#!/usr/bin/env python3
"""Emit the NPU prefill attention kernel (XDNA2 array program + core leaves, Loom low asm) for loom-compile.

One call: 2 mc cores (mc pairs, mc <= 16) each hold 16 query rows of one KV head and stream the same nb K / V records
(the 4 multicast streams: K / V x lower / upper half of a record; each streams the call's payloads, bindings 4 + mc ..,
then the KV head's records, bindings 0 .. 3, which every call of the head shares: HRX patch 0068's constrain.prefix).
A body = one K record (32 keys): QK round S = Q K^T + C (bfp16ebs8 MMAs, f32 chains from the row constants C), the S
blocks parked, softmaxed one body later (P = 2^(U - 128), one accumulator, pushed as bfp16), and one 64-step P.V round
of the P of the 16 bodies before (O += P V). Each core of
a pair receives one half of every K / V record (the lower / upper 64 fragments) and reads the other half from its
partner's tile; a token channel keeps the pair in step. Row sums l = sum P ride as an MMA chain against half-ones
fragments. Records, scales and the unpack: npu_attn_check.py.

The kernel covers K records -1 .. 16 (G - 1) + 14 for nb = 16 (G + 1) (record -1 = the park's start, P = 1 a key; the
GPU takes the keys after them). The cores read nb from the payload (park 1's first word in record 0), so one image serves
every nb up to the one it was built for: patch_nb() rewrites the 4 record lengths of its command words.

The layout is fixed by the Loom planner switches in LOOM_ENV (pinned ring records, bank map npu_attn_banks.txt, the
locked-body planner controls of engine/hrx/patches 0059-0067). Leaves are scheduled by npu_attn_sched.py's model
(issue order = emission order) and emitted as two unrolled body copies.

usage: gen_npu_attn.py <out.loom> <mc> <nb> [out.xdna]
"""
import os
import re
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import hrx_paths  # noqa: E402
import npu_attn_sched as sched  # noqa: E402

EP = "reg<aie2p.ep>"
M1, M2, M4 = "reg<aie2p.mbms>", "reg<aie2p.mbms x2>", "reg<aie2p.mbms x4>"
X, Y = "reg<aie2p.vec256 x2>", "reg<aie2p.vec256 x4>"
LF_, RF2_ = "reg<aie2p.eldfiforeg>", "reg<aie2p.erf2>"
ER, EPS = "reg<aie2p.er>", "reg<aie2p.eps>"
R27 = "reg<aie2p.mr27_select>"
POOLS = {"acc": 20, "x": 24}                             # register pool budgets of the scheduling model

# core exp2: p = E + (x E) v, v = bf16(b1 + a2 x) ~ 2^(x + 1/2) / a0 on x in [-.5, .5] (Lawson minimax, a2 on the bf16
# grid; the common 1 / a0 cancels in O / l)
EXP_B1, EXP_A2 = 0.70312322192927, 0.23828125

FR = 72                                                  # a bfp16 fragment: 8 rows x (exponent + 8 mantissas)
RG = 64 * FR                                             # a 64-fragment record: Q, a K / V slot, a P region half
RGQ = 64 * FR                                            # Q region / K slot
RGP = 2 * RG                                             # a P region: the pushes of 16 bodies (128 fragments, one 64-step P.V round)
NSL = 2 * RGP // 576                                     # body push slots in R0 | R1
LSB, LSL = 64, 640                                       # (l) TB + LSB: [lo, hi] half-ones fragments x 4, TB + LSL: the l block
# tile memory: banks 0-1 = R1 | Q | park 0 || R0 | park 1 | C | PDW; in1 = the K / V slots of the two body copies
R1, QO, PARK, R0 = 0, 9216, 13824, 16384
PSTR, BSTR = 25600 - PARK, 320                           # park half / block strides (a block: S + the softmax's b1 cell)
CO, PDW = 28160, 28672                                   # C (the QK chain inits), the partner offset word
IN0, OUT0, IN1, OUT1 = 29696, 10240, 18432, 7104
KO = IN0 + OUT0                                          # body copy u: K slot KO + KVS u, V half KO + KVS u + RG
VO, KVS = KO + RG, 2 * RG
OTB = PDW + 64                                           # the O block table (16 words)
TB = KO + IN1 + 6144                                     # timers (+0 t1, +4 t0), l (+LSL)
KB, VB = RG, RG                                          # record sizes (record 0 = the payload: Q | C, park starts)
NEG, EGW = 34, 256                                       # egress a pair: 34 records of 1 KB (O 0-15 x 2 cores, 2 stats)

LOOM_ENV = dict(
    LOOM_EXP_LOCKED_PACK="1", LOOM_EXP_LOCKED_HOIST="8", LOOM_EXP_LOCKED_HOIST_ACC="1", LOOM_EXP_LOCKED_HOIST_ST="1",
    LOOM_EXP_LOCKED_VIEWS="1", LOOM_EXP_NOALIAS_ACC="2", LOOM_EXP_LOCKED_FLOAT="all",
    LOOM_EXP_LATE_READ="vmov.vector512.to.accumulator512", LOOM_EXP_LRU="aie2p.mexa", LOOM_EXP_NOWAR_ACC="1",
    LOOM_EXP_LS_STAGE_DEPTH="6", LOOM_EXP_LOCK_ACQREL="6", LOOM_EXP_DELAYFILL="2", LOOM_EXP_DELAYFILL_EFFD="1",
    LOOM_EXP_DELAYFILL_NOEFF="1", LOOM_EXP_POPDODGE="1", LOOM_EXP_BANKDRIFT="1",
    LOOM_EXP_BANKMAP=os.path.join(HERE, "npu_attn_banks.txt"),
    LOOM_EXP_RING_AT=f"{KB}@{KO + KVS}:{KO},{VB}@{KO + KB}:{KO + KVS + KB},1024@{IN0}:{IN0 + 1024},64@65472")
_SCHED = {}


def schedule(ops):
    """npu_attn_sched.schedule of an op list (the body depends only on the core's half: leaves share the result)"""
    key = "\n".join(f"{o[0]}|{o[1]}|{o[2]}|{o[3]}" for o in ops)
    if key not in _SCHED:
        _SCHED[key] = sched.schedule(list(ops), POOLS)
    return _SCHED[key]


def cat4(q):
    return f"concat({', '.join(q)}) : ({', '.join([M1] * 4)}) -> {M4}"


def softmax(ops, n, U, H, fs, gate, flush, tag, sfx):
    """one S block (U, f32 [8 rows, 8 keys], in one accumulator) -> P = 2^(U - 128) pushed as bfp16 through the store
    FIFO fs: m = U clamped to [128, 254] on the bf16 grid, y = U - m (exact, in [-.5, .5]), x = bf16(y),
    P = E + (x E) v with E = 2^(m - 128) and v = bf16(b1 + a2 x) (fitted, EXP_B1 / EXP_A2). 64-lane MACs on y operands
    (mma.bf16bf16.m8n8k1 / mms.bf16.y-y under the elementwise control word); the e16 exponents spill to the block's slot."""
    a = lambda line, tag=None, after=(), prio=1: ops.append(("  " + line, tag, list(after) + list(gate), prio))
    pair = lambda v, w, t: f"concat({v}, {w}) : ({t}, {t}) -> {M2 if t == M1 else M4 if t == M2 else Y}"
    for nm in ("klo", "khi", "ke", "ks"):
        a(f"{n}{nm} = vbcst.16 %s{nm}{sfx}")
    for nm in ("one", "ka3"):
        for j in range(2):
            a(f"{n}{nm}{j} = vbcst.16 %s{nm}{sfx}")
    for h in range(2):
        m = f"{n}h{h}"
        a(f"{m}tb = {H[h]}", tag=f"{n}tb{h}")
        a(f"{m}tl, {m}tlm = vmax_lt.s16x32 {m}tb, {n}klo")
        a(f"{m}tc, {m}tcm = vmin_ge.s16x32 {m}tl, {n}khi")
        a(f"{m}e16 = vadd.16 {m}tc, {n}ke")
        a(f"{m}esp = mov.scalar-to-address %esa{sfx}")
        a(f"vst.512.bf16x32 {m}e16, {m}esp, {64 * h}", tag=f"{m}est")
    a(f"{n}ym = {pair(f'{n}h0tc', f'{n}h1tc', X)}")
    a(f"{n}yo = {pair(f'{n}one0', f'{n}one1', X)}")
    a(f"{n}Y = mms.bf16.y-y {U}, {n}ym, {n}yo, %cf")                 # y = U - m in [-.5, .5], exact
    for q in range(4):
        a(f"{n}yq{q} = slice {n}Y[{q}] : {M4} -> {M1}")
    for h in range(2):
        m = f"{n}h{h}"
        a(f"{m}y = {pair(f'{n}yq{2 * h}', f'{n}yq{2 * h + 1}', M1)}")
        a(f"{m}x = vconv.bf16.fp32 {m}y", tag=f"{m}xc")
    lvl = [f"{n}h0xc", f"{n}h1xc"]
    for h in range(2):
        m = f"{n}h{h}"
        a(f"{m}ea = vups.2x.x-to-c.unsigned {m}tc, %sh7", after=lvl)
        a(f"{m}e7 = vsrs.2x.c-to-x.unsigned {m}ea, %sh0", tag=f"{m}ebc")
        a(f"{m}xe = vadd.16 {m}x, {m}e7")
    a(f"{n}yx = {pair(f'{n}h0x', f'{n}h1x', X)}")
    lvl, prev = [f"{n}h0ebc", f"{n}h1ebc"], f"{n}yk"
    a(f"{n}yk = {pair(f'{n}ka30', f'{n}ka31', X)}")
    nm = "v"                                         # v = bf16(b1 + x a2): b1 from the block's b1 cell (4 loads, none CSE'd)
    a(f"{n}{nm}kp = mov.scalar-to-address %esb{sfx}", after=lvl)
    ks = []
    for q in range(4):
        a(f"{n}{nm}k{q} = vlda.acc {n}{nm}kp, 192", after=lvl)
        ks.append(f"{n}{nm}k{q}")
    a(f"{n}{nm}k = {cat4(ks)}")
    a(f"{n}{nm} = mma.bf16bf16.m8n8k1 {n}{nm}k, {n}yx, {prev}, %cf")
    for q in range(4):
        a(f"{n}{nm}q{q} = slice {n}{nm}[{q}] : {M4} -> {M1}")
    for h in range(2):
        a(f"{n}h{h}{nm}l = {pair(f'{n}{nm}q{2 * h}', f'{n}{nm}q{2 * h + 1}', M1)}")
        a(f"{n}h{h}{nm}b = vconv.bf16.fp32 {n}h{h}{nm}l", tag=f"{n}h{h}{nm}c")
    a(f"{n}y{nm} = {pair(f'{n}h0{nm}b', f'{n}h1{nm}b', X)}")
    lvl, prev = [f"{n}h0{nm}c", f"{n}h1{nm}c"], f"{n}y{nm}"
    for h in range(2):                                               # p = E + (x E) v
        a(f"{n}h{h}elp = mov.scalar-to-address %esb{sfx}", after=lvl)
        a(f"{n}h{h}pa = vlda.ups.2x.x-to-c.unsigned %sh23, {n}h{h}elp, {64 * h - 64}", after=lvl + [f"{n}h{h}est"])
    a(f"{n}E = {pair(f'{n}h0pa', f'{n}h1pa', M2)}")
    a(f"{n}yxe = {pair(f'{n}h0xe', f'{n}h1xe', X)}")
    a(f"{n}P = mms.bf16.y-y {n}E, {n}yxe, {prev}, %cf")
    a(f"{n}f1, {n}op1, {n}q1 = vst.push.bfp16ebs8.from.fp32 {fs[0]}, {n}P, {fs[1]}, {fs[2]}", tag=None if flush else tag)
    if not flush:
        return (f"{n}f1", f"{n}op1", f"{n}q1")
    a(f"{n}ff, {n}fp, {n}fq = vst.flush.512 {n}f1, {n}op1, {n}q1", tag=tag)
    return (f"{n}ff", f"{n}fp", f"{n}fq")


def leaf(mc, pair, half):
    """core leaf 2 pair + half (half 1 parks q0 into blocks 4-7 and starts its P.V lanes at the P region middle)"""
    mcf = mc + (1 - mc % 2)                              # prefix records (+ 1 pad if mc is even: body 0 lands in copy 0)
    pdl = -0x10000 if half == 0 else -0x30000            # the partner tile, as an offset from this one
    L = [f"low.func.def schedule(locked) target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @leaf{2 * pair + half}() asm {{"]
    e = lambda s: L.append("  " + s)
    e(f"%tok = resource<native_pointer> {{index = 4, source_type = buffer}} : {EP}")
    e("%am1 = mova.i32 -1")
    for p_ in ((0, 1, 2) if half == 0 else (0, 2)):
        e(f"acq %am1, {p_}")
    e("set.rounding 12")
    e("set.saturation 0")
    uqn = [0]
    CLUK = [False]
    def uk(nm, v):                                       # a prologue constant no body constant can CSE onto
        uqn[0] += 1; q_ = 7 * (901 + 2 * uqn[0])
        e(f"%{nm}_u0 = mov.i32 {v + q_}"); e(f"%{nm}_u1 = mov.i32 {-q_}"); e(f"%{nm} = add.rr %{nm}_u0, %{nm}_u1")
        return f"%{nm}"
    e("%ins0 = mov.address-to-scalar %tok")
    e(f"%ins = add.rr %ins0, {uk('insk', 0x70000 - (0x4FFC0 if pdl == -0x30000 else 0x7FFC0))}")   # (the tile base, from
    e(f"%outs = add.rr %ins, {uk('outk', IN0)}")     #  the token port's pointer)
    def cloop(nm, n, carried, body):                 # a counted loop: carried = [(name, init, type)]; body(e) -> next values
        if CLUK[0]:                                  # (epilogue loops: no constant CSE'd onto a prologue one,
            uk(f"{nm}i0", 0); uk(f"{nm}n", n); uk(f"{nm}s", 1)   #  which would live across the main loop)
        else:
            e(f"%{nm}i0 = mov.short 0"); uk(f"{nm}n", n); e(f"%{nm}s = mov.short 1")
        L.append(f"  low.br ^{nm}h(%{nm}i0: {ER}" + "".join(f", {i_}: {ty_}" for v_, i_, ty_ in carried) + ")")
        L.append(f"^{nm}h(%{nm}i: {ER}" + "".join(f", %{v_}: {ty_}" for v_, i_, ty_ in carried) + "):")
        e(f"%{nm}t = lt %{nm}i, %{nm}n"); e(f"low.cond_br %{nm}t, ^{nm}b, ^{nm}x : {ER}")
        L.append(f"^{nm}b:")
        nx = body()
        e(f"%{nm}i1 = add.rr %{nm}i, %{nm}s")
        L.append(f"  low.br ^{nm}h(%{nm}i1: {ER}" + "".join(f", {x_}: {ty_}" for x_, (v_, i_, ty_) in zip(nx, carried)) + ")")
        L.append(f"^{nm}x:")
    e("%sz0 = acc.clear.f32x64"); e(f"%szq = slice %sz0[0] : {M4} -> {M1}")
    e("%szp = mov.scalar-to-address %ins")
    cloop("sz", IN0 // 64, [("szc", "%szp", EP)], lambda: (e("%szc1 = vst.acc.post %szq, %szc, 64"), ["%szc1"])[1])
    def seg(nm, src, dst, n):
        e(f"%{nm}sa = add.rr %ins, {uk(nm + 'sk', src)}"); e(f"%{nm}sp = mov.scalar-to-address %{nm}sa")
        e(f"%{nm}da = add.rr %ins, {uk(nm + 'dk', dst)}"); e(f"%{nm}dp = mov.scalar-to-address %{nm}da")
        def body():
            e(f"%{nm}v, %{nm}c1 = vlda.acc.post %{nm}c, 64")
            e(f"%{nm}e1 = vst.acc.post %{nm}v, %{nm}e, 64")
            return [f"%{nm}c1", f"%{nm}e1"]
        cloop(nm, n, [(f"{nm}c", f"%{nm}sp", EP), (f"{nm}e", f"%{nm}dp", EP)], body)
    PL = KO + KVS                                    # the payload: Q | C | park 0 starts (8 x 256) | park 1 starts
    PLV = KO + KB                                    # (its V half: C | park starts; nb in park 1's first word)
    assert RGQ == KB
    if pair % 2:
        PL, PLV = KO, KO + KVS + KB
    for r_ in range(pair):
        e("rel %pone0, 0") if r_ else (e("%pone0 = mova.i32 1"), e("%pm10 = mova.i32 -1"), e("rel %pone0, 0"))
        e("rel %pone0, 2"); e("acq %pm10, 0"); e("acq %pm10, 2")
    seg("cq", PL, QO, RGQ // 64)
    seg("cc", PLV, CO, 8)
    car_ = []                                        # park starts: one loop over the 8 blocks, both halves
    for h_ in range(2):
        e(f"%pk{h_}sa = add.rr %ins, {uk(f'pk{h_}sk', PLV + 512 + 2048 * h_)}"); e(f"%pk{h_}sp = mov.scalar-to-address %pk{h_}sa")
        e(f"%pk{h_}da = add.rr %ins, {uk(f'pk{h_}dk', PARK + PSTR * h_)}"); e(f"%pk{h_}dp = mov.scalar-to-address %pk{h_}da")
        car_ += [(f"pk{h_}s", f"%pk{h_}sp", EP), (f"pk{h_}d", f"%pk{h_}dp", EP)]
    def pbody():
        nx = []
        for h_ in range(2):
            s_, d_ = f"%pk{h_}s", f"%pk{h_}d"
            for q_ in range(4):              # (the last store steps BSTR - 192: the next block's start)
                e(f"%pk{h_}v{q_}, {s_}{q_ + 1} = vlda.acc.post {s_ if q_ == 0 else s_ + str(q_)}, 64")
                e(f"{d_}{q_ + 1} = vst.acc.post %pk{h_}v{q_}, {d_ if q_ == 0 else d_ + str(q_)}, {64 if q_ < 3 else BSTR - 192}")
            nx += [f"{s_}4", f"{d_}4"]
        return nx
    e(f"%nbs = add.rr %ins, {uk('nbk', PLV + 512 + 2048)}")   #  body 0 overwrites that block before any read)
    e("%nbp = mov.scalar-to-address %nbs")
    e("%end = lda %nbp, 0")
    cloop("pkl", 8, car_, pbody)
    e("%pone = mova.i32 1")
    e("%pm1 = mova.i32 -1")
    for r_ in range(pair + 1, mcf):                   # (MC) the records after this core's
        e("rel %pone, 0"); e("rel %pone, 2"); e("acq %pm1, 0"); e("acq %pm1, 2")
    if pdl == -0x10000:
        e('acq %pm1, 4')                         # (core 0 drains the token channel's start credit)
    e("acq %pm1, 0")   # body 0's K (/ V) landed (copy 0)
    e('rel %pone, 4'); e('acq %pm1, 4')       # both cores: payload copied, body 0 data in both tiles
    e("rel %pone, 0")                                 # the payload slots back to the DMA (body 1's records;
    e("%zo = acc.clear.f32x64")
    e(f"%zq = slice %zo[0] : {M4} -> {M1}")
    e("%pz0 = mov.scalar-to-address %outs")
    cloop("oz", OUT0 // 64, [("ozp", "%pz0", EP)], lambda: (e("%ozp1 = vst.acc.post %zq, %ozp, 64"), ["%ozp1"])[1])
    e(f"%o1k = mov.i32 {KO + IN1}")
    e("%o1s = add.rr %ins, %o1k")       # (O addresses keep the out buffers' provenance: Loom
    e(f"%o1n = mov.i32 {-OUT0}")                   # then knows O stores never alias the in-buffer loads)
    e("%o1b = add.rr %o1s, %o1n")
    e("%py0 = mov.scalar-to-address %o1s")
    cloop("oy", 96, [("oyp", "%py0", EP)], lambda: (e("%oyp1 = vst.acc.post %zq, %oyp, 64"), ["%oyp1"])[1])
    e(f"%lsba = add.rr %ins, {uk('lsbk', TB + LSB)}")
    e("%lz0 = mov.scalar-to-address %lsba")
    CLUK[0] = True                               # (loop constants via uk: the body's 4 / 13 must not CSE onto them)
    cloop("lz", (LSL + 256 - LSB) // 64, [("lzp", "%lz0", EP)], lambda: (e("%lzp1 = vst.acc.post %zq, %lzp, 64"), ["%lzp1"])[1])
    l40, l144, l8, l112 = uk("l40", 0x40404040), uk("l144", 144), uk("l8", 8), uk("l112", 112)   # (uk: no body constant CSEs onto them)
    def lwb():                                   # one [lo, hi] pair: lo words 0-8 (rows 0-3), hi words 27-35
        e(f"%lwx = add.rr %lwc, {l8}"); e("%lwa = mov.scalar-to-address %lwx")   # (st offsets: -32 .. 28)
        e(f"%lwy = add.rr %lwc, {l112}"); e("%lwb = mov.scalar-to-address %lwy")
        for w_ in range(9):
            e(f"st {l40}, %lwa, {4 * w_ - 8}")
            e(f"st {l40}, %lwb, {4 * w_ - 4}")
        e(f"%lwc1 = add.rr %lwc, {l144}")
        return ["%lwc1"]
    cloop("lw", 4, [("lwc", "%lsba", ER)], lwb)
    CLUK[0] = False
    e("%cf = mova.i32 60")
    e("%conf = mova.i32 780")
    e("%sh23 = mov.shift 23")
    e("%sh7 = mov.shift 7")
    e("%sh0 = mov.shift 0")
    f32b = lambda v: int(np.float32(v).view(np.uint32))
    # the softmax's 16-bit constants (made per use, see softmax_next): clamp bounds of m's bf16 bits, the exponent
    # bias, the bf16 ones, a2; b1 (f32) sits in each park block's b1 cell
    SKV = dict(sklo=0x4300, skhi=0x437E, ske=48511, sks=49280, ska3=f32b(EXP_A2) >> 16, sone=0x3F80)
    e("%sa2 = mov.i32 0")
    e(f"%sb1 = mov.i32 {f32b(EXP_B1)}")
    e("%vkv = vbcst.32 %sb1")
    for n in range(8):
        for hh in (0, 1):
            e(f"%vkk{n}_{hh} = mov.i32 {PARK + PSTR * hh + BSTR * n + 256}")
            e(f"%vks{n}_{hh} = add.rr %ins, %vkk{n}_{hh}")
            e(f"%vkp{n}_{hh} = mov.scalar-to-address %vks{n}_{hh}")
            e(f"vst.512.f32x16 %vkv, %vkp{n}_{hh}, 0")
    e("%step = mov.short 1")
    e("%j0 = mov.short 0")
    e('%pu0 = mov.i32 0')
    e('%oo0 = mov.i32 14336')
    e(f"%pk0 = mov.i32 {PSTR}")
    for nm, off in (("push0", R0), ("pr0", PARK)):
        e(f"%k_{nm} = mov.i32 {off}")
        e(f"%{nm} = add.rr %ins, %k_{nm}")
    e("%t0w = mov.cycles")
    e(f"%t0 = slice %t0w[0] : reg<aie2p.er x2> -> {ER}")
    e(f"%kt0 = mov.i32 {TB + 4}")
    e('%at0 = add.rr %ins, %kt0')
    e("%pt0 = mov.scalar-to-address %at0")
    e("st %t0, %pt0, 0")
    car = lambda *v: ", ".join(f"{x}: {ER}" for x in v)
    fty = (EPS, LF_, RF2_)                           # a lane's state over the back edge: pointer, FIFO, position
    fcar = lambda nms: "".join(f", {x}: {ty}" for nm in nms for x, ty in zip(nm, fty))
    for nm, off, kk in (("Rb", KO, "b"), ("Ra", QO, "a")):
        e(f"%{nm}k = mov.i32 {off + 6937}")              # (a value no body constant has: identical
        e(f"%{nm}k2 = mov.i32 {-6937}")                   # mov.i32 are CSE'd into one loop-long register)
        e(f"%{nm}s0 = add.rr %ins, %{nm}k")
        e(f"%{nm}s = add.rr %{nm}s0, %{nm}k2")
        e(f"%{nm}b = mov.scalar-to-address %{nm}s")
        e(f"%{nm}lo = vlda.load-fifo.low512 %{nm}b, 0")
        e(f"%{nm}fz = vlda.load-fifo.high512 %{nm}b, %{nm}lo, 64")
        e(f"%{nm}pz = copy %{nm}b : {EP} -> {EPS}")
        e(f"%{nm}qz = mova.fifo.position 0")
        e(f"%{nm}pi, %{nm}fi, %{nm}qi = vld{kk}.fill.512 %{nm}pz, %{nm}fz, %{nm}qz")
    rin = [[f"%Rb{x}" for x in ("pi", "fi", "qi")], [f"%Ra{x}" for x in ("pi", "fi", "qi")]]
    rh = [[f"%hb{x}" for x in "pfq"], [f"%ha{x}" for x in "pfq"]]
    e(f"low.br ^h({car('%j0', '%pu0', '%oo0', '%pk0', '%push0', '%pr0', *())}{fcar(rin)})")
    L.append(f"^h({car('%j', '%pu', '%oo', '%pk', '%push', '%pr', *())}{fcar(rh)}):")
    e("%t = lt %j, %end")
    e(f"low.cond_br %t, ^b, ^x : {ER}")
    L.append("^b:")
    pass                                             # (the body end releases, then waits for the partner)

    ops = []
    a = lambda line, tag=None, after=(), prio=0: ops.append(("  " + line, tag, list(after), prio))
    kn = [0]
    sgate = []                                           # scalar address ops wait for this (keep them near their use)

    def k(v):
        """a fresh constant op (the second body copy's are made distinct from the first's after the duplication: Loom
        CSEs identical mov.i32 values into one long-lived register)"""
        kn[0] += 1
        a(f"%k{kn[0]} = mov.i32 {v}", after=sgate)
        return f"%k{kn[0]}"

    def addr(base, off):
        kv = k(off)
        kn[0] += 1
        a(f"%a{kn[0]} = add.rr {base}, {kv}", after=sgate)
        return f"%a{kn[0]}"

    def addv(x, y):
        kn[0] += 1
        a(f"%a{kn[0]} = add.rr {x}, {y}", after=sgate)
        return f"%a{kn[0]}"

    lastsel = [None]

    def sel(cond, x, y):                                 # cond != 0 ? x : y (r27 is one register: one select at a time)
        kn[0] += 1
        a(f"%cs{kn[0]} = copy {cond} : {ER} -> {R27}", after=([lastsel[0]] if lastsel[0] else []) + sgate)
        a(f"%sl{kn[0]} = sel.nez {x}, {y}, %cs{kn[0]}", tag=f"sel{kn[0]}")
        lastsel[0] = f"sel{kn[0]}"
        return f"%sl{kn[0]}"

    # per-body addresses (needed from round 0's end on)
    sgate[:] = ["m90"]
    kr = k(RGP)
    a(f"%cR = lt %pu, {kr}")                            # pushes go to R0 this body
    push, pr = "%push", "%pr"
    own = sel("%cR", addr("%ins", R1), addr("%ins", R0))
    kp8 = k(8192)
    a(f"%cP = lt %oo, {kp8}")                          # this body's P.V rounds read the partner's rows
    pvp = own                                        # the P.V rounds' A base (P of the region this body does not push to)
    a(f"%obt = lt %oo, {k(OUT0)}", after=sgate)      # O blocks 10-15 sit 18432 B further (out1): a mask, not a
    a(f"%obm = sub %obt, {k(1)}", after=sgate)       # select (a select takes the one select register and Loom
    a(f"%obd = and %obm, {k(KO + IN1 - IN0 - OUT0)}", after=sgate)   # then remakes the MMA config every MMA)
    ob = addv(addv("%outs", "%oo"), "%obd")
    pw = addv(addr("%ins", PARK), "%pk")
    JOUT = {}                                        # round -> its last pops' modifier registers (A, B)
    qa_ = addr("%ins", QO)                           # constant, A (Q <-> this body's P) from pvp
    kn[0] += 1
    jn = kn[0]
    a(f"%ja{jn} = sub {pvp}, {qa_}", after=sgate)
    a(f"%jq{jn} = sub {qa_}, {pvp}", after=sgate)
    jqp, jpq = addv(f"%ja{jn}", k(-RGQ)), addv(f"%jq{jn}", k(-RG))
    # round end jumps: A (Q -> Q -> P -> Q), B immediates (q0 -> partner K slot, q1 -> own V half, p1 -> the next copy's K)
    jl = (("q0", k(-RGQ)), ("q1", addv(f"%ja{jn}", k(RG - RGQ)) if (half == 1) else jqp), ("p1", jpq if (half == 1) else addv(jpq, k(-RG))))
    for nm_, sa_ in jl:
        a(f"%jA{nm_} = mov.modifier {sa_}", after=sgate)
        v_ = {"q0": -RGQ + pdl, "q1": -pdl + KVS, "p1": -pdl - KVS}[nm_]   # (copy 1's are fixed after the duplication)
        a(f"%jB{nm_} = mov.modifier.immediate {v_}")
        JOUT[nm_] = (f"%jA{nm_}", f"%jB{nm_}")
    a(f"%jBmid = mov.modifier.immediate {pdl - RG}")
    if (half == 1):
        a(f"%jAmid = mov.modifier.immediate {-RGP}")   # (P region end -> its start)
    # the next body's loop-carried values, scheduled into this body's free slots (late: short live ranges)
    sgate[:] = ["m440"]
    a('%j1 = add.rr %j, %step', after=sgate)
    pu1 = addv("%pu", k(576))
    kw = k(2 * RGP)
    a(f"%cw = lt {pu1}, {kw}", after=sgate)
    a(f"%pu1m = sub {pu1}, {kw}", after=sgate)
    pu2 = sel("%cw", pu1, "%pu1m")
    a(f"%oo1 = add.rr %oo, {k(1024)}", after=sgate)   # (PV64: one P.V round a body)
    a(f"%oo2 = and %oo1, {k(16383)}", after=sgate)
    a(f"%pk1 = xor %pk, {k(PSTR)}", after=sgate)
    krn = k(RGP)
    a(f"%cRn = lt {pu2}, {krn}", after=sgate)
    push2 = sel("%cRn", addv(addr("%ins", R0), pu2), addv(addr("%ins", R1 - RGP), pu2))

    lastpop = {}                                         # lane -> its last pop's tag

    def pop(st, kk, name, last, jump=None, extra=()):
        s, cnt = st
        u = kk                                           # a round's first 2 steps of A pops issue on the B unit
        m_ = re.match(r"xa[qp]\d+_(\d+)_\d$", name)       # (one FIFO register class): the A slot is left to the chain inits
        if m_ and int(m_.group(1)) < 2:
            u = "b"
        g = ([f"m{mm[0] - 10}"] if mm[0] >= 10 else []) + list(extra)   # a pop runs G MMAs ahead of its reader
        o = "b" if kk == "a" else "a"
        if lastpop.get(o):                      # ... and after the other lane's last pop (bounded lead)
            g = g + [lastpop[o]]
        if jump:                                         # the group's last pop moves the load pointer by `jump`: the
            a(f"%{name}, %{name}p, %{name}f, %{name}q = vld{u}.pop.bfp16ebs8.post.modifier {s[0]}, {s[1]}, {s[2]}, {jump}",
              tag=f"pop{name}", after=g)                 # fill after it loads the next region
        else:
            a(f"%{name}, %{name}p, %{name}f, %{name}q = vld{u}.pop.bfp16ebs8 {s[0]}, {s[1]}, {s[2]}", tag=f"pop{name}", after=g)
        lastpop[kk] = f"pop{name}"
        s, cnt = [f"%{name}p", f"%{name}f", f"%{name}q"], cnt + 1
        if cnt % 8 == 0 and not last:
            a(f"%{name}rp, %{name}rf, %{name}rq = vld{u}.fill.512 {s[0]}, {s[1]}, {s[2]}")
            s = [f"%{name}rp", f"%{name}rf", f"%{name}rq"]
        return s, cnt
    a(f"%sb = mov.scalar-to-address {push}")
    a("%sl = vlda.store-fifo.low512 %sb, 0")
    a("%sf0 = vlda.store-fifo.high512 %sb, %sl, 64", tag="sethi")
    a(f"%sp0 = copy %sb : {EP} -> reg<aie2p.mpfs>", tag="setsp", after=["sethi"])
    a("%sq0 = mova.fifo.store.position 0")
    fs = ("%sf0", "%sp0", "%sq0")
    a(f"%ul0 = mov.scalar-to-address {pr}")
    smx_q = list(range(8))                              # softmaxes still to place
    nu = [0]
    last_sm = [None]

    def softmax_next(gate):
        nonlocal fs
        if not smx_q:
            return
        n = 8 - len(smx_q)
        smx_q.pop(0)
        g = ([last_sm[0]] if last_sm[0] else []) + list(gate)
        H = []
        for q in range(4):
            a(f"%ux{n}_{q}, %ul{nu[0] + 1} = vlda.acc.post %ul{nu[0]}, {64 if q < 3 else BSTR - 192}", after=g)
            nu[0] += 1
        for h in range(2):
            a(f"%H{n}_{h} = concat(%ux{n}_{2 * h}, %ux{n}_{2 * h + 1}) : ({M1}, {M1}) -> {M2}")
            H.append(f"vconv.bf16.fp32 %H{n}_{h}")
        a(f"%U{n} = {cat4([f'%ux{n}_{q}' for q in range(4)])}")
        i0 = len(ops)
        # its own copies of the 16-bit constants, each right before its broadcast (short scalar lives): vbcst.16 reads
        # the low half, a tag in the high half keeps identical constants apart (else Loom CSEs them into one register
        # and their broadcasts live all body). zp / z, sks and the sa2 / sb1 copies are dead (Loom drops them) but
        # shape the model's issue order: without them leaf 0 needs 24 accumulator registers of 20
        a(f"%zp{n} = mov.address-to-scalar %ul{nu[0]}", after=g)
        a(f"%zk{n} = mov.i32 {63 - n}", after=g)
        a(f"%z{n} = and %zp{n}, %zk{n}")
        zg = {"sklo": None, "skhi": None, "ske": "tb0", "sone": "tb1", "sks": "h0xc", "ska3": "h0xc", "sa2": "h0xc", "sb1": "h0xc"}
        for base, lv in zg.items():
            if base in ("sa2", "sb1"):
                a(f"%{base}_{n} = add.rr %{base}, %z{n}", after=g + ([f"%s{n}{lv}"] if lv else []))
                continue
            kn[0] += 1
            a(f"%{base}_{n} = mov.i32 {SKV[base] | ((kn[0] & 0x7FFF) << 16)}", after=g + ([f"%s{n}{lv}"] if lv else []))
        kn[0] += 1                                       # its e16 spill slot: its own park block (dead after the U loads)
        a(f"%esk{n} = mov.i32 {PARK + BSTR * n}", after=g + [f"%s{n}tb0"])
        a(f"%esa_{n} = add.rr %ins, %esk{n}")
        a(f"%eskb{n} = mov.i32 {PARK + BSTR * n + 64}", after=g + [f"%s{n}h1xc"])
        a(f"%esb_{n} = add.rr %ins, %eskb{n}")      # (b1 = esb + 192: the block's own b1 cell)
        fs = softmax(ops, f"%s{n}", f"%U{n}", H, fs, g, n == 7,
                       f"sm{n}", f"_{n}")
        for i in range(i0, len(ops)):                    # broadcast constants issue near their use (vector pool)
            ln, tg, af, pr = ops[i]
            m = re.match(rf"  %s{n}(\w+?)\d? = vbcst", ln)
            if m:
                for nm, lv in (('ke', 'tb0'), ('one', 'tb1'), ('ks', 'h0xc'), ('ka3', 'h0xc')):
                    if m.group(1) == nm:
                        ops[i] = (ln, tg, af + [f"%s{n}{lv}"], pr)
        for i in range(i0 - 16, len(ops)):   # its ops go before the stream's (short live ranges, short span)
            ops[i] = (ops[i][0], ops[i][1], ops[i][2], 2)
        last_sm[0] = f"sm{n}"

    mm = [0]
    prev_store = {}                                      # chain -> tag of its last store (the accumulator is free after it)

    def mma(acc, x, y, c):
        a(f"%m{mm[0]} = mma.bfp16ebs8.m8n8k8 {acc}, {x}, {y}, %conf", tag=f"m{mm[0]}", prio=1, after=[f"m{mm[0] - 1}"] if mm[0] else [])
        mm[0] += 1
        return f"%m{mm[0] - 1}"

    carry = {}

    def round_(kind, s, lbase, sbase, soft_at, nst=32):
        """one round: 4 chains c (QK: c = 2 kbp + rb from C[rb]; P.V: c = 2 dbp + rbp from O); A pops [step][rb],
        B pops [step][kbp / dbp]; the chains' blocks stored after; softmaxes started at the given steps"""
        nm = f"{kind}{s}"
        if kind == "q" and s == 0:                       # the lanes carry their FIFO state from the previous body
            bst, ast = (rh[0], 0), (rh[1], 0)
        else:
            ast, bst = carry["a"], carry["b"]
        m0 = mm[0]
        early = [f"m{m0 - 6}"] if m0 >= 6 else []
        cur = [None] * 4
        if kind == "q":
            a(f"%cl{nm} = mov.scalar-to-address {lbase}", after=early)
        else:
            a(f"%ol{nm}_0 = mov.scalar-to-address {lbase}", after=early)
        nl = 0
        for st in range(nst):
            if st in soft_at:
                softmax_next([f"m{mm[0] - 1}"] if mm[0] else [])
            xa, xb = f"%xa{nm}_{st}_", f"%xb{nm}_{st}_"
            for c in (0, 1, 2, 3):
                rb, kp = c % 2, c // 2
                if st == 0 and kind == "q" and c >= 2:   # chains (rb, kp 1) start from the same loaded C as (rb, kp 0)
                    cur[c] = f"%c{nm}_{c - 2}"
                elif st == 0:                            # chain start: C (QK) or the O block (P.V)
                    g = [prev_store[c]] if c in prev_store else []
                    if kind == "q" and c + 2 in prev_store:
                        g = g + [prev_store[c + 2]]
                    for q in range(4):
                        if kind == "q":
                            a(f"%l{nm}_{c}_{q} = vlda.acc %cl{nm}, {256 * rb + 64 * q}", after=g)
                        else:
                            a(f"%l{nm}_{c}_{q}, %ol{nm}_{nl + 1} = vlda.acc.post %ol{nm}_{nl}, 64", after=g)
                            nl += 1
                    a(f"%c{nm}_{c} = {cat4([f'%l{nm}_{c}_{q}' for q in range(4)])}")
                    cur[c] = f"%c{nm}_{c}"
                if c == 0:
                    ast = pop(ast, "a", f"{xa}0"[1:], False)
                    bst = pop(bst, "b", f"{xb}0"[1:], False)
                jo = JOUT.get(nm, (None, None)) if st == nst - 1 else (None, None)   # the round's last pops: to the next region
                if (half == 1) and kind == "p" and st == nst // 2 - 1:
                    jo = ("%jAmid", jo[1])
                if c == 1:
                    ast = pop(ast, "a", f"{xa}1"[1:], False, jo[0])
                if kind == "p" and st == nst // 2 - 1:
                    jo = (jo[0], "%jBmid")
                if c == 2:
                    bst = pop(bst, "b", f"{xb}1"[1:], False, jo[1])
                cur[c] = mma(cur[c], f"{xa}{rb}", f"{xb}{kp}", c)
        carry["a"], carry["b"] = ast, bst
        a(f"%sw{nm}_0 = mov.scalar-to-address {sbase}", after=[f"m{m0 + 4 * nst - 8}"])
        n = 0
        for c in range(4):                               # drain the chains fast (their accumulators free the softmax's)
            for q in range(4):
                a(f"%o{nm}_{c}_{q} = slice {cur[c]}[{q}] : {M4} -> {M1}")
                inc = 64 if kind == "p" or q < 3 else BSTR - 192
                a(f"%sw{nm}_{n + 1} = vst.acc.post %o{nm}_{c}_{q}, %sw{nm}_{n}, {inc}", tag=f"st{nm}_{c}" if q == 3 else None, prio=0)
                n += 1
            prev_store[c] = f"st{nm}_{c}"
    sgate[:] = []
    round_("q", 0, addr("%ins", CO), addv(pw, k(4 * BSTR)) if (half == 1) else pw, (6, 22))
    sgate[:] = ["m88"]
    round_("q", 1, addr("%ins", CO), pw if (half == 1) else addv(pw, k(4 * BSTR)), (6, 22))
    sgate[:] = ["m216"]
    round_("p", 1, ob, ob, (6, 22), nst=64)
    while smx_q:
        softmax_next([])
    sgate[:] = [f"m{mm[0] - 24}"]                    # (their addresses near their use: short scalar lives)
    rout = [carry["b"][0], carry["a"][0]]
    # the body is emitted twice (copy 0 then copy 1, names + "_u"): the second reads the first's carried values
    bout = ['%j1', pu2, '%oo2', '%pk1', push2, pw]
    bin_ = ['%j', '%pu', '%oo', '%pk', '%push', '%pr']   # reading the first's carried values
    defs_ = {d for o in ops for d in re.findall(r"%[\w.]+", o[0].partition(" = ")[0]) if " = " in o[0]}
    tags_ = {o[1] for o in ops if o[1]}
    amap = dict(zip(bin_, bout[:6]))
    for hh, oo_ in zip(rh, rout):
        amap.update(zip(hh, oo_))
    rnm = lambda v: v + "_u" if v in defs_ else amap.get(v, v)
    sub_ = lambda s: re.sub(r"%[\w.]+", lambda m: rnm(m.group(0)), s)
    lastm = f"m{mm[0] - 1}"
    pos = {}                                         # MMA where its twin issued in a one-body schedule
    l1, _, at1 = schedule(ops)
    mcyc = sorted(c for ln, c in zip(l1, at1) if "mma.bfp16ebs8" in ln)
    import bisect
    cyc_of = {ln: c for ln, c in zip(l1, at1)}
    for ln, tg, af, pr in ops:
        if not af and ln in cyc_of:
            pos[ln] = bisect.bisect_left(mcyc, cyc_of[ln])
    ops2 = []
    for ln, tg, af, pr in ops:
        af2 = [x + "_u" if x in tags_ else x for x in af]
        if tg == "m0":
            af2.append(lastm)
        if "store-fifo" in ln or "fifo.store.position" in ln:   # one store FIFO: after the first body's flush
            af2.append('sm7')
        if not af2 and ln in pos:            # at its twin's place in the second body
            kq = pos[ln] - 4
            af2 = [f"m{kq}_u" if kq >= 0 else f"m{mm[0] + kq}"]
        elif not af2:                                # (its gates fell off the body's start: the first body's end)
            af2 = [f"m{mm[0] - 1 - 8}"]
        ops2.append((sub_(ln), tg + "_u" if tg else None, af2, pr))
    for i, (ln, tg, af, pr) in enumerate(ops2):
        m_ = re.match(r"(\s*%eskb?\d+_u = mov\.i32 )(-?\d+)\s*$", ln)
        if m_:
            ops2[i] = (f"{m_.group(1)}{int(m_.group(2)) + PSTR}", tg, af, pr)
    nfx = 0
    for i, (ln, tg, af, pr) in enumerate(ops2):
        m_ = re.match(r"(\s*)%jBp1_u = mov\.modifier\.immediate (-?\d+)\s*$", ln)
        if m_:
            ops2[i] = (f"{m_.group(1)}%jBp1_u = mov.modifier.immediate {-KVS - pdl}", tg, af, pr); nfx += 1
        m_ = re.match(r"(\s*)%jBq1_u = mov\.modifier\.immediate (-?\d+)\s*$", ln)
        if m_:      # (copy 1: own V at K + RG - KVS)
            ops2[i] = (f"{m_.group(1)}%jBq1_u = mov.modifier.immediate {-pdl - KVS}", tg, af, pr); nfx += 1
    assert nfx == 2, nfx
    L2 = []
    l2 = lambda ln, tag=None, after=(): L2.append(("  " + ln, tag, list(after), 0))
    pu_ = rnm("%pu")
    ja = next(i for i, o in enumerate(ops2) if re.match(r"\s*%jAp1_u = mov\.modifier ", o[0]))
    J = []                                       # (computed at the P.V round's last pops: short scalar lives)
    gl_ = [f"m{mm[0] - 16}_u"]
    jl_ = lambda ln: J.append(("  " + ln, None, list(gl_), 0))
    jl_(f"%lpk = mov.i32 {RGP}"); jl_(f"%lpc = lt {pu_}, %lpk"); jl_("%lp1 = mov.i32 1")
    jl_("%lpm = sub %lpc, %lp1"); jl_("%lpa = and %lpm, %lpk"); jl_(f"%lpo = sub {pu_}, %lpa")   # pu mod RGP
    jl_(f"%lpjk = mov.i32 {-576 - RGP + (RG if (half == 1) else 0)}"); jl_("%lpj = add.rr %lpo, %lpjk")   # P.V end -> this pair of slots
    jt_ = ops2[ja][1]
    del ops2[ja]
    jp = next(i for i, o in enumerate(ops2) if "%jAp1_u" in o[0])   # the round's last A pop
    ops2[jp:jp] = J + [("  %jAp1_u = mov.modifier %lpj", jt_, list(gl_), 0)]
    jb = next(i for i, o in enumerate(ops2) if re.match(r"\s*%jBp1_u = mov\.modifier\.immediate ", o[0]))
    v1 = -KVS - pdl                    # (V end -> K: v1; V end -> the fragments)
    ops2[jb] = (f"  %jBp1_u = mov.modifier.immediate {TB + LSB - KO + v1}",) + tuple(ops2[jb][1:])
    g0 = [f"m{mm[0] - 24}_u"]                    # (late: short scalar / pointer lives)
    def ljumps():                                # (just before the last pops: few modifier registers, %conf
        gj = ["ml9"]   #  holds one for the whole loop)
        # (from pu alone: no P.V base / slot offset held across the round) = QO - R1 - 576 - pu + (pu >= RGP ? RGP - R0 + R1 : 0)
        l2(f"%lqk = mov.i32 {RGP}", after=gj); l2(f"%lqc = lt {pu_}, %lqk", after=gj); l2("%lq1 = mov.i32 1", after=gj)
        l2("%lqm = sub %lqc, %lq1", after=gj); l2(f"%lqr = mov.i32 {RGP - R0 + R1}", after=gj); l2("%lqa = and %lqm, %lqr", after=gj)
        l2(f"%lqq = mov.i32 {QO - R1 - 576}", after=gj); l2(f"%lqb = sub %lqq, {pu_}", after=gj); l2("%lqf = add.rr %lqb, %lqa", after=gj)
        l2("%jAl1 = mov.modifier %lqf", after=gj)                # l end -> Q (the next body's q0)
        l2(f"%jBl1 = mov.modifier.immediate {KO - (TB + LSB + 576)}", after=gj)   # fragments end -> K copy 0
    l2(f"%lkk = mov.i32 {TB + LSL}", after=g0); l2("%lka = add.rr %ins, %lkk", after=g0)
    l2("%lb0 = mov.scalar-to-address %lka", after=g0)
    for q in range(4):
        l2(f"%lv{q}, %lb{q + 1} = vlda.acc.post %lb{q}, 64", after=g0)
    l2(f"%lc0 = {cat4([f'%lv{q}' for q in range(4)])}")
    sa = [[rnm(x) for x in carry["a"][0]], carry["a"][1]]
    sb = [[rnm(x) for x in carry["b"][0]], carry["b"][1]]
    assert sa[1] % 8 == 0 and sb[1] % 8 == 0
    def lpop(s, kk, nm, jump=None):
        st_, cnt = s
        if jump:
            l2(f"%{nm}, %{nm}p, %{nm}f, %{nm}q = vld{kk}.pop.bfp16ebs8.post.modifier {st_[0]}, {st_[1]}, {st_[2]}, {jump}", tag=f"pop{nm}")
        else:
            l2(f"%{nm}, %{nm}p, %{nm}f, %{nm}q = vld{kk}.pop.bfp16ebs8 {st_[0]}, {st_[1]}, {st_[2]}", tag=f"pop{nm}")
        st_, cnt = [f"%{nm}p", f"%{nm}f", f"%{nm}q"], cnt + 1
        if cnt % 8 == 0:
            l2(f"%{nm}rp, %{nm}rf, %{nm}rq = vld{kk}.fill.512 {st_[0]}, {st_[1]}, {st_[2]}")
            st_ = [f"%{nm}rp", f"%{nm}rf", f"%{nm}rq"]
        return [st_, cnt]
    acc, prv = "%lc0", f"m{mm[0] - 1}_u"         # l: one chain over the 16 slots (2 per kb pair) x [lo, hi]
    for kp in range(4):                          # kb pair: A (kb 2kp: rb 0, 1), B lo, hi, A (kb 2kp + 1: rb 0, 1)
        if kp == 3:
            ljumps()
        sa = lpop(sa, "a", f"la{kp}_0"); sa = lpop(sa, "a", f"la{kp}_1")
        sb = lpop(sb, "b", f"lo{kp}"); sb = lpop(sb, "b", f"lh{kp}", "%jBl1" if kp == 3 else None)
        sa = lpop(sa, "a", f"la{kp}_2"); sa = lpop(sa, "a", f"la{kp}_3", "%jAl1" if kp == 3 else None)
        for i_, y_ in ((0, "lo"), (1, "lh"), (2, "lo"), (3, "lh")):
            t_ = f"ml{4 * kp + i_}"
            l2(f"%{t_} = mma.bfp16ebs8.m8n8k8 {acc}, %la{kp}_{i_}, %{y_}{kp}, %conf", tag=t_, after=[prv])
            acc, prv = f"%{t_}", t_
    l2(f"%lkk2 = mov.i32 {TB + LSL}", after=[prv]); l2("%lka2 = add.rr %ins, %lkk2", after=[prv])
    l2("%lw0 = mov.scalar-to-address %lka2", after=[prv])
    for q in range(4):
        l2(f"%lo{q}x = slice {acc}[{q}] : {M4} -> {M1}")
        l2(f"%lw{q + 1} = vst.acc.post %lo{q}x, %lw{q}, 64", tag="lst" if q == 3 else None)
    ops2 += L2
    rout_l = [sb[0], sa[0]]                          # the back edge's lane state (the l chain's)
    Q = 117440512                                  # (loop-long live ranges): make them distinct values. k + Q with
    rhs_ = lambda s: re.findall(r"%[\w.]+", (s.partition(" = ")[2] if " = " in s else s).split(" : ")[0])
    kd = {}                                      # Q a multiple of 2^16 broadcasts the same 16-bit vector; %ins + k
    for i, (ln, tg, af, pr) in enumerate(ops2):  # = %insQ + (k + Q); other uses get (k + Q) + (-Q)
        m_ = re.match(r"\s*(%[\w.]+) = mov\.i32 (-?\d+)\s*$", ln)
        if m_:
            kd[m_.group(1)] = (i, int(m_.group(2)))
    uses_ = {v: [] for v in kd}
    for i, (ln, tg, af, pr) in enumerate(ops2):
        for v in rhs_(ln):
            if v in uses_:
                uses_[v].append(i)
    uq = [0]
    for v, (i, val) in kd.items():
        us = [ops2[j][0] for j in uses_[v]]
        vb = all(re.match(r"\s*%[\w.]+ = vbcst\.16 ", u) for u in us)
        ia = all(re.match(rf"\s*%[\w.]+ = add\.rr %ins, {re.escape(v)}\s*$", u) for u in us)
        ln, tg, af, pr = ops2[i]
        if vb or ia:
            ops2[i] = (ln.replace(f"mov.i32 {val}", f"mov.i32 {val + Q}"), tg, af, pr)
            if ia:
                for j in uses_[v]:
                    l2, t2, a2, p2 = ops2[j]
                    ops2[j] = (l2.replace("add.rr %ins,", "add.rr %insQ,"), t2, a2, p2)
        else:
            uq[0] += 1
            ops2[i] = (ln.replace(f"{v} = mov.i32 {val}", f"{v}Q = mov.i32 {val + Q}"), tg, af, pr)
            ops2.insert(i + 1, (f"  {v} = add.rr {v}Q, %negQ", None, list(af), pr))
            for w, (i2, val2) in kd.items():  # indices after the insert shift by one
                if i2 > i:
                    kd[w] = (i2 + 1, val2)
            for w in uses_:
                uses_[w] = [j + 1 if j > i else j for j in uses_[w]]
    hi_ = next(n_ for n_, l_ in enumerate(L) if l_.startswith("  low.br ^h("))
    L[hi_:hi_] = [f"  %negQ = mov.i32 {-Q}", "  %insQ = add.rr %ins, %negQ"]
    hi_ += 2
    for n_ in range(hi_ - 1, -1, -1):            # loop seeds: a constant only the entry branch reads merges
        m_ = re.match(r"  (%[\w.]+) = mov\.i32 (-?\d+)$", L[n_])   # with any body constant of its value
        if not m_:                               # and then lives the whole loop: build it un-CSE-able
            continue
        v_, val_ = m_.group(1), int(m_.group(2))
        rd = [l_ for l_ in L if re.search(re.escape(v_) + r"\b", l_) and not l_.startswith(f"  {v_} =")]
        if len(rd) == 1 and rd[0].startswith("  low.br ^h("):
            L[n_:n_ + 1] = [f"  {v_}a = mov.i32 {val_ + 5 * Q}", f"  {v_}b = mov.i32 {-5 * Q}", f"  {v_} = add.rr {v_}a, {v_}b"]
    ops = ops + ops2
    lines, cyc, at0 = schedule(ops)
    fifo = lambda s: ".pop." in s or ".fill." in s
    defs = [re.findall(r"%[\w.]+", s.partition(" = ")[0]) if " = " in s else [] for s in lines]
    uses = [re.findall(r"%[\w.]+", (s.partition(" = ")[2] if " = " in s else s).split(" : ")[0]) for s in lines]
    users = {}
    for i, us in enumerate(uses):
        for u in us:
            users.setdefault(u, []).append(i)
    late = [fifo(s) and not (".fill." in s and re.match(r"\s*%\w+rp,", s)) for s in lines]   # store can't hoist above it)
    for i in reversed(range(len(lines))):
        us = [j for d in defs[i] for j in users.get(d, [])]
        if not late[i] and defs[i] and us and all(late[j] for j in us) and not (".fill." in lines[i] and re.match(r"\s*%\w+rp,", lines[i])) \
                and not re.match(r"\s*%[\w.]+ = (mov\.i32|mov\.short|add\.rr|lt|sel\.nez|and|sub|copy %\w+ : reg<aie2p\.er> -> reg<aie2p\.mr27)\b", lines[i]):
            late[i] = True
    key = [float(at0[i]) + (8 if late[i] else 0) for i in range(len(lines))]
    isview = [bool(re.match(r"\s*%[\w.]+(, %[\w.]+)* = (slice|concat|copy)\b", s)) and "aie2p.mpfs" not in s for s in lines]
    def ru(i, first=False):                              # real readers of op i's results (through views)
        out = []
        for d in defs[i][:1] if first else defs[i]:
            for j in users.get(d, []):
                out += ru(j) if isview[j] else [j]
        return out
    for i in reversed(range(len(lines))):
        if late[i]:
            us = ru(i, first=bool(defs[i]) and (".pop." in lines[i] or "vlda.acc" in lines[i])) or ru(i)
            if us:
                key[i] = min(key[j] for j in us) - 0.01
    prod = {d: i for i in range(len(lines)) for d in defs[i]}
    for i in range(len(lines)):          # every op after its inputs (views right after)
        ks = [key[prod[u]] for u in uses[i] if u in prod]
        if isview[i] and ks:
            key[i] = max(ks) + 0.001
        elif ks:
            key[i] = max(key[i], max(ks) + 0.0001)
    for i in range(len(lines)):                          # the one select register: its copy right before its select
        if "reg<aie2p.mr27_select>" in lines[i] and " = copy " in lines[i]:
            us = users.get(defs[i][0], [])
            if us:
                key[i] = min(key[j] for j in us) - 1e-5
    order = sorted(range(len(lines)), key=lambda i: (key[i], "vst.acc" in lines[i], i))
    lines = [lines[i] for i in order]
    at = [at0[i] for i in order]
    for sfx in ("", "_u"):                           # landed, barrier (both tiles' landed, both done with this
        fi = [(int(m_.group(1)), i) for i, l_ in enumerate(lines)   # copy), this copy's slots back to the DMA
              for m_ in [re.match(rf"\s*%xbp1_(\d+)_1rp{sfx}\b", l_)] if m_ and "vldb.fill" in l_]
        assert fi, sfx
        i_ = max(fi)[1]
        blk = ["  acq %pm1, 0", "  rel %pone, 4", "  acq %pm1, 4"]
        lines[i_:i_] = blk
        at[i_:i_] = [at[i_]] * len(blk)
        fm = [(int(m_.group(1)), i) for i, l_ in enumerate(lines)   # this V landed; barrier; this K back
              for m_ in [re.match(rf"\s*%xbq1_(\d+)_1rp{sfx}\b", l_)] if m_ and "vldb.fill" in l_]
        i_ = max(fm)[1]
        blk = ["  acq %pm1, 2", "  rel %pone, 4", "  acq %pm1, 4", "  rel %pone, 0", "  rel %pone, 2"]
        lines[i_:i_] = blk
        at[i_:i_] = [at[i_]] * len(blk)
    L.extend(lines)
    e(f"low.br ^h({car(*[rnm(v) for v in bout])}{fcar(rout_l)})")
    L.append("^x:")
    e("%t1w = mov.cycles")
    e(f"%t1 = slice %t1w[0] : reg<aie2p.er x2> -> {ER}")
    e("%dt = mov.short 0")
    e(f"%ka = mov.i32 {TB}")
    e('%ta = add.rr %ins, %ka')
    e("%tp = mov.scalar-to-address %ta")
    e("st %t1, %tp, 0")
    e(f"%kpl = mov.i32 {TB + 200}")
    e("%apl = add.rr %ins, %kpl")
    e("%ppl = mov.scalar-to-address %apl")
    e("%pl0 = mov.address-to-scalar %tok")
    e("st %pl0, %ppl, 0")
    CLUK[0] = True
    e("%one1b = mova.i32 1"); e("%am1b = mova.i32 -1")   # 34 1 KB records: O 0-15, the partner's O 0-15, the stats
    e('rel %one1b, 4'); e('acq %am1b, 4')    # (own, partner) through 2 slots on O blocks 0 / 1
    if half == 0:
        e("rel %one1b, 1"); e("acq %am1b, 1"); e("rel %one1b, 1")   # records 0, 1: O blocks 0, 1 in place
        assert TB == IN0 + 16384 + (KO + IN1 - IN0 - OUT0)   # (stats = "block 16" of the O formula)
        e(f"%egbase = add.rr %ins, {uk('egbk_', IN0)}")
        for nm_, v_ in (("eg1", 1), ("eg2", 2), ("eg10", 10), ("eg17", 17), ("egdel", KO + IN1 - IN0 - OUT0),
                        ("egpd", pdl)):
            uk(nm_, v_)
        def rbody():                                 # record r = i + 2: (own | partner) block b, slot r % 2
            e("%egr = add.rr %egri, %eg2")
            e("%egc = lt %egr, %eg17"); e("%egm = sub %egc, %eg1")        # 0 own, -1 partner
            e("%egb17 = and %egm, %eg17"); e("%egb = sub %egr, %egb17")
            e("%egt = lt %egb, %eg10"); e("%egtm = sub %egt, %eg1")       # 0 below block 10, -1 from it
            e("%egdl = and %egtm, %egdel"); e("%egpo = and %egm, %egpd")
            e("%egbk = lshl %egb, %eg10")
            e("%egs0 = add.rr %egbase, %egbk"); e("%egs1 = add.rr %egs0, %egdl"); e("%egs2 = add.rr %egs1, %egpo")
            e('%egsp = mov.scalar-to-address %egs2')
            e("%egpar = and %egr, %eg1"); e("%egpk = lshl %egpar, %eg10"); e("%egda = add.rr %egbase, %egpk")
            e("%egdp = mov.scalar-to-address %egda")
            e("acq %am1b, 1")
            def b_():
                e("%egv, %egcp1 = vlda.acc.post %egcp, 64"); e("%egcq1 = vst.acc.post %egv, %egcq, 64")
                return ["%egcp1", "%egcq1"]
            cloop("egc", 16, [("egcp", "%egsp", EP), ("egcq", "%egdp", EP)], b_)
            e("rel %one1b, 1")
            return []
        cloop("egr", 32, [], rbody)
    e("%one1 = mova.i32 1")
    e("%am1v = mova.i32 -1")                         # the stream's last V record (a dummy): every ring then ends a
    e("acq %am1v, 2"); e("rel %one1, 2")             # call on copy 0, where the next call (a continuation) starts
    if half == 1:                                    # the token credit the lower core's first acquire took (setup
        e("rel %one1, 4")                            # holds one): back for the next call
    for p_ in (0, 2):
        e(f"rel %one1, {p_}")
    e("return")
    L.append("}")
    hb = next(n_ for n_, l_ in enumerate(L) if l_.startswith("^b:"))   # before the exit test: the JZ's delay
    cb = next(n_ for n_ in range(hb, 0, -1) if L[n_].startswith("  low.cond_br "))   # slots can hold them
    ok_ = ("mov.i32", "mov.scalar-to-address", "mova.fifo.store.position", "add.rr", "vlda.store-fifo.low512",
           "vlda.acc", "concat(", "mova.i32")
    n_ = hb + 1
    while n_ < len(L) and re.match(r"\s*%[\w.]+(?:\s*,\s*%[\w.]+)*\s*=\s*(\S+)", L[n_]) and \
                any(re.match(r"\s*%[\w.]+(?:\s*,\s*%[\w.]+)*\s*=\s*" + re.escape(k), L[n_]) for k in ok_):
        n_ += 1
    mv = L[hb + 1:n_]
    del L[hb + 1:n_]
    tl = next((i_ for i_ in range(cb - 1, 0, -1) if L[i_].startswith("  %t = lt ")), cb)
    L[tl:tl] = mv
    return L


def array_program(mc, nb):
    mcf = mc + (1 - mc % 2)
    nk, nv = mcf + nb + 1, mcf + nb + 1                  # a channel's records: the call's payloads, then the KV head's
    S_ = "reg<aie2p.array.scalar : index>"
    ty_ = lambda kind, n_: f"reg<aie2p.array.{kind} : tile<{n_}xi32>>"
    H = ["aie2p.target<array> @array_target", "aie2p.target<core> @core_target", "",
         "low.func.def public retain target<amd.xdna.aie2p.array>(@array_target) abi(array_program) @probe() asm {"]
    he = lambda s: H.append("  " + s)
    for k_, v_ in [("zero", 0), ("one", 1), ("two", 2), ("nk", nk), ("nv", nv), ("neg", NEG)] + \
                [(f"r{r}", r) for r in range(2, 6)] + [(f"c{c}", c) for c in range(8)]:
        he(f"%{k_} = constant.u32 {v_} : {S_}")
    he("%origin = constant.u64 0 : reg<aie2p.array.offset : offset>")
    for w_ in range(2 * mc):
        p_, h_ = divmod(w_, 2)
        he(f"%g{w_} = group %one"); he(f"%k{w_} = worker %g{w_}, %zero, @leaf{w_}")
        he(f"constrain.location %k{w_}, %c{p_ % 8}, %r{(2 if p_ < 8 else 4) + h_}")
    for s, nm in enumerate(("klo", "khi", "vlo", "vhi")):
        wd, nr, port = (KB // 4, "nk", 0) if s < 2 else (VB // 4, "nv", 2)
        he(f'%b{nm} = binding {s}, "read"')                # the records (nb + 1)
        he(f'%p{nm} = binding {4 + mc + s}, "read"')       # the payload prefix (mcf), HRX patch 0068
        he(f"%pb{nm} = constant.u32 {mcf * (KB if s < 2 else VB)} : {S_}")
        he(f"%a{nm} = sender %b{nm}, 0 : {ty_('sender', f'1x{nk if s < 2 else nv}x{wd}')}")   # the whole stream
        he(f"%x{nm} = partition.sender %a{nm}, %origin, %zero, %one : {ty_('sender', wd)}")
        for p_ in range(mc):
            w_ = 2 * p_ + s % 2
            he(f"%y{nm}{w_} = receiver %k{w_}, {port} : {ty_('receiver', wd)}")
            he(f"%c{nm}{w_} = channel %x{nm}, %y{nm}{w_}, %two, %{nr} : {ty_('channel', wd)}")
            he(f"constrain.leaf_sync %c{nm}{w_}")
            if p_ == 0:
                he(f"constrain.stage %c{nm}{w_}, %c{s % 2}")
                he(f"constrain.prefix %c{nm}{w_}, %p{nm}, %pb{nm}")
    for p_ in range(mc):
        lo, hi = 2 * p_, 2 * p_ + 1
        he(f'%bo{p_} = binding {4 + p_}, "write"')
        he(f"%ao{p_} = receiver %bo{p_}, 0 : {ty_('receiver', f'1x{NEG}x{EGW}')}")
        he(f"%xo{p_} = partition.receiver %ao{p_}, %origin, %zero, %one : {ty_('receiver', EGW)}")
        he(f"%yo{p_} = sender %k{lo}, 1 : {ty_('sender', EGW)}")
        he(f"%co{p_} = channel %yo{p_}, %xo{p_}, %two, %neg : {ty_('channel', EGW)}")
        he(f"constrain.leaf_sync %co{p_}")
        he(f"%ts{p_} = sender %k{lo}, 4 : {ty_('sender', 16)}")
        he(f"%tr{p_} = receiver %k{hi}, 4 : {ty_('receiver', 16)}")
        he(f"%tc{p_} = channel %ts{p_}, %tr{p_}, %one, %one : {ty_('channel', 16)}")
        he(f"constrain.leaf_sync %tc{p_}")
    he("return"); H.append("}")
    return H


def emit(mc, nb):
    """the whole program text: the array program + 2 mc leaves"""
    assert 1 <= mc <= 16 and nb % 2 == 0
    H = array_program(mc, nb)
    for w in range(2 * mc):
        H += [""] + leaf(mc, *divmod(w, 2))
    return "\n".join(H) + "\n"


def compile_xdna(loom, xdna):
    r = subprocess.run([hrx_paths.LOOM_COMPILE, loom, "--root=@probe", "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11",
                        f"--output={xdna}"], capture_output=True, text=True, env=dict(hrx_paths.env(), **LOOM_ENV))
    if r.returncode:
        sys.exit("loom-compile failed\n" + r.stderr[-3500:])


def stream_words(mc, nb):
    """the record descriptors' lengths in i32 words: K, V (each stream's after its payload prefix)"""
    return (nb + 1) * KB // 4, (nb + 1) * VB // 4


def patch_nb(image, mc, nb_built, nb):
    """an image built for nb_built bodies, run for nb <= nb_built: rewrites the 4 record lengths in .xdna.command (the
    host does the same on each call's invocation bytes; record bindings stay sized for nb_built)"""
    b = bytearray(image)
    shoff, = struct.unpack_from("<I", b, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x2E)
    hs = [struct.unpack_from("<IIIIIIIIII", b, shoff + i * shentsize) for i in range(shnum)]
    so = hs[shstrndx][4]
    name = lambda h: b[so + h[0]:b.index(b"\0", so + h[0])].decode()
    rules = dict(zip(stream_words(mc, nb_built), stream_words(mc, nb)))
    n = 0
    for h in hs:
        if name(h) == ".xdna.command":
            for j in range(h[4], h[4] + h[5] - 3, 4):
                w, = struct.unpack_from("<I", b, j)
                if w in rules:
                    struct.pack_into("<I", b, j, rules[w])
                    n += 1
    assert n == 4, n
    return bytes(b)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    out, mc, nb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    with open(out, "w") as f:
        f.write(emit(mc, nb))
    if len(sys.argv) > 4:
        compile_xdna(out, sys.argv[4])


if __name__ == "__main__":
    main()
