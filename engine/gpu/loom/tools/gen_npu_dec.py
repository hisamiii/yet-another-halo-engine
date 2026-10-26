#!/usr/bin/env python3
"""NPU weight decoders for the fill column of gen_npu_gemm (dcol = 2), one per row format (DQ_FMT: IQ4_XS, the down
pack's, or MQ4, the GPU weights' gate and up rows; the module's constants follow DQ_FMT at import, gen_npu_gemm.decoder
reloads it per format).
Standalone check: DQ_FMT=<fmt> gen_npu_dec.py <work> <model> [ports] (decodes 80 rows per port of the first matching
K = 5120 tensor of the GGUF, or the HFQ file for MQ4, and compares every fragment with the oracle, bit for bit).
The IQ4_XS decoder (16-row panel units):
Input record (one panel unit): [16 rows][4 super-blocks x 136 B] (the (pass, slab) unit read from the row-major GGUF
tensor by an iterated shim view). Output: 4 leaf-synchronized records of 64 bfp16 fragments (super-block j of the
unit), in panel order [kb 32][h 2] (h = rows 0-7 / 8-15).
Per super-block: both halves are prepped (header words; qs rows realigned by vshift and transposed to [chunk][row][8 B];
the IQ4_XS scale path into the half's TB), then 8 sub-block iterations of 4 chunk-halves (chunks 2 s, 2 s + 1 x h):
the low-nibble fragments are pushed in order, the high-nibble accumulators parked and pushed after them.
"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import hrx_paths
import npu_dec_emit as Q
I = Q
from npu_dec_emit import Emit, V1, V2, V4, M1, M2, M4
FMT = os.environ.get("DQ_FMT", "IQ4_XS")
assert FMT in ("IQ4_XS", "MQ4"), FMT
BLK = 136                       # bytes per super-block
HBY = 8                         # header bytes before qs
RECROW = 576                    # record row bytes (4 x 144, room for wider formats' 4 super-blocks): image switches
                                # between formats re-program only the decoder tiles (the staging buffers stay identical)
ROWB = RECROW                   # a row's 4 super-blocks (+ ignored tail)
IN_B, NF = 16 * ROWB, 64
OUT_B = 32 * 72                 # one output record (half a super-block: 4 sub-blocks)
GT, HR = 0, 2048                # table; half regions at HR + h HS
HDR, SCM, HB, TAB, PARTS, TB, QT = 0, 64, 192, 256, 384, 896, 2944
HS = 512                       # half regions now hold only the scale-path temporaries (HDR .. PARTS)
STATE, KVS = HR + 2 * HS, HR + 2 * HS + 64
SCR_B = KVS + 64                # storage A (2048-aligned: the table)
GAS, PARTSS, GT2 = 0, 4096, 5120
SCR2_B = GT2 + 2048 + 64        # storage B (gather addresses, T parts, the table's second copy; +64: gathers read a + 32)
TABV, BASEV = [TAB], ["%sp"]
# MQ4 (MQ4G256 as the HFQ loader leaves it): the IQ4_XS decoder with the nibbles' signed codes for kvalues, ls = 33 and
# d [row][f] = the scale of K half f (T = d (ls - 32) = the scale); its positions are IQ4_XS's (sub-block s: low nibbles of
# bytes 16 s .. 16 s + 15, then the high), so position p of sub-block s holds element 32 s + MQ4_ELEM[p]
MQ4 = FMT == "MQ4"
MQ4_ELEM = [2 * p if p < 16 else 2 * (p - 16) + 1 for p in range(32)]
KV = [float(v - 16 * (v >= 8)) for v in range(16)] if MQ4 else I.KV
INLINE = [0]                    # leaf_inline: super-blocks per input record (0: a standalone leaf)
VROLL = [False]                 # leaf_inline: the scale halves rolled whatever the format (program memory)
TONCE = [True]                  # leaf_inline: the table built once per fill (flag at TFLAG), else every invocation
TFLAG = 32                      # leaf_inline: the table-built flag's byte offset in the caller's private storage %fzs
                                # (past scalar lda / st immediates: addressed through a pointer at it)
RREC = [None]                   # leaf_inline: (first, end) output record decoded (records 2 j + r), None: all
TC_IQ4 = (("su", 0x380), ("sv", 0), ("u", 0xC6A0), ("v", 0xC320), ("s", 0xC681), ("1", 0x4A01), ("1", 0x4901))
NPARTS = 2                      # bf16 parts per scale line (T hi / lo)


def leaf(name="dec", nports=1):
    e = Emit()
    e.L.append(f"low.func.def schedule(locked) target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @{name}() asm {{")

    for pp_ in range(nports):
        sfx = "" if pp_ == 0 else str(pp_)
        e(f"%in{sfx} = resource<native_pointer> {{index = {pp_}, source_type = buffer}} : reg<aie2p.ep>")
    for pp_ in range(nports):
        sfx = "" if pp_ == 0 else str(pp_)
        e(f"%out{sfx} = resource<native_pointer> {{index = {nports + pp_}, source_type = buffer}} : reg<aie2p.ep>")
    e(f"%scr = storage {{byte_alignment = 64, byte_length = {SCR_B}}} : low.storage<private>")
    # storage B in its own space: one region per space, so the gathers (table in A) and the address / T-part loads (B)
    # can sit in different banks (both in one bank: a conflict in nearly every hot-loop bundle)
    sp2 = "scratch"
    e(f"%scr2 = storage {{byte_alignment = 64, byte_length = {SCR2_B}}} : low.storage<{sp2}>")
    e(f"%sp2 = storage_address %scr2 : low.storage<{sp2}> -> reg<aie2p.ep>")
    e("%sp = storage_address %scr : low.storage<private> -> reg<aie2p.ep>")
    e("set.unpack-size 0")
    e("set.rounding 12")
    hdr_end = len(e.L)
    k = e.const
    conf = k(60)
    m0c, m1c, m20, m21, m52, m53 = k(0), k(1), k(20), k(21), k(52), k(53)
    m2c, m3c, m4c, m18, m19 = k(2), k(3), k(4), k(18), k(19)
    # ---- helpers (x: the stream to emit into)
    CTV = [0x380, 0, 0xC6A0, 0xC320, 0xC681, 0x4A01, 0x4901, 0x3F80, 0xBF80, 0x4300, 128, 3200, 1024, 0x7C, 0x8000]
    CTP = {}                                            # stream id -> pointer at CT + 512

    def splat16(x, v):
        r_ = x.t("sp")
        if CTP.get("on") and v in CTV:
            x(f"{r_} = vlda.512.i8x64 %ctp, {64 * CTV.index(v) - 512}")
            return r_
        x(f"{r_} = vbcst.16 {k(v)}")
        return r_


    def add16(x, a_, b_):
        r_ = x.t("ad")
        x(f"{r_} = vadd.16 {a_}, {b_}")
        return r_

    def sub16(x, a_, b_):
        r_ = x.t("sb")
        x(f"{r_} = vsub.16 {a_}, {b_}")
        return r_

    def band(x, a_, v):
        r_ = x.t("an")
        x(f"{r_} = vband {a_}, {splat16(x, v)}")
        return r_

    def mac(x, acc, s1, s2):
        r_ = x.t("ac")
        if acc is None:
            x(f"{r_} = mmul.bf16bf16.m8n8k1 {s1}, {s2}, {conf}")
        else:
            x(f"{r_} = mma.bf16bf16.m8n8k1 {acc}, {s1}, {s2}, {conf}")
        return r_


    def shr8(x, a_):
        o, r_ = x.t("o"), x.t("r8")
        x(f"{o} = vshuffle {a_}, {a_}, {m1c}")
        x(f"{r_} = vshuffle {o}, %z8, {m20}")
        return r_


    import npu_lsched as lsched
    WIN = 4

    def run(streams, window_=None, tails=None, ext=(0, 0, 0), hv=None, full=False):   # full: lsched VEC_FULL / ACC_FULL
        e.L += lsched.schedule([x.L for x in streams], window_ or WIN, tails, ext=ext, full=full,
                               hard=(hv or 24, 20, 3))

    def rel(fp, off, hi=448, step=64):
        o = off - fp[1]
        assert -(hi + step) <= o <= hi and o % step == 0, (off, fp)
        return fp[0], o


    for pp_ in range(nports):
        sfx = "" if pp_ == 0 else str(pp_)
        e(f"%ina{sfx} = mov.address-to-scalar %in{sfx}")
        e(f"%outa{sfx} = mov.address-to-scalar %out{sfx}")
    e("%spa = mov.address-to-scalar %sp")
    e("%spb = mov.address-to-scalar %sp2")
    e(f"%cgt = lshl %spa, {k(-3)}")                  # (GT = 0) >> 3: u16 b + it, << 3 = GT + 8 b (GT < 0x7F800)
    e("%sh3 = mov.shift 3")

    e("%z8 = vbcst.8 %k0z")
    e.pre.append("  %k0z = mova.i32 0")
    e.consts["%k0z"] = True

    def P(x, base_scalar, off):
        a_, p_ = x.t("pa"), x.t("pm")
        x(f"{a_} = add.rr {base_scalar}, {k(off)}" if off else f"{a_} = or {base_scalar}, {k(0)}")
        x(f"{p_} = mov.scalar-to-address {a_}")
        return p_

    def Pr(x, scalar):
        p_ = x.t("pm")
        x(f"{p_} = mov.scalar-to-address {scalar}")
        return p_

    lab = [0]

    def label(nm):
        lab[0] += 1
        return f"{nm}{lab[0]}"

    tb_, tdone = label("tbuild"), label("tdone")
    if INLINE[0] and not TONCE[0]:                      # every invocation (program memory)
        e(f"low.br ^{tb_}")
    elif INLINE[0]:   # the storages are the caller's scratch: built once per fill (the caller clears the flag)
        e("%tfp = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
        e("%tfs = mov.address-to-scalar %tfp")
        e(f"%tft = add %tfs, {TFLAG}")
        e("%tfr = mov.scalar-to-address %tft")
        e("%tflag = lda %tfr, 0")
        e(f"low.cond_br %tflag, ^{tdone}, ^{tb_} : reg<aie2p.er>")
    else:
        stp = P(e, "%spa", STATE)
        e(f"%tflag = lda {stp}, 0")
        e(f"%tx = xor %tflag, {k(0x5A5A1234)}")          # private storage starts undefined: a magic marks the table
        e(f"low.cond_br %tx, ^{tb_}, ^{tdone} : reg<aie2p.er>")
    e.L.append(f"^{tb_}:")
    for i in range(8):
        w_ = (I.bf16_bits(KV[2 * i + 1]) << 16) | I.bf16_bits(KV[2 * i])
        pp = P(e, "%spa", KVS + 4 * i)
        e(f"%kw{i} = mov.i32 {w_ - (1 << 32) if w_ >= 1 << 31 else w_}")
        e(f"st %kw{i}, {pp}, 0")
    pk = P(e, "%spa", KVS)
    e(f"%kvv = vlda.512.bf16x32 {pk}, 0")

    def tline(m, kb, pa_, pb_):
        sp_, r_, h0, h1 = e.t("ks"), e.t("kr"), e.t("kh"), e.t("kh")
        e(f"{sp_} = vbcst.16 {kb}")
        e(f"{r_} = vshuffle %kvv, {sp_}, {m18}")
        e(f"{h0} = slice {r_}[0] : {V2} -> {V1}")
        e(f"{h1} = slice {r_}[1] : {V2} -> {V1}")
        # two copies in different banks (storages A and B): a gather (vldb.4x32) costs one stall cycle unless its
        # address lanes 0 / 2 and 1 / 3 hit different banks (measured, gbench.py) -> lanes 4 q + 2, 4 q + 3 use copy 2
        for pg in (pa_, pb_) if m is None else (P(e, b_, o_ + 128 * m) for b_, o_ in (("%spa", GT), ("%spb", GT2))):
            for blk, hv_ in ((0, h0), (1, h1)):
                for half in range(2):
                    e(f"vst.256.i16x16 {hv_}, {pg}, {64 * blk + 32 * half}")
    if INLINE[0]:   # rolled (program memory)
        e(f"%tla0 = add.rr %spa, {k(GT)}")
        e(f"%tlb0 = add.rr %spb, {k(GT2)}")
        e(f"low.br ^tlh({k(0)}: reg<aie2p.er>, %tla0: reg<aie2p.er>, %tlb0: reg<aie2p.er>)")
        e.L.append("^tlh(%tl_i: reg<aie2p.er>, %tla: reg<aie2p.er>, %tlb: reg<aie2p.er>):")
        e(f"%tl_m = lt %tl_i, {k(16)}")
        e("low.cond_br %tl_m, ^tlb, ^tlx : reg<aie2p.er>")
        e.L.append("^tlb:")
        e.L.append("// LOOP-BEGIN")
        e("%tlkb = vextract.16.reg %kvv, %tl_i")       # (not a load: the KV words were just stored)
        e("%tlpa = mov.scalar-to-address %tla")
        e("%tlpb = mov.scalar-to-address %tlb")
        tline(None, "%tlkb", "%tlpa", "%tlpb")
        e(f"%tla1 = add.rr %tla, {k(128)}")
        e(f"%tlb1 = add.rr %tlb, {k(128)}")
        e("%tl_n = add %tl_i, 1")
        e("low.br ^tlh(%tl_n: reg<aie2p.er>, %tla1: reg<aie2p.er>, %tlb1: reg<aie2p.er>)")
        e.L.append("// LOOP-END")
        e.L.append("^tlx:")
        if INLINE[0] and TONCE[0]:
            e("%tfq = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
            e("%tfqs = mov.address-to-scalar %tfq")
            e(f"%tfqt = add %tfqs, {TFLAG}")
            e("%tfqr = mov.scalar-to-address %tfqt")
            e(f"st {k(1)}, %tfqr, 0")
    else:
        for m in range(16):
            e(f"%kb{m} = mov.i32 {I.bf16_bits(KV[m])}")
            tline(m, f"%kb{m}", None, None)
        stp2 = P(e, "%spa", STATE)
        e(f"st {k(0x5A5A1234)}, {stp2}, 0")
    e(f"low.br ^{tdone}")
    e.L.append(f"^{tdone}:")

    # ---- loop helpers
    def loop_begin(nm, n, args=(), start=None):
        """args: [(init, type)]; n: a count or a register bound; start: a register (else 0); returns the loop counter and
        the block arg names."""
        z = start or k(0)
        names = [f"%{nm}_a{i}" for i in range(len(args))]
        ext = "".join(f", {v}: {t}" for v, t in args)
        e(f"low.br ^{nm}h({z}: reg<aie2p.er>{ext})")
        e.L.append(f"^{nm}h(%{nm}_i: reg<aie2p.er>" + "".join(f", {n_}: {t}" for n_, (v, t) in zip(names, args)) + "):")
        e(f"%{nm}_m = lt %{nm}_i, {k(n) if isinstance(n, int) else n}")
        e(f"low.cond_br %{nm}_m, ^{nm}b, ^{nm}x : reg<aie2p.er>")
        e.L.append(f"^{nm}b:")
        e.L.append("// LOOP-BEGIN")
        return f"%{nm}_i", names

    def loop_end(nm, nxt=()):
        e(f"%{nm}_n = add %{nm}_i, 1")
        e(f"low.br ^{nm}h(%{nm}_n: reg<aie2p.er>" + "".join(f", {v}: {t}" for v, t in nxt) + ")")
        e.L.append("// LOOP-END")
        e.L.append(f"^{nm}x:")


    # ---- super-block loop (port 0; other ports are renamed copies of its section)
    INA, OUTA = ["%ina"], ["%outa"]

    def lockop(op, val):
        """acq / rel on the output ring of port 0 (port_copy moves the index)."""
        if INLINE[0]:                                   # leaf_inline: the output is plain memory
            return
        e(f"{op} {val}, {nports}")
    sec0 = len(e.L)
    j, _ = loop_begin("jl", INLINE[0] or 4)
    e(f"%j136 = mul {j}, {k(BLK)}")
    e(f"%jin = add.rr {INA[0]}, %j136")
    # -- halves: straight-line, one stream each (the scale path is one long dependency chain per half: the two
    # chains interleave)
    for h in range(2):
        e(f"%hb0_{h} = add.rr %spa, {k(HR + h * HS)}")              # this half's region (scalar)
        e(f"%hin_{h} = add.rr %jin, {k(h * 8 * ROWB)}" if h else f"%hin_{h} = or %jin, {k(0)}")
        e(f"%hdpa_{h} = add.rr %hb0_{h}, {k(HDR + 32)}")
        e(f"%hdp_{h} = mov.scalar-to-address %hdpa_{h}")
    # header words -> HDR [row][8 B]
    rows_s = []
    for h in range(2):
        for r in range(8):
            x = e.stream()
            rows_s.append(x)
            ra, w0, w1 = x.t("ra"), x.t("w0"), x.t("w1")
            x(f"{ra} = add.rr %hin_{h}, {k(ROWB * r)}" if r else f"{ra} = or %hin_{h}, {k(0)}")
            pr = Pr(x, ra)
            x(f"{w0} = lda {pr}, 0")
            x(f"{w1} = lda {pr}, 4")
            x(f"st {w0}, %hdp_{h}, {8 * r - 32}")         # one shared store pointer per half at HDR + 32
            x(f"st {w1}, %hdp_{h}, {8 * r - 28}")
    run(rows_s, 2)
    halves_s = []
    vroll = VROLL[0] and INLINE[0]   # the halves in a loop (program memory) instead of two interleaved streams
    if vroll:
        hv_i, _ = loop_begin("vl", 2)
    for h in range(1 if vroll else 2):
        x = e.stream()
        halves_s.append(x)
        hb0, hin_ = f"%hb0_{h}", f"%hin_{h}"
        hbp, pph1, pph = x.t("hbp"), x.t("pph"), x.t("pph")
        if vroll:   # half hv_i: its region, rows and parts lines from half 0's
            hm_, pm_, pq_ = x.t("hvm"), x.t("ppm"), x.t("ppq")
            x(f"{hm_} = mul {hv_i}, {k(HS)}")
            hb0 = x.t("hva")
            x(f"{hb0} = add.rr %hb0_0, {hm_}")
            x(f"{pm_} = mul {hv_i}, {k(NPARTS * 64)}")
            x(f"{pq_} = add.rr %spb, {pm_}")
            x(f"{hbp} = mov.scalar-to-address {hb0}")
            x(f"{pph1} = add.rr {pq_}, {k(PARTSS)}")
        else:
            x(f"{hbp} = mov.scalar-to-address {hb0}")
            x(f"{pph1} = add.rr %spb, {k(PARTSS + h * NPARTS * 64)}")
        x(f"{pph} = mov.scalar-to-address {pph1}")
        BASEV[0] = hbp
        # ---- headers, vectorized: ls [row][j] = low bits from the scales_l nibbles | high bits from scales_h, then
        # the lane order [s'][row][f] (j = 4 f + s'); d per [row][f] for the U / V / OFF path
        k43, H, SLv, DSv, Dv, SHv = (x.t(n) for n in ("k43", "H", "SLv", "DSv", "Dv", "SHv"))
        x(f"{k43} = vbcst.8 {k(0x43)}")
        phd = P(x, hb0, HDR)
        def lsplit(LS):
            LT, LU, a_ = x.t("LT"), x.t("LU"), x.t("lsr")
            x(f"{LT} = vshuffle {LS}, {LS}, {k(35)}")            # [j][row] = [f][s'][row]
            x(f"{LU} = vshuffle {LT}, {LT}, {k(11)}")            # f = 1 half to the front
            x(f"{a_} = vshuffle {LT}, {LU}, {m20}")              # [s'][row][f]
            return a_
        if MQ4:   # H [row][scale0, zero0, scale1, zero1]: its even 16-bit lanes are d [row][f] (twice), ls = 33
            x(f"{H} = vlda.512.i8x64 {phd}, 0")
            hv, a = x.t("hv"), x.t("lsr")
            x(f"{hv} = vshuffle {H}, {H}, {m2c}")
            x(f"{a} = vbcst.8 {k(33)}")
            chains = [(a, hv, TC_IQ4, 0, False)]
        else:
            x(f"{H} = vlda.512.i8x64 {phd}, 0")
            x(f"{SLv} = vshuffle {H}, {H}, {k(5)}")              # odd 32-bit lanes: scales_l [row][4]
            x(f"{DSv} = vshuffle {H}, {H}, {m4c}")               # even 32-bit lanes: d | scales_h
            x(f"{Dv} = vshuffle {DSv}, {DSv}, {m2c}")            # d [row]
            x(f"{SHv} = vshuffle {DSv}, {DSv}, {m3c}")           # scales_h [row]
            pp = P(x, hb0, SCM)
            sl0, sh0 = x.t("sl"), x.t("sh")
            x(f"{sl0} = slice {SLv}[0] : {V2} -> {V1}")
            x(f"{sh0} = slice {SHv}[0] : {V2} -> {V1}")
            NL, NH, NH2 = x.t("NL"), x.t("NH"), x.t("NH2")
            x(f"vst.256.i8x32 {sl0}, {pp}, 0")
            x(f"vst.256.i8x32 {sh0}, {pp}, 32")
            x(f"{NL} = vldb.unpack.u4.to.u8x64 {pp}, 0")
            x(f"{NH} = vldb.unpack.u4.to.u8x64 {pp}, 32")
            x(f"{NH2} = vshuffle {NH}, {NH}, {m20}")             # [row][j]: nibble j // 2

            def b8(v):
                r_ = x.t("b8")
                x(f"{r_} = vbcst.8 {k(v)}")
                return r_

            def dbl(v, n):
                for _ in range(n):
                    r_ = x.t("db")
                    x(f"{r_} = vadd.8 {v}, {v}")
                    v = r_
                return v
            he0, ho0, mev, mod, he1, ho1, hp, LS, LT, LU, a, hv = (x.t(n) for n in (
                "he", "ho", "mev", "mod", "he", "ho", "hp", "LS", "LT", "LU", "lsr", "hv"))
            x(f"{he0} = vband {NH2}, {b8(3)}")
            he = dbl(he0, 4)                                     # (n & 3) << 4: even j
            x(f"{ho0} = vband {NH2}, {b8(12)}")
            ho = dbl(ho0, 2)                                     # ((n >> 2) & 3) << 4: odd j
            x(f"{mev} = vbcst.16 {k(0x00FF)}")
            x(f"{mod} = vbcst.16 {k(0xFF00)}")
            x(f"{he1} = vband {he}, {mev}")
            x(f"{ho1} = vband {ho}, {mod}")
            x(f"{hp} = vbor {he1}, {ho1}")
            x(f"{LS} = vbor {NL}, {hp}")             # ls [row][j]
            a = lsplit(LS)
            x(f"{hv} = vshuffle {Dv}, {Dv}, {m18}")              # d [row][f]
            chains = [(a, hv, TC_IQ4, 0, False)]
        for a, hv, tconst, pbase, neg in chains:
            e4 = band(x, shr8(x, hv), 0x7C)
            e128 = e4
            for _ in range(5):
                e128 = add16(x, e128, e128)
            ep128 = x.t("ep")
            x(f"{ep128} = max.u16x32 {e128}, {splat16(x, 128)}")
            offv = add16(x, sub16(x, ep128, splat16(x, 3200)), band(x, hv, 0x8000))
            z128 = sub16(x, ep128, e128)
            z1024 = add16(x, add16(x, z128, z128), add16(x, z128, z128))
            z1024 = add16(x, z1024, z1024)
            sd = add16(x, band(x, hv, 0x3FF), sub16(x, splat16(x, 1024), z1024))
            if neg:                                              # -M: the f16's sign flipped
                offv = add16(x, offv, splat16(x, 0x8000))
            uv = add16(x, shr8(x, add16(x, sd, sd)), splat16(x, 0x4300))
            vv = add16(x, band(x, sd, 127), splat16(x, 0x4300))
            # (T32 reads U / V / OFF from registers: no TAB round trip)

            # ---- T = d (ls - 32) for all 8 sub-blocks in one block (lanes [s'][row][f]), split into bf16 parts
            def sreg():
                slo, shi = x.t("S"), x.t("S")
                x(f"{slo} = vshuffle {a}, {k43}, {m20}")
                x(f"{shi} = vshuffle {a}, {k43}, {m21}")
                return slo, shi

            # two 32-lane chains (sub-block pairs s' 0-1 / 2-3: slo / shi) with vmul / vmac.bf16x32 (accumulator lanes
            # 0..31): no 64-lane operand duplication (wide) or operand concats; the [row][f] tables load once
            U2, V2c, O2 = uv, vv, offv
            sl_, sh_ = sreg()
            sxs = (sl_, sh_)

            def m32(acc, s1, s2):
                r_ = x.t("ac")
                x(f"{r_} = vmul.bf16x32 {s1}, {s2}, {conf}" if acc is None else
                  f"{r_} = vmac.bf16x32 {acc}, {s1}, {s2}, {conf}")
                return r_

            def conv32(acc):
                q0, q1, h_, c_ = x.t("q"), x.t("q"), x.t("h"), x.t("cv")
                x(f"{q0} = slice {acc}[0] : {M4} -> {M1}")
                x(f"{q1} = slice {acc}[1] : {M4} -> {M1}")
                x(f"{h_} = concat({q0}, {q1}) : ({M1}, {M1}) -> {M2}")
                x(f"{c_} = vconv.bf16.fp32 {h_}")
                return c_
            accs = [None, None]
            # the steps in both chains together: each operand built right before its two uses
            for kind, v in tconst:
                ov = add16(x, O2, splat16(x, v))
                one = splat16(x, 0x3F80) if kind == "1" else None
                for c in range(2):
                    if kind == "su":
                        accs[c] = m32(accs[c], U2, add16(x, sxs[c], ov))
                    elif kind == "sv":
                        accs[c] = m32(accs[c], V2c, add16(x, sxs[c], ov))
                    elif kind == "u":
                        accs[c] = m32(accs[c], U2, ov)            # -20480 P U
                    elif kind == "v":
                        accs[c] = m32(accs[c], V2c, ov)           # -160 P V
                    elif kind == "s":
                        accs[c] = m32(accs[c], sxs[c], ov)        # -16512 P S
                    else:
                        accs[c] = m32(accs[c], one, ov)           # (2^21 + 2^14) P, 528384 P
            m1 = splat16(x, 0xBF80)
            for p in range(2):
                cs_ = [conv32(accs[hi_]) for hi_ in range(2)]
                for f in range(2):
                    # packed line [pair][s'][row] (both pairs: broadcast index 2 t + u): PARTS + (f 2 + h) NP 64 + p 64
                    dv = x.t("dv")
                    x(f"{dv} = vshuffle {cs_[0]}, {cs_[1]}, {k(2 + f)}")
                    pp_, o = x.addr(pph, f * 2 * NPARTS * 64 + (pbase + p) * 64, cursor="pph_w")
                    x(f"vst.512.bf16x32 {dv}, {pp_}, {o}")
                if p < 1:
                    accs = [m32(accs[hi_], cs_[hi_], m1) for hi_ in range(2)]
    run(halves_s, 2)
    if vroll:
        loop_end("vl")
    CTP["on"] = False


    # -- hot: two output records (slots 0, 1), each 2 iterations of 2 sub-blocks (u) x 4 chunk-halves; T row-broadcast
    # from the halves' PARTS ((4 p + 2 r + t) 64: record r = sub-block group f, iteration t = sl // 2, u = sl % 2)
    rlo, rhi = RREC[0] or (0, 2 * INLINE[0])
    rs = rn = None
    if INLINE[0] and (rlo, rhi) != (0, 2 * INLINE[0]):  # skipped records: the first super-block's record 0, the last's 1
        assert 0 <= rlo <= 1 and 2 * INLINE[0] - 1 <= rhi <= 2 * INLINE[0]
        if rlo:
            e(f"%rls = lt {j}, {k(1)}")
            rs = "%rls"
        if rhi < 2 * INLINE[0]:
            e(f"%rll = lt {k(INLINE[0] - 2)}, {j}")
            e(f"%rln = sub {k(2)}, %rll")
            rn = "%rln"
    e(f"%pp0a = add.rr %spb, {k(PARTSS)}")              # record r's parts: lines (hh NP + p) 64 at + 2 NP 64 r
    if rs:
        e(f"%pp0o = mul {rs}, {k(2 * NPARTS * 64)}")
        e("%pp0b = add.rr %pp0a, %pp0o")
        e("%pp0 = mov.scalar-to-address %pp0b")
    else:
        e("%pp0 = mov.scalar-to-address %pp0a")
    r_, (rp0,) = loop_begin("rl", rn or 2, [("%pp0", "reg<aie2p.ep>")], start=rs)
    # -- record r: qs rows of both halves realigned (vshift) and transposed for chunks 8 r .. 8 r + 7, then their 64
    # gather addresses each (GT + 8 b = (b + (GT >> 3)) << 3) at GAS + 1024 (local sub-block) + 512 h + 256 cc
    e(f"%r64 = mul {r_}, {k(64)}")
    # 544 = 32 (mod 64): a row's qs realignment depends only on its parity: aligned bases and shifts of rows 0 / 1
    for par in range(2):
        e(f"%qs{par} = add.rr %jin, {k(ROWB * par + HBY)}")
        e(f"%sh{par} = and %qs{par}, {k(63)}")
        e(f"%al{par} = and %qs{par}, {k(-64)}")
        e(f"%ar{par} = add.rr %al{par}, %r64")
    # the table base in 32 accumulator lanes: addresses = (b << 3) + GT on the MAC slot (vadd.acc.integer mode 0)
    e("%gtv1 = vbcst.32 %spa")
    e(f"%gt2s = add.rr %spb, {k(GT2)}")
    e("%gtv2 = vbcst.32 %gt2s")
    e(f"%gtv = vshuffle %gtv1, %gtv2, {k(14)}")   # u32 lanes A A B B ...
    for q in range(4):
        e(f"%gta{q} = vmov.vector512.to.accumulator512 %gtv")
    e(f"%gtacc = concat(%gta0, %gta1, %gta2, %gta3) : ({M1}, {M1}, {M1}, {M1}) -> {M4}")
    e(f"%amode = mova.i32 0")
    qstreams = []
    e(f"%pz = lshl {r_}, {k(-8)}")
    e("%pk14 = add %pz, 14")
    e("%pk15 = add %pz, 15")
    pk14, pk15 = "%pk14", "%pk15"
    for hh in range(2):
        qs_ = []
        for r in range(8):
            x = e.stream()
            qstreams.append(x)
            row = 8 * hh + r
            sh, pa_ = f"%sh{row % 2}", x.t("pa")
            x(f"{pa_} = add.rr %ar{row % 2}, {k(ROWB * (row - row % 2))}" if row > 1 else f"{pa_} = or %ar{row % 2}, {k(0)}")
            pv = Pr(x, pa_)
            va, vb, q = x.t("va"), x.t("vb"), x.t("q")
            x(f"{va} = vlda.512.i8x64 {pv}, 0")
            x(f"{vb} = vlda.512.i8x64 {pv}, 64")
            x(f"{q} = vshift {va}, {vb}, {sh}")
            qs_.append(q)
        x = e.stream()
        qstreams.append(x)
        vs = qs_
        for d in (4, 2, 1):
            out = list(vs)
            for i in range(8):
                jj = i ^ d
                if i < jj:
                    a_, b_ = x.t("tq"), x.t("tq")
                    x(f"{a_} = vshuffle {vs[i]}, {vs[jj]}, {pk14}")
                    x(f"{b_} = vshuffle {vs[i]}, {vs[jj]}, {pk15}")
                    out[i], out[jj] = a_, b_
            vs = out
        for i in range(8):                       # local chunk i: sub-block i // 2, cc = i % 2
            x = e.stream()
            qstreams.append(x)
            pg = P(x, "%spb", GAS + 1024 * (i // 2) + 512 * hh + 512)     # +512: offsets -512 .. 448
            u8s = []
            for hb_ in range(2):                     # bytes 32 hb_ .. + 31 (rows 4 hb_ .. + 3): 32 x u32 b << 3
                bv, u8 = x.t("bv"), x.t("u8")
                x(f"{bv} = slice {vs[i]}[{hb_}] : {V2} -> {V1}")
                x(f"{u8} = vups.4x.w-to-c.unsigned {bv}, %sh3")
                u8s.append(u8)
            u4, a_ = x.t("u4"), x.t("ua")
            x(f"{u4} = concat({u8s[0]}, {u8s[1]}) : ({M2}, {M2}) -> {M4}")
            x(f"{a_} = vadd.acc.integer {u4}, %gtacc, %amode")         # + GT (64 lanes, MAC slot)
            for q2 in range(4):
                q1 = x.t("ga")
                x(f"{q1} = slice {a_}[{q2}] : {M4} -> {M1}")
                x(f"vst.acc {q1}, {pg}, {256 * (i % 2) + 64 * q2 - 512}")
    run(qstreams, 6)
    e(f"%qp0a = add.rr %spb, {k(GAS + 512)}")           # GA of local sub-block 0, +512 (offsets -512 .. 448)
    e("%qp0 = mov.scalar-to-address %qp0a")
    rq = "%qp0"
    e(f"%jso = mul {r_}, {k(OUT_B)}")
    if INLINE[0]:                               # no output ring: super-block j's records at 2 j + r
        e(f"%jsj = mul {j}, {k(2 * OUT_B)}")
        e("%jsr = add.rr %jso, %jsj")
        e(f"%osa = add.rr {OUTA[0]}, %jsr")
    else:
        e(f"%osa = add.rr {OUTA[0]}, %jso")
    e("%osp = mov.scalar-to-address %osa")
    lockop("acq", "%km1o")
    e.pre.append("  %km1o = mov.i32 -1")
    e.consts["%km1o"] = True
    e("%op = copy %osp : reg<aie2p.ep> -> reg<aie2p.mpfs>")
    e("%sl = vlda.store-fifo.low512 %osp, 0")
    e("%f0 = vlda.store-fifo.high512 %osp, %sl, 64")
    e("%pos = mova.fifo.store.position 0")
    # the two sl iterations unrolled into one schedule (-5.5% decoder)
    s_, (hf, hp, hq, sq, sp0) = r_, ("%f0", "%op", "%pos", rq, rp0)
    st = {"fifo": (hf, hp, hq), "nf": 0}
    # loop-local constants: hoisted ones live across the prep and get rematerialized at every use
    nonlocal_consts = {}
    # from an opaque zero (loop counter >> 8): identical constants elsewhere would be CSE'd into one long-lived value,
    # which the allocator then rematerializes at every use (52 extra load-slot ops per iteration)
    e(f"%hz = lshl {s_}, {k(-8)}")
    for v, nm in ((2, "m2c"), (3, "m3c"), (4, "m4c"), (52, "m52"), (53, "m53"), (60, "conf")):
        e(f"%hk{v} = add %hz, {v}")
        nonlocal_consts[nm] = f"%hk{v}"
    m2c, m3c, m4c, m52, m53, conf = (nonlocal_consts[n] for n in ("m2c", "m3c", "m4c", "m52", "m53", "conf"))

    def push(x, wv):
        st["nf"] += 1
        nf, fifo = st["nf"], st["fifo"]
        nxt = (f"%xf{nf}", f"%xp{nf}", f"%xq{nf}")
        x(f"{nxt[0]}, {nxt[1]}, {nxt[2]} = vst.push.bfp16ebs8.from.fp32 {fifo[0]}, {wv}, {fifo[1]}, {fifo[2]}")
        st["fifo"] = nxt

    def flush(x):
        st["nf"] += 1
        nf, fifo = st["nf"], st["fifo"]
        nxt = (f"%xf{nf}", f"%xp{nf}", f"%xq{nf}")
        x(f"{nxt[0]}, {nxt[1]}, {nxt[2]} = vst.flush.512 {fifo[0]}, {fifo[1]}, {fifo[2]}")
        st["fifo"] = nxt

    order = [(0, 0), (0, 1), (1, 0), (1, 1)]           # (chunk cc of the sub-block, half)
    frags, tails = [], []
    gpp = {}
    for t_, u in [(t_, u) for t_ in range(2) for u in range(2)]:
      if u == 0:
        hi_acc = {}
      for i, (cc, hh) in enumerate(order):
          # G streams: per 16 bytes one address vector -> 4 gathers -> compaction (small register footprint, many in
          # flight); C: lo / hi extraction, T, X, MACs, push
          ents = {}
          for half in range(2):
              for blk in range(2):
                  x = e.stream()
                  frags.append(x)
                  tails.append([])
                  ad = x.t("ad")
                  if (t_, u) not in gpp:    # GA of (t, u) from the scalar base: one add + one move
                      gpp[(t_, u)] = P(x, "%qp0a", 2048 * t_ + 1024 * u)
                  gpu_ = gpp[(t_, u)]
                  x(f"{ad} = vlda.512.i8x64 {gpu_}, {512 * hh + 256 * cc + 128 * half + 64 * blk - 512}")
                  gls = []
                  for hf_ in range(2):
                      av, gl, gh = x.t("av"), x.t("gl"), x.t("gh")
                      x(f"{av} = slice {ad}[{hf_}] : {V2} -> {V1}")
                      x(f"{gl} = vldb.4x32.lo {av}")
                      x(f"{gh} = vldb.4x32.hi {av}")
                      gls.append((gl, gh))
                  gcs = []
                  for gl, gh in gls:
                      gc = x.t("gc")
                      x(f"{gc} = concat({gl}, {gh}) : ({V1}, {V1}) -> {V2}")
                      gcs.append(gc)
                  ev = x.t("en")
                  x(f"{ev} = vshuffle {gcs[0]}, {gcs[1]}, {m4c}")
                  ents[(half, blk)] = ev
          x = e.stream()
          frags.append(x)
          ppt = sp0
          lo_p, hi_p = [], []
          lds = []
          for p in range(2 if NPARTS != 4 else 0):  # the T parts first: their load latency overlaps the gathers
              ld = x.t("pl")
              x(f"{ld} = vlda.512.bf16x32 {ppt}, {(hh * NPARTS + p) * 64}")
              lds.append(ld)

          def build_xs():
              for half in range(2):
                  for mode, dst in ((m2c, lo_p), (m3c, hi_p)):
                      pv = x.t("xp")
                      x(f"{pv} = vshuffle {ents[(half, 0)]}, {ents[(half, 1)]}, {mode}")
                      dst.append(pv)
              xs_ = []
              for ps in (lo_p, hi_p):
                  x4 = x.t("X")
                  x(f"{x4} = concat({ps[0]}, {ps[1]}) : ({V2}, {V2}) -> {V4}")
                  xs_.append(x4)
              return xs_
          xs = build_xs()
          accs = [None, None]

          for p in range(2):
              ld = lds[p]
              bc, ra, rb, t4 = x.t("bc"), x.t("ra"), x.t("rb"), x.t("TB")
              x(f"{bc} = vbroadcast.bf16x8.to.bf16x32 {ld}, {2 * t_ + u}")
              x(f"{ra} = vshuffle {bc}, {bc}, {m52}")
              x(f"{rb} = vshuffle {bc}, {bc}, {m53}")
              x(f"{t4} = concat({ra}, {rb}) : ({V2}, {V2}) -> {V4}")
              for n_, x4 in enumerate(xs):
                  accs[n_] = mac(x, accs[n_], t4, x4)
          hi_acc[(cc, hh)] = accs[1]
          pt = e.stream()
          # panel k-block order per sub-block: chunk c's low nibbles then its high nibbles, each [h0, h1]:
          # (4s, 4s+2, 4s+1, 4s+3) -- the activations use the same k-block permutation
          push(pt, accs[0])
          if hh == 1:
              push(pt, hi_acc[(cc, 0)])
              push(pt, hi_acc[(cc, 1)])
              if cc == 1:
                  flush(pt)                          # every 8 pushes, and at the end of an iteration
          tails.append(pt.L)
    if frags:
        run(frags, 6, tails, hv=24)
    e(f"%spc0 = copy {sp0} : reg<aie2p.ep> -> reg<aie2p.ep>")
    e(f"%spn0 = padda %spc0, {2 * NPARTS * 64}")
    lockop("rel", "%k1r")
    loop_end("rl", [("%spn0", "reg<aie2p.ep>")])
    e.pre.append("  %k1r = mova.i32 1")
    e.consts["%k1r"] = True
    loop_end("jl")
    sec = e.L[sec0:]
    for pp_ in range(1, nports):
        e.L += port_copy(sec, pp_, nports)
    e("return")
    e.L.append("}")
    return "\n".join(e.L[:hdr_end] + place_consts(e.pre, e.L[hdr_end:]))


def leaf_inline(nsb, name="inl", vroll=False, tonce=True, rec=None):
    """The leaf's body for inlining into another core function: decodes nsb super-blocks of an input record
    ([16 rows][RECROW]) at %in into OUT_B records at %out (2 per super-block, no output ring), storages at %sp
    (SCR_B, 2048-aligned) and %sp2 (SCR2_B), all defined by the caller (the table is rebuilt each call). Returns the
    lines (not renamed; the caller prefixes its values and labels). rec: (first, end) of the records 2 j + r to decode
    (the others are left as they are)."""
    INLINE[0], VROLL[0], TONCE[0], RREC[0] = nsb, vroll, tonce, rec
    try:
        txt = leaf(name, 1)
    finally:
        INLINE[0], VROLL[0], TONCE[0], RREC[0] = 0, False, True, None
    out = []
    for ln in txt.split("\n"):
        t = ln.strip()
        if (t.startswith("global.rodata.def ") or t.startswith("low.func.def ") or " = resource<" in t or
                " = storage " in t or (" = storage_address " in t and "%fzs" not in t) or t in ("return", "}")):
            continue
        out.append(ln)
    return out


def place_consts(pre, body):
    """Constants used inside loops: a copy per innermost loop body (defined at its start, uses renamed), so no constant
    lives across loops (long-lived constants got rematerialized at every use). Elsewhere: right before the first use,
    and before the table build's branch when used both inside and outside that conditional block."""
    import re
    defs = {d.split("=")[0].strip(): d for d in pre}
    # innermost loop of each line
    stack, owner, nloops = [], [], 0
    for l in body:
        if l == "// LOOP-BEGIN":
            nloops += 1
            stack.append(nloops)
            owner.append(None)
        elif l == "// LOOP-END":
            stack.pop()
            owner.append(None)
        else:
            owner.append(stack[-1] if stack else None)
    begin = {}
    cur = 0
    for n_, l in enumerate(body):
        if l == "// LOOP-BEGIN":
            cur += 1
            begin[cur] = n_
    local = {}                      # loop -> constants used in its own lines
    out_body = []
    for l, o in zip(body, owner):
        if o is not None:
            names = [nm for nm in re.findall(r"%k[\w]+", l) if nm in defs]
            for nm in names:
                local.setdefault(o, []).append(nm) if nm not in local.get(o, []) else None
            l = re.sub(r"%k[\w]+", lambda m: f"{m.group(0)}_L{o}" if m.group(0) in defs else m.group(0), l)
        out_body.append(l)
    body = out_body
    first, cond, region_of = {}, None, {}
    for n_, l in enumerate(body):
        t_ = l.strip()
        if t_.startswith("^tbuild") and not (INLINE[0] and not TONCE[0]):   # (inline: conditional on the fill's flag)
            cond = (max(i for i in range(n_) if body[i].strip().startswith("low.cond_br")), "^tdone")
        elif cond and t_.startswith(cond[1]):
            cond = None
        if owner[n_] is not None:
            continue
        for nm in re.findall(r"%k[\w]+", l):
            if nm not in defs:
                continue
            region_of.setdefault(nm, set()).add(cond[0] if cond else None)
            first.setdefault(nm, cond[0] if cond else n_)
    for nm, regs in region_of.items():
        if len(regs) > 1:
            first[nm] = min(first[nm], min(r for r in regs if r is not None))
    at = {}
    for nm, n_ in first.items():
        at.setdefault(n_, []).append(defs[nm])
    # a loop-local copy right before its first use among the loop's own lines (blocks of a loop body run in order)
    for lp, names in local.items():
        for nm in names:
            loc = f"{nm}_L{lp}"
            n_ = next(i for i, l in enumerate(body) if owner[i] == lp and re.search(re.escape(loc) + r"\b", l))
            at.setdefault(n_, []).append(defs[nm].replace(nm + " ", f"{loc} ", 1))
    out = []
    for n_, l in enumerate(body):
        if l == "// LOOP-BEGIN":
            out += at.get(n_, [])
            continue
        out += at.get(n_, [])
        if not l.startswith("// LOOP"):
            out.append(l)
    return out

def array(units, nports=1):
    S = "reg<aie2p.array.scalar : index>"
    RS, NS, P = 20 * BLK // 4, 5, units // 5           # row stride (words): K / 256 super-blocks of BLK bytes
    return f"""aie2p.target<array> @array_target
aie2p.target<core> @core_target

low.func.def public retain target<amd.xdna.aie2p.array>(@array_target) abi(array_program) @probe() asm {{
  %n0 = constant.u32 0 : {S}
  %n1 = constant.u32 1 : {S}
  %n2 = constant.u32 2 : {S}
  %rec = constant.u32 {units} : {S}
  %orec = constant.u32 {8 * units} : {S}
  %origin = constant.u64 0 : reg<aie2p.array.offset : offset>
  %workers = group %n1
  %k = worker %workers, %n0, @dec
  constrain.location %k, %n0, %n2
  %ib = binding 0, "read"
  %ob = binding 1, "write"
  %np = constant.u32 {nports} : {S}
  %o_all = receiver %ob, 0 : reg<aie2p.array.receiver : tile<{nports}x{8 * units}x{OUT_B // 4}xi32>>
""" + "".join(f"""  %io{p} = constant.u64 {p * 80 * RS * 4} : reg<aie2p.array.offset : offset>
  %is{p} = sender %ib, 0 : reg<aie2p.array.sender : tile<{P}x{NS}x16x{ROWB // 4}xi32, #encoding.layout.strided<strides=[{BLK}, {16 * RS}, {RS}, 1]>>>
  %iv{p} = view.sender %is{p}, %io{p} : reg<aie2p.array.sender : tile<16x{ROWB // 4}xi32>>
  %ri{p} = receiver %k, {p} : reg<aie2p.array.receiver : tile<16x{ROWB // 4}xi32>>
  %chi{p} = channel %iv{p}, %ri{p}, %n1, %rec : reg<aie2p.array.channel : tile<16x{ROWB // 4}xi32>>
  %orc{p} = partition.receiver %o_all, %origin, %n{p}, %np : reg<aie2p.array.receiver : tile<{OUT_B // 4}xi32>>
  %so{p} = sender %k, {nports + p} : reg<aie2p.array.sender : tile<{OUT_B // 4}xi32>>
  %cho{p} = channel %so{p}, %orc{p}, %n2, %orec : reg<aie2p.array.channel : tile<{OUT_B // 4}xi32>>
  constrain.leaf_sync %cho{p}
""" for p in range(nports)) + """  return
}
"""


# panel position -> k-block within a pass: per sub-block of 4, (0, 2, 1, 3)
KPERM = [4 * (p // 4) + (0, 2, 1, 3)[p % 4] for p in range(128)]


def oracle(raw):
    """raw [80 rows][20 super-blocks][BLK] -> panel-order fragments [pass][slab][kb 128][h 2][72]."""
    w = weights(raw)
    w = w.reshape(80, 640, 8)                                     # [row][kb][8]
    frag = w.reshape(5, 2, 8, 5, 128, 8).transpose(3, 0, 4, 1, 2, 5)   # [pass][slab][kb][h][row][8]
    frag = frag[:, :, KPERM]                                      # the panel's k-block order
    return Q.bfp_hw(frag.reshape(-1, 8, 8)).reshape(5, 5, 128, 2, 72)


def weights(raw):
    """raw [rows][super-blocks][BLK] -> the decoder's f32 weights [rows][super-blocks][256] (its exact op order)."""
    import numpy as np
    if MQ4:                                           # w = scale (K half) x signed code, at IQ4_XS's positions
        d = np.stack([raw[..., 0:2].copy().view(np.float16)[..., 0], raw[..., 4:6].copy().view(np.float16)[..., 0]],
                     -1).astype(np.float64)
        _, _, q = I.iq4xs_fields(raw)
        w = (np.repeat(d, 128, -1) * np.array(KV, np.float64)[q]).astype(np.float32)
    else:
        d, ls, q = I.iq4xs_fields(raw)
        w = (np.repeat(d[..., None] * (ls - 32), 32, -1) * np.array(KV, np.float64)[q]).astype(np.float32)
    return w


def main():
    import numpy as np
    work = sys.argv[1]
    units = 25
    nports = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    os.makedirs(work, exist_ok=True)
    src = array(units, nports) + "\n" + leaf("dec", nports) + "\n"
    open(f"{work}/p.loom", "w").write(src)
    LC = hrx_paths.LOOM_COMPILE
    env = dict(hrx_paths.env(), LOOM_EXP_LOCKED_PACK="1")
    r = subprocess.run([LC, f"{work}/p.loom", "--root=@probe", "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11",
                        f"--output={work}/p.xdna", "--compile-report=text", f"--compile-report-output={work}/report.txt"],
                       capture_output=True, text=True, env=env)
    if r.returncode:
        sys.exit("compile failed\n" + r.stderr[-4000:])
    for l in open(f"{work}/report.txt"):
        if l.startswith("COMPILE-REPORT: emission"):
            print("compiled:", " ".join(t for t in l.split() if t.split("=")[0] in ("instructions", "code_bytes")))
    if MQ4:   # <model> is the HFQ file: the first K = 5120 MQ4 tensor's rows as the loader leaves them
        import hfq_index
        t = next(x for x in hfq_index.read(sys.argv[2]) if x[1] == 44 and x[2][1] == 5120 and x[2][0] >= 1024)
        raw = hfq_index.mq4_rows(sys.argv[2], t, slice(160, 160 + 80 * nports))
    else:
        import gguf   # llama.cpp's gguf-py on PYTHONPATH
        rd = gguf.GGUFReader(sys.argv[2])
        t = next(x for x in rd.tensors if x.tensor_type.name == FMT and len(x.shape) == 2 and int(x.shape[0]) == 5120
                 and int(x.shape[1]) >= 1024)
        raw = np.asarray(t.data).view(np.uint8).reshape(-1, 20, BLK)[160:160 + 80 * nports].copy()   # 80 rows per port
    np.concatenate([raw.reshape(-1), np.zeros(1024, np.uint8)]).tofile(f"{work}/in.bin")   # + slack: records over-read
    np.zeros(nports * 8 * units * OUT_B, np.uint8).tofile(f"{work}/o0.bin")
    r = subprocess.run([hrx_paths.XDNA_RUN, "--columns=1", f"--image={work}/p.xdna", "--entry=probe", "--binding_memory=system",
                        f"--binding={work}/in.bin", f"--binding={work}/o0.bin", f"--output=1={work}/out.bin"],
                       capture_output=True, text=True, env=env, timeout=60)
    if r.returncode:
        sys.exit("run failed\n" + r.stdout[-1500:] + r.stderr[-1500:])
    got = np.fromfile(f"{work}/out.bin", np.uint8).reshape(nports, 5, 5, 128, 2, 72)
    ref = np.stack([oracle(raw[80 * p:80 * p + 80]) for p in range(nports)])
    bad = np.any(got != ref, axis=-1)
    print(f"fragments differing: {int(bad.sum())} of {bad.size}")
    if bad.any():
        i = np.argwhere(bad)[0]
        print("first bad [pass, slab, kb, h]:", i.tolist(), "\n got", got[tuple(i)][:18].tolist(), "\n ref", ref[tuple(i)][:18].tolist())


def port_copy(sec, p, nports):
    """Port p's section: port 0's with every value / label defined in it suffixed, port 0's resources swapped for
    port p's, and the output ring's acquire / release index moved."""
    import re
    defined = set()
    for l in sec:
        t = l.strip()
        m = re.match(r"((?:%[\w.]+)(?:, %[\w.]+)*) = ", t)
        if m:
            defined.update(re.findall(r"%[\w.]+", m.group(1)))
        m = re.match(r"\^([\w]+)\((.*)\):", t)
        if m:
            defined.update(re.findall(r"(%[\w.]+):", m.group(2)))
    labels = {m for l in sec for m in re.findall(r"\^([\w]+)", l)}
    swap = {"%in": f"%in{p}", "%out": f"%out{p}", "%ina": f"%ina{p}", "%outa": f"%outa{p}"}
    sfx = f"_p{p}"

    def fix(l):
        l = re.sub(r"%[\w.]+", lambda m: swap.get(m.group(0), m.group(0) + sfx if m.group(0) in defined else m.group(0)), l)
        l = re.sub(r"\^([\w]+)", lambda m: "^" + m.group(1) + (sfx if m.group(1) in labels else ""), l)
        l = re.sub(r"^(\s*(?:acq|rel) %\w+, )" + str(nports) + r"$", lambda m: m.group(1) + str(nports + p), l)
        l = l.replace("{index = 0, source_type", f"{{index = {p}, source_type").replace(
            f"{{index = {nports}, source_type", f"{{index = {nports + p}, source_type")
        return l
    return [fix(l) for l in sec]


if __name__ == "__main__":
    main()
