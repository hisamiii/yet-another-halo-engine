#!/usr/bin/env python3
"""Generate the FlashAttention-style causal prefill attention (register softmax).

usage: gen_attn_fa.py [out.loom]

Same launch, bindings and configs as gen_attn_hip.py (kernel yah_attn_wmma), so
the emitter can swap it in (YAH_ATTN_FA=1). Not HIP's arithmetic order: checked
by the T1 numerics gate, not md5.

Why: gen_attn_hip routes every 16-key tile through LDS twice (S partials ->
LDS -> softmax in a 4-lanes-per-row layout -> f16 P -> LDS -> P.V) with three
barriers; at pp8192 that kernel is LDS-bound (37% of wave time in full
lgkmcnt(0) drains, 81% LDS busy, 53% of the WMMA floor).

Layout: block = 32 query tokens x 2 query heads of one GQA group (4 query
blocks of 16), 8 waves: wave w owns query block w/2 and head-dim half w%2.
  S^T = K Q^T  (K as A: rows keys; Q^T as B: columns queries) over the wave's
               128 dims, 8 WMMAs; the pair adds partials through a private
               LDS slot (s = own + partner: f32 add commutes, both waves agree)
  softmax in registers: lane (q = lane%16, h = lane/16) holds keys 8h..8h+7 of
               query q (K rows are permuted at staging: LDS row 2i+h holds key
               8h+i), so the row max is an in-lane max + one xor-16 shuffle and
               the row sum stays lane-partial until the epilogue
  P^T as B   : f16(p) of both halves, one xor-16 shuffle of 4 dwords, concat
  O^T += V^T P^T: V^T as A (rows dims, permuted at staging so element i of
               lane half h is dim 8h+i: contiguous output per lane), 8 WMMAs
  scores are log2-scaled in the exp: p = exp2(s*log2e - m*log2e) (one fma)
K/V tiles are double-buffered in LDS, staged one tile ahead from registers
loaded two tiles ahead; two barriers per tile (A: QK, S store, stage next /
B: softmax, P.V).

Knobs: YAH_ATTN_FA_QKF=n fence after every n QK MMAs (0 = none),
YAH_ATTN_FA_PVF=n the same for P.V, YAH_ATTN_FA_SKIP=1 skips the O rescale when
no row max of the wave changed (exact).
"""
import os
import sys

V4 = "vector<4xf32>"
V8 = "vector<8xf32>"
V8H = "vector<8xf16>"
V16H = "vector<16xf16>"
V4I = "vector<4xi32>"

MAX_TOKENS = int(os.environ.get("YAH_ATTN_MAX_TOKENS", "2048"))
F16OUT = os.environ.get("YAH_ATTN_F16OUT", "1") == "1"
QKF = int(os.environ.get("YAH_ATTN_FA_QKF", "2"))
PVF = int(os.environ.get("YAH_ATTN_FA_PVF", "0"))
SKIP = os.environ.get("YAH_ATTN_FA_SKIP", "0") == "1"
# THR (with SKIP): FA4's conditional rescale threshold in log2 units: the
# running max is only raised (and O, l rescaled) when some lane's tile max
# exceeds it by more than THR, so p can reach 2^THR. 0 = exact skip (only
# when no row max of the wave grew: same values as always rescaling).
THR = float(os.environ.get("YAH_ATTN_FA_THR", "0"))
# S8: store/load the S partial as one 32-byte vector per lane (the two-plane
# split materialized 8 copies; costs a 2-way bank conflict on 4 accesses/tile)
S8 = os.environ.get("YAH_ATTN_FA_S8", "0") == "1"
# MSKIF: apply the causal/context mask only on tiles that need it
# (workgroup-uniform branch) instead of on every tile
MSKIF = os.environ.get("YAH_ATTN_FA_MSKIF", "0") == "1"
# HIPNUM: HIP's rounding wherever it is cheap, so the output stays near the
# HIP-order golden (T1): p = exp2((s - m) * log2e) instead of one fma, the
# row sum as HIP groups it per tile (4-key partials, (a + b) + (c + d) across
# the lane halves, sum = fma(sum, prior, part)) and o / sum by IEEE division.
# Scores already match (two 8-step chains over the head-dim halves, s0 + s1);
# left different: HIP's fmaf chain on partly visible diagonal tiles.
HIPNUM = os.environ.get("YAH_ATTN_FA_HIPNUM", "1") == "1"
# key loop unrolled by 2 with the recurrence schedule: 76.8 -> 73.8 M cycles
# (pp8192, real layer-3 inputs) from fewer back-edge copies
POL = os.environ.get("YAH_ATTN_FA_POL", "unroll(%c2) schedule(recurrence)")
QK2 = os.environ.get("YAH_ATTN_FA_QK2", "0") == "1"
# SWZ: workgroup order with the head pair fastest, so the 3 pairs of a KV head
# (GQA 6) run side by side on the same K/V tiles (L2 hits instead of 3 DRAM
# reads; query block fastest gave 35% L2 hits, 8.2 GB per layer at pp8192).
# LPT: longest query blocks first (FA4's ordering).
# Both on: L2 hits 35% -> 73%, 8.2 -> 2.8 GB read per layer, 73.9 -> 70.3 M.
SWZ = os.environ.get("YAH_ATTN_FA_SWZ", "1") == "1"
LPT = os.environ.get("YAH_ATTN_FA_LPT", "1") == "1"
# PVZ: P.V into a zero accumulator, then O = fma(O, alpha, pv) in place (the
# rescale-then-accumulate form copied all 64 O VGPRs around the back edge)
PVZ = os.environ.get("YAH_ATTN_FA_PVZ", "0") == "1"

# LDS (bytes). The Q stage (prologue only) aliases the rest.
# Single K and V buffers: V(i) is staged in phase A of tile i and read in its
# phase B; K(i+1) is staged in phase B and read in the next phase A: every
# write is a barrier away from the reads on either side. Pitches are
# conflict-free under b128's 8-lane passes (128 B): K rows 528 B, V rows 48 B
# (32-B rows put lanes r and r+4 on the same banks: 39% of LDS-active cycles).
KT_PITCH = 264                       # K: 16 keys x 256 dims (+8 pad)
VT_PITCH = 24                        # V^T: 256 dims x 16 keys (+8 pad)
# K(i+1) and V(i+1) are loaded at the top of phase A and staged in phase B of
# the same tile: nothing in flight crosses the loop back edge, where the
# compiler drains vmcnt(0) (15% of wave time when the loads were carried).
# So V is double-buffered (V(i) is read in phase B while V(i+1) is staged).
K_OFF = 0                            # 16 x 264 x 2 = 8448
V_OFF = 8448                         # 2 x 256 x 24 x 2 = 24576
S_OFF = V_OFF + 2 * 256 * VT_PITCH * 2   # S partials: 2 planes x 256 x 4 f32
POOL = S_OFF + 8192                  # 41216: three workgroups per WGP
Q_PITCH = 264
POOL = max(POOL, 64 * Q_PITCH * 2)   # 33792 (Q stage): three workgroups per WGP
assert POOL <= 65536


def gen():
    L = []
    e = L.append
    e("// GENERATED by tools/gen_attn_fa.py -- edit the generator.")
    e("// FlashAttention-style causal prefill attention, register softmax.")
    e("amdgpu.target<gfx1151> @attn_fa_w32 {subgroup_size = 32}")
    e("")
    e("config.def @attention_prefill.cache_capacity = 2048 : index")
    e(f"config.decl @attention_prefill.token_count : %value: index where [range(%value, 1, {MAX_TOKENS})]")
    e("config.decl @attention_prefill.num_heads : %value: index where [range(%value, 1, 4096)]")
    e("config.decl @attention_prefill.num_kv_heads : %value: index where [range(%value, 1, 4096)]")
    e("config.decl @attention_prefill.gqa : %value: index where [range(%value, 1, 4096)]")
    e("config.decl @attention_prefill.head_dim : %value: index where [range(%value, 1, 1024)]")
    e("config.decl @attention_prefill.start_pos : %value: index where [range(%value, 0, 1073741824)]")
    e("")
    e("kernel.def target(@attn_fa_w32) @yah_attn_wmma() {")
    e("  %token_count = config.get @attention_prefill.token_count : index")
    e("  %c1 = index.constant 1 : index")
    e("  %c2 = index.constant 2 : index")
    e("  %c31 = index.constant 31 : index")
    e("  %c32 = index.constant 32 : index")
    e("  %c256 = index.constant 256 : index")
    e("  %nh = config.get @attention_prefill.num_heads : index")
    e("  %pairs = index.div %nh, %c2 : index")
    e("  %tp = index.add %token_count, %c31 : index")
    e("  %qblocks = index.div %tp, %c32 : index")
    e("  kernel.launch.config workgroups(%qblocks, %pairs, %c1) workgroup_size(%c256, %c1, %c1) : index")
    e("} launch(%query: buffer, %gate: buffer, %key_cache: buffer, %value_cache: buffer, %output: buffer, %lse: buffer) {")
    e("  %base = index.constant 0 : offset")
    for v in (0, 1, 2, 3, 4, 6, 8, 15, 16, 31, 32, 64, 128, 256, 1024, 4096, 6144):
        e(f"  %c{v} = index.constant {v} : index")
    e("  %cache_capacity = config.get @attention_prefill.cache_capacity : index")
    e("  %token_count0 = config.get @attention_prefill.token_count : index")
    e(f"  %B = index.assume %token_count0 [range(%token_count0, 1, {MAX_TOKENS})] : index")
    e("  %start_pos = config.get @attention_prefill.start_pos : index")
    e("  %zero = scalar.constant 0.0 : f32")
    e("  %one = scalar.constant 1.0 : f32")
    e("  %ninf = scalar.constant -3.4028234663852886e+38 : f32")
    e("  %log2e = scalar.constant 1.4426950408889634 : f32")
    e("  %qscale = scalar.constant 0.0625 : f32")
    e("  %zh8 = vector.constant 0.0 : vector<8xf16>")
    e(f"  %zeros8 = vector.constant 0.0 : {V8}")
    e(f"  %ones8 = vector.constant 1.0 : {V8}")
    e(f"  %log2e8 = vector.splat %log2e : {V8}")
    e(f"  %ninf8 = vector.splat %ninf : {V8}")
    e("  %m = index.constant 16 : index")
    e("  %n = index.constant 16 : index")
    e("  %k = index.constant 16 : index")
    e("  %x16 = scalar.constant 16 : i32")
    e("  %x32 = scalar.constant 32 : i32")
    e("  %qtot = index.mul %B, %c6144 : index")
    e("  %kvtot = index.mul %cache_capacity, %c1024 : index")
    e("  %q_na, %g_na, %k_na, %v_na, %o_na = buffer.assume.noalias %query, %gate, %key_cache, %value_cache, %output : buffer, buffer, buffer, buffer, buffer")
    e("  %q_flat = buffer.view %q_na[%base] : buffer -> view<[%qtot]xf32>")
    e("  %g_flat = buffer.view %g_na[%base] : buffer -> view<[%qtot]xf32>")
    oty = "f16" if F16OUT else "f32"
    e(f"  %o_flat = buffer.view %o_na[%base] : buffer -> view<[%qtot]x{oty}>")
    e("  %k_flat = buffer.view %k_na[%base] : buffer -> view<[%kvtot]xf16>")
    e("  %cap15 = index.add %cache_capacity, %c15 : index")
    e("  %vtiles = index.div %cap15, %c16 : index")
    e("  %vpitch = index.mul %vtiles, %c16 : index")
    e("  %vtot = index.mul %vpitch, %c1024 : index")
    e("  %vlast = index.sub %vpitch, %c16 : index")
    e("  %v_flat = buffer.view %v_na[%base] : buffer -> view<[%vtot]xf16>")
    e(f"  %pool_bytes = index.constant {POOL} : offset")
    e("  %pool = buffer.alloca<workgroup> align(16) %pool_bytes : buffer")
    e(f"  %qs_view = buffer.view %pool[%base] : buffer -> view<64x{Q_PITCH}xf16>")
    e(f"  %q_lay = encoding.layout.strided [1, {Q_PITCH}] : encoding<layout>")
    e("  %q_fr = buffer.view %pool[%base] : buffer -> view<256x64xf16, %q_lay>")
    e(f"  %k_o = index.constant {K_OFF} : offset")
    e(f"  %k_view = buffer.view %pool[%k_o] : buffer -> view<16x{KT_PITCH}xf16>")
    e(f"  %v_o = index.constant {V_OFF} : offset")
    e(f"  %v_view = buffer.view %pool[%v_o] : buffer -> view<512x{VT_PITCH}xf16>")
    e(f"  %s_o = index.constant {S_OFF} : offset")
    # two planes of 4 f32 per lane, so each b128 access is contiguous across lanes
    e("  %s_view = buffer.view %pool[%s_o] : buffer -> view<512x4xf32>")
    e("  %s8_view = buffer.view %pool[%s_o] : buffer -> view<256x8xf32>")
    # ids
    if SWZ or LPT:
        e("  %wgx = kernel.workgroup.id<x> : index")
        e("  %wgy = kernel.workgroup.id<y> : index")
        e("  %nh_ = config.get @attention_prefill.num_heads : index")
        e("  %npairs = index.div %nh_, %c2 : index")
        e("  %tpq = index.add %B, %c31 : index")
        e("  %nqb = index.div %tpq, %c32 : index")
        if SWZ:
            e("  %wgl0 = index.mul %wgy, %nqb : index")
            e("  %wgl = index.add %wgl0, %wgx : index")
            e("  %hp = index.rem %wgl, %npairs : index")
            e("  %qb0 = index.div %wgl, %npairs : index")
        else:
            e("  %hp = index.add %wgy, %c0 : index")
            e("  %qb0 = index.add %wgx, %c0 : index")
        if LPT:
            e("  %nqb1 = index.sub %nqb, %c1 : index")
            e("  %qb = index.sub %nqb1, %qb0 : index")
        else:
            e("  %qb = index.add %qb0, %c0 : index")
    else:
        e("  %qb = kernel.workgroup.id<x> : index")
        e("  %hp = kernel.workgroup.id<y> : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %wave = index.div %tid, %c32 : index")
    e("  %lane = index.rem %tid, %c32 : index")
    e("  %sub = index.rem %lane, %c16 : index")
    e("  %half = index.div %lane, %c16 : index")
    e("  %wqb = index.div %wave, %c2 : index")       # query block 0..3
    e("  %hd = index.rem %wave, %c2 : index")        # head-dim half
    e("  %hd128 = index.mul %hd, %c128 : index")
    e("  %hd64 = index.mul %hd, %c64 : index")
    e("  %pt0 = index.add %tid, %c32 : index")
    e("  %ptid = index.sub %pt0, %hd64 : index")     # partner lane (tid ^ 32)
    e("  %tid2 = index.add %tid, %c256 : index")
    e("  %ptid2 = index.add %ptid, %c256 : index")
    e("  %h0 = index.cmp eq, %half, %c0 : index")
    # opaque 1.0 / 0.0 (lane & ~lane is 0, unprovable to the folder)
    e("  %lanei = index.cast %lane : index to i32")
    e("  %xm1 = scalar.constant -1 : i32")
    e("  %onebits = scalar.constant 1065353216 : i32")
    e("  %lanen = scalar.xori %lanei, %xm1 : i32")
    e("  %lz = scalar.andi %lanei, %lanen : i32")
    e("  %lob = scalar.ori %lz, %onebits : i32")
    e("  %one_o = scalar.bitcast %lob : i32 to f32")
    e("  %zero_o = scalar.bitcast %lz : i32 to f32")
    e("  %qs = index.mul %qb, %c32 : index")
    e("  %kvh = index.div %hp, %c3 : index")
    e("  %pig = index.rem %hp, %c3 : index")
    e("  %kvh6 = index.mul %kvh, %c6 : index")
    e("  %pig2 = index.mul %pig, %c2 : index")
    e("  %head0 = index.add %kvh6, %pig2 : index")
    e("  %kvbase = index.mul %kvh, %c256 : index")
    e("  %vhb0 = index.mul %kvh, %vtiles : index")
    e("  %vhb = index.mul %vhb0, %c4096 : index")
    e("  %vlane = index.mul %tid, %c16 : index")
    e("  %vhl = index.add %vhb, %vlane : index")
    e("  %ctx_end = index.add %start_pos, %B : index")
    e("  %qs32 = index.add %qs, %c32 : index")
    e("  %vis0 = index.add %start_pos, %qs32 : index")
    e("  %max_vis = index.min %ctx_end, %vis0 : index")
    e("  %B_1 = index.sub %B, %c1 : index")
    e("  %cap_1 = index.sub %cache_capacity, %c1 : index")
    # this lane's query: block wqb = (head0 + wqb%2, tokens qs + (wqb/2)*16 + sub)
    e("  %wqh = index.rem %wqb, %c2 : index")
    e("  %wqt0 = index.div %wqb, %c2 : index")
    e("  %wqt1 = index.mul %wqt0, %c16 : index")
    e("  %wqt = index.add %qs, %wqt1 : index")
    e("  %r_lq = index.add %wqt, %sub : index")
    e("  %r_abs = index.add %start_pos, %r_lq : index")
    e("  %r_live = index.cmp ult, %r_lq, %B : index")
    e("  %r_head = index.add %head0, %wqh : index")
    # ---- Q stage (as gen_attn_hip): row r = rb*16 + rr
    e("  %qr = index.div %tid, %c4 : index")
    e("  %qpart = index.rem %tid, %c4 : index")
    e("  %qrb = index.div %qr, %c16 : index")
    e("  %qrr = index.rem %qr, %c16 : index")
    e("  %qrb2 = index.rem %qrb, %c2 : index")
    e("  %qrh = index.add %head0, %qrb2 : index")
    e("  %qrq0 = index.div %qrb, %c2 : index")
    e("  %qrq1 = index.mul %qrq0, %c16 : index")
    e("  %qrq2 = index.add %qs, %qrq1 : index")
    e("  %qlq = index.add %qrq2, %qrr : index")
    e("  %qlive = index.cmp ult, %qlq, %B : index")
    e("  %qlqc = index.min %qlq, %B_1 : index")
    e("  %qrow0 = index.mul %qlqc, %c6144 : index")
    e("  %qhb = index.mul %qrh, %c256 : index")
    e("  %qrow = index.add %qrow0, %qhb : index")
    e("  %qdb = index.mul %qpart, %c64 : index")
    e("  %qsc_v = vector.splat %qscale : " + V4)
    for c in range(8):
        e(f"  %qdc{c} = index.constant {8 * c} : index")
        e(f"  %qd{c} = index.add %qdb, %qdc{c} : index")
        e(f"  %qa{c} = index.add %qrow, %qd{c} : index")
        e(f"  %qa{c}b = index.add %qa{c}, %c4 : index")
        e(f"  %qv{c}a = vector.load %q_flat[%qa{c}] : view<[%qtot]xf32> -> {V4}")
        e(f"  %qv{c}b = vector.load %q_flat[%qa{c}b] : view<[%qtot]xf32> -> {V4}")
        e(f"  %qm{c}a = vector.mulf %qv{c}a, %qsc_v : {V4}")
        e(f"  %qm{c}b = vector.mulf %qv{c}b, %qsc_v : {V4}")
        e(f"  %qh{c}a = vector.fptrunc %qm{c}a : {V4} to vector<4xf16>")
        e(f"  %qh{c}b = vector.fptrunc %qm{c}b : {V4} to vector<4xf16>")
        e(f"  %qh{c} = vector.concat<0> %qh{c}a, %qh{c}b : vector<4xf16>, vector<4xf16> -> {V8H}")
        e(f"  %qz{c} = scf.select %qlive, %qh{c}, %zh8 : {V8H}")
        e(f"  vector.store %qz{c}, %qs_view[%qr, %qd{c}] : {V8H}, view<64x{Q_PITCH}xf16>")
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    # Q^T fragments (B: [dim, query]) of this wave's 128 dims, kept in registers
    e("  %wq16 = index.mul %wqb, %c16 : index")
    for c in range(8):
        e(f"  %qfd{c}c = index.constant {16 * c} : index")
        e(f"  %qfd{c} = index.add %hd128, %qfd{c}c : index")
        e(f"  %qf{c} = vector.fragment.load<rhs> %q_fr[%qfd{c}, %wq16] shape [%k, %n] : view<256x64xf16, %q_lay> -> {V16H}")
    # Loom does not drain these LDS loads before the barrier below (no
    # lgkmcnt(0) ahead of s_barrier), and the prologue then stages K/V over the
    # Q stage: a fast wave overwrote Q rows a slow wave was still reading
    # (a few corrupted query lanes per run, varying). An LDS store of a value
    # built from both halves of every fragment forces the drain first.
    acc = None
    for c in range(8):
        for j in (0, 8):
            e(f"  %qx{c}_{j} = vector.extract %qf{c}[{j}] : {V16H} -> f16")
            if acc is None:
                acc = f"%qx{c}_{j}"
            else:
                e(f"  %qy{c}_{j} = scalar.addf {acc}, %qx{c}_{j} : f16")
                acc = f"%qy{c}_{j}"
    e(f"  %qdrain = vector.splat {acc} : {V8H}")
    e("  %qdr_v = buffer.view %pool[%s_o] : buffer -> view<512x8xf16>")
    e("  %qdr_r = index.add %tid, %c256 : index")
    e(f"  vector.store %qdrain, %qdr_v[%qdr_r, %c0] : {V8H}, view<512x8xf16>")
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")

    # staging maps. K: item j = tid + 256nn, key = j/32, d8 = (j%32)*8 (one key
    # row per wave: coalesced global, contiguous LDS); LDS row 2*(key%8) +
    # key/8. V^T: lane = dim, LDS row (dim/16)*16 + 2*(dim%8) + (dim%16)/8.
    e("  %vdl = index.rem %tid, %c16 : index")
    e("  %vdb = index.sub %tid, %vdl : index")
    e("  %vdl8 = index.rem %vdl, %c8 : index")
    e("  %vdh = index.div %vdl, %c8 : index")
    e("  %vdr0 = index.mul %vdl8, %c2 : index")
    e("  %vdr1 = index.add %vdr0, %vdh : index")
    e("  %vrow = index.add %vdb, %vdr1 : index")

    for nn in range(2):
        e(f"  %ki{nn}c = index.constant {256 * nn} : index")
        e(f"  %ki{nn} = index.add %tid, %ki{nn}c : index")
        e(f"  %kk{nn} = index.div %ki{nn}, %c32 : index")
        e(f"  %kd{nn}a = index.rem %ki{nn}, %c32 : index")
        e(f"  %kd{nn} = index.mul %kd{nn}a, %c8 : index")
        e(f"  %kk{nn}l = index.rem %kk{nn}, %c8 : index")
        e(f"  %kk{nn}h = index.div %kk{nn}, %c8 : index")
        e(f"  %kk{nn}r0 = index.mul %kk{nn}l, %c2 : index")
        e(f"  %kr{nn} = index.add %kk{nn}r0, %kk{nn}h : index")

    def load_k(ks, p, ind):
        names = []
        for nn in range(2):
            e(f"{ind}%{p}kp{nn} = index.add {ks}, %kk{nn} : index")
            e(f"{ind}%{p}kpc{nn} = index.min %{p}kp{nn}, %cap_1 : index")
            e(f"{ind}%{p}kr{nn} = index.mul %{p}kpc{nn}, %c1024 : index")
            e(f"{ind}%{p}kr{nn}b = index.add %{p}kr{nn}, %kvbase : index")
            e(f"{ind}%{p}ka{nn} = index.add %{p}kr{nn}b, %kd{nn} : index")
            e(f"{ind}%{p}kv{nn} = vector.load %k_flat[%{p}ka{nn}] : view<[%kvtot]xf16> -> {V8H}")
            names.append(f"%{p}kv{nn}")
        return names

    def load_v(ks, p, ind):
        e(f"{ind}%{p}vks = index.min {ks}, %vlast : index")
        e(f"{ind}%{p}vtb = index.mul %{p}vks, %c256 : index")
        e(f"{ind}%{p}va0 = index.add %vhl, %{p}vtb : index")
        e(f"{ind}%{p}va1 = index.add %{p}va0, %c8 : index")
        return [e(f"{ind}%{p}vv{nn} = vector.load %v_flat[%{p}va{nn}] : view<[%vtot]xf16> -> {V8H}") or f"%{p}vv{nn}" for nn in range(2)]

    def stage_k(ks, cur, p, ind):
        """K tile at key ks (rows permuted, zero past ctx_end) to LDS."""
        for nn in range(2):
            e(f"{ind}%{p}sp{nn} = index.add {ks}, %kk{nn} : index")
            e(f"{ind}%{p}sl{nn} = index.cmp ult, %{p}sp{nn}, %ctx_end : index")
            e(f"{ind}%{p}sv{nn} = scf.select %{p}sl{nn}, {cur[nn]}, %zh8 : {V8H}")
            e(f"{ind}vector.store %{p}sv{nn}, %k_view[%kr{nn}, %kd{nn}] : {V8H}, view<16x{KT_PITCH}xf16>")

    def stage_v(cur, vb, p, ind):
        e(f"{ind}%{p}vr = index.add {vb}, %vrow : index")
        e(f"{ind}vector.store {cur[0]}, %v_view[%{p}vr, %c0] : {V8H}, view<512x{VT_PITCH}xf16>")
        e(f"{ind}vector.store {cur[1]}, %v_view[%{p}vr, %c8] : {V8H}, view<512x{VT_PITCH}xf16>")

    # prologue: K(0), V(0) staged (V buffer 0)
    k0 = load_k("%c0", "p0", "  ")
    v0 = load_v("%c0", "p2", "  ")
    stage_k("%c0", k0, "p0s", "  ")
    stage_v(v0, "%c0", "p0v", "  ")
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")

    onames = [f"%o{f}" for f in range(8)]
    kvn = ["%kvc0", "%kvc1", "%kvc2", "%kvc3"]
    types = ", ".join([V8] * 8 + ["f32", "f32"])

    def body(tail):
        e("    %t16 = index.div %ks_, %c16 : index")
        e("    %par = index.rem %t16, %c2 : index")
        e("    %npar = index.sub %c1, %par : index")
        e("    %vcur = index.mul %par, %c256 : index")
        e("    %vnxt = index.mul %npar, %c256 : index")
        e("    %ks16 = index.add %ks_, %c16 : index")
        nk = load_k("%ks16", "nk", "    ")
        nv = load_v("%ks16", "nv", "    ")
        # ---- A: S^T partial over this wave's 128 dims
        # QK2: two independent accumulator chains (dims c even / odd), summed:
        # an 8-deep dependent WMMA chain waits on each MMA's latency
        e(f"    %zeros8s = vector.fragment<init> %zeros8 shape [%m, %n] : {V8}")
        accs = ["%zeros8s", "%zeros8s"]
        for c in range(8):
            e(f"    %kfc{c}c = index.constant {16 * c} : index")
            e(f"    %kfc{c} = index.add %hd128, %kfc{c}c : index")
            e(f"    %kf{c} = vector.fragment.load<lhs> %k_view[%c0, %kfc{c}] shape [%m, %k] : view<16x{KT_PITCH}xf16> -> {V16H}")
            j = c % 2 if QK2 else 0
            e(f"    %sa{c} = vector.mma %kf{c}, %qf{c}, {accs[j]} : {V16H}, {V16H}, {V8}")
            accs[j] = f"%sa{c}"
            if QKF and c + 1 < 8 and (c + 1) % QKF == 0:
                e("    scf.schedule.fence")
        if QK2:
            e(f"    %sacc = vector.addf {accs[0]}, {accs[1]} : {V8}")
            acc = "%sacc"
        else:
            acc = accs[0]
        if S8:
            e(f"    vector.store {acc}, %s8_view[%tid, %c0] : {V8}, view<256x8xf32>")
        else:
            e(f"    %sst0 = vector.slice {acc}[0] : {V8} -> {V4}")
            e(f"    %sst1 = vector.slice {acc}[4] : {V8} -> {V4}")
            e(f"    vector.store %sst0, %s_view[%tid, %c0] : {V4}, view<512x4xf32>")
            e(f"    vector.store %sst1, %s_view[%tid2, %c0] : {V4}, view<512x4xf32>")
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        # ---- B: K(ks+16) (QK of this tile is done) and V(ks+16) to LDS
        stage_k("%ks16", nk, "st", "    ")
        stage_v(nv, "%vnxt", "stv", "    ")
        if os.environ.get("YAH_ATTN_FA_DBGBAR") == "1":
            e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        # ---- B: softmax
        if S8:
            e(f"    %spart = vector.load %s8_view[%ptid, %c0] : view<256x8xf32> -> {V8}")
        else:
            e(f"    %spa = vector.load %s_view[%ptid, %c0] : view<512x4xf32> -> {V4}")
            e(f"    %spb = vector.load %s_view[%ptid2, %c0] : view<512x4xf32> -> {V4}")
            e(f"    %spart = vector.concat<0> %spa, %spb : {V4}, {V4} -> {V8}")
        e(f"    %s0 = vector.addf {acc}, %spart : {V8}")
        s = "%s0"
        if tail and MSKIF:
            # tile needs the mask iff a key passes the workgroup's first query
            # or the context end (uniform: all 64 rows share qs)
            e("    %mk15 = index.add %ks_, %c15 : index")
            e("    %mneed0 = index.cmp ugt, %mk15, %spq : index")
            e("    %mneed1 = index.cmp uge, %mk15, %ctx_end : index")
            e("    %mneed = scalar.ori %mneed0, %mneed1 : i1")
            e(f"    %smk = scf.if %mneed -> ({V8}) {{")
        if tail:
            # key 8*half + i visible iff < lim = min(r_abs + 1, ctx_end) - ks - 8*half
            e("    %lim0 = index.add %r_abs, %c1 : index")
            e("    %lim1 = index.min %lim0, %ctx_end : index")
            e("    %h8 = index.mul %half, %c8 : index")
            e("    %kh = index.add %ks_, %h8 : index")
            e("    %limi = index.cast %lim1 : index to i32")
            e("    %khi = index.cast %kh : index to i32")
            e("    %lim = scalar.subi %limi, %khi : i32")
            els = []
            for i in range(8):
                e(f"    %mi{i} = scalar.constant {i} : i32")
                e(f"    %mv{i}a = scalar.cmpi slt, %mi{i}, %lim : i32")
                e(f"    %mv{i} = scalar.andi %mv{i}a, %r_live_i1 : i1")
                e(f"    %se{i} = vector.extract %s0[{i}] : {V8} -> f32")
                e(f"    %sm{i} = scf.select %mv{i}, %se{i}, %ninf : f32")
                els.append(f"%sm{i}")
            e(f"    %smask = vector.from_elements {', '.join(els)} : {V8}")
            s = "%smask"
            if MSKIF:
                e(f"      scf.yield %smask : {V8}")
                e("    } else {")
                e(f"      scf.yield %s0 : {V8}")
                e("    }")
                s = "%smk"
        e(f"    %tmax0 = vector.reduce<maxnumf> {s}, %ninf : {V8}, f32")
        e("    %tmi = scalar.bitcast %tmax0 : f32 to i32")
        e("    %tmx, %tmv = kernel.subgroup.shuffle<xor> %tmi, %x16, %x32 : i32, i32, i32")
        e("    %tmf = scalar.bitcast %tmx : i32 to f32")
        e("    %tmax = scalar.maxnumf %tmax0, %tmf : f32")
        if SKIP:
            # one wave-uniform branch: raise the max and rescale O / l only
            # when some row needs it (FA4 conditional rescale)
            if THR > 0:
                e(f"    %thr = scalar.constant {THR / 1.4426950408889634!r} : f32")
                e("    %rthr = scalar.addf %rmax, %thr : f32")
                e("    %grow = scalar.cmpf ogt, %tmax, %rthr : f32")
            else:
                e("    %grow = scalar.cmpf ogt, %tmax, %rmax : f32")
            e("    %anygrow = kernel.subgroup.vote.any %grow : i1")
            otypes = ", ".join([V8] * 8 + ["f32", "f32"])
            e(f"    %rso0, %rso1, %rso2, %rso3, %rso4, %rso5, %rso6, %rso7, %rsm, %rss = scf.if %anygrow -> ({otypes}) {{")
            e("      %gmx = scalar.maxnumf %rmax, %tmax : f32")
            e("      %gpd = scalar.subf %rmax, %gmx : f32")
            e("      %gpdl = scalar.mulf %gpd, %log2e : f32")
            e("      %galpha = scalar.exp2f<afn> %gpdl : f32")
            e(f"      %galpha8 = vector.splat %galpha : {V8}")
            for f in range(8):
                e(f"      %gos{f} = vector.mulf %o{f}, %galpha8 : {V8}")
            e("      %gsum = scalar.mulf %rsum, %galpha : f32")
            e(f"      scf.yield {', '.join(f'%gos{f}' for f in range(8))}, %gmx, %gsum : {otypes}")
            e("    } else {")
            e(f"      scf.yield {', '.join(f'%o{f}' for f in range(8))}, %rmax, %rsum : {otypes}")
            e("    }")
            e("    %nmax = scalar.maxnumf %rsm, %ninf : f32")
        else:
            e("    %nmax = scalar.maxnumf %rmax, %tmax : f32")
        e("    %nml = scalar.mulf %nmax, %log2e : f32")
        e("    %nnml = scalar.negf %nml : f32")
        e("    %pd = scalar.subf %rmax, %nmax : f32")
        e("    %pdl = scalar.mulf %pd, %log2e : f32")
        e("    %alpha = scalar.exp2f<afn> %pdl : f32")
        e(f"    %nnml8 = vector.splat %nnml : {V8}")
        if HIPNUM:
            e(f"    %nmax8 = vector.splat %nmax : {V8}")
            e(f"    %pd8 = vector.subf {s}, %nmax8 : {V8}")
            e(f"    %pl = vector.mulf %pd8, %log2e8 : {V8}")
        else:
            e(f"    %pl = vector.fmaf {s}, %log2e8, %nnml8 : {V8}")
        e(f"    %p = vector.exp2f<afn> %pl : {V8}")
        if HIPNUM and not SKIP:
            # keys 8h..8h+3 and 8h+4..8h+7: HIP's 4-lane segments
            for g in range(2):
                cur = "%zero"
                for i in range(4):
                    e(f"    %pg{g}_{i} = vector.extract %p[{4 * g + i}] : {V8} -> f32")
                    e(f"    %pa{g}_{i} = scalar.addf {cur}, %pg{g}_{i} : f32")
                    cur = f"%pa{g}_{i}"
            e("    %pab = scalar.addf %pa0_3, %pa1_3 : f32")
            e("    %pabi = scalar.bitcast %pab : f32 to i32")
            e("    %pcdi, %pcdv = kernel.subgroup.shuffle<xor> %pabi, %x16, %x32 : i32, i32, i32")
            e("    %pcd = scalar.bitcast %pcdi : i32 to f32")
            e("    %psum = scalar.addf %pab, %pcd : f32")
        else:
            e(f"    %psum = vector.reduce<addf> %p, %zero : {V8}, f32")
        if SKIP:
            e("    %nsum = scalar.addf %rss, %psum : f32")
        else:
            e("    %nsum = scalar.fmaf %rsum, %alpha, %psum : f32")
        # P^T as the B operand: keys 0..7 from the h=0 lane, 8..15 from h=1
        # f16(p) through v_fma_mix (fptrunc(fma(p, 1, 0)) with an opaque 1 and 0):
        # v_cvt_f16_f32 results must sit in v0..v127, where Q^T and O live, and
        # each conversion evicted a Q^T fragment to scratch (as Q4FMIX in the GEMMs)
        for i in range(8):
            e(f"    %pe{i} = vector.extract %p[{i}] : {V8} -> f32")
            e(f"    %pm{i} = scalar.fmaf %pe{i}, %one_o, %zero_o : f32")
            e(f"    %pt{i} = scalar.fptrunc %pm{i} : f32 to f16")
        e(f"    %ph = vector.from_elements {', '.join(f'%pt{i}' for i in range(8))} : {V8H}")
        e(f"    %phi = vector.bitcast %ph : {V8H} to {V4I}")
        e(f"    %ppi, %ppv = kernel.subgroup.shuffle<xor> %phi, %x16, %x32 : {V4I}, i32, i32")
        e(f"    %pph = vector.bitcast %ppi : {V4I} to {V8H}")
        e(f"    %plo = scf.select %h0, %ph, %pph : {V8H}")
        e(f"    %phi2 = scf.select %h0, %pph, %ph : {V8H}")
        e(f"    %pb0 = vector.concat<0> %plo, %phi2 : {V8H}, {V8H} -> {V16H}")
        e(f"    %pb = vector.fragment<rhs> %pb0 shape [%k, %n] : {V16H}")
        # rescale + P.V
        e(f"    %alpha8 = vector.splat %alpha : {V8}")
        outs = []
        for f in range(8):
            e(f"    %vfr{f}c = index.constant {16 * f} : index")
            e(f"    %vfr{f}a = index.add %hd128, %vfr{f}c : index")
            e(f"    %vfr{f} = index.add %vcur, %vfr{f}a : index")
            e(f"    %vf{f} = vector.fragment.load<lhs> %v_view[%vfr{f}, %c0] shape [%m, %k] : view<512x{VT_PITCH}xf16> -> {V16H}")
            if SKIP:
                e(f"    %nx{f} = vector.mma %vf{f}, %pb, %rso{f} : {V16H}, {V16H}, {V8}")
            elif PVZ:
                e(f"    %pz{f} = vector.mma %vf{f}, %pb, %zeros8s : {V16H}, {V16H}, {V8}")
                e(f"    %nx{f} = vector.fmaf %o{f}, %alpha8, %pz{f} : {V8}")
            else:
                e(f"    %os{f} = vector.mulf %o{f}, %alpha8 : {V8}")
            if not PVZ and not SKIP:
                e(f"    %nx{f} = vector.mma %vf{f}, %pb, %os{f} : {V16H}, {V16H}, {V8}")
            if PVF and f + 1 < 8 and (f + 1) % PVF == 0:
                e("    scf.schedule.fence")
            outs.append(f"%nx{f}")
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        return outs, "%nmax", "%nsum", []

    e("  %r_live_i1 = index.cmp ult, %r_lq, %B : index")
    e(f"  %oinit = vector.fragment<init> %zeros8 shape [%m, %n] : {V8}")
    e("  %spq = index.add %start_pos, %qs : index")
    e("  %spq1 = index.add %spq, %c1 : index")
    e("  %split0 = index.div %spq1, %c16 : index")
    e("  %split1 = index.mul %split0, %c16 : index")
    # One masked loop over every tile (YAH_ATTN_FA_PLAIN=1: an unmasked loop
    # up to the diagonal, then the masked one). The two-loop form miscompiled
    # without the unroll policy (O accumulators corrupt across the hand-off
    # between the loops: NaN in ~30% of dims from query block 1 on), so it is
    # diagnosis only.
    if os.environ.get("YAH_ATTN_FA_PLAIN", "0") != "1":
        e("  %split = index.min %c0, %max_vis : index")
    else:
        e("  %split = index.min %split1, %max_vis : index")

    def loop(lo, hi, init_o, init_m, init_s, init_kv, tail, res):
        init = ", ".join(f"{nm} = {v} : {V8}" for nm, v in zip(onames, init_o))
        init += f", %rmax = {init_m} : f32, %rsum = {init_s} : f32"
        e(f"  {', '.join(res)} = scf.for %ks_ = [{lo} to {hi} step %c16]({init}) -> ({types}) {POL} {{")
        outs, nm, ns, nxt = body(tail)
        e(f"    scf.yield {', '.join(outs)}, {nm}, {ns} : {types}")
        e("  }")

    r1 = [f"%of{i}" for i in range(8)] + ["%fmax", "%fsum"]
    loop("%c0", "%split", ["%oinit"] * 8, "%ninf", "%zero", [], os.environ.get("YAH_ATTN_FA_MASKALL") == "1", r1)
    r2 = [f"%og{i}" for i in range(8)] + ["%gmax", "%gsum"]
    loop("%split", "%max_vis", r1[:8], "%fmax", "%fsum", [], True, r2)

    # ---- epilogue: o / sum * sigmoid(gate); lane writes dims 8h..8h+7 of each 16
    if HIPNUM and not SKIP:
        e("  %lsum = scalar.addf %gsum, %zero : f32")   # already the full row sum
    else:
        e("  %lsi = scalar.bitcast %gsum : f32 to i32")
        e("  %lsx, %lsv = kernel.subgroup.shuffle<xor> %lsi, %x16, %x32 : i32, i32, i32")
        e("  %lsf = scalar.bitcast %lsx : i32 to f32")
        e("  %lsum = scalar.addf %gsum, %lsf : f32")
    e("  %lpos = scalar.cmpf ogt, %lsum, %zero : f32")
    e("  %linv0 = scalar.divf %one, %lsum : f32")
    e("  %linv = scf.select %lpos, %linv0, %zero : f32")
    e(f"  %linv8 = vector.splat %linv : {V8}")
    e(f"  %lsum8 = vector.splat %lsum : {V8}")
    e(f"  %nlog2e8 = vector.splat %log2e : {V8}")
    e("  %elc = index.min %r_lq, %B_1 : index")
    e("  %eoff0 = index.mul %elc, %c6144 : index")
    e("  %ehb = index.mul %r_head, %c256 : index")
    e("  %eoff1 = index.add %eoff0, %ehb : index")
    e("  %eh8 = index.mul %half, %c8 : index")
    e("  %eoff2 = index.add %eoff1, %hd128 : index")
    e("  %eoff = index.add %eoff2, %eh8 : index")
    for f in range(8):
        e(f"  %eo{f}c = index.constant {16 * f} : index")
        e(f"  %eo{f} = index.add %eoff, %eo{f}c : index")
        e(f"  %eo{f}b = index.add %eo{f}, %c4 : index")
        e(f"  %eg{f}a = vector.load %g_flat[%eo{f}] : view<[%qtot]xf32> -> {V4}")
        e(f"  %eg{f}b = vector.load %g_flat[%eo{f}b] : view<[%qtot]xf32> -> {V4}")
    for f in range(8):
        e(f"  %eg{f} = vector.concat<0> %eg{f}a, %eg{f}b : {V4}, {V4} -> {V8}")
        e(f"  %egn{f} = vector.negf %eg{f} : {V8}")
        e(f"  %egm{f} = vector.mulf %egn{f}, %nlog2e8 : {V8}")
        e(f"  %egx{f} = vector.exp2f<afn> %egm{f} : {V8}")
        e(f"  %egd{f} = vector.addf %ones8, %egx{f} : {V8}")
        e(f"  %egr{f} = vector.divf %ones8, %egd{f} : {V8}")
        if HIPNUM:
            e(f"  %eod{f} = vector.divf %og{f}, %lsum8 : {V8}")
            e(f"  %eov{f} = scf.select %lpos, %eod{f}, %zeros8 : {V8}")
        else:
            e(f"  %eov{f} = vector.mulf %og{f}, %linv8 : {V8}")
        e(f"  %eout{f} = vector.mulf %eov{f}, %egr{f} : {V8}")
        e(f"  scf.if %r_live {{")
        if F16OUT:
            e(f"    %eoh{f} = vector.fptrunc %eout{f} : {V8} to {V8H}")
            e(f"    vector.store %eoh{f}, %o_flat[%eo{f}] : {V8H}, view<[%qtot]xf16>")
        else:
            e(f"    vector.store %eout{f}, %o_flat[%eo{f}] : {V8}, view<[%qtot]xf32>")
        e("  }")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "yah_attn_fa.loom"
    open(out, "w").write(gen())
    print(out)


if __name__ == "__main__":
    main()
