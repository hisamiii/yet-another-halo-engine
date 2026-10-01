#!/usr/bin/env python3
"""Generate the causal prefill attention in HIP's arithmetic order.

usage: gen_attn_hip.py [out.loom]

A port of WmmaCausalAttention<32, 16, true> (attention_wmma.hip), the kernel
the HIP engine runs at pp2048, meant to be bit-identical to it given the same
q, gate and f16 K/V: tools/attn_vs_hip.sh runs both and compares with atol 0.
The op sequence is taken from its compiled ISA (hipcc -O3):

  block = 32 query tokens x 2 query heads of one GQA group = 64 rows, 256 wave32
  lanes, grid (tokens/32, num_heads/2); key tiles of 16.
  q      = f16(q * 0.0625), once
  S      = two 8-step WMMA chains (Q as A, K as B) over the two halves of the
           head dim; the softmax adds the halves: s = s0 + s1
  softmax, 4 lanes per row, 4 keys each, running max/sum per row:
           next = max(prev, max over the row); prior = isfinite(prev) ?
           exp(prev - next) : 0; w = isfinite(s) ? exp(s - next) : 0;
           part = (((0 + w0) + w1) + w2) + w3, xor-1 and xor-2 butterfly;
           sum = fma(sum, prior, part); p = f16(w)
           exp(x) = exp2(x * 0x3fb8aa3b)          (hipcc's __expf)
  P.V    = o = o * prior (per row); o = WMMA(P as A, V as B, o) -- except on
           the causal boundary tiles, where a row that sees only some of the
           tile's keys gets o = fmaf(f32(p), f32(v), o) over its visible keys
           in key order (HIP's tail path), from the same scaled o
  out    = (sum > 0 ? o / sum : 0) * (1 / (1 + exp(-gate)))   IEEE divisions

Loom reads K from its own token-major cache and V from a V^T copy that
gen_vtrans() (yah_transpose_v16, ~0.06 ms per layer) writes per layer as
[kv head][16-key tile][dim][16], one contiguous 8 KB block per tile, so a lane
stages its dim's 16 keys with two 16-byte stores (HIP's PackAttentionHeads
plays the same role). The next tile's K/V global loads are carried across the
loop. The boundary-tile fmaf chain runs only for row blocks whose tile is
partly visible (wave-uniform), from LDS copies of the accumulator so the
fragment register layout never matters; its per-key condition is dropped
because the softmax already stored p = 0 for every key HIP skips and
fmaf(0, v, o) == o (up to the sign of a zero o, which the atol-0 gate treats
as equal). The epilogue issues each fragment's gate loads together.

pp2048, per layer: 2.67 ms + 0.01 transpose (standalone), 3.07 + 0.06 in the
pipeline (YAH_LOOM_TIME=2) vs HIP 2.86; gen_attn_heads.py H=3 was 4.56.

Debug knobs: YAH_ATTN_DBG=sum|o|rsc|tid stores an intermediate instead of the
output; YAH_ATTN_NOTAIL=1 runs every tile in the plain form (not HIP-exact on
boundary tiles). Two Loom pitfalls met on the way: a lane-divergent scf.if
holding LDS loads inside the loop lost lanes (0.3% of outputs written), and
the fully unrolled 8 x 8 x 16 boundary chains exhausted the SGPRs.
"""
import os
import sys

# V^T input: value_cache holds [kv_head][key tile][256 dims][16 keys] f16 over
# ceil(cache_capacity/16) tiles, tokens past the context zero (gen_vtrans()
# writes it from the token-major cache). A tile is one contiguous 8 KB block,
# one 32-byte row per lane, instead of 16 scalar transposing LDS stores per
# lane. (Plain [dim][token] rows at a 4 KB pitch were slower than the scalar
# stores: 256 rows per tile, cache-aliased.) YAH_ATTN_VT=0 reads the
# token-major cache.
VT = os.environ.get("YAH_ATTN_VT", "1") != "0"

V4 = "vector<4xf32>"
V8 = "vector<8xf32>"
V8H = "vector<8xf16>"
V16H = "vector<16xf16>"

# LDS pool offsets (bytes). The Q stage (prologue only) overlaps the rest.
KT_OFF, KT_PITCH = 0, 264          # K tile: 16 keys x 256 dims (+8 pad)
VT_OFF, VT_PITCH = 8448, 24        # V^T: 256 dims x 16 keys (+8 pad)
S_OFF = 20736                      # scores: 2 halves x 4 row blocks x 16 x 17 f32
P_OFF, P_PITCH = 29440, 24         # P: 64 rows x 16 keys (+8) f16
RS_OFF = 32512                     # per-row scale / final sum, 64 x 16 f32 (replicated)
TL_OFF = 36608                     # boundary-tile scratch: 8 waves x 2 x 16x16 f32
POOL = TL_OFF + 8 * 2 * 1024
# RS2: the per-tile row scales also kept compact, 64 f32 indexed [rb][half][i]
# = scale of row rb*16 + 2i + half, so a lane reads its result fragment's 8
# row scales with one vector<8xf32> load (two ds_load_b128) instead of 8
# ds_load_b32 from the replicated table. Same values: bit-identical.
RS2 = os.environ.get("YAH_ATTN_RS2", "1") == "1"
RS2_OFF = POOL
if RS2:
    POOL += 64 * 4
# LDS2: V^T first, then K next to the scores, so the boundary-tile scratch
# aliases K + S: both are dead during P.V (QK and the softmax finished before
# the barrier ahead of P.V; the next tile's K store and QK follow the loop-top
# barrier). 52992 -> 36608 B (+256 with RS2): three workgroups per WGP instead
# of two (HIP's 20992 B gets three as well, VGPR-bound).
LDS2 = os.environ.get("YAH_ATTN_LDS2", "1") == "1"
# PVFENCE: a schedule fence after each P.V MMA, so each accumulator's rescale
# stays next to its own MMA (probe for the back-edge accumulator copies)
PVFENCE = os.environ.get("YAH_ATTN_PVFENCE", "0") == "1"
if LDS2:
    VT_OFF = 0
    KT_OFF = VT_OFF + 256 * VT_PITCH * 2          # 12288
    assert KT_OFF + 16 * KT_PITCH * 2 == S_OFF    # K tile ends where the scores start
    TL_OFF = KT_OFF                               # 16 KB over K (8448) + S (8704)
    assert TL_OFF + 8 * 2 * 1024 <= P_OFF
    RS2_OFF = RS_OFF + 64 * 16 * 4                # 36608
    POOL = RS2_OFF + (64 * 4 if RS2 else 0)
Q_PITCH = 264                      # Q stage: 64 rows x 256 dims (+8 pad)
assert 64 * Q_PITCH * 2 <= POOL and POOL <= 65536


# Largest prompt the token-count facts admit (range assumptions the uniformity
# and bounds proofs build on). emit_prefill_pp.py sets it to B when B > 2048.
MAX_TOKENS = int(os.environ.get("YAH_ATTN_MAX_TOKENS", "2048"))


def gen():
    L = []
    e = L.append
    e("// GENERATED by tools/gen_attn_hip.py -- edit the generator.")
    e("// Causal prefill attention in HIP's WmmaCausalAttention<32, 16, true> order.")
    e("amdgpu.target<gfx1151> @attn_hip_w32 {subgroup_size = 32}")
    e("")
    e("config.def @attention_prefill.cache_capacity = 2048 : index")
    e(f"config.decl @attention_prefill.token_count : %value: index where [range(%value, 1, {MAX_TOKENS})]")
    e("config.decl @attention_prefill.num_heads : %value: index where [range(%value, 1, 4096)]")
    e("config.decl @attention_prefill.num_kv_heads : %value: index where [range(%value, 1, 4096)]")
    e("config.decl @attention_prefill.gqa : %value: index where [range(%value, 1, 4096)]")
    e("config.decl @attention_prefill.head_dim : %value: index where [range(%value, 1, 1024)]")
    e("config.decl @attention_prefill.start_pos : %value: index where [range(%value, 0, 1073741824)]")
    e("")
    e("kernel.def target(@attn_hip_w32) @yah_attn_wmma() {")
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
    for v in (0, 1, 2, 3, 4, 6, 8, 15, 16, 17, 24, 32, 64, 128, 256, 264, 1024, 6144):
        e(f"  %c{v} = index.constant {v} : index")
    e("  %cache_capacity = config.get @attention_prefill.cache_capacity : index")
    e("  %token_count0 = config.get @attention_prefill.token_count : index")
    e(f"  %B = index.assume %token_count0 [range(%token_count0, 1, {MAX_TOKENS})] : index")
    e("  %start_pos = config.get @attention_prefill.start_pos : index")
    e("  %zero = scalar.constant 0.0 : f32")
    e("  %one = scalar.constant 1.0 : f32")
    # +-FLT_MAX stand in for +-inf: a masked score still exps to exactly 0, and
    # every row sees key 0 in its first tile, so the running max is finite from
    # then on -- the same values as HIP's -INFINITY / isfinite. log2e rounds to
    # 0x3fb8aa3b, the constant hipcc's __expf multiplies by.
    e("  %pinf = scalar.constant 3.4028234663852886e+38 : f32")
    e("  %ninf = scalar.constant -3.4028234663852886e+38 : f32")
    e("  %log2e = scalar.constant 1.4426950408889634 : f32")
    e("  %qscale = scalar.constant 0.0625 : f32")
    e("  %zh8 = vector.constant 0.0 : vector<8xf16>")
    e(f"  %zeros8 = vector.constant 0.0 : {V8}")
    e("  %m = index.constant 16 : index")
    e("  %n = index.constant 16 : index")
    e("  %k = index.constant 16 : index")
    for v in (1, 2):
        e(f"  %x{v} = scalar.constant {v} : i32")
    e("  %x32 = scalar.constant 32 : i32")
    e("  %qtot = index.mul %B, %c6144 : index")
    e("  %kvtot = index.mul %cache_capacity, %c1024 : index")
    e("  %q_na, %g_na, %k_na, %v_na, %o_na = buffer.assume.noalias %query, %gate, %key_cache, %value_cache, %output : buffer, buffer, buffer, buffer, buffer")
    e("  %q_flat = buffer.view %q_na[%base] : buffer -> view<[%qtot]xf32>")
    e("  %g_flat = buffer.view %g_na[%base] : buffer -> view<[%qtot]xf32>")
    e("  %o_flat = buffer.view %o_na[%base] : buffer -> view<[%qtot]xf32>")
    e("  %k_flat = buffer.view %k_na[%base] : buffer -> view<[%kvtot]xf16>")
    if VT:
        e("  %cap15 = index.add %cache_capacity, %c15 : index")
        e("  %vtiles = index.div %cap15, %c16 : index")
        e("  %vpitch = index.mul %vtiles, %c16 : index")
        e("  %vtot = index.mul %vpitch, %c1024 : index")
        e("  %vlast = index.sub %vpitch, %c16 : index")
        e("  %v_flat = buffer.view %v_na[%base] : buffer -> view<[%vtot]xf16>")
    else:
        e("  %v_flat = buffer.view %v_na[%base] : buffer -> view<[%kvtot]xf16>")
    e(f"  %pool_bytes = index.constant {POOL} : offset")
    e("  %pool = buffer.alloca<workgroup> align(16) %pool_bytes : buffer")

    def view(name, off, ty, lay=None):
        e(f"  %{name}_o = index.constant {off} : offset")
        e(f"  %{name} = buffer.view %pool[%{name}_o] : buffer -> {ty}" + (f"" if lay is None else ""))

    e(f"  %qs_view = buffer.view %pool[%base] : buffer -> view<64x{Q_PITCH}xf16>")
    e(f"  %kt_o = index.constant {KT_OFF} : offset")
    e(f"  %kt_st = buffer.view %pool[%kt_o] : buffer -> view<16x{KT_PITCH}xf16>")
    e(f"  %kt_lay = encoding.layout.strided [1, {KT_PITCH}] : encoding<layout>")
    e("  %kt_fr = buffer.view %pool[%kt_o] : buffer -> view<256x16xf16, %kt_lay>")
    e(f"  %vt_o = index.constant {VT_OFF} : offset")
    e(f"  %vt_st = buffer.view %pool[%vt_o] : buffer -> view<256x{VT_PITCH}xf16>")
    e(f"  %vt_lay = encoding.layout.strided [1, {VT_PITCH}] : encoding<layout>")
    e("  %vt_fr = buffer.view %pool[%vt_o] : buffer -> view<16x256xf16, %vt_lay>")
    e(f"  %p_o = index.constant {P_OFF} : offset")
    e(f"  %p_view = buffer.view %pool[%p_o] : buffer -> view<64x{P_PITCH}xf16>")
    e(f"  %rs_o = index.constant {RS_OFF} : offset")
    e("  %rs_view = buffer.view %pool[%rs_o] : buffer -> view<64x16xf32>")
    if RS2:
        e(f"  %rs2_o = index.constant {RS2_OFF} : offset")
        e("  %rs2_view = buffer.view %pool[%rs2_o] : buffer -> view<64xf32>")
    for kh in range(2):
        for rb in range(4):
            off = S_OFF + (kh * 4 + rb) * 16 * 17 * 4
            e(f"  %s{kh}{rb}_o = index.constant {off} : offset")
            e(f"  %s{kh}{rb} = buffer.view %pool[%s{kh}{rb}_o] : buffer -> view<16x17xf32>")
    e(f"  %s_flat_o = index.constant {S_OFF} : offset")
    e("  %s_flat = buffer.view %pool[%s_flat_o] : buffer -> view<2176xf32>")
    # ids
    e("  %qb = kernel.workgroup.id<x> : index")
    e("  %hp = kernel.workgroup.id<y> : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %wave = index.div %tid, %c32 : index")
    e("  %lane = index.rem %tid, %c32 : index")
    e("  %sub = index.rem %lane, %c16 : index")
    e("  %half = index.div %lane, %c16 : index")
    e("  %qs = index.mul %qb, %c32 : index")
    e("  %kvh = index.div %hp, %c3 : index")
    e("  %pig = index.rem %hp, %c3 : index")
    e("  %kvh6 = index.mul %kvh, %c6 : index")
    e("  %pig2 = index.mul %pig, %c2 : index")
    e("  %head0 = index.add %kvh6, %pig2 : index")
    e("  %kvbase = index.mul %kvh, %c256 : index")
    if VT:
        e("  %c4096 = index.constant 4096 : index")
        e("  %vhb0 = index.mul %kvh, %vtiles : index")
        e("  %vhb = index.mul %vhb0, %c4096 : index")
        e("  %vlane = index.mul %tid, %c16 : index")
        e("  %vhl = index.add %vhb, %vlane : index")
    e("  %ctx_end = index.add %start_pos, %B : index")
    e("  %qs32 = index.add %qs, %c32 : index")
    e("  %vis0 = index.add %start_pos, %qs32 : index")
    e("  %max_vis = index.min %ctx_end, %vis0 : index")
    e("  %s_rb = index.rem %wave, %c4 : index")
    e("  %s_kh = index.div %wave, %c4 : index")
    e("  %B_1 = index.sub %B, %c1 : index")
    e("  %cap_1 = index.sub %cache_capacity, %c1 : index")
    # ---- Q stage: row r = rb*16 + rr (head0 + rb%2, token qs + (rb/2)*16 + rr)
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
        e(f"  %qc{c}4 = index.constant 4 : index")
        e(f"  %qa{c}b = index.add %qa{c}, %qc{c}4 : index")
        e(f"  %qv{c}a = vector.load %q_flat[%qa{c}] : view<[%qtot]xf32> -> {V4}")
        e(f"  %qv{c}b = vector.load %q_flat[%qa{c}b] : view<[%qtot]xf32> -> {V4}")
        e(f"  %qm{c}a = vector.mulf %qv{c}a, %qsc_v : {V4}")
        e(f"  %qm{c}b = vector.mulf %qv{c}b, %qsc_v : {V4}")
        e(f"  %qh{c}a = vector.fptrunc %qm{c}a : {V4} to vector<4xf16>")
        e(f"  %qh{c}b = vector.fptrunc %qm{c}b : {V4} to vector<4xf16>")
        els = ", ".join(f"%qe{c}_{j}" for j in range(8))
        for j in range(4):
            e(f"  %qe{c}_{j} = vector.extract %qh{c}a[{j}] : vector<4xf16> -> f16")
            e(f"  %qe{c}_{j + 4} = vector.extract %qh{c}b[{j}] : vector<4xf16> -> f16")
        e(f"  %qh{c} = vector.from_elements {els} : {V8H}")
        e(f"  %qz{c} = scf.if %qlive -> ({V8H}) {{")
        e(f"    scf.yield %qh{c} : {V8H}")
        e("  } else {")
        e(f"    scf.yield %zh8 : {V8H}")
        e("  }")
        e(f"  vector.store %qz{c}, %qs_view[%qr, %qd{c}] : {V8H}, view<64x{Q_PITCH}xf16>")
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e("  %s_rb16 = index.mul %s_rb, %c16 : index")
    e("  %s_kh8 = index.mul %s_kh, %c8 : index")
    for ks in range(8):
        e(f"  %qk{ks}c = index.constant {ks} : index")
        e(f"  %qk{ks}a = index.add %s_kh8, %qk{ks}c : index")
        e(f"  %qk{ks}d = index.mul %qk{ks}a, %c16 : index")
        e(f"  %qf{ks} = vector.fragment.load<lhs> %qs_view[%s_rb16, %qk{ks}d] shape [%m, %k] : view<64x{Q_PITCH}xf16> -> {V16H}")
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    # softmax lane map
    e("  %rg = index.div %tid, %c4 : index")
    e("  %seg = index.rem %tid, %c4 : index")
    e("  %rrb = index.div %rg, %c16 : index")
    e("  %rrow = index.rem %rg, %c16 : index")
    if RS2:
        e("  %rr2 = index.rem %rrow, %c2 : index")
        e("  %rrh = index.div %rrow, %c2 : index")
        e("  %rrb16 = index.mul %rrb, %c16 : index")
        e("  %rr28 = index.mul %rr2, %c8 : index")
        e("  %rs2a = index.add %rrb16, %rr28 : index")
        e("  %rs2i = index.add %rs2a, %rrh : index")
    e("  %rq0 = index.div %rrb, %c2 : index")
    e("  %rq1 = index.mul %rq0, %c16 : index")
    e("  %rq2 = index.add %qs, %rq1 : index")
    e("  %r_lq = index.add %rq2, %rrow : index")
    e("  %r_abs = index.add %start_pos, %r_lq : index")
    e("  %r_live = index.cmp ult, %r_lq, %B : index")
    e("  %seg4 = index.mul %seg, %c4 : index")
    # staging lane maps: K (key = idx/32, d8 = (idx%32)*8, idx = tid + 256n);
    # V (key = tid%16, dims (tid/16)*8 + 128n)
    e("  %vk = index.rem %tid, %c16 : index")
    e("  %vg = index.div %tid, %c16 : index")
    e("  %vd0 = index.mul %vg, %c8 : index")

    def load_kv(ks, p, ind):
        """Global loads of the 16-key K/V tile at ks (rows clamped): 2 K and 2 V
        vectors per lane. Masking to zero happens at the LDS store."""
        names = []
        for nn in range(2):
            e(f"{ind}%{p}ki{nn}c = index.constant {256 * nn} : index")
            e(f"{ind}%{p}ki{nn} = index.add %tid, %{p}ki{nn}c : index")
            e(f"{ind}%{p}kk{nn} = index.div %{p}ki{nn}, %c32 : index")
            e(f"{ind}%{p}kd{nn}a = index.rem %{p}ki{nn}, %c32 : index")
            e(f"{ind}%{p}kd{nn} = index.mul %{p}kd{nn}a, %c8 : index")
            e(f"{ind}%{p}kp{nn} = index.add {ks}, %{p}kk{nn} : index")
            e(f"{ind}%{p}kpc{nn} = index.min %{p}kp{nn}, %cap_1 : index")
            e(f"{ind}%{p}kr{nn} = index.mul %{p}kpc{nn}, %c1024 : index")
            e(f"{ind}%{p}kr{nn}b = index.add %{p}kr{nn}, %kvbase : index")
            e(f"{ind}%{p}ka{nn} = index.add %{p}kr{nn}b, %{p}kd{nn} : index")
            e(f"{ind}%{p}kv{nn} = vector.load %k_flat[%{p}ka{nn}] : view<[%kvtot]xf16> -> {V8H}")
            names.append(f"%{p}kv{nn}")
        if VT:
            # lane = dim: keys ks..ks+15 of V^T row (kv_head*256 + dim)
            e(f"{ind}%{p}vks = index.min {ks}, %vlast : index")
            e(f"{ind}%{p}vtb = index.mul %{p}vks, %c256 : index")
            e(f"{ind}%{p}va0 = index.add %vhl, %{p}vtb : index")
            e(f"{ind}%{p}va1 = index.add %{p}va0, %c8 : index")
            for nn in range(2):
                e(f"{ind}%{p}vv{nn} = vector.load %v_flat[%{p}va{nn}] : view<[%vtot]xf16> -> {V8H}")
                names.append(f"%{p}vv{nn}")
            return names
        e(f"{ind}%{p}vp = index.add {ks}, %vk : index")
        e(f"{ind}%{p}vpc = index.min %{p}vp, %cap_1 : index")
        e(f"{ind}%{p}vr = index.mul %{p}vpc, %c1024 : index")
        e(f"{ind}%{p}vrb = index.add %{p}vr, %kvbase : index")
        for nn in range(2):
            e(f"{ind}%{p}vd{nn}c = index.constant {128 * nn} : index")
            e(f"{ind}%{p}vd{nn} = index.add %vd0, %{p}vd{nn}c : index")
            e(f"{ind}%{p}va{nn} = index.add %{p}vrb, %{p}vd{nn} : index")
            e(f"{ind}%{p}vv{nn} = vector.load %v_flat[%{p}va{nn}] : view<[%kvtot]xf16> -> {V8H}")
            names.append(f"%{p}vv{nn}")
        return names

    def stage_kv(cur):
        """Store the carried K/V tile at %ks_ into LDS (zero past ctx_end)."""
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        for nn in range(2):
            e(f"    %ki{nn}c = index.constant {256 * nn} : index")
            e(f"    %ki{nn} = index.add %tid, %ki{nn}c : index")
            e(f"    %kk{nn} = index.div %ki{nn}, %c32 : index")
            e(f"    %kd{nn}a = index.rem %ki{nn}, %c32 : index")
            e(f"    %kd{nn} = index.mul %kd{nn}a, %c8 : index")
            e(f"    %kp{nn} = index.add %ks_, %kk{nn} : index")
            e(f"    %kl{nn} = index.cmp ult, %kp{nn}, %ctx_end : index")
            e(f"    %kv{nn} = scf.select %kl{nn}, {cur[nn]}, %zh8 : {V8H}")
            e(f"    vector.store %kv{nn}, %kt_st[%kk{nn}, %kd{nn}] : {V8H}, view<16x{KT_PITCH}xf16>")
        if VT:
            e(f"    vector.store {cur[2]}, %vt_st[%tid, %c0] : {V8H}, view<256x{VT_PITCH}xf16>")
            e(f"    vector.store {cur[3]}, %vt_st[%tid, %c8] : {V8H}, view<256x{VT_PITCH}xf16>")
            e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            e("    %ks_next = index.add %ks_, %c16 : index")
            return load_kv("%ks_next", "n", "    ")
        e("    %vp = index.add %ks_, %vk : index")
        e("    %vl = index.cmp ult, %vp, %ctx_end : index")
        for nn in range(2):
            e(f"    %vd{nn}c = index.constant {128 * nn} : index")
            e(f"    %vd{nn} = index.add %vd0, %vd{nn}c : index")
            e(f"    %vv{nn} = scf.select %vl, {cur[2 + nn]}, %zh8 : {V8H}")
            for j in range(8):
                e(f"    %vx{nn}_{j} = vector.extract %vv{nn}[{j}] : {V8H} -> f16")
                e(f"    %vdj{nn}_{j}c = index.constant {j} : index")
                e(f"    %vdj{nn}_{j} = index.add %vd{nn}, %vdj{nn}_{j}c : index")
                e(f"    view.store %vx{nn}_{j}, %vt_st[%vdj{nn}_{j}, %vk] : f16, view<256x{VT_PITCH}xf16>")
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        # next tile's loads fly during this tile's S, softmax and P.V
        e("    %ks_next = index.add %ks_, %c16 : index")
        return load_kv("%ks_next", "n", "    ")

    def isfinite(x, name):
        e(f"    %{name}a = scalar.cmpf ole, {x}, %pinf : f32")
        e(f"    %{name}b = scalar.cmpf oge, {x}, %ninf : f32")
        e(f"    %{name} = scalar.andi %{name}a, %{name}b : i1")

    def exp_(x, name):
        e(f"    %{name}m = scalar.mulf {x}, %log2e : f32")
        e(f"    %{name} = scalar.exp2f<afn> %{name}m : f32")

    def body(tail):
        """One 16-key tile at %ks_. tail: the causal boundary form."""
        nxt = stage_kv(["%kvc0", "%kvc1", "%kvc2", "%kvc3"])
        # S
        e(f"    %sinit = vector.fragment<init> %zeros8 shape [%m, %n] : {V8}")
        acc = "%sinit"
        for ks in range(8):
            e(f"    %kf{ks} = vector.fragment.load<rhs> %kt_fr[%qk{ks}d, %c0] shape [%k, %n] : view<256x16xf16, %kt_lay> -> {V16H}")
            e(f"    %sa{ks} = vector.mma %qf{ks}, %kf{ks}, {acc} : {V16H}, {V16H}, {V8}")
            acc = f"%sa{ks}"
        # store to s[s_kh][s_rb]: offset S_OFF + (s_kh*4 + s_rb)*272 floats
        e("    %st0 = index.mul %s_kh, %c4 : index")
        e("    %st1 = index.add %st0, %s_rb : index")
        e("    %st2 = index.mul %st1, %c272 : index")
        e("    %st3 = index.mul %st2, %c4 : index")
        e(f"    %st4 = index.constant {S_OFF} : index")
        e("    %st5 = index.add %st3, %st4 : index")
        e("    %st6 = index.cast %st5 : index to offset")
        e("    %s_mine = buffer.view %pool[%st6] : buffer -> view<16x17xf32>")
        e(f"    vector.fragment.store<result> {acc}, %s_mine[%c0, %c0] shape [%m, %n] : {V8}, view<16x17xf32>")
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        # softmax
        e("    %sfb0 = index.mul %rrb, %c272 : index")
        e("    %sfr = index.mul %rrow, %c17 : index")
        e("    %sfb = index.add %sfb0, %sfr : index")
        vals = []
        for mm in range(4):
            e(f"    %col{mm}c = index.constant {mm} : index")
            e(f"    %col{mm} = index.add %seg4, %col{mm}c : index")
            e(f"    %kpos{mm} = index.add %ks_, %col{mm} : index")
            e(f"    %cz{mm} = index.cmp ule, %kpos{mm}, %r_abs : index")
            e(f"    %ce{mm} = index.cmp ult, %kpos{mm}, %ctx_end : index")
            e(f"    %cv{mm}a = scalar.andi %r_live, %cz{mm} : i1")
            e(f"    %cv{mm} = scalar.andi %cv{mm}a, %ce{mm} : i1")
            e(f"    %s0i{mm} = index.add %sfb, %col{mm} : index")
            e(f"    %s1i{mm} = index.add %s0i{mm}, %c1088 : index")
            e(f"    %s0v{mm} = view.load %s_flat[%s0i{mm}] : view<2176xf32> -> f32")
            e(f"    %s1v{mm} = view.load %s_flat[%s1i{mm}] : view<2176xf32> -> f32")
            e(f"    %ssum{mm} = scalar.addf %s0v{mm}, %s1v{mm} : f32")
            e(f"    %val{mm} = scf.if %cv{mm} -> (f32) {{")
            e(f"      scf.yield %ssum{mm} : f32")
            e("    } else {")
            e("      scf.yield %ninf : f32")
            e("    }")
            vals.append(f"%val{mm}")
        e(f"    %pm0 = scalar.maxnumf %ninf, {vals[0]} : f32")
        e(f"    %pm1 = scalar.maxnumf %pm0, {vals[1]} : f32")
        e(f"    %pm2 = scalar.maxnumf %pm1, {vals[2]} : f32")
        e(f"    %pm3 = scalar.maxnumf %pm2, {vals[3]} : f32")
        cur = "%pm3"
        for mk in (1, 2):
            e(f"    %pmi{mk} = scalar.bitcast {cur} : f32 to i32")
            e(f"    %pmx{mk}, %pmv{mk} = kernel.subgroup.shuffle<xor> %pmi{mk}, %x{mk}, %x32 : i32, i32, i32")
            e(f"    %pmf{mk} = scalar.bitcast %pmx{mk} : i32 to f32")
            e(f"    %pmr{mk} = scalar.maxnumf {cur}, %pmf{mk} : f32")
            cur = f"%pmr{mk}"
        e(f"    %nmax = scalar.maxnumf %rmax, {cur} : f32")
        isfinite("%rmax", "pfin")
        e("    %pd = scalar.subf %rmax, %nmax : f32")
        exp_("%pd", "pex")
        e("    %prior = scf.if %pfin -> (f32) {")
        e("      scf.yield %pex : f32")
        e("    } else {")
        e("      scf.yield %zero : f32")
        e("    }")
        acc = "%zero"
        for mm in range(4):
            isfinite(vals[mm], f"wf{mm}")
            e(f"    %wd{mm} = scalar.subf {vals[mm]}, %nmax : f32")
            exp_(f"%wd{mm}", f"wx{mm}")
            e(f"    %w{mm} = scf.if %wf{mm} -> (f32) {{")
            e(f"      scf.yield %wx{mm} : f32")
            e("    } else {")
            e("      scf.yield %zero : f32")
            e("    }")
            e(f"    %ps{mm} = scalar.addf {acc}, %w{mm} : f32")
            acc = f"%ps{mm}"
            e(f"    %wh{mm} = scalar.fptrunc %w{mm} : f32 to f16")
            e(f"    view.store %wh{mm}, %p_view[%rg, %col{mm}] : f16, view<64x{P_PITCH}xf16>")
        for mk in (1, 2):
            e(f"    %psi{mk} = scalar.bitcast {acc} : f32 to i32")
            e(f"    %psx{mk}, %psv{mk} = kernel.subgroup.shuffle<xor> %psi{mk}, %x{mk}, %x32 : i32, i32, i32")
            e(f"    %psf{mk} = scalar.bitcast %psx{mk} : i32 to f32")
            e(f"    %psr{mk} = scalar.addf {acc}, %psf{mk} : f32")
            acc = f"%psr{mk}"
        e(f"    %nsum = scalar.fmaf %rsum, %prior, {acc} : f32")
        if RS2:
            e("    view.store %prior, %rs2_view[%rs2i] : f32, view<64xf32>")
        else:
            for mm in range(4):
                e(f"    view.store %prior, %rs_view[%rg, %col{mm}] : f32, view<64x16xf32>")
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        # P.V
        outs = []
        for rb in range(4):
            e(f"    %rb{rb}r = index.constant {16 * rb} : index")
            if RS2:
                e(f"    %sc{rb}h = index.mul %half, %c8 : index")
                e(f"    %sc{rb}i = index.add %rb{rb}r, %sc{rb}h : index")
                e(f"    %sc{rb} = vector.load %rs2_view[%sc{rb}i] : view<64xf32> -> {V8}")
            else:
                e(f"    %sc{rb} = vector.fragment.load<result> %rs_view[%rb{rb}r, %c0] shape [%m, %n] : view<64x16xf32> -> {V8}")
            e(f"    %pf{rb} = vector.fragment.load<lhs> %p_view[%rb{rb}r, %c0] shape [%m, %k] : view<64x{P_PITCH}xf16> -> {V16H}")
            for t in range(2):
                a = f"{rb}{t}"
                e(f"    %dt{a}0 = index.constant {8 * t} : index")
                e(f"    %dt{a} = index.add %wave, %dt{a}0 : index")
                e(f"    %dc{a} = index.mul %dt{a}, %c16 : index")
                e(f"    %vf{a} = vector.fragment.load<rhs> %vt_fr[%c0, %dc{a}] shape [%k, %n] : view<16x256xf16, %vt_lay> -> {V16H}")
                e(f"    %os{a} = vector.mulf %o{a}, %sc{rb} : {V8}")
                e(f"    %nx{a} = vector.mma %pf{rb}, %vf{a}, %os{a} : {V16H}, {V16H}, {V8}")
                if PVFENCE:
                    e("    scf.schedule.fence")
                if not tail:
                    outs.append(f"%nx{a}")
                    continue
                # boundary rows: fmaf chain over visible keys from the scaled o.
                # relative_key = ks - (start_pos + qs + (rb/2)*16), as i32. Rows
                # 0..15 of this block have a partly visible tile only when
                # -15 < rk < 16 (wave-uniform); otherwise the WMMA result stands.
                e(f"    %rk{a}0 = index.constant {(rb // 2) * 16} : index")
                e(f"    %rk{a}1 = index.add %start_pos, %qs : index")
                e(f"    %rk{a}2 = index.add %rk{a}1, %rk{a}0 : index")
                e(f"    %rk{a}i = index.cast %ks_ : index to i32")
                e(f"    %rk{a}j = index.cast %rk{a}2 : index to i32")
                e(f"    %rk{a} = scalar.subi %rk{a}i, %rk{a}j : i32")
                e(f"    %rk{a}15 = scalar.addi %rk{a}, %x15 : i32")
                e(f"    %need{a}a = scalar.cmpi sgt, %rk{a}15, %x0i : i32")
                e(f"    %need{a}b = scalar.cmpi slt, %rk{a}, %x16i : i32")
                e(f"    %need{a} = scalar.andi %need{a}a, %need{a}b : i1")
                e(f"    %nt{a} = scf.if %need{a} -> ({V8}) {{")
                e(f"    %tl{a}w = index.mul %wave, %c2048 : index")
                e(f"    %tl{a}b = index.constant {TL_OFF} : index")
                e(f"    %tl{a}o0 = index.add %tl{a}b, %tl{a}w : index")
                e(f"    %tl{a}o = index.cast %tl{a}o0 : index to offset")
                e(f"    %tl{a}n0 = index.add %tl{a}o0, %c1024 : index")
                e(f"    %tl{a}n = index.cast %tl{a}n0 : index to offset")
                e(f"    %tos{a} = buffer.view %pool[%tl{a}o] : buffer -> view<16x16xf32>")
                e(f"    %tnx{a} = buffer.view %pool[%tl{a}n] : buffer -> view<16x16xf32>")
                e(f"    vector.fragment.store<result> %os{a}, %tos{a}[%c0, %c0] shape [%m, %n] : {V8}, view<16x16xf32>")
                e(f"    vector.fragment.store<result> %nx{a}, %tnx{a}[%c0, %c0] shape [%m, %n] : {V8}, view<16x16xf32>")
                e(f"    %dim{a} = index.add %dc{a}, %sub : index")
                e(f"    %vr{a}0 = vector.load %vt_st[%dim{a}, %c0] : view<256x{VT_PITCH}xf16> -> {V8H}")
                e(f"    %vr{a}1 = vector.load %vt_st[%dim{a}, %c8] : view<256x{VT_PITCH}xf16> -> {V8H}")
                e(f"    %vw{a}0 = vector.extf %vr{a}0 : {V8H} to {V8}")
                e(f"    %vw{a}1 = vector.extf %vr{a}1 : {V8H} to {V8}")
                # rows 2i + half, i = 0..7 (scf.for: unrolled, 8 x 8 x 16 chains
                # exhausted the SGPRs). Every key: the softmax already stored
                # p = 0 for the keys HIP skips (same causal/context mask), and
                # fmaf(0, v, o) == o. (A divergent scf.if holding these LDS loads
                # lost lanes.)
                e(f"    scf.for %ti{a} = [%c0 to %c8 step %c1] {{")
                e(f"      %row{a}0 = index.mul %ti{a}, %c2 : index")
                e(f"      %row{a} = index.add %row{a}0, %half : index")
                e(f"      %rowi{a} = index.cast %row{a} : index to i32")
                e(f"      %part{a} = scalar.cmpi slt, %rowi{a}, %rk{a}15 : i32")
                e(f"      %t0{a} = view.load %tos{a}[%row{a}, %sub] : view<16x16xf32> -> f32")
                e(f"      %prow{a} = index.add %rb{rb}r, %row{a} : index")
                e(f"      %pr{a}0 = vector.load %p_view[%prow{a}, %c0] : view<64x{P_PITCH}xf16> -> {V8H}")
                e(f"      %pr{a}1 = vector.load %p_view[%prow{a}, %c8] : view<64x{P_PITCH}xf16> -> {V8H}")
                e(f"      %pw{a}0 = vector.extf %pr{a}0 : {V8H} to {V8}")
                e(f"      %pw{a}1 = vector.extf %pr{a}1 : {V8H} to {V8}")
                cur = f"%t0{a}"
                for key in range(16):
                    h, j = divmod(key, 8)
                    e(f"      %pe{a}_{key} = vector.extract %pw{a}{h}[{j}] : {V8} -> f32")
                    e(f"      %ve{a}_{key} = vector.extract %vw{a}{h}[{j}] : {V8} -> f32")
                    e(f"      %tf{a}_{key} = scalar.fmaf %pe{a}_{key}, %ve{a}_{key}, {cur} : f32")
                    cur = f"%tf{a}_{key}"
                e(f"      %tn{a} = view.load %tnx{a}[%row{a}, %sub] : view<16x16xf32> -> f32")
                e(f"      %tr{a} = scf.select %part{a}, {cur}, %tn{a} : f32")
                e(f"      view.store %tr{a}, %tnx{a}[%row{a}, %sub] : f32, view<16x16xf32>")
                e("    }")
                e(f"    %ntl{a} = vector.fragment.load<result> %tnx{a}[%c0, %c0] shape [%m, %n] : view<16x16xf32> -> {V8}")
                e(f"    scf.yield %ntl{a} : {V8}")
                e("    } else {")
                e(f"    scf.yield %nx{a} : {V8}")
                e("    }")
                outs.append(f"%nt{a}")
        return outs, "%nmax", "%nsum", nxt

    e("  %c272 = index.constant 272 : index")
    e("  %c2048 = index.constant 2048 : index")
    e("  %c1088 = index.constant 1088 : index")
    e("  %x15 = scalar.constant 15 : i32")
    e("  %x0i = scalar.constant 0 : i32")
    e("  %x16i = scalar.constant 16 : i32")
    for key in range(16):
        e(f"  %x{key}k = scalar.constant {key} : i32")
        e(f"  %ck{key} = index.constant {key} : index")
    e(f"  %oinit = vector.fragment<init> %zeros8 shape [%m, %n] : {V8}")
    # split: tiles with key_start + 15 <= start_pos + qs take the plain form
    e("  %spq = index.add %start_pos, %qs : index")
    e("  %spq1 = index.add %spq, %c1 : index")
    e("  %split0 = index.div %spq1, %c16 : index")
    e("  %split1 = index.mul %split0, %c16 : index")
    e("  %split = index.min %split1, %max_vis : index")
    names = [f"%o{rb}{t}" for rb in range(4) for t in range(2)]
    kvn = ["%kvc0", "%kvc1", "%kvc2", "%kvc3"]
    types = ", ".join([V8] * 8 + ["f32", "f32"] + [V8H] * 4)

    def loop(lo, hi, init_o, init_m, init_s, init_kv, tail, res):
        init = ", ".join(f"{n} = {v} : {V8}" for n, v in zip(names, init_o))
        init += f", %rmax = {init_m} : f32, %rsum = {init_s} : f32, "
        init += ", ".join(f"{n} = {v} : {V8H}" for n, v in zip(kvn, init_kv))
        e(f"  {', '.join(res)} = scf.for %ks_ = [{lo} to {hi} step %c16]({init}) -> ({types}) {{")
        outs, nm, ns, nxt = body(tail)
        e(f"    scf.yield {', '.join(outs)}, {nm}, {ns}, {', '.join(nxt)} : {types}")
        e("  }")

    kv0 = load_kv("%c0", "p0", "  ")
    r1 = [f"%of{i}" for i in range(8)] + ["%fmax", "%fsum"] + [f"%fkv{i}" for i in range(4)]
    loop("%c0", "%split", ["%oinit"] * 8, "%ninf", "%zero", kv0, False, r1)
    r2 = [f"%og{i}" for i in range(8)] + ["%gmax", "%gsum"] + [f"%gkv{i}" for i in range(4)]
    if __import__("os").environ.get("YAH_ATTN_NOTAIL") == "1":
        # debug: every tile in the plain form (numerics differ on boundary tiles)
        e("  %notail_hi = index.add %max_vis, %c0 : index")
        loop("%split", "%notail_hi", r1[:8], "%fmax", "%fsum", r1[10:], False, r2)
    else:
        loop("%split", "%max_vis", r1[:8], "%fmax", "%fsum", r1[10:], True, r2)
    # ---- epilogue: row sums, replicated, then o / sum * sigmoid(gate)
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    for mm in range(4):
        e(f"  %ec{mm}c = index.constant {mm} : index")
        e(f"  %ec{mm} = index.add %seg4, %ec{mm}c : index")
        if __import__("os").environ.get("YAH_ATTN_DBG") in ("rsc", "tid"):
            if mm == 0:
                e("  %dbg_ti = index.cast %tid : index to i32")
                e("  %dbg_tf = scalar.sitofp %dbg_ti : i32 to f32")
            e(f"  view.store %dbg_tf, %rs_view[%rg, %ec{mm}] : f32, view<64x16xf32>")
        else:
            e(f"  view.store %gsum, %rs_view[%rg, %ec{mm}] : f32, view<64x16xf32>")
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e("  %ew = index.mul %wave, %c1024 : index")
    e("  %ewo = index.cast %ew : index to offset")
    e("  %eo_view = buffer.view %pool[%ewo] : buffer -> view<16x16xf32>")
    for rb in range(4):
        for t in range(2):
            a = f"{rb}{t}"
            idx = rb * 2 + t
            e(f"  vector.fragment.store<result> %og{idx}, %eo_view[%c0, %c0] shape [%m, %n] : {V8}, view<16x16xf32>")
            e(f"  %edt{a}0 = index.constant {8 * t} : index")
            e(f"  %edt{a} = index.add %wave, %edt{a}0 : index")
            e(f"  %edc{a} = index.mul %edt{a}, %c16 : index")
            e(f"  %edim{a} = index.add %edc{a}, %sub : index")
            e(f"  %ehead{a} = index.constant {rb % 2} : index")
            e(f"  %eh{a} = index.add %head0, %ehead{a} : index")
            e(f"  %ehb{a} = index.mul %eh{a}, %c256 : index")
            e(f"  %eq{a}0 = index.constant {(rb // 2) * 16} : index")
            e(f"  %eq{a} = index.add %qs, %eq{a}0 : index")
            # 8 rows unrolled, loads unconditional (row clamped) so the gate
            # loads issue together; only the store is guarded
            rows = []
            for i in range(8):
                b = f"{a}_{i}"
                e(f"  %erow{b}c = index.constant {2 * i} : index")
                e(f"  %erow{b} = index.add %erow{b}c, %half : index")
                e(f"  %elq{b} = index.add %eq{a}, %erow{b} : index")
                e(f"  %elive{b} = index.cmp ult, %elq{b}, %B : index")
                e(f"  %elc{b} = index.min %elq{b}, %B_1 : index")
                e(f"  %eoff{b}0 = index.mul %elc{b}, %c6144 : index")
                e(f"  %eoff{b}1 = index.add %eoff{b}0, %ehb{a} : index")
                e(f"  %eoff{b} = index.add %eoff{b}1, %edim{a} : index")
                e(f"  %egv{b} = view.load %g_flat[%eoff{b}] : view<[%qtot]xf32> -> f32")
                rows.append(b)
            for i, b in enumerate(rows):
                e(f"  %eov{b} = view.load %eo_view[%erow{b}, %sub] : view<16x16xf32> -> f32")
                e(f"  %ersr{b}c = index.constant {16 * rb} : index")
                e(f"  %ersr{b} = index.add %ersr{b}c, %erow{b} : index")
                e(f"  %eden{b} = view.load %rs_view[%ersr{b}, %c0] : view<64x16xf32> -> f32")
                e(f"  %epos{b} = scalar.cmpf ogt, %eden{b}, %zero : f32")
                e(f"  %edv{b} = scalar.divf %eov{b}, %eden{b} : f32")
                e(f"  %eval{b} = scf.select %epos{b}, %edv{b}, %zero : f32")
                e(f"  %egn{b} = scalar.negf %egv{b} : f32")
                e(f"  %egm{b} = scalar.mulf %egn{b}, %log2e : f32")
                e(f"  %egx{b} = scalar.exp2f<afn> %egm{b} : f32")
                e(f"  %egd{b} = scalar.addf %one, %egx{b} : f32")
                e(f"  %egr{b} = scalar.divf %one, %egd{b} : f32")
                e(f"  %eout{b} = scalar.mulf %eval{b}, %egr{b} : f32")
                if __import__("os").environ.get("YAH_ATTN_DBG") == "rsc":
                    e(f"  %edbg{b} = view.load %rs_view[%ersr{b}, %sub] : view<64x16xf32> -> f32")
                dbg = {"sum": f"%eden{b}", "o": f"%eov{b}", "rsc": f"%edbg{b}", "tid": "%dbg_tf"}.get(__import__("os").environ.get("YAH_ATTN_DBG", ""), f"%eout{b}")
                e(f"  scf.if %elive{b} {{")
                e(f"    view.store {dbg}, %o_flat[%eoff{b}] : f32, view<[%qtot]xf32>")
                e("  }")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def gen_vtrans():
    """yah_transpose_v16: token-major f16 V cache [token][1024] -> V^T blocked
    [4 kv heads][ceil(capacity/16) tiles][256 dims][16 keys], tokens >=
    token_count zero.
    32 x 32 tiles through LDS; grid (1024/32, pitch/32 rounded up)."""
    L = []
    e = L.append
    e("// GENERATED by tools/gen_attn_hip.py (gen_vtrans) -- edit the generator.")
    e("amdgpu.target<gfx1151> @vtrans_w32 {subgroup_size = 32}")
    e("config.decl @yah_vtrans.token_count : %value: index where [range(%value, 1, 1048576)]")
    e("config.decl @yah_vtrans.cache_capacity : %value: index where [range(%value, 1, 1048576)]")
    e("")
    e("kernel.def target(@vtrans_w32) @yah_transpose_v16() {")
    e("  %cap = config.get @yah_vtrans.cache_capacity : index")
    e("  %c1 = index.constant 1 : index")
    e("  %c31 = index.constant 31 : index")
    e("  %c32 = index.constant 32 : index")
    e("  %c256 = index.constant 256 : index")
    e("  %t0 = index.add %cap, %c31 : index")
    e("  %tiles = index.div %t0, %c32 : index")
    e("  kernel.launch.config workgroups(%c32, %tiles, %c1) workgroup_size(%c256, %c1, %c1) : index")
    e("} launch(%src: buffer, %dst: buffer) {")
    e("  %base = index.constant 0 : offset")
    for v in (0, 1, 4, 8, 15, 16, 32, 1024):
        e(f"  %c{v} = index.constant {v} : index")
    e("  %ntok0 = config.get @yah_vtrans.token_count : index")
    e("  %cap = config.get @yah_vtrans.cache_capacity : index")
    e("  %ntok = index.min %ntok0, %cap : index")
    e("  %p15 = index.add %cap, %c15 : index")
    e("  %p16 = index.div %p15, %c16 : index")
    e("  %pitch = index.mul %p16, %c16 : index")
    e("  %stot = index.mul %cap, %c1024 : index")
    e("  %dtot = index.mul %pitch, %c1024 : index")
    e("  %s_na, %d_na = buffer.assume.noalias %src, %dst : buffer, buffer")
    e("  %s_flat = buffer.view %s_na[%base] : buffer -> view<[%stot]xf16>")
    e("  %d_flat = buffer.view %d_na[%base] : buffer -> view<[%dtot]xf16>")
    e("  %tb = index.constant 2304 : offset")
    e("  %tile = buffer.alloca<workgroup> align(16) %tb : buffer")
    e("  %tv = buffer.view %tile[%base] : buffer -> view<32x36xf16>")
    e("  %cb = kernel.workgroup.id<x> : index")
    e("  %tbk = kernel.workgroup.id<y> : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %col0 = index.mul %cb, %c32 : index")
    e("  %tok0 = index.mul %tbk, %c32 : index")
    e("  %r = index.div %tid, %c8 : index")
    e("  %sg0 = index.rem %tid, %c8 : index")
    e("  %sg = index.mul %sg0, %c4 : index")
    e("  %zh4 = vector.constant 0.0 : vector<4xf16>")
    # load: token tok0 + r, cols col0 + sg .. +3 (zero past the tokens)
    e("  %tok = index.add %tok0, %r : index")
    e("  %live = index.cmp ult, %tok, %ntok : index")
    e("  %cap1 = index.sub %cap, %c1 : index")
    e("  %tokc = index.min %tok, %cap1 : index")
    e("  %sa0 = index.mul %tokc, %c1024 : index")
    e("  %sa1 = index.add %sa0, %col0 : index")
    e("  %sa = index.add %sa1, %sg : index")
    e("  %raw = vector.load %s_flat[%sa] : view<[%stot]xf16> -> vector<4xf16>")
    e("  %val = scf.select %live, %raw, %zh4 : vector<4xf16>")
    e("  vector.store %val, %tv[%r, %sg] : vector<4xf16>, view<32x36xf16>")
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    # store: col col0 + r, tokens tok0 + sg .. +3
    els = []
    for j in range(4):
        e(f"  %tj{j}c = index.constant {j} : index")
        e(f"  %tj{j} = index.add %sg, %tj{j}c : index")
        e(f"  %x{j} = view.load %tv[%tj{j}, %r] : view<32x36xf16> -> f16")
        els.append(f"%x{j}")
    e(f"  %ov = vector.from_elements {', '.join(els)} : vector<4xf16>")
    e("  %col = index.add %col0, %r : index")
    e("  %dt = index.add %tok0, %sg : index")
    e("  %dlive = index.cmp ult, %dt, %pitch : index")
    # [kv_head][tile][dim][16]: (kvh * tiles + dt/16) * 4096 + d * 16 + dt%16
    e("  %c256b = index.constant 256 : index")
    e("  %c4096 = index.constant 4096 : index")
    e("  %kvh = index.div %col, %c256b : index")
    e("  %dd = index.rem %col, %c256b : index")
    e("  %dtl = index.div %dt, %c16 : index")
    e("  %dj = index.rem %dt, %c16 : index")
    e("  %da0 = index.mul %kvh, %p16 : index")
    e("  %da1 = index.add %da0, %dtl : index")
    e("  %da2 = index.mul %da1, %c4096 : index")
    e("  %da3 = index.mul %dd, %c16 : index")
    e("  %da4 = index.add %da2, %da3 : index")
    e("  %da5 = index.add %da4, %dj : index")
    e("  %dlim = index.sub %dtot, %c4 : index")
    e("  %da = index.min %da5, %dlim : index")  # in range already; for the prover
    e("  scf.if %dlive {")
    e("    vector.store %ov, %d_flat[%da] : vector<4xf16>, view<[%dtot]xf16>")
    e("  }")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "vtrans":
        out = sys.argv[2] if len(sys.argv) > 2 else "yah_transpose_v16.loom"
        open(out, "w").write(gen_vtrans())
        print(out)
        return
    out = sys.argv[1] if len(sys.argv) > 1 else "yah_attn_hip.loom"
    open(out, "w").write(gen())
    print(out)


if __name__ == "__main__":
    main()
