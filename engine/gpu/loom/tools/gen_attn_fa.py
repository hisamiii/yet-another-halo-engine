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
# SKIP (default): FA4's conditional rescale, exact form (THR=0): O and the
# sum are rescaled only when some row max of the wave grew. Same bits as
# always rescaling; VALU/WMMA 14.1 -> 11.1, 69.8 -> 67.1 M cycles (pp8192).
SKIP = os.environ.get("YAH_ATTN_FA_SKIP", "1") == "1"
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
# Key-loop policy. unroll(%c2) schedule(recurrence) helped the always-rescale
# form (76.8 -> 73.8 M) but with SKIP it puts 73 O copies on the skip path of
# the second copy; without it the SKIP branch leaves no back-edge copies.
POL = os.environ.get("YAH_ATTN_FA_POL", "")
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
K_OFF = 0                            # KT x 264 x 2 (8448 at KT=16)
V_OFF = 16 * 264 * 2 * (int(os.environ.get("YAH_ATTN_FA_KT", "16")) // 16)
# GQA packing (YAH_ATTN_FA_GQA=1): a workgroup runs all 6 query heads of a KV
# group x 16 tokens (6 query blocks, 12 waves), so each K/V tile is staged
# once for 6 heads instead of 2 and all waves share one causal extent.
# Default: 2 heads x 32 tokens (4 query blocks, 8 waves).
GQAP = os.environ.get("YAH_ATTN_FA_GQA", "0") == "1"
HPW, QT = (6, 16) if GQAP else (2, 32)   # query heads, query tokens per workgroup
NQB = HPW * QT // 16                     # query blocks (wave pairs)
NT = 64 * NQB                            # threads
# VSB: one V buffer, V(i) loaded at the top of phase A and staged at its end
# (phase B reads it); QH: Q staged in two 128-dim halves (each wave loads its
# half in its round). Together LDS drops under 32 KB: four 8-wave
# workgroups per WGP (8 waves/SIMD at <= 192 VGPRs) instead of three.
VSB = os.environ.get("YAH_ATTN_FA_VSB", "1") == "1"   # 67.2 -> 64.6 M (pp8192), same bits
QH = os.environ.get("YAH_ATTN_FA_QH", "0") == "1"
VBUFS = 1 if VSB else 2
# KT=32: 32-key tiles (two 16-key sub-tiles per barrier pair, max/rescale and
# score exchange): half the per-tile overhead, but LDS 54 KB -> two
# workgroups per WGP. The running max moves per 32 keys: not HIP's order.
KT = int(os.environ.get("YAH_ATTN_FA_KT", "16"))
NSUB = KT // 16
if KT == 32:
    assert VSB and SKIP and HIPNUM and not S8 and not MSKIF and not PVZ
    VT_PITCH = 40                        # 32 keys (+8): 80-B rows, conflict-free
# VQ8 (int8 config, V half): V^T stored as uint8 around a per-channel centre
# (yah_vstat / yah_vq8: u = rne((v - c) / s) + 128); staging converts u - 128
# to f16 exactly (P.V stays f16) and the epilogue applies o / l * s + c.
VQ8 = os.environ.get("YAH_ATTN_FA_VQ8", "0") == "1"
VQFENCE = os.environ.get("YAH_ATTN_FA_VQFENCE", "1") == "1"
# VQ4 (kv4 configs, V half): V^T 15-level nibbles per channel per 16-key tile
# with f16 (S, C') = (16 s, c - 23 s) (yah_vq4); staging builds f16 1 + u/16
# (0x3c00 | u << 6) by masks and one packed fma f * S + C'. P.V stays f16.
VQ4 = os.environ.get("YAH_ATTN_FA_VQ4", "0") == "1"
# Quantized K (engine/run/kvq/README.md): the K cache is centred by its
# per-channel prompt mean (yah_kmean) and decoded to f16 while staging; QK^T
# runs the f16 path with an f16 (unquantized) Q.
#   K4 (kv4a16, YAH_ATTN_FA_K4=1): H256 (k - m) as asymmetric int4 per 32-dim
#     group (yah_kq4), decoded as (1 + u/16) * 16 s + lo - 16 s, one packed fma
#     per pair; Q is rotated by the same H256 while staging.
#   K8 (kv8a16, YAH_ATTN_FA_K8=1): int8 per token half (yah_kq8), decoded as
#     (1 + u/256) * 256 s - 384 s.
# The integer-compute paths (iu8 int8, int4->int8 kv4a8, iu4 kv4a4) were
# dropped on 2026-10-02: kv4a4 saved ~0.4% of pp8192 at 96% 99%-precision.
K4 = os.environ.get("YAH_ATTN_FA_K4", "0") == "1"
K8 = os.environ.get("YAH_ATTN_FA_K8", "0") == "1"
KDEC = K4 or K8                       # K decoded to f16 at staging
if KDEC:
    assert KT == 16 and VSB and not GQAP and not QH and not (K4 and K8)
S_OFF = V_OFF + VBUFS * 256 * VT_PITCH * 2   # S partials: 2*NSUB planes x NT x 4 f32
Q_PITCH = 136 if QH else 264
Q_DIMS = 128 if QH else 256
Q_END = NQB * 16 * Q_PITCH * 2           # Q stage (prologue only)
# Q-drain dummy store: past the Q stage, or in the S slots when they are past it
DRAIN_OFF = S_OFF if S_OFF >= Q_END else Q_END
POOL = max(S_OFF + NT * 32 * NSUB, DRAIN_OFF + NT * 16)
# 2 heads: 41216 (three workgroups per WGP); 6 heads: 56832 (two, 24 waves)
assert POOL <= 65536


def had64(e, pre, vecs, xors, scale):
    """Walsh-Hadamard over this thread's 64 dims (16 vector<4xf32> in dim order:
    stages 1..32 in-thread) and across threads (lane xor x = qpart bit log2(x):
    dims 64, 128 apart), times `scale`. Butterfly: lower a + b, upper
    lower - upper, as yah_kq4 (same orthogonal H on both sides)."""
    xs = []
    for c in range(16):
        for i in range(4):
            e(f"  %{pre}0_{4 * c + i} = vector.extract {vecs[c]}[{i}] : {V4} -> f32")
            xs.append(f"%{pre}0_{4 * c + i}")
    st = 0
    for h in (1, 2, 4, 8, 16, 32):
        st += 1
        nx = []
        for i in range(64):
            j = i ^ h
            if i & h:
                e(f"  %{pre}{st}_{i} = scalar.subf {xs[j]}, {xs[i]} : f32")
            else:
                e(f"  %{pre}{st}_{i} = scalar.addf {xs[i]}, {xs[j]} : f32")
            nx.append(f"%{pre}{st}_{i}")
        xs = nx
    cur = []
    for c in range(16):
        e(f"  %{pre}v{c} = vector.from_elements {', '.join(xs[4 * c:4 * c + 4])} : {V4}")
        cur.append(f"%{pre}v{c}")
    for x in xors:
        e(f"  %{pre}x{x}c = scalar.constant {x} : i32")
        e(f"  %{pre}b{x}a = index.div %qpart, %c{x} : index")
        e(f"  %{pre}b{x}b = index.rem %{pre}b{x}a, %c2 : index")
        e(f"  %{pre}u{x} = index.cmp eq, %{pre}b{x}b, %c1 : index")
        nxt = []
        for c in range(16):
            e(f"  %{pre}s{x}_{c}, %{pre}sv{x}_{c} = kernel.subgroup.shuffle<xor> {cur[c]}, %{pre}x{x}c, %x32 : {V4}, i32, i32")
            e(f"  %{pre}a{x}_{c} = vector.addf {cur[c]}, %{pre}s{x}_{c} : {V4}")
            e(f"  %{pre}d{x}_{c} = vector.subf %{pre}s{x}_{c}, {cur[c]} : {V4}")
            e(f"  %{pre}f{x}_{c} = scf.select %{pre}u{x}, %{pre}d{x}_{c}, %{pre}a{x}_{c} : {V4}")
            nxt.append(f"%{pre}f{x}_{c}")
        cur = nxt
    e(f"  %{pre}rs = scalar.constant {scale} : f32")
    e(f"  %{pre}rs4 = vector.splat %{pre}rs : {V4}")
    out = []
    for c in range(16):
        e(f"  %{pre}r{c} = vector.mulf {cur[c]}, %{pre}rs4 : {V4}")
        out.append(f"%{pre}r{c}")
    return out


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
    e(f"  %chpw = index.constant {HPW} : index")
    e(f"  %cqt1 = index.constant {QT - 1} : index")
    e(f"  %cqt = index.constant {QT} : index")
    e(f"  %cnt = index.constant {NT} : index")
    e("  %pairs = index.div %nh, %chpw : index")
    e("  %tp = index.add %token_count, %cqt1 : index")
    e("  %qblocks = index.div %tp, %cqt : index")
    e("  kernel.launch.config workgroups(%qblocks, %pairs, %c1) workgroup_size(%cnt, %c1, %c1) : index")
    e("} launch(%query: buffer, %gate: buffer, %key_cache: buffer, %value_cache: buffer, %output: buffer, %lse: buffer"
      + (", %kscale: buffer" if KDEC else "") + (", %vstat: buffer" if VQ8 else "")
      + (", %vstat4: buffer" if VQ4 else "") + ") {")
    e("  %base = index.constant 0 : offset")
    for v in (0, 1, 2, 3, 4, 6, 8, 15, 16, 31, 32, 64, 128, 256, 1024, 4096, 6144):
        e(f"  %c{v} = index.constant {v} : index")
    e(f"  %chpw = index.constant {HPW} : index")
    e(f"  %cqt = index.constant {QT} : index")
    e(f"  %cqt1 = index.constant {QT - 1} : index")
    e(f"  %cnt = index.constant {NT} : index")
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
    e(f"  %zq16 = vector.constant 0.0 : {V16H}")
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
    if KDEC:   # kv4a16: int4 K [token][128 dwords], f16x2 (16 s, lo - 16 s) [token][32]
        # kv8a16: int8 K [token][256 dwords], f16x2 (256 s, -384 s) [token][8]
        e(f"  %kq32tot = index.mul %cache_capacity, %c{128 if K4 else 256} : index")
        e(f"  %kstot = index.mul %cache_capacity, %c{32 if K4 else 8} : index")
        e("  %k_flat = buffer.view %k_na[%base] : buffer -> view<[%kq32tot]xi32>")
        e("  %ks_na = buffer.assume.noalias %kscale : buffer")
        e("  %ks_flat = buffer.view %ks_na[%base] : buffer -> view<[%kstot]xi32>")
    else:
        e("  %k_flat = buffer.view %k_na[%base] : buffer -> view<[%kvtot]xf16>")
    e("  %cap15 = index.add %cache_capacity, %c15 : index")
    e("  %vtiles = index.div %cap15, %c16 : index")
    e("  %vpitch = index.mul %vtiles, %c16 : index")
    e("  %vtot = index.mul %vpitch, %c1024 : index")
    e("  %vlast = index.sub %vpitch, %c16 : index")
    if VQ4:   # nibble V^T: [kvh][tile][256 dims] x 2 dwords, stats 1 dword
        assert KT == 16 and not VQ8
        e("  %vq4tot0 = index.mul %vtiles, %c1024 : index")
        e("  %vq4tot = index.mul %vq4tot0, %c2 : index")
        e("  %v_flat = buffer.view %v_na[%base] : buffer -> view<[%vq4tot]xi32>")
        e("  %vs4_na = buffer.assume.noalias %vstat4 : buffer")
        e("  %vs4_flat = buffer.view %vs4_na[%base] : buffer -> view<[%vq4tot0]xi32>")
    elif VQ8:   # uint8 V^T as dwords: [kvh][tile][256 dims][16 B] = 4 dwords per dim
        assert KT == 16
        e("  %vq32tot = index.mul %vpitch, %c256 : index")
        e("  %v_flat = buffer.view %v_na[%base] : buffer -> view<[%vq32tot]xi32>")
        e("  %vs_na = buffer.assume.noalias %vstat : buffer")
        e("  %vs_flat = buffer.view %vs_na[%base] : buffer -> view<2048xf32>")
        e("  %vmsk = scalar.constant 16711935 : i32")      # 0x00ff00ff
        e("  %vmag = scalar.constant 1677747200 : i32")    # 0x64006400
        e("  %v8s = scalar.constant 8 : i32")
        e("  %vmsk4 = vector.splat %vmsk : vector<4xi32>")
        e("  %vmag4 = vector.splat %vmag : vector<4xi32>")
        e("  %v8s4 = vector.splat %v8s : vector<4xi32>")
        e("  %c1152 = scalar.constant -1152.0 : f32")
    else:
        e("  %v_flat = buffer.view %v_na[%base] : buffer -> view<[%vtot]xf16>")
    e(f"  %pool_bytes = index.constant {POOL} : offset")
    e("  %pool = buffer.alloca<workgroup> align(16) %pool_bytes : buffer")
    e(f"  %qs_view = buffer.view %pool[%base] : buffer -> view<{NQB * 16}x{Q_PITCH}xf16>")
    e(f"  %q_lay = encoding.layout.strided [1, {Q_PITCH}] : encoding<layout>")
    e(f"  %q_fr = buffer.view %pool[%base] : buffer -> view<{Q_DIMS}x{NQB * 16}xf16, %q_lay>")
    e(f"  %k_o = index.constant {K_OFF} : offset")
    e(f"  %k_view = buffer.view %pool[%k_o] : buffer -> view<{KT}x{KT_PITCH}xf16>")
    e(f"  %v_o = index.constant {V_OFF} : offset")
    e(f"  %v_view = buffer.view %pool[%v_o] : buffer -> view<{256 * VBUFS}x{VT_PITCH}xf16>")
    e(f"  %s_o = index.constant {S_OFF} : offset")
    # two planes of 4 f32 per lane, so each b128 access is contiguous across lanes
    e(f"  %s_view = buffer.view %pool[%s_o] : buffer -> view<{2 * NT}x4xf32>")
    e(f"  %s8_view = buffer.view %pool[%s_o] : buffer -> view<{NT}x8xf32>")
    if KT == 32:
        e(f"  %s32_view = buffer.view %pool[%s_o] : buffer -> view<{4 * NT}x4xf32>")
        e("  %c24 = index.constant 24 : index")
    # ids
    if SWZ or LPT:
        e("  %wgx = kernel.workgroup.id<x> : index")
        e("  %wgy = kernel.workgroup.id<y> : index")
        e("  %nh_ = config.get @attention_prefill.num_heads : index")
        e("  %npairs = index.div %nh_, %chpw : index")
        e("  %tpq = index.add %B, %cqt1 : index")
        e("  %nqb = index.div %tpq, %cqt : index")
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
    e("  %tid2 = index.add %tid, %cnt : index")
    e("  %ptid2 = index.add %ptid, %cnt : index")
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
    e("  %qs = index.mul %qb, %cqt : index")
    if GQAP:
        e("  %kvh = index.add %hp, %c0 : index")
        e("  %head0 = index.mul %kvh, %c6 : index")
    else:
        e("  %kvh = index.div %hp, %c3 : index")
        e("  %pig = index.rem %hp, %c3 : index")
        e("  %kvh6 = index.mul %kvh, %c6 : index")
        e("  %pig2 = index.mul %pig, %c2 : index")
        e("  %head0 = index.add %kvh6, %pig2 : index")
    e("  %kvbase = index.mul %kvh, %c256 : index")
    e("  %vhb0 = index.mul %kvh, %vtiles : index")
    e("  %vhb = index.mul %vhb0, %c4096 : index")
    # V staging lanes: the last 256 threads (lane = dim); K: the first 256
    e(f"  %vtoff = index.constant {NT - 256} : index")
    if NT > 256:   # max(): threads below vtoff are K-only (guarded) but must stay in range
        e("  %vtm = index.max %tid, %vtoff : index")
        e("  %vt = index.sub %vtm, %vtoff : index")
    else:
        e("  %vt = index.add %tid, %c0 : index")
    e("  %vlane = index.mul %vt, %c16 : index")
    e("  %vhl = index.add %vhb, %vlane : index")
    if VQ8:
        e("  %vhl4 = index.div %vhl, %c4 : index")
    if VQ4:
        e("  %vhb4 = index.mul %kvh, %vtiles : index")
        e("  %v4m = scalar.constant 62915520 : i32")        # 0x03c003c0
        e("  %v4g = scalar.constant 1006648320 : i32")     # 0x3c003c00
    e("  %ctx_end = index.add %start_pos, %B : index")
    e("  %qs32 = index.add %qs, %cqt : index")
    e("  %vis0 = index.add %start_pos, %qs32 : index")
    e("  %max_vis = index.min %ctx_end, %vis0 : index")
    e("  %B_1 = index.sub %B, %c1 : index")
    e("  %cap_1 = index.sub %cache_capacity, %c1 : index")
    # this lane's query: block wqb = (head0 + wqb%2, tokens qs + (wqb/2)*16 + sub)
    e("  %wqh = index.rem %wqb, %chpw : index")
    e("  %wqt0 = index.div %wqb, %chpw : index")
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
    e("  %qrb2 = index.rem %qrb, %chpw : index")
    e("  %qrh = index.add %head0, %qrb2 : index")
    e("  %qrq0 = index.div %qrb, %chpw : index")
    e("  %qrq1 = index.mul %qrq0, %c16 : index")
    e("  %qrq2 = index.add %qs, %qrq1 : index")
    e("  %qlq = index.add %qrq2, %qrr : index")
    e("  %qlive = index.cmp ult, %qlq, %B : index")
    e("  %qlqc = index.min %qlq, %B_1 : index")
    e("  %qrow0 = index.mul %qlqc, %c6144 : index")
    e("  %qhb = index.mul %qrh, %c256 : index")
    e("  %qrow = index.add %qrow0, %qhb : index")
    e("  %qsc_v = vector.splat %qscale : " + V4)
    if K4:
        e(f"  %z4h = vector.constant 0.0 : {V4}")
    if QH:
        e("  %sgq = kernel.subgroup.id : index")
        e("  %hdu = index.rem %sgq, %c2 : index")       # = hd, provably wave-uniform
    rounds = 2 if QH else 1
    nchunk = 4 if QH else 8                             # 8-dim chunks per thread
    prev = None
    for r in range(rounds):
        e(f"  %qdb{r}a = index.mul %qpart, %c{8 * nchunk} : index")
        rotq = {}
        if K4:   # H256 of the scaled f32 Q (dims qpart*64 + 8c + 0..7) first
            raw = []
            for c in range(nchunk):
                t = f"{r}_{c}"
                e(f"  %qdc{t} = index.constant {8 * c} : index")
                e(f"  %qd{t} = index.add %qdb{r}a, %qdc{t} : index")
                e(f"  %qa{t} = index.add %qrow, %qd{t} : index")
                e(f"  %qa{t}b = index.add %qa{t}, %c4 : index")
                e(f"  %qv{t}a = vector.load %q_flat[%qa{t}] : view<[%qtot]xf32> -> {V4}")
                e(f"  %qv{t}b = vector.load %q_flat[%qa{t}b] : view<[%qtot]xf32> -> {V4}")
                e(f"  %qm{t}a0 = vector.mulf %qv{t}a, %qsc_v : {V4}")
                e(f"  %qm{t}b0 = vector.mulf %qv{t}b, %qsc_v : {V4}")
                raw += [f"%qm{t}a0", f"%qm{t}b0"]
            rot = had64(e, "fh", raw, (1, 2), 0.0625)
            for c in range(nchunk):
                rotq[c] = (rot[2 * c], rot[2 * c + 1])
        for c in range(nchunk):
            t = f"{r}_{c}"
            if K4:
                e(f"  %qm{t}a = vector.addf {rotq[c][0]}, %z4h : {V4}")
                e(f"  %qm{t}b = vector.addf {rotq[c][1]}, %z4h : {V4}")
            else:
                e(f"  %qdc{t} = index.constant {8 * c} : index")
                e(f"  %qd{t} = index.add %qdb{r}a, %qdc{t} : index")         # dim within the stage
                e(f"  %qdg{t}c = index.constant {128 * r} : index")
                e(f"  %qdg{t} = index.add %qd{t}, %qdg{t}c : index")       # global dim
                e(f"  %qa{t} = index.add %qrow, %qdg{t} : index")
                e(f"  %qa{t}b = index.add %qa{t}, %c4 : index")
                e(f"  %qv{t}a = vector.load %q_flat[%qa{t}] : view<[%qtot]xf32> -> {V4}")
                e(f"  %qv{t}b = vector.load %q_flat[%qa{t}b] : view<[%qtot]xf32> -> {V4}")
                e(f"  %qm{t}a = vector.mulf %qv{t}a, %qsc_v : {V4}")
                e(f"  %qm{t}b = vector.mulf %qv{t}b, %qsc_v : {V4}")
            e(f"  %qh{t}a = vector.fptrunc %qm{t}a : {V4} to vector<4xf16>")
            e(f"  %qh{t}b = vector.fptrunc %qm{t}b : {V4} to vector<4xf16>")
            e(f"  %qh{t} = vector.concat<0> %qh{t}a, %qh{t}b : vector<4xf16>, vector<4xf16> -> {V8H}")
            e(f"  %qz{t} = scf.select %qlive, %qh{t}, %zh8 : {V8H}")
            e(f"  vector.store %qz{t}, %qs_view[%qr, %qd{t}] : {V8H}, view<{NQB * 16}x{Q_PITCH}xf16>")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        # Q^T fragments (B: [dim, query]) of this wave's 128 dims, in registers
        if r == 0:
            e("  %wq16 = index.mul %wqb, %c16 : index")
        names = [f"%qf{c}" if not QH else f"%qr{r}f{c}" for c in range(8)]

        def qload():
            for c in range(8):
                e(f"  %qfd{r}{c}c = index.constant {16 * c} : index")
                if QH:
                    e(f"  %qfd{r}{c} = index.add %qfd{r}{c}c, %c0 : index")
                else:
                    e(f"  %qfd{r}{c} = index.add %hd128, %qfd{r}{c}c : index")
                e(f"  %ql{r}f{c} = vector.fragment.load<rhs> %q_fr[%qfd{r}{c}, %wq16] shape [%k, %n] : view<{Q_DIMS}x{NQB * 16}xf16, %q_lay> -> {V16H}")
            return [f"%ql{r}f{c}" for c in range(8)]
        if QH:
            ty = ", ".join([V16H] * 8)
            e(f"  %qsel{r} = index.cmp eq, %hdu, %c{r} : index")
            e(f"  {', '.join(names)} = scf.if %qsel{r} -> ({ty}) {{")
            got = qload()
            e(f"    scf.yield {', '.join(got)} : {ty}")
            e("  } else {")
            other = prev if prev else ["%zq16"] * 8
            e(f"    scf.yield {', '.join(other)} : {ty}")
            e("  }")
            prev = names
        else:
            got = qload()
            names = got
        # Loom does not drain these LDS loads before the barrier below (no
        # lgkmcnt(0) ahead of s_barrier), and the prologue then stages K/V (or
        # the next Q half) over the Q stage: a fast wave overwrote Q rows a
        # slow wave was still reading (a few corrupted query lanes per run,
        # varying). An LDS store of a value built from both halves of every
        # fragment forces the drain first.
        acc = None
        for c in range(8):
            for j in (0, 8):
                e(f"  %qx{r}{c}_{j} = vector.extract {names[c]}[{j}] : {V16H} -> f16")
                if acc is None:
                    acc = f"%qx{r}{c}_{j}"
                else:
                    e(f"  %qy{r}{c}_{j} = scalar.addf {acc}, %qx{r}{c}_{j} : f16")
                    acc = f"%qy{r}{c}_{j}"
        e(f"  %qdrain{r} = vector.splat {acc} : {V8H}")
        e(f"  %qdr{r}_o = index.constant {DRAIN_OFF} : offset")
        e(f"  %qdr{r}_v = buffer.view %pool[%qdr{r}_o] : buffer -> view<{NT}x8xf16>")
        e(f"  vector.store %qdrain{r}, %qdr{r}_v[%tid, %c0] : {V8H}, view<{NT}x8xf16>")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    if QH:
        for c in range(8):
            e(f"  %qf{c} = vector.fragment<rhs> {prev[c]} shape [%k, %n] : {V16H}")
    else:
        for c in range(8):
            e(f"  %qf{c} = vector.fragment<rhs> %ql0f{c} shape [%k, %n] : {V16H}")

    # staging maps. K: item j = tid + 256nn, key = j/32, d8 = (j%32)*8 (one key
    # row per wave: coalesced global, contiguous LDS); LDS row 2*(key%8) +
    # key/8. V^T: lane = dim, LDS row (dim/16)*16 + 2*(dim%8) + (dim%16)/8.
    e("  %vdl = index.rem %vt, %c16 : index")
    e("  %vdb = index.sub %vt, %vdl : index")
    e("  %vdl8 = index.rem %vdl, %c8 : index")
    e("  %vdh = index.div %vdl, %c8 : index")
    e("  %vdr0 = index.mul %vdl8, %c2 : index")
    e("  %vdr1 = index.add %vdr0, %vdh : index")
    e("  %vrow = index.add %vdb, %vdr1 : index")

    for nn in range(KT // 8):
        e(f"  %ki{nn}c = index.constant {256 * nn} : index")
        e(f"  %ki{nn} = index.add %tid, %ki{nn}c : index")
        e(f"  %kk{nn} = index.div %ki{nn}, %c32 : index")
        e(f"  %kd{nn}a = index.rem %ki{nn}, %c32 : index")
        e(f"  %kd{nn} = index.mul %kd{nn}a, %c8 : index")
        e(f"  %kk{nn}l = index.rem %kk{nn}, %c8 : index")
        e(f"  %kk{nn}h = index.div %kk{nn}, %c8 : index")
        e(f"  %kk{nn}r0 = index.mul %kk{nn}l, %c2 : index")
        if KT == 16:
            e(f"  %kr{nn} = index.add %kk{nn}r0, %kk{nn}h : index")
        else:   # row 16*(key/16) + 2*(key%8) + (key%16)/8
            e(f"  %kk{nn}s = index.div %kk{nn}, %c16 : index")
            e(f"  %kk{nn}m = index.rem %kk{nn}, %c16 : index")
            e(f"  %kk{nn}h2 = index.div %kk{nn}m, %c8 : index")
            e(f"  %kk{nn}s16 = index.mul %kk{nn}s, %c16 : index")
            e(f"  %kr{nn}a = index.add %kk{nn}r0, %kk{nn}h2 : index")
            e(f"  %kr{nn} = index.add %kr{nn}a, %kk{nn}s16 : index")

    if KDEC:
        # thread = (key tid/16, 16-dim chunk tid%16); LDS row 2*(key%8) + key/8
        e("  %kdk = index.div %tid, %c16 : index")
        e("  %kdc = index.rem %tid, %c16 : index")
        e("  %kdkl = index.rem %kdk, %c8 : index")
        e("  %kdkh = index.div %kdk, %c8 : index")
        e("  %kdr0 = index.mul %kdkl, %c2 : index")
        e("  %kdr = index.add %kdr0, %kdkh : index")
        e("  %kdc16 = index.mul %kdc, %c16 : index")
        e("  %kdc16b = index.add %kdc16, %c8 : index")
        if K4:   # 2 dwords at head*32 + chunk*2; scale group head*8 + chunk/2
            e("  %kdh0 = index.mul %kvh, %c32 : index")
            e("  %kdc2 = index.mul %kdc, %c2 : index")
            e("  %kdga = index.add %kdh0, %kdc2 : index")
            e("  %kdh1 = index.mul %kvh, %c8 : index")
            e("  %kdg = index.div %kdc, %c2 : index")
            e("  %kdso = index.add %kdh1, %kdg : index")
            e("  %kdm = scalar.constant 62915520 : i32")     # 0x03c003c0
        else:      # 4 dwords at head*64 + chunk*4; scale head*2 + half (chunk/8)
            e("  %kdh0 = index.mul %kvh, %c64 : index")
            e("  %kdc4 = index.mul %kdc, %c4 : index")
            e("  %kdga = index.add %kdh0, %kdc4 : index")
            e("  %kdh1 = index.mul %kvh, %c2 : index")
            e("  %kdg = index.div %kdc, %c8 : index")
            e("  %kdso = index.add %kdh1, %kdg : index")
            e("  %kdm = scalar.constant 66847740 : i32")     # 0x03fc03fc
        e("  %kdgm = scalar.constant 1006648320 : i32")      # 0x3c003c00

    if NT > 256:
        # wave-uniform guards from the subgroup id (tid-based compares lower as
        # lane-masked regions, which the branch lowering rejects here)
        e("  %sgid = kernel.subgroup.id : index")
        e(f"  %vsg0 = index.constant {(NT - 256) // 32} : index")
        e("  %kstg = index.cmp ult, %sgid, %c8 : index")
        e("  %vstg = index.cmp uge, %sgid, %vsg0 : index")

    def guard(cond, ind, body_fn, n):
        """wave-uniform scf.if around staging loads (n V8H results, zeros else)"""
        if NT == 256:
            return body_fn()
        ty = ", ".join([V8H] * n)
        res = [f"%g{len(L)}_{i}" for i in range(n)]
        e(f"{ind}{', '.join(res)} = scf.if {cond} -> ({ty}) {{")
        names = body_fn()
        e(f"{ind}  scf.yield {', '.join(names)} : {ty}")
        e(f"{ind}}} else {{")
        e(f"{ind}  scf.yield {', '.join(['%zh8'] * n)} : {ty}")
        e(f"{ind}}}")
        return res

    def guard0(cond, ind, body_fn):
        if NT == 256:
            return body_fn()
        e(f"{ind}scf.if {cond} {{")
        body_fn()
        e(f"{ind}}}")

    # Loads stay unconditional (out-of-role waves load clamped, in-range
    # duplicates that hit L1): inside an scf.if the compiler drained vmcnt(0)
    # at the region exit, right after issue (GQA: 66.9 -> 69.8 M). Only the
    # LDS stores are guarded.
    def load_k(ks, p, ind):
        return load_k_(ks, p, ind)

    def load_v(ks, p, ind):
        return load_v_(ks, p, ind)

    def stage_k(ks, cur, p, ind):
        guard0("%kstg", ind, lambda: stage_k_(ks, cur, p, ind))

    def stage_v(cur, vb, p, ind):
        if VQ4 and NT > 256:
            cur = vq4_unpack(cur, p, ind)
            guard0("%vstg", ind, lambda: stage_v_(cur, vb, p, ind, unpacked=True))
            return
        if VQ8 and NT > 256:
            # unpack outside the guard: an scf.if drains vmcnt(0) at entry when
            # it reads loaded registers, which also waited for the K loads
            # issued with V (needed only in phase B): GQA int8 62.8 -> 68.2 M
            cur = vq8_unpack(cur, p, ind)
            guard0("%vstg", ind, lambda: stage_v_(cur, vb, p, ind, unpacked=True))
            return
        guard0("%vstg", ind, lambda: stage_v_(cur, vb, p, ind))

    def load_k_(ks, p, ind):
        if KDEC:
            nw = 2 if K4 else 4
            e(f"{ind}%{p}kp = index.add {ks}, %kdk : index")
            e(f"{ind}%{p}kpc = index.min %{p}kp, %cap_1 : index")
            e(f"{ind}%{p}kr = index.mul %{p}kpc, %c{128 if K4 else 256} : index")
            e(f"{ind}%{p}ka = index.add %{p}kr, %kdga : index")
            e(f"{ind}%{p}kv = vector.load %k_flat[%{p}ka] : view<[%kq32tot]xi32> -> vector<{nw}xi32>")
            e(f"{ind}%{p}sr = index.mul %{p}kpc, %c{32 if K4 else 8} : index")
            e(f"{ind}%{p}sa = index.add %{p}sr, %kdso : index")
            e(f"{ind}%{p}sv = view.load %ks_flat[%{p}sa] : view<[%kstot]xi32> -> i32")
            return [f"%{p}kv", f"%{p}sv"]
        names = []
        for nn in range(KT // 8):
            e(f"{ind}%{p}kp{nn} = index.add {ks}, %kk{nn} : index")
            e(f"{ind}%{p}kpc{nn} = index.min %{p}kp{nn}, %cap_1 : index")
            e(f"{ind}%{p}kr{nn} = index.mul %{p}kpc{nn}, %c1024 : index")
            e(f"{ind}%{p}kr{nn}b = index.add %{p}kr{nn}, %kvbase : index")
            e(f"{ind}%{p}ka{nn} = index.add %{p}kr{nn}b, %kd{nn} : index")
            e(f"{ind}%{p}kv{nn} = vector.load %k_flat[%{p}ka{nn}] : view<[%kvtot]xf16> -> {V8H}")
            names.append(f"%{p}kv{nn}")
        return names

    def load_v_(ks, p, ind):
        if VQ4:   # [kvh][tile][dim]: data 2 dwords, stats 1 dword
            e(f"{ind}%{p}vks = index.min {ks}, %vlast : index")
            e(f"{ind}%{p}vt16 = index.div %{p}vks, %c16 : index")
            e(f"{ind}%{p}vti0 = index.add %vhb4, %{p}vt16 : index")
            e(f"{ind}%{p}vti1 = index.mul %{p}vti0, %c256 : index")
            e(f"{ind}%{p}vti = index.add %{p}vti1, %vt : index")
            e(f"{ind}%{p}vda = index.mul %{p}vti, %c2 : index")
            e(f"{ind}%{p}vq = vector.load %v_flat[%{p}vda] : view<[%vq4tot]xi32> -> vector<2xi32>")
            e(f"{ind}%{p}vst = view.load %vs4_flat[%{p}vti] : view<[%vq4tot0]xi32> -> i32")
            return [f"%{p}vq", f"%{p}vst"]
        if VQ8:   # dword (vhl / 4) + tile * 1024: the lane's 16 keys
            e(f"{ind}%{p}vks = index.min {ks}, %vlast : index")
            e(f"{ind}%{p}vtb = index.mul %{p}vks, %c64 : index")
            e(f"{ind}%{p}va = index.add %vhl4, %{p}vtb : index")
            e(f"{ind}%{p}vq = vector.load %v_flat[%{p}va] : view<[%vq32tot]xi32> -> vector<4xi32>")
            return [f"%{p}vq"]
        if KT == 16:
            e(f"{ind}%{p}vks = index.min {ks}, %vlast : index")
            e(f"{ind}%{p}vtb = index.mul %{p}vks, %c256 : index")
            e(f"{ind}%{p}va0 = index.add %vhl, %{p}vtb : index")
            e(f"{ind}%{p}va1 = index.add %{p}va0, %c8 : index")
            return [e(f"{ind}%{p}vv{nn} = vector.load %v_flat[%{p}va{nn}] : view<[%vtot]xf16> -> {V8H}") or f"%{p}vv{nn}" for nn in range(2)]
        names = []
        for u in range(NSUB):   # 16-key V^T tiles ks + 16u
            e(f"{ind}%{p}vk{u}c = index.constant {16 * u} : index")
            e(f"{ind}%{p}vk{u} = index.add {ks}, %{p}vk{u}c : index")
            e(f"{ind}%{p}vks{u} = index.min %{p}vk{u}, %vlast : index")
            e(f"{ind}%{p}vtb{u} = index.mul %{p}vks{u}, %c256 : index")
            e(f"{ind}%{p}va{u}0 = index.add %vhl, %{p}vtb{u} : index")
            e(f"{ind}%{p}va{u}1 = index.add %{p}va{u}0, %c8 : index")
            for nn in range(2):
                e(f"{ind}%{p}vv{u}{nn} = vector.load %v_flat[%{p}va{u}{nn}] : view<[%vtot]xf16> -> {V8H}")
                names.append(f"%{p}vv{u}{nn}")
        return names

    def stage_k_(ks, cur, p, ind):
        """K tile at key ks (rows permuted, zero past ctx_end) to LDS."""
        if KDEC:
            nw = 2 if K4 else 4
            shifts = (6, 2, -2, -6) if K4 else (2, -6)
            e(f"{ind}%{p}qk = index.add {ks}, %kdk : index")
            e(f"{ind}%{p}ql = index.cmp ult, %{p}qk, %ctx_end : index")
            e(f"{ind}%{p}s1 = vector.from_elements {cur[1]} : vector<1xi32>")
            e(f"{ind}%{p}sh = vector.bitcast %{p}s1 : vector<1xi32> to vector<2xf16>")
            e(f"{ind}%{p}sS = vector.extract %{p}sh[0] : vector<2xf16> -> f16")
            e(f"{ind}%{p}sC = vector.extract %{p}sh[1] : vector<2xf16> -> f16")
            ws = []
            for d in range(nw):
                e(f"{ind}%{p}w{d} = vector.extract {cur[0]}[{d}] : vector<{nw}xi32> -> i32")
                for k, sh in enumerate(shifts):
                    op = "shli" if sh >= 0 else "shrui"
                    e(f"{ind}%{p}t{d}{k}c = scalar.constant {abs(sh)} : i32")
                    e(f"{ind}%{p}t{d}{k} = scalar.{op} %{p}w{d}, %{p}t{d}{k}c : i32")
                    e(f"{ind}%{p}m{d}{k} = scalar.andi %{p}t{d}{k}, %kdm : i32")
                    e(f"{ind}%{p}g{d}{k} = scalar.ori %{p}m{d}{k}, %kdgm : i32")
                    ws.append(f"%{p}g{d}{k}")
            # ONE 16-wide fma (see vq4_unpack: two 8-wide ones miscompiled)
            e(f"{ind}%{p}pk = vector.from_elements {', '.join(ws)} : vector<8xi32>")
            e(f"{ind}%{p}pf = vector.bitcast %{p}pk : vector<8xi32> to {V16H}")
            e(f"{ind}%{p}S16 = vector.splat %{p}sS : {V16H}")
            e(f"{ind}%{p}C16 = vector.splat %{p}sC : {V16H}")
            e(f"{ind}%{p}dk = vector.fmaf %{p}pf, %{p}S16, %{p}C16 : {V16H}")
            e(f"{ind}%{p}dz = scf.select %{p}ql, %{p}dk, %zq16 : {V16H}")
            e(f"{ind}%{p}d0 = vector.slice %{p}dz[0] : {V16H} -> {V8H}")
            e(f"{ind}%{p}d1 = vector.slice %{p}dz[8] : {V16H} -> {V8H}")
            e(f"{ind}vector.store %{p}d0, %k_view[%kdr, %kdc16] : {V8H}, view<{KT}x{KT_PITCH}xf16>")
            e(f"{ind}vector.store %{p}d1, %k_view[%kdr, %kdc16b] : {V8H}, view<{KT}x{KT_PITCH}xf16>")
            return
        for nn in range(KT // 8):
            e(f"{ind}%{p}sp{nn} = index.add {ks}, %kk{nn} : index")
            e(f"{ind}%{p}sl{nn} = index.cmp ult, %{p}sp{nn}, %ctx_end : index")
            e(f"{ind}%{p}sv{nn} = scf.select %{p}sl{nn}, {cur[nn]}, %zh8 : {V8H}")
            e(f"{ind}vector.store %{p}sv{nn}, %k_view[%kr{nn}, %kd{nn}] : {V8H}, view<{KT}x{KT_PITCH}xf16>")

    def vq8_unpack(cur, p, ind):
        # f16 1024 + u, exact: (w & 0x00ff00ff) | 0x64006400 per key pair
        # (yah_vq8 stores each dword's tokens as t0, t2, t1, t3)
        if True:
            # scalar ops with literal masks: vector splats of the constants
            # stayed live across the loop (GQA int8 at 176 VGPRs, past the
            # 170 that 3 x 12-wave workgroups per WGP need)
            for d in range(4):
                e(f"{ind}%{p}vw{d}x = vector.extract {cur[0]}[{d}] : vector<4xi32> -> i32")
                e(f"{ind}%{p}vA{d}m = scalar.andi %{p}vw{d}x, %vmsk : i32")
                e(f"{ind}%{p}vA{d} = scalar.ori %{p}vA{d}m, %vmag : i32")
                e(f"{ind}%{p}vB{d}s = scalar.shrui %{p}vw{d}x, %v8s : i32")
                e(f"{ind}%{p}vB{d}m = scalar.andi %{p}vB{d}s, %vmsk : i32")
                e(f"{ind}%{p}vB{d} = scalar.ori %{p}vB{d}m, %vmag : i32")
            hv = []
            for j in range(2):
                ws = [f"%{p}v{ab}{d}" for d in (2 * j, 2 * j + 1) for ab in ("A", "B")]
                e(f"{ind}%{p}vw{j} = vector.from_elements {', '.join(ws)} : vector<4xi32>")
                e(f"{ind}%{p}vh{j} = vector.bitcast %{p}vw{j} : vector<4xi32> to {V8H}")
                hv.append(f"%{p}vh{j}")
        return hv

    def vq4_unpack(cur, p, ind):
        # nibble pairs (2k, 2k+1) -> f16 1024 + u, then (f - 1031) * s + c
        e(f"{ind}%{p}vs1 = vector.from_elements {cur[1]} : vector<1xi32>")
        e(f"{ind}%{p}vsv = vector.bitcast %{p}vs1 : vector<1xi32> to vector<2xf16>")
        e(f"{ind}%{p}vss = vector.extract %{p}vsv[0] : vector<2xf16> -> f16")
        e(f"{ind}%{p}vsc = vector.extract %{p}vsv[1] : vector<2xf16> -> f16")
        ws = []
        for d in range(2):
            e(f"{ind}%{p}vw{d}x = vector.extract {cur[0]}[{d}] : vector<2xi32> -> i32")
            for k in range(4):
                # nibbles (4k, 16 + 4k) -> bits (6.., 22..): f16 1 + u/16
                sh = 6 - 4 * k
                op = "shli" if sh >= 0 else "shrui"
                e(f"{ind}%{p}vsh{d}{k}c = scalar.constant {abs(sh)} : i32")
                e(f"{ind}%{p}vsh{d}{k} = scalar.{op} %{p}vw{d}x, %{p}vsh{d}{k}c : i32")
                e(f"{ind}%{p}vmk{d}{k} = scalar.andi %{p}vsh{d}{k}, %v4m : i32")
                e(f"{ind}%{p}vmg{d}{k} = scalar.ori %{p}vmk{d}{k}, %v4g : i32")
                ws.append(f"%{p}vmg{d}{k}")
        # ONE 16-wide fma: with two 8-wide fmas Loom CSE'd their identical
        # addend splats and tied both in-place v_pk_fmac to one register, so
        # keys 8..15 got f * S + (keys 0..7's result) (micro-test unpk.loom)
        e(f"{ind}%{p}vpk = vector.from_elements {', '.join(ws)} : vector<8xi32>")
        e(f"{ind}%{p}vf = vector.bitcast %{p}vpk : vector<8xi32> to vector<16xf16>")
        e(f"{ind}%{p}vss16 = vector.splat %{p}vss : vector<16xf16>")
        e(f"{ind}%{p}vsc16 = vector.splat %{p}vsc : vector<16xf16>")
        e(f"{ind}%{p}vd = vector.fmaf %{p}vf, %{p}vss16, %{p}vsc16 : vector<16xf16>")
        hv = []
        for d in range(2):
            e(f"{ind}%{p}vd{d} = vector.slice %{p}vd[{8 * d}] : vector<16xf16> -> {V8H}")
            hv.append(f"%{p}vd{d}")
        return hv

    def stage_v_(cur, vb, p, ind, unpacked=False):
        e(f"{ind}%{p}vr = index.add {vb}, %vrow : index")
        if VQ4 and not unpacked:
            cur = vq4_unpack(cur, p, ind)
        if VQ8 and not unpacked:
            cur = vq8_unpack(cur, p, ind)
        for j, v in enumerate(cur):
            e(f"{ind}vector.store {v}, %v_view[%{p}vr, %c{8 * j}] : {V8H}, view<{256 * VBUFS}x{VT_PITCH}xf16>")

    # prologue: K(0), V(0) staged (V buffer 0)
    k0 = load_k("%c0", "p0", "  ")
    stage_k("%c0", k0, "p0s", "  ")
    if not VSB:
        v0 = load_v("%c0", "p2", "  ")
        stage_v(v0, "%c0", "p0v", "  ")
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")

    onames = [f"%o{f}" for f in range(8)]
    kvn = ["%kvc0", "%kvc1", "%kvc2", "%kvc3"]
    types = ", ".join([V8] * 8 + ["f32", "f32"])

    def body(tail):
        if VSB:
            e("    %vcur = index.add %c0, %c0 : index")
        else:
            e("    %t16 = index.div %ks_, %c16 : index")
            e("    %par = index.rem %t16, %c2 : index")
            e("    %npar = index.sub %c1, %par : index")
            e("    %vcur = index.mul %par, %c256 : index")
            e("    %vnxt = index.mul %npar, %c256 : index")
        e("    %ks16 = index.add %ks_, %c16 : index")
        nk = load_k("%ks16", "nk", "    ")
        nv = load_v("%ks_" if VSB else "%ks16", "nv", "    ")
        # ---- A: S^T partial over this wave's 128 dims
        # QK2: two independent accumulator chains (dims c even / odd), summed:
        # an 8-deep dependent WMMA chain waits on each MMA's latency
        e(f"    %zeros8s = vector.fragment<init> %zeros8 shape [%m, %n] : {V8}")
        accs = ["%zeros8s", "%zeros8s"]
        for c in range(8):
            e(f"    %kfc{c}c = index.constant {16 * c} : index")
            e(f"    %kfc{c} = index.add %hd128, %kfc{c}c : index")
            e(f"    %kf{c} = vector.fragment.load<lhs> %k_view[%c0, %kfc{c}] shape [%m, %k] : view<{KT}x{KT_PITCH}xf16> -> {V16H}")
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
            e(f"    vector.store {acc}, %s8_view[%tid, %c0] : {V8}, view<{NT}x8xf32>")
        else:
            e(f"    %sst0 = vector.slice {acc}[0] : {V8} -> {V4}")
            e(f"    %sst1 = vector.slice {acc}[4] : {V8} -> {V4}")
            e(f"    vector.store %sst0, %s_view[%tid, %c0] : {V4}, view<{2 * NT}x4xf32>")
            e(f"    vector.store %sst1, %s_view[%tid2, %c0] : {V4}, view<{2 * NT}x4xf32>")
        if VSB:   # V(ks): the previous tile's P.V finished at the last barrier
            if (VQ8 or VQ4) and VQFENCE:
                # keep the uint8 unpack after QK: hoisted between the QK MMAs it
                # waited for the V load ~4 MMAs after issue (GQA int8 68.0 M)
                e("    scf.schedule.fence")
            stage_v(nv, "%c0", "stv", "    ")
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        # ---- B: K(ks+16) (QK of this tile is done) and V(ks+16) to LDS
        stage_k("%ks16", nk, "st", "    ")
        if not VSB:
            stage_v(nv, "%vnxt", "stv", "    ")
        if os.environ.get("YAH_ATTN_FA_DBGBAR") == "1":
            e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        # ---- B: softmax
        if S8:
            e(f"    %spart = vector.load %s8_view[%ptid, %c0] : view<{NT}x8xf32> -> {V8}")
        else:
            e(f"    %spa = vector.load %s_view[%ptid, %c0] : view<{2 * NT}x4xf32> -> {V4}")
            e(f"    %spb = vector.load %s_view[%ptid2, %c0] : view<{2 * NT}x4xf32> -> {V4}")
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
            # yields alpha (1.0 when skipped: O * 1.0 and fma(sum, 1.0, part)
            # are exact, so THR=0 matches always rescaling bit for bit)
            e(f"      scf.yield {', '.join(f'%gos{f}' for f in range(8))}, %gmx, %galpha : {otypes}")
            e("    } else {")
            e(f"      scf.yield {', '.join(f'%o{f}' for f in range(8))}, %rmax, %one : {otypes}")
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
        if HIPNUM:
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
            e("    %nsum = scalar.fmaf %rsum, %rss, %psum : f32")
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
            e(f"    %vf{f} = vector.fragment.load<lhs> %v_view[%vfr{f}, %c0] shape [%m, %k] : view<{256 * VBUFS}x{VT_PITCH}xf16> -> {V16H}")
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


    def body32(tail):
        """32-key tile: two 16-key sub-tiles u, one barrier pair, one max /
        rescale decision and one score exchange (YAH_ATTN_FA_KT=32)."""
        e("    %ks32 = index.add %ks_, %c32 : index")
        nk = load_k("%ks32", "nk", "    ")
        nv = load_v("%ks_", "nv", "    ")
        e(f"    %zeros8s = vector.fragment<init> %zeros8 shape [%m, %n] : {V8}")
        accs = []
        for u in range(2):
            acc = "%zeros8s"
            for c in range(8):
                e(f"    %kfc{u}{c}c = index.constant {16 * c} : index")
                e(f"    %kfc{u}{c} = index.add %hd128, %kfc{u}{c}c : index")
                e(f"    %kf{u}{c} = vector.fragment.load<lhs> %k_view[%c{16 * u}, %kfc{u}{c}] shape [%m, %k] : view<{KT}x{KT_PITCH}xf16> -> {V16H}")
                e(f"    %sa{u}{c} = vector.mma %kf{u}{c}, %qf{c}, {acc} : {V16H}, {V16H}, {V8}")
                acc = f"%sa{u}{c}"
                if QKF and c + 1 < 8 and (c + 1) % QKF == 0:
                    e("    scf.schedule.fence")
            accs.append(acc)
        for u in range(2):
            for j in range(2):
                pl = 2 * u + j
                e(f"    %sst{u}{j} = vector.slice {accs[u]}[{4 * j}] : {V8} -> {V4}")
                e(f"    %spl{pl}c = index.constant {pl * NT} : index")
                e(f"    %spl{pl} = index.add %tid, %spl{pl}c : index")
                e(f"    %sppl{pl} = index.add %ptid, %spl{pl}c : index")
                e(f"    vector.store %sst{u}{j}, %s32_view[%spl{pl}, %c0] : {V4}, view<{4 * NT}x4xf32>")
        stage_v(nv, "%c0", "stv", "    ")
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        stage_k("%ks32", nk, "st", "    ")
        ss = []
        for u in range(2):
            e(f"    %spa{u} = vector.load %s32_view[%sppl{2 * u}, %c0] : view<{4 * NT}x4xf32> -> {V4}")
            e(f"    %spb{u} = vector.load %s32_view[%sppl{2 * u + 1}, %c0] : view<{4 * NT}x4xf32> -> {V4}")
            e(f"    %spart{u} = vector.concat<0> %spa{u}, %spb{u} : {V4}, {V4} -> {V8}")
            e(f"    %s{u} = vector.addf {accs[u]}, %spart{u} : {V8}")
            ss.append(f"%s{u}")
        if tail:
            e("    %lim0 = index.add %r_abs, %c1 : index")
            e("    %lim1 = index.min %lim0, %ctx_end : index")
            e("    %limi = index.cast %lim1 : index to i32")
            e("    %h8 = index.mul %half, %c8 : index")
            for u in range(2):
                e(f"    %kh{u}a = index.add %ks_, %h8 : index")
                e(f"    %kh{u} = index.add %kh{u}a, %c{16 * u} : index")
                e(f"    %khi{u} = index.cast %kh{u} : index to i32")
                e(f"    %limu{u} = scalar.subi %limi, %khi{u} : i32")
                els = []
                for i in range(8):
                    e(f"    %mi{u}{i} = scalar.constant {i} : i32")
                    e(f"    %mv{u}{i}a = scalar.cmpi slt, %mi{u}{i}, %limu{u} : i32")
                    e(f"    %mv{u}{i} = scalar.andi %mv{u}{i}a, %r_live_i1 : i1")
                    e(f"    %se{u}{i} = vector.extract %s{u}[{i}] : {V8} -> f32")
                    e(f"    %sm{u}{i} = scf.select %mv{u}{i}, %se{u}{i}, %ninf : f32")
                    els.append(f"%sm{u}{i}")
                e(f"    %smask{u} = vector.from_elements {', '.join(els)} : {V8}")
                ss[u] = f"%smask{u}"
        e(f"    %tmx0 = vector.reduce<maxnumf> {ss[0]}, %ninf : {V8}, f32")
        e(f"    %tmax0 = vector.reduce<maxnumf> {ss[1]}, %tmx0 : {V8}, f32")
        e("    %tmi = scalar.bitcast %tmax0 : f32 to i32")
        e("    %tmx, %tmv = kernel.subgroup.shuffle<xor> %tmi, %x16, %x32 : i32, i32, i32")
        e("    %tmf = scalar.bitcast %tmx : i32 to f32")
        e("    %tmax = scalar.maxnumf %tmax0, %tmf : f32")
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
        e(f"      scf.yield {', '.join(f'%gos{f}' for f in range(8))}, %gmx, %galpha : {otypes}")
        e("    } else {")
        e(f"      scf.yield {', '.join(f'%o{f}' for f in range(8))}, %rmax, %one : {otypes}")
        e("    }")
        e("    %nmax = scalar.maxnumf %rsm, %ninf : f32")
        e(f"    %nmax8 = vector.splat %nmax : {V8}")
        parts = []
        pbs = []
        for u in range(2):
            e(f"    %pd8{u} = vector.subf {ss[u]}, %nmax8 : {V8}")
            e(f"    %pl{u} = vector.mulf %pd8{u}, %log2e8 : {V8}")
            e(f"    %p{u} = vector.exp2f<afn> %pl{u} : {V8}")
            for g in range(2):
                cur = "%zero"
                for i in range(4):
                    e(f"    %pg{u}{g}_{i} = vector.extract %p{u}[{4 * g + i}] : {V8} -> f32")
                    e(f"    %pa{u}{g}_{i} = scalar.addf {cur}, %pg{u}{g}_{i} : f32")
                    cur = f"%pa{u}{g}_{i}"
            e(f"    %pab{u} = scalar.addf %pa{u}0_3, %pa{u}1_3 : f32")
            e(f"    %pabi{u} = scalar.bitcast %pab{u} : f32 to i32")
            e(f"    %pcdi{u}, %pcdv{u} = kernel.subgroup.shuffle<xor> %pabi{u}, %x16, %x32 : i32, i32, i32")
            e(f"    %pcd{u} = scalar.bitcast %pcdi{u} : i32 to f32")
            e(f"    %psum{u} = scalar.addf %pab{u}, %pcd{u} : f32")
            parts.append(f"%psum{u}")
            for i in range(8):
                e(f"    %pe{u}{i} = vector.extract %p{u}[{i}] : {V8} -> f32")
                e(f"    %pm{u}{i} = scalar.fmaf %pe{u}{i}, %one_o, %zero_o : f32")
                e(f"    %pt{u}{i} = scalar.fptrunc %pm{u}{i} : f32 to f16")
            e(f"    %ph{u} = vector.from_elements {', '.join(f'%pt{u}{i}' for i in range(8))} : {V8H}")
            e(f"    %phi{u} = vector.bitcast %ph{u} : {V8H} to {V4I}")
            e(f"    %ppi{u}, %ppv{u} = kernel.subgroup.shuffle<xor> %phi{u}, %x16, %x32 : {V4I}, i32, i32")
            e(f"    %pph{u} = vector.bitcast %ppi{u} : {V4I} to {V8H}")
            e(f"    %plo{u} = scf.select %h0, %ph{u}, %pph{u} : {V8H}")
            e(f"    %phh{u} = scf.select %h0, %pph{u}, %ph{u} : {V8H}")
            e(f"    %pbc{u} = vector.concat<0> %plo{u}, %phh{u} : {V8H}, {V8H} -> {V16H}")
            e(f"    %pb{u} = vector.fragment<rhs> %pbc{u} shape [%k, %n] : {V16H}")
            pbs.append(f"%pb{u}")
        e(f"    %nsum0 = scalar.fmaf %rsum, %rss, {parts[0]} : f32")
        e(f"    %nsum = scalar.addf %nsum0, {parts[1]} : f32")
        outs = []
        for f in range(8):
            e(f"    %vfr{f}c = index.constant {16 * f} : index")
            e(f"    %vfr{f} = index.add %hd128, %vfr{f}c : index")
            cur = f"%rso{f}"
            for u in range(2):
                e(f"    %vf{f}{u} = vector.fragment.load<lhs> %v_view[%vfr{f}, %c{16 * u}] shape [%m, %k] : view<{256 * VBUFS}x{VT_PITCH}xf16> -> {V16H}")
                e(f"    %nx{f}{u} = vector.mma %vf{f}{u}, {pbs[u]}, {cur} : {V16H}, {V16H}, {V8}")
                cur = f"%nx{f}{u}"
            if PVF and f + 1 < 8 and (f + 1) % PVF == 0:
                e("    scf.schedule.fence")
            outs.append(cur)
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
        e(f"  {', '.join(res)} = scf.for %ks_ = [{lo} to {hi} step %c{KT}]({init}) -> ({types}) {POL} {{")
        outs, nm, ns, nxt = (body32 if KT == 32 else body)(tail)
        e(f"    scf.yield {', '.join(outs)}, {nm}, {ns} : {types}")
        e("  }")

    r1 = [f"%of{i}" for i in range(8)] + ["%fmax", "%fsum"]
    loop("%c0", "%split", ["%oinit"] * 8, "%ninf", "%zero", [], os.environ.get("YAH_ATTN_FA_MASKALL") == "1", r1)
    r2 = [f"%og{i}" for i in range(8)] + ["%gmax", "%gsum"]
    loop("%split", "%max_vis", r1[:8], "%fmax", "%fsum", [], True, r2)

    # ---- epilogue: o / sum * sigmoid(gate); lane writes dims 8h..8h+7 of each 16
    if HIPNUM:
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
    if VQ8:   # channel kvh * 256 + hd * 128 + 8h (+ 16 f)
        e("  %vsc0 = index.mul %kvh, %c256 : index")
        e("  %vsc1 = index.add %vsc0, %hd128 : index")
        e("  %vsch = index.add %vsc1, %eh8 : index")
        e(f"  %c1152v = vector.splat %c1152 : {V8}")
    def gate_loads(f):
        e(f"  %eo{f}c = index.constant {16 * f} : index")
        e(f"  %eo{f} = index.add %eoff, %eo{f}c : index")
        e(f"  %eo{f}b = index.add %eo{f}, %c4 : index")
        e(f"  %eg{f}a = vector.load %g_flat[%eo{f}] : view<[%qtot]xf32> -> {V4}")
        e(f"  %eg{f}b = vector.load %g_flat[%eo{f}b] : view<[%qtot]xf32> -> {V4}")
    if not VQ8:   # all gate loads together (latency); VQ8: per fragment (VGPRs)
        for f in range(8):
            gate_loads(f)
    for f in range(8):
        if VQ8:
            gate_loads(f)
        e(f"  %eg{f} = vector.concat<0> %eg{f}a, %eg{f}b : {V4}, {V4} -> {V8}")
        e(f"  %egn{f} = vector.negf %eg{f} : {V8}")
        e(f"  %egm{f} = vector.mulf %egn{f}, %nlog2e8 : {V8}")
        e(f"  %egx{f} = vector.exp2f<afn> %egm{f} : {V8}")
        e(f"  %egd{f} = vector.addf %ones8, %egx{f} : {V8}")
        e(f"  %egr{f} = vector.divf %ones8, %egd{f} : {V8}")
        if HIPNUM:
            e(f"  %eod{f} = vector.divf %og{f}, %lsum8 : {V8}")
            if VQ8:   # V = (u - 128) * s + c per channel: o / l * s + c
                e(f"  %evc{f}c = index.constant {16 * f} : index")
                e(f"  %evc{f} = index.add %vsch, %evc{f}c : index")
                e(f"  %evs{f}i = index.add %evc{f}, %c1024 : index")
                e(f"  %evcn{f} = vector.load %vs_flat[%evc{f}] : view<2048xf32> -> {V8}")
                e(f"  %evsc{f} = vector.load %vs_flat[%evs{f}i] : view<2048xf32> -> {V8}")
                # staged values are 1152 + (u - 128): centre c - 1152 s
                e(f"  %evcc{f} = vector.fmaf %evsc{f}, %c1152v, %evcn{f} : {V8}")
                e(f"  %eod{f}v = vector.fmaf %eod{f}, %evsc{f}, %evcc{f} : {V8}")
                e(f"  %eov{f} = scf.select %lpos, %eod{f}v, %zeros8 : {V8}")
            else:
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
        if VQ8 and f < 7:
            # the scheduler hoisted every fragment's gate + scale/centre loads
            # (192 VGPRs) to the epilogue top: 160 -> 240 VGPRs kernel-wide
            e("  scf.schedule.fence")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "yah_attn_fa.loom"
    open(out, "w").write(gen())
    print(out)


if __name__ == "__main__":
    main()
