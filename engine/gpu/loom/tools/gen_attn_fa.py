#!/usr/bin/env python3
"""Generate yah_attn_wmma, the FlashAttention-style causal prefill attention (softmax in registers), and yah_transpose_v16.

usage: gen_attn_fa.py [out.loom]
       gen_attn_fa.py vtrans [out.loom]

Not HIP's arithmetic order: checked by the T1 numerics gate, not by md5.

Bindings: query and gate (f32 [token][6144]), key_cache, value_cache, output (f16 [token][6144]), lse (not read),
then kscale (quantized K), vstat4 (quantized V) and ptab (paged caches) when those modes are on.
Grid: (query blocks of 32 tokens, head pairs); 256 threads = 8 waves.
One workgroup = 32 query tokens x 2 query heads of one GQA group (4 query blocks of 16).
Wave w owns query block w/2 and head-dim half w%2.
  S^T = K Q^T: K as A (rows = keys), Q^T as B (columns = queries), 8 WMMAs over the wave's 128 dims.
               The wave pair adds partials through a private LDS slot (s = own + partner: f32 add commutes, both waves agree).
  softmax:     lane (q = lane%16, h = lane/16) holds keys 8h..8h+7 of query q.
               K rows are permuted at staging (LDS row 2i+h holds key 8h+i), so the row max is an in-lane max + one xor-16 shuffle.
  P^T as B:    f16(p) of both halves, one xor-16 shuffle of 4 dwords, concat.
  O^T += V^T P^T: V^T as A (rows = dims), permuted at staging so element i of lane half h is dim 8h+i (contiguous output per lane), 8 WMMAs.
K is staged one tile ahead; V(i) is loaded at the top of phase A and staged at its end.
Two barriers per tile: A = QK, S store, stage V; B = stage next K, softmax, P.V.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_kvq  # noqa: E402

V4 = "vector<4xf32>"
V8 = "vector<8xf32>"
V8H = "vector<8xf16>"
V16H = "vector<16xf16>"
V4I = "vector<4xi32>"

# Largest prompt the token-count facts admit; emit_prefill_pp.py sets it to B when B > 2048.
MAX_TOKENS = 2048
# A schedule fence after every QKF QK MMAs caps the K fragments in flight.
QKF = 1    # 1: one K fragment in flight, which with QDIRECT gets the 2 x 32 shape to 192 VGPRs (4 workgroups per WGP)
# PIPE2 (FA3's 2-stage pipeline, same arithmetic per element): the QK of tile j+1 runs beside the softmax and P.V of tile j,
# so a wave's own MMAs fill its softmax's dependent stalls. Region 1: partner S(j), softmax(j), [QK(j+1) from the other
# K buffer, loads of V(j+1), K(j+2)], P.V(j); barrier; region 2: store S(j+1), stage V(j+1), stage K(j+2); barrier.
# K is double-buffered (K2_OFF, inside the prologue Q stage's footprint).
PIPE2 = False
# VSWZ: V^T staging thread t (of a group of 16) stages dim (t%2)*8 + t/2 into LDS row t, instead of dim t into row
# 2*(t%8) + t/8: the same LDS contents, but each 8-lane pass of the b128 stores writes 8 consecutive 48-B rows (8
# disjoint 4-bank groups) instead of rows 0, 2, .. 14 (two lanes per bank group).
VSWZ = True
VSWZ_T = True    # VSWZ also in TILED_OUT builds: there it costs the 4th workgroup (236 VGPRs) but measured 72.3 vs 77.9 M (pp2048)
VSWZR = True     # VSWZ: recompute the staged dim's global offset per tile instead of keeping it live through the loop
VSWZF = 0        # formulation of the swizzled dim (same values; the register allocator is sensitive to it)
# QDIRECT (Q16 only): each lane loads its Q^T fragments straight from the f16 query (lane = query row, 16 contiguous dims
# = one 32-B load per fragment), no LDS Q stage: no prologue stores (4-way bank conflicts), barriers or drain store, and
# the LDS high-water mark drops to the loop's (K, V, S). Same values (f16(q / 16), zero for rows past the chunk).
QDIRECT = None    # None: follow Q16
# QNORM (QDIRECT only): the query arrives raw (the f32 q projection) and the prologue does RoPE's Q work for its rows:
# the head RMS norm (the sum of squares in yah_fused_qk_rope_batched's order: a balanced tree over the 256 dims), x inv
# x weight, NeoX RoPE on dims < 64 (pairs p, p + 32), f16(q / 16). Lane half h takes the 16-dim fragments g = h (mod 2)
# of its row (both 128-dim halves for the sum, its own half for the values: 16 RoPE pairs per lane half) and trades
# subtree sums and finished fragments with its xor-16 partner. Same bits as RoPE + rope_q16; RoPE then does K / V only.
QNORM = False
# QG16: q and gate arrive as f16 (the kqg GEMM's f16 epilogue, emit_prefill_pp.QG16); with QNORM the prologue widens q
QG16 = False
QX_OFF = 0          # QNORM: LDS offset of the sums slot (set by gen)
QW_OFF = 0          # R2P_QPF: LDS offset of the norm weights (set by gen)
# PIPE2_EARLY: the QK of tile j+1 goes in before the softmax's rescale branch (its loads stay after it: a branch join
# drains outstanding global loads)
PIPE2_EARLY = False
# The O rescale is skipped when no row max of the wave grew (FA4 conditional rescale).
# The skip path yields alpha = 1.0, so the result has the same bits as always rescaling.
# Rounding follows HIP where it is cheap, so the output stays near the HIP-order golden (T1):
# p = exp2((s - m) * log2e) instead of one fma, the row sum grouped as HIP does per tile
# (4-key partials, (a + b) + (c + d) across the lane halves, sum = fma(sum, prior, part)), and o / sum by IEEE division.
# Workgroup order: head pair fastest, so the 3 pairs of a KV head (GQA 6) run side by side on the same K/V tiles (L2 hits).
# Longest query blocks go first (LPT).

# LDS (bytes). The Q stage (prologue only) aliases the rest.
# Pitches are conflict-free under the 8-lane passes of b128 accesses (128 B): K rows 528 B, V rows 48 B.
# 32-B V rows put lanes r and r+4 on the same banks.
KT_PITCH = 264                       # K: 16 keys x 256 dims (+8 pad)
VT_PITCH = 24                        # V^T: 256 dims x 16 keys (+8 pad)
# K(i+1) is loaded at the top of phase A and staged in phase B of the same tile.
# So nothing in flight crosses the loop back edge, where the compiler drains vmcnt(0).
K_OFF = 0                            # 16 x 264 x 2
V_OFF = 16 * 264 * 2
HPW, QT = 2, 32                      # query heads, query tokens per workgroup
NQB = HPW * QT // 16                 # query blocks (wave pairs)
NT = 64 * NQB                        # threads
KT = 16                              # keys per tile
# VQ8 (V half of kv8a16): V^T as 255-level bytes per channel per 16-key tile, f16 (S, C') = (256 s, c - 384 s) (yah_vq8).
# Staging builds f16 1 + u/256 (0x3c00 | u << 2) with masks, then one packed fma f * S + C'.
# Ranges are per tile, so they stream with chunked prefill (no prompt-wide statistics).
VQ8 = gen_kvq.kv_bits()[1] == 8
# VQ4 (V half of the kv4 configs): V^T as 15-level nibbles per channel per 16-key tile, f16 (S, C') = (16 s, c - 23 s) (yah_vq4).
# Staging builds f16 1 + u/16 (0x3c00 | u << 6) with masks, then one packed fma f * S + C'. P.V stays f16.
VQ4 = gen_kvq.kv_bits()[1] == 4
# Quantized K (engine/run/kvq/README.md): the K cache is centred by its per-channel prompt mean (yah_kmean).
# Staging decodes it to f16, and QK^T runs the f16 path with an unquantized f16 Q.
#   K4 (kv4a16): H256 (k - m) as asymmetric int4 per 32-dim group (yah_kq4).
#     Decoded as (1 + u/16) * 16 s + lo - 16 s, one packed fma per pair; Q is rotated by the same H256 at staging.
#   K8 (kv8a16): int8 per token half (yah_kq8), decoded as (1 + u/256) * 256 s - 384 s.
K4 = gen_kvq.kv_bits()[0] == 4
K8 = gen_kvq.kv_bits()[0] == 8
KDEC = K4 or K8                       # K decoded to f16 at staging
# PAGED: the K / V caches are paged in 256-token pages; ptab[logical page] = physical page.
# K rows (and K scales): row' = ptab[row / 256] * 256 + row % 256.
# V^T tiles (and V stats): tile' = ptab[tile / 16] * 16 + tile % 16. The layouts do not change.
# Every 16-key tile lies in one page: one uniform table load per K tile and per V tile.
# emit_prefill_pp.py clears it when the context is not a multiple of 256.
PAGED = True
# TILED_OUT: store the output fragment-major for an afrag o-projection (gen_gemm_tile Tile.atiled over K = 6144): 16 x 16
# tiles (token / 16, column / 16) of 256 halves, column fastest. A lane's 8 halves sit inside one tile row.
TILED_OUT = False
# GATE_EACH: the epilogue loads each fragment's gate right before its use instead of all 8 up front (64 fewer live
# registers at the epilogue peak; the paged fragment-major build otherwise allocates 236 VGPRs)
GATE_EACH = False
# GATE_FENCE: every gate address first, a schedule fence, then the loads: without it the allocator recycles each address
# register right after its load and Loom waits for that load (vmcnt) before the overwrite, serializing the 16 loads
GATE_FENCE = True
# DIAG_MASK: the causal / context mask (8 compares, ands and selects per lane) under a wave-uniform branch taken only when
# some lane of the wave has a key past its limit or a dead row (the diagonal and last tiles). Same values: an unmasked
# tile's selects pass S through. The branch sits after the K staging and the S partner loads (the mask needs them), so
# its join drains nothing.
DIAG_MASK = True
DIAG_SWAP = True    # the mask branch as 'all lanes unmasked' with the pass-through arm first (layout order)
KCLAMP_S = True     # K tile rows clamped by one scalar min of the tile start (cap - 16), not a per-lane min
# CTX16: start_pos and token_count are multiples of 16 (emit_prefill_pp sets it), so a tile start ks < max_vis <= ctx_end
# has all 16 keys before ctx_end: the K staging's zero-past-ctx_end selects are dead
CTX16 = False
# MAXNN: the row max without the -FLT_MAX seed and with nnan (no NaN reaches it; without nnan LLVM canonicalizes both
# operands of every maxnum, v_max x, x); the seed max folds (ninf: both arms yield a value >= -FLT_MAX). Same values.
MAXNN = False
SUM0 = False        # the row sum's 4-key partials start from the first p, not 0 + p (p >= 0: same values)
# PV_FENCE: a schedule fence before the P.V: the first V fragments' loads hoisted above the softmax landed in split
# registers and cost 16 v_mov per tile into the MMA operand tuples. Same values
PV_FENCE = False
# PBP: P^T's two halves by index shuffles (lane gets lane & 15 / lane | 16: ds_bpermute) instead of an xor-16 shuffle
# and two per-lane selects. Same values; lost (+1.2% cycles: the bpermute latency).
PBP = False
# EPI_FAST: the epilogue's o / sum as Markstein's correction of o * RN(1 / sum) (q = o r; q + (o - q sum) r, correctly
# rounded when r is) and the sigmoid's 1 / (1 + e) as rcp + one Newton step (gen_gemm_tile.recip1), instead of two IEEE
# divisions per element
EPI_FAST = False
# PRO_FAST: the prologue's divisions without IEEE division sequences, same bits: x / 64 and x / 256 as x * 2^-k (exact
# for normal results), RoPE's 1 / theta^e (in [1, 1e14]) as EPI_FAST's rcp + one Newton step. With R2P the epilogue
# also drops the per-element sum > 0 select: every stored row is live and sees its own key, so its sum is >= 1
PRO_FAST = False
# R2P_EPI_LITE: the R2P epilogue (EPI_FAST, PRO_FAST) as o * RN(1 / sum) * rcp(1 + e): no Newton step, no Markstein
# correction (within ~1 f32 ulp before the f16 store; not bit-exact)
R2P_EPI_LITE = os.environ.get("R2P_EPI_LITE", "0") == "1"
KVMAJOR = False     # workgroup order KV head slowest (lost: +67% at 96K, see docs/results.md)
# ATTN_KVG=g (diag, PERSIST_DYN): the queue deals units g KV heads at a time (heads interleaved, then pairs, then query
# blocks), so the 40 slots share fewer K / V streams in L2; per-unit math unchanged (bit-identical)
KVG = int(os.environ.get("ATTN_KVG", "0"))
# ATTN_KRAW (R2P, paged f16 pools, kv8 sets): the K pool holds the raw codes u - 128 (exact f16 integers: few mantissa
# bits, cheaper WMMA operands) and binding kscale their scales s, f16 [KV head][128-dim half][pool row]. Each QK wave's
# partials span one half: the P.V waves stage the tile's scales (f32) into LDS beside K; after its chain the wave scales
# the partner's partial by its half's s before the exchange and forms S = fma(own, s, partner's) after it. Exact
# products; not bit-identical to the f16 s (u - 128) pool (no rounding of K).
KRAW = os.environ.get("ATTN_KRAW", "0") == "1"
# ATTN_KHALF=1 (with ATTN_KRAW=1): kdeq8 writes each row's larger-scale half as the exact codes u - 128 and the other
# half as f16((u - 128) s / s_big), kscale s_big (both halves); attention forms S = (own + partner's) s_big after
# barrier A (no exchange-side work) and folds the old (S - m) r log2e into fma(S, r log2e, -m r log2e).
KH = os.environ.get("ATTN_KHALF", "0") == "1"
assert not KH or KRAW
_krs = os.environ.get("ATTN_KRAW_STEP", "")   # KRAW: the QK chain step after which the scales are read ("" = after it)
KRAW_RD_STEP = int(_krs) if _krs else None
# R2P: reuse-2 role split (8 waves = 2 groups of 2 query blocks x (2 QK waves + 2 P.V waves), each group = the 2 heads of
# one 16-row query half). A QK wave holds Q of both heads for its head-dim half: each K fragment feeds 2 MMAs, the
# other head's partial goes to its QK partner through LDS, and it runs the softmax of its own head once (P^T / alpha to
# LDS, 2 slots by tile parity). A P.V wave holds O of both heads for its head-dim half: each V^T fragment feeds 2 MMAs,
# one tile behind, beside the softmax. Per tile: phase 1 QK || the P.V waves stage V(ks - 16) and load K(ks + 16),
# barrier, phase 2 softmax || K(ks + 16) stored + P.V(ks - 16), barrier. Half the LDS fragment reads of the head_dim
# split, no duplicated softmax, 191 VGPRs (4 workgroups per WGP). The main kernel's arithmetic (bit-identical).
R2P = False
R2P_QKF = int(os.environ.get("R2P_QKF", "1"))   # R2P: a schedule fence every R2P_QKF K fragments (2 MMAs each; 2 -> 198 VGPRs, 3 WGs: +6.8%)
R2P_PVF = int(os.environ.get("R2P_PVF", "1"))   # R2P: the same for the V^T fragments
# R2P_PVDB: the P.V wave's V^T fragments double buffered in a fenced order (load f + 1, then fragment f's MMAs), so
# a fragment load never waits for the MMAs reading the previous one (the ATT trace: a single buffer leaves each wave
# ~35 ticks of register reuse + a 42-tick LDS round trip between MMA pairs). Same arithmetic
R2P_PVDB = os.environ.get("R2P_PVDB", "0") == "1"
# R2P_KPOS: where the P.V wave stores K(ks + 16) in phase B: "p" (default) after its P^T loads, "" right after barrier A
# (ahead of them in the in-order LDS queue: the first P.V MMA waited for the K stores), "0".."7" after V^T fragment f's
# MMAs (from "1": 203 VGPRs). "p": busiest SQ +0.6 / +0.1 / +0.4..0.9 at 64K / 8K / 0, bit-identical; with it the
# softmax shows in the loop (nosoft +2.0 at 8K and 64K, was ~0)
R2P_KPOS = os.environ.get("R2P_KPOS", "p")
# R2P_PRIO: the QK waves' issue priority (kernel.priority, 1..3) from barrier A to barrier B: the softmax chain is the
# tile's critical path and loses issue slots to other waves' MMA bursts (ATT trace). 0: off
R2P_PRIO = int(os.environ.get("R2P_PRIO", "0"))
# R2P_PRIOQ: the same for the QK waves' MMA section (tile start to barrier A): the per-workgroup critical path is
# QK MMAs -> softmax, and the P.V waves' MMAs (beside the softmax, off it) compete for the same WMMA pipe
R2P_PRIOQ = int(os.environ.get("R2P_PRIOQ", "0"))
# R2P_TPRIO: issue priority N (1..3) from the epilogue through the next unit's prologue, 0 from the tile loop on: a
# transitioning wave's dependent VALU chain otherwise waits behind the other workgroups' WMMAs on its SIMD (ATT trace:
# ~20 K ticks of epilogue for ~300 instructions). 0: off
R2P_TPRIO = int(os.environ.get("R2P_TPRIO", "0"))
R2P_PSB = True      # R2P: P^T / alpha slot stores under a parity branch
# R2P_K2: a second K slot (tile n in slot n % 2), so the P.V waves stage K(n + 1) before barrier A instead of after it.
# Room for it (31744 B / WG, 4 WG / WGP): V^T unpadded as two 4 KB planes of 16-B rows (keys 0..7, keys 8..15: any 8
# consecutive rows hit 8 distinct 16-B bank groups, b128 reads and stores conflict-free, one lane address each), and
# the S partials' plane 0 in tile n's P^T slot (each QK lane reads its partner's partial from exactly the 16 B its P^T
# store overwrites later), plane 1 in its own 2 KB. 8K +0.3, 64K +0.5 (bit-identical).
R2P_K2 = os.environ.get("R2P_K2", "1") == "1"
# R2P_UNR: the split role loops unrolled by 2 (tile parity a constant in each body: no slot branches or slot math)
R2P_UNR = {"0": "", "1": "qp"}.get(os.environ.get("R2P_UNR", "0"), os.environ.get("R2P_UNR", "0"))   # q / p: one role
# R2P_VPAD: N opaque lane values (queue zero words) live through the unit, added to the epilogue's row-sum index: N
# VGPRs, to move the register file's fragment base by N (WMMA operand banks)
R2P_VPAD = int(os.environ.get("R2P_VPAD", "0"))
R2P_PVB = os.environ.get("R2P_PVB", "1") == "1"   # R2P, P.V: P^T / alpha loads through byte views, loop-invariant lane
                    # byte address + the slot's uniform byte offset (1 v_add per base; a row index costs 3 per load)
R2P_PEEL = os.environ.get("R2P_PEEL", "1") == "1"   # R2P: tile 0 peeled off the loop without the P.V of tile -1
# R2P_SPLIT: one loop per role under a top-level role branch (same trip count, 2 barriers per tile each), instead of
# one loop with a role branch per tile: the QK loop carries only max / sum (Q is loop-invariant), the P.V loop only
# the O banks; no per-tile role join. The roles join once per unit, before the last tile's P.V
R2P_SPLIT = os.environ.get("R2P_SPLIT", "1") == "1"
# GSIG: the gate arrives as sigmoid(gate), applied by the kqg GEMM's epilogue (gen_gemm_tile KQG_GSIG; the emitter
# sets both): "u16" round(s * 65535) as unorm16 bits in the f16 gate buffer (gate N: KL 1.07e-6, half the gate bytes, no
# exp / reciprocal here; depth 0 +2.2% of floor), "16" f16 (fails gate N), "32" f32 (bit-identical). Same offsets as
# the f32 gate, in f16 elements for u16 / 16
GSIG = os.environ.get("GSIG", "")
GSIG16 = GSIG in ("16", "u16")
R2P_EPIG = os.environ.get("R2P_EPIG", "1") == "1"   # R2P: epilogue math only in the storing (P.V) waves
# R2P_LASTB: the last tile's P.V in the P.V waves only (the QK waves' 16 MMAs of it were discarded, beside the real ones)
R2P_LASTB = os.environ.get("R2P_LASTB", "1") == "1"
# R2P_STEER (PERSIST_DYN R2P, register allocation steering, same arithmetic): the next unit's queue atomic before the
# tile loops (its fetch after them) and one VGPR word carried through the unit loop (tid & an opaque zero, added to the
# next unit's Q row). With R2P_LASTB (all three: no pair or single one is enough) Loom's allocation gives both tile
# loops' WMMAs distinct VGPR banks ({A, B, C, C+1} % 4) in every engine chunk HAL surveyed (249 of 249 at contexts
# 2K..256K, floor/c/rsurvey.py + cover.py; the old default shared banks in ~95% of them: -0.5..-1.3 pts at 4K..64K).
# Steering is by survey, not by design: re-run it after any change to this kernel
R2P_STEER = os.environ.get("R2P_STEER", "1") == "1"
# R2P_GATEPF=D (R2P_EPIG): the whole epilogue under one store guard and each gate fragment loaded D fragments ahead of
# its use (a schedule fence after it). With a guard per fragment, each join drains vmcnt(0): the 16 gate loads ran one
# after another, ~14 K ticks per unit of the P.V waves' epilogue (ATT trace). 0: a guard per fragment. Same arithmetic
R2P_GATEPF = int(os.environ.get("R2P_GATEPF", "1"))
# R2P_DIAG (diagnostics only, wrong results, never deployed): comma list of LDS ablations: nox (no S partial exchange,
# s = own + other head's partial), halfk / halfv (odd K / V^T fragments reuse the even one's registers); nobar (no
# barriers in the R2P tile loop: races, no waits); nosoft (p = s, no subtract / scale / exp);
# nosum (no row sum); nobara / nobarb (only barrier A / B); chain4 (4 QK accumulator chains of 4); occ3 (LDS
# floor for 3 workgroups per WGP)
R2P_DIAG = set(filter(None, os.environ.get("R2P_DIAG", "").split(",")))
# R2P_QPF (R2P_K2 QNORM PERSIST_DYN STEER): the next unit's Q prologue in the QK waves' tail, under the P.V waves' last
# P.V and epilogue (the QK waves sat at the next unit's first barrier: ~14 % of the depth-0 wave time was the Q prologue,
# ATT trace). The tail stages the next unit's K(0), loads both head blocks' raw f32 Q (this wave's head-dim half) and
# stores their sums of squares (K slot 1: idle until the P.V waves stage K(1) after tile 0's barrier A); the raw Q rides
# the unit loop. The unit top: phase 2 of both blocks in the QK waves only (own sums read back), no halves exchange.
R2P_QPF = os.environ.get("R2P_QPF", "0") == "1"
# (R2P_QPF moves 64 VGPRs of Q state across the unit loop: 200..240 VGPRs, spills; the tile loops' own peak is ~191)
# R2P_QTOUCH (QNORM PERSIST_DYN STEER): in the tail, the QK waves (idle: the P.V waves run the last P.V and epilogue)
# touch every 128-B line of the next unit's f32 Q rows (8 b32 loads a lane, folded into one LDS word) so the unit
# top's Q loads hit L2: ~8 K ticks of their wait per unit were DRAM misses (ATT trace, depth 0)
R2P_QTOUCH = os.environ.get("R2P_QTOUCH", "0") == "1"
# R2P_SUMLATE: the QK wave's row sum after its P^T store (the sum is next tile's; the store feeds barrier B)
R2P_SUMLATE = os.environ.get("R2P_SUMLATE", "0") == "1"
# R2P_PST4: P^T stored as 4 dwords (f16 pairs): the converts' halves need no copy into a b128 tuple (4 v_mov + nop)
R2P_PST4 = os.environ.get("R2P_PST4", "0") == "1"
# R2P_QSPLIT (QNORM): the kqg GEMM did the Q head norm's per-element part. The q buffer (f32-sized) holds f16 q at the
# f32 q's indices (= f16(q w / 16); dims 0..63 unused), the raw f32 rotary dims [token][head][64] at byte B * 12288,
# and f32 sums of q^2 per 16-dim group [token][head][16] at B * 18432 (ends at B * 19968 <= the f32 q's B * 24576):
# one f16 rounding per element, as the attention's own. The attention RoPEs dims 0..63 (f32), and the norm's 1 / rms scales S (r log2e in place of log2e).
R2P_QSPLIT = os.environ.get("R2P_QSPLIT", "0") == "1"
# R2P_QROT16 (R2P_QSPLIT): RoPE from the f16 halves (f16(q w / 16) x 16) instead of the raw f32 rotary dims and the
# norm weights: 16 of the unit's 52 KB of Q reads gone, a second rounding on the 64 rotary dims (gate N)
R2P_QROT16 = os.environ.get("R2P_QROT16", "0") == "1"
R2P_QPFW = os.environ.get("R2P_QPFW", "0") == "1"   # R2P_QPF: the norm weights from an LDS copy (+1 KB: a 32 KB pool)
# R2P_QFOLD (R2P_QSPLIT): the traded Q fragments are scaled once per unit by f16(r log2e), so S is in log2 units and
# the softmax drops its per-element scale (8 muls / tile) and the scalar log2e muls (gate N: one more f16 rounding of Q)
R2P_QFOLD = int(os.environ.get("R2P_QFOLD", "0"))   # 1: at the dead-row select; 2: each wave scales its own halves
# before the trade (the RoPE ones in f32, folded into the 0.0625)
R2P_LAZY = float(os.environ.get("R2P_LAZY", "5"))   # R2P: lazy rescale threshold in nats (0: exact, bit-identical)
QB_ZIGZAG = False   # query blocks launch long, short, long, ... instead of longest first
# PERSIST = P > 0: P workgroups (one per slot: 20 WGPs x 4) each run a list of units (query block, head pair) in a
# loop: the units sorted longest first (QB_ZIGZAG: in its long, short, ... order), dealt round by round, the
# direction alternating (snake). Per-unit code as before (bit-identical); the last partial round of whole units no
# longer idles slots behind a launch order.
PERSIST = 0
# PERSIST_REV (with PERSIST): slots 20-39 and 60-79 (WGP s % 20's second and fourth) run their snake rounds last to
# first: the same units per slot (the snake's balance), but every WGP runs short units beside long ones throughout,
# so the short units' Q / gate / O traffic overlaps compute instead of forming a DRAM-bound tail
PERSIST_REV = False
# PERSIST_DYN (with PERSIST): binding pqueue (after the others), PQ_WORDS i32 (word 0 the counter), zero at launch: each
# slot takes its next unit from it with an atomic add, after its last unit one fetch past the end, so a launch makes
# exactly units + P fetches; the one that returns units + P - 1 is the last and resets the counter to zero. Units go
# long, short, long, ... (QB_ZIGZAG's order) for start_pos < PDYN_ZZ_BELOW, else longest first (busiest SQ, QSPLIT:
# zigzag wins at 2K..6K by 2.0..0.8 points, longest first at 8K / 12K by 1.0 / 1.25).
PERSIST_DYN = False
PQ_WORDS = 264      # PERSIST_DYN: the counter and 256 lanes' zero words (1056 B)
# ATTN_KSPLIT=K (R2P_SPLIT PERSIST_DYN): the last partial round's units (units % P, the tail) run as K key pieces each
# (32-key aligned, piece-major after the whole units in the queue), so every slot ends with the same work. Binding
# ksplit (after pqueue): 64 monotonic i32 counters, then per piece and P.V lane 132 f32: O (16 fragments), the row sums
# and maxes x log2e of both heads. The K-th arriver of a unit merges the K partials in piece order and runs the epilogue
# (deterministic, not bit-identical: gate N). Off below start_pos KSPLIT_MIN (pieces of short units)
KSPLIT = int(os.environ.get("ATTN_KSPLIT", "0"))
KSPLIT_MIN = 4096


def ksplit_floats():
    """The ksplit binding's partials (f32) after its 64 counters: (P - 1) K pieces of 32 NQB P.V lanes x 132"""
    return (PERSIST - 1) * KSPLIT * 32 * NQB * 132
PDYN_ZZ_BELOW = int(os.environ.get("R2P_ZZ_BELOW", "8192"))
# OCNT: binding ocnt (after the others): after a workgroup barrier lane 0 adds 1 to counter c tt + t (release) for
# 1024-feature chunk c and 128-token tile t of its output, tt = token_count / 128 (ocnt_need workgroups per counter),
# so a flag-waiting rotq (gen_rot.gen_qp flag) runs without a graph barrier behind this kernel (every lane adding,
# as the GEMMs do, made the kernel ~4% slower)
OCNT = False


def ocnt_need():
    """The lanes adding to one OCNT counter: query blocks per 128-token tile x workgroups per 1024-feature chunk."""
    return (128 // QT) * (1024 // (256 * HPW))
DOWHILE_MAIN = False   # the same for the head_dim-split loop (measured neutral)
# SOFT1..4 (measured, not adopted, docs/results.md): keep the head_dim split and run the softmax in wave A only, P^T and
# alpha through LDS. Bit-identical, -35% VALU, but the waiting partner raises barrier idle: +1.4..+4.4% at 126K.
SOFT1 = False
SOFT2 = False       # P.V(j - 1) beside QK(j) in phase 1, softmax(j) in phase 2 (2 barriers)
SOFT3 = False       # SOFT2 with wave B's P.V moved into phase 2 (overlapping A's softmax)
SOFT4 = False       # phase 1 QK(j), phase 2 both P.V(j - 1) + A's softmax(j)
S1_PASS_FIRST = True
SOFT1_MAP = "mod"
# DEC_LSHADD: in the int4 / int8 K and V decoders, a left-shifted nibble pair ((w << s) & M) | C becomes
# ((w & (M >> s)) << s) + C (the fields do not overlap, so | is +), which Loom selects as v_and_b32 + v_lshl_add_u32:
# 2 VALU instead of 3, same bits
DEC_LSHADD = True


def dec_pair(e, ind, out, w, sh, mask, gname):
    """out = ((w << sh or w >> -sh) & mask) | g, as i32 (mask the f16 mantissa field of both lanes, then OR the 1.0)"""
    if DEC_LSHADD and sh > 0:
        e(f"{ind}{out}_m = scalar.constant {mask >> sh} : i32")
        e(f"{ind}{out}_a = scalar.andi {w}, {out}_m : i32")
        e(f"{ind}{out}_p = scalar.constant {1 << sh} : i32")
        e(f"{ind}{out} = scalar.fmai {out}_a, {out}_p, {gname} : i32")
        return
    op = "shli" if sh >= 0 else "shrui"
    e(f"{ind}{out}_c = scalar.constant {abs(sh)} : i32")
    e(f"{ind}{out}_t = scalar.{op} {w}, {out}_c : i32")
    e(f"{ind}{out}_mk = scalar.constant {mask} : i32")
    e(f"{ind}{out}_x = scalar.andi {out}_t, {out}_mk : i32")
    e(f"{ind}{out} = scalar.ori {out}_x, {gname} : i32")
# Page lookups are scalar (SMEM) loads of the global table; an LDS copy of the table was slower.
# No in-kernel clamp (it cost 1%): the host validates every entry (< npages) before upload,
# and the cache writers clamp page indices into their pools.
assert not (K4 and K8)
# Q16: the query arrives as f16(q * 0.0625) from RoPE (emit_prefill_pp.rope_q16), the scale and rounding this kernel
# applied to the f32 query: half the Q bytes, same bits. Not with K4 (its H256 rotates the scaled f32 Q first).
Q16 = not K4
# QROT (kv4): the Q prologue's H256 rotation runs in its own kernel (gen_qrot, the same operations), which writes f16
# rotated Q; the attention then takes it as Q16 with direct Q fragments (no 33.8 KB Q stage: 4 workgroups per WGP).
# Set QROT and Q16 together.
QROT = False
S_OFF = V_OFF + 256 * VT_PITCH * 2   # S partials: 2 planes x NT x 4 f32
Q_PITCH = 264
Q_END = NQB * 16 * Q_PITCH * 2           # Q stage (prologue only)
# Q-drain dummy store: past the Q stage, or in the S slots when they are past it
DRAIN_OFF = S_OFF if S_OFF >= Q_END else Q_END
K2_OFF = S_OFF + NT * 32                # second K buffer (PIPE2)
POOL = max(S_OFF + NT * 32, DRAIN_OFF + NT * 16)
assert POOL <= 65536


def configure(hpw, qt):
    """Workgroup shape: hpw query heads (of one GQA group) x qt query tokens; recomputes the derived sizes.
    The default is 2 x 32; 6 x 16 packs a whole GQA group (each K / V tile staged once for all 6 heads)."""
    global HPW, QT, NQB, NT, Q_END, DRAIN_OFF, POOL, K2_OFF
    HPW, QT = hpw, qt
    NQB = HPW * QT // 16
    NT = 64 * NQB
    Q_END = NQB * 16 * Q_PITCH * 2
    DRAIN_OFF = S_OFF if S_OFF >= Q_END else Q_END
    K2_OFF = S_OFF + NT * 32
    POOL = max(S_OFF + NT * 32, DRAIN_OFF + NT * 16)
    # QDIRECT has no Q stage or drain slot: gen() sizes the pool S_OFF + NT * 32 then
    assert (S_OFF + NT * 32 if (Q16 if QDIRECT is None else QDIRECT) else POOL) <= 65536


def had64(e, pre, vecs, xors, scale):
    """Walsh-Hadamard times `scale`: stages 1..32 inside this thread's 64 dims (16 vector<4xf32> in dim order),
    then across threads (lane xor x flips qpart bit log2(x): dims 64, 128 apart).
    Butterfly as in yah_kq4 (lower = a + b, upper = lower - upper), so both sides use the same orthogonal H."""
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


def qnorm_load(e, dst, view, vt, base, ks, f16=False):
    """QNORM: the 16-dim f32 runs at base + 32 k (k in ks) as V4s; all loads issued before any use"""
    out = {}
    for k in ks:
        if f16:   # one 32-B load, widened
            e(f"  %{dst}{k}_a = index.add {base}, %qo{32 * k} : index")
            e(f"  %{dst}{k}_h = vector.load {view}[%{dst}{k}_a] : view<[{vt}]xf16> -> {V16H}")
            e(f"  %{dst}{k}_w = vector.extf %{dst}{k}_h : {V16H} to vector<16xf32>")
            out[k] = [f"%{dst}{k}_w"]
            continue
        for j in range(4):
            e(f"  %{dst}{k}_{j}a = index.add {base}, %qo{32 * k + 4 * j} : index")
            e(f"  %{dst}{k}_{j} = vector.load {view}[%{dst}{k}_{j}a] : view<[{vt}]xf32> -> {V4}")
        out[k] = [f"%{dst}{k}_{j}" for j in range(4)]
    return out


def qnorm_scalars(e, dst, vecs):
    out = []
    for j in range(16):
        if len(vecs) == 1:
            e(f"  %{dst}e{j} = vector.extract {vecs[0]}[{j}] : vector<16xf32> -> f32")
        else:
            e(f"  %{dst}e{j} = vector.extract {vecs[j // 4]}[{j % 4}] : {V4} -> f32")
        out.append(f"%{dst}e{j}")
    return out


def qnorm_tree(e, p, lvl):
    """pairwise levels (RoPE's reduction order) -> the root"""
    d = 0
    while len(lvl) > 1:
        nxt = []
        for i in range(len(lvl) // 2):
            e(f"  %{p}{d}_{i} = scalar.addf {lvl[2 * i]}, {lvl[2 * i + 1]} : f32")
            nxt.append(f"%{p}{d}_{i}")
        lvl, d = nxt, d + 1
    return lvl[0]


def NKS():
    return (4 if NQB == 4 else 2) if R2P else 2


def staging_maps(e):
    """Staging maps. K: item j = tid + 256nn, key = j/32, d8 = (j%32)*8, LDS row 2*(key%8) + key/8.
    One key row per wave: coalesced global, contiguous LDS.
    V^T: lane = dim, LDS row (dim/16)*16 + 2*(dim%8) + (dim%16)/8."""
    e("  %vdl = index.rem %vt, %c16 : index")
    e("  %vdb = index.sub %vt, %vdl : index")
    e("  %vdl8 = index.rem %vdl, %c8 : index")
    e("  %vdh = index.div %vdl, %c8 : index")
    e("  %vdr0 = index.mul %vdl8, %c2 : index")
    e("  %vdr1 = index.add %vdr0, %vdh : index")
    if VSWZ and (VSWZ_T or not TILED_OUT):   # with TILED_OUT the paged build allocates 236 VGPRs (192 without)
        e("  %vrow = index.add %vt, %c0 : index")
        if R2P and R2P_K2:   # V^T key halves in two 4 KB planes of 16-B rows: the store lane's row bytes
            e("  %vgs = index.mul %vrow, %c16 : index")
            # loop-invariant lane addresses, the plane / row block in the view base (an immediate): stores (row lo, hi,
            # row + 128 lo, hi), fragment f's lo / hi plane loads (rows 16 f ..)
            for nm, off in zip(("s0", "s1", "s2", "s3"), VSPL_ST):
                e(f"  %vgv{nm}_o = index.constant {V_OFF + off} : offset")
                e(f"  %vgv{nm} = buffer.view %pool[%vgv{nm}_o] : buffer -> view<{VT_BYTES - off}xi8>")
            for f, off in enumerate([256 * f for f in range(8)] + [4096 + 256 * f for f in range(8)]):
                e(f"  %vgv{f}_o = index.constant {V_OFF + off} : offset")
                e(f"  %vgv{f} = buffer.view %pool[%vgv{f}_o] : buffer -> view<{VT_BYTES - off}xi8>")
    else:
        e("  %vrow = index.add %vdb, %vdr1 : index")

    if R2P and NQB == 8:   # 256 threads, 2 items each: role wave r2w = 4 sh + w stages the 128-thread map's items
        # 2 sh, 2 sh + 1 (keys w + 8 sh + 4 j, LDS rows 2 w + sh + 8 j); V^T: its dims + 128 sh
        e("  %kd0 = index.mul %lane, %c8 : index")
        e("  %r2w4 = index.rem %r2w, %c4 : index")
        e("  %r2sh = index.div %r2w, %c4 : index")
        e("  %kr2w0 = index.mul %r2w4, %c2 : index")
        e("  %kr2w = index.add %kr2w0, %r2sh : index")
        e("  %kk0a = index.mul %r2sh, %c8 : index")
        e("  %kk0 = index.add %r2w4, %kk0a : index")
        e("  %kk1 = index.add %kk0, %c4 : index")
        e("  %kr0 = index.add %kr2w, %c0 : index")
        e("  %kr1 = index.add %kr2w, %c8 : index")
        e("  %kd1 = index.add %kd0, %c0 : index")
        e("  %r2vsh = index.mul %r2sh, %c2048 : index")   # V^T: + 128 dims (global halves / LDS bytes)
        if R2P_K2:
            e("  %vgs8 = index.add %vgs, %r2vsh : index")
        e("  %vhbs = index.add %vhb, %r2vsh : index")   # the V lane base of this wave's dim half (load_v)
        e("  %vhls = index.add %vhl, %r2vsh : index")
        return
    if R2P:   # one role's 128 threads stage a tile, 4 items each: key = r2w + 4 nn (uniform), d8 = lane * 8
        e("  %kd0 = index.mul %lane, %c8 : index")
        e("  %kr2w = index.mul %r2w, %c2 : index")
        for nn in range(4):
            e(f"  %kk{nn}c = index.constant {4 * nn} : index")
            e(f"  %kk{nn} = index.add %r2w, %kk{nn}c : index")
            e(f"  %kr{nn}c = index.constant {(0, 8, 1, 9)[nn]} : index")
            e(f"  %kr{nn} = index.add %kr2w, %kr{nn}c : index")
            if nn:
                e(f"  %kd{nn} = index.add %kd0, %c0 : index")
        return
    for nn in range(NKS()):
        e(f"  %ki{nn}c = index.constant {256 * nn} : index")
        e(f"  %ki{nn} = index.add {'%tidr' if R2P else '%tid'}, %ki{nn}c : index")
        e(f"  %kk{nn} = index.div %ki{nn}, %c32 : index")
        e(f"  %kd{nn}a = index.rem %ki{nn}, %c32 : index")
        e(f"  %kd{nn} = index.mul %kd{nn}a, %c8 : index")
        e(f"  %kk{nn}l = index.rem %kk{nn}, %c8 : index")
        e(f"  %kk{nn}h = index.div %kk{nn}, %c8 : index")
        e(f"  %kk{nn}r0 = index.mul %kk{nn}l, %c2 : index")
        e(f"  %kr{nn} = index.add %kk{nn}r0, %kk{nn}h : index")


def qnorm_phase1(e):
    """QNORM before the prologue barrier: issue this lane's own f32 dim loads; returns (own, sums), sums() emitting
    (just before the barrier: the first K tile's loads go out under these) the sum of squares of its 128 head dims into
    the LDS slot for the other head-dim wave of the pair"""
    nown = 4    # own fragments per lane half (local fragment c = 2k + half)
    e("  %qx16 = scalar.constant 16 : i32")
    e("  %qx32 = scalar.constant 32 : i32")
    e("  %qh16 = index.mul %half, %c16 : index")
    e("  %qhis0 = index.cmp eq, %half, %c0 : index")
    e("  %qnb = index.add %qdb, %qh16 : index")
    for n in sorted({32 * k + 4 * j for k in range(nown) for j in range(4)}):
        e(f"  %qo{n} = index.constant {n} : index")
    if R2P_QSPLIT:
        return qsplit_phase1(e, nown), lambda: None
    vecs = qnorm_load(e, "qxv", "%q_flat", "%qtot", "%qnb", range(nown), QG16 or "q16" in R2P_DIAG)
    own = [[f"%qx{k}e{j}" for j in range(16)] for k in range(nown)]
    return own, lambda: qnorm_sums(e, vecs, nown)


def qsplit_phase1(e, nown):
    """R2P_QSPLIT: the block's final f16 halves (two V8H per local fragment), the raw f32 rotary dims of fragments 0
    and 1 (used by the head-dim-0 waves only), and the lane's query's sums of squares (its own head for QK waves)"""
    e("  %qsp_c4 = index.constant 4 : index")
    e("  %qsp_c128 = index.constant 128 : index")
    e("  %qsp_c8 = index.constant 8 : index")
    halves = []
    for k in range(nown):
        e(f"  %qsa{k} = index.add %qnb, %qo{32 * k} : index")
        e(f"  %qsb{k} = index.add %qsa{k}, %qsp_c8 : index")
        e(f"  %qslo{k} = vector.load %q_flat[%qsa{k}] : view<[%qtot]xf16> -> {V8H}")
        e(f"  %qshi{k} = vector.load %q_flat[%qsb{k}] : view<[%qtot]xf16> -> {V8H}")
        halves.append((f"%qslo{k}", f"%qshi{k}"))
    e("  %qsph = scf.select %isA, %r_head, %r_head_o : index")   # the block's head (as %qdbn)
    e("  %qsp_c1536b = index.constant 1536 : index")
    e("  %qsp_c64 = index.constant 64 : index")
    e("  %qsp_c48b = index.constant 384 : index")
    e("  %qsp_c2 = index.constant 16 : index")
    e("  %qrt = index.mul %qdq, %qsp_c1536b : index")
    e("  %qrh = index.mul %qsph, %qsp_c64 : index")
    e("  %qrbase1 = index.add %qrt, %qrh : index")    # (token * 24 + head) * 64
    e("  %qrbase = index.add %qrbase1, %qh16 : index")
    rot = []
    for k in range(0 if R2P_QROT16 else 2):
        vs = []
        for j in range(4):
            e(f"  %qrx{k}_{j}a = index.add %qrbase, %qo{32 * k + 4 * j} : index")
            e(f"  %qrx{k}_{j} = vector.load %qr_flat[%qrx{k}_{j}a] : view<[%qrtot]xf32> -> {V4}")
            vs.append(f"%qrx{k}_{j}")
        rot.append(vs)
    e("  %qst = index.mul %qdq, %qsp_c48b : index")
    e("  %qsh = index.mul %qsph, %qsp_c2 : index")
    e("  %qsbase = index.add %qst, %qsh : index")   # (token * 24 + head) * 16
    for x in range(4):
        e(f"  %qsbc{x} = index.constant {4 * x} : index")
        e(f"  %qsbase{x} = index.add %qsbase, %qsbc{x} : index")
        e(f"  %qssl{x} = vector.load %qs_flat[%qsbase{x}] : view<[%qsstot]xf32> -> {V4}")
    return halves, rot


def qnorm_sums(e, vecs, nown):
    own = [qnorm_scalars(e, f"qx{k}", vecs[k]) for k in range(nown)]
    s16 = []
    for k in range(nown):
        sq = []
        for j, x in enumerate(own[k]):
            e(f"  %qs{k}q{j} = scalar.mulf {x}, {x} : f32")
            sq.append(f"%qs{k}q{j}")
        s16.append(qnorm_tree(e, f"qs{k}l", sq))
    # S32 of fragment pair (2k, 2k + 1) = mine + the xor-16 partner's (commutative: both lanes hold the same bits)
    s32 = []
    for i in range(0, nown, 4):
        e(f"  %qsv{i} = vector.from_elements {', '.join(s16[i:i + 4])} : {V4}")
        e(f"  %qsp{i}, %qspv{i} = kernel.subgroup.shuffle<xor> %qsv{i}, %qx16, %qx32 : {V4}, i32, i32")
        for j in range(4):
            e(f"  %qsp{i}_{j} = vector.extract %qsp{i}[{j}] : {V4} -> f32")
            e(f"  %qs32_{i + j} = scalar.addf {s16[i + j]}, %qsp{i}_{j} : f32")
            s32.append(f"%qs32_{i + j}")
    root = qnorm_tree(e, "qsl", s32)
    e(f"  %qxs_o = index.constant {QX_OFF} : offset")
    e(f"  %qxs_view = buffer.view %pool[%qxs_o] : buffer -> view<{NQB * 2 * 16}xf32>")
    e("  %qxs0 = index.mul %wqb, %c32 : index")
    e("  %qxs1 = index.add %qxs0, %sub : index")
    e("  %qxs_me = index.add %qxs1, %hd16q : index")
    e(f"  view.store {root}, %qxs_view[%qxs_me] : f32, view<{NQB * 2 * 16}xf32>")
    return root


QN_NW = []   # (qnorm_phase2_norm's normed x weight rows, read back by qnorm_phase2)


def qnorm_phase2_norm(e, own, root, wv, nown):
    """QNORM: the head sum of squares, x inv x weight (the f16 conversion and RoPE follow in qnorm_phase2)"""
    e("  %qxs_ot0 = index.sub %c16, %hd16q : index")
    e("  %qxs_ot = index.add %qxs1, %qxs_ot0 : index")
    e(f"  %qxs_other = view.load %qxs_view[%qxs_ot] : view<{NQB * 2 * 16}xf32> -> f32")
    e(f"  %qstot = scalar.addf {root}, %qxs_other : f32")
    total = "%qstot"
    e("  %qhd256 = scalar.constant 256.0 : f32")
    if PRO_FAST:
        e("  %qhd256i = scalar.constant 0.00390625 : f32")
        e(f"  %qmean = scalar.mulf {total}, %qhd256i : f32")
    else:
        e(f"  %qmean = scalar.divf {total}, %qhd256 : f32")
    e("  %qme = scalar.addf %qmean, %qeps_s : f32")
    e("  %qinv = scalar.rsqrtf %qme : f32")
    e("  %qsc16 = scalar.constant 0.0625 : f32")
    def norm_row(k):
        w = qnorm_scalars(e, f"qw{k}", wv[k])
        row = []
        for j in range(16):
            e(f"  %qn{k}_{j} = scalar.mulf {own[k][j]}, %qinv : f32")
            e(f"  %qnw{k}_{j} = scalar.mulf %qn{k}_{j}, {w[j]} : f32")
            row.append(f"%qnw{k}_{j}")
        return row
    QN_NW[:] = [[None] * nown if R2P_QPF else ([norm_row(k) for k in range(nown)]), norm_row]


def qnorm_phase2(e, own, root):
    """QNORM after the prologue barrier: the head sum, x inv x weight, RoPE, f16(q / 16), traded into %qdv{c}"""
    if R2P_QSPLIT:
        nown = len(own[0])
    else:
        nown = len(own)
    e("  %qw_na, %qe_na = buffer.assume.noalias %qnw, %qeps : buffer, buffer")
    e("  %qw_flat = buffer.view %qw_na[%base] : buffer -> view<[%c256]xf32>")
    e("  %qe_flat = buffer.view %qe_na[%base] : buffer -> view<[%c1]xf32>")
    e("  %qwb = index.add %hd128, %qh16 : index")
    wv = {} if R2P_QPF or R2P_QROT16 else qnorm_load(e, "qwv", "%qw_flat", "%c256", "%qwb", range(2 if R2P_QSPLIT else nown))
    e("  %qeps_s = view.load %qe_flat[%c0] : view<[%c1]xf32> -> f32")
    if R2P_QSPLIT:
        qs_nw, qs_halves = qsplit_phase2(e, own, wv)
    else:
        qs_nw = qs_halves = None
        qnorm_phase2_norm(e, own, root, wv, nown)
    nw = qs_nw + [None] * (nown - 2) if R2P_QSPLIT else QN_NW[0]

    def f16frag(p, xs, ind="  "):
        """f16(x / 16) of 16 dims -> two V8H"""
        hs = []
        for j in range(16):
            e(f"{ind}%{p}s{j} = scalar.mulf {xs[j]}, %qsc16 : f32")
            e(f"{ind}%{p}h{j} = scalar.fptrunc %{p}s{j} : f32 to f16")
            hs.append(f"%{p}h{j}")
        e(f"{ind}%{p}lo = vector.from_elements {', '.join(hs[:8])} : {V8H}")
        e(f"{ind}%{p}hi = vector.from_elements {', '.join(hs[8:])} : {V8H}")
        return f"%{p}lo", f"%{p}hi"

    def rope(ind):
        """fragments k = 0, 1 (dims p, p + 32 with p = 16 half + j) rotated, as f16 halves"""
        e(f"{ind}%qrcur0 = index.add %start_pos, %qdq : index")
        e(f"{ind}%qrcur1 = index.cast %qrcur0 : index to i32")
        e(f"{ind}%qrcur = scalar.sitofp %qrcur1 : i32 to f32")
        e(f"{ind}%qrtwo = scalar.constant 2.0 : f32")
        e(f"{ind}%qrone = scalar.constant 1.0 : f32")
        e(f"{ind}%qrdim = scalar.constant 64.0 : f32")
        e(f"{ind}%qrtheta = scalar.constant 10000000.0 : f32")
        r0, r1 = [], []
        for j in range(16):
            e(f"{ind}%qrp{j}c = index.constant {j} : index")
            e(f"{ind}%qrp{j} = index.add %qh16, %qrp{j}c : index")
            e(f"{ind}%qrpi{j} = index.cast %qrp{j} : index to i32")
            e(f"{ind}%qrpf{j} = scalar.sitofp %qrpi{j} : i32 to f32")
            e(f"{ind}%qr2p{j} = scalar.mulf %qrtwo, %qrpf{j} : f32")
            if PRO_FAST:
                e(f"{ind}%qrdimi{j} = scalar.constant 0.015625 : f32")
                e(f"{ind}%qrex{j} = scalar.mulf %qr2p{j}, %qrdimi{j} : f32")
                e(f"{ind}%qrif{j} = scalar.powf<afn> %qrtheta, %qrex{j} : f32")
                e(f"{ind}%qrr0{j} = scalar.divf<nnan|ninf|nsz|arcp> %qrone, %qrif{j} : f32")
                e(f"{ind}%qrnd{j} = scalar.negf %qrif{j} : f32")
                e(f"{ind}%qrer{j} = scalar.fmaf %qrnd{j}, %qrr0{j}, %qrone : f32")
                e(f"{ind}%qriv{j} = scalar.fmaf %qrer{j}, %qrr0{j}, %qrr0{j} : f32")
            else:
                e(f"{ind}%qrex{j} = scalar.divf %qr2p{j}, %qrdim : f32")
                e(f"{ind}%qrif{j} = scalar.powf<afn> %qrtheta, %qrex{j} : f32")
                e(f"{ind}%qriv{j} = scalar.divf %qrone, %qrif{j} : f32")
            if "norope" in R2P_DIAG:   # diag: no rotation (cos 1, sin 0); the angle math above is dead
                e(f"{ind}%qrc{j} = scalar.constant 1.0 : f32")
                e(f"{ind}%qrs{j} = scalar.constant 0.0 : f32")
            else:
                e(f"{ind}%qran{j} = scalar.mulf %qrcur, %qriv{j} : f32")
                e(f"{ind}%qrc{j} = scalar.cosf<afn> %qran{j} : f32")
                e(f"{ind}%qrs{j} = scalar.sinf<afn> %qran{j} : f32")
            a, b = nw[0][j], nw[1][j]
            e(f"{ind}%qra0_{j} = scalar.mulf {a}, %qrc{j} : f32")
            e(f"{ind}%qra1_{j} = scalar.mulf {b}, %qrs{j} : f32")
            e(f"{ind}%qrr0_{j} = scalar.subf %qra0_{j}, %qra1_{j} : f32")
            e(f"{ind}%qrb0_{j} = scalar.mulf {a}, %qrs{j} : f32")
            e(f"{ind}%qrb1_{j} = scalar.mulf {b}, %qrc{j} : f32")
            e(f"{ind}%qrr1_{j} = scalar.addf %qrb0_{j}, %qrb1_{j} : f32")
            r0.append(f"%qrr0_{j}")
            r1.append(f"%qrr1_{j}")
        return f16frag("qrf0", r0, ind) + f16frag("qrf1", r1, ind)

    def rope_halves():
        # only the head-dim-0 wave of a pair holds dims < 64
        e("  %qsg = kernel.subgroup.id : index")
        e("  %qsgh = index.rem %qsg, %c2 : index")
        e("  %qrot0 = index.cmp eq, %qsgh, %c0 : index")
        e(f"  %qrh0, %qrh1, %qrh2, %qrh3 = scf.if %qrot0 -> ({V8H}, {V8H}, {V8H}, {V8H}) {{")
        e(f"    scf.yield {', '.join(rope('    '))} : {V8H}, {V8H}, {V8H}, {V8H}")
        e("  } else {")
        pl = (list(qs_halves[0]) + list(qs_halves[1]) if R2P_QSPLIT else
              f16frag("qp0", nw[0], "    ") + f16frag("qp1", nw[1], "    "))
        e(f"    scf.yield {', '.join(pl)} : {V8H}, {V8H}, {V8H}, {V8H}")
        e("  }")
        return ("%qrh0", "%qrh1", "%qrh2", "%qrh3")

    def trade(k, hk):
        """local fragment 2k + b is mine when b == half, else the partner's"""
        theirs = []
        for t, hv in enumerate(hk):
            e(f"  %qfb{k}_{t} = vector.bitcast {hv} : {V8H} to {V4}")
            e(f"  %qfp{k}_{t}, %qfpv{k}_{t} = kernel.subgroup.shuffle<xor> %qfb{k}_{t}, %qx16, %qx32 : {V4}, i32, i32")
            e(f"  %qfq{k}_{t} = vector.bitcast %qfp{k}_{t} : {V4} to {V8H}")
            theirs.append(f"%qfq{k}_{t}")
        e(f"  %qfm{k} = vector.concat<0> {hk[0]}, {hk[1]} : {V8H}, {V8H} -> {V16H}")
        e(f"  %qft{k} = vector.concat<0> {theirs[0]}, {theirs[1]} : {V8H}, {V8H} -> {V16H}")
        e(f"  %qdv{2 * k} = scf.select %qhis0, %qfm{k}, %qft{k} : {V16H}")
        e(f"  %qdv{2 * k + 1} = scf.select %qhis0, %qft{k}, %qfm{k} : {V16H}")

    if R2P_QPF:   # one 16-dim group at a time, its Q and weights reloaded (the tail's VGPR peak: the sums' loads die)
        own = list(own)

        def grp(k):
            wv.update(qnorm_load(e, "qwv", "%qw_flat", "%c256", "%qwb", [k]))
            ov = qnorm_load(e, "qy", "%q_flat", "%qtot", "%qnb", [k])
            own[k] = qnorm_scalars(e, f"qy{k}", ov[k])
            nw[k] = QN_NW[1](k)
        halves = [None] * nown
        for k in range(2, nown):
            grp(k)
            halves[k] = f16frag(f"qf{k}", nw[k])
            e("  scf.schedule.fence")
        grp(0)
        grp(1)
    else:
        halves = [None, None] + ([qs_halves[k] for k in range(2, nown)] if R2P_QSPLIT else
                                 [f16frag(f"qf{k}", nw[k]) for k in range(2, nown)])
    h = rope_halves()
    halves[0], halves[1] = h[0:2], h[2:4]
    for k in range(nown):
        trade(k, halves[k])


def qsplit_phase2(e, own, wv):
    """R2P_QSPLIT after the prologue barrier: r log2e from the sums (%qsl2e), and q w of the rotary dims (fragments 0
    and 1; RoPE'd by the head-dim-0 waves only); returns (those two rows, the loaded halves)"""
    halves, rot = own
    e(f"  %qssv = vector.addf %qssl0, %qssl1 : {V4}")
    e(f"  %qssw = vector.addf %qssl2, %qssl3 : {V4}")
    e(f"  %qssa = vector.addf %qssv, %qssw : {V4}")
    for x in range(4):
        e(f"  %qss{x} = vector.extract %qssa[{x}] : {V4} -> f32")
    e("  %qss01 = scalar.addf %qss0, %qss1 : f32")
    e("  %qss23 = scalar.addf %qss2, %qss3 : f32")
    e("  %qstot = scalar.addf %qss01, %qss23 : f32")
    e("  %qhd256i = scalar.constant 0.00390625 : f32")
    e("  %qmean = scalar.mulf %qstot, %qhd256i : f32")
    e("  %qme = scalar.addf %qmean, %qeps_s : f32")
    e("  %qinv = scalar.rsqrtf %qme : f32")
    e("  %qsl2e = scalar.mulf %qinv, %log2e : f32")
    if R2P_LAZY and R2P_QFOLD:   # S in log2 units: R2P_LAZY nats = R2P_LAZY log2e
        e(f"  %qslzth = scalar.constant {R2P_LAZY * 1.4426950408889634} : f32")
    elif R2P_LAZY:   # the lazy rescale threshold in S' = S / r units: R2P_LAZY rms (= R2P_LAZY / r), or p outgrows f16
        e("  %qsrms = scalar.mulf %qme, %qinv : f32")
        e(f"  %qslzc = scalar.constant {R2P_LAZY} : f32")
        e("  %qslzth = scalar.mulf %qsrms, %qslzc : f32")
    if R2P_QFOLD == 2:   # f16(q w / 16) x f16(r log2e) for the halves loaded before the barrier; RoPE ones fold in f32
        e("  %qsc16b = scalar.constant 0.0625 : f32")
        e("  %qsc16 = scalar.mulf %qsc16b, %qsl2e : f32")
        e("  %qfl2h = scalar.fptrunc %qsl2e : f32 to f16")
        e(f"  %qfl2v8 = vector.splat %qfl2h : {V8H}")
        sh = []
        for k, hk in enumerate(halves):
            row = []
            for t, hv in enumerate(hk):
                e(f"  %qsf{k}_{t} = vector.mulf {hv}, %qfl2v8 : {V8H}")
                row.append(f"%qsf{k}_{t}")
            sh.append(row)
        halves = sh
    else:
        e("  %qsc16 = scalar.constant 0.0625 : f32")
    nw = []
    if R2P_QROT16:   # q w = f32(f16(q w / 16)) x 16 (exact scaling)
        e("  %qr16c = scalar.constant 16.0 : f32")
        for k in range(2):
            row = []
            for j in range(16):
                e(f"  %qrh{k}_{j} = vector.extract {halves[k][j // 8]}[{j % 8}] : {V8H} -> f16")
                e(f"  %qrf{k}_{j} = scalar.extf %qrh{k}_{j} : f16 to f32")
                e(f"  %qnw{k}_{j} = scalar.mulf %qrf{k}_{j}, %qr16c : f32")
                row.append(f"%qnw{k}_{j}")
            nw.append(row)
        return nw, halves
    for k in range(2):
        xs = qnorm_scalars(e, f"qrx{k}", rot[k])
        w = qnorm_scalars(e, f"qw{k}", wv[k])
        row = []
        for j in range(16):
            e(f"  %qnw{k}_{j} = scalar.mulf {xs[j]}, {w[j]} : f32")
            row.append(f"%qnw{k}_{j}")
        nw.append(row)
    return nw, halves


VT_BYTES = 0   # R2P_K2: the V^T region: keys 0..7 of rows 0..255 (16 B each), then keys 8..15
VSPL_ST = (0, 4096, 2048, 6144)   # R2P_K2 V^T stores: row lo, row hi, row + 128 lo, row + 128 hi


def k2_layout():
    global VT_PITCH, V_OFF, S_OFF, VT_BYTES
    VT_PITCH = 16
    V_OFF = 2 * 16 * KT_PITCH * 2
    VT_BYTES = 32 * 256
    S_OFF = V_OFF + VT_BYTES   # R2P_K2: P^T slots (S plane 0), then S plane 1, alpha


TOKR = re.compile(r"%[A-Za-z0-9_.]+")


def _defs(ln):
    m = re.match(r"\s*(%[A-Za-z0-9_.]+(?:, %[A-Za-z0-9_.]+)*) = ", ln)
    return TOKR.findall(m.group(1)) if m else []


def _uses(ln):
    return TOKR.findall(ln.split(" = ", 1)[1] if _defs(ln) else ln)


def _ren(ln, mp):
    return TOKR.sub(lambda m: mp.get(m.group(0), m.group(0)), ln)


def slice_copy(L, lo, hi, virt, roots, sfx):
    """virt's lines plus the flat defs in L[lo:hi] they need that depend on a root, renamed (sfx), roots substituted"""
    flat = lambda l: l.startswith("  %") and not l.startswith("   ") and not l.rstrip().endswith("{")
    dl = {}
    for i in range(lo, hi):
        for d in _defs(L[i]):
            dl[d] = i if flat(L[i]) else -1
    vdefs = {d for ln in virt for d in _defs(ln)}
    need, seen, dep = [], set(), {}

    def rooted(n):
        if n in roots:
            return True
        if n not in dl or dl[n] < 0:
            return False
        if n not in dep:
            dep[n] = False
            dep[n] = any(rooted(u) for u in _uses(L[dl[n]]))
        return dep[n]

    def pull(n):
        if n in seen or n in vdefs or n in roots or n not in dl or not rooted(n):
            return
        seen.add(n)
        assert dl[n] >= 0, f"slice: {n} is defined by a region op"
        for u in _uses(L[dl[n]]):
            pull(u)
        need.append(dl[n])
    for ln in virt:
        for u in _uses(ln):
            pull(u)
    mp = dict(roots)
    for i in sorted(set(need)):
        for d in _defs(L[i]):
            mp[d] = d + sfx
    for ln in virt:
        for d in _defs(ln):
            mp[d] = d + sfx
    return [_ren("  " + L[i].strip(), mp) for i in sorted(set(need))] + [_ren(ln, mp) for ln in virt]


def qtouch_pass(L, pval_line):
    """R2P_QTOUCH: the QK waves' tail touches the next unit's Q lines (see R2P_QTOUCH)"""
    i_pv = L.index(pval_line)
    i_top = next(i for i in range(i_pv, len(L)) if L[i].startswith("  %wgl = index.assume %wglp"))
    i_sb = next(i for i in range(i_top, len(L)) if "kernel.barrier<workgroup>" in L[i])
    i_y = next(i for i in range(i_sb, len(L)) if L[i].startswith("  scf.yield %pqnu, %pws"))
    v = ["  %qtl = index.rem %tid, %c32 : index", "  %qth = index.div %qtl, %c16 : index",
         "  %qth16 = index.mul %qth, %c16 : index"]   # (recomputed: the unit top's lane terms would live across the loops)
    for t, qdb in (("a", "%qdb"), ("b", "%qdbo")):
        v.append(f"  %qtb_{t} = index.add {qdb}, %qth16 : index")
        for k in range(4):
            v.append(f"  %qta{k}_{t} = index.add %qtb_{t}, %qo{32 * k} : index")
            v.append(f"  %qtv{k}_{t} = view.load %q_flat[%qta{k}_{t}] : view<[%qtot]xf32> -> f32")
    xs = [f"%qtv{k}_{t}" for t in "ab" for k in range(4)]
    acc = xs[0]
    for j, x in enumerate(xs[1:]):
        v.append(f"  %qtm{j} = scalar.maxnumf {acc}, {x} : f32")
        acc = f"%qtm{j}"
    v.append(f"  %qtw_o = index.constant {16 * KT_PITCH * 2} : offset")   # K slot 1: idle from the last V tile to tile 0's barrier A
    v.append("  %qtw_view = buffer.view %pool[%qtw_o] : buffer -> view<256xf32>")
    v.append(f"  view.store {acc}, %qtw_view[%tid] : f32, view<256xf32>")
    body = slice_copy(L, i_pv + 1, i_sb, v, {"%wglp": "%qtu", "%pwc": "%pws"}, "_t")
    tail = ["  %qtc1 = index.sub %pun, %c1 : index", "  %qtu = index.min %pqnu, %qtc1 : index", "  scf.if %isA {"]
    tail += ["  " + ln for ln in body] + ["  }"]
    at = os.environ.get("R2P_QTAT", "y")   # (placement: y = after the epilogue, b = after the last V tile's barrier)
    if at == "b":
        i_lv = next(i for i in range(i_sb, i_y) if L[i].startswith("  %lsvvsx3 = "))
        i_y = next(i for i in range(i_lv, i_y) if "kernel.barrier<workgroup>" in L[i]) + 1
    return L[:i_y] + tail + L[i_y:]


def qpf_pass(L, pval_line):
    """R2P_QPF: the unit's Q prologue (K(0) stage, both head blocks' norm / RoPE / f16 fragments) in the previous
    unit's tail, in the QK waves; the fragments ride the unit loop as %rinit (see R2P_QPF)."""
    V8I = "vector<8xi32>"

    def find(pred, start=0):
        return next(i for i in range(start, len(L)) if pred(L[i]))
    i_wh = find(lambda l: "= scf.while(%pcur" in l)
    i_pv = L.index(pval_line)
    i_top = find(lambda l: l.startswith("  %wgl = index.assume %wglp"), i_pv)
    i_st = find(lambda l: "view.store" in l and "%qxs_view[%qxs_me]" in l, i_top)
    i_sb = i_st + 1
    assert "kernel.barrier<workgroup>" in L[i_sb]
    i_e0 = find(lambda l: l.startswith("  %qx0e0 = vector.extract"), i_top)
    i_q0 = find(lambda l: l.startswith("  %qdz0 = scf.select %r_live"), i_sb)
    i_xb1 = find(lambda l: "kernel.barrier" in l, i_q0)
    i_xb2 = find(lambda l: "kernel.barrier" in l, i_xb1 + 1)
    i_y = find(lambda l: l.startswith("  scf.yield %pqnu, %pws"), i_xb2)
    assert all("kernel.barrier" not in L[i] for i in range(i_top, i_sb))
    flat = lambda l: l.startswith("  %") and not l.startswith("   ") and not l.rstrip().endswith("{")
    dl = {}   # the unit top's defs before the sums barrier: name -> line (-1: a region op's)
    for i in range(i_pv + 1, i_sb):
        for d in _defs(L[i]):
            dl[d] = i if flat(L[i]) else -1
    rawl = [i for i in range(i_top, i_e0) if re.match(r"  %qxv\d+_\d+ = vector\.load %q_flat", L[i])]
    raw = [_defs(L[i])[0] for i in rawl]
    kld = [i for i in range(i_top, i_e0) if re.match(r"  %p0kv\d+ = vector\.load", L[i])]
    kst = [i for i in range(i_top, i_e0) if re.match(r"  vector\.store %p0kv\d+, %k_view", L[i])]
    assert len(raw) == 16 and len(kld) == 4 and len(kst) == 4, (len(raw), len(kld), len(kst))
    S = [i for i in range(i_e0, i_st) if not re.match(r"  %qxs(_o|_view|0|1|_me) = ", L[i])]   # sums, no slot
    p2 = list(range(i_sb + 1, i_q0))
    ib = next(i for i in p2 if L[i].startswith("  %bzero = "))
    root_ln = next(L[i] for i in p2 if L[i].startswith("  %qstot = scalar.addf"))
    root = TOKR.findall(root_ln.split(" = ", 1)[1])[0]
    p2 = [i for i in p2 if i != ib and not re.match(r"  %qxs_(ot0|ot|other) = ", L[i])]
    i_tr = next(i for i in p2 if L[i].startswith("  %qfb0_0 = vector.bitcast"))
    p2t = [i for i in p2 if i >= i_tr]   # the halves trade (xor 16): at the unit top, from the compact halves
    p2 = [i for i in p2 if i < i_tr]
    cpt = [TOKR.findall(L[i].split(" = ", 1)[1])[0] for i in p2t if re.match(r"  %qfb\d+_\d+ = vector\.bitcast", L[i])]
    assert len(cpt) == 8
    cone = [dl["%qnb"]] + [dl[u] for i in rawl for u in _uses(L[i]) if u.startswith("%qxv")] + rawl
    cone_d = {d for i in cone for d in _defs(L[i])}
    S_d = {d for i in S for d in _defs(L[i])}
    p2_d = {d for i in p2 for d in _defs(L[i])}

    def blk(t, qdb, wq):
        """block t: the other head-dim half's row sums (the partner wave's tree, its addresses), then this half's
        loads, sums, phase 2 and fragments; names suffixed t"""
        o = [f"  %qdbt_{t} = index.sub %{qdb}, %hd128 : index", f"  %qdbx_{t} = index.add %qdbt_{t}, %hd128o : index"]
        mo = {d: d + f"_{t}o" for d in cone_d | S_d}
        mo["%qdbn"] = f"%qdbx_{t}"
        o += [_ren(L[i], mo) for i in cone + S]
        o.append("  scf.schedule.fence")   # (each stage's loads after the previous stage: VGPR peak)
        mm = {d: d + f"_{t}" for d in cone_d | S_d | p2_d}
        mm["%qdbn"], mm["%wqn"] = f"%{qdb}", f"%{wq}"
        o += [_ren(L[i], mm) for i in cone + S]
        mm["%qxs_other"] = root + f"_{t}o"
        o += [_ren(L[i], mm) for i in p2]
        o.append("  scf.schedule.fence")
        return o
    virt = [L[i] for i in kld + kst] + ["  scf.schedule.fence", "  %hd128o = index.sub %c128, %hd128 : index"]
    virt += blk("a", "qdb", "wqb") + blk("b", "qdbo", "wqbo")
    yv = [c + "_a" for c in cpt] + [c + "_b" for c in cpt]
    ts16 = ", ".join([V8I] * 16)
    V8H = "vector<8xf16>"
    th16 = ", ".join([V8H] * 16)

    def instance(roots, sfx, alldeps, lds_w, other=None):
        vdefs = {d for ln in virt for d in _defs(ln)}
        need, seen, dep_root = [], set(), {}

        def rooted(n):
            if n in roots:
                return True
            if n not in dl or dl[n] < 0:
                return False
            if n not in dep_root:
                dep_root[n] = False
                dep_root[n] = any(rooted(u) for u in _uses(L[dl[n]]))
            return dep_root[n]

        def pull(n):
            if n in seen or n in vdefs or n in roots or n not in dl:
                return
            seen.add(n)
            if not (alldeps or rooted(n)):
                return
            assert dl[n] >= 0, f"qpf: {n} is defined by a region op"
            for u in _uses(L[dl[n]]):
                pull(u)
            need.append(dl[n])
        vl = virt
        if lds_w:   # the weights from the LDS copy
            wl = re.compile(r"vector\.load %qw_flat_[ab]\[(.*)\] : view<\[%c256\]xf32>")
            vl = [wl.sub(r"vector.load %qwl_view[\1] : view<256xf32>", ln) for ln in virt]
        for ln in vl:
            for u in _uses(ln):
                pull(u)
        pull("%isA")
        mp = dict(roots)
        for i in sorted(set(need)):
            for d in _defs(L[i]):
                mp[d] = d + sfx
        for ln in vl:
            for d in _defs(ln):
                mp[d] = d + sfx
        res = [f"%qpr{sfx}{k}" for k in range(16)]
        o = ["  " + L[i].strip() for i in sorted(set(need))]
        o = [_ren(ln, mp) for ln in o]
        o.append(f"  {', '.join(res)} = scf.if {mp.get('%isA', '%isA')} -> ({th16}) {{")
        o += ["  " + _ren(ln, mp) for ln in vl]
        o.append(f"    scf.yield {', '.join(mp[y] for y in yv)} : {th16}")
        o.append("  } else {")
        if other:   # the P.V waves: halves of values live here anyway (a zero constant would live across the loops)
            for k in range(16):
                o.append(f"    %qpo{k} = vector.bitcast {other[k // 2]} : {V8I} to vector<16xf16>")
                o.append(f"    %qps{k} = vector.slice %qpo{k}[{8 * (k % 2)}] : vector<16xf16> -> {V8H}")
            o.append(f"    scf.yield {', '.join(f'%qps{k}' for k in range(16))} : {th16}")
        else:
            o.append(f"    scf.yield {', '.join(['%qpzh'] * 16)} : {th16}")
        o.append("  }")
        return o, res

    # the tail's lane terms from tid % 128 (= tid in the QK waves) + an opaque loop-variant zero (queue word 2 & the
    # fetched id): else loop-invariant code motion hoists them out of the unit loop, live across the tile loops
    tail, resn = instance({"%wglp": "%qpfu", "%pwc": "%pws", "%tid": "%qtidv"}, "_n", False, R2P_QPFW,
                          [f"%fpo{k}" for k in range(8)])
    tail = ["  %qpfc1 = index.sub %pun, %c1 : index", "  %qpfu = index.min %pqnu, %qpfc1 : index",
            "  %qpzs = scalar.andi %pqns, %pwz0 : i32", "  %qpzx = index.cast %qpzs : i32 to index",
            "  %qtidv0 = index.add %tidr, %qpzx : index",
            "  %qtidv = index.assume %qtidv0 [range(%qtidv0, 0, 128)] : index"] + tail
    pro, resi = instance({"%wglp": "%qpfi", "%pwc": "%pwi"}, "_i", True, R2P_QPFW)
    wcp = ["  %qpwt = kernel.workitem.id<x> : index",
           "  %qpw_flat = buffer.view %qnw[%base] : buffer -> view<[%c256]xf32>",
           "  %qpwv = view.load %qpw_flat[%qpwt] : view<[%c256]xf32> -> f32",
           f"  %qwl_o = index.constant {QW_OFF} : offset",
           "  %qwl_view = buffer.view %pool[%qwl_o] : buffer -> view<256xf32>",
           "  view.store %qpwv, %qwl_view[%qpwt] : f32, view<256xf32>",
           "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)"]   # (all waves, once)
    pro = ["  %qpzh = vector.constant 0.0 : vector<8xf16>"] + (wcp if R2P_QPFW else []) + [
           "  %qpfc1i = index.sub %pun, %c1 : index", "  %qpfi = index.min %pqiu, %qpfc1i : index"] + pro
    drop = set(rawl + kld + kst + list(range(i_e0, i_sb + 1)) + list(range(i_sb + 1, i_xb2 + 1)))
    car = [f"%qcp{k}" for k in range(16)]
    newtop = [L[ib], "  %qw_na, %qe_na = buffer.assume.noalias %qnw, %qeps : buffer, buffer",   # (harness layout)
              "  %qw_flat = buffer.view %qw_na[%base] : buffer -> view<[%c256]xf32>",
              "  %qe_flat = buffer.view %qe_na[%base] : buffer -> view<[%c1]xf32>",
              f"  {', '.join(f'%rinit{c}' for c in range(16))} = scf.if %isA -> ({ts16}) {{"]
    for bi, t in enumerate("ab"):
        tm = {d: d + f"_{t}" for i in p2t for d in _defs(L[i])}
        tm.update({c: car[8 * bi + k] for k, c in enumerate(cpt)})
        newtop += ["  " + _ren(L[i], tm) for i in p2t]
        for c in range(8):
            newtop.append(f"    %qdz{c}_{t} = scf.select %r_live, %qdv{c}_{t}, %zq16x : vector<16xf16>")
            newtop.append(f"    %qbits{c}_{t} = vector.bitcast %qdz{c}_{t} : vector<16xf16> to {V8I}")
    newtop += [f"    scf.yield {', '.join([f'%qbits{c}_a' for c in range(8)] + [f'%qbits{c}_b' for c in range(8)])} : {ts16}",
               "  } else {", f"    scf.yield {', '.join(['%bzero'] * 16)} : {ts16}", "  }"]
    pq = [f"%pqx{k}" for k in range(16)]
    pd = [f"%pqd{k}" for k in range(16)]
    out = []
    for i, ln in enumerate(L):
        if i == i_wh:
            out += pro
            assert ln == "  %pfin, %pwf = scf.while(%pcur = %pqiu : index, %pwc = %pwi : i32) -> (index, i32) {", ln
            args = ", ".join(f"{c} = {r} : {V8H}" for c, r in zip(car, resi))
            fins = ", ".join(f"%qpf{k}" for k in range(16))
            ln = f"  %pfin, %pwf, {fins} = scf.while(%pcur = %pqiu : index, %pwc = %pwi : i32, {args}) -> (index, i32, {th16}) {{"
        elif i == i_pv:
            assert ln == "  %pnext, %pwn = scf.if %pval -> (index, i32) {", ln
            ln = f"  %pnext, %pwn, {', '.join(pq)} = scf.if %pval -> (index, i32, {th16}) {{"
        elif i in drop:
            if i == i_xb2:
                out += newtop
            continue
        elif i == i_y:
            out += tail
            assert ln == "  scf.yield %pqnu, %pws : index, i32", ln
            ln = f"  scf.yield %pqnu, %pws, {', '.join(resn)} : index, i32, {th16}"
        elif ln == "  scf.yield %pun, %pwc : index, i32":
            ln = f"  scf.yield %pun, %pwc, {', '.join(car)} : index, i32, {th16}"   # (exits: no unit)
        elif ln == "  scf.condition %pcont, %pnext, %pwn : i1, index, i32":
            ln = f"  scf.condition %pcont, %pnext, %pwn, {', '.join(pq)} : i1, index, i32, {th16}"
        elif ln == "  } do(%pw: index, %pwd: i32) {":
            ln = f"  }} do(%pw: index, %pwd: i32, {', '.join(f'{d}: {V8H}' for d in pd)}) {{"
        elif ln == "  scf.yield %pw, %pwd : index, i32":
            ln = f"  scf.yield %pw, %pwd, {', '.join(pd)} : index, i32, {th16}"
        out.append(ln)
    return out


def gen():
    L = []
    e = L.append

    def lsum_load():   # R2P_VPAD: + the pad's (+0.0) values (row sums > 0: exact)
        nm = "%lsum_l0" if R2P_VPAD else "%lsum_l"
        e(f"  {nm} = view.load %ls_view[%pa_own] : view<{16 * NQB}xf32> -> f32")
        for i in range(R2P_VPAD):
            e(f"  %vpf{i} = scalar.bitcast %vpv{i} : i32 to f32")
            e(f"  %lsum_l{i + 1 if i + 1 < R2P_VPAD else ''} = scalar.addf %lsum_l{i}, %vpf{i} : f32")
    def ksplit_merge(i0):
        """KS, after the last P.V: the unit's O / row sums (defined from L[i0] on) are renamed to its partial; a piece
        stores its partial (P.V lanes), takes its arrival, and the K-th arriver loads all K in piece order and merges:
        M = max m_i, f_i = 2^(m_i - M), O = sum f_i O_i, l = sum f_i l_i (m: row max x log2e, so f is the softmax's own
        exponent base). %kpwr: this workgroup writes the unit's output (a whole unit, or the merging piece)."""
        tk = re.compile(r"%(og\d+|lsum_lo|lsum_l)(?![A-Za-z0-9_])")
        L[i0:] = [tk.sub(lambda m: "%k_" + m.group(1), ln) for ln in L[i0:]]
        K = KSPLIT
        R = 32 * NQB * 132   # one piece's floats
        rt = f"{', '.join([V8] * 16)}, f32, f32"
        e(f"  %lm_own = view.load %lm_view[%pa_own] : view<{16 * NQB}xf32> -> f32")
        e(f"  %lm_oth = view.load %lm_view[%pa_oth] : view<{16 * NQB}xf32> -> f32")
        outs = [f"%og{f}" for f in range(16)] + ["%lsum_l", "%lsum_lo", "%kpwr"]
        own = [f"%k_og{f}" for f in range(16)] + ["%k_lsum_l", "%k_lsum_lo"]
        if "ksnoep" in R2P_DIAG:   # diag (compile bisect): the pieces' epilogue as a whole unit's (wrong results)
            for f in range(16):
                e(f"  %og{f} = vector.bitcast %k_og{f} : {V8} to {V8}")
            e("  %lsum_l = scalar.addf %k_lsum_l, %zero : f32")
            e("  %lsum_lo = scalar.addf %k_lsum_lo, %zero : f32")
            e("  %kpwr = index.cmp eq, %c0, %c0 : index")
            return
        e(f"  {', '.join(outs)} = scf.if %kpis -> ({rt}, i1) {{")
        e(f"    %kpvt0 = index.rem %tid, %c{32 * NQB} : index")   # the P.V lane (QK lanes: an alias, never stored)
        e("    %kpr0 = index.mul %kpvt0, %c132k : index")
        e(f"    %kprp = index.constant {R} : index")
        e("    %kprq = index.mul %kpq, %kprp : index")
        e("    %kpr = index.add %kprq, %kpr0 : index")
        ksm = "ksmerge" in R2P_DIAG
        if not ksm:   # kernel 2 stores no partial of its own (its dummy range is unused)
            e("    scf.if %isB {")
            for f in range(16):
                e(f"      %kpa{f} = index.add %kpr, %c{8 * f}k : index")
                e(f"      vector.store %k_og{f}, %kspv[%kpa{f}] : {V8}, view<{ksplit_floats()}xf32>")
            for i, v in enumerate(("%k_lsum_l", "%k_lsum_lo", "%lm_own", "%lm_oth")):
                e(f"      %kpa{16 + i} = index.add %kpr, %c{128 + i}k : index")
                e(f"      view.store {v}, %kspv[%kpa{16 + i}] : f32, view<{ksplit_floats()}xf32>")
            e("    }")
        if "ksstore" in R2P_DIAG:   # two-kernel split, kernel 1: a piece stores its partial and writes no output (no
            # arrival, no barrier, no atomic: a separate merge kernel reads the partials after this launch)
            e("    %kpzu = index.cast %wglp : index to i32")
            e("    %kpzi = scalar.andi %pwz0, %kpzu : i32")
            e("    %kpzf = scalar.bitcast %kpzi : i32 to f32")
            e(f"    %kpz8 = vector.splat %kpzf : {V8}")
            e("    %kpwf = index.cmp ne, %c0, %c0 : index")
            e(f"    scf.yield {', '.join(['%kpz8'] * 16 + ['%one', '%one'])}, %kpwf : {rt}, i1")
            e("  } else {")
            e("    %kpwt = index.cmp eq, %c0, %c0 : index")
            e(f"    scf.yield {', '.join(own)}, %kpwt : {rt}, i1")
            e("  }")
            return
        if ksm:   # kernel 2: kernel 1's stores are complete (the launch boundary), this workgroup's P.V waves merge.
            # Only them: a QK lane aliases a P.V lane's record (tid % 256), and its merged stores over piece 0 raced the
            # P.V wave's loads of it (fragments merged twice, per wave and fragment; measured on the chain run)
            e("    %kplast = scalar.andi %isB, %isB : i1")
        # gen_rot.last_arriver: release barrier, lane 0's device add (monotonic: last when old % K == K - 1), acquire
        if not ksm:
            e("    kernel.barrier<global> scope(workgroup) ordering(acq_rel)")
            e("    %kpfb = index.constant 4 : offset")
            e("    %kpfp = buffer.alloca<workgroup> align(4) %kpfb : buffer")
            e("    %kpfv = buffer.view %kpfp[%base] : buffer -> view<1xi32>")
            e("    %kpt0 = index.cmp eq, %tid, %c0 : index")
            e("    scf.if %kpt0 {")
            e("      %kpone = scalar.constant 1 : i32")
            e("      %kpold = view.atomic.rmw<addi> %kpone, %kscv[%kpt] {ordering = acq_rel, scope = device} : i32, view<64xi32> -> i32")
            e(f"      %kpkk = scalar.constant {K} : i32")
            e("      %kpm = scalar.remui %kpold, %kpkk : i32")
            e("      view.store %kpm, %kpfv[%c0] : i32, view<1xi32>")
            e("    }")
            e("    kernel.barrier<global> scope(workgroup) ordering(acquire)")
            e("    %kpmv = view.load %kpfv[%c0] : view<1xi32> -> i32")
            e(f"    %kpkl = scalar.constant {K - 1} : i32")
            e("    %kplast = scalar.cmpi eq, %kpmv, %kpkl : i32")
        if "ksnolast" in R2P_DIAG:   # diag (compile bisect): stores + arrival, no merge
            e(f"    scf.yield {', '.join(['%zeros8'] * 16 + ['%one', '%one'])}, %kplast : {rt}, i1")
            e("  } else {")
            e("    %kpwt = index.cmp eq, %c0, %c0 : index")
            e(f"    scf.yield {', '.join(own)}, %kpwt : {rt}, i1")
            e("  }")
            return
        e(f"    {', '.join(o + 'm' for o in outs[:18])} = scf.if %kplast -> ({rt}) {{")
        # piece i of tail unit t: q = i x tail + t
        for i in range(K):
            e(f"      %kqi{i} = index.constant {i} : index")
            e(f"      %kqx{i} = index.mul %kqi{i}, %kstd : index")
            e(f"      %kqq{i} = index.add %kqx{i}, %kpt : index")
            e(f"      %kqb{i} = index.mul %kqq{i}, %kprp : index")
            e(f"      %kqr{i} = index.add %kqb{i}, %kpr0 : index")
            for j, nm in enumerate(("l", "lo", "m", "mo")):
                e(f"      %kqa{nm}{i} = index.add %kqr{i}, %c{128 + j}k : index")
                e(f"      %kq{nm}{i} = view.load %kspv[%kqa{nm}{i}] : view<{ksplit_floats()}xf32> -> f32")
        fac = {}
        for nm, ln in (("m", "l"), ("mo", "lo")):
            cur = f"%kq{nm}0"
            for i in range(1, K):
                e(f"      %kqM{nm}{i} = scalar.maxnumf {cur}, %kq{nm}{i} : f32")
                cur = f"%kqM{nm}{i}"
            fs = []
            for i in range(K):
                e(f"      %kqd{nm}{i} = scalar.subf %kq{nm}{i}, {cur} : f32")
                e(f"      %kqf{nm}{i} = scalar.exp2f<afn> %kqd{nm}{i} : f32")
                fs.append(f"%kqf{nm}{i}")
            e(f"      %kqs{ln}0 = scalar.mulf %kq{ln}0, {fs[0]} : f32")
            for i in range(1, K):
                e(f"      %kqs{ln}{i} = scalar.fmaf %kq{ln}{i}, {fs[i]}, %kqs{ln}{i - 1} : f32")
            fac[nm] = fs
        # streamed: one fragment at a time in scalar f32 (no factor splats), the merged fragment over piece 0's
        # (this lane's own record: same-lane store then load), then all 16 reloaded for the epilogue
        for f in range(16):
            fv = fac["m" if f < 8 else "mo"]
            for i in range(K):
                e(f"      %kqo{f}a{i} = index.add %kqr{i}, %c{8 * f}k : index")
                e(f"      %kqo{f}v{i} = vector.load %kspv[%kqo{f}a{i}] : view<{ksplit_floats()}xf32> -> {V8}")
            els = []
            for x in range(8):
                for i in range(K):
                    e(f"      %kqe{f}_{x}_{i} = vector.extract %kqo{f}v{i}[{x}] : {V8} -> f32")
                e(f"      %kqt{f}_{x}_0 = scalar.mulf %kqe{f}_{x}_0, {fv[0]} : f32")
                for i in range(1, K):
                    e(f"      %kqt{f}_{x}_{i} = scalar.fmaf %kqe{f}_{x}_{i}, {fv[i]}, %kqt{f}_{x}_{i - 1} : f32")
                els.append(f"%kqt{f}_{x}_{K - 1}")
            e(f"      %kqw{f} = vector.from_elements {', '.join(els)} : {V8}")
            e(f"      vector.store %kqw{f}, %kspv[%kqo{f}a0] : {V8}, view<{ksplit_floats()}xf32>")
            e("      scf.schedule.fence")
        res = []
        for f in range(16):
            e(f"      %kqg{f} = vector.load %kspv[%kqo{f}a0] : view<{ksplit_floats()}xf32> -> {V8}")
            res.append(f"%kqg{f}")
        res += [f"%kqsl{K - 1}", f"%kqslo{K - 1}"]
        e(f"      scf.yield {', '.join(res)} : {rt}")
        e("    } else {")   # not the last: never written (zeros, so the own partial is dead after its stores)
        # an opaque unit-variant zero (queue word 2 & the unit id), not %zeros8: sharing that constant with the QK
        # MMAs' zero init kept one 8-VGPR zero live across the tile loops (211 VGPRs)
        e("      %kpzu = index.cast %wglp : index to i32")
        e("      %kpzi = scalar.andi %pwz0, %kpzu : i32")
        e("      %kpzf = scalar.bitcast %kpzi : i32 to f32")
        e(f"      %kpz8 = vector.splat %kpzf : {V8}")
        e(f"      scf.yield {', '.join(['%kpz8'] * 16 + ['%one', '%one'])} : {rt}")
        e("    }")
        e("    %kptrue = index.cmp eq, %c0, %c0 : index")
        e(f"    %kpwm = scalar.andi %kplast, %kptrue : i1")
        e(f"    scf.yield {', '.join(o + 'm' for o in outs[:18])}, %kpwm : {rt}, i1")
        e("  } else {")
        e("    %kpwt = index.cmp eq, %c0, %c0 : index")
        e(f"    scf.yield {', '.join(own)}, %kpwt : {rt}, i1")
        e("  }")

    if R2P and R2P_K2:
        k2_layout()
    STEER = PERSIST and PERSIST_DYN and R2P and R2P_STEER
    KS = bool(KSPLIT and R2P and PERSIST and PERSIST_DYN)
    if KS:
        assert STEER and R2P_SPLIT and R2P_PEEL and not (R2P_QPF or KRAW or SOFT1 or "tailfill" in R2P_DIAG)
    PVAL_LINE = ("  %pnext, %pwn = scf.if %pval -> (index, i32) {" if STEER else
                 "  %pnext = scf.if %pval -> (index) {" if PERSIST_DYN else "  scf.if %pval {")
    punq = "%punq" if ("tailfill" in R2P_DIAG and PERSIST_DYN) or KS else "%pun"   # the queue's bound

    def persist_issue(t):
        """PERSIST_DYN: every lane's atomic add, %pq<t>o: lane 0 adds 1 to the counter (word 0), the others add 0 to a word
        of their own (1 + tid, always zero). No branch, so its return is waited for only at its use. The unit's own
        stores must not be in flight: the atomic waits for them (vscnt) first."""
        e(f"  %pq{t}tid = kernel.workitem.id<x> : index")
        e(f"  %pq{t}t0 = index.cmp eq, %pq{t}tid, %c0 : index")
        if NT > 256:   # lanes t, t + 256 share a zero word (PQ_WORDS stays 264)
            e(f"  %pq{t}tr = index.rem %pq{t}tid, %c256 : index")
        e(f"  %pq{t}w0 = index.add %pq{t}{'tr' if NT > 256 else 'tid'}, %c1 : index")
        e(f"  %pq{t}w = scf.select %pq{t}t0, %c0, %pq{t}w0 : index")
        e(f"  %pq{t}d = scf.select %pq{t}t0, %pq_one, %pq_zero : i32")
        e(f"  %pq{t}o = view.atomic.rmw<addi> %pq{t}d, %pqv[%pq{t}w] {{ordering = relaxed, scope = device}} : i32, view<{PQ_WORDS}xi32> -> i32")

    def persist_finish(t):
        """PERSIST_DYN: %pq<t>u = the fetched unit id, workgroup uniform (a workgroup max), in an SGPR; the fetch of id
        units + P - 1 is the launch's last: its lane 0 resets the counter"""
        e(f"  %pq{t}r = kernel.workgroup.reduce<maxui> %pq{t}o : i32")
        e(f"  %pq{t}s = kernel.subgroup.broadcast.first %pq{t}r : i32")   # an SGPR: the unit's index math stays scalar
        e(f"  %pq{t}l = scalar.cmpi eq, %pq{t}s, %pq_last : i32")
        e(f"  %pq{t}lx = scalar.andi %pq{t}l, %pq{t}t0 : i1")
        e(f"  scf.if %pq{t}lx {{")
        e(f"    %pq{t}x = view.atomic.rmw<xchgi> %pq_zero, %pqv[%c0] {{ordering = relaxed, scope = device}} : i32, view<{PQ_WORDS}xi32> -> i32")
        e("  }")
        e(f"  %pq{t}u = index.cast %pq{t}s : i32 to index")
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
    if PERSIST:
        assert not (KVMAJOR or OCNT or (PERSIST_REV and QB_ZIGZAG)) and (PERSIST_DYN or not KVG)
        assert not (PERSIST_DYN and (PERSIST_REV or QB_ZIGZAG))
        e(f"  %cpst = index.constant {PERSIST} : index")
        e("  kernel.launch.config workgroups(%cpst, %c1, %c1) workgroup_size(%cnt, %c1, %c1) : index")
    else:
        e("  kernel.launch.config workgroups(%qblocks, %pairs, %c1) workgroup_size(%cnt, %c1, %c1) : index")
    e("} launch(%query: buffer, %gate: buffer, %key_cache: buffer, %value_cache: buffer, %output: buffer, %lse: buffer"
      + (", %kscale: buffer" if KDEC else "")
      + (", %vstat4: buffer" if VQ4 or VQ8 else "") + (", %ptab: buffer" if PAGED else "")
      + (", %qnw: buffer, %qeps: buffer" if QNORM else "") + (", %kscale: buffer" if KRAW else "")
      + (", %ocnt: buffer" if OCNT else "") + (", %pqueue: buffer" if PERSIST and PERSIST_DYN else "")
      + (", %ksplit: buffer" if KS else "") + ") {")
    e("  %base = index.constant 0 : offset")
    for v in (0, 1, 2, 3, 4, 6, 8, 15, 16, 31, 32, 64, 128, 256, 1024, 4096, 6144):
        e(f"  %c{v} = index.constant {v} : index")
    e(f"  %chpw = index.constant {HPW} : index")
    e(f"  %cqt = index.constant {QT} : index")
    e(f"  %cqt1 = index.constant {QT - 1} : index")
    e(f"  %cnt = index.constant {NT} : index")
    if R2P and NQB == 8:
        e("  %c512 = index.constant 512 : index")
        e("  %c2048 = index.constant 2048 : index")
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
    CAPV = "%cache_capacity"
    if PAGED:
        e("  %c255p = index.constant 255 : index")
        e("  %npg0 = index.add %cache_capacity, %c255p : index")
        e("  %npages = index.div %npg0, %c256 : index")
        e("  %pcap = index.mul %npages, %c256 : index")       # pool rows: whole pages
        CAPV = "%pcap"
    e(f"  %kvtot = index.mul {CAPV}, %c1024 : index")
    if KRAW:
        assert R2P and PAGED and QNORM
        e("  %ksc_na = buffer.assume.noalias %kscale : buffer")
        e("  %kscn = index.mul %pcap, %c8 : index")
        e("  %ksc_view = buffer.view %ksc_na[%base] : buffer -> view<[%kscn]xf16>")
    e("  %q_na, %g_na, %k_na, %v_na, %o_na = buffer.assume.noalias %query, %gate, %key_cache, %value_cache, %output : buffer, buffer, buffer, buffer, buffer")
    if OCNT:   # without it the atomic may alias every input: their scalar loads become vector loads, the code shifts
        e("  %oc_na = buffer.assume.noalias %ocnt : buffer")
    # (diag q16: f16 q only, the gate as configured: the Q bytes' share of the prologue; wrong values)
    q16 = Q16 and (not QNORM or QG16 or "q16" in R2P_DIAG or R2P_QSPLIT)
    e(f"  %q_flat = buffer.view %q_na[%base] : buffer -> view<[%qtot]x{'f16' if q16 else 'f32'}>")
    if R2P_QSPLIT:
        e("  %qsp_c1536 = index.constant 1536 : index")
        e("  %qsp_c48 = index.constant 384 : index")
        e("  %qsp_c12288 = index.constant 12288 : index")
        e("  %qsp_c18432 = index.constant 18432 : index")
        e("  %qrtot = index.mul %B, %qsp_c1536 : index")
        e("  %qsstot = index.mul %B, %qsp_c48 : index")
        e("  %qrb_oi = index.mul %B, %qsp_c12288 : index")
        e("  %qrb_o = index.cast %qrb_oi : index to offset")
        e("  %qss_oi = index.mul %B, %qsp_c18432 : index")
        e("  %qss_o = index.cast %qss_oi : index to offset")
        e("  %qr_flat = buffer.view %q_na[%qrb_o] : buffer -> view<[%qrtot]xf32>")
        e("  %qs_flat = buffer.view %q_na[%qss_o] : buffer -> view<[%qsstot]xf32>")
    e(f"  %g_flat = buffer.view %g_na[%base] : buffer -> view<[%qtot]x{'f16' if QG16 or GSIG16 else 'f32'}>")
    e("  %o_flat = buffer.view %o_na[%base] : buffer -> view<[%qtot]xf16>")
    if "stamp" in R2P_DIAG:   # diag: progress records in the gate buffer (wrong gate; see the QK loop)
        e("  %st_v = buffer.view %g_na[%base] : buffer -> view<262144xi32>")
        e("  %c1024st = index.constant 1024 : index")
        e("  %c3st = index.constant 3 : index")
    # kv4a16: int4 K [token][128 dwords], f16x2 (16 s, lo - 16 s) [token][32]
    # kv8a16: int8 K [token][256 dwords], f16x2 (256 s, -384 s) [token][8]
    if KDEC:
        e(f"  %kq32tot = index.mul {CAPV}, %c{128 if K4 else 256} : index")
        e(f"  %kstot = index.mul {CAPV}, %c{32 if K4 else 8} : index")
        e("  %k_flat = buffer.view %k_na[%base] : buffer -> view<[%kq32tot]xi32>")
        e("  %ks_na = buffer.assume.noalias %kscale : buffer")
        e("  %ks_flat = buffer.view %ks_na[%base] : buffer -> view<[%kstot]xi32>")
    else:
        e("  %k_flat = buffer.view %k_na[%base] : buffer -> view<[%kvtot]xf16>")
    if PAGED:
        e("  %pt_na = buffer.assume.noalias %ptab : buffer")
        e("  %pt_flat = buffer.view %pt_na[%base] : buffer -> view<[%npages]xi32>")
    e(f"  %cap15 = index.add {CAPV}, %c15 : index")
    e("  %vtiles = index.div %cap15, %c16 : index")
    e("  %vpitch = index.mul %vtiles, %c16 : index")
    e("  %vtot = index.mul %vpitch, %c1024 : index")
    e("  %vlast = index.sub %vpitch, %c16 : index")
    if VQ4 or VQ8:   # V^T [kvh][tile][256 dims] x 2 (nibbles) or 4 (bytes) dwords, stats 1 dword
        assert not (VQ4 and VQ8)
        e("  %vq4tot0 = index.mul %vtiles, %c1024 : index")
        e(f"  %vq4tot = index.mul %vq4tot0, %c{2 if VQ4 else 4} : index")
        e("  %v_flat = buffer.view %v_na[%base] : buffer -> view<[%vq4tot]xi32>")
        e("  %vs4_na = buffer.assume.noalias %vstat4 : buffer")
        e("  %vs4_flat = buffer.view %vs4_na[%base] : buffer -> view<[%vq4tot0]xi32>")
    else:
        e("  %v_flat = buffer.view %v_na[%base] : buffer -> view<[%vtot]xf16>")
    pool = S_OFF + NT * 32 if (Q16 if QDIRECT is None else QDIRECT) else POOL      # QDIRECT: no Q stage / drain slot
    pool = max(pool, K2_OFF + KT * KT_PITCH * 2) if PIPE2 else pool
    if R2P:    # S partials 2 planes x 4 blocks x 32 lanes x 16 B, P^T 2 slots x 4 blocks x 512 B, alpha 2 x 256 B; the
        # final row sums alias the partials (post-loop), the QNORM sums P^T slot 0 (prologue: first written in tile 0's
        # phase 2, after barrier A)
        assert (NQB == 4 or (NQB == 8 and HPW == 2 and R2P_K2)) and not (SOFT1 or PIPE2 or KDEC or VQ4 or VQ8)
        pool = S_OFF + 4096 + 4096 + 512
        if R2P_K2:   # P^T 2 slots (= S plane 0) 1024 NQB, S plane 1 512 NQB, alpha 2 slots 128 NQB
            pool = S_OFF + 1664 * NQB
        if NQB == 8:   # the prologue's Q exchange: 256 rows x 16 B per role wave from the second K slot
            pool = max(pool, 16 * KT_PITCH * 2 + 4096 * NQB)
    global PPV, ALV
    nsl = 1
    PPV, ALV = f"view<{64 * NQB * nsl}x8xf16>", f"view<{16 * NQB * nsl}xf32>"
    if SOFT1:  # S partials (B's), then P^T 4 pairs x 512 B, alpha 256 B, row sums 256 B (x2 slots under SOFT3)
        pool = S_OFF + NT * 32 + (4096 + 768 if SOFT3 or SOFT4 else 2048 + 512)
    global QX_OFF
    global QW_OFF
    if QNORM and R2P and R2P_QPF and R2P_QPFW:   # the norm weights' LDS copy (256 f32). (A pool of 32 KB costs occupancy: the
        # compiler's VGPR budget rises past 192 at once)
        QW_OFF = pool
        pool += 1024
    if QNORM and R2P and NQB == 8:   # past the Q exchange (the sums are read while other waves store into it)
        QX_OFF = pool
        pool += NQB * 2 * 16 * 4
    elif QNORM and R2P:
        QX_OFF = S_OFF + 4096
    elif QNORM:   # QNORM: the head-dim halves' sums of squares, [query block][half][row] f32
        QX_OFF = pool
        pool += NQB * 2 * 16 * 4
    global KSL_OFF
    if KRAW:   # the K scales' LDS slots: [K slot][128-dim half][16 keys] f32 (as K, staged a tile ahead)
        KSL_OFF = (pool + 15) // 16 * 16
        pool = KSL_OFF + 256
    if "occ3" in R2P_DIAG:   # (diag: the 3-workgroup LDS floor, to separate occupancy from other effects)
        pool = max(pool, 33792)
    assert pool <= 65536
    e(f"  %pool_bytes = index.constant {pool} : offset")
    e("  %pool = buffer.alloca<workgroup> align(16) %pool_bytes : buffer")
    e(f"  %qs_view = buffer.view %pool[%base] : buffer -> view<{NQB * 16}x{Q_PITCH}xf16>")
    if KRAW:
        e(f"  %ksl_o = index.constant {KSL_OFF} : offset")
        e("  %ksl_view = buffer.view %pool[%ksl_o] : buffer -> view<64xf32>")
        e("  %ksl_o2 = index.constant " + str(KSL_OFF + 128) + " : offset")   # KH: the slots' second copy (a separate
        e("  %ksl_view2 = buffer.view %pool[%ksl_o2] : buffer -> view<32xf32>")    # load after barrier A, not CSEd)
    e(f"  %q_lay = encoding.layout.strided [1, {Q_PITCH}] : encoding<layout>")
    e(f"  %q_fr = buffer.view %pool[%base] : buffer -> view<256x{NQB * 16}xf16, %q_lay>")
    e(f"  %k_o = index.constant {K_OFF} : offset")
    e(f"  %k_view = buffer.view %pool[%k_o] : buffer -> view<{KT}x{KT_PITCH}xf16>")
    e(f"  %v_o = index.constant {V_OFF} : offset")
    e(f"  %v_view = buffer.view %pool[%v_o] : buffer -> view<256x{VT_PITCH}xf16>")
    e(f"  %s_o = index.constant {S_OFF} : offset")
    # two planes of 4 f32 per lane, so each b128 access is contiguous across lanes
    e(f"  %s_view = buffer.view %pool[%s_o] : buffer -> view<{2 * NT}x4xf32>")
    e(f"  %s8_view = buffer.view %pool[%s_o] : buffer -> view<{NT}x8xf32>")
    PRS = 256 if (SOFT3 or SOFT4) else 128   # SOFT1 P^T rows (2 slots under SOFT3)
    PRS2 = PRS // 2
    if SOFT1:
        PR = 256 if (SOFT3 or SOFT4) else 128
        e(f"  %pp_o = index.constant {S_OFF + NT * 32} : offset")
        e(f"  %pp_view = buffer.view %pool[%pp_o] : buffer -> view<{PR}x8xf16>")
        e(f"  %al_o = index.constant {S_OFF + NT * 32 + PR * 16} : offset")
        e(f"  %al_view = buffer.view %pool[%al_o] : buffer -> view<{PR // 2}xf32>")
        e(f"  %ls_o = index.constant {S_OFF + NT * 32 + PR * 16 + PR * 2} : offset")
        e("  %ls_view = buffer.view %pool[%ls_o] : buffer -> view<64xf32>")
    if R2P and R2P_K2:
        e(f"  %xs_view = buffer.view %pool[%s_o] : buffer -> view<{96 * NQB}x4xf32>")
        e("  %pp_o = index.constant {S_OFF} : offset".replace("{S_OFF}", str(S_OFF)))
        e(f"  %pp_view = buffer.view %pool[%pp_o] : buffer -> view<{64 * NQB}x8xf16>")
        e(f"  %al_o = index.constant {S_OFF + 1536 * NQB} : offset")
        e(f"  %al_view = buffer.view %pool[%al_o] : buffer -> view<{32 * NQB}xf32>")
        e(f"  %ls_o = index.constant {S_OFF + 1024 * NQB} : offset")
        e(f"  %ls_view = buffer.view %pool[%ls_o] : buffer -> view<{16 * NQB}xf32>")
        if KS:   # the row maxes x log2e for the merge: the K slot (idle from the last tile's barrier B to the next unit)
            e(f"  %lm_view = buffer.view %pool[%base] : buffer -> view<{16 * NQB}xf32>")
        e(f"  %k2_view = buffer.view %pool[%k_o] : buffer -> view<32x{KT_PITCH}xf16>")
        e(f"  %xsp0_view = buffer.view %pool[%pp_o] : buffer -> view<{32 * NQB}x4xf32>")   # S plane 0 in P^T slot 0 / 1
        e(f"  %xsp1_o = index.constant {S_OFF + 512 * NQB} : offset")
        e(f"  %xsp1_view = buffer.view %pool[%xsp1_o] : buffer -> view<{32 * NQB}x4xf32>")
    elif R2P:
        e("  %xs_view = buffer.view %pool[%s_o] : buffer -> view<256x4xf32>")
        e(f"  %pp_o = index.constant {S_OFF + 4096} : offset")
        e("  %pp_view = buffer.view %pool[%pp_o] : buffer -> view<256x8xf16>")
        e(f"  %al_o = index.constant {S_OFF + 8192} : offset")
        e("  %al_view = buffer.view %pool[%al_o] : buffer -> view<128xf32>")
        e(f"  %ls_o = index.constant {S_OFF} : offset")
        e("  %ls_view = buffer.view %pool[%ls_o] : buffer -> view<64xf32>")
    # ids: head pair fastest, longest query blocks first
    if PERSIST:
        e("  %nh_ = config.get @attention_prefill.num_heads : index")
        e("  %npairs = index.div %nh_, %chpw : index")
        e("  %tpq = index.add %B, %cqt1 : index")
        e("  %nqb = index.div %tpq, %cqt : index")
        e("  %pun0 = index.mul %nqb, %npairs : index")
        if "halfunits" in R2P_DIAG:   # diag (with unilen): half the units, each twice as long
            e("  %pun = index.div %pun0, %c2 : index")
        else:
            e("  %pun = index.add %pun0, %c0 : index")              # units
        e(f"  %pP = index.constant {PERSIST} : index")
        e(f"  %pP1 = index.constant {PERSIST - 1} : index")
        e("  %prn0 = index.add %pun, %pP1 : index")
        e("  %prn = index.div %prn0, %pP : index")                 # rounds
        if "tailfill" in R2P_DIAG and PERSIST_DYN:   # diag: the queue runs whole rounds, ids past the units wrap (same stores)
            e("  %punq = index.mul %prn, %pP : index")
        if KS:   # ids < %ksw: whole units; then %kst tail units x KSPLIT pieces, piece-major
            e("  %kst0 = index.rem %pun, %pP : index")
            e(f"  %ksmin = index.constant {KSPLIT_MIN} : index")
            e("  %kson = index.cmp uge, %start_pos, %ksmin : index")
            e("  %kst = scf.select %kson, %kst0, %c0 : index")
            e("  %ksw = index.sub %pun, %kst : index")
            e(f"  %kskc = index.constant {KSPLIT} : index")
            e("  %kspn = index.mul %kst, %kskc : index")
            if "ksmerge" in R2P_DIAG:   # two-kernel split, kernel 2: the queue hands out the tail units only
                e("  %punq = index.add %kst, %c0 : index")
            else:
                e("  %punq = index.add %ksw, %kspn : index")
            e("  %kstd = index.max %kst, %c1 : index")
        e("  %pslot = kernel.workgroup.id<x> : index")
        e("  %pslotr = index.sub %pP1, %pslot : index")
    if KS:
        e("  %ks_na = buffer.assume.noalias %ksplit : buffer")
        e("  %kscv = buffer.view %ks_na[%base] : buffer -> view<64xi32>")
        e("  %ksp_o = index.constant 256 : offset")
        e(f"  %kspv = buffer.view %ks_na[%ksp_o] : buffer -> view<{ksplit_floats()}xf32>")
        for v in sorted({132, 128, 129, 130, 131} | {8 * f for f in range(16)}):
            e(f"  %c{v}k = index.constant {v} : index")
    if PERSIST and PERSIST_DYN:
        e("  %pq_na = buffer.assume.noalias %pqueue : buffer")   # else the atomic may alias every input (vector loads)
        e(f"  %pqv = buffer.view %pq_na[%base] : buffer -> view<{PQ_WORDS}xi32>")
        e("  %pq_one = scalar.constant 1 : i32")
        e("  %pq_zero = scalar.constant 0 : i32")
        e(f"  %pql0 = index.add {punq}, %pP1 : index")
        e("  %pq_last = index.cast %pql0 : index to i32")
        persist_issue("i")
        persist_finish("i")
        if STEER:   # the steering word (R2P_STEER) and its opaque zero
            e(f"  %pwz0 = view.load %pqv[%c2] : view<{PQ_WORDS}xi32> -> i32")   # (queue word 2: always zero)
            e("  %pwi = scalar.constant 0 : i32")
            e("  %pfin, %pwf = scf.while(%pcur = %pqiu : index, %pwc = %pwi : i32) -> (index, i32) {")
        else:
            e("  %pfin = scf.while(%pcur = %pqiu : index) -> (index) {")
        e("  %wglp = index.add %pcur, %c0 : index")
    elif PERSIST:
        e("  scf.for %pr = [%c0 to %prn step %c1] {")
    if PERSIST and not PERSIST_DYN:
        if PERSIST_REV:   # odd groups of 20 slots: round prn - 1 - pr
            e("  %pc20 = index.constant 20 : index")
            e("  %pgr = index.div %pslot, %pc20 : index")
            e("  %pgp = index.rem %pgr, %c2 : index")
            e("  %pgo = index.cmp ne, %pgp, %c0 : index")
            e("  %prn1 = index.sub %prn, %c1 : index")
            e("  %prr = index.sub %prn1, %pr : index")
            e("  %prx = scf.select %pgo, %prr, %pr : index")
        else:
            e("  %prx = index.add %pr, %c0 : index")
        e("  %prp = index.rem %prx, %c2 : index")
        e("  %prev = index.cmp eq, %prp, %c0 : index")
        e("  %psl = scf.select %prev, %pslot, %pslotr : index")    # snake: odd rounds deal the slots in reverse
        e("  %prb = index.mul %prx, %pP : index")
        e("  %wglp = index.add %prb, %psl : index")
    if PERSIST:
        e(f"  %pval = index.cmp ult, %wglp, {punq} : index")
        e(PVAL_LINE)
        # the previous unit's LDS readers (epilogue row sums, P^T / V^T slots) are done before this unit stages
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        if "tailfill" in R2P_DIAG and PERSIST_DYN:
            e("  %wglw = index.rem %wglp, %pun : index")
            e("  %wgl = index.assume %wglw [range(%wglw, 0, 1048576), lt(%wglw, %pun)] : index")
        elif KS:   # a piece: tail unit %kpt (id %ksw + %kpt), key piece %kpj
            if "ksmerge" in R2P_DIAG:   # kernel 2: id t = tail unit t's merge (piece 0's slot, every id a piece)
                e("  %kpis = index.cmp ult, %wglp, %punq : index")
                e("  %kpq = index.add %wglp, %c0 : index")
            else:
                e("  %kpis = index.cmp uge, %wglp, %ksw : index")
                e("  %kpc = index.max %wglp, %ksw : index")
                e("  %kpq = index.sub %kpc, %ksw : index")
            e("  %kpj = index.div %kpq, %kstd : index")
            e("  %kpt = index.rem %kpq, %kstd : index")
            e("  %kpu = index.add %ksw, %kpt : index")
            e("  %wglk = scf.select %kpis, %kpu, %wglp : index")
            e("  %wgl = index.assume %wglk [range(%wglk, 0, 1048576), lt(%wglk, %pun)] : index")
        else:
            e("  %wgl = index.assume %wglp [range(%wglp, 0, 1048576), lt(%wglp, %pun)] : index")
    else:
        e("  %wgx = kernel.workgroup.id<x> : index")
        e("  %wgy = kernel.workgroup.id<y> : index")
        e("  %nh_ = config.get @attention_prefill.num_heads : index")
        e("  %npairs = index.div %nh_, %chpw : index")
        e("  %tpq = index.add %B, %cqt1 : index")
        e("  %nqb = index.div %tpq, %cqt : index")
        e("  %wgl0 = index.mul %wgy, %nqb : index")
        e("  %wgl = index.add %wgl0, %wgx : index")
    if KVG:
        e("  %nkvh_ = config.get @attention_prefill.num_kv_heads : index")
        e("  %pkv = index.div %npairs, %nkvh_ : index")
        e(f"  %kvg = index.constant {KVG} : index")
        e("  %kvblk0 = index.mul %pkv, %nqb : index")
        e("  %kvblk = index.mul %kvblk0, %kvg : index")
        e("  %kvgi = index.div %wgl, %kvblk : index")
        e("  %kvr0 = index.rem %wgl, %kvblk : index")
        e("  %kvin = index.rem %kvr0, %kvg : index")
        e("  %kvr = index.div %kvr0, %kvg : index")
        e("  %kvhb = index.mul %kvgi, %kvg : index")
        e("  %kvh_m = index.add %kvhb, %kvin : index")
        e("  %pig_m = index.rem %kvr, %pkv : index")
        e("  %qb0 = index.div %kvr, %pkv : index")
        e("  %hp0 = index.mul %kvh_m, %pkv : index")
        e("  %hp = index.add %hp0, %pig_m : index")
    elif KVMAJOR:   # KV head slowest: co-resident workgroups stream the same K / V (one KV head's pairs fastest)
        e("  %nkvh_ = config.get @attention_prefill.num_kv_heads : index")
        e("  %pkv = index.div %npairs, %nkvh_ : index")
        e("  %kvblk = index.mul %pkv, %nqb : index")
        e("  %kvh_m = index.div %wgl, %kvblk : index")
        e("  %kvr = index.rem %wgl, %kvblk : index")
        e("  %pig_m = index.rem %kvr, %pkv : index")
        e("  %qb0 = index.div %kvr, %pkv : index")
        e("  %hp0 = index.mul %kvh_m, %pkv : index")
        e("  %hp = index.add %hp0, %pig_m : index")
    else:
        e("  %hp = index.rem %wgl, %npairs : index")
        e("  %qb0 = index.div %wgl, %npairs : index")
    e("  %nqb1 = index.sub %nqb, %c1 : index")
    if QB_ZIGZAG or (PERSIST and PERSIST_DYN):   # launch order long, short, long, ... (the short blocks' Q / gate / O
        # traffic spread over the run)
        e("  %qzh = index.div %qb0, %c2 : index")
        e("  %qzo = index.rem %qb0, %c2 : index")
        e("  %qzl = index.sub %nqb1, %qzh : index")
        e("  %qzodd = index.cmp ne, %qzo, %c0 : index")
        if QB_ZIGZAG:
            e("  %qb = scf.select %qzodd, %qzh, %qzl : index")
        else:
            e("  %qbz = scf.select %qzodd, %qzh, %qzl : index")
            e("  %qbl = index.sub %nqb1, %qb0 : index")
            e(f"  %pqzzc = index.constant {PDYN_ZZ_BELOW} : index")
            e("  %pqzz = index.cmp ult, %start_pos, %pqzzc : index")
            e("  %qb = scf.select %pqzz, %qbz, %qbl : index")
    else:
        e("  %qb = index.sub %nqb1, %qb0 : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %wave = index.div %tid, %c32 : index")
    e("  %lane = index.rem %tid, %c32 : index")
    e("  %sub = index.rem %lane, %c16 : index")
    if PERSIST and PERSIST_DYN:
        for i in range(R2P_VPAD):   # words 1 + 8 i + (lane & 7) <= 128: zero
            e(f"  %vpl{i} = index.rem %lane, %c8 : index")
            e(f"  %vpo{i} = index.constant {1 + 8 * i} : index")
            e(f"  %vpw{i} = index.add %vpl{i}, %vpo{i} : index")
            e(f"  %vpv{i} = view.load %pqv[%vpw{i}] : view<{PQ_WORDS}xi32> -> i32")
    e("  %half = index.div %lane, %c16 : index")
    if SOFT1 or R2P:   # the role from the subgroup id: wave-uniform to Loom, so the role branch is a scalar branch
        e(f"  %psg{''} = kernel.subgroup.id : index")
    if R2P:   # waves p, p + 4 share a SIMD: QK p, P.V p + 4; within a role: group = query half, hd = head-dim half
        # (the subgroup id lowers to tid >> 5 in a VGPR: broadcast, its role / block / staging terms are scalar)
        e("  %psgi = index.cast %psg : index to i32")
        e("  %psgb = kernel.subgroup.broadcast.first %psgi : i32")
        e("  %psgu0 = index.cast %psgb : i32 to index")
        e(f"  %psgu = index.assume %psgu0 [range(%psgu0, 0, {2 * NQB})] : index")
        e(f"  %prole = index.div %psgu, %c{NQB} : index")
        e(f"  %r2w = index.rem %psgu, %c{NQB} : index")
        e("  %hd = index.rem %r2w, %c2 : index")
        e("  %r2g = index.div %r2w, %c2 : index")
        e("  %r2g2 = index.mul %r2g, %c2 : index")
        e("  %wqb = index.add %r2g2, %hd : index")         # own block: the head hd of the group (its softmax, bank 0..7)
        e("  %r2g21 = index.add %r2g2, %c1 : index")
        e("  %wqbo = index.sub %r2g21, %hd : index")      # the other head of the group (bank 8..15)
    elif SOFT1 and SOFT1_MAP == "mod":
        e("  %wqb = index.rem %psg, %c4 : index")
        e("  %hd = index.div %psg, %c4 : index")
    elif SOFT1:
        e("  %wqb = index.div %psg, %c2 : index")
        e("  %hd = index.rem %psg, %c2 : index")
    else:
        e("  %wqb = index.div %wave, %c2 : index")       # query block 0..3
        e("  %hd = index.rem %wave, %c2 : index")        # head-dim half
    if R2P:
        e("  %isA = index.cmp eq, %prole, %c0 : index")
    if SOFT1:
        e("  %isA = index.cmp eq, %hd, %c0 : index")
        e("  %isBs = index.cmp ne, %hd, %c0 : index")
    e("  %hd128 = index.mul %hd, %c128 : index")
    if R2P and R2P_K2:   # V^T fragment lane bytes: row hd128 + (lane & 15) of a plane
        e("  %vglr = index.add %hd128, %sub : index")
        e("  %vgl = index.mul %vglr, %c16 : index")
    if QNORM:
        e("  %hd16q = index.mul %hd, %c16 : index")
    e("  %hd64 = index.mul %hd, %c64 : index")
    if SOFT1:   # only A reads the partner's (B's) partial
        e(f"  %pt0 = index.add %tid, %c{128 if SOFT1_MAP == 'mod' else 32} : index")
        e("  %ptid = index.add %pt0, %c0 : index")
    else:
        e("  %pt0 = index.add %tid, %c32 : index")
        e("  %ptid = index.sub %pt0, %hd64 : index")     # partner lane (tid ^ 32)
    e("  %tid2 = index.add %tid, %cnt : index")
    e("  %ptid2 = index.add %ptid, %cnt : index")
    e("  %h0 = index.cmp eq, %half, %c0 : index")
    if PBP:
        e("  %pbp_l15 = index.rem %lane, %c16 : index")
        e("  %pbp_lo = index.cast %pbp_l15 : index to i32")
        e("  %pbp_h16 = index.add %pbp_l15, %c16 : index")
        e("  %pbp_hi = index.cast %pbp_h16 : index to i32")
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
    if HPW == 6:   # one workgroup = a whole GQA group
        e("  %kvh = index.add %hp, %c0 : index")
        e("  %head0 = index.mul %kvh, %c6 : index")
    else:
        assert HPW == 2
        e("  %kvh = index.div %hp, %c3 : index")
        e("  %pig = index.rem %hp, %c3 : index")
        e("  %kvh6 = index.mul %kvh, %c6 : index")
        e("  %pig2 = index.mul %pig, %c2 : index")
        e("  %head0 = index.add %kvh6, %pig2 : index")
    e("  %kvbase = index.mul %kvh, %c256 : index")
    if KRAW:   # this lane's scale run: [kvh][hd][pool row] + its 8 keys (8 half)
        e("  %ksc0 = index.mul %kvh, %c2 : index")
        e("  %ksc1 = index.add %ksc0, %hd : index")
        e("  %ksc2 = index.mul %ksc1, %pcap : index")
        e("  %ksc3 = index.mul %half, %c8 : index")
        e("  %kscb = index.add %ksc2, %ksc3 : index")
        e("  %kslo0 = index.mul %hd, %c16 : index")      # QK lanes: [slot][hd][8 half] own, [slot][1 - hd][8 half]
        e("  %kslo = index.add %kslo0, %ksc3 : index")
        e("  %kslp0 = index.sub %c16, %kslo0 : index")
        e("  %kslp = index.add %kslp0, %ksc3 : index")
        e(f"  %kstj = index.rem %tid, %c{2 if KH else 4} : index")       # stager item j < 4: half j / 2, keys 8 (j % 2)
        e("  %kstd = index.div %kstj, %c2 : index")   # (KH: j < 2, half 0 = s_big)
        e("  %kstk0 = index.rem %kstj, %c2 : index")
        e("  %kstk = index.mul %kstk0, %c8 : index")
        e("  %kstg0 = index.mul %kvh, %c2 : index")
        e("  %kstg1 = index.add %kstg0, %kstd : index")
        e("  %kstg2 = index.mul %kstg1, %pcap : index")
        e("  %kstg = index.add %kstg2, %kstk : index")
        e("  %kstl0 = index.mul %kstd, %c16 : index")   # (KH: slots of 16)
        e("  %kstl = index.add %kstl0, %kstk : index")
        e("  %kscp0 = index.sub %c1, %hd : index")   # the partner wave's half
        e("  %kscp1 = index.add %ksc0, %kscp0 : index")
        e("  %kscp2 = index.mul %kscp1, %pcap : index")
        e("  %kscbp = index.add %kscp2, %ksc3 : index")
    e("  %vhb0 = index.mul %kvh, %vtiles : index")
    e("  %vhb = index.mul %vhb0, %c4096 : index")
    # V staging: lane = dim, the last 256 threads (K: the first 256)
    e(f"  %vtoff = index.constant {NT - 256} : index")
    if NT > 256 and not R2P:   # max(): threads below vtoff only stage K (their V stores are guarded) but stay in range
        e("  %vtm = index.max %tid, %vtoff : index")
        e("  %vt = index.sub %vtm, %vtoff : index")
    elif R2P:
        e("  %tidr = index.rem %tid, %c128 : index")
        e("  %vt = index.add %tidr, %c0 : index")
    else:
        e("  %vt = index.add %tid, %c0 : index")
    if VSWZ and (VSWZ_T or not TILED_OUT):   # with TILED_OUT the paged build allocates 236 VGPRs (192 without)
        if VSWZF == 0:
            e("  %vsl = index.rem %vt, %c16 : index")
            e("  %vsg = index.div %vt, %c16 : index")
            e("  %vsb = index.mul %vsg, %c16 : index")
            e("  %vsl2 = index.rem %vsl, %c2 : index")
            e("  %vsl8 = index.mul %vsl2, %c8 : index")
            e("  %vslh = index.div %vsl, %c2 : index")
            e("  %vdimo = index.add %vsl8, %vslh : index")
            e("  %vdim0 = index.add %vsb, %vdimo : index")
        elif VSWZF == 1:   # dim = (t & ~15) | ((t & 1) << 3) | ((t >> 1) & 7) on the lane id
            e("  %vsl2 = index.rem %vt, %c2 : index")
            e("  %vsl8 = index.mul %vsl2, %c8 : index")
            e("  %vsh = index.div %vt, %c2 : index")
            e("  %vsh8 = index.rem %vsh, %c8 : index")
            e("  %vsg = index.div %vt, %c16 : index")
            e("  %vsb = index.mul %vsg, %c16 : index")
            e("  %vdimo = index.add %vsl8, %vsh8 : index")
            e("  %vdim0 = index.add %vsb, %vdimo : index")
        else:              # as a permutation of the lane only: the 16-group base comes from the unswizzled t
            e("  %vsl = index.rem %vt, %c16 : index")
            e("  %vsb = index.sub %vt, %vsl : index")
            e("  %vsl2 = index.rem %vt, %c2 : index")
            e("  %vsl8 = index.mul %vsl2, %c8 : index")
            e("  %vslh = index.div %vsl, %c2 : index")
            e("  %vdimo = index.add %vsl8, %vslh : index")
            e("  %vdim0 = index.add %vsb, %vdimo : index")
        e("  %c255v = index.constant 255 : index")
        e("  %vdim = index.min %vdim0, %c255v : index")   # always in range; states it for the bound proof
    else:
        e("  %vdim = index.add %vt, %c0 : index")
    e("  %vlane = index.mul %vdim, %c16 : index")
    e("  %vhl = index.add %vhb, %vlane : index")
    if VQ4 or VQ8:
        e("  %vhb4 = index.mul %kvh, %vtiles : index")
        if VQ4:
            e("  %v4m = scalar.constant 62915520 : i32")        # 0x03c003c0
        else:
            e("  %v8m = scalar.constant 66847740 : i32")        # 0x03fc03fc
        e("  %v4g = scalar.constant 1006648320 : i32")     # 0x3c003c00
    e("  %ctx_end = index.add %start_pos, %B : index")
    e("  %qs32 = index.add %qs, %cqt : index")
    e("  %vis0 = index.add %start_pos, %qs32 : index")
    if "unilen" in R2P_DIAG:   # diag: every unit start_pos + (B + QT) / 2 keys (16-aligned): the causal total, uniform
        e("  %ulv0 = index.add %B, %cqt : index")
        e("  %ulv1 = index.div %ulv0, %c32 : index")
        e(f"  %ulv2 = index.mul %ulv1, %c{32 if 'halfunits' in R2P_DIAG else 16} : index")   # halfunits: twice as long
        e("  %ulv3 = index.add %start_pos, %ulv2 : index")
        e("  %max_vis = index.min %ctx_end, %ulv3 : index")
    elif KS:   # a piece: 32-key pairs [T j / K, T (j + 1) / K) of the unit's T
        e("  %max_visu = index.min %ctx_end, %vis0 : index")
        e("  %kvt0 = index.add %max_visu, %c31 : index")
        e("  %kvt = index.div %kvt0, %c32 : index")
        e("  %kvl0 = index.mul %kvt, %kpj : index")
        e("  %kvl1 = index.div %kvl0, %kskc : index")
        e("  %kvlo = index.mul %kvl1, %c32 : index")
        e("  %kpj1 = index.add %kpj, %c1 : index")
        e("  %kvh0 = index.mul %kvt, %kpj1 : index")
        e("  %kvh1 = index.div %kvh0, %kskc : index")
        e("  %kvh2 = index.mul %kvh1, %c32 : index")
        e("  %kvhi = index.min %kvh2, %max_visu : index")
        if "ksk0" in R2P_DIAG:   # diag (compile bisect): every unit from key 0
            e("  %kbeg = index.add %c0, %c0 : index")
        else:
            e("  %kbeg = scf.select %kpis, %kvlo, %c0 : index")
        if "ksmv" in R2P_DIAG:   # diag (compile bisect): every unit to its end
            e("  %max_vis = index.add %max_visu, %c0 : index")
        else:   # (stated: with both piece ends opaque the allocation spilled, 254 VGPRs; either one alone 187)
            if "ksmerge" in R2P_DIAG:   # kernel 2: a dummy two-tile range (keys 0..31, piece 0's start), its O unused
                e("  %max_vis0 = scf.select %kpis, %c32, %max_visu : index")
            else:
                e("  %max_vis0 = scf.select %kpis, %kvhi, %max_visu : index")
            e("  %max_visu1 = index.add %max_visu, %c1 : index")
            e("  %max_vis = index.assume %max_vis0 [range(%max_vis0, 1, 1048576), lt(%max_vis0, %max_visu1)] : index")
        # every piece has >= 32 floor(T / K) >= 32 keys (T >= 128 from KSPLIT_MIN on), a whole unit >= 17: stated so the
        # tile loops' one-tile branch folds as at tile 0 (left opaque, its join tipped the allocation into spills)
        e("  %kbeg16a = index.add %kbeg, %c16 : index")
        e("  %kbeg16 = index.assume %kbeg16a [range(%kbeg16a, 16, 1048576), lt(%kbeg16a, %max_vis)] : index")
    else:
        e("  %max_vis = index.min %ctx_end, %vis0 : index")
    e("  %B_1 = index.sub %B, %c1 : index")
    e("  %cap_1 = index.sub %cache_capacity, %c1 : index")
    if KRAW:
        e("  %cap_16 = index.sub %cache_capacity, %c16 : index")
    if KCLAMP_S:   # (only then: even an unused op perturbs the allocator)
        e("  %klast = index.sub %cache_capacity, %c16 : index")
    # this lane's query: block wqb = (head0 + wqb%2, tokens qs + (wqb/2)*16 + sub)
    e("  %wqh = index.rem %wqb, %chpw : index")
    e("  %wqt0 = index.div %wqb, %chpw : index")
    e("  %wqt1 = index.mul %wqt0, %c16 : index")
    e("  %wqt = index.add %qs, %wqt1 : index")
    e("  %r_lq = index.add %wqt, %sub : index")
    e("  %r_abs = index.add %start_pos, %r_lq : index")
    e("  %r_live = index.cmp ult, %r_lq, %B : index")
    e("  %r_head = index.add %head0, %wqh : index")
    qdirect = Q16 if QDIRECT is None else QDIRECT
    qnorm_before = qnorm_after = None
    if qdirect:
        assert Q16 and (not K4 or QROT)
        if STEER:
            e("  %pwca = index.cast %pwc : i32 to index")
            e("  %pwq = index.assume %pwca [range(%pwca, 0, 256)] : index")
            e("  %r_lqw = index.add %r_lq, %pwq : index")
            e("  %qdq = index.min %r_lqw, %B_1 : index")
        else:
            e("  %qdq = index.min %r_lq, %B_1 : index")
        if "qhot" in R2P_DIAG:   # diag: Q rows read from the first 64-row block (cache hot), RoPE positions unchanged
            e("  %qhot64 = index.constant 64 : index")
            e("  %qdqh = index.rem %qdq, %qhot64 : index")
            e("  %qdrow = index.mul %qdqh, %c6144 : index")
        elif "qspread" in R2P_DIAG:   # diag: qhot's 64 rows' lines, spread over the batch (row (r % 64) * 64)
            e("  %qsp64 = index.constant 64 : index")
            e("  %qdqs0 = index.rem %qdq, %qsp64 : index")
            e("  %qdqs1 = index.mul %qdqs0, %qsp64 : index")
            e("  %qdqs = index.min %qdqs1, %B_1 : index")
            e("  %qdrow = index.mul %qdqs, %c6144 : index")
        else:
            e("  %qdrow = index.mul %qdq, %c6144 : index")
        e("  %qdh = index.mul %r_head, %c256 : index")
        e("  %qdb0 = index.add %qdrow, %qdh : index")
        e("  %qdb = index.add %qdb0, %hd128 : index")
        if R2P:   # the group's other head (bank 8..15)
            e("  %r2ho = index.rem %wqbo, %chpw : index")
            e("  %r_head_o = index.add %head0, %r2ho : index")
            e("  %qdho = index.mul %r_head_o, %c256 : index")
            e("  %qdbo0 = index.add %qdrow, %qdho : index")
            e("  %qdbo = index.add %qdbo0, %hd128 : index")
            # QNORM block: QK waves their own head's dims, P.V waves the other head's (the 4 waves of a group cover
            # the 2 heads x 2 dim halves once, as the main kernel's 4 waves); P.V waves pass theirs on through LDS
            e("  %wqn = scf.select %isA, %wqb, %wqbo : index")
            e("  %qdbn = scf.select %isA, %qdb, %qdbo : index")
        e(f"  %zq16x = vector.concat<0> %zh8, %zh8 : {V8H}, {V8H} -> {V16H}")
        def qfrags_r2p():
            """bank 0..7: this wave's fragments (QK waves); 8..15: the P.V partner's, through the V^T + partial region
            (lanes 0..15: a fragment's 16 columns, both halves alike), bracketed by barriers"""
            bk = "vector<8xi32>"
            e(f"  %bzero = vector.bitcast %zeros8 : {V8} to {bk}")
            e(f"  %qxo = index.constant {16 * KT_PITCH * 2 if R2P_K2 else V_OFF} : offset")
            e(f"  %qx_view = buffer.view %pool[%qxo] : buffer -> view<{256 * NQB}x8xf16>")
            e("  %qxw0 = index.mul %r2w, %c256 : index")
            e("  %qxs2 = index.mul %sub, %c2 : index")
            e("  %qxw = index.add %qxw0, %qxs2 : index")
            e("  %qxlo = index.cmp ult, %lane, %c16 : index")
            e("  %qxst = index.cmp ne, %prole, %c0 : index")
            e("  %qxst1 = scalar.andi %qxst, %qxlo : i1")
            if R2P_QFOLD == 1:   # Q x f16(r log2e) of the head this wave prepared, 0 for a dead row (the select's job)
                e("  %qfz = scalar.constant 0.0 : f32")
                e("  %qfl2 = scf.select %r_live, %qsl2e, %qfz : f32")
                e("  %qfl2h = scalar.fptrunc %qfl2 : f32 to f16")
                e(f"  %qfl2v = vector.splat %qfl2h : {V16H}")
            for c in range(8):
                if R2P_QFOLD == 1:
                    e(f"  %qdz{c} = vector.mulf %qdv{c}, %qfl2v : {V16H}")
                else:
                    e(f"  %qdz{c} = scf.select %r_live, %qdv{c}, %zq16x : {V16H}")
                e(f"  %qbits{c} = vector.bitcast %qdz{c} : {V16H} to {bk}")
                e(f"  %rinit{c} = scf.select %isA, %qbits{c}, %bzero : {bk}")
            e("  scf.if %qxst1 {")
            for c in range(8):
                e(f"    %qxr{c}c = index.constant {32 * c} : index")
                e(f"    %qxr{c} = index.add %qxw, %qxr{c}c : index")
                e(f"    %qxr{c}b = index.add %qxr{c}, %c1 : index")
                e(f"    %qxh{c}a = vector.slice %qdz{c}[0] : {V16H} -> {V8H}")
                e(f"    %qxh{c}b = vector.slice %qdz{c}[8] : {V16H} -> {V8H}")
                e(f"    vector.store %qxh{c}a, %qx_view[%qxr{c}, %c0] : {V8H}, view<{256 * NQB}x8xf16>")
                e(f"    vector.store %qxh{c}b, %qx_view[%qxr{c}b, %c0] : {V8H}, view<{256 * NQB}x8xf16>")
            e("  }")
            e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            for c in range(8, 16):
                e(f"  %qxl{c}c = index.constant {32 * (c - 8)} : index")
                e(f"  %qxl{c} = index.add %qxw, %qxl{c}c : index")
                e(f"  %qxl{c}b = index.add %qxl{c}, %c1 : index")
                e(f"  %qxla{c} = vector.load %qx_view[%qxl{c}, %c0] : view<{256 * NQB}x8xf16> -> {V8H}")
                e(f"  %qxlb{c} = vector.load %qx_view[%qxl{c}b, %c0] : view<{256 * NQB}x8xf16> -> {V8H}")
                e(f"  %qdz{c} = vector.concat<0> %qxla{c}, %qxlb{c} : {V8H}, {V8H} -> {V16H}")
                e(f"  %qbits{c} = vector.bitcast %qdz{c} : {V16H} to {bk}")
                e(f"  %rinit{c} = scf.select %isA, %qbits{c}, %bzero : {bk}")
            e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")

        def qfrags():
            if R2P:
                if not QNORM:   # pre-rotated f16 Q (qrot): the fragments of the wave's QNORM block, loaded as is
                    for c in range(8):
                        e(f"  %qdc{c} = index.constant {16 * c} : index")
                        e(f"  %qda{c} = index.add %qdbn, %qdc{c} : index")
                        e(f"  %qdv{c} = vector.load %q_flat[%qda{c}] : view<[%qtot]xf16> -> {V16H}")
                return qfrags_r2p()
            for c in range(8):
                if not QNORM:
                    e(f"  %qdc{c} = index.constant {16 * c} : index")
                    e(f"  %qda{c} = index.add %qdb, %qdc{c} : index")
                    e(f"  %qdv{c} = vector.load %q_flat[%qda{c}] : view<[%qtot]xf16> -> {V16H}")
                e(f"  %qdz{c} = scf.select %r_live, %qdv{c}, %zq16x : {V16H}")
                e(f"  %qf{c} = vector.fragment<rhs> %qdz{c} shape [%k, %n] : {V16H}")
        if QNORM and R2P and "noq" in R2P_DIAG:   # diag: no per-unit Q prep (loads, norm, RoPE, exchange): Q = 0
            def qzero():   # opaque zeros (the queue's zero words 1 + 8c .. 8 + 8c, c < 16: words <= 128 < 264)
                e("  %qw_na, %qe_na = buffer.assume.noalias %qnw, %qeps : buffer, buffer")   # unused (harness layout)
                e("  %qw_flat = buffer.view %qw_na[%base] : buffer -> view<[%c256]xf32>")
                e("  %qe_flat = buffer.view %qe_na[%base] : buffer -> view<[%c1]xf32>")
                for c in range(16):
                    e(f"  %qzw{c} = index.constant {1 + 8 * c} : index")
                    e(f"  %rinit{c} = vector.load %pqv[%qzw{c}] : view<{PQ_WORDS}xi32> -> vector<8xi32>")
                if R2P_QSPLIT:   # (r = 1)
                    e("  %qsl2e = scalar.constant 1.4426950408889634 : f32")
                    e(f"  %qslzth = scalar.constant {R2P_LAZY} : f32")
            qnorm_after = qzero
        elif QNORM and R2P and R2P_QSPLIT and R2P_DIAG & {"qdirect", "qdirz"} and not R2P_QFOLD:
            # diag (Q-from-kqg bound): QK waves load their head's 16 fragments straight from the f16 store (dims 0..63
            # unroped: wrong values, in bounds), P.V waves none; sums and r log2e as QSPLIT; no RoPE, trade, exchange
            e("  %qsp_c384 = index.constant 384 : index")
            e("  %qsp_c16 = index.constant 16 : index")
            e("  %qst = index.mul %qdq, %qsp_c384 : index")
            e("  %qsh = index.mul %r_head, %qsp_c16 : index")
            e("  %qsbase = index.add %qst, %qsh : index")   # (token * 24 + head) * 16
            for x in range(4):
                e(f"  %qsbc{x} = index.constant {4 * x} : index")
                e(f"  %qsbase{x} = index.add %qsbase, %qsbc{x} : index")
                e(f"  %qssl{x} = vector.load %qs_flat[%qsbase{x}] : view<[%qsstot]xf32> -> {V4}")
            tys = ", ".join([V16H] * 16)
            e(f"  {', '.join(f'%qdv{c}' for c in range(16))} = scf.if %isA -> ({tys}) {{")
            for c in range(16):
                e(f"    %qdc{c} = index.constant {16 * c} : index")
                e(f"    %qda{c} = index.add %qdb0, %qdc{c} : index")
                e(f"    %qdl{c} = vector.load %q_flat[%qda{c}] : view<[%qtot]xf16> -> {V16H}")
            e(f"    scf.yield {', '.join(f'%qdl{c}' for c in range(16))} : {tys}")
            e("  } else {")
            e(f"    scf.yield {', '.join(['%zq16x'] * 16)} : {tys}")
            e("  }")

            def qdir_after():
                e("  %qw_na, %qe_na = buffer.assume.noalias %qnw, %qeps : buffer, buffer")
                e("  %qw_flat = buffer.view %qw_na[%base] : buffer -> view<[%c256]xf32>")
                e("  %qe_flat = buffer.view %qe_na[%base] : buffer -> view<[%c1]xf32>")
                e("  %qeps_s = view.load %qe_flat[%c0] : view<[%c1]xf32> -> f32")
                e(f"  %qssv = vector.addf %qssl0, %qssl1 : {V4}")
                e(f"  %qssw = vector.addf %qssl2, %qssl3 : {V4}")
                e(f"  %qssa = vector.addf %qssv, %qssw : {V4}")
                for x in range(4):
                    e(f"  %qss{x} = vector.extract %qssa[{x}] : {V4} -> f32")
                e("  %qss01 = scalar.addf %qss0, %qss1 : f32")
                e("  %qss23 = scalar.addf %qss2, %qss3 : f32")
                e("  %qstot = scalar.addf %qss01, %qss23 : f32")
                e("  %qhd256i = scalar.constant 0.00390625 : f32")
                e("  %qmean = scalar.mulf %qstot, %qhd256i : f32")
                e("  %qme = scalar.addf %qmean, %qeps_s : f32")
                e("  %qinv = scalar.rsqrtf %qme : f32")
                e("  %qsl2e = scalar.mulf %qinv, %log2e : f32")
                if R2P_LAZY:
                    e("  %qsrms = scalar.mulf %qme, %qinv : f32")
                    e(f"  %qslzc = scalar.constant {R2P_LAZY} : f32")
                    e("  %qslzth = scalar.mulf %qsrms, %qslzc : f32")
                qlv = "%r_live"
                if "qdirz" in R2P_DIAG:   # diag: the loaded Q dropped for zeros (opaque: queue word 1 is 0), as noq's Q
                    e("  %qzw = vector.load %pqv[%c1] : view<{PQ_WORDS}xi32> -> vector<8xi32>".replace("{PQ_WORDS}", str(PQ_WORDS)))
                    e("  %qzw0 = vector.extract %qzw[0] : vector<8xi32> -> i32")
                    e("  %qzm = vector.splat %qzw0 : vector<8xi32>")
                for c in range(16):
                    e(f"  %qdz{c} = scf.select {qlv}, %qdv{c}, %zq16x : {V16H}")
                    if "qdirz" in R2P_DIAG:
                        e(f"  %qdzb{c} = vector.bitcast %qdz{c} : {V16H} to vector<8xi32>")
                        e(f"  %rinit{c} = vector.andi %qdzb{c}, %qzm : vector<8xi32>")
                        continue
                    e(f"  %rinit{c} = vector.bitcast %qdz{c} : {V16H} to vector<8xi32>")
            qnorm_after = qdir_after
        elif QNORM and R2P:   # the main kernel's per-block prologue on this wave's QNORM block
            assert not K4
            en = lambda ln: e(re.sub(r"%qdb\b", "%qdbn", re.sub(r"%wqb\b", "%wqn", ln)))
            qn_own, qn_sums = qnorm_phase1(en)
            qn_root = []
            qnorm_before = lambda: qn_root.append(qn_sums())
            qnorm_after = lambda: (qnorm_phase2(en, qn_own, qn_root[0]), qfrags())
        elif QNORM:   # sums before the prologue barrier (the pair's LDS slot), the fragments after it
            assert not K4
            qn_own, qn_sums = qnorm_phase1(e)
            qn_root = []
            qnorm_before = lambda: qn_root.append(qn_sums())
            qnorm_after = lambda: (qnorm_phase2(e, qn_own, qn_root[0]), qfrags())
        else:
            qfrags()
    else:
        # ---- Q stage: row r = rb*16 + rr
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
        e("  %qdb0a = index.mul %qpart, %c64 : index")
        rotq = {}
        if K4:   # H256 of the scaled f32 Q (dims qpart*64 + 8c + 0..7) first
            raw = []
            for c in range(8):
                t = f"0_{c}"
                e(f"  %qdc{t} = index.constant {8 * c} : index")
                e(f"  %qd{t} = index.add %qdb0a, %qdc{t} : index")
                e(f"  %qa{t} = index.add %qrow, %qd{t} : index")
                e(f"  %qa{t}b = index.add %qa{t}, %c4 : index")
                e(f"  %qv{t}a = vector.load %q_flat[%qa{t}] : view<[%qtot]xf32> -> {V4}")
                e(f"  %qv{t}b = vector.load %q_flat[%qa{t}b] : view<[%qtot]xf32> -> {V4}")
                e(f"  %qm{t}a0 = vector.mulf %qv{t}a, %qsc_v : {V4}")
                e(f"  %qm{t}b0 = vector.mulf %qv{t}b, %qsc_v : {V4}")
                raw += [f"%qm{t}a0", f"%qm{t}b0"]
            rot = had64(e, "fh", raw, (1, 2), 0.0625)
            for c in range(8):
                rotq[c] = (rot[2 * c], rot[2 * c + 1])
        for c in range(8):
            t = f"0_{c}"
            if K4:
                e(f"  %qm{t}a = vector.addf {rotq[c][0]}, %z4h : {V4}")
                e(f"  %qm{t}b = vector.addf {rotq[c][1]}, %z4h : {V4}")
            else:
                e(f"  %qdc{t} = index.constant {8 * c} : index")
                e(f"  %qd{t} = index.add %qdb0a, %qdc{t} : index")
                e(f"  %qdg{t}c = index.constant 0 : index")
                e(f"  %qdg{t} = index.add %qd{t}, %qdg{t}c : index")
                e(f"  %qa{t} = index.add %qrow, %qdg{t} : index")
                if Q16:
                    e(f"  %qh{t} = vector.load %q_flat[%qa{t}] : view<[%qtot]xf16> -> {V8H}")
                else:
                    e(f"  %qa{t}b = index.add %qa{t}, %c4 : index")
                    e(f"  %qv{t}a = vector.load %q_flat[%qa{t}] : view<[%qtot]xf32> -> {V4}")
                    e(f"  %qv{t}b = vector.load %q_flat[%qa{t}b] : view<[%qtot]xf32> -> {V4}")
                    e(f"  %qm{t}a = vector.mulf %qv{t}a, %qsc_v : {V4}")
                    e(f"  %qm{t}b = vector.mulf %qv{t}b, %qsc_v : {V4}")
            if not (Q16 and not K4):
                e(f"  %qh{t}a = vector.fptrunc %qm{t}a : {V4} to vector<4xf16>")
                e(f"  %qh{t}b = vector.fptrunc %qm{t}b : {V4} to vector<4xf16>")
                e(f"  %qh{t} = vector.concat<0> %qh{t}a, %qh{t}b : vector<4xf16>, vector<4xf16> -> {V8H}")
            e(f"  %qz{t} = scf.select %qlive, %qh{t}, %zh8 : {V8H}")
            e(f"  vector.store %qz{t}, %qs_view[%qr, %qd{t}] : {V8H}, view<{NQB * 16}x{Q_PITCH}xf16>")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        # Q^T fragments (B: [dim, query]) of this wave's 128 dims, in registers
        e("  %wq16 = index.mul %wqb, %c16 : index")
        names = []
        for c in range(8):
            e(f"  %qfd0{c}c = index.constant {16 * c} : index")
            e(f"  %qfd0{c} = index.add %hd128, %qfd0{c}c : index")
            e(f"  %ql0f{c} = vector.fragment.load<rhs> %q_fr[%qfd0{c}, %wq16] shape [%k, %n] : view<256x{NQB * 16}xf16, %q_lay> -> {V16H}")
            names.append(f"%ql0f{c}")
        # Loom does not drain these LDS loads before the barrier below (no lgkmcnt(0) ahead of s_barrier).
        # The prologue then stages K/V over the Q stage, so a fast wave can overwrite Q rows a slow wave still reads.
        # An LDS store of a value built from both halves of every fragment forces the drain first.
        acc = None
        for c in range(8):
            for j in (0, 8):
                e(f"  %qx0{c}_{j} = vector.extract {names[c]}[{j}] : {V16H} -> f16")
                if acc is None:
                    acc = f"%qx0{c}_{j}"
                else:
                    e(f"  %qy0{c}_{j} = scalar.addf {acc}, %qx0{c}_{j} : f16")
                    acc = f"%qy0{c}_{j}"
        e(f"  %qdrain0 = vector.splat {acc} : {V8H}")
        e(f"  %qdr0_o = index.constant {DRAIN_OFF} : offset")
        e(f"  %qdr0_v = buffer.view %pool[%qdr0_o] : buffer -> view<{NT}x8xf16>")
        e(f"  vector.store %qdrain0, %qdr0_v[%tid, %c0] : {V8H}, view<{NT}x8xf16>")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        for c in range(8):
            e(f"  %qf{c} = vector.fragment<rhs> %ql0f{c} shape [%k, %n] : {V16H}")

    staging_maps(e)

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

    def page_of(start, p, ind, tag):
        """Physical page of the (uniform) tile start; start is already clamped."""
        if "page0" in R2P_DIAG:   # diag: every tile on physical page 0 (no page-table load; wrong rows, in bounds)
            return "%c0"
        e(f"{ind}%{p}{tag}lp = index.div {start}, %c256 : index")
        e(f"{ind}%{p}{tag}pg0 = view.load %pt_flat[%{p}{tag}lp] : view<[%npages]xi32> -> i32")
        e(f"{ind}%{p}{tag}pgr = index.cast %{p}{tag}pg0 : i32 to index")
        e(f"{ind}%{p}{tag}pg = index.assume %{p}{tag}pgr [range(%{p}{tag}pgr, 0, 65535), lt(%{p}{tag}pgr, %npages)] : index")
        return f"%{p}{tag}pg"

    def phys_row(kc, ks, p, ind, tag):
        """Row kc (clamped, in the same page as the clamped ks) -> physical row."""
        if not PAGED:
            return kc
        e(f"{ind}%{p}{tag}ksc = index.min {ks}, %cap_1 : index")
        pg = page_of(f"%{p}{tag}ksc", p, ind, tag)
        e(f"{ind}%{p}{tag}pb = index.mul {pg}, %c256 : index")
        e(f"{ind}%{p}{tag}po = index.rem {kc}, %c256 : index")
        e(f"{ind}%{p}{tag}pr = index.add %{p}{tag}pb, %{p}{tag}po : index")
        return f"%{p}{tag}pr"

    def phys_tilestart(vks, p, ind, tag):
        """V^T tile start vks (a multiple of 16) -> physical tile start (in keys)."""
        if not PAGED:
            return vks
        pg = page_of(vks, p, ind, tag)
        # physical tile = page * 16 + (tile % 16): bounded by npages * 16 = vtiles
        e(f"{ind}%{p}{tag}pb = index.mul {pg}, %c16 : index")
        e(f"{ind}%{p}{tag}t16 = index.div {vks}, %c16 : index")
        e(f"{ind}%{p}{tag}po = index.rem %{p}{tag}t16, %c16 : index")
        e(f"{ind}%{p}{tag}pt = index.add %{p}{tag}pb, %{p}{tag}po : index")
        e(f"{ind}%{p}{tag}pr = index.mul %{p}{tag}pt, %c16 : index")
        return f"%{p}{tag}pr"


    def load_k(ks, p, ind):
        if KDEC:
            nw = 2 if K4 else 4
            e(f"{ind}%{p}kp = index.add {ks}, %kdk : index")
            if PAGED:
                e(f"{ind}%{p}kpc0 = index.min %{p}kp, %cap_1 : index")
                kpc = phys_row(f"%{p}kpc0", ks, p, ind, "kq")
                e(f"{ind}%{p}kpc = index.add {kpc}, %c0 : index")
            else:
                e(f"{ind}%{p}kpc = index.min %{p}kp, %cap_1 : index")
            e(f"{ind}%{p}kr = index.mul %{p}kpc, %c{128 if K4 else 256} : index")
            e(f"{ind}%{p}ka = index.add %{p}kr, %kdga : index")
            e(f"{ind}%{p}kv = vector.load %k_flat[%{p}ka] : view<[%kq32tot]xi32> -> vector<{nw}xi32>")
            e(f"{ind}%{p}sr = index.mul %{p}kpc, %c{32 if K4 else 8} : index")
            e(f"{ind}%{p}sa = index.add %{p}sr, %kdso : index")
            e(f"{ind}%{p}sv = view.load %ks_flat[%{p}sa] : view<[%kstot]xi32> -> i32")
            return [f"%{p}kv", f"%{p}sv"]
        names = []
        if KCLAMP_S:   # rows past ctx_end are zeroed at staging from the unclamped key, so any in-range row will do
            e(f"{ind}%{p}ksb = index.min {ks}, %klast : index")
        for nn in range(NKS()):
            e(f"{ind}%{p}kp{nn} = index.add {f'%{p}ksb' if KCLAMP_S else ks}, %kk{nn} : index")
            if PAGED and KCLAMP_S:
                kpc = phys_row(f"%{p}kp{nn}", f"%{p}ksb", p, ind, f"k{nn}")
                e(f"{ind}%{p}kpc{nn} = index.add {kpc}, %c0 : index")
            elif PAGED:
                e(f"{ind}%{p}kpc{nn}0 = index.min %{p}kp{nn}, %cap_1 : index")
                kpc = phys_row(f"%{p}kpc{nn}0", ks, p, ind, f"k{nn}")
                e(f"{ind}%{p}kpc{nn} = index.add {kpc}, %c0 : index")
            else:
                e(f"{ind}%{p}kpc{nn} = index.min %{p}kp{nn}, %cap_1 : index")
            e(f"{ind}%{p}kr{nn} = index.mul %{p}kpc{nn}, %c1024 : index")
            e(f"{ind}%{p}kr{nn}b = index.add %{p}kr{nn}, %kvbase : index")
            e(f"{ind}%{p}ka{nn} = index.add %{p}kr{nn}b, %kd{nn} : index")
            e(f"{ind}%{p}kv{nn} = vector.load %k_flat[%{p}ka{nn}] : view<[%kvtot]xf16> -> {V8H}")
            names.append(f"%{p}kv{nn}")
        return names

    def load_v(ks, p, ind):
        if VQ4 or VQ8:   # [kvh][tile][dim]: data 2 (VQ4) or 4 (VQ8) dwords, stats 1 dword
            nw = 2 if VQ4 else 4
            if PAGED:
                e(f"{ind}%{p}vks0 = index.min {ks}, %vlast : index")
                vks = phys_tilestart(f"%{p}vks0", p, ind, "vq")
                e(f"{ind}%{p}vks = index.add {vks}, %c0 : index")
            else:
                e(f"{ind}%{p}vks = index.min {ks}, %vlast : index")
            e(f"{ind}%{p}vt16 = index.div %{p}vks, %c16 : index")
            e(f"{ind}%{p}vti0 = index.add %vhb4, %{p}vt16 : index")
            e(f"{ind}%{p}vti1 = index.mul %{p}vti0, %c256 : index")
            e(f"{ind}%{p}vti = index.add %{p}vti1, %vdim : index")
            e(f"{ind}%{p}vda = index.mul %{p}vti, %c{nw} : index")
            e(f"{ind}%{p}vq = vector.load %v_flat[%{p}vda] : view<[%vq4tot]xi32> -> vector<{nw}xi32>")
            e(f"{ind}%{p}vst = view.load %vs4_flat[%{p}vti] : view<[%vq4tot0]xi32> -> i32")
            return [f"%{p}vq", f"%{p}vst"]
        if PAGED:
            e(f"{ind}%{p}vks0 = index.min {ks}, %vlast : index")
            vks = phys_tilestart(f"%{p}vks0", p, ind, "v")
            e(f"{ind}%{p}vks = index.add {vks}, %c0 : index")
        else:
            e(f"{ind}%{p}vks = index.min {ks}, %vlast : index")
        e(f"{ind}%{p}vtb = index.mul %{p}vks, %c256 : index")
        w64 = R2P and NQB == 8   # dims + 128 sh (the staging map's half), folded into the lane base
        if VSWZR:
            e(f"{ind}%{p}vln = index.mul %vdim, %c16 : index")
            e(f"{ind}%{p}vhl = index.add %{'vhbs' if w64 else 'vhb'}, %{p}vln : index")
            e(f"{ind}%{p}va0{'s' if w64 else ''} = index.add %{p}vhl, %{p}vtb : index")
        else:
            e(f"{ind}%{p}va0{'s' if w64 else ''} = index.add %{'vhls' if w64 else 'vhl'}, %{p}vtb : index")
        if w64:
            e(f"{ind}%{p}va1s = index.add %{p}va0s, %c8 : index")
            names = []
            for nn in range(2):
                e(f"{ind}%{p}vv{nn} = vector.load %v_flat[%{p}va{nn}s] : view<[%vtot]xf16> -> {V8H}")
                names.append(f"%{p}vv{nn}")
            return names
        e(f"{ind}%{p}va1 = index.add %{p}va0, %c8 : index")
        if R2P:   # dims + 128
            e(f"{ind}%{p}vd128 = index.constant 2048 : index")
            e(f"{ind}%{p}va2 = index.add %{p}va0, %{p}vd128 : index")
            e(f"{ind}%{p}va3 = index.add %{p}va2, %c8 : index")
        names = []
        for nn in range(4 if R2P else 2):
            e(f"{ind}%{p}vv{nn} = vector.load %v_flat[%{p}va{nn}] : view<[%vtot]xf16> -> {V8H}")
            names.append(f"%{p}vv{nn}")
        return names

    if NT > 256 and not R2P:
        # wave-uniform role guards from the subgroup id; loads stay unconditional (out-of-role threads load clamped,
        # in-range rows), only the LDS stores are guarded: an scf.if around loads drains vmcnt(0) at its exit
        e("  %sgid = kernel.subgroup.id : index")
        e(f"  %vsg0 = index.constant {(NT - 256) // 32} : index")
        e("  %kstg = index.cmp ult, %sgid, %c8 : index")
        e("  %vstg = index.cmp uge, %sgid, %vsg0 : index")

    def stage_ks(ks, slot32, cond, p, ind):
        """KRAW: lanes with cond stage tile ks's K scales (item %kstj) into LDS slot slot32 / 32 (cond None: every lane,
        the same values to the same addresses: no divergent region, which the 2x32 config's lowering rejects here)."""
        e(f"{ind}scf.if {cond} {{" if cond else f"{ind}scf.schedule.fence")
        e(f"{ind}  %{p}kk = index.min {ks}, %cap_16 : index")   # (a 16-key run past cap - 16 would leave the buffer)
        pg = page_of(f"%{p}kk", p, ind + "  ", "ks")
        e(f"{ind}  %{p}pb = index.mul {pg}, %c256 : index")
        e(f"{ind}  %{p}po = index.rem %{p}kk, %c256 : index")
        e(f"{ind}  %{p}pr = index.add %{p}pb, %{p}po : index")
        e(f"{ind}  %{p}gi0 = index.add %kstg, %{p}pr : index")
        e(f"{ind}  %{p}gi = index.assume %{p}gi0 [range(%{p}gi0, 0, 1073741824), lt(%{p}gi0, %kscn)] : index")
        e(f"{ind}  %{p}sv = vector.load %ksc_view[%{p}gi] : view<[%kscn]xf16> -> {V8H}")
        e(f"{ind}  %{p}sf = vector.extf %{p}sv : {V8H} to {V8}")
        e(f"{ind}  %{p}li = index.add {slot32}, %kstl : index")
        e(f"{ind}  vector.store %{p}sf, %ksl_view[%{p}li] : {V8}, view<64xf32>")
        if KH:
            e(f"{ind}  vector.store %{p}sf, %ksl_view2[%{p}li] : {V8}, view<32xf32>")
        if cond:
            e(f"{ind}}}")

    def stage_k(ks, cur, p, ind, kv="%k_view", roff=None):
        if NT == 256 or R2P:
            return stage_k_(ks, cur, p, ind, kv, roff)
        e(f"{ind}scf.if %kstg {{")
        stage_k_(ks, cur, p, ind + "  ", kv, roff)
        e(f"{ind}}}")

    def stage_k_(ks, cur, p, ind, kv="%k_view", roff=None):
        """K tile at key ks (rows permuted, zero past ctx_end) to LDS (rows + roff: R2P_K2's slot)."""
        kvt = f"view<{32 if roff else KT}x{KT_PITCH}xf16>"
        kr = {}
        for nn in range(NKS()):
            kr[nn] = f"%kr{nn}"
            if roff:
                e(f"{ind}%{p}kro{nn} = index.add %kr{nn}, {roff} : index")
                kr[nn] = f"%{p}kro{nn}"
        assert not (roff and KDEC)
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
                    if DEC_LSHADD:
                        dec_pair(e, ind, f"%{p}g{d}{k}", f"%{p}w{d}", sh, 0x03c003c0 if K4 else 0x03fc03fc, "%kdgm")
                        ws.append(f"%{p}g{d}{k}")
                        continue
                    op = "shli" if sh >= 0 else "shrui"
                    e(f"{ind}%{p}t{d}{k}c = scalar.constant {abs(sh)} : i32")
                    e(f"{ind}%{p}t{d}{k} = scalar.{op} %{p}w{d}, %{p}t{d}{k}c : i32")
                    e(f"{ind}%{p}m{d}{k} = scalar.andi %{p}t{d}{k}, %kdm : i32")
                    e(f"{ind}%{p}g{d}{k} = scalar.ori %{p}m{d}{k}, %kdgm : i32")
                    ws.append(f"%{p}g{d}{k}")
            # ONE 16-wide fma: two 8-wide ones miscompile (see vq4_unpack)
            e(f"{ind}%{p}pk = vector.from_elements {', '.join(ws)} : vector<8xi32>")
            e(f"{ind}%{p}pf = vector.bitcast %{p}pk : vector<8xi32> to {V16H}")
            e(f"{ind}%{p}S16 = vector.splat %{p}sS : {V16H}")
            e(f"{ind}%{p}C16 = vector.splat %{p}sC : {V16H}")
            e(f"{ind}%{p}dk = vector.fmaf %{p}pf, %{p}S16, %{p}C16 : {V16H}")
            e(f"{ind}%{p}dz = scf.select %{p}ql, %{p}dk, %zq16 : {V16H}")
            e(f"{ind}%{p}d0 = vector.slice %{p}dz[0] : {V16H} -> {V8H}")
            e(f"{ind}%{p}d1 = vector.slice %{p}dz[8] : {V16H} -> {V8H}")
            e(f"{ind}vector.store %{p}d0, {kv}[%kdr, %kdc16] : {V8H}, view<{KT}x{KT_PITCH}xf16>")
            e(f"{ind}vector.store %{p}d1, {kv}[%kdr, %kdc16b] : {V8H}, view<{KT}x{KT_PITCH}xf16>")
            return
        for nn in range(NKS()):
            if CTX16:
                e(f"{ind}vector.store {cur[nn]}, {kv}[{kr[nn]}, %kd{nn}] : {V8H}, {kvt}")
                continue
            e(f"{ind}%{p}sp{nn} = index.add {ks}, %kk{nn} : index")
            e(f"{ind}%{p}sl{nn} = index.cmp ult, %{p}sp{nn}, %ctx_end : index")
            e(f"{ind}%{p}sv{nn} = scf.select %{p}sl{nn}, {cur[nn]}, %zh8 : {V8H}")
            e(f"{ind}vector.store %{p}sv{nn}, {kv}[{kr[nn]}, %kd{nn}] : {V8H}, {kvt}")

    def vq8_unpack(cur, p, ind):
        # bytes (k0, k2, k1, k3) per dword -> f16 1 + u/256 pairs (keys 2k, 2k+1) via (w << 2, w >> 6) & 0x03fc03fc | 0x3c003c00
        # then ONE 16-wide fma f * S + C' (see vq4_unpack)
        e(f"{ind}%{p}vs1 = vector.from_elements {cur[1]} : vector<1xi32>")
        e(f"{ind}%{p}vsv = vector.bitcast %{p}vs1 : vector<1xi32> to vector<2xf16>")
        e(f"{ind}%{p}vss = vector.extract %{p}vsv[0] : vector<2xf16> -> f16")
        e(f"{ind}%{p}vsc = vector.extract %{p}vsv[1] : vector<2xf16> -> f16")
        ws = []
        for d in range(4):
            e(f"{ind}%{p}vw{d}x = vector.extract {cur[0]}[{d}] : vector<4xi32> -> i32")
            for k, sh in enumerate((2, -6)):
                op = "shli" if sh >= 0 else "shrui"
                e(f"{ind}%{p}vsh{d}{k}c = scalar.constant {abs(sh)} : i32")
                e(f"{ind}%{p}vsh{d}{k} = scalar.{op} %{p}vw{d}x, %{p}vsh{d}{k}c : i32")
                e(f"{ind}%{p}vmk{d}{k} = scalar.andi %{p}vsh{d}{k}, %v8m : i32")
                e(f"{ind}%{p}vmg{d}{k} = scalar.ori %{p}vmk{d}{k}, %v4g : i32")
                ws.append(f"%{p}vmg{d}{k}")
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
                if DEC_LSHADD:
                    dec_pair(e, ind, f"%{p}vmg{d}{k}", f"%{p}vw{d}x", sh, 0x03c003c0, "%v4g")
                    ws.append(f"%{p}vmg{d}{k}")
                    continue
                op = "shli" if sh >= 0 else "shrui"
                e(f"{ind}%{p}vsh{d}{k}c = scalar.constant {abs(sh)} : i32")
                e(f"{ind}%{p}vsh{d}{k} = scalar.{op} %{p}vw{d}x, %{p}vsh{d}{k}c : i32")
                e(f"{ind}%{p}vmk{d}{k} = scalar.andi %{p}vsh{d}{k}, %v4m : i32")
                e(f"{ind}%{p}vmg{d}{k} = scalar.ori %{p}vmk{d}{k}, %v4g : i32")
                ws.append(f"%{p}vmg{d}{k}")
        # ONE 16-wide fma: with two 8-wide fmas Loom CSEs their identical addend splats
        # and ties both in-place v_pk_fmac to one register, so keys 8..15 get f * S + (the result of keys 0..7).
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

    def stage_v(cur, vb, p, ind, col0=0):
        if VQ4:   # unpack outside the guard (an scf.if reading loaded registers drains vmcnt(0) at entry)
            cur = vq4_unpack(cur, p, ind)
        if VQ8:
            cur = vq8_unpack(cur, p, ind)
        if NT == 256 or R2P:
            return stage_v_(cur, vb, p, ind, col0)
        e(f"{ind}scf.if %vstg {{")
        stage_v_(cur, vb, p, ind + "  ", col0)
        e(f"{ind}}}")

    def stage_v_(cur, vb, p, ind, col0=0):
        e(f"{ind}%{p}vr = index.add {vb}, %vrow : index")
        vp = VT_PITCH
        if R2P:   # dims + 128: LDS rows + 128
            e(f"{ind}%{p}vr2 = index.add %{p}vr, %c128 : index")
            if R2P_K2:
                assert vb == "%c0"
                assert len(cur) == (4 if NQB == 4 else 2)
                for j, v in enumerate(cur):   # row lo, row hi, row + 128 lo, hi (NQB 8: lo, hi of dims + 128 sh)
                    off = VSPL_ST[j]
                    e(f"{ind}%{p}vsx{j} = vector.bitcast {v} : {V8H} to vector<16xi8>")
                    e(f"{ind}vector.store %{p}vsx{j}, %vgvs{j}[{'%vgs' if NQB == 4 else '%vgs8'}] : vector<16xi8>, view<{VT_BYTES - off}xi8>")
                return
            for j, v in enumerate(cur):
                e(f"{ind}vector.store {v}, %v_view[{'%' + p + 'vr2' if j >= 2 else '%' + p + 'vr'}, %c{8 * (j % 2)}] : {V8H}, view<256x{vp}xf16>")
            return
        for j, v in enumerate(cur):
            cc = col0 + 8 * j
            if cc in (0, 8, 16):
                col = f"%c{cc}"
            else:
                e(f"{ind}%{p}vc{j} = index.constant {cc} : index")
                col = f"%{p}vc{j}"
            e(f"{ind}vector.store {v}, %v_view[%{p}vr, {col}] : {V8H}, view<256x{vp}xf16>")

    # prologue: K(0) staged
    ks0, ks0b = ("%kbeg", "%kbeg16") if KS else ("%c0", "%c16")
    k0 = load_k(ks0, "p0", "  ")
    if PIPE2:
        k1 = load_k("%c16", "p1", "  ")
        v0 = load_v("%c0", "pv0", "  ")
    stage_k(ks0, k0, "p0s", "  ")
    if KRAW:
        stage_ks(ks0, "%c0", None, "kp0", "  ")
    if SOFT1 and (SOFT2 or SOFT4):
        stage_v_(["%zh8", "%zh8"], "%c0", "pz", "  ")
        e("  %pzlo = index.cmp ult, %tid, %c128 : index")
        e("  scf.if %pzlo {")
        e(f"    %pzrow = index.add %tid, %c{128 if SOFT3 or SOFT4 else 0} : index")
        e(f"    vector.store %zh8, %pp_view[%pzrow, %c0] : vector<8xf16>, view<{PRS}x8xf16>")
        e("  }")
        e("  %pza = index.cmp ult, %tid, %c64 : index")
        e("  scf.if %pza {")
        e(f"    %pzai = index.add %tid, %c{64 if SOFT3 or SOFT4 else 0} : index")
        e(f"    view.store %one, %al_view[%pzai] : f32, view<{PRS2}xf32>")
        e("  }")
    def r2p_init():   # P^T slot 1 = tile -1: 0, alpha 1 (V: the first iteration stages V(0) for P.V(-1))
        e(f"  %pzlo = index.cmp ult, %tid, %c{32 * NQB} : index")
        e("  scf.if %pzlo {")
        e(f"    %pzr = index.add %tid, %c{32 * NQB} : index")
        e(f"    vector.store %zh8, %pp_view[%pzr, %c0] : vector<8xf16>, view<{64 * NQB}x8xf16>")
        e("  }")
        e(f"  %pza = index.cmp ult, %tid, %c{16 * NQB} : index")
        e("  scf.if %pza {")
        e(f"    %pzai = index.add %tid, %c{16 * NQB} : index")
        e(f"    view.store %one, %al_view[%pzai] : f32, view<{32 * NQB}xf32>")
        e("  }")
    if R2P and NQB == 4:
        r2p_init()
    if qnorm_before:
        qnorm_before()
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    if qnorm_after:
        qnorm_after()
    if R2P and NQB == 8:   # read only after later barriers
        r2p_init()
    if PIPE2:
        e(f"  %k2_o = index.constant {K2_OFF} : offset")
        e(f"  %k_view2 = buffer.view %pool[%k2_o] : buffer -> view<{KT}x{KT_PITCH}xf16>")

    onames = [f"%o{f}" for f in range(8)]
    types = ", ".join([V8] * 8 + ["f32", "f32"])

    def emit_qk(kv, pp, ind):
        """S^T partial over this wave's 128 dims: one 8-MMA chain from zero (the accumulation order is fixed)."""
        e(f"{ind}%{pp}zeros8s = vector.fragment<init> %zeros8 shape [%m, %n] : {V8}")
        acc = f"%{pp}zeros8s"
        for c in range(8):
            e(f"{ind}%{pp}kfc{c}c = index.constant {16 * c} : index")
            e(f"{ind}%{pp}kfc{c} = index.add %hd128, %{pp}kfc{c}c : index")
            e(f"{ind}%{pp}kf{c} = vector.fragment.load<lhs> {kv}[%c0, %{pp}kfc{c}] shape [%m, %k] : view<{KT}x{KT_PITCH}xf16> -> {V16H}")
            e(f"{ind}%{pp}sa{c} = vector.mma %{pp}kf{c}, %qf{c}, {acc} : {V16H}, {V16H}, {V8}")
            acc = f"%{pp}sa{c}"
            if c + 1 < 8 and (c + 1) % QKF == 0:
                e(f"{ind}scf.schedule.fence")
        return acc

    def emit_softmax_pv(acc, tail, mid=None, early=None, pair_slot=None):
        """Softmax of this tile (own partial acc + the partner's from LDS) and P.V; returns the O accumulators.
        mid(): emitted right after the rescale branch (the pipelined QK of the next tile and its loads).
        pair_slot (the QK role): %s0 is the full S; P^T and alpha go to that LDS slot (rows, alpha base); no P.V."""
        L2E = "%r2l2e" if R2P and R2P_QSPLIT else "%log2e"   # (QSPLIT: S is unscaled; r log2e scales it)
        fold = R2P and R2P_QSPLIT and R2P_QFOLD   # S already in log2 units: no scale op
        def l2mul(d, x, ind):
            if fold:
                return x
            e(f"{ind}{d} = scalar.mulf {x}, {L2E} : f32")
            return d
        if pair_slot is None:
            e(f"    %spa = vector.load %s_view[%ptid, %c0] : view<{2 * NT}x4xf32> -> {V4}")
            e(f"    %spb = vector.load %s_view[%ptid2, %c0] : view<{2 * NT}x4xf32> -> {V4}")
            e(f"    %spart = vector.concat<0> %spa, %spb : {V4}, {V4} -> {V8}")
            e(f"    %s0 = vector.addf {acc}, %spart : {V8}")
        s = "%s0"
        if tail:
            def limit(ind):   # key 8*half + i visible iff < lim = min(r_abs + 1, ctx_end) - ks - 8*half
                e(f"{ind}%lim0 = index.add %r_abs, %c1 : index")
                e(f"{ind}%lim1 = index.min %lim0, %ctx_end : index")
                e(f"{ind}%h8 = index.mul %half, %c8 : index")
                e(f"{ind}%kh = index.add %ks_, %h8 : index")
                e(f"{ind}%limi = index.cast %lim1 : index to i32")
                e(f"{ind}%khi = index.cast %kh : index to i32")
                e(f"{ind}%lim = scalar.subi %limi, %khi : i32")
            r2m = R2P and DIAG_MASK and DIAG_SWAP   # R2P: the pass test on the wave's uniform bound (%r2mlb, pre-loop)
            if not r2m:
                limit("    ")
            ind = "    "
            if DIAG_MASK:
                # some lane has a key at or past its limit (lim < 8) or a dead row
                e("    %mc8 = scalar.constant 8 : i32")
                if r2m:   # min over live lanes of (lim + ks) - ks >= 8, no dead row: exactly the vote below
                    e("    %mksi = index.cast %ks_ : index to i32")
                    e("    %mrem = scalar.subi %r2mlb, %mksi : i32")
                    if "nomask" in R2P_DIAG:   # diag: never mask (diagonal / context-end scores pass through)
                        e("    %mallok = index.cmp eq, %c0, %c0 : index")
                    else:
                        e("    %mallok = scalar.cmpi sge, %mrem, %mc8 : i32")
                    e(f"    %smask = scf.if %mallok -> ({V8}) {{")
                    e(f"      scf.yield %s0 : {V8}")
                    e("    } else {")
                    limit("      ")
                elif DIAG_SWAP:   # no lane has a key at or past its limit and no dead row: the scores pass through
                    e("    %mok8 = scalar.cmpi sge, %lim, %mc8 : i32")
                    e("    %mlive = index.cmp ult, %r_lq, %B : index")
                    e("    %mok = scalar.andi %mok8, %mlive : i1")
                    e("    %mallok = kernel.subgroup.vote.all %mok : i1")
                    e(f"    %smask = scf.if %mallok -> ({V8}) {{")
                    e(f"      scf.yield %s0 : {V8}")
                    e("    } else {")
                else:
                    e("    %mpart = scalar.cmpi slt, %lim, %mc8 : i32")
                    e("    %mdead = index.cmp uge, %r_lq, %B : index")
                    e("    %mneed = scalar.ori %mpart, %mdead : i1")
                    e("    %manyneed = kernel.subgroup.vote.any %mneed : i1")
                    e(f"    %smask = scf.if %manyneed -> ({V8}) {{")
                ind = "      "
            els = []
            for i in range(8):
                e(f"{ind}%mi{i} = scalar.constant {i} : i32")
                e(f"{ind}%mv{i}a = scalar.cmpi slt, %mi{i}, %lim : i32")
                e(f"{ind}%mv{i} = scalar.andi %mv{i}a, %r_live_i1 : i1")
                e(f"{ind}%se{i} = vector.extract %s0[{i}] : {V8} -> f32")
                e(f"{ind}%sm{i} = scf.select %mv{i}, %se{i}, %ninf : f32")
                els.append(f"%sm{i}")
            if DIAG_MASK:
                e(f"      %smk = vector.from_elements {', '.join(els)} : {V8}")
                e(f"      scf.yield %smk : {V8}")
                if not DIAG_SWAP:
                    e("    } else {")
                    e(f"      scf.yield %s0 : {V8}")
                e("    }")
            else:
                e(f"    %smask = vector.from_elements {', '.join(els)} : {V8}")
            s = "%smask"
        if MAXNN:
            for i in range(8):
                e(f"    %tme{i} = vector.extract {s}[{i}] : {V8} -> f32")
            if R2P:   # a tree (max is exact in any order): pairs of independent maxes can dual-issue
                lv = [f"%tme{i}" for i in range(8)] if "nomax" not in R2P_DIAG else ["%tme0"]   # (diag nomax: element 0)
                while len(lv) > 1:
                    nx = []
                    for a, b in zip(lv[::2], lv[1::2]):
                        e(f"    %tmt{len(lv)}_{len(nx)} = scalar.maxnumf<nnan> {a}, {b} : f32")
                        nx.append(f"%tmt{len(lv)}_{len(nx)}")
                    lv = nx
                cur = lv[0]
            else:
                cur = "%tme0"
                for i in range(1, 8):
                    e(f"    %tmc{i} = scalar.maxnumf<nnan> {cur}, %tme{i} : f32")
                    cur = f"%tmc{i}"
            e(f"    %tmi = scalar.bitcast {cur} : f32 to i32")
            e("    %tmx, %tmv = kernel.subgroup.shuffle<xor> %tmi, %x16, %x32 : i32, i32, i32")
            e("    %tmf = scalar.bitcast %tmx : i32 to f32")
            e(f"    %tmax = scalar.maxnumf<nnan> {cur}, %tmf : f32")
        else:
            if "nomax" in R2P_DIAG:   # diag: the row max of element 0 only (no reduce; wrong values)
                e(f"    %tmax0 = vector.extract {s}[0] : {V8} -> f32")
            else:
                e(f"    %tmax0 = vector.reduce<maxnumf> {s}, %ninf : {V8}, f32")
            e("    %tmi = scalar.bitcast %tmax0 : f32 to i32")
            e("    %tmx, %tmv = kernel.subgroup.shuffle<xor> %tmi, %x16, %x32 : i32, i32, i32")
            e("    %tmf = scalar.bitcast %tmx : i32 to f32")
            e("    %tmax = scalar.maxnumf %tmax0, %tmf : f32")
        if early:
            early()
        # one wave-uniform branch: raise the max and rescale O / l only when some row needs it (FA4 conditional rescale)
        if R2P and R2P_LAZY:   # the max is raised only past rmax + R2P_LAZY: p up to e^R2P_LAZY (f16 has room), O
            # and l carry the same stale scale, which the epilogue's O / l cancels (rounding points move: gate N)
            if R2P_QSPLIT:
                e("    %tmlz = scalar.subf %tmax, %qslzth : f32")
            else:
                e(f"    %lzth = scalar.constant {R2P_LAZY} : f32")
                e("    %tmlz = scalar.subf %tmax, %lzth : f32")
            if "nogrow" in R2P_DIAG:   # diag: the max never raised past the first (opaque: rmax + 1e30; wrong values)
                e("    %grbig = scalar.constant 1.0e30 : f32")
                e("    %grth = scalar.addf %rmax, %grbig : f32")
                e("    %grow = scalar.cmpf ogt, %tmlz, %grth : f32")
            else:
                e("    %grow = scalar.cmpf ogt, %tmlz, %rmax : f32")
        else:
            e("    %grow = scalar.cmpf ogt, %tmax, %rmax : f32")
        e("    %anygrow = kernel.subgroup.vote.any %grow : i1")
        if pair_slot is not None:   # O lives in wave B: only the max and alpha here
            e("    %rsm, %rss = scf.if %anygrow -> (f32, f32) {")
            e("      %gmx = scalar.maxnumf %rmax, %tmax : f32")
            e("      %gpd = scalar.subf %rmax, %gmx : f32")
            e(f"      %galpha = scalar.exp2f<afn> {l2mul('%gpdl', '%gpd', '      ')} : f32")
            e("      scf.yield %gmx, %galpha : f32, f32")
            e("    } else {")
            e("      scf.yield %rmax, %one : f32, f32")
            e("    }")
        else:
            otypes = ", ".join([V8] * 8 + ["f32", "f32"])
            e(f"    %rso0, %rso1, %rso2, %rso3, %rso4, %rso5, %rso6, %rso7, %rsm, %rss = scf.if %anygrow -> ({otypes}) {{")
            e(f"      %gmx = scalar.maxnumf{'<nnan>' if MAXNN else ''} %rmax, %tmax : f32")
            e("      %gpd = scalar.subf %rmax, %gmx : f32")
            e(f"      %galpha = scalar.exp2f<afn> {l2mul('%gpdl', '%gpd', '      ')} : f32")
            e(f"      %galpha8 = vector.splat %galpha : {V8}")
            for f in range(8):
                e(f"      %gos{f} = vector.mulf %o{f}, %galpha8 : {V8}")
            # yields alpha; 1.0 when skipped: O * 1.0 and fma(sum, 1.0, part) are exact, so the skip is bit-identical
            e(f"      scf.yield {', '.join(f'%gos{f}' for f in range(8))}, %gmx, %galpha : {otypes}")
            e("    } else {")
            e(f"      scf.yield {', '.join(f'%o{f}' for f in range(8))}, %rmax, %one : {otypes}")
            e("    }")
        if mid:
            mid()
        # MAXNN: both arms yield max(rmax, ..) >= -FLT_MAX, so with ninf LLVM folds the seed max away
        e(f"    %nmax = scalar.maxnumf{'<nnan|ninf>' if MAXNN else ''} %rsm, %ninf : f32")
        e(f"    %nnml = scalar.negf {l2mul('%nml', '%nmax', '    ')} : f32")
        e("    %pd = scalar.subf %rmax, %nmax : f32")
        e(f"    %alpha = scalar.exp2f<afn> {l2mul('%pdl', '%pd', '    ')} : f32")
        e(f"    %nnml8 = vector.splat %nnml : {V8}")
        e(f"    %nmax8 = vector.splat %nmax : {V8}")
        pad = {k[:4]: int(k[4:]) for k in R2P_DIAG if k[:4] in ("vpad", "xpad")}
        if pad:   # diag: vpadN = N extra V8 mul ops (4 chains), xpadN = N extra V8 exp2 ops (2 chains), fed into s
            ends = []
            nv, nx = pad.get("vpad", 0), pad.get("xpad", 0)
            for k in range(4 if nv else 0):
                e(f"    %vpk{k} = scalar.constant {k + 1}.5 : f32")
                e(f"    %vpc{k} = vector.splat %vpk{k} : {V8}")
                e(f"    %vp{k}_0 = vector.addf {s}, %vpc{k} : {V8}")
                for j in range(1, nv // 4):
                    e(f"    %vp{k}_{j} = vector.mulf %vp{k}_{j - 1}, {s} : {V8}")
                ends.append(f"%vp{k}_{nv // 4 - 1}")
            for k in range(2 if nx else 0):
                cur = s
                for j in range(nx // 2):
                    e(f"    %xp{k}_{j} = vector.exp2f<afn> {cur} : {V8}")
                    cur = f"%xp{k}_{j}"
                    if j == 0 and k == 1:   # the second chain starts from a different value
                        e(f"    %xpo = vector.addf %xp1_0, {s} : {V8}")
                        cur = "%xpo"
                ends.append(cur)
            for i, x in enumerate(ends):
                e(f"    %spad{i} = vector.addf {s if i == 0 else f'%spad{i - 1}'}, {x} : {V8}")
            s = f"%spad{len(ends) - 1}"
        xq = [int(k[2:]) for k in R2P_DIAG if k.startswith("xq")]
        if xq:   # diag: xqN = N extra V8 exp2 ops in 4 chains (shallow: more independent exps than xpad)
            ends = []
            for k in range(4):
                cur = s
                if k:
                    e(f"    %xqk{k} = scalar.constant {k}.25 : f32")
                    e(f"    %xqc{k} = vector.splat %xqk{k} : {V8}")
                    e(f"    %xqs{k} = vector.addf {s}, %xqc{k} : {V8}")
                    cur = f"%xqs{k}"
                for j in range(xq[0] // 4):
                    e(f"    %xq{k}_{j} = vector.exp2f<afn> {cur} : {V8}")
                    cur = f"%xq{k}_{j}"
                ends.append(cur)
            for i, x in enumerate(ends):
                e(f"    %sxq{i} = vector.addf {s if i == 0 else f'%sxq{i - 1}'}, {x} : {V8}")
            s = f"%sxq{len(ends) - 1}"
        sp = [int(k[4:]) for k in R2P_DIAG if k.startswith("spad")]
        if sp:   # diag: spadN = N extra scalar index ops (4 chains of mul / add on runtime values), one select of s
            ends = []
            for k in range(4):
                e(f"    %spk{k} = index.constant {k + 3} : index")
                e(f"    %sp{k}_0 = index.add %ks_, %spk{k} : index")
                for j in range(1, sp[0] // 4):
                    op, b = ("index.max", "%start_pos") if j % 2 else ("index.add", "%B")
                    e(f"    %sp{k}_{j} = {op} %sp{k}_{j - 1}, {b} : index")
                ends.append(f"%sp{k}_{sp[0] // 4 - 1}")
            cur = ends[0]
            for i, x in enumerate(ends[1:]):
                e(f"    %spx{i} = index.add {cur}, {x} : index")
                cur = f"%spx{i}"
            e(f"    %spm0 = index.constant 1024 : index")
            e(f"    %spr = index.rem {cur}, %spm0 : index")
            e(f"    %spm1 = index.constant 5 : index")
            e(f"    %spcnd = index.cmp eq, %spr, %spm1 : index")
            e(f"    %ssp = scf.select %spcnd, %zeros8, {s} : {V8}")
            s = "%ssp"
        if "nosoft" in R2P_DIAG:   # p = s: no subtract / scale / exp (two free bitcasts)
            e(f"    %pdi = vector.bitcast {s} : {V8} to vector<8xi32>")
            e(f"    %p = vector.bitcast %pdi : vector<8xi32> to {V8}")
        elif "nosub" in R2P_DIAG:   # diag: p = exp2(s) (no subtract / scale: the bound of log2e-in-Q + C = -m; wrong values)
            e(f"    %p = vector.exp2f<afn> {s} : {V8}")
        else:
            e(f"    %pd8 = vector.subf {s}, %nmax8 : {V8}")
            if "nol2e" in R2P_DIAG:   # diag: no log2e multiply (p = exp2(s - m): wrong values)
                e(f"    %pl = vector.addf %pd8, %zeros8 : {V8}")
            elif "nomul" in R2P_DIAG:   # diag: p = exp2(s - m), no scale op at all (the schedule of r log2e folded into Q)
                e(f"    %pli8 = vector.bitcast %pd8 : {V8} to vector<8xi32>")
                e(f"    %pl = vector.bitcast %pli8 : vector<8xi32> to {V8}")
            elif KH and R2P and "khfma" in R2P_DIAG:   # (diag khfma: one fma for sub + mul: v_fma_f32 has no VOPD form,
                # so 8 single issues for 4 dual subs, and it breaks the adds' / maxes' pairing: +5% cycles)
                e(f"    %pl = vector.fmaf {s}, %r2l2e8, %nnml8 : {V8}")
            elif fold:
                e(f"    %pli8 = vector.bitcast %pd8 : {V8} to vector<8xi32>")
                e(f"    %pl = vector.bitcast %pli8 : vector<8xi32> to {V8}")
            else:
                e(f"    %pl = vector.mulf %pd8, {'%r2l2e8' if R2P else '%log2e8'} : {V8}")
            if "noexp" in R2P_DIAG:   # diag: p = the exponent itself (no exp2; wrong values, same max / rescale path)
                e(f"    %p = vector.addf %pl, %zeros8 : {V8}")
            else:
                e(f"    %p = vector.exp2f<afn> %pl : {V8}")
        def emit_sum():
            # keys 8h..8h+3 and 8h+4..8h+7: HIP's 4-lane segments
            for g in range(2 if "nosum" not in R2P_DIAG else 0):
                cur = "%zero"
                for i in range(4):
                    e(f"    %pg{g}_{i} = vector.extract %p[{4 * g + i}] : {V8} -> f32")
                    if SUM0 and i == 0:
                        cur = f"%pg{g}_0"
                        continue
                    e(f"    %pa{g}_{i} = scalar.addf {cur}, %pg{g}_{i} : f32")
                    cur = f"%pa{g}_{i}"
            if "nosum" in R2P_DIAG:   # no row sum this tile (the carried sum only rescaled)
                e("    %nsum = scalar.fmaf %rsum, %rss, %rsum : f32")
            else:
                e("    %pab = scalar.addf %pa0_3, %pa1_3 : f32")
                e("    %pabi = scalar.bitcast %pab : f32 to i32")
                e("    %pcdi, %pcdv = kernel.subgroup.shuffle<xor> %pabi, %x16, %x32 : i32, i32, i32")
                e("    %pcd = scalar.bitcast %pcdi : i32 to f32")
                e("    %psum = scalar.addf %pab, %pcd : f32")
                e("    %nsum = scalar.fmaf %rsum, %rss, %psum : f32")
        sum_late = R2P_SUMLATE and pair_slot is not None and R2P and R2P_PSB
        if not sum_late:
            emit_sum()
        # P^T as the B operand: keys 0..7 from the h=0 lane, 8..15 from h=1.
        # f16(p) as fptrunc(fma(p, 1, 0)) with an opaque 1 and 0, which selects v_fma_mix (as Q4FMIX in the GEMMs).
        # v_cvt_f16_f32 results must sit in v0..v127, where Q^T and O live: each conversion evicts a Q^T fragment to scratch.
        if "nocvt" in R2P_DIAG:   # diag: P^T bits = 4 of the f32 p, no conversion (wrong values)
            for i in range(4):
                e(f"    %pe{i} = vector.extract %p[{i}] : {V8} -> f32")
            e("    %pv4 = vector.from_elements %pe0, %pe1, %pe2, %pe3 : vector<4xf32>")
            e(f"    %ph = vector.bitcast %pv4 : vector<4xf32> to {V8H}")
        else:
            for i in range(8):
                e(f"    %pe{i} = vector.extract %p[{i}] : {V8} -> f32")
                e(f"    %pm{i} = scalar.fmaf %pe{i}, %one_o, %zero_o : f32")
                e(f"    %pt{i} = scalar.fptrunc %pm{i} : f32 to f16")
            e(f"    %ph = vector.from_elements {', '.join(f'%pt{i}' for i in range(8))} : {V8H}")
            for k in range(4 if R2P_PST4 else 0):
                e(f"    %phw{k} = vector.from_elements %pt{2 * k}, %pt{2 * k + 1} : vector<2xf16>")
        pst = lambda r: ([f"      vector.store %phw{k}, %pp_view[{r}, %c{2 * k}] : vector<2xf16>, view<{64 * NQB}x8xf16>"
                          for k in range(4)] if R2P_PST4 and "nocvt" not in R2P_DIAG else
                         [f"      vector.store %ph, %pp_view[{r}, %c0] : {V8H}, view<{64 * NQB}x8xf16>"])
        if pair_slot is not None and R2P and R2P_PSB:
            # the slot by a uniform branch on the tile parity: each arm's addresses are loop-invariant + an immediate
            # (as a runtime row index Loom rebuilds them every tile: v_mov of the parity, 2 lshl_add, 2 lshl)
            e("    %psl0 = index.cmp eq, %kpar, %c0 : index")
            e("    scf.if %psl0 {")
            for ln in pst("%prow_own"):
                e(ln)
            e(f"      view.store %rss, %al_view[%pa_own] : f32, view<{32 * NQB}xf32>")
            e("    } else {")
            e("      %pwr1 = index.add %prow_own, %c128p : index")
            e(f"      %pwa1 = index.add %pa_own, %c{16 * NQB} : index")
            for ln in pst("%pwr1"):
                e(ln)
            e(f"      view.store %rss, %al_view[%pwa1] : f32, view<{32 * NQB}xf32>")
            e("    }")
            if sum_late:
                emit_sum()
            return None
        if pair_slot is not None:
            rows, alb = pair_slot
            e(f"    %pwr = index.add {rows}, %prow_own : index")
            prows = PRS if SOFT1 else 256
            e(f"    vector.store %ph, %pp_view[%pwr, %c0] : {V8H}, view<{prows}x8xf16>")
            e(f"    %pwa = index.add {alb}, %pa_own : index")
            e(f"    view.store %rss, %al_view[%pwa] : f32, view<{PRS2 if SOFT1 else 128}xf32>")
            return None
        if PBP:   # lane l: keys 0..7 from lane l & 15 (h = 0), keys 8..15 from lane (l & 15) + 16 (h = 1)
            e(f"    %phi = vector.bitcast %ph : {V8H} to {V4I}")
            e(f"    %pli, %plv = kernel.subgroup.shuffle<index> %phi, %pbp_lo, %x32 : {V4I}, i32, i32")
            e(f"    %phii, %phiv = kernel.subgroup.shuffle<index> %phi, %pbp_hi, %x32 : {V4I}, i32, i32")
            e(f"    %plo = vector.bitcast %pli : {V4I} to {V8H}")
            e(f"    %phi2 = vector.bitcast %phii : {V4I} to {V8H}")
        else:
            e(f"    %phi = vector.bitcast %ph : {V8H} to {V4I}")
            e(f"    %ppi, %ppv = kernel.subgroup.shuffle<xor> %phi, %x16, %x32 : {V4I}, i32, i32")
            e(f"    %pph = vector.bitcast %ppi : {V4I} to {V8H}")
            e(f"    %plo = scf.select %h0, %ph, %pph : {V8H}")
            e(f"    %phi2 = scf.select %h0, %pph, %ph : {V8H}")
        e(f"    %pb0 = vector.concat<0> %plo, %phi2 : {V8H}, {V8H} -> {V16H}")
        e(f"    %pb = vector.fragment<rhs> %pb0 shape [%k, %n] : {V16H}")
        # P.V into the (conditionally) rescaled O
        if PV_FENCE:
            e("    scf.schedule.fence")
        e(f"    %alpha8 = vector.splat %alpha : {V8}")
        outs = []
        for f in range(8):
            e(f"    %vfr{f}c = index.constant {16 * f} : index")
            e(f"    %vfr{f}a = index.add %hd128, %vfr{f}c : index")
            e(f"    %vfr{f} = index.add %vcur, %vfr{f}a : index")
            e(f"    %vf{f} = vector.fragment.load<lhs> %v_view[%vfr{f}, %c0] shape [%m, %k] : view<256x{VT_PITCH}xf16> -> {V16H}")
            e(f"    %nx{f} = vector.mma %vf{f}, %pb, %rso{f} : {V16H}, {V16H}, {V8}")
            outs.append(f"%nx{f}")
        return outs


    NFR = 16 if R2P else 8
    if SOFT1 or R2P:
        e("  %pw32 = index.mul %wqb, %c32 : index")
        e("  %pw16 = index.mul %wqb, %c16 : index")
        e("  %prow_own = index.add %pw32, %lane : index")     # A: this lane's P^T row
        e("  %prow_lo = index.add %pw32, %sub : index")       # B: the h0 lane's row ...
        e("  %prow_hi = index.add %prow_lo, %c16 : index")    # ... and the h1 lane's
        e("  %pa_own = index.add %pw16, %sub : index")
        e(f"  %c128p = index.constant {32 * NQB} : index")   # one P^T subtile slot (128 rows at NQB 4)
    if R2P:
        e("  %pw32o = index.mul %wqbo, %c32 : index")
        e("  %pw16o = index.mul %wqbo, %c16 : index")
        e("  %prow_lo_o = index.add %pw32o, %sub : index")   # P.V: the other head's P^T rows
        e("  %prow_hi_o = index.add %prow_lo_o, %c16 : index")
        e("  %pa_oth = index.add %pw16o, %sub : index")
        if R2P_PVB:
            e("  %pvb16 = index.constant 16 : index")
            e("  %pvb4 = index.constant 4 : index")
            e(f"  %ppb_view = buffer.view %pool[%pp_o] : buffer -> view<{1024 * NQB}xi8>")
            # the h1 lanes' rows (16 rows = 256 B past the h0 lanes'): the same lane address, the view base an immediate
            e(f"  %pph_o = index.constant {S_OFF + (0 if R2P_K2 else 4096) + 256} : offset")
            e(f"  %pph_view = buffer.view %pool[%pph_o] : buffer -> view<{1024 * NQB - 256}xi8>")
            e(f"  %alb_view = buffer.view %pool[%al_o] : buffer -> view<{128 * NQB}xi8>")
            e("  %pvb_lo = index.mul %prow_lo, %pvb16 : index")
            e("  %pvb_lo_o = index.mul %prow_lo_o, %pvb16 : index")
            e("  %pab_own = index.mul %pa_own, %pvb4 : index")
            e("  %pab_oth = index.mul %pa_oth, %pvb4 : index")
        e("  %xso_r0 = index.add %pw32o, %lane : index")     # QK: the other head's partial (plane 0 / 1 rows) ...
        e(f"  %xso_r1 = index.add %xso_r0, {f'%c{64 * NQB}' if R2P_K2 else '%c128'} : index")
        e("  %xsm_r0 = index.add %pw32, %lane : index")      # ... and the partner's of the own head
        e(f"  %xsm_r1 = index.add %xsm_r0, {f'%c{64 * NQB}' if R2P_K2 else '%c128'} : index")
    if R2P:
        rnames = [f"%rb{f}" for f in range(16)]
        BK = "vector<8xi32>" if R2P else V8
        rtypes = ", ".join([BK] * 16)

    def r2p_pv(rows, alb, R, pp, ind, hook=None):
        """R2P, P.V wave: rescale each head's half of the bank (0..7: own head, 8..15: the other) by its alpha of that
        tile unless all its rows' are 1 (x 1.0 is exact), both heads' P^T from the slot, then 8 V^T fragments of this
        head-dim half, each feeding both heads' MMAs. Returns the new bank names."""
        if R2P_PVB:
            e(f"{ind}%{pp}rb = index.mul {rows}, %pvb16 : index")
            e(f"{ind}%{pp}ab = index.mul {alb}, %pvb4 : index")
            for nm, base in (("al", "%pab_own"), ("alo", "%pab_oth")):
                e(f"{ind}%{pp}{nm}a = index.add {base}, %{pp}ab : index")
                e(f"{ind}%{pp}{nm}v = vector.load %alb_view[%{pp}{nm}a] : view<{128 * NQB}xi8> -> vector<4xi8>")
                e(f"{ind}%{pp}{nm}w = vector.bitcast %{pp}{nm}v : vector<4xi8> to vector<1xf32>")
                e(f"{ind}%{pp}{nm} = vector.extract %{pp}{nm}w[0] : vector<1xf32> -> f32")
        else:
            e(f"{ind}%{pp}ai = index.add {alb}, %pa_own : index")
            e(f"{ind}%{pp}al = view.load %al_view[%{pp}ai] : view<{32 * NQB}xf32> -> f32")
            e(f"{ind}%{pp}aio = index.add {alb}, %pa_oth : index")
            e(f"{ind}%{pp}alo = view.load %al_view[%{pp}aio] : view<{32 * NQB}xf32> -> f32")
        Rf = []
        for f in range(16):
            e(f"{ind}%{pp}rf{f} = vector.bitcast {R[f]} : {BK} to {V8}")
            Rf.append(f"%{pp}rf{f}")
        ftypes = ", ".join([V8] * 8)
        sc = [f"%{pp}sc{f}" for f in range(16)]
        for t, a in enumerate((f"%{pp}al", f"%{pp}alo")):   # per head: its fragments only when its rows' max grew
            e(f"{ind}%{pp}gr{t} = scalar.cmpf oeq, {a}, %one : f32")
            e(f"{ind}%{pp}ag{t} = kernel.subgroup.vote.all %{pp}gr{t} : i1")
            fs = range(8 * t, 8 * t + 8)
            e(f"{ind}{', '.join(sc[f] for f in fs)} = scf.if %{pp}ag{t} -> ({ftypes}) {{")
            e(f"{ind}  scf.yield {', '.join(Rf[f] for f in fs)} : {ftypes}")
            e(f"{ind}}} else {{")
            e(f"{ind}  %{pp}al8_{t} = vector.splat {a} : {V8}")
            for f in fs:
                e(f"{ind}  %{pp}os{f} = vector.mulf {Rf[f]}, %{pp}al8_{t} : {V8}")
            e(f"{ind}  scf.yield {', '.join(f'%{pp}os{f}' for f in fs)} : {ftypes}")
            e(f"{ind}}}")
        pvp = [int(k[5:]) for k in R2P_DIAG if k.startswith("pvpad")]
        if pvp:   # diag: pvpadN = N extra V8 mul ops (4 chains) on fragment 0's accumulator, in the P.V wave
            ends = []
            for k in range(4):
                e(f"{ind}%{pp}vpk{k} = scalar.constant {k + 1}.5 : f32")
                e(f"{ind}%{pp}vpc{k} = vector.splat %{pp}vpk{k} : {V8}")
                e(f"{ind}%{pp}vp{k}_0 = vector.addf {sc[0]}, %{pp}vpc{k} : {V8}")
                for j in range(1, pvp[0] // 4):
                    e(f"{ind}%{pp}vp{k}_{j} = vector.mulf %{pp}vp{k}_{j - 1}, {sc[0]} : {V8}")
                ends.append(f"%{pp}vp{k}_{pvp[0] // 4 - 1}")
            cur = sc[0]
            for i, x in enumerate(ends):
                e(f"{ind}%{pp}spv{i} = vector.addf {cur}, {x} : {V8}")
                cur = f"%{pp}spv{i}"
            sc[0] = cur
        pbs = []
        for t, (lo, hi) in enumerate((("%prow_lo", "%prow_hi"), ("%prow_lo_o", "%prow_hi_o"))):
            if R2P_PVB:
                e(f"{ind}%{pp}rl{t} = index.add {('%pvb_lo', '%pvb_lo_o')[t]}, %{pp}rb : index")
                e(f"{ind}%{pp}rlb{t} = vector.load %ppb_view[%{pp}rl{t}] : view<{1024 * NQB}xi8> -> vector<16xi8>")
                e(f"{ind}%{pp}rhb{t} = vector.load %pph_view[%{pp}rl{t}] : view<{1024 * NQB - 256}xi8> -> vector<16xi8>")
                e(f"{ind}%{pp}plo{t} = vector.bitcast %{pp}rlb{t} : vector<16xi8> to {V8H}")
                e(f"{ind}%{pp}phi{t} = vector.bitcast %{pp}rhb{t} : vector<16xi8> to {V8H}")
                e(f"{ind}%{pp}pb0{t} = vector.concat<0> %{pp}plo{t}, %{pp}phi{t} : {V8H}, {V8H} -> {V16H}")
                e(f"{ind}%{pp}pb{t} = vector.fragment<rhs> %{pp}pb0{t} shape [%k, %n] : {V16H}")
                pbs.append(f"%{pp}pb{t}")
                continue
            e(f"{ind}%{pp}rl{t} = index.add {rows}, {lo} : index")
            e(f"{ind}%{pp}rh{t} = index.add {rows}, {hi} : index")
            e(f"{ind}%{pp}plo{t} = vector.load %pp_view[%{pp}rl{t}, %c0] : view<{64 * NQB}x8xf16> -> {V8H}")
            e(f"{ind}%{pp}phi{t} = vector.load %pp_view[%{pp}rh{t}, %c0] : view<{64 * NQB}x8xf16> -> {V8H}")
            e(f"{ind}%{pp}pb0{t} = vector.concat<0> %{pp}plo{t}, %{pp}phi{t} : {V8H}, {V8H} -> {V16H}")
            e(f"{ind}%{pp}pb{t} = vector.fragment<rhs> %{pp}pb0{t} shape [%k, %n] : {V16H}")
            pbs.append(f"%{pp}pb{t}")
        if hook:
            hook("p")
        outs = [None] * 16

        def vload(f):
            e(f"{ind}%{pp}vr{f}c = index.constant {16 * f} : index")
            e(f"{ind}%{pp}vr{f} = index.add %hd128, %{pp}vr{f}c : index")
            if R2P_K2:   # the lane's row's key halves from the two planes (fragment row = lane & 15)
                for h in range(2):
                    off = 4096 * h + 256 * f
                    e(f"{ind}%{pp}vx{f}{h} = vector.load %vgv{f + 8 * h}[%vgl] : view<{VT_BYTES - off}xi8> -> vector<16xi8>")
                    e(f"{ind}%{pp}vh{f}{h} = vector.bitcast %{pp}vx{f}{h} : vector<16xi8> to {V8H}")
                e(f"{ind}%{pp}vc{f} = vector.concat<0> %{pp}vh{f}0, %{pp}vh{f}1 : {V8H}, {V8H} -> {V16H}")
                e(f"{ind}%{pp}vf{f} = vector.fragment<lhs> %{pp}vc{f} shape [%m, %k] : {V16H}")
                return
            e(f"{ind}%{pp}vf{f} = vector.fragment.load<lhs> %v_view[%{pp}vr{f}, %c0] shape [%m, %k] : view<256x{VT_PITCH}xf16> -> {V16H}")
        if R2P_PVDB:
            assert "halfv" not in R2P_DIAG
            vload(0)
            e(f"{ind}scf.schedule.fence")
        for f in range(8):
            vc = f - 1 if "halfv" in R2P_DIAG and f % 2 else f
            if R2P_PVDB:
                if f < 7:
                    vload(f + 1)
                    e(f"{ind}scf.schedule.fence")
            elif vc == f:
                vload(f)
            if "dblv" in R2P_DIAG:   # diag: the second head's MMA from its own fragment load (cols + 8, in the pad)
                e(f"{ind}%{pp}vf{f}b = vector.fragment.load<lhs> %v_view[%{pp}vr{f}, %c8] shape [%m, %k] : view<256x{VT_PITCH}xf16> -> {V16H}")
            for t in range(2):
                o = 8 * t + f
                vfr = f"%{pp}vf{f}b" if t and "dblv" in R2P_DIAG else f"%{pp}vf{vc}"
                e(f"{ind}%{pp}oi{o} = vector.fragment<init> {sc[o]} shape [%m, %n] : {V8}")
                e(f"{ind}%{pp}nx{o} = vector.mma {vfr}, {pbs[t]}, %{pp}oi{o} : {V16H}, {V16H}, {V8}")
                e(f"{ind}%{pp}nxi{o} = vector.bitcast %{pp}nx{o} : {V8} to {BK}")
                outs[o] = f"%{pp}nxi{o}"
            if hook:
                hook(str(f))
            if f < 7 and (R2P_PVDB or (f + 1) % R2P_PVF == 0):
                e(f"{ind}scf.schedule.fence")
        return outs

    def r2p_qk(R, ind):
        """R2P, QK wave: both heads' S^T partials over this head-dim half, each K fragment feeding both heads' chains
        (each chain the main kernel's emit_qk order). Returns (own head's, other head's)."""
        for c in range(16):
            e(f"{ind}%qb16_{c} = vector.bitcast {R[c]} : {BK} to {V16H}")
            e(f"{ind}%qf{c} = vector.fragment<rhs> %qb16_{c} shape [%k, %n] : {V16H}")
        e(f"{ind}%r2z = vector.fragment<init> %zeros8 shape [%m, %n] : {V8}")
        if R2P_K2:   # K(ks) from slot (ks / 16) % 2: a runtime row (a parity branch's join costs 8 v_mov)
            return r2p_qk_chain(ind, "%k2_view", "%kslot")
        return r2p_qk_chain(ind, "%k_view")

    def r2p_qk_chain(ind, kv, krow="%c0"):
        ch4 = "chain4" in R2P_DIAG   # 4 chains: per head, even and odd K fragments, summed at the end (sum order moves)
        acc = ["%r2z"] * (4 if ch4 else 2)
        for c in range(8):
            e(f"{ind}%kfc{c}c = index.constant {16 * c} : index")
            e(f"{ind}%kfc{c} = index.add %hd128, %kfc{c}c : index")
            kc = c - 1 if "halfk" in R2P_DIAG and c % 2 else c
            if kc == c:
                kvt = f"view<{32 if kv == '%k2_view' else KT}x{KT_PITCH}xf16>"
                e(f"{ind}%kf{c} = vector.fragment.load<lhs> {kv}[{krow}, %kfc{c}] shape [%m, %k] : {kvt} -> {V16H}")
            if "dblk" in R2P_DIAG:   # diag: the second head's MMA from its own fragment load (cols + 8, in the pad)
                e(f"{ind}%kfc{c}b = index.add %kfc{c}, %c8 : index")
                e(f"{ind}%kf{c}b = vector.fragment.load<lhs> %k_view[%c0, %kfc{c}b] shape [%m, %k] : view<{KT}x{KT_PITCH}xf16> -> {V16H}")
            for t in range(2):
                a = 2 * t + c % 2 if ch4 else t
                kfr = f"%kf{c}b" if t and "dblk" in R2P_DIAG else f"%kf{kc}"
                e(f"{ind}%sa{t}_{c} = vector.mma {kfr}, %qf{8 * t + c}, {acc[a]} : {V16H}, {V16H}, {V8}")
                acc[a] = f"%sa{t}_{c}"
            if CHAIN_HOOK[0] is not None:
                CHAIN_HOOK[0](c)
            if c < 7 and (c + 1) % R2P_QKF == 0:
                e(f"{ind}scf.schedule.fence")
        if ch4:
            for t in range(2):
                e(f"{ind}%sc4_{t} = vector.addf {acc[2 * t]}, {acc[2 * t + 1]} : {V8}")
            return [f"%sc4_{t}" for t in range(2)]
        return acc

    UNR_PAR = [None]
    CHAIN_HOOK = [None]   # called after each K step's MMAs (KRAW: the scale reads, so they overlap the last steps)

    def body_r2p(first=False, role=None):
        """R2P tile ks, one role branch holding both phases (and both barriers: every wave meets each once).
        QK waves: QK(ks), the other head's partial to LDS | barrier A | K(ks + 16) loads, own S = own + partner partial,
        softmax -> P^T / alpha slot ks % 2 | barrier B.
        P.V waves: stage V(ks - 16) (V(0) at ks = 0, read by P.V(-1) with P = 0), K(ks + 16) loads | barrier A | stage
        K(ks + 16), P.V(ks - 16) from the other slot | barrier B. They stage both tiles (128 threads, 4 loads each): the
        QK waves' peak (Q of 2 heads + S + the softmax) has no room for staging registers."""
        I = "    "
        e(f"{I}%ks16 = index.add %ks_, %c16 : index")
        e(f"{I}%kt16 = index.div %ks_, %c16 : index")
        if UNR_PAR[0] is None:
            e(f"{I}%kpar = index.rem %kt16, %c2 : index")
            e(f"{I}%kpar1 = index.sub %c1, %kpar : index")
        else:   # R2P_UNR: this body's tile parity
            e(f"{I}%kpar = index.constant {UNR_PAR[0]} : index")
            e(f"{I}%kpar1 = index.constant {1 - UNR_PAR[0]} : index")
        e(f"{I}%slot_rows = index.mul %kpar, %c128p : index")
        e(f"{I}%slot_al = index.mul %kpar, %c{16 * NQB} : index")
        e(f"{I}%prev_rows = index.mul %kpar1, %c128p : index")
        e(f"{I}%prev_al = index.mul %kpar1, %c{16 * NQB} : index")
        if R2P_K2:
            e(f"{I}%kslot = index.mul %kpar, %c16 : index")
            e(f"{I}%kslot1 = index.mul %kpar1, %c16 : index")
        nb = [f"%nb{f}" for f in range(16)]
        J = I + "  "
        if role is None:
            e(f"{I}{', '.join(nb)}, %qmax, %qsum = scf.if %isA -> ({rtypes}, f32, f32) {{")
        if role == "B":
            return body_r2p_pv(first, I, J)
        if R2P_PRIOQ:
            e(f"{J}kernel.priority {{level = {R2P_PRIOQ}}}")
        assert not (KRAW and "nox" in R2P_DIAG)
        kry = "kry" in R2P_DIAG   # (diag kry: no pre-scale; both halves' scales read and applied after barrier A)

        def kr_loads(t, bases=("%kslo",), view="%ksl_view", vt="view<64xf32>"):   # f32 scales of the lane's 8 keys, from slot kpar (as %krh{t}, %krh{t+1})
            sl = f"%krsl{t}"
            e(f"{J}{sl} = index.mul %kpar, %c{16 if KH else 32} : index")
            for t, base in enumerate(bases, t):
                e(f"{J}%kri{t} = index.add {sl}, {base} : index")
                if "krconst" in R2P_DIAG:   # diag: the scales' VALU without their loads (s = the lane's tile parity + 1)
                    e(f"{J}%krc{t}i = index.add %kpar, %c1 : index")
                    e(f"{J}%krc{t}c = index.cast %krc{t}i : index to i32")
                    e(f"{J}%krc{t}f = scalar.sitofp %krc{t}c : i32 to f32")
                    e(f"{J}%krh{t} = vector.splat %krc{t}f : {V8}")
                else:
                    e(f"{J}%krh{t} = vector.load {view}[%kri{t}] : {vt} -> {V8}")
        krc = KRAW_RD_STEP if KRAW and not kry and not KH else None
        if krc is not None:
            CHAIN_HOOK[0] = lambda c: kr_loads(0) if c == krc else None
        own, oth = r2p_qk(rnames, J)
        CHAIN_HOOK[0] = None
        if KRAW and not kry and not KH:   # the partner's partial leaves scaled (same keys, same half: this half's s), so the softmax-critical
            # section after barrier A keeps 4 dual ops: S = fma(own, s, partner's) for the old own + partner's
            e(f"{J}scf.schedule.fence")
            if krc is None:
                kr_loads(0)
            e(f"{J}%othk = vector.mulf {oth}, %krh0 : {V8}")
            oth = "%othk"
        if "nox" in R2P_DIAG:
            if not R2P_DIAG & {"nobar", "nobara"}:
                e(f"{J}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            e(f"{J}%s0 = vector.addf {own}, {oth} : {V8}")
        else:
            e(f"{J}%xst0 = vector.slice {oth}[0] : {V8} -> {V4}")
            e(f"{J}%xst1 = vector.slice {oth}[4] : {V8} -> {V4}")
            if R2P_K2:   # plane 0 rows + the tile's P^T slot (under the parity branch: immediate offsets)
                xv = f"view<{96 * NQB}x4xf32>"
                e(f"{J}%xsl0 = index.cmp eq, %kpar, %c0 : index")
                e(f"{J}scf.if %xsl0 {{")
                e(f"{J}  vector.store %xst0, %xs_view[%xso_r0, %c0] : {V4}, {xv}")
                e(f"{J}}} else {{")
                e(f"{J}  %xso_r0s = index.add %xso_r0, %c128p : index")
                e(f"{J}  vector.store %xst0, %xs_view[%xso_r0s, %c0] : {V4}, {xv}")
                e(f"{J}}}")
                e(f"{J}vector.store %xst1, %xs_view[%xso_r1, %c0] : {V4}, {xv}")
                if KH and "khpre" in R2P_DIAG:   # after the partner's partial left: s_big and own s_big in the
                    # barrier-A wait (S = fmac(partner's, s_big, own s_big) after it: the base's op count there)
                    e(f"{J}scf.schedule.fence")   # (else the read hoists above the stores: 203 VGPRs)
                    kr_loads(1, ("%ksc3",))
                    e(f"{J}%ownk = vector.mulf {own}, %krh1 : {V8}")
            else:
                e(f"{J}vector.store %xst0, %xs_view[%xso_r0, %c0] : {V4}, view<256x4xf32>")
                e(f"{J}vector.store %xst1, %xs_view[%xso_r1, %c0] : {V4}, view<256x4xf32>")
            if not R2P_DIAG & {"nobar", "nobara"}:
                e(f"{J}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            if R2P_PRIO or R2P_PRIOQ:
                e(f"{J}kernel.priority {{level = {R2P_PRIO}}}")
            if KH and "khpre" not in R2P_DIAG:   # (s_big read after barrier A, beside the partials'; diag khpre: own x s_big in the barrier-A wait, -2.4% cycles worse)
                kr_loads(1, ("%ksc3",))
            elif KH and "khlive" not in R2P_DIAG:   # read again (live across the barrier: 203 VGPRs; diag khlive)
                kr_loads(2, ("%ksc3",), "%ksl_view2", "view<32xf32>")
            elif kry:
                kr_loads(2, ("%kslp",))
            elif KRAW and "krre" in R2P_DIAG:   # (diag krre: read again after barrier A: its post-barrier LDS burst
                # (every wave's partials at once) is bandwidth-bound: +2 b128 per lane cost ~50 ticks per tile)
                kr_loads(1)
            if R2P_K2:   # plane 0 from the tile's slot under the parity branch, one view per slot (immediate offsets: a
                # runtime row puts 3 dependent VALU between barrier A and the softmax: -2.2 at 8K, -4.3 at 64K)
                xv = f"view<{96 * NQB}x4xf32>"
                e(f"{J}%spb = vector.load %xs_view[%xsm_r1, %c0] : {xv} -> {V4}")
                e(f"{J}%xsl1 = index.cmp eq, %kpar, %c0 : index")
                e(f"{J}%spa = scf.if %xsl1 -> ({V4}) {{")
                e(f"{J}  %spa0 = vector.load %xsp0_view[%xsm_r0, %c0] : view<{32 * NQB}x4xf32> -> {V4}")
                e(f"{J}  scf.yield %spa0 : {V4}")
                e(f"{J}}} else {{")
                e(f"{J}  %spa1 = vector.load %xsp1_view[%xsm_r0, %c0] : view<{32 * NQB}x4xf32> -> {V4}")
                e(f"{J}  scf.yield %spa1 : {V4}")
                e(f"{J}}}")
            else:
                e(f"{J}%spa = vector.load %xs_view[%xsm_r0, %c0] : view<256x4xf32> -> {V4}")
                e(f"{J}%spb = vector.load %xs_view[%xsm_r1, %c0] : view<256x4xf32> -> {V4}")
            e(f"{J}%spart = vector.concat<0> %spa, %spb : {V4}, {V4} -> {V8}")
            if KH and "khpre" not in R2P_DIAG and "khfmac" in R2P_DIAG:   # (diag khfmac: partner's x s, then fmac own)
                e(f"{J}%s0a = vector.mulf %spart, %krh1 : {V8}")
                e(f"{J}%s0 = vector.fmaf {own}, %krh1, %s0a : {V8}")
            elif KH and "khpre" not in R2P_DIAG:
                e(f"{J}%s0a = vector.addf {own}, %spart : {V8}")
                e(f"{J}%s0 = vector.mulf %s0a, %krh1 : {V8}")
            elif KH:
                e(f"{J}%s0 = vector.fmaf %spart, {'%krh1' if 'khlive' in R2P_DIAG else '%krh2'}, %ownk : {V8}")
            elif KRAW:
                if kry:   # the partner's half first, then (fenced: one 8-VGPR scale live) this half's
                    e(f"{J}%sprt = vector.mulf %spart, %krh2 : {V8}")
                    e(f"{J}scf.schedule.fence")
                    kr_loads(1)
                    e(f"{J}%s0 = vector.fmaf {own}, %krh1, %sprt : {V8}")
                else:
                    e(f"{J}%s0 = vector.fmaf {own}, {'%krh1' if 'krre' in R2P_DIAG else '%krh0'}, %spart : {V8}")
            else:
                e(f"{J}%s0 = vector.addf {own}, %spart : {V8}")
        emit_softmax_pv(None, True, pair_slot=("%slot_rows", "%slot_al"))
        if R2P_PRIO:
            e(f"{J}kernel.priority {{level = 0}}")
        if not R2P_DIAG & {"nobar", "nobarb"}:
            e(f"{J}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        if role == "A":
            return ["%nmax", "%nsum"]
        e(f"{J}scf.yield {', '.join(rnames)}, %nmax, %nsum : {rtypes}, f32, f32")
        e(f"{I}}} else {{")
        outs = body_r2p_pv(first, I, J)
        e(f"{J}scf.yield {', '.join(outs)}, %rmax, %rsum : {rtypes}, f32, f32")
        e(f"{I}}}")
        return nb

    def body_r2p_pv(first, I, J):
        """body_r2p's P.V-wave half. Returns the O bank names."""
        e(f"{J}%vks1 = index.max %ks_, %c16 : index")
        e(f"{J}%vks = index.sub %vks1, %c16 : index")
        nostg = "nostg" in R2P_DIAG   # (diag nostg: no K / V loads or staging stores in the loop: stale tiles)
        if not nostg:
            lv, lk = "%vks", "%ks16"
            if "kvhot" in R2P_DIAG:   # diag: K / V loads from the first 64 rows (tile start min 48) (cache-hot; staging keys unchanged)
                e(f"{J}%kvh64 = index.constant 48 : index")
                e(f"{J}%vksh = index.min %vks, %kvh64 : index")
                e(f"{J}%ks16h = index.min %ks16, %kvh64 : index")
                lv, lk = "%vksh", "%ks16h"
            nv = ["%zh8"] * 4 if R2P_DIAG & {"novl", "nostv"} else load_v(lv, "nv", J)   # (diag novl: V staged as zeros)
            nk = ["%zh8"] * NKS() if R2P_DIAG & {"nokl", "nostk"} else load_k(lk, "nk", J)   # (diag nokl: K staged as zeros)
            if "nostv" not in R2P_DIAG:   # (diag nostv / nostk: no V / K loads or LDS stores: stale tiles)
                stage_v(nv, "%c0", "stv", J)
            if "kstA" in R2P_DIAG:   # diag: K stored in phase A, before barrier A (races the QK reads: wrong values)
                stage_k("%ks16", nk, "st", J)
        def k2_store():   # slot (ks / 16 + 1) % 2 under a parity branch: loop-invariant addresses + an immediate
            e(f"{J}%kst0 = index.cmp eq, %kpar, %c1 : index")
            e(f"{J}scf.if %kst0 {{")
            stage_k("%ks16", nk, "st", J + "  ", kv="%k2_view", roff="%c0")
            e(f"{J}}} else {{")
            stage_k("%ks16", nk, "st1", J + "  ", kv="%k2_view", roff="%c16")
            e(f"{J}}}")
        if not nostg:
            if R2P_K2 and "k2late" not in R2P_DIAG:   # (diag k2late: after barrier A, as with one slot)
                k2_store()
            if KRAW and "nokst" not in R2P_DIAG:   # P.V wave 0's lanes 0..3 (diag nokst: no scale staging: stale)
                assert R2P_K2 and "k2late" not in R2P_DIAG
                e(f"{J}%kspl = index.rem %tid, %c{32 * NQB} : index")
                e(f"{J}%kspc = index.cmp ult, %kspl, %c{2 if KH else 4} : index")
                e(f"{J}%ksps = index.mul %kpar1, %c{16 if KH else 32} : index")
                stage_ks("%ks16", "%ksps", "%kspc", "ksp", J)
        if not R2P_DIAG & {"nobar", "nobara"}:
            e(f"{J}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        if not nostg and R2P_K2 and "k2late" in R2P_DIAG:
            k2_store()
        kst = not nostg and not R2P_DIAG & {"nostk", "kstA"} and not R2P_K2
        if kst and (first or not R2P_KPOS):
            stage_k("%ks16", nk, "st", J)
        hook = (lambda at: stage_k("%ks16", nk, "st", J) if at == R2P_KPOS else None) if kst and R2P_KPOS else None
        outs = rnames if first else r2p_pv("%prev_rows", "%prev_al", rnames, "bp", J, hook)   # R2P_PEEL: no P.V(-1)
        if not R2P_DIAG & {"nobar", "nobarb"}:
            e(f"{J}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        return outs




    def soft1_pv(ind, onm=None, px="s1", rows=None, alb=None):
        """SOFT1, both waves: rescale the O half by A's alpha (pass-through first), P^T from LDS, 8 P.V MMAs."""
        onm = onm or onames
        if alb is not None:
            e(f"{ind}%{px}ai = index.add {alb}, %pa_own : index")
        e(f"{ind}%{px}al = view.load %al_view[{f'%{px}ai' if alb is not None else '%pa_own'}] : view<{PRS2}xf32> -> f32")
        sc = [f"%{px}sc{f}" for f in range(8)]
        otypes8 = ", ".join([V8] * 8)

        def arm_pass():
            e(f"{ind}  scf.yield {', '.join(onm)} : {otypes8}")

        def arm_mul():
            e(f"{ind}  %{px}al8 = vector.splat %{px}al : {V8}")
            for f in range(8):
                e(f"{ind}  %{px}os{f} = vector.mulf {onm[f]}, %{px}al8 : {V8}")
            e(f"{ind}  scf.yield {', '.join(f'%{px}os{f}' for f in range(8))} : {otypes8}")
        if S1_PASS_FIRST:
            e(f"{ind}%{px}gr = scalar.cmpf oeq, %{px}al, %one : f32")
            e(f"{ind}%{px}ag = kernel.subgroup.vote.all %{px}gr : i1")
        else:
            e(f"{ind}%{px}gr = scalar.cmpf une, %{px}al, %one : f32")
            e(f"{ind}%{px}ag = kernel.subgroup.vote.any %{px}gr : i1")
        e(f"{ind}{', '.join(sc)} = scf.if %{px}ag -> ({otypes8}) {{")
        (arm_pass if S1_PASS_FIRST else arm_mul)()
        e(f"{ind}}} else {{")
        (arm_mul if S1_PASS_FIRST else arm_pass)()
        e(f"{ind}}}")
        rl, rh = "%prow_lo", "%prow_hi"
        if rows is not None:
            e(f"{ind}%{px}rl = index.add {rows}, %prow_lo : index")
            e(f"{ind}%{px}rh = index.add {rows}, %prow_hi : index")
            rl, rh = f"%{px}rl", f"%{px}rh"
        e(f"{ind}%{px}plo = vector.load %pp_view[{rl}, %c0] : view<{PRS}x8xf16> -> {V8H}")
        e(f"{ind}%{px}phi = vector.load %pp_view[{rh}, %c0] : view<{PRS}x8xf16> -> {V8H}")
        e(f"{ind}%{px}pb0 = vector.concat<0> %{px}plo, %{px}phi : {V8H}, {V8H} -> {V16H}")
        e(f"{ind}%{px}pb = vector.fragment<rhs> %{px}pb0 shape [%k, %n] : {V16H}")
        outs = []
        for f in range(8):
            e(f"{ind}%{px}vc{f} = index.constant {16 * f} : index")
            e(f"{ind}%{px}vr{f} = index.add %hd128, %{px}vc{f} : index")
            e(f"{ind}%{px}vf{f} = vector.fragment.load<lhs> %v_view[%{px}vr{f}, %c0] shape [%m, %k] : view<256x{VT_PITCH}xf16> -> {V16H}")
            e(f"{ind}%{px}nx{f} = vector.mma %{px}vf{f}, %{px}pb, {sc[f]} : {V16H}, {V16H}, {V8}")
            outs.append(f"%{px}nx{f}")
        return outs

    def body_soft1(tail):
        I = "    "
        e(f"{I}%vcur = index.add %c0, %c0 : index")
        e(f"{I}%ks16 = index.add %ks_, %c16 : index")
        nk = load_k("%ks16", "nk", I)
        nv = load_v("%ks_", "nv", I)
        acc = emit_qk("%k_view", "", I)
        e(f"{I}scf.if %isBs {{")
        e(f"{I}  %sst0 = vector.slice {acc}[0] : {V8} -> {V4}")
        e(f"{I}  %sst1 = vector.slice {acc}[4] : {V8} -> {V4}")
        e(f"{I}  vector.store %sst0, %s_view[%tid, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"{I}  vector.store %sst1, %s_view[%tid2, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"{I}}}")
        stage_v(nv, "%c0", "stv", I)
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        stage_k("%ks16", nk, "st", I)
        e(f"{I}%qmax, %qsum = scf.if %isA -> (f32, f32) {{")
        e(f"{I}  %spa = vector.load %s_view[%ptid, %c0] : view<{2 * NT}x4xf32> -> {V4}")
        e(f"{I}  %spb = vector.load %s_view[%ptid2, %c0] : view<{2 * NT}x4xf32> -> {V4}")
        e(f"{I}  %spart = vector.concat<0> %spa, %spb : {V4}, {V4} -> {V8}")
        e(f"{I}  %s0 = vector.addf {acc}, %spart : {V8}")
        emit_softmax_pv(None, tail, pair_slot=("%c0", "%c0"))
        e(f"{I}  scf.yield %nmax, %nsum : f32, f32")
        e(f"{I}}} else {{")
        e(f"{I}  scf.yield %rmax, %rsum : f32, f32")
        e(f"{I}}}")
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        outs = soft1_pv(I)
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        return outs


    def body_soft2(tail):
        I = "    "
        e(f"{I}%vcur = index.add %c0, %c0 : index")
        e(f"{I}%ks16 = index.add %ks_, %c16 : index")
        nk = load_k("%ks16", "nk", I)
        nv = load_v("%ks_", "nv", I)
        acc = emit_qk("%k_view", "", I)
        e(f"{I}scf.if %isBs {{")
        e(f"{I}  %sst0 = vector.slice {acc}[0] : {V8} -> {V4}")
        e(f"{I}  %sst1 = vector.slice {acc}[4] : {V8} -> {V4}")
        e(f"{I}  vector.store %sst0, %s_view[%tid, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"{I}  vector.store %sst1, %s_view[%tid2, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"{I}}}")
        otypes8 = ", ".join([V8] * 8)
        cur_slot = prev_slot = (None, None)
        if SOFT3:   # P^T / alpha slot: tile parity (A writes this tile's, P.V reads the previous tile's)
            e(f"{I}%s3t = index.div %ks_, %c16 : index")
            e(f"{I}%s3p = index.rem %s3t, %c2 : index")
            e(f"{I}%s3q = index.sub %c1, %s3p : index")
            e(f"{I}%s3rows = index.mul %s3p, %c128 : index")
            e(f"{I}%s3al = index.mul %s3p, %c64 : index")
            e(f"{I}%s3prows = index.mul %s3q, %c128 : index")
            e(f"{I}%s3pal = index.mul %s3q, %c64 : index")
            cur_slot, prev_slot = ("%s3rows", "%s3al"), ("%s3prows", "%s3pal")
        if SOFT3:   # phase 1: only A's P.V(j - 1)
            ph1 = [f"%p1o{f}" for f in range(8)]
            # pass-through arm first: O's last use is then the P.V, which accumulates in place
            e(f"{I}{', '.join(ph1)} = scf.if %isBs -> ({otypes8}) {{")
            e(f"{I}  scf.yield {', '.join(onames)} : {otypes8}")
            e(f"{I}}} else {{")
            o1 = soft1_pv(I + "  ", onames, "pa", *prev_slot)
            e(f"{I}  scf.yield {', '.join(o1)} : {otypes8}")
            e(f"{I}}}")
            outs = ph1
        else:
            outs = soft1_pv(I)            # P.V of the previous tile (zero P / V before the first)
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        stage_v_late = SOFT3   # under SOFT3, B still reads V(j - 1) in phase 2: stage V(j) after the barrier below
        if not stage_v_late:
            stage_v(nv, "%c0", "stv", I)
        stage_k("%ks16", nk, "st", I)
        e(f"{I}%qmax, %qsum = scf.if %isA -> (f32, f32) {{")
        e(f"{I}  %spa = vector.load %s_view[%ptid, %c0] : view<{2 * NT}x4xf32> -> {V4}")
        e(f"{I}  %spb = vector.load %s_view[%ptid2, %c0] : view<{2 * NT}x4xf32> -> {V4}")
        e(f"{I}  %spart = vector.concat<0> %spa, %spb : {V4}, {V4} -> {V8}")
        e(f"{I}  %s0 = vector.addf {acc}, %spart : {V8}")
        emit_softmax_pv(None, tail, pair_slot=cur_slot if SOFT3 else ("%c0", "%c0"))
        e(f"{I}  scf.yield %nmax, %nsum : f32, f32")
        e(f"{I}}} else {{")
        e(f"{I}  scf.yield %rmax, %rsum : f32, f32")
        e(f"{I}}}")
        if SOFT3:   # phase 2: B's P.V(j - 1) (P^T(j - 1) / alpha(j - 1) are only rewritten after this barrier pair)
            ph2 = [f"%p2o{f}" for f in range(8)]
            e(f"{I}{', '.join(ph2)} = scf.if %isA -> ({otypes8}) {{")
            e(f"{I}  scf.yield {', '.join(outs)} : {otypes8}")
            e(f"{I}}} else {{")
            o2 = soft1_pv(I + "  ", outs, "pb", *prev_slot)
            e(f"{I}  scf.yield {', '.join(o2)} : {otypes8}")
            e(f"{I}}}")
            outs = ph2
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        if SOFT3:   # V(j) after everyone's P.V(j - 1); a third barrier before the next tile reads it
            stage_v(nv, "%c0", "stv", I)
            e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        return outs


    def body_soft4(tail):
        I = "    "
        e(f"{I}%vcur = index.add %c0, %c0 : index")
        e(f"{I}%ks16 = index.add %ks_, %c16 : index")
        e(f"{I}%ksm0 = index.max %ks_, %c16 : index")
        e(f"{I}%ksm = index.sub %ksm0, %c16 : index")      # V(j - 1); tile 0 reloads tile 0 (finite, times a zero P)
        nk = load_k("%ks16", "nk", I)
        nv = load_v("%ksm", "nv", I)
        e(f"{I}%s3t = index.div %ks_, %c16 : index")
        e(f"{I}%s3p = index.rem %s3t, %c2 : index")
        e(f"{I}%s3q = index.sub %c1, %s3p : index")
        e(f"{I}%s3rows = index.mul %s3p, %c128 : index")
        e(f"{I}%s3al = index.mul %s3p, %c64 : index")
        e(f"{I}%s3prows = index.mul %s3q, %c128 : index")
        e(f"{I}%s3pal = index.mul %s3q, %c64 : index")
        acc = emit_qk("%k_view", "", I)
        e(f"{I}scf.if %isBs {{")
        e(f"{I}  %sst0 = vector.slice {acc}[0] : {V8} -> {V4}")
        e(f"{I}  %sst1 = vector.slice {acc}[4] : {V8} -> {V4}")
        e(f"{I}  vector.store %sst0, %s_view[%tid, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"{I}  vector.store %sst1, %s_view[%tid2, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"{I}}}")
        stage_v(nv, "%c0", "stv", I)
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        stage_k("%ks16", nk, "st", I)
        outs = soft1_pv(I, onames, "s4", "%s3prows", "%s3pal")
        e(f"{I}%qmax, %qsum = scf.if %isA -> (f32, f32) {{")
        e(f"{I}  %spa = vector.load %s_view[%ptid, %c0] : view<{2 * NT}x4xf32> -> {V4}")
        e(f"{I}  %spb = vector.load %s_view[%ptid2, %c0] : view<{2 * NT}x4xf32> -> {V4}")
        e(f"{I}  %spart = vector.concat<0> %spa, %spb : {V4}, {V4} -> {V8}")
        e(f"{I}  %s0 = vector.addf {acc}, %spart : {V8}")
        emit_softmax_pv(None, tail, pair_slot=("%s3rows", "%s3al"))
        e(f"{I}  scf.yield %nmax, %nsum : f32, f32")
        e(f"{I}}} else {{")
        e(f"{I}  scf.yield %rmax, %rsum : f32, f32")
        e(f"{I}}}")
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        return outs

    def store_s(acc, pp, ind):
        e(f"{ind}%{pp}sst0 = vector.slice {acc}[0] : {V8} -> {V4}")
        e(f"{ind}%{pp}sst1 = vector.slice {acc}[4] : {V8} -> {V4}")
        e(f"{ind}vector.store %{pp}sst0, %s_view[%tid, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"{ind}vector.store %{pp}sst1, %s_view[%tid2, %c0] : {V4}, view<{2 * NT}x4xf32>")

    def body2(tail):
        """PIPE2 tile ks: region 1 (softmax(ks) + QK(ks+16) + P.V(ks)), barrier, region 2 (stores), barrier."""
        I = "    "
        e(f"{I}%vcur = index.add %c0, %c0 : index")
        e(f"{I}%ks16 = index.add %ks_, %c16 : index")
        e(f"{I}%ks32 = index.add %ks_, %c32 : index")
        # tile t = ks / 16 sits in K buffer t % 2: QK(t+1) reads buffer (t+1) % 2, K(t+2) is staged into t % 2
        e(f"{I}%kt16 = index.div %ks_, %c16 : index")
        e(f"{I}%kpar = index.rem %kt16, %c2 : index")
        e(f"{I}%kpe = index.cmp eq, %kpar, %c0 : index")
        e(f"{I}%kqo0 = index.constant {K_OFF} : index")
        e(f"{I}%kqo1 = index.constant {K2_OFF} : index")
        e(f"{I}%kqoi = scf.select %kpe, %kqo1, %kqo0 : index")
        e(f"{I}%ksoi = scf.select %kpe, %kqo0, %kqo1 : index")
        e(f"{I}%kqo = index.cast %kqoi : index to offset")
        e(f"{I}%kso = index.cast %ksoi : index to offset")
        e(f"{I}%kq_view = buffer.view %pool[%kqo] : buffer -> view<{KT}x{KT_PITCH}xf16>")
        e(f"{I}%ks_view = buffer.view %pool[%kso] : buffer -> view<{KT}x{KT_PITCH}xf16>")
        got = {}

        def qk_next():
            got["s"] = emit_qk("%kq_view", "nq", I)

        def mid():
            if not PIPE2_EARLY:
                qk_next()
            got["v"] = load_v("%ks16", "nv", I)
            got["k"] = load_k("%ks32", "nk", I)
        outs = emit_softmax_pv("%sown", tail, mid, qk_next if PIPE2_EARLY else None)
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        store_s(got["s"], "ns", I)
        stage_v(got["v"], "%c0", "stv", I)
        stage_k("%ks32", got["k"], "st", I, kv="%ks_view")
        e(f"{I}kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        return outs, got["s"]

    def body(tail):
        e("    %vcur = index.add %c0, %c0 : index")
        e("    %ks16 = index.add %ks_, %c16 : index")
        nk = load_k("%ks16", "nk", "    ")
        nv = load_v("%ks_", "nv", "    ")
        acc = emit_qk("%k_view", "", "    ")
        e(f"    %sst0 = vector.slice {acc}[0] : {V8} -> {V4}")
        e(f"    %sst1 = vector.slice {acc}[4] : {V8} -> {V4}")
        e(f"    vector.store %sst0, %s_view[%tid, %c0] : {V4}, view<{2 * NT}x4xf32>")
        e(f"    vector.store %sst1, %s_view[%tid2, %c0] : {V4}, view<{2 * NT}x4xf32>")
        # V(ks): the previous tile's P.V finished at the last barrier
        if VQ8 or VQ4:
            # keep the unpack after QK: hoisted between the QK MMAs, it waits for the V load ~4 MMAs after issue
            e("    scf.schedule.fence")
        stage_v(nv, "%c0", "stv", "    ")
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        # ---- B: K(ks+16) (QK of this tile is done) to LDS
        stage_k("%ks16", nk, "st", "    ")
        outs = emit_softmax_pv(acc, tail)
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        return outs











    e("  %r_live_i1 = index.cmp ult, %r_lq, %B : index")
    e(f"  %oinit = vector.fragment<init> %zeros8 shape [%m, %n] : {V8}")
    e("  %spq = index.add %start_pos, %qs : index")
    e("  %spq1 = index.add %spq, %c1 : index")
    e("  %split0 = index.div %spq1, %c16 : index")
    e("  %split1 = index.mul %split0, %c16 : index")
    # One masked loop over every tile (split = 0, so the unmasked loop runs no iteration).
    # Two loops (unmasked up to the diagonal, then masked) corrupt the O accumulators across the hand-off (Loom).
    e("  %split = index.min %c0, %max_vis : index")

    def loop(lo, hi, init_o, init_m, init_s, tail, res):
        init = ", ".join(f"{nm} = {v} : {V8}" for nm, v in zip(onames, init_o))
        init += f", %rmax = {init_m} : f32, %rsum = {init_s} : f32"
        if DOWHILE_MAIN:
            e(f"  %wkf, {', '.join(res)} = scf.while(%ksw = {lo} : index, {init}) -> (index, {types}) {{")
            e("    %ks_ = index.assume %ksw [range(%ksw, 0, 1048576), lt(%ksw, %max_vis)] : index")
            if SOFT1:
                outs = (body_soft4 if SOFT4 else body_soft2 if SOFT2 else body_soft1)(tail)
                nm_, ns_ = "%qmax", "%qsum"
            else:
                outs = body(tail)
                nm_, ns_ = "%nmax", "%nsum"
            e(f"    %ksn = index.add %ks_, %c{KT} : index")
            e(f"    %kcont = index.cmp ult, %ksn, {hi} : index")
            e(f"    scf.condition %kcont, %ksn, {', '.join(outs)}, {nm_}, {ns_} : i1, index, {types}")
            wargs = ", ".join(["%wk: index"] + [f"%wo{f}: {V8}" for f in range(8)] + ["%wm: f32", "%ws: f32"])
            e(f"  }} do({wargs}) {{")
            e(f"    scf.yield %wk, {', '.join(f'%wo{f}' for f in range(8))}, %wm, %ws : index, {types}")
            e("  }")
            return
        e(f"  {', '.join(res)} = scf.for %ks_ = [{lo} to {hi} step %c{KT}]({init}) -> ({types})  {{")
        if SOFT1:
            outs = (body_soft4 if SOFT4 else body_soft2 if SOFT2 else body_soft1)(tail)
            e(f"    scf.yield {', '.join(outs)}, %qmax, %qsum : {types}")
        else:
            outs = body(tail)
            e(f"    scf.yield {', '.join(outs)}, %nmax, %nsum : {types}")
        e("  }")

    if STEER:
        persist_issue("n")
    if R2P:
        # log2e x opaque 1 (exact): as a literal it keeps the exponent multiplies from pairing (VOPD)
        e(f"  %r2l2e = scalar.mulf {'%qsl2e' if R2P_QSPLIT else '%log2e'}, %one_o : f32")
        e(f"  %r2l2e8 = vector.splat %r2l2e : {V8}")
        if DIAG_MASK and DIAG_SWAP:   # the softmax mask test's wave-uniform terms
            e("  %r2ml0 = index.add %r_abs, %c1 : index")
            e("  %r2ml1 = index.min %r2ml0, %ctx_end : index")
            e("  %r2mh8 = index.mul %half, %c8 : index")
            e("  %r2mli = index.cast %r2ml1 : index to i32")
            e("  %r2mhi = index.cast %r2mh8 : index to i32")
            e("  %r2mll0 = scalar.subi %r2mli, %r2mhi : i32")
            e("  %r2mneg = scalar.constant -1073741824 : i32")   # a dead row never passes
            e("  %r2mll = scf.select %r_live, %r2mll0, %r2mneg : i32")
            e("  %r2mlr = kernel.subgroup.reduce<minsi> %r2mll : i32")
            e("  %r2mlb = kernel.subgroup.broadcast.first %r2mlr : i32")   # (scalar: the per-tile test is SALU)
        init = ", ".join(f"%rb{f} = %rinit{f} : {BK}" for f in range(16)) + ", %rmax = %ninf : f32, %rsum = %zero : f32"
        fb = [f"%fb{f}" for f in range(16)]
        k0 = "%c0"
        tok = re.compile(r"%[A-Za-z0-9_.]+")

        def renamed(i0, sub, suffix):
            """Rename every value defined in L[i0:] with suffix and apply sub (in place)."""
            sub = dict(sub)
            defs = set()
            for ln in L[i0:]:
                if " = " in ln:
                    defs.update(re.findall(r"%[A-Za-z0-9_.]+", ln.split(" = ")[0]))
            sub.update({d: d + suffix for d in defs if d not in sub})
            L[i0:] = [tok.sub(lambda m: sub.get(m.group(0), m.group(0)), ln) for ln in L[i0:]]
            return sub

        if R2P_SPLIT:   # (R2P_SPLIT=0 for the R2P_TPRIO / R2P_PRIO / R2P_PRIOQ / unpeeled diags)
            assert R2P_PEEL and not (R2P_TPRIO or R2P_PRIOQ or R2P_PRIO)
            qb = {f"%rb{f}": f"%rinit{f}" for f in range(16)}   # QK waves: Q, loop-invariant
            e(f"  {', '.join(fb)}, %gmax, %gsum = scf.if %isA -> ({rtypes}, f32, f32) {{")
            # QK role: tile 0, then tiles 16.. (2 barriers each)
            i0 = len(L)
            if KS:   # pieces start 32-key aligned: the peeled tile's parity is 0, as at tile 0 (keeps its branches folded)
                UNR_PAR[0] = 0
            q0 = body_r2p(first=True, role="A")
            if KS:
                UNR_PAR[0] = None
            sm = renamed(i0, {**qb, "%ks_": ks0, "%rmax": "%ninf", "%rsum": "%zero"}, "_qa0")
            q0 = [sm.get(x, x) for x in q0]
            e(f"  %pmore_qa = index.cmp ult, {ks0b}, %max_vis : index")
            e("  %gmax_qa, %gsum_qa = scf.if %pmore_qa -> (f32, f32) {")
            e(f"  %wkf_qa, %gmaxL_qa, %gsumL_qa = scf.while(%ksw_qa = {ks0b} : index, %rmax_qa = {q0[0]} : f32, %rsum_qa = {q0[1]} : f32) -> (index, f32, f32) {{")
            i0 = len(L)
            e("    %ks_ = index.assume %ksw_qa [range(%ksw_qa, 0, 1048576), lt(%ksw_qa, %max_vis)] : index")
            if "q" in R2P_UNR:   # tiles 1, 3, .. (odd: parity 1), then the even one after it when there is one
                UNR_PAR[0] = 1
                q1 = body_r2p(role="A")
                sa = renamed(i0, {**qb, "%rmax": "%rmax_qa", "%rsum": "%rsum_qa", "%ksw_qa": "%ksw_qa"}, "_qa")
                q1 = [sa.get(x, x) for x in q1]
                e(f"    %ks2_qa = index.add %ks__qa, %c{KT} : index")
                e("    %more2_qa = index.cmp ult, %ks2_qa, %max_vis : index")
                e("    %rmax2_qa, %rsum2_qa = scf.if %more2_qa -> (f32, f32) {")
                i1 = len(L)
                e("    %ks_ = index.assume %ks2_qa [range(%ks2_qa, 0, 1048576), lt(%ks2_qa, %max_vis)] : index")
                UNR_PAR[0] = 0
                q2 = body_r2p(role="A")
                UNR_PAR[0] = None
                sb = renamed(i1, {**qb, "%rmax": q1[0], "%rsum": q1[1], "%ks2_qa": "%ks2_qa"}, "_qb")
                q2 = [sb.get(x, x) for x in q2]
                e(f"    scf.yield {q2[0]}, {q2[1]} : f32, f32")
                e("    } else {")
                e(f"    scf.yield {q1[0]}, {q1[1]} : f32, f32")
                e("    }")
                e(f"    %ksn_qa = index.add %ks__qa, %c{2 * KT} : index")
                e("    %kcont_qa = index.cmp ult, %ksn_qa, %max_vis : index")
                e("    scf.condition %kcont_qa, %ksn_qa, %rmax2_qa, %rsum2_qa : i1, index, f32, f32")
            else:
                q1 = body_r2p(role="A")
                if "stamp" in R2P_DIAG:   # diag: every 64 tiles lane 0 takes a ticket (queue word 260) and records
                    # (unit, key, slot, 1) at ticket x 4 of the gate buffer: a global order of tile progress (co-reader drift)
                    e("    %stm = index.rem %ks_, %c1024st : index")
                    e("    %stz = index.cmp eq, %stm, %c0 : index")
                    e("    %stl = index.cmp eq, %tid, %c0 : index")
                    e("    %stx = scalar.andi %stz, %stl : i1")
                    e("    scf.if %stx {")
                    e("      %stw = index.constant 260 : index")
                    e("      %stt = view.atomic.rmw<addi> %pq_one, %pqv[%stw] {ordering = relaxed, scope = device} : i32, view<264xi32> -> i32")
                    e("      %stmk = scalar.constant 65535 : i32")
                    e("      %stt16 = scalar.andi %stt, %stmk : i32")
                    e("      %sti0 = index.cast %stt16 : i32 to index")
                    e("      %sti1 = index.assume %sti0 [range(%sti0, 0, 65536)] : index")
                    e("      %stc4 = index.constant 4 : index")
                    e("      %sti = index.mul %sti1, %stc4 : index")
                    e("      %sti_1 = index.add %sti, %c1 : index")
                    e("      %sti_2 = index.add %sti, %c2 : index")
                    e("      %sti_3 = index.add %sti, %c3st : index")
                    e("      %stu = index.cast %wgl : index to i32")
                    e("      %stks = index.cast %ks_ : index to i32")
                    e("      %stsl = index.cast %pslot : index to i32")
                    e("      view.store %stu, %st_v[%sti] : i32, view<262144xi32>")
                    e("      view.store %stks, %st_v[%sti_1] : i32, view<262144xi32>")
                    e("      view.store %stsl, %st_v[%sti_2] : i32, view<262144xi32>")
                    e("      view.store %pq_one, %st_v[%sti_3] : i32, view<262144xi32>")
                    e("    }")
                e(f"    %ksn = index.add %ks_, %c{KT} : index")
                e("    %kcont = index.cmp ult, %ksn, %max_vis : index")
                e(f"    scf.condition %kcont, %ksn, {q1[0]}, {q1[1]} : i1, index, f32, f32")
                renamed(i0, {**qb, "%rmax": "%rmax_qa", "%rsum": "%rsum_qa", "%ksw_qa": "%ksw_qa"}, "_qa")
            e("  } do(%wk_qa: index, %wm_qa: f32, %ws_qa: f32) {")
            e("    scf.yield %wk_qa, %wm_qa, %ws_qa : index, f32, f32")
            e("  }")
            e("  scf.yield %gmaxL_qa, %gsumL_qa : f32, f32")
            e("  } else {")
            e(f"  scf.yield {q0[0]}, {q0[1]} : f32, f32")
            e("  }")
            e(f"  scf.yield {', '.join(f'%rinit{f}' for f in range(16))}, %gmax_qa, %gsum_qa : {rtypes}, f32, f32")
            e("  } else {")
            # P.V role: tile 0 (staging only), then tiles 16..
            i0 = len(L)
            if KS:   # pieces start 32-key aligned: the peeled tile's parity is 0, as at tile 0 (keeps its branches folded)
                UNR_PAR[0] = 0
            p0 = body_r2p(first=True, role="B")
            if KS:
                UNR_PAR[0] = None
            sm = renamed(i0, {**{f"%rb{f}": f"%rinit{f}" for f in range(16)}, "%ks_": ks0}, "_pv0")
            p0 = [sm.get(x, x) for x in p0]
            e(f"  %pmore_pv = index.cmp ult, {ks0b}, %max_vis : index")
            pl = [f"%fbL{f}_pv" for f in range(16)]
            pr = [f"%fbR{f}_pv" for f in range(16)]
            e(f"  {', '.join(pr)} = scf.if %pmore_pv -> ({rtypes}) {{")
            pinit = ", ".join(f"%rb{f}_pvc = {p0[f]} : {BK}" for f in range(16))
            e(f"  %wkf_pv, {', '.join(pl)} = scf.while(%ksw_pv = {ks0b} : index, {pinit}) -> (index, {rtypes}) {{")
            i0 = len(L)
            e("    %ks_ = index.assume %ksw_pv [range(%ksw_pv, 0, 1048576), lt(%ksw_pv, %max_vis)] : index")
            if "p" in R2P_UNR:
                UNR_PAR[0] = 1
                p1 = body_r2p(role="B")
                sa = renamed(i0, {**{f"%rb{f}": f"%rb{f}_pvc" for f in range(16)}, "%ksw_pv": "%ksw_pv"}, "_pv")
                p1 = [sa.get(x, x) for x in p1]
                e(f"    %ks2_pv = index.add %ks__pv, %c{KT} : index")
                e("    %more2_pv = index.cmp ult, %ks2_pv, %max_vis : index")
                p2n = [f"%u2b{f}_pv" for f in range(16)]
                e(f"    {', '.join(p2n)} = scf.if %more2_pv -> ({rtypes}) {{")
                i1 = len(L)
                e("    %ks_ = index.assume %ks2_pv [range(%ks2_pv, 0, 1048576), lt(%ks2_pv, %max_vis)] : index")
                UNR_PAR[0] = 0
                p2 = body_r2p(role="B")
                UNR_PAR[0] = None
                sb = renamed(i1, {**{f"%rb{f}": p1[f] for f in range(16)}, "%ks2_pv": "%ks2_pv"}, "_pb")
                p2 = [sb.get(x, x) for x in p2]
                e(f"    scf.yield {', '.join(p2)} : {rtypes}")
                e("    } else {")
                e(f"    scf.yield {', '.join(p1)} : {rtypes}")
                e("    }")
                e(f"    %ksn_pv = index.add %ks__pv, %c{2 * KT} : index")
                e("    %kcont_pv = index.cmp ult, %ksn_pv, %max_vis : index")
                e(f"    scf.condition %kcont_pv, %ksn_pv, {', '.join(p2n)} : i1, index, {rtypes}")
            else:
                p1 = body_r2p(role="B")
                e(f"    %ksn = index.add %ks_, %c{KT} : index")
                e("    %kcont = index.cmp ult, %ksn, %max_vis : index")
                e(f"    scf.condition %kcont, %ksn, {', '.join(p1)} : i1, index, {rtypes}")
                renamed(i0, {**{f"%rb{f}": f"%rb{f}_pvc" for f in range(16)}, "%ksw_pv": "%ksw_pv"}, "_pv")
            wargs = ", ".join(["%wk_pv: index"] + [f"%wb{f}_pv: {BK}" for f in range(16)])
            e(f"  }} do({wargs}) {{")
            e(f"    scf.yield %wk_pv, {', '.join(f'%wb{f}_pv' for f in range(16))} : index, {rtypes}")
            e("  }")
            e(f"  scf.yield {', '.join(pl)} : {rtypes}")
            e("  } else {")
            e(f"  scf.yield {', '.join(p0)} : {rtypes}")
            e("  }")
            e(f"  scf.yield {', '.join(pr)}, %ninf, %zero : {rtypes}, f32, f32")
            e("  }")
        elif R2P_PEEL:   # tile 0 before the loop, without the P.V of tile -1 (P = 0: 16 MMAs of nothing per workgroup)
            i0 = len(L)
            body_r2p(first=True)
            sub = {"%ks_": "%c0", "%rmax": "%ninf", "%rsum": "%zero"}
            sub.update({f"%rb{f}": f"%rinit{f}" for f in range(16)})
            defs = set()
            for ln in L[i0:]:
                if " = " in ln:
                    defs.update(re.findall(r"%[A-Za-z0-9_.]+", ln.split(" = ")[0]))
            sub.update({d: d + "_p0" for d in defs})
            tok = re.compile(r"%[A-Za-z0-9_.]+")
            L[i0:] = [tok.sub(lambda m: sub.get(m.group(0), m.group(0)), ln) for ln in L[i0:]]
            init = ", ".join(f"%rb{f} = %nb{f}_p0 : {BK}" for f in range(16)) + ", %rmax = %qmax_p0 : f32, %rsum = %qsum_p0 : f32"
            k0 = "%c16"
            e("  %pmore = index.cmp ult, %c16, %max_vis : index")
            e(f"  %wkf, {', '.join(fb)}, %gmax, %gsum = scf.if %pmore -> (index, {rtypes}, f32, f32) {{")
            res = f"%wkfL, {', '.join(f + 'L' for f in fb)}, %gmaxL, %gsumL"
        else:
            res = f"%wkf, {', '.join(fb)}, %gmax, %gsum"
        if not R2P_SPLIT:
            if R2P_TPRIO:
                e("  kernel.priority {level = 0}")
            e(f"  {res} = scf.while(%ksw = {k0} : index, {init}) -> (index, {rtypes}, f32, f32) {{")
            e("    %ks_ = index.assume %ksw [range(%ksw, 0, 1048576), lt(%ksw, %max_vis)] : index")
            nb = body_r2p()
            e(f"    %ksn = index.add %ks_, %c{KT} : index")
            e("    %kcont = index.cmp ult, %ksn, %max_vis : index")
            e(f"    scf.condition %kcont, %ksn, {', '.join(nb)}, %qmax, %qsum : i1, index, {rtypes}, f32, f32")
            wargs = ", ".join([f"%wk: index"] + [f"%wb{f}: {BK}" for f in range(16)] + ["%wm: f32", "%ws: f32"])
            e(f"  }} do({wargs}) {{")
            e(f"    scf.yield %wk, {', '.join(f'%wb{f}' for f in range(16))}, %wm, %ws : index, {rtypes}, f32, f32")
            e("  }")
            if R2P_PEEL:
                e(f"  scf.yield {res} : index, {rtypes}, f32, f32")
                e("  } else {")
                e(f"  scf.yield %c16, {', '.join(f'%nb{f}_p0' for f in range(16))}, %qmax_p0, %qsum_p0 : index, {rtypes}, f32, f32")
                e("  }")
        if STEER:
            persist_finish("n")
        e("  scf.if %isA {")
        e(f"    view.store %gsum, %ls_view[%pa_own] : f32, view<{16 * NQB}xf32>")
        if KS:
            if R2P_QSPLIT and R2P_QFOLD:   # the max is in log2 units already
                e(f"    view.store %gmax, %lm_view[%pa_own] : f32, view<{16 * NQB}xf32>")
            else:
                e(f"    %gmcl = scalar.mulf %gmax, {'%r2l2e' if R2P_QSPLIT else '%log2e'} : f32")
                e(f"    view.store %gmcl, %lm_view[%pa_own] : f32, view<{16 * NQB}xf32>")
        e("  }")
        # the last tile: its V (P.V waves read it after the barrier), its P^T / alpha in slot (tiles - 1) % 2
        e("  %ltl0 = index.add %max_vis, %c15 : index")
        e("  %ltl1 = index.div %ltl0, %c16 : index")
        e("  %ltl = index.sub %ltl1, %c1 : index")
        e("  %lks = index.mul %ltl, %c16 : index")
        lv = load_v("%lks", "lv", "  ")
        if STEER:
            e("  %pwti = index.cast %tid : index to i32")
            e("  %pws = scalar.andi %pwti, %pwz0 : i32")
        stage_v(lv, "%c0", "lsv", "  ")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        e("  %lpar = index.rem %ltl, %c2 : index")
        e("  %lrows = index.mul %lpar, %c128p : index")
        e(f"  %lal = index.mul %lpar, %c{16 * NQB} : index")
        e("  %isB = index.cmp ne, %prole, %c0 : index")
        # every wave: no merge (the QK waves' results are never stored). The epilogue in the P.V waves only (scf.if
        # %isB around it) measured +3.5% at 8K with an identical loop: not understood
        if R2P_LASTB:
            pvo = [f"%fpo{f}" for f in range(16)]
            e(f"  {', '.join(pvo)} = scf.if %isB -> ({rtypes}) {{")
            outs = r2p_pv("%lrows", "%lal", fb, "fp", "    ")
            e(f"    scf.yield {', '.join(outs)} : {rtypes}")
            e("  } else {")
            e(f"    scf.yield {', '.join(fb)} : {rtypes}")
            e("  }")
            outs = pvo
        else:
            outs = r2p_pv("%lrows", "%lal", fb, "fp", "  ")
        i0 = len(L)
        for f in range(16):
            e(f"  %og{f} = vector.bitcast {outs[f]} : {BK} to {V8}")
        lsum_load()
        e(f"  %lsum_lo = view.load %ls_view[%pa_oth] : view<{16 * NQB}xf32> -> f32")
        if KS:
            ksplit_merge(i0)
            e("  %ewr0 = scalar.andi %r_live, %isB : i1")
            e("  %ewr = scalar.andi %ewr0, %kpwr : i1")
        else:
            e("  %ewr = scalar.andi %r_live, %isB : i1")
    elif PIPE2:
        # prologue part 2: QK(0) -> own S partial (carried), store it, stage V(0) and K(1) (second buffer)
        s0 = emit_qk("%k_view", "p0q", "  ")
        store_s(s0, "p0s", "  ")
        stage_v(v0, "%c0", "p0v", "  ")
        stage_k("%c16", k1, "p1s", "  ", kv="%k_view2")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        init = ", ".join(f"{nm} = %oinit : {V8}" for nm in onames)
        init += f", %rmax = %ninf : f32, %rsum = %zero : f32, %sown = {s0} : {V8}"
        r2 = [f"%og{i}" for i in range(8)] + ["%gmax", "%gsum", "%gsown"]
        e(f"  {', '.join(r2)} = scf.for %ks_ = [%c0 to %max_vis step %c{KT}]({init}) -> ({types}, {V8})  {{")
        outs, snext = body2(True)
        e(f"    scf.yield {', '.join(outs)}, %nmax, %nsum, {snext} : {types}, {V8}")
        e("  }")
    else:
        r1 = [f"%of{i}" for i in range(8)] + ["%fmax", "%fsum"]
        if DOWHILE_MAIN:   # the unmasked loop runs no iteration: its results are its inits
            r1 = ["%oinit"] * 8 + ["%ninf", "%zero"]
        else:
            loop("%c0", "%split", ["%oinit"] * 8, "%ninf", "%zero", False, r1)
        r2 = [f"%og{'l' if SOFT1 and (SOFT2 or SOFT4) else ''}{i}" for i in range(8)] + ["%gmax", "%gsum"]
        loop("%split", "%max_vis", r1[:8], r1[8], r1[9], True, r2)
        if SOFT1 and SOFT4:   # the last tile: stage its V, then its P.V from slot (tiles - 1) % 2
            e("  %f4a = index.add %max_vis, %c15 : index")
            e("  %f4b = index.div %f4a, %c16 : index")
            e("  %f4c = index.sub %f4b, %c1 : index")
            e("  %f4ks = index.mul %f4c, %c16 : index")
            fv = load_v("%f4ks", "fv", "  ")
            stage_v(fv, "%c0", "fsv", "  ")
            e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            e("  %f4p = index.rem %f4c, %c2 : index")
            e("  %f4rows = index.mul %f4p, %c128 : index")
            e("  %f4al = index.mul %f4p, %c64 : index")
            fin = soft1_pv("  ", [f"%ogl{i}" for i in range(8)], "fin", "%f4rows", "%f4al")
            for i in range(8):
                e(f"  %og{i} = vector.bitcast {fin[i]} : {V8} to {V8}")
        if SOFT1 and SOFT2 and not SOFT4:   # the last tile's P.V (its P^T / alpha / V are in LDS after the loop's last barrier)
            fslot = (None, None)
            if SOFT3:
                e("  %fs0 = index.add %max_vis, %c15 : index")
                e("  %fs1 = index.div %fs0, %c16 : index")
                e("  %fs2 = index.sub %fs1, %c1 : index")
                e("  %fs3 = index.rem %fs2, %c2 : index")
                e("  %fsrows = index.mul %fs3, %c128 : index")
                e("  %fsal = index.mul %fs3, %c64 : index")
                fslot = ("%fsrows", "%fsal")
            fin = soft1_pv("  ", [f"%ogl{i}" for i in range(8)], "fin", *fslot)
            for i in range(8):
                e(f"  %og{i} = vector.bitcast {fin[i]} : {V8} to {V8}")
        if SOFT1:
            e("  scf.if %isA {")
            e("    view.store %gsum, %ls_view[%pa_own] : f32, view<64xf32>")
            e("  }")
            e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            lsum_load()

    if PERSIST and PERSIST_DYN and not STEER:   # the next unit's atomic before the epilogue's stores
        persist_issue("n")
    if R2P and R2P_TPRIO:
        e(f"  kernel.priority {{level = {R2P_TPRIO}}}")
    # ---- epilogue: o / sum * sigmoid(gate); lane writes dims 8h..8h+7 of each 16
    e(f"  %lsum = scalar.addf {'%lsum_l' if SOFT1 or R2P else '%gsum'}, %zero : f32")   # already the full row sum
    for bs in ("", "_o") if R2P else ("",):   # R2P: fragments 8..15 are the other head's (row sum %lsum_lo)
        if bs:
            e("  %lsum_o = scalar.addf %lsum_lo, %zero : f32")
        e(f"  %lpos{bs} = scalar.cmpf ogt, %lsum{bs}, %zero : f32")
        e(f"  %linv0{bs} = scalar.divf %one, %lsum{bs} : f32")
        e(f"  %linv{bs} = scf.select %lpos{bs}, %linv0{bs}, %zero : f32")
        e(f"  %linv8{bs} = vector.splat %linv{bs} : {V8}")
        e(f"  %lsum8{bs} = vector.splat %lsum{bs} : {V8}")
        if EPI_FAST:
            e(f"  %elr8{bs} = vector.splat %linv0{bs} : {V8}")
            e(f"  %enls{bs} = scalar.negf %lsum{bs} : f32")
            e(f"  %enls8{bs} = vector.splat %enls{bs} : {V8}")
    e(f"  %nlog2e8 = vector.splat %log2e : {V8}")
    if EPI_FAST:
        e("  %efmax = scalar.constant 3.4028234663852886e+38 : f32")
        e(f"  %efmax8 = vector.splat %efmax : {V8}")
        e("  %enone = scalar.constant -1.0 : f32")
        e(f"  %enone8 = vector.splat %enone : {V8}")
    e("  %elc = index.min %r_lq, %B_1 : index")
    if "gatehot" in R2P_DIAG:   # diag: gate rows read from the first 64-row block (cache hot)
        e("  %ghot64 = index.constant 64 : index")
        e("  %elch = index.rem %elc, %ghot64 : index")
        e("  %eoff0 = index.mul %elch, %c6144 : index")
    elif "gspread" in R2P_DIAG:   # diag: gatehot's 64 rows' lines, spread over the batch (row (r % 64) * 64)
        e("  %gsp64 = index.constant 64 : index")
        e("  %elcs0 = index.rem %elc, %gsp64 : index")
        e("  %elcs1 = index.mul %elcs0, %gsp64 : index")
        e("  %elcs = index.min %elcs1, %B_1 : index")
        e("  %eoff0 = index.mul %elcs, %c6144 : index")
    else:
        e("  %eoff0 = index.mul %elc, %c6144 : index")
    e("  %ehb = index.mul %r_head, %c256 : index")
    e("  %eoff1 = index.add %eoff0, %ehb : index")
    e("  %eh8 = index.mul %half, %c8 : index")
    e("  %eoff2 = index.add %eoff1, %hd128 : index")
    e("  %eoff = index.add %eoff2, %eh8 : index")
    if R2P:   # fragments 8..15: the other head, the same dims
        e("  %ehbo = index.mul %r_head_o, %c256 : index")
        e("  %eoffo1 = index.add %eoff0, %ehbo : index")
        e("  %eoffo2 = index.add %eoffo1, %hd128 : index")
        e("  %eoffo = index.add %eoffo2, %eh8 : index")
    def gate_addr(f):
        e(f"  %eo{f}c = index.constant {16 * (f % 8 if R2P else f)} : index")
        e(f"  %eo{f} = index.add {'%eoffo' if R2P and f >= 8 else '%eoff'}, %eo{f}c : index")
        e(f"  %eo{f}b = index.add %eo{f}, %c4 : index")

    def gate_load(f):
        if QG16 or GSIG16:
            e(f"  %eg{f}h = vector.load %g_flat[%eo{f}] : view<[%qtot]xf16> -> {V8H}")
            return
        e(f"  %eg{f}a = vector.load %g_flat[%eo{f}] : view<[%qtot]xf32> -> {V4}")
        e(f"  %eg{f}b = vector.load %g_flat[%eo{f}b] : view<[%qtot]xf32> -> {V4}")

    def gsig_decode(f):
        """GSIG: %eg{f} = the loaded gate fragment's sigmoid as f32"""
        if GSIG == "u16":   # u | 0x4B00 << 16 = 2^23 + u; fma(2^23 + u, c, -2^23 c) = round(u c), c = f32(1 / 65535)
            e(f"  %egk{f} = vector.constant 14.0 : {V8H}")
            e(f"  %egc{f} = vector.constant 1.5259021893143654e-05 : {V8}")
            e(f"  %egd{f} = vector.constant -128.001953125 : {V8}")
            e(f"  %egi{f} = vector.interleave<0> %eg{f}h, %egk{f} : {V8H}, {V8H} -> vector<16xf16>")
            e(f"  %egx{f} = vector.bitcast %egi{f} : vector<16xf16> to {V8}")
            e(f"  %eg{f} = vector.fmaf %egx{f}, %egc{f}, %egd{f} : {V8}")
        elif GSIG16:
            e(f"  %eg{f} = vector.extf %eg{f}h : {V8H} to {V8}")
        else:
            e(f"  %eg{f} = vector.concat<0> %eg{f}a, %eg{f}b : {V4}, {V4} -> {V8}")

    def gate_loads(f):
        gate_addr(f)
        gate_load(f)
    gate_each = GATE_EACH or R2P   # R2P: 16 output fragments are live, so each gate fragment right before its use

    def gate_batch(fs):
        # all gate loads together (latency); GATE_EACH: each fragment's right before its use (fewer live registers)
        if gate_each:
            return
        if GATE_FENCE and not GATE_EACH:
            for f in fs:
                gate_addr(f)
            e("  scf.schedule.fence")
            for f in fs:
                gate_load(f)
        else:
            for f in (() if GATE_EACH else fs):
                gate_loads(f)
    gate_batch(range(8))
    if TILED_OUT:
        e("  %et16 = index.constant 16 : index")
        e("  %et384 = index.constant 384 : index")
        e("  %etk = index.div %elc, %et16 : index")
        e("  %etr = index.rem %elc, %et16 : index")
        e("  %etkb = index.mul %etk, %et384 : index")
        e("  %etrb = index.mul %etr, %et16 : index")
        e("  %etlast = index.sub %qtot, %c8 : index")
        nfh = 8 if R2P else NFR   # fragments per head (R2P: 8..15 are the other head's)
        e(f"  %et1792 = index.constant {256 * (nfh - 1)} : index")
        e("  %etmax = index.sub %etlast, %et1792 : index")
        for bs, hb in (("", "%ehb"), ("_o", "%ehbo")) if R2P else (("", "%ehb"),):
            e(f"  %etc0a{bs} = index.add {hb}, %hd128 : index")   # the lane's column: head * 256 + d
            e(f"  %etc0{bs} = index.add %etc0a{bs}, %eh8 : index")
            # fragment f's column is etc0 + 16 f: tile column etc0 / 16 + f, offset etc0 % 16, so address base + 256 f
            e(f"  %ett0{bs} = index.div %etc0{bs}, %et16 : index")
            e(f"  %etm0{bs} = index.rem %etc0{bs}, %et16 : index")
            e(f"  %eti0{bs} = index.add %etkb, %ett0{bs} : index")
            e(f"  %etb0{bs} = index.mul %eti0{bs}, %c256 : index")
            e(f"  %ets0{bs} = index.add %etb0{bs}, %etrb : index")
            e(f"  %etu0{bs} = index.add %ets0{bs}, %etm0{bs} : index")
            e(f"  %etbase{bs} = index.min %etu0{bs}, %etmax : index")   # always in range; states it for the bound proof

    def tiled_addr(f):
        """fragment-major output address of fragment f: base + 256 f"""
        o = "_o" if R2P and f >= 8 else ""
        e(f"    %eto{f}c = index.constant {256 * (f % 8 if R2P else f)} : index")
        e(f"    %eto{f} = index.add %etbase{o}, %eto{f}c : index")
    epig = R2P and R2P_EPIG   # the whole fragment (gate loads, math, store) under the store guard: QK waves skip it
    gpf = R2P_GATEPF if epig else 0
    if gpf:
        e("  scf.if %ewr {")
        for f in range(min(gpf, NFR)):
            gate_loads(f)
        e("  scf.schedule.fence")
    for f in range(NFR):
        if f == 8:
            gate_batch(range(8, 16))
        if epig and not gpf:
            e("  scf.if %ewr {")
        if gpf:
            if f + gpf < NFR:
                gate_loads(f + gpf)
                e("  scf.schedule.fence")
        elif gate_each and ("noepi" in R2P_DIAG or "nogate" in R2P_DIAG and R2P and EPI_FAST):
            gate_addr(f)
        elif gate_each:
            gate_loads(f)
        if "noepi" in R2P_DIAG:   # diag: no gate loads or epilogue math, raw O stored (wrong output)
            e(f"  %eout{f} = vector.addf %og{f}, %zeros8 : {V8}")
        elif GSIG and R2P:   # o / sum (Markstein, as EPI_FAST) times the ready sigmoid
            assert EPI_FAST and PRO_FAST
            bs = "_o" if f >= 8 else ""
            gsig_decode(f)
            e(f"  %eq{f} = vector.mulf %og{f}, %elr8{bs} : {V8}")
            e(f"  %eqe{f} = vector.fmaf %eq{f}, %enls8{bs}, %og{f} : {V8}")
            e(f"  %eod{f} = vector.fmaf %eqe{f}, %elr8{bs}, %eq{f} : {V8}")
            e(f"  %eout{f} = vector.mulf %eod{f}, %eg{f} : {V8}")
        elif GSIG:   # the head_dim split kernel: o / sum (as below) times the ready sigmoid
            gsig_decode(f)
            if EPI_FAST:
                e(f"  %eq{f} = vector.mulf %og{f}, %elr8 : {V8}")
                e(f"  %eqe{f} = vector.fmaf %eq{f}, %enls8, %og{f} : {V8}")
                e(f"  %eod{f} = vector.fmaf %eqe{f}, %elr8, %eq{f} : {V8}")
            else:
                e(f"  %eod{f} = vector.divf %og{f}, %lsum8 : {V8}")
            e(f"  %eov{f} = scf.select %lpos, %eod{f}, %zeros8 : {V8}")
            e(f"  %eout{f} = vector.mulf %eov{f}, %eg{f} : {V8}")
        elif "nogate" in R2P_DIAG and R2P and EPI_FAST:   # diag: o / sum without the gate (loads, sigmoid moved out)
            bs = "_o" if f >= 8 else ""
            e(f"  %eq{f} = vector.mulf %og{f}, %elr8{bs} : {V8}")
            e(f"  %eqe{f} = vector.fmaf %eq{f}, %enls8{bs}, %og{f} : {V8}")
            e(f"  %eout{f} = vector.fmaf %eqe{f}, %elr8{bs}, %eq{f} : {V8}")
        else:
            if QG16:
                e(f"  %eg{f} = vector.extf %eg{f}h : {V8H} to {V8}")
            else:
                e(f"  %eg{f} = vector.concat<0> %eg{f}a, %eg{f}b : {V4}, {V4} -> {V8}")
            e(f"  %egn{f} = vector.negf %eg{f} : {V8}")
            e(f"  %egm{f} = vector.mulf %egn{f}, %nlog2e8 : {V8}")
            e(f"  %egx{f} = vector.exp2f<afn> %egm{f} : {V8}")
            e(f"  %egd{f} = vector.addf %ones8, %egx{f} : {V8}")
            bs = "_o" if R2P and f >= 8 else ""
            if R2P_EPI_LITE:
                assert R2P and EPI_FAST and PRO_FAST
                e(f"  %egr{f} = vector.divf<nnan|nsz|arcp> %ones8, %egd{f} : {V8}")
                e(f"  %eod{f} = vector.mulf %og{f}, %elr8{bs} : {V8}")
            elif EPI_FAST:
                # 1 / dn by rcp + one Newton step: bit-exact for dn in [1, 2^126] (gen_gemm_tile FAST_RCP); dn = inf clamps to
                # FLT_MAX, whose tiny reciprocal times any |o| <= 65504 rounds to the same signed f16 zero as o * 0
                e(f"  %egc{f} = vector.minnumf<nnan> %egd{f}, %efmax8 : {V8}")
                e(f"  %egr0{f} = vector.divf<nnan|ninf|nsz|arcp> %ones8, %egc{f} : {V8}")
                e(f"  %egnd{f} = vector.mulf %egc{f}, %enone8 : {V8}")
                e(f"  %eger{f} = vector.fmaf %egnd{f}, %egr0{f}, %ones8 : {V8}")
                e(f"  %egr{f} = vector.fmaf %eger{f}, %egr0{f}, %egr0{f} : {V8}")
                # o / sum (Markstein): q = o r with r = RN(1 / sum), then q + (o - q sum) r, correctly rounded
                e(f"  %eq{f} = vector.mulf %og{f}, %elr8{bs} : {V8}")
                e(f"  %eqe{f} = vector.fmaf %eq{f}, %enls8{bs}, %og{f} : {V8}")
                e(f"  %eod{f} = vector.fmaf %eqe{f}, %elr8{bs}, %eq{f} : {V8}")
            else:
                e(f"  %egr{f} = vector.divf %ones8, %egd{f} : {V8}")
                e(f"  %eod{f} = vector.divf %og{f}, %lsum8{bs} : {V8}")
            if PRO_FAST and R2P and EPI_FAST:   # stored rows have sum >= 1
                e(f"  %eout{f} = vector.mulf %eod{f}, %egr{f} : {V8}")
            else:
                e(f"  %eov{f} = scf.select %lpos{bs}, %eod{f}, %zeros8 : {V8}")
                e(f"  %eout{f} = vector.mulf %eov{f}, %egr{f} : {V8}")
        if not epig:
            e(f"  scf.if {'%ewr' if R2P else '%r_live'} {{")
        if TILED_OUT:
            tiled_addr(f)
        e(f"    %eoh{f} = vector.fptrunc %eout{f} : {V8} to {V8H}")
        e(f"    vector.store %eoh{f}, %o_flat[%{'eto' if TILED_OUT else 'eo'}{f}] : {V8H}, view<[%qtot]xf16>")
        if not gpf:
            e("  }")
    if gpf:
        e("  }")
    if OCNT:
        assert not TILED_OUT and 128 % QT == 0 and 1024 % (256 * HPW) == 0
        e(f"  %oc_qpt = index.constant {128 // QT} : index")
        e(f"  %oc_ppc = index.constant {1024 // (256 * HPW)} : index")
        e("  %oc_t = index.div %qb, %oc_qpt : index")
        e("  %oc_c = index.div %hp, %oc_ppc : index")
        e("  %oc_tt = index.div %B, %c128 : index")
        e("  %oc_i0 = index.mul %oc_c, %oc_tt : index")
        e("  %oc_i = index.add %oc_i0, %oc_t : index")
        e("  %oc_nc = index.div %npairs, %oc_ppc : index")
        e("  %oc_n = index.mul %oc_nc, %oc_tt : index")
        e("  %oc_one = scalar.constant 1 : i32")
        e("  %oc_v = buffer.view %oc_na[%base] : buffer -> view<[%oc_n]xi32>")
        # every wave's O stores before lane 0's add; release only: an acquiring barrier marks global memory as
        # interfered, so every input's scalar loads would become vector loads (the whole kernel's code shifts)
        e("  kernel.barrier<global> scope(workgroup) ordering(release)")
        e("  %oc_l0 = index.cmp eq, %tid, %c0 : index")
        e("  scf.if %oc_l0 {")
        e("    view.atomic.reduce<addi> %oc_one, %oc_v[%oc_i] {ordering = release, scope = device} : i32, view<[%oc_n]xi32>")
        e("  }")
    if PERSIST and PERSIST_DYN:
        if not STEER:
            persist_finish("n")
        w, wt = (", %pws", ", i32") if STEER else ("", "")
        e(f"  scf.yield %pqnu{w} : index{wt}")
        e("  } else {")
        e(f"  scf.yield {punq}{', %pwc' if STEER else ''} : index{wt}")
        e("  }")
        e(f"  %pcont = index.cmp ult, %pnext, {punq} : index")
        e(f"  scf.condition %pcont, %pnext{', %pwn' if STEER else ''} : i1, index{wt}")
        e(f"  }} do(%pw: index{', %pwd: i32' if STEER else ''}) {{")
        e(f"  scf.yield %pw{', %pwd' if STEER else ''} : index{wt}")
        e("  }")
    elif PERSIST:
        e("  }")   # scf.if %pval
        e("  }")   # scf.for %pr
    if PERSIST:
        # constants defined before the unit loop stay live across all of it (Loom does not rematerialize them there):
        # float / vector constants that nothing before the loop uses move to the top of the unit body
        top = next(i for i, ln in enumerate(L) if ln.startswith("} launch("))
        body = next(i for i, ln in enumerate(L) if "kernel.barrier" in ln and i > L.index(PVAL_LINE))
        tok = re.compile(r"%[A-Za-z0-9_.]+")
        moved = []
        for i in range(top + 1, L.index(PVAL_LINE)):
            m = re.match(r"\s*(%[A-Za-z0-9_.]+) = (scalar\.constant .*: f(16|32)|vector\.constant |vector\.splat )", L[i])
            if m and not any(m.group(1) in tok.findall(x.split(" = ", 1)[-1]) for x in L[i + 1:L.index(PVAL_LINE)]):
                moved.append(i)
        mv = [L[i] for i in moved]
        for i in reversed(moved):
            del L[i]
        body = next(i for i, ln in enumerate(L) if "kernel.barrier" in ln and i > L.index(PVAL_LINE))
        L[body + 1:body + 1] = mv
    if R2P_QPF:
        assert QNORM and STEER and R2P_K2
        L[:] = qpf_pass(L, PVAL_LINE)
    if R2P_QTOUCH:
        assert QNORM and STEER and R2P_K2 and not R2P_QPF
        L[:] = qtouch_pass(L, PVAL_LINE)
    e("  kernel.return")
    e("}")
    if "noallbar" in R2P_DIAG:   # diag: no workgroup barriers anywhere (races: wrong values)
        L = [ln for ln in L if "kernel.barrier<workgroup>" not in ln]
    return "\n".join(L) + "\n"


def gen_qrot():
    """yah_qrot (QROT): the kv4 attention's Q prologue as its own kernel: per (token, head) row of 256 dims, 4 threads of
    64 dims compute f16(H256(q * 1/16) * 1/16) with the staged path's operations (had64, the +0.0, the rounding), so
    the attention loads the same bits as direct Q fragments. f32 query [token][6144] -> f16 [token][6144]."""
    L = []
    e = L.append
    e("// GENERATED by tools/gen_attn_fa.py (gen_qrot) -- edit the generator.")
    e("amdgpu.target<gfx1151> @qrot_w32 {subgroup_size = 32}")
    e("config.decl @yah_qrot.token_count : %value: index where [range(%value, 1, 1048576)]")
    e("")
    e("kernel.def target(@qrot_w32) @yah_qrot() {")
    e("  %ntok = config.get @yah_qrot.token_count : index")
    e("  %c1 = index.constant 1 : index")
    e("  %c96 = index.constant 96 : index")
    e("  %c255 = index.constant 255 : index")
    e("  %c256 = index.constant 256 : index")
    e("  %th0 = index.mul %ntok, %c96 : index")       # 24 rows x 4 threads per token
    e("  %th1 = index.add %th0, %c255 : index")
    e("  %nwg = index.div %th1, %c256 : index")
    e("  kernel.launch.config workgroups(%nwg, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index")
    e("} launch(%src: buffer, %dst: buffer) {")
    e("  %base = index.constant 0 : offset")
    for v in (0, 1, 2, 4, 24, 64, 256, 6144):
        e(f"  %c{v} = index.constant {v} : index")
    e("  %x32 = scalar.constant 32 : i32")
    e("  %ntok = config.get @yah_qrot.token_count : index")
    e("  %qtot = index.mul %ntok, %c6144 : index")
    e("  %rows = index.mul %ntok, %c24 : index")
    e("  %rows1 = index.sub %rows, %c1 : index")
    e("  %s_na, %d_na = buffer.assume.noalias %src, %dst : buffer, buffer")
    e("  %q_flat = buffer.view %s_na[%base] : buffer -> view<[%qtot]xf32>")
    e("  %o_flat = buffer.view %d_na[%base] : buffer -> view<[%qtot]xf16>")
    e("  %wg = kernel.workgroup.id<x> : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %g0 = index.mul %wg, %c256 : index")
    e("  %g = index.add %g0, %tid : index")
    e("  %row = index.div %g, %c4 : index")
    e("  %qpart = index.rem %g, %c4 : index")
    e("  %live = index.cmp ult, %row, %rows : index")
    e("  %rowc = index.min %row, %rows1 : index")
    e("  %qrow = index.mul %rowc, %c256 : index")    # [token][head][256]: row-major rows of 256
    e("  %qscale = scalar.constant 0.0625 : f32")
    e(f"  %qsc_v = vector.splat %qscale : {V4}")
    e(f"  %z4h = vector.constant 0.0 : {V4}")
    e("  %qdb0a = index.mul %qpart, %c64 : index")
    raw = []
    for c in range(8):
        t = f"0_{c}"
        e(f"  %qdc{t} = index.constant {8 * c} : index")
        e(f"  %qd{t} = index.add %qdb0a, %qdc{t} : index")
        e(f"  %qa{t} = index.add %qrow, %qd{t} : index")
        e(f"  %qa{t}b = index.add %qa{t}, %c4 : index")
        e(f"  %qv{t}a = vector.load %q_flat[%qa{t}] : view<[%qtot]xf32> -> {V4}")
        e(f"  %qv{t}b = vector.load %q_flat[%qa{t}b] : view<[%qtot]xf32> -> {V4}")
        e(f"  %qm{t}a0 = vector.mulf %qv{t}a, %qsc_v : {V4}")
        e(f"  %qm{t}b0 = vector.mulf %qv{t}b, %qsc_v : {V4}")
        raw += [f"%qm{t}a0", f"%qm{t}b0"]
    rot = had64(e, "fh", raw, (1, 2), 0.0625)
    e("  scf.if %live {")
    for c in range(8):
        t = f"0_{c}"
        e(f"    %qm{t}a = vector.addf {rot[2 * c]}, %z4h : {V4}")
        e(f"    %qm{t}b = vector.addf {rot[2 * c + 1]}, %z4h : {V4}")
        e(f"    %qh{t}a = vector.fptrunc %qm{t}a : {V4} to vector<4xf16>")
        e(f"    %qh{t}b = vector.fptrunc %qm{t}b : {V4} to vector<4xf16>")
        e(f"    %qh{t} = vector.concat<0> %qh{t}a, %qh{t}b : vector<4xf16>, vector<4xf16> -> {V8H}")
        e(f"    vector.store %qh{t}, %o_flat[%qa{t}] : {V8H}, view<[%qtot]xf16>")
    e("  }")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def _deq_header(e, name, sym, extra_params, out_cap=False):
    e(f"// GENERATED by tools/gen_attn_fa.py ({name}) -- edit the generator.")
    e("amdgpu.target<gfx1151> @deq_w32 {subgroup_size = 32}")
    e(f"config.decl @{sym}.cache_capacity : %value: index where [range(%value, 1, 1048576)]")
    if out_cap:
        e(f"config.decl @{sym}.out_capacity : %value: index where [range(%value, 1, 1048576)]")
    e(f"config.decl @{sym}.rows : %value: index where [range(%value, 1, 1048576)]")
    e("")
    e(f"kernel.def target(@deq_w32) @{sym}() {{")
    e(f"  %rows = config.get @{sym}.rows : index")
    for v in (1, 15, 16, 64, 255, 256):
        e(f"  %c{v} = index.constant {v} : index")
    e(extra_params)
    e("  %th1 = index.add %th, %c255 : index")
    e("  %nwg = index.div %th1, %c256 : index")
    e("  kernel.launch.config workgroups(%nwg, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index")


def _deq_common(e, sym):
    e("  %base = index.constant 0 : offset")
    for v in (0, 1, 2, 8, 15, 16, 32, 64, 128, 255, 256, 1024, 4096):
        e(f"  %c{v} = index.constant {v} : index")
    e(f"  %cap = config.get @{sym}.cache_capacity : index")
    e(f"  %rows0 = config.get @{sym}.rows : index")
    e("  %rows = index.min %rows0, %cap : index")
    e("  %npg0 = index.add %cap, %c255 : index")
    e("  %npages = index.div %npg0, %c256 : index")
    e("  %pcap = index.mul %npages, %c256 : index")
    e("  %wg = kernel.workgroup.id<x> : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %g0 = index.mul %wg, %c256 : index")
    e("  %g = index.add %g0, %tid : index")


def gen_kdeq(bits=4, raw=False, khalf=False):
    """yah_kdeq4 (kv4 prefill): the paged int4 K pool -> the paged f16 K pool [physical row][1024] (rows 0..rows-1), with the
    attention's own staging decode (stage_k_ K4: nibble pairs to f16 1 + u/16, one packed fma with the row group's
    (16 s, lo - 16 s)), so the fp16 attention reading it computes what the kv4 attention did. Thread: (row, KV head,
    16-dim chunk). Bindings: kq (i32), kscale (i32), ptab, out (f16).
    raw (kv8, the attention's ATTN_KRAW): out holds u - 128 exactly and binding kso (f16 [KV head][128-dim half][pool
    row], f16) the scale s = (256 s) / 256 of each row half."""
    assert bits in (4, 8) and not (raw and bits != 8) and not (khalf and not raw)
    sym = f"yah_kdeq{bits}"
    nw = 2 if bits == 4 else 4                  # dwords per 16-dim chunk
    rw, sw = (128, 32) if bits == 4 else (256, 8)   # code / scale dwords per row
    L = []
    e = L.append
    _deq_header(e, f"gen_kdeq{bits}", sym, "  %th = index.mul %rows, %c64 : index")
    e("} launch(%kq: buffer, %ks: buffer, %ptab: buffer, %out: buffer" + (", %kso: buffer" if raw else "") + ") {")
    _deq_common(e, sym)
    if bits == 8:
        e("  %c4 = index.constant 4 : index")
    e(f"  %kq32tot = index.mul %pcap, %c{rw} : index")
    e(f"  %kstot = index.mul %pcap, %c{sw} : index")
    e("  %otot = index.mul %pcap, %c1024 : index")
    e("  %kq_na, %ks_na, %pt_na, %o_na = buffer.assume.noalias %kq, %ks, %ptab, %out : buffer, buffer, buffer, buffer")
    e("  %kq_flat = buffer.view %kq_na[%base] : buffer -> view<[%kq32tot]xi32>")
    e("  %ks_flat = buffer.view %ks_na[%base] : buffer -> view<[%kstot]xi32>")
    e("  %pt_flat = buffer.view %pt_na[%base] : buffer -> view<[%npages]xi32>")
    e("  %o_flat = buffer.view %o_na[%base] : buffer -> view<[%otot]xf16>")
    e("  %r0 = index.div %g, %c64 : index")
    e("  %j = index.rem %g, %c64 : index")
    e("  %live = index.cmp ult, %r0, %rows : index")
    e("  %cap1 = index.sub %cap, %c1 : index")
    e("  %r = index.min %r0, %cap1 : index")
    e("  %h = index.div %j, %c16 : index")
    e("  %ch = index.rem %j, %c16 : index")
    e("  %lp = index.div %r, %c256 : index")
    e("  %pg0 = view.load %pt_flat[%lp] : view<[%npages]xi32> -> i32")
    e("  %pgr = index.cast %pg0 : i32 to index")
    e("  %pg = index.assume %pgr [range(%pgr, 0, 65535), lt(%pgr, %npages)] : index")
    e("  %pb = index.mul %pg, %c256 : index")
    e("  %po = index.rem %r, %c256 : index")
    e("  %pr = index.add %pb, %po : index")
    e(f"  %kr = index.mul %pr, %c{rw} : index")
    e(f"  %kh = index.mul %h, %c{16 * nw} : index")
    e(f"  %kc = index.mul %ch, %c{nw} : index")
    e("  %ka0 = index.add %kr, %kh : index")
    e("  %ka = index.add %ka0, %kc : index")
    e(f"  %kv = vector.load %kq_flat[%ka] : view<[%kq32tot]xi32> -> vector<{nw}xi32>")
    e(f"  %sr = index.mul %pr, %c{sw} : index")
    e(f"  %sh8 = index.mul %h, %c{sw // 4} : index")
    e(f"  %sc2 = index.div %ch, %c{2 if bits == 4 else 8} : index")
    e("  %sa0 = index.add %sr, %sh8 : index")
    e("  %sa = index.add %sa0, %sc2 : index")
    e("  %sv = view.load %ks_flat[%sa] : view<[%kstot]xi32> -> i32")
    e(f"  %kdm = scalar.constant {62915520 if bits == 4 else 66847740} : i32")       # 0x03c003c0 / 0x03fc03fc
    e("  %kdgm = scalar.constant 1006648320 : i32")    # 0x3c003c00
    e("  %s1 = vector.from_elements %sv : vector<1xi32>")
    e("  %sh = vector.bitcast %s1 : vector<1xi32> to vector<2xf16>")
    e("  %sS = vector.extract %sh[0] : vector<2xf16> -> f16")
    e("  %sC = vector.extract %sh[1] : vector<2xf16> -> f16")
    ws = []
    for d in range(nw):
        e(f"  %w{d} = vector.extract %kv[{d}] : vector<{nw}xi32> -> i32")
        for k, sh in enumerate((6, 2, -2, -6) if bits == 4 else (2, -6)):
            dec_pair(e, "  ", f"%g{d}{k}", f"%w{d}", sh, 0x03c003c0 if bits == 4 else 0x03fc03fc, "%kdgm")
            ws.append(f"%g{d}{k}")
    e(f"  %pk = vector.from_elements {', '.join(ws)} : vector<8xi32>")
    e(f"  %pf = vector.bitcast %pk : vector<8xi32> to {V16H}")
    if raw:   # (1 + u / 256) 256 - 384 = u - 128, exact
        e("  %rS = scalar.constant 256.0 : f16")
        e("  %rC = scalar.constant -384.0 : f16")
        e(f"  %S16 = vector.splat %rS : {V16H}")
        e(f"  %C16 = vector.splat %rC : {V16H}")
    else:
        e(f"  %S16 = vector.splat %sS : {V16H}")
        e(f"  %C16 = vector.splat %sC : {V16H}")
    e(f"  %dk0 = vector.fmaf %pf, %S16, %C16 : {V16H}")
    dk = "%dk0"
    if khalf:   # the other half's (256 s', ..) pair; r = s / max(s, s') (exactly 1 on the larger half)
        e("  %c1h = index.constant 1 : index")
        e("  %soh = index.sub %c1h, %sc2 : index")    # the other half (kv8: dword sr + 2 h + half)
        e("  %sao = index.add %sr, %sh8 : index")
        e("  %sao3 = index.add %sao, %soh : index")
        e("  %svo = view.load %ks_flat[%sao3] : view<[%kstot]xi32> -> i32")
        e("  %so1 = vector.from_elements %svo : vector<1xi32>")
        e("  %soo = vector.bitcast %so1 : vector<1xi32> to vector<2xf16>")
        e("  %sSo = vector.extract %soo[0] : vector<2xf16> -> f16")
        e("  %sbig = scalar.maxnumf %sS, %sSo : f16")
        e("  %sSf = scalar.extf %sS : f16 to f32")
        e("  %sbf = scalar.extf %sbig : f16 to f32")
        e("  %one32 = scalar.constant 1.0 : f32")
        e("  %rq = scalar.divf %sSf, %sbf : f32")
        e("  %isbig = scalar.cmpf oeq, %sS, %sbig : f16")
        e("  %rr = scf.select %isbig, %one32, %rq : f32")
        e("  %dkf = vector.extf %dk0 : " + V16H + " to vector<16xf32>")
        e("  %rr16 = vector.splat %rr : vector<16xf32>")
        e("  %dkr = vector.mulf %dkf, %rr16 : vector<16xf32>")
        e("  %dk = vector.fptrunc %dkr : vector<16xf32> to " + V16H)
        dk = "%dk"
    e("  %orr = index.mul %pr, %c1024 : index")   # the f16 pool row = the int4 pool's physical row
    e("  %oh = index.mul %h, %c256 : index")
    e("  %oc = index.mul %ch, %c16 : index")
    e("  %oa0 = index.add %orr, %oh : index")
    e("  %oa = index.add %oa0, %oc : index")
    e("  scf.if %live {")
    e(f"    vector.store {dk}, %o_flat[%oa] : {V16H}, view<[%otot]xf16>")
    e("  }")
    if raw:   # the half's first chunk stores s at [h][half][pool row]
        e("  %c8r = index.constant 8 : index")
        e("  %rsn = index.mul %pcap, %c8r : index")
        e("  %rs_na = buffer.assume.noalias %kso : buffer")
        e("  %rs_flat = buffer.view %rs_na[%base] : buffer -> view<[%rsn]xf16>")
        e("  %rhf = index.div %ch, %c8r : index")
        e("  %rcm = index.rem %ch, %c8r : index")
        e("  %rc0 = index.cmp eq, %rcm, %c0 : index")
        e("  %rw = scalar.andi %live, %rc0 : i1")
        e("  %ri0 = index.mul %h, %c2 : index")
        e("  %ri1 = index.add %ri0, %rhf : index")
        e("  %ri2 = index.mul %ri1, %pcap : index")
        e("  %ri = index.add %ri2, %pr : index")
        e("  %r256 = scalar.constant 0.00390625 : f16")
        e(f"  %rs = scalar.mulf {'%sbig' if khalf else '%sS'}, %r256 : f16")
        e("  scf.if %rw {")
        e("    view.store %rs, %rs_flat[%ri] : f16, view<[%rsn]xf16>")
        e("  }")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def gen_vdeq(bits=4):
    """yah_vdeq4 (kv4 prefill): the paged int4 V^T pool -> the paged f16 V^T pool [KV head][physical tile][256 dims]
    [16 keys] (vtpage's layout), tiles 0..ceil(rows / 16) - 1, with the attention's own staging decode (vq4_unpack). Thread: (KV head,
    tile, dim). Bindings: vq (i32), vstat (i32), ptab, out (f16)."""
    assert bits in (4, 8)
    sym = f"yah_vdeq{bits}"
    nw = 2 if bits == 4 else 4                  # dwords per (tile, dim)
    L = []
    e = L.append
    _deq_header(e, f"gen_vdeq{bits}", sym,
                "  %ntl0 = index.add %rows, %c15 : index\n  %ntl = index.div %ntl0, %c16 : index\n"
                "  %th0 = index.mul %ntl, %c256 : index\n  %c4 = index.constant 4 : index\n  %th = index.mul %th0, %c4 : index",
                out_cap=True)
    e("} launch(%vq: buffer, %vs: buffer, %ptab: buffer, %out: buffer) {")
    _deq_common(e, sym)
    e("  %c4 = index.constant 4 : index")
    e("  %vtiles = index.div %pcap, %c16 : index")          # pool tiles per KV head
    e("  %vq4tot0 = index.mul %vtiles, %c1024 : index")
    e(f"  %vq4tot = index.mul %vq4tot0, %c{nw} : index")
    e("  %otl0 = index.add %cap, %c15 : index")
    e("  %otl = index.div %otl0, %c16 : index")              # logical tiles per KV head
    # the f16 pool's capacity (the emitter passes the context)
    e(f"  %ocap = config.get @{sym}.out_capacity : index")
    e("  %onp0 = index.add %ocap, %c255 : index")
    e("  %onp = index.div %onp0, %c256 : index")
    e("  %ovt = index.mul %onp, %c16 : index")
    e("  %otot0 = index.mul %ovt, %c4096 : index")
    e("  %otot = index.mul %otot0, %c4 : index")
    e("  %vq_na, %vs_na, %pt_na, %o_na = buffer.assume.noalias %vq, %vs, %ptab, %out : buffer, buffer, buffer, buffer")
    e("  %vq_flat = buffer.view %vq_na[%base] : buffer -> view<[%vq4tot]xi32>")
    e("  %vs_flat = buffer.view %vs_na[%base] : buffer -> view<[%vq4tot0]xi32>")
    e("  %pt_flat = buffer.view %pt_na[%base] : buffer -> view<[%npages]xi32>")
    e("  %o_flat = buffer.view %o_na[%base] : buffer -> view<[%otot]xf16>")
    e("  %ntl0 = index.add %rows, %c15 : index")
    e("  %ntl = index.div %ntl0, %c16 : index")
    e("  %dim = index.rem %g, %c256 : index")
    e("  %gt = index.div %g, %c256 : index")
    e("  %t0 = index.rem %gt, %ntl : index")
    e("  %kvh0 = index.div %gt, %ntl : index")
    e("  %live = index.cmp ult, %kvh0, %c4 : index")
    e("  %c3 = index.constant 3 : index")
    e("  %kvh = index.min %kvh0, %c3 : index")
    e("  %otl1 = index.sub %otl, %c1 : index")
    e("  %t = index.min %t0, %otl1 : index")
    e("  %vks = index.mul %t, %c16 : index")
    e("  %lp = index.div %vks, %c256 : index")
    e("  %pg0 = view.load %pt_flat[%lp] : view<[%npages]xi32> -> i32")
    e("  %pgr = index.cast %pg0 : i32 to index")
    e("  %pg = index.assume %pgr [range(%pgr, 0, 65535), lt(%pgr, %npages)] : index")
    e("  %pb = index.mul %pg, %c16 : index")
    e("  %po = index.rem %t, %c16 : index")
    e("  %pt = index.add %pb, %po : index")
    e("  %vti0a = index.mul %kvh, %vtiles : index")
    e("  %vti0 = index.add %vti0a, %pt : index")
    e("  %vti1 = index.mul %vti0, %c256 : index")
    e("  %vti = index.add %vti1, %dim : index")
    e(f"  %vda = index.mul %vti, %c{nw} : index")
    e(f"  %vqv = vector.load %vq_flat[%vda] : view<[%vq4tot]xi32> -> vector<{nw}xi32>")
    e("  %vst = view.load %vs_flat[%vti] : view<[%vq4tot0]xi32> -> i32")
    e("  %v4g = scalar.constant 1006648320 : i32")    # 0x3c003c00
    e("  %vs1 = vector.from_elements %vst : vector<1xi32>")
    e("  %vsv = vector.bitcast %vs1 : vector<1xi32> to vector<2xf16>")
    e("  %vss = vector.extract %vsv[0] : vector<2xf16> -> f16")
    e("  %vsc = vector.extract %vsv[1] : vector<2xf16> -> f16")
    ws = []
    for d in range(nw):
        e(f"  %vw{d} = vector.extract %vqv[{d}] : vector<{nw}xi32> -> i32")
        for k, sh in enumerate((6, 2, -2, -6) if bits == 4 else (2, -6)):
            dec_pair(e, "  ", f"%vmg{d}{k}", f"%vw{d}", sh, 0x03c003c0 if bits == 4 else 0x03fc03fc, "%v4g")
            ws.append(f"%vmg{d}{k}")
    e(f"  %vpk = vector.from_elements {', '.join(ws)} : vector<8xi32>")
    e(f"  %vf = vector.bitcast %vpk : vector<8xi32> to {V16H}")
    e(f"  %vss16 = vector.splat %vss : {V16H}")
    e(f"  %vsc16 = vector.splat %vsc : {V16H}")
    e(f"  %vd = vector.fmaf %vf, %vss16, %vsc16 : {V16H}")
    e("  %oi0a = index.mul %kvh, %ovt : index")   # the f16 pool tile = the int4 pool's physical tile (own stride)
    e("  %oi0 = index.add %oi0a, %pt : index")
    e("  %oi1 = index.mul %oi0, %c256 : index")
    e("  %oi2 = index.add %oi1, %dim : index")
    e("  %oi = index.mul %oi2, %c16 : index")
    e("  scf.if %live {")
    e(f"    vector.store %vd, %o_flat[%oi] : {V16H}, view<[%otot]xf16>")
    e("  }")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def gen_vtrans():
    """yah_transpose_v16: token-major f16 V cache [token][1024] -> V^T blocked [4 kv heads][ceil(capacity/16) tiles][256 dims][16 keys].
    Tokens >= token_count are zero. 32 x 32 tiles through LDS; grid (1024/32, pitch/32 rounded up)."""
    L = []
    e = L.append
    e("// GENERATED by tools/gen_attn_fa.py (gen_vtrans) -- edit the generator.")
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
    out = sys.argv[1] if len(sys.argv) > 1 else "yah_attn_fa.loom"
    open(out, "w").write(gen())
    print(out)


if __name__ == "__main__":
    main()


gen_kdeq4 = gen_kdeq
gen_vdeq4 = gen_vdeq
