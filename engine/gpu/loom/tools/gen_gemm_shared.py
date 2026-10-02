#!/usr/bin/env python3
"""Generate the shared-decode kStore GEMM: several wave64 waves share one decoded
64-row weight tile, and the decode is row-per-lane and branch-free.

usage: gen_gemm_shared.py <fmt> [out.loom]      env: YAH_SD_NW (waves, default 2)
                                                     YAH_SD_PAD (f16 row pad, default 0), YAH_SD_KSUB (default 128)

Same ABI (weight, input, wstage, ostage, output), output layout and kStore
semantics as the chained yah_ffn_gemm_<fmt> kernel emit_prefill._chain builds,
and the same arithmetic in the same order, so the output is meant to be
bit-identical to it:

  * each weight element is decoded with the same f32 ops (d*sc, then *code)
    and rounded to f16 once;
  * every accumulator sees the same MMA sequence, K ascending in steps of 16,
    with the same lhs (16 rows of the decoded tile) and rhs (16 tokens) operands.

What changes is how much work sits between the MMAs:

  chained   one wave64 per workgroup (64 rows x 128 tokens); every 16-wide K
            step re-stages and re-decodes 64x16 elements, one element per lane
            per step, with a 4-level branch tree for the IQ4 codebook and five
            LDS byte loads per element, behind three barriers -- the decode was
            42% of the kernel in the corrected sweep (LOOM_RUNTIME.md).
  shared    NW waves per workgroup (64 rows x 128*NW tokens), each wave the same
            64x128 accumulator tile. The 64x256 block is decoded ONCE into LDS
            per 256-wide K block, shared by all NW waves, row-per-lane: a lane
            loads a group's scales once and its 16 qs bytes as one vector, and
            the codebook is a vector.table.lookup. Two barriers per block.

Decode per output falls by NW, and per element by the vectorised row-per-lane
form. The workgroup covers 128*NW tokens, so the HAL's dispatch.txt tile is
128*NW; the grid is (m_tiles/4, B/(128*NW)).
"""
import os
import struct
import sys

NW = int(os.environ.get("YAH_SD_NW", "2"))
# Row groups per workgroup: NR x NW waves, the decoded tile 64*NR rows. Waves in
# different row groups but the same token slice load identical activation
# fragments in the same barrier-aligned phase.
NR = int(os.environ.get("YAH_SD_NR", "1"))
# 16-row tiles per wave (4 = 64 rows). Fewer serve matrices whose row count is
# not a multiple of 64, e.g. the 48-row ssm_alpha/ssm_beta (m_tiles=3).
MT = int(os.environ.get("YAH_SD_MT", "4"))
LR = 16 * MT * NR
PAD = int(os.environ.get("YAH_SD_PAD", "0"))
# K columns decoded per phase. A whole 256-wide block (KSUB=256) is a 64x256 f16
# tile, ~34 KB of LDS per workgroup, which dropped residency to ~1.5 waves/SIMD
# and made the kernel slower than the chained one (13.98 -> 16.5 ms/dispatch at
# pp2048, bit-identical). A narrower phase shrinks the tile at the cost of two
# barriers per phase.
# YAH_TG_IQ4F16=1: IQ4_XS codebook looked up as f16 (bit-identical: extf of an
# exact f16 integer equals sitofp of the int8). YAH_TG_IQ4MULF=<flags> adds
# fast-math flags to the scale multiply, e.g. "<contract|nnan|nsz>" lets the
# multiply and the f16 rounding fuse into v_fma_mix (one rounding, as HIP's
# ISA does -- a numerics change).
IQ4_F16 = os.environ.get("YAH_TG_IQ4F16", "0") == "1" or os.environ.get("YAH_TG_IQ4PK", "0") == "1"
IQ4_MULF = os.environ.get("YAH_TG_IQ4MULF", "<contract|nnan|nsz>")
# Q4_K/Q5_K nibbles on 32-bit words (bit-identical; with the tile GEMM's
# decode-ahead Q4_K 6.18 -> 5.96 ms standalone). YAH_TG_Q4UITOFP=0 restores.
Q4_UITOFP = os.environ.get("YAH_TG_Q4UITOFP", "1") == "1"
# SPLIT: lanes per decoded 32-element group (IQ4_XS). The tile generator sets
# it; each lane then decodes 32/SPLIT elements (%sub_i: its part). 1 = a lane
# decodes the whole group.
SPLIT = 1
# IQ4_HDR: load the IQ4_XS block header as one 8-byte vector (bit-identical)
IQ4_HDR = os.environ.get("YAH_TG_IQ4HDR", "1") == "1"
# Q4_HDR: the Q4_K/Q5_K header (d, dmin, scales[12]) as one 16-byte load.
# Each byte load carried its own address clamp: Q4_K VALU per WMMA 10.4 -> 6.6,
# 5.93 -> 5.47 ms standalone. Q5_K is neutral (10.76 -> 10.86), so the tile
# generator turns it on for Q4_K only unless YAH_TG_Q4HDR is set.
Q4_HDR_ENV = os.environ.get("YAH_TG_Q4HDR")
Q4_HDR = Q4_HDR_ENV == "1"
# Q3_W: Q3_K header in one load and the 2+1-bit quant assembly on 32-bit words
Q3_W = os.environ.get("YAH_TG_Q3W", "1") == "1"
IQ4_W = os.environ.get("YAH_TG_IQ4W", "0") == "1"
# IQ4_U8: the codebook offset by +128 (unsigned bytes), uitofp, then -128.0:
# the same exact f32 value (small integers), so bit-identical; uitofp of a
# byte lowers to v_cvt_f32_ubyteN where sitofp needs v_bfe_i32 + v_cvt_f32_i32
IQ4_U8 = os.environ.get("YAH_TG_IQ4U8", "0") == "1"
# IQ4_U8F (implies IQ4_U8, needs IQ4_MULF): the -128 folded into the fused
# multiply: fptrunc(fma(s, u, -128*s)) = fptrunc((u-128)*s), whose exact value
# fits f32 (<= 24 significant bits), so bit-identical; uitofp of the byte is one
# v_cvt_f32_ubyteN and the bias rides v_fma_mix's f32 addend (no literal).
# Default on with the fused multiply (IQ4_XS kstore 17408x5120, real bytes:
# 26.40 -> 25.96 M cycles, decode block 254 -> 182 VALU, VALU/WMMA 8.39 -> 7.26).
IQ4_U8F = os.environ.get("YAH_TG_IQ4U8F", "1") == "1" and bool(IQ4_MULF)
IQ4_U8 = IQ4_U8 or IQ4_U8F
# IQ4_PK (implies the f16 codebook tables of IQ4_F16): the group scale rounded
# to f16 once and the multiply done in f16 (v_pk_mul_f16, two weights per op).
# NOT bit-identical (~1 f16 ulp per weight): judged by engine/run/accgate2.py.
IQ4_PK = os.environ.get("YAH_TG_IQ4PK", "0") == "1"

KSUB = PH = GPP = GPL = ROWP = None


def set_geometry(mt=None, tok=None, nw=None):
    """Override the per-wave geometry (16-row tiles, tokens, waves) for the next
    gen() calls and return the previous values, so an emitter can switch shape
    per HAL: (MT, TOK, NW)."""
    global MT, TOK, NW, NT, NA, LR
    prev = (MT, TOK, NW)
    if mt is not None:
        MT = mt
    if tok is not None:
        TOK = tok
    if nw is not None:
        NW = nw
    NT = TOK // 16
    NA = MT * NT
    LR = 16 * MT * NR
    return prev


def configure(fmt):
    """Set the per-format phase geometry. YAH_SD_KSUB overrides the format's
    measured default (FMTS[fmt]["ksub"])."""
    global KSUB, ROWP, PH, GPP, GPL
    KSUB = int(os.environ.get("YAH_SD_KSUB", FMTS[fmt]["ksub"]))
    ROWP = KSUB + PAD           # f16 per LDS row
    PH = 256 // KSUB            # phases per 256-wide block
    GPP = KSUB // 32            # 32-element groups per row per phase
    GPL = GPP // NW             # groups decoded per lane per phase
    assert 256 % KSUB == 0 and GPP % NW == 0 and GPL >= 1


# Tokens per wave. 128 gives 32 vector<4xf32> accumulators (128 VGPRs). 64 halves
# the accumulators and removes the scratch spills, and is slower anyway (IQ3_S
# 19.77 vs 15.86 ms, IQ4_XS 15.52 vs 9.71 at NW=4): the 64x128 per-wave tile's
# operand reuse is worth more than the residency and the spill traffic.
TOK = int(os.environ.get("YAH_SD_TOK", "128"))
NT = TOK // 16              # 16-token sub-tiles per wave
NA = MT * NT                # accumulators per wave
# Mechanism probes (numerically meaningless, timing only): decode = skip the
# weight decode, rhs = feed the MMAs the LDS lhs fragments instead of loading
# the activation from global, mma = drop the MMAs (accumulators pass through),
# rhsfix = load the activation fragments from K=0 every step (cache-resident).
ABLATE = os.environ.get("YAH_SD_ABLATE", "")
PREFETCH = os.environ.get("YAH_SD_PREFETCH", "1") == "1"
EPI = os.environ.get("YAH_SD_EPI", "direct")
# Carry prefetched byte vectors as packed i32 words. A vector<Nxi8> is lowered
# one byte per VGPR, so a carried qs[8] costs 8 registers across the MMA loop.
PACK = os.environ.get("YAH_SD_PACK", "1") == "1"
# Store each grid word's 4 elements as soon as they are decoded. Measured slower
# (IQ3_S 13.81 vs 12.57 ms at KSUB=64): the scheduler interleaves anyway and the
# extra stores add spill reloads. Off.
STORE4 = os.environ.get("YAH_SD_STORE4", "0") == "1"
GRID_FIRST = os.environ.get("YAH_SD_GRID_FIRST", "0") == "1"
# Inner MMA loop policy, e.g. "pipeline(%c2)" or "unroll(%c2) schedule(recurrence)".
KPOL = os.environ.get("YAH_SD_KPOL", "")
# MMA order in the K step: "" = all loads then lhs-major MMAs; "fence" =
# rhs-major with schedule fences (see the MMA loop).
KORDER = os.environ.get("YAH_SD_KORDER", "")
# Double-buffered weight tile, one barrier per phase (emit_db_loop). Measured
# slower everywhere (IQ4_XS 9.12 -> 11.51 ms at KSUB=128, 12.31 at 64; IQ3_S
# 12.02 -> 13.15): the decode is issue-bound -- WMMA and the decode share the
# VALU -- not latency-bound, so overlapping them buys nothing and the doubled
# tile costs residency. Off.
DB = os.environ.get("YAH_SD_DB", "0") == "1"
# rhs (activation) fragments software-pipelined one K step ahead.
RPF = os.environ.get("YAH_SD_RPF", "0") == "1"
# IQ3_S / IQ3_XXS: decode each sign byte's 8 elements with i8/f32 vector ops.
VDEC = os.environ.get("YAH_SD_VDEC", "1") == "1"
VDEC_W = os.environ.get("YAH_SD_VDECW", "0") == "1"
# IQ3_U8F (IQ3_S / IQ3_XXS, word path): the signed magnitude bytes XOR 0x80 are
# u = mag + 128 as unsigned bytes; v_cvt_f32_ubyteN reads each one directly and
# the -128 rides the fused multiply's f32 addend: fptrunc(fma(dsc, u, -128*dsc)).
# (u-128)*dsc has <= 24 significant bits (|mag| <= 127, dsc = d * odd <= 5 bits),
# so it is exact in f32 and the single rounding equals today's: bit-identical.
# Replaces sign-extend + sitofp + mulf + fptrunc per element.
IQ3_U8F = os.environ.get("YAH_SD_IQ3U8F", "0") == "1"
# VDECW_FR: the word path's sign spread without quarter-rate v_mul_lo_u32:
# nibble * 0x00204081 as an index multiply (both fit 24 bits: v_mul_u32_u24,
# full rate) and s1 * 255 as (s1 << 8) - s1 (each set byte becomes 0xFF; the top
# byte wraps mod 2^32). Same integers: bit-identical.
VDECW_FR = os.environ.get("YAH_SD_VDECW_FR", "0") == "1"
# Q4FMIX (Q4_K/Q5_K): the decode's subtract-and-narrow as fptrunc(fma(e, 1, -dm))
# per element instead of fptrunc(e - dm). The product by 1 is exact, so it is
# bit-identical, and it selects v_fma_mix{lo,hi}: v_cvt_f16_f32 writes only
# v0..v127, and with 128 VGPRs of accumulators live (4 x 2 waves) the allocator
# evicted accumulators to scratch for every conversion result. The 1.0 is built
# from gb & ~gb so the canonicalizer cannot fold the fma back into a subf.
Q4FMIX = os.environ.get("YAH_SD_Q4FMIX", "0") == "1"
# IQ3F16 (IQ3_S, word path): decode through two workgroup tables built at
# setup: the grid as f16 (4 KiB, 2 dwords per entry) and, per sign byte, the
# f16 sign bits of its 8 elements as four XOR masks (4 KiB). Per 8 elements:
# one 16-byte sign load, two 8-byte grid loads, four xors, eight
# fptrunc(fma(extf(mag), dsc, -0)). The product dsc*mag is exact in f32, so
# the single rounding equals the current fptrunc(f32 product): bit-identical.
IQ3F16 = os.environ.get("YAH_SD_IQ3F16", "0") == "1"


def _i8n(ty):
    import re as _re
    m = _re.fullmatch(r"vector<(\d+)xi8>", ty)
    return int(m.group(1)) if m and int(m.group(1)) % 4 == 0 else 0


def pack_vals(e, vals, tag):
    """Bitcast carried vector<Nxi8> values to vector<N/4xi32>; returns new vals."""
    if not PACK:
        return vals
    out = []
    for nm, ty in vals:
        n = _i8n(ty)
        if n:
            pk = f"{nm}_{tag}pk"
            e(f"    {pk} = vector.bitcast {nm} : {ty} to vector<{n // 4}xi32>")
            out.append((pk, f"vector<{n // 4}xi32>"))
        else:
            out.append((nm, ty))
    return out


def unpack_vals(e, cur, orig):
    """Inverse of pack_vals for loop-carried names cur with original types orig."""
    names = []
    for (nm, ty), (_, oty) in zip(cur, orig):
        n = _i8n(oty)
        if PACK and n:
            e(f"    {nm}_u = vector.bitcast {nm} : {ty} to {oty}")
            names.append(f"{nm}_u")
        else:
            names.append(nm)
    return names


IQ4_KVALUES = [-127, -104, -83, -65, -49, -35, -22, -10,
               1, 13, 25, 38, 53, 69, 89, 113]


def iq4xs_loads(p, blk, gb):
    """Issue this lane's raw-byte loads for one phase of row %drow.

    blk: i32 SSA byte offset of the row's current block; gb: i32 SSA index of
    the lane's first group in the block. Returns (lines, [(name, type)]) -- the
    loaded values, which the caller either decodes at once or carries into the
    next iteration as a prefetch.
    block_iq4_xs: d f16 @0, scales_h u16 @2, scales_l[4] @4, qs[128] @8.
    """
    L = []
    e = L.append
    vals = []
    if IQ4_HDR:
        # the 8-byte header (d, scales_h, scales_l[4]) as one load, as HIP's
        # CacheIqHeader does: one VMEM op instead of four byte/half loads
        e(f"    %{p}hd_ix = index.cast {blk} : i32 to index")
        e(f"    %{p}hd_lo = index.max %{p}hd_ix, %c0 : index")
        e(f"    %{p}hd_idx = index.min %{p}hd_lo, %w_lim8 : index")
        e(f"    %{p}hdr = vector.load %w_view[%{p}hd_idx] : view<[%w_bytes]xi8> -> vector<8xi8>")
        vals.append((f"%{p}hdr", "vector<8xi8>"))
    e(f"    %{p}d_h_i = scalar.shrui {blk}, %c1i : i32")
    e(f"    %{p}d_ix = index.cast %{p}d_h_i : i32 to index")
    e(f"    %{p}d_lo = index.max %{p}d_ix, %c0 : index")
    e(f"    %{p}d_idx = index.min %{p}d_lo, %w_half_last : index")
    if not IQ4_HDR:
        e(f"    %{p}dh = view.load %w_f16_view[%{p}d_idx] : view<[%w_halfs]xf16> -> f16")
        vals.append((f"%{p}dh", "f16"))
    for o in (() if IQ4_HDR else (2, 3)):
        e(f"    %{p}sh{o}_i = scalar.addi {blk}, %c{o}i : i32")
        e(f"    %{p}sh{o}_ix = index.cast %{p}sh{o}_i : i32 to index")
        e(f"    %{p}sh{o}_lo = index.max %{p}sh{o}_ix, %c0 : index")
        e(f"    %{p}sh{o}_idx = index.min %{p}sh{o}_lo, %w_last : index")
        e(f"    %{p}s{o} = view.load %w_view[%{p}sh{o}_idx] : view<[%w_bytes]xi8> -> i8")
        vals.append((f"%{p}s{o}", "i8"))
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}gh{u} = scalar.shrui %{p}g{u}, %c1i : i32")
        e(f"    %{p}sl_i{u} = scalar.addi {blk}, %c4i : i32")
        e(f"    %{p}sl_j{u} = scalar.addi %{p}sl_i{u}, %{p}gh{u} : i32")
        e(f"    %{p}sl_ix{u} = index.cast %{p}sl_j{u} : i32 to index")
        e(f"    %{p}sl_lo{u} = index.max %{p}sl_ix{u}, %c0 : index")
        e(f"    %{p}sl_idx{u} = index.min %{p}sl_lo{u}, %w_last : index")
        if not IQ4_HDR:
            e(f"    %{p}sl{u} = view.load %w_view[%{p}sl_idx{u}] : view<[%w_bytes]xi8> -> i8")
            vals.append((f"%{p}sl{u}", "i8"))
        e(f"    %{p}g16_{u} = scalar.shli %{p}g{u}, %c4i : i32")
        e(f"    %{p}qs_a{u} = scalar.addi {blk}, %c8i : i32")
        e(f"    %{p}qs_b{u} = scalar.addi %{p}qs_a{u}, %{p}g16_{u} : i32")
        e(f"    %{p}qs_ix{u} = index.cast %{p}qs_b{u} : i32 to index")
        e(f"    %{p}qs_lo{u} = index.max %{p}qs_ix{u}, %c0 : index")
        if SPLIT > 1:
            # this lane's 16/SPLIT qs bytes: low nibbles are elements
            # sub*n.., high nibbles 16 + sub*n..
            n = 16 // SPLIT
            e(f"    %{p}qs_s{u} = scalar.muli %sub_i, %c{n}i : i32")
            e(f"    %{p}qs_c{u} = scalar.addi %{p}qs_b{u}, %{p}qs_s{u} : i32")
            e(f"    %{p}qs_cx{u} = index.cast %{p}qs_c{u} : i32 to index")
            e(f"    %{p}qs_cl{u} = index.max %{p}qs_cx{u}, %c0 : index")
            e(f"    %{p}qs_ci{u} = index.min %{p}qs_cl{u}, %w_lim{n} : index")
            e(f"    %{p}q{u} = vector.load %w_view[%{p}qs_ci{u}] : view<[%w_bytes]xi8> -> vector<{n}xi8>")
            vals.append((f"%{p}q{u}", f"vector<{n}xi8>"))
            continue
        e(f"    %{p}qs_idx{u} = index.min %{p}qs_lo{u}, %w_lim : index")
        e(f"    %{p}q{u} = vector.load %w_view[%{p}qs_idx{u}] : view<[%w_bytes]xi8> -> vector<16xi8>")
        vals.append((f"%{p}q{u}", "vector<16xi8>"))
    return L, vals


def iq4xs_compute(v, gb):
    """Decode loaded values v (as iq4xs_loads returned them, possibly renamed to
    loop-carried names) into the LDS tile. Element g*32 + w: L = w%16,
    nib = w<16 ? qs[g*16+L]&15 : qs[g*16+L]>>4,
    sc = ((scales_l[g/2] >> 4*(g%2)) & 15 | ((scales_h >> 2g) & 3) << 4) - 32,
    value = (d*sc) * kvalues[nib] -- the chained kernel's f32 op order."""
    L = []
    e = L.append
    it = iter(v)
    if IQ4_HDR:
        hdr = next(it)
        e(f"    %hdw = vector.bitcast {hdr} : vector<8xi8> to vector<2xi32>")
        e("    %hdw0 = vector.extract %hdw[0] : vector<2xi32> -> i32")
        e("    %hdw1 = vector.extract %hdw[1] : vector<2xi32> -> i32")
        e("    %hd16 = scalar.trunci %hdw0 : i32 to i16")
        e("    %hdf = scalar.bitcast %hd16 : i16 to f16")
        e("    %d = scalar.extf %hdf : f16 to f32")
        e("    %shv = scalar.shrui %hdw0, %c16i_h : i32")
    else:
        dh = next(it); s2 = next(it); s3 = next(it)
        e(f"    %d = scalar.extf {dh} : f16 to f32")
        e(f"    %sh2_v = scalar.extui {s2} : i8 to i32")
        e(f"    %sh3_v = scalar.extui {s3} : i8 to i32")
        e("    %sh3_s = scalar.shli %sh3_v, %c8i : i32")
        e("    %shv = scalar.ori %sh2_v, %sh3_s : i32")
    for u in range(GPL):
        if IQ4_HDR:
            q = next(it)
        else:
            sl = next(it); q = next(it)
        e(f"    %g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %gp{u} = scalar.andi %g{u}, %c1i : i32")
        e(f"    %sh4_{u} = scalar.shli %gp{u}, %c2i : i32")
        e(f"    %sh2_{u} = scalar.shli %g{u}, %c1i : i32")
        if IQ4_HDR:
            # scales_l[g/2] is byte g/2 of the header's second word
            e(f"    %slg{u} = scalar.shrui %g{u}, %c1i : i32")
            e(f"    %sls{u} = scalar.shli %slg{u}, %c3i : i32")
            e(f"    %slw{u} = scalar.shrui %hdw1, %sls{u} : i32")
            e(f"    %slb{u} = scalar.andi %slw{u}, %c255i_h : i32")
        else:
            e(f"    %slb{u} = scalar.extui {sl} : i8 to i32")
        e(f"    %sc_sh{u} = scalar.shrui %slb{u}, %sh4_{u} : i32")
        e(f"    %sc_l{u} = scalar.andi %sc_sh{u}, %c15i : i32")
        e(f"    %sc_ha{u} = scalar.shrui %shv, %sh2_{u} : i32")
        e(f"    %sc_h{u} = scalar.andi %sc_ha{u}, %c3i : i32")
        e(f"    %sc_h4{u} = scalar.shli %sc_h{u}, %c4i : i32")
        e(f"    %sc6_{u} = scalar.ori %sc_l{u}, %sc_h4{u} : i32")
        e(f"    %sc{u} = scalar.subi %sc6_{u}, %c32i : i32")
        e(f"    %sc_f{u} = scalar.sitofp %sc{u} : i32 to f32")
        e(f"    %dsc{u} = scalar.mulf %d, %sc_f{u} : f32")
        if SPLIT > 1:
            assert not IQ4_F16 and not IQ4_MULF
            n = 16 // SPLIT
            e(f"    %dsc_vs{u} = vector.splat %dsc{u} : vector<{n}xf32>")
            e(f"    %m15s{u} = vector.splat %c15b_iq : vector<{n}xi8>")
            e(f"    %s4s{u} = vector.splat %c4b_iq : vector<{n}xi8>")
            e(f"    %nlo{u} = vector.andi {q}, %m15s{u} : vector<{n}xi8>")
            e(f"    %nhi{u} = vector.shrui {q}, %s4s{u} : vector<{n}xi8>")
            for half, nib in (("lo", f"%nlo{u}"), ("hi", f"%nhi{u}")):
                e(f"    %c{half}{u} = vector.table.lookup %kvt[{nib}] : vector<16xi8>, vector<{n}xi8> -> vector<{n}xi8>")
                e(f"    %f{half}{u} = vector.sitofp %c{half}{u} : vector<{n}xi8> to vector<{n}xf32>")
                e(f"    %v{half}{u} = vector.mulf %dsc_vs{u}, %f{half}{u} : vector<{n}xf32>")
                e(f"    %h{half}{u} = vector.fptrunc %v{half}{u} : vector<{n}xf32> to vector<{n}xf16>")
            _col_of(e, u)
            e(f"    %subn{u}_i = scalar.muli %sub_i, %c{n}i : i32")
            e(f"    %subn{u} = index.cast %subn{u}_i : i32 to index")
            e(f"    %cols{u} = index.add %col{u}, %subn{u} : index")
            e(f"    %colsh{u} = index.add %cols{u}, %c16 : index")
            e(f"    vector.store %hlo{u}, %wl_view[%drow, %cols{u}] : vector<{n}xf16>, view<{LR}x{ROWP}xf16>")
            e(f"    vector.store %hhi{u}, %wl_view[%drow, %colsh{u}] : vector<{n}xf16>, view<{LR}x{ROWP}xf16>")
            continue
        e(f"    %dsc_v{u} = vector.splat %dsc{u} : vector<16xf32>")
        if IQ4_W:
            # nibbles on 32-bit words (i8-vector shifts/masks lower per element)
            # only the shift moves to words: the final i8 `& 15` stays, since the
            # table lookup lowers to v_perm only with that index-range fact
            # (through a bitcast it became 960 v_cndmask: 9.44 -> 13.21 ms)
            e(f"    %qw{u} = vector.bitcast {q} : vector<16xi8> to vector<4xi32>")
            e(f"    %nhs{u} = vector.shrui %qw{u}, %s4w_iq : vector<4xi32>")
            e(f"    %nhb{u} = vector.bitcast %nhs{u} : vector<4xi32> to vector<16xi8>")
            e(f"    %nlo{u} = vector.andi {q}, %m15v : vector<16xi8>")
            e(f"    %nhi{u} = vector.andi %nhb{u}, %m15v : vector<16xi8>")
        else:
            e(f"    %nlo{u} = vector.andi {q}, %m15v : vector<16xi8>")
            e(f"    %nhi{u} = vector.shrui {q}, %s4v : vector<16xi8>")
        if IQ4_F16:
            # the codebook as f16 (every entry is an exact small integer), as
            # HIP's DecodeIqRaw looks it up: extf gives the same f32 as sitofp
            # low and high bytes of the f16 bit patterns looked up separately
            # (byte tables lower to v_perm), interleaved into f16 values
            for part, nib in (("lo", f"%nlo{u}"), ("hi", f"%nhi{u}")):
                e(f"    %cb{part}{u}l = vector.table.lookup %kvtl[{nib}] : vector<16xi8>, vector<16xi8> -> vector<16xi8>")
                e(f"    %cb{part}{u}h = vector.table.lookup %kvth[{nib}] : vector<16xi8>, vector<16xi8> -> vector<16xi8>")
                e(f"    %cb{part}{u}lw = vector.extui %cb{part}{u}l : vector<16xi8> to vector<16xi16>")
                e(f"    %cb{part}{u}hw = vector.extui %cb{part}{u}h : vector<16xi8> to vector<16xi16>")
                e(f"    %cb{part}{u}hs = vector.shli %cb{part}{u}hw, %s8w : vector<16xi16>")
                e(f"    %cb{part}{u}i = vector.ori %cb{part}{u}lw, %cb{part}{u}hs : vector<16xi16>")
                e(f"    %c{part}{u} = vector.bitcast %cb{part}{u}i : vector<16xi16> to vector<16xf16>")
                e(f"    %f{part}{u} = vector.extf %c{part}{u} : vector<16xf16> to vector<16xf32>")
        elif IQ4_U8:
            for part in ("lo", "hi"):
                e(f"    %cu{part}{u} = vector.table.lookup %kvtu[%n{part}{u}] : vector<16xi8>, vector<16xi8> -> vector<16xi8>")
                e(f"    %fu{part}{u} = vector.uitofp %cu{part}{u} : vector<16xi8> to vector<16xf32>")
                if not IQ4_U8F:
                    e(f"    %f{part}{u} = vector.subf %fu{part}{u}, %c128v_iq : vector<16xf32>")
            if IQ4_U8F:
                assert IQ4_MULF, "IQ4_U8F folds the bias into the fused multiply (YAH_TG_IQ4MULF)"
                e(f"    %nb{u} = scalar.mulf %dsc{u}, %cm128f_iq : f32")
        else:
            e(f"    %clo{u} = vector.table.lookup %kvt[%nlo{u}] : vector<16xi8>, vector<16xi8> -> vector<16xi8>")
            e(f"    %chi{u} = vector.table.lookup %kvt[%nhi{u}] : vector<16xi8>, vector<16xi8> -> vector<16xi8>")
            e(f"    %flo{u} = vector.sitofp %clo{u} : vector<16xi8> to vector<16xf32>")
            e(f"    %fhi{u} = vector.sitofp %chi{u} : vector<16xi8> to vector<16xf32>")
        if IQ4_PK:
            e(f"    %dsch{u} = scalar.fptrunc %dsc{u} : f32 to f16")
            e(f"    %dschv{u} = vector.splat %dsch{u} : vector<16xf16>")
            e(f"    %hlo{u} = vector.mulf %dschv{u}, %clo{u} : vector<16xf16>")
            e(f"    %hhi{u} = vector.mulf %dschv{u}, %chi{u} : vector<16xf16>")
        elif IQ4_MULF:
            # scalar form: fptrunc(mulf<contract>) pairs feeding from_elements
            # are what AMDGPU source-to-low selects as v_fma_mix{lo,hi}_f16
            for part in ("lo", "hi"):
                hs = []
                for j in range(16):
                    if IQ4_F16:
                        e(f"    %x{part}{u}_{j} = vector.extract %c{part}{u}[{j}] : vector<16xf16> -> f16")
                        e(f"    %y{part}{u}_{j} = scalar.extf %x{part}{u}_{j} : f16 to f32")
                    elif IQ4_U8F:
                        e(f"    %y{part}{u}_{j} = vector.extract %fu{part}{u}[{j}] : vector<16xf32> -> f32")
                    else:
                        e(f"    %y{part}{u}_{j} = vector.extract %f{part}{u}[{j}] : vector<16xf32> -> f32")
                    if IQ4_U8F:
                        e(f"    %m{part}{u}_{j} = scalar.fmaf %dsc{u}, %y{part}{u}_{j}, %nb{u} : f32")
                    else:
                        e(f"    %m{part}{u}_{j} = scalar.mulf{IQ4_MULF} %dsc{u}, %y{part}{u}_{j} : f32")
                    e(f"    %t{part}{u}_{j} = scalar.fptrunc %m{part}{u}_{j} : f32 to f16")
                    hs.append(f"%t{part}{u}_{j}")
                e(f"    %h{part}{u} = vector.from_elements {', '.join(hs)} : vector<16xf16>")
        else:
            e(f"    %vlo{u} = vector.mulf{IQ4_MULF} %dsc_v{u}, %flo{u} : vector<16xf32>")
            e(f"    %vhi{u} = vector.mulf{IQ4_MULF} %dsc_v{u}, %fhi{u} : vector<16xf32>")
            e(f"    %hlo{u} = vector.fptrunc %vlo{u} : vector<16xf32> to vector<16xf16>")
            e(f"    %hhi{u} = vector.fptrunc %vhi{u} : vector<16xf32> to vector<16xf16>")
        e(f"    %col_i{u} = scalar.shli %gl{u}, %c5i : i32")
        e(f"    %col_x{u} = index.cast %col_i{u} : i32 to index")
        e(f"    %col_l{u} = index.max %col_x{u}, %c0 : index")
        e(f"    %col{u} = index.min %col_l{u}, %ccolmax : index")
        e(f"    %colh{u} = index.add %col{u}, %c16 : index")
        e(f"    vector.store %hlo{u}, %wl_view[%drow, %col{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
        e(f"    vector.store %hhi{u}, %wl_view[%drow, %colh{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
    return L


def _ld8(e, p, name, off):
    """Load one weight byte at i32 SSA offset `off` into %{p}{name} (i8)."""
    e(f"    %{p}{name}_ix = index.cast {off} : i32 to index")
    e(f"    %{p}{name}_lo = index.max %{p}{name}_ix, %c0 : index")
    e(f"    %{p}{name}_idx = index.min %{p}{name}_lo, %w_last : index")
    e(f"    %{p}{name} = view.load %w_view[%{p}{name}_idx] : view<[%w_bytes]xi8> -> i8")


def _ldv(e, p, name, off, n):
    """Load n consecutive weight bytes at i32 SSA offset `off` as vector<nxi8>.

    The clamp must be w_bytes - n for THIS width. Clamping every width to
    w_bytes - 16 moved the last row's final sign bytes (IQ3_S +74+4g, within the
    last 16 bytes of the tensor) to the wrong address: one wrong row per
    tensor, argmax 29779 at pp2048, caught by the r64t256 fixture at row 63."""
    e(f"    %{p}{name}_ix = index.cast {off} : i32 to index")
    e(f"    %{p}{name}_lo = index.max %{p}{name}_ix, %c0 : index")
    e(f"    %{p}{name}_idx = index.min %{p}{name}_lo, %w_lim{n} : index")
    e(f"    %{p}{name} = view.load %w_view[%{p}{name}_idx] : view<[%w_bytes]xi8> -> vector<{n}xi8>".replace("view.load", "vector.load"))


def _ldd(e, p, blk):
    """Load the block's f16 d (offset 0) into %{p}dh."""
    e(f"    %{p}d_h_i = scalar.shrui {blk}, %c1i : i32")
    e(f"    %{p}d_ix = index.cast %{p}d_h_i : i32 to index")
    e(f"    %{p}d_lo = index.max %{p}d_ix, %c0 : index")
    e(f"    %{p}d_idx = index.min %{p}d_lo, %w_half_last : index")
    e(f"    %{p}dh = view.load %w_f16_view[%{p}d_idx] : view<[%w_halfs]xf16> -> f16")


def _store_group(e, u, hs):
    """Store the 32 f16 SSA values hs of group u (local column %gl{u}*32)."""
    e(f"    %hlo{u} = vector.from_elements " + ", ".join(hs[:16]) + " : vector<16xf16>")
    e(f"    %hhi{u} = vector.from_elements " + ", ".join(hs[16:]) + " : vector<16xf16>")
    e(f"    %col_i{u} = scalar.shli %gl{u}, %c5i : i32")
    e(f"    %col_x{u} = index.cast %col_i{u} : i32 to index")
    e(f"    %col_l{u} = index.max %col_x{u}, %c0 : index")
    e(f"    %col{u} = index.min %col_l{u}, %ccolmax : index")
    e(f"    %colh{u} = index.add %col{u}, %c16 : index")
    e(f"    vector.store %hlo{u}, %wl_view[%drow, %col{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
    e(f"    vector.store %hhi{u}, %wl_view[%drow, %colh{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")


def _vdec_pair(e, t, gw0, gw1, sgb8, dsc_v8, col, u, p, dsc_s=None):
    """Vector decode of the 8 elements one sign byte covers (grid words lw=2p and
    2p+1): mags = bytes of [gw0, gw1], s = bits of the sign byte (LSB first,
    matching element lw*4 + b), mag = (g ^ -s) + s in i8 -- exact because grid
    magnitudes are < 128 -- then one sitofp, mulf and fptrunc for all eight."""
    if VDEC_W:
        # on the two grid words: s1 = sign bit i in byte i (the nibble times
        # 0x00204081 puts bit i at bit 8i), m = s1 * 255, mag = (g ^ m) + s1 --
        # per byte 256 - g for a set bit, never a carry since every grid
        # magnitude is > 0. (The i8-vector form lowers element by element.)
        e(f"    %wsb_{t} = scalar.extui {sgb8} : i8 to i32")
        for h, gw in ((0, gw0), (1, gw1)):
            if h:
                e(f"    %wsn{h}_{t}0 = scalar.shrui %wsb_{t}, %c4i : i32")
            else:
                e(f"    %wsn{h}_{t}0 = scalar.addi %wsb_{t}, %c0i : i32")
            e(f"    %wsn{h}_{t} = scalar.andi %wsn{h}_{t}0, %c15i : i32")
            if VDECW_FR:
                # n * 0x00204081 & 0x01010101 == (t | t << 14) & 0x01010101 with
                # t = n | n << 7 (n <= 15: bit i lands at 8i, OR cannot carry)
                e(f"    %wst7{h}_{t} = scalar.shli %wsn{h}_{t}, %c7i : i32")
                e(f"    %wst{h}_{t} = scalar.ori %wsn{h}_{t}, %wst7{h}_{t} : i32")
                e(f"    %wst14{h}_{t} = scalar.shli %wst{h}_{t}, %c14i_vdw : i32")
                e(f"    %wsp{h}_{t} = scalar.ori %wst{h}_{t}, %wst14{h}_{t} : i32")
            else:
                e(f"    %wsp{h}_{t} = scalar.muli %wsn{h}_{t}, %vdw_spread : i32")
            e(f"    %ws1{h}_{t} = scalar.andi %wsp{h}_{t}, %vdw_ones : i32")
            if VDECW_FR:
                e(f"    %wsm8{h}_{t} = scalar.shli %ws1{h}_{t}, %c8i : i32")
                e(f"    %wsm{h}_{t} = scalar.subi %wsm8{h}_{t}, %ws1{h}_{t} : i32")
            else:
                e(f"    %wsm{h}_{t} = scalar.muli %ws1{h}_{t}, %vdw_ff : i32")
            e(f"    %wx{h}_{t} = scalar.xori {gw}, %wsm{h}_{t} : i32")
            e(f"    %wm{h}_{t} = scalar.addi %wx{h}_{t}, %ws1{h}_{t} : i32")
        e(f"    %vmw_{t} = vector.from_elements %wm0_{t}, %wm1_{t} : vector<2xi32>")
        e(f"    %vm_{t} = vector.bitcast %vmw_{t} : vector<2xi32> to vector<8xi8>")
    else:
        e(f"    %vg_{t} = vector.from_elements {gw0}, {gw1} : vector<2xi32>")
        e(f"    %vb_{t} = vector.bitcast %vg_{t} : vector<2xi32> to vector<8xi8>")
        e(f"    %vs1_{t} = vector.from_elements {sgb8} : vector<1xi8>")
        e(f"    %vs_{t} = vector.bitunpacku<1> %vs1_{t} : vector<1xi8> -> vector<8xi8>")
        e(f"    %vn_{t} = vector.subi %z8v, %vs_{t} : vector<8xi8>")
        e(f"    %vx_{t} = vector.xori %vb_{t}, %vn_{t} : vector<8xi8>")
        e(f"    %vm_{t} = vector.addi %vx_{t}, %vs_{t} : vector<8xi8>")
    if IQ3_U8F and VDEC_W and dsc_s is not None:
        e(f"    %wu0_{t} = scalar.xori %wm0_{t}, %c80x4_iq3 : i32")
        e(f"    %wu1_{t} = scalar.xori %wm1_{t}, %c80x4_iq3 : i32")
        e(f"    %vuw_{t} = vector.from_elements %wu0_{t}, %wu1_{t} : vector<2xi32>")
        e(f"    %vub_{t} = vector.bitcast %vuw_{t} : vector<2xi32> to vector<8xi8>")
        e(f"    %vuf_{t} = vector.uitofp %vub_{t} : vector<8xi8> to vector<8xf32>")
        e(f"    %unb_{t} = scalar.mulf {dsc_s}, %cm128f_iq3 : f32")
        hs = []
        for j in range(8):
            e(f"    %uy_{t}_{j} = vector.extract %vuf_{t}[{j}] : vector<8xf32> -> f32")
            e(f"    %um_{t}_{j} = scalar.fmaf {dsc_s}, %uy_{t}_{j}, %unb_{t} : f32")
            e(f"    %uh_{t}_{j} = scalar.fptrunc %um_{t}_{j} : f32 to f16")
            hs.append(f"%uh_{t}_{j}")
        e(f"    %vh_{t} = vector.from_elements {', '.join(hs)} : vector<8xf16>")
    else:
        e(f"    %vf_{t} = vector.sitofp %vm_{t} : vector<8xi8> to vector<8xf32>")
        e(f"    %vv_{t} = vector.mulf {dsc_v8}, %vf_{t} : vector<8xf32>")
        e(f"    %vh_{t} = vector.fptrunc %vv_{t} : vector<8xf32> to vector<8xf16>")
    e(f"    %vc_{t} = index.constant {8 * p} : index")
    e(f"    %vco_{t} = index.add {col}, %vc_{t} : index")
    e(f"    vector.store %vh_{t}, %wl_view[%drow, %vco_{t}] : vector<8xf16>, view<{LR}x{ROWP}xf16>")


def _vdec16_pair(e, t, gid0, gid1, sgb8, dsc, col, p):
    """IQ3F16 decode of the 8 elements one sign byte covers (grid words 2p,
    2p+1): sign masks from %sgn16 (4 dwords per sign byte, dword j = elements
    2j, 2j+1 as f16 sign bits), magnitudes from %g16 (2 dwords per grid entry,
    4 f16), value = fptrunc(fma(extf(+-mag), dsc, -0))."""
    e(f"    %fsb_{t} = scalar.extui {sgb8} : i8 to i32")
    e(f"    %fsx_{t} = index.cast %fsb_{t} : i32 to index")
    e(f"    %fsm_{t} = index.min %fsx_{t}, %c255_f16 : index")
    e(f"    %fso_{t} = index.mul %fsm_{t}, %c4 : index")
    e(f"    %fsv_{t} = vector.load %sgn16[%fso_{t}] : view<1024xi32> -> vector<4xi32>")
    hs = []
    for h, gid in ((0, gid0), (1, gid1)):
        e(f"    %fgo{h}_{t} = index.mul {gid}, %c2 : index")
        e(f"    %fgv{h}_{t} = vector.load %g16[%fgo{h}_{t}] : view<1024xi32> -> vector<2xi32>")
        e(f"    %fm0{h}_{t} = vector.extract %fsv_{t}[{2 * h}] : vector<4xi32> -> i32")
        e(f"    %fm1{h}_{t} = vector.extract %fsv_{t}[{2 * h + 1}] : vector<4xi32> -> i32")
        e(f"    %fmv{h}_{t} = vector.from_elements %fm0{h}_{t}, %fm1{h}_{t} : vector<2xi32>")
        e(f"    %fxv{h}_{t} = vector.xori %fgv{h}_{t}, %fmv{h}_{t} : vector<2xi32>")
        e(f"    %fhv{h}_{t} = vector.bitcast %fxv{h}_{t} : vector<2xi32> to vector<4xf16>")
        for b in range(4):
            x = f"{h}{b}_{t}"
            e(f"    %fe{x} = vector.extract %fhv{h}_{t}[{b}] : vector<4xf16> -> f16")
            e(f"    %ff{x} = scalar.extf %fe{x} : f16 to f32")
            e(f"    %fq{x} = scalar.fmaf %ff{x}, {dsc}, %negzero_f16 : f32")
            e(f"    %fo{x} = scalar.fptrunc %fq{x} : f32 to f16")
            hs.append(f"%fo{x}")
    e(f"    %vh_{t} = vector.from_elements {', '.join(hs)} : vector<8xf16>")
    e(f"    %vc_{t} = index.constant {8 * p} : index")
    e(f"    %vco_{t} = index.add {col}, %vc_{t} : index")
    e(f"    vector.store %vh_{t}, %wl_view[%drow, %vco_{t}] : vector<8xf16>, view<{LR}x{ROWP}xf16>")


def _iq3f16_tables():
    """Build %g16 (f16 grid) from %grid_g and %sgn16 (sign masks) in
    workgroup memory; the first K phase's barrier publishes them."""
    L = ["  %g16_bytes = index.constant 4096 : offset",
         "  %g16_l = buffer.alloca<workgroup> align(16) %g16_bytes : buffer",
         "  %g16 = buffer.view %g16_l[%base] : buffer -> view<1024xi32>",
         "  %sgn16_l = buffer.alloca<workgroup> align(16) %g16_bytes : buffer",
         "  %sgn16 = buffer.view %sgn16_l[%base] : buffer -> view<1024xi32>",
         "  %c255_f16 = index.constant 255 : index",
         "  %negzero_f16 = scalar.constant -0.0 : f32",
         "  %f16_c255 = scalar.constant 255 : i32", "  %f16_c16 = scalar.constant 16 : i32",
         "  %f16_c1 = scalar.constant 1 : i32", "  %f16_c15 = scalar.constant 15 : i32",
         "  %f16_c31 = scalar.constant 31 : i32",
         f"  %f16_step = index.constant {64 * NW} : index",
         "  %f16_s1 = scf.for %fi = [%c0 to %c512 step %f16_step](%fm = %c0 : index) -> (index) {",
         "    %fe0 = index.add %fi, %tid : index",
         "    %fe = index.min %fe0, %c511 : index",
         "    %fw = view.load %grid_g[%fe] : view<512xi32> -> i32"]
    hs = []
    for b in range(4):
        L += [f"    %fsh{b} = scalar.constant {8 * b} : i32",
              f"    %fb0{b} = scalar.shrui %fw, %fsh{b} : i32",
              f"    %fb{b} = scalar.andi %fb0{b}, %f16_c255 : i32",
              f"    %fbf{b} = scalar.uitofp %fb{b} : i32 to f32",
              f"    %fbh{b} = scalar.fptrunc %fbf{b} : f32 to f16",
              f"    %fbi{b} = scalar.bitcast %fbh{b} : f16 to i16",
              f"    %fbz{b} = scalar.extui %fbi{b} : i16 to i32"]
    L += ["    %fhi1 = scalar.shli %fbz1, %f16_c16 : i32", "    %fd0 = scalar.ori %fbz0, %fhi1 : i32",
          "    %fhi3 = scalar.shli %fbz3, %f16_c16 : i32", "    %fd1 = scalar.ori %fbz2, %fhi3 : i32",
          "    %fo0 = index.mul %fe, %c2 : index", "    %fo1 = index.add %fo0, %c1 : index",
          "    view.store %fd0, %g16[%fo0] : i32, view<1024xi32>",
          "    view.store %fd1, %g16[%fo1] : i32, view<1024xi32>",
          "    scf.yield %fm : index", "  }",
          "  %f16_s2 = scf.for %si = [%c0 to %c256 step %f16_step](%sm = %c0 : index) -> (index) {",
          "    %se0 = index.add %si, %tid : index",
          "    %se = index.min %se0, %c255_f16 : index",
          "    %sei = index.cast %se : index to i32"]
    for j in range(4):
        L += [f"    %sl{j}c = scalar.constant {2 * j} : i32",
              f"    %sh{j}c = scalar.constant {2 * j + 1} : i32",
              f"    %sl{j}a = scalar.shrui %sei, %sl{j}c : i32", f"    %sl{j} = scalar.andi %sl{j}a, %f16_c1 : i32",
              f"    %sh{j}a = scalar.shrui %sei, %sh{j}c : i32", f"    %sh{j} = scalar.andi %sh{j}a, %f16_c1 : i32",
              f"    %sl{j}s = scalar.shli %sl{j}, %f16_c15 : i32", f"    %sh{j}s = scalar.shli %sh{j}, %f16_c31 : i32",
              f"    %sm{j} = scalar.ori %sl{j}s, %sh{j}s : i32",
              f"    %so{j}b = index.mul %se, %c4 : index", f"    %so{j}c = index.constant {j} : index",
              f"    %so{j} = index.add %so{j}b, %so{j}c : index",
              f"    view.store %sm{j}, %sgn16[%so{j}] : i32, view<1024xi32>"]
    L += ["    scf.yield %sm : index", "  }"]
    return L


def _col_of(e, u):
    e(f"    %col_i{u} = scalar.shli %gl{u}, %c5i : i32")
    e(f"    %col_x{u} = index.cast %col_i{u} : i32 to index")
    e(f"    %col_l{u} = index.max %col_x{u}, %c0 : index")
    e(f"    %col{u} = index.min %col_l{u}, %ccolmax : index")


def iq3s_loads(p, blk, gb):
    """block_iq3_s (110 B): d f16 @0, qs[64] @2, qh[8] @66, signs[32] @74,
    scales[4] @106. Group g reads qs[8g..8g+7], qh[g], signs[4g..4g+3] and the
    scale byte g/2."""
    L = []
    e = L.append
    vals = []
    _ldd(e, p, blk)
    vals.append((f"%{p}dh", "f16"))
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}g8_{u} = scalar.shli %{p}g{u}, %c3i : i32")
        e(f"    %{p}qo{u} = scalar.addi {blk}, %{p}g8_{u} : i32")
        e(f"    %{p}qo2_{u} = scalar.addi %{p}qo{u}, %c2i : i32")
        _ldv(e, p, f"qs{u}", f"%{p}qo2_{u}", 8)
        vals.append((f"%{p}qs{u}", "vector<8xi8>"))
        e(f"    %{p}ho{u} = scalar.addi {blk}, %{p}g{u} : i32")
        e(f"    %{p}ho2_{u} = scalar.addi %{p}ho{u}, %c66i : i32")
        _ld8(e, p, f"qh{u}", f"%{p}ho2_{u}")
        vals.append((f"%{p}qh{u}", "i8"))
        e(f"    %{p}g4_{u} = scalar.shli %{p}g{u}, %c2i : i32")
        e(f"    %{p}so{u} = scalar.addi {blk}, %{p}g4_{u} : i32")
        e(f"    %{p}so2_{u} = scalar.addi %{p}so{u}, %c74i : i32")
        _ldv(e, p, f"sg{u}", f"%{p}so2_{u}", 4)
        vals.append((f"%{p}sg{u}", "vector<4xi8>"))
        e(f"    %{p}gd{u} = scalar.shrui %{p}g{u}, %c1i : i32")
        e(f"    %{p}co{u} = scalar.addi {blk}, %{p}gd{u} : i32")
        e(f"    %{p}co2_{u} = scalar.addi %{p}co{u}, %c106i : i32")
        _ld8(e, p, f"sc{u}", f"%{p}co2_{u}")
        vals.append((f"%{p}sc{u}", "i8"))
    return L, vals


def iq3s_compute(v, gb):
    """The chained kernel's IQ3_S element decode, one row per lane. Element
    lw*4 + b of group g (lw = 2*l + which):
      gidx = qs[8g+lw] | ((qh[g] >> lw) & 1) << 8,  gword = grid[gidx]
      sign_nib = (signs[4g+l] >> 4*which) & 15,     s = (sign_nib >> b) & 1
      mag = ((gword >> 8b) & 255 ^ -s) + s
      value = (d * f32(1 + 2*((scales[g/2] >> 4*(g%2)) & 15))) * f32(mag)"""
    L = []
    e = L.append
    it = iter(v)
    dh = next(it)
    e(f"    %d = scalar.extf {dh} : f16 to f32")
    for u in range(GPL):
        qs = next(it); qh = next(it); sg = next(it); sc = next(it)
        e(f"    %g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %odd{u} = scalar.andi %g{u}, %c1i : i32")
        e(f"    %nb_sh{u} = scalar.shli %odd{u}, %c2i : i32")
        e(f"    %scb{u} = scalar.extui {sc} : i8 to i32")
        e(f"    %nb_t{u} = scalar.shrui %scb{u}, %nb_sh{u} : i32")
        e(f"    %nib{u} = scalar.andi %nb_t{u}, %c15i : i32")
        e(f"    %nib2_{u} = scalar.shli %nib{u}, %c1i : i32")
        e(f"    %onep{u} = scalar.addi %nib2_{u}, %c1i : i32")
        e(f"    %scf{u} = scalar.sitofp %onep{u} : i32 to f32")
        e(f"    %dsc{u} = scalar.mulf %d, %scf{u} : f32")
        e(f"    %qhb{u} = scalar.extui {qh} : i8 to i32")
        if VDEC:
            e(f"    %dsc_v8_{u} = vector.splat %dsc{u} : vector<8xf32>")
            _col_of(e, u)
            gws = []
            for lw in range(8):
                t = f"{u}_{lw}"
                e(f"    %qlo8_{t} = vector.extract {qs}[{lw}] : vector<8xi8> -> i8")
                e(f"    %qlo_{t} = scalar.extui %qlo8_{t} : i8 to i32")
                e(f"    %hb0_{t} = scalar.shrui %qhb{u}, %c{lw}i : i32")
                e(f"    %hb_{t} = scalar.andi %hb0_{t}, %c1i : i32")
                e(f"    %hb8_{t} = scalar.shli %hb_{t}, %c8i : i32")
                e(f"    %gi_{t} = scalar.ori %qlo_{t}, %hb8_{t} : i32")
                e(f"    %gix_{t} = index.cast %gi_{t} : i32 to index")
                e(f"    %gil_{t} = index.max %gix_{t}, %c0 : index")
                e(f"    %gid_{t} = index.min %gil_{t}, %c511 : index")
                if IQ3F16:
                    gws.append(f"%gid_{t}")
                    continue
                e(f"    %gw_{t} = view.load %grid_view[%gid_{t}] : view<512xi32> -> i32")
                gws.append(f"%gw_{t}")
            for pp in range(4):
                e(f"    %sgb8_{u}_{pp} = vector.extract {sg}[{pp}] : vector<4xi8> -> i8")
                if IQ3F16:
                    _vdec16_pair(e, f"{u}_{pp}", gws[2 * pp], gws[2 * pp + 1], f"%sgb8_{u}_{pp}", f"%dsc{u}", f"%col{u}", pp)
                else:
                    _vdec_pair(e, f"{u}_{pp}", gws[2 * pp], gws[2 * pp + 1], f"%sgb8_{u}_{pp}", f"%dsc_v8_{u}", f"%col{u}", u, pp, f"%dsc{u}")
            continue
        hs = []
        for lw in (range(8) if GRID_FIRST else ()):
            # all eight lookups first, so they can be in flight together instead
            # of each one being drained before its element math
            t = f"{u}_{lw}"
            e(f"    %qlo8_{t} = vector.extract {qs}[{lw}] : vector<8xi8> -> i8")
            e(f"    %qlo_{t} = scalar.extui %qlo8_{t} : i8 to i32")
            e(f"    %hb0_{t} = scalar.shrui %qhb{u}, %c{lw}i : i32")
            e(f"    %hb_{t} = scalar.andi %hb0_{t}, %c1i : i32")
            e(f"    %hb8_{t} = scalar.shli %hb_{t}, %c8i : i32")
            e(f"    %gi_{t} = scalar.ori %qlo_{t}, %hb8_{t} : i32")
            e(f"    %gix_{t} = index.cast %gi_{t} : i32 to index")
            e(f"    %gil_{t} = index.max %gix_{t}, %c0 : index")
            e(f"    %gid_{t} = index.min %gil_{t}, %c511 : index")
            e(f"    %gw_{t} = view.load %grid_view[%gid_{t}] : view<512xi32> -> i32")
        for lw in range(8):
            l, which = lw // 2, lw % 2
            t = f"{u}_{lw}"
            if GRID_FIRST:
                pass
            else:
                e(f"    %qlo8_{t} = vector.extract {qs}[{lw}] : vector<8xi8> -> i8")
            if not GRID_FIRST:
                e(f"    %qlo_{t} = scalar.extui %qlo8_{t} : i8 to i32")
                e(f"    %hb0_{t} = scalar.shrui %qhb{u}, %c{lw}i : i32")
                e(f"    %hb_{t} = scalar.andi %hb0_{t}, %c1i : i32")
                e(f"    %hb8_{t} = scalar.shli %hb_{t}, %c8i : i32")
                e(f"    %gi_{t} = scalar.ori %qlo_{t}, %hb8_{t} : i32")
                e(f"    %gix_{t} = index.cast %gi_{t} : i32 to index")
                e(f"    %gil_{t} = index.max %gix_{t}, %c0 : index")
                e(f"    %gid_{t} = index.min %gil_{t}, %c511 : index")
                e(f"    %gw_{t} = view.load %grid_view[%gid_{t}] : view<512xi32> -> i32")
            if which == 0:
                e(f"    %sgb8_{u}_{l} = vector.extract {sg}[{l}] : vector<4xi8> -> i8")
                e(f"    %sgb_{u}_{l} = scalar.extui %sgb8_{u}_{l} : i8 to i32")
            e(f"    %snt_{t} = scalar.shrui %sgb_{u}_{l}, %c{4 * which}i : i32")
            e(f"    %sn_{t} = scalar.andi %snt_{t}, %c15i : i32")
            for b in range(4):
                x = f"{t}_{b}"
                e(f"    %gb0_{x} = scalar.shrui %gw_{t}, %c{8 * b}i : i32")
                e(f"    %gby_{x} = scalar.andi %gb0_{x}, %c255i : i32")
                e(f"    %sb0_{x} = scalar.shrui %sn_{t}, %c{b}i : i32")
                e(f"    %sb_{x} = scalar.andi %sb0_{x}, %c1i : i32")
                e(f"    %ng_{x} = scalar.subi %c0i, %sb_{x} : i32")
                e(f"    %mg0_{x} = scalar.xori %gby_{x}, %ng_{x} : i32")
                e(f"    %mg_{x} = scalar.addi %mg0_{x}, %sb_{x} : i32")
                e(f"    %mf_{x} = scalar.sitofp %mg_{x} : i32 to f32")
                e(f"    %vv_{x} = scalar.mulf %dsc{u}, %mf_{x} : f32")
                e(f"    %hv_{x} = scalar.fptrunc %vv_{x} : f32 to f16")
                hs.append(f"%hv_{x}")
            if STORE4:
                # store this grid word's 4 elements now: collecting all 32 for two
                # 16-wide stores kept them live across the whole group decode and
                # pushed the accumulators into scratch
                if lw == 0:
                    e(f"    %col_i{u} = scalar.shli %gl{u}, %c5i : i32")
                    e(f"    %col_x{u} = index.cast %col_i{u} : i32 to index")
                    e(f"    %col_l{u} = index.max %col_x{u}, %c0 : index")
                    e(f"    %col{u} = index.min %col_l{u}, %ccolmax : index")
                e(f"    %h4_{t} = vector.from_elements " + ", ".join(hs[-4:]) + " : vector<4xf16>")
                e(f"    %c4w_{t} = index.constant {4 * lw} : index")
                e(f"    %col4_{t} = index.add %col{u}, %c4w_{t} : index")
                e(f"    vector.store %h4_{t}, %wl_view[%drow, %col4_{t}] : vector<4xf16>, view<{LR}x{ROWP}xf16>")
        if not STORE4:
            _store_group(e, u, hs)
    return L


GRID_LDS = os.environ.get("YAH_SD_GRID_LDS", "1") == "1"


def iq3s_setup():
    """The 512-word grid as %grid_view. With GRID_LDS (default) it is copied into
    2 KiB of workgroup memory once, so the eight lookups per 32-element group are
    ds_reads instead of dependent global gathers in front of every phase barrier.
    The first K phase starts with a barrier, which publishes the copy."""
    L = ["  %grid_g = buffer.view %grid_na[%base] : buffer -> view<512xi32>",
         "  %c511 = index.constant 511 : index",
         "  %vdw_spread = scalar.constant 2113665 : i32", "  %vdw_ones = scalar.constant 16843009 : i32",
         "  %vdw_ff = scalar.constant 255 : i32",
         "  %c80x4_iq3 = scalar.constant -2139062144 : i32", "  %cm128f_iq3 = scalar.constant -128.0 : f32",
         "  %c14i_vdw = scalar.constant 14 : i32"]
    if IQ3F16:
        return L + _iq3f16_tables()
    if not GRID_LDS:
        return L + ["  %grid_view = buffer.view %grid_na[%base] : buffer -> view<512xi32>"]
    L += ["  %grid_bytes = index.constant 2048 : offset",
          "  %grid_l = buffer.alloca<workgroup> align(16) %grid_bytes : buffer",
          "  %grid_view = buffer.view %grid_l[%base] : buffer -> view<512xi32>",
          f"  %cgstep = index.constant {64 * NW} : index",
          "  %gsink = scf.for %gi = [%c0 to %c512 step %cgstep](%gm = %c0 : index) -> (index) {",
          "    %gidx0 = index.add %gi, %tid : index",
          "    %gidx = index.min %gidx0, %c511 : index",
          "    %gv = view.load %grid_g[%gidx] : view<512xi32> -> i32",
          "    view.store %gv, %grid_view[%gidx] : i32, view<512xi32>",
          "    scf.yield %gm : index",
          "  }"]
    return L


def iq3xxs_loads(p, blk, gb):
    """block_iq3_xxs (98 B): d f16 @0, qs[64] @2 (grid indices), aux[32] @66
    (one LE32 word per 32-element group). Group g reads qs[8g..8g+7] and aux
    bytes 66+4g..69+4g."""
    L = []
    e = L.append
    vals = []
    _ldd(e, p, blk)
    vals.append((f"%{p}dh", "f16"))
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}g8_{u} = scalar.shli %{p}g{u}, %c3i : i32")
        e(f"    %{p}qo{u} = scalar.addi {blk}, %{p}g8_{u} : i32")
        e(f"    %{p}qo2_{u} = scalar.addi %{p}qo{u}, %c2i : i32")
        _ldv(e, p, f"qs{u}", f"%{p}qo2_{u}", 8)
        vals.append((f"%{p}qs{u}", "vector<8xi8>"))
        e(f"    %{p}g4_{u} = scalar.shli %{p}g{u}, %c2i : i32")
        e(f"    %{p}ao{u} = scalar.addi {blk}, %{p}g4_{u} : i32")
        e(f"    %{p}ao2_{u} = scalar.addi %{p}ao{u}, %c66i : i32")
        _ldv(e, p, f"ax{u}", f"%{p}ao2_{u}", 4)
        vals.append((f"%{p}ax{u}", "vector<4xi8>"))
    return L, vals


def iq3xxs_compute(v, gb):
    """The chained kernel's IQ3_XXS element decode, one row per lane. Element
    lw*4 + b of group g (lw = 2*l + which):
      gword = grid[qs[8g+lw]],  aux = LE32(aux bytes of g)
      sign_nib = (ksigns[(aux >> 7l) & 127] >> 4*which) & 15, s = (sign_nib >> b) & 1
      mag = ((gword >> 8b) & 255 ^ -s) + s
      value = (d * ((f32(aux >> 28) + 0.5) * 0.5)) * f32(mag)"""
    L = []
    e = L.append
    it = iter(v)
    dh = next(it)
    e(f"    %d = scalar.extf {dh} : f16 to f32")
    for u in range(GPL):
        qs = next(it); ax = next(it)
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %axw{u} = vector.bitcast {ax} : vector<4xi8> to vector<1xi32>")
        e(f"    %aux{u} = vector.extract %axw{u}[0] : vector<1xi32> -> i32")
        e(f"    %n4_{u} = scalar.shrui %aux{u}, %c28i : i32")
        e(f"    %n4f_{u} = scalar.sitofp %n4_{u} : i32 to f32")
        e(f"    %hp_{u} = scalar.addf %n4f_{u}, %fhalf : f32")
        e(f"    %hp2_{u} = scalar.mulf %hp_{u}, %fhalf : f32")
        e(f"    %dsc{u} = scalar.mulf %d, %hp2_{u} : f32")
        if VDEC:
            e(f"    %dsc_v8_{u} = vector.splat %dsc{u} : vector<8xf32>")
            _col_of(e, u)
            gws = []
            for lw in range(8):
                t = f"{u}_{lw}"
                e(f"    %qlo8_{t} = vector.extract {qs}[{lw}] : vector<8xi8> -> i8")
                e(f"    %qlo_{t} = scalar.extui %qlo8_{t} : i8 to i32")
                e(f"    %gix_{t} = index.cast %qlo_{t} : i32 to index")
                e(f"    %gil_{t} = index.max %gix_{t}, %c0 : index")
                e(f"    %gid_{t} = index.min %gil_{t}, %c255 : index")
                e(f"    %gw_{t} = view.load %grid_view[%gid_{t}] : view<256xi32> -> i32")
                gws.append(f"%gw_{t}")
            for pp in range(4):
                e(f"    %sid0_{u}_{pp} = scalar.shrui %aux{u}, %c{7 * pp}i : i32")
                e(f"    %sid_{u}_{pp} = scalar.andi %sid0_{u}_{pp}, %c127i : i32")
                e(f"    %sidx_{u}_{pp} = index.cast %sid_{u}_{pp} : i32 to index")
                e(f"    %sidl_{u}_{pp} = index.max %sidx_{u}_{pp}, %c0 : index")
                e(f"    %sidc_{u}_{pp} = index.min %sidl_{u}_{pp}, %c127 : index")
                e(f"    %ks8_{u}_{pp} = view.load %ksigns_view[%sidc_{u}_{pp}] : view<128xi8> -> i8")
                _vdec_pair(e, f"{u}_{pp}", gws[2 * pp], gws[2 * pp + 1], f"%ks8_{u}_{pp}", f"%dsc_v8_{u}", f"%col{u}", u, pp, f"%dsc{u}")
            continue
        hs = []
        for lw in range(8):
            l, which = lw // 2, lw % 2
            t = f"{u}_{lw}"
            e(f"    %qlo8_{t} = vector.extract {qs}[{lw}] : vector<8xi8> -> i8")
            e(f"    %qlo_{t} = scalar.extui %qlo8_{t} : i8 to i32")
            e(f"    %gix_{t} = index.cast %qlo_{t} : i32 to index")
            e(f"    %gil_{t} = index.max %gix_{t}, %c0 : index")
            e(f"    %gid_{t} = index.min %gil_{t}, %c255 : index")
            e(f"    %gw_{t} = view.load %grid_view[%gid_{t}] : view<256xi32> -> i32")
            if which == 0:
                e(f"    %sid0_{u}_{l} = scalar.shrui %aux{u}, %c{7 * l}i : i32")
                e(f"    %sid_{u}_{l} = scalar.andi %sid0_{u}_{l}, %c127i : i32")
                e(f"    %sidx_{u}_{l} = index.cast %sid_{u}_{l} : i32 to index")
                e(f"    %sidl_{u}_{l} = index.max %sidx_{u}_{l}, %c0 : index")
                e(f"    %sidc_{u}_{l} = index.min %sidl_{u}_{l}, %c127 : index")
                e(f"    %ks8_{u}_{l} = view.load %ksigns_view[%sidc_{u}_{l}] : view<128xi8> -> i8")
                e(f"    %sgb_{u}_{l} = scalar.extui %ks8_{u}_{l} : i8 to i32")
            e(f"    %snt_{t} = scalar.shrui %sgb_{u}_{l}, %c{4 * which}i : i32")
            e(f"    %sn_{t} = scalar.andi %snt_{t}, %c15i : i32")
            for b in range(4):
                x = f"{t}_{b}"
                e(f"    %gb0_{x} = scalar.shrui %gw_{t}, %c{8 * b}i : i32")
                e(f"    %gby_{x} = scalar.andi %gb0_{x}, %c255i : i32")
                e(f"    %sb0_{x} = scalar.shrui %sn_{t}, %c{b}i : i32")
                e(f"    %sb_{x} = scalar.andi %sb0_{x}, %c1i : i32")
                e(f"    %ng_{x} = scalar.subi %c0i, %sb_{x} : i32")
                e(f"    %mg0_{x} = scalar.xori %gby_{x}, %ng_{x} : i32")
                e(f"    %mg_{x} = scalar.addi %mg0_{x}, %sb_{x} : i32")
                e(f"    %mf_{x} = scalar.sitofp %mg_{x} : i32 to f32")
                e(f"    %vv_{x} = scalar.mulf %dsc{u}, %mf_{x} : f32")
                e(f"    %hv_{x} = scalar.fptrunc %vv_{x} : f32 to f16")
                hs.append(f"%hv_{x}")
        _store_group(e, u, hs)
    return L


def _stage_table(name, src, n, ty, bytes_per):
    """Copy an n-entry read-only table into workgroup memory once; the first K
    phase's leading barrier publishes it. YAH_SD_GRID_LDS=0 reads the table
    from global memory instead (HIP's __device__ tables; no LDS traffic next to
    the MMA fragment loads)."""
    if not GRID_LDS:
        return [f"  %{name}_view = buffer.view {src}[%base] : buffer -> view<{n}x{ty}>"]
    return [f"  %{name}_g = buffer.view {src}[%base] : buffer -> view<{n}x{ty}>",
            f"  %{name}_bytes = index.constant {n * bytes_per} : offset",
            f"  %{name}_l = buffer.alloca<workgroup> align(16) %{name}_bytes : buffer",
            f"  %{name}_view = buffer.view %{name}_l[%base] : buffer -> view<{n}x{ty}>",
            f"  %{name}_n1 = index.constant {n - 1} : index",
            f"  %{name}_step = index.constant {64 * NW} : index",
            f"  %{name}_cnt = index.constant {n} : index",
            f"  %{name}_sink = scf.for %{name}_i = [%c0 to %{name}_cnt step %{name}_step](%{name}_m = %c0 : index) -> (index) {{",
            f"    %{name}_x0 = index.add %{name}_i, %tid : index",
            f"    %{name}_x = index.min %{name}_x0, %{name}_n1 : index",
            f"    %{name}_v = view.load %{name}_g[%{name}_x] : view<{n}x{ty}> -> {ty}",
            f"    view.store %{name}_v, %{name}_view[%{name}_x] : {ty}, view<{n}x{ty}>",
            f"    scf.yield %{name}_m : index",
            "  }"]


def iq3xxs_setup():
    return ["  %vdw_spread = scalar.constant 2113665 : i32", "  %vdw_ones = scalar.constant 16843009 : i32", "  %vdw_ff = scalar.constant 255 : i32",
            "  %c80x4_iq3 = scalar.constant -2139062144 : i32", "  %cm128f_iq3 = scalar.constant -128.0 : f32",
            "  %c14i_vdw = scalar.constant 14 : i32"] + ((["  %fhalf = scalar.constant 0.5 : f32"])
            + _stage_table("grid", "%grid_na", 256, "i32", 4)
            + _stage_table("ksigns", "%ksigns_na", 128, "i8", 1))


def q4k_loads(p, blk, gb, q5=False):
    """block_q4_K (144 B): d f16 @0, dmin f16 @2, scales[12] @4, qs[128] @16.
    Sub-block g (32 elements) reads qs[32*(g/2) .. +31] (low nibbles for even g,
    high for odd), and scale bytes 4+g, 8+g and g (get_scale_min_k4). A lane's
    groups come in even/odd pairs (gb is even when GPL is), which share qs."""
    # The nibble half is chosen by u's parity, which is g's parity only when a
    # lane's first group is even, i.e. GPL even. At KSUB=64 (GPL=1) odd groups
    # took the low nibbles: argmax 13 at pp2048.
    # GPL=1 is fine here (g>>1 picks the qs pair at run time); q4k_compute then
    # picks the nibble at run time too.
    L = []
    e = L.append
    vals = []
    if Q4_HDR:
        # d, dmin and scales[12] as one 16-byte load (blocks are 16-aligned)
        e(f"    %{p}hd_ix = index.cast {blk} : i32 to index")
        e(f"    %{p}hd_lo = index.max %{p}hd_ix, %c0 : index")
        e(f"    %{p}hd_idx = index.min %{p}hd_lo, %w_lim16 : index")
        e(f"    %{p}hdr = vector.load %w_view[%{p}hd_idx] : view<[%w_bytes]xi8> -> vector<16xi8>")
        vals.append((f"%{p}hdr", "vector<16xi8>"))
    else:
        _ldd(e, p, blk)
        vals.append((f"%{p}dh", "f16"))
    if not Q4_HDR:
        e(f"    %{p}dm_h_i = scalar.addi %{p}d_h_i, %c1i : i32")
    if not Q4_HDR:
        e(f"    %{p}dm_ix = index.cast %{p}dm_h_i : i32 to index")
        e(f"    %{p}dm_lo = index.max %{p}dm_ix, %c0 : index")
        e(f"    %{p}dm_idx = index.min %{p}dm_lo, %w_half_last : index")
    if not Q4_HDR:
        e(f"    %{p}dmh = view.load %w_f16_view[%{p}dm_idx] : view<[%w_halfs]xf16> -> f16")
        vals.append((f"%{p}dmh", "f16"))
    if q5:
        # the 32-byte qh plane (offset 16) is shared by every group of the block
        e(f"    %{p}qh_a = scalar.addi {blk}, %c16i : i32")
        e(f"    %{p}qh_b = scalar.addi {blk}, %c32i : i32")
        _ldv(e, p, "qha_v", f"%{p}qh_a", 16)
        _ldv(e, p, "qhb_v", f"%{p}qh_b", 16)
        vals.append((f"%{p}qha_v", "vector<16xi8>"))
        vals.append((f"%{p}qhb_v", "vector<16xi8>"))
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        if u % 2 == 0:
            e(f"    %{p}gp{u} = scalar.shrui %{p}g{u}, %c1i : i32")
            e(f"    %{p}g32_{u} = scalar.shli %{p}gp{u}, %c5i : i32")
            e(f"    %{p}qo{u} = scalar.addi {blk}, %{p}g32_{u} : i32")
            qsb = 48 if q5 else 16
            e(f"    %{p}qa{u} = scalar.addi %{p}qo{u}, %c{qsb}i : i32")
            e(f"    %{p}qb{u} = scalar.addi %{p}qo{u}, %c{qsb + 16}i : i32")
            _ldv(e, p, f"qa_v{u}", f"%{p}qa{u}", 16)
            _ldv(e, p, f"qb_v{u}", f"%{p}qb{u}", 16)
            vals.append((f"%{p}qa_v{u}", "vector<16xi8>"))
            vals.append((f"%{p}qb_v{u}", "vector<16xi8>"))
        if Q4_HDR:
            continue
        e(f"    %{p}ga{u} = scalar.addi {blk}, %{p}g{u} : i32")
        e(f"    %{p}la_o{u} = scalar.addi %{p}ga{u}, %c4i : i32")
        e(f"    %{p}lb_o{u} = scalar.addi %{p}ga{u}, %c8i : i32")
        _ld8(e, p, f"la{u}", f"%{p}la_o{u}")
        _ld8(e, p, f"lb{u}", f"%{p}lb_o{u}")
        _ld8(e, p, f"lc{u}", f"%{p}ga{u}")
        vals += [(f"%{p}la{u}", "i8"), (f"%{p}lb{u}", "i8"), (f"%{p}lc{u}", "i8")]
    return L, vals


def q4k_compute(v, gb, q5=False):
    """The chained kernel's Q4_K element decode, one row per lane:
      value = (d * f32(sc)) * f32(q) - dmin * f32(m)"""
    L = []
    e = L.append
    it = iter(v)
    if Q4FMIX:
        # an opaque 1.0 (see Q4FMIX): gb & ~gb is 0, unprovable to the folder
        e(f"    %q4nb = scalar.xori {gb}, %q4m1 : i32")
        e(f"    %q4z = scalar.andi {gb}, %q4nb : i32")
        e("    %q4ob = scalar.ori %q4z, %q4one_b : i32")
        e("    %q4one = scalar.bitcast %q4ob : i32 to f32")
    if Q4_HDR:
        hdr = next(it)
        e(f"    %hdw = vector.bitcast {hdr} : vector<16xi8> to vector<4xi32>")
        for w in range(4):
            e(f"    %hdw{w} = vector.extract %hdw[{w}] : vector<4xi32> -> i32")
        e("    %hdd16 = scalar.trunci %hdw0 : i32 to i16")
        e("    %hdm0 = scalar.shrui %hdw0, %c16i_q : i32")
        e("    %hdm16 = scalar.trunci %hdm0 : i32 to i16")
        e("    %hddf = scalar.bitcast %hdd16 : i16 to f16")
        e("    %hdmf = scalar.bitcast %hdm16 : i16 to f16")
        dh, dmh = "%hddf", "%hdmf"
    else:
        dh = next(it); dmh = next(it)
    if q5:
        qha = next(it); qhb = next(it)
    e(f"    %d = scalar.extf {dh} : f16 to f32")
    e(f"    %dmin = scalar.extf {dmh} : f16 to f32")
    for u in range(GPL):
        if u % 2 == 0:
            qa = next(it); qb = next(it)
        if not Q4_HDR:
            la = next(it); lb = next(it); lc = next(it)
        e(f"    %g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        if Q4_HDR:
            # header byte k = 4 + g (la), 8 + g (lb), g (lc): word k/4, byte k%4;
            # g < 8, so each picks between two adjacent words on g/4
            e(f"    %hq{u} = scalar.shrui %g{u}, %c2i : i32")
            e(f"    %hq1_{u} = scalar.cmpi eq, %hq{u}, %c1i : i32")
            e(f"    %hr{u} = scalar.andi %g{u}, %c3i : i32")
            e(f"    %hs{u} = scalar.shli %hr{u}, %c3i : i32")
            for nm, w0 in (("la", 1), ("lb", 2), ("lc", 0)):
                e(f"    %{nm}w{u} = scf.select %hq1_{u}, %hdw{w0 + 1}, %hdw{w0} : i32")
                e(f"    %{nm}s{u} = scalar.shrui %{nm}w{u}, %hs{u} : i32")
                e(f"    %{nm}_{u} = scalar.andi %{nm}s{u}, %c255i_q : i32")
        else:
            e(f"    %la_{u} = scalar.extui {la} : i8 to i32")
            e(f"    %lb_{u} = scalar.extui {lb} : i8 to i32")
            e(f"    %lc_{u} = scalar.extui {lc} : i8 to i32")
        e(f"    %slo{u} = scalar.cmpi slt, %g{u}, %c4i : i32")
        e(f"    %sc{u}, %mn{u} = scf.if %slo{u} -> (i32, i32) {{")
        e(f"      %s_a{u} = scalar.andi %la_{u}, %c63i : i32")
        e(f"      %m_a{u} = scalar.andi %lb_{u}, %c63i : i32")
        e(f"      scf.yield %s_a{u}, %m_a{u} : i32, i32")
        e("    } else {")
        e(f"      %lbl{u} = scalar.andi %lb_{u}, %c15i : i32")
        e(f"      %lch{u} = scalar.shrui %lc_{u}, %c6i : i32")
        e(f"      %lch4{u} = scalar.shli %lch{u}, %c4i : i32")
        e(f"      %s_b{u} = scalar.ori %lbl{u}, %lch4{u} : i32")
        e(f"      %lbh{u} = scalar.shrui %lb_{u}, %c4i : i32")
        e(f"      %lah{u} = scalar.shrui %la_{u}, %c6i : i32")
        e(f"      %lah4{u} = scalar.shli %lah{u}, %c4i : i32")
        e(f"      %m_b{u} = scalar.ori %lbh{u}, %lah4{u} : i32")
        e(f"      scf.yield %s_b{u}, %m_b{u} : i32, i32")
        e("    }")
        e(f"    %scf{u} = scalar.sitofp %sc{u} : i32 to f32")
        e(f"    %mf{u} = scalar.sitofp %mn{u} : i32 to f32")
        e(f"    %dsc{u} = scalar.mulf %d, %scf{u} : f32")
        e(f"    %dm{u} = scalar.mulf %dmin, %mf{u} : f32")
        e(f"    %dsc_v{u} = vector.splat %dsc{u} : vector<16xf32>")
        e(f"    %dm_v{u} = vector.splat %dm{u} : vector<16xf32>")
        op = "vector.andi" if u % 2 == 0 else "vector.shrui"
        k = "%m15v" if u % 2 == 0 else "%s4v"
        rt = GPL % 2 == 1
        if rt:
            # one group per lane: its parity is only known at run time, so the
            # nibble is (q >> 4*(g & 1)) & 15 -- the same value as the even
            # (q & 15) and odd (q >> 4) forms, so bit-identical
            e(f"    %gpar{u} = scalar.andi %g{u}, %c1i : i32")
            e(f"    %gsh{u} = scalar.shli %gpar{u}, %c2i : i32")
            e(f"    %gsh8_{u} = scalar.trunci %gsh{u} : i32 to i8")
            e(f"    %gshv{u} = vector.splat %gsh8_{u} : vector<16xi8>")
        if q5:
            # fifth bit: quant = nibble + ((qh[lane] >> g) & 1) * 16
            e(f"    %g8_{u} = scalar.trunci %g{u} : i32 to i8")
            e(f"    %g8v_{u} = vector.splat %g8_{u} : vector<16xi8>")
        if Q4_UITOFP:
            # nibbles on whole 32-bit words, (w >> s) & 0x0f0f0f0f, as HIP
            # does: per-byte i8 shifts and masks lower element by element
            if rt:
                e(f"    %gshw{u} = vector.splat %gsh{u} : vector<4xi32>")
            else:
                e(f"    %gshw{u} = vector.splat %q4sh{0 if u % 2 == 0 else 4} : vector<4xi32>")
        for half, q, qh in (("lo", qa, qha if q5 else None), ("hi", qb, qhb if q5 else None)):
            if Q4_UITOFP:
                e(f"    %qw{half}{u} = vector.bitcast {q} : vector<16xi8> to vector<4xi32>")
                e(f"    %qws{half}{u} = vector.shrui %qw{half}{u}, %gshw{u} : vector<4xi32>")
                e(f"    %qwm{half}{u} = vector.andi %qws{half}{u}, %m0f4 : vector<4xi32>")
                e(f"    %nq{half}{u} = vector.bitcast %qwm{half}{u} : vector<4xi32> to vector<16xi8>")
            elif rt:
                e(f"    %nqs{half}{u} = vector.shrui {q}, %gshv{u} : vector<16xi8>")
                e(f"    %nq{half}{u} = vector.andi %nqs{half}{u}, %m15v : vector<16xi8>")
            else:
                e(f"    %nq{half}{u} = {op} {q}, {k} : vector<16xi8>")
            src = f"%nq{half}{u}"
            if q5:
                if Q4_UITOFP:
                    # the fifth bit on words too: ((qh >> g) & 0x01010101) << 4,
                    # OR the nibbles (the bits do not overlap, so it is the add)
                    e(f"    %hw{half}{u} = vector.bitcast {qh} : vector<16xi8> to vector<4xi32>")
                    e(f"    %hgw{half}{u} = vector.splat %g{u} : vector<4xi32>")
                    e(f"    %hs{half}{u} = vector.shrui %hw{half}{u}, %hgw{half}{u} : vector<4xi32>")
                    e(f"    %hb{half}{u} = vector.andi %hs{half}{u}, %m014 : vector<4xi32>")
                    e(f"    %h16{half}{u} = vector.shli %hb{half}{u}, %s44 : vector<4xi32>")
                    e(f"    %n5w{half}{u} = vector.ori %qwm{half}{u}, %h16{half}{u} : vector<4xi32>")
                    e(f"    %n5{half}{u} = vector.bitcast %n5w{half}{u} : vector<4xi32> to vector<16xi8>")
                else:
                    e(f"    %hs{half}{u} = vector.shrui {qh}, %g8v_{u} : vector<16xi8>")
                    e(f"    %hb{half}{u} = vector.andi %hs{half}{u}, %one8v : vector<16xi8>")
                    e(f"    %h16{half}{u} = vector.shli %hb{half}{u}, %s4v : vector<16xi8>")
                    e(f"    %n5{half}{u} = vector.addi %nq{half}{u}, %h16{half}{u} : vector<16xi8>")
                src = f"%n5{half}{u}"
            # nibbles are 0..15 (0..31 with q5's high bit): uitofp is the same
            # value and can select v_cvt_f32_ubyteN (no sign-extend)
            e(f"    %fq{half}{u} = vector.{'uitofp' if Q4_UITOFP else 'sitofp'} {src} : vector<16xi8> to vector<16xf32>")
            e(f"    %sq{half}{u} = vector.mulf %dsc_v{u}, %fq{half}{u} : vector<16xf32>")
            if Q4FMIX:
                if half == "lo":
                    e(f"    %ndm{u} = scalar.negf %dm{u} : f32")
                hs = []
                for j in range(16):
                    t = f"{half}{u}_{j}"
                    e(f"    %qe{t} = vector.extract %sq{half}{u}[{j}] : vector<16xf32> -> f32")
                    e(f"    %qm{t} = scalar.fmaf %qe{t}, %q4one, %ndm{u} : f32")
                    e(f"    %qt{t} = scalar.fptrunc %qm{t} : f32 to f16")
                    hs.append(f"%qt{t}")
                e(f"    %h{half}{u} = vector.from_elements {', '.join(hs)} : vector<16xf16>")
            else:
                e(f"    %vq{half}{u} = vector.subf %sq{half}{u}, %dm_v{u} : vector<16xf32>")
                e(f"    %h{half}{u} = vector.fptrunc %vq{half}{u} : vector<16xf32> to vector<16xf16>")
        e(f"    %col_i{u} = scalar.shli %gl{u}, %c5i : i32")
        e(f"    %col_x{u} = index.cast %col_i{u} : i32 to index")
        e(f"    %col_l{u} = index.max %col_x{u}, %c0 : index")
        e(f"    %col{u} = index.min %col_l{u}, %ccolmax : index")
        e(f"    %colh{u} = index.add %col{u}, %c16 : index")
        e(f"    vector.store %hlo{u}, %wl_view[%drow, %col{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
        e(f"    vector.store %hhi{u}, %wl_view[%drow, %colh{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
    return L


def q4k_setup():
    return ["  %c15b = scalar.constant 15 : i8", "  %c4b = scalar.constant 4 : i8", "  %c1b = scalar.constant 1 : i8",
            "  %m15v = vector.splat %c15b : vector<16xi8>", "  %s4v = vector.splat %c4b : vector<16xi8>",
            "  %one8v = vector.splat %c1b : vector<16xi8>",
            "  %c0f4 = scalar.constant 252645135 : i32", "  %m0f4 = vector.splat %c0f4 : vector<4xi32>",
            "  %q4sh0 = scalar.constant 0 : i32", "  %q4sh4 = scalar.constant 4 : i32",
            "  %c16i_q = scalar.constant 16 : i32", "  %c255i_q = scalar.constant 255 : i32",
            "  %c014 = scalar.constant 16843009 : i32", "  %m014 = vector.splat %c014 : vector<4xi32>",
            "  %s44 = vector.splat %q4sh4 : vector<4xi32>",
            "  %q4m1 = scalar.constant -1 : i32", "  %q4one_b = scalar.constant 1065353216 : i32"]


def iq2xxs_loads(p, blk, gb):
    """block_iq2_xxs (66 B): d f16 @0, then per 32-element group g eight bytes at
    2 + 8g: four grid codes (bytes 0..3) and one LE32 aux word (bytes 4..7)."""
    L = []
    e = L.append
    vals = []
    _ldd(e, p, blk)
    vals.append((f"%{p}dh", "f16"))
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}g8_{u} = scalar.shli %{p}g{u}, %c3i : i32")
        e(f"    %{p}qo{u} = scalar.addi {blk}, %{p}g8_{u} : i32")
        e(f"    %{p}qo2_{u} = scalar.addi %{p}qo{u}, %c2i : i32")
        _ldv(e, p, f"qs{u}", f"%{p}qo2_{u}", 8)
        vals.append((f"%{p}qs{u}", "vector<8xi8>"))
    return L, vals


def iq2xxs_compute(v, gb):
    """The chained kernel's IQ2_XXS decode, 8 elements per grid code li:
      gw = grid words (2*code, 2*code+1), s = bits of ksigns[(aux >> 7li) & 127]
      mag = (g ^ -s) + s, value = (d * ((f32(aux >> 28) + 0.5) * 0.25)) * f32(mag)"""
    L = []
    e = L.append
    it = iter(v)
    dh = next(it)
    e(f"    %d = scalar.extf {dh} : f16 to f32")
    for u in range(GPL):
        qs = next(it)
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %qw{u} = vector.bitcast {qs} : vector<8xi8> to vector<2xi32>")
        e(f"    %aux{u} = vector.extract %qw{u}[1] : vector<2xi32> -> i32")
        e(f"    %n4_{u} = scalar.shrui %aux{u}, %c28i : i32")
        e(f"    %n4f_{u} = scalar.sitofp %n4_{u} : i32 to f32")
        e(f"    %hp_{u} = scalar.addf %n4f_{u}, %fhalf : f32")
        e(f"    %hp2_{u} = scalar.mulf %hp_{u}, %fquarter : f32")
        e(f"    %dsc{u} = scalar.mulf %d, %hp2_{u} : f32")
        e(f"    %dsc_v8_{u} = vector.splat %dsc{u} : vector<8xf32>")
        _col_of(e, u)
        for li in range(4):
            t = f"{u}_{li}"
            e(f"    %cd8_{t} = vector.extract {qs}[{li}] : vector<8xi8> -> i8")
            e(f"    %cd_{t} = scalar.extui %cd8_{t} : i8 to i32")
            e(f"    %w0_{t} = scalar.shli %cd_{t}, %c1i : i32")
            e(f"    %w1_{t} = scalar.addi %w0_{t}, %c1i : i32")
            for w in (0, 1):
                e(f"    %wx{w}_{t} = index.cast %w{w}_{t} : i32 to index")
                e(f"    %wl{w}_{t} = index.max %wx{w}_{t}, %c0 : index")
                e(f"    %wc{w}_{t} = index.min %wl{w}_{t}, %c511 : index")
                e(f"    %gw{w}_{t} = view.load %grid_view[%wc{w}_{t}] : view<512xi32> -> i32")
            e(f"    %sid0_{t} = scalar.shrui %aux{u}, %c{7 * li}i : i32")
            e(f"    %sid_{t} = scalar.andi %sid0_{t}, %c127i : i32")
            e(f"    %sidx_{t} = index.cast %sid_{t} : i32 to index")
            e(f"    %sidl_{t} = index.max %sidx_{t}, %c0 : index")
            e(f"    %sidc_{t} = index.min %sidl_{t}, %c127 : index")
            e(f"    %ks8_{t} = view.load %ksigns_view[%sidc_{t}] : view<128xi8> -> i8")
            _vdec_pair(e, t, f"%gw0_{t}", f"%gw1_{t}", f"%ks8_{t}", f"%dsc_v8_{u}", f"%col{u}", u, li)
    return L


def iq2xs_loads(p, blk, gb):
    """block_iq2_xs (74 B): d f16 @0, qs[32] u16 @2, scales[8] @66. Group g
    (32 elements) reads the four codes qs[4g..4g+3] (bytes 2 + 8g) and scale
    byte 66 + g."""
    L = []
    e = L.append
    vals = []
    _ldd(e, p, blk)
    vals.append((f"%{p}dh", "f16"))
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}g8_{u} = scalar.shli %{p}g{u}, %c3i : i32")
        e(f"    %{p}qo{u} = scalar.addi {blk}, %{p}g8_{u} : i32")
        e(f"    %{p}qo2_{u} = scalar.addi %{p}qo{u}, %c2i : i32")
        _ldv(e, p, f"qs{u}", f"%{p}qo2_{u}", 8)
        vals.append((f"%{p}qs{u}", "vector<8xi8>"))
        e(f"    %{p}so{u} = scalar.addi {blk}, %{p}g{u} : i32")
        e(f"    %{p}so2_{u} = scalar.addi %{p}so{u}, %c66i : i32")
        _ld8(e, p, f"sc{u}", f"%{p}so2_{u}")
        vals.append((f"%{p}sc{u}", "i8"))
    return L, vals


def iq2xs_compute(v, gb):
    """The chained kernel's IQ2_XS decode (yah_ffn_gemm_iq2xs_f32.loom), 8
    elements per code l (elements 8l..8l+7 of the group):
      gw = grid words (2*(code & 511), +1), s = bits of ksigns[code >> 9],
      mag = (g ^ -s) + s, nib = scales[g] low nibble for l < 2, high for l >= 2,
      value = (d * ((f32(nib) + 0.5) * 0.25)) * f32(mag)"""
    L = []
    e = L.append
    it = iter(v)
    dh = next(it)
    e(f"    %d = scalar.extf {dh} : f16 to f32")
    for u in range(GPL):
        qs = next(it); sc = next(it)
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %qw{u} = vector.bitcast {qs} : vector<8xi8> to vector<2xi32>")
        e(f"    %scb{u} = scalar.extui {sc} : i8 to i32")
        for hv in (0, 1):
            if hv:
                e(f"    %nb{hv}_{u}0 = scalar.shrui %scb{u}, %c4i : i32")
            else:
                e(f"    %nb{hv}_{u}0 = scalar.andi %scb{u}, %c15i : i32")
            e(f"    %nbf{hv}_{u} = scalar.sitofp %nb{hv}_{u}0 : i32 to f32")
            e(f"    %hp{hv}_{u} = scalar.addf %nbf{hv}_{u}, %fhalf : f32")
            e(f"    %hq{hv}_{u} = scalar.mulf %hp{hv}_{u}, %fquarter : f32")
            e(f"    %dsc{hv}_{u} = scalar.mulf %d, %hq{hv}_{u} : f32")
            e(f"    %dsc_v8_{hv}_{u} = vector.splat %dsc{hv}_{u} : vector<8xf32>")
        _col_of(e, u)
        for l in range(4):
            t = f"{u}_{l}"
            e(f"    %cw_{t} = vector.extract %qw{u}[{l // 2}] : vector<2xi32> -> i32")
            if l % 2:
                e(f"    %cd_{t} = scalar.shrui %cw_{t}, %c16i_2 : i32")
            else:
                e(f"    %cd_{t} = scalar.andi %cw_{t}, %c65535i_2 : i32")
            e(f"    %gi_{t} = scalar.andi %cd_{t}, %c511i_2 : i32")
            e(f"    %w0_{t} = scalar.shli %gi_{t}, %c1i : i32")
            e(f"    %w1_{t} = scalar.addi %w0_{t}, %c1i : i32")
            for w in (0, 1):
                e(f"    %wx{w}_{t} = index.cast %w{w}_{t} : i32 to index")
                e(f"    %wl{w}_{t} = index.max %wx{w}_{t}, %c0 : index")
                e(f"    %wc{w}_{t} = index.min %wl{w}_{t}, %c1023 : index")
                e(f"    %gw{w}_{t} = view.load %grid_view[%wc{w}_{t}] : view<1024xi32> -> i32")
            e(f"    %sid0_{t} = scalar.shrui %cd_{t}, %c9i_2 : i32")
            e(f"    %sid_{t} = scalar.andi %sid0_{t}, %c127i : i32")
            e(f"    %sidx_{t} = index.cast %sid_{t} : i32 to index")
            e(f"    %sidl_{t} = index.max %sidx_{t}, %c0 : index")
            e(f"    %sidc_{t} = index.min %sidl_{t}, %c127 : index")
            e(f"    %ks8_{t} = view.load %ksigns_view[%sidc_{t}] : view<128xi8> -> i8")
            _vdec_pair(e, t, f"%gw0_{t}", f"%gw1_{t}", f"%ks8_{t}", f"%dsc_v8_{l // 2}_{u}", f"%col{u}", u, l)
    return L


def iq2xs_setup():
    return (["  %fhalf = scalar.constant 0.5 : f32", "  %fquarter = scalar.constant 0.25 : f32",
             "  %c1023 = index.constant 1023 : index",
             "  %c16i_2 = scalar.constant 16 : i32", "  %c65535i_2 = scalar.constant 65535 : i32",
             "  %c511i_2 = scalar.constant 511 : i32", "  %c9i_2 = scalar.constant 9 : i32",
             "  %vdw_spread = scalar.constant 2113665 : i32", "  %vdw_ones = scalar.constant 16843009 : i32",
             "  %vdw_ff = scalar.constant 255 : i32"]
            + _stage_table("grid", "%grid_na", 1024, "i32", 4)
            + _stage_table("ksigns", "%ksigns_na", 128, "i8", 1))


def iq2xxs_setup():
    return (["  %fhalf = scalar.constant 0.5 : f32", "  %fquarter = scalar.constant 0.25 : f32",
             "  %c511 = index.constant 511 : index"]
            + _stage_table("grid", "%grid_na", 512, "i32", 4)
            + _stage_table("ksigns", "%ksigns_na", 128, "i8", 1))


def q6k_loads(p, blk, gb):
    """block_q6_K (210 B): ql[128] @0, qh[64] @128, int8 scales[16] @192, d @208.
    Group g = 4*half + seg reads 32 ql bytes at 64*half + 32*(seg&1), 32 qh bytes
    at 128 + 32*half, and the two scales 192 + 8*half + 2*seg (+1)."""
    L = []
    e = L.append
    vals = []
    e(f"    %{p}d_h0 = scalar.shrui {blk}, %c1i : i32")
    e(f"    %{p}d_h_i = scalar.addi %{p}d_h0, %c104i : i32")
    e(f"    %{p}d_ix = index.cast %{p}d_h_i : i32 to index")
    e(f"    %{p}d_lo = index.max %{p}d_ix, %c0 : index")
    e(f"    %{p}d_idx = index.min %{p}d_lo, %w_half_last : index")
    e(f"    %{p}dh = view.load %w_f16_view[%{p}d_idx] : view<[%w_halfs]xf16> -> f16")
    vals.append((f"%{p}dh", "f16"))
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}hf{u} = scalar.shrui %{p}g{u}, %c2i : i32")
        e(f"    %{p}sg{u} = scalar.andi %{p}g{u}, %c3i : i32")
        e(f"    %{p}so{u} = scalar.andi %{p}sg{u}, %c1i : i32")
        e(f"    %{p}h64_{u} = scalar.shli %{p}hf{u}, %c6i : i32")
        e(f"    %{p}s32_{u} = scalar.shli %{p}so{u}, %c5i : i32")
        e(f"    %{p}qlo{u} = scalar.addi %{p}h64_{u}, %{p}s32_{u} : i32")
        e(f"    %{p}qla{u} = scalar.addi {blk}, %{p}qlo{u} : i32")
        e(f"    %{p}qlb{u} = scalar.addi %{p}qla{u}, %c16i : i32")
        _ldv(e, p, f"qla_v{u}", f"%{p}qla{u}", 16)
        _ldv(e, p, f"qlb_v{u}", f"%{p}qlb{u}", 16)
        e(f"    %{p}h32_{u} = scalar.shli %{p}hf{u}, %c5i : i32")
        e(f"    %{p}qh0_{u} = scalar.addi {blk}, %{p}h32_{u} : i32")
        e(f"    %{p}qha{u} = scalar.addi %{p}qh0_{u}, %c128i : i32")
        e(f"    %{p}qhb{u} = scalar.addi %{p}qha{u}, %c16i : i32")
        _ldv(e, p, f"qha_v{u}", f"%{p}qha{u}", 16)
        _ldv(e, p, f"qhb_v{u}", f"%{p}qhb{u}", 16)
        e(f"    %{p}h8_{u} = scalar.shli %{p}hf{u}, %c3i : i32")
        e(f"    %{p}s2_{u} = scalar.shli %{p}sg{u}, %c1i : i32")
        e(f"    %{p}sc0_{u} = scalar.addi %{p}h8_{u}, %{p}s2_{u} : i32")
        e(f"    %{p}sca{u} = scalar.addi {blk}, %{p}sc0_{u} : i32")
        e(f"    %{p}sca2_{u} = scalar.addi %{p}sca{u}, %c192i : i32")
        e(f"    %{p}scb2_{u} = scalar.addi %{p}sca2_{u}, %c1i : i32")
        _ld8(e, p, f"sa{u}", f"%{p}sca2_{u}")
        _ld8(e, p, f"sb{u}", f"%{p}scb2_{u}")
        vals += [(f"%{p}qla_v{u}", "vector<16xi8>"), (f"%{p}qlb_v{u}", "vector<16xi8>"),
                 (f"%{p}qha_v{u}", "vector<16xi8>"), (f"%{p}qhb_v{u}", "vector<16xi8>"),
                 (f"%{p}sa{u}", "i8"), (f"%{p}sb{u}", "i8")]
    return L, vals


def q6k_compute(v, gb):
    """The chained kernel's Q6_K element decode, one row per lane:
      code = low4 | ((qh >> 2*seg) & 3) << 4
      value = (d * f32(int8 scale)) * f32(code - 32)"""
    L = []
    e = L.append
    it = iter(v)
    dh = next(it)
    e(f"    %d = scalar.extf {dh} : f16 to f32")
    for u in range(GPL):
        qla = next(it); qlb = next(it); qha = next(it); qhb = next(it); sa = next(it); sb = next(it)
        e(f"    %g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %sg{u} = scalar.andi %g{u}, %c3i : i32")
        e(f"    %hi{u} = scalar.shrui %sg{u}, %c1i : i32")
        e(f"    %qls{u} = scalar.shli %hi{u}, %c2i : i32")
        e(f"    %qhs{u} = scalar.shli %sg{u}, %c1i : i32")
        e(f"    %qls8_{u} = scalar.trunci %qls{u} : i32 to i8")
        e(f"    %qhs8_{u} = scalar.trunci %qhs{u} : i32 to i8")
        e(f"    %qlsv{u} = vector.splat %qls8_{u} : vector<16xi8>")
        e(f"    %qhsv{u} = vector.splat %qhs8_{u} : vector<16xi8>")
        _col_of(e, u)
        e(f"    %colh{u} = index.add %col{u}, %c16 : index")
        for half, ql, qh, sc, col in (("lo", qla, qha, sa, f"%col{u}"), ("hi", qlb, qhb, sb, f"%colh{u}")):
            t = f"{half}{u}"
            e(f"    %scf_{t} = scalar.sitofp {sc} : i8 to f32")
            e(f"    %dsc_{t} = scalar.mulf %d, %scf_{t} : f32")
            e(f"    %dsv_{t} = vector.splat %dsc_{t} : vector<16xf32>")
            e(f"    %l0_{t} = vector.shrui {ql}, %qlsv{u} : vector<16xi8>")
            e(f"    %l_{t} = vector.andi %l0_{t}, %m15v : vector<16xi8>")
            e(f"    %h0_{t} = vector.shrui {qh}, %qhsv{u} : vector<16xi8>")
            e(f"    %h1_{t} = vector.andi %h0_{t}, %m3v : vector<16xi8>")
            e(f"    %h4_{t} = vector.shli %h1_{t}, %s4v : vector<16xi8>")
            e(f"    %cd_{t} = vector.ori %l_{t}, %h4_{t} : vector<16xi8>")
            e(f"    %bs_{t} = vector.subi %cd_{t}, %m32v : vector<16xi8>")
            e(f"    %bf_{t} = vector.sitofp %bs_{t} : vector<16xi8> to vector<16xf32>")
            e(f"    %vv_{t} = vector.mulf %dsv_{t}, %bf_{t} : vector<16xf32>")
            e(f"    %hv_{t} = vector.fptrunc %vv_{t} : vector<16xf32> to vector<16xf16>")
            e(f"    vector.store %hv_{t}, %wl_view[%drow, {col}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
    return L


def q6k_setup():
    return ["  %c15b = scalar.constant 15 : i8", "  %c4b = scalar.constant 4 : i8",
            "  %c3b = scalar.constant 3 : i8", "  %c32b = scalar.constant 32 : i8",
            "  %m15v = vector.splat %c15b : vector<16xi8>", "  %s4v = vector.splat %c4b : vector<16xi8>",
            "  %m3v = vector.splat %c3b : vector<16xi8>", "  %m32v = vector.splat %c32b : vector<16xi8>"]


def q8_0_loads(p, blk, gb):
    """Q8_0 in 256-element super-blocks of 8 blocks (272 B): group g is block g,
    d f16 at 34g, qs int8[32] at 34g + 2."""
    L = []
    e = L.append
    vals = []
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}g34_{u} = scalar.muli %{p}g{u}, %c34i : i32")
        e(f"    %{p}bo{u} = scalar.addi {blk}, %{p}g34_{u} : i32")
        e(f"    %{p}dh_i{u} = scalar.shrui %{p}bo{u}, %c1i : i32")
        e(f"    %{p}dx{u} = index.cast %{p}dh_i{u} : i32 to index")
        e(f"    %{p}dl{u} = index.max %{p}dx{u}, %c0 : index")
        e(f"    %{p}dc{u} = index.min %{p}dl{u}, %w_half_last : index")
        e(f"    %{p}dh{u} = view.load %w_f16_view[%{p}dc{u}] : view<[%w_halfs]xf16> -> f16")
        e(f"    %{p}qa{u} = scalar.addi %{p}bo{u}, %c2i : i32")
        e(f"    %{p}qb{u} = scalar.addi %{p}bo{u}, %c18i : i32")
        _ldv(e, p, f"qa_v{u}", f"%{p}qa{u}", 16)
        _ldv(e, p, f"qb_v{u}", f"%{p}qb{u}", 16)
        vals += [(f"%{p}dh{u}", "f16"), (f"%{p}qa_v{u}", "vector<16xi8>"), (f"%{p}qb_v{u}", "vector<16xi8>")]
    return L, vals


def q8_0_compute(v, gb):
    """The chained kernel's Q8_0 decode: value = d * f32(q)."""
    L = []
    e = L.append
    it = iter(v)
    for u in range(GPL):
        dh = next(it); qa = next(it); qb = next(it)
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %d{u} = scalar.extf {dh} : f16 to f32")
        e(f"    %dv{u} = vector.splat %d{u} : vector<16xf32>")
        _col_of(e, u)
        e(f"    %colh{u} = index.add %col{u}, %c16 : index")
        for half, q, col in (("lo", qa, f"%col{u}"), ("hi", qb, f"%colh{u}")):
            t = f"{half}{u}"
            e(f"    %qf_{t} = vector.sitofp {q} : vector<16xi8> to vector<16xf32>")
            e(f"    %vv_{t} = vector.mulf %dv{u}, %qf_{t} : vector<16xf32>")
            e(f"    %hv_{t} = vector.fptrunc %vv_{t} : vector<16xf32> to vector<16xf16>")
            e(f"    vector.store %hv_{t}, %wl_view[%drow, {col}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
    return L


def iq4xs_setup():
    L = [f"  %kv{i} = scalar.constant {v} : i8" for i, v in enumerate(IQ4_KVALUES)]
    L += ["  %c15b_iq = scalar.constant 15 : i8", "  %c4b_iq = scalar.constant 4 : i8",
          "  %c16i_h = scalar.constant 16 : i32", "  %c255i_h = scalar.constant 255 : i32",
          "  %c0f4_iq = scalar.constant 252645135 : i32", "  %m0f4_iq = vector.splat %c0f4_iq : vector<4xi32>",
          "  %c4w_iq = scalar.constant 4 : i32", "  %s4w_iq = vector.splat %c4w_iq : vector<4xi32>"]
    L.append("  %kvt = vector.from_elements " + ", ".join(f"%kv{i}" for i in range(16)) + " : vector<16xi8>")
    if IQ4_U8:
        L += [f"  %kvu{i} = scalar.constant {(v + 128) - 256 if v + 128 > 127 else v + 128} : i8" for i, v in enumerate(IQ4_KVALUES)]
        L.append("  %kvtu = vector.from_elements " + ", ".join(f"%kvu{i}" for i in range(16)) + " : vector<16xi8>")
        L += ["  %c128f_iq = scalar.constant 128.0 : f32", "  %c128v_iq = vector.splat %c128f_iq : vector<16xf32>",
              "  %cm128f_iq = scalar.constant -128.0 : f32"]
    if IQ4_F16:
        bits = [struct.unpack("<H", struct.pack("<e", float(v)))[0] for v in IQ4_KVALUES]
        for nm, part in (("l", [b & 255 for b in bits]), ("h", [b >> 8 for b in bits])):
            L += [f"  %kvb{nm}{i} = scalar.constant {x - 256 if x > 127 else x} : i8" for i, x in enumerate(part)]
            L.append(f"  %kvt{nm} = vector.from_elements " + ", ".join(f"%kvb{nm}{i}" for i in range(16)) + " : vector<16xi8>")
        L += ["  %c8w = scalar.constant 8 : i16", "  %s8w = vector.splat %c8w : vector<16xi16>"]
    L += ["  %c15b = scalar.constant 15 : i8", "  %c4b = scalar.constant 4 : i8",
          "  %m15v = vector.splat %c15b : vector<16xi8>", "  %s4v = vector.splat %c4b : vector<16xi8>"]
    return L


def q3k_loads(p, blk, gb):
    """block_q3_K (110 B): hmask[32] @0, qs[64] @32, scales[12] @96, d f16 @108.
    Group g (32 elements) of the block: half = g/4, sp = g%4; element h16*16 + j
    reads qs[32*half + 16*h16 + j] (bits 2sp..2sp+1) and hmask[16*h16 + j] (bit
    g); its two 16-element halves use scales si = 2g and 2g+1."""
    L = []
    e = L.append
    vals = []
    e(f"    %{p}dq_h = scalar.shrui {blk}, %c1i : i32")
    e(f"    %{p}dq_i = scalar.addi %{p}dq_h, %q3c54i : i32")
    e(f"    %{p}dq_ix = index.cast %{p}dq_i : i32 to index")
    e(f"    %{p}dq_lo = index.max %{p}dq_ix, %c0 : index")
    e(f"    %{p}dq_idx = index.min %{p}dq_lo, %w_half_last : index")
    if Q3_W:
        # scales[12] @96 and d @108 as one 16-byte load at @94 (inside the block)
        e(f"    %{p}q3h_o = scalar.addi {blk}, %q3c94i : i32")
        _ldv(e, p, "q3hdr", f"%{p}q3h_o", 16)
        vals.append((f"%{p}q3hdr", "vector<16xi8>"))
    else:
        e(f"    %{p}dh = view.load %w_f16_view[%{p}dq_idx] : view<[%w_halfs]xf16> -> f16")
        vals.append((f"%{p}dh", "f16"))
    e(f"    %{p}hm_b = scalar.addi {blk}, %c16i : i32")
    _ldv(e, p, "hma", blk, 16)
    _ldv(e, p, "hmb", f"%{p}hm_b", 16)
    vals += [(f"%{p}hma", "vector<16xi8>"), (f"%{p}hmb", "vector<16xi8>")]
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}hf{u} = scalar.shrui %{p}g{u}, %c2i : i32")
        e(f"    %{p}hf32_{u} = scalar.shli %{p}hf{u}, %c5i : i32")
        e(f"    %{p}qo{u} = scalar.addi {blk}, %{p}hf32_{u} : i32")
        e(f"    %{p}qa_o{u} = scalar.addi %{p}qo{u}, %c32i : i32")
        e(f"    %{p}qb_o{u} = scalar.addi %{p}qo{u}, %c48i : i32")
        _ldv(e, p, f"qa{u}", f"%{p}qa_o{u}", 16)
        _ldv(e, p, f"qb{u}", f"%{p}qb_o{u}", 16)
        vals += [(f"%{p}qa{u}", "vector<16xi8>"), (f"%{p}qb{u}", "vector<16xi8>")]
        # low-nibble scale bytes scales[2*(g&3)] and +1, high bytes scales[8+2*(g&1)] and +1
        e(f"    %{p}g3_{u} = scalar.andi %{p}g{u}, %c3i : i32")
        e(f"    %{p}g3x2_{u} = scalar.shli %{p}g3_{u}, %c1i : i32")
        e(f"    %{p}sl0_{u} = scalar.addi {blk}, %q3c96i : i32")
        e(f"    %{p}sla_o{u} = scalar.addi %{p}sl0_{u}, %{p}g3x2_{u} : i32")
        e(f"    %{p}slb_o{u} = scalar.addi %{p}sla_o{u}, %c1i : i32")
        e(f"    %{p}g1_{u} = scalar.andi %{p}g{u}, %c1i : i32")
        e(f"    %{p}g1x2_{u} = scalar.shli %{p}g1_{u}, %c1i : i32")
        e(f"    %{p}sh0_{u} = scalar.addi {blk}, %c104i : i32")
        e(f"    %{p}sha_o{u} = scalar.addi %{p}sh0_{u}, %{p}g1x2_{u} : i32")
        e(f"    %{p}shb_o{u} = scalar.addi %{p}sha_o{u}, %c1i : i32")
        if not Q3_W:
            _ld8(e, p, f"la{u}", f"%{p}sla_o{u}")
            _ld8(e, p, f"lb{u}", f"%{p}slb_o{u}")
            _ld8(e, p, f"ha{u}", f"%{p}sha_o{u}")
            _ld8(e, p, f"hb{u}", f"%{p}shb_o{u}")
        if not Q3_W:
            vals += [(f"%{p}la{u}", "i8"), (f"%{p}lb{u}", "i8"), (f"%{p}ha{u}", "i8"), (f"%{p}hb{u}", "i8")]
    return L, vals


def q3k_compute(v, gb):
    """The chained kernel's Q3_K element decode (yah_ffn_gemm_q3k_f32.loom):
      low = (qs >> 2sp) & 3, bit = (hmask >> g) & 1, quant = (low | bit<<2) - 4
      low4 = (scales[si&7] >> 4*(si>>3)) & 15, high2 = (scales[8+si%4] >> 2*(si>>2)) & 3
      scale = (low4 | high2<<4) - 32, value = (f32(d) * f32(scale)) * f32(quant)
    The integer steps are exact, so doing them 16 lanes of a vector at a time
    gives the chained kernel's values; the f32 products keep its order."""
    L = []
    e = L.append
    it = iter(v)
    if Q3_W:
        hdr = next(it); hma = next(it); hmb = next(it)
        # window bytes 94..109: word i = bytes 94+4i..97+4i; d = word 3 >> 16
        e(f"    %q3w = vector.bitcast {hdr} : vector<16xi8> to vector<4xi32>")
        for w in range(4):
            e(f"    %q3w{w} = vector.extract %q3w[{w}] : vector<4xi32> -> i32")
        e("    %q3dw = scalar.shrui %q3w3, %q3c16i : i32")
        e("    %q3d16 = scalar.trunci %q3dw : i32 to i16")
        e("    %q3dh = scalar.bitcast %q3d16 : i16 to f16")
        e("    %d = scalar.extf %q3dh : f16 to f32")
    else:
        dh = next(it); hma = next(it); hmb = next(it)
        e(f"    %d = scalar.extf {dh} : f16 to f32")
    for u in range(GPL):
        qa = next(it); qb = next(it)
        if not Q3_W:
            la = next(it); lb = next(it); ha = next(it); hb = next(it)
        e(f"    %g{u} = scalar.addi {gb}, %c{u}i : i32")
        if Q3_W:
            # scales[2*(g&3)], +1 at window bytes 2+2*(g&3); scales[8+2*(g&1)],
            # +1 at 10+2*(g&1): 16-bit pairs from words 0..3
            e(f"    %q3g3_{u} = scalar.andi %g{u}, %c3i : i32")
            e(f"    %q3g1_{u} = scalar.andi %g{u}, %c1i : i32")
            e(f"    %q3z_{u} = scalar.cmpi eq, %q3g3_{u}, %c0i : i32")
            e(f"    %q3t_{u} = scalar.cmpi eq, %q3g3_{u}, %c3i : i32")
            e(f"    %q3wa0_{u} = scf.select %q3t_{u}, %q3w2, %q3w1 : i32")
            e(f"    %q3wa_{u} = scf.select %q3z_{u}, %q3w0, %q3wa0_{u} : i32")
            e(f"    %q3odd_{u} = scalar.cmpi eq, %q3g1_{u}, %c1i : i32")
            e(f"    %q3wah_{u} = scalar.shrui %q3wa_{u}, %q3c16i : i32")
            e(f"    %q3pa_{u} = scf.select %q3odd_{u}, %q3wa_{u}, %q3wah_{u} : i32")
            e(f"    %q3wb_{u} = scf.select %q3odd_{u}, %q3w3, %q3w2 : i32")
            e(f"    %q3wbh_{u} = scalar.shrui %q3wb_{u}, %q3c16i : i32")
            e(f"    %q3pb_{u} = scf.select %q3odd_{u}, %q3wb_{u}, %q3wbh_{u} : i32")
            e(f"    %q3la_{u} = scalar.andi %q3pa_{u}, %c255i_3 : i32")
            e(f"    %q3lb0_{u} = scalar.shrui %q3pa_{u}, %c8i : i32")
            e(f"    %q3lb_{u} = scalar.andi %q3lb0_{u}, %c255i_3 : i32")
            e(f"    %q3ha_{u} = scalar.andi %q3pb_{u}, %c255i_3 : i32")
            e(f"    %q3hb0_{u} = scalar.shrui %q3pb_{u}, %c8i : i32")
            e(f"    %q3hb_{u} = scalar.andi %q3hb0_{u}, %c255i_3 : i32")
            la, lb, ha, hb = f"%q3la_{u}", f"%q3lb_{u}", f"%q3ha_{u}", f"%q3hb_{u}"
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %q3sp{u} = scalar.andi %g{u}, %c3i : i32")
        e(f"    %q3ls{u} = scalar.shli %q3sp{u}, %c1i : i32")
        e(f"    %q3ls8_{u} = scalar.trunci %q3ls{u} : i32 to i8")
        e(f"    %q3lsv{u} = vector.splat %q3ls8_{u} : vector<16xi8>")
        e(f"    %q3bs8_{u} = scalar.trunci %g{u} : i32 to i8")
        e(f"    %q3bsv{u} = vector.splat %q3bs8_{u} : vector<16xi8>")
        if Q3_W:
            e(f"    %q3lsw{u} = vector.splat %q3ls{u} : vector<4xi32>")
            e(f"    %q3bsw{u} = vector.splat %g{u} : vector<4xi32>")
        e(f"    %q3hf{u} = scalar.shrui %g{u}, %c2i : i32")
        e(f"    %q3s4{u} = scalar.shli %q3hf{u}, %c2i : i32")
        e(f"    %q3g2{u} = scalar.shrui %g{u}, %c1i : i32")
        e(f"    %q3s2{u} = scalar.shli %q3g2{u}, %c1i : i32")
        e(f"    %col_i{u} = scalar.shli %gl{u}, %c5i : i32")
        e(f"    %col_x{u} = index.cast %col_i{u} : i32 to index")
        e(f"    %col_l{u} = index.max %col_x{u}, %c0 : index")
        e(f"    %col{u} = index.min %col_l{u}, %ccolmax : index")
        e(f"    %colh{u} = index.add %col{u}, %c16 : index")
        for hn, q, hm, lo8, hi8, col in (("lo", qa, hma, la, ha, f"%col{u}"), ("hi", qb, hmb, lb, hb, f"%colh{u}")):
            t = f"{hn}{u}"
            if Q3_W:
                # on 32-bit words: per byte (q >> 2sp) & 3 | ((hm >> g) & 1) << 2,
                # minus 4 as (x | 0x80) - 4 ^ 0x80 (no borrow leaves a byte)
                e(f"    %q3qw{t} = vector.bitcast {q} : vector<16xi8> to vector<4xi32>")
                e(f"    %q3hw{t} = vector.bitcast {hm} : vector<16xi8> to vector<4xi32>")
                e(f"    %q3lw{t} = vector.shrui %q3qw{t}, %q3lsw{u} : vector<4xi32>")
                e(f"    %q3low{t} = vector.andi %q3lw{t}, %q3m3w : vector<4xi32>")
                e(f"    %q3bw{t} = vector.shrui %q3hw{t}, %q3bsw{u} : vector<4xi32>")
                e(f"    %q3bit{t} = vector.andi %q3bw{t}, %q3m1w : vector<4xi32>")
                e(f"    %q3b2{t} = vector.shli %q3bit{t}, %q3s2w : vector<4xi32>")
                e(f"    %q3lb{t} = vector.ori %q3low{t}, %q3b2{t} : vector<4xi32>")
                e(f"    %q3o8{t} = vector.ori %q3lb{t}, %q3m80w : vector<4xi32>")
                e(f"    %q3s4w{t} = vector.subi %q3o8{t}, %q3m4w : vector<4xi32>")
                e(f"    %q3x8{t} = vector.xori %q3s4w{t}, %q3m80w : vector<4xi32>")
                e(f"    %q3qn{t} = vector.bitcast %q3x8{t} : vector<4xi32> to vector<16xi8>")
            else:
                e(f"    %q3lw{t} = vector.shrui {q}, %q3lsv{u} : vector<16xi8>")
                e(f"    %q3low{t} = vector.andi %q3lw{t}, %q3m3v : vector<16xi8>")
                e(f"    %q3bw{t} = vector.shrui {hm}, %q3bsv{u} : vector<16xi8>")
                e(f"    %q3bit{t} = vector.andi %q3bw{t}, %q3m1v : vector<16xi8>")
                e(f"    %q3b2{t} = vector.shli %q3bit{t}, %q3s2v : vector<16xi8>")
                e(f"    %q3lb{t} = vector.ori %q3low{t}, %q3b2{t} : vector<16xi8>")
                e(f"    %q3qn{t} = vector.subi %q3lb{t}, %q3f4v : vector<16xi8>")
            e(f"    %q3qf{t} = vector.sitofp %q3qn{t} : vector<16xi8> to vector<16xf32>")
            if Q3_W:
                e(f"    %q3l8{t} = scalar.addi {lo8}, %c0i : i32")
            else:
                e(f"    %q3l8{t} = scalar.extui {lo8} : i8 to i32")
            e(f"    %q3l4s{t} = scalar.shrui %q3l8{t}, %q3s4{u} : i32")
            e(f"    %q3l4{t} = scalar.andi %q3l4s{t}, %c15i : i32")
            if Q3_W:
                e(f"    %q3h8{t} = scalar.addi {hi8}, %c0i : i32")
            else:
                e(f"    %q3h8{t} = scalar.extui {hi8} : i8 to i32")
            e(f"    %q3h2s{t} = scalar.shrui %q3h8{t}, %q3s2{u} : i32")
            e(f"    %q3h2{t} = scalar.andi %q3h2s{t}, %c3i : i32")
            e(f"    %q3h4{t} = scalar.shli %q3h2{t}, %c4i : i32")
            e(f"    %q3s6{t} = scalar.ori %q3l4{t}, %q3h4{t} : i32")
            e(f"    %q3sc{t} = scalar.subi %q3s6{t}, %c32i : i32")
            e(f"    %q3scf{t} = scalar.sitofp %q3sc{t} : i32 to f32")
            e(f"    %q3dsc{t} = scalar.mulf %d, %q3scf{t} : f32")
            e(f"    %q3dv{t} = vector.splat %q3dsc{t} : vector<16xf32>")
            e(f"    %q3v{t} = vector.mulf %q3dv{t}, %q3qf{t} : vector<16xf32>")
            e(f"    %h{hn}{u} = vector.fptrunc %q3v{t} : vector<16xf32> to vector<16xf16>")
        e(f"    vector.store %hlo{u}, %wl_view[%drow, %col{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
        e(f"    vector.store %hhi{u}, %wl_view[%drow, %colh{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
    return L


def q3k_setup():
    return ["  %q3c54i = scalar.constant 54 : i32", "  %q3c96i = scalar.constant 96 : i32",
            "  %q3c3b = scalar.constant 3 : i8", "  %q3c1b = scalar.constant 1 : i8",
            "  %q3c2b = scalar.constant 2 : i8", "  %q3c4b = scalar.constant 4 : i8",
            "  %q3m3v = vector.splat %q3c3b : vector<16xi8>", "  %q3m1v = vector.splat %q3c1b : vector<16xi8>",
            "  %q3s2v = vector.splat %q3c2b : vector<16xi8>", "  %q3f4v = vector.splat %q3c4b : vector<16xi8>",
            "  %q3c94i = scalar.constant 94 : i32", "  %q3c16i = scalar.constant 16 : i32",
            "  %c255i_3 = scalar.constant 255 : i32",
            "  %q3k3 = scalar.constant 50529027 : i32", "  %q3m3w = vector.splat %q3k3 : vector<4xi32>",
            "  %q3k1 = scalar.constant 16843009 : i32", "  %q3m1w = vector.splat %q3k1 : vector<4xi32>",
            "  %q3k2 = scalar.constant 2 : i32", "  %q3s2w = vector.splat %q3k2 : vector<4xi32>",
            "  %q3k80 = scalar.constant -2139062144 : i32", "  %q3m80w = vector.splat %q3k80 : vector<4xi32>",
            "  %q3k4 = scalar.constant 67372036 : i32", "  %q3m4w = vector.splat %q3k4 : vector<4xi32>"]


FMTS = {
    # fmt: block bytes, (loads, compute), extra buffer bindings after %weight, setup
    # ksub: measured best phase width at pp2048 (mean ms per dispatch, NW=2):
    #   iq4xs 128 -> 9.71 (64: 12.08, 256: 12.53)
    #   iq3s   64 -> 12.57 (128: 15.47) -- at 128 the 64-element decode per lane
    #          pushes 8 accumulators into scratch (3151 private ops/work-item)
    "iq4xs": dict(bb=136, ksub=128, decode=(iq4xs_loads, iq4xs_compute), extra=[], setup=iq4xs_setup),
    "iq3s": dict(bb=110, ksub=64, decode=(iq3s_loads, iq3s_compute), extra=["grid"], setup=iq3s_setup),
    "q4k": dict(bb=144, ksub=128, decode=(q4k_loads, q4k_compute), extra=[], setup=q4k_setup),
    # Q5_K: Q4_K plus a fifth bit from the qh plane; qs at 48 instead of 16
    "q5k": dict(bb=176, ksub=128, decode=(lambda p, b, g: q4k_loads(p, b, g, q5=True),
                                          lambda v, g: q4k_compute(v, g, q5=True)),
                extra=[], setup=q4k_setup),
    "q8_0": dict(bb=272, kdiv=8, ksub=64, decode=(q8_0_loads, q8_0_compute), extra=[], setup=lambda: []),
    "q6k": dict(bb=210, ksub=64, decode=(q6k_loads, q6k_compute), extra=[], setup=q6k_setup),
    "q3k": dict(bb=110, ksub=64, decode=(q3k_loads, q3k_compute), extra=[], setup=q3k_setup),
    "iq2xxs": dict(bb=66, ksub=64, decode=(iq2xxs_loads, iq2xxs_compute), extra=["grid", "ksigns"], setup=iq2xxs_setup),
    "iq2xs": dict(bb=74, ksub=64, decode=(iq2xs_loads, iq2xs_compute), extra=["grid", "ksigns"], setup=iq2xs_setup),
    "iq3xxs": dict(bb=98, ksub=64, decode=(iq3xxs_loads, iq3xxs_compute), extra=["grid", "ksigns"], setup=iq3xxs_setup),
}


def emit_db_loop(e, L, loads, compute, toks, types, V4):
    """Double-buffered K loop: one barrier per phase, decode overlapping MMAs.

    The weight tile is two LR-row halves. Iteration kp (0..kphases) decodes phase
    min(kp, last) into half kp&1 from bytes loaded one iteration earlier, issues
    the loads for phase kp+1, then -- for kp >= 1 -- runs phase kp-1's MMAs from
    the other half, then barriers. The barrier at the end of kp-1 is what makes
    both the half being decoded (last read in kp-1) and the half being read
    (written in kp-1) safe. Each accumulator sees the same MMA sequence as the
    single-buffered loop, so the result is bit-identical to it."""
    ca = ", ".join(f"%a{i} = %init : {V4}" for i in range(NA))
    L0, vals0 = loads("pf_", "%row_off_i", "%gl_i")
    L.extend(L0)
    orig0 = vals0
    vals0 = pack_vals(e, vals0, "0")
    ca += ", " + ", ".join(f"%cv{x} = {nm} : {ty}" for x, (nm, ty) in enumerate(vals0))
    carried_t = types + ", " + ", ".join(ty for _, ty in vals0)
    res = ", ".join(f"%acc{i}" for i in range(NA)) + ", " + ", ".join(f"%cvout{x}" for x in range(len(vals0)))
    # publish workgroup tables staged by the format setup (IQ grids/ksigns): the
    # single-buffered loop's leading barrier did this, but here the first decode
    # runs before the first barrier (IQ3_S read an unpublished grid: md5 b6dc21d0)
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e("  %kph1 = index.add %kphases, %c1 : index")
    e(f"  %clr_db = index.constant {LR} : index")
    e("  " + res + f" = scf.for %kp = [%c0 to %kph1 step %c1]({ca}) -> ({carried_t}) {{")
    e("    %kp_last = index.sub %kphases, %c1 : index")
    e("    %kpd = index.min %kp, %kp_last : index")
    e("    %kb = index.div %kpd, %cph : index")
    e("    %ph = index.rem %kpd, %cph : index")
    e("    %kb_i = index.cast %kb : index to i32")
    e("    %ph_i = index.cast %ph : index to i32")
    e("    %blk_off0 = scalar.muli %kb_i, %cbbi : i32")
    e("    %phg_i = scalar.muli %ph_i, %cgppi : i32")
    e("    %gb_i = scalar.addi %phg_i, %gl_i : i32")
    e("    %blk_i = scalar.addi %row_off_i, %blk_off0 : i32")
    e("    %bsel = index.rem %kp, %c2 : index")
    e("    %boff = index.mul %bsel, %clr_db : index")
    e("    %drowb = index.add %drow, %boff : index")
    cur = [(f"%cv{x}", ty) for x, (_, ty) in enumerate(vals0)]
    cur_names = unpack_vals(e, cur, orig0)
    dec = compute(cur_names, "%gb_i")
    vt_old, vt_new = f"view<{LR}x{ROWP}xf16>", f"view<{2 * LR}x{ROWP}xf16>"
    L.extend(l.replace("%wl_view[%drow,", "%wl_view[%drowb,").replace(vt_old, vt_new) for l in dec)
    # loads for the next phase (clamped: the extra iterations' loads are in bounds)
    e("    %kp_n0 = index.add %kp, %c1 : index")
    e("    %kp_n = index.min %kp_n0, %kp_last : index")
    e("    %kb_n = index.div %kp_n, %cph : index")
    e("    %ph_n = index.rem %kp_n, %cph : index")
    e("    %kb_ni = index.cast %kb_n : index to i32")
    e("    %ph_ni = index.cast %ph_n : index to i32")
    e("    %blk_off0n = scalar.muli %kb_ni, %cbbi : i32")
    e("    %blk_n = scalar.addi %row_off_i, %blk_off0n : i32")
    e("    %phg_n = scalar.muli %ph_ni, %cgppi : i32")
    e("    %gb_n = scalar.addi %phg_n, %gl_i : i32")
    Ln, nxt = loads("nx_", "%blk_n", "%gb_n")
    L.extend(Ln)
    nxt = pack_vals(e, nxt, "n")
    # MMAs of phase kp-1 from the other half
    e("    %has_mma = index.cmp ne, %kp, %c0 : index")
    e("    %kp1 = index.max %kp, %c1 : index")
    e("    %kpm = index.sub %kp1, %c1 : index")
    e("    %kb_k = index.mul %kpm, %cksub : index")
    e("    %bm = index.sub %c1, %bsel : index")
    e("    %moff = index.mul %bm, %clr_db : index")
    res2 = ", ".join(f"%r{i}" for i in range(NA))
    e(f"    {res2} = scf.if %has_mma -> ({types}) {{")
    cb = ", ".join(f"%b{i} = %a{i} : {V4}" for i in range(NA))
    e("      " + ", ".join(f"%q{i}" for i in range(NA)) + f" = scf.for %ks = [%c0 to %cksub step %c16]({cb}) -> ({types}) {KPOL}{{")
    e("        %kk = index.add %kb_k, %ks : index")
    for i in range(MT):
        e(f"        %lr0_{i} = index.add %rg64, %c{16 * i} : index")
        e(f"        %lr{i} = index.add %lr0_{i}, %moff : index")
        e(f"        %lhs{i} = vector.fragment.load<lhs> %wl_view[%lr{i}, %ks] shape [%m, %k] : {vt_new} -> vector<16xf16>")
    for j in range(NT):
        e(f"        %rhs{j} = vector.fragment.load<rhs> %a_t_view[%kk, {toks[j]}] shape [%k, %n] : view<[%ktot]x[%tokens]xf16, %a_layout> -> vector<16xf16>")
    for i in range(MT):
        for j in range(NT):
            n = i * NT + j
            e(f"        %n{n} = vector.mma %lhs{i}, %rhs{j}, %b{n} : vector<16xf16>, vector<16xf16>, {V4}")
    e("        scf.yield " + ", ".join(f"%n{i}" for i in range(NA)) + f" : {types}")
    e("      }")
    e("      scf.yield " + ", ".join(f"%q{i}" for i in range(NA)) + f" : {types}")
    e("    } else {")
    e("      scf.yield " + ", ".join(f"%a{i}" for i in range(NA)) + f" : {types}")
    e("    }")
    e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e("    scf.yield " + res2 + ", " + ", ".join(nm for nm, _ in nxt) + f" : {carried_t}")
    e("  }")


def gen(fmt, kind="kstore"):
    """kind: "kstore" (f32 token-major output) or "swiglu" (the ffn_up arm:
    f16 output = round_f16(silu(gate) * acc), gate the f32 gate projection in
    the same token-major layout -- yah_ffn_gemm_<fmt>_swiglu_f16.loom's epilogue)."""
    F = FMTS[fmt]
    configure(fmt)
    bb, decode = F["bb"], F["decode"]
    sw = kind == "swiglu"
    kr = kind == "kres"
    bufs = (["weight"] + F["extra"] + ["input"] + (["gate"] if sw else []) + (["resid"] if kr else [])
            + ["wstage", "ostage", "output"])
    sym = f"yah_ffn_gemm_{fmt}" + ("_swiglu" if sw else "") + ("_kres" if kr else "")
    wgs = 64 * NW * NR
    wtok = TOK * NW
    L = []
    e = L.append
    e(f"// GENERATED by tools/gen_gemm_shared.py {fmt} (NW={NW}, PAD={PAD}) -- edit the generator.")
    e("//")
    e(f"// Shared-decode kStore GEMM for {fmt}: {NW} wave64 waves share one decoded 64x256")
    e("// weight tile per K block; each wave accumulates 64 rows x 128 tokens. Same ABI,")
    e("// output layout and arithmetic order as the chained kernel; see the generator.")
    e("amdgpu.target<gfx1151> @yah_gemm_w64 {subgroup_size = 64}")
    e("")
    for c in ("m_tiles", "k_blocks", "token_tiles"):
        e(f"config.decl @{sym}.{c} : %value: index where [range(%value, 1, 4096)]")
    e("")
    e(f"kernel.def target(@yah_gemm_w64) @{sym}() {{")
    e("  %unit = index.constant 1 : index")
    e(f"  %m_tiles = config.get @{sym}.m_tiles : index")
    e(f"  %token_tiles = config.get @{sym}.token_tiles : index")
    e(f"  %wgs = index.constant {wgs} : index")
    e(f"  %rowgrp = index.constant {MT * NR} : index")
    e("  %m_groups = index.div %m_tiles, %rowgrp : index")
    e("  kernel.launch.config workgroups(%m_groups, %token_tiles, %unit) workgroup_size(%wgs, %unit, %unit) : index")
    e("} launch(" + ", ".join(f"%{b}: buffer" for b in bufs) + ") {")
    e("  %base = index.constant 0 : offset")
    for v in (0, 1, 2, 4, 6, 7, 8, 16, 32, 48, 63, 64, 80, 96, 112, 127, 128, 224, 255, 256, 512):
        e(f"  %c{v} = index.constant {v} : index")
    for v in (0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 18, 21, 24, 28, 32, 34, 48, 63, 64, 66, 74, 104, 106, 127, 128, 192, 255):
        e(f"  %c{v}i = scalar.constant {v} : i32")
    e(f"  %cbb = index.constant {bb} : index")
    e(f"  %cbbh = index.constant {bb // 2} : index")
    e(f"  %cbbi = scalar.constant {bb} : i32")
    e(f"  %cwtok = index.constant {wtok} : index")
    e(f"  %cksub = index.constant {KSUB} : index")
    e(f"  %cph = index.constant {PH} : index")
    e(f"  %ccolmax = index.constant {KSUB - 32} : index")
    e(f"  %cgppi = scalar.constant {GPP} : i32")
    e("  %m = index.constant 16 : index")
    e("  %n = index.constant 16 : index")
    e("  %k = index.constant 16 : index")
    e(f"  %m_tiles = config.get @{sym}.m_tiles : index")
    kdiv = F.get("kdiv", 1)
    if kdiv == 1:
        e(f"  %k_blocks = config.get @{sym}.k_blocks : index")
    else:
        # the config counts the format's own blocks (32 elements for Q8_0); the
        # kernel works in 256-element super-blocks of kdiv of them. Without this
        # the Q8_0 kernel walked 8x past its weights and hung the gfx ring.
        e(f"  %k_blocks_cfg = config.get @{sym}.k_blocks : index")
        e(f"  %ckdiv = index.constant {kdiv} : index")
        e("  %k_blocks = index.div %k_blocks_cfg, %ckdiv : index")
    e(f"  %token_tiles = config.get @{sym}.token_tiles : index")
    e("  %ktot = index.mul %k_blocks, %c256 : index")
    e("  %tokens = index.mul %token_tiles, %cwtok : index")
    e("  %m_rows = index.mul %m_tiles, %c16 : index")
    e("  %bpr = index.mul %k_blocks, %cbb : index")
    e("  %hpr = index.mul %k_blocks, %cbbh : index")
    e("  %w_bytes = index.mul %m_rows, %bpr : index")
    e("  %w_halfs = index.mul %m_rows, %hpr : index")
    e("  %w_last = index.sub %w_bytes, %c1 : index")
    e("  %w_lim = index.sub %w_bytes, %c16 : index")
    for n in (4, 8, 16):
        e(f"  %cw{n} = index.constant {n} : index")
        e(f"  %w_lim{n} = index.sub %w_bytes, %cw{n} : index")
    e("  %w_half_last = index.sub %w_halfs, %c1 : index")
    e("  %out_total = index.mul %m_rows, %tokens : index")
    e("  %stage_rows = index.mul %m_tiles, %c16 : index")
    e("  %stage_last = index.sub %stage_rows, %c1 : index")
    e("  %a_layout = encoding.layout.strided [%c1, %ktot] : encoding<layout>")
    e("  " + ", ".join(f"%{b}_na" for b in bufs) + " = buffer.assume.noalias "
      + ", ".join(f"%{b}" for b in bufs) + " : " + ", ".join(["buffer"] * len(bufs)))
    e("  %w_view = buffer.view %weight_na[%base] : buffer -> view<[%w_bytes]xi8>")
    e("  %w_f16_view = buffer.view %weight_na[%base] : buffer -> view<[%w_halfs]xf16>")
    e("  %a_t_view = buffer.view %input_na[%base] : buffer -> view<[%ktot]x[%tokens]xf16, %a_layout>")
    if not sw:
        # the swiglu output is f16: an f32 view of it would declare twice its size
        e("  %out_view = buffer.view %output_na[%base] : buffer -> view<[%out_total]xf32>")
    e("  %ostage_view = buffer.view %ostage_na[%base] : buffer -> view<[%stage_rows]x[%tokens]xf32>")
    wl_bytes = LR * ROWP * 2 * (2 if DB else 1)
    if sw:
        wl_bytes = max(wl_bytes, NW * NR * 16 * TOK * 4)   # the epilogue's f32 slabs
    e(f"  %wl_bytes = index.constant {wl_bytes} : offset")
    e("  %wl = buffer.alloca<workgroup> align(16) %wl_bytes : buffer")
    e(f"  %wl_view = buffer.view %wl[%base] : buffer -> view<{LR * (2 if DB else 1)}x{ROWP}xf16>")
    e("  %wg_x = kernel.workgroup.id<x> : index")
    e("  %wg_y = kernel.workgroup.id<y> : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %wave = index.div %tid, %c64 : index")
    e("  %l64 = index.rem %tid, %c64 : index")
    if NR == 1:
        e(f"  %clr0 = index.constant {LR} : index")
        e("  %m_origin = index.mul %wg_x, %clr0 : index")
        e("  %wv = index.add %wave, %c0 : index")
        e("  %rg64 = index.add %c0, %c0 : index")
    else:
        # wave = rg * NW + wv: row group rg, token slice wv
        e(f"  %cnw = index.constant {NW} : index")
        e(f"  %clr = index.constant {LR} : index")
        e("  %rg = index.div %wave, %cnw : index")
        e("  %wv = index.rem %wave, %cnw : index")
        e("  %rg64 = index.mul %rg, %c64 : index")
        e("  %wg_row = index.mul %wg_x, %clr : index")
        e("  %m_origin = index.add %wg_row, %rg64 : index")
    e("  %wtb = index.mul %wg_y, %cwtok : index")
    e(f"  %ctok = index.constant {TOK} : index")
    e("  %wave_tok = index.mul %wv, %ctok : index")
    e("  %token_base = index.add %wtb, %wave_tok : index")
    # decode lane map: lane l64 of row group rg owns tile row 64*rg + l64 (global
    # row m_origin + l64); token slice wv owns groups [wv*GPL, wv*GPL+GPL)
    e(f"  %clr1 = index.constant {LR - 1} : index")
    e("  %drow0 = index.add %l64, %rg64 : index")
    e("  %drow = index.min %drow0, %clr1 : index")
    e("  %drow_i = index.cast %drow : index to i32")
    e("  %m_origin_i = index.cast %m_origin : index to i32")
    if NR == 1:
        # the decode row is clamped to the tile, so lanes past a short tile (MT < 4)
        # re-decode its last row instead of reading past the matrix
        e("  %grow_i = scalar.addi %m_origin_i, %drow_i : i32")
    else:
        e("  %l64_d = index.cast %l64 : index to i32")
        e("  %grow_i = scalar.addi %m_origin_i, %l64_d : i32")
    e("  %k_blocks_i = index.cast %k_blocks : index to i32")
    e("  %bpr_i = scalar.muli %k_blocks_i, %cbbi : i32")
    e("  %row_off_i = scalar.muli %grow_i, %bpr_i : i32")
    e("  %wave_i = index.cast %wv : index to i32")
    e(f"  %cgpl = scalar.constant {GPL} : i32")
    e("  %gl_i = scalar.muli %wave_i, %cgpl : i32")
    e("  %kphases = index.mul %k_blocks, %cph : index")
    L.extend(F["setup"]())
    e("  %z8s = scalar.constant 0 : i8")
    e("  %z8v = vector.splat %z8s : vector<8xi8>")
    e("  %zeros = vector.constant 0.0 : vector<4xf32>")
    e("  %init = vector.fragment<init> %zeros shape [%m, %n] : vector<4xf32>")
    for j in range(1, NT):
        e(f"  %t{16 * j} = index.add %token_base, %c{16 * j} : index")
    toks = ["%token_base"] + [f"%t{16 * j}" for j in range(1, NT)]
    V4 = "vector<4xf32>"
    types = ", ".join([V4] * NA)
    loads, compute = decode
    if DB:
        assert PREFETCH and ABLATE == "" and KORDER == ""
        emit_db_loop(e, L, loads, compute, toks, types, V4)
    else:
        ca = ", ".join(f"%a{i} = %init : {V4}" for i in range(NA))
        carried_t = types
        if PREFETCH:
            # Phase 0's raw bytes are loaded before the loop; each iteration decodes
            # the carried bytes and then issues the NEXT phase's loads, so their DRAM
            # latency runs under this phase's MMAs instead of in front of the decode.
            L0, vals0 = loads("pf_", "%row_off_i", "%gl_i")
            L.extend(L0)
            orig0 = vals0
            vals0 = pack_vals(e, vals0, "0")
            ca += ", " + ", ".join(f"%cv{x} = {nm} : {ty}" for x, (nm, ty) in enumerate(vals0))
            carried_t = types + ", " + ", ".join(ty for _, ty in vals0)
        res = ", ".join(f"%acc{i}" for i in range(NA))
        if PREFETCH:
            res += ", " + ", ".join(f"%cvout{x}" for x in range(len(vals0)))
        e("  " + res + f" = scf.for %kp = [%c0 to %kphases step %c1]({ca}) -> ({carried_t}) {{")
        e("    // every wave has finished reading the previous phase's tile")
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        e("    %kb = index.div %kp, %cph : index")
        e("    %ph = index.rem %kp, %cph : index")
        e("    %kb_i = index.cast %kb : index to i32")
        e("    %ph_i = index.cast %ph : index to i32")
        e("    %blk_off0 = scalar.muli %kb_i, %cbbi : i32")
        e("    %phg_i = scalar.muli %ph_i, %cgppi : i32")
        e("    %gb_i = scalar.addi %phg_i, %gl_i : i32")
        e("    %kb_k = index.mul %kp, %cksub : index")
        e("    %blk_i = scalar.addi %row_off_i, %blk_off0 : i32")
        if PREFETCH:
            cur = [(f"%cv{x}", ty) for x, (_, ty) in enumerate(vals0)]
            cur_names = unpack_vals(e, cur, orig0)
        else:
            Lc, cur = loads("cu_", "%blk_i", "%gb_i")
            L.extend(Lc)
            cur_names = [nm for nm, _ in cur]
        if ABLATE != "decode":
            L.extend(compute(cur_names, "%gb_i"))
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        if PREFETCH:
            # next phase, clamped to the last one (its loads are then redundant but
            # in bounds, and the carried values are never decoded)
            e("    %kp_n0 = index.add %kp, %c1 : index")
            e("    %kp_last = index.sub %kphases, %c1 : index")
            e("    %kp_n = index.min %kp_n0, %kp_last : index")
            e("    %kb_n = index.div %kp_n, %cph : index")
            e("    %ph_n = index.rem %kp_n, %cph : index")
            e("    %kb_ni = index.cast %kb_n : index to i32")
            e("    %ph_ni = index.cast %ph_n : index to i32")
            e("    %blk_off0n = scalar.muli %kb_ni, %cbbi : i32")
            e("    %blk_n = scalar.addi %row_off_i, %blk_off0n : i32")
            e("    %phg_n = scalar.muli %ph_ni, %cgppi : i32")
            e("    %gb_n = scalar.addi %phg_n, %gl_i : i32")
            Ln, nxt = loads("nx_", "%blk_n", "%gb_n")
            L.extend(Ln)
            nxt = pack_vals(e, nxt, "n")
        cb = ", ".join(f"%b{i} = %a{i} : {V4}" for i in range(NA))
        if RPF:
            # rhs software pipeline: this step's activation fragments were loaded
            # one K step earlier and are carried in; the next step's are issued
            # before this step's MMAs (clamped on the last step: a redundant,
            # in-bounds load). The ATT trace put 56% of the kernel in this loop,
            # nearly all of it waiting on the step's own rhs global loads.
            VF = "vector<16xf16>"
            for j in range(NT):
                e(f"    %rp0_{j} = vector.fragment.load<rhs> %a_t_view[%kb_k, {toks[j]}] shape [%k, %n] : view<[%ktot]x[%tokens]xf16, %a_layout> -> {VF}")
            cb += ", " + ", ".join(f"%rc{j} = %rp0_{j} : {VF}" for j in range(NT))
            rtypes = types + ", " + ", ".join([VF] * NT)
            e("    %ks_last = index.sub %cksub, %c16 : index")
            e("    " + ", ".join(f"%r{i}" for i in range(NA)) + ", " + ", ".join(f"%rco{j}" for j in range(NT))
              + f" = scf.for %ks = [%c0 to %cksub step %c16]({cb}) -> ({rtypes}) {KPOL}{{")
            e("      %ks_n0 = index.add %ks, %c16 : index")
            e("      %ks_n = index.min %ks_n0, %ks_last : index")
            e("      %kk_n = index.add %kb_k, %ks_n : index")
        else:
            e("    " + ", ".join(f"%r{i}" for i in range(NA)) + f" = scf.for %ks = [%c0 to %cksub step %c16]({cb}) -> ({types}) {KPOL}{{")
        e("      %kk = index.add %kb_k, %ks : index")
        for i in range(MT):
            e(f"      %lr{i} = index.add %rg64, %c{16 * i} : index")
            e(f"      %lhs{i} = vector.fragment.load<lhs> %wl_view[%lr{i}, %ks] shape [%m, %k] : view<{LR}x{ROWP}xf16> -> vector<16xf16>")
        for j in range(NT):
            if ABLATE == "rhs":
                e(f"      %rhs{j} = vector.fragment.load<rhs> %wl_view[%c{16 * (j % 4)}, %ks] shape [%k, %n] : view<{LR}x{ROWP}xf16> -> vector<16xf16>")
                continue
            if ABLATE == "rhsfix":
                # probe: same rhs addresses every K step (always cache-resident)
                e(f"      %rhs{j} = vector.fragment.load<rhs> %a_t_view[%c0, {toks[j]}] shape [%k, %n] : view<[%ktot]x[%tokens]xf16, %a_layout> -> vector<16xf16>")
                continue
            if RPF:
                e(f"      %rhs{j} = vector.fragment.load<rhs> %a_t_view[%kk_n, {toks[j]}] shape [%k, %n] : view<[%ktot]x[%tokens]xf16, %a_layout> -> vector<16xf16>")
                continue
            e(f"      %rhs{j} = vector.fragment.load<rhs> %a_t_view[%kk, {toks[j]}] shape [%k, %n] : view<[%ktot]x[%tokens]xf16, %a_layout> -> vector<16xf16>")
        if KORDER == "fence" and ABLATE == "":
            # rhs-major with fences: rhs_{j+1} is issued, then rhs_j's four MMAs,
            # then a fence, so at most two activation fragments (plus the four lhs)
            # are live instead of all eight. Each accumulator still sees exactly one
            # MMA per K step, in K order.
            rhs_lines = {}
            body = L[-NT:]
            del L[-NT:]
            for j, line in enumerate(body):
                rhs_lines[j] = line
            e(rhs_lines[0])
            for j in range(NT):
                if j + 1 < NT:
                    e(rhs_lines[j + 1])
                for i in range(MT):
                    n = i * NT + j
                    e(f"      %n{n} = vector.mma %lhs{i}, %rhs{j}, %b{n} : vector<16xf16>, vector<16xf16>, {V4}")
                e("      scf.schedule.fence")
        for i in (range(MT) if not (KORDER == "fence" and ABLATE == "") else ()):
            for j in range(NT):
                n = i * NT + j
                if ABLATE == "mma":
                    e(f"      %n{n} = vector.fragment<init> %zeros shape [%m, %n] : {V4}") if False else None
                    e(f"      %n{n} = vector.addf %b{n}, %b{n} : {V4}")
                    continue
                e(f"      %n{n} = vector.mma %lhs{i}, {'%rc' if RPF else '%rhs'}{j}, %b{n} : vector<16xf16>, vector<16xf16>, {V4}")
        if RPF:
            e("      scf.yield " + ", ".join(f"%n{i}" for i in range(NA)) + ", " + ", ".join(f"%rhs{j}" for j in range(NT)) + f" : {rtypes}")
        else:
            e("      scf.yield " + ", ".join(f"%n{i}" for i in range(NA)) + f" : {types}")
        e("    }")
        yv = ", ".join(f"%r{i}" for i in range(NA))
        if PREFETCH:
            yv += ", " + ", ".join(nm for nm, _ in nxt)
        e("    scf.yield " + yv + f" : {carried_t}")
        e("  }")
    e("  %mo16 = index.add %m_origin, %c16 : index")
    e("  %mo32 = index.add %m_origin, %c32 : index")
    e("  %mo48 = index.add %m_origin, %c48 : index")
    rows = ["%m_origin", "%mo16", "%mo32", "%mo48"]
    if sw:
        # SwiGLU epilogue: out[t*m + r] = f16(silu(gate[t*m + r]) * acc[r][t]), with
        # the chained kernel's scalar ops in its order (bit-identity).
        # It cannot store an f16 result fragment: that store ignores the strided
        # token-major layout and writes the tile row-major (probe: (t=16, r=17)
        # received (t=17, r=16); pipeline hidden cosine ~0.6). A fully unrolled
        # per-element form ran out of SGPRs (peak 215 of 106). So, per 16-row slab:
        # the f32 fragments go to a per-wave LDS tile through a row-fastest strided
        # view (f32 result stores honour layouts), then a real loop walks it with
        # lane-contiguous rows, so the gate loads and f16 stores coalesce into 64 B
        # runs. The weight tile's LDS is reused; it is sized for this in gen().
        e(f"  %ep_lay = encoding.layout.strided [%c1, %c16] : encoding<layout>")
        e(f"  %ep_wbytes = index.constant {16 * TOK * 4} : index")
        e("  %ep_off_i = index.mul %wave, %ep_wbytes : index")
        e("  %ep_off = index.cast %ep_off_i : index to offset")
        e(f"  %ep_view = buffer.view %wl[%ep_off] : buffer -> view<16x{TOK}xf32, %ep_lay>")
        e(f"  %ep_flat = buffer.view %wl[%ep_off] : buffer -> view<{16 * TOK}xf32>")
        e("  %gate_view = buffer.view %gate_na[%base] : buffer -> view<[%out_total]xf32>")
        e("  %out_h = buffer.view %output_na[%base] : buffer -> view<[%out_total]xf16>")
        e("  %negone = scalar.constant -1.0 : f32")
        e("  %one = scalar.constant 1.0 : f32")
        e("  %out_last = index.sub %out_total, %c1 : index")
        e(f"  %ep_n = index.constant {16 * TOK // 64} : index")
        e(f"  %ep_last = index.constant {16 * TOK - 1} : index")
        tl = ["%c0"] + [f"%c{16 * j}" for j in range(1, NT)]
        for i in range(MT):
            e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            for j in range(NT):
                e(f"  vector.fragment.store<result> %acc{i * NT + j}, %ep_view[%c0, {tl[j]}] shape [%m, %n] : {V4}, view<16x{TOK}xf32, %ep_lay>")
            e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            e(f"  %eps{i} = scf.for %ee{i} = [%c0 to %ep_n step %c1](%em{i} = %c0 : index) -> (index) {{")
            e(f"    %e64_{i} = index.mul %ee{i}, %c64 : index")
            e(f"    %ef0_{i} = index.add %e64_{i}, %l64 : index")
            e(f"    %ef_{i} = index.min %ef0_{i}, %ep_last : index")
            e(f"    %er_{i} = index.rem %ef_{i}, %c16 : index")
            e(f"    %et_{i} = index.div %ef_{i}, %c16 : index")
            e(f"    %v_{i} = view.load %ep_flat[%ef_{i}] : view<{16 * TOK}xf32> -> f32")
            e(f"    %grow_{i} = index.add {rows[i]}, %er_{i} : index")
            e(f"    %gtok_{i} = index.add %token_base, %et_{i} : index")
            e(f"    %gto_{i} = index.mul %gtok_{i}, %m_rows : index")
            e(f"    %gix0_{i} = index.add %gto_{i}, %grow_{i} : index")
            e(f"    %gix_{i} = index.min %gix0_{i}, %out_last : index")
            e(f"    %g_{i} = view.load %gate_view[%gix_{i}] : view<[%out_total]xf32> -> f32")
            e(f"    %ng_{i} = scalar.mulf %g_{i}, %negone : f32")
            e(f"    %ex_{i} = scalar.expf<afn> %ng_{i} : f32")
            e(f"    %dn_{i} = scalar.addf %one, %ex_{i} : f32")
            e(f"    %iv_{i} = scalar.divf %one, %dn_{i} : f32")
            e(f"    %sg_{i} = scalar.mulf %g_{i}, %iv_{i} : f32")
            e(f"    %ac_{i} = scalar.mulf %sg_{i}, %v_{i} : f32")
            e(f"    %h_{i} = scalar.fptrunc %ac_{i} : f32 to f16")
            e(f"    view.store %h_{i}, %out_h[%gix_{i}] : f16, view<[%out_total]xf16>")
            e(f"    scf.yield %em{i} : index")
            e("  }")
    elif kr:
        # Fused residual: out = resid + acc, element for element. resid is read as
        # an f32 result fragment through the same token-major strided view (f32
        # result loads honour the layout), so it arrives in the accumulator's
        # register layout. Same f32 add, same operand order as yah_residual_1d
        # (a + b with a the running hidden state), so bit-identical to the
        # kStore-into-partial + residual pass it replaces.
        e("  %out_layout = encoding.layout.strided [%c1, %m_rows] : encoding<layout>")
        e("  %res_t_view = buffer.view %resid_na[%base] : buffer -> view<[%m_rows]x[%tokens]xf32, %out_layout>")
        e("  %out_t_view = buffer.view %output_na[%base] : buffer -> view<[%m_rows]x[%tokens]xf32, %out_layout>")
        for i in range(MT):
            for j in range(NT):
                a = i * NT + j
                e(f"  %rf{a} = vector.fragment.load<result> %res_t_view[{rows[i]}, {toks[j]}] shape [%m, %n] : view<[%m_rows]x[%tokens]xf32, %out_layout> -> {V4}")
                e(f"  %rs{a} = vector.addf %rf{a}, %acc{a} : {V4}")
                e(f"  vector.fragment.store<result> %rs{a}, %out_t_view[{rows[i]}, {toks[j]}] shape [%m, %n] : {V4}, view<[%m_rows]x[%tokens]xf32, %out_layout>")
    elif EPI == "direct":
        # Result fragments go straight to the token-major output through a
        # strided [m_rows]x[tokens] view (element (row, t) at t*m_rows + row),
        # as emit_prefill.direct_kstore_epilogue does for the unchained kernels.
        # The staged form wrote the f32 tile to ostage, re-read it and wrote it
        # again transposed: 3x the output bytes (426 MB for a 17408x2048 gate).
        e("  %out_layout = encoding.layout.strided [%c1, %m_rows] : encoding<layout>")
        e("  %out_t_view = buffer.view %output_na[%base] : buffer -> view<[%m_rows]x[%tokens]xf32, %out_layout>")
        for i in range(MT):
            for j in range(NT):
                e(f"  vector.fragment.store<result> %acc{i * NT + j}, %out_t_view[{rows[i]}, {toks[j]}] shape [%m, %n] : {V4}, view<[%m_rows]x[%tokens]xf32, %out_layout>")
    else:
        assert TOK == 128, "the staged epilogue's copy loop assumes 128 tokens per wave"
        for i in range(MT):
            for j in range(NT):
                e(f"  vector.fragment.store<result> %acc{i * NT + j}, %ostage_view[{rows[i]}, {toks[j]}] shape [%m, %n] : {V4}, view<[%stage_rows]x[%tokens]xf32>")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        e("  %l64_i = index.cast %l64 : index to i32")
        e("  %store_sink = scf.for %j2 = [%c0 to %c128 step %c1](%mk2 = %c0 : index) -> (index) {")
        e("    %j2_i = index.cast %j2 : index to i32")
        e("    %j32b = scalar.shli %j2_i, %c6i : i32")
        e("    %e2_i = scalar.addi %l64_i, %j32b : i32")
        e("    %r2_i = scalar.shrui %e2_i, %c7i : i32")
        e("    %tok_i = scalar.andi %e2_i, %c127i : i32")
        e("    %r2_ix = index.cast %r2_i : i32 to index")
        e("    %r2 = index.max %r2_ix, %c0 : index")
        e("    %tok_ix = index.cast %tok_i : i32 to index")
        e("    %tok = index.max %tok_ix, %c0 : index")
        e("    %gr2_ix = index.add %m_origin, %r2 : index")
        e("    %gr2 = index.min %gr2_ix, %stage_last : index")
        e("    %tok_g = index.add %token_base, %tok : index")
        e("    %val = view.load %ostage_view[%gr2, %tok_g] : view<[%stage_rows]x[%tokens]xf32> -> f32")
        e("    %tok_off = index.mul %tok_g, %m_rows : index")
        e("    %gflat = index.add %tok_off, %gr2 : index")
        e("    view.store %val, %out_view[%gflat] : f32, view<[%out_total]xf32>")
        e("    scf.yield %mk2 : index")
        e("  }")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def main():
    fmt = sys.argv[1]
    kind = os.environ.get("YAH_SD_KIND", "kstore")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        f"yah_ffn_gemm_{fmt}_shared_f32.loom")
    open(out, "w").write(gen(fmt, kind))
    print(out)


if __name__ == "__main__":
    main()
