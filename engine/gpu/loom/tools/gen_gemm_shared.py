#!/usr/bin/env python3
"""Shared-decode GEMM yah_ffn_gemm_<fmt>[_swiglu|_kres]: NW wave64 waves share one decoded f16 weight tile in LDS.

Workgroup: 64*NW lanes, 16*MT rows x TOK*NW tokens; each wave owns a 16*MT x TOK accumulator tile.
Grid: (m_tiles / MT, token_tiles); the HAL's dispatch.txt token tile is TOK*NW.
Bindings: weight, [grid, ksigns,] input, [gate | resid,] wstage, ostage, output (wstage, ostage unused: the .loom ABI).
input: f16, element (k, t) at t*K + k. output: element (row, t) at t*m_rows + row; f32, or f16 for swiglu.
Each K phase decodes KSUB columns row-per-lane, branch-free, into LDS and runs the MMAs; the next phase's bytes load meanwhile.
The f32 decode ops (one f16 rounding) and the MMA order (K ascending by 16) match the hand-written yah_ffn_gemm_<fmt>.
So the output is bit-identical to that kernel.
tools/gen_gemm_tile.py reuses the decode helpers (FMTS); emit_prefill_pp.py uses this kernel only for shapes the tile GEMM skips.
"""

NW = 2                      # waves per workgroup, all on one decoded tile
# 16-row tiles per wave. Fewer than 4 serve row counts not a multiple of 64, e.g. 48-row ssm_alpha/ssm_beta (m_tiles=3).
MT = 4
LR = 16 * MT
PAD = 0                     # f16 of padding per LDS weight row
# Per-format decode variants. Off for this kernel; tools/gen_gemm_tile.py turns them on per format (its configure()).
# Q4_HDR: the Q4_K/Q5_K header (d, dmin, scales[12]) as one 16-byte load instead of byte loads that each carry an address clamp.
Q4_HDR = False
# VDEC_W: IQ3/IQ2 sign application on the two grid words instead of i8 vectors.
VDEC_W = False
# IQ3_U8F (IQ3_S / IQ3_XXS, word path): mag bytes XOR 0x80 are u = mag + 128 as unsigned bytes, converted by v_cvt_f32_ubyteN.
# The -128 rides the f32 addend: fptrunc(fma(dsc, u, -128*dsc)).
# (u-128)*dsc has <= 24 significant bits (|mag| <= 127, dsc = d * odd <= 5 bits), so it is exact in f32: bit-identical.
IQ3_U8F = False
# VDECW_FR: the word path's sign spread without quarter-rate v_mul_lo_u32.
# nibble * 0x00204081 becomes shifts and ORs, s1 * 255 becomes (s1 << 8) - s1 (top byte wraps). Same integers: bit-identical.
VDECW_FR = False
# Q4FMIX (Q4_K/Q5_K): narrow as fptrunc(fma(e, 1, -dm)) instead of fptrunc(e - dm). The product by 1 is exact: bit-identical.
# It selects v_fma_mix{lo,hi}. v_cvt_f16_f32 writes only v0..v127: with 128 VGPRs of accumulators live, each result spills one.
# The 1.0 comes from gb & ~gb so the canonicalizer cannot fold the fma back into a subf.
Q4FMIX = False

KSUB = PH = GPP = GPL = ROWP = None


def set_geometry(mt=None, tok=None, nw=None):
    """Override MT, TOK and NW for the next gen() calls; return the previous (MT, TOK, NW)."""
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
    LR = 16 * MT
    return prev


def configure(fmt):
    """Set the per-format phase geometry: KSUB = FMTS[fmt]["ksub"] K columns decoded per phase.
    A whole 256-wide block is a ~34 KB LDS tile (residency ~1.5 waves/SIMD); a narrower phase costs two barriers per phase."""
    global KSUB, ROWP, PH, GPP, GPL
    KSUB = FMTS[fmt]["ksub"]
    ROWP = KSUB + PAD           # f16 per LDS row
    PH = 256 // KSUB            # phases per 256-wide block
    GPP = KSUB // 32            # 32-element groups per row per phase
    GPL = GPP // NW             # groups decoded per lane per phase
    assert 256 % KSUB == 0 and GPP % NW == 0 and GPL >= 1


# Tokens per wave. 128 gives 32 vector<4xf32> accumulators (128 VGPRs) and some spills.
# TOK=64 removes the spills but is slower (IQ4_XS 15.52 vs 9.71 ms): the operand reuse of the 64x128 tile is worth more.
TOK = 128
NT = TOK // 16              # 16-token sub-tiles per wave
NA = MT * NT                # accumulators per wave


def _i8n(ty):
    import re as _re
    m = _re.fullmatch(r"vector<(\d+)xi8>", ty)
    return int(m.group(1)) if m and int(m.group(1)) % 4 == 0 else 0


def pack_vals(e, vals, tag):
    """Bitcast carried vector<Nxi8> values to vector<N/4xi32> and return the new (name, type) list.
    A vector<Nxi8> lowers to one byte per VGPR, so a carried qs[8] would hold 8 registers across the MMA loop."""
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
        if n:
            e(f"    {nm}_u = vector.bitcast {nm} : {ty} to {oty}")
            names.append(f"{nm}_u")
        else:
            names.append(nm)
    return names


IQ4_KVALUES = [-127, -104, -83, -65, -49, -35, -22, -10,
               1, 13, 25, 38, 53, 69, 89, 113]


def iq4xs_loads(p, blk, gb):
    """Emit this lane's raw-byte loads for one phase of row %drow; return (lines, [(name, type)]) of the loaded values.
    blk: i32 SSA byte offset of the row's current block; gb: i32 SSA index of the lane's first group in the block.
    block_iq4_xs: d f16 @0, scales_h u16 @2, scales_l[4] @4, qs[128] @8. All *_loads share this contract.
    """
    L = []
    e = L.append
    vals = []
    # the 8-byte header (d, scales_h, scales_l[4]) as one VMEM load instead of four byte/half loads
    e(f"    %{p}hd_ix = index.cast {blk} : i32 to index")
    e(f"    %{p}hd_lo = index.max %{p}hd_ix, %c0 : index")
    e(f"    %{p}hd_idx = index.min %{p}hd_lo, %w_lim8 : index")
    e(f"    %{p}hdr = vector.load %w_view[%{p}hd_idx] : view<[%w_bytes]xi8> -> vector<8xi8>")
    vals.append((f"%{p}hdr", "vector<8xi8>"))
    e(f"    %{p}d_h_i = scalar.shrui {blk}, %c1i : i32")
    e(f"    %{p}d_ix = index.cast %{p}d_h_i : i32 to index")
    e(f"    %{p}d_lo = index.max %{p}d_ix, %c0 : index")
    e(f"    %{p}d_idx = index.min %{p}d_lo, %w_half_last : index")
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}gh{u} = scalar.shrui %{p}g{u}, %c1i : i32")
        e(f"    %{p}sl_i{u} = scalar.addi {blk}, %c4i : i32")
        e(f"    %{p}sl_j{u} = scalar.addi %{p}sl_i{u}, %{p}gh{u} : i32")
        e(f"    %{p}sl_ix{u} = index.cast %{p}sl_j{u} : i32 to index")
        e(f"    %{p}sl_lo{u} = index.max %{p}sl_ix{u}, %c0 : index")
        e(f"    %{p}sl_idx{u} = index.min %{p}sl_lo{u}, %w_last : index")
        e(f"    %{p}g16_{u} = scalar.shli %{p}g{u}, %c4i : i32")
        e(f"    %{p}qs_a{u} = scalar.addi {blk}, %c8i : i32")
        e(f"    %{p}qs_b{u} = scalar.addi %{p}qs_a{u}, %{p}g16_{u} : i32")
        e(f"    %{p}qs_ix{u} = index.cast %{p}qs_b{u} : i32 to index")
        e(f"    %{p}qs_lo{u} = index.max %{p}qs_ix{u}, %c0 : index")
        e(f"    %{p}qs_idx{u} = index.min %{p}qs_lo{u}, %w_lim : index")
        e(f"    %{p}q{u} = vector.load %w_view[%{p}qs_idx{u}] : view<[%w_bytes]xi8> -> vector<16xi8>")
        vals.append((f"%{p}q{u}", "vector<16xi8>"))
    return L, vals


def iq4xs_compute(v, gb):
    """Decode the loaded values v (iq4xs_loads order) into the LDS tile, in the .loom kernel's f32 op order.
    Element g*32 + w, L = w%16: nib = w<16 ? qs[g*16+L]&15 : qs[g*16+L]>>4
      sc = ((scales_l[g/2] >> 4*(g%2)) & 15 | ((scales_h >> 2g) & 3) << 4) - 32, value = (d*sc) * kvalues[nib]"""
    L = []
    e = L.append
    it = iter(v)
    hdr = next(it)
    e(f"    %hdw = vector.bitcast {hdr} : vector<8xi8> to vector<2xi32>")
    e("    %hdw0 = vector.extract %hdw[0] : vector<2xi32> -> i32")
    e("    %hdw1 = vector.extract %hdw[1] : vector<2xi32> -> i32")
    e("    %hd16 = scalar.trunci %hdw0 : i32 to i16")
    e("    %hdf = scalar.bitcast %hd16 : i16 to f16")
    e("    %d = scalar.extf %hdf : f16 to f32")
    e("    %shv = scalar.shrui %hdw0, %c16i_h : i32")
    for u in range(GPL):
        q = next(it)
        e(f"    %g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %gp{u} = scalar.andi %g{u}, %c1i : i32")
        e(f"    %sh4_{u} = scalar.shli %gp{u}, %c2i : i32")
        e(f"    %sh2_{u} = scalar.shli %g{u}, %c1i : i32")
        # scales_l[g/2] is byte g/2 of the header's second word
        e(f"    %slg{u} = scalar.shrui %g{u}, %c1i : i32")
        e(f"    %sls{u} = scalar.shli %slg{u}, %c3i : i32")
        e(f"    %slw{u} = scalar.shrui %hdw1, %sls{u} : i32")
        e(f"    %slb{u} = scalar.andi %slw{u}, %c255i_h : i32")
        e(f"    %sc_sh{u} = scalar.shrui %slb{u}, %sh4_{u} : i32")
        e(f"    %sc_l{u} = scalar.andi %sc_sh{u}, %c15i : i32")
        e(f"    %sc_ha{u} = scalar.shrui %shv, %sh2_{u} : i32")
        e(f"    %sc_h{u} = scalar.andi %sc_ha{u}, %c3i : i32")
        e(f"    %sc_h4{u} = scalar.shli %sc_h{u}, %c4i : i32")
        e(f"    %sc6_{u} = scalar.ori %sc_l{u}, %sc_h4{u} : i32")
        e(f"    %sc{u} = scalar.subi %sc6_{u}, %c32i : i32")
        e(f"    %sc_f{u} = scalar.sitofp %sc{u} : i32 to f32")
        e(f"    %dsc{u} = scalar.mulf %d, %sc_f{u} : f32")
        e(f"    %dsc_v{u} = vector.splat %dsc{u} : vector<16xf32>")
        e(f"    %nlo{u} = vector.andi {q}, %m15v : vector<16xi8>")
        e(f"    %nhi{u} = vector.shrui {q}, %s4v : vector<16xi8>")
        # The codebook +128 as unsigned bytes: uitofp of a byte is one v_cvt_f32_ubyteN.
        # fptrunc(fma(s, u, -128*s)) = fptrunc((u-128)*s) exactly (<= 24 significant bits): bit-identical.
        # The bias rides v_fma_mix's f32 addend (no literal).
        for part in ("lo", "hi"):
            e(f"    %cu{part}{u} = vector.table.lookup %kvtu[%n{part}{u}] : vector<16xi8>, vector<16xi8> -> vector<16xi8>")
            e(f"    %fu{part}{u} = vector.uitofp %cu{part}{u} : vector<16xi8> to vector<16xf32>")
        e(f"    %nb{u} = scalar.mulf %dsc{u}, %cm128f_iq : f32")
        # scalar form: fptrunc(fma) pairs feeding from_elements select as v_fma_mix{lo,hi}_f16
        for part in ("lo", "hi"):
            hs = []
            for j in range(16):
                e(f"    %y{part}{u}_{j} = vector.extract %fu{part}{u}[{j}] : vector<16xf32> -> f32")
                e(f"    %m{part}{u}_{j} = scalar.fmaf %dsc{u}, %y{part}{u}_{j}, %nb{u} : f32")
                e(f"    %t{part}{u}_{j} = scalar.fptrunc %m{part}{u}_{j} : f32 to f16")
                hs.append(f"%t{part}{u}_{j}")
            e(f"    %h{part}{u} = vector.from_elements {', '.join(hs)} : vector<16xf16>")
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
    The clamp must be w_bytes - n for this n: w_bytes - 16 moves loads near the tensor end (IQ3_S signs) to a wrong address."""
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


def _vdec_pair(e, t, gw0, gw1, sgb8, dsc_v8, col, u, p, dsc_s=None):
    """Decode the 8 elements one sign byte covers (grid words lw=2p and 2p+1) and store them to the LDS tile.
    mags = bytes of [gw0, gw1], s = sign bits LSB first, mag = (g ^ -s) + s in i8 (exact: grid magnitudes are < 128)."""
    if VDEC_W:
        # On the two grid words: s1 = sign bit i in byte i (nibble * 0x00204081), m = s1 * 255, mag = (g ^ m) + s1.
        # Per byte that is 256 - g for a set bit, no carry since grid magnitudes are > 0. (i8 vectors lower element by element.)
        e(f"    %wsb_{t} = scalar.extui {sgb8} : i8 to i32")
        for h, gw in ((0, gw0), (1, gw1)):
            if h:
                e(f"    %wsn{h}_{t}0 = scalar.shrui %wsb_{t}, %c4i : i32")
            else:
                e(f"    %wsn{h}_{t}0 = scalar.addi %wsb_{t}, %c0i : i32")
            e(f"    %wsn{h}_{t} = scalar.andi %wsn{h}_{t}0, %c15i : i32")
            if VDECW_FR:
                # n * 0x00204081 & 0x01010101 == (t | t << 14) & 0x01010101, t = n | n << 7 (n <= 15: OR cannot carry)
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


def _col_of(e, u):
    e(f"    %col_i{u} = scalar.shli %gl{u}, %c5i : i32")
    e(f"    %col_x{u} = index.cast %col_i{u} : i32 to index")
    e(f"    %col_l{u} = index.max %col_x{u}, %c0 : index")
    e(f"    %col{u} = index.min %col_l{u}, %ccolmax : index")


def iq3s_loads(p, blk, gb):
    """block_iq3_s (110 B): d f16 @0, qs[64] @2, qh[8] @66, signs[32] @74, scales[4] @106.
    Group g reads qs[8g..8g+7], qh[g], signs[4g..4g+3] and the scale byte g/2."""
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
    """IQ3_S element decode, one row per lane, in the .loom kernel's op order. Element lw*4 + b of group g (lw = 2*l + which):
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
            e(f"    %gw_{t} = view.load %grid_view[%gid_{t}] : view<512xi32> -> i32")
            gws.append(f"%gw_{t}")
        for pp in range(4):
            e(f"    %sgb8_{u}_{pp} = vector.extract {sg}[{pp}] : vector<4xi8> -> i8")
            _vdec_pair(e, f"{u}_{pp}", gws[2 * pp], gws[2 * pp + 1], f"%sgb8_{u}_{pp}", f"%dsc_v8_{u}", f"%col{u}", u, pp, f"%dsc{u}")
    return L


def iq3s_setup():
    """Copy the 512-word grid into 2 KiB of LDS (%grid_view): the eight lookups per group are ds_reads, not global gathers.
    The first K phase starts with a barrier, which publishes the copy."""
    return ["  %grid_g = buffer.view %grid_na[%base] : buffer -> view<512xi32>",
            "  %c511 = index.constant 511 : index",
            "  %vdw_spread = scalar.constant 2113665 : i32", "  %vdw_ones = scalar.constant 16843009 : i32",
            "  %vdw_ff = scalar.constant 255 : i32",
            "  %c80x4_iq3 = scalar.constant -2139062144 : i32", "  %cm128f_iq3 = scalar.constant -128.0 : f32",
            "  %c14i_vdw = scalar.constant 14 : i32",
            "  %grid_bytes = index.constant 2048 : offset",
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


def iq3xxs_loads(p, blk, gb):
    """block_iq3_xxs (98 B): d f16 @0, qs[64] @2 (grid indices), aux[32] @66 (one LE32 word per 32-element group).
    Group g reads qs[8g..8g+7] and aux bytes 66+4g..69+4g."""
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
    """IQ3_XXS element decode, one row per lane, in the .loom kernel's op order. Element lw*4 + b of group g (lw = 2*l + which):
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
    return L


def _stage_table(name, src, n, ty, bytes_per):
    """Copy an n-entry read-only table into LDS once; the first K phase's leading barrier publishes it."""
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
    Sub-block g reads qs[32*(g/2) .. +31] (low nibbles for even g, high for odd) and scale bytes 4+g, 8+g, g (get_scale_min_k4).
    With GPL even a lane's groups are even/odd pairs that share qs; with GPL odd, q4k_compute picks the nibble at run time."""
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
    """Q4_K / Q5_K element decode, one row per lane, in the .loom kernel's op order:
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
            # header byte k = 4 + g (la), 8 + g (lb), g (lc) is word k/4, byte k%4; g < 8: two adjacent words, picked on g/4
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
        rt = GPL % 2 == 1
        if rt:
            # odd GPL: g's parity is known only at run time, so the nibble is (q >> 4*(g & 1)) & 15
            e(f"    %gpar{u} = scalar.andi %g{u}, %c1i : i32")
            e(f"    %gsh{u} = scalar.shli %gpar{u}, %c2i : i32")
            e(f"    %gsh8_{u} = scalar.trunci %gsh{u} : i32 to i8")
            e(f"    %gshv{u} = vector.splat %gsh8_{u} : vector<16xi8>")
        if q5:
            # fifth bit: quant = nibble + ((qh[lane] >> g) & 1) * 16
            e(f"    %g8_{u} = scalar.trunci %g{u} : i32 to i8")
            e(f"    %g8v_{u} = vector.splat %g8_{u} : vector<16xi8>")
        # nibbles on whole 32-bit words, (w >> s) & 0x0f0f0f0f: per-byte i8 shifts and masks lower element by element
        if rt:
            e(f"    %gshw{u} = vector.splat %gsh{u} : vector<4xi32>")
        else:
            e(f"    %gshw{u} = vector.splat %q4sh{0 if u % 2 == 0 else 4} : vector<4xi32>")
        for half, q, qh in (("lo", qa, qha if q5 else None), ("hi", qb, qhb if q5 else None)):
            e(f"    %qw{half}{u} = vector.bitcast {q} : vector<16xi8> to vector<4xi32>")
            e(f"    %qws{half}{u} = vector.shrui %qw{half}{u}, %gshw{u} : vector<4xi32>")
            e(f"    %qwm{half}{u} = vector.andi %qws{half}{u}, %m0f4 : vector<4xi32>")
            e(f"    %nq{half}{u} = vector.bitcast %qwm{half}{u} : vector<4xi32> to vector<16xi8>")
            src = f"%nq{half}{u}"
            if q5:
                # the fifth bit on words too: ((qh >> g) & 0x01010101) << 4, ORed in (the bits do not overlap, so OR is the add)
                e(f"    %hw{half}{u} = vector.bitcast {qh} : vector<16xi8> to vector<4xi32>")
                e(f"    %hgw{half}{u} = vector.splat %g{u} : vector<4xi32>")
                e(f"    %hs{half}{u} = vector.shrui %hw{half}{u}, %hgw{half}{u} : vector<4xi32>")
                e(f"    %hb{half}{u} = vector.andi %hs{half}{u}, %m014 : vector<4xi32>")
                e(f"    %h16{half}{u} = vector.shli %hb{half}{u}, %s44 : vector<4xi32>")
                e(f"    %n5w{half}{u} = vector.ori %qwm{half}{u}, %h16{half}{u} : vector<4xi32>")
                e(f"    %n5{half}{u} = vector.bitcast %n5w{half}{u} : vector<4xi32> to vector<16xi8>")
                src = f"%n5{half}{u}"
            # nibbles are 0..15 (0..31 with q5's bit): uitofp gives the same value and selects v_cvt_f32_ubyteN
            e(f"    %fq{half}{u} = vector.uitofp {src} : vector<16xi8> to vector<16xf32>")
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
    """block_iq2_xxs (66 B): d f16 @0, then per 32-element group g eight bytes at 2 + 8g.
    The eight bytes are four grid codes (bytes 0..3) and one LE32 aux word (bytes 4..7)."""
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
    """IQ2_XXS decode in the .loom kernel's op order, 8 elements per grid code li:
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
    """block_iq2_xs (74 B): d f16 @0, qs[32] u16 @2, scales[8] @66.
    Group g (32 elements) reads the four codes qs[4g..4g+3] (bytes 2 + 8g) and scale byte 66 + g."""
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
    """IQ2_XS decode in yah_ffn_gemm_iq2xs_f32.loom's op order, 8 elements per code l (elements 8l..8l+7 of the group):
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
    Group g = 4*half + seg reads 32 ql bytes at 64*half + 32*(seg&1) and 32 qh bytes at 128 + 32*half.
    Its two scales are at 192 + 8*half + 2*seg (+1)."""
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
    """Q6_K element decode, one row per lane, in the .loom kernel's op order:
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
    """Q8_0 in 256-element super-blocks of 8 blocks (272 B): group g is block g, d f16 at 34g, qs int8[32] at 34g + 2."""
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
    """Q8_0 decode in the .loom kernel's op order: value = d * f32(q)."""
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
    L += [f"  %kvu{i} = scalar.constant {(v + 128) - 256 if v + 128 > 127 else v + 128} : i8" for i, v in enumerate(IQ4_KVALUES)]
    L.append("  %kvtu = vector.from_elements " + ", ".join(f"%kvu{i}" for i in range(16)) + " : vector<16xi8>")
    L += ["  %c128f_iq = scalar.constant 128.0 : f32", "  %c128v_iq = vector.splat %c128f_iq : vector<16xf32>",
          "  %cm128f_iq = scalar.constant -128.0 : f32"]
    L += ["  %c15b = scalar.constant 15 : i8", "  %c4b = scalar.constant 4 : i8",
          "  %m15v = vector.splat %c15b : vector<16xi8>", "  %s4v = vector.splat %c4b : vector<16xi8>"]
    return L


def q3k_loads(p, blk, gb):
    """block_q3_K (110 B): hmask[32] @0, qs[64] @32, scales[12] @96, d f16 @108.
    Group g, half = g/4, sp = g%4: element h16*16 + j reads bits 2sp, 2sp+1 of qs[32*half + 16*h16 + j], bit g of hmask[16*h16 + j].
    The group's two 16-element halves use scales si = 2g and 2g+1."""
    L = []
    e = L.append
    vals = []
    e(f"    %{p}dq_h = scalar.shrui {blk}, %c1i : i32")
    e(f"    %{p}dq_i = scalar.addi %{p}dq_h, %q3c54i : i32")
    e(f"    %{p}dq_ix = index.cast %{p}dq_i : i32 to index")
    e(f"    %{p}dq_lo = index.max %{p}dq_ix, %c0 : index")
    e(f"    %{p}dq_idx = index.min %{p}dq_lo, %w_half_last : index")
    # scales[12] @96 and d @108 as one 16-byte load at @94 (inside the block)
    e(f"    %{p}q3h_o = scalar.addi {blk}, %q3c94i : i32")
    _ldv(e, p, "q3hdr", f"%{p}q3h_o", 16)
    vals.append((f"%{p}q3hdr", "vector<16xi8>"))
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
    return L, vals


def q3k_compute(v, gb):
    """Q3_K element decode in yah_ffn_gemm_q3k_f32.loom's op order:
      low = (qs >> 2sp) & 3, bit = (hmask >> g) & 1, quant = (low | bit<<2) - 4
      low4 = (scales[si&7] >> 4*(si>>3)) & 15, high2 = (scales[8+si%4] >> 2*(si>>2)) & 3
      scale = (low4 | high2<<4) - 32, value = (f32(d) * f32(scale)) * f32(quant)
    The integer steps are exact, so 16 at a time in a vector gives the same values; the f32 products keep the .loom order."""
    L = []
    e = L.append
    it = iter(v)
    hdr = next(it); hma = next(it); hmb = next(it)
    # window bytes 94..109: word i = bytes 94+4i..97+4i; d = word 3 >> 16
    e(f"    %q3w = vector.bitcast {hdr} : vector<16xi8> to vector<4xi32>")
    for w in range(4):
        e(f"    %q3w{w} = vector.extract %q3w[{w}] : vector<4xi32> -> i32")
    e("    %q3dw = scalar.shrui %q3w3, %q3c16i : i32")
    e("    %q3d16 = scalar.trunci %q3dw : i32 to i16")
    e("    %q3dh = scalar.bitcast %q3d16 : i16 to f16")
    e("    %d = scalar.extf %q3dh : f16 to f32")
    for u in range(GPL):
        qa = next(it); qb = next(it)
        e(f"    %g{u} = scalar.addi {gb}, %c{u}i : i32")
        # scales[2*(g&3)], +1 at window bytes 2+2*(g&3); scales[8+2*(g&1)], +1 at 10+2*(g&1): 16-bit pairs from words 0..3
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
            # on 32-bit words: per byte (q >> 2sp) & 3 | ((hm >> g) & 1) << 2, minus 4 as (x | 0x80) - 4 ^ 0x80 (no borrow)
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
            e(f"    %q3qf{t} = vector.sitofp %q3qn{t} : vector<16xi8> to vector<16xf32>")
            e(f"    %q3l8{t} = scalar.addi {lo8}, %c0i : i32")
            e(f"    %q3l4s{t} = scalar.shrui %q3l8{t}, %q3s4{u} : i32")
            e(f"    %q3l4{t} = scalar.andi %q3l4s{t}, %c15i : i32")
            e(f"    %q3h8{t} = scalar.addi {hi8}, %c0i : i32")
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
    # bb: block bytes; kdiv: format blocks per 256-element super-block; decode: (loads, compute); extra: bindings after %weight
    # ksub: best measured phase width at pp2048 (NW=2). At 128 the IQ3_S decode pushes 8 accumulators into scratch.
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


def gen(fmt, kind="kstore"):
    """Return the kernel text for fmt. kind: "kstore" (f32 output = acc), "kres" (f32 output = resid + acc) or "swiglu".
    swiglu is the ffn_up arm: f16 output = round_f16(silu(gate) * acc), gate the f32 gate projection in the output layout."""
    F = FMTS[fmt]
    configure(fmt)
    bb, decode = F["bb"], F["decode"]
    sw = kind == "swiglu"
    kr = kind == "kres"
    bufs = (["weight"] + F["extra"] + ["input"] + (["gate"] if sw else []) + (["resid"] if kr else [])
            + ["wstage", "ostage", "output"])
    sym = f"yah_ffn_gemm_{fmt}" + ("_swiglu" if sw else "") + ("_kres" if kr else "")
    wgs = 64 * NW
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
    e(f"  %rowgrp = index.constant {MT} : index")
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
        # The config counts the format's own blocks (32 elements for Q8_0); the kernel walks super-blocks of kdiv of them.
        # Without the division the kernel reads 8x past its weights and hangs the gfx ring.
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
    wl_bytes = LR * ROWP * 2
    if sw:
        wl_bytes = max(wl_bytes, NW * 16 * TOK * 4)   # the epilogue's f32 slabs
    e(f"  %wl_bytes = index.constant {wl_bytes} : offset")
    e("  %wl = buffer.alloca<workgroup> align(16) %wl_bytes : buffer")
    e(f"  %wl_view = buffer.view %wl[%base] : buffer -> view<{LR}x{ROWP}xf16>")
    e("  %wg_x = kernel.workgroup.id<x> : index")
    e("  %wg_y = kernel.workgroup.id<y> : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %wave = index.div %tid, %c64 : index")
    e("  %l64 = index.rem %tid, %c64 : index")
    e(f"  %clr0 = index.constant {LR} : index")
    e("  %m_origin = index.mul %wg_x, %clr0 : index")
    e("  %wv = index.add %wave, %c0 : index")
    e("  %rg64 = index.add %c0, %c0 : index")
    e("  %wtb = index.mul %wg_y, %cwtok : index")
    e(f"  %ctok = index.constant {TOK} : index")
    e("  %wave_tok = index.mul %wv, %ctok : index")
    e("  %token_base = index.add %wtb, %wave_tok : index")
    # decode lane map: lane l64 owns tile row l64 (global row m_origin + l64); wave wv owns groups [wv*GPL, wv*GPL+GPL)
    e(f"  %clr1 = index.constant {LR - 1} : index")
    e("  %drow0 = index.add %l64, %rg64 : index")
    e("  %drow = index.min %drow0, %clr1 : index")
    e("  %drow_i = index.cast %drow : index to i32")
    e("  %m_origin_i = index.cast %m_origin : index to i32")
    # the decode row is clamped: lanes past a short tile (MT < 4) re-decode its last row instead of reading past the matrix
    e("  %grow_i = scalar.addi %m_origin_i, %drow_i : i32")
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
    # Phase 0's raw bytes load before the loop. Each iteration decodes the carried bytes, then issues the next phase's loads.
    # So the DRAM latency runs under this phase's MMAs instead of in front of the decode.
    ca = ", ".join(f"%a{i} = %init : {V4}" for i in range(NA))
    L0, vals0 = loads("pf_", "%row_off_i", "%gl_i")
    L.extend(L0)
    orig0 = vals0
    vals0 = pack_vals(e, vals0, "0")
    ca += ", " + ", ".join(f"%cv{x} = {nm} : {ty}" for x, (nm, ty) in enumerate(vals0))
    carried_t = types + ", " + ", ".join(ty for _, ty in vals0)
    res = ", ".join(f"%acc{i}" for i in range(NA))
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
    cur = [(f"%cv{x}", ty) for x, (_, ty) in enumerate(vals0)]
    cur_names = unpack_vals(e, cur, orig0)
    L.extend(compute(cur_names, "%gb_i"))
    e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    # next phase, clamped to the last one (its loads are then redundant but in bounds, and the carried values are never decoded)
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
    e("    " + ", ".join(f"%r{i}" for i in range(NA)) + f" = scf.for %ks = [%c0 to %cksub step %c16]({cb}) -> ({types}) {{")
    e("      %kk = index.add %kb_k, %ks : index")
    for i in range(MT):
        e(f"      %lr{i} = index.add %rg64, %c{16 * i} : index")
        e(f"      %lhs{i} = vector.fragment.load<lhs> %wl_view[%lr{i}, %ks] shape [%m, %k] : view<{LR}x{ROWP}xf16> -> vector<16xf16>")
    for j in range(NT):
        e(f"      %rhs{j} = vector.fragment.load<rhs> %a_t_view[%kk, {toks[j]}] shape [%k, %n] : view<[%ktot]x[%tokens]xf16, %a_layout> -> vector<16xf16>")
    for i in range(MT):
        for j in range(NT):
            n = i * NT + j
            e(f"      %n{n} = vector.mma %lhs{i}, %rhs{j}, %b{n} : vector<16xf16>, vector<16xf16>, {V4}")
    e("      scf.yield " + ", ".join(f"%n{i}" for i in range(NA)) + f" : {types}")
    e("    }")
    yv = ", ".join(f"%r{i}" for i in range(NA))
    yv += ", " + ", ".join(nm for nm, _ in nxt)
    e("    scf.yield " + yv + f" : {carried_t}")
    e("  }")
    e("  %mo16 = index.add %m_origin, %c16 : index")
    e("  %mo32 = index.add %m_origin, %c32 : index")
    e("  %mo48 = index.add %m_origin, %c48 : index")
    rows = ["%m_origin", "%mo16", "%mo32", "%mo48"]
    if sw:
        # SwiGLU epilogue: out[t*m + r] = f16(silu(gate[t*m + r]) * acc[r][t]), the .loom kernel's scalar ops in order.
        # An f16 result-fragment store ignores the strided layout, and a fully unrolled per-element form runs out of SGPRs.
        # So per 16-row slab: f32 fragments go to a per-wave LDS slab via a strided view (f32 result stores honour layouts),
        # then a loop walks it with lane-contiguous rows, so gate loads and f16 stores coalesce. The slab reuses the weight LDS.
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
        # Fused residual: out = resid + acc. resid loads as an f32 result fragment through the output view: accumulator layout.
        # Same f32 add and operand order as yah_residual_add_1d (hidden state first), so bit-identical to kstore + that kernel.
        e("  %out_layout = encoding.layout.strided [%c1, %m_rows] : encoding<layout>")
        e("  %res_t_view = buffer.view %resid_na[%base] : buffer -> view<[%m_rows]x[%tokens]xf32, %out_layout>")
        e("  %out_t_view = buffer.view %output_na[%base] : buffer -> view<[%m_rows]x[%tokens]xf32, %out_layout>")
        for i in range(MT):
            for j in range(NT):
                a = i * NT + j
                e(f"  %rf{a} = vector.fragment.load<result> %res_t_view[{rows[i]}, {toks[j]}] shape [%m, %n] : view<[%m_rows]x[%tokens]xf32, %out_layout> -> {V4}")
                e(f"  %rs{a} = vector.addf %rf{a}, %acc{a} : {V4}")
                e(f"  vector.fragment.store<result> %rs{a}, %out_t_view[{rows[i]}, {toks[j]}] shape [%m, %n] : {V4}, view<[%m_rows]x[%tokens]xf32, %out_layout>")
    else:
        # result fragments go straight to the output through a strided [m_rows]x[tokens] view
        e("  %out_layout = encoding.layout.strided [%c1, %m_rows] : encoding<layout>")
        e("  %out_t_view = buffer.view %output_na[%base] : buffer -> view<[%m_rows]x[%tokens]xf32, %out_layout>")
        for i in range(MT):
            for j in range(NT):
                e(f"  vector.fragment.store<result> %acc{i * NT + j}, %out_t_view[{rows[i]}, {toks[j]}] shape [%m, %n] : {V4}, view<[%m_rows]x[%tokens]xf32, %out_layout>")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"

