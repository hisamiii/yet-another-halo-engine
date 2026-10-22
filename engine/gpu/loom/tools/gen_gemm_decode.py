#!/usr/bin/env python3
"""Weight decoder for the tile GEMM (tools/gen_gemm_tile.py): the MQ4 loads and the branch-free f32 decode of one K phase
into f16 LDS rows, and the format table FMTS (block bytes, phase width, extra bindings).
The f32 decode ops (one f16 rounding) match the reference dequant, so the GEMMs are bit-identical to it.
gen_gemm_tile.configure() sets the geometry below before it emits a kernel. The GGUF formats' decoders went with the GGUF
model path (tag pre-cleanup-2026-10-20).
"""

NW = 2                      # decoding waves per workgroup
LR = 64                     # weight rows per decoded tile
PAD = 0                     # f16 of padding per LDS weight row

KSUB = PH = GPP = GPL = ROWP = None


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


def _ldv(e, p, name, off, n):
    """Load n consecutive weight bytes at i32 SSA offset `off` as vector<nxi8>.
    The clamp must be w_bytes - n for this n: w_bytes - 16 moves loads near the tensor end (IQ3_S signs) to a wrong address."""
    e(f"    %{p}{name}_ix = index.cast {off} : i32 to index")
    e(f"    %{p}{name}_lo = index.max %{p}{name}_ix, %c0 : index")
    e(f"    %{p}{name}_idx = index.min %{p}{name}_lo, %w_lim{n} : index")
    e(f"    %{p}{name} = view.load %w_view[%{p}{name}_idx] : view<[%w_bytes]xi8> -> vector<{n}xi8>".replace("view.load", "vector.load"))


def _col_of(e, u):
    e(f"    %col_i{u} = scalar.shli %gl{u}, %c5i : i32")
    e(f"    %col_x{u} = index.cast %col_i{u} : i32 to index")
    e(f"    %col_l{u} = index.max %col_x{u}, %c0 : index")
    e(f"    %col{u} = index.min %col_l{u}, %ccolmax : index")


def mq4_loads(p, blk, gb):
    """MQ4G256 (hipfire MQ4G256V2, nibbles stored signed by the HFQ loader): f16 scale of K 0-127 @0, of K 128-255 @4,
    nibble bytes @8 (byte j: element 2j low, 2j + 1 high). Group g reads scale g / 4 and bytes 16g .. 16g + 15."""
    L = []
    e = L.append
    vals = []
    for u in range(GPL):
        e(f"    %{p}g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %{p}gs{u} = scalar.shrui %{p}g{u}, %c2i : i32")
        e(f"    %{p}gs2_{u} = scalar.shli %{p}gs{u}, %c1i : i32")
        e(f"    %{p}bh{u} = scalar.shrui {blk}, %c1i : i32")
        e(f"    %{p}dh_i{u} = scalar.addi %{p}bh{u}, %{p}gs2_{u} : i32")
        e(f"    %{p}dx{u} = index.cast %{p}dh_i{u} : i32 to index")
        e(f"    %{p}dl{u} = index.max %{p}dx{u}, %c0 : index")
        e(f"    %{p}dc{u} = index.min %{p}dl{u}, %w_half_last : index")
        e(f"    %{p}dh{u} = view.load %w_f16_view[%{p}dc{u}] : view<[%w_halfs]xf16> -> f16")
        e(f"    %{p}g16_{u} = scalar.shli %{p}g{u}, %c4i : i32")
        e(f"    %{p}qa{u} = scalar.addi {blk}, %c8i : i32")
        e(f"    %{p}qb{u} = scalar.addi %{p}qa{u}, %{p}g16_{u} : i32")
        _ldv(e, p, f"q_v{u}", f"%{p}qb{u}", 16)
        vals += [(f"%{p}dh{u}", "f16"), (f"%{p}q_v{u}", "vector<16xi8>")]
    return L, vals


def mq4_compute(v, gb):
    """MQ4G256 decode: value = f16(d * n), n the signed nibble, as fma(d, n ^ 8, -8 d) from the unsigned code (exact in f32)."""
    L = []
    e = L.append
    it = iter(v)
    for u in range(GPL):
        dh = next(it); q = next(it)
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %d{u} = scalar.extf {dh} : f16 to f32")
        e(f"    %nb{u} = scalar.mulf %d{u}, %cm8f_mq : f32")
        e(f"    %qc{u} = vector.xori {q}, %m88v_mq : vector<16xi8>")
        e(f"    %nlo{u} = vector.andi %qc{u}, %m15v_mq : vector<16xi8>")
        e(f"    %nhi{u} = vector.shrui %qc{u}, %s4v_mq : vector<16xi8>")
        e(f"    %flo{u} = vector.uitofp %nlo{u} : vector<16xi8> to vector<16xf32>")
        e(f"    %fhi{u} = vector.uitofp %nhi{u} : vector<16xi8> to vector<16xf32>")
        hs = []
        for j in range(32):
            src = f"%flo{u}" if j % 2 == 0 else f"%fhi{u}"
            e(f"    %y{u}_{j} = vector.extract {src}[{j // 2}] : vector<16xf32> -> f32")
            e(f"    %m{u}_{j} = scalar.fmaf %d{u}, %y{u}_{j}, %nb{u} : f32")
            e(f"    %t{u}_{j} = scalar.fptrunc %m{u}_{j} : f32 to f16")
            hs.append(f"%t{u}_{j}")
        e(f"    %hlo{u} = vector.from_elements {', '.join(hs[:16])} : vector<16xf16>")
        e(f"    %hhi{u} = vector.from_elements {', '.join(hs[16:])} : vector<16xf16>")
        _col_of(e, u)
        e(f"    %colh{u} = index.add %col{u}, %c16 : index")
        e(f"    vector.store %hlo{u}, %wl_view[%drow, %col{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
        e(f"    vector.store %hhi{u}, %wl_view[%drow, %colh{u}] : vector<16xf16>, view<{LR}x{ROWP}xf16>")
    return L


def mq4_setup():
    return ["  %c88b_mq = scalar.constant -120 : i8", "  %m88v_mq = vector.splat %c88b_mq : vector<16xi8>",
            "  %c15b_mq = scalar.constant 15 : i8", "  %m15v_mq = vector.splat %c15b_mq : vector<16xi8>",
            "  %c4b_mq = scalar.constant 4 : i8", "  %s4v_mq = vector.splat %c4b_mq : vector<16xi8>",
            "  %cm8f_mq = scalar.constant -8.0 : f32"]


FMTS = {
    # bb: block bytes; decode: (loads, compute); extra: bindings after %weight; ksub: K columns decoded per phase
    # hipfire MQ4G256V2 (HFQ models): scale per 128, signed nibbles
    "mq4": dict(bb=136, ksub=64, decode=(mq4_loads, mq4_compute), extra=[], setup=mq4_setup),
}
