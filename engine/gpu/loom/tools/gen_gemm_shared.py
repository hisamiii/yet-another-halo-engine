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
import sys

NW = int(os.environ.get("YAH_SD_NW", "2"))
PAD = int(os.environ.get("YAH_SD_PAD", "0"))
# K columns decoded per phase. A whole 256-wide block (KSUB=256) is a 64x256 f16
# tile, ~34 KB of LDS per workgroup, which dropped residency to ~1.5 waves/SIMD
# and made the kernel slower than the chained one (13.98 -> 16.5 ms/dispatch at
# pp2048, bit-identical). A narrower phase shrinks the tile at the cost of two
# barriers per phase.
KSUB = int(os.environ.get("YAH_SD_KSUB", "128"))
ROWP = KSUB + PAD           # f16 per LDS row
TOK = 128                   # tokens per wave
PH = 256 // KSUB            # phases per 256-wide block
GPP = KSUB // 32            # 32-element groups per row per phase
GPL = GPP // NW             # groups decoded per lane per phase
# Mechanism probes (numerically meaningless, timing only): decode = skip the
# weight decode, rhs = feed the MMAs the LDS lhs fragments instead of loading
# the activation from global, mma = drop the MMAs (accumulators pass through).
ABLATE = os.environ.get("YAH_SD_ABLATE", "")
PREFETCH = os.environ.get("YAH_SD_PREFETCH", "1") == "1"
EPI = os.environ.get("YAH_SD_EPI", "direct")
assert 256 % KSUB == 0 and GPP % NW == 0 and GPL >= 1

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
    e(f"    %{p}d_h_i = scalar.shrui {blk}, %c1i : i32")
    e(f"    %{p}d_ix = index.cast %{p}d_h_i : i32 to index")
    e(f"    %{p}d_lo = index.max %{p}d_ix, %c0 : index")
    e(f"    %{p}d_idx = index.min %{p}d_lo, %w_half_last : index")
    e(f"    %{p}dh = view.load %w_f16_view[%{p}d_idx] : view<[%w_halfs]xf16> -> f16")
    vals.append((f"%{p}dh", "f16"))
    for o in (2, 3):
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
        e(f"    %{p}sl{u} = view.load %w_view[%{p}sl_idx{u}] : view<[%w_bytes]xi8> -> i8")
        vals.append((f"%{p}sl{u}", "i8"))
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
    """Decode loaded values v (as iq4xs_loads returned them, possibly renamed to
    loop-carried names) into the LDS tile. Element g*32 + w: L = w%16,
    nib = w<16 ? qs[g*16+L]&15 : qs[g*16+L]>>4,
    sc = ((scales_l[g/2] >> 4*(g%2)) & 15 | ((scales_h >> 2g) & 3) << 4) - 32,
    value = (d*sc) * kvalues[nib] -- the chained kernel's f32 op order."""
    L = []
    e = L.append
    it = iter(v)
    dh = next(it); s2 = next(it); s3 = next(it)
    e(f"    %d = scalar.extf {dh} : f16 to f32")
    e(f"    %sh2_v = scalar.extui {s2} : i8 to i32")
    e(f"    %sh3_v = scalar.extui {s3} : i8 to i32")
    e("    %sh3_s = scalar.shli %sh3_v, %c8i : i32")
    e("    %shv = scalar.ori %sh2_v, %sh3_s : i32")
    for u in range(GPL):
        sl = next(it); q = next(it)
        e(f"    %g{u} = scalar.addi {gb}, %c{u}i : i32")
        e(f"    %gl{u} = scalar.addi %gl_i, %c{u}i : i32")
        e(f"    %gp{u} = scalar.andi %g{u}, %c1i : i32")
        e(f"    %sh4_{u} = scalar.shli %gp{u}, %c2i : i32")
        e(f"    %sh2_{u} = scalar.shli %g{u}, %c1i : i32")
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
        e(f"    %dsc_v{u} = vector.splat %dsc{u} : vector<16xf32>")
        e(f"    %nlo{u} = vector.andi {q}, %m15v : vector<16xi8>")
        e(f"    %nhi{u} = vector.shrui {q}, %s4v : vector<16xi8>")
        e(f"    %clo{u} = vector.table.lookup %kvt[%nlo{u}] : vector<16xi8>, vector<16xi8> -> vector<16xi8>")
        e(f"    %chi{u} = vector.table.lookup %kvt[%nhi{u}] : vector<16xi8>, vector<16xi8> -> vector<16xi8>")
        e(f"    %flo{u} = vector.sitofp %clo{u} : vector<16xi8> to vector<16xf32>")
        e(f"    %fhi{u} = vector.sitofp %chi{u} : vector<16xi8> to vector<16xf32>")
        e(f"    %vlo{u} = vector.mulf %dsc_v{u}, %flo{u} : vector<16xf32>")
        e(f"    %vhi{u} = vector.mulf %dsc_v{u}, %fhi{u} : vector<16xf32>")
        e(f"    %hlo{u} = vector.fptrunc %vlo{u} : vector<16xf32> to vector<16xf16>")
        e(f"    %hhi{u} = vector.fptrunc %vhi{u} : vector<16xf32> to vector<16xf16>")
        e(f"    %col_i{u} = scalar.shli %gl{u}, %c5i : i32")
        e(f"    %col_x{u} = index.cast %col_i{u} : i32 to index")
        e(f"    %col_l{u} = index.max %col_x{u}, %c0 : index")
        e(f"    %col{u} = index.min %col_l{u}, %ccolmax : index")
        e(f"    %colh{u} = index.add %col{u}, %c16 : index")
        e(f"    vector.store %hlo{u}, %wl_view[%drow, %col{u}] : vector<16xf16>, view<64x{ROWP}xf16>")
        e(f"    vector.store %hhi{u}, %wl_view[%drow, %colh{u}] : vector<16xf16>, view<64x{ROWP}xf16>")
    return L


FMTS = {
    # fmt: (block bytes, decode emitter, extra setup lines)
    "iq4xs": (136, (iq4xs_loads, iq4xs_compute)),
}


def gen(fmt):
    bb, decode = FMTS[fmt]
    sym = f"yah_ffn_gemm_{fmt}"
    wgs = 64 * NW
    wtok = TOK * NW
    L = []
    e = L.append
    e(f"// GENERATED by tools/gen_gemm_shared.py {fmt} (NW={NW}, PAD={PAD}) -- edit the generator.")
    e("//")
    e(f"// Shared-decode kStore GEMM for {fmt}: {NW} wave64 waves share one decoded 64x256")
    e("// weight tile per K block; each wave accumulates 64 rows x 128 tokens. Same ABI,")
    e("// output layout and arithmetic order as the chained kernel; see the generator.")
    e("amdgpu.target<gfx11-generic> @yah_gemm_w64 {subgroup_size = 64}")
    e("")
    for c in ("m_tiles", "k_blocks", "token_tiles"):
        e(f"config.decl @{sym}.{c} : %value: index where [range(%value, 1, 4096)]")
    e("")
    e(f"kernel.def target(@yah_gemm_w64) @{sym}() {{")
    e("  %unit = index.constant 1 : index")
    e(f"  %m_tiles = config.get @{sym}.m_tiles : index")
    e(f"  %token_tiles = config.get @{sym}.token_tiles : index")
    e(f"  %wgs = index.constant {wgs} : index")
    e("  %rowgrp = index.constant 4 : index")
    e("  %m_groups = index.div %m_tiles, %rowgrp : index")
    e("  kernel.launch.config workgroups(%m_groups, %token_tiles, %unit) workgroup_size(%wgs, %unit, %unit) : index")
    e("} launch(%weight: buffer, %input: buffer, %wstage: buffer, %ostage: buffer, %output: buffer) {")
    e("  %base = index.constant 0 : offset")
    for v in (0, 1, 2, 4, 6, 7, 8, 16, 32, 48, 63, 64, 80, 96, 112, 127, 128, 224, 256):
        e(f"  %c{v} = index.constant {v} : index")
    for v in (0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 32):
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
    e(f"  %k_blocks = config.get @{sym}.k_blocks : index")
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
    e("  %w_half_last = index.sub %w_halfs, %c1 : index")
    e("  %out_total = index.mul %m_rows, %tokens : index")
    e("  %stage_rows = index.mul %m_tiles, %c16 : index")
    e("  %stage_last = index.sub %stage_rows, %c1 : index")
    e("  %a_layout = encoding.layout.strided [%c1, %ktot] : encoding<layout>")
    e("  %weight_na, %input_na, %wstage_na, %ostage_na, %output_na = buffer.assume.noalias %weight, %input, %wstage, %ostage, %output : buffer, buffer, buffer, buffer, buffer")
    e("  %w_view = buffer.view %weight_na[%base] : buffer -> view<[%w_bytes]xi8>")
    e("  %w_f16_view = buffer.view %weight_na[%base] : buffer -> view<[%w_halfs]xf16>")
    e("  %a_t_view = buffer.view %input_na[%base] : buffer -> view<[%ktot]x[%tokens]xf16, %a_layout>")
    e("  %out_view = buffer.view %output_na[%base] : buffer -> view<[%out_total]xf32>")
    e("  %ostage_view = buffer.view %ostage_na[%base] : buffer -> view<[%stage_rows]x[%tokens]xf32>")
    e(f"  %wl_bytes = index.constant {64 * ROWP * 2} : offset")
    e("  %wl = buffer.alloca<workgroup> align(16) %wl_bytes : buffer")
    e(f"  %wl_view = buffer.view %wl[%base] : buffer -> view<64x{ROWP}xf16>")
    e("  %wg_x = kernel.workgroup.id<x> : index")
    e("  %wg_y = kernel.workgroup.id<y> : index")
    e("  %m_origin = index.mul %wg_x, %c64 : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %wave = index.div %tid, %c64 : index")
    e("  %l64 = index.rem %tid, %c64 : index")
    e("  %wtb = index.mul %wg_y, %cwtok : index")
    e("  %wave_tok = index.mul %wave, %c128 : index")
    e("  %token_base = index.add %wtb, %wave_tok : index")
    # decode lane map: lane l64 owns row l64; wave w owns groups [w*GPL, w*GPL+GPL)
    e("  %drow = index.min %l64, %c63 : index")
    e("  %drow_i = index.cast %drow : index to i32")
    e("  %m_origin_i = index.cast %m_origin : index to i32")
    e("  %grow_i = scalar.addi %m_origin_i, %drow_i : i32")
    e("  %k_blocks_i = index.cast %k_blocks : index to i32")
    e("  %bpr_i = scalar.muli %k_blocks_i, %cbbi : i32")
    e("  %row_off_i = scalar.muli %grow_i, %bpr_i : i32")
    e("  %wave_i = index.cast %wave : index to i32")
    e(f"  %cgpl = scalar.constant {GPL} : i32")
    e("  %gl_i = scalar.muli %wave_i, %cgpl : i32")
    e("  %kphases = index.mul %k_blocks, %cph : index")
    if fmt.startswith("iq4"):
        for i, v in enumerate(IQ4_KVALUES):
            e(f"  %kv{i} = scalar.constant {v} : i8")
        e("  %kvt = vector.from_elements " + ", ".join(f"%kv{i}" for i in range(16)) + " : vector<16xi8>")
        e("  %c15b = scalar.constant 15 : i8")
        e("  %c4b = scalar.constant 4 : i8")
        e("  %m15v = vector.splat %c15b : vector<16xi8>")
        e("  %s4v = vector.splat %c4b : vector<16xi8>")
    e("  %zeros = vector.constant 0.0 : vector<4xf32>")
    e("  %init = vector.fragment<init> %zeros shape [%m, %n] : vector<4xf32>")
    for j in range(1, 8):
        e(f"  %t{16 * j} = index.add %token_base, %c{16 * j} : index")
    toks = ["%token_base"] + [f"%t{16 * j}" for j in range(1, 8)]
    V4 = "vector<4xf32>"
    types = ", ".join([V4] * 32)
    loads, compute = decode
    ca = ", ".join(f"%a{i} = %init : {V4}" for i in range(32))
    carried_t = types
    if PREFETCH:
        # Phase 0's raw bytes are loaded before the loop; each iteration decodes
        # the carried bytes and then issues the NEXT phase's loads, so their DRAM
        # latency runs under this phase's MMAs instead of in front of the decode.
        L0, vals0 = loads("pf_", "%row_off_i", "%gl_i")
        L.extend(L0)
        ca += ", " + ", ".join(f"%cv{x} = {nm} : {ty}" for x, (nm, ty) in enumerate(vals0))
        carried_t = types + ", " + ", ".join(ty for _, ty in vals0)
    res = ", ".join(f"%acc{i}" for i in range(32))
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
    else:
        Lc, cur = loads("cu_", "%blk_i", "%gb_i")
        L.extend(Lc)
    if ABLATE != "decode":
        L.extend(compute([nm for nm, _ in cur], "%gb_i"))
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
    cb = ", ".join(f"%b{i} = %a{i} : {V4}" for i in range(32))
    e("    " + ", ".join(f"%r{i}" for i in range(32)) + f" = scf.for %ks = [%c0 to %cksub step %c16]({cb}) -> ({types}) {{")
    e("      %kk = index.add %kb_k, %ks : index")
    for i in range(4):
        e(f"      %lhs{i} = vector.fragment.load<lhs> %wl_view[%c{16 * i}, %ks] shape [%m, %k] : view<64x{ROWP}xf16> -> vector<16xf16>")
    for j in range(8):
        if ABLATE == "rhs":
            e(f"      %rhs{j} = vector.fragment.load<rhs> %wl_view[%c{16 * (j % 4)}, %ks] shape [%k, %n] : view<64x{ROWP}xf16> -> vector<16xf16>")
            continue
        e(f"      %rhs{j} = vector.fragment.load<rhs> %a_t_view[%kk, {toks[j]}] shape [%k, %n] : view<[%ktot]x[%tokens]xf16, %a_layout> -> vector<16xf16>")
    for i in range(4):
        for j in range(8):
            n = i * 8 + j
            if ABLATE == "mma":
                e(f"      %n{n} = vector.fragment<init> %zeros shape [%m, %n] : {V4}") if False else None
                e(f"      %n{n} = vector.addf %b{n}, %b{n} : {V4}")
                continue
            e(f"      %n{n} = vector.mma %lhs{i}, %rhs{j}, %b{n} : vector<16xf16>, vector<16xf16>, {V4}")
    e("      scf.yield " + ", ".join(f"%n{i}" for i in range(32)) + f" : {types}")
    e("    }")
    yv = ", ".join(f"%r{i}" for i in range(32))
    if PREFETCH:
        yv += ", " + ", ".join(nm for nm, _ in nxt)
    e("    scf.yield " + yv + f" : {carried_t}")
    e("  }")
    e("  %mo16 = index.add %m_origin, %c16 : index")
    e("  %mo32 = index.add %m_origin, %c32 : index")
    e("  %mo48 = index.add %m_origin, %c48 : index")
    rows = ["%m_origin", "%mo16", "%mo32", "%mo48"]
    if EPI == "direct":
        # Result fragments go straight to the token-major output through a
        # strided [m_rows]x[tokens] view (element (row, t) at t*m_rows + row),
        # as emit_prefill.direct_kstore_epilogue does for the unchained kernels.
        # The staged form wrote the f32 tile to ostage, re-read it and wrote it
        # again transposed: 3x the output bytes (426 MB for a 17408x2048 gate).
        e("  %out_layout = encoding.layout.strided [%c1, %m_rows] : encoding<layout>")
        e("  %out_t_view = buffer.view %output_na[%base] : buffer -> view<[%m_rows]x[%tokens]xf32, %out_layout>")
        for i in range(4):
            for j in range(8):
                e(f"  vector.fragment.store<result> %acc{i * 8 + j}, %out_t_view[{rows[i]}, {toks[j]}] shape [%m, %n] : {V4}, view<[%m_rows]x[%tokens]xf32, %out_layout>")
    else:
        for i in range(4):
            for j in range(8):
                e(f"  vector.fragment.store<result> %acc{i * 8 + j}, %ostage_view[{rows[i]}, {toks[j]}] shape [%m, %n] : {V4}, view<[%stage_rows]x[%tokens]xf32>")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        e("  %l64_i = index.cast %l64 : index to i32")
        e("  %store_sink = scf.for %j2 = [%c0 to %c128 step %c1](%mk2 = %c0 : index) -> (index) {")
        e("    %j2_i = index.cast %j2 : index to i32")
        e("    %j32b = scalar.shli %j2_i, %c6i : i32")
        e("    %e2_i = scalar.addi %l64_i, %j32b : i32")
        e("    %r2_i = scalar.shrui %e2_i, %c7i : i32")
        e("    %c127i = scalar.constant 127 : i32")
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
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        f"yah_ffn_gemm_{fmt}_shared_f32.loom")
    open(out, "w").write(gen(fmt))
    print(out)


if __name__ == "__main__":
    main()
