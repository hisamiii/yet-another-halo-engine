#!/usr/bin/env python3
"""Generate the tile GEMM: wave32, many waves, both operands staged in LDS.

usage: gen_gemm_tile.py <fmt> [out.loom]     env: YAH_TG_KIND=kstore|swiglu|kres
                                                  YAH_TG_KSUB (default per format)
                                                  YAH_TG_BM/BN/WM/WN, YAH_TG_APAD/WPAD

Same ABI and output as gen_gemm_shared.py's kernels, and the same decode
arithmetic and per-accumulator MMA order, so the output is bit-identical to
them (hidden f837e614ff55d1d1 at pp2048). What changes is the latency structure,
modelled on HIP's HalfPrefillGemmKernel<256, 256, 8, 4>:

  * BM x BN per workgroup (default 128 rows x 256 tokens) over WM x WN wave32
    waves (default 4 x 4), each owning a 32-row x 64-token tile: 8 accumulators;
  * per K phase the decoded weight tile (BM x KSUB) AND the activation tile
    (BN tokens x KSUB) sit in LDS, so the MMA loop reads only LDS;
  * the next phase's weight bytes and activation rows are loaded from global
    into registers while this phase computes, and stored to LDS after it.

Why the shared kernel lost to HIP (ATT, IQ3_S 17408x5120 kStore): 80% of its
wave time was s_waitcnt, mostly vmcnt on activation loads issued a few
instructions before the WMMA that consumed them, with ~2 wave64 waves per SIMD
to cover it. HIP's kernel waits just as much per wave but runs 8 waves per SIMD.

Why the first version of this kernel (64 x 256, 8 waves) lost too (19.9 ms vs
the shared kernel's 14.9): LDS bank conflicts. Unpadded, a decoded weight row is
KSUB*2 = 128 B, so the 16 rows of an lhs fragment load hit 2 bank groups and
every ds_load_b128 stalled at issue (62% of wave time in lgkmcnt waits). Padding
each weight row by 8 f16 (WPAD) and each activation row by 8 f16 (APAD) takes
it to 11.1 ms; 128 x 256 over 16 waves to 10.5 ms (HIP: ~11.3 ms in the
pipeline). WPAD=4 is 19.6 ms, APAD=0 28 ms; 256 x 128 over 16 waves 14.4 ms;
unroll(2|4) on the K step no better. 256 x 256 does not fit: 64 KB of tiles
plus the IQ grid table staged in LDS is over the 64 KB workgroup limit.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_gemm_shared as G  # noqa: E402

# Workgroup geometry (env; defaults = the measured best, 128 x 256 over 4 x 4
# waves). HIP's HalfPrefillGemmKernel<256, 256, 8, 4> is BM=BN=256, WM=8, WN=4,
# which does not fit here: 64 KB of tiles plus the IQ grid table staged in LDS
# is over gfx11's 64 KB per workgroup.
BN = int(os.environ.get("YAH_TG_BN", "256"))   # tokens per workgroup
BM = int(os.environ.get("YAH_TG_BM", "128"))   # rows per workgroup
WM = int(os.environ.get("YAH_TG_WM", "4"))     # waves along rows
WN = int(os.environ.get("YAH_TG_WN", "4"))     # waves along tokens
TM, TN = BM // WM, BN // WN   # per-wave tile
FM, FN = TM // 16, TN // 16   # fragments per wave
NWAVE = WM * WN
LANES = 32 * NWAVE
assert BM % (16 * WM) == 0 and BN % (16 * WN) == 0 and LANES >= BM and LANES % BN == 0
ROWGRP = BM // 16             # m_tiles per workgroup
APL = LANES // BN             # lanes staging one token row of the activation tile
APAD = int(os.environ.get("YAH_TG_APAD", "8"))
# Global activation row pitch pad (f16). K*2-byte rows are multiples of 1024 B at
# every model K, which aliases the token rows of one tile in the cache.
AGPAD = int(os.environ.get("YAH_TG_AGPAD", "0"))

# f16 of padding per decoded weight row: unpadded rows are 128 B apart at
# KSUB=64, so a 16-lane lhs fragment load hits 2 bank groups (8-way conflicts)
WPAD = int(os.environ.get("YAH_TG_WPAD", "8"))
# FRAG=1: both LDS tiles fragment-major, as HIP's kernel lays them out: every
# 16 x 16 block is 512 contiguous bytes, so each lane of a fragment load reads
# its own contiguous 32 B and the loads are conflict-free with no padding
# (APAD/WPAD are ignored). That is what lets 256 x 256 fit in 64 KB.
FRAG = os.environ.get("YAH_TG_FRAG", "0") == "1"
FENCE = os.environ.get("YAH_TG_FENCE", "1") == "1"
# SWZ=G: grouped launch order -- consecutive workgroups cover G row groups x all
# token tiles, so a weight tile is reused by the 8 token tiles while it is still
# cached and only G weight tiles and the token tiles' activations are live.
# Needs m_groups % G == 0 (the emitter passes it only then). 0 = plain order.
# MEASURED, OFF: G=4 is 10.25 -> 10.12 ms standalone (IQ4_XS 17408x5120) but
# +83/+90 ms on the pp2048 GEMM total, interleaved (YAH_TILE_SWZ=4 to emit it).
SWZ = int(os.environ.get("YAH_TG_SWZ", "0"))
# EPI_LDS=1: kstore/kres epilogue through a wave-private LDS slab (TM rows x 16
# tokens, token-major) so each lane stores 16 contiguous rows of one token with
# b128 stores. The direct fragment store writes each lane's 8 values at an
# 8-byte row stride: 64 global_store_b32 per wave, ~10% of IQ4_XS wave time.
# IQ4_XS 17408x5120 10.26 -> 10.03 ms standalone; pp2048 GEMMs 3449 -> 3438 ms.
EPI_LDS = os.environ.get("YAH_TG_EPI_LDS", "1") == "1"
# inner K-step loop policy, e.g. "unroll(%c2) schedule(recurrence)"
KPOL = os.environ.get("YAH_TG_KPOL", "")
# groups decoded per decoding lane (q4k's even/odd pairing needs 2)
# q4k/q5k decode one group per lane (run-time nibble choice): twice the
# decoding lanes of the even/odd pairing. YAH_TG_Q4GPL=2 restores the pairing.
GPL_OF = {"q4k": int(os.environ.get("YAH_TG_Q4GPL", "1")), "q5k": int(os.environ.get("YAH_TG_Q4GPL", "1"))}
# KSUB=64 everywhere: at 128 the 128 x 256 tiles need ~104 KB of LDS. q4k/q5k
# keep their even group count per lane (GPL=2) with one decoding slot.
KSUB_OF = {}


def configure(fmt):
    ksub = int(os.environ.get("YAH_TG_KSUB", KSUB_OF.get(fmt, 64)))
    G.KSUB = ksub
    G.PAD = 0 if FRAG else WPAD
    G.ROWP = ksub + G.PAD
    G.PH = 256 // ksub
    G.GPP = ksub // 32
    G.GPL = GPL_OF.get(fmt, 1)
    G.LR = BM
    G.NW = NWAVE // 2          # table-staging stride 64*NW = LANES
    assert G.GPP % G.GPL == 0
    assert (G.GPP // G.GPL) * BM <= LANES, "not enough lanes to decode a phase in one pass"
    assert (ksub // 8) % APL == 0, "activation row does not split evenly over its lanes"
    return ksub


def geometry():
    """(tokens per workgroup, m_tiles per workgroup) for dispatch.txt."""
    return BN, ROWGRP


def set_geometry(bm=None, bn=None, wm=None, wn=None):
    """Switch the workgroup geometry for the next gen() (the emitter uses a
    16-row tile for the 48-row matrices); returns the previous (BM, BN, WM, WN)."""
    global BM, BN, WM, WN, TM, TN, FM, FN, NWAVE, LANES, ROWGRP, APL
    prev = (BM, BN, WM, WN)
    BM, BN, WM, WN = bm or BM, bn or BN, wm or WM, wn or WN
    TM, TN = BM // WM, BN // WN
    FM, FN = TM // 16, TN // 16
    NWAVE = WM * WN
    LANES = 32 * NWAVE
    assert BM % (16 * WM) == 0 and BN % (16 * WN) == 0 and LANES >= BM and LANES % BN == 0
    ROWGRP = BM // 16
    APL = LANES // BN
    return prev


def gen(fmt, kind="kstore"):
    F = G.FMTS[fmt]
    ksub = configure(fmt)
    bb, (loads, compute) = F["bb"], F["decode"]
    kr = kind == "kres"
    sw = kind == "swiglu"
    bufs = (["weight"] + F["extra"] + ["input"] + (["gate"] if sw else []) + (["resid"] if kr else [])
            + ["wstage", "ostage", "output"])
    sym = f"yah_ffn_gemm_{fmt}" + ("_swiglu" if sw else "") + ("_kres" if kr else "")
    slots = G.GPP // G.GPL          # decoding lane groups of 64 per phase
    arow = ksub + (0 if FRAG else APAD)   # f16 per LDS activation row
    aseg = ksub // 8                # 16-byte segments per token row
    aspl = aseg // APL              # of which one staging lane loads
    V8 = "vector<8xf32>"
    VF = "vector<16xf16>"
    L = []
    e = L.append
    e(f"// GENERATED by tools/gen_gemm_tile.py {fmt} {kind} (KSUB={ksub}) -- edit the generator.")
    e("//")
    e(f"// Tile GEMM for {fmt}: {NWAVE} wave32 waves over a {BM} x {BN} tile, {TM} x {TN} per wave,")
    e("// decoded weights and staged activations both in LDS. See the generator.")
    e("amdgpu.target<gfx11-generic> @yah_tile_w32 {subgroup_size = 32}")
    e("")
    for c in ("m_tiles", "k_blocks", "token_tiles"):
        e(f"config.decl @{sym}.{c} : %value: index where [range(%value, 1, 4096)]")
    e("")
    e(f"kernel.def target(@yah_tile_w32) @{sym}() {{")
    e("  %unit = index.constant 1 : index")
    e(f"  %m_tiles = config.get @{sym}.m_tiles : index")
    e(f"  %token_tiles = config.get @{sym}.token_tiles : index")
    e(f"  %wgs = index.constant {LANES} : index")
    e(f"  %rowgrp = index.constant {ROWGRP} : index")
    e("  %m_groups = index.div %m_tiles, %rowgrp : index")
    e("  kernel.launch.config workgroups(%m_groups, %token_tiles, %unit) workgroup_size(%wgs, %unit, %unit) : index")
    e("} launch(" + ", ".join(f"%{b}: buffer" for b in bufs) + ") {")
    e("  %base = index.constant 0 : offset")
    for v in sorted({0, 1, 2, 4, 6, 7, 8, 16, 32, 48, 63, 64, 80, 96, 112, 127, 128, 224, 255, 256, 512, BM, BM - 1, BN - 1}):
        e(f"  %c{v} = index.constant {v} : index")
    # the same i32 constants gen_gemm_shared defines (q8_0 needs 18 and 34)
    for v in (0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 18, 21, 24, 28, 32, 34, 48, 63, 64, 66, 74, 104, 106, 127, 128, 192, 255):
        e(f"  %c{v}i = scalar.constant {v} : i32")
    e(f"  %cbb = index.constant {bb} : index")
    e(f"  %cbbh = index.constant {bb // 2} : index")
    e(f"  %cbbi = scalar.constant {bb} : i32")
    e(f"  %cwtok = index.constant {BN} : index")
    e(f"  %cksub = index.constant {ksub} : index")
    e(f"  %cph = index.constant {G.PH} : index")
    e(f"  %ccolmax = index.constant {ksub - 32} : index")
    e(f"  %cgppi = scalar.constant {G.GPP} : i32")
    e("  %m = index.constant 16 : index")
    e("  %n = index.constant 16 : index")
    e("  %k = index.constant 16 : index")
    e(f"  %m_tiles = config.get @{sym}.m_tiles : index")
    # q8_0's k_blocks counts 32-wide blocks; the decode works in 256-wide ones
    # (bb=272 = 8 x 34), as gen_gemm_shared does it. A missed division here walks
    # 8x past the weights -- the ring hang of 2026-09-29.
    kdiv = F.get("kdiv", 1)
    if kdiv == 1:
        e(f"  %k_blocks = config.get @{sym}.k_blocks : index")
    else:
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
    for nb in (4, 8, 16):
        e(f"  %cw{nb} = index.constant {nb} : index")
        e(f"  %w_lim{nb} = index.sub %w_bytes, %cw{nb} : index")
    e("  %w_half_last = index.sub %w_halfs, %c1 : index")
    e("  %out_total = index.mul %m_rows, %tokens : index")
    e(f"  %cagpad = index.constant {AGPAD} : index")
    e("  %apitch = index.add %ktot, %cagpad : index")
    e("  %a_total = index.mul %tokens, %apitch : index")
    e("  %a_last8 = index.sub %a_total, %c8 : index")
    e("  %a_layout = encoding.layout.strided [%c1, %ktot] : encoding<layout>")
    e("  " + ", ".join(f"%{b}_na" for b in bufs) + " = buffer.assume.noalias "
      + ", ".join(f"%{b}" for b in bufs) + " : " + ", ".join(["buffer"] * len(bufs)))
    e("  %w_view = buffer.view %weight_na[%base] : buffer -> view<[%w_bytes]xi8>")
    e("  %w_f16_view = buffer.view %weight_na[%base] : buffer -> view<[%w_halfs]xf16>")
    e("  %a_flat = buffer.view %input_na[%base] : buffer -> view<[%a_total]xf16>")
    # LDS: decoded weight tile and staged activation tile
    e(f"  %wl_bytes = index.constant {BM * G.ROWP * 2} : offset")
    e("  %wl = buffer.alloca<workgroup> align(16) %wl_bytes : buffer")
    e(f"  %wl_view = buffer.view %wl[%base] : buffer -> view<{BM}x{G.ROWP}xf16>")
    e(f"  %al_bytes = index.constant {BN * arow * 2} : offset")
    e("  %al = buffer.alloca<workgroup> align(16) %al_bytes : buffer")
    e(f"  %al_rows = buffer.view %al[%base] : buffer -> view<{BN}x{arow}xf16>")
    e(f"  %carow = index.constant {arow} : index")
    e(f"  %cksubi = index.constant {ksub} : index")
    e("  %al_layout = encoding.layout.strided [%c1, %carow] : encoding<layout>")
    e(f"  %al_t = buffer.view %al[%base] : buffer -> view<{ksub}x{BN}xf16, %al_layout>")
    if FRAG:
        # fragment-major views: row r of block b is row b*16 + r of a 16-wide view
        e(f"  %wl_fm = buffer.view %wl[%base] : buffer -> view<{BM * ksub // 16}x16xf16>")
        e(f"  %al_fm = buffer.view %al[%base] : buffer -> view<{BN * ksub // 16}x16xf16>")
        e("  %fm_lay = encoding.layout.strided [%c1, %c16] : encoding<layout>")
        e(f"  %al_fmt = buffer.view %al[%base] : buffer -> view<16x{BN * ksub // 16}xf16, %fm_lay>")
    if SWZ:
        e("  %wg_x0 = kernel.workgroup.id<x> : index")
        e("  %wg_y0 = kernel.workgroup.id<y> : index")
        e("  %sw_mg = index.div %m_tiles, %c" + str(ROWGRP) + " : index")
        e("  %sw_ly = index.mul %wg_y0, %sw_mg : index")
        e("  %sw_lin = index.add %sw_ly, %wg_x0 : index")
        e(f"  %sw_g = index.constant {SWZ} : index")
        e("  %sw_gt = index.mul %sw_g, %token_tiles : index")
        e("  %sw_grp = index.div %sw_lin, %sw_gt : index")
        e("  %sw_in = index.rem %sw_lin, %sw_gt : index")
        e("  %sw_rg0 = index.mul %sw_grp, %sw_g : index")
        e("  %sw_rgi = index.rem %sw_in, %sw_g : index")
        e("  %wg_x = index.add %sw_rg0, %sw_rgi : index")
        e("  %wg_y = index.div %sw_in, %sw_g : index")
    else:
        e("  %wg_x = kernel.workgroup.id<x> : index")
        e("  %wg_y = kernel.workgroup.id<y> : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %wave = index.div %tid, %c32 : index")
    e(f"  %cwn = index.constant {WN} : index")
    e("  %wr = index.div %wave, %cwn : index")
    e("  %wt = index.rem %wave, %cwn : index")
    e(f"  %wg_row = index.mul %wg_x, %c{BM} : index")
    e(f"  %ctm = index.constant {TM} : index")
    e(f"  %ctn = index.constant {TN} : index")
    e("  %wr_off = index.mul %wr, %ctm : index")
    e("  %wt_off = index.mul %wt, %ctn : index")
    e("  %m_origin = index.add %wg_row, %wr_off : index")
    e("  %wtb = index.mul %wg_y, %cwtok : index")
    e("  %token_base = index.add %wtb, %wt_off : index")
    # decode lane map: lane tid -> weight row tid % BM, slot tid / BM decodes
    # groups [slot*GPL, slot*GPL+GPL) of the phase; slots >= GPP/GPL idle.
    e(f"  %l64 = index.rem %tid, %c{BM} : index")
    e(f"  %slot = index.div %tid, %c{BM} : index")
    e(f"  %drow = index.min %l64, %c{BM - 1} : index")
    if FRAG:
        e(f"  %cksub_fm = index.constant {ksub} : index")
        e("  %drow_b = index.div %drow, %c16 : index")
        e("  %drow_bk = index.mul %drow_b, %cksub_fm : index")
        e("  %drow_r = index.rem %drow, %c16 : index")
        e("  %drow_fm = index.add %drow_bk, %drow_r : index")
    e("  %drow_i = index.cast %drow : index to i32")
    e("  %wg_row_i = index.cast %wg_row : index to i32")
    e("  %grow_i = scalar.addi %wg_row_i, %drow_i : i32")
    e("  %k_blocks_i = index.cast %k_blocks : index to i32")
    e("  %bpr_i = scalar.muli %k_blocks_i, %cbbi : i32")
    e("  %row_off_i = scalar.muli %grow_i, %bpr_i : i32")
    e(f"  %cslots = index.constant {slots} : index")
    e("  %slot_c = index.min %slot, %cslots : index")
    e("  %decoder = index.cmp ult, %slot, %cslots : index")
    e("  %slot_i = index.cast %slot_c : index to i32")
    e(f"  %cgpl = scalar.constant {G.GPL} : i32")
    e("  %gl_i = scalar.muli %slot_i, %cgpl : i32")
    e("  %kphases = index.mul %k_blocks, %cph : index")
    # activation staging map: lane tid stages segments [aseg0, aseg0+aspl) of
    # token row tid % BN of the tile
    e(f"  %atok = index.rem %tid, %c{BN} : index")
    if FRAG:
        e(f"  %cksub_a = index.constant {ksub} : index")
        e("  %atok_b = index.div %atok, %c16 : index")
        e("  %atok_bk = index.mul %atok_b, %cksub_a : index")
        e("  %atok_r = index.rem %atok, %c16 : index")
        e("  %atok_fm = index.add %atok_bk, %atok_r : index")
    e(f"  %apart = index.div %tid, %c{BN} : index")
    e(f"  %caspl8 = index.constant {8 * aspl} : index")
    e("  %aseg0 = index.mul %apart, %caspl8 : index")
    e("  %atok_g = index.add %wtb, %atok : index")
    e("  %arow_g = index.mul %atok_g, %apitch : index")
    L.extend(F["setup"]())
    e("  %z8s = scalar.constant 0 : i8")
    e("  %z8v = vector.splat %z8s : vector<8xi8>")
    e("  %zeros = vector.constant 0.0 : vector<8xf32>")
    e(f"  %init = vector.fragment<init> %zeros shape [%m, %n] : {V8}")
    NA = FM * FN
    types = ", ".join([V8] * NA)

    def a_loads(p, kbase):
        """This lane's token row of the next phase's activation tile, as aseg
        16-byte vectors (clamped: the extra iteration's loads are in bounds)."""
        vals = []
        e(f"    %{p}ab = index.add %arow_g, {kbase} : index")
        e(f"    %{p}ao = index.add %{p}ab, %aseg0 : index")
        for sg in range(aspl):
            e(f"    %{p}ac{sg}0 = index.constant {8 * sg} : index")
            e(f"    %{p}aq{sg} = index.add %{p}ao, %{p}ac{sg}0 : index")
            e(f"    %{p}aqc{sg} = index.min %{p}aq{sg}, %a_last8 : index")
            e(f"    %{p}av{sg} = vector.load %a_flat[%{p}aqc{sg}] : view<[%a_total]xf16> -> vector<8xf16>")
            vals.append((f"%{p}av{sg}", "vector<8xf16>"))
        return vals

    # prefetch phase 0: weight bytes and activation row
    L0, wv0 = loads("pf_", "%row_off_i", "%gl_i")
    L.extend(L0)
    orig0 = wv0
    wv0 = G.pack_vals(e, wv0, "0")
    av0 = a_loads("pa_", "%c0")
    carried = wv0 + av0
    ca = ", ".join(f"%a{i} = %init : {V8}" for i in range(NA))
    ca += ", " + ", ".join(f"%cv{x} = {nm} : {ty}" for x, (nm, ty) in enumerate(carried))
    carried_t = types + ", " + ", ".join(ty for _, ty in carried)
    res = ", ".join(f"%acc{i}" for i in range(NA)) + ", " + ", ".join(f"%cvo{x}" for x in range(len(carried)))
    e("  " + res + f" = scf.for %kp = [%c0 to %kphases step %c1]({ca}) -> ({carried_t}) {{")
    e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e("    %kb = index.div %kp, %cph : index")
    e("    %ph = index.rem %kp, %cph : index")
    e("    %ph_i = index.cast %ph : index to i32")
    e("    %phg_i = scalar.muli %ph_i, %cgppi : i32")
    e("    %gb_i = scalar.addi %phg_i, %gl_i : i32")
    e("    %kb_k = index.mul %kp, %cksub : index")
    cur_w = [(f"%cv{x}", ty) for x, (_, ty) in enumerate(wv0)]
    cur_a = [f"%cv{len(wv0) + x}" for x in range(len(av0))]
    # decode (only the decoding slots) into the weight tile
    e("    scf.if %decoder {")
    names = G.unpack_vals(e, cur_w, orig0)
    if FRAG:
        L.extend(frag_stores(compute(names, "%gb_i"), ksub))
    else:
        L.extend(compute(names, "%gb_i"))
    e("    }")
    # stage the activation row into the LDS activation tile
    for sg, nm in enumerate(cur_a):
        e(f"    %as{sg}c = index.constant {8 * sg} : index")
        e(f"    %as{sg} = index.add %aseg0, %as{sg}c : index")
        if FRAG:
            # token t, k = %as: block (t/16, k/16), row t%16, column k%16
            e(f"    %afq{sg} = index.div %as{sg}, %c16 : index")
            e(f"    %afr{sg} = index.mul %afq{sg}, %c16 : index")
            e(f"    %afrow{sg} = index.add %atok_fm, %afr{sg} : index")
            e(f"    %afc{sg} = index.rem %as{sg}, %c16 : index")
            e(f"    vector.store {nm}, %al_fm[%afrow{sg}, %afc{sg}] : vector<8xf16>, view<{BN * ksub // 16}x16xf16>")
        else:
            e(f"    vector.store {nm}, %al_rows[%atok, %as{sg}] : vector<8xf16>, view<{BN}x{arow}xf16>")
    # next phase's loads. FENCE=1 keeps them below this phase's LDS stores: the
    # scheduler otherwise hoists them above the stores and then has to wait
    # vmcnt(0) -- for the loads it just issued -- before the first store (31%
    # of IQ4_XS wave time in the ATT trace).
    if FENCE:
        e("    scf.schedule.fence")
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
    nxt = G.pack_vals(e, nxt, "n")
    e("    %kk_n = index.mul %kp_n, %cksub : index")
    anx = a_loads("na_", "%kk_n")
    e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    cb = ", ".join(f"%b{i} = %a{i} : {V8}" for i in range(NA))
    e("    " + ", ".join(f"%r{i}" for i in range(NA)) + f" = scf.for %ks = [%c0 to %cksub step %c16]({cb}) -> ({types}) {KPOL} {{")
    for i in range(FM):
        e(f"      %lr{i} = index.add %wr_off, %c{16 * i} : index")
        if FRAG:
            # block (lr/16, ks/16) starts at row (lr/16)*KSUB + ks
            e(f"      %lrb{i} = index.div %lr{i}, %c16 : index")
            e(f"      %lrk{i} = index.mul %lrb{i}, %cksub : index")
            e(f"      %lrow{i} = index.add %lrk{i}, %ks : index")
            e(f"      %lhs{i} = vector.fragment.load<lhs> %wl_fm[%lrow{i}, %c0] shape [%m, %k] : view<{BM * ksub // 16}x16xf16> -> {VF}")
        else:
            e(f"      %lhs{i} = vector.fragment.load<lhs> %wl_view[%lr{i}, %ks] shape [%m, %k] : view<{BM}x{G.ROWP}xf16> -> {VF}")
    for j in range(FN):
        e(f"      %tc{j} = index.add %wt_off, %c{16 * j} : index")
        if FRAG:
            e(f"      %tcb{j} = index.div %tc{j}, %c16 : index")
            e(f"      %tck{j} = index.mul %tcb{j}, %cksub : index")
            e(f"      %tcol{j} = index.add %tck{j}, %ks : index")
            e(f"      %rhs{j} = vector.fragment.load<rhs> %al_fmt[%c0, %tcol{j}] shape [%k, %n] : view<16x{BN * ksub // 16}xf16, %fm_lay> -> {VF}")
        else:
            e(f"      %rhs{j} = vector.fragment.load<rhs> %al_t[%ks, %tc{j}] shape [%k, %n] : view<{ksub}x{BN}xf16, %al_layout> -> {VF}")
    for i in range(FM):
        for j in range(FN):
            n = i * FN + j
            e(f"      %n{n} = vector.mma %lhs{i}, %rhs{j}, %b{n} : {VF}, {VF}, {V8}")
    e("      scf.yield " + ", ".join(f"%n{i}" for i in range(NA)) + f" : {types}")
    e("    }")
    e("    scf.yield " + ", ".join(f"%r{i}" for i in range(NA)) + ", "
      + ", ".join(nm for nm, _ in nxt + anx) + f" : {carried_t}")
    e("  }")
    if sw:
        swiglu_epilogue(e, arow)
        e("  kernel.return")
        e("}")
        return "\n".join(L) + "\n"
    if EPI_LDS and TM == 32:
        lds_epilogue(e, kr, V8)
        e("  kernel.return")
        e("}")
        return "\n".join(L) + "\n"
    e("  %out_layout = encoding.layout.strided [%c1, %m_rows] : encoding<layout>")
    e("  %out_t_view = buffer.view %output_na[%base] : buffer -> view<[%m_rows]x[%tokens]xf32, %out_layout>")
    if kr:
        e("  %res_t_view = buffer.view %resid_na[%base] : buffer -> view<[%m_rows]x[%tokens]xf32, %out_layout>")
    for i in range(FM):
        e(f"  %or{i} = index.add %m_origin, %c{16 * i} : index")
    for j in range(FN):
        e(f"  %ot{j} = index.add %token_base, %c{16 * j} : index")
    for i in range(FM):
        for j in range(FN):
            a = i * FN + j
            val = f"%acc{a}"
            if kr:
                e(f"  %rf{a} = vector.fragment.load<result> %res_t_view[%or{i}, %ot{j}] shape [%m, %n] : view<[%m_rows]x[%tokens]xf32, %out_layout> -> {V8}")
                e(f"  %rs{a} = vector.addf %rf{a}, %acc{a} : {V8}")
                val = f"%rs{a}"
            e(f"  vector.fragment.store<result> {val}, %out_t_view[%or{i}, %ot{j}] shape [%m, %n] : {V8}, view<[%m_rows]x[%tokens]xf32, %out_layout>")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def frag_stores(lines, ksub):
    """Rewrite the shared decode helpers' weight-tile stores
    (vector.store V, %wl_view[%drow, COL] : vector<Nxf16>, view<..>) into the
    fragment-major tile: element (r, c) is row (r/16)*KSUB + (c/16)*16 + r%16,
    column c%16 of %wl_fm. A store never crosses a 16-column block."""
    import re
    pat = re.compile(r"^(\s*)vector\.store (\S+), %wl_view\[%drow, (\S+)\] : (vector<\d+xf16>), view<[^>]*>$")
    out, n = [], 0
    for l in lines:
        m = pat.match(l)
        if not m:
            assert "%wl_view" not in l, l
            out.append(l)
            continue
        ind, val, col, vt = m.groups()
        t = f"fs{n}"
        n += 1
        out += [f"{ind}%{t}q = index.div {col}, %c16 : index",
                f"{ind}%{t}o = index.mul %{t}q, %c16 : index",
                f"{ind}%{t}r = index.add %drow_fm, %{t}o : index",
                f"{ind}%{t}c = index.rem {col}, %c16 : index",
                f"{ind}vector.store {val}, %wl_fm[%{t}r, %{t}c] : {vt}, view<{BM * ksub // 16}x16xf16>"]
    assert n > 0, "no weight-tile stores found to rewrite"
    return out


def lds_epilogue(e, kr, V8):
    """out[t*m + r] (+ resid) for the wave's TM x TN tile, one 16-token column
    of fragments at a time: fragments -> LDS slab (element (r, t) at t*TM + r)
    -> each lane reads 16 contiguous rows of one token -> 4 b128 stores. Two
    lanes cover a token's TM=32 rows (128 contiguous bytes). Same values, and
    for kres the same resid + acc add, so bit-identical."""
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e(f"  %es_ctm = index.constant {TM} : index")
    e("  %es_lay = encoding.layout.strided [%c1, %es_ctm] : encoding<layout>")
    e(f"  %es_wb = index.constant {TM * 16 * 4} : index")
    e("  %es_off_i = index.mul %wave, %es_wb : index")
    e("  %es_off = index.cast %es_off_i : index to offset")
    e(f"  %es_view = buffer.view %al[%es_off] : buffer -> view<{TM}x16xf32, %es_lay>")
    e(f"  %es_flat = buffer.view %al[%es_off] : buffer -> view<{TM * 16}xf32>")
    e("  %out_flat = buffer.view %output_na[%base] : buffer -> view<[%out_total]xf32>")
    if kr:
        e("  %res_flat = buffer.view %resid_na[%base] : buffer -> view<[%out_total]xf32>")
    e("  %es_lane = index.rem %tid, %c32 : index")
    e("  %es_t = index.div %es_lane, %c2 : index")
    e("  %es_h0 = index.rem %es_lane, %c2 : index")
    e("  %es_h = index.mul %es_h0, %c16 : index")
    e(f"  %es_tt = index.mul %es_t, %es_ctm : index")
    e("  %es_rd = index.add %es_tt, %es_h : index")
    e("  %es_row = index.add %m_origin, %es_h : index")
    for j in range(FN):
        for i in range(FM):
            e(f"  %es_r{i}_{j} = index.constant {16 * i} : index")
            e(f"  vector.fragment.store<result> %acc{i * FN + j}, %es_view[%es_r{i}_{j}, %c0] shape [%m, %n] : {V8}, view<{TM}x16xf32, %es_lay>")
        e(f"  %es_tc{j} = index.constant {16 * j} : index")
        e(f"  %es_tk{j}0 = index.add %token_base, %es_tc{j} : index")
        e(f"  %es_tk{j} = index.add %es_tk{j}0, %es_t : index")
        e(f"  %es_tm{j} = index.mul %es_tk{j}, %m_rows : index")
        e(f"  %es_ob{j} = index.add %es_tm{j}, %es_row : index")
        for q in range(4):
            e(f"  %es_q{j}_{q}c = index.constant {4 * q} : index")
            e(f"  %es_ri{j}_{q} = index.add %es_rd, %es_q{j}_{q}c : index")
            e(f"  %es_v{j}_{q} = vector.load %es_flat[%es_ri{j}_{q}] : view<{TM * 16}xf32> -> vector<4xf32>")
            e(f"  %es_oi{j}_{q} = index.add %es_ob{j}, %es_q{j}_{q}c : index")
            val = f"%es_v{j}_{q}"
            if kr:
                e(f"  %es_rf{j}_{q} = vector.load %res_flat[%es_oi{j}_{q}] : view<[%out_total]xf32> -> vector<4xf32>")
                e(f"  %es_rs{j}_{q} = vector.addf %es_rf{j}_{q}, %es_v{j}_{q} : vector<4xf32>")
                val = f"%es_rs{j}_{q}"
            e(f"  vector.store {val}, %out_flat[%es_oi{j}_{q}] : vector<4xf32>, view<[%out_total]xf32>")


def swiglu_epilogue(e, arow):
    """out[t*m + r] = f16(silu(gate[t*m + r]) * acc[r][t]), the chained kernel's
    scalar ops in its order (bit-identity), as in gen_gemm_shared: an f16 result
    fragment store ignores the token-major strided layout, so each 16-row x
    ES-token slab goes through a per-wave f32 LDS tile (f32 result stores honour
    layouts) and a real loop walks it with lane-contiguous rows. The slab is half
    the wave's tokens so all waves' tiles fit in the activation tile's LDS."""
    ES = next(x for x in (32, 16) if x <= TN and NWAVE * 16 * x * 4 <= BN * arow * 2)
    assert TN % ES == 0
    V8 = "vector<8xf32>"
    e("  %ep_lay = encoding.layout.strided [%c1, %c16] : encoding<layout>")
    e(f"  %ep_wbytes = index.constant {16 * ES * 4} : index")
    e("  %ep_off_i = index.mul %wave, %ep_wbytes : index")
    e("  %ep_off = index.cast %ep_off_i : index to offset")
    e(f"  %ep_view = buffer.view %al[%ep_off] : buffer -> view<16x{ES}xf32, %ep_lay>")
    e(f"  %ep_flat = buffer.view %al[%ep_off] : buffer -> view<{16 * ES}xf32>")
    e("  %gate_view = buffer.view %gate_na[%base] : buffer -> view<[%out_total]xf32>")
    e("  %out_h = buffer.view %output_na[%base] : buffer -> view<[%out_total]xf16>")
    e("  %negone = scalar.constant -1.0 : f32")
    e("  %one = scalar.constant 1.0 : f32")
    e("  %out_last = index.sub %out_total, %c1 : index")
    e("  %lane = index.rem %tid, %c32 : index")
    e(f"  %ep_n = index.constant {16 * ES // 32} : index")
    e(f"  %ep_last = index.constant {16 * ES - 1} : index")
    for i in range(FM):
        e(f"  %sr{i} = index.add %m_origin, %c{16 * i} : index")
        for h in range(TN // ES):
            q = f"{i}_{h}"
            e(f"  %st{q}c = index.constant {h * ES} : index")
            e(f"  %st{q} = index.add %token_base, %st{q}c : index")
            e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            for jj in range(ES // 16):
                j = h * ES // 16 + jj
                e(f"  %sc{q}_{jj} = index.constant {16 * jj} : index")
                e(f"  vector.fragment.store<result> %acc{i * FN + j}, %ep_view[%c0, %sc{q}_{jj}] shape [%m, %n] : {V8}, view<16x{ES}xf32, %ep_lay>")
            e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
            e(f"  %eps{q} = scf.for %ee{q} = [%c0 to %ep_n step %c1](%em{q} = %c0 : index) -> (index) {{")
            e(f"    %e32_{q} = index.mul %ee{q}, %c32 : index")
            e(f"    %ef0_{q} = index.add %e32_{q}, %lane : index")
            e(f"    %ef_{q} = index.min %ef0_{q}, %ep_last : index")
            e(f"    %er_{q} = index.rem %ef_{q}, %c16 : index")
            e(f"    %et_{q} = index.div %ef_{q}, %c16 : index")
            e(f"    %v_{q} = view.load %ep_flat[%ef_{q}] : view<{16 * ES}xf32> -> f32")
            e(f"    %grow_{q} = index.add %sr{i}, %er_{q} : index")
            e(f"    %gtok_{q} = index.add %st{q}, %et_{q} : index")
            e(f"    %gto_{q} = index.mul %gtok_{q}, %m_rows : index")
            e(f"    %gix0_{q} = index.add %gto_{q}, %grow_{q} : index")
            e(f"    %gix_{q} = index.min %gix0_{q}, %out_last : index")
            e(f"    %g_{q} = view.load %gate_view[%gix_{q}] : view<[%out_total]xf32> -> f32")
            e(f"    %ng_{q} = scalar.mulf %g_{q}, %negone : f32")
            e(f"    %ex_{q} = scalar.expf<afn> %ng_{q} : f32")
            e(f"    %dn_{q} = scalar.addf %one, %ex_{q} : f32")
            e(f"    %iv_{q} = scalar.divf %one, %dn_{q} : f32")
            e(f"    %sg_{q} = scalar.mulf %g_{q}, %iv_{q} : f32")
            e(f"    %ac_{q} = scalar.mulf %sg_{q}, %v_{q} : f32")
            e(f"    %h_{q} = scalar.fptrunc %ac_{q} : f32 to f16")
            e(f"    view.store %h_{q}, %out_h[%gix_{q}] : f16, view<[%out_total]xf16>")
            e(f"    scf.yield %em{q} : index")
            e("  }")


def main():
    fmt = sys.argv[1]
    kind = os.environ.get("YAH_TG_KIND", "kstore")
    out = sys.argv[2] if len(sys.argv) > 2 else f"yah_tile_{fmt}_{kind}.loom"
    open(out, "w").write(gen(fmt, kind))
    print(out)


if __name__ == "__main__":
    main()
