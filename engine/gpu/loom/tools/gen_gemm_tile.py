#!/usr/bin/env python3
"""Tile GEMM yah_ffn_gemm_<fmt>[_swiglu|_kres|_kqg]: wave32, many waves, decoded weights and activations both in LDS.

Workgroup: BM rows x BN tokens (128 x 256) over WM x WN wave32 waves (4 x 4, or 4 x 2 for WAVE_FMTS), each owning a TM x TN tile.
Grid: (m_tiles / ROWGRP, token_tiles); geometry() gives the dispatch.txt tile.
Bindings and layouts as gen_gemm_decode.py, plus gate_out for kqg.
The decode arithmetic and per-accumulator MMA order match gen_gemm_decode.py, so the output is bit-identical to it.
Per K phase the decoded weight tile (BM x KSUB) and the activation tile (BN x KSUB) sit in LDS, so the MMA loop reads only LDS.
The next phase's weight bytes and activation rows load into registers during this phase and go to LDS after it.
Many waves per SIMD hide the load latency that the wave64 shared kernel (about 2 waves per SIMD) leaves exposed.
"""
import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_gemm_decode as G  # noqa: E402

# Default workgroup: 128 x 256 over 4 x 4 waves (WAVE_FMTS: 4 x 2).
# 256 x 256 does not fit: 64 KB of tiles plus the IQ grid table in LDS is over gfx11's 64 KB per workgroup.
WS = 32
APAD = 8                      # f16 of padding per LDS activation row
# Decode-ahead: after the barrier the decoding waves decode the next phase into a second weight tile.
# So their VALU overlaps every wave's MMAs of the current phase. Needs two weight tiles in LDS: KSUB=32 at 128 x 256.
# Off for the IQ3 formats: their LDS grid lookups contend with the MMA fragment loads.
# Off for Q5_K: its read-ahead loads meet s_waitcnt vmcnt(0) drains inside the K loop, which serialize the prefetch.
DECAHEAD_FMTS = ("iq4xs", "q4k", "q6k")
# Short-K residual GEMMs keep the plain schedule: with decode-ahead, 45% of wave time is s_waitcnt vmcnt(0) in the K loop.
# These full drains serialize the read-ahead.
DECAHEAD_SKIP = {("iq4xs", "kres", 24), ("q4k", "kres", 24)}
# The swiglu epilogue through lds_epilogue (one barrier, wave-private slabs, 4-row vector loads/stores).
# swiglu_epilogue issues one dependent gate load per element in a rolled loop; with one workgroup per WGP nothing hides it.
# IQ4_XS keeps swiglu_epilogue: neutral at 2x the code.
SWEPI_FMTS = ("iq3s", "iq3xxs")
# STAGGER: in the first round the second workgroup on each WGP runs STAGGER barriers before it starts.
# So co-resident workgroups drift out of lockstep; else on short-K kres they all hit the residual epilogue at once.
# 8000 barriers is ~380k cycles of offset, past the ~150-200k cycle epilogue burst. Results are unchanged.
# Only grids of >= STG_MINWG workgroups (8 rounds at 2 per WGP) stagger, so the delay is a one-time cost.
STAGGER = 8000
STG_MINWG = 320
STG_NWGP = 20                 # WGPs on gfx1151
# rhs-outer MMA order with a fence every n rhs groups, per format (fewer live fragments at 4 x 2)
RHSO_FMTS = {"q4k": 1, "q5k": 1}

# f16 of padding per decoded weight row (APAD likewise per activation row).
# Unpadded rows are 128 B apart at KSUB=64, so a 16-lane lhs fragment load hits 2 bank groups (8-way conflicts).
WPAD = 8
# KSL: the k steps of a phase straight-line in program order (no read-ahead).
# Low CSE then shares the fragment loads' address math (13 VALU per 8 WMMAs in the rolled loop).
# A fence keeps step s+1's loads after step s's MMAs.
# Off for Q5_K: the latch copies' vmcnt(0) sits between the steps; with DECLOAD, at the 144-VGPR cap, ~48 moves appear.
# Q6_K, Q2_K, Q8_0 and IQ2_* are not measured.
KSL_FMTS = ("iq4xs", "q4k", "iq3s", "iq3xxs", "q3k")
# DECLOAD (on where KSL is): under decode-ahead only the decoding waves issue the phase's weight loads, not every wave.
# With KSL it also moves the prefetch's latch copies (and their vmcnt(0)) from between the k steps to after the last MMA.
# EPAD: pad of the LDS epilogue slab's token pitch (f32).
# With pitch TM the 16 lanes storing a fragment row are 128 B apart (one or two banks).
EPAD = 4
# Formats that run 4 x 2 waves (32 x 128 per wave) on the 128 x 256 geometry: fewer fragment loads per WMMA (1.5 -> 1.25).
# 2 x 4 (64 x 64) has fewer instructions still but drops off the issue bound (exposed latency).
# Q4_K / Q5_K fit in VGPRs at 4 x 2 only with Q4FMIX, IQ3_S only with the word-path decode (w3).
# IQ3_XXS swiglu at 4 x 2 needs the LDS epilogue (SWEPI_FMTS).
WAVE_FMTS = ("iq4xs", "iq3xxs", "q3k", "iq3s", "q4k", "q5k")


@dataclass(frozen=True)
class Tile:
    """The knobs of one tile GEMM. default_tile() gives the shipped choice; a tuner may pick another legal Tile.
    No knob changes the per-accumulator MMA order or the decode arithmetic, so every legal Tile gives bit-identical output."""
    bm: int = 128              # weight rows per workgroup
    bn: int = 256              # tokens per workgroup
    wm: int = 4                # waves along rows
    wn: int = 4                # waves along tokens
    ksub: int = 64             # K per phase
    decahead: bool = False     # decode the next phase during this phase's MMAs (two weight tiles)
    ksl: bool = False          # straight-line k steps (KSL_FMTS)
    decload: bool = False      # under decode-ahead only the decoding waves load weights (DECLOAD)
    rhs_outer: bool = False    # rhs-outer MMA order
    rhs_fence: int = 0         # fence every n rhs groups under rhs_outer
    w3: bool = False           # IQ3 word-path decode (gen_gemm_decode VDEC_W, IQ3_U8F, VDECW_FR)
    q4fmix: bool = False       # Q4_K/Q5_K subtract-and-narrow through v_fma_mix (gen_gemm_decode Q4FMIX)
    swepi: bool = False        # swiglu through lds_epilogue
    stagger: int = STAGGER     # barriers the second workgroup per WGP waits in the first round

    @property
    def tm(self):
        return self.bm // self.wm

    @property
    def tn(self):
        return self.bn // self.wn

    @property
    def nwave(self):
        return self.wm * self.wn

    @property
    def lanes(self):
        return WS * self.nwave

    @property
    def rowgrp(self):
        """m_tiles per workgroup."""
        return self.bm // 16

    @property
    def apl(self):
        """Lanes staging one token row of the activation tile."""
        return self.lanes // self.bn


def default_tile(fmt, kind, kb, geom=None):
    """The shipped Tile for fmt / kind at k_blocks kb; geom=(BM, BN, WM, WN) overrides the geometry (16-row tiles)."""
    if geom:
        bm, bn, wm, wn = geom
    else:
        bm, bn, wm, wn = (128, 256, 4, 2) if fmt in WAVE_FMTS else (128, 256, 4, 4)
    # the decoding lanes must be whole waves (a wave-uniform branch): not so for the 16-row tiles, which keep the plain schedule
    decahead = fmt in DECAHEAD_FMTS and (fmt, kind, kb) not in DECAHEAD_SKIP and bm % WS == 0
    # IQ3 word-path decode, bit-identical: IQ3_XXS, and IQ3_S at 4 x 2.
    # At 4 x 4 IQ3_S gains nothing: the longer dependent chain is exposed between barriers, where every wave decodes at once.
    w3 = fmt == "iq3xxs" or (fmt == "iq3s" and (wm, wn) == (4, 2))
    return Tile(bm, bn, wm, wn,
                # KSUB=64 (32 for decode-ahead's two weight tiles): at 128 the 128 x 256 tiles need ~104 KB of LDS
                ksub=32 if decahead else 64,
                decahead=decahead, ksl=fmt in KSL_FMTS, decload=fmt in KSL_FMTS,
                rhs_outer=fmt in RHSO_FMTS, rhs_fence=RHSO_FMTS.get(fmt, 0), w3=w3,
                # Q4_K: this lets Q4_K run at 4 x 2 without spills
                q4fmix=fmt in ("q4k", "q5k"),
                swepi=kind == "swiglu" and fmt in SWEPI_FMTS and bm // wm == 32)


def check(t):
    """Raise ValueError if t cannot be emitted (shape rules only; the compiler rejects LDS or VGPR overflow)."""
    rules = [
        (t.bm % (16 * t.wm) == 0 and t.bn % (16 * t.wn) == 0, "per-wave tile is not whole 16 x 16 fragments"),
        (t.lanes >= t.bm and t.lanes % t.bn == 0, "lanes do not cover the weight rows and token rows"),
        (t.ksub in (32, 64, 128) and (not t.decahead or t.bm % WS == 0), "KSUB or decode-ahead geometry"),
        (t.ksub // 32 * t.bm <= t.lanes, "not enough lanes to decode a phase in one pass"),
        ((t.ksub // 8) % t.apl == 0, "activation row does not split evenly over its lanes"),
        (not t.swepi or t.tm == 32, "the LDS swiglu epilogue needs 32 rows per wave"),
    ]
    for ok, why in rules:
        if not ok:
            raise ValueError(f"{t}: {why}")


def configure(fmt, t):
    """Set gen_gemm_decode's decode geometry and switches for fmt under tile t."""
    G.KSUB = t.ksub
    G.PAD = WPAD
    G.ROWP = t.ksub + G.PAD
    G.PH = 256 // t.ksub
    G.GPP = t.ksub // 32
    # one group per decoding lane (q4k/q5k pick the nibble at run time)
    G.GPL = 1
    G.Q4_HDR = fmt in ("q4k", "q5k")
    G.VDEC_W = G.IQ3_U8F = G.VDECW_FR = t.w3
    G.Q4FMIX = t.q4fmix
    G.LR = t.bm
    G.NW = t.nwave // 2        # table-staging stride 64*NW = LANES


def gen(fmt, kind="kstore", tile=None, masked=False):
    """Return the kernel text for fmt; kind as gen_gemm_decode.gen() plus "kqg". tile defaults to default_tile().
    masked: the token count (config "tokens") need not be a multiple of BN. The last token tile clamps its activation loads
    and skips every per-token load and store past it, so the valid tokens are computed exactly as unmasked."""
    t = tile or default_tile(fmt, kind, 0)
    check(t)
    if masked and t.tm != 32:
        raise ValueError(f"{t}: a masked token tile needs the LDS epilogue (32 rows per wave)")
    configure(fmt, t)
    return _gen(fmt, kind, t, masked)


def _gen(fmt, kind, t, masked):
    BM, BN, WM, WN, TM, TN = t.bm, t.bn, t.wm, t.wn, t.tm, t.tn
    FM, FN, NWAVE, LANES, ROWGRP, APL = TM // 16, TN // 16, t.nwave, t.lanes, t.rowgrp, t.apl
    DECAHEAD, KSL, DECLOAD, RHS_OUTER, RHS_FENCE = t.decahead, t.ksl, t.decload, t.rhs_outer, t.rhs_fence
    F = G.FMTS[fmt]
    ksub = t.ksub
    bb, (loads, compute) = F["bb"], F["decode"]
    kr = kind == "kres"
    sw = kind == "swiglu"
    # kqg: the attention q projection (rows = heads x [256 q | 256 gate]) writes q and gate to [tokens][heads*256] buffers.
    # Same values as the separate yah_unpack_qg pass, so bit-identical.
    qg = kind == "kqg"
    bufs = (["weight"] + F["extra"] + ["input"] + (["gate"] if sw else []) + (["resid"] if kr else [])
            + ["wstage", "ostage", "output"] + (["gate_out"] if qg else []))
    sym = f"yah_ffn_gemm_{fmt}" + ("_swiglu" if sw else "") + ("_kres" if kr else "") + ("_kqg" if qg else "")
    slots = G.GPP                   # decoding lane groups of BM per phase
    arow = ksub + APAD              # f16 per LDS activation row
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
    e(f"amdgpu.target<gfx1151> @yah_tile_w32 {{subgroup_size = {WS}}}")
    e("")
    for c in ("m_tiles", "k_blocks", "token_tiles") + (("tokens",) if masked else ()):
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
    for v in sorted({0, 1, 2, 4, 6, 7, 8, 16, 32, 48, 63, 64, 80, 96, 112, 127, 128, 224, 255, 256, 512, BM, BM - 1, BN}):
        e(f"  %c{v} = index.constant {v} : index")
    # the same i32 constants gen_gemm_decode defines (q8_0 needs 18 and 34)
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
    # q8_0's k_blocks counts 32-wide blocks; the decode walks 256-wide ones (bb=272 = 8 x 34), as in gen_gemm_decode.
    # Without the division the kernel reads 8x past the weights and hangs the ring.
    kdiv = F.get("kdiv", 1)
    if kdiv == 1:
        e(f"  %k_blocks = config.get @{sym}.k_blocks : index")
    else:
        e(f"  %k_blocks_cfg = config.get @{sym}.k_blocks : index")
        e(f"  %ckdiv = index.constant {kdiv} : index")
        e("  %k_blocks = index.div %k_blocks_cfg, %ckdiv : index")
    e(f"  %token_tiles = config.get @{sym}.token_tiles : index")
    e("  %ktot = index.mul %k_blocks, %c256 : index")
    if masked:
        e(f"  %tokens = config.get @{sym}.tokens : index")
    else:
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
    e("  %cagpad = index.constant 0 : index")
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
    e(f"  %wl_bytes = index.constant {BM * G.ROWP * 2 * (2 if DECAHEAD else 1)} : offset")
    e(f"  %wl_tb = index.constant {BM * G.ROWP * 2} : index")
    e("  %wl = buffer.alloca<workgroup> align(16) %wl_bytes : buffer")
    e(f"  %wl_view = buffer.view %wl[%base] : buffer -> view<{BM}x{G.ROWP}xf16>")
    # the LDS epilogue's wave-private TM x 16 f32 slabs live in the activation tile
    al_bytes = BN * arow * 2
    slabs = NWAVE * (TM + EPAD) * 16 * 4
    if TM == 32:
        al_bytes = max(al_bytes, slabs)
    e(f"  %al_bytes = index.constant {al_bytes} : offset")
    e("  %al = buffer.alloca<workgroup> align(16) %al_bytes : buffer")
    e(f"  %al_rows = buffer.view %al[%base] : buffer -> view<{BN}x{arow}xf16>")
    e(f"  %carow = index.constant {arow} : index")
    e(f"  %cksubi = index.constant {ksub} : index")
    e("  %al_layout = encoding.layout.strided [%c1, %carow] : encoding<layout>")
    e(f"  %al_t = buffer.view %al[%base] : buffer -> view<{ksub}x{BN}xf16, %al_layout>")
    e("  %wg_x = kernel.workgroup.id<x> : index")
    e("  %wg_y = kernel.workgroup.id<y> : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e(f"  %wave = index.div %tid, %c{WS} : index")
    e(f"  %cwn = index.constant {WN} : index")
    e("  %wr = index.div %wave, %cwn : index")
    e("  %wt = index.rem %wave, %cwn : index")
    e("  %stg_row0 = index.cmp eq, %wg_y, %c0 : index")
    # the second workgroup on each WGP in the first round: linear dispatch ids [NWGP, 2*NWGP) (co-resident pairs are (i, i + 20))
    e("  %stg_gx = kernel.workgroup.count<x> : index")
    e("  %stg_rx = kernel.workgroup.id<x> : index")
    e("  %stg_ry = kernel.workgroup.id<y> : index")
    e("  %stg_l0 = index.mul %stg_ry, %stg_gx : index")
    e("  %stg_lin = index.add %stg_l0, %stg_rx : index")
    e("  %stg_gy = kernel.workgroup.count<y> : index")
    e("  %stg_tot = index.mul %stg_gx, %stg_gy : index")
    e(f"  %stg_minwg = index.constant {STG_MINWG} : index")
    e("  %stg_big = index.cmp uge, %stg_tot, %stg_minwg : index")
    e(f"  %stg_w = index.constant {STG_NWGP} : index")
    e(f"  %stg_w2 = index.constant {2 * STG_NWGP} : index")
    e("  %stg_ge = index.cmp uge, %stg_lin, %stg_w : index")
    e("  %stg_lt = index.cmp ult, %stg_lin, %stg_w2 : index")
    e(f"  %stg_n = index.constant {t.stagger} : index")
    e("  %stg_n1 = scf.select %stg_ge, %stg_n, %c0 : index")
    e("  %stg_n2 = scf.select %stg_lt, %stg_n1, %c0 : index")
    e("  %stg_iters = scf.select %stg_big, %stg_n2, %c0 : index")
    # workgroup-uniform trip count: every wave of the workgroup takes it
    e("  scf.for %stg_i = [%c0 to %stg_iters step %c1] {")
    e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e("  }")
    e(f"  %wg_row = index.mul %wg_x, %c{BM} : index")
    e(f"  %ctm = index.constant {TM} : index")
    e(f"  %ctn = index.constant {TN} : index")
    e("  %wr_off = index.mul %wr, %ctm : index")
    e("  %wt_off = index.mul %wt, %ctn : index")
    e("  %m_origin = index.add %wg_row, %wr_off : index")
    e("  %wtb = index.mul %wg_y, %cwtok : index")
    e("  %token_base = index.add %wtb, %wt_off : index")
    # decode lane map: lane tid decodes weight row tid % BM, group slot tid / BM of the phase; slots >= GPP idle
    e(f"  %l64 = index.rem %tid, %c{BM} : index")
    e(f"  %slot = index.div %tid, %c{BM} : index")
    e(f"  %drow = index.min %l64, %c{BM - 1} : index")
    e("  %drow_i = index.cast %drow : index to i32")
    e("  %wg_row_i = index.cast %wg_row : index to i32")
    e("  %grow_i = scalar.addi %wg_row_i, %drow_i : i32")
    e("  %k_blocks_i = index.cast %k_blocks : index to i32")
    e("  %bpr_i = scalar.muli %k_blocks_i, %cbbi : i32")
    e("  %row_off_i = scalar.muli %grow_i, %bpr_i : i32")
    e(f"  %cslots = index.constant {slots} : index")
    e("  %slot_c = index.min %slot, %cslots : index")
    e("  %decoder = index.cmp ult, %slot, %cslots : index")
    e("  %slot_i0 = index.cast %slot_c : index to i32")
    e("  %csplit = scalar.constant 1 : i32")
    e("  %slot_i = scalar.divui %slot_i0, %csplit : i32")
    e("  %sub_i = scalar.remui %slot_i0, %csplit : i32")
    e("  %cgpl = scalar.constant 1 : i32")
    e("  %gl_i = scalar.muli %slot_i, %cgpl : i32")
    e("  %kphases = index.mul %k_blocks, %cph : index")
    # activation staging map: lane tid stages segments [aseg0, aseg0+aspl) of token row tid % BN of the tile
    e(f"  %atok = index.rem %tid, %c{BN} : index")
    e(f"  %apart = index.div %tid, %c{BN} : index")
    e(f"  %caspl8 = index.constant {8 * aspl} : index")
    e("  %aseg0 = index.mul %apart, %caspl8 : index")
    e("  %atok_g = index.add %wtb, %atok : index")
    e("  %arow_g = index.mul %atok_g, %apitch : index")
    L.extend(F["setup"]())
    e("  %z8s = scalar.constant 0 : i8")
    e("  %z8v = vector.splat %z8s : vector<8xi8>")
    e(f"  %zeros = vector.constant 0.0 : {V8}")
    e(f"  %init = vector.fragment<init> %zeros shape [%m, %n] : {V8}")
    NA = FM * FN
    types = ", ".join([V8] * NA)

    def a_loads(p, kbase):
        """Load this lane's part of its token row of a phase's activation tile as 16-byte vectors.
        Clamped, so the extra iteration's loads stay in bounds."""
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

    if DECAHEAD:
        # phase 0 decoded into weight tile 0 now; phase 1's bytes carried
        L0, w00 = loads("pf_", "%row_off_i", "%gl_i")
        L.extend(L0)
        e("  scf.if %decoder {")
        L.extend(compute([nm for nm, _ in w00], "%gl_i"))
        e("  }")
        e("  %kp1_l = index.sub %kphases, %c1 : index")
        e("  %kp1 = index.min %c1, %kp1_l : index")
        e("  %kb1 = index.div %kp1, %cph : index")
        e("  %ph1 = index.rem %kp1, %cph : index")
        e("  %kb1_i = index.cast %kb1 : index to i32")
        e("  %ph1_i = index.cast %ph1 : index to i32")
        e("  %blk1o = scalar.muli %kb1_i, %cbbi : i32")
        e("  %blk1 = scalar.addi %row_off_i, %blk1o : i32")
        e("  %gb1g = scalar.muli %ph1_i, %cgppi : i32")
        e("  %gb1 = scalar.addi %gb1g, %gl_i : i32")
        L1, wv0 = loads("p1_", "%blk1", "%gb1")
        L.extend(L1)
    else:
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
    e("  " + res + f" = scf.for %kp = [%c0 to %kphases step %c1]({ca}) -> ({carried_t})  {{")
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
    if not DECAHEAD:
        e("    scf.if %decoder {")
        names = G.unpack_vals(e, cur_w, orig0)
        L.extend(compute(names, "%gb_i"))
        e("    }")
    # stage the activation row into the LDS activation tile
    for sg, nm in enumerate(cur_a):
        e(f"    %as{sg}c = index.constant {8 * sg} : index")
        e(f"    %as{sg} = index.add %aseg0, %as{sg}c : index")
        e(f"    vector.store {nm}, %al_rows[%atok, %as{sg}] : vector<8xf16>, view<{BN}x{arow}xf16>")
    # Next phase's loads. The fence keeps them below this phase's LDS stores.
    # Else the scheduler hoists them above the stores and then waits vmcnt(0) for them before the first store.
    e("    scf.schedule.fence")
    e(f"    %kp_n0 = index.add %kp, %c{2 if DECAHEAD else 1} : index")
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
    if DECAHEAD and DECLOAD:
        # wave-uniform: the decoding lanes are whole waves
        wt = ", ".join(ty for _, ty in cur_w)
        e("    %sg_ld = kernel.subgroup.id : index")
        e(f"    %cdl = index.constant {slots * BM // WS} : index")
        e("    %ld_wave = index.cmp ult, %sg_ld, %cdl : index")
        e("    " + ", ".join(f"%nxw{x}" for x in range(len(cur_w))) + f" = scf.if %ld_wave -> ({wt}) {{")
        L.extend(Ln)
        packed = G.pack_vals(e, nxt, "n")
        e("      scf.yield " + ", ".join(nm for nm, _ in packed) + f" : {wt}")
        e("    } else {")
        e("      scf.yield " + ", ".join(nm for nm, _ in cur_w) + f" : {wt}")
        e("    }")
        nxt = [(f"%nxw{x}", ty) for x, (_, ty) in enumerate(cur_w)]
    else:
        L.extend(Ln)
        nxt = G.pack_vals(e, nxt, "n")
    if DECAHEAD:
        e("    %kp_a0 = index.add %kp, %c1 : index")
        e("    %kp_a = index.min %kp_a0, %kp_last : index")
        e("    %kk_n = index.mul %kp_a, %cksub : index")
    else:
        e("    %kk_n = index.mul %kp_n, %cksub : index")
    anx = a_loads("na_", "%kk_n")
    e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    if DECAHEAD:
        # phase kp+1 into tile (kp+1)%2 while every wave multiplies tile kp%2
        e("    %kp_d0 = index.add %kp, %c1 : index")
        e("    %kp_d = index.min %kp_d0, %kp_last : index")
        e("    %ph_d = index.rem %kp_d, %cph : index")
        e("    %ph_di = index.cast %ph_d : index to i32")
        e("    %phg_d = scalar.muli %ph_di, %cgppi : i32")
        e("    %gb_d = scalar.addi %phg_d, %gl_i : i32")
        e("    %buf_d = index.rem %kp_d0, %c2 : index")
        e("    %off_d0 = index.mul %buf_d, %wl_tb : index")
        e("    %off_d = index.cast %off_d0 : index to offset")
        e(f"    %wl_dec = buffer.view %wl[%off_d] : buffer -> view<{BM}x{G.ROWP}xf16>")
        e("    %buf_m = index.rem %kp, %c2 : index")
        e("    %off_m0 = index.mul %buf_m, %wl_tb : index")
        e("    %off_m = index.cast %off_m0 : index to offset")
        e(f"    %wl_mma = buffer.view %wl[%off_m] : buffer -> view<{BM}x{G.ROWP}xf16>")
        # decoding lanes are whole waves (tid < slots*BM): branch on the wave id so the branch is uniform.
        # A divergent branch right before the MMA loop is rejected (divergent_loop_single_entry).
        assert (slots * BM) % WS == 0
        e("    %sg_id = kernel.subgroup.id : index")
        e(f"    %cdecw = index.constant {slots * BM // WS} : index")
        e("    %dec_wave = index.cmp ult, %sg_id, %cdecw : index")
        e("    scf.if %dec_wave {")
        names = G.unpack_vals(e, cur_w, orig0)
        L.extend(l.replace("%wl_view[", "%wl_dec[") for l in compute(names, "%gb_d"))
        e("    }")
    wlv = "%wl_mma" if DECAHEAD else "%wl_view"
    alv = "%al_t"
    cb = ", ".join(f"%b{i} = %a{i} : {V8}" for i in range(NA))
    if KSL:
        # the k steps straight-line in one block: low CSE shares the fragment address math and the step offset becomes an immediate
        nst = ksub // 16
        acc = [f"%a{i}" for i in range(NA)]
        for st in range(nst):
            if st:
                e("    scf.schedule.fence")
            e(f"    %sks{st} = index.constant {16 * st} : index")
            for i in range(FM):
                e(f"    %slr{st}_{i} = index.add %wr_off, %c{16 * i} : index")
                e(f"    %slhs{st}_{i} = vector.fragment.load<lhs> {wlv}[%slr{st}_{i}, %sks{st}] shape [%m, %k] : view<{BM}x{G.ROWP}xf16> -> {VF}")

            def srhs(j):
                e(f"    %stc{st}_{j} = index.add %wt_off, %c{16 * j} : index")
                e(f"    %srhs{st}_{j} = vector.fragment.load<rhs> {alv}[%sks{st}, %stc{st}_{j}] shape [%k, %n] : view<{ksub}x{BN}xf16, %al_layout> -> {VF}")

            def smma(i, j):
                n = i * FN + j
                name = f"%r{n}" if st == nst - 1 else f"%sn{st}_{n}"
                e(f"    {name} = vector.mma %slhs{st}_{i}, %srhs{st}_{j}, {acc[n]} : {VF}, {VF}, {V8}")
            if RHS_OUTER:
                # rhs-outer: each activation fragment dies after its FM MMAs, so a step holds FM + ~RHS_FENCE fragments, not FM + FN
                for j in range(FN):
                    srhs(j)
                    for i in range(FM):
                        smma(i, j)
                    if RHS_FENCE and j + 1 < FN and (j + 1) % RHS_FENCE == 0:
                        e("    scf.schedule.fence")
            else:
                for j in range(FN):
                    srhs(j)
                for i in range(FM):
                    for j in range(FN):
                        smma(i, j)
            acc = [f"%sn{st}_{n}" for n in range(NA)]
        e("    scf.yield " + ", ".join(f"%r{i}" for i in range(NA)) + ", "
          + ", ".join(nm for nm, _ in nxt + anx) + f" : {carried_t}")
        e("  }")
    else:
        e("    " + ", ".join(f"%r{i}" for i in range(NA)) + f" = scf.for %ks = [%c0 to %cksub step %c16]({cb}) -> ({types})  {{")
        for i in range(FM):
            e(f"      %lr{i} = index.add %wr_off, %c{16 * i} : index")
            e(f"      %lhs{i} = vector.fragment.load<lhs> {wlv}[%lr{i}, %ks] shape [%m, %k] : view<{BM}x{G.ROWP}xf16> -> {VF}")

        def rhs_load(j):
            e(f"      %tc{j} = index.add %wt_off, %c{16 * j} : index")
            e(f"      %rhs{j} = vector.fragment.load<rhs> {alv}[%ks, %tc{j}] shape [%k, %n] : view<{ksub}x{BN}xf16, %al_layout> -> {VF}")
        if RHS_OUTER:
            # rhs-outer: each rhs fragment dies after its FM MMAs, so only the lhs fragments and one or two rhs are live.
            # Every accumulator still takes one MMA per k step: same values.
            for j in range(FN):
                rhs_load(j)
                for i in range(FM):
                    n = i * FN + j
                    e(f"      %n{n} = vector.mma %lhs{i}, %rhs{j}, %b{n} : {VF}, {VF}, {V8}")
                if RHS_FENCE and j + 1 < FN and (j + 1) % RHS_FENCE == 0:
                    e("      scf.schedule.fence")
        else:
            for j in range(FN):
                rhs_load(j)
            for i in range(FM):
                for j in range(FN):
                    n = i * FN + j
                    e(f"      %n{n} = vector.mma %lhs{i}, %rhs{j}, %b{n} : {VF}, {VF}, {V8}")
        e("      scf.yield " + ", ".join(f"%n{i}" for i in range(NA)) + f" : {types}")
        e("    }")
        e("    scf.yield " + ", ".join(f"%r{i}" for i in range(NA)) + ", "
          + ", ".join(nm for nm, _ in nxt + anx) + f" : {carried_t}")
        e("  }")
    if t.swepi:
        lds_epilogue(e, t, kr, V8, sw=True, masked=masked)
        e("  kernel.return")
        e("}")
        return "\n".join(L) + "\n"
    if sw:
        swiglu_epilogue(e, t, arow, masked)
        e("  kernel.return")
        e("}")
        return "\n".join(L) + "\n"
    if TM == 32:
        lds_epilogue(e, t, kr, V8, qg=qg, masked=masked)
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


def lds_epilogue(e, t, kr, V8, sw=False, qg=False, masked=False):
    """Store out[t*m + r] (+ resid) for the wave's TM x TN tile, one 16-token column of fragments at a time.
    Fragments go to an LDS slab; each lane reads 16 contiguous rows of one token (two lanes per token) and writes 4 b128 stores.
    A direct fragment store writes each lane's values at an 8-byte row stride instead. Same values: bit-identical."""
    TM, FM, FN = t.tm, t.tm // 16, t.tn // 16
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e(f"  %es_ctm = index.constant {TM + EPAD} : index")
    e("  %es_lay = encoding.layout.strided [%c1, %es_ctm] : encoding<layout>")
    e(f"  %es_wb = index.constant {(TM + EPAD) * 16 * 4} : index")
    e("  %es_off_i = index.mul %wave, %es_wb : index")
    e("  %es_off = index.cast %es_off_i : index to offset")
    e(f"  %es_view = buffer.view %al[%es_off] : buffer -> view<{TM}x16xf32, %es_lay>")
    e(f"  %es_flat = buffer.view %al[%es_off] : buffer -> view<{(TM + EPAD) * 16}xf32>")
    if sw:
        # swiglu: out f16 = f16(silu(gate) * acc), swiglu_epilogue's scalar ops per element (bit-identical), 4 rows per load/store
        e("  %gate_view = buffer.view %gate_na[%base] : buffer -> view<[%out_total]xf32>")
        e("  %out_h = buffer.view %output_na[%base] : buffer -> view<[%out_total]xf16>")
        e("  %negone = scalar.constant -1.0 : f32")
        e("  %one = scalar.constant 1.0 : f32")
    elif qg:
        # row r = head*512 + half*256 + d goes to (q|gate)[t][head*256 + d]; a wave's TM=32 rows sit inside one half
        e("  %qg_tot = index.div %out_total, %c2 : index")
        e("  %qg_rows = index.div %m_rows, %c2 : index")
        e("  %q_flat = buffer.view %output_na[%base] : buffer -> view<[%qg_tot]xf32>")
        e("  %g_flat = buffer.view %gate_out_na[%base] : buffer -> view<[%qg_tot]xf32>")
        e("  %qg_last4 = index.sub %qg_tot, %c4 : index")
        e("  %qg_c512 = index.constant 512 : index")
        e("  %qg_c256 = index.constant 256 : index")
    else:
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
    if qg:
        e("  %qg_head = index.div %es_row, %qg_c512 : index")
        e("  %qg_w = index.rem %es_row, %qg_c512 : index")
        e("  %qg_half = index.div %qg_w, %qg_c256 : index")
        e("  %qg_d = index.rem %qg_w, %qg_c256 : index")
        e("  %qg_hb = index.mul %qg_head, %qg_c256 : index")
        e("  %qg_col = index.add %qg_hb, %qg_d : index")
        e("  %qg_isq = index.cmp eq, %qg_half, %c0 : index")
    for j in range(FN):
        for i in range(FM):
            e(f"  %es_r{i}_{j} = index.constant {16 * i} : index")
            e(f"  vector.fragment.store<result> %acc{i * FN + j}, %es_view[%es_r{i}_{j}, %c0] shape [%m, %n] : {V8}, view<{TM}x16xf32, %es_lay>")
        e(f"  %es_tc{j} = index.constant {16 * j} : index")
        e(f"  %es_tk{j}0 = index.add %token_base, %es_tc{j} : index")
        e(f"  %es_tk{j} = index.add %es_tk{j}0, %es_t : index")
        e(f"  %es_tm{j} = index.mul %es_tk{j}, %m_rows : index")
        e(f"  %es_ob{j} = index.add %es_tm{j}, %es_row : index")
        if qg:
            e(f"  %qg_tm{j} = index.mul %es_tk{j}, %qg_rows : index")
            e(f"  %qg_ob{j} = index.add %qg_tm{j}, %qg_col : index")
        if masked:
            # this lane's token past the last valid one: no residual / gate load, no store
            e(f"  %es_ok{j} = index.cmp ult, %es_tk{j}, %tokens : index")
            e(f"  scf.if %es_ok{j} {{")
        for q in range(4):
            e(f"  %es_q{j}_{q}c = index.constant {4 * q} : index")
            e(f"  %es_ri{j}_{q} = index.add %es_rd, %es_q{j}_{q}c : index")
            e(f"  %es_v{j}_{q} = vector.load %es_flat[%es_ri{j}_{q}] : view<{(TM + EPAD) * 16}xf32> -> vector<4xf32>")
            e(f"  %es_oi{j}_{q} = index.add %es_ob{j}, %es_q{j}_{q}c : index")
            val = f"%es_v{j}_{q}"
            if kr:
                e(f"  %es_rf{j}_{q} = vector.load %res_flat[%es_oi{j}_{q}] : view<[%out_total]xf32> -> vector<4xf32>")
                e(f"  %es_rs{j}_{q} = vector.addf %es_rf{j}_{q}, %es_v{j}_{q} : vector<4xf32>")
                val = f"%es_rs{j}_{q}"
            if sw:
                e(f"  %es_g{j}_{q} = vector.load %gate_view[%es_oi{j}_{q}] : view<[%out_total]xf32> -> vector<4xf32>")
                hs = []
                for x in range(4):
                    y = f"{j}_{q}_{x}"
                    e(f"  %g_{y} = vector.extract %es_g{j}_{q}[{x}] : vector<4xf32> -> f32")
                    e(f"  %v_{y} = vector.extract %es_v{j}_{q}[{x}] : vector<4xf32> -> f32")
                    e(f"  %ng_{y} = scalar.mulf %g_{y}, %negone : f32")
                    e(f"  %ex_{y} = scalar.expf<afn> %ng_{y} : f32")
                    e(f"  %dn_{y} = scalar.addf %one, %ex_{y} : f32")
                    e(f"  %iv_{y} = scalar.divf %one, %dn_{y} : f32")
                    e(f"  %sg_{y} = scalar.mulf %g_{y}, %iv_{y} : f32")
                    e(f"  %ac_{y} = scalar.mulf %sg_{y}, %v_{y} : f32")
                    e(f"  %h_{y} = scalar.fptrunc %ac_{y} : f32 to f16")
                    hs.append(f"%h_{y}")
                e(f"  %hv{j}_{q} = vector.from_elements {', '.join(hs)} : vector<4xf16>")
                e(f"  vector.store %hv{j}_{q}, %out_h[%es_oi{j}_{q}] : vector<4xf16>, view<[%out_total]xf16>")
            elif qg:
                # always in range; the clamp states it for the bound proof
                e(f"  %qg_oir{j}_{q} = index.add %qg_ob{j}, %es_q{j}_{q}c : index")
                e(f"  %qg_oi{j}_{q} = index.min %qg_oir{j}_{q}, %qg_last4 : index")
                e(f"  scf.if %qg_isq {{")
                e(f"    vector.store {val}, %q_flat[%qg_oi{j}_{q}] : vector<4xf32>, view<[%qg_tot]xf32>")
                e("  } else {")
                e(f"    vector.store {val}, %g_flat[%qg_oi{j}_{q}] : vector<4xf32>, view<[%qg_tot]xf32>")
                e("  }")
            else:
                e(f"  vector.store {val}, %out_flat[%es_oi{j}_{q}] : vector<4xf32>, view<[%out_total]xf32>")
        if masked:
            e("  }")


def swiglu_epilogue(e, t, arow, masked=False):
    """out[t*m + r] = f16(silu(gate[t*m + r]) * acc[r][t]), the .loom kernel's scalar ops in order (bit-identical).
    As in gen_gemm_decode, each 16-row x ES-token slab goes through a per-wave f32 LDS tile; a loop walks it lane-contiguous.
    ES is the larger of 32, 16 for which all waves' slabs fit in the activation tile's LDS."""
    BN, TN, NWAVE, FM, FN = t.bn, t.tn, t.nwave, t.tm // 16, t.tn // 16
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
            if masked:
                e(f"    %gok_{q} = index.cmp ult, %gtok_{q}, %tokens : index")
                e(f"    scf.if %gok_{q} {{")
            e(f"    %g_{q} = view.load %gate_view[%gix_{q}] : view<[%out_total]xf32> -> f32")
            e(f"    %ng_{q} = scalar.mulf %g_{q}, %negone : f32")
            e(f"    %ex_{q} = scalar.expf<afn> %ng_{q} : f32")
            e(f"    %dn_{q} = scalar.addf %one, %ex_{q} : f32")
            e(f"    %iv_{q} = scalar.divf %one, %dn_{q} : f32")
            e(f"    %sg_{q} = scalar.mulf %g_{q}, %iv_{q} : f32")
            e(f"    %ac_{q} = scalar.mulf %sg_{q}, %v_{q} : f32")
            e(f"    %h_{q} = scalar.fptrunc %ac_{q} : f32 to f16")
            e(f"    view.store %h_{q}, %out_h[%gix_{q}] : f16, view<[%out_total]xf16>")
            if masked:
                e("    }")
            e(f"    scf.yield %em{q} : index")
            e("  }")
