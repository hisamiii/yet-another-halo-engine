#!/usr/bin/env python3
"""Single-token decode GEMV kernels, one per (kind, weight formats, M, K).

    y[m] = sum_k W[m, k] * x[k]          W packed in a GGUF quant format, x f32

The decode arithmetic is HIP's (Q8KBlockGEMVKernel_2Rows over QuantWarpBlockDot,
quant_ops.hpp): a weight row is cut into 16-element sub-blocks, lane l of a wave
owns sub-blocks l, l + 32, ..., and each sub-block decodes (DecodeQuantSub16)
to sixteen small integers q, a scale and an offset, so that

    w[j] = scale * q[j] - offset          partial += scale * dot(q, x) - offset * sum(x)

followed by a butterfly reduction over the 32 lanes. Differences from HIP: the
format and shape are compile-time (one kernel per tensor shape, like the prefill
GEMMs), one wave carries R rows at once so every x load serves R rows, and the
per-lane decode constants (which sub-block of the 256-element block a lane owns,
shifts, scale indices) are loop invariant because K / 16 is a multiple of 32.

Kinds:
  plain   y[m] = W x
  resid   y[m] += W x                    (the residual add fused, y read and written)
  swiglu  y[m] = silu(G x) * (U x)       G, U may have different formats; HIP's
                                         (g * (1 / (1 + exp(-g)))) * u
Usage (module): gen(kind, fmts, M, K, R=2, W=4) -> Loom source text.
Bindings: weights (one per format in fmts), then the IQ tables the formats need
(TABLE_ORDER), then x, then y.
"""
import os
import re
import sys

# fmt: (elements per block, bytes per block, tables)
FMT = {
    "q8_0": (32, 34, ()),
    "q2k": (256, 84, ()),
    "q3k": (256, 110, ()),
    "q4k": (256, 144, ()),
    "q5k": (256, 176, ()),
    "q6k": (256, 210, ()),
    "iq4xs": (256, 136, ()),
    "iq3s": (256, 110, ("grid_iq3s",)),
    "iq3xxs": (256, 98, ("grid_iq3xxs", "ksigns")),
    "iq2xxs": (256, 66, ("grid_iq2xxs", "ksigns")),
    "iq2xs": (256, 74, ("grid_iq2xs", "ksigns")),
}
# table: (element type, elements); files in the HAL dir (emit_prefill writes them)
TABLES = {"grid_iq3s": ("i32", 512), "grid_iq3xxs": ("i32", 256), "grid_iq2xxs": ("i32", 512),
          "grid_iq2xs": ("i32", 1024), "ksigns": ("i8", 128)}
TABLE_ORDER = ["grid_iq3s", "grid_iq3xxs", "grid_iq2xxs", "grid_iq2xs", "ksigns"]
OFFSET_FMTS = ("q4k", "q5k", "q2k")       # formats with an explicit minimum
IQ4_KVALUES = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
# YAH_GV_GRID_LDS=1: each workgroup copies its IQ tables to LDS before the loop.
GRID_LDS = os.environ.get("YAH_GV_GRID_LDS", "0") == "1"
# YAH_GV_ABL=nogrid: ablation (wrong results) -- grid/ksigns loads replaced by their index.
ABL = os.environ.get("YAH_GV_ABL", "")
# YAH_GV_PIPE=d / YAH_GV_UNROLL=u: Loom read-ahead / unroll on the sub-block loop.
# Full decode, one round each (2026-10-02, word decode): no pipeline 68.2 ms/token,
# depth 2 63.1, depth 3 66.1, depth 4 67.8, depth 2 + unroll 2 63.5, unroll 2 70.2.
PIPE = int(os.environ.get("YAH_GV_PIPE", "2"))
# YAH_GV_ASSUME=1: in-range offsets are promised with index.assume (no VALU) instead of
# being clamped with index.max / index.min (2 VALU per load). Every offset is in range by
# construction (rows < M, blocks < K / qk, grid indices < 512, ksigns < 128); gemv_check
# covers every (format, K) and kind.
ASSUME = os.environ.get("YAH_GV_ASSUME", "0") == "1"
# YAH_GV_SMASK_LDS=1: sign masks (4 sign bits -> 4 bytes of 0x00 / 0xFF) from a 16-entry
# i32 table in LDS (HIP's kDeviceIq3sSignMask) instead of two v_mul_lo_u32 (quarter rate).
# Full decode, one round each (2026-10-02): base 61.25, table 61.69, table + ASSUME 61.47
# ms/token -- neutral: the v_mul_lo_u32 (67 -> 7) were not the limit, the GEMVs are not
# VALU-bound after word decode. Off by default; kept as switches.
SMASK_LDS = os.environ.get("YAH_GV_SMASK_LDS", "0") == "1"
# YAH_GV_G2=1: a lane decodes a whole 32-element group (both sub-blocks) per iteration,
# sharing the group's header loads (d, scales, qh, aux) and merging the two halves'
# byte loads into one wider load each.
# Per unit of work 11-31% fewer VALU and 14-43% fewer loads, but VGPRs go from 60-75 to
# 103-114 (occupancy): full decode 61.77 vs 60.48 ms/token (with pipeline(0): 68.39).
# Off by default; the same trade HIP's quant_ops.hpp notes for its 32-element variant.
G2 = os.environ.get("YAH_GV_G2", "0") == "1"
# YAH_GV_PERSIST=G (megakernel feasibility): launch G workgroups that loop over the row
# groups (rg = wg, wg + G, ...) instead of one workgroup per row group.
# Full decode (2026-10-02): G = 160 / 320 / 640 -> 62.43 / 62.20 / 61.67 vs 60.48 ms/token;
# kernels with one row group per workgroup are unchanged, SwiGLU (3.4 groups each at
# G = 640) +3.7%: static round-robin leaves a ragged last round. Off by default.
PERSIST = int(os.environ.get("YAH_GV_PERSIST", "0"))
def _s32(v):
    return v - (1 << 32) if v >= (1 << 31) else v
UNROLL = int(os.environ.get("YAH_GV_UNROLL", "0"))
# YAH_GV_WORD=1: decode on 32-bit words (vector<4xi32>) to unsigned byte codes plus a
# bias folded into the offset, converted with uitofp. The byte-vector form
# (vector<16xi8> shifts / masks) lowers per byte: ~200-240 VALU per sub-block.
WORD = os.environ.get("YAH_GV_WORD", "1") == "1"
# (tried 2026-10-02: the 16-element dot as an explicit fma chain -- same VALU count,
# 1040 vs 1051 on swiglu iq3s/iq3xxs: the compiler already contracts mul + reduce)
GGML = {8: "q8_0", 10: "q2k", 11: "q3k", 12: "q4k", 13: "q5k", 14: "q6k", 16: "iq2xxs", 17: "iq2xs",
        18: "iq3xxs", 21: "iq3s", 23: "iq4xs"}


def tables_for(fmts):
    need = set(t for f in fmts for t in FMT[f][2])
    return [t for t in TABLE_ORDER if t in need]


def row_bytes(fmt, K):
    qk, bb, _ = FMT[fmt]
    assert K % qk == 0
    return K // qk * bb


class E:
    def __init__(self):
        self.L = []
        self.consts = {}

    def __call__(self, s):
        self.L.append(s)

    def ci(self, v):           # index constant
        n = f"%ci{v}"
        self.consts[n] = f"  {n} = index.constant {v} : index"
        return n

    def cs(self, v, ty="i32"):  # scalar constant
        t = {"i32": "w", "i8": "b", "f32": "f"}[ty]
        nm = f"%c{t}{str(v).replace('-', 'm').replace('.', 'p')}"
        self.consts[nm] = f"  {nm} = scalar.constant {v} : {ty}"
        return nm

    def splat8(self, v):        # vector<16xi8> splat of an i8 constant
        nm = f"%sv{str(v).replace('-', 'm')}"
        s = self.cs(v, "i8")
        self.consts[nm] = f"  {nm} = vector.splat {s} : vector<16xi8>"
        return nm


def gen(kind, fmts, M, K, R=2, W=4, name="yah_gemv", _e=None, _parts=None):
    assert kind in ("plain", "resid", "resid_norm", "swiglu")
    assert len(fmts) == (2 if kind == "swiglu" else 1)
    assert K % 512 == 0, "K / 16 sub-blocks must split evenly over 32 lanes"
    rows_wg = R * W
    assert M % rows_wg == 0, (M, rows_wg)
    NT = K // 512                    # sub-blocks per lane
    tabs = tables_for(fmts)
    e = _e if _e is not None else E()
    body = []
    b = body.append

    # ---- per-lane, loop-invariant decode setup, and the per-iteration decode ----
    def lane_setup(f, p):
        """Emit lane constants for format f (prefix p) into body; return a dict."""
        qk, bb, _ = FMT[f]
        S = {}
        if qk == 32:     # q8_0: block = sb / 2 = lane / 2 + 16 t; half = lane % 2
            b(f"  %{p}lb = index.div %lane, {e.ci(2)} : index")
            b(f"  %{p}lboff = index.mul %{p}lb, {e.ci(bb)} : index")
            b(f"  %{p}hlf = index.rem %lane, {e.ci(2)} : index")
            b(f"  %{p}qo0 = index.mul %{p}hlf, {e.ci(16)} : index")
            b(f"  %{p}qo = index.add %{p}qo0, {e.ci(2)} : index")
            S["tstep"] = 16 * bb
            return S
        # 256-element blocks: block = lane / 16 + 2 t, sb32 = (lane / 2) % 8, hlf = lane % 2
        b(f"  %{p}lb = index.div %lane, {e.ci(16)} : index")
        b(f"  %{p}lboff = index.mul %{p}lb, {e.ci(bb)} : index")
        b(f"  %{p}l2 = index.div %lane, {e.ci(2)} : index")
        b(f"  %{p}s32 = index.rem %{p}l2, {e.ci(8)} : index")
        b(f"  %{p}hlf = index.rem %lane, {e.ci(2)} : index")
        b(f"  %{p}s32i = index.cast %{p}s32 : index to i32")
        b(f"  %{p}hlfi = index.cast %{p}hlf : index to i32")
        b(f"  %{p}l16 = index.mul %{p}hlf, {e.ci(16)} : index")       # lane0 = hlf * 16
        S["tstep"] = 2 * bb

        def sh8(name, expr_i32):
            """splat an i32 shift amount (0..7) to vector<16xi8>"""
            b(f"  %{p}{name}b = scalar.trunci {expr_i32} : i32 to i8")
            b(f"  %{p}{name} = vector.splat %{p}{name}b : vector<16xi8>")
            b(f"  %{p}{name}w = vector.splat {expr_i32} : vector<4xi32>")
            return f"%{p}{name}"

        if f in ("q4k", "q5k"):
            # q bytes: base = (sb32 / 2) * 32 + hlf * 16 (+16 or +48), shift = 4 * (sb32 & 1)
            b(f"  %{p}s32h = index.div %{p}s32, {e.ci(2)} : index")
            b(f"  %{p}qb0 = index.mul %{p}s32h, {e.ci(32)} : index")
            b(f"  %{p}qb1 = index.add %{p}qb0, %{p}l16 : index")
            b(f"  %{p}qo = index.add %{p}qb1, {e.ci(48 if f == 'q5k' else 16)} : index")
            b(f"  %{p}sha = scalar.andi %{p}s32i, {e.cs(1)} : i32")
            b(f"  %{p}shi = scalar.shli %{p}sha, {e.cs(2)} : i32")
            S["sh"] = sh8("shv", f"%{p}shi")
            if f == "q5k":
                b(f"  %{p}ho = index.add %{p}l16, {e.ci(16)} : index")
                S["hsh"] = sh8("hshv", f"%{p}s32i")
            # scales (HIP SharedScales form): base = sb32 & 3 at +4, +8, +12; upper = sb32 >> 2
            b(f"  %{p}sbase = index.rem %{p}s32, {e.ci(4)} : index")
            b(f"  %{p}so0 = index.add %{p}sbase, {e.ci(4)} : index")
            b(f"  %{p}so1 = index.add %{p}sbase, {e.ci(8)} : index")
            b(f"  %{p}so2 = index.add %{p}sbase, {e.ci(12)} : index")
            b(f"  %{p}upi = scalar.shrui %{p}s32i, {e.cs(2)} : i32")
            b(f"  %{p}up = scalar.subi {e.cs(0)}, %{p}upi : i32")                # all-ones when sb32 >= 4
        elif f == "q6k":
            b(f"  %{p}half = index.div %{p}s32, {e.ci(4)} : index")
            b(f"  %{p}seg = index.rem %{p}s32, {e.ci(4)} : index")
            b(f"  %{p}segi = index.cast %{p}seg : index to i32")
            b(f"  %{p}s1 = index.rem %{p}seg, {e.ci(2)} : index")
            b(f"  %{p}qlb0 = index.mul %{p}half, {e.ci(64)} : index")
            b(f"  %{p}qlb1 = index.mul %{p}s1, {e.ci(32)} : index")
            b(f"  %{p}qlb2 = index.add %{p}qlb0, %{p}qlb1 : index")
            b(f"  %{p}qo = index.add %{p}qlb2, %{p}l16 : index")
            b(f"  %{p}qsh0 = scalar.shrui %{p}segi, {e.cs(1)} : i32")
            b(f"  %{p}qshi = scalar.shli %{p}qsh0, {e.cs(2)} : i32")         # (seg >= 2) * 4
            S["sh"] = sh8("shv", f"%{p}qshi")
            b(f"  %{p}hshi = scalar.shli %{p}segi, {e.cs(1)} : i32")          # 2 * seg
            S["hsh"] = sh8("hshv", f"%{p}hshi")
            b(f"  %{p}hb0 = index.mul %{p}half, {e.ci(32)} : index")
            b(f"  %{p}hb1 = index.add %{p}hb0, %{p}l16 : index")
            b(f"  %{p}ho = index.add %{p}hb1, {e.ci(128)} : index")
            b(f"  %{p}sc0 = index.mul %{p}half, {e.ci(8)} : index")
            b(f"  %{p}sc1 = index.mul %{p}seg, {e.ci(2)} : index")
            b(f"  %{p}sc2 = index.add %{p}sc0, %{p}sc1 : index")
            b(f"  %{p}sc3 = index.add %{p}sc2, %{p}hlf : index")
            b(f"  %{p}so = index.add %{p}sc3, {e.ci(192)} : index")
        elif f == "q3k":
            b(f"  %{p}half = index.div %{p}s32, {e.ci(4)} : index")
            b(f"  %{p}sp = index.rem %{p}s32, {e.ci(4)} : index")
            b(f"  %{p}spi = index.cast %{p}sp : index to i32")
            b(f"  %{p}halfi = index.cast %{p}half : index to i32")
            b(f"  %{p}shi = scalar.shli %{p}spi, {e.cs(1)} : i32")
            S["sh"] = sh8("shv", f"%{p}shi")
            b(f"  %{p}hs0 = scalar.shli %{p}halfi, {e.cs(2)} : i32")
            b(f"  %{p}hsi = scalar.addi %{p}hs0, %{p}spi : i32")
            S["hsh"] = sh8("hshv", f"%{p}hsi")
            b(f"  %{p}qb0 = index.mul %{p}half, {e.ci(32)} : index")
            b(f"  %{p}qb1 = index.add %{p}qb0, %{p}l16 : index")
            b(f"  %{p}qo = index.add %{p}qb1, {e.ci(32)} : index")
            b(f"  %{p}ho = index.add %{p}l16, {e.ci(0)} : index")
            # scale index si = half * 8 + sp * 2 + hlf; low nibble byte s[si & 7] >> 4 * (si >> 3);
            # high pair (s[8 + si % 4] >> 2 * (si / 4)) & 3
            b(f"  %{p}si0 = index.mul %{p}half, {e.ci(8)} : index")
            b(f"  %{p}si1 = index.mul %{p}sp, {e.ci(2)} : index")
            b(f"  %{p}si2 = index.add %{p}si0, %{p}si1 : index")
            b(f"  %{p}si = index.add %{p}si2, %{p}hlf : index")
            b(f"  %{p}sii = index.cast %{p}si : index to i32")
            b(f"  %{p}sl7 = index.rem %{p}si, {e.ci(8)} : index")
            b(f"  %{p}slo = index.add %{p}sl7, {e.ci(96)} : index")
            b(f"  %{p}sls0 = scalar.shrui %{p}sii, {e.cs(3)} : i32")
            b(f"  %{p}sls = scalar.shli %{p}sls0, {e.cs(2)} : i32")
            b(f"  %{p}sh4 = index.rem %{p}si, {e.ci(4)} : index")
            b(f"  %{p}sho = index.add %{p}sh4, {e.ci(104)} : index")
            b(f"  %{p}shs0 = scalar.shrui %{p}sii, {e.cs(2)} : i32")
            b(f"  %{p}shs = scalar.shli %{p}shs0, {e.cs(1)} : i32")
        elif f == "q2k":
            # sub = sb % 16 = lane % 16; shift = 2 * ((sub % 8) / 2); qs base = (sub / 8) * 32 + (sub % 2) * 16
            b(f"  %{p}sub = index.rem %lane, {e.ci(16)} : index")
            b(f"  %{p}s8 = index.rem %{p}sub, {e.ci(8)} : index")
            b(f"  %{p}s82 = index.div %{p}s8, {e.ci(2)} : index")
            b(f"  %{p}s82i = index.cast %{p}s82 : index to i32")
            b(f"  %{p}shi = scalar.shli %{p}s82i, {e.cs(1)} : i32")
            S["sh"] = sh8("shv", f"%{p}shi")
            b(f"  %{p}sd8 = index.div %{p}sub, {e.ci(8)} : index")
            b(f"  %{p}qb0 = index.mul %{p}sd8, {e.ci(32)} : index")
            b(f"  %{p}qb1 = index.add %{p}qb0, %{p}l16 : index")
            b(f"  %{p}qo = index.add %{p}qb1, {e.ci(16)} : index")
            b(f"  %{p}so = index.add %{p}sub, {e.ci(0)} : index")
        elif f == "iq4xs":
            b(f"  %{p}qb0 = index.mul %{p}s32, {e.ci(16)} : index")
            b(f"  %{p}qo = index.add %{p}qb0, {e.ci(8)} : index")
            b(f"  %{p}shi = scalar.shli %{p}hlfi, {e.cs(2)} : i32")
            S["sh"] = sh8("shv", f"%{p}shi")
            b(f"  %{p}sl0 = index.div %{p}s32, {e.ci(2)} : index")
            b(f"  %{p}so = index.add %{p}sl0, {e.ci(4)} : index")
            b(f"  %{p}sl1 = scalar.andi %{p}s32i, {e.cs(1)} : i32")
            b(f"  %{p}sls = scalar.shli %{p}sl1, {e.cs(2)} : i32")              # 4 * (sb32 % 2)
            b(f"  %{p}shs = scalar.shli %{p}s32i, {e.cs(1)} : i32")              # 2 * sb32
        elif f in ("iq3s", "iq3xxs", "iq2xxs", "iq2xs"):
            # sign of element j: bit (j % 8) of sign byte l = 2 * hlf + j / 8
            e.consts["%sgnsh"] = "  %sgnsh = vector.from_elements " + ", ".join(e.cs(j % 8, "i8") for j in range(16)) + " : vector<16xi8>"
            b(f"  %{p}l0 = index.mul %{p}hlf, {e.ci(2)} : index")                # l of the first eight
            if f == "iq3s":
                b(f"  %{p}qb0 = index.mul %{p}s32, {e.ci(8)} : index")
                b(f"  %{p}qb1 = index.mul %{p}hlf, {e.ci(4)} : index")
                b(f"  %{p}qb2 = index.add %{p}qb0, %{p}qb1 : index")
                b(f"  %{p}qo = index.add %{p}qb2, {e.ci(2)} : index")             # qs[sb32*8 + 2l]
                b(f"  %{p}ho = index.add %{p}s32, {e.ci(66)} : index")            # qh[sb32]
                b(f"  %{p}sg0 = index.mul %{p}s32, {e.ci(4)} : index")
                b(f"  %{p}sg1 = index.add %{p}sg0, %{p}l0 : index")
                b(f"  %{p}sgo = index.add %{p}sg1, {e.ci(74)} : index")           # signs[sb32*4 + l]
                b(f"  %{p}sl0 = index.div %{p}s32, {e.ci(2)} : index")
                b(f"  %{p}so = index.add %{p}sl0, {e.ci(106)} : index")
                b(f"  %{p}sl1 = scalar.andi %{p}s32i, {e.cs(1)} : i32")
                b(f"  %{p}sls = scalar.shli %{p}sl1, {e.cs(2)} : i32")
                # qh shift for grid index l: (qh << (8 - 2l)) & 256 and (qh << (7 - 2l)) & 256
                b(f"  %{p}l0i = index.cast %{p}l0 : index to i32")
                for qq in (0, 1):
                    b(f"  %{p}l2x{qq} = scalar.addi %{p}l0i, {e.cs(qq)} : i32")
                    b(f"  %{p}l2y{qq} = scalar.shli %{p}l2x{qq}, {e.cs(1)} : i32")
                    b(f"  %{p}qsA{qq} = scalar.subi {e.cs(8)}, %{p}l2y{qq} : i32")
                    b(f"  %{p}qsB{qq} = scalar.subi {e.cs(7)}, %{p}l2y{qq} : i32")
            elif f == "iq3xxs":
                b(f"  %{p}qb0 = index.mul %{p}s32, {e.ci(8)} : index")
                b(f"  %{p}qb1 = index.mul %{p}hlf, {e.ci(4)} : index")
                b(f"  %{p}qb2 = index.add %{p}qb0, %{p}qb1 : index")
                b(f"  %{p}qo = index.add %{p}qb2, {e.ci(2)} : index")             # qs[8 ib32 + 2l]
                b(f"  %{p}ax0 = index.mul %{p}s32, {e.ci(4)} : index")
                b(f"  %{p}ao = index.add %{p}ax0, {e.ci(66)} : index")            # aux32 at 66 + 4 ib32
            elif f == "iq2xxs":
                b(f"  %{p}ax0 = index.mul %{p}s32, {e.ci(8)} : index")
                b(f"  %{p}ao = index.add %{p}ax0, {e.ci(6)} : index")             # high u32 = bytes[4..8)
                b(f"  %{p}qb1 = index.add %{p}ax0, %{p}l0 : index")
                b(f"  %{p}qo = index.add %{p}qb1, {e.ci(2)} : index")             # bytes[l], bytes[l + 1]
            else:  # iq2xs
                b(f"  %{p}qb0 = index.mul %{p}s32, {e.ci(8)} : index")
                b(f"  %{p}qb1 = index.mul %{p}l0, {e.ci(2)} : index")
                b(f"  %{p}qb2 = index.add %{p}qb0, %{p}qb1 : index")
                b(f"  %{p}qo = index.add %{p}qb2, {e.ci(2)} : index")             # qs u16[4 ib32 + l]
                b(f"  %{p}so = index.add %{p}s32, {e.ci(66)} : index")
                b(f"  %{p}sls = scalar.shli %{p}hlfi, {e.cs(2)} : i32")
            if f in ("iq3xxs", "iq2xxs"):
                b(f"  %{p}l0i = index.cast %{p}l0 : index to i32")
                for li in (0, 1):
                    b(f"  %{p}lx{li} = scalar.addi %{p}l0i, {e.cs(li)} : i32")
                    b(f"  %{p}ls{li} = scalar.muli %{p}lx{li}, {e.cs(7)} : i32")      # 7 * l
        else:
            raise SystemExit("format " + f)
        return S

    def ld(view, nbytes, off, p, nm, n=1, ty="i8"):
        """load n x ty at byte (or element) offset off from view, clamped."""
        lim = e.ci(nbytes - n)
        if ABL == "nogrid" and view.startswith("%t_"):
            b(f"    %{p}{nm}x = index.cast {off} : index to i32")
            if ty == "i8":
                b(f"    %{p}{nm} = scalar.trunci %{p}{nm}x : i32 to i8")
            elif n == 1:
                b(f"    %{p}{nm} = scalar.addi %{p}{nm}x, {e.cs(0)} : i32")
            else:
                b(f"    %{p}{nm} = vector.from_elements " + ", ".join([f"%{p}{nm}x"] * n) + f" : vector<{n}xi32>")
            return f"%{p}{nm}"
        if ASSUME:
            b(f"    %{p}{nm}c = index.assume {off} [range({off}, 0, {nbytes - n + 1})] : index")
        else:
            b(f"    %{p}{nm}z = index.max {off}, {e.ci(0)} : index")
            b(f"    %{p}{nm}c = index.min %{p}{nm}z, {lim} : index")
        if n == 1:
            b(f"    %{p}{nm} = view.load {view}[%{p}{nm}c] : view<{nbytes}x{ty}> -> {ty}")
        else:
            b(f"    %{p}{nm} = vector.load {view}[%{p}{nm}c] : view<{nbytes}x{ty}> -> vector<{n}x{ty}>")
        return f"%{p}{nm}"

    def ld_u8(view, nbytes, off, p, nm):
        v = ld(view, nbytes, off, p, nm + "8")
        b(f"    %{p}{nm} = scalar.extui {v} : i8 to i32")
        return f"%{p}{nm}"

    def ld_u32(view, nbytes, off, p, nm):     # 4 bytes, any alignment
        v = ld(view, nbytes, off, p, nm + "v", 4)
        b(f"    %{p}{nm}w = vector.bitcast {v} : vector<4xi8> to vector<1xi32>")
        b(f"    %{p}{nm} = vector.extract %{p}{nm}w[0] : vector<1xi32> -> i32")
        return f"%{p}{nm}"

    def ld_d(hview, nhalf, boff, doff, p, nm):
        b(f"    %{p}{nm}a = index.add {boff}, {e.ci(doff)} : index")
        b(f"    %{p}{nm}h = index.div %{p}{nm}a, {e.ci(2)} : index")
        v = ld(hview, nhalf, f"%{p}{nm}h", p, nm + "x", 1, "f16")
        b(f"    %{p}{nm} = scalar.extf {v} : f16 to f32")
        return f"%{p}{nm}"

    def signs16(p, s0, s1):
        """vector<16xi8> negate masks from sign bytes s0 (elements 0-7) and s1 (8-15), i32 operands."""
        b(f"    %{p}sg0 = scalar.trunci {s0} : i32 to i8")
        b(f"    %{p}sg1 = scalar.trunci {s1} : i32 to i8")
        b(f"    %{p}sgv = vector.from_elements " + ", ".join([f"%{p}sg0"] * 8 + [f"%{p}sg1"] * 8) + " : vector<16xi8>")
        b(f"    %{p}sgs = vector.shrui %{p}sgv, %sgnsh : vector<16xi8>")
        b(f"    %{p}sgb = vector.andi %{p}sgs, {e.splat8(1)} : vector<16xi8>")
        b(f"    %{p}neg = vector.subi {e.splat8(0)}, %{p}sgb : vector<16xi8>")
        return f"%{p}neg"

    def apply_signs(p, g, neg):
        b(f"    %{p}gx = vector.xori {g}, {neg} : vector<16xi8>")
        b(f"    %{p}q = vector.subi %{p}gx, {neg} : vector<16xi8>")
        return f"%{p}q"


    def wspl(v):
        nm = f"%ws{v}"
        c = e.cs(v)
        e.consts[nm] = f"  {nm} = vector.splat {c} : vector<4xi32>"
        return nm

    def tow(p, v16, nm):
        b(f"    %{p}{nm} = vector.bitcast {v16} : vector<16xi8> to vector<4xi32>")
        return f"%{p}{nm}"

    def wop(p, nm, op, a, c):
        b(f"    %{p}{nm} = vector.{op} {a}, {c} : vector<4xi32>")
        return f"%{p}{nm}"

    def tob(p, w4, nm):
        b(f"    %{p}{nm} = vector.bitcast {w4} : vector<4xi32> to vector<16xi8>")
        return f"%{p}{nm}"

    if SMASK_LDS:
        for v in [0, 1, 2, 3] + [_s32(255 << (8 * j)) for j in range(4)]:
            e.cs(v)
        e.ci(16)

    def spread(p, nm, nib):
        """i32 nibble (4 sign bits) -> byte mask word (0x00 / 0xFF per byte)"""
        if SMASK_LDS:
            e.consts["%smask_used"] = ""
            b(f"    %{p}{nm}x = index.cast {nib} : i32 to index")
            b(f"    %{p}{nm}c = index.assume %{p}{nm}x [range(%{p}{nm}x, 0, 16)] : index")
            b(f"    %{p}{nm} = view.load %smask[%{p}{nm}c] : view<16xi32> -> i32")
            return f"%{p}{nm}"
        b(f"    %{p}{nm}a = scalar.muli {nib}, {e.cs(2113665)} : i32")          # 0x00204081
        b(f"    %{p}{nm}b = scalar.andi %{p}{nm}a, {e.cs(16843009)} : i32")      # 0x01010101
        b(f"    %{p}{nm} = scalar.muli %{p}{nm}b, {e.cs(255)} : i32")
        return f"%{p}{nm}"

    def signed_grid(p, gw, s0, s1):
        """grid words gw (vector<4xi32>, unsigned magnitudes <= 62) with sign bytes s0 (words 0, 1)
        and s1 (words 2, 3) -> 64 + sign * g per byte (borrow-free: 64 - g >= 2)"""
        ms = []
        for k, (sb, sh) in enumerate(((s0, 0), (s0, 4), (s1, 0), (s1, 4))):
            if sh:
                b(f"    %{p}nb{k}s = scalar.shrui {sb}, {e.cs(4)} : i32")
                b(f"    %{p}nb{k} = scalar.andi %{p}nb{k}s, {e.cs(15)} : i32")
            else:
                b(f"    %{p}nb{k} = scalar.andi {sb}, {e.cs(15)} : i32")
            ms.append(spread(p, f"mk{k}", f"%{p}nb{k}"))
        b(f"    %{p}mask = vector.from_elements {', '.join(ms)} : vector<4xi32>")
        g2 = wop(p, "gsh2", "shli", gw, wspl(1))
        neg = wop(p, "gn", "andi", g2, f"%{p}mask")
        t = wop(p, "gt", "addi", gw, wspl(0x40404040))
        u = wop(p, "gu", "subi", t, neg)
        return tob(p, u, "q")


    # ---------------- G2: one 32-element group per lane per iteration ----------------
    def lane_setup2(f, p):
        """lane l owns group g = l + 32 t: for 256-element blocks, block l / 8 + 4 t and
        sb32 = l % 8; for q8_0 block l + 32 t."""
        qk, bb, _ = FMT[f]
        S = {}
        if qk == 32:
            b(f"  %{p}lboff = index.mul %lane, {e.ci(bb)} : index")
            S["tstep"] = 32 * bb
            return S
        b(f"  %{p}lb = index.div %lane, {e.ci(8)} : index")
        b(f"  %{p}lboff = index.mul %{p}lb, {e.ci(bb)} : index")
        b(f"  %{p}s32 = index.rem %lane, {e.ci(8)} : index")
        b(f"  %{p}s32i = index.cast %{p}s32 : index to i32")
        S["tstep"] = 4 * bb

        def splw(name, expr_i32, n=8):
            b(f"  %{p}{name} = vector.splat {expr_i32} : vector<{n}xi32>")
            return f"%{p}{name}"
        if f in ("q4k", "q5k"):
            b(f"  %{p}s32h = index.div %{p}s32, {e.ci(2)} : index")
            b(f"  %{p}qb0 = index.mul %{p}s32h, {e.ci(32)} : index")
            b(f"  %{p}qo = index.add %{p}qb0, {e.ci(48 if f == 'q5k' else 16)} : index")
            b(f"  %{p}sha = scalar.andi %{p}s32i, {e.cs(1)} : i32")
            b(f"  %{p}shi = scalar.shli %{p}sha, {e.cs(2)} : i32")
            S["sh"] = splw("shw", f"%{p}shi")
            if f == "q5k":
                S["hsh"] = splw("hshw", f"%{p}s32i")
            b(f"  %{p}sbase = index.rem %{p}s32, {e.ci(4)} : index")
            for k, o_ in enumerate((4, 8, 12)):
                b(f"  %{p}so{k} = index.add %{p}sbase, {e.ci(o_)} : index")
            b(f"  %{p}upi = scalar.shrui %{p}s32i, {e.cs(2)} : i32")
            b(f"  %{p}up = scalar.subi {e.cs(0)}, %{p}upi : i32")
        elif f == "q6k":
            b(f"  %{p}half = index.div %{p}s32, {e.ci(4)} : index")
            b(f"  %{p}seg = index.rem %{p}s32, {e.ci(4)} : index")
            b(f"  %{p}segi = index.cast %{p}seg : index to i32")
            b(f"  %{p}s1 = index.rem %{p}seg, {e.ci(2)} : index")
            b(f"  %{p}qlb0 = index.mul %{p}half, {e.ci(64)} : index")
            b(f"  %{p}qlb1 = index.mul %{p}s1, {e.ci(32)} : index")
            b(f"  %{p}qo = index.add %{p}qlb0, %{p}qlb1 : index")
            b(f"  %{p}qsh0 = scalar.shrui %{p}segi, {e.cs(1)} : i32")
            b(f"  %{p}qshi = scalar.shli %{p}qsh0, {e.cs(2)} : i32")
            S["sh"] = splw("shw", f"%{p}qshi")
            b(f"  %{p}hshi = scalar.shli %{p}segi, {e.cs(1)} : i32")
            S["hsh"] = splw("hshw", f"%{p}hshi")
            b(f"  %{p}hb0 = index.mul %{p}half, {e.ci(32)} : index")
            b(f"  %{p}ho = index.add %{p}hb0, {e.ci(128)} : index")
            b(f"  %{p}sc0 = index.mul %{p}half, {e.ci(8)} : index")
            b(f"  %{p}sc1 = index.mul %{p}seg, {e.ci(2)} : index")
            b(f"  %{p}sc2 = index.add %{p}sc0, %{p}sc1 : index")
            b(f"  %{p}so = index.add %{p}sc2, {e.ci(192)} : index")
        elif f == "q3k":
            b(f"  %{p}half = index.div %{p}s32, {e.ci(4)} : index")
            b(f"  %{p}sp = index.rem %{p}s32, {e.ci(4)} : index")
            b(f"  %{p}spi = index.cast %{p}sp : index to i32")
            b(f"  %{p}halfi = index.cast %{p}half : index to i32")
            b(f"  %{p}shi = scalar.shli %{p}spi, {e.cs(1)} : i32")
            S["sh"] = splw("shw", f"%{p}shi")
            b(f"  %{p}hs0 = scalar.shli %{p}halfi, {e.cs(2)} : i32")
            b(f"  %{p}hsi = scalar.addi %{p}hs0, %{p}spi : i32")
            S["hsh"] = splw("hshw", f"%{p}hsi")
            b(f"  %{p}qb0 = index.mul %{p}half, {e.ci(32)} : index")
            b(f"  %{p}qo = index.add %{p}qb0, {e.ci(32)} : index")
            b(f"  %{p}ho = index.add %{p}lb, {e.ci(0)} : index")      # placeholder, hmask at 0
            for h in (0, 1):    # si = half * 8 + sp * 2 + h
                b(f"  %{p}si0_{h} = index.mul %{p}half, {e.ci(8)} : index")
                b(f"  %{p}si1_{h} = index.mul %{p}sp, {e.ci(2)} : index")
                b(f"  %{p}si2_{h} = index.add %{p}si0_{h}, %{p}si1_{h} : index")
                b(f"  %{p}si_{h} = index.add %{p}si2_{h}, {e.ci(h)} : index")
                b(f"  %{p}sii_{h} = index.cast %{p}si_{h} : index to i32")
                b(f"  %{p}sl7_{h} = index.rem %{p}si_{h}, {e.ci(8)} : index")
                b(f"  %{p}slo_{h} = index.add %{p}sl7_{h}, {e.ci(96)} : index")
                b(f"  %{p}sls0_{h} = scalar.shrui %{p}sii_{h}, {e.cs(3)} : i32")
                b(f"  %{p}sls_{h} = scalar.shli %{p}sls0_{h}, {e.cs(2)} : i32")
                b(f"  %{p}sh4_{h} = index.rem %{p}si_{h}, {e.ci(4)} : index")
                b(f"  %{p}sho_{h} = index.add %{p}sh4_{h}, {e.ci(104)} : index")
                b(f"  %{p}shs0_{h} = scalar.shrui %{p}sii_{h}, {e.cs(2)} : i32")
                b(f"  %{p}shs_{h} = scalar.shli %{p}shs0_{h}, {e.cs(1)} : i32")
        elif f == "q2k":
            b(f"  %{p}l4 = index.rem %lane, {e.ci(4)} : index")
            b(f"  %{p}l4i = index.cast %{p}l4 : index to i32")
            b(f"  %{p}shi = scalar.shli %{p}l4i, {e.cs(1)} : i32")
            S["sh"] = splw("shw", f"%{p}shi")
            b(f"  %{p}sd4 = index.div %{p}s32, {e.ci(4)} : index")
            b(f"  %{p}qb0 = index.mul %{p}sd4, {e.ci(32)} : index")
            b(f"  %{p}qo = index.add %{p}qb0, {e.ci(16)} : index")
            b(f"  %{p}so = index.mul %{p}s32, {e.ci(2)} : index")          # scales[2 sb32 + h]
        elif f == "iq4xs":
            b(f"  %{p}qb0 = index.mul %{p}s32, {e.ci(16)} : index")
            b(f"  %{p}qo = index.add %{p}qb0, {e.ci(8)} : index")
            b(f"  %{p}sl0 = index.div %{p}s32, {e.ci(2)} : index")
            b(f"  %{p}so = index.add %{p}sl0, {e.ci(4)} : index")
            b(f"  %{p}sl1 = scalar.andi %{p}s32i, {e.cs(1)} : i32")
            b(f"  %{p}sls = scalar.shli %{p}sl1, {e.cs(2)} : i32")
            b(f"  %{p}shs = scalar.shli %{p}s32i, {e.cs(1)} : i32")
        elif f == "iq3s":
            b(f"  %{p}qb0 = index.mul %{p}s32, {e.ci(8)} : index")
            b(f"  %{p}qo = index.add %{p}qb0, {e.ci(2)} : index")
            b(f"  %{p}ho = index.add %{p}s32, {e.ci(66)} : index")
            b(f"  %{p}sg0 = index.mul %{p}s32, {e.ci(4)} : index")
            b(f"  %{p}sgo = index.add %{p}sg0, {e.ci(74)} : index")
            b(f"  %{p}sl0 = index.div %{p}s32, {e.ci(2)} : index")
            b(f"  %{p}so = index.add %{p}sl0, {e.ci(106)} : index")
            b(f"  %{p}sl1 = scalar.andi %{p}s32i, {e.cs(1)} : i32")
            b(f"  %{p}sls = scalar.shli %{p}sl1, {e.cs(2)} : i32")
        elif f == "iq3xxs":
            b(f"  %{p}qb0 = index.mul %{p}s32, {e.ci(8)} : index")
            b(f"  %{p}qo = index.add %{p}qb0, {e.ci(2)} : index")
            b(f"  %{p}ax0 = index.mul %{p}s32, {e.ci(4)} : index")
            b(f"  %{p}ao = index.add %{p}ax0, {e.ci(66)} : index")
        elif f in ("iq2xxs", "iq2xs"):
            b(f"  %{p}qb0 = index.mul %{p}s32, {e.ci(8)} : index")
            b(f"  %{p}qo = index.add %{p}qb0, {e.ci(2)} : index")
            if f == "iq2xs":
                b(f"  %{p}so = index.add %{p}s32, {e.ci(66)} : index")
        else:
            raise SystemExit("lane_setup2 " + f)
        return S

    def wspl8(v):
        nm = f"%wt{v}".replace("-", "m")
        c = e.cs(v)
        e.consts[nm] = f"  {nm} = vector.splat {c} : vector<8xi32>"
        return nm

    def halves(p, w8, nm):
        """vector<8xi32> -> two vector<16xi8> (words 0..3, 4..7)"""
        out = []
        for h in (0, 1):
            b(f"    %{p}{nm}w{h} = vector.slice {w8}[{4 * h}] : vector<8xi32> -> vector<4xi32>")
            out.append(tob(p, f"%{p}{nm}w{h}", f"{nm}b{h}"))
        return out

    def ld32w(p, wv, nb, boff, off, nm):
        b(f"    %{p}{nm}a = index.add {boff}, {off} : index")
        v = ld(wv, nb, f"%{p}{nm}a", p, nm + "v", 32)
        b(f"    %{p}{nm} = vector.bitcast {v} : vector<32xi8> to vector<8xi32>")
        return f"%{p}{nm}"

    def kscale(p, low, mid, high, lp):
        """Q4_K / Q5_K 6-bit scale and min (HIP SharedScales form)"""
        b(f"    %{p}lsc = scalar.andi {low}, {e.cs(63)} : i32")
        b(f"    %{p}usc0 = scalar.andi {high}, {e.cs(15)} : i32")
        b(f"    %{p}usc1 = scalar.shrui {low}, {e.cs(6)} : i32")
        b(f"    %{p}usc2 = scalar.shli %{p}usc1, {e.cs(4)} : i32")
        b(f"    %{p}usc = scalar.ori %{p}usc0, %{p}usc2 : i32")
        b(f"    %{p}lmn = scalar.andi {mid}, {e.cs(63)} : i32")
        b(f"    %{p}umn0 = scalar.shrui {high}, {e.cs(4)} : i32")
        b(f"    %{p}umn1 = scalar.shrui {mid}, {e.cs(6)} : i32")
        b(f"    %{p}umn2 = scalar.shli %{p}umn1, {e.cs(4)} : i32")
        b(f"    %{p}umn = scalar.ori %{p}umn0, %{p}umn2 : i32")
        for nm, lo_, hi_ in (("sc", "lsc", "usc"), ("mn", "lmn", "umn")):
            b(f"    %{p}{nm}x = scalar.xori %{p}{lo_}, %{p}{hi_} : i32")
            b(f"    %{p}{nm}m = scalar.andi %{p}{nm}x, %{lp}up : i32")
            b(f"    %{p}{nm} = scalar.xori %{p}{lo_}, %{p}{nm}m : i32")
        return f"%{p}sc", f"%{p}mn"

    def decode2(f, S, p, wv, hv, nb, boff, lp):
        """both 16-element halves of the lane's 32-element group:
        [(unsigned codes vector<16xi8>, scale, total offset)] x 2"""
        nh = nb // 2
        res = []
        if f == "q8_0":
            w = ld32w(p, wv, nb, boff, e.ci(2), "q")
            b(f"    %{p}qx = vector.xori {w}, {wspl8(-2139062144)} : vector<8xi32>")   # + 128 per byte
            qs = halves(p, f"%{p}qx", "q")
            d = ld_d(hv, nh, boff, 0, p, "d")
            b(f"    %{p}bo_ = scalar.mulf {d}, {e.cs(128.0, 'f32')} : f32")
            return [(qs[0], d, f"%{p}bo_"), (qs[1], d, f"%{p}bo_")]
        if f in ("q4k", "q5k", "q2k"):
            w = ld32w(p, wv, nb, boff, f"%{lp}qo", "qr")
            b(f"    %{p}qs = vector.shrui {w}, {S['sh']} : vector<8xi32>")
            b(f"    %{p}ql = vector.andi %{p}qs, {wspl8(0x0F0F0F0F if f != 'q2k' else 0x03030303)} : vector<8xi32>")
            lo = f"%{p}ql"
            if f == "q5k":
                hr = ld32w(p, wv, nb, boff, e.ci(16), "hr")
                b(f"    %{p}hs = vector.shrui {hr}, {S['hsh']} : vector<8xi32>")
                b(f"    %{p}hb = vector.andi %{p}hs, {wspl8(0x01010101)} : vector<8xi32>")
                b(f"    %{p}h4 = vector.shli %{p}hb, {wspl8(4)} : vector<8xi32>")
                b(f"    %{p}q5 = vector.ori %{p}ql, %{p}h4 : vector<8xi32>")
                lo = f"%{p}q5"
            qs = halves(p, lo, "q")
            if f == "q2k":
                b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
                sv = ld(wv, nb, f"%{p}sa", p, "scv", 2)
                d = ld_d(hv, nh, boff, 80, p, "d")
                dm = ld_d(hv, nh, boff, 82, p, "dm")
                for h in (0, 1):
                    b(f"    %{p}sc8_{h} = vector.extract {sv}[{h}] : vector<2xi8> -> i8")
                    b(f"    %{p}sc_{h} = scalar.extui %{p}sc8_{h} : i8 to i32")
                    b(f"    %{p}sl_{h} = scalar.andi %{p}sc_{h}, {e.cs(15)} : i32")
                    b(f"    %{p}sh_{h} = scalar.shrui %{p}sc_{h}, {e.cs(4)} : i32")
                    b(f"    %{p}slf_{h} = scalar.sitofp %{p}sl_{h} : i32 to f32")
                    b(f"    %{p}shf_{h} = scalar.sitofp %{p}sh_{h} : i32 to f32")
                    b(f"    %{p}scale_{h} = scalar.mulf {d}, %{p}slf_{h} : f32")
                    b(f"    %{p}offs_{h} = scalar.mulf {dm}, %{p}shf_{h} : f32")
                    res.append((qs[h], f"%{p}scale_{h}", f"%{p}offs_{h}"))
                return res
            vals = []
            for k in range(3):
                b(f"    %{p}sa{k} = index.add {boff}, %{lp}so{k} : index")
                vals.append(ld_u8(wv, nb, f"%{p}sa{k}", p, f"sb{k}"))
            sc, mn = kscale(p, *vals, lp)
            d = ld_d(hv, nh, boff, 0, p, "d")
            dm = ld_d(hv, nh, boff, 2, p, "dm")
            b(f"    %{p}scf = scalar.sitofp {sc} : i32 to f32")
            b(f"    %{p}mnf = scalar.sitofp {mn} : i32 to f32")
            b(f"    %{p}scale = scalar.mulf {d}, %{p}scf : f32")
            b(f"    %{p}offs = scalar.mulf {dm}, %{p}mnf : f32")
            return [(qs[0], f"%{p}scale", f"%{p}offs"), (qs[1], f"%{p}scale", f"%{p}offs")]
        if f in ("q6k", "q3k"):
            lr = ld32w(p, wv, nb, boff, f"%{lp}qo", "lr")
            hr = ld32w(p, wv, nb, boff, f"%{lp}ho" if f == "q6k" else e.ci(0), "hr")
            b(f"    %{p}ls = vector.shrui {lr}, {S['sh']} : vector<8xi32>")
            b(f"    %{p}hs = vector.shrui {hr}, {S['hsh']} : vector<8xi32>")
            if f == "q6k":
                b(f"    %{p}lo = vector.andi %{p}ls, {wspl8(0x0F0F0F0F)} : vector<8xi32>")
                b(f"    %{p}hb = vector.andi %{p}hs, {wspl8(0x03030303)} : vector<8xi32>")
                b(f"    %{p}h4 = vector.shli %{p}hb, {wspl8(4)} : vector<8xi32>")
            else:
                b(f"    %{p}lo = vector.andi %{p}ls, {wspl8(0x03030303)} : vector<8xi32>")
                b(f"    %{p}hb = vector.andi %{p}hs, {wspl8(0x01010101)} : vector<8xi32>")
                b(f"    %{p}h4 = vector.shli %{p}hb, {wspl8(2)} : vector<8xi32>")
            b(f"    %{p}u = vector.ori %{p}lo, %{p}h4 : vector<8xi32>")
            qs = halves(p, f"%{p}u", "q")
            if f == "q6k":
                b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
                sv = ld(wv, nb, f"%{p}sa", p, "scv", 2)
                d = ld_d(hv, nh, boff, 208, p, "d")
                for h in (0, 1):
                    b(f"    %{p}s8_{h} = vector.extract {sv}[{h}] : vector<2xi8> -> i8")
                    b(f"    %{p}si_{h} = scalar.extsi %{p}s8_{h} : i8 to i32")
                    b(f"    %{p}sf_{h} = scalar.sitofp %{p}si_{h} : i32 to f32")
                    b(f"    %{p}scale_{h} = scalar.mulf {d}, %{p}sf_{h} : f32")
                    b(f"    %{p}bo_{h} = scalar.mulf %{p}scale_{h}, {e.cs(32.0, 'f32')} : f32")
                    res.append((qs[h], f"%{p}scale_{h}", f"%{p}bo_{h}"))
                return res
            d = ld_d(hv, nh, boff, 108, p, "d")
            for h in (0, 1):
                b(f"    %{p}sla_{h} = index.add {boff}, %{lp}slo_{h} : index")
                lb = ld_u8(wv, nb, f"%{p}sla_{h}", p, f"slb_{h}")
                b(f"    %{p}sha_{h} = index.add {boff}, %{lp}sho_{h} : index")
                hb2 = ld_u8(wv, nb, f"%{p}sha_{h}", p, f"shb_{h}")
                b(f"    %{p}lw4_{h} = scalar.shrui {lb}, %{lp}sls_{h} : i32")
                b(f"    %{p}l4_{h} = scalar.andi %{p}lw4_{h}, {e.cs(15)} : i32")
                b(f"    %{p}hw2_{h} = scalar.shrui {hb2}, %{lp}shs_{h} : i32")
                b(f"    %{p}h2_{h} = scalar.andi %{p}hw2_{h}, {e.cs(3)} : i32")
                b(f"    %{p}h2s_{h} = scalar.shli %{p}h2_{h}, {e.cs(4)} : i32")
                b(f"    %{p}s6_{h} = scalar.ori %{p}l4_{h}, %{p}h2s_{h} : i32")
                b(f"    %{p}sc_{h} = scalar.subi %{p}s6_{h}, {e.cs(32)} : i32")
                b(f"    %{p}scf_{h} = scalar.sitofp %{p}sc_{h} : i32 to f32")
                b(f"    %{p}scale_{h} = scalar.mulf {d}, %{p}scf_{h} : f32")
                b(f"    %{p}bo_{h} = scalar.mulf %{p}scale_{h}, {e.cs(4.0, 'f32')} : f32")
                res.append((qs[h], f"%{p}scale_{h}", f"%{p}bo_{h}"))
            return res
        if f == "iq4xs":
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            raw = tow(p, ld(wv, nb, f"%{p}qa", p, "qr", 16), "qw")
            qs = []
            for h in (0, 1):
                sh = wop(p, f"qs{h}", "shrui", raw, wspl(4 * h)) if h else raw
                nibw = wop(p, f"nw{h}", "andi", sh, wspl(0x0F0F0F0F))
                nib0 = tob(p, nibw, f"nb0_{h}")
                b(f"    %{p}nb{h} = vector.andi {nib0}, {e.splat8(15)} : vector<16xi8>")
                b(f"    %{p}q{h} = vector.table.lookup %kvtu[%{p}nb{h}] : vector<16xi8>, vector<16xi8> -> vector<16xi8>")
                qs.append(f"%{p}q{h}")
            b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
            sl = ld_u8(wv, nb, f"%{p}sa", p, "slb")
            b(f"    %{p}h0a = index.add {boff}, {e.ci(2)} : index")
            hv2 = ld(wv, nb, f"%{p}h0a", p, "hv", 2)
            b(f"    %{p}h08 = vector.extract {hv2}[0] : vector<2xi8> -> i8")
            b(f"    %{p}h18 = vector.extract {hv2}[1] : vector<2xi8> -> i8")
            b(f"    %{p}h0 = scalar.extui %{p}h08 : i8 to i32")
            b(f"    %{p}h1 = scalar.extui %{p}h18 : i8 to i32")
            b(f"    %{p}h1s = scalar.shli %{p}h1, {e.cs(8)} : i32")
            b(f"    %{p}shw = scalar.ori %{p}h0, %{p}h1s : i32")
            b(f"    %{p}lw = scalar.shrui {sl}, %{lp}sls : i32")
            b(f"    %{p}l4 = scalar.andi %{p}lw, {e.cs(15)} : i32")
            b(f"    %{p}hw = scalar.shrui %{p}shw, %{lp}shs : i32")
            b(f"    %{p}h2 = scalar.andi %{p}hw, {e.cs(3)} : i32")
            b(f"    %{p}h2s = scalar.shli %{p}h2, {e.cs(4)} : i32")
            b(f"    %{p}s6 = scalar.ori %{p}l4, %{p}h2s : i32")
            b(f"    %{p}sc = scalar.subi %{p}s6, {e.cs(32)} : i32")
            b(f"    %{p}scf = scalar.sitofp %{p}sc : i32 to f32")
            d = ld_d(hv, nh, boff, 0, p, "d")
            b(f"    %{p}scale = scalar.mulf {d}, %{p}scf : f32")
            b(f"    %{p}bo_ = scalar.mulf %{p}scale, {e.cs(128.0, 'f32')} : f32")
            return [(qs[0], f"%{p}scale", f"%{p}bo_"), (qs[1], f"%{p}scale", f"%{p}bo_")]
        # grid formats: 8 index bytes (iq3s / iq3xxs) or 4 + high word (iq2xxs) or 4 u16 codes (iq2xs)
        b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
        qv = ld(wv, nb, f"%{p}qa", p, "qv", 8)
        d = ld_d(hv, nh, boff, 0, p, "d")
        words, signs = [], []
        if f == "iq3s":
            b(f"    %{p}hqa = index.add {boff}, %{lp}ho : index")
            qh = ld_u8(wv, nb, f"%{p}hqa", p, "qh")
            b(f"    %{p}sga = index.add {boff}, %{lp}sgo : index")
            sg = ld_u32(wv, nb, f"%{p}sga", p, "sgw")
            for k in range(8):
                l_, which = k // 2, k % 2
                b(f"    %{p}qi8_{k} = vector.extract {qv}[{k}] : vector<8xi8> -> i8")
                b(f"    %{p}qi_{k} = scalar.extui %{p}qi8_{k} : i8 to i32")
                b(f"    %{p}qhs_{k} = scalar.shli {qh}, {e.cs(8 - 2 * l_ - which)} : i32")
                b(f"    %{p}qhb_{k} = scalar.andi %{p}qhs_{k}, {e.cs(256)} : i32")
                b(f"    %{p}gi_{k} = scalar.ori %{p}qi_{k}, %{p}qhb_{k} : i32")
                b(f"    %{p}gx_{k} = index.cast %{p}gi_{k} : i32 to index")
                words.append(ld("%t_grid_iq3s", 512, f"%{p}gx_{k}", p, f"g{k}", 1, "i32"))
            for l_ in range(4):
                b(f"    %{p}sgs{l_} = scalar.shrui {sg}, {e.cs(8 * l_)} : i32")
                b(f"    %{p}sgb{l_} = scalar.andi %{p}sgs{l_}, {e.cs(255)} : i32")
                signs.append(f"%{p}sgb{l_}")
            b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
            sb_ = ld_u8(wv, nb, f"%{p}sa", p, "scb")
            b(f"    %{p}nw = scalar.shrui {sb_}, %{lp}sls : i32")
            b(f"    %{p}nib = scalar.andi %{p}nw, {e.cs(15)} : i32")
            b(f"    %{p}n2 = scalar.shli %{p}nib, {e.cs(1)} : i32")
            b(f"    %{p}n21 = scalar.addi %{p}n2, {e.cs(1)} : i32")
            b(f"    %{p}nf = scalar.sitofp %{p}n21 : i32 to f32")
            b(f"    %{p}scale = scalar.mulf {d}, %{p}nf : f32")
            scales = [f"%{p}scale"] * 2
        elif f in ("iq3xxs", "iq2xxs"):
            if f == "iq3xxs":
                b(f"    %{p}aa = index.add {boff}, %{lp}ao : index")
                aux = ld_u32(wv, nb, f"%{p}aa", p, "aux")
                for k in range(8):
                    b(f"    %{p}qi8_{k} = vector.extract {qv}[{k}] : vector<8xi8> -> i8")
                    b(f"    %{p}qi_{k} = scalar.extui %{p}qi8_{k} : i8 to i32")
                    b(f"    %{p}gx_{k} = index.cast %{p}qi_{k} : i32 to index")
                    words.append(ld("%t_grid_iq3xxs", 256, f"%{p}gx_{k}", p, f"g{k}", 1, "i32"))
            else:
                b(f"    %{p}qw2 = vector.bitcast {qv} : vector<8xi8> to vector<2xi32>")
                b(f"    %{p}aux = vector.extract %{p}qw2[1] : vector<2xi32> -> i32")
                aux = f"%{p}aux"
                for k in range(4):
                    b(f"    %{p}qi8_{k} = vector.extract {qv}[{k}] : vector<8xi8> -> i8")
                    b(f"    %{p}qi_{k} = scalar.extui %{p}qi8_{k} : i8 to i32")
                    b(f"    %{p}gx_{k} = index.cast %{p}qi_{k} : i32 to index")
                    b(f"    %{p}gx2_{k} = index.mul %{p}gx_{k}, {e.ci(2)} : index")
                    w2 = ld("%t_grid_iq2xxs", 512, f"%{p}gx2_{k}", p, f"g{k}", 2, "i32")
                    for hh in (0, 1):
                        b(f"    %{p}g{k}_{hh} = vector.extract {w2}[{hh}] : vector<2xi32> -> i32")
                        words.append(f"%{p}g{k}_{hh}")
            for l_ in range(4):
                b(f"    %{p}ksh{l_} = scalar.shrui {aux}, {e.cs(7 * l_)} : i32")
                b(f"    %{p}ksi{l_} = scalar.andi %{p}ksh{l_}, {e.cs(127)} : i32")
                b(f"    %{p}ksx{l_} = index.cast %{p}ksi{l_} : i32 to index")
                signs.append(ld_u8("%t_ksigns", 128, f"%{p}ksx{l_}", p, f"ks{l_}"))
            b(f"    %{p}a28 = scalar.shrui {aux}, {e.cs(28)} : i32")
            b(f"    %{p}a28f = scalar.uitofp %{p}a28 : i32 to f32")
            b(f"    %{p}ah = scalar.addf %{p}a28f, {e.cs(0.5, 'f32')} : f32")
            b(f"    %{p}dah = scalar.mulf {d}, %{p}ah : f32")
            b(f"    %{p}scale = scalar.mulf %{p}dah, {e.cs(0.5 if f == 'iq3xxs' else 0.25, 'f32')} : f32")
            scales = [f"%{p}scale"] * 2
        else:   # iq2xs
            b(f"    %{p}qw2 = vector.bitcast {qv} : vector<8xi8> to vector<2xi32>")
            for l_ in range(4):
                b(f"    %{p}cw{l_} = vector.extract %{p}qw2[{l_ // 2}] : vector<2xi32> -> i32")
                if l_ % 2 == 0:
                    b(f"    %{p}cd{l_} = scalar.andi %{p}cw{l_}, {e.cs(65535)} : i32")
                else:
                    b(f"    %{p}cd{l_} = scalar.shrui %{p}cw{l_}, {e.cs(16)} : i32")
                b(f"    %{p}gi{l_} = scalar.andi %{p}cd{l_}, {e.cs(511)} : i32")
                b(f"    %{p}gix{l_} = index.cast %{p}gi{l_} : i32 to index")
                b(f"    %{p}gx2_{l_} = index.mul %{p}gix{l_}, {e.ci(2)} : index")
                w2 = ld("%t_grid_iq2xs", 1024, f"%{p}gx2_{l_}", p, f"g{l_}", 2, "i32")
                for hh in (0, 1):
                    b(f"    %{p}g{l_}_{hh} = vector.extract {w2}[{hh}] : vector<2xi32> -> i32")
                    words.append(f"%{p}g{l_}_{hh}")
                b(f"    %{p}ks{l_}i = scalar.shrui %{p}cd{l_}, {e.cs(9)} : i32")
                b(f"    %{p}ksx{l_} = index.cast %{p}ks{l_}i : i32 to index")
                signs.append(ld_u8("%t_ksigns", 128, f"%{p}ksx{l_}", p, f"ks{l_}"))
            b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
            sb_ = ld_u8(wv, nb, f"%{p}sa", p, "scb")
            scales = []
            for h in (0, 1):
                b(f"    %{p}nw_{h} = scalar.shrui {sb_}, {e.cs(4 * h)} : i32")
                b(f"    %{p}nib_{h} = scalar.andi %{p}nw_{h}, {e.cs(15)} : i32")
                b(f"    %{p}nf_{h} = scalar.uitofp %{p}nib_{h} : i32 to f32")
                b(f"    %{p}nh_{h} = scalar.addf %{p}nf_{h}, {e.cs(0.5, 'f32')} : f32")
                b(f"    %{p}dn_{h} = scalar.mulf {d}, %{p}nh_{h} : f32")
                b(f"    %{p}scale_{h} = scalar.mulf %{p}dn_{h}, {e.cs(0.25, 'f32')} : f32")
                scales.append(f"%{p}scale_{h}")
        for h in (0, 1):
            ph = f"{p}h{h}"
            b(f"    %{ph}gw = vector.from_elements " + ", ".join(words[4 * h:4 * h + 4]) + " : vector<4xi32>")
            q = signed_grid(ph, f"%{ph}gw", signs[2 * h], signs[2 * h + 1])
            b(f"    %{ph}bo_ = scalar.mulf {scales[h]}, {e.cs(64.0, 'f32')} : f32")
            res.append((q, scales[h], f"%{ph}bo_"))
        return res

    def decode_w(f, S, p, wv, hv, nb, boff, lp):
        """Word-level decode: (unsigned byte codes vector<16xi8>, scale, total offset) with
        w = scale * code - offset (biases folded into offset)."""
        nh = nb // 2
        bias = 0
        offs = None
        if f == "q8_0":
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            raw = tow(p, ld(wv, nb, f"%{p}qa", p, "q", 16), "qw")
            q = tob(p, wop(p, "qx", "xori", raw, wspl(-2139062144)), "qu")      # ^ 0x80808080: + 128
            scale = ld_d(hv, nh, boff, 0, p, "d")
            bias = 128
        elif f in ("q4k", "q5k", "q2k"):
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            raw = tow(p, ld(wv, nb, f"%{p}qa", p, "qr", 16), "qw")
            sh = wop(p, "qs", "shrui", raw, S["sh"] + "w")
            lo = wop(p, "ql", "andi", sh, wspl(0x0F0F0F0F if f != "q2k" else 0x03030303))
            if f == "q5k":
                b(f"    %{p}ha = index.add {boff}, %{lp}ho : index")
                hr = tow(p, ld(wv, nb, f"%{p}ha", p, "hr", 16), "hw")
                hs = wop(p, "hs", "shrui", hr, S["hsh"] + "w")
                hb = wop(p, "hb", "andi", hs, wspl(0x01010101))
                h4 = wop(p, "h4", "shli", hb, wspl(4))
                lo = wop(p, "q5", "ori", lo, h4)
            q = tob(p, lo, "qu")
            if f == "q2k":
                b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
                sc = ld_u8(wv, nb, f"%{p}sa", p, "scb")
                b(f"    %{p}sl = scalar.andi {sc}, {e.cs(15)} : i32")
                b(f"    %{p}sh2 = scalar.shrui {sc}, {e.cs(4)} : i32")
                b(f"    %{p}slf = scalar.sitofp %{p}sl : i32 to f32")
                b(f"    %{p}shf = scalar.sitofp %{p}sh2 : i32 to f32")
                d = ld_d(hv, nh, boff, 80, p, "d")
                dm = ld_d(hv, nh, boff, 82, p, "dm")
                b(f"    %{p}scale = scalar.mulf {d}, %{p}slf : f32")
                b(f"    %{p}offs = scalar.mulf {dm}, %{p}shf : f32")
                return q, f"%{p}scale", f"%{p}offs"
            vals = []
            for k in range(3):
                b(f"    %{p}sa{k} = index.add {boff}, %{lp}so{k} : index")
                vals.append(ld_u8(wv, nb, f"%{p}sa{k}", p, f"sb{k}"))
            low, mid, high = vals
            b(f"    %{p}lsc = scalar.andi {low}, {e.cs(63)} : i32")
            b(f"    %{p}usc0 = scalar.andi {high}, {e.cs(15)} : i32")
            b(f"    %{p}usc1 = scalar.shrui {low}, {e.cs(6)} : i32")
            b(f"    %{p}usc2 = scalar.shli %{p}usc1, {e.cs(4)} : i32")
            b(f"    %{p}usc = scalar.ori %{p}usc0, %{p}usc2 : i32")
            b(f"    %{p}lmn = scalar.andi {mid}, {e.cs(63)} : i32")
            b(f"    %{p}umn0 = scalar.shrui {high}, {e.cs(4)} : i32")
            b(f"    %{p}umn1 = scalar.shrui {mid}, {e.cs(6)} : i32")
            b(f"    %{p}umn2 = scalar.shli %{p}umn1, {e.cs(4)} : i32")
            b(f"    %{p}umn = scalar.ori %{p}umn0, %{p}umn2 : i32")
            for nm, lo_, hi_ in (("sc", "lsc", "usc"), ("mn", "lmn", "umn")):
                b(f"    %{p}{nm}x = scalar.xori %{p}{lo_}, %{p}{hi_} : i32")
                b(f"    %{p}{nm}m = scalar.andi %{p}{nm}x, %{lp}up : i32")
                b(f"    %{p}{nm} = scalar.xori %{p}{lo_}, %{p}{nm}m : i32")
            d = ld_d(hv, nh, boff, 0, p, "d")
            dm = ld_d(hv, nh, boff, 2, p, "dm")
            b(f"    %{p}scf = scalar.sitofp %{p}sc : i32 to f32")
            b(f"    %{p}mnf = scalar.sitofp %{p}mn : i32 to f32")
            b(f"    %{p}scale = scalar.mulf {d}, %{p}scf : f32")
            b(f"    %{p}offs = scalar.mulf {dm}, %{p}mnf : f32")
            return q, f"%{p}scale", f"%{p}offs"
        elif f in ("q6k", "q3k"):
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            lr = tow(p, ld(wv, nb, f"%{p}qa", p, "lr", 16), "lw")
            b(f"    %{p}ha = index.add {boff}, %{lp}ho : index")
            hr = tow(p, ld(wv, nb, f"%{p}ha", p, "hr", 16), "hw")
            ls = wop(p, "ls", "shrui", lr, S["sh"] + "w")
            hs = wop(p, "hs", "shrui", hr, S["hsh"] + "w")
            if f == "q6k":
                lo = wop(p, "lo", "andi", ls, wspl(0x0F0F0F0F))
                hb = wop(p, "hb", "andi", hs, wspl(0x03030303))
                h4 = wop(p, "h4", "shli", hb, wspl(4))
                q = tob(p, wop(p, "u", "ori", lo, h4), "qu")
                bias = 32
                b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
                s8 = ld(wv, nb, f"%{p}sa", p, "s8")
                b(f"    %{p}si = scalar.extsi {s8} : i8 to i32")
                b(f"    %{p}sf = scalar.sitofp %{p}si : i32 to f32")
                d = ld_d(hv, nh, boff, 208, p, "d")
                b(f"    %{p}scale0 = scalar.mulf {d}, %{p}sf : f32")
            else:
                lo = wop(p, "lo", "andi", ls, wspl(0x03030303))
                hb = wop(p, "hb", "andi", hs, wspl(0x01010101))
                h4 = wop(p, "h4", "shli", hb, wspl(2))
                q = tob(p, wop(p, "u", "ori", lo, h4), "qu")
                bias = 4
                b(f"    %{p}sla = index.add {boff}, %{lp}slo : index")
                lb = ld_u8(wv, nb, f"%{p}sla", p, "slb")
                b(f"    %{p}sha = index.add {boff}, %{lp}sho : index")
                hb2 = ld_u8(wv, nb, f"%{p}sha", p, "shb")
                b(f"    %{p}lw4 = scalar.shrui {lb}, %{lp}sls : i32")
                b(f"    %{p}l4 = scalar.andi %{p}lw4, {e.cs(15)} : i32")
                b(f"    %{p}hw2 = scalar.shrui {hb2}, %{lp}shs : i32")
                b(f"    %{p}h2 = scalar.andi %{p}hw2, {e.cs(3)} : i32")
                b(f"    %{p}h2s = scalar.shli %{p}h2, {e.cs(4)} : i32")
                b(f"    %{p}s6 = scalar.ori %{p}l4, %{p}h2s : i32")
                b(f"    %{p}sc = scalar.subi %{p}s6, {e.cs(32)} : i32")
                b(f"    %{p}scf = scalar.sitofp %{p}sc : i32 to f32")
                d = ld_d(hv, nh, boff, 108, p, "d")
                b(f"    %{p}scale0 = scalar.mulf {d}, %{p}scf : f32")
            scale = f"%{p}scale0"
        elif f == "iq4xs":
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            raw = tow(p, ld(wv, nb, f"%{p}qa", p, "qr", 16), "qw")
            sh = wop(p, "qs", "shrui", raw, S["sh"] + "w")
            nibw = wop(p, "nw", "andi", sh, wspl(0x0F0F0F0F))
            nib0 = tob(p, nibw, "nb0")
            b(f"    %{p}nb = vector.andi {nib0}, {e.splat8(15)} : vector<16xi8>")   # index range for the lookup
            b(f"    %{p}q = vector.table.lookup %kvtu[%{p}nb] : vector<16xi8>, vector<16xi8> -> vector<16xi8>")
            q = f"%{p}q"
            bias = 128
            b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
            sl = ld_u8(wv, nb, f"%{p}sa", p, "slb")
            b(f"    %{p}h0a = index.add {boff}, {e.ci(2)} : index")
            h0 = ld_u8(wv, nb, f"%{p}h0a", p, "h0")
            b(f"    %{p}h1a = index.add {boff}, {e.ci(3)} : index")
            h1 = ld_u8(wv, nb, f"%{p}h1a", p, "h1")
            b(f"    %{p}h1s = scalar.shli {h1}, {e.cs(8)} : i32")
            b(f"    %{p}shw = scalar.ori {h0}, %{p}h1s : i32")
            b(f"    %{p}lw = scalar.shrui {sl}, %{lp}sls : i32")
            b(f"    %{p}l4 = scalar.andi %{p}lw, {e.cs(15)} : i32")
            b(f"    %{p}hw = scalar.shrui %{p}shw, %{lp}shs : i32")
            b(f"    %{p}h2 = scalar.andi %{p}hw, {e.cs(3)} : i32")
            b(f"    %{p}h2s = scalar.shli %{p}h2, {e.cs(4)} : i32")
            b(f"    %{p}s6 = scalar.ori %{p}l4, %{p}h2s : i32")
            b(f"    %{p}sc = scalar.subi %{p}s6, {e.cs(32)} : i32")
            b(f"    %{p}scf = scalar.sitofp %{p}sc : i32 to f32")
            d = ld_d(hv, nh, boff, 0, p, "d")
            b(f"    %{p}scale0 = scalar.mulf {d}, %{p}scf : f32")
            scale = f"%{p}scale0"
        elif f in ("iq3s", "iq3xxs", "iq2xxs", "iq2xs"):
            # reuse the byte-path grid gathers / sign bytes / scale, then word sign application
            q_old, scale, _ = decode_grid_parts(f, S, p, wv, hv, nb, boff, lp)
            gw, s0, s1 = q_old
            q = signed_grid(p, gw, s0, s1)
            bias = 64
        else:
            raise SystemExit("decode_w " + f)
        b(f"    %{p}bo_ = scalar.mulf {scale}, {e.cs(float(bias), 'f32')} : f32")
        return q, scale, f"%{p}bo_"

    def decode_grid_parts(f, S, p, wv, hv, nb, boff, lp):
        return decode(f, S, p, wv, hv, nb, boff, lp, parts=True)

    def decode(f, S, p, wv, hv, nb, boff, lp, parts=False):
        """Emit one sub-block decode; return (q vector<16xi8>, scale f32, offset f32 or None)."""
        nh = nb // 2
        if f == "q8_0":
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            q = ld(wv, nb, f"%{p}qa", p, "q", 16)
            return q, ld_d(hv, nh, boff, 0, p, "d"), None
        if f in ("q4k", "q5k"):
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            raw = ld(wv, nb, f"%{p}qa", p, "qr", 16)
            b(f"    %{p}qs = vector.shrui {raw}, {S['sh']} : vector<16xi8>")
            b(f"    %{p}ql = vector.andi %{p}qs, {e.splat8(15)} : vector<16xi8>")
            q = f"%{p}ql"
            if f == "q5k":
                b(f"    %{p}ha = index.add {boff}, %{lp}ho : index")
                hr = ld(wv, nb, f"%{p}ha", p, "hr", 16)
                b(f"    %{p}hs = vector.shrui {hr}, {S['hsh']} : vector<16xi8>")
                b(f"    %{p}hb = vector.andi %{p}hs, {e.splat8(1)} : vector<16xi8>")
                b(f"    %{p}h4 = vector.shli %{p}hb, {e.splat8(4)} : vector<16xi8>")
                b(f"    %{p}q5 = vector.ori %{p}ql, %{p}h4 : vector<16xi8>")
                q = f"%{p}q5"
            vals = []
            for k in range(3):
                b(f"    %{p}sa{k} = index.add {boff}, %{lp}so{k} : index")
                vals.append(ld_u8(wv, nb, f"%{p}sa{k}", p, f"sb{k}"))
            low, mid, high = vals
            b(f"    %{p}lsc = scalar.andi {low}, {e.cs(63)} : i32")
            b(f"    %{p}usc0 = scalar.andi {high}, {e.cs(15)} : i32")
            b(f"    %{p}usc1 = scalar.shrui {low}, {e.cs(6)} : i32")
            b(f"    %{p}usc2 = scalar.shli %{p}usc1, {e.cs(4)} : i32")
            b(f"    %{p}usc = scalar.ori %{p}usc0, %{p}usc2 : i32")
            b(f"    %{p}lmn = scalar.andi {mid}, {e.cs(63)} : i32")
            b(f"    %{p}umn0 = scalar.shrui {high}, {e.cs(4)} : i32")
            b(f"    %{p}umn1 = scalar.shrui {mid}, {e.cs(6)} : i32")
            b(f"    %{p}umn2 = scalar.shli %{p}umn1, {e.cs(4)} : i32")
            b(f"    %{p}umn = scalar.ori %{p}umn0, %{p}umn2 : i32")
            for nm, lo, hi in (("sc", "lsc", "usc"), ("mn", "lmn", "umn")):   # lo ^ ((lo ^ hi) & up)
                b(f"    %{p}{nm}x = scalar.xori %{p}{lo}, %{p}{hi} : i32")
                b(f"    %{p}{nm}m = scalar.andi %{p}{nm}x, %{lp}up : i32")
                b(f"    %{p}{nm} = scalar.xori %{p}{lo}, %{p}{nm}m : i32")
            d = ld_d(hv, nh, boff, 0, p, "d")
            dm = ld_d(hv, nh, boff, 2, p, "dm")
            b(f"    %{p}scf = scalar.sitofp %{p}sc : i32 to f32")
            b(f"    %{p}mnf = scalar.sitofp %{p}mn : i32 to f32")
            b(f"    %{p}scale = scalar.mulf {d}, %{p}scf : f32")
            b(f"    %{p}offs = scalar.mulf {dm}, %{p}mnf : f32")
            return q, f"%{p}scale", f"%{p}offs"
        if f == "q6k":
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            lr = ld(wv, nb, f"%{p}qa", p, "lr", 16)
            b(f"    %{p}ha = index.add {boff}, %{lp}ho : index")
            hr = ld(wv, nb, f"%{p}ha", p, "hr", 16)
            b(f"    %{p}ls = vector.shrui {lr}, {S['sh']} : vector<16xi8>")
            b(f"    %{p}lo = vector.andi %{p}ls, {e.splat8(15)} : vector<16xi8>")
            b(f"    %{p}hs = vector.shrui {hr}, {S['hsh']} : vector<16xi8>")
            b(f"    %{p}hb = vector.andi %{p}hs, {e.splat8(3)} : vector<16xi8>")
            b(f"    %{p}h4 = vector.shli %{p}hb, {e.splat8(4)} : vector<16xi8>")
            b(f"    %{p}u = vector.ori %{p}lo, %{p}h4 : vector<16xi8>")
            b(f"    %{p}q = vector.subi %{p}u, {e.splat8(32)} : vector<16xi8>")
            b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
            s8 = ld(wv, nb, f"%{p}sa", p, "s8")
            b(f"    %{p}si = scalar.extsi {s8} : i8 to i32")
            b(f"    %{p}sf = scalar.sitofp %{p}si : i32 to f32")
            d = ld_d(hv, nh, boff, 208, p, "d")
            b(f"    %{p}scale = scalar.mulf {d}, %{p}sf : f32")
            return f"%{p}q", f"%{p}scale", None
        if f == "q3k":
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            lr = ld(wv, nb, f"%{p}qa", p, "lr", 16)
            b(f"    %{p}ha = index.add {boff}, %{lp}ho : index")
            hr = ld(wv, nb, f"%{p}ha", p, "hr", 16)
            b(f"    %{p}ls = vector.shrui {lr}, {S['sh']} : vector<16xi8>")
            b(f"    %{p}lo = vector.andi %{p}ls, {e.splat8(3)} : vector<16xi8>")
            b(f"    %{p}hs = vector.shrui {hr}, {S['hsh']} : vector<16xi8>")
            b(f"    %{p}hb = vector.andi %{p}hs, {e.splat8(1)} : vector<16xi8>")
            b(f"    %{p}h4 = vector.shli %{p}hb, {e.splat8(2)} : vector<16xi8>")
            b(f"    %{p}u = vector.addi %{p}lo, %{p}h4 : vector<16xi8>")
            b(f"    %{p}q = vector.subi %{p}u, {e.splat8(4)} : vector<16xi8>")
            b(f"    %{p}sla = index.add {boff}, %{lp}slo : index")
            lb = ld_u8(wv, nb, f"%{p}sla", p, "slb")
            b(f"    %{p}sha = index.add {boff}, %{lp}sho : index")
            hb = ld_u8(wv, nb, f"%{p}sha", p, "shb")
            b(f"    %{p}lw = scalar.shrui {lb}, %{lp}sls : i32")
            b(f"    %{p}l4 = scalar.andi %{p}lw, {e.cs(15)} : i32")
            b(f"    %{p}hw = scalar.shrui {hb}, %{lp}shs : i32")
            b(f"    %{p}h2 = scalar.andi %{p}hw, {e.cs(3)} : i32")
            b(f"    %{p}h2s = scalar.shli %{p}h2, {e.cs(4)} : i32")
            b(f"    %{p}s6 = scalar.ori %{p}l4, %{p}h2s : i32")
            b(f"    %{p}sc = scalar.subi %{p}s6, {e.cs(32)} : i32")
            b(f"    %{p}scf = scalar.sitofp %{p}sc : i32 to f32")
            d = ld_d(hv, nh, boff, 108, p, "d")
            b(f"    %{p}scale = scalar.mulf {d}, %{p}scf : f32")
            return f"%{p}q", f"%{p}scale", None
        if f == "q2k":
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            raw = ld(wv, nb, f"%{p}qa", p, "qr", 16)
            b(f"    %{p}qs = vector.shrui {raw}, {S['sh']} : vector<16xi8>")
            b(f"    %{p}q = vector.andi %{p}qs, {e.splat8(3)} : vector<16xi8>")
            b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
            sc = ld_u8(wv, nb, f"%{p}sa", p, "scb")
            b(f"    %{p}sl = scalar.andi {sc}, {e.cs(15)} : i32")
            b(f"    %{p}sh = scalar.shrui {sc}, {e.cs(4)} : i32")
            b(f"    %{p}slf = scalar.sitofp %{p}sl : i32 to f32")
            b(f"    %{p}shf = scalar.sitofp %{p}sh : i32 to f32")
            d = ld_d(hv, nh, boff, 80, p, "d")
            dm = ld_d(hv, nh, boff, 82, p, "dm")
            b(f"    %{p}scale = scalar.mulf {d}, %{p}slf : f32")
            b(f"    %{p}offs = scalar.mulf {dm}, %{p}shf : f32")
            return f"%{p}q", f"%{p}scale", f"%{p}offs"
        if f == "iq4xs":
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            raw = ld(wv, nb, f"%{p}qa", p, "qr", 16)
            b(f"    %{p}qs = vector.shrui {raw}, {S['sh']} : vector<16xi8>")
            b(f"    %{p}nb = vector.andi %{p}qs, {e.splat8(15)} : vector<16xi8>")
            b(f"    %{p}q = vector.table.lookup %kvt[%{p}nb] : vector<16xi8>, vector<16xi8> -> vector<16xi8>")
            b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
            sl = ld_u8(wv, nb, f"%{p}sa", p, "slb")
            b(f"    %{p}h0a = index.add {boff}, {e.ci(2)} : index")
            h0 = ld_u8(wv, nb, f"%{p}h0a", p, "h0")
            b(f"    %{p}h1a = index.add {boff}, {e.ci(3)} : index")
            h1 = ld_u8(wv, nb, f"%{p}h1a", p, "h1")
            b(f"    %{p}h1s = scalar.shli {h1}, {e.cs(8)} : i32")
            b(f"    %{p}shw = scalar.ori {h0}, %{p}h1s : i32")
            b(f"    %{p}lw = scalar.shrui {sl}, %{lp}sls : i32")
            b(f"    %{p}l4 = scalar.andi %{p}lw, {e.cs(15)} : i32")
            b(f"    %{p}hw = scalar.shrui %{p}shw, %{lp}shs : i32")
            b(f"    %{p}h2 = scalar.andi %{p}hw, {e.cs(3)} : i32")
            b(f"    %{p}h2s = scalar.shli %{p}h2, {e.cs(4)} : i32")
            b(f"    %{p}s6 = scalar.ori %{p}l4, %{p}h2s : i32")
            b(f"    %{p}sc = scalar.subi %{p}s6, {e.cs(32)} : i32")
            b(f"    %{p}scf = scalar.sitofp %{p}sc : i32 to f32")
            d = ld_d(hv, nh, boff, 0, p, "d")
            b(f"    %{p}scale = scalar.mulf {d}, %{p}scf : f32")
            return f"%{p}q", f"%{p}scale", None
        if f == "iq3s":
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            qv = ld(wv, nb, f"%{p}qa", p, "qv", 4)                       # qs[sb32*8 + 2l .. + 4)
            b(f"    %{p}hqa = index.add {boff}, %{lp}ho : index")
            qh = ld_u8(wv, nb, f"%{p}hqa", p, "qh")
            words = []
            for qq in (0, 1):
                for gi, shn in ((0, "A"), (1, "B")):
                    k = 2 * qq + gi
                    b(f"    %{p}qi8_{k} = vector.extract {qv}[{k}] : vector<4xi8> -> i8")
                    b(f"    %{p}qi_{k} = scalar.extui %{p}qi8_{k} : i8 to i32")
                    b(f"    %{p}qhs_{k} = scalar.shli {qh}, %{lp}qs{shn}{qq} : i32")
                    b(f"    %{p}qhb_{k} = scalar.andi %{p}qhs_{k}, {e.cs(256)} : i32")
                    b(f"    %{p}gi_{k} = scalar.ori %{p}qi_{k}, %{p}qhb_{k} : i32")
                    b(f"    %{p}gx_{k} = index.cast %{p}gi_{k} : i32 to index")
                    words.append(ld("%t_grid_iq3s", 512, f"%{p}gx_{k}", p, f"g{k}", 1, "i32"))
            b(f"    %{p}gw = vector.from_elements " + ", ".join(words) + " : vector<4xi32>")
            b(f"    %{p}g = vector.bitcast %{p}gw : vector<4xi32> to vector<16xi8>")
            sg = []
            for qq in (0, 1):
                b(f"    %{p}sga{qq} = index.add {boff}, %{lp}sgo : index")
                b(f"    %{p}sgb{qq} = index.add %{p}sga{qq}, {e.ci(qq)} : index")
                sg.append(ld_u8(wv, nb, f"%{p}sgb{qq}", p, f"sgn{qq}"))
            if parts:
                q = (f"%{p}gw", sg[0], sg[1])
            else:
                neg = signs16(p, sg[0], sg[1])
                q = apply_signs(p, f"%{p}g", neg)
            b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
            sb_ = ld_u8(wv, nb, f"%{p}sa", p, "scb")
            b(f"    %{p}nw = scalar.shrui {sb_}, %{lp}sls : i32")
            b(f"    %{p}nib = scalar.andi %{p}nw, {e.cs(15)} : i32")
            b(f"    %{p}n2 = scalar.shli %{p}nib, {e.cs(1)} : i32")
            b(f"    %{p}n21 = scalar.addi %{p}n2, {e.cs(1)} : i32")
            b(f"    %{p}nf = scalar.sitofp %{p}n21 : i32 to f32")
            d = ld_d(hv, nh, boff, 0, p, "d")
            b(f"    %{p}scale = scalar.mulf {d}, %{p}nf : f32")
            return q, f"%{p}scale", None
        if f in ("iq3xxs", "iq2xxs"):
            b(f"    %{p}aa = index.add {boff}, %{lp}ao : index")
            aux = ld_u32(wv, nb, f"%{p}aa", p, "aux")
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            nq = 4 if f == "iq3xxs" else 2
            qv = ld(wv, nb, f"%{p}qa", p, "qv", nq)
            words = []
            gname, gn = ("%t_grid_iq3xxs", 256) if f == "iq3xxs" else ("%t_grid_iq2xxs", 512)
            for k in range(nq):
                b(f"    %{p}qi8_{k} = vector.extract {qv}[{k}] : vector<{nq}xi8> -> i8")
                b(f"    %{p}qi_{k} = scalar.extui %{p}qi8_{k} : i8 to i32")
                b(f"    %{p}gx_{k} = index.cast %{p}qi_{k} : i32 to index")
                if f == "iq3xxs":
                    words.append(ld(gname, gn, f"%{p}gx_{k}", p, f"g{k}", 1, "i32"))
                else:      # 8-byte grid entry = words 2k, 2k + 1
                    b(f"    %{p}gx2_{k} = index.mul %{p}gx_{k}, {e.ci(2)} : index")
                    w2 = ld(gname, gn, f"%{p}gx2_{k}", p, f"g{k}", 2, "i32")
                    for h in (0, 1):
                        b(f"    %{p}g{k}_{h} = vector.extract {w2}[{h}] : vector<2xi32> -> i32")
                        words.append(f"%{p}g{k}_{h}")
            b(f"    %{p}gw = vector.from_elements " + ", ".join(words) + " : vector<4xi32>")
            b(f"    %{p}g = vector.bitcast %{p}gw : vector<4xi32> to vector<16xi8>")
            sg = []
            for li in (0, 1):
                b(f"    %{p}ksh{li} = scalar.shrui {aux}, %{lp}ls{li} : i32")
                b(f"    %{p}ksi{li} = scalar.andi %{p}ksh{li}, {e.cs(127)} : i32")
                b(f"    %{p}ksx{li} = index.cast %{p}ksi{li} : i32 to index")
                sg.append(ld_u8("%t_ksigns", 128, f"%{p}ksx{li}", p, f"ks{li}"))
            if parts:
                q = (f"%{p}gw", sg[0], sg[1])
            else:
                neg = signs16(p, sg[0], sg[1])
                q = apply_signs(p, f"%{p}g", neg)
            b(f"    %{p}a28 = scalar.shrui {aux}, {e.cs(28)} : i32")
            b(f"    %{p}a28f = scalar.uitofp %{p}a28 : i32 to f32")
            b(f"    %{p}ah = scalar.addf %{p}a28f, {e.cs(0.5, 'f32')} : f32")
            d = ld_d(hv, nh, boff, 0, p, "d")
            b(f"    %{p}dah = scalar.mulf {d}, %{p}ah : f32")
            b(f"    %{p}scale = scalar.mulf %{p}dah, {e.cs(0.5 if f == 'iq3xxs' else 0.25, 'f32')} : f32")
            return q, f"%{p}scale", None
        if f == "iq2xs":
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            qv = ld(wv, nb, f"%{p}qa", p, "qv", 4)                      # two u16 codes
            b(f"    %{p}qw = vector.bitcast {qv} : vector<4xi8> to vector<1xi32>")
            b(f"    %{p}qww = vector.extract %{p}qw[0] : vector<1xi32> -> i32")
            words, sg = [], []
            for li in (0, 1):
                if li == 0:
                    b(f"    %{p}cd{li} = scalar.andi %{p}qww, {e.cs(65535)} : i32")
                else:
                    b(f"    %{p}cd{li} = scalar.shrui %{p}qww, {e.cs(16)} : i32")
                b(f"    %{p}gi{li} = scalar.andi %{p}cd{li}, {e.cs(511)} : i32")
                b(f"    %{p}gix{li} = index.cast %{p}gi{li} : i32 to index")
                b(f"    %{p}gx2_{li} = index.mul %{p}gix{li}, {e.ci(2)} : index")
                w2 = ld("%t_grid_iq2xs", 1024, f"%{p}gx2_{li}", p, f"g{li}", 2, "i32")
                for h in (0, 1):
                    b(f"    %{p}g{li}_{h} = vector.extract {w2}[{h}] : vector<2xi32> -> i32")
                    words.append(f"%{p}g{li}_{h}")
                b(f"    %{p}ks{li}i = scalar.shrui %{p}cd{li}, {e.cs(9)} : i32")
                b(f"    %{p}ksx{li} = index.cast %{p}ks{li}i : i32 to index")
                sg.append(ld_u8("%t_ksigns", 128, f"%{p}ksx{li}", p, f"ks{li}"))
            b(f"    %{p}gw = vector.from_elements " + ", ".join(words) + " : vector<4xi32>")
            b(f"    %{p}g = vector.bitcast %{p}gw : vector<4xi32> to vector<16xi8>")
            if parts:
                q = (f"%{p}gw", sg[0], sg[1])
            else:
                neg = signs16(p, sg[0], sg[1])
                q = apply_signs(p, f"%{p}g", neg)
            b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
            sb_ = ld_u8(wv, nb, f"%{p}sa", p, "scb")
            b(f"    %{p}nw = scalar.shrui {sb_}, %{lp}sls : i32")
            b(f"    %{p}nib = scalar.andi %{p}nw, {e.cs(15)} : i32")
            b(f"    %{p}nf = scalar.uitofp %{p}nib : i32 to f32")
            b(f"    %{p}nh = scalar.addf %{p}nf, {e.cs(0.5, 'f32')} : f32")
            d = ld_d(hv, nh, boff, 0, p, "d")
            b(f"    %{p}dn = scalar.mulf {d}, %{p}nh : f32")
            b(f"    %{p}scale = scalar.mulf %{p}dn, {e.cs(0.25, 'f32')} : f32")
            return q, f"%{p}scale", None
        raise SystemExit("decode " + f)

    # ---- kernel body ----
    nw = len(fmts)
    wnb = [row_bytes(f, K) * M for f in fmts]
    b("  %tid = kernel.workitem.id<x> : index")
    b("  %wg = kernel.workgroup.id<x> : index")
    b(f"  %lane = index.rem %tid, {e.ci(32)} : index")
    b(f"  %wave = index.div %tid, {e.ci(32)} : index")
    b(f"  %wgw = index.mul %wg, {e.ci(W)} : index")
    b("  %wid = index.add %wgw, %wave : index")
    b(f"  %row0 = index.mul %wid, {e.ci(R)} : index")
    if G2:
        assert K % 1024 == 0 and WORD
        NT = K // 1024
    S = [(lane_setup2 if G2 else lane_setup)(f, f"f{i}") for i, f in enumerate(fmts)]
    # row byte bases: row (row0 + r) * row_bytes + lane block offset
    for i, f in enumerate(fmts):
        rb = row_bytes(f, K)
        for r in range(R):
            b(f"  %f{i}row{r} = index.add %row0, {e.ci(r)} : index")
            b(f"  %f{i}rb{r} = index.mul %f{i}row{r}, {e.ci(rb)} : index")
            b(f"  %f{i}base{r} = index.add %f{i}rb{r}, %f{i}lboff : index")
    b(f"  %xl16 = index.mul %lane, {e.ci(32 if G2 else 16)} : index")
    accs = [f"%a{i}_{r}" for i in range(nw) for r in range(R)]
    anyoff = WORD or any(f in OFFSET_FMTS for f in fmts)
    b("  %zf = scalar.constant 0.0 : f32")
    inits = ", ".join(f"{a} = %zf : f32" for a in accs)
    res = [f"%res{j}" for j in range(len(accs))]
    sched = (f" pipeline({e.ci(PIPE)})" if PIPE > 1 else "") + (f" unroll({e.ci(UNROLL)})" if UNROLL > 1 else "")
    b(f"  {', '.join(res)} = scf.for %t = [{e.ci(0)} to {e.ci(NT)} step {e.ci(1)}]({inits}) -> ({', '.join(['f32'] * len(accs))}){sched} {{")
    b(f"    %xo0 = index.mul %t, {e.ci(1024 if G2 else 512)} : index")
    b("    %xo = index.add %xo0, %xl16 : index")
    xv = ld("%xv", K, "%xo", "", "xa", 16, "f32")      # (not "xv": that would shadow the view)
    if anyoff:
        b(f"    %xsum = vector.reduce<addf> {xv}, %zf : vector<16xf32>, f32")
    nxt = []
    if G2:
        b(f"    %xo2 = index.add %xo, {e.ci(16)} : index")
        xv2 = ld("%xv", K, "%xo2", "", "xb", 16, "f32")
        b(f"    %xsum2 = vector.reduce<addf> {xv2}, %zf : vector<16xf32>, f32")
        for i, f in enumerate(fmts):
            for r in range(R):
                p = f"d{i}_{r}"
                b(f"    %{p}to = index.mul %t, {e.ci(S[i]['tstep'])} : index")
                b(f"    %{p}bo = index.add %f{i}base{r}, %{p}to : index")
                parts = decode2(f, S[i], p, f"%wv{i}", f"%wh{i}", wnb[i], f"%{p}bo", f"f{i}")
                acc = f"%a{i}_{r}"
                for h, ((q, scale, offs), xx, xs) in enumerate(zip(parts, (xv, xv2), ("%xsum", "%xsum2"))):
                    ph = f"{p}H{h}"
                    b(f"    %{ph}qf = vector.uitofp {q} : vector<16xi8> to vector<16xf32>")
                    b(f"    %{ph}pr = vector.mulf %{ph}qf, {xx} : vector<16xf32>")
                    b(f"    %{ph}dot = vector.reduce<addf> %{ph}pr, %zf : vector<16xf32>, f32")
                    b(f"    %{ph}sd = scalar.mulf {scale}, %{ph}dot : f32")
                    b(f"    %{ph}ox = scalar.mulf {offs}, {xs} : f32")
                    b(f"    %{ph}c = scalar.subf %{ph}sd, %{ph}ox : f32")
                    b(f"    %{ph}n = scalar.addf {acc}, %{ph}c : f32")
                    acc = f"%{ph}n"
                nxt.append(acc)
    for i, f in enumerate([] if G2 else fmts):
        for r in range(R):
            p = f"d{i}_{r}"
            b(f"    %{p}to = index.mul %t, {e.ci(S[i]['tstep'])} : index")
            b(f"    %{p}bo = index.add %f{i}base{r}, %{p}to : index")
            dec = decode_w if WORD else decode
            q, scale, offs = dec(f, S[i], p, f"%wv{i}", f"%wh{i}", wnb[i], f"%{p}bo", f"f{i}")
            cvt = "uitofp" if WORD else "sitofp"
            b(f"    %{p}qf = vector.{cvt} {q} : vector<16xi8> to vector<16xf32>")
            b(f"    %{p}pr = vector.mulf %{p}qf, {xv} : vector<16xf32>")
            b(f"    %{p}dot = vector.reduce<addf> %{p}pr, %zf : vector<16xf32>, f32")
            b(f"    %{p}sd = scalar.mulf {scale}, %{p}dot : f32")
            if offs:
                b(f"    %{p}ox = scalar.mulf {offs}, %xsum : f32")
                b(f"    %{p}c = scalar.subf %{p}sd, %{p}ox : f32")
                b(f"    %{p}n = scalar.addf %a{i}_{r}, %{p}c : f32")
            else:
                b(f"    %{p}n = scalar.addf %a{i}_{r}, %{p}sd : f32")
            nxt.append(f"%{p}n")
    b(f"    scf.yield {', '.join(nxt)} : {', '.join(['f32'] * len(accs))}")
    b("  }")
    # butterfly reduction (HIP __shfl_xor 16, 8, 4, 2, 1)
    tot = []
    for j in range(len(accs)):
        cur = f"%res{j}"
        for m in (16, 8, 4, 2, 1):
            tg = f"%rd{j}_{m}"
            b(f"  {tg}i = scalar.bitcast {cur} : f32 to i32")
            b(f"  {tg}x, {tg}v = kernel.subgroup.shuffle<xor> {tg}i, {e.cs(m)}, {e.cs(32)} : i32, i32, i32")
            b(f"  {tg}f = scalar.bitcast {tg}x : i32 to f32")
            b(f"  {tg}s = scalar.addf {cur}, {tg}f : f32")
            cur = f"{tg}s"
        tot.append(cur)
    b(f"  %is0 = index.cmp eq, %lane, {e.ci(0)} : index")
    b("  scf.if %is0 {")
    for r in range(R):
        b(f"    %orow{r} = index.add %row0, {e.ci(r)} : index")
        b(f"    %orc{r} = index.min %orow{r}, {e.ci(M - 1)} : index")
        if kind == "plain":
            v = tot[r]
        elif kind in ("resid", "resid_norm"):
            b(f"    %old{r} = view.load %yv[%orc{r}] : view<{M}xf32> -> f32")
            b(f"    %new{r} = scalar.addf %old{r}, {tot[r]} : f32")
            v = f"%new{r}"
        else:
            g, u = tot[r], tot[R + r]
            b(f"    %ng{r} = scalar.negf {g} : f32")
            b(f"    %eg{r} = scalar.expf<afn> %ng{r} : f32")
            b(f"    %dg{r} = scalar.addf %eg{r}, {e.cs(1.0, 'f32')} : f32")
            b(f"    %sg{r} = scalar.divf {e.cs(1.0, 'f32')}, %dg{r} : f32")
            b(f"    %gs{r} = scalar.mulf {g}, %sg{r} : f32")
            b(f"    %sw{r} = scalar.mulf %gs{r}, {u} : f32")
            v = f"%sw{r}"
        b(f"    view.store {v}, %yv[%orc{r}] : f32, view<{M}xf32>")
    b("  }")
    if _parts is not None:               # gen_bands: hand back the body, constants stay in e
        _parts.update(body=body, wnb=wnb, tabs=tabs)
        return None
    if kind == "resid_norm":
        # The last workgroup to finish (device-scope counter, acq_rel) computes the next
        # RMSNorm of the updated vector: nout = (y * rsqrt(mean(y^2) + eps)) * nw, and
        # resets the counter. Every thread does its own device-scope acquire before reading
        # y (a workgroup may span two CUs with separate L0 caches).
        NT = 32 * W
        assert M % NT == 0
        per = M // NT
        nwg_ = M // rows_wg
        b("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        b(f"  %rn_t0 = index.cmp eq, %tid, {e.ci(0)} : index")
        b("  scf.if %rn_t0 {")
        b(f"    %rn_old = view.atomic.rmw<addi> {e.cs(1)}, %cntv[{e.ci(0)}] {{ordering = acq_rel, scope = device}} : i32, view<1xi32> -> i32")
        b(f"    %rn_last = scalar.cmpi eq, %rn_old, {e.cs(nwg_ - 1)} : i32")
        b("    %rn_fl = scf.if %rn_last -> (i32) {")
        b(f"      scf.yield {e.cs(1)} : i32")
        b("    } else {")
        b(f"      scf.yield {e.cs(0)} : i32")
        b("    }")
        b(f"    view.store %rn_fl, %rnflag[{e.ci(0)}] : i32, view<4xi32>")
        b("  }")
        b("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        b(f"  %rn_flv = view.load %rnflag[{e.ci(0)}] : view<4xi32> -> i32")
        b(f"  %rn_is = scalar.cmpi eq, %rn_flv, {e.cs(1)} : i32")
        b("  scf.if %rn_is {")
        b(f"    %rn_acq = view.atomic.load %cntv[{e.ci(0)}] {{ordering = acquire, scope = device}} : view<1xi32> -> i32")
        acc = "%zf"
        for k in range(per):
            b(f"    %rn_i{k} = index.add %tid, {e.ci(k * NT)} : index")
            b(f"    %rn_x{k} = view.load %yv[%rn_i{k}] : view<{M}xf32> -> f32")
            b(f"    %rn_s{k} = scalar.fmaf %rn_x{k}, %rn_x{k}, {acc} : f32")
            acc = f"%rn_s{k}"
        b(f"    %rn_ws = kernel.subgroup.reduce<addf> {acc} : f32")
        b(f"    %rn_l0 = index.cmp eq, %lane, {e.ci(0)} : index")
        b("    scf.if %rn_l0 {")
        b(f"      view.store %rn_ws, %rnpart[%wave] : f32, view<{W}xf32>")
        b("    }")
        b("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        tot_ = "%zf"
        for w in range(W):
            b(f"    %rn_p{w} = view.load %rnpart[{e.ci(w)}] : view<{W}xf32> -> f32")
            b(f"    %rn_a{w} = scalar.addf {tot_}, %rn_p{w} : f32")
            tot_ = f"%rn_a{w}"
        b(f"    %rn_mean = scalar.mulf {tot_}, {e.cs(1.0 / M, 'f32')} : f32")
        b(f"    %rn_sh = scalar.addf %rn_mean, {e.cs(1e-6, 'f32')} : f32")
        b("    %rn_inv = scalar.rsqrtf %rn_sh : f32")
        for k in range(per):
            b(f"    %rn_w{k} = view.load %nwv[%rn_i{k}] : view<{M}xf32> -> f32")
            b(f"    %rn_n{k} = scalar.mulf %rn_x{k}, %rn_inv : f32")
            b(f"    %rn_y{k} = scalar.mulf %rn_n{k}, %rn_w{k} : f32")
            b(f"    view.store %rn_y{k}, %noutv[%rn_i{k}] : f32, view<{M}xf32>")
        b("    scf.if %rn_t0 {")
        b(f"      view.store {e.cs(0)}, %cntv[{e.ci(0)}] : i32, view<1xi32>")
        b("    }")
        b("  }")
    b("  kernel.return")

    # ---- assemble ----
    out = []
    o = out.append
    o(f"// GENERATED by tools/gen_gemv.py ({kind} {'/'.join(fmts)} M={M} K={K} R={R} W={W}) -- edit the generator.")
    o("amdgpu.target<gfx1151> @gv32 {subgroup_size = 32}")
    o("")
    o(f"kernel.def target(@gv32) @{name}() {{")
    o("  %u1 = index.constant 1 : index")
    o(f"  %nwg = index.constant {min(PERSIST, M // rows_wg) if PERSIST else M // rows_wg} : index")
    o(f"  %wgs = index.constant {32 * W} : index")
    o("  kernel.launch.config workgroups(%nwg, %u1, %u1) workgroup_size(%wgs, %u1, %u1) : index")
    params = [f"%w{i}: buffer" for i in range(nw)] + [f"%{t}: buffer" for t in tabs] + ["%x: buffer", "%y: buffer"]
    if kind == "resid_norm":
        params += ["%nw: buffer", "%nout: buffer", "%cnt: buffer"]
    o(f"}} launch({', '.join(params)}) {{")
    o("  %base = index.constant 0 : offset")
    names = [p.split(":")[0] for p in params]
    na = [n + "_na" for n in names]
    o(f"  {', '.join(na)} = buffer.assume.noalias {', '.join(names)} : {', '.join(['buffer'] * len(names))}")
    for i in range(nw):
        o(f"  %wv{i} = buffer.view %w{i}_na[%base] : buffer -> view<{wnb[i]}xi8>")
        o(f"  %wh{i} = buffer.view %w{i}_na[%base] : buffer -> view<{wnb[i] // 2}xf16>")
    for t in tabs:
        ty, n = TABLES[t]
        if not GRID_LDS:
            o(f"  %t_{t} = buffer.view %{t}_na[%base] : buffer -> view<{n}x{ty}>")
            continue
        eb = 4 if ty == "i32" else 1
        o(f"  %g_{t} = buffer.view %{t}_na[%base] : buffer -> view<{n}x{ty}>")
        o(f"  %lb_{t} = index.constant {n * eb} : offset")
        o(f"  %lp_{t} = buffer.alloca<workgroup> align(16) %lb_{t} : buffer")
        o(f"  %t_{t} = buffer.view %lp_{t}[%base] : buffer -> view<{n}x{ty}>")
        o(f"  %ctid_{t} = kernel.workitem.id<x> : index")
        for k in range((n + 32 * W - 1) // (32 * W)):
            o(f"  %cpo_{t}{k} = index.constant {k * 32 * W} : index")
            o(f"  %cpi_{t}{k}a = index.add %ctid_{t}, %cpo_{t}{k} : index")
            o(f"  %cpm_{t}{k} = index.constant {n - 1} : index")
            o(f"  %cpi_{t}{k} = index.min %cpi_{t}{k}a, %cpm_{t}{k} : index")
            o(f"  %cpv_{t}{k} = view.load %g_{t}[%cpi_{t}{k}] : view<{n}x{ty}> -> {ty}")
            o(f"  view.store %cpv_{t}{k}, %t_{t}[%cpi_{t}{k}] : {ty}, view<{n}x{ty}>")
    if GRID_LDS and tabs:
        o("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    o(f"  %xv = buffer.view %x_na[%base] : buffer -> view<{K}xf32>")
    o(f"  %yv = buffer.view %y_na[%base] : buffer -> view<{M}xf32>")
    if kind == "resid_norm":
        o(f"  %nwv = buffer.view %nw_na[%base] : buffer -> view<{M}xf32>")
        o(f"  %noutv = buffer.view %nout_na[%base] : buffer -> view<{M}xf32>")
        o("  %cnt_g = buffer.assume.memory_space<global> %cnt_na : buffer")
        o("  %cntv = buffer.view %cnt_g[%base] : buffer -> view<1xi32>")
        o("  %rnfb = index.constant 16 : offset")
        o("  %rnfp = buffer.alloca<workgroup> align(16) %rnfb : buffer")
        o("  %rnflag = buffer.view %rnfp[%base] : buffer -> view<4xi32>")
        o(f"  %rnpb = index.constant {max(16, 4 * W)} : offset")
        o("  %rnpp = buffer.alloca<workgroup> align(16) %rnpb : buffer")
        o(f"  %rnpart = buffer.view %rnpp[%base] : buffer -> view<{W}xf32>")
    if "iq4xs" in fmts:
        o("\n".join(f"  %kv{i} = scalar.constant {v} : i8" for i, v in enumerate(IQ4_KVALUES)))
        o("  %kvt = vector.from_elements " + ", ".join(f"%kv{i}" for i in range(16)) + " : vector<16xi8>")
        o("\n".join(f"  %kvu{i} = scalar.constant {v + 128 - 256 if v + 128 > 127 else v + 128} : i8" for i, v in enumerate(IQ4_KVALUES)))
        o("  %kvtu = vector.from_elements " + ", ".join(f"%kvu{i}" for i in range(16)) + " : vector<16xi8>")
    for k in sorted(e.consts, key=lambda n: (not n.startswith("%c"), n)):
        if not k.startswith("%sv") and k != "%sgnsh" and not k.startswith("%ws") and not k.startswith("%wt") and k != "%smask_used":
            o(e.consts[k])
    for k in sorted(e.consts):
        if k.startswith("%sv") or k == "%sgnsh" or k.startswith("%ws") or k.startswith("%wt"):
            o(e.consts[k])
    if "%smask_used" in e.consts:      # 16-entry sign-mask table in LDS, filled by lanes 0..15
        o("  %smb = index.constant 64 : offset")
        o("  %smp = buffer.alloca<workgroup> align(16) %smb : buffer")
        o("  %smask = buffer.view %smp[%base] : buffer -> view<16xi32>")
        o("  %smt = kernel.workitem.id<x> : index")
        o("  %smt16 = index.cmp ult, %smt, %ci16 : index")
        o("  scf.if %smt16 {")
        o("    %smti = index.cast %smt : index to i32")
        # mask(i) = sum over set bits j of 0xFF << 8 j, computed arithmetically once per lane
        o("    %smv0 = scalar.constant 0 : i32")
        cur = "%smv0"
        for j in range(4):
            o(f"    %smb{j} = scalar.shrui %smti, %cw{j} : i32")
            o(f"    %smc{j} = scalar.andi %smb{j}, %cw1 : i32")
            o(f"    %smd{j} = scalar.muli %smc{j}, %cw{str(_s32(255 << (8 * j))).replace('-', 'm')} : i32")
            o(f"    %sme{j} = scalar.ori {cur}, %smd{j} : i32")
            cur = f"%sme{j}"
        o(f"    view.store {cur}, %smask[%smt] : i32, view<16xi32>")
        o("  }")
        o("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    if PERSIST and kind != "resid_norm":
        G_ = min(PERSIST, M // rows_wg)
        assert body[-1].strip() == "kernel.return"
        inner = [l.replace("%wg = kernel.workgroup.id<x> : index", "%wg = index.add %rg, %pz : index") for l in body[:-1]]
        o("  %wgid = kernel.workgroup.id<x> : index")
        o("  %pz = index.constant 0 : index")
        o(f"  %pG = index.constant {G_} : index")
        o(f"  %pN = index.constant {M // rows_wg} : index")
        o("  %prs = scf.for %rg = [%wgid to %pN step %pG](%pk = %pz : index) -> (index) {")
        out += inner
        o("    scf.yield %pk : index")
        o("  }")
        o("  kernel.return")
    else:
        out += body
    o("}")
    return "\n".join(out) + "\n"


_SHARED = re.compile(r"%(c[iwbf]\w*|ws\w*|wt\w*|sv\w*|sgnsh|kvtu?|t_\w+|xv|smask)$")


def gen_bands(fmts, Ms, K, R=2, W=4, name="yah_gemv"):
    """Several plain GEMVs over one input in one dispatch: band b is fmts[b] with Ms[b]
    rows, written to its own output; workgroup ranges map to bands. Bindings: the
    weights, the IQ tables of all bands (TABLE_ORDER), x, then the outputs."""
    rows_wg = R * W
    e = E()
    bodies, wnbs, starts = [], [], []
    start = 0
    for bi, (f, M) in enumerate(zip(fmts, Ms)):
        assert M % rows_wg == 0, (f, M)
        parts = {}
        gen("plain", [f], M, K, R, W, _e=e, _parts=parts)
        pre = f"b{bi}_"
        def ren(m):
            nm = m.group(0)
            return nm if _SHARED.match(nm) else "%" + pre + nm[1:]
        lines = [re.sub(r"%[A-Za-z_]\w*", ren, l) for l in parts["body"]]
        wgl = f"%{pre}wg = kernel.workgroup.id<x> : index"
        assert sum(1 for l in lines if l.strip() == wgl) == 1
        lines = [f"  %{pre}wg = index.sub %wgg, {e.ci(start)} : index" if l.strip() == wgl else l for l in lines]
        bodies.append(lines)
        wnbs.append(parts["wnb"][0])
        starts.append(start)
        start += M // rows_wg
    nwg = start
    tabs = tables_for(fmts)
    out = []
    o = out.append
    o(f"// GENERATED by tools/gen_gemv.py (bands {'/'.join(fmts)} M={'/'.join(map(str, Ms))} K={K} R={R} W={W}) -- edit the generator.")
    o("amdgpu.target<gfx1151> @gv32 {subgroup_size = 32}")
    o("")
    o(f"kernel.def target(@gv32) @{name}() {{")
    o("  %u1 = index.constant 1 : index")
    o(f"  %nwg = index.constant {nwg} : index")
    o(f"  %wgs = index.constant {32 * W} : index")
    o("  kernel.launch.config workgroups(%nwg, %u1, %u1) workgroup_size(%wgs, %u1, %u1) : index")
    params = [f"%w{i}: buffer" for i in range(len(fmts))] + [f"%{t}: buffer" for t in tabs] + ["%x: buffer"] + \
             [f"%y{i}: buffer" for i in range(len(fmts))]
    o(f"}} launch({', '.join(params)}) {{")
    o("  %base = index.constant 0 : offset")
    names = [p.split(":")[0] for p in params]
    o(f"  {', '.join(n + '_na' for n in names)} = buffer.assume.noalias {', '.join(names)} : {', '.join(['buffer'] * len(names))}")
    for bi in range(len(fmts)):
        o(f"  %b{bi}_wv0 = buffer.view %w{bi}_na[%base] : buffer -> view<{wnbs[bi]}xi8>")
        o(f"  %b{bi}_wh0 = buffer.view %w{bi}_na[%base] : buffer -> view<{wnbs[bi] // 2}xf16>")
        o(f"  %b{bi}_yv = buffer.view %y{bi}_na[%base] : buffer -> view<{Ms[bi]}xf32>")
    for t in tabs:
        ty, n = TABLES[t]
        o(f"  %t_{t} = buffer.view %{t}_na[%base] : buffer -> view<{n}x{ty}>")
    o(f"  %xv = buffer.view %x_na[%base] : buffer -> view<{K}xf32>")
    if "iq4xs" in fmts:
        o("\n".join(f"  %kv{i} = scalar.constant {v} : i8" for i, v in enumerate(IQ4_KVALUES)))
        o("  %kvt = vector.from_elements " + ", ".join(f"%kv{i}" for i in range(16)) + " : vector<16xi8>")
        o("\n".join(f"  %kvu{i} = scalar.constant {v + 128 - 256 if v + 128 > 127 else v + 128} : i8" for i, v in enumerate(IQ4_KVALUES)))
        o("  %kvtu = vector.from_elements " + ", ".join(f"%kvu{i}" for i in range(16)) + " : vector<16xi8>")
    for k in sorted(e.consts, key=lambda n: (not n.startswith("%c"), n)):
        if not k.startswith("%sv") and k != "%sgnsh" and not k.startswith("%ws") and not k.startswith("%wt") and k != "%smask_used":
            o(e.consts[k])
    for k in sorted(e.consts):
        if k.startswith("%sv") or k == "%sgnsh" or k.startswith("%ws") or k.startswith("%wt"):
            o(e.consts[k])
    if "%smask_used" in e.consts:      # 16-entry sign-mask table in LDS, filled by lanes 0..15
        o("  %smb = index.constant 64 : offset")
        o("  %smp = buffer.alloca<workgroup> align(16) %smb : buffer")
        o("  %smask = buffer.view %smp[%base] : buffer -> view<16xi32>")
        o("  %smt = kernel.workitem.id<x> : index")
        o("  %smt16 = index.cmp ult, %smt, %ci16 : index")
        o("  scf.if %smt16 {")
        o("    %smti = index.cast %smt : index to i32")
        # mask(i) = sum over set bits j of 0xFF << 8 j, computed arithmetically once per lane
        o("    %smv0 = scalar.constant 0 : i32")
        cur = "%smv0"
        for j in range(4):
            o(f"    %smb{j} = scalar.shrui %smti, %cw{j} : i32")
            o(f"    %smc{j} = scalar.andi %smb{j}, %cw1 : i32")
            o(f"    %smd{j} = scalar.muli %smc{j}, %cw{str(_s32(255 << (8 * j))).replace('-', 'm')} : i32")
            o(f"    %sme{j} = scalar.ori {cur}, %smd{j} : i32")
            cur = f"%sme{j}"
        o(f"    view.store {cur}, %smask[%smt] : i32, view<16xi32>")
        o("  }")
        o("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    o("  %wgg = kernel.workgroup.id<x> : index")
    # if wgg < end_0 { band 0 } else { if wgg < end_1 { band 1 } else { ... } }
    depth = 0
    for bi in range(len(fmts)):
        last = bi == len(fmts) - 1
        if not last:
            end = starts[bi + 1]
            o(f"  %inb{bi} = index.cmp ult, %wgg, {e.ci(end)} : index")
            o(f"  scf.if %inb{bi} {{")
            out += bodies[bi]
            o("  } else {")
            depth += 1
        else:
            out += bodies[bi]
    for _ in range(depth):
        o("  }")
    o("  kernel.return")
    o("}")
    # constants referenced after the chain was opened (e.ci(end) above) are already in e.consts:
    text = "\n".join(out) + "\n"
    missing = [c for c in e.consts if c not in text.split("%wgg = kernel.workgroup.id<x> : index")[0]]
    if missing:   # re-emit with the late constants
        head, tail = text.split("  %wgg = kernel.workgroup.id<x> : index\n")
        text = head + "".join(e.consts[c] + "\n" for c in missing) + "  %wgg = kernel.workgroup.id<x> : index\n" + tail
    return text


def footprint_bands(fmts, Ms, K):
    sizes = [row_bytes(f, K) * M for f, M in zip(fmts, Ms)]
    for t in tables_for(fmts):
        ty, n = TABLES[t]
        sizes.append(n * (4 if ty == "i32" else 1))
    return sizes + [K * 4] + [M * 4 for M in Ms]


def footprint(kind, fmts, M, K):
    """Exact binding sizes in bytes, in binding order (weights, tables, x, y[, nw, nout, cnt])."""
    sizes = [row_bytes(f, K) * M for f in fmts]
    for t in tables_for(fmts):
        ty, n = TABLES[t]
        sizes.append(n * (4 if ty == "i32" else 1))
    return sizes + [K * 4, M * 4] + ([M * 4, M * 4, 4] if kind == "resid_norm" else [])


if __name__ == "__main__":
    # gen_gemv.py <kind> <fmt[,fmt]> <M> <K> <out.loom> [R] [W]
    kind, fmts, M, K, outp = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    R = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    W = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    open(outp, "w").write(gen(kind, fmts, M, K, R, W))
    print(outp)
