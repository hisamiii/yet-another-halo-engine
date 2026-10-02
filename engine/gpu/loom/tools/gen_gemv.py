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


def gen(kind, fmts, M, K, R=2, W=4, name="yah_gemv"):
    assert kind in ("plain", "resid", "swiglu")
    assert len(fmts) == (2 if kind == "swiglu" else 1)
    assert K % 512 == 0, "K / 16 sub-blocks must split evenly over 32 lanes"
    rows_wg = R * W
    assert M % rows_wg == 0, (M, rows_wg)
    NT = K // 512                    # sub-blocks per lane
    tabs = tables_for(fmts)
    e = E()
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

    def decode(f, S, p, wv, hv, nb, boff, lp):
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
    S = [lane_setup(f, f"f{i}") for i, f in enumerate(fmts)]
    # row byte bases: row (row0 + r) * row_bytes + lane block offset
    for i, f in enumerate(fmts):
        rb = row_bytes(f, K)
        for r in range(R):
            b(f"  %f{i}row{r} = index.add %row0, {e.ci(r)} : index")
            b(f"  %f{i}rb{r} = index.mul %f{i}row{r}, {e.ci(rb)} : index")
            b(f"  %f{i}base{r} = index.add %f{i}rb{r}, %f{i}lboff : index")
    b(f"  %xl16 = index.mul %lane, {e.ci(16)} : index")
    accs = [f"%a{i}_{r}" for i in range(nw) for r in range(R)]
    anyoff = any(f in OFFSET_FMTS for f in fmts)
    b("  %zf = scalar.constant 0.0 : f32")
    inits = ", ".join(f"{a} = %zf : f32" for a in accs)
    res = [f"%res{j}" for j in range(len(accs))]
    b(f"  {', '.join(res)} = scf.for %t = [{e.ci(0)} to {e.ci(NT)} step {e.ci(1)}]({inits}) -> ({', '.join(['f32'] * len(accs))}) {{")
    b(f"    %xo0 = index.mul %t, {e.ci(512)} : index")
    b("    %xo = index.add %xo0, %xl16 : index")
    xv = ld("%xv", K, "%xo", "", "xv", 16, "f32")
    if anyoff:
        b(f"    %xsum = vector.reduce<addf> {xv}, %zf : vector<16xf32>, f32")
    nxt = []
    for i, f in enumerate(fmts):
        for r in range(R):
            p = f"d{i}_{r}"
            b(f"    %{p}to = index.mul %t, {e.ci(S[i]['tstep'])} : index")
            b(f"    %{p}bo = index.add %f{i}base{r}, %{p}to : index")
            q, scale, offs = decode(f, S[i], p, f"%wv{i}", f"%wh{i}", wnb[i], f"%{p}bo", f"f{i}")
            b(f"    %{p}qf = vector.sitofp {q} : vector<16xi8> to vector<16xf32>")
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
        elif kind == "resid":
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
    b("  kernel.return")

    # ---- assemble ----
    out = []
    o = out.append
    o(f"// GENERATED by tools/gen_gemv.py ({kind} {'/'.join(fmts)} M={M} K={K} R={R} W={W}) -- edit the generator.")
    o("amdgpu.target<gfx1151> @gv32 {subgroup_size = 32}")
    o("")
    o(f"kernel.def target(@gv32) @{name}() {{")
    o("  %u1 = index.constant 1 : index")
    o(f"  %nwg = index.constant {M // rows_wg} : index")
    o(f"  %wgs = index.constant {32 * W} : index")
    o("  kernel.launch.config workgroups(%nwg, %u1, %u1) workgroup_size(%wgs, %u1, %u1) : index")
    params = [f"%w{i}: buffer" for i in range(nw)] + [f"%{t}: buffer" for t in tabs] + ["%x: buffer", "%y: buffer"]
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
        o(f"  %t_{t} = buffer.view %{t}_na[%base] : buffer -> view<{n}x{ty}>")
    o(f"  %xv = buffer.view %x_na[%base] : buffer -> view<{K}xf32>")
    o(f"  %yv = buffer.view %y_na[%base] : buffer -> view<{M}xf32>")
    if "iq4xs" in fmts:
        o("\n".join(f"  %kv{i} = scalar.constant {v} : i8" for i, v in enumerate(IQ4_KVALUES)))
        o("  %kvt = vector.from_elements " + ", ".join(f"%kv{i}" for i in range(16)) + " : vector<16xi8>")
    for k in sorted(e.consts, key=lambda n: (not n.startswith("%c"), n)):
        if not k.startswith("%sv") and k != "%sgnsh":
            o(e.consts[k])
    for k in sorted(e.consts):
        if k.startswith("%sv") or k == "%sgnsh":
            o(e.consts[k])
    out += body
    o("}")
    return "\n".join(out) + "\n"


def footprint(kind, fmts, M, K):
    """Exact binding sizes in bytes, in binding order (weights, tables, x, y)."""
    sizes = [row_bytes(f, K) * M for f in fmts]
    for t in tables_for(fmts):
        ty, n = TABLES[t]
        sizes.append(n * (4 if ty == "i32" else 1))
    return sizes + [K * 4, M * 4]


if __name__ == "__main__":
    # gen_gemv.py <kind> <fmt[,fmt]> <M> <K> <out.loom> [R] [W]
    kind, fmts, M, K, outp = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    R = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    W = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    open(outp, "w").write(gen(kind, fmts, M, K, R, W))
    print(outp)
