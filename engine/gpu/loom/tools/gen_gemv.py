#!/usr/bin/env python3
"""Single-token decode GEMV kernels, one per (kind, weight formats, M, K).

    y[m] = sum_k W[m, k] * x[k]          W packed in a GGUF quant format, x f32

A weight row is cut into 16-element sub-blocks; lane l of a wave owns sub-blocks l, l + 32, ...
Each sub-block decodes to 16 small integers q, a scale and an offset:

    w[j] = scale * q[j] - offset          partial += scale * dot(q, x) - offset * sum(x)

A butterfly reduction over the 32 lanes follows.
Format and shape are compile-time (one kernel per tensor shape). One wave carries R rows, so every x load serves R rows.
The per-lane decode constants (sub-block of the 256-element block, shifts, scale indices) are loop invariant because K / 16 is a multiple of 32.

Kinds:
  plain   y[m] = W x
  resid   y[m] += W x                    (residual add fused: y is read and written)
  swiglu  y[m] = silu(G x) * (U x)       G, U may have different formats; order (g * (1 / (1 + exp(-g)))) * u
Usage (module): gen(kind, fmts, M, K, R=2, W=4) -> Loom source text.
Bindings: weights (one per format in fmts), then the IQ tables the formats need (TABLE_ORDER), then x, then y.
"""
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
    "iq2s": (256, 82, ("grid_iq2s",)),
    "iq4nl": (32, 18, ()),
}
# table: (element type, elements); files in the HAL dir (emit_prefill writes them)
TABLES = {"grid_iq3s": ("i32", 512), "grid_iq3xxs": ("i32", 256), "grid_iq2xxs": ("i32", 512),
          "grid_iq2xs": ("i32", 1024), "ksigns": ("i8", 128), "grid_iq2s": ("i32", 2048)}
TABLE_ORDER = ["grid_iq3s", "grid_iq3xxs", "grid_iq2xxs", "grid_iq2xs", "ksigns", "grid_iq2s"]
IQ4_KVALUES = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
# Loom read-ahead depth of the sub-block loop: 2 is the fastest on the full decode (none and 3 are slower).
PIPE = 2
GGML = {8: "q8_0", 10: "q2k", 11: "q3k", 12: "q4k", 13: "q5k", 14: "q6k", 16: "iq2xxs", 17: "iq2xs",
        18: "iq3xxs", 20: "iq4nl", 21: "iq3s", 22: "iq2s", 23: "iq4xs"}


def rows_per_wg(R=2, W=4):
    """Output rows per workgroup of a gen() / gen_bands() kernel: launch exactly M / rows_per_wg workgroups.

    Loom takes workgroup.id < grid from the launch config and may drop the row clamps: a larger grid reads out of bounds.
    """
    return R * W


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
        t = {"i32": "w", "i8": "b", "f32": "f", "i64": "q"}[ty]
        nm = f"%c{t}{str(v).replace('-', 'm').replace('.', 'p')}"
        self.consts[nm] = f"  {nm} = scalar.constant {v} : {ty}"
        return nm

    def splat8(self, v):        # vector<16xi8> splat of an i8 constant
        nm = f"%sv{str(v).replace('-', 'm')}"
        s = self.cs(v, "i8")
        self.consts[nm] = f"  {nm} = vector.splat {s} : vector<16xi8>"
        return nm


def gen(kind, fmts, M, K, R=2, W=4, name="yah_gemv", _e=None, _parts=None, peel=0, amax=False, kseg=None, rseg=0):
    """kseg (K-segmented weights, kseg_layout): [(first block, blocks)] covering K's 256-wide blocks, the weight rows
    stored segment by segment ([segment][row][its blocks]); the same loop and summation order (bit-identical results).
    rseg (swiglu, F): row-segmented gate / up (RSEG), both weight bindings the gate tensor's start over gate and up.
    amax (plain): also the argmax of y into a token (bindings ak: u64 key + i32 counter, tok: i32 after y; amax_tail).
    peel: the first peel K iterations run before the loop, their weight loads and decode (which do not read x) ahead
    of a "// chain.wait" marker (gen_chain puts the producer wait there), the x part after it (the same ops)."""
    assert kind in ("plain", "resid", "swiglu")
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
        if qk == 32:     # q8_0, iq4nl: block = sb / 2 = lane / 2 + 16 t; half = lane % 2
            b(f"  %{p}lb = index.div %lane, {e.ci(2)} : index")
            b(f"  %{p}lboff = index.mul %{p}lb, {e.ci(bb)} : index")
            b(f"  %{p}hlf = index.rem %lane, {e.ci(2)} : index")
            if f == "iq4nl":   # qs[0..16) at 2: half 0 the low nibbles, half 1 the high ones
                b(f"  %{p}qo = index.add {e.ci(0)}, {e.ci(2)} : index")
                b(f"  %{p}hlfi = index.cast %{p}hlf : index to i32")
                b(f"  %{p}shi = scalar.shli %{p}hlfi, {e.cs(2)} : i32")
                b(f"  %{p}shvw = vector.splat %{p}shi : vector<4xi32>")
                S["sh"] = f"%{p}shvw"
            else:
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

        def shsplat(name, expr_i32):
            """splat an i32 shift amount (0..7) to vector<4xi32>"""
            b(f"  %{p}{name}w = vector.splat {expr_i32} : vector<4xi32>")
            return f"%{p}{name}w"

        if f in ("q4k", "q5k"):
            # q bytes: base = (sb32 / 2) * 32 + hlf * 16 (+16 or +48), shift = 4 * (sb32 & 1)
            b(f"  %{p}s32h = index.div %{p}s32, {e.ci(2)} : index")
            b(f"  %{p}qb0 = index.mul %{p}s32h, {e.ci(32)} : index")
            b(f"  %{p}qb1 = index.add %{p}qb0, %{p}l16 : index")
            b(f"  %{p}qo = index.add %{p}qb1, {e.ci(48 if f == 'q5k' else 16)} : index")
            b(f"  %{p}sha = scalar.andi %{p}s32i, {e.cs(1)} : i32")
            b(f"  %{p}shi = scalar.shli %{p}sha, {e.cs(2)} : i32")
            S["sh"] = shsplat("shv", f"%{p}shi")
            if f == "q5k":
                b(f"  %{p}ho = index.add %{p}l16, {e.ci(16)} : index")
                S["hsh"] = shsplat("hshv", f"%{p}s32i")
            # scales: base = sb32 & 3 at +4, +8, +12; upper = sb32 >> 2
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
            S["sh"] = shsplat("shv", f"%{p}qshi")
            b(f"  %{p}hshi = scalar.shli %{p}segi, {e.cs(1)} : i32")          # 2 * seg
            S["hsh"] = shsplat("hshv", f"%{p}hshi")
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
            S["sh"] = shsplat("shv", f"%{p}shi")
            b(f"  %{p}hs0 = scalar.shli %{p}halfi, {e.cs(2)} : i32")
            b(f"  %{p}hsi = scalar.addi %{p}hs0, %{p}spi : i32")
            S["hsh"] = shsplat("hshv", f"%{p}hsi")
            b(f"  %{p}qb0 = index.mul %{p}half, {e.ci(32)} : index")
            b(f"  %{p}qb1 = index.add %{p}qb0, %{p}l16 : index")
            b(f"  %{p}qo = index.add %{p}qb1, {e.ci(32)} : index")
            b(f"  %{p}ho = index.add %{p}l16, {e.ci(0)} : index")
            # scale index si = half * 8 + sp * 2 + hlf.
            # low nibble: byte s[si & 7] >> 4 * (si >> 3); high pair: (s[8 + si % 4] >> 2 * (si / 4)) & 3
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
            S["sh"] = shsplat("shv", f"%{p}shi")
            b(f"  %{p}sd8 = index.div %{p}sub, {e.ci(8)} : index")
            b(f"  %{p}qb0 = index.mul %{p}sd8, {e.ci(32)} : index")
            b(f"  %{p}qb1 = index.add %{p}qb0, %{p}l16 : index")
            b(f"  %{p}qo = index.add %{p}qb1, {e.ci(16)} : index")
            b(f"  %{p}so = index.add %{p}sub, {e.ci(0)} : index")
        elif f == "iq4xs":
            b(f"  %{p}qb0 = index.mul %{p}s32, {e.ci(16)} : index")
            b(f"  %{p}qo = index.add %{p}qb0, {e.ci(8)} : index")
            b(f"  %{p}shi = scalar.shli %{p}hlfi, {e.cs(2)} : i32")
            S["sh"] = shsplat("shv", f"%{p}shi")
            b(f"  %{p}sl0 = index.div %{p}s32, {e.ci(2)} : index")
            b(f"  %{p}so = index.add %{p}sl0, {e.ci(4)} : index")
            b(f"  %{p}sl1 = scalar.andi %{p}s32i, {e.cs(1)} : i32")
            b(f"  %{p}sls = scalar.shli %{p}sl1, {e.cs(2)} : i32")              # 4 * (sb32 % 2)
            b(f"  %{p}shs = scalar.shli %{p}s32i, {e.cs(1)} : i32")              # 2 * sb32
        elif f in ("iq3s", "iq3xxs", "iq2xxs", "iq2xs", "iq2s"):
            # sign of element j: bit (j % 8) of sign byte l = 2 * hlf + j / 8
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
            elif f == "iq2s":
                # qs[4 ib32 + l] at 2, signs[4 ib32 + l] at 34, qh[ib32] at 66, scales[ib32] at 74
                b(f"  %{p}qb0 = index.mul %{p}s32, {e.ci(4)} : index")
                b(f"  %{p}qb1 = index.add %{p}qb0, %{p}l0 : index")
                b(f"  %{p}qo = index.add %{p}qb1, {e.ci(2)} : index")
                b(f"  %{p}sgo = index.add %{p}qb1, {e.ci(34)} : index")
                b(f"  %{p}ho = index.add %{p}s32, {e.ci(66)} : index")
                b(f"  %{p}so = index.add %{p}s32, {e.ci(74)} : index")
                b(f"  %{p}sls = scalar.shli %{p}hlfi, {e.cs(2)} : i32")
                # grid index l: qs[l] | (qh << (8 - 2l)) & 0x300
                b(f"  %{p}l0i = index.cast %{p}l0 : index to i32")
                for qq in (0, 1):
                    b(f"  %{p}l2x{qq} = scalar.addi %{p}l0i, {e.cs(qq)} : i32")
                    b(f"  %{p}l2y{qq} = scalar.shli %{p}l2x{qq}, {e.cs(1)} : i32")
                    b(f"  %{p}qsA{qq} = scalar.subi {e.cs(8)}, %{p}l2y{qq} : i32")
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

    def spread(p, nm, nib):
        """i32 nibble (4 sign bits) -> byte mask word (0x00 / 0xFF per byte)"""
        b(f"    %{p}{nm}a = scalar.muli {nib}, {e.cs(2113665)} : i32")          # 0x00204081
        b(f"    %{p}{nm}b = scalar.andi %{p}{nm}a, {e.cs(16843009)} : i32")      # 0x01010101
        b(f"    %{p}{nm} = scalar.muli %{p}{nm}b, {e.cs(255)} : i32")
        return f"%{p}{nm}"

    def signed_grid(p, gw, s0, s1):
        """Grid words gw (vector<4xi32>, magnitudes <= 62) + sign bytes s0 (words 0, 1), s1 (words 2, 3) -> 64 + sign * g per byte.

        No borrow crosses bytes because 64 - g >= 2.
        """
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


    def decode_w(f, S, p, wv, hv, nb, boff, lp):
        """Word-level decode -> (unsigned byte codes vector<16xi8>, scale, total offset), w = scale * code - offset.

        Code biases are folded into the offset.
        """
        nh = nb // 2
        bias = 0
        if f == "q8_0":
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            raw = tow(p, ld(wv, nb, f"%{p}qa", p, "q", 16), "qw")
            q = tob(p, wop(p, "qx", "xori", raw, wspl(-2139062144)), "qu")      # ^ 0x80808080: + 128
            scale = ld_d(hv, nh, boff, 0, p, "d")
            bias = 128
        elif f in ("q4k", "q5k", "q2k"):
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            raw = tow(p, ld(wv, nb, f"%{p}qa", p, "qr", 16), "qw")
            sh = wop(p, "qs", "shrui", raw, S["sh"])
            lo = wop(p, "ql", "andi", sh, wspl(0x0F0F0F0F if f != "q2k" else 0x03030303))
            if f == "q5k":
                b(f"    %{p}ha = index.add {boff}, %{lp}ho : index")
                hr = tow(p, ld(wv, nb, f"%{p}ha", p, "hr", 16), "hw")
                hs = wop(p, "hs", "shrui", hr, S["hsh"])
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
            ls = wop(p, "ls", "shrui", lr, S["sh"])
            hs = wop(p, "hs", "shrui", hr, S["hsh"])
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
            sh = wop(p, "qs", "shrui", raw, S["sh"])
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
        elif f == "iq4nl":
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            raw = tow(p, ld(wv, nb, f"%{p}qa", p, "qr", 16), "qw")
            sh = wop(p, "qs", "shrui", raw, S["sh"])
            nibw = wop(p, "nw", "andi", sh, wspl(0x0F0F0F0F))
            nib0 = tob(p, nibw, "nb0")
            b(f"    %{p}nb = vector.andi {nib0}, {e.splat8(15)} : vector<16xi8>")
            b(f"    %{p}q = vector.table.lookup %kvtu[%{p}nb] : vector<16xi8>, vector<16xi8> -> vector<16xi8>")
            q = f"%{p}q"
            bias = 128
            scale = ld_d(hv, nh, boff, 0, p, "d")
        elif f in ("iq3s", "iq3xxs", "iq2xxs", "iq2xs", "iq2s"):
            (gw, s0, s1), scale = decode_grid(f, S, p, wv, hv, nb, boff, lp)
            q = signed_grid(p, gw, s0, s1)
            bias = 64
        else:
            raise SystemExit("decode_w " + f)
        b(f"    %{p}bo_ = scalar.mulf {scale}, {e.cs(float(bias), 'f32')} : f32")
        return q, scale, f"%{p}bo_"

    def decode_grid(f, S, p, wv, hv, nb, boff, lp):
        """grid formats: ((grid words vector<4xi32>, sign byte 0, sign byte 1), scale)"""
        nh = nb // 2
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
            sg = []
            for qq in (0, 1):
                b(f"    %{p}sga{qq} = index.add {boff}, %{lp}sgo : index")
                b(f"    %{p}sgb{qq} = index.add %{p}sga{qq}, {e.ci(qq)} : index")
                sg.append(ld_u8(wv, nb, f"%{p}sgb{qq}", p, f"sgn{qq}"))
            q = (f"%{p}gw", sg[0], sg[1])
            b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
            sb_ = ld_u8(wv, nb, f"%{p}sa", p, "scb")
            b(f"    %{p}nw = scalar.shrui {sb_}, %{lp}sls : i32")
            b(f"    %{p}nib = scalar.andi %{p}nw, {e.cs(15)} : i32")
            b(f"    %{p}n2 = scalar.shli %{p}nib, {e.cs(1)} : i32")
            b(f"    %{p}n21 = scalar.addi %{p}n2, {e.cs(1)} : i32")
            b(f"    %{p}nf = scalar.sitofp %{p}n21 : i32 to f32")
            d = ld_d(hv, nh, boff, 0, p, "d")
            b(f"    %{p}scale = scalar.mulf {d}, %{p}nf : f32")
            return q, f"%{p}scale"
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
            sg = []
            for li in (0, 1):
                b(f"    %{p}ksh{li} = scalar.shrui {aux}, %{lp}ls{li} : i32")
                b(f"    %{p}ksi{li} = scalar.andi %{p}ksh{li}, {e.cs(127)} : i32")
                b(f"    %{p}ksx{li} = index.cast %{p}ksi{li} : i32 to index")
                sg.append(ld_u8("%t_ksigns", 128, f"%{p}ksx{li}", p, f"ks{li}"))
            q = (f"%{p}gw", sg[0], sg[1])
            b(f"    %{p}a28 = scalar.shrui {aux}, {e.cs(28)} : i32")
            b(f"    %{p}a28f = scalar.uitofp %{p}a28 : i32 to f32")
            b(f"    %{p}ah = scalar.addf %{p}a28f, {e.cs(0.5, 'f32')} : f32")
            d = ld_d(hv, nh, boff, 0, p, "d")
            b(f"    %{p}dah = scalar.mulf {d}, %{p}ah : f32")
            b(f"    %{p}scale = scalar.mulf %{p}dah, {e.cs(0.5 if f == 'iq3xxs' else 0.25, 'f32')} : f32")
            return q, f"%{p}scale"
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
            q = (f"%{p}gw", sg[0], sg[1])
            b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
            sb_ = ld_u8(wv, nb, f"%{p}sa", p, "scb")
            b(f"    %{p}nw = scalar.shrui {sb_}, %{lp}sls : i32")
            b(f"    %{p}nib = scalar.andi %{p}nw, {e.cs(15)} : i32")
            b(f"    %{p}nf = scalar.uitofp %{p}nib : i32 to f32")
            b(f"    %{p}nh = scalar.addf %{p}nf, {e.cs(0.5, 'f32')} : f32")
            d = ld_d(hv, nh, boff, 0, p, "d")
            b(f"    %{p}dn = scalar.mulf {d}, %{p}nh : f32")
            b(f"    %{p}scale = scalar.mulf %{p}dn, {e.cs(0.25, 'f32')} : f32")
            return q, f"%{p}scale"
        if f == "iq2s":
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            qv = ld(wv, nb, f"%{p}qa", p, "qv", 2)                       # qs[4 ib32 + l], l = l0, l0 + 1
            b(f"    %{p}hqa = index.add {boff}, %{lp}ho : index")
            qh = ld_u8(wv, nb, f"%{p}hqa", p, "qh")
            words = []
            for qq in (0, 1):
                b(f"    %{p}qi8_{qq} = vector.extract {qv}[{qq}] : vector<2xi8> -> i8")
                b(f"    %{p}qi_{qq} = scalar.extui %{p}qi8_{qq} : i8 to i32")
                b(f"    %{p}qhs_{qq} = scalar.shli {qh}, %{lp}qsA{qq} : i32")
                b(f"    %{p}qhb_{qq} = scalar.andi %{p}qhs_{qq}, {e.cs(768)} : i32")
                b(f"    %{p}gi_{qq} = scalar.ori %{p}qi_{qq}, %{p}qhb_{qq} : i32")
                b(f"    %{p}gx_{qq} = index.cast %{p}gi_{qq} : i32 to index")
                b(f"    %{p}gx2_{qq} = index.mul %{p}gx_{qq}, {e.ci(2)} : index")   # 8-byte grid entry
                w2 = ld("%t_grid_iq2s", 2048, f"%{p}gx2_{qq}", p, f"g{qq}", 2, "i32")
                for h in (0, 1):
                    b(f"    %{p}g{qq}_{h} = vector.extract {w2}[{h}] : vector<2xi32> -> i32")
                    words.append(f"%{p}g{qq}_{h}")
            b(f"    %{p}gw = vector.from_elements " + ", ".join(words) + " : vector<4xi32>")
            sg = []
            for qq in (0, 1):
                b(f"    %{p}sga{qq} = index.add {boff}, %{lp}sgo : index")
                b(f"    %{p}sgb{qq} = index.add %{p}sga{qq}, {e.ci(qq)} : index")
                sg.append(ld_u8(wv, nb, f"%{p}sgb{qq}", p, f"sgn{qq}"))
            q = (f"%{p}gw", sg[0], sg[1])
            b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
            sb_ = ld_u8(wv, nb, f"%{p}sa", p, "scb")
            b(f"    %{p}nw = scalar.shrui {sb_}, %{lp}sls : i32")
            b(f"    %{p}nib = scalar.andi %{p}nw, {e.cs(15)} : i32")
            b(f"    %{p}nf = scalar.uitofp %{p}nib : i32 to f32")
            b(f"    %{p}nh = scalar.addf %{p}nf, {e.cs(0.5, 'f32')} : f32")
            d = ld_d(hv, nh, boff, 0, p, "d")
            b(f"    %{p}dn = scalar.mulf {d}, %{p}nh : f32")
            b(f"    %{p}scale = scalar.mulf %{p}dn, {e.cs(0.25, 'f32')} : f32")
            return q, f"%{p}scale"
        raise SystemExit("decode_grid " + f)

    # ---- kernel body ----
    nw = len(fmts)
    wnb = [row_bytes(f, K) * M * (2 if rseg else 1) for f in fmts]
    assert not rseg or (kind == "swiglu" and fmts[0] == fmts[1] and not kseg and rseg_ok(M, rseg)), (kind, fmts, rseg)
    b("  %tid = kernel.workitem.id<x> : index")
    b("  %wg = kernel.workgroup.id<x> : index")
    b(f"  %lane = index.rem %tid, {e.ci(32)} : index")
    b(f"  %wave = index.div %tid, {e.ci(32)} : index")
    b(f"  %wgw = index.mul %wg, {e.ci(W)} : index")
    b("  %wid = index.add %wgw, %wave : index")
    b(f"  %row0 = index.mul %wid, {e.ci(R)} : index")
    S = [lane_setup(f, f"f{i}") for i, f in enumerate(fmts)]
    # row byte bases: row (row0 + r) * row_bytes + lane block offset
    if kseg:
        assert len(fmts) == 1 and FMT[fmts[0]][0] == 256 and kseg_ok(kseg, K), kseg
    for i, f in enumerate(fmts):
        rb = row_bytes(f, K)
        for r in range(R):
            b(f"  %f{i}row{r} = index.add %row0, {e.ci(r)} : index")
            if kseg:   # segment s: (M - 1) k0 bb + row w bb + lane block offset (+ t 2 bb in the loop)
                bb = FMT[f][1]
                for si, (k0, w) in enumerate(kseg):
                    b(f"  %f{i}rb{r}s{si} = index.mul %f{i}row{r}, {e.ci(w * bb)} : index")
                    b(f"  %f{i}rc{r}s{si} = index.add %f{i}rb{r}s{si}, {e.ci((M - 1) * k0 * bb)} : index")
                    b(f"  %f{i}base{r}s{si} = index.add %f{i}rc{r}s{si}, %f{i}lboff : index")
                continue
            if rseg:   # (RSEG) logical row row + i M at its physical row
                lg, f0_ = f"%f{i}row{r}" if not i else f"%f{i}lg{r}", M - rseg
                if i:
                    b(f"  {lg} = index.add %f{i}row{r}, {e.ci(M)} : index")
                q_ = f"%f{i}q{r}"
                bs_ = e.ci(M + f0_ if i else f0_)   # (gate rows below M, up rows from M: the NPU's from f0 on)
                b(f"  {q_}npu = index.cmp uge, {lg}, {bs_} : index")
                b(f"  {q_}lgc = index.max {lg}, {bs_} : index")   # (clamps: the range proof's bounds)
                b(f"  {q_}rel = index.sub {q_}lgc, {bs_} : index")
                # (the block by compares: this target refuses index.div / rem by a non-power of two)
                g_ = e.ci(0)
                for k_ in range(1, rseg // RSEG_B):
                    b(f"  {q_}c{k_} = index.cmp uge, {q_}rel, {e.ci(RSEG_B * k_)} : index")
                    b(f"  {q_}d{k_} = scf.select {q_}c{k_}, {e.ci(1)}, {e.ci(0)} : index")
                    b(f"  {q_}g{k_} = index.add {g_}, {q_}d{k_} : index")
                    g_ = f"{q_}g{k_}"
                b(f"  {q_}gb = index.mul {g_}, {e.ci(RSEG_B)} : index")
                b(f"  {q_}w0 = index.sub {q_}rel, {q_}gb : index")
                b(f"  {q_}w1 = index.max {q_}w0, {e.ci(0)} : index")
                b(f"  {q_}w = index.min {q_}w1, {e.ci(RSEG_B - 1)} : index")
                b(f"  {q_}s0 = index.mul {g_}, {e.ci(2)} : index")
                b(f"  {q_}s = index.add {q_}s0, {e.ci(i)} : index")
                b(f"  {q_}sb = index.mul {q_}s, {e.ci(RSEG_B)} : index")
                b(f"  {q_}p0 = index.add {q_}sb, {q_}w : index")
                b(f"  {q_}p = index.add {q_}p0, {e.ci(f0_)} : index")
                lo_ = lg
                if i:   # (the GPU's up rows: F rows on, after the NPU's run)
                    lo_ = f"{q_}lo"
                    b(f"  {lo_} = index.add {lg}, {e.ci(rseg)} : index")
                b(f"  {q_}r = scf.select {q_}npu, {q_}p, {lo_} : index")
                b(f"  %f{i}rb{r} = index.mul {q_}r, {e.ci(rb)} : index")
                b(f"  %f{i}base{r} = index.add %f{i}rb{r}, %f{i}lboff : index")
                continue
            b(f"  %f{i}rb{r} = index.mul %f{i}row{r}, {e.ci(rb)} : index")
            b(f"  %f{i}base{r} = index.add %f{i}rb{r}, %f{i}lboff : index")
    b(f"  %xl16 = index.mul %lane, {e.ci(16)} : index")
    accs = [f"%a{i}_{r}" for i in range(nw) for r in range(R)]
    b("  %zf = scalar.constant 0.0 : f32")
    inits = ", ".join(f"{a} = %zf : f32" for a in accs)
    res = [f"%res{j}" for j in range(len(accs))]
    b(f"  {', '.join(res)} = scf.for %t = [{e.ci(0)} to {e.ci(NT)} step {e.ci(1)}]({inits}) -> ({', '.join(['f32'] * len(accs))}) pipeline({e.ci(PIPE)}) {{")
    b(f"    %xo0 = index.mul %t, {e.ci(512)} : index")
    b("    %xo = index.add %xo0, %xl16 : index")
    xa = ld("%xv", K, "%xo", "", "xa", 16, "f32")   # (not "xv": that would shadow the view)
    b(f"    %xsum = vector.reduce<addf> {xa}, %zf : vector<16xf32>, f32")
    nxt = [None] * len(accs)
    for i, f in enumerate(fmts):
        for r in range(R):
            p = f"d{i}_{r}"
            b(f"    %{p}to = index.mul %t, {e.ci(S[i]['tstep'])} : index")
            base = f"%f{i}base{r}"
            if kseg:   # the step's segment (its two blocks share one: segments start on even blocks)
                base = f"%f{i}base{r}s0"
                for si, (k0, _) in enumerate(kseg[1:], 1):
                    b(f"    %{p}in{si} = index.cmp uge, %t, {e.ci(k0 // 2)} : index")
                    b(f"    %{p}bs{si} = scf.select %{p}in{si}, %f{i}base{r}s{si}, {base} : index")
                    base = f"%{p}bs{si}"
            b(f"    %{p}bo = index.add {base}, %{p}to : index")
            q, scale, offs = decode_w(f, S[i], p, f"%wv{i}", f"%wh{i}", wnb[i], f"%{p}bo", f"f{i}")
            b(f"    %{p}qf = vector.uitofp {q} : vector<16xi8> to vector<16xf32>")
            b(f"    %{p}pr = vector.mulf %{p}qf, {xa} : vector<16xf32>")
            b(f"    %{p}dot = vector.reduce<addf> %{p}pr, %zf : vector<16xf32>, f32")
            b(f"    %{p}sd = scalar.mulf {scale}, %{p}dot : f32")
            b(f"    %{p}ox = scalar.mulf {offs}, %xsum : f32")
            b(f"    %{p}c = scalar.subf %{p}sd, %{p}ox : f32")
            b(f"    %{p}n = scalar.addf %a{i}_{r}, %{p}c : f32")
            nxt[i * R + r] = f"%{p}n"
    b(f"    scf.yield {', '.join(nxt)} : {', '.join(['f32'] * len(accs))}")
    b("  }")
    if peel:
        peel_loop(body, e, res, accs, inits, nxt, peel)
    # butterfly reduction over the 32 lanes: xor 16, 8, 4, 2, 1
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
        oc = f"%orc{r}"
        b(f"    %orow{r} = index.add %row0, {e.ci(r)} : index")
        b(f"    %orc{r} = index.min %orow{r}, {e.ci(M - 1)} : index")
        if kind == "plain":
            v = tot[r]
        elif kind == "resid":
            b(f"    %old{r} = view.load %yv[{oc}] : view<{M}xf32> -> f32")
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
        b(f"    view.store {v}, %yv[{oc}] : f32, view<{M}xf32>")
    b("  }")
    if amax:
        assert kind == "plain"
        amax_tail(b, e, M, R, W, tot)
    if _parts is not None:               # gen_bands: hand back the body, constants stay in e
        _parts.update(body=body, wnb=wnb, tabs=tabs)
        return None
    b("  kernel.return")

    # ---- assemble ----
    out = []
    o = out.append
    o(f"// GENERATED by tools/gen_gemv.py ({kind} {'/'.join(fmts)} M={M} K={K} R={R} W={W})"
      " -- edit the generator.")
    o("amdgpu.target<gfx1151> @gv32 {subgroup_size = 32}")
    o("")
    o(f"kernel.def target(@gv32) @{name}() {{")
    o("  %u1 = index.constant 1 : index")
    o(f"  %nwg = index.constant {M // rows_wg} : index")
    o(f"  %wgs = index.constant {32 * W} : index")
    o("  kernel.launch.config workgroups(%nwg, %u1, %u1) workgroup_size(%wgs, %u1, %u1) : index")
    params = [f"%w{i}: buffer" for i in range(nw)] + [f"%{t}: buffer" for t in tabs] + ["%x: buffer", "%y: buffer"]
    params += ["%ak: buffer", "%atok: buffer"] if amax else []
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
    if amax:
        o("  %akv = buffer.view %ak_na[%base] : buffer -> view<2xi64>")
        o("  %acv = buffer.view %ak_na[%base] : buffer -> view<4xi32>")
        o("  %atv = buffer.view %atok_na[%base] : buffer -> view<1xi32>")
    if "iq4xs" in fmts or "iq4nl" in fmts:
        o("\n".join(f"  %kvu{i} = scalar.constant {v + 128 - 256 if v + 128 > 127 else v + 128} : i8" for i, v in enumerate(IQ4_KVALUES)))
        o("  %kvtu = vector.from_elements " + ", ".join(f"%kvu{i}" for i in range(16)) + " : vector<16xi8>")
    for k in sorted(e.consts, key=lambda n: (not n.startswith("%c"), n)):
        if not k.startswith("%sv") and not k.startswith("%ws"):
            o(e.consts[k])
    for k in sorted(e.consts):
        if k.startswith("%sv") or k.startswith("%ws"):
            o(e.consts[k])
    out += body
    o("}")
    return "\n".join(out) + "\n"


def kseg_layout(kb, f_npu, parts):
    """The K segments of a weight whose last f_npu columns the NPU reads in parts ([(first pass, passes)], a pass 4
    blocks; emit_prefill_pp ffnblk_parts): [(0, the GPU's blocks)] + [(its first block, its blocks)] a part."""
    g = kb - f_npu // 256
    return [(0, g)] + [(g + 4 * p0, 4 * n) for p0, n in parts]


def kseg_ok(kseg, K):
    """Segments that tile K's 256-wide blocks in order, each starting on an even block (a GEMV step's two blocks)."""
    k = 0
    for k0, w in kseg:
        if k0 != k or w <= 0 or k0 % 2:
            return False
        k += w
    return k * 256 == K


# Row segments (RSEG): an FFN layer whose last F gate / up features the NPU's 5-slab gate / up image reads (emit_prefill_pp
# FFNSW5, gen_npu_q qsw5). Its 16-row records take 8 gate rows and the same features' 8 up rows, which the shim DMA
# reaches by a step of under 2^20 words: so those rows are stored in blocks of RSEG_B, gate block g then up block g, a
# pair per 2 RSEG_B rows, in one run of 2 F rows from the gate tensor's last F rows on; the up tensor's other rows (the
# GPU's) after the run (LoomWeights, in place at load; any F / RSEG_B, the pairs never straddle the GPU's up rows).
# Logical row L over gate and up ([0, M) gate, [M, 2M) up) at physical row rseg_phys(L), both from the gate tensor's
# start: gate rows below M - F stay, up rows below M + M - F move F rows on.
RSEG_B = 1280


def rseg_ok(M, F):
    return 0 < F < M and F % RSEG_B == 0


def rseg_phys(L, M, F):
    """The physical row (from the gate tensor's start) of logical row L over gate and up."""
    f0 = M - F
    base = M + f0 if L >= M else f0
    if L < base:
        return L + (F if L >= M else 0)
    g, w = divmod(L - base, RSEG_B)
    return f0 + RSEG_B * (2 * g + (L >= M)) + w


def rseg_emit_i32(e, src, dst, M, F, off=0):
    """%dst (i32) = rseg_phys(%src + off) (gen_gemm_tile RSEG); e: the emitter's line function."""
    assert rseg_ok(M, F), (M, F)
    f0 = M - F
    c = lambda n, v: e(f"  %rs_{n} = scalar.constant {v} : i32")
    c("off", off), c("m", M), c("b0", f0), c("b1", M + f0), c("bb", RSEG_B), c("two", 2), c("one", 1), c("zero", 0)
    c("f", F)
    for ln in (f"%rs_l = scalar.addi {src}, %rs_off : i32",
               "%rs_up = scalar.cmpi uge, %rs_l, %rs_m : i32",
               "%rs_base = scf.select %rs_up, %rs_b1, %rs_b0 : i32",
               "%rs_npu = scalar.cmpi uge, %rs_l, %rs_base : i32",
               "%rs_rel = scalar.subi %rs_l, %rs_base : i32",
               "%rs_g = scalar.divui %rs_rel, %rs_bb : i32",
               "%rs_gb = scalar.muli %rs_g, %rs_bb : i32",
               "%rs_w = scalar.subi %rs_rel, %rs_gb : i32",
               "%rs_g2 = scalar.muli %rs_g, %rs_two : i32",
               "%rs_ui = scf.select %rs_up, %rs_one, %rs_zero : i32",
               "%rs_s = scalar.addi %rs_g2, %rs_ui : i32",
               "%rs_sb = scalar.muli %rs_s, %rs_bb : i32",
               "%rs_p0 = scalar.addi %rs_sb, %rs_w : i32",
               "%rs_p = scalar.addi %rs_p0, %rs_b0 : i32",
               "%rs_uf = scf.select %rs_up, %rs_f, %rs_zero : i32",   # (the GPU's up rows: F rows on)
               "%rs_lo = scalar.addi %rs_l, %rs_uf : i32",
               f"{dst} = scf.select %rs_npu, %rs_p, %rs_lo : i32"):
        e("  " + ln)


def rseg_suffix(F):
    """A kernel name's suffix for row-segmented gate / up weights."""
    return "_rseg%d" % F if F else ""


def kseg_suffix(kseg):
    """A kernel name's suffix for K-segmented weights: "_kseg" and the segments' block counts."""
    return "_kseg" + "_".join(str(w) for _, w in kseg) if kseg else ""


def peel_loop(body, e, res, accs, inits, nxt, peel):
    """gen peel: copies of the loop body for t = 0 .. peel - 1 before the loop, split by dataflow: lines that read x
    (the %xv loads and everything after them, the accumulators) go after the chain.wait marker."""
    fi = next(i for i, l in enumerate(body) if l.startswith(f"  {', '.join(res)} = scf.for %t = "))
    yi = next(i for i in range(fi, len(body)) if body[i].strip().startswith("scf.yield"))
    inner = body[fi + 1:yi]
    defre = re.compile(r"^\s*(%\w+(?:\s*,\s*%\w+)*)\s*=")

    def outs(l):
        m = defre.match(l)
        return set(n.strip() for n in m.group(1).split(",")) if m else set()

    defs = set().union(*(outs(l) for l in inner))
    pre, post, prev = [], [], ["%zf"] * len(accs)
    for k in range(peel):
        ren = {n: f"{n}_p{k}" for n in defs}
        ren["%t"] = e.ci(k)
        ren.update(dict(zip(accs, prev)))
        dep = set(prev) - {"%zf"} | {"%xv"} | set(accs)
        early, late = [], []
        for l in inner:
            out = outs(l)
            nl = re.sub(r"%\w+", lambda mm: ren.get(mm.group(0), mm.group(0)), l).replace("    ", "  ", 1)
            if (set(re.findall(r"%\w+", l)) - out) & dep:
                dep |= out
                late.append(nl)
            else:
                early.append(nl)
        pre += early
        post += late
        prev = [ren[n] for n in nxt]
    loop = body[fi].replace(f"[{e.ci(0)} to", f"[{e.ci(peel)} to").replace(
        f"({inits})", "(" + ", ".join(f"{a} = {v} : f32" for a, v in zip(accs, prev)) + ")")
    body[fi:fi + 1] = pre + ["  // chain.wait"] + post + [loop]


def amax_tail(b, e, M, R, W, tot):
    """gen amax: yah_argmax_f32's result (NaN and +-inf skipped, ties keep the lowest index, -0 = +0) without its pass.
    Each row's key is (order-preserving bits of y, ~row) as a u64 (skipped rows 0); the wave max goes through LDS to
    tid 0, which takes an atomic max into ak[0] and counts the workgroup in ak[2]. The last workgroup swaps the key
    back to 0, resets the count and stores the key's row (M if every row was skipped) to tok[0]."""
    b(f"  %akl = index.constant {8 * W} : offset")
    b("  %akp = buffer.alloca<workgroup> align(16) %akl : buffer")
    b(f"  %akw = buffer.view %akp[%base] : buffer -> view<{W}xi64>")
    b("  %amlo = scalar.constant -3.0000000000000001e+38 : f32")   # yah_argmax_f32's bounds
    b("  %amhi = scalar.constant 3.0000000000000001e+38 : f32")
    keys = []
    for r in range(R):
        p = f"%am{r}"
        b(f"  {p}lo = scalar.cmpf oge, {tot[r]}, %amlo : f32")
        b(f"  {p}hi = scalar.cmpf ole, {tot[r]}, %amhi : f32")
        b(f"  {p}ok = scalar.andi {p}lo, {p}hi : i1")
        b(f"  {p}b = scalar.bitcast {tot[r]} : f32 to i32")
        b(f"  {p}s = scalar.shrsi {p}b, {e.cs(31)} : i32")
        b(f"  {p}m = scalar.ori {p}s, {e.cs(-2147483648)} : i32")
        b(f"  {p}u0 = scalar.xori {p}b, {p}m : i32")             # negative: ~bits, else bits | sign
        b(f"  {p}nz = scalar.cmpi eq, {p}u0, {e.cs(2147483647)} : i32")   # -0 sorts as +0
        b(f"  {p}u = scf.if {p}nz -> (i32) {{")
        b(f"    scf.yield {e.cs(-2147483648)} : i32")
        b("  } else {")
        b(f"    scf.yield {p}u0 : i32")
        b("  }")
        b(f"  {p}hw = scalar.extui {p}u : i32 to i64")
        b(f"  {p}h = scalar.shli {p}hw, {e.cs(32, 'i64')} : i64")
        b(f"  {p}row = index.add %row0, {e.ci(r)} : index")
        b(f"  {p}ri = index.cast {p}row : index to i32")
        b(f"  {p}rn = scalar.xori {p}ri, {e.cs(-1)} : i32")
        b(f"  {p}lw = scalar.extui {p}rn : i32 to i64")
        b(f"  {p}k0 = scalar.ori {p}h, {p}lw : i64")
        b(f"  {p}k = scf.if {p}ok -> (i64) {{")
        b(f"    scf.yield {p}k0 : i64")
        b("  } else {")
        b(f"    scf.yield {e.cs(0, 'i64')} : i64")
        b("  }")
        keys.append(f"{p}k")
    wk = keys[0]
    for i, k in enumerate(keys[1:]):
        b(f"  %amw{i} = scalar.maxui {wk}, {k} : i64")
        wk = f"%amw{i}"
    b("  scf.if %is0 {")
    b(f"    view.store {wk}, %akw[%wave] : i64, view<{W}xi64>")
    b("  }")
    b("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    b(f"  %amt0 = index.cmp eq, %tid, {e.ci(0)} : index")
    b("  scf.if %amt0 {")
    gk = None
    for w in range(W):
        b(f"    %amg{w} = view.load %akw[{e.ci(w)}] : view<{W}xi64> -> i64")
        if gk:
            b(f"    %amx{w} = scalar.maxui {gk}, %amg{w} : i64")
            gk = f"%amx{w}"
        else:
            gk = f"%amg{w}"
    b(f"    view.atomic.reduce<maxui> {gk}, %akv[{e.ci(0)}] {{ordering = relaxed, scope = device}} : i64, view<2xi64>")
    b(f"    %amo = view.atomic.rmw<addi> {e.cs(1)}, %acv[{e.ci(2)}] {{ordering = acq_rel, scope = device}} : i32, view<4xi32> -> i32")
    b(f"    %amlast = scalar.cmpi eq, %amo, {e.cs(M // (R * W) - 1)} : i32")
    b("    scf.if %amlast {")
    b(f"      %amk = view.atomic.rmw<xchgi> {e.cs(0, 'i64')}, %akv[{e.ci(0)}] {{ordering = acq_rel, scope = device}} : i64, view<2xi64> -> i64")
    b(f"      %amc = view.atomic.rmw<xchgi> {e.cs(0)}, %acv[{e.ci(2)}] {{ordering = relaxed, scope = device}} : i32, view<4xi32> -> i32")
    b("      %amkl = scalar.trunci %amk : i64 to i32")
    b(f"      %amr = scalar.xori %amkl, {e.cs(-1)} : i32")
    b(f"      %amnone = scalar.cmpi eq, %amk, {e.cs(0, 'i64')} : i64")
    b("      %amtok = scf.if %amnone -> (i32) {")
    b(f"        scf.yield {e.cs(M)} : i32")
    b("      } else {")
    b("        scf.yield %amr : i32")
    b("      }")
    b(f"      view.store %amtok, %atv[{e.ci(0)}] : i32, view<1xi32>")
    b("    }")
    b("  }")


_SHARED = re.compile(r"%(c[iwbf]\w*|ws\w*|sv\w*|kvtu|t_\w+|xv)$")


def gen_bands(fmts, Ms, K, R=2, W=4, name="yah_gemv", peel=0):
    """Several plain GEMVs over one input in one dispatch; workgroup ranges map to bands.

    Band b is fmts[b] with Ms[b] rows, written to its own output.
    Bindings: the weights, the IQ tables of all bands (TABLE_ORDER), x, then the outputs.
    """
    rows_wg = R * W
    e = E()
    bodies, wnbs, starts = [], [], []
    start = 0
    for bi, (f, M) in enumerate(zip(fmts, Ms)):
        assert M % rows_wg == 0, (f, M)
        parts = {}
        gen("plain", [f], M, K, R, W, _e=e, _parts=parts, peel=peel)
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
    if "iq4xs" in fmts or "iq4nl" in fmts:
        o("\n".join(f"  %kvu{i} = scalar.constant {v + 128 - 256 if v + 128 > 127 else v + 128} : i8" for i, v in enumerate(IQ4_KVALUES)))
        o("  %kvtu = vector.from_elements " + ", ".join(f"%kvu{i}" for i in range(16)) + " : vector<16xi8>")
    for k in sorted(e.consts, key=lambda n: (not n.startswith("%c"), n)):
        if not k.startswith("%sv") and not k.startswith("%ws"):
            o(e.consts[k])
    for k in sorted(e.consts):
        if k.startswith("%sv") or k.startswith("%ws"):
            o(e.consts[k])
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
    # Constants created after the header was built (e.ci(end) above) are in e.consts but not in the text: insert them.
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
