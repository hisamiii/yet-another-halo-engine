#!/usr/bin/env python3
"""The row 5 swap's weight decode folded into the GPU's IQ4_XS gate / up GEMM (its NPU split, "gemm_ffn_iq4xs_<mt>_20
.af.to.npu.hal"): besides its own rows the GEMM decodes the NPU's 8192 gate + 8192 up rows of another layer into the
swap's weight stream (the layout gen_iq4xs_bfp stream mode writes, ks 32 x 20 passes, tile 64), the way the GEMM
decodes its own weights:
  - the biased codebook (kv + 128, u8) through one table lookup, byte pairs as f16 subnormals c 2^-24, and
    w = fma(s 2^24, c 2^-24, -128 s) = s kv exactly in f32 (gen_gemm_decode._iq4_f16's trick);
  - per 8 weights E = the exponent of max |w|, the bump if the largest positive w rounds to 128 (bfp_hw), and
    m = roundeven(w 2^(133 - E')) as fma(w, 2^(133 - E'), magic): one rounding of an exact product.
Work (the slab map): workgroup G < 512 owns stream row groups q = 2 G, 2 G + 1 (tensor q / 512, rows 16 (q % 512) ..),
its wave w the k-blocks 2 w, 2 w + 1 of every pass; K phase p writes the slab of (q = 2 G + p % 2, pass p / 2) whole,
lane L of wave w its 9-byte record at 288 w + 9 L. Each lane loads its next phase's raw bytes one phase ahead.
Measured (Scratch/gpudec/fuse/fuse4.py, slab map + FMF + UG, 11-20 09:3x): bit-exact, the fused GEMM ~0.1 ms faster
than the GEMM alone. Two rules held for every build there: a lane value used inside the K loop beyond the record
offset makes the compiler allocate 180 VGPRs (+0.3 ms); cross-lane store assembly loses.
Two bindings follow the GEMM's: the raw weights (gate then up, each N rows of K / 256 x 136 B; the NPU rows the last
F of each) and the stream (2 F K / 8 records of 9 B)."""
import os
import re

import gen_iq4xs_bfp as G

K, F = 5120, 8192
RBG = K // 256 * 136          # 2720 B a raw row
KB = K // 8                   # 640 k-blocks (8 weights, one 9 B record) a row
NACT = 2 * F * KB // 40       # 262144 decode lanes, one record a K phase
KS, PASSES, TILE = 32, 20, 64
SB = 144 * KS                 # 4608 B a slab
PHASES = 40                   # the GEMM's K phases (K / 128)
LANES, WGS = 512, 512         # decode lanes a workgroup, workgroups that decode
KEEP = {"kp", "tid", "stg_lin", "c0", "base"}
# The VGPR budget to allocate in (emit_prefill YAH_VGPR_TRY): on the fragment-major (tout) split at the default 192
# the allocator squeezes the fused kernel to 180 VGPRs and spreads the GEMM's weight ds_stores over 134
# instructions (+0.66 ms); at 208 it takes 192 and the K loop matches the row-major fuse4 build (757 instructions,
# ds_store span 45 vs 40).
VGPR_TRY = "208"


def out_bytes():
    return 2 * F * KB * 9


def in_bytes(n):
    return 2 * n * RBG


def _pre(lines, p="dz_"):
    return [re.sub(r"%([A-Za-z_]\w*)", lambda m: "%" + (m.group(1) if m.group(1) in KEEP or m.group(1).startswith(p)
                                                         else p + m.group(1)), ln) for ln in lines]


def _frame_prologue(e, n, nin, nout):
    """The constants, views and lane map every payload shares (the slab map and the record's bfp math)."""
    for nm, v in (("nin", nin), ("nout", nout), ("ldmax", nin - 8), ("stmax", nout - 9), ("lanes", LANES),
                  ("cact", NACT), ("c1", 1), ("c2", 2), ("c4", 4), ("c8", 8), ("c9", 9), ("c16", 16), ("c20", 20),
                  ("csb", SB), ("c32", 32), ("c512", 512), ("c288", 288), ("c511u", WGS - 1)):
        e(f"  %{nm} = index.constant {v} : index")
    e("  %wv = buffer.view %dz_w_na[%base] : buffer -> view<[%nin]xi8>")
    e("  %ov = buffer.view %dz_out_na[%base] : buffer -> view<[%nout]xi8>")
    e("  %L0 = index.mul %stg_lin, %lanes : index")
    e("  %L = index.add %L0, %tid : index")
    e("  %act = index.cmp ult, %L, %cact : index")
    # the workgroup and lane from the workgroup id and tid (scalar record base); inactive workgroups clamp to 511
    e("  %Gc = index.min %stg_lin, %c511u : index")
    e("  %tl = index.add %tid, %c0 : index")
    e("  %ww = index.div %tl, %c32 : index")
    e("  %Ll = index.rem %tl, %c32 : index")
    e("  %bb = index.div %Ll, %c16 : index")
    e("  %r16 = index.rem %Ll, %c16 : index")
    e("  %jp = index.div %ww, %c2 : index")        # sub-block (k-blocks 4 j' .. 4 j' + 3)
    e("  %whi = index.rem %ww, %c2 : index")       # nibble half
    e("  %G2 = index.mul %Gc, %c2 : index")
    e("  %ow = index.mul %ww, %c288 : index")
    e("  %oL = index.mul %Ll, %c9 : index")
    e("  %olane = index.add %ow, %oL : index")
    for nm, v in (("i1", 1), ("i23", 23), ("iemask", 0x7F800000), ("ib260", (260 << 23) - 2 ** 32),
                  ("ione", 0x800000), ("i0", 0)):
        e(f"  %{nm} = scalar.constant {v} : i32")
    e("  %f1275 = scalar.constant 127.5 : f32")
    e("  %magic = scalar.constant 12582912.0 : f32")


def _frame_loads(e, kpv, t, ind):
    """Phase kpv's row group and tensor: %<t>pq (the pass, K / 256 super-block), %<t>tt (0 gate, 1 up; workgroup
    uniform), %<t>g16 (row group in the tensor: rows 16 g16 + r16)."""
    x = lambda s: e(ind + s)
    x(f"%{t}pq = index.div {kpv}, %c2 : index")
    x(f"%{t}pp = index.rem {kpv}, %c2 : index")
    x(f"%{t}q = index.add %G2, %{t}pp : index")
    x(f"%{t}tt = index.div %{t}q, %c512 : index")
    x(f"%{t}g16 = index.rem %{t}q, %c512 : index")


class IQ4XS:
    """The IQ4_XS payload: per lane and phase its row's super-block header (8 B) and qs half (8 B), carried as two
    vector<2xi32>; 8 weights s kv exact through the biased codebook as f16 subnormals."""
    buffers = []           # extra launch buffers (names, or (name, view type)), bound after the raw weights
    row_bytes = RBG        # bytes a raw row (the raw binding: gate's n rows, then up's)

    def __init__(self, tensor_offset=0, prefix=""):
        self.toff, self.p = tensor_offset, prefix   # (a pair's up: its rows after the gate's n; names prefixed)

    def prologue(self, e, n, nin, nout):
        p = self.p
        for nm, v in (("c136", 136), ("ctb", n * RBG), ("cr0", (n - F) * RBG), ("crb", RBG)):
            e(f"  %{p}{nm} = index.constant {v} : index")
        e(f"  %{p}rbo = index.mul %r16, %{p}crb : index")
        e(f"  %{p}qj = index.mul %jp, %c16 : index")
        e(f"  %{p}qbb = index.mul %bb, %c8 : index")
        e(f"  %{p}qj8 = index.add %{p}qj, %c8 : index")
        e(f"  %{p}qoff = index.add %{p}qj8, %{p}qbb : index")
        for nm, v in (("i2", 2), ("i3", 3), ("i15", 15), ("i16", 16), ("i32", 32), ("im00ff", 0x00FF00FF)):
            e(f"  %{p}{nm} = scalar.constant {v} : i32")
        e(f"  %{p}f2p24 = scalar.constant 16777216.0 : f32")
        e(f"  %{p}fm128 = scalar.constant -128.0 : f32")
        e(f"  %{p}c15b = scalar.constant 15 : i8")
        e(f"  %{p}m15v = vector.splat %{p}c15b : vector<8xi8>")
        for i, v in enumerate(G.KVALUES):
            u = v + 128
            e(f"  %{p}kvu{i} = scalar.constant {u - 256 if u > 127 else u} : i8")
        e(f"  %{p}kvtu = vector.from_elements " + ", ".join(f"%{p}kvu{i}" for i in range(16)) + " : vector<16xi8>")

    def loads(self, e, kpv, t, ind, ft=None):
        """After _frame_loads(kpv, ft): the header and qs half of super-block %<ft>pq (t: this payload's tag)."""
        x = lambda s: e(ind + s)
        p, f = self.p, t if ft is None else ft
        x(f"%{t}r0 = index.mul %{f}tt, %{p}ctb : index")
        x(f"%{t}r1 = index.add %{t}r0, %{p}cr0 : index")
        x(f"%{t}g16r = index.mul %{f}g16, %c16 : index")
        x(f"%{t}r2 = index.mul %{t}g16r, %{p}crb : index")
        x(f"%{t}r3 = index.add %{t}r1, %{t}r2 : index")
        x(f"%{t}r4 = index.add %{t}r3, %{p}rbo : index")
        x(f"%{t}po = index.mul %{f}pq, %{p}c136 : index")
        x(f"%{t}ha = index.add %{t}r4, %{t}po : index")
        x(f"%{t}qa = index.add %{t}ha, %{p}qoff : index")
        x(f"%{t}hl = index.min %{t}ha, %ldmax : index")
        x(f"%{t}ql = index.min %{t}qa, %ldmax : index")
        x(f"%{t}hv = vector.load %wv[%{t}hl] : view<[%nin]xi8> -> vector<8xi8>")
        x(f"%{t}hp = vector.bitcast %{t}hv : vector<8xi8> to vector<2xi32>")
        x(f"%{t}qv = vector.load %wv[%{t}ql] : view<[%nin]xi8> -> vector<8xi8>")
        x(f"%{t}qp = vector.bitcast %{t}qv : vector<8xi8> to vector<2xi32>")
        return [(f"%{t}hp", "vector<2xi32>"), (f"%{t}qp", "vector<2xi32>")]

    def decode(self, e, carried, ind, t=""):
        """The 8 weights (f32 names, exact) of k-block 32 %pq + 4 jp + 2 whi + bb from the carried header and qs;
        every local name tagged with t, prologue names with the payload prefix."""
        p = self.p
        local = re.compile(r"%(?!(?:whi|jp|c0|i0|i1|" + "|".join(
            f"{p}{c}" for c in ("i2", "i3", "i15", "i16", "i32", "im00ff", "f2p24", "fm128", "m15v", "kvtu"))
            + r")\b)([A-Za-z_]\w*)")
        x = lambda s: e(ind + local.sub(lambda m: "%" + t + m.group(1), s))
        hd, qd = carried
        x("%phi = index.add %whi, %c0 : index")
        # scale of sub-block j': ((scales_l >> 4 j') & 15 | ((scales_h >> 2 j') & 3) << 4) - 32, times d
        e(ind + f"%{t}hw0 = vector.extract {hd}[0] : vector<2xi32> -> i32")
        e(ind + f"%{t}hw1 = vector.extract {hd}[1] : vector<2xi32> -> i32")
        x("%d16 = scalar.trunci %hw0 : i32 to i16")
        x("%dh = scalar.bitcast %d16 : i16 to f16")
        x("%d = scalar.extf %dh : f16 to f32")
        x("%jpi = index.cast %jp : index to i32")
        x(f"%sls = scalar.shli %jpi, %{p}i2 : i32")
        x("%slw = scalar.shrui %hw1, %sls : i32")
        x(f"%scl = scalar.andi %slw, %{p}i15 : i32")
        x("%shs0 = scalar.shli %jpi, %i1 : i32")
        x(f"%shs = scalar.addi %shs0, %{p}i16 : i32")
        x("%sha = scalar.shrui %hw0, %shs : i32")
        x(f"%sch = scalar.andi %sha, %{p}i3 : i32")
        x(f"%ls = scalar.fmai %sch, %{p}i16, %scl : i32")
        x(f"%sc = scalar.subi %ls, %{p}i32 : i32")
        x("%scf = scalar.sitofp %sc : i32 to f32")
        x("%s = scalar.mulf %d, %scf : f32")
        x(f"%s24 = scalar.mulf %s, %{p}f2p24 : f32")
        x(f"%nb = scalar.mulf %s, %{p}fm128 : f32")
        # 8 nibbles of this k-block: the low or high nibbles of the carried 8 bytes
        x("%phii = index.cast %phi : index to i32")
        x(f"%nsh = scalar.shli %phii, %{p}i2 : i32")
        e(ind + f"%{t}qx0 = vector.extract {qd}[0] : vector<2xi32> -> i32")
        e(ind + f"%{t}qx1 = vector.extract {qd}[1] : vector<2xi32> -> i32")
        x("%qy0 = scalar.shrui %qx0, %nsh : i32")
        x("%qy1 = scalar.shrui %qx1, %nsh : i32")
        x("%nw = vector.from_elements %qy0, %qy1 : vector<2xi32>")
        x("%nc = vector.bitcast %nw : vector<2xi32> to vector<8xi8>")
        x(f"%nb8 = vector.andi %nc, %{p}m15v : vector<8xi8>")
        x(f"%cu = vector.table.lookup %{p}kvtu[%nb8] : vector<16xi8>, vector<8xi8> -> vector<8xi8>")
        x("%cuw = vector.bitcast %cu : vector<8xi8> to vector<2xi32>")
        ws = [None] * 8
        for w in range(2):
            x(f"%cw{w} = vector.extract %cuw[{w}] : vector<2xi32> -> i32")
            x(f"%cwe{w} = scalar.andi %cw{w}, %{p}im00ff : i32")          # bytes 0, 2 as f16 subnormals
            x(f"%cwz{w} = vector.from_elements %cw{w}, %i0 : vector<2xi32>")
            x(f"%cwb{w} = vector.bitcast %cwz{w} : vector<2xi32> to vector<8xi8>")
            x(f"%cwq{w} = vector.shuffle<[1, 4, 3, 4, 4, 4, 4, 4]> %cwb{w} : vector<8xi8>")   # bytes 1, 3
            x(f"%cwr{w} = vector.bitcast %cwq{w} : vector<8xi8> to vector<2xi32>")
            x(f"%cwo{w} = vector.extract %cwr{w}[0] : vector<2xi32> -> i32")
            for hh, src in ((0, f"%cwe{w}"), (1, f"%cwo{w}")):
                x(f"%cv{w}_{hh} = vector.from_elements {src} : vector<1xi32>")
                x(f"%ch{w}_{hh} = vector.bitcast %cv{w}_{hh} : vector<1xi32> to vector<2xf16>")
                for half in range(2):
                    j = 4 * w + 2 * half + hh                            # byte b = 2 half + hh of word w
                    x(f"%yh{j} = vector.extract %ch{w}_{hh}[{half}] : vector<2xf16> -> f16")
                    x(f"%y{j} = scalar.extf %yh{j} : f16 to f32")
                    x(f"%w{j} = scalar.fmaf %s24, %y{j}, %nb : f32")    # s kv, exact
                    ws[j] = f"%{t}w{j}"
        return ws


def _pays(pay):
    """pay: None (IQ4_XS), one payload for both tensors, or (gate, up)."""
    if pay is None:
        pay = IQ4XS()
    return tuple(pay) if isinstance(pay, tuple) else (pay,)


def _bufname(b):
    return b[0] if isinstance(b, tuple) else b


def _loads(e, pays, kpv, t, ind):
    """Phase kpv's carried values (names, types): one payload's loads, or the gate's / up's under an scf.if on the
    workgroup-uniform tensor index (their carried types must match)."""
    _frame_loads(e, kpv, t, ind)
    if len(pays) == 1:
        got = pays[0].loads(e, kpv, t, ind)
        return [nm for nm, _ in got], [ty for _, ty in got]
    a, b = [], []
    ga = pays[0].loads(a.append, kpv, t + "g", ind + "  ", ft=t)
    ub = pays[1].loads(b.append, kpv, t + "u", ind + "  ", ft=t)
    tys = [ty for _, ty in ga]
    assert tys == [ty for _, ty in ub], ("gate and up payloads must carry the same types", tys, ub)
    names = [f"%{t}cv{i}" for i in range(len(tys))]
    e(ind + f"%{t}isg = index.cmp eq, %{t}tt, %c0 : index")
    e(f"{ind}{', '.join(names)} = scf.if %{t}isg -> ({', '.join(tys)}) {{")
    for ln in a:
        e(ln)
    e(f"{ind}  scf.yield {', '.join(nm for nm, _ in ga)} : {', '.join(tys)}")
    e(ind + "} else {")
    for ln in b:
        e(ln)
    e(f"{ind}  scf.yield {', '.join(nm for nm, _ in ub)} : {', '.join(tys)}")
    e(ind + "}")
    return names, tys


def _record(x, got, t, FMF):
    """The 9-byte record of phase %kp from 8 exact weights (the shared bfp rounding) or a payload's own (E, words);
    stored at its slab offset. Local names tagged with t."""
    if isinstance(got, tuple):   # the payload's own E (i32) and mantissa words fma(w, jq, magic) as i32
        e1, ms = got[0], list(got[1])
    else:
        ws = list(got)

        def tree(op, vs, nm):
            n = 0
            while len(vs) > 1:
                nxt = []
                for i in range(0, len(vs) - 1, 2):
                    x(f"%{t}{nm}{n} = scalar.{op}{FMF} {vs[i]}, {vs[i + 1]} : f32")
                    nxt.append(f"%{t}{nm}{n}")
                    n += 1
                if len(vs) % 2:
                    nxt.append(vs[-1])
                vs = nxt
            return vs[0]
        pm = tree("maxnumf", list(ws), "px")    # max |w| = max(max w, -min w): no fabs
        nm = tree("minnumf", list(ws), "nx")
        x(f"%{t}nn = scalar.negf {nm} : f32")
        x(f"%{t}amx = scalar.maxnumf{FMF} {pm}, %{t}nn : f32")
        x(f"%{t}amb = scalar.bitcast %{t}amx : f32 to i32")
        x(f"%{t}eb = scalar.andi %{t}amb, %iemask : i32")
        x(f"%{t}iqb = scalar.subi %ib260, %{t}eb : i32")                 # 2^(133 - E) (bits (260 - E) << 23)
        x(f"%{t}iq = scalar.bitcast %{t}iqb : i32 to f32")
        x(f"%{t}ps = scalar.mulf {pm}, %{t}iq : f32")
        x(f"%{t}bu = scalar.cmpf oge, %{t}ps, %f1275 : f32")             # roundeven(ps) > 127
        x(f"%{t}E = scalar.shrui %{t}eb, %i23 : i32")
        x(f"%{t}Ep = scalar.addi %{t}E, %i1 : i32")
        x(f"%{t}E1 = scf.select %{t}bu, %{t}Ep, %{t}E : i32")
        x(f"%{t}jqm = scalar.subi %{t}iqb, %ione : i32")
        x(f"%{t}jqb = scf.select %{t}bu, %{t}jqm, %{t}iqb : i32")
        x(f"%{t}jq = scalar.bitcast %{t}jqb : i32 to f32")
        ms = []
        for j in range(8):
            x(f"%{t}m{j} = scalar.fmaf {ws[j]}, %{t}jq, %magic : f32")
            x(f"%{t}mb{j} = scalar.bitcast %{t}m{j} : f32 to i32")
            ms.append(f"%{t}mb{j}")
        e1 = f"%{t}E1"
    words = []
    for w in range(2):
        a, b, c, d = ms[4 * w:4 * w + 4]
        for nm_, p, q in (("pa", a, b), ("pb", c, d)):
            x(f"%{t}{nm_}{w} = vector.from_elements {p}, {q} : vector<2xi32>")
            x(f"%{t}{nm_}c{w} = vector.bitcast %{t}{nm_}{w} : vector<2xi32> to vector<8xi8>")
            x(f"%{t}{nm_}s{w} = vector.shuffle<[0, 4, 0, 4, 0, 0, 0, 0]> %{t}{nm_}c{w} : vector<8xi8>")
            x(f"%{t}{nm_}w{w} = vector.bitcast %{t}{nm_}s{w} : vector<8xi8> to vector<2xi32>")
            x(f"%{t}{nm_}x{w} = vector.extract %{t}{nm_}w{w}[0] : vector<2xi32> -> i32")
        x(f"%{t}pm2_{w} = vector.from_elements %{t}pax{w}, %{t}pbx{w} : vector<2xi32>")
        x(f"%{t}pmc{w} = vector.bitcast %{t}pm2_{w} : vector<2xi32> to vector<8xi8>")
        x(f"%{t}pms{w} = vector.shuffle<[0, 1, 4, 5, 0, 0, 0, 0]> %{t}pmc{w} : vector<8xi8>")
        x(f"%{t}pmw{w} = vector.bitcast %{t}pms{w} : vector<8xi8> to vector<2xi32>")
        x(f"%{t}o{w} = vector.extract %{t}pmw{w}[0] : vector<2xi32> -> i32")
        words.append(f"%{t}o{w}")
    # the record: slab ((blk 20 + P) 4 + slab) SB + 288 w + 9 L, blk = g16 / 2, slab = 2 t + g16 % 2 of q = 2 G + p % 2
    for ln in ("%sp = index.rem %kp, %c2 : index", "%sq = index.add %G2, %sp : index",
               "%stt = index.div %sq, %c512 : index", "%sg = index.rem %sq, %c512 : index",
               "%sblk = index.div %sg, %c2 : index", "%sl0 = index.rem %sg, %c2 : index",
               "%sl1 = index.mul %stt, %c2 : index", "%sslab = index.add %sl1, %sl0 : index",
               "%s0 = index.mul %sblk, %c20 : index", "%s1 = index.add %s0, %pq : index",
               "%s2 = index.mul %s1, %c4 : index", "%s3 = index.add %s2, %sslab : index",
               "%s4 = index.mul %s3, %csb : index", "%rd = index.add %s4, %olane : index",
               "%rec = index.min %rd, %stmax : index", "%rec1 = index.add %rec, %c1 : index"):
        x(re.sub(r"%(sp|sq|stt|sg|sblk|sl0|sl1|sslab|s0|s1|s2|s3|s4|rd|rec|rec1)\b", lambda m: f"%{t}" + m.group(1), ln))
    x(f"%{t}eb8 = scalar.trunci {e1} : i32 to i8")
    x(f"%{t}ev = vector.from_elements %{t}eb8 : vector<1xi8>")
    x(f"vector.store %{t}ev, %ov[%{t}rec] : vector<1xi8>, view<[%nout]xi8>")
    x(f"%{t}mw = vector.from_elements {words[0]}, {words[1]} : vector<2xi32>")
    x(f"%{t}mv = vector.bitcast %{t}mw : vector<2xi32> to vector<8xi8>")
    x(f"vector.store %{t}mv, %ov[%{t}rec1] : vector<8xi8>, view<[%nout]xi8>")


def _decode(e, pays, carried, ind):
    """Phase %kp's record from the carried values. A (gate, up) pair decodes in two guarded blocks without results
    (each with its own rounding and store): scf.if with results has miscompiled in Loom (kresq-fusion, 10-15)."""
    FMF = "<nnan|nsz>"   # finite values; only the max's exponent bits and the 127.5 compare are read
    L = []
    x = L.append
    noif = os.environ.get("R5_NOIF") == "1"   # (diag) no %act guard: clamped workgroups rewrite workgroup 511's records
    if not noif:
        x("scf.if %act {")
    x("%pq = index.div %kp, %c2 : index")
    if len(pays) == 1:
        _record(x, pays[0].decode(x, carried, ""), "", FMF)
    else:
        x("%dsp = index.rem %kp, %c2 : index")
        x("%dsq = index.add %G2, %dsp : index")
        x("%dtt = index.div %dsq, %c512 : index")
        x("%disg = index.cmp eq, %dtt, %c0 : index")
        x("%disu = index.cmp ne, %dtt, %c0 : index")
        for k, (pay, cond) in enumerate(zip(pays, ("%disg", "%disu"))):
            tg = "gu"[k]
            x(f"scf.if {cond} {{")
            y = lambda s: x("  " + s)
            _record(y, pay.decode(y, carried, "", tg), tg, FMF)
            x("}")
    if not noif:
        x("}")
    for ln in L:
        e(ind + ln)


def fuse(text, n, nwg, lanes, pay=None):
    """The GEMM text (n rows a tensor in the full shape, nwg workgroups of lanes lanes) with the decode in its decoder
    section: buffer arguments appended to its launch (the raw weights %dz_w, any payload buffers, the stream
    %dz_out), the prologue before its K loop, phase p's record decoded from the values loaded at phase p - 1 (the
    loop carries them).
    pay: the payload format: None (IQ4XS(), byte-identical to the IQ4_XS-only build), one object for both tensors,
    or a (gate, up) pair (raw binding: gate's n rows, then up's). A payload has
      buffers: launch buffers after %dz_w (names, or (name, view type)), as %dz_<name> / %dz_<name>_na; a pair's
        union, the gate's first; its prologue makes the views;
      row_bytes: bytes a raw row;
      prologue(e, n, nin, nout): its constants, tables and views; the frame's lane map (%G2 %ww %Ll %bb %r16 %jp %whi
        %olane %act) and constants are in scope;
      loads(e, kpv, t, ind, ft=None) -> [(name, type)]: the carried values, after the frame's %<ft>pq %<ft>tt
        %<ft>g16 (ft None: t); every new name tagged with t (a pair: t = ft + "g" / ft + "u");
      decode(e, carried, ind, t="") -> 8 f32 weight names (exact), or (E i32, 8 i32 mantissa words fma(w, jq, magic))
        when the payload rounds itself; local names tagged with t.
    Names a payload's prologue defines must not clash with the other payload's in a pair."""
    assert lanes == LANES and nwg >= WGS, (lanes, nwg)
    pays = _pays(pay)
    nin, nout = n * (2 * pays[0].row_bytes if len(pays) == 1 else pays[0].row_bytes + pays[1].row_bytes), out_bytes()
    extra = []
    for p in pays:
        extra += [_bufname(b) for b in p.buffers if _bufname(b) not in extra]
    L = text.rstrip("\n").split("\n")
    assert sum("= scf.for %kp = [" in ln for ln in L) == 1
    li = next(i for i, ln in enumerate(L) if ln.startswith("} launch("))
    L[li] = L[li].replace(") {", ", %dz_w: buffer" + "".join(f", %dz_{b}: buffer" for b in extra)
                          + ", %dz_out: buffer) {")
    ai = next(i for i, ln in enumerate(L) if "buffer.assume.noalias" in ln)
    lhs, rhs = L[ai].split(" = buffer.assume.noalias ")
    names, types = rhs.split(" : ")
    L[ai] = (lhs + ", %dz_w_na" + "".join(f", %dz_{b}_na" for b in extra) + ", %dz_out_na = buffer.assume.noalias "
             + names + ", %dz_w" + "".join(f", %dz_{b}" for b in extra) + ", %dz_out : " + types
             + ", buffer" * (2 + len(extra)))
    ki = next(i for i, ln in enumerate(L) if "= scf.for %kp = [" in ln)
    P = []
    _frame_prologue(P.append, n, nin, nout)
    for p in pays:
        p.prologue(P.append, n, nin, nout)
    init, tys = _loads(P.append, pays, "%c0", "p", "  ")
    P, init = _pre(P), _pre(init)
    cur = ["%dz_ch", "%dz_cq"] if len(init) == 2 else [f"%dz_cc{i}" for i in range(len(init))]
    res = ["%dz_r0", "%dz_r1"] if len(init) == 2 else [f"%dz_rr{i}" for i in range(len(init))]
    f = L[ki]
    f = f.replace(" = scf.for %kp", "".join(", " + r for r in res) + " = scf.for %kp", 1)
    f = f.replace(") -> (", "".join(f", {c} = {i} : {ty}" for c, i, ty in zip(cur, init, tys)) + ") -> (", 1)
    f = re.sub(r"\)\s*\{\s*$", "".join(f", {ty}" for ty in tys) + ")  {", f)
    L[ki] = f
    yi = next(i for i in range(ki, len(L)) if L[i].strip().startswith("scf.yield %r0"))
    ri = next(i for i in range(ki, yi) if L[i].strip().startswith("%r0 = vector.mma"))
    NX = ["    %kp1 = index.add %kp, %c1 : index"]
    nxt, _ = _loads(NX.append, pays, "%kp1", "n", "    ")
    NX, nxt = _pre(NX), _pre(nxt)
    ops, ytypes = L[yi].split(" : ", 1)
    L[yi] = ops + ", " + ", ".join(nxt) + " : " + ytypes + "".join(f", {ty}" for ty in tys)
    b2 = [i for i in range(ki, yi) if L[i].strip().startswith("kernel.barrier")][1]
    assert L[b2 - 1].strip() == "}"
    D = []
    _decode(D.append, pays, [c.replace("%dz_", "%") for c in cur], "    ")
    D = _pre(D)
    if os.environ.get("R5_LDTOP") == "1":   # (diag) the next phase's loads at the top of the phase
        L = L[:ki] + P + L[ki:ki + 1] + NX + L[ki + 1:b2 - 1] + D + L[b2 - 1:]
    else:
        L = L[:ki] + P + L[ki:b2 - 1] + D + L[b2 - 1:ri] + NX + L[ri:]
    return "\n".join(L) + "\n"


def lane_records(n, lanes=LANES):
    """Every (lane, phase) record offset and raw load address of the kernel's index math, unclamped (numpy): the
    bounds and the coverage before any run. Returns (records, header loads, qs loads), each [NACT][PHASES]."""
    import numpy as np
    import bfp16_check as BC
    Lv = np.arange(NACT)
    Gw, tl = Lv // lanes, Lv % lanes
    ww, Ll = tl // 32, tl % 32
    bb, r16 = Ll // 16, Ll % 16
    jp, whi = ww // 2, ww % 2
    p = np.arange(PHASES)
    q = 2 * Gw[:, None] + (p % 2)[None, :]
    tt, g16 = q // 512, q % 512
    Pp = (p // 2)[None, :]
    row = 16 * g16 + r16[:, None]
    blk, slab = g16 // 2, 2 * tt + g16 % 2
    rec = ((blk * 20 + Pp) * 4 + slab) * SB + 288 * ww[:, None] + 9 * Ll[:, None]
    ha = tt * n * RBG + (n - F) * RBG + row * RBG + Pp * 136
    qa = ha + 8 + 16 * jp[:, None] + 8 * bb[:, None]
    kbv = 32 * Pp + 4 * jp[:, None] + 2 * whi[:, None] + bb[:, None]
    g8 = row // 8
    sg = 8 * (g8 // 4) + 4 * tt + g8 % 4
    off = BC.offsets("act", 2 * F, [KS], PASSES, TILE, True)
    assert np.array_equal(rec, off[sg, kbv] + 9 * (row % 8)), "record offsets differ from the stream layout"
    return rec, ha, qa
