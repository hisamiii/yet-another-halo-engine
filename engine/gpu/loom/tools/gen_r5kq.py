#!/usr/bin/env python3
"""Row 5 fused decode payloads for the non-IQ4_XS gate / up formats (gen_r5fuse.fuse(..., pay=...)).
A payload decodes the slab map's unit: one row's k-block of 8 weights a K phase (row 16 g16 + r16 of the tensor,
k-block kk = 4 jp + 2 whi + bb of super-block pq), with gen_kq_bfp's exact weight math (the frame rounds to bfp16).
Each carries 2 x vector<2xi32>, the IQ4_XS payload's types, so any two can pair."""
import re

K, F = 5120, 8192
SBS = K // 256   # super-blocks a row
CARRY = "vector<2xi32>"


class IQ3S:
    """IQ3_S, 110 B blocks: d @0, qs 2 B at 2 + 2 kk, qh byte 66 + kk / 4, signs byte 74 + kk, scales 106 + kk / 8.
    Carried: (d | qs << 16, qh | signs << 8 | scales << 16) and a copy (the pair's types)."""
    blk = 110
    row_bytes = SBS * blk
    buffers = [("grid_iq3s", "view<512xi32>")]
    CONST = ("rb", "ctb", "blk", "r0", "ld2", "ld1", "c66", "c74", "c106", "gv", "kk", "oqs", "oqh", "osg", "osc",
             "rr", "i2", "i8", "i15", "i16", "i256", "hsh", "nsh")

    def __init__(self, tensor_offset=None, prefix="q3"):
        self.toff, self.p = tensor_offset, prefix   # (a pair's up: its rows after the gate's n; names prefixed)

    def prologue(self, e, n, nin, nout):
        p, rb = self.p, self.row_bytes
        base = (n - F) * rb + (self.toff or 0)
        for nm, v in (("rb", rb), ("ctb", n * rb), ("blk", self.blk), ("r0", base), ("ld2", nin - 2),
                      ("ld1", nin - 1), ("c66", 66), ("c74", 74), ("c106", 106)):
            e(f"  %{p}{nm} = index.constant {v} : index")
        e(f"  %{p}gv = buffer.view %dz_grid_iq3s_na[%base] : buffer -> view<512xi32>")
        e(f"  %{p}k0 = index.mul %jp, %c4 : index")
        e(f"  %{p}k1 = index.mul %whi, %c2 : index")
        e(f"  %{p}k2 = index.add %{p}k0, %{p}k1 : index")
        e(f"  %{p}kk = index.add %{p}k2, %bb : index")
        e(f"  %{p}kk2 = index.mul %{p}kk, %c2 : index")
        e(f"  %{p}oqs = index.add %{p}kk2, %c2 : index")
        e(f"  %{p}k4 = index.div %{p}kk, %c4 : index")
        e(f"  %{p}oqh = index.add %{p}k4, %{p}c66 : index")
        e(f"  %{p}osg = index.add %{p}kk, %{p}c74 : index")
        e(f"  %{p}k8 = index.div %{p}kk, %c8 : index")
        e(f"  %{p}osc = index.add %{p}k8, %{p}c106 : index")
        e(f"  %{p}rr = index.mul %r16, %{p}rb : index")
        for nm, v in (("i2", 2), ("i8", 8), ("i15", 15), ("i16", 16), ("i256", 256)):
            e(f"  %{p}{nm} = scalar.constant {v} : i32")
        e(f"  %{p}kq = index.rem %{p}kk, %c4 : index")
        e(f"  %{p}kqi = index.cast %{p}kq : index to i32")
        e(f"  %{p}hsh = scalar.shli %{p}kqi, %i1 : i32")           # qh bits 2 (kk % 4), + 1
        e(f"  %{p}k4h = index.rem %{p}k4, %c2 : index")
        e(f"  %{p}k4hi = index.cast %{p}k4h : index to i32")
        e(f"  %{p}nsh0 = scalar.shli %{p}k4hi, %{p}i2 : i32")      # the scale nibble 4 ((kk / 4) % 2) of byte 2
        e(f"  %{p}nsh = scalar.addi %{p}nsh0, %{p}i16 : i32")

    def loads(self, e, kpv, t, ind, ft=None):
        """After the frame's %<f>pq / %<f>tt / %<f>g16: super-block pq's bytes of the lane's row and k-block."""
        p, f = self.p, t if ft is None else ft
        x = lambda s: e(ind + s)
        if self.toff is None:   # (one payload for both tensors: up's rows n rows on)
            x(f"%{t}a0 = index.mul %{f}tt, %{p}ctb : index")
            x(f"%{t}a1 = index.add %{t}a0, %{p}r0 : index")
        else:
            x(f"%{t}a1 = index.add %{p}r0, %c0 : index")
        x(f"%{t}a2 = index.mul %{f}g16, %c16 : index")
        x(f"%{t}a3 = index.mul %{t}a2, %{p}rb : index")
        x(f"%{t}a4 = index.add %{t}a1, %{t}a3 : index")
        x(f"%{t}a5 = index.add %{t}a4, %{p}rr : index")
        x(f"%{t}a6 = index.mul %{f}pq, %{p}blk : index")
        x(f"%{t}b = index.add %{t}a5, %{t}a6 : index")       # the block
        v = []
        for nm, off, n_, lim in (("d", None, 2, "ld2"), ("qs", "oqs", 2, "ld2"), ("qh", "oqh", 1, "ld1"),
                                 ("sg", "osg", 1, "ld1"), ("sc", "osc", 1, "ld1")):
            a = f"%{t}b"
            if off is not None:
                a = f"%{t}{nm}a"
                x(f"{a} = index.add %{t}b, %{p}{off} : index")
            x(f"%{t}{nm}l = index.min {a}, %{p}{lim} : index")
            if n_ == 2:
                x(f"%{t}{nm}v = vector.load %wv[%{t}{nm}l] : view<[%nin]xi8> -> vector<2xi8>")
                x(f"%{t}{nm}w = vector.bitcast %{t}{nm}v : vector<2xi8> to vector<1xi16>")
                x(f"%{t}{nm}h = vector.extract %{t}{nm}w[0] : vector<1xi16> -> i16")
                x(f"%{t}{nm} = scalar.extui %{t}{nm}h : i16 to i32")
            else:
                x(f"%{t}{nm}b = view.load %wv[%{t}{nm}l] : view<[%nin]xi8> -> i8")
                x(f"%{t}{nm} = scalar.extui %{t}{nm}b : i8 to i32")
            v.append(f"%{t}{nm}")
        d, qs, qh, sg, sc = v
        x(f"%{t}qs16 = scalar.shli {qs}, %{p}i16 : i32")
        x(f"%{t}w0 = scalar.ori {d}, %{t}qs16 : i32")
        x(f"%{t}sg8 = scalar.shli {sg}, %{p}i8 : i32")
        x(f"%{t}sc16 = scalar.shli {sc}, %{p}i16 : i32")
        x(f"%{t}w1a = scalar.ori {qh}, %{t}sg8 : i32")
        x(f"%{t}w1 = scalar.ori %{t}w1a, %{t}sc16 : i32")
        x(f"%{t}cw = vector.from_elements %{t}w0, %{t}w1 : vector<2xi32>")
        return [(f"%{t}cw", CARRY), (f"%{t}cw", CARRY)]

    def decode(self, e, carried, ind, t=""):
        """The 8 exact weights s (-1)^sign grid of the lane's k-block (gen_kq_bfp.row_iq3s's math)."""
        p = self.p
        keep = "|".join(["c0", "i0", "i1"] + [p + c for c in self.CONST])
        local = re.compile(r"%(?!(?:" + keep + r")\b)([A-Za-z_]\w*)")
        x = lambda s: e(ind + local.sub(lambda m: "%" + t + m.group(1), s))
        cw = carried[0]
        e(ind + f"%{t}w0 = vector.extract {cw}[0] : vector<2xi32> -> i32")
        e(ind + f"%{t}w1 = vector.extract {cw}[1] : vector<2xi32> -> i32")
        x("%d16 = scalar.trunci %w0 : i32 to i16")
        x("%dh = scalar.bitcast %d16 : i16 to f16")
        x("%d = scalar.extf %dh : f16 to f32")
        x(f"%q32 = scalar.shrui %w0, %{p}i16 : i32")
        x(f"%scs = scalar.shrui %w1, %{p}nsh : i32")
        x(f"%scn = scalar.andi %scs, %{p}i15 : i32")
        x("%sc2 = scalar.shli %scn, %i1 : i32")
        x("%sc1 = scalar.addi %sc2, %i1 : i32")
        x("%scf = scalar.sitofp %sc1 : i32 to f32")
        x("%s = scalar.mulf %d, %scf : f32")                       # d (1 + 2 sc), exact
        x(f"%qhs = scalar.shrui %w1, %{p}hsh : i32")
        gws = []
        for h in range(2):
            x(f"%gl{h} = scalar.bitfield.extractu %q32 {{offset = {8 * h}, width = 8}} : i32 -> i32")
            x(f"%gh{h} = scalar.bitfield.extractu %qhs {{offset = {h}, width = 1}} : i32 -> i32")
            x(f"%gi{h} = scalar.fmai %gh{h}, %{p}i256, %gl{h} : i32")
            x(f"%gq{h} = index.cast %gi{h} : i32 to index")
            x(f"%gd{h} = index.assume %gq{h} [range(%gq{h}, 0, 511)] : index")
            x(f"%gw{h} = view.load %{p}gv[%gd{h}] : view<512xi32> -> i32")
            gws.append(f"%gw{h}")
        ws = []
        for j in range(8):
            x(f"%g{j} = scalar.bitfield.extractu {gws[j // 4]} {{offset = {8 * (j % 4)}, width = 8}} : i32 -> i32")
            x(f"%sb{j} = scalar.bitfield.extractu %w1 {{offset = {8 + j}, width = 1}} : i32 -> i32")
            x(f"%sn{j} = scalar.subi %i0, %sb{j} : i32")
            x(f"%gx{j} = scalar.xori %g{j}, %sn{j} : i32")
            x(f"%gs{j} = scalar.addi %gx{j}, %sb{j} : i32")          # (-1)^sign g
            x(f"%gf{j} = scalar.sitofp %gs{j} : i32 to f32")
            x(f"%wt{j} = scalar.mulf %s, %gf{j} : f32")               # exact (16 x 8 bits)
            ws.append(f"%{t}wt{j}")
        return ws




class _Pay:
    """Shared frame for the other formats: in emitted text, %@x is a local (tagged with t), %$x a prologue name
    (the payload prefix). Carried: (w0, w1) as vector<2xi32>, twice (the IQ4_XS payload's types)."""
    grid = None    # (fmt name, words)

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls.row_bytes = SBS * cls.blk

    def __init__(self, tensor_offset=None, prefix=None):
        self.toff, self.p = tensor_offset, prefix or self.fmt
        self.buffers = [(f"grid_{self.fmt}", f"view<{self.grid}xi32>")] if self.grid else []

    def _x(self, e, ind, t):
        p = self.p
        return lambda s: e(ind + s.replace("%@", "%" + t + "x_").replace("%$", "%" + p))

    def prologue(self, e, n, nin, nout):
        x = self._x(e, "  ", "")
        rb = self.row_bytes
        for nm, v in (("rb", rb), ("ctb", n * rb), ("blk", self.blk), ("r0", (n - F) * rb + (self.toff or 0)),
                      ("ld1", nin - 1), ("ld2", nin - 2), ("ld4", nin - 4), ("ld8", nin - 8)):
            x(f"%${nm} = index.constant {v} : index")
        if self.grid:
            x(f"%$gv = buffer.view %dz_grid_{self.fmt}_na[%base] : buffer -> view<{self.grid}xi32>")
        x("%$k0 = index.mul %jp, %c4 : index")
        x("%$k1 = index.mul %whi, %c2 : index")
        x("%$k2 = index.add %$k0, %$k1 : index")
        x("%$kk = index.add %$k2, %bb : index")      # the k-block in the super-block
        x("%$rr = index.mul %r16, %$rb : index")
        for nm, v in (("i2", 2), ("i3", 3), ("i4", 4), ("i6", 6), ("i7", 7), ("i8", 8), ("i9", 9), ("i12", 12),
                      ("i14", 14), ("i15", 15), ("i16", 16), ("i24", 24), ("i28", 28), ("i32", 32), ("i63", 63),
                      ("i127", 127), ("i128", 128), ("i255", 255), ("i256", 256), ("i511", 511), ("i7fff", 0x7FFF),
                      ("ihi", -65536), ("m01", 0x01010101), ("m03", 0x03030303)):
            x(f"%${nm} = scalar.constant {v} : i32")
        for nm, v in (("f0p25", 0.25), ("f0p125", 0.125), ("fm128", -128.0)):
            x(f"%${nm} = scalar.constant {v} : f32")
        self.consts(x)

    def ci(self, x, nm, v):
        x(f"%${nm} = index.constant {v} : index")

    def toi32(self, x, nm, src):
        x(f"%${nm} = index.cast {src} : index to i32")

    def loads(self, e, kpv, t, ind, ft=None):
        f = t if ft is None else ft
        x = self._x(e, ind, t)
        if self.toff is None:
            x(f"%@a0 = index.mul %{f}tt, %$ctb : index")
            x("%@a1 = index.add %@a0, %$r0 : index")
        else:
            x("%@a1 = index.add %$r0, %c0 : index")
        x(f"%@a2 = index.mul %{f}g16, %c16 : index")
        x("%@a3 = index.mul %@a2, %$rb : index")
        x("%@a4 = index.add %@a1, %@a3 : index")
        x("%@a5 = index.add %@a4, %$rr : index")
        x(f"%@a6 = index.mul %{f}pq, %$blk : index")
        x("%@b = index.add %@a5, %@a6 : index")
        w0, w1 = self.load(x)
        x(f"%@cw = vector.from_elements {w0}, {w1} : vector<2xi32>")
        return [(f"%{t}x_cw", CARRY), (f"%{t}x_cw", CARRY)]

    def ld(self, x, nm, off, nbytes):
        """The u8 / u16 / u32 at block + off (an index name, or None) -> %@<nm> (i32); 8 B -> %@<nm>0, %@<nm>1."""
        a = "%@b"
        if off is not None:
            a = f"%@{nm}a"
            x(f"{a} = index.add %@b, {off} : index")
        x(f"%@{nm}l = index.min {a}, %$ld{nbytes} : index")
        if nbytes == 1:
            x(f"%@{nm}q = view.load %wv[%@{nm}l] : view<[%nin]xi8> -> i8")
            x(f"%@{nm} = scalar.extui %@{nm}q : i8 to i32")
            return
        x(f"%@{nm}v = vector.load %wv[%@{nm}l] : view<[%nin]xi8> -> vector<{nbytes}xi8>")
        if nbytes == 2:
            x(f"%@{nm}w = vector.bitcast %@{nm}v : vector<2xi8> to vector<1xi16>")
            x(f"%@{nm}h = vector.extract %@{nm}w[0] : vector<1xi16> -> i16")
            x(f"%@{nm} = scalar.extui %@{nm}h : i16 to i32")
        elif nbytes == 4:
            x(f"%@{nm}w = vector.bitcast %@{nm}v : vector<4xi8> to vector<1xi32>")
            x(f"%@{nm} = vector.extract %@{nm}w[0] : vector<1xi32> -> i32")
        else:
            x(f"%@{nm}w = vector.bitcast %@{nm}v : vector<8xi8> to vector<2xi32>")
            for h in range(2):
                x(f"%@{nm}{h} = vector.extract %@{nm}w[{h}] : vector<2xi32> -> i32")

    def bits4(self, x, nm, src, sh):
        """Bit sh of each byte of word src -> 4 bits (%@<nm>)."""
        x(f"%@{nm}_s = scalar.shrui {src}, {sh} : i32")
        x(f"%@{nm}_m = scalar.andi %@{nm}_s, %$m01 : i32")
        x(f"%@{nm}_y0 = scalar.shrui %@{nm}_m, %$i7 : i32")
        x(f"%@{nm}_y = scalar.ori %@{nm}_m, %@{nm}_y0 : i32")
        x(f"%@{nm}_z0 = scalar.shrui %@{nm}_y, %$i14 : i32")
        x(f"%@{nm}_z = scalar.ori %@{nm}_y, %@{nm}_z0 : i32")
        x(f"%@{nm} = scalar.andi %@{nm}_z, %$i15 : i32")

    def f16(self, x, nm, w, hi):
        if hi:
            x(f"%@{nm}s = scalar.shrui {w}, %$i16 : i32")
            w = f"%@{nm}s"
        x(f"%@{nm}t = scalar.trunci {w} : i32 to i16")
        x(f"%@{nm}h = scalar.bitcast %@{nm}t : i16 to f16")
        x(f"%@{nm} = scalar.extf %@{nm}h : f16 to f32")

    def decode(self, e, carried, ind, t=""):
        x = self._x(e, ind, t)
        cw = carried[0]
        x(f"%@w0 = vector.extract {cw}[0] : vector<2xi32> -> i32")
        x(f"%@w1 = vector.extract {cw}[1] : vector<2xi32> -> i32")
        return [f"%{t}x_{w[2:]}" for w in self.weights(x)]

    def signed_grid(self, x, s, gws, sgn):
        """w_j = s (-1)^(bit j of sgn) grid byte j (gws: the 2 grid words), exact."""
        ws = []
        for j in range(8):
            x(f"%@g{j} = scalar.bitfield.extractu {gws[j // 4]} {{offset = {8 * (j % 4)}, width = 8}} : i32 -> i32")
            x(f"%@sb{j} = scalar.bitfield.extractu {sgn} {{offset = {j}, width = 1}} : i32 -> i32")
            x(f"%@sn{j} = scalar.subi %i0, %@sb{j} : i32")
            x(f"%@gx{j} = scalar.xori %@g{j}, %@sn{j} : i32")
            x(f"%@gs{j} = scalar.addi %@gx{j}, %@sb{j} : i32")
            x(f"%@gf{j} = scalar.sitofp %@gs{j} : i32 to f32")
            x(f"%@wt{j} = scalar.mulf {s}, %@gf{j} : f32")
            ws.append(f"%@wt{j}")
        return ws

    def gload(self, x, nm, idx):
        x(f"%@{nm}x = index.cast {idx} : i32 to index")
        x(f"%@{nm}d = index.assume %@{nm}x [range(%@{nm}x, 0, {self.grid - 1})] : index")
        x(f"%@{nm} = view.load %$gv[%@{nm}d] : view<{self.grid}xi32> -> i32")

    def scale(self, x, scn, fold):
        """s = d (1 + 2 scn) fold (exact); d the f16 low half of w0."""
        self.f16(x, "d", "%@w0", False)
        x(f"%@sc2 = scalar.shli {scn}, %i1 : i32")
        x("%@sc1 = scalar.addi %@sc2, %i1 : i32")
        x("%@scf = scalar.sitofp %@sc1 : i32 to f32")
        x("%@s0 = scalar.mulf %@d, %@scf : f32")
        if fold is None:
            return "%@s0"
        x(f"%@s = scalar.mulf %@s0, {fold} : f32")
        return "%@s"

    def ksign(self, x, f):
        x(f"%@kpa = scalar.shrui {f}, %$i4 : i32")
        x(f"%@kpb = scalar.xori {f}, %@kpa : i32")
        x("%@kpc = scalar.shrui %@kpb, %$i2 : i32")
        x("%@kpd = scalar.xori %@kpb, %@kpc : i32")
        x("%@kpe = scalar.shrui %@kpd, %i1 : i32")
        x("%@kpf = scalar.xori %@kpd, %@kpe : i32")
        x("%@kpg = scalar.andi %@kpf, %i1 : i32")
        x("%@kph = scalar.shli %@kpg, %$i7 : i32")
        x(f"%@ks = scalar.ori {f}, %@kph : i32")
        return "%@ks"

    def kq(self, x):
        """%$kq = kk % 4, %$k4 = kk / 4, %$k2h = (kk / 2) % 2 and their i32 forms."""
        x("%$kq = index.rem %$kk, %c4 : index")
        x("%$k4 = index.div %$kk, %c4 : index")
        x("%$kd2 = index.div %$kk, %c2 : index")
        x("%$k2h = index.rem %$kd2, %c2 : index")
        x("%$kqi = index.cast %$kq : index to i32")
        x("%$k2hi = index.cast %$k2h : index to i32")
        x("%$nsh = scalar.shli %$k2hi, %$i2 : i32")   # scale nibble 4 ((kk / 2) % 2)
        x("%$kk2 = index.mul %$kk, %c2 : index")
        x("%$o2kk = index.add %$kk2, %c2 : index")       # u16 at 2 + 2 kk


class IQ3XXS(_Pay):
    """98 B: d, q u16 at 2 + 2 kk (two grid indices), aux u32 at 66 + 4 (kk / 4). Carried (d | q << 16, aux)."""
    fmt, blk, grid = "iq3xxs", 98, 256

    def consts(self, x):
        self.kq(x)
        self.ci(x, "c66", 66)
        x("%$k44 = index.mul %$k4, %c4 : index")
        x("%$oax = index.add %$k44, %$c66 : index")
        x("%$fsh = scalar.muli %$kqi, %$i7 : i32")

    def load(self, x):
        self.ld(x, "d", None, 2)
        self.ld(x, "q", "%$o2kk", 2)
        self.ld(x, "ax", "%$oax", 4)
        x("%@q16 = scalar.shli %@q, %$i16 : i32")
        x("%@w0 = scalar.ori %@d, %@q16 : i32")
        return "%@w0", "%@ax"

    def weights(self, x):
        x("%@scn = scalar.shrui %@w1, %$i28 : i32")
        x("%@fs = scalar.shrui %@w1, %$fsh : i32")
        x("%@f = scalar.andi %@fs, %$i127 : i32")
        sg = self.ksign(x, "%@f")
        x("%@q = scalar.shrui %@w0, %$i16 : i32")
        x("%@gi0 = scalar.andi %@q, %$i255 : i32")
        x("%@gi1 = scalar.shrui %@q, %$i8 : i32")
        self.gload(x, "gwa", "%@gi0")
        self.gload(x, "gwb", "%@gi1")
        return self.signed_grid(x, self.scale(x, "%@scn", "%$f0p25"), ["%@gwa", "%@gwb"], sg)


class IQ2XS(_Pay):
    """74 B: d, q u16 at 2 + 2 kk (entry q & 511, sign field q >> 9), scale byte 66 + kk / 4. Carried (d | q << 16, sc)."""
    fmt, blk, grid = "iq2xs", 74, 1024

    def consts(self, x):
        self.kq(x)
        self.ci(x, "c66", 66)
        x("%$osc = index.add %$k4, %$c66 : index")

    def load(self, x):
        self.ld(x, "d", None, 2)
        self.ld(x, "q", "%$o2kk", 2)
        self.ld(x, "sc", "%$osc", 1)
        x("%@q16 = scalar.shli %@q, %$i16 : i32")
        x("%@w0 = scalar.ori %@d, %@q16 : i32")
        return "%@w0", "%@sc"

    def weights(self, x):
        x("%@q = scalar.shrui %@w0, %$i16 : i32")
        x("%@gi = scalar.andi %@q, %$i511 : i32")
        x("%@f = scalar.shrui %@q, %$i9 : i32")
        sg = self.ksign(x, "%@f")
        x("%@scs = scalar.shrui %@w1, %$nsh : i32")
        x("%@scn = scalar.andi %@scs, %$i15 : i32")
        x("%@gw2 = scalar.shli %@gi, %i1 : i32")
        x("%@gw3 = scalar.addi %@gw2, %i1 : i32")
        self.gload(x, "gwa", "%@gw2")
        self.gload(x, "gwb", "%@gw3")
        return self.signed_grid(x, self.scale(x, "%@scn", "%$f0p125"), ["%@gwa", "%@gwb"], sg)


class IQ2S(_Pay):
    """82 B: d, qs byte 2 + kk, sign byte 34 + kk, qh byte 66 + kk / 4 (crumb 2 (kk % 4)), scale byte 74 + kk / 4.
    Carried (d | qs << 16 | qh << 24, sg | sc << 8)."""
    fmt, blk, grid = "iq2s", 82, 2048

    def consts(self, x):
        self.kq(x)
        for nm, v in (("c34", 34), ("c66", 66), ("c74", 74)):
            self.ci(x, nm, v)
        x("%$oqs = index.add %$kk, %c2 : index")
        x("%$osg = index.add %$kk, %$c34 : index")
        x("%$oqh = index.add %$k4, %$c66 : index")
        x("%$osc = index.add %$k4, %$c74 : index")
        x("%$hsh = scalar.shli %$kqi, %i1 : i32")

    def load(self, x):
        self.ld(x, "d", None, 2)
        for nm, o in (("qs", "%$oqs"), ("qh", "%$oqh"), ("sg", "%$osg"), ("sc", "%$osc")):
            self.ld(x, nm, o, 1)
        x("%@qs16 = scalar.shli %@qs, %$i16 : i32")
        x("%@qh24 = scalar.shli %@qh, %$i24 : i32")
        x("%@w0a = scalar.ori %@d, %@qs16 : i32")
        x("%@w0 = scalar.ori %@w0a, %@qh24 : i32")
        x("%@sc8 = scalar.shli %@sc, %$i8 : i32")
        x("%@w1 = scalar.ori %@sg, %@sc8 : i32")
        return "%@w0", "%@w1"

    def weights(self, x):
        x("%@qs = scalar.bitfield.extractu %@w0 {offset = 16, width = 8} : i32 -> i32")
        x("%@qh = scalar.shrui %@w0, %$i24 : i32")
        x("%@qhs = scalar.shrui %@qh, %$hsh : i32")
        x("%@qh2 = scalar.andi %@qhs, %$i3 : i32")
        x("%@gi = scalar.fmai %@qh2, %$i256, %@qs : i32")
        x("%@scb = scalar.shrui %@w1, %$i8 : i32")
        x("%@scs = scalar.shrui %@scb, %$nsh : i32")
        x("%@scn = scalar.andi %@scs, %$i15 : i32")
        x("%@gw2 = scalar.shli %@gi, %i1 : i32")
        x("%@gw3 = scalar.addi %@gw2, %i1 : i32")
        self.gload(x, "gwa", "%@gw2")
        self.gload(x, "gwb", "%@gw3")
        return self.signed_grid(x, self.scale(x, "%@scn", "%$f0p125"), ["%@gwa", "%@gwb"], "%@w1")


class Q3K(_Pay):
    """110 B: hmask 8 B at 8 (kk % 4) (bit hbs), qs 8 B at 32 + 32 (kk / 16) + 8 (kk % 4) (crumb qcs), sub-block
    i = kk / 2: scale bytes 96 + i % 8 (nibble 4 (i / 8)), 104 + i % 4 (crumb 2 (i / 4)), d @108.
    Carried (d | sl << 16 | sh << 24, 8 crumbs | 8 hbits << 16)."""
    fmt, blk = "q3k", 110

    def consts(self, x):
        for nm, v in (("c96", 96), ("c104", 104), ("c108", 108), ("c16i", 16), ("c32i", 32), ("c8i", 8)):
            self.ci(x, nm, v)
        x("%$kq = index.rem %$kk, %c4 : index")
        x("%$kh = index.div %$kk, %$c16i : index")
        x("%$k16 = index.rem %$kk, %$c16i : index")
        x("%$kc = index.div %$k16, %c4 : index")
        x("%$si = index.div %$kk, %c2 : index")
        x("%$si8 = index.rem %$si, %$c8i : index")
        x("%$siq = index.div %$si, %$c8i : index")
        x("%$si4 = index.rem %$si, %c4 : index")
        x("%$sid = index.div %$si, %c4 : index")
        x("%$hmo = index.mul %$kq, %$c8i : index")
        x("%$kho = index.mul %$kh, %$c32i : index")
        x("%$qs0 = index.add %$kho, %$hmo : index")
        x("%$qso = index.add %$qs0, %$c32i : index")
        x("%$slo = index.add %$si8, %$c96 : index")
        x("%$sho = index.add %$si4, %$c104 : index")
        for nm in ("kh", "kc", "siq", "sid"):
            self.toi32(x, nm + "_i", f"%${nm}")
        x("%$kh4 = scalar.shli %$kh_i, %$i2 : i32")
        x("%$hbs = scalar.addi %$kh4, %$kc_i : i32")
        x("%$qcs = scalar.shli %$kc_i, %i1 : i32")
        x("%$lsh = scalar.shli %$siq_i, %$i2 : i32")
        x("%$hsh2 = scalar.shli %$sid_i, %i1 : i32")

    def load(self, x):
        self.ld(x, "d", "%$c108", 2)
        self.ld(x, "sl", "%$slo", 1)
        self.ld(x, "sh", "%$sho", 1)
        self.ld(x, "hm", "%$hmo", 8)
        self.ld(x, "qs", "%$qso", 8)
        x("%@sl16 = scalar.shli %@sl, %$i16 : i32")
        x("%@sh24 = scalar.shli %@sh, %$i24 : i32")
        x("%@w0a = scalar.ori %@d, %@sl16 : i32")
        x("%@w0 = scalar.ori %@w0a, %@sh24 : i32")
        cr = []
        for h in range(2):   # crumb qcs of each byte -> 8 bits
            x(f"%@qy{h} = scalar.shrui %@qs{h}, %$qcs : i32")
            x(f"%@qm{h} = scalar.andi %@qy{h}, %$m03 : i32")
            x(f"%@qa{h} = scalar.shrui %@qm{h}, %$i6 : i32")
            x(f"%@qb{h} = scalar.ori %@qm{h}, %@qa{h} : i32")
            x(f"%@qc{h} = scalar.shrui %@qb{h}, %$i12 : i32")
            x(f"%@qd{h} = scalar.ori %@qb{h}, %@qc{h} : i32")
            x(f"%@qe{h} = scalar.andi %@qd{h}, %$i255 : i32")
            self.bits4(x, f"hb{h}", f"%@hm{h}", "%$hbs")
            cr.append(h)
        x("%@qe1s = scalar.shli %@qe1, %$i8 : i32")
        x("%@hb1s = scalar.shli %@hb1, %$i4 : i32")
        x("%@cq = scalar.ori %@qe0, %@qe1s : i32")
        x("%@hh = scalar.ori %@hb0, %@hb1s : i32")
        x("%@hh16 = scalar.shli %@hh, %$i16 : i32")
        x("%@w1 = scalar.ori %@cq, %@hh16 : i32")
        return "%@w0", "%@w1"

    def weights(self, x):
        self.f16(x, "d", "%@w0", False)
        x("%@sl = scalar.bitfield.extractu %@w0 {offset = 16, width = 8} : i32 -> i32")
        x("%@sh = scalar.shrui %@w0, %$i24 : i32")
        x("%@sls = scalar.shrui %@sl, %$lsh : i32")
        x("%@sln = scalar.andi %@sls, %$i15 : i32")
        x("%@shs = scalar.shrui %@sh, %$hsh2 : i32")
        x("%@shn = scalar.andi %@shs, %$i3 : i32")
        x("%@scv = scalar.fmai %@shn, %$i16, %@sln : i32")
        x("%@scm = scalar.subi %@scv, %$i32 : i32")
        x("%@scf = scalar.sitofp %@scm : i32 to f32")
        x("%@s = scalar.mulf %@d, %@scf : f32")
        ws = []
        for j in range(8):
            x(f"%@cr{j} = scalar.bitfield.extractu %@w1 {{offset = {2 * j}, width = 2}} : i32 -> i32")
            x(f"%@hb{j} = scalar.bitfield.extractu %@w1 {{offset = {16 + j}, width = 1}} : i32 -> i32")
            x(f"%@qq{j} = scalar.fmai %@hb{j}, %$i4, %@cr{j} : i32")
            x(f"%@qm{j} = scalar.subi %@qq{j}, %$i4 : i32")
            x(f"%@qf{j} = scalar.sitofp %@qm{j} : i32 to f32")
            x(f"%@wv{j} = scalar.mulf %@s, %@qf{j} : f32")
            ws.append(f"%@wv{j}")
        return ws


class Q4K(_Pay):
    """144 B (Q5_K 176 B): d | dmin u32 @0, scale bytes 4 + s % 4, 8 + s % 4, 12 + s % 4 of sub-block s = kk / 4;
    qs 8 B at QS + 32 (s / 2) + 8 (kk % 4) (nibble 4 (s % 2)); Q5_K qh 8 B at 16 + 8 (kk % 4) (bit s).
    Carried (d | dmin << 16, b1 | b2 << 8 | b3 << 16 | qh8 << 24, qs0, qs1). gen_kq_bfp.row's math (GF order)."""
    fmt, blk, qs = "q4k", 144, 16

    def consts(self, x):
        for nm, v in (("c4i", 4), ("c8i", 8), ("c12i", 12), ("c16i", 16), ("c32i", 32), ("cqs", self.qs)):
            self.ci(x, nm, v)
        x("%$s = index.div %$kk, %c4 : index")
        x("%$kq = index.rem %$kk, %c4 : index")
        x("%$sj = index.div %$s, %c2 : index")
        x("%$shf = index.rem %$s, %c2 : index")
        x("%$sa = index.rem %$s, %c4 : index")
        x("%$sg = index.div %$s, %c4 : index")
        x("%$sjo = index.mul %$sj, %$c32i : index")
        x("%$kqo = index.mul %$kq, %$c8i : index")
        x("%$qs0 = index.add %$sjo, %$kqo : index")
        x("%$qso = index.add %$qs0, %$cqs : index")
        x("%$qho = index.add %$kqo, %$c16i : index")
        x("%$b1o = index.add %$sa, %$c4i : index")
        x("%$b2o = index.add %$sa, %$c8i : index")
        x("%$b3o = index.add %$sa, %$c12i : index")
        for nm in ("s", "shf", "sg"):
            self.toi32(x, nm + "_i", f"%${nm}")
        x("%$nsh = scalar.shli %$shf_i, %$i2 : i32")
        x("%$selm = scalar.subi %i0, %$sg_i : i32")

    def load(self, x):
        self.ld(x, "dd", None, 4)
        for h in (1, 2, 3):
            self.ld(x, f"b{h}", f"%$b{h}o", 1)
        self.ld(x, "qs", "%$qso", 8)
        x("%@b2s = scalar.shli %@b2, %$i8 : i32")
        x("%@b3s = scalar.shli %@b3, %$i16 : i32")
        x("%@w1a = scalar.ori %@b1, %@b2s : i32")
        x("%@w1b = scalar.ori %@w1a, %@b3s : i32")
        if self.fmt == "q5k":
            self.ld(x, "qh", "%$qho", 8)
            self.bits4(x, "hb0", "%@qh0", "%$s_i")
            self.bits4(x, "hb1", "%@qh1", "%$s_i")
            x("%@hb1s = scalar.shli %@hb1, %$i4 : i32")
            x("%@hh = scalar.ori %@hb0, %@hb1s : i32")
            x("%@hh24 = scalar.shli %@hh, %$i24 : i32")
            x("%@w1 = scalar.ori %@w1b, %@hh24 : i32")
        else:
            x("%@w1 = scalar.ori %@w1b, %c0i32 : i32".replace("%c0i32", "%i0"))
        self._qs = ("%@qs0", "%@qs1")
        return "%@dd", "%@w1"

    def loads(self, e, kpv, t, ind, ft=None):
        got = _Pay.loads(self, e, kpv, t, ind, ft)
        x = self._x(e, ind, t)
        x("%@cq = vector.from_elements %@qs0, %@qs1 : vector<2xi32>")
        return [got[0], (f"%{t}x_cq", CARRY)]

    def decode(self, e, carried, ind, t=""):
        self._cq = carried[1]
        return _Pay.decode(self, e, carried, ind, t)

    def rne(self, x, n, v):
        x(f"%@{n}_u = scalar.bitcast {v} : f32 to i32")
        x(f"%@{n}_l = scalar.shrui %@{n}_u, %$i16 : i32")
        x(f"%@{n}_o = scalar.andi %@{n}_l, %i1 : i32")
        x(f"%@{n}_a = scalar.addi %@{n}_u, %$i7fff : i32")
        x(f"%@{n}_b = scalar.addi %@{n}_a, %@{n}_o : i32")
        x(f"%@{n}_c = scalar.andi %@{n}_b, %$ihi : i32")
        x(f"%@{n} = scalar.bitcast %@{n}_c : i32 to f32")

    def parts2(self, x, n, v):
        self.rne(x, f"{n}0", v)
        x(f"%@{n}r = scalar.subf {v}, %@{n}0 : f32")
        self.rne(x, f"{n}1", f"%@{n}r")

    def weights(self, x):
        self.f16(x, "d", "%@w0", False)
        self.f16(x, "dm", "%@w0", True)
        for h in (1, 2, 3):
            x(f"%@b{h} = scalar.bitfield.extractu %@w1 {{offset = {8 * (h - 1)}, width = 8}} : i32 -> i32")
        x("%@scl = scalar.andi %@b1, %$i63 : i32")
        x("%@mnl = scalar.andi %@b2, %$i63 : i32")
        x("%@t1 = scalar.shrui %@b1, %$i6 : i32")
        x("%@t1s = scalar.shli %@t1, %$i4 : i32")
        x("%@b3l = scalar.andi %@b3, %$i15 : i32")
        x("%@sch = scalar.ori %@b3l, %@t1s : i32")
        x("%@t2 = scalar.shrui %@b2, %$i6 : i32")
        x("%@t2s = scalar.shli %@t2, %$i4 : i32")
        x("%@b3h = scalar.shrui %@b3, %$i4 : i32")
        x("%@mnh = scalar.ori %@b3h, %@t2s : i32")
        for v in ("sc", "mn"):
            x(f"%@{v}x = scalar.xori %@{v}l, %@{v}h : i32")
            x(f"%@{v}y = scalar.andi %@{v}x, %$selm : i32")
            x(f"%@{v} = scalar.xori %@{v}l, %@{v}y : i32")
        x("%@scf = scalar.sitofp %@sc : i32 to f32")
        x("%@mnf = scalar.sitofp %@mn : i32 to f32")
        x("%@T = scalar.mulf %@d, %@scf : f32")
        x("%@Mp = scalar.mulf %@dm, %@mnf : f32")
        x("%@M = scalar.negf %@Mp : f32")
        self.parts2(x, "Tp", "%@T")
        self.parts2(x, "Mq", "%@M")
        for h in range(2):
            x(f"%@qx{h} = vector.extract {self._cq}[{h}] : vector<2xi32> -> i32")
            x(f"%@qy{h} = scalar.shrui %@qx{h}, %$nsh : i32")
        x("%@wz0 = scalar.addf %@Mq0, %@Mq1 : f32")
        x("%@wz1 = scalar.fmaf %@Tp0, %$fm128, %@wz0 : f32")
        x("%@wzb = scalar.fmaf %@Tp1, %$fm128, %@wz1 : f32")
        ws = []
        for j in range(8):
            x(f"%@q{j} = scalar.bitfield.extractu %@qy{j // 4} {{offset = {8 * (j % 4)}, width = 4}} : i32 -> i32")
            q = f"%@q{j}"
            if self.fmt == "q5k":
                x(f"%@h{j} = scalar.bitfield.extractu %@w1 {{offset = {24 + j}, width = 1}} : i32 -> i32")
                x(f"%@qh{j} = scalar.fmai %@h{j}, %$i16, %@q{j} : i32")
                q = f"%@qh{j}"
            x(f"%@xi{j} = scalar.addi {q}, %$i128 : i32")
            x(f"%@X{j} = scalar.sitofp %@xi{j} : i32 to f32")
            x(f"%@wt{j} = scalar.fmaf %@Tp0, %@X{j}, %@wzb : f32")
            x(f"%@wv{j} = scalar.fmaf %@Tp1, %@X{j}, %@wt{j} : f32")
            ws.append(f"%@wv{j}")
        return ws


class Q5K(Q4K):
    fmt, blk, qs = "q5k", 176, 48


PAYLOADS = {"iq3s": IQ3S, "iq3xxs": IQ3XXS, "iq2xs": IQ2XS, "iq2s": IQ2S, "q3k": Q3K, "q4k": Q4K, "q5k": Q5K}
