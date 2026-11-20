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


PAYLOADS = {"iq3s": IQ3S}
