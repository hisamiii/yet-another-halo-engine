#!/usr/bin/env python3
"""Emit the GPU encoders of the NPU attention's input streams (gen_npu_attn.py): K records, V records and the payloads.

usage: gen_npu_attn_encode.py k|v|q <out.loom> key=value ...   (the parameters of Call)

A call covers KV head g with 2 mc cores of 16 query rows each and nb bodies: the K records of keys k0 .. k0 + 32 M - 1
(M = 16 (nb / 16 - 1) - 1). Each of its four streams is a payload prefix (the call's) and records (the KV head's,
shared by every call of the head), all written in place:
  ks [2 halves][nb + 1][4608 B]    K records
  vs [2 halves][nb + 1][4608 B]    V records and a dummy (not written)
  kq [2 halves][mcf][4608 B]       payloads 0 .. mc - 1 (Q), the pad slot (mcf > mc)
  vq [2 halves][mcf][4608 B]       payloads (C, park starts, nb), the pad slot
Record and payload formats: npu_attn_check.py. Fragments are bfp16ebs8 as gen_bfp16_encode.emit_encode writes them
(m = clamp(roundeven(x 2^(133 - E)), -128, 127)).

  yah_npuattn_k  (kpool f16 [rows][1024], ptab, ks, kn): one work item per fragment, a workgroup per record (128).
                 Records j >= M are zero. kn[j] = the record's max squared key norm (of the encoded keys).
  yah_npuattn_v  (vpool f16 V^T [4][tiles][256][16], ptab, vs): one work item per fragment, a workgroup per record.
  yah_npuattn_q  (q, qnw, qeps, kn, kq, vq): a workgroup per core (64 items, one Q fragment each) and one for the pad
                 slots. q is the split-norm query of the attention (gen_attn_fa R2P_QSPLIT): f16(q w / 16)
                 [token][24][256], the raw f32 rotary dims [token][24][64] at byte B * 12288, f32 sums of q^2 per 16
                 dims [token][24][16] at B * 18432. The NPU's Q = f16 half x r log2e with r = 1 / rms, the rotary
                 dims RoPE'd as the attention does (f32 RoPE of q w, then f16(x / 16)). Row constant
                 C = 248.5 - |q| max|k| (Cauchy-Schwarz: U = S + C <= 248.5, inside the core exp's range).
Core w's rows: head heads[w % len(heads)], tokens t0 + 16 (w // len(heads)) + 0 .. 15 of the chunk (position
start_pos + token).
"""
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_bfp16_encode as BE  # noqa: E402

D, KVH, H, PAGE = 256, 4, 24, 256
REC = 4608                                               # a stream slot: 64 fragments of 72 B
LOG2E = 1.4426950408889634


@dataclass
class Call:
    g: int = 0                                           # KV head
    k0: int = 32                                         # the first key (the GPU keeps 0 .. k0 - 1: the sink at 0 makes
                                                         # bfp16 Q's rounding of S its largest error)
    mc: int = 1                                          # pairs (2 mc cores)
    nb: int = 48                                         # bodies
    pages: int = 512                                     # pool pages (rows = 256 pages)
    B: int = 2048                                        # chunk tokens (the q buffer's)
    start_pos: int = 0
    t0: int = 0
    heads: tuple = (0, 1, 2, 3, 4, 5)
    nbs: int = 0                                         # the records' half stride is nbs + 1 slots (the nb an image was
                                                         # built for: its stream views are fixed); 0: nb
    kvn: int = 0                                         # the layer form: KV heads g = 4 - kvn + g' in one dispatch
                                                         # (g unused). k / v: workgroup g' records + j, ks / vs / kn
                                                         # [kvn][...]. q (t0, heads unused; mc 16): call
                                                         # c = (3 g' + p) (B / 256) + t / 256 (tokens t, heads
                                                         # 6 (4 - kvn) + 2 (3 g' + p) + 0 / 1); kq / vq
                                                         # [calls][2][mcf][4608 B]
    qlay: int = 0                                        # q only: 0 the split-norm query (R2P_QSPLIT), 1 the raw f16
                                                         # query [token][24][256] (A4 sets: the attention does the head
                                                         # norm), normed here

    @property
    def mcf(self):
        return self.mc + (1 - self.mc % 2)

    @property
    def M(self):
        return 16 * (self.nb // 16 - 1) - 1

    def check(self):
        assert self.nb % 16 == 0 and self.nb >= 32 and 1 <= self.mc <= 16 and 0 <= self.g < KVH
        assert self.nbs == 0 or self.nbs >= self.nb
        assert self.kvn == 0 or (self.mc == 16 and self.kvn <= KVH and self.B % 256 == 0)
        assert self.k0 % 32 == 0 and self.pages * PAGE >= self.k0 + 32 * self.M, "the pool pages must hold the keys"

    @property
    def records(self):
        """a stream's records: nb + 1 K records (nb V records and a dummy)"""
        return self.nb + 1

    @property
    def slots(self):
        """a half's record slots (its stride)"""
        return (self.nbs or self.nb) + 1

    @property
    def calls(self):
        """the layer form's calls (1: one call)"""
        return 3 * self.kvn * (self.B // 256) if self.kvn else 1

    def ks_bytes(self):
        return 2 * self.slots * REC

    def vs_bytes(self):
        return 2 * self.slots * REC

    def payload_bytes(self):
        """kq or vq: the payloads and the pad slot of each half (of every call in the layer form)"""
        return self.calls * 2 * self.mcf * REC

    def kn_floats(self):
        return max(self.kvn, 1) * max(self.M, 1)

    def footprints(self, kind):
        """the bytes each binding's views span, in binding order (hal_run's min sizes)"""
        kn, nh = 4 * self.kn_floats(), max(self.kvn, 1)
        return {"k": [self.pages * PAGE * 2048, 4 * self.pages, nh * self.ks_bytes(), kn],
                "v": [KVH * self.pages * PAGE * D * 2, 4 * self.pages, nh * self.vs_bytes()],
                "q": [self.B * (12288 if self.qlay else 19968), 4 * D, 4, kn, self.payload_bytes(),
                      self.payload_bytes()]}[kind]

    def grid(self, kind):
        """(workgroups, workgroup size)"""
        nh = max(self.kvn, 1)
        return {"k": (nh * (self.nb + 1), 128), "v": (nh * self.nb, 128),
                "q": (self.calls * (2 * self.mc + (self.mcf > self.mc)), 64)}[kind]


def head(e, name, wgs, wg, args, wgid="%wg"):
    e("// GENERATED by tools/gen_npu_attn_encode.py -- edit the generator.")
    e("amdgpu.target<gfx1151> @yah_wave32 {subgroup_size = 32}")
    e("")
    e(f"kernel.def target(@yah_wave32) @{name}() {{")
    e("  %unit = index.constant 1 : index")
    e(f"  %wgs = index.constant {wgs} : index")
    e(f"  %wgsz = index.constant {wg} : index")
    e("  kernel.launch.config workgroups(%wgs, %unit, %unit) workgroup_size(%wgsz, %unit, %unit) : index")
    e(f"}} launch({', '.join(f'%{a}: buffer' for a in args)}) {{")
    e("  %base = index.constant 0 : offset")
    e(f"  {wgid} = kernel.workgroup.id<x> : index")
    e("  %tid = kernel.workitem.id<x> : index")


def layer_head(e, c, per):
    """the record kernels' layer form: workgroup = KV head g' (%hg, its head %hgg) x per + %wg"""
    if c.kvn:
        e(f"  %cper = index.constant {per} : index")
        e("  %hg = index.div %wga, %cper : index")
        e("  %wg = index.rem %wga, %cper : index")
        e(f"  %hg0 = index.constant {KVH - c.kvn} : index")
        e("  %hgg = index.add %hg, %hg0 : index")


def head_off(e, c, out, a, b, per_head):
    """out = a + b, plus g' per_head in the layer form"""
    if not c.kvn:
        e(f"  {out} = index.add {a}, {b} : index")
        return
    e(f"  {out}h = index.add {a}, {b} : index")
    e(f"  %hgb = index.constant {per_head} : index")
    e("  %hgo = index.mul %hg, %hgb : index")
    e(f"  {out} = index.add {out}h, %hgo : index")


def consts(e, *vals):
    for v in vals:
        e(f"  %c{v} = index.constant {v} : index")


def paged(e, p, unit, per):
    """the physical index of a logical unit (a K row, per = 256 a page, or a V^T tile, per = 16): ptab[u / per] per +
    u % per"""
    e(f"  %{p}lp = index.div {unit}, %c{per} : index")
    e(f"  %{p}pg0 = view.load %pt_flat[%{p}lp] : view<[%npages]xi32> -> i32")
    e(f"  %{p}pgr = index.cast %{p}pg0 : i32 to index")
    e(f"  %{p}pg = index.assume %{p}pgr [range(%{p}pgr, 0, 65535), lt(%{p}pgr, %npages)] : index")
    e(f"  %{p}pb = index.mul %{p}pg, %c{per} : index")
    e(f"  %{p}po = index.rem {unit}, %c{per} : index")
    e(f"  %{p}u = index.add %{p}pb, %{p}po : index")
    return f"%{p}u"


def gen_k(c):
    """K records: workgroup j, item f = 64 s + 2 d + kp: keys 32 j + 16 s + 8 kp + r, dims 8 d .. of head g"""
    L = []
    e = L.append
    head(e, "yah_npuattn_k", *c.grid("k"), ("kp", "ptab", "ks", "kn"), "%wga" if c.kvn else "%wg")
    layer_head(e, c, c.nb + 1)
    consts(e, 0, 2, 8, 16, 32, 64, 72, 256, 1024, REC)
    e(f"  %npages = index.constant {c.pages} : index")
    e(f"  %nkp = index.constant {c.pages * PAGE * 1024} : index")
    e(f"  %nks = index.constant {max(c.kvn, 1) * c.ks_bytes()} : index")
    e(f"  %nkn = index.constant {c.kn_floats()} : index")
    e("  %kp_na, %pt_na, %ks_na, %kn_na = buffer.assume.noalias %kp, %ptab, %ks, %kn : buffer, buffer, buffer, buffer")
    e("  %kp_flat = buffer.view %kp_na[%base] : buffer -> view<[%nkp]xf16>")
    e("  %pt_flat = buffer.view %pt_na[%base] : buffer -> view<[%npages]xi32>")
    e("  %ks_flat = buffer.view %ks_na[%base] : buffer -> view<[%nks]xi8>")
    e("  %kn_flat = buffer.view %kn_na[%base] : buffer -> view<[%nkn]xf32>")
    e("  %s = index.div %tid, %c64 : index")
    e("  %f64 = index.rem %tid, %c64 : index")
    e("  %d = index.div %f64, %c2 : index")
    e("  %kq = index.rem %f64, %c2 : index")
    e(f"  %cM = index.constant {c.M} : index")
    e("  %live = index.cmp ult, %wg, %cM : index")
    e(f"  %klast = index.constant {c.k0 + 32 * c.M - 1} : index")
    e(f"  %kfirst = index.constant {c.k0} : index")
    e("  %k0w = index.mul %wg, %c32 : index")
    e("  %k0a = index.add %k0w, %kfirst : index")
    e("  %k0b = index.mul %s, %c16 : index")
    e("  %k0c = index.mul %kq, %c8 : index")
    e("  %k0d = index.add %k0a, %k0b : index")
    e("  %k0 = index.add %k0d, %k0c : index")
    if c.kvn:
        e("  %dcol = index.mul %hgg, %c256 : index")
    else:
        e(f"  %dcol = index.constant {256 * c.g} : index")
    e("  %d8 = index.mul %d, %c8 : index")
    e("  %col = index.add %dcol, %d8 : index")
    e("  %zh = scalar.constant 0.0 : f16")
    e("  %zv = vector.splat %zh : vector<8xf16>")

    def load(r):
        e(f"  %kr{r}a = index.constant {r} : index")
        e(f"  %kr{r}b = index.add %k0, %kr{r}a : index")
        e(f"  %kr{r} = index.min %kr{r}b, %klast : index")
        row = paged(e, f"p{r}_", f"%kr{r}", PAGE)
        e(f"  %ka{r}a = index.mul {row}, %c1024 : index")
        e(f"  %ka{r} = index.add %ka{r}a, %col : index")
        e(f"  %kv{r} = vector.load %kp_flat[%ka{r}] : view<[%nkp]xf16> -> vector<8xf16>")
        e(f"  %kx{r} = scf.select %live, %kv{r}, %zv : vector<8xf16>")
        return f"%kx{r}"
    e(f"  %slots = index.constant {c.slots} : index")
    e("  %fo1 = index.mul %s, %slots : index")
    e("  %fo3 = index.add %fo1, %wg : index")
    e("  %fo4 = index.mul %fo3, %c4608 : index")
    e("  %fo5 = index.mul %f64, %c72 : index")
    head_off(e, c, "%frag", "%fo4", "%fo5", c.ks_bytes())
    sums = BE.emit_encode(e, load, "%frag", "%ks_flat", "%nks", p="e_", ind="  ", norms=True)
    # the record's max key norm: partial sums [item][8 rows] in LDS; every wave sums key (tid % 32)'s 32 partials in d
    # order and takes the max over its 32 lanes; thread 0 stores it
    e("  %lb = index.constant 4096 : offset")
    e("  %lbuf = buffer.alloca<workgroup> align(16) %lb : buffer")
    e("  %lv = buffer.view %lbuf[%base] : buffer -> view<1024xf32>")
    e("  %lt8 = index.mul %tid, %c8 : index")
    e(f"  %ls = vector.from_elements {', '.join(sums)} : vector<8xf32>")
    e("  vector.store %ls, %lv[%lt8] : vector<8xf32>, view<1024xf32>")
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e("  %tk = index.rem %tid, %c32 : index")            # key 16 s + 8 kp + r: item 64 s + 2 d + kp, row r
    e("  %ts = index.div %tk, %c16 : index")
    e("  %tr16 = index.rem %tk, %c16 : index")
    e("  %tkp = index.div %tr16, %c8 : index")
    e("  %tr = index.rem %tr16, %c8 : index")
    e("  %ti0 = index.mul %ts, %c64 : index")
    e("  %ti1 = index.add %ti0, %tkp : index")
    e("  %ti2 = index.mul %ti1, %c8 : index")
    e("  %ti = index.add %ti2, %tr : index")              # item d's partial at + 16 d
    acc = None
    for d in range(32):
        e(f"  %ta{d}c = index.constant {16 * d} : index")
        e(f"  %ta{d} = index.add %ti, %ta{d}c : index")
        e(f"  %tv{d} = view.load %lv[%ta{d}] : view<1024xf32> -> f32")
        if acc:
            e(f"  %tn{d} = scalar.addf {acc}, %tv{d} : f32")
        acc = f"%tn{d}" if acc else f"%tv{d}"
    wave_max(e, acc, "km")
    e("  %t0 = index.cmp eq, %tid, %c0 : index")
    e("  scf.if %t0 {")
    e("    scf.if %live {")
    if c.kvn:
        e(f"      %knm = index.constant {max(c.M, 1)} : index")
        e("      %knh = index.mul %hg, %knm : index")
        e("      %knw = index.add %knh, %wg : index")
    e(f"      view.store %kmmax, %kn_flat[{'%knw' if c.kvn else '%wg'}] : f32, view<[%nkn]xf32>")
    e("    }")
    e("  }")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def wave_max(e, v, p, ind="  "):
    """max of a non-negative f32 over the wave (as i32 bits) -> %{p}max in every lane"""
    e(f"{ind}%{p}w32 = scalar.constant 32 : i32")
    e(f"{ind}%{p}b0 = scalar.bitcast {v} : f32 to i32")
    cur = f"%{p}b0"
    for k, m in enumerate((16, 8, 4, 2, 1)):
        e(f"{ind}%{p}o{k} = scalar.constant {m} : i32")
        e(f"{ind}%{p}x{k}, %{p}v{k} = kernel.subgroup.shuffle<xor> {cur}, %{p}o{k}, %{p}w32 : i32, i32, i32")
        e(f"{ind}%{p}b{k + 1} = scalar.maxsi {cur}, %{p}x{k} : i32")
        cur = f"%{p}b{k + 1}"
    e(f"{ind}%{p}max = scalar.bitcast {cur} : i32 to f32")


def gen_v(c):
    """V records: workgroup b (P.V group n = b / 16, dims 16 rp .. with rp = (14 + b) % 16), item f = 2 st + dp,
    st = 4 bs + 2 s + kq over K record 16 (n - 1) - 1 + bs: row i = dim 16 rp + 8 dp + i, its 8 keys 16 s + 8 kq .."""
    L = []
    e = L.append
    head(e, "yah_npuattn_v", *c.grid("v"), ("vp", "ptab", "vs"), "%wga" if c.kvn else "%wg")
    layer_head(e, c, c.nb)
    consts(e, 2, 4, 8, 14, 16, 17, 32, 64, 72, 256, REC)
    e(f"  %npages = index.constant {c.pages} : index")
    e(f"  %nvp = index.constant {KVH * c.pages * PAGE * D} : index")
    e(f"  %nvs = index.constant {max(c.kvn, 1) * c.vs_bytes()} : index")
    e("  %vp_na, %pt_na, %vs_na = buffer.assume.noalias %vp, %ptab, %vs : buffer, buffer, buffer")
    e("  %vp_flat = buffer.view %vp_na[%base] : buffer -> view<[%nvp]xf16>")
    e("  %pt_flat = buffer.view %pt_na[%base] : buffer -> view<[%npages]xi32>")
    e("  %vs_flat = buffer.view %vs_na[%base] : buffer -> view<[%nvs]xi8>")
    e("  %n = index.div %wg, %c16 : index")
    e("  %rp0 = index.add %wg, %c14 : index")
    e("  %rp = index.rem %rp0, %c16 : index")
    e("  %st = index.div %tid, %c2 : index")
    e("  %dp = index.rem %tid, %c2 : index")
    e("  %bs = index.div %st, %c4 : index")
    e("  %st4 = index.rem %st, %c4 : index")
    e("  %s = index.div %st4, %c2 : index")
    e("  %kq = index.rem %st4, %c2 : index")
    # K record rr = 16 n + bs - 17 (live for 17 <= 16 n + bs < M + 17)
    e("  %n16 = index.mul %n, %c16 : index")
    e("  %rr17 = index.add %n16, %bs : index")
    e("  %lo = index.cmp uge, %rr17, %c17 : index")
    e(f"  %cM17 = index.constant {c.M + 17} : index")
    e("  %hi = index.cmp ult, %rr17, %cM17 : index")
    e("  %live = scalar.andi %lo, %hi : i1")
    e("  %rrc = index.max %rr17, %c17 : index")
    e("  %rr0 = index.sub %rrc, %c17 : index")
    e(f"  %Mm1 = index.constant {max(c.M - 1, 0)} : index")
    e("  %rr = index.min %rr0, %Mm1 : index")
    e(f"  %kfirst = index.constant {c.k0} : index")
    e("  %k0r = index.mul %rr, %c32 : index")
    e("  %k0a = index.add %k0r, %kfirst : index")
    e("  %k0b = index.mul %s, %c16 : index")
    e("  %k0c = index.mul %kq, %c8 : index")
    e("  %k0d = index.add %k0a, %k0b : index")
    e("  %k0 = index.add %k0d, %k0c : index")              # first key (multiple of 8)
    e("  %kt = index.div %k0, %c16 : index")
    e("  %ko = index.rem %k0, %c16 : index")
    tile = paged(e, "pt_", "%kt", PAGE // 16)
    if c.kvn:
        e(f"  %gtp = index.constant {c.pages * 16} : index")
        e("  %gt = index.mul %hgg, %gtp : index")
    else:
        e(f"  %gt = index.constant {c.g * c.pages * 16} : index")
    e("  %tt = index.add %gt, " + tile + " : index")
    e("  %tb = index.mul %tt, %c256 : index")              # (head g, tile) row 0
    e("  %rp16 = index.mul %rp, %c16 : index")
    e("  %dp8 = index.mul %dp, %c8 : index")
    e("  %dim0 = index.add %rp16, %dp8 : index")
    e("  %zh = scalar.constant 0.0 : f16")
    e("  %zv = vector.splat %zh : vector<8xf16>")

    def load(r):
        e(f"  %vd{r}c = index.constant {r} : index")
        e(f"  %vd{r} = index.add %dim0, %vd{r}c : index")
        e(f"  %va{r}a = index.add %tb, %vd{r} : index")
        e(f"  %va{r}b = index.mul %va{r}a, %c16 : index")
        e(f"  %va{r} = index.add %va{r}b, %ko : index")
        e(f"  %vv{r} = vector.load %vp_flat[%va{r}] : view<[%nvp]xf16> -> vector<8xf16>")
        e(f"  %vx{r} = scf.select %live, %vv{r}, %zv : vector<8xf16>")
        return f"%vx{r}"
    e(f"  %slots = index.constant {c.slots} : index")
    e("  %h = index.div %tid, %c64 : index")
    e("  %f64 = index.rem %tid, %c64 : index")
    e("  %fo1 = index.mul %h, %slots : index")
    e("  %fo3 = index.add %fo1, %wg : index")
    e("  %fo4 = index.mul %fo3, %c4608 : index")
    e("  %fo5 = index.mul %f64, %c72 : index")
    head_off(e, c, "%frag", "%fo4", "%fo5", c.vs_bytes())
    BE.emit_encode(e, load, "%frag", "%vs_flat", "%nvs", p="e_", ind="  ")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def gen_q(c):
    """payloads: workgroup w < 2 mc = core w (item f = 2 d + rb: rows 8 rb .., dims 8 d ..); workgroup 2 mc zeroes the
    pad slots (mcf > mc)"""
    nh = 2 if c.kvn else len(c.heads)
    if not c.kvn:
        assert (2 * c.mc) % nh == 0 and all(6 * c.g <= h < 6 * c.g + 6 for h in c.heads)
        assert c.t0 + 16 * (2 * c.mc // nh) <= c.B
    L = []
    e = L.append
    pad = c.mcf > c.mc
    head(e, "yah_npuattn_q", *c.grid("q"), ("q", "qnw", "qeps", "kn", "kq", "vq"))
    consts(e, 0, 1, 2, 8, 16, 24, 32, 64, 72, 256, REC)
    e(f"  %nq = index.constant {c.B * (12288 if c.qlay else 19968) // 2} : index")
    e(f"  %nkq = index.constant {c.payload_bytes()} : index")
    e(f"  %nvq = index.constant {c.payload_bytes()} : index")
    e(f"  %nkn = index.constant {c.kn_floats()} : index")
    e("  %q_na, %w_na, %e_na, %kn_na, %kq_na, %vq_na = buffer.assume.noalias %q, %qnw, %qeps, %kn, %kq, %vq : "
      "buffer, buffer, buffer, buffer, buffer, buffer")
    e("  %qh_flat = buffer.view %q_na[%base] : buffer -> view<[%nq]xf16>")
    if not c.qlay:
        e(f"  %nq32 = index.constant {c.B * 19968 // 4} : index")
        e("  %qf_flat = buffer.view %q_na[%base] : buffer -> view<[%nq32]xf32>")
    e("  %w_flat = buffer.view %w_na[%base] : buffer -> view<[%c256]xf32>")
    e("  %e_flat = buffer.view %e_na[%base] : buffer -> view<[%c1]xf32>")
    e("  %kn_flat = buffer.view %kn_na[%base] : buffer -> view<[%nkn]xf32>")
    e("  %kq_flat = buffer.view %kq_na[%base] : buffer -> view<[%nkq]xi8>")
    e("  %vq_flat = buffer.view %vq_na[%base] : buffer -> view<[%nvq]xi8>")
    e(f"  %ncore = index.constant {2 * c.mc} : index")
    wg = "%wg"
    if c.kvn:   # workgroup = call cl, its core (or pad) wgc
        e(f"  %cper = index.constant {2 * c.mc + pad} : index")
        e("  %cl = index.div %wg, %cper : index")
        e("  %wgc = index.rem %wg, %cper : index")
        e(f"  %ccb = index.constant {2 * c.mcf * REC} : index")
        e("  %cbase = index.mul %cl, %ccb : index")      # the call's kq / vq bytes
        wg = "%wgc"
    e(f"  %core = index.cmp ult, {wg}, %ncore : index")
    e("  scf.if %core {")
    q_core(e, c, nh, wg)
    e("  }")
    if pad:
        e(f"  %padwg = index.cmp eq, {wg}, %ncore : index")
        e("  scf.if %padwg {")
        e("    %zb = scalar.constant 0 : i8")
        e("    %z8 = vector.splat %zb : vector<8xi8>")
        imgs = (("%kq_flat", "%nkq"), ("%vq_flat", "%nvq"))
        for hv, (buf, nb_) in enumerate(imgs):
            for h in range(2):
                e(f"    %pz{hv}{h}{'c' if c.kvn else ''} = index.constant {(h * c.mcf + c.mc) * REC} : index")
                if c.kvn:
                    e(f"    %pz{hv}{h} = index.add %pz{hv}{h}c, %cbase : index")
                for k in range(REC // 8 // 64):
                    e(f"    %pzo{hv}{h}_{k}a = index.constant {8 * 64 * k} : index")
                    e(f"    %pzo{hv}{h}_{k}b = index.mul %tid, %c8 : index")
                    e(f"    %pzo{hv}{h}_{k}c = index.add %pzo{hv}{h}_{k}a, %pzo{hv}{h}_{k}b : index")
                    e(f"    %pzo{hv}{h}_{k} = index.add %pz{hv}{h}, %pzo{hv}{h}_{k}c : index")
                    e(f"    vector.store %z8, {buf}[%pzo{hv}{h}_{k}] : vector<8xi8>, view<[{nb_}]xi8>")
        e("  }")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def q_core(e, c, nh, wg):
    def x(s):
        e("    " + s)
    def based(name):   # the layer form adds the call's base to a payload byte offset
        if not c.kvn:
            return name
        x(f"{name} = index.add {name}c, %cbase : index")
        return None
    sfx = "c" if c.kvn else ""
    x(f"%cnh = index.constant {nh} : index")
    x(f"%wh = index.rem {wg}, %cnh : index")
    x(f"%wt = index.div {wg}, %cnh : index")
    if c.kvn:   # call cl = (3 g' + p) ncb + t0 / 256: heads 6 (4 - kvn) + 2 (3 g' + p) + wh
        x(f"%cncb = index.constant {c.B // 256} : index")
        x("%hpq = index.div %cl, %cncb : index")
        x("%hpq2 = index.mul %hpq, %c2 : index")
        x(f"%hpb = index.constant {6 * (KVH - c.kvn)} : index")
        x("%hpc = index.add %hpq2, %hpb : index")
        x("%hph = index.add %hpc, %wh : index")
        x(f"%hsel = index.assume %hph [range(%hph, 0, {H}), lt(%hph, %c24)] : index")
        hv = "%hsel"
        x("%ctb = index.rem %cl, %cncb : index")
        x("%ct0 = index.mul %ctb, %c256 : index")
        x(f"%cn3 = index.constant {3 * (c.B // 256)} : index")
        x("%kng = index.div %cl, %cn3 : index")          # g': its kn
        x(f"%knm = index.constant {max(c.M, 1)} : index")
        x("%knb = index.mul %kng, %knm : index")
    else:
        x(f"%hsel0 = index.constant {c.heads[0]} : index")
        hv = "%hsel0"
        for i, hh in enumerate(c.heads[1:], 1):          # head = heads[w % nh]
            x(f"%hc{i} = index.constant {i} : index")
            x(f"%hk{i} = index.constant {hh} : index")
            x(f"%hq{i} = index.cmp eq, %wh, %hc{i} : index")
            x(f"%hsel{i} = scf.select %hq{i}, %hk{i}, {hv} : index")
            hv = f"%hsel{i}"
        x(f"%ct0 = index.constant {c.t0} : index")
    x("%wt16 = index.mul %wt, %c16 : index")
    x("%tok0 = index.add %ct0, %wt16 : index")            # the core's first token
    x("%d = index.div %tid, %c2 : index")
    x("%rb = index.rem %tid, %c2 : index")
    x("%rb8 = index.mul %rb, %c8 : index")
    x("%tokb = index.add %tok0, %rb8 : index")            # the fragment's first row's token
    x("%rot = index.cmp ult, %d, %c8 : index")
    x("%d8 = index.mul %d, %c8 : index")
    x("%two = scalar.constant 2.0 : f32")
    x("%one = scalar.constant 1.0 : f32")
    x("%rdim = scalar.constant 64.0 : f32")
    x("%theta = scalar.constant 10000000.0 : f32")
    x("%c16f = scalar.constant 0.0625 : f32")
    x(f"%log2e = scalar.constant {LOG2E} : f32")
    x("%sc256 = scalar.constant 0.00390625 : f32")
    x("%eps = view.load %e_flat[%c0] : view<[%c1]xf32> -> f32")
    x(f"%csp = index.constant {c.start_pos} : index")
    # RoPE pairs of this fragment's dims (d < 8): dims 8 d + i, p = (8 d + i) % 32, partner dim p + 32 or p - 32
    x("%dlo = index.rem %d8, %c32 : index")
    x("%hi32 = index.cmp uge, %d8, %c32 : index")
    inv = []
    for i in range(8):
        x(f"%rp{i}c = index.constant {i} : index")
        x(f"%rp{i} = index.add %dlo, %rp{i}c : index")
        x(f"%rp{i}i = index.cast %rp{i} : index to i32")
        x(f"%rp{i}f = scalar.sitofp %rp{i}i : i32 to f32")
        x(f"%rp{i}2 = scalar.mulf %two, %rp{i}f : f32")
        x(f"%rp{i}ex = scalar.divf %rp{i}2, %rdim : f32")
        x(f"%rp{i}if = scalar.powf<afn> %theta, %rp{i}ex : f32")
        x(f"%rp{i}iv = scalar.divf %one, %rp{i}if : f32")
        inv.append(f"%rp{i}iv")
        x(f"%ws{i}a = index.add %dlo, %rp{i}c : index")
        x(f"%ws{i}b = index.add %ws{i}a, %c32 : index")
        x(f"%wa{i} = view.load %w_flat[%ws{i}a] : view<[%c256]xf32> -> f32")
        x(f"%wb{i} = view.load %w_flat[%ws{i}b] : view<[%c256]xf32> -> f32")

    def load(r):
        x(f"%t{r}c = index.constant {r} : index")
        x(f"%t{r} = index.add %tokb, %t{r}c : index")
        x(f"%th{r}a = index.mul %t{r}, %c24 : index")
        x(f"%th{r} = index.add %th{r}a, {hv} : index")      # token * 24 + head
        # r log2e from the 16 sums (gen_attn_fa qsplit_phase2's order)
        x(f"%sb{r} = index.mul %th{r}, %c16 : index")
        x(f"%sb{r}o = index.constant {c.B * 18432 // 4} : index")
        x(f"%sb{r}a = index.add %sb{r}, %sb{r}o : index")
        sl = []
        for k in range(4):
            x(f"%sb{r}_{k}c = index.constant {4 * k} : index")
            x(f"%sb{r}_{k} = index.add %sb{r}a, %sb{r}_{k}c : index")
            x(f"%sl{r}_{k} = vector.load %qf_flat[%sb{r}_{k}] : view<[%nq32]xf32> -> vector<4xf32>")
            sl.append(f"%sl{r}_{k}")
        x(f"%sv{r} = vector.addf {sl[0]}, {sl[1]} : vector<4xf32>")
        x(f"%sw{r} = vector.addf {sl[2]}, {sl[3]} : vector<4xf32>")
        x(f"%sa{r} = vector.addf %sv{r}, %sw{r} : vector<4xf32>")
        for k in range(4):
            x(f"%ss{r}_{k} = vector.extract %sa{r}[{k}] : vector<4xf32> -> f32")
        x(f"%s01_{r} = scalar.addf %ss{r}_0, %ss{r}_1 : f32")
        x(f"%s23_{r} = scalar.addf %ss{r}_2, %ss{r}_3 : f32")
        x(f"%stot{r} = scalar.addf %s01_{r}, %s23_{r} : f32")
        x(f"%smean{r} = scalar.mulf %stot{r}, %sc256 : f32")
        x(f"%sme{r} = scalar.addf %smean{r}, %eps : f32")
        x(f"%sinv{r} = scalar.rsqrtf %sme{r} : f32")
        x(f"%sl2e{r} = scalar.mulf %sinv{r}, %log2e : f32")
        # the stored f16 halves (dims 8 d ..)
        x(f"%qa{r}a = index.mul %th{r}, %c256 : index")
        x(f"%qa{r} = index.add %qa{r}a, %d8 : index")
        x(f"%qv{r} = vector.load %qh_flat[%qa{r}] : view<[%nq]xf16> -> vector<8xf16>")
        # the rotary dims: raw f32 q w of dims p and p + 32 (8 each), RoPE at position start_pos + token
        x(f"%rb{r}a = index.mul %th{r}, %c64 : index")
        x(f"%rb{r}o = index.constant {c.B * 12288 // 4} : index")
        x(f"%rb{r}b = index.add %rb{r}a, %rb{r}o : index")
        x(f"%rb{r} = index.add %rb{r}b, %dlo : index")
        x(f"%rb{r}h = index.add %rb{r}, %c32 : index")
        x(f"%ra{r}v = vector.load %qf_flat[%rb{r}] : view<[%nq32]xf32> -> vector<8xf32>")
        x(f"%rh{r}v = vector.load %qf_flat[%rb{r}h] : view<[%nq32]xf32> -> vector<8xf32>")
        x(f"%pos{r}a = index.add %csp, %t{r} : index")
        x(f"%pos{r}i = index.cast %pos{r}a : index to i32")
        x(f"%pos{r} = scalar.sitofp %pos{r}i : i32 to f32")
        out = []
        for i in range(8):
            x(f"%ra{r}_{i}x = vector.extract %ra{r}v[{i}] : vector<8xf32> -> f32")
            x(f"%rh{r}_{i}x = vector.extract %rh{r}v[{i}] : vector<8xf32> -> f32")
            x(f"%a{r}_{i} = scalar.mulf %ra{r}_{i}x, %wa{i} : f32")
            x(f"%b{r}_{i} = scalar.mulf %rh{r}_{i}x, %wb{i} : f32")
            x(f"%an{r}_{i} = scalar.mulf %pos{r}, {inv[i]} : f32")
            x(f"%co{r}_{i} = scalar.cosf<afn> %an{r}_{i} : f32")
            x(f"%si{r}_{i} = scalar.sinf<afn> %an{r}_{i} : f32")
            x(f"%x0{r}_{i} = scalar.mulf %a{r}_{i}, %co{r}_{i} : f32")
            x(f"%x1{r}_{i} = scalar.mulf %b{r}_{i}, %si{r}_{i} : f32")
            x(f"%lo{r}_{i} = scalar.subf %x0{r}_{i}, %x1{r}_{i} : f32")
            x(f"%y0{r}_{i} = scalar.mulf %a{r}_{i}, %si{r}_{i} : f32")
            x(f"%y1{r}_{i} = scalar.mulf %b{r}_{i}, %co{r}_{i} : f32")
            x(f"%up{r}_{i} = scalar.addf %y0{r}_{i}, %y1{r}_{i} : f32")
            x(f"%rr{r}_{i} = scf.select %hi32, %up{r}_{i}, %lo{r}_{i} : f32")
            x(f"%rs{r}_{i} = scalar.mulf %rr{r}_{i}, %c16f : f32")
            x(f"%rt{r}_{i} = scalar.fptrunc %rs{r}_{i} : f32 to f16")
            x(f"%qe{r}_{i} = vector.extract %qv{r}[{i}] : vector<8xf16> -> f16")
            x(f"%qs{r}_{i} = scf.select %rot, %rt{r}_{i}, %qe{r}_{i} : f16")
            x(f"%qx{r}_{i} = scalar.extf %qs{r}_{i} : f16 to f32")
            x(f"%qy{r}_{i} = scalar.mulf %qx{r}_{i}, %sl2e{r} : f32")
            out.append(f"%qy{r}_{i}")
        x(f"%qrow{r} = vector.from_elements {', '.join(out)} : vector<8xf32>")
        return f"%qrow{r}"
    if c.qlay:
        load = q_raw(e, c, hv, inv)
    x(f"%slots = index.constant {c.mcf} : index")
    x(f"%wl = index.rem {wg}, %c2 : index")
    x(f"%wp = index.div {wg}, %c2 : index")
    x("%fo1 = index.mul %wl, %slots : index")
    x("%fo2 = index.add %fo1, %wp : index")
    x("%fo3 = index.mul %fo2, %c4608 : index")
    x("%fo4 = index.mul %tid, %c72 : index")
    x(f"%frag{sfx} = index.add %fo3, %fo4 : index")
    based("%frag")
    sums = BE.emit_encode(e, load, "%frag", "%kq_flat", "%nkq", p="e_", ind="    ", src="f32", norms=True)
    # row norms: partials [item][8] -> row 8 rb + r sums its 32 items in d order; kmax: max of kn over the wave
    x("%lb = index.constant 2048 : offset")
    x("%lbuf = buffer.alloca<workgroup> align(16) %lb : buffer")
    x("%lv = buffer.view %lbuf[%base] : buffer -> view<512xf32>")
    x("%lt8 = index.mul %tid, %c8 : index")
    x(f"%ls = vector.from_elements {', '.join(sums)} : vector<8xf32>")
    x("vector.store %ls, %lv[%lt8] : vector<8xf32>, view<512xf32>")
    x("%lane = index.rem %tid, %c32 : index")
    acc = None
    for k in range(-(-max(c.M, 1) // 32)):              # kn[lane + 32 k] (M of them)
        x(f"%kn{k}c = index.constant {32 * k} : index")
        x(f"%kn{k}a = index.add %lane, %kn{k}c : index")
        x(f"%kn{k}m = index.constant {max(c.M, 1) - 1} : index")
        x(f"%kn{k}i = index.min %kn{k}a, %kn{k}m : index")
        if c.kvn:
            x(f"%kn{k}j = index.add %kn{k}i, %knb : index")
        x(f"%kn{k} = view.load %kn_flat[%kn{k}{'j' if c.kvn else 'i'}] : view<[%nkn]xf32> -> f32")
        if acc:
            x(f"%kx{k} = scalar.maxnumf {acc}, %kn{k} : f32")
        acc = f"%kx{k}" if acc else f"%kn{k}"
    wave_max(e, acc, "km", "    ")
    x("%kmax = scalar.sqrtf %kmmax : f32")
    x("kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    x("%isrow = index.cmp ult, %tid, %c16 : index")
    x("scf.if %isrow {")
    x("  %nrb = index.div %tid, %c8 : index")
    x("  %nr = index.rem %tid, %c8 : index")
    x("  %ni0 = index.mul %nrb, %c8 : index")
    x("  %ni = index.add %ni0, %nr : index")                 # item 2 d + rb, row r: (2 d + rb) 8 + r
    acc = None
    for d in range(32):
        x(f"  %na{d}c = index.constant {16 * d} : index")
        x(f"  %na{d} = index.add %ni, %na{d}c : index")
        x(f"  %nv{d} = view.load %lv[%na{d}] : view<512xf32> -> f32")
        if acc:
            x(f"  %nn{d} = scalar.addf {acc}, %nv{d} : f32")
        acc = f"%nn{d}" if acc else f"%nv{d}"
    x(f"  %qn = scalar.sqrtf {acc} : f32")
    x("  %qk = scalar.mulf %qn, %kmax : f32")
    x("  %cb = scalar.constant 248.5 : f32")
    x("  %cr = scalar.subf %cb, %qk : f32")
    x("  %crv = vector.splat %cr : vector<8xf32>")
    x("  %vo1 = index.mul %wl, %slots : index")
    x("  %vo2 = index.add %vo1, %wp : index")
    x(f"  %vbase{sfx} = index.mul %vo2, %c4608 : index")
    if c.kvn:
        x("  %vbase = index.add %vbasec, %cbase : index")
    x("  %c32b = index.mul %tid, %c32 : index")
    x("  %cbo = index.add %vbase, %c32b : index")
    x("  %crb = vector.bitcast %crv : vector<8xf32> to vector<32xi8>")
    x("  vector.store %crb, %vq_flat[%cbo] : vector<32xi8>, view<[%nvq]xi8>")
    x("}")
    # park starts: 2 x 8 blocks x 256 B of f32 128 (P = 1 a fake key), nb in park half 1's first word; 16 B a store
    x("%f128 = scalar.constant 128.0 : f32")
    x(f"%nbw = scalar.constant {c.nb} : i32")
    x("%nbf = scalar.bitcast %nbw : i32 to f32")
    x("%pvb2 = index.mul %wl, %slots : index")
    x("%pvb3 = index.add %pvb2, %wp : index")
    x(f"%pvb4{sfx} = index.mul %pvb3, %c4608 : index")
    based("%pvb4")
    x("%c512 = index.constant 512 : index")
    x("%pvbase = index.add %pvb4, %c512 : index")
    x("%t16 = index.mul %tid, %c16 : index")
    for k in range(4):                                   # 256 chunks of 16 B, chunk 128 starts with nb
        x(f"%pk{k}c = index.constant {1024 * k} : index")
        x(f"%pk{k}a = index.add %t16, %pk{k}c : index")
        x(f"%pk{k} = index.add %pvbase, %pk{k}a : index")
        if k == 2:
            x("%isnb = index.cmp eq, %tid, %c0 : index")
            x("%nb0 = scf.select %isnb, %nbf, %f128 : f32")
            x("%pkv2 = vector.from_elements %nb0, %f128, %f128, %f128 : vector<4xf32>")
        else:
            x(f"%pkv{k} = vector.splat %f128 : vector<4xf32>")
        x(f"%pkb{k} = vector.bitcast %pkv{k} : vector<4xf32> to vector<16xi8>")
        x(f"vector.store %pkb{k}, %vq_flat[%pk{k}] : vector<16xi8>, view<[%nvq]xi8>")


def q_raw(e, c, hv, inv):
    """qlay 1: the item's 8 rows of the raw f16 query normed as the attention does (q / rms x w, RoPE, f16(x / 16)) x
    log2e. 1 / rms per row from LDS: partials [item][8 rows] of the item's 8 dims, rows summed by threads 0 .. 15."""
    def x(s):
        e("    " + s)
    x("%xw0 = index.mul %tid, %c8 : index")
    for i in range(8):                                   # the own dims' weights (the rotary ones: %wa / %wb)
        x(f"%wo{i}c = index.constant {i} : index")
        x(f"%wo{i}a = index.add %d8, %wo{i}c : index")
        x(f"%wo{i} = view.load %w_flat[%wo{i}a] : view<[%c256]xf32> -> f32")
    x("%pb = index.constant 2112 : offset")
    x("%pbuf = buffer.alloca<workgroup> align(16) %pb : buffer")
    x("%pv = buffer.view %pbuf[%base] : buffer -> view<528xf32>")
    parts = []
    for r in range(8):
        x(f"%xt{r}c = index.constant {r} : index")
        x(f"%xt{r} = index.add %tokb, %xt{r}c : index")
        x(f"%xh{r}a = index.mul %xt{r}, %c24 : index")
        x(f"%xh{r} = index.add %xh{r}a, {hv} : index")
        x(f"%xa{r}a = index.mul %xh{r}, %c256 : index")
        x(f"%xa{r} = index.add %xa{r}a, %d8 : index")
        x(f"%xv{r} = vector.load %qh_flat[%xa{r}] : view<[%nq]xf16> -> vector<8xf16>")
        x(f"%xf{r} = vector.extf %xv{r} : vector<8xf16> to vector<8xf32>")
        x(f"%xs{r} = vector.mulf %xf{r}, %xf{r} : vector<8xf32>")
        acc = None
        for i in range(8):
            x(f"%xs{r}_{i} = vector.extract %xs{r}[{i}] : vector<8xf32> -> f32")
            if acc:
                x(f"%xp{r}_{i} = scalar.addf {acc}, %xs{r}_{i} : f32")
            acc = f"%xp{r}_{i}" if acc else f"%xs{r}_{i}"
        parts.append(acc)
    x(f"%xpv = vector.from_elements {', '.join(parts)} : vector<8xf32>")
    x("vector.store %xpv, %pv[%xw0] : vector<8xf32>, view<528xf32>")
    x("kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    x("%xrow = index.cmp ult, %tid, %c16 : index")
    x("scf.if %xrow {")                                  # row R = 8 rb + r: items 2 d + rb, entry r
    x("  %xrb = index.div %tid, %c8 : index")
    x("  %xr = index.rem %tid, %c8 : index")
    x("  %xi0 = index.mul %xrb, %c8 : index")
    x("  %xi = index.add %xi0, %xr : index")
    acc = None
    for d in range(32):
        x(f"  %xn{d}c = index.constant {16 * d} : index")
        x(f"  %xn{d}a = index.add %xi, %xn{d}c : index")
        x(f"  %xn{d} = view.load %pv[%xn{d}a] : view<528xf32> -> f32")
        if acc:
            x(f"  %xm{d} = scalar.addf {acc}, %xn{d} : f32")
        acc = f"%xm{d}" if acc else f"%xn{d}"
    x(f"  %xmean = scalar.mulf {acc}, %sc256 : f32")
    x("  %xme = scalar.addf %xmean, %eps : f32")
    x("  %xinv = scalar.rsqrtf %xme : f32")
    x("  %xo = index.constant 512 : index")
    x("  %xoa = index.add %xo, %tid : index")
    x("  view.store %xinv, %pv[%xoa] : f32, view<528xf32>")
    x("}")
    x("kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    x("%xq0 = index.constant 512 : index")
    x("%xq1 = index.add %xq0, %rb8 : index")
    for r in range(8):
        x(f"%xi{r}c = index.constant {r} : index")
        x(f"%xi{r}a = index.add %xq1, %xi{r}c : index")
        x(f"%rinv{r} = view.load %pv[%xi{r}a] : view<528xf32> -> f32")

    def load(r):
        # the rotary pair of this fragment's dims (d < 8): dims dlo + i and dlo + 32 + i
        x(f"%ya{r} = index.add %xa{r}a, %dlo : index")
        x(f"%yb{r} = index.add %ya{r}, %c32 : index")
        x(f"%yav{r} = vector.load %qh_flat[%ya{r}] : view<[%nq]xf16> -> vector<8xf16>")
        x(f"%ybv{r} = vector.load %qh_flat[%yb{r}] : view<[%nq]xf16> -> vector<8xf16>")
        x(f"%yaf{r} = vector.extf %yav{r} : vector<8xf16> to vector<8xf32>")
        x(f"%ybf{r} = vector.extf %ybv{r} : vector<8xf16> to vector<8xf32>")
        x(f"%pos{r}a = index.add %csp, %xt{r} : index")
        x(f"%pos{r}i = index.cast %pos{r}a : index to i32")
        x(f"%pos{r} = scalar.sitofp %pos{r}i : i32 to f32")
        out = []
        for i in range(8):
            x(f"%ox{r}_{i} = vector.extract %xf{r}[{i}] : vector<8xf32> -> f32")
            x(f"%on{r}_{i} = scalar.mulf %ox{r}_{i}, %rinv{r} : f32")
            x(f"%ow{r}_{i} = scalar.mulf %on{r}_{i}, %wo{i} : f32")
            x(f"%ax{r}_{i} = vector.extract %yaf{r}[{i}] : vector<8xf32> -> f32")
            x(f"%bx{r}_{i} = vector.extract %ybf{r}[{i}] : vector<8xf32> -> f32")
            x(f"%an{r}_{i}n = scalar.mulf %ax{r}_{i}, %rinv{r} : f32")
            x(f"%bn{r}_{i}n = scalar.mulf %bx{r}_{i}, %rinv{r} : f32")
            x(f"%a{r}_{i} = scalar.mulf %an{r}_{i}n, %wa{i} : f32")
            x(f"%b{r}_{i} = scalar.mulf %bn{r}_{i}n, %wb{i} : f32")
            x(f"%an{r}_{i} = scalar.mulf %pos{r}, {inv[i]} : f32")
            x(f"%co{r}_{i} = scalar.cosf<afn> %an{r}_{i} : f32")
            x(f"%si{r}_{i} = scalar.sinf<afn> %an{r}_{i} : f32")
            x(f"%x0{r}_{i} = scalar.mulf %a{r}_{i}, %co{r}_{i} : f32")
            x(f"%x1{r}_{i} = scalar.mulf %b{r}_{i}, %si{r}_{i} : f32")
            x(f"%lo{r}_{i} = scalar.subf %x0{r}_{i}, %x1{r}_{i} : f32")
            x(f"%y0{r}_{i} = scalar.mulf %a{r}_{i}, %si{r}_{i} : f32")
            x(f"%y1{r}_{i} = scalar.mulf %b{r}_{i}, %co{r}_{i} : f32")
            x(f"%up{r}_{i} = scalar.addf %y0{r}_{i}, %y1{r}_{i} : f32")
            x(f"%rr{r}_{i} = scf.select %hi32, %up{r}_{i}, %lo{r}_{i} : f32")
            x(f"%qs{r}_{i} = scf.select %rot, %rr{r}_{i}, %ow{r}_{i} : f32")
            x(f"%rs{r}_{i} = scalar.mulf %qs{r}_{i}, %c16f : f32")
            x(f"%rt{r}_{i} = scalar.fptrunc %rs{r}_{i} : f32 to f16")
            x(f"%qx{r}_{i} = scalar.extf %rt{r}_{i} : f16 to f32")
            x(f"%qy{r}_{i} = scalar.mulf %qx{r}_{i}, %log2e : f32")
            out.append(f"%qy{r}_{i}")
        x(f"%qrow{r} = vector.from_elements {', '.join(out)} : vector<8xf32>")
        return f"%qrow{r}"
    return load


def parse(argv):
    c = Call()
    for kv in argv:
        k, _, v = kv.partition("=")
        setattr(c, k, tuple(int(x) for x in v.split(",")) if k == "heads" else int(v))
    return c


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("k", "v", "q"):
        sys.exit(__doc__)
    c = parse(sys.argv[3:])
    c.check()
    with open(sys.argv[2], "w") as f:
        f.write({"k": gen_k, "v": gen_v, "q": gen_q}[sys.argv[1]](c))


if __name__ == "__main__":
    main()
