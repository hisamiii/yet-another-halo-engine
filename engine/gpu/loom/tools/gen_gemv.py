#!/usr/bin/env python3
"""Single-token decode GEMV kernels, one per (kind, weight formats, M, K).

    y[m] = sum_k W[m, k] * x[k]          W in MQ4G256 (or Q8_0), x f32

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
Bindings: weights (one per format in fmts), then x, then y.
"""
import gen_rot
import re
import sys

# fmt: (elements per block, bytes per block)
FMT = {"q8_0": (32, 34), "mq4": (256, 136)}
# Loom read-ahead depth of the sub-block loop: 2 is the fastest on the full decode (none and 3 are slower).
PIPE = 2
GGML = {8: "q8_0", 44: "mq4"}


def rows_per_wg(R=2, W=4):
    """Output rows per workgroup of a gen() / gen_bands() kernel: launch exactly M / rows_per_wg workgroups.

    Loom takes workgroup.id < grid from the launch config and may drop the row clamps: a larger grid reads out of bounds.
    """
    return R * W


def row_bytes(fmt, K):
    qk, bb = FMT[fmt]
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

def gen(kind, fmts, M, K, R=2, W=4, name="yah_gemv", _e=None, _parts=None, rot=False, peel=0, amax=False):
    """rot: y feeds an MQ4 GEMV, so rotate it in place (gen_rot.gen in place, f32: y / awq, signs, H256): the last
    workgroup to finish each 256-row block (its monotonic counter, bindings awq, signs, cnt after y) rotates it.
    amax (plain): also the argmax of y into a token (bindings ak: u64 key + i32 counter, tok: i32 after y; amax_tail).
    peel: the first peel K iterations run before the loop, their weight loads and decode (which do not read x) ahead
    of a "// chain.wait" marker (gen_chain puts the producer wait there), the x part after it (the same ops)."""
    assert kind in ("plain", "resid", "swiglu")
    assert len(fmts) == (2 if kind == "swiglu" else 1)
    assert K % 512 == 0, "K / 16 sub-blocks must split evenly over 32 lanes"
    rows_wg = R * W
    assert M % rows_wg == 0, (M, rows_wg)
    NT = K // 512                    # sub-blocks per lane
    e = _e if _e is not None else E()
    body = []
    b = body.append

    # ---- per-lane, loop-invariant decode setup, and the per-iteration decode ----
    def lane_setup(f, p):
        """Emit lane constants for format f (prefix p) into body; return a dict."""
        qk, bb = FMT[f]
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

        if f == "mq4":
            # sub-block (s32, hlf): nibble bytes 16 s32 + 8 hlf .. + 7 at 8, scale f16 at 4 (s32 / 4)
            b(f"  %{p}qb0 = index.mul %{p}s32, {e.ci(16)} : index")
            b(f"  %{p}qb1 = index.mul %{p}hlf, {e.ci(8)} : index")
            b(f"  %{p}qb2 = index.add %{p}qb0, %{p}qb1 : index")
            b(f"  %{p}qo = index.add %{p}qb2, {e.ci(8)} : index")
            b(f"  %{p}s4 = index.div %{p}s32, {e.ci(4)} : index")
            b(f"  %{p}so = index.mul %{p}s4, {e.ci(4)} : index")
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
        elif f == "mq4":
            # signed nibbles (the HFQ loader's code - 8) ^ 8 = the unsigned codes; byte j holds elements 2j (low), 2j + 1
            b(f"    %{p}qa = index.add {boff}, %{lp}qo : index")
            raw = ld(wv, nb, f"%{p}qa", p, "qr", 8)
            b(f"    %{p}qw = vector.bitcast {raw} : vector<8xi8> to vector<2xi32>")
            b(f"    %{p}qx = vector.xori %{p}qw, %ws2_88 : vector<2xi32>")
            b(f"    %{p}ql = vector.andi %{p}qx, %ws2_0f : vector<2xi32>")
            b(f"    %{p}qs = vector.shrui %{p}qx, %ws2_4 : vector<2xi32>")
            b(f"    %{p}qh = vector.andi %{p}qs, %ws2_0f : vector<2xi32>")
            for nm, cv in (("%ws2_88", -2004318072), ("%ws2_0f", 0x0F0F0F0F), ("%ws2_4", 4)):
                e.consts[nm] = f"  {nm} = vector.splat {e.cs(cv)} : vector<2xi32>"
            for k in range(2):
                b(f"    %{p}ql{k} = vector.extract %{p}ql[{k}] : vector<2xi32> -> i32")
                b(f"    %{p}qh{k} = vector.extract %{p}qh[{k}] : vector<2xi32> -> i32")
            b(f"    %{p}qlh = vector.from_elements %{p}ql0, %{p}ql1, %{p}qh0, %{p}qh1 : vector<4xi32>")
            qb = tob(p, f"%{p}qlh", "qlb")
            b(f"    %{p}q = vector.shuffle<[0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]> {qb} : vector<16xi8>")
            q = f"%{p}q"
            bias = 8
            b(f"    %{p}sa = index.add {boff}, %{lp}so : index")
            scale = ld_d(hv, nh, f"%{p}sa", 0, p, "d")
        else:
            raise SystemExit("decode_w " + f)
        b(f"    %{p}bo_ = scalar.mulf {scale}, {e.cs(float(bias), 'f32')} : f32")
        return q, scale, f"%{p}bo_"

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
    b("  %zf = scalar.constant 0.0 : f32")
    inits = ", ".join(f"{a} = %zf : f32" for a in accs)
    res = [f"%res{j}" for j in range(len(accs))]
    b(f"  {', '.join(res)} = scf.for %t = [{e.ci(0)} to {e.ci(NT)} step {e.ci(1)}]({inits}) -> ({', '.join(['f32'] * len(accs))}) pipeline({e.ci(PIPE)}) {{")
    b(f"    %xo0 = index.mul %t, {e.ci(512)} : index")
    b("    %xo = index.add %xo0, %xl16 : index")
    xv = ld("%xv", K, "%xo", "", "xa", 16, "f32")      # (not "xv": that would shadow the view)
    b(f"    %xsum = vector.reduce<addf> {xv}, %zf : vector<16xf32>, f32")
    nxt = []
    for i, f in enumerate(fmts):
        for r in range(R):
            p = f"d{i}_{r}"
            b(f"    %{p}to = index.mul %t, {e.ci(S[i]['tstep'])} : index")
            b(f"    %{p}bo = index.add %f{i}base{r}, %{p}to : index")
            q, scale, offs = decode_w(f, S[i], p, f"%wv{i}", f"%wh{i}", wnb[i], f"%{p}bo", f"f{i}")
            b(f"    %{p}qf = vector.uitofp {q} : vector<16xi8> to vector<16xf32>")
            b(f"    %{p}pr = vector.mulf %{p}qf, {xv} : vector<16xf32>")
            b(f"    %{p}dot = vector.reduce<addf> %{p}pr, %zf : vector<16xf32>, f32")
            b(f"    %{p}sd = scalar.mulf {scale}, %{p}dot : f32")
            b(f"    %{p}ox = scalar.mulf {offs}, %xsum : f32")
            b(f"    %{p}c = scalar.subf %{p}sd, %{p}ox : f32")
            b(f"    %{p}n = scalar.addf %a{i}_{r}, %{p}c : f32")
            nxt.append(f"%{p}n")
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
    if rot:   # the rotation's awq / sign loads beside the stores (gen_rot.block_rot_pre)
        b(f"  %rblk = index.div %wg, {e.ci(256 // rows_wg)} : index")
        gen_rot.block_rot_pre(b, "rr", "%rblk", "%tid", M, "%rav", "%rsv")
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
    if rot:
        rot_tail(b, e, M, rows_wg)
    if amax:
        assert kind == "plain"
        amax_tail(b, e, M, R, W, tot)
    if _parts is not None:               # gen_bands: hand back the body, constants stay in e
        _parts.update(body=body, wnb=wnb)
        return None
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
    params = [f"%w{i}: buffer" for i in range(nw)] + ["%x: buffer", "%y: buffer"]
    params += ["%rawq: buffer", "%rsg: buffer", "%rcnt: buffer"] if rot else []
    params += ["%ak: buffer", "%atok: buffer"] if amax else []
    o(f"}} launch({', '.join(params)}) {{")
    o("  %base = index.constant 0 : offset")
    names = [p.split(":")[0] for p in params]
    na = [n + "_na" for n in names]
    o(f"  {', '.join(na)} = buffer.assume.noalias {', '.join(names)} : {', '.join(['buffer'] * len(names))}")
    for i in range(nw):
        o(f"  %wv{i} = buffer.view %w{i}_na[%base] : buffer -> view<{wnb[i]}xi8>")
        o(f"  %wh{i} = buffer.view %w{i}_na[%base] : buffer -> view<{wnb[i] // 2}xf16>")
    o(f"  %xv = buffer.view %x_na[%base] : buffer -> view<{K}xf32>")
    o(f"  %yv = buffer.view %y_na[%base] : buffer -> view<{M}xf32>")
    if amax:
        o("  %akv = buffer.view %ak_na[%base] : buffer -> view<2xi64>")
        o("  %acv = buffer.view %ak_na[%base] : buffer -> view<4xi32>")
        o("  %atv = buffer.view %atok_na[%base] : buffer -> view<1xi32>")
    if rot:
        o(f"  %rav = buffer.view %rawq_na[%base] : buffer -> view<{M}xf32>")
        o("  %rsv = buffer.view %rsg_na[%base] : buffer -> view<512xf32>")
        o(f"  %rcv = buffer.view %rcnt_na[%base] : buffer -> view<{M // 256}xi32>")
    for k in sorted(e.consts, key=lambda n: (not n.startswith("%c"), n)):
        if not k.startswith("%sv") and not k.startswith("%ws"):
            o(e.consts[k])
    for k in sorted(e.consts):
        if k.startswith("%sv") or k.startswith("%ws"):
            o(e.consts[k])
    out += body
    o("}")
    return "\n".join(out) + "\n"


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


def rot_tail(b, e, M, rows_wg):
    """gen rot: the block's last workgroup (gen_rot.last_arriver) rotates it in place (gen_rot.block_rot_post)."""
    assert M % 256 == 0 and 256 % rows_wg == 0
    last = gen_rot.last_arriver(b, "rl", "%rcv", f"view<{M // 256}xi32>", "%rblk", 256 // rows_wg, "%tid")
    b(f"  %rln = index.cmp ult, %tid, {e.ci(16)} : index")
    b(f"  %rgo = scalar.andi {last}, %rln : i1")
    gen_rot.block_rot_post(b, "rr", "%rgo", "%yv", "%yv", M)


_SHARED = re.compile(r"%(c[iwbf]\w*|ws\w*|sv\w*|xv)$")


def gen_bands(fmts, Ms, K, R=2, W=4, name="yah_gemv", peel=0):
    """Several plain GEMVs over one input in one dispatch; workgroup ranges map to bands.

    Band b is fmts[b] with Ms[b] rows, written to its own output.
    Bindings: the weights, x, then the outputs.
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
    params = [f"%w{i}: buffer" for i in range(len(fmts))] + ["%x: buffer"] + \
             [f"%y{i}: buffer" for i in range(len(fmts))]
    o(f"}} launch({', '.join(params)}) {{")
    o("  %base = index.constant 0 : offset")
    names = [p.split(":")[0] for p in params]
    o(f"  {', '.join(n + '_na' for n in names)} = buffer.assume.noalias {', '.join(names)} : {', '.join(['buffer'] * len(names))}")
    for bi in range(len(fmts)):
        o(f"  %b{bi}_wv0 = buffer.view %w{bi}_na[%base] : buffer -> view<{wnbs[bi]}xi8>")
        o(f"  %b{bi}_wh0 = buffer.view %w{bi}_na[%base] : buffer -> view<{wnbs[bi] // 2}xf16>")
        o(f"  %b{bi}_yv = buffer.view %y{bi}_na[%base] : buffer -> view<{Ms[bi]}xf32>")
    o(f"  %xv = buffer.view %x_na[%base] : buffer -> view<{K}xf32>")
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


def footprint(kind, fmts, M, K):
    """Exact binding sizes in bytes, in binding order (weights, x, y)."""
    return [row_bytes(f, K) * M for f in fmts] + [K * 4, M * 4]


if __name__ == "__main__":
    # gen_gemv.py <kind> <fmt[,fmt]> <M> <K> <out.loom> [R] [W]
    kind, fmts, M, K, outp = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    R = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    W = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    open(outp, "w").write(gen(kind, fmts, M, K, R, W))
    print(outp)
