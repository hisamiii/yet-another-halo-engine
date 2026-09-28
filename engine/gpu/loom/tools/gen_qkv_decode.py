#!/usr/bin/env python3
"""Generate yah_qkv_decode_f32.loom (mixed-format decode QKV) and its fixtures.

The decode QKV launch has mixed per-band formats, so the Loom kernel dispatches
the band format through a per-format func.def. Each band buffer is read through
ONE i8 and ONE f16 view sized to the maximum supported format stride; the check
case binds max-sized buffers, so the footprint analysis is satisfied and the
decode still only touches its own format rows.
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
LOOM = os.path.dirname(HERE)

# name -> (gemv file, stride bytes, block bytes, per32)
FMTS = {
    "q4k":   ("yah_gemv_q4k_f32.loom", 2880, 144, False),
    "iq4xs": ("yah_gemv_iq4xs_f32.loom", 2720, 136, False),
    "iq4nl": ("yah_gemv_iq4nl_f32.loom", 2880, 144, True),
    "q5k":   ("yah_gemv_q5k_f32.loom", 3520, 176, False),
    "q6k":   ("yah_gemv_q6k_f32.loom", 4200, 210, False),
}
ID = {"q4k": 1, "iq4xs": 2, "iq4nl": 3, "q5k": 4, "q6k": 5, "q8_0": 6}


def extract(fname):
    lines = open(os.path.join(LOOM, fname)).read().split("\n")
    consts = []
    seen = set()
    for ln in lines:
        m = re.match(r"\s+(%[A-Za-z0-9_]+) = (?:scalar|index)\.constant", ln)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            consts.append(ln.rstrip())
    start = end = None
    for i, ln in enumerate(lines):
        if "%i_i = scalar.addi %lane_i, %j32" in ln:
            start = i + 1
        if "%prod = scalar.mulf %value, %xv : f32" in ln:
            end = i
    assert start is not None and end is not None, fname
    return consts, lines[start:end + 1]


def rename(ln):
    for a, b in [("view<[%w_bytes]xi8>", "view<[%wn]xi8>"),
                 ("view<[%w_halfs]xf16>", "view<[%whn]xf16>"),
                 ("view<[%x_elems]xf32>", "view<[%xn]xf32>"),
                 ("%w_f16_view", "%wh"), ("%w_view", "%wv"),
                 ("%w_half_last", "%whlast"), ("%w_last", "%wlast"),
                 ("%x_view", "%xs"), ("%x_last", "%xlast")]:
        ln = ln.replace(a, b)
    return ln


def fmt_func(name):
    fname, stride, block, per32 = FMTS[name]
    consts, body = extract(fname)
    out = ["func.def @%s_step(%%wn: index, %%whn: index, %%xn: index, %%wv: view<[%%wn]xi8>, %%wh: view<[%%whn]xf16>, %%xs: view<[%%xn]xf32>, %%row_i: i32, %%kblk_i: i32, %%i_i: i32) -> (f32) {" % name]
    out += consts
    out.append("  %wlast = index.sub %wn, %c1 : index")
    out.append("  %whlast = index.sub %whn, %c1 : index")
    out.append("  %xlast = index.sub %xn, %c1 : index")
    if per32:
        out.append("  %row_off = scalar.muli %row_i, %c" + str(stride) + "i : i32")
        out.append("  %grp_off0 = scalar.muli %kblk_i, %c" + str(block) + "i : i32")
        out.append("  %grp_off = scalar.addi %row_off, %grp_off0 : i32")
        out.append("  %blk8 = scalar.shrui %i_i, %c5i : i32")
        out.append("  %blkj = scalar.muli %blk8, %c18i : i32")
        out.append("  %blk_off = scalar.addi %grp_off, %blkj : i32")
    else:
        out.append("  %row_off = scalar.muli %row_i, %c" + str(stride) + "i : i32")
        out.append("  %blk_off0 = scalar.muli %kblk_i, %c" + str(block) + "i : i32")
        out.append("  %blk_off = scalar.addi %row_off, %blk_off0 : i32")
    out.append("  %kblk256 = scalar.shli %kblk_i, %c8i : i32")
    out += [rename(l) for l in body]
    out.append("  func.return %prod : f32")
    out.append("}")
    out.append("")
    return out


Q8_0 = """func.def @q8_0_step(%wn: index, %whn: index, %xn: index, %wv: view<[%wn]xi8>, %wh: view<[%whn]xf16>, %xs: view<[%xn]xf32>, %row_i: i32, %kblk_i: i32, %i_i: i32) -> (f32) {
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c0i = scalar.constant 0 : i32
  %c1i = scalar.constant 1 : i32
  %c2i = scalar.constant 2 : i32
  %c5i = scalar.constant 5 : i32
  %c8i = scalar.constant 8 : i32
  %c31i = scalar.constant 31 : i32
  %c34i = scalar.constant 34 : i32
  %c272i = scalar.constant 272 : i32
  %c5440i = scalar.constant 5440 : i32
  %wlast = index.sub %wn, %c1 : index
  %whlast = index.sub %whn, %c1 : index
  %xlast = index.sub %xn, %c1 : index
  %row_off = scalar.muli %row_i, %c5440i : i32
  %grp_off = scalar.muli %kblk_i, %c272i : i32
  %base0 = scalar.addi %row_off, %grp_off : i32
  %blk8 = scalar.shrui %i_i, %c5i : i32
  %blkj = scalar.muli %blk8, %c34i : i32
  %blk_off = scalar.addi %base0, %blkj : i32
  %within = scalar.andi %i_i, %c31i : i32
  %qs0 = scalar.addi %blk_off, %c2i : i32
  %qs_i = scalar.addi %qs0, %within : i32
  %qs_ix = index.cast %qs_i : i32 to index
  %qs_lo = index.max %qs_ix, %c0 : index
  %qs_idx = index.min %qs_lo, %wlast : index
  %qb8 = view.load %wv[%qs_idx] : view<[%wn]xi8> -> i8
  %qb = scalar.extsi %qb8 : i8 to i32
  %q_f = scalar.sitofp %qb : i32 to f32
  %blk_half0 = scalar.shrui %blk_off, %c1i : i32
  %d_ix = index.cast %blk_half0 : i32 to index
  %d_lo = index.max %d_ix, %c0 : index
  %d_idx = index.min %d_lo, %whlast : index
  %d_half_v = view.load %wh[%d_idx] : view<[%whn]xf16> -> f16
  %d = scalar.extf %d_half_v : f16 to f32
  %value = scalar.mulf %d, %q_f : f32
  %kblk256 = scalar.shli %kblk_i, %c8i : i32
  %kidx_i = scalar.addi %kblk256, %i_i : i32
  %kidx_ix = index.cast %kidx_i : i32 to index
  %kidx_lo = index.max %kidx_ix, %c0 : index
  %kidx = index.min %kidx_lo, %xlast : index
  %xv_v = view.load %xs[%kidx] : view<[%xn]xf32> -> f32
  %prod = scalar.mulf %value, %xv_v : f32
  func.return %prod : f32
}
"""


def call(name, wv, wh, wn, whn, row, kblk, i, xv, xn, ind):
    pre = " " * ind
    sig = ("(index, index, index, view<[%%%s]xi8>, view<[%%%s]xf16>, view<[%%%s]xf32>, i32, i32, i32) -> (f32)"
           % (wn, whn, xn))
    return "%s%%r = func.call @%s_step(%%%s, %%%s, %%%s, %%%s, %%%s, %%%s, %%%s, %%%s, %%%s) : %s" % (
        pre, name, wn, whn, xn, wv, wh, xv, row, kblk, i, sig)


def emit_dispatch(formats, isvar, vars_, ind):
    """Return list of lines assigning %d = nested scf.if over formats."""
    out = []
    fmt = formats[0]
    wv, wh, wn, whn, row, kblk, i, xv, xn = vars_
    out.append(" " * ind + "%%d = scf.if %%%s -> (f32) {" % isvar[0])
    out.append(call(fmt, wv, wh, wn, whn, row, kblk, i, xv, xn, ind + 2))
    out.append(" " * (ind + 2) + "scf.yield %r : f32")
    out.append(" " * ind + "} else {")
    out += emit_nested(formats[1:], isvar[1:], vars_, ind + 2)
    out.append(" " * ind + "}")
    return out


def emit_nested(formats, isvars, vars_, ind):
    wv, wh, wn, whn, row, kblk, i, xv, xn = vars_
    if len(formats) == 1:
        return [call(formats[0], wv, wh, wn, whn, row, kblk, i, xv, xn, ind),
                " " * ind + "scf.yield %r : f32"]
    fmt = formats[0]
    out = []
    out.append(" " * ind + "%%dn = scf.if %%%s -> (f32) {" % isvars[0])
    out.append(call(fmt, wv, wh, wn, whn, row, kblk, i, xv, xn, ind + 2))
    out.append(" " * (ind + 2) + "scf.yield %r : f32")
    out.append(" " * ind + "} else {")
    out += emit_nested(formats[1:], isvars[1:], vars_, ind + 2)
    out.append(" " * ind + "}")
    out.append(" " * ind + "scf.yield %dn : f32")
    return out


def emit_band(band, row_expr, wv, wh, wn, whn, typearg, outview, outext, outidx, formats, ind, tail):
    """Emit scf.if branches for one band. Returns list of lines."""
    o = []
    pad = " " * ind
    isvar = ["is_%s_%s" % (band, f) for f in formats]
    for f, v in zip(formats, isvar):
        o.append("%s%%%s = scalar.cmpi eq, %%%s, %%%s : i32" % (pad, v, typearg, "id_" + f))
    o.append("%s%%row_i_%s = index.cast %s : index to i32" % (pad, band, row_expr))
    o.append("%s%%acc_%s = scf.for %%kblk_%s = [%%c0 to %%k_groups step %%c1](%%a0_%s = %%zero : f32) -> (f32) {" % (pad, band, band, band))
    o.append("%s  %%kblk_i_%s = index.cast %%kblk_%s : index to i32" % (pad, band, band))
    o.append("%s  %%acc1_%s = scf.for %%j_%s = [%%c0 to %%c8 step %%c1](%%a1_%s = %%a0_%s : f32) -> (f32) {" % (pad, band, band, band, band))
    o.append("%s    %%j_i_%s = index.cast %%j_%s : index to i32" % (pad, band, band))
    o.append("%s    %%j32_%s = scalar.shli %%j_i_%s, %%c5i : i32" % (pad, band, band))
    o.append("%s    %%i_i_%s = scalar.addi %%lane_i, %%j32_%s : i32" % (pad, band, band))
    vars_ = (wv, wh, wn, whn, "row_i_" + band, "kblk_i_" + band, "i_i_" + band, "xv", "x_elems")
    o += emit_dispatch(formats, isvar, vars_, ind + 4)
    o.append("%s    %%next_%s = scalar.addf %%a1_%s, %%d : f32" % (pad, band, band))
    o.append("%s    scf.yield %%next_%s : f32" % (pad, band))
    o.append("%s  }" % pad)
    o.append("%s  scf.yield %%acc1_%s : f32" % (pad, band))
    o.append("%s}" % pad)
    o.append("%s%%sum_%s = kernel.subgroup.reduce<addf> %%acc_%s : f32" % (pad, band, band))
    o.append("%s%%is_zero_%s = index.cmp eq, %%lane, %%c0 : index" % (pad, band))
    o.append("%s%%st_%s = scf.if %%is_zero_%s -> (f32) {" % (pad, band, band))
    o.append("%s  view.store %%sum_%s, %%%s[%s] : f32, view<[%%%s]xf32>" % (pad, band, outview, outidx, outext))
    o.append("%s  scf.yield %%zero : f32" % pad)
    o.append("%s} else {" % pad)
    o.append("%s  scf.yield %%zero : f32" % pad)
    o.append("%s}" % pad)
    if tail:
        o.append("%s  scf.yield %%zero : f32" % pad)
    return o


def emit_check(name, qf, kf, vf):
    idm = {"q4k": 1, "iq4xs": 2, "iq4nl": 3, "q5k": 4, "q6k": 5, "q8_0": 6}
    o = []
    o.append("// q=%s k=%s v=%s; all three bands share the same sparse x." % (qf, kf, vf))
    o.append("check.case public @%s {" % name)
    for tag, fmt in [("q_w", qf), ("k_w", kf), ("v_w", vf)]:
        o.append('  %%%s = check.file.read.npy path("fixtures/qkv_decode/%s_pad.npy") : tensor<87040xi8>' % (tag, fmt))
    o.append('  %x = check.file.read.npy path("fixtures/qkv_decode/x.npy") : tensor<5120xf32>')
    o.append("  %q_out = check.generate.fill value(0.0) : tensor<16xf32>")
    o.append("  %k_out = check.generate.fill value(0.0) : tensor<16xf32>")
    o.append("  %v_out = check.generate.fill value(0.0) : tensor<16xf32>")
    for tag, fmt in [("eq", qf), ("ek", kf), ("ev", vf)]:
        o.append('  %%%s = check.file.read.npy path("fixtures/qkv_decode/%s_expected.npy") : tensor<16xf32>' % (tag, fmt))
    o.append("  %%tq = check.literal value(%d) : i32" % idm[qf])
    o.append("  %%tk = check.literal value(%d) : i32" % idm[kf])
    o.append("  %%tv = check.literal value(%d) : i32" % idm[vf])
    o.append("  kernel.launch @yah_qkv_decode(%q_w, %k_w, %v_w, %x, %q_out, %k_out, %v_out, %tq, %tk, %tv) : (tensor<87040xi8>, tensor<87040xi8>, tensor<87040xi8>, tensor<5120xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, i32, i32, i32)")
    for actual, exp in [("q_out", "eq"), ("k_out", "ek"), ("v_out", "ev")]:
        o.append("  check.expect.close actual(%%%s) expected(%%%s) atol(1.0000000000000001e-05) rtol(1.0000000000000001e-05) nan(same) : tensor<16xf32>" % (actual, exp))
    o.append("  check.return")
    o.append("}")
    o.append("")
    o.append("check.benchmark<@%s> @%s_bench" % (name, name))
    o.append("")
    return o

def main():
    out = []
    out.append("// YAH mixed-format decode QKV projection, Loom port of")
    out.append("// Wave32FusedQKVProjectionsKernel_1Row (qkv.hip). One workgroup per row over")
    out.append("// q_dim + 2*kv_dim rows; each band dispatches its own runtime GgmlType.")
    out.append("// On this shard q in {Q4_K,IQ4_XS,IQ4_NL,Q6_K} and k,v in {Q4_K,Q5_K,Q6_K,Q8_0}.")
    out.append("// The per-band weight buffer is read through ONE i8 and ONE f16 view sized to")
    out.append("// the maximum supported stride, so the buffers must be at least")
    out.append("// q_rows*4200 and kv_rows*5440 bytes respectively.")
    out.append("//")
    out.append("// Required config: --config=yah_qkv_decode.q_dim=16")
    out.append("//                  --config=yah_qkv_decode.kv_dim=16")
    out.append("//                  --config=yah_qkv_decode.k_groups=20")
    out.append("amdgpu.target<gfx11-generic> @yah_wave32 {subgroup_size = 32}")
    out.append("")
    for f in FMTS:
        out += fmt_func(f)
    out.append(Q8_0)
    out.append("")
    out.append("config.decl @yah_qkv_decode.q_dim : %value: index where [range(%value, 1, 65536)]")
    out.append("")
    out.append("config.decl @yah_qkv_decode.kv_dim : %value: index where [range(%value, 1, 65536)]")
    out.append("")
    out.append("config.decl @yah_qkv_decode.k_groups : %value: index where [range(%value, 1, 65536)]")
    out.append("")
    out.append("kernel.def target(@yah_wave32) @yah_qkv_decode() {")
    out.append("  %unit = index.constant 1 : index")
    out.append("  %q_dim = config.get @yah_qkv_decode.q_dim : index")
    out.append("  %kv_dim = config.get @yah_qkv_decode.kv_dim : index")
    out.append("  %c2 = index.constant 2 : index")
    out.append("  %c32 = index.constant 32 : index")
    out.append("  %kv2 = index.mul %kv_dim, %c2 : index")
    out.append("  %total = index.add %q_dim, %kv2 : index")
    out.append("  kernel.launch.config workgroups(%total, %unit, %unit) workgroup_size(%c32, %unit, %unit) : index")
    out.append("} launch(%q_w: buffer, %k_w: buffer, %v_w: buffer, %x: buffer, %q_out: buffer, %k_out: buffer, %v_out: buffer, %q_type: i32, %k_type: i32, %v_type: i32) {")
    out.append("  %base = index.constant 0 : offset")
    out.append("  %c0 = index.constant 0 : index")
    out.append("  %c1 = index.constant 1 : index")
    out.append("  %c2i = index.constant 2 : index")
    out.append("  %c4200 = index.constant 4200 : index")
    out.append("  %c2100 = index.constant 2100 : index")
    out.append("  %c5440 = index.constant 5440 : index")
    out.append("  %c2720 = index.constant 2720 : index")
    out.append("  %c256 = index.constant 256 : index")
    out.append("  %c8 = index.constant 8 : index")
    out.append("  %c0i = scalar.constant 0 : i32")
    out.append("  %c8i = scalar.constant 8 : i32")
    out.append("  %c5i = scalar.constant 5 : i32")
    out.append("  %zero = scalar.constant 0.0 : f32")
    out.append("  %id_q4k = scalar.constant 1 : i32")
    out.append("  %id_iq4xs = scalar.constant 2 : i32")
    out.append("  %id_iq4nl = scalar.constant 3 : i32")
    out.append("  %id_q5k = scalar.constant 4 : i32")
    out.append("  %id_q6k = scalar.constant 5 : i32")
    out.append("  %id_q8_0 = scalar.constant 6 : i32")
    out.append("  %q_dim = config.get @yah_qkv_decode.q_dim : index")
    out.append("  %kv_dim = config.get @yah_qkv_decode.kv_dim : index")
    out.append("  %k_groups = config.get @yah_qkv_decode.k_groups : index")
    out.append("  %q_wn = index.mul %q_dim, %c4200 : index")
    out.append("  %q_whn = index.mul %q_dim, %c2100 : index")
    out.append("  %kv_wn = index.mul %kv_dim, %c5440 : index")
    out.append("  %kv_whn = index.mul %kv_dim, %c2720 : index")
    out.append("  %x_elems = index.mul %k_groups, %c256 : index")
    out.append("  %q_na, %k_na, %v_na, %x_na, %qo_na, %ko_na, %vo_na = buffer.assume.noalias %q_w, %k_w, %v_w, %x, %q_out, %k_out, %v_out : buffer, buffer, buffer, buffer, buffer, buffer, buffer")
    out.append("  %qw = buffer.view %q_na[%base] : buffer -> view<[%q_wn]xi8>")
    out.append("  %qwh = buffer.view %q_na[%base] : buffer -> view<[%q_whn]xf16>")
    out.append("  %kw = buffer.view %k_na[%base] : buffer -> view<[%kv_wn]xi8>")
    out.append("  %kwh = buffer.view %k_na[%base] : buffer -> view<[%kv_whn]xf16>")
    out.append("  %vw = buffer.view %v_na[%base] : buffer -> view<[%kv_wn]xi8>")
    out.append("  %vwh = buffer.view %v_na[%base] : buffer -> view<[%kv_whn]xf16>")
    out.append("  %xv = buffer.view %x_na[%base] : buffer -> view<[%x_elems]xf32>")
    out.append("  %qo = buffer.view %qo_na[%base] : buffer -> view<[%q_dim]xf32>")
    out.append("  %ko = buffer.view %ko_na[%base] : buffer -> view<[%kv_dim]xf32>")
    out.append("  %vo = buffer.view %vo_na[%base] : buffer -> view<[%kv_dim]xf32>")
    out.append("  %m = kernel.workgroup.id<x> : index")
    out.append("  %lane = kernel.workitem.id<x> : index")
    out.append("  %lane_i = index.cast %lane : index to i32")
    out.append("  %qk = index.add %q_dim, %kv_dim : index")
    out.append("  %is_q = index.cmp ult, %m, %q_dim : index")
    out.append("  %is_k = index.cmp ult, %m, %qk : index")
    out.append("  %row_k = index.sub %m, %q_dim : index")
    out.append("  %row_v = index.sub %m, %qk : index")
    # q band wraps k/v in its else; k wraps v in its else; v plain.
    oq = emit_band("q", "%m", "qw", "qwh", "q_wn", "q_whn", "q_type", "qo", "q_dim", "%m", ["q4k", "iq4xs", "iq4nl", "q6k"], 2, True)
    out.append("  %q_sink = scf.if %is_q -> (f32) {")
    out += oq
    out.append("  } else {")
    ok = emit_band("k", "%row_k", "kw", "kwh", "kv_wn", "kv_whn", "k_type", "ko", "kv_dim", "%row_k", ["q4k", "q5k", "q6k", "q8_0"], 4, True)
    out.append("    %k_sink = scf.if %is_k -> (f32) {")
    out += ok
    out.append("    } else {")
    ov = emit_band("v", "%row_v", "vw", "vwh", "kv_wn", "kv_whn", "v_type", "vo", "kv_dim", "%row_v", ["q4k", "q5k", "q6k", "q8_0"], 6, False)
    out += ov
    out.append("      scf.yield %zero : f32")
    out.append("    }")
    out.append("    scf.yield %k_sink : f32")
    out.append("  }")
    out.append("  kernel.return")
    out.append("}")
    out.append("")
    out += emit_check("yah_qkv_decode_case_a", "iq4xs", "q5k", "q6k")
    out += emit_check("yah_qkv_decode_case_b", "q4k", "q8_0", "q4k")
    out += emit_check("yah_qkv_decode_case_c", "iq4nl", "q6k", "q5k")
    out.append("// Production shape: q_dim=6144, kv_dim=1024, k_groups=20. All-zero")
    out.append("// max-sized weights decode to zero in every band.")
    out.append("check.case public @yah_qkv_decode_full_case {")
    out.append("  %q_w = check.generate.fill value(0) : tensor<25804800xi8>")
    out.append("  %k_w = check.generate.fill value(0) : tensor<5570560xi8>")
    out.append("  %v_w = check.generate.fill value(0) : tensor<5570560xi8>")
    out.append("  %x = check.generate.fill value(1.0) : tensor<5120xf32>")
    out.append("  %q_out = check.generate.fill value(0.0) : tensor<6144xf32>")
    out.append("  %k_out = check.generate.fill value(0.0) : tensor<1024xf32>")
    out.append("  %v_out = check.generate.fill value(0.0) : tensor<1024xf32>")
    out.append("  %eq = check.generate.fill value(0.0) : tensor<6144xf32>")
    out.append("  %ek = check.generate.fill value(0.0) : tensor<1024xf32>")
    out.append("  %ev = check.generate.fill value(0.0) : tensor<1024xf32>")
    out.append("  %t1 = check.literal value(1) : i32")
    out.append("  kernel.launch @yah_qkv_decode(%q_w, %k_w, %v_w, %x, %q_out, %k_out, %v_out, %t1, %t1, %t1) : (tensor<25804800xi8>, tensor<5570560xi8>, tensor<5570560xi8>, tensor<5120xf32>, tensor<6144xf32>, tensor<1024xf32>, tensor<1024xf32>, i32, i32, i32)")
    out.append("  check.expect.close actual(%q_out) expected(%eq) atol(0.0) rtol(0.0) nan(same) : tensor<6144xf32>")
    out.append("  check.expect.close actual(%k_out) expected(%ek) atol(0.0) rtol(0.0) nan(same) : tensor<1024xf32>")
    out.append("  check.expect.close actual(%v_out) expected(%ev) atol(0.0) rtol(0.0) nan(same) : tensor<1024xf32>")
    out.append("  check.return")
    out.append("}")
    out.append("")
    out.append("check.benchmark<@yah_qkv_decode_full_case> @yah_qkv_decode_full_bench")
    open(os.path.join(LOOM, "yah_qkv_decode_f32.loom"), "w").write("\n".join(out) + "\n")
    print("wrote yah_qkv_decode_f32.loom", len(out), "lines")


if __name__ == "__main__":
    main()