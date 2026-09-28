#!/usr/bin/env python3
"""Generate yah_swiglu_decode_f32.loom (mixed-format decode fused SwiGLU GEMV).

Port of Wave32FusedQuantSwiGLUGEMVKernel_2Rows runtime-dispatch form. Two
row-dot chains (gate and up) over the same activation, combined as
silu(gate_dot)*up_dot. One func.def per format; the band weight buffer is read
through one i8/f16 view sized to the max stride (Q6_K 4200 bytes/row for 16
rows = 67200 bytes). IQ3_S/IQ3_XXS take the grid_s/grid_x/ksigns operands; the
other formats ignore them. IQ2_XS/IQ2_S are documented as still open.
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
LOOM = os.path.dirname(HERE)

# name -> (gemv file, stride, block, per32)
FMTS = {
    "q4k":   ("yah_gemv_q4k_f32.loom", 2880, 144, False),
    "iq4xs": ("yah_gemv_iq4xs_f32.loom", 2720, 136, False),
    "iq4nl": ("yah_gemv_iq4nl_f32.loom", 2880, 144, True),
    "q5k":   ("yah_gemv_q5k_f32.loom", 3520, 176, False),
    "q6k":   ("yah_gemv_q6k_f32.loom", 4200, 210, False),
    "q3k":   ("yah_gemv_q3k_f32.loom", 2200, 110, False),
    "iq3s":  ("yah_gemv_iq3s_f32.loom", 2200, 110, False),
    "iq3xxs":("yah_gemv_iq3xxs_f32.loom", 1960, 98, False),
}
ID = {"q4k": 1, "iq4xs": 2, "iq4nl": 3, "q5k": 4, "q6k": 5, "q3k": 6,
      "iq3s": 7, "iq3xxs": 8}


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


def rename(ln, gridp="%gs"):
    for a, b in [("view<[%w_bytes]xi8>", "view<[%wn]xi8>"),
                 ("view<[%w_halfs]xf16>", "view<[%whn]xf16>"),
                 ("view<[%x_elems]xf32>", "view<[%xn]xf32>"),
                 ("%w_f16_view", "%wh"), ("%w_view", "%wv"),
                 ("%w_half_last", "%whlast"), ("%w_last", "%wlast"),
                 ("%x_view", "%xs"), ("%x_last", "%xlast"),
                 ("%grid_view", gridp), ("%ksigns_view", "%ks")]:
        ln = ln.replace(a, b)
    return ln


def fmt_func(name):
    fname, stride, block, per32 = FMTS[name]
    consts, body = extract(fname)
    out = ["func.def @%s_step(%%wn: index, %%whn: index, %%xn: index, %%wv: view<[%%wn]xi8>, %%wh: view<[%%whn]xf16>, %%xs: view<[%%xn]xf32>, %%gs: view<512xi32>, %%gx: view<256xi32>, %%ks: view<128xi8>, %%row_i: i32, %%kblk_i: i32, %%i_i: i32) -> (f32) {" % name]
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
    gridp = "%gx" if name == "iq3xxs" else "%gs"
    out += [rename(l, gridp) for l in body]
    out.append("  func.return %prod : f32")
    out.append("}")
    out.append("")
    return out


def call(name, chain, wv, wh, wn, whn, row, kblk, i, xv, xn, ind):
    pre = " " * ind
    sig = "(index, index, index, view<[%%%s]xi8>, view<[%%%s]xf16>, view<[%%%s]xf32>, view<512xi32>, view<256xi32>, view<128xi8>, i32, i32, i32) -> (f32)" % (wn, whn, xn)
    return "%s%%r_%s = func.call @%s_step(%%%s, %%%s, %%%s, %%%s, %%%s, %%%s, %%gs, %%gx, %%ks, %%%s, %%%s, %%%s) : %s" % (
        pre, chain, name, wn, whn, xn, wv, wh, xv, row, kblk, i, sig)


def emit_dispatch(chain, formats, isvar, vars_, ind):
    out = []
    wv, wh, wn, whn, row, kblk, i, xv, xn = vars_
    out.append(" " * ind + "%%d_%s = scf.if %%%s -> (f32) {" % (chain, isvar[0]))
    out.append(call(formats[0], chain, wv, wh, wn, whn, row, kblk, i, xv, xn, ind + 2))
    out.append(" " * (ind + 2) + "scf.yield %%r_%s : f32" % chain)
    out.append(" " * ind + "} else {")
    out += emit_nested(chain, formats[1:], isvar[1:], vars_, ind + 2)
    out.append(" " * ind + "}")
    return out


def emit_nested(chain, formats, isvars, vars_, ind):
    wv, wh, wn, whn, row, kblk, i, xv, xn = vars_
    if len(formats) == 1:
        return [call(formats[0], chain, wv, wh, wn, whn, row, kblk, i, xv, xn, ind),
                " " * ind + "scf.yield %%r_%s : f32" % chain]
    out = []
    out.append(" " * ind + "%%dn_%s = scf.if %%%s -> (f32) {" % (chain, isvars[0]))
    out.append(call(formats[0], chain, wv, wh, wn, whn, row, kblk, i, xv, xn, ind + 2))
    out.append(" " * (ind + 2) + "scf.yield %%r_%s : f32" % chain)
    out.append(" " * ind + "} else {")
    out += emit_nested(chain, formats[1:], isvars[1:], vars_, ind + 2)
    out.append(" " * ind + "}")
    out.append(" " * ind + "scf.yield %%dn_%s : f32" % chain)
    return out


def emit_check(name, gf, uf):
    idm = ID
    o = []
    o.append("// gate=%s up=%s; both chains share x." % (gf, uf))
    o.append("check.case public @%s {" % name)
    o.append('  %%gate_w = check.file.read.npy path("fixtures/swiglu_decode/%s_pad.npy") : tensor<67200xi8>' % gf)
    o.append('  %%up_w = check.file.read.npy path("fixtures/swiglu_decode/%s_pad.npy") : tensor<67200xi8>' % uf)
    o.append('  %grid_s = check.file.read.npy path("fixtures/iq3s_gemm/grid.npy") : tensor<512xi32>')
    o.append('  %grid_x = check.file.read.npy path("fixtures/iq3xxs_gemm/grid.npy") : tensor<256xi32>')
    o.append('  %ksigns = check.file.read.npy path("fixtures/iq3xxs_gemm/ksigns.npy") : tensor<128xi8>')
    o.append('  %x = check.file.read.npy path("fixtures/swiglu_decode/x.npy") : tensor<5120xf32>')
    o.append("  %out = check.generate.fill value(0.0) : tensor<16xf32>")
    o.append('  %%expected = check.file.read.npy path("fixtures/swiglu_decode/%s_%s_expected.npy") : tensor<16xf32>' % (gf, uf))
    o.append("  %%tg = check.literal value(%d) : i32" % idm[gf])
    o.append("  %%tu = check.literal value(%d) : i32" % idm[uf])
    o.append("  kernel.launch @yah_swiglu_decode(%gate_w, %up_w, %grid_s, %grid_x, %ksigns, %x, %out, %tg, %tu) : (tensor<67200xi8>, tensor<67200xi8>, tensor<512xi32>, tensor<256xi32>, tensor<128xi8>, tensor<5120xf32>, tensor<16xf32>, i32, i32)")
    o.append("  check.expect.close actual(%out) expected(%expected) atol(0.001) rtol(1.0000000000000001e-05) nan(same) : tensor<16xf32>")
    o.append("  check.return")
    o.append("}")
    o.append("")
    o.append("check.benchmark<@%s> @%s_bench" % (name, name))
    o.append("")
    return o


def main():
    out = []
    out.append("// YAH mixed-format decode fused SwiGLU GEMV, Loom port of")
    out.append("// Wave32FusedQuantSwiGLUGEMVKernel_2Rows (swiglu.hip) runtime-dispatch form.")
    out.append("// gate and up row dots over one activation, combined as silu(gate)*up.")
    out.append("// Each weight buffer is read through one i8/f16 view sized to the max stride")
    out.append("// (Q6_K 4200 B/row), so the buffers must be at least m_rows*4200 bytes.")
    out.append("// iq3s/iq3xxs read grid_s/grid_x/ksigns; other formats ignore them.")
    out.append("// IQ2_XS/IQ2_S gate/up pairs are not in this port yet.")
    out.append("//")
    out.append("// Required config: --config=yah_swiglu_decode.m_rows=16")
    out.append("//                  --config=yah_swiglu_decode.k_groups=20")
    out.append("amdgpu.target<gfx11-generic> @yah_wave32 {subgroup_size = 32}")
    out.append("")
    for f in FMTS:
        out += fmt_func(f)
    out.append("")
    out.append("config.decl @yah_swiglu_decode.m_rows : %value: index where [range(%value, 1, 65536)]")
    out.append("")
    out.append("config.decl @yah_swiglu_decode.k_groups : %value: index where [range(%value, 1, 65536)]")
    out.append("")
    out.append("kernel.def target(@yah_wave32) @yah_swiglu_decode() {")
    out.append("  %unit = index.constant 1 : index")
    out.append("  %m_rows = config.get @yah_swiglu_decode.m_rows : index")
    out.append("  %c32 = index.constant 32 : index")
    out.append("  kernel.launch.config workgroups(%m_rows, %unit, %unit) workgroup_size(%c32, %unit, %unit) : index")
    out.append("} launch(%gate_w: buffer, %up_w: buffer, %grid_s: buffer, %grid_x: buffer, %ksigns: buffer, %x: buffer, %out: buffer, %gate_type: i32, %up_type: i32) {")
    out.append("  %base = index.constant 0 : offset")
    out.append("  %c0 = index.constant 0 : index")
    out.append("  %c1 = index.constant 1 : index")
    out.append("  %c8 = index.constant 8 : index")
    out.append("  %c4200 = index.constant 4200 : index")
    out.append("  %c2100 = index.constant 2100 : index")
    out.append("  %c256 = index.constant 256 : index")
    out.append("  %c0i = scalar.constant 0 : i32")
    out.append("  %c1i = scalar.constant 1 : i32")
    out.append("  %c5i = scalar.constant 5 : i32")
    out.append("  %zero = scalar.constant 0.0 : f32")
    out.append("  %one = scalar.constant 1.0 : f32")
    out.append("  %negone = scalar.constant -1.0 : f32")
    out.append("  %id_q4k = scalar.constant 1 : i32")
    out.append("  %id_iq4xs = scalar.constant 2 : i32")
    out.append("  %id_iq4nl = scalar.constant 3 : i32")
    out.append("  %id_q5k = scalar.constant 4 : i32")
    out.append("  %id_q6k = scalar.constant 5 : i32")
    out.append("  %id_q3k = scalar.constant 6 : i32")
    out.append("  %id_iq3s = scalar.constant 7 : i32")
    out.append("  %id_iq3xxs = scalar.constant 8 : i32")
    out.append("  %m_rows = config.get @yah_swiglu_decode.m_rows : index")
    out.append("  %k_groups = config.get @yah_swiglu_decode.k_groups : index")
    out.append("  %w_bytes = index.mul %m_rows, %c4200 : index")
    out.append("  %w_halfs = index.mul %m_rows, %c2100 : index")
    out.append("  %x_elems = index.mul %k_groups, %c256 : index")
    out.append("  %gw_na, %uw_na, %gs_na, %gx_na, %ks_na, %x_na, %o_na = buffer.assume.noalias %gate_w, %up_w, %grid_s, %grid_x, %ksigns, %x, %out : buffer, buffer, buffer, buffer, buffer, buffer, buffer")
    out.append("  %gw = buffer.view %gw_na[%base] : buffer -> view<[%w_bytes]xi8>")
    out.append("  %gwh = buffer.view %gw_na[%base] : buffer -> view<[%w_halfs]xf16>")
    out.append("  %uw = buffer.view %uw_na[%base] : buffer -> view<[%w_bytes]xi8>")
    out.append("  %uwh = buffer.view %uw_na[%base] : buffer -> view<[%w_halfs]xf16>")
    out.append("  %gs = buffer.view %gs_na[%base] : buffer -> view<512xi32>")
    out.append("  %gx = buffer.view %gx_na[%base] : buffer -> view<256xi32>")
    out.append("  %ks = buffer.view %ks_na[%base] : buffer -> view<128xi8>")
    out.append("  %xs = buffer.view %x_na[%base] : buffer -> view<[%x_elems]xf32>")
    out.append("  %ov = buffer.view %o_na[%base] : buffer -> view<[%m_rows]xf32>")
    out.append("  %m = kernel.workgroup.id<x> : index")
    out.append("  %lane = kernel.workitem.id<x> : index")
    out.append("  %lane_i = index.cast %lane : index to i32")
    out.append("  %row_i = index.cast %m : index to i32")
    for f in FMTS:
        out.append("  %%g_is_%s = scalar.cmpi eq, %%gate_type, %%id_%s : i32" % (f, f))
    for f in FMTS:
        out.append("  %%u_is_%s = scalar.cmpi eq, %%up_type, %%id_%s : i32" % (f, f))
    out.append("  %accg, %accu = scf.for %kblk = [%c0 to %k_groups step %c1](%ag = %zero : f32, %au = %zero : f32) -> (f32, f32) {")
    out.append("    %kblk_i = index.cast %kblk : index to i32")
    out.append("    %a1g, %a1u = scf.for %j = [%c0 to %c8 step %c1](%g0 = %ag : f32, %u0 = %au : f32) -> (f32, f32) {")
    out.append("      %j_i = index.cast %j : index to i32")
    out.append("      %j32 = scalar.shli %j_i, %c5i : i32")
    out.append("      %i_i = scalar.addi %lane_i, %j32 : i32")
    gvars = ("gw", "gwh", "w_bytes", "w_halfs", "row_i", "kblk_i", "i_i", "xs", "x_elems")
    gis = ["g_is_" + f for f in FMTS]
    out += emit_dispatch("g", list(FMTS.keys()), gis, gvars, 6)
    uvars = ("uw", "uwh", "w_bytes", "w_halfs", "row_i", "kblk_i", "i_i", "xs", "x_elems")
    uis = ["u_is_" + f for f in FMTS]
    out += emit_dispatch("u", list(FMTS.keys()), uis, uvars, 6)
    out.append("      %ng = scalar.addf %g0, %d_g : f32")
    out.append("      %nu = scalar.addf %u0, %d_u : f32")
    out.append("      scf.yield %ng, %nu : f32, f32")
    out.append("    }")
    out.append("    scf.yield %a1g, %a1u : f32, f32")
    out.append("  }")
    out.append("  %sumg = kernel.subgroup.reduce<addf> %accg : f32")
    out.append("  %sumu = kernel.subgroup.reduce<addf> %accu : f32")
    out.append("  %negg = scalar.mulf %sumg, %negone : f32")
    out.append("  %eg = scalar.expf<afn> %negg : f32")
    out.append("  %dg = scalar.addf %one, %eg : f32")
    out.append("  %sg = scalar.divf %one, %dg : f32")
    out.append("  %act = scalar.mulf %sumg, %sg : f32")
    out.append("  %res = scalar.mulf %act, %sumu : f32")
    out.append("  %is_zero = index.cmp eq, %lane, %c0 : index")
    out.append("  %st = scf.if %is_zero -> (f32) {")
    out.append("    view.store %res, %ov[%m] : f32, view<[%m_rows]xf32>")
    out.append("    scf.yield %zero : f32")
    out.append("  } else {")
    out.append("    scf.yield %zero : f32")
    out.append("  }")
    out.append("  kernel.return")
    out.append("}")
    out.append("")
    out.append("// Small cases: 16 rows, hidden 5120.")
    out += emit_check("yah_swiglu_decode_case_a", "q4k", "iq4xs")
    out += emit_check("yah_swiglu_decode_case_b", "q3k", "q6k")
    out += emit_check("yah_swiglu_decode_case_c", "iq4nl", "q5k")
    out += emit_check("yah_swiglu_decode_case_d", "iq3s", "iq3s")
    out += emit_check("yah_swiglu_decode_case_e", "iq3xxs", "iq3xxs")
    out.append("// Production shape: intermediate_size 17408, hidden 5120, all-zero weights")
    out.append("// so out = silu(0)*0 = 0.")
    out.append("check.case public @yah_swiglu_decode_full_case {")
    out.append("  %gate_w = check.generate.fill value(0) : tensor<73113600xi8>")
    out.append("  %up_w = check.generate.fill value(0) : tensor<73113600xi8>")
    out.append("  %grid_s = check.generate.fill value(0) : tensor<512xi32>")
    out.append("  %grid_x = check.generate.fill value(0) : tensor<256xi32>")
    out.append("  %ksigns = check.generate.fill value(0) : tensor<128xi8>")
    out.append("  %x = check.generate.fill value(1.0) : tensor<5120xf32>")
    out.append("  %out = check.generate.fill value(0.0) : tensor<17408xf32>")
    out.append("  %expected = check.generate.fill value(0.0) : tensor<17408xf32>")
    out.append("  %t1 = check.literal value(1) : i32")
    out.append("  kernel.launch @yah_swiglu_decode(%gate_w, %up_w, %grid_s, %grid_x, %ksigns, %x, %out, %t1, %t1) : (tensor<73113600xi8>, tensor<73113600xi8>, tensor<512xi32>, tensor<256xi32>, tensor<128xi8>, tensor<5120xf32>, tensor<17408xf32>, i32, i32)")
    out.append("  check.expect.close actual(%out) expected(%expected) atol(0.0) rtol(0.0) nan(same) : tensor<17408xf32>")
    out.append("  check.return")
    out.append("}")
    out.append("")
    out.append("check.benchmark<@yah_swiglu_decode_full_case> @yah_swiglu_decode_full_bench")
    open(os.path.join(LOOM, "yah_swiglu_decode_f32.loom"), "w").write("\n".join(out) + "\n")
    print("wrote yah_swiglu_decode_f32.loom", len(out), "lines")


if __name__ == "__main__":
    main()