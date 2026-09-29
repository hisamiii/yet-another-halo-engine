#!/usr/bin/env python3
"""Emit every HAL the prefill forward needs for a GGUF shard.

Naming convention so the C++ driver needs no manifest:
  <outdir>/gemm_<kind>_<fmt>_<m_tiles>_<k_blocks>.hal
  <outdir>/<fixed>.hal   for the non-GEMM kernels
where kind is kstore|residual|swiglu.
"""
import os, re, shutil, struct, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Token-tile width for the emitted GEMM family. Prefill narrows to 16 (only kB
# real tokens are read); the decode path keeps the original 64.
TOKEN_TILE = int(os.environ.get("YAH_TOKEN_TILE", "16"))
# Diagnostic ablation: replace the decoded weight with a constant so the dequant
# and its weight reads vanish, leaving the MMA path.
ABLATE_DECODE = os.environ.get("YAH_ABLATE_DECODE", "0") == "1"
# "full" (default) = branchfree -> wave64 -> row groups -> LDS staging.
# "w64" = stop after the wave64 port. See _chain.
CHAIN_LEVEL = os.environ.get("YAH_CHAIN_LEVEL", "full")
LOOM = os.path.abspath(os.path.join(HERE, ".."))
EMIT = os.path.join(LOOM, "emit_hal.py")

# ggml type -> (short, port stem, qk)
FMT = {
    12: ("q4k", "q4k", 256), 13: ("q5k", "q5k", 256), 14: ("q6k", "q6k", 256),
    11: ("q3k", "q3k", 256), 23: ("iq4xs", "iq4xs", 256), 21: ("iq3s", "iq3s", 256),
    18: ("iq3xxs", "iq3xxs", 256), 20: ("iq4nl", "iq4nl", 32),
    17: ("iq2xs", "iq2xs", 256), 8: ("q8_0", "q8_0", 32),
    16: ("iq2xxs", "iq2xxs", 256), 10: ("q2k", "q2k", 256),
}
KSTORE = {"attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight",
          "ssm_beta.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
          "ffn_gate.weight"}
RESIDUAL = {"attn_output.weight", "ssm_out.weight", "ffn_down.weight"}
SWIGLU = {"ffn_up.weight"}


def parse(model):
    f = open(model, "rb"); f.read(8)
    nt, nkv = struct.unpack("<QQ", f.read(16))
    SIZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    def rd():
        n = struct.unpack("<Q", f.read(8))[0]; return f.read(n).decode("utf-8", "replace")
    def skip(t):
        if t == 8: rd()
        elif t == 9:
            et = struct.unpack("<I", f.read(4))[0]; n = struct.unpack("<Q", f.read(8))[0]
            if et == 8:
                for _ in range(n): rd()
            else: f.seek(SIZ[et] * n, 1)
        else: f.seek(SIZ[t], 1)
    for _ in range(nkv):
        rd(); t = struct.unpack("<I", f.read(4))[0]; skip(t)
    rows = []
    for _ in range(nt):
        nm = rd(); nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        ty = struct.unpack("<I", f.read(4))[0]
        f.read(8)
        rows.append((nm, dims, ty))
    return rows


def sym_of(loomfile):
    text = open(os.path.join(LOOM, loomfile)).read()
    return re.search(r"config\.decl @([A-Za-z0-9_]+)\.m_tiles", text).group(1)


def narrow_tokens(text):
    """Rewrite a GEMM kernel to a 16-wide token tile.

    The prefill pads kB real tokens to a 64-wide tile, so 3/4 of the rhs loads,
    MMAs and epilogue stores are waste. The element loop maps lane l to column
    l&15, and the epilogue decodes (row, token) with a shift/mask pair, so the
    same source narrows structurally: only n-group 0 survives and the token
    decode becomes 16 wide. Only tokens 0..kB-1 are ever read, so the driver and
    every other kernel are unchanged."""
    keep = []
    for line in text.split(chr(10)):
        if "%tokens = index.mul %token_tiles, %c64" in line or \
           "%token_base = index.mul %wg_y, %c64" in line:
            keep.append(line.replace("%c64", "%c16"))
        elif re.search(r"%rhs[123] = vector\.fragment\.load<rhs>", line):
            continue
        elif re.search(r"%n[123] = vector\.mma", line):
            continue
        elif line.strip().startswith("scf.yield %n0, %n1, %n2, %n3"):
            keep.append(line.replace("scf.yield %n0, %n1, %n2, %n3",
                                     "scf.yield %n0, %a1, %a2, %a3"))
        elif re.search(r"vector\.fragment\.store<result> %acc[123],", line):
            continue
        elif "scf.for %j2 = [%c0 to %c32 step %c1]" in line:
            keep.append(line.replace("%c32", "%c8"))
        elif "%r2_i = scalar.shrui %e2_i, %c6i" in line:
            keep.append(line.replace("%c6i", "%c4i"))
        elif "%tok_i = scalar.andi %e2_i, %c63i" in line:
            keep.append(line.replace("%c63i", "%c15i"))
        else:
            keep.append(line)
    return chr(10).join(keep)

def ablate_decode(text):
    keep = []
    for line in text.split(chr(10)):
        m = re.match(r"^(\s*)(%\w+) = scalar\.fptrunc %\w+ : f32 to f16$", line)
        if m:
            keep.append("%s%s = scalar.constant 1.0 : f16" % (m.group(1), m.group(2)))
        else:
            keep.append(line)
    return chr(10).join(keep)


def direct_residual_epilogue(text):
    """Store the result fragments straight into the token-major output.

    The residual arm staged every 16x16 result fragment through a global ostage
    buffer and then read it back with a scalar transposed copy. That round trip is
    (a) pure overhead -- the lds3 arm showed 1.25x on the kStore -- and (b) the
    only cross-workgroup shared memory in the kernel, which is what makes the arm
    nondeterministic once the grid has more than one token tile (see project
    memory: two runs at 128 tokens differ in exactly this buffer).

    The replacement addresses the output through a strided token-major view:
    view<[m_rows]x[k_split*tokens]> with layout strided [1, m_rows], i.e. element
    (row, col) at row + col*m_rows. The split is folded into the column index, so
    (split, token, row) lands at m_origin + (split*tokens + token_base)*m_rows --
    the same address the readback loop computed. accum=1 is preserved by loading
    the existing tile into the accumulator instead of adding it after the fact.
    """
    lines = text.split(chr(10))
    if not any("%ostage_view = buffer.view" in l for l in lines):
        return text

    view_ty = "view<[%stage_rows_split]x[%tokens]xf32>"
    new_ty = "view<[%m_rows]x[%out_cols]xf32, %out_layout>"
    prologue = [
        "  %out_layout = encoding.layout.strided [%c1, %m_rows] : encoding<layout>",
        "  %out_cols = index.mul %k_split, %tokens : index",
        "  %out_t_view = buffer.view %output_na[%base] : buffer -> view<[%m_rows]x[%out_cols]xf32, %out_layout>",
        "  %split_tok = index.mul %split, %tokens : index",
        "  %kcol0 = index.add %split_tok, %token_base : index",
        "  %kcol1 = index.add %kcol0, %c16 : index",
        "  %kcol2 = index.add %kcol0, %c32 : index",
        "  %kcol3 = index.add %kcol0, %c48 : index",
    ]
    out = []
    i = 0
    n = len(lines)
    while i < n:
        l = lines[i]
        if "%ostage_view = buffer.view" in l:
            out.append(l)
            out.extend(prologue)
            i += 1
            continue
        if l.strip().startswith("vector.fragment.store<result> %acc0, %ostage_view["):
            # the four sub-tile stores, in order
            for k in range(4):
                src = lines[i + k]
                assert "%ostage_view[" in src and ("%%acc%d" % k) in src, src
                out.append("  vector.fragment.store<result> %%acc%d, %%out_t_view[%%m_origin, %%kcol%d] "
                           "shape [%%m, %%n] : vector<8xf32>, %s" % (k, k, new_ty))
            # the barrier and the readback loop that followed
            j = i + 4
            while lines[j].strip() != "kernel.return":
                j += 1
            i = j
            continue
        if l.strip().startswith("%acc0, %acc1, %acc2, %acc3 = scf.for"):
            out.append("  %is_acc = index.cmp eq, %accum, %c1 : index")
            for k in range(4):
                out.append("  %%init%d = scf.if %%is_acc -> (vector<8xf32>) {" % k)
                out.append("    %%il%d = vector.fragment.load<result> %%out_t_view[%%m_origin, "
                           "%%kcol%d] shape [%%m, %%n] : %s -> vector<8xf32>" % (k, k, new_ty))
                out.append("    scf.yield %%il%d : vector<8xf32>" % k)
                out.append("  } else {")
                out.append("    scf.yield %init : vector<8xf32>")
                out.append("  }")
            repl = l
            for k in range(4):
                repl = repl.replace("%%a%d = %%init" % k, "%%a%d = %%init%d" % (k, k))
            out.append(repl)
            i += 1
            continue
        out.append(l)
        i += 1
    return chr(10).join(out)


def direct_kstore_epilogue(text):
    """The same direct token-major store for the f32 kStore arm.

    Same defect and same fix as the residual (see direct_residual_epilogue): the
    result fragments went through a global ostage buffer whose row stride is
    %tokens, then came back through a scalar transposed copy. The kStore has no
    k_split and no accum flag, so the output columns are just token_base + {0,16,32,48}.
    """
    import re
    lines = text.split(chr(10))
    if not any("%ostage_view = buffer.view" in l for l in lines):
        return text
    if any("%k_split" in l for l in lines):
        return text
    store_re = re.compile(r"vector\.fragment\.store<result> %acc(\d), %ostage_view\[([^,]+), ([^\]]+)\]")
    new_ty = "view<[%m_rows]x[%tokens]xf32, %out_layout>"
    prologue = [
        "  %out_layout = encoding.layout.strided [%c1, %m_rows] : encoding<layout>",
        "  %out_t_view = buffer.view %output_na[%base] : buffer -> view<[%m_rows]x[%tokens]xf32, %out_layout>",
        "  %epc16 = index.constant 16 : index",
        "  %epc32 = index.constant 32 : index",
        "  %epc48 = index.constant 48 : index",
        "  %tk1 = index.add %token_base, %epc16 : index",
        "  %tk2 = index.add %token_base, %epc32 : index",
        "  %tk3 = index.add %token_base, %epc48 : index",
    ]
    out = []
    i = 0
    n = len(lines)
    while i < n:
        l = lines[i]
        if "%ostage_view = buffer.view" in l:
            out.append(l)
            out.extend(prologue)
            i += 1
            continue
        if l.strip().startswith("vector.fragment.store<result> %acc0, %ostage_view["):
            # Do NOT reuse the source's column operands. Two conventions exist in
            # this tree: the newer ports stage a [stage_rows]x[%tokens] view at
            # column %token_base, and the older ones (q5k, q3k, q2k, q6k, q8_0,
            # iq2xs...) stage a tile-local [stage_rows]x64 view at columns
            # 0/16/32/48. A tile-local column is wrong for the token-major output
            # as soon as there is more than one token tile -- both tiles write the
            # same 64 columns -- which is bit-identical at token_tiles=1 and racy
            # and wrong above it. Always rebuild the columns from %token_base.
            cols = ["%token_base", "%tk1", "%tk2", "%tk3"]
            for k in range(4):
                m = store_re.search(lines[i + k])
                assert m and int(m.group(1)) == k, lines[i + k]
                out.append("  vector.fragment.store<result> %%acc%d, %%out_t_view[%s, %s] "
                           "shape [%%m, %%n] : vector<8xf32>, %s" % (k, m.group(2), cols[k], new_ty))
            j = i + 4
            while lines[j].strip() != "kernel.return":
                j += 1
            i = j
            continue
        out.append(l)
        i += 1
    return chr(10).join(out)


def local_ostage_epilogue(text):
    """Stage the swiglu epilogue in a tile-local ostage tile.

    The swiglu arm cannot store its fragments straight to the output: the epilogue
    applies silu(gate)*x elementwise, so it needs the tile back in a scalar form.
    What it does NOT need is a staging view whose row stride is %tokens -- the tile
    a workgroup stages is always 16 rows by 64 columns, so the view is
    [stage_rows]x[%c64] and the columns are the tile-local 0/16/32/48. The readback
    then reads column %tok (the local token) while the output index still uses the
    global token. Side effect: the staging footprint stops growing with the prompt
    (stage_rows*64*4 instead of stage_rows*tokens*4).
    """
    import re
    lines = text.split(chr(10))
    if not any("%ostage_view = buffer.view" in l for l in lines):
        return text
    cols = ["%c0", "%c16", "%c32", "%c48"]
    store_re = re.compile(r"(vector\\.fragment\\.store<result> %acc(\\d), %ostage_view\\[)([^,]+), ([^\\]]+)(\\].*)view<\\[%stage_rows\\]x\\[%tokens\\]xf32>")
    out = []
    for l in lines:
        if "%ostage_view = buffer.view" in l:
            out.append(l.replace("view<[%stage_rows]x[%tokens]xf32>", "view<[%stage_rows]x[%c64]xf32>"))
            continue
        if l.strip().startswith("vector.fragment.store<result> %acc") and "%ostage_view[" in l:
            k = int(l.split("%acc", 1)[1][0])
            head, rest = l.split("%ostage_view[", 1)
            row, rest2 = rest.split(",", 1)
            tail = rest2.split("]", 1)[1].replace("view<[%stage_rows]x[%tokens]xf32>", "view<[%stage_rows]x[%c64]xf32>")
            out.append(head + "%ostage_view[" + row + ", " + cols[k] + "]" + tail)
            continue
        if "%val = view.load %ostage_view[" in l and ", %tok_g]" in l:
            out.append(l.replace(", %tok_g]", ", %tok]")
                        .replace("view<[%stage_rows]x[%tokens]xf32>", "view<[%stage_rows]x[%c64]xf32>"))
            continue
        out.append(l)
    return chr(10).join(out)


def deltanet_lds_rewrite(text):
    """Stage the DeltaNet state in LDS instead of re-reading it from global.

    Measured on the scaled case with the source's own exact expectations as the
    gate: the per-token state store in the update loop is 81% of the kernel
    (19.30 -> 3.57 ms when the store alone is removed), the state load a further
    33% (-> 12.85 ms), the readout 17%. Unrolling 2/4/8 is neutral-to-worse and
    transposing the layout so a wave's accesses coalesce is worth 13%, so the cost
    is the per-iteration global round trip, not issue, bandwidth or line count.

    The register-resident form (one state row per lane in 128 registers, which is
    what the HIP reference does) does not compile on this target: 128 named values
    each need their own address value and amdgpu.sgpr runs out (budget 106, peak
    401, 'spill-traffic-register-exhausted').

    THE LAUNCH SHAPE MUST NOT CHANGE. The obvious way to fit the 128x128 f32 block
    (65536 B, the gfx11 per-workgroup LDS limit) is two 64-lane workgroups per
    head, which is what this kernel's own launch config declares and what
    iree-benchmark-loom honours. engine/run/loom_forward_pp.cc does NOT: it passes
    the grid explicitly as Dispatch(..., kTs, 1, 1, 128, 1, 1, b) and takes only
    the workgroup SIZE from the HAL metadata, so a 2x grid silently never launches
    and only the first half of the 48 heads is computed -- a wrong-but-deterministic
    forward, which is exactly how this was found. So: grid = num_heads, 128 lanes,
    one lane per state row, exactly as the original.

    The block is held COLUMN-major (sl[i*128 + row]) so that a wave's 32 lanes
    always touch 32 consecutive floats -- conflict-free in the staging loop, the
    two token-loop reads and the write-back without any padding. Same f32
    operations in the same order, so the results are bit-identical: the B=64
    single-workgroup-per-row-group output is byte-identical with and without this
    pass.

    19.166 -> 5.916 ms on the scaled case (3.24x), gate exact both ways.
    """
    subs = [
        ("  %state_row = index.add %state_base0, %row_k : index",
         "  %state_row = index.add %state_base0, %row_k : index\n"
         "  %dl16384 = index.constant 16384 : index\n"
         "  %dlbytes = index.constant 65536 : offset\n"
         "  %dll = buffer.alloca<workgroup> align(16) %dlbytes : buffer\n"
         "  %dls = buffer.view %dll[%base] : buffer -> view<[%dl16384]xf32>\n"
         "  %dlstage = scf.for %si = [%c0 to %c128 step %c1](%sm = %zero : f32) -> (f32) {\n"
         "    %dlg = index.add %state_row, %si : index\n"
         "    %dlsi = index.mul %si, %c128 : index\n"
         "    %dllidx = index.add %dlsi, %row : index\n"
         "    %dlv = view.load %state_view[%dlg] : view<[%state_total]xf32> -> f32\n"
         "    view.store %dlv, %dls[%dllidx] : f32, view<[%dl16384]xf32>\n"
         "    scf.yield %sm : f32\n"
         "  }"),
        ("      %s_idx = index.add %state_row, %i : index\n"
         "      %k_idx = index.add %k_off, %i : index\n"
         "      %q_idx = index.add %q_off, %i : index\n"
         "      %s = view.load %state_view[%s_idx] : view<[%state_total]xf32> -> f32",
         "      %dli128 = index.mul %i, %c128 : index\n"
         "      %s_idx = index.add %dli128, %row : index\n"
         "      %k_idx = index.add %k_off, %i : index\n"
         "      %q_idx = index.add %q_off, %i : index\n"
         "      %s = view.load %dls[%s_idx] : view<[%dl16384]xf32> -> f32"),
        ("      %js_idx = index.add %state_row, %j : index\n"
         "      %jk_idx = index.add %k_off, %j : index\n"
         "      %sj = view.load %state_view[%js_idx] : view<[%state_total]xf32> -> f32",
         "      %dlj128 = index.mul %j, %c128 : index\n"
         "      %js_idx = index.add %dlj128, %row : index\n"
         "      %jk_idx = index.add %k_off, %j : index\n"
         "      %sj = view.load %dls[%js_idx] : view<[%dl16384]xf32> -> f32"),
        ("      view.store %sj_new, %state_view[%js_idx] : f32, view<[%state_total]xf32>",
         "      view.store %sj_new, %dls[%js_idx] : f32, view<[%dl16384]xf32>"),
        ("  kernel.return",
         "  %dlunstage = scf.for %ui = [%c0 to %c128 step %c1](%um = %zero : f32) -> (f32) {\n"
         "    %dlug = index.add %state_row, %ui : index\n"
         "    %dlui = index.mul %ui, %c128 : index\n"
         "    %dlulidx = index.add %dlui, %row : index\n"
         "    %dluv = view.load %dls[%dlulidx] : view<[%dl16384]xf32> -> f32\n"
         "    view.store %dluv, %state_view[%dlug] : f32, view<[%state_total]xf32>\n"
         "    scf.yield %um : f32\n"
         "  }\n"
         "  kernel.return"),
    ]
    for old, new in subs:
        if text.count(old) != 1:
            raise SystemExit("deltanet_lds_rewrite: anchor %d times: %s"
                             % (text.count(old), old.strip()[:60]))
        text = text.replace(old, new)
    return text

def _chain(text, tile, n_row, loomfile='', level=None):
    """The measured-best kStore rebuild: branchfree -> wave64 -> n_row row groups.

    Every step fails loudly on a structural anchor it cannot find, so a caller
    can probe it. See emit(..., chain=True) for why the launch geometry this
    produces has to travel with the HAL.
    """
    import branchfree_decode as bf
    import wave64_tokens as w64
    import widen_rows as wr
    # The residual and swiglu siblings decode IQ3_S with the old one-column-per-
    # lane mapping, which widen_rows cannot address. Transplanting the kStore's
    # word body into them is a frame-preserving region swap (see
    # port_word_decode), and the arithmetic is identical, so their outputs stay
    # bit-identical.
    # Opt-in: the swiglu's two-operand silu(gate)*up accumulator set does not match
    # widen_rows' single %acc0..%acc{n-1} rewrite yet (it compiles to an undefined
    # %acc8), and the residual's ostage view still fails the LDS anchor. Until both
    # are handled the port stays out of the default path.
    # Both siblings compile with the transplant (round 8). The earlier "undefined
    # %acc8" was not a two-accumulator problem: it was a duplicate SSA name from
    # the transplant, which aborts the parse of the enclosing loop and makes every
    # one of that loop's results read as undefined.
    # OFF by default: the transplant COMPILES (round 8, after port_word_decode was
    # taught to absorb the target's colliding definitions -- the "undefined %acc8"
    # was one duplicate SSA name aborting the parse of the enclosing loop, not a
    # two-accumulator problem), but it is NUMERICALLY WRONG: with it enabled the
    # B=128 forward argmax is 279 instead of 11751. The word body's row/column
    # decode does not land in the same LDS cells the siblings' lhs fragment reads,
    # so this needs a semantic port, not a region swap. Re-enable only with a
    # bit-identity gate on each source.
    _base = os.path.basename(loomfile)
    _want = os.environ.get("YAH_WORD_PORT", "")
    _is_rs = _base.startswith('yah_ffn_gemm_iq3s_residual')
    _is_sw = _base.startswith('yah_ffn_gemm_iq3s_swiglu')
    # The SWIGLU transplant is default-on and verified: B=128 hidden byte-identical
    # to the pre-transplant build, B=2048 deterministic across runs, argmax 11751 ==
    # HIP and logits-vs-HIP identical to the baseline (corr 0.999996809, max|d|
    # 0.0403244) -- the word decode is the same arithmetic in a different lane order.
    # The RESIDUAL is still wrong (combined argmax 279) and stays behind
    # YAH_WORD_PORT=residual until its own gate passes.
    # YAH_WORD_PORT=off emits the pre-transplant geometry, which is what a paired
    # A/B against the current build needs.
    # The source test must come FIRST. Written the other way round, _want == "1"
    # was true for every source, so the port was applied to the whole GEMM family
    # and injected the IQ3_S word body into e.g. q3k (undefined %c66i) -- which is
    # also why the probe results looked env-dependent.
    # Default (no env) ports the swiglu only: that is the verified shipped state.
    # Every branch is scoped to a sibling -- an unscoped _want == "1" applied the
    # port to the whole GEMM family (it injected the IQ3_S word body into q3k,
    # giving undefined %c66i, which also made the probes look env-dependent).
    # HARD GATE (round 16): porting the IQ3_S word body into the RESIDUAL is
    # numerically wrong and is now unreachable in every selector form. Proven
    # this round: the residual's copy loop, %tokens/%m_tiles/%m_rows/%stage_rows/
    # %token_base, its o16/o32/o48 column offsets and its copy-loop lane map
    # (tok = e2 & 15, r2 = e2 >> 4) are ALL byte-identical to the verified kStore,
    # and k_split defaults to 1 (config.def) so split = wg.z = 0, k_off = 0,
    # k_end = k_blocks -- the split path is inert. The only residual-unique feature
    # left is the accum branch. With the port ON, B=128 forward argmax is 279
    # instead of 11751; with it OFF (the shipped geometry, which still gets
    # wave64 + widen_rows + LDS staging) the build is byte-identical. So the
    # defect is confined to the port's row/column decode: it does not land in the
    # same LDS cells the residual's lhs fragment reads. A frame-preserving region
    # swap is not a semantic port. This must stay unreachable until someone writes
    # a real semantic port gated by the residual's own value oracle -- there is no
    # check.case runner in-tree (safe_bench.py is a footprint gate, not an oracle),
    # so that gate does not exist yet.
    #
    # RuntimeError, NOT SystemExit: chain_applies() catches SystemExit and would
    # silently fall back to emitting an UNCHAINED residual instead of refusing.
    if _is_rs and _want in ("1", "residual"):
        raise RuntimeError(
            "YAH_WORD_PORT=%r would port the IQ3_S word body into %s, which is "
            "known-wrong (B=128 argmax 279 vs 11751) and disabled by the round-16 "
            "gate in emit_prefill._chain. Leave YAH_WORD_PORT unset (swiglu only) "
            "or use YAH_WORD_PORT=off." % (_want, _base))
    _sel = (((_want == "1") and _is_sw)
            or ((_want in ("", "swiglu")) and _is_sw))
    if _want != "off" and _sel:
        import port_word_decode as pwd
        text = pwd.port(text, tile)
    if 'scf.if %wd_old' in text:
        text = bf.drop_branch(text, 'word')
    # 'rows' stops short of the wave64 port: the row-group rewrite below works on
    # the wave32 one-column-per-lane map directly (j in [0,32) puts row =
    # (lane>>4) + 2j across all 64 rows of a 4-row-group tile), and wave64 measured
    # SLOWER at B=2048, so bundling it in would hide the row-group result.
    if (level or CHAIN_LEVEL) != 'rows':
        text = w64.wave64(text, tile)
    # YAH_CHAIN_LEVEL=w64 stops here, after the wave64 port. The row-group and LDS
    # staging steps assume the iq3s decode's lane map (lane>>2 spans ROWS_PER_PASS
    # rows, one row per lane over a 64-row tile); the one-column-per-lane formats
    # (iq3xxs/iq4xs) need a decode rewrite before either of them can apply. wave64
    # alone is exact for them -- same elements, same addresses, 64 lanes -- so it
    # is worth shipping on its own while the rest of the port is built.
    # The level is per-source, not global: a caller must keep the FULL chain on the
    # sources that support it, or the emit silently drops their row groups.
    if (level or CHAIN_LEVEL) == 'w64':
        # q5k is EXCLUDED: wave64 produces a numerically wrong kernel for it.
        # Bisected format-by-format at B=128 against argmax 11751 -- iq2xs,
        # iq2xxs, iq3s, iq3xxs, iq4xs, q2k, q3k, q4k, q6k and q8_0 all transform
        # correctly; q5k alone gives 88. Its decode lane map is textually
        # identical to q4k's (same [0,8) loop, same shli-by-5, same
        # lane&15 / e>>4 mapping), and q5k is correct under token widening alone,
        # so the fault is specific to the wave64 step and is not yet understood.
        # Fail loudly rather than emit a silently wrong q5k.
        if 'q5k' in os.path.basename(loomfile):
            raise SystemExit(
                'wave64: q5k is known-wrong (B=128 argmax 88 vs 11751); excluded')
        return text
    text = wr.widen_rows(text, n_row)
    if (level or CHAIN_LEVEL) == 'rows':
        # TRANSFORM COMPLETE, KERNEL DOES NOT YET COMPILE. widen_rows now supports
        # the one-column-per-lane map (row = (lane>>4) + 2j over j in [0,32)), so
        # this produces the right row-group form for iq3xxs/iq4xs/q4k/q3k/q6k.
        # But emit_hal.py inlines configs as constants, and the resulting
        # scattered GLOBAL byte gathers fail the non-negativity proof:
        #   error [SUBRANGE/023]: view.load footprint origin lower bound is not
        #   proven on view axis 0 (the %qlo/%qh/%sg/%sc %w_view loads)
        # The LDS staging step (lds_stage_iq3s) is what makes those loads provable,
        # which is why the chain has always bundled it. Porting that step to the
        # 98-byte IQ3_XXS block is the remaining work: without it this level
        # cannot emit. Failure is loud (the emit aborts); it is not a silent
        # wrong kernel.
        return text
    # Last, and only for the shape the steps above produce: stage the IQ3_S
    # weight block in LDS. That transform is format-specific (110-byte block) and
    # lives in its own generator, which is run as a filter so its proven
    # substitution list stays verbatim. It is also what lets emit_hal.py accept
    # the kernel at all -- with the configs inlined as constants the scattered
    # global byte-gather indices lose their non-negativity proof.
    import subprocess
    import tempfile
    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lds_stage_iq3s.py")
    with tempfile.NamedTemporaryFile("w", suffix=".loom", delete=False) as fh:
        fh.write(text)
        tmp = fh.name
    r = subprocess.run([sys.executable, tool, tmp], capture_output=True, text=True)
    os.unlink(tmp)
    if r.returncode != 0:
        raise SystemExit("lds_stage_iq3s failed: " + r.stderr.strip()[-160:])
    return r.stdout

    hal = r.stdout.strip().splitlines()[-1]
    dst = os.path.join(outdir, outname)
    subprocess.run(["cp", hal, dst], check=True)
    return dst


def chain_applies(loomfile, tile=None, n_row=None, level=None):
    """Can the wave64/row-group chain rebuild this source?

    Only the kStore family that carries the packed word decode has the lane map
    widen_rows assumes; the other GEMM sources use a decode whose row pass it
    cannot address, so they stay on the shipping geometry. Probing the real
    transform is what makes that distinction safe: it raises on any anchor it
    does not recognise.
    """
    if not os.path.basename(loomfile).startswith("yah_ffn_gemm_"):
        return False
    tile = tile or TOKEN_TILE
    n_row = n_row or int(os.environ.get("YAH_ROWGRP", "4"))
    # The probe must see the SAME text emit() will transform: emit() widens the
    # token tile before calling _chain, and probing the raw source instead gave
    # the residual a false negative for two rounds.
    text = open(os.path.join(LOOM, loomfile)).read()
    if tile and tile != 64:
        import widen_tokens as W
        origin = "%m_origin_s" if "%m_origin_s" in text else "%m_origin"
        dims = "[%stage_rows_split]" if "%stage_rows_split" in text else "[%stage_rows]"
        text = W.widen(text, tile // 16, m_origin=origin, stage_dims=dims)
    try:
        _chain(text, tile, n_row, loomfile, level)
        return True
    except SystemExit:
        return False


def emit(loomfile, configs, outname, outdir, widen=0, chain=False, chain_level=None):
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(LOOM, loomfile)
    if os.environ.get("YAH_DELTANET_LDS") == "1" and "yah_deltanet_rowsplit" in loomfile:
        text = open(src).read()
        src_tmp = os.path.join(tmp, os.path.basename(loomfile))
        with open(src_tmp, "w") as fh:
            fh.write(deltanet_lds_rewrite(text))
        src = src_tmp
    if os.path.basename(loomfile).startswith("yah_ffn_gemm_"):
        text = open(src).read()
        if TOKEN_TILE == 16:
            text = narrow_tokens(text)
        # Full-prompt emission wants the source's own (or a wider) token tile
        # instead of the 5-token narrowing, so widen_tokens.py is the inverse
        # step here. The two are mutually exclusive.
        if widen and widen != 64 and TOKEN_TILE != 16:
            import widen_tokens as W
            origin = "%m_origin_s" if "%m_origin_s" in text else "%m_origin"
            dims = "[%stage_rows_split]" if "%stage_rows_split" in text else "[%stage_rows]"
            text = W.widen(text, widen // 16, m_origin=origin, stage_dims=dims)
        # chain=True rebuilds the source as the measured-best arm: branchfree ->
        # wave64 -> n_row row groups per workgroup -> IQ3_S block staging in LDS.
        # The launch geometry changes with the row group count, so the resolved
        # geometry travels with the HAL (emit_prefill_pp writes dispatch.txt).
        if chain:
            text = _chain(text, widen, int(os.environ.get("YAH_ROWGRP", "4")), loomfile,
                          chain_level)
        # YAH_SIMPLE_FILL=1 drops the unroll/schedule annotation from the LDS fill.
        if os.environ.get("YAH_SIMPLE_FILL") == "1":
            text = text.replace("unroll(%c2) schedule(interleaved)", "")
        # YAH_DIRECT_EPI=1 removes the residual's global ostage round trip. The
        # chain rebuilds the epilogue itself, so the two are per-source alternatives.
        if os.environ.get("YAH_DIRECT_EPI") == "1" and not chain:
            if "_residual_" in loomfile:
                text = direct_residual_epilogue(text)
            elif loomfile.endswith("_f32.loom"):
                text = direct_kstore_epilogue(text)
        if ABLATE_DECODE:
            text = ablate_decode(text)
        src_tmp = os.path.join(tmp, os.path.basename(loomfile))
        with open(src_tmp, "w") as fh:
            fh.write(text)
        src = src_tmp
    r = subprocess.run([sys.executable, EMIT, src, tmp] + configs,
                       capture_output=True, text=True)
    if r.returncode != 0:
        # Surface the ORDERED HEAD of the failure: assembling the message as
        # stdout+stderr pushes the primary diagnostic out of view (the emit helper
        # writes progress without a trailing newline, so the old message began
        # mid-token) and three rounds were lost reading the repeated tail symptom.
        # stderr carries the compiler diagnostics, so it comes first.
        log = os.path.join("/tmp", "emit_fail_" + os.path.basename(outname) + ".log")
        with open(log, "w") as fh:
            fh.write("cmd: %s" % " ".join([sys.executable, EMIT, src, tmp] + list(configs)))
            fh.write(chr(10) + "--- stderr ---" + chr(10) + r.stderr
                     + chr(10) + "--- stdout ---" + chr(10) + r.stdout)
        head = (r.stderr.strip() + chr(10) + r.stdout.strip()).strip().splitlines()[:24]
        raise SystemExit("emit failed for " + loomfile + " (full log: " + log + "):"
                         + (chr(10) + "  ").join(head))
    hal = r.stdout.strip().splitlines()[-1]
    dst = os.path.join(outdir, outname)
    subprocess.run(["cp", hal, dst], check=True)
    return dst


def main():
    model, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    rows = parse(model)
    combos = set()
    for nm, dims, ty in rows:
        suffix = nm.split(".", 2)[2] if nm.startswith("blk.") else nm
        info = FMT.get(ty)
        if not info: continue
        fmt, port, qk = info
        if suffix in KSTORE: kind = "kstore"
        elif suffix in RESIDUAL: kind = "residual"
        elif suffix in SWIGLU: kind = "swiglu"
        else: continue
        mt, kb = dims[1] // 16, dims[0] // qk
        combos.add((kind, fmt, port, mt, kb))
    n = 0
    for kind, fmt, port, mt, kb in sorted(combos):
        if kind == "kstore":
            f = f"yah_ffn_gemm_{port}_f32.loom"; sym = sym_of(f)
            cfg = [f"{sym}.m_tiles={mt}", f"{sym}.k_blocks={kb}", f"{sym}.token_tiles=1"]
            if port == "iq3s":
                # The four-elements-per-lane word decode is a clear in-situ win at
                # the attention geometries (m_tiles 384/640/768) and an in-situ loss
                # at m_tiles=1088, measured against a paired full-pipeline control
                # with bit-identical output. The isolated hal_bench numbers invert
                # the ordering, so the selection is per geometry, not global; the
                # kernel folds the config away, so each HAL is one of the two exact
                # machine codes. See engine/run/LOOM_RUNTIME.md.
                cfg.append(f"{sym}.word_decode={0 if mt == 1088 else 1}")
            emit(f, cfg, f"gemm_kstore_{fmt}_{mt}_{kb}.hal", outdir); n += 1
        elif kind == "residual":
            f = f"yah_ffn_gemm_{port}_residual_f32.loom"; sym = sym_of(f)
            # Every residual arm has m_tiles=320 (the projection writes hidden),
            # so a 4-way K split raises the grid from 320 to 1280 workgroups.
            emit(f, [f"{sym}.m_tiles={mt}", f"{sym}.k_blocks={kb}",
                     f"{sym}.token_tiles=1", f"{sym}.k_split=4", f"{sym}.accum=0"],
                 f"gemm_residual_{fmt}_{mt}_{kb}.hal", outdir); n += 1
        else:
            f = f"yah_ffn_gemm_{port}_swiglu_f16.loom"; sym = sym_of(f)
            emit(f, [f"{sym}.m_tiles={mt}", f"{sym}.k_blocks={kb}", f"{sym}.token_tiles=1"],
                 f"gemm_swiglu_{fmt}_{mt}_{kb}.hal", outdir); n += 1
    # The residual reduction: hidden += sum_s partial[s], dim = hidden * 64 tokens.
    emit("yah_residual_add_1d_f32.loom", ["yah_residual_1d.dim=%d" % (5120 * TOKEN_TILE)],
         "accum.hal", outdir); n += 1
    # The fixed prefill kernels at the shard's shapes.
    fixed = [
        ("yah_half_norm_f16.loom", "norm.hal",
         ["yah_half_norm.rows=5", "yah_half_norm.dim=5120", "yah_half_norm.eps=1e-06"]),
        ("yah_ssm_conv_f32.loom", "conv.hal",
         ["yah_ssm_conv.batch=5", "yah_ssm_conv.qkv_dim=10240"]),
        ("yah_deltanet_prep_kq_f32.loom", "prepkq.hal",
         ["yah_deltanet_prep_kq.batch=5", "yah_deltanet_prep_kq.num_key_heads=16",
          "yah_deltanet_prep_kq.qkv_size=10240"]),
        ("yah_deltanet_prep_ab_f32.loom", "prepab.hal",
         ["yah_deltanet_prep_ab.batch=5", "yah_deltanet_prep_ab.qkv_size=10240",
          "yah_deltanet_prep_ab.num_heads=48"]),
        ("yah_deltanet_rowsplit_f32.loom", "rowsplit.hal",
         ["yah_deltanet.batch=5", "yah_deltanet.qkv_size=10240",
          "yah_deltanet.inner_size=6144", "yah_deltanet.num_key_heads=16",
          "yah_deltanet.num_heads=48"]),
        ("yah_ssm_postnorm_gate_f16.loom", "postnorm.hal",
         ["yah_ssm_postnorm_fp16.head_count=240"]),
        ("yah_unpack_qg_f32.loom", "unpack.hal",
         ["yah_unpack_qg.batch=5", "yah_unpack_qg.num_heads=24",
          "yah_unpack_qg.head_dim=256"]),
        ("yah_fused_qk_rope_batched_f32.loom", "rope.hal", [
            "yah_fused_qk_rope_batched.start_pos=0",
            "yah_fused_qk_rope_batched.batch=5",
            "yah_fused_qk_rope_batched.layer_idx=0",
            "yah_fused_qk_rope_batched.max_context=8",
            "yah_fused_qk_rope_batched.num_heads=24",
            "yah_fused_qk_rope_batched.num_kv_heads=4",
            "yah_fused_qk_rope_batched.head_dim=256",
            "yah_fused_qk_rope_batched.rotary_dim=64",
            "yah_fused_qk_rope_batched.q_elems=30720",
            "yah_fused_qk_rope_batched.kv_elems=5120",
            "yah_fused_qk_rope_batched.cache32_elems=8192",
            "yah_fused_qk_rope_batched.cache16_elems=8192"]),
        ("yah_attn_wmma_f32.loom", "wmma.hal", [
            "yah_attn_wmma.layer_idx=0", "yah_attn_wmma.start_pos=0",
            "yah_attn_wmma.batch_size=5", "yah_attn_wmma.max_context=8",
            "yah_attn_wmma.num_heads=24", "yah_attn_wmma.num_kv_heads=4",
            "yah_attn_wmma.head_dim=256", "yah_attn_wmma.gqa=6",
            "yah_attn_wmma.score_capacity=8", "yah_attn_wmma.kv_padded=8",
            "yah_attn_wmma.has_gate=1", "yah_attn_wmma.has_lse=0",
            "yah_attn_wmma.head_major=0"]),
        ("yah_half_cast.loom", "cast.hal", ["yah_half_cast.num_elements=30720"]),
        ("yah_rmsnorm_f32.loom", "rmsnorm.hal",
         ["yah_rmsnorm.rows=1", "yah_rmsnorm.eps=1e-06"]),
        ("yah_gemv_q6k_f32.loom", "gemv.hal",
         ["yah_gemv_q6k.m_rows=248320", "yah_gemv_q6k.k_blocks=20"]),
        ("yah_argmax_f32.loom", "argmax.hal", ["yah_argmax.vocab=248320"]),
    ]
    for loom, outname, configs in fixed:
        emit(loom, configs, outname, outdir); n += 1
    # The IQ grid and sign tables, committed under loom/tables/.
    tables = os.path.join(LOOM, "tables")
    for src, dst in [("grid_iq3s.bin", "grid_iq3s.bin"),
                     ("grid_iq3xxs.bin", "grid_iq3xxs.bin"),
                     ("grid_iq2xxs.bin", "grid_iq2xxs.bin"),
                     ("grid_iq2xs.bin", "grid_iq2xs.bin"),
                     ("ksigns_iq2xs.bin", "ksigns_iq3xxs.bin"),
                     ("ksigns_iq2xs.bin", "ksigns_iq2xxs.bin")]:
        shutil.copy(os.path.join(tables, src), os.path.join(outdir, dst))
    print("emitted", n, "GEMM + fixed prefill HALs")


if __name__ == "__main__":
    main()