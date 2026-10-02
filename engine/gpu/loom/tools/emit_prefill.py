#!/usr/bin/env python3
"""Helpers for the HAL emitters (emit_prefill_pp.py, emit_decode.py): the GGUF
tensor table, the format map and emit() (one Loom source at one config -> one HAL).

Naming convention so the C++ driver needs no manifest:
  <outdir>/gemm_<kind>_<fmt>_<m_tiles>_<k_blocks>.hal
  <outdir>/<fixed>.hal   for the non-GEMM kernels
"""
import os, re, struct, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
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

def emit(loomfile, configs, outname, outdir):
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(LOOM, loomfile)
    # DEFAULT ON. The shipped HAL set was emitted with this rewrite and it is worth
    # 1.53x END-TO-END at B=2048: interleaved, 15 s gaps, sets differing in nothing
    # but rowsplit.hal -- LDS 21862.2/22773.3 ms vs non-LDS 33314.6/35351.8 ms,
    # argmax 11751 on all four runs. But the gate defaulted OFF, so a plain re-emit
    # (the env documented for reproducing the shipped set) silently produced the
    # SLOW variant and lost 11.5 s. YAH_DELTANET_LDS=0 restores that old default.
    #
    # Superseded by DEFAULT: rowsplit.hal is built from yah_deltanet_regtile_f32.loom
    # (tools/gen_deltanet_regtile.py), which carries each lane's state row in
    # registers across the token loop instead of round-tripping it through LDS.
    # Same ABI, grid and f32 operation order: at B=2048, interleaved, 15 s gaps,
    # sets differing only in rowsplit.hal, LDS 9844.0/9917.4 ms vs regtile
    # 7722.7/7747.2 ms, argmax 11751 and hidden f837e614ff55d1d1 on all four.
    # YAH_DELTANET=lds selects the LDS form below instead.
    if "yah_deltanet_rowsplit" in loomfile and os.environ.get("YAH_DELTANET", "regtile") == "regtile":
        src = os.path.join(LOOM, "yah_deltanet_regtile_f32.loom")
    elif os.environ.get("YAH_DELTANET_LDS", "1") != "0" and "yah_deltanet_rowsplit" in loomfile:
        text = open(src).read()
        src_tmp = os.path.join(tmp, os.path.basename(loomfile))
        with open(src_tmp, "w") as fh:
            fh.write(deltanet_lds_rewrite(text))
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
