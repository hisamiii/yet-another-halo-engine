#!/usr/bin/env python3
"""Helpers for the HAL emitters (emit_prefill_pp.py, emit_decode.py): the GGUF tensor table, the format map and emit().

Naming convention, so the C++ driver needs no manifest:
  <outdir>/gemm_<kind>_<fmt>_<m_tiles>_<k_blocks>.hal
  <outdir>/<fixed>.hal   for the non-GEMM kernels
"""
import json, os, re, struct, subprocess, sys, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
LOOM = os.path.abspath(os.path.join(HERE, ".."))
EMIT = os.path.join(LOOM, "emit_hal.py")

# ggml type -> (short name, short name, qk)
FMT = {
    12: ("q4k", "q4k", 256), 13: ("q5k", "q5k", 256), 14: ("q6k", "q6k", 256),
    11: ("q3k", "q3k", 256), 23: ("iq4xs", "iq4xs", 256), 21: ("iq3s", "iq3s", 256),
    18: ("iq3xxs", "iq3xxs", 256), 20: ("iq4nl", "iq4nl", 32),
    17: ("iq2xs", "iq2xs", 256), 8: ("q8_0", "q8_0", 32),
    16: ("iq2xxs", "iq2xxs", 256), 22: ("iq2s", "iq2s", 256), 10: ("q2k", "q2k", 256),
    44: ("mq4", "mq4", 256),   # hipfire MQ4G256 (HFQ models; not a ggml type)
}
KSTORE = {"attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight",
          "ssm_beta.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
          "ffn_gate.weight"}
RESIDUAL = {"attn_output.weight", "ssm_out.weight", "ffn_down.weight"}
SWIGLU = {"ffn_up.weight"}


def parse(model):
    """Return [(name, dims, type)] for every tensor of a model as the engine sees it: an HFQ model or a GGUF one with
    its projections converted to MQ4G256 at load (yah-dump --table, core/gguf_convert.cpp), or with YAH_NATIVE=1 a
    GGUF's own tensors (only token_embd converted, to Q8_0: the embed kernel's format)."""
    f = open(model, "rb")
    if f.read(4) == b"HFQM" or os.environ.get("YAH_NATIVE") != "1":
        dump = os.path.join(LOOM, "..", "..", "build", "yah-dump")
        out = subprocess.run([dump, model, "--table"], capture_output=True, text=True, check=True).stdout
        return [(p[0], [int(d) for d in p[2:]], int(p[1])) for p in (ln.split() for ln in out.splitlines())]
    f.seek(8)
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
        rows.append((nm, dims, 8 if nm == "token_embd.weight" else ty))
    return rows


def sym_of(loomfile):
    text = open(os.path.join(LOOM, loomfile)).read()
    return re.search(r"config\.decl @([A-Za-z0-9_]+)\.m_tiles", text).group(1)


def emit(loomfile, configs, outname, outdir, env=None):
    """Compile one Loom source (relative to engine/gpu/loom, or absolute) at one config to <outdir>/<outname>.
    env: extra environment for this compile (per-kernel compiler options, e.g. LOOM_EXP_LICM)."""
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(LOOM, loomfile)
    keep = tempfile.mkdtemp(prefix="emit_keep_", dir=tmp)   # the specialized source, for its envelopes
    r = subprocess.run([sys.executable, EMIT, src, tmp] + configs,
                       capture_output=True, text=True, env={**os.environ, **(env or {}), "YAH_EMIT_KEEP": keep})
    if r.returncode != 0:
        # stderr first: it holds the compiler diagnostics; stdout is progress output.
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
    envelopes(os.path.join(keep, os.listdir(keep)[0]), dst + ".fp")
    shutil.rmtree(keep, True)
    return dst


# Roots whose envelope is not a bound on what the kernel touches, so their .fp line is left out: <kernel>.<root>: why.
UNBOUNDED = {
    # its ring loads run under i < qkv_size (ISA: after the s_and_saveexec on that compare), but the analysis bounds
    # them by the grid's alpha / beta lanes: 10x the state view
    "yah_deltanet_prep_ab.state": "guarded loads",
}


def envelopes(src, fp):
    """Write <hal>.fp: "<argument> <bytes> <root>" per root of the HAL's one kernel, the bytes it may touch through that
    binding over its compiled grid ("grid <x> <y> <z>"; a dispatch of fewer workgroups touches that share). engine/run/hrx_dry.c checks every view the engine binds
    against it: a view shorter than its kernel's envelope reads unmapped VA, and the ring hangs. A module of several
    kernels gets none (hrx_dry lists it unchecked)."""
    sys.path.insert(0, LOOM)
    import hrx_paths
    with tempfile.TemporaryDirectory(prefix="emit_fp_") as t:
        rep = os.path.join(t, "report.json")
        r = subprocess.run([hrx_paths.LOOM_COMPILE, src, "--target=amdgpu:gfx1151", "--format=amdgpu-hsaco",
                            "--output=" + os.path.join(t, "k.hsaco"), "--compile-report=details",
                            "--compile-report-output=" + rep], env=hrx_paths.env(), capture_output=True, text=True)
        found, grid = [], None

        def walk(node):
            if isinstance(node, dict):
                roots = node.get("roots")
                if isinstance(roots, list) and roots and all(isinstance(x, dict) and "interval_envelope" in x for x in roots):
                    found.append(roots)
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        if r.returncode == 0:
            report = json.load(open(rep))
            walk(report)
            grid = (report.get("workload") or {}).get("workgroup_count")
    if len(found) != 1:
        if os.path.exists(fp):
            os.remove(fp)
        return
    rows = {}
    for x in found[0]:
        a, b = x.get("source_root_argument_index"), (x.get("interval_envelope") or {}).get("byte_count")
        if "%s.%s" % (x.get("function"), x.get("source_root")) in UNBOUNDED:
            continue
        if a is not None and b is not None and b >= rows.get(a, (0,))[0]:
            rows[a] = (b, x.get("source_root") or "?")
    with open(fp, "w") as fh:
        if grid:
            fh.write("grid %d %d %d\n" % (grid["x"], grid["y"], grid["z"]))
        fh.writelines("%d %d %s\n" % (a, b, n) for a, (b, n) in sorted(rows.items()))
