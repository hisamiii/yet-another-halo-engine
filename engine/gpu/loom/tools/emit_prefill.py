#!/usr/bin/env python3
"""Helpers for the HAL emitters (emit_prefill_pp.py, emit_decode.py): the model's tensor table, the format map and emit().

Naming convention, so the C++ driver needs no manifest:
  <outdir>/gemm_<kind>_<fmt>_<m_tiles>_<k_blocks>.hal
  <outdir>/<fixed>.hal   for the non-GEMM kernels
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOOM = os.path.abspath(os.path.join(HERE, ".."))
EMIT = os.path.join(LOOM, "emit_hal.py")

# tensor type -> (short name, short name, qk); the GEMMs and GEMVs run only MQ4 (the GGUF formats went with the GGUF
# model path, tag pre-cleanup-2026-10-20)
FMT = {44: ("mq4", "mq4", 256)}   # hipfire MQ4G256 (HFQ models; not a ggml type)
KSTORE = {"attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight",
          "ssm_beta.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
          "ffn_gate.weight"}
RESIDUAL = {"attn_output.weight", "ssm_out.weight", "ffn_down.weight"}
SWIGLU = {"ffn_up.weight"}


def parse(model):
    """Return [(name, dims, type)] for every tensor of an HFQ model as the engine sees it (yah-dump --table)."""
    if open(model, "rb").read(4) != b"HFQM":
        raise SystemExit("%s: not an HFQ model (the GGUF model path is gone since tag pre-cleanup-2026-10-20)" % model)
    dump = os.path.join(LOOM, "..", "..", "build", "yah-dump")
    out = subprocess.run([dump, model, "--table"], capture_output=True, text=True, check=True).stdout
    return [(p[0], [int(d) for d in p[2:]], int(p[1])) for p in (ln.split() for ln in out.splitlines())]


def sym_of(loomfile):
    text = open(os.path.join(LOOM, loomfile)).read()
    return re.search(r"config\.decl @([A-Za-z0-9_]+)\.m_tiles", text).group(1)


def emit(loomfile, configs, outname, outdir, env=None):
    """Compile one Loom source (relative to engine/gpu/loom, or absolute) at one config to <outdir>/<outname>.
    env: extra environment for this compile (per-kernel compiler options, e.g. LOOM_EXP_LICM)."""
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(LOOM, loomfile)
    r = subprocess.run([sys.executable, EMIT, src, tmp] + configs,
                       capture_output=True, text=True, env={**os.environ, **env} if env else None)
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
    return dst
