#!/usr/bin/env python3
"""Helpers for the HAL emitters (emit_prefill_pp.py, emit_decode.py): the GGUF tensor table, the format map and emit().

Naming convention, so the C++ driver needs no manifest:
  <outdir>/gemm_<kind>_<fmt>_<m_tiles>_<k_blocks>.hal
  <outdir>/<fixed>.hal   for the non-GEMM kernels
"""
import hashlib, json, os, re, struct, subprocess, sys, tempfile, shutil, threading
from concurrent.futures import ThreadPoolExecutor

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
    """Return [(name, dims, ggml type)] for every tensor of a GGUF file, or of an HFQ one as the engine sees it."""
    f = open(model, "rb")
    if f.read(4) == b"HFQM":
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
        rows.append((nm, dims, ty))
    return rows


def sym_of(loomfile):
    text = open(os.path.join(LOOM, loomfile)).read()
    return re.search(r"config\.decl @([A-Za-z0-9_]+)\.m_tiles", text).group(1)


# Compiles run on a thread pool when ASYNC is set (emit_prefill_pp): a HAL is in place after wait(path), all after
# wait(). Each one is cached in YAH_EMIT_CACHE (default ~/.cache/yah-emit, "0" off) under the hash of its specialized
# source, target, LOOM_* environment, loom-compile binary and UNBOUNDED.
ASYNC = False
CACHE = os.environ.get("YAH_EMIT_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "yah-emit"))
JOBS = int(os.environ.get("YAH_EMIT_JOBS", "16"))
_pool, _pending, _latest, _lock, _lc = None, {}, {}, threading.Lock(), None


def emit(loomfile, configs, outname, outdir, env=None):
    """Compile one Loom source (relative to engine/gpu/loom, or absolute) at one config to <outdir>/<outname>, and its
    envelopes to <outname>.fp. env: extra environment for this compile (per-kernel compiler options, e.g. LOOM_EXP_LICM).
    The source is read now, so the caller may overwrite it once this returns."""
    global _pool, _lc
    sys.path.insert(0, LOOM)
    import emit_hal, hrx_paths
    if _lc is None:
        _lc = hashlib.sha256(open(hrx_paths.LOOM_COMPILE, "rb").read()).hexdigest()
    ce = {**hrx_paths.env(), **(env or {})}
    tgt = ce.get("YAH_LOOM_TARGET", "gfx1151")
    text, _ = emit_hal.specialize(open(os.path.join(LOOM, loomfile)).read(),
                                  {k.lstrip("@"): v for k, _, v in (c.partition("=") for c in configs)}, tgt)
    loomenv = sorted((k, v) for k, v in ce.items() if k.startswith("LOOM_"))
    key = hashlib.sha256(repr((text, tgt, loomenv, _lc, sorted(UNBOUNDED.items()))).encode()).hexdigest()
    dst = os.path.join(outdir, outname)
    os.makedirs(outdir, exist_ok=True)
    with _lock:
        _latest[dst] = seq = _latest.get(dst, 0) + 1
        if _pool is None:
            _pool = ThreadPoolExecutor(JOBS)
        _pending.setdefault(dst, []).append(_pool.submit(_build, loomfile, text, tgt, ce, key, dst, seq))
    if not ASYNC:
        wait(dst)
    return dst


def wait(dst=None):
    """Wait for the compiles of dst (all when None); a failed one exits with its diagnostics."""
    with _lock:
        futs = _pending.pop(dst, []) if dst else [f for v in _pending.values() for f in v]
        if not dst:
            _pending.clear()
    err = None
    for f in futs:
        try:
            f.result()
        except Exception as x:   # the first failure, after the rest finish
            err = err or x
    if err:
        raise SystemExit(str(err))


def _build(loomfile, text, tgt, ce, key, dst, seq):
    import hrx_paths
    hal, meta = os.path.join(CACHE, key + ".hal"), os.path.join(CACHE, key + ".json")
    if CACHE == "0" or not os.path.exists(meta):
        with tempfile.TemporaryDirectory(prefix="emit_") as t:
            src, rep, out = os.path.join(t, "k.loom"), os.path.join(t, "report.json"), os.path.join(t, "k.hal")
            open(src, "w").write(text)
            cmd = [hrx_paths.LOOM_COMPILE, src, "--target=amdgpu:" + tgt, "--format=amdgpu-hsaco", "--output=" + out,
                   "--compile-report=summary", "--compile-report-output=" + rep]
            r = subprocess.run(cmd, env=ce, capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(out):
                # stderr first: it holds the compiler diagnostics
                log = os.path.join("/tmp", "emit_fail_" + os.path.basename(dst) + ".log")
                with open(log, "w") as fh:
                    fh.write("cmd: %s\n--- stderr ---\n%s\n--- stdout ---\n%s" % (" ".join(cmd), r.stderr, r.stdout))
                    fh.write("\n--- source ---\n" + text)
                head = (r.stderr.strip() + "\n" + r.stdout.strip()).strip().splitlines()[:24]
                raise RuntimeError("emit failed for " + loomfile + " (full log: " + log + "):\n  " + "\n  ".join(head))
            fp = envelopes(json.load(open(rep)))
            if CACHE == "0":
                hal, meta = out, None
                _place(dst, seq, out, fp)
                return
            os.makedirs(CACHE, exist_ok=True)
            shutil.copy(out, hal + ".%d" % threading.get_ident())
            os.replace(hal + ".%d" % threading.get_ident(), hal)
            with open(meta + ".%d" % threading.get_ident(), "w") as fh:
                json.dump({"fp": fp, "src": loomfile}, fh)
            os.replace(meta + ".%d" % threading.get_ident(), meta)
    _place(dst, seq, hal, json.load(open(meta))["fp"])


def _place(dst, seq, hal, fp):
    """Copy a built HAL and its .fp (None: no .fp) to dst, unless a later emit of dst replaced this one."""
    with _lock:
        if _latest.get(dst) != seq:
            return
        shutil.copy(hal, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        if fp is None:
            if os.path.exists(dst + ".fp"):
                os.remove(dst + ".fp")
        else:
            with open(dst + ".fp", "w") as fh:
                fh.write(fp)


# Roots whose envelope is not a bound on what the kernel touches, so their .fp line is left out: <kernel>.<root>: why.
UNBOUNDED = {
    # its ring loads run under i < qkv_size (ISA: after the s_and_saveexec on that compare), but the analysis bounds
    # them by the grid's alpha / beta lanes: 10x the state view
    "yah_deltanet_prep_ab.state": "guarded loads",
}


def envelopes(report):
    """The <hal>.fp text of a loom-compile report (None: no .fp): "<argument> <bytes> <root>" per root of the HAL's one kernel, the bytes it may touch through that
    binding over its compiled grid ("grid <x> <y> <z>"; a dispatch of fewer workgroups touches that share). engine/run/hrx_dry.c checks every view the engine binds
    against it: a view shorter than its kernel's envelope reads unmapped VA, and the ring hangs. A module of several
    kernels gets none (hrx_dry lists it unchecked)."""
    found = []

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

    walk(report)
    grid = (report.get("workload") or {}).get("workgroup_count")
    if len(found) != 1:
        return None
    rows = {}
    for x in found[0]:
        a, b = x.get("source_root_argument_index"), (x.get("interval_envelope") or {}).get("byte_count")
        if "%s.%s" % (x.get("function"), x.get("source_root")) in UNBOUNDED:
            continue
        if a is not None and b is not None and b >= rows.get(a, (0,))[0]:
            rows[a] = (b, x.get("source_root") or "?")
    return (("grid %d %d %d\n" % (grid["x"], grid["y"], grid["z"])) if grid else "") + \
        "".join("%d %d %s\n" % (a, b, n) for a, (b, n) in sorted(rows.items()))
