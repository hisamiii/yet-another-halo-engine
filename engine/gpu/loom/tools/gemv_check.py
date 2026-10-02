#!/usr/bin/env python3
"""Check gen_gemv.py kernels on real GGUF tensors against a float64 numpy oracle.

usage: gemv_check.py <model.gguf> <workdir> [kind ...]      (kinds: plain resid swiglu)

For every (format, K) present on the shard it takes the first matching tensor,
generates and emits the kernel, dispatches it once through engine/build/hal_run
(which refuses any binding smaller than gen_gemv.footprint) and compares y with
gguf-py's dequantization: y_ref = W.astype(f64) @ x. Reports the error relative
to max |y_ref|; the f32 accumulation of a 17408-long dot lands near 1e-6.
Needs PYTHONPATH with llama.cpp's gguf-py and a numpy with BLAS.
"""
import collections
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_gemv as G  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
EMIT = os.path.join(HERE, "..", "emit_hal.py")
HALRUN = os.path.join(ROOT, "engine", "build", "hal_run")
GPURUN = os.path.join(ROOT, "engine", "run", "gpu_run.sh")
ONLY = os.environ.get("GEMV_ONLY", "")             # regex on the case tag
BENCH = int(os.environ.get("GEMV_BENCH", "0"))   # N: also time N dispatches per kernel
TABLE_DIR = os.environ.get("YAH_TABLE_DIR", "/home/q/yah-hal-p71")
TABLE_FILE = {"grid_iq3s": "grid_iq3s.bin", "grid_iq3xxs": "grid_iq3xxs.bin", "grid_iq2xxs": "grid_iq2xxs.bin",
              "grid_iq2xs": "grid_iq2xs.bin", "ksigns": "ksigns_iq2xxs.bin"}


def run_kernel(model, work, tag, kind, fmts, M, K, tensors, x, y0=None, R=2, W=4):
    src = os.path.join(work, tag + ".loom")
    open(src, "w").write(G.gen(kind, fmts, M, K, R, W))
    r = subprocess.run([sys.executable, EMIT, src, os.path.join(work, tag), "nop=0"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{tag}: emit failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    hal = r.stdout.strip().splitlines()[0]
    xf = os.path.join(work, tag + ".x")
    x.astype(np.float32).tofile(xf)
    yo = os.path.join(work, tag + ".y")
    binds = [f"t:{t}" for t in tensors] + [f"f:{os.path.join(TABLE_DIR, TABLE_FILE[t])}" for t in G.tables_for(fmts)]
    binds.append(f"f:{xf}")
    if kind == "resid":
        yi = os.path.join(work, tag + ".yi")
        y0.astype(np.float32).tofile(yi)
        binds.append(f"io:{yi}:{yo}")
    else:
        binds.append(f"o:{M * 4}:{yo}")
    mins = ",".join(str(v) for v in G.footprint(kind, fmts, M, K))
    cmd = [GPURUN, "gemv-check", "--", HALRUN, model, hal, str(M // G.rows_per_wg(R, W)), str(32 * W), mins] + binds
    env = dict(os.environ)
    if BENCH:
        env["HAL_RUN_ITERS"] = str(BENCH)
        time.sleep(1)                       # single-kernel timing rule: 1 s gaps
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    if "hal_run: ok" not in r.stdout:
        raise SystemExit(f"{tag}: hal_run failed\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
    y = np.fromfile(yo, np.float32)
    for f in (xf, yo, src):
        os.remove(f)
    ms = None
    for line in r.stdout.splitlines():
        if "ms per dispatch" in line:
            ms = float(line.split()[1])
    return y, ms


def check_bands(model, work, rd, dequantize, rng, layers):
    """band-fused input projections of real layers vs the oracle"""
    worst = 0.0
    t = {x.name: x for x in rd.tensors}
    for l in layers:
        pre = f"blk.{l}."
        names = [pre + n + ".weight" for n in ("attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta")]
        if names[0] not in t:
            names = [pre + n + ".weight" for n in ("attn_q", "attn_k", "attn_v")]
        ts = [t[n] for n in names]
        fmts = [G.GGML[int(x.tensor_type)] for x in ts]
        K = int(ts[0].shape[0]); Ms = [int(x.shape[1]) for x in ts]
        x = rng.standard_normal(K).astype(np.float32)
        tag = f"bands_l{l}"
        src = os.path.join(work, tag + ".loom")
        open(src, "w").write(G.gen_bands(fmts, Ms, K))
        r = subprocess.run([sys.executable, EMIT, src, os.path.join(work, tag), "nop=0"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"{tag}: emit failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
        hal = r.stdout.strip().splitlines()[0]
        xf = os.path.join(work, tag + ".x"); x.tofile(xf)
        outs = [os.path.join(work, f"{tag}.y{i}") for i in range(len(ts))]
        binds = [f"t:{n}" for n in names] + [f"f:{os.path.join(TABLE_DIR, TABLE_FILE[q])}" for q in G.tables_for(fmts)] + [f"f:{xf}"]
        binds += [f"o:{M * 4}:{o}" for M, o in zip(Ms, outs)]
        mins = ",".join(str(v) for v in G.footprint_bands(fmts, Ms, K))
        env = dict(os.environ)
        if BENCH:
            env["HAL_RUN_ITERS"] = str(BENCH); time.sleep(1)
        r = subprocess.run([GPURUN, "gemv-check", "--", HALRUN, model, hal, str(sum(Ms) // G.rows_per_wg()), "128", mins] + binds,
                           capture_output=True, text=True, timeout=120, env=env)
        if "hal_run: ok" not in r.stdout:
            raise SystemExit(f"{tag}: hal_run failed\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
        perf = ""
        for line in r.stdout.splitlines():
            if "ms per dispatch" in line:
                ms = float(line.split()[1]); wb = sum(G.row_bytes(f, K) * M for f, M in zip(fmts, Ms))
                perf = f"   {ms * 1000:7.1f} us  {wb / ms / 1e6:6.1f} GB/s"
        errs = []
        for z, o in zip(ts, outs):
            ref = dequantize(z.data, z.tensor_type).astype(np.float64) @ x.astype(np.float64)
            y = np.fromfile(o, np.float32)
            errs.append(float(np.max(np.abs(y - ref)) / max(np.max(np.abs(ref)), 1e-30)))
            os.remove(o)
        os.remove(xf); os.remove(src)
        worst = max(worst, max(errs))
        print(f"{tag:12s} {'/'.join(fmts):28s} M={'/'.join(map(str, Ms)):22s} max err {max(errs):.2e}"
              f"{'   <-- FAIL' if max(errs) > 1e-4 else ''}{perf}", flush=True)
    return worst


def check_resid_norm(model, work, rd, dequantize, rng, pick):
    """resid_norm: y += W x, then the last workgroup writes rmsnorm(y) * nw and resets the
    counter. One dispatch checked against the oracle, then 50 more back to back: y must
    equal y0 + 51 W x, nout the norm of that, and the counter 0."""
    worst = 0.0
    for (fmt, K), t in pick.items():
        M = int(t.shape[1])
        if M != 5120:
            continue
        x = rng.standard_normal(K).astype(np.float32)
        y0 = rng.standard_normal(M).astype(np.float32)
        nw = (1 + 0.1 * rng.standard_normal(M)).astype(np.float32)
        wx = dequantize(t.data, t.tensor_type).astype(np.float64) @ x.astype(np.float64)
        tag = f"rn_{fmt}_{K}"
        src = os.path.join(work, tag + ".loom")
        open(src, "w").write(G.gen("resid_norm", [fmt], M, K))
        r = subprocess.run([sys.executable, EMIT, src, os.path.join(work, tag), "nop=0"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"{tag}: emit failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
        hal = r.stdout.strip().splitlines()[0]
        fs = {}
        for nm, a in (("x", x), ("y0", y0), ("nw", nw), ("cnt", np.zeros(1, np.int32))):
            fs[nm] = os.path.join(work, f"{tag}.{nm}"); a.tofile(fs[nm])
        errs = []
        for iters in (0, 50):
            env = dict(os.environ)
            if iters:
                env["HAL_RUN_ITERS"] = str(iters)
            binds = [f"t:{t.name}"] + [f"f:{os.path.join(TABLE_DIR, TABLE_FILE[q])}" for q in G.tables_for([fmt])]
            binds += [f"f:{fs['x']}", f"io:{fs['y0']}:{fs['y0']}.out", f"f:{fs['nw']}", f"o:{M * 4}:{fs['nw']}.out", f"io:{fs['cnt']}:{fs['cnt']}.out"]
            mins = ",".join(str(v) for v in G.footprint("resid_norm", [fmt], M, K))
            r = subprocess.run([GPURUN, "gemv-check", "--", HALRUN, model, hal, str(M // G.rows_per_wg()), "128", mins] + binds,
                               capture_output=True, text=True, timeout=120, env=env)
            if "hal_run: ok" not in r.stdout:
                raise SystemExit(f"{tag}: hal_run failed\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
            y = np.fromfile(fs["y0"] + ".out", np.float32)
            nout = np.fromfile(fs["nw"] + ".out", np.float32)
            cnt = int(np.fromfile(fs["cnt"] + ".out", np.int32)[0])
            yref = y0.astype(np.float64) + (1 + iters) * wx
            nref = yref / np.sqrt((yref ** 2).mean() + 1e-6) * nw
            ey = float(np.abs(y - yref).max() / np.abs(yref).max())
            # the norm of the y actually produced (isolates the epilogue from GEMV rounding)
            yy = y.astype(np.float64)
            en = float(np.abs(nout - yy / np.sqrt((yy ** 2).mean() + 1e-6) * nw).max() / np.abs(nref).max())
            errs.append(max(ey, en))
            print(f"{tag:18s} dispatches {1 + iters:3d}: y err {ey:.2e}  norm err {en:.2e}  counter {cnt}"
                  f"{'   <-- FAIL' if max(ey, en) > 1e-4 or cnt != 0 else ''}", flush=True)
            for o in (".out",):
                for nm in ("y0", "nw", "cnt"):
                    if os.path.exists(fs[nm] + o):
                        os.remove(fs[nm] + o)
        for f in fs.values():
            os.remove(f)
        os.remove(src)
        worst = max(worst, max(errs))
    return worst


def main():
    model, work = sys.argv[1], sys.argv[2]
    kinds = sys.argv[3:] or ["plain"]
    os.makedirs(work, exist_ok=True)
    from gguf import GGUFReader
    from gguf.quants import dequantize
    rd = GGUFReader(model)
    pick = collections.OrderedDict()
    for t in rd.tensors:
        fmt = G.GGML.get(int(t.tensor_type))
        if fmt is None or len(t.shape) != 2 or not t.name.startswith("blk."):
            continue
        K, M = int(t.shape[0]), int(t.shape[1])
        if K % 512 or M % 8:
            continue
        pick.setdefault((fmt, K), t)
    rng = np.random.default_rng(1)
    worst = 0.0
    if "resid_norm" in kinds:
        kinds = [k for k in kinds if k != "resid_norm"]
        worst = max(worst, check_resid_norm(model, work, rd, dequantize, rng, pick))
    if "bands" in kinds:
        kinds = [k for k in kinds if k != "bands"]
        worst = max(worst, check_bands(model, work, rd, dequantize, rng, [0, 1, 2, 3, 5, 14, 22, 63]))
    for kind in kinds:
        cases = list(pick.items())
        if kind == "swiglu":   # gate/up pairs from the same layer, formats as found
            ups = {t.name.split(".")[1]: t for t in rd.tensors if t.name.endswith("ffn_up.weight")}
            seen, cases = set(), []
            for t in rd.tensors:
                if t.name.endswith("ffn_gate.weight"):
                    u = ups[t.name.split(".")[1]]
                    key = (G.GGML[int(t.tensor_type)], G.GGML[int(u.tensor_type)])
                    if key not in seen:
                        seen.add(key)
                        cases.append((key, (t, u)))
        for key, t in cases:
            ts = (t,) if hasattr(t, "tensor_type") else t
            fmts = [G.GGML[int(x.tensor_type)] for x in ts]
            K, M = int(ts[0].shape[0]), int(ts[0].shape[1])
            x = rng.standard_normal(K).astype(np.float32)
            ws = [dequantize(z.data, z.tensor_type).astype(np.float64) for z in ts]
            dots = [w @ x.astype(np.float64) for w in ws]
            y0 = None
            if kind == "plain":
                ref = dots[0]
            elif kind == "resid":
                y0 = rng.standard_normal(M).astype(np.float32)
                ref = y0.astype(np.float64) + dots[0]
            else:
                g, u = dots
                ref = g / (1.0 + np.exp(-g)) * u
            tag = f"{kind}_{'_'.join(fmts)}_{K}"
            if ONLY and not re.search(ONLY, tag):
                continue
            y, ms = run_kernel(model, work, tag, kind, fmts, M, K, [z.name for z in ts], x, y0,
                               int(os.environ.get("GEMV_R", "2")), int(os.environ.get("GEMV_W", "4")))
            err = float(np.max(np.abs(y - ref)) / max(np.max(np.abs(ref)), 1e-30))
            worst = max(worst, err)
            perf = ""
            if ms:
                wb = sum(G.row_bytes(f, K) * M for f in fmts)
                perf = f"   {ms * 1000:7.1f} us  {wb / ms / 1e6:6.1f} GB/s ({wb / 1e6:.1f} MB)"
            print(f"{tag:28s} {ts[0].name:26s} M={M:6d} max err / max|y| = {err:.2e}"
                  f"{'   <-- FAIL' if err > 1e-4 or not np.isfinite(err) else ''}{perf}", flush=True)
    print(f"worst {worst:.2e}")


if __name__ == "__main__":
    main()
