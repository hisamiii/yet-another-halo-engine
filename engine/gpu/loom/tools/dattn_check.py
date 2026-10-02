#!/usr/bin/env python3
"""Check gen_decode_attn.py (kvappend, part, reduce) against numpy on random data
with a scrambled page table.

usage: dattn_check.py <model.gguf> <workdir> [T] [pos]
(the model is only opened by hal_run; no tensor is bound)
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_decode_attn as A  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
EMIT = os.path.join(HERE, "..", "emit_hal.py")
HALRUN = os.path.join(ROOT, "engine", "build", "hal_run")
GPURUN = os.path.join(ROOT, "engine", "run", "gpu_run.sh")


def build(work, which, T):
    src = os.path.join(work, f"da_{which}.loom")
    open(src, "w").write(A.gen(which, T))
    r = subprocess.run([sys.executable, EMIT, src, os.path.join(work, "hal"), "nop=0"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stdout[-1500:] + r.stderr[-1500:])
    return r.stdout.strip().splitlines()[0]


def run(model, hal, grid, binds, mins):
    cmd = [GPURUN, "dattn-check", "--", HALRUN, model, hal, grid, "256", ",".join(map(str, mins))] + binds
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if "hal_run: ok" not in r.stdout:
        raise SystemExit(r.stdout[-1500:] + r.stderr[-1500:])


def main():
    model, work = sys.argv[1], sys.argv[2]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    pos = int(sys.argv[4]) if len(sys.argv) > 4 else 700
    os.makedirs(work, exist_ok=True)
    rng = np.random.default_rng(3)
    npg = T // 256
    ptab = rng.permutation(npg).astype(np.int32)
    n = pos + 1
    K = rng.standard_normal((T, 4, 256)).astype(np.float16)
    V = rng.standard_normal((T, 4, 256)).astype(np.float16)
    q = (0.3 * rng.standard_normal((24, 256))).astype(np.float32)
    gate = rng.standard_normal((24, 256)).astype(np.float32)
    kpool = np.zeros((T, 1024), np.float16)
    vt = np.zeros((4, T // 16, 256, 16), np.float16)
    for p in range(T):
        row = ptab[p // 256] * 256 + p % 256
        kpool[row] = K[p].reshape(1024)
        vt[:, row // 16, :, row % 16] = V[p]
    f = lambda nm, a: (a.tofile(os.path.join(work, nm)), os.path.join(work, nm))[1]
    fq, fk, fv, fp = f("q", q), f("kp", kpool), f("vt", vt), f("pt", ptab)
    fpos = f("pos", np.array([pos], np.int32))
    fg = f("gate", gate)
    used = pos // 256 + 1
    acc_b, ml_b = npg * 24 * 256 * 4, npg * 24 * 2 * 4
    hp = build(work, "part", T)
    run(model, hp, f"{used},4", [f"f:{fq}", f"f:{fk}", f"f:{fv}", f"f:{fp}", f"f:{fpos}",
                                  f"o:{acc_b}:{work}/acc", f"o:{ml_b}:{work}/ml"],
        [24 * 256 * 4, T * 2048, T * 2048, npg * 4, 4, acc_b, ml_b])
    hr = build(work, "reduce", T)
    run(model, hr, "24", [f"f:{work}/acc", f"f:{work}/ml", f"f:{fg}", f"f:{fpos}", f"o:{24 * 256 * 4}:{work}/out"],
        [acc_b, ml_b, 24 * 256 * 4, 4, 24 * 256 * 4])
    out = np.fromfile(f"{work}/out", np.float32).reshape(24, 256)
    ref = np.zeros((24, 256))
    for hh in range(24):
        kv = hh // 6
        s = K[:n, kv].astype(np.float64) @ q[hh].astype(np.float64) / 16.0
        pr = np.exp(s - s.max())
        pr /= pr.sum()
        ctx = pr @ V[:n, kv].astype(np.float64)
        ref[hh] = ctx / (1.0 + np.exp(-gate[hh].astype(np.float64)))
    err = np.abs(out - ref).max() / np.abs(ref).max()
    print(f"part+reduce T={T} pos={pos} pages used {used} ptab {ptab.tolist()}: max err / max|ref| = {err:.2e}"
          f"{'   <-- FAIL' if not err < 1e-4 else ''}")
    # kvappend at a second position
    p2 = min(517, T - 1)
    kx = rng.standard_normal(1024).astype(np.float32)
    vx = rng.standard_normal(1024).astype(np.float32)
    ha = build(work, "kvappend", T)
    run(model, ha, "4", [f"f:{f('kx', kx)}", f"f:{f('vx', vx)}", f"io:{fk}:{work}/kp2", f"io:{fv}:{work}/vt2",
                         f"f:{fp}", f"f:{f('pos2', np.array([p2], np.int32))}"],
        [4096, 4096, T * 2048, T * 2048, npg * 4, 4])
    kp2 = np.fromfile(f"{work}/kp2", np.float16).reshape(T, 1024)
    vt2 = np.fromfile(f"{work}/vt2", np.float16).reshape(4, T // 16, 256, 16)
    row = ptab[p2 // 256] * 256 + p2 % 256
    ek = kpool.copy(); ek[row] = kx.astype(np.float16)
    ev = vt.copy(); ev[:, row // 16, :, row % 16] = vx.astype(np.float16).reshape(4, 256)
    ok = np.array_equal(kp2, ek) and np.array_equal(vt2, ev)
    print(f"kvappend pos={p2} row={row}: {'exact' if ok else 'MISMATCH  <-- FAIL'}")
    for nm in ("q", "kp", "vt", "pt", "pos", "gate", "acc", "ml", "out", "kx", "vx", "kp2", "vt2", "pos2"):
        try:
            os.remove(os.path.join(work, nm))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
