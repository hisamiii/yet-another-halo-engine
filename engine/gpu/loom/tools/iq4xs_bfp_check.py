#!/usr/bin/env python3
"""Check gen_iq4xs_bfp.py against the NPU panel's oracle (gen_npu_dec weights -> npu_dec_emit.bfp_hw), byte for byte.

usage: iq4xs_bfp_check.py <model.gguf> <workdir> <layer> [F]

The rows are the NPU's of an IQ4_XS FFN pair: the last F (default 8960) rows of blk.<layer>.ffn_gate, then of ffn_up.
Every work item's read and store range is checked first; the kernel is dispatched only if all of them fit.
HAL_RUN_ITERS=N also times N more dispatches.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_iq4xs_bfp as G  # noqa: E402
import npu_dec_emit as Q  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
EMIT = os.path.join(HERE, "..", "emit_hal.py")
HALRUN = os.path.join(ROOT, "engine", "build", "hal_run")
GPURUN = os.path.join(ROOT, "engine", "run", "gpu_run.sh")


def reference(raw, K):
    """raw [rows][K / 256 * 136] u8 -> the fragments [rows / 8][K / 8][72] u8, 1024 rows at a time."""
    rows = raw.shape[0]
    kv = np.array(G.KVALUES, np.float64)
    out = np.empty((rows // 8, K // 8, 72), np.uint8)
    for r0 in range(0, rows, 1024):
        blk = raw[r0:r0 + 1024].reshape(-1, K // 256, 136)
        d, ls, q = Q.iq4xs_fields(blk)
        w = (np.repeat(d[..., None] * (ls - 32), 32, -1) * kv[q]).astype(np.float32).reshape(-1, K)
        n = w.shape[0]
        frag = w.reshape(n // 8, 8, K // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
        out[r0 // 8:(r0 + n) // 8] = Q.bfp_hw(frag).reshape(n // 8, K // 8, 72)
    return out.reshape(-1)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    import gguf
    model, work, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
    F = int(sys.argv[4]) if len(sys.argv) > 4 else 8960
    ts = {t.name: t for t in gguf.GGUFReader(model).tensors}
    parts = []
    for k in ("ffn_gate", "ffn_up"):
        t = ts[f"blk.{layer}.{k}.weight"]
        assert t.tensor_type.name == "IQ4_XS", f"{t.name} is {t.tensor_type.name}"
        K, N = int(t.shape[0]), int(t.shape[1])
        parts.append(np.asarray(t.data).view(np.uint8).reshape(N, -1)[N - F:])
    raw = np.ascontiguousarray(np.concatenate(parts))
    rows, rb = raw.shape
    assert rb == K // 256 * 136
    nin, nout = raw.size, G.out_bytes(rows, K)
    # the kernel's ranges: reads up to the last row's last block + 136, stores up to item * 72 + 72
    last_read = (rows - 1) * rb + (K // 256 - 1) * 136 + 8 + 16 * 7 + 8 + 8
    items = G.items(rows, K)
    if last_read > nin or items * 72 > nout:
        sys.exit(f"ranges outside the buffers (read {last_read} of {nin}, store {items * 72} of {nout}): not dispatching")
    os.makedirs(work, exist_ok=True)
    tag = "iq4xs_bfp"
    srcf = os.path.join(work, tag + ".loom")
    open(srcf, "w").write(G.gen(rows, K))
    r = subprocess.run([sys.executable, EMIT, srcf, os.path.join(work, tag), "nop=0"], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    hal = r.stdout.strip().splitlines()[0]
    wf, of = os.path.join(work, tag + ".w"), os.path.join(work, tag + ".out")
    raw.tofile(wf)
    cmd = [GPURUN, "iq4xs-bfp-check", "--", HALRUN, model, hal, str((items + G.WG - 1) // G.WG), str(G.WG),
           f"{nin},{nout}", f"f:{wf}", f"o:{nout}:{of}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    os.remove(wf)
    if "hal_run: ok" not in r.stdout:
        sys.exit(f"hal_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    got = np.fromfile(of, np.uint8)
    os.remove(of)
    ref = reference(raw, K)
    bad = np.count_nonzero(got != ref)
    ms = [ln.strip() for ln in r.stdout.splitlines() if "ms per dispatch" in ln]
    print(f"blk.{layer} gate+up rows={rows} K={K}: in {nin / 1e6:.1f} MB, out {nout / 1e6:.1f} MB: "
          f"{bad} of {ref.size} bytes differ" + (f" | {ms[0]}" if ms else ""))
    if bad:
        first = int(np.nonzero(got != ref)[0][0])
        print(f"  first mismatch at byte {first} (fragment {first // 72}, byte {first % 72}): got {got[first]} "
              f"want {ref[first]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
