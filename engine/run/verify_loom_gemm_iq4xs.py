#!/usr/bin/env python3
"""Verify loom_gemm_probe output for an IQ4_XS weight against a float64 oracle.

usage: verify_loom_gemm_iq4xs.py <model.gguf> <file_offset> <out.bin> <tokens> <n> <k> [rows...]
"""
import sys

import numpy as np

KVALS = np.array([-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113], dtype=np.float64)


def decode_iq4xs_row(raw, offset, row, k):
    nb = k // 256
    bpr = nb * 136
    raw.seek(offset + row * bpr)
    data = np.frombuffer(raw.read(bpr), dtype=np.uint8).reshape(nb, 136)
    d = data[:, 0:2].copy().view(np.float16).astype(np.float64).reshape(nb)
    scales_h = (data[:, 2].astype(np.int64) | (data[:, 3].astype(np.int64) << 8)).reshape(nb, 1)
    scales_l = data[:, 4:8].astype(np.int64)
    qs = data[:, 8:136].astype(np.int64).reshape(nb, 8, 16)
    g = np.arange(8)
    sc_l = (scales_l[:, g // 2] >> (4 * (g % 2))) & 15
    sc_hi = (scales_h >> (2 * g)) & 3
    dl = d[:, None] * ((sc_l | (sc_hi << 4)) - 32)
    lo = KVALS[qs & 15]
    hi = KVALS[qs >> 4]
    vals = np.empty((nb, 8, 32), dtype=np.float64)
    vals[:, :, 0:16] = lo
    vals[:, :, 16:32] = hi
    return (dl[:, :, None] * vals).reshape(-1)


def activation(token, k):
    i = np.arange(k)
    return ((token * 7 + i * 13) % 23 - 11) * 0.125


def main():
    model = sys.argv[1]
    offset = int(sys.argv[2])
    out_path = sys.argv[3]
    tokens = int(sys.argv[4])
    n = int(sys.argv[5])
    k = int(sys.argv[6])
    checks = [int(x) for x in sys.argv[7:]] or list(range(0, min(n, 8)))
    raw = open(model, "rb")
    out = np.fromfile(out_path, dtype=np.float32)
    worst_abs = 0.0
    worst_rel = 0.0
    for t in range(tokens):
        x = activation(t, k).astype(np.float16).astype(np.float64)
        for row in checks:
            w = decode_iq4xs_row(raw, offset, row, k).astype(np.float16).astype(np.float64)
            exp = float(np.dot(x, w))
            got = float(out[t * n + row])
            a = abs(got - exp)
            worst_abs = max(worst_abs, a)
            worst_rel = max(worst_rel, a / max(abs(exp), 1e-6))
    print("tokens", tokens, "n", n, "k", k, "checks", len(checks) * tokens)
    print("max_abs", worst_abs)
    print("max_rel", worst_rel)
    ok = worst_rel < 1e-3 and worst_abs < 1e-3
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())