#!/usr/bin/env python3
"""Verify loom_gemm_probe output for a Q4_K weight against a float64 oracle.

The GEMM computes out[token*N + row] = sum_k q4k_decode(W,row,k)*f16(x[token][k]).
The activation tile is the same deterministic pattern the probe generates. The
probe pads tokens to 64 and emits out[token*N+row], so the first tokens*N values
are the real rows.

usage: verify_loom_gemm.py <model.gguf> <file_offset> <out.bin> <tokens> <n> <k> [rows...]
"""
import sys

import numpy as np


def decode_q4k_row(raw, offset, row, k):
    nb = k // 256
    bpr = nb * 144
    raw.seek(offset + row * bpr)
    data = np.frombuffer(raw.read(bpr), dtype=np.uint8).reshape(nb, 144)
    d = data[:, 0:2].copy().view(np.float16).astype(np.float64).reshape(nb)
    dmin = data[:, 2:4].copy().view(np.float16).astype(np.float64).reshape(nb)
    scales = data[:, 4:16].astype(np.int64)
    qs = data[:, 16:144].astype(np.int64)
    i = np.arange(256)
    gg = i // 64
    wv = i % 64
    lane = wv % 32
    low = wv < 32
    qb = qs[:, gg * 32 + lane]
    quant = np.where(low, qb & 15, qb >> 4)
    j = 2 * gg + np.where(low, 0, 1)
    sc = np.empty((nb, 256), dtype=np.int64)
    sm = np.empty((nb, 256), dtype=np.int64)
    for jj in range(8):
        if jj < 4:
            dd = scales[:, jj] & 63
            mm = scales[:, jj + 4] & 63
        else:
            dd = (scales[:, jj + 4] & 0xF) | ((scales[:, jj - 4] >> 6) << 4)
            mm = (scales[:, jj + 4] >> 4) | ((scales[:, jj] >> 6) << 4)
        sel = j == jj
        sc[:, sel] = dd[:, None]
        sm[:, sel] = mm[:, None]
    vals = d[:, None] * sc * quant - dmin[:, None] * sm
    return vals.reshape(-1)

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
            # the kernel stages the decoded weight through f16 before the MMA,
            # so the oracle rounds the same way; only accumulation order differs.
            w = decode_q4k_row(raw, offset, row, k).astype(np.float16).astype(np.float64)
            exp = float(np.dot(x, w))
            got = float(out[t * n + row])
            a = abs(got - exp)
            r = a / max(abs(exp), 1e-6)
            worst_abs = max(worst_abs, a)
            worst_rel = max(worst_rel, r)
    print("tokens", tokens, "n", n, "k", k, "rows", len(checks) * tokens)
    print("max_abs", worst_abs)
    print("max_rel", worst_rel)
    ok = worst_rel < 1e-3 and worst_abs < 1e-3
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())