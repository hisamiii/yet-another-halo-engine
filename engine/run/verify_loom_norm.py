#!/usr/bin/env python3
"""Verify loom_norm_probe output for a fused RMSNorm+residual against a float32 oracle.

The kernel emits fp16, so the oracle is rounded to fp16 before comparing: a correct
kernel may differ from the float64 oracle by one fp16 ulp where the reduction order
crosses a rounding boundary. We therefore require every element within one fp16 ulp
of the oracle and only a handful of elements (if any) off by that single ulp.

usage: verify_loom_norm.py <model.gguf> <file_offset> <hidden.f32> <out.f16> <rows> <dim> <eps>
"""
import sys

import numpy as np


def fp16_ulp(values):
    v16 = np.asarray(values, dtype=np.float16).astype(np.float32)
    up = np.nextafter(v16, np.float32(np.inf))
    dn = np.nextafter(v16, np.float32(-np.inf))
    return np.maximum(np.abs(up - v16), np.abs(dn - v16))


def main():
    model = sys.argv[1]
    offset = int(sys.argv[2])
    hidden_path = sys.argv[3]
    out_path = sys.argv[4]
    rows = int(sys.argv[5])
    dim = int(sys.argv[6])
    eps = np.float32(float(sys.argv[7]))
    w = np.fromfile(model, dtype=np.float32, count=dim, offset=offset)
    if w.size != dim:
        print("FAIL: weight read", w.size, "!=", dim)
        return 1
    hidden = np.fromfile(hidden_path, dtype=np.float32)
    if hidden.size != rows * dim:
        print("FAIL: hidden", hidden.size, "!=", rows * dim)
        return 1
    hidden = hidden.reshape(rows, dim)
    loom = np.fromfile(out_path, dtype=np.float16)
    if loom.size != rows * dim:
        print("FAIL: output", loom.size, "!=", rows * dim)
        return 1
    loom = loom.reshape(rows, dim)

    worst_abs = 0.0
    worst_ratio = 0.0
    mismatches = 0
    tol_fail = 0
    for i in range(rows):
        x = hidden[i].astype(np.float32)
        sumsq = np.sum(x.astype(np.float64) * x.astype(np.float64))
        inv = np.float32(1.0 / np.sqrt(sumsq / dim + float(eps)))
        oracle = (x * inv * w).astype(np.float32)
        expected = oracle.astype(np.float16)
        got = loom[i]
        diff = np.abs(got.astype(np.float32) - expected.astype(np.float32))
        ulp = fp16_ulp(expected)
        worst_abs = max(worst_abs, float(diff.max()))
        worst_ratio = max(worst_ratio, float((diff / ulp).max()))
        mismatches += int(np.count_nonzero(got != expected))
        tol_fail += int(np.count_nonzero(diff > ulp))
    print("rows", rows, "dim", dim, "elems", rows * dim)
    print("f16 mismatches (vs rounded oracle):", mismatches)
    print("elements beyond one f16 ulp:", tol_fail)
    print("max_abs", worst_abs)
    print("max_ulp_ratio", worst_ratio)
    ok = tol_fail == 0 and mismatches <= 16
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())