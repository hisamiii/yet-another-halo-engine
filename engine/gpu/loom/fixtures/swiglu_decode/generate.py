#!/usr/bin/env python3
"""Fixtures for yah_swiglu_decode_f32.loom.

Pads each GEMV weight fixture to the swiglu band max (16 rows * 4200 = 67200
bytes) and saves silu(dot_gate)*dot_up for the pair cases. Every GEMV fixture
shares the same sparse x, so the per-format row dots are the GEMV expectations.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.dirname(HERE)
ROWS = 16
MAXB = 67200
FMTS = ["q4k", "iq4xs", "iq4nl", "q5k", "q6k", "q3k", "iq3s", "iq3xxs"]


def main():
    x = np.load(os.path.join(FIX, "q4k_gemm", "gemv_x.npy")).astype(np.float32)
    np.save(os.path.join(HERE, "x.npy"), x)
    dots = {}
    for fmt in FMTS:
        src = np.load(os.path.join(FIX, fmt + "_gemm", "input_" + fmt + "_gemm.npy")).astype(np.uint8)
        assert src.size <= MAXB, (fmt, src.size)
        out = np.zeros(MAXB, dtype=np.uint8)
        out[:src.size] = src
        np.save(os.path.join(HERE, fmt + "_pad.npy"), out.view(np.int8))
        dots[fmt] = np.load(os.path.join(FIX, fmt + "_gemm", "gemv_expected.npy")).astype(np.float64)
        print(fmt, src.size)
    for g in FMTS:
        for u in FMTS:
            dg = dots[g]
            du = dots[u]
            expected = (dg / (1.0 + np.exp(-dg)) * du).astype(np.float32)
            np.save(os.path.join(HERE, "%s_%s_expected.npy" % (g, u)), expected)
    print("done")


if __name__ == "__main__":
    main()