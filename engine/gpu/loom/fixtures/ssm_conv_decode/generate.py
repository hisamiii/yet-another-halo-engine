#!/usr/bin/env python3
"""Fixtures for yah_ssm_conv_decode_f32.loom (SSMConvKernel, ssm_decode_recurrence.hip).

Per (row, channel): shift the 4-slot conv state, dot with the 4-tap conv weights,
silu, write conv_out, and leave the advanced state in place.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
QKV = 8
ROWS = 2


def main():
    qkv = np.zeros((ROWS, QKV), dtype=np.float32)
    for r in range(ROWS):
        for c in range(QKV):
            qkv[r, c] = np.float32(((r * 3 + c * 5) % 17 - 8) * 0.25)
    w = np.zeros(QKV * 4, dtype=np.float32)
    for c in range(QKV):
        for k in range(4):
            w[c * 4 + k] = np.float32(((c * 7 + k * 3) % 11 - 5) * 0.5)
    state = np.zeros(QKV * 4, dtype=np.float32)
    for c in range(QKV):
        for k in range(4):
            state[c * 4 + k] = np.float32(((c * 2 + k * 5) % 13 - 6) * 0.125)
    st = state.astype(np.float64).copy()
    out = np.zeros((ROWS, QKV), dtype=np.float32)
    for r in range(ROWS):
        for c in range(QKV):
            o = c * 4
            s1, s2, s3 = st[o + 1], st[o + 2], st[o + 3]
            x = np.float64(qkv[r, c])
            dot = s1 * np.float64(w[o]) + s2 * np.float64(w[o + 1]) + s3 * np.float64(w[o + 2]) + x * np.float64(w[o + 3])
            sig = 1.0 / (1.0 + np.exp(-dot))
            out[r, c] = np.float32(dot * sig)
            st[o], st[o + 1], st[o + 2], st[o + 3] = s1, s2, s3, x
    np.save(os.path.join(HERE, "input_qkv.npy"), qkv.reshape(-1).astype(np.float32))
    np.save(os.path.join(HERE, "input_w.npy"), w)
    np.save(os.path.join(HERE, "input_state.npy"), state)
    np.save(os.path.join(HERE, "expected_out.npy"), out.reshape(-1).astype(np.float32))
    np.save(os.path.join(HERE, "expected_state.npy"), st.astype(np.float32))
    print("out", out.reshape(-1)[:4])
    print("state", st[:4])


if __name__ == "__main__":
    main()