#!/usr/bin/env python3
"""Generate the causal SSM conv fixture for yah_ssm_conv_f32.loom.

Reference for BatchedSSMConvKernel in ssm_recurrence.hip. The kernel is a 4-tap
causal convolution whose last weight is applied to the current token and whose
other three fall back to a 4-slot history ring at the start of the batch:

  s3 = x[t][c]
  s2 = t >= 1 ? x[t-1][c] : state[c*4 + 3]
  s1 = t >= 2 ? x[t-2][c] : (t == 1 ? state[c*4 + 3] : state[c*4 + 2])
  s0 = t >= 3 ? x[t-3][c] : (t == 2 ? state[c*4 + 3]
                                : (t == 1 ? state[c*4 + 2] : state[c*4 + 1]))
  dot = s0*w0 + s1*w1 + s2*w2 + s3*w3
  out[t][c] = dot * sigmoid(dot)

The fallback ladder is the whole reason this kernel needs a fixture: the exact
expectation is a transcendental, and the branch structure only shows up in the
first three tokens. The comparison therefore carries a tolerance rather than
being exact, which is the honest contract for an expf on both sides.
"""
import os
import numpy as np

BATCH = 4
QKV = 16
OUT = os.path.dirname(os.path.abspath(__file__))


def pattern(count, modulus, shift):
    idx = np.arange(count, dtype=np.int64)
    return (((idx * 7) % modulus) - shift).astype(np.float32) / np.float32(8.0)


def main():
    # Weights vary per channel so a mis-indexed tap changes the result.
    weights = np.empty(QKV * 4, dtype=np.float32)
    for c in range(QKV):
        weights[c * 4 + 0] = np.float32((c % 3) - 1) * np.float32(0.5)
        weights[c * 4 + 1] = np.float32((c % 5) - 2) * np.float32(0.25)
        weights[c * 4 + 2] = np.float32((c % 4) - 2) * np.float32(0.375)
        weights[c * 4 + 3] = np.float32(1.0) - np.float32(c % 4) * np.float32(0.25)
    state = pattern(QKV * 4, 11, 5)
    x = pattern(BATCH * QKV, 13, 6).reshape(BATCH, QKV)

    out = np.zeros((BATCH, QKV), dtype=np.float32)
    for t in range(BATCH):
        for c in range(QKV):
            w0, w1, w2, w3 = (weights[c * 4 + j] for j in range(4))
            s3 = x[t, c]
            s2 = x[t - 1, c] if t >= 1 else state[c * 4 + 3]
            if t >= 2:
                s1 = x[t - 2, c]
            elif t == 1:
                s1 = state[c * 4 + 3]
            else:
                s1 = state[c * 4 + 2]
            if t >= 3:
                s0 = x[t - 3, c]
            elif t == 2:
                s0 = state[c * 4 + 3]
            elif t == 1:
                s0 = state[c * 4 + 2]
            else:
                s0 = state[c * 4 + 1]
            dot = np.float32(np.float32(s0 * w0) + np.float32(s1 * w1)
                             + np.float32(s2 * w2) + np.float32(s3 * w3))
            out[t, c] = np.float32(dot * np.float32(1.0 / (1.0 + np.exp(-dot))))

    for name, array in (('input_qkv.npy', x.reshape(-1)),
                        ('input_weights.npy', weights),
                        ('input_state.npy', state),
                        ('expected_out.npy', out.reshape(-1))):
        np.save(os.path.join(OUT, name), array)
        print('wrote %-20s shape=%-8s dtype=%s' % (name, array.shape, array.dtype))
    print('  out[0][:4]=%s out[3][:4]=%s' % (out[0][:4], out[3][:4]))


if __name__ == '__main__':
    main()
