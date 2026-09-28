#!/usr/bin/env python3
"""Generate the SSM post-norm + gate fixture for yah_ssm_postnorm_gate_f32.loom.

Reference for BatchedSSMPostNormGateKernel in batched_ssm.hip. One wave per
(token, value head), reducing the squared magnitude over val_dim values:

  mean_sq = sum_i raw[i]^2 / val_dim
  inv     = 1 / sqrt(mean_sq + 1e-6)
  out[i]  = (raw[i] * inv) * weight[i] * (g * sigmoid(g)),  g = gate[i]

The gate factor makes the output transcendental, so the comparison carries a
tolerance. The weight vector is shared across heads and tokens in the kernel, and
the fixture keeps it that way so a mis-indexed weight load changes the result.
"""
import os
import numpy as np

BATCH = 2
HEADS = 2
VAL = 128
INNER = HEADS * VAL
OUT = os.path.dirname(os.path.abspath(__file__))


def pattern(count, modulus, shift):
    idx = np.arange(count, dtype=np.int64)
    return (((idx * 5) % modulus) - shift).astype(np.float32) / np.float32(4.0)


def main():
    raw = pattern(BATCH * INNER, 17, 8)
    gate = pattern(BATCH * INNER, 13, 6)
    weight = pattern(VAL, 9, 4)
    out = np.zeros(BATCH * INNER, dtype=np.float32)
    for t in range(BATCH):
        for h in range(HEADS):
            base = t * INNER + h * VAL
            vals = raw[base:base + VAL].astype(np.float32)
            mean_sq = np.float32(np.sum(vals * vals, dtype=np.float32) / np.float32(VAL))
            inv = np.float32(1.0) / np.sqrt(mean_sq + np.float32(1e-6))
            for i in range(VAL):
                normed = np.float32(vals[i] * inv)
                weighted = np.float32(normed * weight[i])
                g = gate[base + i]
                sig = np.float32(1.0) / np.float32(1.0 + np.exp(-np.float64(g)))
                out[base + i] = np.float32(weighted * np.float32(g * sig))
    np.save(os.path.join(OUT, 'input_raw.npy'), raw)
    np.save(os.path.join(OUT, 'input_gate.npy'), gate)
    np.save(os.path.join(OUT, 'input_weight.npy'), weight)
    np.save(os.path.join(OUT, 'expected_out.npy'), out)
    print('wrote raw=%s gate=%s weight=%s expected=%s'
          % (raw.shape, gate.shape, weight.shape, out.shape))
    print('  out[0][:4]=%s' % out[:4])


if __name__ == '__main__':
    main()
