#!/usr/bin/env python3
"""Generate the fp16 SSM post-norm + gate fixture for yah_ssm_postnorm_gate_f16.loom.

Reference for BatchedSSMPostNormGateFp16Kernel in ssm_row_split.hip. Heads are
flat (token * num_heads + head), each owning 128 values:

  sum = sum_i raw[head*128 + i]^2
  inv = 1 / sqrt(sum/128 + 1e-6)
  out[head*128 + i] = f16((raw[i] * inv) * weight[i] * (g * sigmoid(g)))
  g = gate[head*128 + i]

The gate makes the output transcendental and the store rounds to fp16, so the
comparison carries a tolerance. The fixture records the f16 result so the check
exercises the conversion as well as the arithmetic.
"""
import os
import numpy as np

HEADS = 3
VAL = 128
OUT = os.path.dirname(os.path.abspath(__file__))


def pattern(count, modulus, shift):
    idx = np.arange(count, dtype=np.int64)
    return (((idx * 5) % modulus) - shift).astype(np.float32) / np.float32(4.0)


def main():
    raw = pattern(HEADS * VAL, 17, 8)
    gate = pattern(HEADS * VAL, 13, 6)
    weight = pattern(VAL, 9, 4)
    out = np.zeros(HEADS * VAL, dtype=np.float16)
    for head in range(HEADS):
        base = head * VAL
        vals = raw[base:base + VAL].astype(np.float32)
        total = np.float32(np.sum(vals * vals, dtype=np.float32))
        inv = np.float32(1.0) / np.sqrt(total / np.float32(VAL) + np.float32(1e-6))
        for i in range(VAL):
            result = np.float32(np.float32(vals[i] * inv) * weight[i])
            gv = gate[base + i]
            sig = np.float32(1.0) / np.float32(1.0 + np.exp(-np.float64(gv)))
            result = np.float32(result * np.float32(gv * sig))
            out[base + i] = np.float16(result)
    np.save(os.path.join(OUT, 'input_raw.npy'), raw)
    np.save(os.path.join(OUT, 'input_gate.npy'), gate)
    np.save(os.path.join(OUT, 'input_weight.npy'), weight)
    np.save(os.path.join(OUT, 'expected_out.npy'), out)
    print('wrote raw=%s gate=%s weight=%s expected=%s (%s)'
          % (raw.shape, gate.shape, weight.shape, out.shape, out.dtype))
    print('  out[:4]=%s' % out[:4])


if __name__ == '__main__':
    main()
