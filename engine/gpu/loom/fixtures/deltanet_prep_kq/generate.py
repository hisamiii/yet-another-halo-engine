#!/usr/bin/env python3
"""Generate the DeltaNet K/Q-norm prologue fixture for yah_deltanet_prep_kq_f32.loom.

Reference for BatchedDeltaNetPrepKqKernel in ssm_row_split.hip. One wave per
(token, key head) reduces three quantities over the 128 keys and lane 0 writes:

  k_sq = sum_i k[i]^2,  q_sq = sum_i q[i]^2,  kq = sum_i k[i]*q[i]
  out[0] = 1 / sqrt(k_sq + 1e-6)
  out[1] = (1 / sqrt(128)) * (1 / sqrt(q_sq + 1e-6))
  out[2] = kq

Every input is a power of two, so each product and every partial sum is exact in
f32 and the butterfly order the GPU uses cannot change the sum. That leaves only
the sqrt and the reciprocal, which is why the comparison carries a small tolerance
rather than being bit-exact.
"""
import os
import numpy as np

BATCH = 3
KEY_HEADS = 2
DIM = 128
QKV = 2 * KEY_HEADS * DIM
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    conv = np.zeros(BATCH * QKV, dtype=np.float32)
    expected = np.zeros(BATCH * KEY_HEADS * 3, dtype=np.float32)
    inv_sqrt_dim = np.float32(1.0) / np.sqrt(np.float32(DIM))
    for t in range(BATCH):
        for kh in range(KEY_HEADS):
            k_value = np.float32(0.25 if (t + kh) % 2 == 0 else 0.5)
            q_value = np.float32(0.125 if (t + kh) % 2 == 0 else 0.25)
            base = t * QKV
            conv[base + kh * DIM: base + kh * DIM + DIM] = q_value
            k_start = base + KEY_HEADS * DIM + kh * DIM
            conv[k_start: k_start + DIM] = k_value
            k_sq = np.float32(DIM) * k_value * k_value
            q_sq = np.float32(DIM) * q_value * q_value
            kq = np.float32(DIM) * k_value * q_value
            at = (t * KEY_HEADS + kh) * 3
            expected[at + 0] = np.float32(1.0) / np.sqrt(k_sq + np.float32(1e-6))
            expected[at + 1] = inv_sqrt_dim / np.sqrt(q_sq + np.float32(1e-6))
            expected[at + 2] = kq
    np.save(os.path.join(OUT, 'input_conv.npy'), conv)
    np.save(os.path.join(OUT, 'expected_scales.npy'), expected)
    print('wrote input_conv shape=%s expected_scales shape=%s'
          % (conv.shape, expected.shape))
    print('  scales=%s' % expected.reshape(BATCH, KEY_HEADS, 3)[0])


if __name__ == '__main__':
    main()
