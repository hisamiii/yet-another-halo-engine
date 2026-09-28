#!/usr/bin/env python3
"""Generate the DeltaNet alpha/beta prep fixture for yah_deltanet_prep_ab_f32.loom.

Reference for BatchedDeltaNetPrepAlphaBetaKernel in ssm_row_split.hip, which does
two things in one launch:

  1. advance the conv history ring for every channel,
  2. compute the per-(token, head) alpha/beta pair for every token-head.

  for i < count, h = i % num_heads:
    alpha_biased   = alpha_buf[i] + ssm_dt[h]
    alpha_softplus = alpha_biased > 20 ? alpha_biased : log1p(exp(alpha_biased))
    ab[2i]     = exp(alpha_softplus * ssm_a[h])
    ab[2i + 1] = 1 / (1 + exp(-beta_buf[i]))

  for channel c < qkv_size:
    tail[j]      = batch + j >= 4 ? qkv_in[(batch + j - 4)*qkv_size + c]
                                 : conv_state[c*4 + batch + j]
    new_state[c*4 + j] = tail[j]

The history read index always exceeds the write index for batch >= 1, so the
in-place update is order-safe and the reference can do the same. The alpha/beta
half is transcendental and carries a tolerance; the history half is pure indexing
and is compared exactly.
"""
import os
import numpy as np

BATCH = 2
QKV = 16
HEADS = 4
COUNT = BATCH * HEADS
OUT = os.path.dirname(os.path.abspath(__file__))


def pattern(count, modulus, shift):
    idx = np.arange(count, dtype=np.int64)
    return (((idx * 7) % modulus) - shift).astype(np.float32) / np.float32(8.0)


def main():
    alpha = np.array([0.5, -0.25, 25.0, 1.0, 2.0, -1.5, 0.0, 3.0], dtype=np.float32)
    beta = pattern(COUNT, 9, 4)
    ssm_a = np.array([0.5, -1.0, -0.05, 2.0], dtype=np.float32)
    ssm_dt = np.array([0.0, 0.25, -0.5, 0.125], dtype=np.float32)
    qkv = pattern(BATCH * QKV, 13, 6)
    state = pattern(QKV * 4, 11, 5)

    ab = np.zeros(COUNT * 2, dtype=np.float32)
    for i in range(COUNT):
        h = i % HEADS
        biased = np.float32(alpha[i] + ssm_dt[h])
        if biased > np.float32(20.0):
            softplus = biased
        else:
            softplus = np.float32(np.log1p(np.exp(np.float64(biased))))
        ab[i * 2] = np.float32(np.exp(np.float64(softplus) * np.float64(ssm_a[h])))
        ab[i * 2 + 1] = np.float32(1.0 / (1.0 + np.exp(-np.float64(beta[i]))))

    new_state = state.copy()
    for c in range(QKV):
        tail = []
        for j in range(4):
            if BATCH + j >= 4:
                tail.append(qkv[(BATCH + j - 4) * QKV + c])
            else:
                tail.append(state[c * 4 + BATCH + j])
        for j in range(4):
            new_state[c * 4 + j] = tail[j]

    for name, array in (('input_alpha.npy', alpha), ('input_beta.npy', beta),
                        ('input_ssm_a.npy', ssm_a), ('input_ssm_dt.npy', ssm_dt),
                        ('input_qkv.npy', qkv), ('input_state.npy', state),
                        ('expected_ab.npy', ab),
                        ('expected_state.npy', new_state)):
        np.save(os.path.join(OUT, name), array)
        print('wrote %-22s shape=%s' % (name, array.shape))
    print('  clamped alpha (25.0) -> ab[4:6]=%s' % ab[4:6])
    print('  state ch0 %s -> %s' % (state[:4], new_state[:4]))


if __name__ == '__main__':
    main()
