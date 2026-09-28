#!/usr/bin/env python3
"""Generate the batched attention expectation for yah_attn_batched_f32.loom.

Reference for BatchedAttentionKernel in attention_batched.hip, f32 and f16 cache
paths. One workgroup per (head, token):

  seq_len   = start_pos + token + 1
  scale     = 1 / sqrt(head_dim)
  scores[p] = dot(q, k[p]) * scale            p < seq_len
  s         = softmax(scores[0..seq_len))
  ctx[d]    = sum_p s[p] * v[p][d]
  out[d]    = ctx[d] * sigmoid(gate[d])

gate is all 100, so exp(-100) flushes to zero in fp32 and sigmoid is exactly 1.0:
the gate is a no-op that still has to be read and applied. Every k, v and q value
is a multiple of 1/4 so the f16 cache conversion is exact too. The only inexact
step is the softmax exponential, which is why the comparison carries a tolerance.
"""
import os
import numpy as np

HEAD_DIM = 4
MAX_CTX = 2
START = 1
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    q = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    k = np.array([0.5, 0.25, 0.0, -0.25, 1.0, 2.0, -1.0, 0.5], dtype=np.float32)
    v = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    seq_len = START + 1
    scale = np.float32(1.0) / np.sqrt(np.float32(HEAD_DIM))
    scores = np.zeros(seq_len, dtype=np.float32)
    for p in range(seq_len):
        acc = np.float32(0.0)
        for d in range(HEAD_DIM):
            acc = np.float32(acc + np.float32(q[d] * k[p * HEAD_DIM + d]))
        scores[p] = np.float32(acc * scale)
    top = np.float32(np.max(scores))
    exp_scores = np.array([np.exp(np.float64(s - top)) for s in scores])
    total = np.float32(np.sum(exp_scores, dtype=np.float32))
    weights = exp_scores / np.float64(total)
    ctx = np.zeros(HEAD_DIM, dtype=np.float32)
    for d in range(HEAD_DIM):
        acc = np.float64(0.0)
        for p in range(seq_len):
            acc += weights[p] * np.float64(v[p * HEAD_DIM + d])
        ctx[d] = np.float32(acc)
    for name, array in (('input_q.npy', q), ('input_gate.npy', np.full(HEAD_DIM, 100.0, dtype=np.float32)),
                        ('input_k.npy', k), ('input_v.npy', v),
                        # The f16 cache case reads these. Every value is a multiple
                        # of 1/4, so the conversion is exact and the same
                        # expectation applies to both cache types.
                        ('input_k_f16.npy', k.astype(np.float16)),
                        ('input_v_f16.npy', v.astype(np.float16)),
                        ('expected_out.npy', ctx)):
        np.save(os.path.join(OUT, name), array)
        print('wrote %-20s shape=%s' % (name, array.shape))
    print('  scores=%s weights=%s' % (scores, weights))
    print('  expected=%s' % ctx)


if __name__ == '__main__':
    main()
