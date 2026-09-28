#!/usr/bin/env python3
"""Generate the decode QG-unpack expectation.

Reference for UnpackQGKernel in unpack.hip. The interleaved buffer holds two
head_dim vectors per head, Q then gate:

  q[tid]    = qg[head*2*head_dim + d]
  gate[tid] = qg[head*2*head_dim + head_dim + d]

with tid = head*head_dim + d. Two heads of three is enough to show the
de-interleave: the flat Q output is not an arithmetic sequence across the head
boundary, so a fixture is what pins it.
"""
import os
import numpy as np

NUM_HEADS = 2
HEAD_DIM = 3
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    total = NUM_HEADS * HEAD_DIM
    qg = np.arange(NUM_HEADS * 2 * HEAD_DIM, dtype=np.float32)
    q = np.zeros(total, dtype=np.float32)
    gate = np.zeros(total, dtype=np.float32)
    for h in range(NUM_HEADS):
        for d in range(HEAD_DIM):
            tid = h * HEAD_DIM + d
            q[tid] = qg[h * 2 * HEAD_DIM + d]
            gate[tid] = qg[h * 2 * HEAD_DIM + HEAD_DIM + d]
    np.save(os.path.join(OUT, 'expected_q.npy'), q)
    np.save(os.path.join(OUT, 'expected_gate.npy'), gate)
    print('wrote expected_q=%s expected_gate=%s' % (q, gate))


if __name__ == '__main__':
    main()
