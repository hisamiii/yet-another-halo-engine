#!/usr/bin/env python3
"""Generate the vision AttentionRows expectation.

Reference for AttentionRows in vision/encoder.hip. The attention output is
head-major and this pass transposes it to token-major:

  output[(begin + token)*kHidden + head*kHeadDim + d]
      = Bf16(input[(head*rows + token)*kHeadDim + d])

One token and one head at a time means the flat result is a permutation, not an
arithmetic sequence, and bf16 has no numpy dtype, so the expectation is written as
the raw 16-bit patterns of an int16 array and compared through check.tensor.view.
"""
import os
import numpy as np

HEADS = 2
HEAD_DIM = 3
ROWS = 2
BEGIN = 0
OUT = os.path.dirname(os.path.abspath(__file__))


def to_bf16_bits(values):
    u = values.astype(np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + bias) >> np.uint32(16)).astype(np.uint16)


def main():
    hidden = HEADS * HEAD_DIM
    source = np.arange(HEADS * ROWS * HEAD_DIM, dtype=np.float32)
    out = np.zeros((BEGIN + ROWS) * hidden, dtype=np.float32)
    for token in range(ROWS):
        for head in range(HEADS):
            for d in range(HEAD_DIM):
                src = (head * ROWS + token) * HEAD_DIM + d
                dst = (BEGIN + token) * hidden + head * HEAD_DIM + d
                out[dst] = source[src]
    bits = to_bf16_bits(out).astype(np.int16)
    np.save(os.path.join(OUT, 'expected_bits.npy'), bits)
    print('wrote expected_bits.npy %s' % bits)
    print('  values %s' % out)


if __name__ == '__main__':
    main()
