#!/usr/bin/env python3
"""Generate the causal softmax masked-row expectation.

Reference for CausalSoftmaxKernel in attention_batched.hip when the row is not yet
fully visible. With uniform scores the max subtracts out exactly, exp(0) is exactly
1, and the sum is the visible count, so no transcendental is evaluated away from
0 and the expectation is exact rather than a tolerance. Only the causal-zero tail
needs recording, because the softmax prefix is a constant that a fill cannot
mix with zeros.
"""
import os
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    # start_pos = 0, query = 0, sequence_length = 4 -> visible = 1.
    np.save(os.path.join(OUT, 'expected_masked.npy'),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    print('wrote expected_masked.npy = [1, 0, 0, 0]')


if __name__ == '__main__':
    main()
