#!/usr/bin/env python3
"""Generate the -inf expectation for the non-finite branch of PrepareSamplingKernel.

Reference for PrepareSamplingKernel in sample.hip. The finite branch replaces a
non-finite logit with -infinity:

  adjusted[i] = isfinite(logits[i]) ? logits[i] : -INFINITY
  token_ids[i] = i

The finite half of the check needs no fixture -- an iota input comes back as the same
iota and ids are the index. The non-finite half does: the input is a fill of 3.5e38,
which overflows fp32 to +infinity, and every output must then be -infinity. numpy can
hold that, and check.expect.equal compares it exactly.
"""
import os
import numpy as np

VOCAB = 8
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    expected = np.full(VOCAB, -np.inf, dtype=np.float32)
    np.save(os.path.join(OUT, 'expected_neg_inf.npy'), expected)
    print('wrote expected_neg_inf.npy %s' % expected)


if __name__ == '__main__':
    main()
