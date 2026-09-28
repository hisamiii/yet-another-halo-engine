#!/usr/bin/env python3
"""Generate the ATB packed-head expectations for the expand and add kernels.

Reference for AtbExpandHeadFp16Kernel and AtbAddHeadFp32Kernel in
prefill_fp16.hip. Both walk a packed [rows, head_cols] region and touch the first
head_cols columns of a full-width [rows, full_cols] row:

  expand: out[r*full_cols + c]  = packed[r*head_cols + c]
  add:    out[r*full_cols + c] += packed[r*head_cols + c]

for c < head_cols; the remaining columns are the other producer's business and must
be left alone. That untouched tail is what makes the expectation inexpressible as a
fill or an iota, so it is recorded here. The packed values are an iota the Loom case
can generate, so only the expected array needs a fixture.
"""
import os
import numpy as np

ROWS = 2
HEAD_COLS = 3
FULL_COLS = 5
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    packed = np.arange(ROWS * HEAD_COLS, dtype=np.float32)
    expand = np.full(ROWS * FULL_COLS, -1.0, dtype=np.float32)
    add = np.full(ROWS * FULL_COLS, 10.0, dtype=np.float32)
    for r in range(ROWS):
        for c in range(HEAD_COLS):
            expand[r * FULL_COLS + c] = packed[r * HEAD_COLS + c]
            add[r * FULL_COLS + c] += packed[r * HEAD_COLS + c]
    for name, array in (('expected_expand.npy', expand.astype(np.float16)),
                        ('expected_add.npy', add)):
        np.save(os.path.join(OUT, name), array)
        print('wrote %-22s %s' % (name, array))


if __name__ == '__main__':
    main()
