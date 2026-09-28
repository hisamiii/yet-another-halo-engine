#!/usr/bin/env python3
"""Generate the vision InjectRows expectation.

Reference for InjectRows in vision/device_input.hip:

  row = i / (width*hc)
  hidden[i] = embedding[row*width + i % width]

so every embedding row is broadcast hc times into the hidden rows. With hc = 1 the
result is the identity and any stride passes; hc = 2 makes the broadcast real, and
the flat result is then a repeating pattern inside each half rather than one
arithmetic sequence, which is why it is a fixture.
"""
import os
import numpy as np

ROWS = 2
WIDTH = 3
HC = 2
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    embedding = np.arange(ROWS * WIDTH, dtype=np.float32)
    total = ROWS * WIDTH * HC
    hidden = np.zeros(total, dtype=np.float32)
    for i in range(total):
        row = i // (WIDTH * HC)
        hidden[i] = embedding[row * WIDTH + i % WIDTH]
    np.save(os.path.join(OUT, 'expected_hidden.npy'), hidden)
    print('wrote expected_hidden.npy %s' % hidden)


if __name__ == '__main__':
    main()
