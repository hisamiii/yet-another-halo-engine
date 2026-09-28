#!/usr/bin/env python3
"""Reference for the Loom ATB C-operand decoder (fixtures/atb_decode_c).

groups = 2, l1_cols = 1, n_full = 256. shared = 133 makes mult = 1.0; the two
groups decode to 1..8 and -1..-8 at rows 0 and 1."""
import numpy as np


def main():
    packed = np.zeros(18, dtype=np.uint8)
    packed[0] = 133
    packed[1:9] = np.arange(1, 9)
    packed[9] = 133
    packed[10:18] = np.arange(255, 247, -1)
    np.save("packed.npy", packed.view(np.int8))
    expected = np.zeros((2, 256), dtype=np.float32)
    expected[0, 0:8] = np.arange(1, 9)
    expected[1, 0:8] = -np.arange(1, 9)
    np.save("expected.npy", expected)
    print(packed, expected[0, :8], expected[1, :8])


if __name__ == "__main__":
    main()
