#!/usr/bin/env python3
"""Reference for the Loom sparse sampling penalties (fixtures/sparse_penalties).

logits = 0..7; penalties = {(1, count 2, repeated), (3, count 0, fresh),
(5, count 4, repeated)}; (repeat, frequency, presence) = (2, 0.5, 1). Every
operation is exact in f32, so the row is compared with atol = 0."""
import numpy as np


def main():
    np.save("penalties.npy", np.array([[1, 2, 1], [3, 0, 0], [5, 4, 1]], dtype=np.int32))
    np.save("params.npy", np.array([2.0, 0.5, 1.0, 0.0], dtype=np.float32))
    expected = np.arange(8, dtype=np.float32)
    expected[1] = np.float32(-1.5)
    expected[3] = np.float32(3.0)
    expected[5] = np.float32(-0.5)
    np.save("expected.npy", expected)
    print(expected)


if __name__ == "__main__":
    main()
