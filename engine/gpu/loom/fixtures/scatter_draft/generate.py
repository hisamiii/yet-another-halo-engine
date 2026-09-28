#!/usr/bin/env python3
"""Reference for the Loom draft-probability scatter (fixtures/scatter_draft)."""
import numpy as np


def main():
    np.save("ids.npy", np.array([2, 5, 2, 5], dtype=np.int32))
    np.save("probs.npy", np.array([0.25, 0.5, 0.25, 0.5], dtype=np.float32))
    expected = np.zeros(8, dtype=np.float32)
    expected[2] = np.float32(0.5)
    expected[5] = np.float32(1.0)
    np.save("expected.npy", expected)
    print(expected)


if __name__ == "__main__":
    main()
