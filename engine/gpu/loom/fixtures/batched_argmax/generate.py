#!/usr/bin/env python3
"""Reference for the Loom batched argmax (fixtures/batched_argmax)."""
import numpy as np


def main():
    logits = np.array([[1, 3, 2, 3, 0, 5, 5, 4],
                       [0, -1, 7, 7, 7, 2, 3, 4]], dtype=np.float32)
    np.save("logits.npy", logits)
    np.save("expected.npy", np.array([5, 2], dtype=np.int32))
    print(logits)


if __name__ == "__main__":
    main()
