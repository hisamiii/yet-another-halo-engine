#!/usr/bin/env python3
"""Reference for the Loom ATB gate+up SwiGLU decoder (fixtures/atb_decode_swiglu).

groups = 1, n_full = 16. shared = 133 makes both scales 1.0. Gate bytes 100..107,
up bytes 1..8, so the first eight outputs are exactly (100+i)*(1+i); the rest of
the row is untouched and stays zero."""
import numpy as np


def main():
    gate = np.zeros(9, dtype=np.uint8)
    gate[0] = 133
    gate[1:9] = np.arange(100, 108)
    up = np.zeros(9, dtype=np.uint8)
    up[0] = 133
    up[1:9] = np.arange(1, 9)
    np.save("gate.npy", gate.view(np.int8))
    np.save("up.npy", up.view(np.int8))
    expected = np.zeros((1, 16), dtype=np.float16)
    expected[0, 0:8] = (np.arange(100, 108) * np.arange(1, 9)).astype(np.float16)
    np.save("expected.npy", expected)
    print(expected)


if __name__ == "__main__":
    main()
