#!/usr/bin/env python3
"""Reference bits for the Loom QkvRope rotate case (fixtures/vision_qkv_rope).

Mirrors models/qwen/vision/encoder.hip QkvRope for count=4, grid_width=2 with
input = 1 and bias = 0. Then q0 = q1 = k0 = k1 = 1, so q and k are exactly the
rotation matrix entries: cos - sin in the first 18-wide band, sin + cos in the
second. v is 1.0 everywhere and is checked in-source.

Output is a single int16 array of 4608 elements. q and k are bit-identical
because q0 = q1 = k0 = k1 = 1, so one fixture covers both. bf16 has no numpy
dtype, so the raw 16-bit patterns are stored and the case reinterprets the
kernel output with check.tensor.view.
"""
import math
import numpy as np

COUNT = 4
GRID_WIDTH = 2
HIDDEN = 1152
HEADDIM = 72


def bf16_bits(value):
    """Round a float to bfloat16 (nearest even) and return its signed int16 bits."""
    bits = int(np.float32(value).view(np.uint32))
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFF0000
    hi = (bits >> 16) & 0xFFFF
    return hi - 0x10000 if hi >= 0x8000 else hi


def main():
    out = np.zeros(2 * COUNT * HIDDEN, dtype=np.int16)
    for token in range(COUNT):
        merge = token // 4
        y = (merge // (GRID_WIDTH // 2)) * 2 + (token % 4) // 2
        x = (merge % (GRID_WIDTH // 2)) * 2 + token % 2
        for head in range(16):
            for d in range(HEADDIM):
                pair = d % 36
                kk = pair % 18
                position = y if pair < 18 else x
                angle = position / (10000.0 ** (kk / 18.0))
                cosine = np.float32(math.cos(angle))
                sine = np.float32(math.sin(angle))
                if d < 36:
                    rotated = np.float32(cosine - sine)
                else:
                    rotated = np.float32(sine + cosine)
                target = (head * COUNT + token) * HEADDIM + d
                out[target] = bf16_bits(rotated)
                out[COUNT * HIDDEN + target] = bf16_bits(rotated)
    np.save("expected_bits.npy", out[:COUNT * HIDDEN])
    print(out[:COUNT * HIDDEN].shape, out.dtype, out[:6])


if __name__ == "__main__":
    main()
