#!/usr/bin/env python3
"""Reference for the Loom PatchPosition port (fixtures/vision_patch_position).

Replicates models/qwen/vision/encoder.hip PatchPosition on the same config the
check uses: count=4, grid_height=48, grid_width=48. With a 48x48 grid the two
scales are exactly 1.0, so wy = wx = 0 and only the (y0, x0) sample contributes;
the four decompositions land on rows 0, 1, 48 and 49.

Every tensor is `e mod 17`, so the reference only has to mirror the integer
indices and the three bf16 round trips, not any transcendental.
"""
import numpy as np

HIDDEN = 1152
COUNT = 4
GRID_HEIGHT = 48
GRID_WIDTH = 48
PERIOD = 17


def bf16(x):
    """Round a float32 array to bfloat16 and back, matching float(Bf16(x))."""
    a = np.asarray(x, dtype=np.float32)
    bits = a.view(np.uint32)
    # Round to nearest even on the low 16 bits.
    lsb = (bits >> 16) & 1
    rounding = 0x7FFF + lsb
    bits = (bits + rounding) & 0xFFFF0000
    return bits.view(np.float32)


def main():
    out = np.zeros(COUNT * HIDDEN, dtype=np.float32)
    for patch in range(COUNT):
        merge = patch // 4
        y = (merge // (GRID_WIDTH // 2)) * 2 + (patch % 4) // 2
        x = (merge % (GRID_WIDTH // 2)) * 2 + patch % 2
        fy = float(y) * (47.0 / (GRID_HEIGHT - 1))
        fx = float(x) * (47.0 / (GRID_WIDTH - 1))
        y0 = min(int(fy), 47)
        x0 = min(int(fx), 47)
        row = y0 * 48 + x0
        for dim in range(HIDDEN):
            e = patch * HIDDEN + dim
            h = float(e % PERIOD)
            b = float(dim % PERIOD)
            pos = float((row * HIDDEN + dim) % PERIOD)
            value = bf16(np.float32(bf16(np.float32(h)) + np.float32(b)))
            total = bf16(value + bf16(np.float32(pos)))
            out[e] = total
    np.save("expected.npy", out)


if __name__ == "__main__":
    main()
