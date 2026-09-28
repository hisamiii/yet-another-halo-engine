#!/usr/bin/env python3
"""Reference for the Loom Q8_1 tail zeroer (fixtures/q8_act_tail).

batch = 17, num_blocks = 1 gives two 16-token tiles and tail = 1. The kernel keeps
token 0 of the second tile and zeroes the payload, the scale and both sums of
tokens 1..15. The input is all 0xFF."""
import numpy as np

BATCH = 17
NUM_BLOCKS = 1
TILE_BYTES = 576
TILE_WORDS = 704
TILES = (BATCH + 15) // 16


def main():
    total = TILES * NUM_BLOCKS * TILE_WORDS
    data = np.full(total, 0xFF, dtype=np.uint8)
    expected = data.copy()
    tail = BATCH % 16
    lanes = 16 - tail
    tt = TILES - 1
    for idx in range(NUM_BLOCKS * lanes):
        kb = idx // lanes
        tl = tail + (idx % lanes)
        tile = tt * NUM_BLOCKS + kb
        base = tile * TILE_BYTES
        expected[base + tl * 16:base + tl * 16 + 16] = 0
        expected[base + 256 + tl * 16:base + 256 + tl * 16 + 16] = 0
        expected[base + 512 + tl * 4:base + 512 + tl * 4 + 4] = 0
        sidecar = TILES * NUM_BLOCKS * TILE_BYTES
        slot = sidecar + (tile * 16 + tl) * 2 * 4
        expected[slot:slot + 8] = 0
    np.save("input_bytes.npy", data.view(np.int8))
    np.save("expected_bytes.npy", expected.view(np.int8))
    print(data.shape, expected.shape, int(np.count_nonzero(data != expected)))


if __name__ == "__main__":
    main()
