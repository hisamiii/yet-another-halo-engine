#!/usr/bin/env python3
"""Reference for the Loom fused RMSNorm + Q8_1 quantizer (fixtures/fused_rmsnorm_q8).

batch = 1, dim = 64, x = 1, residual = 0, weight = 1, eps = 0. sum_sq = 64, so
mean = 1 and inv_rms = 1 exactly; every normed value is 1. Each of the two blocks
quantizes to q = 127 with d = 1/127, in the same 704-byte-per-tile layout as the
base quantizer."""
import numpy as np

BATCH = 1
DIM = 64
NUM_BLOCKS = DIM // 32
TILES = 1


def main():
    d = np.float32(1.0) / np.float32(127.0)
    q = np.full(32, 127, dtype=np.int8)
    data = np.zeros(TILES * NUM_BLOCKS * 704, dtype=np.uint8)
    for blk in range(NUM_BLOCKS):
        tile = blk  # tt = 0
        base = tile * 576
        for lane in range(32):
            half = lane // 16
            pos = lane % 16
            data[base + half * 256 + 0 * 16 + pos] = q[lane].astype(np.uint8)
        data[base + 512:base + 512 + 4] = np.frombuffer(d.tobytes(), dtype=np.uint8)
        low = int(np.sum(q[0:16].astype(np.int64)))
        high = int(np.sum(q[16:32].astype(np.int64)))
        slot0 = np.float32(d * np.float32(low + high))
        slot1 = np.float32(d * np.float32(low))
        sidecar = TILES * NUM_BLOCKS * 576
        slot = sidecar + (tile * 16 + 0) * 2 * 4
        data[slot:slot + 4] = np.frombuffer(slot0.tobytes(), dtype=np.uint8)
        data[slot + 4:slot + 8] = np.frombuffer(slot1.tobytes(), dtype=np.uint8)
    np.save("expected_bytes.npy", data.view(np.int8))
    print(data.shape)


if __name__ == "__main__":
    main()
