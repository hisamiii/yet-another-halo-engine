#!/usr/bin/env python3
"""Reference for the Loom fused SSM gate Q8_1 quantizer (fixtures/fused_ssm_gate_q8).

num_heads = 1, batch = 1, val_dim = 32, has_gate = 0, raw = 1, ssm_norm = 1, eps = 0.
sum_sq = 32, mean = 1, inv = 1, so the single block quantizes to q = 127 and
d = 1/127 in the tiled layout with the sum sidecar."""
import numpy as np

NUM_BLOCKS = 1
TILES = 1


def main():
    d = np.float32(1.0) / np.float32(127.0)
    q = np.full(32, 127, dtype=np.int8)
    data = np.zeros(TILES * NUM_BLOCKS * 704, dtype=np.uint8)
    for blk in range(NUM_BLOCKS):
        base = blk * 576
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
        slot = sidecar + (blk * 16 + 0) * 2 * 4
        data[slot:slot + 4] = np.frombuffer(slot0.tobytes(), dtype=np.uint8)
        data[slot + 4:slot + 8] = np.frombuffer(slot1.tobytes(), dtype=np.uint8)
    np.save("expected_bytes.npy", data.view(np.int8))
    print(data.shape)


def emit_gated():
    d = np.float32(100.0) / np.float32(127.0)
    q = np.full(32, 127, dtype=np.int8)
    data = np.zeros(TILES * NUM_BLOCKS * 704, dtype=np.uint8)
    for blk in range(NUM_BLOCKS):
        base = blk * 576
        for lane in range(32):
            half = lane // 16
            pos = lane % 16
            data[base + half * 256 + pos] = q[lane].astype(np.uint8)
        data[base + 512:base + 512 + 4] = np.frombuffer(d.tobytes(), dtype=np.uint8)
        low = int(np.sum(q[0:16].astype(np.int64)))
        high = int(np.sum(q[16:32].astype(np.int64)))
        slot0 = np.float32(d * np.float32(low + high))
        slot1 = np.float32(d * np.float32(low))
        sidecar = TILES * NUM_BLOCKS * 576
        slot = sidecar + (blk * 16) * 2 * 4
        data[slot:slot + 4] = np.frombuffer(slot0.tobytes(), dtype=np.uint8)
        data[slot + 4:slot + 8] = np.frombuffer(slot1.tobytes(), dtype=np.uint8)
    np.save("expected_bytes_gated.npy", data.view(np.int8))


if __name__ == "__main__":
    main()
    emit_gated()
