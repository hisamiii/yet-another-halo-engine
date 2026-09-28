#!/usr/bin/env python3
"""Reference bytes for the Loom SwiGLU Q8_1 quantizer (fixtures/q8_act_swiglu).

Mirrors BatchedFusedSwiGLUQuantizeQ8_1Kernel<true> for batch=16, k=32. gate is
20..531 and up is 1.0, so silu(gate) = gate exactly (exp(-20) is below the f32
half-ulp of 1). The output is the same 704-byte tiled layout as the base
quantizer: 576 payload bytes and a 128-byte activation-sum sidecar."""
import numpy as np

BATCH = 16
K = 32


def main():
    j = np.arange(K, dtype=np.float32)
    rows = np.stack([np.float32(20.0) + np.float32(tok * K) + j for tok in range(BATCH)])
    maxabs = np.max(np.abs(rows), axis=1).astype(np.float32)
    d = (maxabs / np.float32(127.0)).astype(np.float32)
    idv = np.where(d != np.float32(0.0), np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    scaled = (rows * idv[:, None]).astype(np.float32)
    q = np.floor(scaled.astype(np.float64) + 0.5).astype(np.int64).astype(np.int8)

    out = np.zeros(704, dtype=np.uint8)
    for tok in range(BATCH):
        for lane in range(32):
            half = lane // 16
            pos = lane % 16
            out[half * 256 + tok * 16 + pos] = q[tok, lane].astype(np.uint8)
        out[512 + tok * 4:512 + tok * 4 + 4] = np.frombuffer(d[tok].tobytes(), dtype=np.uint8)
        low = int(np.sum(q[tok, 0:16].astype(np.int64)))
        high = int(np.sum(q[tok, 16:32].astype(np.int64)))
        slot0 = np.float32(d[tok] * np.float32(low + high))
        slot1 = np.float32(d[tok] * np.float32(low))
        base = 576 + tok * 8
        out[base:base + 4] = np.frombuffer(slot0.tobytes(), dtype=np.uint8)
        out[base + 4:base + 8] = np.frombuffer(slot1.tobytes(), dtype=np.uint8)

    np.save("expected_bytes.npy", out.view(np.int8))
    print(out.shape, d[:3])


if __name__ == "__main__":
    main()
