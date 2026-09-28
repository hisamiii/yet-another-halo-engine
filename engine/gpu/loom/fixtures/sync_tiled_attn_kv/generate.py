#!/usr/bin/env python3
"""Reference for the Loom tiled-attention KV prefix sync (fixtures/sync_tiled_attn_kv)."""
import numpy as np

LAYER = 0
PREFIX = 2
MAX_CONTEXT = 4
KV_HEADS = 2
HEAD_DIM = 8
KV_WIDTH = KV_HEADS * HEAD_DIM


def main():
    k = np.arange(KV_HEADS * MAX_CONTEXT * HEAD_DIM, dtype=np.float32)
    v = k.copy()
    k16 = np.zeros(PREFIX * KV_WIDTH, dtype=np.float16)
    v16 = np.zeros(PREFIX * KV_WIDTH, dtype=np.float16)
    for pos in range(PREFIX):
        for index in range(KV_WIDTH):
            kv_head = index // HEAD_DIM
            dim = index % HEAD_DIM
            f32 = ((LAYER * KV_HEADS + kv_head) * MAX_CONTEXT + pos) * HEAD_DIM + dim
            f16 = ((LAYER * MAX_CONTEXT + pos) * KV_WIDTH) + index
            k16[f16] = np.float16(k[f32])
            v16[f16] = np.float16(v[f32])
    np.save("k16.npy", k16)
    np.save("v16.npy", v16)
    print(k16)


if __name__ == "__main__":
    main()
