#!/usr/bin/env python3
"""Reference for the Loom tiled-attention KV pack (fixtures/pack_tiled_attn_kv)."""
import numpy as np

LAYER = 0
START = 1
BATCH = 2
MAX_CONTEXT = 4
KV_HEADS = 2
HEAD_DIM = 8
KV_WIDTH = KV_HEADS * HEAD_DIM


def main():
    k = np.arange(BATCH * KV_WIDTH, dtype=np.float32)
    v = k.copy()
    k16 = np.zeros(MAX_CONTEXT * KV_WIDTH, dtype=np.float16)
    v16 = np.zeros(MAX_CONTEXT * KV_WIDTH, dtype=np.float16)
    k32 = np.zeros(KV_HEADS * MAX_CONTEXT * HEAD_DIM, dtype=np.float32)
    v32 = np.zeros(KV_HEADS * MAX_CONTEXT * HEAD_DIM, dtype=np.float32)
    for token in range(BATCH):
        for index in range(KV_WIDTH):
            source = token * KV_WIDTH + index
            position = START + token
            kv_head = index // HEAD_DIM
            dim = index % HEAD_DIM
            f16 = ((LAYER * MAX_CONTEXT + position) * KV_WIDTH) + index
            f32 = ((LAYER * KV_HEADS + kv_head) * MAX_CONTEXT + position) * HEAD_DIM + dim
            k16[f16] = np.float16(k[source])
            v16[f16] = np.float16(v[source])
            k32[f32] = k[source]
            v32[f32] = v[source]
    np.save("k16.npy", k16)
    np.save("v16.npy", v16)
    np.save("k32.npy", k32)
    np.save("v32.npy", v32)
    print(k16[16:48])


if __name__ == "__main__":
    main()
