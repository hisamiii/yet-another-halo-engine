#!/usr/bin/env python3
"""Reference for the Loom WMMA attention head pack (fixtures/pack_attn_heads).

head_dim = 8, kv_heads = 1, length = 16. values = iota 0..127, so the packed value
buffer is the 16 keys x 8 dims -> 8 dims x 16 keys transpose."""
import numpy as np

HEAD_DIM = 8
KV_HEADS = 1
LENGTH = 16
PADDED = 16


def main():
    values = np.arange(LENGTH * KV_HEADS * HEAD_DIM, dtype=np.float16)
    pv = np.zeros(KV_HEADS * PADDED * HEAD_DIM, dtype=np.float16)
    for head in range(KV_HEADS):
        for keyblock in range(LENGTH // 16):
            for dim in range(HEAD_DIM):
                for key in range(16):
                    src = ((keyblock * 16 + key) * KV_HEADS + head) * HEAD_DIM + dim
                    dst = (head * PADDED * HEAD_DIM + keyblock * 16 * HEAD_DIM
                           + dim * 16 + key)
                    pv[dst] = values[src]
    np.save("packed_v.npy", pv)
    print(pv)


if __name__ == "__main__":
    main()
