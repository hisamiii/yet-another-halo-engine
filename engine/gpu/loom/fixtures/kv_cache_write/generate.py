#!/usr/bin/env python3
"""Generate the batched KV-cache write expectation for yah_kv_cache_write_f32.loom.

Reference for WriteBatchedKVCacheKernel in attention_batched.hip. For every
(batch b, kv head) it copies head_dim values of k and v to position

  pos = start_pos + b

into two caches that use different layouts:

  f32: index = ((layer*num_kv_heads + kv_head)*max_context + pos)*head_dim + i
  f16: index = ((layer*max_context + pos)*num_kv_heads*head_dim)
              + kv_head*head_dim + i

The two agree only when max_context == num_kv_heads, so this fixture uses
max_context=3 and num_kv_heads=2 to make them disagree. Both are pure indexing and
the f16 conversion is exact for the small integers used, so every expectation is
exact and the comparison carries no tolerance.
"""
import os
import numpy as np

BATCH = 2
NKH = 2
HD = 4
MAX_CTX = 3
LAYER = 0
START = 0
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    k = np.arange(BATCH * NKH * HD, dtype=np.float32)
    v = (np.arange(BATCH * NKH * HD, dtype=np.float32) + np.float32(100))
    size = NKH * MAX_CTX * HD
    k_cache = np.zeros(size, dtype=np.float32)
    v_cache = np.zeros(size, dtype=np.float32)
    k_cache_f16 = np.zeros(size, dtype=np.float16)
    v_cache_f16 = np.zeros(size, dtype=np.float16)
    kv_width = NKH * HD
    for b in range(BATCH):
        pos = START + b
        for kv in range(NKH):
            src = b * NKH * HD + kv * HD
            coff = ((LAYER * NKH + kv) * MAX_CTX + pos) * HD
            foff = ((LAYER * MAX_CTX + pos) * kv_width) + kv * HD
            for i in range(HD):
                k_cache[coff + i] = k[src + i]
                v_cache[coff + i] = v[src + i]
                k_cache_f16[foff + i] = np.float16(k[src + i])
                v_cache_f16[foff + i] = np.float16(v[src + i])
    for name, array in (('expected_k_cache.npy', k_cache),
                        ('expected_v_cache.npy', v_cache),
                        ('expected_k_cache_f16.npy', k_cache_f16),
                        ('expected_v_cache_f16.npy', v_cache_f16)):
        np.save(os.path.join(OUT, name), array)
        print('wrote %-26s shape=%s dtype=%s' % (name, array.shape, array.dtype))
    print('  k_cache      =%s' % k_cache)
    print('  k_cache_f16  =%s' % k_cache_f16)


if __name__ == '__main__':
    main()
