#!/usr/bin/env python3
"""Reference for the Loom embedding lookup with a device token pointer."""
import numpy as np
import os

np.save("token.npy", np.array([2], dtype=np.int32))


def decode_row_q3k(table, token, bpr):
    out = []
    for b in range(bpr):
        off = (int(token) * bpr + b) * 110
        blk = table[off:off + 110]
        hm = blk[0:32]
        qs = blk[32:96]
        sc = blk[96:108]
        d = np.frombuffer(blk[108:110].tobytes(), dtype=np.float16)[0].astype(np.float64)
        for i in range(256):
            group = i >> 5
            half = group >> 2
            sp = group & 3
            h16b = i & 16
            h16 = h16b >> 4
            lane_of = h16b + (i & 15)
            qsb = int(qs[half * 32 + lane_of])
            low = (qsb >> (2 * sp)) & 3
            high = (int(hm[lane_of]) >> (4 * half + sp)) & 1
            quant = (low | (high << 2)) - 4
            si = half * 8 + sp * 2 + h16
            low4 = (int(sc[si & 7]) >> (0 if si < 8 else 4)) & 0xF
            high2 = (int(sc[8 + (si & 3)]) >> (2 * (si >> 2))) & 3
            scale = (low4 | (high2 << 4)) - 32
            out.append(d * scale * float(quant))
    return np.array(out, dtype=np.float32)


here = os.path.dirname(os.path.abspath(__file__))
table = np.load(os.path.join(here, "..", "prefill_embed", "input_table.npy")).astype(np.uint8)
expected = decode_row_q3k(table, 1, 2)
np.save(os.path.join(here, "token_q3k.npy"), np.array([1], dtype=np.int32))
np.save(os.path.join(here, "expected_q3k.npy"), expected)
print("expected_q3k", expected[:6])