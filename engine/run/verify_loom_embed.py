#!/usr/bin/env python3
"""Verify loom_embed_probe output for a Q3_K token_embd against a float64 oracle.

usage: verify_loom_embed.py <model.gguf> <file_offset> <hidden.bin> <id>...
"""
import struct
import sys

import numpy as np


def decode_row(raw, offset, tok, hidden, bpr):
    raw.seek(offset + tok * (bpr * 110))
    row = raw.read(bpr * 110)
    vals = np.empty(hidden, dtype=np.float64)
    for b in range(bpr):
        blk = row[b * 110:(b + 1) * 110]
        hm = blk[0:32]
        qs = blk[32:96]
        sc = blk[96:108]
        d = np.frombuffer(blk[108:110], dtype=np.float16)[0].astype(np.float64)
        base = b * 256
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
            vals[base + i] = d * scale * float(quant)
    return vals


def main():
    model = sys.argv[1]
    offset = int(sys.argv[2])
    hidden_path = sys.argv[3]
    tokens = [int(x) for x in sys.argv[4:]]
    hidden = 5120
    bpr = 20
    raw = open(model, "rb")
    loom = np.fromfile(hidden_path, dtype=np.float32).reshape(len(tokens), hidden)
    worst = 0.0
    for idx, tok in enumerate(tokens):
        exp = decode_row(raw, offset, tok, hidden, bpr).astype(np.float32)
        diff = float(np.max(np.abs(loom[idx] - exp)))
        worst = max(worst, diff)
        print("token", tok, "max_abs", diff)
    print("overall max_abs", worst)
    ok = worst < 1e-5
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())