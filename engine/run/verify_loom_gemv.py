#!/usr/bin/env python3
"""Verify loom_gemv_probe output for a Q6_K tensor against a float64 oracle.

usage: verify_loom_gemv.py <model.gguf> <tensor> <file_offset> <y.bin> [rows]

Reads the tensor bytes straight from the GGUF at the exact file offset the
probe prints (`file_offset=`), decodes Q6_K in float64, dots against the same
x the probe used (x[k] = ((k%17)-8)*0.25), and compares the first `rows`
(default 16) values.
"""
import struct
import sys

import numpy as np


def decode_block(block):
    ql = block[0:128]
    qh = block[128:192]
    scales = np.frombuffer(block[192:208], dtype=np.int8)
    d = np.frombuffer(block[208:210], dtype=np.float16)[0].astype(np.float64)
    out = np.empty(256, dtype=np.float64)
    for i in range(256):
        half = i // 128
        within = i % 128
        segment = within // 32
        lane = within % 32
        ql_base = half * 64
        qh_byte = int(qh[half * 32 + lane]) & 0xFF
        if segment == 0:
            low = int(ql[ql_base + lane]) & 0x0F
            high = (qh_byte >> 0) & 3
        elif segment == 1:
            low = int(ql[ql_base + 32 + lane]) & 0x0F
            high = (qh_byte >> 2) & 3
        elif segment == 2:
            low = (int(ql[ql_base + lane]) & 0xFF) >> 4
            high = (qh_byte >> 4) & 3
        else:
            low = (int(ql[ql_base + 32 + lane]) & 0xFF) >> 4
            high = (qh_byte >> 6) & 3
        sc = int(scales[half * 8 + lane // 16 + segment * 2])
        out[i] = d * sc * (((high << 4) | low) - 32)
    return out


def main():
    model, tensor, offset, y_path = sys.argv[1:5]
    rows = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    offset = int(offset)
    k = None
    with open(model, "rb") as f:
        f.seek(offset)
        # Decode just `rows` rows, reading one 210-byte block at a time.
        x = None
        expected = np.zeros(rows, dtype=np.float64)
        for r in range(rows):
            acc = np.float64(0.0)
            for b in range(20):
                block = f.read(210)
                dec = decode_block(block)
                if x is None:
                    k = 20 * 256
                    x = np.array([((i % 17) - 8) * 0.25 for i in range(k)],
                                 dtype=np.float64)
                acc += np.dot(dec, x[b * 256:(b + 1) * 256])
            expected[r] = acc
    y = np.fromfile(y_path, dtype=np.float32)[:rows].astype(np.float64)
    diff = np.abs(y - expected)
    rel = diff / np.maximum(np.abs(expected), 1e-30)
    print("loom  ", y[:4])
    print("oracle", expected[:4])
    print("max_abs", float(diff.max()), "max_rel", float(rel.max()))
    ok = diff.max() < 1e-3 and rel.max() < 1e-4
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())