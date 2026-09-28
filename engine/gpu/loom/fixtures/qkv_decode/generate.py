#!/usr/bin/env python3
"""Fixtures for yah_qkv_decode_f32.loom.

Pads each existing GEMV weight fixture to the QKV band max size (q band 4200
bytes/row = 67200 for 16 rows; k/v band 5440 bytes/row = 87040) and adds a
hand-built Q8_0 fixture. All GEMV fixtures already share the same sparse x, so
the three expected vectors are just the per-format GEMV expectations.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.dirname(HERE)
ROWS = 16
K = 5120
MAXBYTES = 87040  # kv band max for 16 rows


def pad(fmt):
    src = np.load(os.path.join(FIX, fmt + "_gemm", "input_" + fmt + "_gemm.npy"))
    src = src.astype(np.uint8)
    assert src.size <= MAXBYTES, (fmt, src.size)
    out = np.zeros(MAXBYTES, dtype=np.uint8)
    out[:src.size] = src
    np.save(os.path.join(HERE, fmt + "_pad.npy"), out.view(np.int8))
    exp = np.load(os.path.join(FIX, fmt + "_gemm", "gemv_expected.npy")).astype(np.float32)
    np.save(os.path.join(HERE, fmt + "_expected.npy"), exp)
    print(fmt, "size", src.size, "exp", exp[:3])


def main():
    x = np.load(os.path.join(FIX, "q4k_gemm", "gemv_x.npy")).astype(np.float32)
    assert x.shape == (K,)
    np.save(os.path.join(HERE, "x.npy"), x)
    for fmt in ["q4k", "iq4xs", "iq4nl", "q5k", "q6k"]:
        pad(fmt)
    # Q8_0: block_q8_0 = {half d; int8 qs[32]}, 34 bytes, row stride 20*272?
    # hidden/32 = 160 blocks, row = 160*34 = 5440 bytes, 256-group = 8 blocks = 272.
    packed = bytearray()
    expected = np.zeros(ROWS, dtype=np.float32)
    for r in range(ROWS):
        acc = np.float64(0.0)
        for b in range(160):
            qs = bytes((((r * 7 + b * 11 + k * 3) % 255) - 127) & 0xFF for k in range(32))
            block = bytearray()
            block += np.float16(1.0).tobytes()
            block += qs
            assert len(block) == 34
            packed += block
            for k in range(32):
                idx = b * 32 + k
                q = np.frombuffer(qs, dtype=np.int8)[k]
                acc += np.float64(1.0) * np.float64(q) * np.float64(x[idx])
        expected[r] = np.float32(acc)
    out = np.zeros(MAXBYTES, dtype=np.uint8)
    raw = np.frombuffer(bytes(packed), dtype=np.int8)
    assert raw.size == ROWS * 5440
    out[:raw.size] = raw.view(np.uint8)
    np.save(os.path.join(HERE, "q8_0_pad.npy"), out.view(np.int8))
    np.save(os.path.join(HERE, "q8_0_expected.npy"), expected)
    print("q8_0 expected", expected[:3])


if __name__ == "__main__":
    main()