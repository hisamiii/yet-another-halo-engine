#!/usr/bin/env python3
"""Generate the Q8_0 GEMM expectation for yah_ffn_gemm_q8_0_f32.loom.

block_q8_0 is 34 bytes:
  half d; int8 qs[32]
element e of the block: value = f32(d) * qs[e]

d is 2^-5 and qs is kept in [-63, 63], so every value is a multiple of 2^-5 and
exact in f16; the row sum stays well under 2^24 and is exact in f32. qs varies per
row, block and position and is signed on both sides of zero, so a dropped sign
extension or a swapped block changes the row sums. The activation is all ones and
the output is token-major out[token*16 + row] = sum_k value(row, k).
"""
import os
import numpy as np

ROWS = 16
K = 5120
BLOCK = 32
BLOCKS_PER_ROW = K // BLOCK
TOKENS = 64
OUT = os.path.dirname(os.path.abspath(__file__))


def decode_block(block):
    d = np.frombuffer(block[0:2], dtype=np.float16)[0].astype(np.float64)
    qs = np.frombuffer(block[2:34], dtype=np.int8)
    return [d * float(q) for q in qs]


def main():
    packed = bytearray()
    row_sums = []
    for r in range(ROWS):
        total = np.float64(0.0)
        for b in range(BLOCKS_PER_ROW):
            block = bytearray()
            block += np.float16(2.0 ** -5).tobytes()
            block += bytes((((r * 11 + b * 13 + k * 17) % 127) - 63) & 0xFF for k in range(32))
            assert len(block) == 34, len(block)
            packed += block
            total += np.float64(sum(decode_block(block)))
        row_sums.append(total)
    exp = np.zeros((TOKENS, ROWS), dtype=np.float32)
    for t in range(TOKENS):
        for r in range(ROWS):
            exp[t, r] = np.float32(row_sums[r])
    inp = np.frombuffer(bytes(packed), dtype=np.int8)
    np.save(os.path.join(OUT, "input_q8_0_gemm.npy"), inp)
    np.save(os.path.join(OUT, "expected_out.npy"), exp.reshape(-1).astype(np.float32))
    print("wrote input_q8_0_gemm.npy", inp.shape)
    print("row sums", [round(float(s), 4) for s in row_sums])


if __name__ == "__main__":
    main()