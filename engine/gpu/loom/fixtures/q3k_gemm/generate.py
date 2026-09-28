#!/usr/bin/env python3
"""Generate the Q3_K GEMM expectation for yah_ffn_gemm_q3k_f32.loom.

block_q3_K is 110 bytes:
  uint8 hmask[32]; uint8 qs[64]; uint8 scales[12]; half d

Element i of the 256-wide block is
  group = i//32; half = group//4; sp = group%4; h16 = (i%32)//16; j = i%16
  low  = (qs[half*32 + h16*16 + j] >> (2*sp)) & 3
  bit  = (hmask[h16*16 + j] >> (half*4 + sp)) & 1
  quant = (low | (bit<<2)) - 4                       in [-4, 3]
  si = half*8 + sp*2 + h16
  low4  = si < 8 ? scales[si] & 0xF : (scales[si-8] >> 4) & 0xF
  high2 = (scales[8 + si%4] >> (2*(si//4))) & 3
  scale = (low4 | (high2<<4)) - 32                   6-bit biased, signed
  value = f32(d) * scale * quant

d = 2^-9, so |scale*quant| <= 31*3 = 93 and every product is an exact multiple
of 2^-9, small enough that the f16 decode and the f32 row sum are exact. hmask,
qs and scales all vary per row and block, so a wrong plane, shift or scale nibble
changes the row sums. The activation is all ones and the output is token-major
out[token*16 + row] = sum_k value(row, k).
"""
import os
import numpy as np

ROWS = 16
K = 5120
BLOCK = 256
BLOCKS_PER_ROW = K // BLOCK
TOKENS = 64
OUT = os.path.dirname(os.path.abspath(__file__))


def decode_block(block):
    hmask = block[0:32]
    qs = block[32:96]
    scales = block[96:108]
    d = np.frombuffer(block[108:110], dtype=np.float16)[0].astype(np.float64)
    out = []
    for i in range(256):
        group = i // 32
        half = group // 4
        sp = group % 4
        h16 = (i % 32) // 16
        j = i % 16
        qs_idx = half * 32 + h16 * 16 + j
        hm_idx = h16 * 16 + j
        low = (int(qs[qs_idx]) >> (2 * sp)) & 3
        bit = (int(hmask[hm_idx]) >> (half * 4 + sp)) & 1
        quant = (low | (bit << 2)) - 4
        si = half * 8 + sp * 2 + h16
        low4 = (int(scales[si]) & 0xF) if si < 8 else ((int(scales[si - 8]) >> 4) & 0xF)
        high2 = (int(scales[8 + (si % 4)]) >> (2 * (si // 4))) & 3
        scale = (low4 | (high2 << 4)) - 32
        out.append(d * scale * quant)
    return out


def main():
    packed = bytearray()
    row_sums = []
    for r in range(ROWS):
        total = np.float64(0.0)
        for b in range(BLOCKS_PER_ROW):
            block = bytearray()
            block += bytes(((r * 3 + b * 5 + k * 7) % 256) for k in range(32))
            block += bytes(((r * 11 + b * 13 + k * 17) % 256) for k in range(64))
            block += bytes(((r * 7 + b * 3 + k * 5) % 256) for k in range(12))
            block += np.float16(2.0 ** -9).tobytes()
            assert len(block) == 110, len(block)
            packed += block
            total += np.float64(sum(decode_block(block)))
        row_sums.append(total)
    exp = np.zeros((TOKENS, ROWS), dtype=np.float32)
    for t in range(TOKENS):
        for r in range(ROWS):
            exp[t, r] = np.float32(row_sums[r])
    inp = np.frombuffer(bytes(packed), dtype=np.int8)
    np.save(os.path.join(OUT, "input_q3k_gemm.npy"), inp)
    np.save(os.path.join(OUT, "expected_out.npy"), exp.reshape(-1).astype(np.float32))
    print("wrote input_q3k_gemm.npy", inp.shape)
    print("row sums", [round(float(s), 4) for s in row_sums])


if __name__ == "__main__":
    main()