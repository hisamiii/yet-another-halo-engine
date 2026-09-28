#!/usr/bin/env python3
"""Generate the IQ4_XS GEMM expectation for yah_ffn_gemm_iq4xs_f32.loom.

block_iq4_xs is 136 bytes:
  half d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128]
A block holds 256 weights in eight 32-element groups. Element e:
  group = e//32; within = e%32; L = within%16; high = within >= 16
  qb = qs[group*16 + L]; nib = (qb >> 4) if high else (qb & 15)
  sc = ((scales_l[group//2] >> (4*(group%2))) & 15
        | (((scales_h >> (2*group)) & 3) << 4)) - 32
  value = f32(d) * sc * kValuesIq4Nl[nib]

The codebook is the 16-entry non-linear IQ4 table {-127,-104,...,113}. d is 2^-13
and scales_h is fixed at 0xAAAA, which makes every sc land in [0,15], so
|sc * code| < 2048 and every product is exact in f16. The scales_l nibbles and qs
both vary per row and block, so a swapped group, nibble half or scale nibble
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
CODEBOOK = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
OUT = os.path.dirname(os.path.abspath(__file__))


def decode_block(block):
    d = np.frombuffer(block[0:2], dtype=np.float16)[0].astype(np.float64)
    scales_h = int.from_bytes(block[2:4], "little")
    scales_l = block[4:8]
    qs = block[8:136]
    out = []
    for e in range(256):
        group = e // 32
        within = e % 32
        L = within % 16
        high = within >= 16
        qb = int(qs[group * 16 + L])
        nib = (qb >> 4) if high else (qb & 15)
        sc = (((int(scales_l[group // 2]) >> (4 * (group % 2))) & 15) |
              (((scales_h >> (2 * group)) & 3) << 4)) - 32
        out.append(d * sc * CODEBOOK[nib])
    return out


def main():
    packed = bytearray()
    row_sums = []
    for r in range(ROWS):
        total = np.float64(0.0)
        for b in range(BLOCKS_PER_ROW):
            block = bytearray()
            block += np.float16(2.0 ** -13).tobytes()
            block += (0xAAAA).to_bytes(2, "little")
            block += bytes(((r * 3 + b * 5 + k * 7) % 16) for k in range(4))
            block += bytes(((r * 11 + b * 13 + k * 17) % 256) for k in range(128))
            assert len(block) == 136, len(block)
            packed += block
            total += np.float64(sum(decode_block(block)))
        row_sums.append(total)
    exp = np.zeros((TOKENS, ROWS), dtype=np.float32)
    for t in range(TOKENS):
        for r in range(ROWS):
            exp[t, r] = np.float32(row_sums[r])
    inp = np.frombuffer(bytes(packed), dtype=np.int8)
    np.save(os.path.join(OUT, "input_iq4xs_gemm.npy"), inp)
    np.save(os.path.join(OUT, "expected_out.npy"), exp.reshape(-1).astype(np.float32))
    print("wrote input_iq4xs_gemm.npy", inp.shape)
    print("row sums", [round(float(s), 6) for s in row_sums])


if __name__ == "__main__":
    main()