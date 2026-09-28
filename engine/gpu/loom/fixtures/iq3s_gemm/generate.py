#!/usr/bin/env python3
"""Generate the IQ3_S GEMM expectation for yah_ffn_gemm_iq3s_f32.loom.

block_iq3_s is 110 bytes:
  half d; uint8 qs[64]; uint8 qh[8]; uint8 signs[32]; uint8 scales[4]

Each group of four elements is one entry of the 512-word iq3s grid (four 4-bit
magnitudes packed least-significant byte first), with one sign bit per element
and one 4-bit scale per 32 elements. Element i of the block maps to a grid word
through the HIP DecodeQuantSub16 IQ3_S arm:
  group = i//32; half = (i//16)%2; j = i%16; q = j//8; which = (j//4)%2; b = j%4
  l = 2*half + q
  grid_lo = qs[group*8 + 2*l + which]
  hi_bit  = (qh[group] >> (2*l + which)) & 1
  g_word  = grid[grid_lo | (hi_bit << 8)]
  g_byte  = (g_word >> (8*b)) & 0xFF
  signs_byte = signs[group*4 + l]
  sign_nib = which == 0 ? signs_byte & 0xF : signs_byte >> 4
  sign_bit = (sign_nib >> b) & 1
  mag = sign_bit ? -g_byte : g_byte
  scale_byte = scales[group//2]
  nib = group%2 == 0 ? scale_byte & 0xF : scale_byte >> 4
  value = f32(d) * (1 + 2*nib) * mag

The 512-word iq3s grid is loaded from a fixture, because Loom has no embedded
constant-array form and the table is not arithmetically derivable; the kernel
takes it as an extra read-only operand. d = 2^-6 keeps every product exact in
f16 and every partial sum exact in f32. qs, qh, signs and scales all vary per
row and block, and both signs of magnitude appear, so a wrong grid bit, sign
nibble or scale nibble changes the row sums.
"""
import os
import numpy as np

ROWS = 16
K = 5120
BLOCK = 256
BLOCKS_PER_ROW = K // BLOCK
TOKENS = 64
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    grid = np.load(os.path.join(OUT, "grid.npy")).astype(np.uint32)
    assert grid.shape == (512,)
    packed = bytearray()
    row_sums = []
    for r in range(ROWS):
        total = np.float64(0.0)
        for blk_i in range(BLOCKS_PER_ROW):
            qs = bytes(((r * 11 + blk_i * 13 + k * 17) % 256) for k in range(64))
            qh = bytes(((r * 5 + blk_i * 7 + k * 3) % 256) for k in range(8))
            signs = bytes(((r * 19 + blk_i * 23 + k * 29) % 256) for k in range(32))
            scales = bytes(((r * 31 + blk_i * 37 + k * 41) % 256) for k in range(4))
            block = bytearray()
            block += np.float16(2.0 ** -6).tobytes()
            block += qs
            block += qh
            block += signs
            block += scales
            assert len(block) == 110, len(block)
            packed += block
            for i in range(256):
                group = i // 32
                half = (i // 16) % 2
                j = i % 16
                q = j // 8
                which = (j // 4) % 2
                b = j % 4
                l = 2 * half + q
                grid_lo = qs[group * 8 + 2 * l + which]
                hi_bit = (qh[group] >> (2 * l + which)) & 1
                g_word = int(grid[grid_lo | (hi_bit << 8)])
                g_byte = (g_word >> (8 * b)) & 0xFF
                signs_byte = signs[group * 4 + l]
                sign_nib = (signs_byte & 0xF) if which == 0 else (signs_byte >> 4)
                sign_bit = (sign_nib >> b) & 1
                mag = -g_byte if sign_bit else g_byte
                scale_byte = scales[group // 2]
                nib = (scale_byte & 0xF) if (group % 2 == 0) else (scale_byte >> 4)
                total += np.float64(2.0 ** -6) * np.float64(1 + 2 * nib) * np.float64(mag)
        row_sums.append(total)
    exp = np.zeros((TOKENS, ROWS), dtype=np.float32)
    for t in range(TOKENS):
        for r in range(ROWS):
            exp[t, r] = np.float32(row_sums[r])
    inp = np.frombuffer(bytes(packed), dtype=np.int8)
    np.save(os.path.join(OUT, "input_iq3s_gemm.npy"), inp)
    np.save(os.path.join(OUT, "expected_out.npy"), exp.reshape(-1).astype(np.float32))
    print("wrote input_iq3s_gemm.npy", inp.shape)
    print("row sums", [round(float(s), 3) for s in row_sums])


if __name__ == "__main__":
    main()