#!/usr/bin/env python3
"""Generate the IQ2_XS GEMM expectation for yah_ffn_gemm_iq2xs_f32.loom.

block_iq2_xs is 74 bytes: half d; uint16 qs[32]; uint8 scales[8]. Each uint16
carries a 9-bit iq2xs grid index and a 7-bit sign index. Element i is:
  sub16 = i//16; ib32 = (sub16//2)%8; half = sub16%2
  li = (i%16)//8; j = i%8; l = 2*half + li
  code = qs[4*ib32 + l]
  g_word = iq2xs_grid[code & 511]; g_byte = (g_word >> (8*j)) & 0xFF
  signs = ksigns[code >> 9]; sign_bit = (signs >> j) & 1
  mag = sign_bit ? -g_byte : g_byte
  nib = half == 0 ? scales[ib32] & 0xF : scales[ib32] >> 4
  value = f32(d) * (0.5 + nib) * 0.25 * mag

d = 2^-6 makes the product 2^-9 * (2*nib+1) * mag, exact in f16 and exact in the
f32 row sum. qs and scales vary per row and block. The 512-entry grid and the
128-byte ksigns table are operands.
"""
import os
import numpy as np

ROWS = 16
K = 5120
QK = 256
BLOCKS_PER_ROW = K // QK
TOKENS = 64
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    grid = np.load(os.path.join(OUT, "grid.npy")).astype(np.uint64)
    ksigns = np.load(os.path.join(OUT, "ksigns.npy")).view(np.uint8)
    packed = bytearray()
    row_sums = []
    for r in range(ROWS):
        total = np.float64(0.0)
        for blk_i in range(BLOCKS_PER_ROW):
            qsv = [((r * 7 + blk_i * 11 + k * 13) % 65536) for k in range(32)]
            scales = bytes(((r * 17 + blk_i * 23 + k * 29) % 256) for k in range(8))
            block = bytearray()
            block += np.float16(2.0 ** -6).tobytes()
            block += np.array(qsv, dtype="<u2").tobytes()
            block += scales
            assert len(block) == 74, len(block)
            packed += block
            for i in range(256):
                sub16 = i // 16
                ib32 = (sub16 // 2) % 8
                half = sub16 % 2
                li = (i % 16) // 8
                j = i % 8
                l = 2 * half + li
                code = qsv[4 * ib32 + l]
                g_word = int(grid[code & 511])
                g_byte = (g_word >> (8 * j)) & 0xFF
                signs = int(ksigns[code >> 9])
                sbit = (signs >> j) & 1
                mag = -g_byte if sbit else g_byte
                sc = scales[ib32]
                nib = (sc & 0xF) if half == 0 else (sc >> 4)
                total += np.float64(2.0 ** -6) * (0.5 + nib) * 0.25 * np.float64(mag)
        row_sums.append(total)
    exp = np.zeros((TOKENS, ROWS), dtype=np.float32)
    for t in range(TOKENS):
        for r in range(ROWS):
            exp[t, r] = np.float32(row_sums[r])
    inp = np.frombuffer(bytes(packed), dtype=np.int8)
    np.save(os.path.join(OUT, "input_iq2xs_gemm.npy"), inp)
    np.save(os.path.join(OUT, "expected_out.npy"), exp.reshape(-1).astype(np.float32))
    print("wrote input_iq2xs_gemm.npy", inp.shape, "row bytes", inp.size // ROWS)
    print("row sums", [round(float(s), 3) for s in row_sums])


if __name__ == "__main__":
    main()