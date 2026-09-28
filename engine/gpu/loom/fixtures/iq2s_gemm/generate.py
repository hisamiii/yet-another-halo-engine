#!/usr/bin/env python3
"""Generate the IQ2_S GEMM expectation for yah_ffn_gemm_iq2s_f32.loom.

block_iq2_s is 82 bytes: half d; uint8 qs[64]; uint8 qh[8]; uint8 scales[8].
qs[0..31] are 8-bit grid indices whose two high bits live in qh; qs[32..63] are
the per-group sign bytes. Element i is
  sub16 = i/16; ib32 = (sub16/2)%8; half = sub16%2
  li = (i%16)/8; j = i%8; l = 2*half + li
  low = qs[4*ib32 + l]; high = (qh[ib32] << (8 - 2*l)) & 0x300
  g_word = iq2s_grid[low | high]; g_byte = (g_word >> (8*j)) & 0xFF
  signs = qs[32 + 4*ib32 + l]; sign_bit = (signs >> j) & 1
  mag = sign_bit ? -g_byte : g_byte
  nib = half == 0 ? scales[ib32] & 0xF : scales[ib32] >> 4
  value = f32(d) * (0.5 + nib) * 0.25 * mag

d = 2^-6 keeps every product exact in f16 and every f32 row sum exact. qs, qh
and scales vary per row and block, and the 1024-entry grid is passed as the
little-endian i32 word pairs because Loom has no 64-bit element load here.
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
    packed = bytearray()
    row_sums = []
    for r in range(ROWS):
        total = np.float64(0.0)
        for blk_i in range(BLOCKS_PER_ROW):
            qs = bytes(((r * 7 + blk_i * 11 + k * 13) % 256) for k in range(64))
            qh = bytes(((r * 17 + blk_i * 19 + k * 23) % 256) for k in range(8))
            scales = bytes(((r * 29 + blk_i * 31 + k * 37) % 256) for k in range(8))
            block = bytearray()
            block += np.float16(2.0 ** -6).tobytes()
            block += qs
            block += qh
            block += scales
            assert len(block) == 82, len(block)
            packed += block
            for i in range(256):
                sub16 = i // 16
                ib32 = (sub16 // 2) % 8
                half = sub16 % 2
                li = (i % 16) // 8
                j = i % 8
                l = 2 * half + li
                low = qs[4 * ib32 + l]
                high = (qh[ib32] << (8 - 2 * l)) & 0x300
                g_word = int(grid[low | high])
                g_byte = (g_word >> (8 * j)) & 0xFF
                signs = qs[32 + 4 * ib32 + l]
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
    np.save(os.path.join(OUT, "input_iq2s_gemm.npy"), inp)
    np.save(os.path.join(OUT, "expected_out.npy"), exp.reshape(-1).astype(np.float32))
    print("wrote input_iq2s_gemm.npy", inp.shape, "row bytes", inp.size // ROWS)
    print("row sums", [round(float(s), 3) for s in row_sums])


if __name__ == "__main__":
    main()