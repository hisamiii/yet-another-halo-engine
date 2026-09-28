#!/usr/bin/env python3
"""Generate the IQ4_NL GEMM expectation for yah_ffn_gemm_iq4nl_f32.loom.

block_iq4_nl is 18 bytes with QK=32: half d; uint8 qs[16]. Each qs byte holds
two nibbles; the low nibble is element local and the high nibble is local+16,
and each nibble indexes the 16-entry non-linear IQ4 codebook:

  value = f32(d) * kValuesIq4Nl[(qs[local % 16] >> (4*(local//16))) & 0xF]

with local = i % 32. d = 2^-4 keeps every codebook product exact in f16 and
every f32 row sum exact. qs varies per row and block, and the fixture visits
every one of the 16 codebook entries, so a wrong nibble half or a mis-ordered
codebook changes the sums.
"""
import os
import numpy as np

ROWS = 16
K = 5120
QK = 32
BLOCKS_PER_ROW = K // QK
TOKENS = 64
KVAL = [-127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
        1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0]
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    packed = bytearray()
    row_sums = []
    for r in range(ROWS):
        total = np.float64(0.0)
        for blk_i in range(BLOCKS_PER_ROW):
            qs = bytes(((r * 3 + blk_i * 5 + k * 7) % 256) for k in range(16))
            block = bytearray()
            block += np.float16(2.0 ** -4).tobytes()
            block += qs
            assert len(block) == 18, len(block)
            packed += block
            for local in range(QK):
                shift = 4 * (local // 16)
                nib = (qs[local % 16] >> shift) & 0xF
                total += np.float64(2.0 ** -4) * np.float64(KVAL[nib])
        row_sums.append(total)
    exp = np.zeros((TOKENS, ROWS), dtype=np.float32)
    for t in range(TOKENS):
        for r in range(ROWS):
            exp[t, r] = np.float32(row_sums[r])
    inp = np.frombuffer(bytes(packed), dtype=np.int8)
    np.save(os.path.join(OUT, "input_iq4nl_gemm.npy"), inp)
    np.save(os.path.join(OUT, "expected_out.npy"), exp.reshape(-1).astype(np.float32))
    print("wrote input_iq4nl_gemm.npy", inp.shape, "row bytes", inp.size // ROWS)
    print("row sums", [round(float(s), 3) for s in row_sums])


if __name__ == "__main__":
    main()