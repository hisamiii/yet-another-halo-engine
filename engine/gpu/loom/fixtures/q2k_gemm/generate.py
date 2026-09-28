#!/usr/bin/env python3
"""Generate the Q2_K GEMM fixture for yah_ffn_gemm_q2k_f32.loom.

block_q2_K is 84 bytes: uint8 scales[16]; uint8 qs[64]; half d; half dmin.
Element i: sub=i//16; l=i%16; j=(sub%8)//2; sc=scales[sub];
  q=qs[(sub//8)*32 + (sub%2)*16 + l]
  value = d*(sc&15)*((q>>(2*j))&3) - dmin*(sc>>4)
The kernel rounds each decoded weight to f16 before the MMA, so the expectation
rounds the same way.
"""
import os, numpy as np
ROWS, K, QK, TOKENS = 16, 5120, 256, 64
BLOCKS = K // QK
OUT = os.path.dirname(os.path.abspath(__file__))

def main():
    packed = bytearray(); row_sums = []
    for r in range(ROWS):
        total = np.float64(0.0)
        for b in range(BLOCKS):
            scales = bytes(((r * 17 + b * 23 + k * 29) % 256) for k in range(16))
            qs = bytes(((r * 7 + b * 11 + k * 13) % 256) for k in range(64))
            d = np.float16(2.0 ** -6); dmin = np.float16(2.0 ** -7)
            block = bytearray(); block += scales; block += qs
            block += d.tobytes(); block += dmin.tobytes()
            assert len(block) == 84
            packed += block
            for i in range(256):
                sub = i // 16; l = i % 16; j = (sub % 8) // 2
                sc = scales[sub]
                q = qs[(sub // 8) * 32 + (sub % 2) * 16 + l]
                v = np.float64(d) * (sc & 15) * ((q >> (2 * j)) & 3) - np.float64(dmin) * (sc >> 4)
                total += np.float64(np.float16(v))
        row_sums.append(total)
    exp = np.zeros((TOKENS, ROWS), dtype=np.float32)
    for t in range(TOKENS):
        for r in range(ROWS):
            exp[t, r] = np.float32(row_sums[r])
    np.save(os.path.join(OUT, "input_q2k_gemm.npy"), np.frombuffer(bytes(packed), dtype=np.int8))
    np.save(os.path.join(OUT, "expected_out.npy"), exp.reshape(-1).astype(np.float32))
    print("wrote q2k fixture; row bytes", len(packed) // ROWS, "row0", float(row_sums[0]))

if __name__ == "__main__":
    main()