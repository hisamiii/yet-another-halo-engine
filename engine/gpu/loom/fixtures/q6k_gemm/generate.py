#!/usr/bin/env python3
"""Generate the Q6_K GEMM expectation for yah_ffn_gemm_q6k_f32.loom.

block_q6_K is 210 bytes, and d is stored LAST:
  uint8 ql[128]; uint8 qh[64]; int8 scales[16]; half d

Element i of the 256-wide block unpacks a low nibble from ql and a high two-bit
plane from qh, subtracts a 32 bias, and scales:
  half = i//128; within = i%128; segment = within//32; lane = within%32
  low  = segment 0/2 ? ql[half*64 + lane] & 0xF : ql[half*64 + 32 + lane] & 0xF
         (segments 2/3 take the high nibble of the same two bytes)
  high = (qh[half*32 + lane] >> (2*segment)) & 3
  sc   = scales[half*8 + lane//16 + segment*2]      (signed)
  value = f32(d) * sc * (((high << 4) | low) - 32)

d = 2^-9 and scales in [-15, 15], so |sc*quant| <= 480 and every product is
exact in f16. ql, qh and scales all vary per row and block, so a swapped ql/qh
offset, a wrong nibble half or a wrong scale nibble changes the row sums. The
activation is all ones and the output is token-major out[token*16 + row].
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
    ql = block[0:128]
    qh = block[128:192]
    scales = np.frombuffer(block[192:208], dtype=np.int8)
    d = np.frombuffer(block[208:210], dtype=np.float16)[0].astype(np.float64)
    out = []
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
        out.append(d * sc * (((high << 4) | low) - 32))
    return out


def main():
    packed = bytearray()
    row_sums = []
    for r in range(ROWS):
        total = np.float64(0.0)
        for b in range(BLOCKS_PER_ROW):
            block = bytearray()
            block += bytes(((r * 7 + b * 3 + k * 5) % 256) for k in range(128))
            block += bytes(((r * 13 + b * 11 + k * 17) % 256) for k in range(64))
            block += bytes(((((r * 5 + b * 7 + k * 3) % 31) - 15) & 0xFF) for k in range(16))
            block += np.float16(2.0 ** -9).tobytes()
            assert len(block) == 210, len(block)
            packed += block
            total += np.float64(sum(decode_block(block)))
        row_sums.append(total)
    exp = np.zeros((TOKENS, ROWS), dtype=np.float32)
    for t in range(TOKENS):
        for r in range(ROWS):
            exp[t, r] = np.float32(row_sums[r])
    inp = np.frombuffer(bytes(packed), dtype=np.int8)
    np.save(os.path.join(OUT, "input_q6k_gemm.npy"), inp)
    np.save(os.path.join(OUT, "expected_out.npy"), exp.reshape(-1).astype(np.float32))
    print("wrote input_q6k_gemm.npy", inp.shape)
    print("row sums", [round(float(s), 4) for s in row_sums])


if __name__ == "__main__":
    main()