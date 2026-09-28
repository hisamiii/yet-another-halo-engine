#!/usr/bin/env python3
"""Generate the Q5_K GEMM expectation for yah_ffn_gemm_q5k_f32.loom.

block_q5_K is 176 bytes:
  half d; half dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128]
element i of the 256-wide block:
  gg = i//64; wv = i%64; lane = wv%32; low = wv < 32
  qb = qs[gg*32+lane]; quant4 = low ? qb & 15 : qb >> 4
  bit = 2*gg + (0 if low else 1)
  quant = quant4 + (16 if (qh[lane] >> bit) & 1 else 0)
  (sc, m) = get_scale_min(scales, 2*gg + (0 if low else 1))
  value = d*sc*quant - dmin*m

d = dmin = 0.5 and the scales stay below 64, so every value is a multiple of 0.5
and exact in f16; the activation is all ones and the output is token-major
out[token*16 + row] = sum_k value(row, k). The stride and the qh plane are both
varied per row, so a swapped row or a dropped fifth bit changes the row sums.
"""
import os
import numpy as np

ROWS = 16
K = 5120
BLOCK = 256
BLOCKS_PER_ROW = K // BLOCK
TOKENS = 64
OUT = os.path.dirname(os.path.abspath(__file__))


def get_scale_min(scales, j):
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    return (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4), (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)


def decode_block(block):
    d = np.frombuffer(block[0:2], dtype=np.float16)[0].astype(np.float64)
    dmin = np.frombuffer(block[2:4], dtype=np.float16)[0].astype(np.float64)
    scales = block[4:16]
    qh = block[16:48]
    qs = block[48:176]
    out = []
    for i in range(256):
        gg = i // 64
        wv = i % 64
        lane = wv % 32
        low = wv < 32
        qb = int(qs[gg * 32 + lane])
        q4 = (qb & 0x0F) if low else (qb >> 4)
        bit = 2 * gg + (0 if low else 1)
        quant = q4 + (16 if (int(qh[lane]) >> bit) & 1 else 0)
        sc, m = get_scale_min(scales, 2 * gg + (0 if low else 1))
        out.append(d * sc * quant - dmin * m)
    return out


def main():
    packed = bytearray()
    row_sums = []
    decoded_rows = []
    for r in range(ROWS):
        total = np.float64(0.0)
        row_dec = []
        for b in range(BLOCKS_PER_ROW):
            block = bytearray()
            block += np.float16(0.5).tobytes()
            block += np.float16(0.5).tobytes()
            block += bytes(((r * 5 + b * 7 + k * 3) % 64) for k in range(12))
            block += bytes(((r * 13 + b * 11 + k * 19) % 256) for k in range(32))
            block += bytes(((r * 11 + b * 13 + k * 17) % 256) for k in range(128))
            assert len(block) == 176, len(block)
            packed += block
            dec = decode_block(block)
            row_dec.extend(dec)
            total += np.float64(sum(dec))
        row_sums.append(total)
        decoded_rows.append(row_dec)
    # Quantized GEMV: a sparse activation pins specific k positions across block,
    # nibble and scale-pair boundaries, so the expected row dots are exact sums of
    # a handful of decoded weights. Same pattern as the Q4_K GEMV fixture.
    x = np.zeros(K, dtype=np.float32)
    for pos, val in [(0, 1.0), (1, 0.5), (32, 2.0), (33, -1.0), (64, 0.25),
                     (65, -0.5), (127, 1.5), (240, 0.75), (255, -2.0),
                     (256, 1.0), (257, -0.25), (300, 0.5), (1023, -1.0),
                     (4096, 0.125), (5119, 2.0)]:
        x[pos] = np.float32(val)
    gemv = np.zeros(ROWS, dtype=np.float32)
    for r in range(ROWS):
        acc = np.float64(0.0)
        for k in range(K):
            if x[k] != 0.0:
                acc += decoded_rows[r][k] * np.float64(x[k])
        gemv[r] = np.float32(acc)
    np.save(os.path.join(OUT, "gemv_x.npy"), x)
    np.save(os.path.join(OUT, "gemv_expected.npy"), gemv)
    print("gemv_expected", gemv[:4])
    exp = np.zeros((TOKENS, ROWS), dtype=np.float32)
    for t in range(TOKENS):
        for r in range(ROWS):
            exp[t, r] = np.float32(row_sums[r])
    inp = np.frombuffer(bytes(packed), dtype=np.int8)
    np.save(os.path.join(OUT, "input_q5k_gemm.npy"), inp)
    np.save(os.path.join(OUT, "expected_out.npy"), exp.reshape(-1).astype(np.float32))
    print("wrote input_q5k_gemm.npy", inp.shape)
    print("row sums", [round(float(s), 3) for s in row_sums])


if __name__ == "__main__":
    main()