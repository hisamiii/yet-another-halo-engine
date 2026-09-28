#!/usr/bin/env python3
"""Generate the Q4_K GEMM expectation for yah_ffn_gemm_q4k_f32.loom.

Weight is 16 output rows x 5120 k, packed as block_q4_K (144 bytes per 256
weights, 20 blocks per row). Element k of row r:
  blk = k // 256; i = k % 256
  gg = i//64; wv = i%64; lane = wv%32; low = wv < 32
  qb = qs[gg*32+lane]; quant4 = low ? qb & 15 : qb >> 4
  (sc, m) = get_scale_min(scales, 2*gg + (low ? 0 : 1))
  value = d*sc*quant4 - dmin*m

The scales are all below 64, so the high-branch high bits are zero and both
scale branches are still exercised through sc/m values that differ. d and dmin
are 0.5, so every decoded value is a multiple of 0.5 and the f16 conversion is
exact. The activation is all ones and the output is token-major
out[token*16 + row] = sum_k value(row, k), so each row sum is repeated for all 64
tokens. The sum is exact in f32 (well below 2^24).
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
    qs = block[16:144]
    out = []
    for i in range(256):
        gg = i // 64
        wv = i % 64
        lane = wv % 32
        low = wv < 32
        qb = int(qs[gg * 32 + lane])
        q4 = (qb & 0x0F) if low else (qb >> 4)
        sc, m = get_scale_min(scales, 2 * gg + (0 if low else 1))
        out.append(d * sc * q4 - dmin * m)
    return out


def main():
    packed = bytearray()
    row_sums = []
    decoded_rows = []
    for r in range(ROWS):
        total = np.float64(0.0)
        decoded = []
        for b in range(BLOCKS_PER_ROW):
            block = bytearray()
            block += np.float16(0.5).tobytes()
            block += np.float16(0.5).tobytes()
            block += bytes(((r * 5 + b * 7 + k * 3) % 64) for k in range(12))
            block += bytes(((r * 11 + b * 13 + k * 17) % 256) for k in range(128))
            assert len(block) == 144, len(block)
            packed += block
            vals = decode_block(block)
            decoded.extend(vals)
            total += np.float64(sum(vals))
        row_sums.append(total)
        decoded_rows.append(np.array(decoded, dtype=np.float64))
    exp = np.zeros((TOKENS, ROWS), dtype=np.float32)
    for t in range(TOKENS):
        for r in range(ROWS):
            exp[t, r] = np.float32(row_sums[r])
    inp = np.frombuffer(bytes(packed), dtype=np.int8)
    np.save(os.path.join(OUT, "input_q4k_gemm.npy"), inp)
    # Quantized GEMV: a sparse activation pins specific k positions across block,
    # nibble and scale-pair boundaries, so the expected row dots are exact sums of
    # a handful of decoded weights.
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
    np.save(os.path.join(OUT, "expected_out.npy"), exp.reshape(-1).astype(np.float32))
    # Residual epilogue: the kernel adds the accumulator into the existing f32
    # rows, so the residual case initialises the output to 1000 and the expected
    # is 1000 + the row sum. Both are exact in f32.
    residual_init = np.float32(1000.0)
    np.save(os.path.join(OUT, "expected_residual.npy"),
            (exp + residual_init).reshape(-1).astype(np.float32))
    # SwiGLU epilogue: activation is scaled to 1/16 so the accumulator stays
    # inside the f16 range, gate is 1, and the output is
    # round_f16(g * sigmoid(g) * acc) with g = 1.
    g = np.float32(1.0)
    sig = np.float32(1.0) / (np.float32(1.0) + np.float32(np.exp(np.float32(-g))))
    silu = np.float32(g * sig)
    acc16 = exp / np.float32(16.0)
    swi = np.float16(silu * acc16)
    np.save(os.path.join(OUT, "expected_swiglu.npy"), swi.reshape(-1).astype(np.float16))
    # Paired gate/up epilogue: both weights are this fixture, so the two
    # accumulators are equal, and the activation 2^-13 keeps them near 38 so that
    # silu(acc) = acc and the output fits f16. Expected = round_f16(silu(acc)*acc).
    acc_gateup = exp * np.float64(2.0 ** -13)
    silu_g = acc_gateup / (np.float64(1.0) + np.exp(-acc_gateup))
    gu = np.float16(silu_g * acc_gateup)
    np.save(os.path.join(OUT, "expected_gateup.npy"), gu.reshape(-1).astype(np.float16))
    print("gateup row0 acc", float(acc_gateup[0][0]), "out", float(gu[0][0]))
    print("swiglu row0", float(silu), [float(v) for v in swi[0][:4]])
    print("wrote input_q4k_gemm.npy", inp.shape)
    print("wrote expected_out.npy", exp.reshape(-1).shape)
    print("wrote expected_residual.npy", exp.reshape(-1).shape)
    print("row sums", [round(float(s), 3) for s in row_sums])


if __name__ == "__main__":
    main()