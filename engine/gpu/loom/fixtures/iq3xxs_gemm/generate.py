#!/usr/bin/env python3
"""Generate the IQ3_XXS GEMM expectation for yah_ffn_gemm_iq3xxs_f32.loom.

block_iq3_xxs is 98 bytes:
  half d; uint8 qs[96]
where qs[0..63] index the 256-entry iq3xxs grid (four magnitudes per word) and
qs[64..95] are four aux words, one per 32-element group. Element i maps through
the HIP DecodeQuantSub16 IQ3_XXS arm:
  group = i//32; half = (i//16)%2; j = i%16; q = j//8; which = (j/4)%2; b = j%4
  l = 2*half + q
  g_word = grid[qs[8*group + 2*l + which]]
  g_byte = (g_word >> (8*b)) & 0xFF
  aux32  = LE32(qs[64 + 4*group .. +3])
  signs  = ksigns[(aux32 >> (7*l)) & 127]
  sign_nib = which == 0 ? signs & 0xF : signs >> 4
  sign_bit = (sign_nib >> b) & 1
  mag = sign_bit ? -g_byte : g_byte
  scale = f32(d) * (0.5 + (aux32 >> 28)) * 0.5
  value = scale * mag

Both the grid and the 128-entry ksigns table are operands: Loom has no embedded
constant array. d = 2^-4 makes scale = 2^-5 * (2n+1), so every product is exact
in f16 and every f32 row sum is exact. qs and the aux words vary per row and
block, and both magnitude signs appear.
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
    ksigns = np.load(os.path.join(OUT, "ksigns.npy")).view(np.uint8)
    assert grid.shape == (256,) and ksigns.shape == (128,)
    packed = bytearray()
    row_sums = []
    decoded_rows = []
    for r in range(ROWS):
        total = np.float64(0.0)
        row_dec = []
        for blk_i in range(BLOCKS_PER_ROW):
            qs = bytes(((r * 13 + blk_i * 17 + k * 19) % 256) for k in range(96))
            block = bytearray()
            block += np.float16(2.0 ** -4).tobytes()
            block += qs
            assert len(block) == 98, len(block)
            packed += block
            for i in range(256):
                group = i // 32
                half = (i // 16) % 2
                j = i % 16
                q = j // 8
                which = (j // 4) % 2
                b = j % 4
                l = 2 * half + q
                g_word = int(grid[qs[8 * group + 2 * l + which]])
                g_byte = (g_word >> (8 * b)) & 0xFF
                aux32 = (qs[64 + 4 * group] | (qs[64 + 4 * group + 1] << 8) |
                         (qs[64 + 4 * group + 2] << 16) | (qs[64 + 4 * group + 3] << 24))
                signs = int(ksigns[(aux32 >> (7 * l)) & 127])
                sign_nib = (signs & 0xF) if which == 0 else (signs >> 4)
                sign_bit = (sign_nib >> b) & 1
                mag = -g_byte if sign_bit else g_byte
                scale = (2.0 ** -4) * (0.5 + float(aux32 >> 28)) * 0.5
                v = np.float64(scale) * np.float64(mag)
                row_dec.append(v)
                total += v
        row_sums.append(total)
        decoded_rows.append(row_dec)
    # Quantized GEMV: the same sparse activation as the other GEMV fixtures.
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
    np.save(os.path.join(OUT, "input_iq3xxs_gemm.npy"), inp)
    np.save(os.path.join(OUT, "expected_out.npy"), exp.reshape(-1).astype(np.float32))
    print("wrote input_iq3xxs_gemm.npy", inp.shape)
    print("row sums", [round(float(s), 3) for s in row_sums])


if __name__ == "__main__":
    main()