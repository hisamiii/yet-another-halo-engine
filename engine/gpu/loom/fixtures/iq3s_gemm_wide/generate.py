#!/usr/bin/env python3
"""Generate IQ3_S GEMM expectations for every row-widened tile (widen_rows.py).

usage: generate.py <ROWS>          ROWS in {16, 32, 64}

Same arithmetic as fixtures/iq3s_gemm/generate.py, but the rows are all distinct.
A duplicated fixture would make a wrong row origin self-consistent -- the exact
bug the row widening can introduce -- so the upper rows use different qs/qh/
signs/scales patterns and a different row sum.

The activation is all ones, so the row sums do not depend on the token count and
every expectation is the same ROWS-vector tiled along the token axis; one file is
written per token width.
"""
import os, sys
import numpy as np

K = 5120
BLOCK = 256
BLOCKS_PER_ROW = K // BLOCK
TOKENS = (16, 32, 64, 128, 256)
OUT = os.path.dirname(os.path.abspath(__file__))
GRID = "/home/q/yet-another-halo-engine/engine/gpu/loom/fixtures/iq3s_gemm/grid.npy"


def main(rows):
    # int32, not uint32: the case binds tensor<512xi32> and the materializer
    # compares element types exactly (0x11 vs 0x12).
    grid = np.load(GRID).astype(np.int32)
    assert grid.shape == (512,)
    np.save(os.path.join(OUT, "grid.npy"), grid)
    packed = bytearray()
    row_sums = []
    for r in range(rows):
        total = np.float64(0.0)
        for blk_i in range(BLOCKS_PER_ROW):
            qs = bytes(((r * 11 + blk_i * 13 + k * 17) % 256) for k in range(64))
            qh = bytes(((r * 5 + blk_i * 7 + k * 3) % 256) for k in range(8))
            signs = bytes(((r * 19 + blk_i * 23 + k * 29) % 256) for k in range(32))
            scales = bytes(((r * 31 + blk_i * 37 + k * 41) % 256) for k in range(4))
            block = bytearray()
            block += np.float16(2.0 ** -6).tobytes()
            block += qs + qh + signs + scales
            assert len(block) == 110
            packed += block
            for i in range(256):
                group, half, j = i // 32, (i // 16) % 2, i % 16
                q, which, b = j // 8, (j // 4) % 2, j % 4
                l = 2 * half + q
                grid_lo = qs[group * 8 + 2 * l + which]
                hi_bit = (qh[group] >> (2 * l + which)) & 1
                g_byte = (int(grid[grid_lo | (hi_bit << 8)]) >> (8 * b)) & 0xFF
                signs_byte = signs[group * 4 + l]
                sign_nib = (signs_byte & 0xF) if which == 0 else (signs_byte >> 4)
                mag = -g_byte if ((sign_nib >> b) & 1) else g_byte
                scale_byte = scales[group // 2]
                nib = (scale_byte & 0xF) if (group % 2 == 0) else (scale_byte >> 4)
                total += np.float64(2.0 ** -6) * np.float64(1 + 2 * nib) * np.float64(mag)
        row_sums.append(total)
    inp = np.frombuffer(bytes(packed), dtype=np.int8)
    np.save(os.path.join(OUT, "input_r%d.npy" % rows), inp)
    for tok in TOKENS:
        exp = np.tile(np.array(row_sums, dtype=np.float32), (tok, 1)).reshape(-1)
        np.save(os.path.join(OUT, "expected_r%dt%d.npy" % (rows, tok)), exp)
    print("rows=%d weight=%d sums=[%s]" % (rows, inp.shape[0],
          ", ".join("%.1f" % float(s) for s in row_sums[:6])))


if __name__ == "__main__":
    main(int(sys.argv[1]))
