#!/usr/bin/env python3
"""Generate the IQ2_XXS GEMM fixture for yah_ffn_gemm_iq2xxs_f32.loom.

block_iq2_xxs is 66 bytes: half d; uint16 qs[32]. Each group of 32 elements is
four 8-bit iq2xxs grid indices (bytes 0..3) followed by a word whose high nibble
is the 4-bit scale and whose low 28 bits are four 7-bit sign indices. Element i:
  ib32 = i//32; l = (i%32)//8; j = i%8
  code = qs_bytes[8*ib32 + l]
  high = u32 at qs_bytes[8*ib32 + 4]
  db = f32(d) * (0.5 + (high >> 28)) * 0.25
  mag = grid_byte(code, j); sign = (ksigns[(high >> 7*l) & 127] >> j) & 1
  value = sign ? -db*mag : db*mag
"""
import os, re, numpy as np

ROWS, K, QK, TOKENS = 16, 5120, 256, 64
BLOCKS = K // QK
OUT = os.path.dirname(os.path.abspath(__file__))
HDR = os.path.join(os.path.dirname(OUT), "..", "..", "ported", "src", "core", "quant", "iq_grids.hpp")
HDR = os.path.abspath(HDR)

def load_tables():
    txt = open(HDR).read()
    m = re.search(r"kIq2XxsGrid\[256\] = \{(.*?)\};", txt, re.S)
    grid = np.array([int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]+)ULL", m.group(1))], dtype=np.uint64)
    assert grid.size == 256, grid.size
    m2 = re.search(r"kKsignsIq2xs\[128\] = \{(.*?)\};", txt, re.S)
    ks = np.array([int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]+)U", m2.group(1))], dtype=np.uint8)
    assert ks.size == 128, ks.size
    return grid, ks

def main():
    grid, ks = load_tables()
    gbytes = grid.view(np.uint8).reshape(256, 8)
    np.save(os.path.join(OUT, "grid32.npy"), grid.view(np.int32))
    np.save(os.path.join(OUT, "ksigns.npy"), ks.view(np.int8))
    packed = bytearray(); row_sums = []
    for r in range(ROWS):
        total = np.float64(0.0)
        for b in range(BLOCKS):
            codes = [(r * 7 + b * 11 + k * 13) % 256 for k in range(4)]
            high = 0
            high |= ((r * 3 + b * 5) % 16) << 28
            for l in range(4):
                high |= ((r * 17 + b * 23 + l * 31) % 128) << (7 * l)
            block = bytearray()
            block += np.float16(2.0 ** -6).tobytes()
            for g in range(8):
                for l in range(4):
                    block.append(codes[l])
                block += np.uint32(high).tobytes()
            assert len(block) == 66, len(block)
            packed += block
            for i in range(256):
                ib32 = i // 32; l = (i % 32) // 8; j = i % 8
                code = codes[l]
                db = np.float64(2.0 ** -6) * (0.5 + (high >> 28)) * 0.25
                mag = int(gbytes[code, j])
                sbit = (int(ks[(high >> (7 * l)) & 127]) >> j) & 1
                total += db * (-mag if sbit else mag)
        row_sums.append(total)
    exp = np.zeros((TOKENS, ROWS), dtype=np.float32)
    for t in range(TOKENS):
        for r in range(ROWS):
            exp[t, r] = np.float32(row_sums[r])
    np.save(os.path.join(OUT, "input_iq2xxs_gemm.npy"), np.frombuffer(bytes(packed), dtype=np.int8))
    np.save(os.path.join(OUT, "expected_out.npy"), exp.reshape(-1).astype(np.float32))
    import math
    silu = lambda x: x / (1.0 + math.exp(-x))
    sw = np.zeros((TOKENS, ROWS), dtype=np.float16)
    for t in range(TOKENS):
        for r in range(ROWS):
            sw[t, r] = np.float16(silu(1.0) * 32.0 * row_sums[r])
    np.save(os.path.join(OUT, "expected_swiglu.npy"), sw.reshape(-1).astype(np.float16))
    print("wrote iq2xxs fixture; row bytes", len(packed) // ROWS)

if __name__ == "__main__":
    main()