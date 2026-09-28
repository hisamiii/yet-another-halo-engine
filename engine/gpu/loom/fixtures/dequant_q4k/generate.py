#!/usr/bin/env python3
"""Generate the Q4_K dequant expectation for yah_dequant_q4k_bf16.loom.

block_q4_K is 144 bytes:
  half d; half dmin; uint8 scales[12]; uint8 qs[128]
and element i of the 256-wide block is
  gg = i/64; wv = i%64; lane = wv%32; low = wv < 32
  qb = qs[gg*32 + lane]; quant4 = low ? qb & 15 : qb >> 4
  (sc, m) = GetQKScaleMin(scales, 2*gg + (low ? 0 : 1))
  value = d*sc*quant4 - dmin*m

The fixture sets d = 1, dmin = 0 and the scale bytes so every (sc, m) is (1, 0),
so the value is exactly the 4-bit quant code. qs varies per block, group and lane,
which pins the index mapping. bf16 has no numpy dtype, so the expectation is the
raw 16-bit pattern.
"""
import os
import numpy as np

BLOCKS = 2
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    blocks = bytearray()
    expected = []
    for b in range(BLOCKS):
        scales = bytes([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
        qs = bytes(((i * 7 + b * 3) % 256) for i in range(128))
        block = bytearray()
        block += np.float16(1.0).tobytes()
        block += np.float16(0.0).tobytes()
        block += scales
        block += qs
        assert len(block) == 144, len(block)
        blocks += block
        for i in range(256):
            gg = i // 64
            wv = i % 64
            lane = wv % 32
            low = wv < 32
            qb = qs[gg * 32 + lane]
            q4 = (qb & 0x0F) if low else (qb >> 4)
            bits = (np.float32(q4).view(np.uint32) >> 16).astype(np.uint16)
            expected.append(bits)
    exp = np.array(expected, dtype=np.uint16).view(np.int16)
    inp = np.frombuffer(bytes(blocks), dtype=np.int8)
    np.save(os.path.join(OUT, "input_q4k.npy"), inp)
    np.save(os.path.join(OUT, "expected_bits.npy"), exp)
    print("wrote input_q4k.npy", inp.shape, "expected_bits.npy", exp.shape)


if __name__ == "__main__":
    main()