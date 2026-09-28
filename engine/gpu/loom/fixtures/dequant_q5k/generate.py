#!/usr/bin/env python3
"""Generate the Q5_K block fixture for yah_dequant_q5k_bf16.loom.

Reference for DequantizeQ5KToBf16Kernel / Q5KValueFp in prefill_gemm.hip and
quant_ops.hpp. A block_q5_K is 176 bytes:

  __half d; __half dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128]

and one element is

  gg = i/64; wv = i%64; lane = wv%32; low_half = wv < 32
  qb       = qs[gg*32 + lane]
  quant4   = low_half ? qb & 0x0F : qb >> 4
  bit      = 2*gg + (low_half ? 0 : 1)
  quant    = quant4 + (qh[lane] >> bit & 1 ? 16 : 0)
  sis      = 2*gg + (low_half ? 0 : 1)
  (sc, m)  = scales pair sis from the 12-byte packed array
  value    = f32(d)*sc*quant - f32(dmin)*m

The packed scale decode is the one from GetQKScaleMin (Q5_K shares the Q4_K
encoding). This fixture pins every bit of it: the codes and the high-bit plane are
both varied, and the scale bytes are chosen so every sc is 1 and every m is 0, so
the expected value is exactly the quant code and the index mapping is not hidden
behind a uniform result.

bf16 has no numpy dtype, so the expectation is written as the raw 16-bit patterns
in an int16 array; the Loom case reinterprets the kernel output with
check.tensor.view and compares the bits.
"""
import os
import numpy as np

BLOCKS = 2
ELEMS = 256
BYTES = 176
OUT = os.path.dirname(os.path.abspath(__file__))


def scales_pair(sis, scales):
    if sis < 4:
        return int(scales[sis] & 0x3F), int(scales[sis + 4] & 0x3F)
    sc = ((int(scales[sis + 4]) & 0x0F) | ((int(scales[sis - 4]) >> 6) << 4)) & 0xFF
    m = ((int(scales[sis + 4]) >> 4) | ((int(scales[sis]) >> 6) << 4)) & 0xFF
    return sc, m


def value_fp32(block, index):
    d = np.frombuffer(block[0:2].tobytes(), dtype=np.float16)[0].astype(np.float32)
    dmin = np.frombuffer(block[2:4].tobytes(), dtype=np.float16)[0].astype(np.float32)
    scales = block[4:16]
    qh = block[16:48]
    qs = block[48:176]
    gg = index // 64
    wv = index % 64
    lane = wv % 32
    low_half = wv < 32
    qb = int(qs[gg * 32 + lane])
    quant4 = (qb & 0x0F) if low_half else (qb >> 4)
    bit = 2 * gg + (0 if low_half else 1)
    quant = quant4 + (16 if ((int(qh[lane]) >> bit) & 1) else 0)
    sis = 2 * gg + (0 if low_half else 1)
    sc, m = scales_pair(sis, scales)
    return np.float32(np.float32(d) * np.float32(sc) * np.float32(quant)
                      - np.float32(dmin) * np.float32(m))


def to_bf16_bits(values):
    """Round float32 to bf16 (round to nearest even) and return the 16-bit patterns."""
    u = values.astype(np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + bias) >> np.uint32(16)).astype(np.uint16)


def main():
    buf = np.zeros(BLOCKS * BYTES, dtype=np.uint8)
    scales = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
    qh_ramp = ((np.arange(32) * 7 + 3) & 0xFF).astype(np.uint8)
    qs_ramp = ((np.arange(128) * 11 + 5) & 0xFF).astype(np.uint8)
    for b in range(BLOCKS):
        off = b * BYTES
        buf[off:off + 2] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
        buf[off + 2:off + 4] = np.frombuffer(np.float16(0.0).tobytes(), dtype=np.uint8)
        buf[off + 4:off + 16] = scales
        buf[off + 16:off + 48] = qh_ramp
        buf[off + 48:off + 176] = qs_ramp
    expected = np.zeros(BLOCKS * ELEMS, dtype=np.float32)
    for b in range(BLOCKS):
        block = buf[b * BYTES:(b + 1) * BYTES]
        for i in range(ELEMS):
            expected[b * ELEMS + i] = value_fp32(block, i)
    bits = to_bf16_bits(expected).astype(np.int16)
    np.save(os.path.join(OUT, 'input_q5k.npy'), buf.view(np.int8))
    np.save(os.path.join(OUT, 'expected_bits.npy'), bits)
    print('wrote input_q5k.npy shape=%s expected_bits shape=%s'
          % (buf.shape, bits.shape))
    print('  expected[:8]=%s' % expected[:8])
    print('  quant range=[%d, %d]  bits[:8]=%s'
          % (int(expected.min()), int(expected.max()), bits[:8]))


if __name__ == '__main__':
    main()
