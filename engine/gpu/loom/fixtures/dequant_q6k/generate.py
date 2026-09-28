#!/usr/bin/env python3
"""Generate the Q6_K block fixture for yah_dequant_q6k_bf16.loom.

Reference for DequantizeQ6KToBf16Kernel / Q6KValueFp in prefill_gemm.hip and
quant_ops.hpp. A block_q6_K is 210 bytes:

  uint8 ql[128]; uint8 qh[64]; int8 scales[16]; __half d

and one element is

  half = i/128; within = i%128; segment = within/32; lane = within%32
  qh_byte = qh[half*32 + lane]
  low  = segment 0/2 ? ql[half*64 + lane] & 0xF or >> 4
         segment 1/3 : ql[half*64 + 32 + lane] & 0xF or >> 4
  high = (qh_byte >> (2*segment)) & 3
  scale_index = half*8 + lane/16 + segment*2
  quant = ((high << 4) | low) - 32
  value = f32(d) * f32(scales[scale_index]) * f32(quant)

The scales cycle through 1, 2, 4 and 8 so the scale index is observable while the
product stays inside the range bf16 holds exactly: quant is in [-32, 31] and the
largest scale is 8, so every value is an integer of magnitude at most 248. As with
Q5_K the expectation is written as raw 16-bit patterns in an int16 array.
"""
import os
import numpy as np

BLOCKS = 2
ELEMS = 256
BYTES = 210
OUT = os.path.dirname(os.path.abspath(__file__))


def value_fp32(block, index):
    d = np.frombuffer(block[208:210].tobytes(), dtype=np.float16)[0].astype(np.float32)
    ql = block[0:128]
    qh = block[128:192]
    scales = block[192:208].view(np.int8)
    half = index // 128
    within = index % 128
    segment = within // 32
    lane = within % 32
    ql_base = half * 64
    qh_byte = int(qh[half * 32 + lane])
    if segment == 0:
        low = int(ql[ql_base + lane]) & 0x0F
        high = (qh_byte >> 0) & 3
    elif segment == 1:
        low = int(ql[ql_base + 32 + lane]) & 0x0F
        high = (qh_byte >> 2) & 3
    elif segment == 2:
        low = int(ql[ql_base + lane]) >> 4
        high = (qh_byte >> 4) & 3
    else:
        low = int(ql[ql_base + 32 + lane]) >> 4
        high = (qh_byte >> 6) & 3
    scale_index = half * 8 + (lane // 16) + segment * 2
    quant = ((high << 4) | low) - 32
    return np.float32(np.float32(d) * np.float32(int(scales[scale_index]))
                      * np.float32(quant))


def to_bf16_bits(values):
    u = values.astype(np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + bias) >> np.uint32(16)).astype(np.uint16)


def main():
    buf = np.zeros(BLOCKS * BYTES, dtype=np.uint8)
    scales = np.array([1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8], dtype=np.int8)
    ql_ramp = ((np.arange(128) * 13 + 7) & 0xFF).astype(np.uint8)
    qh_ramp = ((np.arange(64) * 29 + 11) & 0xFF).astype(np.uint8)
    for b in range(BLOCKS):
        off = b * BYTES
        buf[off:off + 128] = ql_ramp
        buf[off + 128:off + 192] = qh_ramp
        buf[off + 192:off + 208] = scales.view(np.uint8)
        buf[off + 208:off + 210] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
    expected = np.zeros(BLOCKS * ELEMS, dtype=np.float32)
    for b in range(BLOCKS):
        block = buf[b * BYTES:(b + 1) * BYTES]
        for i in range(ELEMS):
            expected[b * ELEMS + i] = value_fp32(block, i)
    bits = to_bf16_bits(expected).astype(np.int16)
    np.save(os.path.join(OUT, 'input_q6k.npy'), buf.view(np.int8))
    np.save(os.path.join(OUT, 'expected_bits.npy'), bits)
    print('wrote input_q6k.npy shape=%s expected_bits shape=%s' % (buf.shape, bits.shape))
    print('  expected range=[%d, %d]  sample=%s'
          % (int(expected.min()), int(expected.max()), expected[:6]))


if __name__ == '__main__':
    main()
