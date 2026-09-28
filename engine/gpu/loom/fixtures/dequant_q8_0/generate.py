#!/usr/bin/env python3
"""Generate the Q8_0 block fixture for yah_dequant_q8_0_bf16.loom.

Reference for DequantizeQ8_0ToBf16Kernel in prefill_gemm.hip. A block_q8_0 is
34 bytes: an fp16 scale then 32 int8 values. The kernel computes

  out[b*32 + j] = bf16( float(d) * float(qs[b][j]) )

Each block gets scale 0.5 and the ramp -16..15, so every product is a multiple of
0.5 in [-8, 7.5] and bf16 holds it exactly. Every block is identical, so the Loom
side states the expectation as a periodic iota and only the packed input needs a
fixture.
"""
import os
import numpy as np

BLOCK_ELEMS = 32
BLOCK_BYTES = 34
BLOCKS = 2
SCALE = np.float16(0.5)
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    buf = np.zeros(BLOCKS * BLOCK_BYTES, dtype=np.uint8)
    ramp = np.arange(-16, 16, dtype=np.int8)
    for b in range(BLOCKS):
        off = b * BLOCK_BYTES
        buf[off:off + 2] = np.frombuffer(SCALE.tobytes(), dtype=np.uint8)
        buf[off + 2:off + 2 + BLOCK_ELEMS] = ramp.view(np.uint8)
    np.save(os.path.join(OUT, 'input_q8_0.npy'), buf.view(np.int8))
    print('wrote input_q8_0.npy shape=%s scale=%s ramp=%s'
          % (buf.shape, SCALE, (ramp[:3], ramp[-3:])))


if __name__ == '__main__':
    main()
