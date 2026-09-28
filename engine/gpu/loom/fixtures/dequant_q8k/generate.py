#!/usr/bin/env python3
"""Generate the Q8_K block fixture for yah_dequant_q8k_bf16.loom.

Reference for DequantizeQ8KToBf16Kernel in prefill_gemm.hip. A block_q8_K is
292 bytes: a float scale, 256 int8 values, then 16 int16 block sums. The kernel
only reads the scale and the codes:

  out[b*256 + j] = bf16( d * float(qs[b][j]) )

Each block gets scale 0.5 and the ramp -128..127 for its codes. The product is
then a multiple of 0.5 in [-64, 63.5], and bf16 represents every such value
exactly below 128, so no tolerance is needed. Because every block is identical and
the expected output is the same 256-value ramp repeated, the Loom side can state
the expectation as a periodic iota and only the packed input needs a fixture.
"""
import os
import numpy as np

BLOCK_ELEMS = 256
BLOCK_BYTES = 292
BLOCKS = 2
SCALE = np.float32(0.5)
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    buf = np.zeros(BLOCKS * BLOCK_BYTES, dtype=np.uint8)
    ramp = np.arange(-128, 128, dtype=np.int8)
    for b in range(BLOCKS):
        off = b * BLOCK_BYTES
        buf[off:off + 4] = np.frombuffer(SCALE.tobytes(), dtype=np.uint8)
        buf[off + 4:off + 4 + BLOCK_ELEMS] = ramp.view(np.uint8)
        # bsums[16] stays zero; the kernel does not read it.
    np.save(os.path.join(OUT, 'input_q8k.npy'), buf.view(np.int8))
    print('wrote input_q8k.npy shape=%s bytes per block=%d'
          % (buf.shape, BLOCK_BYTES))
    print('  block 0 head: d=%s qs[:4]=%s'
          % (SCALE, ramp[:4]))


if __name__ == '__main__':
    main()
