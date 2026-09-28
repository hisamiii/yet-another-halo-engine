#!/usr/bin/env python3
"""Generate the bfp16 round-trip fixture for yah_bfp16_roundtrip_f16.loom.

Reference for Bfp16RoundTripFp16Kernel in prefill_fp16.hip. One group of 8
consecutive fp16 activations is snapped to a shared-exponent grid, in place:

  amax     = max_i |f32(x[i])|
  exponent = the frexp exponent of amax, so amax = m * 2^exponent with m in [0.5, 1)
  step     = 2^(exponent - bits)
  x[i]     = f16(rintf(f32(x[i]) / step) * step)

The group is all zeros -> the kernel returns and the data is untouched.

Two groups with different exponents are used (0.9 -> exponent 0, 2.5 -> exponent 2)
and the values are not multiples of the step, so the rounding is observable rather
than an identity. Without that, any step that happened to divide every value would
pass. The values are fp16-exact, so the whole comparison is exact.
"""
import os
import numpy as np

BITS = 3
GROUPS = 2
OUT = os.path.dirname(os.path.abspath(__file__))


def encode(values, bits):
    v = values.astype(np.float32)
    amax = np.float32(np.max(np.abs(v)))
    if amax == 0.0:
        return values
    _, exponent = np.frexp(amax)
    step = np.float32(np.ldexp(np.float32(1.0), int(exponent) - bits))
    snapped = (np.rint((v / step).astype(np.float32)).astype(np.float32) * step)
    return snapped.astype(np.float16)


def main():
    group0 = np.array([0.3, 0.6, 0.9, -0.3, -0.6, -0.9, 0.1, -0.1], dtype=np.float16)
    group1 = np.array([2.5, 1.25, -2.4, -1.25, 0.7, -0.7, 2.0, -2.0], dtype=np.float16)
    data = np.concatenate([group0, group1])
    expected = np.concatenate([encode(group0, BITS), encode(group1, BITS)])
    np.save(os.path.join(OUT, 'input_bfp16.npy'), data)
    np.save(os.path.join(OUT, 'expected_bfp16.npy'), expected)
    print('wrote input_bfp16.npy  %s' % data)
    print('wrote expected_bfp16.npy %s' % expected)
    print('  changed elements: %d of %d'
          % (int(np.count_nonzero(data != expected)), data.size))


if __name__ == '__main__':
    main()
