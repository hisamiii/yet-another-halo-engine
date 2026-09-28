#!/usr/bin/env python3
"""Generate the vision epilogue expectations for the Finish and BiasResidual ports.

Reference for Finish and BiasResidual in vision/encoder.hip. Both end in Round(),
which is float(Bf16(x)) -- a round trip through bf16 -- and both index the bias by
i % width:

  Finish:       out[i] = bf16(out[i] + bias[i % width])
  BiasResidual: hid[i] = bf16(hid[i] + bf16(proj[i] + bias[i % width]))

width is smaller than size on purpose so the modulo is observable; with width ==
size the wraparound never happens and any index expression would do. All values are
small integers, which bf16 holds exactly, so the comparison is exact and the
expected array is the only fixture needed.
"""
import os
import numpy as np

SIZE = 8
WIDTH = 4
OUT = os.path.dirname(os.path.abspath(__file__))


def to_bf16(value):
    u = np.float32(value).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + bias) >> np.uint32(16)).astype(np.uint32) << np.uint32(16)


def round_bf16(values):
    return np.array([np.frombuffer(np.uint32(to_bf16(v)).tobytes(), dtype=np.float32)[0]
                     for v in values], dtype=np.float32)


def main():
    idx = np.arange(SIZE, dtype=np.float32)
    bias = np.arange(WIDTH, dtype=np.float32)
    finish = round_bf16([np.float32(idx[i]) + np.float32(bias[i % WIDTH]) for i in range(SIZE)])
    inner = round_bf16([np.float32(idx[i]) + np.float32(bias[i % WIDTH]) for i in range(SIZE)])
    residual = round_bf16([np.float32(idx[i]) + np.float32(inner[i]) for i in range(SIZE)])
    np.save(os.path.join(OUT, 'expected_finish.npy'), finish)
    np.save(os.path.join(OUT, 'expected_bias_residual.npy'), residual)
    print('wrote expected_finish        =%s' % finish)
    print('wrote expected_bias_residual =%s' % residual)


if __name__ == '__main__':
    main()
