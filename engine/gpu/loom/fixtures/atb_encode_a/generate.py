#!/usr/bin/env python3
"""Reference for the Loom ATB A-operand encoder (fixtures/atb_encode_a).

k = 64, l1_cols = 1, groups = 1024: one 128x64 fp16 L1 tile. a[i] = fp16(i) for
i in 0..8191, so every group has a distinct exponent and code vector. Mirrors
AtbEncodeAKernel exactly, including the frexpf/ldexpf exponent and the
floor(value/step + 0.5) truncation into int8."""
import math
import numpy as np

K = 64
L1_COLS = 1
GROUPS = 1024
A_ELEMS = 128 * K


def main():
    a = np.arange(A_ELEMS, dtype=np.float32).astype(np.float16)
    np.save("a.npy", a)
    out = np.zeros(GROUPS * 9, dtype=np.uint8)
    for g in range(GROUPS):
        tile = g // 1024
        gt = g % 1024
        l1c = tile % L1_COLS
        l1r = tile // L1_COLS
        rib = gt % 8
        t1 = gt // 8
        bis = t1 % 2
        t2 = t1 // 2
        sbc = t2 % 8
        si = t2 // 8
        r = (2 * si + bis) * 8 + rib
        row = l1r * 128 + r
        src = row * K + l1c * 64 + sbc * 8
        vals = a[src:src + 8].astype(np.float32)
        amax = float(np.max(np.abs(vals)))
        if amax == 0.0:
            continue
        mant, exp = math.frexp(amax)
        out[g * 9] = (exp + 126) & 0xFF
        step = np.float32(math.ldexp(1.0, exp - 7))
        for i in range(8):
            ratio = np.float32(np.float32(vals[i]) / step)
            mag = int(math.floor(float(np.float32(ratio + np.float32(0.5)))))
            out[g * 9 + 1 + i] = mag & 0xFF
    np.save("expected_bytes.npy", out.view(np.int8))
    print(a[:4], out[:18])


if __name__ == "__main__":
    main()
