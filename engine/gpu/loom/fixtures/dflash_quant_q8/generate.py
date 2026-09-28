#!/usr/bin/env python3
"""Reference for the Loom DFlash bf16 -> Q8_0 quantizer (fixtures/dflash_quant_q8)."""
import numpy as np


def bf16(x):
    a = np.asarray(x, dtype=np.float32)
    bits = a.view(np.uint32).astype(np.uint64)
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def main():
    src = bf16(np.arange(64, dtype=np.float32))
    out = np.zeros(68, dtype=np.uint8)
    for b in range(2):
        vals = src[b * 32:(b + 1) * 32]
        amax = float(np.max(np.abs(vals)))
        scale = np.float32(amax / 127.0)
        inv = np.float32(1.0 / float(scale)) if scale > 0 else np.float32(0.0)
        off = b * 34
        out[off:off + 2] = np.frombuffer(np.float16(float(scale)).tobytes(), dtype=np.uint8)
        for i in range(32):
            q = np.floor(float(np.float32(np.float32(vals[i]) * inv)) + 0.5)
            if q < -127.0:
                q = -127.0
            if q > 127.0:
                q = 127.0
            out[off + 2 + i] = np.int8(q).astype(np.uint8)
    np.save("expected_bytes.npy", out.view(np.int8))
    print(out[:36])


if __name__ == "__main__":
    main()
