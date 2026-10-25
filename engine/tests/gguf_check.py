#!/usr/bin/env python3
"""The GGUF conversion's CPU pieces (core/gguf_quant.cpp, through build/gguf_rows) against ggml's Python reference
(the gguf package, gguf.quants), and the MQ4 rotation against gen_rot's basis.

usage: gguf_check.py [model.gguf ...]   (needs numpy and gguf: pip install gguf)

  per type: random blocks of every type the loader reads, dequantized by both, must match to f32 rounding
  mq4: random rows with outliers, quantized to MQ4G256 and decoded back to the original basis (s1 H256(s2 / 16 w)),
       must land within 4-bit error (a wrong basis gives a relative error near 1.4)
  each model given: the first rows of one tensor per type in the file, read through the GGUF reader of both
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
from gguf import GGMLQuantizationType as Q, GGUFReader
from gguf.quants import dequantize

BUILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")
TOOL = os.path.join(BUILD, "gguf_rows")
# block bytes, elements, the f16 fields' byte offsets (set to sane scales; the rest of each block is random)
TYPES = {
    Q.Q4_0: (18, 32, [0]), Q.Q4_1: (20, 32, [0, 2]), Q.Q5_0: (22, 32, [0]), Q.Q5_1: (24, 32, [0, 2]),
    Q.Q8_0: (34, 32, [0]), Q.Q2_K: (84, 256, [80, 82]), Q.Q3_K: (110, 256, [108]), Q.Q4_K: (144, 256, [0, 2]),
    Q.Q5_K: (176, 256, [0, 2]), Q.Q6_K: (210, 256, [208]), Q.IQ2_XXS: (66, 256, [0]), Q.IQ2_XS: (74, 256, [0]),
    Q.IQ3_XXS: (98, 256, [0]), Q.IQ3_S: (110, 256, [0]), Q.IQ4_NL: (18, 32, [0]), Q.IQ4_XS: (136, 256, [0]),
}


def run(*args):
    subprocess.run([TOOL, *map(str, args)], check=True)


def ours(tmp, qtype, raw):
    src, dst = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.f32")
    open(src, "wb").write(raw)
    run("dequant", int(qtype), src, dst)
    return np.fromfile(dst, np.float32)


def close(a, b):
    return np.allclose(a, b, rtol=1e-5, atol=1e-7 * max(1.0, float(np.abs(b).max())))


def check_types(tmp, rng):
    bad = 0
    for qtype, (bb, be, halves) in TYPES.items():
        blocks = rng.integers(0, 256, (64, bb), dtype=np.uint8)
        for h in halves:
            blocks[:, h:h + 2] = rng.uniform(0.001, 0.05, 64).astype(np.float16).view(np.uint8).reshape(64, 2)
        ref = dequantize(blocks.reshape(-1), qtype).reshape(-1).astype(np.float32)
        got = ours(tmp, qtype, blocks.tobytes())
        ok = got.shape == ref.shape and close(got, ref)
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {qtype.name:8s} {64 * be} values")
    for qtype, dt in ((Q.F16, np.float16), (Q.BF16, None)):
        x = rng.standard_normal(512).astype(np.float32)
        raw = x.astype(np.float16).tobytes() if dt else (x.view(np.uint32) >> 16).astype(np.uint16).tobytes()
        ref = dequantize(np.frombuffer(raw, np.uint8), qtype).reshape(-1).astype(np.float32)
        ok = close(ours(tmp, qtype, raw), ref)
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {qtype.name:8s} 512 values")
    return bad


def sign_vec(seed):
    s, out = seed, []
    for _ in range(256):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(1.0 if (s >> 16) & 1 else -1.0)
    return np.array(out)


def h256(x):
    y = x.reshape(-1, 256).astype(np.float64)
    h = 1
    while h < 256:
        y = y.reshape(-1, 256 // (2 * h), 2, h)
        y = np.stack([y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]], 2)
        h *= 2
    return y.reshape(x.shape)


def check_mq4(tmp, rng):
    k, rows = 1024, 64
    w = rng.standard_normal((rows, k)).astype(np.float32) * 0.02
    w[rng.integers(0, rows, 40), rng.integers(0, k, 40)] *= 25   # outliers: what the rotation spreads
    src, dst = os.path.join(tmp, "rows.f32"), os.path.join(tmp, "rows.mq4")
    w.tofile(src)
    run("mq4", src, k, dst)
    b = np.fromfile(dst, np.uint8).reshape(rows, k // 256, 136)
    d = b[..., 0:8].copy().view(np.float16).astype(np.float64)   # d0, -8 d0, d1, -8 d1
    q = b[..., 8:].astype(np.int64)
    code = np.stack([q & 15, q >> 4], -1).reshape(rows, k // 256, 256)
    code = np.where(code >= 8, code - 16, code)
    wr = (np.repeat(d[..., [0, 2]], 128, -1) * code).reshape(rows, k)
    s1, s2 = np.tile(sign_vec(42), k // 256), np.tile(sign_vec(1042), k // 256)
    back = s1 * h256(s2 / 16 * wr)
    rel = np.sqrt(((back - w) ** 2).sum() / (w.astype(np.float64) ** 2).sum())
    header = np.allclose(d[..., 1], -8 * d[..., 0], rtol=1e-3) and np.allclose(d[..., 3], -8 * d[..., 2], rtol=1e-3)
    ok = rel < 0.12 and header
    print(f"{'ok  ' if ok else 'FAIL'} mq4      relative rms error {rel:.4f} (4-bit level ~0.09), header {header}")
    return not ok


def check_model(tmp, path):
    bad, seen = 0, set()
    for t in GGUFReader(path).tensors:
        qtype = Q(t.tensor_type)
        if qtype in seen or len(t.shape) != 2 or (qtype not in TYPES and qtype not in (Q.F16, Q.BF16, Q.F32)):
            continue
        seen.add(qtype)
        k = int(t.shape[0])
        ref = dequantize(np.asarray(t.data), qtype).reshape(-1, k)[:4].reshape(-1).astype(np.float32)
        out = os.path.join(tmp, "t.f32")
        run("tensor", path, t.name, 4, out)
        ok = close(np.fromfile(out, np.float32), ref)
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {qtype.name:8s} {t.name} (first 4 rows of {os.path.basename(path)})")
    return bad


def main():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        bad = check_types(tmp, rng) + check_mq4(tmp, rng)
        for path in sys.argv[1:]:
            bad += check_model(tmp, path)
    print("gguf_check: ok" if not bad else f"gguf_check: {bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
