#!/usr/bin/env python3
"""The GGUF loader's CPU pieces (core/gguf_quant.cpp, through build/gguf_rows) against ggml's Python reference
(the gguf package, gguf.quants).

usage: gguf_check.py [model.gguf ...]   (needs numpy and gguf: pip install gguf)

  per type: random blocks of every type the loader reads, dequantized by both, must match to f32 rounding
  q8: random values with outliers to Q8_0 (token_embd's load), decoded by ggml's reference, within half a step
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


def check_q8(tmp, rng):
    x = rng.standard_normal(32 * 256).astype(np.float32) * 0.02
    x[rng.integers(0, x.size, 40)] *= 25
    src, dst = os.path.join(tmp, "x.f32"), os.path.join(tmp, "x.q8")
    x.tofile(src)
    run("q8", src, dst)
    raw = np.fromfile(dst, np.uint8)
    back = dequantize(raw, Q.Q8_0).reshape(-1)
    step = np.repeat(np.abs(x.reshape(-1, 32)).max(1) / 127, 32)
    # (the f16 scale moves a value by up to 127 x its rounding: 2^-11 relative)
    ok = raw.size == x.size // 32 * 34 and bool(np.all(np.abs(back - x) <= step * (0.5 + 1e-3) + np.abs(x) * 2 ** -10))
    print(f"{'ok  ' if ok else 'FAIL'} q8       {x.size} values, max error {np.abs(back - x).max():.3g}")
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
        bad = check_types(tmp, rng) + check_q8(tmp, rng)
        for path in sys.argv[1:]:
            bad += check_model(tmp, path)
    print("gguf_check: ok" if not bad else f"gguf_check: {bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
