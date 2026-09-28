#!/usr/bin/env python3
"""Fixtures for yah_ssm_proj_decode_f32.loom.

The SSM dispatch kernel keeps ONE i8 and ONE f16 view per band buffer, sized at
the band's maximum supported row stride (4200 B/row for the qkv and gate bands,
because Q6_K is the widest decoder there; the Q8_0 alpha/beta stride is
hidden/32*34 = 5440 B/row at hidden 5120).  The single-format gemv fixtures are
therefore zero-padded up to that maximum so the bound tensor is at least as
large as the declared view; the decoder reads each row at its own format stride,
which is always inside the real data prefix, so padding is never touched.

Writes:
  qkv_q4k_pad.npy   q4k   16x5120 -> 67200 B (from fixtures/q4k_gemm)
  qkv_iq4nl_pad.npy iq4nl 16x5120 -> 67200 B (from fixtures/iq4nl_gemm)
  qkv_iq4xs_pad.npy iq4xs 16x5120 -> 67200 B (from fixtures/iq4xs_gemm)
  gate_q4k_pad.npy  q4k   16x5120 -> 67200 B
  gate_iq4xs_pad.npy iq4xs 16x5120 -> 67200 B
  gate_q5k_pad.npy  q5k   16x5120 -> 67200 B
  gate_q6k_pad.npy  q6k   16x5120 -> 67200 B (already 67200)
  q8_0_alpha.npy    Q8_0  rank=2 x 5120 -> 10880 B
  q8_0_beta.npy     Q8_0  rank=2 x 5120 -> 10880 B
  q8_0_expected.npy float32[4] = [alpha0, alpha1, beta0, beta1]

The Q8_0 activation is fixtures/q4k_gemm/gemv_x.npy (same sparse x as every
other gemv fixture), so the expected dots are exact sums of a handful of
decoded weights.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.dirname(HERE)
ROWS = 16
K = 5120
QKV_PAD = 4200 * ROWS          # 67200
RANK = 2
Q8_STRIDE = (K // 32) * 34     # 5440


def pad(src, dst, total):
    data = np.load(src).reshape(-1).astype(np.int8)
    assert data.size <= total, (src, data.size, total)
    out = np.zeros(total, dtype=np.int8)
    out[:data.size] = data
    np.save(os.path.join(HERE, dst), out)
    print("wrote", dst, out.shape, "from", data.size, "B")


def decode_q8_0_row(row_bytes):
    """block_q8_0 = {half d; int8 qs[32]}; 34 B, QK=32."""
    vals = []
    for b in range((K // 32)):
        blk = row_bytes[b * 34:(b + 1) * 34]
        d = np.frombuffer(blk[0:2], dtype=np.float16)[0].astype(np.float64)
        qs = np.frombuffer(blk[2:34], dtype=np.int8)
        vals.extend([d * np.float64(int(q)) for q in qs])
    return vals


def make_q8_0(seed, dst):
    packed = bytearray()
    rows = []
    for r in range(RANK):
        row = bytearray()
        for b in range(K // 32):
            d = np.float16(0.5 * (r + 1) + 0.25 * (b % 3)).tobytes()
            qs = bytes((((seed + r * 11 + b * 7 + j * 3) % 41) - 20) & 0xFF
                       for j in range(32))
            assert len(d) + len(qs) == 34
            row += d + qs
        assert len(row) == Q8_STRIDE, len(row)
        packed += row
        rows.append(decode_q8_0_row(row))
    np.save(os.path.join(HERE, dst), np.frombuffer(bytes(packed), dtype=np.int8))
    print("wrote", dst, len(packed), "B")
    return rows


def main():
    pad(os.path.join(FIX, "q4k_gemm", "input_q4k_gemm.npy"),
        "qkv_q4k_pad.npy", QKV_PAD)
    pad(os.path.join(FIX, "iq4nl_gemm", "input_iq4nl_gemm.npy"),
        "qkv_iq4nl_pad.npy", QKV_PAD)
    pad(os.path.join(FIX, "iq4xs_gemm", "input_iq4xs_gemm.npy"),
        "qkv_iq4xs_pad.npy", QKV_PAD)
    pad(os.path.join(FIX, "q4k_gemm", "input_q4k_gemm.npy"),
        "gate_q4k_pad.npy", QKV_PAD)
    pad(os.path.join(FIX, "iq4xs_gemm", "input_iq4xs_gemm.npy"),
        "gate_iq4xs_pad.npy", QKV_PAD)
    pad(os.path.join(FIX, "q5k_gemm", "input_q5k_gemm.npy"),
        "gate_q5k_pad.npy", QKV_PAD)
    pad(os.path.join(FIX, "q6k_gemm", "input_q6k_gemm.npy"),
        "gate_q6k_pad.npy", QKV_PAD)

    alpha = make_q8_0(0, "q8_0_alpha.npy")
    beta = make_q8_0(97, "q8_0_beta.npy")
    x = np.load(os.path.join(FIX, "q4k_gemm", "gemv_x.npy")).astype(np.float64)
    exp = []
    for rows in (alpha, beta):
        for r in range(RANK):
            acc = np.float64(0.0)
            for k in range(K):
                if x[k] != 0.0:
                    acc += rows[r][k] * x[k]
            exp.append(np.float32(acc))
    np.save(os.path.join(HERE, "q8_0_expected.npy"), np.array(exp, dtype=np.float32))
    np.save(os.path.join(HERE, "q8_0_alpha_expected.npy"),
            np.array(exp[0:RANK], dtype=np.float32))
    np.save(os.path.join(HERE, "q8_0_beta_expected.npy"),
            np.array(exp[RANK:2 * RANK], dtype=np.float32))
    print("q8_0_expected", exp)


if __name__ == "__main__":
    main()
