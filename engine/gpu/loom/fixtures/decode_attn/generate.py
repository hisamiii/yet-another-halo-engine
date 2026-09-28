#!/usr/bin/env python3
"""Generate the decode online attention expectation for yah_decode_attn_f16.loom.

Reference for QwenDecodeOnlineAttentionHalfKernel in attention_decode.hip. One
workgroup per (query head), one row, 32 lanes; each lane owns eight dimensions
(float4 indices lane and lane+32).

  kv_head = head // group_size
  seq_len = start_pos + row + 1
  scale   = 1 / sqrt(256) = 1/16
  score[p] = dot(q[head], k[p]) * scale        p < seq_len
  s        = softmax(score[0..seq_len))
  ctx[d]   = sum_p s[p] * v[p][d]
  out[d]   = ctx[d] * sigmoid(gate[head][d])

The cache is fp16 in the engine layout [layer][position][kv_head][dim]. Every q,
k, v and gate value is a multiple of 1/4 or 1/2, so the fp16 cache is exact; the
softmax exponential is the only inexact step.
"""
import os
import numpy as np

HEADS = 2
KV_HEADS = 1
HEAD_DIM = 256
MAX_CTX = 4
START = 2
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    q = np.zeros((HEADS, HEAD_DIM), dtype=np.float32)
    k = np.zeros((MAX_CTX, HEAD_DIM), dtype=np.float32)
    v = np.zeros((MAX_CTX, HEAD_DIM), dtype=np.float32)
    gate = np.zeros((HEADS, HEAD_DIM), dtype=np.float32)
    for h in range(HEADS):
        for d in range(HEAD_DIM):
            q[h, d] = np.float32(((d % 13) - 6) * 0.25)
            gate[h, d] = np.float32(((d % 9) - 4) * 0.5)
    for p in range(MAX_CTX):
        for d in range(HEAD_DIM):
            k[p, d] = np.float32(((d % 7) - 3) * 0.5)
            v[p, d] = np.float32(((d % 5) - 2) * 0.25)

    expected = np.zeros((HEADS, HEAD_DIM), dtype=np.float32)
    nogate = np.zeros((HEADS, HEAD_DIM), dtype=np.float32)
    scale = 1.0 / 16.0
    seq_len = START + 1
    for h in range(HEADS):
        scores = np.zeros(seq_len, dtype=np.float64)
        for p in range(seq_len):
            acc = np.float64(0.0)
            for d in range(HEAD_DIM):
                acc += np.float64(q[h, d]) * np.float64(k[p, d])
            scores[p] = acc * scale
        mx = np.max(scores)
        ex = np.exp(scores - mx)
        total = np.sum(ex)
        for d in range(HEAD_DIM):
            ctx = np.float64(0.0)
            for p in range(seq_len):
                ctx += (ex[p] / total) * np.float64(v[p, d])
            sig = np.float64(1.0) / (1.0 + np.exp(-np.float64(gate[h, d])))
            expected[h, d] = np.float32(ctx * sig)
            nogate[h, d] = np.float32(ctx)

    np.save(os.path.join(OUT, "input_q.npy"), q.reshape(-1).astype(np.float32))
    np.save(os.path.join(OUT, "input_k.npy"), k.reshape(-1).astype(np.float16))
    np.save(os.path.join(OUT, "input_v.npy"), v.reshape(-1).astype(np.float16))
    np.save(os.path.join(OUT, "input_gate.npy"), gate.reshape(-1).astype(np.float32))
    np.save(os.path.join(OUT, "expected.npy"), expected.reshape(-1).astype(np.float32))
    np.save(os.path.join(OUT, "expected_nogate.npy"), nogate.reshape(-1).astype(np.float32))
    print("seq_len", seq_len, "expected[0][:4]", expected[0][:4])


if __name__ == "__main__":
    main()