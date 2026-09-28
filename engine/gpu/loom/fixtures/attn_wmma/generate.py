#!/usr/bin/env python3
"""Generate the WMMA causal attention expectation for yah_attn_wmma_f32.loom.

Reference for WmmaCausalAttention in attention_wmma.hip. One query head and one
token per workgroup:

  kv_head   = head / gqa
  abs_query = start_pos + token
  seq_len   = abs_query + 1                 causal, keys up to and including self
  scale     = 1 / 16                        the kernel hardcodes 1/16
  scores[p] = dot(q, k[p]) * scale
  s         = softmax(scores[0..seq_len))
  ctx[d]    = sum_p s[p] * v[p][d]
  lse       = max + log(sum)                always written, used when lse is armed
  out[d]    = ctx[d] * sigmoid(gate[d])     only when the gate is armed

The kernel reads the fp16 KV cache. Two layouts are emitted:
  canonical  [position][head][dim]
  head-major [head][padded][dim]

Every q, k, v and gate value is a multiple of 1/4 or 1/2, so the fp16 cache is
exact. The only inexact step is the softmax exponential, which is why the
comparison carries a tolerance.
"""
import os
import numpy as np

NUM_HEADS = 2
NUM_KV_HEADS = 1
GQA = 2
HEAD_DIM = 4
BATCH = 3
START = 0
MAX_CTX = 16
PADDED = 16
SCALE = np.float64(1.0) / 16.0
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    q = np.zeros((BATCH, NUM_HEADS, HEAD_DIM), dtype=np.float32)
    gate = np.zeros((BATCH, NUM_HEADS, HEAD_DIM), dtype=np.float32)
    k = np.zeros((MAX_CTX, HEAD_DIM), dtype=np.float32)
    v = np.zeros((MAX_CTX, HEAD_DIM), dtype=np.float32)
    for t in range(BATCH):
        for h in range(NUM_HEADS):
            for d in range(HEAD_DIM):
                idx = (t * NUM_HEADS + h) * HEAD_DIM + d
                q[t, h, d] = np.float32(((idx % 9) - 4) * 0.25)
                gate[t, h, d] = np.float32(((idx % 5) - 2) * 0.5)
    for p in range(MAX_CTX):
        for d in range(HEAD_DIM):
            idx = p * HEAD_DIM + d
            k[p, d] = np.float32(((idx % 7) - 3) * 0.5)
            v[p, d] = np.float32(((idx % 5) + 1) * 0.25)

    out = np.zeros((BATCH, NUM_HEADS, HEAD_DIM), dtype=np.float32)
    out_nolse = np.zeros((BATCH, NUM_HEADS, HEAD_DIM), dtype=np.float32)
    lse = np.zeros((NUM_HEADS, BATCH), dtype=np.float32)
    for t in range(BATCH):
        seq_len = START + t + 1
        for h in range(NUM_HEADS):
            scores = np.zeros(seq_len, dtype=np.float64)
            for p in range(seq_len):
                acc = np.float64(0.0)
                for d in range(HEAD_DIM):
                    acc += np.float64(q[t, h, d]) * np.float64(k[p, d])
                scores[p] = acc * SCALE
            mx = np.max(scores)
            ex = np.exp(scores - mx)
            total = np.sum(ex)
            lse[h, t] = np.float32(mx + np.log(total))
            for d in range(HEAD_DIM):
                ctx = np.float64(0.0)
                for p in range(seq_len):
                    ctx += (ex[p] / total) * np.float64(v[p, d])
                out_nolse[t, h, d] = np.float32(ctx)
                sig = np.float64(1.0) / (1.0 + np.exp(-np.float64(gate[t, h, d])))
                out[t, h, d] = np.float32(ctx * sig)

    k_can = np.zeros((MAX_CTX, NUM_KV_HEADS, HEAD_DIM), dtype=np.float16)
    v_can = np.zeros((MAX_CTX, NUM_KV_HEADS, HEAD_DIM), dtype=np.float16)
    k_hm = np.zeros((NUM_KV_HEADS, PADDED, HEAD_DIM), dtype=np.float16)
    v_hm = np.zeros((NUM_KV_HEADS, PADDED, HEAD_DIM), dtype=np.float16)
    for kv_h in range(NUM_KV_HEADS):
        for p in range(MAX_CTX):
            k_can[p, kv_h] = k[p].astype(np.float16)
            v_can[p, kv_h] = v[p].astype(np.float16)
        for p in range(PADDED):
            if p < MAX_CTX:
                k_hm[kv_h, p] = k[p].astype(np.float16)
                v_hm[kv_h, p] = v[p].astype(np.float16)

    files = [
        ("input_q.npy", q.reshape(-1)),
        ("input_gate.npy", gate.reshape(-1)),
        ("input_k_f16.npy", k_can.reshape(-1)),
        ("input_v_f16.npy", v_can.reshape(-1)),
        ("input_k_hm_f16.npy", k_hm.reshape(-1)),
        ("input_v_hm_f16.npy", v_hm.reshape(-1)),
        ("expected_out.npy", out.reshape(-1)),
        ("expected_out_nolse.npy", out_nolse.reshape(-1)),
        ("expected_lse.npy", lse.reshape(-1)),
    ]
    for name, array in files:
        np.save(os.path.join(OUT, name), array)
        print("wrote %-24s shape=%s" % (name, array.shape))


if __name__ == "__main__":
    main()