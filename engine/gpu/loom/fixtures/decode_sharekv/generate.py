#!/usr/bin/env python3
"""Fixtures for yah_decode_splitk_sharekv_f16.loom.

Reference for QwenDecodeSplitKAttentionHalfPartialsKernel<true, 32> in
attention_decode.hip. Same split-K partials as the plain arm; the only
difference is the launch shape (6 query heads share one KV head per block).
q and gate are made head-dependent so a wrong head mapping fails.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HEADS = 6
KV_HEADS = 1
HEAD_DIM = 256
MAX_CTX = 8
START = 4
SPLITS = 2
MAX_SPLITS = 2


def oracle(q, k16, v16, heads, kv_heads, head_dim, max_ctx, start, splits, max_splits):
    seq_len = start + 1
    plen = (seq_len + splits - 1) // splits
    stats_elems2 = heads * max_splits * 2
    row_stride2 = heads * max_splits * (head_dim + 2)
    scratch = np.zeros(row_stride2, dtype=np.float32)
    kd = k16.astype(np.float64)
    vd = v16.astype(np.float64)
    scale = 1.0 / 16.0
    for h in range(heads):
        for s in range(splits):
            begin = s * plen
            end = min(begin + plen, seq_len)
            scores = []
            for p in range(begin, end):
                acc = np.float64(0.0)
                for d in range(head_dim):
                    acc += np.float64(q[h, d]) * kd[p, d]
                scores.append(acc * scale)
            mx = max(scores)
            ss = 0.0
            ctx = np.zeros(head_dim, dtype=np.float64)
            for idx, p in enumerate(range(begin, end)):
                w = np.exp(scores[idx] - mx)
                ss += w
                ctx += w * vd[p]
            pi = h * max_splits + s
            scratch[pi * 2 + 0] = np.float32(mx)
            scratch[pi * 2 + 1] = np.float32(ss)
            base = stats_elems2 + pi * head_dim
            scratch[base:base + head_dim] = ctx.astype(np.float32)
    return scratch


def main():
    q = np.zeros((HEADS, HEAD_DIM), dtype=np.float32)
    gate = np.zeros((HEADS, HEAD_DIM), dtype=np.float32)
    for h in range(HEADS):
        for d in range(HEAD_DIM):
            q[h, d] = np.float32(((d % 13) - 6) * 0.25 + h * 0.125)
            gate[h, d] = np.float32(((d % 9) - 4) * 0.5 - h * 0.25)
    k = np.zeros((MAX_CTX, HEAD_DIM), dtype=np.float32)
    v = np.zeros((MAX_CTX, HEAD_DIM), dtype=np.float32)
    for p in range(MAX_CTX):
        for d in range(HEAD_DIM):
            k[p, d] = np.float32(((d % 7) - 3) * 0.5 + p * 0.0625)
            v[p, d] = np.float32(((d % 5) - 2) * 0.25 - p * 0.03125)
    k16 = k.astype(np.float16)
    v16 = v.astype(np.float16)
    scratch = oracle(q, k16, v16, HEADS, KV_HEADS, HEAD_DIM, MAX_CTX, START, SPLITS, MAX_SPLITS)
    np.save(os.path.join(HERE, "input_q.npy"), q.reshape(-1).astype(np.float32))
    np.save(os.path.join(HERE, "input_k.npy"), k16.reshape(-1))
    np.save(os.path.join(HERE, "input_v.npy"), v16.reshape(-1))
    np.save(os.path.join(HERE, "scratch.npy"), scratch.reshape(-1).astype(np.float32))
    print("small scratch", scratch.shape, scratch[:4])
    # Production shape: 24 heads = 6 * 4 KV heads, context 1024, 8 splits.
    ph, pkv, pctx, pstart, psplits, pms = 24, 4, 1024, 1023, 8, 8
    pq = np.zeros((ph, HEAD_DIM), dtype=np.float32)
    pk = np.zeros((pctx, HEAD_DIM), dtype=np.float16)
    pv = np.zeros((pctx, HEAD_DIM), dtype=np.float16)
    pscratch = oracle(pq, pk, pv, ph, pkv, HEAD_DIM, pctx, pstart+1-1, psplits, pms)
    np.save(os.path.join(HERE, "full_scratch.npy"), pscratch.reshape(-1).astype(np.float32))
    print("full scratch", pscratch.shape, pscratch[:4], pscratch[2:4])


if __name__ == "__main__":
    main()