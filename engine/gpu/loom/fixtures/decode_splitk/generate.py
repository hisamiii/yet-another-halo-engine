#!/usr/bin/env python3
"""Generate the split-K decode attention fixtures for
yah_decode_splitk_partials_f16.loom and yah_decode_splitk_reduce_f32.loom.

Reference for QwenDecodeSplitKAttentionHalfPartialsKernel<false, 32> and
QwenDecodeSplitKAttentionReduceKernel in attention_decode.hip. Partitions:

  partition_length = ceil(seq_len / splits)
  partition        = [split*length, min((split+1)*length, seq_len))
  stats[pi]        = (max over partition, sum of exp(score - max))
  partial[pi][d]   = sum_p exp(score[p] - max) * v[p][d]

Scratch per row: num_heads*max_splits*2 statistics then
num_heads*max_splits*head_dim partials, indexed by pi = head*max_splits + split.
The reduce combines the partitions and applies the gate. Same input values as
fixtures/decode_attn, so the two ports can be compared directly.
"""
import os
import numpy as np

HEADS = 2
KV_HEADS = 1
HEAD_DIM = 256
MAX_CTX = 8
START = 4
SPLITS = 2
MAX_SPLITS = 2
ROWS = 1
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
    k16 = k.astype(np.float16)
    v16 = v.astype(np.float16)
    kd = k16.astype(np.float64)
    vd = v16.astype(np.float64)

    seq_len = START + 1
    plen = (seq_len + SPLITS - 1) // SPLITS
    stats_elems2 = HEADS * MAX_SPLITS * 2
    row_stride2 = HEADS * MAX_SPLITS * (HEAD_DIM + 2)
    scratch = np.zeros((ROWS, row_stride2), dtype=np.float32)
    scale = 1.0 / 16.0
    # Per (head, split) partition statistics and unnormalised context.
    part_max = np.zeros((HEADS, MAX_SPLITS), dtype=np.float64)
    part_sum = np.zeros((HEADS, MAX_SPLITS), dtype=np.float64)
    part_ctx = np.zeros((HEADS, MAX_SPLITS, HEAD_DIM), dtype=np.float64)
    for h in range(HEADS):
        for s in range(SPLITS):
            begin = s * plen
            end = min(begin + plen, seq_len)
            scores = []
            for p in range(begin, end):
                acc = np.float64(0.0)
                for d in range(HEAD_DIM):
                    acc += np.float64(q[h, d]) * kd[p, d]
                scores.append(acc * scale)
            mx = max(scores)
            ss = 0.0
            for idx, p in enumerate(range(begin, end)):
                w = np.exp(scores[idx] - mx)
                ss += w
                part_ctx[h, s] += w * vd[p]
            part_max[h, s] = mx
            part_sum[h, s] = ss
            pi = h * MAX_SPLITS + s
            scratch[0, pi * 2 + 0] = np.float32(mx)
            scratch[0, pi * 2 + 1] = np.float32(ss)
            base = stats_elems2 + pi * HEAD_DIM
            scratch[0, base:base + HEAD_DIM] = part_ctx[h, s].astype(np.float32)

    # Reduce.
    expected = np.zeros((HEADS, HEAD_DIM), dtype=np.float32)
    for h in range(HEADS):
        gmax = max(part_max[h, s] for s in range(SPLITS))
        gsum = 0.0
        ctx = np.zeros(HEAD_DIM, dtype=np.float64)
        for s in range(SPLITS):
            sc = np.exp(part_max[h, s] - gmax)
            gsum += part_sum[h, s] * sc
            ctx += part_ctx[h, s] * sc
        ctx /= gsum
        for d in range(HEAD_DIM):
            sig = np.float64(1.0) / (1.0 + np.exp(-np.float64(gate[h, d])))
            expected[h, d] = np.float32(ctx[d] * sig)

    np.save(os.path.join(OUT, "input_q.npy"), q.reshape(-1).astype(np.float32))
    np.save(os.path.join(OUT, "input_k.npy"), k16.reshape(-1))
    np.save(os.path.join(OUT, "input_v.npy"), v16.reshape(-1))
    np.save(os.path.join(OUT, "input_gate.npy"), gate.reshape(-1).astype(np.float32))
    np.save(os.path.join(OUT, "scratch.npy"), scratch.reshape(-1).astype(np.float32))
    np.save(os.path.join(OUT, "expected.npy"), expected.reshape(-1).astype(np.float32))
    print("plen", plen, "scratch", scratch.shape, "expected[0][:4]", expected[0][:4])


if __name__ == "__main__":
    main()