#!/usr/bin/env python3
"""tierA.py: per-layer KV-codec fidelity on a dump (kvcodec.py --dump DIR, with
YAH_KV_HOOK_QSTRIDE so roped Q rows are included).

  tierA.py <dump dir> <kcodec:vcodec> [<kcodec:vcodec> ...] [--layers 0,5,15]

For every attention layer, the codec is applied chunk by chunk (2048 rows, as
the engine streams it) and, for each sampled query row q at position p and each
of the 24 query heads (GQA 6 per KV head), compared with exact attention over
keys 0..p (fp32, the dumped f16 K/V as the reference):
  out_err = |o_cand - o_ref| / |o_ref|   (attention output, pre-gate)
  kl      = KL(softmax_ref || softmax_cand) over the keys
Reported per codec: median / p90 / p99 of out_err, mean KL, and the mean of
out_err in query-position bins, plus K-only (V exact) and V-only (K exact)
splits, so each side of a codec is measured alone.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvcodec import Codec  # noqa: E402

CHUNK = 2048
QSUB = int(os.environ.get("KVQ_QSUB", "4"))     # use every QSUB-th dumped query row
BINS = [0, 2048, 8192, 16384, 32768, 65536]


def apply(codec_name, layer, x):
    c = Codec(codec_name)
    return np.concatenate([c(layer, x[s:s + CHUNK]) for s in range(0, len(x), CHUNK)])


def attn(q, k, v, p):
    """q [24, 256] f32 (already scaled), k/v [T, 4, 256]; keys 0..p"""
    kk, vv = k[:p + 1], v[:p + 1]
    out = np.empty((24, 256), np.float32); w = []
    for h in range(24):
        s = kk[:, h // 6] @ q[h]
        s -= s.max(); e = np.exp(s); e /= e.sum()
        out[h] = e @ vv[:, h // 6]; w.append(e)
    return out, w


def run(dump, specs, layers):
    for spec in specs:
        kn, vn = spec.split(":")
        rows = []
        for layer in layers:
            k = np.load(f"{dump}/k_l{layer:02d}.npy").astype(np.float32)
            v = np.load(f"{dump}/v_l{layer:02d}.npy").astype(np.float32)
            qp = np.load(f"{dump}/qpos_l{layer:02d}.npy"); qs = np.load(f"{dump}/q_l{layer:02d}.npy").astype(np.float32) * 0.0625
            qp, qs = qp[::QSUB], qs[::QSUB]
            kq = apply(kn, layer, k) if kn != "fp16" else k
            vq = apply(vn, layer, v) if vn != "fp16" else v
            for i, p in enumerate(qp):
                o_r, w_r = attn(qs[i], k, v, p)
                o_c, w_c = attn(qs[i], kq, vq, p)
                o_k, _ = attn(qs[i], kq, v, p)          # K only
                o_v, _ = attn(qs[i], k, vq, p)          # V only
                nr = np.linalg.norm(o_r, axis=1)
                kl = np.mean([np.sum(a * (np.log(a + 1e-30) - np.log(b + 1e-30))) for a, b in zip(w_r, w_c)])
                rows.append((layer, p, np.linalg.norm(o_c - o_r, axis=1) / nr, np.linalg.norm(o_k - o_r, axis=1) / nr,
                             np.linalg.norm(o_v - o_r, axis=1) / nr, kl))
        e = np.concatenate([r[2] for r in rows]); ek = np.concatenate([r[3] for r in rows]); ev = np.concatenate([r[4] for r in rows])
        kl = np.array([r[5] for r in rows])
        print(f"== {spec:22s} out_err median {np.median(e):.2e} p90 {np.percentile(e, 90):.2e} p99 {np.percentile(e, 99):.2e}"
              f" | K-only median {np.median(ek):.2e} | V-only median {np.median(ev):.2e} | KL mean {kl.mean():.2e}")
        pos = np.array([r[1] for r in rows])
        line = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (pos >= lo) & (pos < hi)
            if m.any():
                line.append(f"[{lo // 1024}K,{hi // 1024}K) {np.mean(np.concatenate([r[2] for r, mm in zip(rows, m) if mm])):.2e}")
        lay = []
        for layer in layers:
            m = [r for r in rows if r[0] == layer]
            lay.append(f"L{layer}:{np.median(np.concatenate([r[2] for r in m])):.1e}")
        print("   by position: " + "  ".join(line))
        print("   by layer (median): " + " ".join(lay))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dump, specs = args[0], args[1:]
    layers = sorted(int(os.path.basename(f)[3:5]) for f in glob.glob(f"{dump}/k_l*.npy"))
    if "--layers" in sys.argv:
        layers = [int(x) for x in sys.argv[sys.argv.index("--layers") + 1].split(",")]
    run(dump, specs, layers)


if __name__ == "__main__":
    main()
