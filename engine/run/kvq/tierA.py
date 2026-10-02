#!/usr/bin/env python3
"""Per-layer KV-codec fidelity on a dump (kvcodec.py --dump DIR, with YAH_KV_HOOK_QSTRIDE so roped Q rows are included).

  tierA.py <dump dir> <kcodec:vcodec> [<kcodec:vcodec> ...] [--layers 0,5,15]

For every attention layer, the codec runs chunk by chunk (2048 rows, as the engine streams it).
For each sampled query row q at position p and each of the 24 query heads (GQA 6 per KV head), it is compared with exact attention over keys 0..p.
The reference is fp32 attention over the dumped f16 K/V:
  out_err = |o_cand - o_ref| / |o_ref|   (attention output, pre-gate)
  kl      = KL(softmax_ref || softmax_cand) over the keys
Reported per codec: median / p90 / p99 of out_err, mean KL, and the mean of out_err in query-position bins.
K-only (V exact) and V-only (K exact) splits measure each side of a codec alone.
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


def apply(codec_name, layer, x, qp=None, qs=None):
    """Stream x through the codec chunk by chunk, with the chunk's dumped Q rows for codecs that calibrate on Q.

    The Q rows are unscaled, as the engine sends them.
    """
    c = Codec(codec_name)
    out = []
    for s in range(0, len(x), CHUNK):
        q = None
        if qp is not None:
            m = (qp >= s) & (qp < s + CHUNK)
            q = (qp[m], qs[m] / 0.0625)
        out.append(c(layer, x[s:s + CHUNK], q))
    return np.concatenate(out)


def probs(q, k, pos):
    """q [n, 6, 256] (one GQA group), k [T, 256], pos [n]: causal softmax [n*6, T]"""
    s = q.reshape(-1, q.shape[-1]) @ k.T
    p6 = np.repeat(pos, q.shape[1])
    s[np.arange(k.shape[0])[None, :] > p6[:, None]] = -np.inf
    s -= s.max(1, keepdims=True)
    np.exp(s, out=s)
    s /= s.sum(1, keepdims=True)
    return s


def rel(a, b):
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


def run(dump, specs, layers):
    data = {}
    for layer in layers:
        k = np.load(f"{dump}/k_l{layer:02d}.npy").astype(np.float32)
        v = np.load(f"{dump}/v_l{layer:02d}.npy").astype(np.float32)
        qpa = np.load(f"{dump}/qpos_l{layer:02d}.npy")
        qsa = np.load(f"{dump}/q_l{layer:02d}.npy").astype(np.float32) * 0.0625
        data[layer] = (k, v, qpa[::QSUB], qsa[::QSUB], qpa, qsa)
    for spec in specs:
        kn, vn = spec.split(":")
        E, EK, EV, KL, POS, LAY = [], [], [], [], [], []
        for layer in layers:
            k, v, qp, qs, qpa, qsa = data[layer]      # scored subsample / all dumped Q (codec calibration)
            kq = apply(kn, layer, k, qpa, qsa) if kn != "fp16" else k
            vq = apply(vn, layer, v, qpa, qsa) if vn != "fp16" else v
            for h in range(4):
                qg = qs[:, 6 * h:6 * h + 6]
                pr = probs(qg, k[:, h], qp)
                pc = probs(qg, kq[:, h], qp) if kn != "fp16" else pr
                o_r = pr @ v[:, h]
                o_k = pc @ v[:, h]
                o_v = pr @ vq[:, h]
                o_c = pc @ vq[:, h]
                E.append(rel(o_c, o_r)); EK.append(rel(o_k, o_r)); EV.append(rel(o_v, o_r))
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(pr > 0, pr * (np.log(pr) - np.log(np.maximum(pc, 1e-38))), 0)
                KL.append(t.sum(1))
                POS.append(np.repeat(qp, 6)); LAY.append(np.full(len(qp) * 6, layer))
        e, ek, ev, kl = (np.concatenate(x) for x in (E, EK, EV, KL))
        pos, lay = np.concatenate(POS), np.concatenate(LAY)
        print(f"== {spec:22s} out_err median {np.median(e):.2e} p90 {np.percentile(e, 90):.2e} p99 {np.percentile(e, 99):.2e}"
              f" | K-only median {np.median(ek):.2e} | V-only median {np.median(ev):.2e} | softmax KL mean {kl.mean():.2e}")
        line = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (pos >= lo) & (pos < hi)
            if m.any():
                line.append(f"[{lo // 1024}K,{hi // 1024}K) {np.median(e[m]):.2e}")
        print("   median by position: " + "  ".join(line))
        print("   K-only by layer: " + " ".join(f"L{l}:{np.median(ek[lay == l]):.1e}" for l in layers))
        print("   V-only by layer: " + " ".join(f"L{l}:{np.median(ev[lay == l]):.1e}" for l in layers), flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dump, specs = args[0], args[1:]
    layers = sorted(int(os.path.basename(f)[3:5]) for f in glob.glob(f"{dump}/k_l*.npy"))
    if "--layers" in sys.argv:
        layers = [int(x) for x in sys.argv[sys.argv.index("--layers") + 1].split(",")]
    run(dump, specs, layers)


if __name__ == "__main__":
    main()
