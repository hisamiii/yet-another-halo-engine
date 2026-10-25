#!/usr/bin/env python3
"""kld_ref.py <model> <hal set> <ref.bin> <workdir> [chunks]: KL divergence of the engine vs a BF16 reference.

ref.bin is hipfire's HFKLDR v1 (eval_hipfire --ref): per 2048-token chunk the BF16 model's top-256 log-probs and the
residual mass of positions 1024..2046. Each chunk runs as one fresh loom_forward_pp pp2048 (--logits-from 1024),
scored as eval_hipfire scores (prefill mode): KL over the reference's top 256 plus the residual-mass term.
Writes <workdir>/tokens.bin (per position: f64 KL, u8 top-1 match) and prints mean / p99 / p99.9 / max / top-1 / PPL.
Correctness runs only: no timing.
"""
import os
import struct
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def read_ref(path):
    f = open(path, "rb")
    if f.read(8) != b"HFKLDR\0\0":
        raise SystemExit(path + ": not an HFKLDR file")
    version, n_ctx, vocab, n_chunk, top_k, _flags = struct.unpack("<6I", f.read(24))
    if version != 1:
        raise SystemExit("HFKLDR version %d" % version)
    tokens = np.frombuffer(f.read(4 * n_ctx * n_chunk), np.uint32).reshape(n_chunk, n_ctx)
    rec = np.dtype([("idx", "<u4", top_k), ("logp", "<f4", top_k), ("res", "<f4"), ("pad", "<f4")])
    blocks = np.fromfile(f, rec).reshape(n_chunk, n_ctx // 2 - 1)
    return n_ctx, vocab, tokens, blocks


def score(logits, ref, nxt):
    """Per-position KL (f64) and top-1 match of f32 logits [n][vocab] vs reference blocks [n]; log p of tokens nxt."""
    x = logits.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    logz = m[:, 0] + np.log(np.exp(x - m).sum(axis=1))
    lp = np.take_along_axis(x, ref["idx"].astype(np.int64), axis=1) - logz[:, None]
    pr = np.exp(ref["logp"].astype(np.float64))
    kl = (pr * (ref["logp"] - lp)).sum(axis=1)
    res_ref = ref["res"].astype(np.float64)
    res_c = np.maximum(1.0 - np.exp(lp).sum(axis=1), 0.0)
    both = (res_ref > 1e-9) & (res_c > 1e-9)
    kl[both] += res_ref[both] * (np.log(res_ref[both]) - np.log(res_c[both]))
    top = np.take_along_axis(ref["idx"], ref["logp"].argmax(axis=1)[:, None], axis=1)[:, 0]
    match = x.argmax(axis=1) == top
    nll = logz - x[np.arange(len(x)), nxt]
    return np.maximum(kl, 0.0), match, nll


def main():
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    model, hal, refp, work = sys.argv[1:5]
    n_ctx, vocab, tokens, blocks = read_ref(refp)
    chunks = int(sys.argv[5]) if len(sys.argv) > 5 else len(tokens)
    os.makedirs(work, exist_ok=True)
    kls, matches, nlls = [], [], []
    for c in range(chunks):
        ids = os.path.join(work, "ids.txt")
        open(ids, "w").write(" ".join(map(str, tokens[c])) + "\n")
        prefix = os.path.join(work, "c%02d" % c)
        cmd = ["bash", "-c", 'a=("$@"); set --; source engine/hrx-env.sh >/dev/null; exec "${a[@]}"', "-", "engine/run/gpu_run.sh", "kld",
               "--", "engine/build/loom_forward_pp", model, hal, prefix, str(n_ctx), ids,
               "--logits-from", str(n_ctx // 2)]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode or "all_logits" not in r.stdout:
            raise SystemExit("chunk %d failed:\n%s%s" % (c, r.stdout[-1500:], r.stderr[-1500:]))
        n = blocks.shape[1]
        logits = np.fromfile(prefix + ".all_logits", np.float32).reshape(-1, vocab)[:n]
        kl, match, nll = score(logits, blocks[c], tokens[c][n_ctx // 2 + 1:n_ctx // 2 + 1 + n].astype(np.int64))
        kls.append(kl), matches.append(match), nlls.append(nll)
        for ext in (".all_logits", ".logits", ".hidden"):
            os.remove(prefix + ext)
        print("chunk %2d: mean KL %.4f top-1 %.2f%%" % (c, kl.mean(), 100 * match.mean()), flush=True)
    kl, match, nll = np.concatenate(kls), np.concatenate(matches), np.concatenate(nlls)
    rec = np.zeros(len(kl), np.dtype([("kld", "<f8"), ("m", "u1")]))
    rec["kld"], rec["m"] = kl, match
    rec.tofile(os.path.join(work, "tokens.bin"))
    print("n %d mean KL %.4f p99 %.3f p99.9 %.3f max %.2f top-1 %.2f%% PPL %.3f" % (
        len(kl), kl.mean(), np.quantile(kl, .99), np.quantile(kl, .999), kl.max(), 100 * match.mean(),
        np.exp(nll.mean())))


if __name__ == "__main__":
    main()
