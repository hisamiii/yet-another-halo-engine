#!/usr/bin/env python3
"""gate2.py: compare compact row stats (loom_forward_pp YAH_ROWSTATS) of a
candidate KV codec against the fp16-KV reference, per context-position bin,
with block-bootstrap confidence intervals across documents.

  gate2.py <ref1.rs,ref2.rs,...> <cand1.rs,cand2.rs,...> [--name NAME]

Pairs are matched by position. Per position:
  dNLL  = NLL_cand - NLL_ref of the true next token (nats)
  KL    ~ KL(ref || cand) over the reference's top-64 tokens (candidate log-probs
          outside its own top-64 bounded by its 64th logit: a slight underestimate)
  flip  = argmax differs where the reference top-1/top-2 gap exceeds TIE (0.1)
  margin: reference argmax token's logit minus the best other, in the candidate
Bins: [1K,2K) [2K,4K) [4K,8K) [8K,16K) [16K,32K) [32K,64K) ...
CI: 2000 bootstrap resamples of blocks of BLOCK consecutive sampled positions
(per document), 95% percentile interval of the mean.
"""
import sys

import numpy as np

REC = np.dtype([("pos", "<i4"), ("nxt", "<i4"), ("lse", "<f4"), ("lt", "<f4"), ("am", "<i4"),
                ("t1", "<f4"), ("t2", "<f4"), ("top", [("id", "<i4"), ("lg", "<f4")], (64,))])
TIE = 0.1
BLOCK = 32
BINS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def load(path):
    return np.fromfile(path, REC)


def per_pos(r, c):
    assert np.array_equal(r["pos"], c["pos"]), "position sets differ"
    ok = r["nxt"] >= 0
    nll_r = (r["lse"] - r["lt"]).astype(np.float64)
    nll_c = (c["lse"] - c["lt"]).astype(np.float64)
    lp_r = r["top"]["lg"].astype(np.float64) - r["lse"][:, None]
    p_r = np.exp(lp_r)
    # candidate log-probs at the reference's top-64 ids
    lp_c = np.empty_like(lp_r)
    cid, clg = c["top"]["id"], c["top"]["lg"].astype(np.float64)
    floor = clg[:, -1] - c["lse"]
    for i in range(len(r)):
        m = dict(zip(cid[i].tolist(), (clg[i] - c["lse"][i]).tolist()))
        lp_c[i] = [m.get(t, floor[i]) for t in r["top"]["id"][i].tolist()]
    kl = (p_r * (lp_r - lp_c)).sum(1)
    flip = (r["am"] != c["am"]) & ((r["t1"] - r["t2"]) > TIE)
    return dict(pos=r["pos"], ok=ok, dnll=nll_c - nll_r, nll_r=nll_r, kl=kl, flip=flip.astype(np.float64))


def boot_ci(groups, rng, n=2000):
    """groups: list of 1-D arrays (one per document); block bootstrap of the mean"""
    blocks = []
    for g in groups:
        for s in range(0, len(g), BLOCK):
            blocks.append(g[s:s + BLOCK])
    if not blocks:
        return np.nan, np.nan, np.nan
    sums = np.array([b.sum() for b in blocks]); cnts = np.array([len(b) for b in blocks])
    idx = rng.integers(0, len(blocks), (n, len(blocks)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return sums.sum() / cnts.sum(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    refs = sys.argv[1].split(","); cands = sys.argv[2].split(",")
    name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else cands[0]
    rng = np.random.default_rng(0)
    docs = [per_pos(load(a), load(b)) for a, b in zip(refs, cands)]
    if "--summary" in sys.argv:     # one markdown row: PPL, mean / p99 / p99.9 KL, p99.9 precision = 100 exp(-KL)
        kl = np.concatenate([d["kl"][d["ok"]] for d in docs])
        nr = np.concatenate([d["nll_r"][d["ok"]] for d in docs]); dn = np.concatenate([d["dnll"][d["ok"]] for d in docs])
        p999 = np.percentile(kl, 99.9)
        print(f"| {name} | {np.exp(nr.mean() + dn.mean()):.3f} | {max(kl.mean(), 0):.6f} | {np.percentile(kl, 99):.4f} | {p999:.4f} | {100 * np.exp(-p999):.2f}% |")
        return
    print(f"== {name}  ({len(docs)} docs, {sum(len(d['pos']) for d in docs)} positions)")
    print(f"{'bin':>13s} {'n':>6s} {'dNLL mean [95% CI] (nats)':>36s} {'KL mean':>10s} {'KL p99':>9s} {'KL p99.9':>9s} {'flips/1k':>9s}")
    allb = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = [(d, (d["pos"] >= lo) & (d["pos"] < hi) & d["ok"]) for d in docs]
        n = sum(int(m.sum()) for _, m in sel)
        if not n:
            continue
        m, a, b = boot_ci([d["dnll"][s] for d, s in sel], rng)
        kl = np.concatenate([d["kl"][s] for d, s in sel])
        fl = np.concatenate([d["flip"][s] for d, s in sel])
        print(f"{lo // 1024:5d}K-{hi // 1024:3d}K {n:6d} {m:+11.5f} [{a:+.5f}, {b:+.5f}] {kl.mean():10.2e} {np.percentile(kl, 99):9.2e} {np.percentile(kl, 99.9):9.2e} {1000 * fl.mean():9.2f}")
        allb.append((lo, m, a, b))
    sel = [(d, d["ok"]) for d in docs]
    m, a, b = boot_ci([d["dnll"][s] for d, s in sel], rng)
    kl = np.concatenate([d["kl"][s] for d, s in sel]); fl = np.concatenate([d["flip"][s] for d, s in sel])
    ppl_r = np.exp(np.concatenate([d["nll_r"][s] for d, s in sel]).mean())
    print(f"{'all':>13s} {sum(int(s.sum()) for _, s in sel):6d} {m:+11.5f} [{a:+.5f}, {b:+.5f}] {kl.mean():10.2e} {np.percentile(kl, 99):9.2e} {np.percentile(kl, 99.9):9.2e} {1000 * fl.mean():9.2f}"
          f"   (ref PPL {ppl_r:.3f}, cand {ppl_r * np.exp(m):.3f}, {100 * (np.exp(m) - 1):+.2f}%)")
    # tail KL with a block-bootstrap CI (bins hold too few positions for a p99.9 of their own)
    kls = [d["kl"][s] for d, s in sel]
    blocks = [g[i:i + BLOCK] for g in kls for i in range(0, len(g), BLOCK)]
    bs = {q: [] for q in (99, 99.9)}
    for _ in range(1000):
        x = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        for q in bs:
            bs[q].append(np.percentile(x, q))
    def tail(q):
        v, lo, hi = np.percentile(kl, q), np.percentile(bs[q], 2.5), np.percentile(bs[q], 97.5)
        # precision = 100 exp(-KL)
        return (f"p{q}: {v:.4f} [{lo:.4f}, {hi:.4f}] = precision {100 * np.exp(-v):.2f}%"
                f" [{100 * np.exp(-hi):.2f}, {100 * np.exp(-lo):.2f}]")
    print("   KL tail: " + "   ".join(tail(q) for q in bs)
          + f"   max {kl.max():.2e} (n={len(kl)}, ~{len(kl) / 1000:.0f} positions above p99.9)")


if __name__ == "__main__":
    main()
