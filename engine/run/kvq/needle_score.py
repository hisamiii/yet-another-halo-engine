#!/usr/bin/env python3
"""needle_score.py: tier C scoring. For each prompt (gen_needle.py .json) and
run (row stats at the answer positions), per query: log p(answer) = sum over
answer tokens of (logit_target - logsumexp), correct = every answer token is
the argmax, margin = min over answer tokens of (target logit - best other).

  needle_score.py <prompt.json,...> <ref.rs,...> <cand.rs,...> [--name N]
Reports per-query deltas vs the reference with a bootstrap CI over queries.
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gate2 import load  # noqa: E402


def score(js, rs):
    d = json.load(open(js)); r = load(rs)
    at = {int(p): i for i, p in enumerate(r["pos"])}
    out = []
    for q in d["queries"]:
        lp, ok, mg = 0.0, True, np.inf
        for j, t in enumerate(q["tokens"]):
            i = at[q["start"] + j - 1]
            assert r["nxt"][i] == t
            lp += float(r["lt"][i] - r["lse"][i])
            ok &= int(r["am"][i]) == t
            other = r["t2"][i] if int(r["am"][i]) == t else r["t1"][i]
            mg = min(mg, float(r["lt"][i] - other))
        out.append((lp, ok, mg, q["depth"], d["length"], q.get("updated", False)))
    return out


def main():
    js, refs, cands = (a.split(",") for a in sys.argv[1:4])
    name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else cands[0]
    R = [x for j, r in zip(js, refs) for x in score(j, r)]
    C = [x for j, c in zip(js, cands) for x in score(j, c)]
    dlp = np.array([c[0] - r[0] for r, c in zip(R, C)])
    rng = np.random.default_rng(0)
    bs = [dlp[rng.integers(0, len(dlp), len(dlp))].mean() for _ in range(2000)]
    acc_r = np.mean([r[1] for r in R]); acc_c = np.mean([c[1] for c in C])
    mg_r = np.array([r[2] for r in R]); mg_c = np.array([c[2] for c in C])
    print(f"== {name}: {len(dlp)} queries   d log p(answer) {dlp.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"
          f"   worst {dlp.min():+.3f}   exact-match {100 * acc_c:.1f}% (ref {100 * acc_r:.1f}%)"
          f"   min-margin median {np.median(mg_c):.2f} (ref {np.median(mg_r):.2f}), margin<1: {np.mean(mg_c < 1):.2f} (ref {np.mean(mg_r < 1):.2f})")
    dep = np.array([r[3] for r in R]); upd = np.array([r[5] for r in R])
    parts = []
    for lo, hi in ((0, .33), (.33, .66), (.66, 1.01)):
        m = (dep >= lo) & (dep < hi)
        if m.any():
            parts.append(f"depth {lo:.2f}-{min(hi, 1):.2f}: n={m.sum()} dlogp {dlp[m].mean():+.4f} ref logp {np.mean([r[0] for r, mm in zip(R, m) if mm]):+.3f}")
    if upd.any():
        parts.append(f"reassigned keys: n={upd.sum()} dlogp {dlp[upd].mean():+.4f} ref logp {np.mean([r[0] for r, u in zip(R, upd) if u]):+.3f}")
    print("   " + " | ".join(parts))


if __name__ == "__main__":
    main()
