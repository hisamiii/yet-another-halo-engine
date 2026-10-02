#!/usr/bin/env python3
"""Multi-window, all-position correctness gate: candidate logits vs golden logits.

Inputs are directories of per-window f32 logits of the scored positions, `wNN.all_logits` (rows = positions FROM..2047, vocab-major rows).
loom_forward_pp writes them with YAH_LOGITS_FROM=FROM (gate/gate_run.sh).
The token ids of each window (corpus/ids2048_wNN.txt) give the next-token targets for NLL.

  accgate2.py stats     <golden_dir> <cand_dir> [--windows 00,01,...]
  accgate2.py calibrate <golden_dir> <ref_dir> <thresholds.json> [--windows ...]   (writes T1 thresholds)
  accgate2.py check     <golden_dir> <cand_dir> <thresholds.json> [--windows ...]

Per position (f64): KL(golden || cand), logits relative RMS, top-1 flip, NLL of the actual next token in both.
A flip counts only where the golden top-1/top-2 gap exceeds TIE logits.
Summary: KL mean / p99 / p99.9 / max, flips, mean ln PPL ratio (cand vs golden), max logits rel RMS, non-finite count.

Calibration (tier T1, rounding level): thresholds from a reference engine's distance D to the golden.
mean KL <= FACTOR * D_mean (FACTOR default 0.25), p99.9 KL <= D_p99.9, flips <= D_flips, |ln PPL ratio| <= max(|D_lnppl|, 1e-4).
"""
import json
import os
import sys

import numpy as np

V = 248320
FROM = int(os.environ.get("FROM", "1536"))
TIE = float(os.environ.get("TIE", "0.1"))
CORPUS = os.environ.get("CORPUS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate", "corpus"))
CHUNK = 32
# Long windows "L<n>": corpus/ids8192_w<n>.txt, run on an 8192-token HAL set.
# Scored from FROM_L (default 7680: 512 positions, as in the 2048 windows).
FROM_L = int(os.environ.get("FROM_L", "7680"))


def ids_file(w):
    return f"ids8192_w{w[1:]}.txt" if w.startswith("L") else f"ids2048_w{w}.txt"


def window_stats(gdir, cdir, w):
    g = np.memmap(os.path.join(gdir, f"w{w}.all_logits"), dtype=np.float32, mode="r").reshape(-1, V)
    c = np.memmap(os.path.join(cdir, f"w{w}.all_logits"), dtype=np.float32, mode="r").reshape(-1, V)
    assert g.shape == c.shape, (g.shape, c.shape)
    ids = np.array(open(os.path.join(CORPUS, ids_file(w))).read().split(), dtype=np.int64)
    base = FROM_L if w.startswith("L") else FROM
    rows = g.shape[0]
    kl = np.empty(rows); rrms = np.empty(rows); flip = np.zeros(rows, bool); tie = np.zeros(rows, bool)
    nll_g = []; nll_c = []; nonfinite = 0
    for s in range(0, rows, CHUNK):
        a = np.asarray(g[s:s + CHUNK], dtype=np.float64); b = np.asarray(c[s:s + CHUNK], dtype=np.float64)
        nonfinite += int((~np.isfinite(b)).sum())
        ma = a.max(1, keepdims=True); mb = b.max(1, keepdims=True)
        la = a - ma - np.log(np.exp(a - ma).sum(1, keepdims=True))
        lb = b - mb - np.log(np.exp(b - mb).sum(1, keepdims=True))
        pa = np.exp(la)
        kl[s:s + len(a)] = (pa * (la - lb)).sum(1)
        rrms[s:s + len(a)] = np.sqrt(((a - b) ** 2).mean(1)) / np.sqrt((a ** 2).mean(1))
        top2 = np.partition(a, -2, axis=1)[:, -2:]
        gap = top2[:, 1] - top2[:, 0]
        am, bm = a.argmax(1), b.argmax(1)
        tie[s:s + len(a)] = gap <= TIE
        flip[s:s + len(a)] = (am != bm) & (gap > TIE)
        pos = base + s + np.arange(len(a))          # position p predicts token p+1
        ok = pos + 1 < len(ids)
        t = ids[pos[ok] + 1]
        nll_g += list(-la[ok][np.arange(ok.sum()), t]); nll_c += list(-lb[ok][np.arange(ok.sum()), t])
    return dict(kl=kl, rrms=rrms, flips=int(flip.sum()), ties=int(tie.sum()), rows=rows,
                nll_g=np.array(nll_g), nll_c=np.array(nll_c), nonfinite=nonfinite)


def summarize(parts):
    kl = np.concatenate([p["kl"] for p in parts]); rr = np.concatenate([p["rrms"] for p in parts])
    ng = np.concatenate([p["nll_g"] for p in parts]); nc = np.concatenate([p["nll_c"] for p in parts])
    return {
        "positions": int(kl.size),
        "kl_mean": float(kl.mean()), "kl_p99": float(np.percentile(kl, 99)),
        "kl_p999": float(np.percentile(kl, 99.9)), "kl_max": float(kl.max()),
        "flips": int(sum(p["flips"] for p in parts)), "ties": int(sum(p["ties"] for p in parts)),
        "ln_ppl_ratio": float(nc.mean() - ng.mean()),
        "ppl_golden": float(np.exp(ng.mean())), "ppl_cand": float(np.exp(nc.mean())),
        "logits_rrms_mean": float(rr.mean()), "logits_rrms_max": float(rr.max()),
        "nonfinite": int(sum(p["nonfinite"] for p in parts)),
    }


def windows_arg(argv):
    if "--windows" in argv:
        return argv[argv.index("--windows") + 1].split(",")
    return ["00", "01", "02", "03"]


def run(gdir, cdir, ws):
    return summarize([window_stats(gdir, cdir, w) for w in ws])


def main():
    mode = sys.argv[1]
    ws = windows_arg(sys.argv)
    if mode == "stats":
        print(json.dumps(run(sys.argv[2], sys.argv[3], ws), indent=1))
    elif mode == "calibrate":
        d = run(sys.argv[2], sys.argv[3], ws)
        factor = float(os.environ.get("FACTOR", "0.25"))
        th = {"tier": "T1", "windows": ws, "from": FROM, "tie": TIE, "reference": sys.argv[3], "golden": sys.argv[2],
              "reference_distance": d,
              "kl_mean_max": factor * d["kl_mean"], "kl_p999_max": d["kl_p999"],
              "flips_max": d["flips"], "abs_ln_ppl_ratio_max": max(abs(d["ln_ppl_ratio"]), 1e-4)}
        json.dump(th, open(sys.argv[4], "w"), indent=1)
        print(json.dumps(th, indent=1))
    elif mode == "check":
        th = json.load(open(sys.argv[4]))
        ws = windows_arg(sys.argv) if "--windows" in sys.argv else th["windows"]
        d = run(sys.argv[2], sys.argv[3], ws)
        checks = [
            ("kl_mean", d["kl_mean"], th["kl_mean_max"]),
            ("kl_p999", d["kl_p999"], th["kl_p999_max"]),
            ("flips", d["flips"], th["flips_max"]),
            ("abs_ln_ppl_ratio", abs(d["ln_ppl_ratio"]), th["abs_ln_ppl_ratio_max"]),
            ("nonfinite", d["nonfinite"], 0),
        ]
        ok = True
        for name, v, lim in checks:
            good = v <= lim
            ok &= good
            print(f"{name:18s} {v:12.4g}  limit {lim:12.4g}  {'ok' if good else 'FAIL'}")
        print(f"positions {d['positions']}, ppl golden {d['ppl_golden']:.4f} cand {d['ppl_cand']:.4f}, "
              f"logits rrms mean {d['logits_rrms_mean']:.3g} max {d['logits_rrms_max']:.3g}")
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
