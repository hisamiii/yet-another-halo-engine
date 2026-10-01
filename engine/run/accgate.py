#!/usr/bin/env python3
"""accgate.py: tolerance gate for arithmetic-changing optimisations.

Bit-identity (the hidden md5) stays the gate for changes that claim to keep
the arithmetic. A change that alters it (f16 decode math, single-rounding
fma_mix, reordered accumulation, ...) is compared against a FROZEN golden
output instead -- never against the previous candidate, so small errors cannot
pile up -- with thresholds calibrated on HIP: HIP and Loom are two correct
implementations that already disagree slightly, and a candidate must stay well
inside that disagreement.

  accgate.py calibrate <golden-prefix> <hip.logits> [--factor 0.25]
      writes <golden-prefix>.thresholds.json
  accgate.py check <golden-prefix> <candidate-prefix>
      exit 0 = pass, 1 = fail; prints every metric against its threshold

A prefix names <prefix>.logits (last token, vocab f32) and, for Loom,
<prefix>.hidden (B x 5120 f32, token major; loom_forward_pp writes both).

Metrics (last-token logits): argmax equal; KL(golden || candidate) on
float64 softmax; relative RMS of the logits; top-10 overlap. Hidden state
(Loom only, no HIP counterpart): relative RMS over all tokens and the worst
token's relative error -- held to the logits relative-RMS threshold, which is
provisional (both are relative errors of the same final hidden state, but the
scale is not calibrated directly).
"""
import json
import sys

import numpy as np


def load_logits(prefix_or_file):
    path = prefix_or_file if prefix_or_file.endswith(".logits") else prefix_or_file + ".logits"
    return np.fromfile(path, dtype=np.float32).astype(np.float64)


def softmax(x):
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def logit_metrics(g, c):
    pg, pc = softmax(g), softmax(c)
    kl = float(np.sum(pg * (np.log(pg + 1e-300) - np.log(pc + 1e-300))))
    rel = float(np.linalg.norm(c - g) / np.linalg.norm(g))
    top_g = set(np.argsort(-g)[:10]); top_c = set(np.argsort(-c)[:10])
    return {"argmax_golden": int(g.argmax()), "argmax_candidate": int(c.argmax()),
            "kl": kl, "logits_rel_rms": rel, "top10_overlap": len(top_g & top_c),
            "logits_max_abs": float(np.abs(c - g).max())}


def hidden_metrics(gp, cp):
    g = np.fromfile(gp + ".hidden", dtype=np.float32).reshape(-1, 5120).astype(np.float64)
    c = np.fromfile(cp + ".hidden", dtype=np.float32).reshape(-1, 5120).astype(np.float64)
    rel = float(np.linalg.norm(c - g) / np.linalg.norm(g))
    per = np.linalg.norm(c - g, axis=1) / np.maximum(np.linalg.norm(g, axis=1), 1e-30)
    return {"hidden_rel_rms": rel, "hidden_worst_token_rel": float(per.max()),
            "hidden_worst_token": int(per.argmax())}


def main():
    if len(sys.argv) < 4 or sys.argv[1] not in ("calibrate", "check"):
        sys.exit(__doc__)
    mode, gp = sys.argv[1], sys.argv[2]
    if mode == "calibrate":
        factor = float(sys.argv[sys.argv.index("--factor") + 1]) if "--factor" in sys.argv else 0.25
        m = logit_metrics(load_logits(gp), load_logits(sys.argv[3]))
        th = {"factor": factor, "hip_vs_golden": m,
              "kl_max": factor * m["kl"], "logits_rel_rms_max": factor * m["logits_rel_rms"],
              "top10_overlap_min": 9, "hidden_rel_rms_max": factor * m["logits_rel_rms"],
              "hidden_worst_token_rel_max": 4 * factor * m["logits_rel_rms"]}
        json.dump(th, open(gp + ".thresholds.json", "w"), indent=1)
        print(json.dumps(th, indent=1))
        return
    th = json.load(open(gp + ".thresholds.json"))
    cp = sys.argv[3]
    m = logit_metrics(load_logits(gp), load_logits(cp))
    try:
        m.update(hidden_metrics(gp, cp))
    except FileNotFoundError:
        pass
    checks = [
        ("argmax", m["argmax_candidate"] == m["argmax_golden"], f"{m['argmax_candidate']} vs golden {m['argmax_golden']}"),
        ("kl", m["kl"] <= th["kl_max"], f"{m['kl']:.3e} <= {th['kl_max']:.3e}"),
        ("logits_rel_rms", m["logits_rel_rms"] <= th["logits_rel_rms_max"], f"{m['logits_rel_rms']:.3e} <= {th['logits_rel_rms_max']:.3e}"),
        ("top10_overlap", m["top10_overlap"] >= th["top10_overlap_min"], f"{m['top10_overlap']} >= {th['top10_overlap_min']}"),
    ]
    if "hidden_rel_rms" in m:
        checks += [
            ("hidden_rel_rms (provisional)", m["hidden_rel_rms"] <= th["hidden_rel_rms_max"], f"{m['hidden_rel_rms']:.3e} <= {th['hidden_rel_rms_max']:.3e}"),
            ("hidden_worst_token (provisional)", m["hidden_worst_token_rel"] <= th["hidden_worst_token_rel_max"],
             f"{m['hidden_worst_token_rel']:.3e} <= {th['hidden_worst_token_rel_max']:.3e} (token {m['hidden_worst_token']})"),
        ]
    ok = all(p for _, p, _ in checks)
    for name, p, msg in checks:
        print(f"  {'PASS' if p else 'FAIL'}  {name:34s} {msg}")
    hip = th["hip_vs_golden"]
    print(f"  (scale: HIP vs golden KL {hip['kl']:.3e}, logits rel RMS {hip['logits_rel_rms']:.3e}; factor {th['factor']})")
    print("accgate:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
