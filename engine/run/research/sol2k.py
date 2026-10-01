#!/usr/bin/env python3
"""sol2k.py <samples.jsonl> <seq.csv> <tokens>: speed-of-light table in cycles (SQ_BUSY_CYCLES per dispatch / 20
instances), grouped by kernel shape. GEMM floor = (M/16)(B/16)(K/16) WMMAs * 34 cycles / 80 SIMDs; attention =
24 heads * 32 WMMAs per visible causal 16x16 tile."""
import json, re, sys, collections
S, Q, B = sys.argv[1], sys.argv[2], int(sys.argv[3])
samp = sorted((json.loads(l) for l in open(S)), key=lambda r: r["dispatch_event_id"])
seq = [l.rstrip().split(",") for l in open(Q)][1:]
assert len(samp) == len(seq), (len(samp), len(seq))
FMT = {"iq3s": "IQ3_S", "iq3xxs": "IQ3_XXS", "iq4xs": "IQ4_XS", "q4k": "Q4_K", "q5k": "Q5_K", "q6k": "Q6_K", "q3k": "Q3_K",
       "q8_0": "Q8_0", "iq2xxs": "IQ2_XXS", "iq2xs": "IQ2_XS", "q2k": "Q2_K"}
KIND = {"kstore": "plain", "swiglu": "swiglu", "kres": "+resid", "residual": "+resid(old)"}
g = collections.defaultdict(lambda: [0, 0.0, 0.0])
for s, (_, key, gx, gy, gz) in zip(samp, seq):
    cyc = s["value"] / 20
    m = re.match(r"gemm_(\w+?)_(\w+)_(\d+)_(\d+)\.hal", key)
    fl = None
    if m:
        kind, fmt, mt, kb = m[1], m[2], int(m[3]), int(m[4]); M, K = mt * 16, kb * 256
        name = f"{FMT.get(fmt, fmt)} {KIND.get(kind, kind)} {M}x{K}"
        fl = (M / 16) * (B / 16) * (K / 16) * 34 / 80
    elif "attn_wmma" in key:
        T = B // 16; name = "attention"; fl = 24 * 32 * T * (T + 1) / 2 * 34 / 80
    else:
        name = key.replace(".hal", "")
    e = g[name]; e[0] += 1; e[1] += cyc; e[2] += fl if fl is not None else float("nan")
tot = sum(v[1] for v in g.values())
wm = {k: v for k, v in g.items() if v[2] == v[2]}
twm, tfl = sum(v[1] for v in wm.values()), sum(v[2] for v in wm.values())
print(f"pp{B}: {tot/1e6:.0f} M cycles total; WMMA kernels {twm/1e6:.0f} M ({100*twm/tot:.0f}%) at {100*tfl/twm:.0f}% of their floor "
      f"({tfl/1e6:.0f} M); non-WMMA {(tot-twm)/1e6:.0f} M ({100*(tot-twm)/tot:.0f}%)")
print(f"{'kernel':34s} {'n':>4s} {'M cyc':>8s} {'share':>6s} {'floor':>8s} {'%SOL':>5s} {'above':>8s}")
for k, (n, c, f) in sorted(g.items(), key=lambda kv: -(kv[1][1] - (kv[1][2] if kv[1][2] == kv[1][2] else 0))):
    if c / tot < 0.004: continue
    if f == f:
        print(f"{k:34s} {n:4d} {c/1e6:8.1f} {100*c/tot:5.1f}% {f/1e6:8.1f} {100*f/c:4.0f}% {(c-f)/1e6:8.1f}")
    else:
        print(f"{k:34s} {n:4d} {c/1e6:8.1f} {100*c/tot:5.1f}% {'-':>8s} {'-':>5s} {c/1e6:8.1f}")
