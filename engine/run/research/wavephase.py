#!/usr/bin/env python3
"""wavephase.py <ATT ui dir>: per wave on the traced SIMD: begin, prologue (begin -> first WMMA), K loop
(first -> last WMMA), epilogue (last WMMA -> end); plus the SIMD timeline split into those phases."""
import glob, json, os, sys, numpy as np
d = sys.argv[1]
code = json.load(open(os.path.join(d, "code.json")))["code"]; text = {r[2]: r[0] for r in code}
W = []
for f in glob.glob(os.path.join(d, "se*_sm*_sl*_wv*.json")):
    w = json.load(open(f))["wave"]
    wm = [t for t, c, s, du, i in w["instructions"] if "wmma" in text.get(i, "")]
    if wm: W.append((w["begin"], wm[0], wm[-1], w["end"], len(wm)))
W.sort(); t0 = W[0][0]; T = max(w[3] for w in W) - t0
print(f"{len(W)} waves, SIMD window {T} units")
print(" wave  begin  prologue   kloop  epilogue   end   (units rel. to first wave)")
for k, (b, f, l, e, n) in enumerate(W):
    print(f"{k:4d} {b-t0:7d} {f-b:8d} {l-f:8d} {e-l:8d} {e-t0:7d}")
pro = sum(f - b for b, f, l, e, n in W); kl = sum(l - f for b, f, l, e, n in W); ep = sum(e - l for b, f, l, e, n in W)
print(f"sum over waves: prologue {pro}, kloop {kl}, epilogue {ep}  -> prologue+epilogue = {100*(pro+ep)/(pro+kl+ep):.1f}% of wave time")
# SIMD-level: fraction of the window where no resident wave is inside its K loop
cov = np.zeros(T + 1, bool)
for b, f, l, e, n in W: cov[f - t0:l - t0] = True
print(f"SIMD time with no wave in its K loop: {100*(1-cov.mean()):.1f}%")
