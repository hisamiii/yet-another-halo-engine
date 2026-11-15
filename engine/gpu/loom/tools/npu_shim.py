#!/usr/bin/env python3
"""npu_shim.py <run log>: per job family, the shim counters (2 per column) and memtile counters (4 per column) as a share of
the core cycle counter (core counter 0 = event TRUE) of tile (col, row 2). Values per job: 32 tiles x 6 (core 4, mem 2),
then per column memtile 4 + shim 2."""
import collections
import re
import sys

S = collections.defaultdict(lambda: [0.0] * 48)
CYC = collections.defaultdict(float)
MS = collections.defaultdict(float)
N = collections.Counter()
for l in open(sys.argv[1], errors="replace"):
    if not l.startswith("npu prof: job"):
        continue
    f = l.split()
    tag, ms, v = f[4], float(f[5]), list(map(int, f[6:]))
    fam = re.sub(r"[0-9]+$", "", tag.split("/")[0].split(":")[0])
    N[fam] += 1
    MS[fam] += ms
    CYC[fam] += max(v[(c * 4) * 6] for c in range(8))
    for i in range(48):
        S[fam][i] += v[192 + i]
for fam in sorted(S, key=lambda k: -MS[k]):
    c = CYC[fam]
    print(f"{fam:10s} jobs {N[fam]:4d} ms {MS[fam]:7.1f} cyc/job {c / N[fam] / 1e3:8.0f}K  (clock {c / MS[fam] / 1e6:.2f} GHz)")
    for k, name in ((4, "shim e0"), (5, "shim e1"), (0, "mt e0"), (1, "mt e1"), (2, "mt e2"), (3, "mt e3")):
        print(f"   {name:8s}" + " ".join(f"{100 * S[fam][col * 6 + k] / c:5.1f}" for col in range(8)) + " %")
