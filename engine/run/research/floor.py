#!/usr/bin/env python3
"""floor.py <parity table> <tokens> [ghz=2.55]: per kernel row, Loom device ms vs a hardware floor.
GEMM rows (key '<epi> <fmt> M K'): WMMA floor = n * M*K*B/4096 WMMAs * 32 cycles / 80 SIMDs.
attention: causal 16x16 tiles, 24 heads, 16 WMMAs QK + 16 PV per visible (16-query, 16-key) tile.
Other rows: no floor (memory/latency-bound; listed by time). Ranked by ms above floor."""
import re, sys
f = open(sys.argv[1]).read().split("\n"); B = int(sys.argv[2]); ghz = float(sys.argv[3]) if len(sys.argv) > 3 else 2.55
SIMDS, WC = 80, 32
rows = []
for l in f:
    m = re.match(r"^(\S.*?)\s{2,}(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s", l)
    if not m or l.startswith("key"): continue
    key, ln, lms = m[1].strip(), int(m[4]), float(m[5])
    if ln == 0: continue
    g = re.match(r"^(store|resid|ffn_gate\+up|swiglu|gateup)\s+(\S+)\s+(\d+)\s+(\d+)$", key)
    fl = None
    if g:
        M, K = int(g[3]), int(g[4]); w = M * K * B / 4096
        fl = ln * w * WC / SIMDS / (ghz * 1e6)
    elif key == "attention":
        T = B // 16; w = 24 * 32 * T * (T + 1) / 2
        fl = ln * w * WC / SIMDS / (ghz * 1e6)
    rows.append((key, ln, lms, fl))
tot = sum(r[2] for r in rows); tf = sum(r[3] for r in rows if r[3] is not None); tg = sum(r[2] for r in rows if r[3] is not None)
print(f"pp{B}: Loom device total {tot:.0f} ms; WMMA-bound rows {tg:.0f} ms vs floor {tf:.0f} ms ({100*tf/tg:.0f}% of floor), other rows {tot-tg:.0f} ms")
print(f"{'kernel':34s} {'n':>4s} {'ms':>8s} {'share':>6s} {'floor':>8s} {'%floor':>7s} {'above floor':>12s}")
for key, n, ms, fl in sorted(rows, key=lambda r: -((r[2] - r[3]) if r[3] is not None else r[2] * 0.0001))[:22]:
    print(f"{key:34s} {n:4d} {ms:8.1f} {100*ms/tot:5.1f}% " + (f"{fl:8.1f} {100*fl/ms:6.0f}% {ms-fl:11.1f}" if fl is not None else f"{'-':>8s} {'-':>7s} {'-':>11s}"))
print("non-WMMA rows by time:", ", ".join(f"{k} {ms:.0f}" for k, n, ms, fl in sorted(rows, key=lambda r: -r[2]) if fl is None and ms > 5))
