#!/usr/bin/env python3
"""sol.py <sol-dir> <wmma_per_dispatch> [simds=80] [instances=20]: speed-of-light / issue-bound view from solpmc.sh counters.
Issue model (research/hw-measured.md): per SIMD, cycles >= 34*WMMA + VALU_cycles(non-WMMA) + 3*LDS_instr.
SQ_INST_CYCLES_VALU counts 1 per instruction plus extra for multi-cycle ops (it does NOT include WMMA pipe time)."""
import sqlite3, glob, sys, collections
d, W = sys.argv[1], float(sys.argv[2]); S = int(sys.argv[3]) if len(sys.argv) > 3 else 80; I = int(sys.argv[4]) if len(sys.argv) > 4 else 20
tot = {}
for db in sorted(glob.glob(d + "/p*/run_results.db")):
    c = sqlite3.connect(db); per = collections.defaultdict(collections.Counter)
    for dd, cn, v in c.execute("select dispatch_id, counter_name, counter_value from pmc_events"): per[dd][cn] += v
    tot.update(per[max(per)])
cyc = tot["SQ_BUSY_CYCLES"] / I
valu_c = tot["SQ_INST_CYCLES_VALU"] - W; lds = tot["SQ_INSTS_LDS"]
pred = (34 * W + valu_c + 3 * lds) / S
wc = tot["SQ_WAVE_CYCLES"]
print(f"{d}: cycles {cyc/1e6:.2f} M | per WMMA: cycles {cyc*S/W:.1f}, VALU(non-WMMA) {valu_c/W:.2f}, LDS {lds/W:.2f}, SALU {tot.get('SQ_INSTS_SALU',0)/W:.2f}")
print(f"  issue bound {pred/1e6:.2f} M = {100*pred/cyc:.0f}% of measured  [WMMA {34*W/S/1e6:.2f} + VALU {valu_c/S/1e6:.2f} + LDS {3*lds/S/1e6:.2f} M]")
print(f"  wave time: barrier {100*tot['SQ_WAIT_BARRIER']/wc:.0f}%, waiting to issue {100*tot['SQ_WAIT_INST_ANY']/wc:.0f}%, any wait {100*tot['SQ_WAIT_ANY']/wc:.0f}%; waves/SIMD ~{wc/cyc/S:.1f}")
