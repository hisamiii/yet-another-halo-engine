#!/usr/bin/env bash
# l2.sh <hsaco> <tag>: GL2 hit rate + DRAM read bytes for one attention dispatch (pp8192, real inputs)
D=/home/q/yah-scratch/attn8k; B=8192; LH=/home/q/yet-another-halo-engine/engine/build/loomhip; GR=/home/q/yet-another-halo-engine/engine/run/gpu_run.sh
TR=/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0; cd /home/q/yah-scratch/fa; t=$2; rm -rf l2-$t; sleep 1
for g in "GL2C_HIT GL2C_MISS SQ_BUSY_CYCLES" "GL2C_EA_RDREQ_32B GL2C_EA_RDREQ_64B GL2C_EA_RDREQ_96B GL2C_EA_RDREQ_128B"; do n=$((n+1)); rm -rf l2-$t-$n; sleep 1
$GR fal2 -- $TR/bin/rocprofv3 --pmc $g --kernel-include-regex yah_attn_wmma -d l2-$t-$n -o run -- $LH $1 yah_attn_wmma $((B/32)) 12 256 3 f:$D/l3.aq f:$D/l3.agate f:$D/l3.ak16 f:$D/vt.bin z:$((B*6144*2)) z:$((B*24*4)) > l2-$t-$n.log 2>&1; done
python3 - $t <<'P'
import sqlite3, collections, sys, glob
x = collections.Counter()
for db in glob.glob(f"l2-{sys.argv[1]}-*/run_results.db"):
    c = sqlite3.connect(db); per = collections.defaultdict(collections.Counter)
    for d, cn, v in c.execute("select dispatch_id, counter_name, counter_value from pmc_events"): per[d][cn] += v
    x.update(per[max(per)])
sys.argv.insert(1, "")
h, m = x["GL2C_HIT"], x["GL2C_MISS"]
dram = 32*x["GL2C_EA_RDREQ_32B"] + 64*x["GL2C_EA_RDREQ_64B"] + 96*x["GL2C_EA_RDREQ_96B"] + 128*x["GL2C_EA_RDREQ_128B"]
cyc = x["SQ_BUSY_CYCLES"] / 20
print(f"{sys.argv[2]:8s} L2 hit {100*h/(h+m):.1f}%  requests {(h+m)/1e6:.0f} M  DRAM read {dram/1e9:.2f} GB  ({dram/(cyc/2.3e9)/1e9:.0f} GB/s at ~2.3 GHz)  {cyc/1e6:.1f} M cycles")
P
