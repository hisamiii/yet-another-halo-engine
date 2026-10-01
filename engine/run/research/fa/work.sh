#!/usr/bin/env bash
# work.sh <tag> <hsaco> <K> <V> [extra bindings]: attention work counters at pp8192 (4 passes, 1 s gaps)
D=/home/q/yah-scratch/attn8k; B=8192; LH=/home/q/yet-another-halo-engine/engine/build/loomhip; GR=/home/q/yet-another-halo-engine/engine/run/gpu_run.sh
TR=/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0; export ROCPROFILER_METRICS_PATH=/home/q/yah-scratch/rcmetrics
cd /home/q/yah-scratch/fa; t=$1; h=$2; Kf=$3; Vf=$4; shift 4; n=0
for g in "SQ_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_LDS SQ_INSTS_SALU" "GL2C_HIT GL2C_MISS" "GL2C_EA_RDREQ_32B GL2C_EA_RDREQ_64B GL2C_EA_RDREQ_96B GL2C_EA_RDREQ_128B" "SQ_LDS_IDX_ACTIVE SQ_LDS_BANK_CONFLICT SQ_INSTS_VMEM SQ_WAVE_CYCLES"; do
  n=$((n+1)); rm -rf wk-$t-$n; sleep 1
  $GR wk -- $TR/bin/rocprofv3 --pmc $g --kernel-include-regex yah_attn_wmma -d wk-$t-$n -o run -- $LH $h yah_attn_wmma 256 12 256 3 f:$D/l3.aq f:$D/l3.agate f:$Kf f:$Vf z:$((B*6144*2)) z:$((B*24*4)) "$@" > wk-$t-$n.log 2>&1
done
python3 - $t $h <<'P'
import sqlite3, collections, sys, glob, json, os
t = sys.argv[1]; x = collections.Counter()
for db in glob.glob(f"wk-{t}-*/run_results.db"):
    c = sqlite3.connect(db); per = collections.defaultdict(collections.Counter)
    for d, cn, v in c.execute("select dispatch_id, counter_name, counter_value from pmc_events"): per[d][cn] += v
    x.update(per[max(per)])
rep = json.load(open(sys.argv[2].replace(".hsaco", ".json")))
json.dump({"x": dict(x), "vgpr": rep["target_resources"]["vector"]["final"]["register_count"], "occ": rep["target_resources"]["occupancy_percent"]}, open(f"wk-{t}.json", "w"))
P
