#!/usr/bin/env bash
# solpmc.sh <tag> -- <cmd...>: speed-of-light counter passes (extended gfx1151 counter set via ROCPROFILER_METRICS_PATH),
# one pass per group, 15 s gaps; KREGEX selects the kernel. Results: sol-<tag>/pN/run_results.db
tag=$1; shift; [ "$1" = "--" ] && shift
TR=/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0; GR=/home/q/yet-another-halo-engine/engine/run/gpu_run.sh
export ROCPROFILER_METRICS_PATH=/home/q/yah-scratch/rcmetrics
cd /home/q/yah-scratch; rm -rf sol-$tag; i=0
for g in ${GROUPS_:-"SQ_BUSY_CYCLES SQ_WAVE_CYCLES SQ_INST_CYCLES_VALU SQ_INSTS_VALU" "SQ_INST_CYCLES_LDS SQ_INSTS_LDS SQ_WAIT_BARRIER SQ_WAIT_INST_ANY" "SQ_WAIT_ANY SQ_INSTS_SALU SQ_INST_LEVEL_LDS SQ_WAVES"}; do
  i=$((i+1)); sleep ${GAP:-15}
  $GR sol-$tag-$i -- $TR/bin/rocprofv3 --pmc $g --kernel-include-regex "${KREGEX:-.}" -d /home/q/yah-scratch/sol-$tag/p$i -o run -- "$@" > sol-$tag-$i.log 2>&1
  grep -E 'gpu_run: \*\*\*|error|Error' sol-$tag-$i.log | head -2
done
python3 - sol-$tag <<'P'
import sqlite3, glob, sys, collections, statistics
tot = collections.defaultdict(list)
for db in sorted(glob.glob(sys.argv[1] + "/p*/run_results.db")):
    c = sqlite3.connect(db); per = collections.defaultdict(lambda: collections.Counter())
    for d, cn, v in c.execute("select dispatch_id, counter_name, counter_value from pmc_events"):
        per[d][cn] += v
    last = max(per)
    for k, v in per[last].items(): tot[k].append(v)
for k in sorted(tot): print(f"{k:24s} {tot[k][-1]:.4g}")
P
