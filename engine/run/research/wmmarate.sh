#!/usr/bin/env bash
# wmmarate.sh: one PMC round (15 s gap) per variant; cycles per WMMA per SIMD from SQ_BUSY_CYCLES (last dispatch)
cd /home/q/yah-scratch/research; TR=/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0; GR=/home/q/yet-another-halo-engine/engine/run/gpu_run.sh
IT=${IT:-4096}
for v in ${VARIANTS:-f16 bf16 iu8 iu4 f16v1 f16v2 f16v4 f16v8 f16v16 f16v32}; do
  rm -rf wr-$v; sleep 15
  $GR wr-$v -- $TR/bin/rocprofv3 --pmc SQ_BUSY_CYCLES SQ_WAVES --kernel-include-regex 'wmma_' -d wr-$v -o run -- ./wmmarate $v $IT > wr-$v.log 2>&1
  WR_BLOCKS=${WR_BLOCKS:-160} WR_THREADS=${WR_THREADS:-256} python3 - wr-$v/run_results.db $v $IT <<'P'
import sqlite3, sys, collections
c = sqlite3.connect(sys.argv[1]); per = collections.defaultdict(float); n = collections.Counter()
for d, cn, v in c.execute("select dispatch_id, counter_name, counter_value from pmc_events where counter_name='SQ_BUSY_CYCLES'"):
    per[d] += v; n[d] += 1
d = max(per); cyc = per[d] / n[d]
import os; wmma = int(os.environ.get("WR_BLOCKS","160")) * int(os.environ.get("WR_THREADS","256"))//32 * int(sys.argv[3]) * 8
print(f"{sys.argv[2]:7s} {cyc/1e6:8.2f} M cycles  {cyc / (wmma / 80):6.2f} cycles per WMMA per SIMD")
P
done
