#!/usr/bin/env bash
# cyc.sh <tag>: SQ_BUSY_CYCLES / VALU / LDS of <tag>.hsaco (yah_deltanet ABI) on layer-0 pp2048 inputs
G=/home/q/yah-scratch/gdn; LH=/home/q/yet-another-halo-engine/engine/build/loomhip; GR=/home/q/yet-another-halo-engine/engine/run/gpu_run.sh
TR=/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0; cd $G/k; t=$1; rm -rf pc-$t; sleep 1
$GR gdncyc -- $TR/bin/rocprofv3 --pmc SQ_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_LDS SQ_WAVES --kernel-include-regex yah_deltanet -d pc-$t -o run -- $LH $t.hsaco yah_deltanet 2 48 256 3 f:$G/l0.conv f:$G/l0.kq f:$G/l0.ab z:$((48*16384*4)) z:$((2048*6144*4)) > pc-$t.log 2>&1
python3 - pc-$t/run_results.db $t <<'P'
import sqlite3, collections, sys
c = sqlite3.connect(sys.argv[1]); per = collections.defaultdict(collections.Counter)
for d, cn, v in c.execute("select dispatch_id, counter_name, counter_value from pmc_events"): per[d][cn] += v
x = per[max(per)]
print(f"{sys.argv[2]:8s} {x['SQ_BUSY_CYCLES']/20/1e6:7.3f} M cycles  VALU {x['SQ_INSTS_VALU']/1e6:7.2f} M  LDS {x['SQ_INSTS_LDS']/1e6:6.2f} M  waves {x['SQ_WAVES']:.0f}")
P
