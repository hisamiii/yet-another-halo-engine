#!/usr/bin/env bash
# cyc.sh <tag>...: SQ_BUSY_CYCLES of fa/<tag>.hsaco (f16 out) on real pp8192 inputs, 1 s gap, one round each
D=/home/q/yah-scratch/attn8k; B=8192; LH=/home/q/yet-another-halo-engine/engine/build/loomhip; GR=/home/q/yet-another-halo-engine/engine/run/gpu_run.sh
TR=/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0; cd /home/q/yah-scratch/fa
for t in "$@"; do rm -rf ac-$t; sleep 1
  $GR facyc -- $TR/bin/rocprofv3 --pmc SQ_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_LDS --kernel-include-regex yah_attn_wmma -d ac-$t -o run -- $LH $t.hsaco yah_attn_wmma $((B/32)) 12 256 3 f:$D/l3.aq f:$D/l3.agate f:$D/l3.ak16 f:$D/vt.bin z:$((B*6144*2)) z:$((B*24*4)) > ac-$t.log 2>&1
  python3 - ac-$t/run_results.db $t <<'P'
import sqlite3, collections, sys
c = sqlite3.connect(sys.argv[1]); per = collections.defaultdict(collections.Counter)
for d, cn, v in c.execute("select dispatch_id, counter_name, counter_value from pmc_events"): per[d][cn] += v
x = per[max(per)]; T = 512; w = 24 * 32 * T * (T + 1) / 2
print(f"{sys.argv[2]:10s} {x['SQ_BUSY_CYCLES']/20/1e6:7.2f} M cycles  ({100*w*34/80/(x['SQ_BUSY_CYCLES']/20):.0f}% SOL)  VALU/WMMA {x['SQ_INSTS_VALU']/w-1:.2f}  LDS/WMMA {x['SQ_INSTS_LDS']/w:.2f}")
P
done
