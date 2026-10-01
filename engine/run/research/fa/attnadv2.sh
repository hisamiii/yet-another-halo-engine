#!/usr/bin/env bash
# attnadv.sh: extended counters + ATT stall attribution for production attention at pp8192 (real layer-3 inputs)
D=/home/q/yah-scratch/attn8k; B=8192; h=${1:-/home/q/yah-hal-p61-8192/.emit_tmp/yah_attn_hip.hsaco}
TR=/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0; LH=/home/q/yet-another-halo-engine/engine/build/loomhip
GR=/home/q/yet-another-halo-engine/engine/run/gpu_run.sh; export ROCPROFILER_METRICS_PATH=/home/q/yah-scratch/rcmetrics
ARGS=(yah_attn_wmma $((B/32)) 12 256 3 f:$D/l3.aq f:$D/l3.agate f:$D/l3.ak16 f:$D/vt.bin z:$((B*6144*2)) z:$((B*24*4)))
cd /home/q/yah-scratch; rm -rf aa-*
i=0; for g in "SQ_BUSY_CYCLES SQ_WAVE_CYCLES SQ_INST_CYCLES_VALU SQ_INSTS_VALU" "SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE SQ_WAIT_BARRIER" "SQ_WAIT_INST_ANY SQ_WAIT_CNT_ANY SQ_INSTS_VALU_TRANS SQ_WAIT_INST_LDS" "SQC_ICACHE_MISSES SQC_ICACHE_HITS SQ_WAIT_IFETCH SQ_INSTS_SALU"; do
  i=$((i+1)); sleep 1; $GR aa-$i -- $TR/bin/rocprofv3 --pmc $g --kernel-include-regex 'yah_attn_wmma' -d aa-p$i -o run -- $LH $h "${ARGS[@]}" > aa-$i.log 2>&1; done
python3 - <<'P'
import sqlite3, glob, collections
tot = {}
for db in sorted(glob.glob("/home/q/yah-scratch/aa-p*/run_results.db")):
    c = sqlite3.connect(db); per = collections.defaultdict(collections.Counter)
    for d, cn, v in c.execute("select dispatch_id, counter_name, counter_value from pmc_events"): per[d][cn] += v
    tot.update(per[max(per)])
cyc = tot["SQ_BUSY_CYCLES"] / 20; wc = tot["SQ_WAVE_CYCLES"]; S = 80
T = 8192 // 16; wmma = 24 * 32 * T * (T + 1) / 2
print(f"attention pp8192 (one layer): {cyc/1e6:.2f} M cycles; WMMA floor {wmma*34/80/1e6:.2f} M -> {100*wmma*34/80/cyc:.0f}% SOL")
print(f"  per WMMA: VALU instr {tot['SQ_INSTS_VALU']/wmma - 1:.2f} (VALU busy cycles {tot['SQ_INST_CYCLES_VALU']/wmma - 1:.2f}), transcendental {tot['SQ_INSTS_VALU_TRANS']/wmma:.2f}, LDS {tot['SQ_INSTS_LDS']/wmma:.2f}, SALU {tot['SQ_INSTS_SALU']/wmma:.2f}")
print(f"  issue bound (34*WMMA + VALU cycles + 3*LDS) = {(34*wmma + tot['SQ_INST_CYCLES_VALU'] - wmma + 3*tot['SQ_INSTS_LDS'])/80/cyc*100:.0f}% of measured")
print(f"  LDS busy {100*tot['SQ_LDS_IDX_ACTIVE']/(cyc*20):.0f}%, bank conflicts {100*tot['SQ_LDS_BANK_CONFLICT']/max(tot['SQ_LDS_IDX_ACTIVE'],1):.1f}% of LDS-active")
print(f"  wave time: barrier {100*tot['SQ_WAIT_BARRIER']/wc:.0f}%, waitcnt {100*tot['SQ_WAIT_CNT_ANY']/wc:.0f}%, waiting to issue {100*tot['SQ_WAIT_INST_ANY']/wc:.0f}% (LDS issue {100*tot['SQ_WAIT_INST_LDS']/wc:.1f}%), ifetch {100*tot['SQ_WAIT_IFETCH']/wc:.2f}%; waves/SIMD {wc/cyc/S:.1f}")
print(f"  I-cache miss rate {100*tot['SQC_ICACHE_MISSES']/max(tot['SQC_ICACHE_HITS']+tot['SQC_ICACHE_MISSES'],1):.2f}%")
P
