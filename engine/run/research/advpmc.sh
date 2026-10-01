#!/usr/bin/env bash
# advpmc.sh: extended counters on two GEMMs (IQ3_S kstore, Q4_K kres K=6144), 3 passes each, 1 s gaps
cd /home/q/yah-scratch/tools; LH=/home/q/yet-another-halo-engine/engine/build/loomhip
G=("SQ_BUSY_CYCLES SQ_WAVE_CYCLES SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE" "SQ_WAIT_INST_LDS SQ_WAIT_IFETCH SQ_INST_LEVEL_LDS SQ_INSTS_LDS" "SQC_ICACHE_MISSES SQC_ICACHE_HITS SQ_LDS_UNALIGNED_STALL SQ_WAIT_CNT_ANY")
run() { tag=$1 h=$2 S=$3 f=$4 w=$5 in=$6 gx=$7
  BUFS=$(LH_INPUT=$PWD/$in LH_RESID=$PWD/real_l4_resid_f32.bin python3 lhargs.py ${h%.hsaco}.loom $f $PWD/$w z:142606336) || exit 3
  GROUPS_="${G[0]}" KREGEX="$S\$" bash solpmc.sh adv$tag -- $LH $PWD/$h $S $gx 8 256 5 $BUFS > /dev/null 2>&1
  mv /home/q/yah-scratch/sol-adv$tag /home/q/yah-scratch/sol-adv${tag}a
  GROUPS_="${G[1]}" KREGEX="$S\$" bash solpmc.sh adv$tag -- $LH $PWD/$h $S $gx 8 256 5 $BUFS > /dev/null 2>&1
  mv /home/q/yah-scratch/sol-adv$tag /home/q/yah-scratch/sol-adv${tag}b
  GROUPS_="${G[2]}" KREGEX="$S\$" bash solpmc.sh adv$tag -- $LH $PWD/$h $S $gx 8 256 5 $BUFS > /dev/null 2>&1
  mv /home/q/yah-scratch/sol-adv$tag /home/q/yah-scratch/sol-adv${tag}c
  python3 - $tag <<'P'
import sqlite3, glob, collections, sys
t = sys.argv[1]; tot = {}
for db in sorted(glob.glob(f"/home/q/yah-scratch/sol-adv{t}[abc]/p*/run_results.db")):
    c = sqlite3.connect(db); per = collections.defaultdict(collections.Counter)
    for d, cn, v in c.execute("select dispatch_id, counter_name, counter_value from pmc_events"): per[d][cn] += v
    tot.update(per[max(per)])
cyc = tot["SQ_BUSY_CYCLES"] / 20; wc = tot["SQ_WAVE_CYCLES"]; S = 80
print(f"== {t}: {cyc/1e6:.2f} M cycles")
print(f"   LDS: {tot['SQ_INSTS_LDS']/1e6:.1f} M instr; bank-conflict cycles {tot['SQ_LDS_BANK_CONFLICT']/1e6:.2f} M = {100*tot['SQ_LDS_BANK_CONFLICT']/max(tot['SQ_LDS_IDX_ACTIVE'],1):.1f}% of LDS-active cycles; unaligned stall {tot['SQ_LDS_UNALIGNED_STALL']/1e6:.2f} M")
print(f"   LDS busy (IDX_ACTIVE) {100*tot['SQ_LDS_IDX_ACTIVE']/(cyc*20):.0f}% of per-instance cycles (20 instances) ; avg LDS in flight per instr-cycle {tot['SQ_INST_LEVEL_LDS']/max(tot['SQ_INSTS_LDS'],1):.1f} (latency proxy, cycles)")
print(f"   wave time: waiting on LDS issue {100*tot['SQ_WAIT_INST_LDS']/wc:.1f}%, waiting on counters (waitcnt) {100*tot['SQ_WAIT_CNT_ANY']/wc:.1f}%, ifetch {100*tot['SQ_WAIT_IFETCH']/wc:.2f}%")
print(f"   I-cache: hits {tot['SQC_ICACHE_HITS']/1e6:.2f} M, misses {tot['SQC_ICACHE_MISSES']/1e3:.1f} K ({100*tot['SQC_ICACHE_MISSES']/max(tot['SQC_ICACHE_HITS']+tot['SQC_ICACHE_MISSES'],1):.3f}%)")
P
}
run s3k tv_s42w.hsaco yah_ffn_gemm_iq3s iq3s real_iq3s_gate.bin real_l4_fn.bin 136
run q4r24 q4k24n.hsaco yah_ffn_gemm_q4k_kres q4k real_q4k_ssmout33.bin chain_S6144.bin 40
