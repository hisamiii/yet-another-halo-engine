#!/usr/bin/env bash
# run_rowstats.sh <hal set> <tokens> <ids file> <out .rs>: one prefill run writing compact row stats for gate2.py.
# The reference is an fp16-KV chunked set; a candidate is the same set emitted with another YAH_KV.
set_dir=$1; T=$2; ids=$3; out=$4; M=${MODEL:-$HOME/Downloads/qwen3.8-27b.mq4-xts}
# resumable: a run whose log already reports layers_ms is done
if grep -q layers_ms ${out%.rs}.log 2>/dev/null; then echo "$(basename $out): done earlier, skipped"; exit 0; fi
E=$(cd "$(dirname "$0")/../../.." && pwd)
cd $E; set +u; set --; source engine/hrx-env.sh >/dev/null
engine/run/gpu_run.sh kvq -- engine/build/loom_forward_pp $M $set_dir ${out%.rs} $T $ids \
  --rowstats $out --rowstats-from ${RS_FROM:-1024} --rowstats-stride ${RS_STRIDE:-8} > ${out%.rs}.log 2>&1
echo "$(basename $out): $(grep -E 'layers_ms|\*\*\*|rror' ${out%.rs}.log | tr '\n' ' ')"
