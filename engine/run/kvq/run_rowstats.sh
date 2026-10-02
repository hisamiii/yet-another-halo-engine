#!/usr/bin/env bash
# run_rowstats.sh <hal set> <tokens> <ids file> <out .rs>: one prefill run writing compact row stats for gate2.py.
# The reference is an fp16-KV chunked set; a candidate is the same set emitted with another YAH_KV.
set_dir=$1; T=$2; ids=$3; out=$4
# resumable: a run whose log already reports layers_ms is done
if grep -q layers_ms ${out%.rs}.log 2>/dev/null; then echo "$(basename $out): done earlier, skipped"; exit 0; fi
E=$(cd "$(dirname "$0")/../../.." && pwd)
cd $E; set +u; set --; source engine/hrx-env.sh >/dev/null
YAH_ROWSTATS=$out YAH_ROWSTATS_FROM=${RS_FROM:-1024} YAH_ROWSTATS_STRIDE=${RS_STRIDE:-8} \
  engine/run/gpu_run.sh kvq -- engine/build/loom_forward_pp ~/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf $set_dir ${out%.rs} $T $ids > ${out%.rs}.log 2>&1
echo "$(basename $out): $(grep -E 'layers_ms|\*\*\*|rror' ${out%.rs}.log | tr '\n' ' ')"
