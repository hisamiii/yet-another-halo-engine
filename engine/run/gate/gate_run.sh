#!/usr/bin/env bash
# gate_run.sh <hal set> <outdir> [window ids...]: Loom pp2048 on frozen corpus windows (corpus/ids2048_wNN.txt) with
# all-position logits for positions FROM..2047 (YAH_LOGITS_FROM, default 1536) -> <outdir>/wNN.{all_logits,logits,hidden}.
# Correctness runs: no timing, no gaps.
set_dir=$1; out=$2; shift 2; FROM=${FROM:-1536}; C=${CORPUS:-/home/q/yet-another-halo-engine/engine/run/gate/corpus}
mkdir -p $out; E=/home/q/yet-another-halo-engine
cd $E; set +u; set --; source engine/hrx-env.sh >/dev/null
for w in ${WINDOWS:-00 01 02 03}; do
  YAH_LOGITS_FROM=$FROM engine/run/gpu_run.sh gate-w$w -- engine/build/loom_forward_pp ~/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf $set_dir $out/w$w 2048 $C/ids2048_w$w.txt > $out/w$w.log 2>&1
  echo "w$w: $(grep -E 'argmax|all_logits|gpu_run: \*\*\*' $out/w$w.log | tr '\n' ' ')"
done
