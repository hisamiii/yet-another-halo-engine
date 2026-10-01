#!/usr/bin/env bash
# gate_run.sh <hal set> <outdir> [window ids...]: Loom pp2048 on frozen corpus windows (corpus/ids2048_wNN.txt) with
# all-position logits for positions FROM..2047 (YAH_LOGITS_FROM, default 1536) -> <outdir>/wNN.{all_logits,logits,hidden}.
# Correctness runs: no timing, no gaps.
set_dir=$1; out=$2; shift 2; FROM=${FROM:-1536}; C=${CORPUS:-/home/q/yet-another-halo-engine/engine/run/gate/corpus}
mkdir -p $out; E=/home/q/yet-another-halo-engine
cd $E; set +u; set --; source engine/hrx-env.sh >/dev/null
# Long windows (L0 = corpus/ids8192_w0.txt) run 8192 tokens on SET8K (an
# 8192-token HAL set) and keep positions FROM_L..8191 (default 7680).
for w in ${WINDOWS:-00 01 02 03}; do
  if [ "${w:0:1}" = L ]; then
    YAH_LOGITS_FROM=${FROM_L:-7680} engine/run/gpu_run.sh gate-w$w -- engine/build/loom_forward_pp ~/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf ${SET8K:?SET8K: 8192-token HAL set} $out/w$w 8192 $C/ids8192_w${w:1}.txt > $out/w$w.log 2>&1
  else
    YAH_LOGITS_FROM=$FROM engine/run/gpu_run.sh gate-w$w -- engine/build/loom_forward_pp ~/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf $set_dir $out/w$w 2048 $C/ids2048_w$w.txt > $out/w$w.log 2>&1
  fi
  echo "w$w: $(grep -E 'argmax|all_logits|gpu_run: \*\*\*' $out/w$w.log | tr '\n' ' ')"
done
