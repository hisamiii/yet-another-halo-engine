#!/usr/bin/env bash
# gate_hip.sh <outdir> [windows]: HIP yah-run on frozen corpus windows, all-position logits, rows FROM..2047 kept as wNN.all_logits
out=$1; FROM=${FROM:-1536}; C=${CORPUS:-/home/q/yet-another-halo-engine/engine/run/gate/corpus}; E=/home/q/yet-another-halo-engine; mkdir -p $out
for w in ${WINDOWS:-00 01 02 03}; do
  $E/engine/run/gpu_run.sh gatehip-w$w -- $E/engine/build/yah-run ~/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf --ids-file $C/ids2048_w$w.txt --dump-all-logits $out/w$w.full > $out/w$w.log 2>&1
  python3 -c "
import numpy as np; V=248320
a=np.memmap('$out/w$w.full',dtype=np.float32,mode='r').reshape(-1,V); print('w$w rows', a.shape[0]); np.asarray(a[$FROM:]).tofile('$out/w$w.all_logits')"
  rm -f $out/w$w.full
  echo "w$w: $(grep -E 'argmax|gpu_run: \*\*\*' $out/w$w.log | tr '\n' ' ')"
done
