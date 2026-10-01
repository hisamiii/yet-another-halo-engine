#!/usr/bin/env bash
# cnt8kx.sh <set> <tag>: pp8192 with dispatch sequence under HRX counters (SQ_BUSY_CYCLES, GRBM-free), 15 s gap
S=/home/q/yah-scratch; E=/home/q/yet-another-halo-engine; set_dir=$1; tag=$2
sleep ${GAP:-30}
( cd $E; set +u; set --; source engine/hrx-env.sh >/dev/null; rm -f $S/c8-$tag.irpf
  YAH_LOOM_SEQ=$S/c8-$tag.seq HRX_PROFILE_FILE=$S/c8-$tag.irpf HRX_PROFILE_MODE=counters HRX_PROFILE_COUNTERS=SQ_BUSY_CYCLES,SQ_WAVES timeout 1200 engine/run/gpu_run.sh c8-$tag -- \
    engine/build/loom_forward_pp ~/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf $set_dir $S/c8-$tag 8192 $S/ids8192.txt > $S/c8-$tag.log 2>&1
  echo "$tag: $(grep -E 'argmax|layers_ms' $S/c8-$tag.log | tr '\n' ' ')"
  /home/q/hrx/build/cmake/runtime/src/iree/tools/iree-profile/iree-profile counter --format=jsonl $S/c8-$tag.irpf > $S/c8-$tag.jsonl )
