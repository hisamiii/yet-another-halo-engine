#!/usr/bin/env bash
# run_codec.sh <hal set> <tokens> <ids file> <out .rs> <kcodec> <vcodec> [extra env...]:
# one engine run with the KV codec hook (fp16/fp16 = reference) writing compact row stats
set_dir=$1; T=$2; ids=$3; out=$4; kc=$5; vc=$6; shift 6
extra=("$@")   # hrx-env.sh is sourced with empty args below
E=/home/q/yet-another-halo-engine
cd $E; set +u; set --; source engine/hrx-env.sh >/dev/null
hook=""
[ "$kc/$vc" != "fp16/fp16" ] && hook="python3 $E/engine/run/kvq/kvcodec.py --k $kc --v $vc"
env ${hook:+YAH_KV_HOOK="$hook"} YAH_ROWSTATS=$out YAH_ROWSTATS_FROM=${RS_FROM:-1024} YAH_ROWSTATS_STRIDE=${RS_STRIDE:-8} "${extra[@]}" \
  engine/run/gpu_run.sh kvq -- engine/build/loom_forward_pp ~/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf $set_dir ${out%.rs} $T $ids > ${out%.rs}.log 2>&1
echo "$(basename $out): $(grep -E 'layers_ms|\*\*\*|rror' ${out%.rs}.log | tr '\n' ' ')"
