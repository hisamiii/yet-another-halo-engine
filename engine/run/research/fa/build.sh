#!/usr/bin/env bash
# build.sh <tag> [env...]: gen_attn_fa at pp8192 -> fa/<tag>.hsaco, VGPRs/spills/LDS
t=$1; shift; B=8192; L=/home/q/yet-another-halo-engine/engine/gpu/loom/tools
LC=/home/q/hrx/build/cmake/loom/src/loom/tools/loom-compile/loom-compile
cd /home/q/yah-scratch/fa
env YAH_ATTN_MAX_TOKENS=$B "$@" python3 ${GEN:-$L/gen_attn_fa.py} $t.loom > /dev/null || exit 1
$LC $t.loom --root=@yah_attn_wmma --target=amdgpu:gfx1151 --format=amdgpu-hsaco --output=$t.hsaco \
  --config=attention_prefill.cache_capacity=$B --config=attention_prefill.token_count=$B --config=attention_prefill.start_pos=0 \
  --config=attention_prefill.num_heads=24 --config=attention_prefill.num_kv_heads=4 --config=attention_prefill.head_dim=256 \
  --config=attention_prefill.gqa=6 --compile-report=summary --compile-report-output=$t.json 2>&1 | grep -E "error" | head -5
[ -f $t.hsaco ] && python3 -c "import json; d=json.load(open('$t.json')); print('$t: vgpr', d['target_resources']['vector']['final']['register_count'], 'spills', d['allocation']['materialized_spill_store_count'])"
