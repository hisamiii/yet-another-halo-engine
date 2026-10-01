#!/usr/bin/env bash
# Bit-exact check of tools/gen_attn_hip.py (yah_transpose_v16 + yah_attn_wmma)
# against HIP's production prefill attention (PackAttentionHeads +
# WmmaCausalAttention<32, 16, true>) on the same random q, gate and f16 K/V:
# every output element, atol 0. A negative control binds the token-major V where
# the kernel expects V^T and must FAIL, so a vacuous pass cannot hide.
# usage: attn_vs_hip.sh [tokens=200] [workdir]   (200: ragged, not a multiple of 16 or 32)
set -euo pipefail
export YAH_ATTN_F16OUT=0  # the harness compares f32 outputs
B=${1:-200}
root="$(cd "$(dirname "$0")/../../../.." && pwd)"
W=${2:-$(mktemp -d)}
mkdir -p "$W"; cd "$W"
hipcc -std=c++20 -O3 -DENGINE_ENABLE_HIP=1 --offload-arch=gfx1151 -I"$root/engine/gpu/ported" -I"$root/engine" \
  "$root/engine/tests/attn_hip_ref.hip" -o ref -Wl,--unresolved-symbols=ignore-all 2>/dev/null
python3 - "$B" <<'PY'
import sys, numpy as np
B=int(sys.argv[1]); rng=np.random.default_rng(7)
d={'q':rng.standard_normal(B*6144).astype(np.float32)*1.5,'gate':rng.standard_normal(B*6144).astype(np.float32)*1.5,
   'k16':(rng.standard_normal(B*1024)*1.5).astype(np.float16),'v16':(rng.standard_normal(B*1024)*1.5).astype(np.float16)}
for n,a in d.items(): a.tofile(n+'.bin'); np.save(n+'.npy',a)
PY
./ref "$B" >/dev/null
python3 -c "import numpy as np; np.save('ref_out.npy', np.fromfile('ref_out.bin', np.float32))"
python3 "$root/engine/gpu/loom/tools/gen_attn_hip.py" ah.loom >/dev/null
python3 "$root/engine/gpu/loom/tools/gen_attn_hip.py" vtrans vt.loom >/dev/null
P=$(( (B + 15) / 16 * 16 ))
mkcheck() {  # mkcheck <value_cache operand> <out.loom>
cat ah.loom vt.loom > "$2"; cat >> "$2" <<EOC

check.case public @vs_hip {
  %lse = check.generate.fill value(0.0) : tensor<$((B*24))xf32>
  %q = check.file.read.npy path("q.npy") : tensor<$((B*6144))xf32>
  %gate = check.file.read.npy path("gate.npy") : tensor<$((B*6144))xf32>
  %k16 = check.file.read.npy path("k16.npy") : tensor<$((B*1024))xf16>
  %v16 = check.file.read.npy path("v16.npy") : tensor<$((P*1024))xf16>
  %out = check.generate.fill value(0.0) : tensor<$((B*6144))xf32>
  %vt = check.generate.fill value(7.0) : tensor<$((P*1024))xf16>
  kernel.launch @yah_transpose_v16(%v16, %vt) : (tensor<$((P*1024))xf16>, tensor<$((P*1024))xf16>)
  kernel.launch @yah_attn_wmma(%q, %gate, %k16, $1, %out, %lse) : (tensor<$((B*6144))xf32>, tensor<$((B*6144))xf32>, tensor<$((B*1024))xf16>, tensor<$((P*1024))xf16>, tensor<$((B*6144))xf32>, tensor<$((B*24))xf32>)
  %expected = check.file.read.npy path("ref_out.npy") : tensor<$((B*6144))xf32>
  check.expect.close actual(%out) expected(%expected) atol(0.0) rtol(0.0) nan(same) : tensor<$((B*6144))xf32>
  check.return
}
check.benchmark<@vs_hip> @vs_hip_bench
EOC
}
# v16.npy is read as P*1024 halfs: pad it so both operands have the V^T size
python3 -c "
import numpy as np; v=np.load('v16.npy'); P=$P; np.save('v16.npy', np.concatenate([v, np.zeros(P*1024-v.size, v.dtype)]))"
mkcheck %vt pos.loom; mkcheck %v16 neg.loom
set +u; set --; source "$root/engine/hrx-env.sh" >/dev/null; set -u
H=/home/q/hrx
for t in pos neg; do
  timeout 900 $H/build/cmake/loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom $t.loom --device=amdgpu --target=amdgpu:gfx1151 \
    --config=attention_prefill.cache_capacity=$B --config=attention_prefill.token_count=$B --config=attention_prefill.start_pos=0 \
    --config=attention_prefill.num_heads=24 --config=attention_prefill.num_kv_heads=4 --config=attention_prefill.head_dim=256 \
    --config=attention_prefill.gqa=6 --config=yah_vtrans.token_count=$B --config=yah_vtrans.cache_capacity=$B \
    --benchmark=@vs_hip_bench --measure=dispatch_complete --batch-size=1 --iterations=1 --max-batches=1 --output=$t.json >/dev/null 2>&1 || true
done
python3 - <<'PY'
import json, sys
def result(p):  # (samples run, samples failing the comparison)
    w = (json.load(open(p)).get('work_items') or [{}])[0]
    c = w.get('correctness') or {}
    return c.get('sample_count', 0), c.get('failed_sample_count', 0)
pos, neg = result('pos.json'), result('neg.json')
ok = pos == (1, 0) and neg == (1, 1)  # the control must run AND mismatch
print(f"attn_vs_hip: vs HIP {'exact' if pos == (1, 0) else 'MISMATCH %s' % (pos,)}, "
      f"negative control {'fails as it must' if neg == (1, 1) else 'DID NOT FAIL %s' % (neg,)}")
sys.exit(0 if ok else 1)
PY
