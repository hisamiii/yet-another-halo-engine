#!/usr/bin/env bash
# Bit-exact check of tools/gen_deltanet_hip.py against HIP's production DeltaNet
# kernel (BatchedDeltaNetRowSplitKernel<float,16,2,false,false>) on the same
# inputs: output and final state, atol 0. Also runs a negative control (final
# state vs initial state) that must FAIL, so a vacuous pass cannot hide.
# usage: deltanet_vs_hip.sh [tokens=256] [workdir]
set -euo pipefail
B=${1:-256}
root="$(cd "$(dirname "$0")/../../../.." && pwd)"
W=${2:-$(mktemp -d)}
mkdir -p "$W"; cd "$W"
hipcc -std=c++20 -O3 -DENGINE_ENABLE_HIP=1 --offload-arch=gfx1151 -I"$root/engine/gpu/ported" -I"$root/engine" \
  "$root/engine/tests/deltanet_hip_ref.hip" -o ref -Wl,--unresolved-symbols=ignore-all 2>/dev/null
python3 - "$B" <<'PY'
import sys, numpy as np
B=int(sys.argv[1]); rng=np.random.default_rng(7); QKV,KH,H=10240,16,48
conv=(rng.standard_normal((B,QKV))*0.5).astype(np.float32)
q=conv[:, :KH*128].reshape(B,KH,128); k=conv[:, KH*128:2*KH*128].reshape(B,KH,128)
kq=np.zeros((B,KH,3),np.float32)
kq[...,0]=1/np.sqrt((k*k).sum(-1)+1e-6); kq[...,1]=(1/np.sqrt(128))/np.sqrt((q*q).sum(-1)+1e-6); kq[...,2]=(k*q).sum(-1)
ab=np.zeros((B,H,2),np.float32); ab[...,0]=rng.uniform(0.5,1,(B,H)); ab[...,1]=rng.uniform(0,1,(B,H))
st=(rng.standard_normal((H,128,128))*0.1).astype(np.float32)
for n,a in (('conv',conv),('kq',kq),('ab',ab),('state',st)):
    a.tofile(n+'.bin'); np.save(n+'.npy', a.reshape(-1))
PY
./ref "$B" >/dev/null
python3 -c "
import numpy as np
np.save('ref_out.npy', np.fromfile('ref_out.bin', np.float32)); np.save('ref_state.npy', np.fromfile('ref_state.bin', np.float32))"
python3 "$root/engine/gpu/loom/tools/gen_deltanet_hip.py" dh.loom >/dev/null
mkcheck() {  # mkcheck <expected-state npy> <out.loom>
cp dh.loom "$2"; cat >> "$2" <<EOC

check.case public @vs_hip {
  %conv = check.file.read.npy path("conv.npy") : tensor<$((B*10240))xf32>
  %kq = check.file.read.npy path("kq.npy") : tensor<$((B*48))xf32>
  %ab = check.file.read.npy path("ab.npy") : tensor<$((B*96))xf32>
  %state = check.file.read.npy path("state.npy") : tensor<786432xf32>
  %out = check.generate.fill value(0.0) : tensor<$((B*6144))xf32>
  kernel.launch @yah_deltanet(%conv, %kq, %ab, %state, %out) : (tensor<$((B*10240))xf32>, tensor<$((B*48))xf32>, tensor<$((B*96))xf32>, tensor<786432xf32>, tensor<$((B*6144))xf32>)
  %eo = check.file.read.npy path("ref_out.npy") : tensor<$((B*6144))xf32>
  %es = check.file.read.npy path("$1") : tensor<786432xf32>
  check.expect.close actual(%out) expected(%eo) atol(0.0) rtol(0.0) nan(same) : tensor<$((B*6144))xf32>
  check.expect.close actual(%state) expected(%es) atol(0.0) rtol(0.0) nan(same) : tensor<786432xf32>
  check.return
}
check.benchmark<@vs_hip> @vs_hip_bench
EOC
}
mkcheck ref_state.npy pos.loom; mkcheck state.npy neg.loom
set +u; set --; source "$root/engine/hrx-env.sh" >/dev/null; set -u
H=/home/q/hrx
for t in pos neg; do
  timeout 600 $H/build/cmake/loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom $t.loom --device=amdgpu --target=amdgpu:gfx1151 \
    --config=yah_deltanet.batch=$B --config=yah_deltanet.qkv_size=10240 --config=yah_deltanet.inner_size=6144 \
    --config=yah_deltanet.num_key_heads=16 --config=yah_deltanet.num_heads=48 --benchmark=@vs_hip_bench \
    --measure=dispatch_complete --batch-size=1 --iterations=1 --max-batches=1 --output=$t.json >/dev/null 2>&1 || true
done
python3 - <<'PY'
import json
f=lambda t: json.load(open(t+'.json'))['work_items'][0].get('correctness',{}).get('failed_sample_count')
pos, neg = f('pos'), f('neg')
print('bit-identical to HIP' if pos == 0 else 'MISMATCH vs HIP', '| negative control', 'fails as it must' if neg else 'DID NOT FAIL')
raise SystemExit(0 if pos == 0 and neg else 1)
PY
