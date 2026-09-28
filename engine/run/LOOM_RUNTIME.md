# Running the engine on Loom (no HIP)

The goal is to run the full forward pass on the Loom kernels through the HRX
native API and drop HIP. This file records the proven compile -> load ->
dispatch path and the remaining work.

## 1. Compile a Loom kernel to an HRX-loadable HAL executable

`iree-benchmark-loom` can emit the exact artifact `hrx_executable_load_*`
wants, next to the run it already does:

```
source engine/hrx-env.sh
cd engine/gpu/loom
/home/q/hrx/build/cmake/loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom \
  yah_residual_add_1d_f32.loom --device=amdgpu \
  --case=yah_residual_1d_case --benchmark=yah_residual_1d_bench \
  --config=yah_residual_1d.dim=8 \
  --iterations=1 --warmup-iterations=0 --min-time-ms=0 --max-batches=1 \
  --artifact-bundle-dir=/tmp/bundle --artifact-bundle-policy=full
```

This writes `/tmp/bundle/hal_executables/<run>_c0_hal_executable.hal` (an IREE
HAL executable ELF), plus the target ELF and AMDGPU assembly listing. The
`--config` binding must be supplied because `iree-run-loom` has no `--config`
flag; `iree-benchmark-loom` does, and its artifact bundle is the emission path.

## 2. Load and dispatch from C++ through HRX

`engine/run/loom_probe.cc` is the proof, with no HIP anywhere:

```
hrx_gpu_initialize(0);
hrx_gpu_device_get(0, &device);
hrx_stream_create(device, 0, &stream);
hrx_executable_load_file(device, path, "amdgpu", "gfx1151", &executable);
hrx_executable_lookup_export_by_name(executable, "yah_residual_1d", &ordinal);
hrx_buffer_allocate(stream, bytes, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                    HRX_BUFFER_USAGE_DEFAULT, &buffer);
hrx_synchronous_h2d(device, host, buffer, 0, bytes);
hrx_dispatch_config_t config = {.workgroup_count = {1,1,1},
                              .workgroup_size = {256,1,1}, .subgroup_size = 32};
hrx_buffer_ref_t bindings[3] = {{a,0,bytes},{b,0,bytes},{out,0,bytes}};
hrx_stream_dispatch(stream, executable, ordinal, &config, nullptr, 0, bindings,
                    3, HRX_DISPATCH_FLAG_NONE);
hrx_stream_synchronize(stream);
hrx_synchronous_d2h(device, out, 0, host_out, bytes);
hrx_gpu_shutdown();
```

The device advertises two AMDGPU executable targets: `gfx1151` (exact, kind 0)
and `gfx11-generic` (kind 1). Pass the key the artifact was built for; the
benchmark bundle is compiled for the device, i.e. `gfx1151`.

Build (no HIP, no hipcc):

```
g++ -std=c++20 -O2 -I/home/q/hrx/libhrx/include engine/run/loom_probe.cc \
  -o /tmp/loom_probe -L/home/q/hrx/build/cmake/libhrx/src/libhrx -lhrx
source engine/hrx-env.sh && /tmp/loom_probe <bundle>/hal_executables/*.hal gfx1151
```

Verified: `exports: 1 [0] name=yah_residual_1d bindings=3 params=3 consts=0`
and `LOOM PROBE PASS: out = 100 + 2*i for 8 elements`.

## 3. Tooling and runtime layer

`engine/gpu/loom/emit_hal.py <file.loom> <outdir> <symbol=value> ...` is the
production emission path. `iree-run-loom` has `--emit-only`/
`--emit-hal-executable` but no `--config`; `iree-benchmark-loom` has `--config`
but needs a matching `check.case` and dispatches. Neither emits a production
shape whose case bindings are small. So `emit_hal.py` rewrites each
`config.get @symbol` to a constant, drops the matching `config.decl`, and runs
`iree-run-loom --emit-only`, which emits with no dispatch at all:

```
python3 engine/gpu/loom/emit_hal.py engine/gpu/loom/yah_gemv_q6k_f32.loom \
  /tmp/emit_q6k_new yah_gemv_q6k.m_rows=17408 yah_gemv_q6k.k_blocks=20
# -> /tmp/emit_q6k_new/yah_gemv_q6k_f32.hal
```

`engine/gpu/loom/emit_hal.sh` is the older case-backed variant (artifact bundle
via the benchmark tool); it is only usable for configs the file already has a
matching case for.

- `engine/gpu/loom/emit_hal.sh <file.loom> <case> <bench> <outdir> [config=value ...]`
  wraps the benchmark artifact bundle and prints the `.hal` path.
- `engine/model/loom_runtime.hpp` is the engine-facing HRX layer (no HIP):
  `LoomDevice` (initialize/device/stream/load/allocate/copy/dispatch/sync),
  `LoomExecutable` (export name -> ordinal plus binding/parameter/constant
  metadata), `LoomBuffer`, and `LoomDevice::Config(...)`.
  `engine/run/loom_probe.cc` is now written against it and still passes.

## 4. Real-weight verification (GGUF mmap -> HRX -> Loom)

`LoomDevice::Import` wraps `hrx_allocator_import_buffer`; the GGUF tensor-data
mapping is imported once and each tensor is a `(offset, length)` binding into
it, exactly like the HIP path rebases by a pointer delta. `LoomBuffer` usage
must be `HRX_BUFFER_USAGE_DEFAULT` (the dispatch binding validation rejects a
STORAGE_READ-only import).

`engine/run/loom_gemv_probe.cc` opens the shard, imports a page-aligned window
around `blk.64.ffn_gate.weight` (17408x5120, Q6_K), dispatches
`yah_gemv_q6k` (m_rows=17408, k_blocks=20) through HRX with no HIP, and dumps
`y`; `engine/run/verify_loom_gemv.py` decodes the same tensor in float64 and
compares. Result: `max_abs 2.2e-07, max_rel 1.8e-06, PASS`.

```
engine/gpu/loom/emit_hal.sh yah_gemv_q6k_f32.loom yah_gemv_q6k_full_case \
  yah_gemv_q6k_full_bench /tmp/emit_q6k yah_gemv_q6k.m_rows=17408 \
  yah_gemv_q6k.k_blocks=20
g++ -std=c++20 -O2 -Iengine -I/home/q/hrx/libhrx/include \
  engine/run/loom_gemv_probe.cc -o /tmp/loom_gemv_probe \
  engine/build/libyah_core.a -L/home/q/hrx/build/cmake/libhrx/src/libhrx \
  -lhrx -licuuc -lpthread
/tmp/loom_gemv_probe <model.gguf> <hal> blk.64.ffn_gate.weight /tmp/loom_y.bin
python3 engine/run/verify_loom_gemv.py <model.gguf> blk.64.ffn_gate.weight \
  15168895904 /tmp/loom_y.bin 16
```

`engine/run/loom_embed_probe.cc` runs the prefill embedding over the real Q3_K
`token_embd.weight` (248320x5120, row_bytes 2200) for a prompt, importing the
table and dispatching `yah_prefill_embed_q3k`; `engine/run/verify_loom_embed.py`
decodes the selected rows in float64 and compares. Result for prompt
`760 6511 314 9338 369`: `max_abs 0.0` on every token (bit-exact), PASS.

`engine/run/loom_norm_probe.cc` runs the fused RMSNorm+residual prefill kernel
(`yah_half_norm`, fp16 out) over the real F32 `blk.0.attn_norm.weight`
(dim 5120, file_offset 1621892000) using the hidden states emitted by the
embedding probe; `engine/run/verify_loom_norm.py` rounds a float32 oracle to
fp16 before comparing because the kernel output is fp16. Because an fp16
result can legitimately land one ulp away when the reduction order crosses a
rounding boundary, the verifier requires every element within one fp16 ulp of
the rounded oracle. Result: `f16 mismatches 0`, `elements beyond one f16 ulp 0`,
`max_abs 0.0` (bit-exact), PASS.

```
python3 engine/gpu/loom/emit_hal.py engine/gpu/loom/yah_half_norm_f16.loom /tmp/emit_norm
g++ -std=c++20 -O2 -Iengine -I/home/q/hrx/libhrx/include engine/run/loom_norm_probe.cc -o /tmp/loom_norm_probe engine/build/libyah_core.a -L/home/q/hrx/build/cmake/libhrx/src/libhrx -lhrx -licuuc -lpthread
source engine/hrx-env.sh
/tmp/loom_norm_probe <model.gguf> /tmp/emit_norm/yah_half_norm_f16.hal /tmp/loom_hidden.bin /tmp/loom_normed.f16 blk.0.attn_norm.weight 5
python3 engine/run/verify_loom_norm.py <model.gguf> 1621892000 /tmp/loom_hidden.bin /tmp/loom_normed.f16 5 5120 1e-6
```

`engine/run/loom_gemm_probe.cc` runs the format-faithful Q4_K prefill GEMM
(`yah_ffn_gemm_q4k`, kStore) over the real `blk.4.ffn_down.weight` (N=5120,
K=17408) imported from the GGUF mmap, with the activation tile generated
deterministically; `engine/run/verify_loom_gemm.py` decodes the Q4_K weight in
float64 (rounding it to f16 exactly as the kernel stages it) and compares. The
port is now shape-generic (`k_blocks`, `token_tiles`), so the same HAL serves any
K and any multiple-of-64 token batch. Results: 5 tokens, one tile `max_abs
1.76e-05`; 100 tokens, two tiles `max_abs 2.72e-05`; both PASS.

```
python3 engine/gpu/loom/emit_hal.py engine/gpu/loom/yah_ffn_gemm_q4k_f32.loom /tmp/emit_q4k_down yah_ffn_gemm_q4k.m_tiles=320 yah_ffn_gemm_q4k.k_blocks=68 yah_ffn_gemm_q4k.token_tiles=1
g++ -std=c++20 -O2 -Iengine -I/home/q/hrx/libhrx/include engine/run/loom_gemm_probe.cc -o /tmp/loom_gemm_probe engine/build/libyah_core.a -L/home/q/hrx/build/cmake/libhrx/src/libhrx -lhrx -licuuc -lpthread
source engine/hrx-env.sh
/tmp/loom_gemm_probe <model.gguf> /tmp/emit_q4k_down/yah_ffn_gemm_q4k_f32.hal blk.4.ffn_down.weight 5 /tmp/loom_gemm_y.bin
python3 engine/run/verify_loom_gemm.py <model.gguf> 2444240416 /tmp/loom_gemm_y.bin 5 5120 17408 0 1 2 3 100 3000 5119
```

`engine/run/loom_head_probe.cc` runs the graph tail entirely on Loom through
HRX: the output RMSNorm (`yah_rmsnorm`, eps 1e-6) on the final residual row, the
Q4_K output GEMV (`yah_gemv_q4k`, K=5120, vocab 248320) on the real
`output.weight`, and `yah_argmax`. The input is the final residual state dumped
by the HIP engine via `YAH_DUMP_DIR` on the all-Q4_K `base_q4kpure.gguf` for the
prompt `760 6511 314 9338 369`. Result: `argmax=11751`, identical to the HIP
`yah-run` reference. This proves the Q4_K output projection, the norm and the
argmax on real weights, plus multi-HAL load and chained dispatch through HRX.

```
python3 engine/gpu/loom/emit_hal.py engine/gpu/loom/yah_rmsnorm_f32.loom /tmp/emit_head yah_rmsnorm.rows=1 yah_rmsnorm.eps=1e-6
python3 engine/gpu/loom/emit_hal.py engine/gpu/loom/yah_gemv_q4k_f32.loom /tmp/emit_head yah_gemv_q4k.m_rows=248320 yah_gemv_q4k.k_blocks=20
python3 engine/gpu/loom/emit_hal.py engine/gpu/loom/yah_argmax_f32.loom /tmp/emit_head yah_argmax.vocab=248320
g++ -std=c++20 -O2 -Iengine -I/home/q/hrx/libhrx/include engine/run/loom_head_probe.cc -o /tmp/loom_head_probe engine/build/libyah_core.a -L/home/q/hrx/build/cmake/libhrx/src/libhrx -lhrx -licuuc -lpthread
source engine/hrx-env.sh
/tmp/loom_head_probe <model.gguf> /tmp/emit_head/yah_rmsnorm_f32.hal /tmp/emit_head/yah_gemv_q4k_f32.hal /tmp/emit_head/yah_argmax_f32.hal /tmp/hipdump/layer_63.bin 5 /tmp/loom_logits.bin
```

## 5. Remaining work

1. Emit HAL executables for every ported kernel at its production shape
   (a build step; `iree-benchmark-loom` needs a case+benchmark per compile).
2. An HRX-native tensor/weights layer: register the GGUF mmap as an imported
   HRX buffer (or allocate and copy), device buffers for activations and the
   recurrent/KV state.
3. Replace the HIP `Forward` layer (`engine/model/forward.hip`) and the
   `Launch*` calls with HRX dispatches of the Loom kernels; the dispatch
   constants/bindings layout comes from each kernel export metadata.
4. Validate prefill and decode against the recorded HIP argmax and timings,
   then remove the HIP build (`engine/build_gpu.sh`, hipcc) and the HIP
   sources from the engine.