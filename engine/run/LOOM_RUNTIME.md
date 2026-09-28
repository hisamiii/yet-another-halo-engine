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

## 3. Remaining work

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