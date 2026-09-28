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

`engine/run/loom_ssm_layer_probe.cc` runs one Gated DeltaNet layer end to end on
Loom through HRX: `yah_half_norm`, four Q4_K kStore projections (attn_qkv,
attn_gate, ssm_alpha, ssm_beta), `yah_ssm_conv`, `yah_deltanet_prep_kq`,
`yah_deltanet_prep_ab`, `yah_deltanet_rowsplit`, `yah_ssm_postnorm_fp16` and the
Q4_K kResidual output projection. `YAH_DUMP_DIR` in the HIP engine now also
writes the SSM stage buffers (`ssm_qkv`, `ssm_gate`, `ssm_alpha`, `ssm_beta`,
`ssm_raw`, `ssm_postnorm`) and the post-mixer residual (`hidden_mixer`) for layer
0, so the two paths are compared stage by stage. Results against
`base_q4kpure.gguf`, prompt `760 6511 314 9338 369`: every SSM stage within
0.008 of HIP, and the post-mixer residual matches to `max_abs 0.016`,
`mean_abs 5.5e-05`. (The initial 5.05 gap was comparing against the
post-FFN layer dump, not the mixer.)

`engine/run/loom_attn_layer_probe.cc` runs one full-attention layer end to end
on Loom: `yah_half_norm`, the Q/K/V Q4_K kStore projections, `yah_unpack_qg`,
`yah_fused_qk_rope_batched` (QK norm + RoPE + f16 KV cache write), the WMMA
attention core, the fp32->fp16 cast, and the Q4_K kResidual output projection.
`YAH_DUMP_LAYER=3` makes the HIP engine dump the attention stages (`attn_q`,
`attn_k`, `attn_gate`, `attn_out`, `attn_cast`) and `hidden_mixer` for layer 3.
Results on `base_q4kpure.gguf`: every pre-attention stage within 5e-4,
`attn_out max_abs 1.9e-04`, and the post-mixer residual `max_abs 5.6e-04`,
`mean_abs 4.1e-06`.

Two port fixes were needed to get there. The RoPE cache offset added
`bi*kv_width` on top of `cur = start_pos + bi`, double-counting the token index
for every batch > 1 (the in-tree case is batch 1, so it passed); the f16 cache
offset now uses only `kv_h*head_dim`. And the cache stores need `index.min`
clamps to `cache_elems - head_dim`, because the production-scale offset cannot be
proven from the declared config ranges (the same reason other ports clamp their
weight indices).

`engine/run/loom_forward_probe.cc` runs the **whole 64-layer prefill stack** on
Loom through HRX, no HIP: the embedding is Q4_K-dequantized on the host, each
layer runs the Gated DeltaNet or full-attention mixer followed by the paired Q4_K
gate/up and the Q4_K down residual, and the head is output RMSNorm + Q4_K GEMV +
argmax. On `base_q4kpure.gguf` with prompt `760 6511 314 9338 369`:

```
argmax=11751        (identical to the HIP yah-run reference)
final residual vs HIP layer_63: max_abs 0.095, mean 0.0023
```

Timing (untuned, correctness-first ports; two runs): `layers_ms=4461.8` and
`4401.8`, against the HIP `best_ms` of ~347-363 ms. The ~12x gap is the deferred
tuning target, not a correctness issue; the objective records it now.

Each layer individually matches HIP to ~0.05, so the 0.095 end-to-end is
accumulated fp16-staging/reduction-order drift and does not move the argmax.

A race showed up only when the per-layer debug syncs were removed: the small
norm-weight upload buffer was reused across layers, and the synchronous
`hrx_synchronous_h2d` overwrote it while the previous layer's dispatch still
read it. Giving each norm its own weight buffer fixed it; the run is
deterministic with no intervening syncs.

### Target-shard driver (mixed formats)

`engine/gpu/loom/tools/emit_prefill.py <model.gguf> <outdir>` walks the model
tensor table and emits every HAL the prefill needs, named by a convention the C++
driver reconstructs from each tensor's ggml type and shape:
`gemm_{kstore,residual,swiglu}_<fmt>_<m_tiles>_<k_blocks>.hal`. It emits every
GEMM HAL `Qwen3.8-27B-IQ4_XS-3.84bpw.gguf` reaches (50 for this shard), plus the
fixed HALs (norm/conv/prepkq/prepab/rowsplit/postnorm/unpack/rope/wmma/cast/gemv/
rmsnorm/argmax) and the IQ3_S/IQ3_XXS/IQ2_XXS/IQ2_XS grid/ksigns tables.

`engine/run/loom_forward_target.cc` is the format-aware driver: it reads
`Qwen35Config` for the layer schedule, picks the HAL per tensor from its type,
host-dequantizes the embedding, and runs the same 64-layer prefill. Every layer
now matches the HIP shard dumps (all 64 finite; worst max_abs 0.067, mean ~1e-03
at the last layer), and the full run reaches **argmax 11751**, the HIP reference.
`layers_ms=4447` against the HIP `best_ms=393.2`: the port is correct and
untuned, which is the intended stopping point here.

The last two formats on the shard were **IQ2_XXS** (FFN gate/up) and **Q2_K**
(one ssm_beta), now both ported. Q2_K initially produced all-NaN `beta` because
its decoder read d/dmin at the block base (the block_q4_K layout) instead of
bytes 80/82; the in-tree case reported `skipped` and hid it, and localizing the
first non-finite layer on the real shard found it. Emitting every HAL from the
cleaned source and re-running reproduces argmax 11751.

## 5. Remaining work

1. ~~Emit HAL executables for every ported kernel at its production shape~~ done:
   `emit_prefill.py` emits every GEMM HAL the shard reaches plus the fixed HALs.
2. ~~An HRX-native tensor/weights layer~~ done for prefill: the GGUF mmap is
   imported as an HRX buffer and activations plus recurrent/KV state are device
   buffers (`engine/model/loom_runtime.hpp`).
3. ~~Replace the HIP `Forward` layer~~ done for prefill in
   `engine/run/loom_forward_target.cc`; it dispatches the Loom kernels directly
   through the HRX native API with no HIP.
4. Prefill is **validated** (argmax 11751, worst layer max_abs 0.067). Decode is
   the next target: GEMV, decode attention, resident DeltaNet and the
   state-advancing decode convolution, then remove the HIP build
   (`engine/build_gpu.sh`, hipcc) and the HIP sources.
5. Prefill tuning is now unblocked (correctness matches HIP): the quantized GEMM
   staging is the target, see section 6.
## 6. Prefill performance

`YAH_LOOM_TIME=1` makes `loom_forward_target` synchronize at category boundaries
and print a breakdown of the 64-layer loop. On the IQ4_XS shard:

| category | before (ms) | after (ms) | share (after) | covers |
| --- | ---: | ---: | ---: | --- |
| norm | 18.0 | 13.4 | 0.9% | the two `yah_half_norm` dispatches per layer |
| mixer | 1490.5 | 595.7 | 38.2% | attention QKV/rope/wmma/output, or SSM proj/conv/recurrence |
| ffn | 2898.1 | 948.8 | 60.9% | `ffn_gate` kStore + `ffn_up` kSwiGLU + `ffn_down` kResidual |
| total | 4406.7 | 1558.0 | | |

argmax is 11751 in both runs and the final residual matches the HIP dump to
max_abs 0.062, mean 0.002.

Splitting the FFN category per arm gives: `post_attention_norm` 10.6 ms,
`ffn_gate` kStore 242.2, `ffn_up` kSwiGLU 249.1, `ffn_down` kResidual 464.7. The
down arm is the largest because its K is the FFN width (17408, `k_blocks=68`)
while its N is hidden (5120, `m_tiles=320`) — the same total work as the
gate/up arms but only 320 workgroups of 1088 serial K steps, so there is barely
one wave to hide the per-step latency. Split-K (or a shorter critical path) is
the next lever.

The quantized GEMM family (`yah_ffn_gemm_*_kstore/_swiglu/_residual`) is ~99% of
the time. The port staged the decoded weights through a **full-width** global
buffer (`view<[stage_rows]x[ktot]xf16>`, up to 178 MB at production shape): each
K step of 16 wrote a 16x16 tile into a 10 KB-strided row and the fragment load
read it back, so every 32-byte row access touched its own cache line and the
tile was re-fetched across the K loop.

The fix is a **dense per-workgroup staging tile**: the view is
`view<[stage_rows]x[16]xf16>` (512 bytes per workgroup), the decode stores at
`[row, c]` and the fragment load reads at `[m_origin, 0]`, so writes and reads
are contiguous within the tile. Measured on the `iq4xs` kStore at production
shape (m_tiles=1088, k_blocks=20): **14.64 ms -> 2.48 ms (5.9x)**, in-tree case
still passing. Rolled out to the 25 kStore/kSwiGLU/kResidual files, re-emitted,
and the driver's staging allocations dropped from `kFfn*kHidden*2` (178 MB) to
`kFfn*16*2` (557 KB).

The remaining ~4x gap to HIP (390 ms for 5 tokens, 415 ms for 64) is the same
family: HIP's Q4_K kernel was measured at 1.197 ms at m_tiles=1088, so the 2.48
ms iq4xs arm is close for that one, and the residual/SwiGLU arms and the
SSM/attention GEMMs are the next target. The port notes on
`yah_ffn_gemm_q4k_f32.loom` list the deeper levers (decode a whole 256-wide block
once and reuse its scale bytes across 16 K steps; move the tile into LDS). Those
are now unblocked: `engine/gpu/loom/docs/fragment-layout.md` shows lane L holds
logical row `L%16` and that a `buffer.alloca<workgroup>` LDS fragment load is
correct (the port note claiming otherwise was wrong); the LDS form already saves
~10% on the iq4xs kStore.