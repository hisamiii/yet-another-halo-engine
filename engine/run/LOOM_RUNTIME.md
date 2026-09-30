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

`emit_prefill.py <model.gguf> <outdir>` now emits the **complete** prefill HAL
set: every GEMM HAL plus the fixed kernels (`norm`, `conv`, `prepkq`, `prepab`,
`rowsplit`, `postnorm`, `unpack`, `rope`, `wmma`, `cast`, `rmsnorm`, `gemv`,
`argmax`) at the shard's shapes, and the IQ grid/sign tables copied from
`engine/gpu/loom/tables/` (extracted once from the ggml format tables and
committed). `emit_decode.py` reuses it and overwrites the fixed kernels with the
decode variants, so one command each reproduces both HAL sets with no `/tmp`
bootstrap. A whole-engine run is therefore reproducible from the checkout.
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
4. Prefill is **validated** (argmax 11751, worst layer max_abs 0.067). Decode
   runs but is 10.5x off HIP; removing the HIP build waits until tuning is done.
5. Prefill tuning is unblocked (correctness matches HIP): the quantized GEMM
   staging was the first lever, split-K the second, see section 6.
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
### Split-K on the residual arms

The three residual projections (`attn_output`, `ssm_out`, `ffn_down`) are the
lowest-occupancy arms: `m_tiles=320` workgroups of 320/320/1088 serial K steps,
so there is no second wave to hide the per-step latency. They now take a
`k_split` config (grid `workgroups(m_tiles, token_tiles, k_split)`) and write
`k_split` partial tiles instead of adding the residual in the epilogue
(`accum=0`); the driver reduces them with `yah_residual_add_1d` into the
residual. `emit_prefill.py` emits every residual arm with `k_split=4 accum=0`.

This took the full prefill from 1558 ms to **~1190 ms** (`ffn_down` 438 -> 214
ms) with argmax 11751 and the final residual matching HIP to max_abs 0.081.

One arm, `yah_ffn_gemm_q5k_residual_f32.loom`, was missed by the rollout: its
staging view stayed `view<[stage_rows]x64xf32>` and its epilogue loaded
`[gr2, tok]` with no split offset, so a `q5k` residual read another split's
staging tile. The run drifted (layer 0 max_abs 3.8 vs HIP, 65 at layer 63) but
argmax stayed 11751 because the head is robust. Isolating it needed a
controlled old-vs-split comparison at the production shape: emit the arm with
`k_split=1 accum=1` (the pre-split epilogue), run both, and the first divergent
residual names the arm. Fixed by matching the other seven; the in-tree case
does not catch it because it is emitted at `m_tiles=1`.

The reduction ping-pongs `hidden` and a scratch buffer so the accumulator's
`a`/`out` bindings never alias (`kSplit` is even, so the last step lands in
`hidden`); a same-buffer reduction violates the kernel's `noalias`.
### Host-side: import the GGUF tensor region once

The driver used to call `hrx_allocator_import_buffer` once per GEMM dispatch
(~192 times per run) and `hrx_synchronous_h2d` for every per-layer norm/conv/
state weight. Both are driver round-trips, and even though they overlap the GPU
they were the dominant host cost: the run was 22% host-bound (944 ms of GPU work
inside a 1206 ms wall). `Gguf` already exposes the whole contiguous tensor-data
region, so importing it once and binding each tensor as `(offset, bytes)` into
that one buffer removed the per-dispatch imports.

That alone took the prefill from 1206 ms to **963 ms** with identical output
(argmax 11751, final residual vs HIP max_abs 0.081), and the per-dispatch GPU
sum now matches the wall time, i.e. the host is no longer on the critical path.
### IQ3_S decode: hoist the j-invariant block work

The kStore/SwiGLU/Residual element loop maps lane `l` to `(row, col)` with
`e = l + j*32`, so `col = e & 15 = l & 15` is constant across the 8 `j`
iterations and `i = kbase + col` is loop-invariant. The port recomputed the
whole `group/half/q/which/b/l/lw` decode per element anyway. Hoisting it (and
pre-adding the per-block byte offsets, e.g. `qlo_off = 2 + g8 + lw`) out of the
`j` loop, plus `unroll(2) schedule(interleaved)` on it, is worth ~6% across the
three IQ3_S arms with identical output (argmax 11751, final residual 0.081 vs
HIP). The five reused index clamps become `index.assume ... [le(...)]`, which
the view-bound verifier accepts and which the backend already lowered to
`v_cndmask`/`v_max`/`v_min`.

`unroll(%N)` without `schedule(interleaved)` is *slower* (485 insns, VGPR spill);
factor 2 interleaved beats 4 and 8. The arm is latency-bound, not op-count
bound: the clamps and the branchless rewrite measured ~0, and the whole kernel
is ~42% issue-efficient against the 240 G warp-insn/s peak.
### Hoisting the block decode (all formats)

The kStore/SwiGLU/Residual element loop maps lane `l` to `(row, col)` with
`e = l + j*32`, so `col = e & 15 = l & 15` is constant across the 8 `j`
iterations and `i = kbase + col` is loop-invariant. Every port recomputed the
whole `group/half/q/which/l/lw` decode per element anyway. A small dataflow
pass (rewrite `c_i` to `lane & 15`, hoist every statement whose defs no longer
depend on `j`, add `unroll(2) schedule(interleaved)` to the element loop) was
applied to all 27 GEMM files. It is **bit-identical** (final residual max_abs
0.0, argmax 11751) and takes the prefill from ~1206 ms to **~765 ms**, with the
IQ3_S kStore alone 172 -> 117 ms.

`unroll(%N)` without `schedule(interleaved)` is *slower* (485 insns, VGPR spill);
factor 2 interleaved beats 4 and 8. Re-associating the per-block byte offsets
(`blk_off + 2 + g8 + lw` -> `blk_off + (2+g8+lw)`) on top measured ~0.6%, within
run-to-run noise, so it is not kept. The arm is latency-bound: clamps and
branchless rewrites measured ~0, and the kernel is ~42% issue-efficient against
the 240 G warp-insn/s peak.

Current prefill: **~663 ms** against the HIP `best_ms` of 389.8-415 ms, i.e.
~1.65x, down from 4407 ms for the first correctness-first port. The remaining
time is the weight decode (still ~46% of the IQ3_S kStore at `m_tiles=1088`)
and the structural rhs/MMA/epilogue.
### Narrowing the token tile to 16

The prefill pads `kB` real tokens to a 64-wide tile, so 3/4 of the rhs loads,
MMAs and epilogue stores are waste. The element loop maps lane `l` to column
`l&15`, and the epilogue decodes `(row, token)` with a shift/mask pair, so the
same source narrows structurally: drop n-groups 1..3, re-yield the untouched
accumulators, and make the token decode 16 wide. `emit_prefill.py` does this
(`narrow_tokens`) for every `yah_ffn_gemm_*` source; `TOKEN_TILE` (env
`YAH_TOKEN_TILE`) selects 16 or 64 and the residual reduction extent follows
(`kOutTotal = kHidden * YAH_TOKEN_TILE` in the prefill driver).

Result: **bit-identical** output (final residual max_abs 0.0 against the 64-wide
build) and a paired 805 -> 658 ms on the same run, ~18%. The decode path opts
out (`YAH_TOKEN_TILE=64`) because it uses a different geometry, and only tokens
0..kB-1 are ever read, so no other driver change is needed.

Measured splits for the IQ3_S kStore at `m_tiles=1088, k_blocks=20`, via
`engine/run/hal_bench.cc`: baseline 1.93 ms; the two per-K-step workgroup
barriers removed 1.92 ms (free); the whole decode replaced by `d` 0.90 ms, so
the decode is ~53% and the structural rhs/MMA/epilogue ~47%; the 16-wide tile
1.67 ms.
### Where the gap is: the weight dequant

Both engines can be built with the dequant replaced by a constant, so the bit
math and its weight reads vanish and only the MMA path is left: HIP by patching
`DecodeQuantSub16` (`quant_ops.hpp`), HRX by `YAH_ABLATE_DECODE=1` on
`emit_prefill.py`. Interleaved on the same prompt:

| | full | dequant removed | dequant |
| --- | ---: | ---: | ---: |
| HIP (`yah-run`) | 405.7 ms | 333.0 ms | 72.7 ms (18%) |
| HRX (`loom_forward_target`) | 678.8 ms | 139.1 ms | 539.7 ms (79%) |

The HRX **non-dequant path is 2.4x faster than HIP's** (139 vs 333 ms). The
whole gap, and then some, is the weight dequant: 540 ms against 73 ms, roughly
**7x slower per element** (~44 G weight-elements/s against ~330 G/s). Fixing
only the dequant to HIP's rate would put the prefill near 210 ms, about 2x
*faster* than HIP.

The cause is structural. HIP's `DecodeQuantSub16` decodes **16 elements per call
with packed 32-bit integer ops** - four elements per instruction - and hands the
MMA a packed `q[16]`. Our ports decode one element per lane per instruction
sequence: nibble extraction, sign, grid lookup and the `d`/`scale` multiply are
all per element, now with the loop-invariant part hoisted out.
### The word decode, and where it landed

The restructure landed as a **word-per-lane** mapping: one lane owns the four
consecutive columns `cb..cb+3` (`cb = 4*(lane&3)`), which share `group`,
`half`, `l` and `lw` and therefore share the qs byte, the qh byte, the grid
word, the signs byte, the scales nibble and `d`. They differ only in the byte
selector `b = column&3`, applied inside a 4-trip inner loop, and the packed sign
negate `(g_byte ^ (0 - sign_bit)) + sign_bit` replaces the per-element branch.
Two words per lane cover the 16x16 tile: `row = lane>>2` and `(lane>>2)+8`, and
because `kbase` is a multiple of 16 those two words share the whole offset chain,
so only the row offset is recomputed. The old mapping gave each lane one column
and redid all six loads and the offset chain four times: the modeled per-
workgroup weight reads drop 491520 -> 122880 bytes, and the weight loads per
k-step drop from 256 to 64.

One non-obvious requirement: the envelope analysis bounds the group index
loosely, so the scales load wanted `group = 8`, `sc_off = 110`, one byte past
the 35200-byte weight. `%group = scalar.andi %group_raw, %c7i` states the true
bound (`i_i = kbase + cb <= 252`, so `group <= 7`) and is a semantic no-op.
`loom_preflight.py` caught it; without the mask the case is refused.

`engine/run/hal_bench.cc`, isolated, `k_blocks=20`, 30 iters, old -> new:

| m_tiles | old | new | speedup |
| ---: | ---: | ---: | ---: |
| 1088 | 1.6660 | 1.5265 | 1.09x |
| 768 | 1.2622 | 0.8254 | 1.53x |
| 640 | 1.2095 | 0.5620 | 2.15x |
| 384 | 0.8235 | 0.3488 | 2.36x |
| 3 | 0.4480 | 0.1769 | 2.53x |

Correctness: the `m_tiles=1` fixture case (`fixtures/iq3s_gemm/expected_out.npy`,
captured from HIP) passes, the all-zero production case still returns exactly 0,
and the full 64-layer prefill output is **bit-identical** to the pre-word build
(argmax 11751).

**Timing has to be taken at the right grid.** The first conclusion drawn here
was that isolated timing inverts the pipeline ordering. That was wrong, and the
cause is worth recording: the residual arm splits K onto the grid's z dimension
(`workgroups(m_tiles, token_tiles, k_split)`) and the bench harness only had an x
dimension, so the residual was measured at one K-split of four. A quarter grid is
occupancy-starved, where the word decode's shorter instruction stream wins; at
the full grid the ordering matches the pipeline:

| kernel | m_tiles | k_blocks | grid | old | new |
| --- | ---: | ---: | --- | ---: | ---: |
| residual | 320 | 68 | 320x1x4 full | 1.8831 | 2.0629 (**9.6% slower**) |
| residual | 320 | 68 | 320x1x1 quarter | 0.5705 | 0.2430 (2.35x faster -- artifact) |
| kStore raw | 1088 | 20 | 1088x1x1 | 1.9628 | 1.7369 (1.13x faster) |
| kStore narrowed | 1088 | 20 | 1088x1x1 | 1.6746 | 1.5449 (1.08x faster) |
| kStore raw | 384 | 20 | 384x1x1 | 0.9755 | 0.4741 (2.06x faster) |
| kStore raw | 3 | 20 | 3x1x1 | 0.4929 | 0.2078 (2.37x faster) |

The residual and swiglu word decode were reverted: at the full grid they are
slower in isolation *and* in the pipeline, so that decision now rests on both
surfaces rather than one. Note also that the pipeline runs the
`narrow_tokens`-rewritten kernel (16-wide tokens), which is a different kernel
from the raw source; both are listed above.

Paired pipeline A/B, medians of three interleaved runs, every arm bit-identical
(argmax 11751):

| HAL set | layers_ms | mixer | ffn gate |
| --- | ---: | ---: | ---: |
| previous | 643.4 | 292.7 | 109.3 |
| per-geometry selection | **626.5** | 274.2 | 108.9 |
| word decode everywhere | 644.5 | 273.7 | 126.0 |

The kStore word decode is worth ~18 ms in the mixer. It costs ~17 ms in the FFN
gate, which is the one arm still out of step: at `m_tiles=1088` the pipeline says
the word decode loses 16.7 ms while the isolated narrowed kernel at the correct
grid with pattern data says it wins 1.08x. Unresolved; the per-geometry selection
sidesteps it rather than explaining it.

The selection is a config, so the good set is reproducible from source:
`yah_ffn_gemm_iq3s.word_decode` (0 = one element per lane, 1 = word) wraps the
two decode bodies in an `scf.if` on a compile-time constant, and
`emit_prefill.py` binds `0` at `m_tiles=1088` and `1` elsewhere. The fold is
exact: `word_decode=0` emits byte-for-byte the previous kernel's HSACO and
`word_decode=1` the new one, so every emitted HAL is one of those two machine
codes. The emitted set is kept at `/home/q/yah-hal-t16-wd` with
`/home/q/yah-hal-t16` as its drift baseline. The decode set comes from the same
source and `engine/tests/generate_gate.sh` still passes the 20-token reference.

Remaining split at `m_tiles=1088`: 1.729 ms with the word decode and 1.947 ms
without, and 0.861 ms with the decode replaced by `d`, so the decode is ~46% of
the kernel and the structural rhs/MMA/epilogue ~54%. The 2.4x HIP dequant gap
quoted above is closed on paper at 384/640/768 and open at 1088.

### The 1088 geometry, and why the grid table is not the answer

The decode's critical path is two dependent global loads: the qs byte selects a
grid word, and the grid word selects the magnitudes. Staging the 512-word grid
into 2 KB of LDS removes the second one, and in isolation that is worth 6.3% at
`m_tiles=1088` on the word decode (1.5456 -> 1.4451 ms). It does not survive the
pipeline. All four combinations of {global, LDS grid} x {word, one-element}
decode, three interleaved runs each, every arm issuing identical output:

| 1088 kStore | isolation | per-dispatch in situ | gate arm | layers_ms |
| --- | ---: | ---: | ---: | ---: |
| global grid, one element per lane | 1.6925 | 78.7 | **110.2** | **627.8** |
| LDS grid, word decode | 1.4451 | 85.3 | 115.9 | 633.6 |
| global grid, word decode | 1.5456 | 96.7 | 127.0 | 645.9 |
| LDS grid, one element per lane | 2.2992 | 100.4 | 131.7 | 652.7 |

The surfaces agree about the LDS axis -- staging the grid helps the word decode
and wrecks the one-element decode, in both -- and disagree about old versus word
under the global grid, which is exactly the comparison the deployed selection
rests on. The per-dispatch column is an independent measurement
(`YAH_LOOM_TIME=2` synchronizes around every dispatch, so no category boundary can
move work between arms) and it reproduces the gate-arm ordering. The LDS grid was
therefore rejected, and the selection already in place -- word decode at
384/640/768, the original at 1088 -- is the best of the four.

Calibration worth keeping: at 384/640/768 the isolated harness predicts the
pipeline to within 2% (word decode predicted -0.50 ms per dispatch, measured
-0.51 over 35 dispatches). At 1088 it does not, in either direction. Screen with
it at the mixer geometries; confirm at 1088 in situ.

The harness also takes a real weight blob now (`--wfile`), because the grid index
is a function of the stored bytes. Feeding a real `blk.*.ffn_gate.weight` in place
of the pattern moved the 1088 numbers by under 1% (1.5456 against 1.5463), so data
content is not what drives the disagreement either; cache and dispatch context are
the remaining candidates.

### What the decode share does with prompt length

The 5-token question -- how much of prefill is the weight dequant -- has a
non-obvious answer at length, and it is the same for both engines: the share does
not move, because both re-decode the weight once per token tile.

HIP, end-to-end, `yah-run` with and without `DecodeQuantSub16`:

| prompt | full | dequant removed | dequant | share |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 831.3 | 678.2 | 153.1 | 18.4% |
| 2048 | 3432.1 | 2688.2 | 743.9 | 21.7% |
| 2048, second run | 3715.8 | 2842.0 | 873.8 | 23.5% |

HIP is compute-bound from ~512 tokens on (1.6-1.8 ms/token at both), and its
256-token output tile amortizes one weight decode over 256 tokens.

HRX cannot run a 2048-token prefill yet -- `loom_forward_target` is a fixed
5-token batch with hardcoded ids -- but the IQ3_S kStore can be measured at the
2048-token shape. `safe_bench.py --check-only` gates the shape first, then the
same shape runs with the decode ablated (`emit_prefill.ablate_decode`, which
removes the bit math *and* the weight reads it consumed; the ablated build
declares weight 0):

| kStore, m_tiles=1088 | full | decode removed | decode | share |
| --- | ---: | ---: | ---: | ---: |
| 16 tokens, 16-wide tile | 1.541 | 0.193 | 1.348 | 87.5% |
| 2048 tokens, 16-wide tile (narrowed -- what prefill emits today) | 197.2 | 28.3 | 168.9 | 85.6% |
| 2048 tokens, 64-wide tile (the 4-N-subtile kernel) | 82.7 | 33.5 | 49.2 | **59.5%** |

Two things fall out. The share is flat in token count (87.5% -> 85.6%), and that
is the re-decode: HRX decodes a 16x16 weight tile once per workgroup, so with
`n` tokens it decodes every weight element `n/16` times, while the MMA work
scales with `n` in exactly the same way. And the tile width is worth 2.4x on its
own: the narrowed 16-wide tile -- right for a 5-token prefill, where 11 of 16
lanes would be padding -- decodes each weight element 128 times at 2048 tokens,
where the 64-wide tile decodes it 32 times, 82.7 ms against 197.2 ms for the same
work.

So at a realistic prompt length HRX's decode share is 59.5% against HIP's 21.7%,
and the residual gap splits roughly evenly between decode *rate* (about 58 G
weight-elements/s here against about 256 G/s for HIP, both derived from the
ablations above) and decode *count* (4x more: 32 re-decodes per element against
8). The "at long prompts the MMA dominates" intuition does not hold for either
engine; only a wider tile moves the share.

> **Correction, added later -- this span predates two fixes.** The timing tables
> from here through "Trying exactly 192 VGPRs" are historical. The early ones
> predate the word decode; the wave64 ones were taken through `hal_bench`, which
> launched a hardcoded 32-thread workgroup while those kernels declare 64, so the
> launches were half a workgroup. Two conclusions below do not survive: at 256
> tokens wave32 is 34.1 ms against wave64's 39.5 ms (wave64 is slower there, not
> 12% faster), and the 16-row width curve peaks at 64 tokens, not 256. Both are
> superseded by "The sweep, corrected" further down. The VGPR counts, residency
> tiers, fragment-load model and wave64 argument structure are unaffected -- those
> come from the compiled artifact and the source, not from timing. `hal_bench` and
> the driver now take the launch geometry from the executable's export metadata,
> so this class of error cannot recur silently.
>
### The tile width is the decode lever, not the prompt length

Why not just use a 64- or 256-token tile? It is a register budget, and it works.

One accumulator fragment per 16 tokens costs 8 f32 per lane (a 16x16 WMMA output
tile spread over 32 lanes), so the tile a workgroup can hold is set by how many
fragments fit. IQ3_S kStore, m_tiles=1088, k_blocks=20, 2048 tokens, decode
ablated by `emit_prefill.ablate_decode`:

| tile | fragments | VGPRs | full | decode removed | decode | share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 (narrowed) | 1 | 40 | 197.2 | 28.3 | 168.9 | 85.6% |
| 64 | 4 | 80 | 82.7 | 33.5 | 49.2 | 59.5% |
| 128 | 8 | 128 | 52.4 | 34.8 | 17.6 | 33.6% |
| 256 | 16 | 208 | **48.0** | 37.8 | 10.3 | **21.4%** |

The widened sources come from `tools/widen_tokens.py` (the inverse of
`narrow_tokens`: it duplicates the independent N sub-tiles), and every shape was
gated by `safe_bench.py --check-only` before anything was dispatched.

Three things fall out:

- The decode count falls as predicted -- 4x fewer re-decodes at 256 than at 64 --
  and the share falls with it, to 21.4% against HIP end-to-end 21.7%. **At a
  matched tile width the decode share gap is closed.**
- The ceiling is registers. 208 VGPRs is near the architectural 256, and the
  structural part of the kernel gets *slower* as the tile widens (33.5 -> 37.8 ms)
  because occupancy falls about 2.4x. That is why 256 is 1.7x better than 64 and
  not 4x.
- What remains after the tile fix is not the decode. At 256 tokens per workgroup
  the kernel does 365 GFLOP in 48.0 ms, 7.6 TFLOPS, against the 40.6 TFLOPS the
  HIP kernel reaches at the same shape. The shares match; the remaining ~5x is
  GEMM efficiency -- occupancy at 208 VGPRs, no software pipelining, single-
  buffered LDS staging.

Caveats, because these are feasibility numbers and not a shipped kernel. The
widened variants are verified structurally only: they compile, their declared
footprints fit, and they run, but no numerical reference exists for a 128- or
256-token tile, so a widened `check.case` is owed before any of it is trusted.
And the prefill driver still cannot run 2048 tokens at all -- `kB` is 5 with
hardcoded ids, and the attention path has to become causal over 2048 keys before
an end-to-end number exists.

### What the compiler's own analysis says

`loom-compile-report` (a Python tool under `loom/py/loom/tools/compile_report.py`,
run as `PYTHONPATH=/home/q/hrx/loom/py python3 -m loom.tools.compile_report`) turns
a `--compile-report=details` JSON into bounded views. Its `suggest` subcommand runs
the target provider's experiments -- on AMDGPU: residency cliffs, spill traffic,
private memory, LDS bank service, wait serialization, pipeline copy waits, wave
size, fragment packet expansion. It is the right first stop for a kernel question,
and it is what the residency numbers in the section above came from.

Run on the shipping kernel (narrowed 16-token tile, m_tiles=1088), it reports **no
findings**, and the report view says why:

| fact | value |
| --- | --- |
| final vector registers | 40 |
| scheduled vector pressure | 32 |
| resident subgroups per SIMD | 16 (max) |
| modeled occupancy | 100%, limit `max_waves` |
| private memory / spills | 0 B / none |
| instructions | 438 (207 vector ALU, 99 scalar ALU, 1 WMMA) |

So the kernel that actually runs prefill is not occupancy-limited and does not
spill. Whatever explains the 1088 in-situ ordering, the compiler does not attribute
it to residency. The static mix also quantifies the decode share that the ablations
measure: one matrix instruction per k-step against 306 scalar/vector ALU.

The widened variants are where it has something to say, and it is one thing -- a
high-confidence `amdgpu.residency_cliff`:

| tile | VGPRs | subgroups/SIMD | next tier | to reach it |
| ---: | ---: | ---: | ---: | --- |
| 64 | 80 | 12 | 16 | -16 VGPRs (<=64) |
| 128 | 128 | 8 | 9 | -16 (<=112) |
| 256 | 208 | 4 | 5 | -16 (<=192) |

Nothing else is flagged -- no LDS bank service, no spill traffic, no wait
serialization, no fragment packet expansion -- which also means the LDS store
pattern of the word decode is not a bank-conflict problem. The instruction counts
show the widening doing its job (WMMA 1 -> 4 -> 16 while scalar ALU stays at 99 and
vector ALU goes 213 -> 228) and show where the registers went: **register moves
25 -> 65 -> 149**. Cutting those is the path the compiler points at; the note below records that the
obvious way to cut them does not work.

#### The residency suggestion, tested

The compiler asks for 16 fewer VGPRs per tile width. The move table points at a
single `branch_edge` copy whose size tracks the accumulator count (13 / 37 / 69 /
129 units for 1 / 4 / 8 / 16 fragments), and the only branch carrying the
accumulators is the `word_decode` `scf.if` inside the k-loop. So the obvious move
is to delete that branch and let the emitter choose the decode body textually.

Tested, and **it changes nothing**:

| variant | register moves | peak live | `branch_edge` units |
| --- | ---: | ---: | ---: |
| narrowed, config-gated word decode | 25 | 32 | 13 |
| narrowed, branch-free word decode | 25 | 32 | 13 |
| narrowed, branch-free one-element decode | 24 | 31 | 14 |

The `scf.if` folds away completely, as a compile-time condition should, and the
`branch_edge` cause is the k-loop's own backedge carrying the accumulators, which
is a property of the tile width and not of the config knob. The tool is right that
a wider tile pays in residency; there is no cheap 16-VGPR recovery, and splitting
the N sub-tiles across two sequential K loops to halve the live accumulators just
reproduces the 128-token tile, which measured 52.4 ms against 48.0 ms for the
256-token one. The generator for these variants is `tools/widen_tokens.py`; the
branch-removal experiment needed no committed source change.

#### Getting under 192 VGPRs is a structural change, not a tweak

HIP production kernels sit at 192 VGPRs, so the 256-token tile at 208 is over
budget. Two cheap explanations were tested and both are refuted:

| hypothesis | test | result |
| --- | --- | --- |
| the `word_decode` `scf.if` forces the edge copies | generate a branch-free variant (CPU only) | 25 moves / 13 `branch_edge` units with and without; 24 / 14 for the branch-free one-element decode. The `scf.if` folds, and `branch_edge` is the loop backedge carrying the accumulators |
| source ordering of rhs loads vs MMAs drives pressure | generate `batch` (16 loads then 16 MMAs) and `interleave` (load/MMA pairs) | identical: 149 moves, peak 189, tier 4 |

So the pressure is the accumulator set itself. At a 16x256 tile one wave of 32
lanes holds 16 fragments x 8 f32 = 128 VGPRs (`branch_edge` is 129 units, i.e.
those 128 carried across the backedge plus one), and no reordering changes that.
Getting under 192 means fewer accumulator registers *per lane*, and there are two
ways:

- **wave64.** Loom has the schema but nothing uses it -- all 172 kernels in
  `engine/gpu/loom` declare `subgroup_size = 32`, and `tools/widen_tokens.py`
  output flipped to 64 is rejected: `matrix constraint 'wave_size' is not
  satisfied (source_bits=0, target_bits=256)`. The fragment types have to become
  `vector<8xf16>` and `vector<4xf32>`, which changes the layout the LDS staging
  must produce. HIP gets 40.6 TFLOPS from `prefill_quant_wave64`, and the tuning
  notes measure wave64 at 48.50 against 48.35 TFLOPS for the pure WMMA loop, so
  the half-accumulator win is real -- but it is a from-scratch port with no
  in-tree example.
- **Two wave32 subgroups per workgroup, split on N.** Keep `subgroup_size = 32`
  and the proven fragment layout, launch 64 lanes, and let each subgroup own 128
  of the 256 tokens: `wave = lane >> 5`, `lane32 = lane & 31`, the decode's j loop
  becomes one `j` per wave (`r8 = wave * 8`), and each subgroup loads its own rhs
  and stores its own half in the epilogue while reading the same decoded 16x16
  weight tile from LDS. Accumulators per lane halve to 64 VGPRs, the decode is
  still amortized over 256 tokens, and nothing about the fragment schema changes.
  This is the rewrite to attempt.

Prerequisite before trusting any of it: the 128- and 256-token variants have no
numerical reference. The existing fixture input is `check.generate.fill value(1.0)`
over 64 tokens, which makes the widened case cheap to check -- a 128-token case
with the same all-ones input must produce `expected_out.npy` repeated twice, which
verifies the sub-tile indexing directly. Build that case first; a register win on an
unverified tile is not a win.

#### wave64: under 192, and faster

The wave32 256-token tile costs 208 VGPRs, over the 192 that HIP production
kernels use. Loom does support wave64 for this shape -- `types.h` carries
`RDNA3_WMMAR3_F32_16X16X16_F16_W64` and a per-contract wave-size bitset -- and the
port is four mechanical changes (`tools/wave64_tokens.py`):

- `subgroup_size = 32 -> 64`, with `workgroup_size(64)` and `%c64` declared in the
  `kernel.def` scope, since a wave64 workgroup must be a full wave;
- accumulators halve, `vector<8xf32> -> vector<4xf32>`. The `16xf16` operand
  fragments deliberately do **not** change: each lane still supplies the full
  operand, replicated across the two 32-lane halves. Getting this wrong is what
  makes the compiler answer `matrix constraint payload_shape is not satisfied`
  instead of `wave_size`;
- the word decode covers the whole 16x16 tile in one pass, because `lane>>2` spans
  0..15 over 64 lanes rather than 0..7, so the decode j loop collapses from two
  iterations to one;
- the ostage->output copy uses 64 lanes per trip, so its trip count halves and it
  strides by 64.

At the 2048-token shape (m_tiles=1088, k_blocks=20):

| | VGPRs | SGPR | moves | subgroups/SIMD | resident lanes | full | decode | share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wave32 256-tok | 208 | 37 | 149 | 4 | 128 | 48.05 | 10.27 | 21.4% |
| **wave64 256-tok** | **136** | 28 | 75 | 3 | **192** | **42.18** | 5.99 | **14.2%** |

136 VGPRs is inside the 192 budget, residency rises from 128 to 192 lanes per SIMD,
and the kernel is 12% faster. The decode share falls *below* HIP 21.7% because a
latency-bound decode chain hides better with half again as many lanes resident.

Verified: the 64-token wave64 kernel passes the captured HIP fixture
(`@yah_ffn_gemm_iq3s_case`, `state: ok`), which is the case that pins the decode
arithmetic. The 256-token tile still has no numerical reference of its own -- the
widening is orthogonal to the wave-size port, but a widened case is still owed.

#### The width sweep: 192 VGPRs is past the knee

The wave64 port leaves the tile width free, so it was swept at a fixed 2304-token
total (m_tiles=1088, k_blocks=20, decode ablated for the share):

| tile | VGPRs | tier | resident lanes | re-decodes/token | full | decode | share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 80 | 6 | 384 | 18 | 46.80 | 9.34 | 20.0% |
| 256 | 136 | 3 | 192 | 9 | **46.40** | 5.99 | 12.9% |
| 384 | 176 | 2 | 128 | 6 | 50.94 | 3.51 | 6.9% |

and at a 2048-token total:

| tile | VGPRs | resident lanes | full | decode | share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 80 | 384 | 40.81 | 7.85 | 19.2% |
| 256 | 136 | 192 | 41.22 | 6.65 | 16.1% |
| 512 | 216 | 128 | 53.71 | 20.63 | 38.4% |

The decode share falls monotonically with width, because there are fewer
re-decodes, but the total does not: 128 and 256 are a plateau and 384/512 are
10-30% worse. The 384-token tile sits at 176 VGPRs -- essentially the 192 figure
often quoted as this architecture sweet spot -- and it is the slowest of the three.
The reason is in the last column of the fixed-total table: at 128 resident lanes the
decode's dependent loads stop being hidden, so its cost triples even as its count
halves.

So the quantity to maximize is neither VGPRs consumed nor waves resident on their
own, but tokens per resident lane at enough lanes to hide the decode. On this kernel
that is 128-256 tokens per workgroup at 80-136 VGPRs.

#### The widened tiles hid a real bug

The sweep is also what caught a generator bug worth recording. `widen_tokens`
derives the ostage->output epilogue for 32 lanes (trip count tok/2, stride 32), and
the wave64 port only rewrote the *64-token* case, by literal string match. Every
wider wave64 tile was therefore copying a fraction of its output: wrong results,
plausible timings. The 64-token fixture passed throughout because that one case was
correct.

That is exactly the risk flagged when the widened tiles were first measured -- no
numerical reference -- and it materialised. The epilogue is now derived from the
tile width (trip count tok/4, stride 64, row/token decode shifting by log2(tok) and
masking tok-1) and the fixture still passes, but a widened check case remains owed
before any wide-tile number is treated as final.

#### Trying exactly 192 VGPRs

The 384-token tile lands at 176 VGPRs, which is essentially the 192 figure often
quoted as this architecture's sweet spot. It is also the slowest of the widths
measured, and the arithmetic says why: for wave64 the residency tier is
`floor(32768 / (VGPRs * 64))`, so tier 2 spans roughly 171-256 VGPRs and 176 and 192
are the **same tier**. Adding 16 registers to reach 192 buys no occupancy at all.

The numbers worth aiming at are the tier boundaries, and the nearest one is 8
registers below the 256-token tile:

| tile | VGPRs | tier | lanes | to next tier |
| ---: | ---: | ---: | ---: | --- |
| 128 | 80 | 6 | 384 | 7 -> tier 7 (448) |
| 256 | 136 | 3 | 192 | 8 -> tier 4 (256) |
| 384 | 176 | 2 | 128 | 6 -> tier 3 (192) |
| 512 | 216 | 2 | 128 | 46 |

Two attempts to find those 8 registers, both CPU-only:

- reordering rhs loads against their MMAs: **136 VGPRs, 75 moves and a 67-unit
  `branch_edge` either way.** The backend schedules to its own preference and the
  source order does not survive, as it did not on wave32.
- folding the per-sub-tile address registers into immediate offsets: **not
  possible.** The activation layout is strided with k fastest, so consecutive
  16-token sub-tiles are `16 * ktot * 2` = 163840 bytes apart, far beyond the
  4096-byte immediate offset field. Each sub-tile needs its own 32-bit address
  register; the disassembly shows 16 distinct ones, each feeding a 128-bit load
  plus an `offset:16` for the second half.

So 136 is what this tile allocates, and the 128-256 range is a plateau in time
anyway (40.81 against 41.22 ms at a 2048-token total), which puts a bound of a few
percent on winning one more tier.

#### What the HIP kernels actually use

The 192 figure is real, but it came from the ported tree rather than from a
measurement here. Reading it out of the compiled device objects settles it. The
backend writes the allocated count into the AMDGPU metadata (repeated as
`.amdhsa_next_free_vgpr` in the ISA), so `llvm-readelf --notes` on the
`-hip-amdgcn-amd-amdhsa-gfx1151.o` device object is authoritative:

| kernel | wave | tile | VGPRs | allocated | SGPR | spill | private | LDS/wg |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `WKQuantA8BlockedWmmaGEMMKernel` Q4_K | 64 | 128x128 | **189** | **192** | 87 | 0 | 0 B | 20608 |
| ... Q5_K | 64 | 128x128 | 188 | 192 | 87 | 0 | 0 B | 19456 |
| ... Q8_0 | 64 | 128x128 | 176 | 176 | 26 | 0 | 0 B | 18432 |
| ... IQ4_XS | 64 | 128x128 | 176 | 176 | 28 | 0 | 0 B | 18432 |
| ... Q6_K | 64 | 128x128 | 175 | 176 | 87 | 0 | 0 B | 18432 |
| ... IQ3_S | 64 | 128x128 | 169 | 176 | 87 | 0 | 0 B | 18432 |
| ... IQ3_XXS | 64 | 128x128 | 169 | 176 | 87 | 0 | 0 B | 18432 |
| ... Q4_K, wave32 twin | 32 | 128x128 | 205 | 208 | 30 | 0 | 0 B | 30720 |
| ... IQ3_S, wave32 twin | 32 | 128x128 | 181 | 184 | 28 | 0 | 0 B | 27648 |
| ... Q2_K, wave32 twin | 32 | 128x128 | 256 | 256 | 30 | **5** | 24 B | 35840 |
| `HalfPrefillGemmKernel` Q4_K | 32 | 256x256 | **192** | 192 | 42 | 0 | 0 B | 24576 |
| ... IQ3_XXS | 32 | 64x128 | 223 | 224 | 22 | 0 | 0 B | 12288 |

The wave64 `WKQuantA8BlockedWmmaGEMMKernel` is the fast prefill path -- the one
`TryLaunchQuantPrefillWave64` selects for `batch >= 96, m >= 1024` and the one behind
the 3.18 ms/layer reference. It carries **169-189** VGPRs, Q4_K at 189, with 87 SGPRs
and, notably, **zero spills and zero private memory on every arm**. So "HIP prefill
uses 192" is literally correct for Q4_K: 189 used, 192 allocated, the granule being
the only reason the round number appears.

Two consequences that bound the sweep above.

Wave64 does cut the bill at equal tile: the 128x128 IQ3_S arm is 181 VGPRs at wave32
and 169 at wave64, Q4_K 205 -> 189. Halving the accumulator per lane is a real saving.
It just is not a tier: `floor(32768 / (189 * 64)) = 2`, i.e. 128 resident lanes -- the
*same* tier as the 384-token Loom tile at 176 that this section called past the knee.
HIP spends those registers on in-wave ILP and accepts 128 lanes; the speedup lives
inside a wave, not in more resident waves. Register count therefore does not separate
the two implementations, and matching 192 is not itself a goal.

Reproduce with `hipcc --save-temps -std=c++20 -O3 -DENGINE_ENABLE_HIP=1
--offload-arch=gfx1151 -I gpu/ported -I engine -mwavefrontsize64 -c
gpu/ported/src/models/qwen/hip/kernels/prefill_quant_wave64.hip` (drop the flag and
use `prefill_quant_gemm.hip` for the wave32 twins), then `llvm-readelf --notes` on the
emitted device object.

#### Widening rows, not tokens: the rhs fragment is the shared operand

Every widening tried so far grew the token tile. That widens the *lhs*, the
decoded weight fragment read from LDS: one lhs feeds n MMAs and the rhs fragment,
the activation read from global, is re-loaded for each. Row widening does the
opposite, and the two are not interchangeable. For a workgroup tile of n_row*16
rows and tok tokens, per K step of 16:

| quantity | count |
| --- | --- |
| MMAs | n_row * tok/16 |
| lhs fragment loads (LDS) | n_row |
| rhs fragment loads (global) | tok/16 |
| 16x16 weight decode tiles | n_row |

so operand fragment loads per MMA are `1/(tok/16) + 1/n_row`: the lhs term is the
token width's lever and the rhs term is the row width's. Weight decode per output
is `16/tok` and does not move with n_row at all -- which is why the two levers
bend different costs. `tools/widen_rows.py` implements the row transform on top of
`widen_tokens.py` and `wave64_tokens.py`, for either wave size: the decode row map
is `lane>>2`, so a pass covers 16 rows at wave64 and 8 at wave32, and the ostage
copy's trip count divides rows*tok by the wave size. The grid's x dimension
divides by n_row while `m_tiles` keeps its meaning, so every buffer size the
harness derives is unchanged. The correctness gate is new: the fixture generator
under `fixtures/iq3s_gemm_wide/` emits 16/32/64/128 *distinct* rows, because a
duplicated fixture would make a wrong row origin self-consistent. All 22 shapes
below pass it (`state: ok`).

##### A measurement-surface bug, and what it invalidated

Two hardcoded workgroup sizes were wrong, and they invalidated the first pass at
this sweep.

* `hal_bench` was stale relative to its own source and reported numbers ~3.5x
  faster than that source produces. The stale binary said 0.42 ms for the shipping
  wave32 IQ3_S kStore at `m_tiles=1088` with one 16-token tile. Rebuilt, the same
  call gives **1.54 ms**, and the pipeline's own per-dispatch breakdown
  (`YAH_LOOM_TIME=2`) puts `yah_ffn_gemm_iq3s` at `78.2 ms / 62 dispatches =
  1.26 ms` -- agreement to 22%, against the stale binary's 3.7x. The harness now
  prints `wg=` so the launch geometry is in the transcript.
* `loom_forward_target.cc` passed `sx=32` at every GEMM site, so substituting a
  wave64 HAL launched half a workgroup: plausible timing, garbage output, and
  nondeterministic across runs (argmax 11751 -> 248320 and 1076).

Both now read the workgroup size from the executable's own export metadata
(`hrx_executable_export_info_t.workgroup_size`), which is populated (wave32 arms
report 32, wave64 arms 64) and agrees with the caller's value at every non-GEMM
site (norm 32, unpack 256, rope 256, rowsplit 128, accum 256, ...), so the change
is neutral except where it was wrong. `subgroup_size` is left at 32: libhrx never
reads it.

Consequence: every timing in the wave64 subsections above this one came through
the stale harness and is superseded below. The wave-size *direction* changes with
it -- at 256 tokens the corrected measurement has wave32 at 34.1 ms against
wave64's 39.5 ms, the opposite of what those subsections claim.

##### The sweep, corrected: 2048 tokens, 17408 rows, k_blocks=20

`safe_bench --iters=5`, one sample per cell, repeated cells agree to 1-3%:

| rows \ tok | 16 | 32 | 64 | 128 | 256 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **wave64** | | | | | |
| 16 | 60.15 | 36.80 | **30.58** | 37.75 | 39.51 |
| 32 | 61.78 | 33.68 | 23.04 | 23.65 | 21.12 |
| 64 | 60.73 | 33.20 | 22.59 | **17.10** | 232.21 |
| 128 | 73.39 | 33.79 | 25.67 | 159.53 | (will not allocate) |
| **wave32** | | | | | |
| 16 | 64.03 | 40.30 | 35.18 | **33.79** | 34.07 |
| 32 | 61.64 | 36.40 | 25.18 | 23.54 | 176.63 |
| 64 | 64.26 | 37.07 | 24.55 | 122.04 | |

and the decode ablation (the weight decode and its dependent chain removed, so
the delta is the decode's cost):

| arm | full | ablated | decode | share |
| --- | ---: | ---: | ---: | ---: |
| w32 r1 16x16 | 64.03 | 22.53 | 41.5 | 65% |
| w64 r1 16x16 | 60.15 | 34.00 | 26.2 | 43% |
| w64 r2 32x64 | 23.04 | 11.59 | 11.5 | 50% |
| w64 r4 64x64 | 22.59 | 9.65 | 12.9 | 57% |
| w64 r4 64x128 | 17.10 | 9.91 | 7.2 | 42% |

Four things fall out.

First, row widening is a large win where there is a token width to amortise it
against: 60.15 ms at 16 rows x 16 tokens to 17.10 at 64 x 128, 3.5x, and 2.2x
against the same-width 16 x 128 tile.

Second, there is no win at the 16-token tile (60.15 -> 60.73 going from 16 to 64
rows), and the ablation says why: the decode is 65% of the wave32 16-token kernel
and 43% of the wave64 one, and decode per output is `16/tok` -- row widening does
not touch it. The rhs term the model predicts is real; it just is not what binds
at that width.

Third, the corrected 16-row curve peaks at **64 tokens** (30.58), not 256 as the
superseded section above concluded.

Fourth, the register wall is bracketed from both sides: 64 x 256 spends all 256
VGPRs plus 1984 B of private memory and lands at 232 ms; 128 x 128 spends 240
VGPRs plus 1760 B and lands at 159.5 ms; 128 x 256 does not allocate at all.

##### Where the prefill time actually is

`YAH_LOOM_TIME=2` on the shipping HAL set (5-token prefill, 64 layers, ~636 ms
total) by kernel name, top ten:

| kernel | ms | dispatches | ms each |
| --- | ---: | ---: | ---: |
| yah_ffn_gemm_iq3s | 78.2 | 62 | 1.26 |
| yah_ffn_gemm_q4k | 60.2 | 70 | 0.86 |
| yah_ffn_gemm_iq3s_residual | 60.1 | 41 | 1.47 |
| yah_ffn_gemm_iq4xs | 55.8 | 40 | 1.40 |
| yah_ffn_gemm_iq3xxs_residual | 47.2 | 40 | 1.18 |
| yah_ffn_gemm_iq4xs_swiglu | 38.7 | 23 | 1.68 |
| yah_ffn_gemm_iq3s_swiglu | 37.7 | 20 | 1.89 |
| yah_ffn_gemm_iq3xxs_swiglu | 30.1 | 19 | 1.58 |
| yah_ffn_gemm_iq3xxs | 29.9 | 21 | 1.43 |
| yah_ffn_gemm_q5k | 21.0 | 42 | 0.50 |

The single-kernel harness reproduces the iq3s kStore to 22% (1.54 against 1.26),
so the two surfaces now agree. Note the shape of this: the GEMM family is ~500 of
the 636 ms and every dispatch is 0.3-1.9 ms, so the prefill is a long sequence of
small, latency-bound dispatches, not a few large ones. With 1088 workgroups and 64
resident waves per CU the production GEMM is also *under-occupied*: the same
kernel in a 128-token-tile grid (139k workgroups) reaches 0.47 ms per
1088-workgroup equivalent, 3.3x better, which is the throughput regime the
corrected sweep below measures.

##### The production shape, and why wave64 does not ship

The engine's prefill is `constexpr std::uint32_t kB = 5`
(`engine/run/loom_forward_target.cc`) padded to a 16-token tile, and
`emit_prefill.py` binds `token_tiles=1`: one token tile, so the weight is decoded
once per row tile, there is no re-decode to remove, and row widening only divides
the parallelism (1088 -> 272 workgroups). At that geometry (gx=1088, gy=1) the
rebuilt harness measures wave32 1.54 ms against wave64 1.96 ms -- wave64 is 27%
slower.

The paired full-pipeline A/B agrees, and unlike the microbenchmark it is
end-to-end and bit-identical:

| set | wave64 substituted at | sum ms |
| --- | --- | ---: |
| t16-wd (baseline) | -- | 631.9 / 637.5 / 638.8 |
| w64B | m_tiles 3, 384, 640, 768 | 642.5 / 641.5 / 640.5 |
| w64C | the above plus 1088 | 656.0 / 648.0 / 655.8 |

so wave64 is +0.9% at the geometries whose shipping config already uses the word
decode (B) and +2.6% once `m_tiles=1088` is included (C), where the gate category
goes 110.6 -> 125.4 ms. All six runs give argmax 11751 and `cmp` is byte-identical,
so the port is correct; it is simply slower. Its only real value here is halving
the per-lane accumulator, which is what lets a 64-row tile exist at all: the
wave32 64 x 128 tile spills and takes 122 ms.

So neither lever touches what ships today. The row-widening result is real and
large, but it belongs to a 2048-token path the engine does not have yet, and the
current 5-token prefill is unchanged by everything in this section.

### pp2048: the DeltaNet recurrence was 25% of the prefill, now register-resident

`YAH_LOOM_TIME=2` at B=2048 on the ar3 set (9.77 s layers_ms) put `yah_deltanet`
at 2453 ms over 48 dispatches, 51 ms each, against 3.4 ms per layer for HIP's
`BatchedDeltaNetRowSplitKernel` (docs/kernel-tuning.md) -- 15x, and the largest
single item in the profile. The quant GEMMs are ~6.8 s of the rest (~15 TF/s
against HIP's ~27-31).

The rowsplit kernel is one lane per state row, 48 workgroups of 128 lanes (about
one wave per SIMD), and every token walks the lane's 128-key row twice through
memory (LDS after `deltanet_lds_rewrite`). With nothing to hide latency behind,
that is ~200 cycles per inner step. `yah_deltanet_regtile_f32.loom`, generated
by `tools/gen_deltanet_regtile.py`, keeps the same lane-per-row geometry and the
same f32 operation order but carries the row through the token loop as
`vector<8xf32>` values: loaded once, stored once, with only k/q (wave-uniform,
so they lower to scalar loads) and the per-token scalars touching memory per
token. Two compile constraints shaped it: keeping the readout's `s*alpha` and `k`
live into the update overflows VGPRs (they are recomputed/reloaded, as rowsplit
does), and 16-wide chunks overflow SGPRs at B=512 (peak 157 of 106), so the
chunk is 8 wide.

| arm | scaled case (B=512) | pp2048 layers_ms | hidden md5 |
| --- | ---: | ---: | --- |
| rowsplit + LDS rewrite | 5.92 / 5.93 ms | 9844.0 / 9917.4 | f837e614ff55d1d1 |
| regtile | 1.10 / 1.09 ms | **7722.7 / 7747.2** | f837e614ff55d1d1 |

Interleaved, 15 s gaps, sets differing only in `rowsplit.hal`. Bit-identical,
1.28x on the prefill. The emitters now build `rowsplit.hal` from regtile;
`YAH_DELTANET=lds` restores the LDS form.

### Shared-decode kStore: decode once per workgroup, prefetch it, store direct

After the DeltaNet fix the quant GEMMs are ~6.4 of ~7.6 s. The chained kStore
(one wave64 per workgroup, 64 rows x 128 tokens) re-stages and re-decodes a 64x16
weight slice every 16-wide K step behind three barriers, one element per lane
per step; for IQ4_XS the codebook is a 4-level `scf.if` tree and each element
issues five LDS byte loads. `tools/gen_gemm_shared.py` generates a replacement
with the same ABI, output and f32 op order (so the hidden md5 is the gate):

- NW wave64 waves share one decoded 64-row tile, each wave the same 64x128
  accumulator tile as before;
- the tile is decoded once per KSUB-wide phase, row-per-lane: scales once per
  32-element group, the qs bytes as one vector load, the codebook through
  `vector.table.lookup`;
- the next phase's raw bytes are loaded before this phase's MMAs and carried by
  the loop (prefetch);
- the result fragments are stored straight into the token-major output through
  a strided view, instead of an ostage round trip (3x the output bytes).

IQ4_XS kStore, per-dispatch mean over its 67 dispatches at pp2048 (`YAH_LOOM_TIME=2`),
every arm bit-identical (hidden f837e614ff55d1d1):

| arm | ms |
| --- | ---: |
| chained (shipped) | 13.98 |
| shared NW=4 KSUB=256 | 16.52 |
| shared NW=4 KSUB=128 | 15.95 |
| shared NW=2 KSUB=64, no prefetch | 14.71 |
| + prefetch | 13.43 |
| NW=2 KSUB=128 + prefetch | 12.55 |
| **+ direct epilogue** | **9.71** |

The first two rows are the lesson: sharing the decode alone lost, because a 34 KB
tile halves residency and each phase waited out its own DRAM loads. Probes on the
prefetch-less NW=2 kernel (decode removed: 9.74 ms; MMAs removed: 9.71 ms, of 13.49)
showed decode and MMA serialised with ~6 ms left over, which is what pointed at
the epilogue. Weighted over the ~13.9 TFLOP those dispatches do, 14.9 -> 21.4 TF/s.

End to end, interleaved, 15 s gaps: 7917.2 / 7807.4 ms -> 7681.0 / 7601.0 ms,
argmax 11751 and f837e614ff55d1d1 on all four. `emit_prefill_pp.py` builds the
IQ4_XS kStore HALs from the generator (`YAH_SHARED_GEMM=0` keeps the chained
kernel), and a full emit with `YAH_DIRECT_EPI=1 YAH_GEMM_W64=1 YAH_TOKEN_TILE=128
YAH_ROWGRP=4 YAH_CHAIN_LEVEL=rows` reproduces the measured set file for file.
Other formats need their own row-per-lane decoder in the generator.

### Benchmarking took the box down twice

Two reboots on 2026-09-29 came from this harness, not from a kernel bug. Both
were operand extents that the arithmetic said were fine and were not: an input
sized for `k_blocks=20` (655360 B) reused at `k_blocks=68`, and an output of
`m_rows*tokens*4` when a `k_split=4` residual writes `m_rows*k_split*tokens*4`.
An extent past the end of an allocation on this target does not fault -- the
shader reads unmapped VA, never returns, and hangs with no page fault for the
driver to report, so `gfx_0.1.0` times out, MES stops answering `msg=RESET` and
the machine resets.

Three things now stand between that and another reboot:

- `tools/safe_bench.py` compiles the source under the same config, reads the
  footprint the compiler recorded
  (`source_low.memory.roots[].interval_envelope.byte_count`, the surface
  `loom_preflight.py` already uses) and refuses unless every operand fits the
  buffer it will be handed. It is what found the `k_split` output term, with no
  dispatch and no GPU.
- `engine/run/hal_bench.cc` computes no size of its own any more: sizes are
  parameters and the grid is checked against the shape (`gx*16 <= m_rows`,
  `gz <= k_blocks`) before anything is submitted. K splits live on gz, so folding
  one into gx walks the m origin off the end of the weight.
- `engine/run/gpu_run.sh` snapshots `dmesg` before and after a command and tails
  it into a log for the duration, because a bad dispatch kills the process without
  printing anything. After a reboot the previous boot's log survives in the
  journal: `doas journalctl -k -b -1 --no-pager | grep -iE 'amdgpu|MES|reset'`.
  Logs from both incidents are in `/home/q/yah-scratch/gpu-*.dmesg.log`.

`hal_bench` is a hypothesis generator regardless of how it is sized: one kernel,
on a quiet GPU, with no other dispatch in flight. Only the paired full-pipeline
A/B is a result, and the machine drifts enough between batches (up to ~30 ms on
the same HAL set) that only interleaved paired differences should be read.

### Per-kernel parity against HIP, from device timestamps

The target is every Loom kernel at or above its HIP counterpart, so both sides
are measured per dispatch on the device, not by host-synchronized wallclock
(`YAH_LOOM_TIME=2/3` adds a sync and launch latency to every kernel):

```sh
# HIP: rocprofv3 sees the HIP runtime's HSA queue
rocprofv3 --kernel-trace --output-format csv -d hiptrace -o run -- \
  engine/build/yah-run <gguf> --ids-file ids2048.txt
# Loom: HRX's own profiler (rocprofv3 does not see HRX dispatches) plus the
# driver's dispatch order, which names the HAL behind each anonymous executable
YAH_LOOM_SEQ=seq.csv HRX_PROFILE_FILE=p.irpf HRX_PROFILE_MODE=dispatch \
  engine/build/loom_forward_pp <gguf> <hal dir> <out> 2048
iree-profile dispatch --dispatch_events --format=jsonl p.irpf > loom.jsonl
python3 engine/run/kernel_parity.py hiptrace/run_kernel_trace.csv loom.jsonl seq.csv
```

The script keys GEMMs by (epilogue, format, M, K) on both sides: it splits both
streams at the half_norms (two per layer) and pairs each HIP GEMM with the Loom
one of the same format and M in that segment, and it folds HIP's fused gate+up
together with Loom's gate kStore + SwiGLU pair so they are compared as one step.

Two things had to change before the Loom side produced anything. `LoomDevice`
released the device it got from `hrx_gpu_device_get`, which does not retain it,
so by the time `hrx_gpu_shutdown` ended the profiling session the device had no
HAL handle and the file held a session_begin and nothing else. And HRX's native
stream created its command buffers without `RETAIN_PROFILE_METADATA`, so they
got no profile id and the AMDGPU driver skipped every dispatch in them (the HRX
graph executor already sets the flag when profiling is active; the stream in the
local HRX checkout now does the same).

First table, sd12 HAL set against yah-run, IQ4_XS shard, pp2048 (ms summed over
the run, device time):

| category | HIP | Loom | Loom/HIP |
|---|---:|---:|---:|
| GEMMs (M > 64) | 3164 | 4663 | 1.47 |
| DeltaNet | 94 | 259 | 2.76 |
| attention | 46 | 109 | 2.38 |
| half_norm | 37 | 68 | 1.83 |
| small GEMMs (M = 48) | 16 | 42 | 2.67 |
| head GEMV | 4.5 | 15.3 | 3.42 |
| ssm_conv, ssm_postnorm, unpack_qg, prep_kq, half_cast | 94 | 93 | ~1.0 |
| fused QK-norm/RoPE | 17.8 | 11.0 | 0.62 |
| total | 3479 | 5263 | 1.51 |

The GEMM ratio runs from 1.28 (iq3s/iq3xxs attn/ssm out) to 1.9 (iq4xs
ffn_down, q4k 10240-row qkv); Loom is already ahead on iq3s 6144x5120 (0.55) and
q4k ffn_down (0.88).

### The tile GEMM: HIP's structure, and the bank conflict that hid it

ATT traces of the same GEMM on both sides (IQ3_S 17408x5120 kStore; HIP via
`rocprofv3 --att --att-library-path <therock lib>`, Loom via the benchmark
tool's executable traces) showed where the shared kernel's time went. 80% of its
wave time was `s_waitcnt`, mostly vmcnt on the activation loads each wave issues
from global memory a few instructions before the WMMA that consumes them, with
~2 wave64 waves per SIMD to cover it. HIP's `HalfPrefillGemmKernel<256, 256, 8,
4>` stalls about as much per wave (42% waitcnt, 26% barrier), but it has 32
wave32 waves per workgroup, 8 per SIMD. Per K-tile its threads store the
previous tile's decoded weights and activations to LDS, pass a barrier, issue
the next tile's loads and decode into registers, then run 32 WMMAs per wave out
of LDS. Per WMMA the shared kernel actually issued less VALU (3.9 vs 4.7) and
far less LDS (0.34 vs 1.65) work.

`tools/gen_gemm_tile.py` already had that structure (both operands in LDS, next
phase in registers) and had measured slower, 19.9 ms against the shared
kernel's 14.9. Its trace put 62% of wave time in lgkmcnt waits, with the
`ds_load_b128`s themselves stalling at issue. The cause was bank conflicts: an
unpadded decoded weight row is 64 f16 = 128 B, so the 16 rows of an lhs
fragment load fall on two bank groups. Padding each weight row by 8 f16
(`YAH_TG_WPAD`) and each activation row by 8 (`YAH_TG_APAD`, already there)
gives:

| IQ3_S 17408x5120 kStore, standalone | ms |
|---|---:|
| shared (64x256, 2 wave64) | 14.9 |
| tile 64x256, 8 waves, unpadded weights | 19.9 |
| tile 64x256, 8 waves, WPAD=8 | 11.1 |
| tile 128x256, 16 waves (4x4), WPAD=8 | 10.5 |
| same, WPAD=4 / APAD=0 | 19.6 / 28.2 |
| tile 256x128, 16 waves (8x2) | 14.4 |
| HIP 256x256, in the pipeline | ~11.3 |

256x256 does not fit: 64 KB of tiles plus the 2 KB IQ grid table the decode
stages in LDS exceeds gfx11's 64 KB per workgroup (HIP reads its grid from
global). The generator gained a SwiGLU epilogue (per-wave f32 slabs through LDS,
half the wave's tokens at a time so all 16 waves fit) and parameterized
geometry; `emit_prefill_pp.py` uses it for every shape whose m_tiles is a
multiple of 8 in the formats in `TILE_FMTS` (all shared-decode formats but
q8_0, which needs kdiv). Everything else keeps the shared kernel.
`YAH_TILE_GEMM=0` restores it everywhere.

Bit-identical (argmax 11751, hidden f837e614ff55d1d1): each format x kind was
rolled in one at a time behind the gate. Device time at pp2048 (sd12 -> tile set,
same recipe):

| | HIP | shared | tile |
|---|---:|---:|---:|
| GEMMs (M > 64) | 3164 | 4663 | 3938 |
| whole prefill | 3479 | 5263 | 4574 |

Per shape, the grid formats (IQ3_S, IQ3_XXS, IQ2_XXS) are now within ~1.1-1.2x of
HIP, and the IQ3_S 6144-row z projection and q4k ffn_down are ahead of it. IQ4_XS and
Q4_K are still 1.6-1.75x. Their decode runs on fewer lanes: at KSUB=64 a phase
has 2 groups of 32 per row, so with 128 rows half of the 512 lanes decode
IQ4_XS, and a quarter decode Q4_K, whose groups pair up. HIP spreads each tile's
decode over all 1024 threads.

#### Why not 256 x 256 like HIP, and the fence that mattered more

HIP's 256 x 256 tile fits in 64 KB because it needs no padding: its LDS tiles are
fragment-major (each 16 x 16 block 512 contiguous bytes), split into two 256 B
halves (k 0-7, k 8-15) so that each ds_load_b128 of 16 lanes reads 256
contiguous bytes. `YAH_TG_FRAG=1` builds the fragment-major layout Loom can
express (16 x 16 row-major blocks, a plain 16-wide view); its per-lane 32 B rows
leave lanes i and i+8 on the same banks, and the half split is not expressible
as a strided fragment view. Measured (IQ4_XS 17408x5120 kStore, standalone):

| variant | ms |
|---|---:|
| padded 128 x 256, KSUB=64 (shipping) | 12.1 |
| fragment-major 128 x 256 | 14.3 |
| fragment-major 256 x 256, 32 waves | 13.8 |
| padded 256 x 256 at KSUB=32 (fits, 80 B rows) | 14.1 |
| padded 128 x 256, KSUB=64, `YAH_TG_FENCE=1` | 10.3 |

The tile size was not the limit. The trace of the IQ4_XS kernel had 31% of wave
time on one `s_waitcnt vmcnt(0)` at the phase boundary: the scheduler hoisted the
next phase's weight loads above the stores of this phase's activations to LDS,
then waited for every outstanding load, the fresh ones included, before the
first store. A `scf.schedule.fence` between the stores and the next loads
(now the default) takes IQ4_XS 12.1 -> 10.3, Q4_K 12.2 -> 11.6 and IQ3_S 10.6 ->
10.3 ms standalone; in the pipeline, interleaved, GEMM device time 3675/3762 ->
3516/3558 ms (HIP 3164), bit-identical.

### half_norm: the same loop, unrolled

`yah_half_norm` (fused=0) was 1.85x HIP's `HalfNorm5120` (68.7 vs 37.1 ms over
128 calls). HIP moves 63 MB per 2048-row call at ~217 GB/s, near the DRAM
bandwidth; the Loom kernel, one wave32 per row, loaded, squared and added one
element per lane per loop iteration and then walked the row a second time.
`tools/gen_half_norm.py` emits the same kernel fully unrolled: the same ops in
the same order (mulf then addf per lane, then the subgroup reduce), but every
lane issues its 160 loads up front and keeps the values in registers for the
output pass. Interleaved, device time: loop 68.7, `unroll(8)` 61.9, fully
unrolled 42.3 ms (1.14x HIP), bit-identical. The emitter uses it unless
`YAH_NORM_UNROLLED=0`.

## 7. Decode and the HIP removal

The decode forward is `engine/run/yah_hrx.cc`, built by `engine/build_hrx.sh`.
It processes one token at a time (prompt and generation share the path):

- a host IQ4_XS embedding row lookup;
- a persistent fp16 KV cache in `[layer][position][kv_head][dim]` layout,
  written by `yah_fused_qk_rope` (position from a device i32) and read by
  `yah_decode_attn`;
- `yah_ssm_conv_decode` + `yah_deltanet_decode` for the recurrent state;
- the `token_tiles=1` prefill GEMM HALs for every projection (only token 0 of the
  64-token tile is used);
- `yah_rmsnorm` + `yah_gemv_q6k` + `yah_argmax` for the head.

`engine/gpu/loom/tools/emit_decode.py` emits the decode HAL set: it reuses
`emit_prefill.py` for the GEMMs and adds the decode kernels, with one
`yah_decode_attn` HAL per `start_pos`. On Qwen3.8-27B-IQ4_XS the runner
reproduces the recorded HIP sequence exactly (`11751 13 198 760 6511 314 9564
369 19241 13 198 760 6511 314 14898 369`), and `engine/tests/m0_gate.sh` and
`generate_gate.sh` pass against it.

HIP runs **alongside** HRX again. It was removed in `01279d6` before tuning was
finished; `9f8b142` restores the 141 deleted files (a pure deletion, so the
restore is conflict-free), and the pre-removal state is tagged `hip-pre-removal`
and branched `hip-reference`. `engine/build_gpu.sh` still builds
`engine/build/yah-run`, and the HRX path is untouched (`build_hrx.sh` plus the
CMake core tools).

Live baselines on Qwen3.8-27B-IQ4_XS, same prompt, both engines:

| | prefill | decode |
| --- | ---: | ---: |
| HIP (`yah-run`) | 386.5 ms | 14.02 tok/s (71.3 ms/token) |
| HRX (`yah-hrx` / `loom_forward_target`) | ~663 ms | 1.33 tok/s (752 ms/token) |

Prefill is 1.66x; decode is 10.5x. Both reproduce argmax 11751 and the same
16-token sequence. The decode gap is structural: the HRX decode drives the
prefill GEMM HALs, computing a 64-wide token tile per generated token and
discarding 63, while HIP does a single-token GEMV.

One subtle porting bug worth recording: `yah_unpack_qg` has `head_dim` as a
`config.def` defaulting to 1. A decode HAL emitted without
`yah_unpack_qg.head_dim=256` de-interleaves the Q/gate projection with the wrong
stride, so the attention reads a permuted q and the whole model diverges while
the projections still look plausible. `emit_decode.py` now sets it.