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
25 -> 65 -> 149**. Cutting those is the concrete path to the next residency tier on
a wide tile, and it is the one experiment the compiler asks for.

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