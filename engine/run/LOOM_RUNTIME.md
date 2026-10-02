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
and `gfx11-generic` (kind 1). Pass the key the artifact was built for. Every
Loom source, generator and script now targets `gfx1151` (2026-09-30);
`YAH_LOOM_TARGET` in `emit_hal.py` overrides it.

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

### DeltaNet: k/q through LDS instead of SMEM

ATT of the regtile recurrence: 58% of wave time in `s_waitcnt lgkmcnt(0)`. Every
lane owns a whole 128-key state row, so the token's k and q are wave-uniform and
lower to scalar loads; SMEM returns out of order, so each use drains every
outstanding scalar load -- ~6 memory round trips per token, ~7250 cycles per
token against ~1100 of VALU. (HIP's `BatchedDeltaNetRowSplitKernel<float, 16,
2>` splits a row over 8 lanes, so its k/q are ordinary vector loads prefetched a
token ahead; it also sums in a different order, which this port cannot adopt
without giving up bit-identity.)

`YAH_DN_STAGE=lds` (default): during token t the workgroup's 128 lanes fetch
token t+1's k, q and five scalars with lane-dependent (VMEM) loads and write
them to an LDS double buffer; token t reads its values back as broadcast
ds_loads, which return in order, and one barrier per token separates the
buffers. Same arithmetic, same order: bit-identical. Standalone 5.13 -> 3.32 ms
per layer; pipeline, interleaved, 265/272 -> 185/190 ms (HIP 94). Chunk grouping
G=2 stays best (G=0 4.1 ms; G=4 and G=8 spill, 9.1 and 67 ms).

### Attention: three heads per workgroup, V and P staged for contiguous fragments

`yah_attn_wmma` was 2.4x HIP (109 vs 46 ms). ATT at pp2048: 80% of wave time
in vmcnt waits and 9 global loads per WMMA -- the P*V value operand came from
the row-major V cache, and a B fragment needs 16 keys of one channel per lane,
so every fragment was 16 two-byte gathers. Staging each 64-key x 64-channel V
tile transposed in LDS (vt[channel][key], packed key pairs so a wave's stores
cover 64 banks) halved the per-wave time but only moved the kernel 6.8 -> 6.4
ms: with one query head per workgroup, every 16-row workgroup streamed its
whole causal K/V range, ~3.2 GB per call from MALL. HIP shares each K/V tile
across 64 rows.

`tools/gen_attn_heads.py H` regenerates the kernel with H query heads of one
GQA group per workgroup: each K fragment feeds H QK chains and each staged V
tile H P*V chains. Every head keeps its own 16-token tile, the same WMMA
chains, the same wave64 softmax over 64-key blocks and the same full/tail
split, so it is bit-identical (more tokens per workgroup would have moved the
full-block/tail boundary, and with it the softmax numerics). Two more fixes
from the traces: probabilities are staged row-major (the key-major pitch-18
layout made every P fragment 16 ds_load_u16, 32% of wave time), and V tile
t+1 is loaded while tile t computes (tile 0 before QK).

| standalone, ms/call | |
|---|---:|
| source kernel | 6.78 |
| + V staged transposed | 6.43 |
| H=3 | 4.91 |
| + row-major P | 4.24 |
| + V prefetch | 3.84 |

Pipeline, bit-identical: 109 -> 70 ms (HIP 46). H=3 is the largest that fits
64 KB of LDS (the tail stage aliases the V tile). `YAH_ATTN_HEADS=0` restores
the hand-written kernel; dispatch.txt carries H as `wmma.hal 16 <H> <tiles>`.

### DeltaNet in HIP's arithmetic order -- bit-identical to HIP's kernel

The regtile kernel gives each lane a whole 128-key state row and sums it as a
128-step sequential chain; within that order it was stuck at ~2x HIP. HIP's
`BatchedDeltaNetRowSplitKernel<float, 16, 2, false, false>` (the tile it runs
at pp2048) splits each row over 8 lanes of 16 keys and adds the partials with a
DPP butterfly. `tools/gen_deltanet_hip.py` ports it op for op, taken from its
compiled ISA (hipcc -O3, default FP contraction), because the fused
multiply-adds decide the rounding:

    s <- s*alpha;  t_g = fma(s.w,k.w, fma(s.z,k.z, fma(s.x,k.x, s.y*k.y)))
    u = (((0 + t0) + t1) + t2) + t3;  u += xor-1, xor-2, xor-4 partners
    d = beta * fma(-inv_k, u, v);  out = fma(q_scale, p, d * kq_dot)
    kq_dot = (inv_k*q_scale)*kq;  s <- fma(inv_k*d, k, s)

`tools/deltanet_vs_hip.sh` builds HIP's kernel into a harness
(`engine/tests/deltanet_hip_ref.hip`), runs both on the same random inputs and
compares output and final state with atol 0, plus a negative control that must
fail: bit-identical. k/q/v of token t+1 are loaded while token t computes.
Standalone 3.32 (regtile) -> 2.07 ms per layer (HIP 1.96 in its pipeline).

This changes the model's numerics (the whole-model reference md5 is now
1ae50a0e9a5a37a1; the kernel gate is the bit-exact comparison above). Against
HIP's last-token logits at pp2048: KL(HIP||Loom) 1.88e-8 -> 1.46e-8, max
|dlogit| 0.106 -> 0.082, argmax 11751 and the top-10 unchanged -- Loom moved
toward HIP. `YAH_DELTANET_HIP=0` restores the regtile kernel.

**Unrolled token loop.** Side by side, the ATT traces of the two kernels (same
arithmetic) differ in issue count, not in math: HIP is 71% vmcnt waits and 117
VALU per token-iteration per wave, Loom 75% VALU and 233. Per iteration Loom had
84 `v_fma_f32` and 60 `v_mov_b32` where HIP has 29 `v_dual_fmac` + 20 `v_fmac`
and no moves; the compile report's `move_causes` put 33 units on `branch_edge`:
the loop yields the new state and the prefetched k/q/v in fresh registers and
copies them back on every backedge. `unroll(%c4) schedule(recurrence)` on the
token loop lets the iterations alternate registers: static moves per token 81 ->
44, 2.058 -> 1.829 ms standalone (`unroll(%c2)` 1.857), still bit-identical to
HIP's kernel at 256 and 203 tokens. pp2048 pipeline 120 -> 104 ms (HIP 94). The
FMAs are still three-operand `v_fma_f32` (no in-place `v_fmac`, so no dual-issue
pairs); that and the remaining `low_slice` moves are the rest of the gap.

`loom-compile-report suggest` then named the next step
(`vector.compare_componentwise_bank_state`): the loop-carried k/q prefetch
vectors had static component reads plus one whole-vector consumer, the state
update `vector.fmaf(dk, k, s*alpha)`, so their boundary projection was
rejected. Written per element (the same four scalar fmas), the projection is
taken: 1.799 -> 1.716 ms; with `unroll(%c8)` (128 VGPRs, clearing suggest's
next finding, a residency cliff at 132) 1.663 ms. Bit-identical to HIP's kernel.
pp2048 DeltaNet 108 -> 98-100 ms in paired profiles (HIP 94).

### Attention in HIP's arithmetic order -- bit-identical to HIP's kernel

`tools/gen_attn_hip.py` ports HIP's `WmmaCausalAttention<32, 16, true>`: 32
query tokens x 2 heads of one GQA group per workgroup, 16-key tiles, S as two
8-step WMMA chains added in the softmax, the running max/sum and `__expf` as
hipcc compiles them, and HIP's scalar `fmaf` chain on the causal boundary
tiles. `tools/attn_vs_hip.sh [tokens]` builds HIP's kernels into a harness
(`engine/tests/attn_hip_ref.hip`) and compares every output element with atol
0, plus a negative control (token-major V bound where V^T is expected) that
must fail: exact at 200 (ragged) and 2048 tokens.

HIP reads V from a transposed copy (`PackAttentionHeads`); Loom does the same.
`yah_transpose_v16` (also in the generator, `vtrans.hal`) writes the layer's
V as [kv head][16-key tile][dim][16] f16 after the rope kernel, so a tile's
V^T is one contiguous 8 KB block. Plain [dim][token] rows at a 4 KB pitch were
slower than transposing in the kernel (3.38 vs 2.95 ms): 256 rows per tile at
a cache-aliasing stride.

Standalone, per layer at pp2048: first working port 4.54 ms; boundary chain
only for partly visible row blocks, P/V rows as 16-byte loads 3.32; epilogue
gate loads issued together 3.04; next tile's K/V carried in registers 2.95; V^T
2.67 + 0.01 transpose (HIP 2.86 in its pipeline, `gen_attn_heads.py` H=3 3.84).
In the pipeline (`YAH_LOOM_TIME=2`, 16 layers): attention 72.9 -> 49.0 ms plus
1.0 ms of transpose, HIP 46.0.

Numerics: hidden md5 a2145e371ceefd4d; against HIP's last-token logits
KL(HIP||Loom) 1.46e-8 -> 3.46e-9, max |dlogit| 0.082 -> 0.066, argmax 11751.
`YAH_ATTN_HIP=0` restores `gen_attn_heads.py`.

Two Loom pitfalls on the way: a lane-divergent `scf.if` holding LDS loads
inside the key loop lost lanes (0.3% of outputs written); predicate values
instead. The fully unrolled boundary chains (8 fragments x 8 rows x 16 keys)
exhausted the SGPRs; the rows are an `scf.for`.

**The reset this cost.** The driver's attention grid now comes from dispatch.txt
(32 tokens per workgroup). The driver edit was "built" with `cmake --build`,
which does not build `loom_forward_pp` (`engine/build_hrx.sh` does) and printed
nothing, so the 18:06 binary ran the new HAL set: 128 workgroups along x where
the kernel's launch contract declares 64. Loom proves bounds from that contract
and drops clamps it can show are redundant (`min(token, B-1)` on q, gate and
the output), so workgroups 64..127 read ~48 MiB past q/gate: unmapped VA, no
fault, gfx ring timeout, reset. Every declared footprint
(`loom_preflight.declared_envelopes`) fitted the bindings; the grid did not.
`gpu_run.sh` now refuses a `loom_forward_pp`/`hal_bench` binary older than its
source, and the driver refuses an attention grid that disagrees with the token
tiles the emitter recorded.

### Q3_K joins the tile GEMM

Q3_K was the last FFN format on the chained kStore (1.5x HIP). `q3k_loads` /
`q3k_compute` in `tools/gen_gemm_shared.py` decode a 32-element group per lane
with 16-wide vector integer ops -- low bits `(qs >> 2*(g%4)) & 3`, high bit
`(hmask >> g) & 1`, scales `2g` and `2g+1` from the packed 12-byte array -- and
keep the chained kernel's f32 order, `(f32(d) * f32(scale)) * f32(quant)`, so
the tile and shared kernels are bit-identical to it. Q3_K device time at pp2048,
interleaved: 175/185 -> 142/151 ms (HIP 122 ms for the same dispatches).

#### Decode one phase ahead, into a second weight tile

Cutting decode VALU did nothing on its own (IQ4_XS fused multiply 426 -> 387
static VALU, Q4_K word nibbles 622 -> 393: both within 1%), because the decode
sat between two barriers: the decoding waves (half of them) ran a
latency-bound chain while every wave's MMAs waited. HIP decodes the next tile
after its barrier, overlapped with the MMAs. `YAH_TG_DECAHEAD` does the same
inside the tile GEMM's schedule: phase p stores its activations, issues the
loads for p+2 (weights) and p+1 (activations), passes the barrier, and then the
decoding waves -- a wave-uniform branch on `kernel.subgroup.id`, since a
divergent `scf.if` right before the MMA loop is rejected
(`divergent_loop_single_entry`) -- decode phase p+1 into weight tile (p+1)%2
while every wave multiplies tile p%2. Two weight tiles fit at KSUB=32 (53 KB
with the epilogue slab). Same registers, same MMA order: bit-identical.

Standalone (17408x5120, 10240x5120 for Q4_K), ms: IQ4_XS 9.97 -> 9.71, Q4_K
6.18 -> 5.96 (with word nibbles; 6.33 without), Q5_K 11.85 -> 10.94, Q6_K 11.89
-> 11.55, Q3_K 11.41 -> 11.41, IQ3_S 10.10 -> 10.96, IQ3_XXS 10.07 -> 10.79 (their
IQ3 decode is the long pole: the ATT trace of IQ3_S with decode-ahead has LDS
stalls *down* (6.0% -> 1.3% of wave time, so not table contention) and one
barrier at 46%: at KSUB=32 a phase has one 32-element group per row, so only 4
waves decode (8 at KSUB=64) while each phase's MMA work halves, and IQ3_S's
decode, ~11 VALU per WMMA, no longer fits in the MMA shadow. Spreading a group
over two lanes would be the fix). Default on
for IQ4_XS, Q4_K, Q5_K, Q6_K (`DECAHEAD_FMTS`), not for the 16-row tiles (16
decoding lanes are not a wave), and not for IQ4_XS kres at K=6144, which lost in
two paired profiles (`DECAHEAD_SKIP`, emit_prefill_pp.py). The Q4_K/Q5_K word
nibbles (`(w >> s) & 0x0f0f0f0f` on i32, `vector.uitofp` -> `v_cvt_f32_ubyteN`)
are now the default too.

pp2048 device time, paired: TOTAL 3702 -> 3659 ms (HIP 3479, 1.064x -> 1.052x),
hidden md5 unchanged. IQ4_XS gate+up -12, Q4_K resid -8/-5, Q4_K stores -6/-4/-3,
IQ4_XS resid (K=17408) -6.

Note: a clean emit differs from the shipping set in 11 GEMM HALs (IQ2_XS and
Q2_K kStores, the unfused `gemm_residual_*`): the shipping set carries tuned
variants from earlier env-driven builds, and the clean IQ2_XS gate+up is 43 ms
slower. The decode-ahead set was assembled from the shipping set plus the
regenerated IQ4_XS/Q4_K/Q5_K/Q6_K HALs.

#### One header load per block

The decoders fetched a block's header fields one by one -- IQ4_XS `d`,
`scales_h`, `scales_l[g/2]`; Q4_K `d`, `dmin` and three scale bytes per group --
each a 1-2 byte load with its own index arithmetic and clamp. One vector load of
the header (8 bytes for IQ4_XS, 16 for Q4_K, both aligned in their blocks) plus
shifts and masks gives the same values (`YAH_TG_IQ4HDR`, `YAH_TG_Q4HDR`).
Q4_K's trace: VALU per WMMA 10.4 -> 6.6, VMEM ops per WMMA 0.57 -> 0.32;
standalone Q4_K 5.93 -> 5.47, IQ4_XS 9.60 -> 9.45 ms. Q5_K is neutral (10.76 ->
10.86, 160 -> 144 VGPRs) and keeps its loads. pp2048 paired (candidate second):
3598 -> 3578 ms, hidden md5 unchanged.

The same two fixes for Q3_K and Q5_K. Q3_K (`YAH_TG_Q3W`): scales[12] and d
from one 16-byte load at block offset 94, and the 2+1-bit quant assembled on
32-bit words -- `(w >> 2sp) & 0x03030303`, `((hm >> g) & 0x01010101) << 2`, and
the -4 as `(x | 0x80808080) - 0x04040404 ^ 0x80808080` so no borrow crosses a
byte -- where the i8-vector form lowered per element: 11.49 -> 10.32 ms
standalone (decode-ahead on top: 10.61, so it stays off for Q3_K). Q5_K: its
fifth bit on words (`((qh >> g) & 0x01010101) << 4`) 10.76 -> 10.19, and then the
header load, neutral before (the per-byte bit path was the limit), takes it to
9.78. pp2048 paired: the Q3_K/Q5_K rows -18.9 ms while every other row drifted
+30.6 (the second-run bias), hidden md5 unchanged.

IQ2_XS joins the tile GEMM (`iq2xs_loads`/`iq2xs_compute`: four 16-bit codes
per group, 9-bit grid index into 64-bit grid entries, 7-bit ksigns index,
nibble scale per 16 elements, the chained kernel's f32 order): its one pp2048
gate+up dispatch 20.0 -> 11.5 ms (HIP 10.8), hidden md5 unchanged. The
emitter's footprint gate now bounds the grid binding per format as the driver
allocates it (IQ2_XS 4096 B; the flat 2048 refused it).

IQ3 sign application on grid words (`YAH_SD_VDECW`: s1 = the sign nibble times
0x00204081 masked to one bit per byte, mag = (g ^ 255*s1) + s1 -- exact since
every grid magnitude is > 0): static VALU -22% for both IQ3 formats but IQ3_XXS
9.97 -> 9.82, IQ3_S 10.16 -> 10.23 ms -- not decode-VALU-bound; on for IQ3_XXS.
IQ4_XS nibbles on words changed nothing (the constant shift was already
packed), and masking them on i32 hid the index-range fact the codebook lookup
needs for `v_perm`: 960 `v_cndmask`, 9.44 -> 13.21 ms. Off.

Also measured with it: spreading an IQ4_XS group over 2 or 4 lanes
(`YAH_TG_SPLIT`) removes the decode imbalance (the top barrier falls from 49%
to 20% of wave time at 4) but makes the SIMDs VALU-issue-bound (VALU stall
45%): 9.55 / 9.84 ms alone, 9.65 / 9.96 with the header load. The phase-loop
unroll still loses with the header load (10.14): its remaining full drains
are `read_result_reuse` on other loads.

#### Tile GEMM ideas measured and dropped

- Grouped launch order (runs of 4 row groups x all token tiles): 10.25 -> 10.12
  ms standalone, +83/+90 ms on the pp2048 GEMM total. Kept as `YAH_TILE_SWZ`, off.
- Token-major grid: 10.3 -> 11.2 ms standalone (the activation tile stops being
  shared by consecutive workgroups).
- Half-group decode (two lanes per IQ4_XS group, all lanes decoding): 10.27 ->
  10.95 ms -- each lane repeats the group's loads and scale math.
- Double-buffered LDS at KSUB=32 (the only size that fits 64 KB at 128x256):
  IQ4_XS 10.24 -> 9.92, Q4_K 6.40 -> 6.36 standalone, but +160 ms in the
  pipeline and not bit-identical there; not pursued.
- Activation row pitch padding (`YAH_TG_AGPAD`, f16; h3-hrx found 1024-byte
  multiple pitches alias in cache, and ours are 10240 / 34816 B): IQ4_XS at
  K=5120 9.96 -> 10.18/10.29 ms (pad 32/64), at K=17408 9.97 -> 9.92/9.58/9.52/
  9.91 (pad 16/32/48/64). ~4% on the down projection only, and every producer
  would have to write the padded pitch; the knob stays, off.
- The barrier before the MMA loop is 26% of IQ4_XS wave time (ATT): at KSUB=64
  a phase has 128 rows x 2 groups = 256 decode jobs for 512 lanes, so waves
  0-7 decode while 8-15 wait. Spreading it: 256 x 128 tiles (WM=8, WN=2, all
  512 lanes decode, 128 VGPRs, bit-identical) 9.97 -> 10.31 ms -- halving BN
  doubles the decode per MMA (VALU 31% -> 48% of wave time). KSUB=32 (128 VGPRs,
  tier 8) 10.61. Inner k-loop `unroll(%c4)` 10.48, with `schedule(interleaved)`
  11.68. Waves sit round-robin on the SIMDs, so every SIMD has decoding waves;
  and on RDNA3 WMMA issues through the VALU pipe, so decode and MMA compete for
  the same issue slots either way.
- HIP's IQ4_XS decode is 3.6-3.9 VALU per WMMA to our 6.6: an f16 codebook
  assembled with six `v_perm`s per 8 values, then one `v_fma_mix{lo,hi}_f16`
  per element (one rounding). Loom selects `v_fma_mix` for scalar
  `fptrunc(mulf<contract|nnan|nsz>)` feeding `vector.from_elements`
  (`YAH_TG_IQ4MULF`), but not for the vector form; an f16 `vector.table.lookup`
  scalarizes (1520 static VALU), i8 `vector.interleave` is rejected on gfx11,
  and zext/shift/or assembly costs more than it saves (`YAH_TG_IQ4F16`: 10.21
  ms). The int8 codebook with the fused multiply (HIP's rounding, not
  bit-identical to ours) cuts static VALU 426 -> 387 and times 10.08 vs 10.10:
  instruction count is not the limit. Both knobs stay, off.
- 32 waves of 32x32 wave tiles (WM=4, WN=8; 8 waves per SIMD from one
  workgroup, what the residency tier would buy): IQ4_XS 9.47 -> 12.56, Q4_K
  5.49 -> 6.53 ms -- one fragment load per WMMA instead of 0.75, and 4 of 32
  waves decoding. The GEMMs' own residency cliff (144 VGPRs, 128 for tier 8)
  is allocation, not pressure: the scheduled peak is 126-135 but 8-register
  fragments pack to a 144 high-water; rhs-outer MMA order with fences gets 136.
- swiglu epilogue on the LDS epilogue's structure (`YAH_TG_SWEPI=1`: one
  barrier, wave-private slabs, 4-row vector gate loads and f16 stores, the same
  scalar silu per element; md5 unchanged): standalone IQ4_XS 10.22 -> 10.03,
  IQ3_S 10.99 -> 10.64 ms, but neutral in the pipeline (gate+up rows -2.8 /
  +2.6 / -2.5 ms in an order-swapped pair) and it costs VGPRs (IQ3_S 136 ->
  190) and code (6.5 -> 16 KB). Off. The first pair had shown +24 ms: the
  second of two back-to-back pp2048 profiles runs ~30-40 ms slower whichever
  set it is, so pairs are now run in both orders. Retested on the
  decode-ahead kernels: gate+up rows +1.7..+2.2% against +0.3% drift on the
  other rows -- a real pipeline regression, standalone gains notwithstanding.
- The IQ3 grid tables read from global memory instead of LDS (`YAH_SD_GRID_LDS=0`,
  as HIP's `__device__` tables): IQ3_S 10.10 -> 11.74, IQ3_XXS 10.07 -> 11.73 ms
  standalone; with decode-ahead 12.08 / 11.51. Traced: s_waitcnt 16.5% -> 27.2%
  of wave time, the new stall a `vmcnt(2)` in the decoding waves only -- the
  eight lookups per group are a dependent chain (bytes -> index -> table ->
  element), and a global L0 hit is slower than a ds_read, with too few waves to
  hide it.
- K-phase loop `unroll(%c2) schedule(recurrence)` (`YAH_TG_PPOL`, the move fix
  that worked for DeltaNet): IQ4_XS 9.60 -> 10.14, Q4_K 5.97 -> 6.48 ms. Moves
  per phase do fall (56 -> 36), but the trace shows two new `s_waitcnt
  vmcnt(0)` sites inside the MMA loops (12.8% + 6.8% of wave time): in the
  unrolled body the prefetched global loads are drained before the multiplies
  instead of at the next phase's LDS stores.
- IQ4_XS structure probes (with decode-ahead and the header load, 9.47 ms
  standalone), each traced:
  - dedicated decode waves (`YAH_TG_DECW=4`: 16 MMA waves + 4 decode-only, 640
    threads; the MMA waves never decode) 9.73: the barrier falls 47% -> 33% of
    wave time but VALU stall rises to 33% -- the SIMDs are issue-bound, so what
    counts is total VALU per WMMA, not which wave issues it;
  - the 2-step inner loop unrolled (`YAH_TG_KPOL=unroll(%c2)`) 9.85 (Q4_K 5.49 ->
    6.02): VALU per WMMA 7.40 -> 6.15 (the per-step address math halves), but a
    new full `vmcnt(0)` takes 27% of wave time -- the compile report's extra
    `amdgpu.ssa_use` wait has a register copy of a carried prefetch value as
    consumer: the copy into the loop-carried register now sits before the MMAs
    and drains the prefetch. Same mechanism as the phase-loop unroll;
  - grouped launch order, in the pipeline this time (`YAH_TILE_SWZ=8` on the
    IQ4_XS HALs): IQ4_XS rows +4.0% against +0.4% drift on the others. Our tiles
    reuse the activation tile across all 136 row groups in plain order;
  - weights copied into a device-local buffer instead of importing the GGUF
    mapping (`YAH_LOOM_WEIGHTS_DEVICE=1`): pp2048 3590 -> 3706 ms, slower.
  Standalone kernels run ~20% faster than the same HAL in the pp2048 pipeline
  (IQ4_XS 6144x5120: 3.5 vs 4.2 ms; DeltaNet 1.80 vs 2.17). Not cold weights (4
  rotating 16.7 MB weight buffers: 3.52 ms), and clocks explain only part: the
  pipeline settles at ~2.2 GHz / ~125 W, a sustained standalone run at p50 2.26
  GHz (peaks 2.46) under 110 W. HIP's kernels pay the same, so pipeline rows are
  the only fair comparison.
- wave64 tile GEMM (`YAH_TG_W64=1`: accumulators `vector<4xf32>`, operand
  fragments unchanged, direct epilogue), IQ4_XS 17408x5120 with decode-ahead,
  all bit-identical, against 9.71 ms wave32: 8 waves of 32x128 (152 VGPRs)
  10.97, 8 waves of 64x64 (160) 11.02, 16 waves of 32x64 in a 1024-lane
  workgroup (96 VGPRs) 13.28.
- Load cache hints (`{cache_scope = cu, cache_temporal =
  non_temporal_high_temporal}`, TH_LOAD_NT_HT on gfx12): Loom's gfx11 encoding
  (`gfx11_glc_slc_dlc`) accepts only device/regular and drops anything else
  silently -- the ISA is byte-identical. Nothing to measure on gfx1151.

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

### IQ4_XS, measured instead of inferred (2026-09-30)

**Target.** Everything compiles for `gfx1151` now. `gfx11-generic`'s residency
model is the 1024-VGPR RDNA3 parts (granule 16, so 144 VGPRs read as tier 7:
one 16-wave workgroup per WGP); gfx1151 has the full 1536-VGPR file (granule
24). The hardware ran two workgroups either way (ATT: 7.9 resident waves per
SIMD), and the whole pp2048 set measured the same on both targets (ABBA
3606.8 vs 3622.8 ms, md5 unchanged); the new default costs nothing.

**Real data.** Standalone cases used constant fills. The kernel runs the same
cycle count on any data (26.7 M cycles for 17408x5120x2048), but real weights
and activations drop the clock from ~2.56 to ~1.93 GHz. So ms against HIP on
synthetic data compared clocks, not kernels. Real layer-4 bytes now come from
`YAH_DUMP_LAYER=4` (`.fn`) plus the GGUF tensor; `gemm_bench` takes
`YAH_BENCH_W/UP/A` files. Same bytes, 20 launches: HIP 9.00 ms, Loom 10.40.

**Profiling Loom code with rocprofv3.** rocprofv3 cannot see HRX dispatches,
and HRX's counter backend has no occupancy or LDS counters. `engine/gpu/loomhip.hip`
(`build_gpu.sh loomhip`) launches a Loom hsaco through HIP, which gives the
full rocprofv3 set: PMC, derived occupancy, per-wave ATT timelines. Its output
for the IQ4_XS kstore is bit-identical to the pipeline's layer-4 gate.

**Diagnosis (per-wave ATT, same tool on both kernels).** Same occupancy (8.0
waves/SIMD) and the same WMMA count per wave (2560). Loom needs 37.3 time
units per WMMA per SIMD against HIP's 30.3. HIP's waves queue on a busy
matrix pipe (27 units/WMMA stalled on WMMA issue); ours find it idle and wait
at barriers instead (86.5 vs 23.0 units/WMMA). Per barrier interval the four
decoding waves run 771 units of VALU and the twelve MMA-only waves wait 835
at the barrier. The end-of-phase barrier's last arriver spends 61% of its
last 40 instructions in `s_waitcnt`, mostly `lgkmcnt(1)` right after the 7
fragment loads each k-step issues: after every barrier all ~32 waves on a WGP
load their fragments at once, and the matrix pipe idles for that queue.
HIP pays one barrier per 16 WMMAs per wave, we pay one per 8.

**Tried against that diagnosis, all bit-identical, one round each, real data:**
- decode spread over all 16 waves (`YAH_TG_SPLIT=4`, decode-ahead kept):
  10.83 vs 10.57 ms. Every wave decodes evenly (483 units VALU) but still
  waits 676 at each barrier, so the imbalance explained who waits, not why
  the phase is long.
- Loom read-ahead on the fragment loop (`YAH_TG_KPOL='pipeline(%c2)'`):
  interleaves step 1's loads under step 0's WMMAs, 160 VGPRs, 10.58 vs 10.37.
  The burst after the barrier stays.
- one barrier per phase with a double-buffered activation tile
  (`YAH_TG_ONEBAR=1`): no change. MMAs queued across the barrier HIP's way
  (`YAH_TG_XBAR=1`): 160 VGPRs, +18%.
- HIP's shape (256x256, 32 waves of 32x64, K=64 per stage, all 1024 lanes
  decode 16 weights, fragment-major LDS tiles, fully unrolled K loop):
  12.49 ms. Its K loop now interleaves loads and WMMAs much like HIP's, but
  `s_waitcnt` costs 80.4 units per WMMA against HIP's 14.1, spread over every
  `lgkmcnt` in the loop. HIP keeps loads several WMMAs ahead at 192 VGPRs;
  Loom's `pipeline(%c2)` on this shape went to 208 VGPRs (too many for a
  32-wave workgroup) and still waited `lgkmcnt(0)` before its WMMAs.
  (Fixed on the way: fragment-major 8- and 16-wide stores now prove their
  alignment and stay b128.)

**Attribution in GPU cycles (phase-pricing ablations, 2026-09-30).** The same
ablations on both kernels, real layer-4 bytes, `GRBM_GUI_ACTIVE` per dispatch
(rocprofv3 --pmc; Loom through `loomhip`; HIP via `gemm_bench ablate:iq4xs`,
Loom via `YAH_TG_ABL`, same bits). WMMA floor 17.83 M cycles:

| variant | HIP M cycles | Loom M cycles | Loom/HIP |
|---|---:|---:|---:|
| control | 23.80 | 27.46 | 1.154 |
| noBarrier (1) | 22.11 | 30.06 | 1.360 |
| noCommit (2): no decode, no LDS stores | 19.43 | 23.21 | 1.194 |
| noFetch (4) | 21.23 | 26.16 | 1.233 |
| noStore (8): no epilogue store | 23.61 | 26.70 | 1.131 |

**Correction (2026-10-01):** HIP's `noCommit` row is not comparable. With
nothing written to LDS the compiler deleted HIP's fragment loads and then the
fetch that fed them (its ablated kernel has 0 `ds_load` and 0 `global_load`;
both operands of every WMMA are `v[0:7]`), so it measures bare WMMA
throughput, while ours still loads fragments. The same holds for the stacked
variants. HIP's `noBarrier`, `noFetch` and `noStore` kernels keep all their
work (48 `ds_load`, the decode's `v_perm`s, the fetch) and do compare: there
we are 1.37x, 1.25x and 1.14x. Removing the fetch saves HIP 2.5 M cycles and
us 0.9 M. Removing the barriers saves HIP 1.7 M and costs us 2.7 M. How the
gap splits between decode and the rest is still open; a fair split needs a
no-decode variant that keeps the LDS traffic on both sides (HIP's `Fp16W`
ablation does that on its side).

Per-wave ATT views of the same kernels (scratch tools `critpath.py`,
`simdtl.py`): on the traced SIMD HIP's matrix pipe is busy 52.9% of the time
and ours 43.0%; we lose it mostly to fully idle time (36.7% vs 30.4%: more
barrier waiting, plus the decoders' dependent VALU chains stalling). The
barrier-interval critical path is 31% decode VALU for us, where HIP's is
mostly WMMA. Next: price the skeleton pieces separately (LDS fragment reads
vs barriers vs epilogue), starting with the 3.8x epilogue.

**The epilogue slab held every LDS bank conflict (2026-10-01).** Stacked
ablations in robust cycles (`SQ_BUSY_CYCLES` per SQ; `GRBM_GUI_ACTIVE` reads 0
in some captures and `SQ_WAVES` is unreliable): with commit, fetch, store and
barriers all removed, HIP's K loop runs at 99% of the WMMA floor and ours at
84% (87% with `pipeline(%c2)` or a full unroll). rocprofv3's LDS counters,
which HRX cannot read, then showed ~20 M bank-conflict cycles per dispatch
for us against HIP's 0. Without the LDS epilogue: 0, and our bare skeleton
at 91%. (HIP's 99% "bare loop" turned out to have no fragment loads left, see
the correction above, so 91% vs 99% is not a like-for-like comparison.) The slab stored element (r, t) at `t*TM + r`, so the 16 lanes writing a
fragment row were 128 B apart. A token pitch of TM+4 floats (`YAH_TG_EPAD`,
default 4; keeps the b128 read-back aligned, LDS 52 -> 56 KiB) cuts the
conflicts 94% and IQ4_XS to 26.39 M cycles (from 26.77). Bit-identical. pp2048,
every tile GEMM re-emitted, one round each: 3598.7 -> 3537.1 ms; GEMM rows
-58.8 ms against -2.5 on the rest (IQ3_S gate+up -15.6, IQ3_S residual -10.0,
Q3_K -5.7, IQ4_XS -4.5). IQ3_XXS did not move.

**IQ3_S vs IQ4_XS, and the raster order (2026-10-01).** Same activations,
real weights, cycles: IQ3_S Loom 27.96 M (64% of the WMMA floor) vs HIP
25.91 M (69%), 1.079x; IQ4_XS Loom 26.45 M (67%) vs HIP 23.15 M (77%),
1.143x. Our two kernels are about equally efficient; HIP's IQ4_XS kernel is
the outlier. Its IQ4_XS-specific grouped raster (`group_shift=5`, credited
+12% there) does not transfer: `YAH_TG_SWZ` 17/34/68 (divisors of the 136 row
groups) costs +1-2% in cycles, and SWZ=34 on the IQ4_XS 17408-row kernels in
the pipeline went 385.0 -> 394.1 ms (one round each, other rows +0.9%). A
3-launch ms reading that looked faster was clock noise. The swizzle now clamps
the row group, so a group size that does not divide the row groups gives wrong
results instead of out-of-bounds stores.

**A tolerance gate for arithmetic-changing work (2026-10-01).** The hidden md5
stays the gate for changes that claim to keep the arithmetic. Changes that
alter it go through `engine/run/accgate.py`: a frozen golden output (the md5-
verified production run: last-token logits + the 2048 x 5120 final hidden
state) and thresholds calibrated on HIP. HIP vs golden: last-token KL 3.4e-9,
logits relative RMS 1.24e-3, same top-10 and argmax. A candidate must stay
within 0.25 of that (KL <= 8.6e-10, logits rel RMS <= 3.1e-4); the hidden-state
limits reuse the logits scale and are provisional, since HIP writes no hidden
state. `accgate.py calibrate <golden> <hip.logits>`, `accgate.py check <golden>
<candidate>`.

First uses:
- The fused `fma_mix` IQ4 decode (`YAH_TG_IQ4MULF='<contract>'`) is bit-identical.
  The IQ4_XS scale is an f16 times a 6-bit integer (<= 17 significant bits),
  times a 7-bit codebook value: the f32 product is exact, so one rounding and
  two give the same f16. The md5 gate never blocked it.
- Packed f16 decode (`YAH_TG_IQ4PK`: scale rounded to f16 per group, the
  codebook looked up as f16, `v_pk_mul_f16`) fails: last-token KL 3.9e-9 (HIP's
  own distance), hidden rel RMS 1.7e-2, 259 of 2048 tokens off by > 1% and
  some by ~30%. One rounded scale per 32-weight group is a shared bias that
  compounds through 64 layers. It is also slower (26.98 vs 26.38 M cycles): the
  byte-table join lowers per element where HIP's uses one `v_perm` per pair.

**Causal accounting, and the fused IQ4_XS decode (2026-10-01).** Instead of
trying variants, every instruction and every unit of wave time was attributed
to a role:

- Instructions per role come from the final low IR, weighted by block trip
  counts (`causal_loom.py`), or from ATT hit counts by source line for HIP
  (`causal_hip.py`). Both close against SQ_INSTS_* to within the post-RA
  copies.
- Time is attributed along the critical path: for each barrier interval, the
  last wave to arrive, split by role (`critrole.py`).

The decoding wave was the straggler: its decode VALU was 29% of Loom's
critical path against 13.5% for HIP. HIP looks up the f16 bit patterns of the
codebook and needs one `v_fma_mix` per weight. Loom's in-loop decode block
took 254 VALU per 32 weights: sign-extend, `v_cvt_f32_i32`, mul,
`v_cvt_f16_f32`.

Two exact rewrites, now the default:

- `YAH_TG_IQ4MULF=<contract|nnan|nsz>` fuses the multiply and the rounding
  into `v_fma_mix{lo,hi}` (215 VALU). Plain `<contract>` does not fuse.
- `YAH_TG_IQ4U8F=1` looks up the codebook +128 as unsigned bytes and folds the
  -128 into the fma's f32 addend: `fptrunc(fma(s, u, -128*s))`.
  `v_cvt_f32_ubyteN` reads the byte directly, giving 182 VALU with no bfe,
  ashr or i32 converts. (u-128)*s has at most 24 significant bits, so it is
  exact in f32 and the f16 rounding is the same.

Results:

- Bit-identical on real bytes. The pp2048 hidden md5 is unchanged and
  argmax stays 11751.
- SQ_INSTS_VALU per WMMA 8.39 -> 7.26.
- kstore 17408x5120: 26.40 -> 25.96 M cycles (mf alone 26.02).
- kres down projection on real layer-33 `ffn_down` (input = layer-4 swiglu
  output, `chain.sh`/`krcyc.sh`): 26.34 -> 25.94.
- kres K=6144 on real `ssm_out`: 13.24 -> 13.26, neutral.
- pp2048, one round each: IQ4_XS rows -0.70% against +0.64% drift on the
  others. Production set p43 = p42 + these 12 HALs.

What the model says now:

- ATT slows this kernel ~1.5x. The WMMA spacing is 16 units = 32 cycles, so
  simdtl's 37% "idle" is mostly tracing overhead. Use ATT for relative
  attribution only.
- Untraced, the SIMD is close to issue-bound: cycles are about
  (32 per WMMA + ~1 per other VALU) / utilization. Predicted and measured
  savings:
  - mf: -0.34 M predicted, -0.38 M measured.
  - u8f: -0.63 M predicted, -0.44 M measured.
- Every non-WMMA VALU costs SIMD time whichever wave issues it, which is why
  SPLIT=4 and DECW=4 (placement only) lost. The remaining gap to HIP (25.78
  vs 23.17 M under PMC) is about 56% VALU (6.26 vs 3.63 non-WMMA VALU per
  WMMA) and 44% utilization.
- The largest VALU line left is the K-step address math: 13 VALU per 8
  WMMAs, two of them quarter-rate `v_mul_lo_u32`. It is recomputed from
  loop-invariant lane and buffer terms on every step. Next: the fma_mix
  pair's 30 moves (a zero `v_mov` before each tied `mixlo`, plus copies into
  the `ds_store_b128` tuple).

**Straight-line k steps (`YAH_TG_KSL`, with `YAH_TG_DECLOAD`), default for
IQ4_XS (2026-10-01).**

The cause:

- The rolled 2-step k loop recomputed the fragment loads' lane and base
  addresses on every step: `(lane&15)*80`, `wr*2560 + base` and `wt*5120`.
  That is 13 VALU per 8 WMMAs, two of them quarter-rate `v_mul_lo_u32`.
- `fragment.load` lowers its address math at the load site, in source-to-low,
  and the pipeline has no loop-invariant motion.

The fix:

- `KSL` writes the steps straight-line in program order, with a schedule
  fence between them (unlike PF, there is no read-ahead and VGPRs stay at
  144). Both steps then share one block, low CSE merges the math to 8 VALU
  per 16 WMMAs, and the k offset becomes an immediate.
- Alone it put the prefetch's latch copies between the two steps, behind a
  `vmcnt(0)`: kstore 25.96 -> 25.00 M cycles.
- `DECLOAD` (only decoding waves issue the weight loads) moves the copies and
  their wait to after the last MMA: 24.50 M (HIP 23.15).
- `DECLATE` instead put the wait before step 0.

Results:

- Bit-identical; md5 a2145e371ceefd4d.
- pp2048, one round each: IQ4_XS rows 712.6 -> 675.2 ms (-5.25%) against
  -0.73% drift on the others (p43 -> p44).
- The same rolled address math is in every tile GEMM. The other formats keep
  the old loop until measured (`KSL_FMTS`).

**KSL on the other formats.** Standalone, M cycles on real bytes, one round:

| format (real tensor) | baseline | KSL | KSL + DECLOAD |
|---|---:|---:|---:|
| Q4_K (`attn_qkv`) | 15.49 | 15.83 | 15.21 |
| IQ3_S (gate) | 28.07 | 27.06 | (no decode-ahead) |
| IQ3_XXS (gate) | 26.99 | 25.99 | (no decode-ahead) |
| Q3_K (`blk.54.ffn_gate`) | 29.31 | 28.62 | (no decode-ahead) |
| Q5_K (`blk.27.attn_q`) | 19.34 | 19.59 | 19.60 |

- Q5_K stays off:
  - KSL alone puts the latch copies' `vmcnt(0)` between the steps.
  - With DECLOAD, the allocator, at the 144-VGPR cap, interleaves ~48 moves
    between the MMA pairs.
- IQ3_S kres does not gain (real `ffn_down` 28.74 -> 28.69, `ssm_out`
  14.29 -> 14.40). The kres harness (`krfmt.sh`) uses the layer-4 swiglu
  output as input.
- The md5 gate passed on every kind.
- pp2048 p44 -> p45, one round each: Q4_K/IQ3_S/IQ3_XXS/Q3_K rows
  2447.5 -> 2418.1 ms (-1.20%) against +1.22% drift on the others.
  - The IQ3_S kres row read +11.6 ms. Standalone does not reproduce it
    (see above).

**Kernel parity in cycles, same real bytes (2026-10-01).** Per-kernel ratios
against the old HIP trace (2026-09-30) were stale; HIP's plain pp2048 swings
3289-3538 ms between sessions. Measured kernel against kernel instead
(rocprofv3, Loom through loomhip):

| kernel | Loom M cycles (% WMMA floor) | HIP M cycles (% floor) | Loom/HIP |
|---|---:|---:|---:|
| IQ4_XS 17408x5120 | 26.45 (67%) | 23.15 (77%) | 1.14 |
| Q4_K 10240x5120 | 15.50 (68%) | 13.71 (77%) | 1.13 |
| IQ3_S 17408x5120 | 27.96 (64%) | 25.91 (69%) | 1.08 |
| IQ3_XXS 17408x5120 | 26.99 (66%) | 25.80 (69%) | 1.05 |
| DeltaNet B=2048 | 4.756 | 4.762 | 1.00 |

Ours sit at 64-68% of the floor for every format. HIP sits at 69% except on
the two formats where it defers the decode (raw bytes held in registers,
decoded by all waves after the barrier while queued WMMAs drain): 77%.
DeltaNet is at parity, so the HIP-order constraint costs nothing there.
`YAH_TG_DECLATE` (off) moves the decoders' next weight loads after their
decode so the old and new prefetch are never live together: bit-identical,
the back-edge `vmcnt(0)` (0.9% of wave time) goes, a `vmcnt(1)` (2.8%) comes
(the late loads queue behind the activation loads), 26.43 M cycles: flat.

## 7. Decode and the HIP removal

### Loom decode (2026-10-02): GEMV-based, prefill handoff

`engine/model/loom_decoder.hpp` (`LoomDecoder`) is the single-token step, driven
by `engine/run/loom_decode.cc` (prompt fed through decode) or by
`loom_forward_pp` after a prefill (`YAH_GEN=N YAH_DECODE_HAL=<set>`). The HAL set
comes from `tools/emit_decode.py <model> <dir> <max_context>`:

- `tools/gen_gemv.py`: one GEMV per (kind, formats, M, K) on the shard. HIP's
  sub16 decode for Q2_K..Q8_0 and IQ2_XXS / IQ2_XS / IQ3_XXS / IQ3_S / IQ4_XS,
  done on 32-bit words to unsigned byte codes (biases folded into the offset
  term), two rows per wave sharing x, `pipeline(2)` read-ahead, butterfly
  reduce. Kinds: plain, resid (residual add fused), swiglu (gate + up fused).
  `tools/gemv_check.py`: every (format, K) against a float64 gguf-py oracle,
  worst 1.7e-6.
- `tools/gen_decode_attn.py`: attention over the prefill's paged fp16 pools
  (kvappend; split-K part, one workgroup per (kv head, 256-key page) with the
  six GQA heads; reduce with the sigmoid gate). `tools/dattn_check.py` checks
  it against numpy with scrambled page tables.
- `tools/gen_decode_misc.py`: 512-lane RMSNorm and DeltaNet decode (the ports'
  math), and the IQ4_XS embedding row from a device token stream.
- The step is enqueued back to back: argmax writes the next token into the
  stream the next embedding reads, positions come from a device array, so the
  host waits only after the prompt and at the end. Independent input
  projections go without an ordering barrier.

Prefill handoff: a chunked prefill set (`YAH_CTX`) may now process fewer tokens
than it was emitted for (a multiple of the chunk), leaving pool room; the
decoder binds the prefill's K / V^T pools, page table, conv state (saved after
the last chunk too) and DeltaNet state, whose layouts it shares. The decode
set's context must equal the prefill pool rows.

Results (IQ4_XS shard, 64 greedy tokens, every token identical to HIP's):

| context after prefill | Loom | HIP |
| --- | ---: | ---: |
| 5 (prompt fed through decode) | 16.43 tok/s (60.9 ms) | 14.58 tok/s (68.6 ms) |
| 2048 | 15.97 tok/s (62.6 ms) | 14.07 tok/s (71.1 ms) |
| 8192 | 15.16 tok/s (65.9 ms) | 13.46 tok/s (74.3 ms) |
| 30720 (arXiv 2608.13365) | 13.31 tok/s (75.2 ms) | 11.75 tok/s (85.1 ms) |

Roofline: 12.40 GB of weights per token / 240 GB/s = 51.7 ms. The GEMV
families run at 214-236 GB/s.

What mattered, in order (each measured on the full decode):

- **Low-bit GEMVs were VALU-bound.** `vector<16xi8>` shifts and masks lower
  byte by byte (unpack, shift, repack), ~240 VALU per 16 weights on IQ3_S; a
  VALU-count model predicted the measured 115-175 GB/s per format. Word-level
  decode: 77.1 -> 68.2 ms.
- **Read-ahead.** `pipeline(2)` on the GEMV sub-block loop: 68.2 -> 63.1 ms
  (depth 3 66.1, 4 67.8: the queue costs registers).
- **No host round trip** (GPU embedding, device token stream): -0.9 ms;
  barrier-free independent projections: -1.4 ms.
- **Attention at long context.** Read-ahead on the K and V^T loops (depth 3 / 2)
  and a (head, page) grid order so the four heads of a page read each 2 KB K
  row together: 128 MB of K / V per call at 32K went 2955 -> 689 us
  (43 -> 195 GB/s). Standalone timings below ~32 MB are flattered by the
  32 MB MALL; benchmark attention at 32K.
- **Band-fused input projections** (`gen_gemv.gen_bands`, one dispatch per
  layer for qkv / gate / alpha / beta or q / k / v): 234-237 GB/s, 176 fewer
  dispatches per token, wall time neutral (61.17 vs 61.20 ms): the separate
  no-barrier dispatches were already ~92% efficient.
- Where the rest goes (61.1 ms, short context): GEMVs 54.9 ms (50.9 at
  240 GB/s; SwiGLU ~2.2 ms and the residual GEMVs ~0.9 ms of the excess),
  dependent-dispatch gaps 3.2 ms (563 x ~4 us), DeltaNet 1.9 ms (~1.3 floor).
  HRX graphs do not shorten the gaps: 3.32 us per dependent node replayed vs
  3.57 us on the stream (yah-scratch/decode/probe/graph_probe.cc); only fewer
  dependent dispatches do. An explicit fma chain for the 16-element dot gave
  the same VALU count (the compiler already contracts mul + reduce).
- **RMSNorm folded into the residual GEMV** (`resid_norm`: the last workgroup,
  found with a device-scope acq_rel counter, writes the next layer's normed
  input; stress-tested over 51 back-to-back dispatches): neutral, 61.30 vs
  61.18 ms. 128 fewer dispatches (gaps 3.20 -> 2.76 ms) but each call grows
  ~5 us by the serial norm tail. Off by default (`YAH_DEC_RESNORM=1`).
- **DeltaNet:** state loads hoisted above the q / k norm phase (61.2 -> 60.87 ms)
  and the decode conv fused in (`deltanet_conv`, 60.87 -> 60.59): lanes 0..127
  convolve their q / k / v channels; the conv state ping-pongs between two
  buffers per token, so the three value heads sharing a key head never read a
  half-updated state. Bit-identical to the separate conv + DeltaNet.
- No gain, measured: SwiGLU rows / waves per workgroup (1x4 .. 2x8, all within
  0.2 ms: geometry is not the limit), an LDS sign-mask table in place of the
  quarter-rate v_mul_lo_u32 (67 -> 7 per kernel, neutral: not VALU-bound after
  word decode), index.assume in place of load clamps (-4% VALU, neutral).
- **32-element groups per lane** (`YAH_GV_G2`, header loads shared by both
  sub-blocks): 11-31% fewer VALU and 14-43% fewer loads per unit of work, but
  60-75 -> 103-114 VGPRs; slower (61.77 vs 60.48 ms), the occupancy loss
  outweighs the saved issue. Residual GEMV geometry (R x W): the default 2 x 4
  wins by 1-1.5 ms over 1 x 4, 2 x 8, 1 x 8, 4 x 4.
- **Persistent megakernel, feasibility (not built).** A grid barrier inside a
  resident kernel (atomic arrive with release, spin on an acquire load; spin
  capped so missing residency cannot hang; `research/gen_gbar.py`,
  `hal_run HAL_RUN_TIME1`) costs 0.26 us at 20-40 workgroups, 0.37 at 80, 0.49 at
  160, vs ~3.5-4 us per dependent dispatch: at most ~1.3 ms/token to gain from
  the ~436 boundaries. But persistent GEMVs (`YAH_GV_PERSIST=G`, workgroups
  looping over row groups) lose: SwiGLU +3.7% at G = 640 (static round-robin's
  ragged last round), 61.67 vs 60.48 ms overall; one-group-per-workgroup
  kernels unchanged. With that loss and the megakernel's register count being
  the maximum over all its phases, the expected net is ~0; not pursued.
- Profiling: `HRX_PROFILE_MODE=dispatch` (timestamps only) costs ~1%;
  counters mode inflates dispatch gaps ~4x. A kernel right after a no-barrier
  group shows the group's tail in its own duration.

The previous decode runner (`yah-hrx`, removed) drove the prefill GEMM HALs with a
64-token tile per generated token (1.46 tok/s).

### History

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

**Attention at long context: row scales and LDS footprint (2026-10-01).**

pp8192 per-kernel parity put attention at 1.33x HIP (748 vs 562 ms), the
largest ratio and growing with context.

The setup:

- `YAH_DUMP_LAYER=3` now also dumps the attention inputs (roped q, gate, the
  layer's f16 K/V slots) and its output.
- Real layer-3 pp8192 inputs went through Loom (loomhip, `attnpmc.sh`) and
  HIP's kernel (`engine/tests/attn_hip_ref.hip`). The outputs are
  bit-identical, so all the extra work is overhead.

Counters, one dispatch:

| | Loom | HIP |
|---|---:|---:|
| cycles | 106.6 M | 79.4 M |
| VALU | 1.825 G | 1.119 G |
| LDS instructions | 575 M | 369 M |
| LDS bank-conflict cycles | 205 M | 113 M |
| resident waves (SQ_WAVE_CYCLES/SQ_BUSY_CYCLES) | 16.0 | 23.2 |

ATT showed where it goes. The hottest region (36% of latency) is the
per-tile O rescale before P.V:

- 32 `ds_load_b32` per tile from a 16x-replicated 64x16 row-scale table,
  each group behind a 30-45 cycle `lgkmcnt` wait.
- 48 latch `v_mov` copies of accumulators rescaled out of place.

Also:

- No VOPD pairing.
- LDS 52,992 B per workgroup (HIP 20,992) means two workgroups per WGP,
  so 4 waves/SIMD against HIP's 6.

Fixes, both bit-identical and now the defaults:

- `YAH_ATTN_RS2`: a compact 64-float row-scale table `[rb][half][i]` = row
  rb*16 + 2i + half. A lane reads its fragment's 8 scales with one
  vector<8xf32> load. 106.6 -> 100.3 M cycles, VALU -16%, LDS -30%, and the
  8-byte scratch spill is gone.
- `YAH_ATTN_LDS2`: V^T first, K next to the scores, and the 16 KB
  boundary-tile scratch aliasing K + S, which are dead during P.V. LDS
  36,864 B gives three workgroups per WGP. 100.3 -> 86.5 M cycles; resident
  waves 16.0 -> 23.6.

Results:

- pp2048: md5 a2145e371ceefd4d unchanged; the attention row 51.7 -> 43.5 ms
  (-15.9%) against +1.25% drift on the rest. That is under HIP's 48.6.
- The rest of the gap at 8192 is the 33 latch copies per tile (accumulators
  not coalesced across the back edge) and unpaired VALU.

The pp8192 emit needed the attention token-count facts to follow B
(`YAH_ATTN_MAX_TOKENS`, 09ed5c4).

**Q4_K and the remaining copies; loop-invariant hoisting (2026-10-01).**

Q4_K kstore on real `attn_qkv` bytes (M=10240): Loom 15.38 M cycles against
HIP 13.64 (1.128x). Occupancy and LDS traffic are equal. Per WMMA:

| | Loom | HIP |
|---|---:|---:|
| VALU | 7.08 | 4.37 |
| SALU | 2.26 | 0.24 |
| `v_mov` (ATT) | 1.64 | 0.05 |
| barrier share of latency | 50% | 22% |

The copies are the carried weight prefetch bouncing between two register sets
every phase: 12 + 12 moves per 16 WMMAs. Old and new prefetch are live
together, so the allocator's edge-alias check refuses the relocation
(traced with LOOM_EXP_RELOC_DEBUG in the worktree).

Measured, all bit-identical:

- `DECLATE` (loads after the decode): 15.91. The decoder waves wait
  `vmcnt(0)` right after issuing.
- `YAH_TG_P2` (the phase loop unrolled 2x in the generated text, off):
  removes the back-edge ping-pong, but loses.
  - Without DECLOAD: 16.82. Body B's prefetch is copied into the carried
    registers behind a `vmcnt(0)` between its k-steps. The relocation pass
    meets a permutation chain (12->18, ..., 30->14) blocked at v20/v21 and
    refuses the whole group.
  - With DECLOAD: 18.01. The merge copies sit behind a wait after body A.

The fix needs the allocator to relocate a chain jointly. That is not done
yet.

Hoisting (worktree compiler, `low-licm`, LOOM_EXP_LICM=1, local commit
9ab6a00):

- What it does: moves pure, effect-free VALU/SALU ops (no state writes, exec
  unchanged in the loop) whose operands are loop-invariant into the
  preheader.
- IQ4_XS kstore: 24.50 -> 23.66 M cycles (HIP 23.15). The k-loop goes from
  13 to 8 VALU, the quarter-rate `v_mul_lo` leave, and VGPRs 144 -> 160
  (LDS-bound occupancy, so free).
- Neutral or worse elsewhere: IQ3_S 27.06 -> 27.03, IQ3_XXS 25.99 -> 25.88,
  Q4_K 15.21 -> 15.26, Q3_K 28.62 -> 29.20.
- Attention (45 of 204 VALU per tile are invariant) spills at any setting
  that hoists VALU, because it sits at its 232-VGPR ceiling. It needs
  registers freed first.

**Superseded, not production:** production artifacts must build with stock
HRX, and p47 needed the experimental compiler. For the record, p47 = p46 +
the IQ4_XS kstore/swiglu/kres HALs built with the worktree compiler and
hoisting on:

    YAH_LOOM_HOME=/home/q/hrx-wt LOOM_EXP_LICM=1 rollout.sh <base> iq4xs

That compiler is HRX branch loom-wait-sched, local commit 9ab6a00; the HALs
run on the stock runtime. md5 a2145e371ceefd4d unchanged. pp2048, one round
each: IQ4_XS rows 673.2 -> 659.7 ms (-2.0%) against +0.8% drift.

**Attention QK^T fence (`YAH_ATTN_QKFENCE`, default 1), stock compiler.**

The cause: at the VGPR peak, 5 of the 8 K fragments of the QK^T chain were
live, because the scheduler issued their loads early (40 VGPRs).

The fix: a schedule fence after each (K fragment load, MMA) pair.

- VGPRs 232 -> 216.
- pp8192 real layer-3 inputs: 86.52 -> 80.23 M cycles (HIP 79.4, so 1.01x).
  A fence every 2 pairs gives 80.76, every 4 gives 81.94.
- Bit-identical to HIP's kernel.
- p48 = p46 + this wmma.hal: md5 a2145e371ceefd4d unchanged, attention row
  44.4 -> 41.3 ms (-7.0%) against +0.75% drift.

Hoisting the remaining loop-invariant address math would bring VALU to HIP's
count (1.126 vs 1.119 G), but needs compiler changes (see p47). Production
stays stock-buildable.

**Issue-bound view and the IQ3 decode (2026-10-01, after the research pass).**

Every production kernel runs at 98-102% of its issue bound:
34·WMMA + VALU cycles + ~3·LDS per SIMD (`research/sol.py` with the extended
counters; research/hw-measured.md). Barrier waits are absorbed by other waves,
so only instructions per WMMA move cycles. Two cautions:

- Dependent VALU chains cost 2-3x when every wave decodes in the same phase.
- A decode-free f16 GEMM is falsified: HIP's f16-weight Q4_K kernel is 8%
  slower than its quantized one.

`YAH_SD_IQ3U8F` + `YAH_SD_VDECW_FR`, default for IQ3_XXS:

- What it does: the signed magnitude bytes XOR 0x80 are read with
  `v_cvt_f32_ubyteN`, and the -128 rides the fused multiply's f32 addend
  (`fptrunc(fma(dsc, u, -128*dsc))`; the product is exact in f32).
- The sign spread drops its quarter-rate `v_mul_lo_u32`: n*0x204081 becomes
  two shift-ORs, and *255 becomes a shift-subtract.
- Bit-identical. IQ3_XXS kstore on real bytes 25.99 -> 25.41 M cycles.
- p49 = p48 + the IQ3_XXS HALs: md5 a2145e371ceefd4d unchanged. pp2048
  IQ3_XXS rows 766.9 -> 750.8 ms (-2.1%) against +1.6% drift.

IQ3_S with the same path loses (26.79 -> 27.23). VALU/WMMA falls 7.64 -> 6.58,
but the issue bound drops from 98% to 94% of measured: the longer dependent
chain is exposed between barriers.

**Wave layout 4 x 2 for IQ4_XS (32 x 128 per wave, 8 waves; `WAVE_FMTS`).**

- The cause: LDS instructions cost ~3 cycles each next to WMMA, and fragment
  loads are 2·(FM+FN) per FM·FN WMMAs per k step.
- The 4 x 2 layout cuts LDS per WMMA 1.72 -> 1.47 and VALU 5.52 -> 4.82 (less
  address math per WMMA).
- Kstore 24.50 -> 23.12 M cycles (HIP 23.15), bit-identical.
- 2 x 4 (64 x 64) has fewer instructions still (LDS 1.19) but falls to 93% of
  its issue bound, so latency is exposed: 25.88.
- p50 = p49 + IQ4_XS kstore/swiglu/kres at 4 x 2: md5 a2145e371ceefd4d
  unchanged. pp2048 IQ4_XS rows 680.7 -> 654.7 ms (-3.8%) against +0.9%
  drift.

4 x 2 waves on other formats (standalone kstore, real bytes, bit-identical):

| format | 4 x 4 | 4 x 2 | |
|---|---:|---:|---|
| IQ3_XXS | 25.41 | 24.23 | win |
| Q3_K | 28.62 | 25.12 | win |
| IQ3_S | 27.06 | 52.78 | spills (256 VGPRs, 58 scratch instructions) |
| Q4_K | 15.21 | 44.05 | spills (240 VGPRs, 65 scratch instructions) |

- IQ3_XXS swiglu loses at 4 x 2 (27.39 -> 27.76; pp2048 row +8.6 ms), so the
  default is per kind.
- p52 = p50 + IQ3_XXS kstore/kres and Q3_K at 4 x 2: md5 unchanged. pp2048
  rows 861.9 -> 819.7 ms (-4.9%) against +1.1% drift.

**Pipeline tax and host CPU (2026-10-01).**

- **No cycle tax.** With HRX counters in the real pipeline
  (`HRX_PROFILE_MODE=counters`, a local profiling-only libhrx addition), the
  kernels run the same cycles as standalone: IQ4_XS kstore 23.05 vs 23.12 M,
  IQ3_S 27.01 vs 27.06, IQ3_XXS 24.21 vs 24.23, Q4_K 15.26 vs 15.21.
- **The ms tax is clock.** gpu_metrics during pp2048: gfx clock median
  ~2270 MHz (min 1630; allowed max pulled to ~2480 of 2900), GPU 95-98 C,
  socket ~115 W, of which the GPU is ~38 W.
- **The host keeps the GPU fed.** GPU idle 0.5% of the prefill; dispatch
  gaps have a median of 9 us. Launch overhead is negligible.
- **The host spin was the final wait, now fixed.** One core ran at ~104% for
  the whole prefill (HIP too): ~400-460k `AMDKFD_IOC_WAIT_EVENTS` ioctls.
  - Where: `YAH_LOOM_DISPATCH_TIMING` attributes host gaps to the next
    dispatch. All 947 dispatches enqueue in ~1 ms each; 3356 of the 3368 ms
    of gaps sit before the head's `yah_rmsnorm`, i.e. in the post-layer
    `gpu.Synchronize()`, where ROCr's wait busy-polls. (An earlier note
    placed it inside the layer loop; that was wrong.)
  - Why the first sleep-poll did nothing: `hrx_stream_query` reports
    complete while `stream->timepoint == 0`, which it is for plain
    dispatches, so it fell straight through to the spinning wait.
  - Fix: `LoomDevice::SetSleepSync(us)` records an event at the stream tail
    and sleep-polls `hrx_event_query`. `loom_forward_pp` turns it on at
    200 us (not under `YAH_LOOM_TIME`); `YAH_LOOM_SLEEP_SYNC_US=N`
    overrides it (0 = runtime wait). Probes keep the runtime wait.
  - Result (one round each, 15 s gap): host CPU 104% -> 3-4%, process CPU
    3.97 s -> 0.56 s, socket 111 -> 102-107 W. md5 a2145e371ceefd4d.
- **Freeing that power does not raise the GPU clock.** The GPU is
  hotspot-thermal-limited, not power-limited. `thr_thm_gfx` climbs at a
  constant rate from the first 50 ms sample, and Tgfx is pinned at 93-98 C
  from t=0 (the hotspot saturates within milliseconds). The PPT limits never
  fire (fppt +7 in a spin run, 0 with sleep-sync). Clock and layers_ms move
  within run-to-run noise (2094-2311 MHz, 3457-3526 ms). Clock comes only
  from less GPU energy per unit of work.
- **Load time (warm page cache):** device 180 ms, weights import 250-450
  ms, buffers 55 ms, embed + first HALs 20 ms, exit ~110 ms (`YAH_TRACE_LOAD`
  prints `[phase]` lines). The import is the kernel faulting 3.2M 4 KB PTEs
  and registering them with amdgpu.
  - **Folio size of the page cache sets the import cost.** Same 3.2M faults:
    245-450 ms after a normal mmap cold load, 830 ms after warming the cache
    with `cat` (small read() folios).
  - **Lost: prefault on a thread overlapping device init**
    (`MADV_POPULATE_READ`). Device init went 180 -> 437 ms; the populate
    holds `mmap_lock`, which ROCr's init mmaps wait on.
  - **Lost: `MADV_HUGEPAGE` on the GGUF mapping.** Warm: import -80 ms.
    Cold: +0.6 s, more major faults.
  - **Lost: `MADV_POPULATE_READ` before the import.** -30 ms.
  - **Lost: weights as a 2 MB-page THP copy** (`YAH_WEIGHTS_COPY`-style) or
    a device-local copy (`YAH_LOOM_WEIGHTS_DEVICE`). layers_ms 3505 / 3522
    vs 3527 (noise); load +5 s / +43 s. Weight TLB reach is not a factor:
    the GEMMs are issue-bound and reuse each weight tile across 2048 tokens.
  - What remains is driver cost (ROCr init, KFD userptr registration). It is
    not reachable from our code with the stock runtime.

**IQ3_S at 4 x 2 with the word-path decode (2026-10-01).** 4 x 2 spilled with
IQ3_S's element decode (256 VGPRs), and the word-path U8F/FR decode lost at
4 x 4 (exposed latency). Together they fit (224-248 VGPRs, no scratch), and
the decode block drops from 467 VALU per 32 WMMA to 353 per 64. Standalone,
real bytes, M cycles, one round each, bit-identical:

| kind | p52 | p53 |
|---|---:|---:|
| kstore 17408x5120 | 27.08 | 24.15 (HIP 25.91) |
| swiglu | 28.52 | 27.76 |
| kres K=17408 | 28.90 | 24.18 |
| kres K=6144 | 14.16 | 9.86 |

- p53 = p52 + IQ3_S kstore/swiglu/kres (`WAVE_FMTS`; `configure()` turns the
  word path on for IQ3_S at 4 x 2). md5 a2145e371ceefd4d on every kind.
- Clean pp2048, one round each, 15 s gaps: p52 3431.1 ms, p53 3347.6 ms
  (-2.4%). HIP 3463.3 ms in a separate round.
  - HIP's first round in that sequence read 16686 ms. No GPU messages in
    dmesg; it did not reproduce (3463, then 3436 with `YAH_WEIGHTS_COPY`).
    Cause unverified; possibly outside GPU use.

**pp8192 with the p53 kernels (2026-10-01).** Re-emitted at B=8192 with the
current generator (`/home/q/yah-hal-p53-8192`); hidden md5 e94924b79ae21e57,
equal to clean8192c's.

- Untraced ms, one round each, 15 s gaps, in order: clean8192c 14877.0,
  p53-8192 16246.6, HIP 16532.0. Under HRX counters: p53-8192 15378.7, then
  clean8192c 15677.7. The ms ordering follows run order, not set.
- Cycles (SQ_BUSY_CYCLES summed per kernel, both sets under counters):
  32704.5 -> 30621.9 M (-6.37%).
  - IQ3_S rows -9 to -29%.
  - IQ3_XXS kstore/kres, IQ4_XS and Q3_K rows -4 to -12% (the p50/p52 4 x 2
    layouts, also new to the 8192 set).
  - Attention, DeltaNet and Q4_K unchanged (±0.3%).
- Why ms lies at 8192: a run is ~15 s at full load, so a 15 s gap does not
  return the GPU to the same thermal state. The second run of a pair starts
  hotter (the GPU is hotspot-thermal-limited, see above).
- With 30 s gaps (one round each, same order): clean8192c 15169.5,
  p53-8192 14260.2 (-6.0%, in line with cycles), HIP 15709.3. pp8192 timing
  uses 30 s gaps from here.

**Swiglu: the LDS epilogue for IQ3 (2026-10-01).** SOL counters, real bytes:
IQ3_S swiglu ran at 90% of its issue bound (kstore 100%), +3.3 M cycles over
kstore of which only ~0.5 M is extra instructions. The cause is the scalar
epilogue (`swiglu_epilogue`): one dependent global gate load per element in a
rolled loop, behind two barriers per slab. With one workgroup per WGP nothing
overlaps it. `YAH_TG_SWEPI` (vector gate loads through the LDS epilogue,
measured neutral in ms back at p40) re-measured in cycles, bit-identical:

| swiglu | scalar epilogue | LDS epilogue |
|---|---:|---:|
| IQ3_S (4 x 2) | 27.56 | 24.85 |
| IQ3_XXS 4 x 4 | 27.27 | 27.04 |
| IQ3_XXS 4 x 2 | (lost before) | 25.13 |
| IQ4_XS (4 x 2) | 24.81 | 24.71 (left off) |

- Default for IQ3_S/IQ3_XXS (`SWEPI_FMTS`); IQ3_XXS swiglu moves to 4 x 2.
- p54 = p53 + those two swiglu HALs: md5 a2145e371ceefd4d. pp2048 3316.4 ->
  3285.8 ms (one round each, 15 s gaps).
- pp8192 (p54-8192, md5 e94924b79ae21e57 unchanged): cycles 30776.3 ->
  30454.3 M (-1.05%; swiglu rows -9.2 / -7.5%). Untraced ms at 30 s gaps read
  13900.5 (p53, first run after an idle emit) vs 15584.7: at 8192 the first
  run after idle is far cooler than any later one, so ms needs cycles beside it.

**Q4_K at 4 x 2: blocked by back-edge allocation, not pressure (2026-10-01).**
Every 4 x 2 / 2 x 4 variant spills (compile-only, `--compile-report=text-details`
spill rows), even at 171-208 VGPRs with peak live 171-190:

- What spills: carried K-loop values. These are 2-4 accumulator tuples, the
  carried global prefetch (`nxw*` from the DECLOAD `scf.if`, or `cv4-10`
  without decode-ahead), plus a few loop invariants. Each is stored twice and
  reloaded once or twice: the allocator cannot place the yielded values in the
  loop-argument registers. This is the same back-edge problem that costs Q4_K
  24 `v_mov` per phase at 4 x 4. 11 repair iterations vs 0 for IQ4_XS at 4 x 2.
- Not the cause: KSL, DECLOAD, decode-ahead, the 16-byte header (each off:
  still spills), P2 phase unroll (spills), 2 x 4 (spills).
- Found on the way: in KSL the whole step's fragments (2 weight + 8 activation,
  80 units) were live at the peak. `YAH_TG_RHSO=1` now works in the KSL path
  (each activation fragment loaded right before its MMAs; `YAH_TG_RHSF=n`
  fences every n): peak 190 -> 171, VGPRs 208 -> 184, spills unchanged. Off by
  default; defaults emit identical text.
- A fix needs either allocator work (excluded: stock compiler) or a K loop
  whose global prefetch is not loop-carried. Upside is ~0.5% of the prefill
  (Q4_K ~9.5% of cycles; 4 x 2 gave IQ4_XS -5.6%, Q3_K -12%). Parked.

**Q4_K at 4 x 2: the v0..v127 conversion window (2026-10-01).** The 4 x 2
spills above were not back-edge allocation. Found by reading the stock
allocator (diagnosis only):

- Q4_K's decode narrows with `vector.subf` + `vector.fptrunc`, which lowers to
  `v_cvt_f16_f32`. That instruction's result can only be v0..v127
  (`descriptors/alu.py` operand window; `allocation/target_constraints.c`
  `apply_operand_window` caps the interval at 128).
- At 4 x 2 the 16 accumulators are exactly 128 VGPRs, live across the decode,
  and the linear scan fills low registers first. So every conversion result
  evicts an accumulator (`interval_assignment.c` `find_free_location` fails ->
  victim search), and the evictions become scratch spill storage (`frame.c`).
- The register budget is 256; the final count (176-208) was not a cap.
- IQ4_XS never hit it: its decode narrows through `v_fma_mix{lo,hi}`, which can
  address all 256 VGPRs.
- Fix (`YAH_SD_Q4FMIX`, default for Q4_K): per element
  `fptrunc(fma(e, one, -dm))`. The product by 1 is exact, so it is
  bit-identical. `one` is built as `bitcast((gb & ~gb) | 0x3f800000)` because
  a provable 1.0 is folded back to `subf` by the canonicalizer.
- With it, the default (carrying) K loop fits at 4 x 2 (200-248 VGPRs, no
  scratch). The no-carry loop (`YAH_TG_NOCARRY`, kept, off) was not needed and
  loses on its own: IQ4_XS 23.26 -> 24.76, Q4_K 4 x 4 15.26 -> 15.56. Its
  loads issued at the phase top are less hidden, and `suggest` shows 17/27 LDS
  waits fully draining.

Real bytes, M cycles, bit-identical (1 s gaps for single kernels from here):

| Q4_K | p54 | p55 |
|---|---:|---:|
| kstore 10240x5120 | 15.21 | 13.54 (HIP 13.64-13.71) |
| swiglu | 27.52 | 24.85 |
| kres K=17408 | 25.75 | 23.71 |
| kres K=6144 | 12.06-12.13 | 11.95-12.35 (neutral) |

- 4 x 2 + `RHSO`/`RHSF=1` (`RHSO_FMTS`): 13.77 -> 13.54.
- p55 = p54 + Q4_K kstore/swiglu/kres: md5 a2145e371ceefd4d. pp2048 3290.8 ->
  3262.8 ms (-0.85%).
- pp8192 (p55-8192, md5 e94924b79ae21e57): cycles 30381.1 -> 30081.4 M
  (-0.99%). Q4_K rows -7.7..-12%. Unchanged kernels moved up to ±5% between
  captures (IQ3_S swiglu 1864.5 vs 1961.8 M on the same HAL), so per-row
  cycle noise is ~5%.
- Any decode that narrows with `v_cvt_f16_f32` inside a loop holding 128 VGPRs
  of accumulators will hit the same window (Q5_K shares this decode; Q6_K,
  Q2_K, IQ2_* unchecked).

**Speed-of-light table in cycles, p56 pp2048 (2026-10-01).**
`research/sol2k.py` pairs per-dispatch SQ_BUSY_CYCLES (HRX counters,
`iree-profile counter --counter_samples`) with the `YAH_LOOM_SEQ` shapes. GEMM
floor = (M/16)(B/16)(K/16) WMMAs x 34 / 80 SIMDs. This view is clock-free.

- Total 7465 M cycles: WMMA kernels 6876 M (92%) at 76% of their floor;
  non-WMMA 589 M (8%).
- Big 17408-row / K=17408 GEMMs sit at 75-83% of floor. Q4_K plain is at
  81-83%, IQ4_XS plain 80-82%.
- Weak shapes:
  - K=6144 residual GEMMs at 60-72%: IQ4_XS 60%, Q4_K 64%, IQ3_XXS 70%,
    IQ3_S 72%. Short K leaves the prologue/epilogue less amortized.
  - Q5_K 1024-row at 46%.
  - Attention at 46%.
- Non-WMMA: DeltaNet 226 M (3.0%), half_norm 99, ssm_conv 85,
  ssm_postnorm 64, unpack_qg 31.

**What limits the GPU clock (2026-10-01).** Measured with
`research/clockprobe.hip` (5 s of one instruction mix) and `research/clocklog.py`
(gpu_metrics v3 + hwmon at 20 ms), 30 s gaps. Steady state:

| load | clock MHz | SMU cap | Tgfx C | socket W | throttle |
|---|---:|---:|---:|---:|---|
| WMMA only | 2622 | 2640 | 94.8 | 113 | thm_gfx |
| VALU FMA only | 2067 | 2092 | 94.8 | 101 | thm_gfx |
| LDS only | 2634 | 2665 | 94.8 | 98 | thm_gfx |
| memory stream | 2841 | 2900 | 67.1 | 62 | none |
| IQ3_S GEMM (ours) | 2276 | 2330 | 94.8 | 111 | thm_gfx |

- The limiter is the SMU's GFX thermal controller holding Tgfx at ~95 C.
  - Tgfx is a fast local junction sensor: 47 -> 87 C within ~30 ms of load,
    97 C by ~90 ms. `current_gfx_maxfreq` (the enforced ceiling) drops exactly
    then; thm_gfx residency counts ~1000/s (once per metrics update).
  - The cap never drops before the temperature does, so a current limit
    (EDC/TDC: present in firmware, not exported) is not the sustained limiter.
  - Fast PPT acts only in the first ~160 ms (the 2.9 -> 2.66 step).
  - STAPM/SPL, slow PPT and PROCHOT stay at 0.
- The sustained clock is set by heat per cycle. VALU FMA is the hottest
  (2.07 GHz), WMMA 2.62, our GEMM (WMMA + decode VALU + LDS) 2.28. Fewer
  instructions per WMMA raise the clock as well as cutting cycles.
- Not a sampling artifact: `time_filter_alphavalue` = 1 s smooths every
  `average_*` field, but the cycle-based clock (SQ_BUSY_CYCLES / device ticks)
  and the cap agree. hwmon vddgfx is hard-coded 0; `pp_power_profile_mode`
  and `power1_cap` are not implemented for SMU 14.0.x; amd-smi throttle fields
  are N/A.
- Cooling helps only through the base temperature. Our runs rise ~45-50 C
  locally on top of a 41-49 C base. Pre-run Tgfx vs ceiling correlates
  r = -0.54 (40 C -> 2478 MHz, 49.5 C -> 2322 MHz). A custom fan curve reportedly
  holds 78 C / 2820 MHz on Strix Halo (nathanmarlor/strix-halo-fan-control),
  likely on lighter loads.
- Levers not tried (system changes, the user's call):
  - fan curve
  - `ryzenadj --tctl-temp`: needs root plus `ryzen_smu` (DKMS built only for
    6.18) or `iomem=relaxed`
  - GFX undervolt: `--set-cogfx` reportedly does not work on Strix Halo
- `doas ryzenadj --tctl-temp=105` (user's request) is accepted through the SMU
  mailbox. The PM table is unreadable (no `ryzen_smu` for 7.3, `/dev/mem`
  blocked), so the old value is unknown; it resets on reboot. The firmware
  holds Tgfx at ~99.6 C instead of 94.8, i.e. it clamps near Tjmax (100 C).
  - GEMM probe: 2276 -> 2425 MHz (+6.5%), 10.79 -> 10.25 ms/launch.
  - VALU: 2067 -> 2188 MHz.
  - Socket power 111 -> 124 W.
  - pp2048 p56: 3186.2 ms (md5 unchanged), HIP 3388.4 (one round each,
    15 s gaps). Earlier same session: p55 3262.8, HIP ~3463.
- **tctl 105 trips PROCHOT on sustained load; use 99.**
  - Fan curve set (user): 35:0, 44:46, 55:112, 64:139, 72:167, 79:196, 85:255,
    95:255 (old: ..., 86:219, 95:248). The EC ramps the fans 5300 -> 8700 rpm
    over ~18 s, too slow to matter for a 3 s prefill.
  - At tctl 105, after ~4-6 s with Tgfx at ~100 C, PROCHOT (external, likely
    the EC at APU ~99 C) fires every ~6 s for ~1 s and pins GFX at 600 MHz.
    20 s GEMM: mean cap 2021 MHz, PROCHOT 3660.
  - At tctl 99: hold ~99 C, no PROCHOT in 20 s, mean cap 2173 MHz.
  - pp2048 p56 at tctl 99, starting from APU 54 C: 3255.6 ms (HIP 3616.9,
    run second). Right after sustained probes (APU 80 C): 3408.2. Start
    temperature moves pp2048 by several percent, so compare only from a
    matched idle temperature.
- Power breakdown (`research/powerlog.py`, gpu_metrics, medians after 2 s):
  - Socket: idle 13.8 W, WMMA 106, VALU 110, LDS 105, memory stream 69,
    IQ3_S GEMM 116.
  - The sub-domains (gfx / all_core / sys) do not add up to socket (the
    remainder runs -26..+58 W) and gfx swings 27 W (WMMA) .. 93 W (LDS) at
    similar socket power. They look like SMU model estimates; only socket
    power is trustworthy.
  - DRAM read/write counters read 37+36 for a 208 GB/s stream (unknown units).
  - Under load fclk ~2.0 GHz, uclk 1.0 GHz, socclk 1.45 GHz; Tsoc ~74 C while
    Tgfx sits at 100.
  - Real per-rail power needs the PM table (ryzenadj; blocked by /dev/mem on
    this kernel without `iomem=relaxed` or a `ryzen_smu` build for 7.3).

**Where the power goes: ryzen_smu PM table (2026-10-01).** ryzen_smu
(187.0bb95d9) built for 7.3.0-rc4-perfopt:
- Built against a clean copy of the tree with the running kernel's
  `/proc/config.gz` (clang ThinLTO) and `Module.symvers`, at
  `~/.cache/kbuild-perfopt`. One-line fix: `<asm/cpuid/api.h>` on >= 6.15.
- Signed with the kernel build key; installed to `extra/` with
  `modules-load.d`. Source in `~/yah-scratch/ryzen_smu-7.3`.

`research/pmlog.py` samples the raw table (float32[916], table 0x64020c) at
50 ms through idle and the five probe loads. Fields matched by behaviour
(ryzenadj maps only limits/temperatures for this version):

| idx | idle | WMMA | VALU | LDS | mem | GEMM | reading |
|---|---:|---:|---:|---:|---:|---:|---|
| 1/3/5 | 8 | 112 | 110 | 109 | 61 | 118 | STAPM / fast / slow PPT value (W) |
| 13 (lim 12 = 120) | 4.6 | 99.7 | 113.8 | 93.1 | 34.8 | 111.6 | main compute rail (TDC-style limit 120) |
| 203 | 2.6 | 97.9 | 95.0 | 93.5 | 35.2 | 98.2 | smoothed twin of 13 |
| 17 (lim 16 = 40) | 1.2 | 3.8 | 3.8 | 3.8 | 9.6 | 6.6 | SoC/memory rail (rises with DRAM traffic) |
| 33 | 0.83 | 1.08 | 0.95 | 1.10 | 1.05 | 0.99 | voltage-like (V) |
| 22/23, 340 | 95/36 | 95/95 | 95/95 | 95/95 | 95/62 | 95/95 | GFX temperature limit / value |
| 342/343 | 688 | 2730 | 2324 | 2792 | 2900 | 2452 | GFX target / effective clock (MHz) |
| 348-351 | | 2000 | 1000 | 2000 | 8000 | | fclk, uclk, -, memory MT/s |
| 53 | 1.4 | 32 | 112 | 112 | 29 | 53 | the gpu_metrics-style GFX power estimate |

- Socket power = 1.06 x [203] + 9.4 W over every sample of all six runs
  (rms 5.6 W). The compute rail carries nearly all the variable power: about
  98 of 117 W in the IQ3_S GEMM, on a ~9 W base (SoC, fabric, memory, idle
  CPU).
- The memory stream instead loads the SoC side ([17] and friends).
- [53] (and gpu_metrics' gfx power) fits with a 30 W offset and 31 W rms: an
  activity model, not a measurement.
- [13] reaches 113.8 against its limit of 120 on VALU FMA. The thermal limit
  binds first in all our runs (thm_gfx counters).
- PM table rail decomposition (added a 5 s all-core CPU load, `openssl speed
  -multi 32`): socket = 1.01 x [203] + 0.56 x sum(per-core [740..755]) +
  2.0 x [17] + 2.8 W, rms 1.5 W over all seven runs.
  - Per-core fields: power [740..755], voltage [756..771], temperature
    [772..787], clock [788..803] (4.86 / 4.55 GHz by CCD), C-state
    residency [820..867].
  - CPU rail [15] (limit [14] = 80): CPU load 71, GEMM 4.3.
  - The big rail [13]/[203] also draws ~60 W under the CPU-only load, so it
    is not GPU-only (possibly a shared IOD/fabric VDD); identity unconfirmed.
  - IQ3_S GEMM: ~99 of 118 W on [203], CPU cores 5 W, SoC side ~7 W.

**Short-K residual GEMMs: lockstep epilogue bursts, workgroup stagger
(2026-10-01).**

- SOL counters: kres at K=17408 runs at 100-102% of its issue bound, at
  K=6144 IQ4_XS 81% and IQ3_S 89%.
- ATT (`tools/wavephase.py`) on IQ4_XS kres K=6144: the 32 waves of the traced
  SIMD run in 8 lockstep rounds (2 workgroups per WGP start and end
  together). 14.5% of SIMD time has no wave in its K loop. 80% of the
  epilogue is the 32 residual `global_load` per wave stalling at issue: every
  resident workgroup reads its 128 KB residual tile at once.
- `YAH_TG_STAGGER=N` (default 8000; `STG_SEL=pair`): the second workgroup on
  each WGP in the first round (linear ids [20, 40)) runs N workgroup barriers
  before starting. That gives ~380k cycles of offset, past the ~150-200k
  epilogue burst. It applies only on grids of >= 320 workgroups (runtime
  check), since it is a one-time cost. Bit-identical.
  - ATT: time with no wave in its K loop 14.5% -> 0.7%, summed epilogue
    4.73 -> 2.91 M units.
  - Standalone M cycles (none -> 8000): IQ4_XS kres K=6144 10.27 -> 9.51,
    IQ3_S kres K=6144 9.71 -> 9.57, IQ3_S kstore 24.14 -> 23.83, IQ4_XS
    swiglu 24.62 -> 24.25, IQ3_S kres K=17408 neutral.
- Production-set pitfall: `tools/try.sh` (and `rollout.sh`) compiles one
  generated kernel for every shape of a format/kind, ignoring the emitter's
  per-shape choices.
  - p50-p56 therefore ran IQ4_XS kres K=6144 with decode-ahead, which
    `DECAHEAD_SKIP` excludes (12.79 vs 10.88 M cycles).
  - The 48-row `kstore_*_3_20` kernels also differ from the emitter's.
  - A full re-emit with the stagger off reproduces p56 on all other 77 HALs.
  - Production sets are now built by a full `emit_prefill_pp.py` emit.
- p57 = full emit (stagger on): md5 a2145e371ceefd4d. pp2048 3153.9 ->
  3142.7 ms (one round each).
  - Cycles (counters, one capture each): total 7505 -> 7366 M (-1.84%),
    GEMMs -1.94%.
  - IQ4_XS kres K=6144 -21% (decode-ahead fix + stagger), Q4_K kres K=6144
    -6.2%, swiglus -1.7..-3.5%.
- p57-8192 emitted with the same generator.

**DeltaNet: dual-issue FMAs via `vector.dotf` (2026-10-01, lost).**
- SOL counters (p57 kernel, B=2048 harness): VALU busy 75% of SIMD cycles,
  waves waiting to issue 59%, ~9.2 waves/SIMD: VALU-throughput bound with
  some exposed latency. `suggest`: only a residency cliff (128 -> 120
  VGPRs), but all 768 waves are already resident.
- The token loop (unroll 8) has 672 three-operand `v_fma_f32`, which VOPD
  cannot pair; HIP's ISA uses `v_dual_fmac`.
  - Loom lowers `scalar.fmaf` to `v_fma_f32` always.
  - `vector.dotf` lowers to `v_fma` + `v_fmac` chain in strict element order.
- `YAH_DN_DOTF=1` writes each u/p dot group as `vector.dotf` over (1, 0, 2, 3)
  from -0.0, i.e. HIP's fma(s3,x3, fma(s2,x2, fma(s0,x0, s1*x1))).
  `fma(a,b,-0)` is a*b including the sign of zero. Bit-identical to HIP's
  kernel at 256 and 203 tokens (`deltanet_vs_hip.sh`, negative control fails).
- Loop: 81 `v_dual_fmac` + 220 `v_fmac`, `v_fma_f32` 672 -> 416, but 43 fewer
  `v_dual_mul`; VALU issues 1432 -> 1382.
- Result: 4.754 -> 4.863 M cycles (dncyc), 4.732 -> 4.809 (SOL run).
  - VALU instructions -3.5% and VALU busy cycles -3.3% (75% -> 72% of SIMD
    time), yet total cycles +1.6..2.3%.
  - Cause: a VOPD pair issues only when both halves are ready. Pairing FMAs
    from different dot chains stretches the latency-sensitive per-token path
    more than the issue slots saved. Off.
- The state update (fma with s*alpha as addend) has no source-level fmac
  form. Further DeltaNet gains need the chunked algorithm, which changes
  numerics.

**Non-matrix kernels are at DRAM bandwidth; q/gate unpack fused (2026-10-01).**
Bytes from the bindings over measured time (HRX does not map `TCC_EA0_*` on
gfx1151, `profile_counters.c:752`) against ~210 GB/s streaming:
- ssm_conv ~213 GB/s, ssm_postnorm ~208, unpack_qg ~230, half_norm ~180 (63
  MB/call; it never touches its `reszero` / `sumout` bindings).
- Only fewer bytes helps these, i.e. fusion.

`kqg` GEMM kind (`gen_gemm_tile`, emitted beside every 768-tile kstore, i.e.
the attention q projections): the LDS epilogue writes row r = head*512 +
half*256 + d straight to (q|gate)[t][head*256 + d]. A wave's 32 rows sit in
one half, so the q/gate branch is uniform; the clamped index is for the bound
proof only. `loom_forward_pp` prefers it (`run_kqg`) and skips
`yah_unpack_qg`; `YAH_KQG=0` restores the two-pass path.
- p58 = full emit: md5 a2145e371ceefd4d. pp2048 cycles 7372 -> 7339 M
  (-0.45%), 947 -> 931 dispatches.
- The 16 unpack dispatches (32.9 M) are gone, and the kqg GEMMs cost what the
  kstore ones did (263.6 vs 263.3 M).

**GEMM residency via KSUB=32 (2026-10-01, lost).** `suggest` flags the IQ3
4 x 2 kernels as LDS-limited (57 KB -> 4 waves/SIMD). `YAH_TG_KSUB=32` raises
residency to 5-7 but loses on real bytes (M cycles):
- IQ3_S kstore 23.84 -> 27.34, IQ3_S kres K=6144 9.66 -> 11.97, IQ3_XXS
  kstore 23.81 -> 25.06.
- SOL (IQ3_S kstore): VALU/WMMA 4.74 -> 5.80, SALU 1.15 -> 1.70, barrier
  wait 5% -> 27%, issue bound 100% -> 91%. Twice the phases bring twice the
  per-phase overhead and barriers. The kernels were already at their issue
  bound, so more waves have nothing to hide.

Not pursued, with reasons:
- Stagger delay per kind: flat 2000-8000 except IQ4_XS kres K=6144.
- Fragment-order activations (LSE): a 16x16 f16 fragment is 32 B/lane = 2
  `ds_load_b128`, the minimum, and the 4 x 2 kernels issue exactly that (1.25
  fragment loads/WMMA + decode stores / grid reads = 1.59 measured).

**IQ3_S f16-table decode (2026-10-01, kept off: tiny).**
`YAH_SD_IQ3F16=1` decodes through two workgroup tables built at setup:
- the grid as f16 (4 KiB, 2 dwords per entry);
- per sign byte, the f16 sign bits of its 8 elements as 4 XOR masks (4 KiB).

Per 8 elements: one 16-byte sign load, two 8-byte grid loads, 4 xors, 8
`fptrunc(fma(extf(+-mag), dsc, -0))`, which select `v_fma_mixlo/hi` straight
from the f16 halves. The product is exact in f32, so it is bit-identical.
- K block non-WMMA VALU 289 -> 193 per 64 WMMA; VALU/WMMA 4.74 -> 3.25 (SOL).
- LDS 57 -> 63.5 KB, still 4 waves/SIMD.
- Real bytes, M cycles, bit-identical on all four kinds: kstore 24.13 ->
  23.96, swiglu 24.35 -> 24.32, kres K=17408 24.02 -> 23.69, K=6144 9.63 ->
  9.67.
- The issue model predicted -3%. Measured, each removed decode VALU saved
  ~0.23 cycles, not ~1.1: most decode VALU already overlaps other waves'
  34-cycle WMMAs. The kernel went from 100% to 98% of the model.
- Lesson: the per-instruction VALU price (hw-measured.md: ~1.1 next to WMMA)
  overstates decode cost in the full kernels. More decode trimming has low
  payoff; what remains over the floor is mostly LDS and synchronization.

**Wave64 tile GEMM revisited (2026-10-01, falsified).** Loom's wave64 WMMA
keeps the full 16-half operand per lane (replicated across both 32-lane
halves); only the accumulator halves (4 VGPRs per fragment).
- IQ3_S kstore, compile (`YAH_TG_W64=1`):
  - wave64 4 x 2: 184 VGPRs (wave32 224), ds_load/WMMA 1.38 (same),
    VALU/WMMA 5.33 (4.52).
  - wave64 2 x 2 (64 x 128 per wave): ds_load/WMMA 0.81, VALU/WMMA 3.72, but
    256 VGPRs with 22 spill stores.
  - 4 x 1 and 2 x 1 hit generator gaps.
- `research/wm64.hip`: WMMA is 34.0 cycles per 16x16x16 per SIMD in both
  wave sizes.
- Wave64 4 x 2: bit-identical, 23.93 -> 31.27 M cycles. SOL per instruction:
  - VALU 1.03 -> 1.76 cycles, LDS 1.49 -> 2.53.
  - Barrier wait 5 -> 17%, waiting to issue 10 -> 51%.
- A wave64 instruction costs ~1.7x a wave32 one (0.88x per lane). Our
  overhead is counted per WMMA, and a wave64 WMMA does the same work, so it
  all grows ~1.7x.
- Even the 2 x tile only breaks even on LDS (0.81 x 2.53 vs 1.38 x 1.49) and
  loses ~1.8 cycles/WMMA on VALU. The earlier 12% wave64 gain ("wave64: under
  192") was against a far weaker wave32 kernel.

**Wave64 where it fits: DeltaNet (2026-10-01).** Pure FP32 FMA probe
(`research/valu64.hip`, 8 independent chains): wave32 reaches 56.6
lane-FMAs/cycle/SIMD only because the compiler pairs 32 of 40 FMAs into
`v_dual_fmaak`. Wave64 reaches 60.0 with no pairing: single FP32
instructions run on both ALU halves.
- DeltaNet's 672 three-operand `v_fma_f32` per 8 tokens cannot pair in wave32
  (and forcing pairs via `vector.dotf` hurt latency).
- `YAH_DN_W64` (default on): each 64-lane wave takes 16 rows (second row 8
  below). Every row's arithmetic, including the 8-lane xor butterfly, is
  unchanged. Bit-identical to HIP's kernel at 256 and 203 tokens.
- 124 VGPRs, 5 waves/SIMD. B=2048 harness 4.752 -> 4.184 M cycles. Unroll
  2/4/8/16: 4.59 / 4.31 / 4.18 / 4.12 (8 kept).
- p59 = full emit: md5 a2145e371ceefd4d. Pipeline DeltaNet 226.7 -> 189.2 M
  cycles (-16.5%), total 7339 -> 7315 M.
- Rule of thumb: wave64 pays where cost is per lane and FP32 (pure VALU
  kernels). It loses where cost hangs off WMMAs (GEMMs, see above).

**Fused FFN gate+up: sized, not built (2026-10-01).**
- gate/up formats per layer: same in 36 of 64 (IQ3_XXS 15, IQ3_S 12, IQ4_XS
  9). Mixed in 28 (IQ3_S/IQ4_XS 14, Q3_K/IQ3_S 7, 7 others), which would need
  two decoders in one kernel.
- ATT, IQ3_XXS 1088x20 (stagger on), per wave:
  - gate kstore: prologue 2.4%, epilogue 2.2%.
  - swiglu: prologue 2.4%, epilogue 7.2% (reads the f32 gate).
  - SIMD time with no wave in its K loop: 0.3% / 0.4%. Prologues and
    epilogues already overlap other waves' K loops; the 142 MB gate round
    trip overlaps compute-bound kernels.
- A fused kernel would remove one prologue, the gate epilogue and the gate
  read: upper bound ~4-5% of FFN time if none were hidden, realistically
  ~1-2% (~0.5% of the prefill). Not worth two-decoder kernels now.

**Concurrent dispatch prototype (2026-10-01).** See
`research/upstream-candidates.md` item 1. With a local libhrx patch (dispatch
flag bit 2 skips the stream's trailing ordering barrier) and `YAH_CONCUR=1`,
DeltaNet and the SSM z-projection GEMM overlap fully. md5 unchanged; the pair
goes 270.6 -> 251.9 ms over 48 layers (~0.6% of pp2048). Driver hook:
`LoomDevice::NoBarrierNext()`; it must stay off against stock HRX, which
rejects unknown flags.

**Attention writes f16 for the o projection (2026-10-01).**
- `gen_attn_hip` `F16OUT` (default; off under `YAH_ATTN_DBG` and in
  `attn_vs_hip.sh`, which compares f32): the epilogue stores
  fptrunc(o/sum * sigmoid(gate)) into the f16 GEMM input. That is the same
  rounding `yah_half_cast` applied in a separate pass.
- The emitter marks the set (`dispatch.txt` row `attn_f16out`); the driver
  then binds `scratch` and skips the cast. Older sets keep the old path.
- p60 = full emit: md5 a2145e371ceefd4d, 931 -> 915 dispatches, half_cast
  12.8 M cycles gone, attention 93.9 -> 93.8 M (~0.2% of pp2048).

**prep_kq fused into the SSM conv (2026-10-01).** `yah_ssm_conv_kq_f32.loom`
remaps channels: workgroup w < num_key_heads takes key head w's q (lanes
0..127) and k (lanes 128..255) channels, the rest take v 256 at a time; each
channel's conv is unchanged. The q/k workgroups put their results in LDS, and
wave 0 runs `yah_deltanet_prep_kq` op for op (per-lane sequential sums of 4,
three `subgroup.reduce`, lane 0 stores inv_k / q_scale / k.q). The driver uses
it when the set has `convkq.hal`; `YAH_CONVKQ=0` restores conv + prep_kq.
- p61 = full emit: md5 a2145e371ceefd4d, 915 -> 867 dispatches.
- Rows: conv 83.5 + prep_kq 23.5 -> conv_kq 84.2 M cycles (-22.8 M, ~0.3%).

Skipped: wave64 for `yah_fused_qk_rope_batched`. It moves ~140 MB per call in
0.77 ms (~180 GB/s, ~86% of DRAM bandwidth), so it is memory-bound.

**Advanced profiling: decode-ahead serializes on short K (2026-10-01).**
Extended counters (`ROCPROFILER_METRICS_PATH`, gfx1151-valid subset; no WMMA
pipe-busy counter on this chip), `tools/advpmc.sh`:

| | IQ3_S kstore (79% SOL) | Q4_K kres K=6144 (64%) |
|---|---:|---:|
| LDS busy (IDX_ACTIVE) | 58% | 37% |
| LDS bank-conflict cycles / LDS-active | 5.0% | 0.4% |
| wave time in s_waitcnt (counters) | 8.8% | 38.7% |
| ifetch wait / I-cache miss rate | 0.08% / 0.5% | 0.49% / 0% |

- ATT stall attribution on the Q4_K kres: 45% of all wave time is
  `s_waitcnt vmcnt(0)` inside the K loop (plus 12% `lgkmcnt(0)`). These full
  drains wait for the read-ahead loads just issued, serializing the
  decode-ahead prefetch.
- Decode-ahead on -> off, real bytes, M cycles, bit-identical:
  - Gains when off: Q4_K kres K=6144 11.34 -> 9.51; Q5_K kres K=6144
    11.82 -> 9.65; Q5_K kstore 768 18.43 -> 17.23.
  - Keeps decode-ahead: IQ4_XS swiglu 24.27 -> 25.29 and Q4_K swiglu 24.13
    -> 24.99 (both worse off).
  - Wash: IQ4_XS kstore/kres68 and Q4_K kstore/kres68.
- Q5_K left `DECAHEAD_FMTS`; `(q4k, kres, 24)` joined the emitter's
  `DECAHEAD_SKIP`.
- p62 = full emit: md5 a2145e371ceefd4d.
  - Rows: Q4_K kres K=6144 134.8 -> 119.0, Q5_K kres K=6144 33.6 -> 28.5,
    Q5_K 1024-row 33.6 -> 23.5 M.
  - Other Q5_K shapes +0.3..0.6 M. Net ~-30 M cycles (~0.4%).
- IQ3_S kstore, ATT (`simdtl.py`): window 51.8% WMMA pipe, 13.5% VALU-only,
  29.7% idle. At idle instants the waves wait on lgkm 13%, barrier 11%, VALU
  issue 6%.
- By instruction time: `s_waitcnt lgkmcnt(1)` 19% (fragment loads just
  issued) and `s_barrier` 19% (two barriers per phase, decode and MMA strictly
  separated inside a workgroup). No-K-loop time 2.5%; bank conflicts 5% of
  LDS-active cycles.
- Hiding the LDS latency needs registers 4 x 2 does not have:
  - `KSL_FENCE=0` 23.84 -> 23.79 (no change);
  - `PF=1` 256 VGPRs, 28 spills, 150.96;
  - `PF=1 PF_FENCE=0` 248 VGPRs, 24.45.
  Decode-ahead for IQ3 needs a second weight tile (+17 KB on 57 KB LDS).
  What is left over the floor here is structural (barrier-separated decode /
  MMA phases, LDS latency). The way past it would be producer/consumer wave
  specialization (decode waves feeding MMA waves through LDS without
  workgroup barriers), a larger redesign.

**Attention profile at pp8192 (2026-10-01).** Production `gen_attn_hip`
kernel, real layer-3 inputs:
- 80.26 M cycles per layer against a 42.87 M WMMA floor (53% SOL). The issue
  model explains 98%.
- Per WMMA: VALU 15.1 instr (16.6 cycles), LDS 4.0, transcendental 0.36,
  SALU 1.8.
- LDS 81% busy, bank conflicts 9.9% of LDS-active cycles. I-cache misses 5.9%
  (fetch waits 0.8%).
- ATT instruction time: `s_waitcnt lgkmcnt(0)` 37.2% (full LDS drains),
  `s_barrier` 16.2%, `v_mov_b32` 9.1% (back-edge copies). In the timeline the
  WMMA pipe is idle ~42%; at idle instants waves wait on lgkm 28%, VALU issue
  14%, barrier 11%.
- `YAH_ATTN_POL` (key-loop policy) added, bit-identical:
  - `unroll(%c2) schedule(recurrence)` 80.59 -> 79.03 M with 31 spill stores;
  - `unroll(%c2)` 82.63 M.
  Not adopted.
- Levers: cut LDS traffic (P round trip, more query rows per K/V fragment),
  bank-conflict padding of the K/V/P pitches, partial instead of full LDS
  waits, skip the O rescale when no row max changed (exact), dual-issue of the
  softmax FP32.

**FlashAttention-style attention, `gen_attn_fa.py` (2026-10-01).** Opt-in with
`YAH_ATTN_FA=1` in the emitter. Same grid, bindings and f16 output as
`gen_attn_hip`.

Layout:
- 8 waves per workgroup, as wave pairs: each pair owns 16 queries and each
  wave one half of the head dim. Q^T lives in registers (64 VGPRs).
- S^T = K Q^T, so a lane owns one query column. The pair adds its partial
  scores through a private LDS slot.
- The softmax runs in registers.
- P^T becomes the B operand through one xor-16 swizzle. K rows are permuted
  at staging so a lane holds keys 8h..8h+7.
- V^T rows are permuted so each lane writes 8 contiguous output dims.
- K/V tiles are loaded at the top of phase A and staged in phase B of the
  same tile. V is double-buffered, LDS 41 KB, 3 workgroups per WGP.

Steps, M cycles (pp8192, real layer-3 inputs, production 80.1):

| step | M cycles | cause, measured |
|---|---:|---|
| first build | 178.6 | v_cvt_f16_f32 low-window evictions of Q^T to scratch; scratch vmcnt(0) drained the prefetch |
| v_fma_mix narrowing of P | 81.9 | 0 spills |
| V rows padded to 48 B, single K/V buffers | 76.8 | LDS 104% busy, 39% bank conflicts -> 4.8% |
| unroll(2) recurrence | 73.8 | |
| loads used in the same tile (no vmcnt(0) back-edge drain, 15% of wave time) | 73.9 | correct after the Q-drain fix below |
| head-pair-fastest + LPT order | **70.3** | L2 hits 35% -> 73%, DRAM 8.2 -> 2.8 GB/layer (it was at ~250 GB/s, bandwidth-bound) |
| HIPNUM (HIP's exp / sum / division rounding) | 69.8 | 13x closer to HIP (rel 9e-8, 59% of elements bit-identical) |

Pipeline (SQ_BUSY_CYCLES, one round each):
- pp8192: attention 1264.0 -> 1111.6 M (-12.1%), total -0.57%.
- pp2048: attention -2.1%.

Lost or neutral, single rounds: fences (QKF/PVF), two QK chains, PVZ, S8,
MSKIF, exact rescale skip (SKIP; it also miscompiled alone). The kernel sits
at 103% of the issue bound: ~13 VALU per WMMA, of which ~46 v_mov per tile
are O back-edge rotation (allocator, upstream #4) and ~35 address ops (no
LICM, upstream #2).

Numerics (T1 gate, golden2):
- FA: kl_mean 3.64e-7 against a limit of 1.32e-7 -> FAIL. Passes kl_p999,
  0 flips, PPL 6.6946 vs 6.6947.
- **Control: production HIP-order attention with one change, o * (1/sum)
  instead of o / sum: kl_mean 4.24e-7 (FAIL), 0 flips.** kl_mean barely moves
  with the size of the attention error (FA rel 1.2e-6 -> 9e-8 gave 3.70e-7 ->
  3.64e-7). Any non-bit-exact attention change saturates at ~4e-7 through the
  64 layers, so T1's kl_mean limit amounts to bit-exactness for attention.

Loom miscompiles hit on the way (all worked around; see upstream-candidates):
- **`kernel.barrier` does not drain earlier LDS loads.** There was no
  lgkmcnt(0) ahead of the s_barrier that followed the Q^T fragment loads. Other
  waves then staged K/V over the aliased Q stage while the loads were still in
  flight: a few corrupted query lanes per run, varying. Fixed by an LDS store
  of a value built from every fragment before the barrier.
- Two sequential loops without the unroll policy: O accumulators corrupted
  across the loop-to-loop hand-off (NaN in ~30% of dims from query block 1
  on). The kernel now uses one masked loop.
- SKIP alone: NaN everywhere; correct when combined with S8 and MSKIF.

**FA attention, round 2: the FA1-4 leftovers (2026-10-01).** Standalone
pp8192 M cycles, real layer-3 inputs; previous best 69.8 (HIPNUM):

| change | M cycles | verdict / measured cause |
|---|---:|---|
| FA4 conditional rescale, exact (SKIP, THR=0; yields alpha, 1.0 when skipped) | **67.1** | default. VALU/WMMA 14.1 -> 11.1; same bits as always rescaling. Without the unroll policy the branch also removed the 57 back-edge O copies. Under unroll(2) the skip path of the second copy copied O (73 v_mov) and lost (70.6), so POL now defaults off. |
| threshold THR=8 / 4 (FA4) | 69.2 / - | no gain over exact; numerics change; off |
| single V buffer (VSB: V(i) loaded top of phase A, staged at its end) | **64.6** | default, same bits. Drops the buffer parity (VALU 11.1 -> 10.6). Same 3 workgroups per WGP. |
| + Q staged in halves (QH) | 66.1 | VGPRs 192 -> 200: the 4th workgroup still does not fit; prologue overhead. Off. |
| 32-key tiles (KT=32, two sub-tiles per barrier pair) | 72.5 | LDS 53.8 KB -> 2 workgroups per WGP (4 waves/SIMD instead of 6), despite VALU/WMMA 10.1. K, V and the score slots are live in the same phase, so 3 workgroups cannot fit. Off; also not HIP-order (rel 7e-6). |
| GQA packing (GQA=1: 6 heads x 16 tokens, 12 waves) | 66.8 | same bits. DRAM 2.8 -> 0.74 GB/layer, L2 requests 83 -> 32 M. 2.8% behind 2 heads: 2 x 12 waves per WGP give less phase diversity than 3 x 8. Kept as an option; it pays once per-KV work grows (e.g. int8 KV dequant once per 6 heads). |

GQA notes:
- With the staging loads inside wave-uniform scf.if guards, the compiler
  drained vmcnt(0) at the region exit, right after issue (69.8). Loads are now
  unconditional and only the LDS stores are guarded.
- Guards use the subgroup id: tid compares lowered as lane-masked regions,
  which branch lowering rejected.
- Query block fastest (SWZ=0) beats KV head fastest for the packed kernel.

Pipeline pp8192 (one round, idle start), production default config vs p62:
- Attention row 1264.0 -> 1024.8 M cycles (-18.9%).
- Hidden md5 = p62fa (bit-identical to the gated FA build).
- The total (-0.5%) is clock-confounded: this run clocked higher (12988 vs
  13505 ms), which adds cycles to the memory-bound rows.

**Quantized KV, int8 config (2026-10-01).** User configs: fp16, int8, kv4a8,
kv4a4. Measured rates (research/hw-measured.md): iu8 WMMA = f16 (34
cycles/SIMD), iu4 = 2x. Loom lowers both; signedness comes from an operand
schema (`element_format=i8|u8`, payload vector<4xi32>).

Pieces (gen_kvq.py + gen_attn_fa.py; emitter YAH_ATTN_FA_KQ8 / _VQ8, markers
attn_kq8 / attn_vq8):
- **K:** `yah_kmean` (deterministic per-channel prompt mean, no atomics) +
  `yah_kq8`: int8 of K - mean, one scale per (token, kv head, 128-dim half).
  - Subtracting a per-channel constant shifts each query row's scores
    uniformly, so softmax is unchanged (exact).
  - Recon rel RMS 5.7e-3 (7.7e-3 without centring).
- **Q:** quantized in-kernel per (row, half) while staging.
- **QK^T:** iu8 WMMA, then S = i32 * s_q * s_k.
- **V:** `yah_vstat` (per-channel min/max) + `yah_vq8`: uint8 V^T around the
  midrange, per-channel scale, recon rel RMS 4.4e-3.
  - Staging builds f16 1024 + u by integer ops (`(w & 0x00ff00ff) |
    0x64006400`, dword token order t0, t2, t1, t3), so P.V stays f16.
  - The epilogue applies o / l * s + (c - 1152 s).
- **Epilogue fences:** VGPRs 240 -> 176. The scheduler had hoisted every
  fragment's gate + scale loads.

Gate (golden3 = p63 fp16 on 16 windows + the 8192 window L0, T2):
- int8 K: kl_mean 3.6e-5, 0 flips, PPL 5.5723 vs 5.5718; L0 KL 1.4e-5,
  0 flips.
- int8 K+V: kl_mean 5.0e-5, 0 flips, PPL 5.5730; L0 KL 1.9e-5, 0 flips,
  PPL 6.0855 vs 6.0879. **PASS.**

Work, standalone pp8192 layer 3 (research/fa/work.sh):

| | fp16 | int8 K | int8 K+V |
|---|---:|---:|---:|
| M cycles | 64.2 | 61.9 | 62.8 |
| DRAM GB | 4.17 | 4.73 | 2.42 |
| L2 requests M | 93.1 | 75.4 | 51.1 |
| LDS instr/WMMA | 2.89 | 2.45 | 2.45 |
| LDS busy | 86% | 75% | 74% |
| VALU/WMMA | 10.50 | 11.13 | 12.42 |
| VGPR (waves/SIMD) | 192 (6) | 160 (8) | 176 (8) |

- Pipeline pp8192: attention 1024.8 -> 998.3 M (-2.6%); with KV prep -2.5%
  (quantizers 8.4 M vs the f16 transpose 7.7 M).
- The kernel is issue-bound, so the scale/unpack VALU eats most of the
  memory/LDS savings.
- Gate tooling: accgate2.py / gate_run.sh accept long windows `L<n>` (8192
  tokens, positions FROM_L=7680.., SET8K = an 8192 set).
- The driver dispatches 384-thread attention when the emitter records
  `attn_wg384` (GQA packing).

**GQA packing + int8 (2026-10-01).** Bit-identical to the non-GQA int8 kernel;
the 384-thread dispatch path (`attn_wg384`) is validated in the pipeline
(hidden md5 equal at pp8192). Two fixes on the way:
- **Occupancy cliff:** 12-wave workgroups need <= 170 VGPRs for 3 per WGP.
  int8 K+V sat at 176: all 8 fragments' gate loads were issued before the
  per-fragment epilogue (56 VGPRs, found with --compile-report=details
  pressure_origin_rows). VQ8 now issues them per fenced fragment: 160 VGPRs,
  GQA and non-GQA.
- The V unpack runs outside the wave guard. An scf.if reading loaded
  registers drained vmcnt(0) at its entry, which also waited on the K loads.
  The unpack is scalar ops with literal masks.

| standalone pp8192 M cycles | non-GQA | GQA |
|---|---:|---:|
| int8 K | 62.1 | 61.9 |
| int8 K+V | 63.7 | 63.2 |

Pipeline pp8192 attention: fp16 1024.8, int8 998.3, int8+GQA 1006.5 M
(+0.8%, noise level). GQA trims VALU/WMMA 5% (V unpack once per 6 heads) but
gives up phase diversity (2 x 12 waves). Kept as an option for configs with
heavier per-KV work.

**Quantized KV, kv4a8 and kv4a4 (basic) (2026-10-01).** The user asked for
basic implementations first; rotation and other compression work comes later.

Numerics prototype (research/fa/proto*.py): layer-3 attention-output error vs
exact, float64.

| change | rel RMS |
|---|---:|
| int8 config | 3.3e-3 |
| + uint8 P | 5.1e-2 (16x worse: weights under 1/510 of the row max round to 0) |

So P stays f16 in every config. Int4 options:
- K4: per token-half 3.2e-2; with H128 rotation 2.1e-2; H256 asym 32-group
  1.25e-2; H128 + 64 sink tokens in fp16 1.5e-2.
- V4: per-channel per 16-token tile 2.1e-2; per-channel over the prompt
  3.5e-2; per-token 5.8e-2.
- Q8 vs Q f16: no difference.

Kernels (gen_kvq.py yah_kq4 / yah_vq4, gen_attn_fa.py KQ4 / VQ4 / KA4 / KROT):
- **K int4:** centred, one scale per (token, half), signed nibbles. Byte j of
  each 8-dim dword holds dims j | j+4 << 4.
  - kv4a8: `(w << 4) & 0xf0f0f0f0` / `w & 0xf0f0f0f0` give signed bytes 16 k
    with no zero point; the 1/16 folds into s_q. iu8 QK^T.
  - kv4a4: the nibbles as stored, Q int4 packed in the same order, iu4 WMMA.
- **V int4:** 15 levels per channel per 16-key tile, f16 (S, C') = (16 s,
  c - 23 s). Staging builds f16 `1 + u/16` = `0x3c00 | u << 6` by masks, then
  one 16-wide packed fma. P.V stays f16.
- **KROT=1:** Hadamard H128 per half on K (quantizer) and Q (staging).
  Default off (basic).

Results (layer 3 standalone, pp8192 pipeline, gate vs golden3 = fp16):

| config | attn rel RMS | pipeline attention (vs fp16 1024.8 M) | mean KL | PPL | flips / 8192 | L0 (8192 ctx) |
|---|---:|---:|---:|---:|---:|---|
| int8 | 2.7e-3 | 998.3 M | 5.0e-5 | +0.02% | 0 | KL 1.9e-5 |
| kv4a8 basic | 3.0e-2 | 1050.3 M | 4.6e-3 | +0.37% | 65 | KL 2.6e-3, PPL +0.21% |
| kv4a8 + KROT | 2.6e-2 | ~1050 M | 3.0e-3 | +0.27% | 45 | KL 1.1e-3 |
| **kv4a4 basic** | 5.0e-2 | **876.6 M (-14.5%)** | 7.9e-3 | +0.52% | 139 | KL 5.0e-3, PPL -0.15% |
| K8 + V4 (mix) | | | 3.0e-4 | +0.05% | 1 | T2 PASS |
| K4 + V8 (mix) | | | 1.8e-3 | +0.24% | 39 | |

- K error costs ~6x the KL of V error at equal attention-output error (it
  moves softmax weights). The int4 quality lever is K.
- kv4a8 is VALU-issue-bound: +40 VALU/tile over int8, ~33 of it the V nibble
  decode (rocprof studio profile_run + wave capture diff).
- kv4a4's iu4 QK^T halves the QK matrix cycles.

Loom miscompiles on the way (upstream-candidates #10):
- Two 8-wide f16 `vector.fmaf` with identical addend splats: CSE merged the
  splats and both in-place `v_pk_fmac_f16` were tied to one register. The
  second fma read the first's result (silent wrong output; micro-test
  research/fa/unpk.loom). A shared-addend variant failed allocation instead
  (coalescing.c:1637). Workaround: one 16-wide fma.
- f16 `vector.subf` / `uitofp` are rejected (vector_f32 constraint);
  `vector.fmaf` on f16 works (v_pk_fmac_f16).

### Paged KV (2026-10-02): default (p71); YAH_KV_PAGED=0 for the contiguous cache

- **Default:** the emitter pages unless YAH_KV_PAGED=0, and falls back to the
  contiguous cache (with a note) when the context is not a multiple of 256 or
  attention is not the FA kernel; an explicit YAH_KV_PAGED=1 then errors. The
  decision is pinned in the env before gen_attn_fa / gen_kvq are imported.

- **Layout:** 256-token pages. One page table per sequence (logical -> physical
  page, shared by all layers) renumbers K rows (and K scales) and V^T tiles
  (and V stats); the layouts are otherwise unchanged. Every 16-key attention
  tile lies in one page, so attention does one uniform page-table load per K
  tile and per V tile. The host validates every table entry (< npages) before
  upload; attention assumes the bound, the cache writers also clamp to it.
- **Writers:** the f16 KV cache is always a one-layer, one-chunk scratch (RoPE
  cache_start). Per chunk:
  - fp16: RoPE stores K rows straight into their page (emitter transform
    rope_kpaged, geometry marker "rope_kpaged") and yah_vtpage writes V^T
    tiles, instead of the whole-cache yah_transpose_v16 per layer;
  - quantized: yah_kq8 / yah_kq4 / yah_vq8 / yah_vq4 in paged form.
  The driver fills the table (identity; YAH_PAGE_SCRAMBLE=<seed> shuffles it
  for testing). The context must be a multiple of 256; FA attention only.
- **Bit-identical to non-paged**, including scrambled page maps:
  - fp16 pp2048 md5 ac36332b6b5092a4 and pp8192 963b7396625e2333;
  - kv8a16 / kv4a16 one-pass 8K and chunked 8K;
  - fp16 32K chunked (arXiv row stats).
- **Cost (one round each, SQ_BUSY_CYCLES):** first version (separate yah_kpage
  copy) +0.14% (pp2048) / +0.28% (pp8192). With RoPE writing K into its page:
  -0.25% / -0.48%, i.e. within run-to-run noise (GEMMs alone moved 0.4%);
  attention still +2% from the page lookups, RoPE +5% (27.9 M at pp2048), the
  K copy is gone and vtpage ~ vtrans. In chunked long-context runs paging
  saves the per-chunk whole-cache V^T re-transpose (~16x redundant at 32K).
  - Wall time is not the metric here: pp8192 paged ran +7% wall in the same
    round, with identical idle (0.5%) and 7% fewer SQ cycles per device tick
    across every kernel, from a mid-run clock step (octile 4). The previous
    round had the same step on the non-paged run instead.
  - Tried and reverted: the page table in LDS (attention +1.1% over SMEM; the
    paging cost is the index math, not the scalar load, see gen_attn_fa.py),
    and one shared page lookup per K tile (no gain). The LDS version hung the
    GPU (ring gfx timeout, MODE2 reset) while the prologue read the table
    before the barrier that publishes it.
