# Porting the GPU kernel set to Loom

Status: in progress. Goal is coverage and correctness first; performance tuning is
explicitly deferred.

## Rules for each port

1. Read the HIP source and state the exact semantics in the header comment.
2. Keep the same buffer signature and the same output dtype.
3. Give it an in-source `check.case` with distinct input values, and an expectation
   computed analytically where one exists (identity and all-zero patterns do not
   catch a transposed index).
4. Verify correctness and measure through `./loom_run.sh <file> <case> <benchmark> <config>`.
   It runs the correctness case and the benchmark under **one** config, and
   refuses any config whose declared operand footprint exceeds what the case binds.
   Do not hand-roll `iree-benchmark-loom` invocations for a check.case: re-running a
   validated case under a larger config is an out-of-bounds access on this target and
   hangs the GPU (docs/gpu-ring-hang-qdq.md).
5. Do not tune during the port. Record the number and move on.

## Status

| Loom file | HIP source | kernel | status |
| --- | --- | --- | --- |
| yah_ffn_gemm_f16.loom | prefill_fp16.hip | batched f16 FFN GEMM | ported, 1.819 ms, 1.34x behind HIP |
| yah_residual_add_f32.loom | prefill_residual.hip | batched residual add | ported, 0.0189 ms at 327680 elements |
| yah_rmsnorm_f32.loom | prefill_norm.hip | batched RMSNorm | ported, 0.0821 ms at 64x5120 |
| yah_perhead_rmsnorm_f32.loom | prefill_norm.hip | batched per-head RMSNorm | ported, 0.0154 ms at 8x4x128 |
| yah_swiglu_f32.loom | prefill_swiglu.hip | SwiGLU activation (split form) | ported, 0.0126 ms at 327680 elements |
| yah_rope_f32.loom | rope.hip | RoPE (text path) | ported, 0.0091 ms at 384 pairs |
| yah_embed_f32.loom | embed.hip | embedding lookup (f32 table) | ported, 0.0087 ms at hidden 5120 |
| yah_unpack_qg_f32.loom | prefill_unpack.hip | batched QG unpack | ported, 0.0149 ms at 64x5120 |
| yah_ssm_proj_f32.loom | prefill_ssm.hip | fused SSM input projections (GEMV) | ported, 0.0102 ms |
| - | prefill_ssm.hip, ssm_row_split.hip | DeltaNet conv/recurrence | todo |
| - | prefill_attention*.hip, attention_wmma.hip | batched attention | todo |
| - | qkv.hip | QKV projection | todo |
| - | gemv.hip, gemv_quant.hip | decode GEMV | todo |
| yah_argmax_f32.loom | sample.hip | argmax over logits | ported, 0.0145 ms at vocab 1024; sampling variants todo |
| yah_hadamard_f32.loom | engine/kv/kv_quant.hip | in-place Hadamard over a KV block | ported, 0.0139 ms at rows=1 |
| yah_qdq_f32.loom | engine/kv/kv_quant.hip | quantize/dequantize over 32-element blocks | ported, 0.0546 ms at 10240 blocks (327680 f32) |
| yah_qdq_f16.loom | engine/kv/kv_quant.hip | fp16 quantize/dequantize over 32-element blocks | ported, 0.0434 ms at 10240 blocks (327680 f16) |
| yah_kv_quant_q8_f16.loom | engine/kv/kv_quant.hip | KV cache pack to q8 blocks | ported, 0.0260 ms at 10240 blocks; byte-exact vs fixture |
| yah_kv_dequant_q8_f16.loom | engine/kv/kv_quant.hip | KV cache unpack from q8 blocks | ported, 0.0275 ms at 10240 blocks; byte-exact vs fixture |
| yah_kv_quant_q4_f16.loom | engine/kv/kv_quant.hip | KV cache pack to q4 blocks | ported, 0.0266 ms at 10240 blocks; byte-exact vs fixture |
| yah_kv_dequant_q4_f16.loom | engine/kv/kv_quant.hip | KV cache unpack from q4 blocks | ported, 0.0239 ms at 10240 blocks; byte-exact vs fixture |
| yah_hadamard_f16.loom | engine/kv/kv_quant.hip | in-place Hadamard over an fp16 KV block | ported, 0.0222 ms at rows=1 |
| - | vision/encoder.hip, device_input.hip | vision tower | todo |

## Remaining inventory

From `grep -c '__global__ void'` over `engine/gpu/ported/src/models/qwen`. Roughly
100 kernels; 18 are ported. Ordered by share of prefill time where the model-level
profile gives one, so the expensive paths move first rather than the convenient ones.

| Area | File | Kernels |
| --- | --- | --- |
| DeltaNet / SSM (5.1%) | ssm_row_split.hip | BatchedDeltaNetRowSplitKernel, BatchedDeltaNetPrepAlphaBetaKernel, BatchedSSMConvKernel, BatchedSSMPostNormGateKernel, BatchedSSMPostNormGateFp16Kernel |
| DeltaNet / SSM | ssm_recurrence.hip | BatchedDeltaNetRecurrenceKernel, BatchedSSMConvKernel, BatchedSSMPostNormGateKernel |
| DeltaNet / SSM | batched_ssm.hip, ssm.hip, ssm_decode_recurrence.hip | BatchedSSMPostNormGateKernel, FusedSSMInputProjectionsKernel, SSMConvKernel, CaptureBatchedSsmReplayKernel |
| attention (2.1%) | attention_wmma.hip | PackAttentionHeads, PackTiledAttentionKvKernel, SyncTiledAttentionKvPrefixKernel |
| attention | attention_batched.hip | BatchedAttentionKernel, CausalSoftmaxKernel, WriteBatchedKVCacheKernel, ApplyAttentionGateKernel |
| attention | attention_tile.hip, attention_decode_graph.hip, attention_decode.hip | tiled, decode-online, split-K, and KV-write variants |
| QKV projection | qkv.hip | FusedQKVProjectionsKernel |
| fused RoPE | prefill_rope.hip | BatchedFusedQKNormRoPEKvWriteKernel, BatchedRoPEKernel |
| fused.hip | fused.hip | FusedQKNormRoPEKvWriteKernel |
| dequant to bf16 | prefill_gemm.hip | Q4_K/Q5_K/Q6_K/Q8_0/Q8_1 and elementwise dequant, FloatToBfloat16Kernel |
| W8A8 + fused quant | prefill_quant_gemm.hip | W8A8BlockedWmmaGEMMKernel, QuantizeActivationToQ8_1Kernel, RequantizeActivationInt4Kernel, and three fused RMSNorm/SwiGLU/SSM-norm quantize kernels, ZeroQ8ActTailKernel, BatchedQuantGEMVKernel |
| f16 conversion set | prefill_fp16.hip | AtbEncodeA/DecodeC/SwiGLU, AtbRepack(+Slice), AtbAddHeadFp32, AtbExpandHeadFp16, HalfCast, HalfNorm, HalfNorm5120, Bfp16RoundTripFp16 |
| GEMV (1.5%) | gemv.hip, gemv_quant.hip | FastGEMVBlockKernel, and the quantized variant |
| sampling | sample.hip | 13 more kernels: batched argmax, sparse penalties, linear/sorted sampling, and the speculative segment set |
| vision | vision/encoder.hip, vision/device_input.hip | Patchify, PatchPosition, QkvRope, AttentionRows, Softmax, LayerNorm, BiasResidual, Activate, Finish, InjectRows |
| decode leftovers | embed.hip, rope.hip, norm.hip, residual.hip, unpack.hip, swiglu.hip | the Ptr and decode-side variants |
| dflash | dflash_kernels.hip | grouped convolution, non-causal attention (2), q8_0 quantize, silu_mul, and four selector kernels |
| benchmark scaffolding | core/hip/allocation_benchmark.hip | not part of the engine kernel set |

## Notes carried over from the FFN GEMM port

- `--measure=auto` selects `case_end_to_end` for a `check.case`. Kernel time needs
  `--measure=dispatch_complete`.
- `loom-format --in-place` is the reliable canonicaliser; `loom-check` roundtrip
  disagreed with it on one file.
- `index.ceildiv` does not exist; compute ceilings with add and div.
- A loop with a declared result must bind a name even when the result is unused.
- `check.generate.iota` takes float literals for a float tensor; integer literals
  fail verification ("attribute step has kind 1, expected 2").
- **`config.get` must be repeated inside the launch body.** The launch-config region
  is a separate scope, so a value fetched only there is undefined in the body.
- **Not every scalar op has a target contract.** `scalar.isfinitef` compiles under
  loom-check and then fails with "target 'x' has no target-low contract for
  'scalar.isfinitef'". A bound comparison covers NaN and +inf instead. Check the
  contract for any op beyond arithmetic before building a kernel around it.
- **The formatter rewrites large float literals** (-3e38 becomes -3.0000000000000001e+38),
  so a later string replace against the original spelling silently fails. Re-read the
  file after formatting before patching it.
- There is no `scalar.select`; use `scf.if` with results. `vector.select` exists.
- **`index.cmp` returns i1, and `index.andi` does not accept i1.** Select a row band with
  one biased unsigned comparison instead: `(m - band_start) < band_size` is true exactly
  inside the band and wraps out of range below it.
- Where a permuted or non-uniform expectation is needed, `check.generate.iota` cannot
  express it. Choose a shape where the expectation collapses to an arithmetic sequence
  (the QG unpack case uses head_dim=1 so the interleave is src = 2*idx), or accept a
  captured fixture.
- Float literals in expectations are canonicalised by the formatter (1.0e-4 -> 0.0001).
- Fragment loads from global memory are what this compiler is good at; LDS staging
  measured slower in all three variants tried.
- **A kernel cannot learn its operand size on this target.** `buffer.length` exists
  but has no AMDGPU target-low contract, launch arguments are not in scope in the
  launch-config region, and a raw buffer carries no device-visible length. Every
  extent comes from `config.*`, so the config is a promise from the caller. An
  over-declared config is an out-of-bounds write that wedges the gfx ring --
  that is the `yah_qdq_f32` hang, and `tools/loom_preflight.py` now blocks it.
- **A dynamic view extent is only provable from a `config.decl ... where [range(...)]`.**
  `index.assume` on a buffer-derived value did not satisfy the subrange verifier
  (`SUBRANGE/024`, "view_bound is <dynamic>"). Keep the config-derived bound and
  derive nothing about extents from runtime queries.
- **`index.div` by a power of two can be rejected on AMDGPU.** `index.div %x, 128`
  lowers to `index.shrui`, which hits `TARGET/004` / `low_register_unit_count`.
  Compare in the offset domain with `index.scale` and a byte length instead.
- `--compile-report=details` reports the operand footprint the kernel was told to
  touch (`source_low.memory.roots[].interval_envelope.byte_count`). It is the
  cheapest way to see a config/operand mismatch before it reaches the GPU.
- **Float width changes use `scalar.extf` and `scalar.fptrunc`.** `scalar.truncf`
  is C `truncf` (round toward zero at the same width), not a narrowing conversion:
  `scalar.extf %v : f16 to f32` widens, `scalar.fptrunc %v : f32 to f16` narrows
  with the usual round-to-nearest-even. `scalar.truncf` on a float tensor is a
  different operation and would silently round the wrong way.
- `scalar.roundf` rounds ties away from zero, which is what `lroundf` and
  `__float2half_rn`-adjacent code expect; pair it with `scalar.fptosi`, whose own
  conversion rounds toward zero. The q8 pack fixture deliberately keeps two products
  that land exactly on a `.5` tie, which is what pins `scalar.roundf` to
  ties-away-from-zero and therefore to `lroundf`.
- **Packed byte layouts are checked against a committed generator, not a blob.**
  `check.file.read.npy` resolves relative to the module, so fixtures live beside the
  `.loom` file. `fixtures/kv_quant_q8/generate.py` replicates the HIP kernel in the
  same widths and writes both the input and the expected bytes; regenerate with
  `python3 fixtures/kv_quant_q8/generate.py`. This is the pattern for any kernel
  whose output is a layout rather than a value: a written-down reference is the
  oracle and the check compares bytes exactly. A generic elementwise
  `check.oracle.call` provider would be nicer, but the shipped tool registers only
  `reference.matmul` and `reference.tiled_matmul`; a scalar oracle is an embedding
  hook the CLI does not wire up.
- **When an fp16 result would round, rescale the input so the expectation stays an
  arithmetic sequence.** The unscaled f16 Hadamard sums to `15872 + 32*e`, which
  fp16 cannot represent, so no single `check.generate.iota` describes it. Dividing
  `h` by 16 makes the sum `992 + 2*e`, exactly representable and still an iota. The
  transform itself is unchanged; only the fixture's magnitude moves.
