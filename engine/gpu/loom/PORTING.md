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
| yah_deltanet_rowsplit_f32.loom | ssm_row_split.hip | DeltaNet row-split recurrence | ported, 19.22 ms at batch 512 / 8 heads (untilable port; tuning deferred) |
| yah_deltanet_prep_kq_f32.loom | ssm_row_split.hip | DeltaNet K/Q-norm prologue | ported, 0.0076 ms at 3 tokens / 2 key heads |
| yah_ssm_conv_f32.loom | ssm_recurrence.hip | causal SSM convolution + gate | ported, 0.0148 ms at 4x16; fixture + 1e-6 tolerance |
| yah_ssm_postnorm_gate_f32.loom | batched_ssm.hip | SSM post-norm + gate epilogue | ported, 0.0173 ms at 2x2x128; fixture + 1e-6 tolerance |
| yah_deltanet_prep_ab_f32.loom | ssm_row_split.hip | DeltaNet alpha/beta prep + conv history advance | ported, 0.0087 ms; ab within 1e-5, history exact |
| yah_ssm_postnorm_gate_f16.loom | ssm_row_split.hip | fp16 SSM post-norm + gate epilogue | ported, 0.0079 ms at 3 heads; fixture + 1e-6 tolerance |
| yah_attn_gate_f32.loom | attention_batched.hip | attention output gate | ported, two exact sigmoid points, no fixture |
| yah_attn_softmax_f32.loom | attention_batched.hip | causal softmax + causal zeroing | ported, exact at both mask ends |
| yah_kv_cache_write_f32.loom | attention_batched.hip | batched KV cache write, f32 + f16 layouts | ported, 0.0113 ms; exact fixture (the two layouts differ) |
| yah_attn_batched_f32.loom | attention_batched.hip | batched attention core, f32 + f16 cache | ported, 0.0091/0.0120 ms; fixture + 1e-6 |
| - | attention_batched.hip, prefill_attention*.hip, attention_wmma.hip | tiled/WMMA attention, KV prefix sync, head packing, bf16 output | todo |
| yah_qkv_proj_f32.loom | qkv.hip | fused QKV projection, f32 weight path | ported, 0.0060 ms at 3+2+2 rows; bf16/q8_0/quant paths todo |
| yah_cast_f32_to_bf16.loom | prefill_gemm.hip | f32 to bf16 cast | ported, 0.0068 ms; exact, no fixture |
| yah_dequant_q8k_bf16.loom | prefill_gemm.hip | Q8_K weight dequant to bf16 | ported, 0.0076 ms; exact periodic expectation |
| yah_dequant_q8_0_bf16.loom | prefill_gemm.hip | Q8_0 weight dequant to bf16 | ported, 0.0081 ms; exact periodic expectation |
| yah_dequant_q5k_bf16.loom | prefill_gemm.hip | Q5_K dequant to bf16 (packed 6-bit scales) | ported, 0.0078 ms; bit-exact via `check.tensor.view` |
| yah_dequant_q6k_bf16.loom | prefill_gemm.hip | Q6_K dequant to bf16 | ported, 0.0638 ms; bit-exact via `check.tensor.view` |
| - | prefill_gemm.hip | generic sub-16 element decoder to bf16 | deferred: a thin wrapper over the whole `QuantBlockElement` format table, not a single format |
| yah_half_cast.loom | prefill_fp16.hip | fp32 to fp16 cast | ported, 0.0074 ms; exact, no fixture |
| yah_atb_head_expand_f16.loom | prefill_fp16.hip | ATB packed-head expansion (fp16) | ported, 0.0074 ms; fixture for the untouched tail |
| yah_atb_head_add_f32.loom | prefill_fp16.hip | ATB packed-head accumulate (fp32) | ported, 0.0074 ms; fixture for the untouched tail |
| yah_half_norm_f16.loom | prefill_fp16.hip | fp16-output norm with residual and sum_out | ported, 0.0224 ms at dim 512; exact both buffers |
| yah_half_norm5120_f16.loom | prefill_fp16.hip | fp16-output norm, width-specialized | ported, 0.0235/0.0788 ms at dim 512/5120 |
| yah_bfp16_roundtrip_f16.loom | prefill_fp16.hip | bfp16 shared-exponent round trip (diagnostic) | ported, 0.0072 ms; bit-exact vs fixture |
| yah_residual_add_1d_f32.loom | residual.hip | 1-D residual add (decode) | ported, 0.0075 ms; exact, no fixture |
| yah_unpack_qg_id_f32.loom | unpack.hip | decode QG de-interleave | ported, 0.0075 ms; exact fixture |
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
| yah_vision_finish_f32.loom | vision/encoder.hip | vision encoder finish (bias + bf16 round) | ported, 0.0065 ms; exact fixture |
| yah_vision_bias_residual_f32.loom | vision/encoder.hip | vision bias + residual (two bf16 rounds) | ported, 0.0062 ms; exact fixture |
| yah_vision_layer_norm_bf16.loom | vision/encoder.hip | vision layer norm (bf16 output) | ported, 0.0085 ms; exact on the affine path, scale path unverified |
| yah_vision_softmax_bf16.loom | vision/encoder.hip | vision attention softmax (bf16 output) | ported, 0.0070 ms; exact at uniform scores |
| yah_vision_attention_rows_bf16.loom | vision/encoder.hip | vision attention head->token transpose | ported, 0.0070 ms; bit-exact vs fixture |
| yah_vision_patchify_bf16.loom | vision/encoder.hip | patch embedding (image to 16x16 patches) | ported, 0.0070 ms; bit-exact vs fixture |
| yah_vision_inject_rows_f32.loom | vision/device_input.hip | embedding row injection (broadcast hc) | ported, 0.0065 ms; exact fixture |
| yah_vision_patch_position_f32.loom | vision/encoder.hip | learned position embedding, bilinear | ported, 0.0080 ms; exact fixture |
| yah_vision_activate_bf16.loom | vision/encoder.hip | GELU epilogue (tanh and erf) | ported, 0.0060/0.0064 ms; saturation exact, no fixture |
| yah_vision_qkv_rope_bf16.loom | vision/encoder.hip | QKV projection + RoPE | ported, 0.0233 ms at count=4; identity exact + rotation bit fixture |
| yah_gemv_f32.loom | gemv.hip | decode GEMV, f32 weight path | ported, 0.0063 ms; exact, no fixture; bf16 path todo |
| yah_rmsnorm_decode_f32.loom | norm.hip | decode RMSNorm (single row) | ported, 0.0065 ms; exact, no fixture |
| yah_perhead_rmsnorm_decode_f32.loom | norm.hip | decode per-head RMSNorm | ported, 0.0070 ms; exact, no fixture |
| yah_sample_prepare_f32.loom | sample.hip | sampling prep (non-finite filter + token map) | ported, 0.0068/0.0073 ms; finite exact, non-finite fixture |
| yah_quant_act_q8_f32.loom | prefill_quant_gemm.hip | Q8_1 activation quantizer, tiled ATB layout | ported, 0.0063 ms; byte-exact fixture |
| yah_quant_act_swiglu_q8_f32.loom | prefill_quant_gemm.hip | SwiGLU epilogue + Q8_1 quantize | ported, 0.0054/0.0077 ms; byte-exact fixture |
| yah_requant_int4_f32.loom | prefill_quant_gemm.hip | Q8_1 to int4 requantize in place | ported, 0.0128 ms; byte-exact in/out fixtures; env clip path skipped |
| yah_zero_q8_act_tail_i8.loom | prefill_quant_gemm.hip | zero the slack tokens of the last Q8_1 tile | ported, 0.0061 ms; byte-exact in/out fixtures |
| yah_fused_rmsnorm_q8_f32.loom | prefill_quant_gemm.hip | fused RMSNorm + Q8_1 quantize | ported, 0.0068 ms; byte-exact fixture (constant row) |
| - | vision/encoder.hip, device_input.hip | vision tower: Patchify, PatchPosition, QkvRope, AttentionRows, Softmax, LayerNorm, Activate, InjectRows | all eight **ported** |

## Remaining inventory

From `grep -c '__global__ void'` over `engine/gpu/ported/src/models/qwen`. Roughly
100 kernels; 61 are ported. Ordered by share of prefill time where the model-level
profile gives one, so the expensive paths move first rather than the convenient ones.

| Area | File | Kernels |
| --- | --- | --- |
| DeltaNet / SSM (5.1%) | ssm_row_split.hip | all five kernels **ported** (row split, prep alpha/beta, prep K/Q, conv, post-norm gate fp32 + fp16). The 5.1% prefill block is now covered end to end. |
| DeltaNet / SSM | ssm_recurrence.hip | BatchedDeltaNetRecurrenceKernel, BatchedSSMConvKernel, BatchedSSMPostNormGateKernel |
| DeltaNet / SSM | batched_ssm.hip, ssm.hip, ssm_decode_recurrence.hip | BatchedSSMPostNormGateKernel, FusedSSMInputProjectionsKernel, SSMConvKernel, CaptureBatchedSsmReplayKernel |
| attention (2.1%) | attention_wmma.hip | PackAttentionHeads, PackTiledAttentionKvKernel, SyncTiledAttentionKvPrefixKernel |
| attention | attention_batched.hip | BatchedAttentionKernel, CausalSoftmaxKernel, WriteBatchedKVCacheKernel, ApplyAttentionGateKernel |
| attention | attention_tile.hip, attention_decode_graph.hip, attention_decode.hip | tiled, decode-online, split-K, and KV-write variants |
| QKV projection | qkv.hip | FusedQKVProjectionsKernel |
| fused RoPE | prefill_rope.hip | BatchedFusedQKNormRoPEKvWriteKernel, BatchedRoPEKernel |
| fused.hip | fused.hip | FusedQKNormRoPEKvWriteKernel |
| dequant to bf16 | prefill_gemm.hip | Q4_K/Q5_K/Q6_K/Q8_0/Q8_1 and elementwise dequant, FloatToBfloat16Kernel |
| W8A8 + fused quant | prefill_quant_gemm.hip | **ported**: QuantizeActivationToQ8_1Kernel, BatchedFusedSwiGLUQuantizeQ8_1Kernel, RequantizeActivationInt4Kernel (no-clip path), ZeroQ8ActTailKernel, BatchedFusedRMSNormQuantizeQ8_1Kernel (tiled layout + sum sidecar). **todo**: W8A8BlockedWmmaGEMMKernel, the clip variant, the fused SSM-norm quantize kernel, BatchedQuantGEMVKernel |
| f16 conversion set | prefill_fp16.hip | **ported**: HalfCast, AtbExpandHeadFp16, AtbAddHeadFp32, HalfNorm, HalfNorm5120, Bfp16RoundTripFp16. **todo**: AtbEncodeA, AtbDecodeC, AtbDecodeSwiGLU, AtbRepack(+Slice) |
| GEMV (1.5%) | gemv.hip, gemv_quant.hip | **ported**: FastGEMVBlockKernel f32 path. **todo**: its bf16 path, and all of gemv_quant.hip |
| sampling | sample.hip | **ported**: PrepareSamplingKernel. **todo**: PrepareCandidateLogits, ApplySparsePenalties, batched argmax, linear/sorted sampling, the speculative segment set |
| vision | vision/encoder.hip, vision/device_input.hip | all eight kernels **ported** (Patchify, PatchPosition, QkvRope, AttentionRows, Softmax, LayerNorm, Activate, BiasResidual, Finish, InjectRows) |
| decode leftovers | embed.hip, rope.hip, norm.hip, residual.hip, unpack.hip, swiglu.hip | **ported**: residual.hip, unpack.hip, norm.hip (both). **todo**: RoPEPtr, EmbeddingLookupPtr, FastFusedSwiGLUGEMVBlockKernel |
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
  fail verification ("attribute step has kind 1, expected 2"). The converse also
  holds: an integer tensor (`tensor<4608xi8>`) wants integer literals, and float
  literals fail with "attribute 'step' has kind 2, expected 1".
- **`config.get` must be repeated inside the launch body.** The launch-config region
  is a separate scope, so a value fetched only there is undefined in the body.
- **Not every scalar op has a target contract.** `scalar.isfinitef` compiles under
  loom-check and then fails with "target 'x' has no target-low contract for
  'scalar.isfinitef'". A bound comparison covers NaN and +inf instead. Check the
  contract for any op beyond arithmetic before building a kernel around it.
  Transcendentals additionally need the fast-math flag under the amdgpu-math
  policy: `scalar.expf` fails with "rejected scalar.expf for expf in scalar lanes
  of f32 under math.exp.exact_f32" and is fixed by `scalar.expf<afn>`. The same
  applies to `sinf`, `cosf` and `powf`.
- **The formatter rewrites large float literals** (-3e38 becomes -3.0000000000000001e+38),
  so a later string replace against the original spelling silently fails. Re-read the
  file after formatting before patching it.
- There is no `scalar.select`; use `scf.if` with results. `vector.select` exists.
- **`index.cmp` returns i1, and `index.andi` does not accept i1.** Select a row band with
  one biased unsigned comparison instead: `(m - band_start) < band_size` is true exactly
  inside the band and wraps out of range below it. Equality is spelled `eq`
  (`index.cmp eq, %lane, %c0`), not `ueq` or `ieq`; an earlier note here claimed
  there was no equality predicate at all, and that was wrong. The biased
  `(t - N) < 1` form is still useful when a single `ult` has to carry the test.
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
- **`index.assume` does not feed the memory-footprint analysis, and neither does
  `index.rem`.** The footprint bounds come from `config.decl ... where [range(...)]`
  and from branch conditions the analysis can correlate. A read indexed by an
  expression the analysis cannot bound (here `state[c*4 + batch + j]` inside a
  branch that guarantees `batch + j < 4`) makes it declare a footprint past the end
  of the buffer. `loom_preflight.py` refuses to run that, correctly. The fix is to
  make the memory indices static: load the four ring slots with constant indices
  and select among the registers, rather than indexing memory with the runtime slot.
  The same kernel went from REFUSING to OK with no change in behavior.
- **`scalar.log1pf` has no AMDGPU target-low contract.** Use
  `scalar.logf<afn>(1 + scalar.expf<afn>(x))`; the `logf` recipe carries the afn
  requirement through. `scalar.softplusf` and `scalar.logisticf` exist but expand
  to `exp2`/`log2` recipes that then need fast-math propagation the source op does
  not give them, so the explicit form is the one that compiles today.
- **`index.assume` needs a predicate list.** `index.assume %v : index` fails to
  parse (`unexpected token ':'`); it is `index.assume %v [range(%v, 0, 3)] : index`.
  Prefer removing the need for the assume over writing one, since the footprint
  analysis does not consume it anyway.
- **A softmax at uniform scores has an exact expectation.** The max subtracts out
  exactly, `exp(0)` is exactly 1, and the sum is the visible count, so no
  transcendental is ever evaluated away from zero. That makes a causal softmax
  checkable without a tolerance: a fully visible row is exactly
  `1/sequence_length`, and a one-key row is exactly `1` followed by zeros. The
  kernel's own `+1e-9` guard is harmless in fp32 because `1 + 1e-9` and `4 + 1e-9`
  both round back to their integer.
- **`buffer.alloca<workgroup>` takes an `offset` byte length, not an `index`.**
  Build the size as an index and cast:
  `%bytes = index.cast %len : index to offset`. Passing an index is
  `TYPE/003: operand 'byte_length' has type index, expected offset`. A runtime
  size is accepted, so a workgroup scratch can be sized by a `config` instead of
  being a fixed constant.
- **`scalar.sitofp` rejects `index`.** Cast first:
  `%i = index.cast %n : index to i32` then `%f = scalar.sitofp %i : i32 to f32`.
  The failure is `TYPE/003: operand 'input' has type index, expected integer`.
- **`kernel.subgroup.reduce<addf>` returns the reduction in every lane**, not just
  lane 0, so a port can keep the HIP `if (lane == 0)` guard verbatim and have it
  write the same value the butterflies produced. The three reductions in the K/Q
  prologue share one guard.
- **When an fp16 result would round, rescale the input so the expectation stays an
  arithmetic sequence.** The unscaled f16 Hadamard sums to `15872 + 32*e`, which
  fp16 cannot represent, so no single `check.generate.iota` describes it. Dividing
  `h` by 16 makes the sum `992 + 2*e`, exactly representable and still an iota. The
  transform itself is unchanged; only the fixture's magnitude moves.
- **A per-block kernel whose every block is identical has a periodic expectation.**
  The dequant kernels multiply one scale by a per-block code ramp, so if every block
  carries the same scale and the same codes, the flat output is that ramp repeated.
  `check.generate.iota offset(...) step(...) period(block_elems)` states it exactly,
  and only the packed input needs a fixture. Choosing the scale so the products stay
  exact in bf16 (multiples of 0.5 below 128) keeps the comparison at `atol=0`.
- **`bf16` is a first-class check type.** `check.generate.fill`, `check.generate.iota`
  and `check.expect.close` all accept `tensor<...xbf16>`, and `scalar.fptrunc %v : f32
  to bf16` is a direct narrowing with no separate lowering recipe. Eight explicit
  mantissa bits means every integer up to 256 round-trips exactly, which is what
  lets a bf16 conversion be checked with `atol=0` and no fixture.
- **A bf16 expectation can be a bit fixture, and `check.tensor.view` is how.** numpy
  has no bf16 dtype, so a bf16 value the oracle computes cannot be written to an
  `.npy` directly. Write the raw 16-bit patterns as `int16` and reinterpret the
  kernel output in the case:
  `%bits = check.tensor.view %out offset(0) : tensor<512xbf16> -> tensor<512xi16>`,
  then `check.expect.equal` against the fixture. This made the Q5_K port checkable
  exactly, and it also confirmed that `scalar.fptrunc f32 to bf16` and
  `hip_bfloat16(float)` round identically (round to nearest even).
- **Split a power-of-two index division into loop levels instead of dividing.**
  `index.div` by a constant lowers to `index.shrui`, which the AMDGPU register-unit
  constraint rejects. The Q6_K decoder needs `segment & 1`, `segment >> 1` and
  `lane / 16`; writing the element loop as nested passes over `seg_hi`, `seg_lo`,
  `l16` and `lane16` supplies all three as loop variables and removes the division
  entirely. The Q5_K decoder does the same for its `/64` and `/32`.
- **`RoundActivation` in `prefill_fp16.hip` is `scalar.fptrunc`.** It is an empty
  `asm volatile` barrier followed by `__float2half_rn`, so it pins which rounding
  boundary the source sees and emits no instruction. Ports of `HalfNorm`,
  `HalfNorm5120` and `AtbDecodeSwiGLU` can use `scalar.fptrunc %v : f32 to f16`
  directly; the barrier is a codegen concern Loom does not need a source form for.
- **`frexpf` and `ldexpf` have no AMDGPU target contract either.** Both are reachable
  from the fp32 bit pattern: the frexp exponent of a normal positive `a` is
  `((bitcast<a> >> 23) & 0xFF) - 126`, and `2^(e - bits)` is an fp32 whose exponent
  field is `(e - bits + 127) << 23`, built with `scalar.bitcast` in both directions.
  `scalar.roundevenf` is `rintf`. Subnormal `amax` or a subnormal step are outside
  this substitution and are not covered.
- **A failing correctness expectation can be reported as `state: "skipped"`.**
  `loom_run.sh` treats anything other than `ok` as a failure, so it stops, but the
  benchmark row alone is misleading: the detail is in
  `work_items[].correctness.failed_sample_count` and in `failed_samples[].expectation
  failures`, which name the element index, the actual value and the expected value.
  Read those before assuming a case was not applicable.
- **Re-tile a tuned kernel freely for the first port; say so in the header.**
  `BatchedDeltaNetRowSplitKernel` is templated over four orthogonal tile choices
  with DPP reductions and LDS staging. The port picks the instantiation whose
  `kLanesPerRow` is 1, which drops every cross-lane reduction and every barrier,
  and documents that choice. Coverage is what the port owes; the HIP file's own
  comments own the tiling, and a re-tile that is 20x slower is still correct.
- **Fold the token count held constant when a recurrence would diverge.** The
  DeltaNet state walks `sigma' = 1 - 127*sigma`, which leaves exact f32 after three
  tokens, so the analytic case stops at two. Repeating `(alpha, beta) = (0, 1)` via
  `check.generate.iota ... period(2)` zeroes the state readout, which decouples the
  trajectory from its history and keeps a production-shaped case exact for any
  batch. Use the small case to prove the state dependence and the large one to prove
  the port holds at scale.
- **Infinity comes from an *overflowing* literal, not from the largest finite one.**
  `scalar.constant 3.4028234663852886e+38 : f32` is FLT_MAX and stays finite;
  `4.0e+38` overflows to +infinity and `-4.0e+38` to -infinity. The constant folder
  turns the overflowing spelling into the infinity, and the check only catches the
  mistake when it distinguishes finite from infinite, which is exactly what an
  `isfinite` substitution does. `scalar.isfinitef` has no target-low contract, so
  the source is `cmpf ole %|v|, 3.4028234663852886e+38`; a fill of the finite
  maximum passes that test, a fill of `3.5e38` does not.
- **A derived index needs an explicit two-sided clamp before it reaches memory.**
  The footprint analysis proves origins from the config ranges and the arithmetic
  that produced them; it does not use an `scf.if` condition, a loop guard, or a
  float-to-int conversion to narrow a range. Three separate cases in the vision
  ports show the three forms: `(unsigned)fy` via `scalar.fptosi` gave an unbounded
  lower bound (clamp with `scalar.maxsi %v, %c0i` *and* `scalar.minsi`), the taken
  arm of `scf.if %lt -> (index)` for `pair = d < 36 ? d : d - 36` gave an unbounded
  lower bound in the not-taken arm (compute the subtract in `i32` and clamp the
  result), and the `d < 72` workgroup guard did not narrow `d = lane + pass*32` for
  the analysis (it still saw `d <= 95`). The fix for the last one is a clamped copy
  of the index used *only* for addressing: `dc = min(d, 71)` inside the guarded
  branch is bit-identical to `d` on every live lane, but it gives the analysis the
  provable bound the guard does not. Without these the compile fails `SUBRANGE/023`
  (lower bound) or `SUBRANGE/024` (upper bound), or the declared envelope silently
  grows past the case binding and `loom_preflight.py` refuses the run.
- **A transcendental bit-fixture can be exact even though the op is approximate.**
  `yah_vision_qkv_rope_bf16.loom` compares bf16 `cos`/`sin`/`powf` output against a
  fixture computed with libm, and it is bit-identical: eight bf16 mantissa bits
  absorb the afn approximation error (a few f32 ulps) for every element on this
  case. That is a property of the case, not a guarantee -- an element sitting on a
  bf16 rounding boundary would still differ -- but it is enough to pin the rotation
  formula, the band switch and the angle, which is what the port owes.
- **`scalar.negi` is integer negate; the float negate is `scalar.negf`.** Reaching
  for `negi` on an f32 fails with `TYPE/003: operand input has type f32, expected
  integer`, and `scalar.subf 0.0, %x` also works. (An earlier revision of this
  note claimed there was no `negf`; that was wrong.)
- **A `check.tensor.view` may change element size.** `yah_quant_act_q8_f32.loom`
  writes one buffer that is int8 in its payload region and fp32 in its scale and
  sum regions; `check.tensor.view %y offset(0) : tensor<704xi8> -> tensor<176xf32>`
  is legal (the op only requires the byte range to fit), so the whole mixed-layout
  output can still be compared as raw bytes against one fixture.
- **A stale report file can turn a failed run into a pass.** `iree-benchmark-loom`
  exits non-zero and writes no output when a `check.file.read.npy` target is missing,
  so a runner that only checks the JSON *after* the run reads back the previous
  case report and prints `state: ok`. This was a real false pass on
  `yah_zero_q8_act_tail_i8.loom` before `loom_run.sh` was fixed. The runner now
  writes each report to a fresh `mktemp` file and fails if it is empty. Any new
  harness that runs a case must clear or recreate its report path first; a missing
  fixture is otherwise indistinguishable from a passing one.
- **A whole-workgroup reduction is one op, and `index.andi` is the lane mask.**
  `kernel.workgroup.reduce<addf> %v : f32` replaces a HIP shared-memory tree and
  broadcasts to every thread, so a fused norm needs no LDS staging to share
  `inv_rms`. `index.and` does not exist; the mask is `index.andi %tid, %c31`. A
  constant `index.div %tid, %c32` is still rejected, so the wave index comes from
  `scalar.shrui` on the i32 cast.
