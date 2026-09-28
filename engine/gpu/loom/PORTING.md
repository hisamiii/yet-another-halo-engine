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
| - | engine/kv/kv_quant.hip | fp16 quantize/dequantize, q8/q4 block formats | todo |
| - | vision/encoder.hip, device_input.hip | vision tower | todo |

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
