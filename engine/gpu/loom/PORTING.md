# Porting the GPU kernel set to Loom

Status: in progress. Goal is coverage and correctness first; performance tuning is
explicitly deferred.

## Rules for each port

1. Read the HIP source and state the exact semantics in the header comment.
2. Keep the same buffer signature and the same output dtype.
3. Give it an in-source `check.case` with distinct input values, and an expectation
   computed analytically where one exists (identity and all-zero patterns do not
   catch a transposed index).
4. Verify with `loom-check`, then measure with `iree-benchmark-loom --measure=dispatch_complete`.
5. Do not tune during the port. Record the number and move on.

## Status

| Loom file | HIP source | kernel | status |
| --- | --- | --- | --- |
| yah_ffn_gemm_f16.loom | prefill_fp16.hip | batched f16 FFN GEMM | ported, 1.819 ms, 1.34x behind HIP |
| yah_residual_add_f32.loom | prefill_residual.hip | batched residual add | ported, 0.0189 ms at 327680 elements |
| yah_rmsnorm_f32.loom | prefill_norm.hip | batched RMSNorm | ported, 0.0821 ms at 64x5120 |
| - | prefill_norm.hip | per-head RMSNorm | todo |
| yah_swiglu_f32.loom | prefill_swiglu.hip | SwiGLU activation (split form) | ported, 0.0126 ms at 327680 elements |
| yah_rope_f32.loom | rope.hip | RoPE (text path) | ported, 0.0091 ms at 384 pairs |
| yah_embed_f32.loom | embed.hip | embedding lookup (f32 table) | ported, 0.0087 ms at hidden 5120 |
| - | prefill_unpack.hip | QG unpack | todo |
| - | prefill_ssm.hip, ssm_row_split.hip | DeltaNet recurrence | todo |
| - | prefill_attention*.hip, attention_wmma.hip | batched attention | todo |
| - | qkv.hip | QKV projection | todo |
| - | gemv.hip, gemv_quant.hip | decode GEMV | todo |
| - | sample.hip | argmax / sampling | todo |
| - | engine/kv/kv_quant.hip | KV quantization | todo |
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
- Float literals in expectations are canonicalised by the formatter (1.0e-4 -> 0.0001).
- Fragment loads from global memory are what this compiler is good at; LDS staging
  measured slower in all three variants tried.
