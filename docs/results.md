# Results

RESULTS_PLACEHOLDER

## HIP baselines (final)

The HIP engine (`yah-run`, the hand-written HIP port this project started from) was removed on 2026-10-02. Its last state is tagged `hip-final`. These are its final numbers, measured on the same box, model (Qwen3.8-27B IQ4_XS) and prompts as Loom, one round each with the cool-down rules in [AGENTS.md](../AGENTS.md).

Prefill, 2048-token chunks:

| tokens | HIP | Loom (p71) |
|---|---:|---:|
| 2048 | 3381.7 ms | 3132.2 ms |
| 8192 | 14254.3 ms | 13154.0 ms |
| 8192 (chunked, before decode) | 14431.7 ms | - |
| 30720 (chunked, before decode) | 62596.1 ms | - |

Decode, 64 greedy tokens after the prompt (HIP's GEMV decode; Loom numbers from the same day, before quantized KV):

| context | HIP | Loom |
|---|---:|---:|
| 5 (prompt fed through decode) | 14.58 tok/s (68.6 ms) | 16.43 tok/s (60.9 ms) |
| 2048 | 14.07 tok/s (71.1 ms) | 15.97 tok/s (62.6 ms) |
| 8192 | 13.46 tok/s (74.3 ms) | 15.16 tok/s (65.9 ms) |
| 30720 | 11.75 tok/s (85.1 ms) | 13.31 tok/s (75.2 ms) |

Both engines produced the same 64 tokens at every context.

Prefill kernels, same real bytes, rocprofv3 cycles (Loom kernels launched through `loomhip`), 2026-10-01:

| kernel | Loom M cycles (% of WMMA floor) | HIP M cycles (% of floor) | Loom / HIP |
|---|---:|---:|---:|
| IQ4_XS 17408x5120 | 26.45 (67%) | 23.15 (77%) | 1.14 |
| Q4_K 10240x5120 | 15.50 (68%) | 13.71 (77%) | 1.13 |
| IQ3_S 17408x5120 | 27.96 (64%) | 25.91 (69%) | 1.08 |
| IQ3_XXS 17408x5120 | 26.99 (66%) | 25.80 (69%) | 1.05 |
| DeltaNet B=2048 | 4.756 | 4.762 | 1.00 |
| Attention, pp8192 real layer-3 inputs | 80.23 | 79.4 | 1.01 |

HIP reaches 77% of the WMMA floor on IQ4_XS and Q4_K by deferring the weight decode (raw bytes held in registers, decoded after the barrier while queued WMMAs drain). Loom wins the whole prefill despite slower GEMMs through fusion and fewer passes.
