# gfx1151 measured hardware facts (2026-10-01, research/wmmarate.hip, PMC SQ_BUSY_CYCLES, one round each, 15 s gaps)

Setup: 160 workgroups x 8 waves (16 waves/SIMD requested, 69-82 VGPRs), 8 independent accumulator chains per wave, 4096 iterations.

| variant | cycles per WMMA per SIMD |
|---|---:|
| v_wmma_f32_16x16x16_f16 | 34.01 |
| v_wmma_f32_16x16x16_bf16 | 34.04 |
| v_wmma_i32_16x16x16_iu8 | 34.01 |
| v_wmma_i32_16x16x16_iu4 | 17.01 |

f16 WMMA + V independent v_fma_f32 per WMMA (inline asm, 4 rotating registers):

| V | cycles/WMMA | extra cycles per VALU |
|---:|---:|---:|
| 0 | 34.01 | - |
| 1 | 36.02 | 2.01 |
| 2 | 36.84 | 1.42 |
| 4 | 38.60 | 1.15 |
| 8 | 42.57 | 1.07 |
| 16 | 50.72 | 1.04 |
| 32 | 66.92 | 1.03 |

Conclusions: the matrix pipe takes ~34 cycles per 16x16x16 WMMA per SIMD (f16/bf16/iu8 equal), iu4 is 2x. Every VALU op issued alongside costs ~1 cycle of that pipe (no overlap): the issue model holds. The WMMA floor used in floor.py (32) is ~6% optimistic.

## More (same setup unless noted)

| variant | cycles/WMMA | note |
|---|---:|---|
| dependent chain (1 accumulator), 16 waves/SIMD | 34.00 | latency hidden by other waves |
| dependent chain, 1 wave/SIMD (80 x 32 threads) | 34.07 | back-to-back accumulation has no extra latency (f16 independent chains at 1 wave: 34.12) |
| +1 VOPD pair (v_dual_fmac x2) per WMMA | 36.99 | |
| +4 VOPD pairs | 41.77 | ~1.94 cycles per pair |
| +8 VOPD pairs | 49.38 | ~1.92 cycles per pair: dual issue saves nothing next to WMMA |
| +1 ds_load_b128 per WMMA (conflict-free, 16 B/lane contiguous) | 36.80 | +2.8 |
| +2 ds_load_b128 | 40.60 | +3.3 each |
| +4 ds_load_b128 | 64.65 | +7.7 each: LDS bandwidth-bound, ~128 B/clk per WGP (4 SIMDs x 4 x 512 B / 64.65 cycles) |
| +1/2/4 ds_load_b128 with 64 B lane stride (bank conflicts) | 55.9 / 112 / 224 | conflicts cost ~4x |

Implication: our tile GEMMs issue ~1.7 b128 fragment loads per WMMA (SQ_INSTS_LDS/WMMA ~1.72), i.e. ~100 B/clk per WGP if all are b128: possibly ~80% of LDS bandwidth. If confirmed (SQC_LDS_IDX_ACTIVE vs cycles), fragment bytes (int8 operands, larger per-wave tiles) become a real lever.
