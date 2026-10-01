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

## Price list next to WMMA (wmma_price<KIND,N>, N per WMMA, 16 waves/SIMD; base 34.01 cycles/WMMA)

| instruction | x1 per WMMA | x4 per WMMA (extra cycles each) |
|---|---:|---:|
| ds_load_b32 | +2.2 | +6.1 |
| ds_load_b64 | | +6.1 |
| ds_load_b128 | +2.4 | +7.7 |
| ds_store_b128 | +0.5 | +5.3 |
| global_load_b128 (L1-resident) | +2.5 | +6.4 |
| v_perm_b32 / v_cvt_f32_ubyte0 / v_bfe_u32 / v_and_b32 / v_cvt_f16_f32, independent | | +1.09..1.18 |
| v_perm_b32, dependent chain | | +2.8 |
| v_cvt_f32_ubyte0, reading a chained register | | +2.2 |
| v_mul_lo_u32, dependent chain | | +4.2 |
| v_fma_mixlo_f16 (tied dst) | | +1.1 |

- LDS cost is per instruction, not per byte (b32 = b64 = most of b128 at 4/WMMA): fewer, wider LDS accesses.
- Global loads are not cheaper than LDS loads.
- Dependent VALU chains cost 2-3x an independent op even at 16 waves/SIMD: decode chains are pricier than their count.

## Issue-bound check of production kernels (tools/sol.py: 34*WMMA + VALU cycles + 3*LDS per SIMD vs SQ_BUSY_CYCLES)

| kernel (real bytes) | cycles/WMMA | non-WMMA VALU/WMMA | LDS/WMMA | bound/measured | barrier wait (wave time) |
|---|---:|---:|---:|---:|---:|
| IQ4_XS kstore 17408x5120 | 43.9 | 5.52 | 1.72 | 102% | 30% |
| IQ3_S kstore | 48.1 | 7.64 | 1.84 | 98% | 19% |
| IQ3_XXS kstore | 46.7 | 6.34 | 1.91 | 99% | 17% |
| Q4_K store 10240x5120 | 46.3 | 6.51 | 1.72 | 99% | 32% |
| attention pp8192 | 63.9 | 16.61 | 4.01 | 98% | 9% |

All at the issue bound: barrier waits and stalls are absorbed by other waves; only instructions per WMMA (VALU, LDS) move cycles. Extended counters: SQ_INST_CYCLES_VALU counts one per instruction (+extra for multi-cycle ops), not WMMA pipe time.
