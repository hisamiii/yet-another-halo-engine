# Low-precision (int8/int4 WMMA) prefill GEMMs on gfx1151: research notes

2026-10-01. No GPU work was run. The only local commands were a CPU-only `clang -S` and `llvm-mc`, plus the AMD matrix calculator script.
Labels: **[F]** = fact from spec, source code or docs. **[M]** = measured by someone, with the source given. **[I]** = my inference or estimate.

## TL;DR
- **[F]** On RDNA3/3.5, `v_wmma_i32_16x16x16_iu8` has the **same rate as f16**: 32 cycles and 1024 ops/WGP/clk. `iu4` runs at 2x: 16 cycles and 2048 ops/WGP/clk. WMMA cannot co-execute with VALU. So int8 gives no extra arithmetic throughput on gfx1151. The only wins are fewer VALU ops for decode and fewer LDS/VGPR bytes.
- **[I]** For IQ4_XS, applying the per-32 sub-block scales has a floor of about 1 VALU issue per output element per 32 K. That is **about 4 VALU per WMMA**, plus about 1–2 for decode, so **about 5–7 per WMMA in total**. This roughly matches today's 4–7 and gives no clear win. Q6_K and Q3_K, with 16-element sub-blocks, are strictly worse.
- **[M]** llama.cpp already ships int8-WMMA MMQ on RDNA3.5. On Strix Halo, Llama-8B IQ4_XS pp2048 runs at 1162 t/s, which is only about 27% of f16 WMMA peak by my estimate **[I]**. llama.cpp falls back to dequantize + hipBLAS for Q6_K/Q2_K at large batch.
- **[I]** A more promising idea is to **dequantize each layer to an f16 scratch buffer and run a decode-free f16 GEMM** at large M. The details are in §5.

## 0. What Loom supports (read from /home/q/hrx/loom)
- **[F]** The RDNA3 descriptor set (`py/loom/target/arch/amdgpu/descriptors/sets.py:1471-1500`, overlays at `matrix.py:310-452`) contains these, each with a `.acc_zero` form:
  - `v_wmma_i32_16x16x16_iu8`: A and B are 4 VGPRs each, the accumulator is 8 VGPRs.
  - `..._iu8_w64`: the accumulator is 4 VGPRs.
  - `v_wmma_i32_16x16x16_iu4`: A and B are 2 VGPRs each.
  - `..._iu4_w64`.
  - All are VOP3P with opcodes 0x44 and 0x45. `OP_SEL_HI` is fixed at 7. The accumulator is destructive (tied). Immediates are `neg_lo` (3 bits, sign-select), `neg_hi` (mirrored) and `clamp` (default 0).
  - The RDNA3.5 set derives from the RDNA3 base (`rdna3.py:138`).
- **[F]** `types.h` defines these numeric kinds: `IU8=8` ("per-operand sign-selected"), `I8=7`, `IU4=10`, `I4=9`. A request for `I8`/`I4` matches an `IU8`/`IU4` descriptor through `CONTRACT_FLAG_SIGN_SELECT` (`contract.c:338-345`).
- **[F]** The fragment layouts are `RDNA3_WMMAR3_I32_16X16X16_IU8` (66), `_IU8_W64` (67), `_IU4` (68) and `_IU4_W64` (69). `contract_test.cc:328-344` maps gfx11 wave32 iu8 to 4 input VGPRs × 16 elements with an 8-VGPR accumulator, and iu4 to 2 input VGPRs.
- **[F]** Source form, from `test/source_low/source_low_mma.loom-test:3246-3264` (gfx1100):
  ```
  %lhs_schema = encoding.define #encoding.operand<element_format=i8, payload_elements=16, payload_registers=4> : encoding<schema>
  %lhs = vector.fragment<lhs> %lhs_data shape [%m, %k] using {schema = %lhs_schema : encoding<schema>} : vector<4xi32>
  %rhs = vector.fragment<rhs> %rhs_data shape [%k, %n] using {schema = %rhs_schema ...} : vector<4xi32>
  %init = vector.fragment<init> %init_data shape [%m, %n] : vector<8xi32>
  %result = vector.mma %lhs, %rhs, %init : vector<4xi32>, vector<4xi32>, vector<8xi32>
  // lowers to: v_wmma_i32_16x16x16_iu8 %lhs, %rhs, %acc {neg_lo = 3, neg_hi = 3}
  ```
  `element_format=u8` clears the sign bit. This is shown in the gfx12 swmmac tests, where an i8 lhs with a u8 rhs gives `neg_lo = 1`. **[I]** gfx11 dense iu8 should behave the same, but no gfx11 mixed-sign test exists.
- **[F] Possible bug:** on gfx11, Loom sets **`neg_hi = 3`** (`mirrors_sign_select_to_high_halves=True`).
  - clang 22 for gfx1151 emits only `neg_lo:[1,1,0]` (CPU compile I ran).
  - AMD's matrix calculator says "NEG_HI must be 0 for integers".
  - `llvm-mc` does encode `neg_hi` (byte 0x43 instead of 0x40).
  - **Run a mixed-sign correctness test first, or force `neg_hi = 0`.**
- **[F]** Loom's RDNA3 schedule model gives every WMMA a 32-cycle issue separation (`rdna3.py:89-97`), so it models iu4 too pessimistically. RDNA3 int WMMA descriptors are tagged `INTEGER_MATRIX_COEXECUTION_SPACING`, but the gfx1100 test verifies cleanly; the error appears only in the gfx1250-a0 errata test.
- **[F]** The C/D layout matters for the scale design. Per the calculator, `D[i][j]` is in VGPR `i/2`, lane `16*(i%2)+j`. **Every lane holds a single column `j`.** A and B are replicated across lanes 0-15 and 16-31.

## 1. Throughput: spec vs measured
- **[F spec]** AMD matrix instruction calculator, RDNA3 (I ran it locally):
  - f16→f32: 32 cycles, 1024 FLOPs/WGP/clk.
  - iu8: 32 cycles, 1024 ops/WGP/clk.
  - iu4: 16 cycles, 2048 ops/WGP/clk.
  - "Can co-execute with VALU: False" for all three.
- **[F spec]** gfx1151 at 40 CUs and 2.9 GHz: about 59 TFLOPS f16 and int8, and about 119 TOPS int4. These are computed numbers, not measured ([1bit-MONSTER PR #160](https://github.com/1bit-MONSTER/engine/pull/160)).
- **[F]** RDNA4 differs: iu8 is 2048 ops/CU/clk versus 1024 for f16, which is where int8 really pays ([zolotukhin.ai](https://zolotukhin.ai/blog/2026-07-12-int8-wmma-doubles-rdna4-matrix-rate-q4-k-block-scales-take-half-back/)).
- **[F]** `v_dot4_i32_iu8` gives 4 MAC/lane/clk, which is 128 MAC/clk/SIMD — the same as WMMA iu8 (4096 MAC / 32 cycles). It is not VOPD-capable.
- **[M]** I found no published gfx1151 microbenchmark of iu8 vs f16 WMMA. "Same rate" is spec only.

## 2. How llama.cpp does quantized prefill on RDNA3/3.5
Read from the local checkout `/home/q/llama.cpp-upstream-stock` @ 8172e6577.
- **[F]** `AMD_WMMA_AVAILABLE` is defined for RDNA3 and RDNA4.
  - On RDNA3, `mma()` (`mma.cuh:1326-1331`) issues **2× `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(signed, A, signed, B, acc, clamp=true)`** per 16×16×32 int tile.
  - RDNA3.5 has its own config table (`mmq-config-rdna3-5.cuh`). IQ4_XS uses 128/256 threads, 64/128-row tiles and `MMQ_ITER_K=256`.
- **[F] Scale handling** (`mmq-vec-dot.cuh:143-200`): for each 32-K block, `sum[l] += C.x[l]*dA*dB`.
  - `dA` is the per-32 weight scale, read from LDS per output element.
  - `dB` is the q8_1 per-32 activation scale.
  - That means cvt + mul + fma per element per 32 K: **about 12 VALU per WMMA**, plus LDS reads.
  - Activations are **q8_1: int8 per 32 values with an f16 scale and an f16 `d*Σq` sum**. k-quant mins use the sum.
- **[F] IQ4_XS** (`mmq-load-tiles.cuh:1428`): the codebook is converted to int8 in LDS with `get_int_from_table_16`, which is 6× `v_perm_b32` plus masks, about 15 VALU per 8 weights. The scale `d*(ls-32)` is stored as f32 per 32. This is exactly Q8_0, so the stored representation is exact.
- **[F] Selection** (`mmq.cu:356-381`): on RDNA3.x, MMQ is used for all types except these, which go to hipBLAS above certain batch sizes:
  - Q2_K: above ne11 128.
  - Q6_K: above 128 on RDNA3.0 and 256 on RDNA3.5.
  - IQ2_XS/S: above 128 on RDNA3.0 only.
- **[M]** [PR #18666](https://github.com/ggml-org/llama.cpp/pull/18666), Strix Halo, Llama 8B, pp2048:
  - IQ4_XS ub512: 1162 t/s (MMQ).
  - Q4_K_S ub1024: 1099–1102 t/s.
  - Q6_K ub1024: 736 t/s with MMQ vs **877 t/s with hipBLAS**.
  - **[I]** 2·7e9·1162 ≈ 16 TFLOP/s, about 27% of f16 peak.
- **[M]** [PR #17576](https://github.com/ggml-org/llama.cpp/pull/17576) (int8 WMMA MMQ for RDNA3) measured 3.0–3.4× over dp4a MMQ on Strix Halo at batch 64.
- **[M]** [Issue #17917](https://github.com/ggml-org/llama.cpp/issues/17917): GPT-OSS-120B MXFP4 pp2048 on Strix Halo dropped from 900 to 543 t/s after that change. The tuning in [PR #18537](https://github.com/ggml-org/llama.cpp/pull/18537) followed.
- **[M]** On RDNA4 (RX 9070 XT, Qwen3.5-9B) an independent engine measured ([zolotukhin.ai](https://zolotukhin.ai/blog/2026-07-12-int8-wmma-doubles-rdna4-matrix-rate-q4-k-block-scales-take-half-back/)):
  - fp16 WMMA: 665 t/s.
  - int8 with fp32 descale: 700.
  - Adding integer sub-scale accumulation: 775.
  - Adding K=32 fusion and packed unpack: 815.
  - llama.cpp: 973.
  - The descale "tax" took most of the 2× hardware gain there. **[I]** On RDNA3, with a 1× hardware gain, the same tax is a net loss unless it is driven to about 0.

## 3. The per-32 scale problem: options and VALU/WMMA estimates
**[F]** Each 16×16 WMMA tile spans 32 lanes × 8 accumulator values. Per 32 K (2 WMMAs), each lane needs 8 scale applications. **With 1 VALU op per element, the floor is 4 VALU per WMMA.** There is no `v_pk_fma_f32` on gfx11, and llvm-mc rejects it. VOPD has `v_dual_fmac_f32`, `v_dual_sub_f32`, `v_dual_add_nc_u32` and `v_dual_dot2acc_f32_f16`, but **no `v_dual_mad_i32_i24`** (verified with llvm-mc).

**[F] Exact folding into int8 is impossible for GGUF formats**, because each format's integer product exceeds int8:
- IQ4_XS: |cb|·|ls−32| ≤ 127·32 = 4064.
- Q4_K: 15·63.
- IQ3_S: 15·31.
- IQ3_XXS: 62·31.

**[I]** Requantizing to a per-256 int8 scale is lossy. Sub-block scales within a super-block differ by up to 32×, so small sub-blocks would drop to about 3 bits. Only Q8_0-like per-32 int8 is exact.

**Survey:**
- **QServe** ([2405.04532](https://arxiv.org/abs/2405.04532)): two-level quantization. A per-channel f16 scale gives int8, and then int8 is split into u4 codes with an *integer* group scale and zero. The "protective range" guarantees the dequantized value fits in 8 bits. Subtraction happens after multiplication. The main loop is pure int8 MMA, and the per-channel scale is applied only in the epilogue.
- **LiquidGEMM** ([2509.01229](https://arxiv.org/abs/2509.01229)): the same idea in a UINT8 domain. It needs 2 instructions (IMAD + XOR) per 4 weights, and is up to 2.9× faster than earlier W4A8 kernels.
- **QQQ** ([2406.09904](https://arxiv.org/abs/2406.09904)): per-group W4A8 built on Marlin. The f16 group scale is divided by the channel scale ("FusedDequantQuant") and the weight is requantized to int8 in registers.
- **[I] What these share:** they all *design the weight format* so the 4→8-bit step is exact in int8. GGUF formats are not designed this way, so these tricks only apply after a lossy requantization.

**Concrete IQ4_XS-on-iu8 designs.** All assume weights are the **B operand**, so each lane's column `n` has one scale per sub-block, and **per-token (or per-256) activation scales**.

| Variant | Per element per 32 K | VALU/WMMA [I] |
|---|---|---|
| A. llama.cpp style (per-32 act scale) | cvt + mul + fma | ~12 |
| B1. per-token act, f32 | cvt + fmac | ~8 |
| B2. int super-block accumulator | `v_mad_i32_i24` (\|C\|≤516k fits in 24 bits; ls−32 is signed 6-bit); then per 256: cvt + fma | ~4 + ~1 |
| B3. cvt-free | init WMMA C to `0x4B400000`, so D reinterpreted as f32 is 1.5·2²³+dot exactly (\|dot\|<2²²); then VOPD pairs `v_dual_sub_f32` / `v_dual_fmac_f32` | ~4 issues, if VOPD bank rules allow |
| Decode IQ4_XS→int8 LDS (LUT, no cvt) | ≈240/M_tile | ~1.9 at M_t=128 |

- **[I] Net:** about 5.5–7 VALU/WMMA, compared with today's 4–7. Breakeven at best.
- **[I]** B3 must subtract the magic number *before* scaling. Folding the offset into the f32 accumulator loses about 7–13 mantissa bits.
- **[I]** Accumulator VGPRs grow: a temp i32 plus an i32 or f32 accumulator, compared with a single f32 accumulator today.
- **[I]** Q6_K/Q3_K have a scale every 16 K, so the scale cost doubles to about 8 per WMMA.
- **[I]** For the Q4_K/Q5_K mins (dmin·Σm_j·S_j), use a skinny side GEMM over K/32 with activation block sums. That is about 3% extra matrix work, compared with about 4 VALU/WMMA if done per element.

## 4. Activation quantization quality
- **[F]** In q8_1, `d = absmax/127` per 32 values. llama.cpp uses it by default on every GPU with int8 MMA. **[I]** Broad use is evidence that per-32 A8 is benign on top of 4-bit weights.
- **[F]** SmoothQuant ([2211.10438](https://arxiv.org/abs/2211.10438)) and LLM.int8() ([2208.07339](https://arxiv.org/abs/2208.07339)) show that per-token int8 breaks down because of systematic channel outliers in large models.
  - "Massive activations" ([2402.17762](https://arxiv.org/abs/2402.17762)) are 10³–10⁴× outliers in a few dimensions and tokens.
  - **[I]** The down_proj input (the SwiGLU output) is the riskiest place for per-token scales.
- **[I]** Per-256 activation scales aligned to IQ4_XS super-blocks cost only about 1–1.5 VALU/WMMA in B2. They are a reasonable compromise between per-token and per-32. Validate with `llama-perplexity --kl-divergence` against the f16 path.
- **[F/M]** Int4 activations need rotations.
  - Plain W4A4 with RTN is catastrophic: Llama-3-70B perplexity on WikiText-2 is about 60 vs 2.86.
  - QuaRot ([2404.00456](https://arxiv.org/abs/2404.00456)) keeps Llama-2-70B within 0.47 ppl using Hadamard rotations folded into the weights. SpinQuant ([2405.16406](https://arxiv.org/abs/2405.16406)) learns the rotations.
  - **[I]** iu4 also cannot represent the IQ4_XS non-linear codebook, which needs int8. Splitting A8 or W8 into nibbles costs ≥2 iu4 ops, which equals 1 iu8, so there is no gain.
  - Verdict: **iu4 is not viable** for this project without retraining or rotation.

## 5. Other ideas
1. **[I] Decode-free f16 GEMM using JIT dequantization to scratch.** This is the strongest idea.
   - For 5120×17408 at M=2048: 365 GFLOP, which is about 6.2 ms at peak or about 10 ms at 60% efficiency.
   - f16 weights are 178 MB. Reading them takes about 0.85 ms at about 210 GB/s, so the GEMM stays compute-bound.
   - A dequantize pass (read 47 MB, write 178 MB) takes about 1.1 ms. That is about 10% of the GEMM at M=2048 and about 3% at M=8192.
   - Fused decode at 4–7 VALU per 32-cycle WMMA costs up to about 12–22% of issue time.
   - So at M≥4096 the separate pass should win, *if* a decode-free kernel actually runs faster.
   - **Falsifying measurement:** time the current kernel against a variant that loads pre-dequantized f16 straight into LDS. The existing `yah-hal-sd-ab{decode,none,mma}` ablations may already answer this.
2. **[I] VALU is not the whole gap.** 4–7 one-cycle VALU per 32-cycle WMMA explains at most about 18% of a 37–44% gap. Profile LDS and waits before investing in int8.
3. **[I] Int8 in LDS halves LDS bytes and `ds_read`s per WMMA**: one `ds_read_b128` fills a whole iu8 fragment. This matters only if the kernel is LDS-bound. Test that first.
4. **[I] `v_dot4_i32_iu8`** runs at the same peak as WMMA iu8 and cannot overlap it. It is useful only for M<16 tails.
5. **[I] Store an exact Q8_0-style int8 shadow copy** (1.125 B per weight) to remove the LUT decode. This leaves only the ~4 VALU/WMMA scale floor, so an f16 scratch copy (idea 1) is strictly better on RDNA3.5.

## Sources
All linked inline. Local: llama.cpp-upstream-stock @8172e6577, Loom at /home/q/hrx/loom, and the [AMD matrix calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) (run on the CPU). Also see [GPUOpen WMMA on RDNA3](https://gpuopen.com/learn/wmma_on_rdna3/).
