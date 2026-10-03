# Results

Qwen3.8-27B IQ4_XS on Strix Halo (gfx1151), Loom kernels through HRX. One round per number, APU cooled to <= 55 C first, 15 s (pp2048) or 30 s (longer) gap, no clock pinning; the p71 prefill rows at tctl 95 (rules in [build-and-run.md](build-and-run.md)). HIP baselines are in the section below this one.

## Current results

Prefill, default set (FA attention, chunked DeltaNet, paged KV; HRX pin a02a5ab94 with engine/hrx/patches), `layers_ms`:

| prompt | set | Loom | HIP (hip-final) | measured |
|---|---|---:|---:|---|
| 2048 | one pass, B = 2048 | 2952.1 ms (694 tok/s) | 3381.7 ms | 2026-10-03, + half_norm 4 rows per workgroup (~4 ms of the 48 ms gain; the round started at 42 C) |
| 8192 | one pass, B = 8192 | 12877-13556 ms (604-636 tok/s) | 14254.3 ms | 2026-10-02, f16 operand placement; see below |
| 8192 | chunked (2048), 32K pools, fp16 KV | 13085.3 ms | 14431.7 ms | 2026-10-02, run before decode |
| 30720 | chunked (2048), 32K pools, fp16 KV | 59004.2 ms (521 tok/s) | 62596.1 ms | 2026-10-02, run before decode |
| 8192 / 30720 | same, kv8a16 | 13270.9 / 60182.6 ms | - | 2026-10-02 |
| 8192 / 30720 | same, kv4a16 | 14111.8 / 60315.3 ms | - | 2026-10-02 |
| 65536 | chunked (2048), fp16 / kv8a16 / kv4a16 | 232.8 / 242.1 / 234.7 s | - | 2026-10-02 |

pp8192 rounds from the same <= 55 C start vary by up to ~5% in wall time with the GPU clock (2.0-2.1 GHz) while their counter cycles match to 0.03%, so compare pp8192 on cycles.

The quantized-KV prefill rows are within run-to-run noise of fp16: their attention kernels cost +12-15% (int-to-f16 decode at staging), and attention is ~3.5% of pp8192.

A prompt that ends inside a chunk (chunked 32K set, fp16 KV, `layers_ms`; the last token's logits are bit-identical to the padded chunk):

| prompt | partial chunk | padded chunk | measured |
|---:|---:|---:|---|
| 500 | 969.7 ms | 3082.4 ms | 2026-10-02 |
| 1025 | 2034.6 ms | 3092.1 ms | 2026-10-02 |
| 1537 | 2786.5 ms | 3104.6 ms | 2026-10-02 |

Through `yah_server` (kv8 serve sets): an 18-token prompt prefills in 0.42 s (0.63 s before the narrow tiles), a 621-token one in 1.33 s (both were ~3.1 s with padded chunks).

Decode, greedy, ms per generated token (mean over 63 steps after the prompt; `loom_forward_pp` with `YAH_GEN=64`, prefill sets as above):

| context | fp16 KV | kv8a16 | kv4a16 | HIP | measured |
|---|---:|---:|---:|---:|---|
| 5 (prompt fed through `loom_decode`) | 60.2 ms (16.6 tok/s) | - | - | 68.6 ms | 2026-10-02 |
| 2048 | 62.6 ms (15.97 tok/s) | - | - | 71.1 ms | 2026-10-02, before the fused DeltaNet conv (-0.6 ms) |
| 8192 | 65.17 ms (15.35 tok/s) | 63.90 ms (15.65 tok/s) | 66.86 ms (14.96 tok/s; profiled run 65.56) | 74.3 ms | 2026-10-02 |
| 30720 | 74.10 ms (13.50 tok/s) | 70.56 ms (14.17 tok/s) | 70.79 ms (14.13 tok/s) | 85.1 ms | 2026-10-02 |

- Tokens: fp16 matches HIP token for token at every context. kv8a16 and kv4a16 give the same 64 tokens as fp16 at 8K; at 30K both switch at token 40 to the same alternative (a near-tie).
- kv4a16's 8K round is above fp16 although its attention kernels are 1.1 ms/token cheaper in the profile: run-to-run variation.

Quantized KV quality, 32K chunked prefill, real kernels, vs fp16 KV (one document each; 99% precision = 100 exp(-KLD at the 99th percentile)):

| document | format | dPPL | mean KLD | 99% precision | same top-1 |
|---|---|---:|---:|---:|---:|
| books | kv8a16 | -0.00% | 0.000010 | 99.91% | 99.80% |
| books | kv4a16 | -0.01% | 0.000994 | 98.87% | 98.36% |
| code | kv8a16 | -0.01% | 0.000034 | 99.92% | 99.90% |
| code | kv4a16 | -0.05% | 0.001040 | 98.34% | 99.52% |
| arXiv paper | kv8a16 | -0.02% | 0.000073 | 99.88% | 99.80% |
| arXiv paper | kv4a16 | +0.21% | 0.003154 | 96.07% | 97.51% |

KV memory, GTT peak above idle (prefill, pg1023 book):

| format | 32K | 64K |
|---|---:|---:|
| fp16 | +4133 MiB | +6504 MiB |
| kv8a16 | +3269 MiB | +4752 MiB |
| kv4a16 | +2816 MiB | +3827 MiB |

## Where the time goes

Prefill, share of `SQ_BUSY_CYCLES` per kernel group (p71 set, HRX counters, one capture each, 2026-10-02):

| kernels | pp2048 | pp8192 |
|---|---:|---:|
| GEMM kstore (q/k/v, qkv, z, alpha/beta, ffn_gate) | 37.5% | 36.8% |
| GEMM kres (attn_output, ssm_out, ffn_down, residual fused) | 29.5% | 28.8% |
| GEMM swiglu (ffn_up) | 21.5% | 21.3% |
| GEMM kqg (attn_q) | 3.7% | 3.6% |
| all GEMMs | 92.2% | 90.5% |
| attention | 1.3% | 3.6% |
| Gated DeltaNet | 2.4% | 2.3% |
| half_norm (2 per layer) | 1.4% | 1.2% |
| conv + prep_kq | 1.2% | 1.1% |
| postnorm | 0.9% | 0.9% |
| RoPE | 0.4% | 0.4% |
| output head GEMV | 0.2% | 0.1% |

- The GEMMs run at ~76% of their WMMA floor (34 cycles per WMMA over 80 SIMDs); the big K = 17408 shapes at 75-83%, the K = 6144 residuals at 60-72% (p56, 2026-10-01). Every GEMM is at 98-102% of its issue bound, so only fewer instructions per WMMA make them faster.
- The non-GEMM kernels are at DRAM bandwidth (180-230 GB/s); only fusion removes their time.
- GPU idle during pp2048: 0.5%.

Decode, short context (60-61 ms per token, 2026-10-02):

| part | ms per token |
|---|---:|
| GEMVs (12.40 GB at 214-236 GB/s) | 54.9 (50.9 at 240 GB/s; SwiGLU ~2.2 and the residual GEMVs ~0.9 of the excess) |
| dependent-dispatch gaps (~560 x ~4 us) | 3.2 |
| DeltaNet | 1.9 (~1.3 at bandwidth) |
| norms, attention, head, rest | ~1 |

- Roofline: 12.40 GB of weights per token / 240 GB/s = 51.7 ms (19.4 tok/s). Realistic floor ~56 ms (weights 50.9 + DeltaNet / attention / norms ~2 + launch gaps ~2.8).
- Attention per token at 8K (dispatch timestamps): fp16 3.55 ms, kv8a16 2.25, kv4a16 2.44.
- Decode attention `part` kernel per call at 30.7K: fp16 651.5 us, kv8a16 362 us, kv4a16 347 us. kv4 is latency-bound at ~91 GB/s (kv8 ~144 GB/s); bandwidth-bound it would be ~160 us, -3 ms per token at 30K.

## Wins that got us here

One line each: what, the measured effect, when the set was current. Prefill numbers are pp2048 `layers_ms` or cycles; decode numbers ms per token.

Prefill:

- Activations straight from global memory instead of an LDS tile (`Tile.afrag`, off): only the decoded weights live in LDS, each wave does 128 rows x 32 tokens (2 activation fragments per k step, loaded from the input), 16 waves x 32 = 512 tokens per decoded weight (decode VALU per WMMA halves). Bit-identical. With the input row-major [tokens][K] it is +36% (a fragment spans 16 token rows 10 KB apart: 16 pages and partial lines per load); with a fragment-major input (`Tile.atiled`: each 16 x 16 tile 512 contiguous bytes; the producer would have to write it) and KSUB 128 so all 16 waves decode (KSUB 64: half do, the rest wait at the barrier): kstore IQ4_XS -6.5%, IQ3_S -2.5%, IQ3_XXS -0.7%, but kres at K = 17408 +3.9..4.5% (not activation traffic: the same with cache-resident tokens; cause open). 1024-token tiles +9%, A-fragment streaming no gain.
- Swiglu gate loads one column ahead in the LDS epilogue (`SW_GATE_AHEAD = 1`): IQ3_S swiglu -0.93%, IQ3_XXS -0.51% cycles in pp2048, bit-identical. Lookahead 2 and 4 are slower: the swiglu epilogue cost (IQ3: ~7% of the kernel vs kstore, IQ4_XS ~1.7%) is mostly a bandwidth burst (142 MB f32 gate + 71 MB f16 out per call, concentrated at workgroup ends), not load latency.
- Decode-free GEMMs, built and measured, off by default (`YAH_DECODE_FREE=1`): a dequant kernel (`kind="dequant"`, the tile's own decode) writes f16 weights once per chunk and f16 tile GEMMs (`fmt="f16"`, token tiles fastest so L2 shares the weight panel) multiply them; the driver records each dequant beside the previous GEMM. Bit-identical end to end, but +12% pp2048 as built. Standalone (clock-free): kstore -8.9% (IQ3_S) / -9.4% (IQ4_XS), kqg -6%; down projection (K = 17408) +-0 (f16 operands this long stream from DRAM either launch order); swiglu +5..6% (its gate-stream burst lands in lockstep under token-fastest order); the dequant pass ~25% of a GEMM (latency-bound, ~2.5x its bandwidth floor). The old "decode-free falsified" note measured HIP's f16 kernel, not this design. Second round, kstore / kqg only: the dequant became persistent (20 workgroups, one per WGP, 64-row double-buffered tile = 20 KB of LDS, the room two GEMM workgroups leave; 1.38 ms per 178 MB standalone, from 6.3) and is recorded *before* the GEMM it runs beside (a dispatch's workgroups launch only after the previous dispatch's have all launched) and ordered after the GEMM's inputs (so the graph's full-drain barrier falls before both). 409 ms of dequant then overlap GEMMs, 54 ms do not, bit-identical; but pp2048 3062 vs 2993 ms (+2.3%): kstore GEMMs gain only -4% beside a running dequant (they share SIMDs, LDS and memory), the dequant costs ~354 M counter cycles, and fused decode was already ~0.23 VALU per weight. Parked.
- 24-bit address multiplies (HRX patch 0007): when value facts prove both operands fit 24 bits, address math uses `v_mul_u32_u24` (full rate) instead of `v_mul_lo_u32` (quarter rate); 476 of 663 multiplies in the prefill set. Prefill cycles -0.26% (GEMMs -0.28%), bit-identical; GPU references identical.
- Partial chunks with quantized KV no longer depend on the padding rows: `yah_kmean` averaged K over all 2048 rows of the first chunk and the V quantizer counted padding rows into the last tile, so a short prompt's result changed with the GEMM tile that ran and with the previous prompt in a server (found by the calibration, which runs different tiles per layer: 1 of 40 server answers differed). Full chunks are bit-identical; KL(fp16 KV || kv8) on short prompts 2e-8 .. 6e-5, same top token.
- Decode-ahead outside IQ4_XS / Q4_K / Q6_K is refused by the generator: on IQ3_XXS / IQ3_S / IQ2_XS it gave nondeterministic output (1 of 4 runs differed; cause not traced). The production decode-ahead GEMMs agreed in 448 hashed runs.
- Autotuned GEMM tiles (`engine/tune/tune.py`, 1680 candidates, 7426 measurements, 7 minutes): prefill cycles -0.13% (GEMM -0.18%; the tuner predicted -0.21%), bit-identical. The full-chunk defaults were already best for the large GEMMs; the wins are the small 1024-row GEMMs (Q4_K -35%, IQ4_XS -33%, Q6_K -13%), IQ2_XXS (-4.6% / -2.8%), IQ4_XS kqg and Q8_0. A first run with the weights resident in the last-level cache chose decode-ahead off for IQ4_XS (-8.5% in the bench, 0% in the pipeline); the bench now streams weights like the pipeline. Wall-time rounds vary about 3% with the starting temperature (2952 ms from 42 C, 3030 ms from 47 C).
- Narrow GEMM token tiles for short chunks (chunked sets carry 128- and 64-token variants; the driver picks per chunk by padded rows x cost per row): 18 tokens 595 -> 404 ms, 100 tokens ~595 -> 456 ms, 300 tokens 909 -> 801 ms; 1000+ tokens unchanged; logits identical. Through `yah_server` an 18-token prompt prefills in 415 ms (was 607-630).
- `yah_half_norm` with 4 rows (waves) per workgroup instead of 1: 46.1 -> 41.9 ms per pp2048 (-3.5% of its cycles); bit-identical.
- HRX pin moved to a02a5ab94 plus the f16 WMMA operand placement (patch 0005; upstream enables #1160 for bf16 only): GEMM cycles -3.2%, prefill cycles -2.8%, pp2048 3047 -> 3000.5 ms; bit-identical. Decode 61.1 -> 60.8 ms (noise).
- Prefill as one HRX graph per chunk, independent kernels overlapping (stock HRX): pp2048 ~3090 -> 3032 ms, a 500-token prompt 970 -> 926 ms; bit-identical.
- Tile GEMM (128 x 256 per workgroup, 16 wave32 waves, both operands in padded LDS) replacing the shared-decode GEMM: pp2048 device time 5263 -> 4574 ms.
- LDS row padding (+8 f16 per weight row): IQ3_S kstore 19.9 -> 11.1 ms standalone (bank conflicts gone).
- Schedule fence between a phase's LDS stores and the next phase's loads: IQ4_XS kstore 12.1 -> 10.3 ms; GEMM device time 3675 -> 3516 ms.
- Decode one phase ahead into a second weight tile (IQ4_XS, Q4_K, Q6_K): 3702 -> 3659 ms.
- One vector header load per weight block: Q4_K 5.93 -> 5.47 ms; pp2048 3598 -> 3578 ms.
- Epilogue slab padding (EPAD 4): LDS bank conflicts -94%; pp2048 3598.7 -> 3537.1 ms.
- 4 x 2 wave layout (32 x 128 per wave): IQ4_XS rows -3.8%; IQ3_XXS and Q3_K rows -4.9%.
- IQ3_S at 4 x 2 with the word-path decode: pp2048 3431.1 -> 3347.6 ms.
- LDS swiglu epilogue for IQ3_S / IQ3_XXS: 3316.4 -> 3285.8 ms.
- Q4_K narrowing via `v_fma_mix` (avoids the v0..v127 `v_cvt` window), enabling 4 x 2: 3290.8 -> 3262.8 ms.
- Workgroup stagger on large grids plus the K = 6144 decode-ahead exclusion: cycles -1.84% (3153.9 -> 3142.7 ms).
- IQ3 grid indices from 32-bit words with `index.assume` ranges (one `v_bfe_u32`, no sign-extend / mask / clamp): IQ3 GEMM cycles -0.64%, pp2048 cycles -0.43%.
- Fused residual (`kres`) and hidden / hidden2 swap: removed 128 zero-add passes of 126 MB each per pass.
- q / gate unpack fused into the q-projection epilogue (`kqg`): cycles -0.45%, 16 dispatches fewer.
- Attention writes f16 for the o projection: half_cast gone, 16 dispatches fewer (~0.2%).
- prep_kq fused into the SSM conv: -22.8 M cycles (~0.3%), 48 dispatches fewer.
- half_norm fully unrolled: 68.7 -> 42.3 ms over 128 calls (bit-identical).
- FlashAttention-style attention (register softmax, head-pair-fastest + longest-first order, exact conditional rescale): pp8192 attention 1264 -> 1025 M cycles (-18.9%).
- Chunked WY Gated DeltaNet: DeltaNet -10.5% (pp2048) / -12.9% (pp8192) cycles.
- Paged KV: cycle-neutral (-0.25% / -0.48%); removes the per-chunk whole-cache V^T re-transpose at long context (16x redundant at 32K).
- Raised tctl limit (system setting, not code): at 105 the GEMM probe ran 2276 -> 2425 MHz, but PROCHOT fires under sustained load; 99 or 95 used since.
- Sleep-polled final wait: host CPU 104% -> 4% of a core, socket -4..9 W (no GPU speed change).

Decode:

- GEMV decode replacing the prefill tile GEMMs at one token: 687 -> 77.1 ms.
- Word-level weight decode (no `vector<16xi8>` bit ops): 77.1 -> 68.2 ms.
- `pipeline(2)` read-ahead on the GEMV sub-block loop: 68.2 -> 63.1 ms.
- GPU embedding + device token stream (no host round trip): -0.9 ms.
- Independent input projections without an ordering barrier: -1.4 ms (HRX patch 0001; see build-and-run.md, HRX patches).
- Long-context attention read-ahead and (head, page) grid order: 128 MB per call at 32K, 2955 -> 689 us (43 -> 195 GB/s).
- DeltaNet state loads hoisted above the q / k norm: 61.2 -> 60.87 ms.
- Decode conv fused into DeltaNet (ping-pong conv state): 60.87 -> 60.59 ms.
- Quantized KV in decode: 30K 74.10 -> 70.56 ms (kv8a16).

## Tried and lost

One line each: what, the number, why.

Prefill GEMMs:

- Cycle ablations of IQ3_S kstore (instructions replaced by `s_nop`, clock-free, standalone): no WMMA -33.7% (the rest alone is 26.9 cycles per WMMA slot, mostly overlapped), no barriers -5.2%, no fragment `ds_load_b128` -1.5%, no global loads -1.3%, no LDS stores -1.1%, no dead zero-init moves -0.2% (~1 cycle per VALU issue). Never nop `s_waitcnt`: scalar descriptor loads then land late and the kernel hangs the GPU.
- Wave64 tile GEMM: IQ3_S kstore 23.93 -> 31.27 M cycles; wave64 VALU / LDS instructions cost ~1.7x and operand fragments do not shrink.
- KSUB = 32 for more residency: IQ3_S ffn_gate +6.8% cycles in pp2048; twice the phases, twice the barriers, and the kernel was already at its issue bound.
- Token tiles of 160 / 192 / 224 / 320 / 384 (any multiple of 16 now emits, masked when it does not divide the chunk, bit-identical): IQ3_S ffn_gate +13.8 / +14.3 / +16.5 / +38 / +23.5% cycles vs 256. Part is the general activation-staging map (+6.5% on the same 256 tile, +2% at 128), the rest is smaller per-wave tiles (more fragment loads per WMMA), KSUB 32 above ~300 tokens, and padding (224: 9.4% of the tokens).
- Stagger on the short-K residual GEMMs (0 / 4000 / 16000 barriers vs 8000): within 0.3%; the lockstep it fixed no longer shows.
- Q4_K MMA order and fences (rhs-outer fence 0 / 2 / 4, lhs-outer): within 0.3% of the default; without decode-ahead +2.7%. The compiler report's 43 full LDS drains per phase are hidden by the other waves.
- 128-token tiles (8 waves of 32 x 64, KSUB 64, 128 VGPRs, 6 waves per SIMD instead of 4): IQ3_S ffn_gate +10.5% cycles, VALU +64%, bit-identical. The kernel is issue-bound (~80% WMMA), so more resident waves hide nothing, while decode per WMMA doubles and fragment loads go 0.625 -> 0.75 per WMMA. With the 512-token result, 256 is the sweet spot at full chunks; narrow tiles only for short token counts.
- 512-token tiles (16 waves, 4 x 4 of 32 x 128, KSUB 32 so LDS fits): IQ3_S ffn_gate +16.7% cycles in pp2048, bit-identical. Decode per WMMA halves (VALU -13%) but is mostly hidden already; KSUB 32 costs ~7% and each phase waits for the slowest of 16 waves (~9% at equal KSUB). With decode-ahead (63.5 KB LDS): +23.7%, its LDS grid lookups contend with the fragment loads. The LDS-staged design cannot amortize decode over more tokens.
- 2 x 4 wave layout (64 x 64 per wave) for IQ4_XS: 25.88 vs 23.12 M cycles (4 x 2); fewer instructions but latency exposed (93% of issue bound).
- IQ3_S word-path decode at 4 x 4: 26.79 -> 27.23 M cycles; longer dependent chain exposed between barriers.
- IQ3_S f16 table decode: -0.7% cycles only; each removed decode VALU saves ~0.23 cycles (kept off).
- Decode-free f16 GEMM (pre-dequantized weights): HIP's f16-weight Q4_K kernel is 8% slower than its quantized one.
- HIP's 256 x 256 tile shape in Loom: 12.49 vs 10.37 ms; Loom's read-ahead cannot keep loads ahead at 192 VGPRs.
- Fragment-major LDS layout (128 x 256): 12.1 -> 14.3 ms; lanes i and i+8 share banks.
- 256 x 128 tiles: 9.97 -> 10.31 ms; half the tokens doubles decode per MMA.
- 32 waves of 32 x 32 tiles: IQ4_XS 9.47 -> 12.56 ms; more fragment loads per WMMA, few decoding waves.
- Dedicated decode waves (16 MMA + 4 decode): 9.47 -> 9.73 ms; SIMDs are issue-bound, so total VALU per WMMA counts, not who issues it.
- Decode spread over all waves / two lanes per group: 10.27 -> 10.95 ms and 10.57 -> 10.83 ms; loads and scale math repeated.
- Phase-loop or inner-loop unroll(2): IQ4_XS 9.60 -> 10.14 ms; new `vmcnt(0)` drains of the carried prefetch.
- One barrier per phase with double-buffered activations: no change. MMAs queued across the barrier: +18%.
- Double-buffered LDS, one barrier per phase (`Tile.dbuf`, off by default): KSUB 32 only, since two KSUB-64 tiles need ~108 KB of LDS. IQ3_S kstore +9.0%, IQ3_XXS kres +6.2% cycles vs KSUB 64, bit-identical. Against plain KSUB 32 (+16.3%) it recovers ~7%, the barrier coupling the barrier ablation showed (-5.2%), but KSUB 32's per-phase overhead is larger. The 2026-09 prototype's "+160 ms, not bit-identical" was a missing barrier between the LDS grid-table staging and the first decode. Kept as an enabler: it wins once per-phase overhead at KSUB 32 drops by more than ~9%.
- Why KSUB 32 loses with `dbuf` (IQ3_S kstore): per WMMA in the phase loop VALU 4.44 -> 5.36 (address +0.58, moves +0.31) and SALU 0.83 -> 1.41, about +1.5 of the +3.6 cycles. Not load latency: the gap is the same with cache-resident weights (+8.2%) as streaming (+8.5%). The likely rest: at KSUB 32 one 32-weight group per row means only 4 of 8 waves decode, so the MMA-only waves wait at the barrier (not measured per wave). 256 x 128 tiles put all 8 waves on decode but double decode per WMMA: with `dbuf` +12.3% (IQ3_XXS kres) / +46.9% (IQ3_S), without +26% / +60%.
- Grouped launch order (row-group swizzle): IQ4_XS rows +4.0% in the pipeline; plain order already reuses the activation tile.
- Token-major grid: 10.3 -> 11.2 ms; consecutive workgroups stop sharing the activation tile.
- IQ3 grid tables in global memory instead of LDS: 10.10 -> 11.74 ms; a dependent lookup chain behind global latency.
- Activation row pitch padding: ~4% on ffn_down only, needs every producer to write the padded pitch; not pursued.
- Q4_K carried-prefetch rework (loads after decode, unrolled phases): 15.38 -> 15.91 / 16.82 M cycles; the allocator copies the carried registers behind full drains.
- Fused FFN gate + up: sized at ~0.5% of prefill at best (prologues and epilogues already overlap other waves), and 28 of 64 layers would need two decoders; not built.
- Widening the token tile of unchained GEMMs: wrong forward (argmax 220 vs 11751); cause not found.
- int4 (iu4) GEMMs: need 4-bit activations (4-11% error without rotation) and cannot hold the IQ4_XS codebook. int8: per-32 scales cost ~4 VALU per WMMA, break-even at best.
- Load cache hints: Loom's gfx11 encoding drops them; the ISA is byte-identical.
- Source-level LICM in the AMDGPU pipeline (stock `licm` before `scf-to-cfg`): hoists almost nothing in the GEMMs (the address math is created by the backend at each fragment load), and the DeltaNet kernel spills (248 -> 255 VGPRs, 16 B scratch). Not adopted.
- Activation (B) fragments straight from global memory instead of the LDS activation tile: GEMM cycles +48% (every format +30..87%), bit-identical. A fragment is 16 scattered 32-byte reads (token rows 10 KB apart) instead of one coalesced row load shared through LDS, 64-bit address math adds 8-20% VALU, and the load latency sits inside the barrier-synchronized K loop. The L1-resident `global_load` microbenchmark (2.5 cycles) does not transfer.
- Decoding weights straight into the WMMA operand registers (no LDS weight tile): not built; each weight would be decoded by both token-waves and every 16-row fragment per wave, ~4x the decode instructions per WMMA, which outweighs the ~1.3 cycles/WMMA of LDS traffic it saves.
- Hoisting the LDS fragment row offset out of the K loop (per-wave views): no change. The 4 `v_mul_lo_u32` per phase are the fragment loads' per-lane addresses (lane row x 144-byte padded row), generated inside the compiler: no LICM, and quarter-rate `v_mul_lo_u32` instead of a 24-bit multiply. Compiler-side, like the zero-init `v_mov` before each `v_fma_mix` pair (~0.5 per WMMA) and the carried-register copies (Q3_K / Q5_K / Q4_K latches, IQ4_XS decode-ahead operands); together <= 3-4% of GEMM cycles.

Prefill attention, DeltaNet and pipeline:

- DeltaNet gate projection recorded beside the DeltaNet in the prefill graph: neutral (3036.8 vs 3032.0 ms). A graph barrier drains all earlier work, and the GEMM's workgroups only start once the DeltaNet's 96 are all placed (1.2 ms in).
- HIP-order attention, key-loop unroll(2): 80.59 -> 79.03 M cycles with 31 spill stores; not adopted.
- FA attention, 32-key tiles: 64.6 -> 72.5 M cycles; LDS 53.8 KB allows only 2 workgroups per WGP.
- FA attention, GQA packing (6 heads x 16 tokens, 12 waves): 64.6 -> 66.8 M cycles; 4x less DRAM, but less phase diversity with 2 x 12 waves.
- FA attention, threshold rescale (FA4, THR 8): 69.2 vs 67.1 M cycles exact; no gain and changes numerics.
- FA attention, Q staged in halves: 64.6 -> 66.1 M cycles; 200 VGPRs still miss the 4th workgroup.
- Integer-compute attention (kv4a4 iu4, kv4a8, int8 iu8): kv4a4 -14.5% attention (~0.4% of pp8192) for 96% 99%-precision; dropped for the a16 formats.
- uint8 P in attention: 16x the attention error (small weights round to 0); P stays f16.
- Page table in LDS: attention +1.1% (the cost is index math, not the scalar load), and it hung the GPU when read before its barrier.
- DeltaNet dual-issue FMAs (`vector.dotf`): +1.6-2.3% cycles; VOPD pairs across chains stretch the per-token latency path.
- FLA-style 3-kernel DeltaNet split: per-chunk state snapshots ~150 ms over 48 layers at pp8192 vs ~350 ms for all of DeltaNet.
- Chunked DeltaNet packed f16 pair stores: 3.702 -> 3.744 M cycles; the stores queue behind LDS traffic.
- Weights as a device-local copy or a 2 MB THP copy: 3505 / 3522 vs 3527 ms (noise), load +5 s / +43 s; GEMMs reuse each weight tile 2048 times.
- Load-time tricks on the GGUF mapping: prefault on a thread (device init 180 -> 437 ms, it holds `mmap_lock`), `MADV_HUGEPAGE` (cold load +0.6 s), `MADV_POPULATE_READ` (-30 ms only).
- Freeing host power (no busy-poll) for GPU clock: no clock gain; the GPU is thermal-limited, not power-limited.
- HIP graph capture / persistent kernels for prefill: ~0.02%; a kernel boundary is free when the kernel has real work.

Decode:

- Band-fused input projections: 61.17 vs 61.20 ms, neutral (176 fewer dispatches; the separate no-barrier dispatches were already ~92% efficient). On by default.
- RMSNorm folded into the residual GEMV's last workgroup: 61.30 vs 61.18 ms; the ~5 us serial norm tail cancels the ~3.4 us launch it removes.
- Read-ahead depth 3 / 4 on the GEMV loop: 66.1 / 67.8 vs 63.1 ms (depth 2); the queue costs registers.
- 32-element groups per lane: 61.77 vs 60.48 ms; 11-31% fewer VALU but 60-75 -> 103-114 VGPRs.
- 16 lanes per row: 63.49 vs 60.19 ms; shorter rows lose bandwidth.
- Persistent GEMVs (workgroups loop over row groups): 61.67 vs 60.48 ms; ragged last round (SwiGLU +3.7%).
- Persistent megakernel: grid barrier 0.26-0.49 us would save at most ~1.3 ms per token, but persistent GEMVs lose and the register count is the max over all phases; not built.
- HRX graph replay: 3.32 vs 3.57 us per dependent dispatch; only fewer dependent dispatches help.
- Band and plain GEMV geometry (rows / waves per workgroup 1 x 4, 2 x 8, 4 x 4, 4 x 2; `YAH_TILES` "rw"): 60.6-62.6 vs 60.27 ms for the default 2 x 4; every step's logits identical.
- SwiGLU geometry (rows / waves per workgroup 1 x 4 .. 2 x 8): all within 0.2 ms; residual GEMV 2 x 4 beats 1 x 4, 2 x 8, 1 x 8, 4 x 4 by 1-1.5 ms.
- LDS sign-mask table instead of `v_mul_lo_u32`: neutral; after word decode the GEMVs are not VALU-bound.
- `index.assume` instead of load clamps: -4% VALU, neutral.
- Explicit fma chain for the 16-element dot: same VALU count; the compiler already contracts.
- No x loads at all (ablation): -0.45 ms; x traffic does not limit the low-bit GEMVs.
- kv4a16 decode attention vs kv8a16: no faster (347 vs 362 us at 30.7K) despite 40% fewer bytes; per-workgroup q rotation, barriers and group sums make it latency-bound.

KV codecs (fake-quantized through a KV hook since removed, 32K):

- UltraQuant MXFP4 K and V: mean KL 1.0e-2, dPPL +0.47%; per-token MXFP4 V is 4x worse than per-channel tile V.
- WUSH calibrated transform for int4 K: only ~7% better than a fixed H256 Hadamard; the rotation does the work.
- KVarN Sinkhorn balancing: as good as a rotation alone, redundant with one.
- Quantile clipping for int4 K: 4x worse than range shrink.

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

Prefill kernels, same real bytes, cycles per call (rocprofv3; Loom kernels launched through `loomhip`), latest Loom numbers:

| kernel | Loom M cycles | HIP M cycles | Loom / HIP |
|---|---:|---:|---:|
| IQ4_XS 17408x5120 | 23.12 | 23.15 | 1.00 |
| Q4_K 10240x5120 | 13.54 | 13.71 | 0.99 |
| IQ3_S 17408x5120 | 24.15 | 25.91 | 0.93 |
| IQ3_XXS 17408x5120 | 24.23 | 25.80 | 0.94 |
| Gated DeltaNet, B=2048 (Loom: chunked WY) | 3.57 | 4.76 | 0.75 |
| Attention, pp8192 real layer-3 inputs (Loom: FA) | 64.6 | 79.4 | 0.81 |
