# Results

Qwen3.8-27B IQ4_XS on Strix Halo (gfx1151), Loom kernels through HRX. One round per number, APU cooled to <= 55 C first, 15 s (pp2048) or 30 s (longer) gap, no clock pinning; the p71 prefill rows at tctl 95 (rules in [build-and-run.md](build-and-run.md)). HIP baselines are in the section below this one.

## Current results

Prefill, default set (p71: FA attention, chunked DeltaNet, paged KV), `layers_ms`:

| prompt | set | Loom | HIP (hip-final) | measured |
|---|---|---:|---:|---|
| 2048 | one pass, B = 2048 | 3132.2 ms (654 tok/s) | 3381.7 ms | 2026-10-02 |
| 8192 | one pass, B = 8192 | 13154.0 ms (623 tok/s) | 14254.3 ms | 2026-10-02 |
| 8192 | chunked (2048), 32K pools, fp16 KV | 13085.3 ms | 14431.7 ms | 2026-10-02, run before decode |
| 30720 | chunked (2048), 32K pools, fp16 KV | 59004.2 ms (521 tok/s) | 62596.1 ms | 2026-10-02, run before decode |
| 8192 / 30720 | same, kv8a16 | 13270.9 / 60182.6 ms | - | 2026-10-02 |
| 8192 / 30720 | same, kv4a16 | 14111.8 / 60315.3 ms | - | 2026-10-02 |
| 65536 | chunked (2048), fp16 / kv8a16 / kv4a16 | 232.8 / 242.1 / 234.7 s | - | 2026-10-02 |

The quantized-KV prefill rows are within run-to-run noise of fp16: their attention kernels cost +12-15% (int-to-f16 decode at staging), and attention is ~3.5% of pp8192.

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
- Independent input projections without an ordering barrier: -1.4 ms (needs the local HRX dispatch flag; see build-and-run.md, Local HRX patches).
- Long-context attention read-ahead and (head, page) grid order: 128 MB per call at 32K, 2955 -> 689 us (43 -> 195 GB/s).
- DeltaNet state loads hoisted above the q / k norm: 61.2 -> 60.87 ms.
- Decode conv fused into DeltaNet (ping-pong conv state): 60.87 -> 60.59 ms.
- Quantized KV in decode: 30K 74.10 -> 70.56 ms (kv8a16).

## Tried and lost

One line each: what, the number, why.

Prefill GEMMs:

- Wave64 tile GEMM: IQ3_S kstore 23.93 -> 31.27 M cycles; wave64 VALU / LDS instructions cost ~1.7x and operand fragments do not shrink.
- KSUB = 32 for more residency: IQ3_S kstore 23.84 -> 27.34 M cycles; twice the phases, twice the barriers, and the kernel was already at its issue bound.
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
- Double-buffered LDS at KSUB = 32: -3% standalone but +160 ms in the pipeline.
- Grouped launch order (row-group swizzle): IQ4_XS rows +4.0% in the pipeline; plain order already reuses the activation tile.
- Token-major grid: 10.3 -> 11.2 ms; consecutive workgroups stop sharing the activation tile.
- IQ3 grid tables in global memory instead of LDS: 10.10 -> 11.74 ms; a dependent lookup chain behind global latency.
- Activation row pitch padding: ~4% on ffn_down only, needs every producer to write the padded pitch; not pursued.
- Q4_K carried-prefetch rework (loads after decode, unrolled phases): 15.38 -> 15.91 / 16.82 M cycles; the allocator copies the carried registers behind full drains.
- Fused FFN gate + up: sized at ~0.5% of prefill at best (prologues and epilogues already overlap other waves), and 28 of 64 layers would need two decoders; not built.
- Widening the token tile of unchained GEMMs: wrong forward (argmax 220 vs 11751); cause not found.
- int4 (iu4) GEMMs: need 4-bit activations (4-11% error without rotation) and cannot hold the IQ4_XS codebook. int8: per-32 scales cost ~4 VALU per WMMA, break-even at best.
- Load cache hints: Loom's gfx11 encoding drops them; the ISA is byte-identical.

Prefill attention, DeltaNet and pipeline:

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
- Concurrent DeltaNet + z projection (no ordering barrier): -0.6% of pp2048, but needs a local HRX patch; stock HRX cannot.
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
