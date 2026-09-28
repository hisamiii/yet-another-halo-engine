# Kernel tuning: measured rooflines

Everything here was measured on the same box during the engine work. The full log, with the harnesses and the dead ends, is `docs/models/qwen3.8-27b/EXPERIMENTS.md` in [gufo](https://github.com/gufo-org/gufo); the section names are given so each number can be traced.

## GPU: the ceiling is a measured instruction rate, not a guess

Wall-clock harness, eight live accumulator chains (a first attempt read two of eight, let the compiler delete six MMAs, and reported 4x high):

| instruction | TMAC/s | TFLOPS |
| --- | ---: | ---: |
| WMMA fp16 | 24.17 | 48.35 |
| WMMA bf16 | 23.73 | 47.47 |
| WMMA iu8 | 25.15 | 50.31 |

wave64 computes the same 4096 MACs and issues over two cycles, so it measures identically (48.50 vs 48.35). It buys half the accumulator registers and no throughput.

Production efficiency:

| kernel | TFLOPS | share of its ceiling |
| --- | ---: | ---: |
| fp16 `HalfPrefillGemmKernel` 256x256 bk4, production shape | ~30-32 | 63-66% of 48.35 |
| the same, paired `kGateUp`, isolated at m=17408 k=5120 batch=2048 | **40.6** | **84%** |
| int8 `WKQuantA8BlockedWmmaGEMMKernel`, same shape | ~24.2 | 48% of 50.31 |

**The tile space is swept and exhausted.** Only the 256x256 family is competitive; 128x256, 256x192, 256x128 and 128x128 are 15-40% slower. BK is fixed at 4 and the two stages already fill the 64 KiB LDS budget, so there is one block per CU and 1024 threads is the hardware workgroup limit. `Complete` (drop the tail predicates) is worth 4% on Q4_K, neutral on Q6_K/IQ3_XXS and costs 15% on IQ4_XS, which is why it is gated.

Ablations on the real kernel: weight and activation staging costs ~10% on Q4_K and ~22% on IQ3_XXS; the store epilogue costs **+0.13%**, i.e. nothing. With staging removed the kernel runs at 34.9 TFLOPS, 72% of ceiling, and the remaining ~28% is localised to the K loop (six swizzled LDS reads, eight WMMA per K step, loop overhead across forty stages, barriers) but is **not explained**. The obvious next structure, a reduced inner-loop microbenchmark, measured *slower* than the real kernel and is not a valid reference.

### The aggregate is ~30 TFLOPS, not 40.6

`40.6` is one kernel on one shape. Across the whole prefill kernel set the GPU delivers **~30 TFLOPS aggregate**, about **62% of the 48.35 instruction ceiling**, against the best kernel's 84%. The recorded model-level gap versus the plain-store bench is **27.7%**: part of it is the IQ3_XXS decode (a third of the FFN on the 3.84 bpw target and the most expensive decoder measured, +26% loop instructions over Q4_K), part is the non-FFN GEMMs, and part is shape and `Complete` choices. The batch-2048 Q4_K against IQ3_XXS A/B that would separate the format from the rest was never run.

### The per-format table, and why the aggregate is ~30

The section above attributes the aggregate gap partly to the IQ3_XXS decoder,
partly to the non-FFN GEMMs and partly to shape and `Complete` choices without
splitting the terms. `engine/gpu/gemm_bench.hip` splits the first one: it runs
the kernel the prefill path runs, one weight format at a time, at the two real
FFN shapes, batch 2048, best of three.

| type | gate/up M=17408 K=5120 | down M=5120 K=17408 |
| --- | ---: | ---: |
| **Q4_K** | **38.05** | **37.60** |
| IQ4_XS | 31.30 | 33.18 |
| Q5_K | 32.60 | 32.10 |
| Q8_0 | 30.89 | 31.83 |
| Q2_K | 29.78 | 31.79 |
| Q3_K | 30.83 | 29.96 |
| IQ4_NL | 31.24 | 30.33 |
| IQ3_S | 30.09 | 30.67 |
| IQ3_XXS | 28.99 | 29.79 |
| IQ2_XS | 28.94 | 29.41 |
| IQ2_XXS | 28.85 | 28.85 |
| Q6_K | 30.24 | 30.50 |
| IQ2_S | 28.86 | 28.43 |

Two things stand out. **Q4_K is in a class of its own** — every other format
loses 15-25% to it at the same tile. And the IQ3 decoders are *not* an outlier:
IQ3_XXS at 28.99 is 1.2% below IQ3_S and 7% below IQ4_XS, so "IQ3_XXS is the
most expensive decoder" costs tens of percent of loop instructions but only a
few percent of wall time. What Q4_K has that the rest do not is the cheapest
decoder plus, uniquely in this list, the tail-free `Complete` variant (worth
4% on Q4_K by the ablation above, and gated to Q4_K/Q5_K at line 1412).

At `m >= 4096` every type takes the same 256x256 tile, so the table is
tile-matched and the spread is decoder cost. Weighting the measured rates by the
tensor types of the actual artifact, taking the harmonic mean per layer because
gate, up and down have identical FLOPs:

| | TFLOPS |
| --- | ---: |
| census-weighted FFN prediction, 64/64 layers | **30.43** |
| measured model aggregate (section above) | ~30 |
| if those layers were all Q4_K | 37.90 |

The prediction lands within a few percent of the aggregate with no free
parameter, so the model-level gap against the 48.35 ceiling is **the weight
format mix**, not an unexplained K-loop residual. The artifact is a mixed
IQ3/IQ4 shard despite the `IQ4_XS` in its filename: IQ3_XXS and IQ3_S are 207 of
its 866 tensors and dominate the FFN, while the Q4_K present is in the
projections. `gemm_bench` measures components with synthetic weights, so the
agreement is not a release measurement — but it removes the need for a free
parameter to explain the aggregate.

**Measured at the model level.** An existing pure-Q4_K artifact of the same
model settles whether the component table composes
(`/home/q/models/gufo-sweep/base_q4kpure.gguf`, 506 Q4_K tensors, 866 tensors,
every shape identical). `--ids-file` 2048 tokens, `--chunk 2048`, `--repeat 1`,
warm-up discarded, arm order alternated, a fixed 15 s gap and Tctl before every
timed run:

| rep | order | mixed IQ3/IQ4 (ms) | pure Q4_K (ms) | Q4_K |
| ---: | --- | ---: | ---: | ---: |
| 1 | mix first | 3848.1 (55 °C) | 3554.1 (55 °C) | +8.3% |
| 2 | Q4_K first | 3730.1 (53 °C) | 3490.2 (54 °C) | +6.9% |
| 3 | mix first | 3662.6 (53 °C) | 3389.0 (53 °C) | +8.1% |
| 4 | Q4_K first | 3729.8 (54 °C) | 3517.8 (53 °C) | +6.0% |

Q4_K wins every rep in both orders: **+7.3% on the means**. The format is a real
model-level lever, and the component table over-predicts it by about 2x.

The two files differ in two ways and the second is ruled out: the pure file
takes the dual gate/up kernel on **64/64** blocks against the mixed artifact's
**9/64**, and raising the mixed one to 29/64 measures +1.5% with the sign
flipping (section below). So the 7.3% is **decoder cost**.

**That fixes the target, and it is not the artifact.** Requantizing the shard is
out of scope by decision, so the only way to recover the 7.3% is the decoder.

**The obvious explanation is already refuted by the table above.** Q4_K differs
from the IQ types in that it takes the deferred-decode path (`DecodeStage == 2`,
raw weight registers held across the pipeline: `prefill_fp16.hip` line 264),
while `IQ3_XXS`/`IQ3_S` take `DecodeStage == 0` and decode inline in the fetch
stage. But `Q5_K`, `Q6_K` and `Q8_0` *also* take `DecodeStage == 2` and measure
32.60, 30.24 and 30.89 against Q4_K's 38.05 — decode scheduling does not carry
the advantage. The one-line version of the experiment is not available anyway:
`RawWeights<Type>` is `static_assert`-gated to those four types, and the IQ raw
helper covers only `IQ4_XS`/`IQ4_NL`, so a deferred IQ3 decode would be new
code, not a switch.

What is left is the per-weight decode arithmetic: Q4_K's four-bit unpack with
precomputed scale words is measurably cheaper than the IQ3 grid lookup on this
part. A raw form for `IQ3_XXS`/`IQ3_S` — grouped grid index, sign byte and
block scale held in registers and expanded by table lookup the way
`DecodeIqRaw` does for `IQ4_NL` — is the only route to it. That is a real
kernel change with a numerics re-validation (deferring a decode changes the
dot-product rounding) against a model-level ceiling of 7.3%, only part of which
is IQ3 (the mixed file has non-FFN IQ3 tensors too). Not obviously the next
place to spend the machine now that decode at depth is 3x faster and the format
question is answered.

**This is clock, and the clock is a function of the kernel.** Efficiency per clock agrees to **0.3%** across the two harnesses that disagreed -- 0.0144 TF/MHz in both -- so there is no hidden code difference. The part has three SCLK levels (600 / 1408 / **2900 MHz**) and **never reaches the top one**: measured across power limits, 80 W gives 1799 MHz / 25.9 TF and 130 W gives 2016 MHz / 28.6 TF, work-per-clock is flat (0.0144, -1.5% between them), `gpu_busy` is ~85% at both, and the sampled maximum is 2221 MHz. The log's own line is that "what is broken is the conversion of watts into clock".

So the 26% is **not a fixed ceiling**: the clock is coupled to the kernel's power draw, and zeroed operands raise it ~20%. A kernel that toggles fewer bits and issues fewer instructions for the same FLOPs raises both the work per clock and the clock itself. At the 2900 MHz level the part never reaches, the peak is on the order of **60 TFLOPS**; at a realistic ~2200 MHz it is ~45, and the production ~30 is about two thirds of that. Treating ~31.5 as "the real rate, no headroom" was the wrong read. (The NPU's 32.4 TF, by contrast, is data-independent, so the two engines are at parity *at the current clock*.)

int4 is the only measured lever above 1.5x: 52.6 TMAC/s, 2.1x int8 and 2.2x fp16. It requires four-bit activations, so it is a quality decision, and it is eligibility-bound (Q4_K/Q3_K, a low-double-digit percent of these shards).

**So "autotune the GPU" is not an open tile search.** The tile space is closed and the structural limits (LDS, workgroup size) are hit. What remains is the unexplained K-loop residual and format-level choices, not parameter sweeps.

## NPU: the stock 15% capture is a tiling artifact

MAC-array roofline, register-resident operands, no DMA:

| datapath | 32 tiles | note |
| --- | ---: | --- |
| int8 `mac_8x8_8x8` | **56.9 TOPS** | 1.07 MAC/cycle at ~1.62 GHz, the architectural issue rate, matching AMD's 50 TOPS figure |
| bfp16 via `bf16` conversion | **38.9 TOPS** | conversion inside the loop, so a lower bound |
| native bf16 (`2x mac_4x8_8x8_bf16`) | 2.8 TOPS | 14x slower; avoid this path |

These are three different quantities, and any planning number has to say which one it is:

| level | quantity | value |
| --- | --- | --- |
| silicon | MAC-array issue, register-resident, no DMA | 56.9 TOPS int8, 38.9 bfp16 |
| kernel | a real GEMM at a tiling (ATB config3, bfp16) | 32.4 TFLOPS = **83% of the bfp16 issue rate** |
| engine | that GEMM driven by the inference loop | ~25 TFLOPS gate/up in-engine |

ATB config3 is **still a GEMM** -- it moves operands through L1 and tiles the reduction -- so 32.4 is a kernel result, not the array's ceiling, and it is exposed to the same structural loss the stock kernels are, only less of it. The log warns about exactly this conflation: an early pass "measured a kernel and called it a ceiling" on the stock design, and the same error was already corrected once on the GPU side when the fp16 WMMA ceiling had to be measured separately from the kernel.

Two implications. The bfp16 GEMM still leaves **17%** to its own instruction ceiling, and more importantly the **int8 version of this tiling has never been built**. bfp16 is used because it consumes fp16-class activations directly, with no activation quantisation -- a quality and engineering convenience, not the silicon limit. An int8 tiling would target the 56.9 TOPS issue rate (roughly 47 if the same 83% transfer, which is an estimate and not a measurement), at the cost of activation quantisation and per-element metadata arithmetic, so its practical tiling ceiling is unknown. On paper the datatype swap alone is ~1.45x on the NPU side; the earlier note that the int8 path "has more headroom than fp16" is the same observation from the kernel side.

The stock whole-array kernels capture ~15% of their datatype's peak, and the capture is **datatype-independent** (i8 8.74/56.9, bf16 5.81/38.9, W4A16 5.93/38.9). A datatype-independent plateau is a dataflow signature, not silicon. It is also **not the operand feed**: one vector L1 load per MAC measures 95% of peak. The suspect is the C-accumulator round-trip to L1 plus the ObjectFIFO lock and DMA handshake per k-chunk.

Asymmetric tile buffering (ATB) beats it: **30.6 TFLOPS verified**, 79% of the bfp16 peak, by decoupling the A and C L1 buffering. It was shape-specific until a hardcoded `constexpr int K_Problemsize = 4096` in the design was made overridable (five lines); after that the NPU runs our exact FFN shapes correctly at **32.4 TFLOPS (gate/up) and 32.9 (down)** -- 83% of the bfp16 MAC peak and 80% of the iGPU on the same shape. That kernel family is what the engine uses.

## Both engines at once: the coupling is power, not bandwidth

| phase | GPU | NPU | SoC power mean / max |
| --- | --- | --- | --- |
| GPU solo | 497.80 tok/s | -- | 73.2 / 95.9 W |
| NPU solo | -- | 31.1 TFLOPS | 30.3 / 61.9 W |
| concurrent | 491.13 (-1.3%) | 24.6 (-20.8%) | 61.9 / 93.0 W |

The GPU is unperturbed and the NPU gives up 21%. The concurrent power ceiling is no higher than the GPU-solo ceiling, so the SoC is power-capped and the budget is being split, not the bandwidth. The NPU is by far the more efficient engine: 31 TFLOPS for ~30 W against the iGPU's ~40 TFLOPS for ~73-96 W.

## Where the prefill ceiling actually is

From the profile: GEMM is **91.9%** of prefill, non-GEMM **8.1%** (SSM 4.9, attention 1.7, elementwise ~3), and GPU idle is **0.7%**. The FFN is **67% of GEMM FLOPs**.

With the measured rates (GPU 40.6, concurrent NPU 24.6, balance at 0.377) the FFN split is **1.61x on the FFN GEMM**, which translates to

```
1 / (1 - 0.919 x 0.67 x (1 - 1/1.61)) = ~1.30x on prefill
```

and not the lazier `1.65x` that an earlier pass in the log derived by treating all GEMM as splittable. The log carries both; 1.30x is the corrected one.

This matters for planning. It means:

- an FFN-only split is worth about **1.3x**, and the token split already measured at ~1.30x over the derived GPU-only baseline is therefore close to this ceiling -- the remaining headroom is the overlap, not more FFN.
- splitting the rest of the GEMM (attention projections) raises the multiplier toward the GEMM-level 1.61x, i.e. roughly **1.45-1.55x on prefill**.
- the **2.0x** figure is the bound for splitting *all* work with equal engines, and it is not reachable while 33% of the GEMM stays on one engine and the NPU is derated 21% under concurrency.

## The weight format is a real GPU lever

The shipped shard's FFN is not one quant. A byte-weighted census gives seven types across 195 FFN tensors: IQ3_S 36.9%, IQ3_XXS 33.3%, IQ4_XS 21.0%, plus Q3_K, Q4_K, IQ2_XXS, IQ2_XS (3.48 bpw effective). IQ3_XXS is the most expensive decode in the ISA study (+26% loop instructions over Q4_K) and is a third of the FFN, which lines up with a 27.7% model-level gap against the plain-store bench. Compounding it, the paired gate/up kernel has **no IQ3_XXS case**, so those layers drop to two separate `kStore` launches with tail predicates.

**Falsifier on record:** a batch-2048 A/B of Q4_K against IQ3_XXS at m=17408 k=5120. Near +25% means the gap is the decoder and repacking/coverage is the lever; near +3% means the format hypothesis is dead too. A batch-512 run gave only +2.8%, but at that batch the shape is weight-bandwidth-bound and hides decode cost.

**Falsified: the missing paired kernel for IQ3_XXS is not the gap.** The census
above says this shard's FFN is not one quant, and the paired gate/up kernel had
no `IQ3_XXS`, `Q3_K` or `IQ2_*` case, so those layers ran as two launches with
the gate round-tripping through a 143 MB fp32 buffer — a reading that made
coverage look like the lever. A tensor-type census of this artifact (§ above)
puts the gate/up pairs at IQ3_XXS/IQ3_XXS 16, IQ3_S/IQ4_XS 14, IQ3_S/IQ3_S 12,
IQ4_XS/IQ4_XS 9, Q3_K/IQ3_S 7, and seven singletons: **only 9 of 64 blocks hit
the paired kernel**. Adding same-type `IQ3_XXS`/`Q3_K`/`IQ2_XXS`/`IQ2_XS` and
four mixed pairs took coverage to 29/64. The rest are `IQ3_S`-gated, which the
paired epilogue excludes by `static_assert` because `IQ3_S` feeds the A
operand instead of B.

A first A/B of this used `--repeat 2` with no fixed gap, which is not the
controlled protocol in [methodology.md](methodology.md): the arm that ran first
won every rep and the sign tracked the order, so it could only say "no effect
detected". Re-run under the protocol — every arm warmed and discarded, `-r 1`,
arm order alternated, a fixed 15 s gap and Tctl before every timed run — on the
same artifact, so the pairing is the only variable:

| rep | order | paired 29 (ms) | paired 9 (ms) | new |
| ---: | --- | ---: | ---: | ---: |
| 1 | new first | 3460.6 (50 °C) | 3481.3 (51 °C) | +0.6% |
| 2 | old first | 3657.5 (53 °C) | 3479.9 (51 °C) | -4.9% |
| 3 | new first | 3661.3 (53 °C) | 3799.8 (53 °C) | +3.6% |
| 4 | old first | 3497.0 (53 °C) | 3752.8 (53 °C) | +6.8% |

The new-vs-old mean is **+1.5% with the sign flipping in both order arms**:
nothing. The ~286 MB per layer of round-trip traffic is real but is ~2.7% of a
chunk against a run-order effect of the same size. The change was reverted
again. `YAH_PAIRED_STATS=1` was kept: it prints one character per block for
the FFN gate/up projection, so coverage is confirmed rather than assumed — the
same discipline that caught a split-K A/B that had silently run a stale binary.

## The two matrix units are shaped on different axes

The same datatype switch costs differently on the GPU and the NPU because the
two units are sized differently:

- **GPU WMMA is element-count-shaped.** `wmma_f32_16x16x16_f16` and
  `wmma_i32_16x16x16_iu8` both do 4096 MACs per instruction and issue at about
  the same rate, so fp16 and int8 measure 48.35 and 50.31 TF — a 4% spread.
  int8's narrower operands buy nothing. Only int4 changes the shape
  (`..._16x16x32_iu4`, 8192 MACs), which is where 52.6 TMAC/s, 2.1x int8,
  comes from.
- **NPU AIE2P is bit-width-shaped.** `I512.I512.ACC1024.acc32.mac` moves 512-bit
  operands (64 int8) and `I1024.I1024.ACC2048.bf.mac` moves 1024-bit operands
  (64 bf16). Same MAC count per instruction, twice the operand bits, so bf16
  does half the work per operand fetch: 56.9 TOPS int8 against 38.9 bf16,
  1.46x. There is **no int4 MAC**; the AIE2P has an int4 **unpack**
  (`llvm.aie2p.unpack.I512.I8.I4`, 64 packed nibbles to 64 int8) whose lowering
  names its purpose — AWQ packed weights with the signed offset folded into a
  per-group zero point. So NPU int4 is w4a8 with packed weights, a byte-width
  format, not a rate.

So on the GPU there is nothing between fp16 and int4, while on the NPU int8 is
a free 1.46x over bfp16 before any tiling work. The shipped split runs bfp16 B,
i.e. it sits on the 38.9 TOPS rung when 56.9 is available with fewer bytes per
weight; int4-B then attacks the operand-traffic gap, not the MAC rate.

**Falsifier:** build the same ATB shape with an int8 B and compare against the
bfp16 build at the same M/K/N. Near 1.4x on the achieved rate confirms the
datatype rung; near 1.0x means the kernel was already operand-bound and the
byte width, not the MAC width, is the lever.
## Method notes worth keeping

- Measure the instruction ceiling separately from any kernel; a peak harness that read two of eight accumulators reported 4x the real rate.
- An ablation that skips a fetch without putting valid data in its place is not free: it reported above the measured ceiling and was briefly misread as "staging is the entire deficit".
- A reduced microbenchmark can be slower than the real kernel. It is not a reference.
- All-ones operands cannot detect a layout bug: any permutation of A or B is invisible, and a plumbing failure reads as near-zero output rather than misplaced data.
- GPU-busy versus span is the idle measure, but it is a different quantity from per-cycle stall; a kernel can be resident and stalled at the same time.
