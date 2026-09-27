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
