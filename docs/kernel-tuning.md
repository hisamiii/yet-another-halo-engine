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

`40.6` is one kernel on one shape. Across the whole prefill kernel set the GPU delivers **~30 TFLOPS aggregate**, about **62% of the 48.35 instruction ceiling**, against the best kernel's 84%. The recorded model-level gap versus the plain-store bench is **27.7%**. It used to be attributed partly to the IQ3_XXS decode -- "a third of the FFN on the 3.84 bpw target and the most expensive decoder measured, +26% loop instructions over Q4_K" -- and that attribution is now **withdrawn**: measured wall-clock, IQ3_XXS is at parity with Q4_K (paired table below). The section that replaced it with a per-format explanation is withdrawn too. The format term is small; the gap against the instruction ceiling is not, and it is still unexplained.

### Per-format deltas, paired against Q4_K

The section above attributes the aggregate gap partly to the IQ3_XXS decoder,
partly to the non-FFN GEMMs and partly to shape and Complete choices without
splitting the terms. engine/gpu/gemm_bench.hip runs the kernel the prefill path
runs, one weight format at a time, at the real FFN shape, batch 2048.

**The measurement has to be paired, not a sweep.** The first attempt ran every
type inside one process (gemm_bench all) and reported Q4_K at 38.05 against
28.9-32.6 for everything else. That was not the format: the session decays more
than 25% across a three-minute sweep, so the type that runs *first* takes the
boosted runs and every later type is measured on a hotter part. Re-measuring
each type in its own process, immediately before or after Q4_K with the order
reversed on the second pair so the decay cancels inside each pair:

| type | vs Q4_K | type | vs Q4_K |
| --- | ---: | --- | ---: |
| Q5_K | **+4.0%** | Q8_0 | -6.4% |
| IQ3_XXS | **-0.8%** | Q3_K | -6.9% |
| Q2_K | -1.2% | IQ2_S | -8.4% |
| IQ4_NL | -2.5% | IQ2_XS | -11.2% |
| IQ4_XS | -3.3% | IQ2_XXS | -11.6% |
| IQ3_S | -5.7% | | |

The corrected picture is the opposite of the sweep's. **Q4_K is not special** --
Q5_K measures 4% faster -- and **IQ3_XXS, which dominates this shard's FFN, is at
parity (-0.8%)**. Only the 2-bit formats carry a real penalty, at -8 to -12%. The
per-type decoder rewrite that was planned against the sweep's -24% is moot:
there is no gap to recover. The census-weighted FFN prediction built on the
contaminated table (30.43 TFLOPS) is withdrawn with it, along with the claim
that it explains the aggregate -- it matched for the wrong reason. Any future
per-type number from gemm_bench must come from a paired or rotated design, not
from "... all".

**Epilogues, and the harness noise floor.** gemm_bench originally timed only
the kStore epilogue, but the model runs gate as kStore when the pair is not
fused, up as kSwiGLU, down as kResidual and nine layers as the paired kGateUp.
The bench now measures all four, which produced two things.

The epilogue is not expensive. kSwiGLU and kResidual cost a few percent against
kStore at most: measured adjacently, store/swiglu/residual reads
31.62/29.02/28.13 for IQ3_XXS and 33.84/30.07/28.78 for Q4_K, and taking the
same comparison in the reverse mode order narrows store's lead to 0-6%. And
kGateUp is not available to IQ3_XXS or IQ3_S at all -- the dual launcher returns
false for them, the same coverage gap as above.

More useful, and worth stating plainly: **this harness cannot resolve
differences below about 10%.** gemm_bench times three blocks per process and the
first is a boosted one; parsing the peak instead of the settled block inflated
the earlier table by several points and flipped signs (Q8_0 read -6.4% on the
peak and +1.0% on the settled block, IQ3_S -5.7% and -12.0%). Reversing the arm
order moves a per-mode delta by ~5 points. The paired table above should
therefore be read as "within about a tenth of Q4_K", not to the digit, and no
conclusion should rest on one cell. The whole-model A/Bs are unaffected: they
run -r 1 after a discarded warm-up, so no block in them is a first block.

**Instruction mix: the decoder's extra work is hidden.** rocprofv3 PMC on the
production kernel (HalfPrefillGemmKernel<256,256,8,4,IQ3_XXS,kStore,...>: VGPR
144, LDS 64 KiB, 1024 threads, grid 8192x68, GRBM_GUI_ACTIVE at 100% of
GRBM_COUNT) gives the issue mix. The WMMA count is analytic (M*K*N/4096), so
SQ_INSTS_VALU divided by it is instructions per matrix op:

| type | VALU/WMMA | LDS/WMMA | non-WMMA VALU/WMMA |
| --- | ---: | ---: | ---: |
| Q4_K | 3.97 | 1.62 | 2.97 |
| IQ4_XS | 4.64 | 1.62 | 3.64 |
| IQ3_XXS | 5.36 | 1.62 | 4.36 |

IQ3_XXS issues **47% more non-matrix VALU per WMMA** than Q4_K and measures only
about 3% slower, so the kernel is **not issue-bound**: the decoder's extra
instructions are largely hidden behind other latency. That rules out the obvious
phase-two lever -- shaving the IQ3 decode -- without writing it.

It also sharpens what is left. The ~35% between the achieved ~30 TFLOPS and the
48.35 instruction ceiling is not the format (<=10% spread), not the epilogue (a
few percent), and not issue pressure (a 47% instruction increase costs 3%). That
leaves latency and the K loop's dependency structure, which is where the earlier
ablations already pointed. This part exposes no SQ_WAIT_* stall counters through
rocprofv3, so the next instrument is not a counter sweep.

**The K-loop scheduling hint is not it either.** The kernel applies
__builtin_amdgcn_iglp_opt conditionally per type: iglp_opt(0) to the IQ and
Q3/Q2 types, nothing to Q4_K/Q5_K/Q6_K/Q8_0. The bench's ablation bits 16 / 2048
/ 8192 select 0 / 1 / 2, and the sweep times all three in one process with the
order rotated per round. Minimum over rounds, in TFLOPS:

| type | iglp0 | iglp1 | iglp2 |
| --- | ---: | ---: | ---: |
| IQ3_XXS | 31.78 | 31.32 | 31.70 |
| IQ3_S | 30.05 | 30.11 | 29.93 |
| Q4_K | 33.29 | 32.42 | 31.03 |

On the IQ types the three are inside the noise and the sign of every adjacent
delta tracks which variant ran first (iglp0 reads +4.3% over iglp1 when it runs
first and -3.5% when it runs last). Only Q4_K shows a repeatable order,
iglp_opt(1) over iglp_opt(2) by 4-8% in all three rounds including the round
where iglp2 ran first -- and the production gating never applies iglp2 to
anything, so that is a property of the ablation, not a lever in the build.

Four explanations are now eliminated for the gap between the achieved ~30
TFLOPS and the 48.35 instruction ceiling: the weight format (<=10% spread and
IQ3_XXS at parity), the epilogue (a few percent), issue pressure (47% more
non-matrix VALU per WMMA costs 3%), and the K-loop scheduling hint (inside the
noise on the types that use it). What is left is the loop's dependency and
latency structure, and this part exposes no SQ_WAIT_* counters to measure it.

**Measured at the model level.** An existing pure-Q4_K artifact of the same
model settles whether the format matters end to end
(/home/q/models/gufo-sweep/base_q4kpure.gguf, 506 Q4_K tensors, 866 tensors,
every shape identical). --ids-file 2048 tokens, --chunk 2048, --repeat 1,
warm-up discarded, arm order alternated, a fixed 15 s gap and Tctl before every
timed run:

| rep | order | mixed IQ3/IQ4 (ms) | pure Q4_K (ms) | Q4_K |
| ---: | --- | ---: | ---: | ---: |
| 1 | mix first | 3848.1 (55 C) | 3554.1 (55 C) | +8.3% |
| 2 | Q4_K first | 3730.1 (53 C) | 3490.2 (54 C) | +6.9% |
| 3 | mix first | 3662.6 (53 C) | 3389.0 (53 C) | +8.1% |
| 4 | Q4_K first | 3729.8 (54 C) | 3517.8 (53 C) | +6.0% |

Q4_K wins every rep in both orders: **+7.3% on the means**. That part stands.
The *attribution* does not: with the per-type decoders at parity, "the mixed
file decodes more slowly" cannot be the explanation, and the paired-coverage
difference is separately measured at ~0. The 7.3% is therefore
**unattributed**. The candidates left are the epilogues and the non-FFN GEMMs,
and gemm_bench measures neither -- it times the kStore epilogue, while the model
runs gate as kStore, up as kSwiGLU, down as kResidual, and nine layers as the
paired kGateUp. The next measurement has to cover those, not the decoder.

**And the mechanism is work per clock, not clock.** Re-taken with SCLK and
package power sampled inside the prefill window (tail of the "tokens=" to
"run 0:" bracket, so the registration and Reset() memsets are excluded):

| rep | arm | ms | SCLK mean | SCLK max | power | TFLOPS | TF/MHz |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | mix | 3768.2 | 2307 | 2844 | 95.7 W | 29.35 | 0.01272 |
| 1 | Q4_K | 3552.9 | 2216 | 2664 | 91.0 W | 31.13 | 0.01404 |
| 2 | Q4_K | 3471.7 | 2216 | 2533 | 96.7 W | 31.85 | 0.01438 |
| 2 | mix | 3546.8 | 2399 | 2666 | 89.0 W | 31.18 | 0.01300 |
| 3 | mix | 3531.7 | 2396 | 2664 | 90.0 W | 31.31 | 0.01307 |
| 3 | Q4_K | 3495.2 | 2276 | 2759 | 98.4 W | 31.64 | 0.01390 |

Q4_K wins all three reps again, but by 3.1% on the means rather than 7.3%, and
the split is the result: **Q4_K does 9.1% more work per clock** (mean TF/MHz
0.01411 against 0.01293, ahead in every rep) while **running 5.5% lower clock**
(2236 MHz against 2367 MHz). The two partly cancel. The lower clock is
consistent with the pure-Q4_K artifact being 18% larger in bytes (15.39 GiB
against 13.08 GiB): it wins on instructions per weight and loses on power per
weight, and the operating point follows the power.

That is why a wall-clock-only comparison of this pair has read anywhere from +3%
to +8% across sessions while the work-per-clock figure sits near +9%. It is the
concrete case behind the methodology rule: an efficiency delta and a clock delta
are separate results, and only TF/MHz survives a session change.

**What that leaves.** With per-format decoder differences mostly under 6%, the
dominant unexplained term is the within-type efficiency gap: the ~30 TFLOPS
aggregate against the 48.35 measured instruction ceiling. That needs a profile,
not another sweep, and it is the last item on this page.

### The aggregate is accounted for, shape by shape

Earlier text on this page left part of the ~30 TFLOPS aggregate unattributed. It
is not unattributed. Parsing the artifact's GEMM inventory -- 866 tensors,
grouped by role, type, k and m, excluding token_embd (an embedding lookup) and
output (a GEMV on the last token in prefill) -- gives **49.55 GFLOP/token**.
Measuring every distinct (type, shape) with gemm_bench in one session, with the
FFN gate shape interleaved as a drift reference every five probes, and weighting
the rates by FLOPs:

| | |
| --- | ---: |
| real prefill GEMM FLOPs/token | 49.55 GFLOP |
| **predicted aggregate** | **30.41 TFLOPS** |
| measured aggregate | ~30 TFLOPS |
| instruction ceiling | 48.35 TFLOPS |

No free parameter, and it lands within 1.4%. There is therefore **no model-level
loss beyond the kernels**: the aggregate is the FLOP-weighted average of
per-shape rates, and every shape measured between 25 and 32 TFLOPS in that
session, i.e. 52-66% of the instruction ceiling.

Three details worth keeping. ssm_out at IQ3_S is the slowest real shape
(~25 TFLOPS) and attn_qkv at Q4_K the fastest (~31). The 96 tiny
ssm_alpha/ssm_beta projections (m=48) run at 11-14 TFLOPS, less than half of
everything else -- but their FLOP share is 0.05 GFLOP, about 0.2% of prefill
time, so they are not the problem the gufo investigation found them to be there.
And FFN is 70% of prefill GEMM FLOPs, not 67%.

Caveat on the correction: the interleaved reference drifted 36.5 -> 27.6 across
the session (24%), which is large, and each measurement is rescaled linearly to
the first reference. The aggregate is robust to that because numerator and
denominator come from the same run, but no single cell should be read to better
than a tenth.

**So the remaining target is one kernel's efficiency, not the model's
composition.** Every shape sits at 52-66% of 48.35 and the reason is uniform,
which matches the earlier ablation: staging costs 10% on Q4_K and 22% on
IQ3_XXS, and removing it reaches 34.9 TFLOPS (72%). What is left after that is
the K loop's LDS-read and WMMA dependency chain -- and this page has now
eliminated the weight format, the epilogue, issue pressure and the scheduling
hint as explanations for it.

**BK is confirmed at 4, not merely fixed at 4.** The tile space was swept but
BKDepth never was, and it is the parameter that sets LDS per stage: BK=4 fills
the 64 KiB budget and allows one block per CU, while BK=2 halves LDS and would
allow two. Swept with the order rotated per round, best TFLOPS per BK:

| type | BK=1 | BK=2 | BK=3 | BK=4 |
| --- | ---: | ---: | ---: | ---: |
| Q4_K | 21.63 | 29.36 | 29.42 | **31.72** |
| IQ4_XS | 27.58 | 28.66 | 28.58 | **30.39** |
| IQ3_XXS | 21.40 | 26.01 | 27.68 | **27.77** |

Monotone in every type. The extra loop iterations and barriers at lower BK cost
more than the second resident block returns, so occupancy is not the binding
constraint here and the one-block-per-CU design is not an oversight. IQ3_S cannot
take BK<4 at all: its LDS-transpose epilogue asserts Slots >= WM*WN with
Slots = 8*BK.

**This is clock, and the clock is a function of the kernel.** Efficiency per clock agrees to **0.3%** across the two harnesses that disagreed -- 0.0144 TF/MHz in both -- so there is no hidden code difference. The part has three SCLK levels (600 / 1408 / **2900 MHz**) and **never reaches the top one**: measured across power limits, 80 W gives 1799 MHz / 25.9 TF and 130 W gives 2016 MHz / 28.6 TF, work-per-clock is flat (0.0144, -1.5% between them), `gpu_busy` is ~85% at both, and the sampled maximum is 2221 MHz. The log's own line is that "what is broken is the conversion of watts into clock".

So the 26% is **not a fixed ceiling**: the clock is coupled to the kernel's power draw, and zeroed operands raise it ~20%. A kernel that toggles fewer bits and issues fewer instructions for the same FLOPs raises both the work per clock and the clock itself. At the 2900 MHz level the part never reaches, the peak is on the order of **60 TFLOPS**; at a realistic ~2200 MHz it is ~45, and the production ~30 is about two thirds of that. Treating ~31.5 as "the real rate, no headroom" was the wrong read. (The NPU's
32.4 TF, by contrast, is data-independent, so the two engines are at parity *at
the current clock*.)

**Correction: the clock does reach the top level, and it falls as load rises.**
Re-measured with SCLK and package power sampled at 20 Hz *inside the timed
prefill window* rather than over the whole process — the whole process is
dominated by weight registration and by `Reset()`'s memsets, which are high
clock and low information, see [methodology.md](methodology.md):

| prefill, mixed artifact | |
| --- | ---: |
| SCLK mean / max | ~2300 MHz / **2809 MHz** |
| package power mean | **~95 W** against a 130 W cap |
| Tctl | ~94 °C |

Two of the numbers above have to be read against that. "The 2900 MHz level the
part never reaches" is a property of whatever was being sampled, not of the
part: a plain prefill approaches it, and the DPM table is not even fixed — level
1 read 827 MHz and later 1027 MHz. And at ~95 W mean the 130 W cap was not
binding, so the 80 W / 130 W pair was not measuring a power ceiling either.

The coupling runs opposite to the naive reading, and it matters for tuning:
**clock falls as load rises.** A light region with few active CUs sits near
2700 MHz; adding work draws more power and pulls it back to ~2300. So an
efficient kernel is rewarded twice — more work per clock *and* a better
operating point — and a wall-clock delta cannot on its own be attributed to
instructions. Work per clock is the quantity that survives a session change.

int4 is the only measured lever above 1.5x: 52.6 TMAC/s, 2.1x int8 and 2.2x fp16. It requires four-bit activations, so it is a quality decision, and it is eligibility-bound (Q4_K/Q3_K, a low-double-digit percent of these shards).

### Where the block's cycles actually go

The kernel carries its own per-phase clock (gated by GUFO_FP16_PROTO_CLOCK)
which accumulates, for block (0,0), the cycles spent in each phase. Enabled on
the bench:

| phase | share of block cycles |
| --- | ---: |
| K loop (LDS read + WMMA) | 68-70% |
| commit (register -> LDS) | 7.1% |
| barrier | 4.5-4.9% |
| fetch (global -> register) | 4.2-4.9% |
| prologue and epilogue | ~14% |

Two of six runs came back wrapped (the end timestamp below the start), so the
profiler's writes race and only the stable runs should be read; the split was
consistent across those. The operational consequence is the part that matters:
**staging plus barriers is ~16% of the block**, so a warp-specialised or
double-buffered dataflow can win at most that much. The other 69% is inside the
K loop, which is a dataflow question rather than a parameter.

### The K loop's operand grouping is mis-set

group_shift controls how many row tiles share a token tile inside one grid
group, that is, which operand stays resident while the other streams. It is
hand-set per type -- 5 for IQ4_XS, 5-or-3 for Q5_K, 3 otherwise -- and exposed
as the GShift template parameter, so it can be swept without touching the
default. Swept 0-6 with the order rotated every round, best TFLOPS:

| type | 0 | 1 | 2 | 3 | 4 | 5 | 6 | production |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| IQ3_XXS | 32.16 | 32.25 | 31.36 | 31.37 | 30.88 | 30.45 | 30.46 | 3 |
| IQ4_XS | 31.66 | 28.53 | 28.34 | 28.52 | 28.39 | 28.19 | 28.62 | 5 |

Identity order (group_shift 0) is worth +2.5% on IQ3_XXS and +12% on IQ4_XS in
the bench, in the opposite direction to the hand-set values. A first pass read
it the other way round: gshift 0-6 were run in a fixed order, which hands the
boosted first block to gshift 0 and reads the session's decay back as a
preference. The rotated sweep is the one above.

A bench result on one isolated GEMM is not a model result, and this one does
not transfer. Setting group_shift to 0 and measuring the model (warm-up
discarded, -r 1, order alternated, fixed 15 s gap before every timed run):

| rep | order | identity order (ms) | production 3/5 (ms) |
| ---: | --- | ---: | ---: |
| 1 | identity first | 3937.3 | 3795.5 |
| 2 | production first | 3939.7 | 3843.3 |
| 3 | identity first | 3988.8 | 3917.5 |
| 4 | production first | 3951.1 | 3841.7 |

Production wins every rep, by 2.7% on the means, including the reps where the
variant ran first. So the hand-set grouping is right and the bench is measuring
something the model does not see: in the model the same L2 is shared with the
attention, SSM and paired-kernel traffic, so an isolated GEMM's operand
residency is not the model's. The change was reverted.

The general lesson is worth more than the result. This bench reproduces the
*kernel's* instruction and phase behaviour well -- the per-phase clock above,
the VALU-per-WMMA counts, the BK ordering all agreed with the model -- but not
its memory-system behaviour. Grouping, residency and reuse must be measured end
to end.

### The K loop is stalled, not bandwidth-bound

The natural structural fix for the K loop is to stop storing decoded fp16
weights in LDS and keep them packed 4-bit, cutting the A operand's LDS traffic
and capacity fourfold. Measured, that premise is false. rocprofv3 PM counters
on the production kernel, per dispatch:

| | Q4_K | IQ3_XXS |
| --- | ---: | ---: |
| LDS/TA busy against GRBM cycles | **25.9%** | **38.4%** |
| SQ issue slots busy (of 160) | 12.8% | 12.2% |
| LDS reads per WMMA | 1.62 | 1.62 |
| total instructions per WMMA | 5.61 | 6.99 |

LDS is at a quarter to a third of capacity, so packing the weights would
optimise a resource with 60% headroom. The issue slots are at 12-13%, so the
kernel is **stalled roughly 87% of the time** -- which is consistent with the
K loop's own LDS-read-to-WMMA dependency chain, and inconsistent with any
bandwidth account. The design was not built.

What would help a stalled kernel is more independent work in flight, and the
configuration is already at the limits that provide it: one block per CU set by
64 KiB of LDS, 8 waves per SIMD, and 8 independent accumulator chains
(WRS=2 x WTS=4). The tile/thread combinations that would add either more waves
or more chains -- w4n4 at 16 accumulators, w8n8 at 2048 threads, w16n4 at 2048 --
exceed the VGPR budget or the 1024-thread limit. With WMMA = WRS*WTS and LDS
reads = WRS + WTS, an 8-accumulator tile needs 6 reads however it is split, so
the current split is already the minimum for this shape.

> **Corrected.** The w4n4 half of that sentence is false, and it was checkable
> without running anything: 256x256 split 4x4 over 512 threads compiles to **242
> VGPR with zero spills** (250 with the iglp hint off) against the 256 the
> 512-thread budget allows, i.e. it fits with 14 registers to spare. It was
> therefore never a budget exclusion, it was an unmeasured assumption. It has now
> been timed, and it is a wash (-0.56%) rather than a loss, which still leaves the
> shipped split in place but for a different reason -- see "The cost of
> scheduling" below.

### The K loop is already near its own limit; the deficit is everything else

The phase profile above shows the K loop taking 69% of the block and staging
plus barriers about 16%, which reads as "optimise the K loop". A synthetic
replica of the K loop's own instruction pattern says otherwise. With the same
mix -- twelve ds_read_b128 feeding eight WMMA per step, six fragments, eight
accumulator chains -- but ideal conflict-free LDS and nothing else, at three grid
sizes (engine/gpu/klook_replica.hip, built with build_gpu.sh klook_replica):

| blocks | waves | TFLOPS | share of the 48.35 ceiling |
| ---: | ---: | ---: | ---: |
| 40 | 320 | 42.70 | 88% |
| 160 | 1280 | 42.52 | 88% |
| 640 | 5120 | 45.59 | 94% |

So the K loop's instruction mix is not the limit: it reaches 88-94% of the
ceiling by itself. And 94% of the 69% the K loop occupies is 65%, which is the
66% the whole kernel measures. **The K loop is already running at 90%-plus of
what it can do, and the deficit is the other 31% of the block**: commit 7.1%,
barrier 4.5%, fetch 4.2%, prologue and epilogue about 14%.

That inverts the target. The work is not the inner loop, it is the staging and
the epilogue -- which is also why every inner-loop change above measured
neutral. The epilogue is the largest single item at ~14%, and one part of it is
concrete: the kStore path writes its output as **fp32**, 143 MB for one FFN gate
GEMM, which at 187 GB/s is about 7.6% of that GEMM on its own. The kSwiGLU path
already writes fp16 for the same shape.

Two harness notes so this is reproducible. The MAC count is per **wave**, not per
thread -- a WMMA is a wave operation, and counting threads reports 35x the
ceiling. And the fragment index needs an opaque barrier: (s*6+f)&63 has period
32 steps, so without it the compiler folds the whole loop into a multiply.

### The staging chain is the deficit, and it is stall-bound

Disabling individual phases by ablation prices them, and the answer is not the
K loop:

| ablation | Q4_K | IQ3_XXS |
| --- | ---: | ---: |
| control | 40.66 | 33.60 |
| no commit (register -> LDS) | **53.62** | **52.00** |
| no fetch (global -> register) | 40.87 | 37.93 |
| no barrier | 40.58 | 33.78 |
| no store | 34.66 | 32.60 |

Removing the commit is worth **+32% (Q4_K) and +55% (IQ3_XXS)**. That number must
not be read as the LDS write cost, for two reasons, both checked.

First, the replica: adding the commit's LDS writes to the synthetic loop leaves
it at 94-96% of ceiling against 95-98% for reads alone, so LDS writes are
essentially free here and the write-contention account is wrong.

Second, and this is the trap: **with the commit gone, ra and rb have no
consumer**, so the compiler deletes the global loads and the decode with it. The
ablation prices the entire staging chain at once -- global load, decode and LDS
store -- including the latency stalls in it. The phase clock cannot see this: it
counts *issue* cycles per phase and reported fetch at 4.2% and commit at 7.1%,
while removing the chain they belong to is worth 32-55%. Issue cycles are not
stall cycles.

So the corrected target is the staging chain, and it is **stall-bound rather than
issue-bound**. That is why every inner-loop change measured neutral while the K
loop already sits at 95%-plus of its own limit: the kernel is waiting on global
weight loads and their decode, and the K loop is not the thing to optimise.

The move that follows is to overlap the staging with the compute, which is what
double-buffering would do -- and the upstream log records that as blocked by
registers rather than LDS. That constraint deserves re-deriving in this engine
rather than inheriting, since a different engine is free to spend registers
differently.

The noStore arm is invalid in the other direction: it stays *slower* than the
control, because the store lambda keeps the accumulators live and compares them
instead of discarding them.

### The alternative instruction classes are all slower

Everything on this page is a tile GEMM on WMMA. The last open algorithmic
question is whether a different execution class does more MACs per cycle, since
that would change the algorithm rather than its parameters. Measured with eight
live chains at three grid sizes (engine/gpu/dot_peak.hip, built with
build_gpu.sh dot_peak):

| instruction | TMAC/s | vs fp16 WMMA |
| --- | ---: | ---: |
| v_dot8_i32_i4 | 28.99 | 1.20x |
| v_dot4_i32_i8 | 14.48 | 0.60x |
| v_dot2_f32_f16 | 11.67 | 0.48x |
| v_pk_fma_f16 | 12.04 | 0.50x |
| WMMA fp16 (reference) | 24.17 | 1.00x |
| WMMA iu8 (reference) | 25.15 | 1.04x |
| WMMA iu4 (reference) | 52.6 | 2.18x |

Three things follow. **WMMA is the right execution class**: no dot instruction
beats it, and the vector classes are half its rate, so replacing WMMA with plain
packed FMA is a 2x loss rather than a freedom. **v_dot4_i32_i8 is slower than
fp16 WMMA** at 0.60x, which kills the w8a8 route on rate alone -- that route
would otherwise have side-stepped the 4-bit activation quality cost, since int8
activations would be 16x finer. And **v_dot8_i32_i4 reaches only 1.20x where int4
WMMA reaches 2.18x**, so even accepting 4-bit activations the dot path is the
worse way to spend them.

That closes the instruction-class axis: the only lever above WMMA is the int4
WMMA, and it is closed on quality above.

### The pipelining alternatives are all worse

The shipped K loop consumes each B fragment immediately after loading it -- a
load-to-use distance of zero -- and does not interleave the next stage's global
fetch into the K loop. Both alternatives exist behind ablation bits and had never
been swept. Best TFLOPS over four rotated rounds:

| variant | IQ3_XXS | Q4_K |
| --- | ---: | ---: |
| shipped (control) | **36.60** | **36.61** |
| batch every B load before the MMA block (4096) | 33.90 | 31.89 |
| interleave the next global fetch (256) | 25.33 | 26.36 |
| iglp_opt(1) instead of (0) (2048) | 33.01 | 33.81 |

Both lose, the fetch interleave badly. Zero load-to-use distance is not a defect
here: the scoreboard covers it, and batching the loads raises register pressure
enough to cost more than the latency it hides. The ordering held in every round,
so it is not position.

### Independent confirmation of the upstream closure

Between this section, the BK sweep, the tile sweep, the grouping test, the phase
profile and the counters, the fp16 prefill inner loop has now been probed from
the parameter, scheduling, memory and occupancy directions *in this engine*, and
the shipped structure wins on all of them:

- parameters: tile family closed, BK=4 monotone best, epilogues a few percent;
- scheduling: iglp values inside noise, B-load batching and fetch interleaving
  both worse;
- memory: LDS capacity full but units at 26-45%, grouping changes do not
  transfer from the bench;
- occupancy: one block per CU by LDS capacity, and doubling it via BK=2 loses.

That reproduces gufo's conclusion -- 66% of a 48.3 TFLOPS ceiling -- from an
independent engine and a different codebase. Worth having: it means the closure
is a property of the hardware and this problem shape, not of gufo's ABI or its
compatibility constraints. The one lever that is not structural is int4, and it
is closed on quality above.

### This was already settled upstream

Most of the section above re-derives a conclusion the reference engine's
experiment log already records (gufo
docs/models/qwen3.8-27b/EXPERIMENTS.md: the tile sweep, the double-buffering
section and the wave64 section). Stated once here so it is not re-derived a
third time:

- **LDS is saturated in capacity, not in bandwidth.** 64 KiB per block is the
  whole per-CU LDS, and that is what fixes one block per CU and 8 waves per SIMD.
  The TA-busy counters read 26-45% of cycles, so the LDS *units* keep headroom
  even though the allocation is full. Conflating those two is easy and wrong:
  capacity saturation blocks double-buffering, unit headroom means extra reads
  would not help.
- **Occupancy is not what is binding.** The tile sweep, the BK sweep and the
  wave arithmetic all agree.
- **Double-buffering the staging is blocked by registers, not by LDS.** The
  upstream variant table has the LDS arithmetic working and occupancy improving,
  with the accumulator-and-operand VGPR pair running out.
- **wave64 is not a lever.** wmma_f32_16x16x16_f16_w64 measures 48.50 TFLOPS
  against w32's 48.35 -- the same 4096 MACs issued over two cycles, half the
  accumulator registers and no throughput.
- **The inner loop cannot be widened.** LDS reads per WMMA are
  (WRS + WTS) / (WRS * WTS). For an 8-accumulator tile that is 0.75 per lane
  fragment, 1.62 measured ds_read per WMMA, and it can only fall by adding
  accumulators -- the same register wall.

So the fp16 prefill efficiency question is closed at ~66% of a 48.3 TFLOPS
ceiling, and it was closed before this page was written.

What this page adds is the other half of the int4 decision. The upstream log
calls int4 a quality decision without pricing it, and notes it is eligibility
bound on these shards (Q4_K and Q3_K only, ~15.6% of the 3.84 bpw artifact).
Priced above: four-bit activations alone cost +3.52% perplexity, and the IQ3/IQ4
FFN tensors are non-uniform codebooks that would have to be requantized to
uniform int4 on top. Both halves of the int4 trade are now measured, and it
loses on this artifact.

### The int4 lever is priced, and it is expensive

int4 WMMA is the only measured lever above 1.5x on this part -- 52.6 TMAC/s,
2.1x int8 and 2.2x fp16. It is not a scheduling or format choice, it is an
accuracy choice, and the instruction takes four-bit operands on **both** sides:
there is no w4a16 int4 path, so a w4a4 operand error has to be accepted.

That error is measurable before the kernel exists. YAH_QFFN=4 rounds the FFN
activations in place with the same per-32 block scale and clamp the cache
quantizer uses, which is the operand error the MMA would introduce; the
accumulation is fp32 either way. Next-token KLD against the f16 reference over
512 tokens of repo prose, all arms on f16 KV so the activation rounding is the
only variable:

| config | KLD mean | KLD p99 | top-1 same | implied perplexity |
| --- | ---: | ---: | ---: | ---: |
| FFN activations 4-bit | 0.034560 | 0.219102 | 90.62% | **+3.52%** |
| FFN activations 3-bit | 0.127120 | 0.762251 | 83.98% | +13.56% |
| FFN activations 2-bit | 1.433313 | 6.846752 | 51.76% | +319% |
| q4 KV, for scale | 0.005012 | 0.028196 | 97.85% | +0.50% |

Four-bit activations alone cost seven times what a q4 KV cache costs, and 3-bit
is worse than anything else measured here. That is *before* the weight side: the
instruction needs uniform 4-bit weights, and this artifact's FFN is IQ3/IQ4,
which are non-uniform codebooks, so those tensors would have to be requantized
to uniform int4 as well -- a second loss on top, and a requantization the
project has ruled out. The standard mitigation is an offline Hadamard rotation
absorbed into the weights (W_rot = W @ H), which costs nothing at run time but
requires storing transformed weights; it is not tested here.

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

## Where the prefill deficit actually is: a component flow

The kernel is ~59% of the fp16 instruction ceiling at the production shape, and
the deficit had been attributed to "staging" without splitting it. Adding one
ablate bit per feeder and weaving them makes the arms cumulative, so the deltas
are additive and read as a flow. Q4_K, 17408x5120, batch 2048, BK=4, one process
per arm, order rotated per round, settled arm = the tail rounds.

| arm | ablate | what it adds | TFLOPS | share of ceiling | step cost |
| --- | ---: | --- | ---: | ---: | ---: |
| K loop alone, stale LDS | 6 | `ds_load` + WMMA only | 50.7-51.7 | ~105% | — |
| + global fetch | 2 | next stage's global loads | 51.0-51.9 | ~106% | **~0%** |
| + decode, no LDS store | 36 | the Q4_K unpack + scale | 47.8-48.2 | ~99% | **-6%** |
| + LDS store of raw | 68 | `ds_store` with no decode | 38.2-38.6 | ~79% | **-25%** |
| + decode + LDS store | 4 | both (full commit) | 32.1-34.3 | ~68% | **-12%** |
| + global fetch on top | 16 | production control | 29.4-31.9 | ~63% | **-7%** |

The K loop on its own is *at* the ceiling, so there is no headroom inside it and
every point of the deficit is feeder work. Two things are not obvious from the
aggregate:

- **the global fetch is free on its own.** It is already one stage ahead, so its
  latency is covered; it only costs ~7% once the commit exists, i.e. it competes
  for something the commit is already using.
- **the two costs are super-additive.** Decode is -6% alone and the store is
  -25% alone, but together they are -37%.

### The store is the locus, and it is not throughput

Splitting the commit by side, all with the global fetch present:

| arm | ablate | ms | vs full |
| --- | ---: | ---: | ---: |
| full | 16 | 12.41 | — |
| no decode (keep both stores) | 80 | 9.56 | -23% |
| no A store (keep decode) | 48 | 8.30 | -33% |
| no B store | 16400 | 8.57 | -31% |
| no A or B store | 16432 | 7.81 | -37% |
| no commit at all | 18 | 7.14 | -42% |

The commit writes two things per stage: the decoded weights (A) and a plain copy
of the activations (B), 4 `ds_store_b128` per thread per stage. Removing either
stream alone recovers ~85% of the total store saving, so the cost is **not**
proportional to the bytes or the instruction count — it is a shared serialization
that either stream is enough to trigger. B alone, which needs no decode at all,
is worth -31%: the activation staging copy costs as much as the entire weight
pipeline.

### Mechanisms tested and falsified

Every mechanism that would explain a per-stage store cost this large was tested
and came back negative:

| candidate | test | result |
| --- | --- | --- |
| LDS port throughput | counters | LDS busy 26-45%, never saturated |
| bank conflicts | `LDSBankConflict` PMC | **0** across all 13 dispatches |
| register spills | `.vgpr_spill_count` | **0** for Q4_K; only Q8_0 spills, 4 instrs |
| the VGPR cap | `.vgpr_count` | control uses 192 of 256 available, so the compiler was not capped |
| store->load `s_waitcnt` drain | double-buffer the LDS stage | -0% to -6%, i.e. no help |
| the drain being conservative | double-buffer with the stage loop unrolled by two so the buffer index is compile-time and the two staging regions are provably disjoint | **17% worse** |
| the barrier itself | ablate 17 (no barrier) | -5% only |
| decode in the wrong place | ablate 512 (decode in the K loop) | parity; 1024 (staggered) is 8-18% *worse* |
| B load-to-use distance of zero | ablate 4096 (batch all B loads ahead of the MMA block) | +0.5%, within noise |
| interleaving the fetch | ablate 256 (fetchmid) | **22% worse** |
| iglp_opt(1)/(2) | ablate 2048/8192 | 5% worse |

The double-buffer result is the informative one. On the double-buffered build the
stores *are* issued at the top of the stage and consumed a stage later, exactly
as intended -- the ISA confirms it -- but the compiler still emits
`s_waitcnt lgkmcnt(0)` before the barrier, so the drain is unchanged. Three full
`lgkmcnt(0)` drains per stage appear in the control kernel's schedule. That
looked like workaround-able compiler conservatism, so the stage loop was
unrolled by two to make the buffer index a compile-time constant and the two
staging regions provably disjoint. It came back 17% *worse* than the
single-buffered BK=2 arm, which closes the drain explanation: even if the wait is
what the schedule costs, removing it does not recover the time.

**Conclusion:** the deficit is the LDS store stream, it is a whole-schedule
effect rather than any single micro-mechanism, and the ablations show it is worth
~40% of the kernel -- more than every other feeder combined. Any fix has to make
the staging stores disappear or make them overlap, not make them cheaper per
store.

**Falsifier for the next attempt:** a change that keeps the A/B staging stores
and still beats 31 TF is either not measuring the same thing or has moved the
cost into the K loop; a change that removes the staging stores must come back
near 50 TF (the K-loop-only arm) to be believed.


### The store cost is triggered by writing two regions, not by traffic

The obvious next cut is: does the penalty track DS store **bytes** or DS store
**instruction count**? Holding the tile and shape fixed, the commit's 32 bytes per
call can be reached with 2 x ds_store_b128, 4 x ds_store_b64 or 8 x ds_store_b32
(the swizzle is identical), so bytes are constant while the instruction count
doubles and then quadruples:

| arm | DS stores/thread/stage | bytes | ms |
| --- | ---: | ---: | ---: |
| b128_64B | 4 | 64 | 12.3 |
| b64_64B | 8 | 64 | 12.1 |
| b32_64B | 16 | 64 | 11.9 |
| b128_32B | 2 | 32 | 8.12 |
| b64_32B | 4 | 32 | 8.10 |
| b32_32B | 8 | 32 | 8.07 |

Within a byte row the time is **flat across a 4x change in store instruction
count**, so instruction issue is not the mechanism. Bytes do correlate -- but then
the byte model fails immediately, because it is not the bytes either:

| arm | arrays written | bytes | DS stores | ms |
| --- | --- | ---: | ---: | ---: |
| none | - | 0 | 0 | 8.48 |
| halfA | s_b only | 32 | 2 | 8.74 |
| sameRegion | s_a twice | **64** | **4** | **8.89** |
| halfB | s_a only | 32 | 2 | 9.40 |
| bothHalf | s_a + s_b | 32 | 2 | 12.10 |
| full | s_a + s_b | 64 | 4 | 13.67 |

\`sameRegion\` writes twice the bytes with twice the store instructions of
\`halfA\` -- identical to \`full\`'s traffic -- and costs 8.89 ms, barely above no
stores at all. \`bothHalf\` writes half of \`full\`'s bytes with half its store
instructions and still costs 12.10 ms. Writing the same staging array twice is
nearly free; writing the second array at all costs ~4 ms.

That is not a footprint effect: the same absolute ~4.3 ms appears at BK=2, where
both staging tiles are half the size and the whole working set is 32 KiB.

| arm | stores | ms |
| --- | --- | ---: |
| bk4 none | off | 7.7 |
| bk2 none | off | 7.8 |
| bk4 full | on | 11.9 |
| bk2 full | on | 12.5 |

**SUPERSEDED -- this reading was an artifact of dead-code elimination. See
"The ablation lattice was invalid" below.** The store-removal arms also deleted
the dependent LDS *loads*, so what looked like a store cost was mostly load count
plus schedule. The measurements above are real; their attribution is not.

### Occupancy and register pressure are ruled out

Two reads of the same lattice looked promising and are both wrong. Removing a
store stream lets the compiler delete that array, so the declared LDS size falls
64 KiB -> 32 KiB -> 0 and the VGPR count falls 192 -> ~110. That suggested the
cost was really the occupancy cliff at 64 KiB (one block per CU) or at 192 VGPRs.
Neither survives:

| tile | threads | declared LDS | VGPR | none | full | gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256x256 w8n4 | 1024 | 65536 | 192 | 7.91 | 11.10 | 3.19 |
| 128x128 w4n4 | 512 | 32768 | 112 | 8.48 | 12.57 | 4.09 |
| 128x128 w8n4 | 1024 | 32768 | **80** | 8.73 | **15.03** | **6.30** |

A 128x128 tile declares half the LDS and needs far fewer accumulator registers,
so it is the configuration that should be able to hold two blocks per CU -- and it
has the *largest* store gap, not the smallest. The 128x128 w8n4 arm has the
**lowest** VGPR count of any full-store arm and is the **slowest**, so VGPR count
does not track the cost either. Without staging, all three tiles land at 7.9-8.7 ms,
i.e. the tile barely matters; with staging, the tile matters a lot.

### Where this leaves it

The mechanism is unresolved, and the reason is instrumentation: on gfx1151 this
ROCm exposes no `SQ_WAIT_*` stall counters and no working `OccupancyPercent`,
so every conclusion here is black-box wall-clock, and each ablation changes the
whole compiled artifact rather than one quantity. What can be said precisely is
that the staging chain costs 30-60% of the kernel, that the trigger is writing a
second distinct staging region rather than any measurable property of the stores
themselves, and that no store-side tuning helps.

Note also that the absolute gap is not stable across sessions: the same
full-vs-none pair measured 5.19 ms in one process and 3.19 ms in another, because
the box's operating point drifts. Only within-session comparisons are valid here.

**Falsifier:** any change that keeps both staging arrays separate and still runs
under ~10 ms at this shape falsifies the two-region reading; the reduction must
come from not writing a second distinct array, not from writing the two cheaper.

### Working on the second region: reading the activations from global

The constructive reading of the two-region result is to stop staging the
activation tile at all. That is more reachable than it looks, because the commit
already lays B into LDS as `[K16][token][16 halves]`, which is a plain blocked
layout. So a pre-blocked global buffer `[k/16][batch][16]` lets each lane fetch
its whole WMMA fragment as one contiguous 32-byte load, with the warp's sixteen
distinct lanes spanning 512 contiguous bytes. Lane `sl` needs token
`((wt*WTS)+j)*16 + (sl^(ks&3))`, and one j step is sixteen tokens, i.e. 256
halves, so the address is one 64-bit computation per K step plus a 32-bit add per
fragment. Ablation 1048576 does this and drops the LDS store and fetch for B; the
buffer rides in through the kernel's unused `up` argument.

| arm | ms |
| --- | ---: |
| noStoreB (B staged and read from LDS, but never written) | 8.27 |
| full (B staged) | 10.29 |
| globalB, address recomputed per load | 13.25 |
| globalB, address hoisted per K step | 12.0 |

The repack costs **0.20 ms** for the whole 20 MiB activation matrix, which is
negligible next to the ~2 ms the second staging region costs. The per-load 64-bit
address was worth 1.25 ms of pure overhead, so the first globalB row was partly
addressing.

What the hoisted version shows is that the global path reaches **parity** with the
LDS staging (11.5 vs 12.0, full ahead in three of four paired rounds) but does not
pass it, and it stays far above the noStoreB floor. The reason is the reuse
factor: staging collapses the activation reads to one fetch per block, so the
K loop's eight M-warps read LDS; without staging each of those warps issues its
own global loads and the read volume goes up 8x. That 8x is the price of not
staging, and it is about exactly what the staging costs.

The reuse factor was the obvious suspect, so it was swept. With B in global the
B reuse factor is exactly WM (each M-warp re-reads the same activations), so a
low-WM/high-WN aspect should cut the redundant reads, at the cost of more A reads
from LDS which are cheap:

| arm | mean ms |
| --- | ---: |
| full w8n4 (LDS staging) | 12.46 |
| globalB w8n4 | 12.60 |
| globalB w4n8 | 12.64 |
| globalB w2n16 | 12.96 |
| noStoreB floor | 8.76 |

Cutting the redundancy 4x changes nothing, so the redundant global reads are not
the driver either. The two costs simply match: the second staging region costs
12.46 - 8.76 = **3.70 ms** and the global-load path that replaces it costs
12.60 - 8.76 = **3.84 ms**. Staging through LDS and re-reading from global are the
same price here, which is why the second region cannot be removed for free.

### Spending the freed LDS: tested, and it buys nothing

Removing the activation staging frees 32 KiB of the 64 KiB budget, and the weight
tile alone is BK * RS * 16 * 16 halves, so BK=8 is exactly the whole budget. That
halves the stage count and with it the barriers and the per-stage commit
overhead, at unchanged total byte traffic. It was measured twice:

| arm | session A (mean) | session B (mean) |
| --- | ---: | ---: |
| full BK=4 (production) | 11.50 | 12.74 |
| globalB BK=4 | 11.83 | 12.77 |
| globalB BK=8 w4n8 | **11.11** (won 3 of 3 rounds) | 13.16 (won 1 of 4) |
| globalB BK=8 w8n4 | 11.65 | -- |
| noStoreB floor | 8.24 | 8.97 |

Session A read as a 3.4% win over production and session B reverses it. The effect
is smaller than the between-session drift, so **it is not established** and the
apparent win in session A was noise. BK=8 on its own (w8n4) is neutral, so the
depth is not paying for itself even where it fits.

The same freed budget makes BK=4 genuinely double buffered for the first time (two
32 KiB weight buffers), which is the configuration the original double-buffer
attempt could never reach. It is a **2x regression**: 24.0 ms against 12.7 for the
single-buffered control, in every round. Unrolling the stage loop by two to make
the buffer index compile-time duplicates a BK=4 K loop twice over, and the result
is instruction-bound rather than store-bound. Double buffering is now closed at
BK=2, at BK=4 with a runtime index, and at BK=4 unrolled.

**Net result:** the second staging region can be replaced at parity, and the LDS it
frees has no configuration that reliably beats the current design. The activation
staging stays.

### The ablation lattice was invalid

Every "removing a store stream costs X" number in this document was measured with
an ablation that also deleted work. Dropping the store makes the corresponding
staging array dead, LLVM elides the array, and then **the K loop's loads of that
array are eliminated too**. Counting the LDS instructions per kernel makes it
plain:

| arm | ablate | `ds_load` | `ds_store` | wmma | ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 16 | 48 | 4 | 32 | 13.67 |
| noStoreB (A store only) | 16400 | 16 | 2 | 32 | 9.40 |
| noStoreA (B store only) | 48 | 32 | 2 | 32 | 8.74 |
| noStoreAB | 16432 | 0 | 0 | 32 | 8.48 |
| noCommit | 18 | 0 | 0 | 32 | 7.14 |
| sameRegion | 32784 | 16 | 2 | 32 | 8.89 |

`sameRegion` was meant to write one staging array twice and it did not even do
that: the two stores coalesced to two instructions and the B loads vanished, which
is why it looked cheap. `noStoreAB` is a kernel with *no LDS traffic at all*, not
a kernel with the stores removed.

### What survives when the counts are held fixed

Two comparisons do hold the LDS instruction counts constant, and they reverse the
earlier conclusion:

| arm | ablate | `ds_load` | `ds_store` | ms |
| --- | ---: | ---: | ---: | ---: |
| full | 16 | 48 | 4 | 13.67 |
| bothHalf (each store half width) | 524304 | 48 | 2 | 12.10 |
| no decode | 80 | 48 | 4 | 9.56 |

- **Stores are nearly free.** Halving the store instructions at a fixed 48 loads
  is worth 1.57 ms; the earlier ladder attributed ~4 ms to the same change. The
  store *width* sweep is consistent: b128, b64 and b32 move the same bytes in the
  same time.
- **The decode is the largest single component: 13.67 -> 9.56 = 4.11 ms (23%)**
  with byte-identical LDS counts. This is the one attribution in this document
  that a count audit supports.

Reducing the decode's instruction count does not recover it, though. Rewriting the
per-element fp32 mul/sub/convert chain as a single fp16 fused multiply-add halves
its VALU count and is worth only **2.8%** (11.40 vs 11.73 ms, faster in three of
four paired rounds, again with identical LDS counts). So the decode costs 23% of
the kernel but its instructions are not the reason.

That is the same shape as every other result here: the cost is real and
reproducible, and it resists being removed by making the work cheaper. The most
consistent explanation left is register live range and scheduling -- the decode's
outputs must stay live across the commit -- but this part exposes no counter that
can confirm it.

**Consequence:** the activation staging stays, and LDS restructuring is the wrong
target. The measurements that were valid all say the same thing -- BK, tile shape,
warp aspect, store width, staging strategy and double buffering are neutral or
worse, and the LDS stores themselves cost ~0.3 ms. What remains unaddressed is the
decode's 23%, and the only lever with a known mechanism is removing the decode
from the kernel entirely by materialising fp16 weights (the `no decode` arm is
that upper bound).


**Falsifier:** if a variant that keeps both staging arrays separate and spends the
freed LDS on more K depth runs under the noStoreB floor (~8.5 ms at this shape),
the two-region cost is depth-limited rather than store-limited and this whole line
is mis-framed.




### Materialised fp16 weights: tried, bit-exact, and a wash

The `no decode` arm said the decode was worth 23%. The honest test of that is to
remove the decode for real: keep the weights pre-dequantised as fp16 and read them
directly, so the kernel never unpacks anything. Ablation 16777216 does this, and
the fp16 tensor is built with the kernel's own `LoadRaw`/`DecodeRaw` and the same
fp32 scale/offset math, so the two kernels must agree exactly. They do:

```
fp16w check: worst_abs=0 bit_mismatch=0/35651584 nonfinite=0     (all three formats)
```

Bit-identical output, identical LDS instruction counts (48 loads, 4 stores), same
32 WMMA per stage, rotated paired rounds:

| format | bpw | fp16/bpw | quantised | fp16 weights | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Q4_K | 4.50 | 3.6x | 10.30 | 10.29 | wash |
| Q5_K | 5.50 | 2.9x | 10.70 | 13.66 | **-25%** |
| Q6_K | 6.56 | 2.4x | 11.27 | 10.59 | **+6%** (4 of 4 rounds) |

So the decode is not worth 23%: removing it outright on Q4_K returns nothing, and
the ablate-80 figure does not reproduce when the decode actually leaves the kernel.
What materialising does is trade the decode ALU for a wider weight read, and the
exchange rate depends on the source format -- 3.6x the bytes is break-even on
Q4_K, 2.9x loses badly on Q5_K, 2.4x wins modestly on Q6_K. It is a bit-width
trade, not a decode trade.

**Verdict: not worth it.** The prefill FFN on the primary target is the IQ family
(IQ3_S, IQ3_XXS, IQ4_XS at 3.4-4.25 bpw), which is *wider* than 2 bytes per
element in fp16 by more than Q4_K is -- so the exchange rate there is worse than
the Q4_K wash. Against that, per-layer materialisation costs 510 MiB for the FFN
and 731 MiB for a whole layer on a 24 GiB budget. Paying memory for a wash, or a
loss, is the wrong trade.


## Instrumentation: what actually works on gfx1151

Two things were wrong with the profiling earlier in this document, and both cost
real time.

**PC sampling is unavailable.** `rocprofv3-avail list --pc-sampling` reports no
agents on gfx1151, so per-instruction stall attribution is not possible. Every
conclusion has to come from counters plus controlled experiments.

**Only one PMC is collected per `rocprofv3` run.** Passing several `--pmc`
flags does not collect several counters: all but one are silently dropped and the
run still exits 0. My earlier multi-counter attempts returned a single counter and
I read that as "the counter is unsupported". Run one counter per invocation. The
suite that works:

```
SQ_INSTS_VALU  SQ_INSTS_SALU  SQ_INSTS_LDS  SQ_INSTS_SMEM  SQ_INSTS_FLAT
SQ_INSTS_WAVE32_VALU  SQ_INSTS_WAVE32_LDS  VALUInsts  SALUInsts
GRBM_COUNT  SQ_WAVE_CYCLES  SQ_WAVES  SQ_BUSY_CYCLES  GPUBusy
MemUnitBusy  L2CacheHit  OccupancyPercent  MeanOccupancyPerCU
TA_BUSY_avr  LDSBankConflict
```

Two caveats found the hard way. `MeanOccupancyPerCU` and `OccupancyPercent`
return **0** when the kernel is short -- a 40-block launch at 0.65 ms reports
nothing -- so they can only be read at production grid size. And the occupancy
metric equals `SQ_WAVE_CYCLES / (GRBM_COUNT * cu_count)`, which is a
wave-residency average, not a stall-weighted one.

### The production kernel, measured

Per dispatch, Q4_K 256x256 BK=4, M=17408 K=5120 batch=2048:

| counter | value | reading |
| --- | ---: | --- |
| `GRBM_COUNT` | 23.67 M cyc | 8.45 ms, matches wall clock |
| `MeanOccupancyPerCU` | **15.59 waves** | of 32 slots = 49% |
| `SQ_WAVE_CYCLES` | 1.4846e10 | /(GRBM_COUNT x 40) = 15.68, confirms the metric |
| `SQ_INSTS_VALU` | 195.4 M | 140 per warp per stage |
| `SQ_INSTS_LDS` | 72.4 M | 48 per warp per stage |
| `SQ_INSTS_SALU` | 10.7 M | |
| `MemUnitBusy` | 30.8% | memory pipe not saturated |
| `L2CacheHit` | 77.5% | |
| `LDSBankConflict` | 0 | confirms the swizzle |

VALU outnumbers LDS 2.7:1 and WMMA 4.4:1, so the issue stream is dominated by the
decode and addressing, not the matrix work. Issue is only ~15% occupied, so the
kernel is stall-bound.

### Occupancy is gated by LDS -- and raising it does not pay

`max_waves_per_simd` is 16 and a 1024-thread block is 32 waves, so one block
fills half the wave slots; a second block would fill them. LDS is 64 KiB per CU
and the staging arrays are the whole 64 KiB, so only one block is ever resident.
Measured across tiles and staging sizes:

| config | LDS | VGPR | waves/CU | occ% | ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256x256 BK4 1024t (production) | 65536 | 192 | 15.69 | 48.6 | 9.209 |
| 256x256 BK4 no staging | 0 | 104 | 15.66 | 48.8 | 7.331 |
| 256x256 BK2 1024t | 32768 | 152 | 21.46 | 66.9 | 10.894 |
| 256x256 BK2 no staging | 0 | 96 | 31.39 | 98.2 | 7.563 |
| 128x128 BK4 512t | 32768 | 112 | 29.25 | 90.1 | 10.758 |
| 128x128 BK4 512t no staging | 0 | 72 | 33.56 | 105.0 | 7.908 |

Occupancy does track LDS: halving the staging takes it from 49% to 67-90%, and
removing it takes it to 98-105%. **But every configuration with higher occupancy
is slower.** BK=2 buys 18 points of occupancy and loses 18% of wall clock; the
128x128 512-thread tile buys 41 points and loses 17%. Shrinking the staging to fit
a second block also halves the arithmetic intensity, and the extra LDS traffic per
MAC costs more than the latency hiding returns.

That is also why the ablation lattice read as it did. At *equal* occupancy (48.6%
vs 48.8%) removing the staging work is worth 20% (9.209 -> 7.331), so the cost is
the work and not the occupancy. The arms that dropped the work and the arms that
dropped the array are not separable by occupancy, and the ones that raised
occupancy were never faster.

**Conclusion.** The kernel sits at a real local optimum: 64 KiB of staging is what
caps occupancy at ~50%, and 64 KiB of staging is also what buys the arithmetic
intensity that makes the 256x256 tile efficient. Nothing measured so far moves
either term without paying more on the other. Any further attempt has to break
that coupling -- sustain 256x256 reuse at BK=4 while presenting less than 32 KiB
of LDS per block -- which is what a warp-specialised or register-staged design
would have to do, and the counters now make it possible to tell whether such a
design actually works.


### Testing the occupancy axis directly, and killing it

The tile table above changes two things at once (occupancy *and* tile
efficiency), so it cannot separate them. Ablation 33554432 (GlobalA) does: it
reads the weight operand straight from a pre-blocked fp16 buffer,
`[k/16][m][16]`, so the weight tile is never staged and the LDS allocation drops
to the activation tile alone. Same tile, same BK, same 1024 threads, same VGPR
count -- only the LDS footprint and the operand's source change.

The fp16 tensor is built with the kernel's own decode, so this is checkable, and
it caught two real bugs before any timing was believed:

| attempt | check | LDS | occ% | ms |
| --- | --- | ---: | ---: | ---: |
| XOR applied to the row index | `bit_mismatch=17825791/35651584` | | | 16.3 |
| XOR removed (row is `16a+sl`) | `bit_mismatch=17825791/35651584` | | | 16.3 |
| plus canonical half order | `worst_abs=0 bit_mismatch=0` | 32768 | **72.6** | **12.32** |
| quantised reference | -- | 65536 | -- | 9.28 |

Two lessons are in that table. The fragment's tile-local row is `((wr*WRS)+i)*16
+ sl`: the `sl^(ks&3)` in the load is the *fourth* index (which K16 slot holds
the row), not part of the row. And `LoadSwizzled`'s lane-dependent half swap is
**cancelled** by the matching `StoreSwizzled` swap, so a global reader must read
canonically -- replicating only the load half reverses the two eight-half groups
for every `shift==1` lane, which is exactly half of them. That was the observed
half mismatch. It also moved the timing 16.3 -> 12.3 ms, so the wrong half order
was skewing the address pattern too: a wrong-but-plausible kernel measured 33%
off, which is the size of the effects this document is trying to resolve.

With it correct: **GlobalA halves LDS, raises occupancy from 48.6% to 72.6%, and
is 20-33% slower.** The occupancy lever works exactly as predicted and does not
pay. Every higher-occupancy configuration measured -- BK=2, 128x128, GlobalA --
is slower than the one it displaces.

### So what is actually binding

Not occupancy (just falsified directly), and not any individual pipe:
LDS reads need ~3072 cycles of the ~26000-cycle stage at 128 B/clk (12%), DRAM
traffic is ~20 GB/s of 187, issue is ~15% occupied, bank conflicts are 0, spills
are 0, and the memory pipe reads 30.8% busy.


### Not available on this part: vmem-to-LDS staging (global_load_lds)

The obvious way to remove the staging VGPR round-trip -- every staged byte is
currently written to a register and read back -- is the Ampere-style direct
global-to-LDS copy. It is not available on gfx1151. Probe:

| arch | \`__builtin_amdgcn_global_load_lds\` |
| --- | --- |
| gfx942 (CDNA3) | **selects**, \`global_load_lds\` in the ISA |
| gfx90a | compiles, lowered to a regular load plus \`ds_write\` |
| gfx1100 (RDNA3) | \`Cannot select\` |
| **gfx1151** | \`Cannot select\`; assembler: "instruction not supported on this GPU" |

The subtarget feature \`vmem-to-lds-load-insts\` *is* listed for gfx1151 and the
builtin's feature gate passes under
\`__attribute__((target("vmem-to-lds-load-insts")))\`, but none of the size
encodings (1, 2, 4) select and no \`lds\`-modifier form of \`global_load\`
assembles. The instruction is CDNA3-only in this toolchain, so the staging
register round-trip is not removable on Strix Halo. That closes the last
instruction-stream lever on the staging path.

The design is at the **traffic-optimal point for the LDS budget**: staging cost
per MAC is `(BM*WN + BN*WM) / (BM*BN*BK*16)`, and under the constraint
`(BM+BN)*BK*32 <= 65536` the shipped 256x256 BK4 maximises the denominator for

## The model-level view: the kernel is not the problem

Everything above is one GEMM. To find the model-level gap, a real 2048-token
prefill was traced per dispatch. `--kernel-trace` records no durations on this
part (`Start_Timestamp == End_Timestamp` on all 1625 rows) and its timestamps are
*enqueue* times, not execution times -- 1602 of 1625 consecutive deltas are under
0.1 ms and the only two large ones are the weight-upload buffers -- so kernel time
has to come from `--pmc GRBM_COUNT`, one counter per run, aggregated per
dispatch.

Per-kernel GPU cycles for one 2048-token prefill (1625 dispatches):

| category | n | ms at 2.8 GHz | share |
| --- | ---: | ---: | ---: |
| batched GEMM (`HalfPrefillGemmKernel`) | 487 | 2603.5 | **86.8%** |
| GEMV (the single decode step) | 194 | 45.9 | 1.5% |
| SSM / DeltaNet | 336 | 153.4 | 5.1% |
| norm | 338 | 98.5 | 3.3% |
| attention / RoPE | 96 | 61.7 | 2.1% |
| residual add | 128 | 28.3 | 0.9% |
| setup / convert | 26 | 4.2 | 0.1% |

So the batched GEMM is 87% of GPU time. The kernel this document has been tuning
**is** the prefill; there is no large unexamined kernel.

### The GEMM is already at 79% of its clock-scaled ceiling

Analytic prefill work: 24.242 B linear parameters (embed is a lookup and
`output.weight` applies to the last token only), so 2 x 2048 x 24.242e9 = **99.3
TFLOP** for the 2048-token pass.

**The clock assumption was wrong and it matters.** The GEMM's 2603.5 ms was
converted at 2.8 GHz; the measured clock during a real prefill is **1979 MHz**
(mean; max 2630, min 1692, 93.1 W, Tctl 94.2 C). At the real clock the GEMM takes
3684 ms and delivers **26.9 TFLOPS**, against a clock-scaled ceiling of
48.35 x 1979/2809 = 34.1 TFLOPS -- **79%, the same fraction the bench kernel
reaches.** The kernel's efficiency is clock-invariant and it is already near its
limit.

The same correction removes an apparent finding: summed GPU cycles looked like
3000 ms of a 3650 ms wall, i.e. 18% idle. At the measured 1979 MHz those cycles
are 4245 ms against a 4403 ms wall -- **96% GPU-busy**. There is no idle to
recover; the 18% was the clock assumption.

### What that leaves

**Correction, and it is the one already recorded above.** An earlier draft of
this section read the 1979 MHz mean against the 2630 MHz max as a machine-level
lever to be recovered with cooling or a power profile. That is the naive reading

## How clock management actually behaves on this part

Measured rather than inferred, because the earlier "the sampler says 1979 MHz
mean" reading conflated gated and loaded time.

**The DPM level's frequency is continuously recomputed, and the top level is unused
under load.** `pp_dpm_sclk` exposes three levels and the middle one carries the
live operating point:

```
idle:    0: 600Mhz   1: 966Mhz *   2: 2900Mhz
loaded:  0: 600Mhz   1: 1755Mhz *  2: 2900Mhz
```

Level 1 read 774, 966, 1727, 1744 and 1755 MHz in different samples -- it is not a
table entry but an SMU-computed operating point. **Level 2 at 2900 MHz is never
selected under sustained load.**

**Under sustained dense load the clock is ~1.7 GHz, tightly distributed.** A
3000-iteration GEMM with SCLK sampled at 100 Hz and filtered by
`gpu_busy_percent`:

| | |
| --- | ---: |
| `gpu_busy_percent` mean | 94.9 |
| SCLK mean / max / min | **1691 / 1865 / 1593 MHz** |
| samples above 95% busy | 1706 of 1978, mean 1693 MHz |
| `power1_average` (PPT) | 85.9 W |
| `temp1_input` (edge) | 94.2 C |
| chassis fans | 6600 / 6800 RPM, `pwm1_enable` = 2 (auto) |

The histogram is a single tight mode at 1600-1800 MHz; there is not one sample
anywhere near 2600-2900. So the 2.2-2.8 GHz readings seen in mixed workloads come
from *lighter* phases, and the sustained heavy operating point is ~1.7 GHz.

**What that means for the two competing readings.** Both are true and they
reconcile:

- *Clock falls as load rises* -- confirmed directly. A sustained dense GEMM sits
  at 1.7 GHz; a mixed prefill averages 2.0-2.2; light phases reach 2.8.
- The load is pressing against an **envelope of roughly (86 W PPT, 94 C edge)**,
  with the enclosure fans already at 6.6-6.8k RPM in automatic mode. That is why
  raising the peak power cap from 80 W to 130 W bought only +12% clock (recorded
  above): the peak cap was never the binding term.

**The knobs, and who owns them:**

| knob | path | access | status |
| --- | --- | --- | --- |
| SCLK overdrive, range 600-2900 MHz | `pp_od_clk_voltage` | root only | not tested |
| DPM performance level (`auto`/`high`) | `power_dpm_force_performance_level` | root only | not tested |
| fan curve | chassis EC hwmon8, `pwm1_enable=2` | root only | auto, 6.6-6.8k RPM |
| work per clock in the kernel | code | this document | 0.01269 vs 0.0144 plateau |

`thermal_throttling_logging` reports "enabled, with interval 60 seconds" and
records no events, so it does not confirm or deny throttling; there is no
`power1_cap` or `freq1_max` exposed in hwmon, and `rocm-smi --showmetrics`
cannot parse this device's metrics version, so the SMU's own throttle bitmask is
not readable without more work than it is worth.

**The thermal experiment has been run.** Maximum fan speed does not move the
sustained operating point materially. Together with the 80 W -> 130 W sweep showing
only +12% clock, that closes both cooling and the power cap as clock levers: the
~1.7 GHz sustained point under heavy load is an SMU/power-management behaviour for
this load, not something the enclosure can be made to give back.

**The code lever is unchanged and is the only one available from here.** Work per
clock is 0.01269 TF/MHz in the real prefill against a demonstrated 0.0144 plateau --
about 12% of avoidable work per clock. Because clock is a function of load, that
gain is paid twice: the same work in fewer instructions and fewer toggled bits
raises the work per clock *and* lowers the power draw that is holding the clock at
1.7 GHz.

and it is wrong. The section "clock falls as load rises" earlier in this document
has the mechanism: the clock mean sits below the max *because the workload draws
power*, the DPM pulls the clock back, and it is not power-limited (93 W of 130 W)
nor thermally tripped. So the clock deficit is not exogenous -- it is a
consequence of the kernel's own load, and the honest efficiency measure is work
per clock, which survives a session change.

On that measure, from the bracketed prefill window:

| measure | value |
| --- | --- |
| prefill, this session | 0.01269 TF/MHz (25.12 TF at ~1979 MHz) |
| recorded plateau for this engine | 0.0144 TF/MHz |

**The recoverable gap is ~12% in work per clock, not 25-40% in clock.** And it is
rewarded twice: a kernel that toggles fewer bits and issues fewer instructions per
FLOP raises work per clock *and* lets the DPM hold a higher clock, so the
wall-clock gain is larger than the TF/MHz gain alone.

One discrepancy is left open and is now the measurement to make: the bench GEMM at
~9.2 ms implies ~2750 MHz if work per clock is still the recorded 0.0144, while
the sampler read ~1930 MHz mean over the same process. Either the sampler is
reading low for a 400-iteration bench or work per clock has moved. Settling it
needs the bench bracketed the way `clockrun.py` brackets the engine -- SCLK and
power sampled inside the timed window -- rather than over the whole process.


## The ceiling itself was not measurable, and that invalidated the efficiency claims

Every "X% of the fp16 WMMA ceiling" in this document rests on 24.17 TMAC/s
(48.35 TFLOPS). That number was a **hardcoded printf in `dot_peak`**, carried from
an earlier session -- the tool measured only `sdot4_i8`, `sdot8_i4`,
`fdot2_f16` and `pk_fma_f16`, and had no WMMA kernel at all. So no claim of the
form "the kernel is at N% of the ceiling" could be checked, and each was in fact a
cross-session comparison.

A `wmma_f16` peak kernel was added (eight independent accumulator chains, all
eight read back, or the compiler deletes six and the rate reads ~4x high; MACs
counted **per warp**, since one WMMA is a wave operation computing 4096 MACs in
total, not per lane -- counting per thread reads 32x high). Measured in the same
session as the kernels it is meant to bound:

| | TMAC/s | TFLOPS |
| --- | ---: | ---: |
| `wmma_f16` blocks=40 | 27.73 | 55.46 |
| `wmma_f16` blocks=160 | 26.39 | 52.78 |
| `wmma_f16` blocks=640 | 26.94 | 53.87 |
| hardcoded reference | 24.17 | 48.35 |

**The ceiling is 10-13% higher than the recorded value.** That is the same
magnitude as every effect this document has tried to resolve, so the drift was
large enough to manufacture or hide results: the same kernel reads 74% against a
current-session ceiling and 82% against the recorded one, and a "the ceiling is
fixed at 48.35" assumption silently bakes in a session's operating point.

Two consequences, both methodological:

- **The ceiling must be measured in the same session as the kernel it bounds.**
  `dot_peak` now measures it, so this is checkable rather than assumed.
- **Clock-normalised comparisons across workloads are also unsafe.** The bench GEMM
  at a sampled ~1.7 GHz appeared to deliver 82% of a ceiling measured at a higher
  clock, i.e. to beat a clock-scaled bound, which is impossible. Either the
  `freq1_input` sampler is not comparable across workloads or the clock at which
  the reference was taken is unknown; either way, the quantity that is safe is the
  **same-session ratio of measured rates**, not rate divided by a sampled clock.

This is the concrete answer to "we need to profile what costs what better": the
instrument that decided what costs what was reporting a number from a previous
session.

| lever | size | evidence |
| --- | --- | --- |
| GEMM at 77.9% of the same-session fp16 WMMA peak | ~22% | paired protocol, ~1% reproducible |
| non-GEMM kernels (norms, SSM, attention, residual) | 11.6% of GPU time | 396 ms, unexamined |
| decode GEMV | 1.5% of GPU time | off-target |
| operating clock under sustained load | not recoverable | max fans and the 80->130 W sweep both fail to move it |

that numerator. Shrinking LDS to fit a second block necessarily lowers that ratio,
which is why every occupancy win is a traffic loss and every traffic loss is
bigger than the occupancy win. Moving an operand to global instead makes the
traffic loss worse still, by the operand's reuse factor.

**What remains unexplained is the gap between traffic-optimal and the ceiling.**
The kernel reaches 77.9% of a same-session fp16 WMMA peak; none of the counters
above, nor any of the ~25 interventions in this document, accounts for the
remaining 22%. Answering that needs stall attribution, which gfx1151 does not
expose, or a structural change that alters the instruction stream rather than its
resources.

### Efficiency, measured by the protocol that already existed

The numbers in this section were first taken wrongly, and the error is worth
recording because it produced a false conclusion about the whole line of work.

`methodology.md` already specifies the controlled protocol: warm **every** arm,
take **one** repetition per rep (`-r 1`), alternate which arm runs first, put a
fixed **15 s gap before every timed run**, record SCLK and package power, and
report work per clock. Its point is explicit: "within a multi-repetition run the
first repetition is the boosted one, so an average mixes regimes. Pairing removes
the drift; averaging does not."

The first attempt at this section violated it -- `one:q4k ... 30` averaged thirty
launches back to back with no gap, and the peak kernel averaged five. Both mixed
the boosted regime with the settled one. Re-run under the protocol, pair by pair,
the same two arms are stable to well under a percent:

| rep | order | peak TMAC/s | GEMM TMAC/s | efficiency |
| ---: | --- | ---: | ---: | ---: |
| 0 | peak > gemm | 26.06 | 19.89 | 76.3% |
| 1 | gemm > peak | 26.03 | 20.20 | 77.6% |
| 2 | peak > gemm | 26.00 | 20.32 | 78.2% |
| 3 | gemm > peak | 25.99 | 20.31 | 78.1% |

**Peak spreads 0.13%, GEMM spreads 1.1%, and the efficiency is 77.9%** -- against
the ~10% the same pair showed when averaged. So the "34% spread between processes"
recorded earlier in this section, and the conclusion drawn from it that only large
structural effects are measurable on this box, were **artifacts of the protocol
violation, not properties of the machine.** A 1-2% effect is resolvable here; the
instrument was being used wrong.

Two corrections follow from the same measurement:

- **The GEMM is at ~78% of the same-session fp16 WMMA peak, not 59%.** That agrees
  with the ~80% recorded for the best kernel in earlier sessions, so the original
  efficiency reading was right and the 59% was wrong.
- **The stale ceiling is real but small.** The in-session peak is 25.99-26.06
  TMAC/s against the hardcoded 24.17 -- about 8% high, not the 10-13% estimated
  from averaged runs. It is still worth measuring in-session, but it was not the
  source of a large error.

One caveat on the SCLK column, which this harness does not yet report usefully: the
sampler spans the whole process, and a `-r 1` GEMM run is ~9 ms of kernel inside a
process dominated by the 89 MB weight fill, so the recorded clock is the fill's, not
the kernel's (the runs above show 810-1419 MHz and 15-19 W for the GEMM against
1755-1800 MHz and 37-43 W for the peak). Reporting work per clock for a short timed
run needs the sampler bracketed on the timed launch, not the process.

**The decomposition is left unreconciled.** The phase clock puts the K loop at 68-70%
of block cycles at ~95% of its own limit, and `0.68 x 0.95 = 0.65` was previously
used to "explain" a 59-65% whole-kernel efficiency. With the whole kernel now
measured at 78%, that arithmetic does not hold and the two decompositions have not

## Why the parameter search returned null: the parameters were already at the family optimum

This is a derivation, not a measurement, and it explains the run of null results
better than "the instrument was noisy" does.

Per warp per K step the kernel issues `WRS` A-fragment loads and `WTS`
B-fragment loads (each a `ds_read_b128` pair) feeding `WRS*WTS` WMMAs. So

```
LDS operand loads per WMMA = 1/WRS + 1/WTS
WRS*WTS = (BM*BN) / (256 * numWarps)   -- the accumulator fragments per warp
```

For BM=BN=256, 32 warps: `WRS*WTS = 8`. Minimising `1/WRS + 1/WTS` subject to
that product gives WRS = WTS = 2.83; the integer optima are (2,4) and (4,2), both
**0.75 loads per WMMA** against an ideal 0.707 -- a 6% gap. The shipped assignment
(WM=8, WN=4 -> WRS=2, WTS=4) is already at that integer optimum.

The tile cannot grow either. `(BM+BN)*BK*32 <= 64 KiB` caps `BM+BN <= 512`, and
area is maximised at 256x256. Registers would allow 384x384 (144 VGPRs of
accumulator) but its staging is 96 KiB and does not fit.

So the reachable parameter space around this design has a **6% ceiling on the
quantity most of the interventions were trying to move**. Twenty-five parameter
changes returning null is not a measurement failure; it is what an exhausted local
optimum looks like. Better measurement would not have found a win there.

**The corollary is what to do instead.** A win now has to change the *family*, and
the families are few:

| family | operand loads per WMMA | ceiling | status |
| --- | --- | --- | --- |
| fp16 WMMA + LDS staging, 256x256 | 0.75 (integer optimum) | 26.0 TMAC/s | shipped, at the family bound |
| same, 384x384 | 0.47 | -- | **does not fit** (96 KiB staging) |
| int4 WMMA 16x16x32 | halves it (8192 MACs/instr) | 52.6 TMAC/s recorded | **not on this part**: the wide shape is gfx12-only; gfx1151 has only 16x16x16 iu4, the same 4096 MACs as fp16 |
| packed int4 dots (`v_dot8_i32_i4`) | per-lane, no wave staging | 28.99 TMAC/s | **untried**, no weight-format change |

The packed-dot family is the only one that is neither already-shipped nor blocked
by a quality decision, and `dot_peak`'s own header has flagged it since it was
written: "Packed dots are a different execution class and may issue more MACs per
cycle, in which case a GEMM built on them beats the current kernel without touching
the weight format." On the paired protocol it measures 28.99 against fp16 WMMA's
26.0 -- **11% above the current instruction class, in-session.** Whether a GEMM
built on it keeps that advantage is the open question, because a per-lane dot has
no wave-level operand sharing and the data movement is different in kind.

been brought together.

### Under the protocol: the decode is 16%, and materialising it is a loss

Re-measured with warm-up, `-r 1`, a 15 s gap before every timed run and
alternating order:

| pair | reps | median |
| --- | --- | ---: |
| full (16) -> decode removed, value kept live by asm (80) | -15.3, -16.6, -16.1, -16.0 | **-16.07%** |
| quantised -> materialised fp16 weights | +7.7, +5.7, +5.5, +2.3 | **+5.61%** |
| fp32 decode -> fp16 FMA decode (first attempt) | +4.54 | +4.54% |

Three things follow.

- **The decode is worth 16%, reproducibly.** The earlier 23% came from the
  averaged-regime protocol and was inflated; 16% is the paired number, and its
  spread across four reps is 1.3 points.
- **Materialising the weights is a loss, and this is now settled rather than
  marginal.** It removes the 16% decode and adds ~22% in weight bytes (fp16 is
  2 B/element against 0.5625 for this artifact's IQ4_XS-class FFN), for a net
  **+5.6%**. The earlier "wash" was the averaged protocol hiding a real regression.
- **The first fp16-decode arm was my bug, not a result.** It routed through
  `__builtin_fmaf` on floats, which is an fp32 FMA plus two extra converts, and it
  measured *slower* than the fp32 chain it was supposed to beat. It is rewritten as
  native `_Float16` arithmetic left to contract, and re-measured below.

So the useful finding is narrow and precise: **the decode's 16% is real and
irreducible per element** (`A*q - B` has to be applied per element, and folding it
into the accumulator would need a partial sum per 32-element block). The only
question left is whether the same work can be done in fewer instructions, which is
what the native-fp16 arm tests.


### The decode is a closed question: 16%, and unreachable three ways

With the native-fp16 arm fixed and the wide tile measured, all three routes to the
decode's 16% are closed, each by a paired measurement:

| intervention | what it changes | reps | median |
| --- | --- | --- | ---: |
| decode removed, value kept live by asm | the whole decode | -15.3 -16.6 -16.1 -16.0 | **-16.07%** |
| materialised fp16 weights | decode -> bytes | +7.7 +5.7 +5.5 +2.3 | **+5.61%** |
| native fp16 FMA decode | halves its instruction count | -0.4 -0.6 +0.2 -0.2 | **-0.28%** |
| wide 128x384 tile | amortises it (total decode is 1/BN) | +14.9 +12.3 +13.2 +16.4 | **+14.08%** |

Read together these say something sharper than "the decode is expensive":

- **Removing it is worth 16%**, so it is genuinely the cost.
- **Halving its instructions is worth nothing** (-0.28%, inside the noise over four
  reps). So the 16% is not instruction count.
- **Amortising it over a wider tile is worse by 14%**, even though the total decode
  work really does fall by a third (it is proportional to 1/BN, and 128+384 = 512 is
  exactly the LDS budget at BK=4). The tile's extra store and read traffic costs
  more than the amortisation saves.
- **Precomputing it is worse by 5.6%**, because fp16 is 3.5x the bytes of this
  artifact's FFN formats.

So the 16% is a **scheduling cost, not an instruction cost**: the decode's live
ranges and its interleaving with the K loop are what the matrix pipe waits on, not
the arithmetic. Since it cannot be made cheaper, amortised, or precomputed, that 16%
is structurally locked in for a quantized weight format on this part.

One measurement note: the fp16-decode pair's rep 0 read 17.98 ms against 8.9-9.0 for
every other rep, a 2x outlier. The median is used rather than the mean, and the
other three reps agree to 0.4%. A mean over the four would have reported -3.5% and
been wrong.

## The cost of scheduling, measured: a register budget that is already spent

The decode closure ended with "the 16% is a scheduling cost, not an instruction
cost", which is a label rather than a mechanism. Pricing that scheduling directly
gives a resource ceiling, not a placement error.

### Moving work between phases is the most expensive thing that can be done

If the commit phase is a post-barrier bubble the K loop could absorb, then moving
its work into the K loop should recover part of it. It does the opposite, by up to
6.3x. Paired arms, `one:q4k 17408 5120 2048 1 0 store <ablate>`, four interleaved
cycles with a 15 s gap before every run, and the baseline arm measured both first
and last in each cycle:

| ablate | what moves | ms (4 reps) | spread | vs baseline |
| --- | --- | ---: | ---: | ---: |
| 16 | nothing (baseline) | 8.91-9.07 | 1.8% | -- |
| 272 | global weight fetch into the K loop | 12.37-12.53 | 1.3% | **+38.0%** |
| 528 | weight decode into the K loop | 18.48-18.77 | 1.6% | **+106.4%** |
| 1040 | decode staggered per warp | 57.04-57.22 | 0.3% | **+533.0%** |
| 16 | (measured straight after 1040) | 8.89-8.99 | 1.1% | -1.2% |

The arms are deterministic to 0.3-1.8% over four cycles, and the baseline taken
immediately after the 57 ms arm returns to 8.9 ms, so nothing is leaking between
runs. An earlier attempt at this pair was discarded as instrument error when the
baselines came back at 18-22 ms; on re-measurement with the clock bracketed the
same arms reproduce to 0.3%, so the discard was the mistake and these are the
numbers.

### The mechanism is register spilling, and it is monotone in the spill count

Read out of the linked code object (`llvm-objcopy --dump-section .hip_fatbin`,
`clang-offload-bundler --unbundle`, then `llvm-readelf --notes`), with the
static scratch instruction count and its span over the function body from
`llvm-objdump -d`:

| ablate | ms | VGPR | private seg | spilled VGPR | static scratch insns |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 9.03 | 192 | 0 B | **0** | 0 |
| 272 | 12.46 | 192 | 68 B | **16** | 31 (span 36% of body) |
| 528 | 18.63 | 192 | 148 B | **74** | 65 (span 43%) |
| 1040 | 57.15 | 192 | 648 B | **230** | 168 (span 49%) |

`vgpr_count` is pinned at exactly 192 in all four: the allocator does not exceed
the budget, it spills to scratch, and the scratch instructions are spread across
36-49% of the body rather than confined to a cold prologue, so they execute in the
stage loop. The cost is monotone in spilled registers, 0 -> 16 -> 74 -> 230.

### Why 192 is a hard budget, and why it cannot be bought

The cap is a property of the workgroup size, not of an occupancy target, and that
is testable without running anything. Setting the second `__launch_bounds__`
argument explicitly to 1 leaves the baseline at **192 VGPR / 0 spills**; setting
it to 2, which under a budget of RF/(threads x blocks) would force 96, leaves it
at **192 / 0 spills** as well. The allocator is therefore not trading occupancy
against registers here: a 1024-thread workgroup on this part is simply capped at
192, i.e. 196608 VGPRs divided by 1024 threads. Halving the workgroup halves the
demand and the cap lifts to the 256 architectural maximum, which is why the
512-thread arm of the same tile uses 242 and the 256-thread arm 221. This is the
same constraint the occupancy section found from the other side -- "one block per
CU set by 64 KiB of LDS, 8 waves per SIMD" -- read numerically: LDS and VGPR are
saturated by the **same** block, so neither can be relieved without the other, and
no reordering of the staging pipeline exists that does not need more
simultaneously-live values than 192.

### The two levers that follow are both dead

**Chain count is not a limiter, so there is no dependent-latency term.** The
`dot_peak` chain sweep, re-run per process with the 15 s gap, shows every chain
count from 1 to 16 reaching **27.55-27.72 TMAC/s** in at least one cycle, and every
count also dropping to 24.5, sometimes 22.0, in others. The dips move between
cycles and are not attached to any chain count, so they are an uncontrolled clock
state, not a chain effect -- and the earlier in-process sweep that read as "chain
8 is 12% slow" was exactly that artefact. A single accumulator chain already
reaches 27.57 TMAC/s, so the GEMM's eight chains (WRS=2 x WTS=4) cover the WMMA
latency many times over. This also moves the ceiling the GEMM is scored against:
the recorded 25.99-26.06 was an eight-chain in-process reading and sits between the
two observed states, while the reproducible maximum is **27.6**, which puts the
GEMM's 20.32 TMAC/s at **74%** of ceiling. Against the low state (24.5) the same
kernel is 83%. The ceiling being two-state is itself the largest measurement
caveat in this document.

**LDS operand reads are not a limiter either.** The 256x256 tile split 4x4 over 512
threads (WRS=WTS=4) reads 0.50 fragments per WMMA against the shipped 0.75, 33%
fewer -- statically 64 `ds_load_b128` per 64 WMMA against 48 per 32 -- at 242 VGPR
with zero spills and 14 registers of headroom. Paired against the shipped arm, six
reps with alternating order and 15 s gaps: **-0.56% median** (all: -0.51 -0.08
+0.53 -0.87 -0.62 -1.88), a wash. Going the other way under the same protocol,
128x384 (0.833) is **+13.3%** and 128x128 w4n4 (1.0) is **+15.7%**. So loads/WMMA
is a threshold at 0.75 and not the critical path: halving it buys nothing, raising
it costs.

### Where the deficit actually is

The commit decomposition, paired against the baseline on one basis, three cycles,
15 s gaps:

| arm | ms | vs baseline |
| --- | ---: | ---: |
| 16 full commit | 8.97 | -- |
| 80 decode removed, LDS store kept | 7.49 | **-16.5%** |
| 48 decode kept live, LDS store removed | 7.92 | **-11.7%** |
| 2 no commit at all | 6.90 | **-23.1%** |
| 4 no global fetch | 7.84 | **-12.6%** |

The two terms of the commit overlap rather than adding (16.5 + 11.7 > 23.1): each
one alone already recovers most of the other's slack, which is what a shared
resource looks like. The decode is the larger term at 16.5% of wall time while
being worth only ~7% of block cycles in the phase clock, and halving its
instruction count is worth -0.28% -- so its cost is not the arithmetic but the
occupancy of the one pipeline stage the commit can run in, at a register budget
that forbids overlapping it with the K loop.

### Wave quantization is not part of the deficit either

The real grid for this shape is 8 x 68 = 544 blocks, and if the machine ran a fixed
number of blocks concurrently the 14th round would be 40% empty, which is where a
~3% tail would live. It is not there. Holding the tile, K and batch fixed and
varying only the block count -- 520, 528 and 560 blocks (m = 16640, 16896, 17920;
five cycles, 15 s gaps) -- gives medians of 8.31, 8.52 and 8.92 ms, ratios of
**1.000 : 1.025 : 1.074**. Scaling with block count predicts 1.000 : 1.015 : 1.077;
a 40-block wave predicts 1.000 : **1.077** : 1.077; an 80-block wave predicts
1.000 : 1.000 : 1.000. The 528-block point sits closer to the no-quantization value
than to either quantized one, and its within-arm spread is 0.96%, so throughput is
proportional to block count and the grid is not tail-limited. That removes the last
scheduling term outside the kernel.

**Falsifier:** a configuration that keeps this staging structure and beats 20.3
TMAC/s moves the cost off the commit phase. Every arm that changes the staging
structure has landed at parity or worse -- activations from global, weights from
global, double buffering in three forms, BK=2, the 512-thread split -- which is
what the register budget predicts. The one structure not yet tried is removing the
LDS round trip for the weights entirely by keeping the decoded fragments in
registers across the barrier, which needs the accumulator footprint to fall enough
to pay for it; 192 - 64 (accumulators) = 128 registers is the space that makes it
possible or not.


