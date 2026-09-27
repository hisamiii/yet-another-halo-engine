# How much speedup is actually available

The question: if the GPU+NPU split is implemented perfectly, and the two engines match in speed, is the gain 1.5–1.8x?

**Answer: about 1.6x, but only against the GPU-only engine. Against the split engine that already exists, only about 1.2x remains.** The reason is that the split does not divide the whole model 50/50 — it divides only the FFN, and the GPU keeps all of the rest.

## The two engines do match in speed

From the in-engine numbers, both sustain roughly the same 19 TFLOPS on their respective FFN halves:

- NPU: half the FFN is 5.48e11 FLOPs per layer, and the gate/up plus down chain measures 28.5 ms per layer, so 19.2 TFLOPS.
- GPU: the token-split chunk has ~3.0 s of GPU busy time (6449 ms profiled, and the profiled span runs ~2.09x the real chunk). Subtracting the non-FFN share derived below leaves ~28.6 ms per layer for the same 5.48e11 FLOPs, so also ~19.2 TFLOPS.

So the premise holds. That is why the token-count sweep is flat around M=1024 and falls off on both sides: at M=512 the GPU becomes the long chain, at M=1536 the NPU does.

## The model

Write the per-layer GPU work as `A` (attention, SSM, norms, the parts the NPU never touches) and the full FFN as `F`. At a 50/50 split with equal engine rates `R`:

```
T_gpu_only = (A + F) / R
T_split    = max( (A + F/2) / R , (F/2) / R )
```

`A > 0` always makes the GPU side the longer chain, so `T_split = (A + F/2) / R`. With `f = F / (A + F)`, the FFN's share of the GPU-only time:

```
speedup = T_gpu_only / T_split = 1 / (1 - f/2)
```

For 1.5x you need `f >= 0.67`. For 1.8x you need `f >= 0.89`. For 2.0x you need `f = 1`, i.e. no non-FFN work at all.

## What this box measures

| quantity | value | how |
| --- | ---: | --- |
| token split | 3.72 s / 551 tok/s | measured |
| GPU busy, token split | ~3.0 s | profiled 6449 ms / 2.09 |
| NPU chain, token split | ~1.82 s | 64 x 28.5 ms |
| GPU-only (derived) | ~4.82 s / 425 tok/s | GPU busy + the NPU's half moved back |
| `f` (FFN share) | ~0.76 | `F/R = 2 x 1.82`, over 4.82 |
| current speedup vs GPU-only | **1.30x** | 4.82 / 3.72 |
| ceiling `1/(1-f/2)` | **1.61x** | 1 / 0.622 |
| perfect overlap | ~3.0 s / ~683 tok/s | `max(GPU, NPU)` |

## So

- Against a **GPU-only** engine, the FFN-only split tops out near **1.6x**, and a perfect implementation reaches it. 1.5x is attainable; 1.8x is not, unless `f` is much higher than measured.
- Against the **split engine that exists today**, the remaining headroom is the 20.9% GPU idle: about **1.2x** at most, taking 551 to roughly 660–680 tok/s.
- The 1.5–1.8x range is therefore right only for the first reading, and it sits at the optimistic end of the ceiling rather than in the middle.

## Splitting all the work, not just the FFN

The `1/(1-f/2)` bound exists only because today's split stops at the FFN. Give the NPU a token prefix for every projection and the layer is divided 50/50, so the same model gives:

```
T_split = max( (A/2 + F/2) / R , (A/2 + F/2) / R ) = (A + F) / (2R)
speedup = 2.0x  (hard bound: two engines, equal rates)
```

Two engines is the bound. Splitting more finely cannot exceed it, because every fine split still has at most two engines to place work on.

Which operations can actually join:

| operation | split | cross-engine cost |
| --- | --- | --- |
| QKV, output, FFN gate/up/down | token prefix | none; both engines write full-width rows |
| causal attention core | token prefix | the GPU's tokens need the NPU's K/V: a one-way write into the KV cache the engine already keeps, not a reduction |
| recurrent / SSM layers | not by token; the recurrence is sequential | must split by head, which needs the two partial outputs summed before the output projection |
| norms, RoPE, softmax, residuals, quantize | leave alone | memory-bound on one shared pool; moving them adds sync and no bandwidth |

So the GEMM-shaped work and the attention core both join cheaply; the SSM recurrence is the one that needs a real reduction, and the elementwise work should not be touched. The attention case is the important one, because it is what makes the split 50/50 of the layer rather than 50/50 of the FFN.

The cost is that the NPU kernel set grows from two block-GEMMs to a handful that include softmax and, for the SSM path, a scan, and the number of NPU submissions per layer multiplies. That is the custom engine in full, and it is what the `block_datatypes` examples are a starting point for.

And 2.0x assumes both engines are equally good at everything, which is false. If the NPU is much slower at softmax or at a scan, the balance shifts back toward the GPU and the achieved number lands between 1.6x and 2.0x. The practical target is therefore: give the NPU the token prefix for the GEMMs and the attention, keep the irregular ops on the GPU, and rebalance per layer type.

## What would move the ceiling

The bound comes from `f`: the NPU only ever removes FFN time while all `A` stays on the GPU. Two things break it:

1. **Offload some non-FFN work.** Attention and SSM are fixed shapes too, so they can be given to the NPU as additional xclbins. If the whole layer were split 50/50, the ceiling becomes 2.0x.
2. **Make the NPU faster per FLOP.** The ceiling above assumes the NPU matches the GPU, so the optimum stays at 50/50. If an int4/int8 B kernel made the NPU twice as fast, the optimum shifts to the NPU taking ~88% of the FFN and the ceiling rises to about **3x**, because the GPU is then left with mostly `A`.

That second one is the real prize, and it is a kernel project rather than a scheduler project. The 20.9% idle is the cheap part; the datatype is the large part.

## Caveats

- `T_gpu_only` is **derived, not measured**. The harness `npu_ceiling.sh` would measure it in one paired session and should be run before this model is trusted for planning.
- The model ignores the smaller-batch penalty the GPU pays on its half and the bf16 repack, both of which push the achieved number below the ceiling.
- The 19 TFLOPS figure is the in-engine rate, not the 32.4 TFLOPS standalone ceiling; the gap is unexplained and would raise every number here if closed.
