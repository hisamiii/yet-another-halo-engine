# Scope: the greenfield engine

**Decision: greenfield.** A new codebase and a new execution architecture, not a fork. The reason to go greenfield is precisely the thing the measurements exposed: the NPU is currently a bolt-on behind environment variables in a general engine, and its costs (a serial chain, a host-blocking wait, a duplicated launch, a per-layer repack) are structural. In a new engine the two engines and the power budget are the architecture, not an optimisation.

## Frozen scope

- **One architecture**: Qwen3.8 27B, dense text/image, hybrid Gated DeltaNet plus full attention, hidden 5120, intermediate 17408.
- **Two GGUFs and one projector**, all already on the machine: `Qwen3.8-27B-IQ4_XS-3.84bpw.gguf`, `Qwen3.8-27B-UD-Q4_K_S.gguf`, `mmproj-F16.gguf`. Twelve quant decoders cover both ([targets.md](targets.md)).
- **Three routing modes**: GPU-only, NPU-only, GPU+NPU.
- **Minimal memory**: mmap the GGUF as the single source of truth, quantized KV, transient NPU operands.

## What is ported, and what is not

Greenfield means new structure, not new arithmetic for its own sake, but the kernel set is **not** an untouchable asset. The distinction matters:

- **Port the framework**: the WMMA tiling, the LDS and staging structure, the decoder scaffolding, the GGUF reader and the tokenizer. Those encode measured instruction behaviour and rewriting them is pure delay.
- **Do not port the efficiency claim.** The best single kernel (paired gate/up, 256x256) measures 40.6 TFLOPS, but **the aggregate across all prefill kernels is ~30 TFLOPS** -- about 62% of the measured 48.35 instruction ceiling. That ~18-TFLOPS gap is a first-class workstream. The recorded 27.7% model-level gap against the plain-store bench is split between the IQ3_XXS decode (a third of the primary target's FFN, and the most expensive decoder measured), the non-FFN GEMMs, and shape and `Complete` choices; the batch-2048 A/B that would separate them was never run.

So the kernel workstream is: port the framework, then raise the aggregate toward the instruction ceiling with the twelve decoders and the fixed shapes. That is comparable in value to the NPU's int8 work and cheaper to reach.

Everything above the kernels is new: the execution graph, the two-engine scheduler, the NPU executor, the memory policy.

## Architecture commitments

- **Fixed shapes.** M, N and K per projection are constants; one binary per model family. No runtime shape dispatch, no route table.
- **Two async queues.** A GPU stream and an NPU context, joined by explicit events. The host never blocks on the NPU except at the point the data is actually consumed; this is what removes the measured 20.9% GPU idle.
- **Per-layer-type split.** Full-attention layers take a token prefix end to end (the NPU's tokens attend only to tokens it owns; the GPU reads the NPU's K/V from the shared cache). Gated DeltaNet layers split by head, because the recurrence is sequential in tokens.
- **Power as the resource.** The two engines share a 130 W envelope and the NPU is derated ~21% under concurrency. The split fraction is a power decision, not just a FLOP decision.

## Workstreams

| # | workstream | new or ported | rough |
| --- | --- | --- | --- |
| 1 | GGUF reader, mmap, tensor table, tokenizer | ported | 1 wk |
| 2 | GPU kernel set: port the framework, 12 decoders, paired coverage, per-type `Complete`, raise the aggregate 30 -> 40+ | ported + workstream | 3-5 wk |
| 3 | Model graph: Gated DeltaNet, attention, RoPE, norms, SwiGLU FFN, KV | new | 2-3 wk |
| 4 | NPU executor: xclbins, dma-buf operands, async launch, join | new | 2 wk |
| 5 | Scheduler: phase routing, per-layer-type split, overlap, power budget | new | 1-2 wk |
| 6 | Memory: quantized KV, single-copy weights, transient operand pool | new | 1 wk |
| 7 | int8 or int4 ATB GEMM (the only item that raises the ceiling) | new kernel | 4-8 wk |
| 8 | Vision: mmproj projector, image tokens, preprocessing | new | 2-3 wk |
| 9 | Server, sampler, CLI | ported | 1 wk |

## Milestones and gates

| milestone | content | gate |
| --- | --- | --- |
| **M0** | text runs end to end on the iGPU, correct | token-for-token match on a fixed prompt against a reference; pp2048 in the 400+ tok/s class |
| **M1** | NPU executor + FFN token split, async and overlapped | validation passes at every chunk length **including the all-NPU case**; GPU stream idle < 5%; >= 1.3x over M0 |
| **M2** (parallel from week 1) | int8 / int4 ATB GEMM | >= 40 TFLOPS at both FFN shapes, numerics pass, power inside 130 W |
| **M2g** (parallel) | GPU kernel set: aggregate efficiency on all target types | >= 38-40 TFLOPS **aggregate** at model level (from ~30), top-1 validation unchanged |
| **M3** | split the non-FFN GEMM (attention projections) | >= 1.5x over M0 |
| **M4** (parallel) | memory: quantized KV, single-copy weights | peak RSS = mmap + KV + transients, measured |
| **M5** | vision via mmproj-F16 | image prompt produces correct output; measured |

**Critical path is M0 -> M1 -> M3.** M2 runs in parallel from day one because it is the long pole and the only item that moves the ceiling; M4 also parallel; M5 is after M1 and off the critical path.

## What we deliberately do not build

- No models other than Qwen3.8 27B, no general shape support, no plugin or backend abstraction.
- No training, no quantization tooling, no model conversion.
- No speculative decoding, no MTP, no multi-user serving in phase 1 (all exist in the reference engine and are decode-side levers).
- No multi-vendor: this is gfx1151 plus XDNA2.

## Risks

1. **The all-NPU correctness failure is not understood.** It must be diagnosed, not copied: a greenfield engine that reproduces the same placement bug inherits the same silent wrong answers. This is a blocker for M1's gate.
2. **The int8 ATB GEMM is unproven** at our shapes and carries activation-quantisation quality risk. It is the only high-ceiling item, which is why it starts first.
3. **The model graph is intricate** -- Gated DeltaNet is not a plain transformer, and a subtle state-handling error is invisible until logits diverge. M0's token match is the gate.
4. **Greenfield cost.** M0 is the long pole; without the ported kernels it roughly triples.

## Rules that keep it fast

- Port the kernel framework and the GGUF/tokenizer; do not rewrite the measured instruction structure, but do not inherit the aggregate either (~30 TFLOPS is the starting point, not the ceiling).
- One binary, one architecture, hardcoded shapes.
- Iterate on the 3.84 bpw shard (13 GB, loads faster, and is the harder decoder set); gate on UD-Q4_K_S before promoting.
- Keep the measurement protocol from [methodology.md](methodology.md): warm-up, alternating arms, one repetition, pairs only.
- Every change is gated by the validate-prefill top-1 check, not by throughput.
