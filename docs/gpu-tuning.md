# GPU tuning under a narrow scope

The current engine already does most of the *scheduling* fusion: fused RMSNorm + quantize, a dual gate/up GEMM with the SwiGLU epilogue, SwiGLU folded into the quantize, residual adds folded into the next norm, and a template instantiation per weight type with tail-free variants when the shapes are tile-aligned. The narrow scope does not add more of the same; it changes the *kind* of tuning that is possible, from runtime heuristics to a build-time search over a tiny, known shape set.

## What specialization buys

- **Compile-time shapes.** A general engine passes `m`, `k` and `batch` as runtime integers and carries bounds checks for whatever it is handed. A fixed model has `M`, `N`, `K` per projection as constants: the grid is static, the K loop unrolls fully, and the tail-free kernel is always the correct one. This is codegen headroom, not a new algorithm.
- **Offline autotuning.** With a handful of `(type, M, N, K)` tuples the tile configuration can be searched exhaustively at build time -- `BM`/`BN`/`BK`, the pipeline depth, the row-group shift, the wave assignment -- and the winners baked into a table. The runtime cost of the search disappears, and the search space is small enough to be exhaustive rather than heuristic. The current engine's `DirectGemm` already selects among three tile shapes by hand-tuned rules; this replaces the rules with measurement.
- **Wave-exact grids.** The iGPU has a fixed CU count, so a decomposition that is an exact multiple of the wave size wastes nothing on a partial final wave. With fixed shapes that decomposition is chosen once.
- **Kernel selection by measurement, persisted.** Each candidate is run once at build time and the winner recorded, so the shipped engine never dispatches speculatively.

## Fusion for a fixed graph

A general engine must keep a fallback for every fusion it performs. A fixed graph can assume all of them:

- **Attention**: QKV projection, RoPE, the score matrix, softmax and the value product, without materializing the score matrix at all. In a long-context or image prefill the score matrix is the largest transient in the layer, so this is a memory win as much as a time win -- directly relevant to the minimal-memory goal.
- **SSM / recurrent layers**: a fused chunked scan that never materializes the per-token intermediates.
- **The residual chain**: norm and residual folded across the whole layer rather than per operation.

## Formats

The engine currently moves K-quant weights through int8 WMMA with per-block scales. If the target supports a wider matrix-core path (fp8/bf8 on the same instruction, or a lower-precision accumulator), that doubles the arithmetic per instruction but changes numerics, so it has to be validated the same way the NPU path was. Because the scope is narrow, a second format can be carried for the one or two projections where it wins, rather than everywhere.

A second option the narrow scope permits is to *choose the quant family* instead of accepting a mixed shard. A model in one or two formats needs one or two kernel instantiations instead of a route table over many, which both shrinks the tuning surface and removes the mixed-pair specialisations. This changes the model artifact, so it is a trade to make deliberately.

## Launch overhead and graph capture

A chunk is on the order of 3,600 dispatches. At a few microseconds each that is single-digit milliseconds, and the 3-deep pacing hides most of it while the GPU is saturated. A captured and replayed graph removes it entirely and, more usefully, makes the overlap explicit rather than dependent on the host staying ahead. With a fixed graph this is a build-time artifact.

## What tuning does not fix

The GPU stream is already ~96% busy in the column split, so there is no idle to reclaim on the GPU side. The token split wins because it *removes* GPU work, not because the GPU was idle. GPU tuning therefore raises efficiency (TFLOPS per shape) rather than occupancy, and its ceiling is the hardware peak that `tools/bench/gfx1151_peak.hip` measures.

## Interaction with the split

Tuning the GPU moves the split balance. Both engines currently sustain about the same ~19 TFLOPS on their FFN halves, which is why the optimum sits at 50/50. A faster GPU should take more of the split, so `M` has to be re-tuned after GPU tuning, not before. It also raises the GPU-only baseline, which shrinks the NPU's *relative* contribution even as the absolute chunk time falls -- both are worth having, but the two wins are not additive.

## Order of work

1. Close the all-NPU numerics item.
2. Measure the GPU-only baseline and the gfx1151 peak. Without both, neither the split nor the tuning has a known ceiling.
3. Autotune and constant-fold the GPU kernels for the fixed shapes; re-measure.
4. Re-tune `M` against the faster GPU.
5. Only then the int4/int8 B NPU kernel, which is what raises the split's ceiling past 2.0x.
