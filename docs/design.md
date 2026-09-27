# Design sketch

A GGUF engine scoped to text and image, routing across the iGPU and the NPU, with resident memory kept to the model plus KV cache.

## Non-goals

- Training or fine-tuning.
- Arbitrary architectures. Pick a small family (a text LLM plus a vision tower), fix the layer shapes, and bake them.
- Portability to other NPUs. The AIE shape constraints are the design, not an obstacle to abstract away.
- A general tensor-graph executor. A hand-written layer loop is enough at this scope and removes the framework overhead.

## Weight storage

The GGUF file is mmapped and is the only copy of the weights. No converted or repacked weight set is written to disk or held resident. If an engine needs a different operand layout it streams the repack per layer through a small reusable scratch buffer, which is what the current engine does for the NPU's bf16 B; the goal of the int4/int8 B kernel below is to remove even that.

## Two GEMM backends, one router

| phase | engines | rule |
| --- | --- | --- |
| text prefill | GPU + NPU | split by tokens, roughly 50/50 |
| text decode | GPU | NPU gated off below a minimum batch |
| image prefill | GPU + NPU | its own baked shapes |

Splitting by tokens rather than output columns matters: a full-width GEMM on each engine needs no gather or scatter, and the measurement above shows the gather/scatter a column split needs costs more than the wider B operand the token split pays for.

The router should be explicit about precision. The NPU path is bf16 today, and the unresolved all-NPU numerics item means the engine should carry a policy for how much of the sequence is allowed on the NPU, rather than a boolean.

## The NPU executor

- One xclbin per projection role (gate/up shares one; down has its own), each with M, K and N baked.
- A, B and C allocated as dma-buf exports from the iGPU allocator so neither engine copies an operand.
- Double-buffered, so layer L+1's operands can be encoded while layer L is still running.
- Asynchronous submission with the wait deferred until the result is actually needed. This is the change that removes the 20.9% GPU idle measured above; it needs no new kernel.
- The forward activation encode and the output decode should be folded into the surrounding GPU kernels rather than run as separate passes.

## Cross-request pipelining

Because the NPU shapes are fixed and the GPU is the flexible engine, two concurrent requests can occupy different engines: request A's NPU chain while request B runs on the GPU. This is the principal reason to write a purpose-built scheduler, and it is what returns the GPU to full occupancy without making the NPU faster.

## Memory budget

The model is 12.18 GiB and the pool is 28 GB shared with the CPU. The design should be explicit about:

- weights: 12.18 GiB, mmapped, single copy;
- KV cache: the only per-request growth, so its dtype and page size are a first-class choice;
- NPU operands: transient, on the order of 250 MB at M=1024 (A 5.6/19.1, B 95.6 x2, C 19.1/5.6 MB), reused across layers;
- vision tower: shares the same pool and its activations are large per image, so the image path is the peak-memory case.

## Milestones

1. Close the all-NPU numerics item. It currently disqualifies a mode, so it is not a tuning task.
2. Measure the GPU-only baseline, which bounds everything the NPU can be worth.
3. Build an int4/int8 B AIE GEMM with per-block scales and compare its per-layer time and operand traffic against the bf16 repack path.
4. Move the NPU wait off the critical path and re-run the timeline to confirm the GPU idle falls.
5. Measure a vision tower on this hardware before designing around the assumption that it behaves like text prefill.
