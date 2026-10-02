# Roadmap

What is next, and what is planned but not built. Numbers behind the leads are in [results.md](results.md).

## Next leads

- Decode, kv4a16 attention: `part_q` with kv4 is latency-bound (~91 GB/s, 347 us per call at 30.7K). Bandwidth-bound it would be ~160 us, about -3 ms per token at 30K. The per-workgroup q rotation, barriers and group sums are the suspects.
- Decode, dispatch overlap: re-measure the no-barrier overlap (-1.4 ms when it was added). A single uncooled round after the cleanup showed no gain.
- Speculative decoding (drafting): planned.
- Upstream the local HRX patches (no-ordering-barrier dispatch flag, stream profile metadata, counters mode), with the owner's agreement. See build-and-run.md.
- Other models of the same family: `UD-Q4_K_S` fails to emit (its q5k `ffn_up` has no `yah_ffn_gemm_q5k_swiglu_f16.loom`).

## Planned, not built

- NPU (XDNA2 through XRT): xclbin set for the frozen shapes (ATB config3 family; the `K_Problemsize` override is required, else only K = 4096 verifies), A / B / C as dma-buf imports from the iGPU allocator, async submission with the wait at the join. Blocker: the all-NPU placement failed the top-1 check at some chunk lengths (cosine ~0.95, should be > 0.99).
- GPU + NPU scheduling: prefill split across both engines, decode on the GPU. Full-attention layers split by token prefix (each engine's tokens attend only to tokens it owns; K/V in the shared cache); Gated DeltaNet layers split by head (the recurrence is sequential in tokens). Two async queues joined by events. Both engines share the 130 W budget. The NPU keeps full performance under concurrency as long as the thermal power cap is below the max power cap (the concurrency derate is then ignored). Gate: GPU stream idle < 5% with the split on.
- Vision: `mmproj-F16.gguf` (0.86 GiB) is the only projector; image tokens (64-16384 per image) merge after it. Gate: an image prompt gives correct output.
- Serving: CLI, HTTP and a sampler (greedy only today). Keep it thin.
