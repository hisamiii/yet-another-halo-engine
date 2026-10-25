# Roadmap

What is next, and what is planned but not built. Numbers behind the leads are in [results.md](results.md).

## Next: user-facing

- A GGUF exporter: write the converted model (MQ4G256 in the rotated basis, the MTP layer kept) to a file, so a GGUF converts once instead of at every start (~22 s).
- Speculative decoding for sampled requests: today only greedy requests draft. Sampling needs the draft's distribution and rejection sampling in the verify (the MTP head's logits are already on the GPU).
- One-command setup: fetch the ROCm runtime and HRX, build, emit the default sets and download the model in one script, with clear errors for a missing piece.

## Then

- Vision: the projector (`mmproj-F16.gguf`, 0.86 GiB) and image tokens (64-16384 per image) merged after it. Gate: an image prompt gives correct output.
- Stronger drafters than the MTP layer, both published for this model (looked at 2026-10-26):
  - EAGLE-4 (`EAGLE-Corp/Qwen3.8-27B-EAGLE4`, 6.2 GB BF16): one Llama layer (hidden 5120, 40 heads x 128, MHA, FFN 16384, sliding window 4096) over the target's hidden states of layers 3, 31 and 59, and a rank-128 low-rank head that scores 128 candidates. Needs: capturing those hidden states in the verify and the prefill, a 40 x 128 attention (ours is built for 24 / 4 x 256), the drafter's weights converted to MQ4, and tree drafts (a tree verify: a mask over the T tokens in the attention kernels).
  - DFlash 2 (`z-lab/Qwen3.8-27B-DFlash2`): a block diffusion drafter, 5 non-causal Qwen3 layers (32 / 8 heads x 128, FFN 17408, window 2048) that fill a block of 8 mask tokens in one pass from the target's hidden states of layers 5, 19, 33, 47 and 61. Needs the same hidden-state capture and weight conversion, a non-causal attention for its geometry, and verify of 8 drafts (wider T-token GEMVs).
  - Both pay only if the verify gets cheaper: at 4 drafts it costs 1.45 steps today (Performance leads), and their longer accepted runs need wider verifies.

## Performance leads

- Verify cost: a 4-token verify takes 90 ms against a 62 ms step, where reading the weights once would give ~65 ms. The T-token GEMVs re-read x per wave (R 4 rows per wave took 14% off); x shared through LDS per workgroup, SwiGLU at R 4 (its two weights spill there today) and the per-token small kernels batched over the T tokens are next. Every ms there raises the speedup of every drafter.
- Prefill, partial chunks: only the GEMMs shrink to the real tokens; the norms and other small kernels still run the whole chunk. Shrinking them too changed the last token's logits by ~1e-4 when it sat alone in its 16-token group (likely the 32-token DeltaNet chunk); find that first.
- Decode, kv4a16 attention: `part_q` with kv4 is latency-bound at long context (~91 GB/s at 30.7K). Bandwidth-bound it would save about 3 ms per token at 30K.
- NPU FFN block: switching the NPU between the gate / up and the down image costs ~1.1 ms per chunk-layer (~15% of its time; ~3400 setup ops per switch). Fewer switches (both images' work for two layers per command) would let the NPU take a larger share.
- Upstream the local HRX patches (see build-and-run.md), with the owner's agreement.
