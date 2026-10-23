# Roadmap

What is next, and what is planned but not built. Numbers behind the leads are in [results.md](results.md).

## Next: user-facing

- Sampler: temperature / top-p sampling exists on the GPU (`gen_sample`); add the usual knobs (top-k, min-p, repetition and presence penalties, stop strings) and a seed that reproduces a run.
- Multi-turn KV reuse: keep the previous turn's KV pools and recurrent state (the prefill's slots already save and load them) and prefill only the new tokens of the next turn, instead of the whole conversation.
- An OpenAI-compatible streaming server: `yah_server` speaks the Responses API today; add `/v1/chat/completions` with SSE streaming, so common clients work unchanged.
- One-command setup: fetch the ROCm runtime and HRX, build, emit the default sets and download the model in one script, with clear errors for a missing piece.

## Then

- Vision: the projector (`mmproj-F16.gguf`, 0.86 GiB) and image tokens (64-16384 per image) merged after it. Gate: an image prompt gives correct output.
- Speculative decoding: a drafter (the model's MTP layer, `nextn`, is not run today) and a batched verify step.

## Performance leads

- Prefill, partial chunks: only the GEMMs shrink to the real tokens; the norms and other small kernels still run the whole chunk. Shrinking them too changed the last token's logits by ~1e-4 when it sat alone in its 16-token group (likely the 32-token DeltaNet chunk); find that first.
- Decode, kv4a16 attention: `part_q` with kv4 is latency-bound at long context (~91 GB/s at 30.7K). Bandwidth-bound it would save about 3 ms per token at 30K.
- Upstream the local HRX patches (see build-and-run.md), with the owner's agreement.

## Parked

- The NPU (XDNA2) prefill split: parked on branch `npu` with its HRX patches and docs. Its last state (2026-10-13, IQ4_XS model): every large GEMM split by output columns, the NPU FFN block and token halves, pp2048 ~2300 ms with the NPU as the critical path (busy ~2190 ms). It was written for the IQ4_XS model and does not run on the MQ4 model; the A4 GPU path is faster than it was.
