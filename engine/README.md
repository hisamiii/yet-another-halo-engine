# engine/

The greenfield implementation. The layout maps to the workstreams in [../docs/scope.md](../docs/scope.md).

| dir | contents | status |
| --- | --- | --- |
| `core/` | GGUF reader, mmap, tensor table, config, tokenizer | reader/config/**tokenizer done** |
| `model/` | Qwen3.8 27B graph: Gated DeltaNet, attention, RoPE, norms, FFN, state | weight table done; attention, FFN and DeltaNet checked; graph pending |
| `gpu/` | ported WMMA framework, per-type bench, fusions | **ported + benched**, correctness pending |
| `npu/` | XRT executor, xclbin set, dma-buf operands, async launch | not started |
| `sched/` | phase routing, token split, overlap, power budget | not started |
| `kv/` | paged quantized KV cache | not started |
| `vision/` | mmproj projector, image preprocessing | not started |
| `serve/` | CLI, HTTP, sampler | not started |

Milestone mapping: **M0** core + model + gpu + serve. **M1** npu + sched. **M2** npu (int8). **M2g** gpu. **M3** sched + model. **M4** kv. **M5** vision.

Rules: one binary, one architecture, hardcoded shapes. Port the framework, not the efficiency claim. Every change is gated by the top-1 validation check, not by throughput.
