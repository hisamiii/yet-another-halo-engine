# engine/

The greenfield implementation. The layout maps to the workstreams in [../docs/scope.md](../docs/scope.md).

| dir | contents | status |
| --- | --- | --- |
| `core/` | GGUF reader, mmap, tensor table, config, tokenizer | reader/config/**tokenizer done** |
| `model/` | Qwen3.8 27B graph: Gated DeltaNet, attention, RoPE, norms, FFN, state | **full prefill and decode graph done, token-gated**; per-stage checks all pass |
| `gpu/` | ported WMMA framework, per-type bench, fusions | **ported, benched, correctness-checked** |
| `npu/` | XRT executor, xclbin set, dma-buf operands, async launch | not started |
| `sched/` | phase routing, token split, overlap, power budget | not started |
| `kv/` | paged quantized KV cache | not started |
| `vision/` | mmproj projector, image preprocessing | not started |
| `serve/` | CLI, HTTP, sampler | CLI done (`yah-run`); HTTP pending |

Milestone mapping: **M0** core + model + gpu + serve. **M1** npu + sched. **M2** npu (int8). **M2g** gpu. **M3** sched + model. **M4** kv. **M5** vision.

Rules: one binary, one architecture, hardcoded shapes. Port the framework, not the efficiency claim. Every change is gated by the top-1 validation check, not by throughput.

## The M0 gate

`tests/m0_gate.sh` and `tests/generate_gate.sh` are the milestone gate. The
first requires the engine's greedy next token to equal the reference engine's on
three fixed prompts; the second requires 20 greedy tokens to match the
reference's, token for token. Both pass on the IQ4_XS artifact. Prefill is
chunked and carries KV plus recurrent state across chunks; decode uses the GEMV
and single-token kernels and runs at about 14 tok/s.
