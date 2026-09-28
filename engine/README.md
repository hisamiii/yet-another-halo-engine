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

Weights are a single registered mapping: the GGUF's own `mmap` is registered
with HIP from its page-aligned base, so there is no second copy. Peak RSS on a
5-token run is 12.4 GiB against 24.6 GiB when the tensor region is copied,
which is the headroom a long context needs.

## Runtime: HRX

The target runtime is **HRX** (`ROCm/hrx`), an alternative HIP implementation that
also carries an XDNA/NPU HAL driver, so one runtime serves both the GPU and the NPU
paths. `source engine/hrx-env.sh` selects it for the current shell;
`engine/hrx-env.sh --check` verifies the install and prints the devices, and
`--fetch` downloads and extracts the pinned packages (~174 MB, no system install).
Nothing in `/opt/rocm` is modified, so ROCm stays usable in another shell.

**Pin the official ROCm Core SDK 10.0.0 release; never a nightly.** HRX's AMDGPU
driver calls `hsa_amd_queue_create`, which the 7.13 install and the 7.14.0.dev0
nightly wheel do not export — only the 10.0.0 release does. Judging HRX's
requirements against a nightly produced a wrong "the GPU cannot work" conclusion,
so the rule is written down rather than remembered: full account in
[../docs/hrx-evaluation.md](../docs/hrx-evaluation.md).

Validated on this box: all seven check binaries pass under HRX
(`yah-gemm-check`, `yah-ssm-check`, `yah-attention-check`, `yah-rope-check`,
`yah-unpack-check`, `yah-ffn-check`, `kv-quant-check`), and a 2048-token prefill
runs at parity with ROCm (3130 / 3138 ms against 3126 ms) with an identical greedy
argmax.

### Reaching the HRX-native API

`libhrx/include/hrx_runtime.h` exposes more than HIP: graphs with explicit
dependencies, `hrx_stream_dispatch`, timeline semaphores, fences for submission
batching, and memory pools. Reaching it is **not** a re-link —
`hrx_executable_load_*` takes a native executable package selected by
`target_family` / `target_key`, not a hipcc HSACO — so it needs kernels produced
by the Loom/MLIR/TileLang path, which is the same path the NPU work requires. HIP
graph capture *is* implemented by HRX and maps onto its native graph, but it is
worth ~0.02% of prefill: dispatch is not a lever on this part.
