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

## Kernel authoring: Loom

Loom is the compile path that HRX's native API consumes, and it is functional on
this box for the AMDGPU target. Its mandate is the thing we are short of — "the
performance of its emitted programs: outperform tuned HIP and hand-authored
assembly on AMD GPUs and AIE on XDNA" — and it targets both halves:
`LOOM_TARGET_AMDGPU=ON` and `LOOM_TARGET_XDNA=ON` are both set in our CMake
build. Loom is configured by our existing HRX build; the tools came in with it.

Tools (no extra build needed, they land with `build_gpu.sh`/`cmake --build`):

```
build/cmake/loom/src/loom/tools/{loom-compile,loom-check,loom-link}
build/cmake/loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom
build/cmake/loom/src/loom/tools/iree-run-loom/iree-run-loom
```

Authoring corpus, which already contains shapes close to ours
(`mlp_down_projection_residual_bf16.loom`, `ffn_gate_up_swiglu_q6q8.loom`):

```
cd /home/q/hrx && source engine/hrx-env.sh

# plan only, no GPU needed
iree-benchmark-loom loom/src/loom/test/corpus/authoring/mlp_down_projection_residual_bf16.loom \
  --dry-run --config=mlp_down_projection_residual_bf16.row_capacity=3584

# compile to AMDGPU and dispatch through HRX's amdgpu HAL
iree-benchmark-loom loom/src/loom/test/corpus/authoring/mlp_down_projection_residual_bf16.loom \
  --config=mlp_down_projection_residual_bf16.row_capacity=3584 --device=amdgpu \
  --measure=dispatch_complete --iterations=1 --warmup-iterations=0 --batch-size=1 \
  --min-time-ms=0 --max-batches=1 --input-ring-count=1
```

Verified on gfx1151: the dry run plans 1 case / 3 benchmarks; the GPU run lowers to
`amdgpu-rdna3-5` / snapshot `amdgpu-rdna3-5-low` / config `amdgpu.rdna3_5.core` at
subgroup 32, selects real packets (`global_load_b128_saddr`, `global_store_b32_saddr`,
`v_mul_f32`, `s_lshl_b32`, `workgroup_reduce.publication.lds`), dispatches, and
reports all 4 benchmarks `state: ok` with exit 0. It also emits a full report —
static and dynamic instruction mix, memory roots with byte envelopes, and per-source-op
lowering selections — which is the "keep source facts, target facts and performance
evidence in one system" claim, and is more visibility than the HIP path gives us.

### Why this is the right next move

The HIP surface is exhausted: dispatch, graphs, cooperative launch and persistent
kernels all measured to ~0, and the remaining kernel deficits are shape problems
that our hand-written HIP cannot express (the DeltaNet recurrence is a rank-128
matvec with a per-token cross-lane reduction, and it needs to become chunked
matmuls). Loom is a compiler whose stated purpose is beating tuned HIP, it can
express that restructuring, and the same source path targets the NPU — so the
kernel work and the NPU work stop being two projects.

### First measurement: the corpus example is not yet competitive

The cheapest test of Loom's mandate is to find a corpus example whose shape we can
reproduce on our HIP side and run both. `ffn_gate_up_swiglu_q6q8.loom` is a q6_K
weight x q8_0 activation gate+up SwiGLU, the same structure as our FFN kernel, so
the shapes decode exactly:

- `gate_weight = up_weight = 55,695,360 B` of q6_K at 210 B / 256 weights =
  67,895,296 weights
- output `31 x 18944` f32, so **N = 18944** and **K = 3584** (67,895,296 / 18944)
- input `124,992 B` / 31 rows = 4032 B/row = 112 blocks x 36 B, i.e. q8 with fp32
  scales
- gate+up = 2 x 2 x 31 x 3584 x 18944 = **8.42 GFLOP**

```
same shape (31 x 3584 -> 18944, q6_K weights, gate+up SwiGLU)
  Loom    ffn_gate_up_swiglu_q6q8_zero   3.938 ms   2.14 TFLOP/s   state ok, 1/1
  our HIP one:q6k 18944 3584 31 ... gateup  0.883 ms   9.54 TFLOP/s
```

**Our HIP kernel is 4.5x faster at the same shape and the same weight format.**
Three caveats keep this from being a verdict on Loom:

1. This is an **authoring corpus sample**, not a tuned kernel. Its job is to
   demonstrate the language -- template providers, `check.case`, `check.benchmark`
   -- and Loom's mandate is about where its emitted programs end up, not where the
   examples start.
2. The Loom side is **one cold launch** (warmup 0, iterations 1, host_wall domain)
   against our two warmups plus one timed launch, so it carries cold-start cost.
3. The activation path differs: Loom uses q8_0 integer dots, the
   `v_dot4_i32_i8` class that `dot_peak` measures at 14.2-14.5 TMAC/s (28.4
   TFLOP/s), while ours is fp16 WMMA at 27.6 TMAC/s (55.2 TFLOP/s). Loom's
   instruction class has **half** the ceiling, which explains a factor of two but
   not 4.5x.

What it does establish is the thing worth knowing before committing: **the switch
is not a free win.** A Loom rewrite starts from a kernel 4.5x slower than what we
have, on shapes where our HIP is already reasonable, so the authoring and tuning
effort to reach parity is real and has to be budgeted. The place Loom is most
likely to pay is still where HIP is structurally stuck -- the DeltaNet recurrence's
chunked restructuring -- not where HIP is merely imperfect.

**There is no tuned Loom GEMM to compare against.** `loom/binding/c/benchmark/kernels/`
looks like the right corpus -- it holds `routed_gate_up_swiglu_q4k_q8_amdgpu.loom`
(real Q4_K decode, 144 B blocks) and `attention/prefill_f16_wmma_amdgpu.loom` -- but
its own README says it is "a smoke suite, not a benchmark corpus or model zoo",
and the drivers beside it (`compile_throughput_benchmark`, `link_throughput_benchmark`,
`workload_compile_benchmark`) measure **compiler** throughput, not device runtime.
They exist to prove the production compilation path is exercised, and the Q4_K one is
a routed MoE shape anyway (128 experts, 8 routes, N=768, 16 tokens), not our dense
FFN. So the authoring sample above is the only device number Loom currently offers,
and a fair tuned comparison requires authoring one ourselves.

That is the decision to make deliberately: authoring a Loom kernel at our production
shape (Q4_K, 2048 x 5120 -> 17408) and comparing against our HIP's 41 TFLOP/s is a
bounded experiment that either validates the direction or settles it cheaply. If Loom
lands near parity with modest effort, the DeltaNet restructuring is worth doing there;
if it is multiples off after real work, the HIP path stays.

### The NPU case is the compelling one, and it is exact

The strategic argument for Loom does not rest on the GPU sample above. It rests on
`loom/src/loom/target/arch/amd/xdna`, which is real: 44 C files, ~20k lines, with
`aie2p` arch support (AIE2P **is** the XDNA2 core in Strix Halo), legalization
tables, descriptors, emission and low verification.

From the target's own README:

- Loom "compiles tile programs and their array transport into a native `.xdna`
  executable. The product contains AIE2P instructions, initialized data, array
  configuration, explicit storage and binding requirements, and native invocation
  ranges. It executes through **libamdf**; compilation does not invoke the AIE SDK,
  LLVM, Python, or an external linker."
- **Exact device profiles**, and ours is listed:

  | device | compiler profile | PCI |
  | --- | --- | --- |
  | Strix NPU4 | `amd.xdna.strix.17f0_10` | `17f0:10` |
  | **Strix Halo NPU5** | **`amd.xdna.strix_halo.17f0_11`** | **`17f0:11`** |

  The NPU on this box reports as "Strix Halo Neural Processing Unit (rev 11)", i.e.
  `17f0:11` -- an exact profile match, not a family approximation.
- `libamdf` is the execution path, and it is already verified working here (XDNA
  memory benchmark runs, 13/13 XDNA CTS pass).
- The same source compiles for both halves: the comparable-workload fixture
  `BM_FfnGateUpQuadraticBF16` "compiles the same one-output gate/up operation on
  AMDGPU and XDNA at K=512, 1,024, and 4,096".

That is the NPU argument in one line: Loom is a compiler with an exact profile for
this NPU, a native image format, and a working runtime underneath it, which is
precisely what the NPU work needs and what hand-written HIP cannot provide at all.

So the two halves should be judged separately. **For the NPU, adopt Loom** -- the
evidence is strong and the alternative is nothing. **For the GPU, the sample is
4.5x behind and unproven**, so that half stays a bounded experiment: author one
kernel at our production shape and see whether retuning closes the gap. The user's
read that it "just needs retuning" is the right shape of hypothesis -- the sample
is a language demo, not a tuned kernel -- but the deficit to close is 4.5x, and
both sides already ran on the same runtime.

### The GPU experiment requires authoring the kernel: Loom has no dense prefill GEMM

Searching every Loom corpus, the runnable kernels are all small or non-dense, and
the prefill-class ones cannot be executed:

| corpus | contents | runnable |
| --- | --- | --- |
| `authoring/` | mlp **GEMV** (one input row, K=18944), ffn q6q8 **31 tokens**, routed **MoE** | yes |
| `authoring/hip/` | **GEMM primitives**: shared-memory tiles, q8 load widths, packed fields, cluster multicast | yes |
| `checked_benchmarks/` | paged attention, online softmax, MoE routing | yes |
| `binding/c/benchmark/kernels/` | Q4_K MoE + `prefill_f16_wmma_amdgpu` | **compile-only fixtures** |

`prefill_f16_wmma_amdgpu.loom` is a genuine tiled causal attention prefill kernel
(WMMA, four-wave workgroups, LDS staging, online softmax) but carries **zero
`check.benchmark` rows** -- it exists to time the compiler, and its own header says
the execution corpus lives in a separate module. The dense GEMM we would want to
compare against does not exist in the tree at all.

The runnable GEMM primitives do work on the device, but they are microbenchmarks,
not throughput evidence:

```
shared_memory_vector_tile roundtrip   0.0064 ms   correctness pass
q8_load_width                         0.0066 ms   correctness pass
template_math_legalization            0.0084 ms   correctness pass
cluster_b128_multicast                FAILED: "selected amdgpu HAL device has no
                                      Loom-supported native target"
```

That last one is a real capability gap worth knowing: RDNA has no clusters, so
Loom's cluster/multicast path does not apply to gfx1151.

**So the experiment cannot be run by configuration.** Comparing a Loom GEMM at our
production shape (Q4_K, 2048 x 5120 -> 17408) against our 41 TFLOP/s means
authoring that GEMM in Loom first -- the comparable artifacts in the tree are
300-450 lines of dense Loom IR (`routed_gate_up_swiglu_q4k_q8_amdgpu.loom` at 303,
`prefill_f16_wmma_amdgpu.loom` at 448), so it is a multi-day authoring task on an
early-stage compiler, not a benchmark run.

Given that, the ordering argument is: **the NPU half has an exact device profile and
a working runtime behind it, and the GPU half has no asset at all today.** If the
point of adopting Loom is the NPU -- and that is where the evidence is -- the NPU
path is the cheaper thing to prove end to end, and it simultaneously teaches the
language and the toolchain that the GPU GEMM would need anyway.

### Authored: our first Loom GEMM, working and measured

`engine/gpu/loom/yah_ffn_gemm_f16.loom` is a f16 WMMA GEMM doing
`out[m][t] = sum_k W[m][k] * A[t][k]` at the production FFN shape (M=17408,
K=5120), one 16x16 tile per workgroup, one wave, f32 accumulator. The activation
buffer is token-major, so the rhs operand is a *strided view* of it as [k][t]
(`encoding.layout.strided [1, 5120]`) rather than a transposed copy.

It compiles for gfx1151, dispatches through HRX, and passes its correctness case.
The case uses all-ones operands, so every output must equal exactly 5120 in f32 --
which checks the K reduction and the operand indexing, not just the absence of NaNs.
1956 instructions emitted.

| | weight format | time | TFLOP/s |
| --- | --- | ---: | ---: |
| our HIP, tuned (`one:q4k 17408 5120 64`) | Q4_K, 4.5 bits | **1.197 ms** | **9.53** |
| our authored Loom GEMM | f16, 16 bits | **6.109 ms** | **1.87** |

So the first authored version is 5.1x behind -- and the gap decomposes into two
parts that are both understood rather than mysterious:

1. **Format.** Ours reads 3.5x fewer weight bytes (0.5625 vs 2 B/weight), and this
   shape is weight-traffic sensitive. A Q4_K Loom kernel starts ~3.5x closer
   before any tiling work.
2. **Tile shape, deliberately not tuned.** 16x16 with a single wave runs at ~3.4%
   of peak. The levers are known and unwritten: a 16x128 or larger N tile so the
   weight tile is reused across more tokens (at batch 64 the current tile re-reads
   each weight tile 4 times, at batch 2048 it would be 128 times), LDS staging of
   the weight tile via `buffer.alloca<workgroup>` + `kernel.barrier<workgroup>`,
   multi-wave workgroups, and vectorized fragment loads.

The point of this artifact is that it moves "can Loom do this at all" from unknown
to **yes, verified on the device** -- and it turns the tuning argument into a list
of specific, independently testable changes rather than a hope. Both halves are now
on the same footing: the NPU path has an exact target profile and a working runtime,
and the GPU path has a working kernel we authored and can iterate on.

### Second Loom GEMM version: the N tile was worth 2x, and the model was only half right

The first authored kernel tiled the token axis across four workgroups, so each of
the 1088 M tiles read its 16x5120 weight tile four times. Collapsing the token
axis into one workgroup (M16 x N64, four accumulators sharing one weight fragment
load per K step) makes the weight tensor read exactly once.

| kernel | tile | weight traffic | time | TFLOP/s |
| --- | --- | ---: | ---: | ---: |
| Loom v1 | 16x16, n_tiles=4 | 713 MB | 5.979 ms | 1.909 |
| Loom v2 | 16x64, n_tiles=1 | **178 MB** | **3.000 ms** | **3.805** |
| our HIP, fp16 weights | 256x256 | 178 MB | 1.353 ms | 8.43 |

Both Loom numbers are iree-benchmark-loom with --measure=dispatch_complete
--warmup-iterations=2 --iterations=5 --max-batches=5, run serially in one session,
with the correctness case passing in both.

**The traffic model predicted 1.5 ms and the kernel took 3.0 ms, so it was only
half right.** v1 moved 713 MB in 5.979 ms (119 GB/s), but v2 moves 178 MB in
3.000 ms (59 GB/s) -- v2 is not weight-bandwidth bound at all. Every WMMA operand
comes straight from global memory with no pipelining across K steps, so each of
the 320 iterations exposes a full memory latency; across 40 CUs that is about one
WMMA per 121 cycles, which is latency-bound, not issue-bound. Against our tuned
HIP kernel at the same 16-bit precision the gap is now 2.22x, and the remaining
levers are identified rather than speculative: more independent accumulator chains
per workgroup (a larger M tile, which also raises the WMMA-per-load ratio) and LDS
staging of the operands.

Measurement note: --measure=auto silently selects case_end_to_end for a check.case,
which times the tensor fills and the correctness compare as well as the kernel.
Kernel time requires --measure=dispatch_complete; the two modes gave 38.6 ms and
5.979 ms for the same v1 binary, so the mode must be stated with any Loom number
quoted.
