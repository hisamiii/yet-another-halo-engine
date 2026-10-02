# engine/

The greenfield implementation. The layout maps to the workstreams in [../docs/scope.md](../docs/scope.md).

| dir | contents | status |
| --- | --- | --- |
| `core/` | GGUF reader, mmap, tensor table, config, tokenizer | reader/config/**tokenizer done** |
| `model/` | Qwen3.8 27B graph: Gated DeltaNet, attention, RoPE, norms, FFN, state | **HRX-native prefill and decode, token-gated** |
| `gpu/` | Loom kernel ports, HRX HAL executables | **all shard formats ported; prefill argmax and 16-token decode match the reference** |
| `npu/` | XRT executor, xclbin set, dma-buf operands, async launch | not started |
| `sched/` | phase routing, token split, overlap, power budget | not started |
| `kv/` | paged quantized KV cache | fp16 decode cache on the HRX path; paged quant not started |
| `vision/` | mmproj projector, image preprocessing | not started |
| `serve/` | CLI, HTTP, sampler | CLI done (`loom_decode`); HTTP pending |

Milestone mapping: **M0** core + model + gpu + serve. **M1** npu + sched. **M2** npu (int8). **M2g** gpu. **M3** sched + model. **M4** kv. **M5** vision.

Rules: one binary, one architecture, hardcoded shapes. Port the framework, not the efficiency claim. Every change is gated by the top-1 validation check, not by throughput.

## The M0 gate

`tests/m0_gate.sh` and `tests/generate_gate.sh` are the milestone gate, and both
run on the HRX-native `loom_decode` runner. The first requires the engine's greedy
next token to equal the recorded reference on three fixed prompts; the second
requires 20 greedy tokens to match, token for token. Both pass on the IQ4_XS
artifact: `engine/build_hrx.sh <model>` then `engine/tests/*.sh <model>`. The
runner processes one token at a time, so the prompt and the generated tokens
share one path; the fp16 KV cache and the recurrent state persist across steps.

Weights are the GGUF's own `mmap`, imported once as an HRX device-visible buffer
from its page-aligned base, so there is no second copy.

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

Validated on this box: the HRX-native engine reproduces the recorded reference
sequence token for token (`11751 13 198 760 ... 14898 369`) and both gates pass.
The old HIP GPU tools (`yah-*-check`, `yah-run`, the `gpu/ported` kernel tree and
`build_gpu.sh`) were removed with HIP; correctness is now gated end to end by the
HRX runner.

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

Tools (they land with the HRX build; `engine/build_hrx.sh` uses them):

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
`out[t][m] = sum_k W[m][k] * A[t][k]` at the production FFN shape (M=17408,
K=5120), one 16x16 tile per workgroup, one wave, f32 accumulator. Both the
activation and the result are token-major, matching HIP `y[token*m + row]`. The
activation is therefore a *strided view* as [k][t] (`encoding.layout.strided
[1, 5120]`) and the result is a strided view as `[1, 17408]`, rather than a
transposed copy in either direction.

It compiles for gfx1151, dispatches through HRX, and passes its correctness case.
The original case used all-ones operands, so every output equalled 5120 in f32.
That checked the K reduction but not the output layout, because a constant is its
own transpose: the result was in fact written `out[row][token]` through a default
row-major view, a different buffer layout from HIP. The case is now
layout-sensitive -- the weight varies with row parity, so even rows sum to
13104640 and odd rows to 39319040, and the expected value is an iota of period 2
over the flat token-major index. Reverting the result view to row-major fails the
case.

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

### Loom GEMM progression, and three levers that were falsified

All numbers: gfx1151, iree-benchmark-loom --measure=dispatch_complete, 2 warmups
and 5 timed batches, run serially in one session, correctness passing in every
arm. Our HIP kernel at the same fp16 precision is the control.

| kernel | time | TFLOP/s | vs previous |
| --- | ---: | ---: | --- |
| Loom v1: 16x16 tile, n_tiles=4 | 5.979 ms | 1.909 | - |
| Loom v2: 16x64 tile, weights read once | 3.000 ms | 3.805 | 2.00x |
| Loom v3: + pipeline(depth 3, unroll 2) | **1.819 ms** | **6.271** | 1.65x |
| our HIP, fp16 weights | 1.353 ms | 8.43 | 1.34x ahead of v3 |

**Falsified: M32 x N64 tile.** Sharing one set of activation fragments across two
weight fragments cuts loads per WMMA from 1.25 to 0.75 and halves total activation
traffic, and it is 30% slower (2.370 ms). It compiles to 256 vector registers, the
per-thread maximum, with zero spills -- halving resident warps, and the occupancy
loss beats the reuse gain. Depth two on the same tile is worse still (3.159 ms).

**Falsified: pipeline depth four.** 1.918 ms, 5% slower than depth three. The
read-ahead queue costs vector registers (80 at depth one, 200 at depth three, 240
at depth four) and therefore resident warps; past depth three the occupancy loss
overtakes the overlap gained. Depth two is worse than depth three as well (2.501 ms).

**Falsified: transposing the activation to [k][t].** The rhs operand dominates
issued loads, and the compile report shows v3 reading it *gapped*: 512 bytes
requested, 16 discontiguous regions, 10 KiB maximum gap. Handing the kernel the
activation already transposed makes that access dense with one contiguous region
and zero gaps, and drops vector registers from 200 to 152 -- every static signal
improves, and the kernel is **3.13x slower** (5.701 ms). The token-major
B-fragment path is evidently much cheaper to gather than the k-major one, so the
strided view was never the bottleneck. A static access-geometry argument is not
evidence about this kernel.

So the remaining 1.34x against HIP is not the activation access pattern and not
the tile shape. The untested lever with a mechanism behind it is the weight
operand: v3 reads it row-major through the same gapped pattern (512 requested, 16
regions), and our HIP kernel avoids that by repacking weights into the blocked
layout the fragments want. That is the next experiment.

### The tile is at a local optimum, and the reason is DRAM bytes vs L2 bytes

v6 tested the one remaining tiling lever suggested by the compile report. With four
accumulator fragments per thread the tile area is fixed at 16x16x4 = 1024 and total
issued traffic goes as 1/BM + 1/BN, so 32x32 (1/32 + 1/32) issues 20% fewer bytes
than 16x64 (1/16 + 1/64) and needs fewer loads per WMMA (4/4 instead of 5/4). The
report confirms it exactly: 168 vector registers instead of 200, and total issued
read 1.426 GB instead of 1.783 GB. It is **2.04x slower** (3.713 ms against 1.819).

The resolution is which operand the bytes belong to. The weight tensor is
17408 * 5120 * 2 = 178,257,920 bytes, and the report's issued-byte figure carries a
constant 2x lane-accounting factor (the access geometry shows `requested=512`
against `unique=256` for a single packet), so issued/2 is the number of times the
tensor is read:

| kernel | weight issued | / tensor size | weight reads | activation issued |
| --- | ---: | ---: | ---: | ---: |
| v3 16x64 | 356,515,840 | 2.00x | **1** | 1426 MB |
| v6 32x32 | 713,031,680 | 4.00x | **2** | 713 MB |

v3 reads the weights exactly once and its activation operand is only 655 KiB, which
is L2-resident and therefore nearly free no matter how many of the 1088 workgroups
re-read it. v6 trades 713 MB of that cheap L2-resident activation traffic for a
second, expensive pass over 178 MB of DRAM-streamed weights. Fewer issued bytes,
much worse bytes. Minimising issued bytes is the wrong objective; minimising DRAM
traffic is right, and v3 is already at the floor for these operand sizes -- one read
of the weights, and the activation read once from DRAM and then from L2.

This one mechanism explains every falsification in this section. The M32 16x64 tile
lost to occupancy. Depth four lost to occupancy. The transposed activation lost
because the token-major B-fragment gather is cheaper than the k-major one. And
32x32 lost because it converted cheap L2 bytes into expensive DRAM bytes. v3 sits at
a local optimum on tile shape, and the remaining 1.34x against our HIP kernel is
memory-system efficiency on the same 178 MB (v3 moves it at 98 GB/s, HIP at 132
GB/s), not tile geometry.

Going further needs one of two different things, neither of which is a tile tweak:
either a multi-wave workgroup with LDS staging to raise achieved bandwidth, or the
production w4a16 format, where the weights shrink about 3.5x and the whole balance
changes. The f16 kernel has served its purpose, which was to establish that Loom can
express this kernel and to find where the time actually goes.

### Instruction count is not the limit, and the schedule space is exhausted

The v3 compile report shows a suspicious instruction mix per work-item: 12,823
register moves against 1,280 WMMA, i.e. ten moves per matrix op, making moves 45%
of all instructions. The Loom loop-schedule guide predicts exactly this shape and
prescribes the fix -- `unroll(%factor) schedule(recurrence)` should give allocation a
larger repeating body with a copy-free steady backedge, and its gfx1151 example finds
depth three / factor four. v3 had only been swept at unroll two.

| schedule | vector VGPR | register moves | moves/WMMA | time |
| --- | ---: | ---: | ---: | ---: |
| d3 u1 | 160 | 25,543 | 19.96 | 2.347 ms |
| d3 u2 | 200 | 12,823 | 10.02 | **1.819 ms** |
| d3 u4 | 200 | 271 | **0.21** | 1.821 ms |
| d2 u4 | 200 | 191 | 0.15 | 1.922 ms |

Depth three / factor four does exactly what was predicted -- register moves fall by
98%, to a near-copy-free backedge -- and the kernel is **no faster**: 1.821 ms against
1.819 ms. So the register moves are a symptom of register pressure, not a cost, and
the kernel is not issue-bound at all. That is consistent with the issue budget:
roughly 28k instructions per warp across 1088 warps is about 7% of what four SIMD
units per CU could retire in 1.82 ms.

With that, three directions are closed on this kernel and only one is left open:

- **DRAM traffic** is at its floor. Counters (see docs/hrx-evaluation.md) put DRAM
  reads at 157.7 MB against a 178.3 MB weight tensor, so the weights stream once and
  the activation re-reads are entirely L2-served.
- **Issue / instruction count** is not the limit, as the table above shows directly.
- **Tile shape** is exhausted, and every reshape that raises reuse raises DRAM weight
  traffic in exchange for L2 traffic that is already free.
- **Open: operand delivery.** The kernel stalls on memory latency with a register-
  limited residency, and pipeline depth saturates at three. Raising in-flight bytes
  further requires moving the operand queue out of registers and into shared memory
  in a multi-wave workgroup -- wide coalesced global loads into LDS, MMAs reading
  LDS, with several warps sharing one staged activation tile. That is the one
  structural change left, it is a rewrite rather than a tweak, and the v4/v6 results
  are a warning that the register/occupancy tradeoff around here is tight.

### Loom development practice: the methodology we should be following

Loom ships a real method, not just syntax docs. The centrepiece is
`loom/docs/src/workflows/agent-driven-kernel-development.md`, with
`tune-loop-schedules.md`, `search-loop-schedules.md`, `benchmark.md`,
`compile-reports.md` and `format-and-verify.md` owning the focused contracts. The
parts that matter most here:

1. **Keep three evidence classes independent.** Numerical (does it implement the
   operation), compiler (what did lowering emit), physical (what did the device do).
   "A green check does not make a kernel fast. Fewer VGPRs do not make a kernel
   faster. A short timestamp does not make a kernel correct."
2. **Bound the regime with mechanism probes before hill climbing.** The named probes
   are a load-only proxy (same addresses and publication path, minimal arithmetic),
   a cache-resident proxy (same compute and schedule, operands from an intended cache
   level), and a dispatch-only proxy. Their decision rule: "If the candidate is below
   the known oracle and far from every relevant roofline, it is probably missing a
   structural schedule rather than a locally better integer tile."
3. **Two coupled optimization phases.** Create headroom (shorter live ranges, fewer
   VGPR/LDS, no spills or barriers, less traffic), then fill it (independent work,
   accumulator chains, coalesced requests, staged prefetch, overlap). The first phase
   "can make a report look cleaner while making the kernel slower".
4. **Reject attractive non-evidence.** The table names our exact traps: fewer
   registers or higher modelled occupancy, and a shorter native listing, are not
   wins. A losing candidate is still valuable when it falsifies a mechanism.
5. **One causal hypothesis per edit, stated before compiling**, in a candidate record
   (production boundary, independent variable, hypothesis, expected compiler
   consequence, correctness gates, discriminator, predeclared stop threshold).
6. **Ask the compiler before the GPU**: `show`, `suggest`, strict `diff`, and only then
   IR or ISA. An empty suggestion list is not a win.
7. **Correctness cases need distinct values.** Identity, all-ones and all-zero
   patterns can let a swapped or transposed binding pass.
8. **Compare with interleaved A/B** (`--compare=@base,@cand --interleave=ABABA`)
   rather than serial runs, and keep the time domain in the result identity.

Where this kernel stands against it:

| practice | status |
| --- | --- |
| independent numerical / compiler / physical evidence | followed; v4, v5, v6 and d3u4 were each rejected on physical evidence |
| reject attractive non-evidence | followed; fewer registers (v6) and fewer instructions (d3u4) were both rejected as non-wins |
| regime probes before hill climbing | **not done** -- no load-only or cache-resident proxy has been run |
| ask the compiler first | **partly** -- `loom-compile-report` is not built here, so `suggest` and `diff` are unavailable and the raw report JSON had to be read directly |
| distinct correctness values | **not done** -- the case uses all-ones on both operands, so an operand indexing transposition would still pass |
| interleaved A/B comparison | **not done** -- arms were run serially with a 15 s gap |
| `loom-format --check` | followed; passes |

The compiler evidence is more interesting than expected. `wait_plan` reports 4 full
drains with `max_full_drain_outstanding_before = 32`, which is the pathology the guide
warns about ("a full wait before such a copy can finish future loads earlier than the
arithmetic needs them"). The peak live value in `allocation_high_water_rows` is the
`rhs2` fragment, origin `concat`: the strided `[k][t]` activation view lowers each rhs
fragment load to a concatenation of register pieces, which is exactly where the 12,823
register moves come from (8 per fragment). Unroll four removes those moves and the
full drains change count (6 vs 4) -- and the runtime does not move, so neither is the
binding constraint.

That makes the load-only proxy the next step rather than another tile or schedule:
nothing in the compiler evidence isolates the cause, and the probes are the part of
the method designed for exactly that. It is also the cheapest remaining experiment.

### Load-only proxy: the kernel is 95% memory path

The Loom method prescribes a load-only proxy before further hill climbing, and it
is decisive here. `engine/gpu/loom/yah_ffn_gemm_loadonly_probe.loom` is v3 with the
four `vector.mma` operations replaced by four independent elementwise `vector.addf`
accumulations -- that is, identical loads at identical offsets, identical K loop,
identical pipeline depth and unroll, identical grid, and no matrix op at all.

The mechanism-survival evidence is compiler-side, as the method requires:

| | v3 real kernel | load-only probe |
| --- | ---: | ---: |
| `global_load_count` per work-item | 3200 | **3200** |
| `wmma_count` | 1280 | **0** |
| issued read bytes | 1,782,579,200 | **1,782,579,200** |
| vector VGPR | 200 | 204 |
| **device time** | **1.819 ms** | **1.727 ms** |

Removing all 1,393,000 WMMA operations -- the entire matrix workload -- saves 5%.
The kernel is therefore about 95% memory-path bound, and the corollary the method
states directly applies: "if a load-only proxy is already near the candidate,
compute rewrites cannot recover much." Any further tiling, scheduling or
accumulator-chain work is bounded by 5% before it starts.

This also reframes the remaining 1.34x against our HIP kernel (1.353 ms). HIP does
the same loads *and* the matrix work in less time than our load path alone needs,
while moving the same 178 MB of weight DRAM traffic. So the gap is not DRAM
bandwidth, not compute, and not traffic volume: it is how well the memory path
generates and coalesces outstanding requests. That is precisely what a multi-wave
workgroup with LDS-staged operands changes -- wide coalesced global fills instead of
16-byte-per-lane strided fragment gathers, and many more bytes in flight per issue
slot -- and the probe gives it a hard target: the load path must go from 1.727 ms to
about 1.35 ms.

### `loom-compile-report` works here, and its headline suggestion is already falsified

It is not a C++ tool in `src/loom/tools/` -- it is Python, at
`loom/py/loom/tools/compile_report.py`, and it runs under the system Python 3.14.7:

```shell
cd /home/q/hrx/loom/py && python3 -m loom.tools.compile_report suggest report.json
```

The report for v3 is available and makes one high-confidence recommendation:

```
[amdgpu.residency_cliff]  occupancy_percent: 25
  Reduce amdgpu.vgpr by at least 8 registers/subgroup (to at most 192).
  Recompile and benchmark the modeled transition 4 -> 5 subgroups/SIMD.
```

v3 sits at 200 vector registers, so it misses the 192 threshold that would move it
from 4 to 5 subgroups per SIMD by a margin of 8.

192 is the budget our HIP prefill GEMM actually runs at, verified from the device
object with `llvm-readobj --notes gemm_bench.0.hipv4-amdgcn-amd-amdhsa--gfx1151`:

| tile | threads | type | VGPR | spills |
| --- | ---: | --- | ---: | ---: |
| 256x256 w8n4 | 1024 | Q4_K | **192** | 0 |
| 256x256 w8n4 | 1024 | IQ4_XS | 192 | 0 |
| 256x256 w8n4 | 1024 | Q8_0 | 187 | 0 |
| 256x256 w8n4 | 1024 | Q5_K | 168 | 0 |
| 256x256 w8n4 | 1024 | Q6_K | 163 | 0 |
| 256x256 w4n4 | 512 | Q4_K | 250 | 0 |
| 256x256 w4n2 | 256 | Q4_K | 221 | 0 |

The 1024-thread arm is pinned at 196608 / 1024 = 192 with zero spills, so it is at its
launch-bound ceiling rather than stopping short of it. The 512- and 256-thread arms
exceed 192 because their per-thread ceilings are higher.

**Correction: an earlier revision of this note claimed the suggestion was already
falsified, and that was wrong.** The two variants it cited changed more than the
register count. d3 u1 changes the unroll factor, and therefore the schedule (19.96
register moves per WMMA against 10.02). v5 changes the activation layout, so its
addresses differ entirely. Neither isolates "200 -> 192 with everything else fixed",
and neither is evidence about the cliff. The honest measurement set is:

| variant | vector VGPR | subgroups/SIMD | occupancy | time |
| --- | ---: | ---: | ---: | ---: |
| d1 u1 | 80 | 12 | 75% | 2.895 ms |
| d3 u1 | 160 | 6 | 37% | 2.347 ms |
| **d3 u2 (v3)** | **200** | **4** | **25%** | **1.819 ms** |
| d3 u4 | 200 | 4 | 25% | 1.821 ms |
| d4 u2 | 240 | 4 | 25% | 1.918 ms |
| v5, contiguous activation | 152 | 6 | 37% | 5.701 ms |

Every row above 25% occupancy is slower -- but each reached that occupancy either by
giving up read-ahead or by changing the addresses, so occupancy is confounded with
something else in every one of them. The cell the suggestion actually names, tier 5 at
depth 3 / unroll 2 with unchanged addresses, has never been measured.

Reaching it is not a flag. The AMDGPU target attribute accepts only `subgroup_size`,
the target configs are build tooling, and the residency model is derived from generated
target tables in `planning/occupancy.c` rather than from author input. In this kernel
register pressure *is* the pipeline depth: the queue holds 2 records x 5 values x 8
registers = 80 registers, plus 32 for the four accumulators, so every source-level way
to cut 8 registers also cuts read-ahead.

That makes the LDS rewrite the natural test rather than a separate idea. Moving the
operand queue out of registers and into shared memory lowers per-thread registers while
*preserving* read-ahead, so it should land under 192 and exercise the cliff as a side
effect. It is also what the load-only proxy independently demands, being the only change
that alters the memory path itself. Its target remains 1.727 ms down to about 1.35 ms.

So the compiler and the counters agree on what is *not* the limit (compute, DRAM
traffic, issued bytes, register count as such), and the load-only proxy says what is
(the memory path, at 95%). The one untested mechanism that changes the memory path
itself -- transaction shape, via LDS staging with wide coalesced fills -- remains the
next experiment, with the probe as its hard target: 1.727 ms down to about 1.35 ms.

### Correction: the fp16 control uses no LDS, so LDS staging is not the differentiator

Reading the exact kernels rather than the tile family changes the plan. After forcing a
device rebuild, the code object was extracted from build/obj/gemm_bench.o with
llvm-objcopy --dump-section .hip_fatbin followed by clang-offload-bundler --unbundle,
and the notes were read for all 374 kernels in the current build:

| kernel | VGPR | SGPR | LDS | spills | threads |
| --- | ---: | ---: | ---: | ---: | ---: |
| one:q4k control (ablate 16) | **192** | 42 | 64 KiB | 0 | 1024 |
| fp16w:q4k control (ablate 16777232) | **190** | 42 | **0** | 0 | 1024 |

Two things follow. First, the Q4_K kernel we benchmark against sits at exactly 192
VGPR, which is also its launch-bound ceiling (196608 / 1024), so 192 is a real
operating point rather than a synthetic target and is worth testing the Loom kernel
at. Second, and more importantly, the fp16 arm we compare Loom against uses no shared
memory at all: it reads its operands directly from global memory, exactly as the Loom
kernel does.

That retracts the mechanism proposed in the previous section. LDS staging with wide
coalesced fills cannot be what HIP does differently in the comparison that matters,
because in that comparison it does not stage anything. What remains different is
narrower:

- **workgroup shape**: 1024 threads across 32 waves, against 32 threads in one wave,
  at essentially the same per-thread budget (190 against 200 VGPR);
- **tile**: 256x256 against 16x64, so 8 accumulator fragments per warp against 4;
- **per-warp operand traffic**: every Loom warp issues 5 fragment loads per K step,
  each a 16-byte-per-lane gather, where the HIP warp covers a 32x64 sub-tile.

The next experiment is therefore the workgroup shape rather than LDS: more waves per
workgroup, each keeping the same per-warp schedule, so that many more independent
requests are in flight at the same per-thread register budget. That also makes the
residency boundary load-bearing rather than incidental, since the whole point is to
add waves without crossing roughly 192 registers.

### Reversal: both control kernels use 64 KiB of LDS, and LDS staging is the difference

The previous section claimed the fp16 control uses no shared memory and that LDS
staging therefore cannot explain the gap. **That claim was wrong, and it is retracted
here.** It came from a parsing bug: each kernel note block prints
`.group_segment_fixed_size` *before* `.name`, but the extraction script searched only
the text after `.name`, so it reported the following kernel LDS value. Parsing each
block from `.group_segment_fixed_size` to the next one gives:

| kernel | VGPR | SGPR | LDS | spills |
| --- | ---: | ---: | ---: | ---: |
| one:q4k control (ablate 16) | 192 | 42 | **65536** | 0 |
| fp16w:q4k control (ablate 16777232) | 190 | 42 | **65536** | 0 |

The VGPR figures were unaffected (that field really does follow `.name`), so 192 and
190 stand. Only the LDS column was wrong.

The disassembly settles it independently, and this is the comparison that matters:

| instruction | Loom v3 | HIP fp16 control | HIP Q4_K control |
| --- | ---: | ---: | ---: |
| global_load_b128 | **40** | 8 | 6 |
| ds_load_b128 (LDS) | **0** | **48** | 48 |
| ds_store_b128 | 0 | 4 | 4 |
| v_wmma_f32_16x16x16_f16 | 16 | 32 | 32 |

The HIP kernel fetches each operand byte once from global memory into 64 KiB of
shared memory and then serves 48 of its operand reads per body from LDS, where the
Loom kernel issues 40 global gathers and never touches shared memory. That is the
structural difference, and it explains the numbers without any new assumption: the
Loom load path alone costs 1.727 ms while HIP completes the same loads *and* the
matrix work in 1.353 ms, because HIP pays L2/L1 bandwidth and latency for each
operand once and then reads LDS, while every Loom MMA operand goes back out to the
memory hierarchy.

It also retroactively explains the whole falsification list. Every experiment in this
document reshuffled global accesses -- tile, schedule, layout, workgroup packing --
and none of them introduced shared-memory staging, so none of them could touch the
operand-delivery path that the load-only proxy measured at 95% of the time.

Method note worth keeping: a plausible-looking number from a hand-written parser was
wrong for two rounds and produced a confidently-stated retraction of a correct
claim. The native disassembly is what caught it. Compiler metadata and native
evidence are not interchangeable, which is exactly why the Loom development guide
keeps them as separate evidence classes.

### LDS staging implemented: the mechanism worked and still lost

engine/gpu/loom/yah_ffn_gemm_lds.loom stages operands through shared memory the way
the HIP control does: a BK=64 block per fill, 32 lanes assigned four-per-row so one
fill instruction covers 8 runs of 128 contiguous bytes, two barriers per 64-K block,
and every MMA operand read from LDS. It compiles, verifies, executes, and passes the
all-ones correctness case. Measured:

| metric | v3 (no LDS) | v7 (LDS staged) |
| --- | ---: | ---: |
| device time | **1.819 ms** | 2.160 ms |
| vector VGPR | 200 | **64** |
| global loads per work-item | 3200 | **1600** |
| LDS bytes per workgroup | 0 | 10240 |
| fragment access max gap | 10224 B | **112 B** |
| issued read bytes | 1.78 GB | 2.67 GB |
| residency tier | 4 | **3** |

So the intended mechanism did happen. Global fragment gathers halved, vector registers
fell by two thirds, and operand access became local (the worst-case gap in a fragment
access dropped from 10224 bytes to 112). It is still 19% slower, and the report says
why: amdgpu.lds becomes the residency limiter. LDS is a pooled resource in an
occupancy domain (pool 131072 B, granularity 512), so 10240 B per single-wave
workgroup costs more residency than the 200 registers it saves, and the tier falls to
3. Issued read bytes also rose by half, because the fill reads wider per lane than the
gathers it replaced.

That makes the next experiment a footprint question rather than a mechanism question:
the mechanism is demonstrated, so shrink the staged tile (stage only the activation,
or a smaller BK), and fold several waves into one LDS allocation so the 128-byte-run
fetch is paid once and reused, rather than being charged to one wave alone.

Tooling note: loom-check roundtrip reports this file as mismatched while loom-format
--in-place reports it unchanged and verifies it. The two tools disagree on the
canonical form for this file, so compilation and measurement were used as the gate.

### The LDS line is closed: staging loses in all three variants

Following the disassembly finding (HIP serves 48 operand reads per body from LDS while
Loom serves none), three staged kernels were built and measured. All are correct and
all lose to the plain global-gather kernel:

| kernel | staging | VGPR | global loads/wi | LDS B | LDS/MMA | tier | time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 | none | 200 | 3200 | 0 | 0 | 4 | **1.819 ms** |
| v7 | BK=64, 1 wave | 64 | 1600 | 10240 | 3.75 | 3 | 2.160 ms |
| v8 | BK=64, shared by 4 waves | 64 | 480 | 10240 | 5.50 | 12 | 3.100 ms |
| v9 | v8 with the row padded to 66 | 64 | 480 | 10560 | 5.50 | 12 | 3.872 ms |

Each variant fixed the previous objection and lost anyway:

- v7 demonstrated the mechanism -- the worst-case fragment gap fell from 10224 bytes to
  112, global loads halved, registers fell by two thirds -- but 10240 B of staging
  charged to a single wave made amdgpu.lds the residency limiter at tier 3.
- v8 amortised the same 10240 B across four waves, which fixed residency outright:
  tier 12, 75% occupancy, global loads down 6.7x. It is 42% slower than v7. So
  neither occupancy nor global-load count is the limiter, and 5.5 LDS operations per
  MMA is a cost the global path does not pay.
- v9 padded the shared tile row from 64 to 66 halves to break what looked like a
  16-way bank conflict (a 128-byte row stride is exactly the bank width, so lanes
  stepping one row apart would all land on bank 0). It is slower again, which says the
  padded access did not pay for the extra footprint.

Conclusion: on this hardware with this compiler, routing operands through workgroup
memory costs more than reading them as global fragment gathers, in every shape tried.
The HIP kernel LDS result is real but it is not transferable by construction -- its
advantage comes from a hand-tuned 256x256 tile at 1024 threads with BK=4 sub-blocks,
not from the mere presence of shared memory.

The standing result is therefore v3 at 1.819 ms against the HIP fp16 control at
1.353 ms, a 1.34x gap. Its own load path alone measures 1.727 ms, so the compute is
5% and the entire remaining question is memory-path efficiency that none of the
twelve variants or three probes has been able to move in the right direction.

### Loop schedule choice: recurrence is the best of the four

Only schedule(recurrence) had been used. The other accepted forms were swept at depth
3 / unroll 2 with everything else fixed:

| schedule | vector VGPR | spills | time |
| --- | ---: | ---: | ---: |
| **recurrence** | 200 | 0 | **1.819 ms** |
| interleaved | 200 | 0 | 1.9895 ms |
| linear | 200 | 0 | 1.9895 ms |
| locked | - | - | rejected: not valid on scf.for |

interleaved and linear produce byte-identical timings (same mean, min, max and p50),
so they lower to the same program here. recurrence is 9% faster than either. The
schedule knob is therefore closed as well.

### What is left is below the source language

With tile, schedule, depth, unroll, operand layout, workgroup packing and LDS staging
all swept, the remaining 1.34x lives in how Loom lowers a global fragment load. The
report records a null strategy on those packets and the fragment-memory machinery sits
in target/arch/amdgpu/lower/fragment_memory/, with no author-selectable strategy:
mixed_fragment_memory_strategy is a diagnostic reason key, not a knob. The compile
report offers only two suggestions for this kernel, scf.compare_pipeline_depth and
amdgpu.residency_cliff, and both have been run to a result.

So the honest summary of the Loom GPU experiment: the authoring path works, the
correctness case holds, the kernel went from 5.979 ms to 1.819 ms over twelve variants,
and it now sits 1.34x behind a hand-tuned HIP kernel whose own load path alone costs
less than this kernel load path by a margin no source-level change has been able to
close. Further progress needs a change in the fragment-load lowering, not in the
kernel.

### The 192-register question, tested from both sides of the cliff

The claim had been made that the residency-cliff suggestion was falsified without ever
measuring 192 or 190. That was fair. What the depth/unroll grid actually contains:

| d.u | VGPR | tier | | d.u | VGPR | tier |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| 2.1 | 120 | 8 | | 3.2 | 200 | 4 |
| 2.2 | 168 | 5 | | 3.3 | 240 | 4 |
| 2.3 | 161 | 5 | | 3.4 | 200 | 4 |
| 3.1 | 160 | 6 | | 4.1 | 200 | 4 |

There is no 192 or 190 cell: the grid jumps 168 to 200. And Loom offers no register
cap to force one. The obvious route was tried -- the HIP controls sit at 192 and 190
because they declare 1024-thread workgroups, whose per-thread ceiling is
196608/1024 = 192 -- so a variant with the identical per-wave schedule packed into 32
waves (`workgroup_size(1024)`, 34 workgroups of 512 rows) was compiled. It still
allocates **200** VGPR at tier 4. Loom does not derive a register ceiling from the
declared workgroup size, so HIP register number is a consequence of HIP own codegen
and not a constraint Loom would reproduce by construction.

The cliff has therefore been measured from both sides rather than at the exact number,
and the result is unambiguous:

| config | VGPR | tier | time |
| --- | ---: | ---: | ---: |
| d1 u1 | 80 | 12 | 2.895 ms |
| d2 u2 | 168 | **5** | 2.388 ms |
| d2 u3 | 161 | **5** | 2.398 ms |
| d3 u1 | 160 | 6 | 2.347 ms |
| **d3 u2 (v3)** | **200** | **4** | **1.819 ms** |
| d3 u4 | 200 | 4 | 1.821 ms |
| HIP fp16 control | 190 | 5 | **1.353 ms** |
| HIP Q4_K control | 192 | 5 | 1.197 ms |

Every config that reaches tier 5 or better is around 2.35 to 2.40 ms, roughly 30%
slower than the tier-4 config. They reach the higher tier by giving up read-ahead --
depth 2 or unroll 1 -- not by being more efficient at the same schedule. So crossing
the cliff is not a win here, and the suggestion was correctly left alone, though for a
better-supported reason than the one originally given.

The sharper point is the last two rows. HIP reaches **tier 5 at 1.353 ms** while our
tier-5 configs sit at 2.39 ms. The tier is not what makes HIP fast, and matching the
register number without matching the code buys nothing.

### Resolution: the suggestion was actionable, and it was taken

The open question was why the compiler would name a specific register reduction if
there were no way to perform it. Re-reading the suggestion text answers it:

> Reduce amdgpu.vgpr by at least 8 registers/subgroup (to at most 192). Recompile and
> benchmark the modeled transition 4 -> 5 subgroups/SIMD; higher modeled residency is
> not a throughput guarantee.

It reports a resource delta against the residency model and explicitly disclaims a
throughput guarantee. It never claimed a source transformation for one fixed schedule.

The reduction is reachable, and the model even names the right size. The pipeline queue
holds 2 records x 5 values (one lhs plus four rhs fragment loads), and each value is a
vector<16xf16> occupying 8 registers, so dropping a single value per record is exactly
the 8 registers the suggestion asks for. Configurations that cross the tier boundary at
depth 3 were not available, but they are at other points in the grid and were measured:
d2 u3 at 161 VGPR and d2 u2 at 168 VGPR both reach tier 5. So the suggestion was
actionable, the transition was performed, and the result was 2.39 ms against 1.819 ms.

Attempts to reach it without moving depth or unroll, which would have isolated the
residency variable, all left the allocation at 200: bounding the row origin with
index.assume, declaring 16-byte alignment on the buffers, and reordering the five loads
so the lhs is issued last. The queue is what sets the number, and the queue length is
the read-ahead depth.

So the correct statement is not that the suggestion was unfounded. It is that the
suggestion was right about the resource, the transition it names costs about 30% of
throughput on this kernel, and the tool says as much in its own action text. The
earlier framing of this section treated an unisolatable variable as a bogus suggestion,
and that was the mistake.
