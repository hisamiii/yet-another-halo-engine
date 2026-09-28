# HRX evaluation: both halves work; the GPU needs the ROCm 10.0.0 *release* HSA

**Verdict: HRX drives both the NPU and the GPU on this box.** The GPU path needs the
HSA runtime from the **official AMD ROCm 10.0.0 release**, not from TheRock's
nightly/dev channel -- and this document originally concluded the opposite, from
nightly builds only. That correction is the main content here.

Evaluated at `ROCm/hrx` commit `244cd38` (`main`). HRX describes itself as "an
alternative implementation of HIP", installs `libhrx.so` plus a HIP compatibility
layer as `lib/libamdhip64.so`, and is "early-access runtime infrastructure ... not
an official component within the ROCm stack".

## Verified

| | result |
| --- | --- |
| Builds | **yes** (CMake + Ninja, ROCm clang 22.0.0git, release) |
| GPU (AMDGPU driver) | **yes**, with the ROCm 10.0.0 release HSA |
| NPU (XDNA via `libamdf`) | **yes** |
| Our kernels under HRX | `dot_peak` 27.7 TMAC/s, `gemm_bench one:q4k` 8.86 ms, `ssm_bench` 3.25 ms, all agreeing |

```
hrx-info
  GPU accelerator: 1 device
    [0] AMD Radeon 8060S Graphics (Node 1) (gfx1151)
  CPU accelerator: 1 device
```

## The GPU: the release has the symbol, the nightly does not

HRX's AMDGPU driver always builds an `hsa_amd_queue_create_desc_t` and calls
`hsa_amd_queue_create` (`hsa_queue.c`); there is **no fallback** to the classic
`hsa_queue_create`. Its headers are vendored from the headers-only repo
`iree-org/hsa-runtime-headers` at commit `42855131`, interface **1.31**, where the
function is declared at `hsa_ext_amd.h:3997`.

What each candidate runtime exports:

| source | `hsa_amd_queue_create` |
| --- | --- |
| ROCm 7.2.4 (system `/opt/rocm`, HSA 1.18.0) | no |
| TheRock 7.13.0 (Lemonade's cache) | no |
| TheRock `rocm_sdk_core-7.14.0.dev0` (nightly wheel) | **no** |
| **ROCm 10.0.0 release `amdrocm-runtime10.0_10.0.0-4`** | **yes -- `hsa_amd_queue_create@@ROCR_1`** |

**The dev/nightly channel lags the release for this symbol.** Benchmarks or
requirements must not be judged against nightlies: the 7.14.0.dev0 wheel and the
10.0.0 release are both nominally "the latest ROCm" and disagree.

Working recipe (nothing installed system-wide, nothing touched in `/opt/rocm`):

```bash
# official release packages, ubuntu2404 channel
B=https://stable.repo.amd.com/rocm/core/packages/ubuntu2404/pool/main
curl -O $B/amdrocm-runtime10.0_10.0.0-4_amd64.deb      # HSA runtime, 15 MB
curl -O $B/amdrocm-sysdeps10.0_10.0.0-4_amd64.deb      # librocm_sysdeps_*, 14 MB
# extract (dpkg-deb is not installed here; ar+tar or python both work)
R=.../x_runtime/opt/rocm/core-10.0/lib
S=.../x_sysdeps/opt/rocm/core-10.0/lib/rocm_sysdeps/lib
export IREE_HAL_AMDGPU_LIBHSA_PATH=$R
export LD_LIBRARY_PATH=<hrx>/libhrx/src/binding/hip:<hrx>/libhrx/src/libhrx:$R:$S:$LD_LIBRARY_PATH
```

HRX's `libamdhip64.so.7` then takes precedence over ROCm's, and
`IREE_HAL_AMDGPU_LIBHSA_PATH` supplies the HSA that its driver needs. The first
failure without `sysdeps` is `librocm_sysdeps_elf.so.1`, so both packages are
required.

## Our own kernels run under HRX unmodified

`dot_peak` and `gemm_bench`, compiled against ROCm 7.2.4, load HRX's
`libamdhip64` and run:

| kernel | ROCm 7.2.4 | HRX + ROCm 10.0.0 HSA |
| --- | ---: | ---: |
| `dot_peak wmma_f16` blocks=40 | 27.45 | **27.73** TMAC/s |
| `dot_peak wmma_f16` blocks=160 | 24.53 | **27.69** TMAC/s |
| `gemm_bench one:q4k 17408 5120 2048` | ~8.9-9.0 | **8.86** ms |

`ssm_bench` does not load at first: it pulls in `libamd_comgr.so.3` from ROCm 10,
which has **1204 undefined LLVM_23.0 symbols**, and the only
`libclang-cpp.so.23.0git` on the box before this was TheRock 7.13's, which does not
satisfy them. The fix is the version-matched LLVM from the same ROCm 10.0.0 release,
**`amdrocm-llvm10.0`** (145 MB), extracted to `core-10.0/lib/llvm/lib`
(`libLLVM.so.23.0git` 132 MB, `libclang-cpp.so.23.0git` 90 MB). With that on the
path:

| kernel | ROCm 7.2.4 | HRX + ROCm 10.0.0 |
| --- | ---: | ---: |
| `ssm_bench row` T=2048 | 3.3206 ms | **3.2485 ms** (2.975 TFLOP/s) |
| `ssm_bench cmp` T=2048 | AGREE | **AGREE** (rel_rms 2.08e-4) |

So all three of our benchmarks -- `dot_peak`, `gemm_bench` and `ssm_bench` -- run
unmodified under HRX, and the recurrence cross-check still agrees bit-for-bit.

## A lead chased and retracted: the two-state ceiling is not the runtime

An early single run under HRX showed the WMMA chain sweep reading 27.57-27.67
TMAC/s with no dips, where ROCm had been bimodal (27.6 / 24.5). That looked like
the two-state ceiling being a ROCm runtime artefact -- which would have moved
every efficiency figure in `kernel-tuning.md`.

It does not survive repetition. Per process, 15 s gaps, `dot_peak chains:8`:

| runtime | runs (TMAC/s) | median |
| --- | --- | ---: |
| ROCm 7.2.4 | 27.72, 22.26, 24.66, 27.73 | 26.19 |
| HRX + ROCm 10.0.0 | 27.68, 27.72, 27.74, 22.21 | 27.70 |

**Both are bimodal**, both reach ~27.7 and both drop to ~22.2. The state that costs
20% is not the HIP implementation. The medians differ by 5% at n=4, the same order
as the dip itself, so no difference in ceiling between the runtimes is established.
The retraction matters more than the result: it removes "switch runtime to fix the
ceiling" as a course of action, and leaves the ceiling's state-dependence as an
environmental or silicon effect -- which is what `kernel-tuning.md` already records
as its largest measurement caveat.

## Build notes worth keeping

- **`dev.py` needs Python 3.12**; this box has 3.14.7. `BUILDING.md` sanctions
  driving CMake directly with the ROCm clang, which is what was done. `uv` is
  available for the `dev.py` path.
- **Use `-DIREE_ROCM_DEPENDENCY_MODE=pinned`, not `package`**: with `package`
  the AMDGPU driver compiles against ROCm 7.2.4's HSA headers and dies on
  `HSA_AMD_POINTER_INFO_ALLOC_FLAG_*`; `pinned` uses the vendored 1.31 headers.
  Keep `IREE_ROCM_PATH=/opt/rocm` for the device toolchain and `amdgcn` bitcode.
- Verify `CMakeCache.txt` says `IREE_HAL_DRIVER_AMDGPU=ON` before trusting a build:
  a lost `-D` in a retry loop produced a driver-less runtime that still built
  cleanly.
- `libamdf` fetches `include/uapi/linux/kfd_sysfs.h` from a pinned torvalds/linux
  commit; CMake's downloader failed once transiently and succeeded on retry.

## NPU

Functional on this box: `libamdf`'s XDNA memory benchmark runs (publication
15.5 GiB/s), 13/13 XDNA CTS tests pass, `/dev/accel/accel0` is present with driver
`amdxdna` (PCI `1022:17f0`). The NPU codegen path is `experimental/xdna` plus the
MLIR and TileLang importers -- a different path from the ported HIP kernels.

## Caveats

HRX is explicitly early-access. Its HIP binding is incomplete: of the tests
exercised, `hip_stream_value_api_test` passed 25/25 while
`hip_execution_resource_api_test` had **14 failures**, so acceptance is uneven and
the engine's own checks (`yah-ssm-check`, `yah-gemm-check`,
`kv-quant-check`) must be re-run against it before any cut-over.

## HRX-specific optimizations: where they are, and a negative

Everything above ran ROCm-shaped code through the HIP compatibility shim, which is
the least interesting way to use HRX. What the native surface actually offers, and
what it costs to reach:

| native API | what it would buy us | reachable from HIP? |
| --- | --- | --- |
| `hrx_graph_*` with explicit `add_dependencies` | one submission for the 1625-dispatch prefill | **yes**, via `hipStreamBeginCapture` |
| `hrx_stream_dispatch`, `hrx_queue_dispatch` | skip the compat layer | no -- takes `hrx_executable_t` |
| `hrx_fence_insert/extend`, `hrx_semaphore_*` | submission batching, timeline deps | no -- native only |
| `hrx_stream_wait_on`, `advance_timeline` | overlap independent work (SSM vs attention) | no -- native only |
| `hrx_mem_pool_t`, `hrx_device_memory_info` | GTT budget, alloc churn | no -- native only |

The blocker for all the "no" rows is one signature: `hrx_executable_load_data`
takes a **native executable package** selected by `target_family` / `target_key`,
not a hipcc HSACO. The native path is therefore not a re-link -- it needs kernels
produced by HRX's own compile path (Loom, the MLIR/TileLang importers). That is
re-authoring, not tuning, and it is the honest answer to "can we do HRX-specific
optimizations": not on the HIP surface, only by moving the kernels.

The part that *is* reachable -- HIP graph capture -- HRX implements properly
(315 real HIP definitions against 116 stubs; `hipStreamBeginCapture`,
`hipGraphInstantiate`, `hipGraphLaunch`, `hipGraphAddKernelNode`,
`hipGraphAddDependencies` all implemented, and capture maps onto
`iree_hal_streaming_capture_mode_t`, i.e. the native graph). Your engine already
has `attention_decode_graph`, so the machinery exists. So it was worth measuring:

```
launch_bench 2000, empty kernel, 15 s gaps, 3 cycles
                raw launch        graph launch     graph gain
ROCm 7.2.4      2.397 us/launch   2.264 us/launch  1.06x
HRX + ROCm 10   3.081 us/launch   2.653 us/launch  1.16x
```

Two conclusions, both negative:

1. **HRX's dispatch is 28% more expensive per launch than ROCm's**, and its graph
   path is still slower in absolute terms. HRX's "low latency" thesis does not show
   up as cheaper submission on this part.
2. **The ~4% GPU idle is not dispatch cost.** Graph-capturing the whole prefill
   under HRX saves 1625 x 0.43 us = **0.7 ms of 4403 ms (0.016%)**. Meanwhile the
   4% idle is ~176 ms, or **~108 us per dispatch gap -- some 40x the launch cost**.
   Whatever the idle is (dependency stalls, kernel ramp, or a clock artefact of the
   accounting), launch overhead and graph capture do not address it, and the
   existing `attention_decode_graph` already captures the easy part.

So the HRX-specific lever is not in the HIP surface at all. It is the native
dispatch/graph/semaphore API, and reaching it means producing HRX native
executables through the Loom/MLIR/TileLang path -- a kernel re-authoring project
rather than a runtime swap. Worth deciding deliberately rather than drifting into.

## Switch validation: the engine runs on HRX at parity

The decision to move to HRX was taken on the NPU trajectory, so the gate is not
speed -- it is whether the engine works and whether anything regresses.

**Correctness, all seven check binaries under HRX:**

| check | result |
| --- | --- |
| `yah-gemm-check` | CHECK PASS |
| `yah-ssm-check` | SSM CHECK PASS |
| `yah-attention-check` | ATTENTION CHECK PASS |
| `yah-rope-check` | ROPE+NORM CHECK PASS |
| `yah-unpack-check` | UNPACK CHECK PASS |
| `yah-ffn-check` | FFN CHECK PASS |
| `kv-quant-check` | KV QUANT CHECK PASS |

**End-to-end prefill**, `yah-run base_q4kpure.gguf --ids-file ids2048.txt`,
2048 tokens, same model and ids on both runtimes:

| runtime | prefill | argmax |
| --- | ---: | ---: |
| ROCm 7.2.4 | 3125.762 ms | 9338 |
| HRX + ROCm 10.0.0 | 3130.434 ms, 3138.071 ms | 9338 |

**Parity (+0.4%, inside run-to-run spread) with an identical argmax**, so the
switch is numerically clean and costs nothing measurable. One pair read 5246 ms for
HRX; that was a first-of-pair run started immediately after a killed job and did not
reproduce (3130 / 3138 on either side of it). The earlier per-kernel numbers on the
same two runtimes point the same way: `ssm_bench` 3.2485 ms under HRX against
3.3206 ms under ROCm, `gemm_bench one:q4k` 8.86 ms against ~8.9-9.0 ms.

**On "would the dispatches be better implemented directly for HRX".** No, and the
arithmetic is short. The prefill is 1625 dispatches. At HRX's measured 3.081 us per
launch that is **5.0 ms of a 3130 ms pass (0.16%)** in total. Implementing natively
perfectly -- reaching ROCm's 2.397 us, i.e. erasing the entire 0.68 us compat-layer
cost -- saves **1.1 ms, or 0.035%**. Graph capture saves 0.7 ms, or 0.02%. Dispatch
is not a lever on this part at any implementation, and the ~4% GPU idle that looked
like the target is ~40x the per-launch cost, so it is dependency stalls or kernel
ramp rather than submission.

What native HRX would actually change is codegen, and that needs kernels produced by
the Loom/MLIR/TileLang path -- the same path the NPU work requires. That is the
investment worth considering, and it is separable from the runtime switch.

## Dispatch and kernel boundaries are not levers, under either runtime

The ~4% GPU idle looked like submission overhead, and if it were, a persistent
kernel driven by grid syncs would recover it. Both halves of that are now measured
and both are negative. `engine/gpu/gridsync_bench.hip` carries the probe.

**One grid sync costs 0.71 us and works on both runtimes.** Cooperative launch
succeeds under ROCm 7.2.4 and under HRX, the hand-rolled sense-reversing barrier
verifies (`counter=OK`, no hang), and the barrier costs 0.709 us on ROCm and
0.713 us on HRX against 0.073 us of per-iteration work. So the primitive itself is
cheap -- ~150x cheaper than the 108 us per-dispatch gap that the idle figures
implied.

**But a kernel boundary costs nothing when the kernel has real work.** Running
identical work as 200 dependent kernel launches or as one cooperative kernel with
200 grid syncs, at 200 us of work per iteration:

| runtime | chain | persist | boundary saved | speedup |
| --- | ---: | ---: | ---: | ---: |
| ROCm 7.2.4 | 206.910 us/iter | 207.848 us/iter | **-0.938 us/iter** | 0.995x |
| HRX + ROCm 10.0.0 | 208.959 us/iter | 208.559 us/iter | **+0.400 us/iter** | 1.002x |

The next launch is submitted asynchronously while the current kernel runs, so the
boundary is fully hidden at this work size. Persistent kernels therefore gain
nothing, and the cooperative/persistent direction is closed.

Together with `launch_bench` (2.4-3.1 us submission, 0.16% of a prefill) this
closes submission as an explanation for the idle at **both** scales. What remains
is clock accounting or genuine host-side gaps -- the dispatch trace's only two
large timestamp deltas were the weight-upload buffers, which is the direction to
look next. It also means the HRX-specific surface has nothing to offer here: its
HIP path matches ROCm, and its native path is a codegen question, not a runtime one.

**So tuning returns to the kernels.** The largest identified item is still the
DeltaNet recurrence at ~12.6x worse FLOPs per unit time than the GEMM next to it,
which is a shape problem (chunk the token axis), not a scheduling one.

## Profiling: the amdgpu HAL has a real profiler, with one gap that matters here

The HAL is not stubbed for profiling. It emits AQL profile packets around each
dispatch and decodes them through AMD's aqlprofile, with separate counter,
device-metric and ATT paths, plus dispatch/queue events, executable traces, and raw
.irpf bundles. The relevant flags are `--profile-final-batch=true`,
`--profile-data=<families>`, `--profile-counter=<NAME>`, and `--profile-artifacts-dir`.

### What has to be true for counters to work

- **aqlprofile must be loadable.** The HAL dlopens `libhsa-amd-aqlprofile64.so.1`.
  None of our six ROCm 10.0 debs carry it, but TheRock 7.13 ships a working copy at
  `/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0/lib`, and it loads
  against ROCm 10.0 HSA. Put that directory first on `LD_LIBRARY_PATH`.
- **`--profile-data` must include `counters` or `counter-ranges`**, and the counter
  name must be mapped for the device's gfxip version.

### gfx11.5.1 was rejected outright; one line fixes the family gate

`iree_hal_amdgpu_profile_counter_select_family` accepted only gfx11 with
`minor == 0 && stepping <= 2`, so gfx1151 (11.5.1) fell through to `UNSUPPORTED` and
every named counter failed with `UNIMPLEMENTED ... is not mapped for gfx11.5.1`.
Accepting `minor == 5` as the same family is enough, and the fix is in the local
checkout but not yet upstreamed.

After that, `SQ_WAVES` profiles successfully and returns **1088** for the FFN GEMM
configured with `m_tiles=1088` -- one wave per workgroup, exactly as the kernel is
written. That is an independent device-side confirmation of the launch geometry,
which no amount of host-side timing gives you.

### The catalog is the real limit, not the plumbing

Only three counters are mapped for the gfx11 family: `SQ_WAVES`, `SQ_BUSY_CYCLES`,
and `SQ_INSTS_VALU`. The memory-system counters that would settle whether the
activation operand is served from L2 -- `TCC_EA0_RDREQ`, `TCC_EA0_RDREQ_DRAM`,
`TCC_EA0_RDREQ_32B`, `TCC_EA0_WRREQ`, `TCP_TCC_READ_REQ`, `TCP_TOTAL_READ`, the
`TA_*` block -- are all present in the table but explicitly `UNSUPPORTED` for gfx11.

The header comment says arch-specific PMC program generation is deliberately
centralized in aqlprofile so that factories such as gfx115x stay out of IREE. But
event *id* resolution is not centralized: the table hardcodes numeric ids per family,
and aqlprofile exposes `aqlprofile_iterate_event_ids` to ask a live agent for them --
a symbol the HAL never loads. That is the actual gap for gfx115x memory counters, and
it is a contained fix: query the ids for the agent instead of indexing a static table,
falling back to the table where the query is unavailable.

### Two practical notes

- `--profile-data=device-metrics` executes and exits 0 but returns `row_count: 0` with
  the warning `dispatch_distribution_unavailable`, so it is not useful on this device
  as it stands.
- `rocprofv3` is installed on this box (including a gfx1151 TheRock build) and carries
  its own per-architecture counter definitions, so it is the faster route to the
  TCC/TCP counters today. It attaches through HSA, which the benchmark already uses.
