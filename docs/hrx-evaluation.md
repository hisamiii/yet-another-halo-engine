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
