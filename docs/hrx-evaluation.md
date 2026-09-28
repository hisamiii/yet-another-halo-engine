# HRX evaluation: both halves work; the GPU needs the ROCm 10.0.0 *release* HSA

**Verdict: HRX drives both the NPU and the GPU on this box.** The GPU path needs the
HSA runtime from the **official AMD ROCm 10.0.0 release**, not from TheRock's
nightly/dev channel -- and this document originally concluded the opposite, from
nightly builds only. That correction is the main content here.

Evaluated at \`ROCm/hrx\` commit \`244cd38\` (\`main\`). HRX describes itself as "an
alternative implementation of HIP", installs \`libhrx.so\` plus a HIP compatibility
layer as \`lib/libamdhip64.so\`, and is "early-access runtime infrastructure ... not
an official component within the ROCm stack".

## Verified

| | result |
| --- | --- |
| Builds | **yes** (CMake + Ninja, ROCm clang 22.0.0git, release) |
| GPU (AMDGPU driver) | **yes**, with the ROCm 10.0.0 release HSA |
| NPU (XDNA via \`libamdf\`) | **yes** |
| Our kernels under HRX | \`dot_peak\` 27.73 TMAC/s, \`gemm_bench one:q4k\` 8.86 ms |

\`\`\`
hrx-info
  GPU accelerator: 1 device
    [0] AMD Radeon 8060S Graphics (Node 1) (gfx1151)
  CPU accelerator: 1 device
\`\`\`

## The GPU: the release has the symbol, the nightly does not

HRX's AMDGPU driver always builds an \`hsa_amd_queue_create_desc_t\` and calls
\`hsa_amd_queue_create\` (\`hsa_queue.c\`); there is **no fallback** to the classic
\`hsa_queue_create\`. Its headers are vendored from the headers-only repo
\`iree-org/hsa-runtime-headers\` at commit \`42855131\`, interface **1.31**, where the
function is declared at \`hsa_ext_amd.h:3997\`.

What each candidate runtime exports:

| source | \`hsa_amd_queue_create\` |
| --- | --- |
| ROCm 7.2.4 (system \`/opt/rocm\`, HSA 1.18.0) | no |
| TheRock 7.13.0 (Lemonade's cache) | no |
| TheRock \`rocm_sdk_core-7.14.0.dev0\` (nightly wheel) | **no** |
| **ROCm 10.0.0 release \`amdrocm-runtime10.0_10.0.0-4\`** | **yes -- \`hsa_amd_queue_create@@ROCR_1\`** |

**The dev/nightly channel lags the release for this symbol.** Benchmarks or
requirements must not be judged against nightlies: the 7.14.0.dev0 wheel and the
10.0.0 release are both nominally "the latest ROCm" and disagree.

Working recipe (nothing installed system-wide, nothing touched in \`/opt/rocm\`):

\`\`\`bash
# official release packages, ubuntu2404 channel
B=https://stable.repo.amd.com/rocm/core/packages/ubuntu2404/pool/main
curl -O $B/amdrocm-runtime10.0_10.0.0-4_amd64.deb      # HSA runtime, 15 MB
curl -O $B/amdrocm-sysdeps10.0_10.0.0-4_amd64.deb      # librocm_sysdeps_*, 14 MB
# extract (dpkg-deb is not installed here; ar+tar or python both work)
R=.../x_runtime/opt/rocm/core-10.0/lib
S=.../x_sysdeps/opt/rocm/core-10.0/lib/rocm_sysdeps/lib
export IREE_HAL_AMDGPU_LIBHSA_PATH=$R
export LD_LIBRARY_PATH=<hrx>/libhrx/src/binding/hip:<hrx>/libhrx/src/libhrx:$R:$S:$LD_LIBRARY_PATH
\`\`\`

HRX's \`libamdhip64.so.7\` then takes precedence over ROCm's, and
\`IREE_HAL_AMDGPU_LIBHSA_PATH\` supplies the HSA that its driver needs. The first
failure without \`sysdeps\` is \`librocm_sysdeps_elf.so.1\`, so both packages are
required.

## Our own kernels run under HRX unmodified

\`dot_peak\` and \`gemm_bench\`, compiled against ROCm 7.2.4, load HRX's
\`libamdhip64\` and run:

| kernel | ROCm 7.2.4 | HRX + ROCm 10.0.0 HSA |
| --- | ---: | ---: |
| \`dot_peak wmma_f16\` blocks=40 | 27.45 | **27.73** TMAC/s |
| \`dot_peak wmma_f16\` blocks=160 | 24.53 | **27.69** TMAC/s |
| \`gemm_bench one:q4k 17408 5120 2048\` | ~8.9-9.0 | **8.86** ms |

\`ssm_bench\` does not load under this arrangement: it pulls in \`libamd_comgr\`
from ROCm 10, which wants LLVM 23.0 symbols that the available
\`libclang-cpp.so.23.0git\` (TheRock 7.13) does not provide the same way. That is a
library-path tangle, not an HRX limitation, and it is unresolved.

## A lead worth chasing

Under HRX the WMMA chain sweep read **27.57-27.67 TMAC/s at chains 1-6 with no
dips**. Under ROCm the same sweep is bimodal -- 27.6 in some cycles and 24.5 in
others, with the dips moving between runs -- and that bimodality is recorded in
\`kernel-tuning.md\` as the largest measurement caveat in the document. This is a
single run and must be repeated under the protocol before anything is concluded,
but if it holds, **the two-state ceiling is a ROCm runtime artefact rather than
silicon**, and every efficiency figure in the kernel docs is scored against the
wrong ceiling.

## Build notes worth keeping

- **\`dev.py\` needs Python 3.12**; this box has 3.14.7. \`BUILDING.md\` sanctions
  driving CMake directly with the ROCm clang, which is what was done. \`uv\` is
  available for the \`dev.py\` path.
- **Use \`-DIREE_ROCM_DEPENDENCY_MODE=pinned\`, not \`package\`**: with \`package\`
  the AMDGPU driver compiles against ROCm 7.2.4's HSA headers and dies on
  \`HSA_AMD_POINTER_INFO_ALLOC_FLAG_*\`; \`pinned\` uses the vendored 1.31 headers.
  Keep \`IREE_ROCM_PATH=/opt/rocm\` for the device toolchain and \`amdgcn\` bitcode.
- Verify \`CMakeCache.txt\` says \`IREE_HAL_DRIVER_AMDGPU=ON\` before trusting a build:
  a lost \`-D\` in a retry loop produced a driver-less runtime that still built
  cleanly.
- \`libamdf\` fetches \`include/uapi/linux/kfd_sysfs.h\` from a pinned torvalds/linux
  commit; CMake's downloader failed once transiently and succeeded on retry.

## NPU

Functional on this box: \`libamdf\`'s XDNA memory benchmark runs (publication
15.5 GiB/s), 13/13 XDNA CTS tests pass, \`/dev/accel/accel0\` is present with driver
\`amdxdna\` (PCI \`1022:17f0\`). The NPU codegen path is \`experimental/xdna\` plus the
MLIR and TileLang importers -- a different path from the ported HIP kernels.

## Caveats

HRX is explicitly early-access. Its HIP binding is incomplete: of the tests
exercised, \`hip_stream_value_api_test\` passed 25/25 while
\`hip_execution_resource_api_test\` had **14 failures**, so acceptance is uneven and
the engine's own checks (\`yah-ssm-check\`, \`yah-gemm-check\`,
\`kv-quant-check\`) must be re-run against it before any cut-over.
