# HRX evaluation: the NPU path works today, the GPU path does not

**Verdict:** HRX is worth adopting, but for the **opposite half** of the machine
from the one we are currently working on. As built on this box it drives the
**NPU** and cannot initialise the **GPU**.

Evaluated at \`ROCm/hrx\` commit \`244cd38\` (\`main\`, 2026-09). HRX describes itself as
"an alternative implementation of HIP", installs \`libhrx.so\` plus a HIP
compatibility layer as \`lib/libamdhip64.so\`, and is "early-access runtime
infrastructure ... not an official component within the ROCm stack".

## What was verified on this machine

| | result |
| --- | --- |
| Builds | **yes**, CMake + Ninja, ROCm clang 22.0.0git, release |
| GPU (AMDGPU driver) | **no** -- driver cannot initialise |
| NPU (XDNA via \`libamdf\`) | **yes** -- memory benchmark runs, 13/13 CTS pass |
| \`hrx-info\` | sees the CPU task device only |
| NPU hardware | \`/dev/accel/accel0\`, driver \`amdxdna\`, PCI \`1022:17f0\` |

XDNA evidence, with \`LD_LIBRARY_PATH\` pointing at the build tree:

\`\`\`
libamdf/benchmarks/xdna/memory_benchmark
  XdnaMemory/Allocation/1048576/real_time      234712 ns
  XdnaMemory/Publication/1048576/real_time      62844 ns   15.54 GiB/s
libamdf/libamdf_cts_xdna_xdna_extension_shared_bin
  [  PASSED  ] 13 tests.
\`\`\`

## The GPU blocker, exactly

\`hrx-info\` reports:

\`\`\`
GPU accelerator: unavailable (NOT_FOUND; symbol 'hsa_amd_queue_create' not found in
library; using /opt/rocm/lib/libhsa-runtime64.so.1)
\`\`\`

HRX's vendored HSA headers document \`hsa_amd_queue_create\` as HSA interface
**1.26**. What is installed:

| source | HSA | exports \`hsa_amd_queue_create\` |
| --- | --- | --- |
| \`/opt/rocm\` (ROCm 7.2.4) | 1.18.0 | no |
| Lemonade's TheRock \`gfx1151-7.13.0\` | 1.21.0 | no |

So the GPU driver needs an HSA runtime newer than anything on the box.

**And grabbing the newest ROCm does not fix it.** The chain, checked end to end:

| source | HSA interface | has \`hsa_amd_queue_create\` |
| --- | --- | --- |
| ROCm 7.2.4 (system \`/opt/rocm\`) | 1.18 | no |
| TheRock 7.13.0 (Lemonade's cache) | 1.21 | no |
| **ROCm 7.14 / TheRock 10.0-era, \`rocm_sdk_core-7.14.0.dev0\`** | **1.21** | **no** |
| ROCR-Runtime \`amd-staging\` / \`develop\` | 1.12 / 1.19 | no (only the \`_flag_t\` enum) |
| **HRX's vendored headers** | **1.31** | **yes, declared** |

"ROCm Core SDK 10.0.0" is TheRock's *product* version; its component ROCm version is
7.14.x, and its runtime is HSA 1.21. 7.14 does ship the \`hsa_amd_queue_create_flag_t\`
enum but not the function. HRX takes its headers from a **headers-only** repo,
\`iree-org/hsa-runtime-headers\` at commit \`42855131\`, which is at interface 1.31 --
ahead of every published *implementation*. There is nothing to load.

HRX also has no fallback: \`hsa_queue.c\` always builds an
\`hsa_amd_queue_create_desc_t\` and calls \`hsa_amd_queue_create\`; it never falls back
to the classic \`hsa_queue_create\`, which every version above does export. The
comments even say the descriptor mask "is not honored by all ROCr implementations
exposing hsa_amd_queue_create", so the code only supports ROCr that has it.

**The practical route to GPU-on-HRX is therefore a patch, not a download**: add a
fallback in \`hsa_queue.c\` to the classic \`hsa_queue_create\` when the descriptor
function is absent, losing whatever the descriptor-only fields express (engine type,
explicit queue sizing, priority). Whether that is faithful enough depends on whether
descriptor creation also needs new **kernel** support: this box runs
\`7.3.0-rc4-perfopt\` and \`kfd_ioctl.h\` still exposes only the classic
\`AMDKFD_IOC_CREATE_QUEUE\`/\`DESTROY_QUEUE\`/\`UPDATE_QUEUE\`.

## Two build notes worth keeping

- **\`dev.py\` needs Python 3.12**; this box has 3.14.7. \`BUILDING.md\` says
  "Anything \`dev.py\` does must also be possible with the underlying tools
  directly", so drive CMake directly with the ROCm clang. \`uv\` is available if
  the \`dev.py\` path is wanted.
- **Use \`-DIREE_ROCM_DEPENDENCY_MODE=pinned\`, not \`package\`.** With
  \`package\` the AMDGPU driver is compiled against ROCm 7.2.4's HSA headers and
  fails on the missing \`HSA_AMD_POINTER_INFO_ALLOC_FLAG_*\`; \`pinned\` uses the
  vendored HSA headers and builds. \`IREE_ROCM_PATH\` should still point at
  \`/opt/rocm\` for the device toolchain and \`amdgcn\` bitcode.
- \`libamdf\` fetches \`include/uapi/linux/kfd_sysfs.h\` from a pinned torvalds/linux
  commit; CMake's downloader failed once transiently (the URL returns 200 to curl)
  and succeeded on retry.

## What this means for the plan

The split is convenient: **NPU work can start on HRX now, GPU work stays on ROCm.**

- One runtime does not have to serve both. HRX installs as \`libhrx.so\` plus a
  compat \`libamdhip64.so\`, selected by \`LD_LIBRARY_PATH\`, so the existing ROCm
  build is untouched.
- For the GPU, HRX changes **codegen nothing**: it is a runtime, so the 192-VGPR
  workgroup cap, the 27.6 TMAC/s WMMA ceiling, VALU counts, the reduction chain and
  the 64 KiB LDS budget are all unaffected. The only GPU effect available is
  dispatch/launch overhead.
- For the NPU, expect a **different codegen path** -- \`experimental/xdna\`, \`libamdf\`
  and the MLIR and TileLang importers (\`requirements-importers-mlir\`,
  \`requirements-importers-tilelang\`). The ported HIP kernels do not carry over.
