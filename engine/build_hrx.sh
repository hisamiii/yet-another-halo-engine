#!/usr/bin/env bash
# Builds the HRX-native runners (loom_forward_pp, loom_decode, hal_run) without HIP or hipcc.
#
#   engine/build_hrx.sh                 build libyah_core, the core tools, the runners and the serving binaries
#   engine/build_hrx.sh <model>         also emit the decode HAL set into engine/hal
#   engine/build_hrx.sh <model> <dir>   emit into <dir>
#
# HRX comes from $YAH_HRX (default external/hrx: the pinned revision plus engine/hrx/patches, made by
# engine/hrx/bootstrap.sh), libraries from its build tree $YAH_HRX_BUILD. Source engine/hrx-env.sh before a run.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
export YAH_HRX="${YAH_HRX:-$root/external/hrx}"
H="${YAH_HRX_BUILD:-$YAH_HRX/build/cmake}"
inc="$YAH_HRX/libhrx/include"
# Build only against the pinned, patched HRX, so anyone can reproduce the binaries.
"$root/engine/hrx/bootstrap.sh" --check >/dev/null || { "$root/engine/hrx/bootstrap.sh" --check; exit 1; }
libhrx="$H/libhrx/src/libhrx"

# Host code: clang, -O3 -march=native. -ffp-contract=off keeps host float results the same as
# without FMA, so outputs stay bit-identical across compilers and flags.
# Release builds (engine/release/build.sh) set CXX, YAH_MARCH (znver5: every gfx1151 host) and YAH_LDFLAGS.
CXX="${CXX:-clang++}"
cxxflags=(-std=c++20 -O3 -march="${YAH_MARCH:-native}" -ffp-contract=off)
read -ra ldflags <<< "${YAH_LDFLAGS:-}"
cache="$root/engine/build/CMakeCache.txt"
if [ -f "$cache" ] && ! grep -q "^CMAKE_CXX_COMPILER:[A-Z]*=$(command -v "$CXX")$" "$cache"; then
  rm -rf "$cache" "$root/engine/build/CMakeFiles"  # the compiler changed: CMake needs a fresh configure
fi
cmake -S "$root/engine" -B "$root/engine/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$(command -v "$CXX")" \
    -DCMAKE_CXX_FLAGS_RELEASE="${cxxflags[*]:1}" -DCMAKE_EXE_LINKER_FLAGS="${YAH_LDFLAGS:-}" >/dev/null
cmake --build "$root/engine/build" -j"$(nproc)" >/dev/null
# The runners below are not CMake targets. Build them here so a stale binary never runs against a new HAL set.
# NPU (libamdf + the xdna loader, static from an HRX build with the xdna targets, engine/hrx/bootstrap.sh): the NPU FFN
# block of the prefill driver (YAH_NPU=1) and the server (--npu). Without them both are GPU-only.
npu_flags=() npu_libs=()
if [ -f "$H/libamdf/libamdf_static.a" ]; then
  for a in experimental/xdna/libiree_experimental_xdna_amdf_status.a experimental/xdna/libiree_experimental_xdna_executable.a \
           libamdf/libamdf_static.a runtime/src/iree/hal/drivers/amd/xdna/image/aie2p/libiree_hal_drivers_amd_xdna_image_aie2p_npu2.a \
           runtime/src/iree/hal/drivers/amd/xdna/image/libiree_hal_drivers_amd_xdna_image_image.a \
           runtime/src/iree/hal/drivers/amd/xdna/image/libiree_hal_drivers_amd_xdna_image_validation.a \
           runtime/src/iree/hal/drivers/amd/xdna/image/libiree_hal_drivers_amd_xdna_image_directory.a \
           runtime/src/iree/hal/drivers/amd/xdna/image/libiree_hal_drivers_amd_xdna_image_tables.a \
           runtime/src/iree/schemas/libiree_schemas_xdna_executable.a runtime/src/iree/hal/libiree_hal_hal.a \
           runtime/src/iree/io/libiree_io_file_handle.a runtime/src/iree/async/libiree_async_async.a \
           runtime/src/iree/base/threading/libiree_base_threading_threading.a \
           runtime/src/iree/base/internal/libiree_base_internal_memory.a runtime/src/iree/hal/memory/libiree_hal_memory_asan.a \
           runtime/src/iree/hal/utils/libiree_hal_utils_platform_topology.a \
           runtime/src/iree/base/internal/libiree_base_internal_sysfs.a runtime/src/iree/base/internal/libiree_base_internal_path.a \
           runtime/src/iree/base/internal/libiree_base_internal_time.a runtime/src/iree/base/libiree_base_base.a; do
    npu_libs+=("$H/$a")
  done
  npu_flags=(-DYAH_WITH_NPU -I"$YAH_HRX" -I"$YAH_HRX/runtime/src" -I"$H/runtime/src" -I"$YAH_HRX/libamdf/include" -I"$H"
             -isystem "$H/_deps/linux_uapi/include" -DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl)
  npu_libs=(-Wl,--start-group "${npu_libs[@]}" -Wl,--end-group -lm -ldl)
fi
# Prefill driver.
"$CXX" "${cxxflags[@]}" -I"$root/engine" -I"$inc" "${npu_flags[@]}" "$root/engine/run/loom_forward_pp.cc" \
    -o "$root/engine/build/loom_forward_pp" "$root/engine/build/libyah_core.a" \
    "${npu_libs[@]}" -L"$libhrx" -lhrx -licuuc -lpthread "${ldflags[@]}"
# Decode driver (tools/emit_decode.py sets).
"$CXX" "${cxxflags[@]}" -I"$root/engine" -I"$inc" "$root/engine/run/loom_decode.cc" \
    -o "$root/engine/build/loom_decode" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread "${ldflags[@]}"
# One-dispatch correctness harness for generated kernels (the tools/*_check.py oracles).
"$CXX" "${cxxflags[@]}" -I"$root/engine" -I"$inc" "$root/engine/run/hal_run.cc" \
    -o "$root/engine/build/hal_run" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread "${ldflags[@]}"
# The Responses API server (engine/serve).
"$CXX" "${cxxflags[@]}" -I"$root/engine" -I"$inc" "${npu_flags[@]}" \
    "$root/engine/serve/yah_server.cc" "$root/engine/serve/chat_template.cpp" "$root/engine/serve/responses.cpp" \
    "$root/engine/serve/chat_completions.cpp" \
    "$root/engine/serve/http.cpp" \
    -o "$root/engine/build/yah_server" "$root/engine/build/libyah_core.a" \
    "${npu_libs[@]}" -L"$libhrx" -lhrx -licuuc -lpthread "${ldflags[@]}"
# GPU test of the engine's multi-turn reuse.
"$CXX" "${cxxflags[@]}" -I"$root/engine" -I"$inc" "${npu_flags[@]}" "$root/engine/tests/engine_reuse.cc" \
    -o "$root/engine/build/engine_reuse" "$root/engine/build/libyah_core.a" \
    "${npu_libs[@]}" -L"$libhrx" -lhrx -licuuc -lpthread "${ldflags[@]}"
# GPU test of speculative decoding through the engine (spec == plain).
"$CXX" "${cxxflags[@]}" -I"$root/engine" -I"$inc" "${npu_flags[@]}" "$root/engine/tests/spec_engine.cc" \
    -o "$root/engine/build/spec_engine" "$root/engine/build/libyah_core.a" \
    "${npu_libs[@]}" -L"$libhrx" -lhrx -licuuc -lpthread "${ldflags[@]}"
# GPU test of the speculative verify against plain decode steps.
"$CXX" "${cxxflags[@]}" -I"$root/engine" -I"$inc" "$root/engine/tests/spec_check.cc" \
    -o "$root/engine/build/spec_check" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread "${ldflags[@]}"
if [ "$#" -ge 1 ]; then
  hal="${2:-$root/engine/hal}"
  python3 "$root/engine/gpu/loom/tools/emit_decode.py" "$1" "$hal"
fi
echo "built $root/engine/build/{loom_forward_pp,loom_decode,hal_run,yah_server,engine_reuse,spec_check,spec_engine} (and the CMake targets)"