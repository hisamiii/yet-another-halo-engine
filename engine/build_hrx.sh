#!/usr/bin/env bash
# Builds the HRX-native runners (loom_forward_pp, loom_decode, hal_bench, hal_run) without HIP or hipcc.
#
#   engine/build_hrx.sh                 build libyah_core, the core tools, the runners and the serving binaries
#   engine/build_hrx.sh <model.gguf>    also emit the decode HAL set into engine/hal
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

# Host code: clang, -O3 -march=native. -ffp-contract=off keeps float results (e.g. the host embedding dequant) the same as
# without FMA, so outputs stay bit-identical across compilers and flags.
CXX="${CXX:-clang++}"
cxxflags=(-std=c++20 -O3 -march=native -ffp-contract=off)
cache="$root/engine/build/CMakeCache.txt"
if [ -f "$cache" ] && ! grep -q "^CMAKE_CXX_COMPILER:[A-Z]*=$(command -v "$CXX")$" "$cache"; then
  rm -rf "$cache" "$root/engine/build/CMakeFiles"  # the compiler changed: CMake needs a fresh configure
fi
cmake -S "$root/engine" -B "$root/engine/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$(command -v "$CXX")" \
    -DCMAKE_CXX_FLAGS_RELEASE="${cxxflags[*]:1}" >/dev/null
cmake --build "$root/engine/build" -j"$(nproc)" >/dev/null
# The runners below are not CMake targets. Build them here so a stale binary never runs against a new HAL set.
# Single-kernel timing harness.
"$CXX" "${cxxflags[@]}" -I"$root/engine" -I"$inc" "$root/engine/run/hal_bench.cc" \
    -o "$root/engine/build/hal_bench" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread
# Prefill driver.
"$CXX" "${cxxflags[@]}" -I"$root/engine" -I"$inc" "$root/engine/run/loom_forward_pp.cc" \
    -o "$root/engine/build/loom_forward_pp" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread
# Decode driver (tools/emit_decode.py sets).
"$CXX" "${cxxflags[@]}" -I"$root/engine" -I"$inc" "$root/engine/run/loom_decode.cc" \
    -o "$root/engine/build/loom_decode" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread
# Tile-GEMM variant bench for the autotuner (engine/tune).
"$CXX" "${cxxflags[@]}" -I"$root/engine" -I"$inc" "$root/engine/run/gemm_bench.cc" \
    -o "$root/engine/build/gemm_bench" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread
# One-dispatch correctness harness for generated kernels (tools/gemv_check.py).
"$CXX" "${cxxflags[@]}" -I"$root/engine" -I"$inc" "$root/engine/run/hal_run.cc" \
    -o "$root/engine/build/hal_run" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread
# The Responses API server (engine/serve).
"$CXX" "${cxxflags[@]}" -I"$root/engine" -I"$root/engine/third_party" -I"$inc" "$root/engine/serve/yah_server.cc" \
    "$root/engine/serve/chat_template.cpp" "$root/engine/serve/responses.cpp" \
    "$root/engine/third_party/httplib/httplib.cpp" \
    -o "$root/engine/build/yah_server" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread
# Terminal chat client for yah_server.
"$CXX" "${cxxflags[@]}" -I"$root/engine" -I"$root/engine/third_party" "$root/engine/serve/yah_chat.cc" \
    "$root/engine/third_party/httplib/httplib.cpp" -o "$root/engine/build/yah_chat" -lpthread
# Serving tests: the chat template golden and yah_server --fake over HTTP. No GPU, no libhrx.
"$CXX" "${cxxflags[@]}" -I"$root/engine" -I"$root/engine/third_party" "$root/engine/serve/yah_serve_test.cc" \
    "$root/engine/serve/chat_template.cpp" "$root/engine/serve/responses.cpp" \
    "$root/engine/third_party/httplib/httplib.cpp" -o "$root/engine/build/yah_serve_test" -lpthread
if [ "$#" -ge 1 ]; then
  hal="${2:-$root/engine/hal}"
  python3 "$root/engine/gpu/loom/tools/emit_decode.py" "$1" "$hal"
fi
echo "built $root/engine/build/{loom_forward_pp,loom_decode,hal_bench,gemm_bench,hal_run,yah_server,yah_chat,yah_serve_test}"