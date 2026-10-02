#!/usr/bin/env bash
# Builds the HRX-native runners (loom_forward_pp, loom_decode, hal_bench, hal_run) without HIP or hipcc.
#
#   engine/build_hrx.sh                 build libyah_core, the core tools, the runners and yah_server
#   engine/build_hrx.sh <model.gguf>    also emit the decode HAL set into engine/hal
#   engine/build_hrx.sh <model> <dir>   emit into <dir>
#
# HRX libraries come from $YAH_HRX_BUILD (default: the HRX cmake build tree). Source engine/hrx-env.sh before a run.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
H="${YAH_HRX_BUILD:-/home/q/hrx/build/cmake}"
inc="/home/q/hrx/libhrx/include"
libhrx="$H/libhrx/src/libhrx"

cmake -S "$root/engine" -B "$root/engine/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$root/engine/build" -j"$(nproc)" >/dev/null
# The runners below are not CMake targets. Build them here so a stale binary never runs against a new HAL set.
# Single-kernel timing harness.
g++ -std=c++20 -O2 -I"$root/engine" -I"$inc" "$root/engine/run/hal_bench.cc" \
    -o "$root/engine/build/hal_bench" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread
# Prefill driver.
g++ -std=c++20 -O2 -I"$root/engine" -I"$inc" "$root/engine/run/loom_forward_pp.cc" \
    -o "$root/engine/build/loom_forward_pp" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread
# Decode driver (tools/emit_decode.py sets).
g++ -std=c++20 -O2 -I"$root/engine" -I"$inc" "$root/engine/run/loom_decode.cc" \
    -o "$root/engine/build/loom_decode" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread
# One-dispatch correctness harness for generated kernels (tools/gemv_check.py).
g++ -std=c++20 -O2 -I"$root/engine" -I"$inc" "$root/engine/run/hal_run.cc" \
    -o "$root/engine/build/hal_run" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread
# The Responses API server (engine/serve).
g++ -std=c++20 -O2 -I"$root/engine" -I"$root/engine/third_party" -I"$inc" "$root/engine/serve/yah_server.cc" \
    "$root/engine/serve/chat_template.cpp" "$root/engine/serve/responses.cpp" \
    "$root/engine/third_party/httplib/httplib.cpp" \
    -o "$root/engine/build/yah_server" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread
if [ "$#" -ge 1 ]; then
  hal="${2:-$root/engine/hal}"
  python3 "$root/engine/gpu/loom/tools/emit_decode.py" "$1" "$hal"
fi
echo "built $root/engine/build/{loom_forward_pp,loom_decode,hal_bench,hal_run,yah_server}"