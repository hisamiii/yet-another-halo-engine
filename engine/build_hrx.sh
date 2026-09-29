#!/usr/bin/env bash
# Builds the HRX-native engine runner (yah-hrx) with no HIP and no hipcc.
#
#   engine/build_hrx.sh                 build libyah_core, the core tools and yah-hrx
#   engine/build_hrx.sh <model.gguf>    also emit the HAL set into engine/hal
#   engine/build_hrx.sh <model> <dir>   emit into <dir>
#
# The HRX runtime libraries are found under $YAH_HRX_BUILD (default the HRX
# cmake build tree). Source engine/hrx-env.sh before running the binary.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
H="${YAH_HRX_BUILD:-/home/q/hrx/build/cmake}"
inc="/home/q/hrx/libhrx/include"
libhrx="$H/libhrx/src/libhrx"

cmake -S "$root/engine" -B "$root/engine/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$root/engine/build" -j"$(nproc)" >/dev/null
g++ -std=c++20 -O2 -I"$root/engine" -I"$inc" "$root/engine/run/yah_hrx.cc" \
    -o "$root/engine/build/yah-hrx" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread
# Single-kernel timing harness. Built here so it cannot drift from the tree; it
# is not covered by the CMake target and it is the tool whose missing z dimension
# wedged the GPU ring (see engine/run/hal_bench.cc).
g++ -std=c++20 -O2 -I"$root/engine" -I"$inc" "$root/engine/run/hal_bench.cc" \
    -o "$root/engine/build/hal_bench" "$root/engine/build/libyah_core.a" \
    -L"$libhrx" -lhrx -licuuc -lpthread
if [ "$#" -ge 1 ]; then
  hal="${2:-$root/engine/hal}"
  python3 "$root/engine/gpu/loom/tools/emit_decode.py" "$1" "$hal"
fi
echo "built $root/engine/build/yah-hrx"