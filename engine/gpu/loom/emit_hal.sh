#!/usr/bin/env bash
# Emit the HRX-loadable HAL executable for one Loom check case.
#
# usage: emit_hal.sh <file.loom> <case> <benchmark> <outdir> [config=value ...]
#
# iree-run-loom can emit a HAL executable (--emit-hal-executable) but has no
# --config flag, so a config-parameterised kernel cannot be built with it.
# iree-benchmark-loom has --config and its artifact bundle carries the same HAL
# executable, so that is the emission path. The bundle is written to <outdir>,
# and the .hal path is printed on stdout.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TR=/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0
R=/home/q/rocm10/x_runtime/opt/rocm/core-10.0/lib
LLVM_LIB=/home/q/rocm10/x_llvm/opt/rocm/core-10.0/lib/llvm/lib
SYSDEPS=/home/q/rocm10/x_sysdeps/opt/rocm/core-10.0/lib/rocm_sysdeps/lib
H=/home/q/hrx
export IREE_HAL_AMDGPU_LIBHSA_PATH="$R"
export LD_LIBRARY_PATH="$TR/lib:$H/libhrx/src/binding/hip:$H/libhrx/src/libhrx:$R:$LLVM_LIB:$SYSDEPS:/opt/rocm/lib"
BENCH="$H/build/cmake/loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom"

SOURCE=${1:?usage: emit_hal.sh <file.loom> <case> <benchmark> <outdir> [config=value ...]}
CASE=${2:?missing case}
BENCH_NAME=${3:?missing benchmark}
OUTDIR=${4:?missing outdir}
shift 4

CONFIG_FLAGS=()
for config in "$@"; do CONFIG_FLAGS+=("--config=$config"); done

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"
"$BENCH" "$SOURCE" --device=amdgpu --case="$CASE" --benchmark="$BENCH_NAME" \
  "${CONFIG_FLAGS[@]}" --iterations=1 --warmup-iterations=0 --min-time-ms=0 \
  --max-batches=1 --artifact-bundle-dir="$OUTDIR" --artifact-bundle-policy=full \
  >/dev/null 2>&1
HAL=$(ls "$OUTDIR"/hal_executables/*.hal | head -1)
printf "%s\n" "$HAL"