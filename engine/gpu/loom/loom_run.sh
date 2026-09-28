#!/usr/bin/env bash
# Run one Loom check.case and its benchmark under ONE config, guarded.
#
# usage: loom_run.sh <file.loom> <case> <benchmark|-> <config> [config...]
#   e.g. loom_run.sh yah_qdq_f32.loom @yah_qdq_full @yah_qdq_full_bench yah_qdq.blocks=10240
#
# This exists because a config mismatch is not a soft error on this target. A
# kernel cannot query its operand size (buffer.length has no AMDGPU target-low
# contract), so the config is the only size channel; over-declaring it produces
# out-of-bounds global accesses that fault UTCL2 and wedge the gfx ring until the
# watchdog resets the GPU. Benchmarking a validated 1024-element case with the
# 10240-block config is what took this box down (docs/gpu-ring-hang-qdq.md).
#
# Two rules are enforced here rather than left to discipline:
#   1. correctness and timing always run under the same config;
#   2. tools/loom_preflight.py refuses any config whose declared operand
#      footprint exceeds what the case binds.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TR=/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0
R=/home/q/rocm10/x_runtime/opt/rocm/core-10.0/lib
LLVM_LIB=/home/q/rocm10/x_llvm/opt/rocm/core-10.0/lib/llvm/lib
SYSDEPS=/home/q/rocm10/x_sysdeps/opt/rocm/core-10.0/lib/rocm_sysdeps/lib
H=/home/q/hrx
export IREE_HAL_AMDGPU_LIBHSA_PATH="$R"
export LD_LIBRARY_PATH="$TR/lib:$H/libhrx/src/binding/hip:$H/libhrx/src/libhrx:$R:$LLVM_LIB:$SYSDEPS:/opt/rocm/lib"

COMPILE="$H/build/cmake/loom/src/loom/tools/loom-compile/loom-compile"
BENCH="$H/build/cmake/loom/src/loom/tools/iree-benchmark-loom/iree-benchmark-loom"

SOURCE=${1:?usage: loom_run.sh <file.loom> <case> <benchmark|-> <config> [config...]}
CASE=${2:?missing case}
BENCH_NAME=${3:?missing benchmark name, or - to skip timing}
shift 3

CONFIG_FLAGS=()
for config in "$@"; do
  CONFIG_FLAGS+=("--config=$config")
done

ROOT=$(sed -n 's/.*kernel\.def.*\(@[A-Za-z0-9_]*\)(.*/\1/p' "$SOURCE" | head -1)
if [ -z "$ROOT" ]; then echo "could not find a kernel.def symbol in $SOURCE"; exit 1; fi

echo "== compile $SOURCE root=$ROOT ${CONFIG_FLAGS[*]:-}"
REPORT=$(mktemp /tmp/loom_preflight_report.XXXXXX.json)
if ! "$COMPILE" "$SOURCE" --root="$ROOT" --target=amdgpu:gfx11-generic --format=amdgpu-hsaco \
    "${CONFIG_FLAGS[@]}" --output=/tmp/loom_run.hsaco \
    --compile-report=details --compile-report-output="$REPORT" >/tmp/loom_run_compile.log 2>&1; then
  grep -E 'error' /tmp/loom_run_compile.log | head -5
  echo "compile failed, see /tmp/loom_run_compile.log"
  exit 1
fi
echo "   compiled ok"

echo "== preflight $CASE"
python3 "$HERE/tools/loom_preflight.py" "$SOURCE" "$CASE" "$REPORT"

echo "== correctness $CASE"
timeout 300 "$BENCH" "$SOURCE" --device=amdgpu --target=amdgpu:gfx11-generic \
  "${CONFIG_FLAGS[@]}" --case="$CASE" --measure=case_end_to_end \
  --iterations=1 --warmup-iterations=0 --batch-size=1 --min-time-ms=0 \
  --max-batches=1 --input-ring-count=1 --output=/tmp/loom_run_check.json >/dev/null 2>&1 || true
python3 - <<'PY'
import json
rows = json.load(open('/tmp/loom_run_check.json')).get('benchmarks', [])
states = sorted({str(row.get('state')) for row in rows})
print('   state:', ', '.join(states) or 'no benchmarks ran')
if states != ['ok']:
    raise SystemExit('correctness did not pass')
PY

if [ "$BENCH_NAME" != "-" ]; then
  echo "== timing $BENCH_NAME"
  timeout 300 "$BENCH" "$SOURCE" --device=amdgpu --target=amdgpu:gfx11-generic \
    "${CONFIG_FLAGS[@]}" --case="$CASE" --benchmark="$BENCH_NAME" \
    --measure=dispatch_complete --iterations=1 --warmup-iterations=2 \
    --batch-size=1 --min-time-ms=0 --max-batches=1 --input-ring-count=1 \
    --output=/tmp/loom_run_bench.json >/dev/null 2>&1 || true
  python3 - <<'PY'
import json, re
text = json.dumps(json.load(open('/tmp/loom_run_bench.json')))
found = re.findall(r'mean_physical_dispatch_duration_ns[^0-9]*([0-9.]+)', text)
if not found:
    raise SystemExit('no dispatch duration in the report')
ns = float(found[0])
print('   mean dispatch: %.0f ns  (%.6f ms)' % (ns, ns / 1e6))
PY
fi

rm -f "$REPORT"
echo "== done"
