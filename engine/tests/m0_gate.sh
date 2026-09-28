#!/usr/bin/env bash
# M0 gate: the engine's full 64-block text path must emit the same greedy next
# token as the recorded reference on a fixed prompt set. Runs on the HRX-native
# runner (no HIP).
#
# Usage: m0_gate.sh <model.gguf> [hal-dir]
set -euo pipefail
model="${1:?usage: m0_gate.sh <model.gguf> [hal-dir]}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
bin="$root/engine/build/yah-hrx"
hal="${2:-${YAH_HAL:-$root/engine/hal}}"
[[ -x "$bin" ]] || { echo "m0_gate: build yah-hrx first (engine/build_hrx.sh)" >&2; exit 2; }
[[ -f "$hal/norm.hal" ]] || { echo "m0_gate: no HALs in $hal; run engine/build_hrx.sh <model> $hal" >&2; exit 2; }

fail=0
check() {
  local text="$1" ids="$2" want="$3"
  local got
  got="$("$bin" "$model" --hal "$hal" --ids "$ids" --gen 1 2>/dev/null \
    | sed -n 's/^generated_ids=//p' | awk '{print $1}')"
  if [[ "$got" == "$want" ]]; then
    printf 'ok   %-28s -> %s\n' "$text" "$got"
  else
    printf 'FAIL %-28s -> %s (want %s)\n' "$text" "$got" "$want"
    fail=1
  fi
}

check "The capital of France is" "760 6511 314 9338 369" 11751
check "def fibonacci(n):"         "727 73111 1393 1590"  198
check "Once upon a time"          "12162 5028 264 854"    11

if (( fail == 0 )); then echo "M0 GATE PASS"; else echo "M0 GATE FAIL"; fi
exit "$fail"