#!/usr/bin/env bash
# M0 gate: the engine's full 64-block text prefill must emit the same greedy
# next token as the reference engine on a fixed prompt set. Each expected id
# was captured from the reference (`gufo prompt --raw -n 1 -t 0`) and
# confirmed against the text it printed, so the check is on a token, not on a
# floating-point threshold.
#
# Usage: m0_gate.sh <model.gguf> [--rowsplit]
set -euo pipefail

model="${1:?usage: m0_gate.sh <model.gguf> [--rowsplit]}"
shift || true
root="$(cd "$(dirname "$0")/../.." && pwd)"
bin="$root/engine/build/yah-run"
[[ -x "$bin" ]] || { echo "m0_gate: build yah-run first ($bin)" >&2; exit 2; }

export YAH_SSM_ROWSPLIT=${YAH_SSM_ROWSPLIT:-}
if [[ "${1:-}" == "--rowsplit" ]]; then export YAH_SSM_ROWSPLIT=1; fi

fail=0
check() {
  local text="$1" ids="$2" want="$3"
  local got
  got="$("$bin" "$model" --ids "$ids" --max-context 4096 2>/dev/null \
    | sed -n 's/^argmax=\([0-9]*\)$/\1/p')"
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
