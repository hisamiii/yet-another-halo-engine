#!/usr/bin/env bash
# GPU generation gate: 20 greedy tokens from a fixed prompt must equal the
# recorded reference, token for token. Runs on the HRX-native runner (no HIP).
#
# Usage: generate_gate.sh <model.gguf> [hal-dir]
set -euo pipefail
model="${1:?usage: generate_gate.sh <model.gguf> [hal-dir]}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
bin="$root/engine/build/loom_decode"
hal="${2:-${YAH_HAL:-$root/engine/hal}}"
[[ -x "$bin" ]] || { echo "generate_gate: build loom_decode first (engine/build_hrx.sh)" >&2; exit 2; }
[[ -f "$hal/decode.txt" ]] || { echo "generate_gate: no decode HALs in $hal; run engine/build_hrx.sh <model> $hal" >&2; exit 2; }

want="11751 13 198 760 6511 314 9564 369 19241 13 198 760 6511 314 14898 369 21047 13 198 760"
got="$("$bin" "$model" "$hal" --ids "760 6511 314 9338 369" --gen 20 2>/dev/null \
  | sed -n 's/^generated_ids=//p')"

if [[ "$got" == "$want" ]]; then
  echo "GENERATE GATE PASS"
  echo "  $got"
  exit 0
fi
echo "GENERATE GATE FAIL"
echo "  want: $want"
echo "  got:  $got"
exit 1