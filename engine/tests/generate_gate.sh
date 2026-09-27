#!/usr/bin/env bash
# GPU generation gate: 20 greedy tokens from a fixed prompt must equal the
# reference engine's, token for token. The expected ids were captured from
# `gufo prompt --raw -n 20 -t 0` on the same prompt.
#
# Usage: generate_gate.sh <model.gguf>
set -euo pipefail

model="${1:?usage: generate_gate.sh <model.gguf>}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
bin="$root/engine/build/yah-run"
[[ -x "$bin" ]] || { echo "generate_gate: build yah-run first ($bin)" >&2; exit 2; }

want="11751 13 198 760 6511 314 9564 369 19241 13 198 760 6511 314 14898 369 21047 13 198 760"
got="$("$bin" "$model" --ids "760 6511 314 9338 369" --gen 20 \
  --max-context 4096 2>/dev/null \
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
