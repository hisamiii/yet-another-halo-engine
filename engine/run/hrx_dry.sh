#!/usr/bin/env bash
# Run a GPU command against hrx_dry.c instead of libhrx: nothing runs on the GPU, every binding view, graph node and copy
# is checked against its buffer, and the run's memory peak is printed. Exit 3: a check failed. Run it before any new set.
#
# usage: hrx_dry.sh <command...>   (after engine/hrx-env.sh)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/engine/build/hrx_dry"
real=""
IFS=: read -ra dirs <<< "${LD_LIBRARY_PATH:-}"
for d in "${dirs[@]}"; do
  if [ -f "$d/libhrx.so.0" ] && [ "$d" != "$OUT" ]; then real="$d/libhrx.so.0"; break; fi
done
[ -n "$real" ] || { echo "hrx_dry: no libhrx.so.0 on LD_LIBRARY_PATH (source engine/hrx-env.sh)" >&2; exit 2; }
inc="$(dirname "$real")"
while [ "$inc" != / ] && [ ! -f "$inc/libhrx/include/hrx_runtime.h" ]; do inc="$(dirname "$inc")"; done
[ -f "$inc/libhrx/include/hrx_runtime.h" ] || inc="${YAH_HRX:?}"
mkdir -p "$OUT"
if [ ! -f "$OUT/libhrx.so.0" ] || [ "$ROOT/engine/run/hrx_dry.c" -nt "$OUT/libhrx.so.0" ]; then
  cc -std=gnu11 -O1 -g -shared -fPIC -Wall -Wextra -I"$inc/libhrx/include" -o "$OUT/libhrx.so.0" \
    "$ROOT/engine/run/hrx_dry.c" -ldl
fi
HRX_DRY_REAL="$real" LD_LIBRARY_PATH="$OUT:$LD_LIBRARY_PATH" exec "$@"
