#!/bin/bash
# Usage: bench_with_clock.sh <command...>
# Runs a command, samples the iGPU SCLK during the run, and prints the clock range after the command output.
# SCLK follows the kernel's power draw (three levels; sustained work does not reach the top), so a rate needs its clock.
set -u
SCLK=/sys/class/drm/card1/device/pp_dpm_sclk
if [ ! -r "$SCLK" ]; then
  echo "bench_with_clock: $SCLK is not readable" >&2
  exit 2
fi

out=$(mktemp)
"$@" >"$out" 2>&1 &
pid=$!

samples=0
sum=0
max=0
min=0
while kill -0 "$pid" 2>/dev/null; do
  mhz=$(grep '\*' "$SCLK" 2>/dev/null | grep -oE '[0-9]+Mhz' | grep -oE '[0-9]+' | head -1)
  if [ -n "${mhz:-}" ]; then
    samples=$((samples + 1))
    sum=$((sum + mhz))
    if [ "$samples" -eq 1 ] || [ "$mhz" -gt "$max" ]; then max=$mhz; fi
    if [ "$samples" -eq 1 ] || [ "$mhz" -lt "$min" ]; then min=$mhz; fi
  fi
  sleep 0.05
done
wait "$pid"
rc=$?

cat "$out"
rm -f "$out"
if [ "$samples" -gt 0 ]; then
  printf 'sclk: mean %d MHz, min %d, max %d, %d samples\n' \
    "$((sum / samples))" "$min" "$max" "$samples"
else
  echo 'sclk: no samples'
fi
exit "$rc"