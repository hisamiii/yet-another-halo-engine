#!/usr/bin/env bash
# Run a GPU command with the kernel log captured on both sides.
#
# usage: gpu_run.sh <tag> -- <command...>
#   env: YAH_GPU_LOG_DIR   where to write the log (default /home/q/yah-scratch)
#
# A bad dispatch on this target does not fail in-process. An access past an
# allocation reaches unmapped VA and the shader hangs with NO page fault, so
# there is nothing for the driver to report to the process: gfx_0.1.0 times out,
# MES stops answering msg=RESET, the GPU reset fails, and the machine goes down.
# The process prints nothing and the console is gone, so the kernel log is the
# only record of what happened. On 2026-09-29 that cost a reboot and the crash
# was only recoverable after the fact with `journalctl -k -b -1`.
#
# So: snapshot the log before, tail it into the same file for the duration (a
# follower is flushed as it writes, unlike an after-the-fact dump), snapshot
# again after, and always print where the log went. On a fresh boot the previous
# boot's log is still in the journal:
#   doas journalctl -k -b -1 --no-pager | grep -iE 'amdgpu|timeout|reset|MES'
set -uo pipefail

LOGDIR="${YAH_GPU_LOG_DIR:-/home/q/yah-scratch}"
if [ "$#" -lt 3 ] || [ "$2" != "--" ]; then
  echo "usage: gpu_run.sh <tag> -- <command...>" >&2
  exit 2
fi
TAG="$1"; shift 2

# Refuse a driver binary older than its source. On 2026-09-30 an edit to
# loom_forward_pp.cc was "built" with cmake --build, which does not build this
# target (engine/build_hrx.sh does), and printed nothing. The 18:06 binary then
# launched a new attention HAL on the old 16-token grid: twice the workgroups the
# kernel's launch contract declares. Loom had used that contract to drop its
# token clamps, so the extra workgroups read ~48 MiB past q/gate, and the ring hung.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
for pair in "loom_forward_pp:engine/run/loom_forward_pp.cc" "hal_bench:engine/run/hal_bench.cc"; do
  bin="${pair%%:*}"; src="$ROOT/${pair#*:}"
  for arg in "$@"; do
    case "$arg" in
      */"$bin")
        if [ -f "$arg" ] && [ -f "$src" ] && [ "$src" -nt "$arg" ]; then
          echo "gpu_run: $arg is older than $src -- rebuild with engine/build_hrx.sh" >&2
          exit 3
        fi ;;
    esac
  done
done
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOGDIR/gpu-${TAG}-${STAMP}.dmesg.log"

if [ "$(id -u)" = "0" ]; then DMESG="dmesg"; else DMESG="doas dmesg"; fi

{
  echo "### $(date -Is) tag=$TAG"
  echo "### cmd: $*"
  echo "### uptime: $(uptime)"
  echo "### --- dmesg before ---"
} > "$LOG" 2>&1
$DMESG >> "$LOG" 2>&1

# Follower: its writes are visible in the file as they happen.
$DMESG -W > "$LOGDIR/.dmesg-w-${STAMP}.log" 2>&1 &  # -W: new messages only (-w replays the buffer)
FOLLOWER=$!

sync
"$@"
RC=$?

kill "$FOLLOWER" 2>/dev/null
wait "$FOLLOWER" 2>/dev/null

{
  echo "### --- exit code: $RC ---"
  echo "### --- dmesg follower ---"
  cat "$LOGDIR/.dmesg-w-${STAMP}.log" 2>/dev/null
  echo "### --- dmesg after ---"
} >> "$LOG" 2>&1
$DMESG >> "$LOG" 2>&1
rm -f "$LOGDIR/.dmesg-w-${STAMP}.log"

echo "gpu_run: exit=$RC log=$LOG"
# only lines logged during the run (the follower section): the before/after
# snapshots repeat the whole boot's history, old warnings included
DURING="$(sed -n '/^### --- dmesg follower ---/,/^### --- dmesg after ---/p' "$LOG")"
FAULT="$(grep -icE 'timeout|GPU reset|MES failed|wedged|page fault|ring .* reset' <<<"$DURING" || true)"
if [ "$FAULT" != "0" ]; then
  echo "gpu_run: *** $FAULT GPU fault line(s) logged during the run ***"
  grep -iE 'timeout|GPU reset|MES failed|wedged|page fault|ring .* reset' <<<"$DURING" | tail -10
fi
exit $RC
