#!/usr/bin/env bash
# Builds engine/build/loomhip: runs a Loom hsaco through HIP so rocprofv3 (PMC, occupancy, ATT) can see it. HRX dispatches are invisible to rocprofv3.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/engine/build"
hipcc -std=c++20 -O2 --offload-arch=gfx1151 "$root/engine/gpu/loomhip.hip" -o "$root/engine/build/loomhip"
echo "built $root/engine/build/loomhip"
