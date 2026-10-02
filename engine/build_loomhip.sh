#!/usr/bin/env bash
# Builds engine/build/loomhip, which runs a Loom hsaco through HIP so rocprofv3 (PMC, occupancy, ATT) can see it.
# rocprofv3 does not see HRX dispatches.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/engine/build"
hipcc -std=c++20 -O2 --offload-arch=gfx1151 "$root/engine/gpu/loomhip.hip" -o "$root/engine/build/loomhip"
echo "built $root/engine/build/loomhip"
