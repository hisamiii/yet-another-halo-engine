#!/usr/bin/env bash
# Builds the release binaries and their bundle (engine/release/bundle.py build) on Ubuntu 24.04, the oldest glibc (2.39)
# the bundle runs on. The CI release job (.github/workflows/release.yml) runs it; on a desktop, run it in an Ubuntu
# 24.04 container. HAL sets are added after, on a gfx1151 box (bundle.py hal, then smoke.py, then bundle.py pack).
#
#   engine/release/build.sh <version> <out dir>
#
# Needs: g++-14, cmake, ninja, git, curl, python3, patchelf, libicu-dev; and the copyright files of the Ubuntu packages
# whose upstream projects the ROCm sysdeps are (libdrm2, libelf1t64, libnuma1, zlib1g, libzstd1, liblzma5, libbz2-1.0).
set -euo pipefail
version="${1:?usage: build.sh <version> <out dir>}"
out="$(mkdir -p "${2:?usage: build.sh <version> <out dir>}" && cd "$2" && pwd)"
root="$(cd "$(dirname "$0")/../.." && pwd)"
hrx="${YAH_HRX:-$root/external/hrx}"
ref="$(awk '$1 == "ref" { print $2 }' "$root/engine/hrx/PIN")"
repo="$(awk '$1 == "repo" { print $2 }' "$root/engine/hrx/PIN")"

"$root/engine/hrx-env.sh" --fetch
# Only the pinned commit (the full history is large); bootstrap.sh then applies the patches.
if [ ! -e "$hrx" ]; then
  git init --quiet "$hrx"
  git -C "$hrx" remote add origin "$repo"
  git -C "$hrx" fetch --quiet --depth 1 origin "$ref"
  git -C "$hrx" checkout --quiet --detach FETCH_HEAD
fi
YAH_HRX_TARGETS=libhrx/src/libhrx/libhrx.so "$root/engine/hrx/bootstrap.sh"
# znver5: every gfx1151 host is a Zen 5 Strix Halo. libstdc++ and libgcc go in statically: the target's may be older.
CXX=g++-14 YAH_MARCH=znver5 YAH_LDFLAGS="-static-libstdc++ -static-libgcc" "$root/engine/build_hrx.sh"

lic="$out/licenses"
mkdir -p "$lic"
curl -sfL --max-time 120 -o "$lic/rocr-runtime.txt" \
  https://raw.githubusercontent.com/ROCm/ROCR-Runtime/rocm-7.0.0/LICENSE.txt
for pkg in libdrm2 libelf1t64 libnuma1 zlib1g libzstd1 liblzma5 libbz2-1.0 "libicu$(dpkg -l 'libicu[0-9]*' | awk '/^ii/ { sub(/^libicu/, "", $2); sub(/:.*/, "", $2); print $2; exit }')"; do
  cp "/usr/share/doc/$pkg/copyright" "$lic/$pkg.txt"
done
python3 "$root/engine/release/bundle.py" build "$out" "$version" --licenses "$lic"
