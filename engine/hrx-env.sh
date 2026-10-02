#!/usr/bin/env bash
# HRX runtime environment for the YAH engine.
#
#   source engine/hrx-env.sh      select HRX for this shell
#   engine/hrx-env.sh --check     verify the install and print what resolves
#   engine/hrx-env.sh --fetch     download and extract the pinned packages
#
# HRX is an alternative HIP runtime (libhrx.so plus a compatible libamdhip64.so) with an XDNA/NPU HAL driver.
# Selecting it is an environment change only: /opt/rocm is not touched and ROCm stays usable in another shell.
#
# PINNED: the AMD ROCm Core SDK 10.0.0 release (ROCm 7.14.x). Do not use TheRock nightlies.
# HRX calls hsa_amd_queue_create; only the 10.0.0 release exports it (see docs/build-and-run.md).
#
# Three packages, ~174 MB, extracted under $YAH_HRX_ROOT (no system install):
#   amdrocm-runtime10.0   HSA runtime; carries hsa_amd_queue_create@@ROCR_1
#   amdrocm-sysdeps10.0   librocm_sysdeps_* (the HSA library links against them)
#   amdrocm-llvm10.0      LLVM 23 matching libamd_comgr.so.3

YAH_HRX_REPO=https://stable.repo.amd.com/rocm/core/packages/ubuntu2404/pool/main
YAH_HRX_PKGS="amdrocm-runtime10.0_10.0.0-4_amd64.deb amdrocm-sysdeps10.0_10.0.0-4_amd64.deb amdrocm-llvm10.0_10.0.0-4_amd64.deb"

if [ -z "$YAH_HRX_ROOT" ]; then YAH_HRX_ROOT=/home/q/rocm10; fi
if [ -z "$YAH_HRX_BUILD" ]; then YAH_HRX_BUILD=/home/q/hrx/build/cmake; fi

yah_hrx_libhsa_dir="$YAH_HRX_ROOT/x_runtime/opt/rocm/core-10.0/lib"
yah_hrx_llvm_dir="$YAH_HRX_ROOT/x_llvm/opt/rocm/core-10.0/lib/llvm/lib"
yah_hrx_sysdeps_dir="$YAH_HRX_ROOT/x_sysdeps/opt/rocm/core-10.0/lib/rocm_sysdeps/lib"
yah_hrx_hip_dir="$YAH_HRX_BUILD/libhrx/src/binding/hip"
yah_hrx_libhrx_dir="$YAH_HRX_BUILD/libhrx/src/libhrx"

yah_hrx_fetch() {
  local p d
  mkdir -p "$YAH_HRX_ROOT"
  cd "$YAH_HRX_ROOT" || return 1
  for p in $YAH_HRX_PKGS; do
    [ -f "$p" ] || curl -sL --max-time 900 -o "$p" "$YAH_HRX_REPO/$p" || return 1
    d=""
    case "$p" in
      amdrocm-runtime10.0*) d=x_runtime ;;
      amdrocm-sysdeps10.0*) d=x_sysdeps ;;
      amdrocm-llvm10.0*)    d=x_llvm ;;
      *) continue ;;
    esac
    [ -d "$d" ] && continue
    echo "hrx-env: extracting $p -> $d"
    # dpkg-deb is not on every host. A deb is an ar archive with control.tar.* and data.tar.*; unpack it in python.
    python3 - "$p" "$d" <<'PY'
import io, os, sys, tarfile
def read_ar(path):
    out = {}
    with open(path, 'rb') as f:
        assert f.read(8) == b'!<arch>\n'
        while True:
            hdr = f.read(60)
            if len(hdr) < 60:
                break
            name = hdr[0:16].decode().strip()
            size = int(hdr[48:58].decode().strip())
            out[name] = f.read(size)
            if size % 2:
                f.read(1)
    return out
deb, dest = sys.argv[1], sys.argv[2]
a = read_ar(deb)
key = [k for k in a if k.startswith('data.tar')][0]
os.makedirs(dest, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(a[key]), mode='r:*').extractall(dest)
PY
  done
  echo "hrx-env: fetched into $YAH_HRX_ROOT"
}

yah_hrx_env() {
  export IREE_HAL_AMDGPU_LIBHSA_PATH="$yah_hrx_libhsa_dir"
  # HRX's libamdhip64 must win over ROCm's; ROCm's own libs stay reachable.
  export LD_LIBRARY_PATH="$yah_hrx_hip_dir:$yah_hrx_libhrx_dir:$yah_hrx_libhsa_dir:$yah_hrx_llvm_dir:$yah_hrx_sysdeps_dir:/opt/rocm/lib:$LD_LIBRARY_PATH"
}

yah_hrx_check() {
  local miss=0 d hsa
  for d in "$yah_hrx_libhsa_dir" "$yah_hrx_llvm_dir" "$yah_hrx_sysdeps_dir" "$yah_hrx_hip_dir"; do
    if [ -d "$d" ]; then echo "  ok      $d"; else echo "  MISSING $d"; miss=1; fi
  done
  if [ "$miss" != 0 ]; then
    echo "  -> run: engine/hrx-env.sh --fetch   (and build HRX into $YAH_HRX_BUILD)"
    return 1
  fi
  yah_hrx_env
  hsa="$yah_hrx_libhsa_dir/libhsa-runtime64.so.1"
  if nm -D --defined-only "$hsa" 2>/dev/null | grep -q hsa_amd_queue_create; then
    echo "  ok      hsa_amd_queue_create present in $hsa"
  else
    echo "  MISSING hsa_amd_queue_create in $hsa <- wrong ROCm: need the 10.0.0 release, not a nightly"
    return 1
  fi
  echo "  HIP     $yah_hrx_hip_dir/libamdhip64.so.7"
  if [ -x "$YAH_HRX_BUILD/libhrx/tools/hrx-info" ]; then
    "$YAH_HRX_BUILD/libhrx/tools/hrx-info" 2>&1 | sed 's/^/  /'
  fi
}

yah_hrx_arg="$1"
if [ -z "$yah_hrx_arg" ]; then yah_hrx_arg=--env; fi
case "$yah_hrx_arg" in
  --fetch) yah_hrx_fetch ;;
  --check) yah_hrx_check ;;
  --env)   yah_hrx_env; echo "hrx-env: HRX selected (libhsa=$yah_hrx_libhsa_dir)" ;;
  *) echo "usage: hrx-env.sh [--env|--check|--fetch]" >&2; exit 2 ;;
esac
