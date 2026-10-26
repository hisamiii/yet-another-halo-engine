#!/usr/bin/env python3
"""The yah release bundle: the serving binaries with every library they load, and one HAL set pair.

usage: bundle.py build <out dir> <version> [--licenses <dir>]
           from the built tree (engine/build_hrx.sh): <out dir>/yah-<version>-gfx1151/ without hal/
       bundle.py hal <bundle dir> <prefill set> <decode set> [--npu <prefill set with the NPU block>]
           add a HAL set pair (validated on a gfx1151 box) as hal/<ctx>-<kv>/ (its decode set's context and KV
           format: yah_server --ctx / --kv), and the set yah_server --npu runs (prefill-npu/); once per pair
       bundle.py pack <bundle dir>
           check the bundle, then <bundle dir>.tar.xz beside it

The bundle:
  bin/        yah_server, yah_chat (RPATH $ORIGIN/../lib)
  lib/        libhrx, ICU, the HSA runtime with librocprofiler-register and the ROCm sysdeps it loads (in
              lib/rocm_sysdeps/lib, where the HSA runtime's own RPATH looks)
  hal/        <ctx>-<kv>/prefill and decode/ (decode-mtp/: with the MTP layer's kernels), prefill-npu/
              (emit_npu_ffn.py); 32768-kv8 is the default
  licenses/   this repository's, HRX's, the ROCm parts', ICU's and the sysdeps' (--licenses: extra files)
  README.md   engine/release/README.md

build must run where the binaries were built: it copies the libraries they resolve there (ldd), so build them on the
oldest glibc the bundle should run on (engine/release/build.sh: Ubuntu 24.04, glibc 2.39). Needs patchelf.
"""
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BIN = ["yah_server", "yah_chat"]
# Libraries every glibc system has: never bundled.
SYSTEM = re.compile(r"^(linux-vdso|ld-linux|libc|libm|libdl|libpthread|librt|libgcc_s|libstdc\+\+)\.so")


def rocm_prefix():
    rocm = os.environ.get("YAH_ROCM", os.path.join(ROOT, "external", "rocm10"))
    return os.path.join(rocm, "root", "opt", "rocm", "core-10.0")


def hrx_lib_dir():
    hrx = os.environ.get("YAH_HRX", os.path.join(ROOT, "external", "hrx"))
    build = os.environ.get("YAH_HRX_BUILD", os.path.join(hrx, "build", "cmake"))
    return os.path.join(build, "libhrx", "src", "libhrx")


def ldd(path, lib_path=""):
    """{soname: resolved path or None} of path's dependencies (LD_PRELOAD dropped: it is not a dependency)."""
    env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "LD_LIBRARY_PATH")}
    if lib_path:
        env["LD_LIBRARY_PATH"] = lib_path
    out = subprocess.run(["ldd", path], capture_output=True, text=True, env=env).stdout
    deps = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\S+) => (not found|(\S+))", line)
        if m:
            deps[m.group(1)] = m.group(3)
    return deps


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.realpath(src), dst)   # the file itself under the soname, not a symlink
    os.chmod(dst, 0o755)


def patchelf(*args):
    subprocess.run(["patchelf", *args], check=True)


def build(out, version, licenses):
    b = os.path.join(out, f"yah-{version}-gfx1151")
    if os.path.exists(b):
        sys.exit(f"bundle: {b} exists; move it away first")
    lib = os.path.join(b, "lib")
    for name in BIN:
        copy(os.path.join(ROOT, "engine", "build", name), os.path.join(b, "bin", name))
    # what the binaries load: libhrx and ICU (with their own dependencies), nothing a glibc system has
    todo = [os.path.join(ROOT, "engine", "build", n) for n in BIN]
    seen = set()
    while todo:
        for soname, path in ldd(todo.pop(), hrx_lib_dir()).items():
            if SYSTEM.match(soname) or soname in seen:
                continue
            if path is None:
                sys.exit(f"bundle: {soname} not found")
            seen.add(soname)
            copy(path, os.path.join(lib, soname))
            todo.append(path)
    # libhrx dlopens the HSA runtime by directory (yah_server sets IREE_HAL_AMDGPU_LIBHSA_PATH to lib/); copy it with
    # every ROCm library it loads, at their paths relative to the prefix's lib/
    pre = os.path.join(rocm_prefix(), "lib")
    todo, seen = [os.path.join(pre, "libhsa-runtime64.so.1")], set()
    copy(todo[0], os.path.join(lib, "libhsa-runtime64.so.1"))
    while todo:
        for soname, path in ldd(todo.pop()).items():
            if SYSTEM.match(soname) or soname in seen:
                continue
            if path is None or not os.path.realpath(path).startswith(os.path.realpath(pre) + os.sep):
                sys.exit(f"bundle: the HSA runtime's {soname} resolves outside {pre}: {path}")
            seen.add(soname)
            rel = os.path.relpath(os.path.join(os.path.dirname(path), soname), pre)
            copy(path, os.path.join(lib, rel))
            todo.append(path)
    for name in BIN:
        patchelf("--force-rpath", "--set-rpath", "$ORIGIN/../lib", os.path.join(b, "bin", name))
    for f in os.listdir(lib):   # libhrx and ICU find each other beside them
        p = os.path.join(lib, f)
        if os.path.isfile(p) and not f.startswith(("libhsa-runtime64", "librocprofiler-register")):
            patchelf("--force-rpath", "--set-rpath", "$ORIGIN", p)
    lic = os.path.join(b, "licenses")
    os.makedirs(lic)
    for src, name in ((os.path.join(ROOT, "LICENSE"), "yah.txt"), (os.path.join(ROOT, "external", "hrx", "LICENSE"), "hrx.txt"),
                      (os.path.join(rocm_prefix(), "share", "doc", "rocprofiler-register", "LICENSE.md"),
                       "rocprofiler-register.md")):
        shutil.copy(src, os.path.join(lic, name))
    for f in sorted(os.listdir(licenses)) if licenses else []:
        shutil.copy(os.path.join(licenses, f), os.path.join(lic, f))
    shutil.copy(os.path.join(ROOT, "engine", "release", "README.md"), os.path.join(b, "README.md"))
    print(b)


def set_name(decode):
    """(<ctx>-<kv>, decode dir name) from a decode set's decode.txt."""
    ctx, kv, mtp = None, "fp16", False
    for line in open(os.path.join(decode, "decode.txt")):
        f = line.split()
        if f[:1] == ["ctx"]:
            ctx = int(f[1])
        if f[:2] == ["kv", "q"]:
            kv = {("8", "8"): "kv8", ("4", "4"): "kv4", ("8", "4"): "k8v4", ("4", "8"): "k4v8"}[(f[2], f[3])]
        mtp |= f[:1] == ["mtp"]
    return f"{ctx}-{kv}", "decode-mtp" if mtp else "decode"


def hal(b, prefill, decode, npu=None):
    for src, name, marker in ((prefill, "prefill", "dispatch.txt"), (decode, "decode", "decode.txt"),
                              (npu, "NPU prefill", "npu.txt")):
        if src and not os.path.isfile(os.path.join(src, marker)):
            sys.exit(f"bundle: {src} is not a {name} set (no {marker})")
    pair, dec = set_name(decode)
    sets = [(prefill, "prefill"), (decode, dec)] + ([(npu, "prefill-npu")] if npu else [])
    for src, name in sets:
        dst = os.path.join(b, "hal", pair, name)
        if os.path.exists(dst):
            sys.exit(f"bundle: {dst} exists")
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".emit_tmp", "*.log"))


def pack(b):
    b = os.path.abspath(b)
    bad, glibc = [], set()
    for d, _, files in os.walk(b):
        for f in files:
            p = os.path.join(d, f)
            with open(p, "rb") as fh:
                if fh.read(4) != b"\x7fELF":
                    continue
            glibc |= set(re.findall(r"GLIBC_(\d+\.\d+)", subprocess.run(["objdump", "-T", p], capture_output=True,
                                                                           text=True).stdout))
            # with a clean environment every dependency resolves, and none outside the bundle but the system's
            for soname, path in ldd(p).items():
                if path is None:
                    bad.append(f"{os.path.relpath(p, b)}: {soname} not found")
                elif not SYSTEM.match(soname) and not os.path.realpath(path).startswith(b + os.sep):
                    bad.append(f"{os.path.relpath(p, b)}: {soname} from {path}")
    for need in ("hal/32768-kv8/prefill/dispatch.txt", "bin/yah_server", "lib/libhsa-runtime64.so.1"):
        if not os.path.exists(os.path.join(b, need)):
            bad.append(f"missing {need}")
    if bad:
        sys.exit("bundle: " + "\n  ".join(["not self-contained:"] + bad))
    need = max(glibc, key=lambda v: tuple(map(int, v.split("."))))
    tar = b + ".tar.xz"
    subprocess.run(["tar", "-C", os.path.dirname(b), "-cJf", tar, os.path.basename(b)], check=True,
                   env={**os.environ, "XZ_OPT": "-T0 -6"})
    print(f"{tar}: {os.path.getsize(tar) / 2**20:.1f} MB, needs glibc {need}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["build"] and len(a) in (3, 5) and (len(a) == 3 or a[3] == "--licenses"):
        build(os.path.abspath(a[1]), a[2], a[4] if len(a) == 5 else None)
    elif a[:1] == ["hal"] and (len(a) == 4 or len(a) == 6 and a[4] == "--npu"):
        hal(a[1], a[2], a[3], a[5] if len(a) == 6 else None)
    elif a[:1] == ["pack"] and len(a) == 2:
        pack(a[1])
    else:
        sys.exit(__doc__)
