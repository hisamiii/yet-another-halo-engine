#!/usr/bin/env python3
"""Emit an HRX-loadable HAL executable for a Loom kernel at a fixed config.

usage: emit_hal.py <file.loom> <outdir> <symbol=value> [symbol=value ...]

iree-run-loom can emit a HAL executable (--emit-only --emit-hal-executable) but
has no --config flag, and iree-benchmark-loom needs a matching check.case
before it will dispatch. Neither fits a production shape whose case bindings
are small. So this tool rewrites each `config.get @symbol` to a constant, drops
the corresponding config.decl, and lets iree-run-loom emit the kernel with no
dispatch at all. The kernel is therefore built for exactly the given config.

Prints the .hal path, then one line per export from the KERNEL ABI metadata.
"""
import os
import re
import subprocess
import sys
import tempfile

H = "/home/q/hrx"
TR = "/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0"
R = "/home/q/rocm10/x_runtime/opt/rocm/core-10.0/lib"
LLVM_LIB = "/home/q/rocm10/x_llvm/opt/rocm/core-10.0/lib/llvm/lib"
SYSDEPS = "/home/q/rocm10/x_sysdeps/opt/rocm/core-10.0/lib/rocm_sysdeps/lib"


def env():
    e = dict(os.environ)
    e["IREE_HAL_AMDGPU_LIBHSA_PATH"] = R
    e["LD_LIBRARY_PATH"] = ":".join([
        TR + "/lib", H + "/libhrx/src/binding/hip", H + "/libhrx/src/libhrx",
        R, LLVM_LIB, SYSDEPS, "/opt/rocm/lib"])
    return e


def main():
    if len(sys.argv) < 4:
        sys.stderr.write(__doc__)
        return 2
    source, outdir = sys.argv[1], sys.argv[2]
    bindings = {}
    for arg in sys.argv[3:]:
        key, _, value = arg.partition("=")
        if not _:
            sys.stderr.write("bad config binding: %s\n" % arg)
            return 2
        bindings[key.lstrip("@")] = value

    lines = open(source).read().split("\n")
    out = []
    used = set()
    get_re = re.compile(
        r"^(\s*)(%\w+) = config\.get @([\w.]+) : (index|i32|u32|f32)$")
    for line in lines:
        if line.lstrip().startswith("config.decl @"):
            symbol = line.split("@", 1)[1].split(" ")[0]
            if symbol in bindings:
                continue
        m = get_re.match(line.rstrip())
        if m and m.group(3) in bindings:
            indent, name, symbol, ty = m.groups()
            value = bindings[symbol]
            if ty == "index":
                out.append("%s%s = index.constant %s : index" %
                           (indent, name, value))
            else:
                out.append("%s%s = scalar.constant %s : %s" %
                           (indent, name, value, ty))
            used.add(symbol)
        else:
            out.append(line)
    missing = set(bindings) - used
    if missing:
        sys.stderr.write("unused config bindings (symbol not in module): %s\n" %
                         ", ".join(sorted(missing)))

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(source))[0]
    rewritten = os.path.join(tempfile.gettempdir(), stem + "_emit.loom")
    with open(rewritten, "w") as f:
        f.write("\n".join(out) + "\n")

    hal_path = os.path.join(outdir, stem + ".hal")
    target = os.path.join(outdir, stem + ".hsaco")
    cmd = [H + "/build/cmake/loom/src/loom/tools/iree-run-loom/iree-run-loom",
           rewritten, "--device=amdgpu", "--target=amdgpu:gfx11-generic",
           "--emit-only", "--emit-hal-executable=" + hal_path,
           "--emit-target-artifact=" + target]
    result = subprocess.run(cmd, env=env(), capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(hal_path):
        sys.stderr.write(result.stderr[-4000:])
        return 1
    print(hal_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())