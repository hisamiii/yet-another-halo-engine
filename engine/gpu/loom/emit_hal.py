#!/usr/bin/env python3
"""Emit an HRX-loadable HAL executable for a Loom kernel at a fixed config.

usage: emit_hal.py <file.loom> <outdir> <symbol=value> [symbol=value ...]

This tool rewrites each `config.get @symbol` to a constant, drops its config.decl, and compiles with loom-compile (no
device: the HAL executable of one kernel is its hsaco). The kernel is built for exactly the given config. Prints the .hal
path.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hrx_paths  # noqa: E402


def specialize(text, bindings, tgt="gfx1151"):
    """The source with each bound `config.get` a constant and its config.decl dropped, for target tgt. Returns
    (text, the bindings no config.get used)."""
    lines = text.split("\n")
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
    text = "\n".join(out) + "\n"
    if tgt != "gfx1151":
        text = text.replace("amdgpu.target<gfx1151>", "amdgpu.target<%s>" % tgt)
    return text, set(bindings) - used


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

    text, missing = specialize(open(source).read(), bindings, os.environ.get("YAH_LOOM_TARGET", "gfx1151"))
    if missing:
        sys.stderr.write("unused config bindings (symbol not in module): %s\n" %
                         ", ".join(sorted(missing)))

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(source))[0]
    # a per-call directory: emits may run in parallel (engine/tune)
    tmpdir = tempfile.mkdtemp(prefix="emit_hal_")
    rewritten = os.path.join(tmpdir, stem + "_emit.loom")
    # YAH_LOOM_TARGET selects the compile target: default gfx1151 (1536-VGPR file, granule 24).
    # gfx11-generic models the 1024-VGPR RDNA3 parts.
    tgt = os.environ.get("YAH_LOOM_TARGET", "gfx1151")
    with open(rewritten, "w") as f:
        f.write(text)
    # YAH_EMIT_KEEP=<dir>: keep the specialized source (for loom-compile --compile-report)
    if os.environ.get("YAH_EMIT_KEEP"):
        shutil.copy(rewritten, os.environ["YAH_EMIT_KEEP"])

    hal_path = os.path.join(outdir, stem + ".hal")
    target = os.path.join(outdir, stem + ".hsaco")
    cmd = [hrx_paths.LOOM_COMPILE, rewritten, "--target=amdgpu:" + tgt, "--format=amdgpu-hsaco", "--output=" + hal_path]
    result = subprocess.run(cmd, env=hrx_paths.env(), capture_output=True, text=True)
    shutil.rmtree(tmpdir, ignore_errors=True)
    if result.returncode == 0 and os.path.exists(hal_path):
        shutil.copy(hal_path, target)
    if result.returncode != 0 or not os.path.exists(hal_path):
        # write the whole diagnostic: the first error is at the head, and a tail cut drops it
        sys.stderr.write(result.stderr)
        return 1
    print(hal_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())