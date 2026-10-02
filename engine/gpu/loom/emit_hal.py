#!/usr/bin/env python3
"""Emit an HRX-loadable HAL executable for a Loom kernel at a fixed config.

usage: emit_hal.py <file.loom> <outdir> <symbol=value> [symbol=value ...]

iree-run-loom can emit a HAL executable (--emit-only --emit-hal-executable) but has no --config flag.
So this tool rewrites each `config.get @symbol` to a constant, drops its config.decl, and lets iree-run-loom emit without a dispatch.
The kernel is built for exactly the given config. Prints the .hal path.
"""
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hrx_paths  # noqa: E402


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
    # YAH_LOOM_TARGET selects the compile target: default gfx1151 (1536-VGPR file, granule 24).
    # gfx11-generic models the 1024-VGPR RDNA3 parts.
    tgt = os.environ.get("YAH_LOOM_TARGET", "gfx1151")
    text = "\n".join(out) + "\n"
    if tgt != "gfx1151":
        text = text.replace("amdgpu.target<gfx1151>", "amdgpu.target<%s>" % tgt)
    with open(rewritten, "w") as f:
        f.write(text)

    hal_path = os.path.join(outdir, stem + ".hal")
    target = os.path.join(outdir, stem + ".hsaco")
    cmd = [hrx_paths.IREE_RUN_LOOM,
           rewritten, "--device=amdgpu", "--target=amdgpu:" + tgt,
           "--emit-only", "--emit-hal-executable=" + hal_path,
           "--emit-target-artifact=" + target]
    result = subprocess.run(cmd, env=hrx_paths.env(), capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(hal_path):
        # write the whole diagnostic: the first error is at the head, and a tail cut drops it
        sys.stderr.write(result.stderr)
        return 1
    print(hal_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())