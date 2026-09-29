#!/usr/bin/env python3
"""Gate hal_bench on the compiler's own footprint analysis.

usage: safe_bench.py <source.loom> <hal> <m_rows> <k_blocks> <block_bytes>
                     <tokens_per_wg> <token_tiles> <k_split> <gx> <gy> <gz> <config=value>... [--iters=N]

hal_bench derives its buffer sizes from the shape, but "the shape I derived" and
"the footprint the kernel was told to touch" are two different statements, and
checking the first against the second is the whole job. loom-compile already
publishes the second: source_low.memory.roots[].interval_envelope.byte_count.
This compiles the source under the same config, reads that envelope, and refuses
to dispatch unless every operand fits the buffer hal_bench will hand it.

Why it matters here: an over-declared extent on this target does not fault. The
shader reads unmapped VA, never returns, and the wave hangs with no page fault
for the driver to report -- gfx times out, MES stops answering msg=RESET, and
the box resets. Two such mistakes on 2026-09-29 each cost a reboot before this
gate existed.

Buffer order is hal_bench's binding order:
  0 weight (m_rows*k_blocks*block_bytes)   1 grid (2048)
  2 input  (k_blocks*256*tokens*2)         3 wstage (64 MiB)
  4 ostage (64 MiB)                        5 output (m_rows*tokens*4)
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loom_preflight as lp  # noqa: E402

H = "/home/q/hrx"
TR = "/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0"
R = "/home/q/rocm10/x_runtime/opt/rocm/core-10.0/lib"
LLVM_LIB = "/home/q/rocm10/x_llvm/opt/rocm/core-10.0/lib/llvm/lib"
SYSDEPS = "/home/q/rocm10/x_sysdeps/opt/rocm/core-10.0/lib/rocm_sysdeps/lib"
COMPILE = H + "/build/cmake/loom/src/loom/tools/loom-compile/loom-compile"


def env():
    e = dict(os.environ)
    e["IREE_HAL_AMDGPU_LIBHSA_PATH"] = R
    e["LD_LIBRARY_PATH"] = ":".join([
        TR + "/lib", H + "/libhrx/src/binding/hip", H + "/libhrx/src/libhrx",
        R, LLVM_LIB, SYSDEPS, "/opt/rocm/lib"])
    return e


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    if len(args) < 12:
        sys.stderr.write(__doc__)
        return 2
    source, hal = args[0], args[1]
    m_rows, k_blocks, block_bytes = (int(args[i]) for i in range(2, 5))
    tokens_per_wg, token_tiles, k_split = (int(args[i]) for i in range(5, 8))
    gx, gy, gz = (int(args[i]) for i in range(8, 11))
    bindings = args[11:]
    tokens = tokens_per_wg * token_tiles
    iters = 50
    wfile = ""
    for f in flags:
        if f.startswith("--iters="):
            iters = int(f.split("=", 1)[1])
        elif f.startswith("--wfile="):
            wfile = f.split("=", 1)[1]

    # The formulas the kernels themselves use. A split-K arm writes its partials
    # at split*m_rows*tokens, so its output and ostage extents scale with k_split
    # -- that term is exactly what was missing when m_rows*tokens*4 was used for a
    # k_split=4 residual and the kernel wrote 3.9 MiB past the allocation.
    W = m_rows * k_blocks * block_bytes
    IN = k_blocks * 256 * tokens * 2
    OUT = m_rows * k_split * tokens * 4
    size = {0: W, 1: 2048, 2: IN, 3: 64 << 20, 4: 64 << 20, 5: OUT}
    name = {0: "weight", 1: "grid", 2: "input", 3: "wstage", 4: "ostage", 5: "output"}

    text = open(source).read()
    m = re.search(r"kernel\.def.*?(@[A-Za-z0-9_]*)\(", text)
    if not m:
        sys.stderr.write("safe_bench: no kernel.def root in %s\n" % source)
        return 2
    root = m.group(1)

    report = "/tmp/safe_bench_report.json"
    cmd = [COMPILE, source, "--root=" + root, "--target=amdgpu:gfx11-generic",
           "--format=amdgpu-hsaco", "--output=/tmp/safe_bench.hsaco",
           "--compile-report=details", "--compile-report-output=" + report]
    cmd += ["--config=" + b for b in bindings]
    res = subprocess.run(cmd, env=env(), capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write("safe_bench: compile failed\n" + res.stderr[-2000:])
        return 2
    declared = lp.declared_envelopes(report)
    if declared is None:
        sys.stderr.write("safe_bench: report carried no interval envelopes; refusing\n")
        return 3

    bad = []
    print("safe_bench: %s root=%s grid=%dx%dx%d tokens=%d (per_wg=%d x tiles=%d) k_split=%d"
          % (source, root, gx, gy, gz, tokens, tokens_per_wg, token_tiles, k_split))
    print("  %-8s %12s %12s" % ("operand", "declared", "buffer"))
    for i in sorted(size):
        want = declared.get(i, 0)
        have = size[i]
        mark = ""
        if want > have:
            mark = "  <-- OVERRUN"
            bad.append((name[i], want, have))
        print("  %-8s %12d %12d%s" % (name[i], want, have, mark))
    if bad:
        print("safe_bench: REFUSING to dispatch; nothing was submitted.")
        for n, want, have in bad:
            print("  %s: declared %d B > buffer %d B" % (n, want, have))
        return 3
    if (gx * 16 > m_rows or gz > k_blocks or gy != token_tiles
            or k_split < gz or k_split % gz):
        print("safe_bench: REFUSING: grid inconsistent with the shape")
        return 3
    extra = [str(x) for x in (W, IN, OUT, gx, gy, gz, m_rows, k_blocks, tokens, iters)]
    if wfile:
        extra.append(wfile)
    rc = subprocess.call(["/home/q/yah-bin/hal_bench", hal] + extra)
    return rc


if __name__ == "__main__":
    sys.exit(main())
