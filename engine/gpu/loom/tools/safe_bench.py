#!/usr/bin/env python3
"""Gate hal_bench on the compiler's own footprint analysis.

usage: safe_bench.py <source.loom> <hal> <m_rows> <k_blocks> <block_bytes>
                     <tokens_per_wg> <token_tiles> <k_split>
                     <gx> <gy> <gz> <config=value>... [--iters=N]
                     [--wfile=PATH] [--check-only]

The point of this tool is that the sizes it verifies are the sizes it hands to
the harness. Having the two computed separately is how this went wrong three
times: an input sized for k_blocks=20 reused at 68, an output of
m_rows*tokens*4 for a k_split=4 residual that writes m_rows*k_split*tokens*4, and
a scratch sized 64 MiB in the harness but max(64 MiB, 2*OUT) in the checker. Each
one passed the check and then read past the allocation.

So: compile the source under the same config, read the footprint the compiler
recorded (source_low.memory.roots[].interval_envelope.byte_count, the surface
loom_preflight.py already uses), derive every buffer size, and refuse unless the
declared footprint fits. The data operands (weight, input, output) are checked
against the derived size; wstage and ostage are per-workgroup scratch that is
never read back, so they are sized to the declared need and the check is exact by
construction. Then the same numbers are passed to hal_bench, which allocates
exactly what it is given and computes nothing itself.

Why any of this matters: an extent past the end of an allocation does not fault
on this target. The shader reads unmapped VA, never returns, and hangs with no
page fault for the driver to report -- gfx times out, MES stops answering
msg=RESET, and the box resets. That cost three reboots on 2026-09-29.

--check-only stops after the verification and dispatches nothing.

SAFE_BENCH_WRAP (env, optional) is a whitespace-separated command prefix placed
in front of the hal_bench argv, e.g. "rocprofv3 --pmc SQ_INSTS_VALU --". It exists
so a profiler can be pointed at the SAME argv this tool derived: the wrap never
changes an operand, so the safety argument above is untouched. It is for
profiling only -- the timings it produces are not comparable.
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
SCRATCH_MIN = 64 << 20


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

    iters, wfile, check_only, token_first = 50, "", False, False
    for f in flags:
        if f.startswith("--iters="):
            iters = int(f.split("=", 1)[1])
        elif f.startswith("--wfile="):
            wfile = f.split("=", 1)[1]
        elif f == "--check-only":
            check_only = True
        elif f == "--token-first":
            # The kernel dispatches workgroups(token_tiles, m_groups, 1) instead of
            # (m_groups, token_tiles, 1), so consecutive workgroups are the token
            # tiles of one row-group and share its weight slice in L2. Only the two
            # grid-role sanity checks change; the overrun/envelope checks do not.
            token_first = True

    tokens = tokens_per_wg * token_tiles
    # The formulas the kernels themselves use.
    W = m_rows * k_blocks * block_bytes
    IN = k_blocks * 256 * tokens * 2
    OUT = m_rows * k_split * tokens * 4

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

    # Scratch is sized to the declared need, so the check below is exact for it.
    scratch = max([SCRATCH_MIN, declared.get(3, 0), declared.get(4, 0)])
    size = {0: W, 1: 2048, 2: IN, 3: scratch, 4: scratch, 5: OUT}
    name = {0: "weight", 1: "grid", 2: "input", 3: "wstage", 4: "ostage",
            5: "output"}

    print("safe_bench: %s root=%s grid=%dx%dx%d k_split=%d tokens=%d "
          "(per_wg=%d x tiles=%d)"
          % (source, root, gx, gy, gz, k_split, tokens, tokens_per_wg, token_tiles))
    bad = []
    for i in sorted(size):
        want = declared.get(i, 0)
        have = size[i]
        mark = ""
        if want > have:
            mark = "  <-- OVERRUN"
            bad.append((name[i], want, have))
        print("  %-8s %12d %12d%s" % (name[i], want, have, mark))

    grid_bad = []
    if token_first:
        if gx != token_tiles:
            grid_bad.append("--token-first: gx=%d must equal token_tiles=%d"
                            % (gx, token_tiles))
        if gy * 16 > m_rows:
            grid_bad.append("--token-first: gy=%d covers %d rows but m_rows=%d"
                            % (gy, gy * 16, m_rows))
    else:
        if gx * 16 > m_rows:
            grid_bad.append("gx=%d covers %d rows but m_rows=%d (K splits go on gz)"
                            % (gx, gx * 16, m_rows))
        if gy != token_tiles:
            grid_bad.append("gy=%d must equal token_tiles=%d" % (gy, token_tiles))
    if gz < 1 or gz > k_split:
        grid_bad.append("gz=%d must be in 1..%d" % (gz, k_split))
    if k_split % gz:
        grid_bad.append("k_split=%d is not a multiple of gz=%d" % (k_split, gz))
    for g in grid_bad:
        print("  grid: " + g)

    if bad or grid_bad:
        print("safe_bench: REFUSING to dispatch; nothing was submitted.")
        return 3

    if check_only:
        print("safe_bench: CHECK-ONLY OK -- footprint fits, nothing dispatched.")
        return 0

    extra = [str(x) for x in (W, IN, OUT, size[3], size[4], gx, gy, gz,
                              m_rows, k_blocks, tokens, iters)]
    if wfile:
        extra.append(wfile)
    wrap = os.environ.get("SAFE_BENCH_WRAP", "").split()
    return subprocess.call(wrap + ["/home/q/yah-bin/hal_bench", hal] + extra)


if __name__ == "__main__":
    sys.exit(main())
