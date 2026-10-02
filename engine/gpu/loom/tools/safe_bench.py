#!/usr/bin/env python3
"""Gate hal_bench on the compiler's own footprint analysis.

usage: safe_bench.py <source.loom> <hal> <m_rows> <k_blocks> <block_bytes>
                     <tokens_per_wg> <token_tiles> <k_split>
                     <gx> <gy> <gz> <config=value>... [--iters=N]
                     [--wfile=PATH] [--outfile=PATH] [--check-only]

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
msg=RESET, and the box resets. That cost three reboots on 2026-09-29, and a
fourth when this tool was refused a correct dispatch and then bypassed by hand
sizing instead of being fixed. Two gates were added for that fourth reset:

  * The token stride is READ FROM THE HAL, never taken on trust. dispatch.txt
    beside the HAL records "<hal> <baked_tile> <rowgrp> <tt>" with tt = B/tile,
    and the kernel bakes token_tiles as a compile-time constant, so the token
    stride the kernel walks is tt*baked_tile -- NOT the per-workgroup tile. A
    caller who takes the per-workgroup tile for the token count under-allocates
    the output by tt and hangs the ring. tokens_per_wg/token_tiles must agree
    with the HAL's own line or --check-only refuses.

  * Sizes are keyed by the compiler's own root NAME ('weight', 'input',
    'wstage', 'ostage', 'output'), not by a fixed 6-slot driver position. The
    GEMM family does not share one argument order: the IQ grid/signs formats are
    (weight, grid, [ksigns], input, wstage, ostage, output) = 7 or 6, the rest
    are (weight, input, wstage, ostage, output) = 5. Keying by driver slot
    compared the input's envelope against the 2 KiB grid table and refused a
    good dispatch with a phantom "grid OVERRUN". The kernel's argument count is
    also validated against the variant it claims to be.

--check-only stops after the verification and dispatches nothing.
--outfile=PATH D2H-dumps the raw output after the warmup dispatches. The output
is column-major (out[token * m_rows + row]), so two HALs that claim the same tile
can be dispatched on identical operands and compared elementwise -- a single-HAL
oracle that costs milliseconds instead of a 20 s pp2048 forward.

SAFE_BENCH_WRAP (env, optional) is a whitespace-separated command prefix placed
in front of the hal_bench argv, e.g. "rocprofv3 --pmc SQ_INSTS_VALU --". It exists
so a profiler can be pointed at the SAME argv this tool derived: the wrap never
changes an operand, so the safety argument above is untouched. It is for
profiling only -- the timings it produces are not comparable.
SAFE_BENCH_BIN (env, optional) overrides the harness binary (default
engine/build/hal_bench).
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
HAL_BENCH = os.environ.get("SAFE_BENCH_BIN", os.path.join(HERE, "..", "..", "..", "build", "hal_bench"))
# The argument order each variant of the GEMM family uses, so a root name can be
# turned back into its argument index and the count can be validated.
ROLE_ORDERS = {
    5: ["weight", "input", "wstage", "ostage", "output"],
    6: ["weight", "grid", "input", "wstage", "ostage", "output"],
    7: ["weight", "grid", "ksigns", "input", "wstage", "ostage", "output"],
}


def env():
    e = dict(os.environ)
    e["IREE_HAL_AMDGPU_LIBHSA_PATH"] = R
    e["LD_LIBRARY_PATH"] = ":".join([
        TR + "/lib", H + "/libhrx/src/binding/hip", H + "/libhrx/src/libhrx",
        R, LLVM_LIB, SYSDEPS, "/opt/rocm/lib"])
    return e


def dispatch_geometry(hal):
    """(baked_tile, rowgrp, tt) for a HAL, from dispatch.txt beside it.

    None when there is no dispatch.txt or no line for this HAL. The line is
    written by emit_prefill_pp from the same resolved geometry the emitter baked
    into the kernel, so it is the authority on the token stride.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(hal)), "dispatch.txt")
    if not os.path.isfile(path):
        return None
    want = os.path.basename(hal)
    try:
        for line in open(path):
            p = line.split()
            if len(p) >= 4 and p[0] == want:
                return int(p[1]), int(p[2]), int(p[3])
    except (ValueError, OSError):
        return None
    return None


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

    iters, wfile, outfile, check_only, token_first = 50, "", "", False, False
    for f in flags:
        if f.startswith("--iters="):
            iters = int(f.split("=", 1)[1])
        elif f.startswith("--wfile="):
            wfile = f.split("=", 1)[1]
        elif f.startswith("--outfile="):
            outfile = f.split("=", 1)[1]
        elif f == "--check-only":
            check_only = True
        elif f == "--token-first":
            # The kernel dispatches workgroups(token_tiles, m_groups, 1) instead of
            # (m_groups, token_tiles, 1), so consecutive workgroups are the token
            # tiles of one row-group and share its weight slice in L2. Only the two
            # grid-role sanity checks change; the overrun/envelope checks do not.
            token_first = True

    # The HAL's own geometry, not the caller's word for it. See dispatch_geometry.
    baked_hal = dispatch_geometry(hal)
    geom_bad, geom_warn = [], []
    if baked_hal is None:
        geom_warn.append("no dispatch.txt line for %s -- cannot confirm the "
                         "token stride it was compiled for"
                         % os.path.basename(hal))
    else:
        baked_tile, baked_rowgrp, baked_tt = baked_hal
        if tokens_per_wg != baked_tile or token_tiles != baked_tt:
            geom_bad.append(
                "HAL bakes tile=%d tt=%d (token stride %d) but argv says "
                "per_wg=%d tiles=%d -> tokens=%d"
                % (baked_tile, baked_tt, baked_tile * baked_tt,
                   tokens_per_wg, token_tiles, tokens_per_wg * token_tiles))

    tokens = tokens_per_wg * token_tiles
    # The formulas the kernels themselves use.
    role_size = {
        "weight": m_rows * k_blocks * block_bytes,
        "input": k_blocks * 256 * tokens * 2,
        "output": m_rows * k_split * tokens * 4,
        # Fixed tables the driver binds verbatim: the IQ grid is 512 x i32 and
        # the ksigns table is 128 bytes (see loom_forward_pp run_kstore).
        "grid": 2048,
        "ksigns": 128,
        # Scratch is sized from the declared need below, so its check is exact.
        "wstage": SCRATCH_MIN,
        "ostage": SCRATCH_MIN,
    }

    text = open(source).read()
    m = re.search(r"kernel\.def.*?(@[A-Za-z0-9_]*)\(", text)
    if not m:
        sys.stderr.write("safe_bench: no kernel.def root in %s\n" % source)
        return 2
    root = m.group(1)

    report = "/tmp/safe_bench_report.json"
    cmd = [COMPILE, source, "--root=" + root, "--target=amdgpu:gfx1151",
           "--format=amdgpu-hsaco", "--output=/tmp/safe_bench.hsaco",
           "--compile-report=details", "--compile-report-output=" + report]
    cmd += ["--config=" + b for b in bindings]
    res = subprocess.run(cmd, env=env(), capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write("safe_bench: compile failed\n" + res.stderr[-2000:])
        return 2
    declared, root_name = lp.declared_envelopes(report, with_names=True)
    if declared is None:
        sys.stderr.write("safe_bench: report carried no interval envelopes; refusing\n")
        return 3
    if not root_name:
        sys.stderr.write("safe_bench: report roots carried no source_root names; "
                         "cannot key the sizes by role; refusing\n")
        return 3

    nb = max(root_name) + 1
    present = [root_name[a] for a in sorted(root_name)]
    order_bad = []
    if nb not in ROLE_ORDERS:
        order_bad.append("kernel declares %d arguments (%s); expected one of %s"
                         % (nb, ", ".join(present), sorted(ROLE_ORDERS)))
    else:
        order = ROLE_ORDERS[nb]
        # The compiler omits a root it proved is never touched (a 5-argument
        # kernel drops 'wstage' once the chain stages activations in LDS), so
        # require the names to appear in the expected RELATIVE order rather than
        # to be complete.
        if present != [r for r in order if r in present]:
            order_bad.append("argument order %s is not a subsequence of the "
                             "expected %s" % (present, order))

    # Key every declared envelope by the ROLE the compiler named, so neither the
    # count nor the order is assumed.
    by_role = {}
    for argument, size in declared.items():
        role = root_name.get(argument)
        if role is not None:
            by_role[role] = max(by_role.get(role, 0), size)
    for role in ("wstage", "ostage"):
        if role in by_role:
            role_size[role] = max(SCRATCH_MIN, by_role[role])

    print("safe_bench: %s root=%s args=%d(%s) grid=%dx%dx%d k_split=%d tokens=%d "
          "(per_wg=%d x tiles=%d)"
          % (source, root, nb, ",".join(present), gx, gy, gz, k_split, tokens,
             tokens_per_wg, token_tiles))
    bad = []
    for role in ROLE_ORDERS[nb] if nb in ROLE_ORDERS else present:
        want = by_role.get(role, 0)
        have = role_size.get(role, 0)
        mark = ""
        if want > have:
            mark = "  <-- OVERRUN"
            bad.append((role, want, have))
        elif role not in by_role:
            mark = "  (untouched)"
        print("  %-8s %12d %12d%s" % (role, want, have, mark))

    grid_bad = list(geom_bad)
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
    for g in order_bad:
        print("  args: " + g)
    for g in geom_warn:
        print("  note: " + g)

    if bad or grid_bad or order_bad:
        print("safe_bench: REFUSING to dispatch; nothing was submitted.")
        return 3

    if check_only:
        print("safe_bench: CHECK-ONLY OK -- footprint fits, nothing dispatched.")
        return 0

    # hal_bench's argv is role-keyed, not slot-keyed, so the sizes travel by name
    # and its own binding list follows the export's binding_count.
    extra = [str(x) for x in (role_size["weight"], role_size["input"],
                              role_size["output"], role_size["wstage"],
                              role_size["ostage"], gx, gy, gz,
                              m_rows, k_blocks, tokens, iters)]
    if wfile or outfile:
        extra.append(wfile)
    if outfile:
        extra.append(outfile)
    wrap = os.environ.get("SAFE_BENCH_WRAP", "").split()
    return subprocess.call(wrap + [HAL_BENCH, hal] + extra)


if __name__ == "__main__":
    sys.exit(main())
