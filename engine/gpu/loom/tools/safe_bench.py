#!/usr/bin/env python3
"""Gate hal_bench on the compiler's own footprint analysis.

usage: safe_bench.py <source.loom> <hal> <m_rows> <k_blocks> <block_bytes>
                     <tokens_per_wg> <token_tiles> <k_split>
                     <gx> <gy> <gz> <config=value>... [--iters=N]
                     [--wfile=PATH] [--outfile=PATH] [--check-only]

The sizes it verifies are the sizes it hands to hal_bench, which allocates exactly what it is given.
It compiles the source under the same config and reads the declared footprint (source_low.memory.roots[].interval_envelope.byte_count).
It refuses unless weight, input and output fit their derived sizes.
wstage and ostage are scratch that is never read back, so they are sized to the declared need.
An extent past an allocation does not fault on this target: the shader reads unmapped VA, the gfx ring times out and the box resets.

Two more gates:
  * The token stride comes from the dispatch.txt line beside the HAL ("<hal> <baked_tile> <rowgrp> <tt>"), not from argv.
    The kernel bakes token_tiles, so it walks tt*baked_tile tokens; tokens_per_wg and token_tiles must agree with that line.
  * Sizes are keyed by the compiler's root name ('weight', 'input', 'wstage', 'ostage', 'output'), not by argument position.
    The GEMM family's order varies (ROLE_ORDERS); the argument count is validated too.

--check-only stops after the verification and dispatches nothing.
--token-first: the kernel's grid is (token_tiles, m_groups, 1) instead of (m_groups, token_tiles, 1).
--outfile=PATH dumps the raw output (out[token * m_rows + row]) after the warmup, so two HALs can be compared on identical operands.
SAFE_BENCH_WRAP (env) is a command prefix for the hal_bench argv, e.g. "rocprofv3 --pmc SQ_INSTS_VALU --".
It never changes an operand; its timings are not comparable.
SAFE_BENCH_BIN (env) overrides the harness binary (default engine/build/hal_bench).
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loom_preflight as lp  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))
import hrx_paths  # noqa: E402

COMPILE = hrx_paths.LOOM_COMPILE
SCRATCH_MIN = 64 << 20
HAL_BENCH = os.environ.get("SAFE_BENCH_BIN", os.path.join(HERE, "..", "..", "..", "build", "hal_bench"))
# argument order of each GEMM-family variant, by argument count
ROLE_ORDERS = {
    5: ["weight", "input", "wstage", "ostage", "output"],
    6: ["weight", "grid", "input", "wstage", "ostage", "output"],
    7: ["weight", "grid", "ksigns", "input", "wstage", "ostage", "output"],
}


def env():
    return hrx_paths.env()


def dispatch_geometry(hal):
    """Return (baked_tile, rowgrp, tt) for a HAL from the dispatch.txt beside it, or None.
    emit_prefill_pp writes that line from the geometry it baked into the kernel, so it is the authority on the token stride.
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
            # grid (token_tiles, m_groups, 1): consecutive workgroups share a row group's weight slice in L2.
            # Only the two grid-role checks change; the envelope checks do not.
            token_first = True

    # the HAL's own geometry, not the caller's word for it
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
    # the formulas the kernels themselves use
    role_size = {
        "weight": m_rows * k_blocks * block_bytes,
        "input": k_blocks * 256 * tokens * 2,
        "output": m_rows * k_split * tokens * 4,
        # fixed tables the driver binds: the IQ grid is 512 x i32, the ksigns table 128 bytes
        "grid": 2048,
        "ksigns": 128,
        # scratch is sized from the declared need below, so its check is exact
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
        # The compiler omits a root it proves is never touched (e.g. 'wstage' when activations stage in LDS).
        # So require the expected relative order, not a complete list.
        if present != [r for r in order if r in present]:
            order_bad.append("argument order %s is not a subsequence of the "
                             "expected %s" % (present, order))

    # key every declared envelope by the role the compiler named: neither the count nor the order is assumed
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

    # hal_bench's argv is role-keyed; its own binding list follows the export's binding_count
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
