#!/usr/bin/env python3
"""Per-family AIE counter shares from a YAH_NPU_PROF run (model/npu_prof.hpp): each counter's cycles as a share of the
TRUE counter's (event 1, the core cycles between a job's gate and its counters' read), per core row and per column.

npu_prof.py <run log> [family]   (with a family: the full column x row grid of each counter)
npu_prof.py --pc <image.loom> <diag log> > <image>.pc   (emit_prefill_pp writes these with LOOM_DIAG_PC=1)

The log holds "npu prof: events c0,c1,c2,c3/m0,m1" and one "npu prof: job <i> <family> <ms> <values>" line per job,
values per core tile (columns 0-7, rows 2-5, column-major): the core module's four counters, the memory module's two.
Event ids (AIE2P): core 1 TRUE, 23 memory stall, 24 stream stall, 25 cascade stall, 26 lock stall, 28 active,
37 vector instruction, 38 load, 39 store, 20 / 21 PC in range 0-1 / 2-3; memory module 31 / 32 S2MM 0 / 1 stalled on
a lock, 33 MM2S 0 stalled on a lock, 35 / 36 S2MM 0 / 1 starved of stream data, 37 MM2S 0 backpressure.

--pc: the PC ranges of a fused GEMM image's cores (gen_npu_gemm leaves), from loom-compile's LOOM_DIAG_PC stderr (HRX
patch 0047) for its .loom: range 0-1 (event 20) the GEMM loop [^fzgemm, end), range 2-3 (event 21) the fill's decode
[^dq_tbuild1, ^fzrr). Leaves without them (the tails) get none. A leaf's blocks are its labels in source order after
the entry block; loom merges an empty fall-through block into the next (^outer into ^obody), so only labels up to
^fzgemm are trusted."""
import collections, re, sys

COLS, ROWS, REGS = 8, 4, 6
NAMES = {1: "true", 20: "gemm", 21: "decode", 23: "memstall", 24: "strmstall", 25: "cascstall", 26: "lockstall",
         28: "active", 37: "vector", 38: "load", 39: "store"}
MNAMES = {23: "s2mm0bd", 24: "s2mm1bd", 31: "s2mm0lock", 32: "s2mm1lock", 33: "mm2s0lock", 35: "s2mm0starve",
          36: "s2mm1starve", 37: "mm2s0bp"}


def load(path):
    events, fam = None, collections.defaultdict(lambda: [0.0, 0, [0] * (COLS * ROWS * REGS)])
    for line in open(path):
        if line.startswith("npu prof: events"):
            c, m = line.split()[3].split("/")
            events = [int(x) for x in c.split(",")] + [int(x) for x in m.split(",")]
        elif line.startswith("npu prof: job"):
            f = line.split()
            acc = fam[f[4]]
            acc[0] += float(f[5])
            acc[1] += 1
            for k, v in enumerate(f[6:]):
                acc[2][k] += int(v)
    names = [NAMES.get(e, f"c{e}") for e in events[:4]] + [MNAMES.get(e, f"m{e}") for e in events[4:]]
    return events, names, fam


def pc_ranges(loom, diag):
    """The PC file's lines for an image: its .loom source and its compile's LOOM_DIAG_PC stderr lines."""
    text = open(loom).read()
    labels = {}
    for m in re.finditer(r"^low\.func\.def [^\n]*@(\w+)\(\) asm \{(.*?)^\}", text, re.M | re.S):
        labels[m.group(1)] = ["<entry>"] + re.findall(r"^\^(\w+)", m.group(2), re.M)
    blocks, place = {}, {}
    for line in diag:
        f = line.split()
        if f and f[0] == "PCBLOCKS":
            blocks[f[1]] = (int(f[2]), [int(x) for x in f[3:]])
        elif f and f[0] == "PCTEXT" and f[3].startswith(".text.npu_gemm"):
            place[f[3][len(".text."):]] = (int(f[1]), int(f[2]), int(f[4]))
    out = []
    workers = re.findall(r"%k\d+_\d+ = worker %workers, %\w+, @(\w+)", text)
    for n, fn in enumerate(workers):
        size, offs = blocks[f"npu_gemm$worker${n}"]
        col, row, base = place[f"npu_gemm$worker${n}"]
        lab = labels[fn]
        at = lambda name: base + offs[lab.index(name)]
        r = [0, 0, 0, 0]
        if "fzgemm" in lab:
            r[0], r[1] = at("fzgemm"), base + size
        if "dq_tbuild1" in lab and "fzrr" in lab:
            r[2], r[3] = at("dq_tbuild1"), at("fzrr")
        out.append(" ".join(map(str, (col, row, *r))))
    return out


def main():
    if sys.argv[1] == "--pc":
        return print("\n".join(pc_ranges(sys.argv[2], open(sys.argv[3]).read().splitlines())))
    events, names, fam = load(sys.argv[1])
    t = events.index(1)   # the TRUE counter normalizes the others
    tile = lambda v, c, r, k: v[(c * ROWS + r) * REGS + k]
    print(f"{'family':8s} {'jobs':>4s} {'ms':>7s} {'Mcyc/col':>8s}  per row 2-5: " +
          " | ".join(names[k] for k in range(REGS) if k != t))
    for name, (ms, jobs, v) in sorted(fam.items()):
        cyc = sum(tile(v, c, r, t) for c in range(COLS) for r in range(ROWS)) / (COLS * ROWS)
        rows = []
        for k in range(REGS):
            if k == t:
                continue
            share = [sum(tile(v, c, r, k) for c in range(COLS)) / max(1, sum(tile(v, c, r, t) for c in range(COLS)))
                     for r in range(ROWS)]
            rows.append(" ".join(f"{100 * s:3.0f}" for s in share))
        print(f"{name:8s} {jobs:4d} {ms:7.1f} {cyc / 1e6:8.1f}  " + " | ".join(rows))
    if len(sys.argv) > 2:
        ms, jobs, v = fam[sys.argv[2]]
        for k in range(REGS):
            print(f"\n{names[k]} (% of true; rows 2-5 down, columns 0-7 across)" if k != t else
                  f"\n{names[k]} (Mcycles)")
            for r in range(ROWS):
                vals = [tile(v, c, r, k) / (1e6 if k == t else max(1, tile(v, c, r, t)) / 100) for c in range(COLS)]
                print(f"  r{r + 2} " + " ".join(f"{x:6.1f}" for x in vals))


if __name__ == "__main__":
    main()
