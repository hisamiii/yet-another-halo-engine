import re, sys, subprocess
OD = "/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0/lib/llvm/bin/llvm-objdump"
def regs(tok):
    tok = tok.strip()
    m = re.match(r"v\[(\d+):(\d+)\]", tok)
    if m: return set(range(int(m[1]), int(m[2]) + 1))
    m = re.match(r"v(\d+)(\.[lh])?$", tok)
    return {int(m[1])} if m else set()
def dsts(n):
    out = regs(n.split(None, 1)[1].split(",")[0]) if " " in n else set()
    if n.startswith("v_dual") and "::" in n:
        out |= regs(n.split("::")[1].split(None, 1)[1].split(",")[0])
    return out
for f in sys.argv[1:]:
    ins = [l.split("//")[0].strip() for l in subprocess.run([OD, "-d", f], capture_output=True, text=True).stdout.splitlines()]
    ins = [l for l in ins if re.match(r"^[vsdgb][a-z_0-9]+ ", l) or l in ("s_barrier",)]
    hits = []
    for i, l in enumerate(ins):
        if not (l.startswith("ds_") or l.startswith("global_store") or l.startswith("global_load")): continue
        ops = [o.strip() for o in l.split(None, 1)[1].split(",")]
        # sources: for loads only the address; for stores/swizzle address + data
        srcs = ops[1:] if l.startswith(("ds_load", "global_load", "ds_swizzle", "ds_bpermute")) else ops
        if l.startswith("ds_swizzle"): srcs = ops[1:2]
        src = set().union(*(regs(o) for o in srcs))
        for j in range(i + 1, min(i + 40, len(ins))):
            n = ins[j]
            if n.startswith("s_waitcnt") and ("lgkmcnt(0)" in n or "vmcnt(0)" in n or "depctr" in n): break
            if n.startswith("s_waitcnt_depctr"): break
            if n.startswith("v_") and (dsts(n) & src):
                hits.append((j - i, l[:44], n[:44])); break
    print(f"{f}: {len(hits)} DS/VMEM-source overwrites before a wait")
    for h in sorted(hits)[:4]: print("   ", h)
