import re, sys, subprocess
def regs(tok):
    m = re.match(r"v\[(\d+):(\d+)\]", tok)
    if m: return set(range(int(m[1]), int(m[2]) + 1))
    m = re.match(r"v(\d+)", tok)
    return {int(m[1])} if m else set()
for f in sys.argv[1:]:
    ins = [l.split("//")[0].strip() for l in subprocess.run(["/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0/lib/llvm/bin/llvm-objdump", "-d", f], capture_output=True, text=True).stdout.splitlines()]
    ins = [l for l in ins if re.match(r"^[vsdgb][a-z_0-9]+", l)]
    hits = []
    for i, l in enumerate(ins):
        if not l.startswith("v_wmma"): continue
        ops = [o.strip() for o in l.split(None, 1)[1].split(",")]
        src = set().union(*(regs(o) for o in ops[1:4]))
        valu = 0
        for j in range(i + 1, min(i + 12, len(ins))):
            n = ins[j]
            if n.startswith("v_wmma"): break
            if not n.startswith("v_"): continue
            valu += 1
            dst = regs(n.split(None, 1)[1].split(",")[0].strip())
            if n.startswith("v_dual"):
                dst |= regs(n.split("::")[1].split(None, 1)[1].split(",")[0].strip())
            if dst & src:
                hits.append((valu, l[:48], n[:40])); break
    print(f"{f}: {len(hits)} WMMA-source overwrites by VALU (min distance {min((h[0] for h in hits), default='-')})")
    for h in hits[:3]: print("   ", h)
