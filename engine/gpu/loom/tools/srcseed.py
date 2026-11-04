"""Loom source reordering for the register allocator: each run of pure one-line definitions outside the tile loops
(scf.while over %ksw*) in a random topological order. The allocator's VGPR banks follow definition order, so a seed can
turn a build's WMMA bank conflicts clean (check with a bank survey; any source change needs a new seed)."""
import random
import re

PURE = re.compile(r"\s*(%[A-Za-z0-9_.]+(?:, %[A-Za-z0-9_.]+)*) = (index\.|scalar\.(?!.*atomic)|vector\.(constant|splat|"
                  r"bitcast|concat|slice|extract|from_elements|broadcast)|buffer\.view|config\.get|kernel\.workitem|"
                  r"kernel\.workgroup\.id|encoding\.)")
TOK = re.compile(r"%[A-Za-z0-9_.]+")


def seed_order(text, seed):
    """text with its pure definition runs reordered by seed (0: unchanged)"""
    if not seed:
        return text
    rnd = random.Random(seed)
    out, run, stack = [], [], []   # stack: per open region, inside a tile loop

    def flush():
        defs = {}
        for i, ln in enumerate(run):
            for d in TOK.findall(ln.split(" = ", 1)[0]):
                defs[d] = i
        deps = [{defs[u] for u in TOK.findall(ln.split(" = ", 1)[1]) if u in defs and defs[u] != i}
                for i, ln in enumerate(run)]
        done, order = set(), []
        while len(order) < len(run):
            i = rnd.choice([i for i in range(len(run)) if i not in done and deps[i] <= done])
            done.add(i)
            order.append(i)
        out.extend(run[i] for i in order)
        run.clear()
    for ln in text.split("\n"):
        hot = bool(stack and stack[-1])
        if not hot and "{" not in ln and "}" not in ln and PURE.match(ln):
            run.append(ln)
            continue
        flush()
        out.append(ln)
        st = ln.strip()
        if st.startswith("}"):
            inner = stack.pop() if stack else False
            if st.endswith("{"):   # "} else {" / "} do(...) {": the same construct's next region
                stack.append(inner)
        elif st.endswith("{"):
            stack.append(hot or bool(re.search(r"scf\.while\(%ksw", ln)))
    flush()
    return "\n".join(out)
