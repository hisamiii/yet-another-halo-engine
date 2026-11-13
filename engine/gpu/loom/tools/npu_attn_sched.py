"""Issue-order model for gen_npu_attn.py's locked core leaves (schedule(locked) + LOOM_EXP_LOCKED_PACK: emission order is
issue order; ops of different slots share a bundle).

Cycle-driven list scheduling of a straight-line Low block: one op a slot a cycle; an op issues once its inputs are ready
(lat(): producer -> consumer distances read off real images), its itinerary resources are free (npu_lsched.resources():
e.g. the one accumulator write port, shared by accumulator loads, vups, vconv.fp32.bf16 and accumulator moves at
different stages), its extra predecessors have issued and its result fits the register pools. Views (slice / concat /
register copies) follow their inputs."""
import re

import npu_lsched as LS

VIEW = ("slice", "concat", "copy")
MVB = ("vadd.16", "vsub.16", "vbcst", "vmax", "vmin", "vband", "vbor")      # mv ops whose results ride the MV bypass
MVC = ("vadd.16", "vsub.16", "vmax", "vmin", "vband", "vbor", "vconv.fp32.bf16")   # ... and whose inputs take it


def lat(p, i, c, j):
    """cycles from producer p's issue (its result i) to consumer c's (operand j of c), as Loom's locked images show
    them: a result is usable after its write stage (loads 7, popped fragments 8, vups / push 3, other mv / st ops 2),
    one sooner over the MV bypass; vec results chain into the next accumulate in 3 and reach anything else in 7. A
    pop's pointer, FIFO and position results feed the next pop or fill of the lane a cycle later."""
    mn = p["mn"]
    if ".post" in mn and i == (1 if mn.startswith("vld") else 0):       # the walked pointer
        return 1
    acc = c["slot"] == "vec" and (j == 0 or (j == 1 and c["mn"].startswith(("vsub.f", "vadd.f", "vadd.acc"))))
    if p["slot"] == "vec":
        return 3 if acc else 7
    if ".pop." in mn or ".fill." in mn or "fifo" in mn:
        if ".pop." in c["mn"] or ".fill." in c["mn"]:
            return 1
        return 8 if ".pop." in mn and i == 0 else 7
    w = 7 if mn.startswith(("vlda", "vldb")) else 3 if mn.startswith(("vups", "vst.push")) else 2
    if not acc and mn.startswith(MVB) and c["mn"].startswith(MVC):
        return w - 1
    return w


# each defining mnemonic's first-result size in register pool units (the allocator's unspillable classes)
UNITS = (("mma.bfp16ebs8", "acc", 4), ("mms.bf16", "acc", 4), ("mma.bf16bf16", "acc", 4), ("vlda.acc", "acc", 1),
         ("vmov.accumulator512", "acc", 1), ("vmov.vector512.to.accumulator512", "acc", 1), ("vups.2x", "acc", 2),
         ("vlda.pop", "x", 2), ("vldb.pop", "x", 2), ("vbcst", "x", 2), ("vconv.bf16.fp32", "x", 2), ("vadd.16", "x", 2),
         ("vsub.16", "x", 2), ("vmax_lt", "x", 2), ("vmin_ge", "x", 2), ("vldb.512", "x", 2), ("vlda.512", "x", 2))


def units(mn):
    for pre, pool, n in UNITS:
        if mn.startswith(pre):
            return pool, n
    return None


def schedule(ops, pools):
    """ops: [(line, tag, after tags, priority)]: tag names the op for others' after lists (or None). Returns the lines
    in issue order, the model's cycle count and each returned line's issue cycle.
    pools: {pool: budget}: an op issues only if its result fits the pool once the values it reads last are gone
    (values live from their op's issue to their last real reader's, through views)."""
    M, RES = LS.model(), LS.resources()
    P = []
    for i, (line, tag, after, prio) in enumerate(ops):
        lhs, eq, rhs = line.partition(" = ")
        if not eq:
            lhs, rhs = "", line
        mn = rhs.split()[0]
        view = mn.startswith(VIEW) and "aie2p.mpfs" not in rhs          # a pointer copy is a real st-slot move
        slot = None if view else (M.get(mn) or ("st",))[0]
        uses = re.findall(r"%[\w.]+", rhs.split(" : ")[0])
        P.append(dict(line=line, mn=mn, tag=tag, after=after, prio=prio, slot=slot, view=view, uses=uses,
                      defs=re.findall(r"%[\w.]+", lhs), res=() if view else RES.get(mn, ())))
    by = {d: i for i, o in enumerate(P) for d in o["defs"]}
    ri = {d: k for o in P for k, d in enumerate(o["defs"])}            # which result of its op a value is
    tags = {o["tag"]: i for i, o in enumerate(P) if o["tag"]}

    def roots(u):
        """the real producers behind value u (through views)"""
        k = by.get(u)
        if k is None:
            return []
        return [k] if not P[k]["view"] else [r for v in P[k]["uses"] for r in roots(v)]
    def vorder(u):
        """the order constraints of the views behind value u (a delayed pointer copy holds its users back)"""
        k = by.get(u)
        if k is None or not P[k]["view"]:
            return set()
        return {tags[t] for t in P[k]["after"]}.union(*(vorder(v) for v in P[k]["uses"]))
    for i, o in enumerate(P):
        o["data"] = {}                                  # producer -> required distance
        for j, u in enumerate(o["uses"]):
            for k in ([by[u]] if o["view"] and u in by else roots(u)):
                if k != i:
                    o["data"][k] = max(o["data"].get(k, 0), 0 if o["view"] or P[k]["view"] else lat(P[k], ri.get(u, 0), o, j))
        o["order"] = {tags[t] for t in o["after"]}.union(*(vorder(u) for u in o["uses"]))
    succ = [[] for _ in P]
    for i, o in enumerate(P):
        for k in set(o["data"]) | o["order"]:
            succ[k].append(i)
    cons = [set() for _ in P]                           # real readers of each op's first result (through views)
    first = {o["defs"][0]: k for k, o in enumerate(P) if o["defs"] and not o["view"]}

    def froots(u):
        k = by.get(u)
        if k is None:
            return []
        if not P[k]["view"]:
            return [k] if first.get(u) == k else []
        return [r for v in P[k]["uses"] for r in froots(v)]
    for i, o in enumerate(P):
        if not o["view"]:
            for u in o["uses"]:
                for k in froots(u):
                    if k != i:
                        cons[k].add(i)
    rd = [set() for _ in P]                             # the producers each real op reads
    for k, cs in enumerate(cons):
        for i in cs:
            rd[i].add(k)
    un = [None if o["view"] else units(o["mn"]) for o in P]
    left = [len(c) for c in cons]
    live = {q: 0 for q in pools}

    def net(i):
        """pool units op i would add: its result minus the producers it is the last reader of"""
        d = {q: 0 for q in live}
        if un[i] and un[i][0] in d and cons[i]:
            d[un[i][0]] += un[i][1]
        for k in rd[i]:
            if un[k] and un[k][0] in d and done[k]:
                if left[k] == 1:
                    d[un[k][0]] -= un[k][1]
        return d
    h = [0] * len(P)
    for i in reversed(range(len(P))):
        h[i] = max((P[j]["data"].get(i, 0) + h[j] for j in succ[i]), default=0)
    done, when, out, at, cyc = [False] * len(P), [0] * len(P), [], [], 0
    req, rsv = set(), set()
    while not all(done):
        if cyc >= 50 * len(P) + 100:
            rdy = [i for i, o in enumerate(P) if not done[i] and all(done[k] for k in set(o["data"]) | o["order"])]
            hold = [(P[k]["line"].strip()[:60], left[k]) for k in range(len(P)) if done[k] and left[k] and un[k]]
            raise AssertionError(f"schedule stuck: live {live}; ready {[(P[i]['line'].strip()[:70], net(i)) for i in rdy[:12]]}; "
                                 f"held {hold[:30]}; blocked "
                                 f"{[(P[i]['line'].strip()[:40], [P[k]['line'].strip()[:50] for k in set(P[i]['data']) | P[i]['order'] if not done[k]]) for i in range(len(P)) if not done[i] and 'mma.bfp16' in P[i]['line']][:3]}")
        used = set()
        while True:
            views = [i for i, o in enumerate(P) if not done[i] and o["view"] and all(done[k] for k in set(o["data"]) | o["order"])]
            cand = views or [i for i, o in enumerate(P) if not done[i] and not o["view"] and o["slot"] not in used and
                             all(done[k] and when[k] + d <= cyc for k, d in o["data"].items()) and all(done[k] for k in o["order"]) and
                             not any((cyc + off + j, r) in req or (need and (cyc + off + j, r) in rsv)
                                     for r, off, n, need in o["res"] for j in range(n)) and
                             all(live[q] + v <= pools[q] for q, v in net(i).items() if v > 0)]
            if not cand:
                break
            i = cand[0] if views else max(cand, key=lambda k: (P[k]["prio"], h[k], -k))
            if not views:
                for q, v in net(i).items():
                    live[q] += v
                for k in rd[i]:
                    left[k] -= 1
            done[i] = True
            when[i] = cyc
            out.append(P[i]["line"])
            at.append(cyc)
            if not views:
                used.add(P[i]["slot"])
                for r, off, n, need in P[i]["res"]:
                    for j in range(n):
                        (req if need else rsv).add((cyc + off + j, r))
        cyc += 1
    return out, cyc, at
