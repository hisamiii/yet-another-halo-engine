"""Modulo schedule of the GMIX grid hot loop (gen_npu_dec GMOD): op offsets in a unit (pr, hh), one unit per II cycles.
CP-SAT model (pip install ortools) of the unit: Loom's issue distances, one op per cycle per resource modulo II, the
register budget (concat groups hold their whole aligned width) and the pushes in a window without loads.
Usage: npu_modsched.py [II] [seconds] -> prints GMOD; checks the table over a whole record (16 units)."""
import collections, sys
from ortools.sat.python import cp_model

II = int(sys.argv[1]) if len(sys.argv) > 1 else 29
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 300
VEC, ACC = 22, 20      # vec256 units (Loom fails at 24 with fragmentation), mbms units

# unit ops: name -> (resources, inputs, (register class, units)); P = the load port (vlda and gathers share it)
OPS = {}
for pos in range(2):
    for k in "gs":
        OPS[f"a{pos}{k}"] = (("P",), [], ("v", 2))                              # address vector loads
    for k in "sg":
        OPS[f"m{k}{pos}"] = (("MV", "RM"), [f"a{pos}g", f"a{pos}s"], ("v", 2))  # vsel lane mixing
    for hf in range(2):
        for k in "sg":
            for lh in "lh":
                OPS[f"g{pos}{k}{hf}{lh}"] = (("P",), [f"m{k}{pos}"], ("v", 1))  # gathers
        OPS[f"b{pos}{hf}"] = (("MV",), [f"g{pos}{k}{hf}{lh}" for k in "sg" for lh in "lh"], ("v", 2))
for p in range(2):
    OPS[f"t{p}"] = (("P",), [], ("v", 2))                                       # T part load, broadcast, shuffles
    OPS[f"c{p}"] = (("MV",), [f"t{p}"], ("v", 2))
    OPS[f"r{p}a"] = (("MV", "RM"), [f"c{p}"], ("v", 2))
    OPS[f"r{p}b"] = (("MV", "RM"), [f"c{p}"], ("v", 2))
for pos in range(2):
    OPS[f"m0{pos}"] = (("V",), ["r0a", "r0b", f"b{pos}0", f"b{pos}1"], ("a", 4))
for pos in range(2):
    OPS[f"m1{pos}"] = (("V",), [f"m0{pos}", "r1a", "r1b", f"b{pos}0", f"b{pos}1"], ("a", 4))
GROUPS = [[f"g{pos}{k}{hf}l", f"g{pos}{k}{hf}h"] for pos in range(2) for k in "sg" for hf in range(2)]
GROUPS += [[f"b{pos}0", f"b{pos}1"] for pos in range(2)] + [[f"r{p}a", f"r{p}b"] for p in range(2)]
GROUPED = {n for g in GROUPS for n in g}
# pushes of a pr (in its hh = 1 unit): p0 / p2 take the hh = 0 unit's accumulators (one unit earlier)
PUSH = {"p0": ("m10", II), "p1": ("m10", 0), "p2": ("m11", II), "p3": ("m11", 0), "fl": (None, 0)}


def lat(i):
    """Issue distance from producer i to a consumer (Loom's output; mma -> vst.push is 6)."""
    return 7 if i[0] in "agt" else 1 if i[0] == "c" else 4 if i.startswith("m0") else 2


def check(O):
    """Whole record (16 units): (violations, peak vec, peak acc)."""
    res, ev, viol = collections.Counter(), [], 0
    for u in range(16):
        t = u * II
        last = {}
        for n, (rs, ins, sz) in OPS.items():
            for i in ins:
                last[i] = max(last.get(i, -1), O[n])
                viol += O[i] + lat(i) > O[n]
        for n, (rs, ins, sz) in OPS.items():
            for r in rs + (("LD",) if rs == ("P",) else ()):
                res[(r, t + O[n])] += 1
            if n not in GROUPED and not n.startswith("m1"):
                ev.append((t + O[n], t + last[n]) + sz)
        for g in GROUPS:
            ev.append((t + min(O[n] for n in g), t + max(last[n] for n in g), "v", sum(OPS[n][2][1] for n in g)))
        if u % 2:
            for p, (src, back) in PUSH.items():
                if p != "fl" or u % 4 == 3:
                    res[("ST", t + O[p])] += 1; res[("RM", t + O[p])] += 1
                if src:
                    viol += O[p] + back < O[src] + 6
                    ev.append((t - back + O[src], t + O[p], "a", 4))
    viol += sum(v - 1 for (r, c), v in res.items() if v > 1 and r != "LD")
    st = [c for (r, c) in res if r == "ST"]
    viol += sum(1 for c in st for k in (1, 2, 3) if ("LD", c + k) in res)      # loads wait 4 cycles after a store
    viol += sum(1 for c in st for k in range(5) if ("V", c - k) in res)        # a push 5 cycles after a MAC
    live = {"v": collections.Counter(), "a": collections.Counter()}
    for a, b, cls, z in ev:
        for x in range(a, b):
            live[cls][x] += z
    return viol, max(live["v"].values()), max(live["a"].values())


m = cp_model.CpModel()
names = list(OPS) + list(PUSH)
est = {}
for n in OPS:
    est[n] = max([est[i] + lat(i) for i in OPS[n][1]] or [0])
for n, (src, back) in PUSH.items():
    est[n] = est[src] + 6 - back if src else 0
hi = {n: est[n] + 40 + (II if n in PUSH else 0) for n in names}
s = {n: m.NewIntVar(est[n], hi[n], n) for n in names}
r = {}
for n in names:
    r[n] = m.NewIntVar(0, II - 1, "")
    m.AddModuloEquality(r[n], s[n], II)
for n, (rs, ins, sz) in OPS.items():
    for i in ins:
        m.Add(s[n] >= s[i] + lat(i))
pl = list(PUSH)
for a, b in zip(pl, pl[1:]):
    m.Add(s[b] >= s[a] + 1)
for n, (src, back) in PUSH.items():
    if src:
        m.Add(s[n] >= s[src] - back + 6)
users = collections.defaultdict(list)
for n, (rs, ins, sz) in OPS.items():
    for x in rs:
        users[x].append(n)
users["ST"] += pl; users["RM"] += pl
for x, us in users.items():
    m.AddAllDifferent([r[n] for n in us])


def moddiff(a, b, dom):
    d = m.NewIntVar(0, II - 1, "")
    m.AddModuloEquality(d, r[a] - r[b] + II, II)
    m.AddLinearExpressionInDomain(d, cp_model.Domain.FromIntervals(dom))
for p in pl:
    for n in users["P"]:
        moddiff(n, p, [[0, 0], [4, II - 1]])
    for n in users["V"]:
        moddiff(p, n, [[5, II - 1]])
cons = collections.defaultdict(list)
for n, (rs, ins, sz) in OPS.items():
    for i in ins:
        cons[i].append(n)
vals = [(sz[0], sz[1], [n], cons[n]) for n, (rs, ins, sz) in OPS.items() if n not in GROUPED and not n.startswith("m1")]
vals += [("v", sum(OPS[n][2][1] for n in g), g, sorted({c for n in g for c in cons[n]})) for g in GROUPS]
press = {"v": collections.defaultdict(list), "a": collections.defaultdict(list)}
cost = []


def live(d, e, lo, hi, cls, sz):
    for t in range(lo, hi):
        lv, a, b = m.NewBoolVar(""), m.NewBoolVar(""), m.NewBoolVar("")
        m.Add(d <= t).OnlyEnforceIf(a); m.Add(d > t).OnlyEnforceIf(a.Not())
        m.Add(e > t).OnlyEnforceIf(b); m.Add(e <= t).OnlyEnforceIf(b.Not())
        m.AddBoolAnd([a, b]).OnlyEnforceIf(lv); m.AddBoolOr([a.Not(), b.Not()]).OnlyEnforceIf(lv.Not())
        press[cls][t % II].append(sz * lv)
for cls, sz, defs, cs in vals:
    d = m.NewIntVar(0, 1000, ""); e = m.NewIntVar(0, 1000, "")
    m.AddMinEquality(d, [s[n] for n in defs]); m.AddMaxEquality(e, [s[n] for n in cs])
    cost.append(sz * (e - d))
    live(d, e, min(est[n] for n in defs), max(hi[n] for n in cs), cls, sz)
for k in range(2):     # the accumulators: to the later of their two pushes
    src, d = f"m1{k}", s[f"m1{k}"]
    e = m.NewIntVar(0, 1000, "")
    m.AddMaxEquality(e, [s[f"p{2 * k}"] + II, s[f"p{2 * k + 1}"]])
    cost.append(4 * (e - d))
    live(d, e, est[src], hi[f"p{2 * k}"] + II + 1, "a", 4)
peak = m.NewIntVar(0, VEC, "")
for cls, cap in (("v", peak), ("a", ACC)):
    for t, terms in press[cls].items():
        m.Add(sum(terms) <= cap)
span = m.NewIntVar(0, 1000, "")
m.AddMaxEquality(span, [s[n] for n in names])
m.Minimize(400 * peak + sum(cost) + 2 * span)
sol = cp_model.CpSolver()
sol.parameters.max_time_in_seconds = SECONDS
sol.parameters.num_workers = 16
st = sol.Solve(m)
if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    sys.exit(f"no schedule at II {II}: {sol.StatusName(st)}")
O = {n: sol.Value(s[n]) for n in names}
viol, pv, pa = check(O)
print(f"II {II}: {sol.StatusName(st)}, record {15 * II + max(O.values()) - min(O.values())} cycles, vec {pv}, acc {pa}, "
      f"violations {viol}")
print("GMOD = dict(" + ", ".join(f"{n}={c}" for n, c in sorted(O.items(), key=lambda kv: (kv[1], kv[0]))) + ")")
