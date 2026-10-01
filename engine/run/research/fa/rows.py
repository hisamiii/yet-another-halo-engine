import json, collections, sys
def rows(pfx):
    s = sorted((json.loads(l) for l in open(f"{pfx}.samples")), key=lambda r: r["dispatch_event_id"])
    q = [l.rstrip().split(",") for l in open(f"{pfx}.seq")][1:]
    g = collections.defaultdict(float)
    for x, (_, k, *r) in zip(s, q): g[k.replace(".hal", "")] += x["value"] / 20
    return g
a, b = rows(sys.argv[1]), rows(sys.argv[2])
ta, tb = sum(a.values()), sum(b.values())
print(f"total {ta/1e6:.0f} -> {tb/1e6:.0f} M cycles ({(tb/ta-1)*100:+.2f}%)")
for k in sorted(set(a) | set(b), key=lambda k: -abs(b.get(k, 0) - a.get(k, 0)))[:6]:
    print(f"  {k:28s} {a.get(k,0)/1e6:8.1f} -> {b.get(k,0)/1e6:8.1f} M ({(b.get(k,0)/a[k]-1)*100 if a.get(k) else 0:+.1f}%)")
