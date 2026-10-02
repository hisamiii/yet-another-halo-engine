#!/usr/bin/env bash
# att.sh <tag>: ATT trace of <tag>.hsaco (yah_deltanet ABI, layer-0 pp2048); top instruction time
G=/home/q/yah-scratch/gdn; TR=/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0
LH=/home/q/yet-another-halo-engine/engine/build/loomhip; GR=/home/q/yet-another-halo-engine/engine/run/gpu_run.sh
cd $G/k; t=$1; rm -rf att-$t; sleep 1
$GR gdnatt -- $TR/bin/rocprofv3 --att --att-library-path $TR/lib --kernel-include-regex yah_deltanet -d att-$t -o run -- $LH $t.hsaco yah_deltanet 2 48 256 1 f:$G/l0.conv f:$G/l0.kq f:$G/l0.ab z:$((48*16384*4)) z:$((2048*6144*4)) > att-$t.log 2>&1
D=$(dirname $(find att-$t -name code.json | head -1))
python3 - $D <<'P'
import glob, json, os, sys, collections
d = sys.argv[1]
code = json.load(open(os.path.join(d, "code.json")))["code"]; text = {r[2]: r[0] for r in code}
stall = collections.Counter(); per = collections.Counter(); tot = 0
for f in glob.glob(os.path.join(d, "se*_sm*_sl*_wv*.json")):
    for t, c, s, du, i in json.load(open(f))["wave"]["instructions"]:
        txt = text.get(i, "").strip(); tot += du; op = txt.split()[0] if txt else "?"
        stall[txt if op.startswith(("s_waitcnt", "s_barrier")) else op] += du; per[i] += du
print("top instruction time (incl. stalls):")
for k, v in stall.most_common(22): print(f"  {100*v/tot:5.1f}%  {k[:70]}")
P
