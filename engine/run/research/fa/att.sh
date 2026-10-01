#!/usr/bin/env bash
# att.sh <tag>: ATT trace of fa/<tag>.hsaco at pp8192; SIMD timeline + top instruction time
D8=/home/q/yah-scratch/attn8k; B=8192; TR=/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0
LH=/home/q/yet-another-halo-engine/engine/build/loomhip; GR=/home/q/yet-another-halo-engine/engine/run/gpu_run.sh
cd /home/q/yah-scratch/fa; t=$1; rm -rf att-$t; sleep 1
$GR faatt -- $TR/bin/rocprofv3 --att --att-library-path $TR/lib --kernel-include-regex yah_attn_wmma -d att-$t -o run -- $LH $t.hsaco yah_attn_wmma $((B/32)) 12 256 1 f:$D8/l3.aq f:$D8/l3.agate f:$D8/l3.ak16 f:$D8/vt.bin z:$((B*6144*2)) z:$((B*24*4)) > att-$t.log 2>&1
D=$(dirname $(find att-$t -name code.json | head -1)); python3 /home/q/yah-scratch/tools/simdtl.py $D $t 2>&1 | head -16
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
for k, v in stall.most_common(18): print(f"  {100*v/tot:5.1f}%  {k[:70]}")
print("top single instructions:")
for i, v in per.most_common(8): print(f"  {100*v/tot:5.1f}%  @{i:#x} {text.get(i,'')[:70]}")
P
