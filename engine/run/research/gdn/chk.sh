#!/usr/bin/env bash
# chk.sh <tag>: run <tag>.hsaco (yah_deltanet ABI) on layer-0 pp2048 inputs (zero state), compare with the recurrent kernel's output
G=/home/q/yah-scratch/gdn; B=2048; LH=/home/q/yet-another-halo-engine/engine/build/loomhip; GR=/home/q/yet-another-halo-engine/engine/run/gpu_run.sh
cd $G/k; t=$1
$GR gdnchk -- $LH $t.hsaco yah_deltanet 2 48 256 1 f:$G/l0.conv f:$G/l0.kq f:$G/l0.ab o:$((48*16384*4)):$t.state o:$((B*6144*4)):$t.out 2>&1 | grep -E "\*\*\*|refus|exit|error"
python3 - $t.out $G/l0.raw <<'P'
import numpy as np, sys
a = np.fromfile(sys.argv[1], np.float32).reshape(2048, 48, 128); r = np.fromfile(sys.argv[2], np.float32).reshape(2048, 48, 128)
d = np.abs(a - r)
print(f"nan {np.isnan(a).sum()}  rel {np.sqrt(np.nanmean(d**2)/(r**2).mean()):.2e}  max|d| {np.nanmax(d):.3e}")
print("rel by token block of 256:", " ".join(f"{np.sqrt(np.nanmean(d[i:i+256]**2)/(r[i:i+256]**2).mean()):.1e}" for i in range(0, 2048, 256)))
print("rel by head (first 12):", " ".join(f"{np.sqrt(np.nanmean(d[:, h]**2)/(r[:, h]**2).mean()):.0e}" for h in range(12)))
P
