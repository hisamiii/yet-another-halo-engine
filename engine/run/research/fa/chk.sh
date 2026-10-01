#!/usr/bin/env bash
# chk.sh <tag>: run fa/<tag>.hsaco (f32 out) on real pp8192 layer-3 inputs, compare with HIP's f32 output
D=/home/q/yah-scratch/attn8k; B=8192; LH=/home/q/yet-another-halo-engine/engine/build/loomhip; GR=/home/q/yet-another-halo-engine/engine/run/gpu_run.sh
cd /home/q/yah-scratch/fa; t=$1
$GR fachk -- $LH $t.hsaco yah_attn_wmma $((B/32)) 12 256 1 f:$D/l3.aq f:$D/l3.agate f:$D/l3.ak16 f:$D/vt.bin o:$((B*6144*4)):$t.out z:$((B*24*4)) 2>&1 | grep -E "\*\*\*|refus|exit"
python3 - $t.out $D/ref_out.bin <<'P'
import numpy as np, sys
a = np.fromfile(sys.argv[1], np.float32).reshape(8192, 24, 256); r = np.fromfile(sys.argv[2], np.float32).reshape(8192, 24, 256)
d = np.abs(a - r); print(f"nan {np.isnan(a).sum()}  max|d| {np.nanmax(d):.3e}  rms d {np.sqrt(np.nanmean(d**2)):.3e}  rms ref {np.sqrt((r**2).mean()):.3e}  rel {np.sqrt(np.nanmean(d**2)/(r**2).mean()):.2e}")
w = np.unravel_index(np.nanargmax(d), d.shape); print("worst (token, head, dim)", w, a[w], r[w])
bt = np.sqrt(np.nanmean(d**2, axis=(1,2))); print("rms by token block of 1024:", " ".join(f"{x:.1e}" for x in bt.reshape(8,1024).mean(1)))
bh = np.sqrt(np.nanmean(d**2, axis=(0,2))); print("rms by head:", " ".join(f"{x:.0e}" for x in bh))
P
