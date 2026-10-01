#!/usr/bin/env python3
"""wush_calib.py <dump dir> <out.npz> [rows]: static WUSH-KV transforms per
attention layer and KV head from a tier A dump (first `rows` K rows, default
2048, and the Q rows among them), for KVQ_WUSH_FILE."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvcodec import wush_transform  # noqa: E402

d, out = sys.argv[1], sys.argv[2]
rows = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
gamma = float(os.environ.get("KVQ_WUSH_GAMMA", "1.0"))
z = {}
for f in sorted(glob.glob(f"{d}/k_l*.npy")):
    layer = int(os.path.basename(f)[3:5])
    k = np.load(f)[:rows].astype(np.float64)
    qp = np.load(f"{d}/qpos_l{layer:02d}.npy"); q = np.load(f"{d}/q_l{layer:02d}.npy")[qp < rows].astype(np.float64)
    m = k.mean(0, keepdims=True)
    z[f"mean{layer}"] = m.astype(np.float32)
    for h in range(4):
        z[f"T{layer}_{h}"] = wush_transform(k[:, h] - m[:, h], q[:, 6 * h:6 * h + 6].reshape(-1, 256), gamma)[0]
np.savez(out, **z)
print(f"{out}: {len(z)} arrays from {rows} rows")
