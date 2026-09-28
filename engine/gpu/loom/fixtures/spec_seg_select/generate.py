#!/usr/bin/env python3
"""Reference for the Loom speculative segment select (fixtures/spec_seg_select).

Scratch layout: maxima[256]=0, target_sums[256]=4, residual_sums[256]=0,
target_total=0. residual_uniform=0.5."""
import numpy as np
scratch = np.zeros(769, dtype=np.float32)
scratch[256:512] = 4.0
np.save("scratch.npy", scratch)
np.save("params.npy", np.array([0.5], dtype=np.float32))
print("ok")
