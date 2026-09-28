#!/usr/bin/env python3
"""Reference for the Loom speculative segment weights (fixtures/spec_seg_weights)."""
import numpy as np

np.save("params.npy", np.array([1.0, 0.0], dtype=np.float32))
print("ok")
