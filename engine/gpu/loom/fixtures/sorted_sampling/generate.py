#!/usr/bin/env python3
"""Reference for the Loom sorted sampling (fixtures/sorted_sampling)."""
import numpy as np
np.save("params_f.npy", np.array([1.0, 1.0, 0.0, 0.5], dtype=np.float32))
np.save("params_i.npy", np.array([0, 1], dtype=np.int32))
print("ok")
