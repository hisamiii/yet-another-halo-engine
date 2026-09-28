#!/usr/bin/env python3
"""Reference for the Loom embedding lookup with a device token pointer."""
import numpy as np

np.save("token.npy", np.array([2], dtype=np.int32))
print("ok")
