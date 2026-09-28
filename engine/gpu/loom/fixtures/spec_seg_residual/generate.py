#!/usr/bin/env python3
"""Reference for the Loom speculative residual kernel (fixtures/spec_seg_residual)."""
import numpy as np
np.save("accept.npy", np.array([0.0, 0.5], dtype=np.float32))
np.save("reject.npy", np.array([1.0, 1.0], dtype=np.float32))
np.save("token.npy", np.array([5], dtype=np.int32))
print("ok")
