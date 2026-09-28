#!/usr/bin/env python3
"""Reference for the Loom DFlash selector partial top-k (fixtures/dflash_topk)."""
import numpy as np

logits = np.array([0.0, 7.0, 1.0, 6.0, 2.0, 5.0, 3.0, 4.0], dtype=np.float32)
order = np.argsort(-logits)[:4]
np.save("logits.npy", logits)
np.save("scores.npy", logits[order].astype(np.float32))
np.save("ids.npy", order.astype(np.int32))
print(logits[order], order)
