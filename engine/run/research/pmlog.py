#!/usr/bin/env python3
"""pmlog.py <out.npy> <interval_s>: raw ryzen_smu pm_table (float32[916]) + timestamp per sample, until SIGTERM."""
import sys, time, signal, numpy as np
stop = [False]; signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__(0, True))
rows = []; t0 = time.time(); iv = float(sys.argv[2])
while not stop[0]:
    a = np.frombuffer(open("/sys/kernel/ryzen_smu_drv/pm_table", "rb").read(), dtype=np.float32)
    rows.append(np.concatenate([[time.time() - t0], a])); time.sleep(iv)
np.save(sys.argv[1], np.array(rows, dtype=np.float64))
