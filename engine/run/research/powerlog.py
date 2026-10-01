#!/usr/bin/env python3
"""powerlog.py <out.csv> <interval_s>: gpu_metrics v3 power/clock/temperature breakdown until killed."""
import sys, time, signal
sys.path.insert(0, "/home/q/yah-scratch/tools"); import gpumetrics as G
out = open(sys.argv[1], "w"); iv = float(sys.argv[2]); stop = [False]
signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__(0, True))
F = ["average_socket_power", "average_gfx_power", "average_all_core_power", "average_sys_power", "average_ipu_power",
     "average_dram_reads", "average_dram_writes", "average_gfxclk_frequency", "current_gfx_maxfreq", "average_fclk_frequency",
     "average_uclk_frequency", "average_socclk_frequency", "temperature_gfx", "temperature_soc", "average_gfx_activity",
     "throttle_residency_thm_gfx", "throttle_residency_fppt"]
out.write("t," + ",".join(F) + "\n"); t0 = time.time()
while not stop[0]:
    m = G.read(); out.write(f"{time.time()-t0:.3f}," + ",".join(str(getattr(m, f)) for f in F) + "\n"); out.flush(); time.sleep(iv)
