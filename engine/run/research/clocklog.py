#!/usr/bin/env python3
"""clocklog.py <out.csv> <interval_s>: gpu_metrics v3 + hwmon (vddgfx, sclk, PPT, edge) until killed."""
import sys, time, glob, signal
sys.path.insert(0, "/home/q/yah-scratch/tools"); import gpumetrics as G
hw = glob.glob("/sys/class/drm/card1/device/hwmon/hwmon*")[0]
def rd(f):
    try: return int(open(f"{hw}/{f}").read())
    except Exception: return -1
out = open(sys.argv[1], "w"); iv = float(sys.argv[2]); stop = [False]
signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__(0, True))
out.write("t,gfx_avg,gfx_cap,sclk,vddgfx,Tgfx,Tedge,gfx_mW,sock_mW,ppt_uW,busy,prochot,spl,fppt,sppt,thm_core,thm_gfx,thm_soc\n")
t0 = time.time()
while not stop[0]:
    m = G.read()
    out.write(f"{time.time()-t0:.3f},{m.average_gfxclk_frequency},{m.current_gfx_maxfreq},{rd('freq1_input')//1000000},{rd('in0_input')},"
              f"{m.temperature_gfx/100:.1f},{rd('temp1_input')/1000:.1f},{m.average_gfx_power},{m.average_socket_power},{rd('power1_input')},"
              f"{m.average_gfx_activity},{m.throttle_residency_prochot},{m.throttle_residency_spl},{m.throttle_residency_fppt},"
              f"{m.throttle_residency_sppt},{m.throttle_residency_thm_core},{m.throttle_residency_thm_gfx},{m.throttle_residency_thm_soc}\n")
    out.flush(); time.sleep(iv)
