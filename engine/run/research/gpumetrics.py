#!/usr/bin/env python3
"""Decode amdgpu gpu_metrics v3.0 (Strix Halo / SMU 14.0.x APU). Read-only.
usage: gpumetrics.py            -> one decoded dump
       gpumetrics.py -i 0.05 -n 200 -> CSV samples (t_ns,gfx_MHz,gfxmax,socket_mW,gfx_mW,Tgfx,Tsoc,busy,thr_spl,thr_fppt,thr_sppt,thr_thm_gfx)
"""
import ctypes as C, sys, time, argparse
P = "/sys/class/drm/card1/device/gpu_metrics"
class Hdr(C.Structure):
    _fields_ = [("structure_size", C.c_uint16), ("format_revision", C.c_uint8), ("content_revision", C.c_uint8)]
U16, U32, U64 = C.c_uint16, C.c_uint32, C.c_uint64
class V30(C.Structure):  # natural alignment, mirrors include/kgd_pp_interface.h
    _fields_ = [("hdr", Hdr),
        ("temperature_gfx", U16), ("temperature_soc", U16), ("temperature_core", U16*16), ("temperature_skin", U16),
        ("average_gfx_activity", U16), ("average_vcn_activity", U16), ("average_ipu_activity", U16*8),
        ("average_core_c0_activity", U16*16), ("average_dram_reads", U16), ("average_dram_writes", U16),
        ("average_ipu_reads", U16), ("average_ipu_writes", U16), ("system_clock_counter", U64),
        ("average_socket_power", U32), ("average_ipu_power", U16), ("average_apu_power", U32),
        ("average_gfx_power", U32), ("average_dgpu_power", U32), ("average_all_core_power", U32),
        ("average_core_power", U16*16), ("average_sys_power", U16), ("stapm_power_limit", U16),
        ("current_stapm_power_limit", U16),
        ("average_gfxclk_frequency", U16), ("average_socclk_frequency", U16), ("average_vpeclk_frequency", U16),
        ("average_ipuclk_frequency", U16), ("average_fclk_frequency", U16), ("average_vclk_frequency", U16),
        ("average_uclk_frequency", U16), ("average_mpipu_frequency", U16),
        ("current_coreclk", U16*16), ("current_core_maxfreq", U16), ("current_gfx_maxfreq", U16),
        ("throttle_residency_prochot", U32), ("throttle_residency_spl", U32), ("throttle_residency_fppt", U32),
        ("throttle_residency_sppt", U32), ("throttle_residency_thm_core", U32), ("throttle_residency_thm_gfx", U32),
        ("throttle_residency_thm_soc", U32), ("time_filter_alphavalue", U32)]
def read():
    b = open(P, "rb").read()
    h = Hdr.from_buffer_copy(b[:4])
    assert (h.format_revision, h.content_revision) == (3, 0), (h.format_revision, h.content_revision)
    assert h.structure_size == C.sizeof(V30) == len(b), (h.structure_size, C.sizeof(V30), len(b))
    return V30.from_buffer_copy(b)
def dump(m):
    for n, _ in V30._fields_[1:]:
        v = getattr(m, n)
        print(f"{n:32s} {list(v) if hasattr(v, '__len__') else v}")
if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("-i", type=float); a.add_argument("-n", type=int, default=0)
    o = a.parse_args()
    if o.i is None: dump(read()); sys.exit()
    print("t_ns,gfx_MHz,gfx_maxMHz,socket_mW,gfx_mW,Tgfx_c,Tsoc_c,gfx_busy,thr_spl,thr_fppt,thr_sppt,thr_thm_gfx")
    k = 0
    while o.n == 0 or k < o.n:
        m = read(); k += 1
        print(m.system_clock_counter, m.average_gfxclk_frequency, m.current_gfx_maxfreq, m.average_socket_power,
              m.average_gfx_power, m.temperature_gfx / 100, m.temperature_soc / 100, m.average_gfx_activity,
              m.throttle_residency_spl, m.throttle_residency_fppt, m.throttle_residency_sppt,
              m.throttle_residency_thm_gfx, sep=",", flush=True)
        time.sleep(o.i)
