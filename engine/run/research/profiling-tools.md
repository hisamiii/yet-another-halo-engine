# Profiling tools for gfx1151 (Strix Halo): what works, what it would give us

Researched 2026-10-01. Nothing here launched a kernel or changed GPU state; local checks were sysfs reads, `--help`/`--version`, and counter *listings*.
Labels: **[V]** verified locally, **[D]** documented by AMD/upstream, **[I]** inference.
TR = `/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0`.

## TL;DR

1. **Biggest win: we're already sitting on ~400 gfx1151 counters.** rocprof-compute 3.6.0 in TR ships its own counter YAML. It has gfx1151 event IDs for `SQ_INST_CYCLES_VALU`, `SQ_INST_CYCLES_LDS`, `SQ_WAIT_BARRIER`, `SQ_WAIT_INST_ANY`, `SPI_RA_*` occupancy limiters, SQC I$/K$ and more. Plain `rocprofv3 --pmc` can use them by pointing `ROCPROFILER_METRICS_PATH` at that YAML. **[V, listing only]**
2. rocprof-compute supports gfx1151, including the Speed-of-Light, memory chart and WGP panels. The roofline has no peaks and no WMMA row. **[V+D]**
3. PC sampling is **not available**: zero agents support it locally, and upstream KFD has no PC-sampling ioctl. **[V]**
4. RGP can't do HIP/OpenCL on Linux. **[D]** RGA (binary analysis, gfx1151) and ROCprof Compute Viewer are the useful GPUOpen/ROCm GUIs. **[D]**
5. Monitoring: `gpu_metrics` is v3.0. I wrote a decoder: `/home/q/yah-scratch/tools/gpumetrics.py`. **[V]** Per the user's scope change, clock pinning is not pursued.

---

## 1. Clock/power monitoring (read-only, optional diagnostic)

- **[V]** card1, PCI 1002:1586, MP1 (SMU) 14.0.1, GC 11.5.1. Perf level `auto`. `pp_dpm_sclk` has 3 fine-grain rows (600 / current / 2900 MHz). `pp_power_profile_mode` doesn't exist on this APU. hwmon gives `PPT` power, `edge` temp and `sclk` freq; `vddgfx`/`vddnb` read 0.
- **[V]** `gpu_metrics` header `08 01 03 00` means 264 bytes, **format 3, content 0 → `struct gpu_metrics_v3_0`** (`/home/q/linux-7.3rc/drivers/gpu/drm/amd/include/kgd_pp_interface.h`). The decoder checks that its size matches.
  - Fields: gfx/soc/skin temps, `average_gfxclk_frequency`, `current_gfx_maxfreq`, socket/APU/gfx power, DRAM read/write MB/s, plus **cumulative throttle-residency counters** (`spl`, `fppt`, `sppt`, `thm_core`, `thm_gfx`, `thm_soc`).
  - At idle today `throttle_residency_thm_gfx` = 306735 and `fppt` = 34055 since boot. The units aren't documented. **[I]** Diffing these counters before and after a timed round is the cheapest way to tell whether a slow round was throttled.
  - `time_filter_alphavalue` = 1000000 (field comment says µs). **[I]** If so, the averages are smoothed over about 1 s, so this is good for one row per run and useless per kernel.
- Usage: `gpumetrics.py` dumps every field once. `gpumetrics.py -i 0.1 -n 0 > run.csv &` samples in the background (one 264-byte sysfs read per sample, negligible cost **[I]**).
- **[V]** `amd-smi` 26.4.0 **is** installed at `$TR/bin/amd-smi` (the inventory says it's absent; that's wrong). `amd-smi metric -c -p -t` reads socket power, FCLK and edge temp, but no GFX clock. The decoder gives more. **[D]** The 7.13 notes do advertise richer APU telemetry for amd-smi.
- FYI only (not pursued): **[V, kernel source]** on SMU 14.0.1, `profile_standard` sets SCLK to the UMD p-state of 700 MHz, FCLK 1800 and SOCCLK 678. `profile_peak` sets them to max via *soft* limits. Both are far from real-world behaviour, which supports the user's call.

## 2. RGP / Radeon Developer Panel / ROCprof Compute Viewer

- **RGP: no HIP on Linux. [D]** The RGP manual lists the compute APIs (OpenCL, HIP) as Windows 11 only. On Linux, RGP only supports Vulkan on Ubuntu 24.04.
  - RGP 2.7 (June 2026) adds RDNA3.5 APU support, but for the supported APIs only.
  - **[V]** Earlier `/home/q/incoh/out/rgp/*.rgp` captures were llama-bench Vulkan/RADV traces, not HIP.
  - Verdict: not usable for yah-run/Loom. Adoption cost: n/a.
- **ROCprof Compute Viewer (RCV). [D]** A Qt GUI that reads rocprofv3 `ui_output_agent_*_dispatch_*` dirs, which we already produce. It shows per-wave timelines, instruction latency with memory→waitcnt mapping, a hotspot histogram, hidden-latency estimates for gfx10+, and a flamegraph.
  - The ROCm thread-trace blog lists gfx1100 and gfx1150 as fully supported. **[I]** gfx1151 should work, since our ATT decode already works.
  - Not installed. Linux binaries are on the releases page.
  - Verdict: documented for RDNA3/3.5; gfx1151 unknown but likely. Gives an interactive per-wave view instead of attwave.py-style scripts. Cost: download plus Qt6 libs. It does **not** fix the ~1.5× ATT distortion.
- **ATT knobs [V]** (`rocprofv3 --help`): `--att-target-cu`, `--att-simd-select` (on gfx10+ this is one SIMD ID), `--att-shader-engine-mask` (default 0x1), `--att-serialize-all`.
  - `--att-perfcounters` and `--att-activity` are **gfx9-only**, so there's no SPM counter stream on gfx1151.
  - **[D]** gfx11+ supports `s_ttracedata`/`s_ttracedata_imm` markers, which can bracket phases in traces.

## 3. rocprof-compute (ex-Omniperf) 3.6.0

- **[V]** `$TR/libexec/rocprofiler-compute/rocprof_compute_soc/soc_gfx1151.py` exists ("RDNA3.5": l2_banks=8, lds_banks_per_cu=32). Analysis panels exist for: 0000 Top Stats, 0100 System Info, **0200 System Speed-of-Light**, 0300 Memory Chart, 0400 roofline, 0500 CPC, 0600 SPI, 0700 WGP, 0800 TCP, 1100 GL1C, 1300 GL2C, 1500 GCEA, 1700 GRBM.
  - The empirical roofline benchmark is skipped for gfx1151 (`soc_base.py`: "Roofline not supported on Strix Halo").
  - **[D]** The 7.13 release notes say "Added AMD Ryzen AI Max 300 series (gfx1151) support". Known issue: the roofline 'peak' column is N/A.
- **The SOL panel is crude for our kernels [V].**
  - VALU FLOPs = `SQ_INSTS_VALU × 64 / time`, and WMMA is explicitly "not a dedicated row".
  - IPC = `(SQ_INSTS_ALL − INTERNAL) / SQ_BUSY_CYCLES`, peak 5.
  - Occupancy, L2 hit rate, L2–fabric bytes with peak=None, wave dependency/issue wait %, TCP/GL1C/SQC hit rates and bandwidth.
- **The real value is its counter YAML** (`profile_configs/sdk_config.yaml`, 344 gfx1151 entries, 203 raw). **[V]** Key gfx1151 raw events:
  - `SQ_INST_CYCLES_VALU` SQ103 ("cycles needed to execute VALU ops (SIMD cycles)")
  - `SQ_INST_CYCLES_LDS` SQ109
  - `SQ_WAIT_BARRIER` SQ42, `SQ_WAIT_CNT_ANY` SQ36, `SQ_WAIT_INST_ANY` SQ26, `SQ_WAIT_ANY` SQ35, `SQ_WAIT_IFETCH` SQ41
  - `SQ_INSTS_VALU_TRANS` SQ170, `SQ_INSTS_ALL` SQ48, `SQ_INST_LEVEL_LDS` SQ88
  - `SQ_LDS_UNALIGNED_STALL` SQ258
  - `SPI_RA_{VGPR_SIMD,WAVE_SIMD,LDS_CU,BAR_CU,TGLIM_CU}_FULL_CSN`, `SPI_RA_WVLIM_STALL_CSN` (which resource blocks wave launch)
  - `SQC_ICACHE_*`/`SQC_DCACHE_*`, `CPC_*`, `GRBM_*_BUSY`
- Verdict: works on gfx1151 per the docs, and the install is present. **[V]** `--version` runs.
  - Not yet run. It replays the app once per counter pass (several passes), so use it on loomhip standalone cases, not on the full prefill.
  - Cost: low. Panel output still needs our own WMMA peak.

## 4. PC sampling (rocprofv3 `--pc-sampling-*`)

- **[V]** The options exist: `--pc-sampling-beta-enabled`, `--pc-sampling-method {stochastic,host_trap}`, `--pc-sampling-unit {instructions,cycles,time}`, `--pc-sampling-interval`.
- **[V]** `rocprofv3-avail list --pc-sampling` prints the "Agents supporting PC Sampling" header with an empty list.
- **[V]** `linux-7.3rc` has no PC-sampling code in `amdkfd` or `kfd_ioctl.h`. **[D]** The SDK docs list host-trap for MI200/MI300/MI350 and stochastic for MI300+ only. RDNA isn't mentioned. Host-trap has up to 2 instructions of skid.
- Verdict: **not supported** on our stack. Cost to adopt: n/a (it would need the DKMS KFD, and even then RDNA isn't documented).

## 5. Radeon GPU Analyzer (RGA)

- **[V]** Not installed. `/usr/bin/rga` is ripgrep-all, a name clash to watch for.
- **[D]** RGA 2.14.2 (June 2026) runs on Linux and Windows. It added gfx1151 targets in 2.12.
  - Binary Analysis mode loads precompiled code objects and shows ISA, VGPR pressure (live-register analysis, jump to max pressure) and static resource allocation. It's offline and needs no driver.
  - HIP code objects are documented (2.9.1, for MI300).
- **[I]** A raw Loom `.hsaco` for gfx1151 should load, since it's a standard AMDGPU ELF code object. Unverified.
- Verdict: documented, gfx1151 binary mode probably works. It would give a per-instruction live-VGPR curve for Loom vs HIP kernels, a cross-check for the 144 vs 192 VGPR story and the scheduler cost-model work.
- Cost: download a tarball (no GPU, safe to try while the timing session runs). It gives no timing or latency information.

## 6. rocprof-sys (Omnitrace) 1.6.0

- **[V]** Installed: `rocprof-sys-run|sample|instrument|causal|avail`, v1.6.0 built against ROCm 7.13. **[D]** gfx1150–1153 are listed as supported (gfx1151 since ROCm 7.1).
- What it gives for the "pipeline tax":
  - A Perfetto timeline with CPU call-stack sampling, HIP/HSA API calls, kernel dispatches, and **AMD-SMI sampled GPU tracks** (`ROCPROFSYS_SAMPLING_GPUS`, `ROCPROFSYS_AMD_SMI_METRICS`).
  - **[D]** Periodic device-wide PMC sampling "without serializing kernel dispatches".
  - That lets us line up clock/power dips, queue gaps and host stalls against each prefill kernel.
- **[I]** Per the inventory, rocprofv3 can't see HRX dispatches, so rocprof-sys GPU tracing of HRX almost certainly can't either. It only helps for yah-run/HIP, or loomhip runs. HRX needs `iree-profile` device-metrics plus `gpumetrics.py` running alongside.
- **[I]** For the cheaper option first: `rocprofv3 --kernel-trace` → `rocpd2pftrace` (installed **[V]**) already gives a Perfetto kernel timeline without instrumenting the binary.
- Verdict: documented to work. Cost: medium (config env vars; `rocprof-sys-instrument` rewrites binaries; overhead depends on settings). Try `rocprof-sys-sample` first.

## 7. Counters on gfx1151

- **[V]** The stock rocprofv3 listing has 123 entries. Many are agent constants. The raw SQ set is only BUSY_CYCLES, WAVES, WAVE_CYCLES, INSTS_{VALU,SALU,SMEM,LDS,FLAT,TEX_*,WAVE32*}, LDS_BANK_CONFLICT/IDX_ACTIVE, plus GL2C_*, TA_TA_BUSY and GRBM. Derived: MeanOccupancy*, OccupancyPercent, GPUBusy.
- **[V]** The stock `config.yaml` defines `SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY`, `SQ_INST_LEVEL_LDS`, `ALUStalledByLDS` and similar for `gfx11`/gfx1100–1102 but **not gfx1151**. The SDK matches the exact arch name.
- **[V] How to unlock them** (listing verified; collection not run, per the no-GPU rule):
  ```
  mkdir -p ~/yah-scratch/rcmetrics
  cp $TR/libexec/rocprofiler-compute/rocprof_compute_soc/profile_configs/sdk_config.yaml ~/yah-scratch/rcmetrics/config.yaml
  ROCPROFILER_METRICS_PATH=~/yah-scratch/rcmetrics rocprofv3 --pmc SQ_INST_CYCLES_VALU SQ_BUSY_CYCLES ... -- <app>
  ```
  - With this, a patched `rocprofv3-avail` lists **403** counters for gfx1151. The stock `rocprofv3-avail` hard-codes `ROCPROFILER_METRICS_PATH`, so I tested with a patched copy in my scratchpad.
  - The `rocprofv3` CLI itself doesn't override the variable **[V, grep]**. rocprof-compute sets it the same way (`utils_profile.py`).
- **"Issue utilization" [I]:** RDNA3 WMMA executes on the VALU, so `SQ_INST_CYCLES_VALU / (SQ_BUSY_CYCLES × SIMDs)` should be a direct VALU/WMMA-pipe busy fraction. `SQ_INST_CYCLES_LDS` gives the same for LDS, and `SQ_WAIT_BARRIER / SQ_WAVE_CYCLES` gives barrier tax. That's essentially the speed-of-light view we lack, and it costs no ATT perturbation.
  - There is **no WMMA-specific counter** in any gfx1151 YAML **[V]**.
  - Validate first. On a kernel with known WMMA count × cycles-per-WMMA, check that `SQ_INST_CYCLES_VALU` matches. Watch the units (several SQ counters count quad-cycles) and the per-SIMD vs per-SE aggregation.
- Caveat **[D]**: AMDResearch's intellikit/metrix PR #189 (merged 2026-09-21) found some gfx11 assumptions wrong on gfx1151 hardware. For example, it says TA_BUFFER_* wavefront counters aren't exposed. **[V]** HRX nonetheless programs TA_BUFFER_* with gfx11 IDs. Treat any counter that reads 0 or is implausible as unsupported, and cross-check against known instruction counts.
- **[I]** Other cheap static tools: RGA (above). `compile_report` already covers residency.

## Suggested adoption order

1. `gpumetrics.py` throttle-residency diff around each timed round. Zero risk, explains the 4–9% drift or rules it out.
2. Point `ROCPROFILER_METRICS_PATH` at rocprof-compute's YAML and validate `SQ_INST_CYCLES_VALU`/`_LDS`/`SQ_WAIT_BARRIER`/`SPI_RA_*` on one loomhip case. If they're sane, build `sol.py` (VALU/WMMA %, LDS %, barrier %, DRAM GB/s vs 256 GB/s).
3. `rocprof-compute profile/analyze` on loomhip for the panel view.
4. RGA binary mode for VGPR-pressure curves.
5. RCV for per-wave ATT browsing.
6. rocprof-sys or `rocpd2pftrace` timelines for the pipeline tax (HIP side).

## Sources

- gpu_metrics v3.0 struct: `/home/q/linux-7.3rc/drivers/gpu/drm/amd/include/kgd_pp_interface.h`
- SMU14 APU perf levels and OD: `drivers/gpu/drm/amd/pm/swsmu/smu14/smu_v14_0_0_ppt.c`, `smu_v14_0.c`
- RGP manual (API/OS matrix): https://gpuopen.com/manuals/rgp_manual/rgp_manual-index/
- RGP releases: https://github.com/GPUOpen-Tools/radeon_gpu_profiler/releases
- RGA releases: https://github.com/GPUOpen-Tools/radeon_gpu_analyzer/releases
- RGA manual: https://gpuopen.com/manuals/rga_manual/rga_manual-index/
- ROCprof Compute Viewer: https://github.com/ROCm/rocprof-compute-viewer
- Thread trace blog: https://rocm.blogs.amd.com/software-tools-optimization/thread-trace/README.html
- PC sampling docs: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html
- ROCm 7.13 release notes: https://rocm.docs.amd.com/en/7.13.0-preview/about/release-notes.html
- rocprof-sys options: https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/configuring-runtime-options.html
- intellikit metrix gfx1151 PR: https://github.com/AMDResearch/intellikit/pull/189
