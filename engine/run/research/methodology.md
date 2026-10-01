# Kernel optimization methodology: static models, microbenchmarks, evidence pack

2026-10-01. Labels: **[F]** published fact, **[M]** measured on this box (CPU only, no GPU), **[R]** recommendation or inference.

## 0. Where we stand

- Inputs: LOOM_RUNTIME.md "Causal accounting" onward, plus the scratch tools.
- The current model: cycles ~ (32 per WMMA + ~1 per non-WMMA VALU) / utilization.
  - It predicted mf at -0.34 M (measured -0.38) and u8f at -0.63 M (measured -0.44).
  - It has no term for waits, barriers, LDS queueing or occupancy, so utilization is a fudge factor.
- Many lost variants were visible in the ISA before any GPU run:
  - KSL alone put a `vmcnt(0)` between the k steps.
  - Q5_K + DECLOAD put ~48 `v_mov` between the MMA pairs.
  - Packed f16 decode raised the VALU count per WMMA.
  - SPLIT/DECW only moved VALU between waves; the issue model says that is net zero.
  - `pipeline(%c2)` needed 208 VGPRs, too many for a 32-wave workgroup.
- **[R]** About half the losers could have been rejected with zero GPU rounds.

## 1. Static performance modeling

**llvm-mca on AMDGPU**

- **[F]** llvm-mca accepts `amdgcn`.
- **[F]** An AMDGPU `CustomBehaviour` for `s_waitcnt` landed in 2021 ([D104730](https://reviews.llvm.org/D104730), [D104149](https://reviews.llvm.org/D104149); [header](https://llvm.org/doxygen/AMDGPUCustomBehaviour_8h.html)).
- **[M]** LLVM 22.1.8 and ROCm's llvm-mca with `-mcpu=gfx1151` agree, and are unusable for us:
  - A dependent chain of `v_wmma_f32_16x16x16_f16` costs 5 cycles per WMMA (latency 5, RThroughput 1). The real figure is 32.
  - `v_mul_lo_u32` is modeled as full rate.
  - A VOPD pair counts as 2 uops, i.e. *slower* than two plain VALU ops.
  - `global_load` has a fixed latency of 320 and `ds_load` of 20.
  - `s_waitcnt vmcnt(0)` with no consumer register does **not** stall (348 vs 339 cycles over 10 iterations; the timeline shows dispatch right after the load). It stalls only through register dependencies.
  - It models one wave: no co-resident waves, barriers or occupancy.
- **[R]** Do not invest in llvm-mca. Fixing it would mean patching `SISchedule.td` and the CB, and it would still be single-wave.

**AMD tools**

- **[F]** RGA has per-instruction live-VGPR analysis ("VGPR pressure" column, with hints for how many VGPRs to cut to save a block) and lists gfx1151 as a target ([GPUOpen](https://gpuopen.com/learn/visualizing-vgpr-pressure-with-rga-2-6/), [releases](https://github.com/GPUOpen-Tools/radeon_gpu_analyzer/releases/)).
  - RGA gives no cycle estimates for compute.
  - It compiles source; it does not take a foreign hsaco.
  - **[F]** [amd_matrix_instruction_calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) covers RDNA3 WMMA: register layouts and rates.
  - f16, bf16 and iu8 run at 1024 ops/WGP/clk; iu4 at 2048.
  - These are table values. Nobody has published gfx1151 measurements (see the [1bit-MONSTER PR #160](https://github.com/1bit-MONSTER/engine/pull/160) review).

**Simulators and analytical models**

- **[F]** Cycle-level simulators:
  - [NaviSim](https://bu-icsg.github.io/publications/2022/navisim_pact_2022.pdf) (PACT'22, Akita/MGPUSim lineage) is the only RDNA simulator: RDNA1/2, 9.9% mean error on RX 5500 XT and W6800. It has no WMMA and no RDNA3 VOPD.
  - MGPUSim (ISCA'19) and gem5's GPU model cover GCN/CDNA ([gem5 MFMA, 2025](https://arxiv.org/pdf/2501.18113)).
  - Accel-Sim/GPGPU-Sim are NVIDIA SASS only.
- **[F]** Analytical models:
  - Hong & Kim's MWP/CWP (ISCA'09), GPUMech (MICRO'14, interval analysis) and MDM (MICRO'20, memory divergence) are NVIDIA-calibrated.
  - [GCoM](https://doi.org/10.1145/3470496.3527384) (ISCA'22) adds per-sub-core structural stalls (functional-unit and bank limits) and gets 10% MAE against Accel-Sim, versus 44.9% for earlier models.
  - [Jarmusch & Chandrasekaran 2026](https://arxiv.org/abs/2605.04178): microbenchmark-parameterized analytical models on MI300A/B200 with ~1% MAE, where naive roofline was off by more than 95%.
- **[R]** The lesson from GCoM and Jarmusch: accuracy comes from modeling *structural* contention, which is exactly what our fudge factor hides. Microbenchmark-measured parameters beat generic models.

**Recommendation: build a small per-WGP discrete-event simulator ("wgpsim") [R]**

Simulate one WGP rather than one SIMD. Barriers, LDS and the workgroup span 4 SIMDs in WGP mode, and IQ4_XS's problem is the cross-SIMD LDS burst after a barrier.

Inputs:
- The hsaco disassembly (llvm-objdump), basic blocks mapped to roles (causal_time already does this).
- Trip counts from the low IR (causal_loom).
- Wave-id-dependent branches (decoder vs MMA waves) resolved per wave.
- Occupancy from the kernel descriptor (VGPRs with granule 24, LDS).

Model, one in-order instruction stream per wave:
- **Issue arbiter per SIMD.** Each cycle it picks the oldest ready wave per issue class (VALU/WMMA, SALU, LDS, VMEM, branch). Whether it is oldest-first or round-robin is a parameter.
- **VALU and WMMA.** A WMMA blocks the VALU path for P cycles (P=32 for f16, per type). Quarter-rate ops take 4 cycles, trans ops are separate, a VOPD pair takes 1 cycle, wave64 doubles. Dependent latency comes from `s_delay_alu` and is checked against measurement.
- **`s_waitcnt`.** Exact counter semantics: vmcnt returns in order; lgkmcnt returns out of order when SMEM is mixed in.
- **LDS.** One shared server per WGP. Service time = bytes / bandwidth × conflict degree. The conflict degree is computed statically by evaluating each `ds_*` address per lane; the generators know the layouts, or the VALU address chain can be interpreted.
- **VMEM.** Latency sampled from a measured distribution (L2/MALL/DRAM mix), with a bandwidth token bucket set to 1/20 of the device.
- **`s_barrier`.** Releases when the last wave arrives, plus a release cost.
- **Output.** Steady-state cycles per phase × phases, plus prologue and epilogue, × grid waves / (20 WGPs × resident WGs), with tail quantization.
- **Emitted diagnostics.** The same views we get from ATT (critrole, simdtl, wmmagaps), but *untraced*, without ATT's 1.5x distortion.

Size: ~1k lines Python.

Validation plan:
1. **Ledger.** Collect every variant we already measured, with hsaco and measured cycles (more than 20 points):
   - IQ4_XS kstore: 26.40 / 26.02 (mf) / 25.96 / 25.00 / 24.50 / 23.66 (LICM) / 26.98 (PK) / 26.77→26.39 (EPAD).
   - Attention: 106.6 / 100.3 / 86.5 / 80.2.
   - Q4_K KSL: 15.49 / 15.83 / 15.21.
   - Q5_K: 19.34 / 19.59 / 19.60.
   - IQ3_S, IQ3_XXS and Q3_K KSL pairs.
   - The ablation rows (noBarrier, noFetch, noStore).
2. **Hard checks first.** Simulated per-class instruction counts must equal SQ_INSTS_VALU/LDS/SALU per WMMA, since the counts are deterministic. Then compare the simulated WMMA-gap histogram and role wait shares against ATT *shares*.
3. **Fit at most 5 free parameters** (VMEM mean latency, LDS service, barrier release, arbitration policy, VALU-under-WMMA overlap) on half the ledger; test on the other half.
4. **Acceptance targets.**
   - Sign of Δ correct for every pair with |Δ| > 2%.
   - Kendall τ ≥ 0.8 within each kernel family.
   - Δ error under 1.5% of kernel cycles.
   - Absolute error matters less than ranking.
5. **Known hard cases** that must come out right: EPAD (bank conflicts), noBarrier costing us +2.7 M, and the KSL `vmcnt(0)` placement.

Do this in tiers. Tier 0 (ISA lint) and tier 1 (automated issue model) come first. They are hours of work and catch most of the documented losers. Build wgpsim after them.

## 2. Microbenchmarks worth running on gfx1151

- **[F]** Prior art:
  - [Chips and Cheese RDNA3](https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture): LDS and cache latencies, and VOPD rules. Operands in the same slot must use different banks, one destination must be even and one odd, no 3-source ops, and the compiler rarely pairs FMAs.
  - Strix Halo [memory](https://chipsandcheese.com/p/strix-halos-memory-subsystem-tackling) and [Infinity Cache](https://chipsandcheese.com/p/evaluating-the-infinity-cache-in) articles.
  - [RDNA3 power saving skews latency tests](https://chipsandcheese.com/p/latency-testing-is-hard-rdna-3-power-saving).
  - Methodology templates: Jia et al. "Dissecting Volta" (arXiv 1804.06826), Abdelkhalik et al. "Demystifying Ampere" (arXiv 2208.11174), [RDNA3 ISA guide](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf).
- **[R]** Method: Loom or inline-asm kernels.
  - Time in-kernel with `s_getreg_b32 HW_REG_SHADER_CYCLES` (20-bit) or ATT on a single WGP.
  - Run long enough to escape clock ramps. Report cycles, never ms.
  - These are not candidate timings, but still respect the 15 s gap between GPU runs.

What to measure, in priority order (each feeds a simulator parameter):

1. **WMMA**
   - Back-to-back issue interval per type: f16→f32, f16→f16, bf16, iu8, iu4 (expected 2x).
   - One wave vs 2-8 waves per SIMD.
   - Dependent D→C chain latency, and the D→A/B hazard cost (`v_nop`s).
   - **Key question: can another wave's VALU issue during the 32-cycle WMMA window?** That decides whether "+1 per VALU" is additive or hideable.
2. **VALU**
   - Per-op cost: `v_fma_mix`, `v_perm`, `v_cvt_*ubyte`, `v_mul_lo_u32`, trans ops.
   - Actual VOPD rate, and pairing during WMMA.
   - Dependent-VALU latency and the cost of `s_delay_alu`.
3. **LDS**
   - Latency for b32, b64 and b128.
   - WGP bandwidth.
   - Conflict cost vs stride (sweep the pitch the way EPAD did).
   - A 32-wave burst right after a barrier (the IQ4_XS pattern).
4. **VMEM**
   - Pointer-chase latency through L0/L1/L2/MALL/LPDDR5X.
   - Latency under load at 8 waves/SIMD.
   - Achieved bandwidth.
   - Whether vmcnt returns in order.
5. **Barrier** release cost with all waves pre-arrived, across 4 SIMDs.
6. **Clock vs data.** We already saw 2.56 vs 1.93 GHz on real data. Keep cycles as the metric.

## 3. Field methodology

- **Speed-of-light and issue-slot accounting.**
  - **[F]** Nsight Compute's SOL section reports each unit's throughput as a % of peak. rocprof-compute has SOL panels for MI parts; gfx1151 support is unverified.
  - **[F]** Instruction roofline: Ding & Williams (PMBS'19); [AMD version](https://arxiv.org/html/2110.08221v2) built on SQ_INSTS_VALU/SALU.
  - **[R]** For us the roofs are the WMMA pipe, VALU issue, LDS bandwidth, VMEM bandwidth and the barrier rate. Report cycles split into WMMA-busy, VALU-busy and idle-by-reason, so the parts sum to 100%.
- **Ablation.** Our knob ablations are standard practice. **[R]** Automatically check that each ablated ISA still contains the work: compare per-class instruction counts against the control. This is HIP's DCE lesson.
- **Autotuning**
  - **[F]** Ansor ([arXiv 2006.06762](https://arxiv.org/abs/2006.06762)) uses a learned cost model to pick which candidates to measure.
  - **[F]** [Kernel Tuner](https://arxiv.org/html/2407.11488) supports HIP, verification and observers. OpenTuner provides ensembles of search strategies. Triton's `autotune` times every config with `do_bench`.
  - **[R]** Under the one-round/15 s rule:
    - (a) Correctness first: md5 or accgate, ungapped.
    - (b) Static tiers rank the candidates.
    - (c) Time only the top k, one round each.
    - (d) Use the single round's per-dispatch spread (e.g. MAD over its 20 launches) as the noise band.
    - (e) A Δ inside the band is "no decision": resolve it on a production pp2048 row, not by repeating.
    - (f) Every result goes into the ledger and becomes model validation data.
  - **[R]** Report spread and units per Hoefler & Belli (SC'15).

## 4. How serious teams work

- **[F]** [HipKittens](https://arxiv.org/abs/2511.08083) (AMD CDNA):
  - Diagnose with rocprofv3 PMC (`SQ_LDS_BANK_CONFLICT`).
  - Solve for LDS access phases empirically.
  - Pin registers to avoid compiler copies (855→1024 TFLOPS).
  - 8-wave ping-pong (compute and memory roles swap per barrier) for balanced kernels; 4-wave interleave for imbalanced ones.
  - Producer/consumer wave specialization loses on AMD because registers are statically split.
- **[F]** [ThunderKittens](https://arxiv.org/abs/2410.20399): small tile primitives, kernels written against hardware roofs.
- **[F]** [FlashAttention-3](https://arxiv.org/abs/2407.08608): ping-pong scheduling between warpgroups and intra-warpgroup softmax/GEMM overlap. This relies on asynchronous tensor cores.
  - **[R]** On RDNA, VALU and WMMA share issue, so overlap only hides *latency*, never VALU *count*. This matches our finding that SPLIT and DECW lost.
- **[F]** CUTLASS profiler and Composable Kernel's ckProfiler sweep tile configs offline against a reference check.
- **[F]** llama.cpp perf PRs post `test-backend-ops -m perf` / `compare-llama-bench.py` tables per backend.
- **[R]** The common pattern:
  - a fixed harness per op family, never per kernel;
  - correctness before timing;
  - a ledger of every config;
  - explicit control (pinned registers, schedule fences) where the compiler is weak.

## 5. Proposal: `evpack`

**Input:** a generator invocation (format, knobs), the real-data registry entry, and optionally the HIP counterpart.

**Stages:**

1. **Build and static analysis (no GPU)**
   - Compile report and residency tier.
   - Descriptor (VGPR, LDS, waves/SIMD).
   - causal_loom roles × classes per WMMA.
   - **ISA lint:**
     - `vmcnt(0)`/`lgkmcnt(0)` before the first WMMA of a k-step;
     - `v_mov` per WMMA inside the loop;
     - quarter-rate ops inside the loop;
     - loop-invariant VALU recomputed every iteration;
     - static LDS conflict degree;
     - VGPR margin to the next tier.
   - Tier-1 issue-model prediction (and wgpsim's once it exists), as a Δ against the parent in the ledger.
2. **Correctness gate, ungapped:** bit check or accgate.
3. **One timed PMC round through a generic launcher.**
   - Buffer lists come from `lhargs.py`, never written by hand.
   - Read cycles, SQ_INSTS_*, LDS bank conflicts and sclk.
   - Report floor %.
4. **Optional ATT round:** critrole, simdtl, wmmagaps, opmix, used for shares only.
5. **Report**
   - SOL table and issue-slot accounting.
   - Predicted vs measured, with the residual.
   - Top causes ranked by cycles.
   - Diff against the parent and against HIP.
   - Appended to `ledger.jsonl`.

**What to automate first, by time saved [R]:**

1. **Generic harness plus real-data registry.** Removes the per-kernel harness, which costs the most person-hours.
2. **ISA lint plus tier-1 model as a pre-filter.** Seconds per candidate. Retroactively it flags KSL-alone, Q5_K DECLOAD, PK and SPLIT/DECW, and the occupancy check catches `pipeline(%c2)`.
3. **Ledger, backfilled from LOOM_RUNTIME.md.** Required to validate any model.
4. **Automatic DCE check on ablations.**
5. **gfx1151 microbenchmarks.** Ask the user before running them: they use the GPU.
6. **wgpsim**, validated per §1.
