# Prefill research pass (2026-10-01)

Where the Loom prefill stands after reaching HIP parity, what the hardware
actually does, which ideas are worth pursuing, and how to work from here.
Supporting material is in `research/`:

- `hw-measured.md` and `wmmarate.hip` / `wmmarate.sh`: the microbenchmark
  and its results.
- `correctness.md`: correctness tiers.
- `lowprecision.md`: int8/int4.
- `profiling-tools.md`: tools on gfx1151.
- `methodology.md`: static models and workflow.
- `floor.py`: the floor table.
- `gpumetrics.py`: gpu_metrics decoder.

## Ground rules (user decisions)

- **Stock compiler only.** Production artifacts and committed code must build
  with stock HRX/Loom. The experimental worktree (`/home/q/hrx-wt`) is a
  diagnostic instrument only. A compiler-side fix is reported and, if
  wanted, proposed upstream; it is never shipped from a fork.
- **Timing.** One round per candidate, a 15 s gap, no repeats and no clock
  pinning: pinned clocks diverge from real use. Compare cycles
  (SQ_BUSY_CYCLES). Late-session drift is handled by comparing changed rows
  against the drift of the others.
- **Correctness.** md5 bit-identity for exact rewrites. Numerics changes go
  through the tiered gate below.
- **Low precision.** On this chip int8 is only interesting for
  memory/LDS/decode, and only if those can't be worked around more simply.

## 1. Where we are

Untraced, one round each, today:

| | pp2048 | pp8192 |
|---|---:|---:|
| Loom | 3450 ms | 15256 ms |
| HIP | 3430 ms | 15701 ms |

Parity. HIP's own pp8192 number moves ±9% between rounds.

The WMMA floor table (`floor.py`) uses device times from the p48 profile (2048)
and the clean-8192 set, against a floor of 34 cycles per WMMA (measured, §2):

- **pp2048.** WMMA-bound rows are 3267 ms of 3524 ms, at ~63% of the
  floor, which leaves ~1200 ms of headroom.
  - The five big FFN GEMM rows hold over half of it:

    | row | above floor |
    |---|---:|
    | IQ3_S gate+up | 230 ms |
    | IQ3_XXS gate+up | 141 ms |
    | IQ3_S down | 140 ms |
    | IQ4_XS gate+up | 140 ms |
    | IQ3_XXS down | 103 ms |

  - The worst efficiency is the IQ3_S K=6144 down projection, at ~47%.
  - Non-WMMA rows add 257 ms: DeltaNet 104, half_norm 42, ssm_conv 36,
    ssm_postnorm 27.
- **pp8192.** The same picture; attention climbs to 6th place (315 ms above
  its floor, ~48% efficient) because it is quadratic.
- **Pipeline tax.** IQ4_XS kstore reaches ~78% of the floor standalone but
  ~66% inside the pipeline. Standalone kernels have run 15-20% faster than
  the same HAL in pp2048 before. The cause is unknown, and it is a lever
  across every kernel at once.

## 2. Measured hardware facts (gfx1151)

All from `research/wmmarate.hip` under PMC, one round each.

- **WMMA rate.**
  - f16, bf16 and iu8 each take **34 cycles** per WMMA per SIMD; iu4 takes
    **17**. AMD's tables say 32/16.
  - Dependent back-to-back accumulation costs nothing extra, even with one
    wave.
- **VALU next to WMMA costs ~1 cycle each**, with nothing hidden: +1 VALU
  adds 2.0 cycles; +8 add 1.07 each; +32 add 1.03 each. This confirms the
  issue model `cycles ≈ 34·WMMA + Σ VALU issue`, so instruction count is the
  currency.
- **VOPD (dual issue) saves nothing next to WMMA:** about 1.9 cycles per
  pair, the same as two plain VALUs. HIP's paired moves and multiplies were
  no advantage.
- **LDS.**
  - A conflict-free `ds_load_b128` costs ~2.8-3.3 cycles of WMMA time while
    under bandwidth.
  - Bandwidth is about **128 B/clk per WGP**: at 4 b128 per WMMA it costs
    7.7 cycles each.
  - Bank conflicts (64 B lane stride) cost about 4x.
  - Our tile GEMMs issue ~1.7 LDS instructions per WMMA. If those are
    mostly b128, that is ~100 B/clk per WGP, possibly **~80% of LDS
    bandwidth**. This is the deciding measurement for any bytes-reduction
    idea (§4).

## 3. Correctness: a tiered gate (build this first)

Full design in `research/correctness.md`.

The current gate is weaker than it looks. The `ids{2048,8192}` prompts
contain 10 distinct tokens ("The capital of France is Paris..." repeated),
so a last-token KL is near zero by construction.

- **T0, exact:** hidden md5, as today.
- **Tk, kernel unit test:** NMSE against an f64 numpy reference on real
  captured activations, plus adversarial rows: the outlier channel, zero
  blocks, values near the f16 maximum.
- **T1, rounding-level** (f16 intermediates, fused rounding):
  - Data: 4 wikitext windows of 2048 tokens, scoring positions 1024-2047.
  - Metrics: full-f32 KL per position (mean, p99.9, max), tie-aware top-1
    flips, and per-layer residual relative RMS.
  - Thresholds calibrated from HIP vs the frozen golden on the same text.
  - Cost: about 1-1.5 min.
- **T2, quantization-level** (int8 activations and similar):
  - Data: 16 windows plus one 8192-token window.
  - Metrics: mean KLD with a bootstrap CI, p99.9, same-top-1, ln PPL ratio.
  - Provisional thresholds: KLD ≤ 1e-3, same-top-1 ≥ 99%.
  - References: the Loom golden, plus llama.cpp ROCm on the same GGUF,
    which already uses int8 q8_1 activations on gfx1151.
- **T3, release:** wikitext-2 PPL at ctx 2048 against llama.cpp.

Engine changes needed:

- All-position logits from `loom_forward_pp` (HIP's `yah-run` already has
  `--dump-all-logits`).
- A per-layer residual dump in Loom (HIP has `YAH_DUMP_DIR`).
- `accgate.py` v2 with multiple windows and confidence intervals.
- A frozen tokenized wikitext corpus (`wiki.test.raw` is not on disk yet;
  ~4 MB download).
- llama.cpp's uint16 KLD file format cannot resolve rounding-level changes
  (its floor is 5.5e-4), so compare against our own f32 logits.

Numerics hazards found:

- Activation channel 3994 is a large outlier (58x RMS in GEMM inputs; 429 in
  the residual).
- Per-32 int8 activations add ~1% relative error, 20-40x f16 rounding, and
  ruin the outlier's own block.
- int4 activations add 4-11% error, so they are not viable without outlier
  handling or rotation.
- RMSNorm's sum of squares must stay f32: it reaches ~2.7e5, above the f16
  maximum.

## 4. Low precision: verdict

Details in `research/lowprecision.md`.

- **int4 (iu4, 2x rate): not viable.**
  - It needs 4-bit *activations*, which require rotations (QuaRot/SpinQuant)
    for quality.
  - It cannot represent the IQ4_XS non-linear codebook.
  - Splitting into nibbles costs ≥2 iu4 ops, which equals one iu8.
- **int8 (iu8, same rate as f16): roughly breakeven at best.**
  - Decode gets cheaper: a codebook LUT and no conversions.
  - The GGUF per-32 (Q6_K/Q3_K: per-16) sub-block scales cannot be folded
    exactly into int8; IQ4_XS's product reaches 4064.
  - Applying them costs ≥ about 4 VALU per WMMA, so the total is ~5.5-7
    against today's 4-7.
  - llama.cpp's int8 MMQ on Strix Halo (IQ4_XS pp2048, Llama-8B) reaches an
    estimated ~27% of f16 peak; on Q6_K it loses to its own
    dequantize+hipBLAS path.
- **Possible bug if we ever use it:** on gfx11, stock Loom emits
  `neg_hi = 3` for iu8. clang emits 0, and AMD says integer forms need 0.
  Test mixed signs first.
- **The one remaining reason for int8** is halving fragment bytes, if §2's
  LDS-bandwidth suspicion is confirmed. Larger per-wave tiles (better
  fragment reuse) attack the same bytes without changing numerics.
- **The more promising alternative is decode-free GEMMs.**
  - Dequantize each layer's weights to an f16 scratch buffer once, then run
    a pure f16 GEMM.
  - Estimate: the pass costs ~10% of the GEMM at M=2048 and ~3% at M=8192,
    against 12-22% of issue time spent on fused decode today.
  - It should win at long context and is cheap to falsify: time a kernel
    that loads pre-dequantized f16. The old `yah-hal-sd-ab*` ablations may
    already answer it.
  - It is exact: the same f16 values, so md5-gateable.

## 5. Profiling: what to adopt

Details in `research/profiling-tools.md`.

- **400 gfx1151 counters are already installed.** Point
  `ROCPROFILER_METRICS_PATH` at rocprof-compute's `sdk_config.yaml`
  (procedure in the report).
  - This unlocks `SQ_INST_CYCLES_VALU`, `SQ_INST_CYCLES_LDS`,
    `SQ_WAIT_BARRIER`, `SQ_WAIT_INST_ANY` and the `SPI_RA_*` occupancy
    limiters.
  - That is a speed-of-light view (VALU/WMMA busy, LDS busy, barrier tax)
    **without ATT's 1.5x distortion**.
  - Validate first against the microbenchmark, where the WMMA and VALU
    counts are known.
- **rocprof-compute 3.6** supports gfx1151: Speed-of-Light, memory chart and
  WGP panels. It has no WMMA roofline, and it replays per counter pass, so
  use it on loomhip standalone cases.
- **Useful, not yet installed:**
  - **RGA** (binary mode, per-instruction live-VGPR curves; offline).
  - **ROCprof Compute Viewer** (an interactive browser for our existing ATT
    output).
- **rocprof-sys / `rocpd2pftrace` timelines** cover the pipeline-tax
  question on the HIP side. HRX dispatches are invisible to them, so use
  `iree-profile` plus `gpumetrics.py` throttle-counter diffs on the Loom
  side.
- **Unavailable:** PC sampling (no gfx11 support in this KFD) and RGP for
  HIP (Windows only).

## 6. Workflow

Details in `research/methodology.md`.

About half of today's losing variants were visible in the ISA before any GPU
round. Examples: a `vmcnt(0)` between k-steps, ~48 `v_mov` between MMA pairs,
VALU merely moved between waves, a VGPR-tier jump. llvm-mca cannot help: it
models a WMMA as 5 cycles, VOPD as slower than scalar, and one wave only.

Plan, in order of time saved:

1. **Generic evidence-pack harness (`evpack`) plus a real-data registry.**
   - Inputs: a generator invocation and a real-input registry entry.
   - Stages: compile report, ISA lint, issue-model prediction, correctness
     gate, one PMC round, optional ATT.
   - Output: the report plus a `ledger.jsonl` row.
   - It ends hand-written harnesses: the buffer lists come from
     `lhargs.py`.
2. **ISA lint plus issue-model pre-filter.** Seconds per candidate; time
   only the survivors.
3. **A ledger backfilled from LOOM_RUNTIME.md.** It is the validation set
   for any model.
4. **Automatic DCE check on ablations.**
5. **`wgpsim` later:** a ~1k-line per-WGP discrete-event model (4 SIMDs, a
   shared LDS server, barriers, exact waitcnt) with ≤5 fitted parameters.
   - Parameters come from §2's microbenchmarks.
   - Accept it if it gets the sign and ranking of past deltas right.

## 7. Ranked next steps

1. **Correctness tiers T1/T2** (corpus, all-position logits, residual dump,
   accgate v2). This unlocks every numerics experiment.
2. **SOL counters plus `evpack` and lint.** Verify the LDS-bandwidth
   suspicion and measure the VALU/LDS/barrier busy split per big GEMM.
3. **Decode-free GEMM probe** (pre-dequantized f16 → pure f16 tile GEMM) on
   IQ3_S and IQ4_XS: exact, and potentially a large win at long context.
4. **Pipeline-tax investigation:** standalone vs in-pipeline, using
   throttle counters, L2 counters and dispatch gaps.
5. **IQ3_S GEMMs,** the biggest single headroom at ~47-63% of the floor,
   and the K=6144 down projection.
6. **Long-context attention** (~48% of the floor at 8192).
7. **Q4_K** carried-prefetch copies: only with a generator-side
   restructuring, since the allocator route is excluded.
