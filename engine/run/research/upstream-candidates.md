# Loom / HRX patches worth proposing upstream

Every item comes from a limitation we hit while tuning the prefill and either
worked around or left on the table. Production stays on stock HRX/Loom: these
are candidates to propose upstream (never pushed without the user's OK). Local
prototypes are diagnosis only, like the `HRX_PROFILE_MODE=counters` patch.

Ranked by expected value for this engine.

## 1. HRX: per-dispatch "no trailing ordering barrier" flag (concurrency)

**Limitation.** Kernels never overlap.
- `hrx_stream_dispatch` records an execution barrier after every dispatch
  (`libhrx/src/libhrx/stream.c`, `hrx_stream_record_ordering_barrier`).
- In the AMDGPU AQL command buffer, a dispatch packet gets the barrier bit
  exactly when an execution barrier is pending
  (`runtime/.../amdgpu/aql_program_builder.c:556-563`).
- The driver also provisions a single hardware queue per GPU
  (`IREE_HAL_AMDGPU_DEFAULT_GPU_AGENT_QUEUE_COUNT = 1`), and `hrx_queue_dispatch`
  sets the barrier bit on every packet (`host_queue_dispatch.c:879-888`).

**Patch.** A flag such as `HRX_DISPATCH_FLAG_NO_ORDERING_BARRIER` on
`hrx_stream_dispatch` that skips the trailing ordering barrier. The next
dispatch's packet is then emitted without the barrier bit and may start while
this one runs. The caller keeps ordering with explicit
`hrx_stream_execution_barrier`. It is small and opt-in; nothing changes for
existing callers.

**Use here.**
- SSM block: dispatch DeltaNet flagged, then the z-projection GEMM (6144 rows,
  ~9 M cycles, WMMA-bound) behind it with no barrier, then `postnorm`
  (barrier). DeltaNet (~4.2 M cycles, VALU-bound) and the z GEMM use
  different units. Up to ~4 M cycles per SSM layer if fully overlapped: ~2.6%
  of the pp2048 prefill, realistically about half.
- Also: the two 48-row alpha/beta projections (~0.8 M cycles per layer,
  latency-bound) beside a big GEMM, attention k/v projections beside the q
  projection, and every kernel's fill/drain tail.
- A barrier-bit packet waits for all earlier packets, so overlap pairs must
  be arranged by dispatch order. Real cross-dependencies would need
  barrier-AND packets or multiple hardware queues (item 1b).

**1b (bigger).** Provision 2+ hardware queues and let streams pick one, with
cross-stream ordering through `hrx_stream_wait_event` (semaphores). This gives
general dependency graphs, but it is a larger runtime change.

**Prototype result (2026-10-01).** Local libhrx patch (`stream.c` skips the
trailing ordering barrier for flag bit 2; never committed upstream).
`loom_forward_pp` with `YAH_CONCUR=1` dispatches DeltaNet flagged, then the z
GEMM, then postnorm.
- md5 unchanged.
- Device timestamps: the z GEMM starts ~1 us after DeltaNet and overlaps all
  of it in all 48 SSM layers. Each slows while sharing (DeltaNet ~+50%, GEMM
  ~+20-30%).
- DeltaNet + z GEMM over 48 layers: 270.6 ms serial -> 251.9 ms concurrent
  (-6.9%, -18.6 ms, ~0.6% of pp2048). Below the ~1.4% first estimate: the
  GEMM is near its issue bound, so DeltaNet's VALU steals its issue slots.
- Untraced wall time (single runs, idle start): +2.3% / -1.5% in two pairs,
  i.e. inside the ~±2.5% run-to-run noise. The clock was unchanged (2218 vs
  2209 MHz), so no thermal penalty.
- Verdict: real and exact, but modest for this pairing. Further pairs (the
  48-row alpha/beta GEMMs, attention-side prep, kernel fill/drain tails) are
  each smaller. Worth proposing upstream as a general capability, not on this
  number alone.

## 2. Loom: loop-invariant code motion after view linearization

**Limitation.** Address math recomputed inside K loops.
- Attention: hoisting the loop-invariant address math would bring VALU to
  HIP's count (1.126 vs 1.119 G, LOOM_RUNTIME "Attention ... bit-identical").
- GEMMs: the worktree LICM build (`/home/q/hrx-wt`, `LOOM_EXP_LICM=1`, p47)
  gave IQ4_XS kstore 23.66 M cycles and pp2048 IQ4_XS rows -2%. It was
  withdrawn because it was not a stock compiler.
- LSE carries the same idea as a two-line pipeline patch
  (`mac-amdgpu/patches/hrx/loop-invariant-motion.patch`: run the existing
  `licm` pass after `linearize-view-accesses` in
  `loom/src/loom/target/pipeline.c`).

**Patch.** That pass-order change, with the register-pressure guard the p47
experiment needed (hoisted values stay live across the loop; attention spilled
when hoisting was unbounded).

**Value.** About 1-2% on GEMMs, and the main known lever for long-context
attention.

## 3. HRX: `hrx_stream_query` correctness + a non-spinning wait

**Bug.** `hrx_stream_query` reports complete when `stream->timepoint == 0`,
even with recorded-but-unflushed work (`stream.c` ~205-214; dispatches stay in
`pending_cb` and do not advance the timepoint). We hit it when a sleep-poll of
`hrx_stream_query` returned at once.

**Limitation.** `hrx_stream_synchronize` -> semaphore wait busy-polls the KFD
(~400-460k `AMDKFD_IOC_WAIT_EVENTS` per prefill, one host core at 104%). We
work around it with an event-tail sleep-poll in `LoomDevice::SetSleepSync`.

**Patch.**
- Query honours `has_pending_work` (flush first, or report incomplete).
- Add a blocking or interrupt-driven wait mode (or a sleep-poll option) for
  long waits.

**Value.** Correctness; ~8 W less host power during long waits. The GPU clock
was not power-limited, so no speed change.

## 4. Loom: back-edge copy coalescing for loop-carried values

**Limitation.**
- Q4_K's K loop copies its carried prefetch every phase (24 `v_mov` per phase
  at 4 x 4).
- Attention has 33 latch copies per tile (accumulators not coalesced across
  the back edge).
- `allocation/edge_alias.c` / `loop_edge_relocation.c` refuse the relocation
  in these cases.

**Patch.** Allow coalescing of yielded values into their block-argument
registers when the live ranges permit (or a cheaper parallel-copy plan).

**Value.** Mostly attention; small on GEMMs now that Q4_K runs 4 x 2.

## 5. Loom: low-window-aware allocation / selection

**Limitation.** `v_cvt_f16_f32` results may only use v0..v127
(`descriptors/alu.py` operand window, applied in
`allocation/target_constraints.c` `apply_operand_window`). With 128 VGPRs of
accumulators live, the linear scan fills the low window first, then evicts
accumulators to scratch for every conversion result. Q4_K at 4 x 2 ran 3x
slower until worked around.

**Workaround in use.** `fptrunc(fma(x, opaque_one, y))` selects `v_fma_mix`
(any VGPR). The 1.0 must be opaque because the canonicalizer folds
`fma(x, 1, y)` back to `addf` (`ops/scalar/canonicalize.c` ~1322-1350).

**Patch (any one).**
- Allocate wide long-lived tuples (accumulators) from the high window when
  low-window-constrained results exist in the loop.
- Select `v_fma_mix*` for `fptrunc` when the low window is under pressure.
- Do not fold `fma(x, 1, y)` when its operands come from a mixed-precision
  narrowing.

**Value.** Robustness; removes a hack and a trap for every future kernel with
128 accumulator VGPRs.

## 6. Loom: two-address FMA (`v_fmac`) formation

**Limitation.** `scalar.fmaf` always lowers to three-operand `v_fma_f32`, which
VOPD cannot pair. Only `vector.dotf` / `vector.reduce` lowerings emit
`v_fmac`.
- DeltaNet's 672 FMAs per 8 tokens could not dual-issue in wave32; forcing
  pairs through `dotf` hurt latency.
- We took the gain with wave64 instead (-16.5%).

**Patch.** Convert `v_fma_f32` to `v_fmac_f32` when the addend dies (as LLVM's
two-address pass does), and let the VOPD planner pair with a latency-aware cost.

**Value.** Small for us now (DeltaNet is wave64); general for wave32 VALU code.

## 7. Loom: scheduler register-pressure awareness for fragment loads

**Limitation.** The scheduler hoists fragment loads early.
- Attention held 5 of 8 K fragments live at the VGPR peak.
- Q4_K's KSL step held all 10 fragments of a step.
- We place `scf.schedule.fence` by hand (QKFENCE, KSL fence, RHSO/RHSF).

**Patch.** A pressure-aware limit on how far loads hoist ahead of their
consumers.

**Value.** Removes hand-placed fences; small direct speedup.

## 8. HRX profiling: counters mode + gfx1151 counter map

**Limitations.**
- `HRX_PROFILE_MODE=counters` exists only as our local patch to
  `libhrx/src/libhrx/runtime.c`.
- `TCC_EA0_*` (DRAM traffic) are not mapped for gfx11.5.1
  (`profile_counters.c:752 UNIMPLEMENTED`).

**Patch.** Upstream the counters mode; map the TCC_EA counters for gfx1151.

**Value.** Tooling. Measured bytes for memory-bound kernels instead of
computed ones.

## 9. Loom: small missing ops / friction

- **A sleep op (`s_sleep`).** The workgroup stagger uses a loop of 8000
  workgroup barriers as a delay.
- **A shader cycle-counter read (`s_memtime` / `s_getreg SHADER_CYCLES`).** For
  in-kernel phase instrumentation; we use ATT traces instead.
- **`scalar.select`** (only `scf.select` on index today).
- **Bound proofs through div/rem index remaps.** The kqg epilogue needed a
  redundant `index.min` clamp to pass SUBRANGE/010.

## 10. Loom: miscompiles found writing the FA attention (2026-10-01)

Correctness bugs, so ahead of the performance items when proposing.
Reproducers: `engine/run/research/fa/` and the `gen_attn_fa.py` knobs.

- **Barrier release does not drain outstanding LDS loads.** `kernel.barrier
  <workgroup> ordering(acq_rel)` emitted `s_barrier` with fragment
  `ds_load`s still in flight (no `s_waitcnt lgkmcnt(0)` first). Waves that
  pass the barrier and overwrite that LDS race with the loads (WAR). Release
  semantics must cover prior reads as well as writes.
- **Values carried between two sequential `scf.for` loops.** Without a loop
  policy the WMMA accumulators came out corrupted at the hand-off into the
  second loop. Fine with `unroll(%c2) schedule(recurrence)` or with one loop.
- **A multi-result `scf.if` (8 x vector<8xf32> + 2 f32) on a
  `subgroup.vote.any` condition** produced NaN everywhere in one
  configuration; the same code is correct in another.
- **Perf: `s_waitcnt vmcnt(0)` at the loop back edge** whenever loaded
  registers are carried across it. That drains the prefetch: 15% of wave time
  in the attention, and the same mechanism hit the GEMM decode-ahead.

- **CSE + tied in-place op.** Two `vector.fmaf` on vector<8xf16> with the
  same addend (a splat) were CSE'd to one value and both `v_pk_fmac_f16`
  results were tied to it without a copy, so the second read the first's
  output. Silent miscompile; repro research/fa/unpk.loom. In another shape the
  same tie hit `coalescing.c:1637 low tied result cannot share the operand
  location`.
- f16 vector arithmetic other than fmaf (subf, uitofp) is rejected by the
  `amdgpu.arithmetic.vector_f32` constraint.

## Not candidates (measured, no gain)

- Wave64 tile GEMMs: wave64 VALU/LDS instructions cost ~1.7x.
- More residency via smaller K steps.
- Decode-free f16 GEMMs.

## Appendix: dual-issue audit (p59 kernels, 2026-10-01)

Busiest VALU block per production kernel (`llvm-objdump`):
- Dual-issue (`v_dual_*`) is 0-9% of VALU everywhere.
- Attention (`yah_attn_hip`): 1935 VALU, 0% dual, 89% single-issue FP32 (the
  softmax). It runs at half the FP32 VALU rate RDNA3 offers; the main
  dual-issue lever left. It is also WMMA-heavy, so plain wave64 is not
  automatically a win.
- GEMM K loops: 2-9% dual, but mostly integer decode (shifts/and/bfe), which is
  largely hidden behind WMMA (~0.23 cycles per removed VALU, measured).
- half_norm / conv / postnorm / rope: single-issue FP32, but memory-bound.
- DeltaNet: wave64, so single instructions already use both ALU halves.

## 11. Integer KV-cache decode lowering (2026-10-02, kv8a16 / kv4a16 attention)

Decoding int8/int4 K/V tiles to f16 while staging costs ~3 VALU per WMMA per
cache (standalone layer 3 at pp8192: fp16 64.49 M cycles; K8 only 69.16, V8
only 68.17, both 72.86; K4 68.96, V4 68.55, both 72.44). Cycles track the extra
VALU at ~1.4 M per +1 VALU/WMMA. Per output f16 pair the stock compiler emits
shift + v_and + v_or + a register copy + v_pk_fmac_f16 (5 ops):
- v_pk_fmac_f16 (tied accumulator) is chosen even when the addend is a splat
  reused by 8 FMAs. Each needs a v_mov/v_or copy of the addend (8 extra ops
  per 16 values). The untied VOP3P v_pk_fma_f16 avoids them.
- (x & M) | G is not fused into v_and_or_b32 / v_bfi_b32. Byte gathers
  ((w >> 8k) & 0x00ff00ff) are not selected as v_perm_b32: the fp8/fp4
  encoding lowerings already have perm/bfi descriptors.
- vector.decode has no AMDGPU lowering for i8/u8/u4 payloads ("no target-low
  contract"), and its auxiliary operands accept only 'scale': no zero point /
  min, although encoding.matches documents affine = scale_plus_min. With
  i8 x scale (symmetric K8) or u4/u8 scale_plus_min (V, asymmetric K4), the
  decode would be ~2 ops per pair (perm/bfi + one fma).
Expected gain if fixed: about half of the a16 overhead (+12-15% attention ->
~+6%).
