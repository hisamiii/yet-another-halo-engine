# Chunked Gated DeltaNet (WY form): groundwork, 2026-10-02

DeltaNet (yah_deltanet, recurrent, HIP order) is 2.58% of pp8192 cycles
(749 M of 29.0 G).

The current kernel's math, with k_hat = inv_k k and q_hat = q_scale q
(prep_kq), and value head h reading key head h mod 16:

    S_t = a_t S_{t-1} + b_t (v_t - a_t S_{t-1} k_hat_t) k_hat_t^T,  o_t = S_t q_hat_t

Layer-0 dumps (pp2048, YAH_DUMP_LAYER=0: .conv .kq .ab .raw) checked by
gdn_ref.py:

| check | relative error |
|---|---|
| float64 recurrence vs the kernel | 0.9-2.8e-7 |
| chunked WY form (C = 64, FLA chunk_gated_delta_rule) vs the recurrence | 4-7e-16 |

The decay goes down to a = 0.149, so a kernel must use log-space gating
(exp(G_i - G_j), i >= j, never a division).

Precision (gdn_prec.py): the chunked form with every matmul input rounded
gives output relative error

| inputs | relative error |
|---|---|
| f16 | 1.2-4.2e-4 |
| bf16 | 0.9-3.4e-3 |
| f32 | ~6e-8 |

End to end, YAH_DN_F16SIM=1 rounds the recurrent kernel's k, q, v, the state
read by both dot products and the update coefficient to f16 (state kept f32).
8K gate on 4 docs: mean KLD 0.000003, 99% precision 99.98%, same top 99.97%.
That is 7x below kv8a16, so a plain f16-WMMA chunked kernel is safe;
compensated (hi+lo) inputs are not needed.

## Kernel: tools/gen_gdn_chunk.py (opt-in, not wired into the emitter)

C = 32, workgroup = (64 value rows, one head), 8 waves (wave32), state in f32
WMMA accumulators. Standalone pp2048 layer 0 (cyc.sh, SQ_BUSY_CYCLES):

| version | M cycles | output rel vs recurrent | change |
|---|---|---|---|
| recurrent yah_deltanet (production) | 4.116 | - | - |
| v1 | 5.270 | 2.1e-4 | first correct version |
| v1, solve ablated (T = I) | 4.045 | (wrong) | the solve was 23% |
| v3 | 4.694 | 2.1e-4 | right-looking solve (independent fmas), O1 = Q S^T overlapping the solve, state f16 copy moved into phase 1 |
| v4 | 3.717 | 2.1e-4 | prefetch of the next chunk's inputs (loop-carried) and the decay scan via kernel.subgroup.scan in one barrier |
| v4, solve ablated | 3.542 | (wrong) | the solve is now ~5% |
| v5 | 3.702 | 2.1e-4 | Vn^T into the dead T1/T2 region, one barrier fewer |
| v6 (reverted) | 3.744 | 2.1e-4 | packed (lane ^ 1 shuffle) f16 pair stores for the state copy: VALU 34 -> 43.5 M for LDS 15.2 -> 14.4 M; the scalar stores queue behind LDS traffic rather than limiting by count |

Final state vs the exact recurrence: 2-6e-4.

v4 ATT: s_barrier 55%, ds_store_b16 10.6%, lgkmcnt waits ~12%. Eight
barrier-separated phases per 32 tokens at 2 workgroups per WGP (LDS ~63 KB),
so waves idle behind the slowest one in every phase.

Bugs found on the way:
- the RDNA3 wave32 WMMA accumulator holds rows 2 i + lane / 16 (interleaved,
  not contiguous): debug mode GDN_DBG=1 dumps LDS after each phase, compared
  in dbgcmp.py;
- log2f needs <afn>;
- index arithmetic must stay provably non-negative (u32 address math).

Next structural step (not done): FLA's 3-kernel split.
1. A, T, W, U for all chunks in parallel (no sequential dependency).
2. A lean sequential state kernel: per chunk Vn = U - W S^T and the update.
3. Outputs O = Q S_c^T + P Vn in parallel over chunks, from per-chunk state
   snapshots (f16: 128 x 128 x 2 B per head per chunk).

## The FLA-style split, evaluated (2026-10-02): not worth it here

- **FLA's per-chunk state snapshots:** at pp8192 that is ~400 MB written and
  read back per layer (f16, C = 32). That is ~3 ms per layer at ~256 GB/s,
  ~150 ms over 48 layers, against ~350 ms for DeltaNet in total. Not viable
  on this APU.
- **Without snapshots:** a parallel prep kernel (K K^T, Q K^T -> P, solve ->
  T', T'') plus a sequential state/output kernel. GDN_ABL=prep measures the
  state/output part alone: 3.356 M cycles vs 3.702 M for the fused v5. The
  prep is only 0.35 M (9%), so the split's best case is about -18% vs
  recurrent, realistically -12-15% with kernel A's own cost.
- **What bounds it:** the state/output phases are LDS-bandwidth- and
  barrier-bound. Per chunk each wave does ~26 MMAs, each with 2 fragment loads
  (2 x b128) from LDS, plus the f16 copy of the 64 x 128 state and the
  K / Kt / V / W / Vn staging stores. With 16 waves per WGP that is ~2x the
  WMMA time in LDS traffic (estimate).
- **Possible further directions:** keep shared operands in registers (S
  fragments shared by X and O1 are loaded twice in v5), C = 64 to halve the
  per-token state-copy cost (but 1 WG per WGP), or a different row/key split.
  Each is worth single-digit percent of DeltaNet.

## Shipped (2026-10-02): p70 = p63 + chunked GDN (v5 + O1LATE + alpha clamp)

- Standalone pp2048: 3.566 M cycles vs 4.116 M recurrent.
- In the pipeline: DeltaNet -10.5% (pp2048) / -12.9% (pp8192).
- Bug found by gate v2 at 8K/32K (not by the layer-0 harness): alpha
  underflows to 0 for some tokens deep in the model. log2 -> -inf and the decay
  differences -inf - -inf give NaN, which then stays in the state (arXiv went
  non-finite from position 4800). Fix: clamp log2 alpha at -100.
- Lesson: the standalone harness needs inputs from several layers / documents,
  not only layer 0 of one window.
