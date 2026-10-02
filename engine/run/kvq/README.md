# Gate v2: KV-cache compression quality at long context

Why a new gate: the T1/T2 gate (engine/run/gate) scores 512 positions of
2048-token windows against a frozen golden.
- Its mean KL saturates for any non-bit-exact change (perturbations grow
  chaotically through 64 layers).
- Its windows are too short for KV compression.
- It has no long-range retrieval test and no confidence intervals.

Design follows the evaluation-methodology digest (arXiv 2607.09683,
"Ablation, Statistical Inference, and Validation for KV-Cache Compression")
adapted to a prefill-only engine on one machine.

## Engine support (loom_forward_pp)

- **Chunked prefill** (emit_prefill_pp.py `YAH_CTX=T` with a chunk size B,
  e.g. 2048). Kernels run at B, the KV cache holds T. One rope/attention HAL
  per chunk (start_pos = i B). The DeltaNet state and the short-conv state
  (yah_conv_state) carry between chunks.
  - Bit-identical to one-pass at 8K (hidden and logits).
  - 32K prefill 68 s, 64K 160 s, GTT peak 7.3 GB at 64K. The 24 GB GTT limit
    is not binding; weights are served from the mmapped model.
- **YAH_KV_HOOK="cmd"**: a persistent child process sees every chunk's freshly
  written f16 K/V rows of every attention layer, plus every n-th roped Q row
  with YAH_KV_HOOK_QSTRIDE=n, and returns them. This is fake quantization:
  quantize -> dequantize in kvcodec.py, then the fp16 attention kernel.
  - Any scheme can be evaluated end to end without an attention kernel.
  - Passthrough is bit-identical to no hook.
- **YAH_ROWSTATS=file**: per scored position, logsumexp, next-token logit,
  argmax, top-1/top-2 and the top-64 (id, logit). That is 540 B per position
  instead of a 1 MB logit row. Positions: every YAH_ROWSTATS_STRIDE from
  YAH_ROWSTATS_FROM, or a list (YAH_ROWSTATS_POS).

## Tiers

**A, per-layer fidelity (tierA.py).**
- Dump all 16 attention layers' K/V at 32K plus sampled roped Q
  (`kvcodec.py --dump`).
- Replay codecs offline, chunk by chunk as the engine streams.
- Per layer: attention-output relative error, softmax KL, by query-position
  bin, with K-only / V-only splits.
- No compounding through 64 layers, so this is the main discriminator between
  schemes.

**B, long-document NLL (gate2.py).**
- Public-domain books (Project Gutenberg, tokenized with the model's
  vocabulary), 32K windows, every 8th position from 1024.
- Per position: dNLL (candidate - reference), KL over the reference top-64,
  flips (reference gap > 0.1).
- Binned by context position; 95% block-bootstrap CI (blocks of 32 sampled
  positions, across documents).

**C, multi-key retrieval (gen_needle.py, needle_score.py).**
- 32 near-duplicate keys ("agent Falcon-17" / "Falcon-71") with 7-digit
  codes, inserted at depths 5-95% of book text. 8 questions at the end.
- Scored teacher-forced: log p(answer), exact match, minimum answer-token
  margin.
- Accuracy alone saturates for a 27B model; margins and log p do not.

**Reference.** fp16 KV through the same engine (fits to 64K here), so only the
codec differs.

**Anchors.**
- int8 (`--k int8 --v int8v`, must pass).
- bad3 (3-bit K per token, must fail).
- Thresholds come from these anchors, not by hand.

## Codecs (kvcodec.py)

| K | |
|---|---|
| int8 | centred, per token-half, 127 levels (our int8 K) |
| kv4 | centred, per token-half, +-7 (our kv4 basic) |
| h128 | + Hadamard H128 per half (our KROT) |
| h256g32 | centred, H256, asym int4 per 32 dims |
| uq | UltraQuant: H256 + MXFP4 per 32 (power-of-two scale, c = 0.156) |
| wush, wushg32 | WUSH-KV: per-head T = Hd Lam^-1/4 U^T L^T from first-chunk K and the GQA group's Q (gamma = KVQ_WUSH_GAMMA, default 1.0), asym int4 per 128 (or 32) with the range x0.96 |
| bad3 | anchor |

| V | |
|---|---|
| int8v | per channel per 16-token tile, 255 levels |
| kv4v | 15 levels per channel per 16-token tile (our kv4 basic) |
| uqv | MXFP4 per 32, no rotation |

## Running

```
kvq/run_codec.sh <chunked set> <T> <ids> <out.rs> <kcodec> <vcodec> [env...]
kvq/gate2.py ref1.rs,ref2.rs cand1.rs,cand2.rs --name X
kvq/needle_score.py p1.json,... ref1.rs,... cand1.rs,... --name X
kvq/tierA.py <dump dir> kv4:kv4v wush:kv4v ...
```

## First results (2026-10-02, pg1023 + pg145 at 32K, fp16-KV reference)

Tier B:

| K / V | KL mean | flips/1k | dPPL [95% CI] |
|---|---|---|---|
| int8 / int8v (anchor) | ~1e-5 | 0 | -0.00% [-0.02, +0.01] |
| h256g32 / kv4v | 9.9e-4 | 1.9 | -0.04% (n.s.) |
| wush (asym g128, per-prompt calib) / kv4v | 1.2e-3 | 1.8 | +0.18% [+0.04, +0.31] |
| kv4 / kv4v (shipped kv4 basic) | 3.7e-3 | 8.8 | +0.42% [+0.20, +0.64] |
| uq / uqv (UltraQuant MXFP4) | 1.0e-2 | 23.6 | +0.47% [+0.14, +0.78] |
| bad3 / kv4v (anchor) | 2.8e-2 | 61 | +2.2% [+1.6, +2.9] |

KL mean separates the codecs best. The KL estimate can come out slightly
negative at the int8 level (top-64 approximation).

Tier C v1 (32 keys, 16K/32K) is saturated: every 4-bit codec matches fp16
(d log p within +-0.01). Only bad3 registers (-0.15 nats [-0.35, -0.02]). The
first query of each prompt is format noise (log p -3), hence v2: 256 keys,
reassignments, 2 warm-up questions (KVQ_NKEYS/NQ/NUPD/WARM).

Offline layer 3 (8K), K-only attention error:

| K codec | error |
|---|---|
| kv4 basic | 4.4e-2 |
| h128 | 2.1e-2 |
| uq | 2.3e-2 |
| h256g32 | 1.27e-2 |
| WUSH + symmetric +-7 per token-half, range .96 | 1.2e-2 |
| WUSH asym g128 | 1.07e-2 |

- The symmetric WUSH variant (`wushs`) stores K in the int4 K kernel's
  existing format. It needs a 256x256 transform of K at quantize time and of
  roped Q per GQA group: about 1% of attention FLOPs at 32K.
- WUSH notes: quantile clipping is 4x worse than range shrink; gamma 1.0
  beats the paper's 0.01 here.
- KVarN-lite (channel balance folded into Q) helps a little; its LS scale
  refit doesn't.
- Open: does a static (cross-document) WUSH calibration hold? (wush_calib.py +
  KVQ_WUSH_FILE). How much of the remaining error is kv4v V (tierA.py K/V split)?

## Round 2 (2026-10-02)

Numpy note: the system numpy uses reference CBLAS (6 GFLOP/s). Run the
analysis tools with /home/q/yah-scratch/venv/bin/python (pip numpy, OpenBLAS,
686 GFLOP/s). Tier A on 16 layers x 8 codecs then takes 7 min.

Tier B, 32K, pg1023 + pg145:

| K / V | KL mean | KL p99.9 [95% CI] | flips/1k | dPPL |
|---|---|---|---|---|
| wushs / kv4v (per-prompt calib) | 1.63e-3 | 3.9e-2 [3.1, 5.8]e-2 | 3.8 | +0.05% (n.s.) |
| wushs-static / kv4v (calibrated on pg1399) | 1.76e-3 | 4.3e-2 [3.6, 5.9]e-2 | 4.2 | +0.16% [+0.01, +0.32] |
| kv4 basic | 3.7e-3 | 8.0e-2 [6.0, 9.2]e-2 | 8.8 | +0.42% |

So a static calibration holds across books.

Tier A, pg1399 32K, all 16 attention layers. Median attention-output error:

| K / V | total | K only | V only |
|---|---|---|---|
| int8 | 3.7e-3 | 3.4e-3 | 1.1e-3 |
| kv4 | 6.8e-2 | 6.3e-2 | 1.9e-2 |
| h128 | 5.3e-2 | 4.6e-2 | |
| h256g32 | 4.0e-2 | 3.1e-2 | |
| wush (asym) | 4.4e-2 | 3.6e-2 | |
| wushs | 4.9e-2 | 4.2e-2 | |
| uq / uqv | 1.1e-1 | 6.8e-2 | 7.5e-2 |
| bad3 | 1.75e-1 | 1.7e-1 | |

- UltraQuant's per-token MXFP4 V is 4x worse than our per-channel tile V; that
  is where it loses.
- Layer sensitivity: the middle attention layers (L6-L13) are 3-10x more
  sensitive than L0, L14, L15.

Tier C v2 (256 keys, 8 reassigned, 3 x 32K, 72 queries): still saturated for
every 4-bit codec (d log p within +-0.03, CIs span 0). bad3 loses at early
depth (-0.66 nats for depths < 0.33, worst -9.3), exact match 86% vs 93%.

h256s (centred, fixed H256, symmetric +-7 per token-half, range .96: the int4
K kernel's format with no calibration):
- tier A K-only 4.4e-2 (wushs 4.2e-2);
- tier B KL 1.89e-3 (wushs-static 1.76e-3), p99.9 5.4e-2 [4.2, 7.6]e-2,
  dPPL +0.01% (n.s.).

WUSH's calibration adds only ~7% over a fixed Hadamard in this format; the
rotation does nearly all of the work.

## Round 3 (2026-10-02): code + arXiv paper, 32K each

Documents: code = HRX loom/src/loom/ir/module.c (reference PPL 1.296); arxiv
= 2608.13365 pdftotext (reference PPL 3.335). Static WUSH is still calibrated on
Anna Karenina. 99% / 99.9% precision = 100 exp(-KLD at that percentile).

| doc | K / V | dPPL | mean KLD | 99% KLD | 99.9% KLD | 99% prec | 99.9% prec | same top |
|---|---|---|---|---|---|---|---|---|
| code | int8 | -0.00% | 0.000032 | 0.0008 | 0.0038 | 99.92% | 99.62% | 99.92% |
| code | h256g32 | -0.10% | 0.001010 | 0.0161 | 0.0792 | 98.41% | 92.39% | 99.45% |
| code | wushs | -0.08% | 0.001360 | 0.0247 | 0.0804 | 97.56% | 92.27% | 99.22% |
| code | wushs-static | +0.12% | 0.001617 | 0.0276 | 0.0687 | 97.28% | 93.36% | 99.19% |
| code | h256s | +0.08% | 0.001695 | 0.0273 | 0.1568 | 97.31% | 85.48% | 99.29% |
| code | kv4 | +0.22% | 0.002900 | 0.0508 | 0.1573 | 95.05% | 85.45% | 99.34% |
| code | uq | +0.66% | 0.009415 | 0.0932 | 0.7080 | 91.10% | 49.26% | 98.79% |
| code | bad3 | +0.89% | 0.019853 | 0.3420 | 1.1028 | 71.04% | 33.19% | 97.56% |
| arxiv | int8 | -0.01% | 0.000087 | 0.0015 | 0.0128 | 99.85% | 98.72% | 99.87% |
| arxiv | h256g32 | +0.12% | 0.003240 | 0.0313 | 0.1548 | 96.92% | 85.65% | 97.73% |
| arxiv | wushs | +0.02% | 0.004123 | 0.0468 | 0.3018 | 95.42% | 73.95% | 97.73% |
| arxiv | h256s | +0.13% | 0.004642 | 0.0517 | 0.2498 | 94.96% | 77.89% | 97.56% |
| arxiv | wushs-static | +0.11% | 0.005013 | 0.0526 | 0.2269 | 94.88% | 79.70% | 96.98% |
| arxiv | kv4 | +0.76% | 0.008968 | 0.0749 | 0.5570 | 92.79% | 57.29% | 96.27% |
| arxiv | uq | +0.76% | 0.016819 | 0.1478 | 0.6495 | 86.26% | 52.23% | 94.41% |
| arxiv | bad3 | +4.49% | 0.059623 | 0.6448 | 1.7137 | 52.48% | 18.02% | 90.40% |

- The ranking holds across all three domains: h256g32 best; the kernel-format
  rotations (h256s / wushs / wushs-static) within ~0.5 points of each other at
  99%; kv4 basic 1.3-2.3 points behind; UltraQuant and bad3 last.
- The arXiv paper is the hardest text.
- The novel-calibrated WUSH transfers to code and paper.
- 99.9% columns rest on ~4 positions per document: use mean and 99% for
  decisions.

## Round 4 (2026-10-02): what fits the int4 K kernel

Tier A, K-only (V exact), 16 layers, 32K:

| K codec | per-group cost in kv4a4 | K-only error | softmax KL |
|---|---|---|---|
| kv4 | - | 0.0628 | 0.0431 |
| ksink (KVarN Sinkhorn tiles, no rotation) | per-tile column scales (kv4a8 only) | 0.0449 | 0.0220 |
| h256e1/e2 (pow2 exponent per 32) | ~4 VALU/WMMA | 0.0454 | 0.0243 |
| h256s (sym per token-half) | none | 0.0440 | 0.0229 |
| h256a (asym per token-half) | ~1 VALU/WMMA (rank-1 zero-point term) | 0.0388 | 0.0178 |
| h256a96 (+ range .96) | ~1 VALU/WMMA | 0.0377 | 0.0167 |
| h256sink (H256 + Sinkhorn) | kv4a8 only | 0.0384 | 0.0169 |
| h256sg32 (sym per 32) | ~12 VALU/WMMA | 0.0379 | 0.0168 |
| h256ag64 | ~6 VALU/WMMA | 0.0350 | 0.0144 |
| h256g32 (asym per 32) | ~12+ VALU/WMMA; ~free in kv4a8 | 0.0306 | 0.0110 |

- After H256 the 32-dim groups' ranges sit within 2x of each other, so
  power-of-two group exponents are almost always 0.
- KVarN's Sinkhorn is about as good as a rotation alone and redundant with
  one.

End to end, h256a96 vs h256s (mean KLD; 99% precision):
- books: 0.00148 vs 0.00189; 98.52% vs 98.16%
- code: 0.00127 vs 0.00170; 97.98% vs 97.31%
- arxiv: 0.00482 vs 0.00464; 95.18% vs 94.96%
- arxiv without the clip (h256a): mean 0.00432, 99% 94.83%

The arXiv 99.9% KLD (h256a96 0.69, h256a 0.34, h256s 0.25) is decided by 3-4
two-way-fork tokens (reference p 0.73/0.27 etc.) that every codec tips,
including h256g32. Not a codec defect; use mean and 99% for decisions. The clip
is a wash: it helps tier A and the 99% column and hurts the arXiv mean.

Pick for kv4a4: H256 + asymmetric int4 per token-half (zero point as a rank-1
correction). For kv4a8: asymmetric per 32-dim group, applied at staging decode.

## Real kernels (2026-10-02, commit b875efe): kv4a4 v2, kv4a16, kv8a16

Gate: 4 docs (both books, code, arXiv) x first 8192 tokens, one-pass B=8192,
vs fp16 (/home/q/yah-hal-p63-8192). Sets: /home/q/yah-hal-p67{kv4a4,kv4a16,kv8a16}-8192.

| config | dPPL | mean KLD | 99% KLD | 99.9% KLD | 99% prec | 99.9% prec | same top |
|---|---|---|---|---|---|---|---|
| int8 old (iu8, Q int8) | -0.00% | 0.000063 | 0.0016 | 0.0044 | 99.84% | 99.56% | 99.80% |
| kv8a16 | -0.03% | 0.000045 | 0.0011 | 0.0046 | 99.89% | 99.54% | 99.83% |
| kv4a16 | +0.02% | 0.001343 | 0.0152 | 0.0677 | 98.49% | 93.45% | 98.30% |
| kv4a4 new (H256 + asym/half) | +0.15% | 0.003753 | 0.0372 | 0.1300 | 96.35% | 87.81% | 97.54% |
| kv4a4 old (basic) | +0.76% | 0.008525 | 0.0903 | 0.1999 | 91.36% | 81.88% | 95.51% |

pp8192 attention cycles (SQ_BUSY_CYCLES, one round each, APU <= 55 C + 30 s):

| config | attention kernel | + quantizers | attention total | vs fp16 |
|---|---|---|---|---|
| fp16 | 1021.6 M | 7.1 M | 1028.7 M | - |
| kv4a4 old | 878.8 M | 9.4 M | 888.3 M | -13.7% |
| kv4a4 new | 900.1 M | 11.1 M | 911.2 M | -11.4% |
| int8 old | 1005.8 M | 8.9 M | 1014.6 M | -1.4% |
| kv8a16 | 1148.0 M | 8.8 M | 1156.7 M | +12.4% |
| kv4a16 | 1169.2 M | 11.1 M | 1180.3 M | +14.7% |

- kv4a4 v2 costs +2.4% attention over v1: the rank-1 fma plus the zero loads.
- The a16 configs pay for the int-to-f16 staging decode (~32 VALU per thread
  per K tile) in an issue-bound kernel. They are memory/quality configs.
- Attention is ~3.5% of pp8192 cycles (1.03 of 29.0 G), so kv4a4 saves
  ~0.4% of pp8192.
- Prefill totals vary +-200 M between runs (non-attention kernels): compare
  the attention rows.
- Quantized KV is not yet supported in chunked prefill (YAH_CTX > B).
