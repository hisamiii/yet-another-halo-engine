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
