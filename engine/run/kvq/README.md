# KV codec gate

Measures how much a KV-cache compression scheme hurts the model at long context (32K-64K), end to end, with confidence intervals. The accuracy gate in `../gate` scores short windows and saturates for any non-bit-exact change, so it cannot judge KV compression. Method follows arXiv 2607.09683 ("Ablation, Statistical Inference, and Validation for KV-Cache Compression"). Results for the shipped formats are in [docs/results.md](../../../docs/results.md).

## Engine hooks (`loom_forward_pp`, chunked sets)

- `YAH_KV_HOOK="cmd"`: a persistent child process gets every chunk's fresh f16 K / V rows of every attention layer (and every n-th roped Q row with `YAH_KV_HOOK_QSTRIDE=n`) and returns them. This is fake quantization: any codec in `kvcodec.py` runs quantize -> dequantize, then the fp16 attention. Passthrough is bit-identical to no hook.
- `YAH_ROWSTATS=file`: per scored position, logsumexp, next-token logit, argmax, top-2 and the top-64 (id, logit), 540 B instead of a 1 MB logit row. Positions: every `YAH_ROWSTATS_STRIDE` from `YAH_ROWSTATS_FROM`, or a list in `YAH_ROWSTATS_POS`.
- Real kernels instead of the hook: emit the set with `YAH_KV=kv8|kv4` (see docs/build-and-run.md).

## Tiers

- A, per-layer fidelity (`tierA.py`): dump all 16 attention layers' K / V at 32K plus sampled Q (`kvcodec.py --dump`), replay codecs offline chunk by chunk, report per-layer attention-output error and softmax KL by query position, K-only and V-only. No compounding through 64 layers, so this separates schemes best.
- B, long-document NLL (`gate2.py`): 32K windows of public-domain books, code and a paper; every 8th position from 1024. Per position dNLL, KL over the reference top-64 and top-1 flips, binned by position, 95% block-bootstrap CI.
- C, multi-key retrieval (`gen_needle.py`, `needle_score.py`): 32 near-duplicate keys with 7-digit codes at depths 5-95%, 8 questions at the end, scored teacher-forced (log p(answer), exact match, minimum answer-token margin). Accuracy alone saturates for a 27B model; margins do not.

The reference is fp16 KV through the same engine, so only the codec differs. Anchors set the thresholds: int8 (`--k int8 --v int8v`) must pass, bad3 (3-bit K per token) must fail.

## Codecs (`kvcodec.py`)

| K | |
|---|---|
| `int8` | centred, per token-half, 127 levels (the kv8a16 K) |
| `kv4` | centred, per token-half, +-7 |
| `h128` | + Hadamard H128 per half |
| `h256g32` | centred, H256, asymmetric int4 per 32 dims (the kv4a16 K) |
| `uq` | H256 + MXFP4 per 32 (power-of-two scale) |
| `wush`, `wushg32` | per-head transform from first-chunk K and the GQA group's Q (`KVQ_WUSH_GAMMA`, `KVQ_WUSH_FILE`), asymmetric int4 per 128 (or 32), range x0.96 |
| `bad3` | anchor |

| V | |
|---|---|
| `int8v` | per channel per 16-token tile, 255 levels (the kv8a16 V) |
| `kv4v` | 15 levels per channel per 16-token tile (the kv4a16 V) |
| `uqv` | MXFP4 per 32, no rotation |

## Run

```
kvq/run_codec.sh <chunked set> <T> <ids> <out.rs> <kcodec> <vcodec> [env...]   # fp16 fp16 = reference
kvq/gate2.py ref1.rs,ref2.rs cand1.rs,cand2.rs --name X
kvq/needle_score.py p1.json,... ref1.rs,... cand1.rs,... --name X
kvq/tierA.py <dump dir> kv4:kv4v wush:kv4v ...
```
