# KV cache gate

Measures how much a KV-cache format hurts the model at long context (32K-64K), end to end, with confidence intervals. The accuracy gate in `../gate` scores short windows and saturates for any non-bit-exact change, so it cannot judge KV compression. Method follows arXiv 2607.09683 ("Ablation, Statistical Inference, and Validation for KV-Cache Compression"). Results for the shipped formats are in [docs/results.md](../../../docs/results.md).

The reference is an fp16-KV chunked prefill set; the candidate is the same set emitted with another `YAH_KV` (kv8, kv4, ...), so only the KV format differs.

- `YAH_ROWSTATS=file` (in `loom_forward_pp`): per scored position, logsumexp, next-token logit, argmax, top-2 and the top-64 (id, logit), 540 B instead of a 1 MB logit row. Positions: every `YAH_ROWSTATS_STRIDE` from `YAH_ROWSTATS_FROM`, or a list in `YAH_ROWSTATS_POS`.
- Long-document NLL (`gate2.py`): 32K windows of public-domain books, code and a paper; every 8th position from 1024. Per position dNLL, KL over the reference top-64 and top-1 flips, binned by position, 95% block-bootstrap CI.
- Multi-key retrieval (`gen_needle.py`, `needle_score.py`): 32 near-duplicate keys with 7-digit codes at depths 5-95%, 8 questions at the end, scored teacher-forced (log p(answer), exact match, minimum answer-token margin). Accuracy alone saturates for a 27B model; margins do not.

Run:

```
kvq/run_rowstats.sh <chunked set> <T> <ids> <out.rs>         # reference and candidate sets
kvq/gate2.py ref1.rs,ref2.rs cand1.rs,cand2.rs --name X
kvq/needle_score.py p1.json,... ref1.rs,... cand1.rs,... --name X
```
