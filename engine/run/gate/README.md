# Tiered correctness gate (T1 rounding-level, T2 quantization-level)

Design: ../RESEARCH_PREFILL.md section 3 and ../research/correctness.md.

The corpus is 16 disjoint 2048-token windows plus one 8192-token window of
wikitext-2. They come from the hipfire frozen slice, tokenized with llama.cpp
`llama-tokenize --ids --no-bos`; see `corpus/manifest.json` for the md5s.

Steps:

1. Golden: frozen Loom output of the reference set, positions 1536..2047.

       gate_run.sh <set> <golden_dir>

   Run with WINDOWS="00 01 02 03" for T1, or 00..15 for T2.
2. Thresholds, calibrated from HIP vs golden on the same windows.

       gate_hip.sh <hip_dir>
       accgate2.py calibrate <golden_dir> <hip_dir> thresholds_T1.json

   `thresholds_T1.json` was calibrated 2026-10-01 against golden p48 (the
   reference distance is recorded inside).
3. Candidate:

       gate_run.sh <candidate set> <dir>
       accgate2.py check <golden_dir> <dir> thresholds_T1.json

Controls, 2026-10-01:

- Self-check passes.
- YAH_TG_IQ4PK fails: KL mean 7.0e-7 against a limit of 1.3e-7.
- HIP vs golden: KL mean 5.3e-7, p99.9 1.8e-5, 0 top-1 flips; PPL 6.6947
  vs 6.6949.

Decision, 2026-10-01: FA attention accepted as the default.

- `gen_attn_fa` (p63) fails only kl_mean: 3.64e-7 against a limit of
  1.32e-7.
- It passes kl_p999 (7.5e-6), flips (0) and PPL (6.6946 vs 6.6947).
- kl_mean saturates for any attention change that is not bit-exact. The
  control is the HIP-order kernel with a single rounding change, o * (1/sum)
  instead of o / sum: kl_mean 4.24e-7, 0 flips. FA sits inside that floor
  (13x closer to HIP's attention output than the control is, and kl_mean
  barely moves).
- Treat kl_mean ~4e-7 as the "one rounding change in attention" floor. New
  exact-change references are p63, hidden md5 f0dbf625bfe0f892 (pp2048) and
  4517ea645cd7dcfe (pp8192), replacing a2145e371ceefd4d / e94924b79ae21e57.

Decision, 2026-10-02: chunked WY Gated DeltaNet (tools/gen_gdn_chunk.py) is the
default (emitter YAH_DN_CHUNK=1; 0 restores the HIP-order recurrent kernel).
- New exact-change references are p70: hidden md5 ac36332b6b5092a4 (pp2048)
  and 963b7396625e2333 (pp8192).
- p70 with YAH_DN_CHUNK=0 reproduces p63's f0dbf625bfe0f892 / 4517ea645cd7dcfe
  exactly: the only difference is DeltaNet.
- Gate v2 on the hardest document (arXiv 2608.13365), vs the p63 fp16 references:
  - 8K: mean KLD 0.000008, 99% precision 99.98%, same top 99.89%;
  - 32K chunked: mean KLD 0.000004, 99.98%, same top 99.95%.
- Speed (SQ_BUSY_CYCLES, one round each): DeltaNet -10.5% at pp2048
  (188.0 -> 168.4 M), -12.9% at pp8192 (749.4 -> 652.5 M), i.e. 0.27% / 0.33%
  of total cycles.
