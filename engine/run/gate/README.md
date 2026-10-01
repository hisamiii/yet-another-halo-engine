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
