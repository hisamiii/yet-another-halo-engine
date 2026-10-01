# Tiered correctness for numerics-changing Loom work (2026-10-01)

Labels: **[F]** = a fact that was checked (by reading source or by a CPU-only measurement on existing dumps). **[R]** = a recommendation. No GPU work was run for this note.

## 1. What the community uses

- **[F] `llama-perplexity` KLD mode** (local source `/home/q/llama.cpp/tools/perplexity/perplexity.cpp` @5816d3bee; [upstream](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/perplexity.cpp), [README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)).
  - **Chunking.** The default is `-c 512` with chunks that do not overlap. The first token of each chunk is replaced by BOS. Only the second half of each chunk is scored (`first = n_ctx/2`, which gives `n_ctx-1-first` positions). `--chunks N` limits the number of chunks.
  - **Base logits file.** `--kl-divergence-base F` stores each scored position as uint16 log-probs over a 16-nat window. That is a step of 16/65535 ≈ 2.4e-4 nat, plus 4 uint16 for the scale and minimum. Base tokens with log p < −16 are dropped from the KL sum.
  - **Reported statistics.** PPL(Q) and PPL(base), mean ln(PPL(Q)/PPL(base)), mean KLD ± SE with max and percentiles, Δp (change in the correct token's probability) percentiles and RMS, and "Same top p" (top-1 agreement).
- **[F] The KLD floor of that format.**
  - The README's "f16" row measures only the uint16 downcast, and it already reads KLD 5.5e-4 on LLaMA-3-8B.
  - For scale: BF16 vs FP16 is 2.5e-5, q8_0 weights 1.36e-3, q6_K 5.5e-3, iq4_XS 3.6e-2.
  - So the llama.cpp file format **cannot resolve rounding-level changes**. Our accgate KLs are around 1e-9.
- **[F] Dataset.** The convention is wikitext-2-raw `wiki.test.raw` ([get-wikitext-2.sh](https://github.com/ggml-org/llama.cpp/blob/master/scripts/get-wikitext-2.sh), about a 4 MB zip). A Qwen3 tokenizer gives 584 chunks at ctx 512, about 149k scored tokens ([#19521](https://github.com/ggml-org/llama.cpp/pull/19521)).
  - The base file costs (n_vocab+4)·2 B per scored token, which is **≈497 KB at vocab 248320**. The full test set would therefore be about 70 GB, so `--chunks` is needed.
- **[F] How llama.cpp judges MMQ and numerics changes.**
  - [#7921](https://github.com/ggml-org/llama.cpp/pull/7921): MMQ (q8_1 int8 activations, 32-element blocks) vs FP16 cuBLAS, KLD vs FP16 on LLaMA-3 q2_K: 0.332531 vs 0.332680. That is a difference of about 1.5e-4, inside the error bars.
    - A q8_1 variant that packed the half-block sums added +0.015 KLD and was reverted. Gaessler: "My first choice for testing correctness is `llama-perplexity`".
  - [#4801](https://github.com/ggml-org/llama.cpp/pull/4801): one int8 scale **per row** gave LLaMA-2 q8_0 a KLD of 2.6e-3, vs 3.8e-4 for cuBLAS. 32-element blocks stayed near cuBLAS.
  - [#4755](https://github.com/ggml-org/llama.cpp/issues/4755): q8 activation quantization of a spiky hidden state produced a nonsense token.
  - Newer reports quote mean KLD, same-top-p, and the rate of flipped top tokens ([#25593](https://github.com/ggml-org/llama.cpp/issues/25593), [#23572](https://github.com/ggml-org/llama.cpp/pull/23572)).
- **[F] `test-backend-ops`** ([source](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-backend-ops.cpp), local lines 1165 and 4849).
  - NMSE = Σ(a−b)²/Σa². The default threshold is 1e-7, and **5e-4 for mul_mat**. slaren in #7921 called 5e-4 "probably already too high".
  - The comparison is against the CPU backend, which itself quantizes activations: IQ4_XS's `vec_dot_type` is Q8_K (local `ggml-cpu.c:389`).
- **[F] llama.cpp on gfx1151 is already an int8-activation engine.** `ggml_cuda_should_use_mmq` (local `ggml-cuda/mmq.cu:266`) uses MMQ on RDNA3/3.5 for every type at any batch, with two exceptions that fall back to dequantize + hipBLAS FP16: Q6_K above 256 tokens and Q2_K above 128.
  - `GGML_CUDA_FORCE_CUBLAS` forces the FP16 path. The docs warn this "may [have] numerical overflows".
- **[F] lm-eval-harness** ([README](https://github.com/EleutherAI/lm-evaluation-harness)): `--model gguf` against llama-server; one request per continuation token, so slow. `llama-perplexity --hellaswag` is the cheap equivalent. Pre-release only.
- **[F] Outliers:** [LLM.int8](https://arxiv.org/abs/2208.07339), [SmoothQuant](https://arxiv.org/abs/2211.10438), [Massive Activations](https://arxiv.org/abs/2402.17762). Qwen FP16 overflow of the FFN-down input: [#27016](https://github.com/ggml-org/llama.cpp/pull/27016).
- **[F] KLD and flips** ([2407.09141](https://arxiv.org/abs/2407.09141), [Unsloth](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs.md)): PPL lets errors cancel, so report the KLD tail and top-1 agreement.

## 2. What exists locally

- **[F] llama.cpp builds.** `/home/q/llama.cpp/build` (CPU only), `build-hip`, `build-rocm` and `build-vulkan` all contain `llama-perplexity`, `llama-tokenize` and `test-backend-ops`.
  - The commit supports `qwen35`, which is this GGUF's `general.architecture` (65 blocks, one of them the nextn layer).
- **[F] Data.** There is no `wiki.test.raw`.
  - There is a frozen wikitext-2 **train** slice (10.5 MB, md5 83b0205a…) at `/home/q/temporary-build-storage/hipfire-src/benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt`.
- **[F] Python.** numpy 2.5.2 on **netlib reference BLAS** (single-threaded): a CPU head GEMM is impractical, but elementwise softmax/KL is fine. No torch, scipy or lm_eval; `gguf-py` is at `/home/q/llama.cpp/gguf-py`.
- **[F] Engine outputs.**
  - `loom_forward_pp` writes `.logits` for the **last token only** (Q6_K GEMV on row B−1) and `.hidden` (B×5120 f32, the final residual before the output norm).
  - `YAH_DUMP_LAYER=l` dumps one layer's stages: `.xn` (f16), `.aq`, `.qkv`, `.conv`, `.aout`, `.ffn`, and so on.
  - HIP `yah-run` already has `--dump-all-logits` (f32 logits for every position of the last chunk, via one GEMV per position), `--text` (built-in tokenizer) and `--chunk`.
  - Loom ports of the HIP Q8_1 quantizers exist (`yah_quant_act_*`), dead on the route.
- **[F] The current gate prompt is degenerate.** `/home/q/yah-scratch/ids{512,2048,8192}.txt` contain **10 distinct token ids** ("The capital of France is Paris…" repeated).
  - Distributions on such text are near one-hot, so KL is tiny and insensitive.
  - The golden is `/home/q/yah-scratch/golden/loom-p42.*`; HIP vs golden KL is 3.4e-9.

## 3. CPU-only measurements on existing dumps

- **[F] Final residual** (golden, 2048 tokens): max |h| = 429, in channel 3994. Median row RMS is 7.3. It fits in f16 storage with about 150× headroom.
  - However, Σx² per row is about 5120·53 ≈ 2.7e5, which is above 65504. **The RMSNorm sum of squares must stay f32.**
- **[F] Normed GEMM inputs.**
  - Layer 4 `.xn`: RMS 0.36, with outlier channels 3994 (21.0, 58× RMS) and 310 (10.6).
  - The 8k-token run's layer 3 `.xn`: channel 3994 reaches 91.6.
- **[F] Simulated q8_1-style quantization** (per-32 block, d = max/127, round to nearest), activation relative RMS error:
  - `.xn`: 1.0e-2. `.fn` (FFN input): 5.9e-3. Worst row: 1.35e-2.
  - In the outlier block (channels 3968–3999), the other 31 channels have **15.7% error and 35% of them round to 0**.
  - Int4 (d = max/7): 4–11% relative error.
  - For scale, f16 storage rounding is about 3e-4 relative.
- So int8 activations add roughly 20–40× the error of f16 rounding, but stay about 10× below 4-bit weight-quantization noise. Int4 activations are comparable to the weight noise.

## 4. Tiers

**T0 exact** (unchanged): the hidden md5, for changes that preserve the arithmetic.

**T-k kernel unit test** (every numerics-changing kernel) [R].
- Compare the kernel against a numpy f64 reference on **real captured activations** (`tools/l4.fn`, `attn8k/l3.*`), plus adversarial rows: one outlier channel, an all-zero block (d = 0), values near f16 max, denormals.
- Metric: NMSE. Gate at ≤ 1.5× the NMSE of the f32-activation kernel, plus the expected quantization NMSE, which is predictable: about (1e-2)² = 1e-4 for int8.
- Also check NaN/Inf count = 0 and max |out|.

**T1 rounding level** (f16 intermediates, fma/single-rounding, reordered sums) [R].
- **Reference:** the frozen Loom golden on real text (never the previous candidate).
- **Data:** 4 windows × 2048 tokens of wiki.test.raw, scoring positions 1024–2047. That is 4096 tokens, every one with at least 1k of context, which exercises the DeltaNet state and the KV.
- **Metrics:** full-f32 KL(golden‖cand) per position, reported as mean, p99, p99.9 and max. Logits relative RMS. Top-1 agreement on non-tie positions. Per-layer hidden relative RMS (see §5).
- **Thresholds**, calibrated **on the same corpus** from HIP vs golden (D): mean KL ≤ 0.25·D_mean (the current accgate factor); p99.9 KL ≤ D_p99.9; top-1 flips ≤ D_flips;
  - per-layer relative RMS ≤ 0.5× HIP's curve at every layer (HIP's `YAH_DUMP_DIR` already writes the f32 residual after every layer)
  - non-finite = 0.
- The packed-f16 IQ4 decode (hidden relative RMS 1.7e-2) would fail T1 clearly. Its correct class is T2.

**T2 quantization level** (int8/int4 activations, f16 accumulation, lower-precision decode) [R].
- **References:**
  - The frozen Loom golden (f32 activations), which measures the added error.
  - llama.cpp `build-rocm` on the same GGUF and the same windows. This is the "accepted int8 cost", because its MMQ already uses q8_1. A `GGML_CUDA_FORCE_CUBLAS` build of it gives the dequant-FP16 counterpart.
  - Score both with our own f32 logits, not the uint16 file, so the 5.5e-4 floor does not apply. That needs a ~50-line libllama program (`llama_get_logits_ith`) to write f32 logits per window.
- **Data:** 16 windows × 2048 (16k scored tokens) plus one 8192-token window (recurrent-state drift).
- **Metrics** (llama.cpp names): mean KLD ± bootstrap CI over windows, KLD p99/p99.9, same-top-p, mean ln(PPL ratio) ± CI, Δp percentiles (asymmetry means the model got worse, not just noisier).
- **Provisional thresholds** (recalibrate after the first measurement): mean KLD ≤ 1e-3 and ≤ llama.cpp-MMQ-vs-golden; p99.9 KLD ≤ 0.05; same-top-p ≥ 99%; |mean ln PPL ratio| ≤ 2e-3.
  - Justification: q8_0 *weights* cost 1.4e-3 on LLaMA-3. Per-row int8 (2.6e-3) was rejected upstream, while 32-block q8_1 was accepted at about 1.5e-4. An activation-noise budget of ≤5% of IQ4_XS-class weight KLD (3–4e-2) gives about 1–2e-3.
- Expect int4 to fail without outlier handling. **Do not loosen thresholds for it.** Instead consider keeping the outlier blocks (3994, 310) in int8/f16, or SmoothQuant/rotation.

**T3 release quality** (before shipping only) [R]: wiki.test.raw PPL at ctx 2048 vs llama.cpp's own run; optionally `--hellaswag` 400 tasks.

## 5. Engine changes needed [R]

1. **Corpus.** Tokenize wiki.test.raw once. Use `llama-tokenize --ids` from the CPU build (`vocab_only`, so no weights are loaded) or `yah-run --text`. Freeze the windows as `ids2048_wN.txt` with an md5. Tokenizer parity does not matter, because every engine reads the same ids.
2. **All-position logits in `loom_forward_pp`.** For example `YAH_LOGITS_FROM=1024` loops the existing Q6_K GEMV over rows [1024, B) and writes n×248320 f32 (1 GB per window).
   - Later, a Loom Q6_K head GEMM, or an on-device log-softmax/KL reduction that writes only per-position (KL, NLL, argmax).
   - The golden for T1 stays f32: 4 GB for 4 windows. **Do not use f16 or uint16 for T1 references**; their floor exceeds T1 signals.
3. **Per-layer residual dump.** For example `YAH_DUMP_RESID=stride` writes the residual after every layer for sampled rows: 128 rows × 64 × 5120 × 4 = 168 MB.
   - The per-layer relative-RMS curve vs golden localizes the first layer whose error jumps. Layer type (full attention every 4th layer, DeltaNet otherwise) and the `YAH_DUMP_LAYER` stage dumps then narrow it to the kernel.
   - Loom has no all-layer residual dump yet. HIP has one: `YAH_DUMP_DIR`, f32 `<tag>_<layer>.bin`.
  - The single-layer probes (`loom_*_layer_probe`), fed golden inputs, separate **local** error from **propagated** error.
4. **`accgate.py` v2.** Multi-window input, per-position metrics, bootstrap CI, a tie-aware flip count, and llama.cpp-style output. Calibration files are keyed by corpus md5.
5. **Range guards.** Count non-finite values and track max |x| per stage. Warn when within 4× of the f16 max.

## 6. Runtime budget [R]

- **T1:** 4 × pp2048 (~3.6 s each) + model load + 4096 head GEMVs (~5 ms each ≈ 20 s, at about 1 GB of Q6_K read each) + numpy KL over 1e9 f64 elements (~15 s) ≈ **1–1.5 min**.
- **T2:** 16 windows + 1 at 8k ≈ **4–5 min**. With a head GEMM and on-device KL, about 1.5 min.
- **One-time references:** HIP `--dump-all-logits` per window (about 10 s per window). llama.cpp ROCm f32 logits via the libllama dumper. A `llama-perplexity -c 2048 --chunks 16` base file is about 8 GB.
- **T3:** about 146 chunks at 2048 → ~10 min prefill + head.
- Per the GPU timing rule: run checks only when no timing session is active. CPU numpy load also shares the APU's memory bandwidth.

## 7. Pitfalls

- One last-token KL on repetitive text is near zero by construction and tells you almost nothing.
- Mean KLD is tail-driven, so always report p99.9 and max. The SE scales as 1/√N; for example q8_0's ±0.44% at about 150k tokens implies about ±3% at 4k.
- Argmax ties: only count a flip when the golden top-1/top-2 logit gap exceeds HIP's max |Δlogit| (0.066).
- llama.cpp CPU and MMQ are not float references, since they quantize activations. A dequant-f32 Loom golden is the cleaner baseline.
- Outlier channel 3994 ruins its own 32-block. Per-token or per-row scales would spread that damage to all 5120 channels (#4801).
- f16 sums of squares and FFN-down inputs can overflow, as seen in llama.cpp #27016.
- Error compounds through 64 layers and through the DeltaNet recurrent state. Test the 8k window, not just pp2048.
