# Measurements

Everything below was taken on one Strix Halo box (Ryzen AI MAX+ 395, 130 W) with Qwen3.8 27B IQ4_XS, pp2048, single stream, using `bench` from gufo. Protocol is the controlled one described in [methodology.md](methodology.md): both arms warmed up, arm order alternated, one repetition per run.

## The token split vs the column split

The NPU can divide the FFN two ways. The **column split** gives it output columns `[n_gu, n_full)` for every token. The **token split** gives it tokens `[0, M)` at full width and the GPU the remaining tokens, also at full width, so neither engine needs to gather or scatter partial rows.

Arms: column split with `GUFO_ATB_GU_NSLICE=8192`; token split with `GUFO_ATB_BATCH=1024` and full-width gate/up and down xclbins. Six alternating-order reps after a discarded warm-up:

| rep | column split (n_gu=8192) | token split (M=1024) | delta |
| ---: | ---: | ---: | ---: |
| 1 | 506.73 | 542.61 | +7.1% |
| 2 | 474.63 | 563.20 | +18.7% |
| 3 | 479.82 | 553.44 | +15.3% |
| 4 | 472.90 | 536.93 | +13.5% |
| 5 | 470.91 | 535.24 | +13.7% |
| 6 | 487.36 | 530.77 | +8.9% |
| **mean** | **482.06** | **543.70** | **+12.8%** |

The win is GPU work removed, not NPU work added. Both arms under a kernel trace:

| | column split | token split |
| --- | ---: | ---: |
| dispatches | 3611 | 3849 |
| GPU stream busy | 7703 ms | 6449 ms |
| GPU stream idle | 6.3% | 20.9% |
| repack kernels | 168 ms (x256) | 526 ms (x384) |

The token split removes ~1254 ms of GPU kernel time over the chunk and adds ~357 ms of full-width repack, so GPU work falls ~900 ms net, which is the whole end-to-end gain (the profiled span is roughly twice the real chunk). The GPU is also four times idler: the NPU now covers half the FFN, so the critical path has moved onto it. The column split's extra cost is the gather/scatter a column division needs: partial-width heads land packed and are expanded, and the down partial is packed before it reaches the hidden state.

## NPU token count (M) sweep

The token split has one shape lever, how many tokens the NPU takes, because its B operand is the full intermediate size for any M. Full-width xclbins were built at M = 512, 1024, 1536 and 2048; every arm ran the same split with only `GUFO_ATB_BATCH` changed. Four rotating-order reps per arm:

| M (NPU tokens) | GPU tokens | mean pp2048 | vs M=1024 |
| ---: | ---: | ---: | ---: |
| 512 | 1536 | 476.74 | -13.5% |
| **1024** | **1024** | **551.45** | -- |
| 1536 | 512 | 482.09 | -12.6% |
| 2048 | 0 | 422.43 | -23.4% |

M=1536 and M=2048 are stable to within ~1 tok/s across reps, which is what a purely NPU-bound arm looks like; M=512 is the noisiest (443–547), which is what a GPU-bound arm looks like. M=1024 sits between them and is 12–13% above both neighbours. Offloading is only worth it while the GPU still has FFN work to overlap.

## Numerics

`--validate-prefill N` compares batched-prefill logits against a sequential reference and requires a top-1 match. Results, all with the token split:

| N | NPU tokens M | GPU remainder | top-1 match | cosine |
| ---: | ---: | ---: | :---: | ---: |
| 512 | 512 | 0 | yes | 0.999993 |
| 1025 | 1024 | 1 | yes | 0.987133 |
| 1300 | 1024 | 276 | yes | 0.999228 |
| 1536 | 1024 | 512 | yes | 0.998556 |
| 2048 | 1024 | 1024 | yes | 0.985100 |
| 1024 | 1024 | 0 | **no** | 0.953139 |
| 2048 | 2048 | 0 | **no** | 0.953542 |

Every shape that leaves at least one token on the GPU passes. The two that give the NPU the whole batch do not, and one of them uses a shape (M=512) that passes exactly when it owns the whole batch. That inconsistency is the open item in [open-questions.md](open-questions.md); until it is closed, the 100%-NPU mode is disqualified on correctness, not just speed.

## NPU standalone ceiling

A standalone driver against the same xclbins, no engine in the loop:

| shape | best | TFLOPS |
| --- | ---: | ---: |
| M=1024 K=5120 N=17408 (gate/up, full width) | 5.635 ms | 32.39 |
| M=1024 K=17408 N=5120 (down, full width) | 5.570 ms | 32.77 |
| M=2048 K=5120 N=8192 (reference) | 5.310 ms | 32.35 |

In the engine the same gate/up pair costs ~14.5 ms per layer rather than the ~11.3 ms the standalone floor implies, i.e. roughly 25 TFLOPS and ~14 GB/s of operand traffic. The standalone floor is not reached once the operands are shared with the GPU through dma-buf. This gap is not explained and is a target for the custom engine.

## Decode attention at depth: the scalar walk dominated until it was split

The engine's `Decode()` called `LaunchAttention` with `split_k_scratch =
nullptr`, so every decode step ran `QwenDecodeOnlineAttentionHalfKernel`: one
32-thread warp per query head walking the visible cache one key per two
iterations, with each of the six query heads in a GQA group re-reading the same
K/V rows. The split-K kernel that fixes both — 32 partitions, each KV tile
staged once for its six heads — was already in the tree and is what the
reference engine's own decode selects; only the caller's scratch was missing.
It costs 0.79 MB.

Measured with `yah-run` (32768-token prompt, `--chunk 2048`,
`--max-context 32900`, `--gen 32`, f16 KV), arms alternated 0/1/1/0:

| context | online (tok/s) | split-K (tok/s) | delta |
| ---: | ---: | ---: | ---: |
| 2048 | 11.47 | 12.82 | +12% |
| 32768 | 3.61 / 3.58 | 10.88 / 11.04 | **3.05x** |

Generated ids are identical across the toggle (32/32) at 32k, and
`GUFO_DISPATCH_TELEMETRY=1` shows the selected backend switching between
`decode_online_fp16` and `decode_split_k_fp16`.

**A run is only real if the binary changed.** The first two A/Bs of this change
reported *no effect*, twice: `build_gpu.sh` compared each object only against
its own `.hip`, so editing an included header (`model/forward.hip`) silently
reused a stale object and both arms executed the previous binary. The cache
check now tracks real prerequisites with `-MMD` depfiles, and the dispatch
telemetry line is the evidence that a chosen kernel actually ran.

## Provenance

- Hardware: `RYZEN AI MAX+ 395`, gfx1151, XDNA2 NPU (`aie2p`, PCI `1022:17f0`).
- Model: Qwen3.8 27B IQ4_XS 3.84 bpw, 12.18 GiB GGUF.
- Harnesses: `token_split_ab.sh`, `token_msweep.sh`, `split_timeline.sh`, `token_split_smoke.sh`, `npu_ceiling.sh` (the last is written but has not been run).
- The engine work, including the full experiment log, is in [gufo](https://github.com/gufo-org/gufo).
