# Strix Halo inference engine — design notes

Design notes and measured evidence for a **narrow-scope GGUF inference engine** on AMD Strix Halo: text and image parsing, three routing modes (GPU, NPU, GPU+NPU), and minimal resident memory.

This is a design and measurement log, not an engine. The numbers come from work on [gufo](https://github.com/gufo-org/gufo), a general GGUF engine, where the NPU is attached behind environment-variable-gated xclbin paths. The point of this repo is to record what that work measured and to argue that the *narrow* scope is what makes a purpose-built engine worth writing.

## Hardware and model

- AMD Ryzen AI MAX+ 395 (Strix Halo): gfx1151 iGPU (RDNA 3.5), XDNA2 NPU (`aie2p`), one shared 28 GB memory pool, 130 W configured package power.
- Qwen3.8 27B, IQ4_XS, 3.84 bits per weight, 12.18 GiB GGUF.
- Workload: single-stream prompt processing, pp2048.

## Why the scope has to be narrow

An AIE/NPU kernel bakes M, K and N into the xclbin, and a process holds one shape per hardware context. That is workable only when the shape set is small. Text plus image gives a handful of prefill shapes; a general framework has to treat every model as a new shape. The narrow scope is not a concession, it is what makes the NPU path affordable to write at all.

## The three routing decisions

### Prefill: GPU + NPU, roughly 50/50

Measured. The NPU takes a prefix of the tokens at full width and the GPU takes the rest, so both engines run full-width GEMMs. Over six paired reps this was +12.8% against the previous column-split placement. The balance point is near 50/50: 75% of the work on the NPU loses 12.6%, 100% loses 23.4%. The NPU is a prefill co-processor, not a replacement.

### Decode: GPU

Inferred, not yet measured. The NPU reads bf16 weights, about twice the model's 3.84 bpw, which is the wrong trade at batch 1. Route by phase and gate the NPU path on a minimum batch.

### Image parsing: its own baked shapes

A vision tower is prefill-shaped (dense GEMMs, different K and N), so the same GPU+NPU structure should apply with its own xclbin pair. Unmeasured on this hardware; possibly the largest of the three.

## What was measured, in one table

| quantity | value |
| --- | --- |
| token split, pp2048, 6-rep mean | **551.45 tok/s** |
| column split (n_gu = 8192), same protocol | 482.06 tok/s |
| token split vs column split | **+12.8%**, 6/6 reps |
| NPU token count sweep | 512: 476.7, **1024: 551.5**, 1536: 482.1, 2048: 422.4 |
| GPU stream busy, token split | 6449 ms (20.9% idle) |
| GPU stream busy, column split | 7703 ms (6.3% idle) |
| bf16 weight repack per chunk | 526 ms (token), 168 ms (column) |
| NPU standalone microbench | 32.4 TFLOPS bf16, M=1024 full width |
| NPU operand traffic in engine | ~14–22 GB/s implied |

Full tables, provenance and the validation results are in [docs/measurements.md](docs/measurements.md). The ceiling on a perfect implementation, and why it is about 1.6x over the GPU-only engine rather than 2x, is in [docs/ceiling.md](docs/ceiling.md).

## Levers a purpose-built engine should take

1. **Overlap the NPU.** Today gate/up then down is strictly serial and the host blocks on each NPU wait, which is why the GPU idles 20.9% of the chunk. Double-buffered operands, async submission and a deferred wait are bounded by the GPU's own busy time: ~3.1 s of a 3.72 s chunk, or about +20%.
2. **Make B the model's own quant.** The NPU now reads a bf16 repack of IQ4_XS: 9 bits per element against 3.84, rebuilt on the GPU every layer. An AIE GEMM with an int4/int8 B and per-block scales deletes the repack and roughly halves the NPU's operand bytes. This is the highest-leverage kernel and the reason the mlir-aie `gemm_asymmetric_tile_buffering` example is the right starting point.
3. **Pipeline across requests.** Fixed NPU shapes plus a flexible GPU means request A can run on the NPU while request B runs on the GPU. This is the structural win a single-stream graph cannot express, and it is what returns the GPU to full occupancy without needing the NPU to be faster.
4. **Keep memory minimal.** Resident memory is dominated by the 12.18 GiB model and the KV cache, not by NPU operands. The ATB operands are transient (~250 MB at M=1024). Minimal memory means mmap the GGUF as the single source of truth, quantize KV, share one pool across both engines, and avoid ever materialising a second copy of the weights.

See [docs/design.md](docs/design.md) for the proposed shape of the engine, [docs/gpu-tuning.md](docs/gpu-tuning.md) for what the narrow scope makes possible on the GPU side, and [docs/kernel-tuning.md](docs/kernel-tuning.md) for the measured GPU and NPU rooflines (48.35/50.31 TFLOPS GPU, 56.9 TOPS int8 / 38.9 TOPS bfp16 NPU, and where the production kernels sit against them).

## Status and open items

- **The all-NPU placement fails numerics at some shapes.** M=1024 with N=1024 and M=2048 with N=2048 both give cosine 0.953 against a sequential reference and flip the top-1 token, while M=512/N=512 is near-exact at 0.99999 and M=1024/N=1025 passes at 0.987. This is unresolved and currently disqualifies the 100%-NPU mode. See [docs/open-questions.md](docs/open-questions.md).
- **The GPU-only baseline is missing.** Every NPU number here is relative to other NPU configurations. The engine has a harness for the GPU-only arm but it has not been run, so the total value of the NPU is not yet bounded.
- **Batch-1 NPU behaviour is unmeasured**, as is the whole vision tower.
- **The AIE int4/int8 B path is unproven.** The compiler example is about block datatypes, but a per-block-scale int4 B GEMM matching GGUF has not been built.

## Measurement discipline

Cross-session drift at fixed settings is 15–35%, and the first run of a session is boosted by about 20%, so only adjacent paired deltas are usable. Warm up both arms, alternate the arm order, use one repetition per run, and never compare a number from another session. Details in [docs/methodology.md](docs/methodology.md).

## Background

The engine these notes come from is [gufo](https://github.com/gufo-org/gufo). The longer experiment log is `docs/models/qwen3.8-27b/EXPERIMENTS.md` in that repository.
