# yet another halo engine

LLM inference for one model on one machine: Qwen3.8-27B (IQ4_XS GGUF) on AMD Strix Halo (Ryzen AI MAX+ 395, gfx1151 iGPU). Every GPU kernel is written in Loom, compiled for the model's exact shapes, and dispatched through HRX, AMD's native GPU runtime. Python generators write the kernels; two small C++ drivers run prefill and decode.

Why: a fixed model on a fixed chip lets every kernel be specialized and every number be measured. The engine beats its own hand-written HIP predecessor on both prefill and decode.

## Status

Measured 2026-10-02, one round each from a cool APU (details and method in [docs/results.md](docs/results.md)).

| workload | Loom | HIP (tag `hip-final`) |
|---|---:|---:|
| prefill, 2048-token prompt | 3132 ms (654 tok/s) | 3382 ms |
| prefill, 8192-token prompt | 13154 ms (623 tok/s) | 14254 ms |
| decode after 2K context | 62.6 ms/token (15.97 tok/s) | 71.1 ms (14.07 tok/s) |
| decode after 8K, fp16 KV | 65.17 ms (15.35 tok/s) | 74.3 ms (13.46 tok/s) |
| decode after 8K, kv8a16 KV | 63.90 ms (15.65 tok/s) | - |
| decode after 30K, fp16 KV | 74.10 ms (13.50 tok/s) | 85.1 ms (11.75 tok/s) |
| decode after 30K, kv8a16 KV | 70.56 ms (14.17 tok/s) | - |

Decode reads 12.40 GB of weights per token; at the measured 240 GB/s peak that is a 51.7 ms floor.

Supported: text only, greedy decoding, chunked prefill measured up to 64K context, KV cache in fp16, kv8a16 or kv4a16 (int8 / int4 storage, f16 math). Not supported: sampling, batching, the MTP layer, vision, the NPU.

## Hardware and model

- AMD Ryzen AI MAX+ 395, Radeon 8060S (gfx1151, 40 CUs), 32 GB LPDDR5X-8000 shared memory. Measured facts: [docs/hardware.md](docs/hardware.md).
- `Qwen3.8-27B-IQ4_XS-3.84bpw.gguf` (12.17 GiB): 64 layers, 16 full attention + 48 Gated DeltaNet. The GGUF mmap is the only copy of the weights.

## Quick start

Needs HRX built at `/home/q/hrx` and the ROCm 10.0.0 runtime packages (`engine/hrx-env.sh --fetch`). See [docs/build-and-run.md](docs/build-and-run.md).

```
source engine/hrx-env.sh
engine/build_hrx.sh                                   # drivers into engine/build/
M=~/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf
cd engine/gpu/loom
python3 tools/emit_prefill_pp.py $M ~/yah-scratch/pp2048 2048           # prefill HAL set
YAH_CTX=32768 python3 tools/emit_prefill_pp.py $M ~/yah-scratch/c32k 2048   # chunked, 32K context
python3 tools/emit_decode.py $M ~/yah-scratch/dec32k 32768               # decode set for it
cd -
engine/run/gpu_run.sh pp -- engine/build/loom_forward_pp $M ~/yah-scratch/pp2048 ~/yah-scratch/pp 2048 ids.txt
YAH_GEN=64 YAH_DECODE_HAL=~/yah-scratch/dec32k \
  engine/run/gpu_run.sh gen -- engine/build/loom_forward_pp $M ~/yah-scratch/c32k ~/yah-scratch/gen 8192 ids.txt
engine/run/gpu_run.sh dec -- engine/build/loom_decode $M ~/yah-scratch/dec32k --ids "760 6511 314 9338 369" --gen 20
```

Run every GPU job through `engine/run/gpu_run.sh`: a bad dispatch on this box hangs the GPU and reboots the machine, and the script keeps the kernel log.

## Docs

- [docs/architecture.md](docs/architecture.md): the model, the prefill pipeline, the decode step, data layouts, HAL sets.
- [docs/build-and-run.md](docs/build-and-run.md): build, emit, run, correctness gates, profiling, GPU safety, measuring rules.
- [docs/results.md](docs/results.md): current numbers, where the time goes, HIP baselines, what was tried and lost.
- [docs/hardware.md](docs/hardware.md): measured facts about Strix Halo.
- [AGENTS.md](AGENTS.md): conventions for code, docs, measuring and commits.

License: see [LICENSE](LICENSE).
