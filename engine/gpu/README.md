# gpu/

The GPU path. The kernel framework is **ported verbatim** from the reference
engine (MIT) into `ported/`, preserving its `src/...` include paths and its
`gufo` namespaces so the port is a copy rather than a rewrite. A later pass can
move the namespaces to `yah`; the value here is that the measured instruction
behaviour transfers unchanged. Everything outside `ported/` is the engine's own
tooling.

## Build

The kernel headers need HIP, so they are built as HIP tools rather than as part
of the CMake C++ target:

    hipcc -std=c++20 -O3 -Iengine/gpu/ported -Iengine \
      engine/gpu/gemm_bench.hip \
      engine/gpu/ported/src/core/quant/ggml_dequant.cpp \
      -o engine/build/yah-gemm-bench --offload-arch=gfx1151

## Per-type rates at the production shape

`yah-gemm-bench all 17408 5120 2048 12` (m=17408, k=5120, batch=2048), median
of three repetitions:

| type | TFLOPS | note |
| --- | ---: | --- |
| Q4_K | 32.7 | fastest; the paired and `Complete` paths target it |
| Q5_K | 30.8 | |
| Q8_0 | 30.2 | activations and small tensors |
| IQ4_NL | 30.0 | |
| Q6_K | 29.6 | `lm_head` on the target |
| IQ4_XS | 29.4 | 27.5% of the target |
| Q2_K | 29.3 | trace only |
| Q3_K | 28.4 | |
| IQ3_S | 28.1 | 27.2% of the target |
| IQ3_XXS | 26.9 | 18.8% of the target; **no paired gate/up case** |
| IQ2_XXS | 26.9 | |
| IQ2_S | 26.4 | |
| IQ2_XS | 26.2 | slowest |

The spread is ~25% top to bottom and the absolute values agree with the
model-level ~30 TF. That agreement is the independent confirmation that the
earlier 38-40 TF kernel figures were a zeroed-operand clock artifact rather than
headroom: with real weight data the part runs near 2100-2200 MHz, and these are
the rates it actually sustains.

The target's FFN types sit mid-pack, and the decoders that are most expensive on
paper (the IQ2 family) are also slowest in practice, so the ordering is
consistent. IQ3_XXS being both a fifth of the target and unpaired is the
concrete first coverage target.

## Correctness

The gate for trusting the rates above. `gemm_check.hip` decodes the same weight
bytes twice -- once by the device kernel, once by the host `DotProduct`
reference -- and compares the whole output matrix:

    hipcc -std=c++20 -O3 -Iengine/gpu/ported -Iengine engine/gpu/gemm_check.hip \
      engine/gpu/ported/src/core/quant/ggml_dequant.cpp \
      -o engine/build/yah-gemm-check --offload-arch=gfx1151
    ./engine/build/yah-gemm-check <type> 512 512 512

All thirteen types pass, relative RMS between **1.7e-4 and 3.6e-4**. That is
fp32 accumulation-order agreement: the device and host decoders are reading the
same bytes the same way, which is what a correct port looks like. Worst is Q5_K
at 3.6e-4; the target's FFN types are 1.8-2.4e-4.
### The FFN launcher chain

`ffn_check.hip` runs a whole FFN block through the production launchers --
RMSNorm -> gate/up SwiGLU -> down with residual -- and compares the output
against the host decoder, reading the device's own FP16 intermediates back so
that only the kernels and launchers are under test:

    ./engine/build/yah-ffn-check <gate> <up> <down> <batch>

| gate/up/down | paired path | norm max abs | rel RMS |
| --- | :---: | ---: | ---: |
| Q4_K | yes | 1.9e-6 | 4.7e-5 |
| IQ4_XS | yes | 1.9e-6 | 8.2e-5 |
| IQ3_XXS | **no** | 1.9e-6 | 8.4e-5 |
| IQ3_S | **no** | 1.9e-6 | 2.1e-4 |

The `paired` column is the one that matters for the target: IQ3_XXS and IQ3_S
still have no case in `TryLaunchBatchedDualQuantGEMMSwiGLUFp16`, so those blocks
fall back to a gate store plus an up SwiGLU. The fallback is numerically
correct; its cost is an extra GEMM pass, which is the coverage gap the kernel
workstream exists to close.
### Causal GQA attention

`attention_check.hip` runs the ported prefill attention at the model's own shapes
(24 query heads, 4 KV heads, head_dim 256) against a from-definition reference:
softmax over the causal prefix with `1/sqrt(head_dim)`, then the output gate.

| batch | rel RMS |
| ---: | ---: |
| 2 | 2.5e-9 |
| 64 | 3.5e-7 |
| 256 | 3.8e-7 |
| 1024 | 3.0e-7 |

Two semantics the reference had to get right, both learned by failing first:

- **The gate is per element, not per head.** The kernel indexes it with the same
  `[token, head, d]` layout as the output. An early reference applied a single
  gate value per head, which read as a 100%-relative-error kernel failure and
  was entirely the reference's fault -- the device values matched the inputs
  exactly.
- **The cache representation is selected by null-ness.** `BatchedAttentionKernel`
  reads the FP32 cache when `k_cache != nullptr` and the FP16 cache when it is
  null, and it only *reads*: the host wrapper writes the cache first unless
  `skip_kv_write`, which is the fused QK-norm/RoPE path. The layout is
  `[layer, kv_head, position, head_dim]`.

The reference is O(batch^2), so this stops at 1024; the kernel is unchanged at
the 2048 production batch.
### Gated DeltaNet (SSM)

`ssm_check.hip` runs the ported prefill recurrence at the model's shapes -- qkv
10240 (q 16x128, k 16x128, v 48x128), 48 value heads, key = value dim 128, conv
kernel 4 -- against a from-definition reference: a causal conv, the delta rule
with exponential decay, then per-head RMSNorm and a SiLU gate.

| batch | rel RMS |
| ---: | ---: |
| 8 | 9.5e-8 |
| 64 | 1.0e-7 |
| 256 | 1.1e-7 |

Three things the reference had to get right, each of which produced a
plausible-looking failure first:

- **The qkv layout is q, then k, then v**, and the kernel's `q_h` reads the
  *first* section while `k_h` reads the second. Swapping them -- which the name
  `qkv` and the module's own `qkv` ordering invite -- gives a 2-4% error that
  reads like a precision problem rather than a layout one.
- **`ssm_a` must be negative.** The decay is `exp(softplus(alpha + dt) * ssm_a)`
  and the kernel's default is -0.05. Random bytes can make it positive, the
  state then grows without bound, and *both* device and reference NaN -- which
  looks like a broken kernel until you notice the reference did it too.
- **The conv is causal, taps `[w0..w3]` at offsets `[-3..0]`, zero initial
  history**, and the launcher updates the conv state only after the conv kernel
  has read it.

The recurrence state is `[layer, head, key_dim, val_dim]`, fp32 or bf16.
## Clock

A rate here is not comparable without the clock it ran at. Use
`engine/tests/bench_with_clock.sh <cmd>`; the part has three SCLK levels
(600 / 1261 / 2900 MHz) and sustained work does not reach the top one.

Measured on the long run (`all`, 24 iterations each): the clock settles at a
**mean 1848 MHz** (max 1995) over 260 samples, and the per-type rate falls with
it -- IQ2_XXS reads 24.3 TF sustained against 26.9 TF in a short burst. A single
type at 12 iterations finishes in well under a second, before the part has
ramped, so its rate and clock are not meaningful on their own. Any measurement
needs several seconds of steady load behind it.

## Next here

The per-type rates and their correctness are settled. What is not wired yet is
the engine's own layer loop, and the promotion gate that samples `pp_dpm_sclk`
around a full prefill rather than a synthetic GEMM.