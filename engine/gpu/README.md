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