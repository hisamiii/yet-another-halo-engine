# kv/

Minimal memory is a requirement, not a nice-to-have.

Implemented: a flat, per-layer cache that can be stored FP16, Q8 or Q4 and is
expanded back to FP16 before attention, so no attention kernel changes. Paging
and the image-token spike are still open.

## Storage widths

`--kv f16|q8|q4` selects the width for K and V together. Both planes are
`attn_layers * max_context * kv_heads * head_dim` values.

| storage | block layout | bytes / 32 values | bits/value | 4k, both planes |
| --- | --- | ---: | ---: | ---: |
| f16 | — | 64 | 16 | 268 MB |
| q8 | fp16 scale + 32 int8 | 34 | 8.5 | 142 MB |
| q4 | fp16 scale + 32 signed nibbles | 18 | 4.5 | 76 MB |

The quantized path also holds a one-layer FP16 scratch (2 x 8.4 MB at 4k),
because attention reads FP16.

## How it works

1. The fused QK-norm/RoPE kernel writes K/V into the scratch.
2. The new rows are packed into the per-layer cache.
3. The whole visible prefix, this chunk included, is expanded back into the
   scratch so attention reads the quantized representation and not the exact
   fp16 the fused write just produced.
4. Attention runs unchanged on the scratch.

Step 3 is O(context) per layer per chunk. At 4k it is small against the FFN,
but at long context or batch-1 decode it is a real cost, and it is why the next
step is in-kernel dequant, which removes both the scratch and the extra pass.

## Rotation

The reference engine (llama.cpp) builds an orthonormal Walsh-Hadamard rotation
and sets `attn_rot_k`/`attn_rot_v` for any quantized KV cache with a head dim
divisible by 64. That is a trap: the flags are generic, but the rotation is only
**applied** by the model builders that read `inp->self_k_rot` / `self_v_rot`,
which are the MLA / lightning-indexer / DSA families (`deepseek32`, `deepseek4`,
`dflash`, `dots3note`, `glm-dsa`, `minimax-m3`, and the experimental
`qwen4exp`). The ordinary Qwen builders — `qwen3.cpp`, `qwen35.cpp`,
`qwen3next.cpp` — contain zero uses. So for a plain Qwen3.8, llama.cpp's
quantized KV is plain per-32 block quantization, no rotation. That is the
behaviour this engine matches.

The engine still implements the K half as an option (`YAH_KV_ROT=1`; Q and K
rotated, which is exact for QK^T) because the mechanism is worth measuring for
larger blocks. Measured on Qwen3.8 IQ4_XS at 2k context it made 4-bit worse, so
it stays off by default:

| | logit cosine vs f16 | 100-token greedy divergence |
| --- | ---: | ---: |
| q8 | 0.99977 | none |
| q4 + rotation | 0.980 | token 47 |
| q4, no rotation | 0.998 | none |

The rotation is implemented correctly — q8 with it is exact — so the difference
is the quantizer, not the transform. Dumping layer 0's real K (2048-token
prefill, 8192 rows x 256; `YAH_DUMP_KV=<prefix>`) and measuring it says why:

| quantity | no rotation | rotated |
| --- | ---: | ---: |
| per-32-block crest (max/rms), mean | **1.405** | 2.781 |
| per-32-block crest, p90 | 1.549 | 5.539 |
| q4 relative MSE of K | **0.0120** | 0.0426 |
| q8 relative MSE of K | **1.4e-5** | 1.8e-4 |
| QK logit error, mean | **0.297** | 0.818 |

K is *flatter than uniform*: a uniform block has crest 1.73 and a Gaussian
max-of-32 about 2.5. At crest 1.4 there is no outlier for the rotation to
remove, and the transform only makes the block more Gaussian, raising the crest
and coarsening the scale. The per-32 block quantizer is already near the 4-bit
floor for this signal (exactly uniform would be 0.51% relative MSE; this is
1.2%).

Per-channel structure does not rescue it either. Across tokens, the per-channel
crest is 1.435 (p90 1.719) and the channel RMS spread is 1.43x (p90/p10), so
KIVI-style per-channel K scales give 0.0121 relative MSE against 0.0120 for the
blocks — no gain. K after QK-norm and RoPE is flat in both axes.

The reconstruction numbers say the rotation should hurt, and on a repeated-prompt
logit comparison it did (cosine 0.980 vs 0.998). But the next-token KLD over
real prose is *mixed*: the rotation lowers the KLD by ~22% (0.0050 vs 0.0064)
and the tail (max 0.033 vs 0.030) is a wash, while the argmax agreement drops
slightly (95.3% vs 95.9%). So a larger reconstruction error can still land
closer in distribution, because only the component along the query matters for
the logits. The rotation is not a clean win here and not a clean loss; it stays
off by default and the flag is there to revisit.

V is not rotated because the attention gate is fused per element inside the
kernel. A rotation mixes exactly the elements the gate treats independently, so
the inverse cannot be applied after it. Matching the reference on V would mean
splitting the gate out of the attention kernel.

## Gate

`kv-quant-check` verifies the pack/unpack against a host reference bit for bit,
the round trip inside the block layout's error bound, and the Hadamard identity
(`H*H = I`). The token gates run with `--kv q8` and `--kv q4`: three prefill
prompts and 20 generated tokens match the f16 result.

## KLD

`tests/kld.sh <model> <text> [tokens]` runs the engine once per configuration,
dumps the next-token logits at every position, and reports
`KL(softmax(f16) || softmax(candidate))` plus the argmax agreement. 512 tokens
of repo prose, mean/p99/max in nats:

| config | KLD mean | KLD p99 | KLD max | top-1 same | implied perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| q8 | 0.000025 | 0.000115 | 0.000235 | 99.80% | +0.002% |
| q4 | 0.006449 | 0.026297 | 0.030372 | 95.90% | +0.65% |
| q4 + rotation | 0.005035 | 0.022502 | 0.033072 | 95.31% | +0.50% |
| q4 + Q4 (w4a4) | 0.008694 | 0.043999 | 0.119704 | 95.70% | +0.87% |
| q4 + Q3 (w4a3) | 0.019152 | 0.131046 | 0.249166 | 91.21% | +1.93% |
| q4 + Q4 + rotation | 0.006567 | 0.043413 | 0.062868 | 94.73% | +0.66% |

`YAH_QATTN=4` rounds Q to the same 4-bit block grid before attention, which
reproduces a w4a4 QK^T's operand error without an int4 MMA kernel (the
accumulation is fp32 either way). It costs +35% KLD over q4 KV; Q at 3 bits
costs 3x. The argmax agrees 95.9% of the time at q4, so the damage is a ~4%
single-step flip rate on a sub-percent perplexity shift, not a collapse.

## Open

- In-kernel dequant, removing the scratch and the O(context) expansion.
- Per-channel K scales (KIVI): keys have per-channel outliers and values
  per-token ones, and a uniform per-32 block ignores that asymmetry.
- Quantized KV paging for the image-token spike.
