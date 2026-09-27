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
is the quantizer, not the transform. The mechanism is the block size: at 32
elements a per-channel outlier already spoils only 32 of the 256 values in a
head, while the rotation spreads that energy into every block and raises the
scale floor for all of them. Rotation should pay off with larger blocks; that
is the next experiment.

V is not rotated because the attention gate is fused per element inside the
kernel. A rotation mixes exactly the elements the gate treats independently, so
the inverse cannot be applied after it. Matching the reference on V would mean
splitting the gate out of the attention kernel.

## Gate

`kv-quant-check` verifies the pack/unpack against a host reference bit for bit,
the round trip inside the block layout's error bound, and the Hadamard identity
(`H*H = I`). The token gates run with `--kv q8` and `--kv q4`: three prefill
prompts and 20 generated tokens match the f16 result.

## Open

- In-kernel dequant, removing the scratch and the O(context) expansion.
- Per-channel K scales (KIVI): keys have per-channel outliers and values
  per-token ones, and a uniform per-32 block ignores that asymmetry.
- Quantized KV paging for the image-token spike.
