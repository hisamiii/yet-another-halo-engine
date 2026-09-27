# Targets

These are the only model artifacts the engine has to serve. All are already on this machine.

| file | size | bpw | role |
| --- | ---: | ---: | --- |
| `Qwen3.8-27B-IQ4_XS-3.84bpw.gguf` | 12.17 GiB | 3.828 | primary target |
| `Qwen3.8-27B-UD-Q4_K_S.gguf` | 14.29 GiB | 4.494 | second target |
| `Qwen3.8-27B-UD-Q4_K_S-requant294-Q4K.gguf` | 14.95 GiB | -- | requant experiment, not yet censused |
| `mmproj-F16.gguf` | 0.86 GiB | 16 | vision projector |

836 tensors each, GGUF v3. Architecture: Qwen3.8 27B, dense text/image, hybrid Gated DeltaNet plus one full-attention layer every `full_attention_interval`; hidden 5120, intermediate 17408.

## Census: bytes by role and quant

### IQ4_XS 3.84 bpw (12.17 GiB)

| role | share | dominant types |
| --- | ---: | --- |
| ffn_gate | 18.9% | IQ3_S 7.9, IQ3_XXS 4.4, IQ4_XS 3.6, Q3_K 2.3 |
| ffn_up | 20.0% | IQ4_XS 8.3, IQ3_S 5.9, IQ3_XXS 5.2 |
| ffn_down | 19.1% | IQ3_S 7.3, IQ3_XXS 7.3, IQ4_XS 2.9, Q4_K 1.5 |
| attn_proj (q/k/v/o) | 7.2% | Q4_K, IQ4_XS, Q5_K, IQ3_S |
| attn_gate | 5.8% | IQ4_XS, IQ3_S, Q4_K |
| ssm (Gated DeltaNet) | 5.8% | Q4_K, IQ3_S, IQ4_XS |
| embed | 5.2% | IQ4_XS |
| lm_head | 8.0% | Q6_K |
| other | 10.1% | Q4_K, IQ4_XS, IQ3_S |

FFN is 58.0% of the file, and its decoders are the **IQ family**: IQ3_S, IQ3_XXS, IQ4_XS, then Q3_K and small IQ2. There is almost no K-quant in the FFN.

### UD-Q4_K_S (14.29 GiB)

| role | share | dominant types |
| --- | ---: | --- |
| ffn_gate | 19.9% | IQ4_XS 10.8, Q5_K 2.8, Q4_K 2.3, Q3_K 1.5, IQ3_S 1.2 |
| ffn_up | 20.5% | IQ4_XS 10.5, Q4_K 3.3, Q5_K 2.8, Q3_K 1.2, Q6_K 1.0 |
| ffn_down | 21.2% | IQ4_XS 8.9, Q5_K 5.6, Q4_K 3.9, IQ3_S 1.5 |
| attn_proj | 7.0% | IQ4_XS, Q5_K, Q4_K, Q6_K |
| attn_gate | 5.5% | IQ4_XS, Q4_K, Q5_K |
| ssm | 6.2% | Q5_K 2.8, IQ4_XS 1.6, Q4_K 1.3 |
| embed | 3.6% | Q3_K |
| lm_head | 6.8% | Q6_K |
| other | 9.3% | -- |

FFN is 61.6% of the file and is **IQ4_XS plus the K-quants** (Q4_K, Q5_K, Q6_K, Q3_K).

## The decoder union

Covering both targets needs **twelve** formats: `IQ4_XS`, `IQ3_S`, `IQ3_XXS`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ4_NL`, `Q2_K`(trace), `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, plus `Q8_0`/`F32` for the projector and small tensors.

`IQ4_XS` is the crossover: 27.5% of the 3.84 bpw file and 41.0% of UD-Q4_K_S. It is the one decoder that must be excellent on both.

## What this costs, and where the current engine is incomplete

- **The 3.84 bpw target is an IQ-family shard.** IQ3_S + IQ3_XXS + IQ4_XS are 73% of it, and IQ3_XXS alone is ~19% of the file. The existing paired gate/up kernel has **no IQ3_XXS case**, so a third of the FFN on the primary target runs unpaired, as two separate store kernels with tail predicates. That is a concrete, target-specific win for a greenfield engine.
- **UD-Q4_K_S wants the opposite**: the Q4_K/Q5_K paired path with the tail-free `Complete` variant is exactly what it needs, and it is the minority case on the 3.84 bpw shard.
- **`Complete` is type-dependent**: worth ~4% on Q4_K, neutral on Q6_K/IQ3_XXS, and **-15% on IQ4_XS**, so the kernel choice has to be per (type, shape), which a fixed-shape engine can bake.
- **`lm_head` is Q6_K at 7-8% of both files** and is decode-only (a GEMV), so it is a `tg` concern, not prefill.
- **`embed` is a lookup**, not a GEMM: IQ4_XS on one target and Q3_K on the other.

## Why this is the right thing to bake

Twelve decoders and two files is a small, closed set. The current engine carries a route table over many more combinations and still has coverage gaps on these two artifacts. A fixed-shape engine can instantiate exactly these, pick `Complete` per type, and skip every format it will never see -- which is the whole argument for the narrow scope.
