# model/

Qwen3.8 27B, one architecture, shapes baked.

- hidden 5120, intermediate 17408, 866 tensors, in both target GGUFs.
- Layer types: full attention every `full_attention_interval`, Gated DeltaNet otherwise. The pattern is read from the GGUF and validated against it.
- Prefill chunk 2048; decode is a separate, batch-1 path.

This is the M0 gate: token-for-token match against a reference on a fixed prompt. Gated DeltaNet is not a plain transformer, and a state-handling error stays invisible until the logits diverge.
