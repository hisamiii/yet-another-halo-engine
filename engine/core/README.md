# core/

GGUF reader, mmap, tensor table and tokenizer. Ported rather than written: the file format is not the interesting part of the engine and the reference implementation is already correct.

- header, metadata KV, tensor table, type and size, mmap of the weight region
- BPE tokenizer and chat template, both from the GGUF's own metadata
- the twelve decoders the two targets need (see [../../docs/targets.md](../../docs/targets.md))

Gate: tensor table and byte sizes match the reference exactly.
