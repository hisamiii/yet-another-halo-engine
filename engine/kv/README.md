# kv/

Minimal memory is a requirement, not a nice-to-have.

- Paged and quantized.
- The model is 12-14 GiB against a 28 GB shared CPU/iGPU pool, so the KV is the only per-request growth.
- Image tokens are long (64-16384 merged per image), so the image prefill spike is the peak-memory case.

Gate: peak RSS = mmap + KV + transient operands, measured rather than argued.
