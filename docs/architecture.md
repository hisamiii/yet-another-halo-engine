# Architecture

The engine runs one model (Qwen3.8-27B, IQ4_XS GGUF) on one GPU (Strix Halo, gfx1151). Every kernel is written in Loom and compiled for the exact shapes of this model. Python generators write the Loom source, `emit_hal.py` compiles it to an HRX HAL executable, and two small C++ drivers dispatch the HAL sets through the HRX native API. There is no general tensor runtime: the layer loop is hand-written.

This page describes the model, the prefill pipeline, the decode step, the data layouts and the HAL sets. Numbers are in [results.md](results.md), measured hardware facts in [hardware.md](hardware.md).

## The model

Qwen3.8-27B is a hybrid: every 4th layer is full (softmax) attention, the others are Gated DeltaNet (a linear-attention recurrence). The GGUF stores it as architecture `qwen35`. `engine/core/config.hpp` reads every field from the file.

| item | value |
|---|---|
| layers | 64 main layers + 1 MTP (`nextn`) layer, which the engine does not run |
| full-attention layers | 16: layers 3, 7, 11, ..., 63 (`(l + 1) % 4 == 0`) |
| Gated DeltaNet layers | 48 |
| hidden | 5120 |
| FFN | SwiGLU, 17408 |
| attention | 24 query heads, 4 KV heads (GQA 6), head dim 256, RoPE on 64 of 256 dims, per-head QK RMSNorm, sigmoid output gate |
| attention q projection | 12288 rows: per head 256 q rows then 256 gate rows |
| DeltaNet | 16 key heads, 48 value heads, head dim 128, state 128 x 128 per value head, short conv width 4 over 10240 channels (q 2048, k 2048, v 6144), z gate 6144, alpha / beta 48 each |
| vocab | 248320; output head Q6_K, token embedding IQ4_XS |
| weights | 12.17 GiB, 3.84 bpw, 11 formats: IQ4_XS, IQ3_S, IQ3_XXS, IQ2_XXS, IQ2_XS, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 |

The weights are the GGUF's own mmap, imported once into HRX as one device-visible buffer. Every tensor is an offset into that import. There is no second copy and no repacking.

## Prefill

`engine/run/loom_forward_pp.cc` runs a prompt through all 64 layers as one batch of B tokens (B = 2048 by default). Every kernel has the token count in its grid, so there is no per-tile loop. All activations are token-major (`[token][row]`). The residual stream is f32; GEMM inputs are f16.

### Kernels per layer, in order

Every layer starts with `yah_half_norm` (RMSNorm, writes the f16 GEMM input).

Full-attention layer:

1. `gemm_kqg` (attn_q, 12288 rows): the GEMM epilogue writes q and gate to separate buffers.
2. `gemm_kstore` attn_k and attn_v (1024 rows each).
3. `yah_fused_qk_rope_batched`: QK RMSNorm and RoPE. It writes roped q, and writes K straight into its page of the K pool. V goes to a one-chunk f16 scratch.
4. KV writers: fp16 `yah_vtpage` (V^T tiles into the paged pool); quantized KV `yah_kmean` (first chunk only), `yah_kq8` or `yah_kq4`, then `yah_vq8` or `yah_vq4`.
5. `yah_attn_wmma` (FlashAttention-style, see below). It writes `f16(o / sum * sigmoid(gate))` straight into the o-projection input.
6. `gemm_kres` attn_output (5120 x 6144): the GEMM reads the residual and writes residual + W x.

Gated DeltaNet layer:

1. `gemm_kstore` attn_qkv (10240 rows), attn_gate (z, 6144), ssm_alpha and ssm_beta (48 rows each).
2. `yah_ssm_conv_kq`: the causal conv over the 10240 channels with the q / k L2 normalization (`prep_kq`) fused in.
3. `yah_conv_state` (chunked sets only): saves the last conv inputs for the next chunk.
4. `yah_deltanet_prep_ab`: decay alpha and beta per token and head.
5. `yah_deltanet`: chunked WY Gated DeltaNet (see below).
6. `yah_ssm_postnorm_fp16`: gated RMSNorm per head with z, f16 out.
7. `gemm_kres` ssm_out (5120 x 6144).

Then the FFN, for every layer:

1. `yah_half_norm` (post_attention_norm).
2. `gemm_kstore` ffn_gate (17408 rows, f32 out).
3. `gemm_swiglu` ffn_up: `silu(gate) * up`, f16 out.
4. `gemm_kres` ffn_down (5120 x 17408).

After the last layer of the last chunk: `yah_rmsnorm` on the last token, the Q6_K output GEMV (`yah_gemv_q6k`, 248320 rows) and `yah_argmax`. A pp2048 pass is 867 dispatches. The token embedding is dequantized on the host and uploaded once per chunk.

### The tile GEMM

All big GEMMs come from `tools/gen_gemm_tile.py`. They are about 90% of prefill time.

- Workgroup tile: 128 rows x 256 tokens, wave32. 16 waves of 32 x 64 by default; 8 waves of 32 x 128 (`WAVE_FMTS`) for IQ4_XS, IQ3_S, IQ3_XXS, Q3_K, Q4_K, Q5_K.
- Per K phase, the decoded weight tile and the activation tile are both in LDS, so the MMA loop reads only LDS. LDS rows are padded (weights +8 f16, activations +8 f16) to remove bank conflicts.
- The next phase's weight bytes and activations are loaded into registers while the current phase computes, then stored to LDS after it. A schedule fence keeps the next loads behind the current LDS stores.
- Decode-ahead (IQ4_XS, Q4_K, Q6_K): the decoding waves decode phase p+1 into a second weight tile while every wave multiplies phase p. Off for the IQ4_XS and Q4_K residual GEMMs at K = 6144, where it serializes on `vmcnt(0)`.
- The weight decode is exact (bit-identical to the reference dequant). Decoders work on 32-bit words, read headers with one vector load per block, and narrow to f16 with `v_fma_mix` where `v_cvt_f16_f32` would hit the v0..v127 window.
- Epilogue kinds: `kstore` (plain store), `kqg` (q / gate split), `swiglu` (reads the f32 gate, LDS epilogue for IQ3), `kres` (fused residual add into a second hidden buffer; the driver swaps the two).
- On grids of 320 or more workgroups, the second workgroup on each WGP in the first round runs 8000 empty workgroup barriers before it starts. This breaks the lockstep epilogue bursts of the short-K residual GEMMs.
- Small shapes (the 48-row ssm_alpha / ssm_beta) use a 16-row x 64-token tile.

HAL names encode the shape: `gemm_<kind>_<fmt>_<m_tiles>_<k_blocks>.hal`, with m_tiles = rows / 16 and k_blocks = K / 256 (Q8_0: K / 32). For example `gemm_kstore_iq4xs_1088_20` is a 17408 x 5120 IQ4_XS GEMM.

### Attention

`tools/gen_attn_fa.py`, FlashAttention-style with the softmax in registers. A workgroup is 32 query tokens x 2 query heads of one GQA group, 8 waves; each wave pair owns 16 queries and each wave one half of the head dim.

- S^T = K Q^T, so a lane owns one query column; the pair adds its partial scores through a private LDS slot.
- P^T becomes the B operand of the P.V WMMA through one xor-16 swizzle. V is read as V^T tiles.
- The O rescale is skipped when no row max of the wave grew (exact).
- Workgroups run head-pair-fastest in longest-first order, so K/V tiles hit L2.
- One V buffer (loaded at the top of phase A, staged at its end); 3 workgroups per WGP.
- Chunked prefill emits one attention HAL per chunk (`wmma_c<i>.hal`, start_pos = i B).

It is not bit-identical to the HIP-order kernel; the tiered accuracy gate accepted it (see build-and-run.md).

### Gated DeltaNet

`tools/gen_gdn_chunk.py`: the chunked WY form (chunk C = 32 tokens). A workgroup takes 64 value rows of one head (grid 2 x 48), 8 waves; the state stays in f32 WMMA accumulators and the matmul inputs are f16. Gating is in log space; log2(alpha) is clamped at -100 because alpha underflows to 0 deep in the model. Output error vs the exact recurrence is ~2e-4 relative; end-to-end KLD ~3e-6.

### Chunked prefill (long context)

A set emitted with `YAH_CTX=T` runs every kernel at the chunk size B but sizes the KV pools for T tokens. The driver runs the prompt in T_run / B passes over the 64 layers and carries the conv state and the DeltaNet state between chunks. T_run may be any multiple of B up to T, which leaves room in the pools for decode. One-pass and chunked runs are bit-identical at 8K.

## Decode

`engine/model/loom_decoder.hpp` (`LoomDecoder`) is the single-token step. `loom_decode` feeds a prompt through it one token at a time; `loom_forward_pp` with `YAH_GEN` runs it after a prefill, on the prefill's KV pools and recurrent state. Decode reads all 12.40 GB of weights (token_embd excluded) once per token, so it is a bandwidth problem: the target is 240 GB/s.

### Step, per layer

1. `rmsnorm` (512 lanes).
2. Input projections as one band-fused GEMV (`gb_*`): attn_q / attn_k / attn_v, or attn_qkv / attn_gate / ssm_alpha / ssm_beta, in one dispatch.
3. Full attention: `unpack` (q / gate), `rope` (QK norm + RoPE), KV append (`dattn_kvappend`, or `dattn_kappend_q` + `dattn_vappend_q`), `dattn_part` (or `dattn_part_q`), `dattn_reduce`, then `gv_resid` attn_output.
4. DeltaNet: `deltanet_conv` (the conv fused into the DeltaNet step, gated norm inside), then `gv_resid` ssm_out.
5. `rmsnorm`, `gv_swiglu` (ffn_gate and ffn_up in one kernel), `gv_resid` ffn_down.

Head: `rmsnorm`, `gv_plain` output (Q6_K, 248320 rows), `argmax`. The argmax writes the next token id into a device token stream; the next step's `embed` kernel (IQ4_XS row) reads it. Positions come from a device array. So steps are enqueued back to back and the host waits only after the prompt and at the end.

### GEMV design

`tools/gen_gemv.py`, one kernel per (kind, formats, rows, K). The arithmetic is HIP's sub-16 decode: a row is cut into 16-element sub-blocks; each decodes to 16 small unsigned integers, a scale and an offset, and the lane accumulates `scale * dot(q, x) - offset * sum(x)`.

- Decode on 32-bit words (`vector<4xi32>`), never on `vector<16xi8>`, which Loom lowers byte by byte.
- Two rows per wave share each x load; 4 waves per workgroup (R x W = 2 x 4).
- `pipeline(2)` read-ahead on the sub-block loop.
- Butterfly reduce over the wave.
- Kinds: `plain`, `resid` (y += W x, the residual add fused), `swiglu` (gate and up of mixed formats in one kernel), and bands (`gen_bands`, several output matrices of one input in one dispatch).

The GEMV families run at 214-236 GB/s of the 240 GB/s peak.

### Decode attention

`tools/gen_decode_attn.py`, split-K over pages:

- `part`: one workgroup per (KV head, 256-key page), with the six query heads of the GQA group. Scores (thread = key), per-head block max and sum, p in LDS, then P.V (thread = dim). Read-ahead depth 3 on the K rows and 2 on the V^T tiles. Grid order (head, page) so the four KV heads of a page read together.
- `reduce`: one workgroup per query head, merges the pages and applies the sigmoid gate.
- Quantized KV (`part_q`): same grid and outputs, the reduce is shared. q goes to LDS in the codes' storage order (kv4: rotated in place by an 8-stage LDS butterfly). Each key thread dequantizes its own row. The open f16 V tile is added once after the tile loop. The K channel mean is never added back: q.m is constant over keys and cancels in the softmax.

## Data layouts

### Paged KV

All KV caches are paged with 256-token pages. One page table per sequence (i32, logical page -> physical page) is shared by all layers. A 16-key attention tile always lies in one page, so attention does one page-table load per K tile and per V tile. The host validates every entry (< number of pages) before upload; attention assumes the bound, the cache writers clamp to it. The context must be a multiple of 256.

Per full-attention layer, at context T (pool rows = T):

| format | K | V | bytes per token per layer |
|---|---|---|---:|
| fp16 | `[T][1024]` f16, row of position p = `ptab[p/256]*256 + p%256` | V^T `[4 kv heads][T/16 tiles][256 dims][16 keys]` f16, physical tile `ptab[t/16]*16 + t%16` | 4096 |
| kv8a16 | int8, one scale per (token, KV head, 128-dim half): codes `[T][1024 B]`, scales `[T][8]` f16 pairs | uint8 per channel per 16-key tile around the tile midrange: codes `[4][tiles][256] x 16 B`, stats `[4][tiles][256]` f16 pairs | 2336 |
| kv4a16 | H256 (Walsh-Hadamard over the whole head) of K, asymmetric int4 per 32-dim group: codes `[T][512 B]`, scales `[T][32]` f16 pairs | 15 levels per channel per 16-key tile: codes `[4][tiles][256] x 8 B`, stats as kv8 | 1408 |

Over 16 layers that is 64 KiB (fp16), 36.5 KiB (kv8a16) and 22 KiB (kv4a16) per token of context; at 32K context 2.0 / 1.14 / 0.69 GiB.

Details of the quantized formats (`tools/gen_kvq.py`):

- K is centred first: `yah_kmean` computes the per-channel mean of the first chunk's K (deterministic, no atomics), kept per layer. Subtracting a per-channel constant shifts each query row's scores uniformly, so softmax is unchanged.
- kv4 rotates K with H256; the attention rotates q the same way, so q.k is unchanged.
- Codes are stored in an order that lets the attention build f16 values with masks and ORs (`1 + u/16`, `1 + u/256`), then dequantize with one fma `f * S + C'`. All attention math stays f16 WMMA ("a16").
- With K and V quantized, the f16 KV cache is a one-layer, one-chunk scratch that RoPE writes and the quantizers read.
- Decode keeps each layer's open 16-key V tile in f16 and quantizes it when the 16th key arrives. Prefill runs end on whole chunks, so a handoff always starts on a tile boundary.

### Recurrent state

| buffer | layout |
|---|---|
| conv state | `[48 layers][10240 channels][4]` f32. Decode ping-pongs between two copies per token, so the three value heads that share a key head never read a half-updated state. |
| DeltaNet state | `[48 layers][48 value heads][128][128]` f32 (3 MiB per layer) |

Prefill and decode use the same layouts, so the decoder binds the prefill's buffers directly.

## HAL sets

A HAL set is a directory of compiled kernels plus one text file that records how to launch them. The drivers read the file and refuse any other grid: Loom drops index clamps it proves redundant from the launch grid, so a larger grid reads out of bounds (see build-and-run.md, GPU safety).

### Prefill set: `tools/emit_prefill_pp.py <model.gguf> <dir> [B]`

- One GEMM HAL per (kind, format, shape) on the shard, the fixed kernels (`norm.hal`, `convkq.hal`, `rowsplit.hal` = DeltaNet, `postnorm.hal`, `rope.hal`, `wmma.hal` = attention, `vtpage.hal`, `rmsnorm.hal`, `gemv.hal`, `argmax.hal`, ...), one rope / attention / KV-writer HAL per chunk, and the IQ grid tables (`grid_*.bin`, `ksigns_*.bin`).
- `dispatch.txt`, one line per HAL: `<hal> <tokens per workgroup> <row groups> <token tiles>`. Marker rows with zeros carry facts: `kv_paged`, `kv16_scratch`, `rope_kpaged`, `attn_f16out`, `attn_kq8` / `attn_kq4` / `attn_vq8` / `attn_vq4`, and `ctx <B> 0 <T>` for chunked sets.
- `tools/footprint_gate.py` runs for every GEMM before it is emitted and refuses a kernel whose declared operand footprint is larger than the buffer the driver binds.

### Decode set: `tools/emit_decode.py <model.gguf> <dir> [max_context]`

- `gv_<kind>_<fmts>_<M>_<K>.hal` and `gb_<fmts>_<Ms>_<K>.hal` (GEMVs and band GEMVs), `dattn_*` (attention), `rmsnorm`, `unpack`, `rope`, `deltanet_conv`, `embed`, `argmax`, and the IQ tables.
- `decode.txt`: `ctx <T>`, `kv q <K bits> <V bits>` for quantized KV, `rw <kind> <R> <W>` (GEMV geometry), and `grid <hal> <workgroups> 0` for every GEMV kernel.
- To decode after a prefill, emit the decode set with the same context as the prefill pools and the same `YAH_KV`. The driver refuses a mismatch.

### From generator to HAL

1. A generator (`tools/gen_*.py`) prints Loom text, one kernel per `gen_*` function. A few kernels are still hand-written `.loom` files in `engine/gpu/loom/`; the emitters compile those with fixed configs.
2. `engine/gpu/loom/emit_hal.py <file.loom> <outdir> sym=value ...` replaces every `config.get` with its constant, drops the `config.decl`, and has `iree-run-loom --emit-only --emit-hal-executable` compile the kernel for gfx1151. The kernel is built for exactly that config. `YAH_LOOM_HOME` picks the HRX/Loom tree (default `/home/q/hrx`).
3. The emitter copies the result into the set and writes `dispatch.txt` or `decode.txt`.

Production sets must build and run with stock HRX/Loom (see build-and-run.md, Local HRX patches). An experimental compiler may be used for diagnosis only.
