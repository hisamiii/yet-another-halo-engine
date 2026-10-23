# Architecture

The engine runs one model (Qwen3.8-27B as hipfire's MQ4 file, `qwen3.8-27b.mq4-xts`) on one GPU (Strix Halo, gfx1151). Every kernel is written in Loom and compiled for the exact shapes of this model. Python generators write the Loom source, `emit_hal.py` compiles it to an HRX HAL executable, and two small C++ drivers dispatch the HAL sets through the HRX native API. There is no general tensor runtime: the layer loop is hand-written.

This page describes the model, the prefill pipeline, the decode step, the data layouts and the HAL sets. Numbers are in [results.md](results.md), measured hardware facts in [hardware.md](hardware.md).

Until 2026-10-20 the engine also ran the IQ4_XS GGUF model (11 quant formats, tile GEMMs with per-format decoders, afrag GEMMs, a tuner) and split prefill GEMMs with the NPU. Both are gone from main; the tag `pre-cleanup-2026-10-20` has the GGUF path and branch `npu` the NPU work.

## The model

Qwen3.8-27B is a hybrid: every 4th layer is full (softmax) attention, the others are Gated DeltaNet (a linear-attention recurrence). `engine/core/config.hpp` reads every field from the file.

| item | value |
|---|---|
| layers | 64 main layers (the file has no MTP layer) |
| full-attention layers | 16: layers 3, 7, 11, ..., 63 (`(l + 1) % 4 == 0`) |
| Gated DeltaNet layers | 48 |
| hidden | 5120 |
| FFN | SwiGLU, 17408 |
| attention | 24 query heads, 4 KV heads (GQA 6), head dim 256, RoPE on 64 of 256 dims, per-head QK RMSNorm, sigmoid output gate |
| attention q projection | 12288 rows: per head 256 q rows then 256 gate rows |
| DeltaNet | 16 key heads, 48 value heads, head dim 128, state 128 x 128 per value head, short conv width 4 over 10240 channels (q 2048, k 2048, v 6144), z gate 6144, alpha / beta 48 each |
| vocab | 248320 |
| weights | 13.96 GiB: every linear layer, the output head included, MQ4G256; the token embedding Q8_0; the rest f16 |

MQ4G256 is symmetric int4 with one f16 scale per 128 weights, 136 bytes per 256. Each 256-wide group of every weight row is stored in a rotated basis (random signs, 256-point Walsh-Hadamard, random signs), so every GEMM input gets the same rotation. Each projection also carries an AWQ activation scale (`blk.N.<weight>.awq`, f32): the input is divided by it before the rotation.

### Loading

`engine/core/hfq.cpp` reads the HFQ container and presents it through `core::Gguf`: GGUF tensor names and dims, GGUF metadata keys for the config and the tokenizer (the tokenizer goldens pass). The file is read into locked anonymous memory (2 MB pages where the kernel has them) and changed in place at load (about 4 s):

| tensor | change at load |
|---|---|
| MQ4G256 weights | nibbles xor 8: signed codes, weight = nibble x scale (the stored zero is always -8 x scale) |
| DeltaNet value heads | GGUF order (value head i uses key head i mod 16; HF groups them by key head): rows of `attn_qkv` (v part), `attn_gate`, `ssm_alpha`, `ssm_beta`, channels of `ssm_conv1d`, `ssm_a`, `ssm_dt` |
| norms | f32, 1 + w as GGUF stores them (DeltaNet `ssm_norm` as is) |
| `ssm_a`, `ssm_dt`, `ssm_conv1d`, AWQ scales | f32 (`ssm_a` = -exp(A_log)), placed after the file's bytes in the same import |

`ssm_out` keeps the HF value-head order on its input axis: a head permutation does not commute with the rotation, so its input is written in HF head order. The tensors are then copied into device-local HRX buffers of about 1 GiB cut at tensor boundaries (`LoomWeights`), each host range released right after its copy: RAM holds one copy of the model. A mapped file does not work: reclaim and compaction scan its page-cache folios, and every scan of a GPU-imported page evicts all of the process's GPU queues (hardware.md).

## Prefill

`engine/model/loom_prefill.hpp` (`LoomPrefill`) runs a prompt through all 64 layers as one batch of B tokens (B = 2048 by default); `loom_forward_pp` is its command-line driver. Every kernel has the token count in its grid, so there is no per-tile loop. All activations are token-major (`[token][row]`). The residual stream is f32.

Two modes, chosen when the set is emitted:

- A4 (the default): the GEMMs run on `v_wmma_i32_16x16x16_iu4` with int4 activations, one f32 scale per 128 values of each token. Accuracy is on par with hipfire's A4 route (results.md).
- W4A16 (`YAH_A16=1`, the reference mode): f16 activations on the f16 WMMA, the weights decoded to f16 in LDS (`gen_gemm_tile`).

### A4 kernels per layer

The chunk graph starts with `yah_embed` (`gen_decode_misc.gen_embed_q8_0`, rows = B), which dequantizes the token_embd rows of the chunk's token ids.

Every GEMM input site has a quantizer: the input rotated as above, then int4 codes and one scale per 128 values (hipfire's recipe: the best of 8 candidate scales (amax / 7)(1 + j / 7) / 2 by squared error, round to nearest even, clamp to [-8, 7]) into `aq_` / `ad_`. The norm sites fuse it with the norm (`normq`, `gen_rot.gen_nq`), the others run `rotq` (`gen_rot.gen_q`) on the f16 producer output, and the DeltaNet postnorm fuses it as `pnq` (`gen_rot.gen_pnq`).

Full-attention layer:

1. `normq` (attn_norm).
2. `iu4_kqg` attn_q (12288 rows): the epilogue writes q and gate (f16) to separate buffers; `iu4_kstore` attn_k and attn_v (1024 rows each).
3. `yah_fused_qk_rope_batched` (the k / v form, `ropekv`): K RMSNorm and RoPE, K and V straight into their pages of the KV pools (f16), or the KV quantizers (`yah_kmean` on the first chunk, `yah_kq8` / `yah_kq4`, then `yah_vq8` / `yah_vq4`) for quantized KV.
4. `yah_attn_wmma` (FlashAttention-style, below): the q RMSNorm and RoPE run inside it (`qnorm`); it writes `f16(o / sum * sigmoid(gate))`.
5. `rotq` (K = 6144), then `iu4_kres` attn_output: the GEMM reads the residual and writes residual + W x into the second hidden buffer; the two swap.

Gated DeltaNet layer:

1. `normq` (attn_norm).
2. `iu4_kstore` ssm_alpha and ssm_beta (48 rows each; first, so they read the norm's output before the big GEMMs evict it), attn_qkv (10240 rows).
3. `yah_ssm_conv_kq` (`gen_conv_kq`): the causal conv of the q / k heads with their L2 normalization, a workgroup per (256 channels, 16 tokens).
4. `yah_deltanet` (`gen_gdn_chunk`, below) in two parts (heads 0-39, then 40-47): it computes the v channels' conv and alpha / beta itself.
5. attn_gate (z, 6144 rows) beside the second DeltaNet part, which leaves most of the GPU free; `iu4_kstoreh` writes it in HF head order.
6. `yah_deltanet_prep_ab` advances the conv ring past the real tokens; `pnq`: the gated RMSNorm per head with z, ssm_out's rotation and quantization in one kernel.
7. `iu4_kres` ssm_out.

Then the FFN, for every layer: `normq` (post_attention_norm), `iu4_gu` (gate and up in one GEMM: a workgroup stages 64 gate rows and the same features' 64 up rows, and the epilogue pairs them through LDS into f16 silu(gate) x up, so the gate never goes to memory), `rotq` (K = 17408), `iu4_kres` ffn_down.

After the last layer of the last chunk: `yah_rmsnorm` on the last token, `rot_head` (the rotation without awq), the MQ4 output GEMV (`gen_gemv`, 248320 rows) and `yah_argmax`. The last layer's tail runs only the GEMM token tiles holding the rows the caller reads (`RunLayers` keep: the last token, or none on chunks whose output nobody reads).

### Overlap through completion counters

A graph edge waits for every workgroup of the producer, so each GEMM's tail (its last, partly filled wave of workgroups) would leave the GPU idle. Most producer / consumer pairs therefore skip the edge: the producer adds 1 to a counter per 128-token tile as it finishes it, and the consumer's workgroups poll the counters of the tiles they read (`FlagSlot` in `loom_prefill.hpp`, a slot per pair, zeroed before the chunk; dispatch.txt markers `nqov`, `nqw`, `nqgu`, `guov`, `rqk`, `dnov`, `dnring`, `aov`, `ropew`). So the normq after a kres runs in the kres's tail, the ffn_down rotq in the gu's tail, the conv in the qkv GEMM's tail, the pnq in the gate GEMM's tail, the GEMMs after a normq start on the first finished token tiles, and so on. Consumers that would otherwise queue thousands of workgroups ahead of the next GEMM run as persistent workgroups (`rotqp`, `rotqop`, `pnq_abfrp`, `convkqp`). Where a consumer writes the activations its producer still reads, the two use the other buffer pair (`aq2_` / `ad2_`) and swap. Results are bit-identical to the barriered graph.

### The iu4 GEMM

`tools/gen_gemm_iu4.py` (`iu4_<kind>_<m_tiles>_<k_blocks>.hal`, kinds kstore, kstoreh, kqg, kres, gu):

- Workgroup 128 rows x 128 tokens, 4 x 2 waves of 32 x 64 (the 48-row alpha / beta: 48 x 128, 3 x 2 waves): at 216-232 VGPRs and 42 KB LDS three workgroups share a WGP, so one's fold, barrier and epilogue stalls overlap the others' WMMAs.
- Per 128-K phase (one scale group) the weight and activation nibbles and scales are staged in LDS, double-buffered with one barrier per phase.
- Each wave runs 8 k steps of i32 WMMAs into zeroed accumulators, then folds them: acc += f32(c) x (weight scale x activation scale). In the iu4 C layout, element i of lane l is row 2i + l / 16 and token l % 16, so the weight scales sit in LDS in that order (one vector load) and the activation scale is one value per lane.
- Epilogues through `gen_gemm_tile.lds_epilogue` (wave-private LDS slabs, 4 rows per vector store); the 48-row tile stores its f32 fragments directly.
- Grouped grids (dispatch.txt `iu4grp_<hal> G`): grid (G, token tiles, row tiles / G), x fastest, so a band of G weight tiles stays in L2 while all token tiles run over it.
- The flagged (`f`) and waiting (`w`) forms count finished token tiles and wait on their producer's counters (above).

HAL names encode the shape: m_tiles = rows / 16, k_blocks = K / 256. For example `iu4_kstore_640_20` is a 10240 x 5120 GEMM (attn_qkv).

### The tile GEMM (W4A16)

`tools/gen_gemm_tile.py` with the MQ4 decoder of `tools/gen_gemm_decode.py` (weight = f16(nibble x scale)):

- Workgroup 128 rows x 256 tokens over 4 x 4 wave32 waves of 32 x 64.
- Per K phase (64), the decoded weight tile and the activation tile are both in LDS, so the MMA loop reads only LDS. LDS rows are padded (+8 f16) against bank conflicts.
- The next phase's weight bytes and activations load into registers while the current phase computes and go to LDS after it; a schedule fence keeps those loads behind the current LDS stores.
- Epilogue kinds: `kstore`, `kqg` (q / gate split), `swiglu` (reads the f32 gate), `kres` (residual add).
- On grids of 320 or more workgroups, the second workgroup on each WGP in the first round runs 8000 empty workgroup barriers before it starts, which breaks the lockstep epilogue bursts of the short-K residual GEMMs.
- The 48-row alpha / beta use a 16-row x 64-token tile.

The W4A16 mode runs `yah_half_norm` and `yah_rot256` (`gen_rot.gen`, in place) where A4 runs normq, and `yah_ssm_postnorm_fp16` plus `rot_heads` (the GGUF to HF value-head gather) for ssm_out.

### Attention

`tools/gen_attn_fa.py`, FlashAttention-style with the softmax in registers. A workgroup is 32 query tokens x 2 query heads of one GQA group, 8 waves; each wave pair owns 16 queries and each wave one half of the head dim.

- S^T = K Q^T, so a lane owns one query column; the pair adds its partial scores through a private LDS slot.
- P^T becomes the B operand of the P.V WMMA through one xor-16 swizzle (`v_permlanex16`, HRX patch 0043). V is read as V^T tiles.
- The O rescale is skipped when no row max of the wave grew (exact).
- Workgroups run head-pair-fastest in longest-first order, so K/V tiles hit L2.
- Chunked prefill emits one attention HAL per chunk (`wmma_c<i>.hal`, start_pos = i B).
- Quantized KV: the attention reads the kv8 / kv4 codes directly, or f16 copies of the pools that `yah_kdeq*` / `yah_vdeq*` write in sets that carry them.

It is not bit-identical to a HIP-order kernel; the accuracy gate accepted it (build-and-run.md).

Block-sparse long-context attention (`YAH_FPS_FROM=<chunk>`, FlashPrefill V2 style, `tools/gen_fpsparse.py`; kv4 sets): from that chunk on, `yah_fpmean` keeps per-128-key block means of K and V, `yah_fpsel` picks the key blocks each 16-query tile attends to from the rotated q against the block means, and the attention runs only those (plus the diagonal and the answer tile of the final chunk exact). Off by default.

### Gated DeltaNet

`tools/gen_gdn_chunk.py`: the chunked WY form (chunk C = 32 tokens). A workgroup takes 64 value rows of one head, 8 waves; the state stays in f32 WMMA accumulators and the matmul inputs are f16. Gating is in log space; log2(alpha) is clamped at -100 because alpha underflows to 0 deep in the model. Output error vs the exact recurrence is ~2e-4 relative; end-to-end KLD ~3e-6.

### One HRX graph per chunk

`RunLayers` records a chunk's dispatches into one HRX graph (`LoomGraph` in `engine/model/loom_runtime.hpp`) and launches it once; layer 0 goes into a graph of its own first, so the GPU starts while the host records the rest. Edges come from the byte ranges each dispatch reads and writes (bindings a pair synchronizes through counters are left untracked). Kernels with no data between them run at the same time. Results are bit-identical to dispatching on the stream.

### Chunked prefill (long context)

A set emitted with `YAH_CTX=T` runs every kernel at the chunk size B but sizes the KV pools for T tokens. The prompt runs in passes of B tokens over the 64 layers, carrying the conv ring and the DeltaNet state between chunks. One-pass and chunked runs are bit-identical at 8K.

A prompt that ends inside a chunk runs a partial last chunk. The GEMMs launch only the token tiles that hold real tokens, and chunked sets carry each GEMM also with narrower token tiles (`LoomPrefill::PickGemm` takes the variant with the fewest padded token rows weighted by its cost per row); every other kernel runs the whole chunk over padding rows, which are zero-filled at allocation and so always finite. `yah_deltanet_prep_ab` and the DeltaNet read the real token count from a device scalar: padding tokens get decay 1 and update 0, so they leave the DeltaNet state unchanged, and the conv ring advances only past the real tokens. Attention is causal, so the real tokens never see the padding, and decode overwrites the padding's KV rows before reading them. The KV quantizers also read the real token count: `yah_kmean` averages K over the real rows only and `yah_vq8` / `yah_vq4` repeat the last real row into a partial 16-key tile. With quantized V, `yah_vseed` copies the real V rows of the last, partial 16-key tile into the decoder's open tile.

Slots: `SaveState` / `LoadState` write and read the sequence state after some chunks (KV pools, page table, conv rings, DeltaNet states) raw, for a prefill of the same set (`YAH_SLOT_SAVE` / `YAH_SLOT_LOAD` / `YAH_SLOT_AT` in `loom_forward_pp`): profiling at depth without re-running the prefix.

## Engine

`engine/model/engine.hpp` (`Engine`) loads the model, a chunked prefill set and a decode set once and serves `Generate(prompt, params, on_token)` calls one at a time (the `TextGenerator` interface in `engine/model/generator.hpp`; the server in `engine/serve/` talks only to that). Per request: reset the recurrent state, prefill the whole chunks, finish the tail with a partial chunk or with decode steps (whichever its measured costs say is cheaper), then decode. Greedy takes the argmax on the GPU; temperature / top_p sample on the GPU too (`gen_sample`).

## Decode

`engine/model/loom_decoder.hpp` (`LoomDecoder`) is the single-token step. `loom_decode` feeds a prompt through it one token at a time; `loom_forward_pp` with `YAH_GEN` runs it after a prefill, on the prefill's KV pools and recurrent state. Decode reads 13.63 GB of weights per token (every MQ4 tensor, one embedding row), so it is a bandwidth problem: the floor is ~58.1 ms at 240 GB/s with the DeltaNet state traffic, and the step runs at 96% of it (results.md).

### Step, per layer

1. `normrot` (`gen_rot.gen_norm`): rmsnorm and the rotation of its output in one kernel.
2. Input projections as one band-fused GEMV (`gb_*`): attn_q / attn_k / attn_v, or attn_qkv / attn_gate / ssm_alpha / ssm_beta, in one dispatch.
3. Full attention: `unpack` (q / gate), `rope` (QK norm + RoPE), KV append (`dattn_kvappend`, or `dattn_kappend_q` + `dattn_vappend_q`), `dattn_part` (or `dattn_part_q`), `dattn_reduce`, the rotation `drot_*`, then `gv_resid` attn_output.
4. DeltaNet: `deltanet_conv` (the conv fused into the DeltaNet step, gated norm inside), `drot_*` (into HF head order), then `gv_resid` ssm_out.
5. `normrot`, `gv_swiglu` (ffn_gate and ffn_up in one kernel, with the rotation of its output: `swrot`, below), `gv_resid` ffn_down.

Head: `normrot` without awq, `gv_plain` output (248320 rows) with the argmax in it. The argmax writes the next token id into a device token stream; the next step's `embed` kernel (Q8_0 row) reads it. Positions come from a device array. So steps are enqueued back to back and the host waits only after the prompt and at the end. Each step is submitted as soon as it is recorded (`LoomDevice::Flush`), so the GPU runs one step while the host records the next.

Rotations inside their producers, bit-identical to separate `drot_*` dispatches:

- `normrot` (`drot_norm_*`): each of the 5 workgroups reduces all of x in rmsnorm's lane order, then waves 0-1 rotate the workgroup's 1024 values.
- `swrot` (`gv_swiglurot_*`): ffn_down's input. After its stores, a SwiGLU workgroup adds 1 to its 256-row block's counter (`gen_rot.last_arriver`); the block's last workgroup (old count & 31 = 31) rotates it in place. The counters are monotonic and never reset.

### Flag-chained dispatch

A stream dispatch ends with an ordering barrier: the next kernel cannot launch until the previous one has drained (~4 us per boundary). With decode.txt `chain` (the default, `tools/gen_chain.py`) every decode kernel instead waits on its producer's counter:

- `gen_chain.chain` adds a counter buffer and three i32 push constants (wait slot, target count, own slot) to a 1-D kernel. Wave 0 polls the wait slot (acquire, device scope) until `(count - target) >= 0` as i32, then a workgroup barrier releases the other waves. At the end, lane 0 of every wave adds 1 to its own slot (release, device scope).
- The host (`LoomDevice::ChainDispatch`) keeps a shadow of the 1024 monotonic counters: each dispatch waits on the previous dispatch's slot for its full count (workgroups x waves) and signals the next slot. Slot 0 is never signaled. kvappend and vappend share one slot (join): they write disjoint caches.
- Workgroups launch in order, so a waiting workgroup never blocks its producer. Every dispatch waits on the one before it, so the order is still linear (no write-after-read hazards).
- Any other stream op (copies, fills, graph launches) records the ordering barrier first (`LoomDevice::Order`).

Logits are byte-identical to the barriered step. On top of the chain:

- Peeled GEMV start (`emit_decode.PEEL`, `gen_gemv` peel): a chained GEMV runs its first K iteration before the loop, split by dataflow at a `// chain.wait` marker. The weight loads and their decode, which do not read x, issue before the wait, so the first weights arrive during the producer's tail. Band GEMVs have one marker per band branch.
- The rotation kernels load their constants (s1, awq, s2, the norm weight) before the wait and x after it, and divide x by awq with a reciprocal and Markstein's three fmas (the IEEE quotient for every normal awq) instead of the IEEE divide.
- DeltaNet state stores (decode.txt `latestate`): `deltanet_conv` stores its updated state after its output, past a `// chain.signal` marker, with gfx11 DLC (no MALL allocation, HRX patch 0045), so the 3 MiB per layer do not get flushed out of the MALL during the next GEMV's weight stream. `LoomDecoder::Step` starts each step with `Order()`: the next token's DeltaNet reads the state after it.
- Greedy argmax in the output GEMV (decode.txt `amax`, `gen_gemv.amax_tail`): each workgroup reduces its rows' keys (order-preserving bits of the logit, then the inverted row, as a u64; NaN and +-inf skipped, -0 as +0) through LDS and takes one atomic max; the last workgroup by count stores the row as the token and resets the key. Ties keep the lowest index. Sampled steps run the output GEMV without it and then `gen_sample`.

### GEMV design

`tools/gen_gemv.py`, one kernel per (kind, rows, K), MQ4 rows (unsigned codes: nibble xor 8 with bias 8, one scale per 128).

- Decode on 32-bit words (`vector<4xi32>`), never on `vector<16xi8>`, which Loom lowers byte by byte.
- Two rows per wave share each x load; 4 waves per workgroup (R x W = 2 x 4, decode.txt `rw`).
- `pipeline(2)` read-ahead on the block loop; butterfly reduce over the wave.
- Kinds: `plain`, `resid` (y += W x, the residual add fused), `swiglu` (gate and up in one kernel), and bands (`gen_bands`, several output matrices of one input in one dispatch).

The GEMVs stream at ~237.6 GB/s of the 240 GB/s peak.

### Decode attention

`tools/gen_decode_attn.py`, split-K over pages:

- `part`: one workgroup per (KV head, 256-key page), with the six query heads of the GQA group. Scores (thread = key), per-head block max and sum, p in LDS, then P.V (thread = dim). Read-ahead depth 3 on the K rows and 2 on the V^T tiles. Grid order (head, page) so the four KV heads of a page read together.
- The last page is partial (keys up to `pos` only). Waves whose 32 keys are all past `pos` skip the scores loop, and P.V runs tiles 8-15 only when a visible key is there. Both guards sit outside the pipelined loops: a branch inside one makes every join wait for all loads (`vmcnt(0)`). The output is bit-identical. `part_q` sits at 256 VGPRs and spills with either guard, so it runs every key.
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

### Prefill set: `tools/emit_prefill_pp.py <model> <dir> [B]`

- One GEMM HAL per (kind, shape), the fixed kernels (`embed.hal`, `normq_5120.hal`, `rotq_*.hal`, `convkq*.hal`, `rowsplit_*.hal` = DeltaNet, `pnq_*.hal`, `prepab*.hal`, `rope.hal`, `wmma.hal` = attention, `rmsnorm.hal`, `rot_head.hal`, `gemv.hal`, `argmax.hal`, ...), one rope / attention / KV-writer HAL per chunk, and `rot_signs.bin`.
- `dispatch.txt`, one line per HAL: `<hal> <tokens per workgroup> <row groups> <token tiles>`. Marker rows with zeros carry the set's features (`a4`, `rot`, `kv_paged`, the overlap markers above, `ctx <B> 0 <T>` for chunked sets, ...). The driver refuses a set without `rot` (a GGUF-era set).
- `tools/footprint_gate.py` runs for every GEMM before it is emitted and refuses a kernel whose declared operand footprint is larger than the buffer the driver binds.

### Decode set: `tools/emit_decode.py <model> <dir> [max_context]`

- `gv_<kind>_<fmts>_<M>_<K>.hal` and `gb_<fmts>_<Ms>_<K>.hal` (GEMVs and band GEMVs), `dattn_*` (attention), `drot_*`, `unpack`, `rope`, `deltanet_conv`, `embed`, `argmax`, `sample`, and `rot_signs.bin`.
- `decode.txt`: `ctx <T>`, `kv q <K bits> <V bits>` for quantized KV, `chain` (flag-chained kernels; sets without it run barriered), `rot` with `normrot`, `swrot`, `amax` (argmax in the output GEMV), `latestate`, `rw <kind> <R> <W>` (GEMV geometry), and `grid <hal> <workgroups> 0` for every GEMV kernel.
- To decode after a prefill, emit the decode set with the same context as the prefill pools and the same `YAH_KV`. The driver refuses a mismatch.

### From generator to HAL

1. A generator (`tools/gen_*.py`) prints Loom text, one kernel per `gen_*` function. A few kernels are still hand-written `.loom` files in `engine/gpu/loom/`; the emitters compile those with fixed configs.
2. `engine/gpu/loom/emit_hal.py <file.loom> <outdir> sym=value ...` replaces every `config.get` with its constant, drops the `config.decl`, and has `iree-run-loom --emit-only --emit-hal-executable` compile the kernel for gfx1151. The kernel is built for exactly that config. The HRX/Loom tree is `external/hrx` (`engine/gpu/loom/hrx_paths.py`; `YAH_LOOM_HOME` points it elsewhere for diagnosis).
3. The emitter copies the result into the set and writes `dispatch.txt` or `decode.txt`.

Production sets build with the pinned HRX/Loom plus `engine/hrx/patches` (see build-and-run.md, HRX patches). Any other compiler change is for diagnosis only until it is a patch there.
