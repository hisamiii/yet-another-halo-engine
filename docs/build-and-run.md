# Build, run, verify, profile

How to build the drivers, emit HAL sets, run prefill and decode, check correctness, profile, and measure without hanging the GPU or fooling yourself. Read the GPU safety section before you write or launch a new kernel.

## Prerequisites

- HRX / Loom: the revision pinned in `engine/hrx/PIN` plus the patches in `engine/hrx/patches/`. `engine/hrx/bootstrap.sh` clones it into `external/hrx` (gitignored; `YAH_HRX` overrides, and a symlink to an existing checkout works), checks out the pin, applies the patches and builds what the engine uses: `libhrx` and its HIP binding, `iree-run-loom`, `loom-compile`, `iree-profile`. `engine/hrx/bootstrap.sh --check` verifies the checkout; `engine/build_hrx.sh` refuses anything else. On this box `main` builds against `/home/q/Development/Projects/hrx-npu` (the pin plus every patch here, the NPU ones 0013-0029 included; `external/hrx` links there) and branch `npu` against `/home/q/Development/Projects/hrx` (that branch's patch set); point `external/hrx` at the one matching the branch you build. Without HRX's XDNA targets the build is GPU-only (`--npu` off).
- The official ROCm Core SDK 10.0.0 packages (runtime, sysdeps, llvm, base, profiler-base), extracted into one prefix (no system install) by `engine/hrx-env.sh --fetch` under `external/rocm10/root` (`YAH_ROCM` overrides). They hold both the GPU runtime and the toolchain the bootstrap builds HRX with; no system ROCm is needed. Do not use a TheRock nightly: HRX needs `hsa_amd_queue_create`, which only the 10.0.0 release exports. `engine/hrx-env.sh --check` verifies the install.
- `source engine/hrx-env.sh` in every shell that runs a GPU job. It only sets `LD_LIBRARY_PATH` and `IREE_HAL_AMDGPU_LIBHSA_PATH`. The Python emitters find the same paths through `engine/gpu/loom/hrx_paths.py`.
- The model: hipfire's MQ4G256 HFQ file of Qwen3.8 27B, `~/Downloads/qwen3.8-27b.mq4-xts` (15 GB, read into resident memory by every driver), or a llama.cpp GGUF of it (any of Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K .. Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, F16, BF16, F32 per tensor), whose projections are re-quantized to MQ4G256 at load (~22 s on 16 cores; architecture.md). Only a GGUF with the MTP layer (`blk.64.nextn.*`, e.g. unsloth's) can draft for speculative decoding. The HAL sets are emitted from the model they run with.
- Python 3. The emitters need only the standard library; the checkers and gate tools need numpy, and an OpenBLAS-backed one (a venv) is ~100x faster there than a reference-BLAS build.

### HRX patches

Each patch says at its top what it is for. Patches stack (a later one may edit an earlier one's lines): the bootstrap builds the pin with every patch applied in order and brings the checkout's patched files to it, and `--check` compares them; a patch that no longer fits the pin stops it. Raise the pin deliberately: bootstrap, rebuild, then check the emits and the GPU references.

| patch | effect |
|---|---|
| `0001-stream-dispatch-no-ordering-barrier` | `HRX_DISPATCH_FLAG_NO_ORDERING_BARRIER`: decode overlaps independent dispatches (`LoomDevice::NoBarrierNext`) |
| `0002-stream-profile-metadata` | `HRX_PROFILE_MODE=dispatch` records stream dispatches |
| `0003-profile-counters-mode` | `HRX_PROFILE_MODE=counters` with `HRX_PROFILE_COUNTERS`: per-dispatch PMC such as `SQ_BUSY_CYCLES`, plus the gfx11 GL2C counters |
| `0005-gfx1151-f16-wmma-operand-placement` | the f16 WMMA spreads its operands over the register banks (upstream does it for bf16 only): prefill GEMM cycles -2.6% |
| `0007-amdgpu-u24-address-multiplies` | address multiplies use full-rate `v_mul_u32_u24` when value facts prove 24-bit operands: prefill cycles -0.26%, bit-identical |
| `0010-loom-vopd-trans-window-lookahead` | VOPD pairs inside TRANS-result windows when a lookahead finds no read that relies on the distance: prefill attention and SwiGLU epilogues -0.1..-0.25% cycles, bit-identical |
| `0011-loom-low-licm` | loop-invariant code motion for low CFGs, only with `LOOM_EXP_LICM=1` (the emitter sets it for the fp16 and kv4 prefill attention): -1.0% at pp2048, -2.15% at 32K, bit-identical |
| `0012-loom-slice-alias-coalescing` | concats of consecutive slices fold, and slice coalescing ignores dead exact-storage aliases: attention loop 29 -> 21 moves, -0.8..-2.3% attention, bit-identical |
| `0013-xdna-npu-cascade-gemm-compiler` | the Loom AIE2P compiler pieces of the NPU cascade GEMM: memory-tile staging, replay and multicast |
| `0014-xdna-continuation-invocation` | the XDNA loader resolves an entry's control-only continuation (`iree_hal_amd_xdna_executable_query_continuation`) |
| `0015-graph-atomic-store-node` | `hrx_graph_add_atomic_store_node`: a 4 / 8 byte store in a graph (the GPU releases NPU jobs and clears their words) |
| `0016-loom-xdna-npu-side-gate` | an NPU job waits for a flag in memory itself, so the GPU and the NPU sync without the host |
| `0017-loom-xdna-gated-call-invocations` | the NPU-side gate for plans that are not streamed (one replay group) |
| `0018-loom-xdna-npu-decoder-column` | array descriptors and bindings for the decoder column (`gen_npu_gemm` dcol 2: weight rows decoded on the cores) |
| `0019-loom-xdna-npu-ffn-block` | array plan, binding (axis chains) and route changes for the NPU FFN block |
| `0020-loom-xdna-npu-fused-fill` | GEMM workers fill their own memory-tile panels, so weights can be decoded on the way in |
| `0021-loom-xdna-pack-low` | low bank packing of read-only leaf data and wide ring records (opt-in `LOOM_EXP_PACK_LOW=1`) |
| `0022-loom-xdna-intake` | intake channels (`constrain.intake`): a staged binding read in panels |
| `0023-loom-xdna-core-stream-dma` | core-to-core streams own no DMA |
| `0024-loom-xdna-core-stream-schedule` | scalar core stream transfers schedule like the cascade |
| `0025-loom-xdna-east-neighbor-intake-depths` | two array fixes for the gate placement that frees the gate column's fill (`gen_npu_gemm` gwest) |
| `0026-loom-xdna-leaf-sync-addresses` | two array fixes for leaf-synchronized channels (`gen_npu_gemm` ghelp) |
| `0027-loom-xdna-drain-lock-requests` | a setup fix for images of other layouts sharing a context (`LOOM_EXP_ROUTE_RESET`) |
| `0028-loom-xdna-deferred-conflicting-rodata` | the planner defers read-only data with bank conflicts (`conflicts([...])`) |
| `0029-loom-xdna-top-aligned-programs` | core programs end at the top of program memory, behind a 16-byte startup jump |
| `0030-loom-amdgpu-tied-fmac` | an f32 fma whose addend dies there becomes the tied `v_fmac_f32`, which VOPD pairs; the iu4 GEMM fold: 31 fewer VALU issue slots per phase, bit-identical |
| `0031-loom-amdgpu-matrix-bank-allocation` | register allocation also tries a first fit that sees instruction operand preferences (matrix operand banks), kept only without spills and without a lower residency tier; adds the gfx1151 iu4 WMMA qualification; bit-identical |
| `0032-loom-kernel-priority` | `kernel.priority {level = N}` lowers to `s_setprio` as an ordered execution barrier; used around the iu4 GEMM fold; bit-identical |
| `0033-loom-amdgpu-iu4-inline-seed-subrev` | an iu4 WMMA accumulator splat of the bits of 1/(2 pi) selects the inline-source form; `x - c` with an inline `c` selects `v_subrev_f32`, which VOPD pairs; bit-identical |
| `0034-loom-amdgpu-inline-accumulator-bank-placement` | the inline-accumulator iu4 WMMA forms get their own gfx1151 placement preference (A and B in different VGPR banks); bit-identical |
| `0035-loom-amdgpu-valu-source-bank-placement` | VALU instructions prefer pairwise different source banks (two reads of one bank issue in 2.5 cycles): GEMM fold bank conflicts 21 -> 4, pp2048 GEMM cycles -0.7%, bit-identical |
| `0036-loom-amdgpu-pack-16bit-pairs-with-perm` | two 16-bit lanes pack with one `v_perm_b32` instead of 3 VALU: iu4 gu epilogue 48 -> 16 packing VALU, bit-identical |
| `0037-loom-wave-uniform-branches-with-matched-barriers` | workgroup barriers under a subgroup-uniform branch are accepted when every region runs the same barrier sequence: one wave runs its own loop (`gen_gdn_ss`'s solver wave) |
| `0038-loom-loop-carried-frame-placement` | loop-carried storage frames are placed together, so lane-decomposed accumulators keep one location around the backedge (`gen_gdn_ss` row loop 100 -> 33 `v_mov`) |
| `0039-loom-amdgpu-repack-reads-constructing-lanes` | a result -> rhs b16 fragment repack reads the constructing lanes directly: 8 fewer VALU per repacked tile, bit-identical |
| `0040-loom-amdgpu-b16-store-takes-low-half-def` | a 16-bit store of a low-half definition skips the `v_bfe_u32` that cleared the upper half: 168 VALU fewer in `gen_gdn_ss`, bit-identical |
| `0041-loom-amdgpu-cross-block-memory-waits-use-alias-summaries` | cross-block memory waits use the source's alias summaries (no `vscnt 0` drain for noalias buffers), and `ds_swizzle` touches no memory |
| `0043-loom-amdgpu-xor16-shuffle-via-permlanex16` | wave32 xor-16 shuffles lower to `v_permlanex16_b32` instead of `ds_swizzle`: attention softmax exchanges -1.1..-3%, bit-identical |
| `0045-loom-amdgpu-gfx11-non-temporal-cache-policy` | gfx11 non-temporal hints encode as GLC / SLC / DLC; decode's DeltaNet state store skips the MALL (+18 us per DeltaNet layer before) |

These are upstream candidates; do not push them to HRX without the owner's agreement.

## Build

```
engine/build_hrx.sh
```

It builds `libyah_core`, the core tools (`yah-tokenize`, `yah-dump`, `yah-weights`), the drivers `loom_forward_pp`, `loom_decode`, `hal_run`, the server `yah_server`, and the CMake targets that need no GPU (`yah_server_fake`, the chat client `yah_chat`, `yah_serve_test`) into `engine/build/`. It compiles with clang (`CXX=` overrides), `-O3 -march=native -ffp-contract=off`. Always use this script: `cmake --build engine/build` does not build the drivers and prints nothing, and a stale driver with a new HAL set launches the wrong grid. `gpu_run.sh` refuses a driver binary older than its source.

## Emit HAL sets

Emitters run on the CPU and take a few minutes. Paths below are relative to `engine/gpu/loom`.

| set | command |
|---|---|
| prefill, one pass of B tokens (A4) | `python3 tools/emit_prefill_pp.py <model> <dir> 2048` (B a multiple of 256; int4 activations on the iu4 WMMA) |
| prefill, W4A16 (the quality mode) | add `YAH_A16=1` (f16 activations, the tile GEMM) |
| prefill, chunked, context T | `YAH_CTX=32768 python3 tools/emit_prefill_pp.py <model> <dir> 2048` (chunk 2048, pools for 32768 tokens; T a multiple of B) |
| prefill with kv8a16 / kv4a16 | add `YAH_KV=kv8` or `YAH_KV=kv4` (or mixed `k8v4`, `k4v8`; `k8` / `v4` alone quantize one side, prefill only) |
| prefill with block-sparse attention | add `YAH_FPS_FROM=<P>` to a chunked kv4 set: chunks from token P on run FlashPrefill (architecture.md) |
| decode | `python3 tools/emit_decode.py <model> <dir> <max_context>` (multiple of 256, default 4096) |
| decode after a prefill | same, with `max_context` = the prefill set's context and the same `YAH_KV` |

The emitters read the model's tensor table from `engine/build/yah-dump --table` (headers only, no weights), so build first. The decode set also holds the 4-token verify GEMVs (`<gemv>_t4`) and, for a model with the MTP layer, its kernels: the set serves speculative decoding (Serving).

`engine/build_hrx.sh <model> [dir]` also emits a decode set (default `engine/hal`), which the end-to-end gates use.

Big sets belong in `~/Development/Scratch` (HAL sets under `~/Development/Scratch/hal/`), not in the repo.

## Run

Every GPU job goes through `engine/run/gpu_run.sh <tag> -- <command>` (see GPU safety). Source `engine/hrx-env.sh` first.

Prefill:

```
engine/run/gpu_run.sh pp -- engine/build/loom_forward_pp <model> <set> <out-prefix> 2048 <ids-file>
```

- `<ids-file>`: whitespace-separated token ids. A short file is repeated to the token count. Make one with `engine/build/yah-tokenize <model> --stdin`.
- The token count must equal the set's B, or for a chunked set be a multiple of the chunk up to T.
- Prints `layers_ms=` (the 64-layer loop, wall clock) and `argmax=`. Writes `<prefix>.logits` (last token, f32) and `<prefix>.hidden` (final hidden, f32, token-major).

Prefill then decode:

```
engine/run/gpu_run.sh gen -- engine/build/loom_forward_pp <model> <chunked set> <out-prefix> 8192 <ids-file> --gen 64 --decode <decode set>
```

- The prefill set must be paged (the default) and leave room: prompt + generated tokens <= T.
- Prints `generated_ids=` and `decode_ms=` / `decode_tok_s=` (mean over the generated steps).

Standalone decode (the prompt goes through the decode path one token at a time):

```
engine/run/gpu_run.sh dec -- engine/build/loom_decode <model> <decode set> --ids "760 6511 314 9338 369" --gen 64
```

`--logits FILE` appends every step's logits (f32) for an external KL check.

## Serving

`engine/build/yah_server` serves the model over the OpenAI Responses and Chat Completions APIs. It runs one generation at a time; other requests wait. The engine behind it (`engine/model/engine.hpp`) needs a chunked, paged prefill set and a decode set with the same context and KV format. `--ctx` (default 32768, a multiple of 2048) and `--kv` (default `kv8`; `fp16`, `kv4`, `k8v4`, `k4v8`) pick them: a release bundle's `hal/<ctx>-<kv>/`, or in a source tree `~/.cache/yah/hal/<ctx>-<kv>/` (`XDG_CACHE_HOME`), emitted there on first use (~20 s). The prefill sets are the same for the HFQ and any GGUF; a decode set emitted from a model with the MTP layer adds its kernels (`decode-mtp/`) and serves either. `--prefill` / `--decode` name sets instead (an NPU set: `tools/emit_npu_ffn.py` on a prefill set, made for one model file).

```
source engine/hrx-env.sh
engine/run/gpu_run.sh serve -- engine/build/yah_server --model <model> [--ctx 32768] [--kv kv8] [--host 127.0.0.1] [--port 8080] \
  [--draft mtp [--draft-len N]] [--npu [--npu-down <pack>]] [--no-reuse] [--prefill <set> --decode <set>]
engine/build/yah_server --model <model> --fake    # canned replies, CPU only: for clients and API tests
```

Before it loads anything the server estimates the memory it will take (the weights from the tensor table, the context's KV pools and scratch, a measured fixed part; `engine/model/memory_plan.hpp`) and refuses to start when that and a 1.5 GB reserve exceed `MemAvailable`: on the APU the GPU's buffers are system RAM, and an engine that does not fit makes the OOM killer end other programs. At 32K kv8 with MTP it takes ~20.5 GB (15.2 GB of weights for the Q4_K_M GGUF), at 8K ~19.3 GB. The log's `loaded in` line gives the load time and the GPU buffers.

A prompt runs as prefill chunks of 2048 tokens. When it ends inside a chunk, the last chunk is a partial one: its GEMMs run only the 256-token tiles that hold real tokens, so it costs about 0.25 s plus 0.36 s per 256 tokens (an 18-token prompt: 0.63 s to the first token). Tails of a few tokens go through decode steps instead (about 62 ms per token). Decode runs two steps ahead of the host: each step picks its token on the GPU (argmax, or for `temperature` > 0 `yah_dec_sample`: softmax at T cut by top-k, then top-p, then min-p, an exact draw by Gumbel-max; 0.84 ms with top_k 20, 1.0 ms with top-k off) into the token stream the next step reads, and the host only reads the tokens (a host-mapped copy) to stream them and stop. A given seed draws other tokens than the earlier host sampler did; the distribution is the same. With penalties, `yah_dec_penalty` first adjusts the logits from the last `repeat_last_n` tokens of the stream (prompt included, up to 1024) the way llama.cpp does, and the argmax then runs as its own kernel. `loom_forward_pp` takes any token count the same way.

Multi-turn chats reuse the cache. The engine keeps the token sequence of the last request (prompt and output) and two snapshots of the recurrent state (conv rings and DeltaNet states, 159 MB each): one at the last 2048-token chunk boundary 32 tokens short of the prompt's end, one at the last boundary the decode crossed. A prompt that starts with that sequence resumes at the deepest snapshot inside the shared prefix: the KV rows below it are still in the pools, so only the rest runs (a 9.3K-token follow-up: 1.7 s to the first token instead of 10 s). The first slot serves the usual follow-up, whose history ends at the old assistant header (thinking turns drop their reasoning there); the second serves a history that keeps the whole output. From the first slot the result is bit-identical to a fresh prefill; from the second it is the continued run's (the tokens past the old prompt were decode steps, not a prefill chunk, so a greedy reply can take another near-tie). `usage.prompt_tokens_details.cached_tokens` (`input_tokens_details` for Responses) reports the reused tokens. `yah_server --no-reuse` turns it off.

Speculative decoding: `--draft mtp` drafts with the model's MTP layer (a GGUF that has it) and verifies the draft and the next token in one pass of the decode set's 4-token GEMVs; every other kernel runs once per token as in a step, so a greedy output is the same token for token as without drafting. A rejected token rolls back the conv rings, the DeltaNet states (one ring slot per verified token) and the open KV tile. The draft length follows the acceptance: each round picks the length with the most expected tokens per ms (from a decayed acceptance rate and the measured round times), down to plain steps where drafting does not pay. `--draft-len N` fixes it. On the Q4_K_M GGUF at 32K kv8 (`spec_engine`): code 15.9 -> 24.3 t/s, a long factual prompt 13.6 -> 22.0, free prose 14.4 -> 13.7 (the acceptance is low there). Sampled requests (temperature > 0) decode without drafts. The request log line carries `accepted=<accepted>/<drafted>`.

The NPU: `--npu` runs the prefill set's FFN block on the XDNA2 NPU (a set with `tools/emit_npu_ffn.py` run on it; its down pack `<prefill>/npu_down.bin` from `tools/npu_down_pack.py`, or `--npu-down`). It needs an unlimited memlock limit and falls back to the GPU alone without what it needs. Long prompts gain (pp8500: +11%); prompts shorter than two chunks do not (the pair schedule needs two).

For quick tests, `engine/build/yah_chat` is a terminal chat client: it streams the reply (reasoning dimmed), keeps the conversation, and prints token counts, time to first token and tok/s after each reply. Commands: `/reset`, `/effort E`, `/temp T`, `/max N`, `/system TEXT`, `/quit`; Ctrl-C stops a reply.

```
engine/build/yah_chat --url http://127.0.0.1:8080 [--effort none|low|medium|high] [--temperature T] [--max N] [--system TEXT]
```

It logs one line per request to stderr: id, prompt and output tokens, prefill ms, decode tok/s, finish reason (`cancelled` when the client left).

```
curl -s localhost:8080/v1/responses -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "input": "Why is the sky blue?", "reasoning": {"effort": "low"}, "max_output_tokens": 2048}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
r = client.responses.create(model="qwen", instructions="Be brief.", input="Why is the sky blue?")
print(r.output_text)
for event in client.responses.create(model="qwen", input="Hi", stream=True):
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
for chunk in client.chat.completions.create(model="qwen", messages=[{"role": "user", "content": "Hi"}], stream=True,
                                            reasoning_effort="none"):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

| | |
|---|---|
| endpoints | `POST /v1/responses` and `POST /v1/chat/completions` (plain and `stream: true` SSE), `GET /v1/models`, `GET /health` |
| input | a string, or a list of `system` / `developer` / `user` / `assistant` messages with string or text-part content; `reasoning`, `function_call` and `function_call_output` items from earlier turns; `instructions` |
| reasoning | `reasoning.effort`: `none` / `minimal` (thinking off), `low`, `medium`, `high` (default, the template's `xhigh`). The thinking text comes back as a `reasoning` output item with `reasoning_text` content |
| tools | `tools` (function tools) with `tool_choice` `auto` or `none`. The model's calls come back as `function_call` output items (streamed with `response.function_call_arguments.delta` / `.done`, each call whole), their arguments typed by the tool's schema |
| sampling | `temperature` (default 0.6), `top_p` (default 0.95), top-k 20, `max_output_tokens` (default: the rest of the context; prompt + max must fit) |
| chat completions | `messages` as above; `max_completion_tokens` or `max_tokens`; `temperature`, `top_p`, `top_k` (default 20; 0 or -1: off), `min_p`, `presence_penalty`, `frequency_penalty` (-2..2), `repetition_penalty`, `repeat_last_n` (default 64, up to 1024), `seed`, `stop` (a string or up to 16), `stream` with `stream_options.include_usage`; `reasoning_effort` or `chat_template_kwargs.enable_thinking`; `ignore_eos` (benchmarks); `tools` (function tools) with `tool_choice` `auto` or `none`, and `assistant` `tool_calls` and `tool` messages in the history. Reasoning comes back as `reasoning_content`, and llama.cpp's `timings` object (prefill and decode ms and tokens/s) comes with the response or the stream's last chunk. The model's calls come back as `tool_calls` (finish reason `tool_calls`; streamed, each call whole in one chunk), their arguments typed by the tool's schema. `n` must be 1 |
| accepted and ignored | `store`, `metadata` (echoed), `user`, `service_tier`, `parallel_tool_calls`, and default values of `tool_choice`, `text`, `truncation` (Responses) |
| not supported (400) | tools other than functions, `tool_choice` `required` or a named function, `previous_response_id` and stored responses, images and files, structured output, logprobs |

All system and developer messages merge into one system message at the start. The chat template is a C++ port of the model's Jinja template for this subset.

Tests (no GPU):

```
engine/build/yah_serve_test <model> engine/build/yah_server_fake   # chat template golden, then --fake over HTTP
```

GPU tests of speculative decoding: `spec_check` runs verify rounds with one wrong draft per round at a rotating index against plain steps (every verified logit bit-identical, `--time` adds the step / verify / draft costs); `spec_engine` runs prompts through the engine greedy with and without drafts and requires the same tokens:

```
engine/run/gpu_run.sh spec -- engine/build/spec_check <model> <decode set> 96 [--time]
engine/run/gpu_run.sh spec -- engine/build/spec_engine <GGUF with MTP> <prefill set> <decode set> [--draft-len N] [--npu]
```

GPU test of the reuse (bit-exact continuation from the decode slot, reused = fresh from the prefill slot; ~2 min):

```
engine/run/gpu_run.sh reuse -- engine/build/engine_reuse <model> <prefill set> <decode set>
```

`yah_server_fake` is `yah_server` built without the engine (`YAH_SERVE_FAKE_ONLY`): only `--fake`, no libhrx. The default server is `yah_server` next to the test, which links libhrx: source `engine/hrx-env.sh` first. `--fake` reads only the tokenizer, so `<model>` can be the tokenizer-only cut `python3 engine/tests/make_tokenizer_fixture.py <model path or URL> <out.hfq>` makes (~14 MB; from a URL it fetches only the metadata ranges). `engine/tests/verify_tokenizer.py --model` takes it too. The CPU CI (`.github/workflows/cpu.yml`) runs both on that cut once the repository variable `YAH_MODEL_URL` names the model's download URL; without it the job only builds.

It checks the C++ chat template byte for byte against `engine/serve/testdata/chat_template_cases.json` and `chat_template_tool_cases.json`, then starts `yah_server --fake` on a free port and checks the response object and stream event shapes (the required fields of the openai SDK models), event order, deltas, `max_output_tokens`, the 400 error objects and client disconnect. The first fixture was frozen from jinja2 on 2026-10-02 against the GGUF's template; regenerating it is not supported in-tree. The tool fixture comes from `engine/tests/make_tool_template_cases.py <model>` (jinja2, the model file's own template; the HFQ's and the unsloth GGUF's render these cases the same).

## Release

A release is one tarball, `yah-<version>-gfx1151.tar.xz`. It holds `yah_server` and `yah_chat`, every library they load (libhrx, ICU, the ROCm 10 HSA runtime and its sysdeps; RPATH `$ORIGIN/../lib`), the 32K kv8 HAL sets and the licenses. It needs glibc 2.39 and no ROCm or root. `yah_server` takes the HAL sets from the bundle's `hal/` and the HSA runtime from its `lib/` when the options and `IREE_HAL_AMDGPU_LIBHSA_PATH` don't name others; `engine/release/README.md` is the bundle's README.

1. CI (`.github/workflows/release.yml`, on a `v*` tag or by hand with a version) runs `engine/release/build.sh` on Ubuntu 24.04. It builds libhrx at the pin, then the binaries with g++-14, `-march=znver5` and static libstdc++, and bundles them without HAL sets (artifact `yah-<version>-gfx1151-nohal`).
2. On a gfx1151 box: emit the serve sets (Serving above), add them, run the smoke test, pack:

```
gh run download <run id> -n yah-<version>-gfx1151-nohal && tar -xJf yah-<version>-gfx1151-nohal.tar.xz
python3 engine/release/bundle.py hal yah-<version>-gfx1151 <prefill set> <decode set>
python3 engine/release/smoke.py yah-<version>-gfx1151 <model>     # ~2 min on the GPU
python3 engine/release/bundle.py pack yah-<version>-gfx1151       # checks it is self-contained, then the tarball
```

`smoke.py` starts the bundle's server with only `HOME` and `PATH` set: chat completions plain and streamed, Responses, sampling with penalties, cache reuse on a follow-up. `bundle.py pack` checks every ELF in the bundle resolves its libraries inside it with a clean environment and prints the glibc it needs.

## Correctness

| tool | what it checks | when |
|---|---|---|
| hidden md5 of `<prefix>.hidden` | bit identity of the whole forward: record the reference of the set before an exact rewrite (`ids2048` / `ids8192` prompts) | every exact rewrite |
| `cmp` of every emitted HAL | byte identity of an emit | every refactor or cleanup of a generator |
| `tools/dattn_check.py <model> <work> [T] [pos]` | fp16 decode attention against numpy, scrambled page table | decode attention changes |
| `tools/sample_check.py <model> <work> [draws]` | the decode sampler against a numpy model of it (token for token), the model against the exact top-p distribution | sampler changes |
| `tools/dattn_q_check.py <model> <work> 8\|4\|kb,vb [T] [pos]` | quantized decode attention and the appends against numpy models of the formats | quantized KV changes |
| `engine/run/gate/kld_ref.py <model> <set> <ref.bin> <workdir> [chunks]` | KL vs the BF16 model: hipfire's HFKLDR reference (24 wikitext-2 chunks of 2048, top-256 log-probs of positions 1024..2046; `~/Development/Scratch/w4/kld/qwen3.8-27b.ref_wt2.bin`), one pp2048 per chunk, scored as hipfire's `eval_hipfire` scores; prints mean / p99 / p99.9 / max KL, top-1, PPL | HFQ model numerics |
| `engine/run/gate/` (`gate_run.sh`, `accgate2.py`) | tiered accuracy gate on wikitext windows, all-position logits (`--logits-from`): T1 rounding level, T2 quantization level. See `engine/run/gate/README.md` | any numerics change |
| `engine/run/kvq/` (`run_rowstats.sh`, `gate2.py`, `needle_score.py`) | KV format quality at 32K: long-document KL / dPPL with bootstrap CIs and multi-key retrieval, from `--rowstats` row stats of an fp16-KV reference set and a `YAH_KV` candidate set. See `engine/run/kvq/README.md` | KV format work |
| `engine/tests/m0_gate.sh <model> [set]` | greedy next token on 3 fixed prompts | after any change, cheap |
| `engine/tests/generate_gate.sh <model> [set]` | 20 greedy tokens equal the reference | after any change, cheap |
| `engine/tests/npu_gate.py <model> <prefill set> [down pack]` | the NPU FFN block runs and matches the GPU alone on a pp4096 pair schedule: top-1 and KL of the last 64 positions, the last chunk's hidden states (loose: catches a hang, a fallback or wrong NPU output; ~25 s) | NPU, prefill or HRX changes |

`hal_run` (`engine/build/hal_run`) dispatches one HAL once with bindings from model tensors or files and writes outputs to files. It refuses any binding smaller than the size you declare. The checkers use it; use it for any new kernel oracle.

Gate notes:

- T1's `kl_mean` saturates at ~4e-7 for any attention change that is not bit-exact (a single rounding change gives the same). Judge such changes on `kl_p999`, flips and PPL.
- The `ids2048` / `ids8192` timing prompts repeat one sentence (10 distinct tokens). They are fine for md5 and timing, useless for accuracy.
- Run a standalone kernel check on inputs from several layers and documents, not only layer 0: an alpha underflow in DeltaNet showed up only at 8K / 32K deep in the model.

## Profiling

| question | tool |
|---|---|
| device time per dispatch in the real pipeline | `HRX_PROFILE_FILE=p.irpf HRX_PROFILE_MODE=dispatch <driver ...>`, then `external/hrx/build/cmake/runtime/src/iree/tools/iree-profile/iree-profile dispatch --dispatch_events --format=jsonl p.irpf`. Costs ~1%. |
| cycles per dispatch (clock-free) | `HRX_PROFILE_MODE=counters HRX_PROFILE_COUNTERS=SQ_BUSY_CYCLES`, then `iree-profile counter --format=jsonl --counter_samples`. Needs `libhsa-amd-aqlprofile64`, which `engine/hrx-env.sh` puts on the path. Inflates dispatch gaps ~4x. |
| registers, spills, residency, schedule | `loom-compile k.loom --root=@k --target=amdgpu:gfx1151 --format=amdgpu-hsaco --output=k.hsaco --compile-report=details --compile-report-output=k.json`, then `PYTHONPATH=external/hrx/loom/py python3 -m loom.tools.compile_report show\|suggest\|diff k.json`. Run `suggest --include-experimental` before hand-tuning. `allocation.materialized_spill_*` shows real spills even when `spill_count` is 0. |
| instruction counts | `llvm-objdump -d --mcpu=gfx1151` on the hsaco (a `.hal` embeds the ELF from the first `\x7fELF`). Run it with `env -u LD_LIBRARY_PATH`. |
| PMC, occupancy, per-wave ATT | rocprofv3 cannot see HRX dispatches. The `loomhip` harness that ran one Loom hsaco through HIP under `rocprofv3 --pmc` / `--att` is on tag `pre-cleanup-2026-10-20`. |
| clocks, temperature, throttling | gpu_metrics v3 (`/sys/class/drm/card1/device/gpu_metrics`), the ryzen_smu PM table, `z13ctl status` for the APU temperature. See hardware.md. |

- ATT stretches the tile GEMM ~1.5x. Use it for shares and attribution, never for absolute idle time; check resource claims with PMC.
- After an ablation, check that the ablated kernel's ISA still contains the work: the compiler deletes loads whose data is never used.
- PC sampling and RGP for compute do not work on this box.

## GPU safety

A bad dispatch on this box does not fault. A read past an allocation reaches unmapped VA, the shader hangs with no page fault, the gfx ring times out, MES fails to reset, and the machine reboots.

Rules:

1. Launch exactly the grid the kernel was compiled for. Loom assumes `workgroup.id` < the launch-config grid and drops index clamps it proves redundant, so a larger grid reads out of bounds. The drivers take grids from `dispatch.txt` / `decode.txt` and refuse others; `gen_gemv.rows_per_wg()` is the single source for GEMV grids. Never compute a grid by hand in a harness.
2. LDS data written by other threads needs a workgroup barrier before the first read, prologues included. (A page table in LDS read before its barrier hung the GPU.)
3. Clamp or validate every index read from memory before it addresses a buffer. The host validates page tables; cache writers clamp page indices into the pool.
4. Clamp a vector load by its own width: `min(addr, N - width)`. A clamp copied from a 4-wide load onto a 2-wide load shifts the last element.
5. Never bind a buffer smaller than the kernel's declared footprint. The prefill emitter runs `tools/footprint_gate.py`; `hal_run` checks declared sizes; `tools/loom_preflight.py` checks the compile report's footprint against the bindings.
6. Rebuild drivers with `engine/build_hrx.sh` after every source change, and do not edit `engine/run/*.cc` during a measuring round (`gpu_run.sh` will then refuse the remaining runs).
7. Run every GPU job through `engine/run/gpu_run.sh`. It snapshots and follows the kernel log into `~/Development/Scratch/gpu-logs/gpu-<tag>-<time>.dmesg.log` and flags timeout / reset lines.
8. Never free memory the GPU may still touch: a GPU read of a freed page hangs the shader and MES.

After a hang and reboot, read the previous boot's kernel log:

```
doas journalctl -k -b -1 --no-pager | grep -iE 'amdgpu|ring .*timeout|reset|MES'
```

`/tmp` is wiped by the reboot; keep logs elsewhere.

## Measuring

End to end, as a client sees it: `engine/bench/bench.py --url http://127.0.0.1:8080 [--lengths 512,2048,8192,16384,30000] [--gen 128] [--json out.json]`. One greedy request per prompt length (random nonce first, so no cache serves it; `ignore_eos`, so every run decodes `--gen` tokens), prefill and decode tokens/s from the server's `timings`. It follows the rules below: it waits for Tctl <= 55 C, then 15 s or 30 s, before each run. Pointed at llama.cpp's `llama-server` it measures that the same way.

- One round per candidate. No interleaving, no ABBA, no repeats, no clock pinning.
- Before each timed run, cool the APU to <= 55 C (`z13ctl status`), then wait 1 s (single kernels), 15 s (pp2048) or 30 s (pp8192 and longer).
- For pp8192 and longer, report `SQ_BUSY_CYCLES` next to ms. A run can take a mid-run clock step worth ~7% of wall time; the first pp8192 run after an idle period reads fast. Cycles per dispatch and per device tick show both.
- Compare per-kernel rows, not only totals: run-to-run noise on a single kernel row is up to ~5% in cycles.
- Use real weights and real activations for standalone kernels: constant fills run the same cycles at a higher clock.
- Benchmark long-context decode attention at 32K: smaller standalone cases fit in the 32 MB MALL and read flattered bandwidth.
- A kernel change must be bit-identical (hidden md5) or pass its oracle check before it is timed. A refactor must keep every emitted HAL byte-identical.
- Before an A/B, `cmp` the two variants' HALs: a script that loses its env overrides measures the same kernel twice.
- The tctl limit set with `ryzenadj` resets on reboot. Re-apply it before timing and compare only runs with the same setting.
- When an optimization loses, find out why before dropping it, and record one line in results.md.
- A timing ablation that skips a kernel changes the data the rest of the run computes on, and GEMM speed depends on the data (power under the thermal cap): skipping the qkv / gate unpacks ran every later GEMM ~4.6% faster, pp2048 -48 ms of which ~11 ms was the unpacks. Measure a kernel's cost with the output unchanged (run it a second time into a dummy buffer) and require the reference md5; a skip that changes the md5 is an upper bound.

## Further reading

- HRX demos: [DEVELOPMENT.md](https://github.com/ROCm/hrx-demos/blob/main/DEVELOPMENT.md), [AGENTS.md](https://github.com/ROCm/hrx-demos/blob/main/AGENTS.md); Loom docs in `external/hrx/loom/docs/`.
- [gfx950 Gluon tutorials](https://github.com/ROCm/gfx950-gluon-tutorials), [HIP performance guidelines](https://rocmdocs.amd.com/projects/HIP/en/develop/how-to/performance_guidelines.html).
