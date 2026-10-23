# Build, run, verify, profile

How to build the drivers, emit HAL sets, run prefill and decode, check correctness, profile, and measure without hanging the GPU or fooling yourself. Read the GPU safety section before you write or launch a new kernel.

## Prerequisites

- HRX / Loom: the revision pinned in `engine/hrx/PIN` plus the patches in `engine/hrx/patches/`. `engine/hrx/bootstrap.sh` clones it into `external/hrx` (gitignored; `YAH_HRX` overrides, and a symlink to an existing checkout works), checks out the pin, applies the patches and builds what the engine uses: `libhrx` and its HIP binding, `iree-run-loom`, `loom-compile`, `iree-profile`. `engine/hrx/bootstrap.sh --check` verifies the checkout; `engine/build_hrx.sh` refuses anything else. On this box `main` builds against `/home/q/hrx-gpu` (the pin plus main's patches; `external/hrx` links there) and branch `npu` against `/home/q/hrx` (the pin plus every patch of that branch, the NPU ones 0013-0029 included); point `external/hrx` at the one matching the branch you build. hrx-gpu was moved there from its build location, so rebuilding HRX itself needs a fresh configure (`engine/hrx/bootstrap.sh`).
- The official ROCm Core SDK 10.0.0 runtime packages, extracted (no system install) by `engine/hrx-env.sh --fetch` into `external/rocm10` (`YAH_ROCM` overrides). Do not use a TheRock nightly: HRX needs `hsa_amd_queue_create`, which only the 10.0.0 release exports. `engine/hrx-env.sh --check` verifies the install.
- `source engine/hrx-env.sh` in every shell that runs a GPU job. It only sets `LD_LIBRARY_PATH` and `IREE_HAL_AMDGPU_LIBHSA_PATH`; `/opt/rocm` stays usable. The Python emitters find the same paths through `engine/gpu/loom/hrx_paths.py`.
- The model: hipfire's MQ4G256 HFQ file of Qwen3.8 27B, `~/Downloads/qwen3.8-27b.mq4-xts` (15 GB, read into resident memory by every driver). The GGUF / IQ4_XS path is gone; tag `pre-cleanup-2026-10-20` has it.
- Python 3 with numpy for the emitters and checkers. Use an OpenBLAS numpy (`/home/q/yah-scratch/venv/bin/python`) for the gate tools; the system numpy is ~100x slower there.

### HRX patches

Each patch says at its top what it is for. Patches stack (a later one may edit an earlier one's lines): the bootstrap builds the pin with every patch applied in order and brings the checkout's patched files to it, and `--check` compares them; a patch that no longer fits the pin stops it. Raise the pin deliberately: bootstrap, rebuild, then check the emits and the GPU references.

| patch | effect |
|---|---|
| `0001-stream-dispatch-no-ordering-barrier` | `HRX_DISPATCH_FLAG_NO_ORDERING_BARRIER`: decode overlaps independent dispatches (`LoomDevice::NoBarrierNext`) |
| `0002-stream-profile-metadata` | `HRX_PROFILE_MODE=dispatch` records stream dispatches |
| `0003-profile-counters-mode` | `HRX_PROFILE_MODE=counters` with `HRX_PROFILE_COUNTERS`: per-dispatch PMC such as `SQ_BUSY_CYCLES`, plus the gfx11 GL2C counters |
| `0004-loom-profile-function-filter` | `LOOM_PROFILE_FUNCTION=<glob>` for executable traces in the Loom HAL benchmark tool |
| `0005-gfx1151-f16-wmma-operand-placement` | the f16 WMMA spreads its operands over the register banks (upstream does it for bf16 only): prefill GEMM cycles -2.6% |
| `0006-dispatch-timestamps-callback` | dispatch device timestamps to an in-process callback (`hrx_device_profile_dispatches_*`); its user, the prefill calibration, is gone (tag `pre-cleanup-2026-10-20`) |
| `0007-amdgpu-u24-address-multiplies` | address multiplies use full-rate `v_mul_u32_u24` when value facts prove 24-bit operands: prefill cycles -0.26%, bit-identical |
| `0008-loom-concat-reservation-unavoidable-tier` | the allocator keeps a concat's result reservation when its residency-tier crossing is unavoidable, instead of placing one half alone and copying both into the tuple later; bit-identical |
| `0009-loom-omit-dead-fma-mix-seeds` | after allocation, drops the `v_mov_b32 0` that seeds an f16 pair built by `v_fma_mixlo` + `v_fma_mixhi`, keeping the schedule and registers; bit-identical |
| `0010-loom-vopd-trans-window-lookahead` | VOPD pairs inside TRANS-result windows when a lookahead finds no read that relies on the distance: prefill attention and SwiGLU epilogues -0.1..-0.25% cycles, bit-identical |
| `0011-loom-low-licm` | loop-invariant code motion for low CFGs, only with `LOOM_EXP_LICM=1` (the emitter sets it for the fp16 and kv4 prefill attention): -1.0% at pp2048, -2.15% at 32K, bit-identical |
| `0012-loom-slice-alias-coalescing` | concats of consecutive slices fold, and slice coalescing ignores dead exact-storage aliases: attention loop 29 -> 21 moves, -0.8..-2.3% attention, bit-identical |
| `0015-graph-atomic-store-node` | `hrx_graph_add_atomic_store_node`: a 4 / 8-byte store recorded in a graph's command buffer (no partition break); written for the NPU split (branch `npu`), unused on main |
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
| `0042-amdgpu-one-mib-command-buffer-blocks` | 1 MiB command-buffer blocks: a pp2048 chunk graph is one submission, no 0.5-0.7 ms idle at a block boundary |
| `0043-loom-amdgpu-xor16-shuffle-via-permlanex16` | wave32 xor-16 shuffles lower to `v_permlanex16_b32` instead of `ds_swizzle`: attention softmax exchanges -1.1..-3%, bit-identical |
| `0044-libhrx-graph-barrier-state-256` | the graph barrier planner tracks 256 commands since the last barrier (was 8), so completion-counter chains do not pay a full barrier for retired edges |
| `0045-loom-amdgpu-gfx11-non-temporal-cache-policy` | gfx11 non-temporal hints encode as GLC / SLC / DLC; decode's DeltaNet state store skips the MALL (+18 us per DeltaNet layer before) |

These are upstream candidates; do not push them to HRX without the owner's agreement.

## Build

```
engine/build_hrx.sh
```

It builds `libyah_core`, the core tools (`yah-tokenize`, `yah-dump`, `yah-weights`), the drivers `loom_forward_pp`, `loom_decode`, `hal_run`, the server `yah_server`, the chat client `yah_chat` and `yah_serve_test` into `engine/build/`. It compiles with clang (`CXX=` overrides), `-O3 -march=native -ffp-contract=off`. Always use this script: `cmake --build engine/build` does not build the drivers and prints nothing, and a stale driver with a new HAL set launches the wrong grid. `gpu_run.sh` refuses a driver binary older than its source.

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

The emitters read the model's tensor table from `engine/build/yah-dump --table` (headers only, no weights), so build first. `YAH_EMIT_KEEP=<dir>` keeps each specialized Loom source for `loom-compile --compile-report`.

`engine/build_hrx.sh <model> [dir]` also emits a decode set (default `engine/hal`), which the end-to-end gates use.

Big sets belong in `~/yah-scratch` or `/home/q/yah-hal-*`, not in the repo.

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
YAH_GEN=64 YAH_DECODE_HAL=<decode set> engine/run/gpu_run.sh gen -- engine/build/loom_forward_pp <model> <chunked set> <out-prefix> 8192 <ids-file>
```

- The prefill set must be paged (the default) and leave room: prompt + generated tokens <= T.
- Prints `generated_ids=` and `decode_ms=` / `decode_tok_s=` (mean over the generated steps).

Standalone decode (the prompt goes through the decode path one token at a time):

```
engine/run/gpu_run.sh dec -- engine/build/loom_decode <model> <decode set> --ids "760 6511 314 9338 369" --gen 64
```

`--logits FILE` appends every step's logits (f32) for an external KL check.

## Serving

`engine/build/yah_server` serves the model over the OpenAI Responses API. It runs one generation at a time; other requests wait. The engine behind it (`engine/model/engine.hpp`) needs a chunked, paged prefill set and a decode set with the same context and `YAH_KV`. The default is 32K context with kv8a16 KV:

```
cd engine/gpu/loom
YAH_CTX=32768 YAH_KV=kv8 python3 tools/emit_prefill_pp.py <model> /home/q/yah-hal-serve 2048
YAH_KV=kv8 python3 tools/emit_decode.py <model> /home/q/yah-hal-serve-dec 32768
cd -
source engine/hrx-env.sh
engine/run/gpu_run.sh serve -- engine/build/yah_server --model <model> --prefill /home/q/yah-hal-serve --decode /home/q/yah-hal-serve-dec [--host 127.0.0.1] [--port 8080]
engine/build/yah_server --model <model> --fake    # canned replies, CPU only: for clients and API tests
```

A prompt runs as prefill chunks of 2048 tokens. When it ends inside a chunk, the last chunk is a partial one: its GEMMs run only the 256-token tiles that hold real tokens, so it costs about 0.25 s plus 0.36 s per 256 tokens (an 18-token prompt: 0.63 s to the first token). Tails of a few tokens go through decode steps instead (about 62 ms per token). Decode runs two steps ahead of the host: each step picks its token on the GPU (argmax, or for `temperature` > 0 `yah_dec_sample`: softmax at T cut to the top-p nucleus, an exact draw by Gumbel-max, ~1 ms) into the token stream the next step reads, and the host only reads the tokens (a host-mapped copy) to stream them and stop. A given seed draws other tokens than the earlier host sampler did; the distribution is the same. `loom_forward_pp` takes any token count the same way.

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
```

| | |
|---|---|
| endpoints | `POST /v1/responses` (plain and `stream: true` SSE), `GET /v1/models`, `GET /health` |
| input | a string, or a list of `system` / `developer` / `user` / `assistant` messages with string or text-part content; `reasoning` items from earlier output; `instructions` |
| reasoning | `reasoning.effort`: `none` / `minimal` (thinking off), `low`, `medium`, `high` (default, the template's `xhigh`). The thinking text comes back as a `reasoning` output item with `reasoning_text` content |
| sampling | `temperature` (default 0.6), `top_p` (default 0.95), `max_output_tokens` (default: the rest of the context; prompt + max must fit) |
| accepted and ignored | `store`, `metadata` (echoed), `user`, `service_tier`, `parallel_tool_calls`, and default values of `tool_choice`, `text`, `truncation` |
| not supported (400) | tools, `previous_response_id` and stored responses, images and files, structured output, logprobs |

All system and developer messages merge into one system message at the start. The chat template is a C++ port of the model's Jinja template for this subset.

Tests (no GPU; source `engine/hrx-env.sh` first, `yah_server` links libhrx):

```
engine/build/yah_serve_test <model> [yah_server]   # chat template golden, then yah_server --fake over HTTP
```

It checks the C++ chat template byte for byte against `engine/serve/testdata/chat_template_cases.json`, then starts `yah_server --fake` on a free port and checks the response object and stream event shapes (the required fields of the openai SDK models), event order, deltas, `max_output_tokens`, the 400 error objects and client disconnect. The fixture was frozen from jinja2 on 2026-10-02 against the GGUF's template; regenerating it is not supported in-tree.

## Correctness

| tool | what it checks | when |
|---|---|---|
| hidden md5 of `<prefix>.hidden` | bit identity of the whole forward: record the reference of the set before an exact rewrite (`ids2048` / `ids8192` prompts) | every exact rewrite |
| `cmp` of every emitted HAL | byte identity of an emit | every refactor or cleanup of a generator |
| `tools/dattn_check.py <model> <work> [T] [pos]` | fp16 decode attention against numpy, scrambled page table | decode attention changes |
| `tools/sample_check.py <model> <work> [draws]` | the decode sampler against a numpy model of it (token for token), the model against the exact top-p distribution | sampler changes |
| `tools/dattn_q_check.py <model> <work> 8\|4\|kb,vb [T] [pos]` | quantized decode attention and the appends against numpy models of the formats | quantized KV changes |
| `engine/run/gate/kld_ref.py <model> <set> <ref.bin> <workdir> [chunks]` | KL vs the BF16 model: hipfire's HFKLDR reference (24 wikitext-2 chunks of 2048, top-256 log-probs of positions 1024..2046; `~/yah-scratch/w4/kld/qwen3.8-27b.ref_wt2.bin`), one pp2048 per chunk, scored as hipfire's `eval_hipfire` scores; prints mean / p99 / p99.9 / max KL, top-1, PPL | HFQ model numerics |
| `engine/run/gate/` (`gate_run.sh`, `accgate2.py`) | tiered accuracy gate on wikitext windows, all-position logits (`YAH_LOGITS_FROM`): T1 rounding level, T2 quantization level. See `engine/run/gate/README.md` | any numerics change |
| `engine/run/kvq/` (`run_rowstats.sh`, `gate2.py`, `needle_score.py`) | KV format quality at 32K: long-document KL / dPPL with bootstrap CIs and multi-key retrieval, from `YAH_ROWSTATS` row stats of an fp16-KV reference set and a `YAH_KV` candidate set. See `engine/run/kvq/README.md` | KV format work |
| `engine/tests/m0_gate.sh <model> [set]` | greedy next token on 3 fixed prompts | after any change, cheap |
| `engine/tests/generate_gate.sh <model> [set]` | 20 greedy tokens equal the reference | after any change, cheap |

`hal_run` (`engine/build/hal_run`) dispatches one HAL once with bindings from model tensors or files and writes outputs to files. It refuses any binding smaller than the size you declare. The checkers use it; use it for any new kernel oracle.

Gate notes:

- T1's `kl_mean` saturates at ~4e-7 for any attention change that is not bit-exact (a single rounding change gives the same). Judge such changes on `kl_p999`, flips and PPL.
- The `ids2048` / `ids8192` timing prompts repeat one sentence (10 distinct tokens). They are fine for md5 and timing, useless for accuracy.
- Run a standalone kernel check on inputs from several layers and documents, not only layer 0: an alpha underflow in DeltaNet showed up only at 8K / 32K deep in the model.

## Profiling

| question | tool |
|---|---|
| device time per dispatch in the real pipeline | `HRX_PROFILE_FILE=p.irpf HRX_PROFILE_MODE=dispatch <driver ...>`, then `external/hrx/build/cmake/runtime/src/iree/tools/iree-profile/iree-profile dispatch --dispatch_events --format=jsonl p.irpf`. Costs ~1%. |
| cycles per dispatch (clock-free) | `HRX_PROFILE_MODE=counters HRX_PROFILE_COUNTERS=SQ_BUSY_CYCLES`, then `iree-profile counter --format=jsonl --counter_samples`. Needs a TheRock `libhsa-amd-aqlprofile64` first on `LD_LIBRARY_PATH` (the copy used so far lived under `/var/lib/lemonade`, now removed; `/opt/rocm/lib` has one, not yet tried with HRX). Inflates dispatch gaps ~4x. |
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
7. Run every GPU job through `engine/run/gpu_run.sh`. It snapshots and follows the kernel log into `~/yah-scratch/gpu-logs/gpu-<tag>-<time>.dmesg.log` and flags timeout / reset lines.
8. Never free memory the GPU may still touch: a GPU read of a freed page hangs the shader and MES.

After a hang and reboot, read the previous boot's kernel log:

```
doas journalctl -k -b -1 --no-pager | grep -iE 'amdgpu|ring .*timeout|reset|MES'
```

`/tmp` is wiped by the reboot; keep logs elsewhere.

## Measuring

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
