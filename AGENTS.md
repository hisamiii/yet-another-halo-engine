# Instructions for yet-another-halo-engine

Conventions for anyone (human or agent) changing this repo. Adapted from llama.cpp's AGENTS.md / CONTRIBUTING.md. Read this before writing code or docs.

## The project

Qwen3.8-27B (IQ4_XS GGUF) inference on Strix Halo (gfx1151), written as Loom kernels run through HRX. Python generators emit the kernels; small C++ drivers run them.

Layout:

- `engine/gpu/loom/tools/`: kernel generators. `emit_prefill_pp.py` and `emit_decode.py` write a HAL set (compiled kernels + `dispatch.txt` / `decode.txt`); `*_check.py` test generated kernels against numpy oracles through `hal_run`.
- `engine/gpu/loom/*.loom`: hand-written Loom kernels the emitters still read; `tables/`: IQ codebooks.
- `engine/run/`: the drivers (`loom_forward_pp.cc` prefill and prefill -> decode, `loom_decode.cc`, `hal_run.cc`, `hal_bench.cc`), `gpu_run.sh` (run every GPU job through it), `gate/` (accuracy gate), `kvq/` (KV codec gate).
- `engine/model/`: `loom_runtime.hpp` (HRX wrappers), `loom_decoder.hpp` (the decode step).
- `engine/core/`: GGUF reader, model config, tokenizer, small CLI tools.
- `engine/tests/`: end-to-end gates.
- `docs/`: see Docs below.

## Code

- Keep it simple. A simpler change that does 90% of the job beats a complex one that does 100%. Plain loops over clever constructs, no templates unless they remove real duplication.
- Reuse what exists (generators, `hal_run`, `gpu_run.sh`, the checkers) before adding a new tool or subsystem.
- Read the surrounding code first and match it: naming, idioms, comment density.
- ASCII only, in code, comments and docs: `-`, `->`, `x`, `...`, not em-dashes, arrows, `×` or `…`.
- C++: C++20, the existing style (types and functions `CamelCase`, locals `snake_case`, members `snake_case_`, constants `kName`), 2-space indent, 120 columns. Format with `clang-format` (`.clang-format` in the root). Sized integers (`std::uint32_t`) for sizes and indices that cross the host/device boundary.
- Python: 4-space indent, the existing style. Generators print Loom text with `e(...)`; keep one kernel per `gen_*` function.
- No dead code. When an experiment loses, delete its switch and code path and record one line in `docs/results.md` (what, the number, why). Git keeps the code if it is ever needed again.
- Environment switches are for real modes only (context size, KV format, debug dumps), not for parked experiments.

### Comments

- Write the code first, then add a comment only where the code can't say it: a non-obvious invariant, a hardware fact, a reason something must stay as it is.
- Keep comments to 1-2 lines. Simple wording (ASD-STE100 Simplified Technical English; write like a caveman if needed).
- Do not hard-wrap comments to a column width. One sentence or thought per line.
- No narration of the task or the session ("fixes the bug we saw", "now faster"). Comments must make sense to a reader who never saw the conversation.
- Numbers in comments only when they justify the code (e.g. "LDS table: 8 -> 3 VGPR, 2% faster"); measurements belong in `docs/results.md`.

```cpp
// GOOD: explains a contract the code can't show
// Loom drops clamps it proves redundant from the launch grid: launch exactly the compiled grid.
CheckGrid(name, gx);

// BAD: restates the code
// check the grid of the kernel against the recorded grid
CheckGrid(name, gx);
```

## Docs

Docs are for a human reader who knows GPUs but not this repo. They hold what matters now, not how we got here.

- Lead with what it is and why it matters, then how. Short sections, plain words, concrete numbers.
- Tables for numbers. Every number says what was measured, on what (context, set), and when it was last measured.
- One paragraph per line in Markdown; no hard wrapping.
- Track only: architecture and data layouts, how to build / run / verify / profile, current results and baselines, rules that prevent GPU hangs or wrong measurements, and a one-line-per-item list of things that were tried and lost.
- No session diaries, no dated "today we..." sections, no step-by-step history. Git history is the history.
- Update the doc in the same commit as the code it describes. Delete text that is no longer true.

Docs live in `docs/`: `architecture.md`, `build-and-run.md`, `results.md` (results, baselines, lost experiments), `hardware.md` (measured facts about the box).

### Local scratchpads

`notes/` in the repo root is gitignored: use it for lab notebooks, TODOs, raw logs and session notes. Big experiment output goes to `~/yah-scratch/<topic>` (delete large outputs after use). Promote only distilled, still-true findings into `docs/`.

## Measuring

- GPU timing: one round per candidate; cool the APU to <= 55 C (`z13ctl status`) and wait 1 s (single kernels), 15 s (pp2048) or 30 s (pp8192 and longer) before each run. No clock pinning.
- A kernel change must be bit-identical (or checked against an oracle: `tools/*_check.py`) before it is timed.
- A cleanup or refactor must keep every emitted HAL byte-identical to the previous emit.
- When an optimization loses, find out why before dropping it, and record the cause.

## GPU safety

A bad dispatch on this box can hang the GPU ring and force a reboot.

- Launch exactly the grid a kernel was compiled for. Loom may drop index clamps it proves redundant from the launch config, so a larger grid reads out of bounds.
- Data that other threads wrote to LDS needs a workgroup barrier before the first read, prologues included.
- Clamp or validate every index read from memory (page tables, token ids) before it addresses a buffer.
- Run GPU jobs through `engine/run/gpu_run.sh`.

## Commits

- Small, focused commits; format-only changes in their own commit.
- Concise message, like llama.cpp: an `area : what changed` subject, and a body only when the why or the measured effect isn't obvious (1-3 short lines). No lists of every file touched, no sales language.
- End with `Assisted-by: Claude Opus 5.5` when an agent helped. Not `Co-authored-by`.

```
// GOOD
decode : quantized KV attention (kv8a16 / kv4a16)

30K context: 74.1 -> 70.6 ms/token (kv8).

Assisted-by: Claude Opus 5.5


// BAD
This commit introduces comprehensive quantized KV cache support for the decode
path, adding three new kernels, wiring them through the decoder and the prefill
handoff, and significantly improving long-context decode performance.
```
