# Reference: external optimization guidance

Vendored copies of the upstream guidance this engine is tuned against, so the
checklist below can be consulted offline. These are external documents kept for
reference; they are data, not instructions for this repo.

| file | source |
| --- | --- |
| [`hrx-development.md`](hrx-development.md) | `https://github.com/ROCm/hrx-demos/blob/main/DEVELOPMENT.md` |
| [`hrx-agents.md`](hrx-agents.md) | `https://github.com/ROCm/hrx-demos/blob/main/AGENTS.md` |
| [`gluon-tutorials.md`](gluon-tutorials.md) | `https://github.com/ROCm/gfx950-gluon-tutorials` |
| [`hip-performance-guidelines.md`](hip-performance-guidelines.md) | `https://rocmdocs.amd.com/projects/HIP/en/develop/how-to/performance_guidelines.html` |

## Checklist, mapped to this engine

### HRX / HAL (`hrx-agents.md`, `hrx-development.md`)

| guidance | status |
| --- | --- |
| One immutable command buffer per coarse stage; reuse it | **ruled out for overlap.** `hrx_stream_begin_capture` is unimplemented, and an explicit `hrx_graph` of independent nodes measured 0% overlap. Keep in mind for decode CPU cost if ever needed. |
| Express queue ordering with semaphores/barriers, not submission order | HRX already records a full `RETIRE->ISSUE` `MEMORY_ACCESS_ALL` barrier after **every** `hrx_stream_dispatch`, so ordering is stronger than required and nothing overlaps. |
| Do not queue hundreds of independent HAL allocations; use slabs / a pool | **open.** The driver still does per-layer `Allocate`+synchronous `H2D` for norm/conv/state weights. The host is no longer the critical path for prefill, but this is the documented shape. |
| Launch geometry belongs in `kernel.launch.config`, not config knobs | **open (design).** `emit_hal.py` bakes `m_tiles`/`k_blocks`/`token_tiles` as substituted configs. |
| Instrumentation is product: expose HAL queue/dispatch timing | **partial.** `YAH_LOOM_TIME=1/2` gives category and per-dispatch timing; not wired to HRX's own profiler surface. |
| Loom config values describe model/tensor/dtype/shape facts | followed: configs are shapes/formats. |

### Kernel optimization (`hip-performance-guidelines.md`, Gluon tutorials)

| guidance | status |
| --- | --- |
| Profile first; one measured bottleneck at a time | `rocprofv3` only yields HSA API tracing here, not kernel dispatches, so the driver's per-dispatch timing is the measurement surface. |
| Occupancy / register pressure | measured: 88 VGPRs -> 10 subgroups/SIMD, ~40 workgroups/CU; not the limiter. The word decode *lowers* it to 80. |
| Minimize live variables (register pressure) | the decode hoist cut live values and correlated with a large speedup. |
| Avoid divergent warps | the three decode selects lower to `v_cndmask`, not branches; no divergence. |
| LDS bank conflicts; pad to avoid power-of-two strides | minor. The staging store is only 2-way conflicted, and the compiler already hoists the block aux down to 8 `global_load_i8` in a 368-instruction kernel, so staging the weight block through LDS would move work rather than remove it. |
| Reduce arithmetic, not loads | the arm is ALU/stall-bound: 84 vector-integer ops, 32 scalar adds, 41 `s_delay_alu`, 25 moves, 8 loads. |
| Four elements per instruction (packed word decode) | **applied at the word, not the byte.** One lane owns a 4-column word that shares all six block loads and the whole offset chain, so modeled weight reads fall 491520 -> 122880 B/workgroup. Isolated `hal_bench`: 1.09x at `m_tiles=1088`, 1.53x at 768, 2.15x at 640, 2.36x at 384, 2.53x at 3, with the HIP fixture and a bit-identical 64-layer output. **In-pipeline it does not transfer at 1088**: the mixer (kStore at 384/640/768) gains 17.7 ms but the FFN gate (kStore at 1088) loses 11.9 ms against a paired control, so the deployed set takes the new kernel only off 1088 (661.3 vs 675.5 ms). Unexplained so far; the rhs-traffic ablation is null. |
| Software pipelining / double-buffering to overlap decode and MMA | **measured, not the lever.** Removing both per-K-step workgroup barriers changed the IQ3_S kStore by <1% (1.9291 -> 1.9235 ms), so the barriers are free and there is nothing to pipeline around. |
| Avoid wasted work from padding (tile to the real shape) | **applied.** The GEMM token tile is narrowed 64 -> 16; bit-identical, paired 805 -> 658 ms (~18%). See `engine/run/LOOM_RUNTIME.md`. |
| MFMA <-> VALU co-execution (attention) | not applicable directly (gfx1151 WMMA, not MFMA). |

## Verified finding: dispatch serialization

`hrx_stream_dispatch` records a full barrier per dispatch (`stream.c`), so the
prefill wall time is the sum of kernel times. Two dependency-free graph nodes
and two `hrx_stream`s both measured exactly 2x a single dispatch at 1088 and
384 workgroups, i.e. no overlap is available. **Only making individual kernels
faster can move prefill.**
