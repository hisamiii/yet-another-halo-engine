# GPU ring hang: yah_qdq_f32 with an over-declared config

Date: 2026-08-14 (host monotonic 71066 s in `/home/q/log`).

Two GPU resets, one wedged device, one `iree-benchmark-loom` segfault. The box
recovered on its own; no reboot was needed. This is the write-up of what caused
it and what now prevents it.

## What the log says

```
[71066.964850] amdgpu 0000:c4:00.0: ring gfx_0.0.0 timeout, signaled seq=3550878, emitted seq=3550881
[71068.968723] amdgpu 0000:c4:00.0: MES failed to respond to msg=RESET
[71068.968733] amdgpu 0000:c4:00.0: Ring gfx_0.0.0 reset failed
[71068.970146] amdgpu: Failed to suspend process pid 1254244      <- iree-benchmark-loom
[71068.970150] amdgpu 0000:c4:00.0: remove_all_kfd_queues_mes: Failed to remove queue 0 for dev 28801
[71071.053823] amdgpu 0000:c4:00.0: MES failed to respond to msg=REMOVE_QUEUE
[71073.347418] [drm:gfx_v11_0_hw_fini...] *ERROR* failed to halt cp gfx
[71074.183566] amdgpu 0000:c4:00.0: GPU reset(1) succeeded!
[71074.184816] iree-benchmark-[1254244]: segfault ... in libhsa-runtime64.so.1.21.0
[71075.333533] amdgpu 0000:c4:00.0: GPU reset(2) succeeded!
```

Three packets were outstanding on the legacy gfx ring, MES could not service a
RESET or a REMOVE_QUEUE, and the queue that could not be removed belonged to the
benchmark process. The benchmark then faulted inside libhsa while the runtime was
being torn down underneath it.

One line is easy to misread: `Process waterfox-bin pid 6005 thread waterfox-b:cs0`
names the queue at the head of the legacy ring, not necessarily the job that
stopped draining. The operations that actually failed are all about our queue,
so the queue that wedged the ring is ours. The uncertainty is recorded rather
than hidden: MES never came back, so there is no clean post-mortem.

## Root cause

The benchmark pass was this:

```
iree-benchmark-loom yah_qdq_f32.loom --config=yah_qdq.blocks=10240 --case=@yah_qdq_case ...
```

`@yah_qdq_case` binds `tensor<1024xf32>` -- 4 KB. But `yah_qdq` took its block
count from `config @yah_qdq.blocks` and used it for **both** the grid and the view
extent, so `blocks=10240` declared a 10240 x 32 = 327680 x f32 = 1,310,720 B view
over a 4096 B buffer. The kernel then stored to byte offsets up to 1,310,716, i.e.
1.3 MB past the end of the allocation, three times per benchmark invocation
(`--warmup-iterations=2 --iterations=1`). The correctness case had been run with
the matching `blocks=32` and passed; only the benchmark was re-run with the
larger number. That is the whole bug: a validated case was re-run under a config
it was never validated against.

The compiler had already computed the footprint. `--compile-report=details`
records it:

| config | `source_low.memory.roots[0].interval_envelope.byte_count` | case binding |
| --- | --- | --- |
| `blocks=32` | 4096 | 4096 |
| `blocks=10240` | 1310720 | 4096 (mismatch) |

## Why the kernel cannot check this itself

The obvious fix -- guard the access with the buffer's real length -- is not
available on this target:

- `buffer.length` exists and returns byte length, but it has **no AMDGPU
  target-low contract**: "target 'yah_wave32' export 'yah_qdq' has no target-low
  contract for 'buffer.length'". It is a host-side property, and a raw buffer
  carries no device-visible length.
- Launch arguments are **not in scope in the launch-config region**, so the grid
  cannot be derived from the operand either ("PARSE/001: undefined SSA value").
- A dynamic view extent is only accepted by the subrange verifier when its bound
  comes from a `config.decl ... where [range(...)]`. `index.assume` on a
  buffer-derived value did not satisfy it (`SUBRANGE/024`, "view_bound is
  <dynamic>"), and `index.div` by a power of two lowers to `index.shrui`, which
  an AMDGPU register-unit constraint rejects (`TARGET/004`, `low_register_unit_count`).

So on AMDGPU the config is the **only** size channel, which makes it a promise
from the caller rather than a fact the kernel can verify. The fix therefore lives
in the tooling around the kernel, not in the kernel.

## What changed

1. `yah_qdq_f32.loom` documents the required config for each case, and the case
   tensors are sized to match it: `@yah_qdq_small` needs `blocks=32`
   (1024 f32), `@yah_qdq_full` needs `blocks=10240` (327680 f32).
2. `tools/loom_preflight.py` reads the compile report's declared operand
   footprint and the byte size of the tensors the case binds, and **refuses**
   when the declaration exceeds the binding. On the lethal pairing it prints the
   1310720 vs 4096 mismatch and exits 2 before anything is dispatched.
3. `loom_run.sh` compiles, preflights, runs the correctness case, and only then
   benchmarks -- always under the same config. Correctness and timing can no
   longer diverge.

```
$ ./loom_run.sh yah_qdq_f32.loom @yah_qdq_small @yah_qdq_small_bench yah_qdq.blocks=10240
preflight: REFUSING to run @yah_qdq_small
  argument 0: config declares 1310720 B of operand, case binds 4096 B
runner exit=2  (no GPU dispatch)
```

## Result

`yah_qdq_f32.loom`, 10240 blocks / 327680 f32, correctness `state: ok`:

| case | config | mean dispatch |
| --- | --- | --- |
| `@yah_qdq_small` | `blocks=32` | 0.0152 ms |
| `@yah_qdq_full` | `blocks=10240` | 0.0546 ms |

## Residual gap

The check data is uniform, and the operation is idempotent on already-quantized
values, so an *under*-declared config would silently leave the tail unquantized
without failing the check. Closing that needs a non-uniform fixture plus a
`check.oracle.call` reference implementation, which is the next methodology item
in `PORTING.md`. The over-declaration direction -- the one that hangs the GPU --
is now mechanically blocked.
