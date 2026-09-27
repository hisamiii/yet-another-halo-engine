# Open questions

Each entry states the observation, the competing explanations, and the cheapest test that separates them.

## 1. The all-NPU placement fails numerics at M=1024 only

**Observation.** `--validate-prefill` compares batched-prefill logits against a sequential reference and requires a top-1 match. Re-run after regenerating the stale xclbins:

| N = M | top-1 | cosine |
| ---: | :---: | ---: |
| 512 | yes | 0.999993 |
| 1536 | yes | 0.997575 |
| 2048 | yes | 0.967327 |
| 1024 | **no** | 0.827 / 0.938 / 0.970 |

The M=2048 point that used to fail now passes. Its xclbin (`npu_gu_t2048_exact`) was regenerated at 17:30, after the slot-tail generator fix; `npu_gu_t1024_exact` was not, which is why M=1024 was the apparent outlier. Regenerating the M=1024 pair with the patched generator does not change the result, so the stale xclbin was a red herring for this shape.

**What the xclbin is not.** The M=1024 xclbins are numerically correct. `atb_npu_run npu_gu_t1024_exact.xclbin ... 1024 5120 17408 3 dev 30` passes 30 varying-data stress iterations at 32.11 TFLOPS, and the down xclbin passes at 32.20. The generator and the kernel are not the fault.

**What is left.** The M=1024 cosine varies run to run with the same binaries and the same inputs (0.827, 0.938, 0.970) while M=512/1536/2048 are stable. That is a race, not a tiling or precision effect. The separating test points the same way: keeping the down projection on the GPU (gate/up all-NPU) is *worse* (0.918) than the full split, so the fault is in gate/up, not the down accumulation. M=1024 is exactly the balance point (GPU 20.9% idle), the one shape where the NPU is the critical path and the GPU has no work to hide a missing dependency, which is consistent with a host-side ordering or dma-buf coherence bug rather than a kernel bug.

**Cheapest separating test.** Serialize the engine's NPU handoff at M=1024: after `LaunchAtbEncodeAFp16` and the repack, insert a full `hipDeviceSynchronize` (not `WaitStreamSleeping`) before `atb->Launch()`, and a full device sync before the decode. If the cosine becomes stable, the fault is the host ordering around the dma-buf handoff; if it stays variable, the fault is in the xclbin's internal scheduling. A second, cheaper check is to drop the concurrent GPU token rows (already zero at M=1024) and confirm the down branch is not writing the wrong rows.

**Impact.** Until closed, the 100%-NPU mode is a correctness risk, and the shipped split is unsound for a prompt whose final chunk is exactly the NPU width.

## 2. The GPU-only baseline is unmeasured

Every number in this repo is relative to another NPU configuration. The harness `npu_ceiling.sh` runs the ordinary GPU path against the best split in one paired session, which would bound the total value of the NPU. Without it, a decision to invest in the NPU cannot be justified against investing in the GPU path instead.

## 3. The NPU does not reach its standalone ceiling in the engine

Standalone the xclbins sustain 32.4 TFLOPS; in the engine the same gate/up pair implies about 25 TFLOPS and ~14 GB/s of operand traffic, against ~22 GB/s standalone. The operand buffers are shared via dma-buf, so the loss could be (a) the NPU reading from GTT rather than a local allocation, (b) the A/C interop with the GPU, or (c) the B operand being freshly written and cold. A standalone driver that reads B from the same dma-buf import would separate them.

## 4. An int4/int8 B GEMM with GGUF scales is unproven

The mlir-aie `block_datatypes/gemm_asymmetric_tile_buffering` example is the right starting point, and the shape generator already accepts the required tile counts after a slot-tail fix. What is not shown is a B operand that carries GGUF-style per-block scales and matches the GPU path's numerics. If it works it removes the repack and roughly halves NPU operand bytes; if it does not, the NPU path stays bandwidth-heavy.

## 5. Batch-1 NPU behaviour is unmeasured

Decode is routed to the GPU on reasoning, not evidence. The measurement is cheap: the NPU's GEMV path against the GPU's at batch 1, for the same weight bytes.

## 6. The vision tower is entirely unmeasured

Whether a vision tower on this hardware is prefill-shaped, how large its activations are, and whether the NPU path is even worth having for it are all open. This is the largest unmeasured surface in the scope.
