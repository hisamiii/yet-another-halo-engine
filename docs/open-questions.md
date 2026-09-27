# Open questions

Each entry states the observation, the competing explanations, and the cheapest test that separates them.

## 1. The all-NPU placement fails numerics at some shapes

**Observation.** `--validate-prefill` compares batched-prefill logits against a sequential reference and requires a top-1 match.

| N | NPU tokens M | GPU remainder | top-1 | cosine |
| ---: | ---: | ---: | :---: | ---: |
| 512 | 512 | 0 | yes | 0.999993 |
| 1025 | 1024 | 1 | yes | 0.987133 |
| 1300 | 1024 | 276 | yes | 0.999228 |
| 1536 | 1024 | 512 | yes | 0.998556 |
| 2048 | 1024 | 1024 | yes | 0.985100 |
| 1024 | 1024 | 0 | **no** | 0.953139 |
| 2048 | 2048 | 0 | **no** | 0.953542 |

**Why it is not simple.** The obvious hypotheses both fail. It is not "the last token was computed on the NPU": N=1025 leaves exactly one token on the GPU and passes, while M=512/N=512 puts every token on the NPU and is near-exact. It is not bf16 precision: the same bf16 path gives 0.99999 at M=512.

**What is left.** Either a host-path interaction when the chunk length equals the xclbin's M at M >= 1024 (the down projection accumulating across the whole hidden state with no GPU residual to compare against), or a generator/xclbin issue that only appears at those exact shapes and not at M=512. The two failing cosines being nearly identical (0.9531 and 0.9535) across different M values points away from a shape-specific packing bug and toward something systematic.

**Cheapest separating test.** Run M=1024 with N=1024 but with the down projection disabled (gate/up on the NPU, down on the GPU). If it passes, the gate/up all-NPU path is sound and the fault is in the down all-NPU accumulation; if it fails, the fault is earlier and precision-related. Also worth running M=1536 with N=1536, which gives a third all-NPU point at a different M.

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
