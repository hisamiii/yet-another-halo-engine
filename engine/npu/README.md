# npu/

XDNA2 through XRT, first-class rather than bolted on.

- xclbin set for the frozen shapes, ATB config3 family. The `K_Problemsize` override is required; without it only K=4096 verifies.
- A, B and C as dma-buf imports from the iGPU allocator. The GPU encodes A and repacks B in place.
- Async submission with the wait deferred to the join. The measured 20.9% GPU idle comes from blocking on that wait.
- The down role needs **one** run, not two. Verify the duplicate launch in the reference before porting the pattern.

**Blocker:** the all-NPU placement fails the top-1 check at some chunk lengths (cosine ~0.95 where it should be >0.99). Diagnose it before building on it.
