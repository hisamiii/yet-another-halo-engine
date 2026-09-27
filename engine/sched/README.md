# sched/

The reason to go greenfield: the two engines and the power budget are the architecture.

- Phase routing: prefill splits across engines, decode stays on the GPU.
- Per-layer-type split: full-attention layers by token prefix end to end (the NPU's tokens attend only to tokens it owns; the GPU reads its K/V from the shared cache); Gated DeltaNet layers by head, because the recurrence is sequential in tokens.
- Two async queues joined by events; the host blocks only where the data is consumed.
- Power is the resource: both engines share 130 W and the NPU is derated ~21% under concurrency.

Gate: GPU stream idle < 5% with the split engaged.
