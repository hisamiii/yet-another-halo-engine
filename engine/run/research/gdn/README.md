# Chunked Gated DeltaNet (WY form): groundwork, 2026-10-02

DeltaNet (yah_deltanet, recurrent, HIP order) is 2.58% of pp8192 cycles
(749 M of 29.0 G).

The current kernel's math, with k_hat = inv_k k and q_hat = q_scale q
(prep_kq), and value head h reading key head h mod 16:

    S_t = a_t S_{t-1} + b_t (v_t - a_t S_{t-1} k_hat_t) k_hat_t^T,  o_t = S_t q_hat_t

Layer-0 dumps (pp2048, YAH_DUMP_LAYER=0: .conv .kq .ab .raw) checked by
gdn_ref.py:

| check | relative error |
|---|---|
| float64 recurrence vs the kernel | 0.9-2.8e-7 |
| chunked WY form (C = 64, FLA chunk_gated_delta_rule) vs the recurrence | 4-7e-16 |

The decay goes down to a = 0.149, so a kernel must use log-space gating
(exp(G_i - G_j), i >= j, never a division).

Precision (gdn_prec.py): the chunked form with every matmul input rounded
gives output relative error

| inputs | relative error |
|---|---|
| f16 | 1.2-4.2e-4 |
| bf16 | 0.9-3.4e-3 |
| f32 | ~6e-8 |

End to end, YAH_DN_F16SIM=1 rounds the recurrent kernel's k, q, v, the state
read by both dot products and the update coefficient to f16 (state kept f32).
8K gate on 4 docs: mean KLD 0.000003, 99% precision 99.98%, same top 99.97%.
That is 7x below kv8a16, so a plain f16-WMMA chunked kernel is safe;
compensated (hi+lo) inputs are not needed.
