# gpu/

Port the WMMA framework, then work the aggregate.

- Tiling, LDS and staging are ported. The tile space is swept and exhausted; do not re-sweep it.
- Twelve decoders for the two targets. IQ4_XS is the crossover (27.5% of one, 41% of the other) and must be excellent on both.
- Paired gate/up coverage for every target type. IQ3_XXS has no paired case today and is ~19% of the primary target.
- `Complete` (tail-free) is per (type, shape): +4% on Q4_K, neutral on Q6_K/IQ3_XXS, **-15% on IQ4_XS**.

Starting point is **~30 TFLOPS aggregate**, not the 40.6 of the best single kernel. M2g target is 38-40+. Gate: top-1 validation unchanged.
