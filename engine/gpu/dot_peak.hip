// Instruction-ceiling probe for the packed dot and packed-FMA classes.
//
// The prefill GEMM's ceiling on this part is the WMMA rate: 24.17 TMAC/s
// (48.35 TFLOPS), measured with eight live accumulator chains. WMMA is
// element-count-shaped, so fp16 and int8 measure the same. Packed dots are a
// different execution class and may issue more MACs per cycle, in which case a
// GEMM built on them beats the current kernel without touching the weight
// format. Eight independent chains again, or the compiler deletes the work.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <chrono>
#include <cstdint>
#include <cstdio>

#define CHAINS 8
#define DOT_FEATURES

DOT_FEATURES __global__ void Dot4I8Kernel(const unsigned a, const unsigned b,
                                          int* out, int iters) {
  int acc[CHAINS] = {};
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < CHAINS; ++j)
      asm volatile("v_dot4_i32_i8 %0, %1, %2, %0"
                   : "+v"(acc[j])
                   : "v"(a), "v"(b));
  }
  int s = 0;
#pragma unroll
  for (int j = 0; j < CHAINS; ++j) s += acc[j];
  if (s == 123456789) out[0] = s;
}

DOT_FEATURES __global__ void Dot8I4Kernel(const unsigned a, const unsigned b,
                                          int* out, int iters) {
  int acc[CHAINS] = {};
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < CHAINS; ++j)
      asm volatile("v_dot8_i32_i4 %0, %1, %2, %0"
                   : "+v"(acc[j])
                   : "v"(a), "v"(b));
  }
  int s = 0;
#pragma unroll
  for (int j = 0; j < CHAINS; ++j) s += acc[j];
  if (s == 123456789) out[0] = s;
}

DOT_FEATURES __global__ void Dot2F16Kernel(const unsigned a, const unsigned b,
                                           float* out, int iters) {
  float acc[CHAINS] = {};
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < CHAINS; ++j)
      asm volatile("v_dot2_f32_f16 %0, %1, %2, %0"
                   : "+v"(acc[j])
                   : "v"(a), "v"(b));
  }
  float s = 0;
#pragma unroll
  for (int j = 0; j < CHAINS; ++j) s += acc[j];
  if (s == 123456789.0F) out[0] = s;
}

__global__ void PkFmaF16Kernel(const unsigned a, const unsigned b,
                               unsigned* out, int iters) {
  unsigned acc[CHAINS] = {};
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < CHAINS; ++j)
      asm volatile("v_pk_fma_f16 %0, %1, %2, %0"
                   : "+v"(acc[j])
                   : "v"(a), "v"(b));
  }
  unsigned s = 0;
#pragma unroll
  for (int j = 0; j < CHAINS; ++j) s += acc[j];
  if (s == 123456789U) out[0] = s;
}

// The prefill GEMM's real ceiling: fp16 WMMA. This was previously only a
// hardcoded reference line from an earlier session, so every "X% of ceiling"
// claim in the docs was a cross-session comparison and could not be checked.
// Eight independent accumulator chains, all eight read back, or the compiler
// deletes six of them and the rate reads ~4x high. One WMMA is a *wave*
// operation and computes 16*16*16 = 4096 MACs total, not per lane, so MACs are
// counted per warp, not per thread -- counting per thread reads 32x high.
using v16bf = _Float16 __attribute__((ext_vector_type(16)));
using v8f = float __attribute__((ext_vector_type(8)));

__global__ void WmmaF16Kernel(float* out, int iters) {
  union {
    v16bf f;
    uint4 q[2];
  } pa, pb;
  pa.q[0] = pa.q[1] = make_uint4(0x3c003c00U, 0x3c003c00U, 0x3c003c00U,
                                 0x3c003c00U);
  pb.q[0] = pb.q[1] = pa.q[0];
  const v16bf a = pa.f, b = pb.f;
  v8f acc[CHAINS] = {};
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < CHAINS; ++j)
      acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[j]);
  }
  float s = 0;
#pragma unroll
  for (int j = 0; j < CHAINS; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) s += acc[j][e];
  if (s == 123456789.0F) out[0] = s;
}

#define BENCH(NAME, KERNEL, OUTPTR, MACS)                                      \
  do {                                                                         \
    for (int blocks : {40, 160, 640}) {                                        \
      for (int i = 0; i < 2; ++i)                                              \
        KERNEL<<<blocks, 256>>>(0x01020304U, 0x05060708U, OUTPTR, iters);      \
      hipDeviceSynchronize();                                                  \
      const auto t0 = std::chrono::steady_clock::now();                        \
      for (int i = 0; i < 5; ++i)                                              \
        KERNEL<<<blocks, 256>>>(0x01020304U, 0x05060708U, OUTPTR, iters);      \
      hipDeviceSynchronize();                                                  \
      const auto t1 = std::chrono::steady_clock::now();                        \
      const double ms =                                                        \
          std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;    \
      const double macs =                                                      \
          double(blocks) * 256.0 * iters * CHAINS * (MACS);                    \
      const double tmacs = macs / (ms * 1e9);                                  \
      std::printf("%-10s blocks=%4d  ms=%9.3f  TMAC/s=%7.2f  TFLOPS=%7.2f\n",  \
                  NAME, blocks, ms, tmacs, 2.0 * tmacs);                       \
    }                                                                          \
  } while (0)

int main() {
  int* iout = nullptr;
  float* fout = nullptr;
  unsigned* uout = nullptr;
  hipMalloc(&iout, 64);
  hipMalloc(&fout, 64);
  hipMalloc(&uout, 64);
  const int iters = 20000;
  std::printf("WMMA reference (previous session, not re-measured): fp16 24.17, "
              "iu8 25.15, iu4 52.6 TMAC/s\n\n");
  // -r 1: two warm-up launches, then ONE timed launch. Averaging several
  // launches mixes the boosted first repetition with the settled ones, which
  // methodology.md forbids; the drift between regimes is larger than the effects
  // being measured, so a single warmed repetition is the number to pair.
  for (int blocks : {40, 160, 640}) {
    for (int i = 0; i < 2; ++i) WmmaF16Kernel<<<blocks, 256>>>(fout, iters);
    hipDeviceSynchronize();
    const auto w0 = std::chrono::steady_clock::now();
    WmmaF16Kernel<<<blocks, 256>>>(fout, iters);
    hipDeviceSynchronize();
    const auto w1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(w1 - w0).count();
    const double macs =
        double(blocks) * (256.0 / 32.0) * iters * CHAINS * 4096.0;
    const double tmacs = macs / (ms * 1e9);
    std::printf("%-10s blocks=%4d  ms=%9.3f  TMAC/s=%7.2f  TFLOPS=%7.2f\n",
                "wmma_f16", blocks, ms, tmacs, 2.0 * tmacs);
  }
  std::printf("\n");
  BENCH("sdot4_i8", Dot4I8Kernel, iout, 4.0);
  BENCH("sdot8_i4", Dot8I4Kernel, iout, 8.0);
  BENCH("fdot2_f16", Dot2F16Kernel, fout, 2.0);
  BENCH("pk_fma_f16", PkFmaF16Kernel, uout, 2.0);
  return 0;
}
