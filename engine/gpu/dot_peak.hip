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
#include <cstdlib>
#include <string>

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

// Chain-count sweep. The GEMM keeps eight accumulator chains per thread
// (WRS=2 x WTS=4); if the achievable WMMA rate rises with the number of
// independent chains then the "ceiling" the GEMM is scored against is not a
// hardware constant but a function of how much dependent latency the kernel
// can cover -- which is the same quantity the phase-placement arms failed to
// move. The kernel is otherwise the same five instructions.
template <int C>
__global__ void WmmaF16ChainsKernel(float* out, int iters) {
  union {
    v16bf f;
    uint4 q[2];
  } pa, pb;
  pa.q[0] = pa.q[1] =
      make_uint4(0x3c003c00U, 0x3c003c00U, 0x3c003c00U, 0x3c003c00U);
  pb.q[0] = pb.q[1] = pa.q[0];
  const v16bf a = pa.f, b = pb.f;
  v8f acc[C] = {};
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < C; ++j)
      acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[j]);
  }
  float s = 0;
#pragma unroll
  for (int j = 0; j < C; ++j)
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

// Chain count is only meaningful under the controlled protocol: the sweep above
// runs its arms back to back and the observed dips move between runs, which is
// clock drift and not a chain effect -- except at 8 chains, which has never once
// reached the 27.5 TMAC/s the other counts reach. Since the GEMM uses exactly
// eight chains (WRS=2 x WTS=4), that has to be settled per process with the 15 s
// gap, so this entry point runs one chain count and exits.
template <int C>
void RunOneChainCount(float* fout, int iters) {
  for (int i = 0; i < 2; ++i) WmmaF16ChainsKernel<C><<<160, 256>>>(fout, iters);
  hipDeviceSynchronize();
  const auto c0 = std::chrono::steady_clock::now();
  WmmaF16ChainsKernel<C><<<160, 256>>>(fout, iters);
  hipDeviceSynchronize();
  const auto c1 = std::chrono::steady_clock::now();
  const double cms =
      std::chrono::duration<double, std::milli>(c1 - c0).count();
  const double cmacs = 160.0 * 8.0 * iters * C * 4096.0;
  std::printf("chains=%-3d ms=%9.3f  TMAC/s=%7.2f\n", C, cms, cmacs / (cms * 1e9));
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]).rfind("chains:", 0) == 0) {
    const int c = std::atoi(argv[1] + 7);
    float* f = nullptr;
    hipMalloc(&f, 64);
    const int it = 20000;
    switch (c) {
      case 1: RunOneChainCount<1>(f, it); break;
      case 2: RunOneChainCount<2>(f, it); break;
      case 3: RunOneChainCount<3>(f, it); break;
      case 4: RunOneChainCount<4>(f, it); break;
      case 5: RunOneChainCount<5>(f, it); break;
      case 6: RunOneChainCount<6>(f, it); break;
      case 7: RunOneChainCount<7>(f, it); break;
      case 8: RunOneChainCount<8>(f, it); break;
      case 10: RunOneChainCount<10>(f, it); break;
      case 12: RunOneChainCount<12>(f, it); break;
      case 16: RunOneChainCount<16>(f, it); break;
      default: std::fprintf(stderr, "chains: unsupported %d\n", c); break;
    }
    return 0;
  }
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
  std::printf("\nWMMA rate vs independent accumulator chains "
              "(blocks=160, 256 threads, one warmed launch):\n");
#define CHAIN_SWEEP(C)                                                         \
  do {                                                                         \
    for (int i = 0; i < 2; ++i)                                                \
      WmmaF16ChainsKernel<C><<<160, 256>>>(fout, iters);                       \
    hipDeviceSynchronize();                                                    \
    const auto c0 = std::chrono::steady_clock::now();                          \
    WmmaF16ChainsKernel<C><<<160, 256>>>(fout, iters);                         \
    hipDeviceSynchronize();                                                    \
    const auto c1 = std::chrono::steady_clock::now();                          \
    const double cms = std::chrono::duration<double, std::milli>(c1 - c0)      \
                           .count();                                           \
    const double cmacs = 160.0 * 8.0 * iters * C * 4096.0;                     \
    std::printf("  chains=%-3d ms=%9.3f  TMAC/s=%7.2f\n", C, cms,              \
                cmacs / (cms * 1e9));                                          \
  } while (0)
  CHAIN_SWEEP(1);
  CHAIN_SWEEP(2);
  CHAIN_SWEEP(3);
  CHAIN_SWEEP(4);
  CHAIN_SWEEP(6);
  CHAIN_SWEEP(8);
  CHAIN_SWEEP(12);
  CHAIN_SWEEP(16);
  CHAIN_SWEEP(24);
  CHAIN_SWEEP(32);
#undef CHAIN_SWEEP
  std::printf("\n");
  BENCH("sdot4_i8", Dot4I8Kernel, iout, 4.0);
  BENCH("sdot8_i4", Dot8I4Kernel, iout, 8.0);
  BENCH("fdot2_f16", Dot2F16Kernel, fout, 2.0);
  BENCH("pk_fma_f16", PkFmaF16Kernel, uout, 2.0);
  return 0;
}
