// clockprobe <mode> <seconds>: back-to-back kernels of one instruction mix for N seconds; prints launches,
// achieved rate and the clock implied by it (wmma: 34 cycles/WMMA/SIMD; valu: 1 FMA/cycle/lane).
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
__global__ void __launch_bounds__(256) k_wmma(float* out, int iters) {
  h16 a, b; for (int i = 0; i < 16; ++i) { a[i] = (_Float16)(threadIdx.x * 0.001f + i); b[i] = (_Float16)(i * 0.5f); }
  f8 c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  for (int it = 0; it < iters; ++it) {
    c0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c0); c1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c1);
    c2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c2); c3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c3);
    c4 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c4); c5 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c5);
    c6 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c6); c7 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c7);
  }
  f8 s = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = 0; for (int i = 0; i < 8; ++i) t += s[i];
  if (t == 1234.5f) out[threadIdx.x] = t;
}
__global__ void __launch_bounds__(256) k_valu(float* out, int iters) {
  float x[8]; for (int i = 0; i < 8; ++i) x[i] = threadIdx.x * 0.001f + i;
  const float m = 0.9999f, a = 0.0001f;
  for (int it = 0; it < iters; ++it)
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = __builtin_fmaf(x[i], m, a);
  float t = 0; for (int i = 0; i < 8; ++i) t += x[i];
  if (t == 1234.5f) out[threadIdx.x] = t;
}
__global__ void __launch_bounds__(256) k_lds(float* out, int iters) {
  __shared__ float4 s[2048];
  for (int i = threadIdx.x; i < 2048; i += 256) s[i] = make_float4(i, i, i, i);
  __syncthreads();
  float4 acc = make_float4(0, 0, 0, 0); int idx = threadIdx.x;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int r = 0; r < 8; ++r) { float4 v = s[(idx + r * 256) & 2047]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
    idx = (idx + 37) & 2047;
  }
  if (acc.x == 1234.5f) out[threadIdx.x] = acc.y + acc.z + acc.w;
}
__global__ void __launch_bounds__(256) k_mem(const float4* in, float4* out, size_t n) {
  for (size_t i = blockIdx.x * 256 + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) out[i] = in[i];
}
int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "wmma"; double secs = argc > 2 ? atof(argv[2]) : 5.0;
  float* out; hipMalloc(&out, 4096);
  size_t n = (size_t)256 << 20 >> 4; float4 *a = nullptr, *b = nullptr;
  if (!strcmp(mode, "mem")) { hipMalloc(&a, n * 16); hipMalloc(&b, n * 16); hipMemset(a, 0, n * 16); }
  const int blocks = 160 * 4, iters = !strcmp(mode, "wmma") ? 2048 : !strcmp(mode, "valu") ? 8192 : 4096;
  auto launch = [&] {
    if (!strcmp(mode, "wmma")) hipLaunchKernelGGL(k_wmma, dim3(blocks), dim3(256), 0, 0, out, iters);
    else if (!strcmp(mode, "valu")) hipLaunchKernelGGL(k_valu, dim3(blocks), dim3(256), 0, 0, out, iters);
    else if (!strcmp(mode, "lds")) hipLaunchKernelGGL(k_lds, dim3(blocks), dim3(256), 0, 0, out, iters);
    else hipLaunchKernelGGL(k_mem, dim3(1024), dim3(256), 0, 0, a, b, n);
  };
  launch(); hipDeviceSynchronize();
  auto t0 = std::chrono::steady_clock::now(); long nl = 0; double el = 0;
  while (el < secs) { for (int i = 0; i < 4; ++i) launch(); hipDeviceSynchronize(); nl += 4;
    el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
  double per = el / nl; printf("%s launches=%ld per_launch_ms=%.3f", mode, nl, per * 1e3);
  if (!strcmp(mode, "wmma")) { double w = (double)blocks * 8 * iters * 8; printf(" implied_GHz=%.3f", w / 80 * 34 / per / 1e9); }
  if (!strcmp(mode, "valu")) { double f = (double)blocks * 256 * iters * 32; printf(" implied_GHz=%.3f", f / (80 * 32) / per / 1e9); }
  if (!strcmp(mode, "mem")) printf(" GB/s=%.1f", 2.0 * n * 16 / per / 1e9);
  printf("\n"); return 0;
}
