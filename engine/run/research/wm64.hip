// wm64.hip: WMMA rate in wave64 vs wave32 (8 independent accumulators, f16 16x16x16)
#include <hip/hip_runtime.h>
#include <cstdio>
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
#ifdef W64
typedef float fa __attribute__((ext_vector_type(4)));
#define WMMA __builtin_amdgcn_wmma_f32_16x16x16_f16_w64
#else
typedef float fa __attribute__((ext_vector_type(8)));
#define WMMA __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
#endif
__global__ void __launch_bounds__(256) wmma_rate(float* out, int iters) {
  h16 a, b; for (int i = 0; i < 16; ++i) { a[i] = (_Float16)(threadIdx.x * 0.001f + i); b[i] = (_Float16)(i * 0.5f); }
  fa c0 = {}, c1 = {}, c2 = {}, c3 = {}, c4 = {}, c5 = {}, c6 = {}, c7 = {};
  for (int it = 0; it < iters; ++it) {
    c0 = WMMA(a, b, c0); c1 = WMMA(a, b, c1); c2 = WMMA(a, b, c2); c3 = WMMA(a, b, c3);
    c4 = WMMA(a, b, c4); c5 = WMMA(a, b, c5); c6 = WMMA(a, b, c6); c7 = WMMA(a, b, c7);
  }
  fa s = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7; float t = 0; for (int i = 0; i < (int)(sizeof(fa) / 4); ++i) t += s[i];
  if (t == 1234.5f) out[threadIdx.x] = t;
}
int main() { float* o; hipMalloc(&o, 4096); hipLaunchKernelGGL(wmma_rate, dim3(160), dim3(256), 0, 0, o, 4096); hipDeviceSynchronize();
  for (int i = 0; i < 3; ++i) hipLaunchKernelGGL(wmma_rate, dim3(160), dim3(256), 0, 0, o, 4096); hipDeviceSynchronize(); printf("ok\n"); }
