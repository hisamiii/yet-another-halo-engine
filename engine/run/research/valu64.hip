// valu64.hip: pure f32 FMA throughput (8 independent chains), wave32 vs wave64 (compile with -mwavefrontsize64)
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void __launch_bounds__(256) valu_rate(float* out, int iters) {
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
int main() { float* o; (void)hipMalloc(&o, 4096); for (int i = 0; i < 4; ++i) hipLaunchKernelGGL(valu_rate, dim3(640), dim3(256), 0, 0, o, 8192); (void)hipDeviceSynchronize(); printf("ok\n"); }
