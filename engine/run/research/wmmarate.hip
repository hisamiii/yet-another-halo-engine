// wmmarate.hip: gfx1151 matrix-pipe microbenchmark. Each wave runs 8 independent
// WMMA accumulator chains for ITER iterations (no dependent-WMMA stalls); optional
// V independent VALU FMAs per WMMA (inline asm, own registers) to measure what a
// VALU op next to WMMAs costs. Cycles come from rocprofv3 SQ_BUSY_CYCLES; the
// kernel name encodes the variant.  usage: wmmarate <variant> [iters]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));

#define CHAINS 8

template <int V>
__device__ __forceinline__ void valu_ops(float& x0, float& x1, float& x2, float& x3) {
  if constexpr (V >= 1) asm volatile("v_fma_f32 %0, %0, %0, 1.0" : "+v"(x0));
  if constexpr (V >= 2) asm volatile("v_fma_f32 %0, %0, %0, 1.0" : "+v"(x1));
  if constexpr (V >= 3) asm volatile("v_fma_f32 %0, %0, %0, 1.0" : "+v"(x2));
  if constexpr (V >= 4) asm volatile("v_fma_f32 %0, %0, %0, 1.0" : "+v"(x3));
  if constexpr (V >= 5) valu_ops<V - 4>(x0, x1, x2, x3);
}

template <int V>
__global__ void __launch_bounds__(256) wmma_f16(float* out, int iters) {
  v16h a, b;
  for (int i = 0; i < 16; ++i) { a[i] = (_Float16)(threadIdx.x * 0.001f + i); b[i] = (_Float16)(i * 0.5f); }
  v8f c[CHAINS] = {};
  float x0 = threadIdx.x, x1 = 1.f, x2 = 2.f, x3 = 3.f;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < CHAINS; ++k) {
      c[k] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c[k]);
      valu_ops<V>(x0, x1, x2, x3);
    }
  }
  float s = x0 + x1 + x2 + x3;
  for (int k = 0; k < CHAINS; ++k) for (int j = 0; j < 8; ++j) s += c[k][j];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void __launch_bounds__(256) wmma_bf16(float* out, int iters) {
  v16s a, b;
  for (int i = 0; i < 16; ++i) { a[i] = (short)(threadIdx.x + i); b[i] = (short)(i * 3); }
  v8f c[CHAINS] = {};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < CHAINS; ++k) c[k] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c[k]);
  }
  float s = 0;
  for (int k = 0; k < CHAINS; ++k) for (int j = 0; j < 8; ++j) s += c[k][j];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void __launch_bounds__(256) wmma_iu8(float* out, int iters) {
  v4i a, b;
  for (int i = 0; i < 4; ++i) { a[i] = threadIdx.x * 0x01010101 + i; b[i] = 0x02030405 + i; }
  v8i c[CHAINS] = {};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < CHAINS; ++k) c[k] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c[k], false);
  }
  int s = 0;
  for (int k = 0; k < CHAINS; ++k) for (int j = 0; j < 8; ++j) s += c[k][j];
  out[blockIdx.x * blockDim.x + threadIdx.x] = (float)s;
}

__global__ void __launch_bounds__(256) wmma_iu4(float* out, int iters) {
  v2i a, b;
  for (int i = 0; i < 2; ++i) { a[i] = threadIdx.x * 0x11111111 + i; b[i] = 0x12345671 + i; }
  v8i c[CHAINS] = {};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < CHAINS; ++k) c[k] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c[k], false);
  }
  int s = 0;
  for (int k = 0; k < CHAINS; ++k) for (int j = 0; j < 8; ++j) s += c[k][j];
  out[blockIdx.x * blockDim.x + threadIdx.x] = (float)s;
}

// one dependent chain: c = wmma(a, b, c) back to back (latency, not throughput)
__global__ void __launch_bounds__(256) wmma_dep(float* out, int iters) {
  v16h a, b;
  for (int i = 0; i < 16; ++i) { a[i] = (_Float16)(threadIdx.x * 0.001f + i); b[i] = (_Float16)(i * 0.5f); }
  v8f c = {};
  for (int it = 0; it < iters * CHAINS; ++it) c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
  float s = 0;
  for (int j = 0; j < 8; ++j) s += c[j];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// V VOPD pairs (two independent FMACs in one dual-issue instruction) per WMMA
template <int V>
__global__ void __launch_bounds__(256) wmma_vopd(float* out, int iters) {
  v16h a, b;
  for (int i = 0; i < 16; ++i) { a[i] = (_Float16)(threadIdx.x * 0.001f + i); b[i] = (_Float16)(i * 0.5f); }
  v8f c[CHAINS] = {};
  float x0 = threadIdx.x, x1 = 1.f, x2 = 2.f, x3 = 3.f, y = 0.5f, z = 0.25f, w = 0.75f, u = 0.125f;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < CHAINS; ++k) {
      c[k] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c[k]);
#pragma unroll
      for (int p = 0; p < V; ++p) {
        if (p & 1) asm volatile("v_dual_fmac_f32 %0, %2, %3 :: v_dual_fmac_f32 %1, %4, %5" : "+v"(x2), "+v"(x3) : "v"(y), "v"(z), "v"(w), "v"(u));
        else asm volatile("v_dual_fmac_f32 %0, %2, %3 :: v_dual_fmac_f32 %1, %4, %5" : "+v"(x0), "+v"(x1) : "v"(y), "v"(z), "v"(w), "v"(u));
      }
    }
  }
  float s = x0 + x1 + x2 + x3;
  for (int k = 0; k < CHAINS; ++k) for (int j = 0; j < 8; ++j) s += c[k][j];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// V ds_load_b128 per WMMA from a wave-private LDS slice (conflict-free, results summed)
template <int V>
__global__ void __launch_bounds__(256) wmma_lds(float* out, int iters) {
  __shared__ float4 lds[256 * 4 + 64];
  v16h a, b;
  for (int i = 0; i < 16; ++i) { a[i] = (_Float16)(threadIdx.x * 0.001f + i); b[i] = (_Float16)(i * 0.5f); }
  for (int i = 0; i < 4; ++i) lds[threadIdx.x + 256 * i] = make_float4(threadIdx.x, i, 1, 2);
  __syncthreads();
  v8f c[CHAINS] = {};
  float4 acc = make_float4(0, 0, 0, 0);
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < CHAINS; ++k) {
      c[k] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c[k]);
#pragma unroll
      for (int p = 0; p < V; ++p) {
        float4 r;
        asm volatile("ds_load_b128 %0, %1 offset:%2" : "=v"(r) : "v"((unsigned)(threadIdx.x * 16)), "i"(4096 * (p & 3)));
        acc.x += r.x;
      }
    }
  }
  asm volatile("s_waitcnt lgkmcnt(0)");
  float s = acc.x;
  for (int k = 0; k < CHAINS; ++k) for (int j = 0; j < 8; ++j) s += c[k][j];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main(int argc, char** argv) {
  const char* v = argc > 1 ? argv[1] : "f16";
  const int iters = argc > 2 ? atoi(argv[2]) : 4096;
  const int blocks = getenv("WR_BLOCKS") ? atoi(getenv("WR_BLOCKS")) : 160, threads = getenv("WR_THREADS") ? atoi(getenv("WR_THREADS")) : 256;  // 160 x 8 waves over 80 SIMDs: 16 waves/SIMD requested
  float* out; hipMalloc(&out, sizeof(float) * blocks * threads);
  auto run = [&](auto kern) {
    hipLaunchKernelGGL(kern, dim3(blocks), dim3(threads), 0, 0, out, iters);
    hipDeviceSynchronize();
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a);
    hipLaunchKernelGGL(kern, dim3(blocks), dim3(threads), 0, 0, out, iters);
    hipEventRecord(b); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b);
    double wmma = (double)blocks * (threads / 32) * iters * CHAINS;
    printf("%s iters=%d wmma=%.3e ms=%.3f  wmma/s/SIMD=%.3e\n", v, iters, wmma, ms, wmma / 80 / (ms * 1e-3));
  };
  if (!strcmp(v, "f16")) run(wmma_f16<0>);
  else if (!strcmp(v, "f16v1")) run(wmma_f16<1>);
  else if (!strcmp(v, "f16v2")) run(wmma_f16<2>);
  else if (!strcmp(v, "f16v4")) run(wmma_f16<4>);
  else if (!strcmp(v, "f16v8")) run(wmma_f16<8>);
  else if (!strcmp(v, "f16v16")) run(wmma_f16<16>);
  else if (!strcmp(v, "f16v32")) run(wmma_f16<32>);
  else if (!strcmp(v, "bf16")) run(wmma_bf16);
  else if (!strcmp(v, "iu8")) run(wmma_iu8);
  else if (!strcmp(v, "iu4")) run(wmma_iu4);
  else if (!strcmp(v, "dep")) run(wmma_dep);
  else if (!strcmp(v, "vopd1")) run(wmma_vopd<1>);
  else if (!strcmp(v, "vopd4")) run(wmma_vopd<4>);
  else if (!strcmp(v, "vopd8")) run(wmma_vopd<8>);
  else if (!strcmp(v, "lds1")) run(wmma_lds<1>);
  else if (!strcmp(v, "lds2")) run(wmma_lds<2>);
  else if (!strcmp(v, "lds4")) run(wmma_lds<4>);
  else { fprintf(stderr, "unknown variant\n"); return 1; }
  return 0;
}
