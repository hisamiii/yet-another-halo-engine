// Synthetic replica of the prefill K loop's instruction pattern.
//
// The production kernel stalls ~87% of cycles with LDS units at 26-45% and
// issue slots at 12%, so the question is whether the K loop's pattern itself
// caps out -- 12 ds_read_b128 feeding 8 WMMA per step -- or whether the
// surrounding work is the deficit. This reproduces the mix with ideal,
// conflict-free LDS reads and nothing else, so the answer is clean.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>

using v16bf = _Float16 __attribute__((ext_vector_type(16)));
using v8f = float __attribute__((ext_vector_type(8)));

__global__ void KLoopReplica(int* sink, int steps) {
  __shared__ _Float16 buf[32768];  // 64 KiB, same as the real kernel
  for (int i = threadIdx.x; i < 32768; i += blockDim.x)
    buf[i] = _Float16(0.001f);
  __syncthreads();
  const int lane = threadIdx.x & 31;
  v8f acc[8] = {};
  for (int s = 0; s < steps; ++s) {
    v16bf frag[6];
#pragma unroll
    for (int f = 0; f < 6; ++f) {
      // Fragment f occupies 1024 bytes; lane L reads two consecutive
      // 16-byte chunks, which is the canonical conflict-free ds_read_b128
      // pattern. 64 fragments fill the 64 KiB buffer exactly.
      //
      // The index is made opaque on purpose. (s*6+f)&63 has period 32 steps,
      // so without the barrier the compiler proves the operands repeat, folds
      // 4000 iterations into a multiply, and reports 35x the ceiling.
      int idx = (s * 6 + f) & 63;
      asm volatile("" : "+v"(idx));
      const int fb = (idx * 512) + (lane * 8);
      union {
        v16bf v;
        uint4 q[2];
      } bits;
      bits.q[0] = *reinterpret_cast<const uint4*>(buf + fb);
      bits.q[1] = *reinterpret_cast<const uint4*>(buf + fb + 256);
      frag[f] = bits.v;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        acc[(i * 4) + j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            frag[i], frag[2 + j], acc[(i * 4) + j]);
      }
    }
  }
  float t = 0.0F;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int e = 0; e < 8; ++e) t += acc[i][e];
  }
  if (t == 123456789.0F) sink[0] = 1;
}

// Same instruction mix plus the register->LDS commit: 32 bytes per thread
// every fourth step, which is what the production kernel's commit writes per
// stage. If the K loop is LDS-bandwidth-bound and the commit shares that
// bandwidth, this should drop the achieved rate by about the write share.
__global__ void KLoopWithCommit(int* sink, int steps) {
  __shared__ _Float16 buf[32768];
  for (int i = threadIdx.x; i < 32768; i += blockDim.x)
    buf[i] = _Float16(0.001f);
  __syncthreads();
  const int lane = threadIdx.x & 31;
  v8f acc[8] = {};
  for (int s = 0; s < steps; ++s) {
    v16bf frag[6];
#pragma unroll
    for (int f = 0; f < 6; ++f) {
      int idx = (s * 6 + f) & 63;
      asm volatile("" : "+v"(idx));
      const int fb = (idx * 512) + (lane * 8);
      union {
        v16bf v;
        uint4 q[2];
      } bits;
      bits.q[0] = *reinterpret_cast<const uint4*>(buf + fb);
      bits.q[1] = *reinterpret_cast<const uint4*>(buf + fb + 256);
      frag[f] = bits.v;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        acc[(i * 4) + j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            frag[i], frag[2 + j], acc[(i * 4) + j]);
      }
    }
    if ((s & 3) == 0) {
      // 32 bytes per thread, one commit per four steps.
      const int wb = ((s >> 2) & 15) * 1024 + (threadIdx.x * 8);
      uint4 w0 = make_uint4(s, s + 1, s + 2, s + 3);
      *reinterpret_cast<uint4*>(buf + wb) = w0;
      *reinterpret_cast<uint4*>(buf + wb + 256) = w0;
    }
  }
  float t = 0.0F;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int e = 0; e < 8; ++e) t += acc[i][e];
  }
  if (t == 123456789.0F) sink[0] = 1;
}

template<typename K>
void Run(K kernel, int* sink, int steps, const char* name) {
  for (int blocks : {40, 160, 640}) {
    for (int i = 0; i < 2; ++i) kernel<<<blocks, 256>>>(sink, steps);
    hipDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 5; ++i) kernel<<<blocks, 256>>>(sink, steps);
    hipDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;
    const double waves = double(blocks) * 8.0;
    const double macs = waves * steps * 8.0 * 4096.0;
    const double tmacs = macs / (ms * 1e9);
    std::printf("%-8s blocks=%4d  TMAC/s=%6.2f  TFLOPS=%6.2f  (%.0f%% of ceiling)\n",
                name, blocks, tmacs, 2.0 * tmacs, 100.0 * tmacs / 24.17);
  }
}

int main() {
  int* sink = nullptr;
  hipMalloc(&sink, 64);
  const int steps = 4000;
  std::printf("WMMA fp16 reference ceiling: 24.17 TMAC/s (48.35 TFLOPS)\n");
  std::printf("production K loop: ~30-32 TFLOPS over the whole kernel\n\n");
  Run(KLoopReplica, sink, steps, "reads");
  Run(KLoopWithCommit, sink, steps, "commit");
  return 0;
}
