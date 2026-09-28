// Kernel-boundary cost vs grid-sync cost, for the persistent-kernel question.
//
// gridsync_bench prices one grid sync at ~0.71 us. launch_bench showed launch
// submission is ~2.4-3.1 us, which cannot explain the ~4% GPU idle in a prefill.
// The remaining candidate is the kernel *boundary*: drain, retire, relaunch,
// ramp, and the dependency wait. This measures it directly by running identical
// work two ways and comparing:
//
//   chain   - N dependent kernel launches, each reading what the last wrote
//   persist - one cooperative kernel doing N iterations with a grid sync between
//
// The difference is what a persistent design would actually recover.
//
// Usage: gridsync_bench [blocks] [iters] [workloops]
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

__device__ unsigned int g_arrive = 0;
__device__ unsigned int g_sense = 0;

__device__ __forceinline__ void GridBarrier(int nblocks) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned int sense = atomicAdd(&g_sense, 0u);
    __threadfence();
    const unsigned int ticket = atomicAdd(&g_arrive, 1u);
    if (ticket == (unsigned int)(nblocks - 1)) {
      g_arrive = 0;
      __threadfence();
      atomicAdd(&g_sense, 1u);
    } else {
      while (atomicAdd(&g_sense, 0u) == sense) {
        __threadfence();
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ float Work(float acc, int workloops) {
  for (int i = 0; i < workloops; ++i) acc = fmaf(acc, 1.0000001f, 1.0e-7f);
  return acc;
}

// One unit of work, reading and writing its own slot so launches are dependent.
__global__ void ChainKernel(float* buf, int workloops) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  buf[i] = Work(buf[i], workloops);
}

__global__ void PersistentKernel(float* buf, int iters, int workloops,
                                 int nblocks) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = buf[i];
  for (int it = 0; it < iters; ++it) {
    acc = Work(acc, workloops);
    buf[i] = acc;
    GridBarrier(nblocks);
  }
}

int main(int argc, char** argv) {
  const int blocks = argc > 1 ? std::atoi(argv[1]) : 40;
  const int iters = argc > 2 ? std::atoi(argv[2]) : 200;
  const int workloops = argc > 3 ? std::atoi(argv[3]) : 20000;
  const int threads = 256;

  float* buf = nullptr;
  hipMalloc(&buf, (size_t)blocks * threads * sizeof(float));
  hipMemset(buf, 0, (size_t)blocks * threads * sizeof(float));

  // chain: N dependent launches on one stream
  for (int i = 0; i < 5; ++i) ChainKernel<<<blocks, threads>>>(buf, workloops);
  hipDeviceSynchronize();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) ChainKernel<<<blocks, threads>>>(buf, workloops);
  hipDeviceSynchronize();
  auto t1 = std::chrono::steady_clock::now();
  const double chain_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  // persist: same work, one cooperative kernel, grid sync between iterations
  int iters_arg = iters, work_arg = workloops, blocks_arg = blocks;
  void* args[] = {&buf, &iters_arg, &work_arg, &blocks_arg};
  hipError_t e = hipLaunchCooperativeKernel((void*)PersistentKernel,
                                            dim3(blocks), dim3(threads), args, 0, 0);
  if (e != hipSuccess) {
    std::printf("cooperative launch UNSUPPORTED: %s\n", hipGetErrorString(e));
    return 0;
  }
  hipDeviceSynchronize();
  auto t2 = std::chrono::steady_clock::now();
  e = hipLaunchCooperativeKernel((void*)PersistentKernel, dim3(blocks),
                                 dim3(threads), args, 0, 0);
  hipDeviceSynchronize();
  auto t3 = std::chrono::steady_clock::now();
  if (e != hipSuccess) {
    std::printf("persistent FAILED: %s\n", hipGetErrorString(e));
    return 0;
  }
  const double persist_ms =
      std::chrono::duration<double, std::milli>(t3 - t2).count();

  std::printf(
      "blocks=%d iters=%d work=%d  chain=%.3f ms (%.3f us/iter)  persist=%.3f ms (%.3f us/iter)  boundary saved=%.3f us/iter  speedup=%.3fx\n",
      blocks, iters, workloops, chain_ms, chain_ms * 1000.0 / iters, persist_ms,
      persist_ms * 1000.0 / iters,
      (chain_ms - persist_ms) * 1000.0 / iters, chain_ms / persist_ms);
  return 0;
}
