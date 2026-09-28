// Launch-submission microbenchmark, built to be run under either HIP runtime.
//
// The prefill issues 1625 dispatches for a 2048-token pass and the model-level
// trace leaves ~4% of the wall clock as GPU idle. This separates submission cost
// from execution cost: an empty kernel makes the loop CPU/submission bound, so
// N launches measures dispatch overhead directly, and the same N captured into a
// graph measures the graph path. HRX maps hipStreamBeginCapture onto its native
// graph, so this is reachable without touching any kernel.
//
// Usage: launch_bench [N]
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

__global__ void EmptyKernel(int* p) {
  if (p != nullptr) *p = 1;
}

static double ms_since(const std::chrono::steady_clock::time_point& a) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - a)
      .count();
}

int main(int argc, char** argv) {
  const int N = argc > 1 ? std::atoi(argv[1]) : 2000;
  int* d = nullptr;
  hipMalloc(&d, sizeof(int));
  hipMemset(d, 0, sizeof(int));

  for (int i = 0; i < 100; ++i) EmptyKernel<<<1, 64>>>(d);
  hipDeviceSynchronize();

  // Raw stream submission: N launches, then wait.
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) EmptyKernel<<<1, 64>>>(d);
  hipDeviceSynchronize();
  const double raw = ms_since(t0);

  // Same work captured into a graph and launched once.
  hipStream_t s = nullptr;
  hipStreamCreate(&s);
  hipGraph_t g = nullptr;
  hipGraphExec_t ge = nullptr;
  const hipError_t cap = hipStreamBeginCapture(s, hipStreamCaptureModeGlobal);
  if (cap != hipSuccess) {
    std::printf("N=%d raw=%.4f ms (%.4f us/launch)  graph=UNSUPPORTED (%s)\n", N,
                raw, raw * 1000.0 / N, hipGetErrorString(cap));
    return 0;
  }
  for (int i = 0; i < N; ++i) EmptyKernel<<<1, 64, 0, s>>>(d);
  hipStreamEndCapture(s, &g);
  const hipError_t inst = hipGraphInstantiate(&ge, g, nullptr, nullptr, 0);
  if (inst != hipSuccess) {
    std::printf("N=%d raw=%.4f ms (%.4f us/launch)  graph=INSTANTIATE FAILED (%s)\n",
                N, raw, raw * 1000.0 / N, hipGetErrorString(inst));
    return 0;
  }
  hipGraphLaunch(ge, s);
  hipStreamSynchronize(s);  // warm
  auto t1 = std::chrono::steady_clock::now();
  hipGraphLaunch(ge, s);
  hipStreamSynchronize(s);
  const double gr = ms_since(t1);

  std::printf(
      "N=%d raw=%.4f ms (%.4f us/launch)  graph=%.4f ms (%.4f us/launch)  speedup=%.2fx\n",
      N, raw, raw * 1000.0 / N, gr, gr * 1000.0 / N, raw / gr);
  return 0;
}
