// loomhip: launch a Loom-compiled kernel (hsaco from loom-compile) through the
// HIP runtime, so rocprofv3 (kernel trace, PMC incl. occupancy, ATT per-wave)
// can profile Loom code with the same tools as the HIP reference kernels.
// rocprofv3 cannot see dispatches made through HRX.
//
// usage: loomhip <hsaco> <kernel> <gx> <gy> <block_x> <iters> <buf>...
//   buf: f:<path>            device buffer initialised from a raw file
//        z:<bytes>           zero-filled device buffer
//        o:<bytes>:<path>    zero-filled, written to <path> after the first launch
// Kernel arguments are the buffers' device pointers, in order (Loom's ABI for
// kernels without scalar arguments: 8 bytes per binding).
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x)                                                              \
  do {                                                                        \
    hipError_t e_ = (x);                                                      \
    if (e_ != hipSuccess) {                                                   \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,          \
                   hipGetErrorString(e_));                                    \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

static std::vector<char> ReadFile(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<char> v(n);
  if (std::fread(v.data(), 1, n, f) != static_cast<size_t>(n)) std::exit(1);
  std::fclose(f);
  return v;
}

int main(int argc, char** argv) {
  if (argc < 8) {
    std::fprintf(stderr, "usage: loomhip <hsaco> <kernel> <gx> <gy> <block_x> <iters> <buf>...\n");
    return 1;
  }
  const std::string hsaco = argv[1], kernel = argv[2];
  const unsigned gx = std::atoi(argv[3]), gy = std::atoi(argv[4]), bx = std::atoi(argv[5]);
  const int iters = std::atoi(argv[6]);

  hipModule_t mod;
  CHECK(hipModuleLoad(&mod, hsaco.c_str()));
  hipFunction_t fn;
  CHECK(hipModuleGetFunction(&fn, mod, kernel.c_str()));

  std::vector<void*> ptrs;
  std::vector<std::pair<int, std::string>> dumps;
  std::vector<size_t> sizes;
  for (int i = 7; i < argc; ++i) {
    std::string s = argv[i];
    void* d = nullptr;
    size_t bytes = 0;
    if (s.rfind("f:", 0) == 0) {
      auto v = ReadFile(s.substr(2));
      bytes = v.size();
      CHECK(hipMalloc(&d, bytes));
      CHECK(hipMemcpy(d, v.data(), bytes, hipMemcpyHostToDevice));
    } else if (s.rfind("z:", 0) == 0 || s.rfind("o:", 0) == 0) {
      std::string rest = s.substr(2);
      size_t c = rest.find(':');
      bytes = std::stoull(rest.substr(0, c));
      CHECK(hipMalloc(&d, bytes));
      CHECK(hipMemset(d, 0, bytes));
      if (s[0] == 'o') dumps.push_back({static_cast<int>(ptrs.size()), rest.substr(c + 1)});
    } else {
      std::fprintf(stderr, "bad buffer spec %s\n", s.c_str());
      return 1;
    }
    ptrs.push_back(d);
    sizes.push_back(bytes);
  }
  std::vector<char> kernarg(ptrs.size() * sizeof(void*));
  std::memcpy(kernarg.data(), ptrs.data(), kernarg.size());
  size_t kbytes = kernarg.size();
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg.data(),
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &kbytes, HIP_LAUNCH_PARAM_END};
  auto launch = [&] {
    CHECK(hipModuleLaunchKernel(fn, gx, gy, 1, bx, 1, 1, 0, nullptr, nullptr, config));
  };
  launch();
  CHECK(hipDeviceSynchronize());
  for (auto& [idx, path] : dumps) {
    std::vector<char> h(sizes[idx]);
    CHECK(hipMemcpy(h.data(), ptrs[idx], sizes[idx], hipMemcpyDeviceToHost));
    FILE* f = std::fopen(path.c_str(), "wb");
    std::fwrite(h.data(), 1, h.size(), f);
    std::fclose(f);
  }
  // wall clock around a device sync (hipEvent timing on the null stream read 0)
  CHECK(hipDeviceSynchronize());
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) launch();
  CHECK(hipDeviceSynchronize());
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("loomhip %s: %d launches, %.3f ms/launch\n", kernel.c_str(), iters, iters ? ms / iters : 0.f);
  return 0;
}
