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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x)                                                                             \
  do {                                                                                       \
    hipError_t e_ = (x);                                                                     \
    if (e_ != hipSuccess) {                                                                  \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
      std::exit(1);                                                                          \
    }                                                                                        \
  } while (0)

static std::vector<char> ReadFile(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<char> v(n);
  if (std::fread(v.data(), 1, n, f) != static_cast<size_t>(n)) std::exit(1);
  std::fclose(f);
  return v;
}

// The kernel descriptor's kernarg_size (amd_kernel_code_t-style .kd symbol:
// group_segment u32 @0, private_segment u32 @4, kernarg_size u32 @8), read
// straight from the ELF. Returns -1 if the symbol is missing.
static long KernargSize(const std::vector<char>& elf, const std::string& kernel) {
  auto u16 = [&](size_t o) {
    uint16_t v;
    std::memcpy(&v, elf.data() + o, 2);
    return v;
  };
  auto u32 = [&](size_t o) {
    uint32_t v;
    std::memcpy(&v, elf.data() + o, 4);
    return v;
  };
  auto u64 = [&](size_t o) {
    uint64_t v;
    std::memcpy(&v, elf.data() + o, 8);
    return v;
  };
  if (elf.size() < 64 || std::memcmp(elf.data(),
                                     "\x7f"
                                     "ELF",
                                     4) != 0)
    return -1;
  const uint64_t shoff = u64(40);
  const unsigned shentsize = u16(58), shnum = u16(60);
  auto sh = [&](unsigned i) { return shoff + static_cast<uint64_t>(i) * shentsize; };
  const std::string want = kernel + ".kd";
  for (unsigned i = 0; i < shnum; ++i) {
    if (u32(sh(i) + 4) != 2) continue;  // SHT_SYMTAB
    const uint64_t sym_off = u64(sh(i) + 24), sym_size = u64(sh(i) + 32);
    const uint64_t str_off = u64(sh(u32(sh(i) + 40)) + 24);
    for (uint64_t o = sym_off; o + 24 <= sym_off + sym_size; o += 24) {
      if (std::strcmp(elf.data() + str_off + u32(o), want.c_str()) != 0) continue;
      const unsigned shndx = u16(o + 6);
      const uint64_t value = u64(o + 8);
      const uint64_t file = u64(sh(shndx) + 24) + (value - u64(sh(shndx) + 16));
      return u32(file + 8);
    }
  }
  return -1;
}

int main(int argc, char** argv) {
  if (argc < 8) {
    std::fprintf(stderr, "usage: loomhip <hsaco> <kernel> <gx> <gy> <block_x> <iters> <buf>...\n");
    return 1;
  }
  const std::string hsaco = argv[1], kernel = argv[2];
  const unsigned gx = std::atoi(argv[3]), gy = std::atoi(argv[4]), bx = std::atoi(argv[5]);
  const int iters = std::atoi(argv[6]);

  // Refuse a launch whose buffer count does not match the kernel's argument
  // block. On 2026-10-01 an IQ3_S kernel (6 bindings, kernarg_size 48) was
  // launched with 5 buffers: it read its sixth pointer past the 40 given bytes,
  // stored through it, and hung the GPU (ring timeouts, GPU reset) -- on this
  // target a wild access hangs instead of faulting.
  const long kas = KernargSize(ReadFile(hsaco), kernel);
  const long given = static_cast<long>(argc - 7) * 8;
  if (kas < 0) {
    std::fprintf(stderr, "loomhip: no %s.kd in %s; refusing to launch\n", kernel.c_str(), hsaco.c_str());
    return 2;
  }
  if (kas != given) {
    std::fprintf(stderr,
                 "loomhip: %s takes %ld bytes of kernel arguments (%ld buffers), %d given; refusing to launch\n",
                 kernel.c_str(), kas, kas / 8, argc - 7);
    return 2;
  }

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
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &kbytes,
                    HIP_LAUNCH_PARAM_END};
  auto launch = [&] { CHECK(hipModuleLaunchKernel(fn, gx, gy, 1, bx, 1, 1, 0, nullptr, nullptr, config)); };
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
