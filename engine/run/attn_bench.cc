// attn_bench: times attention HALs (or any HAL whose bindings are listed) interleaved in one process.
//
// usage: attn_bench <iters> <nbuf> <buf>... <hal>@<gx>,<gy>[,<out>]...
//   buf: f:<path>             the file's bytes
//        z:<bytes>            zeros
//        h:<bytes>:<seed>     f16 in [-1, 1)
//        s:<bytes>:<seed>     f32 in [-1, 1)
//        c:<count>:<value>    f32 constant
//        p:<pages>[:<window>] i32 identity page table (window: page j -> j % window, the K / V reads stay in window pages)
//   <out>: the binding whose bytes are hashed after each HAL's first dispatch (default 4, the attention output)
// A HAL that binds fewer buffers gets the first ones. ATTN_BENCH_ZERO=<i>: buffer i must be all zero after the run
// (a self-resetting work queue counter).
// ATTN_BENCH_CHAIN=1: only HAL 0's output is zeroed before its first dispatch (HAL k > 0 reads what HAL k - 1 wrote).
//
// Do not compute the buffer sizes by hand: run this through tools/attn_bench.py, which derives them from the kernel's
// configs. An access past an allocation does not fault on this target: the shader hangs and the box resets.
// Prints one line per HAL: "hal <i> <path> hash <fnv64> ms <median>". Rounds go HAL 0, 1, ..., so a clock change hits
// every HAL alike; counter cycles come from the HRX profile (HRX_PROFILE_MODE=counters), not from these host times.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "model/loom_runtime.hpp"

using namespace yah::model;

static uint16_t F16(float f) {  // round to nearest even; [-1, 1) only, so no overflow or subnormal handling
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
  if (exp <= 0) return static_cast<uint16_t>(sign);
  uint32_t mant = x & 0x7fffffu;
  uint32_t h = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  const uint32_t rest = mant & 0x1fffu;
  if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
  return static_cast<uint16_t>(h);
}

static std::vector<std::string> Split(const std::string& s, char c) {
  std::vector<std::string> out;
  size_t a = 0;
  for (size_t b; (b = s.find(c, a)) != std::string::npos; a = b + 1) out.push_back(s.substr(a, b - a));
  out.push_back(s.substr(a));
  return out;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: attn_bench <iters> <nbuf> <buf>... <hal>@<gx>,<gy>[,<out>]...\n");
    return 2;
  }
  const int iters = atoi(argv[1]);
  const int nbuf = atoi(argv[2]);
  if (iters < 1 || iters > 10000 || nbuf < 1 || 3 + nbuf >= argc) {
    std::fprintf(stderr, "attn_bench: bad iters / nbuf\n");
    return 2;
  }
  LoomDevice gpu;
  std::vector<LoomBuffer> bufs;
  std::vector<size_t> sizes;
  for (int i = 0; i < nbuf; ++i) {
    const auto f = Split(argv[3 + i], ':');
    std::vector<uint8_t> host;
    if (f[0] == "f" && f.size() == 2) {
      FILE* fp = std::fopen(f[1].c_str(), "rb");
      if (!fp) {
        std::fprintf(stderr, "attn_bench: cannot open %s\n", f[1].c_str());
        return 2;
      }
      std::fseek(fp, 0, SEEK_END);
      host.resize(static_cast<size_t>(std::ftell(fp)));
      std::fseek(fp, 0, SEEK_SET);
      if (std::fread(host.data(), 1, host.size(), fp) != host.size()) return 2;
      std::fclose(fp);
    } else if ((f[0] == "z" && f.size() == 2) || ((f[0] == "h" || f[0] == "s") && f.size() == 3)) {
      host.assign(strtoull(f[1].c_str(), nullptr, 10), 0);
      if (f[0] != "z") {
        uint32_t seed = static_cast<uint32_t>(strtoul(f[2].c_str(), nullptr, 10)) * 2654435761u + 1u;
        auto next = [&] {
          seed = seed * 1664525u + 1013904223u;
          return static_cast<float>(seed >> 8) / 8388608.0f - 1.0f;
        };
        if (f[0] == "h")
          for (size_t j = 0; j + 1 < host.size(); j += 2) {
            const uint16_t v = F16(next());
            std::memcpy(&host[j], &v, 2);
          }
        else
          for (size_t j = 0; j + 3 < host.size(); j += 4) {
            const float v = next();
            std::memcpy(&host[j], &v, 4);
          }
      }
    } else if (f[0] == "c" && f.size() == 3) {
      const size_t n = strtoull(f[1].c_str(), nullptr, 10);
      const float v = strtof(f[2].c_str(), nullptr);
      host.resize(n * 4);
      for (size_t j = 0; j < n; ++j) std::memcpy(&host[4 * j], &v, 4);
    } else if (f[0] == "p" && (f.size() == 2 || f.size() == 3)) {
      const size_t n = strtoull(f[1].c_str(), nullptr, 10);
      const size_t w = f.size() == 3 ? strtoull(f[2].c_str(), nullptr, 10) : n;   // p:<pages>:<window>: page j -> j % window
      if (w < 1 || w > n) {
        std::fprintf(stderr, "attn_bench: bad page window %s\n", argv[3 + i]);
        return 2;
      }
      host.resize(n * 4);
      for (size_t j = 0; j < n; ++j) {
        const int32_t v = static_cast<int32_t>(j % w);
        std::memcpy(&host[4 * j], &v, 4);
      }
    } else {
      std::fprintf(stderr, "attn_bench: bad buffer spec %s\n", argv[3 + i]);
      return 2;
    }
    if (host.empty()) {
      std::fprintf(stderr, "attn_bench: empty buffer %s\n", argv[3 + i]);
      return 2;
    }
    bufs.push_back(gpu.Allocate(host.size()));
    sizes.push_back(host.size());
    gpu.H2D(bufs.back(), host.data(), host.size());
  }
  struct Hal {
    std::string path;
    LoomExecutable exe;
    uint32_t ordinal = 0, gx = 0, gy = 0, nb = 0;
    int out = 4;
    std::vector<double> ms;
  };
  std::vector<Hal> hals;
  for (int i = 3 + nbuf; i < argc; ++i) {
    const std::string a = argv[i];
    const size_t at = a.rfind('@');
    if (at == std::string::npos) {
      std::fprintf(stderr, "attn_bench: %s has no @<gx>,<gy>\n", a.c_str());
      return 2;
    }
    const auto g = Split(a.substr(at + 1), ',');
    Hal h;
    h.path = a.substr(0, at);
    h.gx = static_cast<uint32_t>(atoi(g[0].c_str()));
    h.gy = g.size() > 1 ? static_cast<uint32_t>(atoi(g[1].c_str())) : 1;
    if (g.size() > 2) h.out = atoi(g[2].c_str());
    h.exe = gpu.Load(h.path);
    h.ordinal = h.exe.OrdinalOrZero(h.exe.names.empty() ? "" : h.exe.names[0]);
    const uint32_t nb = h.exe.BindingCount(h.ordinal);
    if (nb > static_cast<uint32_t>(nbuf) || nb == 0 || !h.gx || !h.gy || h.out < 0 || h.out >= static_cast<int>(nb)) {
      std::fprintf(stderr, "attn_bench: REFUSING %s: export binds %u buffers, %d given (grid %ux%u, out %d)\n",
                   h.path.c_str(), nb, nbuf, h.gx, h.gy, h.out);
      return 3;
    }
    h.nb = nb;
    hals.push_back(std::move(h));
  }
  std::vector<hrx_buffer_ref_t> refs;
  for (int i = 0; i < nbuf; ++i) refs.push_back({bufs[i].handle, 0, sizes[i]});
  // One synchronized dispatch per HAL first: a hang shows here, and the hash is of a single run on zeroed output.
  for (size_t k = 0; k < hals.size(); ++k) {
    Hal& h = hals[k];
    std::vector<uint8_t> zero(sizes[h.out], 0), back(sizes[h.out]);
    if (k == 0 || !std::getenv("ATTN_BENCH_CHAIN")) gpu.H2D(bufs[h.out], zero.data(), zero.size());
    const uint32_t ws = h.exe.WorkgroupSize(h.ordinal);
    gpu.Dispatch(h.exe, h.ordinal, LoomDevice::Config(h.gx, h.gy, 1, ws ? ws : 256, 1, 1), nullptr, 0, refs.data(),
                 h.nb);
    gpu.Synchronize();
    gpu.D2H(bufs[h.out], back.data(), back.size());
    uint64_t fnv = 1469598103934665603ull;
    for (uint8_t b : back) fnv = (fnv ^ b) * 1099511628211ull;
    std::printf("first %zu %016llx\n", k, static_cast<unsigned long long>(fnv));
    if (const char* dir = std::getenv("ATTN_BENCH_DUMP")) {   // the first dispatch's output, as <dir>/out<k>.bin
      FILE* fp = std::fopen((std::string(dir) + "/out" + std::to_string(k) + ".bin").c_str(), "wb");
      if (!fp || std::fwrite(back.data(), 1, back.size(), fp) != back.size()) return 2;
      std::fclose(fp);
    }
  }
  for (int r = 0; r < iters; ++r)
    for (Hal& h : hals) {
      const uint32_t ws = h.exe.WorkgroupSize(h.ordinal);
      const auto t0 = std::chrono::steady_clock::now();
      gpu.Dispatch(h.exe, h.ordinal, LoomDevice::Config(h.gx, h.gy, 1, ws ? ws : 256, 1, 1), nullptr, 0, refs.data(),
                   h.nb);
      gpu.Synchronize();
      h.ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
  if (const char* z = std::getenv("ATTN_BENCH_ZERO")) {
    const int i = atoi(z);
    if (i < 0 || i >= nbuf) return 2;
    std::vector<uint8_t> back(sizes[i]);
    gpu.D2H(bufs[i], back.data(), back.size());
    const bool zero = std::all_of(back.begin(), back.end(), [](uint8_t b) { return b == 0; });
    std::printf("zero %d %s\n", i, zero ? "ok" : "NOT ZERO");
    if (!zero) return 4;
  }
  for (size_t k = 0; k < hals.size(); ++k) {
    auto v = hals[k].ms;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    std::printf("hal %zu %s ms %.4f\n", k, hals[k].path.c_str(), v[v.size() / 2]);
  }
  return 0;
}
