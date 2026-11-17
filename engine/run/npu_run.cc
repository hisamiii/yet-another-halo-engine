// Runs one NPU image invocation through LoomNpu: the in-engine stand-in for HRX's iree-xdna-run, which HRX main no
// longer has (tools/npu_gemm_check.py and npu_attn_check.py drive it).
//
// usage: npu_run <image.xdna> <entry> <columns> <binding>=<output file>[,...] <binding file>...
//
// Runs the entry's first invocation (iree-xdna-run --invocation_count=1). Each binding file becomes host pages the NPU
// reads and writes, in order; after the run, each listed binding's bytes (its file's length) go to its output file. A
// context width the image does not admit fails with the loader's message ("does not match the admitted native
// context"): the caller retries a wider one.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <string>
#include <utility>
#include <vector>

#include "model/loom_npu.hpp"
#include "model/loom_runtime.hpp"

using yah::model::LoomDevice;
using yah::model::LoomError;
using yah::model::LoomNpu;

namespace {

std::vector<char> ReadAll(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw LoomError("cannot open " + path);
  std::vector<char> data(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(data.data(), static_cast<std::streamsize>(data.size()));
  return data;
}

// The host pages are cached write-back and the NPU's DMA does not snoop the CPU caches (LoomNpu flushes its own
// command storage the same way): write the copied-in bytes back before the run, and drop the lines before reading
// what the NPU wrote.
void FlushLines(const void* p, std::size_t n) {
  const auto* c = static_cast<const char*>(p);
  for (std::size_t i = 0; i < n; i += 64) _mm_clflushopt(const_cast<char*>(c + i));
  _mm_sfence();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr,
                 "usage: npu_run <image.xdna> <entry> <columns> <binding>=<output file>[,...] <binding file>...\n");
    return 2;
  }
  try {
    const std::string image = argv[1], entry = argv[2];
    const std::uint32_t columns = std::atoi(argv[3]);
    std::vector<std::pair<std::size_t, std::string>> outputs;
    for (std::string spec = argv[4]; !spec.empty();) {
      const std::size_t comma = spec.find(','), eq = spec.find('=');
      const std::string one = spec.substr(0, comma);
      if (eq == std::string::npos || eq > one.size()) throw LoomError("output spec " + one + ": want <binding>=<file>");
      outputs.emplace_back(std::stoul(one.substr(0, eq)), one.substr(eq + 1));
      spec = comma == std::string::npos ? "" : spec.substr(comma + 1);
    }
    std::vector<std::vector<char>> files;
    for (int i = 5; i < argc; ++i) files.push_back(ReadAll(argv[i]));
    for (const auto& [b, path] : outputs)
      if (b >= files.size()) throw LoomError("output binding " + std::to_string(b) + " out of range");

    LoomDevice gpu;
    LoomNpu npu(gpu, columns);
    std::vector<LoomNpu::View> views;
    for (const auto& f : files) {
      LoomNpu::Shared& s = npu.CreateShared(f.size() ? f.size() : 4, true);
      std::memcpy(s.host, f.data(), f.size());
      FlushLines(s.host, s.bytes);
      views.push_back({&s, 0, f.size()});
    }
    // NPU_RUN_REPEAT=<n>: n runs, each one's host wall time (load + run) printed; timing by difference of two
    // problem sizes cancels the fixed part
    const char* rep = std::getenv("NPU_RUN_REPEAT");
    for (int i = 0, n = rep ? std::max(1, std::atoi(rep)) : 1; i < n; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      if (const char* spec = std::getenv("YAH_NPU_PROF")) {   // the counters (NpuProf spec c0,c1,c2,c3/m0,m1)
        const std::vector<std::uint32_t> v =   // (NPU_RUN_GATED=<calls>: one gated job of that many calls)
            std::getenv("NPU_RUN_GATED")
                ? npu.RunGatedOnce(image, entry, views, std::max(1, std::atoi(std::getenv("NPU_RUN_GATED"))))
                : npu.RunOnceProf(image, entry, views);
        std::printf("npu_run: prof events %s\n", spec);
        std::size_t at = 0;
        for (std::uint32_t c = 0; c < columns; ++c)
          for (std::uint32_t r = 0; r < 4; ++r, at += 6)
            std::printf("npu_run: prof tile %u %u %u %u %u %u %u %u\n", c, r + 2, v[at], v[at + 1], v[at + 2],
                        v[at + 3], v[at + 4], v[at + 5]);
        for (std::uint32_t c = 0; c < columns; ++c, at += 6)
          std::printf("npu_run: prof memtile %u %u %u %u %u shim %u %u\n", c, v[at], v[at + 1], v[at + 2], v[at + 3],
                      v[at + 4], v[at + 5]);
      } else if (std::getenv("NPU_RUN_GATED")) {   // (diagnosis) a gated image: its setup, then one job of one call
        npu.RunGatedOnce(image, entry, views, std::max(1, std::atoi(std::getenv("NPU_RUN_GATED"))));
      } else {
        npu.RunOnce(image, entry, views);
      }
      const auto t1 = std::chrono::steady_clock::now();
      if (rep) std::printf("npu_run: run %d %.1f us\n", i, std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    for (const auto& [b, path] : outputs) {
      FlushLines(views[b].shared->host, views[b].shared->bytes);
      std::ofstream out(path, std::ios::binary);
      out.write(static_cast<const char*>(views[b].shared->host), static_cast<std::streamsize>(files[b].size()));
      if (!out) throw LoomError("cannot write " + path);
    }
    std::printf("npu_run: ok\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "npu_run: %s\n", e.what());
    return 1;
  }
  return 0;
}
