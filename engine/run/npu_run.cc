// Runs one NPU image invocation through LoomNpu: the in-engine stand-in for HRX's iree-xdna-run, which HRX main no
// longer has (tools/npu_gemm_check.py and npu_attn_check.py drive it).
//
// usage: npu_run <image.xdna> <entry> <columns> <binding>=<output file>[,...] <binding file>...
//
// Runs the entry's first invocation (iree-xdna-run --invocation_count=1). Each binding file becomes host pages the NPU
// reads and writes, in order; after the run, each listed binding's bytes (its file's length) go to its output file. A
// context width the image does not admit fails with the loader's message ("does not match the admitted native
// context"): the caller retries a wider one.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
      views.push_back({&s, 0, f.size()});
    }
    npu.RunOnce(image, entry, views);
    for (const auto& [b, path] : outputs) {
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
