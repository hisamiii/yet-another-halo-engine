// The GGUF conversion's CPU pieces on files, for tests/gguf_check.py:
//   gguf_rows dequant <ggml type id> <blocks.bin> <out.f32>       raw blocks of a type to f32
//   gguf_rows mq4 <rows.f32> <row length> <out.mq4>                original-basis rows to MQ4G256 (engine form)
//   gguf_rows tensor <model.gguf> <tensor> <rows> <out.f32>        a tensor's first rows to f32
//   gguf_rows load <model>                                         the engine's resident open (conversion), timed
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "core/gguf_quant.hpp"

using namespace yah::core;

namespace {

std::vector<std::uint8_t> Read(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error(std::string("cannot read ") + path);
  return {std::istreambuf_iterator<char>(f), {}};
}
void Write(const char* path, const void* p, std::size_t n) {
  std::ofstream(path, std::ios::binary).write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  try {
    if (mode == "dequant" && argc == 5) {
      const auto t = static_cast<GgmlType>(std::strtoul(argv[2], nullptr, 10));
      const auto in = Read(argv[3]);
      const std::uint64_t n = in.size() / BlockBytes(t) * BlockElements(t);
      std::vector<float> out(n);
      Dequantize(t, in.data(), out.data(), n);
      Write(argv[4], out.data(), n * 4);
    } else if (mode == "mq4" && argc == 5) {
      auto in = Read(argv[2]);
      const std::uint64_t k = std::strtoull(argv[3], nullptr, 10), rows = in.size() / 4 / k;
      std::vector<std::uint8_t> out(rows * k / 256 * 136);
      for (std::uint64_t r = 0; r < rows; ++r)
        QuantizeMq4Row(reinterpret_cast<float*>(in.data()) + r * k, k, out.data() + r * k / 256 * 136);
      Write(argv[4], out.data(), out.size());
    } else if (mode == "tensor" && argc == 6) {
      const Gguf g = Gguf::Open(argv[2]);
      const TensorInfo* t = g.Find(argv[3]);
      if (!t) throw std::runtime_error(std::string("no tensor ") + argv[3]);
      const std::uint64_t n = std::strtoull(argv[4], nullptr, 10) * t->dims[0];
      std::vector<float> out(n);
      Dequantize(t->type, g.Data(*t), out.data(), n);
      Write(argv[5], out.data(), n * 4);
    } else if (mode == "load" && argc == 3) {
      const auto t0 = std::chrono::steady_clock::now();
      const Gguf g = Gguf::OpenResident(argv[2]);
      std::printf("%zu tensors, %.2f GiB in %.1f s\n", g.tensors().size(), g.tensor_data_size() / 1073741824.0,
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    } else {
      std::fprintf(stderr, "usage: gguf_rows dequant <type> <blocks> <out> | mq4 <rows.f32> <k> <out> | "
                           "tensor <model> <name> <rows> <out> | load <model>\n");
      return 2;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gguf_rows: %s\n", e.what());
    return 1;
  }
  return 0;
}
