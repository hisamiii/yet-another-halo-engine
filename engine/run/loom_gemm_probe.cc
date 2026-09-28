// loom_gemm_probe: run the ported format-faithful prefill GEMM (kStore) over a
// real GGUF tensor through HRX, with no HIP. The weight is imported from the
// GGUF mmap, the activation tile is generated deterministically, and the output
// is written for an external oracle. Exercises an arbitrary K (k_blocks) and a
// multi-token activation tile (token_tiles), which the fixture-shape port did
// not before parameterization.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/gguf.hpp"
#include "model/loom_runtime.hpp"

using namespace yah::model;  // NOLINT(google-build-using-namespace)

static std::uint16_t F16(float v) {
  _Float16 h = static_cast<_Float16>(v);
  std::uint16_t u;
  std::memcpy(&u, &h, sizeof(u));
  return u;
}

static float Activation(std::uint64_t token, std::uint64_t k) {
  const std::uint64_t v = (token * 7 + k * 13) % 23;
  return (static_cast<float>(v) - 11.0f) * 0.125f;
}

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr,
                 "usage: loom_gemm_probe <model.gguf> <hal.hal> <tensor> "
                 "<tokens> <out_y> [token_tiles]\n");
    return 2;
  }
  const char* model = argv[1];
  const char* hal = argv[2];
  const char* tensor_name = argv[3];
  const std::uint64_t tokens = std::strtoul(argv[4], nullptr, 10);
  const char* out_y = argv[5];
  const std::uint64_t token_tiles =
      argc > 6 ? std::strtoul(argv[6], nullptr, 10) : 1;
  try {
    auto gguf = yah::core::Gguf::Open(model);
    const auto* tensor = gguf.Find(tensor_name);
    if (!tensor) { std::fprintf(stderr, "not found: %s\n", tensor_name); return 1; }
    const std::uint64_t k = tensor->dims.size() > 0 ? tensor->dims[0] : 0;
    const std::uint64_t n = tensor->dims.size() > 1 ? tensor->dims[1] : 1;
    std::fprintf(stderr, "tensor %s n=%llu k=%llu type=%s bytes=%llu\n",
                 tensor_name, static_cast<unsigned long long>(n),
                 static_cast<unsigned long long>(k),
                 yah::core::TypeName(tensor->type),
                 static_cast<unsigned long long>(tensor->bytes));
    std::fprintf(stderr, "file_offset=%llu\n",
                 static_cast<unsigned long long>(gguf.tensor_data_offset() +
                                                tensor->offset));
    if (k % 256 != 0) { std::fprintf(stderr, "k not a multiple of 256\n"); return 1; }
    if (n % 16 != 0) { std::fprintf(stderr, "n not a multiple of 16\n"); return 1; }
    const std::uint64_t k_blocks = k / 256;
    const std::uint64_t m_tiles = n / 16;
    const std::uint64_t padded = token_tiles * 64;
    if (tokens > padded) { std::fprintf(stderr, "tokens > padded\n"); return 1; }

    LoomDevice gpu;
    const std::uint8_t* data = gguf.Data(*tensor);
    const std::uintptr_t page = 4096;
    const std::uintptr_t start =
        reinterpret_cast<std::uintptr_t>(data) & ~(page - 1);
    const size_t window = static_cast<size_t>(tensor->bytes) +
                          (reinterpret_cast<std::uintptr_t>(data) - start);
    LoomBuffer weight = gpu.Import(reinterpret_cast<void*>(start), window);
    const size_t weight_offset =
        reinterpret_cast<std::uintptr_t>(data) - start;

    std::vector<std::uint16_t> x(static_cast<size_t>(padded) * k);
    for (std::uint64_t t = 0; t < padded; ++t) {
      for (std::uint64_t i = 0; i < k; ++i) {
        x[t * k + i] = F16(Activation(t, i));
      }
    }
    LoomBuffer a = gpu.Allocate(x.size() * sizeof(std::uint16_t));
    LoomBuffer wstage = gpu.Allocate(static_cast<size_t>(n) * k * sizeof(std::uint16_t));
    LoomBuffer ostage = gpu.Allocate(static_cast<size_t>(n) * 64 * sizeof(float));
    LoomBuffer out = gpu.Allocate(static_cast<size_t>(n) * padded * sizeof(float));
    gpu.H2D(a, x.data(), x.size() * sizeof(std::uint16_t));

    LoomExecutable executable = gpu.Load(hal);
    const uint32_t ordinal = executable.OrdinalOrZero("yah_ffn_gemm_q4k");
    const hrx_buffer_ref_t bindings[5] = {
        {weight.handle, weight_offset, static_cast<size_t>(tensor->bytes)},
        {a.handle, 0, x.size() * sizeof(std::uint16_t)},
        {wstage.handle, 0, static_cast<size_t>(n) * k * sizeof(std::uint16_t)},
        {ostage.handle, 0, static_cast<size_t>(n) * 64 * sizeof(float)},
        {out.handle, 0, static_cast<size_t>(n) * padded * sizeof(float)}};
    gpu.Dispatch(executable, ordinal,
                 LoomDevice::Config(static_cast<uint32_t>(m_tiles),
                                    static_cast<uint32_t>(token_tiles), 1, 32, 1, 1),
                 nullptr, 0, bindings, 5);
    gpu.Synchronize();
    std::vector<float> y(static_cast<size_t>(n) * padded, 0.0f);
    gpu.D2H(out, y.data(), y.size() * sizeof(float));
    FILE* file = std::fopen(out_y, "wb");
    if (!file) { std::fprintf(stderr, "cannot write %s\n", out_y); return 1; }
    std::fwrite(y.data(), sizeof(float), y.size(), file);
    std::fclose(file);
    std::fprintf(stderr, "wrote %zu f32 values to %s\n", y.size(), out_y);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "loom_gemm_probe: %s\n", error.what());
    return 1;
  }
}