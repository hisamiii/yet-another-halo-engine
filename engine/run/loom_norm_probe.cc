// loom_norm_probe: run the prefill RMSNorm (fp16 out) through HRX on real
// attn_norm weights. Reads the f32 hidden produced by the embedding probe.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/gguf.hpp"
#include "model/loom_runtime.hpp"

using namespace yah::model;  // NOLINT(google-build-using-namespace)

static std::vector<std::uint8_t> ReadFile(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
  std::fseek(f, 0, SEEK_END);
  const long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<std::uint8_t> data(static_cast<size_t>(size));
  if (std::fread(data.data(), 1, data.size(), f) != data.size()) {
    std::fprintf(stderr, "short read %s\n", path); std::exit(1);
  }
  std::fclose(f);
  return data;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: loom_norm_probe <model.gguf> <hal.hal> <hidden.f32> "
                 "<out.f16> <tensor> <rows>\n");
    return 2;
  }
  const char* model = argv[1];
  const char* hal = argv[2];
  const char* hidden_path = argv[3];
  const char* out_path = argv[4];
  const char* weight_name = argv[5];
  const std::uint32_t rows = static_cast<std::uint32_t>(std::stoul(argv[6]));
  try {
    auto gguf = yah::core::Gguf::Open(model);
    const auto* tensor = gguf.Find(weight_name);
    if (!tensor) { std::fprintf(stderr, "not found: %s\n", weight_name); return 1; }
    const std::uint64_t dim = tensor->dims[0];
    std::fprintf(stderr, "weight %s dim=%llu type=%s\n", weight_name,
                 static_cast<unsigned long long>(dim),
                 yah::core::TypeName(tensor->type));
    std::fprintf(stderr, "file_offset=%llu\n",
                 static_cast<unsigned long long>(gguf.tensor_data_offset() +
                                                tensor->offset));

    std::vector<std::uint8_t> hidden_bytes = ReadFile(hidden_path);
    const size_t elems = static_cast<size_t>(rows) * dim;
    if (hidden_bytes.size() != elems * sizeof(float)) {
      std::fprintf(stderr, "hidden size %zu != %zu\n", hidden_bytes.size(),
                   elems * sizeof(float));
      return 1;
    }

    LoomDevice gpu;
    const std::uint8_t* wdata = gguf.Data(*tensor);
    const std::uintptr_t page = 4096;
    const std::uintptr_t wstart =
        reinterpret_cast<std::uintptr_t>(wdata) & ~(page - 1);
    const size_t wwindow = static_cast<size_t>(tensor->bytes) +
                           (reinterpret_cast<std::uintptr_t>(wdata) - wstart);
    LoomBuffer weight = gpu.Import(reinterpret_cast<void*>(wstart), wwindow);
    const size_t weight_offset =
        reinterpret_cast<std::uintptr_t>(wdata) - wstart;

    LoomBuffer x = gpu.Allocate(elems * sizeof(float));
    LoomBuffer residual = gpu.Allocate(elems * sizeof(float));
    LoomBuffer sum_out = gpu.Allocate(elems * sizeof(float));
    LoomBuffer out = gpu.Allocate(elems * sizeof(std::uint16_t));
    std::vector<float> zeros(elems, 0.0f);
    gpu.H2D(x, hidden_bytes.data(), hidden_bytes.size());
    gpu.H2D(residual, zeros.data(), zeros.size() * sizeof(float));

    LoomExecutable executable = gpu.Load(hal);
    const uint32_t ordinal = executable.OrdinalOrZero("yah_half_norm");
    const hrx_buffer_ref_t bindings[5] = {
        {x.handle, 0, elems * sizeof(float)},
        {residual.handle, 0, elems * sizeof(float)},
        {weight.handle, weight_offset, static_cast<size_t>(tensor->bytes)},
        {sum_out.handle, 0, elems * sizeof(float)},
        {out.handle, 0, elems * sizeof(std::uint16_t)}};
    gpu.Dispatch(executable, ordinal, LoomDevice::Config(rows, 1, 1, 32, 1, 1),
                 nullptr, 0, bindings, 5);
    gpu.Synchronize();
    std::vector<std::uint16_t> host(elems, 0);
    gpu.D2H(out, host.data(), elems * sizeof(std::uint16_t));
    FILE* file = std::fopen(out_path, "wb");
    std::fwrite(host.data(), sizeof(std::uint16_t), host.size(), file);
    std::fclose(file);
    std::fprintf(stderr, "wrote %zu f16 values to %s\n", host.size(), out_path);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "loom_norm_probe: %s\n", error.what());
    return 1;
  }
}