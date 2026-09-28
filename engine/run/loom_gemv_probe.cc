// loom_gemv_probe: run the ported Q6_K GEMV over a real GGUF tensor through
// HRX, with no HIP. Imports the tensor as an HRX buffer, dispatches the Loom
// kernel, and writes the first rows of x and y for an external oracle.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/gguf.hpp"
#include "model/loom_runtime.hpp"

using namespace yah::model;  // NOLINT(google-build-using-namespace)

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: loom_gemv_probe <model.gguf> <hal.hal> <tensor> [out_y]\n");
    return 2;
  }
  const char* model = argv[1];
  const char* hal = argv[2];
  const char* tensor_name = argv[3];
  const char* out_y = argc > 4 ? argv[4] : "/tmp/loom_y.bin";
  try {
    auto gguf = yah::core::Gguf::Open(model);
    const auto* tensor = gguf.Find(tensor_name);
    if (!tensor) {
      std::fprintf(stderr, "tensor not found: %s\n", tensor_name);
      return 1;
    }
    const std::uint64_t rows = tensor->dims.size() > 1 ? tensor->dims[1] : 1;
    const std::uint64_t elems = tensor->dims[0];
    std::fprintf(stderr, "tensor %s rows=%llu k=%llu type=%s bytes=%llu\n",
                 tensor_name, static_cast<unsigned long long>(rows),
                 static_cast<unsigned long long>(elems),
                 yah::core::TypeName(tensor->type),
                 static_cast<unsigned long long>(tensor->bytes));
    std::fprintf(stderr, "file_offset=%llu\n",
                 static_cast<unsigned long long>(gguf.tensor_data_offset() +
                                                tensor->offset));

    LoomDevice gpu;
    // Import a page-aligned window containing the tensor.
    const std::uint8_t* data = gguf.Data(*tensor);
    const std::uintptr_t page = 4096;
    const std::uintptr_t start =
        reinterpret_cast<std::uintptr_t>(data) & ~(page - 1);
    const size_t window = static_cast<size_t>(tensor->bytes) +
                          (reinterpret_cast<std::uintptr_t>(data) - start);
    LoomBuffer weight = gpu.Import(reinterpret_cast<void*>(start), window);
    const size_t weight_offset =
        reinterpret_cast<std::uintptr_t>(data) - start;

    const size_t k_blocks = static_cast<size_t>(elems / 256);
    const size_t x_elems = k_blocks * 256;
    std::vector<float> x(x_elems);
    for (size_t i = 0; i < x_elems; ++i) {
      x[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25f;
    }
    std::vector<float> y(static_cast<size_t>(rows), 0.0f);
    LoomBuffer xb = gpu.Allocate(x.size() * sizeof(float));
    LoomBuffer yb = gpu.Allocate(y.size() * sizeof(float));
    gpu.H2D(xb, x.data(), x.size() * sizeof(float));

    LoomExecutable executable = gpu.Load(hal);
    const uint32_t ordinal = executable.OrdinalOrZero("yah_gemv_q6k");
    const hrx_buffer_ref_t bindings[3] = {
        {weight.handle, weight_offset, static_cast<size_t>(tensor->bytes)},
        {xb.handle, 0, x.size() * sizeof(float)},
        {yb.handle, 0, y.size() * sizeof(float)}};
    gpu.Dispatch(executable, ordinal,
                 LoomDevice::Config(static_cast<uint32_t>(rows), 1, 1, 32, 1, 1),
                 nullptr, 0, bindings, 3);
    gpu.Synchronize();
    gpu.D2H(yb, y.data(), y.size() * sizeof(float));

    FILE* file = std::fopen(out_y, "wb");
    if (!file) { std::fprintf(stderr, "cannot write %s\n", out_y); return 1; }
    std::fwrite(y.data(), sizeof(float), y.size(), file);
    std::fclose(file);
    std::fprintf(stderr, "wrote %zu y values to %s\n", y.size(), out_y);
    std::printf("y[0]=%f y[1]=%f y[2]=%f\n", y[0], y[1], y[2]);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "loom_gemv_probe: %s\n", error.what());
    return 1;
  }
}