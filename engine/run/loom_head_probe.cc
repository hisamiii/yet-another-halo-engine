// loom_head_probe: run the graph tail on Loom through HRX with no HIP: the
// output RMSNorm, the Q4_K output GEMV and the argmax, over the real
// output_norm / output weights and a hidden state dumped by the HIP engine.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/gguf.hpp"
#include "model/loom_runtime.hpp"

using namespace yah::model;  // NOLINT(google-build-using-namespace)

int main(int argc, char** argv) {
  if (argc < 8) {
    std::fprintf(stderr,
                 "usage: loom_head_probe <model.gguf> <rms.hal> <gemv.hal> "
                 "<argmax.hal> <hidden.f32> <rows> <logits_out>\n");
    return 2;
  }
  const char* model = argv[1];
  const char* rms_hal = argv[2];
  const char* gemv_hal = argv[3];
  const char* argmax_hal = argv[4];
  const char* hidden_path = argv[5];
  const std::uint64_t rows = std::strtoull(argv[6], nullptr, 10);
  const char* logits_out = argv[7];
  try {
    auto gguf = yah::core::Gguf::Open(model);
    const auto* norm = gguf.Find("output_norm.weight");
    const auto* out = gguf.Find("output.weight");
    if (!norm || !out) { std::fprintf(stderr, "head tensors not found\n"); return 1; }
    const std::uint64_t hidden = norm->dims[0];
    const std::uint64_t vocab = out->dims[1];
    std::fprintf(stderr, "hidden=%llu vocab=%llu out_off=%llu\n",
                 static_cast<unsigned long long>(hidden),
                 static_cast<unsigned long long>(vocab),
                 static_cast<unsigned long long>(gguf.tensor_data_offset() + out->offset));

    std::vector<float> host_hidden(static_cast<size_t>(rows) * hidden);
    FILE* fh = std::fopen(hidden_path, "rb");
    if (!fh) { std::fprintf(stderr, "cannot open %s\n", hidden_path); return 1; }
    if (std::fread(host_hidden.data(), sizeof(float), host_hidden.size(), fh) !=
        host_hidden.size()) { std::fprintf(stderr, "short hidden\n"); return 1; }
    std::fclose(fh);

    LoomDevice gpu;
    const std::uint8_t* data = gguf.Data(*out);
    const std::uintptr_t page = 4096;
    const std::uintptr_t start =
        reinterpret_cast<std::uintptr_t>(data) & ~(page - 1);
    const size_t window = static_cast<size_t>(out->bytes) +
                          (reinterpret_cast<std::uintptr_t>(data) - start);
    LoomBuffer weight = gpu.Import(reinterpret_cast<void*>(start), window);
    const size_t weight_offset =
        reinterpret_cast<std::uintptr_t>(data) - start;

    LoomBuffer x = gpu.Allocate(hidden * sizeof(float));
    LoomBuffer wnorm = gpu.Allocate(hidden * sizeof(float));
    LoomBuffer normed = gpu.Allocate(hidden * sizeof(float));
    LoomBuffer logits = gpu.Allocate(vocab * sizeof(float));
    LoomBuffer token = gpu.Allocate(sizeof(std::uint32_t));
    const float* last = host_hidden.data() + (rows - 1) * hidden;
    gpu.H2D(x, last, hidden * sizeof(float));
    gpu.H2D(wnorm, gguf.Data(*norm), hidden * sizeof(float));

    LoomExecutable rms = gpu.Load(rms_hal);
    {
      const hrx_buffer_ref_t b[3] = {
          {x.handle, 0, hidden * sizeof(float)},
          {wnorm.handle, 0, hidden * sizeof(float)},
          {normed.handle, 0, hidden * sizeof(float)}};
      gpu.Dispatch(rms, rms.OrdinalOrZero("yah_rmsnorm"),
                   LoomDevice::Config(1, 1, 1, 32, 1, 1), nullptr, 0, b, 3);
    }
    LoomExecutable gemv = gpu.Load(gemv_hal);
    {
      const hrx_buffer_ref_t b[3] = {
          {weight.handle, weight_offset, static_cast<size_t>(out->bytes)},
          {normed.handle, 0, hidden * sizeof(float)},
          {logits.handle, 0, vocab * sizeof(float)}};
      gpu.Dispatch(gemv, gemv.OrdinalOrZero("yah_gemv_q4k"),
                   LoomDevice::Config(static_cast<uint32_t>(vocab), 1, 1, 32, 1, 1),
                   nullptr, 0, b, 3);
    }
    LoomExecutable arg = gpu.Load(argmax_hal);
    {
      const hrx_buffer_ref_t b[2] = {
          {logits.handle, 0, vocab * sizeof(float)},
          {token.handle, 0, sizeof(std::uint32_t)}};
      gpu.Dispatch(arg, arg.OrdinalOrZero("yah_argmax"),
                   LoomDevice::Config(1, 1, 1, 32, 1, 1), nullptr, 0, b, 2);
    }
    gpu.Synchronize();
    std::uint32_t tok = 0;
    gpu.D2H(token, &tok, sizeof(tok), 0);
    std::vector<float> host_logits(vocab, 0.0f);
    gpu.D2H(logits, host_logits.data(), vocab * sizeof(float), 0);
    FILE* fo = std::fopen(logits_out, "wb");
    std::fwrite(host_logits.data(), sizeof(float), host_logits.size(), fo);
    std::fclose(fo);
    std::printf("argmax=%u\n", tok);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "loom_head_probe: %s\n", error.what());
    return 1;
  }
}