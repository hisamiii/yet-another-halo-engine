// loom_embed_probe: run the ported prefill embedding (Q3_K table) through HRX.
// No HIP. Uploads token ids, imports the token_embd table, dispatches, dumps
// the batch*hidden fp32 hidden states.
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
                 "usage: loom_embed_probe <model.gguf> <hal.hal> <out.bin> <id>...\n");
    return 2;
  }
  const char* model = argv[1];
  const char* hal = argv[2];
  const char* out_path = argv[3];
  std::vector<std::uint32_t> tokens;
  for (int i = 4; i < argc; ++i) tokens.push_back(
      static_cast<std::uint32_t>(std::strtoul(argv[i], nullptr, 10)));
  if (tokens.empty()) { std::fprintf(stderr, "no tokens\n"); return 2; }
  try {
    auto gguf = yah::core::Gguf::Open(model);
    const auto* tensor = gguf.Find("token_embd.weight");
    if (!tensor) { std::fprintf(stderr, "token_embd not found\n"); return 1; }
    const std::uint64_t hidden = tensor->dims[0];
    const std::uint64_t vocab = tensor->dims[1];
    const std::uint64_t row_bytes = tensor->bytes / vocab;
    std::fprintf(stderr,
                 "token_embd rows=%llu hidden=%llu type=%s row_bytes=%llu\n",
                 static_cast<unsigned long long>(vocab),
                 static_cast<unsigned long long>(hidden),
                 yah::core::TypeName(tensor->type),
                 static_cast<unsigned long long>(row_bytes));
    std::fprintf(stderr, "file_offset=%llu\n",
                 static_cast<unsigned long long>(gguf.tensor_data_offset() +
                                                tensor->offset));
    if (row_bytes % 110 != 0) { std::fprintf(stderr, "not Q3_K rows\n"); return 1; }
    const std::uint64_t bpr = row_bytes / 110;

    LoomDevice gpu;
    const std::uint8_t* data = gguf.Data(*tensor);
    const std::uintptr_t page = 4096;
    const std::uintptr_t start =
        reinterpret_cast<std::uintptr_t>(data) & ~(page - 1);
    const size_t window = static_cast<size_t>(tensor->bytes) +
                          (reinterpret_cast<std::uintptr_t>(data) - start);
    LoomBuffer table = gpu.Import(reinterpret_cast<void*>(start), window);
    const size_t table_offset =
        reinterpret_cast<std::uintptr_t>(data) - start;

    const size_t batch = tokens.size();
    const size_t out_elems = batch * hidden;
    LoomBuffer tok = gpu.Allocate(batch * sizeof(std::uint32_t));
    LoomBuffer out = gpu.Allocate(out_elems * sizeof(float));
    gpu.H2D(tok, tokens.data(), batch * sizeof(std::uint32_t));

    LoomExecutable executable = gpu.Load(hal);
    const uint32_t ordinal = executable.OrdinalOrZero("yah_prefill_embed_q3k");
    const hrx_buffer_ref_t bindings[3] = {
        {table.handle, table_offset, static_cast<size_t>(tensor->bytes)},
        {tok.handle, 0, batch * sizeof(std::uint32_t)},
        {out.handle, 0, out_elems * sizeof(float)}};
    gpu.Dispatch(executable, ordinal,
                 LoomDevice::Config(static_cast<uint32_t>(bpr),
                                    static_cast<uint32_t>(batch), 1, 32, 1, 1),
                 nullptr, 0, bindings, 3);
    gpu.Synchronize();
    std::vector<float> host(out_elems, 0.0f);
    gpu.D2H(out, host.data(), out_elems * sizeof(float));
    FILE* file = std::fopen(out_path, "wb");
    std::fwrite(host.data(), sizeof(float), host.size(), file);
    std::fclose(file);
    std::fprintf(stderr, "wrote %zu hidden values to %s\n", host.size(), out_path);
    std::printf("h[0][0..3]=%f %f %f %f\n", host[0], host[1], host[2], host[3]);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "loom_embed_probe: %s\n", error.what());
    return 1;
  }
}