#include <cinttypes>
#include <cstdio>
#include <map>
#include <string>

#include "core/gguf.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: yah-dump <model.gguf>\n");
    return 2;
  }
  auto gguf = yah::core::Gguf::Open(argv[1]);
  std::printf("version=%u tensors=%zu metadata=%zu file=%.2f GiB data_offset=%zu\n",
              gguf.version(), gguf.tensors().size(), gguf.metadata_count(),
              static_cast<double>(gguf.file_size()) / 1073741824.0,
              gguf.tensor_data_offset());

  struct Totals {
    std::uint64_t bytes{0};
    std::uint64_t elements{0};
    std::uint64_t count{0};
  };
  std::map<std::string, Totals> by_type;
  std::uint64_t total_bytes = 0;
  std::uint64_t total_elements = 0;
  for (const auto& t : gguf.tensors()) {
    auto& entry = by_type[yah::core::TypeName(t.type)];
    entry.bytes += t.bytes;
    entry.elements += t.elements;
    entry.count += 1;
    total_bytes += t.bytes;
    total_elements += t.elements;
  }
  for (const auto& [name, totals] : by_type) {
    std::printf("  %-9s %7.2f GiB  %5.2f%%  %7.3f bpw  x%" PRIu64 "\n",
                name.c_str(), static_cast<double>(totals.bytes) / 1073741824.0,
                100.0 * static_cast<double>(totals.bytes) /
                    static_cast<double>(total_bytes),
                8.0 * static_cast<double>(totals.bytes) /
                    static_cast<double>(totals.elements),
                totals.count);
  }
  std::printf("total %.2f GiB  %.3f bpw\n",
              static_cast<double>(total_bytes) / 1073741824.0,
              8.0 * static_cast<double>(total_bytes) /
                  static_cast<double>(total_elements));

  for (const char* name : {"token_embd.weight", "blk.0.ffn_gate.weight",
                           "blk.0.ffn_down.weight", "output.weight"}) {
    const auto* t = gguf.Find(name);
    if (t == nullptr) continue;
    std::printf("  %-24s %-8s dims=", t->name.c_str(),
                yah::core::TypeName(t->type));
    for (std::size_t i = 0; i < t->dims.size(); ++i) {
      std::printf("%s%" PRIu64, i == 0 ? "" : "x", t->dims[i]);
    }
    std::printf(" bytes=%" PRIu64 "\n", t->bytes);
  }
  return 0;
}
