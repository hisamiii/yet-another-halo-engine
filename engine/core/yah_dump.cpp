#include <cinttypes>
#include <cstdio>
#include <map>
#include <string>

#include "core/config.hpp"
#include "core/gguf.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: yah-dump <model.gguf>\n");
    return 2;
  }
  auto gguf = yah::core::Gguf::Open(argv[1]);
  std::printf("version=%u tensors=%zu metadata=%zu file=%.2f GiB data_offset=%zu\n", gguf.version(),
              gguf.tensors().size(), gguf.metadata_count(), static_cast<double>(gguf.file_size()) / 1073741824.0,
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
    std::printf("  %-9s %7.2f GiB  %5.2f%%  %7.3f bpw  x%" PRIu64 "\n", name.c_str(),
                static_cast<double>(totals.bytes) / 1073741824.0,
                100.0 * static_cast<double>(totals.bytes) / static_cast<double>(total_bytes),
                8.0 * static_cast<double>(totals.bytes) / static_cast<double>(totals.elements), totals.count);
  }
  std::printf("total %.2f GiB  %.3f bpw\n", static_cast<double>(total_bytes) / 1073741824.0,
              8.0 * static_cast<double>(total_bytes) / static_cast<double>(total_elements));

  for (const char* name : {"token_embd.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "output.weight"}) {
    const auto* t = gguf.Find(name);
    if (t == nullptr) continue;
    std::printf("  %-24s %-8s dims=", t->name.c_str(), yah::core::TypeName(t->type));
    for (std::size_t i = 0; i < t->dims.size(); ++i) {
      std::printf("%s%" PRIu64, i == 0 ? "" : "x", t->dims[i]);
    }
    std::printf(" bytes=%" PRIu64 "\n", t->bytes);
  }

  const auto* arch = gguf.Meta("general.architecture");
  if (arch != nullptr && arch->kind == yah::core::MetadataValue::Kind::kString && arch->s == "qwen35") {
    const auto cfg = yah::core::Qwen35Config::FromGguf(gguf);
    std::printf("config: arch=%s blocks=%u ctx=%u hidden=%u ffn=%u\n", cfg.architecture.c_str(), cfg.block_count,
                cfg.context_length, cfg.embedding_length, cfg.feed_forward_length);
    std::printf("config: heads=%u kv=%u key=%u val=%u q_dim=%u kv_dim=%u\n", cfg.head_count, cfg.head_count_kv,
                cfg.key_length, cfg.value_length, cfg.attention_q_dim(), cfg.attention_kv_dim());
    std::printf("config: rope_dim=%u rope_base=%.1f eps=%.1e full_attn_every=%u first_full=%u\n",
                cfg.rope_dimension_count, static_cast<double>(cfg.rope_freq_base), static_cast<double>(cfg.rms_eps),
                cfg.full_attention_interval, cfg.full_attention_interval);
    std::printf("config: ssm conv=%u state=%u groups=%u dt_rank=%u inner=%u\n", cfg.ssm_conv_kernel, cfg.ssm_state_size,
                cfg.ssm_group_count, cfg.ssm_time_step_rank, cfg.ssm_inner_size);
    std::printf("config: full_attention_layers=%u recurrent_layers=%u nextn=%u\n", cfg.AttentionLayers(),
                cfg.RecurrentLayers(), cfg.nextn_predict_layers);
    const auto tok = yah::core::TokenizerConfig::FromGguf(gguf);
    std::printf("tokenizer: model=%s pre=%s bos=%u eos=%u pad=%u add_bos=%d\n", tok.model.c_str(), tok.pre.c_str(),
                tok.bos_id, tok.eos_id, tok.padding_id, tok.add_bos ? 1 : 0);
    std::printf("tokenizer: tokens=%zu merges=%zu template=%zu bytes\n", tok.tokens != nullptr ? tok.tokens->size() : 0,
                tok.merges != nullptr ? tok.merges->size() : 0, tok.chat_template.size());
  }
  return 0;
}
