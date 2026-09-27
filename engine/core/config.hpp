#ifndef YAH_CORE_CONFIG_HPP_
#define YAH_CORE_CONFIG_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "core/gguf.hpp"

namespace yah::core {

// Qwen3.8 27B, which stores its architecture as "qwen35". Every field is read
// from the file's own metadata; nothing is hard-coded except the key names.
struct Qwen35Config {
  std::string architecture;
  std::uint32_t block_count{0};
  std::uint32_t context_length{0};
  std::uint32_t embedding_length{0};
  std::uint32_t feed_forward_length{0};
  std::uint32_t head_count{0};
  std::uint32_t head_count_kv{0};
  std::uint32_t key_length{0};
  std::uint32_t value_length{0};
  std::uint32_t rope_dimension_count{0};
  float rope_freq_base{0.0F};
  float rms_eps{0.0F};
  std::uint32_t full_attention_interval{0};
  std::uint32_t nextn_predict_layers{0};
  std::uint32_t ssm_conv_kernel{0};
  std::uint32_t ssm_state_size{0};
  std::uint32_t ssm_group_count{0};
  std::uint32_t ssm_time_step_rank{0};
  std::uint32_t ssm_inner_size{0};

  // Layer types alternate on a fixed interval: one full-attention layer every
  // full_attention_interval, Gated DeltaNet otherwise. layer is 0-based.
  [[nodiscard]] bool IsFullAttention(std::uint32_t layer) const {
    return full_attention_interval != 0 &&
           (layer + 1) % full_attention_interval == 0;
  }
  // The MTP head is one of the block_count and is not part of the main stack.
  [[nodiscard]] std::uint32_t main_block_count() const {
    return block_count - nextn_predict_layers;
  }
  [[nodiscard]] std::uint32_t AttentionLayers() const {
    return full_attention_interval == 0
               ? 0
               : main_block_count() / full_attention_interval;
  }
  [[nodiscard]] std::uint32_t RecurrentLayers() const {
    return main_block_count() - AttentionLayers();
  }
  // Dense index of a Gated DeltaNet layer among the recurrent layers, which is
  // the slot its convolution and recurrent states live in.
  [[nodiscard]] std::uint32_t SsmLayerIndex(std::uint32_t layer) const {
    return full_attention_interval == 0
               ? layer
               : layer - layer / full_attention_interval;
  }
  // Width of one DeltaNet value head group, ssm_inner_size / time_step_rank.
  [[nodiscard]] std::uint32_t SsmValueSize() const {
    return ssm_time_step_rank == 0 ? 0 : ssm_inner_size / ssm_time_step_rank;
  }
  // Attention width (q for each token), and the fused qkv width.
  [[nodiscard]] std::uint32_t attention_q_dim() const {
    return head_count * key_length;
  }
  [[nodiscard]] std::uint32_t attention_kv_dim() const {
    return head_count_kv * key_length;
  }

  static Qwen35Config FromGguf(const Gguf& gguf);
};

// Tokenizer pieces, left as pointers into the metadata map so the 248k strings
// are neither copied nor re-allocated. The map owns them for the Gguf's life.
struct TokenizerConfig {
  std::string model;
  std::string pre;
  std::uint32_t bos_id{0};
  std::uint32_t eos_id{0};
  std::uint32_t padding_id{0};
  bool add_bos{false};
  std::string chat_template;
  const std::vector<MetadataValue>* tokens{nullptr};
  const std::vector<MetadataValue>* token_types{nullptr};
  const std::vector<MetadataValue>* merges{nullptr};

  static TokenizerConfig FromGguf(const Gguf& gguf);
};

}  // namespace yah::core

#endif  // YAH_CORE_CONFIG_HPP_
