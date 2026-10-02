#ifndef YAH_MODEL_WEIGHTS_HPP_
#define YAH_MODEL_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"

namespace yah::model {

// A view of one tensor in the mapping, in GGUF order.
// ne0 is the row length (the reduction dimension), ne1 the row count (the output dimension).
struct TensorRef {
  const std::uint8_t* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t ne0{0};
  std::uint64_t ne1{0};
  std::uint64_t bytes{0};
  bool present{false};

  [[nodiscard]] std::uint64_t rows() const { return ne1; }
  [[nodiscard]] std::uint64_t cols() const { return ne0; }
};

struct Qwen35Layer {
  TensorRef attn_norm;
  TensorRef post_attention_norm;

  // Full-attention layers only. attn_q holds the query and the output gate fused: ne1 is twice the attention width.
  TensorRef attn_q, attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm;

  // Gated DeltaNet layers only.
  TensorRef attn_qkv, attn_gate;
  TensorRef ssm_a, ssm_alpha, ssm_beta, ssm_conv1d, ssm_dt, ssm_norm, ssm_out;

  TensorRef ffn_gate, ffn_up, ffn_down;

  bool is_full_attention{false};
};

struct Qwen35Weights {
  TensorRef token_embd;
  TensorRef output_norm;
  TensorRef output;
  std::vector<Qwen35Layer> layers;

  // The MTP head's tensors, present only on block_count - 1. Not resolved for the text path.
  TensorRef nextn_eh_proj, nextn_enorm, nextn_hnorm, nextn_shared_head_norm;

  // Resolves every tensor the layer loop needs and checks it against the config (row lengths, counts, layer types).
  // Throws on anything missing or inconsistent; never returns a partial table.
  static Qwen35Weights FromGguf(const core::Gguf& gguf, const core::Qwen35Config& config);

  // Shifts every resolved tensor pointer by `delta` bytes, e.g. from the file mapping onto the device copy.
  void Rebase(std::ptrdiff_t delta);

  [[nodiscard]] std::uint64_t bytes() const;
};

}  // namespace yah::model

#endif  // YAH_MODEL_WEIGHTS_HPP_
