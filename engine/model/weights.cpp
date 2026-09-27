#include "model/weights.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace yah::model {
namespace {

TensorRef Resolve(const core::Gguf& gguf, const std::string& name) {
  TensorRef ref;
  const core::TensorInfo* info = gguf.Find(name);
  if (info == nullptr) return ref;
  ref.data = gguf.Data(*info);
  ref.type = info->type;
  ref.ne0 = info->dims.size() > 0 ? info->dims[0] : 0;
  ref.ne1 = info->dims.size() > 1 ? info->dims[1] : 1;
  ref.bytes = info->bytes;
  ref.present = true;
  return ref;
}

void Require(const TensorRef& ref, const std::string& name) {
  if (!ref.present) {
    throw std::runtime_error("weights: missing tensor " + name);
  }
}

void RequireCols(const TensorRef& ref, const std::string& name,
                 std::uint64_t expected) {
  Require(ref, name);
  if (ref.cols() != expected) {
    throw std::runtime_error("weights: " + name + " row length " +
                             std::to_string(ref.cols()) + " != " +
                             std::to_string(expected));
  }
}

std::string LayerName(std::uint32_t layer, const char* suffix) {
  return "blk." + std::to_string(layer) + "." + suffix;
}

}  // namespace

Qwen35Weights Qwen35Weights::FromGguf(const core::Gguf& gguf,
                                      const core::Qwen35Config& config) {
  const std::uint64_t hidden = config.embedding_length;
  const std::uint64_t ffn = config.feed_forward_length;
  const std::uint64_t q_dim = config.attention_q_dim();
  const std::uint64_t kv_dim = config.attention_kv_dim();

  Qwen35Weights weights;
  weights.token_embd = Resolve(gguf, "token_embd.weight");
  weights.output_norm = Resolve(gguf, "output_norm.weight");
  weights.output = Resolve(gguf, "output.weight");
  RequireCols(weights.token_embd, "token_embd.weight", hidden);
  Require(weights.output_norm, "output_norm.weight");
  RequireCols(weights.output, "output.weight", hidden);

  weights.layers.resize(config.block_count);
  for (std::uint32_t i = 0; i < config.block_count; ++i) {
    Qwen35Layer& layer = weights.layers[i];
    // The main stack alternates on the interval; the trailing MTP block is a
    // separate full-attention layer and must not be classified by the pattern.
    layer.is_full_attention =
        i >= config.main_block_count() || config.IsFullAttention(i);

    layer.attn_norm = Resolve(gguf, LayerName(i, "attn_norm.weight"));
    layer.post_attention_norm =
        Resolve(gguf, LayerName(i, "post_attention_norm.weight"));
    RequireCols(layer.attn_norm, LayerName(i, "attn_norm.weight"), hidden);
    RequireCols(layer.post_attention_norm,
                LayerName(i, "post_attention_norm.weight"), hidden);

    layer.ffn_gate = Resolve(gguf, LayerName(i, "ffn_gate.weight"));
    layer.ffn_up = Resolve(gguf, LayerName(i, "ffn_up.weight"));
    layer.ffn_down = Resolve(gguf, LayerName(i, "ffn_down.weight"));
    RequireCols(layer.ffn_gate, LayerName(i, "ffn_gate.weight"), hidden);
    RequireCols(layer.ffn_up, LayerName(i, "ffn_up.weight"), hidden);
    RequireCols(layer.ffn_down, LayerName(i, "ffn_down.weight"), ffn);

    if (layer.is_full_attention) {
      layer.attn_q = Resolve(gguf, LayerName(i, "attn_q.weight"));
      layer.attn_k = Resolve(gguf, LayerName(i, "attn_k.weight"));
      layer.attn_v = Resolve(gguf, LayerName(i, "attn_v.weight"));
      layer.attn_output = Resolve(gguf, LayerName(i, "attn_output.weight"));
      layer.attn_q_norm = Resolve(gguf, LayerName(i, "attn_q_norm.weight"));
      layer.attn_k_norm = Resolve(gguf, LayerName(i, "attn_k_norm.weight"));
      RequireCols(layer.attn_q, LayerName(i, "attn_q.weight"), hidden);
      RequireCols(layer.attn_k, LayerName(i, "attn_k.weight"), hidden);
      RequireCols(layer.attn_v, LayerName(i, "attn_v.weight"), hidden);
      RequireCols(layer.attn_output, LayerName(i, "attn_output.weight"),
                  q_dim);
      if (layer.attn_q.rows() != 2 * q_dim) {
        throw std::runtime_error("weights: " + LayerName(i, "attn_q.weight") +
                                 " rows " + std::to_string(layer.attn_q.rows()) +
                                 " != 2 x attention width");
      }
      if (layer.attn_k.rows() != kv_dim || layer.attn_v.rows() != kv_dim) {
        throw std::runtime_error("weights: " + LayerName(i, "attn_k/v.weight") +
                                 " rows != kv width");
      }
    } else {
      layer.attn_qkv = Resolve(gguf, LayerName(i, "attn_qkv.weight"));
      layer.attn_gate = Resolve(gguf, LayerName(i, "attn_gate.weight"));
      layer.ssm_a = Resolve(gguf, LayerName(i, "ssm_a"));
      layer.ssm_alpha = Resolve(gguf, LayerName(i, "ssm_alpha.weight"));
      layer.ssm_beta = Resolve(gguf, LayerName(i, "ssm_beta.weight"));
      layer.ssm_conv1d = Resolve(gguf, LayerName(i, "ssm_conv1d.weight"));
      layer.ssm_dt = Resolve(gguf, LayerName(i, "ssm_dt.bias"));
      layer.ssm_norm = Resolve(gguf, LayerName(i, "ssm_norm.weight"));
      layer.ssm_out = Resolve(gguf, LayerName(i, "ssm_out.weight"));
      RequireCols(layer.attn_qkv, LayerName(i, "attn_qkv.weight"), hidden);
      RequireCols(layer.attn_gate, LayerName(i, "attn_gate.weight"), hidden);
      RequireCols(layer.ssm_alpha, LayerName(i, "ssm_alpha.weight"), hidden);
      RequireCols(layer.ssm_beta, LayerName(i, "ssm_beta.weight"), hidden);
      RequireCols(layer.ssm_out, LayerName(i, "ssm_out.weight"),
                  config.ssm_inner_size);
      if (layer.ssm_alpha.rows() != config.ssm_time_step_rank ||
          layer.ssm_beta.rows() != config.ssm_time_step_rank) {
        throw std::runtime_error("weights: " + LayerName(i, "ssm_alpha/beta") +
                                 " rows != time_step_rank");
      }
    }
  }

  const std::uint32_t mtp = config.block_count - 1;
  if (config.nextn_predict_layers > 0) {
    weights.nextn_eh_proj = Resolve(gguf, LayerName(mtp, "nextn.eh_proj.weight"));
    weights.nextn_enorm = Resolve(gguf, LayerName(mtp, "nextn.enorm.weight"));
    weights.nextn_hnorm = Resolve(gguf, LayerName(mtp, "nextn.hnorm.weight"));
    weights.nextn_shared_head_norm =
        Resolve(gguf, LayerName(mtp, "nextn.shared_head_norm.weight"));
  }
  return weights;
}

std::uint64_t Qwen35Weights::bytes() const {
  std::uint64_t total = token_embd.bytes + output_norm.bytes + output.bytes;
  for (const auto& layer : layers) {
    total += layer.attn_norm.bytes + layer.post_attention_norm.bytes;
    total += layer.attn_q.bytes + layer.attn_k.bytes + layer.attn_v.bytes +
             layer.attn_output.bytes + layer.attn_q_norm.bytes +
             layer.attn_k_norm.bytes;
    total += layer.attn_qkv.bytes + layer.attn_gate.bytes + layer.ssm_a.bytes +
             layer.ssm_alpha.bytes + layer.ssm_beta.bytes +
             layer.ssm_conv1d.bytes + layer.ssm_dt.bytes + layer.ssm_norm.bytes +
             layer.ssm_out.bytes;
    total += layer.ffn_gate.bytes + layer.ffn_up.bytes + layer.ffn_down.bytes;
  }
  return total;
}

}  // namespace yah::model
