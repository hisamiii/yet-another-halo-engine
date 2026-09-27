#include "core/config.hpp"

#include <stdexcept>

namespace yah::core {
namespace {

const MetadataValue& Require(const Gguf& gguf, const std::string& key) {
  const auto* value = gguf.Meta(key);
  if (value == nullptr) {
    throw std::runtime_error("config: missing metadata key " + key);
  }
  return *value;
}

std::uint32_t U32(const Gguf& gguf, const std::string& key) {
  const auto& value = Require(gguf, key);
  if (value.kind != MetadataValue::Kind::kUInt) {
    throw std::runtime_error("config: " + key + " is not an unsigned integer");
  }
  return static_cast<std::uint32_t>(value.u);
}

float F32(const Gguf& gguf, const std::string& key) {
  const auto& value = Require(gguf, key);
  if (value.kind != MetadataValue::Kind::kFloat) {
    throw std::runtime_error("config: " + key + " is not a float");
  }
  return static_cast<float>(value.f);
}

std::string Str(const Gguf& gguf, const std::string& key) {
  const auto& value = Require(gguf, key);
  if (value.kind != MetadataValue::Kind::kString) {
    throw std::runtime_error("config: " + key + " is not a string");
  }
  return value.s;
}

const std::vector<MetadataValue>* Array(const Gguf& gguf,
                                        const std::string& key) {
  const auto* value = gguf.Meta(key);
  if (value == nullptr) return nullptr;
  if (value->kind != MetadataValue::Kind::kArray) {
    throw std::runtime_error("config: " + key + " is not an array");
  }
  return &value->array;
}

std::uint32_t U32Or(const Gguf& gguf, const std::string& key,
                       std::uint32_t fallback) {
  const auto* value = gguf.Meta(key);
  if (value == nullptr) return fallback;
  if (value->kind != MetadataValue::Kind::kUInt) {
    throw std::runtime_error("config: " + key + " is not an unsigned integer");
  }
  return static_cast<std::uint32_t>(value->u);
}

std::string StrOr(const Gguf& gguf, const std::string& key,
                  const std::string& fallback) {
  const auto* value = gguf.Meta(key);
  if (value == nullptr) return fallback;
  if (value->kind != MetadataValue::Kind::kString) {
    throw std::runtime_error("config: " + key + " is not a string");
  }
  return value->s;
}

std::string Key(const std::string& prefix, const char* suffix) {
  return prefix + suffix;
}

}  // namespace

Qwen35Config Qwen35Config::FromGguf(const Gguf& gguf) {
  Qwen35Config config;
  config.architecture = Str(gguf, "general.architecture");
  const std::string prefix = config.architecture + ".";
  config.block_count = U32(gguf, Key(prefix, "block_count"));
  config.context_length = U32(gguf, Key(prefix, "context_length"));
  config.embedding_length = U32(gguf, Key(prefix, "embedding_length"));
  config.feed_forward_length = U32(gguf, Key(prefix, "feed_forward_length"));
  config.head_count = U32(gguf, Key(prefix, "attention.head_count"));
  config.head_count_kv = U32(gguf, Key(prefix, "attention.head_count_kv"));
  config.key_length = U32(gguf, Key(prefix, "attention.key_length"));
  config.value_length = U32(gguf, Key(prefix, "attention.value_length"));
  config.rope_dimension_count = U32(gguf, Key(prefix, "rope.dimension_count"));
  config.rope_freq_base = F32(gguf, Key(prefix, "rope.freq_base"));
  config.rms_eps = F32(gguf, Key(prefix, "attention.layer_norm_rms_epsilon"));
  config.full_attention_interval =
      U32(gguf, Key(prefix, "full_attention_interval"));
  config.nextn_predict_layers =
      U32Or(gguf, Key(prefix, "nextn_predict_layers"), 0);
  config.ssm_conv_kernel = U32(gguf, Key(prefix, "ssm.conv_kernel"));
  config.ssm_state_size = U32(gguf, Key(prefix, "ssm.state_size"));
  config.ssm_group_count = U32(gguf, Key(prefix, "ssm.group_count"));
  config.ssm_time_step_rank = U32(gguf, Key(prefix, "ssm.time_step_rank"));
  config.ssm_inner_size = U32(gguf, Key(prefix, "ssm.inner_size"));
  return config;
}

TokenizerConfig TokenizerConfig::FromGguf(const Gguf& gguf) {
  // Only the model name and the two special ids are assumed present. The other
  // shard on this machine omits add_bos_token entirely, so every other key
  // falls back rather than failing the load.
  TokenizerConfig tokenizer;
  tokenizer.model = Str(gguf, "tokenizer.ggml.model");
  tokenizer.pre = StrOr(gguf, "tokenizer.ggml.pre", "");
  tokenizer.bos_id = U32(gguf, "tokenizer.ggml.bos_token_id");
  tokenizer.eos_id = U32(gguf, "tokenizer.ggml.eos_token_id");
  tokenizer.padding_id =
      U32Or(gguf, "tokenizer.ggml.padding_token_id", tokenizer.bos_id);
  if (const auto* add_bos = gguf.Meta("tokenizer.ggml.add_bos_token")) {
    if (add_bos->kind != MetadataValue::Kind::kBool) {
      throw std::runtime_error("config: add_bos_token is not a bool");
    }
    tokenizer.add_bos = add_bos->b;
  }
  tokenizer.chat_template =
      StrOr(gguf, "tokenizer.chat_template", "");
  tokenizer.tokens = Array(gguf, "tokenizer.ggml.tokens");
  tokenizer.token_types = Array(gguf, "tokenizer.ggml.token_type");
  tokenizer.merges = Array(gguf, "tokenizer.ggml.merges");
  return tokenizer;
}

}  // namespace yah::core
