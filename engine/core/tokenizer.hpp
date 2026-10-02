#ifndef YAH_CORE_TOKENIZER_HPP_
#define YAH_CORE_TOKENIZER_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"

namespace yah::core {

using TokenId = std::uint32_t;
inline constexpr TokenId kInvalidTokenId = 0xFFFFFFFFU;

struct TokenizerOptions {
  bool add_bos{false};
  bool add_eos{false};
  bool parse_special_tokens{true};
};

// Byte-level BPE for Qwen3.8, matching the HF tokenizer. The "qwen35"
// pre-tokenizer is a Unicode split with contractions, so it needs ICU for the
// general-category predicates and for NFC. Ported from the reference engine,
// which validates it against HF golden token hashes.
class Tokenizer {
 public:
  static Tokenizer FromGguf(const Gguf& gguf, const TokenizerConfig& config);

  [[nodiscard]] std::vector<TokenId> Encode(std::string_view text, const TokenizerOptions& options = {}) const;
  [[nodiscard]] std::string Decode(const std::vector<TokenId>& tokens) const;
  [[nodiscard]] std::string DecodeToken(TokenId id) const;
  [[nodiscard]] bool IsSpecial(TokenId id) const { return is_special_.contains(id); }
  [[nodiscard]] std::optional<TokenId> FindSpecial(std::string_view text) const;
  [[nodiscard]] std::size_t vocab_size() const { return id_to_token_.size(); }
  [[nodiscard]] TokenId bos_id() const { return bos_id_; }
  [[nodiscard]] TokenId eos_id() const { return eos_id_; }

 private:
  struct Merge {
    std::uint32_t rank{0};
    TokenId token{kInvalidTokenId};
  };

  [[nodiscard]] std::vector<TokenId> BpeMergeChunk(std::string_view chunk) const;
  [[nodiscard]] std::vector<TokenId> BpeEncodeText(std::string_view text) const;
  void InitializeDecodedTokens();

  std::vector<std::string> id_to_token_;
  std::vector<std::string> id_to_decoded_token_;
  std::unordered_map<std::string, TokenId> token_to_id_;
  std::unordered_map<std::uint64_t, Merge> merge_ranks_;
  std::array<TokenId, 256> byte_tokens_{};
  std::vector<std::pair<std::string, TokenId>> special_token_list_;
  std::unordered_map<std::string, TokenId> special_to_id_;
  std::unordered_set<TokenId> is_special_;
  TokenId bos_id_{kInvalidTokenId};
  TokenId eos_id_{kInvalidTokenId};
  bool qwen35_{false};
};

}  // namespace yah::core

#endif  // YAH_CORE_TOKENIZER_HPP_
