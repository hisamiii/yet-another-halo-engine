#include "core/tokenizer.hpp"

#include <unicode/bytestream.h>
#include <unicode/normalizer2.h>
#include <unicode/uchar.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>

namespace yah::core {
namespace {

int HexCharToInt(char c) noexcept {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// GPT-2 byte-level alphabet: every byte maps to a printable codepoint so the
// vocabulary can be stored as text. Bytes outside the printable ranges become
// U+0100 + n in order.
std::string Utf8FromCodepoint(int cp) {
  std::string out;
  out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
  out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  return out;
}

std::string ByteToGpt2Utf8(std::uint8_t b) {
  static const auto table = []() {
    std::array<std::string, 256> map;
    for (int i = '!'; i <= '~'; ++i) map[i] = std::string(1, static_cast<char>(i));
    // The three printable ranges map to their own codepoint; every other byte
    // maps to U+0100 + its index within the gap list, in order.
    for (int i = 161; i <= 172; ++i) map[i] = Utf8FromCodepoint(i);
    for (int i = 174; i <= 255; ++i) map[i] = Utf8FromCodepoint(i);
    int n = 0;
    for (int i = 0; i < 256; ++i) {
      if ((i < '!' || i > '~') && (i < 161 || i > 172) && (i < 174 || i > 255)) {
        map[i] = Utf8FromCodepoint(256 + n);
        ++n;
      }
    }
    return map;
  }();
  return table[b];
}

std::string UnescapeGpt2Bytes(std::string_view text) {
  std::string result;
  result.reserve(text.size());
  std::size_t i = 0;
  while (i < text.size()) {
    const auto b0 = static_cast<unsigned char>(text[i]);
    char32_t cp = b0;
    std::size_t len = 1;
    if ((b0 & 0xE0) == 0xC0 && i + 1 < text.size()) {
      cp = ((b0 & 0x1F) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3F);
      len = 2;
    } else if ((b0 & 0xF0) == 0xE0 && i + 2 < text.size()) {
      cp = ((b0 & 0x0F) << 12) | ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
           (static_cast<unsigned char>(text[i + 2]) & 0x3F);
      len = 3;
    }
    if (cp >= 0x100 && cp <= 0x1FF) {
      // Reverse of the b2u gap list: 0x100 + n maps back to the nth gap byte.
      int n = static_cast<int>(cp - 0x100);
      for (int b = 0; b < 256; ++b) {
        if ((b < '!' || b > '~') && (b < 161 || b > 172) && (b < 174 || b > 255)) {
          if (n == 0) {
            result.push_back(static_cast<char>(b));
            break;
          }
          --n;
        }
      }
    } else if (cp < 0x80 || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
      result.push_back(static_cast<char>(cp));
    } else {
      result.append(text.substr(i, len));
    }
    i += len;
  }
  return result;
}

struct Utf8CodePoint {
  UChar32 value{0};
  std::size_t length{1};
};

Utf8CodePoint DecodeUtf8(std::string_view text, std::size_t offset) noexcept {
  const auto first = static_cast<std::uint8_t>(text[offset]);
  if (first < 0x80U) return {.value = first, .length = 1};
  std::size_t length = 0;
  UChar32 value = 0;
  if ((first & 0xE0U) == 0xC0U) {
    length = 2;
    value = static_cast<UChar32>(first & 0x1FU);
  } else if ((first & 0xF0U) == 0xE0U) {
    length = 3;
    value = static_cast<UChar32>(first & 0x0FU);
  } else if ((first & 0xF8U) == 0xF0U) {
    length = 4;
    value = static_cast<UChar32>(first & 0x07U);
  } else {
    return {.value = first, .length = 1};
  }
  if (offset + length > text.size()) return {.value = first, .length = 1};
  for (std::size_t index = 1; index < length; ++index) {
    const auto byte = static_cast<std::uint8_t>(text[offset + index]);
    if ((byte & 0xC0U) != 0x80U) return {.value = first, .length = 1};
    value = static_cast<UChar32>((value << 6U) | (byte & 0x3FU));
  }
  return {.value = value, .length = length};
}

bool IsUnicodeLetter(UChar32 value) noexcept {
  return (U_GET_GC_MASK(value) & U_GC_L_MASK) != 0;
}
bool IsUnicodeLetterOrMark(UChar32 value) noexcept {
  return (U_GET_GC_MASK(value) & (U_GC_L_MASK | U_GC_M_MASK)) != 0;
}
bool IsUnicodeNumber(UChar32 value) noexcept {
  const auto category = static_cast<UCharCategory>(u_charType(value));
  return category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER || category == U_OTHER_NUMBER;
}
bool IsUnicodeWhitespace(UChar32 value) noexcept {
  return u_isUWhiteSpace(value) != 0;
}
bool IsNewline(UChar32 value) noexcept {
  return value == '\r' || value == '\n';
}

std::size_t Qwen35ContractionEnd(std::string_view text, std::size_t offset) noexcept {
  if (text[offset] != '\'') return offset;
  constexpr std::array<std::string_view, 7> kSuffixes = {"s", "t", "re", "ve", "m", "ll", "d"};
  for (const std::string_view suffix : kSuffixes) {
    std::size_t end = offset + 1;
    bool matches = true;
    for (const char letter : suffix) {
      if (end == text.size()) {
        matches = false;
        break;
      }
      const auto next = DecodeUtf8(text, end);
      if (u_foldCase(next.value, U_FOLD_CASE_DEFAULT) != letter) {
        matches = false;
        break;
      }
      end += next.length;
    }
    if (matches) return end;
  }
  return offset;
}

std::size_t Qwen35PieceEnd(std::string_view text, std::size_t offset) {
  if (const std::size_t contraction = Qwen35ContractionEnd(text, offset); contraction != offset) {
    return contraction;
  }
  const Utf8CodePoint first = DecodeUtf8(text, offset);
  if (IsUnicodeLetterOrMark(first.value)) {
    std::size_t end = offset + first.length;
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (!IsUnicodeLetterOrMark(next.value)) break;
      end += next.length;
    }
    return end;
  }
  if (!IsNewline(first.value) && !IsUnicodeLetter(first.value) && !IsUnicodeNumber(first.value) &&
      offset + first.length < text.size()) {
    const Utf8CodePoint next = DecodeUtf8(text, offset + first.length);
    if (IsUnicodeLetterOrMark(next.value)) {
      std::size_t end = offset + first.length + next.length;
      while (end < text.size()) {
        const Utf8CodePoint letter = DecodeUtf8(text, end);
        if (!IsUnicodeLetterOrMark(letter.value)) break;
        end += letter.length;
      }
      return end;
    }
  }
  if (IsUnicodeNumber(first.value)) return offset + first.length;

  std::size_t punctuation_start = offset;
  if (first.value == ' ' && offset + first.length < text.size()) {
    const Utf8CodePoint next = DecodeUtf8(text, offset + first.length);
    if (!IsUnicodeWhitespace(next.value) && !IsUnicodeLetterOrMark(next.value) && !IsUnicodeNumber(next.value)) {
      punctuation_start += first.length;
    }
  }
  const Utf8CodePoint punctuation = DecodeUtf8(text, punctuation_start);
  if (!IsUnicodeWhitespace(punctuation.value) && !IsUnicodeLetterOrMark(punctuation.value) &&
      !IsUnicodeNumber(punctuation.value)) {
    std::size_t end = punctuation_start;
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (IsUnicodeWhitespace(next.value) || IsUnicodeLetterOrMark(next.value) || IsUnicodeNumber(next.value)) {
        break;
      }
      end += next.length;
    }
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (!IsNewline(next.value)) break;
      end += next.length;
    }
    return end;
  }
  if (IsUnicodeWhitespace(first.value)) {
    std::size_t end = offset;
    std::size_t last_start = offset;
    std::size_t last_newline_end = offset;
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (!IsUnicodeWhitespace(next.value)) break;
      last_start = end;
      end += next.length;
      if (IsNewline(next.value)) last_newline_end = end;
    }
    if (last_newline_end != offset) return last_newline_end;
    return end < text.size() && last_start > offset ? last_start : end;
  }
  return offset + first.length;
}

std::uint64_t MergeKey(TokenId left, TokenId right) {
  return (static_cast<std::uint64_t>(left) << 32) | right;
}

}  // namespace

Tokenizer Tokenizer::FromGguf(const Gguf& gguf, const TokenizerConfig& config) {
  (void)gguf;
  if (config.tokens == nullptr) {
    throw std::runtime_error("tokenizer: vocabulary missing from metadata");
  }
  Tokenizer tokenizer;
  const auto& tokens = *config.tokens;
  tokenizer.id_to_token_.reserve(tokens.size());
  tokenizer.token_to_id_.reserve(tokens.size());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokenizer.id_to_token_.push_back(tokens[i].s);
    tokenizer.token_to_id_[tokens[i].s] = static_cast<TokenId>(i);
  }

  if (config.merges != nullptr) {
    const auto& merges = *config.merges;
    tokenizer.merge_ranks_.reserve(merges.size());
    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
      const std::string& merge = merges[rank].s;
      const auto space = merge.find(' ');
      if (space == std::string::npos) continue;
      const std::string part1 = merge.substr(0, space);
      const std::string part2 = merge.substr(space + 1);
      const auto it1 = tokenizer.token_to_id_.find(part1);
      const auto it2 = tokenizer.token_to_id_.find(part2);
      if (it1 == tokenizer.token_to_id_.end() || it2 == tokenizer.token_to_id_.end()) {
        continue;
      }
      const auto merged = tokenizer.token_to_id_.find(part1 + part2);
      tokenizer.merge_ranks_[MergeKey(it1->second, it2->second)] = {
          static_cast<std::uint32_t>(rank), merged != tokenizer.token_to_id_.end() ? merged->second : kInvalidTokenId};
    }
  }

  // Special tokens are the CONTROL (3) and USER_DEFINED (4) entries of the
  // GGUF token type array. bos/eos are added regardless so the scanner can emit
  // them even if a shard marks them differently.
  if (config.token_types != nullptr) {
    const auto& types = *config.token_types;
    for (std::size_t i = 0; i < types.size() && i < tokenizer.id_to_token_.size(); ++i) {
      if (types[i].kind != MetadataValue::Kind::kInt) continue;
      if (types[i].i == 3 || types[i].i == 4) {
        tokenizer.is_special_.insert(static_cast<TokenId>(i));
      }
    }
  }
  tokenizer.is_special_.insert(config.bos_id);
  tokenizer.is_special_.insert(config.eos_id);
  tokenizer.bos_id_ = config.bos_id;
  tokenizer.eos_id_ = config.eos_id;
  for (const TokenId id : tokenizer.is_special_) {
    if (id >= tokenizer.id_to_token_.size()) continue;
    tokenizer.special_token_list_.emplace_back(tokenizer.id_to_token_[id], id);
    tokenizer.special_to_id_[tokenizer.id_to_token_[id]] = id;
  }

  for (int b = 0; b < 256; ++b) {
    const std::string key = ByteToGpt2Utf8(static_cast<std::uint8_t>(b));
    const auto it = tokenizer.token_to_id_.find(key);
    tokenizer.byte_tokens_[b] = it != tokenizer.token_to_id_.end() ? it->second : kInvalidTokenId;
  }

  tokenizer.qwen35_ = config.pre == "qwen35";
  tokenizer.InitializeDecodedTokens();
  return tokenizer;
}

void Tokenizer::InitializeDecodedTokens() {
  id_to_decoded_token_.resize(id_to_token_.size());
  for (std::size_t i = 0; i < id_to_token_.size(); ++i) {
    if (is_special_.contains(static_cast<TokenId>(i))) {
      id_to_decoded_token_[i] = id_to_token_[i];
      continue;
    }
    const auto& token = id_to_token_[i];
    if (token.size() == 6 && token.starts_with("<0x") && token.ends_with('>')) {
      const int high = HexCharToInt(token[3]);
      const int low = HexCharToInt(token[4]);
      if (high >= 0 && low >= 0) {
        id_to_decoded_token_[i] = std::string(1, static_cast<char>(static_cast<std::uint8_t>((high << 4) | low)));
        continue;
      }
    }
    id_to_decoded_token_[i] = UnescapeGpt2Bytes(token);
  }
}

std::vector<TokenId> Tokenizer::BpeMergeChunk(std::string_view chunk) const {
  if (chunk.empty()) return {};
  std::vector<TokenId> word_tokens;
  word_tokens.reserve(chunk.size());
  for (const unsigned char c : chunk) {
    const TokenId tid = byte_tokens_[c];
    if (tid != kInvalidTokenId) word_tokens.push_back(tid);
  }
  if (word_tokens.size() <= 1) return word_tokens;

  constexpr std::size_t kNoSymbol = std::numeric_limits<std::size_t>::max();
  const std::size_t count = word_tokens.size();
  std::vector<std::size_t> previous(count);
  std::vector<std::size_t> following(count);
  for (std::size_t i = 0; i < count; ++i) {
    previous[i] = i == 0 ? kNoSymbol : i - 1;
    following[i] = i + 1 < count ? i + 1 : kNoSymbol;
  }

  struct Candidate {
    std::uint32_t rank;
    std::size_t index;
    TokenId left;
    TokenId right;
  };
  const auto later = [](const Candidate& a, const Candidate& b) {
    return a.rank != b.rank ? a.rank > b.rank : a.index > b.index;
  };
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(later)> candidates(later);

  const auto offer = [&](std::size_t index, std::size_t right) {
    const auto it = merge_ranks_.find(MergeKey(word_tokens[index], word_tokens[right]));
    if (it != merge_ranks_.end()) {
      candidates.push({it->second.rank, index, word_tokens[index], word_tokens[right]});
    }
  };
  for (std::size_t i = 0; i + 1 < count; ++i) offer(i, i + 1);

  while (!candidates.empty()) {
    const Candidate best = candidates.top();
    candidates.pop();
    const std::size_t right = following[best.index];
    if (word_tokens[best.index] != best.left || right == kNoSymbol || word_tokens[right] != best.right) {
      continue;
    }
    const auto it = merge_ranks_.find(MergeKey(best.left, best.right));
    if (it == merge_ranks_.end() || it->second.token == kInvalidTokenId) break;
    word_tokens[best.index] = it->second.token;
    const std::size_t after = following[right];
    following[best.index] = after;
    if (after != kNoSymbol) previous[after] = best.index;
    word_tokens[right] = kInvalidTokenId;
    if (previous[best.index] != kNoSymbol) offer(previous[best.index], best.index);
    if (after != kNoSymbol) offer(best.index, after);
  }

  std::vector<TokenId> merged;
  merged.reserve(count);
  for (std::size_t i = 0; i != kNoSymbol; i = following[i]) {
    merged.push_back(word_tokens[i]);
  }
  return merged;
}

std::vector<TokenId> Tokenizer::BpeEncodeText(std::string_view text) const {
  if (!qwen35_) return BpeMergeChunk(text);

  std::string normalized;
  if (std::any_of(text.begin(), text.end(), [](unsigned char byte) { return byte >= 0x80; })) {
    UErrorCode status = U_ZERO_ERROR;
    const auto* nfc = icu::Normalizer2::getNFCInstance(status);
    if (U_FAILURE(status)) {
      throw std::runtime_error("tokenizer: NFC instance unavailable");
    }
    icu::StringByteSink<std::string> sink(&normalized);
    nfc->normalizeUTF8(0, icu::StringPiece(text.data(), static_cast<int32_t>(text.size())), sink, nullptr, status);
    if (U_FAILURE(status)) {
      throw std::runtime_error("tokenizer: NFC normalization failed");
    }
    text = normalized;
  }

  std::vector<TokenId> tokens;
  std::size_t offset = 0;
  while (offset < text.size()) {
    const std::size_t end = Qwen35PieceEnd(text, offset);
    const auto piece_tokens = BpeMergeChunk(text.substr(offset, end - offset));
    tokens.insert(tokens.end(), piece_tokens.begin(), piece_tokens.end());
    offset = end;
  }
  return tokens;
}

std::vector<TokenId> Tokenizer::Encode(std::string_view text, const TokenizerOptions& options) const {
  std::vector<TokenId> tokens;
  if (options.add_bos && bos_id_ != kInvalidTokenId) tokens.push_back(bos_id_);
  if (text.empty()) {
    if (options.add_eos && eos_id_ != kInvalidTokenId) tokens.push_back(eos_id_);
    return tokens;
  }

  if (!options.parse_special_tokens || special_token_list_.empty()) {
    const auto chunk = BpeEncodeText(text);
    tokens.insert(tokens.end(), chunk.begin(), chunk.end());
  } else {
    std::vector<std::size_t> occurrence(special_token_list_.size());
    for (std::size_t i = 0; i < special_token_list_.size(); ++i) {
      occurrence[i] = text.find(special_token_list_[i].first);
    }
    std::size_t pos = 0;
    while (pos < text.size()) {
      std::size_t best_pos = std::string_view::npos;
      std::string_view best_text;
      TokenId best_id = kInvalidTokenId;
      for (std::size_t i = 0; i < special_token_list_.size(); ++i) {
        const auto& entry = special_token_list_[i];
        if (occurrence[i] != std::string_view::npos && occurrence[i] < pos) {
          occurrence[i] = text.find(entry.first, pos);
        }
        const std::size_t found = occurrence[i];
        if (found == std::string_view::npos) continue;
        if (best_pos == std::string_view::npos || found < best_pos ||
            (found == best_pos && entry.first.size() > best_text.size())) {
          best_pos = found;
          best_text = entry.first;
          best_id = entry.second;
        }
      }
      if (best_pos == std::string_view::npos) {
        const auto chunk = BpeEncodeText(text.substr(pos));
        tokens.insert(tokens.end(), chunk.begin(), chunk.end());
        break;
      }
      if (best_pos > pos) {
        const auto chunk = BpeEncodeText(text.substr(pos, best_pos - pos));
        tokens.insert(tokens.end(), chunk.begin(), chunk.end());
      }
      tokens.push_back(best_id);
      pos = best_pos + best_text.size();
    }
  }

  if (options.add_eos && eos_id_ != kInvalidTokenId) tokens.push_back(eos_id_);
  return tokens;
}

std::string Tokenizer::DecodeToken(TokenId id) const {
  if (id < id_to_decoded_token_.size()) return id_to_decoded_token_[id];
  if (id < id_to_token_.size()) return id_to_token_[id];
  return {};
}

std::string Tokenizer::Decode(const std::vector<TokenId>& tokens) const {
  std::string result;
  for (const TokenId id : tokens) result += DecodeToken(id);
  return result;
}

std::optional<TokenId> Tokenizer::FindSpecial(std::string_view text) const {
  const auto it = special_to_id_.find(std::string(text));
  if (it == special_to_id_.end()) return std::nullopt;
  return it->second;
}

}  // namespace yah::core
