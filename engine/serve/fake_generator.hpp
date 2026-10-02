// FakeGenerator: a CPU-only TextGenerator for server tests. It uses the real tokenizer and streams a canned reply.
#ifndef YAH_SERVE_FAKE_GENERATOR_HPP_
#define YAH_SERVE_FAKE_GENERATOR_HPP_

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "model/generator.hpp"

namespace yah::serve {

class FakeGenerator : public model::TextGenerator {
 public:
  // The reply has a character that the tokenizer splits into byte tokens, to test UTF-8 reassembly.
  // It is encoded without special tokens, so </think> arrives split over three tokens.
  static constexpr const char* kReasoning = "The user wants a reply. I give the canned one.";
  static constexpr const char* kAnswer = "Hello! This is a fake reply. \xF0\x9D\x94\x98 caf\xC3\xA9.";
  static constexpr std::uint32_t kContext = 4096;

  explicit FakeGenerator(const std::string& gguf_path)
      : gguf_(core::Gguf::Open(gguf_path)),
        tokenizer_(core::Tokenizer::FromGguf(gguf_, core::TokenizerConfig::FromGguf(gguf_))),
        name_(std::filesystem::path(gguf_path).stem().string()) {}

  [[nodiscard]] const core::Tokenizer& tokenizer() const override { return tokenizer_; }
  [[nodiscard]] std::uint32_t context() const override { return kContext; }
  [[nodiscard]] std::string model_name() const override { return name_; }

  model::GenerateResult Generate(const std::vector<core::TokenId>& prompt, const model::GenerateParams& params,
                                 const std::function<bool(core::TokenId)>& on_token) override {
    model::GenerateResult result;
    result.prompt_tokens = static_cast<std::uint32_t>(prompt.size());
    result.finish_reason = "length";
    // A prompt that ends in an open <think> block gets reasoning first, like the real model.
    const std::size_t tail = std::min<std::size_t>(prompt.size(), 4);
    const std::string end = tokenizer_.Decode(std::vector<core::TokenId>(prompt.end() - tail, prompt.end()));
    const bool thinking = end.ends_with("<think>\n");
    const std::string text = thinking ? std::string(kReasoning) + "\n</think>\n\n" + kAnswer : kAnswer;
    std::vector<core::TokenId> reply = tokenizer_.Encode(text, {.parse_special_tokens = false});
    reply.push_back(*tokenizer_.FindSpecial("<|im_end|>"));

    const auto t0 = std::chrono::steady_clock::now();
    for (const core::TokenId id : reply) {
      if (std::find(params.stop_ids.begin(), params.stop_ids.end(), id) != params.stop_ids.end()) {
        result.finish_reason = "stop";
        break;
      }
      if (result.generated_tokens >= params.max_tokens) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
      ++result.generated_tokens;
      if (!on_token(id)) {
        result.finish_reason = "stop";
        break;
      }
    }
    result.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return result;
  }

 private:
  core::Gguf gguf_;
  core::Tokenizer tokenizer_;
  std::string name_;
};

}  // namespace yah::serve

#endif  // YAH_SERVE_FAKE_GENERATOR_HPP_
