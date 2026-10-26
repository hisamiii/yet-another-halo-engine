// The OpenAI Chat Completions API (POST /v1/chat/completions) on top of model::TextGenerator: request parsing, stop
// strings, and the chat.completion object and its chat.completion.chunk stream.
// Reasoning goes out as reasoning_content (llama.cpp's and vLLM's field), the answer as content, and with tools the
// model's calls as tool_calls, each in one chunk once it is complete.
#ifndef YAH_SERVE_CHAT_COMPLETIONS_HPP_
#define YAH_SERVE_CHAT_COMPLETIONS_HPP_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "model/generator.hpp"
#include "serve/chat_template.hpp"
#include "serve/responses.hpp"

namespace yah::serve {

struct ChatRequest {
  std::string model;
  std::vector<ChatMessage> messages;
  ChatOptions chat;
  std::optional<std::uint32_t> max_tokens;  // max_completion_tokens or max_tokens
  model::SamplingParams sampling;
  std::vector<std::string> stop;
  bool stream = false;
  bool include_usage = false;  // stream_options.include_usage
  bool ignore_eos = false;     // llama.cpp's and vLLM's: generate max_tokens tokens past the end of turn (benchmarks)
};

// llama.cpp's timings object: prompt_n tokens prefilled (the cached ones are cache_n) in prompt_ms, to the first token;
// predicted_n generated, the 2nd to the last in predicted_ms.
[[nodiscard]] Json Timings(const model::GenerateResult& result);

// Defaults are Qwen3.8's recommended sampling (temperature 0.6, top_p 0.95, top_k 20). Throws ApiError for bad or
// unsupported input.
[[nodiscard]] ChatRequest ParseChatRequest(const Json& body);

// Holds back generated text that may be the start of a stop string; text up to a stop string goes out, the string
// itself and everything after it do not (OpenAI's stop).
class StopMatcher {
 public:
  explicit StopMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {}
  // The text that is safe to emit now.
  std::string Feed(std::string_view bytes);
  // The held text, once generation ended without a stop string.
  std::string Finish();
  [[nodiscard]] bool stopped() const { return stopped_; }

 private:
  std::vector<std::string> stops_;
  std::string held_;
  bool stopped_ = false;
};

// One /v1/chat/completions generation. Builds the chat.completion object and, when emit is set, the chunk stream.
// emit gets each chunk already framed as SSE and returns false when the client is gone.
class ChatStream {
 public:
  ChatStream(const ChatRequest& request, std::function<bool(const std::string&)> emit);
  // Each returns false once the client is gone.
  bool Start();
  bool OnToken(std::string_view bytes);
  bool Finish(const model::GenerateResult& result);
  void Fail(const std::string& message);
  [[nodiscard]] Json Response() const;
  [[nodiscard]] const std::string& id() const { return id_; }
  [[nodiscard]] bool client_alive() const { return alive_; }
  // A stop string ended the output: the generator stops at the next token.
  [[nodiscard]] bool stopped() const { return stop_.stopped(); }

 private:
  bool EmitChunk(Json delta, const Json& finish_reason, const Json* usage = nullptr);
  void Apply(const OutputParser::Chunk& chunk);
  void Content(const std::string& text);
  void ApplyTools(ToolCallParser::Chunk chunk);
  static Json CallObject(const ToolCall& call);

  const ChatRequest& request_;
  std::function<bool(const std::string&)> emit_;
  StopMatcher stop_;
  OutputParser parser_;
  std::optional<ToolCallParser> tool_parser_;  // with tools
  std::vector<ToolCall> calls_;
  std::string id_;
  std::int64_t created_;
  std::string reasoning_, answer_;
  std::string finish_reason_;
  Json usage_, timings_;
  bool alive_ = true;
};

}  // namespace yah::serve

#endif  // YAH_SERVE_CHAT_COMPLETIONS_HPP_
