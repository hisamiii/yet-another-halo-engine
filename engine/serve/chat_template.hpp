// The Qwen3.8 chat template (tokenizer.chat_template in the GGUF) in C++, for text chats with tools, without images.
#ifndef YAH_SERVE_CHAT_TEMPLATE_HPP_
#define YAH_SERVE_CHAT_TEMPLATE_HPP_

#include <string>
#include <string_view>
#include <vector>

#include "core/json.hpp"

namespace yah::serve {

// none: enable_thinking=false. low, medium: reasoning_effort=low / medium. high: the template default xhigh.
enum class ReasoningEffort { kNone, kLow, kMedium, kHigh };

struct ToolCall {
  std::string id;
  std::string name;
  core::Json arguments;  // an object
};

struct ChatMessage {
  std::string role;  // system, developer, user, assistant or tool
  std::string content;
  std::string reasoning;               // assistant only: the text inside <think>
  std::vector<ToolCall> tool_calls{};  // assistant only
};

struct ChatOptions {
  ReasoningEffort effort = ReasoningEffort::kHigh;
  bool add_generation_prompt = true;
  core::Json tools = core::Json::array();  // the request's tool definitions, as given
};

// Python str.strip(): Jinja's trim filter, which also strips Unicode whitespace such as U+00A0 and U+3000.
[[nodiscard]] std::string TrimPy(std::string_view text);

// Python's json.dumps(value, ensure_ascii=False): the template's tojson filter as transformers defines it.
[[nodiscard]] std::string PyJson(const core::Json& value);

// The same bytes as the Jinja template with preserve_thinking unset (true).
// System and developer messages merge into one leading system message, trimmed and joined by a blank line.
// Throws std::invalid_argument where the template raises an exception.
[[nodiscard]] std::string RenderChat(const std::vector<ChatMessage>& messages, const ChatOptions& options);

}  // namespace yah::serve

#endif  // YAH_SERVE_CHAT_TEMPLATE_HPP_
