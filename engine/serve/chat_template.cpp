#include "serve/chat_template.hpp"

#include <stdexcept>

namespace yah::serve {
namespace {

constexpr std::string_view kXhighInstructions =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
    "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
constexpr std::string_view kLowInstructions =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without "
    "unnecessary elaboration.";

// The code points Python's str.isspace() accepts.
bool IsPySpace(char32_t c) {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// Decodes the code point at text[i] and returns its byte length. An invalid byte decodes as itself, length 1.
std::size_t DecodeAt(std::string_view text, std::size_t i, char32_t* cp) {
  const auto b = static_cast<unsigned char>(text[i]);
  std::size_t len = b < 0x80 ? 1 : (b >> 5) == 0x6 ? 2 : (b >> 4) == 0xE ? 3 : (b >> 3) == 0x1E ? 4 : 0;
  if (len == 0 || i + len > text.size()) {
    *cp = b;
    return 1;
  }
  char32_t c = len == 1 ? b : b & (0x7F >> len);
  for (std::size_t k = 1; k < len; ++k) c = (c << 6) | (static_cast<unsigned char>(text[i + k]) & 0x3F);
  *cp = c;
  return len;
}

}  // namespace

std::string TrimPy(std::string_view text) {
  std::size_t begin = text.size();
  std::size_t end = 0;
  for (std::size_t i = 0; i < text.size();) {
    char32_t c = 0;
    const std::size_t len = DecodeAt(text, i, &c);
    if (!IsPySpace(c)) {
      if (begin == text.size()) begin = i;
      end = i + len;
    }
    i += len;
  }
  if (begin >= end) return {};
  return std::string(text.substr(begin, end - begin));
}

std::string RenderChat(const std::vector<ChatMessage>& messages, const ChatOptions& options) {
  if (messages.empty()) throw std::invalid_argument("No messages provided.");
  std::string system;
  std::vector<const ChatMessage*> turns;
  for (const ChatMessage& m : messages) {
    if (m.role == "system" || m.role == "developer") {
      const std::string text = TrimPy(m.content);
      if (!text.empty() && !system.empty()) system += "\n\n";
      system += text;
    } else if (m.role == "user" || m.role == "assistant") {
      turns.push_back(&m);
    } else {
      throw std::invalid_argument("Unexpected message role.");
    }
  }

  // A user turn that is only a tool response does not count as a query.
  bool has_query = false;
  for (const ChatMessage* m : turns) {
    if (m->role != "user") continue;
    const std::string text = TrimPy(m->content);
    if (!(text.starts_with("<tool_response>") && text.ends_with("</tool_response>"))) has_query = true;
  }
  if (!has_query) throw std::invalid_argument("No user query found in messages.");

  std::string_view instructions;
  if (options.effort == ReasoningEffort::kHigh) instructions = kXhighInstructions;
  if (options.effort == ReasoningEffort::kLow) instructions = kLowInstructions;

  std::string out;
  if (!system.empty()) {
    out += "<|im_start|>system\n";
    if (!instructions.empty()) out += std::string(instructions) + "\n\n";
    out += system + "<|im_end|>\n";
  } else if (!instructions.empty()) {
    out += "<|im_start|>system\n" + std::string(instructions) + "<|im_end|>\n";
  }
  for (const ChatMessage* m : turns) {
    if (m->role == "user") {
      out += "<|im_start|>user\n" + TrimPy(m->content) + "<|im_end|>\n";
    } else {
      out += "<|im_start|>assistant\n<think>\n" + TrimPy(m->reasoning) + "\n</think>\n\n" + TrimPy(m->content) +
             "<|im_end|>\n";
    }
  }
  if (options.add_generation_prompt) {
    out += "<|im_start|>assistant\n";
    out += options.effort == ReasoningEffort::kNone ? "<think>\n\n</think>\n\n" : "<think>\n";
  }
  return out;
}

}  // namespace yah::serve
