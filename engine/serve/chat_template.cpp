#include "serve/chat_template.hpp"

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace yah::serve {
namespace {

constexpr std::string_view kXhighInstructions =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
    "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
constexpr std::string_view kLowInstructions =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without "
    "unnecessary elaboration.";
constexpr std::string_view kToolsHeader = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
constexpr std::string_view kToolsFormat =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

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

// Python's float repr: the shortest round-trip digits, positional while the exponent is in [-5, 15].
void PyFloat(std::string& out, double d) {
  char buf[40];
  const auto r = std::to_chars(buf, buf + sizeof buf, d, std::chars_format::scientific);
  const std::string_view t(buf, static_cast<std::size_t>(r.ptr - buf));
  const std::size_t epos = t.find('e');
  std::size_t at = 0;
  if (t[0] == '-') out += '-', at = 1;
  std::string digits;
  for (std::size_t i = at; i < epos; ++i)
    if (t[i] != '.') digits += t[i];
  int x = 0;
  std::from_chars(t.data() + epos + (t[epos + 1] == '+' ? 2 : 1), t.data() + t.size(), x);
  const int k = static_cast<int>(digits.size()), n = x + 1;  // the value is 0.d1d2... * 10^n
  if (0 < n && n <= 16) {
    if (k <= n) {
      out += digits, out.append(static_cast<std::size_t>(n - k), '0'), out += ".0";
    } else {
      out += digits.substr(0, static_cast<std::size_t>(n)), out += '.',
          out += digits.substr(static_cast<std::size_t>(n));
    }
  } else if (-4 < n && n <= 0) {
    out += "0.", out.append(static_cast<std::size_t>(-n), '0'), out += digits;
  } else {
    out += digits[0];
    if (k > 1) out += '.', out += digits.substr(1);
    char e[16];
    std::snprintf(e, sizeof e, "e%c%02d", x < 0 ? '-' : '+', std::abs(x));
    out += e;
  }
}

void PyJsonTo(std::string& out, const core::Json& v) {
  using Kind = core::Json::Kind;
  switch (v.kind()) {
    case Kind::kDouble:
      PyFloat(out, v.get<double>());
      break;
    case Kind::kArray:
      out += '[';
      for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out += ", ";
        PyJsonTo(out, v[i]);
      }
      out += ']';
      break;
    case Kind::kObject: {
      out += '{';
      bool first = true;
      for (const auto& [key, value] : v.items()) {
        if (!first) out += ", ";
        first = false;
        out += core::Json(key).dump(-1, ' ', false, core::Json::error_handler_t::replace) + ": ";
        PyJsonTo(out, value);
      }
      out += '}';
      break;
    }
    default:  // null, bool, int and string print as compact JSON does
      out += v.dump(-1, ' ', false, core::Json::error_handler_t::replace);
  }
}

}  // namespace

std::string PyJson(const core::Json& value) {
  std::string out;
  PyJsonTo(out, value);
  return out;
}

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
  bool has_system = false;
  std::vector<const ChatMessage*> turns;
  for (const ChatMessage& m : messages) {
    if (m.role == "system" || m.role == "developer") {
      has_system = true;
      const std::string text = TrimPy(m.content);
      if (!text.empty() && !system.empty()) system += "\n\n";
      system += text;
    } else if (m.role == "user" || m.role == "assistant" || m.role == "tool") {
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
  if (options.tools.is_array() && !options.tools.empty()) {
    out += "<|im_start|>system\n";
    if (!instructions.empty()) out += std::string(instructions) + "\n\n";
    out += kToolsHeader;
    for (const core::Json& tool : options.tools) out += "\n" + PyJson(tool);
    out += kToolsFormat;
    if (!system.empty()) out += "\n\n" + system;
    out += "<|im_end|>\n";
  } else if (!system.empty()) {
    out += "<|im_start|>system\n";
    if (!instructions.empty()) out += std::string(instructions) + "\n\n";
    out += system + "<|im_end|>\n";
  } else if (!instructions.empty()) {
    out += "<|im_start|>system\n" + std::string(instructions) + "<|im_end|>\n";
  }
  for (std::size_t i = 0; i < turns.size(); ++i) {
    const ChatMessage& m = *turns[i];
    const std::string content = TrimPy(m.content);
    if (m.role == "user") {
      out += "<|im_start|>user\n" + content + "<|im_end|>\n";
    } else if (m.role == "assistant") {
      out += "<|im_start|>assistant\n<think>\n" + TrimPy(m.reasoning) + "\n</think>\n\n" + content;
      for (std::size_t c = 0; c < m.tool_calls.size(); ++c) {
        const ToolCall& call = m.tool_calls[c];
        out += c > 0             ? "\n<tool_call>\n<function="
               : content.empty() ? "<tool_call>\n<function="
                                 : "\n\n<tool_call>\n<function=";
        out += call.name + ">\n";
        for (const auto& [key, value] : call.arguments.items()) {
          out += "<parameter=" + key + ">\n" +
                 (value.is_string() ? value.get_ref<const std::string&>() : PyJson(value)) + "\n</parameter>\n";
        }
        out += "</function>\n</tool_call>";
      }
      out += "<|im_end|>\n";
    } else {
      // Consecutive tool results share one user turn. (The template opens none for a tool message that comes first.)
      if (i > 0 ? turns[i - 1]->role != "tool" : has_system) out += "<|im_start|>user";
      out += "\n<tool_response>\n" + content + "\n</tool_response>";
      if (i + 1 == turns.size() || turns[i + 1]->role != "tool") out += "<|im_end|>\n";
    }
  }
  if (options.add_generation_prompt) {
    out += "<|im_start|>assistant\n";
    out += options.effort == ReasoningEffort::kNone ? "<think>\n\n</think>\n\n" : "<think>\n";
  }
  return out;
}

}  // namespace yah::serve
