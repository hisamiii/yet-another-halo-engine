#include "serve/responses.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <random>
#include <set>

namespace yah::serve {
namespace {

constexpr std::string_view kThinkEnd = "</think>";

// llama.cpp's string_strip whitespace.
bool IsSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

std::string StripSpace(std::string_view s, bool leading) {
  std::size_t begin = 0;
  std::size_t end = s.size();
  while (end > 0 && IsSpace(s[end - 1])) --end;
  if (leading) {
    while (begin < end && IsSpace(s[begin])) ++begin;
  }
  return std::string(s.substr(begin, end - begin));
}

std::vector<ChatMessage> ParseInput(const Json& input) {
  if (input.is_string()) return {ChatMessage{"user", input.get<std::string>(), ""}};
  if (!input.is_array() || input.empty()) {
    throw ApiError(400, "'input' must be a string or a non-empty array of messages.", "input");
  }
  std::vector<ChatMessage> messages;
  std::string reasoning;  // from reasoning items, for the next assistant message
  bool assistant_last = false;  // a function_call joins the assistant message before it
  for (std::size_t i = 0; i < input.size(); ++i) {
    const Json& item = input[i];
    const std::string param = "input[" + std::to_string(i) + "]";
    if (!item.is_object()) throw ApiError(400, "Each input item must be an object.", param);
    const Json* type = Field(item, "type");
    const std::string kind = type != nullptr && type->is_string() ? type->get<std::string>() : "message";
    if (kind == "reasoning") {
      const Json* parts = Field(item, "content");
      if (parts == nullptr || parts->empty()) parts = Field(item, "summary");
      if (parts != nullptr && parts->is_array()) {
        for (const Json& part : *parts) {
          if (part.is_object() && part.contains("text") && part["text"].is_string()) {
            reasoning += part["text"].get<std::string>();
          }
        }
      }
      continue;
    }
    if (kind == "function_call") {
      ToolCall call;
      const Json* name = Field(item, "name");
      if (name == nullptr || !name->is_string() || name->get<std::string>().empty()) {
        throw ApiError(400, "A function_call needs a 'name'.", param + ".name");
      }
      call.name = name->get<std::string>();
      if (const Json* id = Field(item, "call_id"); id != nullptr && id->is_string()) call.id = id->get<std::string>();
      call.arguments = Json::object();
      if (const Json* args = Field(item, "arguments")) {
        try {
          if (!args->is_string()) throw Json::parse_error("json: not a string");
          const std::string text = TrimPy(args->get<std::string>());
          if (!text.empty()) call.arguments = Json::parse(text);
        } catch (const Json::parse_error&) {
          call.arguments = nullptr;
        }
        if (!call.arguments.is_object()) {
          throw ApiError(400, "'arguments' must be a JSON object in a string.", param + ".arguments");
        }
      }
      if (!assistant_last) {
        messages.push_back({"assistant", "", std::move(reasoning)});
        reasoning.clear();
      }
      messages.back().tool_calls.push_back(std::move(call));
      assistant_last = true;
      continue;
    }
    if (kind == "function_call_output") {
      const Json* output = Field(item, "output");
      messages.push_back({"tool", output != nullptr ? ContentText(*output, param + ".output") : "", ""});
      assistant_last = false;
      continue;
    }
    if (kind != "message") {
      throw ApiError(400, "Unsupported input item type '" + kind + "'.", param + ".type", "unsupported_parameter");
    }
    const Json* role = Field(item, "role");
    if (role == nullptr || !role->is_string()) throw ApiError(400, "Input message needs a 'role'.", param + ".role");
    ChatMessage m{role->get<std::string>(), "", ""};
    if (m.role != "user" && m.role != "assistant" && m.role != "system" && m.role != "developer") {
      throw ApiError(400, "Unsupported role '" + m.role + "'.", param + ".role", "invalid_value");
    }
    m.content = ContentText(item.contains("content") ? item["content"] : Json(nullptr), param + ".content");
    if (m.role == "assistant") m.reasoning = std::move(reasoning);
    reasoning.clear();
    assistant_last = m.role == "assistant";
    messages.push_back(std::move(m));
  }
  return messages;
}

}  // namespace

std::string RandomId(const char* prefix) {
  thread_local std::mt19937_64 rng(std::random_device{}());
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%016llx%016llx", static_cast<unsigned long long>(rng()),
                static_cast<unsigned long long>(rng()));
  return std::string(prefix) + buf;
}

std::size_t CompleteUtf8(std::string_view s) {
  const std::size_t n = s.size();
  for (std::size_t back = 1; back <= 4 && back <= n; ++back) {
    const auto b = static_cast<unsigned char>(s[n - back]);
    if ((b & 0xC0) == 0x80) continue;
    const std::size_t len = (b >> 5) == 0x6 ? 2 : (b >> 4) == 0xE ? 3 : (b >> 3) == 0x1E ? 4 : 1;
    return len > back ? n - back : n;
  }
  return n;
}

const Json* Field(const Json& object, const char* key) {
  const auto it = object.find(key);
  if (it == object.end() || it->is_null()) return nullptr;
  return &*it;
}

bool IsEmptyValue(const Json& v) {
  return v.is_null() || (v.is_string() && v.get_ref<const std::string&>().empty()) ||
         ((v.is_array() || v.is_object()) && v.empty());
}

bool IsDefaultValue(const std::string& key, const Json& v) {
  if (key == "tool_choice") return v == "auto" || v == "none";
  if (key == "truncation") return v == "disabled";
  if (key == "background") return v == false;
  if (key == "top_logprobs") return v == 0;
  if (key == "text") return v == Json{{"format", {{"type", "text"}}}};
  return false;
}

double Number(const Json& body, const char* key, double fallback, double lo, double hi) {
  const Json* v = Field(body, key);
  if (v == nullptr) return fallback;
  if (!v->is_number()) throw ApiError(400, std::string("Invalid type for '") + key + "': expected a number.", key);
  const double x = v->get<double>();
  if (x < lo || x > hi) {
    char msg[128];
    std::snprintf(msg, sizeof(msg), "Invalid value for '%s': must be between %g and %g.", key, lo, hi);
    throw ApiError(400, msg, key, "invalid_value");
  }
  return x;
}

std::string Dump(const Json& value) {
  return value.dump(-1, ' ', false, Json::error_handler_t::replace);
}

Json ErrorBody(const ApiError& error) {
  return {{"error",
           {{"message", error.what()},
            {"type", error.type},
            {"param", error.param.empty() ? Json(nullptr) : Json(error.param)},
            {"code", error.code.empty() ? Json(nullptr) : Json(error.code)}}}};
}

std::string ContentText(const Json& content, const std::string& param) {
  if (content.is_null()) return {};
  if (content.is_string()) return content.get<std::string>();
  if (!content.is_array()) throw ApiError(400, "Content must be a string or an array of parts.", param);
  std::string text;
  for (std::size_t j = 0; j < content.size(); ++j) {
    const Json& part = content[j];
    const std::string where = param + "[" + std::to_string(j) + "]";
    const Json* type = part.is_object() ? Field(part, "type") : nullptr;
    const std::string kind = type != nullptr && type->is_string() ? type->get<std::string>() : "";
    if (kind != "input_text" && kind != "output_text" && kind != "text") {
      throw ApiError(400, "Unsupported content part type '" + kind + "'; only text is supported.", where + ".type",
                     "unsupported_parameter");
    }
    const Json* value = Field(part, "text");
    if (value == nullptr || !value->is_string())
      throw ApiError(400, "Text part needs a 'text' string.", where + ".text");
    text += value->get<std::string>();
  }
  return text;
}

ReasoningEffort ParseEffort(const std::string& effort) {
  if (effort == "none" || effort == "minimal") return ReasoningEffort::kNone;
  if (effort == "low") return ReasoningEffort::kLow;
  if (effort == "medium") return ReasoningEffort::kMedium;
  if (effort == "high" || effort == "xhigh" || effort == "max") return ReasoningEffort::kHigh;
  throw ApiError(400, "Invalid value for 'reasoning.effort': '" + effort + "'.", "reasoning.effort", "invalid_value");
}

ResponseRequest ParseResponseRequest(const Json& body) {
  if (!body.is_object()) throw ApiError(400, "Request body must be a JSON object.");
  // Implemented, or accepted and ignored.
  // clang-format off
  static const std::set<std::string> kAccepted = {
      "model", "input", "instructions", "max_output_tokens", "temperature", "top_p", "reasoning", "stream", "store",
      "metadata", "user", "service_tier", "safety_identifier", "prompt_cache_key", "stream_options",
      "parallel_tool_calls", "tools", "tool_choice"};
  // clang-format on
  for (const auto& [key, value] : body.items()) {
    if (kAccepted.contains(key) || IsEmptyValue(value) || IsDefaultValue(key, value)) continue;
    throw ApiError(400, "Unsupported parameter: '" + key + "'.", key, "unsupported_parameter");
  }

  ResponseRequest request;
  const Json* model = Field(body, "model");
  if (model == nullptr || !model->is_string()) throw ApiError(400, "Missing required parameter: 'model'.", "model");
  request.model = model->get<std::string>();

  const Json* instructions = Field(body, "instructions");
  if (instructions != nullptr) {
    if (!instructions->is_string()) throw ApiError(400, "'instructions' must be a string.", "instructions");
    request.instructions = *instructions;
    request.messages.push_back({"system", instructions->get<std::string>(), ""});
  }
  const Json* input = Field(body, "input");
  if (input == nullptr) throw ApiError(400, "Missing required parameter: 'input'.", "input");
  for (ChatMessage& m : ParseInput(*input)) request.messages.push_back(std::move(m));

  // Function tools, given to the template in the Chat Completions shape the model was trained on.
  if (const Json* tools = Field(body, "tools")) {
    if (!tools->is_array()) throw ApiError(400, "'tools' must be an array.", "tools");
    for (std::size_t i = 0; i < tools->size(); ++i) {
      const Json& tool = (*tools)[i];
      const std::string param = "tools[" + std::to_string(i) + "]";
      const Json* type = tool.is_object() ? Field(tool, "type") : nullptr;
      if (type == nullptr || *type != Json("function")) {
        throw ApiError(400, "Only function tools are supported.", param + ".type", "unsupported_value");
      }
      const Json* name = Field(tool, "name");
      if (name == nullptr || !name->is_string() || name->get<std::string>().empty()) {
        throw ApiError(400, "A function tool needs a 'name'.", param + ".name");
      }
      Json function = {{"name", *name}};
      for (const char* key : {"description", "parameters"}) {
        if (const Json* v = Field(tool, key)) function[key] = *v;
      }
      request.chat.tools.push_back({{"type", "function"}, {"function", std::move(function)}});
    }
    request.tools = *tools;
  }
  if (const Json* choice = Field(body, "tool_choice")) {
    if (*choice == Json("none")) {
      request.chat.tools = Json::array();  // the model is not told of the tools
      request.tool_choice = "none";
    } else if (*choice != Json("auto")) {
      throw ApiError(400, "Only 'auto' and 'none' are supported for 'tool_choice'.", "tool_choice",
                     "unsupported_value");
    }
  }

  request.effort = "high";
  if (const Json* reasoning = Field(body, "reasoning")) {
    if (!reasoning->is_object()) throw ApiError(400, "'reasoning' must be an object.", "reasoning");
    if (const Json* effort = Field(*reasoning, "effort")) {
      if (!effort->is_string()) throw ApiError(400, "'reasoning.effort' must be a string.", "reasoning.effort");
      request.effort = effort->get<std::string>();
    }
  }
  request.chat.effort = ParseEffort(request.effort);

  if (const Json* max = Field(body, "max_output_tokens")) {
    if (!max->is_number_integer() || max->get<std::int64_t>() < 1 ||
        max->get<std::int64_t>() > std::numeric_limits<std::uint32_t>::max()) {
      throw ApiError(400, "'max_output_tokens' must be a positive integer.", "max_output_tokens", "invalid_value");
    }
    request.max_output_tokens = max->get<std::uint32_t>();
  }
  request.temperature = Number(body, "temperature", request.temperature, 0.0, 2.0);
  request.top_p = Number(body, "top_p", request.top_p, 0.0, 1.0);
  if (const Json* stream = Field(body, "stream")) {
    if (!stream->is_boolean()) throw ApiError(400, "'stream' must be a boolean.", "stream");
    request.stream = stream->get<bool>();
  }
  request.metadata = Json::object();
  if (const Json* metadata = Field(body, "metadata")) {
    if (!metadata->is_object()) throw ApiError(400, "'metadata' must be an object.", "metadata");
    request.metadata = *metadata;
  }
  return request;
}

OutputParser::Chunk OutputParser::Feed(std::string_view bytes) {
  ++tokens_;
  utf8_pending_ += bytes;
  const std::size_t n = CompleteUtf8(utf8_pending_);
  std::string text = utf8_pending_.substr(0, n);
  utf8_pending_.erase(0, n);
  Chunk chunk;
  Process(std::move(text), &chunk);
  return chunk;
}

OutputParser::Chunk OutputParser::Finish() {
  Chunk chunk;
  // Bytes of a sequence the model never finished go out as they are; Dump() replaces them.
  Process(std::move(utf8_pending_), &chunk);
  utf8_pending_.clear();
  if (in_reasoning_) {
    reasoning_tokens_ = tokens_;
    chunk.reasoning += StripSpace(held_, !reasoning_started_);
    held_.clear();
  }
  return chunk;
}

void OutputParser::Process(std::string text, Chunk* chunk) {
  if (!in_reasoning_) {
    EmitAnswer(text, chunk);
    return;
  }
  held_ += text;
  const std::size_t pos = held_.find(kThinkEnd);
  if (pos != std::string::npos) {
    chunk->reasoning += StripSpace(std::string_view(held_).substr(0, pos), !reasoning_started_);
    const std::string rest = held_.substr(pos + kThinkEnd.size());
    held_.clear();
    in_reasoning_ = false;
    reasoning_tokens_ = tokens_;
    chunk->reasoning_ended = true;
    skip_answer_space_ = true;
    EmitAnswer(rest, chunk);
    return;
  }
  // Hold a tail that may be the start of </think>, and the whitespace before it.
  std::size_t keep = 0;
  for (std::size_t k = std::min(held_.size(), kThinkEnd.size() - 1); k > 0; --k) {
    if (std::string_view(held_).substr(held_.size() - k) == kThinkEnd.substr(0, k)) {
      keep = k;
      break;
    }
  }
  std::size_t end = held_.size() - keep;
  while (end > 0 && IsSpace(held_[end - 1])) --end;
  std::size_t begin = 0;
  if (!reasoning_started_) {
    while (begin < end && IsSpace(held_[begin])) ++begin;
  }
  if (begin < end) {
    chunk->reasoning.append(held_, begin, end - begin);
    reasoning_started_ = true;
    held_.erase(0, end);
  }
}

void OutputParser::EmitAnswer(std::string_view text, Chunk* chunk) {
  if (skip_answer_space_) {
    while (!text.empty() && IsSpace(text.front())) text.remove_prefix(1);
    if (text.empty()) return;
    skip_answer_space_ = false;
  }
  chunk->answer += text;
}

namespace {

constexpr std::string_view kCallOpen = "<tool_call>", kCallClose = "</tool_call>";

bool IsAsciiSpace(char c) {
  return c == ' ' || c == '\n' || c == '\t' || c == '\r';
}

}  // namespace

ToolCallParser::Chunk ToolCallParser::Feed(std::string_view text) {
  Chunk chunk;
  held_ += text;
  for (;;) {
    if (in_call_) {
      const std::size_t end = held_.find(kCallClose);
      if (end == std::string::npos) return chunk;
      const std::string block = held_.substr(0, end);
      ToolCall call;
      if (Parse(block, &call)) {
        chunk.calls.push_back(std::move(call));
        called_ = true;
      } else {
        chunk.content += std::string(kCallOpen) + block + std::string(kCallClose);
      }
      held_.erase(0, end + kCallClose.size());
      in_call_ = false;
      continue;
    }
    // Text goes out up to a call or a tail that may begin one, less the whitespace before either.
    std::size_t keep = held_.size();
    const std::size_t open = held_.find(kCallOpen);
    if (open != std::string::npos) {
      keep = open;
    } else {
      for (std::size_t n = std::min(kCallOpen.size() - 1, held_.size()); n > 0; --n) {
        if (held_.compare(held_.size() - n, n, kCallOpen, 0, n) == 0) {
          keep = held_.size() - n;
          break;
        }
      }
    }
    std::size_t cut = keep;
    while (cut > 0 && IsAsciiSpace(held_[cut - 1])) --cut;
    chunk.content += held_.substr(0, cut);
    if (open == std::string::npos) {
      held_.erase(0, cut);
      return chunk;
    }
    held_.erase(0, open + kCallOpen.size());
    in_call_ = true;
  }
}

ToolCallParser::Chunk ToolCallParser::Finish() {
  Chunk chunk;
  if (in_call_) {
    chunk.content = std::string(kCallOpen) + held_;
  } else if (!called_ || !TrimPy(held_).empty()) {
    chunk.content = held_;
  }
  held_.clear();
  in_call_ = false;
  return chunk;
}

bool ToolCallParser::Parse(std::string_view block, ToolCall* call) const {
  constexpr std::string_view kFunction = "<function=", kFunctionEnd = "</function>";
  constexpr std::string_view kParameter = "<parameter=", kParameterEnd = "</parameter>";
  std::size_t at = block.find(kFunction);
  if (at == std::string_view::npos) return false;
  at += kFunction.size();
  const std::size_t name_end = block.find('>', at);
  if (name_end == std::string_view::npos) return false;
  call->name = TrimPy(block.substr(at, name_end - at));
  if (call->name.empty()) return false;
  call->id = RandomId("call_");
  call->arguments = Json::object();
  const std::size_t body_end = std::min(block.find(kFunctionEnd, name_end), block.size());
  for (at = name_end + 1;;) {
    const std::size_t p = block.find(kParameter, at);
    if (p == std::string_view::npos || p >= body_end) break;
    const std::size_t key_at = p + kParameter.size();
    const std::size_t key_end = block.find('>', key_at);
    if (key_end == std::string_view::npos || key_end > body_end) return false;
    // A value ends at its </parameter>, or where the next parameter or the function's end shows it was not closed.
    const std::size_t value_at = key_end + 1;
    const std::size_t value_end =
        std::min({block.find(kParameterEnd, value_at), block.find(kParameter, value_at), body_end});
    std::string_view value = block.substr(value_at, value_end - value_at);
    if (value.starts_with('\n')) value.remove_prefix(1);
    if (value.ends_with('\n')) value.remove_suffix(1);
    const std::string key = TrimPy(block.substr(key_at, key_end - key_at));
    call->arguments[key] = Value(call->name, key, value);
    at = block.compare(value_end, kParameterEnd.size(), kParameterEnd) == 0 ? value_end + kParameterEnd.size()
                                                                            : value_end;
  }
  return true;
}

Json ToolCallParser::Value(const std::string& function, const std::string& key, std::string_view text) const {
  // Strings unless the schema gives the parameter a type that is not string (an undeclared one stays a string).
  bool string = true;
  for (const Json& tool : tools_) {
    const Json* f = Field(tool, "function");
    if (f == nullptr || f->value("name", std::string()) != function) continue;
    const Json* params = Field(*f, "parameters");
    const Json* props = params != nullptr ? Field(*params, "properties") : nullptr;
    const Json* schema = props != nullptr ? Field(*props, key.c_str()) : nullptr;
    if (schema == nullptr) break;
    const Json* type = Field(*schema, "type");
    if (type == nullptr) {
      string = false;
    } else if (type->is_string()) {
      string = type->get<std::string>() == "string";
    } else if (type->is_array()) {
      string = false;
      for (const Json& t : *type) string = string || t == Json("string");
    }
    break;
  }
  if (!string) {
    try {
      return Json::parse(text);
    } catch (const Json::parse_error&) {
    }
  }
  return Json(std::string(text));
}

ResponseStream::ResponseStream(const ResponseRequest& request, std::function<bool(const std::string&)> emit)
    : request_(request),
      emit_(std::move(emit)),
      parser_(request.chat.effort != ReasoningEffort::kNone),
      tool_parser_(request.chat.tools.empty() ? std::nullopt : std::optional<ToolCallParser>(request.chat.tools)),
      id_(RandomId("resp_")),
      reasoning_id_(RandomId("rs_")),
      message_id_(RandomId("msg_")),
      created_at_(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                      .count()) {}

bool ResponseStream::Emit(const std::string& type, Json event) {
  if (!emit_ || !alive_) return alive_;
  Json e = {{"type", type}, {"sequence_number", sequence_++}};
  for (auto& [key, value] : event.items()) e[key] = std::move(value);
  alive_ = emit_("event: " + type + "\ndata: " + Dump(e) + "\n\n");
  return alive_;
}

bool ResponseStream::Start() {
  Emit("response.created", {{"response", ResponseObject("in_progress")}});
  Emit("response.in_progress", {{"response", ResponseObject("in_progress")}});
  if (!parser_.in_reasoning() && !tool_parser_) OpenMessage();
  return alive_;
}

bool ResponseStream::OnToken(std::string_view bytes) {
  Apply(parser_.Feed(bytes));
  return alive_;
}

bool ResponseStream::Finish(const model::GenerateResult& result) {
  Apply(parser_.Finish());
  if (tool_parser_) ApplyTools(tool_parser_->Finish());
  if (reasoning_open_) CloseReasoning();
  status_ = result.finish_reason == "length" && calls_.empty() ? "incomplete" : "completed";
  if (tool_parser_ && message_index_ < 0 && calls_.empty() && status_ == "completed")
    OpenMessage();  // a reply without calls ends in a message, as without tools
  if (message_open_) CloseMessage();
  usage_ = {{"input_tokens", result.prompt_tokens},
            {"input_tokens_details", {{"cached_tokens", result.cached_tokens}, {"cache_write_tokens", 0}}},
            {"output_tokens", result.generated_tokens},
            {"output_tokens_details", {{"reasoning_tokens", parser_.reasoning_tokens()}}},
            {"total_tokens", result.prompt_tokens + result.generated_tokens}};
  Emit("response." + status_, {{"response", ResponseObject(status_)}});
  return alive_;
}

void ResponseStream::Fail(const std::string& message) {
  Emit("error", {{"code", "server_error"}, {"message", message}, {"param", nullptr}});
}

void ResponseStream::Apply(const OutputParser::Chunk& chunk) {
  if (!chunk.reasoning.empty()) {
    if (!reasoning_open_) OpenReasoning();
    reasoning_ += chunk.reasoning;
    Emit("response.reasoning_text.delta", {{"item_id", reasoning_id_},
                                           {"output_index", reasoning_index_},
                                           {"content_index", 0},
                                           {"delta", chunk.reasoning}});
  }
  if (chunk.reasoning_ended) {
    if (reasoning_open_) CloseReasoning();
    if (!tool_parser_) OpenMessage();
  }
  if (!tool_parser_) {
    AnswerText(chunk.answer);
  } else if (!chunk.answer.empty()) {
    ApplyTools(tool_parser_->Feed(chunk.answer));
  }
}

void ResponseStream::AnswerText(const std::string& text) {
  if (text.empty()) return;
  if (!message_open_) OpenMessage();
  answer_ += text;
  Emit("response.output_text.delta", {{"item_id", message_id_},
                                      {"output_index", message_index_},
                                      {"content_index", 0},
                                      {"delta", text},
                                      {"logprobs", Json::array()}});
}

void ResponseStream::ApplyTools(ToolCallParser::Chunk chunk) {
  // Text after a call (the template asks for none) has no message left to go to and is dropped.
  if (calls_.empty()) AnswerText(chunk.content);
  for (ToolCall& call : chunk.calls) {
    if (message_open_) CloseMessage();
    const int index = (reasoning_index_ >= 0) + (message_index_ >= 0) + static_cast<int>(calls_.size());
    calls_.push_back({RandomId("fc_"), std::move(call), index});
    const CallItem& item = calls_.back();
    const std::string arguments = Dump(item.call.arguments);
    Emit("response.output_item.added", {{"output_index", index}, {"item", FunctionCallItem(item, false)}});
    Emit("response.function_call_arguments.delta",
         {{"item_id", item.id}, {"output_index", index}, {"delta", arguments}});
    Emit("response.function_call_arguments.done",
         {{"item_id", item.id}, {"output_index", index}, {"arguments", arguments}});
    Emit("response.output_item.done", {{"output_index", index}, {"item", FunctionCallItem(item, true)}});
  }
}

Json ResponseStream::FunctionCallItem(const CallItem& item, bool done) {
  return {{"id", item.id},
          {"type", "function_call"},
          {"status", done ? "completed" : "in_progress"},
          {"arguments", done ? Dump(item.call.arguments) : ""},
          {"call_id", item.call.id},
          {"name", item.call.name}};
}

void ResponseStream::OpenReasoning() {
  reasoning_index_ = 0;
  reasoning_open_ = true;
  Json item = {{"id", reasoning_id_}, {"type", "reasoning"}, {"summary", Json::array()}, {"content", Json::array()}};
  Emit("response.output_item.added", {{"output_index", reasoning_index_}, {"item", item}});
  Emit("response.content_part.added", {{"item_id", reasoning_id_},
                                       {"output_index", reasoning_index_},
                                       {"content_index", 0},
                                       {"part", {{"type", "reasoning_text"}, {"text", ""}}}});
}

void ResponseStream::CloseReasoning() {
  reasoning_open_ = false;
  Emit("response.reasoning_text.done",
       {{"item_id", reasoning_id_}, {"output_index", reasoning_index_}, {"content_index", 0}, {"text", reasoning_}});
  Emit("response.content_part.done", {{"item_id", reasoning_id_},
                                      {"output_index", reasoning_index_},
                                      {"content_index", 0},
                                      {"part", {{"type", "reasoning_text"}, {"text", reasoning_}}}});
  Emit("response.output_item.done", {{"output_index", reasoning_index_}, {"item", ReasoningItem()}});
}

void ResponseStream::OpenMessage() {
  message_index_ = reasoning_index_ + 1;
  message_open_ = true;
  Json item = {{"id", message_id_},
               {"type", "message"},
               {"status", "in_progress"},
               {"role", "assistant"},
               {"content", Json::array()}};
  Emit("response.output_item.added", {{"output_index", message_index_}, {"item", item}});
  Emit("response.content_part.added",
       {{"item_id", message_id_}, {"output_index", message_index_}, {"content_index", 0}, {"part", OutputText("")}});
}

void ResponseStream::CloseMessage() {
  message_open_ = false;
  Emit("response.output_text.done", {{"item_id", message_id_},
                                     {"output_index", message_index_},
                                     {"content_index", 0},
                                     {"text", answer_},
                                     {"logprobs", Json::array()}});
  Emit("response.content_part.done", {{"item_id", message_id_},
                                      {"output_index", message_index_},
                                      {"content_index", 0},
                                      {"part", OutputText(answer_)}});
  Emit("response.output_item.done", {{"output_index", message_index_}, {"item", MessageItem()}});
}

Json ResponseStream::OutputText(const std::string& text) {
  return {{"type", "output_text"}, {"text", text}, {"annotations", Json::array()}, {"logprobs", Json::array()}};
}

Json ResponseStream::ReasoningItem() const {
  return {{"id", reasoning_id_},
          {"type", "reasoning"},
          {"summary", Json::array()},
          {"content", Json::array({{{"type", "reasoning_text"}, {"text", reasoning_}}})}};
}

Json ResponseStream::MessageItem() const {
  return {{"id", message_id_},
          {"type", "message"},
          {"status", message_open_              ? "in_progress"
                     : status_ == "in_progress" ? "completed"
                                                : status_},
          {"role", "assistant"},
          {"content", Json::array({OutputText(answer_)})}};
}

Json ResponseStream::ResponseObject(const std::string& status) const {
  Json output = Json::array();
  if (reasoning_index_ >= 0) output.push_back(ReasoningItem());
  if (message_index_ >= 0) output.push_back(MessageItem());
  for (const CallItem& item : calls_) output.push_back(FunctionCallItem(item, true));
  return {{"id", id_},
          {"object", "response"},
          {"created_at", created_at_},
          {"status", status},
          {"background", false},
          {"error", nullptr},
          {"incomplete_details", status == "incomplete" ? Json{{"reason", "max_output_tokens"}} : Json(nullptr)},
          {"instructions", request_.instructions},
          {"max_output_tokens", request_.max_output_tokens ? Json(*request_.max_output_tokens) : Json(nullptr)},
          {"metadata", request_.metadata},
          {"model", request_.model},
          {"output", output},
          {"parallel_tool_calls", false},
          {"previous_response_id", nullptr},
          {"reasoning", {{"effort", request_.effort}, {"summary", nullptr}}},
          {"store", false},
          {"temperature", request_.temperature},
          {"text", {{"format", {{"type", "text"}}}}},
          {"tool_choice", request_.tool_choice},
          {"tools", request_.tools},
          {"top_p", request_.top_p},
          {"truncation", "disabled"},
          {"usage", usage_},
          {"user", nullptr}};
}

}  // namespace yah::serve
