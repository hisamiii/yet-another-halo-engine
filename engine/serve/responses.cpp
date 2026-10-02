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

std::string RandomId(const char* prefix) {
  thread_local std::mt19937_64 rng(std::random_device{}());
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%016llx%016llx", static_cast<unsigned long long>(rng()),
                static_cast<unsigned long long>(rng()));
  return std::string(prefix) + buf;
}

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

// Length of the longest prefix of s that does not end inside a UTF-8 sequence.
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

// Fields we do not implement but that are harmless at these values.
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

std::vector<ChatMessage> ParseInput(const Json& input) {
  if (input.is_string()) return {ChatMessage{"user", input.get<std::string>(), ""}};
  if (!input.is_array() || input.empty()) {
    throw ApiError(400, "'input' must be a string or a non-empty array of messages.", "input");
  }
  std::vector<ChatMessage> messages;
  std::string reasoning;  // from reasoning items, for the next assistant message
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
    messages.push_back(std::move(m));
  }
  return messages;
}

}  // namespace

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
      "parallel_tool_calls"};
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

ResponseStream::ResponseStream(const ResponseRequest& request, std::function<bool(const std::string&)> emit)
    : request_(request),
      emit_(std::move(emit)),
      parser_(request.chat.effort != ReasoningEffort::kNone),
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
  if (!parser_.in_reasoning()) OpenMessage();
  return alive_;
}

bool ResponseStream::OnToken(std::string_view bytes) {
  Apply(parser_.Feed(bytes));
  return alive_;
}

bool ResponseStream::Finish(const model::GenerateResult& result) {
  Apply(parser_.Finish());
  if (reasoning_open_) CloseReasoning();
  status_ = result.finish_reason == "length" ? "incomplete" : "completed";
  if (message_open_) CloseMessage();
  usage_ = {{"input_tokens", result.prompt_tokens},
            {"input_tokens_details", {{"cached_tokens", 0}, {"cache_write_tokens", 0}}},
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
    OpenMessage();
  }
  if (!chunk.answer.empty()) {
    if (!message_open_) OpenMessage();
    answer_ += chunk.answer;
    Emit("response.output_text.delta", {{"item_id", message_id_},
                                        {"output_index", message_index_},
                                        {"content_index", 0},
                                        {"delta", chunk.answer},
                                        {"logprobs", Json::array()}});
  }
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
          {"status", message_open_ ? "in_progress" : status_},
          {"role", "assistant"},
          {"content", Json::array({OutputText(answer_)})}};
}

Json ResponseStream::ResponseObject(const std::string& status) const {
  Json output = Json::array();
  if (reasoning_index_ >= 0) output.push_back(ReasoningItem());
  if (message_index_ >= 0) output.push_back(MessageItem());
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
          {"tool_choice", "auto"},
          {"tools", Json::array()},
          {"top_p", request_.top_p},
          {"truncation", "disabled"},
          {"usage", usage_},
          {"user", nullptr}};
}

}  // namespace yah::serve
