#include "serve/chat_completions.hpp"

#include <chrono>
#include <limits>
#include <set>

namespace yah::serve {
namespace {

// The largest penalty window; penalty_last_n beyond it is refused.
constexpr std::int64_t kMaxPenaltyWindow = 1024;

// An assistant message's tool call. Arguments come as a JSON string (OpenAI) or an object.
ToolCall ParseToolCall(const Json& item, const std::string& param) {
  const Json* function = item.is_object() ? Field(item, "function") : nullptr;
  if (function == nullptr || !function->is_object()) throw ApiError(400, "A tool call needs a 'function'.", param);
  ToolCall call;
  const Json* name = Field(*function, "name");
  if (name == nullptr || !name->is_string() || name->get<std::string>().empty()) {
    throw ApiError(400, "Tool call is missing a function name.", param + ".function.name");
  }
  call.name = name->get<std::string>();
  if (const Json* id = Field(item, "id"); id != nullptr && id->is_string()) call.id = id->get<std::string>();
  call.arguments = Json::object();
  if (const Json* args = Field(*function, "arguments")) {
    Json parsed = *args;
    if (args->is_string()) {
      const std::string text = TrimPy(args->get<std::string>());
      try {
        parsed = text.empty() ? Json::object() : Json::parse(text);
      } catch (const Json::parse_error&) {
        parsed = nullptr;
      }
    }
    if (!parsed.is_object()) {
      throw ApiError(400, "Tool call arguments must be a JSON object.", param + ".function.arguments");
    }
    call.arguments = std::move(parsed);
  }
  return call;
}

std::vector<ChatMessage> ParseMessages(const Json& messages) {
  if (!messages.is_array() || messages.empty()) {
    throw ApiError(400, "'messages' must be a non-empty array.", "messages");
  }
  std::vector<ChatMessage> out;
  for (std::size_t i = 0; i < messages.size(); ++i) {
    const Json& item = messages[i];
    const std::string param = "messages[" + std::to_string(i) + "]";
    if (!item.is_object()) throw ApiError(400, "Each message must be an object.", param);
    const Json* role = Field(item, "role");
    if (role == nullptr || !role->is_string()) throw ApiError(400, "Message needs a 'role'.", param + ".role");
    ChatMessage m{role->get<std::string>(), "", ""};
    if (m.role != "user" && m.role != "assistant" && m.role != "system" && m.role != "developer" && m.role != "tool") {
      throw ApiError(400, "Unsupported role '" + m.role + "'.", param + ".role", "invalid_value");
    }
    if (const Json* calls = Field(item, "tool_calls"); calls != nullptr && !IsEmptyValue(*calls)) {
      if (m.role != "assistant" || !calls->is_array()) {
        throw ApiError(400, "'tool_calls' must be an array, on an assistant message.", param + ".tool_calls");
      }
      for (std::size_t c = 0; c < calls->size(); ++c) {
        m.tool_calls.push_back(ParseToolCall((*calls)[c], param + ".tool_calls[" + std::to_string(c) + "]"));
      }
    }
    m.content = ContentText(item.contains("content") ? item["content"] : Json(nullptr), param + ".content");
    if (m.role == "assistant") {
      for (const char* key : {"reasoning_content", "reasoning"}) {
        const Json* r = Field(item, key);
        if (r == nullptr) continue;
        if (!r->is_string()) throw ApiError(400, std::string("'") + key + "' must be a string.", param + "." + key);
        m.reasoning = r->get<std::string>();
        break;
      }
    }
    out.push_back(std::move(m));
  }
  return out;
}

std::int64_t Integer(const Json& body, const char* key, std::int64_t fallback, std::int64_t lo, std::int64_t hi) {
  const Json* v = Field(body, key);
  if (v == nullptr) return fallback;
  if (!v->is_number_integer()) throw ApiError(400, std::string("'") + key + "' must be an integer.", key);
  const std::int64_t x = v->get<std::int64_t>();
  if (x < lo || x > hi) {
    throw ApiError(400,
                   std::string("Invalid value for '") + key + "': must be between " + std::to_string(lo) + " and " +
                       std::to_string(hi) + ".",
                   key, "invalid_value");
  }
  return x;
}

std::string Sse(const Json& data) { return "data: " + Dump(data) + "\n\n"; }

}  // namespace

ChatRequest ParseChatRequest(const Json& body) {
  if (!body.is_object()) throw ApiError(400, "Request body must be a JSON object.");
  // Implemented, or accepted and ignored.
  // clang-format off
  static const std::set<std::string> kAccepted = {
      "model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k", "min_p",
      "presence_penalty", "frequency_penalty", "repetition_penalty", "repeat_last_n", "stop", "seed", "stream",
      "stream_options", "reasoning_effort", "chat_template_kwargs", "n", "user", "store", "metadata", "service_tier",
      "parallel_tool_calls", "prompt_cache_key", "safety_identifier", "ignore_eos", "tools", "tool_choice"};
  // clang-format on
  for (const auto& [key, value] : body.items()) {
    if (kAccepted.contains(key) || IsEmptyValue(value) || IsDefaultValue(key, value)) continue;
    if (key == "logprobs" && value == false) continue;
    if (key == "response_format" && value == Json{{"type", "text"}}) continue;
    throw ApiError(400, "Unsupported parameter: '" + key + "'.", key, "unsupported_parameter");
  }

  ChatRequest request;
  const Json* model = Field(body, "model");
  if (model == nullptr || !model->is_string()) throw ApiError(400, "Missing required parameter: 'model'.", "model");
  request.model = model->get<std::string>();
  const Json* messages = Field(body, "messages");
  if (messages == nullptr) throw ApiError(400, "Missing required parameter: 'messages'.", "messages");
  request.messages = ParseMessages(*messages);

  if (const Json* tools = Field(body, "tools")) {
    if (!tools->is_array()) throw ApiError(400, "'tools' must be an array.", "tools");
    for (std::size_t i = 0; i < tools->size(); ++i) {
      const Json& tool = (*tools)[i];
      const std::string param = "tools[" + std::to_string(i) + "]";
      const Json* type = tool.is_object() ? Field(tool, "type") : nullptr;
      if (type == nullptr || *type != Json("function")) {
        throw ApiError(400, "Only function tools are supported.", param + ".type", "unsupported_value");
      }
      const Json* function = Field(tool, "function");
      const Json* name = function != nullptr && function->is_object() ? Field(*function, "name") : nullptr;
      if (name == nullptr || !name->is_string() || name->get<std::string>().empty()) {
        throw ApiError(400, "A tool needs a 'function' with a 'name'.", param + ".function.name");
      }
    }
    request.chat.tools = *tools;
  }
  if (const Json* choice = Field(body, "tool_choice")) {
    // none: the model is not told of the tools.
    if (*choice == Json("none")) {
      request.chat.tools = Json::array();
    } else if (*choice != Json("auto")) {
      throw ApiError(400, "Only 'auto' and 'none' are supported for 'tool_choice'.", "tool_choice",
                     "unsupported_value");
    }
  }

  if (const Json* effort = Field(body, "reasoning_effort")) {
    if (!effort->is_string()) throw ApiError(400, "'reasoning_effort' must be a string.", "reasoning_effort");
    try {
      request.chat.effort = ParseEffort(effort->get<std::string>());
    } catch (const ApiError& error) {
      throw ApiError(400, "Invalid value for 'reasoning_effort': '" + effort->get<std::string>() + "'.",
                     "reasoning_effort", "invalid_value");
    }
  }
  if (const Json* kwargs = Field(body, "chat_template_kwargs")) {
    if (!kwargs->is_object()) throw ApiError(400, "'chat_template_kwargs' must be an object.", "chat_template_kwargs");
    for (const auto& [key, value] : kwargs->items()) {
      if (key != "enable_thinking" || !value.is_boolean()) {
        throw ApiError(400, "Unsupported chat_template_kwargs entry '" + key + "'; only enable_thinking (bool).",
                       "chat_template_kwargs", "unsupported_parameter");
      }
      if (!value.get<bool>()) request.chat.effort = ReasoningEffort::kNone;
    }
  }

  constexpr std::int64_t kU32 = std::numeric_limits<std::uint32_t>::max();
  for (const char* key : {"max_completion_tokens", "max_tokens"}) {
    if (Field(body, key) != nullptr && !request.max_tokens) {
      request.max_tokens = static_cast<std::uint32_t>(Integer(body, key, 0, 1, kU32));
    }
  }
  if (Integer(body, "n", 1, 1, 1) != 1) throw ApiError(400, "'n' must be 1.", "n", "invalid_value");

  model::SamplingParams& s = request.sampling;
  s.temperature = static_cast<float>(Number(body, "temperature", 0.6, 0.0, 2.0));
  s.top_p = static_cast<float>(Number(body, "top_p", 0.95, 0.0, 1.0));
  const std::int64_t top_k = Integer(body, "top_k", 20, -1, kU32);   // vLLM's -1 and llama.cpp's 0 both mean off
  s.top_k = top_k < 0 ? 0 : static_cast<std::uint32_t>(top_k);
  s.min_p = static_cast<float>(Number(body, "min_p", 0.0, 0.0, 1.0));
  s.presence_penalty = static_cast<float>(Number(body, "presence_penalty", 0.0, -2.0, 2.0));
  s.frequency_penalty = static_cast<float>(Number(body, "frequency_penalty", 0.0, -2.0, 2.0));
  s.repetition_penalty = static_cast<float>(Number(body, "repetition_penalty", 1.0, 0.01, 10.0));
  s.penalty_last_n = static_cast<std::uint32_t>(Integer(body, "repeat_last_n", 64, 0, kMaxPenaltyWindow));
  if (const Json* seed = Field(body, "seed")) {
    if (!seed->is_number_integer()) throw ApiError(400, "'seed' must be an integer.", "seed");
    // Any request seed is a fixed draw; 0 inside the engine means "from the clock".
    const std::uint64_t mixed = static_cast<std::uint64_t>(seed->get<std::int64_t>()) ^ 0x9E3779B97F4A7C15ULL;
    s.seed = mixed ? mixed : 1;
  }

  if (const Json* stop = Field(body, "stop")) {
    if (stop->is_string()) {
      request.stop.push_back(stop->get<std::string>());
    } else if (stop->is_array() && stop->size() <= 16) {
      for (const Json& v : *stop) {
        if (!v.is_string()) throw ApiError(400, "'stop' entries must be strings.", "stop");
        request.stop.push_back(v.get<std::string>());
      }
    } else {
      throw ApiError(400, "'stop' must be a string or an array of at most 16 strings.", "stop");
    }
    std::erase_if(request.stop, [](const std::string& x) { return x.empty(); });
  }
  if (const Json* stream = Field(body, "stream")) {
    if (!stream->is_boolean()) throw ApiError(400, "'stream' must be a boolean.", "stream");
    request.stream = stream->get<bool>();
  }
  if (const Json* ignore = Field(body, "ignore_eos")) {
    if (!ignore->is_boolean()) throw ApiError(400, "'ignore_eos' must be a boolean.", "ignore_eos");
    request.ignore_eos = ignore->get<bool>();
  }
  if (const Json* options = Field(body, "stream_options")) {
    if (!options->is_object()) throw ApiError(400, "'stream_options' must be an object.", "stream_options");
    if (const Json* usage = Field(*options, "include_usage")) {
      if (!usage->is_boolean())
        throw ApiError(400, "'stream_options.include_usage' must be a boolean.", "stream_options.include_usage");
      request.include_usage = usage->get<bool>();
    }
  }
  return request;
}

std::string StopMatcher::Feed(std::string_view bytes) {
  if (stopped_) return {};
  held_ += bytes;
  std::size_t first = std::string::npos;
  for (const std::string& s : stops_) first = std::min(first, held_.find(s));
  if (first != std::string::npos) {
    stopped_ = true;
    std::string out = held_.substr(0, first);
    held_.clear();
    return out;
  }
  // Hold the longest tail that is a proper prefix of a stop string.
  std::size_t keep = 0;
  for (const std::string& s : stops_) {
    for (std::size_t n = std::min(s.size() - 1, held_.size()); n > keep; --n) {
      if (held_.compare(held_.size() - n, n, s, 0, n) == 0) {
        keep = n;
        break;
      }
    }
  }
  std::string out = held_.substr(0, held_.size() - keep);
  held_.erase(0, held_.size() - keep);
  return out;
}

std::string StopMatcher::Finish() {
  std::string out = std::move(held_);
  held_.clear();
  return out;
}

ChatStream::ChatStream(const ChatRequest& request, std::function<bool(const std::string&)> emit)
    : request_(request),
      emit_(std::move(emit)),
      stop_(request.stop),
      parser_(request.chat.effort != ReasoningEffort::kNone),
      tool_parser_(request.chat.tools.empty() ? std::nullopt : std::optional<ToolCallParser>(request.chat.tools)),
      id_(RandomId("chatcmpl-")),
      created_(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                   .count()) {}

bool ChatStream::EmitChunk(Json delta, const Json& finish_reason, const Json* usage) {
  if (!emit_ || !alive_) return alive_;
  Json chunk = {{"id", id_}, {"object", "chat.completion.chunk"}, {"created", created_}, {"model", request_.model}};
  if (usage != nullptr) {
    chunk["choices"] = Json::array();
    chunk["usage"] = *usage;
  } else {
    chunk["choices"] = Json::array(
        {{{"index", 0}, {"delta", std::move(delta)}, {"logprobs", nullptr}, {"finish_reason", finish_reason}}});
    if (request_.include_usage) chunk["usage"] = nullptr;
    if (!finish_reason.is_null()) chunk["timings"] = timings_;
  }
  alive_ = emit_(Sse(chunk));
  return alive_;
}

bool ChatStream::Start() {
  EmitChunk({{"role", "assistant"}, {"content", ""}}, nullptr);
  return alive_;
}

bool ChatStream::OnToken(std::string_view bytes) {
  Apply(parser_.Feed(stop_.Feed(bytes)));
  return alive_ && !stop_.stopped();
}

void ChatStream::Apply(const OutputParser::Chunk& chunk) {
  if (!chunk.reasoning.empty()) {
    reasoning_ += chunk.reasoning;
    EmitChunk({{"reasoning_content", chunk.reasoning}}, nullptr);
  }
  if (!tool_parser_) {
    Content(chunk.answer);
  } else if (!chunk.answer.empty()) {
    ApplyTools(tool_parser_->Feed(chunk.answer));
  }
}

void ChatStream::Content(const std::string& text) {
  if (text.empty()) return;
  answer_ += text;
  EmitChunk({{"content", text}}, nullptr);
}

void ChatStream::ApplyTools(ToolCallParser::Chunk chunk) {
  Content(chunk.content);
  for (ToolCall& call : chunk.calls) {
    Json delta = {{"index", calls_.size()}};
    Json object = CallObject(call);
    for (auto& [key, value] : object.items()) delta[key] = std::move(value);
    EmitChunk({{"tool_calls", Json::array({std::move(delta)})}}, nullptr);
    calls_.push_back(std::move(call));
  }
}

Json ChatStream::CallObject(const ToolCall& call) {
  return {
      {"id", call.id}, {"type", "function"}, {"function", {{"name", call.name}, {"arguments", Dump(call.arguments)}}}};
}

bool ChatStream::Finish(const model::GenerateResult& result) {
  if (const std::string tail = stop_.Finish(); !tail.empty()) Apply(parser_.Feed(tail));
  Apply(parser_.Finish());
  if (tool_parser_) ApplyTools(tool_parser_->Finish());
  finish_reason_ = !calls_.empty()                                        ? "tool_calls"
                   : result.finish_reason == "length" && !stop_.stopped() ? "length"
                                                                          : "stop";
  usage_ = {{"prompt_tokens", result.prompt_tokens},
            {"completion_tokens", result.generated_tokens},
            {"total_tokens", result.prompt_tokens + result.generated_tokens},
            {"prompt_tokens_details", {{"cached_tokens", result.cached_tokens}}},
            {"completion_tokens_details", {{"reasoning_tokens", parser_.reasoning_tokens()}}}};
  timings_ = Timings(result);
  EmitChunk(Json::object(), finish_reason_);
  if (request_.include_usage) EmitChunk(nullptr, nullptr, &usage_);
  if (emit_ && alive_) alive_ = emit_("data: [DONE]\n\n");
  return alive_;
}

void ChatStream::Fail(const std::string& message) {
  if (!emit_ || !alive_) return;
  alive_ = emit_(Sse(ErrorBody(ApiError(500, message, "", "server_error", "server_error")))) &&
           emit_("data: [DONE]\n\n");
}

Json ChatStream::Response() const {
  Json message = {{"role", "assistant"},
                  {"content", answer_.empty() && !calls_.empty() ? Json(nullptr) : Json(answer_)},
                  {"refusal", nullptr}};
  if (!reasoning_.empty()) message["reasoning_content"] = reasoning_;
  if (!calls_.empty()) {
    message["tool_calls"] = Json::array();
    for (const ToolCall& call : calls_) message["tool_calls"].push_back(CallObject(call));
  }
  return {{"id", id_},
          {"object", "chat.completion"},
          {"created", created_},
          {"model", request_.model},
          {"choices",
           Json::array({{{"index", 0}, {"message", message}, {"logprobs", nullptr}, {"finish_reason", finish_reason_}}})},
          {"usage", usage_},
          {"timings", timings_}};
}

Json Timings(const model::GenerateResult& result) {
  const std::uint32_t prompt_n = result.prompt_tokens - result.cached_tokens;
  const std::uint32_t steps = result.generated_tokens > 1 ? result.generated_tokens - 1 : 0;
  return {{"cache_n", result.cached_tokens},
          {"prompt_n", prompt_n},
          {"prompt_ms", result.prefill_ms},
          {"prompt_per_second", result.prefill_ms > 0.0 ? prompt_n * 1000.0 / result.prefill_ms : 0.0},
          {"predicted_n", result.generated_tokens},
          {"predicted_ms", result.decode_ms},
          {"predicted_per_second", result.decode_ms > 0.0 ? steps * 1000.0 / result.decode_ms : 0.0}};
}

}  // namespace yah::serve
