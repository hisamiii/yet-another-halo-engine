// yah_serve_test: tests engine/serve without a GPU.
// 1. The C++ chat template against renders frozen from the GGUF's Jinja template (testdata/chat_template_cases.json).
// 2. yah_server --fake over HTTP: the Responses API object and stream shapes, errors, and client disconnect.
//
// usage: yah_serve_test <model.gguf> [yah_server]   (default: yah_server next to this binary)
#include <netinet/in.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "httplib/httplib.h"
#include "serve/chat_template.hpp"
#include "serve/fake_generator.hpp"
#include "serve/responses.hpp"

namespace {

using namespace yah::serve;

const std::string kReasoning = FakeGenerator::kReasoning;
const std::string kAnswer = FakeGenerator::kAnswer;

struct Failure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

int g_checks = 0;

void Check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) throw Failure(what);
}

void CheckEq(const Json& got, const Json& want, const std::string& what) {
  Check(got == want, what + ": got " + Dump(got) + ", want " + Dump(want));
}

// Chat template golden

int CheckChatTemplate(const std::filesystem::path& fixture, int* total) {
  std::ifstream in(fixture);
  if (!in) throw Failure("cannot open " + fixture.string());
  const Json cases = Json::parse(in);
  *total = static_cast<int>(cases.size());
  int passed = 0;
  for (const Json& c : cases) {
    std::string got, error;
    try {
      std::vector<ChatMessage> messages;
      for (const Json& m : c.at("messages")) {
        messages.push_back({m.at("role").get<std::string>(),
                            ContentText(m.contains("content") ? m["content"] : Json(nullptr), "content"),
                            m.value("reasoning_content", std::string())});
      }
      ChatOptions options;
      options.effort = ParseEffort(c.at("effort").get<std::string>());
      options.add_generation_prompt = c.at("add_generation_prompt").get<bool>();
      got = RenderChat(messages, options);
    } catch (const std::exception& e) {
      error = e.what();
    }
    const std::string name = c.at("name").get<std::string>();
    if (c.contains("error")) {
      if (c["error"] == error) {
        ++passed;
      } else {
        std::fprintf(stderr, "FAIL chat template '%s': want error %s, got error '%s' render %s\n", name.c_str(),
                     Dump(c["error"]).c_str(), error.c_str(), Dump(got).c_str());
      }
      continue;
    }
    const std::string want = c.at("expected").get<std::string>();
    if (error.empty() && got == want) {
      ++passed;
      continue;
    }
    std::size_t at = 0;
    while (at < got.size() && at < want.size() && got[at] == want[at]) ++at;
    std::fprintf(stderr, "FAIL chat template '%s': %s; first difference at byte %zu\n  jinja: %s\n  c++:   %s\n",
                 name.c_str(), error.empty() ? "render differs" : ("error '" + error + "'").c_str(), at,
                 Dump(want).c_str(), Dump(got).c_str());
  }
  return passed;
}

// The yah_server child process

int FreePort() {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socklen_t len = sizeof(addr);
  if (fd < 0 || bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
      getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
    throw Failure("cannot find a free port");
  }
  close(fd);
  return ntohs(addr.sin_port);
}

// Runs yah_server --fake with stderr in a memory file. The destructor kills it.
class ServerProcess {
 public:
  ServerProcess(const std::string& exe, const std::string& model, int port) {
    log_fd_ = memfd_create("yah_server_log", 0);
    if (log_fd_ < 0) throw Failure("memfd_create failed");
    const std::string port_str = std::to_string(port);
    pid_ = fork();
    if (pid_ < 0) throw Failure("fork failed");
    if (pid_ == 0) {
      // The server dies with the test, also when the test is killed.
      prctl(PR_SET_PDEATHSIG, SIGKILL);
      dup2(log_fd_, STDERR_FILENO);
      execl(exe.c_str(), exe.c_str(), "--model", model.c_str(), "--fake", "--port", port_str.c_str(),
            static_cast<char*>(nullptr));
      std::fprintf(stderr, "cannot run %s\n", exe.c_str());
      _exit(127);
    }
  }
  ~ServerProcess() {
    Stop();
    close(log_fd_);
  }
  void Stop() {
    if (pid_ <= 0) return;
    kill(pid_, SIGTERM);
    waitpid(pid_, nullptr, 0);
    pid_ = -1;
  }
  bool Exited() {
    if (pid_ > 0 && waitpid(pid_, nullptr, WNOHANG) == pid_) pid_ = -1;
    return pid_ <= 0;
  }
  [[nodiscard]] std::string Log() const {
    std::string text;
    char buf[4096];
    ssize_t n;
    for (off_t at = 0; (n = pread(log_fd_, buf, sizeof(buf), at)) > 0; at += n) text.append(buf, n);
    return text;
  }

 private:
  pid_t pid_ = -1;
  int log_fd_ = -1;
};

// Response object and stream event shapes: the required fields and types of the openai SDK models
// (Response, ResponseOutputMessage, ResponseReasoningItem, ResponseUsage and the stream events).

enum class Kind { kString, kInteger, kNumber, kBoolean, kArray, kObject };

bool Is(const Json& v, Kind kind) {
  switch (kind) {
    case Kind::kString:
      return v.is_string();
    case Kind::kInteger:
      return v.is_number_integer();
    case Kind::kNumber:
      return v.is_number();
    case Kind::kBoolean:
      return v.is_boolean();
    case Kind::kArray:
      return v.is_array();
    case Kind::kObject:
      return v.is_object();
  }
  return false;
}

// A field the SDK model requires: present, not null, of this kind.
void Required(const Json& obj, const std::string& where, const char* key, Kind kind) {
  Check(obj.is_object() && obj.contains(key) && Is(obj[key], kind),
        where + "." + key + " is missing or has a wrong type");
}

// A field the SDK model allows to be missing or null.
void Optional(const Json& obj, const std::string& where, const char* key, Kind kind) {
  if (!obj.contains(key) || obj[key].is_null()) return;
  Check(Is(obj[key], kind), where + "." + key + " has a wrong type: " + Dump(obj[key]));
}

void OneOf(const Json& obj, const std::string& where, const char* key, const std::vector<std::string>& values) {
  if (!obj.contains(key) || obj[key].is_null()) return;
  bool found = false;
  for (const std::string& v : values) found = found || obj[key] == v;
  Check(found, where + "." + key + " has a value outside the SDK's literals: " + Dump(obj[key]));
}

void CheckItemShape(const Json& item, const std::string& where) {
  Required(item, where, "type", Kind::kString);
  Required(item, where, "id", Kind::kString);
  if (item["type"] == "message") {
    Required(item, where, "content", Kind::kArray);
    Required(item, where, "role", Kind::kString);
    Required(item, where, "status", Kind::kString);
    CheckEq(item["role"], "assistant", where + ".role");
    OneOf(item, where, "status", {"in_progress", "completed", "incomplete"});
    for (std::size_t i = 0; i < item["content"].size(); ++i) {
      const Json& part = item["content"][i];
      const std::string at = where + ".content[" + std::to_string(i) + "]";
      Required(part, at, "type", Kind::kString);
      Required(part, at, "text", Kind::kString);
      Required(part, at, "annotations", Kind::kArray);
      Optional(part, at, "logprobs", Kind::kArray);
      CheckEq(part["type"], "output_text", at + ".type");
    }
  } else if (item["type"] == "reasoning") {
    Required(item, where, "summary", Kind::kArray);
    Optional(item, where, "content", Kind::kArray);
    for (std::size_t i = 0; i < item.value("content", Json::array()).size(); ++i) {
      const Json& part = item["content"][i];
      const std::string at = where + ".content[" + std::to_string(i) + "]";
      Required(part, at, "type", Kind::kString);
      Required(part, at, "text", Kind::kString);
      CheckEq(part["type"], "reasoning_text", at + ".type");
    }
  } else {
    Check(false, where + ".type is unexpected: " + Dump(item["type"]));
  }
}

void CheckResponseShape(const Json& r, const std::string& where) {
  Check(r.is_object(), where + " is not an object");
  Required(r, where, "id", Kind::kString);
  Required(r, where, "created_at", Kind::kNumber);
  Required(r, where, "model", Kind::kString);
  Required(r, where, "object", Kind::kString);
  Required(r, where, "output", Kind::kArray);
  Required(r, where, "parallel_tool_calls", Kind::kBoolean);
  Required(r, where, "tools", Kind::kArray);
  Check(r.contains("tool_choice") && (r["tool_choice"].is_object() || r["tool_choice"] == "none" ||
                                      r["tool_choice"] == "auto" || r["tool_choice"] == "required"),
        where + ".tool_choice is missing or invalid");
  CheckEq(r["object"], "response", where + ".object");
  Optional(r, where, "error", Kind::kObject);
  Optional(r, where, "incomplete_details", Kind::kObject);
  Check(!r.contains("instructions") || r["instructions"].is_null() || r["instructions"].is_string() ||
            r["instructions"].is_array(),
        where + ".instructions has a wrong type");
  Optional(r, where, "metadata", Kind::kObject);
  const Json metadata = r.value("metadata", Json::object());
  for (const auto& [key, value] : metadata.items()) {
    Check(value.is_string(), where + ".metadata." + key + " is not a string");
  }
  Optional(r, where, "temperature", Kind::kNumber);
  Optional(r, where, "top_p", Kind::kNumber);
  Optional(r, where, "background", Kind::kBoolean);
  Optional(r, where, "max_output_tokens", Kind::kInteger);
  Optional(r, where, "previous_response_id", Kind::kString);
  Optional(r, where, "reasoning", Kind::kObject);
  Optional(r, where, "status", Kind::kString);
  Optional(r, where, "text", Kind::kObject);
  Optional(r, where, "truncation", Kind::kString);
  Optional(r, where, "usage", Kind::kObject);
  Optional(r, where, "user", Kind::kString);
  OneOf(r, where, "status", {"completed", "failed", "in_progress", "cancelled", "queued", "incomplete"});
  OneOf(r, where, "truncation", {"auto", "disabled"});
  if (r.contains("incomplete_details") && r["incomplete_details"].is_object()) {
    OneOf(r["incomplete_details"], where + ".incomplete_details", "reason",
          {"max_output_tokens", "max_messages", "content_filter", "steered"});
  }
  if (r.contains("reasoning") && r["reasoning"].is_object()) {
    OneOf(r["reasoning"], where + ".reasoning", "effort", {"none", "minimal", "low", "medium", "high", "xhigh", "max"});
    OneOf(r["reasoning"], where + ".reasoning", "summary", {"auto", "concise", "detailed"});
  }
  for (std::size_t i = 0; i < r["output"].size(); ++i) {
    CheckItemShape(r["output"][i], where + ".output[" + std::to_string(i) + "]");
  }
  if (r.contains("usage") && r["usage"].is_object()) {
    Json u = r["usage"];
    const std::string at = where + ".usage";
    Required(u, at, "input_tokens", Kind::kInteger);
    Required(u, at, "input_tokens_details", Kind::kObject);
    Required(u, at, "output_tokens", Kind::kInteger);
    Required(u, at, "output_tokens_details", Kind::kObject);
    Required(u, at, "total_tokens", Kind::kInteger);
    Required(u["input_tokens_details"], at + ".input_tokens_details", "cached_tokens", Kind::kInteger);
    Required(u["input_tokens_details"], at + ".input_tokens_details", "cache_write_tokens", Kind::kInteger);
    Required(u["output_tokens_details"], at + ".output_tokens_details", "reasoning_tokens", Kind::kInteger);
  }
}

void CheckEventShape(const Json& e, const std::string& where) {
  Required(e, where, "type", Kind::kString);
  Required(e, where, "sequence_number", Kind::kInteger);
  const std::string type = e["type"].get<std::string>();
  const std::string at = where + " (" + type + ")";
  if (type == "response.created" || type == "response.in_progress" || type == "response.completed" ||
      type == "response.incomplete") {
    Required(e, at, "response", Kind::kObject);
    CheckResponseShape(e["response"], at + ".response");
    return;
  }
  Required(e, at, "output_index", Kind::kInteger);
  if (type == "response.output_item.added" || type == "response.output_item.done") {
    Required(e, at, "item", Kind::kObject);
    CheckItemShape(e["item"], at + ".item");
    return;
  }
  Required(e, at, "item_id", Kind::kString);
  Required(e, at, "content_index", Kind::kInteger);
  if (type == "response.content_part.added" || type == "response.content_part.done") {
    Required(e, at, "part", Kind::kObject);
    Required(e["part"], at + ".part", "type", Kind::kString);
    Required(e["part"], at + ".part", "text", Kind::kString);
    OneOf(e["part"], at + ".part", "type", {"output_text", "reasoning_text"});
    if (e["part"]["type"] == "output_text") Required(e["part"], at + ".part", "annotations", Kind::kArray);
  } else if (type == "response.output_text.delta" || type == "response.reasoning_text.delta") {
    Required(e, at, "delta", Kind::kString);
    if (type == "response.output_text.delta") Required(e, at, "logprobs", Kind::kArray);
  } else if (type == "response.output_text.done" || type == "response.reasoning_text.done") {
    Required(e, at, "text", Kind::kString);
    if (type == "response.output_text.done") Required(e, at, "logprobs", Kind::kArray);
  } else {
    Check(false, where + ": unexpected event type " + type);
  }
}

// Responses API checks

bool ValidUtf8(std::string_view s) {
  for (std::size_t i = 0; i < s.size();) {
    const auto b = static_cast<unsigned char>(s[i]);
    const std::size_t len = b < 0x80 ? 1 : (b >> 5) == 0x6 ? 2 : (b >> 4) == 0xE ? 3 : (b >> 3) == 0x1E ? 4 : 0;
    if (len == 0 || i + len > s.size()) return false;
    for (std::size_t k = 1; k < len; ++k) {
      if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
    }
    i += len;
  }
  return true;
}

// The SDK's Response.output_text: the text of every output_text part of every message item.
std::string OutputText(const Json& r) {
  std::string text;
  for (const Json& item : r["output"]) {
    if (item["type"] != "message") continue;
    for (const Json& part : item["content"]) {
      if (part["type"] == "output_text") text += part["text"].get<std::string>();
    }
  }
  return text;
}

Json OutputTypes(const Json& r) {
  Json types = Json::array();
  for (const Json& item : r["output"]) types.push_back(item["type"]);
  return types;
}

class Tester {
 public:
  explicit Tester(int port) : client_("127.0.0.1", port) { client_.set_read_timeout(60); }

  void Run(ServerProcess& server) {
    Health();
    Models();
    CreateStringInput();
    CreateMessageList();
    CreateMultiTurn();
    Streaming();
    MaxOutputTokens();
    Errors();
    Disconnect(server);
  }

 private:
  struct Reply {
    int status = 0;
    std::string content_type;
    std::string body;
  };

  Reply Request(const std::string& method, const std::string& path, const std::string& body = "") {
    httplib::Result res = method == "GET" ? client_.Get(path) : client_.Post(path, body, "application/json");
    Check(static_cast<bool>(res), method + " " + path + ": " + httplib::to_string(res.error()));
    return {res->status, res->get_header_value("Content-Type"), res->body};
  }

  // POST /v1/responses that must succeed: checks the raw JSON against the SDK's Response model.
  Json Create(Json body, const std::string& where) {
    const Reply reply = Request("POST", "/v1/responses", Dump(body));
    Check(reply.status == 200, where + ": status " + std::to_string(reply.status) + ": " + reply.body);
    Json r = Json::parse(reply.body);
    CheckResponseShape(r, where);
    return r;
  }

  // POST /v1/responses with stream true. Checks the SSE framing, every event's shape, sequence numbers from 0,
  // and that no delta splits a UTF-8 sequence.
  std::vector<Json> Stream(Json body, const std::string& where) {
    body["stream"] = true;
    const Reply reply = Request("POST", "/v1/responses", Dump(body));
    Check(reply.status == 200, where + ": status " + std::to_string(reply.status) + ": " + reply.body);
    Check(reply.content_type.starts_with("text/event-stream"), where + ": content type " + reply.content_type);
    std::vector<Json> events;
    std::size_t pos = 0;
    while (pos < reply.body.size()) {
      const std::size_t end = reply.body.find("\n\n", pos);
      Check(end != std::string::npos, where + ": unterminated SSE event at byte " + std::to_string(pos));
      const std::string block = reply.body.substr(pos, end - pos);
      pos = end + 2;
      const std::size_t nl = block.find('\n');
      Check(block.starts_with("event: ") && nl != std::string::npos && block.compare(nl + 1, 6, "data: ") == 0,
            where + ": bad SSE block: " + block);
      const std::string type = block.substr(7, nl - 7);
      const Json e = Json::parse(block.substr(nl + 7));
      const std::string at = where + " event " + std::to_string(events.size());
      CheckEventShape(e, at);
      CheckEq(e["type"], type, at + ": data.type against the SSE event name");
      CheckEq(e["sequence_number"], events.size(), at + ".sequence_number");
      if (e.contains("delta")) {
        const std::string delta = e["delta"].get<std::string>();
        Check(ValidUtf8(delta) && delta.find("\xEF\xBF\xBD") == std::string::npos,
              at + ": delta splits a UTF-8 sequence: " + Dump(e["delta"]));
      }
      events.push_back(e);
    }
    Check(!events.empty(), where + ": no events");
    return events;
  }

  static void CheckUsage(Json r, const std::string& where) {
    Json u = r["usage"];
    Check(u.is_object(), where + ": usage is null");
    CheckEq(u["total_tokens"], u["input_tokens"].get<int>() + u["output_tokens"].get<int>(), where + ": total_tokens");
    Check(u["input_tokens"].get<int>() > 0, where + ": input_tokens is 0");
    CheckEq(u["input_tokens_details"]["cached_tokens"], 0, where + ": cached_tokens");
    const int reasoning = u["output_tokens_details"]["reasoning_tokens"].get<int>();
    Check(reasoning >= 0 && reasoning <= u["output_tokens"].get<int>(), where + ": reasoning_tokens " + Dump(u));
  }

  // A complete reply of the fake generator.
  static void CheckFull(Json r, const std::string& effort, const std::string& where) {
    CheckEq(r["status"], "completed", where + ": status");
    CheckEq(r["error"], nullptr, where + ": error");
    CheckEq(r["incomplete_details"], nullptr, where + ": incomplete_details");
    CheckEq(OutputText(r), kAnswer, where + ": output_text");
    CheckUsage(r, where);
    Json u = r["usage"];
    const int reasoning_tokens = u["output_tokens_details"]["reasoning_tokens"].get<int>();
    if (effort == "none" || effort == "minimal") {
      CheckEq(OutputTypes(r), {"message"}, where + ": output types");
      CheckEq(reasoning_tokens, 0, where + ": reasoning_tokens");
    } else {
      CheckEq(OutputTypes(r), {"reasoning", "message"}, where + ": output types");
      Json item = r["output"][0];
      CheckEq(item["content"], Json::array({{{"type", "reasoning_text"}, {"text", kReasoning}}}),
              where + ": reasoning content");
      CheckEq(item["summary"], Json::array(), where + ": reasoning summary");
      Check(item["id"].get<std::string>().starts_with("rs_"), where + ": reasoning id " + Dump(item["id"]));
      Check(reasoning_tokens > 0 && reasoning_tokens < u["output_tokens"].get<int>(),
            where + ": reasoning_tokens " + Dump(u));
    }
    Json msg = r["output"].back();
    Check(msg["id"].get<std::string>().starts_with("msg_"), where + ": message id " + Dump(msg["id"]));
    CheckEq(msg["status"], "completed", where + ": message status");
    CheckEq(msg["role"], "assistant", where + ": message role");
    Check(r["id"].get<std::string>().starts_with("resp_"), where + ": response id " + Dump(r["id"]));
    CheckEq(r["reasoning"]["effort"], effort, where + ": reasoning.effort");
  }

  // Event types with runs of the same delta event collapsed into one.
  static Json Squash(const std::vector<Json>& events) {
    Json types = Json::array();
    for (const Json& e : events) {
      const std::string t = e["type"].get<std::string>();
      if (!types.empty() && types.back() == t && t.ends_with(".delta")) continue;
      types.push_back(t);
    }
    return types;
  }

  // Checks the event order and the deltas against the done events; returns the final response.
  static Json CheckStreamEvents(const std::vector<Json>& events, const std::string& effort, const std::string& where) {
    Json expect = {"response.created", "response.in_progress"};
    for (const std::string& kind : {std::string("reasoning"), std::string("output")}) {
      if (kind == "reasoning" && effort == "none") continue;
      const std::string text = kind == "reasoning" ? "response.reasoning_text" : "response.output_text";
      for (const std::string t : {"response.output_item.added", "response.content_part.added"}) expect.push_back(t);
      expect.push_back(text + ".delta");
      expect.push_back(text + ".done");
      for (const std::string t : {"response.content_part.done", "response.output_item.done"}) expect.push_back(t);
    }
    expect.push_back("response.completed");
    CheckEq(Squash(events), expect, where + ": event order");

    std::string reasoning, answer, reasoning_done, answer_done;
    for (const Json& e : events) {
      if (e["type"] == "response.reasoning_text.delta") reasoning += e["delta"].get<std::string>();
      if (e["type"] == "response.output_text.delta") answer += e["delta"].get<std::string>();
      if (e["type"] == "response.reasoning_text.done") reasoning_done = e["text"].get<std::string>();
      if (e["type"] == "response.output_text.done") answer_done = e["text"].get<std::string>();
    }
    if (effort != "none") {
      CheckEq(reasoning, kReasoning, where + ": reasoning deltas");
      CheckEq(reasoning_done, kReasoning, where + ": reasoning_text.done");
    }
    CheckEq(answer, kAnswer, where + ": output_text deltas");
    CheckEq(answer_done, kAnswer, where + ": output_text.done");
    const Json done = events.back()["response"];
    CheckEq(done["id"], events.front()["response"]["id"], where + ": final id against response.created");
    CheckAccumulated(events, done, where);
    return done;
  }

  // What the SDK's responses.stream helper does: build the output from added items, parts and deltas.
  // The snapshot must hold the same items and texts as the final response.
  static void CheckAccumulated(const std::vector<Json>& events, Json done, const std::string& where) {
    Json output = events.front()["response"]["output"];
    for (const Json& e : events) {
      const std::string t = e["type"].get<std::string>();
      if (t == "response.output_item.added") {
        CheckEq(e["output_index"], output.size(), where + ": output_item.added output_index");
        output.push_back(e["item"]);
        continue;
      }
      if (!e.contains("output_index") || !e.contains("content_index")) continue;
      const std::size_t index = e["output_index"].get<std::size_t>();
      Check(index < output.size(), where + ": " + t + " for an item that was not added");
      Json& item = output[index];
      CheckEq(e["item_id"], item["id"], where + ": " + t + " item_id");
      if (t == "response.content_part.added") {
        CheckEq(e["content_index"], item["content"].size(), where + ": content_part.added content_index");
        item["content"].push_back(e["part"]);
      } else if (t.ends_with(".delta")) {
        Json& part = item["content"][e["content_index"].get<std::size_t>()];
        part["text"] = part["text"].get<std::string>() + e["delta"].get<std::string>();
      }
    }
    CheckEq(output.size(), done["output"].size(), where + ": accumulated output size");
    for (std::size_t i = 0; i < output.size(); ++i) {
      CheckEq(output[i]["id"], done["output"][i]["id"], where + ": accumulated item id");
      CheckEq(output[i]["type"], done["output"][i]["type"], where + ": accumulated item type");
      Json texts = Json::array(), want = Json::array();
      for (const Json& part : output[i]["content"]) texts.push_back(part["text"]);
      for (const Json& part : done["output"][i]["content"]) want.push_back(part["text"]);
      CheckEq(texts, want, where + ": accumulated texts");
    }
  }

  void Health() {
    const Reply reply = Request("GET", "/health");
    Check(reply.status == 200, "GET /health: status " + std::to_string(reply.status));
    CheckEq(Json::parse(reply.body), {{"status", "ok"}}, "GET /health");
  }

  void Models() {
    const Reply reply = Request("GET", "/v1/models");
    Check(reply.status == 200, "GET /v1/models: status " + std::to_string(reply.status));
    Json models = Json::parse(reply.body);
    CheckEq(models["object"], "list", "GET /v1/models: object");
    Check(models["data"].is_array() && models["data"].size() == 1, "GET /v1/models: data " + Dump(models["data"]));
    Json m = models["data"][0];
    Required(m, "GET /v1/models data[0]", "id", Kind::kString);
    Required(m, "GET /v1/models data[0]", "created", Kind::kInteger);
    Required(m, "GET /v1/models data[0]", "object", Kind::kString);
    Required(m, "GET /v1/models data[0]", "owned_by", Kind::kString);
    CheckEq(m["object"], "model", "GET /v1/models: data[0].object");
    CheckEq(m["owned_by"], "yah", "GET /v1/models: data[0].owned_by");
  }

  void CreateStringInput() {
    const std::string where = "create: string input";
    Json r = Create({{"model", "qwen-test"}, {"input", "Hi there"}}, where);
    CheckFull(r, "high", where);
    CheckEq(r["model"], "qwen-test", where + ": model");
    CheckEq(r["metadata"], Json::object(), where + ": metadata");
    CheckEq(r["tools"], Json::array(), where + ": tools");
    CheckEq(r["parallel_tool_calls"], false, where + ": parallel_tool_calls");
  }

  void CreateMessageList() {
    const std::string where = "create: message list, instructions, effort none";
    const Json input = {
        {{"role", "user"}, {"content", "Hi"}},
        {{"role", "assistant"}, {"content", {{{"type", "output_text"}, {"text", "Hello"}}}}},
        {{"type", "message"}, {"role", "user"}, {"content", {{{"type", "input_text"}, {"text", "Again"}}}}}};
    Json r = Create({{"model", "qwen-test"},
                     {"instructions", "Be brief."},
                     {"input", input},
                     {"reasoning", {{"effort", "none"}}},
                     {"metadata", {{"k", "v"}}},
                     {"temperature", 0.2},
                     {"top_p", 0.5},
                     {"store", false}},
                    where);
    CheckFull(r, "none", where);
    CheckEq(r["instructions"], "Be brief.", where + ": instructions");
    CheckEq(r["metadata"], {{"k", "v"}}, where + ": metadata");
    CheckEq(r["temperature"], 0.2, where + ": temperature");
    CheckEq(r["top_p"], 0.5, where + ": top_p");
  }

  void CreateMultiTurn() {
    const std::string where = "create: output items as input";
    Json first = Create({{"model", "m"}, {"input", "Hi"}, {"reasoning", {{"effort", "low"}}}}, where);
    CheckFull(first, "low", where + " (first, low)");
    Json history = Json::array({{{"role", "user"}, {"content", "Hi"}}});
    for (const Json& item : first["output"]) history.push_back(item);
    history.push_back({{"role", "user"}, {"content", "More"}});
    Json r = Create({{"model", "m"}, {"input", history}, {"reasoning", {{"effort", "medium"}}}}, where);
    CheckFull(r, "medium", where + " (second, medium)");
    Check(r["usage"]["input_tokens"].get<int>() > first["usage"]["input_tokens"].get<int>(),
          where + ": the history did not grow the prompt");
    Json minimal = Create({{"model", "m"}, {"input", "Hi"}, {"reasoning", {{"effort", "minimal"}}}}, where);
    CheckFull(minimal, "minimal", where + " (minimal)");
    for (const std::string effort : {"high", "xhigh", "max"}) {
      CheckFull(Create({{"model", "m"}, {"input", "Hi"}, {"reasoning", {{"effort", effort}}}}, where), effort,
                where + " (" + effort + ")");
    }
  }

  void Streaming() {
    std::vector<Json> events = Stream({{"model", "m"}, {"input", "Hi"}}, "stream: effort high");
    CheckFull(CheckStreamEvents(events, "high", "stream: effort high"), "high", "stream: effort high");
    events = Stream(
        {{"model", "m"}, {"input", {{{"role", "user"}, {"content", "Hi"}}}}, {"reasoning", {{"effort", "none"}}}},
        "stream: effort none");
    CheckFull(CheckStreamEvents(events, "none", "stream: effort none"), "none", "stream: effort none");
  }

  void MaxOutputTokens() {
    std::string where = "max_output_tokens 5 (inside reasoning)";
    Json r = Create({{"model", "m"}, {"input", "Hi"}, {"max_output_tokens", 5}}, where);
    CheckEq(r["status"], "incomplete", where + ": status");
    CheckEq(r["incomplete_details"], {{"reason", "max_output_tokens"}}, where + ": incomplete_details");
    CheckEq(OutputTypes(r), {"reasoning"}, where + ": output types");
    Check(kReasoning.starts_with(r["output"][0]["content"][0]["text"].get<std::string>()),
          where + ": reasoning is not a prefix: " + Dump(r["output"][0]));
    CheckEq(r["usage"]["output_tokens"], 5, where + ": output_tokens");
    CheckEq(r["usage"]["output_tokens_details"]["reasoning_tokens"], 5, where + ": reasoning_tokens");
    CheckEq(r["max_output_tokens"], 5, where + ": max_output_tokens");
    CheckUsage(r, where);

    where = "max_output_tokens 20 (inside the answer)";
    r = Create({{"model", "m"}, {"input", "Hi"}, {"max_output_tokens", 20}}, where);
    CheckEq(r["status"], "incomplete", where + ": status");
    CheckEq(r["incomplete_details"], {{"reason", "max_output_tokens"}}, where + ": incomplete_details");
    CheckEq(OutputTypes(r), {"reasoning", "message"}, where + ": output types");
    CheckEq(r["output"][1]["status"], "incomplete", where + ": message status");
    const std::string text = OutputText(r);
    Check(kAnswer.starts_with(text) && text != kAnswer, where + ": answer is not a strict prefix: " + text);
    CheckEq(r["usage"]["output_tokens"], 20, where + ": output_tokens");
    CheckUsage(r, where);

    where = "max_output_tokens 5, streamed";
    std::vector<Json> events = Stream({{"model", "m"}, {"input", "Hi"}, {"max_output_tokens", 5}}, where);
    CheckEq(events.back()["type"], "response.incomplete", where + ": last event");
    CheckEq(events.back()["response"]["status"], "incomplete", where + ": status");

    where = "max_output_tokens 5, effort none, streamed";
    events =
        Stream({{"model", "m"}, {"input", "Hi"}, {"max_output_tokens", 5}, {"reasoning", {{"effort", "none"}}}}, where);
    CheckEq(events.back()["type"], "response.incomplete", where + ": last event");
    CheckEq(events.back()["response"]["output"][0]["status"], "incomplete", where + ": message status");
  }

  // A 400 with the OpenAI error object. An empty param or code is not checked.
  void Bad(const std::string& body, const std::string& param = "", const std::string& code = "") {
    const std::string where = "error case " + body.substr(0, 120);
    const Reply reply = Request("POST", "/v1/responses", body);
    Check(reply.status == 400, where + ": status " + std::to_string(reply.status) + ": " + reply.body);
    const Json data = Json::parse(reply.body);
    Check(data.is_object() && data.size() == 1 && data.contains("error"), where + ": body " + reply.body);
    const Json& err = data["error"];
    Check(err.is_object() && err.size() == 4 && err.contains("message") && err.contains("type") &&
              err.contains("param") && err.contains("code"),
          where + ": error keys " + Dump(err));
    CheckEq(err["type"], "invalid_request_error", where + ": type");
    Check(err["message"].is_string() && !err["message"].get<std::string>().empty(), where + ": empty message");
    Check(err["param"].is_null() || err["param"].is_string(), where + ": param type");
    Check(err["code"].is_null() || err["code"].is_string(), where + ": code type");
    if (!param.empty()) CheckEq(err["param"], param, where + ": param");
    if (!code.empty()) CheckEq(err["code"], code, where + ": code");
  }

  void Errors() {
    const auto body = [](const Json& extra) {
      Json b = {{"model", "m"}, {"input", "Hi"}};
      for (const auto& [key, value] : extra.items()) b[key] = value;
      return Dump(b);
    };
    const Json tools = Json::array({{{"type", "function"}, {"name", "f"}}});
    Bad(body({{"tools", tools}}), "tools", "unsupported_parameter");
    Bad(body({{"previous_response_id", "resp_1"}}), "previous_response_id");
    Bad(body({{"tool_choice", "required"}}), "tool_choice");
    Bad("{not json");
    Bad("[1, 2]");
    Bad(R"({"input": "Hi"})", "model");
    Bad(R"({"model": "m"})", "input");
    Bad(R"({"model": "m", "input": []})", "input");
    Bad(R"({"model": "m", "input": [{"type": "function_call", "name": "f"}]})", "input[0].type");
    Bad(R"({"model": "m", "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]})",
        "input[0].content[0].type");
    Bad(R"({"model": "m", "input": [{"role": "tool", "content": "x"}]})", "input[0].role");
    Bad(R"({"model": "m", "input": [{"role": "assistant", "content": "x"}]})", "input");
    Bad(body({{"reasoning", {{"effort", "huge"}}}}), "reasoning.effort");
    Bad(body({{"temperature", 3}}), "temperature");
    Bad(body({{"max_output_tokens", 0}}), "max_output_tokens");
    Bad(body({{"max_output_tokens", 100000}}), "max_output_tokens", "context_length_exceeded");
    std::string long_input;
    for (int i = 0; i < 5000; ++i) long_input += "hello ";
    Bad(Dump({{"model", "m"}, {"input", long_input}}), "input", "context_length_exceeded");
    // What the SDK sends for tools with parameters.
    const Json sdk_tools = Json::array({{{"type", "function"}, {"name", "f"}, {"parameters", Json::object()}}});
    Bad(body({{"tools", sdk_tools}}), "tools");

    // Unsupported fields at default or empty values are accepted.
    const Reply ok = Request("POST", "/v1/responses",
                             body({{"tools", Json::array()},
                                   {"tool_choice", "auto"},
                                   {"previous_response_id", nullptr},
                                   {"text", {{"format", {{"type", "text"}}}}},
                                   {"user", "u"},
                                   {"store", true},
                                   {"max_output_tokens", 3}}));
    Check(ok.status == 200, "accepted defaults: status " + std::to_string(ok.status) + ": " + ok.body);

    const Reply missing = Request("GET", "/v1/nothing");
    Check(missing.status == 404, "GET /v1/nothing: status " + std::to_string(missing.status));
    const Json err = Json::parse(missing.body)["error"];
    Check(err["message"].is_string() && !err["message"].get<std::string>().empty(), "GET /v1/nothing: " + missing.body);
  }

  // A client that leaves mid-stream stops the generation; the server keeps serving.
  void Disconnect(ServerProcess& server) {
    std::string received;
    int events = 0;
    const httplib::Result res =
        client_.Post("/v1/responses", httplib::Headers(), Dump({{"model", "m"}, {"input", "Hi"}, {"stream", true}}),
                     "application/json", [&](const char* data, std::size_t n) {
                       received.append(data, n);
                       events = 0;
                       for (std::size_t p = 0; (p = received.find("\n\n", p)) != std::string::npos; p += 2) ++events;
                       return events < 4;
                     });
    Check(!res && res.error() == httplib::Error::Canceled, "disconnect: the request was not cancelled");
    Check(received.starts_with("event: response.created\n"), "disconnect: not a stream: " + received.substr(0, 200));
    // Requests run one at a time, so the cancelled one has logged by the time this one returns.
    CheckFull(Create({{"model", "m"}, {"input", "Hi"}}, "after disconnect"), "high", "after disconnect");
    Check(server.Log().find("finish=cancelled") != std::string::npos,
          "disconnect: the server did not log finish=cancelled");
  }

  httplib::Client client_;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: yah_serve_test <model.gguf> [yah_server]\n");
    return 2;
  }
  const std::filesystem::path dir = std::filesystem::read_symlink("/proc/self/exe").parent_path();
  const std::string model = argv[1];
  const std::string exe = argc > 2 ? argv[2] : (dir / "yah_server").string();
  bool failed = false;

  try {
    int total = 0;
    const int passed = CheckChatTemplate(dir / "../serve/testdata/chat_template_cases.json", &total);
    std::printf("chat template: %d/%d\n", passed, total);
    failed = passed != total;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL chat template: %s\n", e.what());
    failed = true;
  }

  std::fflush(stdout);
  const int port = FreePort();
  ServerProcess server(exe, model, port);
  try {
    httplib::Client health("127.0.0.1", port);
    bool up = false;
    for (int i = 0; i < 300 && !up && !server.Exited(); ++i) {
      const httplib::Result res = health.Get("/health");
      up = res && res->status == 200;
      if (!up) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!up) throw Failure("yah_server did not come up on port " + std::to_string(port));
    Tester(port).Run(server);
    std::printf("responses: %d checks passed\n", g_checks);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL responses: %s\n--- yah_server log ---\n%s", e.what(), server.Log().c_str());
    failed = true;
  }
  return failed ? 1 : 0;
}
