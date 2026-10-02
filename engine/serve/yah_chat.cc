// yah_chat: terminal chat client for yah_server (OpenAI Responses API, streaming).
//
// usage: yah_chat [--url http://127.0.0.1:8080] [--effort none|low|medium|high] [--temperature T] [--max N]
//                 [--system TEXT]
// Type a message and press Enter. Commands: /reset (new conversation), /effort E, /temp T (empty: server default),
// /max N, /system TEXT, /quit. Reasoning streams dimmed; a stats line follows each reply. Ctrl-C stops the reply.
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "httplib/httplib.h"
#include "nlohmann/json.hpp"

namespace {

using Json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

constexpr const char* kDim = "\033[2m";
constexpr const char* kReset = "\033[0m";

std::atomic<bool> g_interrupted{false};

struct Options {
  std::string url = "http://127.0.0.1:8080";
  std::string effort = "none";
  std::optional<double> temperature;
  int max_tokens = 2048;
  std::string system;
};

bool ValidEffort(const std::string& e) {
  return e == "none" || e == "low" || e == "medium" || e == "high";
}

int Usage() {
  std::fprintf(stderr,
               "usage: yah_chat [--url http://127.0.0.1:8080] [--effort none|low|medium|high] [--temperature T] "
               "[--max N] [--system TEXT]\n");
  return 2;
}

// Server-sent events: "event: <type>\ndata: <json>\n\n". Calls on_event(type, data) per complete event.
class SseParser {
 public:
  template <class F>
  void Feed(const char* data, size_t n, F&& on_event) {
    buf_.append(data, n);
    size_t end;
    while ((end = buf_.find("\n\n")) != std::string::npos) {
      std::string block = buf_.substr(0, end);
      buf_.erase(0, end + 2);
      std::string type, payload;
      size_t pos = 0;
      while (pos <= block.size()) {
        size_t nl = block.find('\n', pos);
        if (nl == std::string::npos) nl = block.size();
        const std::string line = block.substr(pos, nl - pos);
        if (line.rfind("event: ", 0) == 0) type = line.substr(7);
        if (line.rfind("data: ", 0) == 0) payload = line.substr(6);
        pos = nl + 1;
      }
      if (!type.empty() && !payload.empty()) on_event(type, Json::parse(payload));
    }
  }
  [[nodiscard]] const std::string& rest() const { return buf_; }

 private:
  std::string buf_;
};

struct Turn {
  std::optional<Json> final;  // the response object of response.completed / incomplete / failed
  std::optional<Clock::time_point> first;
  std::string error;
};

Turn Send(httplib::Client& client, const Json& body) {
  Turn turn;
  SseParser sse;
  std::string raw;  // the body, for error replies
  bool in_reasoning = false;
  const auto on_event = [&](const std::string& type, const Json& d) {
    const bool reasoning = type == "response.reasoning_text.delta";
    if (reasoning || type == "response.output_text.delta") {
      if (!turn.first) turn.first = Clock::now();
      if (reasoning != in_reasoning) std::fputs(reasoning ? kDim : kReset, stdout);
      in_reasoning = reasoning;
      std::fputs(d["delta"].get<std::string>().c_str(), stdout);
      std::fflush(stdout);
    } else if (type == "response.output_item.done" && d["item"]["type"] == "reasoning") {
      std::fputs(kReset, stdout);
      in_reasoning = false;
      std::fputs("\n\n", stdout);
    } else if (type == "response.completed" || type == "response.incomplete" || type == "response.failed") {
      turn.final = d["response"];
    }
  };
  const httplib::Headers headers = {{"Accept", "text/event-stream"}};
  auto res = client.Post("/v1/responses", headers, body.dump(), "application/json", [&](const char* data, size_t n) {
    if (raw.size() < 65536) raw.append(data, n);
    try {
      sse.Feed(data, n, on_event);
    } catch (const std::exception& e) {
      turn.error = std::string("bad event: ") + e.what();
      return false;
    }
    return !g_interrupted.load();
  });
  if (in_reasoning) std::fputs(kReset, stdout);
  if (g_interrupted.load()) {
    turn.error = "interrupted";
  } else if (!res) {
    turn.error = "cannot reach the server: " + httplib::to_string(res.error());
  } else if (res->status != 200) {
    turn.error = "error " + std::to_string(res->status) + ": " + raw;
  } else if (!turn.final && turn.error.empty()) {
    turn.error = "the stream ended without a final response";
  }
  return turn;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool more = i + 1 < argc;
    if (a == "--url" && more) {
      o.url = argv[++i];
    } else if (a == "--effort" && more && ValidEffort(argv[i + 1])) {
      o.effort = argv[++i];
    } else if (a == "--temperature" && more) {
      o.temperature = std::atof(argv[++i]);
    } else if (a == "--max" && more) {
      o.max_tokens = std::atoi(argv[++i]);
    } else if (a == "--system" && more) {
      o.system = argv[++i];
    } else {
      return Usage();
    }
  }
  httplib::Client client(o.url);
  client.set_read_timeout(3600);  // a long reasoning reply can stream for minutes
  // No SA_RESTART: Ctrl-C at the prompt makes getline fail, so the client exits; during a reply it stops the reply.
  struct sigaction sa = {};
  sa.sa_handler = [](int) { g_interrupted = true; };
  sigaction(SIGINT, &sa, nullptr);

  std::vector<Json> history;  // Responses API input items: user messages and the model's previous output items
  std::printf("yah chat (%s), effort=%s. /reset /effort E /temp T /max N /system TEXT /quit\n", o.url.c_str(),
              o.effort.c_str());
  std::string line;
  while (true) {
    std::printf("\n> ");
    std::fflush(stdout);
    g_interrupted = false;
    if (!std::getline(std::cin, line) || g_interrupted) {
      std::printf("\n");
      return 0;
    }
    if (line.empty()) continue;
    if (line[0] == '/') {
      const size_t sp = line.find(' ');
      const std::string cmd = line.substr(0, sp), arg = sp == std::string::npos ? "" : line.substr(sp + 1);
      if (cmd == "/quit") return 0;
      if (cmd == "/reset") {
        history.clear();
        std::printf("(new conversation)\n");
      } else if (cmd == "/effort" && ValidEffort(arg)) {
        o.effort = arg;
      } else if (cmd == "/temp") {
        o.temperature = arg.empty() ? std::nullopt : std::optional<double>(std::atof(arg.c_str()));
      } else if (cmd == "/max" && !arg.empty() && std::atoi(arg.c_str()) > 0) {
        o.max_tokens = std::atoi(arg.c_str());
      } else if (cmd == "/system") {
        o.system = arg;
      } else {
        std::printf("commands: /reset /effort none|low|medium|high /temp T /max N /system TEXT /quit\n");
      }
      continue;
    }
    history.push_back({{"role", "user"}, {"content", line}});
    Json body = {{"model", "yah"},
                 {"input", history},
                 {"stream", true},
                 {"max_output_tokens", o.max_tokens},
                 {"reasoning", {{"effort", o.effort}}}};
    if (o.temperature) body["temperature"] = *o.temperature;
    if (!o.system.empty()) body["instructions"] = o.system;

    const auto t0 = Clock::now();
    Turn turn = Send(client, body);
    if (!turn.error.empty()) {
      std::printf("\n(%s)\n", turn.error.c_str());
      history.pop_back();
      continue;
    }
    const Json& r = *turn.final;
    for (const auto& item : r["output"]) history.push_back(item);  // the model's turn, for the next request
    const Json usage = r.value("usage", Json::object());
    const int in = usage.value("input_tokens", 0), out = usage.value("output_tokens", 0);
    const auto first = turn.first.value_or(t0);
    const double ttft = std::chrono::duration<double>(first - t0).count();
    const double gen_s = std::chrono::duration<double>(Clock::now() - first).count();
    const double rate = out > 1 && gen_s > 0 ? (out - 1) / gen_s : 0.0;
    const std::string status = r.value("status", "");
    std::printf("\n%s[%d in, %d out, first token %.2f s, %.1f tok/s%s]%s\n", kDim, in, out, ttft, rate,
                status == "completed" ? "" : (", " + status).c_str(), kReset);
  }
}
