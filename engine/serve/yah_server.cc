// yah_server: the OpenAI Responses API (POST /v1/responses) and Chat Completions API (POST /v1/chat/completions)
// over one TextGenerator, one generation at a time.
// Built with YAH_SERVE_FAKE_ONLY (the CMake target yah_server_fake) it has only --fake: no Engine, no HRX, no GPU.
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#ifndef YAH_SERVE_FAKE_ONLY
#include "model/engine.hpp"
#endif
#include "model/generator.hpp"
#include "serve/chat_completions.hpp"
#include "serve/chat_template.hpp"
#include "serve/fake_generator.hpp"
#include "serve/http.hpp"
#include "serve/responses.hpp"

namespace {

using yah::core::TokenId;
using yah::model::GenerateParams;
using yah::model::GenerateResult;
using yah::model::SamplingParams;
using yah::model::TextGenerator;
using namespace yah::serve;
namespace http = yah::serve::http;

int Usage() {
  std::fprintf(stderr,
               "usage: yah_server --model <model> [--prefill <hal set> --decode <hal set>] [--host 127.0.0.1] "
               "[--port 8080] [--npu [--npu-down <pack>]] [--draft mtp|off [--draft-len N]]\n"
               "       yah_server --model <model> --fake [--host ...] [--port ...]   canned replies, no GPU\n"
               "In a release bundle the HAL sets default to the bundle's hal/prefill and hal/decode.\n"
               "--npu: the NPU runs the prefill set's FFN block (a set from tools/emit_npu_ffn.py), its down pack from\n"
               "<prefill>/npu_down.bin unless --npu-down; without what it needs the GPU runs everything.\n"
               "--draft mtp: greedy requests decode speculatively, drafting with the model's MTP layer (a GGUF with\n"
               "blk.N.nextn.*) and verifying with the decode set's verify GEMVs; --draft-len N fixes the draft length\n"
               "(default: adapts to the acceptance).\n");
  return 2;
}

// The release bundle (engine/release/bundle.sh): bin/yah_server beside lib/ (libhrx, the HSA runtime) and hal/.
std::string BundleDir() {
  char path[4096];
  const ssize_t n = ::readlink("/proc/self/exe", path, sizeof path - 1);
  if (n <= 0) return {};
  const std::string exe(path, static_cast<std::size_t>(n));
  return exe.substr(0, exe.rfind('/')) + "/..";
}
bool Exists(const std::string& path) { return ::access(path.c_str(), R_OK) == 0; }

class Server {
 public:
  explicit Server(TextGenerator& generator) : generator_(generator) {
    for (const char* stop : {"<|im_end|>", "<|endoftext|>"}) {
      if (auto id = generator_.tokenizer().FindSpecial(stop)) stop_ids_.push_back(*id);
    }
    if (stop_ids_.empty()) throw std::runtime_error("the tokenizer has no <|im_end|>");
    created_ = std::time(nullptr);
  }

  void Responses(const http::Request& req, http::Response& res) {
    try {
      const ResponseRequest request = ParseResponseRequest(ParseBody(req));
      SamplingParams sampling;
      sampling.temperature = static_cast<float>(request.temperature);
      sampling.top_p = static_cast<float>(request.top_p);
      sampling.top_k = 20;   // Qwen3.8's recommended top_k, as /v1/chat/completions defaults to
      std::vector<TokenId> prompt;
      const GenerateParams params =
          Prepare(request.messages, request.chat, request.max_output_tokens, sampling, "input", "max_output_tokens",
                  &prompt);
      Serve<ResponseStream>(request, request.stream, prompt, params, res);
    } catch (const ApiError& error) {
      res.status = error.status;
      res.Set(Dump(ErrorBody(error)), "application/json");
    }
  }

  void ChatCompletions(const http::Request& req, http::Response& res) {
    try {
      const ChatRequest request = ParseChatRequest(ParseBody(req));
      std::vector<TokenId> prompt;
      GenerateParams params =
          Prepare(request.messages, request.chat, request.max_tokens, request.sampling, "messages", "max_tokens", &prompt);
      if (request.ignore_eos) params.stop_ids.clear();
      Serve<ChatStream>(request, request.stream, prompt, params, res);
    } catch (const ApiError& error) {
      res.status = error.status;
      res.Set(Dump(ErrorBody(error)), "application/json");
    }
  }

  void Models(const http::Request&, http::Response& res) {
    const Json model = {
        {"id", generator_.model_name()}, {"object", "model"}, {"created", created_}, {"owned_by", "yah"}};
    res.Set(Dump({{"object", "list"}, {"data", Json::array({model})}}), "application/json");
  }

 private:
  static Json ParseBody(const http::Request& req) {
    try {
      return Json::parse(req.body);
    } catch (const Json::parse_error& error) {
      throw ApiError(400, std::string("Invalid JSON body: ") + error.what());
    }
  }

  // The prompt tokens and the generation limits of a chat; ApiError (400) past the context.
  GenerateParams Prepare(const std::vector<ChatMessage>& messages, const ChatOptions& chat,
                         std::optional<std::uint32_t> max_tokens, const SamplingParams& sampling,
                         const char* input_param, const char* max_param, std::vector<TokenId>* prompt) {
    std::string text;
    try {
      text = RenderChat(messages, chat);
    } catch (const std::invalid_argument& error) {
      throw ApiError(400, error.what(), input_param);
    }
    *prompt = generator_.tokenizer().Encode(text);
    const std::uint32_t context = generator_.context();
    const auto prompt_tokens = static_cast<std::uint32_t>(prompt->size());
    GenerateParams params;
    params.stop_ids = stop_ids_;
    params.sampling = sampling;
    if (prompt_tokens >= context) {
      throw ApiError(400,
                     "The input has " + std::to_string(prompt_tokens) + " tokens; the context is " +
                         std::to_string(context) + ".",
                     input_param, "context_length_exceeded");
    }
    params.max_tokens = max_tokens.value_or(context - prompt_tokens);
    if (params.max_tokens > context - prompt_tokens) {
      throw ApiError(400,
                     std::string(input_param) + " (" + std::to_string(prompt_tokens) + " tokens) + " + max_param +
                         " (" + std::to_string(params.max_tokens) + ") exceeds the context of " +
                         std::to_string(context) + " tokens.",
                     max_param, "context_length_exceeded");
    }
    return params;
  }

  // One generation as a JSON body or, with stream, as server-sent events. S: ResponseStream or ChatStream.
  template <class S, class Request>
  void Serve(const Request& request, bool stream, const std::vector<TokenId>& prompt, const GenerateParams& params,
             http::Response& res) {
    if (!stream) {
      S s(request, nullptr);
      Run(s, prompt, params);
      res.Set(Dump(s.Response()), "application/json");
      return;
    }
    res.headers.emplace_back("Cache-Control", "no-cache");
    res.content_type = "text/event-stream";
    res.stream = [this, request, prompt, params](http::Sink& sink) { Stream<S>(request, prompt, params, sink); };
  }

  template <class S, class Request>
  void Stream(const Request& request, const std::vector<TokenId>& prompt, const GenerateParams& params,
              http::Sink& sink) {
    S stream(request, [&sink](const std::string& event) { return sink.Alive() && sink.Write(event); });
    try {
      Run(stream, prompt, params);
    } catch (const std::exception& error) {
      stream.Fail(error.what());
    }
  }

  template <class S>
  void Run(S& stream, const std::vector<TokenId>& prompt, const GenerateParams& params) {
    std::lock_guard lock(mutex_);
    GenerateResult result;
    result.prompt_tokens = static_cast<std::uint32_t>(prompt.size());
    result.finish_reason = "stop";
    try {
      // A client that left while waiting for the lock costs no generation.
      if (stream.Start()) {
        result = generator_.Generate(
            prompt, params, [&](TokenId id) { return stream.OnToken(generator_.tokenizer().DecodeToken(id)); });
      }
    } catch (const std::exception& error) {
      std::fprintf(stderr, "yah_server: %s failed: %s\n", stream.id().c_str(), error.what());
      throw;
    }
    stream.Finish(result);
    const double tok_s = Timings(result)["predicted_per_second"].get<double>();
    std::fprintf(stderr,
                 "yah_server: %s prompt=%u cached=%u output=%u prefill_ms=%.1f decode_tok_s=%.1f accepted=%u/%u "
                 "finish=%s\n",
                 stream.id().c_str(), result.prompt_tokens, result.cached_tokens, result.generated_tokens,
                 result.prefill_ms, tok_s, result.accepted, result.drafted,
                 stream.client_alive() ? result.finish_reason.c_str() : "cancelled");
  }

  TextGenerator& generator_;
  std::mutex mutex_;
  std::vector<TokenId> stop_ids_;
  std::int64_t created_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  std::string model, prefill, decode, host = "127.0.0.1";
  int port = 8080;
  bool fake = false, npu = false;
  std::string npu_down, draft = "off";
  std::uint32_t draft_len = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = i + 1 < argc;
    if (arg == "--fake") {
      fake = true;
    } else if (arg == "--model" && has_value) {
      model = argv[++i];
    } else if (arg == "--prefill" && has_value) {
      prefill = argv[++i];
    } else if (arg == "--decode" && has_value) {
      decode = argv[++i];
    } else if (arg == "--npu") {
      npu = true;
    } else if (arg == "--npu-down" && has_value) {
      npu_down = argv[++i];
    } else if (arg == "--draft" && has_value) {
      draft = argv[++i];
    } else if (arg == "--draft-len" && has_value) {
      draft_len = static_cast<std::uint32_t>(std::atoi(argv[++i]));
    } else if (arg == "--host" && has_value) {
      host = argv[++i];
    } else if (arg == "--port" && has_value) {
      port = std::atoi(argv[++i]);
    } else {
      return Usage();
    }
  }
  const std::string bundle = BundleDir();
  if (!fake && prefill.empty() && Exists(bundle + "/hal/prefill/dispatch.txt")) prefill = bundle + "/hal/prefill";
  if (!fake && decode.empty() && Exists(bundle + "/hal/decode/decode.txt")) decode = bundle + "/hal/decode";
  // libhrx loads the HSA runtime from this directory: the bundle's own copy unless the caller picked one.
  if (!std::getenv("IREE_HAL_AMDGPU_LIBHSA_PATH") && Exists(bundle + "/lib/libhsa-runtime64.so.1"))
    ::setenv("IREE_HAL_AMDGPU_LIBHSA_PATH", (bundle + "/lib").c_str(), 1);
  if (model.empty() || (!fake && (prefill.empty() || decode.empty()))) return Usage();

  try {
    std::unique_ptr<TextGenerator> generator;
    if (fake) {
      generator = std::make_unique<FakeGenerator>(model);
    } else {
#ifdef YAH_SERVE_FAKE_ONLY
      throw std::runtime_error("built without the GPU engine: only --fake");
#else
      yah::model::Engine::Options options;
      options.model = model;
      options.prefill_hal = prefill;
      options.decode_hal = decode;
      options.npu = npu;
      options.npu_down = npu_down;
      options.draft = draft;
      options.draft_len = draft_len;
      auto engine = std::make_unique<yah::model::Engine>(options);
      if (npu) {
        const std::string& why = engine->npu_status();
        std::fprintf(stderr, "yah_server: npu %s%s\n", why.empty() ? "on" : "off: ", why.c_str());
      }
      if (draft != "off") {
        const std::string& why = engine->draft_status();
        std::fprintf(stderr, "yah_server: draft %s %s%s\n", draft.c_str(), why.empty() ? "on" : "off: ", why.c_str());
      }
      generator = std::move(engine);
#endif
    }
    Server server(*generator);
    http::Server web;
    web.Post("/v1/responses", [&](const http::Request& req, http::Response& res) { server.Responses(req, res); });
    web.Post("/v1/chat/completions",
             [&](const http::Request& req, http::Response& res) { server.ChatCompletions(req, res); });
    web.Get("/v1/models", [&](const http::Request& req, http::Response& res) { server.Models(req, res); });
    web.Get("/health", [](const http::Request&, http::Response& res) { res.Set(R"({"status":"ok"})", "application/json"); });
    web.on_error = [](http::Response& res) {
      const ApiError error(res.status, res.status == 404 ? "Not found." : "HTTP error " + std::to_string(res.status));
      res.Set(Dump(ErrorBody(error)), "application/json");
    };
    web.on_exception = [](http::Response& res, const std::string& message) {
      res.status = 500;
      res.Set(Dump(ErrorBody(ApiError(500, message, "", "server_error", "server_error"))), "application/json");
    };
    std::fprintf(stderr, "yah_server: %s%s on http://%s:%d\n", generator->model_name().c_str(), fake ? " (fake)" : "",
                 host.c_str(), port);
    if (!web.Listen(host, port)) {
      std::fprintf(stderr, "yah_server: cannot listen on %s:%d\n", host.c_str(), port);
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "yah_server: %s\n", error.what());
    return 1;
  }
}
