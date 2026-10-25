// GPU test of speculative decoding through the engine (Engine::Speculate, --draft mtp): each prompt runs greedy with
// plain decode steps, then with MTP drafts and verify rounds, and the outputs must be equal token for token. Prints
// the acceptance and both decode rates. Needs a GGUF with the MTP layer and a decode set from emit_decode.py with
// VERIFY (the verify GEMVs and the MTP kernels).
//
// usage: spec_engine <model> <prefill set> <decode set> [--draft-len N] [--npu]
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "model/engine.hpp"

namespace {

using yah::core::TokenId;
using yah::model::Engine;

struct Out {
  std::vector<TokenId> ids;
  yah::model::GenerateResult r;
};

Out Run(Engine& e, const std::vector<TokenId>& prompt, std::uint32_t max) {
  yah::model::GenerateParams p;
  p.max_tokens = max;
  Out o;
  o.r = e.Generate(prompt, p, [&](TokenId t) {
    o.ids.push_back(t);
    return true;
  });
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: spec_engine <model> <prefill set> <decode set> [--draft-len N] [--npu]\n");
    return 2;
  }
  Engine::Options o;
  o.model = argv[1], o.prefill_hal = argv[2], o.decode_hal = argv[3], o.draft = "mtp";
  for (int i = 4; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--npu") o.npu = true;
    else if (a == "--draft-len" && i + 1 < argc) o.draft_len = static_cast<std::uint32_t>(std::atoi(argv[++i]));
    else return 2;
  }
  try {
    Engine e(o);
    if (!e.draft_status().empty()) throw std::runtime_error("draft off: " + e.draft_status());
    const auto& tok = e.tokenizer();
    const auto chat = [&](const std::string& user) {
      return tok.Encode("<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    };
    std::string facts;
    for (int i = 1; i <= 90; ++i)
      facts += "Item " + std::to_string(i) + ": a crate of " + std::to_string(i * 7 % 13 + 2) + " lamps ships to depot " +
               std::to_string(i % 9) + ".\n";
    struct Case {
      const char* name;
      std::vector<TokenId> prompt;
      std::uint32_t gen;
    };
    std::vector<Case> cases = {
        {"code", chat("Write a Python function that parses an ISO 8601 date string without using datetime, with tests."), 384},
        {"prose", chat("Tell a short story about a lighthouse keeper who finds a message in a bottle."), 384},
        {"facts", chat(facts + "List every item that ships to depot 4, then total their lamps."), 320},
        {"raw", tok.Encode("The history of the printing press begins in the fifteenth century, when"), 256},
    };
    int bad = 0;
    double plain_ms = 0, spec_ms = 0;
    std::uint32_t plain_n = 0, spec_n = 0;
    for (const Case& c : cases) {
      e.set_draft(false);
      const Out a = Run(e, c.prompt, c.gen);
      e.set_draft(true);
      const Out b = Run(e, c.prompt, c.gen);
      std::size_t same = 0;
      while (same < a.ids.size() && same < b.ids.size() && a.ids[same] == b.ids[same]) ++same;
      const bool ok = a.ids == b.ids;
      bad += !ok;
      const double ta = a.r.generated_tokens / a.r.decode_ms * 1000, tb = b.r.generated_tokens / b.r.decode_ms * 1000;
      std::printf("%s %-6s prompt %5zu: %u tokens, same for %zu; accepted %u/%u (%.2f per round); plain %.1f t/s, spec "
                  "%.1f t/s (x%.2f)\n",
                  ok ? "ok  " : "FAIL", c.name, c.prompt.size(), a.r.generated_tokens, same, b.r.accepted, b.r.drafted,
                  b.r.drafted ? 1.0 * b.r.accepted / (b.r.generated_tokens - b.r.accepted) : 0.0, ta, tb, tb / ta);
      plain_ms += a.r.decode_ms, spec_ms += b.r.decode_ms, plain_n += a.r.generated_tokens, spec_n += b.r.generated_tokens;
      if (!ok) std::printf("  plain: %s\n  spec:  %s\n", tok.Decode(a.ids).c_str(), tok.Decode(b.ids).c_str());
    }
    std::printf("%s spec == plain on %zu prompts; decode %.1f -> %.1f t/s\n", bad ? "FAIL" : "ok  ", cases.size(),
                plain_n / plain_ms * 1000, spec_n / spec_ms * 1000);
    return bad ? 1 : 0;
  } catch (const std::exception& err) {
    std::fprintf(stderr, "spec_engine: %s\n", err.what());
    return 1;
  }
}
