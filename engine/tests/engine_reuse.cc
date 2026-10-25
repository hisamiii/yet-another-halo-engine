// GPU test of the engine's multi-turn reuse (engine.hpp): a prompt that extends the cached sequence must resume from
// a snapshot slot and produce exactly the tokens a fresh or continued run produces. Greedy, token ids in and out.
//
// usage: engine_reuse <model> <prefill set> <decode set>    (a 32K set: the sequences reach ~5.1K tokens)
//
//   decode slot: a 1.3K-token prompt decodes past 2048; the prompt + its first k output tokens (2148 tokens) resumes at
//     2048 and runs the rest as decode steps (Tail::kDecode), the same work the first run did, so the next tokens
//     must equal the first run's output from k on, bit for bit.
//   prefill slot: a ~5K-token prompt, then one that shares all but its last tokens resumes at 4096 and must equal a
//     fresh run of it (an unrelated prompt in between drops the cache).
#include <cstdio>
#include <string>
#include <vector>

#include "model/engine.hpp"

namespace {

using yah::core::TokenId;
using yah::model::Engine;

std::vector<TokenId> Run(Engine& e, const std::vector<TokenId>& prompt, std::uint32_t max, std::uint32_t* cached) {
  yah::model::GenerateParams p;
  p.max_tokens = max;
  std::vector<TokenId> out;
  const auto r = e.Generate(prompt, p, [&](TokenId t) {
    out.push_back(t);
    return true;
  });
  if (cached) *cached = r.cached_tokens;
  std::printf("  prompt %zu, cached %u, %u tokens, prefill %.0f ms\n", prompt.size(), r.cached_tokens,
              r.generated_tokens, r.prefill_ms);
  return out;
}

std::string Facts(int k, const char* salt) {
  std::string s;
  for (int i = 1; i <= k; ++i)
    s += "Fact " + std::to_string(i) + ": the " + salt + " number " + std::to_string(i) + " times " +
         std::to_string(i + 3) + " is " + std::to_string(i * (i + 3)) + ".\n";
  return s;
}

int fails = 0;
void Check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  fails += !ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 && !(argc == 5 && std::string(argv[4]) == "--npu")) {
    std::fprintf(stderr, "usage: engine_reuse <model> <prefill set> <decode set> [--npu]\n");
    return 2;
  }
  Engine::Options options{argv[1], argv[2], argv[3]};
  options.npu = argc == 5;
  Engine e(options);
  if (options.npu) {
    if (!e.npu_status().empty()) {
      std::fprintf(stderr, "engine_reuse: npu off: %s\n", e.npu_status().c_str());
      return 1;
    }
    std::printf("npu on\n");
  }
  const auto& tok = e.tokenizer();
  std::uint32_t cached = 0;

  // decode slot
  auto p1 = tok.Encode(Facts(60, "plum") + "\nList facts 1 to 60 verbatim, one per line.\n");
  const std::uint32_t n1 = static_cast<std::uint32_t>(p1.size());
  const std::uint32_t q = 2148, k = q - n1;
  const auto o1 = Run(e, p1, q + 64 - n1, nullptr);
  auto pq = p1;
  pq.insert(pq.end(), o1.begin(), o1.begin() + k);
  e.set_tail(Engine::Tail::kDecode);
  const auto oq = Run(e, pq, 48, &cached);
  e.set_tail(Engine::Tail::kAuto);
  Check(cached == 2048, "decode slot: resumed at " + std::to_string(cached));
  Check(std::vector<TokenId>(o1.begin() + k, o1.begin() + k + 48) == oq, "decode slot: continuation = first run");

  // prefill slot
  auto p2 = tok.Encode(Facts(200, "red") + "\nWhat is fact 137?\n");
  Run(e, p2, 16, nullptr);
  auto p3 = std::vector<TokenId>(p2.begin(), p2.end() - 6);
  const auto tail = tok.Encode("\nAnd what is fact 12? Answer in one sentence.\n");
  p3.insert(p3.end(), tail.begin(), tail.end());
  const auto reused = Run(e, p3, 32, &cached);
  Check(cached == 4096, "prefill slot: resumed at " + std::to_string(cached));
  Run(e, tok.Encode("Say hi."), 4, nullptr);
  std::uint32_t cached_fresh = 1;
  const auto fresh = Run(e, p3, 32, &cached_fresh);
  Check(cached_fresh == 0, "fresh run: cached " + std::to_string(cached_fresh));
  Check(reused == fresh, "prefill slot: reused = fresh");
  std::printf("%s\n", fails ? "engine_reuse: FAIL" : "engine_reuse: ok");
  return fails ? 1 : 0;
}
