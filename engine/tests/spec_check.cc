// The speculative verify against plain decode steps (LoomDecoder::Verify / Accept), on a decode set with "verify":
//   spec_check <model> <decode set> [gen] [--time]
// A greedy reference of gen tokens after a fixed prompt (decode steps, logits kept), then the same through verify
// rounds whose drafts are the reference's tokens with one wrong per round at a rotating index, so every accept count
// 0 .. T - 1 and the rollbacks behind it come up. Every verified token's logits must equal the reference's bit for bit
// and the output must be the reference. --time: then the cost of a step, a verify of 1 .. T tokens and an MTP draft
// (each synced, the median of 9).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "model/loom_decoder.hpp"
#include "model/loom_runtime.hpp"

using namespace yah::model;  // NOLINT(google-build-using-namespace)

int main(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    std::fprintf(stderr, "usage: spec_check <model> <decode set> [gen] [--time]\n");
    return 2;
  }
  const std::uint32_t gen = argc > 3 ? static_cast<std::uint32_t>(std::atoi(argv[3])) : 96;
  const bool timing = argc > 4 && std::string(argv[4]) == "--time";
  try {
    auto gguf = yah::core::Gguf::OpenResident(argv[1]);
    const auto cfg = yah::core::Qwen35Config::FromGguf(gguf);
    const auto tconfig = yah::core::TokenizerConfig::FromGguf(gguf);
    const auto tok = yah::core::Tokenizer::FromGguf(gguf, tconfig);
    const std::vector<std::uint32_t> prompt = tok.Encode(
        "The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined movable "
        "metal type, oil-based ink and a wooden press. Within fifty years");
    LoomDevice gpu;
    LoomWeights weights(gpu, gguf);
    LoomDecoder dec(gpu, gguf, cfg, argv[2], weights);
    const std::uint32_t T = dec.verify_tokens(), n = static_cast<std::uint32_t>(prompt.size()), V = LoomDecoder::kVocab;
    if (!T) throw LoomError("the decode set has no verify GEMVs (emit_decode.py VERIFY)");

    // reference: plain steps, the logits of each kept token
    dec.Bind(dec.OwnState());
    dec.SetTokens(prompt.data(), n, 0);
    std::vector<float> ref_logits(std::size_t{gen} * V);
    for (std::uint32_t pos = 0; pos + 1 < n + gen; ++pos) {
      dec.Step(pos, n);
      if (pos + 1 >= n) {
        gpu.Synchronize();
        dec.CopyLogits(ref_logits.data() + std::size_t{pos + 1 - n} * V);
      }
    }
    gpu.Synchronize();
    const std::vector<std::uint32_t> ref = dec.Tokens(n, n + gen);

    // speculative: the prompt by steps, then verify rounds from its last token
    dec.Bind(dec.OwnState());
    dec.SetTokens(prompt.data(), n, 0);
    for (std::uint32_t pos = 0; pos + 1 < n; ++pos) dec.Step(pos, n);
    std::vector<std::uint32_t> out;
    std::vector<float> lg(V);
    std::uint32_t pos = n - 1, round = 0, mismatched = 0, accepted = 0, counts[16] = {};
    while (out.size() < gen) {
      const std::uint32_t k = std::min<std::uint32_t>(T - 1, gen - 1 - static_cast<std::uint32_t>(out.size()));
      std::vector<std::uint32_t> draft(ref.begin() + out.size(), ref.begin() + out.size() + k);
      const std::uint32_t wrong = round++ % T;  // a wrong draft at this index (none if past the draft)
      if (wrong < k) draft[wrong] = (draft[wrong] + 1) % V;
      if (k) dec.SetTokens(draft.data(), k, pos + 1);
      dec.Verify(pos, k + 1);
      gpu.Synchronize();
      const std::vector<std::uint32_t> g = dec.VerifyResult(k + 1);
      std::uint32_t m = 0;
      while (m < k && draft[m] == g[m]) ++m;
      for (std::uint32_t i = 0; i <= m; ++i) {  // the kept tokens' logits against the reference's
        dec.CopyVerifyLogits(i, lg.data());
        const std::size_t at = out.size() + i;
        if (at < gen && std::memcmp(lg.data(), ref_logits.data() + at * V, std::size_t{V} * 4) != 0) {
          float worst = 0;
          for (std::uint32_t j = 0; j < V; ++j) worst = std::max(worst, std::fabs(lg[j] - ref_logits[at * V + j]));
          if (mismatched++ < 5) std::printf("  logits of token %zu differ (max |d| %g)\n", at, worst);
        }
      }
      dec.Accept(m);
      for (std::uint32_t i = 0; i < m; ++i) out.push_back(draft[i]);
      out.push_back(g[m]);
      dec.SetTokens(&g[m], 1, pos + m + 1);  // (the next round's first token)
      pos += m + 1;
      accepted += m, ++counts[m];
    }
    out.resize(gen);
    std::uint32_t same = 0;
    while (same < gen && out[same] == ref[same]) ++same;
    std::printf("verify T=%u: %u rounds, %u drafts accepted; accept counts", T, round, accepted);
    for (std::uint32_t i = 0; i < T; ++i) std::printf(" %u", counts[i]);
    std::printf("\n%s  output = reference for %u of %u tokens\n", same == gen ? "ok  " : "FAIL", same, gen);
    std::printf("%s  verified logits bit-identical (%u differ)\n", mismatched ? "FAIL" : "ok  ", mismatched);
    std::printf("reference: %s\n", tok.Decode(ref).c_str());
    if (timing) {
      const auto ms = [&](auto&& f) {
        std::vector<double> t;
        for (int r = 0; r < 9; ++r) {
          gpu.Synchronize();
          const auto t0 = std::chrono::steady_clock::now();
          f();
          gpu.Synchronize();
          t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(t.begin(), t.end());
        return t[4];
      };
      std::printf("step %.2f ms\n", ms([&] { dec.Step(pos, 0), ++pos; }));
      for (std::uint32_t k = 1; k <= T; ++k)
        std::printf("verify %u: %.2f ms\n", k, ms([&] {
                      dec.Verify(pos, k);
                      dec.Accept(k - 1);
                      pos += k;
                    }));
      if (dec.has_mtp()) {
        dec.MtpBegin();
        dec.MtpHidden(dec.MainHidden());
        std::uint32_t v = 0;
        std::printf("mtp draft %.2f ms\n", ms([&] { dec.Draft(pos + v, v), ++v; }));
      }
    }
    return same == gen && !mismatched ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "spec_check: %s\n", e.what());
    return 1;
  }
}
