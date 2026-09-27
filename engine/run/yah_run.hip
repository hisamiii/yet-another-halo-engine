// yah-run: the engine's end-to-end text entry point. Loads a GGUF mapping,
// tokenizes a prompt, runs one prefill chunk through the GPU layer loop, and
// prints the argmax token so the run can be compared token-for-token against a
// reference. Timing is reported separately from correctness; the argmax is the
// gate.
#define ENGINE_ENABLE_HIP 1
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "model/forward.hip"
#include "model/weights.hpp"

namespace {

std::vector<std::uint32_t> ParseIds(const std::string& text) {
  std::vector<std::uint32_t> ids;
  std::size_t i = 0;
  while (i < text.size()) {
    while (i < text.size() && (text[i] == ' ' || text[i] == ',')) ++i;
    std::size_t start = i;
    while (i < text.size() && text[i] != ' ' && text[i] != ',') ++i;
    if (i > start) {
      ids.push_back(static_cast<std::uint32_t>(std::stoul(text.substr(start, i - start))));
    }
  }
  return ids;
}

}  // namespace

int main(int argc, char** argv) {
  // Diagnostics must survive a GPU hang, which kills the process before any
  // buffered stream is flushed, so stderr is unbuffered from the first line.
  std::setvbuf(stderr, nullptr, _IONBF, 0);
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: yah-run <model.gguf> [--text TEXT] [--ids \"1 2\"] "
                 "[--repeat N] [--chunk N] [--gen N] [--kv f16|q8|q4] [--max-context N]\n"
                 "       [--dump-logits F] [--dump-all-logits F]\n");
    return 2;
  }
  const std::string model = argv[1];
  std::string text;
  std::string ids_arg;
  std::string dump;
  std::string all_logits;
  std::uint32_t max_context = 4096;
  std::uint32_t repeat = 1;
  std::uint32_t batch = 0;
  std::uint32_t chunk = 0;
  std::uint32_t gen = 0;
  yah::kv::KvStorage kv = yah::kv::KvStorage::kF16;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--text") text = next();
    else if (a == "--ids") ids_arg = next();
    else if (a == "--dump-logits") dump = next();
    else if (a == "--dump-all-logits") all_logits = next();
    else if (a == "--batch") batch = static_cast<std::uint32_t>(std::stoul(next()));
    else if (a == "--max-context") max_context = static_cast<std::uint32_t>(std::stoul(next()));
    else if (a == "--repeat") repeat = static_cast<std::uint32_t>(std::stoul(next()));
    else if (a == "--chunk") chunk = static_cast<std::uint32_t>(std::stoul(next()));
    else if (a == "--gen") gen = static_cast<std::uint32_t>(std::stoul(next()));
    else if (a == "--kv") {
      const std::string value = next();
      if (value == "f16") kv = yah::kv::KvStorage::kF16;
      else if (value == "q8") kv = yah::kv::KvStorage::kQ8;
      else if (value == "q4") kv = yah::kv::KvStorage::kQ4;
      else throw std::runtime_error("yah-run: --kv must be f16, q8 or q4");
    }
    else if (!text.empty()) text += " " + a;
    else text = a;
  }
  try {
    auto gguf = yah::core::Gguf::Open(model);
    const auto config = yah::core::Qwen35Config::FromGguf(gguf);
    auto weights = yah::model::Qwen35Weights::FromGguf(gguf, config);
    // The file mapping is not GPU-visible: a kernel that dereferences an
    // unregistered host mapping stalls the ring. Register a device-visible
    // copy and repoint the table at it before any weight is read.
    const yah::model::HipWeightRegion region(gguf);
    weights.Rebase(region.delta(gguf));
    std::fprintf(stderr,
                 "yah-run: registered %.2f GiB of weights for the GPU\n",
                 static_cast<double>(gguf.tensor_data_size()) / 1073741824.0);

    const auto tconfig = yah::core::TokenizerConfig::FromGguf(gguf);
    const auto tokenizer = yah::core::Tokenizer::FromGguf(gguf, tconfig);
    std::vector<std::uint32_t> tokens;
    if (!ids_arg.empty()) {
      tokens = ParseIds(ids_arg);
    } else {
      tokens = tokenizer.Encode(text);
    }
    if (tokens.empty()) {
      std::fprintf(stderr, "yah-run: empty prompt\n");
      return 1;
    }
    if (batch != 0 && batch != tokens.size()) {
      throw std::runtime_error("yah-run: --batch does not match the prompt");
    }
    batch = static_cast<std::uint32_t>(tokens.size());
    std::fprintf(stderr,
                 "yah-run: layers=%u hidden=%u ffn=%u heads=%u kv_heads=%u "
                 "head_dim=%u vocab=%llu batch=%u context=%u\n",
                 config.main_block_count(), config.embedding_length,
                 config.feed_forward_length, config.head_count,
                 config.head_count_kv, config.key_length,
                 static_cast<unsigned long long>(weights.output.rows()), batch,
                 max_context);
    std::fprintf(stderr, "tokens=%zu\n", tokens.size());
    if (tokens.size() <= 32) {
      for (std::size_t i = 0; i < tokens.size(); ++i) {
        std::fprintf(stderr, "%u%s", tokens[i],
                     i + 1 == tokens.size() ? "\n" : " ");
      }
    }

    // The chunk size bounds one prefill call; the whole prompt is split into
    // chunks, each continuing from the KV and recurrent state the previous one
    // left, exactly as the reference processes a long prompt.
    const std::uint32_t max_batch =
        chunk == 0 ? static_cast<std::uint32_t>(tokens.size())
                   : std::min(chunk, static_cast<std::uint32_t>(tokens.size()));
    yah::model::Forward forward(config, weights, max_batch, max_context, kv);
    std::uint32_t argmax = 0;
    std::uint32_t last_batch = 0;
    double best_ms = 1e30;
    for (std::uint32_t r = 0; r < repeat; ++r) {
      forward.Reset();
      const auto start = std::chrono::steady_clock::now();
      for (std::size_t off = 0; off < tokens.size(); off += max_batch) {
        const std::uint32_t n = static_cast<std::uint32_t>(
            std::min<std::size_t>(max_batch, tokens.size() - off));
        argmax = forward.Prefill(tokens.data() + off, n,
                                 static_cast<std::uint32_t>(off));
        last_batch = n;
      }
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
      if (ms < best_ms) best_ms = ms;
      std::fprintf(stderr, "run %u: %.3f ms\n", r, ms);
    }
    std::printf("argmax=%u\n", argmax);
    std::printf("best_ms=%.3f\n", best_ms);
    if (!all_logits.empty()) {
      const std::size_t count =
          static_cast<std::size_t>(last_batch) * forward.vocab_size();
      std::vector<float> every(count);
      forward.AllLogits(last_batch, every.data());
      std::ofstream out(all_logits, std::ios::binary);
      out.write(reinterpret_cast<const char*>(every.data()),
                static_cast<std::streamsize>(count * sizeof(float)));
      std::fprintf(stderr, "wrote %u x %u all-position logits to %s\n",
                   last_batch, forward.vocab_size(), all_logits.c_str());
    }

    // Generation runs the same kernel chain one token at a time. A dedicated
    // GEMV decode path is a later optimization; the goal of this loop is a
    // working end-to-end text path, and the single-token forward is the
    // chunked prefill with a chunk of one, already exercised above.
    if (gen != 0) {
      std::vector<std::uint32_t> generated;
      std::uint32_t next = argmax;
      std::uint32_t pos = static_cast<std::uint32_t>(tokens.size());
      const auto gen_start = std::chrono::steady_clock::now();
      for (std::uint32_t i = 0; i < gen; ++i) {
        generated.push_back(next);
        if (next == tconfig.eos_id) break;
        if (pos + 1 > max_context) break;
        next = forward.Decode(next, pos);
        ++pos;
      }
      const double gen_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - gen_start)
                                .count();
      std::printf("generated_ids=");
      for (std::size_t i = 0; i < generated.size(); ++i) {
        std::printf("%u%s", generated[i],
                    i + 1 == generated.size() ? "" : " ");
      }
      std::printf("\n");
      std::printf("generated_text=%s\n", tokenizer.Decode(generated).c_str());
      std::fprintf(stderr, "generated %zu tokens in %.3f ms (%.2f tok/s)\n",
                   generated.size(), gen_ms,
                   generated.empty() ? 0.0
                                     : generated.size() * 1000.0 / gen_ms);
    }
    if (!dump.empty()) {
      std::vector<float> logits(forward.vocab_size());
      forward.CopyLogits(logits.data());
      std::ofstream out(dump, std::ios::binary);
      out.write(reinterpret_cast<const char*>(logits.data()),
                static_cast<std::streamsize>(logits.size() * sizeof(float)));
      std::fprintf(stderr, "wrote %zu logits to %s\n", logits.size(),
                   dump.c_str());
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "yah-run: %s\n", error.what());
    return 1;
  }
}
