// TextGenerator: what the server needs from an engine. model::Engine is the GPU implementation.
#ifndef YAH_MODEL_GENERATOR_HPP_
#define YAH_MODEL_GENERATOR_HPP_

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "core/tokenizer.hpp"

namespace yah::model {

// Penalties on the logits, then softmax at temperature cut by top-k, top-p and min-p in that order (vLLM's and
// transformers' order), then an exact draw. All on the GPU (gen_decode_misc gen_penalty, gen_sample).
struct SamplingParams {
  float temperature = 0.0f;  // 0: greedy (argmax on the GPU)
  float top_p = 1.0f;
  std::uint32_t top_k = 0;  // 0: off
  float min_p = 0.0f;       // keep tokens with p >= min_p * p_max; 0: off
  // Over the last penalty_last_n tokens (prompt and output): logit -= presence (once) + frequency * count, and
  // repetition divides a positive logit and multiplies a negative one (1: off). They apply to greedy decoding too.
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  float repetition_penalty = 1.0f;
  std::uint32_t penalty_last_n = 64;
  std::uint64_t seed = 0;  // 0: seed from the clock
  [[nodiscard]] bool penalized() const {
    return penalty_last_n > 0 && (presence_penalty != 0.0f || frequency_penalty != 0.0f || repetition_penalty != 1.0f);
  }
};

struct GenerateParams {
  std::uint32_t max_tokens = 1024;
  SamplingParams sampling;
  std::vector<core::TokenId> stop_ids;  // generation ends at the first of these; it is not reported
};

struct GenerateResult {
  std::uint32_t prompt_tokens = 0;
  std::uint32_t cached_tokens = 0;     // prompt tokens whose KV and state came from the previous request
  std::uint32_t generated_tokens = 0;  // tokens passed to on_token
  std::string finish_reason;           // "stop" (stop id or on_token returned false) or "length"
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  std::uint32_t drafted = 0, accepted = 0;  // speculative decoding: draft tokens verified, and those kept
};

class TextGenerator {
 public:
  virtual ~TextGenerator() = default;
  [[nodiscard]] virtual const core::Tokenizer& tokenizer() const = 0;
  // Largest prompt + generated token count one request can use.
  [[nodiscard]] virtual std::uint32_t context() const = 0;
  [[nodiscard]] virtual std::string model_name() const = 0;
  // Runs the prompt, then calls on_token once per generated token, in order, as soon as it is known.
  // Return false from on_token to stop early. Not reentrant: callers serialize requests.
  virtual GenerateResult Generate(const std::vector<core::TokenId>& prompt, const GenerateParams& params,
                                  const std::function<bool(core::TokenId)>& on_token) = 0;
};

}  // namespace yah::model

#endif  // YAH_MODEL_GENERATOR_HPP_
