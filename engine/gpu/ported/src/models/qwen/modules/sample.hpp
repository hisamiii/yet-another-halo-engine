#ifndef GUFO_MODELS_QWEN_MODULES_SAMPLE_HPP_
#define GUFO_MODELS_QWEN_MODULES_SAMPLE_HPP_

#include <cstdint>
#include <span>

#include "src/models/qwen/modules/module_ctx.hpp"

namespace gufo::models::qwen {

/// Greedy argmax over a logit distribution (sampling-policy module seam).
std::uint32_t SampleForward(const CpuModuleContext& ctx,
                            std::span<const float> logits) noexcept;

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_MODULES_SAMPLE_HPP_
