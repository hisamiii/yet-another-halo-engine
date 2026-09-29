#ifndef GUFO_CORE_HIP_DETAIL_GEMV_DISPATCHER_HPP_
#define GUFO_CORE_HIP_DETAIL_GEMV_DISPATCHER_HPP_

#include <cstdint>
#include <string_view>

namespace gufo::hip::detail {

enum class GemvStrategy : std::uint8_t {
  kBaselineBlock = 0,
  kWave32SingleRow = 1,
};

[[nodiscard]] constexpr std::string_view GemvStrategyName(
    GemvStrategy strategy) noexcept {
  switch (strategy) {
    case GemvStrategy::kBaselineBlock:
      return "baseline_block";
    case GemvStrategy::kWave32SingleRow:
      return "wave32_single_row";
  }
  return "unknown";
}

}  // namespace gufo::hip::detail

#endif  // GUFO_CORE_HIP_DETAIL_GEMV_DISPATCHER_HPP_
