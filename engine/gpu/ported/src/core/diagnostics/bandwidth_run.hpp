#ifndef GUFO_CORE_DIAGNOSTICS_BANDWIDTH_RUN_HPP_
#define GUFO_CORE_DIAGNOSTICS_BANDWIDTH_RUN_HPP_

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "src/core/diagnostics/bandwidth.h"

namespace gufo::diagnostics::detail {

// Construct after warmup. Each path runs for both the requested duration
// and minimum iteration count; allocation and warmup are outside the budget.
class BandwidthRun {
public:
  BandwidthRun(const BandwidthOptions& options, BandwidthPathResult& result)
      : options_(options), result_(result), started_(Clock::now()) {
    if (options.repetitions == 0 || options.working_set_bytes == 0)
      throw std::invalid_argument(
          "bandwidth requires nonzero repetitions and working set");
  }

  [[nodiscard]] bool ShouldContinue() const {
    result_.elapsed_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - started_)
            .count();
    return result_.repetitions < options_.repetitions ||
           result_.elapsed_ms < options_.duration_ms;
  }

  void Record(double seconds, double transferred_bytes) {
    result_.raw_repetitions_gbps.push_back(transferred_bytes / 1e9 /
                                           std::max(seconds, 1e-9));
    ++result_.repetitions;
    result_.elapsed_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - started_)
            .count();
  }

private:
  using Clock = std::chrono::steady_clock;
  const BandwidthOptions& options_;
  BandwidthPathResult& result_;
  Clock::time_point started_;
};

}  // namespace gufo::diagnostics::detail

#endif  // GUFO_CORE_DIAGNOSTICS_BANDWIDTH_RUN_HPP_
