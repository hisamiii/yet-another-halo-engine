// Attention front-end check: per-head QK RMSNorm then RoPE, at the model's
// shapes (24 q heads, 4 kv heads, head_dim 256, rotary_dim 64, theta 1e7).
#define ENGINE_ENABLE_HIP 1
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/token.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

using gufo::test::DeviceBuffer;

namespace {

__global__ void FillValues(float* values, std::size_t count, float scale) {
  const std::size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < count) {
    values[i] = (static_cast<int>((i * 7919) % 2001) - 1000) * scale;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t batch = argc > 1 ? std::stoul(argv[1]) : 64;
  const std::uint32_t heads = 24, kv_heads = 4, head_dim = 256, rotary_dim = 64;
  const float theta = 10000000.0F, eps = 1e-6F;
  const std::uint32_t start_pos = 0;

  DeviceBuffer<float> q(batch * heads * head_dim), k(batch * kv_heads * head_dim),
      q_norm(head_dim), k_norm(head_dim);
  auto fill = [](DeviceBuffer<float>& buffer, float scale) {
    FillValues<<<static_cast<unsigned>((buffer.size() + 255) / 256), 256>>>(
        buffer.data(), buffer.size(), scale);
  };
  fill(q, 0.02F);
  fill(k, 0.02F);
  fill(q_norm, 0.01F);
  fill(k_norm, 0.01F);

  gufo::hip::LaunchBatchedPerHeadRMSNorm(q.data(), q_norm.data(), q.data(),
                                         batch, heads, head_dim, eps, nullptr);
  gufo::hip::LaunchBatchedPerHeadRMSNorm(k.data(), k_norm.data(), k.data(),
                                         batch, kv_heads, head_dim, eps,
                                         nullptr);
  gufo::hip::LaunchBatchedRoPE(q.data(), k.data(), batch, heads, kv_heads,
                               head_dim, rotary_dim, start_pos, theta, nullptr,
                               nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  const auto q_host = q.CopyToHost();
  const auto k_host = k.CopyToHost();

  // Rebuild the inputs independently and run the reference.
  DeviceBuffer<float> q0(batch * heads * head_dim), k0(batch * kv_heads * head_dim);
  fill(q0, 0.02F);
  fill(k0, 0.02F);
  const auto q0_host = q0.CopyToHost();
  const auto k0_host = k0.CopyToHost();
  const auto q_norm_host = q_norm.CopyToHost();
  const auto k_norm_host = k_norm.CopyToHost();

  auto per_head_norm = [&](std::vector<float> x, std::size_t num_heads,
                           const std::vector<float>& weight) {
    for (std::size_t b = 0; b < batch; ++b) {
      for (std::size_t h = 0; h < num_heads; ++h) {
        float* row = x.data() + (b * num_heads + h) * head_dim;
        double sum = 0.0;
        for (std::uint32_t d = 0; d < head_dim; ++d) {
          sum += static_cast<double>(row[d]) * row[d];
        }
        const double inv = 1.0 / std::sqrt(sum / head_dim + eps);
        for (std::uint32_t d = 0; d < head_dim; ++d) {
          row[d] = static_cast<float>(row[d] * inv * weight[d]);
        }
      }
    }
    return x;
  };
  auto rope = [&](std::vector<float> x, std::size_t num_heads) {
    const std::uint32_t half = rotary_dim / 2;
    for (std::size_t b = 0; b < batch; ++b) {
      for (std::size_t h = 0; h < num_heads; ++h) {
        float* row = x.data() + (b * num_heads + h) * head_dim;
        for (std::uint32_t i = 0; i < half; ++i) {
          const double freq =
              std::pow(static_cast<double>(theta),
                       -2.0 * i / static_cast<double>(rotary_dim));
          const double angle =
              static_cast<double>(start_pos + b) * freq;
          const double c = std::cos(angle), s = std::sin(angle);
          const double x0 = row[i], x1 = row[i + half];
          row[i] = static_cast<float>(x0 * c - x1 * s);
          row[i + half] = static_cast<float>(x0 * s + x1 * c);
        }
      }
    }
    return x;
  };

  const auto q_ref = rope(per_head_norm(q0_host, heads, q_norm_host), heads);
  const auto k_ref =
      rope(per_head_norm(k0_host, kv_heads, k_norm_host), kv_heads);

  auto compare = [](const std::vector<float>& got,
                    const std::vector<float>& want) {
    double max_abs = 0.0, sum_sq = 0.0, ref_sq = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      const double diff = std::fabs(static_cast<double>(got[i]) - want[i]);
      max_abs = std::max(max_abs, diff);
      sum_sq += diff * diff;
      ref_sq += static_cast<double>(want[i]) * want[i];
    }
    const double rel =
        ref_sq > 0.0 ? std::sqrt(sum_sq / ref_sq) : 0.0;
    return std::pair<double, double>{max_abs, rel};
  };
  const auto [q_max, q_rel] = compare(q_host, q_ref);
  const auto [k_max, k_rel] = compare(k_host, k_ref);

  std::printf("rope batch=%zu heads=%u kv_heads=%u head_dim=%u rotary=%u\n",
              batch, heads, kv_heads, head_dim, rotary_dim);
  std::printf("q max_abs=%.6g rel=%.3g   k max_abs=%.6g rel=%.3g\n", q_max,
              q_rel, k_max, k_rel);
  const bool finite = std::isfinite(q_rel) && std::isfinite(k_rel) &&
                      q_rel > 0.0 && k_rel > 0.0;
  const bool ok = finite && q_rel < 1e-5 && k_rel < 1e-5;
  std::printf("%s\n", ok ? "ROPE+NORM CHECK PASS" : "ROPE+NORM CHECK FAIL");
  return ok ? 0 : 1;
}
