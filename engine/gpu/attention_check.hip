// Causal grouped-query attention check for the ported prefill kernel, at the
// model's own shapes (24 query heads, 4 KV heads, head_dim 256) and a single
// chunk starting at position 0. The reference is the definition: softmax over
// the causal prefix with 1/sqrt(head_dim), then a sigmoid output gate.
#define ENGINE_ENABLE_HIP 1
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kernels/attention_batched.hip"
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
  const std::size_t batch = argc > 1 ? std::stoul(argv[1]) : 256;
  const std::uint32_t heads = 24, kv_heads = 4, head_dim = 256;
  const std::uint32_t q_width = heads * head_dim;      // 6144
  const std::uint32_t kv_width = kv_heads * head_dim;  // 1024
  const std::uint32_t max_context = static_cast<std::uint32_t>(batch);

  DeviceBuffer<float> q(batch * q_width), k(batch * kv_width),
      v(batch * kv_width), gate(batch * q_width), out(batch * q_width);
  const std::size_t cache_elements =
      static_cast<std::size_t>(kv_heads) * max_context * head_dim;
  DeviceBuffer<float> k_cache(cache_elements), v_cache(cache_elements);
  DeviceBuffer<short> k_cache_f16(cache_elements), v_cache_f16(cache_elements);

  auto fill = [](DeviceBuffer<float>& buffer, float scale) {
    FillValues<<<static_cast<unsigned>((buffer.size() + 255) / 256), 256>>>(
        buffer.data(), buffer.size(), scale);
  };
  fill(q, 0.02F);
  fill(k, 0.02F);
  fill(v, 0.02F);
  fill(gate, 0.05F);
  HIP_CHECK(hipMemset(out.data(), 0, out.size() * sizeof(float)));

  gufo::hip::LaunchBatchedAttention(
      q.data(), k.data(), v.data(), gate.data(), k_cache.data(), v_cache.data(),
      k_cache_f16.data(), v_cache_f16.data(), out.data(), /*layer_idx=*/0,
      /*start_pos=*/0, batch, max_context, heads, kv_heads, head_dim, nullptr,
      /*skip_kv_write=*/false);

  const auto q_host = q.CopyToHost();
  const auto k_host = k.CopyToHost();
  const auto v_host = v.CopyToHost();
  const auto gate_host = gate.CopyToHost();
  const auto out_host = out.CopyToHost();

  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  const std::uint32_t heads_per_kv = heads / kv_heads;
  std::vector<float> scores(batch);
  std::vector<float> reference(out_host.size(), 0.0F);
  for (std::size_t t = 0; t < batch; ++t) {
    for (std::uint32_t h = 0; h < heads; ++h) {
      const std::uint32_t kv_head = h / heads_per_kv;
      const float* q_row = q_host.data() + t * q_width + h * head_dim;
      float max_score = -1e30F;
      for (std::size_t j = 0; j <= t; ++j) {
        const float* k_row = k_host.data() + j * kv_width + kv_head * head_dim;
        double dot = 0.0;
        for (std::uint32_t d = 0; d < head_dim; ++d) {
          dot += static_cast<double>(q_row[d]) * k_row[d];
        }
        scores[j] = static_cast<float>(dot) * scale;
        if (scores[j] > max_score) max_score = scores[j];
      }
      double sum = 0.0;
      for (std::size_t j = 0; j <= t; ++j) {
        scores[j] = std::exp(scores[j] - max_score);
        sum += scores[j];
      }
      float* out_row = reference.data() + t * q_width + h * head_dim;
      for (std::size_t j = 0; j <= t; ++j) {
        const float weight = static_cast<float>(scores[j] / sum);
        const float* v_row =
            v_host.data() + j * kv_width + kv_head * head_dim;
        for (std::uint32_t d = 0; d < head_dim; ++d) {
          out_row[d] += weight * v_row[d];
        }
      }
      // The gate is per element, not per head: the kernel indexes it with the
      // same [t, head, d] layout as the output.
      for (std::uint32_t d = 0; d < head_dim; ++d) {
        const float g = gate_host[t * q_width + h * head_dim + d];
        out_row[d] *= 1.0F / (1.0F + std::exp(-g));
      }
    }
  }

  double max_abs = 0.0, sum_sq = 0.0, ref_sq = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double diff =
        std::fabs(static_cast<double>(out_host[i]) - reference[i]);
    max_abs = std::max(max_abs, diff);
    sum_sq += diff * diff;
    ref_sq += static_cast<double>(reference[i]) * reference[i];
  }
  const double rms = std::sqrt(sum_sq / static_cast<double>(reference.size()));
  const double ref_rms =
      std::sqrt(ref_sq / static_cast<double>(reference.size()));
  const double rel = ref_rms > 0.0 ? rms / ref_rms : 0.0;
  const bool finite = std::isfinite(max_abs) && std::isfinite(rel) && rel > 0.0;
  std::printf("attention batch=%zu heads=%u kv_heads=%u head_dim=%u\n", batch,
              heads, kv_heads, head_dim);
  std::printf("max_abs=%.6g rms=%.6g ref_rms=%.6g rel_rms=%.3g\n", max_abs, rms,
              ref_rms, rel);
  if (batch <= 4) {
    for (std::size_t i = 0; i < 8; ++i) {
      std::printf("  [%zu] device=%.6g ref=%.6g\n", i, out_host[i],
                  reference[i]);
    }
  }
  const bool ok = finite && rel < 1e-4;
  std::printf("%s\n", ok ? "ATTENTION CHECK PASS" : "ATTENTION CHECK FAIL");
  return ok ? 0 : 1;
}
