// Gated DeltaNet (SSM) check for the ported prefill recurrence, at the model's
// shapes: qkv 10240 (q 16x128, k 16x128, v 48x128), 48 value heads, key and
// value dim 128, conv kernel 4. The reference is the from-definition causal
// conv, delta-rule recurrence, per-head RMSNorm and SiLU gate.
#define ENGINE_ENABLE_HIP 1
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "src/models/qwen/hip/ops/ssm.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

using gufo::test::DeviceBuffer;

namespace {

__global__ void FillValues(float* values, std::size_t count, float scale) {
  const std::size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < count) {
    values[i] = (static_cast<int>((i * 7919) % 2001) - 1000) * scale;
  }
}

// A trained ssm_a is negative (the kernel's own default is -0.05), which is
// what makes the recurrence decay. Random bytes can make it positive, and then
// the synthetic state grows without bound and both sides NaN, so bias it.
__global__ void BiasNegative(float* values, std::size_t count, float base,
                             float scale) {
  const std::size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < count) values[i] = base + values[i] * scale;
}

float Silu(float x) { return x / (1.0F + std::exp(-x)); }

}  // namespace

int main(int argc, char** argv) {
  const std::size_t batch = argc > 1 ? std::stoul(argv[1]) : 64;
  const std::uint32_t num_key_heads = 16, num_heads = 48;
  const std::uint32_t key_dim = 128, val_dim = 128;
  const std::size_t qkv_size = 2 * num_key_heads * key_dim + num_heads * val_dim;
  const std::size_t inner_size = num_heads * val_dim;

  DeviceBuffer<float> qkv(batch * qkv_size), conv_w(qkv_size * 4),
      conv_state(qkv_size * 4), conv_out(batch * qkv_size),
      alpha(batch * num_heads), beta(batch * num_heads), ssm_a(num_heads),
      ssm_dt(num_heads), ssm_norm(key_dim), gate(batch * inner_size),
      out(batch * inner_size);
  DeviceBuffer<float> deltanet_state(
      static_cast<std::size_t>(num_heads) * key_dim * val_dim);

  auto fill = [](DeviceBuffer<float>& buffer, float scale) {
    FillValues<<<static_cast<unsigned>((buffer.size() + 255) / 256), 256>>>(
        buffer.data(), buffer.size(), scale);
  };
  fill(qkv, 0.004F);
  fill(conv_w, 0.02F);
  fill(alpha, 0.01F);
  fill(beta, 0.01F);
  fill(ssm_a, 0.0005F);
  BiasNegative<<<1, 64>>>(ssm_a.data(), ssm_a.size(), -0.05F, 0.1F);
  fill(ssm_dt, 0.02F);
  fill(ssm_norm, 0.01F);
  fill(gate, 0.05F);
  HIP_CHECK(hipMemset(conv_state.data(), 0, conv_state.size() * sizeof(float)));
  HIP_CHECK(hipMemset(deltanet_state.data(), 0,
                      deltanet_state.size() * sizeof(float)));
  HIP_CHECK(hipMemset(out.data(), 0, out.size() * sizeof(float)));

  gufo::hip::LaunchBatchedSSMConvRecurrence(
      qkv.data(), conv_w.data(), conv_state.data(), conv_out.data(),
      deltanet_state.data(), alpha.data(), beta.data(), ssm_a.data(),
      ssm_dt.data(), ssm_norm.data(), gate.data(), out.data(), /*layer_idx=*/0,
      batch, qkv_size, num_key_heads, num_heads, key_dim, val_dim, nullptr,
      gufo::hip::QwenRecurrentStateStorage::kFp32);
  HIP_CHECK(hipDeviceSynchronize());

  const auto qkv_host = qkv.CopyToHost();
  const auto conv_w_host = conv_w.CopyToHost();
  const auto alpha_host = alpha.CopyToHost();
  const auto beta_host = beta.CopyToHost();
  const auto ssm_a_host = ssm_a.CopyToHost();
  const auto ssm_dt_host = ssm_dt.CopyToHost();
  const auto ssm_norm_host = ssm_norm.CopyToHost();
  const auto gate_host = gate.CopyToHost();
  const auto out_host = out.CopyToHost();

  // Reference: causal conv with a zero initial history.
  std::vector<float> conv_ref(batch * qkv_size);
  for (std::size_t t = 0; t < batch; ++t) {
    for (std::size_t c = 0; c < qkv_size; ++c) {
      double dot = 0.0;
      for (int k = 0; k < 4; ++k) {
        const long j = static_cast<long>(t) - 3 + k;
        if (j >= 0) {
          dot += static_cast<double>(conv_w_host[c * 4 + k]) *
                 qkv_host[static_cast<std::size_t>(j) * qkv_size + c];
        }
      }
      conv_ref[t * qkv_size + c] = Silu(static_cast<float>(dot));
    }
  }

  // Reference: delta-rule recurrence, then the per-head norm and gate.
  std::vector<float> state(static_cast<std::size_t>(num_heads) * key_dim * val_dim,
                           0.0F);
  std::vector<float> out_ref(batch * inner_size, 0.0F);
  std::vector<double> norm_k(key_dim), norm_q(key_dim), u(val_dim), p(val_dim),
      d(val_dim);
  for (std::size_t t = 0; t < batch; ++t) {
    for (std::uint32_t h = 0; h < num_heads; ++h) {
      const std::uint32_t kh = h % num_key_heads;
      // The qkv layout is q first, then k, then v. The kernel's own variable
      // names are the authority here: q_h reads the first section.
      const float* q_h = conv_ref.data() + t * qkv_size + kh * key_dim;
      const float* k_h =
          conv_ref.data() + t * qkv_size + num_key_heads * key_dim + kh * key_dim;
      const float* v_h = conv_ref.data() + t * qkv_size +
                         2 * num_key_heads * key_dim + h * val_dim;
      double k_sq = 0.0, q_sq = 0.0;
      for (std::uint32_t i = 0; i < key_dim; ++i) {
        k_sq += static_cast<double>(k_h[i]) * k_h[i];
        q_sq += static_cast<double>(q_h[i]) * q_h[i];
      }
      const double inv_k = 1.0 / std::sqrt(k_sq + 1e-6);
      const double q_scale =
          (1.0 / std::sqrt(static_cast<double>(key_dim))) /
          std::sqrt(q_sq + 1e-6);
      double kq = 0.0;
      for (std::uint32_t i = 0; i < key_dim; ++i) {
        norm_k[i] = k_h[i] * inv_k;
        norm_q[i] = q_h[i] * q_scale;
        kq += norm_k[i] * norm_q[i];
      }
      const double alpha_biased = alpha_host[t * num_heads + h] + ssm_dt_host[h];
      const double softplus = alpha_biased > 20.0
                                  ? alpha_biased
                                  : std::log1p(std::exp(alpha_biased));
      const double alpha_h = std::exp(softplus * ssm_a_host[h]);
      const double beta_h = 1.0 / (1.0 + std::exp(-beta_host[t * num_heads + h]));
      float* s_head = state.data() +
                      (static_cast<std::size_t>(h) * key_dim * val_dim);
      for (std::uint32_t j = 0; j < val_dim; ++j) {
        float* row = s_head + static_cast<std::size_t>(j) * key_dim;
        double uj = 0.0, pj = 0.0;
        for (std::uint32_t i = 0; i < key_dim; ++i) {
          row[i] = static_cast<float>(row[i] * alpha_h);
          uj += static_cast<double>(row[i]) * norm_k[i];
          pj += static_cast<double>(row[i]) * norm_q[i];
        }
        u[j] = uj;
        p[j] = pj;
        d[j] = (static_cast<double>(v_h[j]) - uj) * beta_h;
        out_ref[t * inner_size + h * val_dim + j] =
            static_cast<float>(pj + d[j] * kq);
        for (std::uint32_t i = 0; i < key_dim; ++i) {
          row[i] = static_cast<float>(row[i] + d[j] * norm_k[i]);
        }
      }
      // Per-head RMSNorm then the SiLU gate.
      double mean_sq = 0.0;
      for (std::uint32_t j = 0; j < val_dim; ++j) {
        const double value = out_ref[t * inner_size + h * val_dim + j];
        mean_sq += value * value;
      }
      const double inv_rms = 1.0 / std::sqrt(mean_sq / val_dim + 1e-6);
      for (std::uint32_t j = 0; j < val_dim; ++j) {
        const double value = out_ref[t * inner_size + h * val_dim + j];
        const double g = gate_host[t * inner_size + h * val_dim + j];
        const double sig = 1.0 / (1.0 + std::exp(-g));
        out_ref[t * inner_size + h * val_dim + j] = static_cast<float>(
            value * inv_rms * ssm_norm_host[j] * (g * sig));
      }
    }
  }

  double max_abs = 0.0, sum_sq = 0.0, ref_sq = 0.0;
  for (std::size_t i = 0; i < out_ref.size(); ++i) {
    const double diff =
        std::fabs(static_cast<double>(out_host[i]) - out_ref[i]);
    max_abs = std::max(max_abs, diff);
    sum_sq += diff * diff;
    ref_sq += static_cast<double>(out_ref[i]) * out_ref[i];
  }
  const double rms = std::sqrt(sum_sq / static_cast<double>(out_ref.size()));
  const double ref_rms =
      std::sqrt(ref_sq / static_cast<double>(out_ref.size()));
  const double rel = ref_rms > 0.0 ? rms / ref_rms : 0.0;
  const bool finite = std::isfinite(max_abs) && std::isfinite(rel) && rel > 0.0;
  std::printf("ssm batch=%zu qkv=%zu heads=%u key_dim=%u val_dim=%u\n", batch,
              qkv_size, num_heads, key_dim, val_dim);
  std::printf("max_abs=%.6g rms=%.6g ref_rms=%.6g rel_rms=%.3g\n", max_abs, rms,
              ref_rms, rel);
  const bool ok = finite && rel < 5e-4;
  std::printf("%s\n", ok ? "SSM CHECK PASS" : "SSM CHECK FAIL");
  return ok ? 0 : 1;
}
