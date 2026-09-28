// Isolated bench for the ported Gated DeltaNet path, at the model's shapes:
// qkv 10240 (q 16x128, k 16x128, v 48x128), 48 value heads, key and value dim
// 128, conv kernel 4.
//
// The point is the falsifier for the SSM cost model: if the recurrence is bound
// by sequential token depth, the achieved TFLOP/s must be flat in T, because
// both the FLOPs and the sequential depth grow with T and the launch width does
// not. If instead the rate rises with T, the cost is ramp-up and tail and
// chunking the token axis buys nothing.
//
// One arm per process, so a paired harness can put a 15 s gap before each timed
// launch, matching docs/methodology.md. Modes:
//   row  - LaunchBatchedSSMConvRecurrenceRowSplit, the path prefill takes
//          (grid dim3(2, 48) = 96 blocks of 256 threads at these shapes)
//   base - LaunchBatchedSSMConvRecurrence, the unfused cross-check that
//          YAH_SSM_BASELINE selects, launched as <<<48, 256>>>
//   cmp  - run both, reset state between, and compare outputs elementwise
//
// Usage: ssm_bench <tokens> <row|base|cmp> [iters]
#define ENGINE_ENABLE_HIP 1
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// A trained ssm_a is negative, which is what makes the recurrence decay; a
// positive one makes the synthetic state grow without bound.
__global__ void BiasNegative(float* values, std::size_t count, float base,
                             float scale) {
  const std::size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < count) values[i] = base + values[i] * scale;
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t batch = argc > 1 ? std::stoul(argv[1]) : 2048;
  const std::string mode = argc > 2 ? argv[2] : "row";
  const unsigned iters = argc > 3 ? std::stoul(argv[3]) : 1;

  const std::uint32_t num_key_heads = 16, num_heads = 48;
  const std::uint32_t key_dim = 128, val_dim = 128;
  const std::size_t qkv_size = 2 * num_key_heads * key_dim + num_heads * val_dim;
  const std::size_t inner_size = static_cast<std::size_t>(num_heads) * val_dim;

  DeviceBuffer<float> qkv(batch * qkv_size), conv_w(qkv_size * 4),
      conv_state(qkv_size * 4), conv_out(batch * qkv_size),
      alpha(batch * num_heads), beta(batch * num_heads), ssm_a(num_heads),
      ssm_dt(num_heads), ssm_norm(key_dim), gate(batch * inner_size),
      out_base(batch * inner_size), out_row(batch * inner_size),
      kq_scales(batch * num_key_heads * 3),
      alpha_beta(batch * static_cast<std::size_t>(num_heads) * 2);
  DeviceBuffer<__half> out_row_f16(batch * inner_size);
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
  HIP_CHECK(hipDeviceSynchronize());

  const float* ssm_a_p = ssm_a.data();
  const float* ssm_dt_p = ssm_dt.data();
  const float* ssm_norm_p = ssm_norm.data();
  const float* gate_p = gate.data();
  const float* conv_w_p = conv_w.data();
  float* conv_state_p = conv_state.data();
  float* conv_out_p = conv_out.data();
  float* deltanet_p = deltanet_state.data();
  float* qkv_p = qkv.data();
  float* alpha_p = alpha.data();
  float* beta_p = beta.data();
  float* out_base_p = out_base.data();
  float* out_row_p = out_row.data();
  float* kq_p = kq_scales.data();
  float* ab_p = alpha_beta.data();
  __half* out_f16_p = out_row_f16.data();

  auto run_row = [&] {
    gufo::hip::LaunchBatchedSSMConvRecurrenceRowSplit(
        qkv_p, conv_w_p, conv_state_p, conv_out_p, deltanet_p, alpha_p, beta_p,
        ssm_a_p, ssm_dt_p, ssm_norm_p, gate_p, out_row_p, /*q8_out=*/nullptr,
        kq_p, ab_p, /*layer_idx=*/0, batch, qkv_size, num_key_heads, num_heads,
        key_dim, val_dim, nullptr, gufo::hip::QwenRecurrentStateStorage::kFp32,
        /*fp16_out=*/out_f16_p);
  };
  auto run_base = [&] {
    gufo::hip::LaunchBatchedSSMConvRecurrence(
        qkv_p, conv_w_p, conv_state_p, conv_out_p, deltanet_p, alpha_p, beta_p,
        ssm_a_p, ssm_dt_p, ssm_norm_p, gate_p, out_base_p, /*layer_idx=*/0,
        batch, qkv_size, num_key_heads, num_heads, key_dim, val_dim, nullptr,
        gufo::hip::QwenRecurrentStateStorage::kFp32);
  };
  auto reset_state = [&] {
    HIP_CHECK(hipMemset(conv_state_p, 0, conv_state.size() * sizeof(float)));
    HIP_CHECK(
        hipMemset(deltanet_p, 0, deltanet_state.size() * sizeof(float)));
  };

  // The essential recurrence work: three rank-128 operations per head per token.
  const double flops =
      3.0 * static_cast<double>(batch) * num_heads * key_dim * val_dim * 2.0;

  if (mode == "cmp") {
    reset_state();
    run_base();
    HIP_CHECK(hipDeviceSynchronize());
    reset_state();
    run_row();
    HIP_CHECK(hipDeviceSynchronize());
    const auto a = out_base.CopyToHost();
    const auto b = out_row_f16.CopyToHost();
    double max_abs = 0.0, sum_sq = 0.0, ref_sq = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double d = std::fabs(static_cast<double>(a[i]) -
                                 static_cast<double>(b[i]));
      max_abs = std::max(max_abs, d);
      sum_sq += d * d;
      ref_sq += static_cast<double>(a[i]) * a[i];
    }
    const double ref_rms = std::sqrt(ref_sq / static_cast<double>(a.size()));
    const double rel = ref_rms > 0.0 ? std::sqrt(sum_sq / a.size()) / ref_rms : 0.0;
    std::printf("cmp T=%zu max_abs=%.6g rel_rms=%.3g %s\n", batch, max_abs, rel,
                (std::isfinite(rel) && rel < 5e-3) ? "AGREE" : "MISMATCH");
    return 0;
  }

  const bool use_base = (mode == "base");
  auto launch = [&] {
    if (use_base) {
      run_base();
    } else {
      run_row();
    }
  };
  reset_state();
  for (int i = 0; i < 2; ++i) launch();
  HIP_CHECK(hipDeviceSynchronize());
  const auto t0 = std::chrono::steady_clock::now();
  for (unsigned i = 0; i < iters; ++i) launch();
  HIP_CHECK(hipDeviceSynchronize());
  const auto t1 = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  std::printf("ssm mode=%s T=%zu ms=%.6f tflops=%.3f blocks=%s\n", mode.c_str(),
              batch, ms, flops / (ms * 1e9),
              (mode == "base") ? "48" : "96");
  return 0;
}
