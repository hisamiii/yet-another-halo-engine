// End-to-end check of one FFN block through the ported launchers: RMSNorm ->
// gate/up SwiGLU -> down with residual. The reference reuses the device's own
// FP16 intermediate activations and decodes the weights with the host
// DotProduct, so a mismatch is a kernel or launcher fault rather than a
// precision difference in the activation round trip.
#define ENGINE_ENABLE_HIP 1
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/kernels/prefill_fp16.hip"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

using gufo::core::GgmlType;
using gufo::test::DeviceBuffer;

namespace {

constexpr std::size_t kNoMin = std::numeric_limits<std::size_t>::max();

__global__ void FillWeights(unsigned char* data, std::size_t size,
                            std::size_t block_bytes, std::size_t d_offset,
                            std::size_t dmin_offset, int pattern) {
  const std::size_t block = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (block >= size / block_bytes) return;
  auto* bytes = data + (block * block_bytes);
  for (std::size_t i = 0; i < block_bytes; ++i) {
    bytes[i] = pattern == 0
                   ? static_cast<unsigned char>((block * 131 + i * 7) & 0xFF)
                   : static_cast<unsigned char>(pattern);
  }
  reinterpret_cast<__half*>(bytes + d_offset)[0] =
      __float2half(0.005F + (block % 97) * 1e-4F);
  if (dmin_offset != kNoMin) {
    reinterpret_cast<__half*>(bytes + dmin_offset)[0] =
        __float2half(0.004F + (block % 89) * 1e-4F);
  }
}

__global__ void FillValues(float* values, std::size_t count, float scale) {
  const std::size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < count) {
    values[i] = (static_cast<int>((i * 7919) % 2001) - 1000) * scale;
  }
}

std::pair<std::size_t, std::size_t> WeightScaleOffsets(GgmlType type) {
  switch (type) {
    case GgmlType::kQ4_K:
    case GgmlType::kQ5_K:
      return {0, 2};
    case GgmlType::kQ6_K:
      return {208, kNoMin};
    case GgmlType::kQ3_K:
      return {108, kNoMin};
    case GgmlType::kQ2_K:
      return {80, 82};
    default:
      return {0, kNoMin};
  }
}

float ReferenceDot(GgmlType type, const void* row, const float* x,
                   std::size_t k) {
  const std::span<const float> vec(x, k);
  switch (type) {
    case GgmlType::kQ4_K: return gufo::quant::DotProductQ4_K(row, vec, k);
    case GgmlType::kQ5_K: return gufo::quant::DotProductQ5_K(row, vec, k);
    case GgmlType::kQ6_K: return gufo::quant::DotProductQ6_K(row, vec, k);
    case GgmlType::kQ3_K: return gufo::quant::DotProductQ3_K(row, vec, k);
    case GgmlType::kQ2_K: return gufo::quant::DotProductQ2_K(row, vec, k);
    case GgmlType::kQ8_0: return gufo::quant::DotProductQ8_0(row, vec, k);
    case GgmlType::kIQ4_NL: return gufo::quant::DotProductIQ4_NL(row, vec, k);
    case GgmlType::kIQ4_XS: return gufo::quant::DotProductIQ4_XS(row, vec, k);
    case GgmlType::kIQ3_S: return gufo::quant::DotProductIQ3_S(row, vec, k);
    case GgmlType::kIQ3_XXS: return gufo::quant::DotProductIQ3_XXS(row, vec, k);
    case GgmlType::kIQ2_XS: return gufo::quant::DotProductIQ2_XS(row, vec, k);
    case GgmlType::kIQ2_S: return gufo::quant::DotProductIQ2_S(row, vec, k);
    case GgmlType::kIQ2_XXS: return gufo::quant::DotProductIQ2_XXS(row, vec, k);
    default: throw std::runtime_error("no host reference for this type");
  }
}

bool ParseType(const std::string& name, GgmlType* out) {
  const std::pair<const char*, GgmlType> kTypes[] = {
      {"q4k", GgmlType::kQ4_K},     {"q5k", GgmlType::kQ5_K},
      {"q6k", GgmlType::kQ6_K},     {"q3k", GgmlType::kQ3_K},
      {"q8_0", GgmlType::kQ8_0},    {"iq4nl", GgmlType::kIQ4_NL},
      {"iq4xs", GgmlType::kIQ4_XS}, {"iq3s", GgmlType::kIQ3_S},
      {"iq3xxs", GgmlType::kIQ3_XXS}};
  for (const auto& [label, type] : kTypes) {
    if (name == label) {
      *out = type;
      return true;
    }
  }
  return false;
}

float HalfToFloat(std::uint16_t bits) {
  __half h;
  std::memcpy(&h, &bits, sizeof(h));
  return __half2float(h);
}

void FillDevice(DeviceBuffer<unsigned char>& buffer, GgmlType type,
                std::size_t k) {
  const std::size_t block_bytes = gufo::quant::QuantizedRowBytes(
      type, gufo::quant::QuantizedBlockElements(type));
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(type);
  FillWeights<<<static_cast<unsigned>((buffer.size() / block_bytes + 255) / 256),
                256>>>(buffer.data(), buffer.size(), block_bytes, d_offset,
                       dmin_offset, 0);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string gate_name = argc > 1 ? argv[1] : "iq3xxs";
  const std::string up_name = argc > 2 ? argv[2] : gate_name;
  const std::string down_name = argc > 3 ? argv[3] : gate_name;
  const std::size_t batch = argc > 4 ? std::stoul(argv[4]) : 256;
  const std::size_t hidden = 5120, ffn = 17408;
  const float eps = 1e-6F;

  GgmlType gate_type{}, up_type{}, down_type{};
  if (!ParseType(gate_name, &gate_type) || !ParseType(up_name, &up_type) ||
      !ParseType(down_name, &down_type)) {
    std::fprintf(stderr, "unknown type\n");
    return 2;
  }

  const std::size_t gate_row = gufo::quant::QuantizedRowBytes(gate_type, hidden);
  const std::size_t up_row = gufo::quant::QuantizedRowBytes(up_type, hidden);
  const std::size_t down_row = gufo::quant::QuantizedRowBytes(down_type, ffn);

  DeviceBuffer<unsigned char> gate_w(gate_row * ffn), up_w(up_row * ffn),
      down_w(down_row * hidden);
  DeviceBuffer<float> hidden_in(batch * hidden), hidden_weight(hidden),
      gate_fp32(batch * ffn), out(batch * hidden);
  DeviceBuffer<short> normed(batch * hidden), act(batch * ffn);

  FillDevice(gate_w, gate_type, hidden);
  FillDevice(up_w, up_type, hidden);
  FillDevice(down_w, down_type, ffn);
  FillValues<<<static_cast<unsigned>((hidden_in.size() + 255) / 256), 256>>>(
      hidden_in.data(), hidden_in.size(), 0.0009F);
  FillValues<<<static_cast<unsigned>((hidden_weight.size() + 255) / 256), 256>>>(
      hidden_weight.data(), hidden_weight.size(), 0.000004F);
  HIP_CHECK(hipMemcpy(out.data(), hidden_in.data(), hidden_in.size() * sizeof(float),
                      hipMemcpyDeviceToDevice));

  gufo::hip::LaunchBatchedRMSNormFp16(hidden_in.data(), nullptr,
                                      hidden_weight.data(), nullptr,
                                      normed.data(), batch, hidden, eps,
                                      nullptr);
  const bool paired = gufo::hip::TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
      gate_type, up_type, gate_w.data(), up_w.data(), normed.data(), act.data(),
      batch, ffn, hidden, nullptr);
  if (!paired) {
    gufo::hip::LaunchBatchedQuantGEMMFp16(gate_type, gate_w.data(),
                                          normed.data(), gate_fp32.data(), batch,
                                          ffn, hidden, nullptr);
    gufo::hip::LaunchBatchedQuantGEMMSwiGLUFp16(
        up_type, up_w.data(), normed.data(), gate_fp32.data(), act.data(), batch,
        ffn, hidden, nullptr);
  }
  gufo::hip::LaunchBatchedQuantGEMMResidualFp16(down_type, down_w.data(),
                                                act.data(), out.data(), batch,
                                                hidden, ffn, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  const auto hidden_host = hidden_in.CopyToHost();
  const auto weight_host = hidden_weight.CopyToHost();
  const auto normed_host = normed.CopyToHost();
  const auto act_host = act.CopyToHost();
  const auto out_host = out.CopyToHost();
  const auto gate_host = gate_w.CopyToHost();
  const auto up_host = up_w.CopyToHost();
  const auto down_host = down_w.CopyToHost();

  std::vector<float> normed_ref(batch * hidden), act_ref(batch * ffn);
  for (std::size_t i = 0; i < normed_ref.size(); ++i) {
    normed_ref[i] = HalfToFloat(static_cast<std::uint16_t>(normed_host[i]));
  }
  for (std::size_t i = 0; i < act_ref.size(); ++i) {
    act_ref[i] = HalfToFloat(static_cast<std::uint16_t>(act_host[i]));
  }

  double max_abs = 0.0, sum_sq = 0.0, ref_sq = 0.0;
  for (std::size_t b = 0; b < batch; ++b) {
    const float* act_row = act_ref.data() + b * ffn;
    for (std::size_t o = 0; o < hidden; ++o) {
      const float reference =
          hidden_host[b * hidden + o] +
          ReferenceDot(down_type, down_host.data() + o * down_row, act_row, ffn);
      const double got = out_host[b * hidden + o];
      const double diff = std::fabs(got - reference);
      max_abs = std::max(max_abs, diff);
      sum_sq += diff * diff;
      ref_sq += static_cast<double>(reference) * reference;
    }
  }

  // Separately confirm the norm matches the reference formula, reading the
  // device's own FP16 output as the input so only the reduction is compared.
  double norm_max = 0.0;
  for (std::size_t b = 0; b < batch; ++b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < hidden; ++i) {
      const double v = hidden_host[b * hidden + i];
      sum += v * v;
    }
    const double scale = 1.0 / std::sqrt(sum / static_cast<double>(hidden) + eps);
    for (std::size_t i = 0; i < hidden; ++i) {
      const float want = static_cast<float>(
          hidden_host[b * hidden + i] * scale * weight_host[i]);
      const float got = normed_ref[b * hidden + i];
      const double diff = std::fabs(static_cast<double>(want - got));
      norm_max = std::max(norm_max, diff);
    }
  }

  const double count = static_cast<double>(batch) * static_cast<double>(hidden);
  const double rms = std::sqrt(sum_sq / count);
  const double ref_rms = std::sqrt(ref_sq / count);
  const double rel = ref_rms > 0.0 ? rms / ref_rms : 0.0;
  std::printf("gate=%s up=%s down=%s batch=%zu paired=%d\n", gate_name.c_str(),
              up_name.c_str(), down_name.c_str(), batch, paired ? 1 : 0);
  std::printf("norm_max_abs=%.4g  out max_abs=%.6g rms=%.6g ref_rms=%.6g rel_rms=%.3g\n",
              norm_max, max_abs, rms, ref_rms, rel);
  // A NaN makes every comparison false, so finiteness has to be part of the
  // pass condition rather than implied by it.
  const bool finite = std::isfinite(max_abs) && std::isfinite(rms) &&
                      std::isfinite(ref_rms) && std::isfinite(rel) &&
                      std::isfinite(norm_max);
  const bool ok = finite && rel < 5e-3 && norm_max < 1e-3;
  std::printf("%s\n", ok ? "FFN CHECK PASS" : "FFN CHECK FAIL");
  return ok ? 0 : 1;
}
