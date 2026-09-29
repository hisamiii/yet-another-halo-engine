// Correctness check for the ported fp16 prefill GEMM: the same quantized
// weight bytes are decoded twice, once by the device kernel and once by the
// host DotProduct reference, and the two matrices are compared. This is the gate
// for trusting the per-type rates beyond their relative ordering.
#define ENGINE_ENABLE_HIP 1
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>
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

__global__ void FillValues(float* values, std::size_t count) {
  const std::size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < count) {
    values[i] = (static_cast<int>((i * 7919) % 2001) - 1000) * 0.00011F;
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

const char* TypeName(GgmlType type) {
  switch (type) {
    case GgmlType::kQ4_K: return "Q4_K";
    case GgmlType::kQ5_K: return "Q5_K";
    case GgmlType::kQ6_K: return "Q6_K";
    case GgmlType::kQ3_K: return "Q3_K";
    case GgmlType::kQ2_K: return "Q2_K";
    case GgmlType::kQ8_0: return "Q8_0";
    case GgmlType::kIQ4_NL: return "IQ4_NL";
    case GgmlType::kIQ4_XS: return "IQ4_XS";
    case GgmlType::kIQ3_S: return "IQ3_S";
    case GgmlType::kIQ3_XXS: return "IQ3_XXS";
    case GgmlType::kIQ2_XS: return "IQ2_XS";
    case GgmlType::kIQ2_S: return "IQ2_S";
    case GgmlType::kIQ2_XXS: return "IQ2_XXS";
    default: return "?";
  }
}

bool ParseType(const std::string& name, GgmlType* out) {
  const std::pair<const char*, GgmlType> kTypes[] = {
      {"q4k", GgmlType::kQ4_K},     {"q5k", GgmlType::kQ5_K},
      {"q6k", GgmlType::kQ6_K},     {"q3k", GgmlType::kQ3_K},
      {"q2k", GgmlType::kQ2_K},     {"q8_0", GgmlType::kQ8_0},
      {"iq4nl", GgmlType::kIQ4_NL}, {"iq4xs", GgmlType::kIQ4_XS},
      {"iq3s", GgmlType::kIQ3_S},   {"iq3xxs", GgmlType::kIQ3_XXS},
      {"iq2xs", GgmlType::kIQ2_XS}, {"iq2s", GgmlType::kIQ2_S},
      {"iq2xxs", GgmlType::kIQ2_XXS}};
  for (const auto& [label, type] : kTypes) {
    if (name == label) {
      *out = type;
      return true;
    }
  }
  return false;
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

float HalfToFloat(std::uint16_t bits) {
  __half h;
  std::memcpy(&h, &bits, sizeof(h));
  return __half2float(h);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string type_name = argc > 1 ? argv[1] : "q4k";
  const std::size_t m = argc > 2 ? std::stoul(argv[2]) : 512;
  const std::size_t k = argc > 3 ? std::stoul(argv[3]) : 512;
  const std::size_t batch = argc > 4 ? std::stoul(argv[4]) : 512;

  GgmlType type{};
  if (!ParseType(type_name, &type)) {
    std::fprintf(stderr, "unknown type %s\n", type_name.c_str());
    return 2;
  }
  if (k % 256 != 0 || batch == 0 || m == 0) {
    std::fprintf(stderr, "require k %% 256 == 0\n");
    return 2;
  }

  const std::size_t row_bytes = gufo::quant::QuantizedRowBytes(type, k);
  const std::size_t weight_bytes = row_bytes * m;
  const std::size_t block_bytes = gufo::quant::QuantizedRowBytes(
      type, gufo::quant::QuantizedBlockElements(type));
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(type);

  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, 0);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);

  gufo::hip::LaunchBatchedQuantGEMMFp16(type, weights.data(), half_x.data(),
                                        y.data(), batch, m, k, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  const auto weights_host = weights.CopyToHost();
  const auto half_x_host = half_x.CopyToHost();
  const auto y_host = y.CopyToHost();

  std::vector<float> x_float(batch * k);
  for (std::size_t i = 0; i < x_float.size(); ++i) {
    x_float[i] = HalfToFloat(static_cast<std::uint16_t>(half_x_host[i]));
  }

  double max_abs = 0.0;
  double sum_sq = 0.0;
  double ref_sq = 0.0;
  std::size_t worst_b = 0;
  std::size_t worst_r = 0;
  for (std::size_t b = 0; b < batch; ++b) {
    const float* x_row = x_float.data() + b * k;
    for (std::size_t r = 0; r < m; ++r) {
      const void* w_row = weights_host.data() + r * row_bytes;
      const float reference = ReferenceDot(type, w_row, x_row, k);
      const double got = y_host[b * m + r];
      const double diff = std::fabs(got - reference);
      if (diff > max_abs) {
        max_abs = diff;
        worst_b = b;
        worst_r = r;
      }
      sum_sq += diff * diff;
      ref_sq += static_cast<double>(reference) * reference;
    }
  }
  const double count = static_cast<double>(batch) * static_cast<double>(m);
  const double rms = std::sqrt(sum_sq / count);
  const double ref_rms = std::sqrt(ref_sq / count);
  const double rel = ref_rms > 0.0 ? rms / ref_rms : 0.0;

  std::printf("type=%s m=%zu k=%zu batch=%zu row_bytes=%zu\n",
              TypeName(type), m, k, batch, row_bytes);
  std::printf("max_abs=%.6g rms=%.6g ref_rms=%.6g rel_rms=%.3g (worst b=%zu r=%zu)\n",
              max_abs, rms, ref_rms, rel, worst_b, worst_r);
  const bool ok = rel < 1e-3 && std::isfinite(rel);
  std::printf("%s\n", ok ? "CHECK PASS" : "CHECK FAIL");
  return ok ? 0 : 1;
}
