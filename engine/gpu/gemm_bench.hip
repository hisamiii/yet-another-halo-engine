// FP16 prefill GEMM ablations, one weight format at a time. This is the kernel
// LaunchBatchedQuantGEMMFp16 dispatches, so it is the one the chunked prefill
// path actually runs. Build inside Nix with tools/bench/build.sh.
//
//   prefill_fp16_bench <type|all> M K [batch=2048] [iterations=12]
//
// Synthetic device weights; component timings do not replace release model
// measurements with the actual mapped GGUF artifacts.
#define ENGINE_ENABLE_HIP 1
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/kernels/prefill_fp16.hip"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

namespace {
using gufo::core::GgmlType;
using gufo::test::DeviceBuffer;

constexpr std::size_t kNoMin = std::numeric_limits<std::size_t>::max();

__device__ unsigned Hash(unsigned value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

// Random bytes everywhere, then the fp16 scale fields the decoder reads are
// overwritten so the comparison is not dominated by denormals or infinities.
__global__ void FillWeights(unsigned char* data, std::size_t size,
                            std::size_t block_bytes, std::size_t d_offset,
                            std::size_t dmin_offset, int pattern) {
  const std::size_t block = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (block >= size / block_bytes) return;
  auto* bytes = data + (block * block_bytes);
  for (std::size_t i = 0; i < block_bytes; ++i)
    bytes[i] = pattern == 0
                   ? Hash(static_cast<unsigned>(block * block_bytes + i))
                   : static_cast<unsigned char>(pattern);
  reinterpret_cast<__half*>(bytes + d_offset)[0] =
      __float2half(0.007F + (Hash(static_cast<unsigned>(block)) % 1000) * 1e-5F);
  if (dmin_offset != kNoMin)
    reinterpret_cast<__half*>(bytes + dmin_offset)[0] = __float2half(
        0.004F + (Hash(static_cast<unsigned>(block + 19)) % 1000) * 1e-5F);
}

__global__ void FillValues(float* values, std::size_t count) {
  const std::size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < count)
    values[i] = (static_cast<int>(Hash(static_cast<unsigned>(i)) % 20001) -
                 10000) *
                0.00027F;
}

// Scale-field offsets inside one quantization block, mirroring the oracle test.
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

void RunFp16(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
             unsigned iterations, int pattern) {
  using namespace gufo::hip;
  if (m == 0 || k == 0 || k % 256 != 0 || batch < 96 || iterations == 0)
    throw std::invalid_argument("require K % 256 == 0, batch >= 96");
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(type, gufo::quant::QuantizedBlockElements(type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(), nullptr);
  const auto launch = [&] {
    LaunchBatchedQuantGEMMFp16(type, weights.data(), half_x.data(), y.data(),
                               batch, m, k, nullptr);
  };
  launch();
  HIP_CHECK(hipDeviceSynchronize());
  const auto host = y.CopyToHost();
  std::size_t nonfinite = 0;
  for (const float value : host) nonfinite += !std::isfinite(value);
  const auto time = [&] {
    for (unsigned i = 0; i < 2; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  for (unsigned repeat = 0; repeat < 3; ++repeat) {
    const double ms = time();
    const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                          (ms * 1e9);
    std::printf(
        "fp16 type=%s batch=%zu M=%zu K=%zu pattern=%d rep=%u ms=%.6f "
        "tflops=%.2f nonfinite=%zu\n",
        TypeName(type), batch, m, k, pattern, repeat, ms, tflops, nonfinite);
    std::fflush(stdout);
  }
}

struct TypeCase {
  const char* name;
  GgmlType type;
};

const TypeCase kTypes[] = {
    {"q4k", GgmlType::kQ4_K},       {"q5k", GgmlType::kQ5_K},
    {"q6k", GgmlType::kQ6_K},       {"q3k", GgmlType::kQ3_K},
    {"q2k", GgmlType::kQ2_K},       {"q8_0", GgmlType::kQ8_0},
    {"iq4nl", GgmlType::kIQ4_NL},   {"iq4xs", GgmlType::kIQ4_XS},
    {"iq3s", GgmlType::kIQ3_S},     {"iq3xxs", GgmlType::kIQ3_XXS},
    {"iq2xs", GgmlType::kIQ2_XS},   {"iq2s", GgmlType::kIQ2_S},
    {"iq2xxs", GgmlType::kIQ2_XXS},
};
}  // namespace

// ---------------------------------------------------------------------------
// Tile-configuration sweep for the FP16 prefill kernel. The production
// DirectGemm hard-codes one shape per case; this measures a candidate without
// changing model code. BM+BN is capped at 512 because BK is fixed at 4 and the
// two stages must fit the 64 KiB LDS budget.
template<gufo::core::GgmlType Type, int BM, int BN, int WM, int WN,
         bool Complete>
void LaunchTile(const void* w, const short* x, float* y, std::size_t batch,
                std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + BN - 1) / BN),
                  static_cast<unsigned>((m + BM - 1) / BM));
  gufo::hip::HalfPrefillGemmKernel<BM, BN, WM, WN, Type,
                                   gufo::hip::HalfEpilogue::kStore, Complete>
      <<<grid, WM * WN * 32, 0, stream>>>(w, x, y, batch, m, k, nullptr);
}

const char* TileName(int variant) {
  switch (variant) {
    case 0: return "256x256 w8n4";
    case 1: return "256x256 w4n8";
    case 2: return "256x256 w4n4";
    case 3: return "256x256 w16n2";
    case 4: return "256x192 w8n4";
    case 5: return "128x256 w4n8";
    case 6: return "256x128 w8n4";
    case 7: return "128x128 w4n4";
    case 8: return "256x256 w8n4 full";
    default: return "?";
  }
}
constexpr int kTileVariants = 9;

template<gufo::core::GgmlType Type>
void LaunchTileVariant(const void* w, const short* x, float* y,
                       std::size_t batch, std::size_t m, std::size_t k,
                       int variant, hipStream_t stream) {
  switch (variant) {
    case 0: LaunchTile<Type, 256, 256, 8, 4, false>(w, x, y, batch, m, k, stream); break;
    case 1: LaunchTile<Type, 256, 256, 4, 8, false>(w, x, y, batch, m, k, stream); break;
    case 2: LaunchTile<Type, 256, 256, 4, 4, false>(w, x, y, batch, m, k, stream); break;
    case 3: LaunchTile<Type, 256, 256, 16, 2, false>(w, x, y, batch, m, k, stream); break;
    case 4: LaunchTile<Type, 256, 192, 8, 4, false>(w, x, y, batch, m, k, stream); break;
    case 5: LaunchTile<Type, 128, 256, 4, 8, false>(w, x, y, batch, m, k, stream); break;
    case 6: LaunchTile<Type, 256, 128, 8, 4, false>(w, x, y, batch, m, k, stream); break;
    case 7: LaunchTile<Type, 128, 128, 4, 4, false>(w, x, y, batch, m, k, stream); break;
    case 8: LaunchTile<Type, 256, 256, 8, 4, true>(w, x, y, batch, m, k, stream); break;
    default: break;
  }
}

template<gufo::core::GgmlType Type>
void RunTiles(std::size_t m, std::size_t k, std::size_t batch,
              unsigned iterations, int pattern) {
  using namespace gufo::hip;
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m), reference(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(), nullptr);
  LaunchBatchedQuantGEMMFp16(Type, weights.data(), half_x.data(),
                             reference.data(), batch, m, k, nullptr);
  const auto expected = reference.CopyToHost();
  for (int variant = 0; variant < kTileVariants; ++variant) {
    const auto launch = [&] {
      LaunchTileVariant<Type>(weights.data(), half_x.data(), y.data(), batch, m,
                              k, variant, nullptr);
    };
    hipMemset(y.data(), 0, y.size() * sizeof(float));
    launch();
    HIP_CHECK(hipDeviceSynchronize());
    const auto actual = y.CopyToHost();
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i)
      mismatches += actual[i] != expected[i];
    for (unsigned i = 0; i < 2; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(stop - start).count() /
        iterations;
    std::printf(
        "tiles type=%s M=%zu K=%zu variant=%d %-18s ms=%.6f tflops=%.2f "
        "mismatch=%zu\n",
        TypeName(Type), m, k, variant, TileName(variant), ms,
        (2.0 * static_cast<double>(m) * k * batch) / (ms * 1e9), mismatches);
    std::fflush(stdout);
  }
}

void DispatchTiles(GgmlType type, std::size_t m, std::size_t k,
                   std::size_t batch, unsigned iterations, int pattern) {
  switch (type) {
    case GgmlType::kQ4_K:
      RunTiles<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ6_K:
      RunTiles<GgmlType::kQ6_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kIQ4_XS:
      RunTiles<GgmlType::kIQ4_XS>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kIQ3_XXS:
      RunTiles<GgmlType::kIQ3_XXS>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "tiles sweep covers q4k, q6k, iq4xs, iq3xxs\n");
      return;
  }
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: prefill_fp16_bench <type|all> M K [batch] "
                 "[iterations] [pattern]\n");
    return 1;
  }
  std::string name = argv[1];
  const bool sweep = name.rfind("tiles:", 0) == 0;
  if (sweep) name = name.substr(6);
  const std::size_t m = std::stoull(argv[2]);
  const std::size_t k = std::stoull(argv[3]);
  const std::size_t batch = argc > 4 ? std::stoull(argv[4]) : 2048;
  const unsigned iterations = argc > 5 ? std::stoul(argv[5]) : 12;
  const int pattern = argc > 6 ? std::stoi(argv[6]) : 0;
  try {
    if (name == "all") {
      for (const auto& entry : kTypes)
        RunFp16(entry.type, m, k, batch, iterations, pattern);
      return 0;
    }
    for (const auto& entry : kTypes) {
      if (name == entry.name) {
        if (sweep)
          DispatchTiles(entry.type, m, k, batch, iterations, pattern);
        else
          RunFp16(entry.type, m, k, batch, iterations, pattern);
        return 0;
      }
    }
    std::fprintf(stderr, "unknown type: %s\n", name.c_str());
    return 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
