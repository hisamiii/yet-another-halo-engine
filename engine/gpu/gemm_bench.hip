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

// The prefill path does not run the plain store epilogue everywhere: gate is
// kStore only when the pair is not fused, up is kSwiGLU, down is kResidual, and
// the paired gate/up kernel is kGateUp. Each has its own buffer traffic and
// its own output width (kSwiGLU and kGateUp write fp16), so a table measured
// with kStore alone is not the model's kernel.
void RunFp16(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
             unsigned iterations, int pattern, const char* epilogue) {
  using namespace gufo::hip;
  if (m == 0 || k == 0 || k % 256 != 0 || batch < 96 || iterations == 0)
    throw std::invalid_argument("require K % 256 == 0, batch >= 96");
  const bool swiglu = std::strcmp(epilogue, "swiglu") == 0;
  const bool residual = std::strcmp(epilogue, "residual") == 0;
  const bool dual = std::strcmp(epilogue, "gateup") == 0;
  const bool fp16_out = swiglu || dual;
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(type, gufo::quant::QuantizedBlockElements(type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<unsigned char> up_weights(dual ? weight_bytes : 1);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m), gate(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  if (dual) {
    FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                  256>>>(up_weights.data(), weight_bytes, block_bytes,
                          d_offset, dmin_offset, pattern);
  }
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(), nullptr);
  const auto launch = [&] {
    if (swiglu) {
      LaunchBatchedQuantGEMMSwiGLUFp16(type, weights.data(), half_x.data(),
                                       gate.data(), y.data(), batch, m, k,
                                       nullptr);
    } else if (residual) {
      LaunchBatchedQuantGEMMResidualFp16(type, weights.data(), half_x.data(),
                                         y.data(), batch, m, k, nullptr);
    } else if (dual) {
      TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
          type, type, weights.data(), up_weights.data(), half_x.data(),
          y.data(), batch, m, k, nullptr);
    } else {
      LaunchBatchedQuantGEMMFp16(type, weights.data(), half_x.data(), y.data(),
                                 batch, m, k, nullptr);
    }
  };
  launch();
  HIP_CHECK(hipDeviceSynchronize());
  std::size_t nonfinite = 0;
  if (fp16_out) {
    const auto host = y.CopyToHost();
    const auto* halves = reinterpret_cast<const __half*>(host.data());
    for (std::size_t i = 0; i < batch * m; ++i)
      nonfinite += !std::isfinite(__half2float(halves[i]));
  } else {
    const auto host = y.CopyToHost();
    for (const float value : host) nonfinite += !std::isfinite(value);
  }
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
  // kGateUp computes both gate and up, so its FLOPs are twice a single
  // projection's; the other three are one projection each.
  const double flops =
      2.0 * static_cast<double>(m) * k * batch * (dual ? 2.0 : 1.0);
  for (unsigned repeat = 0; repeat < 3; ++repeat) {
    const double ms = time();
    const double tflops = flops / (ms * 1e9);
    std::printf(
        "fp16 type=%s batch=%zu M=%zu K=%zu epi=%s pattern=%d rep=%u "
        "ms=%.6f tflops=%.2f nonfinite=%zu\n",
        TypeName(type), batch, m, k, epilogue, pattern, repeat, ms, tflops,
        nonfinite);
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

// Ablation sweep for the K-loop scheduling hint. Ablate bit 16 is iglp_opt(0),
// which is also what the default path applies to the IQ types, 2048 is
// iglp_opt(1) and 8192 is iglp_opt(2). All three are timed in one process with
// the order rotated per round, because the first timed block in a process is a
// boosted one and a fixed order would hand it to one variant.
template<gufo::core::GgmlType Type, int Ablate>
void LaunchAblate(const void* w, const short* x, float* y, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, Ablate>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr);
}

template<gufo::core::GgmlType Type>
void RunAblate(std::size_t m, std::size_t k, std::size_t batch,
               unsigned iterations, int pattern) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);
  // 16 is the control (iglp_opt(0), which the IQ types get by default).
  // 256 interleaves the next stage's global fetch into the K loop; 4096 batches
  // every B fragment load ahead of the MMA block, which is the pipelining
  // question -- the shipped order consumes each B fragment back to back, with a
  // load-to-use distance of zero.
  // Phase pricing. 16 is the control. Bit 1 drops the K-loop barrier, 2 the
  // register->LDS commit, 4 the global fetch, 8 the epilogue store. They
  // produce wrong results on purpose; the point is to price each phase against
  // the phase clock, which says staging plus barriers is 16% and the untimed
  // prologue/epilogue another 14%.
  const int values[5] = {16, 1, 2, 4, 8};
  const char* names[5] = {"control", "noBarrier", "noCommit", "noFetch",
                          "noStore"};
  const auto launch = [&](int variant) {
    if (variant == 1)
      LaunchAblate<Type, 1>(weights.data(), half_x.data(), y.data(), batch, m, k,
                            nullptr);
    else if (variant == 2)
      LaunchAblate<Type, 2>(weights.data(), half_x.data(), y.data(), batch, m, k,
                            nullptr);
    else if (variant == 4)
      LaunchAblate<Type, 4>(weights.data(), half_x.data(), y.data(), batch, m, k,
                            nullptr);
    else if (variant == 8)
      LaunchAblate<Type, 8>(weights.data(), half_x.data(), y.data(), batch, m, k,
                            nullptr);
    else
      LaunchAblate<Type, 16>(weights.data(), half_x.data(), y.data(), batch, m,
                             k, nullptr);
  };
  const auto timed = [&](int variant) {
    for (unsigned i = 0; i < 2; ++i) launch(variant);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) launch(variant);
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  for (int round = 0; round < 5; ++round) {
    for (int i = 0; i < 5; ++i) {
      const int idx = (i + round) % 5;
      const double ms = timed(values[idx]);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("ablate type=%s variant=%s round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchAblate(GgmlType type, std::size_t m, std::size_t k,
                    std::size_t batch, unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunAblate<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ5_K:
      RunAblate<GgmlType::kQ5_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kIQ4_XS:
      RunAblate<GgmlType::kIQ4_XS>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kIQ3_XXS:
      RunAblate<GgmlType::kIQ3_XXS>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kIQ3_S:
      RunAblate<GgmlType::kIQ3_S>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr,
                   "ablate sweep covers q4k, q5k, iq4xs, iq3xxs, iq3s\n");
      return;
  }
}

// BK (BKDepth) sweep. The production path fixes BK at 4, which fills the 64 KiB
// LDS budget and allows one block per CU. BK also sets how many 16-wide K steps
// one iteration covers, so lowering it halves LDS per stage and doubles the
// resident blocks, at the cost of more loop iterations and barriers. That
// tradeoff is the K loop's latency question and has never been swept.
template<gufo::core::GgmlType Type, int BK>
void LaunchTileBk(const void* w, const short* x, float* y, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   BK>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr);
}

template<gufo::core::GgmlType Type>
void RunBk(std::size_t m, std::size_t k, std::size_t batch, unsigned iterations,
           int pattern) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);
  const auto launch = [&](int bk) {
    switch (bk) {
      case 1:
        LaunchTileBk<Type, 1>(weights.data(), half_x.data(), y.data(), batch, m,
                              k, nullptr);
        break;
      case 2:
        LaunchTileBk<Type, 2>(weights.data(), half_x.data(), y.data(), batch, m,
                              k, nullptr);
        break;
      case 3:
        LaunchTileBk<Type, 3>(weights.data(), half_x.data(), y.data(), batch, m,
                              k, nullptr);
        break;
      default:
        LaunchTileBk<Type, 4>(weights.data(), half_x.data(), y.data(), batch, m,
                              k, nullptr);
        break;
    }
  };
  const auto timed = [&](int bk) {
    for (unsigned i = 0; i < 2; ++i) launch(bk);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) launch(bk);
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  for (int round = 0; round < 4; ++round) {
    for (int i = 0; i < 4; ++i) {
      const int bk = 1 + ((i + round) % 4);
      const double ms = timed(bk);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("bk type=%s bk=%d round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), bk, round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchBk(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunBk<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kIQ4_XS:
      RunBk<GgmlType::kIQ4_XS>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kIQ3_XXS:
      RunBk<GgmlType::kIQ3_XXS>(m, k, batch, iterations, pattern);
      return;
    // IQ3_S is absent: its LDS-transpose epilogue asserts
    // Slots >= WM*WN, and Slots = 8*BK, so BK must be 4 for that type.
    default:
      std::fprintf(stderr, "bk sweep covers q4k, iq4xs, iq3xxs\n");
      return;
  }
}

// Operand-reuse grouping sweep. group_shift controls how many row tiles share a
// token tile inside one grid group, i.e. which operand stays resident while the
// other streams. It is hand-set per type in the kernel (5 for IQ4_XS, 5-or-3 for
// Q5_K, 3 otherwise) and GShift is a template parameter, so it can be swept
// without touching the production default.
template<gufo::core::GgmlType Type, int GShift>
void LaunchTileG(const void* w, const short* x, float* y, std::size_t batch,
                 std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, 0, GShift>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr);
}

template<gufo::core::GgmlType Type>
void RunGShift(std::size_t m, std::size_t k, std::size_t batch,
               unsigned iterations, int pattern) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);
  const auto launch = [&](int g) {
    switch (g) {
      case 0: LaunchTileG<Type, 0>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 1: LaunchTileG<Type, 1>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 2: LaunchTileG<Type, 2>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 3: LaunchTileG<Type, 3>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 4: LaunchTileG<Type, 4>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 5: LaunchTileG<Type, 5>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      default: LaunchTileG<Type, 6>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
    }
  };
  const auto timed = [&](int g) {
    for (unsigned i = 0; i < 2; ++i) launch(g);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) launch(g);
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  // Rotated every round: running 0..6 in a fixed order hands the boosted first
  // block to gshift=0 and reads back the session's decay as a preference.
  for (int round = 0; round < 7; ++round) {
    for (int i = 0; i < 7; ++i) {
      const int g = (i + round) % 7;
      const double ms = timed(g);
      const double tflops =
          (2.0 * static_cast<double>(m) * k * batch) / (ms * 1e9);
      std::printf("gshift type=%s gshift=%d round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), g, round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchGShift(GgmlType type, std::size_t m, std::size_t k,
                    std::size_t batch, unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunGShift<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kIQ4_XS:
      RunGShift<GgmlType::kIQ4_XS>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kIQ3_XXS:
      RunGShift<GgmlType::kIQ3_XXS>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "gshift sweep covers q4k, iq4xs, iq3xxs\n");
      return;
  }
}

// Double-buffered staging sweep. The same tile at BK=2 with one LDS stage
// buffer (control), BK=2 with two stage buffers -- the commit for stage i+1 is
// issued before stage i's K loop, so the LDS writes drain under the matrix pipe
// and the trailing barrier only orders compute -- and the BK=4 production
// control. One variable changes at a time between the first two arms; the
// third says what the current default already gets.
template<gufo::core::GgmlType Type, int BKDepth, bool DblBuf>
void LaunchTileD(const void* w, const short* x, float* y, std::size_t batch,
                 std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   BKDepth, 0, 3, DblBuf>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr);
}

template<gufo::core::GgmlType Type>
void RunDbl(std::size_t m, std::size_t k, std::size_t batch,
            unsigned iterations, int pattern) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);
  const auto launch = [&](int v) {
    switch (v) {
      case 0: LaunchTileD<Type, 2, false>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 1: LaunchTileD<Type, 2, true>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      default: LaunchTileD<Type, 4, false>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
    }
  };
  const auto timed = [&](int v) {
    for (unsigned i = 0; i < 2; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  const char* names[3] = {"bk2_single", "bk2_dblbuf", "bk4_control"};
  // Rotated so no arm inherits the session's opening boost.
  for (int round = 0; round < 3; ++round) {
    for (int i = 0; i < 3; ++i) {
      const int v = (i + round) % 3;
      const double ms = timed(v);
      const double tflops =
          (2.0 * static_cast<double>(m) * k * batch) / (ms * 1e9);
      std::printf("dbl type=%s variant=%s round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[v], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchDbl(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                 unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunDbl<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ5_K:
      RunDbl<GgmlType::kQ5_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kIQ4_XS:
      RunDbl<GgmlType::kIQ4_XS>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kIQ3_XXS:
      RunDbl<GgmlType::kIQ3_XXS>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "dbl sweep covers q4k, q5k, iq4xs, iq3xxs\n");
      return;
  }
}

// Decode-placement sweep for the deferred-decode (DecodeStage==2) types. 16 is
// the shipped arm: decode happens in commit, at the top of the K loop, after the
// barrier and before any matrix work, so its VALU burst runs with the matrix
// pipe idle. 512 decodes stage i+1 inside stage i's K loop instead, and 1024
// staggers that to a warp-dependent sub-stage so each warp enters the decode at
// a different ks. Both are numerically identical to 16 -- the decode moves, it
// does not change pass -- so a win here is free.
template<gufo::core::GgmlType Type, int Ablate>
void LaunchHide(const void* w, const short* x, float* y, std::size_t batch,
                std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, Ablate>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr);
}

template<gufo::core::GgmlType Type>
void RunHide(std::size_t m, std::size_t k, std::size_t batch,
             unsigned iterations, int pattern) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);
  const int values[3] = {16, 512, 1024};
  const char* names[3] = {"commit", "kloop", "stagger"};
  const auto launch = [&](int variant) {
    if (variant == 512)
      LaunchHide<Type, 512>(weights.data(), half_x.data(), y.data(), batch, m, k,
                            nullptr);
    else if (variant == 1024)
      LaunchHide<Type, 1024>(weights.data(), half_x.data(), y.data(), batch, m, k,
                             nullptr);
    else
      LaunchHide<Type, 16>(weights.data(), half_x.data(), y.data(), batch, m, k,
                           nullptr);
  };
  const auto timed = [&](int variant) {
    for (unsigned i = 0; i < 2; ++i) launch(variant);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) launch(variant);
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  for (int round = 0; round < 3; ++round) {
    for (int i = 0; i < 3; ++i) {
      const int idx = (i + round) % 3;
      const double ms = timed(values[idx]);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("hide type=%s arm=%s round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}


// Cumulative "what feeds the K loop" ladder. Each arm adds exactly one feeder
// to the one above it, on the same tile and the same shape, so the deltas are
// additive and read as a flow: bare K loop -> global fetch -> decode -> LDS
// store -> barrier. Ablate is a bitmask (1 noBarrier, 2 noCommit, 4 noFetch,
// 8 noStore-y, 16 iglp_opt(0), 32 noLdsStore, 64 noDecode), so the arms are
// just ORs of removals.
//
//   k_only          6  = noFetch|noCommit        K loop against stale LDS
//   k_fetch         2  = noCommit                + the next stage's global loads
//   k_store        68  = noFetch|noDecode        + the LDS store, undecoded
//   k_decode       36  = noFetch|noLdsStore      + the decode, no store
//   k_decode_store  4  = noFetch                 + the store (full commit)
//   full           16  = control                 + the global fetch
//   full_noBarrier 17  = control without the barrier
template<gufo::core::GgmlType Type, int Ablate>
void LaunchFeed(const void* w, const short* x, float* y, std::size_t batch,
                std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, Ablate>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr);
}

template<gufo::core::GgmlType Type>
void RunFeed(std::size_t m, std::size_t k, std::size_t batch,
             unsigned iterations, int pattern) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);
  const int values[7] = {6, 2, 68, 36, 4, 16, 17};
  const char* names[7] = {"k_only", "k_fetch", "k_store", "k_decode",
                          "k_decode_store", "full", "full_noBarrier"};
  const auto launch = [&](int variant) {
    switch (variant) {
      case 6: LaunchFeed<Type, 6>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 2: LaunchFeed<Type, 2>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 68: LaunchFeed<Type, 68>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 36: LaunchFeed<Type, 36>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 4: LaunchFeed<Type, 4>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 17: LaunchFeed<Type, 17>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      default: LaunchFeed<Type, 16>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
    }
  };
  const auto timed = [&](int variant) {
    for (unsigned i = 0; i < 2; ++i) launch(variant);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) launch(variant);
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  for (int round = 0; round < 7; ++round) {
    for (int i = 0; i < 7; ++i) {
      const int idx = (i + round) % 7;
      const double ms = timed(values[idx]);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("feed type=%s arm=%s ablate=%d round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], values[idx], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchFeed(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                  unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunFeed<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ5_K:
      RunFeed<GgmlType::kQ5_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ6_K:
      RunFeed<GgmlType::kQ6_K>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "feed ladder covers q4k, q5k, q6k\n");
      return;
  }
}


// Store-side component matrix. The commit writes the decoded weights (A) and a
// plain copy of the activations (B); Ablate 32 drops the A store while keeping
// its decode, 64 drops the decode while keeping the store, and 16384 drops the
// B store, so this separates the four pieces of the 25% that the store costs.
template<gufo::core::GgmlType Type, int Ablate>
void LaunchStore(const void* w, const short* x, float* y, std::size_t batch,
                 std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, Ablate>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr);
}

template<gufo::core::GgmlType Type>
void RunStore(std::size_t m, std::size_t k, std::size_t batch,
              unsigned iterations, int pattern) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);
  const int values[6] = {16, 80, 48, 16400, 16432, 18};
  const char* names[6] = {"full", "noDecodeA", "noStoreA", "noStoreB",
                          "noStoreAB", "noCommit"};
  const auto launch = [&](int v) {
    switch (v) {
      case 80: LaunchStore<Type, 80>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 48: LaunchStore<Type, 48>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 16400: LaunchStore<Type, 16400>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 16432: LaunchStore<Type, 16432>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 18: LaunchStore<Type, 18>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      default: LaunchStore<Type, 16>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
    }
  };
  const auto timed = [&](int v) {
    for (unsigned i = 0; i < 2; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  for (int round = 0; round < 6; ++round) {
    for (int i = 0; i < 6; ++i) {
      const int idx = (i + round) % 6;
      const double ms = timed(values[idx]);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("store type=%s arm=%s ablate=%d round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], values[idx], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}


// K-loop scheduling variants, all on top of iglp_opt(0) (Ablate bit 16).
// The shipped order consumes each B fragment back to back with its MMA, so the
// load-to-use distance is zero and every WMMA eats the full LDS latency;
// 4096 batches all WTS B fragments ahead of the MMA block to give them the MMA
// block to land behind. 2048/8192 are the other two iglp_opt immediates and 256
// interleaves the next stage's global fetch into the K loop.
template<gufo::core::GgmlType Type, int Ablate>
void LaunchSched(const void* w, const short* x, float* y, std::size_t batch,
                 std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, Ablate>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr);
}

template<gufo::core::GgmlType Type>
void RunSched(std::size_t m, std::size_t k, std::size_t batch,
              unsigned iterations, int pattern) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);
  const int values[6] = {16, 4112, 4368, 272, 2064, 8208};
  const char* names[6] = {"control", "batchB", "batchB_fetchmid", "fetchmid",
                          "iglp1", "iglp2"};
  const auto launch = [&](int v) {
    switch (v) {
      case 4112: LaunchSched<Type, 4112>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 4368: LaunchSched<Type, 4368>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 272: LaunchSched<Type, 272>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 2064: LaunchSched<Type, 2064>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 8208: LaunchSched<Type, 8208>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      default: LaunchSched<Type, 16>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
    }
  };
  const auto timed = [&](int v) {
    for (unsigned i = 0; i < 2; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  for (int round = 0; round < 6; ++round) {
    for (int i = 0; i < 6; ++i) {
      const int idx = (i + round) % 6;
      const double ms = timed(values[idx]);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("sched type=%s arm=%s ablate=%d round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], values[idx], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchSched(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                   unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunSched<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ5_K:
      RunSched<GgmlType::kQ5_K>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "sched sweep covers q4k, q5k\n");
      return;
  }
}


// Store-instruction vs store-byte sweep. The commit's store byte volume is held
// constant while the number of DS store instructions changes (2 x b128, 4 x b64,
// 8 x b32 per call), and then the same ladder is repeated at half the bytes by
// dropping the weight-store half. The grid is (instructions, bytes):
//
//   b128_64B  16       (4, 64)      b128_32B  48       (2, 32)
//   b64_64B   131088   (8, 64)      b64_32B   131120   (4, 32)
//   b32_64B   262160   (16, 64)     b32_32B   262192   (8, 32)
//
// If the penalty tracks bytes, each row is flat and the 32B row is far cheaper.
// If it tracks DS instruction count, the cost rises left to right within a row.
template<gufo::core::GgmlType Type, int Ablate>
void LaunchSw(const void* w, const short* x, float* y, std::size_t batch,
              std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, Ablate>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr);
}

template<gufo::core::GgmlType Type>
void RunSw(std::size_t m, std::size_t k, std::size_t batch,
           unsigned iterations, int pattern) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);
  const int values[6] = {16, 131088, 262160, 48, 131120, 262192};
  const char* names[6] = {"b128_64B", "b64_64B", "b32_64B",
                          "b128_32B", "b64_32B", "b32_32B"};
  const auto launch = [&](int v) {
    switch (v) {
      case 131088: LaunchSw<Type, 131088>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 262160: LaunchSw<Type, 262160>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 48: LaunchSw<Type, 48>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 131120: LaunchSw<Type, 131120>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 262192: LaunchSw<Type, 262192>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      default: LaunchSw<Type, 16>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
    }
  };
  const auto timed = [&](int v) {
    for (unsigned i = 0; i < 2; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  for (int round = 0; round < 6; ++round) {
    for (int i = 0; i < 6; ++i) {
      const int idx = (i + round) % 6;
      const double ms = timed(values[idx]);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("sw type=%s arm=%s ablate=%d round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], values[idx], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchSw(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunSw<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ5_K:
      RunSw<GgmlType::kQ5_K>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "sw sweep covers q4k, q5k\n");
      return;
  }
}


// Regions x bytes. All arms move 64 B, 32 B or 0 B per thread per stage with the
// same tile, so the only thing that changes is how much data the commit writes
// and whether it writes both staging arrays.
//
//   full        16              both arrays, 64 B, 4 x b128
//   bothHalf    524304          both arrays, 32 B (16 B per call)
//   halfA       48              s_b only,    32 B
//   halfB       16400           s_a only,    32 B
//   sameRegion  32784           s_a twice,   64 B, one array
//   none        16432           no stores
template<gufo::core::GgmlType Type, int Ablate>
void LaunchReg(const void* w, const short* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, Ablate>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr);
}

template<gufo::core::GgmlType Type>
void RunReg(std::size_t m, std::size_t k, std::size_t batch,
            unsigned iterations, int pattern) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);
  const int values[6] = {16, 524304, 48, 16400, 32784, 16432};
  const char* names[6] = {"full", "bothHalf", "halfA", "halfB", "sameRegion",
                          "none"};
  const auto launch = [&](int v) {
    switch (v) {
      case 524304: LaunchReg<Type, 524304>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 48: LaunchReg<Type, 48>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 16400: LaunchReg<Type, 16400>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 32784: LaunchReg<Type, 32784>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 16432: LaunchReg<Type, 16432>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      default: LaunchReg<Type, 16>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
    }
  };
  const auto timed = [&](int v) {
    for (unsigned i = 0; i < 2; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  for (int round = 0; round < 6; ++round) {
    for (int i = 0; i < 6; ++i) {
      const int idx = (i + round) % 6;
      const double ms = timed(values[idx]);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("reg type=%s arm=%s ablate=%d round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], values[idx], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchReg(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                 unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunReg<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ5_K:
      RunReg<GgmlType::kQ5_K>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "reg sweep covers q4k, q5k\n");
      return;
  }
}


// Cross-check of the "the write stream costs because it spans both 32 KiB LDS
// halves" law. At BK=4 the two staging tiles are 32 KiB each and the store
// stream spans all 64 KiB; at BK=2 they are 16 KiB each and the whole working
// set is one half. If the law holds, the full-vs-none gap must collapse at BK=2
// and the BK=2 floor is then set by its own stage overhead (twice the stages,
// twice the barriers), not by the stores.
template<gufo::core::GgmlType Type, int BKDepth, int Ablate>
void LaunchRegBk(const void* w, const short* x, float* y, std::size_t batch,
                 std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   BKDepth, Ablate>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr);
}

template<gufo::core::GgmlType Type>
void RunRegBk(std::size_t m, std::size_t k, std::size_t batch,
              unsigned iterations, int pattern) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);
  const char* names[4] = {"bk4_full", "bk4_none", "bk2_full", "bk2_none"};
  const auto launch = [&](int v) {
    switch (v) {
      case 1: LaunchRegBk<Type, 4, 16432>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 2: LaunchRegBk<Type, 2, 16>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 3: LaunchRegBk<Type, 2, 16432>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      default: LaunchRegBk<Type, 4, 16>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
    }
  };
  const auto timed = [&](int v) {
    for (unsigned i = 0; i < 2; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  for (int round = 0; round < 4; ++round) {
    for (int i = 0; i < 4; ++i) {
      const int idx = (i + round) % 4;
      const double ms = timed(idx);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("regbk type=%s arm=%s round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchRegBk(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                   unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunRegBk<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "regbk sweep covers q4k\n");
      return;
  }
}


// Occupancy cross-check. The 256x256 BK=4 tile declares 64 KiB of LDS and
// compiles to 192 VGPRs, so exactly one 1024-thread block is resident per CU and
// the per-SIMD wave count is halved. A 128x128 BK=4 tile declares 32 KiB and
// needs far fewer accumulator registers, so two blocks fit. If the "store cost"
// is really the 2-blocks-to-1-blocks occupancy transition then the -full versus
// -none gap must collapse here even with both staging stores present, and the
// falsifier recorded for the two-region reading is met by occupancy instead.
template<gufo::core::GgmlType Type, int BM, int BN, int WM, int WN, int Ablate>
void LaunchTt(const void* w, const short* x, float* y, std::size_t batch,
              std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + BN - 1) / BN),
                  static_cast<unsigned>((m + BM - 1) / BM));
  gufo::hip::HalfPrefillGemmKernel<BM, BN, WM, WN, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, Ablate>
      <<<grid, WM * WN * 32, 0, stream>>>(w, x, y, batch, m, k, nullptr);
}

template<gufo::core::GgmlType Type>
void RunTt(std::size_t m, std::size_t k, std::size_t batch,
           unsigned iterations, int pattern) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);
  const char* names[6] = {"t256_full", "t256_none", "t128w8n4_full",
                          "t128w8n4_none", "t128w4n4_full", "t128w4n4_none"};
  const auto launch = [&](int v) {
    switch (v) {
      case 1: LaunchTt<Type, 256, 256, 8, 4, 16432>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 2: LaunchTt<Type, 128, 128, 8, 4, 16>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 3: LaunchTt<Type, 128, 128, 8, 4, 16432>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 4: LaunchTt<Type, 128, 128, 4, 4, 16>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 5: LaunchTt<Type, 128, 128, 4, 4, 16432>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      default: LaunchTt<Type, 256, 256, 8, 4, 16>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
    }
  };
  const auto timed = [&](int v) {
    for (unsigned i = 0; i < 2; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  for (int round = 0; round < 6; ++round) {
    for (int i = 0; i < 6; ++i) {
      const int idx = (i + round) % 6;
      const double ms = timed(idx);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("tt type=%s arm=%s round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchTt(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunTt<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ5_K:
      RunTt<GgmlType::kQ5_K>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "tt sweep covers q4k, q5k\n");
      return;
  }
}


// GlobalB: read the activation operand straight from a pre-blocked global buffer
// instead of staging it through a second LDS array, so only the weight tile is
// staged and there is no second region to write. The buffer is [k/16][batch][16]
// halves -- exactly the layout the commit builds in LDS -- and is passed through
// the kernel's unused \`up\` argument. The repack from row-major [batch][k] is
// timed separately, because it is real extra traffic and has to be paid for.
__global__ void RepackBlockedB(const short* __restrict__ src,
                               short* __restrict__ dst, int batch, int k) {
  // Token-major walk: consecutive threads take consecutive K16 blocks of one
  // token, so the gather from row-major src is coalesced and the scatter into
  // the blocked buffer is strided. A tiled transpose would fix the scatter; this
  // is the pessimistic bound on the repack cost.
  const int kg = (blockIdx.x * blockDim.x + threadIdx.x) % (k / 16);
  const int tok = (blockIdx.x * blockDim.x + threadIdx.x) / (k / 16);
  if (tok >= batch) return;
  const short* s = src + (static_cast<std::size_t>(tok) * k) + kg * 16;
  short* d = dst + (static_cast<std::size_t>(kg) * batch + tok) * 16;
#pragma unroll
  for (int i = 0; i < 16; ++i) d[i] = s[i];
}

template<gufo::core::GgmlType Type, int Ablate>
void LaunchGb(const void* w, const short* x, float* y, std::size_t batch,
              std::size_t m, std::size_t k, hipStream_t stream,
              const short* blocked) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, Ablate>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr, blocked);
}

template<gufo::core::GgmlType Type>
void RunGb(std::size_t m, std::size_t k, std::size_t batch, unsigned iterations,
           int pattern) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k), y(batch * m);
  DeviceBuffer<short> half_x(batch * k);
  DeviceBuffer<short> blocked((k / 16) * batch * 16);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);
  // Price the repack: the whole activation matrix read and rewritten once.
  const unsigned repack_blocks =
      static_cast<unsigned>(((k / 16) * batch + 255) / 256);
  for (int i = 0; i < 2; ++i)
    RepackBlockedB<<<repack_blocks, 256>>>(half_x.data(), blocked.data(),
                                           static_cast<int>(batch),
                                           static_cast<int>(k));
  HIP_CHECK(hipDeviceSynchronize());
  const auto rs = std::chrono::steady_clock::now();
  for (int i = 0; i < 5; ++i)
    RepackBlockedB<<<repack_blocks, 256>>>(half_x.data(), blocked.data(),
                                           static_cast<int>(batch),
                                           static_cast<int>(k));
  HIP_CHECK(hipDeviceSynchronize());
  const auto re = std::chrono::steady_clock::now();
  const double repack_ms =
      std::chrono::duration<double, std::milli>(re - rs).count() / 5.0;
  const int values[3] = {16, 16 | 1048576, 16400};
  const char* names[3] = {"full", "globalB", "noStoreB"};
  const auto launch = [&](int v) {
    if (v == (16 | 1048576))
      LaunchGb<Type, 16 | 1048576>(weights.data(), half_x.data(), y.data(), batch,
                                   m, k, nullptr, blocked.data());
    else if (v == 16400)
      LaunchGb<Type, 16400>(weights.data(), half_x.data(), y.data(), batch, m, k,
                            nullptr, nullptr);
    else
      LaunchGb<Type, 16>(weights.data(), half_x.data(), y.data(), batch, m, k,
                         nullptr, nullptr);
  };
  const auto timed = [&](int v) {
    for (unsigned i = 0; i < 2; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) launch(v);
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  std::printf("gb repack_ms=%.4f\n", repack_ms);
  std::fflush(stdout);
  for (int round = 0; round < 4; ++round) {
    for (int i = 0; i < 3; ++i) {
      const int idx = (i + round) % 3;
      const double ms = timed(values[idx]);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("gb type=%s arm=%s ablate=%d round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], values[idx], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchGb(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunGb<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ5_K:
      RunGb<GgmlType::kQ5_K>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "gb sweep covers q4k, q5k\n");
      return;
  }
}

void DispatchStore(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                   unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunStore<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ5_K:
      RunStore<GgmlType::kQ5_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ6_K:
      RunStore<GgmlType::kQ6_K>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "store matrix covers q4k, q5k, q6k\n");
      return;
  }
}

void DispatchHide(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                  unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunHide<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ6_K:
      RunHide<GgmlType::kQ6_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ8_0:
      RunHide<GgmlType::kQ8_0>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "hide sweep covers q4k, q6k, q8_0\n");
      return;
  }
}

#if defined(GUFO_FP16_PROTO_CLOCK)
// Reads the kernel's own per-phase clock accumulators for block (0,0): how the
// block's cycles split across commit (register -> LDS), barrier, fetch (global
// -> register) and the K loop (LDS read + WMMA). This is the only per-phase
// instrument available and it is the basis for deciding whether a structural
// change is worth making.
template<gufo::core::GgmlType Type>
void RunProto(std::size_t m, std::size_t k, std::size_t batch) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
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
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, 0>
      <<<grid, 1024, 0, nullptr>>>(weights.data(), half_x.data(), y.data(),
                                   batch, m, k, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  unsigned long long host[8] = {};
  HIP_CHECK(hipMemcpyFromSymbol(host, HIP_SYMBOL(gufo::hip::g_proto_clock),
                                sizeof(host)));
  const double total = static_cast<double>(host[5] - host[4]);
  std::printf(
      "proto type=%-8s blocks=%ux%u total=%.0f cyc  commit=%5.1f%% "
      "barrier=%5.1f%% fetch=%5.1f%% kloop=%5.1f%%  raw %llu/%llu/%llu/%llu\n",
      TypeName(Type), grid.x, grid.y, total, 100.0 * host[0] / total,
      100.0 * host[1] / total, 100.0 * host[2] / total, 100.0 * host[3] / total,
      host[0], host[1], host[2], host[3]);
  std::fflush(stdout);
}

void DispatchProto(GgmlType type, std::size_t m, std::size_t k,
                   std::size_t batch) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunProto<GgmlType::kQ4_K>(m, k, batch);
      return;
    case GgmlType::kIQ4_XS:
      RunProto<GgmlType::kIQ4_XS>(m, k, batch);
      return;
    case GgmlType::kIQ3_XXS:
      RunProto<GgmlType::kIQ3_XXS>(m, k, batch);
      return;
    default:
      std::fprintf(stderr, "proto covers q4k, iq4xs, iq3xxs\n");
      return;
  }
}
#endif

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
                 "[iterations] [pattern] [store|swiglu|residual|gateup]\n");
    return 1;
  }
  std::string name = argv[1];
  const bool sweep = name.rfind("tiles:", 0) == 0;
  if (sweep) name = name.substr(6);
  const bool ablate = name.rfind("ablate:", 0) == 0;
  if (ablate) name = name.substr(7);
  const bool bksweep = name.rfind("bk:", 0) == 0;
  if (bksweep) name = name.substr(3);
  const bool proto = name.rfind("proto:", 0) == 0;
  if (proto) name = name.substr(6);
  const bool gsweep = name.rfind("gshift:", 0) == 0;
  if (gsweep) name = name.substr(7);
  const bool dbl = name.rfind("dbl:", 0) == 0;
  if (dbl) name = name.substr(4);
  const bool hide = name.rfind("hide:", 0) == 0;
  if (hide) name = name.substr(5);
  const bool feed = name.rfind("feed:", 0) == 0;
  if (feed) name = name.substr(5);
  const bool storem = name.rfind("store:", 0) == 0;
  if (storem) name = name.substr(6);
  const bool sched = name.rfind("sched:", 0) == 0;
  if (sched) name = name.substr(6);
  const bool swsweep = name.rfind("sw:", 0) == 0;
  if (swsweep) name = name.substr(3);
  const bool regsweep = name.rfind("reg:", 0) == 0;
  if (regsweep) name = name.substr(4);
  const bool regbk = name.rfind("regbk:", 0) == 0;
  if (regbk) name = name.substr(6);
  const bool tt = name.rfind("tt:", 0) == 0;
  if (tt) name = name.substr(3);
  const bool gb = name.rfind("gb:", 0) == 0;
  if (gb) name = name.substr(3);
  const std::size_t m = std::stoull(argv[2]);
  const std::size_t k = std::stoull(argv[3]);
  const std::size_t batch = argc > 4 ? std::stoull(argv[4]) : 2048;
  const unsigned iterations = argc > 5 ? std::stoul(argv[5]) : 12;
  const int pattern = argc > 6 ? std::stoi(argv[6]) : 0;
  const char* epilogue = argc > 7 ? argv[7] : "store";
  try {
    if (name == "all") {
      for (const auto& entry : kTypes)
        RunFp16(entry.type, m, k, batch, iterations, pattern, epilogue);
      return 0;
    }
    for (const auto& entry : kTypes) {
      if (name == entry.name) {
        if (sweep)
          DispatchTiles(entry.type, m, k, batch, iterations, pattern);
        else if (ablate)
          DispatchAblate(entry.type, m, k, batch, iterations, pattern);
        else if (bksweep)
          DispatchBk(entry.type, m, k, batch, iterations, pattern);
        else if (gsweep)
          DispatchGShift(entry.type, m, k, batch, iterations, pattern);
        else if (dbl)
          DispatchDbl(entry.type, m, k, batch, iterations, pattern);
        else if (hide)
          DispatchHide(entry.type, m, k, batch, iterations, pattern);
        else if (feed)
          DispatchFeed(entry.type, m, k, batch, iterations, pattern);
        else if (storem)
          DispatchStore(entry.type, m, k, batch, iterations, pattern);
        else if (sched)
          DispatchSched(entry.type, m, k, batch, iterations, pattern);
        else if (swsweep)
          DispatchSw(entry.type, m, k, batch, iterations, pattern);
        else if (regsweep)
          DispatchReg(entry.type, m, k, batch, iterations, pattern);
        else if (regbk)
          DispatchRegBk(entry.type, m, k, batch, iterations, pattern);
        else if (tt)
          DispatchTt(entry.type, m, k, batch, iterations, pattern);
        else if (gb)
          DispatchGb(entry.type, m, k, batch, iterations, pattern);
#ifdef GUFO_FP16_PROTO_CLOCK
        else if (proto)
          DispatchProto(entry.type, m, k, batch);
#endif
        else
          RunFp16(entry.type, m, k, batch, iterations, pattern, epilogue);
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
