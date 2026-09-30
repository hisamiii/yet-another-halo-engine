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
#include <cstdlib>
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

// Real-data override: YAH_BENCH_W / YAH_BENCH_UP (raw GGUF weight bytes) and
// YAH_BENCH_A (raw fp16 activations, batch x K) replace the synthetic fills, so
// this bench and the Loom case can run the same bytes. Clocks depend on the
// data (real weights run the same cycle count ~25% slower than a constant fill
// on gfx1151), so only same-data comparisons mean anything.
bool LoadOverride(const char* env, void* device, std::size_t bytes) {
  const char* path = std::getenv(env);
  if (path == nullptr) return false;
  FILE* f = std::fopen(path, "rb");
  if (f == nullptr) throw std::runtime_error(std::string("cannot open ") + path);
  std::vector<unsigned char> host(bytes);
  const std::size_t got = std::fread(host.data(), 1, bytes, f);
  std::fclose(f);
  if (got != bytes)
    throw std::runtime_error(std::string(path) + ": short read (" +
                             std::to_string(got) + " of " + std::to_string(bytes) + ")");
  HIP_CHECK(hipMemcpy(device, host.data(), bytes, hipMemcpyHostToDevice));
  std::fprintf(stderr, "%s: %s (%zu bytes)\n", env, path, bytes);
  return true;
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
  HIP_CHECK(hipDeviceSynchronize());
  LoadOverride("YAH_BENCH_W", weights.data(), weight_bytes);
  if (dual) LoadOverride("YAH_BENCH_UP", up_weights.data(), weight_bytes);
  LoadOverride("YAH_BENCH_A", half_x.data(), batch * k * sizeof(short));
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
  HIP_CHECK(hipDeviceSynchronize());
  LoadOverride("YAH_BENCH_W", weights.data(), weight_bytes);
  LoadOverride("YAH_BENCH_A", half_x.data(), batch * k * sizeof(short));
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
  // 6/14/15 stack the ablations (noCommit+noFetch, +noStore, +noBarrier) to
  // price the bare LDS->MMA skeleton against another kernel's.
  constexpr int kVariants = 8;
  const int values[kVariants] = {16, 1, 2, 4, 8, 6, 14, 15};
  const char* names[kVariants] = {"control", "noBarrier", "noCommit", "noFetch",
                                  "noStore", "noCommitFetch", "noCommitFetchStore",
                                  "noAll"};
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
    else if (variant == 6)
      LaunchAblate<Type, 6>(weights.data(), half_x.data(), y.data(), batch, m, k,
                            nullptr);
    else if (variant == 14)
      LaunchAblate<Type, 14>(weights.data(), half_x.data(), y.data(), batch, m, k,
                             nullptr);
    else if (variant == 15)
      LaunchAblate<Type, 15>(weights.data(), half_x.data(), y.data(), batch, m, k,
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
    for (int i = 0; i < kVariants; ++i) {
      const int idx = (i + round) % kVariants;
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


// With the activations read from global, the B reuse factor is WM: every M-warp
// re-reads the same activation fragments. The LDS-era tile sweep picked w8n4
// because LDS reads and A/B symmetry made the aspect near-irrelevant, but with B
// in global a low-WM/high-WN layout halves (or quarters) the B load traffic at
// the cost of more LDS reads for A, which are cheap.
template<gufo::core::GgmlType Type, int WM, int WN, int Ablate>
void LaunchGbA(const void* w, const short* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream,
               const short* blocked) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, WM, WN, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, Ablate>
      <<<grid, WM * WN * 32, 0, stream>>>(w, x, y, batch, m, k, nullptr, blocked);
}

template<gufo::core::GgmlType Type>
void RunGbA(std::size_t m, std::size_t k, std::size_t batch, unsigned iterations,
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
  const unsigned repack_blocks =
      static_cast<unsigned>(((k / 16) * batch + 255) / 256);
  RepackBlockedB<<<repack_blocks, 256>>>(half_x.data(), blocked.data(),
                                         static_cast<int>(batch),
                                         static_cast<int>(k));
  HIP_CHECK(hipDeviceSynchronize());
  // arms: full(w8n4, LDS) ; globalB at w8n4 / w4n8 / w2n16 ; noStoreB floor
  const char* names[5] = {"full_w8n4", "gb_w8n4", "gb_w4n8", "gb_w2n16",
                          "noStoreB"};
  const auto launch = [&](int v) {
    const int A = 16 | 1048576;
    switch (v) {
      case 1: LaunchGbA<Type, 8, 4, A>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, blocked.data()); break;
      case 2: LaunchGbA<Type, 4, 8, A>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, blocked.data()); break;
      case 3: LaunchGbA<Type, 2, 16, A>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, blocked.data()); break;
      case 4: LaunchGbA<Type, 8, 4, 16400>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, nullptr); break;
      default: LaunchGbA<Type, 8, 4, 16>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, nullptr); break;
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
    for (int i = 0; i < 5; ++i) {
      const int idx = (i + round) % 5;
      const double ms = timed(idx);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("gba type=%s arm=%s round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchGbA(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                 unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunGbA<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ5_K:
      RunGbA<GgmlType::kQ5_K>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "gba sweep covers q4k, q5k\n");
      return;
  }
}


// Spend the LDS freed by ablation 1048576 on K depth for the weight tile alone.
// At BM=BN=256 the weight tile is BK * RS * 16 * 16 halves, so BK=8 is exactly
// the 64 KiB budget and the unused activation array is elided. Doubling BK halves
// the stage count, and with it the barriers and the per-stage commit overhead, at
// unchanged total byte traffic.
template<gufo::core::GgmlType Type, int WM, int WN, int BKDepth, int Ablate,
         bool Dbl = false>
void LaunchGb8(const void* w, const short* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream,
               const short* blocked) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, WM, WN, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   BKDepth, Ablate, -1, Dbl>
      <<<grid, WM * WN * 32, 0, stream>>>(w, x, y, batch, m, k, nullptr, blocked);
}

template<gufo::core::GgmlType Type>
void RunGb8(std::size_t m, std::size_t k, std::size_t batch, unsigned iterations,
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
  const unsigned repack_blocks =
      static_cast<unsigned>(((k / 16) * batch + 255) / 256);
  RepackBlockedB<<<repack_blocks, 256>>>(half_x.data(), blocked.data(),
                                         static_cast<int>(batch),
                                         static_cast<int>(k));
  HIP_CHECK(hipDeviceSynchronize());
  const char* names[5] = {"full_bk4", "gb_bk4", "gb_bk8", "gb_bk8_w4n8",
                          "noStoreB_bk4"};
  const auto launch = [&](int v) {
    const int A = 16 | 1048576;
    switch (v) {
      case 1: LaunchGb8<Type, 8, 4, 4, A>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, blocked.data()); break;
      case 2: LaunchGb8<Type, 8, 4, 8, A>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, blocked.data()); break;
      case 3: LaunchGb8<Type, 4, 8, 8, A>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, blocked.data()); break;
      case 4: LaunchGb8<Type, 8, 4, 4, 16400>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, nullptr); break;
      default: LaunchGb8<Type, 8, 4, 4, 16>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, nullptr); break;
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
    for (int i = 0; i < 5; ++i) {
      const int idx = (i + round) % 5;
      const double ms = timed(idx);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("gb8 type=%s arm=%s round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchGb8(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                 unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunGb8<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ5_K:
      RunGb8<GgmlType::kQ5_K>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "gb8 sweep covers q4k, q5k\n");
      return;
  }
}


// The configuration freed LDS makes possible: with only the weight tile staged,
// BK=4 has two 32 KiB buffers, so the commit for stage i+1 can be issued before
// stage i's K loop at full K depth instead of the BK=2 the shared budget forced
// earlier. Compare against the single-buffered BK=8 form that spends the same
// 64 KiB on depth instead of on a second stage.
template<gufo::core::GgmlType Type>
void RunGbDb(std::size_t m, std::size_t k, std::size_t batch,
             unsigned iterations, int pattern) {
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
  const unsigned repack_blocks =
      static_cast<unsigned>(((k / 16) * batch + 255) / 256);
  RepackBlockedB<<<repack_blocks, 256>>>(half_x.data(), blocked.data(),
                                         static_cast<int>(batch),
                                         static_cast<int>(k));
  HIP_CHECK(hipDeviceSynchronize());
  const int A = 16 | 1048576;
  const char* names[5] = {"full_bk4", "gb_bk4", "gbdb_bk4", "gb_bk8_w4n8",
                          "noStoreB"};
  const auto launch = [&](int v) {
    switch (v) {
      case 1: LaunchGb8<Type, 8, 4, 4, A>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, blocked.data()); break;
      case 2: LaunchGb8<Type, 8, 4, 4, A, true>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, blocked.data()); break;
      case 3: LaunchGb8<Type, 4, 8, 8, A>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, blocked.data()); break;
      case 4: LaunchGb8<Type, 8, 4, 4, 16400>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, nullptr); break;
      default: LaunchGb8<Type, 8, 4, 4, 16>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, nullptr); break;
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
  for (int round = 0; round < 5; ++round) {
    for (int i = 0; i < 5; ++i) {
      const int idx = (i + round) % 5;
      const double ms = timed(idx);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("gbdb type=%s arm=%s round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchGbDb(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                  unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunGbDb<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ5_K:
      RunGbDb<GgmlType::kQ5_K>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "gbdb sweep covers q4k, q5k\n");
      return;
  }
}


// Decode cost, measured with the LDS instruction counts held fixed. All three
// arms emit 48 LDS loads and 4 LDS stores per thread per stage, so any delta is
// the decode itself rather than an array that dead-code elimination removed:
//
//   fp32  16          decode in fp32 (mul, sub, two converts)
//   fp16  8388624     the same decode through one fp16 fused multiply-add
//   none  80          decode skipped, value kept live with an opaque asm
template<gufo::core::GgmlType Type, int Ablate>
void LaunchDec(const void* w, const short* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, Ablate>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr);
}

template<gufo::core::GgmlType Type>
void RunDec(std::size_t m, std::size_t k, std::size_t batch, unsigned iterations,
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
  const int values[3] = {16, 16 | 8388608, 80};
  const char* names[3] = {"fp32", "fp16", "none"};
  const auto launch = [&](int v) {
    if (v == (16 | 8388608))
      LaunchDec<Type, 16 | 8388608>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr);
    else if (v == 80)
      LaunchDec<Type, 80>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr);
    else
      LaunchDec<Type, 16>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr);
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
  for (int round = 0; round < 5; ++round) {
    for (int i = 0; i < 3; ++i) {
      const int idx = (i + round) % 3;
      const double ms = timed(values[idx]);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("dec type=%s arm=%s ablate=%d round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], values[idx], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchDec(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                 unsigned iterations, int pattern) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunDec<GgmlType::kQ4_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ5_K:
      RunDec<GgmlType::kQ5_K>(m, k, batch, iterations, pattern);
      return;
    case GgmlType::kQ6_K:
      RunDec<GgmlType::kQ6_K>(m, k, batch, iterations, pattern);
      return;
    default:
      std::fprintf(stderr, "dec sweep covers q4k, q5k, q6k\n");
      return;
  }
}


// Materialised fp16 weights. DequantWeightsToFp16 fills the fp16 tensor using the
// kernel's own LoadRaw/DecodeRaw and the same fp32 scale/offset math the commit
// uses, so the Fp16W kernel and the quantised kernel must produce bit-identical
// output -- that is the correctness gate, not a tolerance. The 178 MiB tensor is
// a per-layer buffer in practice, not a whole-model one.
template<gufo::core::GgmlType Type>
__global__ void DequantWeightsToFp16(const void* __restrict__ w,
                                     short* __restrict__ out, int m, int k) {
  const int nk = k / 16;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int kk = idx % nk, row = idx / nk;
  if (row >= m) return;
  const auto* rowp =
      static_cast<const std::uint8_t*>(w) +
      static_cast<std::size_t>(row) * gufo::hip::QuantRowBytes(Type, k);
  gufo::hip::RawWeights<Type> raw = gufo::hip::LoadRaw<Type>(rowp, (unsigned)kk);
  gufo::hip::QuantSub16 q = gufo::hip::DecodeRaw<Type>(raw);
  union {
    gufo::hip::v16bf f;
    uint4 u[2];
  } packed;
#pragma unroll
  for (int j = 0; j < 16; ++j)
    packed.f[j] = static_cast<_Float16>(__fsub_rn(
        __fmul_rn(q.scale, static_cast<float>(q.q[j])), q.offset));
  uint4* dst = reinterpret_cast<uint4*>(
      out + (static_cast<std::size_t>(row) * nk + kk) * 16);
  dst[0] = packed.u[0];
  dst[1] = packed.u[1];
}

template<gufo::core::GgmlType Type, int Ablate>
void LaunchFp16W(const void* w, const short* x, float* y, std::size_t batch,
                 std::size_t m, std::size_t k, hipStream_t stream,
                 const short* w16) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, Ablate>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr,
                                  reinterpret_cast<const void*>(w16));
}

template<gufo::core::GgmlType Type>
void RunFp16W(std::size_t m, std::size_t k, std::size_t batch,
              unsigned iterations, int pattern, int only_arm) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k);
  DeviceBuffer<short> half_x(batch * k);
  DeviceBuffer<short> w16(m * k);
  DeviceBuffer<float> y_a(batch * m), y_b(batch * m);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);
  {
    const unsigned n = static_cast<unsigned>((m * (k / 16) + 255) / 256);
    DequantWeightsToFp16<Type><<<n, 256>>>(weights.data(), w16.data(),
                                           static_cast<int>(m),
                                           static_cast<int>(k));
    HIP_CHECK(hipDeviceSynchronize());
  }
  // Correctness: the same shape through both kernels, compared exactly.
  LaunchFp16W<Type, 16>(weights.data(), half_x.data(), y_a.data(), batch, m, k,
                        nullptr, nullptr);
  LaunchFp16W<Type, 16 | 16777216>(weights.data(), half_x.data(), y_b.data(),
                                   batch, m, k, nullptr, w16.data());
  HIP_CHECK(hipDeviceSynchronize());
  {
    std::vector<float> ha(batch * m), hb(batch * m);
    HIP_CHECK(hipMemcpy(ha.data(), y_a.data(), sizeof(float) * ha.size(),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hb.data(), y_b.data(), sizeof(float) * hb.size(),
                        hipMemcpyDeviceToHost));
    double worst = 0.0;
    std::size_t mismatch = 0, nonfinite = 0;
    for (std::size_t i = 0; i < ha.size(); ++i) {
      if (!std::isfinite(ha[i]) || !std::isfinite(hb[i])) { ++nonfinite; continue; }
      const double d = std::fabs(double(ha[i]) - double(hb[i]));
      if (d > worst) worst = d;
      if (ha[i] != hb[i]) ++mismatch;
    }
    std::printf("fp16w check: worst_abs=%.6g bit_mismatch=%zu/%zu nonfinite=%zu\n",
                worst, mismatch, ha.size(), nonfinite);
    std::fflush(stdout);
  }
  const auto timed = [&](int v) {
    for (unsigned i = 0; i < 2; ++i)
      if (v == 0)
        LaunchFp16W<Type, 16>(weights.data(), half_x.data(), y_a.data(), batch, m, k, nullptr, nullptr);
      else
        LaunchFp16W<Type, 16 | 16777216>(weights.data(), half_x.data(), y_b.data(), batch, m, k, nullptr, w16.data());
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i)
      if (v == 0)
        LaunchFp16W<Type, 16>(weights.data(), half_x.data(), y_a.data(), batch, m, k, nullptr, nullptr);
      else
        LaunchFp16W<Type, 16 | 16777216>(weights.data(), half_x.data(), y_b.data(), batch, m, k, nullptr, w16.data());
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  const char* names[2] = {"quantised", "fp16_weights"};
  if (only_arm >= 0) {
    // -r 1 single arm for the paired protocol: warm inside, one timed run.
    const double ms = timed(only_arm);
    std::printf("fp16w arm=%s ms=%.6f tflops=%.2f\n", names[only_arm], ms,
                (2.0 * m * k * batch) / (ms * 1e9));
    return;
  }
  for (int round = 0; round < 5; ++round) {
    for (int i = 0; i < 2; ++i) {
      const int idx = (i + round) % 2;
      const double ms = timed(idx);
      const double tflops = (2.0 * static_cast<double>(m) * k * batch) /
                            (ms * 1e9);
      std::printf("fp16w type=%s arm=%s round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], round, ms, tflops);
      std::fflush(stdout);
    }
  }
}

void DispatchFp16W(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                   unsigned iterations, int pattern, int only_arm) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunFp16W<GgmlType::kQ4_K>(m, k, batch, iterations, pattern, only_arm);
      return;
    case GgmlType::kQ5_K:
      RunFp16W<GgmlType::kQ5_K>(m, k, batch, iterations, pattern, only_arm);
      return;
    case GgmlType::kQ6_K:
      RunFp16W<GgmlType::kQ6_K>(m, k, batch, iterations, pattern, only_arm);
      return;
    case GgmlType::kQ8_0:
      RunFp16W<GgmlType::kQ8_0>(m, k, batch, iterations, pattern, only_arm);
      return;
    default:
      std::fprintf(stderr, "fp16w sweep covers q4k, q5k, q6k, q8_0\n");
      return;
  }
}


// Single-arm runner, so one configuration can be profiled per process. The
// counter suite on this part only returns one PMC per run, and a multi-arm mode
// would have to be untangled by dispatch order; one arm per process removes that.
template<gufo::core::GgmlType Type, int Ablate>
void LaunchOne(const void* w, const short* x, float* y, std::size_t batch,
               std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, Ablate>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr);
}

template<gufo::core::GgmlType Type, int BM, int BN, int WM, int WN, int BKDepth>
void LaunchOneT(const void* w, const short* x, float* y, std::size_t batch,
                std::size_t m, std::size_t k, hipStream_t stream, int ablate) {
  const dim3 grid(static_cast<unsigned>((batch + BN - 1) / BN),
                  static_cast<unsigned>((m + BM - 1) / BM));
  switch (ablate) {
    case 16:
      gufo::hip::HalfPrefillGemmKernel<BM, BN, WM, WN, Type,
                                       gufo::hip::HalfEpilogue::kStore, false,
                                       Type, BKDepth, 16>
          <<<grid, WM * WN * 32, 0, stream>>>(w, x, y, batch, m, k, nullptr);
      break;
    case 16432:
      gufo::hip::HalfPrefillGemmKernel<BM, BN, WM, WN, Type,
                                       gufo::hip::HalfEpilogue::kStore, false,
                                       Type, BKDepth, 16432>
          <<<grid, WM * WN * 32, 0, stream>>>(w, x, y, batch, m, k, nullptr);
      break;
    default:
      std::fprintf(stderr, "one: tile run supports ablate 16 / 16432 only\n");
      break;
  }
}

template<gufo::core::GgmlType Type>
void RunOne(std::size_t m, std::size_t k, std::size_t batch, unsigned iterations,
            int pattern, int ablate, int tile_sel) {
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
  const int tile = tile_sel;
  const auto launch = [&] {
    if (tile == 1) { LaunchOneT<Type, 256, 256, 8, 4, 2>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, ablate); return; }
    if (tile == 2) { LaunchOneT<Type, 128, 128, 4, 4, 4>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, ablate); return; }
    if (tile == 3) { LaunchOneT<Type, 128, 128, 8, 4, 4>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, ablate); return; }
    // Wide token tile: 128+384 = 512 is exactly the LDS budget at BK=4, and the
    // total decode work is proportional to 1/BN because every block decodes its
    // own weight tile. Trading BM for BN therefore buys decode amortisation at
    // the cost of operand reuse (WRS=2, WTS=3 -> 0.83 loads/WMMA against 0.75).
    if (tile == 4) { LaunchOneT<Type, 128, 384, 4, 8, 4>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, ablate); return; }
    // tile 5: the shipped 256x256 tile at 512 threads instead of 1024. WRS and
    // WTS both become 4, so LDS operand reads per WMMA fall 0.75 -> 0.50, and
    // each thread carries 16 accumulator chains instead of 8. The tile search
    // that shipped w8n4 excluded this split on the grounds that 16 accumulators
    // "exceed the VGPR budget"; measured from the compiled object it is 250 VGPR
    // against the 256 the 512-thread budget allows, with zero spills, so the
    // exclusion was on a premise that is false and the arm was never timed.
    if (tile == 5) { LaunchOneT<Type, 256, 256, 4, 4, 4>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr, ablate); return; }
    switch (ablate) {
      case 16: LaunchOne<Type, 16>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 48: LaunchOne<Type, 48>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 80: LaunchOne<Type, 80>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 1: LaunchOne<Type, 1>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 2: LaunchOne<Type, 2>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 4: LaunchOne<Type, 4>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 32784: LaunchOne<Type, 32784>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 16400: LaunchOne<Type, 16400>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 16432: LaunchOne<Type, 16432>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 524304: LaunchOne<Type, 524304>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      case 8388624: LaunchOne<Type, 8388624>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      // Scheduling placements, all of which move work between phases rather than
      // changing its amount: 272 puts the global fetch inside the K loop, 528 puts
      // the decode inside it, 1040 staggers the decode across warps.
      case 272: LaunchOne<Type, 272>(weights.data(), half_x.data(), y.data(), batch, m, k, nullptr); break;
      // 512/1024 assert the plain deferred-decode path, which Q5_K is not (it
      // caches the block header), so the instantiation is gated by type instead
      // of breaking the build for that one.
      case 528:
        if constexpr (Type == gufo::core::GgmlType::kQ4_K ||
                      Type == gufo::core::GgmlType::kQ6_K ||
                      Type == gufo::core::GgmlType::kQ8_0)
          LaunchOne<Type, 528>(weights.data(), half_x.data(), y.data(), batch, m,
                               k, nullptr);
        break;
      case 1040:
        if constexpr (Type == gufo::core::GgmlType::kQ4_K ||
                      Type == gufo::core::GgmlType::kQ6_K ||
                      Type == gufo::core::GgmlType::kQ8_0)
          LaunchOne<Type, 1040>(weights.data(), half_x.data(), y.data(), batch, m,
                                k, nullptr);
        break;
      default: std::fprintf(stderr, "one: unsupported ablate %d\n", ablate); break;
    }
  };
  for (unsigned i = 0; i < 2; ++i) launch();
  HIP_CHECK(hipDeviceSynchronize());
  const auto start = std::chrono::steady_clock::now();
  for (unsigned i = 0; i < iterations; ++i) launch();
  HIP_CHECK(hipDeviceSynchronize());
  const auto stop = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(stop - start).count() / iterations;
  std::printf("one type=%s ablate=%d ms=%.6f tflops=%.2f\n", TypeName(Type),
              ablate, ms, (2.0 * m * k * batch) / (ms * 1e9));
}


// GlobalA: stage only the activations in LDS and read the weights straight from a
// pre-blocked fp16 buffer, layout [k/16][m][16]. Halves the LDS allocation to
// 32 KiB, which is what lets a second block be resident, without changing the
// tile or BK. Positions are the mirror of GlobalB: weights are shared by WN
// warps and activations by WM, so the cheaper-to-duplicate operand goes to
// global.
template<gufo::core::GgmlType Type>
__global__ void DequantWeightsToFp16Blocked(const void* __restrict__ w,
                                            short* __restrict__ out, int m, int k) {
  const int nk = k / 16;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int kk = idx % nk, row = idx / nk;
  if (row >= m) return;
  const auto* rowp =
      static_cast<const std::uint8_t*>(w) +
      static_cast<std::size_t>(row) * gufo::hip::QuantRowBytes(Type, k);
  gufo::hip::RawWeights<Type> raw = gufo::hip::LoadRaw<Type>(rowp, (unsigned)kk);
  gufo::hip::QuantSub16 q = gufo::hip::DecodeRaw<Type>(raw);
  union {
    gufo::hip::v16bf f;
    uint4 u[2];
  } packed;
#pragma unroll
  for (int j = 0; j < 16; ++j)
    packed.f[j] = static_cast<_Float16>(__fsub_rn(
        __fmul_rn(q.scale, static_cast<float>(q.q[j])), q.offset));
  uint4* dst = reinterpret_cast<uint4*>(
      out + (static_cast<std::size_t>(kk) * m + row) * 16);
  dst[0] = packed.u[0];
  dst[1] = packed.u[1];
}

template<gufo::core::GgmlType Type, int Ablate>
void LaunchGlobalA(const void* w, const short* x, float* y, std::size_t batch,
                   std::size_t m, std::size_t k, hipStream_t stream,
                   const short* wblk) {
  const dim3 grid(static_cast<unsigned>((batch + 255) / 256),
                  static_cast<unsigned>((m + 255) / 256));
  gufo::hip::HalfPrefillGemmKernel<256, 256, 8, 4, Type,
                                   gufo::hip::HalfEpilogue::kStore, false, Type,
                                   4, Ablate>
      <<<grid, 1024, 0, stream>>>(w, x, y, batch, m, k, nullptr,
                                  reinterpret_cast<const void*>(wblk));
}

template<gufo::core::GgmlType Type>
void RunGlobalA(std::size_t m, std::size_t k, std::size_t batch,
                unsigned iterations, int pattern, int only_arm) {
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(Type, gufo::quant::QuantizedBlockElements(Type));
  const std::size_t weight_bytes = gufo::quant::QuantizedRowBytes(Type, k) * m;
  const auto [d_offset, dmin_offset] = WeightScaleOffsets(Type);
  DeviceBuffer<unsigned char> weights(weight_bytes);
  DeviceBuffer<float> fp32_x(batch * k);
  DeviceBuffer<short> half_x(batch * k);
  DeviceBuffer<short> wblk((k / 16) * m * 16);
  DeviceBuffer<float> y_a(batch * m), y_b(batch * m);
  FillWeights<<<static_cast<unsigned>((weight_bytes / block_bytes + 255) / 256),
                256>>>(weights.data(), weight_bytes, block_bytes, d_offset,
                        dmin_offset, pattern);
  FillValues<<<static_cast<unsigned>((fp32_x.size() + 255) / 256), 256>>>(
      fp32_x.data(), fp32_x.size());
  gufo::hip::LaunchFloatToFp16(fp32_x.data(), half_x.data(), fp32_x.size(),
                               nullptr);
  {
    const unsigned n = static_cast<unsigned>((m * (k / 16) + 255) / 256);
    DequantWeightsToFp16Blocked<Type><<<n, 256>>>(weights.data(), wblk.data(),
                                                  static_cast<int>(m),
                                                  static_cast<int>(k));
    HIP_CHECK(hipDeviceSynchronize());
  }
  const auto run = [&](int arm, float* dst) {
    if (arm == 0)
      LaunchGlobalA<Type, 16>(weights.data(), half_x.data(), dst, batch, m, k,
                              nullptr, nullptr);
    else
      LaunchGlobalA<Type, 16 | 33554432>(weights.data(), half_x.data(), dst,
                                         batch, m, k, nullptr, wblk.data());
  };
  if (only_arm < 0) {
    run(0, y_a.data());
    run(1, y_b.data());
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> ha(batch * m), hb(batch * m);
    HIP_CHECK(hipMemcpy(ha.data(), y_a.data(), sizeof(float) * ha.size(),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hb.data(), y_b.data(), sizeof(float) * hb.size(),
                        hipMemcpyDeviceToHost));
    double worst = 0.0; std::size_t mismatch = 0, nonfinite = 0;
    for (std::size_t i = 0; i < ha.size(); ++i) {
      if (!std::isfinite(ha[i]) || !std::isfinite(hb[i])) { ++nonfinite; continue; }
      const double d = std::fabs(double(ha[i]) - double(hb[i]));
      if (d > worst) worst = d;
      if (ha[i] != hb[i]) ++mismatch;
    }
    std::printf("globalA check: worst_abs=%.6g bit_mismatch=%zu/%zu nonfinite=%zu\n",
                worst, mismatch, ha.size(), nonfinite);
    std::fflush(stdout);
  }
  const auto timed = [&](int arm) {
    for (unsigned i = 0; i < 2; ++i) run(arm, y_a.data());
    HIP_CHECK(hipDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < iterations; ++i) run(arm, y_a.data());
    HIP_CHECK(hipDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           iterations;
  };
  if (only_arm >= 0) {
    const double ms = timed(only_arm);
    std::printf("globalA type=%s arm=%s ms=%.6f tflops=%.2f\n", TypeName(Type),
                only_arm == 0 ? "quantised" : "globalA", ms,
                (2.0 * m * k * batch) / (ms * 1e9));
    return;
  }
  const char* names[2] = {"quantised", "globalA"};
  for (int round = 0; round < 5; ++round) {
    for (int i = 0; i < 2; ++i) {
      const int idx = (i + round) % 2;
      const double ms = timed(idx);
      std::printf("globalA type=%s arm=%s round=%d ms=%.6f tflops=%.2f\n",
                  TypeName(Type), names[idx], round, ms,
                  (2.0 * m * k * batch) / (ms * 1e9));
      std::fflush(stdout);
    }
  }
}

void DispatchGlobalA(GgmlType type, std::size_t m, std::size_t k,
                     std::size_t batch, unsigned iterations, int pattern,
                     int only_arm) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunGlobalA<GgmlType::kQ4_K>(m, k, batch, iterations, pattern, only_arm);
      return;
    case GgmlType::kQ5_K:
      RunGlobalA<GgmlType::kQ5_K>(m, k, batch, iterations, pattern, only_arm);
      return;
    case GgmlType::kQ6_K:
      RunGlobalA<GgmlType::kQ6_K>(m, k, batch, iterations, pattern, only_arm);
      return;
    default:
      std::fprintf(stderr, "globalA sweep covers q4k, q5k, q6k\n");
      return;
  }
}

void DispatchOne(GgmlType type, std::size_t m, std::size_t k, std::size_t batch,
                 unsigned iterations, int pattern, int ablate, int tile_sel) {
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      RunOne<GgmlType::kQ4_K>(m, k, batch, iterations, pattern, ablate, tile_sel);
      return;
    case GgmlType::kQ5_K:
      RunOne<GgmlType::kQ5_K>(m, k, batch, iterations, pattern, ablate, tile_sel);
      return;
    case GgmlType::kQ6_K:
      RunOne<GgmlType::kQ6_K>(m, k, batch, iterations, pattern, ablate, tile_sel);
      return;
    case GgmlType::kQ8_0:
      RunOne<GgmlType::kQ8_0>(m, k, batch, iterations, pattern, ablate, tile_sel);
      return;
    default:
      std::fprintf(stderr, "one: covers q4k, q5k, q6k, q8_0\n");
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
  const bool gba = name.rfind("gba:", 0) == 0;
  if (gba) name = name.substr(4);
  const bool gb8 = name.rfind("gb8:", 0) == 0;
  if (gb8) name = name.substr(4);
  const bool gbdb = name.rfind("gbdb:", 0) == 0;
  if (gbdb) name = name.substr(5);
  const bool decsweep = name.rfind("dec:", 0) == 0;
  if (decsweep) name = name.substr(4);
  const bool fp16w = name.rfind("fp16w:", 0) == 0;
  if (fp16w) name = name.substr(6);
  const bool one = name.rfind("one:", 0) == 0;
  if (one) name = name.substr(4);
  const bool globalA = name.rfind("globalA:", 0) == 0;
  if (globalA) name = name.substr(8);
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
        else if (gba)
          DispatchGbA(entry.type, m, k, batch, iterations, pattern);
        else if (gb8)
          DispatchGb8(entry.type, m, k, batch, iterations, pattern);
        else if (gbdb)
          DispatchGbDb(entry.type, m, k, batch, iterations, pattern);
        else if (decsweep)
          DispatchDec(entry.type, m, k, batch, iterations, pattern);
        else if (fp16w)
          DispatchFp16W(entry.type, m, k, batch, iterations, pattern,
                        argc > 8 ? std::stoi(argv[8]) : -1);
        else if (one)
          DispatchOne(entry.type, m, k, batch, iterations, pattern,
                      argc > 8 ? std::stoi(argv[8]) : 16,
                      argc > 9 ? std::stoi(argv[9]) : 0);
        else if (globalA)
          DispatchGlobalA(entry.type, m, k, batch, iterations, pattern,
                          argc > 8 ? std::stoi(argv[8]) : -1);
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
