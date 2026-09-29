// KV quantization check. The device pack/unpack must reproduce the host
// reference's pack/unpack, and the round trip must stay inside the error the
// block layout implies (amax/levels). The reference is the definition, not the
// kernel, so a shared bug cannot pass.
#define ENGINE_ENABLE_HIP 1
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kv/kv_cache.hpp"
#include "kv/kv_quant.hip"

using yah::kv::DequantizeKvRowHost;
using yah::kv::KvRowBytes;
using yah::kv::KvStorage;
using yah::kv::QuantizeKvRowHost;

namespace {

float HalfToFloat(std::uint16_t h) {
  const auto* half = reinterpret_cast<const __half*>(&h);
  return __half2float(*half);
}

std::uint16_t FloatToHalf(float f) {
  const __half h = __float2half_rn(f);
  return *reinterpret_cast<const std::uint16_t*>(&h);
}

int CheckOne(KvStorage storage, std::size_t rows, std::size_t cols,
             std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<std::uint16_t> source(rows * cols);
  for (std::size_t i = 0; i < source.size(); ++i) {
    float value = normal(rng);
    if ((i % 257) == 0) value *= 8.0F;  // a few outliers in every row
    source[i] = FloatToHalf(value);
  }

  const std::size_t row_bytes = KvRowBytes(storage, cols);
  const std::size_t packed_bytes = row_bytes * rows;

  void* d_source = nullptr;
  void* d_packed = nullptr;
  void* d_round = nullptr;
  hipMalloc(&d_source, source.size() * sizeof(std::uint16_t));
  hipMalloc(&d_packed, packed_bytes);
  hipMalloc(&d_round, source.size() * sizeof(std::uint16_t));
  hipMemcpy(d_source, source.data(), source.size() * sizeof(std::uint16_t),
            hipMemcpyHostToDevice);
  yah::kv::LaunchQuantizeKvRows(d_source, d_packed, rows, cols, storage);
  yah::kv::LaunchDequantizeKvRows(d_packed, d_round, rows, cols, storage);
  hipDeviceSynchronize();

  std::vector<std::uint16_t> device_round(source.size());
  hipMemcpy(device_round.data(), d_round,
            source.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost);

  std::vector<std::uint8_t> device_packed(packed_bytes, 0);
  hipMemcpy(device_packed.data(), d_packed, packed_bytes,
            hipMemcpyDeviceToHost);

  // Host reference over the same bytes.
  std::vector<float> row(cols);
  std::vector<std::uint8_t> packed(row_bytes);
  std::vector<float> reference(cols);
  std::vector<std::uint8_t> host_packed(packed_bytes, 0);
  double max_reference_delta = 0.0;
  double max_round_error = 0.0;
  double signal = 0.0;
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      row[c] = HalfToFloat(source[r * cols + c]);
    }
    QuantizeKvRowHost(row.data(), cols, storage,
                      host_packed.data() + (r * row_bytes));
    DequantizeKvRowHost(host_packed.data() + (r * row_bytes), cols, storage,
                        reference.data());
    for (std::size_t c = 0; c < cols; ++c) {
      const float device = HalfToFloat(device_round[r * cols + c]);
      // The device dequant emits fp16; round the fp32 reference the same way
      // so the comparison is of the quantizer, not of the output precision.
      const float ref = HalfToFloat(FloatToHalf(reference[c]));
      max_reference_delta =
          std::max<double>(max_reference_delta, std::fabs(device - ref));
      max_round_error =
          std::max<double>(max_round_error, std::fabs(device - row[c]));
      signal = std::max<double>(signal, std::fabs(row[c]));
    }
  }
  std::size_t packed_diff = 0;
  std::size_t first_diff = packed_bytes;
  for (std::size_t i = 0; i < packed_bytes; ++i) {
    if (device_packed[i] != host_packed[i]) {
      if (first_diff == packed_bytes) first_diff = i;
      ++packed_diff;
    }
  }
  hipFree(d_source);
  hipFree(d_packed);
  hipFree(d_round);

  const char* name = storage == KvStorage::kF16 ? "F16"
                     : storage == KvStorage::kQ8 ? "Q8"
                                                 : "Q4";
  const double levels = storage == KvStorage::kF16   ? 2048.0
                        : storage == KvStorage::kQ8 ? 127.0
                                                    : 7.0;
  const double allowed = signal / levels * 1.5;
  const bool reference_ok = max_reference_delta == 0.0;
  const bool round_ok = max_round_error <= allowed;
  std::printf(
      "%-3s rows=%zu cols=%zu bytes/row=%zu ref_delta=%.3e "
      "round_err=%.3e allowed=%.3e %s\n",
      name, rows, cols, row_bytes, max_reference_delta, max_round_error,
      allowed, (reference_ok && round_ok) ? "PASS" : "FAIL");
  if (packed_diff != 0) {
    const std::size_t block = first_diff / KvBlockBytes(storage);
    std::printf("     packed_diff=%zu first=%zu block=%zu dev=[%02x %02x] host=[%02x %02x]\n",
                packed_diff, first_diff, block, device_packed[block * KvBlockBytes(storage)],
                device_packed[(block * KvBlockBytes(storage)) + 1],
                host_packed[block * KvBlockBytes(storage)],
                host_packed[(block * KvBlockBytes(storage)) + 1]);
  }
  return (reference_ok && round_ok && packed_diff == 0) ? 0 : 1;
}


int CheckHadamard(std::size_t dim) {
  const std::vector<float> h = yah::kv::GenerateHadamard(dim);
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> x(dim);
  for (float& v : x) v = normal(rng);
  std::vector<float> y_host(dim, 0.0F);
  for (std::size_t e = 0; e < dim; ++e) {
    for (std::size_t d = 0; d < dim; ++d) {
      y_host[e] += x[d] * h[(d * dim) + e];
    }
  }
  float* d_x = nullptr;
  float* d_h = nullptr;
  hipMalloc(&d_x, dim * sizeof(float));
  hipMalloc(&d_h, h.size() * sizeof(float));
  hipMemcpy(d_x, x.data(), dim * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_h, h.data(), h.size() * sizeof(float),
            hipMemcpyHostToDevice);
  yah::kv::LaunchHadamardInPlace(d_x, 1, dim, d_h);
  std::vector<float> y_dev(dim);
  hipMemcpy(y_dev.data(), d_x, dim * sizeof(float), hipMemcpyDeviceToHost);
  yah::kv::LaunchHadamardInPlace(d_x, 1, dim, d_h);
  std::vector<float> round(dim);
  hipMemcpy(round.data(), d_x, dim * sizeof(float), hipMemcpyDeviceToHost);
  hipFree(d_x);
  hipFree(d_h);
  double err_single = 0.0;
  double err_identity = 0.0;
  for (std::size_t i = 0; i < dim; ++i) {
    err_single = std::max<double>(err_single, std::fabs(y_dev[i] - y_host[i]));
    err_identity = std::max<double>(err_identity, std::fabs(round[i] - x[i]));
  }
  const bool ok = err_single < 1e-4 && err_identity < 1e-4;
  std::printf("HAD dim=%-4zu single=%.3e identity=%.3e %s\n", dim,
              err_single, err_identity, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
  const std::size_t cols = argc > 1 ? std::stoul(argv[1]) : 1024;
  const std::size_t rows = argc > 2 ? std::stoul(argv[2]) : 8;
  if (cols % 32 != 0) {
    std::fprintf(stderr, "columns must be a multiple of 32\n");
    return 2;
  }
  int failures = 0;
  failures += CheckOne(KvStorage::kF16, rows, cols, 1);
  failures += CheckOne(KvStorage::kQ8, rows, cols, 2);
  failures += CheckOne(KvStorage::kQ4, rows, cols, 3);
  failures += CheckHadamard(256);
  if (failures == 0) {
    std::printf("KV QUANT CHECK PASS\n");
    return 0;
  }
  std::printf("KV QUANT CHECK FAIL (%d)\n", failures);
  return 1;
}
