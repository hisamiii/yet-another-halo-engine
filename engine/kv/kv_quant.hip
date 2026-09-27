// KV cache pack and unpack. One thread owns one 32-element block: it finds
// the block's magnitude, derives the scale, and writes (or reads) the packed
// codes. The math is FP16 on the attention side, so the unpack target is
// always __half and the attention kernels are untouched.
#ifndef YAH_KV_KV_QUANT_HIP_
#define YAH_KV_KV_QUANT_HIP_

#ifndef ENGINE_ENABLE_HIP
#error "kv/kv_quant.hip requires ENGINE_ENABLE_HIP"
#endif

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "kv/kv_cache.hpp"

namespace yah::kv {

namespace {

__device__ __forceinline__ std::uint16_t HalfBits(float value) {
  return __half_as_ushort(__float2half_rn(value));
}

__global__ void QuantizeQ8Kernel(const __half* __restrict__ src,
                                 std::uint8_t* __restrict__ dst,
                                 std::size_t blocks, std::size_t row_blocks) {
  const std::size_t block = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (block >= blocks) return;
  const std::size_t row = block / row_blocks;
  const std::size_t col = block % row_blocks;
  const __half* in = src + (row * row_blocks + col) * kKvBlock;
  float amax = 0.0F;
  for (std::size_t i = 0; i < kKvBlock; ++i) {
    amax = fmaxf(amax, fabsf(__half2float(in[i])));
  }
  const float scale = amax > 0.0F ? amax / 127.0F : 0.0F;
  const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
  auto* block_out = reinterpret_cast<KvQ8Block*>(dst) + block;
  block_out->d = HalfBits(scale);
  for (std::size_t i = 0; i < kKvBlock; ++i) {
    const long q = lroundf(__half2float(in[i]) * inverse);
    block_out->qs[i] = static_cast<std::int8_t>(max(-127L, min(127L, q)));
  }
}

__global__ void QuantizeQ4Kernel(const __half* __restrict__ src,
                                 std::uint8_t* __restrict__ dst,
                                 std::size_t blocks, std::size_t row_blocks) {
  const std::size_t block = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (block >= blocks) return;
  const std::size_t row = block / row_blocks;
  const std::size_t col = block % row_blocks;
  const __half* in = src + (row * row_blocks + col) * kKvBlock;
  float amax = 0.0F;
  for (std::size_t i = 0; i < kKvBlock; ++i) {
    amax = fmaxf(amax, fabsf(__half2float(in[i])));
  }
  const float scale = amax > 0.0F ? amax / 7.0F : 0.0F;
  const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
  auto* block_out = reinterpret_cast<KvQ4Block*>(dst) + block;
  block_out->d = HalfBits(scale);
  for (std::size_t i = 0; i < kKvBlock; ++i) {
    const long q = lroundf(__half2float(in[i]) * inverse);
    const auto nibble =
        static_cast<std::uint8_t>(max(-8L, min(7L, q)) & 0x0F);
    if ((i & 1U) == 0U) {
      block_out->qs[i / 2] =
          static_cast<std::uint8_t>(block_out->qs[i / 2] & 0xF0U) | nibble;
    } else {
      block_out->qs[i / 2] = static_cast<std::uint8_t>(
          (block_out->qs[i / 2] & 0x0FU) | (nibble << 4U));
    }
  }
}

__global__ void DequantizeQ8Kernel(const std::uint8_t* __restrict__ src,
                                   __half* __restrict__ dst,
                                   std::size_t blocks,
                                   std::size_t row_blocks) {
  const std::size_t block = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (block >= blocks) return;
  const std::size_t row = block / row_blocks;
  const std::size_t col = block % row_blocks;
  const auto* block_in = reinterpret_cast<const KvQ8Block*>(src) + block;
  const float scale = __half2float(__ushort_as_half(block_in->d));
  __half* out = dst + (row * row_blocks + col) * kKvBlock;
  for (std::size_t i = 0; i < kKvBlock; ++i) {
    out[i] = __float2half_rn(scale * static_cast<float>(block_in->qs[i]));
  }
}

__global__ void DequantizeQ4Kernel(const std::uint8_t* __restrict__ src,
                                   __half* __restrict__ dst,
                                   std::size_t blocks,
                                   std::size_t row_blocks) {
  const std::size_t block = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (block >= blocks) return;
  const std::size_t row = block / row_blocks;
  const std::size_t col = block % row_blocks;
  const auto* block_in = reinterpret_cast<const KvQ4Block*>(src) + block;
  const float scale = __half2float(__ushort_as_half(block_in->d));
  __half* out = dst + (row * row_blocks + col) * kKvBlock;
  for (std::size_t i = 0; i < kKvBlock; ++i) {
    const std::uint8_t byte = block_in->qs[i / 2];
    const int raw = (i & 1U) == 0U ? (byte & 0x0FU) : (byte >> 4U);
    const int value = raw >= 8 ? raw - 16 : raw;
    out[i] = __float2half_rn(scale * static_cast<float>(value));
  }
}

__global__ void HadamardInPlaceKernel(float* __restrict__ x,
                                      const float* __restrict__ h,
                                      std::size_t dim) {
  extern __shared__ float row[];
  const std::size_t r = blockIdx.x;
  const std::size_t e = threadIdx.x;
  row[e] = x[(r * dim) + e];
  __syncthreads();
  float acc = 0.0F;
  for (std::size_t d = 0; d < dim; ++d) {
    acc += row[d] * h[(d * dim) + e];
  }
  x[(r * dim) + e] = acc;
}

__global__ void HadamardInPlaceHalfKernel(__half* __restrict__ x,
                                          const float* __restrict__ h,
                                          std::size_t dim) {
  extern __shared__ float row[];
  const std::size_t r = blockIdx.x;
  const std::size_t e = threadIdx.x;
  row[e] = __half2float(x[(r * dim) + e]);
  __syncthreads();
  float acc = 0.0F;
  for (std::size_t d = 0; d < dim; ++d) {
    acc += row[d] * h[(d * dim) + e];
  }
  x[(r * dim) + e] = __float2half_rn(acc);
}

void CheckLaunch(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("kv quant: ") + what + ": " +
                             hipGetErrorString(status));
  }
}

}  // namespace

// Applies the orthonormal Hadamard rotation to each row of a [rows, dim]
// fp32 matrix in place. One block per row, `dim` threads, row staged in shared
// memory so the in-place write cannot race the reads.
inline void LaunchHadamardInPlace(float* x, std::size_t rows, std::size_t dim,
                                  const float* h,
                                  hipStream_t stream = nullptr) {
  if (rows == 0) return;
  HadamardInPlaceKernel<<<static_cast<unsigned int>(rows),
                          static_cast<unsigned int>(dim),
                          dim * sizeof(float), stream>>>(x, h, dim);
  CheckLaunch(hipGetLastError(), "hadamard launch");
}

// Same rotation for an fp16 matrix, used for the cached K and V, which live
// in the fp16 scratch.
inline void LaunchHadamardInPlaceHalf(void* x, std::size_t rows,
                                      std::size_t dim, const float* h,
                                      hipStream_t stream = nullptr) {
  if (rows == 0) return;
  HadamardInPlaceHalfKernel<<<static_cast<unsigned int>(rows),
                               static_cast<unsigned int>(dim),
                               dim * sizeof(float), stream>>>(
      static_cast<__half*>(x), h, dim);
  CheckLaunch(hipGetLastError(), "hadamard half launch");
}

// Packs `rows` rows of `row_elements` fp16 values into the packed layout.
// kF16 is a plain copy so the caller has one code path.
inline void LaunchQuantizeKvRows(const void* src_f16, void* dst_packed,
                                 std::size_t rows, std::size_t row_elements,
                                 KvStorage storage,
                                 hipStream_t stream = nullptr) {
  const std::size_t row_bytes = row_elements * sizeof(std::uint16_t);
  if (storage == KvStorage::kF16) {
    CheckLaunch(hipMemcpyAsync(dst_packed, src_f16, rows * row_bytes,
                               hipMemcpyDeviceToDevice, stream),
                "f16 copy");
    return;
  }
  const std::size_t row_blocks = row_elements / kKvBlock;
  const std::size_t blocks = rows * row_blocks;
  const unsigned int block_size = 128;
  const unsigned int grid =
      static_cast<unsigned int>((blocks + block_size - 1) / block_size);
  if (storage == KvStorage::kQ8) {
    QuantizeQ8Kernel<<<grid, block_size, 0, stream>>>(
        static_cast<const __half*>(src_f16),
        static_cast<std::uint8_t*>(dst_packed), blocks, row_blocks);
  } else {
    QuantizeQ4Kernel<<<grid, block_size, 0, stream>>>(
        static_cast<const __half*>(src_f16),
        static_cast<std::uint8_t*>(dst_packed), blocks, row_blocks);
  }
  CheckLaunch(hipGetLastError(), "quantize launch");
}

inline void LaunchDequantizeKvRows(const void* src_packed, void* dst_f16,
                                   std::size_t rows,
                                   std::size_t row_elements,
                                   KvStorage storage,
                                   hipStream_t stream = nullptr) {
  const std::size_t row_bytes = row_elements * sizeof(std::uint16_t);
  if (storage == KvStorage::kF16) {
    CheckLaunch(hipMemcpyAsync(dst_f16, src_packed, rows * row_bytes,
                               hipMemcpyDeviceToDevice, stream),
                "f16 copy");
    return;
  }
  const std::size_t row_blocks = row_elements / kKvBlock;
  const std::size_t blocks = rows * row_blocks;
  const unsigned int block_size = 128;
  const unsigned int grid =
      static_cast<unsigned int>((blocks + block_size - 1) / block_size);
  if (storage == KvStorage::kQ8) {
    DequantizeQ8Kernel<<<grid, block_size, 0, stream>>>(
        static_cast<const std::uint8_t*>(src_packed),
        static_cast<__half*>(dst_f16), blocks, row_blocks);
  } else {
    DequantizeQ4Kernel<<<grid, block_size, 0, stream>>>(
        static_cast<const std::uint8_t*>(src_packed),
        static_cast<__half*>(dst_f16), blocks, row_blocks);
  }
  CheckLaunch(hipGetLastError(), "dequantize launch");
}

}  // namespace yah::kv

#endif  // YAH_KV_KV_QUANT_HIP_
