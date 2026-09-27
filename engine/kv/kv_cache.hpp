#ifndef YAH_KV_KV_CACHE_HPP_
#define YAH_KV_KV_CACHE_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#endif

namespace yah::kv {

// Storage width of the KV cache. The arithmetic stays FP16; only the cached
// bytes are narrowed, so a wider option can be added without touching the
// attention kernel. kQ8 is a Q8_0-shaped block (fp16 scale, 32 signed bytes)
// and kQ4 a Q4_0-shaped block (fp16 scale, 32 signed nibbles).
enum class KvStorage : std::uint8_t { kF16, kQ8, kQ4 };

inline constexpr std::size_t kKvBlock = 32;

struct KvQ8Block {
  std::uint16_t d;  // fp16 scale
  std::int8_t qs[kKvBlock];
};
static_assert(sizeof(KvQ8Block) == 34, "KvQ8Block must be 34 bytes");

struct KvQ4Block {
  std::uint16_t d;  // fp16 scale
  std::uint8_t qs[kKvBlock / 2];
};
static_assert(sizeof(KvQ4Block) == 18, "KvQ4Block must be 18 bytes");

[[nodiscard]] constexpr std::size_t KvBlockBytes(KvStorage storage) {
  switch (storage) {
    case KvStorage::kF16:
      return kKvBlock * sizeof(std::uint16_t);
    case KvStorage::kQ8:
      return sizeof(KvQ8Block);
    case KvStorage::kQ4:
      return sizeof(KvQ4Block);
  }
  return 0;
}

// Bytes one cache row of `elements` takes, rounded up to whole blocks.
[[nodiscard]] constexpr std::size_t KvRowBytes(KvStorage storage,
                                              std::size_t elements) {
  const std::size_t blocks = (elements + kKvBlock - 1) / kKvBlock;
  return blocks * KvBlockBytes(storage);
}

[[nodiscard]] constexpr std::size_t KvCacheBytes(KvStorage storage,
                                                std::size_t elements) {
  return KvRowBytes(storage, elements);
}

// Orthonormal Walsh-Hadamard matrix of size dim x dim, row-major. It is
// symmetric and H*H = I, so the same matrix rotates and unrotates, and
// rotating Q and K identically leaves their dot product unchanged: the only
// effect is to spread a per-channel outlier over the whole head dimension
// before the 4-bit block quantizer sees it. The reference engine does this for
// every quantized KV cache with a head dimension that is a multiple of 64.
[[nodiscard]] inline std::vector<float> GenerateHadamard(std::size_t dim) {
  if (dim == 0 || (dim & (dim - 1)) != 0) {
    throw std::runtime_error("kv: hadamard dimension is not a power of two");
  }
  std::vector<float> h(dim * dim, 0.0F);
  h[0] = 1.0F;
  for (std::size_t n = 1; n < dim; n *= 2) {
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        const float value = h[(i * dim) + j];
        h[(i * dim) + (j + n)] = value;
        h[((i + n) * dim) + j] = value;
        h[((i + n) * dim) + (j + n)] = -value;
      }
    }
  }
  const float scale = 1.0F / std::sqrt(static_cast<float>(dim));
  for (float& value : h) value *= scale;
  return h;
}

namespace detail {

#if defined(ENGINE_ENABLE_HIP)
// The device pack/unpack uses the platform's fp16 primitive, so the reference
// uses the same one. The quantizer under test is the amax, scale, and code
// math, not the half conversion.
inline std::uint16_t FloatToHalfBits(float value) {
  return __half_as_ushort(__float2half_rn(value));
}

inline float HalfBitsToFloat(std::uint16_t half) {
  return __half2float(__ushort_as_half(half));
}

#else

inline std::uint16_t FloatToHalfBits(float value) {
  const std::uint32_t bits = [](float f) {
    std::uint32_t u = 0;
    __builtin_memcpy(&u, &f, sizeof(u));
    return u;
  }(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
  const std::uint32_t mantissa = bits & 0x7FFFFFU;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<std::uint16_t>(sign);
    const std::uint32_t hidden = mantissa | 0x800000U;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
    const std::uint32_t half = hidden >> shift;
    const std::uint32_t round_bit = (hidden >> (shift - 1)) & 1U;
    return static_cast<std::uint16_t>(sign | (half + round_bit));
  }
  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7C00U);
  // Round to nearest, ties to even, matching __float2half_rn.
  std::uint32_t half = (mantissa + 0x0FFFU + ((mantissa >> 13) & 1U)) >> 13;
  if (half >= 0x400U) {
    half = 0;
    ++exponent;
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  return static_cast<std::uint16_t>(sign |
                                    (static_cast<std::uint32_t>(exponent) << 10) |
                                    half);
}

inline float HalfBitsToFloat(std::uint16_t half) {
  const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000U) << 16;
  std::uint32_t exponent = (half >> 10) & 0x1FU;
  std::uint32_t mantissa = half & 0x3FFU;
  if (exponent == 0) {
    if (mantissa == 0) {
      const std::uint32_t out = sign;
      float f = 0.0F;
      __builtin_memcpy(&f, &out, sizeof(f));
      return f;
    }
    exponent = 1;
    while ((mantissa & 0x400U) == 0) {
      mantissa <<= 1;
      --exponent;
    }
    mantissa &= 0x3FFU;
  } else if (exponent == 31) {
    const std::uint32_t out = sign | 0x7F800000U | (mantissa << 13);
    float f = 0.0F;
    __builtin_memcpy(&f, &out, sizeof(f));
    return f;
  }
  const std::uint32_t out = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  float f = 0.0F;
  __builtin_memcpy(&f, &out, sizeof(f));
  return f;
}

#endif  // ENGINE_ENABLE_HIP

}  // namespace detail

// Host reference: split `row` into fixed blocks, scale to the full signed
// range of the block, and write the packed block. `elements` must be a
// multiple of kKvBlock.
inline void QuantizeKvRowHost(const float* row, std::size_t elements,
                              KvStorage storage, std::uint8_t* out) {
  if (elements % kKvBlock != 0) {
    throw std::runtime_error("kv: row is not a whole number of blocks");
  }
  const std::size_t blocks = elements / kKvBlock;
  for (std::size_t b = 0; b < blocks; ++b) {
    const float* src = row + (b * kKvBlock);
    float amax = 0.0F;
    for (std::size_t i = 0; i < kKvBlock; ++i) {
      amax = std::max(amax, std::fabs(src[i]));
    }
    std::uint8_t* dst = out + (b * KvBlockBytes(storage));
    if (storage == KvStorage::kF16) {
      auto* half = reinterpret_cast<std::uint16_t*>(dst);
      for (std::size_t i = 0; i < kKvBlock; ++i) {
        half[i] = detail::FloatToHalfBits(src[i]);
      }
      continue;
    }
    const int levels = storage == KvStorage::kQ8 ? 127 : 7;
    const float scale = amax > 0.0F ? amax / static_cast<float>(levels) : 0.0F;
    const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
    if (storage == KvStorage::kQ8) {
      auto* block = reinterpret_cast<KvQ8Block*>(dst);
      block->d = detail::FloatToHalfBits(scale);
      for (std::size_t i = 0; i < kKvBlock; ++i) {
        const long q = std::lround(src[i] * inverse);
        block->qs[i] = static_cast<std::int8_t>(std::clamp(q, -127L, 127L));
      }
      continue;
    }
    auto* block = reinterpret_cast<KvQ4Block*>(dst);
    block->d = detail::FloatToHalfBits(scale);
    for (std::size_t i = 0; i < kKvBlock; ++i) {
      const long q = std::lround(src[i] * inverse);
      const auto nibble = static_cast<std::uint8_t>(std::clamp(q, -8L, 7L) & 0x0F);
      if ((i & 1U) == 0U) {
        block->qs[i / 2] = static_cast<std::uint8_t>(block->qs[i / 2] & 0xF0U) | nibble;
      } else {
        block->qs[i / 2] = static_cast<std::uint8_t>((block->qs[i / 2] & 0x0FU) | (nibble << 4U));
      }
    }
  }
}

inline void DequantizeKvRowHost(const std::uint8_t* in, std::size_t elements,
                                KvStorage storage, float* row) {
  if (elements % kKvBlock != 0) {
    throw std::runtime_error("kv: row is not a whole number of blocks");
  }
  const std::size_t blocks = elements / kKvBlock;
  for (std::size_t b = 0; b < blocks; ++b) {
    const std::uint8_t* src = in + (b * KvBlockBytes(storage));
    float* dst = row + (b * kKvBlock);
    if (storage == KvStorage::kF16) {
      const auto* half = reinterpret_cast<const std::uint16_t*>(src);
      for (std::size_t i = 0; i < kKvBlock; ++i) {
        dst[i] = detail::HalfBitsToFloat(half[i]);
      }
      continue;
    }
    if (storage == KvStorage::kQ8) {
      const auto* block = reinterpret_cast<const KvQ8Block*>(src);
      const float scale = detail::HalfBitsToFloat(block->d);
      for (std::size_t i = 0; i < kKvBlock; ++i) {
        dst[i] = scale * static_cast<float>(block->qs[i]);
      }
      continue;
    }
    const auto* block = reinterpret_cast<const KvQ4Block*>(src);
    const float scale = detail::HalfBitsToFloat(block->d);
    for (std::size_t i = 0; i < kKvBlock; ++i) {
      const std::uint8_t byte = block->qs[i / 2];
      const int raw = (i & 1U) == 0U ? (byte & 0x0FU) : (byte >> 4U);
      const int value = raw >= 8 ? raw - 16 : raw;
      dst[i] = scale * static_cast<float>(value);
    }
  }
}

}  // namespace yah::kv

#endif  // YAH_KV_KV_CACHE_HPP_
