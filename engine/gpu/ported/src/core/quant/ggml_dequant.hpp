#ifndef GUFO_CORE_QUANT_GGML_DEQUANT_HPP_
#define GUFO_CORE_QUANT_GGML_DEQUANT_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::quant {

// Canonical quantized block layouts. These are the authoritative field
// layouts for the dequant/dot paths. Relocated verbatim from the internal
// definitions in ggml_dequant.cpp so the header is the single source of
// truth; field order and byte sizes match the HIP path
// (src/models/qwen/hip/quant_ops.hpp) and the historical per-file copies
// (models/qwen/state.hpp, tests/). Do NOT redefine block_* locally. Sizes are
// static_asserted; this is the layout contract (see the parity test that
// proves field order against the byte-level spec).
#pragma pack(push, 1)
struct block_q4_K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};

struct block_q5_K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};

struct block_q6_K {
  std::uint8_t ql[128];
  std::uint8_t qh[64];
  std::int8_t scales[16];
  std::uint16_t d;
};

struct block_q3_K {
  std::uint8_t hmask[32];
  std::uint8_t qs[64];
  std::uint8_t scales[12];
  std::uint16_t d;
};

struct block_q8_K {
  float d;
  std::int8_t qs[256];
  std::int16_t bsums[16];
};

// Q8_0: fp16 scale + 32 int8 quantized values (QK=32).
struct block_q8_0 {
  std::uint16_t d;
  std::int8_t qs[32];
};

// IQ4_NL: fp16 scale + 32 non-linear 4-bit codebook indices (QK=32). The
// codebook is kValuesIq4Nl; the index nibble order matches Q4_0 (low nibbles
// give the first 16 elements, high nibbles the second 16).
struct block_iq4_nl {
  std::uint16_t d;
  std::uint8_t qs[16];
};

// IQ4_XS: super-block of 256 with eight 6-bit sub-block scales split across
// scales_h (2 high bits each) and scales_l (4 low bits each), plus 128 bytes
// of packed non-linear 4-bit codebook indices.
struct block_iq4_xs {
  std::uint16_t d;
  std::uint16_t scales_h;
  std::uint8_t scales_l[4];
  std::uint8_t qs[128];
};

// IQ3_S: super-block of 256. Each group of 8 elements indexes the 512-entry
// kIq3sGrid table (8 low bits in qs, 1 high bit in qh) and carries a per-group
// sign byte; scales holds two 4-bit sub-block scales per byte.
struct block_iq3_s {
  std::uint16_t d;
  std::uint8_t qs[64];
  std::uint8_t qh[8];
  std::uint8_t signs[32];
  std::uint8_t scales[4];
};

// IQ3_XXS: super-block of 256 (98 bytes). The first 64 bytes are two-bit
// indices into the 256-entry iq3xxs grid; the remaining 32 are
// scales_and_signs -- four 7-bit sign indices plus one 4-bit scale per group of
// 32 elements.
struct block_iq3_xxs {
  std::uint16_t d;
  std::uint8_t qs[96];
};

// Q2_K: super-block of 256 (84 bytes). Sixteen 16-element sub-blocks, each
// with a 4-bit scale in the low nibble and a 4-bit min in the high nibble of
// one scales byte; d and dmin are the super-block scale and min.
struct block_q2_K {
  std::uint8_t scales[16];
  std::uint8_t qs[64];
  std::uint16_t d;
  std::uint16_t dmin;
};
static_assert(sizeof(block_q2_K) == 84, "block_q2_K must be 84 bytes");

// IQ2_XXS: super-block of 256 (66 bytes). Each group of 32 elements is four
// 8-bit iq2xxs grid indices followed by one word holding the 4-bit scale and
// four 7-bit sign indices.
struct block_iq2_xxs {
  std::uint16_t d;
  std::uint16_t qs[32];
};
static_assert(sizeof(block_iq2_xxs) == 66, "block_iq2_xxs must be 66 bytes");

// IQ2_XS: super-block of 256 (74 bytes). 32 uint16 codes, each a 9-bit index
// into the 512-entry iq2xs grid plus a 7-bit sign index; scales holds two 4-bit
// sub-block scales per byte.
struct block_iq2_xs {
  std::uint16_t d;
  std::uint16_t qs[32];
  std::uint8_t scales[8];
};

// IQ2_S: super-block of 256 (82 bytes). qs[0..31] are 8-bit grid indices whose
// two high bits come from qh; qs[32..63] are the sign bytes; scales as IQ2_XS.
struct block_iq2_s {
  std::uint16_t d;
  std::uint8_t qs[64];
  std::uint8_t qh[8];
  std::uint8_t scales[8];
};
#pragma pack(pop)

static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");
static_assert(sizeof(block_q5_K) == 176, "block_q5_K must be 176 bytes");
static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 bytes");
static_assert(sizeof(block_q3_K) == 110, "block_q3_K must be 110 bytes");
static_assert(sizeof(block_q8_K) == 292, "block_q8_K must be 292 bytes");
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");
static_assert(sizeof(block_iq4_nl) == 18, "block_iq4_nl must be 18 bytes");
static_assert(sizeof(block_iq4_xs) == 136, "block_iq4_xs must be 136 bytes");
static_assert(sizeof(block_iq3_s) == 110, "block_iq3_s must be 110 bytes");
static_assert(sizeof(block_iq3_xxs) == 98, "block_iq3_xxs must be 98 bytes");
static_assert(sizeof(block_iq2_xs) == 74, "block_iq2_xs must be 74 bytes");
static_assert(sizeof(block_iq2_s) == 82, "block_iq2_s must be 82 bytes");

/// Non-linear 4-bit codebook shared by IQ4_NL and IQ4_XS.
inline constexpr std::int8_t kValuesIq4Nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

/// Returns the logical elements represented by one supported quant block, or
/// zero for dense and unsupported storage types.
[[nodiscard]] constexpr std::size_t QuantizedBlockElements(
    core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kQ4_0:
    case core::GgmlType::kQ4_1:
    case core::GgmlType::kQ5_0:
    case core::GgmlType::kQ5_1:
    case core::GgmlType::kQ8_1:
    case core::GgmlType::kQ8_0:
    case core::GgmlType::kIQ4_NL:
      return 32;
    case core::GgmlType::kQ2_K:
    case core::GgmlType::kIQ2_XXS:
    case core::GgmlType::kIQ2_XS:
    case core::GgmlType::kIQ3_XXS:
    case core::GgmlType::kIQ2_S:
    case core::GgmlType::kIQ4_XS:
    case core::GgmlType::kIQ3_S:
    case core::GgmlType::kQ3_K:
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
    case core::GgmlType::kQ8_K:
      return 256;
    default:
      return 0;
  }
}

/// Returns the encoded byte count for one logical quantized row, or zero when
/// the type is not quantized or the element count is not block aligned.
[[nodiscard]] std::size_t QuantizedRowBytes(core::GgmlType type,
                                            std::size_t elements) noexcept;

/// Returns the physical byte count for a tensor with `elements` logical
/// elements, or zero when the format is unsupported/misaligned.
[[nodiscard]] std::size_t EncodedSizeBytes(core::GgmlType type,
                                           std::size_t elements) noexcept;

// Standard 16-bit float helper
float Fp16ToFloat(std::uint16_t h) noexcept;

// Dequantize row of Q4_K to float
void DequantizeQ4_K(const void* src, float* dst, std::size_t k);

// Dequantize row of Q5_K to float
void DequantizeQ5_K(const void* src, float* dst, std::size_t k);

// Dequantize row of Q6_K to float
void DequantizeQ6_K(const void* src, float* dst, std::size_t k);

// Dequantize row of Q3_K to float
void DequantizeQ3_K(const void* src, float* dst, std::size_t k);

// Dequantize row of Q8_K to float
void DequantizeQ8_K(const void* src, float* dst, std::size_t k);

// Compute dot product of quantized row with FP32 vector
float DotProductQ4_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);
// Compute dot product of Q5_K quantized row with FP32 vector
float DotProductQ5_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);
float DotProductQ6_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);
float DotProductQ3_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);
float DotProductQ8_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);

// Dequantize row of Q8_0 to float
void DequantizeQ8_0(const void* src, float* dst, std::size_t k);

// Dequantize row of IQ4_NL to float
void DequantizeIQ4_NL(const void* src, float* dst, std::size_t k);

// Dequantize row of IQ4_XS to float
void DequantizeIQ4_XS(const void* src, float* dst, std::size_t k);

// Dequantize row of IQ3_S to float
void DequantizeIQ3_S(const void* src, float* dst, std::size_t k);

// Dequantize row of IQ3_XXS to float
void DequantizeIQ3_XXS(const void* src, float* dst, std::size_t k);

// Dequantize row of Q2_K to float
void DequantizeQ2_K(const void* src, float* dst, std::size_t k);

// Dequantize row of IQ2_XXS to float
void DequantizeIQ2_XXS(const void* src, float* dst, std::size_t k);

// Dequantize row of IQ2_XS to float
void DequantizeIQ2_XS(const void* src, float* dst, std::size_t k);

// Dequantize row of IQ2_S to float
void DequantizeIQ2_S(const void* src, float* dst, std::size_t k);

// Compute dot product of IQ4_NL quantized row with FP32 vector
float DotProductIQ4_NL(const void* row_data, std::span<const float> vec,
                       std::size_t k);
// Compute dot product of IQ4_XS quantized row with FP32 vector
float DotProductIQ4_XS(const void* row_data, std::span<const float> vec,
                       std::size_t k);
// Compute dot product of IQ3_S quantized row with FP32 vector
float DotProductIQ3_S(const void* row_data, std::span<const float> vec,
                      std::size_t k);
// Compute dot product of IQ3_XXS quantized row with FP32 vector
float DotProductIQ3_XXS(const void* row_data, std::span<const float> vec,
                        std::size_t k);
// Compute dot product of IQ2_XXS quantized row with FP32 vector
float DotProductIQ2_XXS(const void* row_data, std::span<const float> vec,
                        std::size_t k);
// Compute dot product of Q2_K quantized row with FP32 vector
float DotProductQ2_K(const void* row_data, std::span<const float> vec,
                     std::size_t k);
// Compute dot product of IQ2_XS quantized row with FP32 vector
float DotProductIQ2_XS(const void* row_data, std::span<const float> vec,
                       std::size_t k);
// Compute dot product of IQ2_S quantized row with FP32 vector
float DotProductIQ2_S(const void* row_data, std::span<const float> vec,
                      std::size_t k);

/// Returns the 512-entry IQ3_S grid table (each entry packs four uint8
/// magnitudes).
[[nodiscard]] const std::uint32_t* Iq3sGrid() noexcept;

// Compute dot product of Q8_0 quantized row with FP32 vector
float DotProductQ8_0(const void* row_data, std::span<const float> vec,
                     std::size_t k);

}  // namespace gufo::quant

#endif  // GUFO_CORE_QUANT_GGML_DEQUANT_HPP_
