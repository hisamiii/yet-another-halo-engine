// GGML block formats to f32 on the CPU, and Q8_0 from f32: what loading a GGUF needs (core/gguf_convert.cpp: token_embd
// and a head the GEMVs cannot read become Q8_0, the small tensors f32).
#ifndef YAH_CORE_GGUF_QUANT_HPP_
#define YAH_CORE_GGUF_QUANT_HPP_

#include <cstdint>

#include "core/gguf.hpp"

namespace yah::core {

// Whether Dequantize reads type t.
[[nodiscard]] bool CanDequantize(GgmlType t);
// n elements (whole blocks of t) at src to f32, element order.
void Dequantize(GgmlType t, const std::uint8_t* src, float* dst, std::uint64_t n);

// n elements (a multiple of 32) to ggml Q8_0.
void QuantizeQ8_0(const float* x, std::uint64_t n, std::uint8_t* dst);

}  // namespace yah::core

#endif  // YAH_CORE_GGUF_QUANT_HPP_
