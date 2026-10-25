// GGML block formats to f32 on the CPU, and the engine's own formats from f32: what loading a GGUF needs
// (core/gguf_convert.cpp) to bring any common quant into the MQ4G256 rotated basis the GPU kernels read.
#ifndef YAH_CORE_GGUF_QUANT_HPP_
#define YAH_CORE_GGUF_QUANT_HPP_

#include <cstdint>

#include "core/gguf.hpp"

namespace yah::core {

// Whether Dequantize reads type t.
[[nodiscard]] bool CanDequantize(GgmlType t);
// n elements (whole blocks of t) at src to f32, element order.
void Dequantize(GgmlType t, const std::uint8_t* src, float* dst, std::uint64_t n);

// One row of k (a multiple of 256) original-basis weights to MQ4G256 as the engine holds it (core/hfq.cpp): rotated
// per 256 group (w' = s2 / 16 H256(s1 w), so w' . rot(x) = w . x for gen_rot's rot), then one f16 scale per 128
// (hipfire's header: d, -8 d) and signed 4-bit codes, element 2i in the low nibble of byte i. Writes k / 256 * 136 B.
// The scale is the best of 16 candidates in [amax / 14, amax / 7] by squared error. w is clobbered.
void QuantizeMq4Row(float* w, std::uint64_t k, std::uint8_t* dst);
// n elements (a multiple of 32) to ggml Q8_0.
void QuantizeQ8_0(const float* x, std::uint64_t n, std::uint8_t* dst);

}  // namespace yah::core

#endif  // YAH_CORE_GGUF_QUANT_HPP_
