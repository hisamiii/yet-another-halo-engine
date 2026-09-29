#ifndef GUFO_MODELS_QWEN_HIP_OPS_PREFILL_FP16_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_PREFILL_FP16_HPP_
#include <cstddef>

#include "src/core/gguf_reader.hpp"
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
namespace gufo::hip {
// The half buffers contain IEEE FP16, not BF16. Callers reuse the existing
// scratch allocations after their BF16 contents are no longer live.
void LaunchFloatToFp16(const float* input, void* output, std::size_t elements,
                       hipStream_t stream);
// Optional residual and sum_out preserve the FP32 residual link; sum_out may
// alias input. The normalization reduction matches LaunchBatchedRMSNorm.
// The FP16 output must be disjoint from the FP32 inputs and weights.
void LaunchBatchedRMSNormFp16(const float* input, const float* residual,
                              const float* weight, float* sum_out, void* output,
                              std::size_t batch, std::size_t dim, float eps,
                              hipStream_t stream);
// Diagnostic only: rounds an FP16 activation buffer in place onto a bfp16
// shared-exponent grid with "bits" magnitude bits, emulating the NPU operand
// encoder. A "bits" outside [1, 9] is a no-op, so the default path is
// untouched.
void LaunchBfp16RoundTripFp16InPlace(void* buffer, std::size_t count, int bits,
                                     hipStream_t stream);
// Runtime repack of one quantised weight tensor [n_out, k] into the ATB bfp16 B
// operand, written to out. Removes the need to hold a packed copy of the model:
// the staging buffer is reused per layer. Needs k divisible by 64 and n_out by
// 128, and out to hold n_out * k * 9 / 8 bytes.
void LaunchAtbRepackBfp16(const void* weights, core::GgmlType type,
                          std::size_t n_out, std::size_t k, void* out,
                          hipStream_t stream);
// ATB split support. The NPU consumes bfp16 A and B operands and emits bfp16 C
// in L1 tiles; these are the GPU-side transforms, each verified against the
// vendored host layout code. Rows and n_tiles must be multiples of the ATB L1
// tiles (128 and 64 for A, 128 for B and C), which the callers guarantee by
// construction.
// A: FP16 activations [rows, k] into the ATB A operand, rows * k * 9 / 8 bytes.
void LaunchAtbEncodeAFp16(const void* act_fp16, std::size_t rows, std::size_t k,
                          void* out, hipStream_t stream);
// B: only the n tiles covering [n_offset, n_offset + n_tiles * 128), numbered
// from zero within the slice, which is what the ATB kernel's N refers to.
void LaunchAtbRepackBfp16Slice(const void* weights, core::GgmlType type,
                               std::size_t n_out, std::size_t k,
                               std::size_t n_offset, std::size_t n_tiles,
                               void* out, hipStream_t stream);
// C: the packed slice back into rows of a full-width buffer.
void LaunchAtbDecodeCFp32(const void* packed, std::size_t rows,
                          std::size_t n_slice, std::size_t n_full,
                          std::size_t n_offset, float* out, hipStream_t stream);
void LaunchAtbDecodeCFp16(const void* packed, std::size_t rows,
                          std::size_t n_slice, std::size_t n_full,
                          std::size_t n_offset, void* out, hipStream_t stream);
// Moves the packed head of a partial-width FP16 projection into the first
// columns of a full-width row, for the GPU's share of a split projection.
void LaunchAtbExpandHeadFp16(const void* packed, void* out, std::size_t rows,
                             std::size_t head_cols, std::size_t full_cols,
                             hipStream_t stream);
// The FP32 equivalent for the down projection, whose GPU share is a packed
// partial that has to be added into the full-width residual rows.
void LaunchAtbAddHeadFp32(const float* packed, float* out, std::size_t rows,
                          std::size_t head_cols, std::size_t full_cols,
                          hipStream_t stream);
// C accumulated onto an existing FP32 row, for a projection whose residual the
// GPU branch already wrote.
void LaunchAtbDecodeCAccumulateFp32(const void* packed, std::size_t rows,
                                    std::size_t n_slice, std::size_t n_full,
                                    std::size_t n_offset, float* out,
                                    hipStream_t stream);
// Fused: gate and up packed slices in, FP16 SwiGLU out. Gate and up share a
// layout, so this replaces two decodes plus a separate activation pass.
void LaunchAtbDecodeSwiGLUFp16(const void* gate_packed, const void* up_packed,
                               std::size_t rows, std::size_t n_slice,
                               std::size_t n_full, std::size_t n_offset,
                               void* out, hipStream_t stream);
// Packed GGUF weights are scaled in FP32, rounded to FP16 inside the kernel,
// then multiplied by FP16 activations with FP32 accumulation in K16 order.
// Supports the Qwen27B Q4 shard's native quant formats and K divisible by 256.
void LaunchBatchedQuantGEMMFp16(core::GgmlType type, const void* weights,
                                const void* input, float* output,
                                std::size_t batch, std::size_t m, std::size_t k,
                                hipStream_t stream);
// Fuses the up projection and SwiGLU. gate is FP32; output is FP16 and must
// not alias input. Its storage may reuse the otherwise dead FP32 up buffer.
void LaunchBatchedQuantGEMMSwiGLUFp16(core::GgmlType type, const void* weights,
                                      const void* input, const float* gate,
                                      void* output, std::size_t batch,
                                      std::size_t m, std::size_t k,
                                      hipStream_t stream);
// Adds the completed FP32 dot product to residual in place. Input and residual
// must not alias; accumulation starts at zero, as in the standalone GEMM.
void LaunchBatchedQuantGEMMResidualFp16(core::GgmlType type,
                                        const void* weights, const void* input,
                                        float* residual, std::size_t batch,
                                        std::size_t m, std::size_t k,
                                        hipStream_t stream);
// Qualified gate/up weight pairs share a kernel that emits FP16 SwiGLU
// directly. Returns false without launching for other pairs.
// All buffers are disjoint; the output may reuse the dead FP32 up allocation.
bool TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
    core::GgmlType gate_type, core::GgmlType up_type, const void* gate_weights,
    const void* up_weights, const void* input, void* output, std::size_t batch,
    std::size_t m, std::size_t k, hipStream_t stream);
}  // namespace gufo::hip
#endif
#endif
