#ifndef GUFO_MODELS_QWEN_HIP_ATB_NPU_HPP_
#define GUFO_MODELS_QWEN_HIP_ATB_NPU_HPP_
#include <cstddef>

namespace gufo::hip {

// Which projection a slice engine serves. Gate and up share one shape and so
// one xclbin; down has a different K and N and needs its own, which is legal
// because two ATB hardware contexts coexist on this device.
enum class AtbRole { kGateUp, kDown };

// Geometry of the ATB split. The NPU runs the tail of the FFN gate and up
// output rows; the GPU keeps the head plus the whole down projection, because
// the ATB design bakes K and N into its xclbin and swapping an instruction
// stream from another shape aborts the device queue. One process therefore
// holds exactly one shape: the gate/up slice at K = hidden.
struct AtbSplitGeometry {
  std::size_t batch;      // rows of the A operand
  std::size_t gu_k;       // gate/up reduction width, the hidden size
  std::size_t gu_n_full;  // gate/up output rows, the intermediate size
  std::size_t gu_n_gpu;   // rows [0, gu_n_gpu) stay on the GPU
  // Two ways to divide the projection. The width split gives the NPU output
  // rows [gu_n_gpu, gu_n_full) of every token; the token split gives it tokens
  // [0, batch) at the full width, which is what gu_n_gpu == 0 means. batch is
  // the xclbin's M in both cases, but only the token split allows it to be
  // smaller than the chunk's own batch.
  bool token_split;
};

// Owns the NPU side of the split: the XRT device, the ATB kernel, and the A, B
// and C operands shared with the iGPU through dma-bufs, so no operand is ever
// copied between them. The GPU writes A and B in place and reads C out of the
// same allocations.
//
// Get() returns nullptr when the path is unavailable, which is the default
// unless GUFO_ATB_XCLBIN and GUFO_ATB_INSTS name a matching pair and
// GUFO_ATB_NSLICE agrees with the xclbin.
class AtbNpuOffload {
public:
  // Lazily created singleton per role; nullptr when the NPU path cannot be
  // used.
  static AtbNpuOffload* Get(AtbRole role);

  AtbNpuOffload(const AtbNpuOffload&) = delete;
  AtbNpuOffload& operator=(const AtbNpuOffload&) = delete;
  ~AtbNpuOffload();

  [[nodiscard]] const AtbSplitGeometry& geometry() const { return geometry_; }
  [[nodiscard]] std::size_t n_slice() const {
    return geometry_.gu_n_full - geometry_.gu_n_gpu;
  }
  // Rows [0, gu_n_gpu) are the GPU's share; [gu_n_gpu, gu_n_full) are the
  // NPU's, for gate/up and for down alike.

  // Operands the GPU encoders fill and the decoders read. All are ordinary
  // hipMalloc allocations; the NPU holds them as imported BOs.
  [[nodiscard]] void* activations() const;
  [[nodiscard]] void* gate_weights() const;
  [[nodiscard]] void* up_weights() const;
  [[nodiscard]] void* gate_output() const;
  [[nodiscard]] void* up_output() const;

  // Queues the gate and the up slice. The caller must have completed and
  // synchronised the stream that produced A and B; this does not wait.
  bool Launch();
  // Blocks until both slices have completed.
  bool Wait();

private:
  AtbNpuOffload();
  bool Init(AtbRole role);

  struct Impl;
  Impl* impl_{nullptr};
  AtbSplitGeometry geometry_{};
};
}  // namespace gufo::hip
#endif
