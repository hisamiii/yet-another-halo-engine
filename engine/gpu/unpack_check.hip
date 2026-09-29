// q+gate unpack check: the fused attention query projection emits
// [head][q(head_dim) | gate(head_dim)] per token, and this splits it into the
// query and the per-element sigmoid gate the attention kernel consumes.
#define ENGINE_ENABLE_HIP 1
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "src/models/qwen/hip/ops/token.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

using gufo::test::DeviceBuffer;

namespace {
__global__ void FillValues(float* values, std::size_t count, float scale) {
  const std::size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < count) {
    values[i] = (static_cast<int>((i * 7919) % 2001) - 1000) * scale;
  }
}
}  // namespace

int main(int argc, char** argv) {
  const std::size_t batch = argc > 1 ? std::stoul(argv[1]) : 64;
  const std::uint32_t heads = 24, head_dim = 256;
  const std::size_t total = static_cast<std::size_t>(heads) * head_dim;

  DeviceBuffer<float> qg(batch * total * 2), q(batch * total),
      gate(batch * total);
  FillValues<<<static_cast<unsigned>((qg.size() + 255) / 256), 256>>>(
      qg.data(), qg.size(), 0.02F);
  gufo::hip::LaunchBatchedUnpackQG(qg.data(), q.data(), gate.data(), batch,
                                   heads, head_dim, nullptr);
  HIP_CHECK(hipDeviceSynchronize());

  const auto qg_host = qg.CopyToHost();
  const auto q_host = q.CopyToHost();
  const auto gate_host = gate.CopyToHost();

  double q_err = 0.0, gate_err = 0.0;
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < heads; ++h) {
      for (std::uint32_t d = 0; d < head_dim; ++d) {
        const std::size_t dst = b * total + h * head_dim + d;
        const std::size_t src = b * total * 2 + h * 2 * head_dim + d;
        q_err = std::max(q_err, std::fabs(static_cast<double>(q_host[dst]) -
                                          qg_host[src]));
        gate_err = std::max(
            gate_err, std::fabs(static_cast<double>(gate_host[dst]) -
                                qg_host[src + head_dim]));
      }
    }
  }
  std::printf("unpack batch=%zu heads=%u head_dim=%u q_err=%.3g gate_err=%.3g\n",
              batch, heads, head_dim, q_err, gate_err);
  const bool ok = q_err == 0.0 && gate_err == 0.0;
  std::printf("%s\n", ok ? "UNPACK CHECK PASS" : "UNPACK CHECK FAIL");
  return ok ? 0 : 1;
}
