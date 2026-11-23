// The memory an Engine will take, estimated before it loads anything. On the APU the GPU's buffers are system RAM:
// an engine that does not fit makes the kernel's OOM killer end processes, other programs included.
#ifndef YAH_MODEL_MEMORY_PLAN_HPP_
#define YAH_MODEL_MEMORY_PLAN_HPP_

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "core/gguf.hpp"
#include "model/loom_runtime.hpp"

namespace yah::model {

struct MemoryPlan {
  static constexpr std::uint64_t kGB = 1000ull * 1000 * 1000;
  // left to the rest of the system: the desktop and the page cache it needs
  static constexpr std::uint64_t kReserve = 1500ull * 1000 * 1000;

  std::uint64_t weights = 0, buffers = 0, available = 0;
  std::uint32_t context = 0;

  [[nodiscard]] std::uint64_t need() const { return weights + buffers; }
  [[nodiscard]] bool fits() const { return need() + kReserve <= available; }

  // model: the model file; decode_hal: its decode set (context and KV format); npu_set: the prefill set whose NPU split
  // runs, empty without the NPU.
  static MemoryPlan For(const std::string& model, const std::string& decode_hal, const std::string& npu_set) {
    std::ifstream f(decode_hal + "/decode.txt");
    if (!f) throw LoomError("cannot read " + decode_hal + "/decode.txt: not a decode set (tools/emit_decode.py)");
    std::uint32_t context = 0, kb = 16, vb = 16;
    for (std::string line; std::getline(f, line);) {
      std::istringstream in(line);
      std::string key, kind;
      in >> key;
      if (key == "ctx") in >> context;
      if (key == "kv") in >> kind >> kb >> vb;
    }
    return For(model, context, kb, vb, npu_set);
  }
  // The same for a context and KV bits (K, V: 16, 8 or 4) before the sets exist.
  static MemoryPlan For(const std::string& model, std::uint32_t context, std::uint32_t kb, std::uint32_t vb,
                        const std::string& npu_set) {
    MemoryPlan p;
    p.context = context;
    const core::Gguf index = core::Gguf::OpenIndex(model);
    for (const auto& t : index.tensors()) p.weights += t.bytes;
    // Fitted to the lowest MemAvailable on gfx1151 with the native UD-IQ4_XS sets, kv8 (2026-11-23): yah_server --npu
    // over a load and 3 requests 19.26 / 19.68 / 21.83 GB at 16K / 32K / 64K (the estimate at or above each); the NPU
    // split's own buffers (GEMM + attention) 1.39 GB (loom_forward_pp 32K, with and without --npu). What grows with the
    // context is the KV pools (K and V of 16 layers x 1024 per token, at the format's bits plus scales) and f32 / f16
    // attention scratch.
    const std::uint64_t kv = 16ull * 1024 * (kb + vb) / 8 * 33 / 32;
    p.buffers = kFixed + std::uint64_t{p.context} * (kv + kScratchPerToken);
    if (!npu_set.empty()) {  // the NPU's own rows, read into memory (pre-decoded: npu_bfp_*)
      std::error_code ec;
      for (const auto& e : std::filesystem::directory_iterator(npu_set, ec)) {
        const std::string n = e.path().filename().string();
        if (n.starts_with("npu_bfp_")) p.buffers += e.file_size(ec);
      }
      p.buffers += kNpuFixed;
    }
    p.available = Available();
    return p;
  }

  [[nodiscard]] std::string Describe() const {
    char s[200];
    std::snprintf(s, sizeof s, "needs ~%.1f GB (weights %.1f, buffers %.1f at %u context), %.1f GB available",
                  double(need()) / kGB, double(weights) / kGB, double(buffers) / kGB, context, double(available) / kGB);
    return s;
  }

  // MemAvailable: free memory plus what the kernel can reclaim without swapping.
  static std::uint64_t Available() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    std::uint64_t kb = 0;
    while (f >> key >> kb) {
      if (key == "MemAvailable:") return kb * 1024;
      f.ignore(64, '\n');
    }
    return ~0ull;  // (no /proc: do not refuse)
  }

 private:
  // kFixed: the chunk-sized buffers, state snapshots, the host side and HRX's own memory; kNpuFixed: the NPU split's
  // command arenas, operand and attention buffers (beside its npu_bfp_ rows)
  static constexpr std::uint64_t kFixed = 1650ull * 1000 * 1000, kScratchPerToken = 16500,
                                 kNpuFixed = 1400ull * 1000 * 1000;
};

}  // namespace yah::model

#endif  // YAH_MODEL_MEMORY_PLAN_HPP_
