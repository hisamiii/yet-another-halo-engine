// gemm_bench: runs tile-GEMM HAL variants on real weights, one process, for the autotuner (engine/tune).
//
// usage: gemm_bench <model.gguf> <table dir> <job file>
//
// Each job line: <hal> <export> <tensor> <fmt> <kind> <row groups> <tokens per workgroup> <tokens> <reps> <roles>
//   roles: the kernel's bindings in order, comma separated (weight, grid, ksigns, input, gate, resid, wstage, ostage,
//   output, gate_out). The table dir holds grid_<fmt>.bin / ksigns_iq2xxs.bin (any prefill HAL set).
// Every binding is at least as large as loom_forward_pp's largest use of it (chunk 2048), so a HAL that passed the
// emitter's footprint gate is in bounds here too. The grid is (m_tiles / row groups, ceil(tokens / tile)): a subset of
// the compiled grid whenever tokens <= the chunk.
// Per job it prints "job <i> wall_ms <ms per rep> hash <h>": h hashes the output of the real token rows only (variants
// pad differently), after 2 warmup dispatches. Run under HRX_PROFILE_MODE=counters for clock-free cycles: the
// dispatches appear in job order, 2 warmups + reps per job.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "model/loom_runtime.hpp"

namespace {

using yah::model::LoomBuffer;
using yah::model::LoomDevice;
using yah::model::LoomError;
using yah::model::LoomExecutable;

constexpr std::size_t kChunk = 2048, kFfn = 17408;

struct Job {
  std::string hal, exp, tensor, fmt, kind;
  std::uint32_t rowgrp = 0, tile = 0, tokens = 0, reps = 0;
  std::vector<std::string> roles;
};

std::vector<Job> ReadJobs(const std::string& path) {
  std::vector<Job> jobs;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream s(line);
    Job j;
    std::string roles;
    if (!(s >> j.hal >> j.exp >> j.tensor >> j.fmt >> j.kind >> j.rowgrp >> j.tile >> j.tokens >> j.reps >> roles))
      throw LoomError("bad job line: " + line);
    for (std::size_t a = 0, b; a <= roles.size(); a = b + 1) {
      b = roles.find(',', a);
      if (b == std::string::npos) b = roles.size();
      j.roles.push_back(roles.substr(a, b - a));
    }
    jobs.push_back(std::move(j));
  }
  return jobs;
}

// Small finite values: f16 0.5..1 or f32 -1..1 with a fixed sequence, the same for every variant.
std::vector<std::uint8_t> Pattern(std::size_t bytes, bool f16, std::uint32_t seed) {
  std::vector<std::uint8_t> v(bytes);
  std::uint32_t x = seed;
  if (f16) {
    auto* h = reinterpret_cast<std::uint16_t*>(v.data());
    for (std::size_t i = 0; i < bytes / 2; ++i) {
      x = x * 1664525u + 1013904223u;
      h[i] = static_cast<std::uint16_t>(((x >> 16) & 0x8000u) | 0x3800u | ((x >> 8) & 0x3ffu));
    }
  } else {
    auto* g = reinterpret_cast<float*>(v.data());
    for (std::size_t i = 0; i < bytes / 4; ++i) {
      x = x * 1664525u + 1013904223u;
      g[i] = static_cast<float>(static_cast<std::int32_t>(x >> 8) - (1 << 23)) / static_cast<float>(1 << 23);
    }
  }
  return v;
}

std::uint64_t Hash(const std::vector<std::uint8_t>& v, std::uint64_t h = 1469598103934665603ull) {
  for (std::uint8_t c : v) h = (h ^ c) * 1099511628211ull;
  return h;
}

std::vector<std::uint8_t> ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw LoomError("cannot read " + path);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(f), {});
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: gemm_bench <model.gguf> <table dir> <job file>\n");
    return 2;
  }
  try {
    const auto gguf = yah::core::Gguf::Open(argv[1]);
    const std::string tables = argv[2];
    const std::vector<Job> jobs = ReadJobs(argv[3]);
    LoomDevice gpu;
    // Shared bindings, sized for the largest use in loom_forward_pp at chunk 2048.
    LoomBuffer input = gpu.Allocate(kChunk * kFfn * 2), gate = gpu.Allocate(kChunk * kFfn * 4),
               resid = gpu.Allocate(kChunk * kFfn * 4), wstage = gpu.Allocate(kFfn * 16 * 2),
               ostage = gpu.Allocate(kFfn * kChunk * 4), output = gpu.Allocate(kChunk * kFfn * 4),
               gate_out = gpu.Allocate(kChunk * kFfn * 4), ksigns = gpu.Allocate(128);
    gpu.H2D(input, Pattern(input.size, true, 1).data(), input.size);
    gpu.H2D(gate, Pattern(gate.size, false, 2).data(), gate.size);
    gpu.H2D(resid, Pattern(resid.size, false, 3).data(), resid.size);
    {
      const auto k = ReadFile(tables + "/ksigns_iq2xxs.bin");
      if (k.size() != 128) throw LoomError("ksigns_iq2xxs.bin is not 128 bytes");
      gpu.H2D(ksigns, k.data(), 128);
    }
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      const Job& j = jobs[i];
      const auto* t = gguf.Find(j.tensor);
      if (!t || t->dims.size() < 2) throw LoomError("tensor not found: " + j.tensor);
      const std::uint32_t m_rows = static_cast<std::uint32_t>(t->dims[1]);
      if (j.tokens == 0 || j.tokens > kChunk || m_rows % 16 || (m_rows / 16) % j.rowgrp)
        throw LoomError(j.hal + ": bad shape or token count");
      LoomBuffer weight = gpu.Allocate(t->bytes);
      gpu.H2D(weight, gguf.Data(*t), t->bytes);
      LoomBuffer grid;
      const bool needs_grid = j.fmt == "iq3s" || j.fmt == "iq3xxs" || j.fmt == "iq2xxs" || j.fmt == "iq2xs";
      if (needs_grid) {
        const auto g = ReadFile(tables + "/grid_" + j.fmt + ".bin");
        grid = gpu.Allocate(g.size());
        gpu.H2D(grid, g.data(), g.size());
      }
      std::vector<hrx_buffer_ref_t> b;
      for (const std::string& r : j.roles) {
        const LoomBuffer* x = r == "weight"    ? &weight
                              : r == "grid"    ? (needs_grid ? &grid : nullptr)
                              : r == "ksigns"  ? &ksigns
                              : r == "input"   ? &input
                              : r == "gate"    ? &gate
                              : r == "resid"   ? &resid
                              : r == "wstage"  ? &wstage
                              : r == "ostage"  ? &ostage
                              : r == "output"  ? &output
                              : r == "gate_out" ? &gate_out
                                                : nullptr;
        if (!x) throw LoomError(j.hal + ": unknown binding " + r);
        b.push_back({x->handle, 0, x->size});
      }
      LoomExecutable exe = gpu.Load(j.hal);
      const std::uint32_t ord = exe.OrdinalOrZero(j.exp);
      const std::uint32_t ws = exe.WorkgroupSize(ord);
      if (!ws) throw LoomError(j.hal + ": no workgroup size in the export metadata");
      const auto cfg = LoomDevice::Config(m_rows / 16 / j.rowgrp, (j.tokens + j.tile - 1) / j.tile, 1, ws, 1, 1);
      gpu.Fill(output, 0);
      gpu.Fill(gate_out, 0);
      for (int w = 0; w < 2; ++w) gpu.Dispatch(exe, ord, cfg, nullptr, 0, b.data(), b.size());
      gpu.Synchronize();
      // The real token rows of the outputs: [token][rows], f16 for swiglu, q and gate halves for kqg.
      const std::size_t row_bytes = j.kind == "swiglu" ? std::size_t{m_rows} * 2
                                    : j.kind == "kqg"  ? std::size_t{m_rows} / 2 * 4
                                                       : std::size_t{m_rows} * 4;
      std::vector<std::uint8_t> host(row_bytes * j.tokens);
      gpu.D2H(output, host.data(), host.size());
      std::uint64_t h = Hash(host);
      if (j.kind == "kqg") {
        gpu.D2H(gate_out, host.data(), host.size());
        h = Hash(host, h);
      }
      const auto t0 = std::chrono::steady_clock::now();
      for (std::uint32_t r = 0; r < j.reps; ++r) gpu.Dispatch(exe, ord, cfg, nullptr, 0, b.data(), b.size());
      gpu.Synchronize();
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::printf("job %zu wall_ms %.4f hash %016llx\n", i, j.reps ? ms / j.reps : 0.0,
                  static_cast<unsigned long long>(h));
      std::fflush(stdout);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gemm_bench: %s\n", e.what());
    return 1;
  }
  return 0;
}
