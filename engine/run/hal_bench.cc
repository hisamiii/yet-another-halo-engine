// hal_bench: times one GEMM-shaped HAL repeatedly, isolated from the driver.
//
// usage: hal_bench <hal> <W> <IN> <OUT> <WST> <OST> <gx> <gy> <gz>
//                  [m_rows] [k_blocks] [tokens] [iters] [wfile] [outfile]
//
// Do not compute the operand sizes by hand: run this through tools/safe_bench.py.
// It derives them from the shape and checks them against the compiled kernel's declared footprint.
// On this target an access past an allocation does not fault: the shader hangs, MES stops answering and the box resets.
// Hand formulas look right and are not: e.g. a K-split residual writes m_rows * k_split * tokens * 4 bytes.
//
// With the shape given, the grid is checked before any submit (exit 3): gx*16 <= m_rows, gz <= k_blocks, gy <= tokens.
// K splits live on gz (workgroups(m_tiles, token_tiles, k_split)); a split folded into gx runs past the weight.
//
// Data is deterministic but not degenerate: weight and grid get a byte pattern, the activation is f16 0.5.
// All-0x01 weights make d a subnormal and collapse the grid lookup.
// One kernel on a quiet GPU gives a hypothesis, not a result: confirm with a full-pipeline A/B.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "model/loom_runtime.hpp"
using namespace yah::model;

static uint32_t g_seed = 12345u;
static void FillPattern(std::vector<uint8_t>& v) {
  for (auto& b : v) {
    g_seed = g_seed * 1664525u + 1013904223u;
    b = static_cast<uint8_t>(g_seed >> 24);
  }
}

int main(int argc, char** argv) {
  if (argc < 10) {
    std::fprintf(stderr,
                 "usage: hal_bench <hal> <W> <IN> <OUT> <WST> <OST> <gx> <gy> "
                 "<gz> [m_rows] [k_blocks] [tokens] [iters] [wfile] [outfile]\n");
    return 2;
  }
  const std::string hal = argv[1];
  const size_t W = strtoull(argv[2], nullptr, 10);
  const size_t IN = strtoull(argv[3], nullptr, 10);
  const size_t OUT = strtoull(argv[4], nullptr, 10);
  const size_t WST = strtoull(argv[5], nullptr, 10);
  const size_t OST = strtoull(argv[6], nullptr, 10);
  const uint32_t GX = static_cast<uint32_t>(atoi(argv[7]));
  const uint32_t GY = static_cast<uint32_t>(atoi(argv[8]));
  const uint32_t GZ = static_cast<uint32_t>(atoi(argv[9]));
  const size_t MROWS = argc > 10 ? strtoull(argv[10], nullptr, 10) : 0;
  const size_t KBLK = argc > 11 ? strtoull(argv[11], nullptr, 10) : 0;
  const size_t TOK = argc > 12 ? strtoull(argv[12], nullptr, 10) : 0;
  const int iters = argc > 13 ? atoi(argv[13]) : 50;
  // Optional real weight blob. The IQ grid lookups depend on the stored bytes, and a synthetic pattern gives a flatter
  // access distribution than the real tensor. Use a real tensor to time the production case.
  const char* WFILE = argc > 14 ? argv[14] : "";
  // Optional output dump: run two HALs for the same tile on identical inputs and compare the outputs elementwise.
  // This finds the wrong (row, token) cells without a full forward. Layout: out[token * m_rows + row].
  const char* OUTFILE = argc > 15 ? argv[15] : "";

  int bad = 0;
  if (!W || !IN || !OUT || !WST || !OST) {
    std::fprintf(stderr, "W, IN, OUT, WST and OST must be non-zero\n");
    bad = 1;
  }
  if (!GX || (MROWS && static_cast<size_t>(GX) * 16 > MROWS)) {
    std::fprintf(stderr, "gx=%u covers %zu rows but m_rows=%zu (K splits go on gz)\n", GX, static_cast<size_t>(GX) * 16,
                 MROWS);
    bad = 1;
  }
  if (!GZ || (KBLK && GZ > KBLK)) {
    std::fprintf(stderr, "gz=%u must be in 1..%zu\n", GZ, KBLK);
    bad = 1;
  }
  // This does not know the kernel's token tile, so it checks only gy <= tokens; safe_bench enforces gy == token_tiles.
  if (!GY || (TOK && GY > TOK)) {
    std::fprintf(stderr, "gy=%u is out of range for tokens=%zu\n", GY, TOK);
    bad = 1;
  }
  if (iters < 1 || iters > 100000) {
    std::fprintf(stderr, "iters out of range\n");
    bad = 1;
  }
  if (bad) {
    std::fprintf(stderr, "hal_bench: REFUSING to dispatch; nothing was submitted.\n");
    return 3;
  }

  const size_t GRID = 2048;   // the IQ grid table is exactly 512 x i32
  const size_t KSIGNS = 128;  // the ksigns table, one byte per sign slot
  std::fprintf(stderr, "hal_bench: grid=%ux%ux%u weight=%zu input=%zu output=%zu iters=%d\n", GX, GY, GZ, W, IN, OUT,
               iters);

  LoomDevice gpu;
  LoomExecutable e = gpu.Load(hal);
  LoomBuffer weight = gpu.Allocate(W), grid = gpu.Allocate(GRID), ksigns = gpu.Allocate(KSIGNS),
             input = gpu.Allocate(IN), wstage = gpu.Allocate(WST), ostage = gpu.Allocate(OST),
             output = gpu.Allocate(OUT);
  std::vector<uint8_t> hw(W), hg(GRID), hk(KSIGNS);
  FillPattern(hw);
  FillPattern(hg);
  FillPattern(hk);
  if (WFILE[0]) {
    FILE* wf = std::fopen(WFILE, "rb");
    if (!wf) {
      std::fprintf(stderr, "hal_bench: cannot open %s\n", WFILE);
      return 2;
    }
    const size_t got = std::fread(hw.data(), 1, W, wf);
    std::fclose(wf);
    if (got != W) {
      std::fprintf(stderr, "hal_bench: %s has %zu bytes, need %zu\n", WFILE, got, W);
      return 2;
    }
  }
  std::vector<uint8_t> hin(IN);
  for (size_t i = 0; i + 1 < IN; i += 2) {  // f16 0.5, little endian
    hin[i] = 0x00;
    hin[i + 1] = 0x38;
  }
  gpu.H2D(weight, hw.data(), W);
  gpu.H2D(grid, hg.data(), GRID);
  gpu.H2D(ksigns, hk.data(), KSIGNS);
  gpu.H2D(input, hin.data(), IN);

  const char* name = e.names.empty() ? "?" : e.names[0].c_str();
  const uint32_t ordinal = e.OrdinalOrZero(name);
  // Use the export's workgroup size: a wave64 kernel launched with 32 threads skips half the tile and looks fast.
  const uint32_t ws = e.WorkgroupSize(ordinal);
  auto cfg = LoomDevice::Config(GX, GY, GZ, ws ? ws : 32, 1, 1);
  // Build the binding list from the export's count, as loom_forward_pp.cc run_kstore does.
  // 5: weight, input, wstage, ostage, out. 6 (iq3s): + grid. 7 (iq3xxs/iq2xxs/iq2xs): + grid, ksigns.
  const uint32_t nb = e.BindingCount(ordinal);
  if (nb < 5 || nb > 7) {
    std::fprintf(stderr, "hal_bench: export binds %u buffers; expected 5..7\n", nb);
    return 3;
  }
  std::vector<hrx_buffer_ref_t> b = {{weight.handle, 0, W}};
  if (nb >= 6) b.push_back({grid.handle, 0, GRID});
  if (nb == 7) b.push_back({ksigns.handle, 0, KSIGNS});
  b.push_back({input.handle, 0, IN});
  b.push_back({wstage.handle, 0, WST});
  b.push_back({ostage.handle, 0, OST});
  b.push_back({output.handle, 0, OUT});
  // One dispatch at a time first: a hang then shows here, not several dispatches deep into a wedged ring.
  for (int i = 0; i < 3; ++i) {
    gpu.Dispatch(e, ordinal, cfg, nullptr, 0, b.data(), b.size());
    gpu.Synchronize();
  }
  if (OUTFILE[0]) {
    std::vector<uint8_t> ho(OUT);
    gpu.D2H(output, ho.data(), OUT);
    FILE* of = std::fopen(OUTFILE, "wb");
    if (!of) {
      std::fprintf(stderr, "hal_bench: cannot write %s\n", OUTFILE);
      return 2;
    }
    if (std::fwrite(ho.data(), 1, OUT, of) != OUT) {
      std::fprintf(stderr, "hal_bench: short write to %s\n", OUTFILE);
      std::fclose(of);
      return 2;
    }
    std::fclose(of);
  }
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) gpu.Dispatch(e, ordinal, cfg, nullptr, 0, b.data(), b.size());
  gpu.Synchronize();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
  std::printf("%-28s gx=%-5u gy=%-3u gz=%-3u wg=%-3u %.4f ms\n", name, GX, GY, GZ, ws ? ws : 32, ms);
  return 0;
}
