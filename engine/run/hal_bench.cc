// hal_bench: time one GEMM-shaped HAL repeatedly, isolated from the driver.
//
// usage: hal_bench <hal> <W> <IN> <OUT> <WST> <OST> <gx> <gy> <gz> [m_rows] [k_blocks] [tokens] [iters] [wfile]
//
// The caller supplies the operand sizes. Do NOT compute them by hand: run this
// through tools/safe_bench.py, which derives them from the shape AND checks them
// against the compiled kernel's own declared footprint before anything is
// dispatched here.
//
// Why: on this target an extent past the end of an allocation does not fault. The
// shader reads unmapped VA, never returns, and hangs with no page fault for the
// driver to report: gfx times out, MES stops answering msg=RESET, and the box
// resets. Three sizing mistakes on 2026-09-29 cost two reboots. Two of them were
// formulas that looked right and were not -- an input sized for k_blocks=20
// (655360 B) reused at k_blocks=68, and an output of m_rows*tokens*4 when a
// k_split=4 residual writes m_rows*k_split*tokens*4. The compiler knew both; the
// arithmetic did not. Verify first, dispatch second.
//
// The grid is checked before anything is submitted (exit 3) when the shape is
// given: gx*16 <= m_rows, gz <= k_blocks, gy <= tokens. K splits live on gz
// (workgroups(m_tiles, token_tiles, k_split)), so folding a split into gx walks
// the m origin off the end of the weight.
//
// Data is deterministic but not degenerate: the weight and grid carry a byte
// pattern (all-0x01 weights make d a subnormal and collapse the grid lookup) and
// the activation is a run of f16 0.5. Even so this measures one kernel on a quiet
// GPU: it is a hypothesis generator, not a result. Confirm with a paired
// full-pipeline A/B (engine/run/LOOM_RUNTIME.md section 6).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
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
  // Optional real weight blob. The IQ grid index the decode looks up is a function
  // of the stored bytes, so a synthetic pattern drives a different, flatter access
  // distribution than the real tensor does -- and the 1088 geometry is sensitive to
  // exactly that. Feed a real tensor when the question is about the production case.
  const char* WFILE = argc > 14 ? argv[14] : "";
  // Optional output dump. With it this stops being only a timing harness: two
  // HALs that claim to compute the same tile can be dispatched on identical
  // inputs and their raw outputs compared elementwise, which localises a
  // wrong-but-plausible kernel to specific (row, token) cells instead of paying
  // a full 2048-token forward per hypothesis. The output is column-major:
  // out[token * m_rows + row]. See tools/hal_oracle.py.
  const char* OUTFILE = argc > 15 ? argv[15] : "";

  int bad = 0;
  if (!W || !IN || !OUT || !WST || !OST) {
    std::fprintf(stderr, "W, IN, OUT, WST and OST must be non-zero\n");
    bad = 1;
  }
  if (!GX || (MROWS && static_cast<size_t>(GX) * 16 > MROWS)) {
    std::fprintf(stderr, "gx=%u covers %zu rows but m_rows=%zu (K splits go on gz)\n",
                 GX, static_cast<size_t>(GX) * 16, MROWS);
    bad = 1;
  }
  if (!GZ || (KBLK && GZ > KBLK)) {
    std::fprintf(stderr, "gz=%u must be in 1..%zu\n", GZ, KBLK);
    bad = 1;
  }
  // The per-workgroup token tile is 64 in the raw source and 16 once the emitter
  // narrows it, so this can only check the weak invariant; safe_bench knows which
  // source it is looking at and enforces gy == token_tiles exactly.
  if (!GY || (TOK && GY > TOK)) {
    std::fprintf(stderr, "gy=%u is out of range for tokens=%zu\n", GY, TOK);
    bad = 1;
  }
  if (iters < 1 || iters > 100000) { std::fprintf(stderr, "iters out of range\n"); bad = 1; }
  if (bad) {
    std::fprintf(stderr, "hal_bench: REFUSING to dispatch; nothing was submitted.\n");
    return 3;
  }

  const size_t GRID = 2048;  // the IQ grid table is exactly 512 x i32
  const size_t KSIGNS = 128; // the ksigns table, one byte per sign slot
  std::fprintf(stderr,
               "hal_bench: grid=%ux%ux%u weight=%zu input=%zu output=%zu iters=%d\n",
               GX, GY, GZ, W, IN, OUT, iters);

  LoomDevice gpu;
  LoomExecutable e = gpu.Load(hal);
  LoomBuffer weight = gpu.Allocate(W), grid = gpu.Allocate(GRID),
             ksigns = gpu.Allocate(KSIGNS),
             input = gpu.Allocate(IN), wstage = gpu.Allocate(WST),
             ostage = gpu.Allocate(OST), output = gpu.Allocate(OUT);
  std::vector<uint8_t> hw(W), hg(GRID), hk(KSIGNS);
  FillPattern(hw);
  FillPattern(hg);
  FillPattern(hk);
  if (WFILE[0]) {
    FILE* wf = std::fopen(WFILE, "rb");
    if (!wf) { std::fprintf(stderr, "hal_bench: cannot open %s\n", WFILE); return 2; }
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
  // Opt-in: pre-fill the output with zeros so a cell the kernel never writes is
  // distinguishable from one it computed. A freshly allocated device buffer is
  // not zeroed, so without this an UNWRITTEN cell reads as whatever the previous
  // allocation left there -- often a NaN pattern, which compares "different"
  // against everything (NaN != NaN) and makes an oracle report 100% disagreement
  // between two bit-identical kernels. Set this before reading anything into a
  // comparison.
  if (std::getenv("YAH_BENCH_ZERO_OUT")) {
    size_t scratch = (WST > OST) ? WST : OST;
    std::vector<uint8_t> zeros(scratch, 0);
    gpu.H2D(output, zeros.data(), OUT);
    // wstage and ostage are scratch the kernel reads back, so an unwritten cell
    // there is read as garbage -- and the shader never returns an error for it,
    // it computes a NaN that then lands in the output. Zeroing them separates
    // "the kernel computed this" from "the kernel read something never written".
    gpu.H2D(wstage, zeros.data(), WST);
    gpu.H2D(ostage, zeros.data(), OST);
  }

  const char* name = e.names.empty() ? "?" : e.names[0].c_str();
  const uint32_t ordinal = e.OrdinalOrZero(name);
  // The executable's own workgroup size, not a hardcoded 32. A wave64 kernel
  // launched with 32 threads runs half a wavegroup: it reads as a large
  // speedup because half the output tile is never computed.
  const uint32_t ws = e.WorkgroupSize(ordinal);
  auto cfg = LoomDevice::Config(GX, GY, GZ, ws ? ws : 32, 1, 1);
  // The binding list is the export's, not a fixed six. The IQ grid/signs
  // formats carry extra leading operands (iq3xxs/iq2xxs/iq2xs: weight, grid,
  // ksigns = 7) and every other format carries none (weight, input, wstage,
  // ostage, out = 5), while the driver builds exactly the matching list
  // (engine/run/loom_forward_pp.cc run_kstore). Hardcoding six rejected the
  // entire non-grid family with "expected 5 but got 6".
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
  // One dispatch at a time up front: if the first hangs it shows up here rather
  // than more dispatches deep into an already wedged ring.
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
  for (int i = 0; i < iters; ++i)
    gpu.Dispatch(e, ordinal, cfg, nullptr, 0, b.data(), b.size());
  gpu.Synchronize();
  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
          .count() / iters;
  std::printf("%-28s gx=%-5u gy=%-3u gz=%-3u wg=%-3u %.4f ms\n", name, GX, GY,
              GZ, ws ? ws : 32, ms);
  return 0;
}
