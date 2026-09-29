// hal_bench: time one kStore-shaped HAL repeatedly, isolated from the driver.
// usage: hal_bench <hal> <gx> <weight_bytes> <out_bytes> [iters]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include "model/loom_runtime.hpp"
using namespace yah::model;
int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: hal_bench <hal> <gx> <W> <OUT> [iters]\n"); return 2; }
  const std::string hal = argv[1];
  const uint32_t GX = (uint32_t)atoi(argv[2]);
  const size_t W = strtoull(argv[3], nullptr, 10);
  const size_t OUT = strtoull(argv[4], nullptr, 10);
  const int iters = argc > 5 ? atoi(argv[5]) : 50;
  const size_t GRID = 2048, IN = 655360, WST = 65536, OST = 5242880;
  LoomDevice gpu;
  LoomExecutable e = gpu.Load(hal);
  LoomBuffer weight = gpu.Allocate(W), grid = gpu.Allocate(GRID),
             input = gpu.Allocate(IN), wstage = gpu.Allocate(WST),
             ostage = gpu.Allocate(OST), output = gpu.Allocate(OUT);
  std::vector<uint8_t> hw(W, 1), hg(GRID, 0);
  gpu.H2D(weight, hw.data(), W); gpu.H2D(grid, hg.data(), GRID);
  auto cfg = LoomDevice::Config(GX, 1, 1, 32, 1, 1);
  std::vector<hrx_buffer_ref_t> b = {{weight.handle,0,W},{grid.handle,0,GRID},{input.handle,0,IN},
      {wstage.handle,0,WST},{ostage.handle,0,OST},{output.handle,0,OUT}};
  const char* name = e.names.empty() ? "?" : e.names[0].c_str();
  for (int i = 0; i < 3; ++i) gpu.Dispatch(e, e.OrdinalOrZero(name), cfg, nullptr, 0, b.data(), b.size());
  gpu.Synchronize();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) gpu.Dispatch(e, e.OrdinalOrZero(name), cfg, nullptr, 0, b.data(), b.size());
  gpu.Synchronize();
  const double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/iters;
  std::printf("%-28s gx=%-5u %.4f ms\n", name, GX, ms);
  return 0;
}