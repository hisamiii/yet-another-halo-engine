// Reference harness: run HIP's production DeltaNet kernel
// (BatchedDeltaNetRowSplitKernel<float, 16, 2, false, false>) on inputs from
// .bin files and write its outputs, for a bit-exact comparison with Loom.
#include "src/models/qwen/hip/kernels/ssm_row_split.hip"
#include <cstdio>
#include <vector>
static std::vector<float> rd(const char* p, size_t n) {
  std::vector<float> v(n); FILE* f = fopen(p, "rb"); if (!f || fread(v.data(), 4, n, f) != n) { fprintf(stderr, "read %s\n", p); exit(1); } fclose(f); return v; }
static void wr(const char* p, const std::vector<float>& v) { FILE* f = fopen(p, "wb"); fwrite(v.data(), 4, v.size(), f); fclose(f); }
template <class T> static T* up(const std::vector<float>& v) { T* d; hipMalloc(&d, v.size() * 4); hipMemcpy(d, v.data(), v.size() * 4, hipMemcpyHostToDevice); return d; }
int main(int argc, char** argv) {
  const int B = atoi(argv[1]), QKV = 10240, INNER = 6144, KH = 16, H = 48;
  auto conv = rd("conv.bin", (size_t)B * QKV), kq = rd("kq.bin", (size_t)B * KH * 3),
       ab = rd("ab.bin", (size_t)B * H * 2), st = rd("state.bin", (size_t)H * 16384);
  float *dconv = up<float>(conv), *dkq = up<float>(kq), *dab = up<float>(ab), *dst = up<float>(st);
  std::vector<float> out((size_t)B * INNER, 0.f); float* dout = up<float>(out);
  dim3 g(128 / 64, H);
  hipLaunchKernelGGL((gufo::hip::BatchedDeltaNetRowSplitKernel<float, 16, 2, false, false>), g, dim3(256), 0, 0,
                     dconv, dkq, dab, dst, dout, 0u, B, QKV, INNER, KH, H);
  if (hipDeviceSynchronize() != hipSuccess) { fprintf(stderr, "kernel failed\n"); return 1; }
  hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost);
  hipMemcpy(st.data(), dst, st.size() * 4, hipMemcpyDeviceToHost);
  wr("ref_out.bin", out); wr("ref_state.bin", st);
  printf("ok\n");
}
