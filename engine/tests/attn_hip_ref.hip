// Reference harness: HIP's production prefill attention (PackAttentionHeads +
// WmmaCausalAttention<32, 16, true>, start_pos 0, key_begin 0, gate, no lse)
// on inputs from files; writes the output for a bit-exact comparison with Loom.
#include "src/models/qwen/hip/kernels/attention_wmma.hip"
#include <cstdio>
#include <vector>
template <class T> static std::vector<T> rd(const char* p, size_t n) {
  std::vector<T> v(n); FILE* f = fopen(p, "rb"); if (!f || fread(v.data(), sizeof(T), n, f) != n) { fprintf(stderr, "read %s\n", p); exit(1); } fclose(f); return v; }
template <class T> static T* up(const std::vector<T>& v) { T* d; (void)hipMalloc(&d, v.size() * sizeof(T)); (void)hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice); return d; }
int main(int argc, char** argv) {
  const unsigned B = atoi(argv[1]);
  auto q = rd<float>("q.bin", (size_t)B * 6144), g = rd<float>("gate.bin", (size_t)B * 6144);
  auto k = rd<__half>("k16.bin", (size_t)B * 1024), v = rd<__half>("v16.bin", (size_t)B * 1024);
  float *dq = up(q), *dg = up(g); __half *dk = up(k), *dv = up(v);
  const size_t padded = (B + 15) & ~15u;
  __half *pk, *pv; (void)hipMalloc(&pk, (size_t)4 * B * 256 * 2); (void)hipMalloc(&pv, (size_t)4 * padded * 256 * 2);
  std::vector<float> out((size_t)B * 6144, 0.f); float* dout = up(out);
  gufo::hip::PackAttentionHeads<<<dim3(4, (B + 15) / 16), 256>>>(dk, dv, pk, pv, B);
  gufo::hip::WmmaCausalAttention<32, 16, true><<<dim3((B + 31) / 32, 12), 256>>>(dq, dg, pk, pv, dout, 0u, (size_t)B, nullptr, 0u);
  if (hipDeviceSynchronize() != hipSuccess) { fprintf(stderr, "kernel failed\n"); return 1; }
  (void)hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost);
  FILE* f = fopen("ref_out.bin", "wb"); fwrite(out.data(), 4, out.size(), f); fclose(f);
  printf("ok\n");
}
