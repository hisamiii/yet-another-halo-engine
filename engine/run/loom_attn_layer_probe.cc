// loom_attn_layer_probe: run one full-attention layer of the prefill graph
// through HRX on the Loom kernels, no HIP, and dump the post-mixer residual.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/gguf.hpp"
#include "model/loom_runtime.hpp"

using namespace yah::model;  // NOLINT(google-build-using-namespace)

namespace {
constexpr std::uint32_t kB = 5;
constexpr std::uint32_t kP = 64;
constexpr std::uint32_t kHidden = 5120;
constexpr std::uint32_t kFfn = 17408;
constexpr std::uint32_t kAttn = 6144;
constexpr std::uint32_t kQProj = 12288;
constexpr std::uint32_t kKv = 1024;
constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kCtx = 8;
constexpr std::uint32_t kCache = kCtx * kKvHeads * kHeadDim;

struct Imported { LoomBuffer buf; std::size_t offset; std::size_t bytes; };

Imported ImportTensor(LoomDevice& gpu, const yah::core::Gguf& gguf,
                      const yah::core::TensorInfo& t) {
  const std::uint8_t* data = gguf.Data(t);
  const std::uintptr_t page = 4096;
  const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(data) & ~(page - 1);
  const std::size_t window = static_cast<std::size_t>(t.bytes) +
                             (reinterpret_cast<std::uintptr_t>(data) - start);
  Imported out;
  out.buf = gpu.Import(reinterpret_cast<void*>(start), window);
  out.offset = reinterpret_cast<std::uintptr_t>(data) - start;
  out.bytes = static_cast<std::size_t>(t.bytes);
  return out;
}

void ReadFile(const char* path, void* dst, std::size_t bytes) {
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
  if (std::fread(dst, 1, bytes, f) != bytes) {
    std::fprintf(stderr, "short read %s\n", path); std::exit(1); }
  std::fclose(f);
}

void Dump(LoomDevice& gpu, const LoomBuffer& b, std::size_t bytes,
          const char* path) {
  gpu.Synchronize();
  std::vector<std::uint8_t> host(bytes);
  gpu.D2H(b, host.data(), bytes, 0);
  FILE* f = std::fopen(path, "wb");
  if (!f) { std::fprintf(stderr, "cannot write %s\n", path); std::exit(1); }
  std::fwrite(host.data(), 1, bytes, f);
  std::fclose(f);
}

void Dispatch(LoomDevice& gpu, const LoomExecutable& exe, const char* name,
              std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
              std::uint32_t sx, std::uint32_t sy, std::uint32_t sz,
              const hrx_buffer_ref_t* b, std::size_t n) {
  gpu.Dispatch(exe, exe.OrdinalOrZero(name),
               LoomDevice::Config(gx, gy, gz, sx, sy, sz), nullptr, 0, b, n);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: loom_attn_layer_probe <model.gguf> <hidden_in> <haldir> <out>\n");
    return 2;
  }
  const char* model = argv[1];
  const char* hidden_in = argv[2];
  const std::string dir = argv[3];
  const char* out_path = argv[4];
  const std::string pre = "blk.3.";
  try {
    auto gguf = yah::core::Gguf::Open(model);
    LoomDevice gpu;
    const auto* tq = gguf.Find(pre + "attn_q.weight");
    const auto* tk = gguf.Find(pre + "attn_k.weight");
    const auto* tv = gguf.Find(pre + "attn_v.weight");
    const auto* to = gguf.Find(pre + "attn_output.weight");
    const auto* tn = gguf.Find(pre + "attn_norm.weight");
    const auto* tqn = gguf.Find(pre + "attn_q_norm.weight");
    const auto* tkn = gguf.Find(pre + "attn_k_norm.weight");
    if (!tq || !tk || !tv || !to || !tn || !tqn || !tkn) {
      std::fprintf(stderr, "missing blk.3 tensor\n"); return 1; }
    const Imported wq = ImportTensor(gpu, gguf, *tq);
    const Imported wk = ImportTensor(gpu, gguf, *tk);
    const Imported wv = ImportTensor(gpu, gguf, *tv);
    const Imported wo = ImportTensor(gpu, gguf, *to);

    LoomBuffer hidden = gpu.Allocate(std::size_t{kP} * kHidden * 4);
    LoomBuffer reszero = gpu.Allocate(std::size_t{kB} * kHidden * 4);
    LoomBuffer sumout = gpu.Allocate(std::size_t{kB} * kHidden * 4);
    LoomBuffer scratch = gpu.Allocate(std::size_t{kP} * kFfn * 2);
    LoomBuffer qkv = gpu.Allocate(std::size_t{kP} * kQProj * 4);
    LoomBuffer q = gpu.Allocate(std::size_t{kB} * kAttn * 4);
    LoomBuffer gate = gpu.Allocate(std::size_t{kB} * kAttn * 4);
    LoomBuffer kb = gpu.Allocate(std::size_t{kP} * kKv * 4);
    LoomBuffer vb = gpu.Allocate(std::size_t{kP} * kKv * 4);
    LoomBuffer aout = gpu.Allocate(std::size_t{kB} * kAttn * 4);
    LoomBuffer kc16 = gpu.Allocate(std::size_t{kCache} * 2);
    LoomBuffer vc16 = gpu.Allocate(std::size_t{kCache} * 2);
    LoomBuffer kc32 = gpu.Allocate(std::size_t{kCache} * 4);
    LoomBuffer vc32 = gpu.Allocate(std::size_t{kCache} * 4);
    LoomBuffer lse = gpu.Allocate(std::size_t{kB} * kHeads * 4);
    LoomBuffer eps = gpu.Allocate(4);
    LoomBuffer wstage = gpu.Allocate(std::size_t{kQProj} * kHidden * 2);
    LoomBuffer ostage = gpu.Allocate(std::size_t{kQProj} * kP * 4);
    LoomBuffer wnorm = gpu.Allocate(std::size_t{kHidden} * 4);
    LoomBuffer wqn = gpu.Allocate(std::size_t{kHeadDim} * 4);
    LoomBuffer wkn = gpu.Allocate(std::size_t{kHeadDim} * 4);

    std::vector<float> host_hidden(std::size_t{kB} * kHidden);
    ReadFile(hidden_in, host_hidden.data(), host_hidden.size() * 4);
    gpu.H2D(hidden, host_hidden.data(), host_hidden.size() * 4);
    std::vector<float> zeros(std::size_t{kB} * kHidden, 0.0f);
    gpu.H2D(reszero, zeros.data(), zeros.size() * 4);
    gpu.H2D(wnorm, gguf.Data(*tn), std::size_t{kHidden} * 4);
    gpu.H2D(wqn, gguf.Data(*tqn), std::size_t{kHeadDim} * 4);
    gpu.H2D(wkn, gguf.Data(*tkn), std::size_t{kHeadDim} * 4);
    const float epsv = 1.0e-6f;
    gpu.H2D(eps, &epsv, 4);

    LoomExecutable norm = gpu.Load(dir + "/norm/yah_half_norm_f16.hal");
    LoomExecutable gemm_q = gpu.Load(dir + "/q/yah_ffn_gemm_q4k_f32.hal");
    LoomExecutable gemm_k = gpu.Load(dir + "/k/yah_ffn_gemm_q4k_f32.hal");
    LoomExecutable gemm_v = gpu.Load(dir + "/v/yah_ffn_gemm_q4k_f32.hal");
    LoomExecutable unpack = gpu.Load(dir + "/unpack/yah_unpack_qg_f32.hal");
    LoomExecutable rope = gpu.Load(dir + "/rope/yah_fused_qk_rope_batched_f32.hal");
    LoomExecutable wmma = gpu.Load(dir + "/wmma/yah_attn_wmma_f32.hal");
    LoomExecutable cast = gpu.Load(dir + "/cast/yah_half_cast.hal");
    LoomExecutable res = gpu.Load(dir + "/res/yah_ffn_gemm_q4k_residual_f32.hal");

    // 1. attn_norm.
    {
      const hrx_buffer_ref_t b[5] = {
          {hidden.handle, 0, std::size_t{kB} * kHidden * 4},
          {reszero.handle, 0, std::size_t{kB} * kHidden * 4},
          {wnorm.handle, 0, std::size_t{kHidden} * 4},
          {sumout.handle, 0, std::size_t{kB} * kHidden * 4},
          {scratch.handle, 0, std::size_t{kP} * kFfn * 2}};
      Dispatch(gpu, norm, "yah_half_norm", kB, 1, 1, 32, 1, 1, b, 5);
    }
    // 2. Q/K/V projections.
    auto kstore = [&](const LoomExecutable& exe, const Imported& w,
                      const LoomBuffer& out, std::uint32_t m_tiles,
                      std::uint32_t m_rows) {
      const hrx_buffer_ref_t b[5] = {
          {w.buf.handle, w.offset, w.bytes},
          {scratch.handle, 0, std::size_t{kP} * kFfn * 2},
          {wstage.handle, 0, std::size_t{kQProj} * kHidden * 2},
          {ostage.handle, 0, std::size_t{kQProj} * kP * 4},
          {out.handle, 0, std::size_t{m_rows} * kP * 4}};
      Dispatch(gpu, exe, "yah_ffn_gemm_q4k", m_tiles, 1, 1, 32, 1, 1, b, 5);
    };
    kstore(gemm_q, wq, qkv, 768, kQProj);
    kstore(gemm_k, wk, kb, 64, kKv);
    kstore(gemm_v, wv, vb, 64, kKv);
    // 3. Split q+gate.
    {
      const hrx_buffer_ref_t b[3] = {
          {qkv.handle, 0, std::size_t{kB} * kQProj * 4},
          {q.handle, 0, std::size_t{kB} * kAttn * 4},
          {gate.handle, 0, std::size_t{kB} * kAttn * 4}};
      Dispatch(gpu, unpack, "yah_unpack_qg", 24, kB, 1, 256, 1, 1, b, 3);
    }
    Dump(gpu, q, std::size_t{kB} * kAttn * 4, "/tmp/loomdump/attn_q.bin");
    Dump(gpu, kb, std::size_t{kB} * kKv * 4, "/tmp/loomdump/attn_k.bin");
    Dump(gpu, gate, std::size_t{kB} * kAttn * 4, "/tmp/loomdump/attn_gate.bin");
    // 4. QK norm + RoPE + KV cache write.
    {
      const hrx_buffer_ref_t b[12] = {
          {q.handle, 0, std::size_t{kB} * kAttn * 4},
          {kb.handle, 0, std::size_t{kB} * kKv * 4},
          {vb.handle, 0, std::size_t{kB} * kKv * 4},
          {wqn.handle, 0, std::size_t{kHeadDim} * 4},
          {wkn.handle, 0, std::size_t{kHeadDim} * 4},
          {q.handle, 0, std::size_t{kB} * kAttn * 4},
          {kb.handle, 0, std::size_t{kB} * kKv * 4},
          {kc32.handle, 0, std::size_t{kCache} * 4},
          {vc32.handle, 0, std::size_t{kCache} * 4},
          {kc16.handle, 0, std::size_t{kCache} * 2},
          {vc16.handle, 0, std::size_t{kCache} * 2},
          {eps.handle, 0, 4}};
      Dispatch(gpu, rope, "yah_fused_qk_rope_batched", 28, kB, 1, 256, 1, 1, b, 12);
    }
    // 5. Attention over the f16 cache.
    {
      const hrx_buffer_ref_t b[6] = {
          {q.handle, 0, std::size_t{kB} * kAttn * 4},
          {gate.handle, 0, std::size_t{kB} * kAttn * 4},
          {kc16.handle, 0, std::size_t{kCache} * 2},
          {vc16.handle, 0, std::size_t{kCache} * 2},
          {aout.handle, 0, std::size_t{kB} * kAttn * 4},
          {lse.handle, 0, std::size_t{kB} * kHeads * 4}};
      Dispatch(gpu, wmma, "yah_attn_wmma", kHeads, kB, 1, 32, 1, 1, b, 6);
    }
    Dump(gpu, aout, std::size_t{kB} * kAttn * 4, "/tmp/loomdump/attn_out.bin");
    // 6. Cast the attention output to f16.
    {
      const hrx_buffer_ref_t b[2] = {
          {aout.handle, 0, std::size_t{kB} * kAttn * 4},
          {scratch.handle, 0, std::size_t{kP} * kFfn * 2}};
      Dispatch(gpu, cast, "yah_half_cast", 120, 1, 1, 256, 1, 1, b, 2);
    }
    Dump(gpu, scratch, std::size_t{kB} * kAttn * 2, "/tmp/loomdump/attn_cast.bin");
    // 7. attn_output projection + residual.
    {
      const hrx_buffer_ref_t b[5] = {
          {wo.buf.handle, wo.offset, wo.bytes},
          {scratch.handle, 0, std::size_t{kP} * kFfn * 2},
          {wstage.handle, 0, std::size_t{kQProj} * kHidden * 2},
          {ostage.handle, 0, std::size_t{kQProj} * kP * 4},
          {hidden.handle, 0, std::size_t{kP} * kHidden * 4}};
      Dispatch(gpu, res, "yah_ffn_gemm_q4k_residual", 320, 1, 1, 32, 1, 1, b, 5);
    }
    gpu.Synchronize();
    std::vector<float> out(std::size_t{kB} * kHidden, 0.0f);
    gpu.D2H(hidden, out.data(), out.size() * 4);
    FILE* fo = std::fopen(out_path, "wb");
    std::fwrite(out.data(), 4, out.size(), fo);
    std::fclose(fo);
    std::fprintf(stderr, "wrote %zu f32 to %s\n", out.size(), out_path);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "loom_attn_layer_probe: %s\n", error.what());
    return 1;
  }
}