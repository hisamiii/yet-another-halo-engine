// loom_ssm_layer_probe: run one Gated DeltaNet (SSM) layer of the prefill graph
// entirely through HRX on the Loom kernels, no HIP, and dump the residual so it
// can be compared with the HIP engine's YAH_DUMP_DIR layer output.
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
constexpr std::uint32_t kB = 5;        // real prompt tokens
constexpr std::uint32_t kP = 64;       // padded token tile
constexpr std::uint32_t kHidden = 5120;
constexpr std::uint32_t kFfn = 17408;
constexpr std::uint32_t kQkv = 10240;  // 2*16*128 + 6144
constexpr std::uint32_t kInner = 6144;
constexpr std::uint32_t kTs = 48;
constexpr std::uint32_t kKh = 16;      // ssm group count
constexpr std::uint32_t kState = 128;

struct Imported {
  LoomBuffer buf;
  std::size_t offset;
  std::size_t bytes;
};

Imported ImportTensor(LoomDevice& gpu, const yah::core::Gguf& gguf,
                      const yah::core::TensorInfo& t) {
  const std::uint8_t* data = gguf.Data(t);
  const std::uintptr_t page = 4096;
  const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(data) & ~(page - 1);
  const std::size_t window =
      static_cast<std::size_t>(t.bytes) +
      (reinterpret_cast<std::uintptr_t>(data) - start);
  Imported out;
  out.buf = gpu.Import(reinterpret_cast<void*>(start), window);
  out.offset = reinterpret_cast<std::uintptr_t>(data) - start;
  out.bytes = static_cast<std::size_t>(t.bytes);
  return out;
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

void ReadFile(const char* path, void* dst, std::size_t bytes) {
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
  if (std::fread(dst, 1, bytes, f) != bytes) {
    std::fprintf(stderr, "short read %s\n", path); std::exit(1); }
  std::fclose(f);
}

void Dispatch(LoomDevice& gpu, const LoomExecutable& exe, const char* name,
              std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
              std::uint32_t sx, std::uint32_t sy, std::uint32_t sz,
              const hrx_buffer_ref_t* b, std::size_t n) {
  const uint32_t ordinal = exe.OrdinalOrZero(name);
  gpu.Dispatch(exe, ordinal, LoomDevice::Config(gx, gy, gz, sx, sy, sz), nullptr, 0,
               b, n);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: loom_ssm_layer_probe <model.gguf> <hidden_in> "
                 "<haldir> <out>\n");
    return 2;
  }
  const char* model = argv[1];
  const char* hidden_in = argv[2];
  const std::string dir = argv[3];
  const char* out_path = argv[4];
  const std::string pre = "blk.0.";
  try {
    auto gguf = yah::core::Gguf::Open(model);
    LoomDevice gpu;

    const yah::core::TensorInfo* t_qkv = gguf.Find(pre + "attn_qkv.weight");
    const yah::core::TensorInfo* t_gate = gguf.Find(pre + "attn_gate.weight");
    const yah::core::TensorInfo* t_alpha = gguf.Find(pre + "ssm_alpha.weight");
    const yah::core::TensorInfo* t_beta = gguf.Find(pre + "ssm_beta.weight");
    const yah::core::TensorInfo* t_out = gguf.Find(pre + "ssm_out.weight");
    const yah::core::TensorInfo* t_norm = gguf.Find(pre + "attn_norm.weight");
    const yah::core::TensorInfo* t_conv = gguf.Find(pre + "ssm_conv1d.weight");
    const yah::core::TensorInfo* t_a = gguf.Find(pre + "ssm_a");
    const yah::core::TensorInfo* t_dt = gguf.Find(pre + "ssm_dt.bias");
    const yah::core::TensorInfo* t_sn = gguf.Find(pre + "ssm_norm.weight");
    if (!t_qkv || !t_gate || !t_alpha || !t_beta || !t_out || !t_norm ||
        !t_conv || !t_a || !t_dt || !t_sn) {
      std::fprintf(stderr, "missing blk.0 tensor\n"); return 1;
    }
    const Imported w_qkv = ImportTensor(gpu, gguf, *t_qkv);
    const Imported w_gate = ImportTensor(gpu, gguf, *t_gate);
    const Imported w_alpha = ImportTensor(gpu, gguf, *t_alpha);
    const Imported w_beta = ImportTensor(gpu, gguf, *t_beta);
    const Imported w_out = ImportTensor(gpu, gguf, *t_out);

    // Activations, padded to a full token tile where a GEMM reads them.
    LoomBuffer hidden = gpu.Allocate(std::size_t{kP} * kHidden * 4);
    LoomBuffer reszero = gpu.Allocate(std::size_t{kB} * kHidden * 4);
    LoomBuffer sumout = gpu.Allocate(std::size_t{kB} * kHidden * 4);
    LoomBuffer scratch = gpu.Allocate(std::size_t{kP} * kFfn * 2);
    LoomBuffer ssm_qkv = gpu.Allocate(std::size_t{kP} * kQkv * 4);
    LoomBuffer ssm_gate = gpu.Allocate(std::size_t{kP} * kInner * 4);
    LoomBuffer alpha = gpu.Allocate(std::size_t{kP} * kTs * 4);
    LoomBuffer beta = gpu.Allocate(std::size_t{kP} * kTs * 4);
    LoomBuffer conv_out = gpu.Allocate(std::size_t{kB} * kQkv * 4);
    LoomBuffer kq = gpu.Allocate(std::size_t{kB} * kKh * 3 * 4);
    LoomBuffer ab = gpu.Allocate(std::size_t{kB} * kTs * 2 * 4);
    LoomBuffer raw = gpu.Allocate(std::size_t{kB} * kInner * 4);
    LoomBuffer conv_state = gpu.Allocate(std::size_t{kQkv} * 4 * 4);
    LoomBuffer state = gpu.Allocate(std::size_t{kTs} * kState * kState * 4);
    LoomBuffer wstage = gpu.Allocate(std::size_t{kQkv} * kHidden * 2);
    LoomBuffer ostage = gpu.Allocate(std::size_t{kQkv} * kP * 4);
    LoomBuffer wnorm = gpu.Allocate(std::size_t{kHidden} * 4);
    LoomBuffer wconv = gpu.Allocate(std::size_t{4} * kQkv * 4);
    LoomBuffer wa = gpu.Allocate(std::size_t{kTs} * 4);
    LoomBuffer wdt = gpu.Allocate(std::size_t{kTs} * 4);
    LoomBuffer wsn = gpu.Allocate(std::size_t{kState} * 4);

    std::vector<float> host_hidden(std::size_t{kB} * kHidden);
    ReadFile(hidden_in, host_hidden.data(), host_hidden.size() * 4);
    gpu.H2D(hidden, host_hidden.data(), host_hidden.size() * 4);
    Dump(gpu, hidden, host_hidden.size() * 4, "/tmp/loomdump/hidden_in.bin");
    std::vector<float> zeros(std::size_t{kB} * kHidden, 0.0f);
    gpu.H2D(reszero, zeros.data(), zeros.size() * 4);
    gpu.H2D(wnorm, gguf.Data(*t_norm), std::size_t{kHidden} * 4);
    gpu.H2D(wconv, gguf.Data(*t_conv), std::size_t{4} * kQkv * 4);
    gpu.H2D(wa, gguf.Data(*t_a), std::size_t{kTs} * 4);
    gpu.H2D(wdt, gguf.Data(*t_dt), std::size_t{kTs} * 4);
    gpu.H2D(wsn, gguf.Data(*t_sn), std::size_t{kState} * 4);
    std::vector<std::uint8_t> zconv(std::size_t{kQkv} * 4 * 4, 0);
    gpu.H2D(conv_state, zconv.data(), zconv.size());
    std::vector<std::uint8_t> zstate(std::size_t{kTs} * kState * kState * 4, 0);
    gpu.H2D(state, zstate.data(), zstate.size());

    LoomExecutable norm = gpu.Load(dir + "/norm/yah_half_norm_f16.hal");
    LoomExecutable gemm_qkv = gpu.Load(dir + "/qkv/yah_ffn_gemm_q4k_f32.hal");
    LoomExecutable gemm_gate = gpu.Load(dir + "/gate/yah_ffn_gemm_q4k_f32.hal");
    LoomExecutable gemm_alpha = gpu.Load(dir + "/alpha/yah_ffn_gemm_q4k_f32.hal");
    LoomExecutable gemm_beta = gpu.Load(dir + "/beta/yah_ffn_gemm_q4k_f32.hal");
    LoomExecutable gemm_res = gpu.Load(dir + "/res/yah_ffn_gemm_q4k_residual_f32.hal");
    LoomExecutable conv = gpu.Load(dir + "/conv/yah_ssm_conv_f32.hal");
    LoomExecutable prepkq = gpu.Load(dir + "/prepkq/yah_deltanet_prep_kq_f32.hal");
    LoomExecutable prepab = gpu.Load(dir + "/prepab/yah_deltanet_prep_ab_f32.hal");
    LoomExecutable rowsplit = gpu.Load(dir + "/rowsplit/yah_deltanet_rowsplit_f32.hal");
    LoomExecutable postnorm = gpu.Load(dir + "/postnorm/yah_ssm_postnorm_gate_f16.hal");

    Dump(gpu, hidden, std::size_t{kB} * kHidden * 4, "/tmp/loomdump/hidden_pre_norm.bin");
    // 1. attn_norm (f16 out).
    {
      const hrx_buffer_ref_t b[5] = {
          {hidden.handle, 0, std::size_t{kB} * kHidden * 4},
          {reszero.handle, 0, std::size_t{kB} * kHidden * 4},
          {wnorm.handle, 0, std::size_t{kHidden} * 4},
          {sumout.handle, 0, std::size_t{kB} * kHidden * 4},
          {scratch.handle, 0, std::size_t{kP} * kFfn * 2}};
      Dispatch(gpu, norm, "yah_half_norm", kB, 1, 1, 32, 1, 1, b, 5);
    }
    Dump(gpu, scratch, std::size_t{kB} * kHidden * 2, "/tmp/loomdump/norm_out.bin");
    Dump(gpu, sumout, std::size_t{kB} * kHidden * 4, "/tmp/loomdump/norm_sum.bin");
    // 2. Projections: attn_qkv, attn_gate, ssm_alpha, ssm_beta (Q4_K kStore).
    auto kstore = [&](const LoomExecutable& exe, const Imported& w, const LoomBuffer& out,
                      std::uint32_t m_tiles, std::uint32_t k_blocks,
                      std::uint32_t m_rows) {
      const hrx_buffer_ref_t b[5] = {
          {w.buf.handle, w.offset, w.bytes},
          {scratch.handle, 0, std::size_t{kP} * kFfn * 2},
          {wstage.handle, 0, std::size_t{kQkv} * kHidden * 2},
          {ostage.handle, 0, std::size_t{kQkv} * kP * 4},
          {out.handle, 0, std::size_t{m_rows} * kP * 4}};
      Dispatch(gpu, exe, "yah_ffn_gemm_q4k", m_tiles, 1, 1, 32, 1, 1, b, 5);
      (void)k_blocks;
    };
    kstore(gemm_qkv, w_qkv, ssm_qkv, 640, 20, kQkv);
    kstore(gemm_gate, w_gate, ssm_gate, 384, 20, kInner);
    kstore(gemm_alpha, w_alpha, alpha, 3, 20, kTs);
    kstore(gemm_beta, w_beta, beta, 3, 20, kTs);
    Dump(gpu, ssm_qkv, std::size_t{kB} * kQkv * 4, "/tmp/loomdump/ssm_qkv.bin");
    Dump(gpu, ssm_gate, std::size_t{kB} * kInner * 4, "/tmp/loomdump/ssm_gate.bin");
    Dump(gpu, alpha, std::size_t{kB} * kTs * 4, "/tmp/loomdump/ssm_alpha.bin");
    Dump(gpu, beta, std::size_t{kB} * kTs * 4, "/tmp/loomdump/ssm_beta.bin");
    // 3. Convolution over the qkv stream (reads the old history).
    {
      const hrx_buffer_ref_t b[4] = {
          {ssm_qkv.handle, 0, std::size_t{kB} * kQkv * 4},
          {wconv.handle, 0, std::size_t{4} * kQkv * 4},
          {conv_state.handle, 0, std::size_t{kQkv} * 4 * 4},
          {conv_out.handle, 0, std::size_t{kB} * kQkv * 4}};
      Dispatch(gpu, conv, "yah_ssm_conv", 40, kB, 1, 256, 1, 1, b, 4);
    }
    // 4. K/Q norm prologue.
    {
      const hrx_buffer_ref_t b[2] = {
          {conv_out.handle, 0, std::size_t{kB} * kQkv * 4},
          {kq.handle, 0, std::size_t{kB} * kKh * 3 * 4}};
      Dispatch(gpu, prepkq, "yah_deltanet_prep_kq", kKh, kB, 1, 32, 1, 1, b, 2);
    }
    // 5. alpha/beta prep, which also advances the conv history.
    {
      const hrx_buffer_ref_t b[7] = {
          {alpha.handle, 0, std::size_t{kP} * kTs * 4},
          {beta.handle, 0, std::size_t{kP} * kTs * 4},
          {wa.handle, 0, std::size_t{kTs} * 4},
          {wdt.handle, 0, std::size_t{kTs} * 4},
          {ssm_qkv.handle, 0, std::size_t{kB} * kQkv * 4},
          {conv_state.handle, 0, std::size_t{kQkv} * 4 * 4},
          {ab.handle, 0, std::size_t{kB} * kTs * 2 * 4}};
      Dispatch(gpu, prepab, "yah_deltanet_prep_ab", 41, 1, 1, 256, 1, 1, b, 7);
    }
    // 6. Row-split recurrence -> raw output.
    {
      const hrx_buffer_ref_t b[5] = {
          {conv_out.handle, 0, std::size_t{kB} * kQkv * 4},
          {kq.handle, 0, std::size_t{kB} * kKh * 3 * 4},
          {ab.handle, 0, std::size_t{kB} * kTs * 2 * 4},
          {state.handle, 0, std::size_t{kTs} * kState * kState * 4},
          {raw.handle, 0, std::size_t{kB} * kInner * 4}};
      Dispatch(gpu, rowsplit, "yah_deltanet", kTs, 1, 1, 128, 1, 1, b, 5);
    }
    Dump(gpu, raw, std::size_t{kB} * kInner * 4, "/tmp/loomdump/ssm_raw.bin");
    // 7. Post-norm + gate -> f16.
    {
      const hrx_buffer_ref_t b[4] = {
          {raw.handle, 0, std::size_t{kB} * kInner * 4},
          {wsn.handle, 0, std::size_t{kState} * 4},
          {ssm_gate.handle, 0, std::size_t{kB} * kInner * 4},
          {scratch.handle, 0, std::size_t{kP} * kFfn * 2}};
      Dispatch(gpu, postnorm, "yah_ssm_postnorm_fp16", 30, 1, 1, 256, 1, 1, b, 4);
    }
    Dump(gpu, scratch, std::size_t{kB} * kInner * 2, "/tmp/loomdump/ssm_postnorm.bin");
    // 8. Output projection + residual (Q4_K kResidual, K=inner).
    {
      const hrx_buffer_ref_t b[5] = {
          {w_out.buf.handle, w_out.offset, w_out.bytes},
          {scratch.handle, 0, std::size_t{kP} * kFfn * 2},
          {wstage.handle, 0, std::size_t{kQkv} * kHidden * 2},
          {ostage.handle, 0, std::size_t{kQkv} * kP * 4},
          {hidden.handle, 0, std::size_t{kP} * kHidden * 4}};
      Dispatch(gpu, gemm_res, "yah_ffn_gemm_q4k_residual", 320, 1, 1, 32, 1, 1, b, 5);
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
    std::fprintf(stderr, "loom_ssm_layer_probe: %s\n", error.what());
    return 1;
  }
}