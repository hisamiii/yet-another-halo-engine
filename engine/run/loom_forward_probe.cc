// loom_forward_probe: the full 64-layer prefill stack on Loom through HRX, no
// HIP. Embedding is dequantized on the host (token_embd is Q4_K), each layer runs
// either the Gated DeltaNet mixer or the full-attention mixer plus the Q4_K FFN,
// and the head runs output RMSNorm + Q4_K GEMV + argmax.
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
constexpr std::uint32_t kInner = 6144;
constexpr std::uint32_t kQkv = 10240;
constexpr std::uint32_t kTs = 48;
constexpr std::uint32_t kKh = 16;
constexpr std::uint32_t kState = 128;
constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kCtx = 8;
constexpr std::uint32_t kCache = kCtx * kKvHeads * kHeadDim;
constexpr std::uint32_t kLayers = 64;
constexpr std::uint32_t kMainBlocks = 64;
constexpr std::uint32_t kRecurrent = 48;
constexpr std::uint32_t kAttnLayers = 16;
constexpr std::uint32_t kInterval = 4;
constexpr std::uint32_t kVocab = 248320;

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

void Dispatch(LoomDevice& gpu, const LoomExecutable& exe, const char* name,
              std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
              std::uint32_t sx, std::uint32_t sy, std::uint32_t sz,
              const hrx_buffer_ref_t* b, std::size_t n) {
  gpu.Dispatch(exe, exe.OrdinalOrZero(name),
               LoomDevice::Config(gx, gy, gz, sx, sy, sz), nullptr, 0, b, n);
}

float Half(const std::uint8_t* p) {
  _Float16 h;
  std::memcpy(&h, p, 2);
  return static_cast<float>(h);
}

// Host Q4_K decode of one token_embd row (block 144 bytes, 256 values).
void DequantQ4KRow(const std::uint8_t* base, std::uint64_t row, float* out) {
  const std::uint32_t nb = kHidden / 256;
  const std::uint8_t* p = base + row * static_cast<std::uint64_t>(nb) * 144;
  for (std::uint32_t b = 0; b < nb; ++b) {
    const std::uint8_t* blk = p + b * 144;
    const float d = Half(blk + 0);
    const float dmin = Half(blk + 2);
    const std::uint8_t* sc = blk + 4;
    const std::uint8_t* qs = blk + 16;
    for (std::uint32_t i = 0; i < 256; ++i) {
      const std::uint32_t gg = i / 64;
      const std::uint32_t wv = i % 64;
      const std::uint32_t lane = wv % 32;
      const bool low = wv < 32;
      const std::uint32_t qb = qs[gg * 32 + lane];
      const std::uint32_t quant = low ? (qb & 15) : (qb >> 4);
      const std::uint32_t j = 2 * gg + (low ? 0 : 1);
      std::uint32_t s, m;
      if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
      else { s = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4);
             m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
      out[b * 256 + i] = d * static_cast<float>(s) * static_cast<float>(quant) -
                         dmin * static_cast<float>(m);
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: loom_forward_probe <model.gguf> <out.bin> <dirs>\n");
    return 2;
  }
  const char* model = argv[1];
  const char* out_path = argv[2];
  const std::string ssm = argv[3];
  const std::string att = argv[4];
  const std::string fwd = argv[5];
  try {
    auto gguf = yah::core::Gguf::Open(model);
    LoomDevice gpu;
    auto find = [&](const std::string& name) {
      const auto* t = gguf.Find(name);
      if (!t) throw LoomError("tensor not found: " + name);
      return t;
    };

    // Activations (reused across layers).
    LoomBuffer hidden = gpu.Allocate(std::size_t{kP} * kHidden * 4);
    LoomBuffer reszero = gpu.Allocate(std::size_t{kB} * kHidden * 4);
    LoomBuffer sumout = gpu.Allocate(std::size_t{kB} * kHidden * 4);
    LoomBuffer scratch = gpu.Allocate(std::size_t{kP} * kFfn * 2);
    LoomBuffer qkv = gpu.Allocate(std::size_t{kP} * kQProj * 4);
    LoomBuffer gate = gpu.Allocate(std::size_t{kP} * kInner * 4);
    LoomBuffer alpha = gpu.Allocate(std::size_t{kP} * kTs * 4);
    LoomBuffer beta = gpu.Allocate(std::size_t{kP} * kTs * 4);
    LoomBuffer q = gpu.Allocate(std::size_t{kB} * kAttn * 4);
    LoomBuffer kbuf = gpu.Allocate(std::size_t{kP} * kKv * 4);
    LoomBuffer vbuf = gpu.Allocate(std::size_t{kP} * kKv * 4);
    LoomBuffer aout = gpu.Allocate(std::size_t{kB} * kAttn * 4);
    LoomBuffer raw = gpu.Allocate(std::size_t{kB} * kInner * 4);
    LoomBuffer conv_out = gpu.Allocate(std::size_t{kB} * kQkv * 4);
    LoomBuffer kqbuf = gpu.Allocate(std::size_t{kB} * kKh * 3 * 4);
    LoomBuffer ab = gpu.Allocate(std::size_t{kB} * kTs * 2 * 4);
    LoomBuffer conv_state = gpu.Allocate(std::size_t{kRecurrent} * kQkv * 4 * 4);
    LoomBuffer state = gpu.Allocate(std::size_t{kRecurrent} * kTs * kState * kState * 4);
    LoomBuffer kv16 = gpu.Allocate(std::size_t{kAttnLayers} * 2 * kCache * 2);
    LoomBuffer kc32 = gpu.Allocate(std::size_t{kCache} * 4);
    LoomBuffer vc32 = gpu.Allocate(std::size_t{kCache} * 4);
    LoomBuffer lse = gpu.Allocate(std::size_t{kB} * kHeads * 4);
    LoomBuffer eps = gpu.Allocate(4);
    LoomBuffer ffnup = gpu.Allocate(std::size_t{kP} * kFfn * 2);
    LoomBuffer gwstage = gpu.Allocate(std::size_t{kFfn} * kHidden * 2);
    LoomBuffer uwstage = gpu.Allocate(std::size_t{kFfn} * kHidden * 2);
    LoomBuffer ogate = gpu.Allocate(std::size_t{kFfn} * kP * 4);
    LoomBuffer oup = gpu.Allocate(std::size_t{kFfn} * kP * 4);
    LoomBuffer wstage = gpu.Allocate(std::size_t{kFfn} * kHidden * 2);
    LoomBuffer ostage = gpu.Allocate(std::size_t{kFfn} * kP * 4);
    LoomBuffer normed = gpu.Allocate(std::size_t{kHidden} * 4);
    LoomBuffer logits = gpu.Allocate(std::size_t{kVocab} * 4);
    LoomBuffer token = gpu.Allocate(4);
    LoomBuffer wsmall = gpu.Allocate(std::size_t{kInner} * 4);

    // Embedding on the host.
    const auto* emb = find("token_embd.weight");
    const std::uint8_t* emb_data = gguf.Data(*emb);
    std::vector<float> host_hidden(std::size_t{kB} * kHidden);
    const std::uint32_t ids[kB] = {760, 6511, 314, 9338, 369};
    for (std::uint32_t t = 0; t < kB; ++t) {
      DequantQ4KRow(emb_data, ids[t], host_hidden.data() + std::size_t{t} * kHidden);
    }
    gpu.H2D(hidden, host_hidden.data(), host_hidden.size() * 4);
    std::vector<float> zeros(std::size_t{kB} * kHidden, 0.0f);
    gpu.H2D(reszero, zeros.data(), zeros.size() * 4);
    const float epsv = 1.0e-6f;
    gpu.H2D(eps, &epsv, 4);
    std::vector<std::uint8_t> zconv(std::size_t{kRecurrent} * kQkv * 4 * 4, 0);
    gpu.H2D(conv_state, zconv.data(), zconv.size());
    std::vector<std::uint8_t> zstate(std::size_t{kRecurrent} * kTs * kState * kState * 4, 0);
    gpu.H2D(state, zstate.data(), zstate.size());
    std::vector<std::uint8_t> zkv(std::size_t{kAttnLayers} * 2 * kCache * 2, 0);
    gpu.H2D(kv16, zkv.data(), zkv.size());

    LoomExecutable e_norm = gpu.Load(ssm + "/norm/yah_half_norm_f16.hal");
    LoomExecutable e_qkv = gpu.Load(ssm + "/qkv/yah_ffn_gemm_q4k_f32.hal");
    LoomExecutable e_gate = gpu.Load(ssm + "/gate/yah_ffn_gemm_q4k_f32.hal");
    LoomExecutable e_alpha = gpu.Load(ssm + "/alpha/yah_ffn_gemm_q4k_f32.hal");
    LoomExecutable e_beta = gpu.Load(ssm + "/beta/yah_ffn_gemm_q4k_f32.hal");
    LoomExecutable e_res = gpu.Load(ssm + "/res/yah_ffn_gemm_q4k_residual_f32.hal");
    LoomExecutable e_conv = gpu.Load(ssm + "/conv/yah_ssm_conv_f32.hal");
    LoomExecutable e_prepkq = gpu.Load(ssm + "/prepkq/yah_deltanet_prep_kq_f32.hal");
    LoomExecutable e_prepab = gpu.Load(ssm + "/prepab/yah_deltanet_prep_ab_f32.hal");
    LoomExecutable e_rowsplit = gpu.Load(ssm + "/rowsplit/yah_deltanet_rowsplit_f32.hal");
    LoomExecutable e_postnorm = gpu.Load(ssm + "/postnorm/yah_ssm_postnorm_gate_f16.hal");
    LoomExecutable e_q = gpu.Load(att + "/q/yah_ffn_gemm_q4k_f32.hal");
    LoomExecutable e_k = gpu.Load(att + "/k/yah_ffn_gemm_q4k_f32.hal");
    LoomExecutable e_v = gpu.Load(att + "/v/yah_ffn_gemm_q4k_f32.hal");
    LoomExecutable e_unpack = gpu.Load(att + "/unpack/yah_unpack_qg_f32.hal");
    LoomExecutable e_rope = gpu.Load(att + "/rope/yah_fused_qk_rope_batched_f32.hal");
    LoomExecutable e_wmma = gpu.Load(att + "/wmma/yah_attn_wmma_f32.hal");
    LoomExecutable e_cast = gpu.Load(att + "/cast/yah_half_cast.hal");
    LoomExecutable e_res_attn = gpu.Load(att + "/res/yah_ffn_gemm_q4k_residual_f32.hal");
    LoomExecutable e_gateup = gpu.Load(fwd + "/gateup/yah_ffn_gemm_q4k_gateup_f16.hal");
    LoomExecutable e_res_down = gpu.Load(fwd + "/res_down/yah_ffn_gemm_q4k_residual_f32.hal");
    LoomExecutable e_gemv = gpu.Load(fwd + "/gemv/yah_gemv_q4k_f32.hal");
    LoomExecutable e_rms = gpu.Load(fwd + "/rmsnorm/yah_rmsnorm_f32.hal");
    LoomExecutable e_argmax = gpu.Load(fwd + "/argmax/yah_argmax_f32.hal");

    auto run_norm = [&](const std::string& wname, const LoomBuffer& x) {
      const auto* tw = find(wname);
      LoomBuffer w = gpu.Allocate(std::size_t{kHidden} * 4);
      gpu.H2D(w, gguf.Data(*tw), std::size_t{kHidden} * 4);
      const hrx_buffer_ref_t b[5] = {
          {x.handle, 0, std::size_t{kB} * kHidden * 4},
          {reszero.handle, 0, std::size_t{kB} * kHidden * 4},
          {w.handle, 0, std::size_t{kHidden} * 4},
          {sumout.handle, 0, std::size_t{kB} * kHidden * 4},
          {scratch.handle, 0, std::size_t{kP} * kFfn * 2}};
      Dispatch(gpu, e_norm, "yah_half_norm", kB, 1, 1, 32, 1, 1, b, 5);
    };
    auto run_kstore = [&](const LoomExecutable& exe, const std::string& wname,
                          const LoomBuffer& out, std::uint32_t m_tiles) {
      const auto* tw = find(wname);
      const Imported w = ImportTensor(gpu, gguf, *tw);
      const hrx_buffer_ref_t b[5] = {
          {w.buf.handle, w.offset, w.bytes},
          {scratch.handle, 0, std::size_t{kP} * kFfn * 2},
          {wstage.handle, 0, std::size_t{kFfn} * kHidden * 2},
          {ostage.handle, 0, std::size_t{kFfn} * kP * 4},
          {out.handle, 0, out.size}};
      Dispatch(gpu, exe, "yah_ffn_gemm_q4k", m_tiles, 1, 1, 32, 1, 1, b, 5);
    };
    auto run_residual = [&](const LoomExecutable& exe, const std::string& wname,
                            const LoomBuffer& input) {
      const auto* tw = find(wname);
      const Imported w = ImportTensor(gpu, gguf, *tw);
      const hrx_buffer_ref_t b[5] = {
          {w.buf.handle, w.offset, w.bytes},
          {input.handle, 0, input.size},
          {wstage.handle, 0, std::size_t{kFfn} * kHidden * 2},
          {ostage.handle, 0, std::size_t{kFfn} * kP * 4},
          {hidden.handle, 0, hidden.size}};
      Dispatch(gpu, exe, "yah_ffn_gemm_q4k_residual", 320, 1, 1, 32, 1, 1, b, 5);
    };

    for (std::uint32_t l = 0; l < kMainBlocks; ++l) {
      const std::string pre = "blk." + std::to_string(l) + ".";
      const bool full = ((l + 1) % kInterval) == 0;
      run_norm(pre + "attn_norm.weight", hidden);
      if (full) {
        const std::uint32_t ai = l / kInterval;
        run_kstore(e_q, pre + "attn_q.weight", qkv, 768);
        run_kstore(e_k, pre + "attn_k.weight", kbuf, 64);
        run_kstore(e_v, pre + "attn_v.weight", vbuf, 64);
        {
          const hrx_buffer_ref_t b[3] = {
              {qkv.handle, 0, std::size_t{kB} * kQProj * 4},
              {q.handle, 0, std::size_t{kB} * kAttn * 4},
              {gate.handle, 0, std::size_t{kB} * kAttn * 4}};
          Dispatch(gpu, e_unpack, "yah_unpack_qg", 24, kB, 1, 256, 1, 1, b, 3);
        }
        const auto* qn = find(pre + "attn_q_norm.weight");
        const auto* kn = find(pre + "attn_k_norm.weight");
        LoomBuffer wqn = gpu.Allocate(std::size_t{kHeadDim} * 4);
        LoomBuffer wkn = gpu.Allocate(std::size_t{kHeadDim} * 4);
        gpu.H2D(wqn, gguf.Data(*qn), std::size_t{kHeadDim} * 4);
        gpu.H2D(wkn, gguf.Data(*kn), std::size_t{kHeadDim} * 4);
        const std::size_t koff = std::size_t{ai} * kCache;
        {
          const hrx_buffer_ref_t b[12] = {
              {q.handle, 0, std::size_t{kB} * kAttn * 4},
              {kbuf.handle, 0, std::size_t{kB} * kKv * 4},
              {vbuf.handle, 0, std::size_t{kB} * kKv * 4},
              {wqn.handle, 0, std::size_t{kHeadDim} * 4},
              {wkn.handle, 0, std::size_t{kHeadDim} * 4},
              {q.handle, 0, std::size_t{kB} * kAttn * 4},
              {kbuf.handle, 0, std::size_t{kB} * kKv * 4},
              {kc32.handle, 0, std::size_t{kCache} * 4},
              {vc32.handle, 0, std::size_t{kCache} * 4},
              {kv16.handle, 0, std::size_t{kCache} * 2},
              {kv16.handle, koff * 2 + std::size_t{kCache} * 2, std::size_t{kCache} * 2},
              {eps.handle, 0, 4}};
          Dispatch(gpu, e_rope, "yah_fused_qk_rope_batched", 28, kB, 1, 256, 1, 1, b, 12);
        }
        {
          const hrx_buffer_ref_t b[6] = {
              {q.handle, 0, std::size_t{kB} * kAttn * 4},
              {gate.handle, 0, std::size_t{kB} * kAttn * 4},
              {kv16.handle, 0, std::size_t{kCache} * 2},
              {kv16.handle, koff * 2 + std::size_t{kCache} * 2, std::size_t{kCache} * 2},
              {aout.handle, 0, std::size_t{kB} * kAttn * 4},
              {lse.handle, 0, std::size_t{kB} * kHeads * 4}};
          Dispatch(gpu, e_wmma, "yah_attn_wmma", kHeads, kB, 1, 32, 1, 1, b, 6);
        }
        {
          const hrx_buffer_ref_t b[2] = {
              {aout.handle, 0, std::size_t{kB} * kAttn * 4},
              {scratch.handle, 0, std::size_t{kP} * kFfn * 2}};
          Dispatch(gpu, e_cast, "yah_half_cast", 120, 1, 1, 256, 1, 1, b, 2);
        }
        run_residual(e_res_attn, pre + "attn_output.weight", scratch);
      } else {
        const std::uint32_t si = l - l / kInterval;
        run_kstore(e_qkv, pre + "attn_qkv.weight", qkv, 640);
        run_kstore(e_gate, pre + "attn_gate.weight", gate, 384);
        run_kstore(e_alpha, pre + "ssm_alpha.weight", alpha, 3);
        run_kstore(e_beta, pre + "ssm_beta.weight", beta, 3);
        const auto* convw = find(pre + "ssm_conv1d.weight");
        LoomBuffer wconv = gpu.Allocate(std::size_t{4} * kQkv * 4);
        gpu.H2D(wconv, gguf.Data(*convw), std::size_t{4} * kQkv * 4);
        const auto* ta = find(pre + "ssm_a");
        const auto* tdt = find(pre + "ssm_dt.bias");
        const auto* tsn = find(pre + "ssm_norm.weight");
        LoomBuffer wa = gpu.Allocate(std::size_t{kTs} * 4);
        LoomBuffer wdt = gpu.Allocate(std::size_t{kTs} * 4);
        LoomBuffer wsn = gpu.Allocate(std::size_t{kState} * 4);
        gpu.H2D(wa, gguf.Data(*ta), std::size_t{kTs} * 4);
        gpu.H2D(wdt, gguf.Data(*tdt), std::size_t{kTs} * 4);
        gpu.H2D(wsn, gguf.Data(*tsn), std::size_t{kState} * 4);
        const std::size_t cs_off = std::size_t{si} * kQkv * 4 * 4;
        const std::size_t st_off = std::size_t{si} * kTs * kState * kState * 4;
        {
          const hrx_buffer_ref_t b[4] = {
              {qkv.handle, 0, std::size_t{kB} * kQkv * 4},
              {wconv.handle, 0, std::size_t{4} * kQkv * 4},
              {conv_state.handle, cs_off, std::size_t{kQkv} * 4 * 4},
              {conv_out.handle, 0, std::size_t{kB} * kQkv * 4}};
          Dispatch(gpu, e_conv, "yah_ssm_conv", 40, kB, 1, 256, 1, 1, b, 4);
        }
        {
          const hrx_buffer_ref_t b[2] = {
              {conv_out.handle, 0, std::size_t{kB} * kQkv * 4},
              {kqbuf.handle, 0, std::size_t{kB} * kKh * 3 * 4}};
          Dispatch(gpu, e_prepkq, "yah_deltanet_prep_kq", kKh, kB, 1, 32, 1, 1, b, 2);
        }
        {
          const hrx_buffer_ref_t b[7] = {
              {alpha.handle, 0, alpha.size},
              {beta.handle, 0, beta.size},
              {wa.handle, 0, std::size_t{kTs} * 4},
              {wdt.handle, 0, std::size_t{kTs} * 4},
              {qkv.handle, 0, std::size_t{kB} * kQkv * 4},
              {conv_state.handle, cs_off, std::size_t{kQkv} * 4 * 4},
              {ab.handle, 0, std::size_t{kB} * kTs * 2 * 4}};
          Dispatch(gpu, e_prepab, "yah_deltanet_prep_ab", 41, 1, 1, 256, 1, 1, b, 7);
        }
        {
          const hrx_buffer_ref_t b[5] = {
              {conv_out.handle, 0, std::size_t{kB} * kQkv * 4},
              {kqbuf.handle, 0, std::size_t{kB} * kKh * 3 * 4},
              {ab.handle, 0, std::size_t{kB} * kTs * 2 * 4},
              {state.handle, st_off, std::size_t{kTs} * kState * kState * 4},
              {raw.handle, 0, std::size_t{kB} * kInner * 4}};
          Dispatch(gpu, e_rowsplit, "yah_deltanet", kTs, 1, 1, 128, 1, 1, b, 5);
        }
        {
          const hrx_buffer_ref_t b[4] = {
              {raw.handle, 0, std::size_t{kB} * kInner * 4},
              {wsn.handle, 0, std::size_t{kState} * 4},
              {gate.handle, 0, std::size_t{kB} * kInner * 4},
              {scratch.handle, 0, std::size_t{kP} * kFfn * 2}};
          Dispatch(gpu, e_postnorm, "yah_ssm_postnorm_fp16", 30, 1, 1, 256, 1, 1, b, 4);
        }
        run_residual(e_res, pre + "ssm_out.weight", scratch);
      }
      // FFN: post_attention_norm, paired Q4_K gate/up, Q4_K down + residual.
      run_norm(pre + "post_attention_norm.weight", hidden);
      {
        const auto* tg = find(pre + "ffn_gate.weight");
        const auto* tu = find(pre + "ffn_up.weight");
        const Imported wg = ImportTensor(gpu, gguf, *tg);
        const Imported wu = ImportTensor(gpu, gguf, *tu);
        const hrx_buffer_ref_t b[8] = {
            {wg.buf.handle, wg.offset, wg.bytes},
            {wu.buf.handle, wu.offset, wu.bytes},
            {scratch.handle, 0, std::size_t{kP} * kFfn * 2},
            {gwstage.handle, 0, std::size_t{kFfn} * kHidden * 2},
            {uwstage.handle, 0, std::size_t{kFfn} * kHidden * 2},
            {ogate.handle, 0, std::size_t{kFfn} * kP * 4},
            {oup.handle, 0, std::size_t{kFfn} * kP * 4},
            {ffnup.handle, 0, std::size_t{kP} * kFfn * 2}};
        Dispatch(gpu, e_gateup, "yah_ffn_gemm_q4k_gateup", 1088, 1, 1, 32, 1, 1, b, 8);
      }
      run_residual(e_res_down, pre + "ffn_down.weight", ffnup);
    }

    // Head: output RMSNorm on the last position, Q4_K GEMV, argmax.
    const auto* onw = find("output_norm.weight");
    const auto* ow = find("output.weight");
    gpu.H2D(wsmall, gguf.Data(*onw), std::size_t{kHidden} * 4);
    {
      const hrx_buffer_ref_t b[3] = {
          {hidden.handle, std::size_t{kB - 1} * kHidden * 4, std::size_t{kHidden} * 4},
          {wsmall.handle, 0, std::size_t{kHidden} * 4},
          {normed.handle, 0, std::size_t{kHidden} * 4}};
      Dispatch(gpu, e_rms, "yah_rmsnorm", 1, 1, 1, 32, 1, 1, b, 3);
    }
    {
      const Imported w = ImportTensor(gpu, gguf, *ow);
      const hrx_buffer_ref_t b[3] = {
          {w.buf.handle, w.offset, w.bytes},
          {normed.handle, 0, std::size_t{kHidden} * 4},
          {logits.handle, 0, std::size_t{kVocab} * 4}};
      Dispatch(gpu, e_gemv, "yah_gemv_q4k", kVocab, 1, 1, 32, 1, 1, b, 3);
    }
    {
      const hrx_buffer_ref_t b[2] = {
          {logits.handle, 0, std::size_t{kVocab} * 4},
          {token.handle, 0, 4}};
      Dispatch(gpu, e_argmax, "yah_argmax", 1, 1, 1, 32, 1, 1, b, 2);
    }
    gpu.Synchronize();
    std::uint32_t tok = 0;
    gpu.D2H(token, &tok, 4, 0);
    std::vector<float> out(std::size_t{kB} * kHidden, 0.0f);
    gpu.D2H(hidden, out.data(), out.size() * 4, 0);
    FILE* fo = std::fopen(out_path, "wb");
    std::fwrite(out.data(), 4, out.size(), fo);
    std::fclose(fo);
    std::printf("argmax=%u\n", tok);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "loom_forward_probe: %s\n", error.what());
    return 1;
  }
}