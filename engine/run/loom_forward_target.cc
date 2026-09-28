// loom_forward_target: the full prefill stack on Loom through HRX for a shard
// with mixed quantized formats. The GEMM HAL is selected per tensor from its
// ggml type and shape, using the names emit_prefill.py writes.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "core/config.hpp"
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
constexpr std::uint32_t kVocab = 248320;

const float kKvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                            1, 13, 25, 38, 53, 69, 89, 113};

struct Fmt { const char* name; std::uint32_t qk; };
bool FmtOf(std::uint32_t type, Fmt* out) {
  switch (type) {
    case 12: *out = {"q4k", 256}; return true;
    case 13: *out = {"q5k", 256}; return true;
    case 14: *out = {"q6k", 256}; return true;
    case 11: *out = {"q3k", 256}; return true;
    case 23: *out = {"iq4xs", 256}; return true;
    case 21: *out = {"iq3s", 256}; return true;
    case 18: *out = {"iq3xxs", 256}; return true;
    case 20: *out = {"iq4nl", 32}; return true;
    case 17: *out = {"iq2xs", 256}; return true;
    case 8: *out = {"q8_0", 32}; return true;
    case 16: *out = {"iq2xxs", 256}; return true;
    case 10: *out = {"q2k", 256}; return true;
    default: return false;
  }
}
bool HasGrid(const std::string& f) { return f == "iq3s" || f == "iq3xxs"; }
bool HasKsigns(const std::string& f) { return f == "iq3xxs"; }

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
              const std::vector<hrx_buffer_ref_t>& b) {
  gpu.Dispatch(exe, exe.OrdinalOrZero(name),
               LoomDevice::Config(gx, gy, gz, sx, sy, sz), nullptr, 0, b.data(),
               b.size());
}

void ReadFile(const char* path, void* dst, std::size_t bytes) {
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
  if (std::fread(dst, 1, bytes, f) != bytes) { std::exit(1); }
  std::fclose(f);
}
float Half(const std::uint8_t* p) { _Float16 h; std::memcpy(&h, p, 2); return (float)h; }

void DequantQ4KRow(const std::uint8_t* base, std::uint64_t row, float* out) {
  const std::uint32_t nb = kHidden / 256;
  const std::uint8_t* p = base + row * static_cast<std::uint64_t>(nb) * 144;
  for (std::uint32_t b = 0; b < nb; ++b) {
    const std::uint8_t* blk = p + b * 144;
    const float d = Half(blk), dmin = Half(blk + 2);
    const std::uint8_t* sc = blk + 4; const std::uint8_t* qs = blk + 16;
    for (std::uint32_t i = 0; i < 256; ++i) {
      const std::uint32_t gg = i / 64, wv = i % 64, lane = wv % 32;
      const bool low = wv < 32;
      const std::uint32_t qb = qs[gg * 32 + lane];
      const std::uint32_t quant = low ? (qb & 15) : (qb >> 4);
      const std::uint32_t j = 2 * gg + (low ? 0 : 1);
      std::uint32_t s, m;
      if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
      else { s = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4);
             m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
      out[b * 256 + i] = d * (float)s * (float)quant - dmin * (float)m;
    }
  }
}

void DequantIq4XsRow(const std::uint8_t* base, std::uint64_t row, float* out) {
  const std::uint32_t nb = kHidden / 256;
  const std::uint8_t* p = base + row * static_cast<std::uint64_t>(nb) * 136;
  for (std::uint32_t b = 0; b < nb; ++b) {
    const std::uint8_t* blk = p + b * 136;
    const float d = Half(blk);
    const std::uint32_t sh = blk[2] | (blk[3] << 8);
    const std::uint8_t* sl = blk + 4; const std::uint8_t* qs = blk + 8;
    for (std::uint32_t g = 0; g < 8; ++g) {
      const std::uint32_t sc = ((sl[g / 2] >> (4 * (g % 2))) & 15) | (((sh >> (2 * g)) & 3) << 4);
      const float dl = d * (float)((int)sc - 32);
      for (std::uint32_t w = 0; w < 32; ++w) {
        const std::uint32_t q = qs[g * 16 + (w & 15)];
        const std::uint32_t nib = (w >= 16) ? (q >> 4) : (q & 15);
        out[b * 256 + g * 32 + w] = dl * kKvalues[nib];
      }
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: loom_forward_target <model.gguf> <haldir> <out.bin>\n");
    return 2;
  }
  const char* model = argv[1];
  const std::string dir = argv[2];
  const char* out_path = argv[3];
  try {
    auto gguf = yah::core::Gguf::Open(model);
    const auto cfg = yah::core::Qwen35Config::FromGguf(gguf);
    LoomDevice gpu;
    std::map<std::string, LoomExecutable> exes;
    auto load = [&](const std::string& path) -> LoomExecutable& {
      auto it = exes.find(path);
      if (it == exes.end()) it = exes.emplace(path, gpu.Load(path)).first;
      return it->second;
    };
    auto find = [&](const std::string& name) {
      const auto* t = gguf.Find(name);
      if (!t) throw LoomError("tensor not found: " + name);
      return t;
    };
    LoomBuffer grid_iq3s = gpu.Allocate(std::size_t{512} * 4);
    LoomBuffer grid_iq3xxs = gpu.Allocate(std::size_t{256} * 4);
    LoomBuffer ksigns_iq3xxs = gpu.Allocate(std::size_t{128});
    LoomBuffer grid_iq2xxs = gpu.Allocate(std::size_t{512} * 4);
    LoomBuffer grid_iq2xs = gpu.Allocate(std::size_t{1024} * 4);
    LoomBuffer ksigns_iq2xxs = gpu.Allocate(std::size_t{128});
    { std::vector<std::uint8_t> v(512 * 4); ReadFile((dir + "/grid_iq3s.bin").c_str(), v.data(), v.size()); gpu.H2D(grid_iq3s, v.data(), v.size()); }
    { std::vector<std::uint8_t> v(256 * 4); ReadFile((dir + "/grid_iq3xxs.bin").c_str(), v.data(), v.size()); gpu.H2D(grid_iq3xxs, v.data(), v.size()); }
    { std::vector<std::uint8_t> v(128); ReadFile((dir + "/ksigns_iq3xxs.bin").c_str(), v.data(), v.size()); gpu.H2D(ksigns_iq3xxs, v.data(), v.size()); }
    { std::vector<std::uint8_t> v(512 * 4); ReadFile((dir + "/grid_iq2xxs.bin").c_str(), v.data(), v.size()); gpu.H2D(grid_iq2xxs, v.data(), v.size()); }
    { std::vector<std::uint8_t> v(1024 * 4); ReadFile((dir + "/grid_iq2xs.bin").c_str(), v.data(), v.size()); gpu.H2D(grid_iq2xs, v.data(), v.size()); }
    { std::vector<std::uint8_t> v(128); ReadFile((dir + "/ksigns_iq2xxs.bin").c_str(), v.data(), v.size()); gpu.H2D(ksigns_iq2xxs, v.data(), v.size()); }

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
    LoomBuffer conv_state = gpu.Allocate(std::size_t{48} * kQkv * 4 * 4);
    LoomBuffer state = gpu.Allocate(std::size_t{48} * kTs * kState * kState * 4);
    LoomBuffer kv16 = gpu.Allocate(std::size_t{16} * 2 * kCache * 2);
    LoomBuffer kc32 = gpu.Allocate(std::size_t{kCache} * 4);
    LoomBuffer vc32 = gpu.Allocate(std::size_t{kCache} * 4);
    LoomBuffer lse = gpu.Allocate(std::size_t{kB} * kHeads * 4);
    LoomBuffer eps = gpu.Allocate(4);
    LoomBuffer ffnup = gpu.Allocate(std::size_t{kP} * kFfn * 2);
    LoomBuffer gateffn = gpu.Allocate(std::size_t{kP} * kFfn * 4);
    LoomBuffer gwstage = gpu.Allocate(std::size_t{kFfn} * kHidden * 2);
    LoomBuffer uwstage = gpu.Allocate(std::size_t{kFfn} * kHidden * 2);
    LoomBuffer ogate = gpu.Allocate(std::size_t{kFfn} * kP * 4);
    LoomBuffer oup = gpu.Allocate(std::size_t{kFfn} * kP * 4);
    LoomBuffer wstage = gpu.Allocate(std::size_t{kFfn} * kHidden * 2);
    LoomBuffer ostage = gpu.Allocate(std::size_t{kFfn} * kP * 4);
    LoomBuffer normed = gpu.Allocate(std::size_t{kHidden} * 4);
    LoomBuffer logits = gpu.Allocate(std::size_t{kVocab} * 4);
    LoomBuffer token = gpu.Allocate(4);

    const auto hb = [](const LoomBuffer& b) { return b.size; };
    std::vector<float> zeros(std::size_t{kB} * kHidden, 0.0f);
    gpu.H2D(reszero, zeros.data(), zeros.size() * 4);
    const float epsv = 1.0e-6f; gpu.H2D(eps, &epsv, 4);
    { std::vector<std::uint8_t> z(std::size_t{48} * kQkv * 4 * 4, 0); gpu.H2D(conv_state, z.data(), z.size()); }
    { std::vector<std::uint8_t> z(std::size_t{48} * kTs * kState * kState * 4, 0); gpu.H2D(state, z.data(), z.size()); }
    { std::vector<std::uint8_t> z(std::size_t{16} * 2 * kCache * 2, 0); gpu.H2D(kv16, z.data(), z.size()); }

    const auto* emb = find("token_embd.weight");
    std::vector<float> host_hidden(std::size_t{kB} * kHidden);
    const std::uint32_t ids[kB] = {760, 6511, 314, 9338, 369};
    const std::uint8_t* emb_data = gguf.Data(*emb);
    for (std::uint32_t t = 0; t < kB; ++t) {
      if (static_cast<std::uint32_t>(emb->type) == 23) DequantIq4XsRow(emb_data, ids[t], host_hidden.data() + std::size_t{t} * kHidden);
      else DequantQ4KRow(emb_data, ids[t], host_hidden.data() + std::size_t{t} * kHidden);
    }
    gpu.H2D(hidden, host_hidden.data(), host_hidden.size() * 4);

    LoomExecutable& e_norm = load(dir + "/norm.hal");
    LoomExecutable& e_conv = load(dir + "/conv.hal");
    LoomExecutable& e_prepkq = load(dir + "/prepkq.hal");
    LoomExecutable& e_prepab = load(dir + "/prepab.hal");
    LoomExecutable& e_rowsplit = load(dir + "/rowsplit.hal");
    LoomExecutable& e_postnorm = load(dir + "/postnorm.hal");
    LoomExecutable& e_unpack = load(dir + "/unpack.hal");
    LoomExecutable& e_rope = load(dir + "/rope.hal");
    LoomExecutable& e_wmma = load(dir + "/wmma.hal");
    LoomExecutable& e_cast = load(dir + "/cast.hal");
    LoomExecutable& e_gemv = load(dir + "/gemv.hal");
    LoomExecutable& e_rms = load(dir + "/rmsnorm.hal");
    LoomExecutable& e_argmax = load(dir + "/argmax.hal");

    auto run_norm = [&](const std::string& wname) {
      const auto* tw = find(wname);
      LoomBuffer w = gpu.Allocate(std::size_t{kHidden} * 4);
      gpu.H2D(w, gguf.Data(*tw), std::size_t{kHidden} * 4);
      std::vector<hrx_buffer_ref_t> b = {
          {hidden.handle, 0, hb(hidden)}, {reszero.handle, 0, hb(reszero)},
          {w.handle, 0, hb(w)}, {sumout.handle, 0, hb(sumout)},
          {scratch.handle, 0, hb(scratch)}};
      Dispatch(gpu, e_norm, "yah_half_norm", kB, 1, 1, 32, 1, 1, b);
    };
    auto run_kstore = [&](const std::string& wname, const LoomBuffer& out) {
      const auto* tw = find(wname);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(tw->type), &f)) throw LoomError("no kStore port for type on " + wname);
      const std::uint32_t mt = static_cast<std::uint32_t>(tw->dims[1] / 16);
      const std::uint32_t kb = static_cast<std::uint32_t>(tw->dims[0] / f.qk);
      const Imported w = ImportTensor(gpu, gguf, *tw);
      LoomExecutable& exe = load(dir + "/gemm_kstore_" + f.name + "_" +
                                 std::to_string(mt) + "_" + std::to_string(kb) + ".hal");
      std::vector<hrx_buffer_ref_t> b = {{w.buf.handle, w.offset, w.bytes}};
      if (f.name == std::string("iq3s")) b.push_back({grid_iq3s.handle, 0, hb(grid_iq3s)});
      if (f.name == std::string("iq3xxs")) b.push_back({grid_iq3xxs.handle, 0, hb(grid_iq3xxs)});
      if (f.name == std::string("iq2xxs")) b.push_back({grid_iq2xxs.handle, 0, hb(grid_iq2xxs)});
      if (f.name == std::string("iq2xs")) b.push_back({grid_iq2xs.handle, 0, hb(grid_iq2xs)});
      if (f.name == std::string("iq3xxs") || f.name == std::string("iq2xxs") || f.name == std::string("iq2xs")) b.push_back({ksigns_iq2xxs.handle, 0, hb(ksigns_iq2xxs)});
      b.push_back({scratch.handle, 0, hb(scratch)});
      b.push_back({wstage.handle, 0, hb(wstage)});
      b.push_back({ostage.handle, 0, hb(ostage)});
      b.push_back({out.handle, 0, hb(out)});
      Dispatch(gpu, exe, ("yah_ffn_gemm_" + std::string(f.name)).c_str(), mt, 1, 1,
               32, 1, 1, b);
    };
    auto run_swiglu = [&](const std::string& wname) {
      const auto* tw = find(wname);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(tw->type), &f)) throw LoomError("no swiglu port for type on " + wname);
      const std::uint32_t mt = static_cast<std::uint32_t>(tw->dims[1] / 16);
      const std::uint32_t kb = static_cast<std::uint32_t>(tw->dims[0] / f.qk);
      const Imported w = ImportTensor(gpu, gguf, *tw);
      LoomExecutable& exe = load(dir + "/gemm_swiglu_" + f.name + "_" +
                                 std::to_string(mt) + "_" + std::to_string(kb) + ".hal");
      std::vector<hrx_buffer_ref_t> b = {{w.buf.handle, w.offset, w.bytes}};
      if (f.name == std::string("iq3s")) b.push_back({grid_iq3s.handle, 0, hb(grid_iq3s)});
      if (f.name == std::string("iq3xxs")) b.push_back({grid_iq3xxs.handle, 0, hb(grid_iq3xxs)});
      if (f.name == std::string("iq2xxs")) b.push_back({grid_iq2xxs.handle, 0, hb(grid_iq2xxs)});
      if (f.name == std::string("iq2xs")) b.push_back({grid_iq2xs.handle, 0, hb(grid_iq2xs)});
      if (f.name == std::string("iq3xxs") || f.name == std::string("iq2xxs") || f.name == std::string("iq2xs")) b.push_back({ksigns_iq2xxs.handle, 0, hb(ksigns_iq2xxs)});
      b.push_back({scratch.handle, 0, hb(scratch)});
      b.push_back({gateffn.handle, 0, hb(gateffn)});
      b.push_back({uwstage.handle, 0, hb(uwstage)});
      b.push_back({ostage.handle, 0, hb(ostage)});
      b.push_back({ffnup.handle, 0, hb(ffnup)});
      Dispatch(gpu, exe, ("yah_ffn_gemm_" + std::string(f.name) + "_swiglu").c_str(),
               mt, 1, 1, 32, 1, 1, b);
    };
    auto run_residual = [&](const std::string& wname, const LoomBuffer& input) {
      const auto* tw = find(wname);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(tw->type), &f)) throw LoomError("no residual port for type on " + wname);
      const std::uint32_t mt = static_cast<std::uint32_t>(tw->dims[1] / 16);
      const std::uint32_t kb = static_cast<std::uint32_t>(tw->dims[0] / f.qk);
      const Imported w = ImportTensor(gpu, gguf, *tw);
      LoomExecutable& exe = load(dir + "/gemm_residual_" + f.name + "_" +
                                 std::to_string(mt) + "_" + std::to_string(kb) + ".hal");
      std::vector<hrx_buffer_ref_t> b = {{w.buf.handle, w.offset, w.bytes}};
      if (f.name == std::string("iq3s")) b.push_back({grid_iq3s.handle, 0, hb(grid_iq3s)});
      if (f.name == std::string("iq3xxs")) b.push_back({grid_iq3xxs.handle, 0, hb(grid_iq3xxs)});
      if (f.name == std::string("iq2xxs")) b.push_back({grid_iq2xxs.handle, 0, hb(grid_iq2xxs)});
      if (f.name == std::string("iq2xs")) b.push_back({grid_iq2xs.handle, 0, hb(grid_iq2xs)});
      if (f.name == std::string("iq3xxs") || f.name == std::string("iq2xxs") || f.name == std::string("iq2xs")) b.push_back({ksigns_iq2xxs.handle, 0, hb(ksigns_iq2xxs)});
      b.push_back({input.handle, 0, hb(input)});
      b.push_back({wstage.handle, 0, hb(wstage)});
      b.push_back({ostage.handle, 0, hb(ostage)});
      b.push_back({hidden.handle, 0, hb(hidden)});
      Dispatch(gpu, exe, ("yah_ffn_gemm_" + std::string(f.name) + "_residual").c_str(),
               mt, 1, 1, 32, 1, 1, b);
    };

    gpu.Synchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (std::uint32_t l = 0; l < cfg.main_block_count(); ++l) {
      const std::string pre = "blk." + std::to_string(l) + ".";
      const bool full = cfg.IsFullAttention(l);
      run_norm(pre + "attn_norm.weight");
      if (full) {
        const std::uint32_t ai = l / cfg.full_attention_interval;
        run_kstore(pre + "attn_q.weight", qkv);
        run_kstore(pre + "attn_k.weight", kbuf);
        run_kstore(pre + "attn_v.weight", vbuf);
        {
          std::vector<hrx_buffer_ref_t> b = {
              {qkv.handle, 0, std::size_t{kB} * kQProj * 4},
              {q.handle, 0, hb(q)}, {gate.handle, 0, hb(gate)}};
          Dispatch(gpu, e_unpack, "yah_unpack_qg", 24, kB, 1, 256, 1, 1, b);
        }
        const auto* qn = find(pre + "attn_q_norm.weight");
        const auto* kn = find(pre + "attn_k_norm.weight");
        LoomBuffer wqn = gpu.Allocate(std::size_t{kHeadDim} * 4);
        LoomBuffer wkn = gpu.Allocate(std::size_t{kHeadDim} * 4);
        gpu.H2D(wqn, gguf.Data(*qn), std::size_t{kHeadDim} * 4);
        gpu.H2D(wkn, gguf.Data(*kn), std::size_t{kHeadDim} * 4);
        const std::size_t koff = std::size_t{ai} * kCache;
        {
          std::vector<hrx_buffer_ref_t> b = {
              {q.handle, 0, hb(q)}, {kbuf.handle, 0, hb(kbuf)},
              {vbuf.handle, 0, hb(vbuf)}, {wqn.handle, 0, hb(wqn)},
              {wkn.handle, 0, hb(wkn)}, {q.handle, 0, hb(q)},
              {kbuf.handle, 0, hb(kbuf)}, {kc32.handle, 0, hb(kc32)},
              {vc32.handle, 0, hb(vc32)}, {kv16.handle, 0, std::size_t{kCache} * 2},
              {kv16.handle, koff * 2 + std::size_t{kCache} * 2, std::size_t{kCache} * 2},
              {eps.handle, 0, 4}};
          Dispatch(gpu, e_rope, "yah_fused_qk_rope_batched", 28, kB, 1, 256, 1, 1, b);
        }
        {
          std::vector<hrx_buffer_ref_t> b = {
              {q.handle, 0, hb(q)}, {gate.handle, 0, hb(gate)},
              {kv16.handle, 0, std::size_t{kCache} * 2},
              {kv16.handle, koff * 2 + std::size_t{kCache} * 2, std::size_t{kCache} * 2},
              {aout.handle, 0, hb(aout)}, {lse.handle, 0, hb(lse)}};
          Dispatch(gpu, e_wmma, "yah_attn_wmma", kHeads, kB, 1, 32, 1, 1, b);
        }
        {
          std::vector<hrx_buffer_ref_t> b = {
              {aout.handle, 0, hb(aout)}, {scratch.handle, 0, hb(scratch)}};
          Dispatch(gpu, e_cast, "yah_half_cast", 120, 1, 1, 256, 1, 1, b);
        }
        run_residual(pre + "attn_output.weight", scratch);
      } else {
        const std::uint32_t si = l - l / cfg.full_attention_interval;
        run_kstore(pre + "attn_qkv.weight", qkv);
        run_kstore(pre + "attn_gate.weight", gate);
        run_kstore(pre + "ssm_alpha.weight", alpha);
        run_kstore(pre + "ssm_beta.weight", beta);
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
          std::vector<hrx_buffer_ref_t> b = {
              {qkv.handle, 0, std::size_t{kB} * kQkv * 4},
              {wconv.handle, 0, hb(wconv)},
              {conv_state.handle, cs_off, std::size_t{kQkv} * 4 * 4},
              {conv_out.handle, 0, hb(conv_out)}};
          Dispatch(gpu, e_conv, "yah_ssm_conv", 40, kB, 1, 256, 1, 1, b);
        }
        {
          std::vector<hrx_buffer_ref_t> b = {
              {conv_out.handle, 0, hb(conv_out)}, {kqbuf.handle, 0, hb(kqbuf)}};
          Dispatch(gpu, e_prepkq, "yah_deltanet_prep_kq", kKh, kB, 1, 32, 1, 1, b);
        }
        {
          std::vector<hrx_buffer_ref_t> b = {
              {alpha.handle, 0, hb(alpha)}, {beta.handle, 0, hb(beta)},
              {wa.handle, 0, hb(wa)}, {wdt.handle, 0, hb(wdt)},
              {qkv.handle, 0, std::size_t{kB} * kQkv * 4},
              {conv_state.handle, cs_off, std::size_t{kQkv} * 4 * 4},
              {ab.handle, 0, hb(ab)}};
          Dispatch(gpu, e_prepab, "yah_deltanet_prep_ab", 41, 1, 1, 256, 1, 1, b);
        }
        {
          std::vector<hrx_buffer_ref_t> b = {
              {conv_out.handle, 0, hb(conv_out)}, {kqbuf.handle, 0, hb(kqbuf)},
              {ab.handle, 0, hb(ab)},
              {state.handle, st_off, std::size_t{kTs} * kState * kState * 4},
              {raw.handle, 0, hb(raw)}};
          Dispatch(gpu, e_rowsplit, "yah_deltanet", kTs, 1, 1, 128, 1, 1, b);
        }
        {
          std::vector<hrx_buffer_ref_t> b = {
              {raw.handle, 0, hb(raw)}, {wsn.handle, 0, hb(wsn)},
              {gate.handle, 0, std::size_t{kB} * kInner * 4},
              {scratch.handle, 0, hb(scratch)}};
          Dispatch(gpu, e_postnorm, "yah_ssm_postnorm_fp16", 30, 1, 1, 256, 1, 1, b);
        }
        run_residual(pre + "ssm_out.weight", scratch);
      }
      run_norm(pre + "post_attention_norm.weight");
      run_kstore(pre + "ffn_gate.weight", gateffn);
      run_swiglu(pre + "ffn_up.weight");
      run_residual(pre + "ffn_down.weight", ffnup);
    }
    gpu.Synchronize();
    const double layer_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("layers_ms=%.1f\n", layer_ms);

    const auto* onw = find("output_norm.weight");
    const auto* ow = find("output.weight");
    LoomBuffer wnorm = gpu.Allocate(std::size_t{kHidden} * 4);
    gpu.H2D(wnorm, gguf.Data(*onw), std::size_t{kHidden} * 4);
    {
      std::vector<hrx_buffer_ref_t> b = {
          {hidden.handle, std::size_t{kB - 1} * kHidden * 4, std::size_t{kHidden} * 4},
          {wnorm.handle, 0, hb(wnorm)}, {normed.handle, 0, hb(normed)}};
      Dispatch(gpu, e_rms, "yah_rmsnorm", 1, 1, 1, 32, 1, 1, b);
    }
    {
      const Imported w = ImportTensor(gpu, gguf, *ow);
      std::vector<hrx_buffer_ref_t> b = {
          {w.buf.handle, w.offset, w.bytes}, {normed.handle, 0, hb(normed)},
          {logits.handle, 0, hb(logits)}};
      Dispatch(gpu, e_gemv, "yah_gemv_q6k", kVocab, 1, 1, 32, 1, 1, b);
    }
    {
      std::vector<hrx_buffer_ref_t> b = {
          {logits.handle, 0, hb(logits)}, {token.handle, 0, 4}};
      Dispatch(gpu, e_argmax, "yah_argmax", 1, 1, 1, 32, 1, 1, b);
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
    std::fprintf(stderr, "loom_forward_target: %s\n", error.what());
    return 1;
  }
}