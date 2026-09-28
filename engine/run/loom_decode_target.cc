// loom_decode_target: single-token decode forward on Loom through HRX, no HIP.
//
// Runs the IQ4_XS shard one token at a time (the 5 prompt tokens then the
// generated ones) using the dedicated decode kernels for attention and the
// recurrent state, and the token_tiles=1 prefill GEMM HALs for the projections
// (only token 0 of each 64-token tile is used, which is correct but wasteful).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "model/loom_runtime.hpp"

using namespace yah::model;  // NOLINT(google-build-using-namespace)

namespace {
constexpr std::uint32_t kP = 64;
constexpr std::uint32_t kHidden = 5120;
constexpr std::uint32_t kFfn = 17408;
constexpr std::uint32_t kAttn = 6144;
constexpr std::uint32_t kQProj = 12288;
constexpr std::uint32_t kKv = 1024;
constexpr std::uint32_t kInner = 6144;
constexpr std::uint32_t kQkv = 10240;
constexpr std::uint32_t kHeadsV = 48;
constexpr std::uint32_t kTs = 48;
constexpr std::uint32_t kKh = 16;
constexpr std::uint32_t kState = 128;
constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kVocab = 248320;
// hidden elements per dispatch: 5120 rows x the 64-token tile.
constexpr std::uint32_t kOutTotal = kHidden * 64;
constexpr std::uint32_t kSplit = 4;
static_assert(kSplit % 2 == 0, "kSplit must be even");
constexpr std::uint32_t kMaxContext = 64;
constexpr std::uint32_t kCacheLayer = kMaxContext * kKvHeads * kHeadDim;
constexpr std::uint32_t kConvState = kQkv * 4;
constexpr std::uint32_t kStateElems = kHeadsV * kState * kState;
constexpr std::uint32_t kFirst = 5;
constexpr std::uint32_t kGen = 16;
constexpr std::uint32_t kSteps = kFirst + kGen - 1;
const std::uint32_t kPrompt[kFirst] = {760, 6511, 314, 9338, 369};

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

struct Imported { hrx_buffer_t handle; std::size_t offset; std::size_t bytes; };


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
    std::fprintf(stderr, "usage: loom_decode_target <model.gguf> <haldir> <out.bin> [gen]\n");
    return 2;
  }
  const char* model = argv[1];
  const std::string dir = argv[2];
  const char* out_path = argv[3];
  const std::uint32_t gen_count = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : kGen;
  try {
    auto gguf = yah::core::Gguf::Open(model);
    const auto cfg = yah::core::Qwen35Config::FromGguf(gguf);
    LoomDevice gpu;
    // Import the whole GGUF tensor-data region once: every tensor is an offset
    // into it, instead of one hrx_allocator_import_buffer per dispatch.
    LoomBuffer weights;
    std::size_t weights_delta = 0;
    {
      const std::uint8_t* wbase = gguf.tensor_data_base();
      const std::uintptr_t page = 4096;
      const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(wbase) & ~(page - 1);
      weights_delta = reinterpret_cast<std::uintptr_t>(wbase) - start;
      weights = gpu.Import(reinterpret_cast<void*>(start),
                           gguf.tensor_data_size() + weights_delta);
    }
    auto ImportTensor = [&](const yah::core::TensorInfo& t) -> Imported {
      return {weights.handle, weights_delta + static_cast<std::size_t>(t.offset),
              static_cast<std::size_t>(t.bytes)};
    };
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
    const auto hb = [](const LoomBuffer& b) { return b.size; };

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
    LoomBuffer scratch = gpu.Allocate(std::size_t{kP} * kHidden * 2);
    LoomBuffer scratch2 = gpu.Allocate(std::size_t{kP} * kAttn * 2);
    LoomBuffer reszero = gpu.Allocate(std::size_t{kHidden} * 4);
    LoomBuffer sumout = gpu.Allocate(std::size_t{kHidden} * 4);
    LoomBuffer ffnup = gpu.Allocate(std::size_t{kP} * kFfn * 2);
    LoomBuffer gateffn = gpu.Allocate(std::size_t{kP} * kFfn * 4);
    LoomBuffer qkv = gpu.Allocate(std::size_t{kP} * kQProj * 4);
    LoomBuffer kbuf = gpu.Allocate(std::size_t{kP} * kKv * 4);
    LoomBuffer vbuf = gpu.Allocate(std::size_t{kP} * kKv * 4);
    LoomBuffer q = gpu.Allocate(std::size_t{kAttn} * 4);
    LoomBuffer gate = gpu.Allocate(std::size_t{kP} * kInner * 4);
    LoomBuffer aout = gpu.Allocate(std::size_t{kAttn} * 4);
    LoomBuffer ssmout = gpu.Allocate(std::size_t{kInner} * 4);
    LoomBuffer convout = gpu.Allocate(std::size_t{kQkv} * 4);
    LoomBuffer alpha = gpu.Allocate(std::size_t{kP} * kTs * 4);
    LoomBuffer beta = gpu.Allocate(std::size_t{kP} * kTs * 4);
    LoomBuffer normed = gpu.Allocate(std::size_t{kHidden} * 4);
    LoomBuffer logits = gpu.Allocate(std::size_t{kVocab} * 4);
    LoomBuffer token = gpu.Allocate(4);
    LoomBuffer wstage = gpu.Allocate(std::size_t{kFfn} * 16 * 2);
    LoomBuffer uwstage = gpu.Allocate(std::size_t{kFfn} * 16 * 2);
    LoomBuffer ostage = gpu.Allocate(std::size_t{4} * kHidden * 64 * 4);
    LoomBuffer partial = gpu.Allocate(std::size_t{4} * kOutTotal * 4);
    LoomBuffer hidden2 = gpu.Allocate(std::size_t{kOutTotal} * 4);
    LoomBuffer kv16 = gpu.Allocate(std::size_t{2} * 16 * kCacheLayer * 2);
    LoomBuffer cache32 = gpu.Allocate(std::size_t{kCacheLayer} * 4);
    LoomBuffer convstate = gpu.Allocate(std::size_t{48} * kConvState * 4);
    LoomBuffer dstates = gpu.Allocate(std::size_t{48} * kStateElems * 4);
    LoomBuffer dpos = gpu.Allocate(4);
    LoomBuffer eps = gpu.Allocate(4);

    { std::vector<float> z(std::size_t{kHidden}, 0.0f); gpu.H2D(reszero, z.data(), z.size() * 4); }
    { std::vector<std::uint8_t> z(std::size_t{48} * kConvState * 4, 0); gpu.H2D(convstate, z.data(), z.size()); }
    { std::vector<std::uint8_t> z(std::size_t{48} * kStateElems * 4, 0); gpu.H2D(dstates, z.data(), z.size()); }
    { std::vector<std::uint8_t> z(std::size_t{2} * 16 * kCacheLayer * 2, 0); gpu.H2D(kv16, z.data(), z.size()); }
    const float epsv = 1.0e-6f; gpu.H2D(eps, &epsv, 4);

    LoomExecutable& e_norm = load(dir + "/norm.hal");
    LoomExecutable& e_unpack = load(dir + "/unpack.hal");
    LoomExecutable& e_cast = load(dir + "/cast.hal");
    LoomExecutable& e_rope = load(dir + "/rope.hal");
    LoomExecutable& e_ssmconv = load(dir + "/ssmconv.hal");
    LoomExecutable& e_dn = load(dir + "/deltanet.hal");
    LoomExecutable& e_rms = load(dir + "/rmsnorm.hal");
    LoomExecutable& e_gemv = load(dir + "/gemv.hal");
    LoomExecutable& e_argmax = load(dir + "/argmax.hal");
    LoomExecutable& e_accum = load(dir + "/accum.hal");

    auto run_norm = [&](const std::string& wname) {
      const auto* tw = find(wname);
      LoomBuffer w = gpu.Allocate(std::size_t{kHidden} * 4);
      gpu.H2D(w, gguf.Data(*tw), std::size_t{kHidden} * 4);
      std::vector<hrx_buffer_ref_t> b = {
          {hidden.handle, 0, hb(hidden)}, {reszero.handle, 0, hb(reszero)},
          {w.handle, 0, hb(w)}, {sumout.handle, 0, hb(sumout)},
          {scratch.handle, 0, hb(scratch)}};
      Dispatch(gpu, e_norm, "yah_half_norm", 1, 1, 1, 32, 1, 1, b);
    };
    auto tables = [&](const std::string& f, std::vector<hrx_buffer_ref_t>* b) {
      if (f == "iq3s") b->push_back({grid_iq3s.handle, 0, hb(grid_iq3s)});
      if (f == "iq3xxs") b->push_back({grid_iq3xxs.handle, 0, hb(grid_iq3xxs)});
      if (f == "iq2xxs") b->push_back({grid_iq2xxs.handle, 0, hb(grid_iq2xxs)});
      if (f == "iq2xs") b->push_back({grid_iq2xs.handle, 0, hb(grid_iq2xs)});
      if (f == "iq3xxs" || f == "iq2xxs" || f == "iq2xs") b->push_back({ksigns_iq2xxs.handle, 0, hb(ksigns_iq2xxs)});
    };
    auto run_kstore = [&](const std::string& wname, const LoomBuffer& out) {
      const auto* tw = find(wname);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(tw->type), &f)) throw LoomError("no kStore port for type on " + wname);
      const std::uint32_t mt = static_cast<std::uint32_t>(tw->dims[1] / 16);
      const std::uint32_t kb = static_cast<std::uint32_t>(tw->dims[0] / f.qk);
      const Imported w = ImportTensor(*tw);
      LoomExecutable& exe = load(dir + "/gemm_kstore_" + f.name + "_" + std::to_string(mt) + "_" + std::to_string(kb) + ".hal");
      std::vector<hrx_buffer_ref_t> b = {{w.handle, w.offset, w.bytes}};
      tables(f.name, &b);
      b.push_back({scratch.handle, 0, hb(scratch)});
      b.push_back({wstage.handle, 0, hb(wstage)});
      b.push_back({ostage.handle, 0, hb(ostage)});
      b.push_back({out.handle, 0, hb(out)});
      Dispatch(gpu, exe, ("yah_ffn_gemm_" + std::string(f.name)).c_str(), mt, 1, 1, 32, 1, 1, b);
    };
    auto run_swiglu = [&](const std::string& wname) {
      const auto* tw = find(wname);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(tw->type), &f)) throw LoomError("no swiglu port for type on " + wname);
      const std::uint32_t mt = static_cast<std::uint32_t>(tw->dims[1] / 16);
      const std::uint32_t kb = static_cast<std::uint32_t>(tw->dims[0] / f.qk);
      const Imported w = ImportTensor(*tw);
      LoomExecutable& exe = load(dir + "/gemm_swiglu_" + f.name + "_" + std::to_string(mt) + "_" + std::to_string(kb) + ".hal");
      std::vector<hrx_buffer_ref_t> b = {{w.handle, w.offset, w.bytes}};
      tables(f.name, &b);
      b.push_back({scratch.handle, 0, hb(scratch)});
      b.push_back({gateffn.handle, 0, hb(gateffn)});
      b.push_back({uwstage.handle, 0, hb(uwstage)});
      b.push_back({ostage.handle, 0, hb(ostage)});
      b.push_back({ffnup.handle, 0, hb(ffnup)});
      Dispatch(gpu, exe, ("yah_ffn_gemm_" + std::string(f.name) + "_swiglu").c_str(), mt, 1, 1, 32, 1, 1, b);
    };
    auto run_residual = [&](const std::string& wname, const LoomBuffer& input) {
      const auto* tw = find(wname);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(tw->type), &f)) throw LoomError("no residual port for type on " + wname);
      const std::uint32_t mt = static_cast<std::uint32_t>(tw->dims[1] / 16);
      const std::uint32_t kb = static_cast<std::uint32_t>(tw->dims[0] / f.qk);
      const Imported w = ImportTensor(*tw);
      LoomExecutable& exe = load(dir + "/gemm_residual_" + f.name + "_" + std::to_string(mt) + "_" + std::to_string(kb) + ".hal");
      std::vector<hrx_buffer_ref_t> b = {{w.handle, w.offset, w.bytes}};
      tables(f.name, &b);
      b.push_back({input.handle, 0, hb(input)});
      b.push_back({wstage.handle, 0, hb(wstage)});
      b.push_back({ostage.handle, 0, hb(ostage)});
      b.push_back({partial.handle, 0, hb(partial)});
      Dispatch(gpu, exe, ("yah_ffn_gemm_" + std::string(f.name) + "_residual").c_str(), mt, 1, 4, 32, 1, 1, b);
      for (std::uint32_t s = 0; s < kSplit; ++s) {
        const LoomBuffer& src = (s % 2 == 0) ? hidden : hidden2;
        const LoomBuffer& dst = (s % 2 == 0) ? hidden2 : hidden;
        std::vector<hrx_buffer_ref_t> r = {
            {src.handle, 0, hb(src)},
            {partial.handle, std::size_t{s} * kOutTotal * 4, std::size_t{kOutTotal} * 4},
            {dst.handle, 0, hb(dst)}};
        Dispatch(gpu, e_accum, "yah_residual_1d", kOutTotal / 256, 1, 1, 256, 1, 1, r);
      }
    };
    const auto* emb = find("token_embd.weight");
    std::vector<float> host_hidden(std::size_t{kHidden}, 0.0f);
    const auto* onw = find("output_norm.weight");
    const auto* ow = find("output.weight");
    LoomBuffer wnorm = gpu.Allocate(std::size_t{kHidden} * 4);
    gpu.H2D(wnorm, gguf.Data(*onw), std::size_t{kHidden} * 4);
    const Imported owt = ImportTensor(*ow);

    std::vector<std::uint32_t> gen;
    for (std::uint32_t pos = 0; pos < kFirst + gen_count - 1; ++pos) {
      const std::uint32_t tok = (pos < kFirst) ? kPrompt[pos] : gen[pos - kFirst];
      if (static_cast<std::uint32_t>(emb->type) == 23) DequantIq4XsRow(gguf.Data(*emb), tok, host_hidden.data());
      else DequantQ4KRow(gguf.Data(*emb), tok, host_hidden.data());
      gpu.H2D(hidden, host_hidden.data(), std::size_t{kHidden} * 4);
      const std::int32_t p32 = static_cast<std::int32_t>(pos);
      gpu.H2D(dpos, &p32, 4);

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
                {qkv.handle, 0, std::size_t{kQProj} * 4},
                {q.handle, 0, hb(q)}, {gate.handle, 0, hb(gate)}};
            Dispatch(gpu, e_unpack, "yah_unpack_qg", kHeads, 1, 1, 256, 1, 1, b);
          }
          const auto* qn = find(pre + "attn_q_norm.weight");
          const auto* kn = find(pre + "attn_k_norm.weight");
          LoomBuffer wqn = gpu.Allocate(std::size_t{kHeadDim} * 4);
          LoomBuffer wkn = gpu.Allocate(std::size_t{kHeadDim} * 4);
          gpu.H2D(wqn, gguf.Data(*qn), std::size_t{kHeadDim} * 4);
          gpu.H2D(wkn, gguf.Data(*kn), std::size_t{kHeadDim} * 4);
          const std::size_t ksl = std::size_t{ai} * kCacheLayer;
          const std::size_t vsl = (std::size_t{16} + ai) * kCacheLayer;
          {
            std::vector<hrx_buffer_ref_t> b = {
                {q.handle, 0, hb(q)}, {kbuf.handle, 0, hb(kbuf)},
                {vbuf.handle, 0, hb(vbuf)}, {wqn.handle, 0, hb(wqn)},
                {wkn.handle, 0, hb(wkn)}, {q.handle, 0, hb(q)},
                {kbuf.handle, 0, hb(kbuf)}, {cache32.handle, 0, hb(cache32)},
                {cache32.handle, 0, hb(cache32)},
                {kv16.handle, ksl * 2, std::size_t{kCacheLayer} * 2},
                {kv16.handle, vsl * 2, std::size_t{kCacheLayer} * 2},
                {dpos.handle, 0, 4}, {eps.handle, 0, 4}};
            Dispatch(gpu, e_rope, "yah_fused_qk_rope", kHeads + kKvHeads, 1, 1, 256, 1, 1, b);
          }
          LoomExecutable& e_attn = load(dir + "/attn_" + std::to_string(pos) + ".hal");
          {
            std::vector<hrx_buffer_ref_t> b = {
                {q.handle, 0, hb(q)},
                {kv16.handle, ksl * 2, std::size_t{kCacheLayer} * 2},
                {kv16.handle, vsl * 2, std::size_t{kCacheLayer} * 2},
                {gate.handle, 0, hb(gate)}, {aout.handle, 0, hb(aout)}};
            Dispatch(gpu, e_attn, "yah_decode_attn", kHeads, 1, 1, 32, 1, 1, b);
          }
          {
            std::vector<hrx_buffer_ref_t> b = {{aout.handle, 0, hb(aout)},
                                               {scratch2.handle, 0, hb(scratch2)}};
            Dispatch(gpu, e_cast, "yah_half_cast", kAttn / 256, 1, 1, 256, 1, 1, b);
          }
          run_residual(pre + "attn_output.weight", scratch2);
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
          const std::size_t cso = std::size_t{si} * kConvState * 4;
          const std::size_t sto = std::size_t{si} * kStateElems * 4;
          {
            std::vector<hrx_buffer_ref_t> b = {
                {qkv.handle, 0, std::size_t{kQkv} * 4}, {wconv.handle, 0, hb(wconv)},
                {convstate.handle, cso, std::size_t{kConvState} * 4},
                {convout.handle, 0, hb(convout)}};
            Dispatch(gpu, e_ssmconv, "yah_ssm_conv_decode", (kQkv + 255) / 256, 1, 1, 256, 1, 1, b);
          }
          {
            std::vector<hrx_buffer_ref_t> b = {
                {convout.handle, 0, hb(convout)},
                {dstates.handle, sto, std::size_t{kStateElems} * 4},
                {alpha.handle, 0, hb(alpha)}, {beta.handle, 0, hb(beta)},
                {wa.handle, 0, hb(wa)}, {wdt.handle, 0, hb(wdt)},
                {wsn.handle, 0, hb(wsn)}, {gate.handle, 0, hb(gate)},
                {ssmout.handle, 0, hb(ssmout)}};
            Dispatch(gpu, e_dn, "yah_deltanet_decode", kHeadsV, 1, 1, 32, 1, 1, b);
          }
          {
            std::vector<hrx_buffer_ref_t> b = {{ssmout.handle, 0, hb(ssmout)},
                                               {scratch2.handle, 0, hb(scratch2)}};
            Dispatch(gpu, e_cast, "yah_half_cast", kInner / 256, 1, 1, 256, 1, 1, b);
          }
          run_residual(pre + "ssm_out.weight", scratch2);
        }
        run_norm(pre + "post_attention_norm.weight");
        run_kstore(pre + "ffn_gate.weight", gateffn);
        run_swiglu(pre + "ffn_up.weight");
        run_residual(pre + "ffn_down.weight", ffnup);
      }

      {
        std::vector<hrx_buffer_ref_t> b = {
            {hidden.handle, 0, hb(hidden)}, {wnorm.handle, 0, hb(wnorm)},
            {normed.handle, 0, hb(normed)}};
        Dispatch(gpu, e_rms, "yah_rmsnorm", 1, 1, 1, 32, 1, 1, b);
      }
      {
        std::vector<hrx_buffer_ref_t> b = {
            {owt.handle, owt.offset, owt.bytes}, {normed.handle, 0, hb(normed)},
            {logits.handle, 0, hb(logits)}};
        Dispatch(gpu, e_gemv, "yah_gemv_q6k", kVocab, 1, 1, 32, 1, 1, b);
      }
      {
        std::vector<hrx_buffer_ref_t> b = {{logits.handle, 0, hb(logits)},
                                           {token.handle, 0, 4}};
        Dispatch(gpu, e_argmax, "yah_argmax", 1, 1, 1, 32, 1, 1, b);
      }
      gpu.Synchronize();
      std::uint32_t out = 0;
      gpu.D2H(token, &out, 4, 0);
      std::fprintf(stderr, "pos %u token %u\n", pos, out);
      if (pos + 1 >= kFirst) gen.push_back(out);
    }

    std::printf("generated_ids=");
    for (std::size_t i = 0; i < gen.size(); ++i)
      std::printf("%u%s", gen[i], i + 1 == gen.size() ? "" : " ");
    std::printf("\n");
    std::printf("n_generated=%zu\n", gen.size());
    FILE* fo = std::fopen(out_path, "wb");
    std::fwrite(gen.data(), 4, gen.size(), fo);
    std::fclose(fo);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "loom_decode_target: %s\n", error.what());
    return 1;
  }
}
