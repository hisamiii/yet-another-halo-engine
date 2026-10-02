// loom_forward_pp: the Loom prefill on HRX. Runs the prompt through the 64 layers, one chunk of B tokens at a time.
//
// usage: loom_forward_pp <model.gguf> <haldir> <out-prefix> [tokens] [ids-file]
// outputs: <out-prefix>.logits (vocab f32, last token) and <out-prefix>.hidden (B x 5120 f32, token major)
// env: YAH_LOGITS_FROM, YAH_ROWSTATS*, YAH_KV_HOOK*, YAH_GEN + YAH_DECODE_HAL (see below)
//
// The HAL set comes from tools/emit_prefill_pp.py; <haldir>/dispatch.txt gives the launch geometry of each HAL.
// Every kernel carries the token dimension in its grid:
//   GEMM family    grid (m_tiles / rowgrp, token_tiles, 1)
//   norm/conv/...  grid (tiles, B) or (B, 1, 1)
//   attention      grid (query-token tiles, head groups), causal over keys 0..token
// Chunk 0 starts with zeroed recurrent state (ssm conv ring, DeltaNet state); later chunks carry it forward.
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "model/loom_decoder.hpp"
#include "model/loom_runtime.hpp"

using namespace yah::model;  // NOLINT(google-build-using-namespace)

namespace {
// GEMM tokens per workgroup for a HAL that dispatch.txt does not list.
#ifndef YAH_TOKEN_TILE
#define YAH_TOKEN_TILE 64
#endif
constexpr std::uint32_t kTile = YAH_TOKEN_TILE;
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
constexpr std::uint32_t kVocab = 248320;
// One KV cache row: all KV heads of one token.
constexpr std::uint32_t kKvRow = kKvHeads * kHeadDim;  // 1024

const float kKvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

struct Fmt {
  const char* name;
  std::uint32_t qk;
};
bool FmtOf(std::uint32_t type, Fmt* out) {
  switch (type) {
    case 12:
      *out = {"q4k", 256};
      return true;
    case 13:
      *out = {"q5k", 256};
      return true;
    case 14:
      *out = {"q6k", 256};
      return true;
    case 11:
      *out = {"q3k", 256};
      return true;
    case 23:
      *out = {"iq4xs", 256};
      return true;
    case 21:
      *out = {"iq3s", 256};
      return true;
    case 18:
      *out = {"iq3xxs", 256};
      return true;
    case 20:
      *out = {"iq4nl", 32};
      return true;
    case 17:
      *out = {"iq2xs", 256};
      return true;
    case 8:
      *out = {"q8_0", 32};
      return true;
    case 16:
      *out = {"iq2xxs", 256};
      return true;
    case 10:
      *out = {"q2k", 256};
      return true;
    default:
      return false;
  }
}

struct Imported {
  hrx_buffer_t handle;
  std::size_t offset;
  std::size_t bytes;
};

std::uint32_t g_b = 0;   // prompt tokens in this run
std::uint32_t g_tt = 0;  // GEMM token tiles = g_b / g_gtile
std::uint32_t g_gtile = kTile;  // fallback GEMM tile when dispatch.txt says nothing

void Dispatch(LoomDevice& gpu, const LoomExecutable& exe, const char* name, std::uint32_t gx, std::uint32_t gy,
              std::uint32_t gz, std::uint32_t sx, std::uint32_t sy, std::uint32_t sz,
              const std::vector<hrx_buffer_ref_t>& b) {
  // The export's own workgroup size wins; sx is only the fallback for metadata without one.
  const std::uint32_t ordinal = exe.OrdinalOrZero(name);
  const std::uint32_t ws = exe.WorkgroupSize(ordinal);
  gpu.Dispatch(exe, ordinal, LoomDevice::Config(gx, gy, gz, ws ? ws : sx, sy, sz), nullptr, 0, b.data(), b.size());
}

// The launch geometry each HAL was compiled for, from <haldir>/dispatch.txt.
// One line per HAL: "<basename> <tokens> <rowgrp> <token_tiles>".
// Marker rows (e.g. "kv_paged 0 0 0") flag features of the set.
// Read the grid from here; never mirror the emitter's arithmetic: a wrong grid silently skips rows.
struct Geom {
  std::uint32_t tokens;  // tokens per workgroup = grid y divisor
  std::uint32_t rowgrp;  // 16-row tiles per workgroup = grid x divisor
  std::uint32_t tt;      // token_tiles the emitter bound into this HAL
};
std::map<std::string, Geom> g_geom;

void LoadDispatch(const std::string& dir) {
  std::ifstream f(dir + "/dispatch.txt");
  if (!f) return;
  std::string name;
  Geom g{};
  while (f >> name >> g.tokens >> g.rowgrp >> g.tt) g_geom[name] = g;
}

// Geometry of one HAL. Exits if the HAL was emitted for another token count: it would compute a silent subset.
Geom GeomOf(const std::string& hal, std::uint32_t B) {
  const auto it = g_geom.find(hal);
  if (it == g_geom.end()) return Geom{kTile, 1, static_cast<std::uint32_t>(B / kTile)};
  const Geom g = it->second;
  if (g.tokens == 0 || g.rowgrp == 0 || B % g.tokens != 0 || B / g.tokens != g.tt) {
    std::fprintf(stderr, "%s: emitted for tile=%u rowgrp=%u token_tiles=%u, but B=%u gives %u tiles\n", hal.c_str(),
                 g.tokens, g.rowgrp, g.tt, B, B / g.tokens);
    std::exit(3);
  }
  return g;
}

void ReadFile(const char* path, void* dst, std::size_t bytes) {
  FILE* f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(1);
  }
  if (std::fread(dst, 1, bytes, f) != bytes) {
    std::exit(1);
  }
  std::fclose(f);
}
float Half(const std::uint8_t* p) {
  _Float16 h;
  std::memcpy(&h, p, 2);
  return (float)h;
}

void DequantQ4KRow(const std::uint8_t* base, std::uint64_t row, float* out) {
  const std::uint32_t nb = kHidden / 256;
  const std::uint8_t* p = base + row * static_cast<std::uint64_t>(nb) * 144;
  for (std::uint32_t b = 0; b < nb; ++b) {
    const std::uint8_t* blk = p + b * 144;
    const float d = Half(blk), dmin = Half(blk + 2);
    const std::uint8_t* sc = blk + 4;
    const std::uint8_t* qs = blk + 16;
    for (std::uint32_t i = 0; i < 256; ++i) {
      const std::uint32_t gg = i / 64, wv = i % 64, lane = wv % 32;
      const bool low = wv < 32;
      const std::uint32_t qb = qs[gg * 32 + lane];
      const std::uint32_t quant = low ? (qb & 15) : (qb >> 4);
      const std::uint32_t j = 2 * gg + (low ? 0 : 1);
      std::uint32_t s, m;
      if (j < 4) {
        s = sc[j] & 63;
        m = sc[j + 4] & 63;
      } else {
        s = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4);
        m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
      }
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
    const std::uint8_t* sl = blk + 4;
    const std::uint8_t* qs = blk + 8;
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

std::vector<std::uint32_t> ParseIds(const char* path) {
  std::ifstream file(path);
  if (!file) throw LoomError(std::string("cannot read ids file ") + path);
  std::vector<std::uint32_t> ids;
  std::uint64_t v;
  while (file >> v) ids.push_back(static_cast<std::uint32_t>(v));
  if (ids.empty()) throw LoomError("ids file is empty");
  return ids;
}
}  // namespace
int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: loom_forward_pp <model.gguf> <haldir> <out-prefix> "
                 "[tokens] [ids-file]\n");
    return 2;
  }
  const char* model = argv[1];
  const std::string dir = argv[2];
  const std::string prefix = argv[3];
  const std::uint32_t want = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 2048;
  const char* ids_path = argc > 5 ? argv[5] : "/home/q/yah-scratch/ids2048.txt";
  try {
    auto gguf = yah::core::Gguf::Open(model);
    const auto cfg = yah::core::Qwen35Config::FromGguf(gguf);
    std::vector<std::uint32_t> ids_all = ParseIds(ids_path);
    g_b = want ? want : static_cast<std::uint32_t>(ids_all.size());
    if (ids_all.size() < g_b) {
      // Pad a short id file by repeating it: a timing run needs only a valid token stream of the right length.
      std::vector<std::uint32_t> grown;
      while (grown.size() < g_b) grown.insert(grown.end(), ids_all.begin(), ids_all.end());
      ids_all.swap(grown);
    }
    LoadDispatch(dir);
    // Chunked prefill: dispatch.txt row "ctx" holds the chunk size (tokens) and the emitted context T_ctx (tt).
    // Every kernel runs at the chunk size, the KV cache holds T_ctx, and the chunks carry the DeltaNet and conv state.
    // T_run (the token count argument) is any multiple of the chunk up to T_ctx; the rest is room for YAH_GEN.
    std::uint32_t T_ctx = g_b;
    const std::uint32_t T_run = g_b;
    if (const auto it = g_geom.find("ctx"); it != g_geom.end()) {
      T_ctx = it->second.tt;
      if (T_run > T_ctx || T_run % it->second.tokens) {
        std::fprintf(stderr,
                     "chunked set: tokens=%u must be a multiple of the chunk %u and at most the emitted context %u\n",
                     T_run, it->second.tokens, T_ctx);
        return 2;
      }
      g_b = it->second.tokens;
    }
    const std::uint32_t n_chunks = T_run / g_b;
    if (g_b % g_gtile) {
      std::fprintf(stderr, "tokens=%u must be a multiple of the token tile=%u\n", g_b, g_gtile);
      return 2;
    }
    g_tt = g_b / g_gtile;
    const std::uint32_t B = g_b;
    const std::size_t kOutTotal = static_cast<std::size_t>(kHidden) * B;
    // One KV slot per full-attention layer. The f16 cache holds kFull K slots, then kFull V slots.
    std::uint32_t kFull = 0;
    for (std::uint32_t l = 0; l < cfg.main_block_count(); ++l)
      if (cfg.IsFullAttention(l)) ++kFull;
    LoomDevice gpu;
    // The layers queue ~950 dispatches and wait once: the runtime wait would busy-poll a host core the whole time.
    gpu.SetSleepSync(200);
    std::fprintf(stderr, "loom_forward_pp: tokens=%u tile=%u token_tiles=%u layers=%u geometry=%zu hal(s)\n", B,
                 g_gtile, g_tt, cfg.main_block_count(), g_geom.size());
    // Import the whole GGUF tensor-data region once; every tensor is an offset into it.
    LoomBuffer weights;
    std::size_t weights_delta = 0;
    {
      const std::uint8_t* wbase = gguf.tensor_data_base();
      const std::uintptr_t page = 4096;
      const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(wbase) & ~(page - 1);
      weights_delta = reinterpret_cast<std::uintptr_t>(wbase) - start;
      const std::size_t wbytes = gguf.tensor_data_size() + weights_delta;
      weights = gpu.Import(reinterpret_cast<void*>(start), wbytes);
    }
    auto ImportTensor = [&](const yah::core::TensorInfo& t) -> Imported {
      return {weights.handle, weights_delta + static_cast<std::size_t>(t.offset), static_cast<std::size_t>(t.bytes)};
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
    LoomBuffer grid_iq3s = gpu.Allocate(std::size_t{512} * 4);
    LoomBuffer grid_iq3xxs = gpu.Allocate(std::size_t{256} * 4);
    LoomBuffer ksigns_iq3xxs = gpu.Allocate(std::size_t{128});
    LoomBuffer grid_iq2xxs = gpu.Allocate(std::size_t{512} * 4);
    LoomBuffer grid_iq2xs = gpu.Allocate(std::size_t{1024} * 4);
    LoomBuffer ksigns_iq2xxs = gpu.Allocate(std::size_t{128});
    {
      std::vector<std::uint8_t> v(512 * 4);
      ReadFile((dir + "/grid_iq3s.bin").c_str(), v.data(), v.size());
      gpu.H2D(grid_iq3s, v.data(), v.size());
    }
    {
      std::vector<std::uint8_t> v(256 * 4);
      ReadFile((dir + "/grid_iq3xxs.bin").c_str(), v.data(), v.size());
      gpu.H2D(grid_iq3xxs, v.data(), v.size());
    }
    {
      std::vector<std::uint8_t> v(128);
      ReadFile((dir + "/ksigns_iq3xxs.bin").c_str(), v.data(), v.size());
      gpu.H2D(ksigns_iq3xxs, v.data(), v.size());
    }
    {
      std::vector<std::uint8_t> v(512 * 4);
      ReadFile((dir + "/grid_iq2xxs.bin").c_str(), v.data(), v.size());
      gpu.H2D(grid_iq2xxs, v.data(), v.size());
    }
    {
      std::vector<std::uint8_t> v(1024 * 4);
      ReadFile((dir + "/grid_iq2xs.bin").c_str(), v.data(), v.size());
      gpu.H2D(grid_iq2xs, v.data(), v.size());
    }
    {
      std::vector<std::uint8_t> v(128);
      ReadFile((dir + "/ksigns_iq2xxs.bin").c_str(), v.data(), v.size());
      gpu.H2D(ksigns_iq2xxs, v.data(), v.size());
    }

    // Activations are token major, [token][row], with row stride = the K extent of the GEMM that reads them.
    // The KV cache has one row per context token; the recurrent state buffers are per layer, not per token.
    const std::size_t kKvCache = static_cast<std::size_t>(T_ctx) * kKvRow;
    LoomBuffer hidden = gpu.Allocate(static_cast<std::size_t>(B) * kHidden * 4);
    LoomBuffer reszero = gpu.Allocate(static_cast<std::size_t>(B) * kHidden * 4);
    LoomBuffer sumout = gpu.Allocate(static_cast<std::size_t>(B) * kHidden * 4);
    LoomBuffer scratch = gpu.Allocate(static_cast<std::size_t>(B) * kFfn * 2);
    LoomBuffer qkv = gpu.Allocate(static_cast<std::size_t>(B) * kQProj * 4);
    LoomBuffer gate = gpu.Allocate(static_cast<std::size_t>(B) * kInner * 4);
    LoomBuffer alpha = gpu.Allocate(static_cast<std::size_t>(B) * kTs * 4);
    LoomBuffer beta = gpu.Allocate(static_cast<std::size_t>(B) * kTs * 4);
    LoomBuffer q = gpu.Allocate(static_cast<std::size_t>(B) * kAttn * 4);
    LoomBuffer kbuf = gpu.Allocate(static_cast<std::size_t>(B) * kKv * 4);
    LoomBuffer vbuf = gpu.Allocate(static_cast<std::size_t>(B) * kKv * 4);
    LoomBuffer aout = gpu.Allocate(static_cast<std::size_t>(B) * kAttn * 4);
    LoomBuffer raw = gpu.Allocate(static_cast<std::size_t>(B) * kInner * 4);
    LoomBuffer conv_out = gpu.Allocate(static_cast<std::size_t>(B) * kQkv * 4);
    LoomBuffer kqbuf = gpu.Allocate(static_cast<std::size_t>(B) * kKh * 3 * 4);
    LoomBuffer ab = gpu.Allocate(static_cast<std::size_t>(B) * kTs * 2 * 4);
    LoomBuffer conv_state = gpu.Allocate(std::size_t{48} * kQkv * 4 * 4);
    LoomBuffer state = gpu.Allocate(std::size_t{48} * kTs * kState * kState * 4);
    // f16 K/V cache: [K | V] x attention layers x context.
    // With "kv16_scratch" (paged or quantized KV) it holds one layer's current chunk only:
    // RoPE writes it (row cur - cache_start) and the paged writers or quantizers read it before the next layer.
    const bool kv16_scratch = g_geom.count("kv16_scratch") != 0;
    const std::size_t kKv16Layer = kv16_scratch ? std::size_t{B} * kKvRow * 2 : kKvCache * 2;
    LoomBuffer kv16 = gpu.Allocate(kv16_scratch ? 2 * kKv16Layer : std::size_t{2} * kFull * kKvCache * 2);
    LoomBuffer kc32 = gpu.Allocate(kKvCache * 4);
    LoomBuffer vc32 = gpu.Allocate(kKvCache * 4);
    LoomBuffer lse = gpu.Allocate(static_cast<std::size_t>(B) * kHeads * 4);
    LoomBuffer eps = gpu.Allocate(4);
    LoomBuffer ffnup = gpu.Allocate(static_cast<std::size_t>(B) * kFfn * 2);
    LoomBuffer gateffn = gpu.Allocate(static_cast<std::size_t>(B) * kFfn * 4);
    // Per-workgroup weight staging: the kernels stage a 16x16 weight tile per K step, not the full 16xK row.
    LoomBuffer gwstage = gpu.Allocate(std::size_t{kFfn} * 16 * 2);
    LoomBuffer uwstage = gpu.Allocate(std::size_t{kFfn} * 16 * 2);
    LoomBuffer wstage = gpu.Allocate(std::size_t{kFfn} * 16 * 2);
    // Epilogue scratch. The compiler declares it over the whole [m_rows][tokens] tile: size it for the widest GEMM.
    LoomBuffer ostage = gpu.Allocate(std::size_t{kFfn} * B * 4);
    LoomBuffer partial = gpu.Allocate(kOutTotal * 4);
    LoomBuffer hidden2 = gpu.Allocate(kOutTotal * 4);
    LoomBuffer normed = gpu.Allocate(std::size_t{kHidden} * 4);
    LoomBuffer logits = gpu.Allocate(std::size_t{kVocab} * 4);
    LoomBuffer token = gpu.Allocate(4);

    const auto hb = [](const LoomBuffer& b) { return b.size; };
    const float epsv = 1.0e-6f;
    gpu.H2D(eps, &epsv, 4);
    {
      std::vector<std::uint8_t> z(std::size_t{48} * kQkv * 4 * 4, 0);
      gpu.H2D(conv_state, z.data(), z.size());
    }
    {
      std::vector<std::uint8_t> z(std::size_t{48} * kTs * kState * kState * 4, 0);
      gpu.H2D(state, z.data(), z.size());
    }
    {
      std::vector<std::uint8_t> z(kv16_scratch ? 2 * kKv16Layer : std::size_t{2} * kFull * kKvCache * 2, 0);
      gpu.H2D(kv16, z.data(), z.size());
    }
    {
      std::vector<std::uint8_t> z(static_cast<std::size_t>(B) * kHidden * 4, 0);
      gpu.H2D(reszero, z.data(), z.size());
    }
    const auto* emb = find("token_embd.weight");
    std::vector<float> host_hidden(static_cast<std::size_t>(B) * kHidden);
    const std::uint8_t* emb_data = gguf.Data(*emb);
    const auto embed = [&](std::uint32_t chunk, LoomBuffer& dst) {
      for (std::uint32_t t = 0; t < B; ++t) {
        const std::uint32_t id = ids_all[std::size_t{chunk} * B + t];
        if (static_cast<std::uint32_t>(emb->type) == 23)
          DequantIq4XsRow(emb_data, id, host_hidden.data() + std::size_t{t} * kHidden);
        else
          DequantQ4KRow(emb_data, id, host_hidden.data() + std::size_t{t} * kHidden);
      }
      gpu.H2D(dst, host_hidden.data(), host_hidden.size() * 4);
    };
    embed(0, hidden);

    LoomExecutable& e_norm = load(dir + "/norm.hal");
    LoomExecutable& e_conv = load(dir + "/conv.hal");
    LoomExecutable& e_prepkq = load(dir + "/prepkq.hal");
    // Optional convkq.hal: the conv with yah_deltanet_prep_kq fused in.
    LoomExecutable* e_convkq = nullptr;
    {
      const std::string path = dir + "/convkq.hal";
      if (FILE* f = std::fopen(path.c_str(), "rb")) {
        std::fclose(f);
        e_convkq = &load(path);
      }
    }
    LoomExecutable& e_prepab = load(dir + "/prepab.hal");
    LoomExecutable& e_rowsplit = load(dir + "/rowsplit.hal");
    LoomExecutable& e_postnorm = load(dir + "/postnorm.hal");
    LoomExecutable& e_unpack = load(dir + "/unpack.hal");
    std::vector<LoomExecutable*> e_ropes, e_wmmas;
    for (std::uint32_t c = 0; c < n_chunks; ++c) {
      e_ropes.push_back(&load(dir + (c ? "/rope_c" + std::to_string(c) + ".hal" : std::string("/rope.hal"))));
      e_wmmas.push_back(&load(dir + (c ? "/wmma_c" + std::to_string(c) + ".hal" : std::string("/wmma.hal"))));
    }
    // Chunked sets only. Runs after every chunk, so the last one leaves the final conv state for the decode handoff.
    LoomExecutable* e_convstate = T_ctx > g_b ? &load(dir + "/convstate.hal") : nullptr;
    // Unpaged set with a "vtrans.hal" row: attention reads V as [kv head][16-key tile][dim][16] f16.
    // yah_transpose_v16 writes that layout per layer.
    // "kv_paged": 256-token pages, one page table per sequence (logical -> physical page, shared by all layers).
    // The table is bound to attention and every cache writer. Per chunk, RoPE stores fp16 K and yah_vtpage fp16 V^T.
    const bool kv_paged = g_geom.count("kv_paged") != 0;
    const std::uint32_t kPages = (T_ctx + 255) / 256;
    LoomBuffer ptab = gpu.Allocate(std::size_t{kv_paged ? kPages : 1} * 4);
    if (kv_paged) {
      std::vector<std::int32_t> pages(kPages);
      for (std::uint32_t i = 0; i < kPages; ++i) pages[i] = static_cast<std::int32_t>(i);
      // Attention does not clamp page table entries: validate them here.
      for (std::int32_t pg : pages)
        if (pg < 0 || pg >= static_cast<std::int32_t>(kPages)) throw LoomError("page table entry out of range");
      gpu.H2D(ptab, pages.data(), pages.size() * 4);
    }
    LoomExecutable* e_vtrans = (!kv_paged && g_geom.count("vtrans.hal")) ? &load(dir + "/vtrans.hal") : nullptr;
    const std::size_t kVtBytes = std::size_t{(T_ctx + 15) / 16 * 16} * kKvRow * 2;
    LoomBuffer vt16 = gpu.Allocate(e_vtrans ? kVtBytes : 4);
    // Quantized K (tools/gen_kvq.py, engine/run/kvq/README.md): "attn_kq8" int8 (kv8a16).
    // "attn_kq4": H256 + asymmetric int4 (kv4a16); its kernel yah_kq4 is in kq8.hal.
    // Per attention layer: codes [T][1024 or 512 B], scales [T][8 or 32] dwords, the channel mean of the first chunk.
    // The quantizers run per chunk on cache slices.
    const bool attn_kq4 = g_geom.count("attn_kq4") != 0;
    const bool attn_kq8 = g_geom.count("attn_kq8") != 0 || attn_kq4;
    LoomExecutable* e_kmean = attn_kq8 ? &load(dir + "/kmean.hal") : nullptr;
    LoomExecutable* e_kq8 = attn_kq8 ? &load(dir + "/kq8.hal") : nullptr;
    const std::size_t kKsBytes = static_cast<std::size_t>(T_ctx) * (attn_kq4 ? 32 : 8) * 4;
    const std::size_t kKqBytes = attn_kq4 ? kKvCache / 2 : kKvCache;
    LoomBuffer kq8buf = gpu.Allocate(attn_kq8 ? std::size_t{kFull} * kKqBytes : 4);
    LoomBuffer ksbuf = gpu.Allocate(attn_kq8 ? std::size_t{kFull} * kKsBytes : 4);
    LoomBuffer kmbuf = gpu.Allocate(attn_kq8 ? std::size_t{kFull} * 4096 : 4);
    // Quantized V^T: "attn_vq8" bytes / "attn_vq4" nibbles per channel per 16-key tile.
    // Per layer: [4][tiles][256] x 16 / 8 B of codes plus (S, C') x 4 B. One HAL per chunk (start_pos): vq8_c<i>.hal.
    const bool attn_vq4 = g_geom.count("attn_vq4") != 0;
    const bool attn_vq8 = g_geom.count("attn_vq8") != 0;
    const bool attn_vqt = attn_vq4 || attn_vq8;
    std::vector<LoomExecutable*> e_vqs;
    for (std::uint32_t c = 0; attn_vqt && c < n_chunks; ++c) {
      const std::string nm = attn_vq8 ? "vq8" : "vq4";
      e_vqs.push_back(&load(dir + "/" + nm + (c ? "_c" + std::to_string(c) : std::string()) + ".hal"));
    }
    const std::size_t kVqBytes = attn_vq8 ? kVtBytes / 2 : kVtBytes / 4, kVqsBytes = kVtBytes / 8;
    LoomBuffer vqbuf = gpu.Allocate(attn_vqt ? std::size_t{kFull} * kVqBytes : 4);
    LoomBuffer vqsbuf = gpu.Allocate(attn_vqt ? std::size_t{kFull} * kVqsBytes : 4);
    // Paged fp16 pools (per layer: K rows / V^T tiles of the whole context) and the per-chunk paged writers.
    const bool paged_f16k = kv_paged && !attn_kq8, paged_f16v = kv_paged && !attn_vqt;
    // "rope_kpaged": RoPE writes the fp16 K rows straight into the paged pool.
    const bool rope_kpaged = g_geom.count("rope_kpaged") != 0;
    if (paged_f16k && !rope_kpaged) throw LoomError("paged fp16 K needs a rope_kpaged HAL set (re-emit)");
    const std::size_t kPoolBytes = std::size_t{kPages} * 256 * kKvRow * 2;
    LoomBuffer kpool = gpu.Allocate(paged_f16k ? std::size_t{kFull} * kPoolBytes : 4);
    LoomBuffer vtpool = gpu.Allocate(paged_f16v ? std::size_t{kFull} * kPoolBytes : 4);
    std::vector<LoomExecutable*> e_vtpages, e_kq8s;
    for (std::uint32_t c = 0; kv_paged && c < n_chunks; ++c) {
      const std::string sfx = c ? "_c" + std::to_string(c) : std::string();
      if (paged_f16v) e_vtpages.push_back(&load(dir + "/vtpage" + sfx + ".hal"));
      if (attn_kq8) e_kq8s.push_back(&load(dir + "/kq8" + sfx + ".hal"));
    }
    const hrx_buffer_ref_t ptab_ref{ptab.handle, 0, std::size_t{kv_paged ? kPages : 1} * 4};
    LoomExecutable& e_cast = load(dir + "/cast.hal");
    LoomExecutable& e_gemv = load(dir + "/gemv.hal");
    LoomExecutable& e_rms = load(dir + "/rmsnorm.hal");
    LoomExecutable& e_argmax = load(dir + "/argmax.hal");
    LoomExecutable& e_accum = load(dir + "/accum.hal");

    // The norm writes the f16 activation into scratch at row stride dim: every GEMM that reads scratch has ktot == dim.
    // Small per-layer weights (norms, conv1d, ssm_a/dt/norm) are views into the one weight import, never copies.
    // Do not refill a shared device buffer here: hrx_synchronous_h2d does not wait for queued dispatches that bind it.
    // Do not allocate per call either: each allocation polls AMDKFD_IOC_WAIT_EVENTS and drains the queue every layer.
    auto run_norm = [&](const std::string& wname) {
      const auto* tw = find(wname);
      const Imported w = ImportTensor(*tw);
      std::vector<hrx_buffer_ref_t> b = {{hidden.handle, 0, hb(hidden)},
                                         {reszero.handle, 0, hb(reszero)},
                                         {w.handle, w.offset, w.bytes},
                                         {sumout.handle, 0, hb(sumout)},
                                         {scratch.handle, 0, hb(scratch)}};
      Dispatch(gpu, e_norm, "yah_half_norm", B, 1, 1, 32, 1, 1, b);
    };
    auto run_kstore = [&](const std::string& wname, const LoomBuffer& out) {
      const auto* tw = find(wname);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(tw->type), &f)) throw LoomError("no kStore port for type on " + wname);
      const std::uint32_t mt = static_cast<std::uint32_t>(tw->dims[1] / 16);
      const std::uint32_t kb = static_cast<std::uint32_t>(tw->dims[0] / f.qk);
      const Imported w = ImportTensor(*tw);
      const std::string hal =
          std::string("gemm_kstore_") + f.name + "_" + std::to_string(mt) + "_" + std::to_string(kb) + ".hal";
      LoomExecutable& exe = load(dir + "/" + hal);
      const Geom gm = GeomOf(hal, B);
      std::vector<hrx_buffer_ref_t> b = {{w.handle, w.offset, w.bytes}};
      if (f.name == std::string("iq3s")) b.push_back({grid_iq3s.handle, 0, hb(grid_iq3s)});
      if (f.name == std::string("iq3xxs")) b.push_back({grid_iq3xxs.handle, 0, hb(grid_iq3xxs)});
      if (f.name == std::string("iq2xxs")) b.push_back({grid_iq2xxs.handle, 0, hb(grid_iq2xxs)});
      if (f.name == std::string("iq2xs")) b.push_back({grid_iq2xs.handle, 0, hb(grid_iq2xs)});
      if (f.name == std::string("iq3xxs") || f.name == std::string("iq2xxs") || f.name == std::string("iq2xs"))
        b.push_back({ksigns_iq2xxs.handle, 0, hb(ksigns_iq2xxs)});
      b.push_back({scratch.handle, 0, hb(scratch)});
      b.push_back({wstage.handle, 0, hb(wstage)});
      b.push_back({ostage.handle, 0, hb(ostage)});
      b.push_back({out.handle, 0, hb(out)});
      Dispatch(gpu, exe, ("yah_ffn_gemm_" + std::string(f.name)).c_str(), mt / gm.rowgrp, B / gm.tokens, 1, 32, 1, 1,
               b);
    };
    // "attn_f16out": the attention HAL stores f16 straight into the o-projection input, so no yah_half_cast pass.
    const bool attn_f16 = g_geom.count("attn_f16out") != 0;
    // gemm_kqg_*: the attention q projection with the q/gate unpack fused in (rows = heads x [256 q | 256 gate]).
    // Returns false if the set has no such HAL; the caller then runs kstore + yah_unpack_qg.
    auto run_kqg = [&](const std::string& wname) -> bool {
      const auto* tw = find(wname);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(tw->type), &f)) return false;
      const std::uint32_t mt = static_cast<std::uint32_t>(tw->dims[1] / 16);
      const std::uint32_t kb = static_cast<std::uint32_t>(tw->dims[0] / f.qk);
      const std::string hal =
          std::string("gemm_kqg_") + f.name + "_" + std::to_string(mt) + "_" + std::to_string(kb) + ".hal";
      if (g_geom.find(hal) == g_geom.end()) return false;
      const Imported w = ImportTensor(*tw);
      LoomExecutable& exe = load(dir + "/" + hal);
      const Geom gm = GeomOf(hal, B);
      std::vector<hrx_buffer_ref_t> b = {{w.handle, w.offset, w.bytes}};
      if (f.name == std::string("iq3s")) b.push_back({grid_iq3s.handle, 0, hb(grid_iq3s)});
      if (f.name == std::string("iq3xxs")) b.push_back({grid_iq3xxs.handle, 0, hb(grid_iq3xxs)});
      if (f.name == std::string("iq2xxs")) b.push_back({grid_iq2xxs.handle, 0, hb(grid_iq2xxs)});
      if (f.name == std::string("iq2xs")) b.push_back({grid_iq2xs.handle, 0, hb(grid_iq2xs)});
      if (f.name == std::string("iq3xxs") || f.name == std::string("iq2xxs") || f.name == std::string("iq2xs"))
        b.push_back({ksigns_iq2xxs.handle, 0, hb(ksigns_iq2xxs)});
      b.push_back({scratch.handle, 0, hb(scratch)});
      b.push_back({wstage.handle, 0, hb(wstage)});
      b.push_back({ostage.handle, 0, hb(ostage)});
      b.push_back({q.handle, 0, hb(q)});
      b.push_back({gate.handle, 0, hb(gate)});
      Dispatch(gpu, exe, ("yah_ffn_gemm_" + std::string(f.name) + "_kqg").c_str(), mt / gm.rowgrp, B / gm.tokens, 1, 32,
               1, 1, b);
      return true;
    };
    auto run_swiglu = [&](const std::string& wname) {
      const auto* tw = find(wname);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(tw->type), &f)) throw LoomError("no swiglu port for type on " + wname);
      const std::uint32_t mt = static_cast<std::uint32_t>(tw->dims[1] / 16);
      const std::uint32_t kb = static_cast<std::uint32_t>(tw->dims[0] / f.qk);
      const Imported w = ImportTensor(*tw);
      const std::string hal =
          std::string("gemm_swiglu_") + f.name + "_" + std::to_string(mt) + "_" + std::to_string(kb) + ".hal";
      LoomExecutable& exe = load(dir + "/" + hal);
      const Geom gm = GeomOf(hal, B);
      std::vector<hrx_buffer_ref_t> b = {{w.handle, w.offset, w.bytes}};
      if (f.name == std::string("iq3s")) b.push_back({grid_iq3s.handle, 0, hb(grid_iq3s)});
      if (f.name == std::string("iq3xxs")) b.push_back({grid_iq3xxs.handle, 0, hb(grid_iq3xxs)});
      if (f.name == std::string("iq2xxs")) b.push_back({grid_iq2xxs.handle, 0, hb(grid_iq2xxs)});
      if (f.name == std::string("iq2xs")) b.push_back({grid_iq2xs.handle, 0, hb(grid_iq2xs)});
      if (f.name == std::string("iq3xxs") || f.name == std::string("iq2xxs") || f.name == std::string("iq2xs"))
        b.push_back({ksigns_iq2xxs.handle, 0, hb(ksigns_iq2xxs)});
      b.push_back({scratch.handle, 0, hb(scratch)});
      b.push_back({gateffn.handle, 0, hb(gateffn)});
      b.push_back({uwstage.handle, 0, hb(uwstage)});
      b.push_back({ostage.handle, 0, hb(ostage)});
      b.push_back({ffnup.handle, 0, hb(ffnup)});
      Dispatch(gpu, exe, ("yah_ffn_gemm_" + std::string(f.name) + "_swiglu").c_str(), mt / gm.rowgrp, B / gm.tokens, 1,
               32, 1, 1, b);
    };
    auto run_residual = [&](const std::string& wname, const LoomBuffer& input) {
      const auto* tw = find(wname);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(tw->type), &f)) throw LoomError("no residual port for type on " + wname);
      const std::uint32_t mt = static_cast<std::uint32_t>(tw->dims[1] / 16);
      const std::uint32_t kb = static_cast<std::uint32_t>(tw->dims[0] / f.qk);
      const Imported w = ImportTensor(*tw);
      // Fused residual (gen_gemm_shared kind "kres"), if the set has it: the GEMM writes hidden + acc into hidden2.
      // No partial buffer and no yah_residual_1d pass; the handles are swapped after.
      const std::string fused_hal =
          std::string("gemm_kres_") + f.name + "_" + std::to_string(mt) + "_" + std::to_string(kb) + ".hal";
      if (g_geom.count(fused_hal)) {
        LoomExecutable& fx = load(dir + "/" + fused_hal);
        const Geom fg = GeomOf(fused_hal, B);
        std::vector<hrx_buffer_ref_t> fb = {{w.handle, w.offset, w.bytes}};
        if (f.name == std::string("iq3s")) fb.push_back({grid_iq3s.handle, 0, hb(grid_iq3s)});
        if (f.name == std::string("iq3xxs")) fb.push_back({grid_iq3xxs.handle, 0, hb(grid_iq3xxs)});
        if (f.name == std::string("iq2xxs")) fb.push_back({grid_iq2xxs.handle, 0, hb(grid_iq2xxs)});
        if (f.name == std::string("iq2xs")) fb.push_back({grid_iq2xs.handle, 0, hb(grid_iq2xs)});
        if (f.name == std::string("iq3xxs") || f.name == std::string("iq2xxs") || f.name == std::string("iq2xs"))
          fb.push_back({ksigns_iq2xxs.handle, 0, hb(ksigns_iq2xxs)});
        fb.push_back({input.handle, 0, hb(input)});
        fb.push_back({hidden.handle, 0, hb(hidden)});
        fb.push_back({wstage.handle, 0, hb(wstage)});
        fb.push_back({ostage.handle, 0, hb(ostage)});
        fb.push_back({hidden2.handle, 0, hb(hidden2)});
        Dispatch(gpu, fx, (std::string("yah_ffn_gemm_") + f.name + "_kres").c_str(), mt / fg.rowgrp, B / fg.tokens, 1,
                 32, 1, 1, fb);
        std::swap(hidden, hidden2);
        return;
      }
      // Otherwise kStore writes the token-major [B][m_rows] product into partial and yah_residual_1d adds it.
      const std::string hal =
          std::string("gemm_kstore_") + f.name + "_" + std::to_string(mt) + "_" + std::to_string(kb) + ".hal";
      LoomExecutable& exe = load(dir + "/" + hal);
      const Geom gm = GeomOf(hal, B);
      std::vector<hrx_buffer_ref_t> b = {{w.handle, w.offset, w.bytes}};
      if (f.name == std::string("iq3s")) b.push_back({grid_iq3s.handle, 0, hb(grid_iq3s)});
      if (f.name == std::string("iq3xxs")) b.push_back({grid_iq3xxs.handle, 0, hb(grid_iq3xxs)});
      if (f.name == std::string("iq2xxs")) b.push_back({grid_iq2xxs.handle, 0, hb(grid_iq2xxs)});
      if (f.name == std::string("iq2xs")) b.push_back({grid_iq2xs.handle, 0, hb(grid_iq2xs)});
      if (f.name == std::string("iq3xxs") || f.name == std::string("iq2xxs") || f.name == std::string("iq2xs"))
        b.push_back({ksigns_iq2xxs.handle, 0, hb(ksigns_iq2xxs)});
      // The activation is the caller's input: ffnup for ffn_down, scratch for attn_output / ssm_out.
      b.push_back({input.handle, 0, hb(input)});
      b.push_back({wstage.handle, 0, hb(wstage)});
      b.push_back({ostage.handle, 0, hb(ostage)});
      b.push_back({partial.handle, 0, hb(partial)});
      Dispatch(gpu, exe, ("yah_ffn_gemm_" + std::string(f.name)).c_str(), mt / gm.rowgrp, B / gm.tokens, 1, 32, 1, 1,
               b);
      // The sum lands in hidden2: swap the handles instead of copying it back. Every consumer names the variable.
      std::vector<hrx_buffer_ref_t> r = {
          {hidden.handle, 0, hb(hidden)}, {partial.handle, 0, kOutTotal * 4}, {hidden2.handle, 0, hb(hidden2)}};
      Dispatch(gpu, e_accum, "yah_residual_1d", static_cast<std::uint32_t>(kOutTotal / 256), 1, 1, 256, 1, 1, r);
      std::swap(hidden, hidden2);
    };
    gpu.Synchronize();
    const auto t0 = std::chrono::steady_clock::now();
    // YAH_LOGITS_FROM=P: f32 logits of absolute positions P..T-1 into <prefix>.all_logits ((T-P) x vocab, row-major).
    // For the correctness gate (engine/run/accgate2.py); gathered after each chunk's layers.
    std::uint32_t logits_from = T_run;
    if (const char* lf = std::getenv("YAH_LOGITS_FROM")) {
      logits_from = static_cast<std::uint32_t>(std::atoi(lf));
      if (logits_from >= T_run) throw LoomError("YAH_LOGITS_FROM must be below the token count");
    }
    const auto* head_onw = find("output_norm.weight");
    const auto* head_ow = find("output.weight");
    LoomBuffer every = gpu.Allocate(std::size_t{T_run - logits_from + (logits_from == T_run)} * kVocab * 4);
    // YAH_ROWSTATS=<file>: compact per-position statistics for the quality gate (engine/run/kvq/gate2.py).
    // Positions: YAH_ROWSTATS_FROM (default 1024) and every YAH_ROWSTATS_STRIDE-th (default 8) after it,
    // or the list in YAH_ROWSTATS_POS (one position per line).
    // Record per position (little endian): int32 pos, int32 next-token id (-1 at the end), f32 logsumexp,
    // f32 logit of the next token, int32 argmax, f32 top1, f32 top2, then 64 x (int32 id, f32 logit), highest first.
    std::FILE* rowstats = nullptr;
    std::vector<char> rs_want(T_ctx, 0);
    if (const char* rf = std::getenv("YAH_ROWSTATS")) {
      rowstats = std::fopen(rf, "wb");
      if (const char* pf = std::getenv("YAH_ROWSTATS_POS")) {
        std::ifstream pfs(pf);
        std::uint32_t pp;
        while (pfs >> pp)
          if (pp < T_ctx) rs_want[pp] = 1;
      } else {
        const std::uint32_t rs_from =
            std::getenv("YAH_ROWSTATS_FROM") ? std::atoi(std::getenv("YAH_ROWSTATS_FROM")) : 1024;
        const std::uint32_t rs_stride =
            std::getenv("YAH_ROWSTATS_STRIDE") ? std::atoi(std::getenv("YAH_ROWSTATS_STRIDE")) : 8;
        for (std::uint32_t pp = rs_from; pp < T_ctx; pp += rs_stride) rs_want[pp] = 1;
      }
    }
    std::uint32_t rs_max = 0;
    for (std::uint32_t c = 0; c < n_chunks; ++c) {
      std::uint32_t n = 0;
      for (std::uint32_t r = 0; r < B; ++r) n += rs_want[std::size_t{c} * B + r];
      rs_max = std::max(rs_max, n);
    }
    LoomBuffer rsbuf = gpu.Allocate(std::size_t{std::max<std::uint32_t>(rs_max, 1)} * kVocab * 4);
    // YAH_KV_HOOK="<cmd>": a persistent child (sh -c cmd) gets each chunk's new f16 K and V rows, per attention layer.
    // It returns them, e.g. quantized-dequantized by a KV codec under study (engine/run/kvq/kvcodec.py).
    // Protocol per (layer, chunk), little endian:
    //   -> int32 magic 0x4b56484b, attn layer, chunk, first row, rows, cols (1024), q_rows,
    //      then K rows x cols f16, V rows x cols f16, then q_rows x (int32 abs position + 6144 f32 roped Q)
    //   <- K rows x cols f16, V rows x cols f16 (written back into the cache)
    // YAH_KV_HOOK_QSTRIDE=n sends the Q rows whose position is a multiple of n (0 = none).
    // A final header with rows = 0 tells the child to finish.
    FILE* hook_in = nullptr;
    FILE* hook_out = nullptr;
    pid_t hook_pid = -1;
    const std::uint32_t hook_qstride =
        std::getenv("YAH_KV_HOOK_QSTRIDE") ? std::atoi(std::getenv("YAH_KV_HOOK_QSTRIDE")) : 0;
    if (const char* hc = std::getenv("YAH_KV_HOOK")) {
      if (kv16_scratch) throw LoomError("YAH_KV_HOOK needs the f16 KV cache (not a quantized-KV set)");
      int to_child[2], from_child[2];
      if (pipe(to_child) || pipe(from_child)) throw LoomError("YAH_KV_HOOK: pipe failed");
      hook_pid = fork();
      if (hook_pid == 0) {
        dup2(to_child[0], 0);
        dup2(from_child[1], 1);
        close(to_child[1]);
        close(from_child[0]);
        execl("/bin/sh", "sh", "-c", hc, static_cast<char*>(nullptr));
        _exit(127);
      }
      close(to_child[0]);
      close(from_child[1]);
      hook_in = fdopen(to_child[1], "wb");
      hook_out = fdopen(from_child[0], "rb");
    }
    const auto kv_hook = [&](std::uint32_t ai, std::uint32_t ci, std::size_t koff, std::size_t voff) {
      if (!hook_in) return;
      gpu.Synchronize();
      const std::size_t first = std::size_t{ci} * B, cols = kKvRow;
      const std::size_t bytes = std::size_t{B} * cols * 2;
      std::vector<std::uint8_t> kk(bytes), vv(bytes);
      gpu.D2H(kv16, kk.data(), bytes, koff + first * cols * 2);
      gpu.D2H(kv16, vv.data(), bytes, voff + first * cols * 2);
      std::vector<std::uint32_t> qpos;
      if (hook_qstride)
        for (std::uint32_t r = 0; r < B; ++r)
          if ((first + r) % hook_qstride == 0) qpos.push_back(r);
      const std::int32_t hdr[7] = {0x4b56484b,
                                   static_cast<std::int32_t>(ai),
                                   static_cast<std::int32_t>(ci),
                                   static_cast<std::int32_t>(first),
                                   static_cast<std::int32_t>(B),
                                   static_cast<std::int32_t>(cols),
                                   static_cast<std::int32_t>(qpos.size())};
      std::fwrite(hdr, 4, 7, hook_in);
      std::fwrite(kk.data(), 1, bytes, hook_in);
      std::fwrite(vv.data(), 1, bytes, hook_in);
      if (!qpos.empty()) {
        std::vector<float> qrow(kAttn);
        for (std::uint32_t r : qpos) {
          gpu.D2H(q, qrow.data(), std::size_t{kAttn} * 4, std::size_t{r} * kAttn * 4);
          const std::int32_t ap = static_cast<std::int32_t>(first + r);
          std::fwrite(&ap, 4, 1, hook_in);
          std::fwrite(qrow.data(), 4, kAttn, hook_in);
        }
      }
      std::fflush(hook_in);
      if (std::fread(kk.data(), 1, bytes, hook_out) != bytes || std::fread(vv.data(), 1, bytes, hook_out) != bytes)
        throw LoomError("YAH_KV_HOOK: short reply from the hook process");
      gpu.H2D(kv16, kk.data(), bytes, koff + first * cols * 2);
      gpu.H2D(kv16, vv.data(), bytes, voff + first * cols * 2);
    };
    for (std::uint32_t ci = 0; ci < n_chunks; ++ci) {
      if (ci) {
        gpu.Synchronize();  // host_hidden is reused by the next embed
        embed(ci, hidden);
      }
      LoomExecutable& e_rope = *e_ropes[ci];
      LoomExecutable& e_wmma = *e_wmmas[ci];
      for (std::uint32_t l = 0; l < cfg.main_block_count(); ++l) {
        const std::string pre = "blk." + std::to_string(l) + ".";
        const bool full = cfg.IsFullAttention(l);
        run_norm(pre + "attn_norm.weight");
        if (full) {
          const std::uint32_t ai = l / cfg.full_attention_interval;
          const bool qg_fused = run_kqg(pre + "attn_q.weight");
          if (!qg_fused) run_kstore(pre + "attn_q.weight", qkv);
          run_kstore(pre + "attn_k.weight", kbuf);
          run_kstore(pre + "attn_v.weight", vbuf);
          if (!qg_fused) {
            std::vector<hrx_buffer_ref_t> b = {
                {qkv.handle, 0, hb(qkv)}, {q.handle, 0, hb(q)}, {gate.handle, 0, hb(gate)}};
            Dispatch(gpu, e_unpack, "yah_unpack_qg", 24, B, 1, 256, 1, 1, b);
          }
          const auto* qn = find(pre + "attn_q_norm.weight");
          const auto* kn = find(pre + "attn_k_norm.weight");
          const Imported w_qn = ImportTensor(*qn);
          const Imported w_kn = ImportTensor(*kn);
          if (ai >= kFull) throw LoomError("full-attention layer index past the KV slot count");
          const std::size_t koff = kv16_scratch ? 0 : std::size_t{ai} * kKvCache * 2;
          const std::size_t voff = kv16_scratch ? kKv16Layer : std::size_t{kFull} * kKvCache * 2 + koff;
          {
            std::vector<hrx_buffer_ref_t> b = {
                {q.handle, 0, hb(q)},
                {kbuf.handle, 0, hb(kbuf)},
                {vbuf.handle, 0, hb(vbuf)},
                {w_qn.handle, w_qn.offset, w_qn.bytes},
                {w_kn.handle, w_kn.offset, w_kn.bytes},
                {q.handle, 0, hb(q)},
                {kbuf.handle, 0, hb(kbuf)},
                {kc32.handle, 0, hb(kc32)},
                {vc32.handle, 0, hb(vc32)},
                rope_kpaged ? hrx_buffer_ref_t{kpool.handle, std::size_t{ai} * kPoolBytes, kPoolBytes}
                            : hrx_buffer_ref_t{kv16.handle, koff, kKv16Layer},
                {kv16.handle, voff, kKv16Layer},
                {eps.handle, 0, 4}};
            if (rope_kpaged) b.push_back(ptab_ref);
            Dispatch(gpu, e_rope, "yah_fused_qk_rope_batched", 28, B, 1, 256, 1, 1, b);
          }
          kv_hook(ai, ci, koff, voff);
          const std::size_t q8off = std::size_t{ai} * kKqBytes;
          const std::size_t ksoff = std::size_t{ai} * kKsBytes;
          const std::size_t kmoff = std::size_t{ai} * 4096;
          const std::size_t first = std::size_t{ci} * B;            // this chunk's first cache row
          const std::size_t f16rows = std::size_t{B} * kKvRow * 2;  // the chunk's f16 K or V rows
          const std::size_t f16first = kv16_scratch ? 0 : first * kKvRow * 2;
          if (attn_kq8) {
            const std::size_t qrow = kKqBytes / T_ctx, srow = kKsBytes / T_ctx;
            if (ci == 0) {  // channel mean of the first chunk, kept for the later ones
              std::vector<hrx_buffer_ref_t> b = {{kv16.handle, koff, f16rows}, {kmbuf.handle, kmoff, 4096}};
              Dispatch(gpu, *e_kmean, "yah_kmean", 4, 1, 1, 256, 1, 1, b);
            }
            std::vector<hrx_buffer_ref_t> b = {{kv16.handle, koff + f16first, f16rows},
                                               {kmbuf.handle, kmoff, 4096},
                                               {kq8buf.handle, q8off + first * qrow, B * qrow},
                                               {ksbuf.handle, ksoff + first * srow, B * srow}};
            if (kv_paged) {  // whole-layer pools, rows placed through the page table
              b[2] = {kq8buf.handle, q8off, kKqBytes};
              b[3] = {ksbuf.handle, ksoff, kKsBytes};
              b.push_back(ptab_ref);
            }
            Dispatch(gpu, kv_paged ? *e_kq8s[ci] : *e_kq8, attn_kq4 ? "yah_kq4" : "yah_kq8", (B + 1) / 2, 1, 1, 256, 1,
                     1, b);
          }
          const std::size_t vqoff = std::size_t{ai} * kVqBytes, vqsoff = std::size_t{ai} * kVqsBytes;
          if (attn_vqt) {
            std::vector<hrx_buffer_ref_t> b = {{kv16.handle, voff + f16first, f16rows},
                                               {vqbuf.handle, vqoff, kVqBytes},
                                               {vqsbuf.handle, vqsoff, kVqsBytes}};
            if (kv_paged) b.push_back(ptab_ref);
            Dispatch(gpu, *e_vqs[ci], attn_vq8 ? "yah_vq8" : "yah_vq4", 4, (B + 15) / 16, 1, 256, 1, 1, b);
          } else if (paged_f16v) {
            std::vector<hrx_buffer_ref_t> b = {
                {kv16.handle, voff, f16rows}, {vtpool.handle, std::size_t{ai} * kPoolBytes, kPoolBytes}, ptab_ref};
            Dispatch(gpu, *e_vtpages[ci], "yah_vtpage", 32, (B + 31) / 32, 1, 256, 1, 1, b);
          } else if (e_vtrans) {
            std::vector<hrx_buffer_ref_t> b = {{kv16.handle, voff, kKvCache * 2}, {vt16.handle, 0, kVtBytes}};
            Dispatch(gpu, *e_vtrans, "yah_transpose_v16", 32, (T_ctx + 31) / 32, 1, 256, 1, 1, b);
          }
          {
            std::vector<hrx_buffer_ref_t> b = {
                {q.handle, 0, hb(q)},
                {gate.handle, 0, hb(gate)},
                attn_kq8     ? hrx_buffer_ref_t{kq8buf.handle, q8off, kKqBytes}
                : paged_f16k ? hrx_buffer_ref_t{kpool.handle, std::size_t{ai} * kPoolBytes, kPoolBytes}
                             : hrx_buffer_ref_t{kv16.handle, koff, kKvCache * 2},
                attn_vqt     ? hrx_buffer_ref_t{vqbuf.handle, vqoff, kVqBytes}
                : paged_f16v ? hrx_buffer_ref_t{vtpool.handle, std::size_t{ai} * kPoolBytes, kPoolBytes}
                : e_vtrans   ? hrx_buffer_ref_t{vt16.handle, 0, kVtBytes}
                             : hrx_buffer_ref_t{kv16.handle, voff, kKvCache * 2},
                {attn_f16 ? scratch.handle : aout.handle, 0, attn_f16 ? hb(scratch) : hb(aout)},
                {lse.handle, 0, hb(lse)}};
            if (attn_kq8) b.push_back({ksbuf.handle, ksoff, kKsBytes});
            if (attn_vqt) b.push_back({vqsbuf.handle, vqsoff, kVqsBytes});
            if (kv_paged) b.push_back(ptab_ref);
            // dispatch.txt "wmma.hal" row: rowgrp = query heads of one GQA group per workgroup,
            // tokens = query tokens per workgroup, tt = query-token tiles.
            const auto attn_geom = g_geom.find("wmma.hal");
            const std::uint32_t attn_hpw =
                attn_geom != g_geom.end() && attn_geom->second.rowgrp ? attn_geom->second.rowgrp : 1;
            if (kHeads % attn_hpw) throw LoomError("wmma.hal heads per workgroup does not divide the heads");
            const std::uint32_t attn_tpw =
                attn_geom != g_geom.end() && attn_geom->second.tokens ? attn_geom->second.tokens : 16;
            // Loom drops bounds clamps it proves from the launch contract: extra workgroups read unmapped VA and hang.
            // Refuse a grid the emitter did not record.
            if (attn_geom != g_geom.end() && attn_geom->second.tt &&
                (B + attn_tpw - 1) / attn_tpw != attn_geom->second.tt)
              throw LoomError("wmma.hal: grid x does not match the emitted token tiles");
            // "attn_wg384": 12-wave workgroups (GQA packing).
            const std::uint32_t attn_wg = g_geom.count("attn_wg384") ? 384 : 256;
            Dispatch(gpu, e_wmma, "yah_attn_wmma", (B + attn_tpw - 1) / attn_tpw, kHeads / attn_hpw, 1, attn_wg, 1, 1,
                     b);
          }
          if (!attn_f16) {
            std::vector<hrx_buffer_ref_t> b = {{aout.handle, 0, hb(aout)}, {scratch.handle, 0, hb(scratch)}};
            Dispatch(gpu, e_cast, "yah_half_cast", 24 * B, 1, 1, 256, 1, 1, b);
          }
          run_residual(pre + "attn_output.weight", scratch);
        } else {
          const std::uint32_t si = l - l / cfg.full_attention_interval;
          run_kstore(pre + "attn_qkv.weight", qkv);
          run_kstore(pre + "attn_gate.weight", gate);
          run_kstore(pre + "ssm_alpha.weight", alpha);
          run_kstore(pre + "ssm_beta.weight", beta);
          const auto* convw = find(pre + "ssm_conv1d.weight");
          const Imported w_conv = ImportTensor(*convw);
          const auto* ta = find(pre + "ssm_a");
          const auto* tdt = find(pre + "ssm_dt.bias");
          const auto* tsn = find(pre + "ssm_norm.weight");
          const Imported w_a = ImportTensor(*ta);
          const Imported w_dt = ImportTensor(*tdt);
          const Imported w_sn = ImportTensor(*tsn);
          const std::size_t cs_off = std::size_t{si} * kQkv * 4 * 4;
          const std::size_t st_off = std::size_t{si} * kTs * kState * kState * 4;
          {
            std::vector<hrx_buffer_ref_t> b = {{qkv.handle, 0, hb(qkv)},
                                               {w_conv.handle, w_conv.offset, w_conv.bytes},
                                               {conv_state.handle, cs_off, std::size_t{kQkv} * 4 * 4},
                                               {conv_out.handle, 0, hb(conv_out)}};
            if (e_convkq) {
              b.push_back({kqbuf.handle, 0, hb(kqbuf)});
              Dispatch(gpu, *e_convkq, "yah_ssm_conv_kq", 40, B, 1, 256, 1, 1, b);
            } else {
              Dispatch(gpu, e_conv, "yah_ssm_conv", 40, B, 1, 256, 1, 1, b);
            }
          }
          if (e_convstate) {  // the next chunk's conv reads this chunk's last 3 inputs
            std::vector<hrx_buffer_ref_t> b = {{qkv.handle, 0, hb(qkv)},
                                               {conv_state.handle, cs_off, std::size_t{kQkv} * 4 * 4}};
            Dispatch(gpu, *e_convstate, "yah_conv_state", (kQkv + 255) / 256, 1, 1, 256, 1, 1, b);
          }
          {
            std::vector<hrx_buffer_ref_t> b = {{conv_out.handle, 0, hb(conv_out)}, {kqbuf.handle, 0, hb(kqbuf)}};
            if (!e_convkq) Dispatch(gpu, e_prepkq, "yah_deltanet_prep_kq", kKh, B, 1, 32, 1, 1, b);
          }
          {
            std::vector<hrx_buffer_ref_t> b = {{alpha.handle, 0, hb(alpha)},
                                               {beta.handle, 0, hb(beta)},
                                               {w_a.handle, w_a.offset, w_a.bytes},
                                               {w_dt.handle, w_dt.offset, w_dt.bytes},
                                               {qkv.handle, 0, hb(qkv)},
                                               {conv_state.handle, cs_off, std::size_t{kQkv} * 4 * 4},
                                               {ab.handle, 0, hb(ab)}};
            // One lane index does two jobs: the conv-history ring for i < qkv_size and alpha/beta for i < B * heads.
            // So the grid covers max() of the two; a fixed grid silently skips alpha/beta channels at large B.
            const std::uint32_t prebab_tiles = (std::max<std::uint32_t>(B * kTs, kQkv) + 255u) / 256u;
            Dispatch(gpu, e_prepab, "yah_deltanet_prep_ab", prebab_tiles, 1, 1, 256, 1, 1, b);
          }
          {
            std::vector<hrx_buffer_ref_t> b = {{conv_out.handle, 0, hb(conv_out)},
                                               {kqbuf.handle, 0, hb(kqbuf)},
                                               {ab.handle, 0, hb(ab)},
                                               {state.handle, st_off, std::size_t{kTs} * kState * kState * 4},
                                               {raw.handle, 0, hb(raw)}};
            // DeltaNet grid: (blocks per head, heads) x 256, blocks per head from the "rowsplit.hal" row group.
            // Without that row: (heads) x 128.
            const auto dn_geom = g_geom.find("rowsplit.hal");
            if (dn_geom != g_geom.end() && dn_geom->second.rowgrp)
              Dispatch(gpu, e_rowsplit, "yah_deltanet", dn_geom->second.rowgrp, kTs, 1, 256, 1, 1, b);
            else
              Dispatch(gpu, e_rowsplit, "yah_deltanet", kTs, 1, 1, 128, 1, 1, b);
          }
          {
            std::vector<hrx_buffer_ref_t> b = {{raw.handle, 0, hb(raw)},
                                               {w_sn.handle, w_sn.offset, w_sn.bytes},
                                               {gate.handle, 0, static_cast<std::size_t>(B) * kInner * 4},
                                               {scratch.handle, 0, hb(scratch)}};
            Dispatch(gpu, e_postnorm, "yah_ssm_postnorm_fp16", 6 * B, 1, 1, 256, 1, 1, b);
          }
          run_residual(pre + "ssm_out.weight", scratch);
        }
        run_norm(pre + "post_attention_norm.weight");
        run_kstore(pre + "ffn_gate.weight", gateffn);
        run_swiglu(pre + "ffn_up.weight");
        run_residual(pre + "ffn_down.weight", ffnup);
      }
      if (logits_from < T_run && std::size_t{ci + 1} * B > logits_from) {
        const Imported wnorm = ImportTensor(*head_onw);
        const Imported w = ImportTensor(*head_ow);
        const std::uint32_t lo = std::max<std::uint32_t>(logits_from, ci * B);
        for (std::uint32_t ra = lo; ra < (ci + 1) * B; ++ra) {
          const std::uint32_t r = ra - ci * B;
          {
            std::vector<hrx_buffer_ref_t> b = {{hidden.handle, std::size_t{r} * kHidden * 4, std::size_t{kHidden} * 4},
                                               {wnorm.handle, wnorm.offset, wnorm.bytes},
                                               {normed.handle, 0, hb(normed)}};
            Dispatch(gpu, e_rms, "yah_rmsnorm", 1, 1, 1, 32, 1, 1, b);
          }
          std::vector<hrx_buffer_ref_t> b = {
              {w.handle, w.offset, w.bytes},
              {normed.handle, 0, hb(normed)},
              {every.handle, std::size_t{ra - logits_from} * kVocab * 4, std::size_t{kVocab} * 4}};
          Dispatch(gpu, e_gemv, "yah_gemv_q6k", kVocab, 1, 1, 32, 1, 1, b);
        }
      }
      if (rowstats) {
        std::vector<std::uint32_t> rows;
        for (std::uint32_t r = 0; r < B; ++r)
          if (rs_want[std::size_t{ci} * B + r]) rows.push_back(r);
        if (!rows.empty()) {
          const Imported wnorm = ImportTensor(*head_onw);
          const Imported w = ImportTensor(*head_ow);
          for (std::size_t i = 0; i < rows.size(); ++i) {
            {
              std::vector<hrx_buffer_ref_t> b = {
                  {hidden.handle, std::size_t{rows[i]} * kHidden * 4, std::size_t{kHidden} * 4},
                  {wnorm.handle, wnorm.offset, wnorm.bytes},
                  {normed.handle, 0, hb(normed)}};
              Dispatch(gpu, e_rms, "yah_rmsnorm", 1, 1, 1, 32, 1, 1, b);
            }
            std::vector<hrx_buffer_ref_t> b = {{w.handle, w.offset, w.bytes},
                                               {normed.handle, 0, hb(normed)},
                                               {rsbuf.handle, i * kVocab * 4, std::size_t{kVocab} * 4}};
            Dispatch(gpu, e_gemv, "yah_gemv_q6k", kVocab, 1, 1, 32, 1, 1, b);
          }
          gpu.Synchronize();
          std::vector<float> host(rows.size() * kVocab);
          gpu.D2H(rsbuf, host.data(), host.size() * 4, 0);
          std::vector<std::uint32_t> idx(kVocab);
          for (std::size_t i = 0; i < rows.size(); ++i) {
            const float* lg = host.data() + i * kVocab;
            const std::uint32_t pos = ci * B + rows[i];
            const std::int32_t nxt = pos + 1 < T_run ? static_cast<std::int32_t>(ids_all[pos + 1]) : -1;
            double mx = lg[0];
            for (std::uint32_t v = 1; v < kVocab; ++v) mx = std::max<double>(mx, lg[v]);
            double se = 0.0;
            for (std::uint32_t v = 0; v < kVocab; ++v) se += std::exp(static_cast<double>(lg[v]) - mx);
            const float lse = static_cast<float>(mx + std::log(se));
            for (std::uint32_t v = 0; v < kVocab; ++v) idx[v] = v;
            std::partial_sort(idx.begin(), idx.begin() + 64, idx.end(), [&](std::uint32_t a, std::uint32_t b2) {
              return lg[a] > lg[b2] || (lg[a] == lg[b2] && a < b2);
            });
            const std::int32_t ipos = static_cast<std::int32_t>(pos), am = static_cast<std::int32_t>(idx[0]);
            const float lt = nxt >= 0 ? lg[nxt] : 0.0f, t1 = lg[idx[0]], t2 = lg[idx[1]];
            std::fwrite(&ipos, 4, 1, rowstats);
            std::fwrite(&nxt, 4, 1, rowstats);
            std::fwrite(&lse, 4, 1, rowstats);
            std::fwrite(&lt, 4, 1, rowstats);
            std::fwrite(&am, 4, 1, rowstats);
            std::fwrite(&t1, 4, 1, rowstats);
            std::fwrite(&t2, 4, 1, rowstats);
            for (int k = 0; k < 64; ++k) {
              const std::int32_t id = static_cast<std::int32_t>(idx[k]);
              std::fwrite(&id, 4, 1, rowstats);
              std::fwrite(&lg[idx[k]], 4, 1, rowstats);
            }
          }
        }
      }
    }  // chunks
    gpu.Synchronize();
    if (rowstats) std::fclose(rowstats);
    if (hook_in) {
      const std::int32_t fin[7] = {0x4b56484b, -1, -1, 0, 0, 0, 0};
      std::fwrite(fin, 4, 7, hook_in);
      std::fclose(hook_in);
      std::fclose(hook_out);
      int status = 0;
      waitpid(hook_pid, &status, 0);
    }
    const double layer_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("layers_ms=%.1f\n", layer_ms);

    {
      std::vector<float> out(static_cast<std::size_t>(B) * kHidden);
      gpu.D2H(hidden, out.data(), out.size() * 4, 0);
      FILE* fo = std::fopen((prefix + ".hidden").c_str(), "wb");
      std::fwrite(out.data(), 4, out.size(), fo);
      std::fclose(fo);
    }
    std::uint32_t tok = 0;
    {
      const auto* onw = find("output_norm.weight");
      const auto* ow = find("output.weight");
      const Imported wnorm = ImportTensor(*onw);
      {
        std::vector<hrx_buffer_ref_t> b = {{hidden.handle, std::size_t{B - 1} * kHidden * 4, std::size_t{kHidden} * 4},
                                           {wnorm.handle, wnorm.offset, wnorm.bytes},
                                           {normed.handle, 0, hb(normed)}};
        Dispatch(gpu, e_rms, "yah_rmsnorm", 1, 1, 1, 32, 1, 1, b);
      }
      {
        const Imported w = ImportTensor(*ow);
        std::vector<hrx_buffer_ref_t> b = {
            {w.handle, w.offset, w.bytes}, {normed.handle, 0, hb(normed)}, {logits.handle, 0, hb(logits)}};
        Dispatch(gpu, e_gemv, "yah_gemv_q6k", kVocab, 1, 1, 32, 1, 1, b);
      }
      {
        std::vector<hrx_buffer_ref_t> b = {{logits.handle, 0, hb(logits)}, {token.handle, 0, 4}};
        Dispatch(gpu, e_argmax, "yah_argmax", 1, 1, 1, 32, 1, 1, b);
      }
      gpu.Synchronize();
      gpu.D2H(token, &tok, 4, 0);
      std::vector<float> logit_host(kVocab);
      gpu.D2H(logits, logit_host.data(), logit_host.size() * 4, 0);
      FILE* fl = std::fopen((prefix + ".logits").c_str(), "wb");
      std::fwrite(logit_host.data(), 4, logit_host.size(), fl);
      std::fclose(fl);
      std::printf("argmax=%u\n", tok);

      // YAH_GEN=N, YAH_DECODE_HAL=<emit_decode.py set, same context>: N greedy tokens, the prefill's argmax first.
      // The GEMV decoder runs on this prefill's paged KV pools, page table, conv and DeltaNet state.
      if (const char* gv = std::getenv("YAH_GEN"); gv && std::atoi(gv) > 0) {
        const std::uint32_t ngen = static_cast<std::uint32_t>(std::atoi(gv));
        const char* ddir = std::getenv("YAH_DECODE_HAL");
        if (!ddir) throw LoomError("YAH_GEN needs YAH_DECODE_HAL (tools/emit_decode.py set)");
        if (!kv_paged) throw LoomError("YAH_GEN needs the paged KV layout (the default set)");
        if (T_run + ngen - 1 > T_ctx) throw LoomError("YAH_GEN: prompt + gen exceeds the emitted context");
        LoomDecoder dec(gpu, gguf, cfg, ddir, weights.handle, weights_delta);
        if (dec.context() != kPages * 256)
          throw LoomError("YAH_GEN: decode set context " + std::to_string(dec.context()) + " != prefill pool rows " +
                          std::to_string(kPages * 256));
        // The decode set's KV format must be this prefill's (fp16, kv8a16 or kv4a16).
        const std::uint32_t kbits = attn_kq4 ? 4 : attn_kq8 ? 8 : 16, vbits = attn_vq4 ? 4 : attn_vq8 ? 8 : 16;
        if (dec.kv_bits() != std::make_pair(kbits, vbits))
          throw LoomError("YAH_GEN: decode set KV bits " + std::to_string(dec.kv_bits().first) + "/" +
                          std::to_string(dec.kv_bits().second) + " != prefill " + std::to_string(kbits) + "/" +
                          std::to_string(vbits) + " (emit_decode.py with the same YAH_ATTN_FA_* switches)");
        LoomDecoderState st;
        for (std::uint32_t ai = 0; ai < kFull; ++ai) {
          if (dec.quant()) {
            st.kq.push_back({kq8buf.handle, std::size_t{ai} * kKqBytes, kKqBytes});
            st.ks.push_back({ksbuf.handle, std::size_t{ai} * kKsBytes, kKsBytes});
            st.km.push_back({kmbuf.handle, std::size_t{ai} * 4096, 4096});
            st.vq.push_back({vqbuf.handle, std::size_t{ai} * kVqBytes, kVqBytes});
            st.vs.push_back({vqsbuf.handle, std::size_t{ai} * kVqsBytes, kVqsBytes});
          } else {
            st.kpool.push_back({kpool.handle, std::size_t{ai} * kPoolBytes, kPoolBytes});
            st.vtpool.push_back({vtpool.handle, std::size_t{ai} * kPoolBytes, kPoolBytes});
          }
        }
        st.ptab = ptab_ref;
        st.convstate = conv_state.handle;
        st.dstate = state.handle;
        dec.Bind(st);
        dec.SetTokens(&tok, 1, T_run);
        gpu.Synchronize();
        const auto tg = std::chrono::steady_clock::now();
        for (std::uint32_t pos = T_run; pos + 1 < T_run + ngen; ++pos) dec.Step(pos, T_run);
        gpu.Synchronize();
        const double gms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tg).count();
        const std::vector<std::uint32_t> gen = dec.Tokens(T_run, T_run + ngen);
        std::printf("generated_ids=");
        for (std::size_t i = 0; i < gen.size(); ++i) std::printf("%u%s", gen[i], i + 1 == gen.size() ? "" : " ");
        std::printf("\n");
        if (ngen > 1)
          std::printf("decode_ms=%.2f decode_tok_s=%.2f (context %u)\n", gms / (ngen - 1), 1000.0 * (ngen - 1) / gms,
                      T_run);
      }

      // Write the YAH_LOGITS_FROM rows gathered after each chunk.
      if (logits_from < T_run) {
        const std::size_t rows = T_run - logits_from;
        gpu.Synchronize();
        std::vector<float> host(rows * kVocab);
        gpu.D2H(every, host.data(), host.size() * 4, 0);
        FILE* fa = std::fopen((prefix + ".all_logits").c_str(), "wb");
        std::fwrite(host.data(), 4, host.size(), fa);
        std::fclose(fa);
        std::printf("all_logits rows=%zu from=%u\n", rows, logits_from);
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "loom_forward_pp: %s\n", error.what());
    return 1;
  }
}
