// loom_forward_pp: the whole prompt through the 64-layer Loom prefill in ONE
// pass, so the Loom prefill can be compared against the HIP prefill at the same
// token count.
//
// loom_forward_target.cc is the 5-token path: kB is a compile-time constant, the
// HAL set is emitted with batch=5, and emit_prefill.narrow_tokens narrows every
// GEMM to a 16-wide token tile because 11 of 16 lanes would be padding. A real
// prompt wants the token dimension carried by the grid instead, so this driver
// takes the token count B from the command line and every kernel is dispatched
// with its token axes multiplied out:
//
//   GEMM family    grid (m_tiles, token_tiles, 1|k_split), token_tiles = B/TILE
//   norm/conv/...  grid (tiles, B) or (B, 1, 1)
//   attention      grid (heads, B), one workgroup per (head, token), keys
//                  0..token causal, KV cache max_context = B deep
//   residual       the K-split partials are token-major over the WHOLE prompt,
//                  so the reduction dim is 5120*B (not 5120*TILE) and the
//                  per-split stride is m_rows*B
//
// One pass means start_pos is 0 everywhere and the recurrent state (the ssm conv
// ring, the DeltaNet state) starts zeroed, so the recurrence is unchanged: this
// is exactly the first-chunk case the 5-token path already validates. No
// token-tile loop, and therefore no per-tile launch overhead.
//
// The HAL set must come from emit_prefill_pp.py at the SAME B and the same
// YAH_TOKEN_TILE; the printed banner states both, because a mismatch is a wrong
// grid and that is not a caught error.
//
// usage: loom_forward_pp <model.gguf> <haldir> <out-prefix> [tokens] [ids-file]
//   env YAH_LOOM_TIME=1|2   category / per-dispatch timing
//   env YAH_SKIP_HEAD=1     skip the output projection and argmax
//
// outputs: <out-prefix>.logits (vocab f32, last token) and <out-prefix>.hidden
//          (B x 5120 f32, token major)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "model/loom_runtime.hpp"

using namespace yah::model;  // NOLINT(google-build-using-namespace)

namespace {
// GEMM tokens per workgroup: the tile the HAL set was emitted with. 64 is the
// shipping source's own tile; 128/256 come from tools/widen_tokens.py.
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
// K-split factor for the residual projections. Runtime, so the emitter's
// YAH_KSPLIT and the driver's YAH_KSPLIT can be swept: the 4-way split is the
// only geometry where the residual arm was ever measured to be nondeterministic.
std::uint32_t g_ksplit = 4;
// One KV slot per prompt token per layer.
constexpr std::uint32_t kKvRow = kKvHeads * kHeadDim;  // 1024

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

std::uint32_t g_b = 0;   // prompt tokens in this run
int g_dump_layer = -1;   // YAH_DUMP_LAYER: checkpoint this layer's stages
int g_layer_now = -1;    // the layer the driver is on
std::uint32_t g_tt = 0;  // GEMM token tiles = g_b / g_gtile
// The GEMM launch geometry the HAL set was emitted with, read from
// <haldir>/dispatch.json. The emitter writes it, so the grid this driver passes
// cannot disagree with the geometry the kernel was compiled for. Mirroring that
// arithmetic in C here is what produced a wrong-but-deterministic forward on
// 2026-09-29: the HAL declared twice the workgroups, the driver launched the old
// count, and half the rows were never computed. docs/reference/hrx-agents.md:
// "Launch geometry is not public stage configuration ... stage authoring code
// must not mirror the arithmetic in C."
std::uint32_t g_gtile = kTile;  // fallback GEMM tile when dispatch.txt says nothing
int g_time = 0;
std::string g_key;  // YAH_LOOM_TIME=3: timing key for the next dispatch
bool g_fused_residual = true;  // YAH_FUSED_RESIDUAL=0 disables the gemm_kres path
std::map<std::string, double> g_per_name;
std::map<std::string, int> g_per_count;
std::chrono::steady_clock::time_point g_mark = std::chrono::steady_clock::now();
// YAH_TRACE_LOAD: wall time since process start at each setup/teardown phase.
const std::chrono::steady_clock::time_point g_start = std::chrono::steady_clock::now();
void Phase(const char* what) {
  static const bool on = std::getenv("YAH_TRACE_LOAD") != nullptr;
  if (on)
    std::fprintf(stderr, "[phase] %8.1f ms  %s\n",
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g_start).count(), what);
}
// YAH_LOOM_SEQ=<path>: every dispatch's key (HAL or kernel name) and grid in
// submission order, written at exit. No synchronization, so it pairs with an
// HRX_PROFILE_FILE=... HRX_PROFILE_MODE=dispatch capture of the same run: the
// profile has device durations per dispatch but only anonymous executable ids.
std::FILE* g_seq = nullptr;
std::size_t g_seq_n = 0;

void Dispatch(LoomDevice& gpu, const LoomExecutable& exe, const char* name,
              std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
              std::uint32_t sx, std::uint32_t sy, std::uint32_t sz,
              const std::vector<hrx_buffer_ref_t>& b) {
  static const bool dt = std::getenv("YAH_LOOM_DISPATCH_TIMING") != nullptr;
  static double in_us = 0, between_us = 0, meta_us = 0; static long n = 0;
  static std::chrono::steady_clock::time_point last_return;
  const auto t_in = std::chrono::steady_clock::now();
  static std::map<std::string, std::pair<double, long>> gap_by_next;
  if (dt && n) {
    const double g = std::chrono::duration<double, std::micro>(t_in - last_return).count();
    between_us += g;
    auto& e = gap_by_next[name]; e.first += g; ++e.second;
  }
  // The executable's own workgroup size is authoritative; sx is only the
  // fallback for metadata that does not carry one.
  const std::uint32_t ordinal = exe.OrdinalOrZero(name);
  const std::uint32_t ws = exe.WorkgroupSize(ordinal);
  if (dt) meta_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_in).count();
  struct Report { ~Report() {
    if (!(dt && n)) return;
    std::fprintf(stderr, "Dispatch(): %ld calls, inside %.1f ms (metadata %.1f ms), between calls %.1f ms\n", n, in_us / 1000, meta_us / 1000, between_us / 1000);
    std::vector<std::pair<double, std::string>> v;
    for (auto& kv : gap_by_next) v.push_back({kv.second.first, kv.first + " (" + std::to_string(kv.second.second) + ")"});
    std::sort(v.rbegin(), v.rend());
    for (std::size_t i = 0; i < v.size() && i < 8; ++i) std::fprintf(stderr, "  host gap before %-40s %8.1f ms\n", v[i].second.c_str(), v[i].first / 1000);
  } };
  static Report report;
  if (g_seq)
    std::fprintf(g_seq, "%zu,%s,%u,%u,%u\n", g_seq_n++,
                 g_key.empty() ? name : g_key.c_str(), gx, gy, gz);
  if (g_time >= 2) gpu.Synchronize();
  g_mark = std::chrono::steady_clock::now();
  gpu.Dispatch(exe, ordinal, LoomDevice::Config(gx, gy, gz, ws ? ws : sx, sy, sz),
               nullptr, 0, b.data(), b.size());
  if (g_time >= 2) {
    gpu.Synchronize();
    // YAH_LOOM_TIME=3 keys the GEMMs by their HAL (format and shape) instead of
    // the kernel name, so per-shape efficiency is visible.
    const std::string key = (g_time >= 3 && !g_key.empty()) ? g_key : std::string(name);
    g_per_name[key] += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - g_mark).count();
    g_per_count[key]++;
  }
  g_key.clear();
  if (dt) {
    last_return = std::chrono::steady_clock::now();
    in_us += std::chrono::duration<double, std::micro>(last_return - t_in).count();
    ++n;
  }
}

// The launch geometry each GEMM HAL was compiled for, read from
// <haldir>/dispatch.txt: one line per HAL, "<basename> <tokens> <rowgrp>
// <token_tiles>". The emitter writes it beside the HAL, so the grid this driver
// passes cannot disagree with the geometry the kernel was built for. Mirroring
// that arithmetic in C here is what produced a wrong-but-deterministic forward on
// 2026-09-29: the HAL declared twice the workgroups, the driver launched the old
// count, and half the rows were never computed. docs/reference/hrx-agents.md:
// "Launch geometry is not public stage configuration ... stage authoring code
// must not mirror the arithmetic in C."
struct Geom {
  std::uint32_t tokens;   // tokens per workgroup = grid y divisor
  std::uint32_t rowgrp;   // 16-row tiles per workgroup = grid x divisor
  std::uint32_t tt;       // token_tiles the emitter bound into this HAL
};
std::map<std::string, Geom> g_geom;

void LoadDispatch(const std::string& dir) {
  std::ifstream f(dir + "/dispatch.txt");
  if (!f) return;
  std::string name;
  Geom g{};
  while (f >> name >> g.tokens >> g.rowgrp >> g.tt) g_geom[name] = g;
}

// Geometry for one HAL, with the check that makes a mismatch loud: a HAL emitted
// for a different token count would otherwise compute a silent subset.
Geom GeomOf(const std::string& hal, std::uint32_t B) {
  const auto it = g_geom.find(hal);
  if (it == g_geom.end())
    return Geom{kTile, 1, static_cast<std::uint32_t>(B / kTile)};
  const Geom g = it->second;
  if (g.tokens == 0 || g.rowgrp == 0 || B % g.tokens != 0 || B / g.tokens != g.tt) {
    std::fprintf(stderr,
                 "%s: emitted for tile=%u rowgrp=%u token_tiles=%u, but B=%u gives %u tiles\n",
                 hal.c_str(), g.tokens, g.rowgrp, g.tt, B, B / g.tokens);
    std::exit(3);
  }
  return g;
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
  { const char* t = std::getenv("YAH_LOOM_TIME");
    g_time = t ? std::min(std::atoi(t), 3) : 0; }
  if (const char* q = std::getenv("YAH_LOOM_SEQ")) {
    g_seq = std::fopen(q, "w");
    if (g_seq) std::fprintf(g_seq, "seq,key,gx,gy,gz\n");
  }
  // YAH_DUMP_LAYER=<l> writes three checkpoints from layer l: the mixer input
  // (the f16 normed activation), hidden after the mixer, and hidden after the
  // FFN. It exists to locate a stage that disagrees between two token counts.
  { const char* t = std::getenv("YAH_DUMP_LAYER"); if (t) g_dump_layer = std::atoi(t); }
  { const char* t = std::getenv("YAH_KSPLIT");
    if (t) { g_ksplit = std::atoi(t); }
    if (g_ksplit < 1 || g_ksplit > 8) throw LoomError("YAH_KSPLIT must be 1..8"); }
  const bool skip_head = std::getenv("YAH_SKIP_HEAD") != nullptr;
  { const char* t = std::getenv("YAH_FUSED_RESIDUAL"); if (t && std::string(t) == "0") g_fused_residual = false; }
  // YAH_KSTORE_RESIDUAL=1 runs the gemm_residual projections (ffn_down,
  // attn_output, ssm_out) on the CHAINED kStore HAL instead of the residual
  // source. The residual source still uses the old one-column-per-lane decode
  // that widen_rows cannot address, so it runs at the shipping geometry; the
  // kStore variant is the chained one. The accumulate is unchanged -- the kStore
  // writes the same token-major [B][m_rows] layout into 'partial' split 0, and
  // the reduction below adds that into the residual exactly as before.
  const char* ks_env = std::getenv("YAH_KSTORE_RESIDUAL");
  const std::string ks_residual = ks_env ? std::string(ks_env) : std::string();
  try {
    auto gguf = yah::core::Gguf::Open(model);
    const auto cfg = yah::core::Qwen35Config::FromGguf(gguf);
    Phase("gguf open");
    std::vector<std::uint32_t> ids_all = ParseIds(ids_path);
    g_b = want ? want : static_cast<std::uint32_t>(ids_all.size());
    if (ids_all.size() < g_b) {
      // Pad a short id file by repeating it: the numbers only have to be a valid
      // token stream of the right length for a timing run, and a caller that
      // wants exact ids passes a long enough file.
      std::vector<std::uint32_t> grown;
      while (grown.size() < g_b) grown.insert(grown.end(), ids_all.begin(), ids_all.end());
      ids_all.swap(grown);
    }
    LoadDispatch(dir);
    if (g_b % g_gtile) {
      std::fprintf(stderr, "tokens=%u must be a multiple of the token tile=%u\n", g_b, g_gtile);
      return 2;
    }
    g_tt = g_b / g_gtile;
    const std::uint32_t B = g_b;
    const std::size_t kOutTotal = static_cast<std::size_t>(kHidden) * B;
    // Only the full-attention layers own a KV slot, and the cache holds two runs
    // of kFull slots -- k then v -- so slot i of layer ai is reached by indexing
    // the run. The 5-token driver hardcoded 8 here and its kv16 had 32 slots by
    // accident; at 16 full-attention layers a slot count of 8 is out of range.
    std::uint32_t kFull = 0;
    for (std::uint32_t l = 0; l < cfg.main_block_count(); ++l)
      if (cfg.IsFullAttention(l)) ++kFull;
    Phase("dispatch table");
    LoomDevice gpu;
    // The layers queue ~950 dispatches and wait once at the end; the runtime
    // wait would busy-poll a core for the whole prefill (LOOM_RUNTIME.md).
    // YAH_LOOM_TIME waits after every stage, so it keeps the runtime wait.
    if (!g_time) gpu.SetSleepSync(200);
    Phase("device");
    std::fprintf(stderr,
                 "loom_forward_pp: tokens=%u tile=%u token_tiles=%u layers=%u geometry=%zu hal(s)\n",
                 B, g_gtile, g_tt, cfg.main_block_count(), g_geom.size());
    // Import the whole GGUF tensor-data region once: every tensor is an offset
    // into it, instead of one hrx_allocator_import_buffer per dispatch.
    LoomBuffer weights;
    std::size_t weights_delta = 0;
    {
      const std::uint8_t* wbase = gguf.tensor_data_base();
      const std::uintptr_t page = 4096;
      const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(wbase) & ~(page - 1);
      weights_delta = reinterpret_cast<std::uintptr_t>(wbase) - start;
      const std::size_t wbytes = gguf.tensor_data_size() + weights_delta;
      if (std::getenv("YAH_LOOM_WEIGHTS_DEVICE")) {
        // A device-local copy instead of the imported mmap (HIP's layout).
        weights = gpu.Allocate(wbytes);
        const std::size_t chunk = std::size_t{256} << 20;
        for (std::size_t off = 0; off < wbytes; off += chunk)
          gpu.H2D(weights, reinterpret_cast<const void*>(start + off),
                  std::min(chunk, wbytes - off), off);
      } else {
        weights = gpu.Import(reinterpret_cast<void*>(start), wbytes);
      }
    }
    Phase("weights import");
    auto ImportTensor = [&](const yah::core::TensorInfo& t) -> Imported {
      return {weights.handle, weights_delta + static_cast<std::size_t>(t.offset),
              static_cast<std::size_t>(t.bytes)};
    };
    std::map<std::string, LoomExecutable> exes;
    auto load = [&](const std::string& path) -> LoomExecutable& {
      auto it = exes.find(path);
      if (it == exes.end()) {
        if (std::getenv("YAH_TRACE_LOAD")) std::fprintf(stderr, "[load] %s\n", path.c_str());
        it = exes.emplace(path, gpu.Load(path)).first;
      }
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

    // Every activation buffer is token major: [token][row], with the row stride
    // equal to the K extent of the GEMM that reads it (k_blocks*256). The KV
    // cache is one slot per prompt token; the per-layer state buffers keep the
    // 5-token layout because they are indexed by the layer, not by the prompt.
    const std::size_t kKvCache = static_cast<std::size_t>(B) * kKvRow;
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
    LoomBuffer kv16 = gpu.Allocate(std::size_t{2} * kFull * kKvCache * 2);
    LoomBuffer kc32 = gpu.Allocate(kKvCache * 4);
    LoomBuffer vc32 = gpu.Allocate(kKvCache * 4);
    LoomBuffer lse = gpu.Allocate(static_cast<std::size_t>(B) * kHeads * 4);
    LoomBuffer eps = gpu.Allocate(4);
    LoomBuffer ffnup = gpu.Allocate(static_cast<std::size_t>(B) * kFfn * 2);
    LoomBuffer gateffn = gpu.Allocate(static_cast<std::size_t>(B) * kFfn * 4);
    // Dense per-workgroup staging: the kernels stage a 16x16 weight tile per K
    // step, not the full 16xK row.
    LoomBuffer gwstage = gpu.Allocate(std::size_t{kFfn} * 16 * 2);
    LoomBuffer uwstage = gpu.Allocate(std::size_t{kFfn} * 16 * 2);
    LoomBuffer wstage = gpu.Allocate(std::size_t{kFfn} * 16 * 2);
    // ostage is the epilogue scratch. The compiler declares it over the whole
    // [m_rows][tokens] tile, so it must cover the widest GEMM (17408 rows) and
    // the 4-way residual split (4 * 5120 rows) at this token count.
    const std::size_t kStageRows = std::max<std::size_t>(kFfn, std::size_t{g_ksplit} * kHidden);
    LoomBuffer ostage = gpu.Allocate(kStageRows * B * 4);
    LoomBuffer partial = gpu.Allocate(kOutTotal * g_ksplit * 4);
    LoomBuffer hidden2 = gpu.Allocate(kOutTotal * 4);
    LoomBuffer normed = gpu.Allocate(std::size_t{kHidden} * 4);
    LoomBuffer logits = gpu.Allocate(std::size_t{kVocab} * 4);
    LoomBuffer token = gpu.Allocate(4);

    const auto hb = [](const LoomBuffer& b) { return b.size; };
    // Diagnostics probe: hold every per-layer weight buffer alive so the
    // allocator cannot hand the same address back while a queued dispatch still
    // references it. If the forward becomes deterministic with this, the
    // nondeterminism is HRX VA reuse, not a kernel race.
    std::vector<LoomBuffer> keep;
    const auto dump_buf = [&](const LoomBuffer& src, std::size_t bytes, const std::string& tag) {
      // Synchronize first. hrx_synchronous_d2h is synchronous for the copy, not
      // for the stream, so a dump taken while dispatches are still queued reads a
      // buffer that is mid-write -- the first bisect with these hooks reported
      // stages as "differing" in an order that contradicted the data flow.
      gpu.Synchronize();
      std::vector<std::uint8_t> v(bytes);
      gpu.D2H(src, v.data(), bytes, 0);
      FILE* f = std::fopen((prefix + tag).c_str(), "wb");
      std::fwrite(v.data(), 1, bytes, f);
      std::fclose(f);
    };
    // dump_buf for a byte range (e.g. one layer's slot of the KV cache)
    const auto dump_range = [&](const LoomBuffer& src, std::size_t offset, std::size_t bytes,
                                const std::string& tag) {
      gpu.Synchronize();
      std::vector<std::uint8_t> v(bytes);
      gpu.D2H(src, v.data(), bytes, offset);
      FILE* f = std::fopen((prefix + tag).c_str(), "wb");
      std::fwrite(v.data(), 1, bytes, f);
      std::fclose(f);
    };
    const float epsv = 1.0e-6f; gpu.H2D(eps, &epsv, 4);
    { std::vector<std::uint8_t> z(std::size_t{48} * kQkv * 4 * 4, 0); gpu.H2D(conv_state, z.data(), z.size()); }
    { std::vector<std::uint8_t> z(std::size_t{48} * kTs * kState * kState * 4, 0); gpu.H2D(state, z.data(), z.size()); }
    { std::vector<std::uint8_t> z(std::size_t{2} * kFull * kKvCache * 2, 0); gpu.H2D(kv16, z.data(), z.size()); }
    { std::vector<std::uint8_t> z(static_cast<std::size_t>(B) * kHidden * 4, 0); gpu.H2D(reszero, z.data(), z.size()); }
    // YAH_ZERO=1 pre-zeroes every scratch buffer. It is a diagnostic: if the
    // forward stops varying between runs with it set, a kernel is reading a
    // region the driver never wrote (HRX device memory is not zeroed, so its
    // contents differ per process).
    if (std::getenv("YAH_ZERO") != nullptr) {
      const LoomBuffer* all[] = {&scratch, &qkv, &gate, &aout, &raw, &conv_out,
                                 &kqbuf, &ab, &ffnup, &gateffn, &partial,
                                 &hidden2, &sumout, &lse, &alpha, &beta, &q,
                                 &kbuf, &vbuf, &ostage, &gwstage, &uwstage,
                                 &wstage};
      std::vector<std::uint8_t> z(1u << 20, 0);
      for (const LoomBuffer* b : all) {
        for (std::size_t off = 0; off < b->size; off += z.size())
          gpu.H2D(*b, z.data(), std::min<std::size_t>(z.size(), b->size - off), off);
      }
    }

    Phase("buffers");
    const auto* emb = find("token_embd.weight");
    std::vector<float> host_hidden(static_cast<std::size_t>(B) * kHidden);
    const std::uint8_t* emb_data = gguf.Data(*emb);
    for (std::uint32_t t = 0; t < B; ++t) {
      if (static_cast<std::uint32_t>(emb->type) == 23) DequantIq4XsRow(emb_data, ids_all[t], host_hidden.data() + std::size_t{t} * kHidden);
      else DequantQ4KRow(emb_data, ids_all[t], host_hidden.data() + std::size_t{t} * kHidden);
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
    // tools/gen_attn_hip.py: a vtrans.hal row means the attention reads V as
    // [kv head][16-key tile][dim][16] f16, written per layer by yah_transpose_v16.
    LoomExecutable* e_vtrans = g_geom.count("vtrans.hal") ? &load(dir + "/vtrans.hal") : nullptr;
    const std::size_t kVtBytes = std::size_t{(B + 15) / 16 * 16} * kKvRow * 2;
    LoomBuffer vt16 = gpu.Allocate(e_vtrans ? kVtBytes : 4);
    LoomExecutable& e_cast = load(dir + "/cast.hal");
    LoomExecutable& e_gemv = load(dir + "/gemv.hal");
    LoomExecutable& e_rms = load(dir + "/rmsnorm.hal");
    LoomExecutable& e_argmax = load(dir + "/argmax.hal");
    LoomExecutable& e_accum = load(dir + "/accum.hal");

    // The norm writes the f16 activation into scratch at row stride dim, so
    // every GEMM reading scratch has ktot == dim: 5120 for the FFN, 6144 for the
    // attention output. That is why scratch is kFfn wide.
    // The small per-layer weights (norms, conv1d, ssm_a/dt/norm) are bound as
    // views into the single imported tensor region instead of being copied into
    // freshly allocated device buffers.
    //
    // The old per-call allocation was deliberate, and the hazard it avoided is
    // real: a SHARED buffer refilled here while the previous layer's dispatch is
    // still queued corrupts the weight that dispatch bound, because
    // hrx_synchronous_h2d is synchronous for the copy but not for the stream
    // (sharing measured a wrong forward, argmax 14). Binding a view avoids that
    // hazard entirely rather than reintroducing it: nothing is ever rewritten,
    // because every tensor already owns a distinct immutable region of the one
    // import -- exactly how run_kstore/run_swiglu and output.weight bind theirs.
    //
    // What the copy cost: ~8 allocations plus H2D transfers per layer. Each
    // allocation runs the HSA map path, which waits by POLLING
    // AMDKFD_IOC_WAIT_EVENTS (~1.1M polls per forward, 99.6% of all syscall
    // time), and mapping memory while dispatches are in flight drains the queue
    // every layer -- the ramp-then-decay in GPU utilisation.
    auto run_norm = [&](const std::string& wname) {
      const auto* tw = find(wname);
      const Imported w = ImportTensor(*tw);
      std::vector<hrx_buffer_ref_t> b = {
          {hidden.handle, 0, hb(hidden)}, {reszero.handle, 0, hb(reszero)},
          {w.handle, w.offset, w.bytes}, {sumout.handle, 0, hb(sumout)},
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
      const std::string hal = std::string("gemm_kstore_") + f.name + "_" +
                              std::to_string(mt) + "_" + std::to_string(kb) + ".hal";
      LoomExecutable& exe = load(dir + "/" + hal);
      if (std::getenv("YAH_TRACE_GEMM")) std::fprintf(stderr, "[hal] %s -> %s%c", wname.c_str(), hal.c_str(), 10);
      const Geom gm = GeomOf(hal, B);
      g_key = hal;
      std::vector<hrx_buffer_ref_t> b = {{w.handle, w.offset, w.bytes}};
      if (f.name == std::string("iq3s")) b.push_back({grid_iq3s.handle, 0, hb(grid_iq3s)});
      if (f.name == std::string("iq3xxs")) b.push_back({grid_iq3xxs.handle, 0, hb(grid_iq3xxs)});
      if (f.name == std::string("iq2xxs")) b.push_back({grid_iq2xxs.handle, 0, hb(grid_iq2xxs)});
      if (f.name == std::string("iq2xs")) b.push_back({grid_iq2xs.handle, 0, hb(grid_iq2xs)});
      if (f.name == std::string("iq3xxs") || f.name == std::string("iq2xxs") || f.name == std::string("iq2xs")) b.push_back({ksigns_iq2xxs.handle, 0, hb(ksigns_iq2xxs)});
      b.push_back({scratch.handle, 0, hb(scratch)});
      b.push_back({wstage.handle, 0, hb(wstage)});
      b.push_back({ostage.handle, 0, hb(ostage)});
      b.push_back({out.handle, 0, hb(out)});
      Dispatch(gpu, exe, ("yah_ffn_gemm_" + std::string(f.name)).c_str(),
               mt / gm.rowgrp, B / gm.tokens, 1, 32, 1, 1, b);
    };
    // The attention q projection with the q/gate unpack fused into its epilogue
    // (gemm_kqg_*: rows = heads x [256 q | 256 gate] stored straight into q and
    // gate, as yah_unpack_qg did). False if the set has no such HAL or
    // YAH_KQG=0; the caller then runs kstore + unpack.
    auto run_kqg = [&](const std::string& wname) -> bool {
      { const char* e = std::getenv("YAH_KQG"); if (e && std::string(e) == "0") return false; }
      const auto* tw = find(wname);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(tw->type), &f)) return false;
      const std::uint32_t mt = static_cast<std::uint32_t>(tw->dims[1] / 16);
      const std::uint32_t kb = static_cast<std::uint32_t>(tw->dims[0] / f.qk);
      const std::string hal = std::string("gemm_kqg_") + f.name + "_" +
                              std::to_string(mt) + "_" + std::to_string(kb) + ".hal";
      if (g_geom.find(hal) == g_geom.end()) return false;
      const Imported w = ImportTensor(*tw);
      LoomExecutable& exe = load(dir + "/" + hal);
      if (std::getenv("YAH_TRACE_GEMM")) std::fprintf(stderr, "[hal] %s -> %s%c", wname.c_str(), hal.c_str(), 10);
      const Geom gm = GeomOf(hal, B);
      g_key = hal;
      std::vector<hrx_buffer_ref_t> b = {{w.handle, w.offset, w.bytes}};
      if (f.name == std::string("iq3s")) b.push_back({grid_iq3s.handle, 0, hb(grid_iq3s)});
      if (f.name == std::string("iq3xxs")) b.push_back({grid_iq3xxs.handle, 0, hb(grid_iq3xxs)});
      if (f.name == std::string("iq2xxs")) b.push_back({grid_iq2xxs.handle, 0, hb(grid_iq2xxs)});
      if (f.name == std::string("iq2xs")) b.push_back({grid_iq2xs.handle, 0, hb(grid_iq2xs)});
      if (f.name == std::string("iq3xxs") || f.name == std::string("iq2xxs") || f.name == std::string("iq2xs")) b.push_back({ksigns_iq2xxs.handle, 0, hb(ksigns_iq2xxs)});
      b.push_back({scratch.handle, 0, hb(scratch)});
      b.push_back({wstage.handle, 0, hb(wstage)});
      b.push_back({ostage.handle, 0, hb(ostage)});
      b.push_back({q.handle, 0, hb(q)});
      b.push_back({gate.handle, 0, hb(gate)});
      Dispatch(gpu, exe, ("yah_ffn_gemm_" + std::string(f.name) + "_kqg").c_str(),
               mt / gm.rowgrp, B / gm.tokens, 1, 32, 1, 1, b);
      return true;
    };
    auto run_swiglu = [&](const std::string& wname) {
      const auto* tw = find(wname);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(tw->type), &f)) throw LoomError("no swiglu port for type on " + wname);
      const std::uint32_t mt = static_cast<std::uint32_t>(tw->dims[1] / 16);
      const std::uint32_t kb = static_cast<std::uint32_t>(tw->dims[0] / f.qk);
      const Imported w = ImportTensor(*tw);
      const std::string hal = std::string("gemm_swiglu_") + f.name + "_" +
                              std::to_string(mt) + "_" + std::to_string(kb) + ".hal";
      LoomExecutable& exe = load(dir + "/" + hal);
      if (std::getenv("YAH_TRACE_GEMM")) std::fprintf(stderr, "[hal] %s -> %s%c", wname.c_str(), hal.c_str(), 10);
      const Geom gm = GeomOf(hal, B);
      g_key = hal;
      std::vector<hrx_buffer_ref_t> b = {{w.handle, w.offset, w.bytes}};
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
               mt / gm.rowgrp, B / gm.tokens, 1, 32, 1, 1, b);
    };
    auto run_residual = [&](const std::string& wname, const LoomBuffer& input) {
      // Narrow the kStore swap to ffn_down until its output layout is proven
      // against the residual's partial. Routing all three residual projections at
      // once made one bad path corrupt the whole residual stream.
      // Default ON: route every residual projection through the chained kStore.
      // YAH_KSTORE_RESIDUAL=off restores the residual source; any other value
      // selects only the projections whose weight name contains it.
      const bool kres = ks_residual.empty()
                            ? true
                            : (ks_residual != "off" && wname.find(ks_residual) != std::string::npos);
      const auto* tw = find(wname);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(tw->type), &f)) throw LoomError("no residual port for type on " + wname);
      const std::uint32_t mt = static_cast<std::uint32_t>(tw->dims[1] / 16);
      const std::uint32_t kb = static_cast<std::uint32_t>(tw->dims[0] / f.qk);
      const Imported w = ImportTensor(*tw);
      // Fused residual (gen_gemm_shared kind "kres"): the GEMM reads hidden and
      // writes hidden + acc into hidden2 itself, so neither the partial buffer nor
      // the yah_residual_1d pass is needed; the handles are swapped after.
      // Taken whenever the HAL set carries one; YAH_FUSED_RESIDUAL=0 disables it.
      const std::string fused_hal = std::string("gemm_kres_") + f.name + "_" +
                                    std::to_string(mt) + "_" + std::to_string(kb) + ".hal";
      if (kres && g_fused_residual && g_geom.count(fused_hal)) {
        LoomExecutable& fx = load(dir + "/" + fused_hal);
        const Geom fg = GeomOf(fused_hal, B);
        g_key = fused_hal;
        std::vector<hrx_buffer_ref_t> fb = {{w.handle, w.offset, w.bytes}};
        if (f.name == std::string("iq3s")) fb.push_back({grid_iq3s.handle, 0, hb(grid_iq3s)});
        if (f.name == std::string("iq3xxs")) fb.push_back({grid_iq3xxs.handle, 0, hb(grid_iq3xxs)});
        if (f.name == std::string("iq2xxs")) fb.push_back({grid_iq2xxs.handle, 0, hb(grid_iq2xxs)});
        if (f.name == std::string("iq2xs")) fb.push_back({grid_iq2xs.handle, 0, hb(grid_iq2xs)});
        if (f.name == std::string("iq3xxs") || f.name == std::string("iq2xxs") || f.name == std::string("iq2xs")) fb.push_back({ksigns_iq2xxs.handle, 0, hb(ksigns_iq2xxs)});
        fb.push_back({input.handle, 0, hb(input)});
        fb.push_back({hidden.handle, 0, hb(hidden)});
        fb.push_back({wstage.handle, 0, hb(wstage)});
        fb.push_back({ostage.handle, 0, hb(ostage)});
        fb.push_back({hidden2.handle, 0, hb(hidden2)});
        Dispatch(gpu, fx, (std::string("yah_ffn_gemm_") + f.name + "_kres").c_str(),
                 mt / fg.rowgrp, B / fg.tokens, 1, 32, 1, 1, fb);
        std::swap(hidden, hidden2);
        return;
      }
      const std::string hal = std::string(kres ? "gemm_kstore_" : "gemm_residual_") +
                              f.name + "_" + std::to_string(mt) + "_" + std::to_string(kb) + ".hal";

      LoomExecutable& exe = load(dir + "/" + hal);
      if (std::getenv("YAH_TRACE_GEMM")) std::fprintf(stderr, "[hal] %s -> %s%c", wname.c_str(), hal.c_str(), 10);
      const Geom gm = GeomOf(hal, B);
      g_key = hal;
      std::vector<hrx_buffer_ref_t> b = {{w.handle, w.offset, w.bytes}};
      if (f.name == std::string("iq3s")) b.push_back({grid_iq3s.handle, 0, hb(grid_iq3s)});
      if (f.name == std::string("iq3xxs")) b.push_back({grid_iq3xxs.handle, 0, hb(grid_iq3xxs)});
      if (f.name == std::string("iq2xxs")) b.push_back({grid_iq2xxs.handle, 0, hb(grid_iq2xxs)});
      if (f.name == std::string("iq2xs")) b.push_back({grid_iq2xs.handle, 0, hb(grid_iq2xs)});
      if (f.name == std::string("iq3xxs") || f.name == std::string("iq2xxs") || f.name == std::string("iq2xs")) b.push_back({ksigns_iq2xxs.handle, 0, hb(ksigns_iq2xxs)});
      // This 3rd binding is the ACTIVATION for both sources. run_kstore hardcodes
      // 'scratch' only because its callers' activation always lives there; for the
      // residual projections the activation is the caller's input (ffnup for
      // ffn_down, scratch for attn_output/ssm_out). Binding scratch unconditionally
      // fed the kStore the wrong activation and produced uncorrelated output.
      b.push_back({input.handle, 0, hb(input)});
      b.push_back({wstage.handle, 0, hb(wstage)});
      b.push_back({ostage.handle, 0, hb(ostage)});
      b.push_back({partial.handle, 0, hb(partial)});
      Dispatch(gpu, exe,
               (std::string("yah_ffn_gemm_") + std::string(f.name) +
                (kres ? "" : "_residual")).c_str(),
               mt / gm.rowgrp, B / gm.tokens, kres ? 1u : g_ksplit, 32, 1, 1, b);
      if (g_dump_layer == g_layer_now && &input == &ffnup) {
        dump_buf(partial, kOutTotal * g_ksplit * 4, ".partial");
        std::fprintf(stderr, "[dump] partial for layer %d\n", g_layer_now);
      }
      // Reduce the k_split partials into the residual: hidden += sum_s partial[s].
      // The partial is token major over the whole prompt, so split s starts at
      // s * (m_rows * B) floats -- that is why the dimension is 5120*B and not
      // 5120*TILE. Ping-pong through hidden2 so the reduction's a and out
      // bindings never alias; the last step lands in hidden (with a final
      // zero-add pass when the split count is odd).
      const std::size_t out_bytes = kOutTotal * 4;
      const LoomBuffer* land = &hidden;
      const std::uint32_t splits = kres ? 1u : g_ksplit;
      for (std::uint32_t s = 0; s < splits; ++s) {
        const LoomBuffer& src = (s % 2 == 0) ? hidden : hidden2;
        const LoomBuffer& dst = (s % 2 == 0) ? hidden2 : hidden;
        std::vector<hrx_buffer_ref_t> r = {
            {src.handle, 0, hb(src)},
            {partial.handle, std::size_t{s} * out_bytes, out_bytes},
            {dst.handle, 0, hb(dst)}};
        Dispatch(gpu, e_accum, "yah_residual_1d", static_cast<std::uint32_t>(kOutTotal / 256), 1, 1, 256, 1, 1, r);
        land = &dst;
      }
      if (land != &hidden) {
        // An odd split count (always, on the kStore-residual path) leaves the
        // running sum in hidden2. Swap the two handles instead of copying it
        // back through a zero-add pass: that pass was half of every residual's
        // dispatches (128 of 256 per forward) and moved 126 MB each. Every
        // consumer names the variable, so it follows the swap.
        std::swap(hidden, hidden2);
      }
    };
    double t_norm = 0.0, t_mixer = 0.0, t_attn = 0.0, t_ssm = 0.0;
    double t_ffn = 0.0, t_ffn_norm = 0.0, t_ffn_gate = 0.0, t_ffn_up = 0.0, t_ffn_down = 0.0;
    std::chrono::steady_clock::time_point mark = std::chrono::steady_clock::now();
    const auto tick = [&]() {
      if (!g_time) return 0.0;
      gpu.Synchronize();
      const double d = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - mark).count();
      mark = std::chrono::steady_clock::now();
      return d;
    };
    // YAH_LOOM_PRELOAD=1: load every GEMM HAL of the set before the timed
    // region instead of lazily at first use (diagnostic for the host-side
    // polling investigation; measured neutral, so off by default to keep
    // layers_ms comparable with earlier runs).
    if (std::getenv("YAH_LOOM_PRELOAD")) {
      for (const auto& kv : g_geom) {
        const std::string path = dir + "/" + kv.first;
        if (FILE* f = std::fopen(path.c_str(), "rb")) {
          std::fclose(f);
          load(path);
        }
      }
    }
    gpu.Synchronize();
    Phase("embed + first HALs");
    if (std::getenv("YAH_TRACE_LOAD")) std::fprintf(stderr, "[load] ---- timed region starts ----\n");
    const auto t0 = std::chrono::steady_clock::now();
    // YAH_LAYERS=N stops after N layers. It exists to bisect a stage that
    // disagrees between runs, which is how the residual-geometry race in the
    // k-split GEMM was localised.
    const char* lv = std::getenv("YAH_LAYERS");
    const std::uint32_t n_layers = lv ? std::min<std::uint32_t>(std::atoi(lv), cfg.main_block_count()) : cfg.main_block_count();
    for (std::uint32_t l = 0; l < n_layers; ++l) {
      const std::string pre = "blk." + std::to_string(l) + ".";
      g_layer_now = static_cast<int>(l);
      const bool full = cfg.IsFullAttention(l);
      run_norm(pre + "attn_norm.weight");
      if (g_dump_layer == static_cast<int>(l))
        dump_buf(scratch, static_cast<std::size_t>(B) * kHidden * 2, ".xn");
      t_norm += tick();
      if (full) {
        const std::uint32_t ai = l / cfg.full_attention_interval;
        const bool qg_fused = run_kqg(pre + "attn_q.weight");
        if (!qg_fused) run_kstore(pre + "attn_q.weight", qkv);
        run_kstore(pre + "attn_k.weight", kbuf);
        run_kstore(pre + "attn_v.weight", vbuf);
        if (!qg_fused) {
          std::vector<hrx_buffer_ref_t> b = {
              {qkv.handle, 0, hb(qkv)},
              {q.handle, 0, hb(q)}, {gate.handle, 0, hb(gate)}};
          Dispatch(gpu, e_unpack, "yah_unpack_qg", 24, B, 1, 256, 1, 1, b);
        }
        const auto* qn = find(pre + "attn_q_norm.weight");
        const auto* kn = find(pre + "attn_k_norm.weight");
        const Imported w_qn = ImportTensor(*qn);
        const Imported w_kn = ImportTensor(*kn);
        if (ai >= kFull) throw LoomError("full-attention layer index past the KV slot count");
        const std::size_t koff = std::size_t{ai} * kKvCache * 2;
        const std::size_t voff = std::size_t{kFull} * kKvCache * 2 + koff;
        {
          std::vector<hrx_buffer_ref_t> b = {
              {q.handle, 0, hb(q)}, {kbuf.handle, 0, hb(kbuf)},
              {vbuf.handle, 0, hb(vbuf)}, {w_qn.handle, w_qn.offset, w_qn.bytes},
              {w_kn.handle, w_kn.offset, w_kn.bytes}, {q.handle, 0, hb(q)},
              {kbuf.handle, 0, hb(kbuf)}, {kc32.handle, 0, hb(kc32)},
              {vc32.handle, 0, hb(vc32)}, {kv16.handle, koff, kKvCache * 2},
              {kv16.handle, voff, kKvCache * 2},
              {eps.handle, 0, 4}};
          Dispatch(gpu, e_rope, "yah_fused_qk_rope_batched", 28, B, 1, 256, 1, 1, b);
        }
        // w_qn/w_kn are views into the import; nothing to keep alive.
        if (g_dump_layer == static_cast<int>(l)) {
          // the attention kernel's inputs: roped q, gate, this layer's f16 K and V
          dump_buf(q, static_cast<std::size_t>(B) * 6144 * 4, ".aq");
          dump_buf(gate, static_cast<std::size_t>(B) * 6144 * 4, ".agate");
          dump_range(kv16, koff, kKvCache * 2, ".ak16");
          dump_range(kv16, voff, kKvCache * 2, ".av16");
        }
        if (e_vtrans) {
          std::vector<hrx_buffer_ref_t> b = {
              {kv16.handle, voff, kKvCache * 2}, {vt16.handle, 0, kVtBytes}};
          Dispatch(gpu, *e_vtrans, "yah_transpose_v16", 32, (B + 31) / 32, 1, 256, 1, 1, b);
        }
        {
          std::vector<hrx_buffer_ref_t> b = {
              {q.handle, 0, hb(q)}, {gate.handle, 0, hb(gate)},
              {kv16.handle, koff, kKvCache * 2},
              e_vtrans ? hrx_buffer_ref_t{vt16.handle, 0, kVtBytes}
                       : hrx_buffer_ref_t{kv16.handle, voff, kKvCache * 2},
              {aout.handle, 0, hb(aout)}, {lse.handle, 0, hb(lse)}};
          // YAH_ATTN_GRID_OLD restores the pre-WMMA attention launch geometry so the
          // two attention kernels can be A/Bd from ONE binary, interleaved, without a
          // rebuild between runs (a failed rebuild leaves a stale binary and a mismatched
          // grid silently produces garbage).
          const bool attn_old_grid = std::getenv("YAH_ATTN_GRID_OLD") != nullptr;
          // tools/gen_attn_heads.py runs H query heads of one GQA group per
          // workgroup; dispatch.txt records H as the "wmma.hal" row group.
          const auto attn_geom = g_geom.find("wmma.hal");
          const std::uint32_t attn_hpw =
              attn_geom != g_geom.end() && attn_geom->second.rowgrp ? attn_geom->second.rowgrp : 1;
          if (kHeads % attn_hpw) throw LoomError("wmma.hal heads per workgroup does not divide the heads");
          // query tokens per workgroup: 16 (gen_attn_heads) or 32 (gen_attn_hip)
          const std::uint32_t attn_tpw =
              attn_geom != g_geom.end() && attn_geom->second.tokens ? attn_geom->second.tokens : 16;
          // The kernel's launch contract fixes its grid, and Loom drops bounds
          // clamps it proves from it: extra workgroups read unmapped VA and hang
          // the ring. Refuse a grid the emitter did not record.
          if (!attn_old_grid && attn_geom != g_geom.end() && attn_geom->second.tt &&
              (B + attn_tpw - 1) / attn_tpw != attn_geom->second.tt)
            throw LoomError("wmma.hal: grid x does not match the emitted token tiles");
          Dispatch(gpu, e_wmma, "yah_attn_wmma",
                   attn_old_grid ? kHeads : (B + attn_tpw - 1) / attn_tpw,
                   attn_old_grid ? B : kHeads / attn_hpw, 1,
                   attn_old_grid ? 32 : 256, 1, 1, b);
        }
        if (g_dump_layer == static_cast<int>(l))
          dump_buf(aout, static_cast<std::size_t>(B) * 6144 * 4, ".aout");
        {
          std::vector<hrx_buffer_ref_t> b = {
              {aout.handle, 0, hb(aout)}, {scratch.handle, 0, hb(scratch)}};
          Dispatch(gpu, e_cast, "yah_half_cast", 24 * B, 1, 1, 256, 1, 1, b);
        }
        run_residual(pre + "attn_output.weight", scratch);
        t_attn += tick();
      } else {
        const std::uint32_t si = l - l / cfg.full_attention_interval;
        run_kstore(pre + "attn_qkv.weight", qkv);
        if (g_dump_layer == static_cast<int>(l))
          dump_buf(qkv, static_cast<std::size_t>(B) * kQkv * 4, ".qkv");
        run_kstore(pre + "attn_gate.weight", gate);
        run_kstore(pre + "ssm_alpha.weight", alpha);
        run_kstore(pre + "ssm_beta.weight", beta);
        if (g_dump_layer == static_cast<int>(l)) {
          dump_buf(alpha, static_cast<std::size_t>(B) * kTs * 4, ".alpha");
          dump_buf(beta, static_cast<std::size_t>(B) * kTs * 4, ".beta");
        }
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
          std::vector<hrx_buffer_ref_t> b = {
              {qkv.handle, 0, hb(qkv)},
              {w_conv.handle, w_conv.offset, w_conv.bytes},
              {conv_state.handle, cs_off, std::size_t{kQkv} * 4 * 4},
              {conv_out.handle, 0, hb(conv_out)}};
          Dispatch(gpu, e_conv, "yah_ssm_conv", 40, B, 1, 256, 1, 1, b);
        }
        if (g_dump_layer == static_cast<int>(l))
          dump_buf(conv_out, static_cast<std::size_t>(B) * kQkv * 4, ".conv");
        // w_conv is a view into the import; nothing to keep alive.
        {
          std::vector<hrx_buffer_ref_t> b = {
              {conv_out.handle, 0, hb(conv_out)}, {kqbuf.handle, 0, hb(kqbuf)}};
          Dispatch(gpu, e_prepkq, "yah_deltanet_prep_kq", kKh, B, 1, 32, 1, 1, b);
        }
        if (g_dump_layer == static_cast<int>(l))
          dump_buf(kqbuf, static_cast<std::size_t>(B) * kKh * 3 * 4, ".kq");
        {
          std::vector<hrx_buffer_ref_t> b = {
              {alpha.handle, 0, hb(alpha)}, {beta.handle, 0, hb(beta)},
              {w_a.handle, w_a.offset, w_a.bytes}, {w_dt.handle, w_dt.offset, w_dt.bytes},
              {qkv.handle, 0, hb(qkv)},
              {conv_state.handle, cs_off, std::size_t{kQkv} * 4 * 4},
              {ab.handle, 0, hb(ab)}};
          // The kernel's own grid is ceil((batch*num_heads + qkv_size)/256); it
          // does two jobs over the same lane index -- the conv-history ring for
          // i < qkv_size and alpha/beta for i < count -- so max() of the two is
          // enough. The 5-token driver hardcoded 41, which silently covers only
          // 10496 lanes: correct at 5..218 tokens, and at 2048 it would compute
          // alpha/beta for 10496 of 98304 channels.
          const std::uint32_t prebab_tiles = (std::max<std::uint32_t>(B * kTs, kQkv) + 255u) / 256u;
          Dispatch(gpu, e_prepab, "yah_deltanet_prep_ab", prebab_tiles, 1, 1, 256, 1, 1, b);
        }
        if (g_dump_layer == static_cast<int>(l))
          dump_buf(ab, static_cast<std::size_t>(B) * kTs * 2 * 4, ".ab");
        // w_a/w_dt are views into the import; nothing to keep alive.
        {
          std::vector<hrx_buffer_ref_t> b = {
              {conv_out.handle, 0, hb(conv_out)}, {kqbuf.handle, 0, hb(kqbuf)},
              {ab.handle, 0, hb(ab)},
              {state.handle, st_off, std::size_t{kTs} * kState * kState * 4},
              {raw.handle, 0, hb(raw)}};
          // tools/gen_deltanet_hip.py (HIP's row-split order) runs (2, heads)
          // workgroups of 256; dispatch.txt says so with a "rowsplit.hal" row whose
          // row-group field is the blocks per head. Without it: the regtile
          // kernel's (heads) x 128.
          const auto dn_geom = g_geom.find("rowsplit.hal");
          if (dn_geom != g_geom.end() && dn_geom->second.rowgrp)
            Dispatch(gpu, e_rowsplit, "yah_deltanet", dn_geom->second.rowgrp, kTs, 1, 256, 1, 1, b);
          else
            Dispatch(gpu, e_rowsplit, "yah_deltanet", kTs, 1, 1, 128, 1, 1, b);
        }
        if (g_dump_layer == static_cast<int>(l))
          dump_buf(raw, static_cast<std::size_t>(B) * kInner * 4, ".raw");
        {
          std::vector<hrx_buffer_ref_t> b = {
              {raw.handle, 0, hb(raw)}, {w_sn.handle, w_sn.offset, w_sn.bytes},
              {gate.handle, 0, static_cast<std::size_t>(B) * kInner * 4},
              {scratch.handle, 0, hb(scratch)}};
          Dispatch(gpu, e_postnorm, "yah_ssm_postnorm_fp16", 6 * B, 1, 1, 256, 1, 1, b);
        }
        if (g_dump_layer == static_cast<int>(l))
          dump_buf(scratch, static_cast<std::size_t>(B) * kInner * 2, ".ssm");
        // w_sn is a view into the import; nothing to keep alive.
        run_residual(pre + "ssm_out.weight", scratch);
        t_ssm += tick();
      }
      t_mixer += tick();
      if (g_dump_layer == static_cast<int>(l))
        dump_buf(hidden, static_cast<std::size_t>(B) * kHidden * 4, ".mixer");
      run_norm(pre + "post_attention_norm.weight");
      if (g_dump_layer == static_cast<int>(l))
        dump_buf(scratch, static_cast<std::size_t>(B) * kHidden * 2, ".fn");
      t_ffn_norm += tick();
      run_kstore(pre + "ffn_gate.weight", gateffn);
      if (g_dump_layer == static_cast<int>(l))
        dump_buf(gateffn, static_cast<std::size_t>(B) * kFfn * 4, ".gate");
      t_ffn_gate += tick();
      run_swiglu(pre + "ffn_up.weight");
      if (g_dump_layer == static_cast<int>(l))
        dump_buf(ffnup, static_cast<std::size_t>(B) * kFfn * 2, ".up");
      t_ffn_up += tick();
      run_residual(pre + "ffn_down.weight", ffnup);
      t_ffn_down += tick();
      if (g_dump_layer == static_cast<int>(l))
        dump_buf(hidden, static_cast<std::size_t>(B) * kHidden * 4, ".ffn");
    }
    gpu.Synchronize();
    const double layer_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("layers_ms=%.1f\n", layer_ms);
    Phase("layers");

    {
      std::vector<float> out(static_cast<std::size_t>(B) * kHidden);
      gpu.D2H(hidden, out.data(), out.size() * 4, 0);
      FILE* fo = std::fopen((prefix + ".hidden").c_str(), "wb");
      std::fwrite(out.data(), 4, out.size(), fo);
      std::fclose(fo);
    }
    std::uint32_t tok = 0;
    if (!skip_head) {
      const auto* onw = find("output_norm.weight");
      const auto* ow = find("output.weight");
      const Imported wnorm = ImportTensor(*onw);
      {
        std::vector<hrx_buffer_ref_t> b = {
            {hidden.handle, std::size_t{B - 1} * kHidden * 4, std::size_t{kHidden} * 4},
            {wnorm.handle, wnorm.offset, wnorm.bytes}, {normed.handle, 0, hb(normed)}};
        Dispatch(gpu, e_rms, "yah_rmsnorm", 1, 1, 1, 32, 1, 1, b);
      }
      {
        const Imported w = ImportTensor(*ow);
        std::vector<hrx_buffer_ref_t> b = {
            {w.handle, w.offset, w.bytes}, {normed.handle, 0, hb(normed)},
            {logits.handle, 0, hb(logits)}};
        Dispatch(gpu, e_gemv, "yah_gemv_q6k", kVocab, 1, 1, 32, 1, 1, b);
      }
      {
        std::vector<hrx_buffer_ref_t> b = {
            {logits.handle, 0, hb(logits)}, {token.handle, 0, 4}};
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

      // YAH_LOGITS_FROM=P: f32 logits of every position P..B-1 into
      // <prefix>.all_logits ((B-P) x vocab, row-major) for the tiered
      // correctness gate (engine/run/accgate2.py). The same head kernels as the
      // last-token path, one row at a time into one device buffer.
      if (const char* lf = std::getenv("YAH_LOGITS_FROM")) {
        const std::uint32_t from = static_cast<std::uint32_t>(std::atoi(lf));
        if (from >= B) throw LoomError("YAH_LOGITS_FROM must be below the token count");
        const std::size_t rows = B - from;
        LoomBuffer every = gpu.Allocate(rows * kVocab * 4);
        const Imported w = ImportTensor(*ow);
        for (std::uint32_t r = from; r < B; ++r) {
          {
            std::vector<hrx_buffer_ref_t> b = {
                {hidden.handle, std::size_t{r} * kHidden * 4, std::size_t{kHidden} * 4},
                {wnorm.handle, wnorm.offset, wnorm.bytes}, {normed.handle, 0, hb(normed)}};
            Dispatch(gpu, e_rms, "yah_rmsnorm", 1, 1, 1, 32, 1, 1, b);
          }
          std::vector<hrx_buffer_ref_t> b = {
              {w.handle, w.offset, w.bytes}, {normed.handle, 0, hb(normed)},
              {every.handle, std::size_t{r - from} * kVocab * 4, std::size_t{kVocab} * 4}};
          Dispatch(gpu, e_gemv, "yah_gemv_q6k", kVocab, 1, 1, 32, 1, 1, b);
        }
        gpu.Synchronize();
        std::vector<float> host(rows * kVocab);
        gpu.D2H(every, host.data(), host.size() * 4, 0);
        FILE* fa = std::fopen((prefix + ".all_logits").c_str(), "wb");
        std::fwrite(host.data(), 4, host.size(), fa);
        std::fclose(fa);
        std::printf("all_logits rows=%zu from=%u\n", rows, from);
      }
    }
    if (g_time) {
      const double ffn = t_ffn_norm + t_ffn_gate + t_ffn_up + t_ffn_down;
      std::fprintf(stderr,
                   "== loom pp timing: norm=%.1f attn=%.1f ssm=%.1f mixer=%.1f "
                   "ffn=%.1f (ffn_norm=%.1f gate=%.1f up=%.1f down=%.1f) sum=%.1f ms ==\n",
                   t_norm, t_attn, t_ssm, t_mixer, ffn, t_ffn_norm, t_ffn_gate,
                   t_ffn_up, t_ffn_down, t_norm + t_mixer + ffn);
    }
    if (g_time >= 2) {
      std::vector<std::pair<double, std::string>> rows;
      double sum = 0.0;
      for (auto& kv : g_per_name) {
        rows.push_back({kv.second, kv.first});
        sum += kv.second;
      }
      std::sort(rows.rbegin(), rows.rend());
      std::fprintf(stderr, "== loom per-dispatch timing sum=%.1f ms ==\n", sum);
      for (auto& r : rows)
        std::fprintf(stderr, "%9.1f  %5d  %7.3f  %s\n", r.first, g_per_count[r.second], r.first / g_per_count[r.second], r.second.c_str());
    }
    if (g_seq) std::fclose(g_seq);
    Phase("head + outputs");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "loom_forward_pp: %s\n", error.what());
    return 1;
  }
}
