// LoomPrefill: the 64-layer Loom prefill on HRX, one chunk of B tokens at a time.
//
// HAL set: tools/emit_prefill_pp.py; <dir>/dispatch.txt gives the launch geometry of each HAL and marker rows for the
// set's features. Every kernel carries the token dimension in its grid:
//   GEMM family    grid (m_tiles / rowgrp, token_tiles, 1)
//   norm/conv/...  grid (tiles, B) or (B, 1, 1)
//   attention      grid (query-token tiles, head groups), causal over keys 0..token
// A chunked set ("ctx" row) runs every kernel at the chunk size B and sizes the KV pools for the context T.
// Reset() zeroes the recurrent state (conv ring, DeltaNet state); later chunks carry it forward.
// DecoderState() hands the KV pools, page table and recurrent state to a LoomDecoder.
#ifndef YAH_MODEL_LOOM_PREFILL_HPP_
#define YAH_MODEL_LOOM_PREFILL_HPP_

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "model/loom_decoder.hpp"
#include "model/loom_runtime.hpp"

namespace yah::model {

class LoomPrefill {
 public:
  static constexpr std::uint32_t kHidden = 5120, kFfn = 17408, kAttn = 6144, kQProj = 12288, kKv = 1024;
  static constexpr std::uint32_t kInner = 6144, kQkv = 10240, kTs = 48, kKh = 16, kState = 128;
  static constexpr std::uint32_t kHeads = 24, kKvHeads = 4, kHeadDim = 256, kVocab = 248320;
  static constexpr std::uint32_t kKvRow = kKvHeads * kHeadDim;  // one KV cache row: all KV heads of one token

  // Called after RoPE of attention layer ai in chunk ci, with the layer's f16 K and V offsets in the KV scratch.
  using KvHook = std::function<void(std::uint32_t ai, std::uint32_t ci, std::size_t koff, std::size_t voff)>;

  // tokens: the token count of a one-pass (unchunked) set; ignored for a chunked set, which fixes B.
  LoomPrefill(LoomDevice& gpu, const core::Gguf& gguf, const core::Qwen35Config& cfg, std::string dir,
              hrx_buffer_t weights, std::size_t delta, std::uint32_t tokens = 0)
      : gpu_(gpu), gguf_(gguf), cfg_(cfg), dir_(std::move(dir)), weights_(weights), delta_(delta) {
    LoadDispatch();
    B_ = tokens;
    T_ = tokens;
    if (const auto it = geom_.find("ctx"); it != geom_.end()) {
      B_ = it->second.tokens;
      T_ = it->second.tt;
    }
    if (B_ == 0) throw LoomError("prefill: a one-pass set needs the token count");
    for (std::uint32_t l = 0; l < cfg_.main_block_count(); ++l)
      if (cfg_.IsFullAttention(l)) ++full_;
    LoadTables();
    AllocateBuffers();
    LoadExecutables();
  }

  [[nodiscard]] std::uint32_t chunk() const { return B_; }
  [[nodiscard]] std::uint32_t context() const { return T_; }
  [[nodiscard]] bool paged() const { return kv_paged_; }
  // KV bits of this set: (16, 16) fp16, else 8 or 4 per side.
  [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> kv_bits() const {
    return {attn_kq4_ ? 4u : attn_kq8_ ? 8u : 16u, attn_vq4_ ? 4u : attn_vq8_ ? 8u : 16u};
  }
  LoomBuffer& hidden() { return *hidden_; }

  // Zero the recurrent state before a new sequence. KV rows need no reset: attention reads only keys <= the query.
  void Reset() {
    Zero(*conv_state_);
    Zero(*state_);
  }

  // Host-dequantize the token_embd rows of ids[0..n) into hidden() for the next RunLayers. n < B pads the chunk with
  // the last token. The GEMMs (~92% of the time) run only the token tiles that hold real tokens; every other kernel runs
  // the whole chunk, so padding rows next to the real ones keep their padded-path values. Shrinking the norms as well
  // changed the last token's logits by ~1e-4 when it sat alone in its 16-token group (cause not found yet). Padding tokens leave the recurrent state
  // unchanged and attention is causal, so the real tokens' results and the state carried forward are exact.
  // Waits for queued work that reads hidden().
  void Embed(const std::uint32_t* ids, std::uint32_t n) {
    if (n == 0 || n > B_) throw LoomError("prefill: chunk token count out of range");
    gpu_.Synchronize();
    const auto* emb = Find("token_embd.weight");
    const std::uint8_t* data = gguf_.Data(*emb);
    for (std::uint32_t t = 0; t < B_; ++t) {
      const std::uint32_t id = ids[std::min(t, n - 1)];
      if (id >= kVocab) throw LoomError("token id " + std::to_string(id) + " is outside the vocabulary");
      if (static_cast<std::uint32_t>(emb->type) == 23)
        DequantIq4XsRow(data, id, host_hidden_.data() + std::size_t{t} * kHidden);
      else
        DequantQ4KRow(data, id, host_hidden_.data() + std::size_t{t} * kHidden);
    }
    gpu_.H2D(*hidden_, host_hidden_.data(), host_hidden_.size() * 4);
    const std::int32_t valid = static_cast<std::int32_t>(n);
    gpu_.H2D(*valid_, &valid, 4);
    n_ = n;
  }

  // Enqueue the 64 layers for chunk ci (absolute positions ci * B ..), on the hidden() rows from Embed().
  void RunLayers(std::uint32_t ci, const KvHook& hook = {}) {
    if (std::size_t{ci + 1} * B_ > T_) throw LoomError("prefill: chunk past the emitted context");
    if (n_ == 0) throw LoomError("prefill: RunLayers before Embed");
    // The layers go into one graph, so kernels with no data between them (the input projections of a layer, the DeltaNet
    // gate projection and the conv / DeltaNet chain) can run at the same time.
    LoomGraph graph(gpu_);
    graph.ReadOnly(weights_);
    for (const LoomBuffer* t : {grid_iq3s_, grid_iq3xxs_, grid_iq2xxs_, grid_iq2xs_, ksigns_, eps_, reszero_})
      graph.ReadOnly(t->handle);
    graph_ = &graph;
    for (std::uint32_t l = 0; l < cfg_.main_block_count(); ++l) {
      const std::string pre = "blk." + std::to_string(l) + ".";
      RunNorm(pre + "attn_norm.weight");
      if (cfg_.IsFullAttention(l))
        RunAttention(l, ci, pre, hook);
      else
        RunDeltaNet(l, pre);
      RunNorm(pre + "post_attention_norm.weight");
      RunKstore(pre + "ffn_gate.weight", *gateffn_);
      RunSwiglu(pre + "ffn_up.weight");
      RunResidual(pre + "ffn_down.weight", *ffnup_);
    }
    graph_ = nullptr;
    graph.Launch();
  }

  // Final norm + output head of hidden() row `row` (this chunk) into dst (kVocab f32).
  void Head(std::uint32_t row, const hrx_buffer_ref_t& dst) {
    const auto* onw = Find("output_norm.weight");
    const auto* ow = Find("output.weight");
    Dispatch(Exe("rmsnorm.hal"), "yah_rmsnorm", 1, 1, 1, 32, 1, 1,
             {{hidden_->handle, std::size_t{row} * kHidden * 4, std::size_t{kHidden} * 4}, TRef(*onw), Ref(*normed_)});
    Dispatch(Exe("gemv.hal"), "yah_gemv_q6k", kVocab, 1, 1, 32, 1, 1, {TRef(*ow), Ref(*normed_), dst});
  }
  // Argmax of kVocab f32 logits into dst (one u32).
  void Argmax(const hrx_buffer_ref_t& logits, const hrx_buffer_ref_t& dst) {
    Dispatch(Exe("argmax.hal"), "yah_argmax", 1, 1, 1, 32, 1, 1, {logits, dst});
  }

  // Quantized V, from a KvHook of the last chunk: copy rows r0..r0+cnt of the layer's f16 V (voff in the KV scratch) into a
  // decoder's open tile dst. params: device i32 (r0, cnt).
  void SeedOpenTile(std::size_t voff, const hrx_buffer_ref_t& params, const hrx_buffer_ref_t& dst) {
    Dispatch(Exe("vseed.hal"), "yah_vseed", 4, 1, 1, 256, 1, 1,
             {{kv16_->handle, voff, std::size_t{B_} * kKvRow * 2}, params, dst});
  }

  // The KV pools, page table and recurrent state, for a LoomDecoder with the same context and KV format.
  [[nodiscard]] LoomDecoderState DecoderState() const {
    if (!kv_paged_) throw LoomError("prefill: decode needs the paged KV layout (the default set)");
    LoomDecoderState st;
    for (std::uint32_t ai = 0; ai < full_; ++ai) {
      if (attn_kq8_) {
        st.kq.push_back({kq8buf_->handle, std::size_t{ai} * kq_bytes_, kq_bytes_});
        st.ks.push_back({ksbuf_->handle, std::size_t{ai} * ks_bytes_, ks_bytes_});
        st.km.push_back({kmbuf_->handle, std::size_t{ai} * 4096, 4096});
      }
      if (attn_vqt_) {
        st.vq.push_back({vqbuf_->handle, std::size_t{ai} * vq_bytes_, vq_bytes_});
        st.vs.push_back({vqsbuf_->handle, std::size_t{ai} * vqs_bytes_, vqs_bytes_});
      }
      if (!attn_kq8_) st.kpool.push_back({kpool_->handle, std::size_t{ai} * pool_bytes_, pool_bytes_});
      if (!attn_vqt_) st.vtpool.push_back({vtpool_->handle, std::size_t{ai} * pool_bytes_, pool_bytes_});
    }
    st.ptab = ptab_ref_;
    st.convstate = conv_state_->handle;
    st.dstate = state_->handle;
    return st;
  }
  [[nodiscard]] std::uint32_t pool_rows() const { return pages_ * 256; }

 private:
  struct Fmt {
    const char* name;
    std::uint32_t qk;
  };
  static bool FmtOf(std::uint32_t type, Fmt* out) {
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
  // dispatch.txt: "<hal> <tokens per workgroup> <16-row tiles per workgroup> <token tiles>" per HAL.
  // Read grids from here; never mirror the emitter's arithmetic: a wrong grid silently skips rows.
  struct Geom {
    std::uint32_t tokens;
    std::uint32_t rowgrp;
    std::uint32_t tt;
  };

  void LoadDispatch() {
    std::ifstream f(dir_ + "/dispatch.txt");
    if (!f) throw LoomError("cannot open " + dir_ + "/dispatch.txt");
    std::string name;
    Geom g{};
    while (f >> name >> g.tokens >> g.rowgrp >> g.tt) geom_[name] = g;
  }
  // Refuses a HAL emitted for another token count: it would compute a silent subset.
  Geom GeomOf(const std::string& hal) const {
    const auto it = geom_.find(hal);
    if (it == geom_.end()) throw LoomError("dispatch.txt has no row for " + hal);
    const Geom g = it->second;
    // A token tile need not divide the chunk: the last one is masked in the kernel.
    if (g.tokens == 0 || g.rowgrp == 0 || (B_ + g.tokens - 1) / g.tokens != g.tt)
      throw LoomError(hal + ": emitted for another token count");
    return g;
  }

  static void ReadFile(const std::string& path, void* dst, std::size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw LoomError("cannot open " + path);
    const bool ok = std::fread(dst, 1, bytes, f) == bytes;
    std::fclose(f);
    if (!ok) throw LoomError("short read: " + path);
  }
  static float Half(const std::uint8_t* p) {
    _Float16 h;
    std::memcpy(&h, p, 2);
    return static_cast<float>(h);
  }
  static void DequantQ4KRow(const std::uint8_t* base, std::uint64_t row, float* out) {
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
        out[b * 256 + i] = d * static_cast<float>(s) * static_cast<float>(quant) - dmin * static_cast<float>(m);
      }
    }
  }
  static void DequantIq4XsRow(const std::uint8_t* base, std::uint64_t row, float* out) {
    static const float kValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
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
        const float dl = d * static_cast<float>(static_cast<int>(sc) - 32);
        for (std::uint32_t w = 0; w < 32; ++w) {
          const std::uint32_t qv = qs[g * 16 + (w & 15)];
          const std::uint32_t nib = (w >= 16) ? (qv >> 4) : (qv & 15);
          out[b * 256 + g * 32 + w] = dl * kValues[nib];
        }
      }
    }
  }

  static hrx_buffer_ref_t Ref(const LoomBuffer& b) { return {b.handle, 0, b.size}; }
  hrx_buffer_ref_t TRef(const core::TensorInfo& t) const {
    return {weights_, delta_ + static_cast<std::size_t>(t.offset), static_cast<std::size_t>(t.bytes)};
  }
  const core::TensorInfo* Find(const std::string& name) const {
    const auto* t = gguf_.Find(name);
    if (!t) throw LoomError("tensor not found: " + name);
    return t;
  }
  // Zero-filled: a partial chunk leaves rows past its last token tile unwritten, and the full-chunk kernels (conv,
  // DeltaNet) read them. They must be finite: the DeltaNet masks a padding token by multiplying it with 0.
  LoomBuffer& Alloc(std::size_t bytes) {
    keep_.push_back(gpu_.Allocate(bytes));
    Zero(keep_.back());
    return keep_.back();
  }
  // Waits: the fill is stream work, while uploads (tables, Embed) are synchronous copies that would otherwise land first
  // and be zeroed.
  void Zero(LoomBuffer& b) {
    gpu_.Fill(b, 0);
    gpu_.Synchronize();
  }
  // GEMM token tiles covering this chunk's real tokens.
  std::uint32_t TokenTiles(const Geom& g) const { return (n_ + g.tokens - 1) / g.tokens; }
  LoomExecutable& Exe(const std::string& hal) {
    auto it = exes_.find(hal);
    if (it == exes_.end()) it = exes_.emplace(hal, gpu_.Load(dir_ + "/" + hal)).first;
    return it->second;
  }
  // The export's own workgroup size wins; sx is only the fallback for metadata without one.
  // writes: the bindings the kernel may write (bit i = binding i), for the graph's dependencies; by default all of them.
  void Dispatch(const LoomExecutable& exe, const char* name, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                std::uint32_t sx, std::uint32_t sy, std::uint32_t sz, const std::vector<hrx_buffer_ref_t>& b,
                std::uint64_t writes = ~std::uint64_t{0}) {
    const std::uint32_t ordinal = exe.OrdinalOrZero(name);
    const std::uint32_t ws = exe.WorkgroupSize(ordinal);
    const hrx_dispatch_config_t config = LoomDevice::Config(gx, gy, gz, ws ? ws : sx, sy, sz);
    if (graph_)
      graph_->Dispatch(exe, ordinal, config, b.data(), b.size(), writes);
    else
      gpu_.Dispatch(exe, ordinal, config, nullptr, 0, b.data(), b.size());
  }
  // A GEMM writes only its outputs; the hand-written Q2_K GEMM also stages its accumulators in ostage.
  std::uint64_t GemmWrites(const std::vector<hrx_buffer_ref_t>& b, const Fmt& f,
                           std::initializer_list<const LoomBuffer*> outs) const {
    std::uint64_t m = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
      for (const LoomBuffer* o : outs)
        if (b[i].buffer == o->handle) m |= std::uint64_t{1} << i;
      if (std::string(f.name) == "q2k" && b[i].buffer == ostage_->handle) m |= std::uint64_t{1} << i;
    }
    return m;
  }
  // A per-chunk HAL: chunk 0 is "<stem>.hal", chunk c "<stem>_c<c>.hal" (start_pos is compiled in).
  LoomExecutable& ChunkExe(const std::string& stem, std::uint32_t c) {
    return Exe(stem + (c ? "_c" + std::to_string(c) : std::string()) + ".hal");
  }

  void LoadTables() {
    const auto table = [&](const char* file, std::size_t bytes) -> LoomBuffer& {
      LoomBuffer& b = Alloc(bytes);
      std::vector<std::uint8_t> v(bytes);
      ReadFile(dir_ + "/" + file, v.data(), bytes);
      gpu_.H2D(b, v.data(), bytes);
      return b;
    };
    grid_iq3s_ = &table("grid_iq3s.bin", 512 * 4);
    grid_iq3xxs_ = &table("grid_iq3xxs.bin", 256 * 4);
    grid_iq2xxs_ = &table("grid_iq2xxs.bin", 512 * 4);
    grid_iq2xs_ = &table("grid_iq2xs.bin", 1024 * 4);
    ksigns_ = &table("ksigns_iq2xxs.bin", 128);
  }

  void AllocateBuffers() {
    const std::size_t B = B_;
    // Activations are token major, [token][row], with row stride = the K extent of the GEMM that reads them.
    kv_cache_ = std::size_t{T_} * kKvRow;
    hidden_ = &Alloc(B * kHidden * 4);
    reszero_ = &Alloc(B * kHidden * 4);
    sumout_ = &Alloc(B * kHidden * 4);
    scratch_ = &Alloc(B * kFfn * 2);
    qkv_ = &Alloc(B * kQProj * 4);
    gate_ = &Alloc(B * kInner * 4);
    alpha_ = &Alloc(B * kTs * 4);
    beta_ = &Alloc(B * kTs * 4);
    q_ = &Alloc(B * kAttn * 4);
    kbuf_ = &Alloc(B * kKv * 4);
    vbuf_ = &Alloc(B * kKv * 4);
    raw_ = &Alloc(B * kInner * 4);
    conv_out_ = &Alloc(B * kQkv * 4);
    kqbuf_ = &Alloc(B * kKh * 3 * 4);
    ab_ = &Alloc(B * kTs * 2 * 4);
    conv_state_ = &Alloc(std::size_t{48} * kQkv * 4 * 4);
    state_ = &Alloc(std::size_t{48} * kTs * kState * kState * 4);
    // f16 K/V cache: [K | V] x attention layers x context. With "kv16_scratch" (paged or quantized KV) it holds one
    // layer's current chunk only: RoPE writes it and the paged writers or quantizers read it before the next layer.
    kv16_scratch_ = geom_.count("kv16_scratch") != 0;
    kv16_layer_ = kv16_scratch_ ? B * kKvRow * 2 : kv_cache_ * 2;
    kv16_ = &Alloc(kv16_scratch_ ? 2 * kv16_layer_ : std::size_t{2} * full_ * kv_cache_ * 2);
    kc32_ = &Alloc(kv_cache_ * 4);
    vc32_ = &Alloc(kv_cache_ * 4);
    lse_ = &Alloc(B * kHeads * 4);
    eps_ = &Alloc(4);
    ffnup_ = &Alloc(B * kFfn * 2);
    gateffn_ = &Alloc(B * kFfn * 4);
    // Per-workgroup weight staging (ABI only: the tile GEMMs stage in LDS) and epilogue scratch sized for the widest
    // GEMM, which the compiler declares over the whole [m_rows][tokens] tile.
    uwstage_ = &Alloc(std::size_t{kFfn} * 16 * 2);
    wstage_ = &Alloc(std::size_t{kFfn} * 16 * 2);
    ostage_ = &Alloc(std::size_t{kFfn} * B * 4);
    partial_ = &Alloc(B * kHidden * 4);
    hidden2_ = &Alloc(B * kHidden * 4);
    normed_ = &Alloc(std::size_t{kHidden} * 4);
    valid_ = &Alloc(4);
    host_hidden_.resize(B * kHidden);
    const float epsv = 1.0e-6f;
    gpu_.H2D(*eps_, &epsv, 4);
    Zero(*conv_state_);
    Zero(*state_);
    Zero(*kv16_);
    Zero(*reszero_);

    // "kv_paged": 256-token pages, one page table per sequence (logical -> physical page, shared by all layers).
    kv_paged_ = geom_.count("kv_paged") != 0;
    pages_ = (T_ + 255) / 256;
    ptab_ = &Alloc(std::size_t{kv_paged_ ? pages_ : 1} * 4);
    if (kv_paged_) {
      std::vector<std::int32_t> pages(pages_);
      for (std::uint32_t i = 0; i < pages_; ++i) pages[i] = static_cast<std::int32_t>(i);
      // Attention does not clamp page table entries: validate them here.
      for (std::int32_t pg : pages)
        if (pg < 0 || pg >= static_cast<std::int32_t>(pages_)) throw LoomError("page table entry out of range");
      gpu_.H2D(*ptab_, pages.data(), pages.size() * 4);
    }
    ptab_ref_ = {ptab_->handle, 0, std::size_t{kv_paged_ ? pages_ : 1} * 4};
    // Unpaged set with a "vtrans.hal" row: attention reads V as [kv head][16-key tile][dim][16] f16.
    vtrans_ = !kv_paged_ && geom_.count("vtrans.hal");
    vt_bytes_ = std::size_t{(T_ + 15) / 16 * 16} * kKvRow * 2;
    vt16_ = &Alloc(vtrans_ ? vt_bytes_ : 4);
    // Quantized K: "attn_kq8" int8 (kv8a16), "attn_kq4" H256 + asymmetric int4 (kv4a16; kernel yah_kq4 in kq8.hal).
    // Per attention layer: codes [T][1024 or 512 B], scales [T][8 or 32] dwords, the channel mean of the first chunk.
    attn_kq4_ = geom_.count("attn_kq4") != 0;
    attn_kq8_ = geom_.count("attn_kq8") != 0 || attn_kq4_;
    ks_bytes_ = std::size_t{T_} * (attn_kq4_ ? 32 : 8) * 4;
    kq_bytes_ = attn_kq4_ ? kv_cache_ / 2 : kv_cache_;
    kq8buf_ = &Alloc(attn_kq8_ ? std::size_t{full_} * kq_bytes_ : 4);
    ksbuf_ = &Alloc(attn_kq8_ ? std::size_t{full_} * ks_bytes_ : 4);
    kmbuf_ = &Alloc(attn_kq8_ ? std::size_t{full_} * 4096 : 4);
    // Quantized V^T: "attn_vq8" bytes / "attn_vq4" nibbles per channel per 16-key tile, plus (S, C') per tile.
    attn_vq4_ = geom_.count("attn_vq4") != 0;
    attn_vq8_ = geom_.count("attn_vq8") != 0;
    attn_vqt_ = attn_vq4_ || attn_vq8_;
    vq_bytes_ = attn_vq8_ ? vt_bytes_ / 2 : vt_bytes_ / 4;
    vqs_bytes_ = vt_bytes_ / 8;
    vqbuf_ = &Alloc(attn_vqt_ ? std::size_t{full_} * vq_bytes_ : 4);
    vqsbuf_ = &Alloc(attn_vqt_ ? std::size_t{full_} * vqs_bytes_ : 4);
    // Paged fp16 pools: per layer, K rows / V^T tiles of the whole context.
    paged_f16k_ = kv_paged_ && !attn_kq8_;
    paged_f16v_ = kv_paged_ && !attn_vqt_;
    // "rope_kpaged": RoPE writes the fp16 K rows straight into the paged pool.
    rope_kpaged_ = geom_.count("rope_kpaged") != 0;
    if (paged_f16k_ && !rope_kpaged_) throw LoomError("paged fp16 K needs a rope_kpaged HAL set (re-emit)");
    pool_bytes_ = std::size_t{pages_} * 256 * kKvRow * 2;
    kpool_ = &Alloc(paged_f16k_ ? std::size_t{full_} * pool_bytes_ : 4);
    vtpool_ = &Alloc(paged_f16v_ ? std::size_t{full_} * pool_bytes_ : 4);
  }

  void LoadExecutables() {
    // The attention HAL stores f16 straight into the o-projection input ("attn_f16out").
    if (!geom_.count("attn_f16out")) throw LoomError("dispatch.txt lacks attn_f16out (re-emit the set)");
    const auto dn = geom_.find("rowsplit.hal");
    if (dn == geom_.end() || !dn->second.rowgrp)
      throw LoomError("dispatch.txt has no rowsplit.hal row group (re-emit the set)");
    dn_rowgrp_ = dn->second.rowgrp;
    const auto at = geom_.find("wmma.hal");
    if (at == geom_.end() || !at->second.rowgrp || !at->second.tokens)
      throw LoomError("dispatch.txt has no wmma.hal geometry (re-emit the set)");
    attn_hpw_ = at->second.rowgrp;
    attn_tpw_ = at->second.tokens;
    if (kHeads % attn_hpw_) throw LoomError("wmma.hal heads per workgroup does not divide the heads");
    // Loom drops bounds clamps it proves from the launch contract: extra workgroups read unmapped VA and hang.
    if (at->second.tt && (B_ + attn_tpw_ - 1) / attn_tpw_ != at->second.tt)
      throw LoomError("wmma.hal: grid x does not match the emitted token tiles");
    for (const char* hal : {"norm.hal", "convkq.hal", "prepab.hal", "rowsplit.hal", "postnorm.hal", "unpack.hal",
                            "gemv.hal", "rmsnorm.hal", "argmax.hal", "accum.hal"})
      Exe(hal);
  }

  // Bindings shared by every GEMM: weights, then the IQ grid / sign tables the format needs.
  std::vector<hrx_buffer_ref_t> GemmWeights(const core::TensorInfo& t, const Fmt& f) const {
    std::vector<hrx_buffer_ref_t> b = {TRef(t)};
    const std::string n = f.name;
    if (n == "iq3s") b.push_back(Ref(*grid_iq3s_));
    if (n == "iq3xxs") b.push_back(Ref(*grid_iq3xxs_));
    if (n == "iq2xxs") b.push_back(Ref(*grid_iq2xxs_));
    if (n == "iq2xs") b.push_back(Ref(*grid_iq2xs_));
    // The IQ3_XXS and IQ2 sign tables are the same 128 bytes.
    if (n == "iq3xxs" || n == "iq2xxs" || n == "iq2xs") b.push_back(Ref(*ksigns_));
    return b;
  }
  // HAL name "<kind>_<fmt>_<m_tiles>_<k_blocks>.hal" of the GEMM on tensor t.
  std::string GemmHal(const char* kind, const core::TensorInfo& t, Fmt* f) const {
    if (!FmtOf(static_cast<std::uint32_t>(t.type), f)) throw LoomError(std::string("no GEMM format for ") + kind);
    const std::uint32_t mt = static_cast<std::uint32_t>(t.dims[1] / 16);
    const std::uint32_t kb = static_cast<std::uint32_t>(t.dims[0] / f->qk);
    return std::string(kind) + "_" + f->name + "_" + std::to_string(mt) + "_" + std::to_string(kb) + ".hal";
  }
  std::uint32_t MTiles(const core::TensorInfo& t) const { return static_cast<std::uint32_t>(t.dims[1] / 16); }

  void RunNorm(const std::string& wname) {
    Dispatch(Exe("norm.hal"), "yah_half_norm", B_, 1, 1, 32, 1, 1,
             {Ref(*hidden_), Ref(*reszero_), TRef(*Find(wname)), Ref(*sumout_), Ref(*scratch_)});
  }
  void RunKstore(const std::string& wname, const LoomBuffer& out) {
    const auto* t = Find(wname);
    Fmt f{};
    const std::string hal = GemmHal("gemm_kstore", *t, &f);
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*t, f);
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{scratch_, wstage_, ostage_}) b.push_back(Ref(*x));
    b.push_back(Ref(out));
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name)).c_str(), MTiles(*t) / g.rowgrp, TokenTiles(g), 1, 32,
             1, 1, b, GemmWrites(b, f, {&out}));
  }
  // The attention q projection with the q / gate unpack fused in (rows = heads x [256 q | 256 gate]).
  // Returns false if the set has no such HAL; the caller then runs kstore + yah_unpack_qg.
  bool RunKqg(const std::string& wname) {
    const auto* t = Find(wname);
    Fmt f{};
    if (!FmtOf(static_cast<std::uint32_t>(t->type), &f)) return false;
    const std::string hal = GemmHal("gemm_kqg", *t, &f);
    if (!geom_.count(hal)) return false;
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*t, f);
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{scratch_, wstage_, ostage_, q_, gate_}) b.push_back(Ref(*x));
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name) + "_kqg").c_str(), MTiles(*t) / g.rowgrp,
             TokenTiles(g), 1, 32, 1, 1, b, GemmWrites(b, f, {q_, gate_}));
    return true;
  }
  void RunSwiglu(const std::string& wname) {
    const auto* t = Find(wname);
    Fmt f{};
    const std::string hal = GemmHal("gemm_swiglu", *t, &f);
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*t, f);
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{scratch_, gateffn_, uwstage_, ostage_, ffnup_}) b.push_back(Ref(*x));
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name) + "_swiglu").c_str(), MTiles(*t) / g.rowgrp,
             TokenTiles(g), 1, 32, 1, 1, b, GemmWrites(b, f, {ffnup_}));
  }
  // hidden += W input. The fused kres GEMM writes hidden + W input into hidden2; otherwise kStore writes W input
  // into partial and yah_residual_1d adds it. Either way the two hidden buffers swap.
  void RunResidual(const std::string& wname, const LoomBuffer& input) {
    const auto* t = Find(wname);
    Fmt f{};
    const std::string fused = GemmHal("gemm_kres", *t, &f);
    if (geom_.count(fused)) {
      const Geom g = GeomOf(fused);
      auto b = GemmWeights(*t, f);
      for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{&input, hidden_, wstage_, ostage_, hidden2_}) b.push_back(Ref(*x));
      Dispatch(Exe(fused), (std::string("yah_ffn_gemm_") + f.name + "_kres").c_str(), MTiles(*t) / g.rowgrp,
               TokenTiles(g), 1, 32, 1, 1, b, GemmWrites(b, f, {hidden2_}));
      std::swap(hidden_, hidden2_);
      return;
    }
    const std::string hal = GemmHal("gemm_kstore", *t, &f);
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*t, f);
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{&input, wstage_, ostage_, partial_}) b.push_back(Ref(*x));
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name)).c_str(), MTiles(*t) / g.rowgrp, TokenTiles(g), 1, 32,
             1, 1, b, GemmWrites(b, f, {partial_}));
    const std::size_t n = std::size_t{B_} * kHidden;
    Dispatch(Exe("accum.hal"), "yah_residual_1d", static_cast<std::uint32_t>(n / 256), 1, 1, 256, 1, 1,
             {Ref(*hidden_), {partial_->handle, 0, n * 4}, Ref(*hidden2_)});
    std::swap(hidden_, hidden2_);
  }

  void RunAttention(std::uint32_t l, std::uint32_t ci, const std::string& pre, const KvHook& hook) {
    const std::uint32_t ai = l / cfg_.full_attention_interval;
    if (ai >= full_) throw LoomError("full-attention layer index past the KV slot count");
    const bool qg_fused = RunKqg(pre + "attn_q.weight");
    if (!qg_fused) RunKstore(pre + "attn_q.weight", *qkv_);
    RunKstore(pre + "attn_k.weight", *kbuf_);
    RunKstore(pre + "attn_v.weight", *vbuf_);
    if (!qg_fused)
      Dispatch(Exe("unpack.hal"), "yah_unpack_qg", 24, B_, 1, 256, 1, 1, {Ref(*qkv_), Ref(*q_), Ref(*gate_)});
    const std::size_t koff = kv16_scratch_ ? 0 : std::size_t{ai} * kv_cache_ * 2;
    const std::size_t voff = kv16_scratch_ ? kv16_layer_ : std::size_t{full_} * kv_cache_ * 2 + koff;
    const hrx_buffer_ref_t kpool_l{kpool_->handle, std::size_t{ai} * pool_bytes_, pool_bytes_};
    const hrx_buffer_ref_t vtpool_l{vtpool_->handle, std::size_t{ai} * pool_bytes_, pool_bytes_};
    {
      std::vector<hrx_buffer_ref_t> b = {Ref(*q_),
                                         Ref(*kbuf_),
                                         Ref(*vbuf_),
                                         TRef(*Find(pre + "attn_q_norm.weight")),
                                         TRef(*Find(pre + "attn_k_norm.weight")),
                                         Ref(*q_),
                                         Ref(*kbuf_),
                                         Ref(*kc32_),
                                         Ref(*vc32_),
                                         rope_kpaged_ ? kpool_l : hrx_buffer_ref_t{kv16_->handle, koff, kv16_layer_},
                                         {kv16_->handle, voff, kv16_layer_},
                                         {eps_->handle, 0, 4}};
      if (rope_kpaged_) b.push_back(ptab_ref_);
      Dispatch(ChunkExe("rope", ci), "yah_fused_qk_rope_batched", 28, B_, 1, 256, 1, 1, b);
    }
    if (hook) hook(ai, ci, koff, voff);
    const std::size_t q8off = std::size_t{ai} * kq_bytes_;
    const std::size_t ksoff = std::size_t{ai} * ks_bytes_;
    const std::size_t kmoff = std::size_t{ai} * 4096;
    const std::size_t first = std::size_t{ci} * B_;              // this chunk's first cache row
    const std::size_t f16rows = std::size_t{B_} * kKvRow * 2;    // the chunk's f16 K or V rows
    const std::size_t f16first = kv16_scratch_ ? 0 : first * kKvRow * 2;
    if (attn_kq8_) {
      const std::size_t qrow = kq_bytes_ / T_, srow = ks_bytes_ / T_;
      if (ci == 0)  // channel mean of the first chunk, kept for the later ones
        Dispatch(Exe("kmean.hal"), "yah_kmean", 4, 1, 1, 256, 1, 1,
                 {{kv16_->handle, koff, f16rows}, {kmbuf_->handle, kmoff, 4096}});
      std::vector<hrx_buffer_ref_t> b = {{kv16_->handle, koff + f16first, f16rows},
                                         {kmbuf_->handle, kmoff, 4096},
                                         {kq8buf_->handle, q8off + first * qrow, B_ * qrow},
                                         {ksbuf_->handle, ksoff + first * srow, B_ * srow}};
      if (kv_paged_) {  // whole-layer pools, rows placed through the page table
        b[2] = {kq8buf_->handle, q8off, kq_bytes_};
        b[3] = {ksbuf_->handle, ksoff, ks_bytes_};
        b.push_back(ptab_ref_);
      }
      Dispatch(kv_paged_ ? ChunkExe("kq8", ci) : Exe("kq8.hal"), attn_kq4_ ? "yah_kq4" : "yah_kq8", (B_ + 1) / 2, 1, 1,
               256, 1, 1, b);
    }
    const std::size_t vqoff = std::size_t{ai} * vq_bytes_, vqsoff = std::size_t{ai} * vqs_bytes_;
    if (attn_vqt_) {
      std::vector<hrx_buffer_ref_t> b = {{kv16_->handle, voff + f16first, f16rows},
                                         {vqbuf_->handle, vqoff, vq_bytes_},
                                         {vqsbuf_->handle, vqsoff, vqs_bytes_}};
      if (kv_paged_) b.push_back(ptab_ref_);
      Dispatch(ChunkExe(attn_vq8_ ? "vq8" : "vq4", ci), attn_vq8_ ? "yah_vq8" : "yah_vq4", 4, (B_ + 15) / 16, 1, 256,
               1, 1, b);
    } else if (paged_f16v_) {
      Dispatch(ChunkExe("vtpage", ci), "yah_vtpage", 32, (B_ + 31) / 32, 1, 256, 1, 1,
               {{kv16_->handle, voff, f16rows}, vtpool_l, ptab_ref_});
    } else if (vtrans_) {
      Dispatch(Exe("vtrans.hal"), "yah_transpose_v16", 32, (T_ + 31) / 32, 1, 256, 1, 1,
               {{kv16_->handle, voff, kv_cache_ * 2}, {vt16_->handle, 0, vt_bytes_}});
    }
    {
      std::vector<hrx_buffer_ref_t> b = {
          Ref(*q_),
          Ref(*gate_),
          attn_kq8_     ? hrx_buffer_ref_t{kq8buf_->handle, q8off, kq_bytes_}
          : paged_f16k_ ? kpool_l
                        : hrx_buffer_ref_t{kv16_->handle, koff, kv_cache_ * 2},
          attn_vqt_     ? hrx_buffer_ref_t{vqbuf_->handle, vqoff, vq_bytes_}
          : paged_f16v_ ? vtpool_l
          : vtrans_     ? hrx_buffer_ref_t{vt16_->handle, 0, vt_bytes_}
                        : hrx_buffer_ref_t{kv16_->handle, voff, kv_cache_ * 2},
          Ref(*scratch_),
          Ref(*lse_)};
      if (attn_kq8_) b.push_back({ksbuf_->handle, ksoff, ks_bytes_});
      if (attn_vqt_) b.push_back({vqsbuf_->handle, vqsoff, vqs_bytes_});
      if (kv_paged_) b.push_back(ptab_ref_);
      Dispatch(ChunkExe("wmma", ci), "yah_attn_wmma", (B_ + attn_tpw_ - 1) / attn_tpw_, kHeads / attn_hpw_, 1, 256, 1,
               1, b);
    }
    RunResidual(pre + "attn_output.weight", *scratch_);
  }

  void RunDeltaNet(std::uint32_t l, const std::string& pre) {
    const std::uint32_t si = l - l / cfg_.full_attention_interval;
    RunKstore(pre + "attn_qkv.weight", *qkv_);
    RunKstore(pre + "attn_gate.weight", *gate_);
    RunKstore(pre + "ssm_alpha.weight", *alpha_);
    RunKstore(pre + "ssm_beta.weight", *beta_);
    const hrx_buffer_ref_t cs{conv_state_->handle, std::size_t{si} * kQkv * 4 * 4, std::size_t{kQkv} * 4 * 4};
    const hrx_buffer_ref_t st{state_->handle, std::size_t{si} * kTs * kState * kState * 4,
                              std::size_t{kTs} * kState * kState * 4};
    // The conv with the q / k L2 norm (prep_kq) fused in.
    Dispatch(Exe("convkq.hal"), "yah_ssm_conv_kq", 40, B_, 1, 256, 1, 1,
             {Ref(*qkv_), TRef(*Find(pre + "ssm_conv1d.weight")), cs, Ref(*conv_out_), Ref(*kqbuf_)});
    // One lane index does two jobs: advance the conv ring past the real tokens (i < qkv_size; the next chunk and the
    // decoder read it) and alpha / beta (i < B * heads). So the grid covers max() of the two.
    const std::uint32_t prepab_tiles = (std::max<std::uint32_t>(B_ * kTs, kQkv) + 255u) / 256u;
    Dispatch(Exe("prepab.hal"), "yah_deltanet_prep_ab", prepab_tiles, 1, 1, 256, 1, 1,
             {Ref(*alpha_), Ref(*beta_), TRef(*Find(pre + "ssm_a")), TRef(*Find(pre + "ssm_dt.bias")), Ref(*qkv_), cs,
              Ref(*ab_), Ref(*valid_)});
    // DeltaNet grid: (blocks per head, heads) x 256, blocks per head from the "rowsplit.hal" row group.
    Dispatch(Exe("rowsplit.hal"), "yah_deltanet", dn_rowgrp_, kTs, 1, 256, 1, 1,
             {Ref(*conv_out_), Ref(*kqbuf_), Ref(*ab_), st, Ref(*raw_)});
    Dispatch(Exe("postnorm.hal"), "yah_ssm_postnorm_fp16", 6 * B_, 1, 1, 256, 1, 1,
             {Ref(*raw_), TRef(*Find(pre + "ssm_norm.weight")), {gate_->handle, 0, std::size_t{B_} * kInner * 4},
              Ref(*scratch_)});
    RunResidual(pre + "ssm_out.weight", *scratch_);
  }

  LoomDevice& gpu_;
  const core::Gguf& gguf_;
  const core::Qwen35Config& cfg_;
  std::string dir_;
  hrx_buffer_t weights_;
  std::size_t delta_;
  std::map<std::string, Geom> geom_;
  std::map<std::string, LoomExecutable> exes_;
  std::deque<LoomBuffer> keep_;  // stable addresses
  std::uint32_t B_ = 0, T_ = 0, full_ = 0, pages_ = 0, dn_rowgrp_ = 0, attn_hpw_ = 0, attn_tpw_ = 0;
  std::uint32_t n_ = 0;  // real tokens of the chunk from the last Embed
  LoomGraph* graph_ = nullptr;  // open while RunLayers records
  bool kv16_scratch_ = false, kv_paged_ = false, vtrans_ = false, rope_kpaged_ = false;
  bool attn_kq4_ = false, attn_kq8_ = false, attn_vq4_ = false, attn_vq8_ = false, attn_vqt_ = false;
  bool paged_f16k_ = false, paged_f16v_ = false;
  std::size_t kv_cache_ = 0, kv16_layer_ = 0, vt_bytes_ = 0, ks_bytes_ = 0, kq_bytes_ = 0, vq_bytes_ = 0,
              vqs_bytes_ = 0, pool_bytes_ = 0;
  hrx_buffer_ref_t ptab_ref_{};
  std::vector<float> host_hidden_;
  LoomBuffer *grid_iq3s_ = nullptr, *grid_iq3xxs_ = nullptr, *grid_iq2xxs_ = nullptr, *grid_iq2xs_ = nullptr,
             *ksigns_ = nullptr;
  LoomBuffer *hidden_ = nullptr, *reszero_ = nullptr, *sumout_ = nullptr, *scratch_ = nullptr, *qkv_ = nullptr,
             *gate_ = nullptr, *alpha_ = nullptr, *beta_ = nullptr, *q_ = nullptr, *kbuf_ = nullptr,
             *vbuf_ = nullptr, *raw_ = nullptr, *conv_out_ = nullptr, *kqbuf_ = nullptr, *ab_ = nullptr,
             *conv_state_ = nullptr, *state_ = nullptr, *kv16_ = nullptr, *kc32_ = nullptr, *vc32_ = nullptr,
             *lse_ = nullptr, *eps_ = nullptr, *ffnup_ = nullptr, *gateffn_ = nullptr, *uwstage_ = nullptr,
             *wstage_ = nullptr, *ostage_ = nullptr, *partial_ = nullptr, *hidden2_ = nullptr, *normed_ = nullptr,
             *ptab_ = nullptr, *vt16_ = nullptr, *kq8buf_ = nullptr, *ksbuf_ = nullptr, *kmbuf_ = nullptr,
             *vqbuf_ = nullptr, *vqsbuf_ = nullptr, *kpool_ = nullptr, *vtpool_ = nullptr, *valid_ = nullptr;
};

}  // namespace yah::model

#endif  // YAH_MODEL_LOOM_PREFILL_HPP_
