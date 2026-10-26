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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "model/loom_decoder.hpp"
#include "model/loom_runtime.hpp"
#include "model/npu_down_pack.hpp"
#include "model/npu_split.hpp"

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
              const LoomWeights& weights, std::uint32_t tokens = 0)
      : gpu_(gpu), gguf_(gguf), cfg_(cfg), dir_(std::move(dir)), weights_(weights) {
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
    WarmExecutables();
  }

  [[nodiscard]] std::uint32_t chunk() const { return B_; }
  [[nodiscard]] std::uint32_t context() const { return T_; }
  [[nodiscard]] bool paged() const { return kv_paged_; }
  // KV bits of this set: (16, 16) fp16, else 8 or 4 per side.
  [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> kv_bits() const {
    return {attn_kq4_ ? 4u : attn_kq8_ ? 8u : 16u, attn_vq4_ ? 4u : attn_vq8_ ? 8u : 16u};
  }
  LoomBuffer& hidden() { return *hidden_; }

  // Chunk pairs (RunPair): two full chunks recorded layer by layer, so the NPU's FFN block of one runs while the GPU
  // runs the other's layer. Each chunk of a pair has its own residual rows, ids and NPU A / C half; SelectChunk(p)
  // picks the one Embed, Head and hidden() see. Before npu_plan() / EnableNpu.
  void EnablePairs() {
    if (pairs_) return;
    pairs_ = true;
    other_ = {&Alloc(std::size_t{B_} * kHidden * 4), &Alloc(std::size_t{B_} * kHidden * 4), &Alloc(std::size_t{B_} * 4),
              &Alloc(4)};
    // a chained DeltaNet tail leaves its outputs to the next layer of its chunk untracked: the other chunk's layer
    // runs between them here
    dnring_ = false;
  }
  [[nodiscard]] bool pairs() const { return pairs_; }
  void SelectChunk(std::uint32_t p) {
    if (p > 1 || (p && !pairs_)) throw LoomError("prefill: no such chunk of a pair");
    if (p == par_) return;
    FlushKres();
    std::swap(hidden_, other_.hidden), std::swap(hidden2_, other_.hidden2), std::swap(ids_, other_.ids);
    std::swap(valid_, other_.valid), std::swap(n_, other_.n), std::swap(keep_rows_, other_.keep);
    std::swap(answer_, other_.answer);
    par_ = p;
  }

  // Zero the recurrent state before a new sequence. KV rows need no reset: attention reads only keys <= the query.
  void Reset() {
    Zero(*conv_state_);
    Zero(*state_);
  }
  // The recurrent state (conv rings, DeltaNet states) that chunks carry forward and DecoderState() hands on.
  [[nodiscard]] hrx_buffer_ref_t conv_state() const { return Ref(*conv_state_); }
  [[nodiscard]] hrx_buffer_ref_t delta_state() const { return Ref(*state_); }
  // fpsparse sets keep block means that a resumed sequence would not rebuild.
  [[nodiscard]] bool fpsparse() const { return fps_on_; }

  // The NPU FFN block (npu.txt, tools/emit_npu_ffn.py): the NPU owns the FFN's last F features of every layer. Its
  // gate / up calls (npu_ffnsw.xdna, F / 256 of them: call j's column c takes features G + 256 j + 32 c + [0, 32) of the
  // G = 17408 - F the GPU leaves, gate rows and up rows decoded on the NPU from the weight chunks) write H =
  // bfp16(silu(gate) up) into C, [M block][pass][16-token slab][128 k-blocks]; its down calls (npu_ffndn<passes>.xdna,
  // per K part of F, kDownCalls calls of 512 rows) read H and decode IQ4_XS rows (npu_down_pack.hpp: down's rows
  // unrotated, its awq folded in, re-quantized) and write their partials after H. The GPU runs gate / up and down over the first G features
  // into hidden2, then npu_unpack_ffn.hal adds the NPU's partials into hidden.
  // npu_plan: the sizes LoomNpuSplit needs (a_bytes 0: the set has no block).
  [[nodiscard]] NpuPlan npu_plan() const {
    NpuPlan p;
    if (!FfnF()) return p;
    const NpuBytes sw = NpuBytesOf("npubytes_ffnsw");
    p.a_bytes = sw.a, p.w_bytes = sw.w;   // (W: the images' panel binding, not read)
    for (const auto& [p0, passes] : FfnParts()) p.w_bytes = std::max(p.w_bytes, NpuBytesOf(DnImage(passes, "npubytes_")).w);
    p.c_bytes = NpuCBytes();
    if (pairs_) p.a_bytes *= 2, p.c_bytes *= 2;   // a half per chunk of a pair
    p.dcol = p.gfirst = true;
    const Geom& g = geom_.at("npugate");
    p.gate_calls = g.rowgrp, p.gate_record = g.tt;
    return p;
  }
  // Full chunks run the block from now on. npu (built from npu_plan()) outlives this object. Packs down's rows for the
  // NPU (~3 s). layers: only the first that many layers (a test's smaller footprint: their down rows alone are packed).
  void EnableNpu(NpuSplit* npu, std::uint32_t layers = ~0u);

  // Stage ids[0..n) for the next RunLayers, whose first kernel (yah_embed) writes their token_embd rows into hidden().
  // n < B pads the chunk with the last token.
  // The GEMMs (~92% of the time) run only the token tiles that hold real tokens; every other kernel runs
  // the whole chunk, so padding rows next to the real ones keep their padded-path values. Shrinking the norms as well
  // changed the last token's logits by ~1e-4 when it sat alone in its 16-token group (cause not found yet). Padding tokens leave the recurrent state
  // unchanged and attention is causal, so the real tokens' results and the state carried forward are exact.
  // The ids and the token count go to the GPU in stream order (stream updates): the host never waits here.
  void Embed(const std::uint32_t* ids, std::uint32_t n) {
    if (n == 0 || n > B_) throw LoomError("prefill: chunk token count out of range");
    for (std::uint32_t t = 0; t < B_; ++t) {
      host_ids_[t] = ids[std::min(t, n - 1)];
      if (host_ids_[t] >= kVocab)
        throw LoomError("token id " + std::to_string(host_ids_[t]) + " is outside the vocabulary");
    }
    gpu_.Update(*ids_, host_ids_.data(), host_ids_.size() * 4);
    const std::int32_t valid = static_cast<std::int32_t>(n);
    gpu_.Update(*valid_, &valid, 4);
    n_ = n;
  }

  // RunLayers' keep: which rows of hidden() the caller reads after the last layer (Head). kAllRows: every row; kNoRows:
  // none, so the last layer's tail (attention / o-proj or postnorm / ssm_out, and the FFN) is skipped (its K / V and
  // recurrent state are still written); a row index: the tail's GEMMs run only the token tile that holds it.
  static constexpr std::int64_t kAllRows = -1, kNoRows = -2;

  // Enqueue the 64 layers for chunk ci (absolute positions ci * B ..), on the hidden() rows from Embed(). answer: the
  // chunk holds the prompt's last token, whose row the caller reads (fpsparse keeps its 16-token block exact).
  void RunLayers(std::uint32_t ci, const KvHook& hook = {}, std::int64_t keep = kAllRows, bool answer = false) {
    keep_rows_ = keep;
    answer_ = answer;
    if (std::size_t{ci + 1} * B_ > T_) throw LoomError("prefill: chunk past the emitted context");
    if (n_ == 0) throw LoomError("prefill: RunLayers before Embed");
    if (flags_) gpu_.Fill(*flags_, 0);   // the chunk's completion counters (FlagSlot), in stream order before its work
    // The layers go into one graph, so kernels with no data between them (the input projections of a layer, the DeltaNet
    // gate projection and the conv / DeltaNet chain) can run at the same time.
    // NPU FFN block: full chunks only, its handoffs graph nodes on flag words (NpuEnqueue / NpuJoin); the jobs are
    // queued once the chunk's one graph launched
    // (with pairs, a lone full chunk runs without the NPU too: alone, its layers would wait for the NPU's block)
    npu_on_ = npu_ && n_ == B_ && !pairs_;
    if (npu_on_) npu_->BeginChunk();
    flagged_.clear(), job_c_.clear();
    npu_cleared_ = npu_joined_ = 0;
    NewGraph();
    // Layer 0 goes first in a graph of its own, so the GPU runs it while the host records the other layers. The split
    // falls before its deferred down GEMM (the first dispatch while recording layer 1), which waits for all of the
    // layer's last kernel anyway.
    split_layer_ = !npu_on_ && cfg_.main_block_count() > 1 ? 1 : 0;
    RecordLayers(ci, hook);
    FlushKres();
    split_layer_ = 0, split_armed_ = false;
    LaunchChunk();
  }

  // Chunks ci and ci + 1, both full and staged (SelectChunk, Embed), recorded layer by layer: chunk 0's layer l, then
  // chunk 1's. A chunk's NPU join and unpack of layer l go before its layer l + 1, after the other chunk's layer l, so
  // the NPU's block of one chunk runs while the GPU runs the other's. keep / answer: per chunk, as RunLayers'.
  void RunPair(std::uint32_t ci, std::int64_t keep0, std::int64_t keep1, bool answer1) {
    if (!pairs_) throw LoomError("prefill: RunPair without EnablePairs");
    if (std::size_t{ci + 2} * B_ > T_) throw LoomError("prefill: chunk past the emitted context");
    SelectChunk(1);
    keep_rows_ = keep1, answer_ = answer1;
    const bool full1 = n_ == B_;
    SelectChunk(0);
    keep_rows_ = keep0, answer_ = false;
    if (n_ != B_ || !full1) throw LoomError("prefill: RunPair needs two full chunks");
    if (flags_) gpu_.Fill(*flags_, 0);
    npu_on_ = npu_ != nullptr;
    if (npu_on_) npu_->BeginChunk();
    flagged_.clear(), job_c_.clear();
    npu_cleared_ = npu_joined_ = 0;
    join_[0] = join_[1] = kNoJoin;
    NewGraph();
    split_layer_ = 0;
    flag_slot_ = 0;
    for (std::uint32_t p = 0; p < 2; ++p) {
      SelectChunk(p);
      Dispatch(Exe("embed.hal"), "yah_embed", GeomOf("embed.hal").tt, 1, 1, kHidden / 16, 1, 1,
               {TRef(*Find("token_embd.weight")), Ref(*ids_), Ref(*hidden_)}, 4);
    }
    for (std::uint32_t l = 0; l < cfg_.main_block_count(); ++l)
      for (std::uint32_t p = 0; p < 2; ++p) {
        SelectChunk(p);
        JoinFfn();
        RecordLayer(l, ci + p, {});
      }
    for (std::uint32_t p = 0; p < 2; ++p) SelectChunk(p), JoinFfn();
    FlushKres();
    LaunchChunk();
  }

  // The recorded chunk's graph to the GPU, its NPU jobs to the NPU.
  void LaunchChunk() {
    graph_ = nullptr;
    std::vector<NpuSplit::Queued> jobs;
    jobs.swap(flagged_);
    // YAH_NPU_CPU=1 (a safe test of the NPU side): the host stands in for the GPU, which runs nothing; it stores each
    // job's ready word, waits up to 1 s for its done word and reports the first job that stalls or fails
    if (!jobs.empty() && std::getenv("YAH_NPU_CPU")) {
      NpuCpuDriven(jobs);
      return;
    }
    // YAH_NPU_FAKE=1 (a timing test): the NPU runs nothing and the graph does not wait for it, so the run takes the
    // GPU's share of the block alone (the NPU's partials are garbage)
    if (std::getenv("YAH_NPU_FAKE")) jobs.clear();
    chunk_graph_->Launch();
    if (!jobs.empty()) {
      try {
        npu_->Enqueue(jobs);   // on a failure it releases the graph's waits for the jobs left
      } catch (...) {
        gpu_.Synchronize();   // the graph reads NPU buffers: it must end before anyone frees them
        throw;
      }
    }
  }

  // The 64 layers of chunk ci, in dispatch order.
  void RecordLayers(std::uint32_t ci, const KvHook& hook) {
    flag_slot_ = 0;
    Dispatch(Exe("embed.hal"), "yah_embed", GeomOf("embed.hal").tt, 1, 1, kHidden / 16, 1, 1,
             {TRef(*Find("token_embd.weight")), Ref(*ids_), Ref(*hidden_)}, 4);
    for (std::uint32_t l = 0; l < cfg_.main_block_count(); ++l) RecordLayer(l, ci, hook);
    JoinFfn();
  }
  void RecordLayer(std::uint32_t l, std::uint32_t ci, const KvHook& hook) {
    const std::string pre = "blk." + std::to_string(l) + ".";
    split_armed_ = split_layer_ && l == split_layer_;
    npu_layer_ = l;
    tail_ = l + 1 == cfg_.main_block_count() ? keep_rows_ : kAllRows;
    nq_want_ = nqw_;
    RunNorm(pre + "attn_norm.weight");
    nq_want_ = false;
    if (cfg_.IsFullAttention(l))
      RunAttention(l, ci, pre, hook);
    else
      RunDeltaNet(l, pre);
    if (tail_ != kNoRows) {
      trim_row_ = tail_;
      RunFfn(pre);
      trim_row_ = kAllRows;
    }
    tail_ = kAllRows;
  }


  // Final norm + output head of hidden() row `row` (this chunk) into dst (kVocab f32).
  void Head(std::uint32_t row, const hrx_buffer_ref_t& dst) {
    const auto* onw = Find("output_norm.weight");
    const auto* ow = Find("output.weight");
    Dispatch(Exe("rmsnorm.hal"), "yah_rmsnorm", 1, 1, 1, 32, 1, 1,
             {{hidden_->handle, std::size_t{row} * kHidden * 4, std::size_t{kHidden} * 4}, TRef(*onw), Ref(*normed_)});
    // the f32 norm row into the weights' basis (no awq), then gen_gemv's tile-free GEMV
    Dispatch(Exe("rot_head.hal"), "yah_rot256", kHidden / 1024, 1, 1, 64, 1, 1, {Ref(*normed_), Ref(*rot_signs_)});
    const Geom& g = geom_.at("gemv.hal");   // (rows per workgroup, lanes, workgroups)
    Dispatch(Exe("gemv.hal"), "yah_gemv", g.tt, 1, 1, g.rowgrp, 1, 1, {TRef(*ow), Ref(*normed_), dst});
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

  // A slot: the sequence state after `chunks` chunks (KV pools, page table, conv rings, DeltaNet states), raw. It loads
  // only into a prefill of the same set (context, chunk size, KV format); LoadState returns the chunks it covers, so the
  // caller continues at that chunk index (marginal prefill at depth without re-running the prefix).
  void SaveState(const std::string& path, std::uint32_t chunks) {
    gpu_.Synchronize();
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw LoomError("prefill: cannot write the slot " + path);
    const auto bufs = StateBuffers();
    const std::uint64_t head[5] = {kSlotMagic, T_, B_, chunks, bufs.size()};
    bool ok = std::fwrite(head, sizeof head, 1, f) == 1;
    std::vector<std::uint8_t> host(std::size_t{64} << 20);
    for (const LoomBuffer* b : bufs) {
      const std::uint64_t n = b->size;
      ok = ok && std::fwrite(&n, 8, 1, f) == 1;
      for (std::size_t o = 0; ok && o < b->size; o += host.size()) {
        const std::size_t k = std::min(host.size(), b->size - o);
        gpu_.D2H(*b, host.data(), k, o);
        ok = std::fwrite(host.data(), 1, k, f) == k;
      }
    }
    if (std::fclose(f) != 0 || !ok) throw LoomError("prefill: short write to the slot " + path);
  }
  std::uint32_t LoadState(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw LoomError("prefill: cannot open the slot " + path);
    const auto bufs = StateBuffers();
    std::uint64_t head[5] = {};
    bool ok = std::fread(head, sizeof head, 1, f) == 1 && head[0] == kSlotMagic && head[1] == T_ && head[2] == B_ &&
              head[4] == bufs.size();
    std::vector<std::uint8_t> host(std::size_t{64} << 20);
    for (const LoomBuffer* b : bufs) {
      std::uint64_t n = 0;
      ok = ok && std::fread(&n, 8, 1, f) == 1 && n == b->size;
      for (std::size_t o = 0; ok && o < b->size; o += host.size()) {
        const std::size_t k = std::min(host.size(), b->size - o);
        ok = std::fread(host.data(), 1, k, f) == k;
        if (ok) gpu_.H2D(*b, host.data(), k, o);
      }
    }
    std::fclose(f);
    if (!ok) throw LoomError("prefill: the slot " + path + " does not match this set (context, chunk, KV format)");
    gpu_.Synchronize();
    return static_cast<std::uint32_t>(head[3]);
  }

 private:
  static constexpr std::uint64_t kSlotMagic = 0x31544f4c53484159ull;  // "YAHSLOT1"
  // Paged sets keep the whole sequence state in these (the f16 K/V buffer is per-chunk scratch there).
  [[nodiscard]] std::vector<const LoomBuffer*> StateBuffers() const {
    if (!kv_paged_ || !kv16_scratch_) throw LoomError("prefill: slots need the paged KV layout (the default set)");
    std::vector<const LoomBuffer*> v = {ptab_, kq8buf_, ksbuf_, kmbuf_, vqbuf_, vqsbuf_, kpool_, vtpool_, conv_state_, state_};
    if (fps_on_) v.insert(v.end(), {kfm_, vfm_});   // the block means (a slot of an fpsparse set)
    return v;
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
    if (!f) throw LoomError("cannot open " + dir_ + "/dispatch.txt: not a prefill set (tools/emit_prefill_pp.py)");
    std::string name;
    Geom g{};
    while (f >> name >> g.tokens >> g.rowgrp >> g.tt) geom_[name] = g;
    std::ifstream n(dir_ + "/npu.txt");   // the NPU FFN block's rows (tools/emit_npu_ffn.py), if the set has it
    while (n >> name >> g.tokens >> g.rowgrp >> g.tt) geom_[name] = g;
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
  static hrx_buffer_ref_t Ref(const LoomBuffer& b) { return {b.handle, 0, b.size}; }
  hrx_buffer_ref_t TRef(const core::TensorInfo& t) const {
    return weights_.Ref(t);
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
  // The last layer's tail on one row (trim_row_ >= 0): a GEMM dispatches only the token tile that holds the row, its
  // token-major bindings b[first..] moved to that tile (row_bytes: bytes per token row, 0 for the others). Each output row
  // depends only on its own input row, so the kept row is the same; the other rows are left unwritten. The
  // fragment-major layout groups 16 rows per tile row, so a tile start (a multiple of 16) has the same byte offset.
  // Returns the token tiles to launch.
  std::uint32_t Trim(std::vector<hrx_buffer_ref_t>& b, std::size_t first, const std::vector<std::size_t>& row_bytes,
                     const Geom& g) const {
    if (trim_row_ < 0) return TokenTiles(g);
    const std::size_t t0 = static_cast<std::size_t>(trim_row_) / g.tokens * g.tokens;
    std::size_t i = first;
    for (const std::size_t rb : row_bytes) {
      if (rb) b[i].offset += t0 * rb, b[i].length -= t0 * rb;
      ++i;
    }
    return 1;
  }
  LoomExecutable& Exe(const std::string& hal) {
    auto it = exes_.find(hal);
    if (it == exes_.end()) it = exes_.emplace(hal, gpu_.Load(dir_ + "/" + hal)).first;
    return it->second;
  }
  // The export's own workgroup size wins; sx is only the fallback for metadata without one.
  // writes: the bindings the kernel may write (bit i = binding i), for the graph's dependencies; by default all of them.
  // also_writes: ranges the dispatch stands for as written though it does not bind them (LoomGraph::Dispatch).
  // untracked: bindings the kernel orders itself (LoomGraph::Dispatch).
  void Dispatch(const LoomExecutable& exe, const char* name, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz,
                std::uint32_t sx, std::uint32_t sy, std::uint32_t sz, const std::vector<hrx_buffer_ref_t>& b,
                std::uint64_t writes = ~std::uint64_t{0}, const std::vector<hrx_buffer_ref_t>* also_writes = nullptr,
                std::uint64_t untracked = 0) {
    if (split_armed_ && graph_ && !warming_) SplitGraph();
    FlushKres();
    if (warming_) return;
    const std::uint32_t ordinal = exe.OrdinalOrZero(name);
    const std::uint32_t ws = exe.WorkgroupSize(ordinal);
    const hrx_dispatch_config_t config = LoomDevice::Config(gx, gy, gz, ws ? ws : sx, sy, sz);
    if (graph_)
      graph_->Dispatch(exe, ordinal, config, b.data(), b.size(), writes, nullptr, also_writes, untracked);
    else
      gpu_.Dispatch(exe, ordinal, config, nullptr, 0, b.data(), b.size());
  }
  // A GEMM writes only its outputs.
  std::uint64_t GemmWrites(const std::vector<hrx_buffer_ref_t>& b, std::initializer_list<const LoomBuffer*> outs) const {
    std::uint64_t m = 0;
    for (std::size_t i = 0; i < b.size(); ++i)
      for (const LoomBuffer* o : outs)
        if (b[i].buffer == o->handle) m |= std::uint64_t{1} << i;
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
    if (!geom_.count("rot"))
      throw LoomError("prefill: not an MQ4 set (an emit_prefill_pp.py set from before 2026-10-20?)");
    a4_ = geom_.count("a4") != 0;
    nqov_ = a4_ && geom_.count("nqov") != 0;
    dnov_ = a4_ && geom_.count("dnov") != 0;
    pnq_ab_ = dnov_ && geom_.count("pnq_abf.hal") != 0;
    guov_ = a4_ && geom_.count("guov") != 0;
    nqgu_ = guov_ && geom_.count("nqgu") != 0;
    rqk_ = nqov_ && geom_.count("rqk") != 0;
    rqp_ = rqk_ && guov_ && geom_.count("rotqp_17408.hal") != 0;
    aov_ = rqk_ && geom_.count("aov") != 0 && std::ifstream(dir_ + "/wmmaf.hal").good();
    ropew_ = geom_.count("ropew") != 0;
    ropekv_ = geom_.count("ropekv") != 0;
    if (const auto it = geom_.find("nqp"); it != geom_.end()) nqp_ = it->second.tokens;
    nqw_ = a4_ && geom_.count("nqw") != 0;
    dnring_ = nqw_ && rqk_ && geom_.count("dnring") != 0;
    rot_signs_ = &table("rot_signs.bin", 512 * 4);
  }

  void AllocateBuffers() {
    const std::size_t B = B_;
    // Activations are token major, [token][row], with row stride = the K extent of the GEMM that reads them.
    kv_cache_ = std::size_t{T_} * kKvRow;
    hidden_ = &Alloc(B * kHidden * 4);
    reszero_ = &Alloc(B * kHidden * 4);
    sumout_ = &Alloc(B * kHidden * 4);
    scratch_ = &Alloc(B * kFfn * 2);
    if (!a4_) rotbuf_ = &Alloc(B * kInner * 2);
    if (a4_) aq_ = &Alloc(B * kFfn / 2), ad_ = &Alloc(B * (kFfn / 128) * 4);
    if (nqov_ || dnov_ || guov_) {
      // the flag-waiting kernels' activation pair (swapped with aq_ / ad_ after each) and their producers' counters
      aq2_ = &Alloc(B * kFfn / 2), ad2_ = &Alloc(B * (kFfn / 128) * 4);
      flags_ = &Alloc(kFlagSlots * TokenTiles128() * 4);
    }
    qkv_ = &Alloc(B * kQProj * 4);
    gate_ = &Alloc(B * kInner * 4);
    alpha_ = &Alloc(B * kTs * 4);
    beta_ = &Alloc(B * kTs * 4);
    q_ = &Alloc(B * kAttn * 4);
    // q16 sets: RoPE writes the attention's f16(q / 16) here instead of the f32 query in place
    q16_ = geom_.count("q16") ? &Alloc(B * kAttn * 2) : q_;
    // kv4 sets at depth (dispatch.txt "qrot <first chunk>"): yah_qrot writes the H256-rotated f16 Q here for the attention
    if (const auto qr = geom_.find("qrot"); qr != geom_.end()) qr16_ = &Alloc(B * kAttn * 2), qrot_first_ = qr->second.tokens;
    kbuf_ = &Alloc(B * kKv * 4);
    vbuf_ = &Alloc(B * kKv * 4);
    raw_ = &Alloc(B * kInner * 4);
    // DeltaNet as two head ranges (dispatch.txt "dnsplit": heads [0, a) and [a, a + b)); the second writes raw2_
    if (const auto ds = geom_.find("dnsplit"); ds != geom_.end()) {
      dnsplit_a_ = ds->second.rowgrp, dnsplit_b_ = ds->second.tt;
      if (dnsplit_a_ + dnsplit_b_ != kTs || (dnsplit_a_ * B) % 8 || (dnsplit_b_ * B) % 8)
        throw LoomError("dispatch.txt dnsplit does not cover the DeltaNet heads");
      raw2_ = &Alloc(B * kInner * 4);
    }
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
    if (!a4_) gateffn_ = &Alloc(B * kFfn * 4);   // A4's gu GEMM keeps the gate in LDS
    // Per-workgroup weight staging (ABI only: the tile GEMMs stage in LDS) and epilogue scratch sized for the widest
    // GEMM, which the compiler declares over the whole [m_rows][tokens] tile.
    uwstage_ = &Alloc(std::size_t{kFfn} * 16 * 2);
    wstage_ = &Alloc(std::size_t{kFfn} * 16 * 2);
    ostage_ = &Alloc(std::size_t{kFfn} * B * 4);
    hidden2_ = &Alloc(B * kHidden * 4);
    normed_ = &Alloc(std::size_t{kHidden} * 4);
    valid_ = &Alloc(4);
    noans_ = &Alloc(4);
    ids_ = &Alloc(B * 4);
    host_ids_.resize(B);
    const float epsv = 1.0e-6f;
    gpu_.H2D(*eps_, &epsv, 4);
    Zero(*conv_state_);
    Zero(*state_);
    Zero(*kv16_);
    Zero(*reszero_);
    Zero(*noans_);

    // "kv_paged": 256-token pages, one page table per sequence (logical -> physical page, shared by all layers).
    kv_paged_ = geom_.count("kv_paged") != 0;
    pages_ = (T_ + 255) / 256;
    // "fpsparse <first chunk> <pages> <list stride>" (gen_fpsparse): block-sparse attention from that chunk on, on the
    // kv4 pools ("fpskv4"). They gain <pages> pages of pooled rows (block j = key pages_ * 256 + j), mapped to themselves
    // by the page table.
    if (const auto fp = geom_.find("fpsparse"); fp != geom_.end())
      fps_on_ = true, fps_first_ = fp->second.tokens, fps_ext_ = fp->second.rowgrp, fps_ls_ = fp->second.tt;
    if (fps_on_ && !geom_.count("fpskv4")) throw LoomError("fpsparse set without fpskv4 (re-emit the set)");
    fps_ord_ = fps_on_ && geom_.count("fpord");   // the attention's launch order (yah_fpord) after the lists in sel
    const std::uint32_t ptab_n = pages_ + fps_ext_;
    ptab_ = &Alloc(std::size_t{kv_paged_ ? ptab_n : 1} * 4);
    if (kv_paged_) {
      std::vector<std::int32_t> pages(ptab_n);
      for (std::uint32_t i = 0; i < ptab_n; ++i) pages[i] = static_cast<std::int32_t>(i);
      // Attention does not clamp page table entries: validate them here.
      for (std::int32_t pg : pages)
        if (pg < 0 || pg >= static_cast<std::int32_t>(ptab_n)) throw LoomError("page table entry out of range");
      gpu_.H2D(*ptab_, pages.data(), pages.size() * 4);
    }
    ptab_ref_ = {ptab_->handle, 0, std::size_t{kv_paged_ ? ptab_n : 1} * 4};
    // Unpaged set with a "vtrans.hal" row: attention reads V as [kv head][16-key tile][dim][16] f16.
    vtrans_ = !kv_paged_ && geom_.count("vtrans.hal");
    vt_bytes_ = std::size_t{(T_ + 15) / 16 * 16} * kKvRow * 2;
    vt16_ = &Alloc(vtrans_ ? vt_bytes_ : 4);
    // Quantized K: "attn_kq8" int8 (kv8a16), "attn_kq4" H256 + asymmetric int4 (kv4a16; kernel yah_kq4 in kq8.hal).
    // Per attention layer: codes [T][1024 or 512 B], scales [T][8 or 32] dwords, the channel mean of the first chunk.
    attn_kq4_ = geom_.count("attn_kq4") != 0;
    attn_kq8_ = geom_.count("attn_kq8") != 0 || attn_kq4_;
    // rows per layer: the context, plus the pooled rows of a fpsparse set
    const std::size_t qrows = std::size_t{T_} + std::size_t{fps_ext_} * 256;
    ks_bytes_ = qrows * (attn_kq4_ ? 32 : 8) * 4;
    kq_bytes_ = qrows * (attn_kq4_ ? 512 : 1024);
    kq8buf_ = &Alloc(attn_kq8_ ? std::size_t{full_} * kq_bytes_ : 4);
    ksbuf_ = &Alloc(attn_kq8_ ? std::size_t{full_} * ks_bytes_ : 4);
    kmbuf_ = &Alloc(attn_kq8_ ? std::size_t{full_} * 4096 : 4);
    // Quantized V^T: "attn_vq8" bytes / "attn_vq4" nibbles per channel per 16-key tile, plus (S, C') per tile.
    attn_vq4_ = geom_.count("attn_vq4") != 0;
    attn_vq8_ = geom_.count("attn_vq8") != 0;
    attn_vqt_ = attn_vq4_ || attn_vq8_;
    // (fpsparse: the pooled rows' tiles after the 4 KV heads' context tiles)
    const std::size_t vrows = std::size_t{(T_ + 15) / 16 * 16} + std::size_t{fps_ext_} * 256;
    vq_bytes_ = vrows * (attn_vq8_ ? 1024 : 512);
    vqs_bytes_ = vrows * 256;
    vqbuf_ = &Alloc(attn_vqt_ ? std::size_t{full_} * vq_bytes_ : 4);
    vqsbuf_ = &Alloc(attn_vqt_ ? std::size_t{full_} * vqs_bytes_ : 4);
    // Paged fp16 pools: per layer, K rows / V^T tiles of the whole context.
    paged_f16k_ = kv_paged_ && !attn_kq8_;
    paged_f16v_ = kv_paged_ && !attn_vqt_;
    // "rope_kpaged": RoPE writes the fp16 K rows straight into the paged pool.
    rope_kpaged_ = geom_.count("rope_kpaged") != 0;
    qnorm_ = geom_.count("qnorm") != 0;
    qg16_ = geom_.count("qg16") != 0;
    vtrope_ = geom_.count("vtrope") != 0;
    if (vtrope_ && !(rope_kpaged_ && paged_f16v_)) throw LoomError("vtrope needs the paged fp16 K and V pools");
    if (paged_f16k_ && !rope_kpaged_) throw LoomError("paged fp16 K needs a rope_kpaged HAL set (re-emit)");
    pool_bytes_ = std::size_t{pages_} * 256 * kKvRow * 2;
    kpool_ = &Alloc(paged_f16k_ ? std::size_t{full_} * pool_bytes_ : 4);
    vtpool_ = &Alloc(paged_f16v_ ? std::size_t{full_} * pool_bytes_ : 4);
    // kv4 / kv8 sets (dispatch.txt "kvdeq"): yah_kdeq / yah_vdeq decode the layer's quantized pools into one f16 pool pair (reused
    // layer to layer) and the attention runs the fp16 kernel on it, as llama.cpp's prefill flash attention does
    if (geom_.count("kvdeq")) {
      if (!attn_kq8_ || !attn_vqt_ || !kv_paged_) throw LoomError("dispatch.txt kvdeq needs the paged kv4 pools");
      deq_bytes_ = std::size_t{pages_ + fps_ext_} * 256 * kKvRow * 2;
      kdeq_ = &Alloc(deq_bytes_);
      vdeq_ = &Alloc(deq_bytes_);
    }
    if (fps_on_) {
      if (!kdeq_ || B_ % 128 || !qr16_) throw LoomError("fpsparse needs a kv4 kvdeq set with 128-token blocks");
      fps_blocks_ = pages_ * 2;
      fps_mean_bytes_ = std::size_t{fps_blocks_} * kKvRow * 2;
      kfm_ = &Alloc(std::size_t{full_} * fps_mean_bytes_);
      vfm_ = &Alloc(std::size_t{full_} * fps_mean_bytes_);
      fsel_ = &Alloc(std::size_t{B_ / 16} * 4 * (fps_ls_ + (fps_ord_ ? 1 : 0)) * 4);
    }
  }

  void LoadExecutables() {
    // The attention HAL stores f16 straight into the o-projection input ("attn_f16out").
    if (!geom_.count("attn_f16out")) throw LoomError("dispatch.txt lacks attn_f16out (re-emit the set)");
    const auto dn = geom_.find("rowsplit.hal");
    if (dn == geom_.end() || !dn->second.rowgrp)
      throw LoomError("dispatch.txt has no rowsplit.hal row group (re-emit the set)");
    dn_rowgrp_ = dn->second.rowgrp;
    // The DeltaNet kernel's workgroup size (gen_gdn_ss runs 9 waves); sets without the row are 256.
    if (const auto t = geom_.find("dnthreads"); t != geom_.end()) dn_threads_ = t->second.tokens;
    const auto at = geom_.find("wmma.hal");
    if (at == geom_.end() || !at->second.rowgrp || !at->second.tokens)
      throw LoomError("dispatch.txt has no wmma.hal geometry (re-emit the set)");
    attn_hpw_ = at->second.rowgrp;
    attn_tpw_ = at->second.tokens;
    if (kHeads % attn_hpw_) throw LoomError("wmma.hal heads per workgroup does not divide the heads");
    // Loom drops bounds clamps it proves from the launch contract: extra workgroups read unmapped VA and hang.
    if (at->second.tt && (B_ + attn_tpw_ - 1) / attn_tpw_ != at->second.tt)
      throw LoomError("wmma.hal: grid x does not match the emitted token tiles");
    // chunks whose attention has a workgroup shape of its own (the KT2 twin) carry their own row
    for (const auto& [name, g] : geom_)
      if (name.rfind("wmma", 0) == 0 && name.find("_c") != std::string::npos &&
          (!g.rowgrp || !g.tokens || kHeads % g.rowgrp || (B_ + g.tokens - 1) / g.tokens != g.tt))
        throw LoomError(name + ": its launch geometry does not fit (re-emit the set)");
    for (const char* hal : {"norm.hal", "convkq.hal", "prepab.hal", "rowsplit.hal", "postnorm.hal", "gemv.hal",
                            "rmsnorm.hal", "argmax.hal", "embed.hal"})
      Exe(hal);
    GeomOf("embed.hal");   // one token per workgroup: refuses a set emitted for another chunk size
    // embed.hal decodes Q8_0
    if (Find("token_embd.weight")->type != core::GgmlType::kQ8_0)
      throw LoomError("prefill: token_embd: its format does not match embed.hal (re-emit the set)");
  }

  // Load every HAL chunk 0 dispatches now (a pass of the layers that dispatches nothing): a first load copies the code to
  // the GPU, ~0.25 ms per HAL, which RunLayers would otherwise spend before the GPU starts.
  void WarmExecutables() {
    LoomBuffer *h = hidden_, *h2 = hidden2_, *aq = aq_, *aq2 = aq2_, *ad = ad_, *ad2 = ad2_;
    n_ = B_, warming_ = true;
    RecordLayers(0, {});
    FlushKres();
    n_ = 0, warming_ = false;
    hidden_ = h, hidden2_ = h2, aq_ = aq, aq2_ = aq2, ad_ = ad, ad2_ = ad2;
  }

  // HAL name "<kind>_mq4_<m_tiles>_<k_blocks>.hal" of the tile GEMM on tensor t.
  std::string GemmHal(const char* kind, const core::TensorInfo& t) const {
    if (t.type != core::GgmlType::kMQ4G256) throw LoomError(std::string("prefill: ") + kind + ": the weights are not MQ4");
    return std::string(kind) + "_mq4_" + std::to_string(t.dims[1] / 16) + "_" + std::to_string(t.dims[0] / 256) + ".hal";
  }
  std::uint32_t MTiles(const core::TensorInfo& t) const { return static_cast<std::uint32_t>(t.dims[1] / 16); }
  // The GEMM variant for this chunk's real tokens: the set may carry "<hal>.t128.hal" / ".t64.hal" with narrower token
  // tiles. Pick the least (padded token rows x cost per row); a narrow tile pads less but costs more per row (measured:
  // x1.12 at 128 tokens, x1.6 at 64). Every variant computes the same values.
  std::string PickGemm(const std::string& hal) {
    const std::string stem = hal.substr(0, hal.size() - 4);
    std::string best = hal;
    double best_cost = 1e30;
    for (const std::string& h : {hal, stem + ".t128.hal", stem + ".t64.hal"}) {
      const auto it = geom_.find(h);
      if (it == geom_.end()) continue;
      const std::uint32_t bn = it->second.tokens;
      const double weight = bn >= 256 ? 1.0 : bn >= 128 ? 1.12 : 1.6;
      const double cost = static_cast<double>((n_ + bn - 1) / bn * bn) * weight;
      if (cost < best_cost) best = h, best_cost = cost;
    }
    return best;
  }

  // The FFN: norm, gate, swiglu (up), down.
  void RunFfn(const std::string& pre) {
    if (RunFfnBlock(pre)) return;
    nq_want_ = nqgu_;
    RunNorm(pre + "post_attention_norm.weight");
    nq_want_ = false;
    if (a4_) {
      // one GEMM for gate and up: silu(gate) * up straight into ffnup_ (gen_gemm_iu4 gu); guov: flagged (a counter per
      // 1024 features and token tile), RunResidual's rotq waits on it
      const auto* up = Find(pre + "ffn_up.weight");
      if (guov_) gu_cnt_ = FlagSlots(kFfn / 1024);
      gu_flagged_ = RunIu4("gu", *Find(pre + "ffn_gate.weight"), {ffnup_}, {std::size_t(up->dims[1]) * 2}, 0b10000, up,
                           guov_ ? &gu_cnt_ : nullptr, &nq_cnt_);
      gu_waited_ = gu_flagged_ && nq_cnt_.buffer;
      nq_cnt_ = {};
    } else {
      RunKstore(pre + "ffn_gate.weight", *gateffn_);
      RunSwiglu(pre + "ffn_up.weight");
    }
    RunResidual(pre + "ffn_down.weight", *ffnup_);
  }

  // The NPU FFN block (npu_plan). npu.txt: "npuffn <F> <B> <K parts>", "npubytes_<image> <a> <w> <c>" (binding bytes),
  // "npudgu <bytes from a gate row to its up row>", "npugate <supply> <calls> <record>" and the GPU kernels' rows.
  static constexpr std::uint32_t kDownCalls = 10, kDownRows = 512;   // down calls per K part, rows per call
  static constexpr std::size_t kHRec = 4 * 128 * 144;   // H bytes per (M block, pass): [slab 4][k-block 128][144]
  struct NpuBytes {
    std::size_t a, w, c;
  };
  NpuBytes NpuBytesOf(const std::string& name) const {
    const Geom& g = geom_.at(name);
    return {g.tokens, g.rowgrp, g.tt};
  }
  std::uint32_t FfnF() const {
    const auto it = geom_.find("npuffn");
    return it == geom_.end() ? 0 : it->second.tokens;
  }
  // (first pass, passes) of down's K parts over F features: the fewest parts of at most 5 passes, sizes balanced
  // (emit_npu_ffn.parts)
  std::vector<std::pair<std::uint32_t, std::uint32_t>> FfnParts() const;
  static std::string DnImage(std::uint32_t passes, const char* pre = "") { return pre + std::string("ffndn") + std::to_string(passes); }
  // H's bytes in C (from 0), then down's partials from FfnDownOff().
  std::size_t FfnDownOff() const { return (std::size_t{B_ / 64} * (FfnF() / 1024) * kHRec + 4095) / 4096 * 4096; }
  std::size_t DownC() const { return NpuBytesOf(DnImage(FfnParts()[0].second, "npubytes_")).c; }
  std::size_t DownRowBytes() const { return std::size_t{FfnF()} / 256 * 136; }   // a down pack row (IQ4_XS)
  static constexpr std::size_t kRecordRow = 576;   // a decoder record's row bytes (gen_npu_dec.RECROW)

  bool RunFfnBlock(const std::string& pre);
  // The selected chunk's pending FFN block: wait for its last NPU job, then hidden = hidden2 + the NPU's partials.
  void JoinFfn();
  std::size_t NpuCBytes() const { return FfnDownOff() + FfnParts().size() * kDownCalls * DownC(); }
  // An iu4 GEMM of the block at its own row count (RunIu4 names the HAL from the tensor's): grid (G, token tiles,
  // row tiles / G) with its iu4grp row, else (row tiles, token tiles).
  void DispatchIu4(const std::string& hal, const char* sfx, std::uint32_t mtiles,
                   const std::vector<hrx_buffer_ref_t>& b, std::uint64_t writes);
  // NPU handoffs are graph nodes on flag words in host memory (NpuSplit::Flags), so the host never splits the chunk.
  // NpuEnqueue: the graph stores the job's gate value to its ready word once a is written and every earlier reader of
  // c is done (release, system scope: a reaches memory first); the NPU waits for it itself. NpuJoin: yah_npu_flag_wait
  // polls the last job's done word; it stands for the NPU's reads of A and writes of C, so what reads C (the unpack) or
  // writes A (the next layer's encoder) orders after it. The jobs are queued right after the chunk graph launches.
  hrx_buffer_ref_t FlagWord(std::uint32_t w) const { return {npu_->Flags()->handle, std::size_t{w} * 4, 4}; }
  void NpuEnqueue(std::vector<std::uint32_t> calls, std::string tag, const hrx_buffer_ref_t& a,
                  const hrx_buffer_ref_t& c);
  void NpuCpuDriven(const std::vector<NpuSplit::Queued>& jobs);
  void NpuJoin(std::size_t job);

  // The RMSNorm of hidden() rows into scratch_ (f16), rotated into the weights' basis (A4: and quantized into aq_ / ad_).
  void RunNorm(const std::string& wname) {
    // A row takes `split` waves (dispatch.txt norm_split, else 1); a workgroup of w waves takes w / split rows.
    const LoomExecutable& exe = Exe("norm.hal");
    const std::uint32_t ws = exe.WorkgroupSize(exe.OrdinalOrZero("yah_half_norm"));
    const auto ns = geom_.find("norm_split");
    const std::uint32_t split = ns != geom_.end() && ns->second.rowgrp ? ns->second.rowgrp : 1;
    const std::uint32_t rows_per_wg = ws ? ws / 32 / split : 1;
    if (!rows_per_wg || (ws && ws % (32 * split))) throw LoomError("norm.hal: workgroup size does not match norm_split");
    if (B_ % rows_per_wg) throw LoomError("norm.hal: rows per workgroup must divide the chunk");
    if (a4_ && geom_.count("normq_5120.hal")) {
      // A4: the norm and its rotq in one kernel (gen_rot.gen_nq), one row per workgroup; the f16 norm output is not kept.
      // nqov: right after its kres (deferred by RunResidual, untrimmed) as kresf + normqf with no graph edge between
      // them: the rows wait on the kres's per-token-tile counters (FlagSlot), so they start in the GEMM's tail. normqf
      // writes the other activation pair (the kres still reads aq_ / ad_), which becomes aq_ / ad_.
      // nq_want_ (nqgu: the FFN norm, nqw: the attention norm): the normq also counts its finished rows in nq_cnt_ for
      // the GEMMs that wait on them
      const bool want = std::exchange(nq_want_, false);
      nq_cnt_ = {};
      if (const PendingKres p = pending_kres_; p.t && p.out == hidden_) {
        pending_kres_ = {};
        const hrx_buffer_ref_t cnt = FlagSlot();
        const std::int64_t trim = std::exchange(trim_row_, p.trim);
        const bool flagged = RunIu4("kres", *p.t, {p.in, p.out}, {std::size_t(p.t->dims[1]) * 4, std::size_t(p.t->dims[1]) * 4},
                                    0b10000, nullptr, &cnt, &p.wcnt, p.chain ? 0b11000 : 0);
        trim_row_ = trim;
        if (flagged) {
          std::vector<hrx_buffer_ref_t> b = {Ref(*hidden_), TRef(*Find(wname)), TRef(*Find(NormAwq(wname))),
                                             Ref(*rot_signs_), Ref(*aq2_), Ref(*ad2_), cnt};
          if (want) b.push_back(nq_cnt_ = FlagSlot());
          // a chained kres (rqk): aq2_ / ad2_'s readers since the last barrier (the gu) finished each token tile
          // before the rotq, the kres and so these rows: untracked too
          const std::uint64_t pair = p.chain && p.wcnt.buffer ? 0b110000 : 0;
          Dispatch(Exe(want ? "normqfo_5120.hal" : "normqf_5120.hal"), "yah_normq", want && nqp_ ? nqp_ : B_, 1, 1,
                   geom_.at("normq_5120.hal").rowgrp, 1, 1, b, want ? 0b10110000 : 0b0110000, nullptr,
                   (want ? 0b11000001 : 0b1000001) | pair);
          std::swap(aq_, aq2_), std::swap(ad_, ad2_);
          return;
        }
      }
      FlushKres();
      std::vector<hrx_buffer_ref_t> b = {Ref(*hidden_), TRef(*Find(wname)), TRef(*Find(NormAwq(wname))),
                                         Ref(*rot_signs_), Ref(*aq_), Ref(*ad_)};
      if (want) b.push_back(nq_cnt_ = FlagSlot());
      Dispatch(Exe(want ? "normqo_5120.hal" : "normq_5120.hal"), "yah_normq", want && nqp_ ? nqp_ : B_, 1, 1,
               geom_.at("normq_5120.hal").rowgrp, 1, 1, b, want ? 0b1110000 : 0b110000, nullptr, want ? 0b1000000 : 0);
      return;
    }
    Dispatch(exe, "yah_half_norm", B_ / rows_per_wg, 1, 1, 32, 1, 1,
             {Ref(*hidden_), Ref(*reszero_), TRef(*Find(wname)), Ref(*sumout_), Ref(*scratch_)});
    if (a4_)
      RotQ(*scratch_, kHidden, NormAwq(wname), false);
    else
      Rotate(*scratch_, nullptr, kHidden, NormAwq(wname));
  }
  // the awq of a norm's input site: the post-attention norm feeds the FFN, attn_norm qkv / gate / alpha / beta or q / k / v
  std::string NormAwq(const std::string& wname) const {
    const std::string pre = wname.substr(0, wname.find('.', 4) + 1);
    const char* w = wname.find("post_attention_norm") != std::string::npos   ? "ffn_gate"
                    : cfg_.IsFullAttention(std::stoul(wname.substr(4))) ? "attn_q"
                                                                        : "attn_qkv";
    return pre + w + ".awq";
  }
  // MQ4 sets: x (B_ rows of k f16) into the weights' basis (gen_rot): in place, or dst != nullptr (k = kInner, ssm_out):
  // the DeltaNet value heads from GGUF into HF order into dst.
  void Rotate(const LoomBuffer& x, const LoomBuffer* dst, std::uint32_t k, const std::string& awq) {
    std::vector<hrx_buffer_ref_t> b = {Ref(x), TRef(*Find(awq)), Ref(*rot_signs_)};
    if (dst) b.push_back(Ref(*dst));
    Dispatch(Exe(dst ? std::string("rot_heads.hal") : "rot_" + std::to_string(k) + ".hal"), "yah_rot256", k / 1024, B_,
             1, 64, 1, 1, b);
  }
  // A4 sets: x (B_ rows of k f16) rotated and quantized into aq_ / ad_ (gen_rot.gen_q, one scale per 128 values);
  // heads: ssm_out's value heads from GGUF into HF order.
  // ocnt (rqk): counters for the kres after it (PendingKres::wcnt) when this rotq has a counting variant
  // (rotqfo_17408, rotqo_6144), else left null. Returns whether it ran flagged after a waiting gu (PendingKres::chain).
  bool RotQ(const LoomBuffer& x, std::uint32_t k, const std::string& awq, bool heads, hrx_buffer_ref_t* ocnt = nullptr) {
    const std::string hal = heads ? std::string("rotq_heads.hal") : "rotq_" + std::to_string(k) + ".hal";
    const Geom& g = geom_.at(hal);
    if (std::exchange(gu_flagged_, false)) {
      // the flagged gu's output, x untracked: the workgroups wait on its counters, so they run in the GEMM's tail; the
      // other activation pair (the gu still reads aq_ / ad_) becomes aq_ / ad_
      // gu_waited_: the gu waited on its normq's counters, so no barrier since the kres that read aq2_ / ad2_ (nqov); a
      // row's token tile of that kres finished before its normq rows, gu tiles and so these workgroups: untracked too
      if (&x != ffnup_ || k != kFfn || heads) throw LoomError("rotqf_17408.hal: not the flagged gu's output");
      const bool chain = std::exchange(gu_waited_, false), counts = ocnt != nullptr;
      std::vector<hrx_buffer_ref_t> b = {Ref(x), TRef(*Find(awq)), Ref(*rot_signs_), Ref(*aq2_), Ref(*ad2_), gu_cnt_};
      if (counts) b.push_back(*ocnt = FlagSlot());
      // rqp: as persistent workgroups (rotqp_17408), so the kres after it is not queued behind 34816 rotq workgroups
      const bool persist = counts && rqp_;
      Dispatch(Exe(persist ? "rotqp_17408.hal" : counts ? "rotqfo_17408.hal" : "rotqf_17408.hal"), "yah_rotq256",
               persist ? geom_.at("rotqp_17408.hal").tokens : g.tt, persist ? 1 : B_, 1, g.rowgrp, 1, 1, b,
               counts ? 0b1011000 : 0b11000, nullptr, (chain ? 0b111001 : 0b100001) | (counts ? 0b1000000 : 0));
      std::swap(aq_, aq2_), std::swap(ad_, ad2_);
      return chain && counts;
    }
    const std::string ohal = "rotqo_" + std::to_string(k) + ".hal";
    const bool counts = ocnt && !heads && k == 6144;
    std::vector<hrx_buffer_ref_t> b = {Ref(x), TRef(*Find(awq)), Ref(*rot_signs_), Ref(*aq_), Ref(*ad_)};
    if (counts) b.push_back(*ocnt = FlagSlot());
    // rotqop_6144: as persistent workgroups, so the kres after it is not queued behind the rotq's grid
    const std::string phal = "rotqop_" + ohal.substr(6);
    // the flagged attention's output (x untracked): the persistent workgroups wait on its counters
    if (const auto it = geom_.find("rotqopf_6144.hal"); std::exchange(attn_flagged_, false) && counts && &x == scratch_ &&
                                                         it != geom_.end()) {
      b.insert(b.begin() + 5, attn_cnt_);
      Dispatch(Exe("rotqopf_6144.hal"), "yah_rotq256", it->second.tokens, 1, 1, g.rowgrp, 1, 1, b, 0b1011000, nullptr,
               0b1100001);
      return false;
    }
    if (const auto it = geom_.find(phal); counts && it != geom_.end()) {
      Dispatch(Exe(phal), "yah_rotq256", it->second.tokens, 1, 1, g.rowgrp, 1, 1, b, 0b111000, nullptr, 0b100000);
      return false;
    }
    Dispatch(Exe(counts ? ohal : hal), "yah_rotq256", g.tt, B_, 1, g.rowgrp, 1, 1, b, counts ? 0b111000 : 0b11000,
             nullptr, counts ? 0b100000 : 0);
    return false;
  }
  // A4 sets: the iu4 GEMM (gen_gemm_iu4) of kind on t over aq_ / ad_; io: the kind's bindings after them (outputs at
  // the bits of writes), row_bytes: their bytes per token.
  // t2: a second weight tensor bound after the activations (gu: up beside gate).
  // cnt (kres, kstore, kstoreh): the flagged GEMM (iu4_<kind>f_*: per-token-tile completion counters in cnt, a FlagSlot,
  // untracked) when the chunk is untrimmed; returns whether it ran flagged.
  // wcnt: the activations' producer's counters (nq_cnt_, a rotq's): untrimmed, the GEMM also waits on them
  // (iu4_<kind>[f]w_*), aq_ / ad_ and wcnt untracked, so no graph barrier separates it from the producer; then also the
  // bindings at the bits of untrack, which the producer's own waits order per token tile.
  bool RunIu4(const char* kind, const core::TensorInfo& t, std::initializer_list<const LoomBuffer*> io,
              std::initializer_list<std::size_t> row_bytes, std::uint64_t writes, const core::TensorInfo* t2 = nullptr,
              const hrx_buffer_ref_t* cnt = nullptr, const hrx_buffer_ref_t* wcnt = nullptr, std::uint64_t untrack = 0) {
    const std::size_t K = t.dims[0];
    const std::string hal = std::string("iu4_") + kind + "_" + std::to_string(MTiles(t)) + "_" + std::to_string(K / 256) + ".hal";
    const Geom g = GeomOf(hal);
    std::vector<hrx_buffer_ref_t> b = {TRef(t), Ref(*aq_), Ref(*ad_)};
    std::vector<std::size_t> rb = {K / 2, K / 128 * 4};
    if (t2) b.push_back(TRef(*t2)), rb.push_back(0);
    for (const LoomBuffer* x : io) b.push_back(Ref(*x));
    rb.insert(rb.end(), row_bytes);
    const std::uint32_t tt = Trim(b, 1, rb, g);
    const char* sfx = std::strcmp(kind, "kres") == 0 ? "_kres" : std::strcmp(kind, "kqg") == 0 ? "_kqg" : std::strcmp(kind, "gu") == 0 ? "_gu" : "";
    // dispatch.txt "iu4grp_<hal> G": grouped grid (G, token tiles, row tiles / G), x fastest = bands of G weight tiles
    std::uint32_t gx = MTiles(t) / g.rowgrp, gz = 1;
    if (const auto it = geom_.find("iu4grp_" + hal); it != geom_.end()) {
      if (it->second.tokens == 0 || gx % it->second.tokens != 0) throw LoomError(hal + ": iu4grp does not divide the row tiles");
      gz = gx / it->second.tokens, gx = it->second.tokens;
    }
    const bool flagged = cnt && tt == g.tt;
    const bool waits = wcnt && wcnt->buffer && tt == g.tt;
    std::uint64_t untracked = 0;
    if (flagged) {
      b.push_back(*cnt);
      writes |= std::uint64_t{1} << (b.size() - 1), untracked |= std::uint64_t{1} << (b.size() - 1);
    }
    if (waits) b.push_back(*wcnt), untracked |= std::uint64_t{1} << (b.size() - 1) | 0b110 | untrack;
    const std::string exe = flagged || waits ? std::string("iu4_") + kind + (flagged ? "f" : "") + (waits ? "w" : "") +
                                                   hal.substr(4 + std::strlen(kind))
                                             : hal;
    Dispatch(Exe(exe), (std::string("yah_iu4_gemm") + sfx).c_str(), gx, tt, gz, 32, 1, 1, b, writes, nullptr, untracked);
    return flagged;
  }
  // Completion counters of a flagged producer and its flag-waiting consumer (gen_gemm_iu4 flag, gen_rot.flag_wait): a
  // word per 128-token tile, a slot per pair in the chunk, zeroed before the chunk (RunLayers; a pair: both chunks). Both bind it untracked.
  static constexpr std::size_t kFlagSlots = 4096;
  std::size_t TokenTiles128() const { return (B_ + 127) / 128; }
  hrx_buffer_ref_t FlagSlot() { return FlagSlots(1); }
  // k consecutive slots (a flagged gu: a counter per 1024 features and token tile)
  hrx_buffer_ref_t FlagSlots(std::size_t k) {
    if (flag_slot_ + k > kFlagSlots) throw LoomError("prefill: out of completion-counter slots");
    const std::size_t n = TokenTiles128() * 4;
    flag_slot_ += k;
    return {flags_->handle, (flag_slot_ - k) * n, k * n};
  }
  // A kres RunResidual deferred (nqov): RunNorm's normq takes it as the flagged pair, anything else runs it plain first.
  // trim: trim_row_ when it was deferred (the GEMM runs as it would have then).
  struct PendingKres {
    const core::TensorInfo* t = nullptr;
    const LoomBuffer* in = nullptr;
    const LoomBuffer* out = nullptr;
    std::int64_t trim = kAllRows;
    // rqk: its rotq's per-token-tile counters (wcnt.buffer null: none); chain: that rotq waited on a gu that waited on
    // its normq (RotQ), so the kres's residual input and output are ordered per token tile too
    hrx_buffer_ref_t wcnt{};
    bool chain = false;
  };
  PendingKres pending_kres_;
  void FlushKres() {
    if (!pending_kres_.t) return;
    const PendingKres p = pending_kres_;
    pending_kres_ = {};
    const std::int64_t trim = std::exchange(trim_row_, p.trim);
    RunIu4("kres", *p.t, {p.in, p.out}, {std::size_t(p.t->dims[1]) * 4, std::size_t(p.t->dims[1]) * 4}, 0b10000);
    trim_row_ = trim;
  }

  // The chunk's graph: the whole chunk (after layer 0's, SplitGraph) is one graph.
  // Replacing a launched graph is safe: the queue retains its command buffers until they complete, and every buffer and
  // executable they use is this object's.
  void NewGraph() {
    chunk_graph_ = std::make_unique<LoomGraph>(gpu_);
    LoomGraph& g = *chunk_graph_;
    weights_.ForEachBuffer([&](hrx_buffer_t h) { g.ReadOnly(h); });
    for (const LoomBuffer* t : {eps_, reszero_})
      g.ReadOnly(t->handle);
    graph_ = &g;
  }
  // Launch the graph recorded so far and record the rest of the chunk into a new one, so the GPU starts while the host
  // records. Stream order runs the new graph after the whole old one: no dependencies cross.
  void SplitGraph() {
    split_armed_ = false;
    chunk_graph_->Launch();
    NewGraph();
  }

  // The tile GEMMs (W4A16) read scratch_; A4 runs the iu4 GEMMs on aq_ / ad_ (RunIu4).
  // cnt (A4): run the flagged GEMM on these counters if it can (RunIu4); returns whether it did.
  bool RunKstore(const std::string& wname, const LoomBuffer& out, const hrx_buffer_ref_t* cnt = nullptr) {
    const auto* t = Find(wname);
    if (a4_) {
      // dnperm sets: the DeltaNet gate in HF head order (kstoreh), as the pnq kernels read it
      const bool hp = geom_.count("dnperm") && wname.ends_with("attn_gate.weight");
      if (hp && !geom_.count("iu4_kstoreh_" + std::to_string(MTiles(*t)) + "_" + std::to_string(t->dims[0] / 256) + ".hal"))
        throw LoomError("dnperm sets need the kstoreh gate GEMM");
      return RunIu4(hp ? "kstoreh" : "kstore", *t, {&out}, {std::size_t(t->dims[1]) * 4}, 0b1000, nullptr, cnt, &iu4_wait_,
                    iu4_untrack_);
    }
    const std::string hal = PickGemm(GemmHal("gemm_kstore", *t));
    const Geom g = GeomOf(hal);
    std::vector<hrx_buffer_ref_t> b = {TRef(*t), Ref(*scratch_), Ref(*wstage_), Ref(*ostage_), Ref(out)};
    const std::uint32_t tt = Trim(b, 1, {std::size_t(t->dims[0]) * 2, 0, 0, std::size_t(t->dims[1]) * 4}, g);
    Dispatch(Exe(hal), "yah_ffn_gemm_mq4", MTiles(*t) / g.rowgrp, tt, 1, 32, 1, 1, b, GemmWrites(b, {&out}));
    return false;
  }
  // The attention q projection with the q / gate unpack fused in (rows = heads x [256 q | 256 gate]).
  void RunKqg(const std::string& wname) {
    const auto* t = Find(wname);
    if (a4_) {
      RunIu4("kqg", *t, {q_, gate_}, {std::size_t(t->dims[1]) * 2, std::size_t(t->dims[1]) * 2}, 0b11000, nullptr, nullptr,
             &iu4_wait_, iu4_untrack_);
      return;
    }
    const std::string hal = PickGemm(GemmHal("gemm_kqg", *t));
    const Geom g = GeomOf(hal);
    const std::vector<hrx_buffer_ref_t> b = {TRef(*t), Ref(*scratch_), Ref(*wstage_), Ref(*ostage_), Ref(*q_), Ref(*gate_)};
    Dispatch(Exe(hal), "yah_ffn_gemm_mq4_kqg", MTiles(*t) / g.rowgrp, TokenTiles(g), 1, 32, 1, 1, b,
             GemmWrites(b, {q_, gate_}));
  }
  void RunSwiglu(const std::string& wname) {
    const auto* t = Find(wname);
    const std::string hal = PickGemm(GemmHal("gemm_swiglu", *t));
    const Geom g = GeomOf(hal);
    std::vector<hrx_buffer_ref_t> b = {TRef(*t), Ref(*scratch_), Ref(*gateffn_), Ref(*uwstage_), Ref(*ostage_), Ref(*ffnup_)};
    const std::size_t M = t->dims[1];
    const std::uint32_t tt = Trim(b, 1, {std::size_t(t->dims[0]) * 2, M * 4, 0, 0, M * 2}, g);
    Dispatch(Exe(hal), "yah_ffn_gemm_mq4_swiglu", MTiles(*t) / g.rowgrp, tt, 1, 32, 1, 1, b, GemmWrites(b, {ffnup_}));
  }
  // hidden += W input: the kres GEMM writes hidden + W input into hidden2, then the two hidden buffers swap.
  // quantized (A4): aq_ / ad_ already hold the rotated input (gen_pnq), in is not read
  // qcnt (quantized, dnring): the pnq's per-token-tile counters, which the kres waits on (a chained PendingKres)
  void RunResidual(const std::string& wname, const LoomBuffer& in, bool quantized = false,
                   const hrx_buffer_ref_t* qcnt = nullptr) {
    const auto* t = Find(wname);
    const bool heads = wname.find("ssm_out") != std::string::npos;
    const std::string awq = wname.substr(0, wname.size() - 6) + "awq";
    if (a4_) {
      hrx_buffer_ref_t rc = qcnt ? *qcnt : hrx_buffer_ref_t{};
      const bool chain =
          qcnt ? true : !quantized && RotQ(in, static_cast<std::uint32_t>(t->dims[0]), awq, heads, rqk_ ? &rc : nullptr);
      // nqov: the next A4 normq takes it as the flagged pair (RunNorm); any other dispatch flushes it first
      pending_kres_ = {t, hidden_, hidden2_, trim_row_, rc, chain};
      if (!nqov_) FlushKres();
      std::swap(hidden_, hidden2_);
      return;
    }
    Rotate(in, heads ? rotbuf_ : nullptr, static_cast<std::uint32_t>(t->dims[0]), awq);
    const std::string hal = PickGemm(GemmHal("gemm_kres", *t));
    const Geom g = GeomOf(hal);
    std::vector<hrx_buffer_ref_t> b = {TRef(*t), Ref(heads ? *rotbuf_ : in), Ref(*hidden_), Ref(*wstage_), Ref(*ostage_),
                                       Ref(*hidden2_)};
    const std::size_t M = t->dims[1];
    const std::uint32_t tt = Trim(b, 1, {std::size_t(t->dims[0]) * 2, M * 4, 0, 0, M * 4}, g);
    Dispatch(Exe(hal), "yah_ffn_gemm_mq4_kres", MTiles(*t) / g.rowgrp, tt, 1, 32, 1, 1, b, GemmWrites(b, {hidden2_}));
    std::swap(hidden_, hidden2_);
  }

  void RunAttention(std::uint32_t l, std::uint32_t ci, const std::string& pre, const KvHook& hook) {
    const std::uint32_t ai = l / cfg_.full_attention_interval;
    if (ai >= full_) throw LoomError("full-attention layer index past the KV slot count");
    // nqw: all three wait on the norm's counters (iu4_wait_); after a chained DeltaNet tail (dn_chained_) gate_'s
    // reader since the last barrier (that tail's pnq, row by row) finished each token tile before it reaches kqg:
    // kqg's outputs untracked too
    iu4_wait_ = std::exchange(nq_cnt_, {});
    iu4_untrack_ = std::exchange(dn_chained_, false) && iu4_wait_.buffer ? 0b11000 : 0;
    RunKqg(pre + "attn_q.weight");
    iu4_untrack_ = 0;
    // ropew: k / v flagged (per-token-tile counters), RoPE polls them instead of a graph barrier. Not when RoPE ends
    // the chunk's graph (tail kNoRows): the next chunk's counter fill is not ordered after its polls
    const bool rw = ropew_ && qnorm_ && n_ == B_ && tail_ != kNoRows;
    if (rw) kv_cnt_[0] = FlagSlot(), kv_cnt_[1] = FlagSlot();
    kv_flagged_ = RunKstore(pre + "attn_k.weight", *kbuf_, rw ? &kv_cnt_[0] : nullptr);
    kv_flagged_ = RunKstore(pre + "attn_v.weight", *vbuf_, rw ? &kv_cnt_[1] : nullptr) && kv_flagged_;
    iu4_wait_ = {}, iu4_untrack_ = 0;
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
                                         Ref(*q16_),
                                         Ref(*kbuf_),
                                         Ref(*kc32_),
                                         Ref(*vc32_),
                                         rope_kpaged_ ? kpool_l : hrx_buffer_ref_t{kv16_->handle, koff, kv16_layer_},
                                         vtrope_ ? vtpool_l : hrx_buffer_ref_t{kv16_->handle, voff, kv16_layer_},
                                         {eps_->handle, 0, 4}};
      if (rope_kpaged_) b.push_back(ptab_ref_);
      // ropew: the k-only RoPE reads k / v (untracked: it waits on their counters) and writes only the pools; its q /
      // q_out / k_out bindings are unused
      const bool rw = std::exchange(kv_flagged_, false) && rope_kpaged_ && vtrope_;
      if (rw) b.push_back(kv_cnt_[0]), b.push_back(kv_cnt_[1]);
      Dispatch(ChunkExe(rw ? "ropew" : "rope", ci), "yah_fused_qk_rope_batched",
               ropekv_ ? 3 * B_ / 4 : qnorm_ ? kKvHeads : kHeads + kKvHeads, ropekv_ ? 1 : B_, 1, 256, 1, 1, b,
               rw ? 0b11000000000 : ~std::uint64_t{0}, nullptr, rw ? 0b110000001100111 : 0);
    }
    if (hook) hook(ai, ci, koff, voff);
    const std::size_t q8off = std::size_t{ai} * kq_bytes_;
    const std::size_t ksoff = std::size_t{ai} * ks_bytes_;
    const std::size_t kmoff = std::size_t{ai} * 4096;
    const std::size_t first = std::size_t{ci} * B_;              // this chunk's first cache row
    const std::size_t f16rows = std::size_t{B_} * kKvRow * 2;    // the chunk's f16 K or V rows
    const std::size_t f16first = kv16_scratch_ ? 0 : first * kKvRow * 2;
    if (attn_kq8_) {
      const std::size_t qrow = attn_kq4_ ? 512 : 1024, srow = attn_kq4_ ? 128 : 32;
      if (ci == 0)  // channel mean of the first chunk, kept for the later ones
        Dispatch(Exe("kmean.hal"), "yah_kmean", 4, 1, 1, 256, 1, 1,
                 {{kv16_->handle, koff, f16rows}, {kmbuf_->handle, kmoff, 4096}, Ref(*valid_)});
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
      b.push_back(Ref(*valid_));
      Dispatch(ChunkExe(attn_vq8_ ? "vq8" : "vq4", ci), attn_vq8_ ? "yah_vq8" : "yah_vq4", 4, (B_ + 15) / 16, 1, 256,
               1, 1, b);
    } else if (paged_f16v_ && !vtrope_) {
      Dispatch(ChunkExe("vtpage", ci), "yah_vtpage", 32, (B_ + 31) / 32, 1, 256, 1, 1,
               {{kv16_->handle, voff, f16rows}, vtpool_l, ptab_ref_});
    } else if (vtrans_) {
      Dispatch(Exe("vtrans.hal"), "yah_transpose_v16", 32, (T_ + 31) / 32, 1, 256, 1, 1,
               {{kv16_->handle, voff, kv_cache_ * 2}, {vt16_->handle, 0, vt_bytes_}});
    }
    if (tail_ == kNoRows) return;  // K / V are written; nothing reads this layer's output
    const bool fps = fps_on_ && ci >= fps_first_;  // the sparse attention reads the kv4 pools
    if (kdeq_) {  // the context so far, this chunk included, as f16 pools for the fp16 attention (fps: yah_fpmean's rows only)
      const std::uint32_t rows = (ci + 1) * B_ - (fps && ci ? (ci - 1) * B_ : 0);
      const char* kd = attn_kq4_ ? "kdeq4" : "kdeq8";
      const char* vd = attn_kq4_ ? "vdeq4" : "vdeq8";
      Dispatch(ChunkExe(kd, ci), attn_kq4_ ? "yah_kdeq4" : "yah_kdeq8", (rows * 64 + 255) / 256, 1, 1, 256, 1, 1,
               {{kq8buf_->handle, q8off, kq_bytes_}, {ksbuf_->handle, ksoff, ks_bytes_}, ptab_ref_, Ref(*kdeq_)});
      Dispatch(ChunkExe(vd, ci), attn_kq4_ ? "yah_vdeq4" : "yah_vdeq8", 4 * ((rows + 15) / 16), 1, 1, 256, 1, 1,
               {{vqbuf_->handle, vqoff, vq_bytes_}, {vqsbuf_->handle, vqsoff, vqs_bytes_}, ptab_ref_, Ref(*vdeq_)});
    }
    const hrx_buffer_ref_t kfm_l{fps_on_ ? kfm_->handle : nullptr, std::size_t{ai} * fps_mean_bytes_, fps_mean_bytes_};
    const hrx_buffer_ref_t vfm_l{fps_on_ ? vfm_->handle : nullptr, std::size_t{ai} * fps_mean_bytes_, fps_mean_bytes_};
    if (fps_on_) {  // block means of this chunk (and the previous one: the last layer skips non-final chunks), the pooled rows
      const std::uint32_t bpc = B_ / 128, first = (ci ? ci - 1 : 0) * bpc, nend = (ci + 1) * bpc;
      Dispatch(ChunkExe("fpmean", ci), "yah_fpmean", nend - first, 1, 1, 256, 1, 1,
               {Ref(*kdeq_), Ref(*vdeq_), ptab_ref_, kfm_l, vfm_l});
      Dispatch(ChunkExe("fpcopyk", ci), "yah_fpcopyk", (nend + 1) / 2, 1, 1, 256, 1, 1,
               {kfm_l, {kq8buf_->handle, q8off, kq_bytes_}, {ksbuf_->handle, ksoff, ks_bytes_}});
      Dispatch(ChunkExe("fpcopyv", ci), "yah_fpcopyv", 4, (nend + 15) / 16, 1, 256, 1, 1,
               {vfm_l, {vqbuf_->handle, vqoff, vq_bytes_}, {vqsbuf_->handle, vqsoff, vqs_bytes_}});
    }
    const bool qrot = qr16_ && ci >= qrot_first_;
    if (qrot)
      Dispatch(Exe("qrot.hal"), "yah_qrot", (B_ * 96 + 255) / 256, 1, 1, 256, 1, 1, {Ref(*q16_), Ref(*qr16_)});
    if (fps) {  // the tile lists, from the rotated Q the attention reads
      if (!qrot) throw LoomError("fpsparse needs the rotated Q (qrot) on its chunks");
      Dispatch(ChunkExe("fpsel", ci), "yah_fpsel", B_ / 16, kKvHeads, 1, 256, 1, 1,
               {Ref(*qr16_), kfm_l, Ref(*valid_), Ref(answer_ ? *valid_ : *noans_), Ref(*fsel_)});
      if (fps_ord_) Dispatch(Exe("fpord.hal"), "yah_fpord", 1, 1, 1, 256, 1, 1, {Ref(*fsel_)});
    }
    {
      std::vector<hrx_buffer_ref_t> b = {
          Ref(qrot ? *qr16_ : qnorm_ ? *q_ : *q16_),
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
      const bool deqb = kdeq_ && !fps;
      if (deqb) b = {b[0], b[1], Ref(*kdeq_), Ref(*vdeq_), b[4], b[5]};  // the fp16 kernel's bindings
      if (attn_kq8_ && !deqb) b.push_back({ksbuf_->handle, ksoff, ks_bytes_});
      if (attn_vqt_ && !deqb) b.push_back({vqsbuf_->handle, vqsoff, vqs_bytes_});
      if (kv_paged_) b.push_back(ptab_ref_);
      if (qnorm_) {
        b.push_back(TRef(*Find(pre + "attn_q_norm.weight")));
        b.push_back({eps_->handle, 0, 4});
      }
      if (fps) b.push_back(Ref(*fsel_));
      const std::string ah = std::string("wmma") + (ci ? "_c" + std::to_string(ci) : std::string()) + ".hal";
      const auto ag = geom_.find(ah);
      const std::uint32_t tpw = ag != geom_.end() ? ag->second.tokens : attn_tpw_,
                          hpw = ag != geom_.end() ? ag->second.rowgrp : attn_hpw_;
      // aov: the counters (untracked) for the rotq after it; wmmaf.hal is chunk 0's main kernel with them
      attn_flagged_ = aov_ && ah == "wmma.hal" && !fps && !qrot && n_ == B_;
      if (attn_flagged_) b.push_back(attn_cnt_ = FlagSlots(6));
      Dispatch(Exe(attn_flagged_ ? "wmmaf.hal" : ah), "yah_attn_wmma", (B_ + tpw - 1) / tpw, kHeads / hpw, 1, 256, 1, 1,
               b, ~std::uint64_t{0}, nullptr, attn_flagged_ ? std::uint64_t{1} << (b.size() - 1) : 0);
    }
    trim_row_ = tail_;
    RunResidual(pre + "attn_output.weight", *scratch_);
    trim_row_ = kAllRows;
  }


  void RunDeltaNet(std::uint32_t l, const std::string& pre) {
    const std::uint32_t si = l - l / cfg_.full_attention_interval;
    const bool dn_chained = std::exchange(dn_chained_, false);   // the last DeltaNet tail ran chained (dnring)
    // alpha / beta first: they read the norm's row-major copy (21 MB), which the qkv / gate GEMMs would evict from the
    // last-level cache (0.18 vs ~0.39 M cycles each)
    // dnov: the qkv GEMM and the conv, the gate GEMM and pnq part a as flagged pairs (FlagSlot), so the conv runs in the
    // qkv GEMM's tail and pnq part a in the gate GEMM's
    hrx_buffer_ref_t qkv_cnt{}, gate_cnt{}, dn_cnt{};
    bool qkv_flagged = false, gate_flagged = false, qkv_waited = false;
    // nqw: alpha, beta and qkv wait on the norm's counters (iu4_wait_); after a chained DeltaNet tail (dn_chained_)
    // their outputs' readers since the last barrier (that tail's DeltaNet part b and ring update) finished before its
    // pnq and so before each token tile reaches them: untracked too
    iu4_wait_ = std::exchange(nq_cnt_, {});
    iu4_untrack_ = dn_chained && iu4_wait_.buffer ? 0b1000 : 0;
    RunKstore(pre + "ssm_alpha.weight", *alpha_);
    RunKstore(pre + "ssm_beta.weight", *beta_);
    // dnvconv: the gate GEMM goes after DeltaNet's second part (below), beside it, and the conv no longer waits for it
    const bool gate_late = geom_.count("dnvconv") != 0;
    if (dnov_ && gate_late) qkv_cnt = FlagSlot();
    qkv_waited = iu4_wait_.buffer && iu4_untrack_;
    qkv_flagged = RunKstore(pre + "attn_qkv.weight", *qkv_, qkv_cnt.buffer ? &qkv_cnt : nullptr);
    if (!gate_late) RunKstore(pre + "attn_gate.weight", *gate_);
    iu4_wait_ = {}, iu4_untrack_ = 0;
    const hrx_buffer_ref_t cs{conv_state_->handle, std::size_t{si} * kQkv * 4 * 4, std::size_t{kQkv} * 4 * 4};
    const hrx_buffer_ref_t st{state_->handle, std::size_t{si} * kTs * kState * kState * 4,
                              std::size_t{kTs} * kState * kState * 4};
    // The conv with the q / k L2 norm (prep_kq) fused in.
    // dispatch.txt "convtb <tokens>": a workgroup per tokens (gen_conv_kq), else per token
    const auto ctb = geom_.find("convtb");
    const std::uint32_t conv_tb = ctb != geom_.end() ? ctb->second.tokens : 1;
    bool dn_done = false;
    // dnvconv: the conv ring update (prep_ab's ring lanes only; its alpha / beta lanes below qkv_size write ab_, which
    // nothing reads). It waits for both DeltaNet parts (they read the ring), so it is queued after the pnq kernels.
    const auto ring = [&] {
      Dispatch(Exe("prepab.hal"), "yah_deltanet_prep_ab", (kQkv + 255u) / 256u, 1, 1, 256, 1, 1,
               {Ref(*alpha_), Ref(*beta_), TRef(*Find(pre + "ssm_a")), TRef(*Find(pre + "ssm_dt.bias")), Ref(*qkv_), cs,
                Ref(*ab_), Ref(*valid_)},
               0b1100000);
    };
    if (const auto dvc = geom_.find("dnvconv"); dvc != geom_.end()) {
      // dispatch.txt "dnvconv <conv workgroups>": the conv takes only the q / k heads; DeltaNet computes the v channels'
      // conv and alpha / beta itself (gen_gdn_chunk vconv), and prep_ab only advances the conv ring, after both parts
      // (they read the ring before token 0)
      if (!dnsplit_a_ || !dvc->second.tokens) throw LoomError("dispatch.txt dnvconv needs dnsplit and a conv grid");
      // dn_chained (and a waiting qkv GEMM): conv_out_ / kqbuf_'s readers since the last barrier (the last DeltaNet tail's
      // part b) finished before the token tiles reached the qkv GEMM: untracked too
      // dispatch.txt "convkqp.hal <workgroups>": the flagged conv as persistent workgroups (gen_conv_kq.gen_p)
      const auto cp = geom_.find("convkqp.hal");
      if (qkv_flagged && cp != geom_.end())
        Dispatch(Exe("convkqp.hal"), "yah_ssm_conv_kq_p", cp->second.tokens, 1, 1, 128, 1, 1,
                 {Ref(*qkv_), TRef(*Find(pre + "ssm_conv1d.weight")), cs, Ref(*conv_out_), Ref(*kqbuf_), qkv_cnt},
                 0b011000, nullptr, dn_chained && qkv_waited ? 0b111001 : 0b100001);
      else if (qkv_flagged)
        Dispatch(Exe("convkqf.hal"), "yah_ssm_conv_kq", dvc->second.tokens, B_ / conv_tb, 1, 256, 1, 1,
                 {Ref(*qkv_), TRef(*Find(pre + "ssm_conv1d.weight")), cs, Ref(*conv_out_), Ref(*kqbuf_), qkv_cnt},
                 0b011000, nullptr, dn_chained && qkv_waited ? 0b111001 : 0b100001);
      else
        Dispatch(Exe("convkq.hal"), "yah_ssm_conv_kq", dvc->second.tokens, B_ / conv_tb, 1, 256, 1, 1,
                 {Ref(*qkv_), TRef(*Find(pre + "ssm_conv1d.weight")), cs, Ref(*conv_out_), Ref(*kqbuf_)});
      const std::vector<hrx_buffer_ref_t> in{Ref(*conv_out_), Ref(*qkv_), TRef(*Find(pre + "ssm_conv1d.weight")), cs,
                                             Ref(*alpha_), Ref(*beta_), TRef(*Find(pre + "ssm_a")),
                                             TRef(*Find(pre + "ssm_dt.bias")), Ref(*valid_)};
      auto ba = in, bb = in;
      if (pnq_ab_) {
        // raw_: part b's compact f16 rows, then part a's (gen_rot.gen_pnq "ab"); part b counts its finished lanes in
        // dn_cnt, which the merged pnq waits on
        const std::size_t b_bytes = std::size_t{B_} * geom_.at("pnq_b.hal").tt * 1024 * 2;
        ba.insert(ba.end(), {st, {raw_->handle, b_bytes, std::size_t{B_} * geom_.at("pnq_a.hal").tt * 1024 * 2}});
        dn_cnt = FlagSlot();
        bb.insert(bb.end(), {st, {raw_->handle, 0, b_bytes}, dn_cnt});
      } else {
        ba.insert(ba.end(), {st, Ref(*raw_)});
        bb.insert(bb.end(), {st, Ref(*raw2_)});
      }
      Dispatch(Exe("rowsplit_a.hal"), "yah_deltanet", dn_rowgrp_, dnsplit_a_, 1, dn_threads_, 1, 1, ba, 0b11000000000);
      Dispatch(Exe("rowsplit_b.hal"), "yah_deltanet", dn_rowgrp_, dnsplit_b_, 1, dn_threads_, 1, 1, bb, 0b111000000000,
               nullptr, pnq_ab_ ? 0b100000000000 : 0);
      // The second part's few workgroups leave most of the GPU to the gate GEMM; pnq part a beside it was DRAM-bound and
      // slowed it about as much, but gained less than the GEMM's whole tail and the conv's wait.
      if (gate_late) {
        if (dnov_) gate_cnt = FlagSlot();
        gate_flagged = RunKstore(pre + "attn_gate.weight", *gate_, gate_cnt.buffer ? &gate_cnt : nullptr);
      }
      dn_done = true;
    } else {
      Dispatch(Exe("convkq.hal"), "yah_ssm_conv_kq", 40, B_ / conv_tb, 1, 256, 1, 1,
               {Ref(*qkv_), TRef(*Find(pre + "ssm_conv1d.weight")), cs, Ref(*conv_out_), Ref(*kqbuf_)});
    }
    if (!dn_done) {
      // One lane index does two jobs: advance the conv ring past the real tokens (i < qkv_size; the next chunk and the
      // decoder read it) and alpha / beta (i < B * heads). So the grid covers max() of the two.
      const std::uint32_t prepab_tiles = (std::max<std::uint32_t>(B_ * kTs, kQkv) + 255u) / 256u;
      Dispatch(Exe("prepab.hal"), "yah_deltanet_prep_ab", prepab_tiles, 1, 1, 256, 1, 1,
               {Ref(*alpha_), Ref(*beta_), TRef(*Find(pre + "ssm_a")), TRef(*Find(pre + "ssm_dt.bias")), Ref(*qkv_), cs,
                Ref(*ab_), Ref(*valid_)});
      // DeltaNet grid: (blocks per head, heads) x 256, blocks per head from the "rowsplit.hal" row group.
      // Split (dnsplit): heads [0, a) into raw_, then [a, kTs) into raw2_. The first part's postnorm depends only on raw_,
      // so it runs beside the second part (whose few workgroups leave most slots free).
      if (dnsplit_a_) {
        Dispatch(Exe("rowsplit_a.hal"), "yah_deltanet", dn_rowgrp_, dnsplit_a_, 1, dn_threads_, 1, 1,
                 {Ref(*conv_out_), Ref(*kqbuf_), Ref(*ab_), st, Ref(*raw_)});
        Dispatch(Exe("rowsplit_b.hal"), "yah_deltanet", dn_rowgrp_, dnsplit_b_, 1, dn_threads_, 1, 1,
                 {Ref(*conv_out_), Ref(*kqbuf_), Ref(*ab_), st, Ref(*raw2_)});
      } else {
        Dispatch(Exe("rowsplit.hal"), "yah_deltanet", dn_rowgrp_, kTs, 1, dn_threads_, 1, 1,
                 {Ref(*conv_out_), Ref(*kqbuf_), Ref(*ab_), st, Ref(*raw_)});
      }
    }
    if (tail_ == kNoRows) {  // the recurrent state is written; nothing reads this layer's output
      if (dn_done) ring();
      return;
    }
    const hrx_buffer_ref_t gate{gate_->handle, 0, std::size_t{B_} * kInner * 4};
    // dispatch.txt "dnperm": the DeltaNet parts run head sets (gen_rot.pn_heads) into compact outputs, which only the
    // pnq kernels read
    const bool pnq = a4_ && dnsplit_a_ && geom_.count("pnq_a.hal");
    if (pnq != (geom_.count("dnperm") != 0)) throw LoomError("the A4 pnq postnorm and dnperm go together");
    if (dn_done && !pnq) throw LoomError("dispatch.txt dnvconv needs the pnq postnorm");
    if (pnq) {
      // A4: the postnorm and ssm_out's rotq in one kernel per DeltaNet part (gen_rot.gen_pnq), each reading its part's
      // output (after the gate GEMM, which runs beside the second DeltaNet part)
      const auto* nw = Find(pre + "ssm_norm.weight");
      const auto* aw = Find(pre + "ssm_out.awq");
      const Geom& ga = geom_.at("pnq_a.hal");
      const Geom& gb = geom_.at("pnq_b.hal");
      // dnov: part a waits on the gate GEMM's counters while the GEMM still reads aq_ / ad_, so both parts write the
      // other pair, which becomes aq_ / ad_
      const LoomBuffer& q = gate_flagged ? *aq2_ : *aq_;
      const LoomBuffer& d = gate_flagged ? *ad2_ : *ad_;
      // dnring: the ring update (prepabr) waits on dn_cnt and counts in ring_cnt, the pnq waits on that and counts its rows
      // in pq_cnt for ssm_out's kres: the whole tail runs without a graph barrier (the next layer's input GEMMs and conv
      // leave their outputs untracked, dn_chained_)
      const bool chained = dnring_ && pnq_ab_ && gate_flagged;
      hrx_buffer_ref_t pq_cnt{};
      if (pnq_ab_) {
        // raw_ untracked: part a's DeltaNet is behind the barrier before part b's, part b's done count is dn_cnt
        if (!dn_done) throw LoomError("pnq_abf.hal needs dnvconv");
        const Geom& gab = geom_.at("pnq_abf.hal");
        if (chained) {
          // cs untracked: the DeltaNet parts read it (part b before dn_cnt is complete, part a behind part b's barrier)
          const hrx_buffer_ref_t ring_cnt = FlagSlot();
          pq_cnt = FlagSlot();
          Dispatch(Exe("prepabr.hal"), "yah_deltanet_prep_ab", (kQkv + 255u) / 256u, 1, 1, 256, 1, 1,
                   {Ref(*qkv_), cs, Ref(*valid_), dn_cnt, ring_cnt}, 0b10010, nullptr, 0b11010);
          // pnq_abfrp: as persistent workgroups, so ssm_out's kres is not queued behind 12288 pnq workgroups
          const auto pp = geom_.find("pnq_abfrp.hal");
          const bool persist = pp != geom_.end();
          Dispatch(Exe(persist ? "pnq_abfrp.hal" : "pnq_abfr.hal"), "yah_pnq", persist ? pp->second.tokens : gab.tt,
                   persist ? 1 : B_, 1, gab.rowgrp, 1, 1,
                   {Ref(*raw_), TRef(*nw), gate, TRef(*aw), Ref(*rot_signs_), Ref(q), Ref(d), gate_cnt, ring_cnt, pq_cnt},
                   0b1001100000, nullptr, 0b1110000101);
        } else if (gate_flagged)
          Dispatch(Exe("pnq_abf.hal"), "yah_pnq", gab.tt, B_, 1, gab.rowgrp, 1, 1,
                   {Ref(*raw_), TRef(*nw), gate, TRef(*aw), Ref(*rot_signs_), Ref(q), Ref(d), gate_cnt, dn_cnt}, 0b1100000,
                   nullptr, 0b110000101);
        else   // a trimmed chunk: the gate GEMM ran unflagged, the graph orders this after it
          Dispatch(Exe("pnq_abd.hal"), "yah_pnq", gab.tt, B_, 1, gab.rowgrp, 1, 1,
                   {Ref(*raw_), TRef(*nw), gate, TRef(*aw), Ref(*rot_signs_), Ref(q), Ref(d), dn_cnt}, 0b1100000,
                   nullptr, 0b10000001);
      } else if (gate_flagged)
        Dispatch(Exe("pnq_af.hal"), "yah_pnq", ga.tt, B_, 1, ga.rowgrp, 1, 1,
                 {Ref(*raw_), TRef(*nw), gate, TRef(*aw), Ref(*rot_signs_), Ref(q), Ref(d), gate_cnt}, 0b1100000, nullptr,
                 0b10000100);
      else
        Dispatch(Exe("pnq_a.hal"), "yah_pnq", ga.tt, B_, 1, ga.rowgrp, 1, 1,
                 {Ref(*raw_), TRef(*nw), gate, TRef(*aw), Ref(*rot_signs_), Ref(q), Ref(d)}, 0b1100000);
      if (!pnq_ab_)
        Dispatch(Exe("pnq_b.hal"), "yah_pnq", gb.tt, B_, 1, gb.rowgrp, 1, 1,
                 {Ref(*raw2_), TRef(*nw), gate, TRef(*aw), Ref(*rot_signs_), Ref(q), Ref(d)}, 0b1100000);
      if (gate_flagged) std::swap(aq_, aq2_), std::swap(ad_, ad2_);
      if (dn_done && !chained) ring();
      trim_row_ = tail_;
      RunResidual(pre + "ssm_out.weight", *scratch_, true, chained ? &pq_cnt : nullptr);
      trim_row_ = kAllRows;
      dn_chained_ = chained;
      return;
    }
    if (dnsplit_a_) {
      Dispatch(Exe("postnorm_a.hal"), "yah_ssm_postnorm_fp16", dnsplit_a_ * B_ / 8, 1, 1, 256, 1, 1,
               {Ref(*raw_), TRef(*Find(pre + "ssm_norm.weight")), gate, Ref(*scratch_)});
      Dispatch(Exe("postnorm_b.hal"), "yah_ssm_postnorm_fp16", dnsplit_b_ * B_ / 8, 1, 1, 256, 1, 1,
               {Ref(*raw2_), TRef(*Find(pre + "ssm_norm.weight")), gate, Ref(*scratch_)});
    } else {
      Dispatch(Exe("postnorm.hal"), "yah_ssm_postnorm_fp16", 6 * B_, 1, 1, 256, 1, 1,
               {Ref(*raw_), TRef(*Find(pre + "ssm_norm.weight")), gate, Ref(*scratch_)});
    }
    trim_row_ = tail_;
    RunResidual(pre + "ssm_out.weight", *scratch_);
    trim_row_ = kAllRows;
  }

  LoomDevice& gpu_;
  const core::Gguf& gguf_;
  const core::Qwen35Config& cfg_;
  std::string dir_;
  const LoomWeights& weights_;
  std::map<std::string, Geom> geom_;
  std::map<std::string, LoomExecutable> exes_;
  std::deque<LoomBuffer> keep_;  // stable addresses
  std::uint32_t B_ = 0, T_ = 0, full_ = 0, pages_ = 0, dn_rowgrp_ = 0, dn_threads_ = 256, attn_hpw_ = 0, attn_tpw_ = 0;
  std::uint32_t dnsplit_a_ = 0, dnsplit_b_ = 0;  // DeltaNet head split (dispatch.txt "dnsplit"; 0: one dispatch)
  std::uint32_t qrot_first_ = 0;  // first chunk whose attention reads yah_qrot's rotated Q (dispatch.txt "qrot")
  // fpsparse (gen_fpsparse): from chunk fps_first_ on; fps_ext_ pooled-row pages; fps_ls_ list dwords per (query block,
  // KV head); per attention layer the block means [fps_blocks_][1024] f16 (kfm_ K, vfm_ V)
  bool fps_on_ = false, fps_ord_ = false;
  std::uint32_t fps_first_ = 0, fps_ext_ = 0, fps_ls_ = 0, fps_blocks_ = 0;
  std::size_t fps_mean_bytes_ = 0, deq_bytes_ = 0;
  LoomBuffer *kfm_ = nullptr, *vfm_ = nullptr, *fsel_ = nullptr;
  std::uint32_t n_ = 0;  // real tokens of the chunk from the last Embed
  LoomGraph* graph_ = nullptr;  // open while RunLayers records
  std::unique_ptr<LoomGraph> chunk_graph_;
  std::uint32_t split_layer_ = 0;  // RunLayers: SplitGraph before this layer's first dispatch (0: none)
  bool split_armed_ = false;       // recording split_layer_: its first dispatch splits
  bool warming_ = false;           // WarmExecutables' pass: Dispatch records nothing
  // The NPU FFN block (EnableNpu, RunFfnBlock)
  NpuSplit* npu_ = nullptr;
  bool npu_on_ = false;            // this chunk runs it (full chunks)
  bool npu_planning_ = false;      // EnableNpu's binding pass: binds the calls, enqueues nothing
  std::vector<NpuSplit::Queued> flagged_;   // the chunk's jobs, in NpuEnqueue order
  std::vector<hrx_buffer_ref_t> job_c_;     // each job's C range (NpuJoin orders on it)
  std::size_t npu_cleared_ = 0, npu_joined_ = 0;   // jobs whose flag words NpuJoin cleared / waited for
  LoomBuffer *npu_down_ = nullptr, *npu_xo_ = nullptr;   // the down pack; normqx's rows in the decoder's positions
  std::size_t down_layer_ = 0;     // the down pack's bytes per layer
  std::uint32_t npu_layer_ = 0;    // the layer being recorded (NpuSplit::Queued::layer)
  std::uint32_t npu_layers_ = 0;   // the layers that run the block (EnableNpu)
  // pairs (EnablePairs): the other chunk's buffers and state (SelectChunk swaps them in), the selected chunk, and each
  // chunk's FFN block job to join before its next layer (kNoJoin: none)
  struct ChunkState {
    LoomBuffer *hidden = nullptr, *hidden2 = nullptr, *ids = nullptr, *valid = nullptr;
    std::uint32_t n = 0;
    std::int64_t keep = -1;
    bool answer = false;
  };
  bool pairs_ = false;
  ChunkState other_;
  std::uint32_t par_ = 0;
  static constexpr std::size_t kNoJoin = ~std::size_t{0};
  std::size_t join_[2] = {kNoJoin, kNoJoin};
  bool kv16_scratch_ = false, kv_paged_ = false, vtrans_ = false, rope_kpaged_ = false;
  bool qnorm_ = false;  // the attention normalizes / rotates its own (raw f32) Q; RoPE does K / V only
  bool qg16_ = false;   // the kqg GEMM stores q and gate as f16 (only the GPU kqg path writes them so)
  bool vtrope_ = false;  // RoPE stores V^T straight into the paged pool (no yah_vtpage pass)
  bool attn_kq4_ = false, attn_kq8_ = false, attn_vq4_ = false, attn_vq8_ = false, attn_vqt_ = false;
  bool paged_f16k_ = false, paged_f16v_ = false;
  std::size_t kv_cache_ = 0, kv16_layer_ = 0, vt_bytes_ = 0, ks_bytes_ = 0, kq_bytes_ = 0, vq_bytes_ = 0,
              vqs_bytes_ = 0, pool_bytes_ = 0;
  hrx_buffer_ref_t ptab_ref_{};
  std::vector<std::uint32_t> host_ids_;  // Embed's staging
  // MQ4 sets ("rot"): the rotation signs (gen_rot.signs) and ssm_out's input in HF head order
  LoomBuffer *rot_signs_ = nullptr, *rotbuf_ = nullptr;
  // A4 sets (dispatch.txt "a4"): int4 activations (B_ x kFfn / 2 bytes) and their scales (one per 128) for the iu4 GEMMs
  bool a4_ = false, nqov_ = false, dnov_ = false;
  bool pnq_ab_ = false;   // dispatch.txt pnq_abf.hal: one flagged pnq for both DeltaNet parts, their outputs in raw_
  // dispatch.txt guov: the flagged gu GEMM and its rotq (rotqf_17408); gu_flagged_: the last gu ran flagged on gu_cnt_
  bool guov_ = false, gu_flagged_ = false;
  hrx_buffer_ref_t gu_cnt_{};
  // dispatch.txt nqgu: the FFN norm's normq counts its finished rows (normq[f]o_5120) in nq_cnt_ when nq_want_, and
  // the flagged gu waits on them (iu4_gufw_*); nq_cnt_.buffer is null unless the last normq counted (nqw: the attention
  // norm's, for iu4_wait_)
  // gu_waited_: the last flagged gu also waited (RotQ)
  bool nqgu_ = false, nq_want_ = false, gu_waited_ = false;
  // dispatch.txt rotqp_17408.hal (rqk, guov): the flagged gu's counting rotq as persistent workgroups (RotQ)
  bool rqp_ = false;
  // dispatch.txt "aov": chunk 0's attention as wmmaf.hal adds to attn_cnt_ (attn_flagged_), the attention-out rotq
  // (rotqopf_6144) waits on it without a graph barrier
  bool aov_ = false, attn_flagged_ = false;
  // dispatch.txt "ropew": the attention's k / v GEMMs flagged (kv_cnt_), RoPE (ropew) waits on them
  bool ropew_ = false, kv_flagged_ = false, ropekv_ = false;
  // dispatch.txt "nqp": the flag-counting normq (normq[f]o) runs as nqp_ persistent workgroups
  std::uint32_t nqp_ = 0;
  hrx_buffer_ref_t kv_cnt_[2]{};
  hrx_buffer_ref_t attn_cnt_{};
  // dispatch.txt rqk: the deferred kres waits on its rotq's counters (rotqfo_17408 / rotqo_<k>, iu4_kresfw_*)
  bool rqk_ = false;
  // dispatch.txt nqw: the attention / DeltaNet input GEMMs wait on their norm's counters (iu4_wait_ = nq_cnt_ around
  // them; iu4_kstorew_*, iu4_kstorefw_*, iu4_kqgw_*)
  bool nqw_ = false;
  hrx_buffer_ref_t iu4_wait_{};
  // iu4_untrack_: bindings the waiting input GEMMs also leave untracked (RunIu4 untrack); dnring: dn_chained_, the last
  // DeltaNet tail ran without a graph barrier (RunDeltaNet)
  std::uint64_t iu4_untrack_ = 0;
  bool dnring_ = false, dn_chained_ = false;
  hrx_buffer_ref_t nq_cnt_{};
  LoomBuffer *aq_ = nullptr, *ad_ = nullptr, *aq2_ = nullptr, *ad2_ = nullptr, *flags_ = nullptr;
  std::size_t flag_slot_ = 0;
  LoomBuffer *hidden_ = nullptr, *reszero_ = nullptr, *sumout_ = nullptr, *scratch_ = nullptr, *qkv_ = nullptr,
             *gate_ = nullptr, *alpha_ = nullptr, *beta_ = nullptr, *q_ = nullptr, *q16_ = nullptr, *kbuf_ = nullptr,
             *vbuf_ = nullptr, *raw_ = nullptr, *raw2_ = nullptr, *conv_out_ = nullptr, *kqbuf_ = nullptr, *ab_ = nullptr,
             *conv_state_ = nullptr, *state_ = nullptr, *kv16_ = nullptr, *kc32_ = nullptr, *vc32_ = nullptr,
             *lse_ = nullptr, *eps_ = nullptr, *ffnup_ = nullptr, *gateffn_ = nullptr, *uwstage_ = nullptr,
             *wstage_ = nullptr, *ostage_ = nullptr, *hidden2_ = nullptr, *normed_ = nullptr,
             *ptab_ = nullptr, *vt16_ = nullptr, *kq8buf_ = nullptr, *ksbuf_ = nullptr, *kmbuf_ = nullptr,
             *vqbuf_ = nullptr, *vqsbuf_ = nullptr, *kpool_ = nullptr, *vtpool_ = nullptr, *valid_ = nullptr, *noans_ = nullptr, *qr16_ = nullptr,
             *kdeq_ = nullptr, *vdeq_ = nullptr, *ids_ = nullptr;
  std::int64_t keep_rows_ = kAllRows, tail_ = kAllRows, trim_row_ = kAllRows;
  bool answer_ = false;
};

}  // namespace yah::model

#include "model/loom_prefill_npu-inl.hpp"

#endif  // YAH_MODEL_LOOM_PREFILL_HPP_
