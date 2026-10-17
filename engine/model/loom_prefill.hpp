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

#include <unistd.h>

#include <chrono>
#include <thread>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "model/loom_decoder.hpp"
#include "model/loom_runtime.hpp"
#include "model/npu_split.hpp"
#include "model/prefill_calib.hpp"

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
    LoadHalves();
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

  // Buffer sizes of the NPU column split this set carries (dispatch.txt "npusplit_<site>", "npubytes_<K>"); zero: none.
  // A job is the NPU work behind one cut: the DeltaNet qkv + gate, or one other site; the buffers fit the largest
  // (W twice).
  [[nodiscard]] NpuPlan npu_plan() const {
    NpuPlan p = WholePlan();
    if (!hgeom_.empty() && p.dcol) {  // token halves: their A slots and C regions (HalfA, HalfC)
      p.a_bytes = std::max(p.a_bytes, 6 * HNpuK(5120).a);
      p.c_bytes = std::max(p.c_bytes, HalfC(1) + HalfCBytes());
    }
    return p;
  }
  // The NPU buffers of the whole-chunk jobs.
  [[nodiscard]] NpuPlan WholePlan() const {
    NpuPlan p;
    const auto need = [&](const std::string& site) {
      const auto it = geom_.find("npusplit_" + site);
      NpuPlan q;
      if (it == geom_.end()) return q;
      const std::size_t calls = it->second.tokens / NpuCallRows(), mats = site == "ffn" ? 2 : 1;
      for (const auto& [k0, K] : NpuChunks(site)) {
        const NpuBytes nb = NpuK(K);
        q.a_bytes += nb.a, q.w_bytes += mats * calls * nb.w, q.c_bytes += mats * calls * nb.c;
      }
      return q;
    };
    const NpuPlan qkv = need("qkv"), gate = need("gate");
    for (const NpuPlan& q : {NpuPlan{std::max(qkv.a_bytes, gate.a_bytes), qkv.w_bytes + gate.w_bytes,
                                     qkv.c_bytes + gate.c_bytes},
                             need("q"), need("out"), need("down"), need("ffn")})
      p.a_bytes = std::max(p.a_bytes, q.a_bytes), p.w_bytes = std::max(p.w_bytes, q.w_bytes),
      p.c_bytes = std::max(p.c_bytes, q.c_bytes);
    p.w_bytes *= 2;   // two slots: a job's weights are decoded while the previous job runs (NpuDecode)
    if (FfnBlkF()) {   // the FFN block: gate / up's activations; H and down's partials in C
      p.a_bytes = std::max(p.a_bytes, NpuK(5120).a);
      const auto parts = FfnBlkParts(FfnBlkF());
      p.c_bytes = std::max(p.c_bytes, FfnBlkDown() + parts.size() * FfnBlkCalls() *
                                                         NpuBytesOf("npubytes_ffndn" + std::to_string(parts[0].second)).c);
    }
    p.dcol = geom_.count("npudcol") != 0;
    p.gfirst = geom_.count("npugfirst") != 0;
    if (p.dcol) {
      // the NPU decodes raw rows: W is one call's unread panel binding; A holds the p4 stream, then the pk one (Q4_K)
      p.w_bytes = NpuK(5120).w;
      p.a_bytes *= 2;
    }
    if (const auto it = geom_.find("npugate"); it != geom_.end())
      p.gate_calls = it->second.rowgrp, p.gate_record = it->second.tt;
    return p;
  }
  // Split the set's NPU sites of full chunks with the NPU from now on: it computes their trailing npusplit rows.
  // npu (built from npu_plan()) outlives this object.
  void EnableNpu(NpuSplit* npu) {
    for (const auto& [name, g] : geom_)
      if (name.rfind("npusplit_", 0) == 0) npu_rows_[name.substr(9)] = g.tokens;
    ffnblk_ = FfnBlkF();
    if (!npu || (npu_rows_.empty() && !ffnblk_)) throw LoomError("prefill: the set has no NPU split");
    if (!geom_.count("npu_flag_wait.hal") || !geom_.count("npugate"))
      throw LoomError("prefill: the set has no gated NPU handoffs (npu_flag_wait.hal, npugate; re-emit it)");
    npu_ = npu;
    npu_dcol_ = geom_.count("npudcol") != 0;
    halves_ = !hgeom_.empty() && npu_dcol_;
    if (halves_ && geom_.count("dnperm")) throw LoomError("dnperm sets have no token-half DeltaNet");
    if (halves_ && (npu_->A().size < 6 * HNpuK(5120).a || npu_->C().size < HalfC(1) + HalfCBytes()))
      throw LoomError("prefill: the NPU buffers lack the token halves' slots (build the split from npu_plan())");
    std::size_t rem = 0;
    for (const auto& [site, rows] : npu_rows_)
      if (NpuRem(site)) rem = std::max<std::size_t>(rem, std::size_t{B_} * rows * 4);
    if (rem) npurem_ = &Alloc(rem);
    if (ffnblk_ && geom_.count("npu_unpack_ffnblk.hal")) ffnrem_ = &Alloc(std::size_t{B_} * kHidden * 4);   // (npuffnrem)
    // Bind every NPU call of a full chunk now (a pass of the layers that dispatches nothing): a first bind loads the
    // image's storage and patches it, milliseconds per call.
    LoomBuffer *h = hidden_, *h2 = hidden2_;
    const std::uint32_t n = n_;
    n_ = B_, npu_on_ = true, npu_planning_ = true;
    planned_dq_.clear();
    npu_job_ = 0;
    NewGraph();
    RecordLayers(0, {});
    npu_->EndBinding();
    chunk_graph_.reset();
    graph_ = nullptr, n_ = n, npu_on_ = false, npu_planning_ = false;
    hidden_ = h, hidden2_ = h2;
  }

  // Zero the recurrent state before a new sequence. KV rows need no reset: attention reads only keys <= the query.
  void Reset() {
    Zero(*conv_state_);
    Zero(*state_);
  }

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
    // YAH_DUMP_ACT / YAH_FPS_STATS / YAH_FPS_DUMP: no graph, so the hooks can synchronize and read outputs back
    if (DumpOn() || std::getenv("YAH_FPS_STATS") || std::getenv("YAH_FPS_DUMP")) {
      if (!df_planned_) PlanDecodeFree(ci);
      nodes_.clear();
      df_on_ = false;
      RecordLayers(ci, hook);
      gpu_.Synchronize();
      return;
    }
    if (!df_planned_) PlanDecodeFree(ci);
    // NPU split: full chunks only, its handoffs graph nodes on flag words (NpuEnqueue / NpuJoin)
    npu_on_ = npu_ && n_ == B_;
    if (npu_on_) npu_->BeginChunk();
    flagged_.clear();
    predecoded_.clear();
    npu_job_ = 0, npu_cleared_ = 0, npu_joined_ = 0;
    job_side_.clear();
    ffn_pending_ = {};
    NewGraph();
    nodes_.clear();
    const bool calibrate = !npu_on_ && calib_ && calib_->BeginChunk(n_);
    // Layer 0 goes first in a graph of its own, so the GPU runs it while the host records the other layers. The split
    // falls before its deferred down GEMM (the first dispatch while recording layer 1), which waits for all of the
    // layer's last kernel anyway. The NPU split's flagged jobs and the calibration's matching keep one graph.
    split_layer_ = !npu_on_ && !calibrate && cfg_.main_block_count() > 1 ? 1 : 0;
    df_on_ = !npu_on_ && !df_plan_.empty() && n_ > kDfMinTokens;
    df_k_ = 0;
    if (df_on_) DispatchDequant(0);
    RecordLayers(ci, hook);
    FlushKres();
    split_layer_ = 0, split_armed_ = false;
    graph_ = nullptr;
    if (calibrate && !gpu_.profiling()) {
      try {
        gpu_.ProfileBegin([this](const hrx_profile_dispatch_t* e, std::size_t n, std::uint64_t) {
          session_.insert(session_.end(), e, e + n);
        });
      } catch (const LoomError& e) {  // e.g. HRX_PROFILE_FILE holds the device's profiler
        std::fprintf(stderr, "prefill calibration off: %s\n", e.what());
        calib_.reset();
      }
    }
    // every graph of the session, to match them in order (one graph per chunk; NPU chunks are not matched)
    if (gpu_.profiling() && flagged_.empty()) session_chunks_.push_back(nodes_);
    std::vector<NpuSplit::Queued> jobs;
    // YAH_NPU_CPU=1 (safe test of the NPU host path): the host stands in for the GPU, which runs nothing; it stores
    // each job's ready word, waits up to 1 s for its done word and reports the first job that stalls or fails
    if (npu_on_ && std::getenv("YAH_NPU_CPU")) {
      jobs.swap(flagged_);
      graph_ = nullptr;
      if (const char* r = std::getenv("YAH_NPU_CPU_JOBS")) {   // "a-b,c-d": only these jobs (whole family runs)
        if (!std::strcmp(r, "list"))
          for (std::size_t i = 0; i < jobs.size(); ++i) std::fprintf(stderr, "npu cpu-driven: job %zu %s\n", i, jobs[i].tag.c_str());
        std::vector<NpuSplit::Queued> kept;
        std::size_t a = 0, b = 0;
        for (int n = 0; std::sscanf(r, "%zu-%zu%n", &a, &b, &n) == 2 && a <= b && b < jobs.size(); r += n) {
          kept.insert(kept.end(), jobs.begin() + a, jobs.begin() + b + 1);
          if (r[n] != ',') break;
          ++r;
        }
        jobs.swap(kept);
        for (const auto& q : jobs) std::fprintf(stderr, "npu cpu-driven: keep %s gate %08x\n", q.tag.c_str(), q.words.gate);
      }
      std::thread enq([&] {   // it blocks while the NPU's queue is full, the NPU waits for the stores below
        try {
          npu_->Enqueue(jobs);
        } catch (const std::exception& e) {
          std::fprintf(stderr, "npu cpu-driven: enqueue: %s\n", e.what());
        }
      });
      struct Join {
        std::thread& t;
        ~Join() { t.join(); }
      } join{enq};
      for (std::size_t i = 0; i < jobs.size(); ++i) {
        const NpuSplit::Job& w = jobs[i].words;
        std::vector<std::uint32_t> before;   // the job's gate slots, to report which words its calls changed
        for (std::uint32_t k = w.ready; k <= w.done + 16; ++k) before.push_back(npu_->HostLoad(k));
        if (const char* h = std::getenv("YAH_NPU_CPU_HOLD"); h && std::strtoul(h, nullptr, 10) == i) {
          // the job parked at its gate (its setup done): the stall command sees the array waiting
          std::this_thread::sleep_for(std::chrono::milliseconds(300));
          if (const char* cmd = std::getenv("YAH_NPU_CPU_STALL_CMD")) std::system(cmd);
        }
        npu_->HostStore(w.ready, w.gate);
        const auto t0 = std::chrono::steady_clock::now();
        std::uint32_t d = 0;
        while (((d = npu_->HostLoad(w.done)) & 0x7fffffffu) < w.gate) {
          if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(1)) break;
        }
        npu_->HostStore(w.ready, 0), npu_->HostStore(w.done, 0);   // as NpuJoin's stores do
        if ((d & 0x7fffffffu) < w.gate || (d & 0x80000000u)) {
          std::fprintf(stderr, "npu cpu-driven: job %zu (%s) stalled or failed: done %08x gate %08x\n", i,
                       jobs[i].tag.c_str(), d, w.gate);
          if (const char* cmd = std::getenv("YAH_NPU_CPU_STALL_CMD")) std::system(cmd);   // e.g. an AIE status dump
          for (std::uint32_t k = w.ready; k <= w.done + 16; ++k)
            if (const std::uint32_t v = npu_->HostLoad(k); v != before[k - w.ready])
              std::fprintf(stderr, "  word +%u: %08x -> %08x\n", k - w.ready, before[k - w.ready], v);
          return;
        }
      }
      std::fprintf(stderr, "npu cpu-driven: %zu jobs done\n", jobs.size());
      if (const char* cmd = std::getenv("YAH_NPU_CPU_DONE_CMD")) std::system(cmd);
      return;
    }
    chunk_graph_->Launch();
    jobs.swap(flagged_);
    if (!jobs.empty()) {
      try {
        npu_->Enqueue(jobs);   // on a failure it releases the graph's waits for the jobs left
      } catch (...) {
        gpu_.Synchronize();   // the graph reads NPU buffers: it must end before anyone frees them
        throw;
      }
    }
  }

  // The last RunLayers graph's dispatches in node order (= the order of their dispatch timestamps' command indices).
  struct Node {
    std::array<std::uint32_t, 3> grid;
    std::string name;
    int tag = -1;  // the calibration's tag (PrefillCalib::Choose)
  };
  [[nodiscard]] const std::vector<Node>& nodes() const { return nodes_; }

  // The 64 layers of chunk ci, in dispatch order.
  void RecordLayers(std::uint32_t ci, const KvHook& hook) {
    flag_slot_ = 0;
    Dispatch(Exe("embed.hal"), "yah_embed", GeomOf("embed.hal").tt, 1, 1, kHidden / 16, 1, 1,
             {TRef(*Find("token_embd.weight")), Ref(*ids_), Ref(*hidden_)}, 4);
    for (std::uint32_t l = 0; l < cfg_.main_block_count(); ++l) {
      const std::string pre = "blk." + std::to_string(l) + ".";
      layer_ = l;
      split_armed_ = split_layer_ && l == split_layer_;
      tail_ = l + 1 == cfg_.main_block_count() ? keep_rows_ : kAllRows;
      if (cfg_.IsFullAttention(l)) {
        const bool q_af = Af("gemm_kqg", pre + "attn_q.weight");
        const bool k_af = Af("gemm_kstore", pre + "attn_k.weight");
        const bool v_af = Af("gemm_kstore", pre + "attn_v.weight");
        const bool q_npu = NpuRows("q") && NpuOk(pre + "attn_q.weight") && !NpuHal(KqgHal(pre + "attn_q.weight", q_af)).empty();
        if (!q_npu || !RunAttentionHalves(l, ci, pre, hook, q_af, k_af, v_af)) {
          FfnUnpackPending(-1);
          nq_want_ = nqw_ && !q_npu;
          RunNorm(pre + "attn_norm.weight",
                  q_af && k_af && v_af   ? NormOut::kTiled
                  : q_af || k_af || v_af ? NormOut::kBoth
                                         : NormOut::kRow,
                  q_npu ? "q" : nullptr);
          nq_want_ = false;
          RunAttention(l, ci, pre, hook, q_af, k_af, v_af);
        }
      } else {
        // afrag qkv / gate read the fragment-major copy; alpha / beta keep the row-major one
        const bool qkv_af = Af("gemm_kstore", pre + "attn_qkv.weight");
        const bool gate_af = Af("gemm_kstore", pre + "attn_gate.weight");
        const bool qkv_npu = NpuRows("qkv") && NpuRows("gate") && NpuOk(pre + "attn_qkv.weight") &&
                             NpuOk(pre + "attn_gate.weight") &&
                             !NpuHal(KstoreHal(pre + "attn_qkv.weight", qkv_af)).empty() &&
                             !NpuHal(KstoreHal(pre + "attn_gate.weight", gate_af)).empty();
        if (!qkv_npu || !RunDeltaNetHalves(l, pre, qkv_af, gate_af)) {
          FfnUnpackPending(-1);
          nq_want_ = nqw_ && !qkv_npu;
          RunNorm(pre + "attn_norm.weight", qkv_af || gate_af ? NormOut::kBoth : NormOut::kRow,
                  qkv_npu ? "qkv" : nullptr);
          nq_want_ = false;
          RunDeltaNet(l, pre, qkv_af, gate_af);
        }
      }
      if (tail_ == kNoRows) continue;
      trim_row_ = tail_;
      RunFfn(pre);
      trim_row_ = kAllRows;
    }
    FfnUnpackPending(-1);
    tail_ = kAllRows;
  }


  // Calibrate the GEMM variants while serving (model/prefill_calib.hpp), state in path. The caller runs Collect() after
  // the chunks of a prompt completed (Synchronize).
  void EnableCalibration(const std::string& path) {
    std::map<std::string, std::uint32_t> hals;
    for (const auto& [name, g] : geom_) hals[name] = g.tokens;
    calib_ = std::make_unique<PrefillCalib>(hals, path);
  }
  // Ends the profile session, if any, and hands each chunk's dispatch timestamps to the calibration.
  void Collect() {
    if (!gpu_.profiling()) return;
    gpu_.ProfileEnd();
    // Each command buffer of the session in launch order; a chunk's graph is the next one with its node count (stream
    // dispatches, e.g. the output head, come in command buffers of their own).
    std::map<std::uint64_t, std::vector<hrx_profile_dispatch_t>> buffers;
    for (const auto& e : session_) buffers[e.command_buffer_id].push_back(e);
    const auto before = calib_->Progress();
    auto chunk = session_chunks_.begin();
    for (auto& [id, ev] : buffers) {
      if (chunk == session_chunks_.end()) break;
      if (ev.size() != chunk->size()) continue;
      std::sort(ev.begin(), ev.end(), [](const auto& a, const auto& b) { return a.command_index < b.command_index; });
      std::vector<int> tags;
      std::vector<std::array<std::uint32_t, 3>> grids;
      for (const Node& nd : *chunk++) tags.push_back(nd.tag), grids.push_back(nd.grid);
      calib_->EndChunk(tags, grids, ev);
    }
    calib_->EndSession();
    const auto after = calib_->Progress();
    if (after != before)
      std::fprintf(stderr, "prefill calibration: %d GEMM buckets settled, %d open\n", after.first, after.second);
    session_.clear();
    session_chunks_.clear();
  }
  [[nodiscard]] const PrefillCalib* calibration() const { return calib_.get(); }

  // Final norm + output head of hidden() row `row` (this chunk) into dst (kVocab f32).
  void Head(std::uint32_t row, const hrx_buffer_ref_t& dst) {
    const auto* onw = Find("output_norm.weight");
    const auto* ow = Find("output.weight");
    Dispatch(Exe("rmsnorm.hal"), "yah_rmsnorm", 1, 1, 1, 32, 1, 1,
             {{hidden_->handle, std::size_t{row} * kHidden * 4, std::size_t{kHidden} * 4}, TRef(*onw), Ref(*normed_)});
    if (rot_) {
      // MQ4 head: the f32 norm row into the weights' basis (no awq), then gen_gemv's tile-free GEMV
      Dispatch(Exe("rot_head.hal"), "yah_rot256", kHidden / 1024, 1, 1, 64, 1, 1, {Ref(*normed_), Ref(*rot_signs_)});
      const Geom& g = geom_.at("gemv.hal");   // (rows per workgroup, lanes, workgroups)
      Dispatch(Exe("gemv.hal"), "yah_gemv", g.tt, 1, 1, g.rowgrp, 1, 1, {TRef(*ow), Ref(*normed_), dst});
      return;
    }
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
      case 44: *out = {"mq4", 256}; return true;
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
    if (split_armed_ && graph_ && !df_planning_ && !npu_planning_ && !warming_) SplitGraph();
    FlushKres();
    if (df_planning_ || npu_planning_ || warming_) return;
    const std::uint32_t ordinal = exe.OrdinalOrZero(name);
    const std::uint32_t ws = exe.WorkgroupSize(ordinal);
    const hrx_dispatch_config_t config = LoomDevice::Config(gx, gy, gz, ws ? ws : sx, sy, sz);
    if (graph_) {
      graph_->Dispatch(exe, ordinal, config, b.data(), b.size(), writes, df_after_.empty() ? nullptr : &df_after_,
                       also_writes, untracked);
      nodes_.push_back({{gx, gy, gz}, name, pending_tag_});
      pending_tag_ = -1;
    } else
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
    rot_ = geom_.count("rot") != 0;
    a4_ = geom_.count("a4") != 0;
    nqov_ = a4_ && geom_.count("nqov") != 0;
    dnov_ = a4_ && geom_.count("dnov") != 0;
    pnq_ab_ = dnov_ && geom_.count("pnq_abf.hal") != 0;
    guov_ = a4_ && geom_.count("guov") != 0;
    nqgu_ = guov_ && geom_.count("nqgu") != 0;
    rqk_ = nqov_ && geom_.count("rqk") != 0;
    nqw_ = a4_ && geom_.count("nqw") != 0;
    if (rot_) rot_signs_ = &table("rot_signs.bin", 512 * 4);
  }

  void AllocateBuffers() {
    const std::size_t B = B_;
    // Activations are token major, [token][row], with row stride = the K extent of the GEMM that reads them.
    kv_cache_ = std::size_t{T_} * kKvRow;
    hidden_ = &Alloc(B * kHidden * 4);
    reszero_ = &Alloc(B * kHidden * 4);
    sumout_ = &Alloc(B * kHidden * 4);
    scratch_ = &Alloc(B * kFfn * 2);
    if (rot_ && !a4_) rotbuf_ = &Alloc(B * kInner * 2);
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
    // the FFN norm's fragment-major copy for the afrag GEMMs (norm_t.hal), only in sets that have them
    bool any_af = false;
    for (const auto& [name, g] : geom_) any_af = any_af || name.size() > 7 && name.compare(name.size() - 7, 7, ".af.hal") == 0;
    if (any_af) normt_ = &Alloc(B * kHidden * 2);
    if (!a4_) gateffn_ = &Alloc(B * kFfn * 4);   // A4's gu GEMM keeps the gate in LDS
    // Per-workgroup weight staging (ABI only: the tile GEMMs stage in LDS) and epilogue scratch sized for the widest
    // GEMM, which the compiler declares over the whole [m_rows][tokens] tile.
    uwstage_ = &Alloc(std::size_t{kFfn} * 16 * 2);
    wstage_ = &Alloc(std::size_t{kFfn} * 16 * 2);
    ostage_ = &Alloc(std::size_t{kFfn} * B * 4);
    partial_ = &Alloc(B * kHidden * 4);
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
    for (const char* hal : {"norm.hal", "convkq.hal", "prepab.hal", "rowsplit.hal", "postnorm.hal", "unpack.hal",
                            "gemv.hal", "rmsnorm.hal", "argmax.hal", "accum.hal", "embed.hal"})
      Exe(hal);
    GeomOf("embed.hal");   // one token per workgroup: refuses a set emitted for another chunk size
    // embed.hal decodes IQ4_XS, or Q8_0 in the HFQ models' sets
    if (static_cast<std::uint32_t>(Find("token_embd.weight")->type) != (rot_ ? 8u : 23u))
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
  // The GEMM variant for this chunk's real tokens: the set may carry "<hal>.t128.hal" / ".t64.hal" with narrower token
  // tiles. Pick the least (padded token rows x cost per row); a narrow tile pads less but costs more per row (measured:
  // x1.12 at 128 tokens, x1.6 at 64). Every variant computes the same values.
  // A tuned set (engine/tune) carries the measured choice instead: "pick:<hal>:<max tokens>" rows map a token bucket to
  // the token tile to use; chunks above the largest bucket take the full tile.
  std::string PickGemm(const std::string& hal) {
    if (calib_ && graph_) {
      int tag = -1;
      std::string v = calib_->Choose(hal, &tag);
      pending_tag_ = tag;
      if (!v.empty()) return v;
    }
    const std::string stem = hal.substr(0, hal.size() - 4);
    const std::string prefix = "pick:" + hal + ":";
    bool tuned = false;
    std::uint32_t bucket = ~0u, bn = 0;
    for (auto it = geom_.lower_bound(prefix); it != geom_.end() && it->first.compare(0, prefix.size(), prefix) == 0;
         ++it) {
      tuned = true;
      if (it->second.tokens >= n_ && it->second.tokens < bucket) bucket = it->second.tokens, bn = it->second.rowgrp;
    }
    if (tuned) {
      const auto full = geom_.find(hal);
      if (bn == 0 || (full != geom_.end() && bn == full->second.tokens)) return hal;
      const std::string v = stem + ".t" + std::to_string(bn) + ".hal";
      return geom_.count(v) ? v : hal;
    }
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

  // The afrag form of a GEMM HAL ("<hal>.af.hal" / ".af.to.hal", emit_prefill_pp.py afrag_variants), or "".
  std::string AfHal(const std::string& hal, const char* suffix = ".af.hal") const {
    const std::string v = hal.substr(0, hal.size() - 4) + suffix;
    return geom_.count(v) ? v : "";
  }
  // The FFN: norm, gate, swiglu (up), down. A GEMM with an afrag form reads its input fragment-major: the norm writes
  // that layout into normt_ for gate / up (and the plain one into scratch_ if one of them has no afrag form), the swiglu
  // writes it into ffnup_ for down when both have afrag forms.
  // afrag GEMMs take 512-token tiles: a short chunk pads more with them than with the narrow tiles PickGemm would take
  // (300 tokens: 384 rows at x1.12, +2.4% with afrag). They run when their padded rows at ~0.93 the cost per row (their
  // clock-free gain) cost no more than the best of the full and the set's narrow tiles, PickGemm's untuned weights.
  bool AfChunk() const {
    const auto padded = [&](std::uint32_t bn) { return static_cast<double>((n_ + bn - 1) / bn * bn); };
    double best = padded(256);
    if (std::any_of(geom_.begin(), geom_.end(),
                    [](const auto& g) { return g.first.find(".t128.hal") != std::string::npos; }))
      best = std::min({best, padded(128) * 1.12, padded(64) * 1.6});
    return padded(512) * 0.93 <= best;
  }
  // Whether the GEMM of this kind on tensor wname runs its afrag form in this chunk.
  bool Af(const char* kind, const std::string& wname) const {
    Fmt f{};
    return AfChunk() && !AfHal(GemmHal(kind, *Find(wname), &f)).empty();
  }
  void RunFfn(const std::string& pre) {
    if (RunFfnBlock(pre)) return;
    if (ffn_normed_) throw LoomError("prefill: the token halves normed an FFN that does not run as the FFN block");
    if (RunFfnFused(pre)) return;
    const bool gate_af = Af("gemm_kstore", pre + "ffn_gate.weight");
    const bool up_af = Af("gemm_swiglu", pre + "ffn_up.weight");
    const bool down_af = up_af && Af("gemm_kres", pre + "ffn_down.weight");
    const std::uint32_t nn = trim_row_ == kAllRows && FfnNpuOk(pre) ? NpuRows("ffn") : 0;
    const std::string gs = nn ? NpuHal(KstoreHal(pre + "ffn_gate.weight", gate_af)) : "";
    const std::string us = nn ? NpuHal(SwigluHal(pre + "ffn_up.weight", up_af, down_af)) : "";
    nq_want_ = nqgu_ && nn == 0;
    RunNorm(pre + "post_attention_norm.weight",
            gate_af && up_af ? NormOut::kTiled : gate_af || up_af ? NormOut::kBoth : NormOut::kRow,
            !gs.empty() && !us.empty() ? "ffn" : nullptr);
    nq_want_ = false;
    if (!gs.empty() && !us.empty()) {
      // the NPU's rows of gate and up; the unpack applies silu(gate) * up
      std::vector<NpuView> ag, au;
      FfnEncode(pre, gate_af && up_af ? *normt_ : *scratch_, gate_af && up_af, ag, au);
      std::size_t w_off = 0, c_off = 0;
      const auto wg = NpuDecode("ffn", pre + "ffn_gate.weight", nn, w_off);
      const auto wu = NpuDecode("ffn", pre + "ffn_up.weight", nn, w_off);
      std::vector<std::uint32_t> calls;
      const NpuView cg = NpuCalls("ffn", ag, wg, c_off, calls, pre + "ffn_gate.weight");
      const NpuView cu = NpuCalls("ffn", au, wu, c_off, calls, pre + "ffn_up.weight");
      NpuEnqueue(std::move(calls), "ffn");
      RunKstoreSplit(pre + "ffn_gate.weight", *gateffn_, gate_af, gs, nn);
      RunSwigluSplit(pre + "ffn_up.weight", up_af, us, nn);
      NpuJoin();
      FfnUnpack(down_af, {cg.offset, cg.length + cu.length});
    } else if (a4_) {
      // one GEMM for gate and up: silu(gate) * up straight into ffnup_ (gen_gemm_iu4 gu); guov: flagged (a counter per
      // 1024 features and token tile), RunResidual's rotq waits on it
      const auto* up = Find(pre + "ffn_up.weight");
      if (guov_) gu_cnt_ = FlagSlots(kFfn / 1024);
      gu_flagged_ = RunIu4("gu", *Find(pre + "ffn_gate.weight"), {ffnup_}, {std::size_t(up->dims[1]) * 2}, 0b10000, up,
                           guov_ ? &gu_cnt_ : nullptr, &nq_cnt_);
      gu_waited_ = gu_flagged_ && nq_cnt_.buffer;
      nq_cnt_ = {};
    } else {
      RunKstore(pre + "ffn_gate.weight", *gateffn_, gate_af);
      RunSwiglu(pre + "ffn_up.weight", up_af, down_af);
    }
    RunResidual(pre + "ffn_down.weight", *ffnup_, down_af);
  }

  // The NPU's FFN block (dispatch.txt "npuffnblk <F>", emit_prefill_pp NPU_FFNBLK): the NPU owns the FFN's last F
  // features. Its gate / up calls (npu_ffnsw_<fmt>.xdna; decoder-column sets: 224 features a call, call j's column c
  // takes features 8 (128 c + 4 j) + [0, 32) of F; fused sets: 256, column c features 256 j + 32 c + [0, 32); gate rows
  // and the up rows one gate tensor further, straight from the model) write H = bfp16(silu(gate) * up) into C, [M block]
  // [pass][16-token slab][128 k-blocks]; its down calls (npu_ffndn<passes>_<fmt>.xdna, per K part of F (FfnBlkParts),
  // 560 / 640 rows each, a last 560-row call overlapping the one before) read H and write their partials after it. Each
  // is a job of one image family. The GPU runs the other features (the fused ffn's split) and down over them
  // (npuffnrem_<fmt>.hal, f32), then npu_unpack_ffnblk.hal writes hidden + that + the partials. Returns false if the
  // layer does not qualify.
  std::uint32_t FfnBlkF() const {
    const auto it = geom_.find("npuffnblk");
    return it == geom_.end() ? 0 : it->second.tokens;
  }
  // dispatch.txt "npuffnblk <F> <gate / up features a call> <down rows a call>" (0: a decoder-column set's 224, 560)
  bool FfnBlkFused() const { return geom_.at("npuffnblk").rowgrp == 256; }
  std::uint32_t FfnBlkRows() const { return geom_.at("npuffnblk").tt ? geom_.at("npuffnblk").tt : 560; }
  std::uint32_t FfnBlkCalls() const { return (kHidden + FfnBlkRows() - 1) / FfnBlkRows(); }   // down calls per K part
  // (first pass, passes) of down's K parts over F features: the fewest parts of at most 5 passes (a fused panel holds a
  // part), sizes balanced (emit_prefill_pp ffnblk_parts)
  static std::vector<std::pair<std::uint32_t, std::uint32_t>> FfnBlkParts(std::uint32_t F) {
    const std::uint32_t P = F / 1024, n = (P + 4) / 5;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> parts;
    for (std::uint32_t i = 0, p0 = 0; i < n; ++i) {
      parts.push_back({p0, P / n + (i < P % n)});
      p0 += parts.back().second;
    }
    return parts;
  }
  // An image's decoder swap image, the image itself where there is none (fused sets)
  static std::string SwapOf(const std::string& image) {
    const std::string swap = image + ".swap.xdna";
    return ::access(swap.c_str(), R_OK) == 0 ? swap : image + ".xdna";
  }
  // H's bytes in C (from 0), then down's partials from FfnBlkDown().
  std::size_t FfnBlkH() const { return std::size_t{B_ / 64} * (FfnBlkF() / 1024) * 4 * 128 * 144; }
  std::size_t FfnBlkDown() const { return (FfnBlkH() + 4095) / 4096 * 4096; }
  static int DecoderId(const char* fmt) {
    static const char* const kDecoders[] = {"iq4xs", "iq3s", "iq3xxs", "q4k", "q3k", "iq2xxs", "iq2xs"};
    for (int i = 0; i < 7; ++i)
      if (std::strcmp(kDecoders[i], fmt) == 0) return i;
    return -1;
  }
  // With norm (the token halves' probe): only whether the layer runs the block and its norm HAL stem; dispatches nothing.
  bool RunFfnBlock(const std::string& pre, const char** norm = nullptr) {
    if (!ffnblk_ || !npu_on_ || trim_row_ != kAllRows || !AfChunk()) return false;
    const auto* tg = Find(pre + "ffn_gate.weight");
    const auto* tu = Find(pre + "ffn_up.weight");
    const auto* td = Find(pre + "ffn_down.weight");
    // gate and up of one format: one image (its columns compute both); else column pairs (gen_npu_gemm ufmt)
    const bool pair = tg->type != tu->type;
    if (tg->dims != tu->dims || (!pair && tu->offset != tg->offset + tg->bytes) || (pair && !geom_.count("npuffnpair")))
      return false;
    Fmt f{}, fu{}, fd{};
    if (!FmtOf(static_cast<std::uint32_t>(tg->type), &f) || !FmtOf(static_cast<std::uint32_t>(tu->type), &fu) ||
        !FmtOf(static_cast<std::uint32_t>(td->type), &fd))
      return false;
    const std::uint32_t F = pair ? geom_.at("npuffnpair").tokens : ffnblk_, M = static_cast<std::uint32_t>(tg->dims[1]);
    // the GPU's features: the fused ffn's split, or (pairs) gate kstore + up swiglu splits, fragment-major into ffnup_
    const std::string hal = pair ? "" : AfHal(GemmHal("gemm_ffn", *tg, &f), ".af.to.hal");
    const std::string sh = pair ? "" : hal.empty() ? "" : NpuHal(hal);
    const std::string gs = pair ? NpuHal(KstoreHal(pre + "ffn_gate.weight", true)) : "";
    const std::string us = pair ? NpuHal(SwigluHal(pre + "ffn_up.weight", true, true)) : "";
    const std::uint32_t kbw = static_cast<std::uint32_t>(td->dims[0] - F) / 256;
    // the GPU's down over its features: f32 for npu_unpack_ffnblk, or (fused sets) the residual GEMM adding the NPU's
    const std::string tail = std::string(fd.name) + "_" + std::to_string(kbw) + ".hal";
    const bool res = geom_.count("npuffnres_" + tail) != 0;
    const std::string rem = (res ? "npuffnres_" : "npuffnrem_") + tail;
    // gate / up write H in the down decoder's k-block order (emit_prefill_pp FFNBLK_HORD: IQ4_XS's P4, Q4_K's; fused sets
    // P4 but Q4_K's)
    const bool q4k = std::strcmp(fd.name, "q4k") == 0;
    const std::string hord = q4k ? "_q4k" : std::strcmp(fd.name, "iq4xs") == 0 || FfnBlkFused() ? "_p4" : "";
    const std::string sw = dir_ + (pair ? "/npu_ffnsp_" + std::string(f.name) + "_" + fu.name : "/npu_ffnsw_" + std::string(f.name)) +
                           hord;
    const auto parts = FfnBlkParts(F);
    std::vector<std::string> dn;
    for (const auto& [p0, passes] : parts) {
      dn.push_back(dir_ + "/npu_ffndn" + std::to_string(passes) + "_" + fd.name);
      if (::access((dn.back() + ".xdna").c_str(), R_OK)) return false;
    }
    if ((pair ? gs.empty() || us.empty() || !Af("gemm_kstore", pre + "ffn_gate.weight") ||
                    !Af("gemm_swiglu", pre + "ffn_up.weight")
              : sh.empty()) ||
        !geom_.count(rem) || ::access((dir_ + "/norm_t_bfp_g1.hal").c_str(), R_OK) ||
        ::access((sw + ".xdna").c_str(), R_OK))
      return false;
    // the norm also writes gate / up's activations (one replay group: that stream layout; "npuffndfr_<fmt>": the
    // deferred-epilogue image's K split, emit_prefill_pp NPU_DFR_KS)
    const bool dfr = !pair && geom_.count("npuffndfr_" + std::string(f.name));
    if (dfr && ::access((dir_ + "/norm_t_bfp_g1s.hal").c_str(), R_OK)) return false;
    if (norm) {
      *norm = dfr ? "norm_t_bfp_g1s" : "norm_t_bfp_g1";
      return true;
    }
    if (!std::exchange(ffn_normed_, false))  // (the token halves ran it: FfnHalfNorm)
      RunNorm(pre + "post_attention_norm.weight", NormOut::kTiled, dfr ? "ffnblk_s" : "ffnblk");
    const NpuView ag{0, NpuK(5120).a};
    const std::uint8_t* base = gguf_.tensor_data_base();
    const std::size_t rb = static_cast<std::size_t>(tg->bytes) / M, blk = rb * 256 / tg->dims[0];
    const std::size_t rbu = static_cast<std::size_t>(tu->bytes) / M, blku = rbu * 256 / tu->dims[0];
    const std::size_t rbd = static_cast<std::size_t>(td->bytes) / td->dims[1], blkd = rbd * 256 / td->dims[0];
    const std::size_t f0 = M - F, dgu = static_cast<std::size_t>(tg->bytes);
    if (npu_planning_) {   // the NPU's gate / up rows (one range when up follows gate), all of down
      const std::size_t end = gguf_.tensor_data_size();
      if (pair) {
        npu_->RegisterRaw(base + tg->offset + f0 * rb, std::min(tg->offset + M * rb + 4096, end) - (tg->offset + f0 * rb));
        npu_->RegisterRaw(base + tu->offset + f0 * rbu, std::min(tu->offset + M * rbu + 4096, end) - (tu->offset + f0 * rbu));
      } else {
        const std::size_t a = tg->offset + f0 * rb;
        npu_->RegisterRaw(base + a, std::min(tu->offset + M * rb + 4096, end) - a);
      }
      npu_->RegisterRaw(base + td->offset, std::min<std::size_t>(td->bytes + 4096, end - td->offset));
    }
    std::vector<std::uint32_t> csw;
    std::vector<std::vector<std::uint32_t>> cdn(parts.size());
    if (pair && FfnBlkFused()) {   // F / 256 calls: call j's pair u (gate column 2 u, up column 2 u + 1) takes features
                                   // 256 j + 64 u + [0, 64), its records at H k-blocks 32 j + 8 u .. (gen_npu_gemm swcol 64)
      const NpuBytes bs = NpuBytesOf("npubytes_ffnsp");
      for (std::uint32_t j = 0; j < F / 256; ++j) {
        const std::size_t r0 = f0 + 256 * j;
        csw.push_back(npu_->BindRawPair(
            sw + ".xdna", SwapOf(sw), 16 + 8 * DecoderId(f.name) + DecoderId(fu.name), ag,
            base + tg->offset + r0 * rb, 255 * rb + 16 * blk + kDcolRecordRow,
            base + tu->offset + r0 * rbu, 255 * rbu + 16 * blku + kDcolRecordRow,
            {(j / 4) * 4 * 128 * 144 + (j % 4) * 32 * 144, bs.c}, 4 * (B_ / 64) * 4 * 1152,
            "ffnsp_" + std::string(f.name) + "_" + fu.name + hord));
      }
    } else if (pair) {   // 32 calls of 192 features: call j's up column u writes passes 2 u + j / 16, k-blocks 8 (j % 16) ..
      const NpuBytes bs = NpuBytesOf("npubytes_ffnsp");
      for (std::uint32_t j = 0; j < F / 192; ++j) {
        const std::size_t r0 = f0 + 1024 * (j / 16) + 64 * (j % 16);
        csw.push_back(npu_->BindRawPair(
            sw + ".xdna", SwapOf(sw), 16 + 8 * DecoderId(f.name) + DecoderId(fu.name), ag,
            base + tg->offset + r0 * rb, (2 * 2048 + 63) * rb + 16 * blk + kDcolRecordRow,
            base + tu->offset + r0 * rbu, (2 * 2048 + 63) * rbu + 16 * blku + kDcolRecordRow,
            {(j / 16) * 4 * 128 * 144 + (j % 16) * 8 * 144, bs.c}, 3 * (B_ / 64) * 4 * 1152,
            "ffnsp_" + std::string(f.name) + "_" + fu.name + hord));
      }
    } else if (FfnBlkFused()) {   // F / 256 calls: call j's column c takes features 256 j + 32 c + [0, 32), its records
                                  // at H k-blocks 32 j + 4 c .. (gen_npu_gemm swcol 32)
      const NpuBytes bs = NpuBytesOf("npubytes_ffnsw");
      for (std::uint32_t j = 0; j < F / 256; ++j)
        csw.push_back(npu_->BindRaw(sw + ".xdna", SwapOf(sw), DecoderId(f.name), ag,
                                    base + tg->offset + (f0 + 256 * j) * rb, dgu + 255 * rb + 16 * blk + kDcolRecordRow,
                                    {(j / 4) * 4 * 128 * 144 + (j % 4) * 32 * 144, bs.c}, "ffnsw" + hord));
    } else {   // 32 calls of 224 features: call j's column c takes features 8 (128 c + 4 j) + [0, 32); Q4_K's order
               // (gen_npu_gemm hord): its two 16-row slabs 32 rows apart from 64 (j / 2) + 16 (j % 2)
      const NpuBytes bs = NpuBytesOf("npubytes_ffnsw");
      for (std::uint32_t j = 0; j < F / 224; ++j)
        csw.push_back(npu_->BindRaw(sw + ".xdna", SwapOf(sw), DecoderId(f.name), ag,
                                    base + tg->offset + (f0 + (q4k ? 64 * (j / 2) + 16 * (j % 2) : 32 * j)) * rb,
                                    dgu + (6 * 1024 + (q4k ? 47 : 31)) * rb + 16 * blk + kDcolRecordRow, {4 * j * 144, bs.c},
                                    "ffnsw" + hord));
    }
    // down: per K part, FfnBlkCalls() calls of FfnBlkRows() rows (a last one ending at the last row)
    const std::uint32_t rows = FfnBlkRows(), calls = FfnBlkCalls();
    const std::size_t doff = FfnBlkDown(), cp = NpuBytesOf("npubytes_ffndn" + std::to_string(parts[0].second)).c;
    for (std::size_t ci = 0; ci < parts.size(); ++ci) {
      const auto [p0, passes] = parts[ci];
      const std::string fam = "ffndn" + std::to_string(passes);
      const NpuBytes bd = NpuBytesOf("npubytes_" + fam);
      if (bd.c != cp) throw LoomError("prefill: the FFN block's down parts differ in C bytes");
      for (std::uint32_t r = 0; r < calls; ++r) {
        const std::size_t r0 = std::min<std::size_t>(r * rows, td->dims[1] - rows), k0 = f0 / 256 + 4 * p0;
        cdn[ci].push_back(npu_->BindRaw(dn[ci] + ".xdna", SwapOf(dn[ci]), DecoderId(fd.name),
                                        {std::size_t{p0} * 4 * 128 * 144, bd.a, true},
                                        base + td->offset + r0 * rbd + k0 * blkd,
                                        (rows - 1) * rbd + 16 * blkd + kDcolRecordRow,
                                        {doff + (ci * calls + r) * cp, cp}, fam));
      }
    }
    NpuEnqueue(std::move(csw), pair ? "ffnsp" : "ffnsw");
    for (std::size_t ci = 0; ci < parts.size(); ++ci) NpuEnqueue(std::move(cdn[ci]), "ffndn" + std::to_string(parts[ci].second));
    if (pair) {
      RunKstoreSplit(pre + "ffn_gate.weight", *gateffn_, true, gs, F);
      RunSwigluSplit(pre + "ffn_up.weight", true, us, F);
    } else {
      const Geom g = GeomOf(hal);
      auto b = GemmWeights(*tg, f);
      b[0].length = static_cast<std::size_t>(tg->bytes + tu->bytes);
      const std::size_t first = b.size();
      for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{normt_, wstage_, ostage_, ffnup_}) b.push_back(Ref(*x));
      const std::uint32_t tt = Trim(b, first, {std::size_t(tg->dims[0]) * 2, 0, 0, std::size_t(M) * 2}, g);
      Dispatch(Exe(sh), ("yah_ffn_gemm_" + std::string(f.name) + "_ffn").c_str(), (M - F) / 16 / GeomOf(sh).rowgrp, tt, 1,
               32, 1, 1, b, GemmWrites(b, f, {ffnup_}));
    }
    if (res) {   // down over them as the residual GEMM, its epilogue adding the NPU's partials once the job is done
      const Geom& gr = geom_.at(rem);
      auto r = GemmWeights(*td, fd);
      for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{ffnup_, hidden_, wstage_, ostage_, hidden2_})
        r.push_back(Ref(*x));
      const std::size_t out = r.size() - 1;
      r.push_back({npu_->C().handle, doff, parts.size() * calls * cp});
      if (!npu_planning_) {
        const NpuWait w = NpuJoinBegin();
        r.insert(r.end(), w.words.begin(), w.words.end());
        Dispatch(Exe(rem), ("yah_ffn_gemm_" + std::string(fd.name) + "_kres").c_str(), MTiles(*td) / gr.rowgrp, gr.tt,
                 1, 32, 1, 1, r, (std::uint64_t{1} << out) | (std::uint64_t{1} << (r.size() - 1)), &w.npu_side);
        NpuJoinEnd();
      }
      std::swap(hidden_, hidden2_);
      return true;
    }
    // down over them: f32 [tokens][5120]
    const Geom& gr = geom_.at(rem);
    auto r = GemmWeights(*td, fd);
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{ffnup_, wstage_, ostage_}) r.push_back(Ref(*x));
    r.push_back({ffnrem_->handle, 0, std::size_t{B_} * kHidden * 4});
    Dispatch(Exe(rem), ("yah_ffn_gemm_" + std::string(fd.name)).c_str(), MTiles(*td) / gr.rowgrp, gr.tt, 1, 32, 1, 1, r,
             std::uint64_t{1} << (r.size() - 1));
    NpuJoin();
    // with token halves the next layer runs it, per half (FfnUnpackPending)
    ffn_pending_ = {{doff, parts.size() * calls * cp}, hidden_, hidden2_};
    if (!halves_ || !geom_.count("npu_unpack_ffnblk_h1.hal")) FfnUnpackPending(-1);
    std::swap(hidden_, hidden2_);
    return true;
  }

  // The ffn site's NPU rows need both matrices on the NPU (the unpack applies silu(gate) * up).
  bool FfnNpuOk(const std::string& pre) const { return NpuOk(pre + "ffn_gate.weight") && NpuOk(pre + "ffn_up.weight"); }
  // The ffn site's activations for gate (ag) and up (au), each in its decoder's k-block order.
  void FfnEncode(const std::string& pre, const LoomBuffer& input, bool tiled, std::vector<NpuView>& ag,
                 std::vector<NpuView>& au) {
    const bool kg = NpuPk(pre + "ffn_gate.weight"), ku = NpuPk(pre + "ffn_up.weight");
    std::vector<NpuView> a4, ak;
    if (!kg || !ku) a4 = NpuEncode("ffn", input, tiled);
    if (kg || ku) ak = NpuEncode("ffn", input, tiled, true);
    ag = kg ? ak : a4, au = ku ? ak : a4;
  }
  // kRow: f16 row-major into scratch_; kTiled: fragment-major into normt_ (afrag GEMMs); kBoth: both, one pass.
  enum class NormOut { kRow, kTiled, kBoth };
  // The FFN with ffn_gate and ffn_up in one afrag GEMM ("gemm_ffn_<fmt>_..", emit_prefill_pp.py ffn_fused): both of one
  // format and ffn_up stored right after ffn_gate, so one binding spans them (the kernel reads up row r at m_rows + r).
  // It writes f16(silu(gate) * up) into ffnup_, fragment-major when ffn_down is afrag too. Returns false if not taken.
  bool RunFfnFused(const std::string& pre) {
    if (!AfChunk()) return false;
    const auto* tg = Find(pre + "ffn_gate.weight");
    const auto* tu = Find(pre + "ffn_up.weight");
    if (tg->type != tu->type || tg->dims != tu->dims || tu->offset != tg->offset + tg->bytes) return false;
    Fmt f{};
    const std::string base = GemmHal("gemm_ffn", *tg, &f);
    const bool down_af = Af("gemm_kres", pre + "ffn_down.weight");
    const std::string hal = AfHal(base, down_af ? ".af.to.hal" : ".af.hal");
    if (hal.empty()) return false;
    RunNorm(pre + "post_attention_norm.weight", NormOut::kTiled,
            trim_row_ == kAllRows && FfnNpuOk(pre) && NpuRows("ffn") && !NpuHal(hal).empty() ? "ffn" : nullptr);
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*tg, f);
    b[0].length = static_cast<std::size_t>(tg->bytes + tu->bytes);
    const std::size_t first = b.size();
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{normt_, wstage_, ostage_, ffnup_}) b.push_back(Ref(*x));
    const std::uint32_t tt = Trim(b, first, {std::size_t(tg->dims[0]) * 2, 0, 0, std::size_t(tg->dims[1]) * 2}, g);
    const std::uint32_t nn = tt == g.tt && FfnNpuOk(pre) ? NpuRows("ffn") : 0;
    const std::string sh = nn ? NpuHal(hal) : "";
    if (!sh.empty()) {
      // the NPU's rows of gate and up (the weights' last rows of each tensor); the unpack applies silu(gate) * up
      std::vector<NpuView> ag, au;
      FfnEncode(pre, *normt_, true, ag, au);
      std::size_t w_off = 0, c_off = 0;
      const auto wg = NpuDecode("ffn", pre + "ffn_gate.weight", nn, w_off);
      const auto wu = NpuDecode("ffn", pre + "ffn_up.weight", nn, w_off);
      std::vector<std::uint32_t> calls;
      const NpuView cg = NpuCalls("ffn", ag, wg, c_off, calls, pre + "ffn_gate.weight");
      const NpuView cu = NpuCalls("ffn", au, wu, c_off, calls, pre + "ffn_up.weight");
      NpuEnqueue(std::move(calls), "ffn");
      Dispatch(Exe(sh), ("yah_ffn_gemm_" + std::string(f.name) + "_ffn").c_str(),
               (MTiles(*tg) - nn / 16) / GeomOf(sh).rowgrp, tt, 1, 32, 1, 1, b, GemmWrites(b, f, {ffnup_}));
      NpuJoin();
      FfnUnpack(down_af, {cg.offset, cg.length + cu.length});
    } else {
      Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name) + "_ffn").c_str(), MTiles(*tg) / g.rowgrp, tt, 1, 32, 1,
               1, b, GemmWrites(b, f, {ffnup_}));
    }
    RunResidual(pre + "ffn_down.weight", *ffnup_, down_af);
    return true;
  }

  // npu_site: the NPU split of a K = 5120 site reads this norm's output; with "npunormbfp" in the set the norm also
  // writes its BFP16 input into A ("<norm>_bfp.hal") and NpuEncode(npu_site) skips the encoder.
  void RunNorm(const std::string& wname, NormOut mode = NormOut::kRow, const char* npu_site = nullptr) {
    // A row takes `split` waves (dispatch.txt norm_split, else 1); a workgroup of w waves takes w / split rows.
    // npu_site "ffnblk": the FFN block's one-replay-group stream (RunFfnBlock, fragment-major output only); "ffnblk_s":
    // in the deferred-epilogue images' K split
    const bool g1s = npu_site && std::strcmp(npu_site, "ffnblk_s") == 0;
    const bool g1 = g1s || (npu_site && std::strcmp(npu_site, "ffnblk") == 0);
    const bool bfp = npu_site && (g1 || geom_.count("npunormbfp"));
    norm_bfp_ = bfp && !g1 ? npu_site : "";
    const std::string hal = mode == NormOut::kTiled ? "norm_t" : mode == NormOut::kBoth ? "norm_rt" : "norm";
    const LoomExecutable& exe = Exe(g1s ? "norm_t_bfp_g1s.hal" : g1 ? "norm_t_bfp_g1.hal" : hal + (bfp ? "_bfp.hal" : ".hal"));
    const std::uint32_t ws = exe.WorkgroupSize(exe.OrdinalOrZero("yah_half_norm"));
    const auto ns = geom_.find("norm_split");
    const std::uint32_t split = ns != geom_.end() && ns->second.rowgrp ? ns->second.rowgrp : 1;
    const std::uint32_t rows_per_wg = ws ? ws / 32 / split : 1;
    if (!rows_per_wg || (ws && ws % (32 * split))) throw LoomError("norm.hal: workgroup size does not match norm_split");
    if (B_ % rows_per_wg) throw LoomError("norm.hal: rows per workgroup must divide the chunk");
    DumpNorm(wname);
    if (a4_ && mode == NormOut::kRow && !bfp && geom_.count("normq_5120.hal")) {
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
          Dispatch(Exe(want ? "normqfo_5120.hal" : "normqf_5120.hal"), "yah_normq", B_, 1, 1,
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
      Dispatch(Exe(want ? "normqo_5120.hal" : "normq_5120.hal"), "yah_normq", B_, 1, 1,
               geom_.at("normq_5120.hal").rowgrp, 1, 1, b, want ? 0b1110000 : 0b110000, nullptr, want ? 0b1000000 : 0);
      return;
    }
    std::vector<hrx_buffer_ref_t> b =
        mode == NormOut::kBoth
            ? std::vector<hrx_buffer_ref_t>{Ref(*hidden_), Ref(*reszero_), TRef(*Find(wname)), Ref(*sumout_),
                                            Ref(*scratch_), Ref(*normt_)}
            : std::vector<hrx_buffer_ref_t>{Ref(*hidden_), Ref(*reszero_), TRef(*Find(wname)), Ref(*sumout_),
                                            Ref(mode == NormOut::kTiled ? *normt_ : *scratch_)};
    if (bfp) b.push_back({npu_->A().handle, 0, NpuK(5120).a});
    Dispatch(exe, "yah_half_norm", B_ / rows_per_wg, 1, 1, 32, 1, 1, b);
    if (rot_) {
      if (mode != NormOut::kRow || bfp) throw LoomError("prefill: MQ4 sets rotate only the row-major norm output");
      if (a4_)
        RotQ(*scratch_, kHidden, NormAwq(wname), false);
      else
        Rotate(*scratch_, nullptr, kHidden, NormAwq(wname));
    }
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
      Dispatch(Exe(counts ? "rotqfo_17408.hal" : "rotqf_17408.hal"), "yah_rotq256", g.tt, B_, 1, g.rowgrp, 1, 1, b,
               counts ? 0b1011000 : 0b11000, nullptr, (chain ? 0b111001 : 0b100001) | (counts ? 0b1000000 : 0));
      std::swap(aq_, aq2_), std::swap(ad_, ad2_);
      return chain && counts;
    }
    const std::string ohal = "rotqo_" + std::to_string(k) + ".hal";
    const bool counts = ocnt && !heads && k == 6144;
    std::vector<hrx_buffer_ref_t> b = {Ref(x), TRef(*Find(awq)), Ref(*rot_signs_), Ref(*aq_), Ref(*ad_)};
    if (counts) b.push_back(*ocnt = FlagSlot());
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
  // word per 128-token tile, a slot per pair in the chunk, zeroed before the chunk (RunLayers). Both bind it untracked.
  static constexpr std::size_t kFlagSlots = 2048;
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
  // Debug (YAH_DUMP_ACT=<prefix>, YAH_DUMP_LAYERS=l0,l1,..): the row-major f16 output of each RMSNorm that feeds a GEMM
  // in those layers (n_ rows x 5120) to <prefix>_<norm weight name>.f16, from an extra norm.hal pass into its own buffer.
  bool DumpOn() const { return std::getenv("YAH_DUMP_ACT") != nullptr; }
  void DumpNorm(const std::string& wname) {
    const char* prefix = std::getenv("YAH_DUMP_ACT");
    if (!prefix || wname.rfind("blk.", 0) != 0) return;
    const std::string layer = wname.substr(4, wname.find('.', 4) - 4);
    const std::string want = std::string(",") + (std::getenv("YAH_DUMP_LAYERS") ? std::getenv("YAH_DUMP_LAYERS") : "") + ",";
    if (want.find("," + layer + ",") == std::string::npos) return;
    if (!dumpbuf_) dumpbuf_ = &Alloc(std::size_t{B_} * kHidden * 2);
    const LoomExecutable& exe = Exe("norm.hal");
    const std::uint32_t ws = exe.WorkgroupSize(exe.OrdinalOrZero("yah_half_norm"));
    const auto ns = geom_.find("norm_split");
    const std::uint32_t split = ns != geom_.end() && ns->second.rowgrp ? ns->second.rowgrp : 1;
    const std::uint32_t rows_per_wg = ws ? ws / 32 / split : 1;
    Dispatch(exe, "yah_half_norm", B_ / rows_per_wg, 1, 1, 32, 1, 1,
             {Ref(*hidden_), Ref(*reszero_), TRef(*Find(wname)), Ref(*sumout_), Ref(*dumpbuf_)});
    gpu_.Synchronize();
    std::vector<std::uint16_t> host(std::size_t{n_} * kHidden);
    gpu_.D2H(*dumpbuf_, host.data(), host.size() * 2);
    std::ofstream(std::string(prefix) + "_" + wname + ".f16", std::ios::binary)
        .write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(host.size() * 2));
  }
  LoomBuffer* dumpbuf_ = nullptr;

  // Decode-free GEMMs (emit_prefill_pp.py decode_free): "dq_<fmt>_<mt>_<kb>.hal" decodes a GEMM's weights to f16 scratch
  // once per chunk with the tile GEMM's own decode (the same f16 values), "gemm_<kind>_f16_<mt>_<kb>.hal" multiplies them
  // without decode. The pass for the next GEMM is recorded right after each GEMM: no edge between them, so they share a
  // graph segment and run together; the barrier before the GEMM's consumer drains both. Three scratch slots rotate, so a
  // slot is rewritten only after the GEMM two before has drained. Chunks of at most kDfMinTokens real tokens keep the
  // fused decode: one token tile decodes each weight once anyway.
  static constexpr std::uint32_t kDfMinTokens = 512;
  struct DfStep {
    std::string dq, gemm;
    const core::TensorInfo* t;
    Fmt f;
  };
  std::vector<DfStep> df_plan_;
  bool df_planned_ = false, df_planning_ = false, df_on_ = false;
  std::size_t df_k_ = 0;
  LoomBuffer* df_w_[3] = {nullptr, nullptr, nullptr};
  std::vector<hrx_buffer_ref_t> df_after_;

  // Records the GEMMs that have a decode-free path, in dispatch order (a dry run: Dispatch records nothing), and sizes
  // the scratch for the largest.
  void PlanDecodeFree(std::uint32_t ci) {
    df_planned_ = true;
    bool any = false;
    for (const auto& [name, g] : geom_) any = any || name.rfind("dq_", 0) == 0;
    if (!any) return;
    LoomBuffer* h = hidden_;
    LoomBuffer* h2 = hidden2_;
    df_planning_ = true;
    RecordLayers(ci, {});
    df_planning_ = false;
    hidden_ = h, hidden2_ = h2;
    std::size_t most = 0;
    for (const DfStep& s : df_plan_) most = std::max<std::size_t>(most, std::size_t{s.t->dims[1]} * s.t->dims[0] * 2);
    for (auto& w : df_w_) w = &Alloc(most);
  }
  // The f16 GEMM to run for this call, or "" for the fused path. Planning: records the call.
  std::string DfGemm(const char* kind, const core::TensorInfo& t, const Fmt& f) {
    const std::uint32_t mt = MTiles(t);
    const std::string dq = "dq_" + std::string(f.name) + "_" + std::to_string(mt) + "_" +
                           std::to_string(t.dims[0] / f.qk) + ".hal";
    const std::string gemm = "gemm_" + std::string(kind) + "_f16_" + std::to_string(mt) + "_" +
                             std::to_string(t.dims[0] / 256) + ".hal";
    if (!geom_.count(dq) || !geom_.count(gemm)) return "";
    if (df_planning_) {
      df_plan_.push_back({dq, gemm, &t, f});
      return "";
    }
    if (!df_on_) return "";
    if (df_k_ >= df_plan_.size() || df_plan_[df_k_].t != &t) throw LoomError("prefill: decode-free plan out of step");
    return gemm;
  }
  // Decode plan step k's weights into its scratch slot. The dequant is persistent: its row's first column is the
  // workgroup count (one per WGP), and its workgroups walk the tensor.
  void DispatchDequant(std::size_t k) {
    const DfStep& s = df_plan_[k];
    const Geom g = geom_.at(s.dq);
    auto b = GemmWeights(*s.t, s.f);
    b.push_back(Ref(*df_w_[k % 3]));
    Dispatch(Exe(s.dq), ("yah_dequant_" + std::string(s.f.name)).c_str(), g.tokens, 1, 1, 256, 1, 1, b,
             std::uint64_t{1} << (b.size() - 1));
  }
  // The weight binding of the current decode-free GEMM.
  std::vector<hrx_buffer_ref_t> DfWeights() const { return {Ref(*df_w_[df_k_ % 3])}; }
  // Before a decode-free GEMM: the next one's weights. Recorded first, so its few workgroups launch before the GEMM's
  // and run beside them (a dispatch's workgroups only launch once the previous dispatch's have all launched).
  // gemm: the GEMM's bindings; the dequant is ordered after its inputs (not its weights) so that the barrier the GEMM
  // needs falls before the dequant and the two share a graph segment.
  void DfAhead(const std::vector<hrx_buffer_ref_t>& gemm) {
    if (df_k_ + 1 >= df_plan_.size()) return;
    df_after_.assign(gemm.begin() + 1, gemm.end());
    DispatchDequant(df_k_ + 1);
    df_after_.clear();
  }
  void DfNext() { ++df_k_; }

  // NPU column split (EnableNpu, emit_prefill_pp.npu_split). A site's NPU rows in this chunk (0: GPU only).
  std::uint32_t NpuRows(const char* site) const {
    const auto it = npu_rows_.find(site);
    return npu_on_ && it != npu_rows_.end() ? it->second : 0;
  }
  // (k offset, K) of a site's NPU calls: one K, or down's 6 + 6 + 5 passes of K = 17408 (its weight panel does not fit
  // a memory tile).
  // The NPU runs one image (K = 5120 per call; an image switch costs ~0.55 ms): a site's K in chunks of 5120, the rest
  // of K (out: 1024, down: 2048) on the GPU ("npurem_<site>_<fmt>.hal"), added by the unpack.
  static std::uint32_t NpuSiteK(const std::string& site) { return site == "down" ? 17408 : site == "out" ? 6144 : 5120; }
  static std::vector<std::pair<std::uint32_t, std::uint32_t>> NpuChunks(const std::string& site) {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> v;
    for (std::uint32_t k = 0; k + 5120 <= NpuSiteK(site); k += 5120) v.push_back({k, 5120});
    return v;
  }
  static std::uint32_t NpuRem(const std::string& site) { return NpuSiteK(site) % 5120; }
  // Chunked sites name their encoders and decoders per chunk ("_c<i>").
  static bool NpuChunked(const std::string& site) { return NpuChunks(site).size() > 1 || NpuRem(site) != 0; }
  // Bytes per NPU call of a K (dispatch.txt "npubytes_<K>": activations of the chunk, a weight panel, a C panel).
  struct NpuBytes {
    std::size_t a, w, c;
  };
  NpuBytes NpuBytesOf(const std::string& name) const {
    const Geom& g = geom_.at(name);
    return {g.tokens, g.rowgrp, g.tt};
  }
  NpuBytes NpuK(std::uint32_t K) const {
    const Geom& g = geom_.at("npubytes_" + std::to_string(K));
    return {g.tokens, g.rowgrp, g.tt};
  }
  // Output rows per NPU call (dispatch.txt "npurows": 8 columns of gen_npu_gemm.TN).
  std::uint32_t NpuCallRows() const { return geom_.at("npurows").tokens; }
  // The GPU's share of GEMM HAL hal ("<hal>.npu.hal": its leading rows at the full stride), or "".
  std::string NpuHal(const std::string& hal) const {
    const std::string v = hal.empty() ? "" : hal.substr(0, hal.size() - 4) + ".npu.hal";
    return geom_.count(v) ? v : "";
  }
  // The kstore HAL a GEMM on wname runs (afrag form with af).
  std::string KstoreHal(const std::string& wname, bool af) const {
    Fmt f{};
    const std::string base = GemmHal("gemm_kstore", *Find(wname), &f);
    return af ? AfHal(base) : base;
  }
  // Decoder-column sets ("npudcol"): the NPU decodes a tensor of a format with a decoder itself, from its raw rows
  // ("npu_dcol_<fmt>_<K>.xdna"; "" for other formats: that site stays on the GPU in that layer). dec: the format's
  // decoder id (the images' swaps).
  std::string DcolImage(const std::string& wname, int* dec = nullptr, bool half = false) const {
    Fmt f{};
    if (!FmtOf(static_cast<std::uint32_t>(Find(wname)->type), &f)) return "";
    static const char* const kDecoders[] = {"iq4xs", "iq3s", "iq3xxs", "q4k", "q3k", "iq2xxs", "iq2xs"};
    const auto it = std::find_if(std::begin(kDecoders), std::end(kDecoders),
                                 [&](const char* d) { return std::strcmp(d, f.name) == 0; });
    if (it == std::end(kDecoders)) return "";
    if (dec) *dec = static_cast<int>(it - std::begin(kDecoders));
    const std::string path =
        (half ? hdir_ : dir_) + "/npu_dcol_" + f.name + "_" + std::to_string(Find(wname)->dims[0]) + ".xdna";
    return ::access(path.c_str(), R_OK) == 0 ? path : "";
  }
  // The NPU can take wname's rows (a decoder-column set needs its format's decoder).
  bool NpuOk(const std::string& wname) const { return !npu_dcol_ || !DcolImage(wname).empty(); }
  // wname's decoder pushes Q4_K's k-block order (the activations' "pk" stream, NpuEncode pk).
  bool NpuPk(const std::string& wname) const {
    Fmt f{};
    return npu_dcol_ && FmtOf(static_cast<std::uint32_t>(Find(wname)->type), &f) && std::strcmp(f.name, "q4k") == 0;
  }
  // input (row-major, or fragment-major with tiled) encoded into A for each of the site's K chunks; their A views.
  // pk (decoder-column sets): the Q4_K k-block order ("_pk" encoders, A's second half; the fused writers write the
  // p4 order of the other decoders).
  std::vector<NpuView> NpuEncode(const std::string& site, const LoomBuffer& input, bool tiled, bool pk = false) {
    const auto chunks = NpuChunks(site);
    // down after an ffn unpack that wrote its columns (FfnUnpack): only each chunk's columns before them
    const bool from_ffn = site == "down" && ffn_bfp_ && !pk;
    ffn_bfp_ = false;
    // a K = 5120 site after a norm that wrote its input (RunNorm npu_site)
    const bool from_norm = norm_bfp_ == site && !pk;
    norm_bfp_.clear();
    const std::uint32_t ffn_cols = from_ffn ? geom_.at("npuffnbfp").tokens : NpuSiteK(site);
    std::vector<NpuView> views;
    std::size_t off = pk ? npu_->A().size / 2 : 0;
    for (std::size_t c = 0; c < chunks.size(); ++c) {
      const std::size_t bytes = NpuK(chunks[c].second).a;
      const auto [k0, K] = chunks[c];
      const std::uint32_t cover = ffn_cols > k0 ? std::min(K, ffn_cols - k0) : 0;
      if (cover && !from_norm) {
        const std::string hal = "npu_enc_" + std::to_string(NpuSiteK(site)) + (NpuChunked(site) ? "_c" + std::to_string(c) : "") +
                                (cover < K ? "p" : "") + (tiled ? "_t" : "") + (pk ? "_pk" : "") + ".hal";
        const Geom& g = geom_.at(hal);
        Dispatch(Exe(hal), "yah_bfp16_encode_act", g.tokens, 1, 1, g.rowgrp, 1, 1,
                 {Ref(input), {npu_->A().handle, off, bytes}}, 2);
      }
      views.push_back({off, bytes});
      off += bytes;
    }
    return views;
  }
  // The ffn site's unpack: f16(silu(gate) * up) into ffnup_ (fragment-major for an afrag down). With "npuffnbfp" (down
  // split too) it also writes down's BFP16 input for the ffn's NPU columns into A at down's chunk offsets, and
  // NpuEncode("down") encodes only the columns before them.
  void FfnUnpack(bool tiled, NpuView c) {
    std::vector<hrx_buffer_ref_t> rest = {Ref(*ffnup_)};
    std::uint64_t writes = 2;
    ffn_bfp_ = geom_.count("npuffnbfp") && NpuRows("down");
    if (ffn_bfp_) {
      std::size_t bytes = 0;
      for (const auto& [k0, K] : NpuChunks("down")) bytes += NpuK(K).a;
      rest.push_back({npu_->A().handle, 0, bytes});
      writes |= 4;
    }
    NpuUnpack(tiled ? "npu_unpack_ffn_t.hal" : "npu_unpack_ffn.hal", c, rest, writes);
  }
  // The NPU's rows of wname (its last rows) decoded straight into W panels from w_off within this job's W slot
  // ("dqbfp_<fmt>_<mt>_<kb>[_c<i>]", dispatch.txt: <workgroups> <workgroup size>); the W views per chunk and call.
  // The decode reads only weights, so it is recorded into the previous job's GPU segment, where it runs beside that
  // segment's GEMM while the NPU works; the two slots alternate per job.
  std::vector<std::vector<NpuView>> NpuDecode(const std::string& site, const std::string& wname, std::uint32_t rows,
                                              std::size_t& w_off) {
    if (npu_dcol_) return NpuRawRows(site, wname, rows);
    const std::size_t slot = (npu_job_ % 2) * (npu_->W().size / 2);
    const auto* t = Find(wname);
    Fmt f{};
    const std::string base = GemmHal("gemm_kstore", *t, &f);
    const auto chunks = NpuChunks(site);
    const std::size_t row_bytes = static_cast<std::size_t>(t->bytes) / static_cast<std::size_t>(t->dims[1]);
    std::vector<std::vector<NpuView>> views;
    for (std::size_t c = 0; c < chunks.size(); ++c) {
      const std::string dq = "dqbfp" + base.substr(std::strlen("gemm_kstore"), base.size() - std::strlen("gemm_kstore") - 4) +
                             (NpuChunked(site) ? "_c" + std::to_string(c) : "") + ".hal";
      const Geom& g = geom_.at(dq);
      const std::size_t panel = NpuK(chunks[c].second).w, calls = rows / NpuCallRows();
      auto b = GemmWeights(*t, f);
      b[0].offset += (static_cast<std::size_t>(t->dims[1]) - rows) * row_bytes;
      b[0].length = std::size_t{rows} * row_bytes;
      b.push_back({npu_->W().handle, slot + w_off, calls * panel});
      PlannedDecode d{dq, "yah_dequant_" + std::string(f.name) + "_bfp16", g.tokens, g.rowgrp, b};
      if (npu_planning_) planned_dq_.resize(npu_job_ + 1), planned_dq_[npu_job_].push_back(d);
      // the previous job's NpuJoin dispatched it already, unless the plan differs here (the last layer's trimmed tail)
      if (!Predecoded(d))
        Dispatch(Exe(dq), d.name.c_str(), d.gx, 1, 1, d.wg, 1, 1, d.b, std::uint64_t{1} << (b.size() - 1));
      views.emplace_back();
      for (std::size_t p = 0; p < calls; ++p) views.back().push_back({slot + w_off + p * panel, panel});
      w_off += calls * panel;
    }
    return views;
  }
  // Decoder-column sets: the NPU's rows of wname (its last rows) as raw GGUF bytes, per chunk and call (offsets from the
  // tensor data base). A call reads, per row and pass, one record of 4 super-blocks from the chunk's first k; the
  // planning pass registers the rows (and the last records' over-read) with the NPU.
  std::vector<std::vector<NpuView>> NpuRawRows(const std::string& site, const std::string& wname, std::uint32_t rows) {
    const auto* t = Find(wname);
    const std::size_t K = t->dims[0], M = t->dims[1], row_bytes = static_cast<std::size_t>(t->bytes) / M;
    const std::size_t blk = row_bytes * 256 / K, first = t->offset + (M - rows) * row_bytes;
    const std::size_t cr = NpuCallRows();
    const std::size_t extent = (cr - 1) * row_bytes + 16 * blk + kDcolRecordRow;   // gen_npu_gemm.fill_inputs
    if (npu_planning_) {
      const std::size_t end = std::min(first + rows * row_bytes + 4096, gguf_.tensor_data_size());
      npu_->RegisterRaw(gguf_.tensor_data_base() + first, end - first);
    }
    std::vector<std::vector<NpuView>> views;
    for (const auto& [k0, Kc] : NpuChunks(site)) {
      views.emplace_back();
      for (std::size_t p = 0; p < rows / cr; ++p)
        views.back().push_back({first + p * cr * row_bytes + k0 / 256 * blk, extent});
    }
    return views;
  }
  // The NPU calls of one matrix: per chunk, per NpuCallRows() rows; C panels from c_off, chunk-major (the unpack sums the chunks).
  // Returns the C view they fill. Decoder-column sets: wname's image, its raw rows w (NpuRawRows). half: the half set's
  // image (token halves).
  NpuView NpuCalls(const std::string& site, const std::vector<NpuView>& a, const std::vector<std::vector<NpuView>>& w,
                   std::size_t& c_off, std::vector<std::uint32_t>& calls, const std::string& wname, bool half = false) {
    const auto chunks = NpuChunks(site);
    const std::size_t start = c_off;
    int dec = -1;
    std::string swap;
    if (npu_dcol_) {
      Fmt f{};
      FmtOf(static_cast<std::uint32_t>(Find(wname)->type), &f);
      swap = (half ? hdir_ : dir_) + "/npu_dcol_" + f.name + ".swap.xdna";
      if (::access(swap.c_str(), R_OK) != 0) swap.clear();   // fused sets (YAH_NPU_FUSE): no decoder tiles to swap
    }
    for (std::size_t c = 0; c < chunks.size(); ++c) {
      const std::string image =
          npu_dcol_ ? DcolImage(wname, &dec, half) : dir_ + "/npu_gemm_" + std::to_string(chunks[c].second) + ".xdna";
      const std::size_t cp = half ? HNpuK(chunks[c].second).c : NpuK(chunks[c].second).c;
      for (const NpuView& wv : w[c]) {
        // fused sets: each image its own family (a family change runs the image's setup; a swap would splice a whole
        // image's first call, which stalls the array)
        calls.push_back(npu_dcol_ ? npu_->BindRaw(image, swap.empty() ? image : swap, dec, a[c],
                                                  gguf_.tensor_data_base() + wv.offset,
                                                  wv.length, {c_off, cp}, swap.empty() ? image : "dcol")
                                  : npu_->Bind(image, a[c], wv, {c_off, cp}));
        c_off += cp;
      }
    }
    return {start, c_off - start};
  }
  // npu_unpack_<site>: C view c, then bindings (writes: the mask over all bindings, C at bit 0). half: the half set's.
  void NpuUnpack(const std::string& hal, NpuView c, std::vector<hrx_buffer_ref_t> rest, std::uint64_t writes,
                 bool half = false) {
    const Geom& g = half ? HGeom(hal) : geom_.at(hal);
    rest.insert(rest.begin(), {npu_->C().handle, c.offset, c.length});
    Dispatch(half ? HExe(hal) : Exe(hal), "yah_npu_unpack", g.tokens, 1, 1, g.rowgrp, 1, 1, rest, writes);
  }
  // The swiglu HAL a GEMM on wname runs (afrag form with af, fragment-major output with tout).
  std::string SwigluHal(const std::string& wname, bool af, bool tout) const {
    Fmt f{};
    const std::string base = GemmHal("gemm_swiglu", *Find(wname), &f);
    return af ? AfHal(base, tout ? ".af.to.hal" : ".af.hal") : base;
  }
  // The kqg HAL a GEMM on wname runs (afrag form with af), "" if the set has none.
  std::string KqgHal(const std::string& wname, bool af) const {
    Fmt f{};
    if (!FmtOf(static_cast<std::uint32_t>(Find(wname)->type), &f)) return "";
    const std::string base = GemmHal("gemm_kqg", *Find(wname), &f);
    return !geom_.count(base) ? "" : af ? AfHal(base) : base;
  }
  // The GPU's rows of the swiglu GEMM on wname (hal: the split HAL), its gate input and output at the full stride.
  void RunSwigluSplit(const std::string& wname, bool af, const std::string& hal, std::uint32_t npu_rows) {
    const auto* t = Find(wname);
    Fmt f{};
    GemmHal("gemm_swiglu", *t, &f);
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*t, f);
    const std::size_t first = b.size();
    for (const LoomBuffer* x :
         std::initializer_list<const LoomBuffer*>{af ? normt_ : scratch_, gateffn_, uwstage_, ostage_, ffnup_})
      b.push_back(Ref(*x));
    const std::size_t M = t->dims[1];
    const std::uint32_t tt = Trim(b, first, {std::size_t(t->dims[0]) * 2, M * 4, 0, 0, M * 2}, g);
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name) + "_swiglu").c_str(), (MTiles(*t) - npu_rows / 16) / g.rowgrp,
             tt, 1, 32, 1, 1, b, GemmWrites(b, f, {ffnup_}));
  }
  // The GPU's heads of the attention q projection (hal: the split kqg HAL) into q_ / gate_ at the full stride.
  void RunKqgSplit(const std::string& wname, bool af, const std::string& hal, std::uint32_t npu_rows) {
    const auto* t = Find(wname);
    Fmt f{};
    GemmHal("gemm_kqg", *t, &f);
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*t, f);
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{af ? normt_ : scratch_, wstage_, ostage_, q_, gate_})
      b.push_back(Ref(*x));
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name) + "_kqg").c_str(), (MTiles(*t) - npu_rows / 16) / g.rowgrp,
             TokenTiles(g), 1, 32, 1, 1, b, GemmWrites(b, f, {q_, gate_}));
  }
  // The GPU's rows [0, rows - npu_rows) of kstore wname (hal: the split HAL); out keeps the full row stride.
  void RunKstoreSplit(const std::string& wname, const LoomBuffer& out, bool af, const std::string& hal,
                      std::uint32_t npu_rows) {
    const auto* t = Find(wname);
    Fmt f{};
    GemmHal("gemm_kstore", *t, &f);
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*t, f);
    const std::size_t first = b.size();
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{af ? normt_ : scratch_, wstage_, ostage_})
      b.push_back(Ref(*x));
    b.push_back(Ref(out));
    const std::uint32_t tt = Trim(b, first, {std::size_t(t->dims[0]) * 2, 0, 0, std::size_t(t->dims[1]) * 4}, g);
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name)).c_str(), (MTiles(*t) - npu_rows / 16) / g.rowgrp, tt, 1,
             32, 1, 1, b, GemmWrites(b, f, {&out}));
  }
  // The chunk's graph: the whole chunk (after layer 0's, SplitGraph) is one graph, the NPU handoffs included
  // (NpuEnqueue / NpuJoin).
  // Replacing a launched graph is safe: the queue retains its command buffers until they complete, and every buffer and
  // executable they use is this object's.
  void NewGraph() {
    chunk_graph_ = std::make_unique<LoomGraph>(gpu_);
    LoomGraph& g = *chunk_graph_;
    weights_.ForEachBuffer([&](hrx_buffer_t h) { g.ReadOnly(h); });
    for (const LoomBuffer* t : {grid_iq3s_, grid_iq3xxs_, grid_iq2xxs_, grid_iq2xs_, ksigns_, eps_, reszero_})
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


  // NPU handoffs are graph nodes on flag words in host memory (NpuSplit::Flags), so the host never splits the chunk.
  // NpuEnqueue: the graph stores the job's gate value to its ready word once its encoders / decoders are done (release,
  // system scope: A and W reach memory first); the NPU waits for it itself. NpuJoin: yah_npu_flag_wait polls the done
  // word; it stands for the NPU's reads of A / W and writes of C, so the unpacks and the next job's encoders and
  // decoders order after it. The jobs are queued right after the chunk graph launches.
  // A weight decode of the planning pass (EnableNpu), dispatched early in one-graph mode.
  struct PlannedDecode {
    std::string hal, name;
    std::uint32_t gx, wg;
    std::vector<hrx_buffer_ref_t> b;
    bool operator==(const PlannedDecode& o) const {
      if (hal != o.hal || gx != o.gx || wg != o.wg || b.size() != o.b.size()) return false;
      for (std::size_t i = 0; i < b.size(); ++i)
        if (b[i].buffer != o.b[i].buffer || b[i].offset != o.b[i].offset || b[i].length != o.b[i].length) return false;
      return true;
    }
  };
  bool Predecoded(const PlannedDecode& d) const {
    return std::find(predecoded_.begin(), predecoded_.end(), d) != predecoded_.end();
  }
  // Job j's half of W (NpuDecode alternates them): the next job's decode must not wait for this job's NPU work.
  hrx_buffer_ref_t WSlot(std::uint32_t job) const {
    const std::size_t half = npu_->W().size / 2;
    return {npu_->W().handle, (job % 2) * half, half};
  }
  hrx_buffer_ref_t FlagWord(std::uint32_t w) const { return {npu_->Flags()->handle, std::size_t{w} * 4, 4}; }
  hrx_buffer_ref_t AView(NpuView v) const { return {npu_->A().handle, v.offset, v.length}; }
  hrx_buffer_ref_t CView(NpuView v) const { return {npu_->C().handle, v.offset, v.length}; }
  // A job's NPU side: the A ranges it reads and the C range it writes (empty a: all of A and C).
  struct NpuSide {
    std::vector<hrx_buffer_ref_t> a;
    hrx_buffer_ref_t c;
  };
  // The job's calls; tag names it in the NPU stats. Returns its index in this chunk (NpuJoin).
  std::uint32_t NpuEnqueue(std::vector<std::uint32_t> calls, std::string tag, NpuSide side = {}) {
    const std::uint32_t job = npu_job_++;
    if (npu_planning_) return job;
    if (side.a.empty()) side = {{Ref(npu_->A())}, Ref(npu_->C())};
    const NpuSplit::Job words = npu_->NewJob(calls);
    std::vector<hrx_buffer_ref_t> in = side.a;
    if (!npu_dcol_) in.push_back(WSlot(job));
    graph_->AtomicStore(FlagWord(words.ready), words.gate, HRX_ATOMIC_FLAG_RELEASE | HRX_ATOMIC_FLAG_SYSTEM_SCOPE, in,
                        {side.c});
    flagged_.push_back({std::move(calls), std::move(tag), words, layer_});
    job_side_.push_back(std::move(side));
    return job;
  }
  // Queue the GPU work that runs beside job (default: the last enqueued one) before this.
  void NpuJoin(std::int64_t job = -1) {
    if (npu_planning_) return;
    const NpuWait w = NpuJoinBegin(job);
    Dispatch(Exe("npu_flag_wait.hal"), "yah_npu_flag_wait", 1, 1, 1, 32, 1, 1, w.words, 4, &w.npu_side);
    NpuJoinEnd(job);
  }
  // NpuJoin's parts around the dispatch that waits for the last enqueued job (yah_npu_flag_wait, or a GEMM whose
  // epilogue waits: gen_gemm_tile.NPUADD): the next job's weight decodes, then what that dispatch binds (the job's flag
  // words done, ready, status; status is written) and orders (the NPU-side buffers, as also-writes).
  struct NpuWait {
    std::vector<hrx_buffer_ref_t> words, npu_side;
  };
  // The wait stands for the NPU work of every job up to job not joined before (the NPU runs them in order).
  NpuWait NpuJoinBegin(std::int64_t job_or_last = -1) {
    const auto job = static_cast<std::uint32_t>(job_or_last < 0 ? flagged_.size() - 1 : job_or_last);
    // the next job's weight decodes run beside this job's GPU work
    predecoded_.clear();
    if (job + 1 < planned_dq_.size())
      for (const PlannedDecode& d : planned_dq_[job + 1]) {
        Dispatch(Exe(d.hal), d.name.c_str(), d.gx, 1, 1, d.wg, 1, 1, d.b, std::uint64_t{1} << (d.b.size() - 1));
        predecoded_.push_back(d);
      }
    const NpuSplit::Queued& j = flagged_.at(job);
    NpuWait w{{FlagWord(j.words.done), FlagWord(j.words.ready), FlagWord(NpuSplit::kFlagStatus)}, {}};
    for (std::size_t i = npu_joined_; i <= job; ++i) {
      w.npu_side.insert(w.npu_side.end(), job_side_[i].a.begin(), job_side_[i].a.end());
      if (!npu_dcol_) w.npu_side.push_back(WSlot(static_cast<std::uint32_t>(i)));
      w.npu_side.push_back(job_side_[i].c);
    }
    return w;
  }
  void NpuJoinEnd(std::int64_t job_or_last = -1) {
    const std::size_t job = job_or_last < 0 ? flagged_.size() - 1 : static_cast<std::size_t>(job_or_last);
    npu_joined_ = std::max(npu_joined_, job + 1);
    // decoder-column sets count jobs per chunk: the joined jobs' words read zero again before the next chunk reuses them
    // (only the joined ones: a later job may still be waiting at its gate)
    if (npu_dcol_)
      for (; npu_cleared_ <= job; ++npu_cleared_)
        for (const std::uint32_t w : {flagged_[npu_cleared_].words.ready, flagged_[npu_cleared_].words.done})
          graph_->AtomicStore(FlagWord(w), 0, HRX_ATOMIC_FLAG_RELEASE | HRX_ATOMIC_FLAG_SYSTEM_SCOPE, {}, {});
  }

  // af: the afrag form, input normt_ (fragment-major)
  // cnt (A4): run the flagged GEMM on these counters if it can (RunIu4); returns whether it did.
  bool RunKstore(const std::string& wname, const LoomBuffer& out, bool af = false, const hrx_buffer_ref_t* cnt = nullptr) {
    const auto* t = Find(wname);
    if (a4_) {
      // dnperm sets: the DeltaNet gate in HF head order (kstoreh), as the pnq kernels read it
      const bool hp = geom_.count("dnperm") && wname.ends_with("attn_gate.weight");
      if (hp && !geom_.count("iu4_kstoreh_" + std::to_string(MTiles(*t)) + "_" + std::to_string(t->dims[0] / 256) + ".hal"))
        throw LoomError("dnperm sets need the kstoreh gate GEMM");
      return RunIu4(hp ? "kstoreh" : "kstore", *t, {&out}, {std::size_t(t->dims[1]) * 4}, 0b1000, nullptr, cnt, &iu4_wait_);
    }
    Fmt f{};
    const std::string base = GemmHal("gemm_kstore", *t, &f);
    const std::string df = af ? "" : DfGemm("kstore", *t, f);
    const std::string hal = af ? AfHal(base) : df.empty() ? PickGemm(base) : df;
    const Geom g = GeomOf(hal);
    auto b = df.empty() ? GemmWeights(*t, f) : DfWeights();
    const std::size_t first = b.size();
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{af ? normt_ : scratch_, wstage_, ostage_})
      b.push_back(Ref(*x));
    b.push_back(Ref(out));
    const std::uint32_t tt = Trim(b, first, {std::size_t(t->dims[0]) * 2, 0, 0, std::size_t(t->dims[1]) * 4}, g);
    if (!df.empty()) DfAhead(b);
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(df.empty() ? f.name : "f16")).c_str(), MTiles(*t) / g.rowgrp,
             tt, 1, 32, 1, 1, b, GemmWrites(b, f, {&out}));
    if (!df.empty()) DfNext();
    return false;
  }
  // The attention q projection with the q / gate unpack fused in (rows = heads x [256 q | 256 gate]).
  // Returns false if the set has no such HAL; the caller then runs kstore + yah_unpack_qg.
  // af: the afrag form, input normt_ (fragment-major)
  bool RunKqg(const std::string& wname, bool af = false) {
    const auto* t = Find(wname);
    if (a4_) {
      RunIu4("kqg", *t, {q_, gate_}, {std::size_t(t->dims[1]) * 2, std::size_t(t->dims[1]) * 2}, 0b11000, nullptr, nullptr,
             &iu4_wait_);
      return true;
    }
    Fmt f{};
    if (!FmtOf(static_cast<std::uint32_t>(t->type), &f)) return false;
    std::string hal = GemmHal("gemm_kqg", *t, &f);
    if (!geom_.count(hal)) return false;
    const std::string df = af ? "" : DfGemm("kqg", *t, f);
    hal = af ? AfHal(hal) : df.empty() ? PickGemm(hal) : df;
    const Geom g = GeomOf(hal);
    auto b = df.empty() ? GemmWeights(*t, f) : DfWeights();
    for (const LoomBuffer* x :
         std::initializer_list<const LoomBuffer*>{af ? normt_ : scratch_, wstage_, ostage_, q_, gate_})
      b.push_back(Ref(*x));
    if (!df.empty()) DfAhead(b);
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(df.empty() ? f.name : "f16") + "_kqg").c_str(),
             MTiles(*t) / g.rowgrp, TokenTiles(g), 1, 32, 1, 1, b, GemmWrites(b, f, {q_, gate_}));
    if (!df.empty()) DfNext();
    return true;
  }
  // af: the afrag form, input normt_; tout: its output fragment-major (for an afrag down projection)
  void RunSwiglu(const std::string& wname, bool af = false, bool tout = false) {
    const auto* t = Find(wname);
    Fmt f{};
    const std::string base = GemmHal("gemm_swiglu", *t, &f);
    const std::string df = af ? "" : DfGemm("swiglu", *t, f);
    const std::string hal = af ? AfHal(base, tout ? ".af.to.hal" : ".af.hal") : df.empty() ? PickGemm(base) : df;
    if (hal.empty()) throw LoomError(base + ": the set lacks its afrag form");
    const Geom g = GeomOf(hal);
    auto b = df.empty() ? GemmWeights(*t, f) : DfWeights();
    const std::size_t first = b.size();
    for (const LoomBuffer* x :
         std::initializer_list<const LoomBuffer*>{af ? normt_ : scratch_, gateffn_, uwstage_, ostage_, ffnup_})
      b.push_back(Ref(*x));
    const std::size_t M = t->dims[1];
    const std::uint32_t tt = Trim(b, first, {std::size_t(t->dims[0]) * 2, M * 4, 0, 0, M * 2}, g);
    if (!df.empty()) DfAhead(b);
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(df.empty() ? f.name : "f16") + "_swiglu").c_str(),
             MTiles(*t) / g.rowgrp, tt, 1, 32, 1, 1, b, GemmWrites(b, f, {ffnup_}));
    if (!df.empty()) DfNext();
  }
  // hidden += W input. The fused kres GEMM writes hidden + W input into hidden2; otherwise kStore writes W input
  // into partial and yah_residual_1d adds it. Either way the two hidden buffers swap.
  // af: the afrag form of the fused kres GEMM (input fragment-major)
  // quantized (A4): aq_ / ad_ already hold the rotated input (gen_pnq), in is not read
  void RunResidual(const std::string& wname, const LoomBuffer& in, bool af = false, bool quantized = false) {
    const auto* t = Find(wname);
    const bool heads = rot_ && wname.find("ssm_out") != std::string::npos;
    const std::string awq = wname.substr(0, wname.size() - 6) + "awq";
    if (a4_) {
      hrx_buffer_ref_t rc{};
      const bool chain = !quantized && RotQ(in, static_cast<std::uint32_t>(t->dims[0]), awq, heads, rqk_ ? &rc : nullptr);
      // nqov: the next A4 normq takes it as the flagged pair (RunNorm); any other dispatch flushes it first
      pending_kres_ = {t, hidden_, hidden2_, trim_row_, rc, chain};
      if (!nqov_) FlushKres();
      std::swap(hidden_, hidden2_);
      return;
    }
    if (rot_) Rotate(in, heads ? rotbuf_ : nullptr, static_cast<std::uint32_t>(t->dims[0]), awq);
    const LoomBuffer& input = heads ? *rotbuf_ : in;
    Fmt f{};
    const std::string fused0 = GemmHal("gemm_kres", *t, &f);
    if (geom_.count(fused0)) {
      const std::string df = af ? "" : DfGemm("kres", *t, f);
      const std::string fused = af ? AfHal(fused0) : df.empty() ? PickGemm(fused0) : df;
      const Geom g = GeomOf(fused);
      auto b = df.empty() ? GemmWeights(*t, f) : DfWeights();
      const std::size_t first = b.size();
      for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{&input, hidden_, wstage_, ostage_, hidden2_}) b.push_back(Ref(*x));
      const std::size_t M = t->dims[1];
      const std::uint32_t tt = Trim(b, first, {std::size_t(t->dims[0]) * 2, M * 4, 0, 0, M * 4}, g);
      if (!df.empty()) DfAhead(b);
      // the persistent kres (".af.p.hal": each workgroup runs every token tile of its row block, grid y = 1) unless trimmed
      const std::string pers = af ? AfHal(fused0, ".af.p.hal") : "";
      const bool use_p = !pers.empty() && tt == g.tt;
      // NPU split (sites "out": ssm_out / attn_output, "down": ffn_down) of untrimmed chunks
      const char* site = wname.find("ffn_down") != std::string::npos ? "down" : "out";
      const std::uint32_t nn = df.empty() && tt == g.tt && NpuOk(wname) ? NpuRows(site) : 0;
      // The persistent kres runs one workgroup per row block (one per CU unsplit), so the split uses the tiled form.
      const std::string sh = nn ? NpuHal(fused) : "";
      if (!sh.empty()) {
        const auto a = NpuEncode(site, input, af, NpuPk(wname));
        std::size_t w_off = 0, c_off = 0;
        const auto w = NpuDecode(site, wname, nn, w_off);
        std::vector<std::uint32_t> calls;
        const NpuView c = NpuCalls(site, a, w, c_off, calls, wname);
        NpuEnqueue(std::move(calls), site);
        const Geom gs = GeomOf(sh);
        const std::string kname = std::string("yah_ffn_gemm_") + f.name + "_kres";
        const std::uint32_t blocks = (MTiles(*t) - nn / 16) / gs.rowgrp;
        // The persistent kres (one workgroup per row block) over all but the last token tile, the tiled kres on that
        // tile beside it (disjoint token ranges: the graph runs them together on the CUs the row blocks leave idle).
        const std::string p3 = fused0.substr(0, fused0.size() - 4) + ".af.p3.npu.hal";   // NPU split only
        if (af && geom_.count(p3) && tt == g.tt) {
          const std::size_t T = tt - 1;
          const std::size_t tile = B_ / tt;
          // per-token bytes of input, residual and output (bindings first, first + 1, first + 4)
          const std::size_t stride[] = {std::size_t(t->dims[0]) * 2, M * 4, 0, 0, M * 4};
          auto part = [&](std::size_t tok0, std::size_t ntok) {
            auto r = b;
            for (std::size_t i = 0; i < 5; ++i)
              if (stride[i]) r[first + i].offset += tok0 * stride[i], r[first + i].length = ntok * stride[i];
            return r;
          };
          const auto b0 = part(0, T * tile), b1 = part(T * tile, tile);
          Dispatch(Exe(p3), kname.c_str(), blocks, 1, 1, 32, 1, 1, b0, GemmWrites(b0, f, {hidden2_}));
          Dispatch(Exe(sh), kname.c_str(), blocks, 1, 1, 32, 1, 1, b1, GemmWrites(b1, f, {hidden2_}));
        } else {
          Dispatch(Exe(sh), kname.c_str(), blocks, tt, 1, 32, 1, 1, b, GemmWrites(b, f, {hidden2_}));
        }
        // the rest of K for the NPU's rows (its last nn weight rows), beside the GPU's rows: f32 [tokens][nn]
        const std::string rh = "npurem_" + std::string(site) + "_" + f.name + ".hal";
        const Geom& gr = geom_.at(rh);
        const std::size_t row_bytes = static_cast<std::size_t>(t->bytes) / M;
        auto rb = GemmWeights(*t, f);
        rb[0].offset += (M - nn) * row_bytes, rb[0].length = std::size_t{nn} * row_bytes;
        for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{&input, wstage_, ostage_}) rb.push_back(Ref(*x));
        rb.push_back({npurem_->handle, 0, std::size_t{B_} * nn * 4});
        Dispatch(Exe(rh), ("yah_ffn_gemm_" + std::string(f.name)).c_str(), nn / 16 / gr.rowgrp, gr.tt, 1, 32, 1, 1, rb,
                 std::uint64_t{1} << (rb.size() - 1));
        NpuJoin();
        NpuUnpack(std::string("npu_unpack_") + site + ".hal", c,
                  {Ref(*hidden_), {npurem_->handle, 0, std::size_t{B_} * nn * 4}, Ref(*hidden2_)}, 8);
        std::swap(hidden_, hidden2_);
        return;
      }
      Dispatch(Exe(use_p ? pers : fused), (std::string("yah_ffn_gemm_") + (df.empty() ? f.name : "f16") + "_kres").c_str(),
               MTiles(*t) / g.rowgrp, use_p ? 1 : tt, 1, 32, 1, 1, b, GemmWrites(b, f, {hidden2_}));
      if (!df.empty()) DfNext();
      std::swap(hidden_, hidden2_);
      return;
    }
    const std::string base = GemmHal("gemm_kstore", *t, &f);
    const std::string df = DfGemm("kstore", *t, f);
    const std::string hal = df.empty() ? PickGemm(base) : df;
    const Geom g = GeomOf(hal);
    auto b = df.empty() ? GemmWeights(*t, f) : DfWeights();
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{&input, wstage_, ostage_, partial_}) b.push_back(Ref(*x));
    if (!df.empty()) DfAhead(b);
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(df.empty() ? f.name : "f16")).c_str(), MTiles(*t) / g.rowgrp,
             TokenTiles(g), 1, 32, 1, 1, b, GemmWrites(b, f, {partial_}));
    if (!df.empty()) DfNext();
    const std::size_t n = std::size_t{B_} * kHidden;
    Dispatch(Exe("accum.hal"), "yah_residual_1d", static_cast<std::uint32_t>(n / 256), 1, 1, 256, 1, 1,
             {Ref(*hidden_), {partial_->handle, 0, n * 4}, Ref(*hidden2_)});
    std::swap(hidden_, hidden2_);
  }

  void RunAttention(std::uint32_t l, std::uint32_t ci, const std::string& pre, const KvHook& hook, bool q_af, bool k_af,
                    bool v_af) {
    const std::uint32_t ai = l / cfg_.full_attention_interval;
    if (ai >= full_) throw LoomError("full-attention layer index past the KV slot count");
    // an afrag o-projection reads the attention output fragment-major (wmma_t)
    const bool o_af = Af("gemm_kres", pre + "attn_output.weight");
    const std::uint32_t nq = NpuOk(pre + "attn_q.weight") ? NpuRows("q") : 0;   // (a format without a decoder: GPU)
    const std::string qs = nq ? NpuHal(KqgHal(pre + "attn_q.weight", q_af)) : "";
    bool qg_fused = !qs.empty();
    if (qg16_ && qg_fused) throw LoomError("f16 q / gate (qg16) needs the GPU kqg GEMM, not the NPU's");
    if (qg_fused) {
      // the NPU's heads of q (whole heads of 512 rows [256 q | 256 gate]) beside the GPU's heads and k / v;
      // the norm wrote the row-major input unless q, k and v are all afrag
      const bool tiled = q_af && k_af && v_af;
      const auto a = NpuEncode("q", tiled ? *normt_ : *scratch_, tiled, NpuPk(pre + "attn_q.weight"));
      std::size_t w_off = 0, c_off = 0;
      const auto w = NpuDecode("q", pre + "attn_q.weight", nq, w_off);
      std::vector<std::uint32_t> calls;
      const NpuView c = NpuCalls("q", a, w, c_off, calls, pre + "attn_q.weight");
      NpuEnqueue(std::move(calls), "q");
      RunKqgSplit(pre + "attn_q.weight", q_af, qs, nq);
      RunKstore(pre + "attn_k.weight", *kbuf_, k_af);
      RunKstore(pre + "attn_v.weight", *vbuf_, v_af);
      NpuJoin();
      NpuUnpack("npu_unpack_q.hal", c, {Ref(*q_), Ref(*gate_)}, 6);
    } else {
      // nqw: all three wait on the norm's counters (iu4_wait_)
      iu4_wait_ = std::exchange(nq_cnt_, {});
      qg_fused = RunKqg(pre + "attn_q.weight", q_af);
      if (qg16_ && !qg_fused) throw LoomError("f16 q / gate (qg16) needs the kqg GEMM");
      if (!qg_fused) RunKstore(pre + "attn_q.weight", *qkv_);
      RunKstore(pre + "attn_k.weight", *kbuf_, k_af);
      RunKstore(pre + "attn_v.weight", *vbuf_, v_af);
      iu4_wait_ = {};
    }
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
                                         Ref(*q16_),
                                         Ref(*kbuf_),
                                         Ref(*kc32_),
                                         Ref(*vc32_),
                                         rope_kpaged_ ? kpool_l : hrx_buffer_ref_t{kv16_->handle, koff, kv16_layer_},
                                         vtrope_ ? vtpool_l : hrx_buffer_ref_t{kv16_->handle, voff, kv16_layer_},
                                         {eps_->handle, 0, 4}};
      if (rope_kpaged_) b.push_back(ptab_ref_);
      Dispatch(ChunkExe("rope", ci), "yah_fused_qk_rope_batched", qnorm_ ? kKvHeads : kHeads + kKvHeads, B_, 1, 256,
               1, 1, b);
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
      if (std::getenv("YAH_FPS_STATS")) FpsStats(ai, ci);
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
      const std::string ah = std::string(o_af ? "wmma_t" : "wmma") + (ci ? "_c" + std::to_string(ci) : std::string()) + ".hal";
      const auto ag = geom_.find(ah);
      const std::uint32_t tpw = ag != geom_.end() ? ag->second.tokens : attn_tpw_,
                          hpw = ag != geom_.end() ? ag->second.rowgrp : attn_hpw_;
      Dispatch(Exe(ah), "yah_attn_wmma", (B_ + tpw - 1) / tpw, kHeads / hpw, 1, 256, 1, 1, b);
      if (fps_on_) FpsDump(ai, ci);
    }
    trim_row_ = tail_;
    RunResidual(pre + "attn_output.weight", *scratch_, o_af);
    trim_row_ = kAllRows;
  }

  // YAH_FPS_STATS: the layer's tile lists (synchronizes): kept prior tiles / the dense prior tiles, correction tiles per list
  void FpsStats(std::uint32_t ai, std::uint32_t ci) {
    gpu_.Synchronize();
    std::vector<std::int32_t> h(std::size_t{B_ / 16} * 4 * fps_ls_);
    gpu_.D2H(*fsel_, h.data(), h.size() * 4, 0);
    const std::uint32_t tpad = pages_ * 256, start = ci * B_;
    double kept = 0, prior = 0, corr = 0;
    for (std::size_t g = 0; g < std::size_t{B_ / 16} * 4; ++g) {
      const std::int32_t* l = h.data() + g * fps_ls_;
      for (std::int32_t t = 0; t < l[0]; ++t) {
        const std::uint32_t ks = static_cast<std::uint32_t>(l[2 + 2 * t]);
        if (ks >= tpad) corr += 1;
        else if (ks < start) kept += 1;
      }
      prior += start / 32;
    }
    std::fprintf(stderr, "fps_stats chunk=%u layer=%u kept=%.4f corr_tiles=%.1f\n", ci, ai, kept / prior,
                 corr / (B_ / 16 * 4));
  }

  // YAH_FPS_DUMP=<prefix> YAH_FPS_DUMP_AT=<chunk>,<attention layer>: that attention call's inputs and output as raw files
  // (<prefix>.q f16 rotated Q, .kfm block means, .kpool / .vpool the f16 pools, .sel lists, .gate f32, .out f16)
  void FpsDump(std::uint32_t ai, std::uint32_t ci) {
    const char* pre = std::getenv("YAH_FPS_DUMP");
    const char* at = std::getenv("YAH_FPS_DUMP_AT");
    if (!pre || !at) return;
    unsigned dc = 0, da = 0;
    // "<chunk>,all": every attention layer of the chunk, the scoring inputs only (.q .kpool .sel) as <prefix>_l<layer>.*
    const bool all = std::sscanf(at, "%u,all", &dc) == 1 && std::strstr(at, "all");
    if (!all && (std::sscanf(at, "%u,%u", &dc, &da) != 2 || da != ai)) return;
    if (dc != ci) return;
    const std::string base = all ? std::string(pre) + "_l" + std::to_string(ai) : std::string(pre);
    gpu_.Synchronize();
    const auto dump = [&](const char* ext, const LoomBuffer& b, std::size_t bytes, std::size_t off) {
      std::vector<std::uint8_t> h(bytes);
      gpu_.D2H(b, h.data(), bytes, off);
      std::FILE* f = std::fopen((base + ext).c_str(), "wb");
      std::fwrite(h.data(), 1, bytes, f);
      std::fclose(f);
    };
    dump(".q", *qr16_, std::size_t{B_} * kAttn * 2, 0);
    dump(".kpool", *kdeq_, deq_bytes_, 0);
    dump(".sel", *fsel_, std::size_t{B_ / 16} * 4 * fps_ls_ * 4, 0);
    if (all && std::getenv("YAH_FPS_DUMP_V")) dump(".vpool", *vdeq_, deq_bytes_, 0);
    if (!all) {
      dump(".kfm", *kfm_, fps_mean_bytes_, std::size_t{ai} * fps_mean_bytes_);
      dump(".vpool", *vdeq_, deq_bytes_, 0);
      dump(".gate", *gate_, std::size_t{B_} * kInner * 4, 0);
      dump(".out", *scratch_, std::size_t{B_} * kAttn * 2, 0);
      // YAH_FPS_DUMP_K4=1: the layer's int4 K / V pools, scales and the page table. The f16 pools above hold only the rows
      // yah_fpmean read this layer (they are shared by the layers) and no pooled rows; the int4 ones are the twin's input
      if (std::getenv("YAH_FPS_DUMP_K4")) {
        dump(".kq4", *kq8buf_, kq_bytes_, std::size_t{ai} * kq_bytes_);
        dump(".ks4", *ksbuf_, ks_bytes_, std::size_t{ai} * ks_bytes_);
        dump(".ptab", *ptab_, ptab_ref_.length, 0);
        dump(".vq4", *vqbuf_, vq_bytes_, std::size_t{ai} * vq_bytes_);
        dump(".vs4", *vqsbuf_, vqs_bytes_, std::size_t{ai} * vqs_bytes_);
      }
    }
    std::fprintf(stderr, "fps_dump chunk=%u layer=%u -> %s.*\n", ci, ai, base.c_str());
  }

  void RunDeltaNet(std::uint32_t l, const std::string& pre, bool qkv_af, bool gate_af) {
    const std::uint32_t si = l - l / cfg_.full_attention_interval;
    // alpha / beta first: they read the norm's row-major copy (21 MB), which the qkv / gate GEMMs would evict from the
    // last-level cache (0.18 vs ~0.39 M cycles each)
    const bool nok = NpuOk(pre + "attn_qkv.weight") && NpuOk(pre + "attn_gate.weight");
    const std::uint32_t nq = nok ? NpuRows("qkv") : 0, ng = nok ? NpuRows("gate") : 0;
    const std::string qkv_split = nq ? NpuHal(KstoreHal(pre + "attn_qkv.weight", qkv_af)) : "";
    const std::string gate_split = ng ? NpuHal(KstoreHal(pre + "attn_gate.weight", gate_af)) : "";
    // conv_c: the conv reads the NPU's qkv columns straight from C ("convkq_c.hal"); the unpack writes only the conv ring
    NpuView conv_c{};
    bool gate_late = false;
    // dnov: the qkv GEMM and the conv, the gate GEMM and pnq part a as flagged pairs (FlagSlot), so the conv runs in the
    // qkv GEMM's tail and pnq part a in the gate GEMM's
    hrx_buffer_ref_t qkv_cnt{}, gate_cnt{}, dn_cnt{};
    bool qkv_flagged = false, gate_flagged = false;
    if (!qkv_split.empty() && !gate_split.empty()) {
      // the NPU's rows of qkv and gate beside alpha / beta and the GPU's rows (one job: the same activations)
      // decoder-column sets: each matrix's activations in its decoder's k-block order
      const bool kq = NpuPk(pre + "attn_qkv.weight"), kg = NpuPk(pre + "attn_gate.weight");
      std::vector<NpuView> a4, ak;
      if (!kq || !kg) a4 = NpuEncode("qkv", *scratch_, false);
      if (kq || kg) ak = NpuEncode("qkv", *scratch_, false, true);
      std::size_t w_off = 0, c_off = 0;
      const auto wq = NpuDecode("qkv", pre + "attn_qkv.weight", nq, w_off);
      const auto wg = NpuDecode("gate", pre + "attn_gate.weight", ng, w_off);
      std::vector<std::uint32_t> calls;
      const NpuView cq = NpuCalls("qkv", kq ? ak : a4, wq, c_off, calls, pre + "attn_qkv.weight");
      // fused sets: a job runs one image (one family: NpuCalls)
      if (npu_dcol_ && DcolImage(pre + "attn_qkv.weight") != DcolImage(pre + "attn_gate.weight") &&
          ::access((dir_ + "/npu_dcol_iq4xs.swap.xdna").c_str(), R_OK) != 0)
        NpuEnqueue(std::exchange(calls, {}), "qkv");
      const NpuView cg = NpuCalls("gate", kg ? ak : a4, wg, c_off, calls, pre + "attn_gate.weight");
      NpuEnqueue(std::move(calls), "qkv");
      RunKstore(pre + "ssm_alpha.weight", *alpha_);
      RunKstore(pre + "ssm_beta.weight", *beta_);
      RunKstoreSplit(pre + "attn_qkv.weight", *qkv_, qkv_af, qkv_split, nq);
      RunKstoreSplit(pre + "attn_gate.weight", *gate_, gate_af, gate_split, ng);
      NpuJoin();
      if (geom_.count("convkq_c.hal")) {
        conv_c = cq;
        NpuUnpack("npu_unpack_qkv_tail.hal", cq, {Ref(*qkv_)}, 2);
      } else {
        NpuUnpack("npu_unpack_qkv.hal", cq, {Ref(*qkv_)}, 2);
      }
      NpuUnpack("npu_unpack_gate.hal", cg, {Ref(*gate_)}, 2);
    } else {
      // nqw: alpha, beta and qkv wait on the norm's counters (iu4_wait_)
      iu4_wait_ = std::exchange(nq_cnt_, {});
      RunKstore(pre + "ssm_alpha.weight", *alpha_);
      RunKstore(pre + "ssm_beta.weight", *beta_);
      // dnvconv: the gate GEMM goes after DeltaNet's second part (below), beside it, and the conv no longer waits for it
      gate_late = geom_.count("dnvconv") != 0;
      if (dnov_ && gate_late) qkv_cnt = FlagSlot();
      qkv_flagged = RunKstore(pre + "attn_qkv.weight", *qkv_, qkv_af, qkv_cnt.buffer ? &qkv_cnt : nullptr);
      if (!gate_late) RunKstore(pre + "attn_gate.weight", *gate_, gate_af);
      iu4_wait_ = {};
    }
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
    if (conv_c.length) {
      const Geom& gg = geom_.at("convkq_g.hal");
      const Geom& gc = geom_.at("convkq_c.hal");
      Dispatch(Exe("convkq_g.hal"), "yah_ssm_conv_kq", gg.tokens, B_ / conv_tb, 1, 256, 1, 1,
               {Ref(*qkv_), TRef(*Find(pre + "ssm_conv1d.weight")), cs, Ref(*conv_out_), Ref(*kqbuf_)});
      Dispatch(Exe("convkq_c.hal"), "yah_ssm_conv_kq_c", gc.tokens, gc.tt, 1, gc.rowgrp, 1, 1,
               {{npu_->C().handle, conv_c.offset, conv_c.length},
                TRef(*Find(pre + "ssm_conv1d.weight")),
                cs,
                Ref(*conv_out_)},
               8);
    } else if (const auto dvc = geom_.find("dnvconv"); dvc != geom_.end()) {
      // dispatch.txt "dnvconv <conv workgroups>": the conv takes only the q / k heads; DeltaNet computes the v channels'
      // conv and alpha / beta itself (gen_gdn_chunk vconv), and prep_ab only advances the conv ring, after both parts
      // (they read the ring before token 0)
      if (!dnsplit_a_ || !dvc->second.tokens) throw LoomError("dispatch.txt dnvconv needs dnsplit and a conv grid");
      if (qkv_flagged)
        Dispatch(Exe("convkqf.hal"), "yah_ssm_conv_kq", dvc->second.tokens, B_ / conv_tb, 1, 256, 1, 1,
                 {Ref(*qkv_), TRef(*Find(pre + "ssm_conv1d.weight")), cs, Ref(*conv_out_), Ref(*kqbuf_), qkv_cnt},
                 0b011000, nullptr, 0b100001);
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
        gate_flagged = RunKstore(pre + "attn_gate.weight", *gate_, gate_af, gate_cnt.buffer ? &gate_cnt : nullptr);
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
    // an afrag ssm_out reads the postnorm output fragment-major (postnorm_t)
    const bool out_af = Af("gemm_kres", pre + "ssm_out.weight");
    const hrx_buffer_ref_t gate{gate_->handle, 0, std::size_t{B_} * kInner * 4};
    // dispatch.txt "dnperm": the DeltaNet parts run head sets (gen_rot.pn_heads) into compact outputs, which only the
    // pnq kernels read
    const bool pnq = a4_ && dnsplit_a_ && !out_af && geom_.count("pnq_a.hal");
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
      if (pnq_ab_) {
        // raw_ untracked: part a's DeltaNet is behind the barrier before part b's, part b's done count is dn_cnt
        if (!dn_done) throw LoomError("pnq_abf.hal needs dnvconv");
        const Geom& gab = geom_.at("pnq_abf.hal");
        if (gate_flagged)
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
      if (dn_done) ring();
      trim_row_ = tail_;
      RunResidual(pre + "ssm_out.weight", *scratch_, false, true);
      trim_row_ = kAllRows;
      return;
    }
    if (dnsplit_a_) {
      const std::string pn = out_af ? "postnorm_t" : "postnorm";
      // heads [0, a) are ssm_out's NPU K chunk: with "npupostnormbfp" this part also writes the NPU's input into A and
      // NpuEncode("out") skips the encoder (RunResidual splits only full chunks; an unused A costs nothing)
      const bool bfp = geom_.count("npupostnormbfp") && NpuRows("out") && trim_row_ == kAllRows && tail_ == kAllRows;
      norm_bfp_ = bfp ? "out" : "";
      std::vector<hrx_buffer_ref_t> b{Ref(*raw_), TRef(*Find(pre + "ssm_norm.weight")), gate, Ref(*scratch_)};
      if (bfp) b.push_back({npu_->A().handle, 0, NpuK(5120).a});
      Dispatch(Exe(pn + (bfp ? "_a_bfp.hal" : "_a.hal")), "yah_ssm_postnorm_fp16", dnsplit_a_ * B_ / 8, 1, 1, 256, 1, 1, b);
      Dispatch(Exe(pn + "_b.hal"), "yah_ssm_postnorm_fp16", dnsplit_b_ * B_ / 8, 1, 1, 256, 1, 1,
               {Ref(*raw2_), TRef(*Find(pre + "ssm_norm.weight")), gate, Ref(*scratch_)});
    } else {
      Dispatch(Exe(out_af ? "postnorm_t.hal" : "postnorm.hal"), "yah_ssm_postnorm_fp16", 6 * B_, 1, 1, 256, 1, 1,
               {Ref(*raw_), TRef(*Find(pre + "ssm_norm.weight")), gate, Ref(*scratch_)});
    }
    trim_row_ = tail_;
    RunResidual(pre + "ssm_out.weight", *scratch_, out_af);
    trim_row_ = kAllRows;
  }

  // ---- Token halves ----
  // A layer's qkv / q and out sites run per half of the chunk, so the NPU computes half 1 while the GPU runs half 0's DeltaNet / attention chain.
  // GPU order: C0 C1 (FFN unpack, attn norm, job) | GPU rows of qkv / q (whole chunk) | A0 o0 A1 o1 (chain, out job, GPU out rows) | B0 B1 (out unpack, FFN norm).
  // Half h runs the kernels of "<dir>/h" (the set at half the chunk) on token views from h * B / 2; half 1 continues the conv ring, DeltaNet state and KV pools like a next chunk.
  // HalfA / HalfC: no GPU kernel reads a job's C range between its ready store and its join (the graph orders it only after the store), nor writes an A range the NPU still reads.
  std::uint32_t HB() const { return B_ / 2; }
  // Half h's rows of a token-major buffer (row_bytes per token; fragment-major layouts group 16 tokens, so the same).
  hrx_buffer_ref_t Tok(const LoomBuffer& b, int h, std::size_t row_bytes) const {
    return {b.handle, std::size_t(h) * HB() * row_bytes, std::size_t{HB()} * row_bytes};
  }
  void LoadHalves() {
    std::ifstream f(dir_ + "/h/dispatch.txt");
    if (!f) return;
    std::string name;
    Geom g{};
    while (f >> name >> g.tokens >> g.rowgrp >> g.tt) hgeom_[name] = g;
    const auto c = hgeom_.find("ctx");
    if (c == hgeom_.end() || 2 * c->second.tokens != B_ || c->second.tt != T_)
      throw LoomError(dir_ + "/h is not this set at half the chunk (re-emit the set)");
    hdir_ = dir_ + "/h";
  }
  const Geom& HGeom(const std::string& hal) const {
    const auto it = hgeom_.find(hal);
    if (it == hgeom_.end()) throw LoomError("h/dispatch.txt has no row for " + hal);
    return it->second;
  }
  // A GEMM HAL of the half set; refuses one emitted for another token count.
  Geom HGeomOf(const std::string& hal) const {
    const Geom g = HGeom(hal);
    if (g.tokens == 0 || g.rowgrp == 0 || (HB() + g.tokens - 1) / g.tokens != g.tt)
      throw LoomError("h/" + hal + ": emitted for another token count");
    return g;
  }
  LoomExecutable& HExe(const std::string& hal) { return Exe("h/" + hal); }
  LoomExecutable& HChunkExe(const std::string& stem, std::uint32_t c) {
    return HExe(stem + (c ? "_c" + std::to_string(c) : std::string()) + ".hal");
  }
  NpuBytes HNpuK(std::uint32_t K) const {
    const Geom& g = HGeom("npubytes_" + std::to_string(K));
    return {g.tokens, g.rowgrp, g.tt};
  }
  bool HalvesOn() const { return halves_ && npu_on_ && n_ == B_ && tail_ == kAllRows && trim_row_ == kAllRows; }
  // Half h's A slot of a K = 5120 stream (S bytes): the attn norm's (in: qkv, gate, q) or the out site's, pk: the Q4_K
  // order. out's half-0 slots reuse the in ones (half 0's in jobs are joined first); its half-1 slots lie past the FFN
  // block's whole-chunk stream [0, 2 S), which the FFN norm halves write while the NPU still reads out's half 1.
  NpuView HalfA(bool out, int h, bool pk) const {
    const std::size_t s = HNpuK(5120).a;
    return {(out && h ? 4 + pk : 2 * pk + (out ? 0 : h)) * s, s};
  }
  // A half job's calls of a site.
  std::size_t HalfCalls(const char* site) const {
    const auto it = geom_.find(std::string("npusplit_") + site);
    return it == geom_.end() ? 0 : it->second.tokens / NpuCallRows() * NpuChunks(site).size();
  }
  std::size_t HalfCBytes() const {
    return std::max({HalfCalls("qkv") + HalfCalls("gate"), HalfCalls("q"), HalfCalls("out")}) * HNpuK(5120).c;
  }
  // Half h's C region (its in and out jobs: out's half-h job runs once that half's in job is unpacked). Half 0's from 0:
  // the FFN block's H, which no GPU kernel reads (its down partials after it are still read by the next layer's half-1
  // unpack while half 0's jobs run); half 1's after every whole-chunk region.
  std::size_t HalfC(int h) const {
    const std::size_t whole = WholePlan().c_bytes, need = HalfCBytes();
    const std::size_t c0 = !FfnBlkF() || need <= FfnBlkDown() ? 0 : (whole + 4095) / 4096 * 4096;
    return h == 0 ? c0 : (std::max(whole, c0 + need) + 4095) / 4096 * 4096;
  }
  // Rows per workgroup of a norm HAL (its workgroup size over dispatch.txt norm_split waves a row).
  std::uint32_t NormRowsPerWg(const LoomExecutable& exe) const {
    const std::uint32_t ws = exe.WorkgroupSize(exe.OrdinalOrZero("yah_half_norm"));
    const auto ns = geom_.find("norm_split");
    const std::uint32_t split = ns != geom_.end() && ns->second.rowgrp ? ns->second.rowgrp : 1;
    const std::uint32_t rows = ws ? ws / 32 / split : 1;
    if (!rows || (ws && ws % (32 * split)) || HB() % rows) throw LoomError("norm: workgroup size does not fit a half");
    return rows;
  }
  // The FFN block's unpack the last layer deferred (RunFfnBlock): half h's tokens, or (h < 0) all of them.
  void FfnUnpackPending(int h) {
    if (!ffn_pending_.in) return;
    const std::size_t r = std::size_t{kHidden} * 4;
    if (h < 0)
      NpuUnpack("npu_unpack_ffnblk.hal", ffn_pending_.c,
                {Ref(*ffn_pending_.in), {ffnrem_->handle, 0, B_ * r}, Ref(*ffn_pending_.out)}, 8);
    else
      NpuUnpack("npu_unpack_ffnblk_h" + std::to_string(h) + ".hal", ffn_pending_.c,
                {Tok(*ffn_pending_.in, h, r), Tok(*ffnrem_, h, r), Tok(*ffn_pending_.out, h, r)}, 8);
    if (h != 0) ffn_pending_ = {};
  }
  // C_h's attn norm over half h, with its NPU stream into half h's in slot (the half set's "<norm>_bfp.hal").
  void HalfNorm(const std::string& wname, NormOut mode, int h) {
    LoomExecutable& exe = HExe(mode == NormOut::kTiled  ? "norm_t_bfp.hal"
                               : mode == NormOut::kBoth ? "norm_rt_bfp.hal"
                                                        : "norm_bfp.hal");
    const std::size_t r = std::size_t{kHidden} * 4, r16 = std::size_t{kHidden} * 2;
    std::vector<hrx_buffer_ref_t> b{Tok(*hidden_, h, r), Tok(*reszero_, h, r), TRef(*Find(wname)), Tok(*sumout_, h, r),
                                    Tok(mode == NormOut::kTiled ? *normt_ : *scratch_, h, r16)};
    if (mode == NormOut::kBoth) b.push_back(Tok(*normt_, h, r16));
    b.push_back(AView(HalfA(false, h, false)));
    Dispatch(exe, "yah_half_norm", HB() / NormRowsPerWg(exe), 1, 1, 32, 1, 1, b);
  }
  // B_h's FFN block norm over half h into the whole chunk's stream ("<stem>_h<h>.hal"); x: the layer's new hidden.
  void FfnHalfNorm(const std::string& pre, const char* stem, const LoomBuffer& x, int h) {
    LoomExecutable& exe = Exe(std::string(stem) + "_h" + std::to_string(h) + ".hal");
    const std::size_t r = std::size_t{kHidden} * 4, r16 = std::size_t{kHidden} * 2;
    Dispatch(exe, "yah_half_norm", HB() / NormRowsPerWg(exe), 1, 1, 32, 1, 1,
             {Tok(x, h, r),
              Tok(*reszero_, h, r),
              TRef(*Find(pre + "post_attention_norm.weight")),
              Tok(*sumout_, h, r),
              Tok(*normt_, h, r16),
              {npu_->A().handle, 0, NpuK(5120).a}});
  }
  // Half h's NPU stream ("npu_enc_*" of the half set) from input's half h rows into slot a.
  void HalfEncode(const std::string& hal, const LoomBuffer& input, std::size_t row_bytes, int h, NpuView a) {
    const Geom& g = HGeom(hal);
    Dispatch(HExe(hal), "yah_bfp16_encode_act", g.tokens, 1, 1, g.rowgrp, 1, 1, {Tok(input, h, row_bytes), AView(a)},
             2);
  }
  // The out site (ssm_out / attn_output wname; af: its afrag form) splits by halves: an NPU decoder for its format and
  // the half set's split HALs.
  bool HalfOutOk(const std::string& wname, bool af) const {
    if (!NpuOk(wname) || !NpuRows("out") || DcolImage(wname, nullptr, true).empty()) return false;
    Fmt f{};
    const std::string fused0 = GemmHal("gemm_kres", *Find(wname), &f);
    const std::string fused = af ? AfHal(fused0) : fused0;
    return geom_.count(fused0) && !fused.empty() && hgeom_.count(fused.substr(0, fused.size() - 4) + ".npu.hal") &&
           hgeom_.count("npurem_out_" + std::string(f.name) + ".hal") && hgeom_.count("npu_unpack_out.hal");
  }
  // o_h: half h of the out site on input (row-major, or fragment-major with af): its NPU job on half h's out slot
  // (from_norm: the postnorm wrote the p4 stream; else the half encoder), the GPU's rows (the kres split) and the rest of
  // K of the NPU's rows. Returns the job; c: its C.
  std::uint32_t OutHalf(const std::string& wname, const LoomBuffer& input, bool af, int h, bool from_norm, NpuView& c) {
    const auto* t = Find(wname);
    Fmt f{};
    const std::string fused0 = GemmHal("gemm_kres", *t, &f);
    const std::string fused = af ? AfHal(fused0) : fused0;
    const std::string sh = fused.substr(0, fused.size() - 4) + ".npu.hal";
    const std::uint32_t nn = NpuRows("out");
    const std::size_t M = t->dims[1], K = t->dims[0], row_bytes = static_cast<std::size_t>(t->bytes) / M;
    const bool pk = NpuPk(wname);
    const NpuView a = HalfA(true, h, pk);
    if (pk || !from_norm)
      HalfEncode(std::string("npu_enc_6144_c0") + (af ? "_t" : "") + (pk ? "_pk" : "") + ".hal", input, K * 2, h, a);
    std::size_t c_off = HalfC(h);
    std::vector<std::uint32_t> calls;
    c = NpuCalls("out", {a}, NpuRawRows("out", wname, nn), c_off, calls, wname, true);
    const std::uint32_t job = NpuEnqueue(std::move(calls), "out", {{AView(a)}, CView(c)});
    const Geom gs = HGeomOf(sh);
    auto b = GemmWeights(*t, f);
    for (const hrx_buffer_ref_t& x :
         {Tok(input, h, K * 2), Tok(*hidden_, h, M * 4), Ref(*wstage_), Ref(*ostage_), Tok(*hidden2_, h, M * 4)})
      b.push_back(x);
    Dispatch(HExe(sh), ("yah_ffn_gemm_" + std::string(f.name) + "_kres").c_str(), (MTiles(*t) - nn / 16) / gs.rowgrp,
             gs.tt, 1, 32, 1, 1, b, GemmWrites(b, f, {hidden2_}));
    const std::string rh = "npurem_out_" + std::string(f.name) + ".hal";
    const Geom gr = HGeomOf(rh);
    auto rb = GemmWeights(*t, f);
    rb[0].offset += (M - nn) * row_bytes, rb[0].length = std::size_t{nn} * row_bytes;
    for (const hrx_buffer_ref_t& x :
         {Tok(input, h, K * 2), Ref(*wstage_), Ref(*ostage_), Tok(*npurem_, h, std::size_t{nn} * 4)})
      rb.push_back(x);
    Dispatch(HExe(rh), ("yah_ffn_gemm_" + std::string(f.name)).c_str(), nn / 16 / gr.rowgrp, gr.tt, 1, 32, 1, 1, rb,
             std::uint64_t{1} << (rb.size() - 1));
    return job;
  }
  // B0 B1: out's unpack per half (hidden2 = hidden + the NPU's columns + their K remainder) and, when the layer's FFN is
  // the NPU's FFN block, its norm over that half (RunFfnBlock then skips it); then the hidden buffers swap.
  void OutHalvesEnd(const std::string& pre, const std::string& wname, const std::uint32_t (&job)[2],
                    const NpuView (&c)[2]) {
    const char* norm = nullptr;
    const bool ffn = RunFfnBlock(pre, &norm) && ::access((dir_ + "/" + norm + "_h1.hal").c_str(), R_OK) == 0;
    const std::size_t M = Find(wname)->dims[1], nn = NpuRows("out");
    for (int h = 0; h < 2; ++h) {
      NpuJoin(job[h]);
      NpuUnpack("npu_unpack_out.hal", c[h],
                {Tok(*hidden_, h, M * 4), Tok(*npurem_, h, nn * 4), Tok(*hidden2_, h, M * 4)}, 8, true);
      if (ffn) FfnHalfNorm(pre, norm, *hidden2_, h);
    }
    ffn_normed_ = ffn;
    std::swap(hidden_, hidden2_);
  }
  // A DeltaNet layer by halves (the NPU's qkv / gate rows, and out's when HalfOutOk); false: run it whole.
  bool RunDeltaNetHalves(std::uint32_t l, const std::string& pre, bool qkv_af, bool gate_af) {
    const std::string wq = pre + "attn_qkv.weight", wg = pre + "attn_gate.weight", wo = pre + "ssm_out.weight";
    if (!HalvesOn() || !dnsplit_a_ || !geom_.count("convkq_c.hal") || DcolImage(wq, nullptr, true).empty() ||
        DcolImage(wg, nullptr, true).empty())
      return false;
    norm_bfp_.clear();
    const std::uint32_t si = l - l / cfg_.full_attention_interval;
    const std::uint32_t nq = NpuRows("qkv"), ng = NpuRows("gate");
    const bool out_af = Af("gemm_kres", wo), out_half = HalfOutOk(wo, out_af);
    const bool bfp = out_half && hgeom_.count("npupostnormbfp");  // the postnorm writes out's p4 stream
    const bool kq = NpuPk(wq), kg = NpuPk(wg);
    // fused sets: a job runs one image; with two, half 1 runs gate first (the array keeps half 0's last image)
    const bool two = DcolImage(wq, nullptr, true) != DcolImage(wg, nullptr, true) &&
                     ::access((hdir_ + "/npu_dcol_iq4xs.swap.xdna").c_str(), R_OK) != 0;
    const auto rq = NpuRawRows("qkv", wq, nq), rg = NpuRawRows("gate", wg, ng);
    // with two images gate runs whole between qkv's halves (one switch and one fill less than its halves): its own A
    // stream in out's half-1 slots (free until A_1) and C past half 0's region (dead past C_1's FFN unpack)
    const std::string genc = std::string("npu_enc_5120") + (kg ? "_pk" : "") + ".hal";
    const NpuView ga{4 * HNpuK(5120).a, NpuK(5120).a};
    const std::size_t gc0 = (HalfC(0) + HalfCBytes() + 4095) / 4096 * 4096;
    const bool gwhole = two && ga.length == 2 * HNpuK(5120).a && geom_.count(genc) && !DcolImage(wg).empty() &&
                        gc0 + HalfCalls("gate") * NpuK(5120).c <= HalfC(1);
    std::uint32_t job[2] = {}, gjob = 0;
    NpuView cq[2], cg[2];
    for (int h = 0; h < 2; ++h) {
      FfnUnpackPending(h);
      HalfNorm(pre + "attn_norm.weight", qkv_af || gate_af ? NormOut::kBoth : NormOut::kRow, h);
      const NpuView a4 = HalfA(false, h, false), ak = HalfA(false, h, true);
      if (kq || (kg && !gwhole)) HalfEncode("npu_enc_5120_pk.hal", *scratch_, std::size_t{kHidden} * 2, h, ak);
      if (gwhole && h == 1) {
        const Geom& g = geom_.at(genc);
        Dispatch(Exe(genc), "yah_bfp16_encode_act", g.tokens, 1, 1, g.rowgrp, 1, 1, {Ref(*scratch_), AView(ga)}, 2);
        std::size_t c_off = gc0;
        std::vector<std::uint32_t> calls;
        cg[0] = NpuCalls("gate", {ga}, rg, c_off, calls, wg);
        gjob = NpuEnqueue(std::move(calls), "qkv", {{AView(ga)}, CView(cg[0])});
      }
      std::size_t c_off = HalfC(h);
      std::vector<std::uint32_t> calls;
      const auto add = [&](bool qkv) {  // one matrix's calls (and job, with two images)
        const std::size_t c0 = c_off;
        const NpuView a = (qkv ? kq : kg) ? ak : a4;
        (qkv ? cq : cg)[h] = NpuCalls(qkv ? "qkv" : "gate", {a}, qkv ? rq : rg, c_off, calls, qkv ? wq : wg, true);
        if (two) job[h] = NpuEnqueue(std::exchange(calls, {}), "qkv", {{AView(a)}, CView({c0, c_off - c0})});
      };
      add(h == 0 || gwhole);
      if (!gwhole) add(h != 0);
      if (!two) {
        std::vector<hrx_buffer_ref_t> as{AView(kq ? ak : a4)};
        if (kq != kg) as.push_back(AView(kg ? ak : a4));
        job[h] = NpuEnqueue(std::move(calls), "qkv", {as, CView({HalfC(h), c_off - HalfC(h)})});
      }
    }
    // the GPU's rows of the whole chunk (as RunDeltaNet)
    RunKstore(pre + "ssm_alpha.weight", *alpha_);
    RunKstore(pre + "ssm_beta.weight", *beta_);
    RunKstoreSplit(wq, *qkv_, qkv_af, NpuHal(KstoreHal(wq, qkv_af)), nq);
    RunKstoreSplit(wg, *gate_, gate_af, NpuHal(KstoreHal(wg, gate_af)), ng);
    const hrx_buffer_ref_t cs{conv_state_->handle, std::size_t{si} * kQkv * 4 * 4, std::size_t{kQkv} * 4 * 4};
    const hrx_buffer_ref_t st{state_->handle, std::size_t{si} * kTs * kState * kState * 4,
                              std::size_t{kTs} * kState * kState * 4};
    const auto* w1d = Find(pre + "ssm_conv1d.weight");
    const Geom& gg = HGeom("convkq_g.hal");
    const Geom& gc = HGeom("convkq_c.hal");
    const auto ctb = hgeom_.find("convtb");
    const std::uint32_t conv_tb = ctb != hgeom_.end() ? ctb->second.tokens : 1;
    const std::uint32_t dn_grp = HGeom("rowsplit.hal").rowgrp;
    const std::uint32_t dn_thr = hgeom_.count("dnthreads") ? HGeom("dnthreads").tokens : 256;
    // per token: qkv / conv f16 [10240], gate / postnorm f16 [6144], kq f32 [16][3], ab f32 [48][2], raw f32 [6144]
    const std::size_t rqkv = std::size_t{kQkv} * 2, rin = std::size_t{kInner} * 2, rkq = std::size_t{kKh} * 3 * 4,
                      rab = std::size_t{kTs} * 2 * 4, rraw = std::size_t{kInner} * 4, rts = std::size_t{kTs} * 4;
    const std::string pn = out_af ? "postnorm_t" : "postnorm";
    std::uint32_t ojob[2] = {};
    NpuView oc[2];
    for (int h = 0; h < 2; ++h) {
      NpuJoin(job[h]);
      // the conv reads the NPU's v channels from C; the unpack writes the conv ring's tokens (prep_ab reads them)
      NpuUnpack("npu_unpack_qkv_tail.hal", cq[h], {Tok(*qkv_, h, rqkv)}, 2, true);
      if (!gwhole) {
        NpuUnpack("npu_unpack_gate.hal", cg[h], {Tok(*gate_, h, rin)}, 2, true);
      } else if (h == 0) {
        NpuJoin(gjob);
        NpuUnpack("npu_unpack_gate.hal", cg[0], {Ref(*gate_)}, 2);
      }
      Dispatch(HExe("convkq_g.hal"), "yah_ssm_conv_kq", gg.tokens, HB() / conv_tb, 1, 256, 1, 1,
               {Tok(*qkv_, h, rqkv), TRef(*w1d), cs, Tok(*conv_out_, h, rqkv), Tok(*kqbuf_, h, rkq)});
      Dispatch(HExe("convkq_c.hal"), "yah_ssm_conv_kq_c", gc.tokens, gc.tt, 1, gc.rowgrp, 1, 1,
               {CView(cq[h]), TRef(*w1d), cs, Tok(*conv_out_, h, rqkv)}, 8);
      // (prep_ab clamps valid to its half)
      Dispatch(HExe("prepab.hal"), "yah_deltanet_prep_ab", (std::max<std::uint32_t>(HB() * kTs, kQkv) + 255u) / 256u, 1,
               1, 256, 1, 1,
               {Tok(*alpha_, h, rts), Tok(*beta_, h, rts), TRef(*Find(pre + "ssm_a")), TRef(*Find(pre + "ssm_dt.bias")),
                Tok(*qkv_, h, rqkv), cs, Tok(*ab_, h, rab), Ref(*valid_)});
      Dispatch(HExe("rowsplit_a.hal"), "yah_deltanet", dn_grp, dnsplit_a_, 1, dn_thr, 1, 1,
               {Tok(*conv_out_, h, rqkv), Tok(*kqbuf_, h, rkq), Tok(*ab_, h, rab), st, Tok(*raw_, h, rraw)});
      Dispatch(HExe("rowsplit_b.hal"), "yah_deltanet", dn_grp, dnsplit_b_, 1, dn_thr, 1, 1,
               {Tok(*conv_out_, h, rqkv), Tok(*kqbuf_, h, rkq), Tok(*ab_, h, rab), st, Tok(*raw2_, h, rraw)});
      const hrx_buffer_ref_t gate = Tok(*gate_, h, rin);
      std::vector<hrx_buffer_ref_t> b{Tok(*raw_, h, rraw), TRef(*Find(pre + "ssm_norm.weight")), gate,
                                      Tok(*scratch_, h, rin)};
      if (bfp) b.push_back(AView(HalfA(true, h, false)));
      Dispatch(HExe(pn + (bfp ? "_a_bfp.hal" : "_a.hal")), "yah_ssm_postnorm_fp16", dnsplit_a_ * HB() / 8, 1, 1, 256, 1,
               1, b);
      Dispatch(HExe(pn + "_b.hal"), "yah_ssm_postnorm_fp16", dnsplit_b_ * HB() / 8, 1, 1, 256, 1, 1,
               {Tok(*raw2_, h, rraw), TRef(*Find(pre + "ssm_norm.weight")), gate, Tok(*scratch_, h, rin)});
      if (out_half) ojob[h] = OutHalf(wo, *scratch_, out_af, h, bfp, oc[h]);
    }
    if (out_half)
      OutHalvesEnd(pre, wo, ojob, oc);
    else
      RunResidual(wo, *scratch_, out_af);
    return true;
  }
  // An attention layer by halves (the NPU's q heads, and out's rows when HalfOutOk; the paged fp16 KV only); false: run
  // it whole.
  bool RunAttentionHalves(std::uint32_t l, std::uint32_t ci, const std::string& pre, const KvHook& hook, bool q_af,
                          bool k_af, bool v_af) {
    const std::string wq = pre + "attn_q.weight", wo = pre + "attn_output.weight";
    if (!HalvesOn() || hook || !kv_paged_ || !paged_f16k_ || !paged_f16v_ || !rope_kpaged_ || !kv16_scratch_ || kdeq_ ||
        qr16_ || attn_kq8_ || attn_vqt_ || (q16_ == q_ && !qnorm_) || qg16_ || DcolImage(wq, nullptr, true).empty())
      return false;
    norm_bfp_.clear();
    const std::uint32_t ai = l / cfg_.full_attention_interval;
    if (ai >= full_) throw LoomError("full-attention layer index past the KV slot count");
    const std::uint32_t nq = NpuRows("q");
    const bool o_af = Af("gemm_kres", wo), out_half = HalfOutOk(wo, o_af);
    const bool tiled = q_af && k_af && v_af, pk = NpuPk(wq);
    const auto rq = NpuRawRows("q", wq, nq);
    std::uint32_t job[2] = {};
    NpuView cq[2];
    for (int h = 0; h < 2; ++h) {
      FfnUnpackPending(h);
      HalfNorm(pre + "attn_norm.weight",
               tiled                  ? NormOut::kTiled
               : q_af || k_af || v_af ? NormOut::kBoth
                                      : NormOut::kRow,
               h);
      const NpuView a = HalfA(false, h, pk);
      if (pk)
        HalfEncode(std::string("npu_enc_5120") + (tiled ? "_t" : "") + "_pk.hal", tiled ? *normt_ : *scratch_,
                   std::size_t{kHidden} * 2, h, a);
      std::size_t c_off = HalfC(h);
      std::vector<std::uint32_t> calls;
      cq[h] = NpuCalls("q", {a}, rq, c_off, calls, wq, true);
      job[h] = NpuEnqueue(std::move(calls), "q", {{AView(a)}, CView(cq[h])});
    }
    // the GPU's q heads, k and v of the whole chunk (as RunAttention)
    RunKqgSplit(wq, q_af, NpuHal(KqgHal(wq, q_af)), nq);
    RunKstore(pre + "attn_k.weight", *kbuf_, k_af);
    RunKstore(pre + "attn_v.weight", *vbuf_, v_af);
    const hrx_buffer_ref_t kpool_l{kpool_->handle, std::size_t{ai} * pool_bytes_, pool_bytes_};
    const hrx_buffer_ref_t vtpool_l{vtpool_->handle, std::size_t{ai} * pool_bytes_, pool_bytes_};
    // per token: q / gate f32 [6144], q16 f16 [6144] (q16 sets), k / v f32 [1024], output f16 [6144], lse f32 [24]
    const std::size_t rq32 = std::size_t{kAttn} * 4, rq16 = std::size_t{kAttn} * 2, rkv = std::size_t{kKv} * 4,
                      vrows = std::size_t{HB()} * kKvRow * 2;
    const Geom& at = HGeom("wmma.hal");
    std::uint32_t ojob[2] = {};
    NpuView oc[2];
    for (int h = 0; h < 2; ++h) {
      const std::uint32_t c = 2 * ci + h;  // the half set's chunk
      NpuJoin(job[h]);
      NpuUnpack("npu_unpack_q.hal", cq[h], {Tok(*q_, h, rq32), Tok(*gate_, h, rq32)}, 6, true);
      // V rows into the half's part of the layer's f16 V scratch, K rows straight into the paged pool
      const hrx_buffer_ref_t v16{kv16_->handle, kv16_layer_ + h * vrows, vrows};
      Dispatch(HChunkExe("rope", c), "yah_fused_qk_rope_batched", qnorm_ ? kKvHeads : kHeads + kKvHeads, HB(), 1, 256, 1, 1,
               {Tok(*q_, h, rq32),
                Tok(*kbuf_, h, rkv),
                Tok(*vbuf_, h, rkv),
                TRef(*Find(pre + "attn_q_norm.weight")),
                TRef(*Find(pre + "attn_k_norm.weight")),
                Tok(*q16_, h, rq16),
                Tok(*kbuf_, h, rkv),
                Ref(*kc32_),
                Ref(*vc32_),
                kpool_l,
                vtrope_ ? vtpool_l : v16,
                {eps_->handle, 0, 4},
                ptab_ref_});
      if (!vtrope_)
        Dispatch(HChunkExe("vtpage", c), "yah_vtpage", 32, (HB() + 31) / 32, 1, 256, 1, 1, {v16, vtpool_l, ptab_ref_});
      std::vector<hrx_buffer_ref_t> ab = {
          qnorm_ ? Tok(*q_, h, rq32) : Tok(*q16_, h, rq16), Tok(*gate_, h, rq32), kpool_l, vtpool_l,
          Tok(*scratch_, h, std::size_t{kAttn} * 2), Tok(*lse_, h, std::size_t{kHeads} * 4), ptab_ref_};
      if (qnorm_) {
        ab.push_back(TRef(*Find(pre + "attn_q_norm.weight")));
        ab.push_back({eps_->handle, 0, 4});
      }
      Dispatch(HChunkExe(o_af ? "wmma_t" : "wmma", c), "yah_attn_wmma", (HB() + at.tokens - 1) / at.tokens,
               kHeads / at.rowgrp, 1, 256, 1, 1, ab);
      if (out_half) ojob[h] = OutHalf(wo, *scratch_, o_af, h, false, oc[h]);
    }
    if (out_half)
      OutHalvesEnd(pre, wo, ojob, oc);
    else
      RunResidual(wo, *scratch_, o_af);
    return true;
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
  NpuSplit* npu_ = nullptr;        // EnableNpu
  bool npu_on_ = false;            // this chunk splits with the NPU
  std::uint32_t layer_ = 0;        // RecordLayers' current layer (NpuSplit::Queued::layer)
  bool npu_planning_ = false;      // EnableNpu's binding pass: Dispatch records nothing
  LoomBuffer* npurem_ = nullptr;   // the GPU's K remainder of the NPU's out / down rows (f32 [tokens][rows])
  std::uint32_t ffnblk_ = 0;       // the NPU's FFN block features (dispatch.txt "npuffnblk"; RunFfnBlock), 0: none
  LoomBuffer* ffnrem_ = nullptr;   // its GPU down over the other features (f32 [tokens][5120])
  bool ffn_bfp_ = false;           // the last ffn unpack wrote down's BFP16 input for its columns (FfnUnpack)
  std::string norm_bfp_;           // the site whose NPU input the last norm wrote (RunNorm npu_site)
  bool npu_dcol_ = false;          // the set's NPU decodes raw weight rows itself (dispatch.txt "npudcol")
  static constexpr std::size_t kDcolRecordRow = 576;   // a decoder record's row bytes (gen_npu_dec.RECROW)
  std::uint32_t npu_job_ = 0;      // NPU jobs of this chunk so far (W slot = job % 2)
  std::size_t npu_cleared_ = 0;    // flagged_ jobs whose flag words NpuJoin cleared
  std::size_t npu_joined_ = 0;     // flagged_ jobs an NpuJoin waited for (in order)
  std::vector<NpuSide> job_side_;  // per flagged_ job: its A / C ranges (NpuEnqueue)
  std::vector<NpuSplit::Queued> flagged_;  // this chunk's NPU jobs (index = job), queued after the launch
  std::vector<std::vector<PlannedDecode>> planned_dq_;  // per job: its weight decodes (planning pass)
  std::vector<PlannedDecode> predecoded_;               // the next job's, dispatched at the last Join
  std::map<std::string, std::uint32_t> npu_rows_;  // NPU rows per site (dispatch.txt "npusplit_<site>")
  // Token halves (LoadHalves): the half set's directory and dispatch.txt, whether this chunk's NPU layers use them
  std::string hdir_;
  std::map<std::string, Geom> hgeom_;
  bool halves_ = false;
  // the FFN block's unpack the next layer runs (RunFfnBlock: per half, or whole)
  struct FfnPending {
    NpuView c;
    LoomBuffer *in = nullptr, *out = nullptr;
  } ffn_pending_;
  bool ffn_normed_ = false;  // the halves ran the next FFN block's norm (OutHalvesEnd)
  // The last RunLayers graph's dispatches in node order.
  std::vector<Node> nodes_;
  std::unique_ptr<PrefillCalib> calib_;
  int pending_tag_ = -1;
  std::vector<hrx_profile_dispatch_t> session_;   // dispatch timestamps of the open profile session
  std::vector<std::vector<Node>> session_chunks_;  // the graphs launched in it
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
  bool rot_ = false;
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
  // dispatch.txt rqk: the deferred kres waits on its rotq's counters (rotqfo_17408 / rotqo_<k>, iu4_kresfw_*)
  bool rqk_ = false;
  // dispatch.txt nqw: the attention / DeltaNet input GEMMs wait on their norm's counters (iu4_wait_ = nq_cnt_ around
  // them; iu4_kstorew_*, iu4_kstorefw_*, iu4_kqgw_*)
  bool nqw_ = false;
  hrx_buffer_ref_t iu4_wait_{};
  hrx_buffer_ref_t nq_cnt_{};
  LoomBuffer *aq_ = nullptr, *ad_ = nullptr, *aq2_ = nullptr, *ad2_ = nullptr, *flags_ = nullptr;
  std::size_t flag_slot_ = 0;
  LoomBuffer *grid_iq3s_ = nullptr, *grid_iq3xxs_ = nullptr, *grid_iq2xxs_ = nullptr, *grid_iq2xs_ = nullptr,
             *ksigns_ = nullptr;
  LoomBuffer *hidden_ = nullptr, *reszero_ = nullptr, *sumout_ = nullptr, *scratch_ = nullptr, *qkv_ = nullptr,
             *gate_ = nullptr, *alpha_ = nullptr, *beta_ = nullptr, *q_ = nullptr, *q16_ = nullptr, *kbuf_ = nullptr,
             *vbuf_ = nullptr, *raw_ = nullptr, *raw2_ = nullptr, *conv_out_ = nullptr, *kqbuf_ = nullptr, *ab_ = nullptr,
             *conv_state_ = nullptr, *state_ = nullptr, *kv16_ = nullptr, *kc32_ = nullptr, *vc32_ = nullptr,
             *lse_ = nullptr, *eps_ = nullptr, *ffnup_ = nullptr, *gateffn_ = nullptr, *uwstage_ = nullptr,
             *wstage_ = nullptr, *ostage_ = nullptr, *partial_ = nullptr, *hidden2_ = nullptr, *normed_ = nullptr,
             *ptab_ = nullptr, *vt16_ = nullptr, *kq8buf_ = nullptr, *ksbuf_ = nullptr, *kmbuf_ = nullptr,
             *vqbuf_ = nullptr, *vqsbuf_ = nullptr, *kpool_ = nullptr, *vtpool_ = nullptr, *valid_ = nullptr, *noans_ = nullptr, *qr16_ = nullptr,
             *kdeq_ = nullptr, *vdeq_ = nullptr, *ids_ = nullptr,
             *normt_ = nullptr;
  std::int64_t keep_rows_ = kAllRows, tail_ = kAllRows, trim_row_ = kAllRows;
  bool answer_ = false;
};

}  // namespace yah::model

#endif  // YAH_MODEL_LOOM_PREFILL_HPP_
