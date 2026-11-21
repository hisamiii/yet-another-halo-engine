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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "core/gguf_quant.hpp"
#include "model/loom_decoder.hpp"
#include "model/loom_runtime.hpp"
#include "model/npu_prof.hpp"
#include "model/npu_split.hpp"

namespace yah::model {

class LoomPrefill {
 public:
  static constexpr std::uint32_t kHidden = 5120, kFfn = 17408, kAttn = 6144, kQProj = 12288, kKv = 1024;
  static constexpr std::uint32_t kInner = 6144, kQkv = 10240, kTs = 48, kKh = 16, kState = 128;
  static constexpr std::uint32_t kHeads = 24, kKvHeads = 4, kHeadDim = 256, kVocab = 248320;
  static constexpr std::uint32_t kKvRow = kKvHeads * kHeadDim;  // one KV cache row: all KV heads of one token

  // The K-segmented ffn_down tensors of the set in dir (dispatch.txt "npudnseg_<fmt> <F> <m tiles> <k blocks>",
  // emit_prefill_pp YAH_NPU_DNSEG): every ffn_down of such a format and shape, its segments the GPU's K part [0,
  // K - F) and the NPU's K parts (FfnBlkParts) in order. LoomWeights holds these rows so; the set's kernels read them so.
  // ("npudnsegl_<layer> <F>": that layer's down in the layout of its own F, emit_prefill_pp YAH_NPU_FFNBLKS: the
  // single-format layers' F)
  static std::map<std::string, KSeg> DownKSegs(const std::string& dir, const core::Gguf& gguf) {
    std::ifstream f(dir + "/dispatch.txt");
    std::map<std::string, Geom> seg;   // fmt -> (F, m tiles, k blocks)
    std::map<std::string, std::uint32_t> lf;   // layer -> F
    std::string name;
    Geom g{};
    while (f >> name >> g.tokens >> g.rowgrp >> g.tt)
      if (name.starts_with("npudnseg_")) seg[name.substr(9)] = g;
      else if (name.starts_with("npudnsegl_")) lf[name.substr(10)] = g.tokens;
    std::map<std::string, KSeg> out;
    for (const auto& t : gguf.tensors()) {
      Fmt fm{};
      if (!t.name.ends_with(".ffn_down.weight") || !FmtOf(static_cast<std::uint32_t>(t.type), &fm)) continue;
      const auto it = seg.find(fm.name);
      if (it == seg.end() || t.dims[1] != std::uint64_t{it->second.rowgrp} * 16 ||
          t.dims[0] != std::uint64_t{it->second.tt} * 256)
        continue;
      const auto l = lf.find(t.name.substr(4, t.name.find('.', 4) - 4));   // ("blk.<N>.ffn_down.weight")
      out[t.name] = KSegFor(l == lf.end() ? it->second.tokens : l->second, it->second.tt);
    }
    return out;
  }
  // The row-segmented gate / up tensors of the set in dir (dispatch.txt "npursegl_<layer> <F>", emit_prefill_pp
  // FFNSW5): both of that layer's, at F (RSegPhys). LoomWeights holds these rows so; the set's kernels read them so.
  static std::map<std::string, std::uint32_t> GateUpRSegs(const std::string& dir) {
    std::ifstream f(dir + "/dispatch.txt");
    std::map<std::string, std::uint32_t> out;
    std::string name;
    Geom g{};
    while (f >> name >> g.tokens >> g.rowgrp >> g.tt)
      if (name.starts_with("npursegl_"))
        for (const char* w : {".ffn_gate.weight", ".ffn_up.weight"}) out["blk." + name.substr(9) + w] = g.tokens;
    return out;
  }
  // The K segments of a down of kb blocks under an FFN block of F features: the GPU's K part [0, K - F), then the
  // NPU's K parts (FfnBlkUnits) in order.
  static KSeg KSegFor(std::uint32_t F, std::uint32_t kb) {
    KSeg s{{0, kb - F / 256}};
    for (const FfnPart& u : FfnBlkUnits(F)) s.push_back({kb - F / 256 + u.u0, u.r});
    return s;
  }

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
    // YAH_PP_HALVES=1 | 2 (port/design-overlap.md): a chunk of 2 B tokens as two halves of this B set in one graph and
    // one NPU queue, each half with its own token buffers (a second bank) and NPU A / C regions; 1: half 0's layers,
    // then half 1's; 2: the halves alternate at every GPU wait for an NPU job (NpuJoin)
    if (const char* h = std::getenv("YAH_PP_HALVES"); h && *h && std::strcmp(h, "0") != 0) {
      halves_ = static_cast<std::uint32_t>(std::atoi(h));
      if ((halves_ != 1 && halves_ != 2) || T_ < 2 * B_)
        throw LoomError("YAH_PP_HALVES: 1 or 2, on a chunked set whose context holds two chunks");
    }
    for (std::uint32_t l = 0; l < cfg_.main_block_count(); ++l)
      if (cfg_.IsFullAttention(l)) ++full_;
    LoadTables();
    AllocateBuffers();
    if (halves_) AllocHalfBank();
    LoadExecutables();
    WarmExecutables();
  }

  [[nodiscard]] std::uint32_t chunk() const { return B_; }
  // the caller's chunk: 2 B with YAH_PP_HALVES (two halves of this set's B), else B
  [[nodiscard]] std::size_t carriers(std::uint32_t h) const { return carriers_[h]; }   // carrier dispatches, half h
  [[nodiscard]] std::uint32_t chunk_tokens() const { return halves_ ? 2 * B_ : B_; }
  [[nodiscard]] std::uint32_t context() const { return T_; }
  [[nodiscard]] bool paged() const { return kv_paged_; }
  // KV bits of this set: (16, 16) fp16, else 8 or 4 per side.
  [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> kv_bits() const {
    return {attn_kq4_ ? 4u : attn_kq8_ ? 8u : 16u, attn_vq4_ ? 4u : attn_vq8_ ? 8u : 16u};
  }
  LoomBuffer& hidden() { return halves_ ? *hs_[1].hidden_ : *hidden_; }

  // Buffer sizes of the NPU column split this set carries (dispatch.txt "npusplit_<site>", "npubytes_<K>"); zero: none.
  // A job is the NPU work behind one cut: the DeltaNet qkv + gate, or one other site; the buffers fit the largest
  // (W: one call's unread panel binding, the NPU decodes raw rows).
  // (YAH_PP_HALVES) A, W and C hold two halves' regions, each one B plan's, 4096-aligned; the swap's weight stream at
  // A's end stays one (both halves run a layer's same weights: the first half to reach the layer decodes it)
  [[nodiscard]] NpuPlan npu_plan() const {
    NpuPlan p = NpuPlanOne();
    if (halves_) {
      if (R5Fuse() && !R5Part())
        throw LoomError("YAH_PP_HALVES: whole r5 carriers do not split by half (emit YAH_NPU_R5PART=1 or YAH_NPU_R5CAR=0)");
      if (PanelQ()) throw LoomError("YAH_PP_HALVES: YAH_NPU_PANELQ is not built for halves");
      p.a_bytes = 2 * HalfA(p) + SwapAll(), p.w_bytes = 2 * Up4k(p.w_bytes), p.c_bytes = 2 * Up4k(p.c_bytes);
    }
    return p;
  }
  std::size_t SwapAll() const { return RSwapOn() ? (R5Fuse() ? 2 : 1) * SwapStreamMax() : 0; }
  // one half's A region: the one-B plan's A before the swap's stream (NpuPlanOne rounds it to 4096 when the swap is on)
  std::size_t HalfA(const NpuPlan& one) const { return Up4k(one.a_bytes - SwapAll()); }
  static std::size_t Up4k(std::size_t b) { return (b + 4095) / 4096 * 4096; }
  [[nodiscard]] NpuPlan NpuPlanOne() const {
    NpuPlan p;
    const auto need = [&](const std::string& site) {
      const auto it = geom_.find("npusplit_" + site);
      NpuPlan q;
      if (it == geom_.end()) return q;
      std::size_t rows = it->second.tokens;   // (and the largest per-layer row, "npusplit_<site>@<L>")
      for (auto j = geom_.lower_bound("npusplit_" + site + "@"); j != geom_.end() && j->first.rfind("npusplit_" + site + "@", 0) == 0; ++j)
        rows = std::max<std::size_t>(rows, j->second.tokens);
      const std::size_t calls = rows / NpuCallRows();
      for (const auto& [k0, K] : NpuChunks(site)) {
        const NpuBytes nb = NpuK(K);
        q.a_bytes += nb.a, q.c_bytes += calls * nb.c;
      }
      if (const auto i4 = geom_.find("npusplit4_" + site); i4 != geom_.end()) {   // (the NP 4 formats' layers)
        std::size_t c4 = 0;
        for (const auto& [k0, K] : NpuChunks(site)) c4 += i4->second.tokens / NpuCallRows(true) * NpuK(K, true).c;
        q.c_bytes = std::max(q.c_bytes, c4);
      }
      return q;
    };
    NpuPlan qkv = need("qkv"), gate = need("gate");   // (C: qkv's + gate's of either family, an upper bound)
    std::size_t sw_a = 0;
    if (const std::uint32_t rq = SwSiteMaxR("qkv"), rg = SwSiteMaxR("gate"); rq || rg) {   // (swap-form sites)
      if (!rq || !rg) throw LoomError("prefill: swap-form qkv sites without gate sites, or the reverse");
      if (halves_) throw LoomError("YAH_PP_HALVES: swap-form sites are not built for halves");
      sw_a = SwSiteGateA() + NpuBytesOf("npubytes_swsite_" + std::to_string(rg)).a;
      qkv.c_bytes = std::max(qkv.c_bytes, SwSiteC(rq)), gate.c_bytes = std::max(gate.c_bytes, SwSiteC(rg));
      p.w_bytes = std::max(p.w_bytes, std::size_t{geom_.at("npu_swpanel.hal").tt});
    }
    // (out: a DeltaNet layer's out C follows its qkv and gate C)
    NpuPlan qp = need("q");
    if (const std::uint32_t r = SwSiteMaxR("q")) {   // (a swap-form q site: its stream at A 0, C B R bf16)
      if (halves_) throw LoomError("YAH_PP_HALVES: swap-form sites are not built for halves");
      qp.a_bytes = std::max(qp.a_bytes, NpuBytesOf("npubytes_swsite_" + std::to_string(r)).a);
      qp.c_bytes = std::max(qp.c_bytes, SwSiteC(r));
      p.w_bytes = std::max(p.w_bytes, std::size_t{geom_.at("npu_swpanel.hal").tt});
    }
    NpuPlan op = need("out");
    if (const std::uint32_t r = SwSiteMaxR("out")) {   // (a swap-form out site: its stream at A 0, C B R bf16)
      if (halves_) throw LoomError("YAH_PP_HALVES: swap-form sites are not built for halves");
      if (!geom_.count("npu_swpanel_o.hal") || !geom_.count("npu_swpanel_o_t.hal"))
        throw LoomError("prefill: swap-form out sites without npu_swpanel_o[_t].hal (a set from before out swap): re-emit it");
      op.a_bytes = std::max(op.a_bytes, NpuBytesOf("npubytes_swsite_" + std::to_string(r)).a);
      op.c_bytes = std::max(op.c_bytes, SwSiteC(r));
      p.w_bytes = std::max(p.w_bytes, std::size_t{geom_.at("npu_swpanel_o.hal").tt});
    }
    for (const NpuPlan& q : {NpuPlan{std::max({qkv.a_bytes, gate.a_bytes, sw_a}), 0, qkv.c_bytes + gate.c_bytes + op.c_bytes},
                             qp, op})
      p.a_bytes = std::max(p.a_bytes, q.a_bytes), p.c_bytes = std::max(p.c_bytes, q.c_bytes);
    if (FfnBlkF()) {   // the FFN block: gate / up's activations; H and down's partials in C
      p.a_bytes = std::max(p.a_bytes, NpuK(5120).a);
      const auto parts = FfnBlkUnits(FfnBlkF());
      const std::string tag = DnTag(parts[0], FfnBlkApass(FfnBlkF()));
      p.c_bytes = std::max(p.c_bytes, FfnBlkDown() + parts.size() * FfnBlkCalls() * NpuBytesOf("npubytes_ffndn" + tag).c);
      if (geom_.count("npuffnblk4"))
        p.c_bytes = std::max(p.c_bytes, FfnBlkDown() + parts.size() * FfnBlkCalls(true) *
                                                           NpuBytesOf("npubytes4_ffndn" + tag).c);
    }
    if (SwDown()) {   // (the down in swap form) its weight streams at A 0, a K part each; C a (part, panel) each
      if (halves_) throw LoomError("YAH_PP_HALVES: the swap-form down is not built for halves");
      const NpuBytes b = NpuBytesOf("npubytes_swdown");
      p.a_bytes = std::max(p.a_bytes, 2 * Up4k(b.a));
      p.c_bytes = std::max(p.c_bytes, FfnBlkDown() + 2 * std::size_t{B_ / 512} * b.c);
    }
    if (OutCAfterH())   // (PanelQ: the out site's C after the FFN block's H, from FfnBlkDown)
      p.c_bytes = std::max(p.c_bytes, FfnBlkDown() + op.c_bytes);
    p.dcol = geom_.count("npudcol") != 0;
    p.gfirst = geom_.count("npugfirst") != 0;
    p.depth1 = geom_.count("npudepth1") != 0;
    if (p.dcol) {
      // the NPU decodes raw rows: W is one call's unread panel binding; A holds the p4 stream, then the pk one (Q4_K)
      p.w_bytes = std::max({p.w_bytes, NpuK(5120).w, geom_.count("npubytes4_5120") ? NpuK(5120, true).w : 0});
      p.a_bytes *= 2;
    }
    if (RSwapOn()) {   // (YAH_NPU_RSWAP) the swap's weight stream at A's end (SwapA), its token panel at W's start
      const NpuBytes bw = NpuBytesOf("npubytes_ffnswap");
      p.a_bytes = (p.a_bytes + 4095) / 4096 * 4096 + (R5Fuse() ? 2 : 1) * SwapStreamMax();
      p.w_bytes = std::max(p.w_bytes, std::size_t{B_} / 512 * bw.w);
    }
    if (const auto it = geom_.find("npugate"); it != geom_.end())
      p.gate_calls = it->second.rowgrp, p.gate_record = it->second.tt;
    if (const auto it = geom_.find("npupanel"); it != geom_.end()) p.panel = it->second.tokens;
    return p;
  }
  // YAH_NPU_WAITLOG: after the run, one line per GPU wait for an NPU job (layer, the job's tag, polls of its done word,
  // ~0.2 us each) and the polls per tag. Waits inside a GEMM epilogue (NPUADD) are not logged.
  void ReportWaitLog(std::FILE* f) {
    if (!waitlog_) return;
    std::vector<std::uint32_t> polls(waitlog_tag_.size());
    gpu_.D2H(*waitlog_, polls.data(), polls.size() * 4);
    std::map<std::string, std::uint64_t> per_tag;
    for (std::size_t i = 0; i < polls.size(); ++i) {
      std::fprintf(f, "npu wait: %zu %s %u\n", i, waitlog_tag_[i].c_str(), polls[i]);
      per_tag[waitlog_tag_[i].substr(waitlog_tag_[i].find(' ') + 1)] += polls[i];
    }
    for (const auto& [tag, n] : per_tag)
      std::fprintf(f, "npu wait: %-8s %llu polls\n", tag.c_str(), static_cast<unsigned long long>(n));
  }
  // YAH_CLK_PROBE: one probe (16-byte slot: t0, t1, c0, c1), ordered after the hidden state's last write as if it read
  // it. The graph runs a dispatch with dependencies after every earlier one (mem-clkffn-gate), so the probe runs alone
  // for its 20 us and the next kernel after it: ~1.5 ms a run with the probe on.
  void ClkProbe() {
    if (!clk_ || npu_planning_ || warming_ || halves_) return;
    ClkDispatch("clk_probe.hal", "yah_clk_probe", {{clk_->handle, ClkSlot() * 16, 16}}, {Ref(*hidden_)});
  }
  std::size_t ClkSlot() {
    if (clk_n_ == kClkSlots) throw LoomError("YAH_CLK_PROBE: more probes than slots");
    return clk_n_++;
  }
  void ClkDispatch(const char* hal, const char* name, const std::vector<hrx_buffer_ref_t>& b,
                   const std::vector<hrx_buffer_ref_t>& after) {
    if (split_armed_ && graph_) SplitGraph();
    const LoomExecutable& exe = Exe(hal);
    const std::uint32_t ordinal = exe.OrdinalOrZero(name);
    const hrx_dispatch_config_t config = LoomDevice::Config(1, 1, 1, 32, 1, 1);
    if (graph_)
      graph_->Dispatch(exe, ordinal, config, b.data(), b.size(), 1, &after);
    else
      gpu_.Dispatch(exe, ordinal, config, nullptr, 0, b.data(), b.size());
  }
  // YAH_CLK_PROBE=ffn, around one FFN gate / up GEMM (bindings b): a probe ordered after what it reads (it runs right
  // before the GEMM: the graph runs a dispatch with dependencies after every earlier one) and one after what it writes.
  // The probe can not run inside the GEMM in this graph (mem-clkffn2-gate: a dispatch with no dependencies launches as
  // a root, ahead of the chunk), so the GEMM's clock is its two edge clocks.
  struct ClkFfn {
    std::uint32_t layer;
    bool carry;
    std::size_t edge, end;
  };
  void ClkFfnBegin(const std::vector<hrx_buffer_ref_t>& b, bool carry) {
    if (!clk_ffn_on_ || !clk_ || npu_planning_ || warming_ || halves_) return;
    clk_ffn_.push_back({layer_, carry, ClkSlot(), 0});
    ClkDispatch("clk_probe.hal", "yah_clk_probe", {{clk_->handle, clk_ffn_.back().edge * 16, 16}}, b);
  }
  void ClkFfnEnd(const std::vector<hrx_buffer_ref_t>& b, std::uint64_t writes) {
    if (!clk_ffn_on_ || !clk_ || npu_planning_ || warming_ || halves_) return;
    std::vector<hrx_buffer_ref_t> out;
    for (std::size_t i = 0; i < b.size(); ++i)
      if ((writes >> i) & 1) out.push_back(b[i]);
    clk_ffn_.back().end = ClkSlot();
    ClkDispatch("clk_probe.hal", "yah_clk_probe", {{clk_->handle, clk_ffn_.back().end * 16, 16}}, out);
  }
  // YAH_CLK_PROBE=ffn: a line per GEMM call ("clk ffn: <layer> <carrier|plain> gemm <ms> edges <MHz> <MHz> <Mcyc>":
  // REALTIME from the probe before it to the one after, the two probes' clocks, ms x their mean) and one per class
  // (calls, mean ms, mean edge clocks, total and mean Mcyc)
  void ReportClkFfn(std::FILE* f, const std::vector<std::uint32_t>& p) {
    if (clk_ffn_.empty()) return;
    const auto mhz = [&](std::size_t i) {
      const std::uint32_t dt = p[4 * i + 1] - p[4 * i];
      return dt ? ((p[4 * i + 3] - p[4 * i + 2]) & 0xfffffu) * 100.0 / dt : 0.0;
    };
    struct Sum { std::size_t n = 0; double ms = 0, before = 0, after = 0, mcyc = 0; };
    std::map<std::string, Sum> cls;
    for (const ClkFfn& c : clk_ffn_) {
      const double ms = (p[4 * c.end] - p[4 * c.edge]) * 1e-5, a = mhz(c.edge), z = mhz(c.end);
      const char* k = c.carry ? "carrier" : "plain";
      const double mc = ms * (a + z) * 0.5e-3;
      std::fprintf(f, "clk ffn: %u %s gemm %.3f ms edges %.0f %.0f MHz %.2f Mcyc\n", c.layer, k, ms, a, z, mc);
      Sum& s = cls[k];
      ++s.n, s.ms += ms, s.before += a, s.after += z, s.mcyc += mc;
    }
    for (const auto& [k, s] : cls)
      std::fprintf(f, "clk ffn: class %s n %zu gemm %.3f ms mean, edges %.0f %.0f MHz mean, %.1f Mcyc total, %.2f mean\n",
                   k.c_str(), s.n, s.ms / s.n, s.before / s.n, s.after / s.n, s.mcyc, s.mcyc / s.n);
  }
  // YAH_CLK_PROBE: after the run, one line: the GPU span from the first probe to the last (REALTIME), the flag waits
  // inside it, work = span - waits, and work in G cycles (each interval between probes at the mean clock of its two
  // probes) with the ms that is at 2600 MHz; the probes' clock mean / min / max. YAH_CLK_PROBE=2: a line per interval.
  void ReportClk(std::FILE* f) {
    if (!clk_ || clk_n_ < 2) return;
    std::vector<std::uint32_t> p(clk_n_ * 4), s(stamp_n_ * 2);
    gpu_.D2H(*clk_, p.data(), p.size() * 4);
    if (stamp_n_) gpu_.D2H(*stamp_, s.data(), s.size() * 4);
    ReportClkFfn(f, p);
    std::vector<double> mhz(clk_n_);
    double fsum = 0, fmin = 1e9, fmax = 0;
    for (std::size_t i = 0; i < clk_n_; ++i) {
      const std::uint32_t dt = p[4 * i + 1] - p[4 * i], dc = (p[4 * i + 3] - p[4 * i + 2]) & 0xfffffu;
      mhz[i] = dt ? dc * 100.0 / dt : 0;
      fsum += mhz[i], fmin = std::min(fmin, mhz[i]), fmax = std::max(fmax, mhz[i]);
    }
    const bool each = std::string(std::getenv("YAH_CLK_PROBE")) == "2";
    const std::uint32_t a0 = p[0], span_t = p[4 * (clk_n_ - 1)] - a0;
    double waits = 0, cyc = 0, outside = 0;
    std::vector<double> wi(clk_n_, 0);
    for (std::size_t k = 0; k < stamp_n_; ++k) {
      const std::uint32_t te = s[2 * k], d = s[2 * k + 1] - te, at = te - a0;
      if (at >= span_t) { outside += d * 1e-5; continue; }
      std::size_t i = 0;
      while (i + 2 < clk_n_ && p[4 * (i + 1)] - a0 <= at) ++i;
      wi[i] += d * 1e-5, waits += d * 1e-5;
    }
    for (std::size_t i = 0; i + 1 < clk_n_; ++i) {
      const double len = (p[4 * (i + 1)] - p[4 * i]) * 1e-5, work = len - wi[i];
      cyc += work * 1e-3 * (mhz[i] + mhz[i + 1]) * 0.5e6;
      if (each) std::fprintf(f, "clk: %zu %.0f MHz, %.3f ms, waits %.3f ms\n", i, mhz[i], len, wi[i]);
    }
    std::fprintf(f, "clk: span %.2f ms, waits %.2f ms (%zu, %.2f ms outside), work %.2f ms, work %.4f Gcyc = %.2f ms at "
                 "2600 MHz; clock %.0f MHz mean, %.0f..%.0f, %zu probes\n", span_t * 1e-5, waits, stamp_n_, outside,
                 span_t * 1e-5 - waits, cyc * 1e-9, cyc / 2.6e6, fsum / clk_n_, fmin, fmax, clk_n_);
    std::fflush(f);   // (before the NPU's report, which writes to stderr: no lines cut into each other)
  }
  // YAH_NPU_PROF in a GPU + NPU run: after it, the first NPU chunk's per-job counters as CPU-driven runs print them
  // (tools/npu_prof.py; no per-job host time: 0).
  void ReportNpuProf() {
    NpuProf* const prof = npu_ ? npu_->Prof() : nullptr;
    if (!prof || prof_tags_.empty()) return;
    if (std::getenv("YAH_NPU_PROF_MID")) {   // (diag) every record in log order (npu recmap: their kinds)
      prof->Finish(0);
      const std::vector<std::uint32_t> v = prof->Peek();
      for (std::size_t r = 0; r * prof->Values() < v.size(); ++r)
        std::fprintf(stderr, "npu prof: rec %zu %u %u %u\n", r, v[r * prof->Values()],
                     v[r * prof->Values() + prof->Values() - 2], v[r * prof->Values() + prof->Values() - 1]);
      return;
    }
    const std::vector<std::uint32_t> v = prof->Finish(prof_tags_.size());
    if (prof->Clock()) {   // (YAH_NPU_PROF=clock) "npu clk: jobs <n> busy <ticks> span <ticks> idle <ticks>": NPU work
      std::uint64_t busy = 0;   // in array cycles, its span from the first gate to the last done, the rest idle;
      std::map<std::string, std::pair<std::size_t, std::uint64_t>> tag;   // then "npu clk: tag <tag> jobs <n> busy <ticks>"
      const auto& ts = prof->times();
      for (std::size_t i = 0; i < ts.size(); ++i) {
        busy += ts[i].done - ts[i].gate;
        auto& [n, b] = tag[i < prof_tags_.size() ? prof_tags_[i] : "?"];
        ++n, b += ts[i].done - ts[i].gate;
      }
      const std::uint64_t span = ts.empty() ? 0 : ts.back().done - ts.front().gate;
      std::fprintf(stderr, "npu clk: jobs %zu busy %llu span %llu idle %llu\n", ts.size(),
                   static_cast<unsigned long long>(busy), static_cast<unsigned long long>(span),
                   static_cast<unsigned long long>(span > busy ? span - busy : 0));
      for (const auto& [t, nb] : tag)
        std::fprintf(stderr, "npu clk: tag %s jobs %zu busy %llu\n", t.c_str(), nb.first,
                     static_cast<unsigned long long>(nb.second));
      PrintNpuTimes(*prof, prof_tags_, prof_layers_);
      return;
    }
    std::fprintf(stderr, "npu prof: events %s\n", std::getenv("YAH_NPU_PROF"));
    for (std::size_t i = 0; i < prof_tags_.size(); ++i) {
      // (the field after the tag: the job's counter setting, YAH_NPU_PROF_ROT)
      std::fprintf(stderr, "npu prof: job %zu %s %u", i, prof_tags_[i].c_str(),
                   i < prof->chosen().size() ? prof->chosen()[i] : 0u);
      for (std::uint32_t k = 0; k < NpuProf::kValues; ++k) std::fprintf(stderr, " %u", v[i * NpuProf::kValues + k]);
      std::fprintf(stderr, "\n");
    }
    PrintNpuTimes(*prof, prof_tags_, prof_layers_);
  }
  // "npu time: <job> <layer> <tag> <gate> <done>": when each job's gate opened and its done landed (array cycles;
  // NpuProf::Times)
  static void PrintNpuTimes(const NpuProf& prof, const std::vector<std::string>& tags,
                            const std::vector<std::uint32_t>& layers) {
    for (std::size_t i = 0; i < tags.size() && i < prof.times().size(); ++i) {
      const NpuProf::Times& t = prof.times()[i];
      std::fprintf(stderr, "npu time: %zu %u %s %llu %llu\n", i, i < layers.size() ? layers[i] : 0u, tags[i].c_str(),
                   static_cast<unsigned long long>(t.gate), static_cast<unsigned long long>(t.done));
    }
  }
  // Split the set's NPU sites of full chunks with the NPU from now on: it computes their trailing npusplit rows.
  // npu (built from npu_plan()) outlives this object.
  void EnableNpu(NpuSplit* npu) {
    for (const auto& [name, g] : geom_)
      if (name.rfind("npusplit_", 0) == 0 && name.find('@') == std::string::npos) npu_rows_[name.substr(9)] = g.tokens;
      else if (name.rfind("npusplit4_", 0) == 0) npu_rows4_[name.substr(10)] = g.tokens;
    ffnblk_ = FfnBlkF();
    if (!npu || (npu_rows_.empty() && !ffnblk_)) throw LoomError("prefill: the set has no NPU split");
    if (!geom_.count("npu_flag_wait.hal") || !geom_.count("npugate"))
      throw LoomError("prefill: the set has no gated NPU handoffs (npu_flag_wait.hal, npugate; re-emit it)");
    npu_ = npu;
    if (!geom_.count("npudcol"))
      throw LoomError("prefill: the set's NPU split decodes no raw rows (npudcol; re-emit it)");
    std::size_t rem = 0;
    for (const auto& [site, rows] : npu_rows_)
      if (NpuRem(site)) rem = std::max<std::size_t>(rem, std::size_t{B_} * rows * 4);
    for (const auto& [name, g] : geom_)   // (and the per-layer rows, "npusplit_<site>@<L>")
      if (name.rfind("npusplit_", 0) == 0 && name.find('@') != std::string::npos &&
          NpuRem(name.substr(9, name.find('@') - 9)))
        rem = std::max<std::size_t>(rem, std::size_t{B_} * g.tokens * 4);
    if (rem) npurem_ = &Alloc(rem);
    if (ffnblk_ && geom_.count("npu_unpack_ffnblk.hal")) ffnrem_ = &Alloc(std::size_t{B_} * kHidden * 4);   // (npuffnrem)
    if (halves_) {   // (YAH_PP_HALVES) half 1's remainders; each half's NPU regions
      const NpuPlan one = NpuPlanOne();
      a_one_ = HalfA(one), w_one_ = Up4k(one.w_bytes), c_one_ = Up4k(one.c_bytes);
      for (std::uint32_t h = 0; h < 2; ++h) {
        hs_[h].npu_a0_ = h * a_one_, hs_[h].npu_w0_ = h * w_one_, hs_[h].npu_c0_ = h * c_one_;
        if (h == 1) {
          if (npurem_) hs_[1].npurem_ = &Alloc(rem);
          if (ffnrem_) hs_[1].ffnrem_ = &Alloc(std::size_t{B_} * kHidden * 4);
        } else {
          hs_[0].npurem_ = npurem_, hs_[0].ffnrem_ = ffnrem_;
        }
      }
      half_ = 0;
      npu_a0_ = 0, npu_w0_ = 0, npu_c0_ = 0;
    }
    if (std::getenv("YAH_NPU_WAITLOG")) {
      if (!geom_.count("npu_flag_wait_log.hal"))
        throw LoomError("YAH_NPU_WAITLOG: the set has no npu_flag_wait_log.hal (re-emit it)");
      waitlog_ = &Alloc(kWaitLogSlots * 4);
    }
    // Bind every NPU call of a full chunk now (a pass of the layers that dispatches nothing): a first bind loads the
    // image's storage and patches it, milliseconds per call.
    LoomBuffer *h = hidden_, *h2 = hidden2_;
    const std::uint32_t n = n_;
    n_ = B_, npu_on_ = true, npu_planning_ = true;
    npu_job_ = 0;
    r5_.clear();
    NewGraph();
    if (halves_) {   // (YAH_PP_HALVES) each half's calls, on its own A / C regions (chunks 0 and 1 of this set)
      for (std::uint32_t hh = 0; hh < 2; ++hh) {
        LoadHalf(hh);
        n_ = B_;
        RecordLayers(hh, {});
        SaveHalf(hh);
      }
      LoadHalf(0);
    } else {
      RecordLayers(0, {});
    }
    npu_->EndBinding();
    chunk_graph_.reset();
    graph_ = nullptr, n_ = n, npu_on_ = false, npu_planning_ = false;
    hidden_ = h, hidden2_ = h2;
  }

  // Split the attention of chunks from the set's first NPU chunk with the NPU (dispatch.txt "npuattn <kvn> <nb built>
  // <first chunk>", emit YAH_NPU_ATTN): per full-attention layer the encoders write the NPU's records and payloads, the
  // graph stores the layer's ready word, kernel 1 (the key-split twin) runs beside the NPU, a wait polls the done word
  // and kernel 2 merges the NPU's piece. npu (a LoomNpuAttn) outlives this object.
  [[nodiscard]] bool npu_attn_set() const { return geom_.count("npuattn") != 0; }
  void EnableNpuAttn(NpuAttn* npu) {
    const auto g = geom_.find("npuattn");
    if (!npu || g == geom_.end())
      throw LoomError("prefill: the set has no NPU attention (emit_prefill_pp YAH_NPU_ATTN)");
    if (!kv_paged_) throw LoomError("prefill: the NPU attention needs paged pools");
    na_kvn_ = g->second.tokens, na_nbs_ = g->second.rowgrp, na_from_ = g->second.tt;
    if (na_from_ == 0 || NaReady(full_) > 1024) throw LoomError("prefill: npuattn row out of range");
    na_ = npu->Setup(dir_ + "/npu_attn.xdna", na_kvn_, na_nbs_, B_);
    npu_attn_ = npu;
  }

  // Zero the recurrent state before a new sequence. KV rows need no reset: attention reads only keys <= the query.
  void Reset() {
    Zero(*conv_state_);
    Zero(*state_);
  }
  // The recurrent state (conv rings, DeltaNet states) that chunks carry forward and DecoderState() hands on.
  [[nodiscard]] hrx_buffer_ref_t conv_state() const { return Ref(*conv_state_); }
  [[nodiscard]] hrx_buffer_ref_t delta_state() const { return Ref(*state_); }

  // Stage ids[0..n) for the next RunLayers, whose first kernel (yah_embed) writes their token_embd rows into hidden().
  // n < B pads the chunk with the last token.
  // The GEMMs (~92% of the time) run only the token tiles that hold real tokens; every other kernel runs
  // the whole chunk, so padding rows next to the real ones keep their padded-path values. Shrinking the norms as well
  // changed the last token's logits by ~1e-4 when it sat alone in its 16-token group (cause not found yet). Padding tokens leave the recurrent state
  // unchanged and attention is causal, so the real tokens' results and the state carried forward are exact.
  // The ids and the token count go to the GPU in stream order (stream updates): the host never waits here.
  void Embed(const std::uint32_t* ids, std::uint32_t n) {
    if (halves_ && !in_half_embed_) {   // (YAH_PP_HALVES) full chunks of 2 B only: each half its own bank's rows
      if (n != 2 * B_) throw LoomError("YAH_PP_HALVES: a chunk of 2 B tokens only");
      static const bool ov = !(std::getenv("YAH_EMBED_OVERLAP") && !std::strcmp(std::getenv("YAH_EMBED_OVERLAP"), "0"));
      if (embed_rows_ && ov) {   // both halves' rows in one gather on the thread (host_rows_ holds 2 B), FinishEmbed
        for (std::uint32_t t = 0; t < 2 * B_; ++t)
          if (ids[t] >= kVocab) throw LoomError("token id " + std::to_string(ids[t]) + " is outside the vocabulary");
        FinishEmbed();
        gather_ids_.assign(ids, ids + 2 * B_);
        half_rows_ = true;
        gather_.t = std::thread([this] {
          try {
            const auto e0 = std::chrono::steady_clock::now();
            LoomDevice::WaitFor(rows_copied_);
            const auto e1 = std::chrono::steady_clock::now();
            gguf_.Rows(*Find("token_embd.weight"), gather_ids_.data(), 2 * B_, host_rows_.data());
            gather_ms_[0] = std::chrono::duration<double, std::milli>(e1 - e0).count();
            gather_ms_[1] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e1).count();
          } catch (...) {
            gather_error_ = std::current_exception();
          }
        });
        for (std::uint32_t t = 0; t < B_; ++t) host_ids_[t] = t;
        const std::int32_t valid = static_cast<std::int32_t>(B_);
        for (std::uint32_t h = 0; h < 2; ++h) {
          LoadHalf(h);
          gpu_.Update(*ids_, host_ids_.data(), host_ids_.size() * 4);
          gpu_.Update(*valid_, &valid, 4);
          n_ = B_;
          SaveHalf(h);
        }
        LoadHalf(0);
        return;
      }
      in_half_embed_ = true;
      for (std::uint32_t h = 0; h < 2; ++h) {
        LoadHalf(h);
        Embed(ids + std::size_t{h} * B_, B_);
        SaveHalf(h);
      }
      in_half_embed_ = false;
      LoadHalf(0);
      return;
    }
    if (n == 0 || n > B_) throw LoomError("prefill: chunk token count out of range");
    for (std::uint32_t t = 0; t < B_; ++t) {
      host_ids_[t] = ids[std::min(t, n - 1)];
      if (host_ids_[t] >= kVocab)
        throw LoomError("token id " + std::to_string(host_ids_[t]) + " is outside the vocabulary");
    }
    FinishEmbed();   // (a gather still running from an Embed with no RunLayers after it)
    // The rows' gather runs on a thread while RunLayers records the chunk (neither reads the other's data); the
    // first graph launch joins it and streams the rows (FinishEmbed). YAH_EMBED_OVERLAP=0: gathered here.
    static const bool overlap0 = !(std::getenv("YAH_EMBED_OVERLAP") && !std::strcmp(std::getenv("YAH_EMBED_OVERLAP"), "0"));
    const bool overlap = overlap0 && !halves_;   // (halves: one gather thread would stream into the wrong bank)
    if (embed_rows_ && overlap) {
      gather_ids_.assign(host_ids_.begin(), host_ids_.end());
      gather_.t = std::thread([this] {
        try {
          const auto e0 = std::chrono::steady_clock::now();
          LoomDevice::WaitFor(rows_copied_);
          const auto e1 = std::chrono::steady_clock::now();
          gguf_.Rows(*Find("token_embd.weight"), gather_ids_.data(), B_, host_rows_.data());
          gather_ms_[0] = std::chrono::duration<double, std::milli>(e1 - e0).count();
          gather_ms_[1] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e1).count();
        } catch (...) {
          gather_error_ = std::current_exception();
        }
      });
      for (std::uint32_t t = 0; t < B_; ++t) host_ids_[t] = t;
    } else if (embed_rows_) {   // token_embd stays on the host: the chunk's rows staged, the embed reads row t for token t
      // (the slot's staging is rewritten once its last copy landed: the GPU a whole chunk behind the host waits here)
      std::vector<std::uint8_t>& rows = host_rows_;
      const auto e0 = std::chrono::steady_clock::now();
      LoomDevice::WaitFor(rows_copied_);
      const auto e1 = std::chrono::steady_clock::now();
      gguf_.Rows(*Find("token_embd.weight"), host_ids_.data(), B_, rows.data());
      const auto e2 = std::chrono::steady_clock::now();
      rows_copied_ = gpu_.StreamH2D(*embed_rows_, rows.data(), embed_rows_->size);   // (halves: one half of rows)
      if (std::getenv("YAH_HOST_TIMES"))
        std::fprintf(stderr, "host: embed wait %.2f ms, gather %.2f ms (%zu B), stream %.2f ms\n",
                     std::chrono::duration<double, std::milli>(e1 - e0).count(),
                     std::chrono::duration<double, std::milli>(e2 - e1).count(), rows.size(),
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e2).count());
      for (std::uint32_t t = 0; t < B_; ++t) host_ids_[t] = t;
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

  // Enqueue the 64 layers for chunk ci (absolute positions ci * B ..), on the hidden() rows from Embed().
  void RunLayers(std::uint32_t ci, const KvHook& hook = {}, std::int64_t keep = kAllRows) {
    keep_rows_ = keep;
    if (std::size_t{ci + 1} * chunk_tokens() > T_) throw LoomError("prefill: chunk past the emitted context");
    if (n_ == 0) throw LoomError("prefill: RunLayers before Embed");
    // The layers go into one graph, so kernels with no data between them (the input projections of a layer, the DeltaNet
    // gate projection and the conv / DeltaNet chain) can run at the same time.
    // NPU split: full chunks only, its handoffs graph nodes on flag words (NpuEnqueue / NpuJoin)
    npu_on_ = npu_ && n_ == B_;
    if (npu_on_) npu_->BeginChunk();
    if (npu_attn_ && ci >= na_from_) ++na_seq_;  // the layers' ready / done values (words are reused per chunk)
    flagged_.clear();
    npu_job_ = 0, npu_cleared_ = 0, npu_joined_ = 0;
    if (ci == 0) clk_n_ = 0, stamp_n_ = 0, clk_ffn_.clear();   // (YAH_CLK_PROBE: ReportClk reads the last pass's chunks)
    swap_layer_.clear(), swap_reader_[0] = swap_reader_[1] = -1, part_done_[0] = part_done_[1] = -1;   // (halves: a chunk's job indices start again; its graph runs after the last)
    job_side_.clear();
    NewGraph();
    // Layer 0 goes first in a graph of its own, so the GPU runs it while the host records the other layers. The split
    // falls before its deferred down GEMM (the first dispatch while recording layer 1), which waits for all of the
    // layer's last kernel anyway. The NPU split's flagged jobs keep one graph.
    split_layer_ = (!npu_on_ || NpuEarly()) && cfg_.main_block_count() > 1 ? 1 : 0;
    npu_enq_ = 0;
    // YAH_HOST_TIMES=1 (diag): the host's recording and launch times per chunk
    static const bool host_times = std::getenv("YAH_HOST_TIMES") != nullptr;
    const auto h0 = std::chrono::steady_clock::now();
    try {
      if (halves_)
        RecordHalves(ci, hook);
      else
        RecordLayers(ci, hook);
    } catch (...) {
      gather_.Join();
      throw;
    }
    const auto h1 = std::chrono::steady_clock::now();
    split_layer_ = 0, split_armed_ = false;
    // (with the NPU split: a thread polls the first job's ready word, which the GPU stores after its first kernels)
    double first_ready = -1, last_ready = -1, last_done = -1;
    std::thread poll;
    if (host_times && npu_on_ && !flagged_.empty()) {
      const NpuSplit::Job w = flagged_[0].words, wl = flagged_.back().words;
      poll = std::thread([this, w, wl, h1, &first_ready, &last_ready, &last_done] {
        const auto ms = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h1).count(); };
        while (std::chrono::steady_clock::now() - h1 < std::chrono::seconds(8)) {
          if (first_ready < 0 && npu_->HostLoad(w.ready) == w.gate) first_ready = ms();
          if (last_ready < 0 && npu_->HostLoad(wl.ready) == wl.gate) last_ready = ms();
          if (last_done < 0 && (npu_->HostLoad(wl.done) & 0x7fffffffu) >= wl.gate) last_done = ms();
          if (last_done >= 0 && first_ready >= 0 && last_ready >= 0) return;
        }
      });
    }
    LaunchChunk();
    const auto h2 = std::chrono::steady_clock::now();
    if (host_times) gpu_.Synchronize();
    if (poll.joinable()) poll.join();
    if (host_times)
      std::fprintf(stderr, "host: chunk %u record %.1f ms, launch + NPU enqueue %.1f ms, GPU done %.1f ms; first ready "
                   "word %.2f, last ready %.1f, last done %.1f ms (after the record)\n", ci,
                   std::chrono::duration<double, std::milli>(h1 - h0).count(),
                   std::chrono::duration<double, std::milli>(h2 - h1).count(),
                   std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h1).count(), first_ready,
                   last_ready, last_done);
  }

  // The recorded chunk's graph to the GPU, its NPU jobs to the NPU.
  void LaunchChunk() {
    graph_ = nullptr;
    FinishEmbed();
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
        std::vector<std::size_t> from;   // each kept job's index in the chunk
        std::size_t a = 0, b = 0;
        for (int n = 0; std::sscanf(r, "%zu-%zu%n", &a, &b, &n) == 2 && a <= b && b < jobs.size(); ++r) {
          kept.insert(kept.end(), jobs.begin() + a, jobs.begin() + b + 1);
          for (std::size_t i = a; i <= b; ++i) from.push_back(i);
          r += n;
          if (*r != ',') break;
        }
        jobs.swap(kept);
        // The gate counts jobs per family run (NpuSplit::NewJob; gate = sequence * 64 + calls, gen_npu_gemm.GATE_CALLS).
        // A job kept right after one of its tag that was not its neighbor in the chunk (no family switch between them
        // now) counts on from that one; every other job keeps the chunk's value.
        for (std::size_t i = 1; i < jobs.size(); ++i)
          if (jobs[i].tag == jobs[i - 1].tag && from[i] != from[i - 1] + 1)
            jobs[i].words.gate = (jobs[i - 1].words.gate / 64 + 1) * 64 + jobs[i].words.gate % 64;
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
      std::map<std::string, double> family_ms;
      NpuProf* prof = npu_->Prof();
      std::vector<double> job_ms;
      for (std::size_t i = 0; i < jobs.size(); ++i) {
        const NpuSplit::Job& w = jobs[i].words;
        std::vector<std::uint32_t> before;   // the job's gate slots, to report which words its calls changed
        for (std::uint32_t k = w.ready; k <= w.done + 16; ++k) before.push_back(npu_->HostLoad(k));
        // (YAH_NPU_CPU_HOLD=all: every job)
        if (const char* h = std::getenv("YAH_NPU_CPU_HOLD"); h && (!std::strcmp(h, "all") || std::strtoul(h, nullptr, 10) == i)) {
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
        // the job's time from its ready store to its done word (its command's setup and image switch included)
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "npu cpu-driven: job %zu %s layer %u %.3f ms\n", i, jobs[i].tag.c_str(), jobs[i].layer, ms);
        if (std::getenv("YAH_NPU_CPU_GO"))   // (diag) the job's go record: [ready value, status, polls, -]
          std::fprintf(stderr, "npu cpu-driven: go %08x %u %u %08x\n", npu_->HostLoad(w.ready + 32),
                       npu_->HostLoad(w.ready + 33), npu_->HostLoad(w.ready + 34), npu_->HostLoad(w.ready + 35));
        job_ms.push_back(ms);
        family_ms[jobs[i].tag] += ms;
        if ((d & 0x7fffffffu) < w.gate || (d & 0x80000000u)) {
          std::fprintf(stderr, "npu cpu-driven: job %zu (%s) stalled or failed: done %08x gate %08x\n", i,
                       jobs[i].tag.c_str(), d, w.gate);
          if (const char* cmd = std::getenv("YAH_NPU_CPU_STALL_CMD")) std::system(cmd);   // e.g. an AIE status dump
          if (NpuProf* const p = npu_->Prof(); p && std::getenv("YAH_NPU_PROF_MID")) {   // (diag) the log so far
            const std::vector<std::uint32_t> v = p->Peek();
            for (std::size_t r = 0; r * p->Values() < v.size(); ++r) {
              std::fprintf(stderr, "npu prof: rec %zu", r);
              for (std::uint32_t k = 0; k < p->Values(); ++k) std::fprintf(stderr, " %u", v[r * p->Values() + k]);
              std::fprintf(stderr, "\n");
            }
          }
          for (std::uint32_t k = w.ready; k <= w.done + 16; ++k)
            if (const std::uint32_t v = npu_->HostLoad(k); v != before[k - w.ready])
              std::fprintf(stderr, "  word +%u: %08x -> %08x\n", k - w.ready, before[k - w.ready], v);
          return;
        }
      }
      std::fprintf(stderr, "npu cpu-driven: %zu jobs done\n", jobs.size());
      if (prof && std::getenv("YAH_NPU_PROF_MID")) {   // (diag) every record, in log order
        prof->Finish(0);
        const std::vector<std::uint32_t> v = prof->Peek();
        for (std::size_t r = 0; r * prof->Values() < v.size(); ++r) {
          std::fprintf(stderr, "npu prof: rec %zu", r);
          for (std::uint32_t k = 0; k < prof->Values(); ++k) std::fprintf(stderr, " %u", v[r * prof->Values() + k]);
          std::fprintf(stderr, "\n");
        }
        prof = nullptr;
      }
      if (prof) {   // one line per job: its counters (NpuProf::Finish order), for tools/npu_prof.py
        const std::vector<std::uint32_t> v = prof->Finish(jobs.size());
        std::fprintf(stderr, "npu prof: events %s\n", std::getenv("YAH_NPU_PROF"));
        for (std::size_t i = 0; i < jobs.size(); ++i) {
          std::fprintf(stderr, "npu prof: job %zu %s %.3f", i, jobs[i].tag.c_str(), job_ms[i]);
          for (std::uint32_t k = 0; k < prof->Values(); ++k) std::fprintf(stderr, " %u", v[i * prof->Values() + k]);
          std::fprintf(stderr, "\n");
        }
        std::vector<std::string> tags;
        std::vector<std::uint32_t> layers;
        for (const auto& j : jobs) tags.push_back(j.tag), layers.push_back(j.layer);
        PrintNpuTimes(*prof, tags, layers);
      }
      for (const auto& [tag, ms] : family_ms) std::fprintf(stderr, "npu cpu-driven: %-8s %8.1f ms\n", tag.c_str(), ms);
      if (const char* cmd = std::getenv("YAH_NPU_CPU_DONE_CMD")) std::system(cmd);
      return;
    }
    if (const char* d = std::getenv("YAH_NPU_CDUMP"); d && npu_ && std::getenv("YAH_NPU_CDUMP_JOB")) {
      const std::size_t j = std::strtoul(std::getenv("YAH_NPU_CDUMP_JOB"), nullptr, 10);   // (diag) C after job j
      if (j < flagged_.size())
        npu_->DiagDumpC(flagged_[j].words.done, flagged_[j].words.gate, std::size_t{16} << 20, d);
    }
    static const bool fake = npu_ && std::getenv("YAH_NPU_FAKE");
    if (fake && !na_jobs_.empty()) throw LoomError("YAH_NPU_FAKE: not with NPU attention jobs");
    chunk_graph_->Launch();
    jobs.swap(flagged_);
    // YAH_NPU_FAKE=1 (diag, wrong output): no NPU work; a host thread stands in for the NPU, storing each job's done
    // word as soon as the GPU stores its ready word (in job order), so the GPU runs its exact split share with no NPU
    // memory traffic (GPU time without contention). Not with NPU attention.
    if (fake) {
      fake_npu_.Join();
      std::vector<NpuSplit::Job> words;
      for (const auto& q : jobs) words.push_back(q.words);
      fake_npu_.t = std::thread([this, words] {
        for (std::size_t i = 0; i < words.size(); ++i) {
          const auto t0 = std::chrono::steady_clock::now();
          while (npu_->HostLoad(words[i].ready) != words[i].gate)
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) {
              std::fprintf(stderr, "npu fake: job %zu's ready word never came (gate %08x)\n", i, words[i].gate);
              npu_->HostStore(words[i].done, NpuSplit::kGateFailed | words[i].gate);   // the GPU waits fail at once
              return;
            }
          npu_->HostStore(words[i].done, words[i].gate);
        }
        std::fprintf(stderr, "npu fake: %zu jobs answered\n", words.size());
      });
      return;
    }
    if (npu_ && npu_->Prof() && !jobs.empty() && prof_tags_.empty())   // the first NPU chunk's jobs (ReportNpuProf)
      for (const auto& q : jobs) prof_tags_.push_back(q.tag), prof_layers_.push_back(q.layer);
    // in recording order: each attention job after the GEMM jobs recorded before it (the two NPU contexts run in the
    // order their commands were built, LoomNpu::SetPeer)
    std::size_t from = npu_enq_;   // (NpuEarly: the jobs before it went at the split)
    npu_enq_ = 0;
    auto gemm_to = [&](std::size_t to) {
      if (to <= from) return;
      try {
        npu_->Enqueue({jobs.begin() + from, jobs.begin() + to});   // on a failure it releases the graph's waits
      } catch (...) {
        gpu_.Synchronize();   // the graph reads NPU buffers: it must end before anyone frees them
        throw;
      }
      from = to;
    };
    for (const NaJob& j : na_jobs_) {
      gemm_to(j.after);
      try {
        npu_attn_->Enqueue(j.nb, NaReady(j.ai), NaReady(j.ai) + 1, na_seq_, "attn");
      } catch (...) {
        gpu_.Synchronize();  // the graph's waits time out (the sticky status word): it reads the NPU's buffers
        throw;
      }
    }
    na_jobs_.clear();
    gemm_to(jobs.size());
  }

  // The 64 layers of chunk ci, in dispatch order.
  void RecordLayers(std::uint32_t ci, const KvHook& hook) {
    Dispatch(Exe("embed.hal"), "yah_embed", GeomOf("embed.hal").tt, 1, 1, kHidden / 16, 1, 1,
             {EmbedTable(), Ref(*ids_), Ref(*hidden_)}, 4);
    for (std::uint32_t l = 0; l < cfg_.main_block_count(); ++l) RecordLayer(l, ci, hook);
    FlushFfn();
    ClkProbe();
  }
  void RecordLayer(std::uint32_t l, std::uint32_t ci, const KvHook& hook) {
    const std::string pre = "blk." + std::to_string(l) + ".";
    split_armed_ = split_layer_ && l == split_layer_;
    layer_ = l;
    ClkProbe();
    tail_ = l + 1 == cfg_.main_block_count() ? keep_rows_ : kAllRows;
    if (cfg_.IsFullAttention(l)) {
      const bool q_af = Af("gemm_kqg", pre + "attn_q.weight");
      const bool k_af = Af("gemm_kstore", pre + "attn_k.weight");
      const bool v_af = Af("gemm_kstore", pre + "attn_v.weight");
      const bool q_npu = NpuRows("q", pre + "attn_q.weight") && NpuOk(pre + "attn_q.weight") &&
                         !NpuHal(KqgHal(pre + "attn_q.weight", q_af)).empty();
      RunLayerNorm(pre + "attn_norm.weight",
                   q_af && k_af && v_af   ? NormOut::kTiled
                   : q_af || k_af || v_af ? NormOut::kBoth
                                          : NormOut::kRow,
                   q_npu ? "q" : nullptr, q_npu && NpuPk(pre + "attn_q.weight"), q_npu && SwSiteOf(pre + "attn_q.weight"));
      RunAttention(l, ci, pre, hook, q_af, k_af, v_af);
    } else {
      // afrag qkv / gate read the fragment-major copy; alpha / beta keep the row-major one
      const bool qkv_af = Af("gemm_kstore", pre + "attn_qkv.weight");
      const bool gate_af = Af("gemm_kstore", pre + "attn_gate.weight");
      const bool qkv_npu = NpuRows("qkv", pre + "attn_qkv.weight") && NpuRows("gate", pre + "attn_gate.weight") &&
                           NpuOk(pre + "attn_qkv.weight") &&
                           NpuOk(pre + "attn_gate.weight") &&
                           !NpuHal(KstoreHal(pre + "attn_qkv.weight", qkv_af)).empty() &&
                           !NpuHal(KstoreHal(pre + "attn_gate.weight", gate_af)).empty();
      // ab_t: alpha / beta read the fragment-major copy too (gemm_kst2 .at.hal), so no row-major copy is written
      const bool ab_t = qkv_npu && qkv_af && gate_af && !AlphaBetaHal(pre, true).empty();
      RunLayerNorm(pre + "attn_norm.weight", ab_t ? NormOut::kTiled : qkv_af || gate_af ? NormOut::kBoth : NormOut::kRow,
                   qkv_npu ? "qkv" : nullptr, qkv_npu && (NpuPk(pre + "attn_qkv.weight") || NpuPk(pre + "attn_gate.weight")),
                   qkv_npu && SwSiteOf(pre + "attn_qkv.weight") && SwSiteOf(pre + "attn_gate.weight"));
      RunDeltaNet(l, pre, qkv_af, gate_af, ab_t);
    }
    if (!pending_norm_.empty()) throw LoomError("prefill: " + pre + " fused norm without its f16 output");
    if (tail_ != kNoRows) {
      trim_row_ = tail_;
      RunFfn(pre);
      trim_row_ = kAllRows;
    }
    FlushOut();
    tail_ = kAllRows;
  }



  // Final norm + output head of hidden() row `row` (this chunk) into dst (kVocab f32).
  void Head(std::uint32_t row, const hrx_buffer_ref_t& dst) {
    if (halves_) {   // (YAH_PP_HALVES) row of the 2 B chunk: its half's bank
      LoadHalf(row / B_);
      row %= B_;
    }
    FlushOut();
    const auto* onw = Find("output_norm.weight");
    const auto* ow = Find("output.weight");
    Dispatch(Exe("rmsnorm.hal"), "yah_rmsnorm", 1, 1, 1, 32, 1, 1,
             {{hidden_->handle, std::size_t{row} * kHidden * 4, std::size_t{kHidden} * 4}, TRef(*onw), Ref(*normed_)});
    if (geom_.count("gemv.hal")) {   // gen_gemv's tile-free GEMV (a head that is not Q6_K)
      const Geom& g = geom_.at("gemv.hal");   // (rows per workgroup, lanes, workgroups)
      Dispatch(Exe("gemv.hal"), "yah_gemv", g.tt, 1, 1, g.rowgrp, 1, 1, {TRef(*ow), Ref(*normed_), dst});
      return;
    }
    if (ow->type != core::GgmlType::kQ6_K)
      throw LoomError("output.weight is not Q6_K, and the set has no gemv.hal row for its head");
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
    return {ptab_, kq8buf_, ksbuf_, kmbuf_, vqbuf_, vqsbuf_, kpool_, vtpool_, conv_state_, state_};
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
      case 22: *out = {"iq2s", 256}; return true;
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
    if (!f) throw LoomError("cannot open " + dir_ + "/dispatch.txt: not a prefill set (tools/emit_prefill_pp.py)");
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
  LoomBuffer& Alloc(std::size_t bytes, std::source_location loc = std::source_location::current()) {
    keep_.push_back(gpu_.Allocate(bytes, loc));
    Zero(keep_.back());
    return keep_.back();
  }
  // The first `bytes` of host as a buffer of its own (the same handle, offset 0): for buffers that are never live at
  // once. The graph orders their users by the overlapping byte range of the one handle. Neither may rely on the other's
  // padding rows: each is fully written before it is read, or read only where it was written.
  LoomBuffer& Share(const LoomBuffer& host, std::size_t bytes) {
    if (bytes > host.size) throw LoomError("prefill: a shared buffer is larger than its host");
    hrx_buffer_retain(host.handle);
    LoomBuffer& b = keep_.emplace_back();
    b.handle = host.handle, b.size = bytes;
    return b;
  }
  // The most rows of a hand-written Q2_K GEMM in the set (yah_ffn_gemm_q2k_f32.loom, emitted only as
  // "...kstore_q2k_<m tiles>_<k blocks>..."), 0 without one. It is the only kernel that writes ostage: its accumulators,
  // [rows][tokens] f32. Its .fp files leave ostage out, so they cannot size it.
  static std::size_t HandQ2kRows(const std::string& name) {
    std::size_t most = 0;
    const std::string tag = "kstore_q2k_";
    for (std::size_t at = name.find(tag); at != std::string::npos; at = name.find(tag, at + 1))
      most = std::max<std::size_t>(most, 16 * std::strtoul(name.c_str() + at + tag.size(), nullptr, 10));
    return most;
  }
  std::size_t HandQ2kRows() const {
    std::size_t most = 0;
    for (const auto& [name, g] : geom_) most = std::max(most, HandQ2kRows(name));
    return most;
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
    if (it == exes_.end()) {
      if (const std::size_t q = HandQ2kRows(hal); q && (!ostage_ || q * B_ * 4 > ostage_->size)) throw LoomError("prefill: ostage is too small for " + hal);
      it = exes_.emplace(hal, gpu_.Load(dir_ + "/" + hal)).first;
    }
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
    if (split_armed_ && graph_ && !npu_planning_ && !warming_) SplitGraph();
    if (npu_planning_ || warming_) return;
    // YAH_DISP_NAMES=1 (diagnostic): each recorded dispatch's HAL, kernel and grid on stderr, in record order
    static const bool disp_names = std::getenv("YAH_DISP_NAMES") != nullptr;
    if (disp_names)
      for (const auto& [hal, e] : exes_)
        if (&e == &exe) { std::fprintf(stderr, "DISPNAME %s %s %u %u %u\n", hal.c_str(), name, gx, gy, gz); break; }
    const std::uint32_t ordinal = exe.OrdinalOrZero(name);
    const std::uint32_t ws = exe.WorkgroupSize(ordinal);
    const hrx_dispatch_config_t config = LoomDevice::Config(gx, gy, gz, ws ? ws : sx, sy, sz);
    if (graph_)
      graph_->Dispatch(exe, ordinal, config, b.data(), b.size(), writes, nullptr, also_writes, untracked);
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
    // a set of the GGUF's own formats (emit_prefill_pp.py writes the "native" row); the removed MQ4 sets lack it
    if (!geom_.count("native")) throw LoomError("prefill: " + dir_ + " is not a native set (an MQ4 set: re-emit it)");
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
    grid_iq2s_ = &table("grid_iq2s.bin", 2048 * 4);
    ksigns_ = &table("ksigns_iq2xxs.bin", 128);
  }

  void AllocateBuffers() {
    const std::size_t B = B_;
    // Activations are token major, [token][row], with row stride = the K extent of the GEMM that reads them.
    kv_cache_ = std::size_t{T_} * kKvRow;
    hidden_ = &Alloc(B * kHidden * 4);
    // The norms' residual and sum_out bindings: the emitted norms (gen_half_norm.gen_split) never touch them (only
    // gen(resadd) does), so one 4-byte buffer stands for both, read-only in the graph (no false waits between norms)
    normnil_ = &Alloc(4);
    // YAH_CLK_PROBE=1 (diag, off by default; mem 11-21): the GPU clock without the dispatch profiler. A 1-wave probe
    // (yah_clk_probe) at each layer start and after the last layer measures the clock over 20 us (REALTIME 100 MHz,
    // SHADER_CYCLES); with the NPU split each flag wait stamps REALTIME at entry and exit (npu_flag_wait_stamp.hal).
    // ReportClk prints the span, the waits and the GPU's work in cycles. Slots restart at each chunk 0.
    if (std::getenv("YAH_CLK_PROBE")) {
      if (!geom_.count("clk_probe.hal")) throw LoomError("YAH_CLK_PROBE: the set has no clk_probe.hal (re-emit it)");
      clk_ = &Alloc(kClkSlots * 16);
      if (geom_.count("npu_flag_wait_stamp.hal")) stamp_ = &Alloc(kWaitLogSlots * 8);
      clk_ffn_on_ = std::string(std::getenv("YAH_CLK_PROBE")) == "ffn";   // (also around each FFN gate / up GEMM)
    }
    scratch_ = &Alloc(B * kFfn * 2);
    qkv_ = &Alloc(B * kQProj * 4);
    gate_ = &Alloc(B * kInner * 4);
    alpha_ = &Alloc(B * kTs * 4);
    beta_ = &Alloc(B * kTs * 4);
    q_ = &Alloc(B * kAttn * 4);
    // q16 sets: RoPE writes the attention's f16(q / 16) here instead of the f32 query in place
    q16_ = geom_.count("q16") ? &Alloc(B * kAttn * 2) : q_;
    // kv4 sets at depth (dispatch.txt "qrot <first chunk>"): yah_qrot writes the H256-rotated f16 Q here for the attention
    if (const auto qr = geom_.find("qrot"); qr != geom_.end()) qr16_ = &Alloc(B * kAttn * 2), qrot_first_ = qr->second.tokens;
    // queued persistent attention (dispatch.txt "persist_<hal> <slots> 1 1"): a 1056-byte unit queue per attention layer
    // (gen_attn_fa PQ_WORDS), zero between calls (the kernel's last fetch resets its counter)
    for (const auto& [name, g] : geom_)
      if (name.rfind("persist_", 0) == 0 && g.tt && !pqueue_) pqueue_ = &Alloc(kPqueueSlots * 2048), gpu_.Fill(*pqueue_, 0);
    // two-kernel key-split tail (dispatch.txt "ksplit_<hal> <partial floats> 0 0"): one partials buffer for every
    // attention call (each call's merge runs right after it), 64 i32 counters (unused) then the f32 partials
    for (const auto& [name, g] : geom_)
      if (name.rfind("ksplit_", 0) == 0) ksplit_bytes_ = std::max<std::size_t>(ksplit_bytes_, 256 + std::size_t{4} * g.tokens);
    if (ksplit_bytes_) ksplit_ = &Alloc(ksplit_bytes_), gpu_.Fill(*ksplit_, 0);
    kbuf_ = &Alloc(B * kKv * 4);
    vbuf_ = &Alloc(B * kKv * 4);
    // FFN / DeltaNet pairs in shared memory (Share): layer l's ffn_down writes hidden, which layer l + 1's norm, qkv
    // GEMM, conv and DeltaNet wait on, so each reuse is ordered (the DeltaNet's raw_ binding is untracked).
    LoomBuffer* up_host = &Alloc(std::max(B * kInner * 4, B * kFfn * 2));
    LoomBuffer* gate_host = &Alloc(std::max(B * kQkv * 4, B * kFfn * 4));
    const auto part = [&](LoomBuffer* host, std::size_t bytes) -> LoomBuffer& { return Share(*host, bytes); };
    raw_ = &part(up_host, B * kInner * 4);
    // DeltaNet as two head ranges (dispatch.txt "dnsplit": heads [0, a) and [a, a + b)); the second writes raw2_
    if (const auto ds = geom_.find("dnsplit"); ds != geom_.end()) {
      dnsplit_a_ = ds->second.rowgrp, dnsplit_b_ = ds->second.tt;
      if (dnsplit_a_ + dnsplit_b_ != kTs || (dnsplit_a_ * B) % 8 || (dnsplit_b_ * B) % 8)
        throw LoomError("dispatch.txt dnsplit does not cover the DeltaNet heads");
      // (YAH_DNC_GROUP) the first part as dispatches of <= g heads ("dnc_a_p<i>"): one workgroup a WGP
      if (const auto dg = geom_.find("dncgroup"); dg != geom_.end()) dnc_group_ = dg->second.tokens;
      raw2_ = &Alloc(B * kInner * 4);
    }
    conv_out_ = &part(gate_host, B * kQkv * 4);
    kqbuf_ = &Alloc(B * kKh * 3 * 4);
    ab_ = &Alloc(B * kTs * 2 * 4);
    conv_state_ = &Alloc(std::size_t{48} * kQkv * 4 * 4);
    state_ = &Alloc(std::size_t{48} * kTs * kState * kState * 4);
    // f16 K/V cache: [K | V] x attention layers x context. With "kv16_scratch" (paged or quantized KV) it holds one
    // layer's current chunk only: RoPE writes it and the paged writers or quantizers read it before the next layer.
    kv16_scratch_ = geom_.count("kv16_scratch") != 0;
    kv16_layer_ = kv16_scratch_ ? B * kKvRow * 2 : kv_cache_ * 2;
    kv16_ = &Alloc(kv16_scratch_ ? 2 * kv16_layer_ : std::size_t{2} * full_ * kv_cache_ * 2);
    // the RoPE kernel's f32 K / V cache copies: bound, never stored (emit_prefill_pp.rope_live), so no context-sized pair
    kc32_ = &Alloc(4);
    vc32_ = &Alloc(4);
    lse_ = &Alloc(B * kHeads * 4);
    eps_ = &Alloc(4);
    ffnup_ = &part(up_host, B * kFfn * 2);
    // the FFN norm's fragment-major copy for the afrag GEMMs (norm_t.hal), only in sets that have them
    bool any_af = false;
    for (const auto& [name, g] : geom_) any_af = any_af || name.size() > 7 && name.compare(name.size() - 7, 7, ".af.hal") == 0;
    if (any_af) normt_ = &Alloc(B * kHidden * 2);
    gateffn_ = &part(gate_host, B * kFfn * 4);
    // Per-workgroup weight staging (ABI only: the tile GEMMs stage in LDS)
    uwstage_ = &Alloc(std::size_t{kFfn} * 16 * 2);
    wstage_ = &Alloc(std::size_t{kFfn} * 16 * 2);
    // ostage: the hand-written Q2_K GEMM's [rows][tokens] f32 accumulators (HandQ2kRows); the tile GEMMs bind it unused
    ostage_ = &Alloc(std::max<std::size_t>(HandQ2kRows() * B * 4, 4));
    partial_ = &Alloc(B * kHidden * 4);
    hidden2_ = &Alloc(B * kHidden * 4);
    normed_ = &Alloc(std::size_t{kHidden} * 4);
    valid_ = &Alloc(4);
    ids_ = &Alloc(B * 4);
    host_ids_.resize(B);
    if (const core::TensorInfo& e = *Find("token_embd.weight"); !weights_.Has(e)) {
      host_rows_.resize(std::size_t{B} * (e.bytes / e.dims[1]));
      embed_rows_ = &Alloc(host_rows_.size());
    }
    const float epsv = 1.0e-6f;
    gpu_.H2D(*eps_, &epsv, 4);
    Zero(*conv_state_);
    Zero(*state_);
    Zero(*kv16_);

    // "kv_paged": 256-token pages, one page table per sequence (logical -> physical page, shared by all layers).
    kv_paged_ = geom_.count("kv_paged") != 0;
    pages_ = (T_ + 255) / 256;
    const std::uint32_t ptab_n = pages_;
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
    const std::size_t qrows = T_;   // rows per layer: the context
    ks_bytes_ = qrows * (attn_kq4_ ? 32 : 8) * 4;
    kq_bytes_ = qrows * (attn_kq4_ ? 512 : 1024);
    kq8buf_ = &Alloc(attn_kq8_ ? std::size_t{full_} * kq_bytes_ : 4);
    ksbuf_ = &Alloc(attn_kq8_ ? std::size_t{full_} * ks_bytes_ : 4);
    kmbuf_ = &Alloc(attn_kq8_ ? std::size_t{full_} * 4096 : 4);
    // Quantized V^T: "attn_vq8" bytes / "attn_vq4" nibbles per channel per 16-key tile, plus (S, C') per tile.
    attn_vq4_ = geom_.count("attn_vq4") != 0;
    attn_vq8_ = geom_.count("attn_vq8") != 0;
    attn_vqt_ = attn_vq4_ || attn_vq8_;
    const std::size_t vrows = (T_ + 15) / 16 * 16;
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
    gsig_ = geom_.count("gsig") != 0;
    qsplit_ = geom_.count("qsplit") != 0;
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
      deq_bytes_ = std::size_t{pages_} * 256 * kKvRow * 2;
      kdeq_ = &Alloc(deq_bytes_);
      vdeq_ = &Alloc(deq_bytes_);
      // dispatch.txt "kraw": kdeq8 writes the raw codes and their scales, f16 [KV head][128-dim half][pool row]
      if (geom_.count("kraw") != 0) kscale_ = &Alloc(std::size_t{pages_} * 256 * 8 * 2);
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
    // embed.hal decodes Q8_0 (core/gguf_convert.cpp makes a GGUF's token_embd Q8_0)
    if (Find("token_embd.weight")->type != core::GgmlType::kQ8_0)
      throw LoomError("prefill: token_embd: its format does not match embed.hal (re-emit the set)");
  }

  // Load every HAL chunk 0 dispatches now (a pass of the layers that dispatches nothing): a first load copies the code to
  // the GPU, ~0.25 ms per HAL, which RunLayers would otherwise spend before the GPU starts.
  void WarmExecutables() {
    LoomBuffer *h = hidden_, *h2 = hidden2_;
    n_ = B_, warming_ = true;
    RecordLayers(0, {});
    n_ = 0, warming_ = false;
    hidden_ = h, hidden2_ = h2;
  }

  // Bindings shared by every GEMM: weights, then the IQ grid / sign tables the format needs.
  std::vector<hrx_buffer_ref_t> GemmWeights(const core::TensorInfo& t, const Fmt& f) const {
    std::vector<hrx_buffer_ref_t> b = {TRef(t)};
    const std::string n = f.name;
    if (n == "iq3s") b.push_back(Ref(*grid_iq3s_));
    if (n == "iq3xxs") b.push_back(Ref(*grid_iq3xxs_));
    if (n == "iq2xxs") b.push_back(Ref(*grid_iq2xxs_));
    if (n == "iq2xs") b.push_back(Ref(*grid_iq2xs_));
    if (n == "iq2s") b.push_back(Ref(*grid_iq2s_));
    // The IQ3_XXS and IQ2 sign tables are the same 128 bytes.
    if (n == "iq3xxs" || n == "iq2xxs" || n == "iq2xs") b.push_back(Ref(*ksigns_));
    return b;
  }
  // HAL name "<kind>_<fmt>_<m_tiles>_<k_blocks>.hal" of the GEMM on tensor t.
  std::string GemmHal(const char* kind, const core::TensorInfo& t, Fmt* f) const {
    if (!FmtOf(static_cast<std::uint32_t>(t.type), f)) throw LoomError(std::string("no GEMM format for ") + kind);
    const std::uint32_t mt = static_cast<std::uint32_t>(t.dims[1] / 16);
    const std::uint32_t kb = static_cast<std::uint32_t>(t.dims[0] / f->qk);
    std::string kseg;   // (a down in another F's K segments than its format's, DownKSegs: "_kseg<widths>")
    const auto it = geom_.find("npudnseg_" + std::string(f->name));
    if (const KSeg* ks = weights_.KSegOf(t); ks && it != geom_.end() && *ks != KSegFor(it->second.tokens, kb)) {
      kseg = "_kseg";
      for (std::size_t i = 0; i < ks->size(); ++i) kseg += (i ? "_" : "") + std::to_string((*ks)[i].second);
    }
    return std::string(kind) + "_" + f->name + "_" + std::to_string(mt) + "_" + std::to_string(kb) + kseg + ".hal";
  }
  std::uint32_t MTiles(const core::TensorInfo& t) const { return static_cast<std::uint32_t>(t.dims[1] / 16); }
  // (LoomWeights RSegOf: a row-segmented gate / up) the GEMM of that layout: "_rseg<F>" before ".hal" (emit_prefill_pp
  // rseg_gemms), its weight binding from the gate tensor's start over gate and up
  std::string RSegHal(std::string hal, const core::TensorInfo& t) const {
    if (const std::uint32_t F = weights_.RSegOf(t)) hal.insert(hal.size() - 4, "_rseg" + std::to_string(F));
    return hal;
  }
  hrx_buffer_ref_t RSegRef(const core::TensorInfo& t) const {   // (t: the gate or up tensor)
    const std::string n = t.name.ends_with("ffn_up.weight") ? t.name.substr(0, t.name.size() - 13) + "ffn_gate.weight"
                                                             : t.name;
    hrx_buffer_ref_t r = TRef(*Find(n));
    r.length = 2 * static_cast<std::size_t>(t.bytes);
    return r;
  }
  // The GEMM variant for this chunk's real tokens: the set may carry "<hal>.t128.hal" / ".t64.hal" with narrower token
  // tiles. Pick the least (padded token rows x cost per row); a narrow tile pads less but costs more per row (measured:
  // x1.12 at 128 tokens, x1.6 at 64). Every variant computes the same values.
  // A tuned set (engine/tune) carries the measured choice instead: "pick:<hal>:<max tokens>" rows map a token bucket to
  // the token tile to use; chunks above the largest bucket take the full tile.
  std::string PickGemm(const std::string& hal) {
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
      const std::size_t n = trim_row_ >= 0 && geom_.count("tailnarrow") ? 1 : n_;   // (a trimmed tail keeps one row)
      const double cost = static_cast<double>((n + bn - 1) / bn * bn) * weight;
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
    if (trim_row_ >= 0 && geom_.count("tailnarrow")) return false;   // (the trimmed tail's plain 64-token tiles)
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
    if (RunFfnFused(pre)) return;
    const bool gate_af = Af("gemm_kstore", pre + "ffn_gate.weight");
    const bool up_af = Af("gemm_swiglu", pre + "ffn_up.weight");
    const bool down_af = up_af && Af("gemm_kres", pre + "ffn_down.weight");
    RunNorm(pre + "post_attention_norm.weight",
            gate_af && up_af ? NormOut::kTiled : gate_af || up_af ? NormOut::kBoth : NormOut::kRow);
    RunKstore(pre + "ffn_gate.weight", *gateffn_, gate_af);
    RunSwiglu(pre + "ffn_up.weight", up_af, down_af);
    RunResidual(pre + "ffn_down.weight", *ffnup_, down_af);
  }

  // The NPU's FFN block (dispatch.txt "npuffnblk <F>", emit_prefill_pp NPU_FFNBLK): the NPU owns the FFN's last F
  // features. Its gate / up calls (npu_ffnsw_<fmt>_<hord>.xdna, 256 features a call: call j's column c features
  // 256 j + 32 c + [0, 32); gate rows and the up rows one gate tensor further, straight from the model) write
  // H = bfp16(silu(gate) * up) into C, [M block][pass][16-token slab][128 k-blocks]; its down calls
  // (npu_ffndn<passes>_<fmt>.xdna, per K part of F (FfnBlkParts), FfnBlkRows() rows each, a last call overlapping the
  // one before) read H and write their partials after it. Each is a job of one image family. The GPU runs the other features (the fused ffn's split) and down over them
  // (npuffnrem_<fmt>.hal, f32), then npu_unpack_ffnblk.hal writes hidden + that + the partials. Returns false if the
  // layer does not qualify.
  // (row 5 stepping stone) on a set with dispatch.txt "npurswap" (emit_prefill_pp RSWAP_EMIT, default on): the FFN
  // block layers' gate / up run the swapped image (RunFfnBlock) when RSwapFmt takes both formats; YAH_NPU_RSWAP=0: off
  bool RSwapOn() const {
    static const bool on = [] { const char* e = std::getenv("YAH_NPU_RSWAP"); return !e || std::strcmp(e, "0") != 0; }();
    return on && geom_.count("npurswap") != 0;
  }
  // A gate / up format the swap decodes: the set has its decoders npu_swdec_<fmt>_g / _u.hal, and YAH_NPU_RSWAP_FMTS
  // (a comma list, unset: all) names it.
  bool RSwapFmt(const char* fmt) const {
    static const std::string only = [] {
      const char* e = std::getenv("YAH_NPU_RSWAP_FMTS");
      return e ? "," + std::string(e) + "," : std::string();
    }();
    const std::string n(fmt);
    return geom_.count("npu_swdec_" + n + "_g.hal") && geom_.count("npu_swdec_" + n + "_u.hal") &&
           (only.empty() || only.find("," + n + ",") != std::string::npos);
  }
  // (row 5) the swap layers' GPU gate / up split decodes the next swap layer's NPU gate / up rows into the other half
  // of a double-buffered stream: dispatch.txt "<split>.af.to.r5.npu.hal" (emit_prefill_pp RSWAP_EMIT, gen_r5fuse; its
  // two buffers after the GEMM's: the next layer's gate + up tensors, the stream half); a chunk's first swap layer, and
  // one after a layer that could not carry it, decode with npu_swdec_<fmt>_g / _u
  // (gen_r5kq payloads: "<split>.af.to.r5_<gate>_<up>.npu.hal" decodes a next layer of other formats; their grid
  // tables bind between the raw weights and the stream, gate's first, each format once)
  // (row 5 swap) the FFN norm writes the token panel: the set has its "_pn" forms; YAH_NPU_RSWAP_PANEL=0: off
  // YAH_NPU_PANELQ=1 with a panel swap image (dispatch.txt "npupanel", npu's YAH_NPU_PANEL emit): a carried swap
  // layer's FFN norm runs in token quarters (YAH_EMIT_PANELQ's "<pn>_q<q>.hal") and opens the swap job's panel q
  // after quarter q (gate - 48 + 16 q, NpuSplit::NewJob's panel form; the job's own final store opens the last).
  // The out site's C then sits after the swap's H (FfnBlkDown): panel 0's H writes must not reach the out C that
  // quarters 1-3 still read.
  bool PanelQ() const {
    static const bool on = [] { const char* e = std::getenv("YAH_NPU_PANELQ"); return e && std::strcmp(e, "1") == 0; }();
    return on && geom_.count("npupanel") != 0;
  }
  // (diag) YAH_NPU_OUTC_AFTER_H=1: only PanelQ's C placement (the out site's C after the FFN block's H)
  bool OutCAfterH() const {
    static const bool on = std::getenv("YAH_NPU_OUTC_AFTER_H") != nullptr;
    return (on || PanelQ()) && FfnBlkF();
  }
  bool PanelNorm() const {
    static const bool off = [] { const char* e = std::getenv("YAH_NPU_RSWAP_PANEL"); return e && std::strcmp(e, "0") == 0; }();
    if (off || !RSwapOn()) return false;
    const std::string oh = OnormHal();
    return ::access((dir_ + "/norm_t_bfp_g1_pn.hal").c_str(), R_OK) == 0 &&
           (!geom_.count(oh) || geom_.count(oh.substr(0, oh.size() - 4) + "_pn.hal"));
  }
  bool R5Fuse() const {
    if (!RSwapOn()) return false;
    static const std::string sfx = ".af.to.r5.npu.hal";
    for (const auto& kv : geom_)
      if (kv.first.size() > sfx.size() && kv.first.compare(kv.first.size() - sfx.size(), sfx.size(), sfx) == 0) return true;
    return R5Part();
  }
  // (YAH_PP_HALVES=2) tile's part carriers (emit YAH_NPU_R5PART, gen_r5fuse part=): "<carrier>.af.to.r5..._p<h>.npu..."
  // decodes half the next layer's stream (part 0: gate rows, part 1: up rows); half h's GEMM runs part h, so the two
  // halves' GEMMs of layer k together write layer k + 1's stream. Without HALVES=2 the set's layers decode standalone.
  bool R5Part() const {
    if (!RSwapOn()) return false;
    for (const auto& kv : geom_)
      if (kv.first.find(".af.to.r5") != std::string::npos && kv.first.find("_p0.npu") != std::string::npos) return true;
    return false;
  }
  // the swap image's weight M blocks (64 stream rows of 5120 x 9 / 8 B) a call: 32 = 1024 features (emit_prefill_pp
  // YAH_NPU_RSWAP_NB); a token panel's calls each take SwapNb() / 32 of today's 1024-feature blocks
  // (share-up, F of 256-feature units: dispatch.txt "npurswap_<F> <F>", its image npu_ffnswap_iq4xs_<F>.xdna, bytes
  // "npubytes_ffnswap_<F>", decoders npu_swdec_<fmt>_<g>_<F>.hal; the set's own F, "npurswap", unsuffixed). F 0: that.
  std::uint32_t SwapF0() const { return geom_.at("npurswap").tokens; }
  std::string SwapTag(std::uint32_t F) const { return !F || F == SwapF0() ? "" : "_" + std::to_string(F); }
  bool RSwapF(std::uint32_t F) const { return F == SwapF0() || geom_.count("npurswap_" + std::to_string(F)) != 0; }
  // (A whole blocks; the share-up's NB = F / 32 need not be a multiple of 32: one call a token panel)
  std::uint32_t SwapNb(std::uint32_t F = 0) const {
    const std::size_t a = NpuBytesOf("npubytes_ffnswap" + SwapTag(F)).a, blk = std::size_t{64} * 5120 * 9 / 8;
    F = F ? F : SwapF0();
    if (a % blk || (a / blk % 32 && a / blk != F / 32) || F / 32 % (a / blk))
      throw LoomError("prefill: npubytes_ffnswap" + SwapTag(F) + "'s A is not whole blocks");
    return static_cast<std::uint32_t>(a / blk);
  }
  std::size_t SwapStream(std::uint32_t F = 0) const {   // one layer's swap weight stream: F / 32 blocks
    const std::uint32_t f = F ? F : SwapF0();
    return std::size_t{f} / 32 / SwapNb(F) * NpuBytesOf("npubytes_ffnswap" + SwapTag(F)).a;
  }
  std::size_t SwapStreamMax() const {   // (A's swap halves: the largest F's stream)
    std::size_t m = SwapStream();
    for (const auto& kv : geom_)
      if (kv.first.starts_with("npurswap_")) m = std::max(m, SwapStream(kv.second.tokens));
    return m;
  }
  // (emit_prefill_pp YAH_NPU_SWDOWN, dispatch.txt "npuswdown") the FFN block's down in swap form: ffnswap writes H as
  // its token panels (GN_QHPAN), so every swap layer's down runs it (npu_swdown, npu_swddec_<fmt>)
  bool SwDown() const { return geom_.count("npuswdown") != 0; }
  std::size_t SwapA(std::uint32_t half = 0) const {   // the swap's weight stream in A: its halves at A's end
    const std::size_t r = SwapStreamMax();
    return npu_->A().size - (R5Fuse() ? 2 : 1) * r + half * r;
  }
  // (share-up) layer pre's NPU FFN features: dispatch.txt "npusplit_ffn@<L> <F'>", else base
  std::uint32_t LayerFfnF(const std::string& pre, std::uint32_t base) const {
    const auto it = geom_.find("npusplit_ffn@" + pre.substr(4, pre.find('.', 4) - 4));
    return it == geom_.end() ? base : it->second.tokens;
  }
  // the pk (Q4_K order) stream's start in A (npu_plan dcol: A holds the p4 stream, then the pk one): the middle of A
  // before the swap's streams at its end (A's size / 2 put it inside the swap's stream since RSwapOn; harmless while
  // each stream was written right before its job, not once the double buffer keeps one across layers)
  std::size_t PkA() const {
    std::size_t s = halves_ ? a_one_ : npu_->A().size;   // (halves: one half's A, its p4 stream then the pk one)
    if (RSwapOn() && !halves_) s -= (R5Fuse() ? 2 : 1) * SwapStreamMax();
    return s / 2;
  }
  struct R5Layer {
    std::string pre;
    std::string carrier;   // its GPU split's name before ".af.to.npu[_n<F>].hal" ("": it cannot carry)
    std::string pay;       // its gate / up as a payload: "" (IQ4_XS), "_<gate>_<up>"; payok: they fit gen_r5fuse's map
    bool payok;            // (17408 x 5120, up after gate, a swap F, no rseg)
    std::uint32_t f = 0;   // (share-up) its F: the carrier's rows "_n<F>", the payload's "_f<F>" (the swap's F: none)
  };
  // (R5Fuse) layer k's split carries layer k + 1's decode: the set has that fused split
  std::string R5Hal(std::size_t k) const {
    if (k + 1 >= r5_.size() || r5_[k].carrier.empty() || !r5_[k + 1].payok) return "";
    static const bool part = R5Part();
    if (part && halves_ != 2) return "";
    const std::string h = r5_[k].carrier + ".af.to.r5" + r5_[k + 1].pay +
                          (SwapTag(r5_[k + 1].f).empty() ? "" : "_f" + SwapTag(r5_[k + 1].f).substr(1)) +
                          (part ? "_p" + std::to_string(half_) : std::string()) + ".npu" +
                          (SwapTag(r5_[k].f).empty() ? "" : "_n" + SwapTag(r5_[k].f).substr(1)) + ".hal";
    return geom_.count(h) ? h : "";
  }
  std::vector<R5Layer> r5_;   // (R5Fuse) the swap layers in order, recorded by the planning pass
  std::uint32_t FfnBlkF() const {
    const auto it = geom_.find("npuffnblk");
    return it == geom_.end() ? 0 : it->second.tokens;
  }
  // dispatch.txt "npuffnblk <F> <gate / up features a call: 256> <down rows a call>"
  // (np4: the NP 4 formats' downs, dispatch.txt "npuffnblk4", "npuffndn4_<fmt>", "npubytes4_ffndn<passes>" and
  // npu_unpack_ffnblk_4.hal)
  std::uint32_t FfnBlkRows(bool np4 = false) const { return geom_.at(np4 ? "npuffnblk4" : "npuffnblk").tt; }
  std::uint32_t FfnBlkCalls(bool np4 = false) const {   // down calls per K part
    return (kHidden + FfnBlkRows(np4) - 1) / FfnBlkRows(np4);
  }
  bool FfnDn4(const char* fmt) const { return geom_.count(std::string("npuffndn4_") + fmt) != 0; }
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
  // A down K part over F (emit_prefill_pp ffnblk_units): its first unit (256 features, a super-block) within F, its
  // units, and the units its image reads (padded: a part of 15 reads 16, the 16th its segment's next row's first block
  // times H's zero pad unit, npu_zpad_<F>.hal).
  struct FfnPart {
    std::uint32_t u0, r, p;
  };
  // F of whole passes: FfnBlkParts. Else (F of whole 1280-feature groups, the NP 5 gate / up): the fewest parts of at
  // most 4 groups, the shorter ones first.
  static std::vector<FfnPart> FfnBlkUnits(std::uint32_t F) {
    std::vector<FfnPart> out;
    if (F % 1024 == 0) {
      for (const auto& [p0, n] : FfnBlkParts(F)) out.push_back({4 * p0, 4 * n, 4 * n});
      return out;
    }
    if (F % 256 == 0 && F % 1280) {   // (share-up F', whole units: ceil(U / 4) passes in the fewest parts of <= 5,
      // shorter first, the first padded: its over-read lands on the next K segment's rows, as FFNSW5's)
      const std::uint32_t U = F / 256, P = (U + 3) / 4, n = (P + 4) / 5;
      for (std::uint32_t i = 0, u0 = 0; i < n; ++i) {
        const std::uint32_t p = 4 * (P / n + (i >= n - P % n)), r = p - (i ? 0 : 4 * P - U);
        out.push_back({u0, r, p});
        u0 += r;
      }
      return out;
    }
    if (F % 1280) throw LoomError("prefill: FFN block of " + std::to_string(F) + " features: not whole units");
    const std::uint32_t G = F / 1280, n = (G + 3) / 4;
    for (std::uint32_t i = 0, u0 = 0; i < n; ++i) {
      const std::uint32_t r = 5 * (G / n + (i >= n - G % n));
      out.push_back({u0, r, (r + 3) / 4 * 4});
      u0 += r;
    }
    return out;
  }
  // H's units a 64-token M block when F's parts differ in padded size (H one [M block][units] layout, a part's A a
  // window of it), else 0 (H in parts, [part][M block][its units]).
  // ("npuhapass", emit_prefill_pp hapass: a swap call's H spans both K parts) H [M block][units] for equal parts too
  std::uint32_t FfnBlkApass(std::uint32_t F) const {
    const auto us = FfnBlkUnits(F);
    std::uint32_t sum = 0;
    bool same = true;
    for (const FfnPart& u : us) sum += u.p, same = same && u.p == us[0].p;
    return same && !geom_.count("npuhapass") ? 0 : sum;
  }
  // A down part image's tag (emit_prefill_pp dn_tag): "npu_ffndn<tag>_<fmt>", "npubytes_ffndn<tag>"
  static std::string DnTag(const FfnPart& u, std::uint32_t apass) {
    return u.r == u.p && !apass ? std::to_string(u.p / 4)
                                : "u" + std::to_string(u.r) + "_" + std::to_string(u.p) + "_" + std::to_string(apass);
  }
  // (share-up F', whole units only, emit_prefill_pp ffnblk_front) the padded first part's pads at its front: H = [pads]
  // [real units] (one run, as the swap's one call a panel writes it); its rows read from p - r blocks before its K
  // segment's row (DNSEG: the previous row's, or the GPU segment's last row's, last blocks), times those zero pads
  static bool FfnBlkFront(std::uint32_t F) { return F % 256 == 0 && F % 1024 && F % 1280; }
  static std::size_t FfnBlkHOf(std::uint32_t B, std::uint32_t F) {
    std::size_t units = 0;
    for (const FfnPart& u : FfnBlkUnits(F)) units += u.p;
    return std::size_t{B / 64} * units * 18432;
  }
  // H's bytes in C (from 0), then down's partials from FfnBlkDown().
  std::size_t FfnBlkH() const {   // (share-up: the largest layer F's)
    std::size_t h = FfnBlkHOf(B_, FfnBlkF());
    for (const auto& kv : geom_)
      if (kv.first.starts_with("npusplit_ffn@")) h = std::max(h, FfnBlkHOf(B_, kv.second.tokens));
    return h;
  }
  std::size_t FfnBlkDown() const { return (FfnBlkH() + 4095) / 4096 * 4096; }
  static int DecoderId(const char* fmt) {
    static const char* const kDecoders[] = {"iq4xs", "iq3s",  "iq3xxs", "q4k",  "q3k",
                                            "iq2xxs", "iq2xs", "q5k",    "iq2s", "bfp",  "iq4nl", "q2k"};
    for (int i = 0; i < 12; ++i)
      if (std::strcmp(kDecoders[i], fmt) == 0) return i;
    return -1;
  }
  bool RunFfnBlock(const std::string& pre) {
    if (!ffnblk_ || !npu_on_ || trim_row_ != kAllRows || !AfChunk()) return false;
    const auto* tg = Find(pre + "ffn_gate.weight");
    const auto* tu = Find(pre + "ffn_up.weight");
    const auto* td = Find(pre + "ffn_down.weight");
    Fmt f{}, fu{}, fd{};
    if (!FmtOf(static_cast<std::uint32_t>(tg->type), &f) || !FmtOf(static_cast<std::uint32_t>(tu->type), &fu) ||
        !FmtOf(static_cast<std::uint32_t>(td->type), &fd))
      return false;
    // gate and up of one format with a fused GPU gate / up: one image (its columns compute both); else a pair: the GPU's
    // side two splits, the NPU's one mixed (or BFP) image
    // (dispatch.txt "npuffncp_<gate>_<up>" of one format, emit_prefill_pp QCP: a Q' column-pair image, so the pair path)
    const bool cprow = geom_.count("npuffncp_" + std::string(f.name) + "_" + fu.name) != 0;
    const std::string hal = tg->type == tu->type && !cprow ? AfHal(GemmHal("gemm_ffn", *tg, &f), ".af.to.hal") : "";
    const bool pair = hal.empty();
    // (bmix: a pair whose NPU rows the emitter pre-decoded, emit_prefill_pp NPU_BFPSW: one BFP image like a single
    // format's, "npu_ffnsw_bfp<hord>", on npu_bfp_<pre>ffn_gate_up.bin)
    const bool bmix = pair && geom_.count("npuffnbx_" + std::string(f.name) + "_" + fu.name) != 0;
    if (tg->dims != tu->dims || (!pair && tu->offset != tg->offset + tg->bytes) || (pair && !geom_.count("npuffnpair")))
      return false;
    // (share-up: the layer's own F, "npusplit_ffn@<L>"; its GPU splits "<hal>.npu_n<F>.hal")
    const std::uint32_t F0 = pair ? geom_.at("npuffnpair").tokens : ffnblk_, F = LayerFfnF(pre, F0),
                        M = static_cast<std::uint32_t>(tg->dims[1]);
    const auto npuhal = [&](const std::string& h) {
      if (F == F0 || h.empty()) return NpuHal(h);
      const std::string v = h.substr(0, h.size() - 4) + ".npu_n" + std::to_string(F) + ".hal";
      return geom_.count(v) ? v : std::string();
    };
    // the GPU's features: the fused ffn's split, or (pairs) gate kstore + up swiglu splits, fragment-major into ffnup_
    // (rseg: the split of that layout, its up rows F rows on, emit_prefill_pp FFUP)
    const std::string sh = pair ? "" : npuhal(weights_.RSegOf(*tg) ? AfHal(RSegHal(GemmHal("gemm_ffn", *tg, &f), *tg), ".af.to.hal") : hal);
    // (a pair whose gate and up rows are contiguous: the fused GPU gate / up's split when the set has it, emit_prefill_pp
    // GN_FFNMIXSPLIT, "gemm_ffn_<gate>[_<up>]_<mt>_<kb>.af.to.npu.hal"; YAH_FFNMIXSPLIT=0: gate kstore + up swiglu)
    static const bool mixsplit = [] { const char* e = std::getenv("YAH_FFNMIXSPLIT"); return !e || std::strcmp(e, "0"); }();
    std::string fh;
    if (pair && mixsplit && tu->offset == tg->offset + tg->bytes) {
      Fmt fx{};
      std::string b = GemmHal("gemm_ffn", *tg, &fx);
      if (tg->type == tu->type) fh = npuhal(AfHal(b, ".af.to.hal"));
      if (fh.empty()) {
        b.insert(std::strlen("gemm_ffn_") + std::strlen(f.name), "_" + std::string(fu.name));
        fh = npuhal(AfHal(b, ".af.to.hal"));
      }
    }
    const std::string gs = pair ? npuhal(KstoreHal(pre + "ffn_gate.weight", true)) : "";
    const std::string us = pair ? npuhal(SwigluHal(pre + "ffn_up.weight", true, true)) : "";
    const std::uint32_t kbw = static_cast<std::uint32_t>(td->dims[0] - F) / 256;
    // the GPU's down over its features: f32 for npu_unpack_ffnblk
    const std::string rem = "npuffnrem_" + std::string(fd.name) + "_" + std::to_string(kbw) + ".hal";
    // gate / up write H in the down decoder's k-block order (emit_prefill_pp FFNBLK_HORD: P4 but Q4_K's; Q', dispatch.txt
    // "npuffnq": P4, the Q4_K down decodes in P4 order)
    const std::string hord = std::strcmp(fd.name, "q4k") == 0 && !geom_.count("npuffnq") ? "_q4k" : "_p4";
    // (mix: a pair whose formats share a decoder, emit_prefill_pp mixed: the GPU's side a pair's, the NPU's one image like
    // a single format's, "npu_ffnsw_<gate>_<up>")
    const std::string mixed = dir_ + "/npu_ffnsw_" + f.name + "_" + fu.name + hord;
    const bool mix = bmix || (pair && ::access((mixed + ".xdna").c_str(), R_OK) == 0);
    if (pair && !mix) return false;   // (a pair needs its mixed image)
    const std::string sw = bmix ? dir_ + "/npu_ffnsw_bfp" + hord : mix ? mixed : dir_ + "/npu_ffnsw_" + f.name + hord;
    const auto parts = FfnBlkUnits(F);
    const std::uint32_t apass = FfnBlkApass(F);
    const bool hap = apass && geom_.count("npuhapass");   // (equal parts, H [M block][units]: the swap's nb 256)
    const bool rsw0 = RSwapOn() && (!pair || mix) && (pair || !geom_.count("npuffnsw5")) && geom_.count("npuffnq") &&
                      !bmix && RSwapFmt(f.name) && RSwapFmt(fu.name) && hord == "_p4" && (apass == 0 || hap) &&
                      RSwapF(F) && B_ % 512 == 0 && parts[0].p % 4 == 0 &&
                      (SwapTag(F).empty() || (geom_.count("npu_swdec_" + std::string(f.name) + "_g" + SwapTag(F) + ".hal") &&
                                              geom_.count("npu_swdec_" + std::string(fu.name) + "_u" + SwapTag(F) + ".hal")));
    std::vector<std::string> dn;
    for (const FfnPart& u : parts) {
      dn.push_back(dir_ + "/npu_ffndn" + DnTag(u, apass) + "_" + fd.name);
      // (a swap-down layer binds none: emit_prefill_pp does not emit them under YAH_NPU_SWDOWN)
      if (!(rsw0 && SwDown()) && ::access((dn.back() + ".xdna").c_str(), R_OK)) return false;
    }
    if ((pair ? gs.empty() || us.empty() || !Af("gemm_kstore", pre + "ffn_gate.weight") ||
                    !Af("gemm_swiglu", pre + "ffn_up.weight")
              : sh.empty()) ||
        !geom_.count(rem) || ::access((dir_ + "/norm_t_bfp_g1.hal").c_str(), R_OK) ||
        ::access((sw + ".xdna").c_str(), R_OK))
      return false;
    // the norm also writes gate / up's activations (one replay group: that stream layout; "npuffndfr_<fmt>": the
    // deferred-epilogue image's K split, emit_prefill_pp NPU_DFR_KS)
    const bool dfr = mix ? geom_.count("npuffndfr_" + std::string(f.name) + "_" + fu.name) != 0
                         : !pair && geom_.count("npuffndfr_" + std::string(f.name));
    if (dfr && ::access((dir_ + "/norm_t_bfp_g1s.hal").c_str(), R_OK)) return false;
    // (row 5 swap) this layer swaps (rswap below, checked equal); panel: its FFN norm writes the swap's token panel
    // into W ("<norm>_pn.hal", gen_half_norm PANEL) and the swap skips npu_swpanel.hal
    const bool panel = rsw0 && !dfr && PanelNorm();
    const bool panelq = panel && PanelQ();   // (the norm runs below, once the swap's calls are bound)
    if (!panelq) RunNorm(pre + "post_attention_norm.weight", NormOut::kTiled, dfr ? "ffnblk_s" : "ffnblk", false, panel);
    const NpuView ag{0, NpuK(5120).a};
    const std::uint8_t* base = gguf_.tensor_data_base();
    // (a mixed pair of one format, emit_prefill_pp NPU_SAME_SW: up follows gate in the GGUF, like a single format's)
    // (cpair: gate and up on column pairs, emit_prefill_pp NPU_COLPAIR: each from its own tensor's rows)
    const bool cpair = mix && geom_.count("npuffncp_" + std::string(f.name) + "_" + fu.name) != 0;
    if (mix && !bmix && ((!cpair && std::strcmp(f.name, fu.name)) || tu->offset != tg->offset + tg->bytes))
      throw LoomError("prefill: " + pre + " mixed image without contiguous rows");
    // (the decoder binding the NPU's gate / up rows: bmix's BFP)
    const char* const gdec = bmix ? "bfp" : f.name;
    const std::size_t blk = bmix ? kBfpBlock : tg->bytes / M * 256 / tg->dims[0], rb = blk * tg->dims[0] / 256;
    const std::size_t blkd = td->bytes / td->dims[1] * 256 / td->dims[0], rbd = blkd * td->dims[0] / 256;
    const std::size_t f0 = M - F, dgu = bmix ? M * rb : static_cast<std::size_t>(tg->bytes);
    // a combined gate / up copy: the NPU's gate rows and up rows, M rows apart (the images' dgu); the up rows between
    // them (the GPU's) are in the range registered for the NPU, which pins its pages: one aliased page run
    const auto gate_up = [&](const std::string& file) {
      auto* p = const_cast<std::uint8_t*>(HostRows(file, 2 * M * rb, {{f0 * rb, F * rb}, {(M + f0) * rb, F * rb}}));
      AliasUnread(p + M * rb, f0 * rb);
      return p;
    };
    const std::uint8_t* pg = bmix ? gate_up("npu_bfp_" + pre + "ffn_gate_up.bin") : base + tg->offset;
    const std::uint8_t* pd = base + td->offset;
    // cpair: per column (gate, up) its rows' address (row 0) and row bytes
    struct ColRows {
      const std::uint8_t* p;
      std::size_t rb;
    };
    const auto col_rows = [&](const auto* t) { return ColRows{base + t->offset, static_cast<std::size_t>(t->bytes) / M}; };
    const ColRows cg = cpair ? col_rows(tg) : ColRows{}, cu = cpair ? col_rows(tu) : ColRows{};
    // (dispatch.txt "npudnseg_<fmt>", emit_prefill_pp YAH_NPU_DNSEG) down's rows in K segments (LoomWeights KSegOf):
    // the down images read their K part's segment, srb bytes a row; else the GGUF's rows (rbd apart)
    const KSeg* ks = weights_.KSegOf(*td);
    if ((ks != nullptr) != (geom_.count("npudnseg_" + std::string(fd.name)) != 0))
      throw LoomError("prefill: " + pre + "ffn_down's K segments differ from the set's (DownKSegs)");
    for (std::size_t ci = 0; ks && ci < parts.size(); ++ci)
      if (ks->size() != parts.size() + 1 ||
          (*ks)[ci + 1] != std::pair<std::uint32_t, std::uint32_t>{static_cast<std::uint32_t>(f0 / 256 + parts[ci].u0),
                                                                   parts[ci].r})
        throw LoomError("prefill: " + pre + "ffn_down's K segments are not the FFN block's parts");
    for (const FfnPart& u : parts)
      if (u.r != u.p && !ks) throw LoomError("prefill: " + pre + "a padded down part needs its rows in K segments");
    const auto srb = [&](std::size_t ci) { return ks ? std::size_t{parts[ci].r} * blkd : rbd; };   // (a part's row stride)
    if (npu_planning_) {   // the NPU's gate / up rows (one range when up follows gate), all of down
      const std::size_t end = gguf_.tensor_data_size();
      if (bmix) {
        npu_->RegisterRaw(pg + f0 * rb, 2 * M * rb - f0 * rb + 8192);
      } else {
        const std::size_t a = tg->offset + f0 * rb;
        npu_->RegisterRaw(base + a, std::min<std::size_t>(tu->offset + tu->bytes + 4096, end) - a);
      }
      npu_->RegisterRaw(pd, std::min<std::size_t>(td->bytes + 4096, end - td->offset));
    }
    std::vector<std::uint32_t> csw;
    std::vector<std::vector<std::uint32_t>> cdn(parts.size());
    // F / 256 calls: call j's column c takes features 256 j + 32 c + [0, 32), its records at H k-blocks 32 j + 4 c ..
    // (gen_npu_gemm swcol 32)
    // (dispatch.txt "npuffnq", layout Q': H in K parts of one size (FfnBlkParts, one down each), a part in down's A
    // layout [M block][its qc calls][slab][column][576 B]: call j's records from part j / qc, (j % qc) x 4 x 4608 B)
    // (dispatch.txt "npuffnsw5", emit_prefill_pp FFNSW5: the single-format layers' image at NP 5, 320 features a call:
    // call j's column c features f0 + 1280 (j / 4) + 64 (j % 4) + 256 b + 8 P8[c] + [0, 8) of its 5 runs b, its records
    // at K part j / q5's H passes 5 ((j % q5) / 4) .. + 5, 16-token slab slots 8 (j % 4) ..; gen_npu_q qsw5)
    const bool np5 = !pair && geom_.count("npuffnsw5") != 0;
    const std::uint32_t fc = np5 ? 320 : 256;   // (features a call)
    const NpuBytes bs = NpuBytesOf(np5 ? "npubytes5_ffnsw" : "npubytes_ffnsw");
    const bool qffn = geom_.count("npuffnq") != 0;
    bool g5 = true;   // (np5: parts of whole 1280-feature groups)
    for (const FfnPart& u : parts) g5 = g5 && u.r % 5 == 0;
    if (np5 && (!qffn || !g5 || weights_.RSegOf(*tg) != F || weights_.RSegOf(*tu) != F))
      throw LoomError("prefill: the NP 5 FFN gate / up needs Q' parts of whole groups and its rows in row segments of F");
    // (Q': a part's calls; its H bytes qpart; hap: H [M block][qu units], qu = both parts' units, one "part")
    const std::uint32_t qc = parts[0].p, qu = hap ? apass : qc;
    const std::size_t qpart = std::size_t{B_ / 64} * qu * 18432;
    for (const FfnPart& u : parts)
      if (qffn && (u.p != parts[0].p || u.r != u.p) && !np5 && !(rsw0 && hap))
        throw LoomError("prefill: the Q' FFN block's K parts differ in size");
    // a part's H start: apass, unit U of the M block's [M block][units] (the image's hpass apass); else its own region
    const auto hbase = [&](std::size_t ci) {
      std::size_t U = 0;
      for (std::size_t i = 0; i < ci; ++i) U += parts[i].p;
      return apass ? U * 18432 : ci * std::size_t{B_ / 64} * parts[ci].p * 18432;
    };
    const auto c_sw = [&](std::uint32_t j) {
      if (np5) {   // (call j: group j / 4, in the part holding units 5 (j / 4) ..)
        const std::uint32_t u = 5 * (j / 4);
        std::size_t ci = 0;
        while (u >= parts[ci].u0 + parts[ci].r) ++ci;
        return NpuView{hbase(ci) + std::size_t{u - parts[ci].u0} * 18432 + (j % 4) * 8 * 144, bs.c};
      }
      return NpuView{qffn ? j / qu * qpart + std::size_t{j % qu} * 4 * 4608 : (j / 4) * 4 * 128 * 144 + (j % 4) * 32 * 144,
                     bs.c};
    };
    // (a column's raw view: 256 rows from row r0, its last record over-read)
    const auto col_bytes = [&](const ColRows& c, const auto* t, const Fmt& x) {
      return 255 * c.rb + 16 * (c.rb * 256 / static_cast<std::size_t>(t->dims[0])) + RecordRow(x.name);
    };
    // (row 5 stepping stone, RSwapOn) the swapped image: the GPU decodes the NPU's gate / up rows into A's stream and
    // encodes the norm's tiled copy into W's token panel; call (t, b) takes tokens 512 t .., features 1024 b .. (today's
    // calls 4 b .. 4 b + 3) and writes H where they do, its 8 M blocks qc x 18432 B apart (gen_npu_q GN_QSWHT)
    // (a single format's layer or a mixed pair's: the image reads bfp16 only, the GPU decodes each tensor's format)
    const bool rswap = RSwapOn() && (!pair || mix) && !np5 && qffn && !bmix && RSwapFmt(f.name) && RSwapFmt(fu.name) &&
                       hord == "_p4" && (apass == 0 || hap) && rsw0 && B_ % 512 == 0 && qc % 4 == 0;
    if (rswap != rsw0) throw LoomError("prefill: " + pre + " swap decided differently before the norm");
    for (std::uint32_t j = 0; j < F / 256 && cpair && !rswap; ++j)
      csw.push_back(npu_->BindRawPair(sw + ".xdna", sw + ".xdna", 100 + 16 * DecoderId(f.name) + DecoderId(fu.name), HA(ag),
                                      cg.p + (f0 + 256 * j) * cg.rb, col_bytes(cg, tg, f),
                                      cu.p + (f0 + 256 * j) * cu.rb, col_bytes(cu, tu, fu), HC(c_sw(j)),
                                      "ffnsw" + hord + (qffn ? std::string("_") + f.name + "_" + fu.name : "")));
    // (R5Fuse) this swap layer's index k, its stream half k % 2; carry: its split decodes swap layer k + 1
    std::size_t r5k = 0;
    bool carry = false, carried = false;
    std::string r5h;
    if (rswap && R5Fuse()) {
      const std::string& cs = pair ? fh : sh;   // (the carrier: the fused gate / up split)
      const bool shape = M == 17408 && RSwapF(F) && tg->dims[0] == 5120 && !weights_.RSegOf(*tg);
      const std::string cx = ".af.to.npu" + (F == F0 ? std::string() : "_n" + std::to_string(F)) + ".hal";
      const bool can = shape && cs.size() > cx.size() && cs.compare(cs.size() - cx.size(), cx.size(), cx) == 0;
      const bool iq4 = std::strcmp(f.name, "iq4xs") == 0 && std::strcmp(fu.name, "iq4xs") == 0;
      // (halves: the planning pass records both halves; the swap layers are listed once)
      if (npu_planning_ && std::none_of(r5_.begin(), r5_.end(), [&](const R5Layer& x) { return x.pre == pre; })) {
        r5_.push_back({pre, can ? cs.substr(0, cs.size() - cx.size()) : "", iq4 ? "" : "_" + std::string(f.name) + "_" + fu.name,
                       shape && tu->offset == tg->offset + tg->bytes, F});
        if (r5_.size() > 1 && !R5Hal(r5_.size() - 2).empty())
          Exe(R5Hal(r5_.size() - 2));   // (loaded here: the timed pass's first carry would load it)
      }
      r5k = std::find_if(r5_.begin(), r5_.end(), [&](const R5Layer& x) { return x.pre == pre; }) - r5_.begin();
      if (r5k < r5_.size()) {
        r5h = R5Hal(r5k);
        carry = !r5h.empty();
        carried = r5k > 0 && !R5Hal(r5k - 1).empty();
        if (npu_planning_ && halves_ && carried) Exe(R5Hal(r5k - 1));   // (halves: each half's part carriers)
      }
    }
    // (R5Fuse) the carried decode's bindings after the GEMM's: the next layer's gate + up, its grids, the stream half
    const auto r5_bind = [&](std::vector<hrx_buffer_ref_t>& b, std::uint64_t& wmask) {
      const std::string& np = r5_[r5k + 1].pre;
      const auto* tgn = Find(np + "ffn_gate.weight");
      const auto* tun = Find(np + "ffn_up.weight");
      hrx_buffer_ref_t wn = TRef(*tgn);
      wn.length = static_cast<std::size_t>(tgn->bytes + tun->bytes);
      b.push_back(wn);
      Fmt fg{}, fn{};
      FmtOf(static_cast<std::uint32_t>(tgn->type), &fg);
      FmtOf(static_cast<std::uint32_t>(tun->type), &fn);
      const std::vector<hrx_buffer_ref_t> gg = GemmWeights(*tgn, fg), gn = GemmWeights(*tun, fn);
      if (gg.size() > 1) b.push_back(gg[1]);
      if (gn.size() > 1 && std::strcmp(fg.name, fn.name)) b.push_back(gn[1]);
      b.push_back(AViewAbs({SwapA((r5k + 1) % 2), SwapStream(r5_[r5k + 1].f)}));
      wmask |= std::uint64_t{1} << (b.size() - 1);
    };
    if (rswap) {
      const NpuBytes bw = NpuBytesOf("npubytes_ffnswap" + SwapTag(F));
      const std::size_t sa = SwapA(R5Fuse() ? r5k % 2 : 0);
      // (a call's 64-row blocks, its H units uc; nbk calls a token panel; one call a panel: its H all qu units of an
      // M block, the pads its image skips)
      const std::uint32_t nbc = SwapNb(F), uc = nbc / 8, nbk = F / 32 / nbc;
      if (uc > qu || (nbk > 1 && qu % uc)) throw LoomError("prefill: a swap call's H spans more than a down part");
      if (R5Fuse() && SwapStream(F) != std::size_t{F} * 5120 * 9 / 4)
        throw LoomError("prefill: the swap stream is not gen_r5fuse's (2 F K / 8 records of 9 B)");
      // (YAH_PP_HALVES) the stream is one: the half that reaches a swap layer second finds its weights decoded (the GPU
      // queue runs the first half's decode before this half's job's ready store); a new layer's decode waits for every
      // swap job that reads the stream (without handing over: the other half must not reach the stream between)
      bool decoded = false;
      if (halves_ && !npu_planning_) {
        decoded = swap_layer_ == pre;
        if (!decoded) {
          SlotFree(R5Fuse() ? r5k % 2 : 0);
          swap_layer_ = pre;
        }
      }
      // (part carriers) layer k's stream comes from both halves' layer k - 1 GEMMs: this half's ready store goes after
      // the other half's part in the in-order GPU queue, so wait for it to be recorded
      if (carried && R5Part() && halves_ == 2 && !npu_planning_)
        while (part_done_[1 - half_] < static_cast<std::int64_t>(r5k) - 1) {
          if (baton_ && baton_->done[1 - half_]) throw LoomError("prefill: the other half ended before its part carrier");
          YieldHalf();
        }
      for (const auto& [t, x, gu] : {std::tuple<const core::TensorInfo*, const Fmt*, const char*>{tg, &f, "g"},
                                     {tu, &fu, "u"}}) {
        if (carried || decoded) break;
        const std::string h = "npu_swdec_" + std::string(x->name) + "_" + gu + SwapTag(F) + ".hal";
        const Geom& gd = geom_.at(h);
        const std::size_t trb = static_cast<std::size_t>(t->bytes) / M;
        hrx_buffer_ref_t wr = TRef(*t);
        wr.offset += f0 * trb, wr.length = F * trb;
        // (gen_kq_bfp: a grid format's grid is its second buffer, the GEMMs' first table)
        std::vector<hrx_buffer_ref_t> db = {wr};
        if (const auto gw = GemmWeights(*t, *x); gw.size() > 1) db.push_back(gw[1]);
        db.push_back(AViewAbs({sa, gd.tt}));
        const bool iq4 = std::strcmp(x->name, "iq4xs") == 0;
        Dispatch(Exe(h), iq4 ? "yah_iq4xs_bfp" : "yah_kq_bfp", (gd.tokens + gd.rowgrp - 1) / gd.rowgrp, 1, 1,
                 gd.rowgrp, 1, 1, db, std::uint64_t{1} << (db.size() - 1));
      }
      const Geom& gp = geom_.at("npu_swpanel.hal");
      if (!panel)
        Dispatch(Exe("npu_swpanel.hal"), "yah_bfp16_encode_wgt", (gp.tokens + gp.rowgrp - 1) / gp.rowgrp, 1, 1,
                 gp.rowgrp, 1, 1, {Ref(*normt_), WRef(0, gp.tt)}, 2);
      const std::uint32_t nt = B_ / 512;
      for (std::uint32_t i = 0; i < nt * nbk; ++i) {
        const std::uint32_t t = i / nbk, b = i % nbk;   // (call b: units uc b ..)
          // (FfnBlkFront: H's real units from the first part's pads on)
          const std::uint32_t j = uc * b, hf = FfnBlkFront(F) ? parts[0].p - parts[0].r : 0;
          csw.push_back(npu_->BindSwap(dir_ + "/npu_ffnswap_iq4xs" + SwapTag(F) + ".xdna", {sa + b * bw.a, bw.a},
                                       HW({t * bw.w, bw.w}),
                                       HC({j / qu * qpart + std::size_t{j % qu + hf} * 18432 + std::size_t{8} * t * qu * 18432,
                                           std::size_t{nbk == 1 ? 8 * qu - hf : 7 * qu + uc} * 18432})));
        }
    }
    for (std::uint32_t j = 0; j < F / fc && !cpair && !rswap; ++j)   // (np5: a call's rows span 4 x 256 + 64)
      csw.push_back(npu_->BindRaw(sw + ".xdna", sw + ".xdna", mix && !bmix ? 100 + 16 * DecoderId(f.name) + DecoderId(fu.name) : DecoderId(gdec), HA(ag),
                                  pg + (np5 ? RSegPhys(f0 + 1280 * (j / 4) + 64 * (j % 4), M, F) : f0 + 256 * j) * rb,
                                  (np5 ? kRSegBlock * rb : dgu) + (np5 ? 1087 : 255) * rb + 16 * blk + RecordRow(gdec),
                                  HC(c_sw(j)), "ffnsw" + hord + (qffn && mix ? std::string("_") + f.name : "") + (np5 ? "_5" : "")));
    // (Q': a format's qsw image is its own family, its array state differs beyond the decoder tiles)
    // down: per K part, FfnBlkCalls() calls of FfnBlkRows() rows (a last one ending at the last row)
    const bool dn4 = FfnDn4(fd.name);
    const std::string bytes = dn4 ? "npubytes4_ffndn" : "npubytes_ffndn";
    const std::uint32_t rows = FfnBlkRows(dn4), calls = FfnBlkCalls(dn4);
    const std::size_t doff = FfnBlkDown(), cp = NpuBytesOf(bytes + DnTag(parts[0], apass)).c;
    std::vector<std::string> dtag;
    // (SwDown) the down in swap form: per K part kp the GPU decodes the NPU's 4096 columns of all 5120 rows to A at
    // kp sa (npu_swddec_<fmt>, H's K order); call (kp, t) reads that stream, token panel t of H (W in C: 8 M blocks
    // apass units apart from part kp's first unit; the image's GN_QWCS) and writes C (kp, t) in swapc form
    const bool swdown = rswap && SwDown();
    if (SwDown() && !rswap) throw LoomError("prefill: " + pre + "a swap-form down set needs the swap gate / up (H as panels)");
    std::vector<std::uint32_t> cswd;
    std::size_t swc = 0, swa = 0;
    if (swdown) {
      if (!hap || apass != 32 || parts.size() != 2 || parts[0].p != 16 || parts[1].p != 16 || parts[0].r != 16 ||
          parts[1].r != 16 || td->dims[0] != 17408 || td->dims[1] != 5120 || B_ % 512)
        throw LoomError("prefill: " + pre + "the swap-form down needs H [M block][32 units], two 4096 parts, 17408 x 5120");
      const NpuBytes b = NpuBytesOf("npubytes_swdown");
      if (b.c != std::size_t{5120} * 1024) throw LoomError("prefill: npubytes_swdown's C is not 5120 x 512 bf16");
      const std::string h = "npu_swddec_" + std::string(fd.name) + ".hal";
      if (!geom_.count(h)) throw LoomError("prefill: the set has no " + h + " (" + pre + "ffn_down, swap form): re-emit it");
      const Geom& gd = geom_.at(h);
      swa = Up4k(b.a);
      const std::uint32_t nt = B_ / 512;
      for (std::uint32_t kp = 0; kp < 2; ++kp) {
        // (the part's first block of row 0: in its K segment, rows 16 blocks wide (npu_swddec without kfull); else in
        // the GGUF's rows)
        const std::size_t k0 = ks ? std::size_t{5120} * (*ks)[kp + 1].first * blkd : (f0 / 256 + parts[kp].u0) * blkd;
        hrx_buffer_ref_t wr = TRef(*td);
        wr.offset += k0, wr.length = (std::size_t{5120} - 1) * srb(kp) + 16 * blkd;
        std::vector<hrx_buffer_ref_t> db = {wr};
        if (const auto gw = GemmWeights(*td, fd); gw.size() > 1) db.push_back(gw[1]);   // (a grid format's grid)
        db.push_back(ARef(kp * swa, gd.tt));
        Dispatch(Exe(h), std::strncmp(fd.name, "iq4", 3) == 0 ? "yah_iq4xs_bfp" : "yah_kq_bfp",
                 (gd.tokens + gd.rowgrp - 1) / gd.rowgrp, 1, 1, gd.rowgrp, 1, 1, db, std::uint64_t{1} << (db.size() - 1));
        for (std::uint32_t t = 0; t < nt; ++t, swc += b.c)
          cswd.push_back(npu_->BindSwap(dir_ + "/npu_swdown.xdna", HA({kp * swa, b.a}),
                                        HA({hbase(kp) + std::size_t{8} * t * qu * 18432, (std::size_t{7} * qu + 16) * 18432, true}),
                                        HC({doff + swc, b.c}), "swdown"));
      }
    }
    for (std::size_t ci = 0; ci < parts.size() && !swdown; ++ci) {
      const FfnPart u = parts[ci];
      dtag.push_back(DnTag(u, apass));
      // (an NP 4 down: its own family, its array state differs from the NP 5 images')
      const std::string fam = "ffndn" + dtag.back() + (dn4 ? std::string("_") + fd.name : "");
      const NpuBytes bd = NpuBytesOf(bytes + dtag.back());
      if (bd.c != cp) throw LoomError("prefill: the FFN block's down parts differ in C bytes");
      for (std::uint32_t r = 0; r < calls; ++r) {
        const std::size_t r0 = std::min<std::size_t>(r * rows, td->dims[1] - rows), k0 = f0 / 256 + u.u0;
        // (a padded part's last row reads p - r blocks on: the next segment's, or past the tensor in its registered pad;
        // FfnBlkFront: its first row reads p - r blocks before, the GPU segment's last row's)
        const std::size_t front = FfnBlkFront(F) && ks ? std::size_t{u.p - u.r} * blkd : 0;
        if (FfnBlkFront(F) && !ks) throw LoomError("prefill: " + pre + "front-padded down parts need K segments");
        cdn[ci].push_back(npu_->BindRaw(dn[ci] + ".xdna", dn[ci] + ".xdna", DecoderId(fd.name),
                                        HA({qffn ? hbase(ci) : std::size_t{u.u0} * 18432, bd.a, true}),
                                        ks ? pd + td->dims[1] * (*ks)[ci + 1].first * blkd + r0 * srb(ci) - front
                                           : pd + r0 * rbd + k0 * blkd,
                                        (rows - 1) * srb(ci) + (16 + u.p - u.r) * blkd + RecordRow(fd.name),
                                        HC({doff + (ci * calls + r) * cp, cp}), fam));
      }
    }
    if (panelq) {   // (PanelQ) a carried layer's weights are decoded already: reserve the job, the quarters open it
      if (carried && !npu_planning_) panel_job_ = npu_->NewJob(csw);
      RunNorm(pre + "post_attention_norm.weight", NormOut::kTiled, "ffnblk", false, panel);
    }
    // (padded parts) H's pad units zeroed (npu_zpad_<F>.hal): the GPU's write of C orders before the job's gate store
    // (NpuEnqueue: the store writes C), after every joined job's C
    const std::string zpad = "npu_zpad_" + std::to_string(F) + ".hal";
    bool padded = false;
    for (const FfnPart& u : parts) padded = padded || u.r != u.p;
    if (padded) {
      if (!geom_.count(zpad)) throw LoomError("prefill: padded down parts without " + zpad);
      if (!npu_planning_ && (halves_ ? last_job_ >= 0 && npu_joined_ <= static_cast<std::size_t>(last_job_)
                                     : npu_joined_ != flagged_.size()))
        throw LoomError("prefill: " + zpad + " with an NPU job not joined (its C writes unordered)");
      Dispatch(Exe(zpad), "yah_npu_zpad", geom_.at(zpad).tokens, 1, 1, geom_.at(zpad).rowgrp, 1, 1,
               {CView({0, FfnBlkHOf(B_, F)})});
    }
    if (rswap) {
      NpuEnqueue(std::move(csw), "ffnswap", {{Ref(npu_->A()), Ref(npu_->W())}, Ref(npu_->C())});
      swap_reader_[R5Fuse() ? r5k % 2 : 0] = last_job_;   // (halves: the slot's latest reader)
    } else
      NpuEnqueue(std::move(csw), "ffnsw");
    // the down's K parts as one job when they run one image (no GPU work between them: a job boundary, ~120 us, saved;
    // YAH_NPU_DNJOBS=1: a job per part)
    static const bool dnjobs = std::getenv("YAH_NPU_DNJOBS") != nullptr;
    const bool one = !dnjobs && std::all_of(dn.begin(), dn.end(), [&](const std::string& d) { return d == dn[0]; });
    if (swdown)
      NpuEnqueue(std::move(cswd), "swdown", {{ARef(0, 2 * swa), CRef(0, FfnBlkH())}, CRef(doff, swc)});
    for (std::size_t ci = 0; ci < parts.size() && !swdown; ++ci) {
      if (one && ci + 1 < parts.size()) {
        cdn[ci + 1].insert(cdn[ci + 1].begin(), cdn[ci].begin(), cdn[ci].end());
        continue;
      }
      NpuEnqueue(std::move(cdn[ci]), "ffndn" + dtag[ci]);
    }
    if (pair && !fh.empty()) {   // (the fused gate / up split; mixed formats bind the up format's tables after the gate's)
      const Geom g = GeomOf(fh);
      auto b = GemmWeights(*tg, f);
      b[0].length = static_cast<std::size_t>(tg->bytes + tu->bytes);
      if (tg->type != tu->type) {
        const auto bu = GemmWeights(*tu, fu);
        b.insert(b.end(), bu.begin() + 1, bu.end());
      }
      const std::size_t first = b.size();
      for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{normt_, wstage_, ostage_, ffnup_}) b.push_back(Ref(*x));
      const std::uint32_t tt = Trim(b, first, {std::size_t(tg->dims[0]) * 2, 0, 0, std::size_t(M) * 2}, g);
      const std::string sym = "yah_ffn_gemm_" + std::string(f.name) + (tg->type != tu->type ? "_" + std::string(fu.name) : "") + "_ffn";
      std::uint64_t wmask = GemmWrites(b, f, {ffnup_});
      if (carry) r5_bind(b, wmask), SlotFree((r5k + 1) % 2);
      ClkFfnBegin(b, carry);
      Dispatch(Exe(carry ? r5h : fh), sym.c_str(), (M - F) / 16 / g.rowgrp, tt, 1, 32, 1, 1, b, wmask);
      ClkFfnEnd(b, wmask);
      if (carry) part_done_[half_] = static_cast<std::int64_t>(r5k), carriers_[half_] += !npu_planning_;
    } else if (pair) {
      RunKstoreSplit(pre + "ffn_gate.weight", *gateffn_, true, gs, F);
      RunSwigluSplit(pre + "ffn_up.weight", true, us, F);
    } else {
      const Geom g = GeomOf(hal);
      auto b = GemmWeights(*tg, f);
      b[0].length = static_cast<std::size_t>(tg->bytes + tu->bytes);
      const std::size_t first = b.size();
      for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{normt_, wstage_, ostage_, ffnup_}) b.push_back(Ref(*x));
      const std::uint32_t tt = Trim(b, first, {std::size_t(tg->dims[0]) * 2, 0, 0, std::size_t(M) * 2}, g);
      std::uint64_t wmask = GemmWrites(b, f, {ffnup_});
      if (carry) r5_bind(b, wmask), SlotFree((r5k + 1) % 2);   // (R5Fuse) swap layer k + 1's NPU rows, other half
      const std::string& gh = carry ? r5h : sh;
      ClkFfnBegin(b, carry);
      Dispatch(Exe(gh), ("yah_ffn_gemm_" + std::string(f.name) + "_ffn").c_str(), (M - F) / 16 / GeomOf(gh).rowgrp, tt, 1,
               32, 1, 1, b, wmask);
      ClkFfnEnd(b, wmask);
      if (carry) part_done_[half_] = static_cast<std::int64_t>(r5k), carriers_[half_] += !npu_planning_;
    }
    // down over them: f32 [tokens][5120]. With "npuffnrem_<fmt>_<kbw>_rr.hal" its residual epilogue adds hidden() (the
    // GPU waits for the NPU's down here anyway), so the sum after the NPU's down reads one f32 tensor less
    // ("npu_unpack_ffnblk[_4]_rr.hal": (hidden + GPU down) + NPU partials, a summation order of its own; YAH_NPU_REMEXACT=1:
    // hidden + (NPU partials + GPU down), bit-exact with the sets before it)
    static const bool remexact = std::getenv("YAH_NPU_REMEXACT") != nullptr;
    const std::string rem_rr = rem.substr(0, rem.size() - 4) + "_rr.hal";
    ffn_rr_ = !remexact && geom_.count(rem_rr) && geom_.count(dn4 ? "npu_unpack_ffnblk_4_rr.hal" : "npu_unpack_ffnblk_rr.hal");
    const Geom& gr = geom_.at(ffn_rr_ ? rem_rr : rem);
    auto r = GemmWeights(*td, fd);
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{ffnup_, wstage_, ostage_}) r.push_back(Ref(*x));
    if (ffn_rr_) r.insert(r.end() - 2, Ref(*hidden_));   // (kres: weights, input, resid, wstage, ostage, out)
    r.push_back({ffnrem_->handle, 0, std::size_t{B_} * kHidden * 4});
    Dispatch(Exe(ffn_rr_ ? rem_rr : rem), ("yah_ffn_gemm_" + std::string(fd.name) + (ffn_rr_ ? "_kres" : "")).c_str(),
             MTiles(*td) / gr.rowgrp, gr.tt, 1, 32, 1, 1, r, std::uint64_t{1} << (r.size() - 1));
    NpuJoin();
    // the sum into hidden() waits for the next layer's norm, which fuses it when it feeds the NPU (RunLayerNorm)
    ffn_pending_ = {doff, swdown ? swc : parts.size() * calls * cp};
    ffn_pending4_ = dn4 && !swdown;
    ffn_pendingsw_ = swdown;
    // (npu_unorm reads the NP 5 downs' C; after the "_rr" sum, "npu_unorm[4]_rr_*" takes any down)
    if (ffn_rr_ ? !geom_.count(UnormRr(false, NormOut::kTiled)) : !geom_.count("npu_unorm.hal") || dn4 || swdown) FlushFfn();
    return true;
  }
  // The FFN block's sum (hidden + the GPU's down + the NPU's partials) into hidden(), if RunFfnBlock left it pending.
  void FlushFfn() {
    FlushOut();
    if (!ffn_pending_.length) return;
    const hrx_buffer_ref_t remv{ffnrem_->handle, 0, std::size_t{B_} * kHidden * 4};
    if (ffn_rr_)
      NpuUnpack(ffn_pendingsw_ ? "npu_unpack_ffnblk_sw_rr.hal" : ffn_pending4_ ? "npu_unpack_ffnblk_4_rr.hal" : "npu_unpack_ffnblk_rr.hal",
                std::exchange(ffn_pending_, {}),
                {remv, Ref(*hidden2_)}, 4);
    else
      NpuUnpack(ffn_pendingsw_ ? "npu_unpack_ffnblk_sw.hal" : ffn_pending4_ ? "npu_unpack_ffnblk_4.hal" : "npu_unpack_ffnblk.hal",
                std::exchange(ffn_pending_, {}),
                {Ref(*hidden_), remv, Ref(*hidden2_)}, 8);
    std::swap(hidden_, hidden2_);
  }

  // The out site's sum (hidden + the NPU's rows' C + their K remainder; the GPU's rows are in hidden2 already) into
  // hidden(), if RunResidual left it pending.
  std::string OnormHal() const { return RowsHal(out_np4_ ? "npu_onorm_4.hal" : "npu_onorm.hal", "out", out_nn_, out_np4_); }
  // A site's row-baked HAL (unpack, out's sum) for nn NPU rows: the set's own at its rows, else the per-layer variant
  // "<hal minus .hal>_n<nn>.hal" (emit_prefill_pp YAH_NPU_SPLIT_LAYERS)
  std::string QkvHal2(const std::string& hal, const std::string& wqkv, std::uint32_t nq) const {
    return RowsHal(hal, "qkv", nq, Np4(wqkv));
  }
  std::string QkvHal(const std::string& base, const std::string& wqkv, std::uint32_t nq) const {
    return RowsHal(Np4Hal(base, wqkv), "qkv", nq, Np4(wqkv));
  }
  std::string RowsHal(const std::string& hal, const char* site, std::uint32_t nn, bool np4) const {
    const auto& m = np4 ? npu_rows4_ : npu_rows_;
    const auto it = m.find(site);
    if (hal.empty() || !nn || it == m.end() || it->second == nn) return hal;
    return hal.substr(0, hal.size() - 4) + "_n" + std::to_string(nn) + ".hal";
  }
  void FlushOut() {
    if (!out_pending_.length) return;
    NpuUnpack(RowsHal(out_np4_ ? "npu_unpack_out_4.hal" : "npu_unpack_out.hal", "out", out_nn_, out_np4_), std::exchange(out_pending_, {}),
              {Ref(*hidden_), {npurem_->handle, 0, std::size_t{B_} * out_nn_ * 4}, Ref(*hidden2_)}, 8);
    std::swap(hidden_, hidden2_);
  }

  // kRow: f16 row-major into scratch_; kTiled: fragment-major into normt_ (afrag GEMMs); kBoth: both, one pass.
  enum class NormOut { kRow, kTiled, kBoth };
  // A layer's input norm (RunNorm with npu_site). After an FFN block whose sum is pending, with npu_site set,
  // "npu_unorm[_pk].hal" adds the sum and writes the NPU's BFP16 input (the Q4_K order too with pk) in one pass; the
  // f16 output(s) for the GPU's GEMMs follow in RunPendingNorm, recorded after the NPU job is released, so the job does
  // not wait for them.
  // The fused sum + norm after the "_rr" sum with the f16 output(s) for mode (YAH_NPU_NOUNORMF=1: none, the sum and
  // the norm as two passes).
  // (ns: the no-stream form, "_ns.hal")
  std::string UnormRr(bool pk, NormOut mode, bool ns = false) const {
    static const bool off = std::getenv("YAH_NPU_NOUNORMF") != nullptr;
    if (off) return "-";
    return std::string(ffn_pendingsw_ ? "npu_unormsw_rr_" : ffn_pending4_ ? "npu_unorm4_rr_" : "npu_unorm_rr_") +
           (mode == NormOut::kRow ? "r" : mode == NormOut::kTiled ? "t" : "rt") + (ns ? "_ns.hal" : pk ? "_pk.hal" : ".hal");
  }
  // swap: the layer's NPU sites are in swap form (their stream the GPU's decoders write): no BFP16 stream, npu_site unused
  void RunLayerNorm(const std::string& wname, NormOut mode, const char* npu_site, bool pk, bool swap = false) {
    FlushOut();
    if (swap) {
      if (ffn_pending_.length && ffn_rr_ && geom_.count(UnormRr(false, mode, true))) {
        std::vector<hrx_buffer_ref_t> b{CView(std::exchange(ffn_pending_, {})),
                                        {ffnrem_->handle, 0, std::size_t{B_} * kHidden * 4}, TRef(*Find(wname)),
                                        Ref(*hidden2_)};
        if (mode != NormOut::kTiled) b.push_back(Ref(*scratch_));
        if (mode != NormOut::kRow) b.push_back(Ref(*normt_));
        const LoomExecutable& exe = Exe(UnormRr(false, mode, true));
        Dispatch(exe, "yah_half_norm", B_ / 4, 1, 1, exe.WorkgroupSize(exe.OrdinalOrZero("yah_half_norm")), 1, 1, b,
                 ~std::uint64_t{0} << 3);
        std::swap(hidden_, hidden2_);
        norm_bfp_.clear(), norm_pk_.clear();
        return;
      }
      FlushFfn();
      RunNorm(wname, mode);
      return;
    }
    if (ffn_pending_.length && ffn_rr_ && npu_site && geom_.count("npunormbfp") && geom_.count(UnormRr(pk, mode))) {
      const std::size_t a = NpuK(5120).a;
      std::vector<hrx_buffer_ref_t> b{CView(std::exchange(ffn_pending_, {})),
                                      {ffnrem_->handle, 0, std::size_t{B_} * kHidden * 4}, TRef(*Find(wname)),
                                      Ref(*hidden2_)};
      if (mode != NormOut::kTiled) b.push_back(Ref(*scratch_));
      if (mode != NormOut::kRow) b.push_back(Ref(*normt_));
      b.push_back(ARef(0, a));
      if (pk) b.push_back(ARef(PkA(), a));
      const LoomExecutable& exe = Exe(UnormRr(pk, mode));
      Dispatch(exe, "yah_half_norm", B_ / 4, 1, 1, exe.WorkgroupSize(exe.OrdinalOrZero("yah_half_norm")), 1, 1, b,
               ~std::uint64_t{0} << 3);
      std::swap(hidden_, hidden2_);
      norm_bfp_ = npu_site, norm_pk_ = pk ? npu_site : "";
      return;
    }
    if (!ffn_pending_.length || !npu_site || !geom_.count("npunormbfp") || ffn_rr_) {
      FlushFfn();
      RunNorm(wname, mode, npu_site, pk);
      return;
    }
    const std::size_t a = NpuK(5120).a;
    std::vector<hrx_buffer_ref_t> b{CView(std::exchange(ffn_pending_, {})), Ref(*hidden_),
                                    {ffnrem_->handle, 0, std::size_t{B_} * kHidden * 4}, TRef(*Find(wname)),
                                    Ref(*hidden2_), ARef(0, a)};
    if (pk) b.push_back(ARef(PkA(), a));
    const LoomExecutable& exe = Exe(pk ? "npu_unorm_pk.hal" : "npu_unorm.hal");
    Dispatch(exe, "yah_half_norm", B_ / 4, 1, 1, exe.WorkgroupSize(exe.OrdinalOrZero("yah_half_norm")), 1, 1, b,
             pk ? 0b1110000 : 0b110000);
    std::swap(hidden_, hidden2_);
    norm_bfp_ = npu_site, norm_pk_ = pk ? npu_site : "";
    pending_norm_ = wname, pending_mode_ = mode;
  }
  // GPU work an NPU job does not read, recorded right after that job's release (RunDeltaNet's postnorm part b)
  std::function<void()> after_release_;
  NpuView gc_read_{};   // (gc) the gate's C range a deferred postnorm part still reads
  std::size_t dn_c_end_ = 0;   // the end of a DeltaNet layer's qkv + gate C: its out job's C starts there
  std::string qc_hal_;   // (qc) the attention HAL that reads q / gate from qc_c_, set by the q split
  NpuView qc_c_{};   // (qc) the q split's C range that chunk 0's attention reads
  void RunAfterRelease() {
    if (after_release_) std::exchange(after_release_, nullptr)();
  }
  void RunPendingNorm() {
    if (!pending_norm_.empty()) RunNorm(std::exchange(pending_norm_, ""), pending_mode_);
  }
  // The FFN with ffn_gate and ffn_up in one afrag GEMM ("gemm_ffn_<fmt>_..", emit_prefill_pp.py ffn_fused): ffn_up
  // stored right after ffn_gate, so one binding spans them (the kernel reads up row r at m_rows + r). Gate and up of one
  // format, or a pair "gemm_ffn_<gate>_<up>_.." (AF_FFN_MIX: mixed formats, which bind the gate format's tables, then
  // the up format's, or one format without an NPU split).
  // It writes f16(silu(gate) * up) into ffnup_, fragment-major when ffn_down is afrag too. Returns false if not taken.
  bool RunFfnFused(const std::string& pre) {
    if (!AfChunk()) return false;
    const auto* tg = Find(pre + "ffn_gate.weight");
    const auto* tu = Find(pre + "ffn_up.weight");
    if (tg->dims != tu->dims || tu->offset != tg->offset + tg->bytes) return false;
    const bool mixed = tg->type != tu->type;
    Fmt f{}, fu{};
    std::string base = RSegHal(GemmHal("gemm_ffn", *tg, &f), *tg);   // (rseg: its layout's GEMM)
    GemmHal("gemm_ffn", *tu, &fu);
    const bool down_af = Af("gemm_kres", pre + "ffn_down.weight");
    const char* sfx = down_af ? ".af.to.hal" : ".af.hal";
    std::string hal = mixed ? "" : AfHal(base, sfx);
    if (hal.empty()) {   // "gemm_ffn_<gate>_<mt>_<kb>.hal" -> "gemm_ffn_<gate>_<up>_<mt>_<kb>.hal"
      base.insert(std::strlen("gemm_ffn_") + std::strlen(f.name), "_" + std::string(fu.name));
      hal = AfHal(base, sfx);
    }
    if (hal.empty()) return false;
    const std::string sym = "yah_ffn_gemm_" + std::string(f.name) + (mixed ? "_" + std::string(fu.name) : "") + "_ffn";
    RunNorm(pre + "post_attention_norm.weight", NormOut::kTiled);
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*tg, f);
    b[0].length = static_cast<std::size_t>(tg->bytes + tu->bytes);
    if (mixed) {
      const auto bu = GemmWeights(*tu, fu);
      b.insert(b.end(), bu.begin() + 1, bu.end());
    }
    const std::size_t first = b.size();
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{normt_, wstage_, ostage_, ffnup_}) b.push_back(Ref(*x));
    const std::uint32_t tt = Trim(b, first, {std::size_t(tg->dims[0]) * 2, 0, 0, std::size_t(tg->dims[1]) * 2}, g);
    Dispatch(Exe(hal), sym.c_str(), MTiles(*tg) / g.rowgrp, tt, 1, 32, 1, 1, b, GemmWrites(b, f, {ffnup_}));
    RunResidual(pre + "ffn_down.weight", *ffnup_, down_af);
    return true;
  }

  // The RMSNorm of hidden() rows into scratch_ (f16; afrag: fragment-major into normt_). npu_site: the NPU split of a
  // K = 5120 site reads this norm's output; with "npunormbfp" in the set the norm also writes its BFP16 input into A ("<norm>_bfp.hal") and
  // NpuEncode(npu_site) skips the encoder; pk: also the Q4_K-order stream into A's second half ("<norm>_bfp_pk.hal",
  // with "npunormbfppk" in the set; NpuEncode pk then skips its encoder too).
  // panel (ffnblk only): the "_pn.hal" form, its BFP16 output the swap's token panel in W (npu_swpanel.hal's bytes)
  void RunNorm(const std::string& wname, NormOut mode = NormOut::kRow, const char* npu_site = nullptr, bool pk = false,
               bool panel = false) {
    // A row takes `split` waves (dispatch.txt norm_split, else 1); a workgroup of w waves takes w / split rows.
    // npu_site "ffnblk": the FFN block's one-replay-group stream (RunFfnBlock, fragment-major output only); "ffnblk_s":
    // in the deferred-epilogue images' K split
    const bool g1s = npu_site && std::strcmp(npu_site, "ffnblk_s") == 0;
    const bool g1 = g1s || (npu_site && std::strcmp(npu_site, "ffnblk") == 0);
    if (panel && (!g1 || g1s || mode != NormOut::kTiled)) throw LoomError("prefill: a panel norm off the FFN block");
    // (the NPU input binding, built only where a norm writes one: npu_ is null on GPU-only norms)
    const auto ab = [&] {
      return panel ? WRef(0, geom_.at("npu_swpanel.hal").tt)
                   : ARef(0, NpuK(5120).a);
    };
    const std::string pn = panel ? "_pn" : "";
    if (out_pending_.length && g1 && !g1s && mode == NormOut::kTiled && geom_.count(OnormHal())) {
      // the out site's sum fused with this norm (gen_half_norm.gen_osum): c, resid, rem, then the norm's bindings with
      // x = hidden2 (the GPU's rows read, the NPU's written by the sum)
      const std::string oh = OnormHal();
      const std::string stem = oh.substr(0, oh.size() - 4) + pn;
      const std::vector<hrx_buffer_ref_t> ob = {
          CView(std::exchange(out_pending_, {})), Ref(*hidden_), {npurem_->handle, 0, std::size_t{B_} * out_nn_ * 4},
          Ref(*hidden2_), Ref(*normnil_), TRef(*Find(wname)), Ref(*normnil_), Ref(*normt_), ab()};
      // YAH_NPU_PANELQ=1 (chunked FFN norm, the swap's panel form): 4 token-quarter dispatches ("<pn>_q<q>.hal",
      // emitted with YAH_EMIT_PANELQ=1), in order (they bind the same buffers), so quarter q's panel is done first
      static const bool panelq = std::getenv("YAH_NPU_PANELQ") && !std::strcmp(std::getenv("YAH_NPU_PANELQ"), "1");
      if (panel && panelq && geom_.count(stem + "_q0.hal")) {
        for (int q = 0; q < 4; ++q) {
          const LoomExecutable& qe = Exe(stem + "_q" + std::to_string(q) + ".hal");
          Dispatch(qe, "yah_half_norm", B_ / 16, 1, 1, qe.WorkgroupSize(qe.OrdinalOrZero("yah_half_norm")), 1, 1, ob,
                   (1u << 3) | (1u << 7) | (1u << 8));
          if (panel_job_ && q < 3 && !npu_planning_)   // (PanelQ) panel q open: its W panel and A are written
            graph_->AtomicStore(FlagWord(panel_job_->ready), panel_job_->gate - 3 * 16 + 16 * q,
                                HRX_ATOMIC_FLAG_RELEASE | HRX_ATOMIC_FLAG_SYSTEM_SCOPE,
                                {Ref(npu_->A()), Ref(npu_->W())}, {Ref(npu_->C())});
        }
      } else {
        const LoomExecutable& exe = Exe(stem + ".hal");
        Dispatch(exe, "yah_half_norm", B_ / 4, 1, 1, exe.WorkgroupSize(exe.OrdinalOrZero("yah_half_norm")), 1, 1, ob,
                 (1u << 3) | (1u << 7) | (1u << 8));
      }
      std::swap(hidden_, hidden2_);
      norm_bfp_.clear(), norm_pk_.clear();
      return;
    }
    FlushOut();
    const bool bfp = npu_site && (g1 || geom_.count("npunormbfp"));
    norm_bfp_ = bfp && !g1 ? npu_site : "";
    const std::string hal = mode == NormOut::kTiled ? "norm_t" : mode == NormOut::kBoth ? "norm_rt" : "norm";
    const bool wpk = bfp && !g1 && pk && geom_.count("npunormbfppk");
    norm_pk_ = wpk ? npu_site : "";
    const LoomExecutable& exe = Exe(g1s ? "norm_t_bfp_g1s.hal" : g1 ? "norm_t_bfp_g1" + pn + ".hal"
                                    : hal + (wpk ? "_bfp_pk.hal" : bfp ? "_bfp.hal" : ".hal"));
    const std::uint32_t ws = exe.WorkgroupSize(exe.OrdinalOrZero("yah_half_norm"));
    const auto ns = geom_.find("norm_split");
    const std::uint32_t split = ns != geom_.end() && ns->second.rowgrp ? ns->second.rowgrp : 1;
    const std::uint32_t rows_per_wg = ws ? ws / 32 / split : 1;
    if (!rows_per_wg || (ws && ws % (32 * split))) throw LoomError("norm.hal: workgroup size does not match norm_split");
    if (B_ % rows_per_wg) throw LoomError("norm.hal: rows per workgroup must divide the chunk");
    std::vector<hrx_buffer_ref_t> b =
        mode == NormOut::kBoth
            ? std::vector<hrx_buffer_ref_t>{Ref(*hidden_), Ref(*normnil_), TRef(*Find(wname)), Ref(*normnil_),
                                            Ref(*scratch_), Ref(*normt_)}
            : std::vector<hrx_buffer_ref_t>{Ref(*hidden_), Ref(*normnil_), TRef(*Find(wname)), Ref(*normnil_),
                                            Ref(mode == NormOut::kTiled ? *normt_ : *scratch_)};
    if (bfp) b.push_back(ab());
    if (wpk) b.push_back(ARef(PkA(), NpuK(5120).a));
    Dispatch(exe, "yah_half_norm", B_ / rows_per_wg, 1, 1, 32, 1, 1, b);
  }
  // NPU column split (EnableNpu, emit_prefill_pp.npu_split). A site's NPU rows in this chunk (0: GPU only).
  std::uint32_t NpuRows(const char* site) const {
    const auto it = npu_rows_.find(site);
    return npu_on_ && it != npu_rows_.end() ? it->second : 0;
  }
  // A site's NPU rows on wname (its format's image family: Np4). A per-layer row (dispatch.txt "npusplit_<site>@<L>",
  // emit_prefill_pp YAH_NPU_SPLIT_LAYERS; L from wname's "blk.<L>.") overrides the set's rows for that layer.
  std::uint32_t NpuRows(const char* site, const std::string& wname) const {
    if (npu_on_ && wname.rfind("blk.", 0) == 0) {
      const auto it = geom_.find(std::string("npusplit_") + site + "@" + wname.substr(4, wname.find('.', 4) - 4));
      if (it != geom_.end()) {
        if (it->second.tokens % NpuCallRows(Np4(wname))) throw LoomError("prefill: " + it->first + " is not whole NPU calls");
        return it->second.tokens;
      }
    }
    if (!Np4(wname)) return NpuRows(site);
    const auto it = npu_rows4_.find(site);
    return npu_on_ && it != npu_rows4_.end() ? it->second : 0;
  }
  // wname's site image is NP 4 (Q' P1K sets: emit_prefill_pp QNP4_FMTS, dispatch.txt "npunp4_<fmt>"): 512-row calls,
  // "npusplit4_<site>" rows, "npubytes4_<K>", and the "_4" unpacks / conv split.
  bool Np4(const std::string& wname) const {
    if (!npu_ || !geom_.count("npurows4")) return false;
    const char* const dec = SiteDec(wname);
    return dec && geom_.count(std::string("npunp4_") + dec) != 0;
  }
  // The unpack / conv HAL name of wname's family ("<base>.hal" or "<base>_4.hal").
  std::string Np4Hal(const std::string& base, const std::string& wname) const {
    return base + (Np4(wname) ? "_4.hal" : ".hal");
  }
  // (k offset, K) of a site's NPU calls: one K, or down's 6 + 6 + 5 passes of K = 17408 (its weight panel does not fit
  // a memory tile).
  // The NPU runs one image (K = 5120 per call; an image switch costs ~0.55 ms): a site's K in chunks of 5120, the rest
  // of K (out: 1024, down: 2048) on the GPU ("npurem_<site>_<fmt>.hal"), added by the unpack.
  static std::uint32_t NpuSiteK(const std::string& site) { return site == "out" ? 6144 : 5120; }
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
  NpuBytes NpuK(std::uint32_t K, bool np4 = false) const {
    const Geom& g = geom_.at((np4 ? "npubytes4_" : "npubytes_") + std::to_string(K));
    return {g.tokens, g.rowgrp, g.tt};
  }
  // Output rows per NPU call (dispatch.txt "npurows": 8 columns of gen_npu_gemm.TN; np4: "npurows4").
  std::uint32_t NpuCallRows(bool np4 = false) const { return geom_.at(np4 ? "npurows4" : "npurows").tokens; }
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
  // decoder id.
  std::string DcolImage(const std::string& wname, int* dec = nullptr) const {
    const char* const name = SiteDec(wname);
    if (!name) return "";
    const int id = DecoderId(name);
    if (id < 0) return "";
    if (dec) *dec = id;
    const std::string path = dir_ + "/npu_dcol_" + name + "_" + std::to_string(Find(wname)->dims[0]) + ".xdna";
    return ::access(path.c_str(), R_OK) == 0 ? path : "";
  }
  // A site tensor whose NPU rows the emitter pre-decoded (emit_prefill_pp NPU_BFPSITE: npu_bfp_<wname>.bin, the
  // BFP decoder's rows).
  bool SiteBfp(const std::string& wname) const { return geom_.count("npusitebx_" + wname) != 0; }
  // The decoder binding wname's NPU rows (its format's, else BFP's), nullptr if none.
  const char* SiteDec(const std::string& wname) const {
    Fmt f{};
    if (SiteBfp(wname)) return "bfp";
    return FmtOf(static_cast<std::uint32_t>(Find(wname)->type), &f) ? f.name : nullptr;
  }
  // The base of NpuRawRows' offsets: the model's tensor data, or a pre-decoded site's rows (NpuRawRows read them).
  const std::uint8_t* RawBase(const std::string& wname) const {
    return SiteBfp(wname) ? bfp_.at("npu_bfp_" + wname + ".bin") : gguf_.tensor_data_base();
  }
  // The NPU can take wname's rows (a decoder-column set needs its format's decoder).
  bool NpuOk(const std::string& wname) const { return !npu_ || SwSiteOf(wname) || !DcolImage(wname).empty(); }
  // A site tensor ("blk.<L>.attn_q / attn_qkv / attn_gate / attn_output / ssm_out.weight") in swap form in its layer
  bool SwSiteOf(const std::string& wname) const {
    if (wname.rfind("blk.", 0) != 0) return false;
    const std::size_t d = wname.find('.', 4);
    static const std::pair<const char*, const char*> kSites[] = {{"attn_q.weight", "q"}, {"attn_qkv.weight", "qkv"},
        {"attn_gate.weight", "gate"}, {"attn_output.weight", "out"}, {"ssm_out.weight", "out"}};
    for (const auto& [t, site] : kSites)
      if (wname.compare(d + 1, std::string::npos, t) == 0)
        return geom_.count(std::string("npuswsite_") + site + "@" + wname.substr(4, d - 4)) != 0;
    return false;
  }
  // wname's decoder pushes Q4_K's k-block order (the activations' "pk" stream, NpuEncode pk).
  bool NpuPk(const std::string& wname) const {
    Fmt f{};
    return npu_ && FmtOf(static_cast<std::uint32_t>(Find(wname)->type), &f) && std::strcmp(f.name, "q4k") == 0;
  }
  // input (row-major, or fragment-major with tiled) encoded into A for each of the site's K chunks; their A views.
  // pk (decoder-column sets): the Q4_K k-block order ("_pk" encoders, A's second half; the fused writers write the
  // p4 order of the other decoders).
  std::vector<NpuView> NpuEncode(const std::string& site, const LoomBuffer& input, bool tiled, bool pk = false) {
    const auto chunks = NpuChunks(site);
    // a K = 5120 site after a norm that wrote its input (RunNorm npu_site; the Q4_K order: RunLayerNorm pk)
    std::string& wrote = pk ? norm_pk_ : norm_bfp_;
    const bool from_norm = wrote == site;
    wrote.clear();
    const std::uint32_t site_k = NpuSiteK(site);
    std::vector<NpuView> views;
    std::size_t off = pk ? PkA() : 0;
    for (std::size_t c = 0; c < chunks.size(); ++c) {
      const std::size_t bytes = NpuK(chunks[c].second).a;
      const auto [k0, K] = chunks[c];
      const std::uint32_t cover = site_k > k0 ? std::min(K, site_k - k0) : 0;
      if (cover && !from_norm) {
        const std::string hal = "npu_enc_" + std::to_string(NpuSiteK(site)) + (NpuChunked(site) ? "_c" + std::to_string(c) : "") +
                                (cover < K ? "p" : "") + (tiled ? "_t" : "") + (pk ? "_pk" : "") + ".hal";
        const Geom& g = geom_.at(hal);
        Dispatch(Exe(hal), "yah_bfp16_encode_act", g.tokens, 1, 1, g.rowgrp, 1, 1,
                 {Ref(input), ARef(off, bytes)}, 2);
      }
      views.push_back({off, bytes});
      off += bytes;
    }
    return views;
  }
  // The NPU's rows of wname (its last rows) as raw GGUF bytes, per chunk and call (offsets from the tensor data base).
  // A call reads, per row and pass, one record of 4 super-blocks from the chunk's first k; the
  // planning pass registers the rows (and the last records' over-read) with the NPU.
  std::vector<std::vector<NpuView>> NpuRawRows(const std::string& site, const std::string& wname, std::uint32_t rows) {
    const auto* t = Find(wname);
    const bool bx = SiteBfp(wname);
    const std::size_t K = t->dims[0], M = t->dims[1];
    const std::size_t row_bytes = bx ? K / 256 * kBfpBlock : static_cast<std::size_t>(t->bytes) / M;
    const std::size_t blk = row_bytes * 256 / K, first = (bx ? 0 : t->offset) + (M - rows) * row_bytes;
    const std::size_t cr = NpuCallRows(Np4(wname));
    const char* const dec = SiteDec(wname);
    if (!dec) throw LoomError("prefill: no NPU format for " + wname);
    const std::size_t extent = (cr - 1) * row_bytes + 16 * blk + RecordRow(dec);   // gen_npu_gemm.fill_inputs
    if (bx) HostRows("npu_bfp_" + wname + ".bin", M * row_bytes, {{first, rows * row_bytes}});
    if (npu_planning_) {
      const std::size_t end = bx ? first + rows * row_bytes + 8192
                                 : std::min(first + rows * row_bytes + 4096, gguf_.tensor_data_size());
      npu_->RegisterRaw(RawBase(wname) + first, end - first);
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
  // Returns the C view they fill. Decoder-column sets: wname's image, its raw rows w (NpuRawRows).
  // (swap-form sites, emit YAH_NPU_SWSITE) layer L's site image rows R: dispatch.txt "npuswsite_<site>@<L> <nb> <real
  // rows>" (R = 64 nb), 0: in-core; the largest R over the layers; a site's C (B / 512 calls of the image's C bytes:
  // panel p at p R 1024 B); gate's stream in A after the largest qkv stream (qkv's at 0)
  std::uint32_t SwSiteR(const std::string& site, std::uint32_t l, std::uint32_t rows) const {
    const auto it = geom_.find("npuswsite_" + site + "@" + std::to_string(l));
    if (it == geom_.end()) return 0;
    if (it->second.rowgrp != rows)
      throw LoomError("prefill: npuswsite_" + site + "@" + std::to_string(l) + "'s rows differ from the split's");
    return it->second.tokens * 64;
  }
  std::uint32_t SwSiteMaxR(const std::string& site) const {
    std::uint32_t m = 0;
    for (auto j = geom_.lower_bound("npuswsite_" + site + "@"); j != geom_.end() && j->first.starts_with("npuswsite_" + site + "@"); ++j)
      m = std::max(m, j->second.tokens * 64);
    return m;
  }
  std::size_t SwSiteC(std::uint32_t r) const {
    const NpuBytes b = NpuBytesOf("npubytes_swsite_" + std::to_string(r));
    if (b.c != std::size_t{r} * 1024) throw LoomError("prefill: npubytes_swsite_" + std::to_string(r) + "'s C is not R 512 bf16");
    return std::size_t{B_} / 512 * b.c;
  }
  std::size_t SwSiteGateA() const { return Up4k(NpuBytesOf("npubytes_swsite_" + std::to_string(SwSiteMaxR("qkv"))).a); }
  // Site rows of wname (its last rows: the NPU's) decoded to the swap's plain stream at A offset a (npu_swsdec_<fmt>_<rows>)
  void SwSiteDecode(const std::string& wname, std::uint32_t rows, std::size_t a) {
    const auto* t = Find(wname);
    Fmt f{};
    FmtOf(static_cast<std::uint32_t>(t->type), &f);
    const std::string h = std::string(t->dims[0] == 6144 ? "npu_swsdeco_" : "npu_swsdec_") + f.name + "_" +
                          std::to_string(rows) + ".hal";
    if (!geom_.count(h)) throw LoomError("prefill: the set has no " + h + " (" + wname + "'s swap-form site): re-emit it");
    const Geom& gd = geom_.at(h);
    const std::size_t M = t->dims[1], trb = static_cast<std::size_t>(t->bytes) / M;
    hrx_buffer_ref_t wr = TRef(*t);
    wr.offset += (M - rows) * trb, wr.length = rows * trb;
    std::vector<hrx_buffer_ref_t> db = {wr};
    if (const auto gw = GemmWeights(*t, f); gw.size() > 1) db.push_back(gw[1]);   // (a grid format's grid)
    db.push_back(ARef(a, gd.tt));
    Dispatch(Exe(h), std::strncmp(f.name, "iq4", 3) == 0 ? "yah_iq4xs_bfp" : "yah_kq_bfp", (gd.tokens + gd.rowgrp - 1) / gd.rowgrp,
             1, 1, gd.rowgrp, 1, 1, db, std::uint64_t{1} << (db.size() - 1));
  }
  // The norm's tiled output as the swap's token panel (npu_swpanel) in W; returns its bytes
  // (out: "npu_swpanel_o[_t].hal" from its input's k 0 .. 5120)
  std::size_t SwSitePanel(const std::string& hal = "npu_swpanel.hal", const LoomBuffer* input = nullptr) {
    const Geom& gp = geom_.at(hal);
    Dispatch(Exe(hal), "yah_bfp16_encode_wgt", (gp.tokens + gp.rowgrp - 1) / gp.rowgrp, 1, 1, gp.rowgrp, 1, 1,
             {Ref(input ? *input : *normt_), WRef(0, gp.tt)}, 2);
    return gp.tt;
  }
  // the layer of a "blk.<L>." name
  static std::uint32_t BlkOf(const std::string& name) { return static_cast<std::uint32_t>(std::stoul(name.substr(4))); }
  // A swap-form site's job: one npu_swsite_<r> call a 512-token panel (stream at A offset a, panel t at W t w, C at
  // c_off on); returns its C view
  NpuView SwSiteCalls(std::uint32_t r, std::size_t a, std::size_t& c_off, std::vector<std::uint32_t>& calls) {
    const NpuBytes b = NpuBytesOf("npubytes_swsite_" + std::to_string(r));
    const std::size_t start = c_off;
    for (std::uint32_t t = 0; t < B_ / 512; ++t, c_off += b.c)
      calls.push_back(npu_->BindSwap(dir_ + "/npu_swsite_" + std::to_string(r) + ".xdna", HA({a, b.a}), HW({t * b.w, b.w}),
                                     HC({c_off, b.c}), "swsite_" + std::to_string(r)));
    return {start, c_off - start};
  }
  NpuView NpuCalls(const std::string& site, const std::vector<NpuView>& a, const std::vector<std::vector<NpuView>>& w,
                   std::size_t& c_off, std::vector<std::uint32_t>& calls, const std::string& wname) {
    const auto chunks = NpuChunks(site);
    const std::size_t start = c_off;
    int dec = -1;
    for (std::size_t c = 0; c < chunks.size(); ++c) {
      const std::string image = DcolImage(wname, &dec);
      const std::size_t cp = NpuK(chunks[c].second, Np4(wname)).c;
      for (const NpuView& wv : w[c]) {
        // each image its own family (a family change runs the image's setup; a shared family would splice a whole
        // image's first call, which stalls the array)
        calls.push_back(npu_->BindRaw(image, image, dec, HA(a[c]), RawBase(wname) + wv.offset, wv.length, HC({c_off, cp}),
                                      image));
        c_off += cp;
      }
    }
    return {start, c_off - start};
  }
  // npu_unpack_<site>: C view c, then bindings (writes: the mask over all bindings, C at bit 0).
  void NpuUnpack(const std::string& hal, NpuView c, std::vector<hrx_buffer_ref_t> rest, std::uint64_t writes) {
    const Geom& g = geom_.at(hal);
    rest.insert(rest.begin(), CRef(c.offset, c.length));
    Dispatch(Exe(hal), "yah_npu_unpack", g.tokens, 1, 1, g.rowgrp, 1, 1, rest, writes);
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
    for (const LoomBuffer* t : {grid_iq3s_, grid_iq3xxs_, grid_iq2xxs_, grid_iq2xs_, grid_iq2s_, ksigns_, eps_, normnil_})
      g.ReadOnly(t->handle);
    graph_ = &g;
  }
  // Launch the graph recorded so far and record the rest of the chunk into a new one, so the GPU starts while the host
  // records. Stream order runs the new graph after the whole old one: no dependencies cross.
  void SplitGraph() {
    split_armed_ = false;
    split_layer_ = 0;   // (once a chunk: with halves both reach the layer)
    FinishEmbed();
    chunk_graph_->Launch();
    if (npu_on_ && npu_enq_ < flagged_.size()) {   // (NpuEarly) the jobs so far, before any later one (NPU order)
      try {
        npu_->Enqueue({flagged_.begin() + static_cast<std::ptrdiff_t>(npu_enq_), flagged_.end()});
      } catch (...) {
        gpu_.Synchronize();
        throw;
      }
      npu_enq_ = flagged_.size();
    }
    NewGraph();
  }
  // YAH_NPU_EARLY=1: with the NPU split on, the chunk's graph splits after layer 0 too (SplitGraph): the GPU starts
  // while the host records the other layers, and the jobs recorded so far go to the NPU at the split (the rest at
  // LaunchChunk, in order). Not with NPU attention (its jobs interleave by recording order), YAH_NPU_FAKE / _CPU.
  bool NpuEarly() const {
    static const bool on = [] { const char* e = std::getenv("YAH_NPU_EARLY"); return e && std::strcmp(e, "1") == 0; }();
    return on && !npu_attn_ && !std::getenv("YAH_NPU_FAKE") && !std::getenv("YAH_NPU_CPU");
  }


  // NPU handoffs are graph nodes on flag words in host memory (NpuSplit::Flags), so the host never splits the chunk.
  // NpuEnqueue: the graph stores the job's gate value to its ready word once its encoders are done (release, system
  // scope: A reaches memory first); the NPU waits for it itself. NpuJoin: yah_npu_flag_wait polls the done word; it
  // stands for the NPU's reads of A and writes of C, so the unpacks and the next job's encoders order after it. The jobs are queued right after the chunk graph launches.
  hrx_buffer_ref_t FlagWord(std::uint32_t w) const { return {npu_->Flags()->handle, std::size_t{w} * 4, 4}; }
  // (YAH_PP_HALVES) this half's A / C regions start at npu_a0_ / npu_c0_; the swap stream (SwapA) is absolute
  // (bounds: a view past its half's region, or past the buffer, throws before anything runs)
  void HalfBound(std::size_t off, std::size_t len, std::size_t r0, std::size_t one, const LoomBuffer& buf,
                 const char* what) const {
    if (off + len > buf.size || (halves_ && (off < r0 || off + len > r0 + one)))
      throw LoomError(std::string("prefill: an NPU ") + what + " view [" + std::to_string(off) + ", +" +
                      std::to_string(len) + ") outside its half's region");
  }
  hrx_buffer_ref_t AView(NpuView v) const { return ARef(v.offset, v.length); }
  hrx_buffer_ref_t AViewAbs(NpuView v) const {
    if (v.offset + v.length > npu_->A().size) throw LoomError("prefill: an NPU A stream view past A");
    hrx_buffer_ref_t r;
    r.buffer = npu_->A().handle, r.offset = v.offset, r.length = v.length;
    return r;
  }
  hrx_buffer_ref_t CView(NpuView v) const { return CRef(v.offset, v.length); }
  // (halves) this half's W region: the swap's token panel
  hrx_buffer_ref_t WRef(std::size_t off, std::size_t len) const {
    HalfBound(npu_w0_ + off, len, npu_w0_, w_one_, npu_->W(), "W");
    hrx_buffer_ref_t r;
    r.buffer = npu_->W().handle, r.offset = npu_w0_ + off, r.length = len;
    return r;
  }
  NpuView HW(NpuView v) const {
    v.offset += npu_w0_;
    HalfBound(v.offset, v.length, npu_w0_, w_one_, npu_->W(), "call W");
    return v;
  }
  hrx_buffer_ref_t ARef(std::size_t off, std::size_t len) const {
    HalfBound(npu_a0_ + off, len, npu_a0_, a_one_, npu_->A(), "A");
    hrx_buffer_ref_t r;
    r.buffer = npu_->A().handle, r.offset = npu_a0_ + off, r.length = len;
    return r;
  }
  hrx_buffer_ref_t CRef(std::size_t off, std::size_t len) const {
    HalfBound(npu_c0_ + off, len, npu_c0_, c_one_, npu_->C(), "C");
    hrx_buffer_ref_t r;
    r.buffer = npu_->C().handle, r.offset = npu_c0_ + off, r.length = len;
    return r;
  }
  // an NPU call's A view (in A, or in C: in_c) and C view in this half's regions
  NpuView HA(NpuView v) const {
    v.offset += v.in_c ? npu_c0_ : npu_a0_;
    if (v.in_c)
      HalfBound(v.offset, v.length, npu_c0_, c_one_, npu_->C(), "call A-in-C");
    else
      HalfBound(v.offset, v.length, npu_a0_, a_one_, npu_->A(), "call A");
    return v;
  }
  NpuView HC(NpuView v) const {
    v.offset += npu_c0_;
    HalfBound(v.offset, v.length, npu_c0_, c_one_, npu_->C(), "call C");
    return v;
  }
  // A job's NPU side: the A ranges it reads and the C range it writes (empty a: all of A and C).
  struct NpuSide {
    std::vector<hrx_buffer_ref_t> a;
    hrx_buffer_ref_t c;
  };
  // The job's calls; tag names it in the NPU stats. Returns its index in this chunk (NpuJoin).
  // (PanelQ) the swap job's words, reserved before its FFN norm's quarters (the next NpuEnqueue takes them)
  std::optional<NpuSplit::Job> panel_job_;
  std::uint32_t NpuEnqueue(std::vector<std::uint32_t> calls, std::string tag, NpuSide side = {}) {
    const std::uint32_t job = npu_job_++;
    if (npu_planning_) return job;
    if (side.a.empty()) side = {{Ref(npu_->A())}, Ref(npu_->C())};
    const NpuSplit::Job words = panel_job_ ? *std::exchange(panel_job_, std::nullopt) : npu_->NewJob(calls);
    graph_->AtomicStore(FlagWord(words.ready), words.gate, HRX_ATOMIC_FLAG_RELEASE | HRX_ATOMIC_FLAG_SYSTEM_SCOPE, side.a,
                        {side.c});
    flagged_.push_back({std::move(calls), std::move(tag), words, layer_});
    job_side_.push_back(std::move(side));
    last_job_ = static_cast<std::int64_t>(flagged_.size() - 1);
    return job;
  }
  // Queue the GPU work that runs beside job (default: the last enqueued one) before this.
  void NpuJoin(std::int64_t job = -1, bool handover = true) {
    if (npu_planning_) return;
    // YAH_PP_YIELD=<tag,..> (diag, HALVES 2): hand over only at joins of jobs whose tag starts with one of these
    static const std::vector<std::string> yield_tags = [] {
      std::vector<std::string> v;
      if (const char* e = std::getenv("YAH_PP_YIELD"))
        for (const char *p = e, *q; *p; p = *q ? q + 1 : q) {
          q = std::strchr(p, ',');
          if (!q) q = p + std::strlen(p);
          v.emplace_back(p, q);
        }
      return v;
    }();
    bool yield = true;
    if (!yield_tags.empty() && halves_) {
      const std::int64_t jj = job < 0 ? last_job_ : job;
      yield = false;
      if (jj >= 0)
        for (const std::string& t : yield_tags) yield = yield || flagged_.at(static_cast<std::size_t>(jj)).tag.rfind(t, 0) == 0;
    }
    if (yield && handover) YieldHalf();
    if (job < 0 && halves_) job = last_job_;
    // (halves) a job before one already joined is done (the NPU runs its queue in order) and its words are cleared:
    // no wait (it would poll a zero done word until the timeout)
    if (halves_ && job >= 0 && static_cast<std::size_t>(job) < npu_joined_) return;
    NpuWait w = NpuJoinBegin(job);
    if (waitlog_) {   // YAH_NPU_WAITLOG: the wait's poll count to its own slot (ReportWaitLog)
      const NpuSplit::Queued& q = flagged_.at(job < 0 ? flagged_.size() - 1 : static_cast<std::size_t>(job));
      if (waitlog_tag_.size() == kWaitLogSlots) throw LoomError("YAH_NPU_WAITLOG: more waits than log slots");
      w.words.push_back({waitlog_->handle, waitlog_tag_.size() * 4, 4});
      waitlog_tag_.push_back(std::to_string(q.layer) + " " + q.tag);
      Dispatch(Exe("npu_flag_wait_log.hal"), "yah_npu_flag_wait", 1, 1, 1, 32, 1, 1, w.words, 12, &w.npu_side);
    } else if (stamp_) {   // YAH_CLK_PROBE: REALTIME at the wait's entry and exit to its own 8-byte slot (ReportClk)
      const bool rec = !npu_planning_ && !warming_;
      if (rec && stamp_n_ == kWaitLogSlots) throw LoomError("YAH_CLK_PROBE: more waits than stamp slots");
      w.words.push_back({stamp_->handle, (rec ? stamp_n_++ : 0) * 8, 8});
      Dispatch(Exe("npu_flag_wait_stamp.hal"), "yah_npu_flag_wait", 1, 1, 1, 32, 1, 1, w.words, 12, &w.npu_side);
    } else {
      Dispatch(Exe("npu_flag_wait.hal"), "yah_npu_flag_wait", 1, 1, 1, 32, 1, 1, w.words, 4, &w.npu_side);
    }
    NpuJoinEnd(job);
  }
  // NpuJoin's parts around the dispatch that waits for the last enqueued job (yah_npu_flag_wait): what that dispatch
  // binds (the job's flag words done, ready, status; status is written) and orders (the NPU-side buffers, as
  // also-writes).
  struct NpuWait {
    std::vector<hrx_buffer_ref_t> words, npu_side;
  };
  // The wait stands for the NPU work of every job up to job not joined before (the NPU runs them in order).
  NpuWait NpuJoinBegin(std::int64_t job_or_last = -1) {
    const auto job = static_cast<std::uint32_t>(job_or_last < 0 ? flagged_.size() - 1 : job_or_last);
    const NpuSplit::Queued& j = flagged_.at(job);
    NpuWait w{{FlagWord(j.words.done), FlagWord(j.words.ready), FlagWord(NpuSplit::kFlagStatus)}, {}};
    for (std::size_t i = npu_joined_; i <= job; ++i) {
      w.npu_side.insert(w.npu_side.end(), job_side_[i].a.begin(), job_side_[i].a.end());
      w.npu_side.push_back(job_side_[i].c);
    }
    return w;
  }
  void NpuJoinEnd(std::int64_t job_or_last = -1) {
    const std::size_t job = job_or_last < 0 ? flagged_.size() - 1 : static_cast<std::size_t>(job_or_last);
    npu_joined_ = std::max(npu_joined_, job + 1);
    // jobs count per chunk: the joined jobs' words read zero again before the next chunk reuses them (only the joined
    // ones: a later job may still be waiting at its gate)
    for (; npu_cleared_ <= job; ++npu_cleared_)
      for (const std::uint32_t w : {flagged_[npu_cleared_].words.ready, flagged_[npu_cleared_].words.done})
        graph_->AtomicStore(FlagWord(w), 0, HRX_ATOMIC_FLAG_RELEASE | HRX_ATOMIC_FLAG_SYSTEM_SCOPE, {}, {});
  }

  // The NPU attention's flag words: layer ai's ready and done at NaReady(ai), + 1 (word NpuSplit::kFlagStatus: status).
  static constexpr std::uint32_t kNaWord0 = 16;
  static std::uint32_t NaReady(std::uint32_t ai) { return kNaWord0 + 2 * ai; }
  hrx_buffer_ref_t NaWord(std::uint32_t w) const { return {na_.flags->handle, std::size_t{w} * 4, 4}; }
  // Chunk ci's NPU attention inputs for layer ai (gen_npu_attn_encode's layer forms, npua{k,v,q}_c<ci>.hal): the KV
  // heads' K / V records of keys [32, k_lo) (k_lo = 512 floor(start / 512): nb = 16 (k_lo / 512 + 1) bodies) and every
  // call's payloads from q, the attention's own; then the layer's ready word.
  void RunNpuAttnEncode(std::uint32_t ci, std::uint32_t ai, hrx_buffer_ref_t q, hrx_buffer_ref_t kpool,
                        hrx_buffer_ref_t vtpool, const std::string& pre) {
    const std::uint32_t nb = 16 * (ci * B_ / 512 + 1), calls = 3 * na_kvn_ * (B_ / 256);
    Dispatch(ChunkExe("npuak", ci), "yah_npuattn_k", na_kvn_ * (nb + 1), 1, 1, 128, 1, 1,
             {kpool, ptab_ref_, Ref(*na_.ks), Ref(*na_.kn)}, 0b1100);
    Dispatch(ChunkExe("npuav", ci), "yah_npuattn_v", na_kvn_ * nb, 1, 1, 128, 1, 1, {vtpool, ptab_ref_, Ref(*na_.vs)},
             0b100);
    Dispatch(
        ChunkExe("npuaq", ci), "yah_npuattn_q", calls * 33, 1, 1, 64, 1, 1,
        {q, TRef(*Find(pre + "attn_q_norm.weight")), {eps_->handle, 0, 4}, Ref(*na_.kn), Ref(*na_.kq), Ref(*na_.vq)},
        0b110000);
    graph_->AtomicStore(NaWord(NaReady(ai)), na_seq_, HRX_ATOMIC_FLAG_RELEASE | HRX_ATOMIC_FLAG_SYSTEM_SCOPE,
                        {Ref(*na_.ks), Ref(*na_.vs), Ref(*na_.kq), Ref(*na_.vq)}, {});
    na_jobs_.push_back({ai, nb, flagged_.size()});
    if (npu_on_) npu_->AfterForeignCommand();   // the GEMM split's next job follows this one on the array
  }

  // af: the afrag form, input normt_ (fragment-major)
  void RunKstore(const std::string& wname, const LoomBuffer& out, bool af = false) {
    const auto* t = Find(wname);
    Fmt f{};
    const std::string base = RSegHal(GemmHal("gemm_kstore", *t, &f), *t);
    const std::string hal = af ? AfHal(base) : PickGemm(base);
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*t, f);
    if (weights_.RSegOf(*t)) b[0] = RSegRef(*t);
    const std::size_t first = b.size();
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{af ? normt_ : scratch_, wstage_, ostage_})
      b.push_back(Ref(*x));
    b.push_back(Ref(out));
    const std::uint32_t tt = Trim(b, first, {std::size_t(t->dims[0]) * 2, 0, 0, std::size_t(t->dims[1]) * 4}, g);
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name)).c_str(), MTiles(*t) / g.rowgrp,
             tt, 1, 32, 1, 1, b, GemmWrites(b, f, {&out}));
  }
  // qsplit: the q head-norm weight beside a layer's attn_q.weight
  static std::string QNormName(const std::string& wq) {
    const std::string suf = "attn_q.weight";
    if (wq.size() < suf.size() || wq.compare(wq.size() - suf.size(), suf.size(), suf) != 0)
      throw LoomError("qsplit: not an attn_q weight: " + wq);
    return wq.substr(0, wq.size() - suf.size()) + "attn_q_norm.weight";
  }
  // A segmented GEMM (emit_prefill_pp SEG, gen_segcat): one dispatch over the tiled norm output in which each part (the
  // large GEMM first) has its own weights, format and outputs; "seg_<kind>_<fmt>_<mt>_<kb>__..". dispatch.txt gives
  // the grid x (all parts' row groups) in the rowgrp column. Returns false if the set lacks it or for a trimmed chunk.
  struct SegPart {
    const char* kind;
    std::string tensor;
    std::vector<const LoomBuffer*> outs;
  };
  bool RunSeg(const std::vector<SegPart>& parts) {
    if (trim_row_ >= 0 || !AfChunk()) return false;
    std::string hal = "seg";
    std::vector<std::pair<const core::TensorInfo*, Fmt>> ts;
    for (const SegPart& p : parts) {
      const auto* t = Find(p.tensor);
      Fmt f{};
      const std::string h = GemmHal(p.kind, *t, &f);   // "gemm_<kind>_<fmt>_<mt>_<kb>.hal"
      hal += (ts.empty() ? "_" : "__") + h.substr(std::strlen("gemm_"), h.size() - std::strlen("gemm_") - 4);
      ts.push_back({t, f});
    }
    hal += ".hal";
    const auto g = geom_.find(hal);
    if (g == geom_.end()) return false;
    std::vector<hrx_buffer_ref_t> b;
    std::uint64_t writes = 0;
    std::size_t ostage = 0;   // the one ostage binding (part 0's): Q2_K writes it (GemmWrites)
    for (std::size_t i = 0; i < parts.size(); ++i) {
      const auto w = GemmWeights(*ts[i].first, ts[i].second);
      b.insert(b.end(), w.begin(), w.end());
      if (i == 0) {
        for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{normt_, wstage_, ostage_}) b.push_back(Ref(*x));
        ostage = b.size() - 1;
      }
      for (const LoomBuffer* o : parts[i].outs) {
        writes |= std::uint64_t{1} << b.size();
        b.push_back(Ref(*o));
      }
      if (qsplit_ && std::string(parts[i].kind) == "gemm_kqg") b.push_back(TRef(*Find(QNormName(parts[i].tensor))));
      if (std::string(ts[i].second.name) == "q2k") writes |= std::uint64_t{1} << ostage;
    }
    Dispatch(Exe(hal), "yah_seg", g->second.rowgrp, TokenTiles(g->second), 1, 512, 1, 1, b, writes);
    return true;
  }

  // ssm_alpha and ssm_beta: in one kstore when the set has "gemm_kst2_<fmt>_<mt>_<kb>.hal" (emit_prefill_pp KST2) and the
  // tensors are of one format and stored back to back (one weight binding spans both; rows of beta go to beta_), else
  // two kstores.
  // The KST2 HAL of layer pre's alpha / beta ("" if none); tiled: its fragment-major-input form (".at.hal", input
  // normt_; YAH_KST2AT=0 turns it off).
  std::string AlphaBetaHal(const std::string& pre, bool tiled, Fmt* fo = nullptr) const {
    static const bool at_off = [] { const char* v = std::getenv("YAH_KST2AT"); return v && std::string(v) == "0"; }();
    const auto* ta = Find(pre + "ssm_alpha.weight");
    const auto* tb = Find(pre + "ssm_beta.weight");
    Fmt f{};
    std::string hal = ta->type != tb->type || ta->dims != tb->dims || tb->offset != ta->offset + ta->bytes
                          ? "" : GemmHal("gemm_kst2", *ta, &f);
    if (fo) *fo = f;
    if (tiled && !hal.empty()) hal = at_off ? "" : hal.substr(0, hal.size() - 4) + ".at.hal";
    return hal.empty() || !geom_.count(hal) ? "" : hal;
  }
  void RunAlphaBeta(const std::string& pre, bool tiled = false) {
    const auto* ta = Find(pre + "ssm_alpha.weight");
    const auto* tb = Find(pre + "ssm_beta.weight");
    Fmt f{};
    const std::string hal = AlphaBetaHal(pre, tiled, &f);
    if (tiled && hal.empty()) throw LoomError("prefill: " + pre + " alpha / beta: no fragment-major-input KST2");
    if (hal.empty()) {
      RunKstore(pre + "ssm_alpha.weight", *alpha_);
      RunKstore(pre + "ssm_beta.weight", *beta_);
      return;
    }
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*ta, f);
    b[0].length = static_cast<std::size_t>(ta->bytes + tb->bytes);
    const std::size_t first = b.size();
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{tiled ? normt_ : scratch_, wstage_, ostage_, alpha_, beta_})
      b.push_back(Ref(*x));
    const std::size_t rb = std::size_t(ta->dims[1]) * 4;
    const std::uint32_t tt = Trim(b, first, {std::size_t(ta->dims[0]) * 2, 0, 0, rb, rb}, g);
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name)).c_str(), 2 * MTiles(*ta) / g.rowgrp, tt, 1, 32, 1, 1, b,
             GemmWrites(b, f, {alpha_, beta_}));
  }

  // The attention q projection with the q / gate unpack fused in (rows = heads x [256 q | 256 gate]).
  // Returns false if the set has no such HAL; the caller then runs kstore + yah_unpack_qg.
  // af: the afrag form, input normt_ (fragment-major)
  bool RunKqg(const std::string& wname, bool af = false) {
    const auto* t = Find(wname);
    Fmt f{};
    if (!FmtOf(static_cast<std::uint32_t>(t->type), &f)) return false;
    std::string hal = GemmHal("gemm_kqg", *t, &f);
    if (!geom_.count(hal)) return false;
    hal = af ? AfHal(hal) : PickGemm(hal);
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*t, f);
    for (const LoomBuffer* x :
         std::initializer_list<const LoomBuffer*>{af ? normt_ : scratch_, wstage_, ostage_, q_, gate_})
      b.push_back(Ref(*x));
    if (qsplit_) b.push_back(TRef(*Find(QNormName(wname))));
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name) + "_kqg").c_str(), MTiles(*t) / g.rowgrp, TokenTiles(g), 1, 32, 1, 1, b, GemmWrites(b, f, {q_, gate_}));
    return true;
  }
  // af: the afrag form, input normt_; tout: its output fragment-major (for an afrag down projection)
  void RunSwiglu(const std::string& wname, bool af = false, bool tout = false) {
    const auto* t = Find(wname);
    Fmt f{};
    const std::string base = RSegHal(GemmHal("gemm_swiglu", *t, &f), *t);
    const std::string hal = af ? AfHal(base, tout ? ".af.to.hal" : ".af.hal") : PickGemm(base);
    if (hal.empty()) throw LoomError(base + ": the set lacks its afrag form");
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*t, f);
    if (weights_.RSegOf(*t)) b[0] = RSegRef(*t);
    const std::size_t first = b.size();
    for (const LoomBuffer* x :
         std::initializer_list<const LoomBuffer*>{af ? normt_ : scratch_, gateffn_, uwstage_, ostage_, ffnup_})
      b.push_back(Ref(*x));
    const std::size_t M = t->dims[1];
    const std::uint32_t tt = Trim(b, first, {std::size_t(t->dims[0]) * 2, M * 4, 0, 0, M * 2}, g);
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name) + "_swiglu").c_str(),
             MTiles(*t) / g.rowgrp, tt, 1, 32, 1, 1, b, GemmWrites(b, f, {ffnup_}));
  }
  // hidden += W input. The fused kres GEMM writes hidden + W input into hidden2; otherwise kStore writes W input
  // into partial and yah_residual_1d adds it. Either way the two hidden buffers swap.
  // af: the afrag form of the fused kres GEMM (input fragment-major)
  void RunResidual(const std::string& wname, const LoomBuffer& input, bool af = false) {
    FlushOut();
    const auto* t = Find(wname);
    Fmt f{};
    const std::string fused0 = GemmHal("gemm_kres", *t, &f);
    if (geom_.count(fused0)) {
      const std::string fused = af ? AfHal(fused0) : PickGemm(fused0);
      const Geom g = GeomOf(fused);
      auto b = GemmWeights(*t, f);
      const std::size_t first = b.size();
      for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{&input, hidden_, wstage_, ostage_, hidden2_}) b.push_back(Ref(*x));
      const std::size_t M = t->dims[1];
      const std::uint32_t tt = Trim(b, first, {std::size_t(t->dims[0]) * 2, M * 4, 0, 0, M * 4}, g);
      // the persistent kres (".af.p.hal": each workgroup runs every token tile of its row block, grid y = 1) unless trimmed
      const std::string pers = af ? AfHal(fused0, ".af.p.hal") : "";
      const bool use_p = !pers.empty() && tt == g.tt;
      // NPU split (site "out": ssm_out / attn_output, not ffn_down) of untrimmed chunks
      const char* site = "out";
      const bool down = wname.find("ffn_down") != std::string::npos;
      const std::uint32_t nn = tt == g.tt && !down && NpuOk(wname) ? NpuRows(site, wname) : 0;
      // The persistent kres runs one workgroup per row block (one per CU unsplit), so the split uses the tiled form.
      // Per-layer rows ("npusplit_out@<L>"): the "_n<rows>" forms (RowsHal). all: the NPU takes every row (no GPU
      // kres; the GPU runs only the K remainder, the unpack adds the residual).
      const bool np4w = nn && Np4(wname), all = nn && nn == M;
      const std::string sh = nn && !all ? RowsHal(NpuHal(fused), site, nn, np4w) : "";
      if (!sh.empty() || (all && geom_.count(RowsHal("npurem_" + std::string(site) + "_" + f.name + ".hal", site, nn, np4w)))) {
        const std::uint32_t ro = SwSiteR(site, BlkOf(wname), nn);
        // (PanelQ) after the FFN block's H; a DeltaNet layer's after its qkv and gate C (dn_c_end_), the later of both
        const std::size_t dn_end = std::exchange(dn_c_end_, 0);
        std::size_t c_off = std::max(OutCAfterH() && !std::strcmp(site, "out") ? FfnBlkDown() : 0, dn_end);
        std::vector<std::uint32_t> calls;
        NpuView c{};
        std::size_t ow = 0;   // (swap form) the panel's W bytes
        if (ro) {   // (swap-form site) the GPU decodes the NPU rows' k 0 .. 5120 to A 0, the input's k 0 .. 5120 to the panel
          SwSiteDecode(wname, nn, 0);
          ow = SwSitePanel(af ? "npu_swpanel_o_t.hal" : "npu_swpanel_o.hal", &input);
          c = SwSiteCalls(ro, 0, c_off, calls);
        } else {
          const auto a = NpuEncode(site, input, af, NpuPk(wname));
          const auto w = NpuRawRows(site, wname, nn);
          c = NpuCalls(site, a, w, c_off, calls, wname);
        }
        // (gc) part b of the DeltaNet postnorm, deferred past this job's release, still reads the gate's C
        if (gc_read_.length && c.offset < gc_read_.offset + gc_read_.length && gc_read_.offset < c.offset + c.length)
          throw LoomError("prefill: the out job's C overlaps the gate's C that the postnorm reads (" + wname + ")");
        if (ro)
          NpuEnqueue(std::move(calls), "swout", {{ARef(0, NpuBytesOf("npubytes_swsite_" + std::to_string(ro)).a), WRef(0, ow)},
                                                 CRef(c.offset, c.length)});
        else
          NpuEnqueue(std::move(calls), site, {{Ref(npu_->A())}, CView(c)});   // (its own C: not after other C readers)
        RunAfterRelease();
        const Geom gs = all ? Geom{} : GeomOf(sh);
        const std::string kname = std::string("yah_ffn_gemm_") + f.name + "_kres";
        const std::uint32_t blocks = all ? 0 : (MTiles(*t) - nn / 16) / gs.rowgrp;
        // The persistent kres (one workgroup per row block) over all but the last token tile, the tiled kres on that
        // tile beside it (disjoint token ranges: the graph runs them together on the CUs the row blocks leave idle).
        const std::string p3 = fused0.substr(0, fused0.size() - 4) + ".af.p3.npu.hal";   // NPU split only
        // (11-19: graph dispatches do not overlap here, so the two run one after the other and the tiled kres over all
        // token tiles, blocks x tt workgroups, is faster: IQ4_XS 1582 -> 1288 us, Q4_K / Q5_K equal; YAH_NPU_OUTP3=1:
        // the persistent part + the tiled last tile)
        static const bool outp3 = std::getenv("YAH_NPU_OUTP3") != nullptr;
        // the rest of K for the NPU's rows (its last nn weight rows), beside the GPU's rows: f32 [tokens][nn]
        const std::string rh = RowsHal("npurem_" + std::string(site) + "_" + f.name + ".hal", site, nn, np4w);
        const std::size_t row_bytes = static_cast<std::size_t>(t->bytes) / M;
        auto rb = GemmWeights(*t, f);
        rb[0].offset += (M - nn) * row_bytes, rb[0].length = std::size_t{nn} * row_bytes;
        for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{&input, wstage_, ostage_}) rb.push_back(Ref(*x));
        rb.push_back({npurem_->handle, 0, std::size_t{B_} * nn * 4});
        // Both as one dispatch when the set has "npuseg_out_<fmt>.hal" (emit_prefill_pp outseg; gen_segcat linear: the
        // kres workgroups first, the remainder's fill its last round); bindings the kres's, then the remainder's weights
        // and output. The same values; 11-21 pp2048: out site GPU kernels 113.9 -> 106.9 ms. YAH_NPU_OUTSEG=0: two.
        static const bool outseg = [] { const char* e = std::getenv("YAH_NPU_OUTSEG"); return !e || std::strcmp(e, "0"); }();
        const std::string segh = RowsHal("npuseg_out_" + std::string(f.name) + ".hal", site, nn, np4w);
        const auto sg = outseg && !outp3 && !all ? geom_.find(segh) : geom_.end();
        if (sg != geom_.end()) {
          auto sb = b;
          sb.insert(sb.end(), rb.begin(), rb.end() - 4);   // (the remainder's weights: all but input, wstage, ostage, out)
          sb.push_back(rb.back());
          Dispatch(Exe(segh), "yah_seg", sg->second.rowgrp, 1, 1, 512, 1, 1, sb,
                   GemmWrites(b, f, {hidden2_}) | std::uint64_t{1} << (sb.size() - 1));
        } else if (all) {
          // (no GPU rows: the K remainder below is the GPU's whole share)
        } else if (outp3 && af && geom_.count(p3) && tt == g.tt && gs.tt == g.tt) {
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
          // its own token tiles (emit_prefill_pp.split_tile: 256 tokens; the split runs untrimmed chunks only)
          Dispatch(Exe(sh), kname.c_str(), blocks, gs.tt, 1, 32, 1, 1, b, GemmWrites(b, f, {hidden2_}));
        }
        if (sg == geom_.end()) {
          const Geom& gr = geom_.at(rh);
          Dispatch(Exe(rh), ("yah_ffn_gemm_" + std::string(f.name)).c_str(), nn / 16 / gr.rowgrp, gr.tt, 1, 32, 1, 1, rb,
                   std::uint64_t{1} << (rb.size() - 1));
        }
        NpuJoin();
        // the sum waits for the FFN block's norm, which fuses it ("npu_onorm[_4].hal", RunNorm; YAH_NPU_NOONORM=1 at
        // emit: the unpack here)
        out_pending_ = c, out_nn_ = nn, out_np4_ = Np4(wname);
        if (!geom_.count(OnormHal())) FlushOut();
        return;
      }
      RunAfterRelease();
      Dispatch(Exe(use_p ? pers : fused), (std::string("yah_ffn_gemm_") + f.name + "_kres").c_str(),
               MTiles(*t) / g.rowgrp, use_p ? 1 : tt, 1, 32, 1, 1, b, GemmWrites(b, f, {hidden2_}));
      std::swap(hidden_, hidden2_);
      return;
    }
    RunAfterRelease();
    const std::string base = GemmHal("gemm_kstore", *t, &f);
    const std::string hal = PickGemm(base);
    const Geom g = GeomOf(hal);
    auto b = GemmWeights(*t, f);
    for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{&input, wstage_, ostage_, partial_}) b.push_back(Ref(*x));
    Dispatch(Exe(hal), ("yah_ffn_gemm_" + std::string(f.name)).c_str(), MTiles(*t) / g.rowgrp,
             TokenTiles(g), 1, 32, 1, 1, b, GemmWrites(b, f, {partial_}));
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
    const std::uint32_t nq = NpuOk(pre + "attn_q.weight") ? NpuRows("q", pre + "attn_q.weight") : 0;   // (a format without a decoder: GPU)
    const std::string qs = nq ? RowsHal(NpuHal(KqgHal(pre + "attn_q.weight", q_af)), "q", nq, Np4(pre + "attn_q.weight")) : "";
    bool qg_fused = !qs.empty();
    if (gsig_ && qg_fused) throw LoomError("the kqg GEMM's sigmoid gate (gsig) needs the GPU kqg GEMM, not the NPU's");
    if (qsplit_ && qg_fused) throw LoomError("the kqg GEMM's split q norm (qsplit) needs the GPU kqg GEMM, not the NPU's");
    if (qg_fused) {
      // the NPU's heads of q (whole heads of 512 rows [256 q | 256 gate]) beside the GPU's heads and k / v;
      // the norm wrote the row-major input unless q, k and v are all afrag
      const bool tiled = q_af && k_af && v_af;
      std::size_t c_off = 0;
      std::vector<std::uint32_t> calls;
      NpuView c{};
      if (const std::uint32_t rq = SwSiteR("q", l, nq)) {
        // (swap-form site) the GPU decodes the NPU's rows to the plain stream at A 0, the norm's tiled output to the panel
        if (!tiled) throw LoomError("prefill: " + pre + " a swap-form q site needs the tiled norm output");
        const NpuBytes bq = NpuBytesOf("npubytes_swsite_" + std::to_string(rq));
        SwSiteDecode(pre + "attn_q.weight", nq, 0);
        const std::size_t pw = SwSitePanel();
        c = SwSiteCalls(rq, 0, c_off, calls);
        NpuEnqueue(std::move(calls), "swq", {{ARef(0, bq.a), WRef(0, pw)}, CRef(c.offset, c.length)});
      } else {
        const auto a = NpuEncode("q", tiled ? *normt_ : *scratch_, tiled, NpuPk(pre + "attn_q.weight"));
        const auto w = NpuRawRows("q", pre + "attn_q.weight", nq);
        c = NpuCalls("q", a, w, c_off, calls, pre + "attn_q.weight");
        NpuEnqueue(std::move(calls), "q");
      }
      RunPendingNorm();
      RunKqgSplit(pre + "attn_q.weight", q_af, qs, nq);
      RunKstore(pre + "attn_k.weight", *kbuf_, k_af);
      RunKstore(pre + "attn_v.weight", *vbuf_, v_af);
      NpuJoin();
      // qc: chunk 0's attention reads the NPU heads' q / gate from C ("wmma[_t]_qc<sfx>.hal", gen_attn_fa QC), so the
      // unpack goes (the next NPU job that writes C, the o-projection's, waits for its ready word after the attention)
      const std::string un = RowsHal(Np4Hal("npu_unpack_q", pre + "attn_q.weight"), "q", nq, Np4(pre + "attn_q.weight"));
      const std::string qch = std::string(o_af ? "wmma_t_qc" : "wmma_qc") + un.substr(12, un.size() - 16) + ".hal";
      if (ci == 0 && geom_.count(qch) && c.length >= std::size_t{B_} * nq * 2)   // (one replay group: bf16 per element; the .fp cbuf footprint)
        qc_hal_ = qch, qc_c_ = c;
      else
        NpuUnpack(un, c, {Ref(*q_), Ref(*gate_)}, 6);
    } else {
      // seg: k and v ride in the kqg GEMM's dispatch (f16 sets)
      const bool seg = q_af && k_af && v_af && RunSeg({{"gemm_kqg", pre + "attn_q.weight", {q_, gate_}},
                                                       {"gemm_kstore", pre + "attn_k.weight", {kbuf_}},
                                                       {"gemm_kstore", pre + "attn_v.weight", {vbuf_}}});
      qg_fused = seg || RunKqg(pre + "attn_q.weight", q_af);
      if (gsig_ && !qg_fused) throw LoomError("the sigmoid gate (gsig) needs the kqg GEMM");
      if (qsplit_ && !qg_fused) throw LoomError("the split q norm (qsplit) needs the kqg GEMM");
      if (!qg_fused) RunKstore(pre + "attn_q.weight", *qkv_);
      if (!seg) {
        RunKstore(pre + "attn_k.weight", *kbuf_, k_af);
        RunKstore(pre + "attn_v.weight", *vbuf_, v_af);
      }
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
      Dispatch(ChunkExe("rope", ci), "yah_fused_qk_rope_batched", qnorm_ ? kKvHeads : kHeads + kKvHeads, B_, 1, 256, 1,
               1, b);
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
    if (tail_ == kNoRows) {  // K / V are written; nothing reads this layer's output
      qc_hal_.clear();
      return;
    }
    if (kdeq_) {  // the context so far, this chunk included, as f16 pools for the fp16 attention
      const std::uint32_t rows = (ci + 1) * B_;
      const char* kd = attn_kq4_ ? "kdeq4" : "kdeq8";
      const char* vd = attn_kq4_ ? "vdeq4" : "vdeq8";
      Dispatch(ChunkExe(kd, ci), attn_kq4_ ? "yah_kdeq4" : "yah_kdeq8", (rows * 64 + 255) / 256, 1, 1, 256, 1, 1,
               kscale_ ? std::vector<hrx_buffer_ref_t>{{kq8buf_->handle, q8off, kq_bytes_}, {ksbuf_->handle, ksoff, ks_bytes_},
                                                       ptab_ref_, Ref(*kdeq_), Ref(*kscale_)}
                       : std::vector<hrx_buffer_ref_t>{{kq8buf_->handle, q8off, kq_bytes_}, {ksbuf_->handle, ksoff, ks_bytes_},
                                                       ptab_ref_, Ref(*kdeq_)});
      Dispatch(ChunkExe(vd, ci), attn_kq4_ ? "yah_vdeq4" : "yah_vdeq8", 4 * ((rows + 15) / 16), 1, 1, 256, 1, 1,
               {{vqbuf_->handle, vqoff, vq_bytes_}, {vqsbuf_->handle, vqsoff, vqs_bytes_}, ptab_ref_, Ref(*vdeq_)});
    }
    const bool qrot = qr16_ && ci >= qrot_first_;
    if (qrot)
      Dispatch(Exe("qrot.hal"), "yah_qrot", (B_ * 96 + 255) / 256, 1, 1, 256, 1, 1, {Ref(*q16_), Ref(*qr16_)});
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
      const bool deqb = kdeq_ != nullptr;
      if (deqb) b = {b[0], b[1], Ref(*kdeq_), Ref(*vdeq_), b[4], b[5]};  // the fp16 kernel's bindings
      if (attn_kq8_ && !deqb) b.push_back({ksbuf_->handle, ksoff, ks_bytes_});
      if (attn_vqt_ && !deqb) b.push_back({vqsbuf_->handle, vqsoff, vqs_bytes_});
      if (kv_paged_) b.push_back(ptab_ref_);
      if (qnorm_) {
        b.push_back(TRef(*Find(pre + "attn_q_norm.weight")));
        b.push_back({eps_->handle, 0, 4});
      }
      if (deqb && kscale_) b.push_back(Ref(*kscale_));
      const std::string ah = std::string(o_af ? "wmma_t" : "wmma") + (ci ? "_c" + std::to_string(ci) : std::string()) + ".hal";
      const auto ag = geom_.find(ah);
      const std::uint32_t tpw = ag != geom_.end() ? ag->second.tokens : attn_tpw_,
                          hpw = ag != geom_.end() ? ag->second.rowgrp : attn_hpw_;
      // a persistent attention HAL (dispatch.txt "persist_<hal> <slots>") runs <slots> workgroups over all the units
      const std::string ahx = qc_hal_.empty() ? ah : qc_hal_;
      const auto pg = geom_.find("persist_" + ahx);
      if (pg != geom_.end() && pg->second.tt) {
        if (ai >= kPqueueSlots) throw LoomError("prefill: more attention layers than queue counters");
        b.push_back({pqueue_->handle, std::size_t{ai} * 2048, 1056});
      }
      if (!qc_hal_.empty()) {   // (gen_attn_fa QC: %cbuf after the others)
        if (pg == geom_.end() || !pg->second.tt || geom_.count("ksplit_" + ahx) || (npu_attn_ && ci >= na_from_))
          throw LoomError("prefill: " + ahx + " needs the plain persistent chunk-0 attention");
        b.push_back(CRef(qc_c_.offset, qc_c_.length));
        qc_hal_.clear();
      }
      // the key-split tail: kernel 1 (this HAL) stores the tail pieces' partials, the merge "wmmak<rest>" writes their output
      const bool ks = geom_.count("ksplit_" + ahx) != 0;
      if (ks) {
        if (pg == geom_.end() || !pg->second.tt || !ksplit_)
          throw LoomError("prefill: ksplit_" + ahx + " without its queue");
        b.push_back({ksplit_->handle, 0, ksplit_bytes_});
      }
      // the NPU's piece (EnableNpuAttn): its encoders, then the ready word, so the NPU runs beside kernel 1
      const bool na = npu_attn_ && ci >= na_from_ && !warming_ && !npu_planning_;
      if (na) {
        if (!ks) throw LoomError("prefill: the NPU attention needs the key-split pair");
        RunNpuAttnEncode(ci, ai, b[0], b[2], b[3], pre);  // the attention's own K / V^T pools (kv8: the f16 copies)
      }
      Dispatch(Exe(ahx), "yah_attn_wmma", pg != geom_.end() ? pg->second.tokens : (B_ + tpw - 1) / tpw,
               pg != geom_.end() ? 1 : kHeads / hpw, 1, 256, 1, 1, b);
      if (ks) {
        const std::string mh = "wmmak" + ahx.substr(4);
        const auto mg = geom_.find("persist_" + mh);
        if (mg == geom_.end() || !mg->second.tt) throw LoomError("prefill: no persist_" + mh);
        if (na) {  // the wait stands for the NPU's reads of the records and payloads and its writes of the outputs
          const std::vector<hrx_buffer_ref_t> side = {Ref(*na_.ks), Ref(*na_.vs), Ref(*na_.kq), Ref(*na_.vq),
                                                      Ref(*na_.out)};
          Dispatch(Exe("npu_flag_wait.hal"), "yah_npu_flag_wait", 1, 1, 1, 32, 1, 1,
                   {NaWord(NaReady(ai) + 1), NaWord(NaReady(ai)), NaWord(NpuSplit::kFlagStatus)}, 4, &side);
          b.push_back(Ref(*na_.out));
          b.push_back(Ref(*na_.vq));
        }
        Dispatch(Exe(mh), "yah_attn_wmma", mg->second.tokens, 1, 1, 256, 1, 1, b);
      }
    }
    trim_row_ = tail_;
    RunResidual(pre + "attn_output.weight", *scratch_, o_af);
    trim_row_ = kAllRows;
  }


  void RunDeltaNet(std::uint32_t l, const std::string& pre, bool qkv_af, bool gate_af, bool ab_t = false) {
    const std::uint32_t si = l - l / cfg_.full_attention_interval;
    // alpha / beta first: they read the norm's row-major copy (21 MB), which the qkv / gate GEMMs would evict from the
    // last-level cache (0.18 vs ~0.39 M cycles each)
    const bool nok = NpuOk(pre + "attn_qkv.weight") && NpuOk(pre + "attn_gate.weight");
    const std::uint32_t nq = nok ? NpuRows("qkv", pre + "attn_qkv.weight") : 0,
                        ng = nok ? NpuRows("gate", pre + "attn_gate.weight") : 0;
    const std::string wqkv = pre + "attn_qkv.weight";
    const std::string qkv_split = nq ? QkvHal2(NpuHal(KstoreHal(pre + "attn_qkv.weight", qkv_af)), pre + "attn_qkv.weight", nq) : "";
    const std::string gate_split = ng ? NpuHal(KstoreHal(pre + "attn_gate.weight", gate_af)) : "";
    // conv_c: the conv reads the NPU's qkv columns straight from C ("convkq_c.hal"); the unpack writes only the conv ring
    NpuView conv_c{}, cg{};
    bool gate_late = false;
    std::int64_t gate_job = -1;   // the NPU's gate rows: joined before the postnorm, the first to read them
    if (!qkv_split.empty() && !gate_split.empty()) {
      // the NPU's rows of qkv and gate beside alpha / beta and the GPU's rows: a qkv job, then a gate job (the same
      // activations), so the GPU's conv and DeltaNet run while the NPU computes gate
      // decoder-column sets: each matrix's activations in its decoder's k-block order
      std::size_t c_off = 0;
      std::vector<std::uint32_t> calls;
      NpuView cq{};
      std::uint32_t qkv_job = 0;
      const std::uint32_t rq = SwSiteR("qkv", l, nq), rg = SwSiteR("gate", l, ng);
      if ((rq == 0) != (rg == 0)) throw LoomError("prefill: " + pre + " has one of qkv / gate in swap form");
      if (rq) {
        // (swap-form sites) the GPU decodes the NPU's rows to the plain bfp16 stream (qkv's at A 0, gate's at
        // SwSiteGateA), the norm's tiled output to the token panel in W; the image reads both (its job's side)
        if (!ab_t) throw LoomError("prefill: " + pre + " swap-form sites need the tiled norm output");
        const std::size_t ag = SwSiteGateA();
        const NpuBytes bq = NpuBytesOf("npubytes_swsite_" + std::to_string(rq)),
                       bg = NpuBytesOf("npubytes_swsite_" + std::to_string(rg));
        SwSiteDecode(pre + "attn_qkv.weight", nq, 0);
        const std::size_t pw = SwSitePanel();
        cq = SwSiteCalls(rq, 0, c_off, calls);
        qkv_job = NpuEnqueue(std::exchange(calls, {}), "swqkv", {{ARef(0, bq.a), WRef(0, pw)}, CRef(cq.offset, cq.length)});
        SwSiteDecode(pre + "attn_gate.weight", ng, ag);
        cg = SwSiteCalls(rg, ag, c_off, calls);
        gate_job = NpuEnqueue(std::move(calls), "swgate", {{ARef(ag, bg.a), WRef(0, pw)}, CRef(cg.offset, cg.length)});
      } else {
        const bool kq = NpuPk(pre + "attn_qkv.weight"), kg = NpuPk(pre + "attn_gate.weight");
        std::vector<NpuView> a4, ak;
        if (!kq || !kg) a4 = NpuEncode("qkv", ab_t ? *normt_ : *scratch_, ab_t);   // (ab_t: no row-major copy)
        if (kq || kg) ak = NpuEncode("qkv", ab_t ? *normt_ : *scratch_, ab_t, true);
        const auto wq = NpuRawRows("qkv", pre + "attn_qkv.weight", nq);
        const auto wg = NpuRawRows("gate", pre + "attn_gate.weight", ng);
        cq = NpuCalls("qkv", kq ? ak : a4, wq, c_off, calls, pre + "attn_qkv.weight");
        qkv_job = NpuEnqueue(std::exchange(calls, {}), "qkv");
        cg = NpuCalls("gate", kg ? ak : a4, wg, c_off, calls, pre + "attn_gate.weight");
        gate_job = NpuEnqueue(std::move(calls), "gate");
      }
      dn_c_end_ = c_off;   // (the out job's C goes after qkv's and gate's: dnc parts read qkv's, postnorm b gate's)
      RunPendingNorm();
      RunAlphaBeta(pre, ab_t);
      RunKstoreSplit(pre + "attn_qkv.weight", *qkv_, qkv_af, qkv_split, nq);
      RunKstoreSplit(pre + "attn_gate.weight", *gate_, gate_af, gate_split, ng);
      NpuJoin(qkv_job);
      if (geom_.count(QkvHal("convkq_c", wqkv, nq))) {
        conv_c = cq;
        NpuUnpack(QkvHal("npu_unpack_qkv_tail", wqkv, nq), cq, {Ref(*qkv_)}, 2);
      } else {
        NpuUnpack(QkvHal("npu_unpack_qkv", wqkv, nq), cq, {Ref(*qkv_)}, 2);
      }
    } else {
      if (ab_t) throw LoomError("prefill: " + pre + " tiled alpha / beta input without the NPU split");
      // seg: alpha and beta ride in the qkv GEMM's dispatch (f16 sets)
      const bool seg = qkv_af && RunSeg({{"gemm_kstore", pre + "attn_qkv.weight", {qkv_}},
                                         {"gemm_kstore", pre + "ssm_alpha.weight", {alpha_}},
                                         {"gemm_kstore", pre + "ssm_beta.weight", {beta_}}});
      if (!seg) RunAlphaBeta(pre);
      // dnvconv: the gate GEMM goes after DeltaNet's second part (below), beside it, and the conv no longer waits for it
      gate_late = geom_.count("dnvconv") != 0;
      if (!seg) RunKstore(pre + "attn_qkv.weight", *qkv_, qkv_af);
      if (!gate_late) RunKstore(pre + "attn_gate.weight", *gate_, gate_af);
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
    // nothing reads). It waits for both DeltaNet parts (they read the ring), so it is queued after the postnorm parts.
    const auto ring = [&] {
      Dispatch(Exe("prepab.hal"), "yah_deltanet_prep_ab", (kQkv + 255u) / 256u, 1, 1, 256, 1, 1,
               {Ref(*alpha_), Ref(*beta_), TRef(*Find(pre + "ssm_a")), TRef(*Find(pre + "ssm_dt.bias")), Ref(*qkv_), cs,
                Ref(*ab_), Ref(*valid_)},
               0b1100000);
    };
    if (const std::string da = QkvHal("dnc_a", wqkv, nq), db = QkvHal("dnc_b", wqkv, nq);
        conv_c.length && geom_.count(da) && geom_.count(db)) {
      // dispatch.txt "dnc_<a|b>.hal" (emit_prefill_pp YAH_DNSSC): DeltaNet's parts on the state-stationary kernel with
      // the v conv and alpha / beta folded in, the NPU's v channels read from C; the conv only takes the q / k heads,
      // and prep_ab only advances the conv ring after both parts (as dnvconv)
      const Geom& gq = geom_.at("convqk.hal");
      Dispatch(Exe("convqk.hal"), "yah_ssm_conv_kq", gq.tokens, B_ / conv_tb, 1, 256, 1, 1,
               {Ref(*qkv_), TRef(*Find(pre + "ssm_conv1d.weight")), cs, Ref(*conv_out_), Ref(*kqbuf_)});
      const hrx_buffer_ref_t cref = CRef(conv_c.offset, conv_c.length);
      const std::vector<hrx_buffer_ref_t> in{Ref(*conv_out_), Ref(*qkv_), TRef(*Find(pre + "ssm_conv1d.weight")), cs,
                                             Ref(*alpha_), Ref(*beta_), TRef(*Find(pre + "ssm_a")),
                                             TRef(*Find(pre + "ssm_dt.bias")), Ref(*valid_)};
      auto ba = in, bb = in;
      ba.insert(ba.end(), {st, Ref(*raw_), cref});
      bb.insert(bb.end(), {st, Ref(*raw2_), cref});
      const Geom& ga = geom_.at(da);
      const Geom& gb = geom_.at(db);
      if (ga.tokens != dnsplit_a_ || gb.tokens != dnsplit_b_) throw LoomError("prefill: dnc parts do not match dnsplit");
      if (dnc_group_) {   // (dncgroup) heads [0, a) in dispatches of <= g heads: the same kernel per workgroup
        std::uint32_t h = 0;
        for (std::uint32_t i = 0; h < dnsplit_a_; ++i) {
          const std::string dp = QkvHal("dnc_a_p" + std::to_string(i), wqkv, nq);
          const Geom& gp = geom_.at(dp);
          if (gp.tokens != std::min(dnc_group_, dnsplit_a_ - h)) throw LoomError("prefill: " + dp + " does not match dncgroup");
          Dispatch(Exe(dp), "yah_deltanet", 1, gp.tokens, 1, gp.rowgrp, 1, 1, ba, 0b11000000000);
          h += gp.tokens;
        }
      } else {
        Dispatch(Exe(da), "yah_deltanet", 1, ga.tokens, 1, ga.rowgrp, 1, 1, ba, 0b11000000000);
      }
      Dispatch(Exe(db), "yah_deltanet", 1, gb.tokens, 1, gb.rowgrp, 1, 1, bb, 0b11000000000);
      dn_done = true;
    } else if (conv_c.length) {
      const std::string hg = QkvHal("convkq_g", wqkv, nq), hc = QkvHal("convkq_c", wqkv, nq);
      const Geom& gg = geom_.at(hg);
      const Geom& gc = geom_.at(hc);
      Dispatch(Exe(hg), "yah_ssm_conv_kq", gg.tokens, B_ / conv_tb, 1, 256, 1, 1,
               {Ref(*qkv_), TRef(*Find(pre + "ssm_conv1d.weight")), cs, Ref(*conv_out_), Ref(*kqbuf_)});
      Dispatch(Exe(hc), "yah_ssm_conv_kq_c", gc.tokens, gc.tt, 1, gc.rowgrp, 1, 1,
               {CRef(conv_c.offset, conv_c.length),
                TRef(*Find(pre + "ssm_conv1d.weight")),
                cs,
                Ref(*conv_out_)},
               8);
    } else if (const auto dvc = geom_.find("dnvconv"); dvc != geom_.end()) {
      // dispatch.txt "dnvconv <conv workgroups>": the conv takes only the q / k heads; DeltaNet computes the v channels'
      // conv and alpha / beta itself (gen_gdn_chunk vconv), and prep_ab only advances the conv ring, after both parts
      // (they read the ring before token 0)
      if (!dnsplit_a_ || !dvc->second.tokens) throw LoomError("dispatch.txt dnvconv needs dnsplit and a conv grid");
      Dispatch(Exe("convkq.hal"), "yah_ssm_conv_kq", dvc->second.tokens, B_ / conv_tb, 1, 256, 1, 1,
               {Ref(*qkv_), TRef(*Find(pre + "ssm_conv1d.weight")), cs, Ref(*conv_out_), Ref(*kqbuf_)});
      const std::vector<hrx_buffer_ref_t> in{Ref(*conv_out_), Ref(*qkv_), TRef(*Find(pre + "ssm_conv1d.weight")), cs,
                                             Ref(*alpha_), Ref(*beta_), TRef(*Find(pre + "ssm_a")),
                                             TRef(*Find(pre + "ssm_dt.bias")), Ref(*valid_)};
      auto ba = in, bb = in;
      ba.insert(ba.end(), {st, Ref(*raw_)});
      bb.insert(bb.end(), {st, Ref(*raw2_)});
      Dispatch(Exe("rowsplit_a.hal"), "yah_deltanet", dn_rowgrp_, dnsplit_a_, 1, dn_threads_, 1, 1, ba, 0b11000000000);
      Dispatch(Exe("rowsplit_b.hal"), "yah_deltanet", dn_rowgrp_, dnsplit_b_, 1, dn_threads_, 1, 1, bb, 0b111000000000);
      // The second part's few workgroups leave most of the GPU to the gate GEMM
      if (gate_late) RunKstore(pre + "attn_gate.weight", *gate_, gate_af);
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
    // an afrag ssm_out reads the postnorm output fragment-major (postnorm_t)
    const bool out_af = Af("gemm_kres", pre + "ssm_out.weight");
    const std::string pn = out_af ? "postnorm_t" : "postnorm";
    // heads [0, a) are ssm_out's NPU K chunk: with "npupostnormbfp" part a also writes the NPU's input into A and
    // NpuEncode("out") skips the encoder (RunResidual splits only full chunks; an unused A costs nothing)
    const std::uint32_t on = NpuRows("out", pre + "ssm_out.weight");
    const bool bfp = dnsplit_a_ && geom_.count("npupostnormbfp") && on && !SwSiteR("out", BlkOf(pre), on) &&
                     trim_row_ == kAllRows && tail_ == kAllRows;
    // (a Q4_K ssm_out reads the Q4_K-order stream: "_a_bfp_pk.hal" writes it into A's second half instead)
    const bool pk = bfp && NpuPk(pre + "ssm_out.weight") && geom_.count("npupostnormbfppk");
    // gc: the postnorm parts read the gate's NPU columns straight from C ("<pn>_a_bfp[_pk]_gc<sfx>.hal", "<pn>_b_gc<sfx>.hal",
    // gen_postnorm_bfp gc), so the gate unpack goes; the out job's C must not overlap the gate's (RunResidual checks)
    std::string gc_sfx;
    // (YAH_PP_HALVES=2) the gate's join hands over to the other half, whose conv of this layer reads the ring next:
    // it advances now (prep_ab reads this half's qkv and writes the ring and ab, which nothing later here reads)
    if (halves_ == 2 && dn_done) ring(), dn_done = false;
    if (gate_job >= 0) {   // (the conv and DeltaNet neither read gate's C nor write the A it reads)
      NpuJoin(gate_job);
      const std::string un = RowsHal(Np4Hal("npu_unpack_gate", pre + "attn_gate.weight"), "gate", ng, Np4(pre + "attn_gate.weight"));
      const std::string sfx = un.substr(15, un.size() - 19);   // "npu_unpack_gate<sfx>.hal"
      // (a swap-form out: part a writes no stream, "<pn>_a_gc<sfx>.hal")
      if ((bfp || (dnsplit_a_ && on && SwSiteR("out", BlkOf(pre), on))) &&
          geom_.count(pn + (pk ? "_a_bfp_pk_gc" : bfp ? "_a_bfp_gc" : "_a_gc") + sfx + ".hal") &&
          geom_.count(pn + "_b_gc" + sfx + ".hal"))
        gc_sfx = "_gc" + sfx, gc_read_ = cg;
      else
        NpuUnpack(un, cg, {Ref(*gate_)}, 2);
    }
    if (tail_ == kNoRows) {  // the recurrent state is written; nothing reads this layer's output
      if (dn_done) ring();
      gc_read_ = {};
      return;
    }
    const hrx_buffer_ref_t gate{gate_->handle, 0, std::size_t{B_} * kInner * 4};
    const hrx_buffer_ref_t gcb = npu_ ? CRef(cg.offset, cg.length) : hrx_buffer_ref_t{gate_->handle, cg.offset, cg.length};
    if (dnsplit_a_) {
      norm_bfp_ = bfp && !pk ? "out" : "";
      norm_pk_ = pk ? "out" : "";
      std::vector<hrx_buffer_ref_t> b{Ref(*raw_), TRef(*Find(pre + "ssm_norm.weight")), gate, Ref(*scratch_)};
      if (bfp) b.push_back(ARef(pk ? PkA() : 0, NpuK(5120).a));
      if (!gc_sfx.empty()) b.push_back(gcb);
      Dispatch(Exe(pn + (pk ? "_a_bfp_pk" : bfp ? "_a_bfp" : "_a") + gc_sfx + ".hal"), "yah_ssm_postnorm_fp16",
               dnsplit_a_ * B_ / 8, 1, 1, 256, 1, 1, b);
      const auto part_b = [this, pn, gate, gc_sfx, gcb, w = TRef(*Find(pre + "ssm_norm.weight"))] {
        std::vector<hrx_buffer_ref_t> bb{Ref(*raw2_), w, gate, Ref(*scratch_)};
        if (!gc_sfx.empty()) bb.push_back(gcb);
        Dispatch(Exe(pn + "_b" + gc_sfx + ".hal"), "yah_ssm_postnorm_fp16", dnsplit_b_ * B_ / 8, 1, 1, 256, 1, 1, bb);
        gc_read_ = {};
      };
      // with the NPU's out job reading part a's A, part b (and the conv ring) follow its release (RunResidual), not
      // precede it: the GPU's own out GEMM still reads them after (YAH_NPU_NODEFER=1: before, as without the NPU)
      static const bool nodefer = std::getenv("YAH_NPU_NODEFER") != nullptr;
      if (bfp && !nodefer) {
        after_release_ = [part_b, &ring, dn_done] {
          part_b();
          if (dn_done) ring();
        };
        dn_done = false;
      } else {
        part_b();
      }
    } else {
      Dispatch(Exe(out_af ? "postnorm_t.hal" : "postnorm.hal"), "yah_ssm_postnorm_fp16", 6 * B_, 1, 1, 256, 1, 1,
               {Ref(*raw_), TRef(*Find(pre + "ssm_norm.weight")), gate, Ref(*scratch_)});
    }
    if (dn_done) ring();
    trim_row_ = tail_;
    RunResidual(pre + "ssm_out.weight", *scratch_, out_af);
    RunAfterRelease();
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
  std::uint32_t dnc_group_ = 0;   // (dispatch.txt "dncgroup") the first DeltaNet part's heads a dispatch
  std::uint32_t dnsplit_a_ = 0, dnsplit_b_ = 0;  // DeltaNet head split (dispatch.txt "dnsplit"; 0: one dispatch)
  std::uint32_t qrot_first_ = 0;  // first chunk whose attention reads yah_qrot's rotated Q (dispatch.txt "qrot")
  std::size_t deq_bytes_ = 0;
  static constexpr std::uint32_t kPqueueSlots = 64;
  LoomBuffer* pqueue_ = nullptr;
  LoomBuffer* ksplit_ = nullptr;  // the key-split tail's partials (dispatch.txt "ksplit_<hal>")
  // the NPU attention (EnableNpuAttn): its KV heads, built nb, first chunk, buffers, the count of NPU chunks so far
  // (the flag words' values) and the chunk's layer jobs (layer, nb), queued once its graph launched
  NpuAttn* npu_attn_ = nullptr;
  NpuAttn::Buffers na_{};
  std::uint32_t na_kvn_ = 0, na_nbs_ = 0, na_from_ = 0, na_seq_ = 0;
  struct NaJob {
    std::uint32_t ai, nb;
    std::size_t after;   // the chunk's GEMM split jobs recorded before it (flagged_)
  };
  std::vector<NaJob> na_jobs_;
  std::size_t ksplit_bytes_ = 0;
  std::uint32_t n_ = 0;  // real tokens of the chunk from the last Embed
  LoomGraph* graph_ = nullptr;  // open while RunLayers records
  std::unique_ptr<LoomGraph> chunk_graph_;
  std::size_t npu_enq_ = 0;        // (NpuEarly) this chunk's jobs already enqueued (SplitGraph)
  std::uint32_t split_layer_ = 0;  // RunLayers: SplitGraph before this layer's first dispatch (0: none)
  bool split_armed_ = false;       // recording split_layer_: its first dispatch splits
  bool warming_ = false;           // WarmExecutables' pass: Dispatch records nothing
  NpuSplit* npu_ = nullptr;        // EnableNpu
  bool npu_on_ = false;            // this chunk splits with the NPU
  std::uint32_t layer_ = 0;        // RecordLayers' current layer (NpuSplit::Queued::layer)
  bool npu_planning_ = false;      // EnableNpu's binding pass: Dispatch records nothing
  LoomBuffer* npurem_ = nullptr;   // the GPU's K remainder of the NPU's out / down rows (f32 [tokens][rows])
  std::uint32_t ffnblk_ = 0;       // the NPU's FFN block features (dispatch.txt "npuffnblk"; RunFfnBlock), 0: none
  bool ffn_rr_ = false;   // (RunFfnBlock: the GPU's down added hidden, FlushFfn's "_rr" sum)
  LoomBuffer* ffnrem_ = nullptr;   // its GPU down over the other features (f32 [tokens][5120])
  std::string norm_bfp_;           // the site whose NPU input the last norm wrote (RunNorm npu_site)
  std::string norm_pk_;            // .. in the Q4_K order (RunLayerNorm pk)
  NpuView ffn_pending_{};          // the FFN block's partials not yet summed into hidden() (RunFfnBlock, FlushFfn)
  NpuView out_pending_{};          // the out site's C not yet summed into hidden() (RunResidual, FlushOut)
  std::uint32_t out_nn_ = 0;       // ... its NPU rows
  bool out_np4_ = false;           // ... from an NP 4 image (npu_unpack_out_4.hal)
  bool ffn_pending4_ = false;      // ... from NP 4 down calls (npu_unpack_ffnblk_4.hal)
  bool ffn_pendingsw_ = false;     // ... from the swap-form down (npu_unpack_ffnblk_sw.hal, npu_unormsw_rr_*)
  std::string pending_norm_;       // a fused norm's f16 output still to record (RunLayerNorm, RunPendingNorm)
  NormOut pending_mode_{};
  static constexpr std::size_t kDcolRecordRow = 576;   // a decoder record's row bytes (gen_npu_dec.RECROW)
  static std::size_t RecordRow(const char* fmt) {
    return std::strcmp(fmt, "q5k") == 0 ? 704 : std::strcmp(fmt, "bfp") == 0 ? 1152 : kDcolRecordRow;
  }
  // Rows without an NPU decoder (gate / up pairs, sites) the emitter wrote pre-decoded (emit_prefill_pp bfp_pair_rows,
  // bfp_site_rows: npu_bfp_<...>.bin, the decoder records' bfp16 fragments, 288 bytes per 256 weights), read once into
  // resident writable memory like the model (RegisterRaw pins it; + 8192 bytes of over-read slack)
  static constexpr std::size_t kBfpBlock = 288;
  // Host-side NPU rows (emit_prefill_pp bfp_pair_rows / bfp_site_rows): file's bytes in order into the spans (offset, bytes) of
  // a buffer of bytes (+ 8 KB the records over-read; pages outside the spans are never touched), kept for the run
  const std::uint8_t* HostRows(const std::string& file, std::size_t bytes,
                               const std::vector<std::pair<std::size_t, std::size_t>>& spans) {
    if (const auto it = bfp_.find(file); it != bfp_.end()) return it->second;
    const std::string path = dir_ + "/" + file;
    void* p = ::mmap(nullptr, bytes + 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw LoomError("prefill: no memory for " + path);
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw LoomError("prefill: cannot open " + path);
    std::size_t at = 0;   // the file's bytes in order, span after span
    for (const auto& [off, n] : spans) {
      for (std::size_t done = 0; done < n;) {
        const ::ssize_t r = ::pread(fd, static_cast<std::uint8_t*>(p) + off + done, n - done, static_cast<off_t>(at + done));
        if (r <= 0) {
          ::close(fd);
          throw LoomError("prefill: short read of " + path);
        }
        done += static_cast<std::size_t>(r);
      }
      at += n;
    }
    ::close(fd);
    return bfp_[file] = static_cast<std::uint8_t*>(p);
  }
  std::map<std::string, std::uint8_t*> bfp_;
  std::uint32_t npu_job_ = 0;      // NPU jobs of this chunk so far
  std::size_t npu_cleared_ = 0;    // flagged_ jobs whose flag words NpuJoin cleared
  std::size_t npu_joined_ = 0;     // flagged_ jobs an NpuJoin waited for (in order)
  std::vector<NpuSide> job_side_;  // per flagged_ job: its A / C ranges (NpuEnqueue)
  std::vector<NpuSplit::Queued> flagged_;  // this chunk's NPU jobs (index = job), queued after the launch
  std::map<std::string, std::uint32_t> npu_rows_;  // NPU rows per site (dispatch.txt "npusplit_<site>")
  std::map<std::string, std::uint32_t> npu_rows4_;  // the NP 4 formats' (Np4: "npusplit4_<site>")
  bool kv16_scratch_ = false, kv_paged_ = false, vtrans_ = false, rope_kpaged_ = false;
  bool qnorm_ = false;  // the attention normalizes / rotates its own (raw f32) Q; RoPE does K / V only
  bool gsig_ = false;   // the kqg GEMM stores the gate's sigmoid (unorm16 bits by default; only the GPU kqg path)
  bool qsplit_ = false;   // the kqg GEMM stores f16 q w / 16 + rotary dims + sums of q^2 (binds the q norm weight)
  bool vtrope_ = false;  // RoPE stores V^T straight into the paged pool (no yah_vtpage pass)
  bool attn_kq4_ = false, attn_kq8_ = false, attn_vq4_ = false, attn_vq8_ = false, attn_vqt_ = false;
  bool paged_f16k_ = false, paged_f16v_ = false;
  std::size_t kv_cache_ = 0, kv16_layer_ = 0, vt_bytes_ = 0, ks_bytes_ = 0, kq_bytes_ = 0, vq_bytes_ = 0,
              vqs_bytes_ = 0, pool_bytes_ = 0;
  hrx_buffer_ref_t ptab_ref_{};
  std::vector<std::uint32_t> host_ids_;  // Embed's staging
  // token_embd off the device (LoomWeights skip): the chunk's rows, staged by Embed (per chunk of a pair)
  LoomBuffer* embed_rows_ = nullptr;
  std::vector<std::uint8_t> host_rows_;          // the chunk's staged token_embd rows
  hrx_timeline_point_t rows_copied_ = {};        // when their last copy landed
  // YAH_EMBED_OVERLAP: the gather thread (joined before host_rows_ goes: declared after it), its ids, error and times
  struct Joined {
    std::thread t;
    void Join() {
      if (t.joinable()) t.join();
    }
    ~Joined() { Join(); }
  } gather_;
  std::vector<std::uint32_t> gather_ids_;
  bool half_rows_ = false;   // (YAH_PP_HALVES) the gather holds both halves' rows (FinishEmbed streams two)
  std::exception_ptr gather_error_;
  double gather_ms_[2] = {};
  // The overlapped gather's end: join it, stream the rows (in stream order before the graph that reads them)
  void FinishEmbed() {
    if (!gather_.t.joinable()) return;
    gather_.Join();
    if (gather_error_) std::rethrow_exception(std::exchange(gather_error_, nullptr));
    const auto e2 = std::chrono::steady_clock::now();
    if (half_rows_) {   // (YAH_PP_HALVES) the rows of half h to its bank (one stream: the later point covers both)
      const std::size_t hb = host_rows_.size() / 2;
      gpu_.StreamH2D(*hs_[0].embed_rows_, host_rows_.data(), hb);
      rows_copied_ = gpu_.StreamH2D(*hs_[1].embed_rows_, host_rows_.data() + hb, hb);
      half_rows_ = false;
    } else {
      rows_copied_ = gpu_.StreamH2D(*embed_rows_, host_rows_.data(), host_rows_.size());
    }
    if (std::getenv("YAH_HOST_TIMES"))
      std::fprintf(stderr, "host: embed wait %.2f ms, gather %.2f ms (%zu B, overlapped), stream %.2f ms\n",
                   gather_ms_[0], gather_ms_[1], host_rows_.size(),
                   std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e2).count());
  }
  hrx_buffer_ref_t EmbedTable() const { return embed_rows_ ? Ref(*embed_rows_) : TRef(*Find("token_embd.weight")); }
  static constexpr std::size_t kWaitLogSlots = 65536;
  static constexpr std::size_t kClkSlots = 4096;
  LoomBuffer* clk_ = nullptr;     // YAH_CLK_PROBE: 16 bytes a probe (ClkProbe, ReportClk)
  LoomBuffer* stamp_ = nullptr;   // YAH_CLK_PROBE: 8 bytes a flag wait (NpuJoin)
  std::size_t clk_n_ = 0, stamp_n_ = 0;
  bool clk_ffn_on_ = false;        // YAH_CLK_PROBE=ffn
  std::vector<ClkFfn> clk_ffn_;
  LoomBuffer* waitlog_ = nullptr;          // YAH_NPU_WAITLOG: a poll count per GPU wait (NpuJoin, ReportWaitLog)
  std::vector<std::string> waitlog_tag_;   // each slot's layer and job tag
  std::vector<std::string> prof_tags_;     // YAH_NPU_PROF: the first NPU chunk's job tags (ReportNpuProf)
  std::vector<std::uint32_t> prof_layers_;  // and their layers
  LoomBuffer *grid_iq3s_ = nullptr, *grid_iq3xxs_ = nullptr, *grid_iq2xxs_ = nullptr, *grid_iq2xs_ = nullptr,
             *grid_iq2s_ = nullptr, *ksigns_ = nullptr;
  LoomBuffer *hidden_ = nullptr, *normnil_ = nullptr, *scratch_ = nullptr, *qkv_ = nullptr,
             *gate_ = nullptr, *alpha_ = nullptr, *beta_ = nullptr, *q_ = nullptr, *q16_ = nullptr, *kbuf_ = nullptr,
             *vbuf_ = nullptr, *raw_ = nullptr, *raw2_ = nullptr, *conv_out_ = nullptr, *kqbuf_ = nullptr, *ab_ = nullptr,
             *conv_state_ = nullptr, *state_ = nullptr, *kv16_ = nullptr, *kc32_ = nullptr, *vc32_ = nullptr,
             *lse_ = nullptr, *eps_ = nullptr, *ffnup_ = nullptr, *gateffn_ = nullptr, *uwstage_ = nullptr,
             *wstage_ = nullptr, *ostage_ = nullptr, *partial_ = nullptr, *hidden2_ = nullptr, *normed_ = nullptr,
             *ptab_ = nullptr, *vt16_ = nullptr, *kq8buf_ = nullptr, *ksbuf_ = nullptr, *kmbuf_ = nullptr,
             *vqbuf_ = nullptr, *vqsbuf_ = nullptr, *kpool_ = nullptr, *vtpool_ = nullptr, *valid_ = nullptr, *qr16_ = nullptr,
             *kdeq_ = nullptr, *vdeq_ = nullptr, *kscale_ = nullptr, *ids_ = nullptr,
             *normt_ = nullptr;
  std::int64_t keep_rows_ = kAllRows, tail_ = kAllRows, trim_row_ = kAllRows;
  // ---- YAH_PP_HALVES (port/design-overlap.md sections 9-10) ----
  std::uint32_t halves_ = 0, half_ = 0;
  bool in_half_embed_ = false;
  std::int64_t last_job_ = -1;              // this half's last enqueued job (NpuJoin's default)
  std::size_t npu_a0_ = 0, npu_w0_ = 0, npu_c0_ = 0;   // this half's A / W / C region starts
  std::size_t a_one_ = 0, w_one_ = 0, c_one_ = 0;      // one half's A / W / C region bytes
  std::string swap_layer_;                  // (shared) the layer whose swap weights the stream holds
  std::size_t carriers_[2] = {0, 0};        // carrier dispatches recorded per half (half 0 without halves), logged
  std::int64_t part_done_[2] = {-1, -1};     // (shared, part carriers) each half's last recorded carrier's r5k
  std::int64_t swap_reader_[2] = {-1, -1};  // (shared) each stream slot's last swap job (it reads the slot)
  // (halves) before a write into stream slot s: its last reader joined (no handover: the other half must not record
  // between); in HALVES=2 order it already is (the NPU queue runs it before this half's joined jobs), so this waits
  // only if that order breaks
  void SlotFree(std::size_t s) {
    if (halves_ && !npu_planning_ && swap_reader_[s] >= 0 && static_cast<std::size_t>(swap_reader_[s]) >= npu_joined_)
      NpuJoin(swap_reader_[s], false);
  }
#define YAH_PF_HALF(X)                                                                                                \
  X(hidden_) X(hidden2_) X(partial_) X(scratch_) X(qkv_) X(gate_) X(alpha_) X(beta_) X(ab_) X(q_) X(q16_) X(qr16_)     \
  X(kbuf_) X(vbuf_) X(kqbuf_) X(lse_) X(normt_) X(raw_) X(raw2_) X(conv_out_) X(ffnup_) X(gateffn_) X(ostage_)         \
  X(npurem_) X(ffnrem_) X(ids_) X(embed_rows_) X(valid_) X(kv16_) X(n_) X(keep_rows_) X(tail_) X(trim_row_) X(layer_) \
  X(norm_bfp_) X(norm_pk_) X(ffn_pending_) X(out_pending_) X(out_nn_) X(out_np4_) X(ffn_pending4_) X(ffn_pendingsw_) X(ffn_rr_)         \
  X(pending_norm_) X(pending_mode_) X(gc_read_) X(qc_hal_) X(qc_c_) X(dn_c_end_) X(after_release_) X(panel_job_)       \
  X(last_job_) X(npu_a0_) X(npu_w0_) X(npu_c0_)
  struct HalfState {
#define YAH_PF_FIELD(m) decltype(LoomPrefill::m) m{};
    YAH_PF_HALF(YAH_PF_FIELD)
#undef YAH_PF_FIELD
  };
  HalfState hs_[2];
  void SaveHalf(std::uint32_t h) {
#define YAH_PF_SAVE(m) hs_[h].m = m;
    YAH_PF_HALF(YAH_PF_SAVE)
#undef YAH_PF_SAVE
  }
  void LoadHalf(std::uint32_t h) {
#define YAH_PF_LOAD(m) m = hs_[h].m;
    YAH_PF_HALF(YAH_PF_LOAD)
#undef YAH_PF_LOAD
    half_ = h;
  }
  // Half 1's token buffers (half 0's are AllocateBuffers'): the same sizes, so both halves together hold today's 2 B
  // footprint; the shared hosts (raw_ / ffnup_, conv_out_ / gateffn_) per half, so no alias crosses halves.
  void AllocHalfBank() {
    SaveHalf(0);
    const std::size_t B = B_;
    const bool q16 = q16_ != q_;
    hidden_ = &Alloc(B * kHidden * 4);
    scratch_ = &Alloc(B * kFfn * 2);
    qkv_ = &Alloc(B * kQProj * 4);
    gate_ = &Alloc(B * kInner * 4);
    alpha_ = &Alloc(B * kTs * 4);
    beta_ = &Alloc(B * kTs * 4);
    q_ = &Alloc(B * kAttn * 4);
    q16_ = q16 ? &Alloc(B * kAttn * 2) : q_;
    if (qr16_) qr16_ = &Alloc(B * kAttn * 2);
    kbuf_ = &Alloc(B * kKv * 4);
    vbuf_ = &Alloc(B * kKv * 4);
    LoomBuffer* up_host = &Alloc(std::max(B * kInner * 4, B * kFfn * 2));
    LoomBuffer* gate_host = &Alloc(std::max(B * kQkv * 4, B * kFfn * 4));
    raw_ = &Share(*up_host, B * kInner * 4);
    if (raw2_) raw2_ = &Alloc(B * kInner * 4);
    conv_out_ = &Share(*gate_host, B * kQkv * 4);
    kqbuf_ = &Alloc(B * kKh * 3 * 4);
    ab_ = &Alloc(B * kTs * 2 * 4);
    if (kv16_scratch_) kv16_ = &Alloc(2 * kv16_layer_), Zero(*kv16_);
    lse_ = &Alloc(B * kHeads * 4);
    ffnup_ = &Share(*up_host, B * kFfn * 2);
    if (normt_) normt_ = &Alloc(B * kHidden * 2);
    gateffn_ = &Share(*gate_host, B * kFfn * 4);
    ostage_ = &Alloc(std::max<std::size_t>(HandQ2kRows() * B * 4, 4));
    partial_ = &Alloc(B * kHidden * 4);
    hidden2_ = &Alloc(B * kHidden * 4);
    valid_ = &Alloc(4);
    ids_ = &Alloc(B * 4);
    if (embed_rows_) embed_rows_ = &Alloc(host_rows_.size()), host_rows_.resize(2 * host_rows_.size());
    SaveHalf(1);
    LoadHalf(0);
  }
  // The recorder's baton: one half's thread records at a time; a half hands over at its NpuJoin (HALVES 2) or at its
  // end (both modes), so the graph and the NPU queue get the halves' work in that order with true job indices.
  struct Baton {
    std::mutex m;
    std::condition_variable cv;
    std::uint32_t turn = 0;
    bool done[2] = {false, false};
    std::exception_ptr error;
  };
  Baton* baton_ = nullptr;
  void YieldHalf() {
    if (!baton_ || halves_ != 2) return;
    const std::uint32_t h = half_;
    SaveHalf(h);
    {
      std::unique_lock<std::mutex> lk(baton_->m);
      if (!baton_->done[1 - h]) baton_->turn = 1 - h;
      baton_->cv.notify_all();
      baton_->cv.wait(lk, [&] { return baton_->turn == h; });
    }
    LoadHalf(h);
  }
  // RunLayers' record of chunk ci (2 B tokens) as halves 2 ci and 2 ci + 1 of this set
  void RecordHalves(std::uint32_t ci, const KvHook& hook) {
    const std::int64_t keep = keep_rows_;
    for (std::uint32_t h = 0; h < 2; ++h) {
      LoadHalf(h);
      keep_rows_ = keep == kAllRows || keep == kNoRows ? keep
                   : keep / static_cast<std::int64_t>(B_) == h ? keep % static_cast<std::int64_t>(B_)
                                                                 : kNoRows;
      last_job_ = -1;
      SaveHalf(h);
    }
    Baton bt;
    baton_ = &bt;
    const auto body = [&](std::uint32_t h) {
      {
        std::unique_lock<std::mutex> lk(bt.m);
        bt.cv.wait(lk, [&] { return bt.turn == h; });
      }
      LoadHalf(h);
      try {
        if (!bt.error) RecordLayers(2 * ci + h, hook);
      } catch (...) {
        std::lock_guard<std::mutex> lk(bt.m);
        if (!bt.error) bt.error = std::current_exception();
      }
      SaveHalf(h);
      std::lock_guard<std::mutex> lk(bt.m);
      bt.done[h] = true;
      bt.turn = 1 - h;
      bt.cv.notify_all();
    };
    std::thread t1(body, 1u);
    body(0);
    t1.join();
    baton_ = nullptr;
    LoadHalf(1);
    if (bt.error) std::rethrow_exception(bt.error);
  }
  // YAH_NPU_FAKE's stand-in thread, joined before the members it reads go away (the last member: destroyed first)
  struct FakeNpu {
    std::thread t;
    void Join() {
      if (t.joinable()) t.join();
    }
    ~FakeNpu() { Join(); }
  } fake_npu_;
};

}  // namespace yah::model


#endif  // YAH_MODEL_LOOM_PREFILL_HPP_
