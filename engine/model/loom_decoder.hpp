// LoomDecoder: the single-token decode step on Loom through HRX, GEMV-based.
//
// HAL set: tools/emit_decode.py (decode.txt "ctx T": the paged KV pools hold T rows).
// Per layer, as HIP's Forward::Decode:
//   rmsnorm -> full attention: attn_q / attn_k / attn_v GEMV, unpack q|gate, QK norm
//              + RoPE, append K / V to the paged pools, split-K attention + reduce
//              (gate), attn_output GEMV += hidden
//           -> recurrent:      attn_qkv / attn_gate / ssm_alpha / ssm_beta GEMV, conv,
//              DeltaNet (gated norm inside), ssm_out GEMV += hidden
//   rmsnorm -> ffn_gate|ffn_up SwiGLU GEMV -> ffn_down GEMV += hidden
// head: rmsnorm -> output GEMV -> argmax into a device token stream that the next
// step's embedding kernel (IQ4_XS token_embd) reads, so steps are enqueued back to
// back with no host round trip.
//
// The recurrent state and the KV pools are external (LoomDecoderState): either the
// decoder's own (OwnState, for a prompt fed through decode) or the prefill's
// (loom_forward_pp with YAH_GEN: its paged pools, page table, conv and DeltaNet state,
// which use the same layouts).
#ifndef YAH_MODEL_LOOM_DECODER_HPP_
#define YAH_MODEL_LOOM_DECODER_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "model/loom_runtime.hpp"

namespace yah::model {

struct LoomDecoderState {
  std::vector<hrx_buffer_ref_t> kpool, vtpool;   // per full-attention layer, T * 1024 f16 each
  hrx_buffer_ref_t ptab{};                        // T / 256 i32 (logical page -> physical page)
  hrx_buffer_t convstate = nullptr;               // per recurrent layer si: si * 10240 * 4 f32
  hrx_buffer_t dstate = nullptr;                  // per recurrent layer si: si * 48 * 128 * 128 f32
};

class LoomDecoder {
 public:
  static constexpr std::uint32_t kHidden = 5120, kFfn = 17408, kAttn = 6144, kQProj = 12288, kKv = 1024;
  static constexpr std::uint32_t kInner = 6144, kQkv = 10240, kHeadsV = 48, kTs = 48, kState = 128;
  static constexpr std::uint32_t kHeads = 24, kKvHeads = 4, kVocab = 248320;
  static constexpr std::uint32_t kConvState = kQkv * 4, kStateElems = kHeadsV * kState * kState;
  static constexpr std::uint32_t kR = 2, kW = 4;   // gen_gemv defaults: rows per wave, waves per workgroup

  LoomDecoder(LoomDevice& gpu, const core::Gguf& gguf, const core::Qwen35Config& cfg, std::string dir,
              hrx_buffer_t weights, std::size_t delta)
      : gpu_(gpu), gguf_(gguf), cfg_(cfg), dir_(std::move(dir)), weights_(weights), delta_(delta) {
    {
      const auto txt = ReadAll(dir_ + "/decode.txt");
      std::istringstream in(std::string(txt.begin(), txt.end()));
      std::string k;
      in >> k >> T_;
      if (k != "ctx" || T_ == 0 || T_ % 256) throw LoomError("bad decode.txt in " + dir_);
    }
    const std::uint32_t npg = T_ / 256;
    if (static_cast<std::uint32_t>(Find("token_embd.weight")->type) != 23)
      throw LoomError("token_embd: only IQ4_XS is wired");
    const char* tnames[5] = {"grid_iq3s.bin", "grid_iq3xxs.bin", "grid_iq2xxs.bin", "grid_iq2xs.bin", "ksigns_iq2xs.bin"};
    for (const char* tn : tnames) {
      const auto data = ReadAll(dir_ + "/" + tn);
      LoomBuffer& b = Alloc(data.size());
      gpu_.H2D(b, data.data(), data.size());
      tabs_.push_back({b.handle, 0, data.size()});
    }
    hidden_ = &Alloc(kHidden * 4); normed_ = &Alloc(kHidden * 4); qg_ = &Alloc(kQProj * 4);
    q_ = &Alloc(kAttn * 4); gate_ = &Alloc(kAttn * 4); kb_ = &Alloc(kKv * 4); vb_ = &Alloc(kKv * 4);
    aout_ = &Alloc(kAttn * 4); qkv_ = &Alloc(kQkv * 4); alpha_ = &Alloc(kTs * 4); beta_ = &Alloc(kTs * 4);
    convout_ = &Alloc(kQkv * 4); ssmout_ = &Alloc(kInner * 4); ffnact_ = &Alloc(kFfn * 4);
    logits_ = &Alloc(std::size_t{kVocab} * 4); sink_ = &Alloc(4); cnt_ = &Alloc(4);
    toks_ = &Alloc((std::size_t{T_} + 1) * 4); posarr_ = &Alloc(std::size_t{T_} * 4); eps_ = &Alloc(4);
    c32a_ = &Alloc(kKv * 4); c32b_ = &Alloc(kKv * 4); c16a_ = &Alloc(kKv * 2); c16b_ = &Alloc(kKv * 2);
    acc_ = &Alloc(std::size_t{npg} * kHeads * 256 * 4); ml_ = &Alloc(std::size_t{npg} * kHeads * 2 * 4);
    const float e = 1.0e-6f;
    gpu_.H2D(*eps_, &e, 4);
    std::vector<std::int32_t> pv(T_);
    for (std::uint32_t i = 0; i < T_; ++i) pv[i] = static_cast<std::int32_t>(i);
    gpu_.H2D(*posarr_, pv.data(), pv.size() * 4);
    trace_ = std::getenv("YAH_DEC_TRACE") != nullptr;
    overlap_ = std::getenv("YAH_DEC_NO_OVERLAP") == nullptr;
    { const char* v = std::getenv("YAH_DEC_BANDS"); bands_ = !(v && std::string(v) == "0"); }
    // YAH_DEC_RESNORM=1: fold each following RMSNorm into the residual GEMV (gen_gemv
    // resid_norm). Off: neutral (61.30 vs 61.18 ms) -- the last workgroup's ~5 us norm
    // tail cancels the ~3.4 us dependent-dispatch latency it removes, 128 times a token.
    { const char* v = std::getenv("YAH_DEC_RESNORM"); resnorm_ = v && std::string(v) == "1"; }
  }

  [[nodiscard]] std::uint32_t context() const { return T_; }
  std::uint32_t full_layers() const { return cfg_.main_block_count() / cfg_.full_attention_interval; }
  std::uint32_t recurrent_layers() const { return cfg_.main_block_count() - full_layers(); }

  // The decoder's own pools / page table (identity) / zeroed recurrent state.
  LoomDecoderState OwnState() {
    LoomDecoderState st;
    const std::uint32_t npg = T_ / 256;
    const std::size_t poolb = std::size_t{T_} * kKv * 2;
    for (std::uint32_t i = 0; i < full_layers(); ++i) {
      st.kpool.push_back(Ref(Alloc(poolb)));
      st.vtpool.push_back(Ref(Alloc(poolb)));
    }
    LoomBuffer& pt = Alloc(std::size_t{npg} * 4);
    std::vector<std::int32_t> pages(npg);
    for (std::uint32_t i = 0; i < npg; ++i) pages[i] = static_cast<std::int32_t>(i);
    gpu_.H2D(pt, pages.data(), pages.size() * 4);
    st.ptab = Ref(pt);
    st.convstate = Alloc(std::size_t{recurrent_layers()} * kConvState * 4).handle;
    st.dstate = Alloc(std::size_t{recurrent_layers()} * kStateElems * 4).handle;
    return st;
  }

  // External state: sizes are checked against this set's context.
  void Bind(const LoomDecoderState& st) {
    const std::size_t poolb = std::size_t{T_} * kKv * 2;
    if (st.kpool.size() != full_layers() || st.vtpool.size() != full_layers())
      throw LoomError("decoder: one K and one V^T pool per full-attention layer expected");
    for (std::uint32_t i = 0; i < full_layers(); ++i)
      if (st.kpool[i].length < poolb || st.vtpool[i].length < poolb)
        throw LoomError("decoder: KV pool smaller than the decode set's context " + std::to_string(T_));
    if (st.ptab.length < std::size_t{T_ / 256} * 4) throw LoomError("decoder: page table too small");
    st_ = st;
  }

  // Tokens into the device stream at positions at .. at + n - 1.
  void SetTokens(const std::uint32_t* toks, std::size_t n, std::uint32_t at) {
    if (at + n > T_ + 1) throw LoomError("decoder: token stream overflow");
    gpu_.H2D(*toks_, toks, n * 4, std::size_t{at} * 4);
  }
  std::vector<std::uint32_t> Tokens(std::uint32_t from, std::uint32_t to) {
    std::vector<std::uint32_t> out(to - from);
    gpu_.D2H(*toks_, out.data(), out.size() * 4, std::size_t{from} * 4);
    for (auto t : out)
      if (t >= kVocab) throw LoomError("decoder: token id out of range: " + std::to_string(t));
    return out;
  }
  void CopyLogits(float* host) { gpu_.D2H(*logits_, host, std::size_t{kVocab} * 4); }

  // Enqueue the step at position pos: reads toks[pos]; its argmax goes to toks[pos + 1]
  // when pos + 1 >= keep_from (generated positions), else to a sink (prompt positions).
  void Step(std::uint32_t pos, std::uint32_t keep_from) {
    if (pos >= T_) throw LoomError("decoder: position past the set's context");
    if (st_.kpool.empty()) throw LoomError("decoder: no state bound");
    cur_pos_ = pos;
    const hrx_buffer_ref_t dposr{posarr_->handle, std::size_t{pos} * 4, 4};
    Dispatch(Load("embed"), 1, 1, kHidden / 16,
             {TRef("token_embd.weight"), {toks_->handle, std::size_t{pos} * 4, 4}, Ref(*hidden_)});
    const std::uint32_t nl = cfg_.main_block_count();
    // with resnorm_, each residual GEMV's last workgroup also writes the next RMSNorm
    // (gen_gemv resid_norm); only layer 0's input norm is a dispatch of its own
    auto next_norm = [&](std::uint32_t l) {
      return l + 1 < nl ? "blk." + std::to_string(l + 1) + ".attn_norm.weight" : std::string("output_norm.weight");
    };
    for (std::uint32_t l = 0; l < nl; ++l) {
      const std::string pre = "blk." + std::to_string(l) + ".";
      if (!resnorm_ || l == 0) Rmsnorm(*hidden_, pre + "attn_norm.weight", *normed_);
      Tr("in", l, *hidden_, kHidden);
      if (cfg_.IsFullAttention(l)) {
        const std::uint32_t ai = l / cfg_.full_attention_interval;
        Project({pre + "attn_q.weight", pre + "attn_k.weight", pre + "attn_v.weight"}, {qg_, kb_, vb_});
        Dispatch(Load("unpack"), kHeads, 1, 256, {Ref(*qg_), Ref(*q_), Ref(*gate_)});
        Dispatch(Load("rope"), kHeads + kKvHeads, 1, 256,
                 {Ref(*q_), Ref(*kb_), Ref(*vb_), TRef(pre + "attn_q_norm.weight"), TRef(pre + "attn_k_norm.weight"),
                  Ref(*q_), Ref(*kb_), Ref(*c32a_), Ref(*c32b_), Ref(*c16a_), Ref(*c16b_), dposr, Ref(*eps_)});
        Dispatch(Load("dattn_kvappend"), kKvHeads, 1, 256,
                 {Ref(*kb_), Ref(*vb_), st_.kpool[ai], st_.vtpool[ai], st_.ptab, dposr});
        Dispatch(Load("dattn_part"), kKvHeads, pos / 256 + 1, 256,
                 {Ref(*q_), st_.kpool[ai], st_.vtpool[ai], st_.ptab, dposr, Ref(*acc_), Ref(*ml_)});
        Dispatch(Load("dattn_reduce"), kHeads, 1, 256, {Ref(*acc_), Ref(*ml_), Ref(*gate_), dposr, Ref(*aout_)});
        Tr("attn", l, *aout_, kAttn);
        Resid(pre + "attn_output.weight", *aout_, pre + "post_attention_norm.weight");
      } else {
        const std::uint32_t si = l - l / cfg_.full_attention_interval;
        Project({pre + "attn_qkv.weight", pre + "attn_gate.weight", pre + "ssm_alpha.weight", pre + "ssm_beta.weight"},
                {qkv_, gate_, alpha_, beta_});
        Dispatch(Load("ssmconv"), (kQkv + 255) / 256, 1, 256,
                 {{qkv_->handle, 0, std::size_t{kQkv} * 4}, TRef(pre + "ssm_conv1d.weight"),
                  {st_.convstate, std::size_t{si} * kConvState * 4, std::size_t{kConvState} * 4}, Ref(*convout_)});
        Dispatch(Load("deltanet"), kHeadsV, 1, 512,
                 {Ref(*convout_), {st_.dstate, std::size_t{si} * kStateElems * 4, std::size_t{kStateElems} * 4},
                  Ref(*alpha_), Ref(*beta_), TRef(pre + "ssm_a"), TRef(pre + "ssm_dt.bias"),
                  TRef(pre + "ssm_norm.weight"), Ref(*gate_), Ref(*ssmout_)});
        Tr("ssm", l, *ssmout_, kInner);
        Resid(pre + "ssm_out.weight", *ssmout_, pre + "post_attention_norm.weight");
      }
      if (!resnorm_) Rmsnorm(*hidden_, pre + "post_attention_norm.weight", *normed_);
      Gemv("swiglu", {pre + "ffn_gate.weight", pre + "ffn_up.weight"}, *normed_, *ffnact_);
      Tr("ffnact", l, *ffnact_, kFfn);
      Resid(pre + "ffn_down.weight", *ffnact_, next_norm(l));
    }
    if (!resnorm_) Rmsnorm(*hidden_, "output_norm.weight", *normed_);
    Gemv("plain", {"output.weight"}, *normed_, *logits_);
    Tr("logits", 99, *logits_, kVocab);
    Dispatch(Load("argmax"), 1, 1, 1024,
             {Ref(*logits_), pos + 1 >= keep_from ? hrx_buffer_ref_t{toks_->handle, std::size_t{pos + 1} * 4, 4}
                                                  : Ref(*sink_)});
  }

 private:
  struct Fmt { const char* name; std::uint32_t qk, bb, tables; };
  // table bits, in gen_gemv.TABLE_ORDER: grid_iq3s, grid_iq3xxs, grid_iq2xxs, grid_iq2xs, ksigns
  static bool FmtOf(std::uint32_t type, Fmt* f) {
    switch (type) {
      case 8: *f = {"q8_0", 32, 34, 0}; return true;
      case 10: *f = {"q2k", 256, 84, 0}; return true;
      case 11: *f = {"q3k", 256, 110, 0}; return true;
      case 12: *f = {"q4k", 256, 144, 0}; return true;
      case 13: *f = {"q5k", 256, 176, 0}; return true;
      case 14: *f = {"q6k", 256, 210, 0}; return true;
      case 16: *f = {"iq2xxs", 256, 66, 4 | 16}; return true;
      case 17: *f = {"iq2xs", 256, 74, 8 | 16}; return true;
      case 18: *f = {"iq3xxs", 256, 98, 2 | 16}; return true;
      case 21: *f = {"iq3s", 256, 110, 1}; return true;
      case 23: *f = {"iq4xs", 256, 136, 0}; return true;
      default: return false;
    }
  }
  static std::vector<char> ReadAll(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw LoomError("cannot open " + path);
    return std::vector<char>(std::istreambuf_iterator<char>(in), {});
  }
  static hrx_buffer_ref_t Ref(const LoomBuffer& b) { return {b.handle, 0, b.size}; }
  LoomBuffer& Alloc(std::size_t bytes) {
    keep_.push_back(gpu_.Allocate(bytes));
    std::vector<char> z(bytes, 0);
    gpu_.H2D(keep_.back(), z.data(), bytes);
    return keep_.back();
  }
  const core::TensorInfo* Find(const std::string& n) const {
    const auto* t = gguf_.Find(n);
    if (!t) throw LoomError("tensor not found: " + n);
    return t;
  }
  hrx_buffer_ref_t TRef(const std::string& n) const {
    const auto* t = Find(n);
    return {weights_, delta_ + static_cast<std::size_t>(t->offset), static_cast<std::size_t>(t->bytes)};
  }
  LoomExecutable& Load(const std::string& name) {
    auto it = exes_.find(name);
    if (it == exes_.end()) {
      it = exes_.emplace(name, gpu_.Load(dir_ + "/" + name + ".hal")).first;
      if (std::getenv("YAH_DEC_LIST")) std::fprintf(stderr, "loaded %zu %s\n", exes_.size(), name.c_str());
    }
    return it->second;
  }
  // workgroup size and binding count come from the compiled export and must agree with
  // the call site (argmax is a 1024-lane kernel: launched with 32 lanes its per-wave LDS
  // slots stay unwritten and it returns garbage)
  void Dispatch(LoomExecutable& e, std::uint32_t gx, std::uint32_t gy, std::uint32_t wg,
                const std::vector<hrx_buffer_ref_t>& b) {
    const std::uint32_t cw = e.WorkgroupSize(0), cb = e.BindingCount(0);
    if ((cw && cw != wg) || (cb && cb != b.size()))
      throw LoomError("dispatch mismatch: workgroup " + std::to_string(wg) + " vs compiled " + std::to_string(cw) +
                      ", bindings " + std::to_string(b.size()) + " vs " + std::to_string(cb));
    gpu_.Dispatch(e, 0, LoomDevice::Config(gx, gy, 1, wg, 1, 1), nullptr, 0, b.data(), b.size());
  }
  // GEMV: names and footprints as tools/gen_gemv.py. overlap: no ordering barrier
  // against the previous dispatch (independent projections of one input); the next
  // barriered dispatch still waits for all of them.
  void Gemv(const char* kind, const std::vector<std::string>& ws, const LoomBuffer& x, const LoomBuffer& y,
            bool overlap = false, const std::vector<hrx_buffer_ref_t>& extra = {}) {
    std::string name = std::string("gv_") + kind;
    std::uint32_t tbits = 0, M = 0, K = 0;
    std::vector<hrx_buffer_ref_t> b;
    for (const auto& w : ws) {
      const auto* t = Find(w);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(t->type), &f)) throw LoomError("no GEMV format for " + w);
      K = static_cast<std::uint32_t>(t->dims[0]);
      M = static_cast<std::uint32_t>(t->dims[1]);
      if (t->bytes != std::uint64_t{K} / f.qk * f.bb * M) throw LoomError("footprint mismatch on " + w);
      name += std::string("_") + f.name;
      tbits |= f.tables;
      b.push_back(TRef(w));
    }
    name += "_" + std::to_string(M) + "_" + std::to_string(K);
    for (int i = 0; i < 5; ++i)
      if (tbits & (1u << i)) b.push_back(tabs_[i]);
    if (x.size < std::size_t{K} * 4 || y.size < std::size_t{M} * 4) throw LoomError("GEMV operand too small: " + name);
    b.push_back({x.handle, 0, std::size_t{K} * 4});
    b.push_back({y.handle, 0, std::size_t{M} * 4});
    b.insert(b.end(), extra.begin(), extra.end());
    LoomExecutable& exe = Load(name);
    if (overlap && overlap_) gpu_.NoBarrierNext();
    Dispatch(exe, M / (kR * kW), 1, 32 * kW, b);
  }
  // A layer's input projections of normed_: one band-fused GEMV (gen_gemv gen_bands), or
  // with YAH_DEC_BANDS=0 one GEMV each, the later ones without an ordering barrier.
  void Project(const std::vector<std::string>& ws, const std::vector<LoomBuffer*>& ys) {
    if (!bands_) {
      for (std::size_t i = 0; i < ws.size(); ++i) Gemv("plain", {ws[i]}, *normed_, *ys[i], i > 0);
      return;
    }
    std::string fn, mn;
    std::uint32_t tbits = 0, K = 0, rows = 0;
    std::vector<hrx_buffer_ref_t> b;
    for (std::size_t i = 0; i < ws.size(); ++i) {
      const auto* t = Find(ws[i]);
      Fmt f{};
      if (!FmtOf(static_cast<std::uint32_t>(t->type), &f)) throw LoomError("no GEMV format for " + ws[i]);
      const std::uint32_t k = static_cast<std::uint32_t>(t->dims[0]), M = static_cast<std::uint32_t>(t->dims[1]);
      if (i && k != K) throw LoomError("bands need one K");
      K = k;
      if (t->bytes != std::uint64_t{K} / f.qk * f.bb * M) throw LoomError("footprint mismatch on " + ws[i]);
      if (M % (kR * kW) || ys[i]->size < std::size_t{M} * 4) throw LoomError("band output: " + ws[i]);
      fn += std::string("_") + f.name;
      mn += "_" + std::to_string(M);
      tbits |= f.tables;
      rows += M;
      b.push_back(TRef(ws[i]));
    }
    for (int i = 0; i < 5; ++i)
      if (tbits & (1u << i)) b.push_back(tabs_[i]);
    b.push_back({normed_->handle, 0, std::size_t{K} * 4});
    for (std::size_t i = 0; i < ws.size(); ++i) {
      const auto* t = Find(ws[i]);
      b.push_back({ys[i]->handle, 0, static_cast<std::size_t>(t->dims[1]) * 4});
    }
    Dispatch(Load("gb" + fn + mn + "_" + std::to_string(K)), rows / (kR * kW), 1, 32 * kW, b);
  }
  // hidden += W x; with resnorm_ the same dispatch also writes normed_ = rmsnorm(hidden) * nw
  void Resid(const std::string& w, const LoomBuffer& x, const std::string& nw) {
    if (!resnorm_) {
      Gemv("resid", {w}, x, *hidden_);
      return;
    }
    Gemv("resid_norm", {w}, x, *hidden_, false, {TRef(nw), Ref(*normed_), Ref(*cnt_)});
  }
  void Rmsnorm(const LoomBuffer& x, const std::string& w, const LoomBuffer& out) {
    Dispatch(Load("rmsnorm"), 1, 1, 512, {Ref(x), TRef(w), Ref(out)});
  }
  // YAH_DEC_TRACE=1: at step 0, sync after every stage and print max |x| / NaNs
  void Tr(const char* what, std::uint32_t l, const LoomBuffer& b, std::size_t n) {
    if (!trace_ || cur_pos_ != 0) return;
    gpu_.Synchronize();
    std::vector<float> h(n);
    gpu_.D2H(b, h.data(), n * 4);
    double mx = 0;
    std::size_t nan = 0;
    for (float v : h) {
      if (v != v) ++nan;
      else mx = std::max(mx, static_cast<double>(std::fabs(v)));
    }
    std::fprintf(stderr, "trace l%-2u %-10s max %.4g nan %zu\n", l, what, mx, nan);
  }

  LoomDevice& gpu_;
  const core::Gguf& gguf_;
  const core::Qwen35Config& cfg_;
  std::string dir_;
  hrx_buffer_t weights_;
  std::size_t delta_;
  std::uint32_t T_ = 0, cur_pos_ = 0;
  bool trace_ = false, overlap_ = true, bands_ = true, resnorm_ = false;
  std::deque<LoomBuffer> keep_;   // stable addresses
  std::map<std::string, LoomExecutable> exes_;
  std::vector<hrx_buffer_ref_t> tabs_;
  LoomDecoderState st_;
  LoomBuffer *hidden_, *normed_, *qg_, *q_, *gate_, *kb_, *vb_, *aout_, *qkv_, *alpha_, *beta_, *convout_, *ssmout_,
      *ffnact_, *logits_, *sink_, *toks_, *posarr_, *eps_, *c32a_, *c32b_, *c16a_, *c16b_, *acc_, *ml_, *cnt_;
};

}  // namespace yah::model

#endif  // YAH_MODEL_LOOM_DECODER_HPP_
