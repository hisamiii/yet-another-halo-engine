// LoomDecoder: the GEMV-based single-token decode step on Loom through HRX.
//
// HAL set: tools/emit_decode.py. decode.txt "ctx T": the paged KV pools hold T rows.
// decode.txt "kv q KB VB": quantized KV in the prefill's kv8a16 / kv4a16 formats (gen_kvq.py).
// Per layer:
//   rmsnorm -> full attention: attn_q / attn_k / attn_v GEMV, unpack q|gate, QK norm + RoPE,
//                              append K / V to the paged pools, split-K attention + reduce (gate),
//                              attn_output GEMV += hidden
//           -> recurrent:      attn_qkv / attn_gate / ssm_alpha / ssm_beta GEMV,
//                              conv + DeltaNet in one kernel (gated norm inside), ssm_out GEMV += hidden
//   rmsnorm -> ffn_gate|ffn_up SwiGLU GEMV -> ffn_down GEMV += hidden
// Head: rmsnorm -> output GEMV -> argmax into a device token stream (decode.txt "amax": inside the output GEMV).
// MQ4 sets (decode.txt "rot"): the GEMV inputs are rotated; "normrot" / "swrot" do it inside the norm and the
// SwiGLU GEMV instead of separate drot_* dispatches.
// The next step's embedding kernel (IQ4_XS token_embd) reads that stream: steps queue back to back, no host round trip.
//
// The recurrent state and the KV pools are external (LoomDecoderState) and use the prefill's layouts.
// Use OwnState() for a prompt fed through decode, or bind the prefill's state (loom_forward_pp with YAH_GEN).
// Quantized KV: each layer's open 16-key V tile stays f16 until its 16th key arrives.
// So a bound state must start on a 16-key tile boundary (prefill runs are whole chunks).
#ifndef YAH_MODEL_LOOM_DECODER_HPP_
#define YAH_MODEL_LOOM_DECODER_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "model/generator.hpp"
#include "core/config.hpp"
#include "core/gguf.hpp"
#include "model/loom_runtime.hpp"

namespace yah::model {

struct LoomDecoderState {
  std::vector<hrx_buffer_ref_t> kpool, vtpool;  // per full-attention layer, T * 1024 f16 each
  // Quantized KV (decode.txt "kv q KB VB"), per full-attention layer; the f16 pools are then unused.
  // K codes, K scales, K channel means subtracted before quantizing (zeros in OwnState), V^T codes, V^T stats.
  std::vector<hrx_buffer_ref_t> kq, ks, km, vq, vs;
  hrx_buffer_ref_t ptab{};           // T / 256 i32 (logical page -> physical page)
  hrx_buffer_t convstate = nullptr;  // per recurrent layer si: si * 10240 * 4 f32
  hrx_buffer_t dstate = nullptr;     // per recurrent layer si: si * 48 * 128 * 128 f32
};

class LoomDecoder {
 public:
  static constexpr std::uint32_t kHidden = 5120, kFfn = 17408, kAttn = 6144, kQProj = 12288, kKv = 1024;
  static constexpr std::uint32_t kInner = 6144, kQkv = 10240, kHeadsV = 48, kTs = 48, kState = 128;
  static constexpr std::uint32_t kHeads = 24, kKvHeads = 4, kVocab = 248320;
  static constexpr std::uint32_t kConvState = kQkv * 4, kStateElems = kHeadsV * kState * kState;
  static constexpr std::uint32_t kR = 2, kW = 4;  // gen_gemv defaults: rows per wave, waves per workgroup

  LoomDecoder(LoomDevice& gpu, const core::Gguf& gguf, const core::Qwen35Config& cfg, std::string dir,
              const LoomWeights& weights)
      : gpu_(gpu), gguf_(gguf), cfg_(cfg), dir_(std::move(dir)), weights_(weights) {
    {
      const auto txt = ReadAll(dir_ + "/decode.txt");
      std::istringstream in(std::string(txt.begin(), txt.end()));
      std::string k;
      in >> k >> T_;
      if (k != "ctx" || T_ == 0 || T_ % 256) throw LoomError("bad decode.txt in " + dir_);
      // "rw <kind> R W": the GEMV launch geometry of each kind in this set.
      std::string kind;
      std::uint32_t r = 0, w = 0;
      while (in >> k >> kind >> r >> w) {
        if (k == "rw") rw_[kind] = {r, w};
        if (k == "grid") grid_[kind] = r;  // the exact grid each GEMV kernel was compiled for
        if (k == "kv") {
          kbits_ = r;
          vbits_ = w;
        }  // "kv q KB VB"
        if (k == "rot") rot_ = true;   // MQ4 model: GEMV inputs are rotated (drot_*)
        if (k == "chain") chain_ = true;       // every kernel waits on its producer's counter (tools/gen_chain.py)
        if (k == "normrot") normrot_ = true;   // rmsnorm + rotation in one kernel (drot_norm_*)
        if (k == "swrot") swrot_ = true;       // the SwiGLU GEMV rotates its output for ffn_down (gv_swiglurot_*)
        if (k == "amax") amax_ = true;         // the output GEMV takes the greedy argmax (gv_plainam_*)
        if (k == "latestate") late_ = true;    // the DeltaNet stores its state after its chain signal
        if (k == "verify") vt_ = r;            // T-token GEMVs (gv_*_t<T>) for Verify
        if (k == "mtp") mtp_ = true;           // the MTP layer's eh_proj GEMV
      }
      const auto ok = [](std::uint32_t b) { return b == 4 || b == 8; };
      if (quant() && !(ok(kbits_) && ok(vbits_))) throw LoomError("decode.txt: kv formats must both be 8 or 4");
    }
    const std::uint32_t npg = T_ / 256;
    if (!rot_) throw LoomError("decoder: not an MQ4 set (an emit_decode.py set from before 2026-10-20?)");
    // embed decodes Q8_0
    if (Find("token_embd.weight")->type != core::GgmlType::kQ8_0)
      throw LoomError("token_embd: its format does not match the decode set's embed (re-emit)");
    hidden_ = &Alloc(kHidden * 4);
    normed_ = &Alloc(kHidden * 4);
    qg_ = &Alloc(kQProj * 4);
    q_ = &Alloc(kAttn * 4);
    gate_ = &Alloc(kAttn * 4);
    kb_ = &Alloc(kKv * 4);
    vb_ = &Alloc(kKv * 4);
    aout_ = &Alloc(kAttn * 4);
    qkv_ = &Alloc(kQkv * 4);
    alpha_ = &Alloc(kTs * 4);
    beta_ = &Alloc(kTs * 4);
    ssmout_ = &Alloc(kInner * 4);
    ffnact_ = &Alloc(kFfn * 4);
    {
      const auto sg = ReadAll(dir_ + "/rot_signs.bin");
      rot_signs_ = &Alloc(sg.size());
      gpu_.H2D(*rot_signs_, sg.data(), sg.size());
    }
    rotx_ = &Alloc(kInner * 4);
    if (swrot_) swcnt_ = &Alloc(kFfn / 256 * 4);   // monotonic per-block counters, never reset
    logits_ = &Alloc(std::size_t{kVocab} * 4);
    if (amax_) amaxk_ = &Alloc(16);   // u64 max key + i32 workgroup count, both back to 0 after each step
    sink_ = &Alloc(4);
    sparams_ = &Alloc(32);
    pparams_ = &Alloc(16);
    toks_ = &Alloc((std::size_t{T_} + 1) * 4);
    posarr_ = &Alloc(std::size_t{T_} * 4);
    eps_ = &Alloc(4);
    c32a_ = &Alloc(kKv * 4);
    c32b_ = &Alloc(kKv * 4);
    c16a_ = &Alloc(kKv * 2);
    c16b_ = &Alloc(kKv * 2);
    acc_ = &Alloc(std::size_t{npg} * kHeads * 256 * 4);
    ml_ = &Alloc(std::size_t{npg} * kHeads * 2 * 4);
    if (quant())
      for (std::uint32_t i = 0; i < full_layers(); ++i) vopen_.push_back(Ref(Alloc(std::size_t{kKv} * 16 * 2)));
    const float e = 1.0e-6f;
    gpu_.H2D(*eps_, &e, 4);
    std::vector<std::int32_t> pv(T_);
    for (std::uint32_t i = 0; i < T_; ++i) pv[i] = static_cast<std::int32_t>(i);
    gpu_.H2D(*posarr_, pv.data(), pv.size() * 4);
    trace_ = std::getenv("YAH_DEC_TRACE") != nullptr;
    Load("sample");    // now, not in the first sampled request (a set emitted before sampling stops here)
    Load("penalty");
  }

  [[nodiscard]] std::uint32_t context() const { return T_; }
  // KV bits of this set: (16, 16) for fp16, else (8 | 4, 8 | 4).
  [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> kv_bits() const { return {kbits_, vbits_}; }
  [[nodiscard]] bool quant() const { return kbits_ != 16; }
  // Quantized V: layer ai's open 16-key tile ([1024 dims][16 keys] f16), for seeding after a prefill that ends mid-tile.
  [[nodiscard]] hrx_buffer_ref_t OpenTile(std::uint32_t ai) const { return vopen_.at(ai); }
  // The next Step may start at pos even inside a tile: the open tiles hold that tile's earlier keys (seeded).
  void ResumeAt(std::uint32_t pos) { next_pos_ = pos; }
  // The conv state the next step reads (the ping-pong buffer the last step wrote); the DeltaNet state is in place.
  [[nodiscard]] hrx_buffer_ref_t conv_state() const {
    return {cs_[cs_cur_], 0, std::size_t{recurrent_layers()} * kConvState * 4};
  }
  // Quantized pool bytes per layer (gen_kvq.py layouts).
  std::size_t KqBytes() const { return std::size_t{T_} * (kbits_ == 4 ? 512 : 1024); }
  std::size_t KsBytes() const { return std::size_t{T_} * (kbits_ == 4 ? 128 : 32); }
  std::size_t VqBytes() const { return std::size_t{T_} * (vbits_ == 4 ? 512 : 1024); }
  std::size_t VsBytes() const { return std::size_t{T_} * 256; }
  std::uint32_t full_layers() const { return cfg_.main_block_count() / cfg_.full_attention_interval; }
  std::uint32_t recurrent_layers() const { return cfg_.main_block_count() - full_layers(); }

  // The decoder's own pools / page table (identity) / zeroed recurrent state.
  LoomDecoderState OwnState() {
    LoomDecoderState st;
    const std::uint32_t npg = T_ / 256;
    const std::size_t poolb = std::size_t{T_} * kKv * 2;
    for (std::uint32_t i = 0; i < full_layers(); ++i) {
      if (quant()) {
        st.kq.push_back(Ref(Alloc(KqBytes())));
        st.ks.push_back(Ref(Alloc(KsBytes())));
        st.km.push_back(Ref(Alloc(4096)));
        st.vq.push_back(Ref(Alloc(VqBytes())));
        st.vs.push_back(Ref(Alloc(VsBytes())));
      } else {
        st.kpool.push_back(Ref(Alloc(poolb)));
        st.vtpool.push_back(Ref(Alloc(poolb)));
      }
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
    const std::uint32_t nf = full_layers();
    if (quant()) {
      if (st.kq.size() != nf || st.ks.size() != nf || st.km.size() != nf || st.vq.size() != nf || st.vs.size() != nf)
        throw LoomError("decoder: quantized K / V pools per full-attention layer expected");
      for (std::uint32_t i = 0; i < nf; ++i)
        if (st.kq[i].length < KqBytes() || st.ks[i].length < KsBytes() || st.km[i].length < 4096 ||
            st.vq[i].length < VqBytes() || st.vs[i].length < VsBytes())
          throw LoomError("decoder: quantized KV pool smaller than the decode set's context " + std::to_string(T_));
    } else {
      if (st.kpool.size() != nf || st.vtpool.size() != nf)
        throw LoomError("decoder: one K and one V^T pool per full-attention layer expected");
      for (std::uint32_t i = 0; i < nf; ++i)
        if (st.kpool[i].length < poolb || st.vtpool[i].length < poolb)
          throw LoomError("decoder: KV pool smaller than the decode set's context " + std::to_string(T_));
    }
    if (st.ptab.length < std::size_t{T_ / 256} * 4) throw LoomError("decoder: page table too small");
    st_ = st;
    next_pos_ = kAnyTile;
    cs_[0] = st.convstate;
    if (!cs_[1]) cs_[1] = Alloc(std::size_t{recurrent_layers()} * kConvState * 4).handle;
    cs_cur_ = 0;
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
  // Token position pos of the device token stream (a step's argmax target).
  [[nodiscard]] hrx_buffer_ref_t TokenRef(std::uint32_t pos) const { return {toks_->handle, std::size_t{pos} * 4, 4}; }
  void CopyLogits(float* host) { gpu_.D2H(*logits_, host, std::size_t{kVocab} * 4); }

  // Queues the step at position pos. It reads toks[pos]; its argmax goes to toks[pos + 1] if pos + 1 >= keep_from.
  // Prompt positions (pos + 1 < keep_from) write the argmax to a sink.
  void Step(std::uint32_t pos, std::uint32_t keep_from) {
    if (pos >= T_) throw LoomError("decoder: position past the set's context");
    if (st_.kpool.empty() && st_.kq.empty()) throw LoomError("decoder: no state bound");
    // The open V tile holds this tile's earlier keys only if the steps were contiguous.
    if (quant() && pos % 16 && pos != next_pos_)
      throw LoomError("decoder: quantized KV steps must be contiguous from a 16-key tile boundary");
    next_pos_ = pos + 1;
    cur_pos_ = pos;
    // "latestate": no chained consumer orders the DeltaNet's state stores (gen_decode_misc LATE_STATE), so the step
    // starts after every dispatch so far completed, the previous step's state stores with them.
    if (late_) gpu_.Order();
    const hrx_buffer_ref_t dposr{posarr_->handle, std::size_t{pos} * 4, 4};
    Dispatch(Load("embed"), 1, 1, kHidden / 16,
             {TRef("token_embd.weight"), {toks_->handle, std::size_t{pos} * 4, 4}, Ref(*hidden_)});
    const std::uint32_t nl = cfg_.main_block_count();
    for (std::uint32_t l = 0; l < nl; ++l) {
      const std::string pre = "blk." + std::to_string(l) + ".";
      NormRotate(*hidden_, pre + "attn_norm.weight", pre + (cfg_.IsFullAttention(l) ? "attn_q" : "attn_qkv"));
      Tr("in", l, *hidden_, kHidden);
      if (cfg_.IsFullAttention(l)) {
        const std::uint32_t ai = l / cfg_.full_attention_interval;
        Project({pre + "attn_q.weight", pre + "attn_k.weight", pre + "attn_v.weight"}, {qg_, kb_, vb_});
        Dispatch(Load("unpack"), kHeads, 1, 256, {Ref(*qg_), Ref(*q_), Ref(*gate_)});
        Dispatch(Load("rope"), kHeads + kKvHeads, 1, 256,
                 {Ref(*q_), Ref(*kb_), Ref(*vb_), TRef(pre + "attn_q_norm.weight"), TRef(pre + "attn_k_norm.weight"),
                  Ref(*q_), Ref(*kb_), Ref(*c32a_), Ref(*c32b_), Ref(*c16a_), Ref(*c16b_), dposr, Ref(*eps_)});
        if (quant()) {
          Dispatch(Load("dattn_kappend_q"), 1, 1, 128,
                   {Ref(*kb_), st_.km[ai], st_.kq[ai], st_.ks[ai], st_.ptab, dposr});
          if (!chain_) gpu_.NoBarrierNext();  // independent of the K append
          Dispatch(Load("dattn_vappend_q"), kKvHeads, 1, 256,
                   {Ref(*vb_), vopen_[ai], st_.vq[ai], st_.vs[ai], st_.ptab, dposr}, true);
          Dispatch(Load("dattn_part_q"), kKvHeads, pos / 256 + 1, 256,
                   {Ref(*q_), st_.kq[ai], st_.ks[ai], st_.vq[ai], st_.vs[ai], vopen_[ai], st_.ptab, dposr, Ref(*acc_),
                    Ref(*ml_)});
        } else {
          Dispatch(Load("dattn_kvappend"), kKvHeads, 1, 256,
                   {Ref(*kb_), Ref(*vb_), st_.kpool[ai], st_.vtpool[ai], st_.ptab, dposr});
          Dispatch(Load("dattn_part"), kKvHeads, pos / 256 + 1, 256,
                   {Ref(*q_), st_.kpool[ai], st_.vtpool[ai], st_.ptab, dposr, Ref(*acc_), Ref(*ml_)});
        }
        Dispatch(Load("dattn_reduce"), kHeads, 1, 256, {Ref(*acc_), Ref(*ml_), Ref(*gate_), dposr, Ref(*aout_)});
        Tr("attn", l, *aout_, kAttn);
        Rotate(*aout_, pre + "attn_output");
        Gemv("resid", {pre + "attn_output.weight"}, *aout_, *hidden_);
      } else {
        const std::uint32_t si = l - l / cfg_.full_attention_interval;
        Project({pre + "attn_qkv.weight", pre + "attn_gate.weight", pre + "ssm_alpha.weight", pre + "ssm_beta.weight"},
                {qkv_, gate_, alpha_, beta_});
        const hrx_buffer_ref_t dst{st_.dstate, std::size_t{si} * kStateElems * 4, std::size_t{kStateElems} * 4};
        // Conv state ping-pong: this step reads cs_[cs_cur_] and writes the other one.
        const std::size_t co = std::size_t{si} * kConvState * 4, cb = std::size_t{kConvState} * 4;
        Dispatch(Load("deltanet_conv"), kHeadsV, 1, 512,
                 {{qkv_->handle, 0, std::size_t{kQkv} * 4},
                  TRef(pre + "ssm_conv1d.weight"),
                  {cs_[cs_cur_], co, cb},
                  {cs_[1 - cs_cur_], co, cb},
                  dst,
                  Ref(*alpha_),
                  Ref(*beta_),
                  TRef(pre + "ssm_a"),
                  TRef(pre + "ssm_dt.bias"),
                  TRef(pre + "ssm_norm.weight"),
                  Ref(*gate_),
                  Ref(*ssmout_)});
        Tr("ssm", l, *ssmout_, kInner);
        Rotate(*ssmout_, pre + "ssm_out");
        Gemv("resid", {pre + "ssm_out.weight"}, *rotx_, *hidden_);
      }
      NormRotate(*hidden_, pre + "post_attention_norm.weight", pre + "ffn_gate");
      Gemv("swiglu", {pre + "ffn_gate.weight", pre + "ffn_up.weight"}, *normed_, *ffnact_,
           swrot_ ? pre + "ffn_down" : std::string());
      Tr("ffnact", l, *ffnact_, kFfn);
      if (!swrot_) Rotate(*ffnact_, pre + "ffn_down");
      Gemv("resid", {pre + "ffn_down.weight"}, *ffnact_, *hidden_);
    }
    if (normrot_) {
      Dispatch(Load("drot_norm_head"), kHidden / 1024, 1, 512, {Ref(*hidden_), TRef("output_norm.weight"),
                                                                Ref(*rot_signs_), Ref(*normed_)});
    } else {
      Rmsnorm(*hidden_, "output_norm.weight", *normed_);
      Dispatch(Load("drot_head"), kHidden / 1024, 1, 64, {Ref(*normed_), Ref(*rot_signs_)});
    }
    const bool sample = sampling_ && pos + 1 >= keep_from;
    const bool penalize = penalized_ && pos + 1 >= keep_from;
    const hrx_buffer_ref_t tok = pos + 1 >= keep_from ? TokenRef(pos + 1) : Ref(*sink_);
    Gemv("plain", {"output.weight"}, *normed_, *logits_, {}, amax_ && !sample && !penalize ? &tok : nullptr);
    Tr("logits", 99, *logits_, kVocab);
    cs_cur_ = 1 - cs_cur_;
    if (penalize) Penalize(Ref(*logits_), pos);
    if (sample)
      Sample(Ref(*logits_), pos, TokenRef(pos + 1));
    else if (!amax_ || penalize)
      Dispatch(Load("argmax"), 1, 1, 1024, {Ref(*logits_), tok});
  }

  // Speculative verify (decode.txt "verify - T 0"): the steps at pos .. pos + n - 1 (n <= verify_tokens()) in one
  // pass of T-token GEMVs, reading toks[pos ..]; the greedy token after each into VerifyTokens()[i]. Every token's
  // arithmetic is the plain Step's (the per-token kernels are Step's own, each GEMV row sums as the T = 1 one), so the
  // tokens are bit for bit what n Steps would pick. Accept(m) then keeps the state after token m: the conv states
  // ping-pong through a ring, the DeltaNet state after each token but the last is copied to a ring slot, and a
  // quantized V open tile a rejected token may have moved past is restored and refilled. Greedy only.
  [[nodiscard]] std::uint32_t verify_tokens() const { return vt_; }
  [[nodiscard]] hrx_buffer_ref_t VerifyTokens() const { return Ref(*vtok_); }
  // The greedy tokens of the last Verify (synchronous read).
  std::vector<std::uint32_t> VerifyResult(std::uint32_t n) {
    std::vector<std::uint32_t> g(n);
    gpu_.D2H(*vtok_, g.data(), std::size_t{n} * 4);
    return g;
  }
  void CopyVerifyLogits(std::uint32_t i, float* host) {
    gpu_.D2H(*lg_n_, host, std::size_t{kVocab} * 4, std::size_t{i} * kVocab * 4);
  }
  // The final hidden state (before output_norm) after verify token i: the MTP layer's input.
  [[nodiscard]] hrx_buffer_ref_t VerifyHidden(std::uint32_t i) const {
    return {hid_n_->handle, std::size_t{i} * kHidden * 4, kHidden * 4};
  }
  void Verify(std::uint32_t pos, std::uint32_t n) {
    if (!vt_ || n == 0 || n > vt_) throw LoomError("decoder: verify of " + std::to_string(n) + " tokens");
    if (pos + n > T_) throw LoomError("decoder: position past the set's context");
    if (st_.kpool.empty() && st_.kq.empty()) throw LoomError("decoder: no state bound");
    if (quant() && pos % 16 && pos != next_pos_)
      throw LoomError("decoder: quantized KV steps must be contiguous from a 16-key tile boundary");
    if (!hid_n_) AllocVerify();
    vpos_ = pos, vn_ = n;
    next_pos_ = pos + n;
    cur_pos_ = pos;
    if (late_) gpu_.Order();
    // a token past a 16-key boundary refills the open V tiles: keep them for a rollback (Accept)
    vcross_ = quant() && pos / 16 != (pos + n - 1) / 16;
    if (vcross_)
      for (std::uint32_t i = 0; i < full_layers(); ++i) gpu_.Copy(vopen_[i], Slice(*vopen_save_, i, vopen_[i].length));
    const auto R = [](const LoomBuffer* b, std::uint32_t i, std::size_t elems) -> hrx_buffer_ref_t {
      return {b->handle, std::size_t{i} * elems * 4, elems * 4};
    };
    const auto dpos = [&](std::uint32_t i) -> hrx_buffer_ref_t {
      return {posarr_->handle, std::size_t{pos + i} * 4, 4};
    };
    for (std::uint32_t i = 0; i < n; ++i)
      Dispatch(Load("embed"), 1, 1, kHidden / 16,
               {TRef("token_embd.weight"), {toks_->handle, std::size_t{pos + i} * 4, 4}, R(hid_n_, i, kHidden)});
    const std::uint32_t nl = cfg_.main_block_count();
    for (std::uint32_t l = 0; l < nl; ++l) {
      const std::string pre = "blk." + std::to_string(l) + ".";
      const bool full = cfg_.IsFullAttention(l);
      for (std::uint32_t i = 0; i < n; ++i)
        NormRotate(R(hid_n_, i, kHidden), pre + "attn_norm.weight", pre + (full ? "attn_q" : "attn_qkv"),
                   R(nrm_n_, i, kHidden));
      if (full) {
        const std::uint32_t ai = l / cfg_.full_attention_interval;
        const LoomBuffer* vs = vsave_[ai];   // this layer's V rows, kept for a refill (Accept)
        GemvT("plain", {pre + "attn_q.weight"}, *nrm_n_, *qg_n_);
        GemvT("plain", {pre + "attn_k.weight"}, *nrm_n_, *k_n_);
        GemvT("plain", {pre + "attn_v.weight"}, *nrm_n_, *vs);
        for (std::uint32_t i = 0; i < n; ++i) {
          const hrx_buffer_ref_t kb = R(k_n_, i, kKv), vb = R(vs, i, kKv);
          Dispatch(Load("unpack"), kHeads, 1, 256, {R(qg_n_, i, kQProj), Ref(*q_), Ref(*gate_)});
          Dispatch(Load("rope"), kHeads + kKvHeads, 1, 256,
                   {Ref(*q_), kb, vb, TRef(pre + "attn_q_norm.weight"), TRef(pre + "attn_k_norm.weight"), Ref(*q_), kb,
                    Ref(*c32a_), Ref(*c32b_), Ref(*c16a_), Ref(*c16b_), dpos(i), Ref(*eps_)});
          if (quant()) {
            Dispatch(Load("dattn_kappend_q"), 1, 1, 128, {kb, st_.km[ai], st_.kq[ai], st_.ks[ai], st_.ptab, dpos(i)});
            if (!chain_) gpu_.NoBarrierNext();
            Dispatch(Load("dattn_vappend_q"), kKvHeads, 1, 256,
                     {vb, vopen_[ai], st_.vq[ai], st_.vs[ai], st_.ptab, dpos(i)}, true);
            Dispatch(Load("dattn_part_q"), kKvHeads, (pos + i) / 256 + 1, 256,
                     {Ref(*q_), st_.kq[ai], st_.ks[ai], st_.vq[ai], st_.vs[ai], vopen_[ai], st_.ptab, dpos(i),
                      Ref(*acc_), Ref(*ml_)});
          } else {
            Dispatch(Load("dattn_kvappend"), kKvHeads, 1, 256, {kb, vb, st_.kpool[ai], st_.vtpool[ai], st_.ptab, dpos(i)});
            Dispatch(Load("dattn_part"), kKvHeads, (pos + i) / 256 + 1, 256,
                     {Ref(*q_), st_.kpool[ai], st_.vtpool[ai], st_.ptab, dpos(i), Ref(*acc_), Ref(*ml_)});
          }
          Dispatch(Load("dattn_reduce"), kHeads, 1, 256, {Ref(*acc_), Ref(*ml_), Ref(*gate_), dpos(i), R(ao_n_, i, kAttn)});
          Rotate(R(ao_n_, i, kAttn), pre + "attn_output", {});
        }
        GemvT("resid", {pre + "attn_output.weight"}, *ao_n_, *hid_n_);
      } else {
        const std::uint32_t si = l - l / cfg_.full_attention_interval;
        GemvT("plain", {pre + "attn_qkv.weight"}, *nrm_n_, *qkv_n_);
        GemvT("plain", {pre + "attn_gate.weight"}, *nrm_n_, *gate_n_);
        GemvT("plain", {pre + "ssm_alpha.weight"}, *nrm_n_, *alpha_n_);
        GemvT("plain", {pre + "ssm_beta.weight"}, *nrm_n_, *beta_n_);
        const std::size_t so = std::size_t{si} * kStateElems * 4, sb = std::size_t{kStateElems} * 4;
        const std::size_t co = std::size_t{si} * kConvState * 4, cb = std::size_t{kConvState} * 4;
        for (std::uint32_t i = 0; i < n; ++i) {
          // conv state: ring slot i (slot 0 the current one) to slot i + 1
          Dispatch(Load("deltanet_conv"), kHeadsV, 1, 512,
                   {R(qkv_n_, i, kQkv), TRef(pre + "ssm_conv1d.weight"), {ConvRing(i), co, cb}, {ConvRing(i + 1), co, cb},
                    {st_.dstate, so, sb}, R(alpha_n_, i, kTs), R(beta_n_, i, kTs), TRef(pre + "ssm_a"),
                    TRef(pre + "ssm_dt.bias"), TRef(pre + "ssm_norm.weight"), R(gate_n_, i, kAttn),
                    R(so_n_, i, kInner)});
          if (i + 1 < n) gpu_.Copy({st_.dstate, so, sb}, {dring_[i]->handle, so, sb});
          Rotate(R(so_n_, i, kInner), pre + "ssm_out", R(rx_n_, i, kInner));
        }
        GemvT("resid", {pre + "ssm_out.weight"}, *rx_n_, *hid_n_);
      }
      for (std::uint32_t i = 0; i < n; ++i)
        NormRotate(R(hid_n_, i, kHidden), pre + "post_attention_norm.weight", pre + "ffn_gate", R(nrm_n_, i, kHidden));
      GemvT("swiglu", {pre + "ffn_gate.weight", pre + "ffn_up.weight"}, *nrm_n_, *act_n_);
      for (std::uint32_t i = 0; i < n; ++i) Rotate(R(act_n_, i, kFfn), pre + "ffn_down", {});
      GemvT("resid", {pre + "ffn_down.weight"}, *act_n_, *hid_n_);
    }
    for (std::uint32_t i = 0; i < n; ++i) {
      if (normrot_) {
        Dispatch(Load("drot_norm_head"), kHidden / 1024, 1, 512,
                 {R(hid_n_, i, kHidden), TRef("output_norm.weight"), Ref(*rot_signs_), R(nrm_n_, i, kHidden)});
      } else {
        Dispatch(Load("rmsnorm"), 1, 1, 512, {R(hid_n_, i, kHidden), TRef("output_norm.weight"), R(nrm_n_, i, kHidden)});
        Dispatch(Load("drot_head"), kHidden / 1024, 1, 64, {R(nrm_n_, i, kHidden), Ref(*rot_signs_)});
      }
    }
    GemvT("plain", {"output.weight"}, *nrm_n_, *lg_n_);
    for (std::uint32_t i = 0; i < n; ++i)
      Dispatch(Load("argmax"), 1, 1, 1024,
               {{lg_n_->handle, std::size_t{i} * kVocab * 4, std::size_t{kVocab} * 4}, Slice(*vtok_, i, 4)});
  }
  // MTP drafting (decode.txt "mtp", a model with the nextn layer blk.<main blocks>.nextn.*): one pass of that layer
  // predicts the token after next from a token and the hidden state before it (vLLM's Qwen3-Next MTP: rmsnorm'd
  // embedding and hidden concatenated, eh_proj, one full-attention block, shared_head_norm, the shared output head).
  // Its KV is its own fp16 pool indexed by a draft position that starts at 0 (MtpBegin): RoPE sees only relative
  // offsets, so this equals the true positions, and the layer never attends to prompt rows it did not see.
  [[nodiscard]] bool has_mtp() const { return mtp_ && cfg_.nextn_predict_layers > 0; }
  void MtpBegin() {
    if (!mtp_k_) AllocMtp();
  }
  // The hidden input of the next Draft: the main model's final hidden h (before output_norm), normalized as the head
  // sees it. Step leaves its own in hidden_ (MainHidden()), Verify one per token (VerifyHidden).
  void MtpHidden(const hrx_buffer_ref_t& h) {
    Dispatch(Load("rmsnorm"), 1, 1, 512, {h, TRef("output_norm.weight"), Ref(*mh_)});
  }
  [[nodiscard]] hrx_buffer_ref_t MainHidden() const { return Ref(*hidden_); }
  // Draft at position q (draft position v): reads toks[q], writes the predicted token to toks[q + 1], and leaves its
  // own output (after shared_head_norm) as the next Draft's hidden input.
  void Draft(std::uint32_t q, std::uint32_t v) {
    if (q + 1 > T_ || v >= T_) throw LoomError("decoder: draft past the set's context");
    const std::string pre = "blk." + std::to_string(cfg_.main_block_count()) + ".";
    const hrx_buffer_ref_t dposr{posarr_->handle, std::size_t{v} * 4, 4};
    Dispatch(Load("embed"), 1, 1, kHidden / 16,
             {TRef("token_embd.weight"), {toks_->handle, std::size_t{q} * 4, 4}, Ref(*normed_)});
    Dispatch(Load("rmsnorm"), 1, 1, 512, {Ref(*normed_), TRef(pre + "nextn.enorm.weight"), Slice(*cat_, 0, kHidden * 4)});
    Dispatch(Load("rmsnorm"), 1, 1, 512, {Ref(*mh_), TRef(pre + "nextn.hnorm.weight"), Slice(*cat_, 1, kHidden * 4)});
    Rotate(Ref(*cat_), pre + "nextn.eh_proj", {});
    Gemv("plain", {pre + "nextn.eh_proj.weight"}, *cat_, *mx_);
    NormRotate(*mx_, pre + "attn_norm.weight", pre + "attn_q");
    Project({pre + "attn_q.weight", pre + "attn_k.weight", pre + "attn_v.weight"}, {qg_, kb_, vb_});
    Dispatch(Load("unpack"), kHeads, 1, 256, {Ref(*qg_), Ref(*q_), Ref(*gate_)});
    Dispatch(Load("rope"), kHeads + kKvHeads, 1, 256,
             {Ref(*q_), Ref(*kb_), Ref(*vb_), TRef(pre + "attn_q_norm.weight"), TRef(pre + "attn_k_norm.weight"),
              Ref(*q_), Ref(*kb_), Ref(*c32a_), Ref(*c32b_), Ref(*c16a_), Ref(*c16b_), dposr, Ref(*eps_)});
    Dispatch(Load("dattn_kvappend"), kKvHeads, 1, 256, {Ref(*kb_), Ref(*vb_), Ref(*mtp_k_), Ref(*mtp_v_), Ref(*mtp_pt_), dposr});
    Dispatch(Load("dattn_part"), kKvHeads, v / 256 + 1, 256,
             {Ref(*q_), Ref(*mtp_k_), Ref(*mtp_v_), Ref(*mtp_pt_), dposr, Ref(*acc_), Ref(*ml_)});
    Dispatch(Load("dattn_reduce"), kHeads, 1, 256, {Ref(*acc_), Ref(*ml_), Ref(*gate_), dposr, Ref(*aout_)});
    Rotate(*aout_, pre + "attn_output");
    Gemv("resid", {pre + "attn_output.weight"}, *aout_, *mx_);
    NormRotate(*mx_, pre + "post_attention_norm.weight", pre + "ffn_gate");
    Gemv("swiglu", {pre + "ffn_gate.weight", pre + "ffn_up.weight"}, *normed_, *ffnact_,
         swrot_ ? pre + "ffn_down" : std::string());
    if (!swrot_) Rotate(*ffnact_, pre + "ffn_down");
    Gemv("resid", {pre + "ffn_down.weight"}, *ffnact_, *mx_);
    Dispatch(Load("rmsnorm"), 1, 1, 512, {Ref(*mx_), TRef(pre + "nextn.shared_head_norm.weight"), Ref(*mh_)});
    if (normrot_) {
      Dispatch(Load("drot_norm_head"), kHidden / 1024, 1, 512,
               {Ref(*mx_), TRef(pre + "nextn.shared_head_norm.weight"), Ref(*rot_signs_), Ref(*normed_)});
    } else {
      gpu_.Copy(Ref(*mh_), Ref(*normed_));
      Dispatch(Load("drot_head"), kHidden / 1024, 1, 64, {Ref(*normed_), Ref(*rot_signs_)});
    }
    const hrx_buffer_ref_t tok = TokenRef(q + 1);
    Gemv("plain", {"output.weight"}, *normed_, *logits_, {}, amax_ ? &tok : nullptr);
    if (!amax_) Dispatch(Load("argmax"), 1, 1, 1024, {Ref(*logits_), tok});
  }

  // Keeps the state after verify token m (0 <= m < n of the last Verify): the next step or verify is at vpos + m + 1.
  void Accept(std::uint32_t m) {
    if (m >= vn_) throw LoomError("decoder: accept past the verified tokens");
    const std::uint32_t nf = full_layers(), nr = recurrent_layers();
    gpu_.Copy({ConvRing(m + 1), 0, std::size_t{nr} * kConvState * 4}, {cs_[cs_cur_], 0, std::size_t{nr} * kConvState * 4});
    if (m + 1 < vn_) {
      gpu_.Copy(Ref(*dring_[m]), {st_.dstate, 0, std::size_t{nr} * kStateElems * 4});
      // the open V tiles: a rejected token past a tile boundary started a new tile over the accepted tokens' one
      if (vcross_ && (vpos_ + m) / 16 != (vpos_ + vn_ - 1) / 16) {
        for (std::uint32_t ai = 0; ai < nf; ++ai) {
          gpu_.Copy(Slice(*vopen_save_, ai, vopen_[ai].length), vopen_[ai]);
          for (std::uint32_t i = 0; i <= m; ++i)
            if ((vpos_ + i) / 16 == (vpos_ + m) / 16)   // the accepted tokens of the tile left open
              Dispatch(Load("dattn_vappend_q"), kKvHeads, 1, 256,
                       {{vsave_[ai]->handle, std::size_t{i} * kKv * 4, kKv * 4}, vopen_[ai], st_.vq[ai], st_.vs[ai],
                        st_.ptab, {posarr_->handle, std::size_t{vpos_ + i} * 4, 4}});
        }
      }
    }
    next_pos_ = vpos_ + m + 1;
    vn_ = 0;
  }

  // From now on the steps whose token is kept sample it (gen_decode_misc.gen_sample: softmax(logits / t) cut by top_k,
  // top_p and min_p, an exact draw seeded by seed and the step's position) instead of taking the argmax; t <= 0:
  // argmax. With penalties (gen_penalty) the logits first get them from the token stream, which must hold the prompt
  // (SetTokens). In stream order, so a run of steps queued before keeps its mode.
  void SetSampling(const SamplingParams& s, std::uint64_t seed) {
    penalized_ = s.penalized();
    if (penalized_) {
      std::uint32_t q[4];
      std::memcpy(&q[0], &s.presence_penalty, 4), std::memcpy(&q[1], &s.frequency_penalty, 4);
      std::memcpy(&q[2], &s.repetition_penalty, 4);
      q[3] = s.penalty_last_n;
      gpu_.Update(*pparams_, q, sizeof q);
    }
    sampling_ = s.temperature > 0.0f;
    if (!sampling_) return;
    std::uint32_t p[8] = {};
    std::memcpy(&p[0], &s.temperature, 4), std::memcpy(&p[1], &s.top_p, 4);
    p[2] = static_cast<std::uint32_t>(seed), p[3] = static_cast<std::uint32_t>(seed >> 32);
    p[4] = s.top_k;
    std::memcpy(&p[5], &s.min_p, 4);
    gpu_.Update(*sparams_, p, sizeof p);
  }
  [[nodiscard]] bool penalized() const { return penalized_; }
  // The penalties (SetSampling) on logits produced by the step at position at, from toks[at + 1 - last_n .. at].
  void Penalize(const hrx_buffer_ref_t& logits, std::uint32_t at) {
    if (at >= T_) throw LoomError("decoder: position past the set's context");
    Dispatch(Load("penalty"), 1, 1, 1024,
             {logits, Ref(*toks_), Ref(*pparams_), {posarr_->handle, std::size_t{at} * 4, 4}});
  }
  // Samples a token from logits (SetSampling's parameters) into dst; at: the position of the step that produced the
  // logits (the draw's counter).
  void Sample(const hrx_buffer_ref_t& logits, std::uint32_t at, const hrx_buffer_ref_t& dst) {
    if (at >= T_) throw LoomError("decoder: position past the set's context");
    Dispatch(Load("sample"), 1, 1, 1024, {logits, Ref(*sparams_), {posarr_->handle, std::size_t{at} * 4, 4}, dst});
  }

 private:
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
  // A GEMV weight: MQ4G256 (136 B per 256), footprint checked.
  const core::TensorInfo* Mq4(const std::string& n) const {
    const auto* t = Find(n);
    if (t->type != core::GgmlType::kMQ4G256 || t->bytes != std::uint64_t(t->dims[0]) / 256 * 136 * t->dims[1])
      throw LoomError("decoder: " + n + " is not an MQ4 weight of its shape");
    return t;
  }
  hrx_buffer_ref_t TRef(const std::string& n) const {
    const auto* t = Find(n);
    return weights_.Ref(*t);
  }
  LoomExecutable& Load(const std::string& name) {
    auto it = exes_.find(name);
    if (it == exes_.end()) it = exes_.emplace(name, gpu_.Load(dir_ + "/" + name + ".hal")).first;
    return it->second;
  }
  // Workgroup size and binding count must match the compiled export.
  // Example: argmax is a 1024-lane kernel; with 32 lanes its per-wave LDS slots stay unwritten and it returns garbage.
  // join: a chained kernel that runs beside the previous one (same producer).
  void Dispatch(LoomExecutable& e, std::uint32_t gx, std::uint32_t gy, std::uint32_t wg,
                const std::vector<hrx_buffer_ref_t>& b, bool join = false) {
    const std::uint32_t cw = e.WorkgroupSize(0), cb = e.BindingCount(0);
    if ((cw && cw != wg) || (cb && cb != b.size() + chain_))
      throw LoomError("dispatch mismatch: workgroup " + std::to_string(wg) + " vs compiled " + std::to_string(cw) +
                      ", bindings " + std::to_string(b.size()) + " vs " + std::to_string(cb));
    if (chain_)
      gpu_.ChainDispatch(e, LoomDevice::Config(gx, gy, 1, wg, 1, 1), b, join);
    else
      gpu_.Dispatch(e, 0, LoomDevice::Config(gx, gy, 1, wg, 1, 1), nullptr, 0, b.data(), b.size());
  }
  // GEMV kernel names and weight footprints follow tools/gen_gemv.py.
  // rot_w: the GEMV also rotates y for weight rot_w's GEMV (gen_gemv rot; swiglu only)
  // rot_w: the SwiGLU output rotated for that weight (gv_swiglurot_*); tok: the argmax of y stored there (gv_plainam_*).
  void Gemv(const char* kind, const std::vector<std::string>& ws, const LoomBuffer& x, const LoomBuffer& y,
            const std::string& rot_w = {}, const hrx_buffer_ref_t* tok = nullptr) {
    std::string name = std::string("gv_") + kind + (rot_w.empty() ? "" : "rot") + (tok ? "am" : "");
    std::uint32_t M = 0, K = 0;
    std::vector<hrx_buffer_ref_t> b;
    for (const auto& w : ws) {
      const auto* t = Mq4(w);
      K = static_cast<std::uint32_t>(t->dims[0]);
      M = static_cast<std::uint32_t>(t->dims[1]);
      name += "_mq4";
      b.push_back(TRef(w));
    }
    name += "_" + std::to_string(M) + "_" + std::to_string(K);
    if (x.size < std::size_t{K} * 4 || y.size < std::size_t{M} * 4) throw LoomError("GEMV operand too small: " + name);
    b.push_back({x.handle, 0, std::size_t{K} * 4});
    b.push_back({y.handle, 0, std::size_t{M} * 4});
    if (!rot_w.empty()) {
      if (M != kFfn) throw LoomError("gv rot: only the SwiGLU output");
      b.push_back(TRef(rot_w + ".awq"));
      b.push_back(Ref(*rot_signs_));
      b.push_back(Ref(*swcnt_));
    }
    if (tok) {
      b.push_back(Ref(*amaxk_));
      b.push_back(*tok);
    }
    LoomExecutable& exe = Load(name);
    auto [R, W] = Rw(kind);
    const std::uint32_t gx = M / (R * W);
    CheckGrid(name, gx);
    Dispatch(exe, gx, 1, 32 * W, b);
  }
  // A layer's input projections of normed_: one band-fused GEMV (gen_gemv gen_bands).
  void Project(const std::vector<std::string>& ws, const std::vector<LoomBuffer*>& ys) {
    std::string fn, mn;
    std::uint32_t K = 0, rows = 0;
    std::vector<hrx_buffer_ref_t> b;
    for (std::size_t i = 0; i < ws.size(); ++i) {
      const auto* t = Mq4(ws[i]);
      const std::uint32_t k = static_cast<std::uint32_t>(t->dims[0]), M = static_cast<std::uint32_t>(t->dims[1]);
      if (i && k != K) throw LoomError("bands need one K");
      K = k;
      if (M % (Rw("bands").first * Rw("bands").second) || ys[i]->size < std::size_t{M} * 4)
        throw LoomError("band output: " + ws[i]);
      fn += "_mq4";
      mn += "_" + std::to_string(M);
      rows += M;
      b.push_back(TRef(ws[i]));
    }
    b.push_back({normed_->handle, 0, std::size_t{K} * 4});
    for (std::size_t i = 0; i < ws.size(); ++i) {
      const auto* t = Find(ws[i]);
      b.push_back({ys[i]->handle, 0, static_cast<std::size_t>(t->dims[1]) * 4});
    }
    auto [R, W] = Rw("bands");
    const std::string bn = "gb" + fn + mn + "_" + std::to_string(K);
    CheckGrid(bn, rows / (R * W));
    Dispatch(Load(bn), rows / (R * W), 1, 32 * W, b);
  }
  // (R, W) of a GEMV kind in this set (decode.txt).
  std::pair<std::uint32_t, std::uint32_t> Rw(const std::string& kind) const {
    auto it = rw_.find(kind);
    return it == rw_.end() ? std::make_pair(kR, kW) : it->second;
  }
  // Loom drops clamps it proves redundant from the launch grid: launch exactly the grid recorded in decode.txt.
  void CheckGrid(const std::string& name, std::uint32_t gx) const {
    auto it = grid_.find(name);
    if (it == grid_.end()) throw LoomError("decode set has no grid record for " + name + " (re-emit)");
    if (it->second != gx)
      throw LoomError("grid mismatch for " + name + ": " + std::to_string(gx) + " vs compiled " +
                      std::to_string(it->second));
  }
  // x (the f32 input of weight w's GEMV, "<pre><w>") into the weights' basis with w's awq: in place, or
  // for ssm_out from GGUF into HF value-head order into rotx_.
  void Rotate(const LoomBuffer& x, const std::string& w) { Rotate(Ref(x), w, Ref(*rotx_)); }
  // (out: ssm_out's gathered input; the other sites rotate x in place)
  void Rotate(const hrx_buffer_ref_t& x, const std::string& w, const hrx_buffer_ref_t& out) {
    const auto* t = Find(w + ".weight");
    const std::uint32_t k = static_cast<std::uint32_t>(t->dims[0]);
    const bool heads = w.find("ssm_out") != std::string::npos;
    std::vector<hrx_buffer_ref_t> b = {{x.buffer, x.offset, std::size_t{k} * 4}, TRef(w + ".awq"), Ref(*rot_signs_)};
    if (heads) b.push_back(out);
    Dispatch(Load(heads ? std::string("drot_heads") : "drot_" + std::to_string(k)), k / 1024, 1, 64, b);
  }
  // rmsnorm(x) * weight nw into normed_, rotated for weight w's GEMV (Rotate)
  void NormRotate(const LoomBuffer& x, const std::string& nw, const std::string& w) {
    NormRotate(Ref(x), nw, w, Ref(*normed_));
  }
  void NormRotate(const hrx_buffer_ref_t& x, const std::string& nw, const std::string& w, const hrx_buffer_ref_t& out) {
    if (!normrot_) {
      Dispatch(Load("rmsnorm"), 1, 1, 512, {x, TRef(nw), out});
      Rotate(out, w, {});
      return;
    }
    Dispatch(Load("drot_norm_5120"), kHidden / 1024, 1, 512, {x, TRef(nw), TRef(w + ".awq"), Ref(*rot_signs_), out});
  }
  void Rmsnorm(const LoomBuffer& x, const std::string& w, const LoomBuffer& out) {
    Dispatch(Load("rmsnorm"), 1, 1, 512, {Ref(x), TRef(w), Ref(out)});
  }
  // A T-token GEMV (gv_<kind>_..._t<T>, gen_gemv T): x [T][K], y [T][M]
  void GemvT(const char* kind, const std::vector<std::string>& ws, const LoomBuffer& x, const LoomBuffer& y) {
    std::string name = std::string("gv_") + kind;
    std::uint32_t M = 0, K = 0;
    std::vector<hrx_buffer_ref_t> b;
    for (const auto& w : ws) {
      const auto* t = Mq4(w);
      K = static_cast<std::uint32_t>(t->dims[0]);
      M = static_cast<std::uint32_t>(t->dims[1]);
      name += "_mq4";
      b.push_back(TRef(w));
    }
    name += "_" + std::to_string(M) + "_" + std::to_string(K) + "_t" + std::to_string(vt_);
    if (x.size < std::size_t{K} * vt_ * 4 || y.size < std::size_t{M} * vt_ * 4)
      throw LoomError("GEMV operand too small: " + name);
    b.push_back({x.handle, 0, std::size_t{K} * vt_ * 4});
    b.push_back({y.handle, 0, std::size_t{M} * vt_ * 4});
    auto [R, W] = Rw(kind);
    const std::uint32_t gx = M / (R * W);
    CheckGrid(name, gx);
    Dispatch(Load(name), gx, 1, 32 * W, b);
  }
  static hrx_buffer_ref_t Slice(const LoomBuffer& b, std::size_t i, std::size_t bytes) {
    return {b.handle, i * bytes, bytes};
  }
  // Verify's conv state ring: slot 0 the current state, slot i + 1 after verify token i.
  hrx_buffer_t ConvRing(std::uint32_t i) const { return i ? cring_[i - 1]->handle : cs_[cs_cur_]; }
  void AllocMtp() {
    const std::size_t poolb = std::size_t{T_} * kKv * 2;
    mtp_k_ = &Alloc(poolb), mtp_v_ = &Alloc(poolb), mtp_pt_ = &Alloc(std::size_t{T_ / 256} * 4);
    std::vector<std::int32_t> pages(T_ / 256);
    for (std::uint32_t i = 0; i < T_ / 256; ++i) pages[i] = static_cast<std::int32_t>(i);
    gpu_.H2D(*mtp_pt_, pages.data(), pages.size() * 4);
    cat_ = &Alloc(2 * kHidden * 4), mx_ = &Alloc(kHidden * 4), mh_ = &Alloc(kHidden * 4);
  }
  void AllocVerify() {
    const std::size_t t = vt_;
    hid_n_ = &Alloc(t * kHidden * 4), nrm_n_ = &Alloc(t * kHidden * 4), qg_n_ = &Alloc(t * kQProj * 4);
    k_n_ = &Alloc(t * kKv * 4), ao_n_ = &Alloc(t * kAttn * 4), qkv_n_ = &Alloc(t * kQkv * 4);
    gate_n_ = &Alloc(t * kAttn * 4), alpha_n_ = &Alloc(t * kTs * 4), beta_n_ = &Alloc(t * kTs * 4);
    so_n_ = &Alloc(t * kInner * 4), rx_n_ = &Alloc(t * kInner * 4), act_n_ = &Alloc(t * kFfn * 4);
    lg_n_ = &Alloc(t * kVocab * 4), vtok_ = &Alloc(t * 4);
    for (std::uint32_t i = 0; i < full_layers(); ++i) vsave_.push_back(&Alloc(t * kKv * 4));
    for (std::uint32_t i = 0; i < vt_; ++i) cring_.push_back(&Alloc(std::size_t{recurrent_layers()} * kConvState * 4));
    for (std::uint32_t i = 0; i + 1 < vt_; ++i)
      dring_.push_back(&Alloc(std::size_t{recurrent_layers()} * kStateElems * 4));
    if (quant()) vopen_save_ = &Alloc(std::size_t{full_layers()} * vopen_[0].length);
  }
  // YAH_DEC_TRACE=1: at step 0, sync after every stage and print max |x| and the NaN count.
  void Tr(const char* what, std::uint32_t l, const LoomBuffer& b, std::size_t n) {
    if (!trace_ || cur_pos_ != 0) return;
    gpu_.Synchronize();
    std::vector<float> h(n);
    gpu_.D2H(b, h.data(), n * 4);
    double mx = 0;
    std::size_t nan = 0;
    for (float v : h) {
      if (v != v)
        ++nan;
      else
        mx = std::max(mx, static_cast<double>(std::fabs(v)));
    }
    std::fprintf(stderr, "trace l%-2u %-10s max %.4g nan %zu\n", l, what, mx, nan);
  }

  LoomDevice& gpu_;
  const core::Gguf& gguf_;
  const core::Qwen35Config& cfg_;
  std::string dir_;
  const LoomWeights& weights_;
  static constexpr std::uint32_t kAnyTile = 0xffffffffu;
  std::uint32_t T_ = 0, cur_pos_ = 0, kbits_ = 16, vbits_ = 16, next_pos_ = kAnyTile;
  std::vector<hrx_buffer_ref_t> vopen_;  // quantized V: each layer's open tile, [1024][16] f16
  bool trace_ = false;
  bool rot_ = false;       // decode.txt "rot"
  bool chain_ = false;     // decode.txt "chain"
  bool normrot_ = false;   // decode.txt "normrot"
  bool swrot_ = false;     // decode.txt "swrot"
  bool amax_ = false;      // decode.txt "amax"
  bool late_ = false;      // decode.txt "latestate"
  bool mtp_ = false;       // decode.txt "mtp"
  std::uint32_t vt_ = 0;   // decode.txt "verify": tokens per T-token GEMV (0: no verify)
  // Verify: its tokens, the state rings, and per-token buffers ([vt_][...])
  std::uint32_t vpos_ = 0, vn_ = 0;
  bool vcross_ = false;
  LoomBuffer *hid_n_ = nullptr, *nrm_n_ = nullptr, *qg_n_ = nullptr, *k_n_ = nullptr, *ao_n_ = nullptr, *qkv_n_ = nullptr,
             *gate_n_ = nullptr, *alpha_n_ = nullptr, *beta_n_ = nullptr, *so_n_ = nullptr, *rx_n_ = nullptr,
             *act_n_ = nullptr, *lg_n_ = nullptr, *vtok_ = nullptr, *vopen_save_ = nullptr;
  std::vector<LoomBuffer*> vsave_, cring_, dring_;
  // MTP: its fp16 K / V^T pools and page table, the concatenated input, its residual stream, the next hidden input
  LoomBuffer *mtp_k_ = nullptr, *mtp_v_ = nullptr, *mtp_pt_ = nullptr, *cat_ = nullptr, *mx_ = nullptr, *mh_ = nullptr;
  bool sampling_ = false;  // SetSampling
  bool penalized_ = false;
  hrx_buffer_t cs_[2] = {nullptr, nullptr};  // conv state ping-pong
  int cs_cur_ = 0;
  std::deque<LoomBuffer> keep_;  // deque: Alloc hands out references that must stay valid
  std::map<std::string, LoomExecutable> exes_;
  std::map<std::string, std::pair<std::uint32_t, std::uint32_t>> rw_;
  std::map<std::string, std::uint32_t> grid_;
  LoomDecoderState st_;
  LoomBuffer *hidden_, *normed_, *qg_, *q_, *gate_, *kb_, *vb_, *aout_, *qkv_, *alpha_, *beta_, *ssmout_, *ffnact_,
      *logits_, *amaxk_ = nullptr, *sink_, *sparams_, *pparams_, *toks_, *posarr_, *eps_, *c32a_, *c32b_, *c16a_, *c16b_, *acc_, *ml_;
  LoomBuffer *rot_signs_ = nullptr, *rotx_ = nullptr, *swcnt_ = nullptr;
};

}  // namespace yah::model

#endif  // YAH_MODEL_LOOM_DECODER_HPP_
