// Engine: the GPU text generator. Loads the model, a chunked prefill HAL set and a decode HAL set once, then serves
// Generate() calls one at a time: prefill the prompt in chunks, then decode.
// A prompt that ends inside a chunk finishes either with a padded (masked) chunk or with decode steps, whichever the
// measured costs say is faster for its tail.
// Multi-turn reuse: a prompt that starts with the previous request's tokens (prompt and output) resumes at the last
// chunk boundary of that shared prefix whose recurrent state a snapshot slot holds; the KV rows below it are still in
// the pools. One slot is taken in the prefill just short of the prompt's end (a thinking turn's history drops its
// reasoning, so the next prompt leaves this one's tokens at its assistant header), the other at the last boundary the
// decode crossed (a no-think turn's history keeps its answer). YAH_NO_REUSE=1 turns it off.
#ifndef YAH_MODEL_ENGINE_HPP_
#define YAH_MODEL_ENGINE_HPP_

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "core/tokenizer.hpp"
#include "model/generator.hpp"
#include "model/loom_decoder.hpp"
#include "model/loom_prefill.hpp"
#include "model/loom_runtime.hpp"
#ifdef YAH_WITH_NPU
#include <sys/resource.h>
#include <unistd.h>

#include <fstream>

#include "model/loom_npu.hpp"
#endif

namespace yah::model {

class Engine : public TextGenerator {
 public:
  struct Options {
    std::string model;        // GGUF path
    std::string prefill_hal;  // emit_prefill_pp.py set (chunked, paged)
    std::string decode_hal;   // emit_decode.py set with the prefill's context and YAH_KV
    // The NPU runs the FFN block of a prefill set that has one (tools/emit_npu_ffn.py), its down weights from
    // npu_down (tools/npu_down_pack.py; default <prefill_hal>/npu_down.bin). Off, or not ready (NpuProblem), the
    // GPU runs everything.
    bool npu = false;
    std::string npu_down;
  };
  // How the tokens after the last whole chunk are run.
  enum class Tail { kAuto, kChunk, kDecode };
  void set_tail(Tail t) { tail_ = t; }

  explicit Engine(const Options& o)
      : gguf_(core::Gguf::OpenResident(o.model)),
        cfg_(core::Qwen35Config::FromGguf(gguf_)),
        tokenizer_(core::Tokenizer::FromGguf(gguf_, core::TokenizerConfig::FromGguf(gguf_))),
        weights_(gpu_, gguf_),
        prefill_(gpu_, gguf_, cfg_, o.prefill_hal, weights_),
        decoder_(gpu_, gguf_, cfg_, o.decode_hal, weights_) {
    if (decoder_.context() != prefill_.pool_rows())
      throw LoomError("engine: decode set context " + std::to_string(decoder_.context()) + " != prefill pool rows " +
                      std::to_string(prefill_.pool_rows()));
    if (decoder_.kv_bits() != prefill_.kv_bits())
      throw LoomError("engine: the decode set's KV format differs from the prefill's (emit with the same YAH_KV)");
    gpu_.SetSleepSync(200);
    logits_ = gpu_.Allocate(std::size_t{LoomPrefill::kVocab} * 4);
    seed_ = gpu_.Allocate(8);
    reuse_ = !prefill_.fpsparse() && !std::getenv("YAH_NO_REUSE");
    if (reuse_)
      for (Snapshot& s : snaps_) {
        s.conv = gpu_.Allocate(prefill_.conv_state().length);
        s.delta = gpu_.Allocate(prefill_.delta_state().length);
      }
    const void* th = nullptr;
    tokens_mirror_ = gpu_.AllocateHost((std::size_t{decoder_.context()} + 1) * 4, &th);
    tokens_host_ = static_cast<const std::uint32_t*>(th);
    const core::MetadataValue* name = gguf_.Meta("general.name");
    name_ = name && !name->s.empty() ? name->s : "qwen3.8-27b";
    if (o.npu) StartNpu(o);
  }
  ~Engine() override { gpu_.Synchronize(); }   // (the graphs wait on NPU words: none may run past npu_)

  // Why the NPU block is off ("" when it runs).
  [[nodiscard]] const std::string& npu_status() const { return npu_status_; }

  [[nodiscard]] const core::Tokenizer& tokenizer() const override { return tokenizer_; }
  [[nodiscard]] std::uint32_t context() const override { return prefill_.context(); }
  [[nodiscard]] std::string model_name() const override { return name_; }

  GenerateResult Generate(const std::vector<core::TokenId>& prompt, const GenerateParams& params,
                          const std::function<bool(core::TokenId)>& on_token) override {
    const std::size_t n = prompt.size();
    if (n == 0) throw LoomError("engine: empty prompt");
    for (core::TokenId id : prompt)
      if (id >= LoomPrefill::kVocab) throw LoomError("engine: token id outside the vocabulary");
    if (std::uint64_t{n} + params.max_tokens > context()) throw LoomError("engine: prompt + max_tokens exceeds context");
    try {
      return Run(prompt, params, on_token);
    } catch (...) {
      Forget();   // the pools and slots may hold a half-run sequence
      throw;
    }
  }

 private:
  GenerateResult Run(const std::vector<core::TokenId>& prompt, const GenerateParams& params,
                     const std::function<bool(core::TokenId)>& on_token) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint32_t n = static_cast<std::uint32_t>(prompt.size());
    GenerateResult r;
    r.prompt_tokens = n;
    const std::uint64_t seed =
        params.sampling.seed ? params.sampling.seed : static_cast<std::uint64_t>(t0.time_since_epoch().count());
    const bool greedy = params.sampling.temperature <= 0.0f;

    if (params.sampling.penalized()) {
      // The penalty window reads the prompt from the decoder's token stream. A host write: the steps the last request
      // queued past its stop must not land after it.
      gpu_.Synchronize();
      decoder_.SetTokens(prompt.data(), n, 0);
    }
    // Reuse: the slot with the deepest boundary inside the prefix shared with the cached sequence (below n: the last
    // prompt token must run for its logits). Slots past the shared prefix hold another sequence's state.
    const std::uint32_t B = prefill_.chunk(), full = n / B * B, tail = n - full;
    std::uint32_t shared = 0;
    while (shared < seq_.size() && shared < n && seq_[shared] == prompt[shared]) ++shared;
    int src = -1;
    for (int i = 0; i < 2; ++i) {
      if (snaps_[i].at > shared) snaps_[i].at = 0;
      if (snaps_[i].at && snaps_[i].at < n && (src < 0 || snaps_[i].at > snaps_[src].at)) src = i;
    }
    const std::uint32_t from = src >= 0 ? snaps_[src].at : 0;
    r.cached_tokens = from;
    seq_.assign(prompt.begin(), prompt.end());
    // The prefill slot: the last boundary kMargin tokens short of the prompt's end. The decode slot is the other one.
    const std::uint32_t at_prefill = reuse_ && n > kMargin ? (n - kMargin) / B * B : 0;
    const int xs = src < 0 ? 0 : at_prefill <= from ? src : 1 - src, ys = 1 - xs;
    if (src >= 0) {
      gpu_.Copy(Ref(snaps_[src].conv), prefill_.conv_state());
      gpu_.Copy(Ref(snaps_[src].delta), prefill_.delta_state());
    } else {
      prefill_.Reset();
    }
    decoder_.Bind(prefill_.DecoderState());
    // Prompt: whole chunks through the prefill; the tail as one more partial chunk or through decode steps, whichever
    // the measured costs say is cheaper (a partial chunk runs only the GEMM token tiles that hold its tokens).
    const bool tail_chunk =
        tail && (tail_ == Tail::kChunk || (tail_ == Tail::kAuto && tail * step_ms_ > chunk_ms_ * ChunkShare(tail)));
    const std::uint32_t chunks = full / B + (tail_chunk ? 1 : 0);
    // The host queues every chunk without waiting: one chunk costs the whole prefill's time over its chunks' shares.
    const auto tp = std::chrono::steady_clock::now();
    double shares = 0.0;
    for (std::uint32_t c = from / B; c < chunks; ++c) {
      const std::uint32_t valid = std::min(B, n - c * B);
      shares += ChunkShare(valid);
      // Two full chunks with no snapshot between them run as a pair (the NPU block of one beside the other's layer)
      if (prefill_.pairs() && c + 1 < chunks && n >= (c + 2) * B && !(at_prefill > from && (c + 1) * B == at_prefill)) {
        const bool head1 = c + 2 == chunks && (tail_chunk || tail == 0);
        shares += 2 * ChunkShare(B);
        for (std::uint32_t p = 0; p < 2; ++p)
          prefill_.SelectChunk(p), prefill_.Embed(prompt.data() + std::size_t{c + p} * B, B);
        prefill_.RunPair(c, LoomPrefill::kNoRows, head1 ? std::int64_t{B} - 1 : LoomPrefill::kNoRows, head1);
        if (head1) prefill_.SelectChunk(1), prefill_.Head(B - 1, Ref(logits_));
        prefill_.SelectChunk(0);
        if (at_prefill > from && (c + 2) * B == at_prefill) Save(xs, at_prefill, prefill_.conv_state());
        if (c + 2 == chunks) {
          gpu_.Synchronize();
          Track(chunk_ms_, (std::chrono::steady_clock::now() - tp) / shares, true);
        }
        ++c;
        continue;
      }
      prefill_.Embed(prompt.data() + std::size_t{c} * B, valid);
      LoomPrefill::KvHook hook;
      if (valid % 16 && decoder_.kv_bits().second != 16) {
        // Quantized V and the prompt ends mid-tile: seed the decoder's open tile with that tile's real V rows.
        const std::int32_t rc[2] = {static_cast<std::int32_t>(valid - valid % 16), static_cast<std::int32_t>(valid % 16)};
        gpu_.Update(seed_, rc, 8);
        hook = [&](std::uint32_t ai, std::uint32_t, std::size_t, std::size_t voff) {
          prefill_.SeedOpenTile(voff, Ref(seed_), decoder_.OpenTile(ai));
        };
      }
      // only the last chunk's last row is read (Head), and only when the prompt ends in it
      const bool head = c + 1 == chunks && (tail_chunk || tail == 0);
      prefill_.RunLayers(c, hook, head ? std::int64_t{valid} - 1 : LoomPrefill::kNoRows, head);
      if (c + 1 == chunks && (tail_chunk || tail == 0)) prefill_.Head(valid - 1, Ref(logits_));
      if (at_prefill > from && (c + 1) * B == at_prefill) Save(xs, at_prefill, prefill_.conv_state());
      if (c + 1 == chunks) {
        gpu_.Synchronize();
        Track(chunk_ms_, (std::chrono::steady_clock::now() - tp) / shares, true);
      }
    }
    // Every token, greedy or sampled (LoomDecoder::SetSampling), is picked on the GPU into the decoder's token stream,
    // which the next step reads, so the steps run kAhead ahead of the token the host reads; made[i] marks the copy of
    // the host's next token + i into tokens_host_. At most kAhead steps past a stop run for nothing.
    decoder_.SetSampling(params.sampling, seed);
    const bool tail_steps = tail && !tail_chunk;
    const auto ts = std::chrono::steady_clock::now();
    if (tail_steps) {
      decoder_.SetTokens(prompt.data() + full, tail, full);
      for (std::uint32_t pos = full; pos < n; ++pos) decoder_.Step(pos, n);   // the last one picks token n
    } else {
      decoder_.ResumeAt(n);
      if (decoder_.penalized()) decoder_.Penalize(Ref(logits_), n - 1);
      if (greedy)
        prefill_.Argmax(Ref(logits_), decoder_.TokenRef(n));
      else
        decoder_.Sample(Ref(logits_), n - 1, decoder_.TokenRef(n));
    }
    std::deque<LoomEvent> made;
    made.push_back(MarkToken(n));
    auto t1 = std::chrono::steady_clock::now();

    // Decode: report each token, then run the step that consumes it.
    r.finish_reason = "length";
    std::uint32_t queued = n;  // the next step to queue
    for (std::uint32_t pos = n;; ++pos) {
      for (; queued < pos + kAhead && queued + 1 < n + params.max_tokens && queued < decoder_.context(); ++queued) {
        decoder_.Step(queued, n);
        if (reuse_ && (queued + 1) % B == 0) Save(ys, queued + 1, decoder_.conv_state());
        made.push_back(MarkToken(queued + 1));
      }
      if (made.empty()) throw LoomError("decoder: position past the set's context");
      gpu_.Wait(made.front());
      made.pop_front();
      const core::TokenId tok = __atomic_load_n(&tokens_host_[pos], __ATOMIC_ACQUIRE);
      if (tok >= LoomDecoder::kVocab) throw LoomError("decoder: token id out of range: " + std::to_string(tok));
      seq_.push_back(tok);   // the step at pos + 1, if queued, consumed it
      if (pos == n) {   // the first token: the prompt's time ends here
        if (tail_steps) Track(step_ms_, (std::chrono::steady_clock::now() - ts) / tail, true);
        t1 = std::chrono::steady_clock::now();
        r.prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      }
      if (std::find(params.stop_ids.begin(), params.stop_ids.end(), tok) != params.stop_ids.end()) {
        r.finish_reason = "stop";
        break;
      }
      ++r.generated_tokens;
      if (!on_token(tok)) {
        r.finish_reason = "stop";
        break;
      }
      if (r.generated_tokens >= params.max_tokens) break;
    }
    r.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    // A slot the steps queued past the stop wrote covers tokens the host never read.
    for (Snapshot& s : snaps_)
      if (s.at > seq_.size()) s.at = 0;
    if (r.generated_tokens > 1) step_ms_ = 0.8 * step_ms_ + 0.2 * r.decode_ms / (r.generated_tokens - 1);
    return r;
  }

  static hrx_buffer_ref_t Ref(const LoomBuffer& b) { return {b.handle, 0, b.size}; }
  void StartNpu(const Options& o) {
#ifdef YAH_WITH_NPU
    npu_status_ = NpuProblem();
    if (npu_status_.empty()) {
      prefill_.EnablePairs();
      const NpuPlan plan = prefill_.npu_plan();
      if (!plan.a_bytes) {
        npu_status_ = "the prefill set has no NPU FFN block (tools/emit_npu_ffn.py)";
      } else {
        npu_ = std::make_unique<LoomNpuSplit>(gpu_, plan);
        prefill_.EnableNpu(npu_.get(), o.npu_down.empty() ? o.prefill_hal + "/npu_down.bin" : o.npu_down);
      }
    }
#else
    (void)o;
    npu_status_ = "this build has no NPU support (build against an HRX with the xdna targets)";
#endif
  }
#ifdef YAH_WITH_NPU
  // What keeps the NPU from running here, or "". Fabric DPM switches corrupt the NPU's outputs, so the GPU's DPM level
  // must be pinned; the NPU's buffers are locked pages.
  static std::string NpuProblem() {
    if (::access("/dev/accel/accel0", R_OK | W_OK) != 0) return "no NPU device (/dev/accel/accel0, amdxdna)";
    rlimit rl{};
    if (getrlimit(RLIMIT_MEMLOCK, &rl) != 0 || rl.rlim_cur != RLIM_INFINITY)
      return "the memlock limit is not unlimited (ulimit -l unlimited, or prlimit --memlock=unlimited:unlimited)";
    for (int card = 0; card < 8; ++card) {
      std::ifstream f("/sys/class/drm/card" + std::to_string(card) + "/device/power_dpm_force_performance_level");
      std::string level;
      if (f >> level)
        return level == "high" ? "" :
               "the GPU's DPM level is " + level + ", not high (echo high | sudo tee /sys/class/drm/card" +
                   std::to_string(card) + "/device/power_dpm_force_performance_level; auto restores it)";
    }
    return "no amdgpu DPM control found";
  }
#endif
  // The prefill slot sits this many tokens short of the prompt's end: the next turn's history replaces the assistant
  // header's tail (<think> and the like).
  static constexpr std::uint32_t kMargin = 32;
  // The recurrent state at sequence position at (0: empty); the KV rows below it stay in the pools.
  struct Snapshot {
    LoomBuffer conv, delta;
    std::uint32_t at = 0;
  };
  // Snapshot the recurrent state in stream order (after the work queued so far): conv from the given buffer, the
  // DeltaNet state from the prefill's in-place one.
  void Save(int slot, std::uint32_t at, const hrx_buffer_ref_t& conv) {
    gpu_.Copy(conv, Ref(snaps_[slot].conv));
    gpu_.Copy(prefill_.delta_state(), Ref(snaps_[slot].delta));
    snaps_[slot].at = at;
  }
  void Forget() {
    seq_.clear();
    for (Snapshot& s : snaps_) s.at = 0;
  }
  // Decode steps queued ahead of the token the host reads: the next step is queued while the current one runs.
  static constexpr std::uint32_t kAhead = 2;
  // Copies token pos of the decoder's stream to tokens_host_ and marks it.
  LoomEvent MarkToken(std::uint32_t pos) {
    gpu_.Copy(decoder_.TokenRef(pos), {tokens_mirror_.handle, std::size_t{pos} * 4, 4});
    return gpu_.Mark();
  }
  // A chunk of n real tokens costs about this share of a full one: the GEMMs (~92% of a full chunk) run whole 256-token
  // tiles, everything else runs the whole chunk.
  double ChunkShare(std::uint32_t n) const {
    const std::uint32_t B = prefill_.chunk(), tiles = (n + 255) / 256 * 256;
    return 0.08 + 0.92 * std::min(1.0, double(tiles) / B);
  }
  // Running estimate of a cost in ms; only timings that include the GPU work (synced) count.
  template <class D>
  static void Track(double& est, D elapsed, bool synced) {
    if (synced) est = 0.8 * est + 0.2 * std::chrono::duration<double, std::milli>(elapsed).count();
  }

  core::Gguf gguf_;
  core::Qwen35Config cfg_;
  core::Tokenizer tokenizer_;
  LoomDevice gpu_;
  LoomWeights weights_;
  LoomPrefill prefill_;
  LoomDecoder decoder_;
#ifdef YAH_WITH_NPU
  std::unique_ptr<LoomNpuSplit> npu_;   // (destroyed before the prefill that points at it)
#endif
  std::string npu_status_ = "off";
  LoomBuffer logits_, seed_;
  LoomBuffer tokens_mirror_;                    // host-local copy of the decoder's token stream (MarkToken)
  const std::uint32_t* tokens_host_ = nullptr;  // its mapping
  Tail tail_ = Tail::kAuto;
  bool reuse_ = false;
  Snapshot snaps_[2];
  std::vector<core::TokenId> seq_;   // the tokens the pools and slots were computed from (prompt, then the output read)
  double chunk_ms_ = 3100.0, step_ms_ = 62.0;  // prefill chunk and decode step costs (pp2048, short context)
  std::string name_;
};

}  // namespace yah::model

#endif  // YAH_MODEL_ENGINE_HPP_
