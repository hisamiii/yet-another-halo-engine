// The NPU FFN block's members of LoomPrefill (the prefill's last F FFN features on the XDNA2 NPU), out of line.
// Included by loom_prefill.hpp only.
#ifndef YAH_MODEL_LOOM_PREFILL_NPU_INL_HPP_
#define YAH_MODEL_LOOM_PREFILL_NPU_INL_HPP_

namespace yah::model {

inline void LoomPrefill::EnableNpu(NpuSplit* npu, std::uint32_t layers) {
  if (!npu || !FfnF() || !(a4_ || NpuNative()))
    throw LoomError("prefill: the set has no NPU FFN block (tools/emit_npu_ffn.py)");
  npu_ = npu;
  npu_layers_ = std::min(layers, cfg_.main_block_count());
  if (!NpuNative()) PackDown();
  // Bind every NPU call of a full chunk now (a pass of the layers that dispatches nothing): a first bind loads the
  // image's storage and patches it, milliseconds per call.
  LoomBuffer *h = hidden_, *h2 = hidden2_, *aq = aq_, *aq2 = aq2_, *ad = ad_, *ad2 = ad2_;
  n_ = B_, warming_ = true, npu_on_ = true, npu_planning_ = true;
  for (par_ = 0; par_ < (pairs_ ? 2u : 1u); ++par_) {  // (pairs: each chunk's half of A and C)
    RecordLayers(0, {});
    FlushKres();
  }
  par_ = 0;
  npu_->EndBinding();
  n_ = 0, warming_ = false, npu_on_ = false, npu_planning_ = false;
  hidden_ = h, hidden2_ = h2, aq_ = aq, aq2_ = aq2, ad_ = ad, ad2_ = ad2;
}

inline void LoomPrefill::PackDown() {
  down_layer_ = std::size_t{kHidden} * DownRowBytes();
  // (+ 4096: the last call's raw view runs a decoder record past its rows)
  npu_down_ = &Alloc(down_layer_ * npu_layers_ + 4096);
  // the down pack from the loaded weights (npu_down_pack.hpp), so the NPU's rows are this model's whatever its file
  std::vector<float> signs(512), awq(kFfn);
  ReadFile(dir_ + "/rot_signs.bin", signs.data(), signs.size() * 4);
  const std::size_t rb = std::size_t{kFfn} / 256 * NpuDownPack::kBlock, G = kFfn - FfnF();
  std::vector<std::uint8_t> rows(std::size_t{kHidden} * rb), pack(down_layer_);
  for (std::uint32_t l = 0; l < npu_layers_; ++l) {
    const std::string pre = "blk." + std::to_string(l) + ".";
    const core::TensorInfo &w = *Find(pre + "ffn_down.weight"), &a = *Find(pre + "ffn_down.awq");
    if (w.bytes != rows.size() || a.bytes != awq.size() * 4) throw LoomError("prefill: " + w.name + " is not MQ4G256");
    gpu_.D2H(TRef(w), rows.data());
    gpu_.D2H(TRef(a), awq.data());
    NpuDownPack::Rows(rows.data() + G / 256 * NpuDownPack::kBlock, rb, kHidden, FfnF(), awq.data() + G, signs.data(),
                      pack.data());
    gpu_.H2D(*npu_down_, pack.data(), pack.size(), std::size_t{l} * down_layer_);
  }
  npu_->ShareGpu(npu_down_->handle, npu_down_->size);
  npu_xo_ = &Alloc(std::size_t{B_} * kHidden * 2);
}

inline std::vector<std::pair<std::uint32_t, std::uint32_t>> LoomPrefill::FfnParts() const {
  const std::uint32_t P = FfnF() / 1024, n = (P + 4) / 5;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> parts;
  for (std::uint32_t i = 0, p0 = 0; i < n; ++i) {
    parts.push_back({p0, P / n + (i < P % n)});
    p0 += parts.back().second;
  }
  return parts;
}

inline bool LoomPrefill::RunFfnBlock(const std::string& pre) {
  if (!npu_on_ || trim_row_ != kAllRows || npu_layer_ >= npu_layers_) return false;
  if (NpuNative()) return RunFfnBlockNative(pre);
  const auto* tg = Find(pre + "ffn_gate.weight");
  const auto* tu = Find(pre + "ffn_up.weight");
  const auto* td = Find(pre + "ffn_down.weight");
  const std::uint32_t F = FfnF(), G = kFfn - F, layer = static_cast<std::uint32_t>(std::stoul(pre.substr(4)));
  const auto parts = FfnParts();
  const std::size_t rb = kHidden / 256 * 136, dgu = static_cast<std::size_t>(tu->offset - tg->offset);
  if (dgu != geom_.at("npudgu").tokens || &weights_.ChunkOf(*tg) != &weights_.ChunkOf(*tu))
    throw LoomError("prefill: " + pre + "ffn_up does not follow ffn_gate as the NPU block's image expects");
  const hrx_buffer_ref_t gate = TRef(*tg);
  const NpuBytes bs = NpuBytesOf("npubytes_ffnsw");
  const std::size_t ab = par_ * bs.a, cb = par_ * NpuCBytes();  // this chunk's half of A / C (pairs)
  if (npu_planning_) npu_->ShareGpu(gate.buffer, weights_.ChunkOf(*tg).size);
  // gate / up: F / 256 calls, call j's raw rows its 256 gate rows through its up rows (and a page the decoder may
  // prefetch); its H records at k-block 32 j (gen_npu_gemm swcol 32)
  std::vector<std::uint32_t> csw;
  for (std::uint32_t j = 0; j < F / 256; ++j)
    csw.push_back(npu_->BindRaw(dir_ + "/npu_ffnsw.xdna", {ab, bs.a},
                                {gate.buffer, gate.offset + (G + 256 * j) * rb, dgu + 256 * rb + 4096},
                                {cb + (j / 4) * kHRec + (j % 4) * 32 * 144, bs.c}, "ffnsw"));
  // down: per K part, kDownCalls calls of 512 rows of the down pack ([layer][row][F / 256 blocks]); a call's raw
  // view from its first row's part through a decoder record past its last row's
  std::vector<std::vector<std::uint32_t>> cdn(parts.size());
  const std::size_t doff = FfnDownOff(), cp = DownC(), rbd = DownRowBytes();
  for (std::size_t ci = 0; ci < parts.size(); ++ci) {
    const auto [p0, passes] = parts[ci];
    const NpuBytes bd = NpuBytesOf(DnImage(passes, "npubytes_"));
    if (bd.c != cp) throw LoomError("prefill: the FFN block's down parts differ in C bytes");
    for (std::uint32_t r = 0; r < kDownCalls; ++r)
      cdn[ci].push_back(npu_->BindRaw(
          dir_ + "/npu_" + DnImage(passes) + ".xdna", {cb + p0 * kHRec, bd.a, true},
          {npu_down_->handle,
           std::size_t{layer} * down_layer_ + std::size_t{r} * kDownRows * rbd + std::size_t{4} * p0 * 136,
           (kDownRows - 1) * rbd + std::size_t{4} * passes * 136 + kRecordRow},
          {cb + doff + (ci * kDownCalls + r) * cp, cp}, DnImage(passes)));
  }
  // the FFN norm's normq also writes the rotated rows in the decoder's positions, encoded into A for gate / up
  FlushKres();
  const std::string nw = pre + "post_attention_norm.weight";
  Dispatch(
      Exe("normqx_5120.hal"), "yah_normq", B_, 1, 1, geom_.at("normqx_5120.hal").rowgrp, 1, 1,
      {Ref(*hidden_), TRef(*Find(nw)), TRef(*Find(NormAwq(nw))), Ref(*rot_signs_), Ref(*aq_), Ref(*ad_), Ref(*npu_xo_)},
      0b1110000);
  const hrx_buffer_ref_t a = {npu_->A().handle, ab, bs.a};
  Dispatch(Exe("npu_enc_act.hal"), "yah_bfp16_encode_act", geom_.at("npu_enc_act.hal").tt, 1, 1, 256, 1, 1,
           {Ref(*npu_xo_), a}, 0b10);
  NpuEnqueue(std::move(csw), "ffnsw", a, {npu_->C().handle, cb, doff});
  for (std::size_t ci = 0; ci < parts.size(); ++ci)
    NpuEnqueue(std::move(cdn[ci]), DnImage(parts[ci].second), a,
               {npu_->C().handle, cb + doff + ci * kDownCalls * cp, kDownCalls * cp});
  // the GPU's features: gate / up over the first G, their rotq (down's awq), down over that K window into hidden2
  const std::string gu = "iu4_gu_" + std::to_string(G / 16) + "_20.hal";
  DispatchIu4(gu, "_gu", G / 16, {TRef(*tg), Ref(*aq_), Ref(*ad_), TRef(*tu), Ref(*ffnup_)}, 0b10000);
  const std::string rq = "rotq_" + std::to_string(G) + ".hal";
  Dispatch(Exe(rq), "yah_rotq256", geom_.at(rq).tt, B_, 1, geom_.at(rq).rowgrp, 1, 1,
           {Ref(*ffnup_), TRef(*Find(pre + "ffn_down.awq")), Ref(*rot_signs_), Ref(*aq_), Ref(*ad_)}, 0b11000);
  DispatchIu4("iu4_kresk_320_" + std::to_string(G / 256) + ".hal", "_kres", kHidden / 16,
              {TRef(*td), Ref(*aq_), Ref(*ad_), Ref(*hidden_), Ref(*hidden2_)}, 0b10000);
  // hidden = hidden2 + the NPU's partials, once its last job is done: now, or (pairs) before the chunk's next layer
  if (!npu_planning_) join_[par_] = flagged_.size() - 1;
  if (!pairs_) JoinFfn();
  return true;
}

// A native set's block (emit_npu_ffn.native): the layers of npu.txt "npulayer_<l>" run it, the NPU decoding the
// model's own rows from the weight chunks: gate / up (npu_ffnsw_<fmt>_<hord>.xdna; H in down's decoder order) and down
// (npu_ffndn<passes>_<fmt>.xdna) over its rows' last F features. The norm's row-major output, encoded, is gate / up's A;
// the GPU runs the fused ffn over the first G features (".npu.hal") and down's residual GEMM over that K window.
inline bool LoomPrefill::RunFfnBlockNative(const std::string& pre) {
  const std::uint32_t layer = static_cast<std::uint32_t>(std::stoul(pre.substr(4)));
  if (!geom_.count("npulayer_" + std::to_string(layer))) return false;
  const auto* tg = Find(pre + "ffn_gate.weight");
  const auto* tu = Find(pre + "ffn_up.weight");
  const auto* td = Find(pre + "ffn_down.weight");
  Fmt f{}, fd{};
  const std::string ffn = GemmHal("gemm_ffn", *tg, &f);
  if (!FmtOf(static_cast<std::uint32_t>(td->type), &fd)) return false;
  const std::uint32_t F = FfnF(), G = kFfn - F;
  const auto parts = FfnParts();
  const std::size_t rb = static_cast<std::size_t>(tg->bytes) / kFfn, dgu = static_cast<std::size_t>(tg->bytes);
  const std::size_t rbd = static_cast<std::size_t>(td->bytes) / kHidden, blkd = rbd / (kFfn / 256);
  if (tu->type != tg->type || tu->offset != tg->offset + tg->bytes || &weights_.ChunkOf(*tg) != &weights_.ChunkOf(*tu))
    throw LoomError("prefill: " + pre + "ffn_up does not follow ffn_gate as the NPU block's image expects");
  const std::string fam = "ffnsw_" + std::string(f.name) + (std::strcmp(fd.name, "q4k") == 0 ? "_q4k" : "_p4");
  const hrx_buffer_ref_t gate = TRef(*tg), down = TRef(*td);
  const NpuBytes bs = NpuBytesOf("npubytes_" + fam);
  const std::size_t ab = par_ * bs.a, cb = par_ * NpuCBytes();  // this chunk's half of A / C (pairs)
  if (npu_planning_) {
    npu_->ShareGpu(gate.buffer, weights_.ChunkOf(*tg).size);
    npu_->ShareGpu(down.buffer, weights_.ChunkOf(*td).size);
  }
  // gate / up: F / 256 calls, call j's raw rows its 256 gate rows through its up rows (and a page the decoder may
  // prefetch); its H records at k-block 32 j (gen_npu_gemm swcol 32)
  std::vector<std::uint32_t> csw;
  for (std::uint32_t j = 0; j < F / 256; ++j)
    csw.push_back(npu_->BindRaw(dir_ + "/npu_" + fam + ".xdna", {ab, bs.a},
                                {gate.buffer, gate.offset + (G + 256 * j) * rb, dgu + 256 * rb + 4096},
                                {cb + (j / 4) * kHRec + (j % 4) * 32 * 144, bs.c}, fam));
  // down: per K part, kDownCalls calls of 512 of its rows, each from its first row's part through a decoder record past
  // its last row's
  std::vector<std::vector<std::uint32_t>> cdn(parts.size());
  std::vector<std::string> dfam;
  const std::size_t doff = FfnDownOff(), cp = DownC();
  for (std::size_t ci = 0; ci < parts.size(); ++ci) {
    const auto [p0, passes] = parts[ci];
    dfam.push_back(DnImage(passes) + "_" + fd.name);
    const NpuBytes bd = NpuBytesOf("npubytes_" + dfam.back());
    if (bd.c != cp) throw LoomError("prefill: the FFN block's down parts differ in C bytes");
    for (std::uint32_t r = 0; r < kDownCalls; ++r)
      cdn[ci].push_back(npu_->BindRaw(
          dir_ + "/npu_" + dfam.back() + ".xdna", {cb + p0 * kHRec, bd.a, true},
          {down.buffer, down.offset + std::size_t{r} * kDownRows * rbd + (G / 256 + std::size_t{4} * p0) * blkd,
           (kDownRows - 1) * rbd + std::size_t{4} * passes * blkd + kRecordRow},
          {cb + doff + (ci * kDownCalls + r) * cp, cp}, dfam.back()));
  }
  FlushKres();
  RunNorm(pre + "post_attention_norm.weight", NormOut::kBoth);
  const hrx_buffer_ref_t a = {npu_->A().handle, ab, bs.a};
  Dispatch(Exe("npu_enc_act.hal"), "yah_bfp16_encode_act", geom_.at("npu_enc_act.hal").tt, 1, 1, 256, 1, 1,
           {Ref(*scratch_), a}, 0b10);
  NpuEnqueue(std::move(csw), fam, a, {npu_->C().handle, cb, doff});
  for (std::size_t ci = 0; ci < parts.size(); ++ci)
    NpuEnqueue(std::move(cdn[ci]), dfam[ci], a, {npu_->C().handle, cb + doff + ci * kDownCalls * cp, kDownCalls * cp});
  // the GPU's features: the fused ffn over the first G (f16(silu(gate) up), fragment-major at the full stride), then
  // down over that K window into hidden2
  const std::string sh = ffn.substr(0, ffn.size() - 4) + ".npu.hal";
  const Geom gs = GeomOf(sh);
  auto b = GemmWeights(*tg, f);
  b[0].length = static_cast<std::size_t>(tg->bytes + tu->bytes);
  for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{normt_, wstage_, ostage_, ffnup_}) b.push_back(Ref(*x));
  Dispatch(Exe(sh), ("yah_ffn_gemm_" + std::string(f.name) + "_ffn").c_str(), G / 16 / gs.rowgrp, gs.tt, 1, 32, 1, 1, b,
           GemmWrites(b, f, {ffnup_}));
  const std::string rw = "gemm_kresw_" + std::string(fd.name) + "_" + std::to_string(G / 256) + ".hal";
  const Geom gr = GeomOf(rw);
  auto r = GemmWeights(*td, fd);
  for (const LoomBuffer* x : std::initializer_list<const LoomBuffer*>{ffnup_, hidden_, wstage_, ostage_, hidden2_})
    r.push_back(Ref(*x));
  Dispatch(Exe(rw), ("yah_ffn_gemm_" + std::string(fd.name) + "_kres").c_str(), MTiles(*td) / gr.rowgrp, gr.tt, 1, 32, 1,
           1, r, GemmWrites(r, fd, {hidden2_}));
  // hidden = hidden2 + the NPU's partials, once its last job is done: now, or (pairs) before the chunk's next layer
  if (!npu_planning_) join_[par_] = flagged_.size() - 1;
  if (!pairs_) JoinFfn();
  return true;
}

inline void LoomPrefill::JoinFfn() {
  const std::size_t job = std::exchange(join_[par_], kNoJoin);
  if (job == kNoJoin) return;
  NpuJoin(job);
  const std::size_t cb = par_ * NpuCBytes(), doff = FfnDownOff();
  const Geom& gn = geom_.at("npu_unpack_ffn.hal");
  Dispatch(Exe("npu_unpack_ffn.hal"), "yah_npu_unpack", gn.tokens, 1, 1, gn.rowgrp, 1, 1,
           {{npu_->C().handle, cb + doff, NpuCBytes() - doff}, Ref(*hidden2_), Ref(*hidden_)}, 0b100);
}

inline void LoomPrefill::DispatchIu4(const std::string& hal, const char* sfx, std::uint32_t mtiles,
                                     const std::vector<hrx_buffer_ref_t>& b, std::uint64_t writes) {
  const Geom g = GeomOf(hal);
  std::uint32_t gx = mtiles / g.rowgrp, gz = 1;
  if (const auto it = geom_.find("iu4grp_" + hal); it != geom_.end()) {
    if (it->second.tokens == 0 || gx % it->second.tokens != 0)
      throw LoomError(hal + ": iu4grp does not divide the row tiles");
    gz = gx / it->second.tokens, gx = it->second.tokens;
  }
  Dispatch(Exe(hal), (std::string("yah_iu4_gemm") + sfx).c_str(), gx, g.tt, gz, 32, 1, 1, b, writes);
}

inline void LoomPrefill::NpuEnqueue(std::vector<std::uint32_t> calls, std::string tag, const hrx_buffer_ref_t& a,
                                    const hrx_buffer_ref_t& c) {
  if (npu_planning_) return;
  const NpuSplit::Job words = npu_->NewJob(calls);
  graph_->AtomicStore(FlagWord(words.ready), words.gate, HRX_ATOMIC_FLAG_RELEASE | HRX_ATOMIC_FLAG_SYSTEM_SCOPE, {a},
                      {c});
  flagged_.push_back({std::move(calls), std::move(tag), words, npu_layer_});
  job_c_.push_back(c);
}

inline void LoomPrefill::NpuCpuDriven(const std::vector<NpuSplit::Queued>& jobs) {
  std::thread enq([&] {  // it blocks while the NPU's queue is full; the NPU waits for the stores below
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
  std::map<std::string, std::pair<double, unsigned>> fam;  // per image family: ms, jobs
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    const NpuSplit::Job& w = jobs[i].words;
    std::vector<std::uint32_t> before;  // the job's gate slots, to report which words its calls changed
    for (std::uint32_t k = w.ready; k <= w.done + 16; ++k) before.push_back(npu_->HostLoad(k));
    npu_->HostStore(w.ready, w.gate);
    const auto t0 = std::chrono::steady_clock::now();
    std::uint32_t d = 0;
    while (((d = npu_->HostLoad(w.done)) & 0x7fffffffu) < w.gate)
      if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(1)) break;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    npu_->HostStore(w.ready, 0), npu_->HostStore(w.done, 0);  // as NpuJoin's stores do
    if ((d & 0x7fffffffu) < w.gate || (d & 0x80000000u)) {
      std::fprintf(stderr, "npu cpu-driven: job %zu (%s, layer %u) stalled or failed: done %08x gate %08x\n", i,
                   jobs[i].tag.c_str(), jobs[i].layer, d, w.gate);
      // YAH_NPU_CPU_STALL_CMD: run while the context lives (tools/npu_aie_status.py)
      if (const char* cmd = std::getenv("YAH_NPU_CPU_STALL_CMD")) (void)!std::system(cmd);
      for (std::uint32_t k = w.ready; k <= w.done + 16; ++k)
        if (const std::uint32_t v = npu_->HostLoad(k); v != before[k - w.ready])
          std::fprintf(stderr, "  word +%u: %08x -> %08x\n", k - w.ready, before[k - w.ready], v);
      return;
    }
    if (i < 6) std::fprintf(stderr, "npu cpu-driven: job %zu (%s) done in %.2f ms\n", i, jobs[i].tag.c_str(), ms);
    if (i >= 6) fam[jobs[i].tag].first += ms, ++fam[jobs[i].tag].second;
  }
  std::fprintf(stderr, "npu cpu-driven: %zu jobs done\n", jobs.size());
  for (const auto& [tag, t] : fam)  // (past the first 6: the image loads)
    std::fprintf(stderr, "npu cpu-driven: %s mean %.3f ms over %u jobs\n", tag.c_str(), t.first / t.second, t.second);
}

inline void LoomPrefill::NpuJoin(std::size_t job) {
  if (npu_planning_) return;
  const NpuSplit::Queued& j = flagged_.at(job);
  std::vector<hrx_buffer_ref_t> side = {Ref(npu_->A())};
  for (std::size_t i = npu_joined_; i <= job; ++i) side.push_back(job_c_[i]);
  if (!std::getenv("YAH_NPU_FAKE"))
    Dispatch(Exe("npu_flag_wait.hal"), "yah_npu_flag_wait", 1, 1, 1, 32, 1, 1,
             {FlagWord(j.words.done), FlagWord(j.words.ready), FlagWord(NpuSplit::kFlagStatus)}, 4, &side);
  npu_joined_ = job + 1;
  // the gate counts jobs per chunk: the joined jobs' words read zero again before the next chunk reuses them
  for (; npu_cleared_ <= job; ++npu_cleared_)
    for (const std::uint32_t w : {flagged_[npu_cleared_].words.ready, flagged_[npu_cleared_].words.done})
      graph_->AtomicStore(FlagWord(w), 0, HRX_ATOMIC_FLAG_RELEASE | HRX_ATOMIC_FLAG_SYSTEM_SCOPE, {}, {});
}

}  // namespace yah::model

#endif  // YAH_MODEL_LOOM_PREFILL_NPU_INL_HPP_
