// loom_forward_pp: the Loom prefill on HRX (model/loom_prefill.hpp) as a command-line tool.
//
// usage: loom_forward_pp <model> <haldir> <out-prefix> <tokens> <ids-file> [options]
// outputs: <out-prefix>.logits (vocab f32, last token) and <out-prefix>.hidden (B x 5120 f32, token major, last chunk;
//          rows past a partial chunk's tokens are padding)
// tokens: any count up to the set's context; the last chunk may be partial (masked prefill, as the Engine runs it).
// A short ids file is repeated to the token count.
//
// options:
//   --gen N --decode <emit_decode.py set>  N greedy tokens after the prompt (the prefill's argmax first)
//   --logits-from P     f32 logits of positions P..tokens-1 into <out-prefix>.all_logits (engine/run/gate)
//   --rowstats <file>   compact per-position logit statistics (engine/run/kvq/gate2.py) at --rowstats-from (1024)
//                       every --rowstats-stride (8), or at the positions listed in --rowstats-pos <file>
//   --slot-save <file> [--slot-at C]  save the sequence state before chunk C (default: after the last)
//   --slot-load <file>  restore a slot and run from the chunk it covers (marginal prefill at depth)
//   --chunk-ms          print each chunk's wall time (synchronizes after every chunk)
//   --pairs             full chunks two at a time, layer by layer (LoomPrefill::RunPair: the NPU block of one chunk
//                       runs while the GPU runs the other's layer)
//   --npu               the NPU runs the set's FFN block (tools/emit_npu_ffn.py)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "model/loom_decoder.hpp"
#ifdef YAH_WITH_NPU
#include "model/loom_npu.hpp"
#endif
#include "model/loom_prefill.hpp"
#include "model/loom_runtime.hpp"

using namespace yah::model;  // NOLINT(google-build-using-namespace)

namespace {
constexpr std::uint32_t kVocab = LoomPrefill::kVocab;

FILE* OpenOut(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw LoomError("cannot write " + path);
  return f;
}

std::vector<std::uint32_t> ParseIds(const char* path) {
  std::ifstream file(path);
  if (!file) throw LoomError(std::string("cannot read ids file ") + path);
  std::vector<std::uint32_t> ids;
  std::uint64_t v;
  while (file >> v) {
    if (v >= kVocab) throw LoomError("token id " + std::to_string(v) + " is outside the vocabulary");
    ids.push_back(static_cast<std::uint32_t>(v));
  }
  if (ids.empty()) throw LoomError("ids file is empty");
  return ids;
}

// One YAH_ROWSTATS record per position (little endian): int32 pos, int32 next-token id (-1 at the end), f32 logsumexp,
// f32 logit of the next token, int32 argmax, f32 top1, f32 top2, then 64 x (int32 id, f32 logit), highest first.
void WriteRowStats(FILE* f, const float* lg, std::uint32_t pos, std::int32_t nxt, std::vector<std::uint32_t>& idx) {
  double mx = lg[0];
  for (std::uint32_t v = 1; v < kVocab; ++v) mx = std::max<double>(mx, lg[v]);
  double se = 0.0;
  for (std::uint32_t v = 0; v < kVocab; ++v) se += std::exp(static_cast<double>(lg[v]) - mx);
  const float lse = static_cast<float>(mx + std::log(se));
  for (std::uint32_t v = 0; v < kVocab; ++v) idx[v] = v;
  std::partial_sort(idx.begin(), idx.begin() + 64, idx.end(),
                    [&](std::uint32_t a, std::uint32_t b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
  const std::int32_t ipos = static_cast<std::int32_t>(pos), am = static_cast<std::int32_t>(idx[0]);
  const float lt = nxt >= 0 ? lg[nxt] : 0.0f, t1 = lg[idx[0]], t2 = lg[idx[1]];
  std::fwrite(&ipos, 4, 1, f);
  std::fwrite(&nxt, 4, 1, f);
  std::fwrite(&lse, 4, 1, f);
  std::fwrite(&lt, 4, 1, f);
  std::fwrite(&am, 4, 1, f);
  std::fwrite(&t1, 4, 1, f);
  std::fwrite(&t2, 4, 1, f);
  for (int k = 0; k < 64; ++k) {
    const std::int32_t id = static_cast<std::int32_t>(idx[k]);
    std::fwrite(&id, 4, 1, f);
    std::fwrite(&lg[idx[k]], 4, 1, f);
  }
}
struct Options {
  std::uint32_t gen = 0, slot_at = ~0u, rowstats_from = 1024, rowstats_stride = 8;
  std::int64_t logits_from = -1;
  std::string decode, rowstats, rowstats_pos, slot_save, slot_load;
  bool chunk_ms = false, pairs = false, npu = false;
};

bool ParseOptions(int argc, char** argv, Options* o) {
  for (int i = 6; i < argc; ++i) {
    const std::string a = argv[i];
    const bool v = i + 1 < argc;
    const auto num = [&] { return static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10)); };
    if (a == "--gen" && v)
      o->gen = num();
    else if (a == "--decode" && v)
      o->decode = argv[++i];
    else if (a == "--logits-from" && v)
      o->logits_from = num();
    else if (a == "--rowstats" && v)
      o->rowstats = argv[++i];
    else if (a == "--rowstats-from" && v)
      o->rowstats_from = num();
    else if (a == "--rowstats-stride" && v)
      o->rowstats_stride = std::max(1u, num());
    else if (a == "--rowstats-pos" && v)
      o->rowstats_pos = argv[++i];
    else if (a == "--slot-save" && v)
      o->slot_save = argv[++i];
    else if (a == "--slot-at" && v)
      o->slot_at = num();
    else if (a == "--slot-load" && v)
      o->slot_load = argv[++i];
    else if (a == "--chunk-ms")
      o->chunk_ms = true;
    else if (a == "--pairs")
      o->pairs = true;
    else if (a == "--npu")
      o->npu = true;
    else
      return std::fprintf(stderr, "loom_forward_pp: unknown option %s\n", a.c_str()), false;
  }
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (argc < 6 || !ParseOptions(argc, argv, &opt)) {
    std::fprintf(stderr,
                 "usage: loom_forward_pp <model> <haldir> <out-prefix> <tokens> <ids-file> [options]"
                 " (see the source's header)\n");
    return 2;
  }
  const std::string prefix = argv[3];
  const std::uint32_t want = std::strtoul(argv[4], nullptr, 10);
  const char* ids_path = argv[5];
  try {
    auto gguf = yah::core::Gguf::OpenResident(argv[1]);
    const auto cfg = yah::core::Qwen35Config::FromGguf(gguf);
    std::vector<std::uint32_t> ids = ParseIds(ids_path);
    LoomDevice gpu;
    // The layers queue ~950 dispatches and wait once: the runtime wait would busy-poll a host core the whole time.
    gpu.SetSleepSync(200);
    LoomWeights weights(gpu, gguf);
    LoomPrefill pf(gpu, gguf, cfg, argv[2], weights, want);
    if (opt.pairs) pf.EnablePairs();
#ifdef YAH_WITH_NPU
    std::unique_ptr<yah::model::LoomNpuSplit> npu;
    if (opt.npu) {
      const auto plan = pf.npu_plan();
      if (plan.a_bytes == 0) {
        std::fprintf(stderr, "--npu: needs a set with the NPU FFN block (emit_npu_ffn.py)\n");
        return 2;
      }
      npu = std::make_unique<yah::model::LoomNpuSplit>(gpu, plan);
      pf.EnableNpu(npu.get());
    }
#else
    if (opt.npu) throw LoomError("--npu: this build has no NPU support");
#endif
    const std::uint32_t B = pf.chunk(), T_run = want ? want : B;
    if (T_run == 0 || T_run > pf.context()) {
      std::fprintf(stderr, "tokens=%u must be at most the context %u\n", T_run, pf.context());
      return 2;
    }
    for (std::size_t i = ids.size(), n = ids.size(); i < T_run; ++i) ids.push_back(ids[i % n]);
    const std::uint32_t n_chunks = (T_run + B - 1) / B, last_n = T_run - (n_chunks - 1) * B;
    std::fprintf(stderr, "loom_forward_pp: tokens=%u chunk=%u context=%u\n", T_run, B, pf.context());

    std::uint32_t logits_from = T_run;
    if (opt.logits_from >= 0) {
      logits_from = static_cast<std::uint32_t>(opt.logits_from);
      if (logits_from >= T_run) throw LoomError("--logits-from must be below the token count");
    }
    LoomBuffer every = gpu.Allocate(std::size_t{T_run - logits_from + (logits_from == T_run)} * kVocab * 4);
    FILE* rowstats = nullptr;
    std::vector<char> rs_want(pf.context(), 0);
    if (!opt.rowstats.empty()) {
      rowstats = OpenOut(opt.rowstats);
      if (!opt.rowstats_pos.empty()) {
        std::ifstream pfs(opt.rowstats_pos);
        if (!pfs) throw LoomError("cannot read " + opt.rowstats_pos);
        std::uint32_t pp;
        while (pfs >> pp)
          if (pp < pf.context()) rs_want[pp] = 1;
      } else {
        for (std::uint32_t pp = opt.rowstats_from; pp < pf.context(); pp += opt.rowstats_stride) rs_want[pp] = 1;
      }
    }
    std::uint32_t rs_max = 1;  // rows per chunk
    for (std::uint32_t c = 0; c < n_chunks; ++c)
      rs_max = std::max<std::uint32_t>(
          rs_max, std::count(rs_want.begin() + std::size_t{c} * B, rs_want.begin() + std::size_t{c + 1} * B, 1));
    LoomBuffer rsbuf = gpu.Allocate(std::size_t{rs_max} * kVocab * 4);

    // Slots (LoomPrefill::SaveState): --slot-save writes the state before chunk --slot-at (default: after the last
    // chunk); --slot-load restores it and runs from the chunk it covers, so layers_ms is the marginal time.
    std::uint32_t ci0 = 0;
    if (!opt.slot_load.empty()) {
      ci0 = pf.LoadState(opt.slot_load);
      if (ci0 >= n_chunks) throw LoomError("the slot covers every chunk of this run");
      std::fprintf(stderr, "loom_forward_pp: slot %s covers %u chunks; running from token %u\n", opt.slot_load.c_str(),
                   ci0, ci0 * B);
    }
    const char* slot_save = opt.slot_save.empty() ? nullptr : opt.slot_save.c_str();
    const std::uint32_t slot_at = std::min(opt.slot_at, n_chunks);

    gpu.Synchronize();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> idx(kVocab);
    const bool chunk_ms = opt.chunk_ms;
    auto tc = t0;
    for (std::uint32_t ci = ci0; ci < n_chunks; ++ci) {
      // --pairs: two full chunks at a time (LoomPrefill::RunPair)
      if (pf.pairs() && ci + 1 < n_chunks && (ci + 2 < n_chunks || last_n == B)) {
        if (rowstats || slot_save) throw LoomError("--pairs runs without --rowstats / --slot-save");
        const auto keep = [&](std::uint32_t c) {
          return logits_from < std::min((c + 1) * B, T_run) ? LoomPrefill::kAllRows
                 : c + 1 == n_chunks                        ? std::int64_t{last_n} - 1
                                                            : LoomPrefill::kNoRows;
        };
        for (std::uint32_t p = 0; p < 2; ++p) pf.SelectChunk(p), pf.Embed(ids.data() + std::size_t{ci + p} * B, B);
        pf.RunPair(ci, keep(ci), keep(ci + 1), ci + 2 == n_chunks);
        for (std::uint32_t p = 0; p < 2; ++p) {
          pf.SelectChunk(p);
          const std::uint32_t c = ci + p;
          for (std::uint32_t ra = std::max(logits_from, c * B); ra < std::min((c + 1) * B, T_run); ++ra)
            pf.Head(ra - c * B, {every.handle, std::size_t{ra - logits_from} * kVocab * 4, std::size_t{kVocab} * 4});
        }
        ++ci;
        continue;
      }
      if (slot_save && ci == slot_at) pf.SaveState(slot_save, ci);
      pf.Embed(ids.data() + std::size_t{ci} * B, ci + 1 == n_chunks ? last_n : B);
      // the last layer's tail only for the rows read below: every row for logits_from / rowstats, else the last row
      const bool rows = rowstats || logits_from < std::min((ci + 1) * B, T_run);
      pf.RunLayers(ci, {}, rows ? LoomPrefill::kAllRows
                       : ci + 1 == n_chunks ? std::int64_t{last_n} - 1
                                            : LoomPrefill::kNoRows,
                   ci + 1 == n_chunks);
      for (std::uint32_t ra = std::max(logits_from, ci * B); ra < std::min((ci + 1) * B, T_run); ++ra)
        pf.Head(ra - ci * B, {every.handle, std::size_t{ra - logits_from} * kVocab * 4, std::size_t{kVocab} * 4});
      if (rowstats) {
        std::vector<std::uint32_t> rows;
        for (std::uint32_t r = 0; r < (ci + 1 == n_chunks ? last_n : B); ++r)
          if (rs_want[std::size_t{ci} * B + r]) rows.push_back(r);
        for (std::size_t i = 0; i < rows.size(); ++i)
          pf.Head(rows[i], {rsbuf.handle, i * kVocab * 4, std::size_t{kVocab} * 4});
        if (!rows.empty()) {
          gpu.Synchronize();
          std::vector<float> host(rows.size() * kVocab);
          gpu.D2H(rsbuf, host.data(), host.size() * 4, 0);
          for (std::size_t i = 0; i < rows.size(); ++i) {
            const std::uint32_t pos = ci * B + rows[i];
            const std::int32_t nxt = pos + 1 < T_run ? static_cast<std::int32_t>(ids[pos + 1]) : -1;
            WriteRowStats(rowstats, host.data() + i * kVocab, pos, nxt, idx);
          }
        }
      }
      if (chunk_ms) {   // per-chunk wall time (synchronizes: the next chunk's recording no longer overlaps this one)
        gpu.Synchronize();
        const auto tn = std::chrono::steady_clock::now();
        std::printf("chunk=%u start=%u ms=%.1f\n", ci, ci * B, std::chrono::duration<double, std::milli>(tn - tc).count());
        std::fflush(stdout);
        tc = tn;
      }
    }
    gpu.Synchronize();
    if (slot_save && slot_at >= n_chunks) pf.SaveState(slot_save, n_chunks);
    if (rowstats) std::fclose(rowstats);
    const double layer_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("layers_ms=%.1f\n", layer_ms);
#ifdef YAH_WITH_NPU
    if (npu) npu->Report(stdout);
#endif

    {
      std::vector<float> out(std::size_t{B} * LoomPrefill::kHidden);
      gpu.D2H(pf.hidden(), out.data(), out.size() * 4, 0);
      FILE* fo = OpenOut(prefix + ".hidden");
      std::fwrite(out.data(), 4, out.size(), fo);
      std::fclose(fo);
    }
    LoomBuffer logits = gpu.Allocate(std::size_t{kVocab} * 4);
    LoomBuffer token = gpu.Allocate(4);
    pf.Head(last_n - 1, {logits.handle, 0, logits.size});
    pf.Argmax({logits.handle, 0, logits.size}, {token.handle, 0, 4});
    gpu.Synchronize();
    std::uint32_t tok = 0;
    gpu.D2H(token, &tok, 4, 0);
    std::vector<float> logit_host(kVocab);
    gpu.D2H(logits, logit_host.data(), logit_host.size() * 4, 0);
    FILE* fl = OpenOut(prefix + ".logits");
    std::fwrite(logit_host.data(), 4, logit_host.size(), fl);
    std::fclose(fl);
    std::printf("argmax=%u\n", tok);

    if (opt.gen > 0) {
      const std::uint32_t ngen = opt.gen;
      if (opt.decode.empty()) throw LoomError("--gen needs --decode (a tools/emit_decode.py set)");
      if (T_run + ngen - 1 > pf.context()) throw LoomError("--gen: prompt + gen exceeds the emitted context");
      LoomDecoder dec(gpu, gguf, cfg, opt.decode, weights);
      if (dec.context() != pf.pool_rows())
        throw LoomError("--gen: decode set context " + std::to_string(dec.context()) + " != prefill pool rows " +
                        std::to_string(pf.pool_rows()));
      if (dec.kv_bits() != pf.kv_bits())
        throw LoomError("--gen: the decode set's KV format differs from the prefill's (emit with the same YAH_KV)");
      dec.Bind(pf.DecoderState());
      dec.SetTokens(&tok, 1, T_run);
      gpu.Synchronize();
      const auto tg = std::chrono::steady_clock::now();
      for (std::uint32_t pos = T_run; pos + 1 < T_run + ngen; ++pos) {
        dec.Step(pos, T_run);
        gpu.Flush();
      }
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

    if (logits_from < T_run) {
      const std::size_t rows = T_run - logits_from;
      std::vector<float> host(rows * kVocab);
      gpu.D2H(every, host.data(), host.size() * 4, 0);
      FILE* fa = OpenOut(prefix + ".all_logits");
      std::fwrite(host.data(), 4, host.size(), fa);
      std::fclose(fa);
      std::printf("all_logits rows=%zu from=%u\n", rows, logits_from);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "loom_forward_pp: %s\n", error.what());
    return 1;
  }
}
