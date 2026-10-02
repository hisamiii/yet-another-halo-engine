// loom_decode: single-token decode forward on Loom through HRX, GEMV-based.
//
// usage: loom_decode <model.gguf> <hal_dir> --ids "1 2 3" [--gen N] [--logits FILE]
//
// The prompt goes through the decode path one token at a time (positions 0..n-1),
// then N tokens are generated greedily. HAL set: tools/emit_decode.py.
// Per layer, as HIP's Forward::Decode:
//   rmsnorm -> full attention: attn_q / attn_k / attn_v GEMV, unpack q|gate, QK norm
//              + RoPE, append K / V to the paged pools, split-K attention + reduce
//              (gate), attn_output GEMV += hidden
//           -> recurrent:      attn_qkv / attn_gate / ssm_alpha / ssm_beta GEMV, conv,
//              DeltaNet (gated norm inside), ssm_out GEMV += hidden
//   rmsnorm -> ffn_gate|ffn_up SwiGLU GEMV -> ffn_down GEMV += hidden
// head: rmsnorm -> output GEMV -> argmax, written into a device token stream that the
// next step's embedding kernel (IQ4_XS token_embd) reads: steps are enqueued back to
// back with no host round trip (the host waits after the prompt and at the end).
// --logits FILE appends every step's 248320 logits (f32) for an external KL gate.
#include <algorithm>
#include <chrono>
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

#include "core/gguf.hpp"
#include "core/config.hpp"
#include "core/tokenizer.hpp"
#include "model/loom_runtime.hpp"

using namespace yah::model;  // NOLINT(google-build-using-namespace)

namespace {
constexpr std::uint32_t kHidden = 5120, kFfn = 17408, kAttn = 6144, kQProj = 12288, kKv = 1024;
constexpr std::uint32_t kInner = 6144, kQkv = 10240, kHeadsV = 48, kTs = 48, kState = 128;
constexpr std::uint32_t kHeads = 24, kKvHeads = 4, kVocab = 248320;
constexpr std::uint32_t kConvState = kQkv * 4, kStateElems = kHeadsV * kState * kState;
constexpr std::uint32_t kR = 2, kW = 4;   // gen_gemv defaults: rows per wave, waves per workgroup

struct Fmt { const char* name; std::uint32_t qk, bb; std::uint32_t tables; };
// table bits, in gen_gemv.TABLE_ORDER: grid_iq3s, grid_iq3xxs, grid_iq2xxs, grid_iq2xs, ksigns
bool FmtOf(std::uint32_t type, Fmt* f) {
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

std::vector<char> ReadAll(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw LoomError("cannot open " + path);
  return std::vector<char>(std::istreambuf_iterator<char>(in), {});
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: loom_decode <model.gguf> <hal_dir> --ids \"1 2\" [--gen N] [--logits FILE]\n");
    return 2;
  }
  const std::string model = argv[1], dir = argv[2];
  std::string ids_arg, logits_path;
  std::uint32_t gen_count = 16;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--ids" && i + 1 < argc) ids_arg = argv[++i];
    else if (a == "--gen" && i + 1 < argc) gen_count = static_cast<std::uint32_t>(std::atoi(argv[++i]));
    else if (a == "--logits" && i + 1 < argc) logits_path = argv[++i];
    else { std::fprintf(stderr, "loom_decode: unknown argument %s\n", a.c_str()); return 2; }
  }
  try {
    auto gguf = yah::core::Gguf::Open(model);
    const auto cfg = yah::core::Qwen35Config::FromGguf(gguf);
    const auto tconfig = yah::core::TokenizerConfig::FromGguf(gguf);
    const auto tokenizer = yah::core::Tokenizer::FromGguf(gguf, tconfig);
    std::vector<std::uint32_t> prompt;
    { std::istringstream in(ids_arg); std::uint32_t v; while (in >> v) prompt.push_back(v); }
    if (prompt.empty()) throw LoomError("--ids is required");
    std::uint32_t T = 0;
    {
      std::istringstream in(std::string(ReadAll(dir + "/decode.txt").data()));
      std::string k; in >> k >> T;
      if (k != "ctx" || T == 0 || T % 256) throw LoomError("bad decode.txt");
    }
    const std::uint32_t steps = static_cast<std::uint32_t>(prompt.size()) + gen_count - 1;
    if (steps > T) throw LoomError("prompt + gen exceeds the set's max context");
    const std::uint32_t npg = T / 256;

    LoomDevice gpu;
    LoomBuffer weights;
    std::size_t delta = 0;
    {
      const std::uint8_t* wbase = gguf.tensor_data_base();
      const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(wbase) & ~std::uintptr_t{4095};
      delta = reinterpret_cast<std::uintptr_t>(wbase) - start;
      weights = gpu.Import(reinterpret_cast<void*>(start), gguf.tensor_data_size() + delta);
    }
    auto find = [&](const std::string& n) {
      const auto* t = gguf.Find(n);
      if (!t) throw LoomError("tensor not found: " + n);
      return t;
    };
    auto tref = [&](const std::string& n) -> hrx_buffer_ref_t {
      const auto* t = find(n);
      return {weights.handle, delta + static_cast<std::size_t>(t->offset), static_cast<std::size_t>(t->bytes)};
    };
    std::map<std::string, LoomExecutable> exes;
    auto load = [&](const std::string& name) -> LoomExecutable& {
      auto it = exes.find(name);
      if (it == exes.end()) {
        it = exes.emplace(name, gpu.Load(dir + "/" + name + ".hal")).first;
        if (std::getenv("YAH_DEC_LIST")) std::fprintf(stderr, "loaded %zu %s\n", exes.size(), name.c_str());
      }
      return it->second;
    };
    std::deque<LoomBuffer> keep;   // stable references (a vector would move them)
    auto alloc = [&](std::size_t bytes) -> LoomBuffer& {
      keep.push_back(gpu.Allocate(bytes));
      std::vector<char> z(bytes, 0);
      gpu.H2D(keep.back(), z.data(), bytes);
      return keep.back();
    };
    const char* tnames[5] = {"grid_iq3s.bin", "grid_iq3xxs.bin", "grid_iq2xxs.bin", "grid_iq2xs.bin", "ksigns_iq2xs.bin"};
    std::vector<hrx_buffer_ref_t> tabs;
    for (const char* tn : tnames) {
      const auto data = ReadAll(dir + "/" + tn);
      LoomBuffer& b = alloc(data.size());
      gpu.H2D(b, data.data(), data.size());
      tabs.push_back({b.handle, 0, data.size()});
    }
    auto ref = [](const LoomBuffer& b) -> hrx_buffer_ref_t { return {b.handle, 0, b.size}; };

    LoomBuffer& hidden = alloc(kHidden * 4);
    LoomBuffer& normed = alloc(kHidden * 4);
    LoomBuffer& qg = alloc(kQProj * 4);
    LoomBuffer& q = alloc(kAttn * 4);
    LoomBuffer& gate = alloc(kAttn * 4);
    LoomBuffer& kb = alloc(kKv * 4);
    LoomBuffer& vb = alloc(kKv * 4);
    LoomBuffer& aout = alloc(kAttn * 4);
    LoomBuffer& qkv = alloc(kQkv * 4);
    LoomBuffer& alpha = alloc(kTs * 4);
    LoomBuffer& beta = alloc(kTs * 4);
    LoomBuffer& convout = alloc(kQkv * 4);
    LoomBuffer& ssmout = alloc(kInner * 4);
    LoomBuffer& ffnact = alloc(kFfn * 4);
    LoomBuffer& logits = alloc(std::size_t{kVocab} * 4);
    LoomBuffer& token = alloc(4);                       // argmax sink for prompt positions
    // device token stream (prompt preloaded; argmax writes position pos + 1) and
    // position array (step pos binds element pos): no host round trip per token
    LoomBuffer& toks = alloc((std::size_t{T} + 1) * 4);
    LoomBuffer& posarr = alloc(std::size_t{T} * 4);
    LoomBuffer& eps = alloc(4);
    LoomBuffer& c32a = alloc(kKv * 4);
    LoomBuffer& c32b = alloc(kKv * 4);
    LoomBuffer& c16a = alloc(kKv * 2);
    LoomBuffer& c16b = alloc(kKv * 2);
    LoomBuffer& acc = alloc(std::size_t{npg} * kHeads * 256 * 4);
    LoomBuffer& ml = alloc(std::size_t{npg} * kHeads * 2 * 4);
    LoomBuffer& ptab = alloc(std::size_t{npg} * 4);
    {
      std::vector<std::int32_t> pages(npg);
      for (std::uint32_t i = 0; i < npg; ++i) pages[i] = static_cast<std::int32_t>(i);
      gpu.H2D(ptab, pages.data(), pages.size() * 4);
      const float e = 1.0e-6f;
      gpu.H2D(eps, &e, 4);
      std::vector<std::int32_t> pv(T);
      for (std::uint32_t i = 0; i < T; ++i) pv[i] = static_cast<std::int32_t>(i);
      gpu.H2D(posarr, pv.data(), pv.size() * 4);
      gpu.H2D(toks, prompt.data(), prompt.size() * 4);
    }
    const std::uint32_t nfull = cfg.main_block_count() / cfg.full_attention_interval;
    const std::uint32_t nssm = cfg.main_block_count() - nfull;
    const std::size_t poolb = std::size_t{T} * kKv * 2;
    std::vector<LoomBuffer*> kpool, vtpool;
    for (std::uint32_t i = 0; i < nfull; ++i) { kpool.push_back(&alloc(poolb)); vtpool.push_back(&alloc(poolb)); }
    LoomBuffer& convstate = alloc(std::size_t{nssm} * kConvState * 4);
    LoomBuffer& dstate = alloc(std::size_t{nssm} * kStateElems * 4);

    // workgroup size and binding count come from the compiled export and must
    // agree with the call site (argmax is a 1024-lane kernel: launched with 32
    // lanes its per-wave LDS slots stay unwritten and it returns garbage)
    auto dispatch = [&](LoomExecutable& e, std::uint32_t gx, std::uint32_t gy, std::uint32_t wg,
                        const std::vector<hrx_buffer_ref_t>& b) {
      const std::uint32_t cw = e.WorkgroupSize(0), cb = e.BindingCount(0);
      if ((cw && cw != wg) || (cb && cb != b.size()))
        throw LoomError("dispatch mismatch: workgroup " + std::to_string(wg) + " vs compiled " + std::to_string(cw) +
                        ", bindings " + std::to_string(b.size()) + " vs " + std::to_string(cb));
      gpu.Dispatch(e, 0, LoomDevice::Config(gx, gy, 1, wg, 1, 1), nullptr, 0, b.data(), b.size());
    };
    // GEMV: names and footprints as tools/gen_gemv.py; checked once per tensor
    // overlap = true: no ordering barrier against the previous dispatch (independent
    // projections of one input); the next barriered dispatch still waits for all of them
    auto gemv = [&](const char* kind, const std::vector<std::string>& ws, const LoomBuffer& x, const LoomBuffer& y,
                    bool overlap = false) {
      std::string name = std::string("gv_") + kind;
      std::uint32_t tbits = 0, M = 0, K = 0;
      std::vector<hrx_buffer_ref_t> b;
      for (const auto& w : ws) {
        const auto* t = find(w);
        Fmt f{};
        if (!FmtOf(static_cast<std::uint32_t>(t->type), &f)) throw LoomError("no GEMV format for " + w);
        K = static_cast<std::uint32_t>(t->dims[0]);
        M = static_cast<std::uint32_t>(t->dims[1]);
        if (t->bytes != std::uint64_t{K} / f.qk * f.bb * M) throw LoomError("footprint mismatch on " + w);
        name += std::string("_") + f.name;
        tbits |= f.tables;
        b.push_back(tref(w));
      }
      name += "_" + std::to_string(M) + "_" + std::to_string(K);
      for (int i = 0; i < 5; ++i)
        if (tbits & (1u << i)) b.push_back(tabs[i]);
      if (x.size < std::size_t{K} * 4 || y.size < std::size_t{M} * 4) throw LoomError("GEMV operand too small: " + name);
      b.push_back({x.handle, 0, std::size_t{K} * 4});
      b.push_back({y.handle, 0, std::size_t{M} * 4});
      LoomExecutable& exe = load(name);
      if (overlap && !std::getenv("YAH_DEC_NO_OVERLAP")) gpu.NoBarrierNext();
      dispatch(exe, M / (kR * kW), 1, 32 * kW, b);
    };
    auto rmsnorm = [&](const LoomBuffer& x, const std::string& w, const LoomBuffer& out) {
      dispatch(load("rmsnorm"), 1, 1, 512, {ref(x), tref(w), ref(out)});
    };

    // YAH_DEC_TRACE=1: at step 0, sync after every stage and print max |x| / NaNs
    const bool trace = std::getenv("YAH_DEC_TRACE") != nullptr;
    std::uint32_t cur_pos = 0;
    auto tr = [&](const char* what, std::uint32_t l, const LoomBuffer& b, std::size_t n) {
      if (!trace || cur_pos != 0) return;
      gpu.Synchronize();
      std::vector<float> h(n);
      gpu.D2H(b, h.data(), n * 4);
      double mx = 0; std::size_t nan = 0;
      for (float v : h) { if (v != v) ++nan; else mx = std::max(mx, static_cast<double>(std::fabs(v))); }
      std::fprintf(stderr, "trace l%-2u %-10s max %.4g nan %zu\n", l, what, mx, nan);
    };
    const auto* emb = find("token_embd.weight");
    if (static_cast<std::uint32_t>(emb->type) != 23) throw LoomError("token_embd: only IQ4_XS is wired");
    std::FILE* lf = logits_path.empty() ? nullptr : std::fopen(logits_path.c_str(), "wb");
    std::vector<float> host_logits(logits_path.empty() ? 0 : kVocab);

    // steps are enqueued back to back; the host waits only after the prompt (to time
    // generation alone) and at the end, unless --logits / YAH_DEC_TRACE / YAH_DEC_SYNC
    const bool sync_steps = lf || trace || std::getenv("YAH_DEC_SYNC");
    std::vector<double> step_ms;
    const std::uint32_t n = static_cast<std::uint32_t>(prompt.size());
    auto tgen = std::chrono::steady_clock::now();
    for (std::uint32_t pos = 0; pos < steps; ++pos) {
      const auto t0 = std::chrono::steady_clock::now();
      cur_pos = pos;
      const hrx_buffer_ref_t dposr{posarr.handle, std::size_t{pos} * 4, 4};
      dispatch(load("embed"), 1, 1, kHidden / 16,
               {tref("token_embd.weight"), {toks.handle, std::size_t{pos} * 4, 4}, ref(hidden)});
      for (std::uint32_t l = 0; l < cfg.main_block_count(); ++l) {
        const std::string pre = "blk." + std::to_string(l) + ".";
        rmsnorm(hidden, pre + "attn_norm.weight", normed);
        tr("in", l, hidden, kHidden);
        tr("normed", l, normed, kHidden);
        if (cfg.IsFullAttention(l)) {
          const std::uint32_t ai = l / cfg.full_attention_interval;
          gemv("plain", {pre + "attn_q.weight"}, normed, qg);
          gemv("plain", {pre + "attn_k.weight"}, normed, kb, true);
          gemv("plain", {pre + "attn_v.weight"}, normed, vb, true);
          tr("qg", l, qg, kQProj); tr("k", l, kb, kKv); tr("v", l, vb, kKv);
          dispatch(load("unpack"), kHeads, 1, 256, {ref(qg), ref(q), ref(gate)});
          dispatch(load("rope"), kHeads + kKvHeads, 1, 256,
                   {ref(q), ref(kb), ref(vb), tref(pre + "attn_q_norm.weight"), tref(pre + "attn_k_norm.weight"),
                    ref(q), ref(kb), ref(c32a), ref(c32b), ref(c16a), ref(c16b), dposr, ref(eps)});
          dispatch(load("dattn_kvappend"), kKvHeads, 1, 256,
                   {ref(kb), ref(vb), ref(*kpool[ai]), ref(*vtpool[ai]), ref(ptab), dposr});
          dispatch(load("dattn_part"), pos / 256 + 1, kKvHeads, 256,
                   {ref(q), ref(*kpool[ai]), ref(*vtpool[ai]), ref(ptab), dposr, ref(acc), ref(ml)});
          tr("q_rope", l, q, kAttn); tr("k_rope", l, kb, kKv);
          dispatch(load("dattn_reduce"), kHeads, 1, 256, {ref(acc), ref(ml), ref(gate), dposr, ref(aout)});
          tr("attn", l, aout, kAttn);
          gemv("resid", {pre + "attn_output.weight"}, aout, hidden);
        } else {
          const std::uint32_t si = l - l / cfg.full_attention_interval;
          gemv("plain", {pre + "attn_qkv.weight"}, normed, qkv);
          gemv("plain", {pre + "attn_gate.weight"}, normed, gate, true);
          gemv("plain", {pre + "ssm_alpha.weight"}, normed, alpha, true);
          gemv("plain", {pre + "ssm_beta.weight"}, normed, beta, true);
          tr("qkv", l, qkv, kQkv); tr("z", l, gate, kInner); tr("alpha", l, alpha, kTs); tr("beta", l, beta, kTs);
          dispatch(load("ssmconv"), (kQkv + 255) / 256, 1, 256,
                   {{qkv.handle, 0, std::size_t{kQkv} * 4}, tref(pre + "ssm_conv1d.weight"),
                    {convstate.handle, std::size_t{si} * kConvState * 4, std::size_t{kConvState} * 4}, ref(convout)});
          dispatch(load("deltanet"), kHeadsV, 1, 512,
                   {ref(convout), {dstate.handle, std::size_t{si} * kStateElems * 4, std::size_t{kStateElems} * 4},
                    ref(alpha), ref(beta), tref(pre + "ssm_a"), tref(pre + "ssm_dt.bias"),
                    tref(pre + "ssm_norm.weight"), ref(gate), ref(ssmout)});
          tr("conv", l, convout, kQkv); tr("ssm", l, ssmout, kInner);
          gemv("resid", {pre + "ssm_out.weight"}, ssmout, hidden);
        }
        rmsnorm(hidden, pre + "post_attention_norm.weight", normed);
        tr("mixed", l, hidden, kHidden);
        gemv("swiglu", {pre + "ffn_gate.weight", pre + "ffn_up.weight"}, normed, ffnact);
        tr("ffnact", l, ffnact, kFfn);
        gemv("resid", {pre + "ffn_down.weight"}, ffnact, hidden);
      }
      rmsnorm(hidden, "output_norm.weight", normed);
      tr("head_in", 99, normed, kHidden);
      gemv("plain", {"output.weight"}, normed, logits);
      tr("logits", 99, logits, kVocab);
      dispatch(load("argmax"), 1, 1, 1024,
               {ref(logits), pos + 1 >= n ? hrx_buffer_ref_t{toks.handle, std::size_t{pos + 1} * 4, 4} : ref(token)});
      if (sync_steps) {
        gpu.Synchronize();
        if (lf) {
          gpu.D2H(logits, host_logits.data(), host_logits.size() * 4);
          std::fwrite(host_logits.data(), 4, host_logits.size(), lf);
        }
        step_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      }
      if (pos + 1 == n) {
        gpu.Synchronize();
        tgen = std::chrono::steady_clock::now();
      }
    }
    gpu.Synchronize();
    const double gen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tgen).count();
    if (lf) std::fclose(lf);
    std::vector<std::int32_t> stream(steps + 1);
    gpu.D2H(toks, stream.data(), stream.size() * 4);
    std::vector<std::uint32_t> gen(stream.begin() + n, stream.end());
    for (auto g : gen)
      if (g >= kVocab) throw LoomError("token id out of range: " + std::to_string(g));
    if (!step_ms.empty()) {
      std::fprintf(stderr, "step_ms=");
      for (std::size_t i = 0; i < step_ms.size(); ++i) std::fprintf(stderr, "%.1f%s", step_ms[i], i + 1 == step_ms.size() ? "\n" : " ");
    }
    // decode rate over the generation steps (positions n .. steps - 1), enqueued back to back
    const std::uint32_t dn = steps - n;
    if (dn) std::fprintf(stderr, "decode_ms=%.2f decode_tok_s=%.2f\n", gen_ms / dn, 1000.0 * dn / gen_ms);
    std::printf("generated_ids=");
    for (std::size_t i = 0; i < gen.size(); ++i) std::printf("%u%s", gen[i], i + 1 == gen.size() ? "" : " ");
    std::printf("\ngenerated_text=%s\n", tokenizer.Decode(gen).c_str());
  } catch (const std::exception& error) {
    std::fprintf(stderr, "loom_decode: %s\n", error.what());
    return 1;
  }
  return 0;
}
