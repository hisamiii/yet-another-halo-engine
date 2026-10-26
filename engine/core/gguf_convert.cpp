// A GGUF model opened as the engine reads the HFQ one (core/hfq.cpp): the GPU kernels take MQ4G256 weights in the
// rotated basis, so at load every projection is dequantized (core/gguf_quant.cpp: the common GGML quants), rotated
// and re-quantized, with all-ones AWQ scales; token_embd becomes Q8_0 and the small tensors f32. GGUF already stores
// the names, dims, norms (1 + w), DeltaNet head order and metadata the engine expects, but for ssm_out's columns: the
// engine gathers its input back into HF value-head order (rot_heads), as the HFQ stores ssm_out, so they are
// permuted from GGUF order here. The tensor layout matches the
// HFQ's where the sets depend on it: a layer's ffn_up starts ffn_gate's bytes plus the HFQ's f16 gate awq after it
// (the NPU FFN block's image reads gate through up rows in one view, at that stride).
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

#include "core/gguf.hpp"
#include "core/gguf_quant.hpp"

namespace yah::core {
namespace {

enum class Kind { kCopy, kMq4, kQ8_0, kF32, kOnes };

struct Item {
  TensorInfo dst;
  const TensorInfo* src;  // null for kOnes
  Kind kind;
  bool heads = false;  // ssm_out: columns from GGUF to HF value-head order
};

// GGUF value head i is HF head order[i] (core/hfq.cpp HeadOrder): GGUF tiles the value heads over the key heads.
struct Heads {
  std::uint32_t value = 0, key = 0, dim = 0;
  [[nodiscard]] std::uint32_t Hf(std::uint32_t i) const { return value / key * (i % key) + i / key; }
};

Heads HeadsOf(const Gguf& g) {
  const auto u = [&](const char* k) -> std::uint64_t {
    const MetadataValue* v = g.Meta(std::string("qwen35.") + k);
    return v ? v->u : 0;
  };
  Heads h{static_cast<std::uint32_t>(u("ssm.time_step_rank")), static_cast<std::uint32_t>(u("ssm.group_count")), 0};
  if (h.value && h.key) h.dim = static_cast<std::uint32_t>(u("ssm.inner_size") / h.value);
  return h;
}

bool IsProjection(const TensorInfo& t) {
  static const char* const kSites[] = {"attn_q",    "attn_k",    "attn_v",       "attn_output", "attn_qkv",
                                       "attn_gate", "ssm_alpha", "ssm_beta",     "ssm_out",     "ffn_gate",
                                       "ffn_up",    "ffn_down",  "nextn.eh_proj"};
  if (t.dims.size() != 2) return false;
  if (t.name == "output.weight") return true;
  if (!t.name.starts_with("blk.") || !t.name.ends_with(".weight")) return false;
  const std::size_t dot = t.name.find('.', 4);
  const std::string site = t.name.substr(dot + 1, t.name.size() - dot - 1 - 7);
  return std::any_of(std::begin(kSites), std::end(kSites), [&](const char* s) { return site == s; });
}

std::vector<Item> Plan(const Gguf& src, std::uint64_t* total) {
  std::map<std::string, const TensorInfo*> by_name;
  for (const auto& t : src.tensors()) by_name[t.name] = &t;
  std::vector<Item> items;
  std::uint64_t off = 0;
  const auto add = [&](const TensorInfo* s, std::string name, std::vector<std::uint64_t> dims, GgmlType type, Kind k) {
    Item it{{}, s, k};
    it.dst.name = std::move(name);
    it.dst.dims = std::move(dims);
    it.dst.type = type;
    it.dst.elements = 1;
    for (const auto d : it.dst.dims) it.dst.elements *= d;
    it.dst.bytes = it.dst.elements / BlockElements(type) * BlockBytes(type);
    it.dst.offset = off;
    off += (it.dst.bytes + 255) / 256 * 256;
    items.push_back(std::move(it));
  };
  const auto awq_of = [&](const TensorInfo& t) {
    const std::string awq = t.name.substr(0, t.name.size() - 7) + ".awq";
    const auto it = by_name.find(awq);
    add(it == by_name.end() ? nullptr : it->second, awq, {t.dims[0]}, GgmlType::kF32,
        it == by_name.end() ? Kind::kOnes : Kind::kCopy);
  };
  const auto projection = [&](const TensorInfo& t, bool awq) {
    if (t.dims[0] % 256) throw std::runtime_error("gguf: " + t.name + " rows are not a multiple of 256 long");
    if (!CanDequantize(t.type))
      throw std::runtime_error(std::string("gguf: tensor type ") + TypeName(t.type) + " of " + t.name +
                               " is not supported (F32, F16, BF16, Q4_0 .. Q8_0, Q2_K .. Q6_K, IQ2_XXS, IQ2_XS, "
                               "IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS are)");
    add(&t, t.name, t.dims, GgmlType::kMQ4G256, t.type == GgmlType::kMQ4G256 ? Kind::kCopy : Kind::kMq4);
    items.back().heads = t.name.ends_with(".ssm_out.weight") && items.back().kind == Kind::kMq4;
    if (awq && t.name != "output.weight") awq_of(t);
  };
  for (const auto& t : src.tensors()) {
    if (t.name.ends_with(".awq")) continue;  // with its projection
    if (t.name.ends_with("ffn_up.weight") && by_name.contains(t.name.substr(0, t.name.size() - 13) + "ffn_gate.weight"))
      continue;  // after ffn_gate
    if (IsProjection(t)) {
      const auto up = t.name.ends_with("ffn_gate.weight")
                          ? by_name.find(t.name.substr(0, t.name.size() - 15) + "ffn_up.weight")
                          : by_name.end();
      projection(t, up == by_name.end());
      if (up != by_name.end()) {
        off += (t.dims[0] * 2 + 255) / 256 * 256;  // where the HFQ keeps the f16 gate awq
        projection(*up->second, true);
        awq_of(t);
      }
    } else if (t.name == "token_embd.weight") {
      if (!CanDequantize(t.type)) throw std::runtime_error(std::string("gguf: token_embd type ") + TypeName(t.type));
      add(&t, t.name, t.dims, GgmlType::kQ8_0, t.type == GgmlType::kQ8_0 ? Kind::kCopy : Kind::kQ8_0);
    } else {
      if (!CanDequantize(t.type)) throw std::runtime_error(std::string("gguf: tensor type ") + TypeName(t.type));
      add(&t, t.name, t.dims, GgmlType::kF32, t.type == GgmlType::kF32 ? Kind::kCopy : Kind::kF32);
    }
  }
  *total = off;
  return items;
}

}  // namespace

Gguf Gguf::OpenConverted(const std::string& path, bool weights, bool lock) {
  Gguf src = Open(path);
  std::uint64_t total = 0;
  std::vector<Item> items = Plan(src, &total);
  const Heads heads = HeadsOf(src);
  for (const Item& it : items)
    if (it.heads && (!heads.dim || it.dst.dims[0] != std::uint64_t{heads.value} * heads.dim))
      throw std::runtime_error("gguf: " + it.dst.name + " does not match the ssm head metadata");
  Gguf g;
  g.version_ = 3;
  g.data_offset_ = 0;
  g.size_ = total;
  g.meta_ = src.meta_;
  for (const Item& it : items) g.tensors_.push_back(it.dst);
  if (!weights) return g;

  constexpr std::size_t kHuge = std::size_t{2} << 20;
  const std::size_t bytes = (total + kHuge - 1) & ~(kHuge - 1);
  g.base_ = Reserve(bytes, path), g.map_bytes_ = bytes;
  ::madvise(src.base_, src.size_, MADV_SEQUENTIAL);

  // jobs: row ranges of about 32 MB of f32 rows
  struct Job {
    const Item* item;
    std::uint64_t r0, r1;
  };
  std::vector<Job> jobs;
  std::uint64_t work = 0;
  bool convert = false;
  for (const Item& it : items) {
    const std::uint64_t k = it.dst.dims[0], rows = it.dst.elements / k;
    const std::uint64_t step = std::max<std::uint64_t>(1, (std::uint64_t{8} << 20) / k);
    for (std::uint64_t r = 0; r < rows; r += step) jobs.push_back({&it, r, std::min(rows, r + step)});
    work += it.dst.bytes;
    convert = convert || it.kind == Kind::kMq4 || it.kind == Kind::kQ8_0;
  }
  const auto t0 = std::chrono::steady_clock::now();
  if (convert) std::fprintf(stderr, "gguf: %s: converting the weights to MQ4 for the GPU kernels\n", path.c_str());
  std::atomic<std::size_t> next{0};
  std::atomic<std::uint64_t> done{0};
  std::atomic<bool> failed{false};
  std::string failure;
  const auto run = [&] {
    std::vector<float> buf, row;
    try {
      for (std::size_t i; !failed && (i = next.fetch_add(1)) < jobs.size();) {
        const Job& j = jobs[i];
        const TensorInfo& d = j.item->dst;
        const std::uint64_t k = d.dims[0], n = (j.r1 - j.r0) * k;
        std::uint8_t* out = g.base_ + d.offset + j.r0 * k / BlockElements(d.type) * BlockBytes(d.type);
        const std::uint8_t* in = nullptr;
        if (j.item->src) {
          const TensorInfo& s = *j.item->src;
          in = src.Data(s) + j.r0 * k / BlockElements(s.type) * BlockBytes(s.type);
        }
        switch (j.item->kind) {
          case Kind::kCopy:
            std::memcpy(out, in, n / BlockElements(d.type) * BlockBytes(d.type));
            break;
          case Kind::kOnes:
            std::fill_n(reinterpret_cast<float*>(out), n, 1.0F);
            break;
          case Kind::kF32:
            Dequantize(j.item->src->type, in, reinterpret_cast<float*>(out), n);
            break;
          case Kind::kQ8_0:
          case Kind::kMq4:
            buf.resize(n);
            Dequantize(j.item->src->type, in, buf.data(), n);
            if (j.item->kind == Kind::kQ8_0) {
              QuantizeQ8_0(buf.data(), n, out);
            } else {
              if (j.item->heads) {
                row.resize(k);
                for (std::uint64_t r = 0; r < j.r1 - j.r0; ++r) {
                  float* x = buf.data() + r * k;
                  for (std::uint32_t i = 0; i < heads.value; ++i)
                    std::memcpy(row.data() + std::size_t{heads.Hf(i)} * heads.dim, x + std::size_t{i} * heads.dim,
                                heads.dim * 4);
                  std::memcpy(x, row.data(), k * 4);
                }
              }
              for (std::uint64_t r = 0; r < j.r1 - j.r0; ++r)
                QuantizeMq4Row(buf.data() + r * k, k, out + r * k / 256 * 136);
            }
            break;
        }
        done += n / BlockElements(d.type) * BlockBytes(d.type);
        if (in) {
          // the source pages are read once: drop them as we go (a GGUF and its converted copy may not both fit)
          const std::size_t len = n / BlockElements(j.item->src->type) * BlockBytes(j.item->src->type);
          const auto lo = (reinterpret_cast<std::uintptr_t>(in) + 4095) & ~std::uintptr_t{4095};
          const auto hi = (reinterpret_cast<std::uintptr_t>(in) + len) & ~std::uintptr_t{4095};
          if (hi > lo) {
            ::madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_DONTNEED);
            ::posix_fadvise(src.fd_, static_cast<off_t>(lo - reinterpret_cast<std::uintptr_t>(src.base_)),
                            static_cast<off_t>(hi - lo), POSIX_FADV_DONTNEED);
          }
        }
      }
    } catch (const std::exception& e) {
      if (!failed.exchange(true)) failure = e.what();
    }
  };
  std::vector<std::thread> pool;
  const unsigned threads = std::max(1U, std::thread::hardware_concurrency());
  for (unsigned i = 0; i < threads; ++i) pool.emplace_back(run);
  // progress: one line rewritten on a terminal, a line per quarter in a log
  const bool tty = ::isatty(2) != 0;
  for (int pct = -1; convert;) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const int now = static_cast<int>(done * 100 / std::max<std::uint64_t>(1, work));
    if (tty && now != pct) std::fprintf(stderr, "\rgguf: converting, %d%%", pct = now);
    if (!tty && now / 25 != pct / 25 && now < 100) std::fprintf(stderr, "gguf: converting, %d%%\n", pct = now);
    if (done == work || failed) break;
  }
  for (auto& th : pool) th.join();
  if (failed) {
    if (convert && tty) std::fprintf(stderr, "\n");
    throw std::runtime_error(failure);
  }
  if (convert)
    std::fprintf(stderr, "%sgguf: converted in %.1f s\n", tty ? "\r" : "",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  if (lock && ::mlock(g.base_, bytes) != 0)
    std::fprintf(stderr,
                 "gguf: cannot lock the model in memory (RLIMIT_MEMLOCK); it may be paged out under memory "
                 "pressure. To lock it: a line '<user> - memlock unlimited' in /etc/security/limits.d/, then log in "
                 "again\n");
  return g;
}

}  // namespace yah::core
