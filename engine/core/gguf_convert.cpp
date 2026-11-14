// A GGUF model as the engine reads it: the projections keep their bytes (the sets run the GGUF's own formats), ffn_up
// right after ffn_gate (the NPU FFN block's image reads gate through up rows in one view); token_embd becomes Q8_0, as
// does an output.weight that is not Q6_K or Q8_0, and the small tensors f32.
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/gguf.hpp"
#include "core/gguf_quant.hpp"

namespace yah::core {
namespace {

enum class Kind { kCopy, kQ8_0, kF32 };

struct Item {
  TensorInfo dst;
  const TensorInfo* src;
  Kind kind;
};

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
  const auto projection = [&](const TensorInfo& t) {
    if (t.dims[0] % 256) throw std::runtime_error("gguf: " + t.name + " rows are not a multiple of 256 long");
    if (!CanDequantize(t.type))
      throw std::runtime_error(std::string("gguf: tensor type ") + TypeName(t.type) + " of " + t.name +
                               " is not supported (F32, F16, BF16, Q4_0 .. Q8_0, Q2_K .. Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, "
                               "IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS are)");
    // the head GEMVs read Q6_K (yah_gemv_q6k) and Q8_0 (gen_gemv): a head of another format becomes Q8_0
    if (t.name == "output.weight" && t.type != GgmlType::kQ6_K && t.type != GgmlType::kQ8_0)
      return add(&t, t.name, t.dims, GgmlType::kQ8_0, Kind::kQ8_0);
    add(&t, t.name, t.dims, t.type, Kind::kCopy);
  };
  for (const auto& t : src.tensors()) {
    if (t.name.ends_with("ffn_up.weight") && by_name.contains(t.name.substr(0, t.name.size() - 13) + "ffn_gate.weight"))
      continue;  // after ffn_gate
    if (IsProjection(t)) {
      const auto up = t.name.ends_with("ffn_gate.weight")
                          ? by_name.find(t.name.substr(0, t.name.size() - 15) + "ffn_up.weight")
                          : by_name.end();
      projection(t);
      if (up != by_name.end()) projection(*up->second);
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

Gguf Gguf::OpenConverted(const std::string& path, bool weights) {
  // the converter's state: the source mapping and its jobs (row ranges of about 32 MB of f32 rows), in copy order
  struct Job {
    const Item* item;
    std::uint64_t r0, r1;
  };
  struct State {
    Gguf src;
    std::vector<Item> items;
    std::vector<Job> jobs;
    std::uint8_t* base = nullptr;
    std::size_t next = 0;   // a lazy fill: jobs [0, next) are done
    // a streamed fill (Stream): the ranges, how many the reader filled and Prepare handed out, the reader's failure
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    std::size_t filled = 0, handed = 0;
    bool stop = false;
    std::string error;
    std::mutex mu;
    std::condition_variable cv;
    std::thread reader;
    ~State() {
      {
        const std::lock_guard<std::mutex> l(mu);
        stop = true;
      }
      cv.notify_all();
      if (reader.joinable()) reader.join();
    }
  };
  auto st = std::make_shared<State>();
  st->src = Open(path);
  std::uint64_t total = 0;
  st->items = Plan(st->src, &total);
  Gguf g;
  g.version_ = 3;
  g.data_offset_ = 0;
  g.size_ = total;
  g.meta_ = st->src.meta_;
  for (const Item& it : st->items) g.tensors_.push_back(it.dst);
  if (!weights) return g;

  constexpr std::size_t kHuge = std::size_t{2} << 20;
  const std::size_t bytes = (total + kHuge - 1) & ~(kHuge - 1);
  g.base_ = Reserve(bytes, path), g.map_bytes_ = bytes;
  st->base = g.base_;
  ::madvise(st->src.base_, st->src.size_, MADV_SEQUENTIAL);
  for (const Item& it : st->items) {
    const std::uint64_t k = it.dst.dims[0], rows = it.dst.elements / k;
    const std::uint64_t step = std::max<std::uint64_t>(1, (std::uint64_t{8} << 20) / k);
    for (std::uint64_t r = 0; r < rows; r += step) st->jobs.push_back({&it, r, std::min(rows, r + step)});
  }
  // jobs [j0, j1) on every core
  const auto run = [](State& s, std::size_t j0, std::size_t j1) {
    std::atomic<std::size_t> next{j0};
    std::atomic<bool> failed{false};
    std::string failure;
    const auto worker = [&] {
      std::vector<float> buf;
      try {
        for (std::size_t i; !failed && (i = next.fetch_add(1)) < j1;) {
          const Job& j = s.jobs[i];
          const TensorInfo& d = j.item->dst;
          const std::uint64_t k = d.dims[0], n = (j.r1 - j.r0) * k;
          std::uint8_t* out = s.base + d.offset + j.r0 * k / BlockElements(d.type) * BlockBytes(d.type);
          const TensorInfo& t = *j.item->src;
          const std::uint8_t* in = s.src.Data(t) + j.r0 * k / BlockElements(t.type) * BlockBytes(t.type);
          switch (j.item->kind) {
            case Kind::kCopy:
              std::memcpy(out, in, n / BlockElements(d.type) * BlockBytes(d.type));
              break;
            case Kind::kF32:
              Dequantize(j.item->src->type, in, reinterpret_cast<float*>(out), n);
              break;
            case Kind::kQ8_0:
              buf.resize(n);
              Dequantize(j.item->src->type, in, buf.data(), n);
              QuantizeQ8_0(buf.data(), n, out);
              break;
          }
          // the source pages are read once: drop them as we go (a GGUF and its copy may not both fit)
          const std::size_t len = n / BlockElements(t.type) * BlockBytes(t.type);
          const auto lo = (reinterpret_cast<std::uintptr_t>(in) + 4095) & ~std::uintptr_t{4095};
          const auto hi = (reinterpret_cast<std::uintptr_t>(in) + len) & ~std::uintptr_t{4095};
          if (hi > lo) {
            ::madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_DONTNEED);
            ::posix_fadvise(s.src.fd_, static_cast<off_t>(lo - reinterpret_cast<std::uintptr_t>(s.src.base_)),
                            static_cast<off_t>(hi - lo), POSIX_FADV_DONTNEED);
          }
        }
      } catch (const std::exception& e) {
        if (!failed.exchange(true)) failure = e.what();
      }
    };
    std::vector<std::thread> pool;
    const unsigned threads = std::max(1U, std::thread::hardware_concurrency());
    for (unsigned i = 0; i < threads; ++i) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    if (failed) throw std::runtime_error(failure);
  };
  {
    // filled range by range as the device upload reads it (Prepare), each range's host pages dropped after (the
    // uploader's ReleaseData): the host never holds the whole copy next to the device's
    for (std::size_t i = 1; i < st->jobs.size(); ++i)
      if (st->jobs[i].item->dst.offset < st->jobs[i - 1].item->dst.offset)
        throw std::runtime_error("gguf: the copy's tensors are out of order");
    // rows from the source (Rows: a tensor the device skipped, never filled)
    g.rows_ = [st](const TensorInfo& t, const std::uint32_t* rows, std::size_t n, std::uint8_t* out) {
      const Item* it = nullptr;
      for (const Item& i : st->items)
        if (i.dst.name == t.name) it = &i;
      if (!it) return false;
      const TensorInfo& s = *it->src;
      const std::uint64_t k = s.dims[0];
      const std::size_t sb = k / BlockElements(s.type) * BlockBytes(s.type), db = k / BlockElements(t.type) * BlockBytes(t.type);
      std::vector<float> buf(k);
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint8_t* in = st->src.Data(s) + std::size_t{rows[i]} * sb;
        if (it->kind == Kind::kCopy) {
          std::memcpy(out + i * db, in, db);
        } else if (it->kind == Kind::kQ8_0) {
          Dequantize(s.type, in, buf.data(), k);
          QuantizeQ8_0(buf.data(), k, out + i * db);
        } else {
          throw std::runtime_error("gguf: Rows of " + t.name + " from the source: not a copy or Q8_0 item");
        }
      }
      return true;
    };
    // a job's bytes in the copy (dst) and in the file (src)
    const auto range = [](const Job& j) {
      const TensorInfo& d = j.item->dst;
      const std::uint64_t rb = d.dims[0] / BlockElements(d.type) * BlockBytes(d.type);
      return std::pair{d.offset + j.r0 * rb, d.offset + j.r1 * rb};
    };
    const auto src_range = [](const State& s, const Job& j) {
      const TensorInfo& t = *j.item->src;
      const std::uint64_t rb = t.dims[0] / BlockElements(t.type) * BlockBytes(t.type);
      const std::size_t at = s.src.data_offset_ + t.offset;
      return std::pair{at + j.r0 * rb, at + j.r1 * rb};
    };
    // Stream: the reader fills range k once Prepare handed out range k - 1 (the host holds at most the range in upload
    // and the next). Copies are read from the file straight into the copy, contiguous jobs in one read, each read's
    // cached pages dropped (as ReadAll); the other jobs convert as the lazy fill does.
    g.stream_ = [st, run, range, src_range](std::vector<std::pair<std::size_t, std::size_t>> ranges) {
      State& s = *st;
      if (s.reader.joinable() || !s.ranges.empty()) throw std::runtime_error("gguf: Stream called twice");
      s.ranges = std::move(ranges);
      ::posix_fadvise(s.src.fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
      s.reader = std::thread([&s, run, range, src_range] {
        try {
          std::size_t j = 0;
          for (std::size_t k = 0; k < s.ranges.size(); ++k) {
            {
              std::unique_lock<std::mutex> l(s.mu);
              s.cv.wait(l, [&] { return s.stop || s.handed >= k; });
              if (s.stop) return;
            }
            const auto [offset, len] = s.ranges[k];
            while (j < s.jobs.size() && range(s.jobs[j]).second <= offset) ++j;
            std::size_t j1 = j;
            while (j1 < s.jobs.size() && range(s.jobs[j1]).first < offset + len) ++j1;
            for (std::size_t i = j; i < j1;) {
              if (s.jobs[i].item->kind != Kind::kCopy) {
                std::size_t e = i + 1;
                while (e < j1 && s.jobs[e].item->kind != Kind::kCopy) ++e;
                run(s, i, e);
                i = e;
                continue;
              }
              auto [d0, d1] = range(s.jobs[i]);
              auto [f0, f1] = src_range(s, s.jobs[i]);
              for (++i; i < j1 && s.jobs[i].item->kind == Kind::kCopy && d1 - d0 < (std::size_t{256} << 20); ++i) {
                const auto [e0, e1] = range(s.jobs[i]);
                const auto [g0, g1] = src_range(s, s.jobs[i]);
                if (e0 != d1 || g0 != f1) break;
                d1 = e1, f1 = g1;
              }
              for (std::size_t done = 0; done < d1 - d0;) {
                const ssize_t n = ::pread(s.src.fd_, s.base + d0 + done, d1 - d0 - done, static_cast<off_t>(f0 + done));
                if (n <= 0) throw std::runtime_error("gguf: read failed in the streamed fill");
                done += static_cast<std::size_t>(n);
              }
              ::posix_fadvise(s.src.fd_, static_cast<off_t>(f0), static_cast<off_t>(f1 - f0), POSIX_FADV_DONTNEED);
            }
            j = j1;
            {
              const std::lock_guard<std::mutex> l(s.mu);
              s.filled = k + 1;
            }
            s.cv.notify_all();
          }
        } catch (const std::exception& e) {
          {
            const std::lock_guard<std::mutex> l(s.mu);
            s.error = e.what();
          }
          s.cv.notify_all();
        }
      });
    };
    // streamed: wait for the reader's range (Prepare takes the ranges in Stream's order); else the jobs overlapping the
    // range run here, those wholly before it skipped (tensors the device does not take)
    g.fill_ = [st, run, range](std::size_t offset, std::size_t len) {
      if (!st->ranges.empty()) {
        State& s = *st;
        std::unique_lock<std::mutex> l(s.mu);
        const std::size_t k = s.handed;
        if (k >= s.ranges.size() || s.ranges[k] != std::pair{offset, len})
          throw std::runtime_error("gguf: Prepare out of the streamed order");
        s.cv.wait(l, [&] { return s.filled > k || !s.error.empty(); });
        if (!s.error.empty()) throw std::runtime_error(s.error);
        s.handed = k + 1;
        l.unlock();
        s.cv.notify_all();
        return;
      }
      std::size_t j0 = st->next;
      while (j0 < st->jobs.size() && range(st->jobs[j0]).second <= offset) ++j0;
      std::size_t j1 = j0;
      while (j1 < st->jobs.size() && range(st->jobs[j1]).first < offset + len) ++j1;
      if (j1 > j0) run(*st, j0, j1);
      st->next = j1;
    };
    return g;
  }
}

}  // namespace yah::core
