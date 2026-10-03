// Prefill GEMM calibration while serving: picks each tile GEMM's variant per token bucket from measurements of real
// prefill chunks, and keeps the picks in a file.
//
// A served HAL set carries, next to each tile GEMM "<gemm>.hal", variants that compute the same values with other tiles:
// "<gemm>.t<BN>.hal" (narrow token tiles) and "<gemm>.m<i>.hal" (the calibration menu, engine/tune/tune.py menu; every
// one verified bit-identical when emitted). While a GEMM's bucket is open, each of its occurrences (layers) in a chunk
// runs the plain GEMM or, by a hash of (GEMM, chunk, occurrence), the open challenger with the fewest samples; GEMMs
// that share a layer vary independently. The device timestamps of the chunk's graph (HRX patch 0006) time each
// occurrence by the span of its overlap group (the dispatches running at the same time as it, transitively), so a
// variant that only takes GPU time from a concurrent kernel gains nothing. A challenger occurrence gives the sample
// log(span / mean span of the plain GEMM's occurrences in the same chunk): the chunk shares one clock, so the ratio does
// not drift with temperature. A challenger is dropped once it is clearly slower than the plain GEMM or the best
// challenger; the bucket closes when one arm is left or every arm has kMaxSamples, and then takes the best challenger
// only if it is clearly faster than the plain GEMM. Outputs are the same whichever variant runs (the K / V quantizers
// ignore the padding rows the tiles differ in), so calibration only moves time.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "hrx_runtime.h"

namespace yah::model {

class PrefillCalib {
 public:
  static constexpr std::uint32_t kBuckets[] = {64, 128, 256, 512, 1024, 2048};
  static constexpr std::uint32_t kMaxSamples = 12;
  static constexpr double kZ = 2.5;          // standard errors for "clearly"
  // Spread of one sample (measured 2026-10-03: median 7%, p90 11-24% by bucket): the floor, and the prior while an
  // arm has one sample (so an arm 40% slower goes after one).
  static constexpr double kMinSd = 0.03, kPriorSd = 0.15;

  // hals: every HAL of the set -> its token tile (dispatch.txt). path: the state file ("": keep nothing).
  PrefillCalib(const std::map<std::string, std::uint32_t>& hals, std::string path) : path_(std::move(path)) {
    for (const auto& [hal, bn] : hals) {
      const auto dot = hal.rfind(".hal");
      if (dot == std::string::npos || dot + 4 != hal.size() || hal.compare(0, 5, "gemm_") != 0) continue;
      const std::string stem = hal.substr(0, dot);
      if (stem.find('.') != std::string::npos) continue;  // a variant, not a GEMM
      std::vector<std::pair<std::string, std::uint32_t>> v{{hal, bn}};
      for (const auto& [h, b] : hals)
        if (h.size() > stem.size() + 1 && h.compare(0, stem.size() + 1, stem + ".") == 0 &&
            (h[stem.size() + 1] == 't' || h[stem.size() + 1] == 'm'))
          v.emplace_back(h, b);
      if (v.size() > 1) variants_[hal] = std::move(v);
    }
    Load();
  }

  [[nodiscard]] static std::uint32_t Bucket(std::uint32_t n) {
    for (std::uint32_t b : kBuckets)
      if (n <= b) return b;
    return kBuckets[std::size(kBuckets) - 1];
  }

  // A new chunk of n real tokens: whether it calibrates anything (then profile its graph).
  bool BeginChunk(std::uint32_t n) {
    bucket_ = Bucket(n);
    seen_.clear();
    ++chunk_;
    bool known = false;  // a bucket no chunk ran yet: calibrate to learn which GEMMs it runs
    for (const auto& [hal, m] : cells_) {
      const auto it = m.find(bucket_);
      if (it == m.end()) continue;
      known = true;
      if (!it->second.closed) return true;
    }
    return !known;
  }

  // The HAL to run for this occurrence of GEMM hal in the chunk, or "" to leave the choice to the caller. tag: what
  // to record for the node (-1: nothing).
  std::string Choose(const std::string& hal, int* tag) {
    *tag = -1;
    const auto it = variants_.find(hal);
    if (it == variants_.end()) return "";
    auto& c = Cell(hal);
    if (c.closed) return c.pick;  // "" for the caller's choice
    const std::uint32_t k = seen_[hal]++;
    int arm = 0;
    if (Mix(hal, chunk_, k) & 1) {  // a challenger: the open one with the fewest samples (including this chunk's)
      std::uint32_t least = ~0u;
      for (std::size_t a = 1; a < c.arms.size(); ++a) {
        const Arm& r = c.arms[a];
        if (r.alive && r.n + r.pending < least) least = r.n + r.pending, arm = static_cast<int>(a);
      }
    }
    ++c.arms[arm].pending;
    *tag = static_cast<int>(tags_.size());
    tags_.push_back({hal, bucket_, arm});
    return c.arms[arm].hal;
  }

  // A chunk's graph ran: node_tags[i] is the tag Choose returned for graph node i (or -1), events are that graph's
  // dispatches sorted by command index; their k-th entry is node k. A chunk whose grids do not match is skipped.
  void EndChunk(const std::vector<int>& node_tags, const std::vector<std::array<std::uint32_t, 3>>& node_grids,
                const std::vector<hrx_profile_dispatch_t>& events) {
    if (events.size() != node_tags.size()) return;
    for (std::size_t i = 0; i < events.size(); ++i)
      for (int d = 0; d < 3; ++d)
        if (events[i].workgroup_count[d] != node_grids[i][d]) return;
    std::map<std::pair<std::string, std::uint32_t>, int> count;  // occurrences per GEMM bucket in this chunk
    for (int t : node_tags)
      if (t >= 0) ++count[{tags_[t].hal, tags_[t].bucket}];
    // The span of each dispatch's overlap group: dispatches sorted by start, merged while they overlap.
    std::vector<std::size_t> order(events.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return events[a].start_tick < events[b].start_tick; });
    std::vector<double> span(events.size());
    for (std::size_t g = 0; g < order.size();) {
      std::uint64_t lo = events[order[g]].start_tick, hi = events[order[g]].end_tick;
      std::size_t e = g + 1;
      while (e < order.size() && events[order[e]].start_tick < hi) hi = std::max(hi, events[order[e]].end_tick), ++e;
      for (std::size_t i = g; i < e; ++i) span[order[i]] = static_cast<double>(hi - lo);
      g = e;
    }
    // per GEMM: the plain GEMM's mean time and the challenger samples
    std::map<std::pair<std::string, std::uint32_t>, std::pair<double, int>> base;
    for (std::size_t i = 0; i < events.size(); ++i) {
      if (node_tags[i] < 0) continue;
      const Tag& t = tags_[node_tags[i]];
      if (t.arm == 0) {
        auto& b = base[{t.hal, t.bucket}];
        b.first += span[i];
        ++b.second;
      }
    }
    std::map<std::pair<std::string, std::uint32_t>, bool> touched;
    for (std::size_t i = 0; i < events.size(); ++i) {
      if (node_tags[i] < 0) continue;
      const Tag& t = tags_[node_tags[i]];
      const auto b = base.find({t.hal, t.bucket});
      if (t.arm == 0 || b == base.end() || !b->second.second) continue;
      const double d = span[i];
      const double d0 = b->second.first / b->second.second;
      if (d <= 0 || d0 <= 0) continue;
      auto& a = cells_[t.hal][t.bucket].arms[t.arm];
      const double x = std::log(d / d0);
      a.n += 1, a.sum += x, a.sum2 += x * x;
      touched[{t.hal, t.bucket}] = true;
    }
    for (const auto& [key, k] : count) {
      auto& c = cells_[key.first][key.second];
      if (k < 2) {  // one run per chunk: no plain GEMM to pair with; leave the choice to the caller
        c.closed = true, c.pick.clear();
      } else if (touched.count(key)) {
        ++c.chunks;
        Decide(c);
      }
    }
  }

  // The profile session's chunks are all recorded: forget their tags and store the state.
  void EndSession() {
    tags_.clear();
    for (auto& [hal, m] : cells_)
      for (auto& [b, c] : m)
        for (Arm& a : c.arms) a.pending = 0;
    Save();
  }

  // For logs: closed / open GEMM buckets.
  [[nodiscard]] std::pair<int, int> Progress() const {
    int closed = 0, open = 0;
    for (const auto& [hal, m] : cells_)
      for (const auto& [b, c] : m) (c.closed ? closed : open)++;
    return {closed, open};
  }

 private:
  struct Arm {
    std::string hal;
    std::uint32_t bn = 0;
    bool alive = true;
    std::uint32_t n = 0, pending = 0;  // samples; occurrences assigned in the open session
    double sum = 0, sum2 = 0;
    [[nodiscard]] double Mean() const { return n ? sum / n : 0.0; }
    [[nodiscard]] double Se() const {
      if (n == 0) return 1e9;
      if (n == 1) return kPriorSd;
      const double var = std::max((sum2 - sum * sum / n) / (n - 1), kMinSd * kMinSd);
      return std::sqrt(var / n);
    }
  };
  struct CellState {
    std::vector<Arm> arms;  // arms[0]: the plain GEMM
    std::uint32_t chunks = 0;
    bool closed = false;
    std::string pick;
  };
  struct Tag {
    std::string hal;
    std::uint32_t bucket;
    int arm;
  };

  CellState& Cell(const std::string& hal) {
    auto& c = cells_[hal][bucket_];
    if (c.arms.empty()) {
      // Arms for this bucket: the variants whose padding at the bucket's token count is at most 2x the least.
      const auto& v = variants_.at(hal);
      std::uint32_t least = ~0u;
      for (const auto& [h, bn] : v) least = std::min(least, (bucket_ + bn - 1) / bn * bn);
      for (const auto& [h, bn] : v)
        if (h == hal || (bucket_ + bn - 1) / bn * bn <= 2 * least) c.arms.push_back({h, bn});
      if (c.arms.size() == 1) c.closed = true, c.pick = hal;
    }
    return c;
  }

  static std::uint64_t Mix(const std::string& s, std::uint64_t a, std::uint64_t b) {
    std::uint64_t h = 1469598103934665603ull;
    for (unsigned char ch : s) h = (h ^ ch) * 1099511628211ull;
    h ^= a * 0x9e3779b97f4a7c15ull + b * 0xc2b2ae3d27d4eb4full;
    h ^= h >> 31, h *= 0x7fb5d329728ea185ull, h ^= h >> 27;
    return h >> 7;
  }

  static void Decide(CellState& c) {
    // best: the lowest mean among arms with enough samples (the plain GEMM is 0 by definition)
    double best = 0.0, best_se = 0.0;
    for (std::size_t a = 1; a < c.arms.size(); ++a) {
      const Arm& r = c.arms[a];
      if (r.alive && r.n >= 2 && r.Mean() < best) best = r.Mean(), best_se = r.Se();
    }
    bool open = false, capped = true;
    for (std::size_t a = 1; a < c.arms.size(); ++a) {
      Arm& r = c.arms[a];
      if (!r.alive) continue;
      if (r.n >= 1 && (r.Mean() - kZ * r.Se() > 0.0 || r.Mean() - kZ * r.Se() > best + kZ * best_se))
        r.alive = false;
      if (r.alive) open = true, capped = capped && r.n >= kMaxSamples;
    }
    if (open && !capped) return;
    c.closed = true;
    c.pick = c.arms[0].hal;
    double pick = 0.0;
    for (std::size_t a = 1; a < c.arms.size(); ++a) {
      const Arm& r = c.arms[a];
      if (r.alive && r.n >= 2 && r.Mean() + kZ * r.Se() < 0.0 && r.Mean() < pick) pick = r.Mean(), c.pick = r.hal;
    }
  }

  // The picks file: "<gemm> <bucket> closed <pick>" or "<gemm> <bucket> arm <hal> <alive> <n> <sum> <sum2>" lines.
  void Load() {
    std::ifstream f(path_);
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream s(line);
      std::string hal, what, h;
      std::uint32_t bucket = 0;
      if (!(s >> hal >> bucket >> what >> h) || !variants_.count(hal)) continue;
      const std::uint32_t saved = bucket_;
      bucket_ = bucket;
      auto& c = Cell(hal);
      bucket_ = saved;
      if (what == "closed") {
        bool known = h == "-";
        for (const Arm& a : c.arms) known = known || a.hal == h;
        if (known) c.closed = true, c.pick = h == "-" ? "" : h;
      } else if (what == "arm") {
        for (Arm& a : c.arms)
          if (a.hal == h) {
            int alive = 1;
            s >> alive >> a.n >> a.sum >> a.sum2 >> c.chunks;
            a.alive = alive != 0;
          }
      }
    }
  }
  void Save() const {
    if (path_.empty()) return;
    const std::string tmp = path_ + ".tmp";
    {
      std::ofstream f(tmp);
      if (!f) return;
      for (const auto& [hal, m] : cells_)
        for (const auto& [b, c] : m) {
          if (c.closed) {
            f << hal << ' ' << b << " closed " << (c.pick.empty() ? "-" : c.pick) << '\n';
            continue;
          }
          for (const Arm& a : c.arms)
            f << hal << ' ' << b << " arm " << a.hal << ' ' << (a.alive ? 1 : 0) << ' ' << a.n << ' ' << a.sum << ' '
              << a.sum2 << ' ' << c.chunks << '\n';
        }
    }
    std::rename(tmp.c_str(), path_.c_str());
  }

  std::string path_;
  std::map<std::string, std::vector<std::pair<std::string, std::uint32_t>>> variants_;
  std::map<std::string, std::map<std::uint32_t, CellState>> cells_;
  std::uint32_t bucket_ = 2048, chunk_ = 0;
  std::map<std::string, std::uint32_t> seen_;
  std::vector<Tag> tags_;
};

}  // namespace yah::model
