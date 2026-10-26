// The NPU FFN block's down weights, built from the loaded model: every layer's ffn_down rows over the NPU's F features
// out of the rotated basis with the awq folded in (w = s1 / awq H256(s2 / 16 w), the signs of rot_signs.bin), then
// re-quantized to the IQ4_XS rows the down image decodes (llama.cpp block_iq4_xs: d, scales_h, scales_l, qs).
// Float64 throughout, in numpy's operation order: the first pack came from a numpy tool and this one equals it.
#ifndef YAH_MODEL_NPU_DOWN_PACK_HPP_
#define YAH_MODEL_NPU_DOWN_PACK_HPP_

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

namespace yah::model {

struct NpuDownPack {
  static constexpr std::size_t kBlock = 136;  // bytes per 256 values, MQ4G256 and IQ4_XS alike
  static constexpr double kIq[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

  // rows: n MQ4 rows (signed nibbles, as the engine holds them) row_bytes apart, from their F features' first block;
  // awq: those features' awq; signs: rot_signs.bin (s1, then s2 / 16). out: n rows of F / 256 IQ4_XS blocks.
  static void Rows(const std::uint8_t* rows, std::size_t row_bytes, std::size_t n, std::size_t F, const float* awq,
                   const float* signs, std::uint8_t* out) {
    const unsigned threads = std::max(1U, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t)
      pool.emplace_back([=] {
        for (std::size_t r = t; r < n; r += threads)
          for (std::size_t b = 0; b < F / 256; ++b)
            Block(rows + r * row_bytes + b * kBlock, awq + b * 256, signs, out + (r * (F / 256) + b) * kBlock);
      });
    for (auto& th : pool) th.join();
  }

 private:
  // The IQ grid index nearest x (ties up): the midpoints at or below x.
  static int Nearest(double x) {
    int i = 0;
    for (int k = 0; k < 15; ++k) i += (kIq[k] + kIq[k + 1]) / 2 <= x;
    return i;
  }
  // kIq[Nearest(x)] without the lookup (the steps are integers: exact), so the candidate loop vectorizes
  static double Value(double x) {
    double v = kIq[0];
    for (int k = 0; k < 15; ++k) v += (kIq[k] + kIq[k + 1]) / 2 <= x ? kIq[k + 1] - kIq[k] : 0;
    return v;
  }
  // numpy's pairwise sum of 32 values: 8 strided accumulators, combined as a tree
  static double Sum32(const double* a) {
    double r[8];
    for (int k = 0; k < 8; ++k) r[k] = a[k];
    for (int i = 8; i < 32; i += 8)
      for (int k = 0; k < 8; ++k) r[k] += a[i + k];
    return ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
  }
  static double F16(const std::uint8_t* p) {
    return static_cast<double>(std::bit_cast<_Float16>(static_cast<std::uint16_t>(p[0] | p[1] << 8)));
  }

  static void Block(const std::uint8_t* mq4, const float* awq, const float* signs, std::uint8_t* out) {
    double w[256];
    const double d2[2] = {F16(mq4), F16(mq4 + 4)};  // the scales of the two halves
    for (int e = 0; e < 256; ++e) {
      const int nib = (mq4[8 + e / 2] >> (4 * (e & 1))) & 15;
      w[e] = static_cast<double>(signs[256 + e]) * (d2[e / 128] * (nib >= 8 ? nib - 16 : nib));
    }
    for (int h = 1; h < 256; h *= 2)  // the natural-order Walsh-Hadamard transform, unnormalized
      for (int i = 0; i < 256; i += 2 * h)
        for (int j = i; j < i + h; ++j) {
          const double a = w[j], b = w[j + h];
          w[j] = a + b, w[j + h] = a - b;
        }
    for (int e = 0; e < 256; ++e) w[e] = static_cast<double>(signs[e]) / static_cast<double>(awq[e]) * w[e];

    // per 32 values the scale that minimizes the squared error over 34 candidates (signed: the grid is asymmetric)
    double sub[8];
    for (int j = 0; j < 8; ++j) {
      const double* x = w + 32 * j;
      double am = 0, err = std::numeric_limits<double>::infinity(), e2[32];
      for (int i = 0; i < 32; ++i) am = std::max(am, std::abs(x[i]));
      sub[j] = 0;
      for (int c = 0; c < 34; ++c) {
        const int i16 = c % 17;  // numpy's linspace(0.8, 1.2, 17), then its negation
        const double f = (i16 == 16 ? 1.2 : i16 * ((1.2 - 0.8) / 16) + 0.8) * (c < 17 ? 1 : -1);
        const double sc = am / 127 * f, div = sc == 0 ? 1 : sc;
        for (int i = 0; i < 32; ++i) {
          const double v = Value(x[i] / div) * sc - x[i];
          e2[i] = v * v;
        }
        if (const double e = Sum32(e2); e < err) sub[j] = sc, err = e;
      }
    }
    double dmax = 0;
    for (const double s : sub) dmax = std::max(dmax, std::abs(s));
    const auto d = static_cast<_Float16>(dmax / 31);
    const double dd = static_cast<double>(d);
    const std::uint16_t dbits = std::bit_cast<std::uint16_t>(d);
    std::uint16_t sh = 0;
    std::fill(out, out + kBlock, 0);
    out[0] = dbits & 255, out[1] = dbits >> 8;
    for (int j = 0; j < 8; ++j) {
      const double ls = std::clamp(std::nearbyint(sub[j] / (dd == 0 ? 1 : dd)), -32.0, 31.0);
      const double s = dd * ls, div = s == 0 ? 1 : s;
      const auto u = static_cast<std::uint16_t>(static_cast<int>(ls) + 32);  // 6 bits per sub-block
      sh |= ((u >> 4) & 3) << (2 * j);
      out[4 + j / 2] |= (u & 15) << (4 * (j % 2));
      for (int i = 0; i < 16; ++i)
        out[8 + 16 * j + i] =
            static_cast<std::uint8_t>(Nearest(w[32 * j + i] / div) | Nearest(w[32 * j + 16 + i] / div) << 4);
    }
    out[2] = sh & 255, out[3] = sh >> 8;
  }
};

}  // namespace yah::model

#endif  // YAH_MODEL_NPU_DOWN_PACK_HPP_
