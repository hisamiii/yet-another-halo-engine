// GGML dequantization follows ggml-quants.c (dequantize_row_*), block layouts from ggml-common.h.
#include "core/gguf_quant.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace yah::core {
namespace {

#include "core/iq_grids.inc"

constexpr std::int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

float Half(const std::uint8_t* p) {
  std::uint16_t h;
  std::memcpy(&h, p, 2);
  return static_cast<float>(std::bit_cast<_Float16>(h));
}
std::uint16_t ToHalf(float f) {
  return std::bit_cast<std::uint16_t>(static_cast<_Float16>(f));
}
std::uint32_t U32(const std::uint8_t* p) {
  std::uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
// The sign pattern of 8 grid values: 7 stored bits, the 8th makes the count of minus signs even (ggml's ksigns_iq2xs).
std::uint8_t Ksigns(std::uint32_t i) {
  return static_cast<std::uint8_t>(i | ((std::popcount(i) & 1U) << 7));
}
float Sign(std::uint8_t signs, int j) {
  return (signs >> j) & 1 ? -1.0F : 1.0F;
}

// Q4_K / Q5_K: 6-bit scale and min j of the 12 packed bytes.
void ScaleMinK4(int j, const std::uint8_t* q, float* d, float* m) {
  if (j < 4) {
    *d = static_cast<float>(q[j] & 63), *m = static_cast<float>(q[j + 4] & 63);
  } else {
    *d = static_cast<float>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *m = static_cast<float>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

void Q4_0(const std::uint8_t* b, float* y) {
  const float d = Half(b);
  for (int j = 0; j < 16; ++j) y[j] = d * ((b[2 + j] & 0xF) - 8), y[j + 16] = d * ((b[2 + j] >> 4) - 8);
}
void Q4_1(const std::uint8_t* b, float* y) {
  const float d = Half(b), m = Half(b + 2);
  for (int j = 0; j < 16; ++j) y[j] = d * (b[4 + j] & 0xF) + m, y[j + 16] = d * (b[4 + j] >> 4) + m;
}
void Q5_0(const std::uint8_t* b, float* y) {
  const float d = Half(b);
  const std::uint32_t qh = U32(b + 2);
  for (int j = 0; j < 16; ++j) {
    const int h0 = static_cast<int>(((qh >> j) << 4) & 0x10), h1 = static_cast<int>((qh >> (j + 12)) & 0x10);
    y[j] = d * (((b[6 + j] & 0xF) | h0) - 16), y[j + 16] = d * (((b[6 + j] >> 4) | h1) - 16);
  }
}
void Q5_1(const std::uint8_t* b, float* y) {
  const float d = Half(b), m = Half(b + 2);
  const std::uint32_t qh = U32(b + 4);
  for (int j = 0; j < 16; ++j) {
    const int h0 = static_cast<int>(((qh >> j) << 4) & 0x10), h1 = static_cast<int>((qh >> (j + 12)) & 0x10);
    y[j] = d * ((b[8 + j] & 0xF) | h0) + m, y[j + 16] = d * ((b[8 + j] >> 4) | h1) + m;
  }
}
void Q8_0(const std::uint8_t* b, float* y) {
  const float d = Half(b);
  for (int j = 0; j < 32; ++j) y[j] = d * static_cast<std::int8_t>(b[2 + j]);
}
void IQ4_NL(const std::uint8_t* b, float* y) {
  const float d = Half(b);
  for (int j = 0; j < 16; ++j) y[j] = d * kIq4nl[b[2 + j] & 0xF], y[j + 16] = d * kIq4nl[b[2 + j] >> 4];
}

// 256-element super-blocks

void Q2_K(const std::uint8_t* b, float* y) {
  const std::uint8_t* sc = b;
  const std::uint8_t* q = b + 16;
  const float d = Half(b + 80), dmin = Half(b + 82);
  int is = 0;
  for (int n = 0; n < 256; n += 128, q += 32) {
    for (int shift = 0; shift < 8; shift += 2) {
      for (int half = 0; half < 2; ++half) {
        const std::uint8_t s = sc[is++];
        const float dl = d * (s & 0xF), ml = dmin * (s >> 4);
        for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l + 16 * half] >> shift) & 3) - ml;
      }
    }
  }
}
void Q3_K(const std::uint8_t* b, float* y) {
  const std::uint8_t* hm = b;
  const std::uint8_t* q = b + 32;
  const float d = Half(b + 108);
  std::uint32_t aux[4];
  std::memcpy(aux, b + 96, 12);
  constexpr std::uint32_t kM1 = 0x03030303, kM2 = 0x0f0f0f0f;
  const std::uint32_t tmp = aux[2];
  aux[2] = ((aux[0] >> 4) & kM2) | (((tmp >> 4) & kM1) << 4);
  aux[3] = ((aux[1] >> 4) & kM2) | (((tmp >> 6) & kM1) << 4);
  aux[0] = (aux[0] & kM2) | (((tmp >> 0) & kM1) << 4);
  aux[1] = (aux[1] & kM2) | (((tmp >> 2) & kM1) << 4);
  const auto* scales = reinterpret_cast<const std::int8_t*>(aux);
  int is = 0;
  std::uint8_t m = 1;
  for (int n = 0; n < 256; n += 128, q += 32) {
    for (int shift = 0; shift < 8; shift += 2, m <<= 1) {
      for (int half = 0; half < 2; ++half) {
        const float dl = d * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          const int i = l + 16 * half;
          *y++ = dl * (((q[i] >> shift) & 3) - ((hm[i] & m) ? 0 : 4));
        }
      }
    }
  }
}
void Q4_K(const std::uint8_t* b, float* y) {
  const float d = Half(b), dmin = Half(b + 2);
  const std::uint8_t* sc = b + 4;
  const std::uint8_t* q = b + 16;
  for (int j = 0, is = 0; j < 256; j += 64, q += 32, is += 2) {
    float s1, m1, s2, m2;
    ScaleMinK4(is, sc, &s1, &m1);
    ScaleMinK4(is + 1, sc, &s2, &m2);
    for (int l = 0; l < 32; ++l) *y++ = d * s1 * (q[l] & 0xF) - dmin * m1;
    for (int l = 0; l < 32; ++l) *y++ = d * s2 * (q[l] >> 4) - dmin * m2;
  }
}
void Q5_K(const std::uint8_t* b, float* y) {
  const float d = Half(b), dmin = Half(b + 2);
  const std::uint8_t* sc = b + 4;
  const std::uint8_t* qh = b + 16;
  const std::uint8_t* ql = b + 48;
  std::uint8_t u1 = 1, u2 = 2;
  for (int j = 0, is = 0; j < 256; j += 64, ql += 32, is += 2, u1 <<= 2, u2 <<= 2) {
    float s1, m1, s2, m2;
    ScaleMinK4(is, sc, &s1, &m1);
    ScaleMinK4(is + 1, sc, &s2, &m2);
    for (int l = 0; l < 32; ++l) *y++ = d * s1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - dmin * m1;
    for (int l = 0; l < 32; ++l) *y++ = d * s2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - dmin * m2;
  }
}
void Q6_K(const std::uint8_t* b, float* y) {
  const std::uint8_t* ql = b;
  const std::uint8_t* qh = b + 128;
  const auto* sc = reinterpret_cast<const std::int8_t*>(b + 192);
  const float d = Half(b + 208);
  for (int n = 0; n < 256; n += 128, y += 128, ql += 64, qh += 32, sc += 8) {
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      y[l] = d * sc[is] * q1, y[l + 32] = d * sc[is + 2] * q2;
      y[l + 64] = d * sc[is + 4] * q3, y[l + 96] = d * sc[is + 6] * q4;
    }
  }
}
void IQ4_XS(const std::uint8_t* b, float* y) {
  const float d = Half(b);
  const std::uint16_t sh = static_cast<std::uint16_t>(b[2] | (b[3] << 8));
  const std::uint8_t* sl = b + 4;
  const std::uint8_t* qs = b + 8;
  for (int ib = 0; ib < 8; ++ib, y += 32, qs += 16) {
    const int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4);
    const float dl = d * (ls - 32);
    for (int j = 0; j < 16; ++j) y[j] = dl * kIq4nl[qs[j] & 0xF], y[j + 16] = dl * kIq4nl[qs[j] >> 4];
  }
}
void IQ2_XXS(const std::uint8_t* b, float* y) {
  const float d = Half(b);
  const std::uint8_t* qs = b + 2;
  for (int ib = 0; ib < 8; ++ib, qs += 8) {
    const std::uint32_t a1 = U32(qs + 4);
    const float db = d * (0.5F + static_cast<float>(a1 >> 28)) * 0.25F;
    for (int l = 0; l < 4; ++l, y += 8) {
      const std::uint64_t g = kIq2xxsGrid[qs[l]];
      const std::uint8_t s = Ksigns((a1 >> 7 * l) & 127);
      for (int j = 0; j < 8; ++j) y[j] = db * static_cast<float>((g >> 8 * j) & 0xFF) * Sign(s, j);
    }
  }
}
void IQ2_XS(const std::uint8_t* b, float* y) {
  const float d = Half(b);
  const std::uint8_t* qs = b + 2;
  const std::uint8_t* sc = b + 66;
  for (int ib = 0; ib < 8; ++ib) {
    const float db[2] = {d * (0.5F + (sc[ib] & 0xF)) * 0.25F, d * (0.5F + (sc[ib] >> 4)) * 0.25F};
    for (int l = 0; l < 4; ++l, y += 8) {
      const std::uint16_t q = static_cast<std::uint16_t>(qs[8 * ib + 2 * l] | (qs[8 * ib + 2 * l + 1] << 8));
      const std::uint64_t g = kIq2xsGrid[q & 511];
      const std::uint8_t s = Ksigns(q >> 9);
      for (int j = 0; j < 8; ++j) y[j] = db[l / 2] * static_cast<float>((g >> 8 * j) & 0xFF) * Sign(s, j);
    }
  }
}
void IQ3_XXS(const std::uint8_t* b, float* y) {
  const float d = Half(b);
  const std::uint8_t* qs = b + 2;
  const std::uint8_t* ss = b + 2 + 64;
  for (int ib = 0; ib < 8; ++ib, qs += 8) {
    const std::uint32_t a = U32(ss + 4 * ib);
    const float db = d * (0.5F + static_cast<float>(a >> 28)) * 0.5F;
    for (int l = 0; l < 4; ++l, y += 8) {
      const std::uint8_t s = Ksigns((a >> 7 * l) & 127);
      const std::uint32_t g1 = kIq3xxsGrid[qs[2 * l]], g2 = kIq3xxsGrid[qs[2 * l + 1]];
      for (int j = 0; j < 4; ++j) {
        y[j] = db * static_cast<float>((g1 >> 8 * j) & 0xFF) * Sign(s, j);
        y[j + 4] = db * static_cast<float>((g2 >> 8 * j) & 0xFF) * Sign(s, j + 4);
      }
    }
  }
}
void IQ3_S(const std::uint8_t* b, float* y) {
  const float d = Half(b);
  const std::uint8_t* qs = b + 2;
  const std::uint8_t* qh = b + 66;
  const std::uint8_t* signs = b + 74;
  const std::uint8_t* sc = b + 106;
  for (int ib = 0; ib < 8; ib += 2) {
    const float db[2] = {d * (1 + 2 * (sc[ib / 2] & 0xF)), d * (1 + 2 * (sc[ib / 2] >> 4))};
    for (int h = 0; h < 2; ++h, qs += 8, signs += 4) {
      for (int l = 0; l < 4; ++l, y += 8) {
        const std::uint32_t g1 = kIq3sGrid[qs[2 * l] | ((qh[h] << (8 - 2 * l)) & 256)];
        const std::uint32_t g2 = kIq3sGrid[qs[2 * l + 1] | ((qh[h] << (7 - 2 * l)) & 256)];
        for (int j = 0; j < 4; ++j) {
          y[j] = db[h] * static_cast<float>((g1 >> 8 * j) & 0xFF) * Sign(signs[l], j);
          y[j + 4] = db[h] * static_cast<float>((g2 >> 8 * j) & 0xFF) * Sign(signs[l], j + 4);
        }
      }
    }
    qh += 2;
  }
}

using Block = void (*)(const std::uint8_t*, float*);
Block BlockFn(GgmlType t) {
  switch (t) {
    case GgmlType::kQ4_0: return Q4_0;
    case GgmlType::kQ4_1: return Q4_1;
    case GgmlType::kQ5_0: return Q5_0;
    case GgmlType::kQ5_1: return Q5_1;
    case GgmlType::kQ8_0: return Q8_0;
    case GgmlType::kQ2_K: return Q2_K;
    case GgmlType::kQ3_K: return Q3_K;
    case GgmlType::kQ4_K: return Q4_K;
    case GgmlType::kQ5_K: return Q5_K;
    case GgmlType::kQ6_K: return Q6_K;
    case GgmlType::kIQ2_XXS: return IQ2_XXS;
    case GgmlType::kIQ2_XS: return IQ2_XS;
    case GgmlType::kIQ3_XXS: return IQ3_XXS;
    case GgmlType::kIQ3_S: return IQ3_S;
    case GgmlType::kIQ4_NL: return IQ4_NL;
    case GgmlType::kIQ4_XS: return IQ4_XS;
    default: return nullptr;
  }
}

// In place over 256 values: the natural-order Walsh-Hadamard transform, unnormalized.
[[gnu::target_clones("avx512f", "avx2", "default")]] void H256(float* x) {
  for (int h = 1; h < 256; h *= 2)
    for (int i = 0; i < 256; i += 2 * h)
      for (int j = i; j < i + h; ++j) {
        const float a = x[j], b = x[j + h];
        x[j] = a + b, x[j + h] = a - b;
      }
}

// Squared error of 128 values coded at scale d (codes clamp to [-8, 7], ties to even as the hardware rounds).
[[gnu::target_clones("avx512f", "avx2", "default")]] float Mq4Error(const float* w, float d) {
  const float inv = 1.0F / d;
  float e = 0;
  for (int i = 0; i < 128; ++i) {
    const float q = std::clamp(std::nearbyint(w[i] * inv), -8.0F, 7.0F);
    const float r = w[i] - q * d;
    e += r * r;
  }
  return e;
}

// hipfire's sign vectors (gen_rot.sign_vec, seeds 42 and 1042): state = state * 1103515245 + 12345 mod 2^31, +1 when
// bit 16 is set.
struct Signs {
  float s1[256], s2[256];
  Signs() {
    for (auto [seed, out] : {std::pair{42U, s1}, std::pair{1042U, s2}}) {
      std::uint32_t s = seed;
      for (int i = 0; i < 256; ++i) {
        s = (s * 1103515245U + 12345U) & 0x7FFFFFFFU;
        out[i] = (s >> 16) & 1 ? 1.0F : -1.0F;
      }
    }
  }
};
const Signs kSigns;

}  // namespace

bool CanDequantize(GgmlType t) {
  return t == GgmlType::kF32 || t == GgmlType::kF16 || t == GgmlType::kBF16 || BlockFn(t) != nullptr;
}

void Dequantize(GgmlType t, const std::uint8_t* src, float* dst, std::uint64_t n) {
  if (t == GgmlType::kF32) {
    std::memcpy(dst, src, n * 4);
  } else if (t == GgmlType::kF16) {
    for (std::uint64_t i = 0; i < n; ++i) dst[i] = Half(src + 2 * i);
  } else if (t == GgmlType::kBF16) {
    for (std::uint64_t i = 0; i < n; ++i) {
      std::uint16_t h;
      std::memcpy(&h, src + 2 * i, 2);
      dst[i] = std::bit_cast<float>(static_cast<std::uint32_t>(h) << 16);
    }
  } else {
    const Block f = BlockFn(t);
    if (f == nullptr) throw std::runtime_error(std::string("gguf: cannot read tensor type ") + TypeName(t));
    const std::uint32_t be = BlockElements(t), bb = BlockBytes(t);
    for (std::uint64_t b = 0; b < n / be; ++b) f(src + b * bb, dst + b * be);
  }
}

void QuantizeMq4Row(float* w, std::uint64_t k, std::uint8_t* dst) {
  for (std::uint64_t g = 0; g < k / 256; ++g, dst += 136) {
    float* x = w + 256 * g;
    for (int i = 0; i < 256; ++i) x[i] *= kSigns.s1[i];
    H256(x);
    for (int i = 0; i < 256; ++i) x[i] *= kSigns.s2[i] / 16.0F;
    std::memset(dst, 0, 136);
    for (int h = 0; h < 2; ++h) {
      const float* v = x + 128 * h;
      float amax = 0;
      for (int i = 0; i < 128; ++i) amax = std::max(amax, std::fabs(v[i]));
      float d = 0;
      if (amax > 0) {
        float best = INFINITY;
        for (int j = 0; j < 16; ++j) {
          // the stored scale is f16: search over what the kernels will read
          const float c = static_cast<float>(static_cast<_Float16>(amax / 7.0F * (0.5F + j / 30.0F)));
          if (c <= 0) continue;
          const float e = Mq4Error(v, c);
          if (e < best) best = e, d = c;
        }
      }
      const std::uint16_t hd = ToHalf(d), hm = ToHalf(-8.0F * d);
      std::memcpy(dst + 4 * h, &hd, 2);
      std::memcpy(dst + 4 * h + 2, &hm, 2);
      const float inv = d > 0 ? 1.0F / d : 0.0F;
      for (int i = 0; i < 128; ++i) {
        const int q = static_cast<int>(std::clamp(std::nearbyint(v[i] * inv), -8.0F, 7.0F));
        dst[8 + 64 * h + i / 2] |= static_cast<std::uint8_t>((q & 15) << (4 * (i & 1)));
      }
    }
  }
}

void QuantizeQ8_0(const float* x, std::uint64_t n, std::uint8_t* dst) {
  for (std::uint64_t b = 0; b < n / 32; ++b, x += 32, dst += 34) {
    float amax = 0;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float d = amax / 127.0F, inv = d > 0 ? 1.0F / d : 0.0F;
    const std::uint16_t hd = ToHalf(d);
    std::memcpy(dst, &hd, 2);
    for (int i = 0; i < 32; ++i) dst[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::lround(x[i] * inv)));
  }
}

}  // namespace yah::core
