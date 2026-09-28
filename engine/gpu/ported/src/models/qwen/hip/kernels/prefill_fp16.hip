#if defined(ENGINE_ENABLE_HIP)
#include <stdexcept>

#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
namespace gufo::hip {
namespace {
__device__ __forceinline__ __half
RoundActivation(float value) {  // Keep the FP32 rounding boundary before the
                                // FP16 conversion. Without this
  // register dependency, combining the producer with conversion changes rare
  // halfway values relative to the standalone FP32 producer. No instruction
  // or scratch memory is emitted by this compiler barrier.
  asm volatile("" : "+v"(value));
  return __float2half_rn(value);
}
using v16bf = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f WmmaHalf(v16bf a, v16bf b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// Permute 16-byte halves and the LDS row to avoid repeated bank conflicts.
__device__ __forceinline__ v16bf LoadSwizzled(const short* p) {
  union {
    v16bf f;
    uint4 q[2];
  } bits;
  const unsigned row = threadIdx.x & 15, shift = (row ^ (row >> 2)) & 1;
  bits.q[0] = reinterpret_cast<const uint4*>(p)[shift];
  bits.q[1] = reinterpret_cast<const uint4*>(p)[shift ^ 1];
  return bits.f;
}
__device__ __forceinline__ void StoreSwizzled(short* p, uint4 lo, uint4 hi,
                                              unsigned row) {
  const unsigned shift = (row ^ (row >> 2)) & 1;
  reinterpret_cast<uint4*>(p)[shift] = lo;
  reinterpret_cast<uint4*>(p)[shift ^ 1] = hi;
}

// Ablation 1048576: one activation fragment straight out of the pre-blocked
// global buffer, layout [k/16][batch][16 halves]. Lane `sl` holds the sixteen K
// values of token (((wt*WTS)+j)*16) + (sl ^ (ks&3)), which is 32 contiguous
// bytes, so the warp's sixteen distinct lanes span 512 contiguous bytes and the
// upper sixteen lanes broadcast. No LDS staging array is involved.
// One 64-bit address computation per K step gives a lane-local base; advancing j
// is then a 32-bit add, because one j step is sixteen tokens, i.e. 16*16 halves.
// Recomputing the full 64-bit address per load cost more than the load itself.
// Ablation 8388608: the same decode, with the multiply and subtract folded into a
// single fp16 fused multiply-add instead of an fp32 mul/sub pair plus two
// converts. It halves the decode's VALU count but rounds the product in half
// precision, so it is a measurement of what the fp32 chain costs, not a default.
template<int Ablate>
__device__ __forceinline__ void DecodeToHalves(const gufo::hip::QuantSub16& q,
                                               v16bf& out) {
  if constexpr ((Ablate & 8388608) != 0) {
    const _Float16 hs = static_cast<_Float16>(q.scale);
    const _Float16 ho = static_cast<_Float16>(q.offset);
#pragma unroll
    for (int j = 0; j < 16; ++j)
      out[j] = static_cast<_Float16>(
          __builtin_fmaf(static_cast<float>(hs),
                         static_cast<float>(static_cast<_Float16>(q.q[j])),
                         -static_cast<float>(ho)));
  } else {
#pragma unroll
    for (int j = 0; j < 16; ++j)
      out[j] = static_cast<_Float16>(__fsub_rn(
          __fmul_rn(q.scale, static_cast<float>(q.q[j])), q.offset));
  }
}

__device__ __forceinline__ v16bf LoadBlockedBAt(const short* src) {
  union {
    v16bf f;
    uint4 q[2];
  } bits;
  // Canonical order, deliberately. LoadSwizzled's lane-dependent half swap is
  // cancelled by the matching StoreSwizzled swap: the store put lo at
  // 16*shift and hi at 16*(shift^1), so reading back at LoadSwizzled's own shift
  // returns lo then hi. Replicating only the load half of that pair reverses the
  // two eight-half groups for every lane with shift==1, which is exactly half of
  // them -- the observed bit_mismatch of half the outputs.
  bits.q[0] = *reinterpret_cast<const uint4*>(src);
  bits.q[1] = *reinterpret_cast<const uint4*>(src + 8);
  return bits.f;
}

// Same 32 bytes and the same swizzle as StoreSwizzled, reached with a different
// number of DS store instructions: 2 x ds_store_b128, 4 x ds_store_b64 or
// 8 x ds_store_b32. Ablation 131072 selects b64 and 262144 selects b32, so the
// store byte volume is held constant while the DS instruction count doubles and
// then quadruples -- which separates "the store costs what it moves" from "the
// store costs what it issues".
template<int Width>
__device__ __forceinline__ void StoreSwizzledWidth(short* p, uint4 lo, uint4 hi,
                                                   unsigned row) {
  const unsigned shift = (row ^ (row >> 2)) & 1;
  if constexpr (Width == 128) {
    reinterpret_cast<uint4*>(p)[shift] = lo;
    reinterpret_cast<uint4*>(p)[shift ^ 1] = hi;
  } else if constexpr (Width == 64) {
    const uint2 lo2[2] = {make_uint2(lo.x, lo.y), make_uint2(lo.z, lo.w)};
    const uint2 hi2[2] = {make_uint2(hi.x, hi.y), make_uint2(hi.z, hi.w)};
    uint2* q = reinterpret_cast<uint2*>(p);
    q[(2 * shift) + 0] = lo2[0];
    q[(2 * shift) + 1] = lo2[1];
    q[(2 * (shift ^ 1)) + 0] = hi2[0];
    q[(2 * (shift ^ 1)) + 1] = hi2[1];
  } else {
    const unsigned lo4[4] = {lo.x, lo.y, lo.z, lo.w};
    const unsigned hi4[4] = {hi.x, hi.y, hi.z, hi.w};
    unsigned* q = reinterpret_cast<unsigned*>(p);
#pragma unroll
    for (int i = 0; i < 4; ++i) q[(4 * shift) + i] = lo4[i];
#pragma unroll
    for (int i = 0; i < 4; ++i) q[(4 * (shift ^ 1)) + i] = hi4[i];
  }
}

template<int Ablate>
__device__ __forceinline__ void StoreCommit(short* p, uint4 lo, uint4 hi,
                                            unsigned row) {
  // Ablation 524288: both staging regions and both store streams stay, but each
  // writes only its low 16 bytes -> 32 B/thread/stage with the same regions
  // touched. Together with bit 32 (one region, 32 B) this separates "the store
  // costs what it moves" from "the store costs because both arrays are written".
  if constexpr ((Ablate & 524288) != 0) {
    const unsigned shift = (row ^ (row >> 2)) & 1;
    reinterpret_cast<uint4*>(p)[shift] = lo;
  } else if constexpr ((Ablate & 262144) != 0)
    StoreSwizzledWidth<32>(p, lo, hi, row);
  else if constexpr ((Ablate & 131072) != 0)
    StoreSwizzledWidth<64>(p, lo, hi, row);
  else
    StoreSwizzled(p, lo, hi, row);
}

struct HalfWeights {
  uint4 low, high;
};

struct RawIqWeights {
  unsigned data[4], shift;
  float scale;
};

template<gufo::core::GgmlType Type>
__device__ __forceinline__ RawIqWeights
LoadIqRaw(const void* row, unsigned sub, const uint2* header = nullptr) {
  static_assert(Type == gufo::core::GgmlType::kIQ4_XS ||
                Type == gufo::core::GgmlType::kIQ4_NL);
  RawIqWeights weights;
  weights.shift = (sub & 1) * 4;
  if constexpr (Type == gufo::core::GgmlType::kIQ4_XS) {
    const auto& block = static_cast<const IQ4XSBlock*>(row)[sub / 16];
    const unsigned sb = (sub / 2) & 7;
    LoadQuantWords16(block.qs + sb * 16, weights.data);
    const unsigned low =
        header ? header->y >> (8 * (sb / 2)) : block.scales_l[sb / 2];
    const unsigned high = header ? header->x >> 16 : block.scales_h;
    const __half d = header ? __ushort_as_half(header->x & 65535) : block.d;
    const int code = static_cast<int>(((low >> (4 * (sb & 1))) & 15) |
                                      (((high >> (2 * sb)) & 3) << 4)) -
                     32;
    weights.scale = __half2float(d) * static_cast<float>(code);
  } else {
    const auto& block = static_cast<const IQ4NLBlock*>(row)[sub / 2];
    LoadQuantWords16(block.qs, weights.data);
    weights.scale = __half2float(block.d);
  }
  return weights;
}

// All 16 IQ4 codebook entries are integers represented exactly in FP16.
// Look up their half bits directly, avoiding signed-byte extraction and
// conversion for each weight. Scaling still uses the original FP32 product.
__device__ __forceinline__ HalfWeights
DecodeIqRaw(const RawIqWeights& weights) {
  union {
    v16bf f;
    uint4 u[2];
  } result;
#pragma unroll
  for (unsigned i = 0; i < 4; ++i) {
    const unsigned n = (weights.data[i] >> weights.shift) & 0x0f0f0f0fU;
    const unsigned index = n & 0x07070707U;
    const unsigned select = ((n & 0x08080808U) >> 3) * 255;
    // Low and high bytes of FP16 {-127, -104, -83, -65, -49, -35,
    // -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}, four entries per word.
    const unsigned lo0 = __builtin_amdgcn_perm(0x00806020U, 0x103080f0U, index);
    const unsigned lo1 = __builtin_amdgcn_perm(0x109050a0U, 0xc0408000U, index);
    const unsigned hi0 = __builtin_amdgcn_perm(0xc9cdd0d2U, 0xd4d5d6d7U, index);
    const unsigned hi1 = __builtin_amdgcn_perm(0x57555452U, 0x504e4a3cU, index);
    const unsigned lo = (lo0 & ~select) | (lo1 & select);
    const unsigned hi = (hi0 & ~select) | (hi1 & select);
    const unsigned c0 = __builtin_amdgcn_perm(hi, lo, 0x05010400U);
    const unsigned c1 = __builtin_amdgcn_perm(hi, lo, 0x07030602U);
    result.f[4 * i] = static_cast<_Float16>(
        __fmul_rn(weights.scale, __half2float(__ushort_as_half(c0 & 65535))));
    result.f[4 * i + 1] = static_cast<_Float16>(
        __fmul_rn(weights.scale, __half2float(__ushort_as_half(c0 >> 16))));
    result.f[4 * i + 2] = static_cast<_Float16>(
        __fmul_rn(weights.scale, __half2float(__ushort_as_half(c1 & 65535))));
    result.f[4 * i + 3] = static_cast<_Float16>(
        __fmul_rn(weights.scale, __half2float(__ushort_as_half(c1 >> 16))));
  }
  return {result.u[0], result.u[1]};
}

template<gufo::core::GgmlType Type>
__device__ __forceinline__ HalfWeights LoadIqHalf(const void* row,
                                                  unsigned sub) {
  return DecodeIqRaw(LoadIqRaw<Type>(row, sub));
}

template<gufo::core::GgmlType Type>
struct RawWeights {
  static_assert(Type == gufo::core::GgmlType::kQ4_K ||
                Type == gufo::core::GgmlType::kQ5_K ||
                Type == gufo::core::GgmlType::kQ6_K ||
                Type == gufo::core::GgmlType::kQ8_0);
  std::uint32_t low[4], high[Type == gufo::core::GgmlType::kQ5_K ||
                                     Type == gufo::core::GgmlType::kQ6_K
                                 ? 4
                                 : 1];
  __half d, dmin;
  unsigned sc0, sc1, sc2, sub;
  int signed_scale;
};
template<gufo::core::GgmlType Type>
__device__ __forceinline__ RawWeights<Type> LoadRaw(
    const void* row, unsigned sub, const uint4* header = nullptr) {
  using namespace gufo::hip;
  using gufo::core::GgmlType;
  RawWeights<Type> r{};
  r.sub = sub;
  const unsigned sb = (sub / 2) % 8, lane = (sub % 2) * 16;
  if constexpr (Type == GgmlType::kQ8_0) {
    const auto& b = static_cast<const Q8_0Block*>(row)[sub / 2];
    r.d = b.d;
    __builtin_memcpy(r.low, b.qs + lane, 16);
  } else if constexpr (Type == GgmlType::kQ4_K || Type == GgmlType::kQ5_K) {
    using Block =
        std::conditional_t<Type == GgmlType::kQ5_K, Q5KBlock, Q4KBlock>;
    const auto& b = static_cast<const Block*>(row)[sub / 16];
    if (header) {
      r.d = __ushort_as_half(header->x & 65535);
      r.dmin = __ushort_as_half(header->x >> 16);
      const unsigned shift = (sb & 3) * 8;
      r.sc0 = (header->y >> shift) & 255;
      r.sc1 = (header->z >> shift) & 255;
      r.sc2 = (header->w >> shift) & 255;
    } else {
      r.d = b.d;
      r.dmin = b.dmin;
      r.sc0 = b.scales[sb & 3];
      r.sc1 = b.scales[(sb & 3) + 4];
      r.sc2 = b.scales[(sb & 3) + 8];
    }
    LoadQuantWords16(b.qs + (sb / 2) * 32 + lane, r.low);
    if constexpr (Type == GgmlType::kQ5_K)
      LoadQuantWords16(b.qh + lane, r.high);
  } else if constexpr (Type == GgmlType::kQ6_K) {
    const auto& b = static_cast<const Q6KBlock*>(row)[sub / 16];
    unsigned half = sb / 4, segment = sb % 4;
    r.d = b.d;
    r.signed_scale = b.scales[half * 8 + segment * 2 + (sub % 2)];
    LoadQuantWords16(b.ql + half * 64 + (segment % 2) * 32 + lane, r.low);
    LoadQuantWords16(b.qh + half * 32 + lane, r.high);
  }
  return r;
}
template<gufo::core::GgmlType Type>
__device__ __forceinline__ gufo::hip::QuantSub16 DecodeRaw(
    const RawWeights<Type>& r) {
  using namespace gufo::hip;
  using gufo::core::GgmlType;
  QuantSub16 q;
  q.offset = 0;
  unsigned sb = (r.sub / 2) % 8;
  if constexpr (Type == GgmlType::kQ8_0) {
    __builtin_memcpy(q.w, r.low, 16);
    q.scale = __half2float(r.d);
  } else if constexpr (Type == GgmlType::kQ4_K || Type == GgmlType::kQ5_K) {
    unsigned mask = 0u - (sb >> 2), lo = r.sc0 & 63,
             up = (r.sc2 & 15) | ((r.sc0 >> 6) << 4), lm = r.sc1 & 63,
             um = (r.sc2 >> 4) | ((r.sc1 >> 6) << 4);
    q.scale = __half2float(r.d) * static_cast<float>(lo ^ ((lo ^ up) & mask));
    q.offset =
        __half2float(r.dmin) * static_cast<float>(lm ^ ((lm ^ um) & mask));
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
      q.w[i] = (r.low[i] >> (4 * (sb & 1))) & 0x0f0f0f0f;
      if constexpr (Type == GgmlType::kQ5_K)
        q.w[i] |= ((r.high[i] >> sb) & 0x01010101) << 4;
    }
  } else if constexpr (Type == GgmlType::kQ6_K) {
    q.scale = __half2float(r.d) * static_cast<float>(r.signed_scale);
    unsigned seg = sb % 4;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
      q.w[i] = SubBytes(((r.low[i] >> (seg >= 2 ? 4 : 0)) & 0x0f0f0f0f) |
                            (((r.high[i] >> (2 * seg)) & 0x03030303) << 4),
                        0x20202020);
  }
  return q;
}

#if defined(GUFO_FP16_PROTO_CLOCK)
// Phase accumulators written by block (0,0) thread 0: commit, barrier,
// fetch, K loop -- in GPU clock cycles, summed over every stage. Slots 4 and 5
// hold that block's start and end timestamps.
__device__ unsigned long long g_proto_clock[8];
// The barrier keeps two reads from being commoned up or reordered; without it
// the compiler is free to fold them and the deltas come back wrapped.
__device__ __forceinline__ unsigned long long ProtoClock() {
  unsigned long long value = clock64();
  asm volatile("" : "+v"(value));
  return value;
}
#endif

enum class HalfEpilogue { kStore, kSwiGLU, kGateUp, kResidual };

// Fully scaled FP16 weights and activations; FP32 accumulation in K16 order.
template<int BM, int BN, int WM, int WN, gufo::core::GgmlType Type,
         HalfEpilogue Epilogue, bool Complete = false,
         gufo::core::GgmlType UpType = Type, int BKDepth = 4, int Ablate = 0,
         int GShift = -1, bool DblBuf = false>
__launch_bounds__(WM * WN * 32) __global__
    void HalfPrefillGemmKernel(const void* __restrict__ w,
                               const short* __restrict__ x,
                               float* __restrict__ y, int batch, int m, int k,
                               const float* __restrict__ gate,
                               const void* __restrict__ up = nullptr) {
  constexpr bool Paired = Epilogue == HalfEpilogue::kGateUp;
  constexpr bool Mixed = Paired && Type != UpType;
  constexpr bool UpRaw = UpType == gufo::core::GgmlType::kQ4_K ||
                         UpType == gufo::core::GgmlType::kQ5_K;
  constexpr int BK = BKDepth, WaveSize = 32;
  // BK > 4 only fits when the activation tile is not staged: at BM=BN=256 the
  // weight tile alone is BK * RS * 16 * 16 halves, so BK=8 is the whole 64 KiB
  // budget and the unused s_b is elided (ablation 1048576).
  static_assert(BK >= 1 && BK <= 8, "BKDepth out of range");
  static_assert((Ablate & 1048576) != 0 || BK <= 4,
                "BK > 4 needs the activation tile to come from global");
  // A 256-weight Q5 block spans four K64 iterations. Keep its header in
  // registers for the Q5 and Q4/Q5 kernels that benefit from this reuse.
  constexpr bool CacheQ5Header = BM == 256 && BN == 256 &&
                                 (Type == gufo::core::GgmlType::kQ5_K ||
                                  (Type == gufo::core::GgmlType::kQ4_K &&
                                   UpType == gufo::core::GgmlType::kQ5_K));
  // These IQ4_XS projections benefit from keeping packed weights until
  // LDS commit and reusing the block header. Q3/IQ4_NL mixed pairs retain
  // immediate decoding.
  constexpr bool CacheIqHeader =
      BM == 256 && BN == 256 &&
      ((Type == gufo::core::GgmlType::kIQ4_XS &&
        (!Mixed || UpType == gufo::core::GgmlType::kQ4_K ||
         UpType == gufo::core::GgmlType::kQ5_K)) ||
       (Type == gufo::core::GgmlType::kQ4_K &&
        UpType == gufo::core::GgmlType::kIQ4_XS));
  // The mixed Q6 gate is qualified with the fetch-stage decoder; deferring
  // its decode changes dot-product rounding in this kernel specialization.
  constexpr int DecodeStage =
      (Type == gufo::core::GgmlType::kQ4_K ||
       Type == gufo::core::GgmlType::kQ5_K ||
       (Type == gufo::core::GgmlType::kQ6_K && !Mixed) ||
       Type == gufo::core::GgmlType::kQ8_0)
          ? 2
          : 0;
  constexpr int kThreads = WM * WN * WaveSize;
  constexpr int RS = BM / 16;
  constexpr int TS = BN / 16;
  constexpr int WRS = RS / WM;
  constexpr int WTS = TS / WN;
  constexpr int KS = 16;
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);

  constexpr int PFA = (BM * BK + kThreads - 1) / kThreads;
  constexpr int PFB = (BN * BK + kThreads - 1) / kThreads;

  // DblBuf holds two LDS stages so the commit for stage i+1 can be issued
  // before stage i's K loop, letting its LDS writes drain while the matrix pipe
  // runs; the trailing barrier then only orders the compute instead of waiting
  // on a commit nothing overlaps. At BK=4 two stages are 128 KiB and do not
  // fit; at BK=2 they are exactly the 64 KiB the single-buffered BK=4 uses.
  constexpr int kBufs = DblBuf ? 2 : 1;
  __shared__ short s_a[kBufs][BK][RS][16][KS];
  __shared__ short s_b[kBufs][BK][TS][16][KS];

  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / WaveSize;
  const int lane = tid % WaveSize;
  const int sl = lane & 15;
  const int wr = wave / WN;
  const int wt = wave % WN;

#if defined(GUFO_FP16_PROTO_CLOCK)
  const unsigned long long proto_block_start = ProtoClock();
#endif
  unsigned row_tile = blockIdx.y, token_tile = blockIdx.x;
  {
    // Wider row groups improve input reuse for these large projections.
    // Long Q5 reductions and mixed gate/up decoders retain smaller groups.
    unsigned group_shift = 3;
    if constexpr (BM == 256 && BN == 256 && !Mixed) {
      if constexpr (Type == gufo::core::GgmlType::kIQ4_XS)
        group_shift = 5;
      else if constexpr (Type == gufo::core::GgmlType::kQ5_K)
        group_shift = k <= 8192 ? 5 : 3;
    }
    if constexpr (GShift >= 0)
      group_shift = GShift;
    const unsigned group = 1U << group_shift;
    unsigned first = (blockIdx.y >> group_shift) << group_shift,
             within = (blockIdx.y & (group - 1)) * gridDim.x + blockIdx.x,
             rows = min(group, gridDim.y - first);
    row_tile = first + within % rows;
    token_tile = within / rows;
  }
  const int r_block = row_tile * (Paired ? BM / 2 : BM),
            t_block = token_tile * BN;
  const int ksteps = k / 16;
  // Ablation 1048576 (GlobalB): the activation operand is read straight from a
  // pre-blocked global buffer instead of through the second LDS staging array,
  // so only the weight tile is staged. The buffer is passed through `up`, which
  // is unused on the kStore path; it holds k/16 blocks of [batch][16 halves] in
  // that order, which is exactly the LDS layout the commit builds, so a lane's
  // fragment is one contiguous 32-byte load.
  const short* b_blocked = (Ablate & 1048576) != 0
                               ? static_cast<const short*>(up)
                               : nullptr;
  // GlobalB and DblBuf combine to the configuration that never fit before: with
  // only the weight tile staged, BK=4 double buffering is 2 * 32 KiB = 64 KiB.
  //
  // Ablation 16777216 (Fp16W): the weights arrive already dequantised as fp16,
  // row-major [m][k], also passed through `up`. The fetch loads 32 bytes per
  // (row, K16) and the commit stores them to LDS unchanged, so the decode leaves
  // the kernel entirely. Restricted to the plain kStore path.
  static_assert((Ablate & 16777216) == 0 || (!Paired && !Mixed),
                "Fp16W covers the plain kStore path");
  static_assert(((Ablate & 16777216) == 0) || ((Ablate & 1048576) == 0),
                "Fp16W and GlobalB both use the up argument");
  const short* w16 = (Ablate & 16777216) != 0 ? static_cast<const short*>(up)
                                              : nullptr;
  const short* w16_row[PFA];
  // Ablation 33554432 (GlobalA): the weight operand is read straight from a
  // pre-blocked fp16 buffer, layout [k/16][m][16], so the weight tile is never
  // staged in LDS and the allocation drops to the activation tile alone -- 32 KiB
  // at BK=4, which is what lets a second block be resident. Also rides in on
  // `up`. This is the mirror of GlobalB: the operand with the *lower* reuse
  // factor (weights are shared by WN warps, activations by WM) is the one to
  // read from global.
  static_assert(((Ablate & 33554432) == 0) || ((Ablate & 1048576) == 0),
                "GlobalA and GlobalB both use the up argument");
  static_assert(((Ablate & 33554432) == 0) || ((Ablate & 16777216) == 0),
                "GlobalA and Fp16W both use the up argument");
  static_assert(((Ablate & 33554432) == 0) || !DblBuf,
                "GlobalA has no weight tile to double buffer");
  const short* w16a = (Ablate & 33554432) != 0 ? static_cast<const short*>(up)
                                               : nullptr;

  v8f acc[WRS][WTS] = {};

  // Hoisted 64-bit row bases, clamped so an out-of-range row reads row m-1 and
  // is masked out at the store instead of branching in the K loop.
  const std::uint8_t* w_row[PFA];
  const std::size_t row_bytes = gufo::hip::QuantRowBytes(Type, k);
  const std::size_t up_row_bytes = gufo::hip::QuantRowBytes(UpType, k);
#pragma unroll
  for (int p = 0; p < PFA; ++p) {
    const int physical_row = ((p * kThreads) + tid) / BK;
    const int r =
        r_block +
        (Paired ? (physical_row / 32) * 16 + physical_row % 16 : physical_row);
    const void* source = Paired && physical_row % 32 >= 16 ? up : w;
    const std::size_t row_sel =
        static_cast<std::size_t>((Complete || r < m) ? r : (m - 1));
    w_row[p] = static_cast<const std::uint8_t*>(source) +
               (row_sel *
                (Mixed && physical_row % 32 >= 16 ? up_row_bytes : row_bytes));
    w16_row[p] = w16 + (row_sel * k);
  }
  const short* x_row[PFB];
#pragma unroll
  for (int p = 0; p < PFB; ++p) {
    const int t = t_block + (((p * kThreads) + tid) / BK);
    x_row[p] =
        x +
        (static_cast<std::size_t>((Complete || t < batch) ? t : (batch - 1)) *
         k);
  }

  uint4 ra[PFA][2], rb[PFB][2], headers[PFA];
  uint2 iq_headers[PFA];
  RawIqWeights raw_iq[PFA];
  struct NoRaw {};
  std::conditional_t<DecodeStage == 2, RawWeights<Type>, NoRaw> raw[PFA];
  std::conditional_t<Mixed && UpRaw, RawWeights<UpType>, NoRaw> raw_up[PFA];
  const auto fetch_iq = [&](int p, int kk) {
    if ((kk & 12) == 0)
      __builtin_memcpy(
          &iq_headers[p],
          reinterpret_cast<const IQ4XSBlock*>(w_row[p]) + unsigned(kk) / 16, 8);
    raw_iq[p] = LoadIqRaw<gufo::core::GgmlType::kIQ4_XS>(w_row[p], unsigned(kk),
                                                         &iq_headers[p]);
  };
  const auto fetch_up = [&](int p, int kk) {
    if constexpr (Mixed) {
      if constexpr (UpRaw) {
        if constexpr (CacheQ5Header && UpType == gufo::core::GgmlType::kQ5_K) {
          if ((kk & 12) == 0)
            __builtin_memcpy(
                &headers[p],
                reinterpret_cast<const Q5KBlock*>(w_row[p]) + unsigned(kk) / 16,
                16);
          raw_up[p] = LoadRaw<UpType>(w_row[p], kk, &headers[p]);
        } else {
          raw_up[p] = LoadRaw<UpType>(w_row[p], kk);
        }
      } else if constexpr (CacheIqHeader &&
                           UpType == gufo::core::GgmlType::kIQ4_XS) {
        fetch_iq(p, kk);
      } else if constexpr (UpType == gufo::core::GgmlType::kIQ4_XS ||
                           UpType == gufo::core::GgmlType::kIQ4_NL) {
        const auto packed = LoadIqHalf<UpType>(w_row[p], unsigned(kk));
        ra[p][0] = packed.low;
        ra[p][1] = packed.high;
      } else {
        gufo::hip::QuantSub16 q;
        gufo::hip::DecodeQuantSub16<true>(UpType, w_row[p], kk, q);
        union {
          v16bf f;
          uint4 u[2];
        } packed;
#pragma unroll
        for (int j = 0; j < 16; ++j)
          packed.f[j] = static_cast<_Float16>(__fsub_rn(
              __fmul_rn(q.scale, static_cast<float>(q.q[j])), q.offset));
        ra[p][0] = packed.u[0];
        ra[p][1] = packed.u[1];
      }
    }
  };
  const auto fetch = [&](int ks0) {
#pragma unroll
    for (int p = 0; p < PFA; ++p) {
      const int idx = p * kThreads + tid, kk = ks0 + idx % BK;
      if ((Ablate & 64) == 0 && (Ablate & 33554432) == 0 && idx < BM * BK &&
          (Complete || kk < ksteps)) {
        if constexpr ((Ablate & 16777216) != 0) {
          const short* src = w16_row[p] + (static_cast<std::size_t>(kk) * 16);
          ra[p][0] = *reinterpret_cast<const uint4*>(src);
          ra[p][1] = *reinterpret_cast<const uint4*>(src + 8);
        } else if (Mixed && (idx / BK) % 32 >= 16)
          fetch_up(p, kk);
        else if constexpr (CacheQ5Header &&
                           Type == gufo::core::GgmlType::kQ5_K) {
          if ((ks0 & 15) == 0)
            __builtin_memcpy(
                &headers[p],
                reinterpret_cast<const Q5KBlock*>(w_row[p]) + unsigned(kk) / 16,
                16);
          raw[p] = LoadRaw<Type>(w_row[p], kk, &headers[p]);
        } else if constexpr (DecodeStage == 2)
          raw[p] = LoadRaw<Type>(w_row[p], kk);
        else if constexpr (CacheIqHeader &&
                           Type == gufo::core::GgmlType::kIQ4_XS) {
          fetch_iq(p, kk);
        } else if constexpr (Type == gufo::core::GgmlType::kIQ4_XS ||
                             Type == gufo::core::GgmlType::kIQ4_NL) {
          const auto packed = LoadIqHalf<Type>(w_row[p], unsigned(kk));
          ra[p][0] = packed.low;
          ra[p][1] = packed.high;
        } else {
          gufo::hip::QuantSub16 q;
          gufo::hip::DecodeQuantSub16<true>(Type, w_row[p], kk, q);
          union {
            v16bf f;
            uint4 u[2];
          } packed;
#pragma unroll
          for (int j = 0; j < 16; ++j)
            packed.f[j] = static_cast<_Float16>(__fsub_rn(
                __fmul_rn(q.scale, static_cast<float>(q.q[j])), q.offset));
          ra[p][0] = packed.u[0];
          ra[p][1] = packed.u[1];
        }
      } else {
        if constexpr (CacheIqHeader)
          raw_iq[p] = {};
        ra[p][0] = make_uint4(0, 0, 0, 0);
        ra[p][1] = make_uint4(0, 0, 0, 0);
        if (Mixed && (idx / BK) % 32 >= 16)
          fetch_up(p, kk);
        else if constexpr (DecodeStage == 2)
          raw[p] = {};
        if constexpr (Mixed && UpRaw)
          raw_up[p] = {};
      }
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      const int idx = p * kThreads + tid, kk = ks0 + idx % BK;
      // Ablations 32/64 route into the zero-fill else branch instead of
      // skipping the loop: skipping leaves raw/rb undefined and lets the
      // compiler fold the dependent decode and delete s_b, which measures
      // something other than the fetch.
      if ((Ablate & 32) == 0 && (Ablate & 1048576) == 0 &&
          idx < BN * BK && (Complete || kk < ksteps)) {
        const short* src =
            (Ablate & 128) != 0
                ? x_row[p] + static_cast<std::size_t>(kk & 3) * 16
                : x_row[p] + static_cast<std::size_t>(kk) * 16;
        rb[p][0] = *reinterpret_cast<const uint4*>(src);
        rb[p][1] = *reinterpret_cast<const uint4*>(src + 8);
      } else {
        rb[p][0] = make_uint4(0, 0, 0, 0);
        rb[p][1] = make_uint4(0, 0, 0, 0);
      }
    }
  };
  const auto commit_up = [&](int p, short* dst, int row) {
    if constexpr (Mixed) {
      if constexpr (UpRaw) {
        gufo::hip::QuantSub16 q = DecodeRaw<UpType>(raw_up[p]);
        union {
          v16bf f;
          uint4 u[2];
        } packed;
        DecodeToHalves<Ablate>(q, packed.f);
        StoreCommit<Ablate>(dst, packed.u[0], packed.u[1], row);
      } else if constexpr (CacheIqHeader &&
                           UpType == gufo::core::GgmlType::kIQ4_XS) {
        const auto packed = DecodeIqRaw(raw_iq[p]);
        StoreCommit<Ablate>(dst, packed.low, packed.high, row);
      } else {
        StoreCommit<Ablate>(dst, ra[p][0], ra[p][1], row);
      }
    }
  };
  const auto commit = [&](int buf) {
#pragma unroll
    for (int p = 0; p < PFA; ++p) {
      const int idx = p * kThreads + tid, row = idx / BK, ks = idx % BK;
      if (idx < BM * BK) {
        const int placed = (row ^ (ks & 3));
        short* dst = &s_a[buf][ks][placed / 16][placed % 16][0];
        if constexpr ((Ablate & 33554432) != 0) {
          // GlobalA: the weight tile is never staged, so s_a is elided entirely.
        } else if (Mixed && row % 32 >= 16) {
          commit_up(p, dst, row);
        } else if constexpr (DecodeStage != 0 &&
                             (Ablate & (512 | 1024 | 64 | 16777216)) == 0) {
          gufo::hip::QuantSub16 q = DecodeRaw<Type>(raw[p]);
          union {
            v16bf f;
            uint4 u[2];
          } packed;
          DecodeToHalves<Ablate>(q, packed.f);
          if constexpr ((Ablate & 32) != 0) {
            // All eight words are readonly-in practice but the asm claims
            // read-write, so the decode above cannot be deleted while no LDS
            // store happens. No arithmetic is added.
            asm volatile("" : "+v"(packed.u[0].x), "+v"(packed.u[0].y),
                                "+v"(packed.u[0].z), "+v"(packed.u[0].w),
                                "+v"(packed.u[1].x), "+v"(packed.u[1].y),
                                "+v"(packed.u[1].z), "+v"(packed.u[1].w));
          } else {
            StoreCommit<Ablate>(dst, packed.u[0], packed.u[1], row);
          }
        } else if constexpr (CacheIqHeader &&
                             Type == gufo::core::GgmlType::kIQ4_XS) {
          const auto packed = DecodeIqRaw(raw_iq[p]);
          StoreCommit<Ablate>(dst, packed.low, packed.high, row);
        } else
          StoreCommit<Ablate>(dst, ra[p][0], ra[p][1], row);
      }
    }
#pragma unroll
    for (int p = 0; p < PFB; ++p) {
      // Ablation 16384 drops the activation-side staging store. The commit
      // writes two things per stage -- the decoded weights and a plain copy of
      // the activations -- and this prices them separately.
      if constexpr ((Ablate & 16384) == 0 && (Ablate & 1048576) == 0) {
        const int idx = p * kThreads + tid, row = idx / BK, ks = idx % BK;
        if (idx < BN * BK) {
          const int placed = (row ^ (ks & 3));
          short* dst = &s_b[buf][ks][placed / 16][placed % 16][0];
          // Ablation 32768: same 64 B and the same four store instructions, but
          // re-based into s_a so only one staging array is ever written.
          if constexpr ((Ablate & 32768) != 0)
            dst = reinterpret_cast<short*>(
                reinterpret_cast<char*>(&s_a[buf][0][0][0][0]) +
                (reinterpret_cast<char*>(dst) -
                 reinterpret_cast<char*>(&s_b[buf][0][0][0][0])));
          StoreCommit<Ablate>(dst, rb[p][0], rb[p][1], row);
        }
      }
    }
  };
  // Ablation 64: keep the register->LDS store but skip the decode, to price the
  // decode ALU against the store. Ablation 2 removes both; if 64 recovers the
  // whole of 2's win then the decode arithmetic is the deficit, and if it
  // recovers none then the LDS store (or the store->load turnaround) is.
  // Ablation 32 is the other half of that 2x2: pay the decode but skip the
  // store. The decoded value has to stay live or the compiler deletes the
  // arithmetic, so it is folded into an opaque register the asm marks
  // read-write; that costs one VALU and keeps the decode unconditional.
  // Ablation 512: decode the loaded weights inside the previous K loop instead
  // of in the commit, so the ~90 VALU of Q4_K decode overlap live matrix work
  // rather than running in the post-barrier bubble with the matrix pipe idle.
  const auto decode_ra = [&]() {
    if constexpr ((Ablate & (512 | 1024)) != 0) {
      static_assert(
          DecodeStage == 2 && !Mixed && !CacheIqHeader && !CacheQ5Header,
          "ablation 512 covers only the plain deferred-decode path");
#pragma unroll
      for (int p = 0; p < PFA; ++p) {
        const int idx = p * kThreads + tid;
        if (idx < BM * BK) {
          gufo::hip::QuantSub16 q = DecodeRaw<Type>(raw[p]);
          union {
            v16bf f;
            uint4 u[2];
          } packed;
          DecodeToHalves<Ablate>(q, packed.f);
          ra[p][0] = packed.u[0];
          ra[p][1] = packed.u[1];
        }
      }
    }
  };
#if defined(GUFO_FP16_PROTO_CLOCK)
  unsigned long long pc_commit = 0, pc_barrier = 0, pc_fetch = 0, pc_loop = 0;
  unsigned long long pc_t0 = 0;
#define PC_START() pc_t0 = ProtoClock()
#define PC_END(acc) (acc) += ProtoClock() - pc_t0
#else
#define PC_START()
#define PC_END(acc)
#endif
  fetch(0);
  decode_ra();
  if constexpr (DblBuf) {
    // Stage i+1's commit is issued before stage i's K loop, so the LDS writes
    // drain while the matrix pipe runs. The stage loop is unrolled by two so the
    // buffer index is a compile-time constant: with a runtime `stage & 1` the
    // pointers into s_a are not provably disjoint and the compiler still emits
    // `s_waitcnt lgkmcnt(0)` before the barrier, which is why the runtime-index
    // version of this measured no faster than the single buffer.
    const auto kloop = [&]<int Buf>(int ks0base) {
      constexpr int buf = Buf;
#pragma unroll
      for (int ks = 0; ks < BK; ++ks) {
        v16bf a[WRS];
#pragma unroll
        for (int i = 0; i < WRS; ++i)
          a[i] =
              LoadSwizzled(&s_a[buf][ks][(wr * WRS) + i][sl ^ (ks & 3)][0]);
        const short* bbase = nullptr;
        if constexpr ((Ablate & 1048576) != 0)
          bbase = b_blocked +
                  (static_cast<std::size_t>(ks0base + ks) * batch + t_block + sl) * 16;
#pragma unroll
        for (int j = 0; j < WTS; ++j) {
          v16bf b;
          if constexpr ((Ablate & 1048576) != 0)
            b = LoadBlockedBAt(bbase + (j * 256));
          else
            b = LoadSwizzled(&s_b[buf][ks][(wt * WTS) + j][sl ^ (ks & 3)][0]);
#pragma unroll
          for (int i = 0; i < WRS; ++i) {
            if constexpr (Type == gufo::core::GgmlType::kIQ3_S)
              acc[i][j] = WmmaHalf(a[i], b, acc[i][j]);
            else
              acc[i][j] = WmmaHalf(b, a[i], acc[i][j]);
          }
        }
      }
    };
    commit(0);
    __syncthreads();
    int ks0 = 0;
    while (ks0 + 2 * BK <= ksteps) {
      if (ks0 + BK < ksteps) {
        fetch(ks0 + BK);
        commit(1);
      }
      kloop.template operator()<0>(ks0);
      __syncthreads();
      if (ks0 + 2 * BK < ksteps) {
        fetch(ks0 + 2 * BK);
        commit(0);
      }
      kloop.template operator()<1>(ks0);
      __syncthreads();
      ks0 += 2 * BK;
    }
    // An odd stage count leaves the last stage in buffer 0, committed above.
    if (ks0 < ksteps) {
      kloop.template operator()<0>(ks0);
      __syncthreads();
    }
  } else {
  for (int ks0 = 0; ks0 < ksteps; ks0 += BK) {
    PC_START();
    if constexpr ((Ablate & 2) == 0)
      commit(0);
    PC_END(pc_commit);
    PC_START();
    if constexpr ((Ablate & 1) == 0)
      __syncthreads();
    PC_END(pc_barrier);
    PC_START();
    if constexpr ((Ablate & 4) == 0 && (Ablate & 256) == 0) {
      if (ks0 + BK < ksteps)
        fetch(ks0 + BK);
    }
    PC_END(pc_fetch);
    PC_START();
#pragma unroll
    for (int ks = 0; ks < BK; ++ks) {
      v16bf a[WRS];
      const short* abase = nullptr;
      if constexpr ((Ablate & 33554432) != 0)
        abase = w16a +
                (static_cast<std::size_t>(ks0 + ks) * m + r_block + sl) * 16;
#pragma unroll
      for (int i = 0; i < WRS; ++i) {
        if constexpr ((Ablate & 33554432) != 0)
          a[i] = LoadBlockedBAt(abase + ((wr * WRS + i) * 256));
        else
          a[i] = LoadSwizzled(&s_a[0][ks][(wr * WRS) + i][sl ^ (ks & 3)][0]);
      }
      // Ablation 4096: batch every B fragment load ahead of the MMA block. In
      // the shipped order each B fragment is loaded and consumed back to back,
      // so its load-to-use distance is zero; batching gives all of them the MMA
      // block to land behind.
      if constexpr ((Ablate & 4096) != 0) {
        v16bf bb[WTS];
#pragma unroll
        for (int j = 0; j < WTS; ++j)
          bb[j] = LoadSwizzled(&s_b[0][ks][(wt * WTS) + j][sl ^ (ks & 3)][0]);
#pragma unroll
        for (int j = 0; j < WTS; ++j) {
#pragma unroll
          for (int i = 0; i < WRS; ++i) {
            if constexpr (Type == gufo::core::GgmlType::kIQ3_S)
              acc[i][j] = WmmaHalf(a[i], bb[j], acc[i][j]);
            else
              acc[i][j] = WmmaHalf(bb[j], a[i], acc[i][j]);
          }
        }
      } else {
        const short* bbase = nullptr;
        if constexpr ((Ablate & 1048576) != 0)
          bbase = b_blocked +
                  (static_cast<std::size_t>(ks0 + ks) * batch + t_block + sl) * 16;
#pragma unroll
        for (int j = 0; j < WTS; ++j) {
          v16bf b;
          if constexpr ((Ablate & 1048576) != 0)
            b = LoadBlockedBAt(bbase + (j * 256));
          else
            b = LoadSwizzled(&s_b[0][ks][(wt * WTS) + j][sl ^ (ks & 3)][0]);
#pragma unroll
          for (int i = 0; i < WRS; ++i) {
            if constexpr (Type == gufo::core::GgmlType::kIQ3_S)
              acc[i][j] = WmmaHalf(a[i], b, acc[i][j]);
            else
              acc[i][j] = WmmaHalf(b, a[i], acc[i][j]);
          }
        }
      }
      if constexpr ((Ablate & 4) == 0 && (Ablate & 256) != 0) {
        if (ks == BK / 2 - 1 && ks0 + BK < ksteps)
          fetch(ks0 + BK);
      }
      if constexpr ((Ablate & 512) != 0) {
        if (ks == BK / 2 - 1 && ks0 + BK < ksteps)
          decode_ra();
      }
      // Ablation 1024: stagger the decode to a warp-dependent sub-stage.
      // Each warp has its own PC on RDNA3, so the warps no longer all enter
      // the decode together after the same barrier and the matrix pipe keeps
      // being fed by the warps still issuing MMAs.
      if constexpr ((Ablate & 1024) != 0) {
        if (ks == (wave & (BK - 1)) && ks0 + BK < ksteps)
          decode_ra();
      }
    }
    if constexpr ((Ablate & (16 | 2048 | 8192)) != 0 ||
                  (BM == 256 && BN == 256 &&
                   ((Type == gufo::core::GgmlType::kIQ4_XS &&
                     (!Mixed || UpType != gufo::core::GgmlType::kQ5_K)) ||
                    (Type == gufo::core::GgmlType::kIQ4_NL &&
                     UpType == gufo::core::GgmlType::kIQ4_XS) ||
                    Type == gufo::core::GgmlType::kQ3_K ||
                    Type == gufo::core::GgmlType::kQ2_K ||
                    Type == gufo::core::GgmlType::kIQ3_S ||
                    Type == gufo::core::GgmlType::kIQ3_XXS ||
                    Type == gufo::core::GgmlType::kIQ2_XS ||
                    Type == gufo::core::GgmlType::kIQ2_S ||
                    Type == gufo::core::GgmlType::kIQ2_XXS))) {
      // Interleave LDS reads with WMMA after the grid decoders. Mixed Q5
      // decoders retain the compiler's default instruction schedule.
      // Ablations 2048/8192 select the other iglp_opt immediates; 16 keeps
      // the value the other instantiations already use.
      if constexpr ((Ablate & 2048) != 0)
        __builtin_amdgcn_iglp_opt(1);
      else if constexpr ((Ablate & 8192) != 0)
        __builtin_amdgcn_iglp_opt(2);
      else
        __builtin_amdgcn_iglp_opt(0);
    }
    PC_END(pc_loop);
    PC_START();
    if constexpr ((Ablate & 1) == 0)
      __syncthreads();
    PC_END(pc_barrier);
  }
  }
#if defined(GUFO_FP16_PROTO_CLOCK)
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    g_proto_clock[0] = pc_commit;
    g_proto_clock[1] = pc_barrier;
    g_proto_clock[2] = pc_fetch;
    g_proto_clock[3] = pc_loop;
  }
#undef PC_START
#undef PC_END
#endif

  // Orient WMMA output as [token][row] for coalesced direct stores. IQ3_S
  // keeps the original orientation and LDS transpose, which is faster for
  // its decoder's register schedule.
  const auto store = [&](std::size_t index, float value) {
    if constexpr ((Ablate & 8) != 0) {
      // Epilogue ablation: keep the accumulators live, never store.
      if (value == 1.0e38F)
        y[index] = value;
    } else if constexpr (Epilogue == HalfEpilogue::kSwiGLU) {
      const float g = gate[index];
      const float sig = 1.0F / (1.0F + expf(-g));
      reinterpret_cast<__half*>(y)[index] =
          RoundActivation(__fmul_rn(g * sig, value));
    } else if constexpr (Epilogue == HalfEpilogue::kResidual) {
      // Add only after the dot product, preserving the FP32 residual link.
      y[index] = __fadd_rn(y[index], value);
    } else
      y[index] = value;
  };
  const int hi = lane / 16;
  if constexpr (Paired) {
    static_assert(WRS % 2 == 0 && Type != gufo::core::GgmlType::kIQ3_S);
    // Adjacent 16-row fragments hold gate and up for the same logical rows.
    // One wave owns both accumulators, so no intermediate store or exchange
    // between waves is needed.
#pragma unroll
    for (int i = 0; i < WRS; i += 2) {
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const int row = r_block + (wr * (WRS / 2) + i / 2) * 16 + sl;
        const int t0 = t_block + (wt * WTS + j) * 16;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          const int token = t0 + 2 * l + hi;
          if (Complete || (row < m && token < batch)) {
            const float g = acc[i][j][l], u = acc[i + 1][j][l];
            const float sig = 1.0F / (1.0F + expf(-g));
            reinterpret_cast<__half*>(y)[std::size_t(token) * m + row] =
                RoundActivation(__fmul_rn(g * sig, u));
          }
        }
      }
    }
  } else if constexpr (Type != gufo::core::GgmlType::kIQ3_S) {
#pragma unroll
    for (int i = 0; i < WRS; ++i) {
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
        const int row = r_block + ((wr * WRS + i) * 16) + sl;
        const int t0 = t_block + ((wt * WTS + j) * 16);
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          const int token = t0 + 2 * l + hi;
          if (Complete || (row < m && token < batch))
            store(std::size_t(token) * m + row, acc[i][j][l]);
        }
      }
    }
  } else {
    __syncthreads();
    constexpr int Slots =
        (sizeof(s_a) >= sizeof(s_b) ? sizeof(s_a) : sizeof(s_b)) / (256 * 4);
    static_assert(Slots >= WM * WN);
    float* scratch = (sizeof(s_a) >= sizeof(s_b)
                          ? reinterpret_cast<float*>(&s_a[0][0][0][0])
                          : reinterpret_cast<float*>(&s_b[0][0][0][0])) +
                     (wave * 256);

#pragma unroll
    for (int i = 0; i < WRS; ++i) {
#pragma unroll
      for (int j = 0; j < WTS; ++j) {
#pragma unroll
        for (int l = 0; l < 256 / WaveSize; ++l) {
          scratch[(sl * 16) + ((WaveSize / 16) * l) + hi] = acc[i][j][l];
        }
        __builtin_amdgcn_wave_barrier();
        const int r0 = r_block + (((wr * WRS) + i) * 16);
        const int t0 = t_block + (((wt * WTS) + j) * 16);
        if (t0 + 16 <= batch && r0 + 16 <= m) {
          const std::size_t base =
              (static_cast<std::size_t>(t0) * m) + static_cast<std::size_t>(r0);
#pragma unroll
          for (int s = 0; s < 256 / WaveSize; ++s) {
            const int flat = (s * WaveSize) + lane;
            store(
                base + (static_cast<std::size_t>(flat >> 4) * m) + (flat & 15),
                scratch[flat]);
          }
        } else {
#pragma unroll
          for (int s = 0; s < 256 / WaveSize; ++s) {
            const int flat = (s * WaveSize) + lane;
            const int tok = t0 + (flat >> 4);
            const int r = r0 + (flat & 15);
            if (tok < batch && r < m) {
              store((static_cast<std::size_t>(tok) * m) + r, scratch[flat]);
            }
          }
        }
        __builtin_amdgcn_wave_barrier();
      }
    }
  }
#if defined(GUFO_FP16_PROTO_CLOCK)
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    g_proto_clock[4] = proto_block_start;
    g_proto_clock[5] = ProtoClock();
  }
#endif
}

__global__ void HalfNorm(const float* x, const float* residual,
                         const float* weight, float* sum_out, __half* out,
                         std::size_t dim, float eps) {
  const std::size_t row = blockIdx.x, tid = threadIdx.x, base = row * dim;
  float sum_sq = 0;
  for (std::size_t i = tid; i < dim; i += 256) {
    const float v =
        residual ? __fadd_rn(x[base + i], residual[base + i]) : x[base + i];
    sum_sq += v * v;
  }
  __shared__ float sums[256];
  sums[tid] = sum_sq;
  __syncthreads();
  for (unsigned stride = 128; stride; stride /= 2) {
    if (tid < stride)
      sums[tid] += sums[tid + stride];
    __syncthreads();
  }
  const float inv = rsqrtf(sums[0] / static_cast<float>(dim) + eps);
  for (std::size_t i = tid; i < dim; i += 256) {
    const float v =
        residual ? __fadd_rn(x[base + i], residual[base + i]) : x[base + i];
    if (sum_out)
      sum_out[base + i] = v;
    out[base + i] = RoundActivation(__fmul_rn(v * inv, weight[i]));
  }
}
__global__ void HalfNorm5120(const float* __restrict__ x,
                             const float* __restrict__ weight,
                             __half* __restrict__ out, float eps) {
  constexpr unsigned Dim = 5120, Elements = Dim / 256;
  const unsigned row = blockIdx.x, tid = threadIdx.x;
  const std::size_t base = std::size_t(row) * Dim;
  float sum_sq = 0;
#pragma unroll
  for (unsigned j = 0; j < Elements; ++j) {
    const float v = x[base + j * 256 + tid];
    // Explicit FMAs preserve which square rounds first when LLVM unrolls
    // the loop; contracting two ordinary products can swap that boundary.
    sum_sq = __fmaf_rn(v, v, sum_sq);
  }
  __shared__ float sums[256];
  sums[tid] = sum_sq;
  __syncthreads();
  for (unsigned stride = 128; stride; stride /= 2) {
    if (tid < stride)
      sums[tid] += sums[tid + stride];
    __syncthreads();
  }
  const float inv = rsqrtf(sums[0] / static_cast<float>(Dim) + eps);
#pragma unroll
  for (unsigned j = 0; j < Elements; ++j) {
    const unsigned i = j * 256 + tid;
    out[base + i] = RoundActivation(__fmul_rn(x[base + i] * inv, weight[i]));
  }
}

__global__ void HalfCast(const float* x, __half* y, std::size_t n) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = __float2half_rn(x[i]);
}

// Diagnostic: round every 8 consecutive FP16 activations onto a shared-exponent
// grid with "bits" magnitude bits, in place. This is what the NPU path's bfp16
// operand encoder does to its A operand, minus the byte packing, so the
// model-level cost of that encoding can be measured with the held-out
// instrument instead of argued from a per-GEMM number. The 8-element group is
// the real one -- consecutive k within one row -- and row lengths are multiples
// of 8, so no group straddles rows.
__global__ void Bfp16RoundTripFp16Kernel(__half* data, std::size_t count,
                                         int bits) {
  const std::size_t group =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t base = group * 8;
  if (base + 8 > count) {
    return;
  }
  float value[8];
  float amax = 0.0F;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    value[i] = __half2float(data[base + i]);
    amax = fmaxf(amax, fabsf(value[i]));
  }
  if (amax == 0.0F) {
    return;
  }
  // amax lands in [0.5, 1) * 2^exponent, so this step reproduces the encoder's
  // 2^(max_biased_exponent - 127 - (mbits - 1)) grid.
  int exponent = 0;
  (void)frexpf(amax, &exponent);
  const float step = ldexpf(1.0F, exponent - bits);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    data[base + i] = __float2half(rintf(value[i] / step) * step);
  }
}

// config3's ATB L1 tile: 64 reduction rows by 128 output columns, emitted
// column-major, each tile shuffled into 1x2 super-blocks of 8x8 column-major
// sub-blocks. Numbers from the vendored layout header.
constexpr unsigned kAtbL1K = 64;
constexpr unsigned kAtbL1N = 128;
constexpr unsigned kAtbGroupsPerTile = (kAtbL1K * kAtbL1N) / 8;

// Runtime repack of one quantised weight tensor into the ATB bfp16 B operand.
// One thread per bfp16 group: 8 consecutive reduction rows at one output
// column, which the shuffle makes contiguous, so each thread writes 9
// contiguous bytes read from 8 weights of one row. This is what removes the
// need to hold a packed copy of the model -- the staging buffer is reused.
__global__ void AtbRepackKernel(const void* __restrict__ w,
                                std::size_t row_bytes, int type_id,
                                std::size_t l1_rows, std::size_t groups,
                                std::uint8_t* __restrict__ out) {
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= groups) {
    return;
  }
  const std::size_t tile = g / kAtbGroupsPerTile;
  const std::size_t gt = g % kAtbGroupsPerTile;
  const std::size_t l1c = tile / l1_rows;
  const std::size_t l1r = tile % l1_rows;
  const unsigned cib = static_cast<unsigned>(gt % 8);
  const unsigned t1 = static_cast<unsigned>(gt / 8);
  const unsigned bis = t1 % 2;
  const unsigned t2 = t1 / 2;
  const unsigned sbr = t2 % 8;
  const unsigned sb = t2 / 8;
  const unsigned sbc = 2 * sb;

  const std::size_t n = l1c * kAtbL1N + (sbc + bis) * 8 + cib;
  const std::size_t k0 = l1r * kAtbL1K + sbr * 8;

  const void* row = static_cast<const std::uint8_t*>(w) + n * row_bytes;
  const core::GgmlType type = static_cast<core::GgmlType>(type_id);
  // k0 is a multiple of 8, so a group is one half of one 16-element sub-block
  // and never straddles two. Decoding per element instead repeats this 8x,
  // which cost 2.8x on the first version.
  QuantSub16 sub;
  DecodeQuantSub16(type, row, k0 / 16, sub);
  const unsigned within = static_cast<unsigned>(k0 % 16);
  float value[8];
  float amax = 0.0F;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    value[i] = sub.scale * static_cast<float>(sub.q[within + i]) - sub.offset;
    amax = fmaxf(amax, fabsf(value[i]));
  }
  std::uint8_t* dst = out + g * 9;
  if (amax == 0.0F) {
    for (int i = 0; i < 9; ++i) {
      dst[i] = 0;
    }
    return;
  }
  int exponent = 0;
  (void)frexpf(amax, &exponent);
  dst[0] = static_cast<std::uint8_t>(exponent + 126);
  const float step = ldexpf(1.0F, exponent - 7);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int magnitude = static_cast<int>(floorf(value[i] / step + 0.5F));
    dst[1 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(magnitude));
  }
}
// Slice form of AtbRepackKernel: packs only the n tiles [n_tile0, ...) into a
// buffer whose own tile numbering starts at zero, which is exactly what the ATB
// kernel expects when its N is the slice width. This is what lets the split
// repack only the NPU's share of a weight tensor.
__global__ void AtbRepackSliceKernel(const void* __restrict__ w,
                                     std::size_t row_bytes, int type_id,
                                     std::size_t l1_rows, std::size_t n_tile0,
                                     std::size_t groups,
                                     std::uint8_t* __restrict__ out) {
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= groups) {
    return;
  }
  const std::size_t tile = g / kAtbGroupsPerTile;
  const std::size_t gt = g % kAtbGroupsPerTile;
  const std::size_t l1c = tile / l1_rows;
  const std::size_t l1r = tile % l1_rows;
  const unsigned cib = static_cast<unsigned>(gt % 8);
  const unsigned t1 = static_cast<unsigned>(gt / 8);
  const unsigned bis = t1 % 2;
  const unsigned t2 = t1 / 2;
  const unsigned sbr = t2 % 8;
  const unsigned sb = t2 / 8;
  const unsigned sbc = 2 * sb;

  const std::size_t n = (n_tile0 + l1c) * kAtbL1N + (sbc + bis) * 8 + cib;
  const std::size_t k0 = l1r * kAtbL1K + sbr * 8;

  const void* row = static_cast<const std::uint8_t*>(w) + n * row_bytes;
  const core::GgmlType type = static_cast<core::GgmlType>(type_id);
  QuantSub16 sub;
  DecodeQuantSub16(type, row, k0 / 16, sub);
  const unsigned within = static_cast<unsigned>(k0 % 16);
  float value[8];
  float amax = 0.0F;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    value[i] = sub.scale * static_cast<float>(sub.q[within + i]) - sub.offset;
    amax = fmaxf(amax, fabsf(value[i]));
  }
  std::uint8_t* dst = out + g * 9;
  if (amax == 0.0F) {
    for (int i = 0; i < 9; ++i) {
      dst[i] = 0;
    }
    return;
  }
  int exponent = 0;
  (void)frexpf(amax, &exponent);
  dst[0] = static_cast<std::uint8_t>(exponent + 126);
  const float step = ldexpf(1.0F, exponent - 7);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int magnitude = static_cast<int>(floorf(value[i] / step + 0.5F));
    dst[1 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(magnitude));
  }
}

// The GEMM kernels index their output as y[token * m + row], so a projection
// that computes only part of the output width writes a packed [batch, m]
// buffer. This moves such a head into the first columns of a full-width row,
// which is what the down projection reads. One pass over the head only: the
// tail columns are filled by the NPU's own decode.
__global__ void AtbExpandHeadFp16Kernel(const __half* __restrict__ packed,
                                        __half* __restrict__ out,
                                        std::size_t head_cols,
                                        std::size_t full_cols,
                                        std::size_t rows) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * head_cols) {
    return;
  }
  const std::size_t r = i / head_cols;
  const std::size_t c = i % head_cols;
  out[r * full_cols + c] = packed[i];
}

// The same packed-head problem for the down projection, whose residual lives
// at a full-width stride: the GPU's partial columns are added into the residual
// rows here while the NPU accumulates the columns after them.
__global__ void AtbAddHeadFp32Kernel(const float* __restrict__ packed,
                                     float* __restrict__ out,
                                     std::size_t head_cols,
                                     std::size_t full_cols,
                                     std::size_t rows) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * head_cols) {
    return;
  }
  const std::size_t r = i / head_cols;
  const std::size_t c = i % head_cols;
  out[r * full_cols + c] += packed[i];
}

// ATB A operand: FP16 activations [rows, k] into the bfp16 A L1 layout, the
// same transform as tools/qwen27b/atb_encode_a.hip. L1 tiles of (128, 64) with
// the k tile varying fastest, each shuffled into 2x1 vertically stacked 8x8
// row-major super-blocks, then one shared exponent per 8 consecutive elements.
// Groups of 8 are consecutive k within one row, so a group never straddles a
// row and the shared exponent is the same set the NPU sees.
constexpr unsigned kAtbAL1M = 128;
constexpr unsigned kAtbAL1K = 64;
constexpr unsigned kAtbAGroupsPerTile = (kAtbAL1M * kAtbAL1K) / 8;

__global__ void AtbEncodeAKernel(const __half* __restrict__ a, int k,
                                 std::size_t l1_cols, std::size_t groups,
                                 std::uint8_t* __restrict__ out) {
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= groups) {
    return;
  }
  const std::size_t tile = g / kAtbAGroupsPerTile;
  const std::size_t gt = g % kAtbAGroupsPerTile;
  const std::size_t l1r = tile / l1_cols;
  const std::size_t l1c = tile % l1_cols;
  const unsigned rib = static_cast<unsigned>(gt % 8);
  const unsigned t1 = static_cast<unsigned>(gt / 8);
  const unsigned bis = t1 % 2;
  const unsigned t2 = t1 / 2;
  const unsigned sbc = t2 % 8;
  const unsigned si = t2 / 8;

  const unsigned r = (2 * si + bis) * 8 + rib;
  const std::size_t row = l1r * kAtbAL1M + r;
  const __half* src = a + row * static_cast<std::size_t>(k) + l1c * kAtbAL1K + sbc * 8;

  float value[8];
  float amax = 0.0F;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    value[i] = __half2float(src[i]);
    amax = fmaxf(amax, fabsf(value[i]));
  }
  std::uint8_t* dst = out + g * 9;
  if (amax == 0.0F) {
    for (int i = 0; i < 9; ++i) {
      dst[i] = 0;
    }
    return;
  }
  int exponent = 0;
  (void)frexpf(amax, &exponent);
  dst[0] = static_cast<std::uint8_t>(exponent + 126);
  const float step = ldexpf(1.0F, exponent - 7);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int magnitude = static_cast<int>(floorf(value[i] / step + 0.5F));
    dst[1 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(magnitude));
  }
}

// ATB C operand: bfp16ebs8 in L1 tiles of (512, 128) laid out row-major, each
// tile 2x2 super-blocks of 8x8 row-major sub-blocks. Consecutive 8 packed
// elements are 8 consecutive output columns of one row, so one thread per
// 8-element group decodes 9 bytes and writes 8 outputs. n_offset and n_full
// place a slice of the output width inside the engine's full FP32/FP16 row.
constexpr unsigned kAtbCL1M = 512;
constexpr unsigned kAtbCL1N = 128;
constexpr unsigned kAtbCGroupsPerTile = (kAtbCL1M * kAtbCL1N) / 8;

__device__ __forceinline__ float Bfp16DecodeByte(std::uint8_t byte,
                                                 float multiplier) {
  const bool negative = (byte & 0x80u) != 0;
  if (negative) {
    const float magnitude =
        static_cast<float>(static_cast<std::uint8_t>(~(byte - 1)));
    return -(magnitude * multiplier);
  }
  return static_cast<float>(byte) * multiplier;
}

__device__ __forceinline__ float Bfp16DecodeScale(std::uint8_t shared) {
  const float multiplier = shared >= 127
                               ? static_cast<float>(1u << (shared - 127))
                               : 1.0F / static_cast<float>(1u << (127 - shared));
  return multiplier / 64.0F;
}

// Where one packed 8-element group of C lands in the engine's full-width row.
struct AtbCGroup {
  std::size_t row;
  std::size_t col0;
};

__device__ __forceinline__ AtbCGroup AtbCGroupIndex(std::size_t g,
                                                    std::size_t l1_cols,
                                                    std::size_t n_offset) {
  const std::size_t tile = g / kAtbCGroupsPerTile;
  const std::size_t gi = g % kAtbCGroupsPerTile;
  const std::size_t l1r = tile / l1_cols;
  const std::size_t l1c = tile % l1_cols;

  const unsigned rib = static_cast<unsigned>(gi % 8);
  const unsigned t1 = static_cast<unsigned>(gi / 8);
  const unsigned block = t1 % 4;
  const unsigned t2 = t1 / 4;
  const unsigned sbc_half = t2 % 8;
  const unsigned sbr_half = t2 / 8;
  const unsigned br = block / 2;
  const unsigned bc = block % 2;

  AtbCGroup out;
  out.row = l1r * kAtbCL1M + (2 * sbr_half + br) * 8 + rib;
  out.col0 = n_offset + l1c * kAtbCL1N + (2 * sbc_half + bc) * 8;
  return out;
}

enum class AtbDecodeOp { kWrite, kAdd };

template <typename OutT, AtbDecodeOp Op>
__global__ void AtbDecodeCKernel(const std::uint8_t* __restrict__ packed,
                                 std::size_t l1_cols, std::size_t groups,
                                 std::size_t n_full, std::size_t n_offset,
                                 OutT* __restrict__ out) {
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= groups) {
    return;
  }
  const AtbCGroup at = AtbCGroupIndex(g, l1_cols, n_offset);
  const std::uint8_t* src = packed + g * 9;
  const float multiplier = Bfp16DecodeScale(src[0]);
  OutT* dst = out + at.row * n_full + at.col0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float value = Bfp16DecodeByte(src[1 + i], multiplier);
    if constexpr (Op == AtbDecodeOp::kAdd) {
      dst[i] = static_cast<OutT>(static_cast<float>(dst[i]) + value);
    } else {
      dst[i] = static_cast<OutT>(value);
    }
  }
}

// Gate and up share a layout, so one thread can decode both and emit the FP16
// SwiGLU the down projection consumes, instead of decoding two FP32 rows, then
// running a separate activation pass over them.
__global__ void AtbDecodeSwiGLUKernel(const std::uint8_t* __restrict__ gate,
                                      const std::uint8_t* __restrict__ up,
                                      std::size_t l1_cols, std::size_t groups,
                                      std::size_t n_full, std::size_t n_offset,
                                      __half* __restrict__ out) {
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= groups) {
    return;
  }
  const AtbCGroup at = AtbCGroupIndex(g, l1_cols, n_offset);
  const std::uint8_t* gs = gate + g * 9;
  const std::uint8_t* us = up + g * 9;
  const float gm = Bfp16DecodeScale(gs[0]);
  const float um = Bfp16DecodeScale(us[0]);
  __half* dst = out + at.row * n_full + at.col0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float value = Bfp16DecodeByte(gs[1 + i], gm);
    const float gate_value = value / (1.0F + expf(-value));
    dst[i] = __float2half_rn(gate_value * Bfp16DecodeByte(us[1 + i], um));
  }
}

}  // namespace
void LaunchBatchedRMSNormFp16(const float* x, const float* residual,
                              const float* weight, float* sum_out, void* out,
                              std::size_t batch, std::size_t dim, float eps,
                              hipStream_t stream) {
  if (dim == 5120 && residual == nullptr && sum_out == nullptr) {
    HalfNorm5120<<<batch, 256, 0, stream>>>(x, weight,
                                            static_cast<__half*>(out), eps);
    return;
  }
  HalfNorm<<<batch, 256, 0, stream>>>(x, residual, weight, sum_out,
                                      static_cast<__half*>(out), dim, eps);
}
void LaunchFloatToFp16(const float* x, void* y, std::size_t n,
                       hipStream_t stream) {
  HalfCast<<<(n + 255) / 256, 256, 0, stream>>>(x, static_cast<__half*>(y), n);
}

void LaunchBfp16RoundTripFp16InPlace(void* buffer, std::size_t count, int bits,
                                     hipStream_t stream) {
  if (buffer == nullptr || count < 8 || bits < 1 || bits >= 10) {
    return;
  }
  const std::size_t groups = count / 8;
  Bfp16RoundTripFp16Kernel<<<static_cast<unsigned int>((groups + 255) / 256),
                             256, 0, stream>>>(static_cast<__half*>(buffer),
                                               groups * 8, bits);
}

void LaunchAtbRepackBfp16(const void* weights, core::GgmlType type,
                          std::size_t n_out, std::size_t k, void* out,
                          hipStream_t stream) {
  if (weights == nullptr || out == nullptr) {
    return;
  }
  const std::size_t row_bytes = quant::QuantizedRowBytes(type, k);
  if (row_bytes == 0 || k % kAtbL1K != 0 || n_out % kAtbL1N != 0) {
    return;
  }
  const std::size_t groups = (n_out * k) / 8;
  const std::size_t blocks = (groups + 255) / 256;
  AtbRepackKernel<<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
      weights, row_bytes, static_cast<int>(type), k / kAtbL1K, groups,
      static_cast<std::uint8_t*>(out));
}
void LaunchAtbEncodeAFp16(const void* act_fp16, std::size_t rows,
                          std::size_t k, void* out, hipStream_t stream) {
  if (act_fp16 == nullptr || out == nullptr || k % kAtbAL1K != 0 ||
      rows % kAtbAL1M != 0) {
    return;
  }
  const std::size_t groups = rows * k / 8;
  const std::size_t blocks = (groups + 255) / 256;
  AtbEncodeAKernel<<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
      static_cast<const __half*>(act_fp16), static_cast<int>(k), k / kAtbAL1K,
      groups, static_cast<std::uint8_t*>(out));
}

void LaunchAtbRepackBfp16Slice(const void* weights, core::GgmlType type,
                               std::size_t n_out, std::size_t k,
                               std::size_t n_offset, std::size_t n_tiles,
                               void* out, hipStream_t stream) {
  if (weights == nullptr || out == nullptr || n_tiles == 0) {
    return;
  }
  const std::size_t row_bytes = quant::QuantizedRowBytes(type, k);
  if (row_bytes == 0 || k % kAtbL1K != 0 ||
      n_offset + n_tiles * kAtbL1N > n_out) {
    return;
  }
  const std::size_t groups = n_tiles * kAtbL1N * k / 8;
  const std::size_t blocks = (groups + 255) / 256;
  AtbRepackSliceKernel<<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
      weights, row_bytes, static_cast<int>(type), k / kAtbL1K,
      n_offset / kAtbL1N, groups, static_cast<std::uint8_t*>(out));
}

void LaunchAtbDecodeCFp32(const void* packed, std::size_t rows,
                          std::size_t n_slice, std::size_t n_full,
                          std::size_t n_offset, float* out,
                          hipStream_t stream) {
  if (packed == nullptr || out == nullptr || rows % kAtbCL1M != 0 ||
      n_slice % kAtbCL1N != 0) {
    return;
  }
  const std::size_t groups = rows * n_slice / 8;
  const std::size_t blocks = (groups + 255) / 256;
  AtbDecodeCKernel<float, AtbDecodeOp::kWrite>
      <<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
      static_cast<const std::uint8_t*>(packed), n_slice / kAtbCL1N, groups,
      n_full, n_offset, out);
}

void LaunchAtbDecodeCFp16(const void* packed, std::size_t rows,
                          std::size_t n_slice, std::size_t n_full,
                          std::size_t n_offset, void* out,
                          hipStream_t stream) {
  if (packed == nullptr || out == nullptr || rows % kAtbCL1M != 0 ||
      n_slice % kAtbCL1N != 0) {
    return;
  }
  const std::size_t groups = rows * n_slice / 8;
  const std::size_t blocks = (groups + 255) / 256;
  AtbDecodeCKernel<__half, AtbDecodeOp::kWrite>
      <<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
      static_cast<const std::uint8_t*>(packed), n_slice / kAtbCL1N, groups,
      n_full, n_offset, static_cast<__half*>(out));
}

void LaunchAtbExpandHeadFp16(const void* packed, void* out,
                               std::size_t rows, std::size_t head_cols,
                               std::size_t full_cols, hipStream_t stream) {
  if (packed == nullptr || out == nullptr || head_cols == 0 ||
      head_cols > full_cols) {
    return;
  }
  const std::size_t count = rows * head_cols;
  const std::size_t blocks = (count + 255) / 256;
  AtbExpandHeadFp16Kernel<<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
      static_cast<const __half*>(packed), static_cast<__half*>(out), head_cols,
      full_cols, rows);
}

void LaunchAtbAddHeadFp32(const float* packed, float* out,
                             std::size_t rows, std::size_t head_cols,
                             std::size_t full_cols, hipStream_t stream) {
  if (packed == nullptr || out == nullptr || head_cols == 0 ||
      head_cols > full_cols) {
    return;
  }
  const std::size_t count = rows * head_cols;
  const std::size_t blocks = (count + 255) / 256;
  AtbAddHeadFp32Kernel<<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
      packed, out, head_cols, full_cols, rows);
}

void LaunchAtbDecodeCAccumulateFp32(const void* packed, std::size_t rows,
                                    std::size_t n_slice, std::size_t n_full,
                                    std::size_t n_offset, float* out,
                                    hipStream_t stream) {
  if (packed == nullptr || out == nullptr || rows % kAtbCL1M != 0 ||
      n_slice % kAtbCL1N != 0) {
    return;
  }
  const std::size_t groups = rows * n_slice / 8;
  const std::size_t blocks = (groups + 255) / 256;
  AtbDecodeCKernel<float, AtbDecodeOp::kAdd>
      <<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
          static_cast<const std::uint8_t*>(packed), n_slice / kAtbCL1N, groups,
          n_full, n_offset, out);
}

void LaunchAtbDecodeSwiGLUFp16(const void* gate_packed, const void* up_packed,
                               std::size_t rows, std::size_t n_slice,
                               std::size_t n_full, std::size_t n_offset,
                               void* out, hipStream_t stream) {
  if (gate_packed == nullptr || up_packed == nullptr || out == nullptr ||
      rows % kAtbCL1M != 0 || n_slice % kAtbCL1N != 0) {
    return;
  }
  const std::size_t groups = rows * n_slice / 8;
  const std::size_t blocks = (groups + 255) / 256;
  AtbDecodeSwiGLUKernel<<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
      static_cast<const std::uint8_t*>(gate_packed),
      static_cast<const std::uint8_t*>(up_packed), n_slice / kAtbCL1N, groups,
      n_full, n_offset, static_cast<__half*>(out));
}

template<HalfEpilogue Epilogue>
void DirectGemm(gufo::core::GgmlType type, const void* w, const void* x,
                float* y, std::size_t batch, std::size_t m, std::size_t k,
                const float* gate, hipStream_t stream) {
  const auto launch = [&]<gufo::core::GgmlType Type>() {
    constexpr bool LargeKvTile = Type == gufo::core::GgmlType::kQ4_K ||
                                 Type == gufo::core::GgmlType::kQ5_K ||
                                 Type == gufo::core::GgmlType::kQ6_K ||
                                 Type == gufo::core::GgmlType::kQ8_0;
    const auto large = [&]<bool Full>() {
      HalfPrefillGemmKernel<256, 256, 8, 4, Type, Epilogue, Full>
          <<<dim3((batch + 255) / 256, (m + 255) / 256), 1024, 0, stream>>>(
              w, static_cast<const short*>(x), y, batch, m, k, gate);
    };
    if (Epilogue == HalfEpilogue::kSwiGLU || m >= 4096 ||
        (LargeKvTile && m >= 1024 && batch >= 1024)) {
      // Only Q4/Q5 benefit from removing the tail checks on complete tiles.
      // Other formats retain the general kernel's better register schedule.
      if constexpr (Type == gufo::core::GgmlType::kQ4_K ||
                    Type == gufo::core::GgmlType::kQ5_K) {
        if (m % 256 == 0 && batch % 256 == 0) {
          large.template operator()<true>();
          return;
        }
      }
      large.template operator()<false>();
    } else if constexpr (Epilogue != HalfEpilogue::kSwiGLU) {
      if (m >= 1024) {
        HalfPrefillGemmKernel<64, 128, 2, 4, Type, Epilogue>
            <<<dim3((batch + 127) / 128, (m + 63) / 64), 256, 0, stream>>>(
                w, static_cast<const short*>(x), y, batch, m, k, nullptr);
      } else {
        HalfPrefillGemmKernel<32, 64, 2, 2, Type, Epilogue>
            <<<dim3((batch + 63) / 64, (m + 31) / 32), 128, 0, stream>>>(
                w, static_cast<const short*>(x), y, batch, m, k, nullptr);
      }
    }
  };
  using gufo::core::GgmlType;
  switch (type) {
    case GgmlType::kQ4_K:
      launch.template operator()<GgmlType::kQ4_K>();
      break;
    case GgmlType::kQ5_K:
      launch.template operator()<GgmlType::kQ5_K>();
      break;
    case GgmlType::kQ6_K:
      launch.template operator()<GgmlType::kQ6_K>();
      break;
    case GgmlType::kIQ4_XS:
      launch.template operator()<GgmlType::kIQ4_XS>();
      break;
    case GgmlType::kIQ4_NL:
      launch.template operator()<GgmlType::kIQ4_NL>();
      break;
    case GgmlType::kQ8_0:
      launch.template operator()<GgmlType::kQ8_0>();
      break;
    case GgmlType::kQ3_K:
      launch.template operator()<GgmlType::kQ3_K>();
      break;
    case GgmlType::kIQ3_S:
      launch.template operator()<GgmlType::kIQ3_S>();
      break;
    case GgmlType::kIQ3_XXS:
      launch.template operator()<GgmlType::kIQ3_XXS>();
      break;
    case GgmlType::kIQ2_XS:
      launch.template operator()<GgmlType::kIQ2_XS>();
      break;
    case GgmlType::kIQ2_S:
      launch.template operator()<GgmlType::kIQ2_S>();
      break;
    case GgmlType::kIQ2_XXS:
      launch.template operator()<GgmlType::kIQ2_XXS>();
      break;
    case GgmlType::kQ2_K:
      launch.template operator()<GgmlType::kQ2_K>();
      break;
    default:
      throw std::runtime_error("unexpected half prefill weight format");
  }
}
void LaunchBatchedQuantGEMMFp16(gufo::core::GgmlType type, const void* w,
                                const void* x, float* y, std::size_t batch,
                                std::size_t m, std::size_t k,
                                hipStream_t stream) {
  DirectGemm<HalfEpilogue::kStore>(type, w, x, y, batch, m, k, nullptr, stream);
}
void LaunchBatchedQuantGEMMSwiGLUFp16(gufo::core::GgmlType type, const void* w,
                                      const void* x, const float* gate, void* y,
                                      std::size_t batch, std::size_t m,
                                      std::size_t k, hipStream_t stream) {
  DirectGemm<HalfEpilogue::kSwiGLU>(type, w, x, static_cast<float*>(y), batch,
                                    m, k, gate, stream);
}

void LaunchBatchedQuantGEMMResidualFp16(gufo::core::GgmlType type,
                                        const void* w, const void* x,
                                        float* residual, std::size_t batch,
                                        std::size_t m, std::size_t k,
                                        hipStream_t stream) {
  DirectGemm<HalfEpilogue::kResidual>(type, w, x, residual, batch, m, k,
                                      nullptr, stream);
}

bool TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
    gufo::core::GgmlType gate_type, gufo::core::GgmlType up_type,
    const void* gate_weights, const void* up_weights, const void* x, void* y,
    std::size_t batch, std::size_t m, std::size_t k, hipStream_t stream) {
  const auto launch =
      [&]<gufo::core::GgmlType Type, gufo::core::GgmlType UpType = Type>() {
        const auto tile = [&]<bool Full>() {
          HalfPrefillGemmKernel<256, 256, 8, 4, Type, HalfEpilogue::kGateUp,
                                Full, UpType>
              <<<dim3((batch + 255) / 256, (m + 127) / 128), 1024, 0, stream>>>(
                  gate_weights, static_cast<const short*>(x),
                  static_cast<float*>(y), batch, m, k, nullptr, up_weights);
        };
        if constexpr ((Type == gufo::core::GgmlType::kQ4_K ||
                       Type == gufo::core::GgmlType::kQ5_K) &&
                      (UpType == gufo::core::GgmlType::kQ4_K ||
                       UpType == gufo::core::GgmlType::kQ5_K)) {
          if (m % 128 == 0 && batch % 256 == 0) {
            tile.template operator()<true>();
            return;
          }
        }
        tile.template operator()<false>();
      };
  using gufo::core::GgmlType;
  // Mixed pairs keep independent row strides and decoders. Qualify each
  // pairing: a shared input alone does not guarantee a faster kernel.
  if (gate_type != up_type) {
    if (gate_type == GgmlType::kQ4_K && up_type == GgmlType::kQ5_K) {
      launch.template operator()<GgmlType::kQ4_K, GgmlType::kQ5_K>();
    } else if (gate_type == GgmlType::kIQ4_XS && up_type == GgmlType::kQ5_K) {
      launch.template operator()<GgmlType::kIQ4_XS, GgmlType::kQ5_K>();
    } else if (gate_type == GgmlType::kQ4_K && up_type == GgmlType::kIQ4_XS) {
      launch.template operator()<GgmlType::kQ4_K, GgmlType::kIQ4_XS>();
    } else if (gate_type == GgmlType::kIQ4_XS && up_type == GgmlType::kQ4_K) {
      launch.template operator()<GgmlType::kIQ4_XS, GgmlType::kQ4_K>();
    } else if (gate_type == GgmlType::kQ6_K && up_type == GgmlType::kQ5_K) {
      launch.template operator()<GgmlType::kQ6_K, GgmlType::kQ5_K>();
    } else if (gate_type == GgmlType::kIQ4_XS && up_type == GgmlType::kQ3_K) {
      launch.template operator()<GgmlType::kIQ4_XS, GgmlType::kQ3_K>();
    } else if (gate_type == GgmlType::kQ3_K && up_type == GgmlType::kIQ4_XS) {
      launch.template operator()<GgmlType::kQ3_K, GgmlType::kIQ4_XS>();
    } else if (gate_type == GgmlType::kIQ4_NL && up_type == GgmlType::kIQ4_XS) {
      launch.template operator()<GgmlType::kIQ4_NL, GgmlType::kIQ4_XS>();
    } else if (gate_type == GgmlType::kIQ4_NL && up_type == GgmlType::kQ5_K) {
      launch.template operator()<GgmlType::kIQ4_NL, GgmlType::kQ5_K>();
    } else if (gate_type == GgmlType::kQ5_K && up_type == GgmlType::kIQ4_NL) {
      launch.template operator()<GgmlType::kQ5_K, GgmlType::kIQ4_NL>();
    } else {
      return false;
    }
    return true;
  }
  switch (gate_type) {
    case GgmlType::kQ4_K:
      launch.template operator()<GgmlType::kQ4_K>();
      return true;
    case GgmlType::kQ5_K:
      launch.template operator()<GgmlType::kQ5_K>();
      return true;
    case GgmlType::kQ6_K:
      launch.template operator()<GgmlType::kQ6_K>();
      return true;
    case GgmlType::kIQ4_XS:
      launch.template operator()<GgmlType::kIQ4_XS>();
      return true;
    default:
      return false;
  }
}

}  // namespace gufo::hip
#endif
