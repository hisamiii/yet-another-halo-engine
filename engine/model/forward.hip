// Single-chunk text prefill for Qwen3.8-27B on gfx1151, built from the
// per-stage GPU kernels that each have an independent numeric check. The
// layer sequence mirrors the reference engine's fp16 prefill path exactly:
// embed, then for each main block a pre-norm, a token mixer (full attention or
// Gated DeltaNet), and a SwiGLU FFN, all accumulating into one FP32 residual.
//
// This file is HIP-only. It includes the op declarations and expects the
// kernel translation units to be compiled and linked alongside it, matching
// how the reference engine builds them.
#ifndef YAH_MODEL_FORWARD_HIP_
#define YAH_MODEL_FORWARD_HIP_

#ifndef ENGINE_ENABLE_HIP
#error "model/forward.hip requires ENGINE_ENABLE_HIP"
#endif

#include <hip/hip_runtime.h>

#include <sys/mman.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "model/weights.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/hip/ops/token.hpp"

namespace yah::model {

// The ported kernels keep the reference engine's namespaces, so the engine's
// own enums have to be bridged at the call sites. Both enums list the GGML
// type codes in the same order, so the conversion is the underlying value.
using namespace gufo::hip;  // NOLINT(google-build-using-namespace)

[[nodiscard]] inline gufo::core::GgmlType Gt(core::GgmlType type) {
  return static_cast<gufo::core::GgmlType>(
      static_cast<std::uint32_t>(type));
}

namespace {

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("yah forward: ") + what + ": " +
                             hipGetErrorString(status));
  }
}

// One allocation for the whole forward pass. Every buffer is sized for the
// single batch the run is constructed with, so there is no per-call memory
// traffic and no arena to grow.
struct Buffers {
  float* hidden{nullptr};
  void* scratch_f16{nullptr};
  float* normed{nullptr};
  float* logits{nullptr};
  float* ssm_qkv{nullptr};
  float* ssm_gate{nullptr};
  float* ssm_out{nullptr};
  float* q{nullptr};
  float* k{nullptr};
  float* v{nullptr};
  float* alpha{nullptr};
  float* beta{nullptr};
  float* conv_out{nullptr};
  float* ffn_gate{nullptr};
  float* ffn_up{nullptr};
  std::uint16_t* kv_f16{nullptr};
  std::uint32_t* tokens{nullptr};
  std::uint32_t* argmax{nullptr};
  float* conv_state{nullptr};
  float* deltanet_state{nullptr};
  float* kq_scales{nullptr};
  float* alpha_beta{nullptr};
};

}  // namespace

// A device-visible view of the GGUF tensor region. The file mapping is
// read-only and is not registered with HIP, so a kernel that dereferences it
// stalls the GPU instead of faulting cleanly. The reference engine copies the
// tensor region into an anonymous mapping and registers that copy; the same
// approach is used here, and the weight table is rebased by the pointer delta.
class HipWeightRegion {
 public:
  explicit HipWeightRegion(const core::Gguf& gguf) {
    const auto* source = gguf.tensor_data_base();
    size_ = gguf.tensor_data_size();
    if (size_ == 0) {
      throw std::runtime_error("yah weights: empty tensor region");
    }
    void* copy = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (copy == MAP_FAILED) {
      throw std::runtime_error("yah weights: anonymous mapping failed");
    }
    host_ = static_cast<std::uint8_t*>(copy);
    (void)::madvise(host_, size_, MADV_HUGEPAGE);
    std::memcpy(host_, source, size_);
    if (hipHostRegister(host_, size_,
                        hipHostRegisterMapped | hipHostRegisterReadOnly) !=
        hipSuccess) {
      ::munmap(host_, size_);
      host_ = nullptr;
      throw std::runtime_error("yah weights: hipHostRegister failed");
    }
    registered_ = true;
    void* device = nullptr;
    if (hipHostGetDevicePointer(&device, host_, 0) != hipSuccess) {
      throw std::runtime_error("yah weights: hipHostGetDevicePointer failed");
    }
    device_ = static_cast<std::uint8_t*>(device);
  }

  ~HipWeightRegion() {
    if (registered_) (void)hipHostUnregister(host_);
    if (host_ != nullptr) (void)::munmap(host_, size_);
  }
  HipWeightRegion(const HipWeightRegion&) = delete;
  HipWeightRegion& operator=(const HipWeightRegion&) = delete;

  [[nodiscard]] std::uint8_t* device_base() const { return device_; }
  [[nodiscard]] std::ptrdiff_t delta(const core::Gguf& gguf) const {
    return device_ - gguf.tensor_data_base();
  }

 private:
  std::uint8_t* host_{nullptr};
  std::uint8_t* device_{nullptr};
  std::size_t size_{0};
  bool registered_{false};
};

class Forward {
 public:
  Forward(const core::Qwen35Config& config, const Qwen35Weights& weights,
          std::uint32_t batch, std::uint32_t max_context)
      : config_(config), weights_(weights), max_batch_(batch),
        max_context_(max_context) {
    if (batch == 0 || max_context == 0) {
      throw std::runtime_error("yah forward: zero batch or context");
    }
    if (config_.full_attention_interval == 0 ||
        config_.main_block_count() % config_.full_attention_interval != 0) {
      throw std::runtime_error("yah forward: bad attention interval");
    }
    const std::uint32_t layers = config_.main_block_count();
    if (weights_.layers.size() < layers) {
      throw std::runtime_error("yah forward: weight table too short");
    }
    hidden_ = config_.embedding_length;
    ffn_ = config_.feed_forward_length;
    attn_ = config_.head_count * config_.key_length;
    q_proj_ = 2 * attn_;
    kv_ = config_.head_count_kv * config_.key_length;
    ssm_qkv_ = 2 * config_.ssm_group_count * config_.ssm_state_size +
               config_.ssm_inner_size;
    ssm_inner_ = config_.ssm_inner_size;
    ts_rank_ = config_.ssm_time_step_rank;
    value_ = ssm_inner_ / ts_rank_;
    attn_layers_ = config_.AttentionLayers();
    recurrent_layers_ = config_.RecurrentLayers();
    vocab_ = static_cast<std::uint32_t>(weights_.output.rows() != 0
                                            ? weights_.output.rows()
                                            : weights_.token_embd.rows());
    Allocate();
    Reset();
  }

  ~Forward() { Free(); }
  Forward(const Forward&) = delete;
  Forward& operator=(const Forward&) = delete;

  // Zeroes the recurrent and convolution states and the KV cache. Call
  // before a fresh sequence.
  void Reset() {
    Check(hipMemsetAsync(buffers_.deltanet_state, 0,
                         static_cast<std::size_t>(recurrent_layers_) *
                             ts_rank_ * config_.ssm_state_size * value_ *
                             sizeof(float),
                         stream_), "memset deltanet");
    Check(hipMemsetAsync(buffers_.conv_state, 0,
                         static_cast<std::size_t>(recurrent_layers_) *
                             ssm_qkv_ * config_.ssm_conv_kernel * sizeof(float),
                         stream_), "memset conv");
    const std::size_t plane = KvPlaneElements();
    Check(hipMemsetAsync(buffers_.kv_f16, 0, 2 * plane * sizeof(std::uint16_t),
                         stream_), "memset kv");
    Check(hipStreamSynchronize(stream_), "sync reset");
  }

  // Runs the main stack over `batch` tokens starting at `start_pos` and
  // returns the argmax token of the final position.
  std::uint32_t Prefill(const std::uint32_t* host_tokens,
                        std::uint32_t batch, std::uint32_t start_pos) {
    if (batch == 0 || batch > max_batch_) {
      throw std::runtime_error("yah forward: chunk exceeds max batch");
    }
    if (start_pos + batch > max_context_ && !config_.IsFullAttention(0)) {
      // The KV cache bounds only the full-attention layers.
    }
    if (start_pos + batch > max_context_) {
      throw std::runtime_error("yah forward: batch exceeds context");
    }
    Check(hipMemcpyAsync(buffers_.tokens, host_tokens,
                         batch * sizeof(std::uint32_t),
                         hipMemcpyHostToDevice, stream_), "copy tokens");

    const bool sync_stages = std::getenv("YAH_SYNC_STAGES") != nullptr;
    const auto stage = [&](const char* what, std::uint32_t l) {
      if (!sync_stages) return;
      Check(hipStreamSynchronize(stream_), what);
      std::fprintf(stderr, "yah forward: layer %u stage %s ok\n", l, what);
      std::fflush(stderr);
    };

    LaunchBatchedEmbeddingLookup(
        weights_.token_embd.data, Gt(weights_.token_embd.type), buffers_.tokens,
        buffers_.hidden, batch, hidden_, stream_);
    stage("embed", 0);

    // Debug staging: YAH_LAYER_LIMIT bounds the stack and YAH_SYNC_LAYERS
    // synchronizes after every block, so a fault localizes to one layer type
    // without a full-stack hang. Both are inert when unset.
    const char* limit_env = std::getenv("YAH_LAYER_LIMIT");
    const std::uint32_t layer_limit =
        limit_env == nullptr
            ? config_.main_block_count()
            : std::min(config_.main_block_count(),
                       static_cast<std::uint32_t>(std::stoul(limit_env)));
    const bool sync_layers = std::getenv("YAH_SYNC_LAYERS") != nullptr;
    for (std::uint32_t l = 0; l < layer_limit; ++l) {
      const Qwen35Layer& layer = weights_.layers[l];
      const bool full = config_.IsFullAttention(l);
      if (full != layer.is_full_attention) {
        throw std::runtime_error("yah forward: layer type disagrees");
      }

      LaunchBatchedRMSNormFp16(
          buffers_.hidden, /*residual=*/nullptr,
          reinterpret_cast<const float*>(layer.attn_norm.data),
          /*sum_out=*/nullptr, buffers_.scratch_f16, batch, hidden_,
          config_.rms_eps, stream_);
      stage("norm", l);

      if (full) {
        const std::uint32_t attn_idx = l / config_.full_attention_interval;
        LaunchBatchedQuantGEMMFp16(Gt(layer.attn_q.type), layer.attn_q.data,
                                   buffers_.scratch_f16, buffers_.ssm_qkv,
                                   batch, q_proj_, hidden_, stream_);
        LaunchBatchedQuantGEMMFp16(Gt(layer.attn_k.type), layer.attn_k.data,
                                   buffers_.scratch_f16, buffers_.k, batch,
                                   kv_, hidden_, stream_);
        LaunchBatchedQuantGEMMFp16(Gt(layer.attn_v.type), layer.attn_v.data,
                                   buffers_.scratch_f16, buffers_.v, batch,
                                   kv_, hidden_, stream_);
        LaunchBatchedUnpackQG(buffers_.ssm_qkv, buffers_.q,
                              buffers_.ssm_gate, batch, config_.head_count,
                              config_.key_length, stream_);
        const std::size_t plane = KvPlaneElements();
        LaunchBatchedFusedQKNormRoPEKvWrite(
            buffers_.q, buffers_.k, buffers_.v,
            reinterpret_cast<const float*>(layer.attn_q_norm.data),
            reinterpret_cast<const float*>(layer.attn_k_norm.data),
            buffers_.q, buffers_.k, /*k_cache=*/nullptr, /*v_cache=*/nullptr,
            buffers_.kv_f16, buffers_.kv_f16 + plane, attn_idx, start_pos,
            batch, max_context_, config_.head_count,
            config_.head_count_kv, config_.key_length,
            config_.rope_dimension_count, config_.rope_freq_base,
            config_.rms_eps, stream_);
        const bool wmma = LaunchQwenWmmaAttention(
            buffers_.q, buffers_.k, buffers_.v, buffers_.ssm_gate,
            /*k_cache=*/nullptr, /*v_cache=*/nullptr, buffers_.kv_f16,
            buffers_.kv_f16 + plane, buffers_.ssm_out, attn_idx, start_pos,
            batch, max_context_, config_.head_count, config_.head_count_kv,
            config_.key_length, stream_, /*lse_out=*/nullptr,
            /*key_begin=*/0, /*skip_kv_write=*/true);
        if (!wmma) {
          LaunchBatchedAttention(
              buffers_.q, buffers_.k, buffers_.v, buffers_.ssm_gate,
              /*k_cache=*/nullptr, /*v_cache=*/nullptr, buffers_.kv_f16,
              buffers_.kv_f16 + plane, buffers_.ssm_out, attn_idx, start_pos,
              batch, max_context_, config_.head_count,
              config_.head_count_kv, config_.key_length, stream_,
              /*skip_kv_write=*/true);
        }
        LaunchFloatToFp16(buffers_.ssm_out, buffers_.scratch_f16,
                          static_cast<std::size_t>(batch) * attn_,
                          stream_);
        LaunchBatchedQuantGEMMResidualFp16(
            Gt(layer.attn_output.type), layer.attn_output.data,
            buffers_.scratch_f16, buffers_.hidden, batch, hidden_, attn_,
            stream_);
        stage("attn", l);
      } else {
        const std::uint32_t ssm_idx = config_.SsmLayerIndex(l);
        LaunchBatchedQuantGEMMFp16(Gt(layer.attn_qkv.type), layer.attn_qkv.data,
                                   buffers_.scratch_f16, buffers_.ssm_qkv,
                                   batch, ssm_qkv_, hidden_, stream_);
        LaunchBatchedQuantGEMMFp16(Gt(layer.attn_gate.type),
                                   layer.attn_gate.data, buffers_.scratch_f16,
                                   buffers_.ssm_gate, batch, ssm_inner_,
                                   hidden_, stream_);
        LaunchBatchedQuantGEMMFp16(Gt(layer.ssm_alpha.type),
                                   layer.ssm_alpha.data, buffers_.scratch_f16,
                                   buffers_.alpha, batch, ts_rank_, hidden_,
                                   stream_);
        LaunchBatchedQuantGEMMFp16(Gt(layer.ssm_beta.type),
                                   layer.ssm_beta.data, buffers_.scratch_f16,
                                   buffers_.beta, batch, ts_rank_, hidden_,
                                   stream_);
        stage("ssm_proj", l);
        // The row-split recurrence fuses the norm, gate, and FP16 conversion
        // and is the path the reference runs by default. The unfused variant
        // the SSM check verified directly stays available behind
        // YAH_SSM_BASELINE as the independent cross-check.
        if (std::getenv("YAH_SSM_BASELINE") == nullptr) {
          LaunchBatchedSSMConvRecurrenceRowSplit(
              buffers_.ssm_qkv,
              reinterpret_cast<const float*>(layer.ssm_conv1d.data),
              buffers_.conv_state, buffers_.conv_out, buffers_.deltanet_state,
              buffers_.alpha, buffers_.beta,
              reinterpret_cast<const float*>(layer.ssm_a.data),
              reinterpret_cast<const float*>(layer.ssm_dt.data),
              reinterpret_cast<const float*>(layer.ssm_norm.data),
              buffers_.ssm_gate, buffers_.ssm_out, /*q8_out=*/nullptr,
              buffers_.kq_scales, buffers_.alpha_beta, ssm_idx, batch,
              ssm_qkv_, config_.ssm_group_count, ts_rank_,
              config_.ssm_state_size, value_, stream_,
              QwenRecurrentStateStorage::kFp32,
              /*fp16_out=*/buffers_.scratch_f16);
        } else {
          LaunchBatchedSSMConvRecurrence(
              buffers_.ssm_qkv,
              reinterpret_cast<const float*>(layer.ssm_conv1d.data),
              buffers_.conv_state, buffers_.conv_out, buffers_.deltanet_state,
              buffers_.alpha, buffers_.beta,
              reinterpret_cast<const float*>(layer.ssm_a.data),
              reinterpret_cast<const float*>(layer.ssm_dt.data),
              reinterpret_cast<const float*>(layer.ssm_norm.data),
              buffers_.ssm_gate, buffers_.ssm_out, ssm_idx, batch, ssm_qkv_,
              config_.ssm_group_count, ts_rank_, config_.ssm_state_size,
              value_, stream_, QwenRecurrentStateStorage::kFp32);
          LaunchFloatToFp16(buffers_.ssm_out, buffers_.scratch_f16,
                            static_cast<std::size_t>(batch) * ssm_inner_,
                            stream_);
        }
        LaunchBatchedQuantGEMMResidualFp16(
            Gt(layer.ssm_out.type), layer.ssm_out.data, buffers_.scratch_f16,
            buffers_.hidden, batch, hidden_, ssm_inner_, stream_);
        stage("ssm", l);
      }
      stage("mixer", l);

      LaunchBatchedRMSNormFp16(
          buffers_.hidden, /*residual=*/nullptr,
          reinterpret_cast<const float*>(layer.post_attention_norm.data),
          /*sum_out=*/nullptr, buffers_.scratch_f16, batch, hidden_,
          config_.rms_eps, stream_);
      const bool paired = TryLaunchBatchedDualQuantGEMMSwiGLUFp16(
          Gt(layer.ffn_gate.type), Gt(layer.ffn_up.type), layer.ffn_gate.data,
          layer.ffn_up.data, buffers_.scratch_f16, buffers_.ffn_up, batch,
          ffn_, hidden_, stream_);
      if (!paired) {
        LaunchBatchedQuantGEMMFp16(Gt(layer.ffn_gate.type), layer.ffn_gate.data,
                                   buffers_.scratch_f16, buffers_.ffn_gate,
                                   batch, ffn_, hidden_, stream_);
        LaunchBatchedQuantGEMMSwiGLUFp16(
            Gt(layer.ffn_up.type), layer.ffn_up.data, buffers_.scratch_f16,
            buffers_.ffn_gate, buffers_.ffn_up, batch, ffn_, hidden_,
            stream_);
      }
      stage("gate_up", l);
      LaunchBatchedQuantGEMMResidualFp16(
          Gt(layer.ffn_down.type), layer.ffn_down.data, buffers_.ffn_up,
          buffers_.hidden, batch, hidden_, ffn_, stream_);
      if (sync_layers) {
        Check(hipStreamSynchronize(stream_), "layer sync");
        std::fprintf(stderr, "yah forward: layer %u (%s) ok\n", l,
                     full ? "attention" : "deltanet");
      }
    }

    const float* final_hidden =
        buffers_.hidden + (static_cast<std::size_t>(batch - 1) * hidden_);
    LaunchRMSNorm(final_hidden,
                  reinterpret_cast<const float*>(weights_.output_norm.data),
                  buffers_.normed, hidden_, config_.rms_eps, stream_);
    if (weights_.output.present) {
      LaunchGEMV(weights_.output.data, Gt(weights_.output.type), buffers_.normed,
                 buffers_.logits, vocab_, hidden_, stream_);
    } else {
      LaunchGEMV(weights_.token_embd.data, Gt(weights_.token_embd.type),
                 buffers_.normed, buffers_.logits, vocab_, hidden_, stream_);
    }
    LaunchGPUArgmax(buffers_.logits, buffers_.argmax, vocab_, stream_);
    std::uint32_t out = 0;
    Check(hipMemcpyAsync(&out, buffers_.argmax, sizeof(std::uint32_t),
                         hipMemcpyDeviceToHost, stream_), "copy argmax");
    Check(hipStreamSynchronize(stream_), "sync prefill");
    return out;
  }


  // One decode step. The math is the same as Prefill with a batch of one,
  // but the projections are GEMVs and the attention and recurrence are the
  // single-token kernels, which is where the reference decode path gets its
  // speed. The KV cache and recurrent state are the ones Prefill left.
  std::uint32_t Decode(std::uint32_t token, std::uint32_t pos) {
    Check(hipMemcpyAsync(buffers_.tokens, &token, sizeof(token),
                         hipMemcpyHostToDevice, stream_),
          "copy decode token");
    LaunchEmbeddingLookup(
        weights_.token_embd.data, Gt(weights_.token_embd.type),
        buffers_.tokens, buffers_.hidden, hidden_, stream_);
    for (std::uint32_t l = 0; l < config_.main_block_count(); ++l) {
      const Qwen35Layer& layer = weights_.layers[l];
      const bool full = config_.IsFullAttention(l);
      LaunchRMSNorm(buffers_.hidden,
                    reinterpret_cast<const float*>(layer.attn_norm.data),
                    buffers_.normed, hidden_, config_.rms_eps, stream_);
      if (full) {
        const std::uint32_t attn_idx = l / config_.full_attention_interval;
        LaunchFusedQKVProjections(
            layer.attn_q.data, Gt(layer.attn_q.type), layer.attn_k.data,
            Gt(layer.attn_k.type), layer.attn_v.data, Gt(layer.attn_v.type),
            buffers_.normed, buffers_.ssm_qkv, buffers_.k, buffers_.v, q_proj_,
            kv_, hidden_, stream_);
        LaunchUnpackQG(buffers_.ssm_qkv, buffers_.q, buffers_.ssm_gate,
                       config_.head_count, config_.key_length, stream_);
        const std::size_t plane = KvPlaneElements();
        LaunchBatchedFusedQKNormRoPEKvWrite(
            buffers_.q, buffers_.k, buffers_.v,
            reinterpret_cast<const float*>(layer.attn_q_norm.data),
            reinterpret_cast<const float*>(layer.attn_k_norm.data),
            buffers_.q, buffers_.k, /*k_cache=*/nullptr, /*v_cache=*/nullptr,
            buffers_.kv_f16, buffers_.kv_f16 + plane, attn_idx, pos,
            /*batch=*/1, max_context_, config_.head_count,
            config_.head_count_kv, config_.key_length,
            config_.rope_dimension_count, config_.rope_freq_base,
            config_.rms_eps, stream_);
        LaunchAttention(
            buffers_.q, buffers_.k, buffers_.v, buffers_.ssm_gate,
            /*k_cache=*/nullptr, /*v_cache=*/nullptr, buffers_.kv_f16,
            buffers_.kv_f16 + plane, buffers_.ssm_out, attn_idx, pos,
            max_context_, config_.head_count, config_.head_count_kv,
            config_.key_length, stream_, /*split_k_scratch=*/nullptr,
            /*skip_kv_write=*/true);
        LaunchGEMV(layer.attn_output.data, Gt(layer.attn_output.type),
                   buffers_.ssm_out, buffers_.normed, hidden_, attn_, stream_);
      } else {
        const std::uint32_t ssm_idx = config_.SsmLayerIndex(l);
        LaunchFusedSSMInputProjections(
            layer.attn_qkv.data, Gt(layer.attn_qkv.type), layer.attn_gate.data,
            Gt(layer.attn_gate.type), layer.ssm_alpha.data,
            Gt(layer.ssm_alpha.type), layer.ssm_beta.data,
            Gt(layer.ssm_beta.type), buffers_.normed, buffers_.ssm_qkv,
            buffers_.ssm_gate, buffers_.alpha, buffers_.beta, hidden_,
            ssm_qkv_, ssm_inner_, ts_rank_, stream_);
        LaunchSSMConvRecurrence(
            buffers_.ssm_qkv,
            reinterpret_cast<const float*>(layer.ssm_conv1d.data),
            buffers_.conv_state, buffers_.conv_out, buffers_.deltanet_state,
            buffers_.alpha, buffers_.beta,
            reinterpret_cast<const float*>(layer.ssm_a.data),
            reinterpret_cast<const float*>(layer.ssm_dt.data),
            reinterpret_cast<const float*>(layer.ssm_norm.data),
            buffers_.ssm_gate, buffers_.ssm_out, ssm_idx, ssm_qkv_,
            config_.ssm_group_count, ts_rank_, config_.ssm_state_size,
            value_, stream_, SsmReplayCapture{},
            QwenRecurrentStateStorage::kFp32);
        LaunchGEMV(layer.ssm_out.data, Gt(layer.ssm_out.type),
                   buffers_.ssm_out, buffers_.normed, hidden_, ssm_inner_,
                   stream_);
      }
      LaunchResidualAdd(buffers_.hidden, buffers_.normed, buffers_.hidden,
                        hidden_, stream_);
      LaunchRMSNorm(
          buffers_.hidden,
          reinterpret_cast<const float*>(layer.post_attention_norm.data),
          buffers_.normed, hidden_, config_.rms_eps, stream_);
      LaunchFusedSwiGLUGEMV(layer.ffn_gate.data, Gt(layer.ffn_gate.type),
                            layer.ffn_up.data, Gt(layer.ffn_up.type),
                            buffers_.normed, buffers_.ffn_gate, ffn_,
                            hidden_, stream_);
      LaunchGEMV(layer.ffn_down.data, Gt(layer.ffn_down.type),
                 buffers_.ffn_gate, buffers_.normed, hidden_, ffn_, stream_);
      LaunchResidualAdd(buffers_.hidden, buffers_.normed, buffers_.hidden,
                        hidden_, stream_);
    }
    LaunchRMSNorm(buffers_.hidden,
                  reinterpret_cast<const float*>(weights_.output_norm.data),
                  buffers_.normed, hidden_, config_.rms_eps, stream_);
    if (weights_.output.present) {
      LaunchGEMV(weights_.output.data, Gt(weights_.output.type),
                 buffers_.normed, buffers_.logits, vocab_, hidden_, stream_);
    } else {
      LaunchGEMV(weights_.token_embd.data, Gt(weights_.token_embd.type),
                 buffers_.normed, buffers_.logits, vocab_, hidden_, stream_);
    }
    LaunchGPUArgmax(buffers_.logits, buffers_.argmax, vocab_, stream_);
    std::uint32_t out = 0;
    Check(hipMemcpyAsync(&out, buffers_.argmax, sizeof(std::uint32_t),
                         hipMemcpyDeviceToHost, stream_),
          "copy decode argmax");
    Check(hipStreamSynchronize(stream_), "sync decode");
    return out;
  }
  // Copies the fp32 logits of the last position to the host.
  void CopyLogits(float* out) {
    Check(hipMemcpyAsync(out, buffers_.logits, vocab_ * sizeof(float),
                         hipMemcpyDeviceToHost, stream_), "copy logits");
    Check(hipStreamSynchronize(stream_), "sync logits");
  }

  [[nodiscard]] std::uint32_t vocab_size() const { return vocab_; }
  [[nodiscard]] std::uint32_t max_batch() const { return max_batch_; }

 private:
  [[nodiscard]] std::size_t KvPlaneElements() const {
    return static_cast<std::size_t>(attn_layers_) * config_.head_count_kv *
           max_context_ * config_.key_length;
  }

  void Allocate() {
    const std::size_t batch = max_batch_;
    hipStreamCreate(&stream_);
    auto alloc = [&](void** p, std::size_t bytes, const char* what) {
      Check(hipMalloc(p, bytes), what);
    };
    alloc(reinterpret_cast<void**>(&buffers_.hidden),
          batch * hidden_ * sizeof(float), "hidden");
    alloc(&buffers_.scratch_f16, batch * ffn_ * sizeof(std::uint16_t),
          "scratch_f16");
    alloc(reinterpret_cast<void**>(&buffers_.normed),
          hidden_ * sizeof(float), "normed");
    alloc(reinterpret_cast<void**>(&buffers_.logits),
          vocab_ * sizeof(float), "logits");
    alloc(reinterpret_cast<void**>(&buffers_.ssm_qkv),
          batch * q_proj_ * sizeof(float), "ssm_qkv");
    const std::size_t gate_width = std::max<std::size_t>(attn_, ssm_inner_);
    alloc(reinterpret_cast<void**>(&buffers_.ssm_gate),
          batch * gate_width * sizeof(float), "ssm_gate");
    alloc(reinterpret_cast<void**>(&buffers_.ssm_out),
          batch * gate_width * sizeof(float), "ssm_out");
    alloc(reinterpret_cast<void**>(&buffers_.q), batch * attn_ * sizeof(float),
          "q");
    alloc(reinterpret_cast<void**>(&buffers_.k), batch * kv_ * sizeof(float),
          "k");
    alloc(reinterpret_cast<void**>(&buffers_.v), batch * kv_ * sizeof(float),
          "v");
    alloc(reinterpret_cast<void**>(&buffers_.alpha),
          batch * ts_rank_ * sizeof(float), "alpha");
    alloc(reinterpret_cast<void**>(&buffers_.beta),
          batch * ts_rank_ * sizeof(float), "beta");
    alloc(reinterpret_cast<void**>(&buffers_.conv_out),
          batch * ssm_qkv_ * sizeof(float), "conv_out");
    alloc(reinterpret_cast<void**>(&buffers_.ffn_gate),
          batch * ffn_ * sizeof(float), "ffn_gate");
    alloc(reinterpret_cast<void**>(&buffers_.ffn_up),
          batch * ffn_ * sizeof(float), "ffn_up");
    const std::size_t plane = KvPlaneElements();
    alloc(reinterpret_cast<void**>(&buffers_.kv_f16),
          2 * plane * sizeof(std::uint16_t), "kv_f16");
    alloc(reinterpret_cast<void**>(&buffers_.tokens),
          batch * sizeof(std::uint32_t), "tokens");
    alloc(reinterpret_cast<void**>(&buffers_.argmax), sizeof(std::uint32_t),
          "argmax");
    alloc(reinterpret_cast<void**>(&buffers_.conv_state),
          static_cast<std::size_t>(recurrent_layers_) * ssm_qkv_ *
              config_.ssm_conv_kernel * sizeof(float),
          "conv_state");
    alloc(reinterpret_cast<void**>(&buffers_.deltanet_state),
          static_cast<std::size_t>(recurrent_layers_) * ts_rank_ *
              config_.ssm_state_size * value_ * sizeof(float),
          "deltanet_state");
    alloc(reinterpret_cast<void**>(&buffers_.kq_scales),
          batch * config_.ssm_group_count * 3 * sizeof(float), "kq_scales");
    alloc(reinterpret_cast<void**>(&buffers_.alpha_beta),
          batch * ts_rank_ * 2 * sizeof(float), "alpha_beta");
  }

  void Free() {
    hipFree(buffers_.hidden);
    hipFree(buffers_.scratch_f16);
    hipFree(buffers_.normed);
    hipFree(buffers_.logits);
    hipFree(buffers_.ssm_qkv);
    hipFree(buffers_.ssm_gate);
    hipFree(buffers_.ssm_out);
    hipFree(buffers_.q);
    hipFree(buffers_.k);
    hipFree(buffers_.v);
    hipFree(buffers_.alpha);
    hipFree(buffers_.beta);
    hipFree(buffers_.conv_out);
    hipFree(buffers_.ffn_gate);
    hipFree(buffers_.ffn_up);
    hipFree(buffers_.kv_f16);
    hipFree(buffers_.tokens);
    hipFree(buffers_.argmax);
    hipFree(buffers_.conv_state);
    hipFree(buffers_.deltanet_state);
    hipFree(buffers_.kq_scales);
    hipFree(buffers_.alpha_beta);
    if (stream_ != nullptr) hipStreamDestroy(stream_);
  }

  const core::Qwen35Config& config_;
  const Qwen35Weights& weights_;
  std::uint32_t max_batch_{0};
  std::uint32_t max_context_{0};
  std::uint32_t hidden_{0};
  std::uint32_t ffn_{0};
  std::uint32_t attn_{0};
  std::uint32_t q_proj_{0};
  std::uint32_t kv_{0};
  std::uint32_t ssm_qkv_{0};
  std::uint32_t ssm_inner_{0};
  std::uint32_t ts_rank_{0};
  std::uint32_t value_{0};
  std::uint32_t attn_layers_{0};
  std::uint32_t recurrent_layers_{0};
  std::uint32_t vocab_{0};
  hipStream_t stream_{nullptr};
  Buffers buffers_;
};

}  // namespace yah::model

#endif  // YAH_MODEL_FORWARD_HIP_
