#include <cstdio>
#include <exception>
#include <string>

#include "core/config.hpp"
#include "core/gguf.hpp"
#include "model/weights.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: yah-weights <model.gguf>\n");
    return 2;
  }
  try {
    auto gguf = yah::core::Gguf::Open(argv[1]);
    const auto config = yah::core::Qwen35Config::FromGguf(gguf);
    const auto weights = yah::model::Qwen35Weights::FromGguf(gguf, config);
    const double gib = static_cast<double>(weights.bytes()) / 1073741824.0;
    std::printf("resolved %zu layers, %.2f GiB of mapped tensors\n",
                weights.layers.size(), gib);
    std::printf("token_embd %s, output %s, output_norm present=%d\n",
                yah::core::TypeName(weights.token_embd.type),
                yah::core::TypeName(weights.output.type),
                weights.output_norm.present ? 1 : 0);
    for (const std::uint32_t index : {0U, 3U}) {
      const auto& layer = weights.layers[index];
      std::printf("blk.%u %s: norm=%s ffn_gate=%s ffn_up=%s ffn_down=%s\n", index,
                  layer.is_full_attention ? "attention" : "deltanet",
                  yah::core::TypeName(layer.attn_norm.type),
                  yah::core::TypeName(layer.ffn_gate.type),
                  yah::core::TypeName(layer.ffn_up.type),
                  yah::core::TypeName(layer.ffn_down.type));
      if (layer.is_full_attention) {
        std::printf("   q=%s[%llu,%llu] k=%s[%llu,%llu] v=%s[%llu,%llu] o=%s\n",
                    yah::core::TypeName(layer.attn_q.type),
                    static_cast<unsigned long long>(layer.attn_q.cols()),
                    static_cast<unsigned long long>(layer.attn_q.rows()),
                    yah::core::TypeName(layer.attn_k.type),
                    static_cast<unsigned long long>(layer.attn_k.cols()),
                    static_cast<unsigned long long>(layer.attn_k.rows()),
                    yah::core::TypeName(layer.attn_v.type),
                    static_cast<unsigned long long>(layer.attn_v.cols()),
                    static_cast<unsigned long long>(layer.attn_v.rows()),
                    yah::core::TypeName(layer.attn_output.type));
      } else {
        std::printf("   qkv=%s[%llu,%llu] gate=%s ssm_out=%s[%llu,%llu]\n",
                    yah::core::TypeName(layer.attn_qkv.type),
                    static_cast<unsigned long long>(layer.attn_qkv.cols()),
                    static_cast<unsigned long long>(layer.attn_qkv.rows()),
                    yah::core::TypeName(layer.attn_gate.type),
                    yah::core::TypeName(layer.ssm_out.type),
                    static_cast<unsigned long long>(layer.ssm_out.cols()),
                    static_cast<unsigned long long>(layer.ssm_out.rows()));
        std::printf("   alpha=%s beta=%s conv1d=%s a=%s dt=%s norm=%s\n",
                    yah::core::TypeName(layer.ssm_alpha.type),
                    yah::core::TypeName(layer.ssm_beta.type),
                    yah::core::TypeName(layer.ssm_conv1d.type),
                    yah::core::TypeName(layer.ssm_a.type),
                    yah::core::TypeName(layer.ssm_dt.type),
                    yah::core::TypeName(layer.ssm_norm.type));
      }
    }
    std::printf("WEIGHTS OK\n");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "yah-weights: %s\n", error.what());
    return 1;
  }
}
