// HFQ containers (hipfire, crates/hipfire-runtime/src/hfq.rs) opened as the GGUF the engine reads.
// Only Qwen3.8 with MQ4G256V2 symmetric weights (mq4-xts) is accepted. The file is read into resident memory and
// rewritten in place: GGUF names and dims, signed weight nibbles, DeltaNet value heads in GGUF order (tiled over the
// key heads), norms as GGUF stores them. Converted small tensors go after the file's bytes.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

#include "core/gguf.hpp"
#include "nlohmann/json.hpp"

namespace yah::core {
namespace {

using Json = nlohmann::json;

constexpr std::uint32_t kOverlayFlag = 0x80000000U;
constexpr std::uint8_t kQtF16 = 1, kQtQ8_0 = 3, kQtMQ4 = 44;
constexpr std::size_t kMq4Block = 136, kMq4Nibbles = 8;  // 4 f16 header, then 128 B of nibbles

struct HfqTensor {
  std::string name;
  std::uint8_t qt;
  std::vector<std::uint64_t> shape;  // HF order: [rows, row length]
  std::uint64_t offset;              // from the start of the file
  std::uint64_t size;
};

float Half(std::uint16_t h) {
  return static_cast<float>(std::bit_cast<_Float16>(h));
}

MetadataValue UInt(std::uint64_t u) {
  MetadataValue v;
  v.kind = MetadataValue::Kind::kUInt;
  v.u = u;
  return v;
}
MetadataValue Int(std::int64_t i) {
  MetadataValue v;
  v.kind = MetadataValue::Kind::kInt;
  v.i = i;
  return v;
}
MetadataValue Float(double f) {
  MetadataValue v;
  v.kind = MetadataValue::Kind::kFloat;
  v.f = f;
  return v;
}
MetadataValue Bool(bool b) {
  MetadataValue v;
  v.kind = MetadataValue::Kind::kBool;
  v.b = b;
  return v;
}
MetadataValue Str(std::string s) {
  MetadataValue v;
  v.kind = MetadataValue::Kind::kString;
  v.s = std::move(s);
  return v;
}
MetadataValue Array(std::vector<MetadataValue> a) {
  MetadataValue v;
  v.kind = MetadataValue::Kind::kArray;
  v.array = std::move(a);
  return v;
}

// The metadata JSON object ends at its balancing brace; the tensor index follows directly.
std::size_t JsonEnd(const std::string& s) {
  int depth = 0;
  bool in_str = false, esc = false;
  for (std::size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (esc) {
      esc = false;
    } else if (in_str && c == '\\') {
      esc = true;
    } else if (c == '"') {
      in_str = !in_str;
    } else if (!in_str && c == '{') {
      ++depth;
    } else if (!in_str && c == '}' && --depth == 0) {
      return i + 1;
    }
  }
  throw std::runtime_error("hfq: metadata is not a JSON object");
}

Json Sub(const Json& meta, const char* key) {
  if (!meta.contains(key)) throw std::runtime_error(std::string("hfq: metadata has no ") + key);
  const Json& v = meta[key];
  return v.is_string() ? Json::parse(v.get<std::string>()) : v;
}

// GGUF metadata for Qwen3.8: the keys Qwen35Config and TokenizerConfig read.
std::map<std::string, MetadataValue> Metadata(const Json& meta) {
  const Json cfg = Sub(meta, "config");
  const Json& t = cfg.contains("text_config") ? cfg["text_config"] : cfg;
  const Json tok = Sub(meta, "tokenizer");
  const Json tc = Sub(meta, "tokenizer_config");
  const Json gen = Sub(meta, "generation_config");
  std::map<std::string, MetadataValue> m;
  const std::string a = "qwen35.";
  const std::uint32_t head_dim = t["head_dim"];
  m["general.architecture"] = Str("qwen35");
  m["general.name"] = Str("Qwen3.8 27B");
  m[a + "block_count"] = UInt(t["num_hidden_layers"]);
  m[a + "context_length"] = UInt(t["max_position_embeddings"]);
  m[a + "embedding_length"] = UInt(t["hidden_size"]);
  m[a + "feed_forward_length"] = UInt(t["intermediate_size"]);
  m[a + "attention.head_count"] = UInt(t["num_attention_heads"]);
  m[a + "attention.head_count_kv"] = UInt(t["num_key_value_heads"]);
  m[a + "attention.key_length"] = UInt(head_dim);
  m[a + "attention.value_length"] = UInt(head_dim);
  m[a + "attention.layer_norm_rms_epsilon"] = Float(static_cast<float>(t["rms_norm_eps"].get<double>()));
  const Json& rope = t["rope_parameters"];
  m[a + "rope.dimension_count"] =
      UInt(static_cast<std::uint64_t>(head_dim * rope["partial_rotary_factor"].get<double>()));
  m[a + "rope.freq_base"] = Float(static_cast<float>(rope["rope_theta"].get<double>()));
  m[a + "full_attention_interval"] = UInt(t["full_attention_interval"]);
  m[a + "nextn_predict_layers"] = UInt(0);  // the container has no MTP tensors
  m[a + "ssm.conv_kernel"] = UInt(t["linear_conv_kernel_dim"]);
  m[a + "ssm.state_size"] = UInt(t["linear_key_head_dim"]);
  m[a + "ssm.group_count"] = UInt(t["linear_num_key_heads"]);
  m[a + "ssm.time_step_rank"] = UInt(t["linear_num_value_heads"]);
  m[a + "ssm.inner_size"] =
      UInt(t["linear_value_head_dim"].get<std::uint64_t>() * t["linear_num_value_heads"].get<std::uint64_t>());

  // Tokenizer as the GGUF converter writes it: vocab + added tokens (special: control 3, else user-defined 4), the
  // ids in neither are [PAD<id>] (unused 5).
  const std::size_t n = t["vocab_size"];
  std::vector<MetadataValue> tokens(n), types(n);
  std::vector<bool> seen(n);
  for (const auto& [s, id] : tok["model"]["vocab"].items()) {
    const std::size_t i = id;
    tokens[i] = Str(s), types[i] = Int(1), seen[i] = true;
  }
  std::map<std::string, std::uint32_t> added;
  for (const Json& at : tok["added_tokens"]) {
    const std::size_t i = at["id"];
    const std::string content = at["content"];
    tokens[i] = Str(content), types[i] = Int(at["special"].get<bool>() ? 3 : 4), seen[i] = true;
    added[content] = static_cast<std::uint32_t>(i);
  }
  for (std::size_t i = 0; i < n; ++i)
    if (!seen[i]) tokens[i] = Str("[PAD" + std::to_string(i) + "]"), types[i] = Int(5);
  std::vector<MetadataValue> merges;
  merges.reserve(tok["model"]["merges"].size());
  for (const Json& mg : tok["model"]["merges"])
    merges.push_back(
        Str(mg.is_string() ? mg.get<std::string>() : mg[0].get<std::string>() + " " + mg[1].get<std::string>()));
  const auto id_of = [&](const Json& v) -> std::uint32_t {
    if (v.is_number()) return v;
    const auto it = added.find(v.get<std::string>());
    if (it == added.end()) throw std::runtime_error("hfq: unknown special token " + v.get<std::string>());
    return it->second;
  };
  const Json& bos = gen["bos_token_id"];
  m["tokenizer.ggml.model"] = Str("gpt2");
  m["tokenizer.ggml.pre"] = Str("qwen35");
  m["tokenizer.ggml.tokens"] = Array(std::move(tokens));
  m["tokenizer.ggml.token_type"] = Array(std::move(types));
  m["tokenizer.ggml.merges"] = Array(std::move(merges));
  m["tokenizer.ggml.bos_token_id"] = UInt(id_of(bos.is_array() ? bos[0] : bos));
  m["tokenizer.ggml.eos_token_id"] = UInt(id_of(tc["eos_token"]));
  m["tokenizer.ggml.padding_token_id"] = UInt(id_of(tc["pad_token"]));
  m["tokenizer.ggml.add_bos_token"] = Bool(tc.value("add_bos_token", false));
  m["tokenizer.chat_template"] = Str(tc.value("chat_template", std::string()));
  return m;
}

enum class Conv { kNone, kNorm1p, kF32, kNegExp, kQ8ToF32 };

struct Target {
  std::string name;  // GGUF name
  Conv conv{Conv::kNone};
  std::uint32_t head_rows{0};   // DeltaNet value-head permutation: rows per head (0: none)
  std::uint64_t head_first{0};  // first row of the permuted heads
};

// HF name -> GGUF name and conversion. Throws on a tensor the engine does not know.
Target Map(const std::string& hf, std::uint32_t key_heads, std::uint32_t key_dim, std::uint32_t value_dim) {
  static const std::map<std::string, std::pair<std::string, Conv>> kTop = {
      {"model.language_model.embed_tokens.weight", {"token_embd.weight", Conv::kNone}},
      {"model.language_model.norm.weight", {"output_norm.weight", Conv::kNorm1p}},
      {"lm_head.weight", {"output.weight", Conv::kNone}},
  };
  static const std::map<std::string, std::pair<std::string, Conv>> kLayer = {
      {"input_layernorm.weight", {"attn_norm.weight", Conv::kNorm1p}},
      {"post_attention_layernorm.weight", {"post_attention_norm.weight", Conv::kNorm1p}},
      {"mlp.gate_proj", {"ffn_gate", Conv::kNone}},
      {"mlp.up_proj", {"ffn_up", Conv::kNone}},
      {"mlp.down_proj", {"ffn_down", Conv::kNone}},
      {"self_attn.q_proj", {"attn_q", Conv::kNone}},
      {"self_attn.k_proj", {"attn_k", Conv::kNone}},
      {"self_attn.v_proj", {"attn_v", Conv::kNone}},
      {"self_attn.o_proj", {"attn_output", Conv::kNone}},
      {"self_attn.q_norm.weight", {"attn_q_norm.weight", Conv::kNorm1p}},
      {"self_attn.k_norm.weight", {"attn_k_norm.weight", Conv::kNorm1p}},
      {"linear_attn.in_proj_qkv", {"attn_qkv", Conv::kNone}},
      {"linear_attn.in_proj_z", {"attn_gate", Conv::kNone}},
      {"linear_attn.in_proj_a", {"ssm_alpha", Conv::kNone}},
      {"linear_attn.in_proj_b", {"ssm_beta", Conv::kNone}},
      {"linear_attn.out_proj", {"ssm_out", Conv::kNone}},
      {"linear_attn.A_log", {"ssm_a", Conv::kNegExp}},
      {"linear_attn.dt_bias", {"ssm_dt.bias", Conv::kF32}},
      {"linear_attn.conv1d.weight", {"ssm_conv1d.weight", Conv::kQ8ToF32}},
      {"linear_attn.norm.weight", {"ssm_norm.weight", Conv::kF32}},
  };
  const auto top = kTop.find(hf);
  if (top != kTop.end()) return {top->second.first, top->second.second};
  const std::string pre = "model.language_model.layers.";
  if (hf.compare(0, pre.size(), pre) == 0) {
    const std::size_t dot = hf.find('.', pre.size());
    const std::string layer = hf.substr(pre.size(), dot - pre.size());
    std::string rest = hf.substr(dot + 1);
    // projections: <site>.weight, and the AWQ activation scale <site>.awq_scale.weight (x is divided by it)
    std::string tail;
    for (const char* suffix : {".awq_scale.weight", ".weight"}) {
      const std::size_t n = std::strlen(suffix);
      if (rest.size() > n && rest.compare(rest.size() - n, n, suffix) == 0 && !kLayer.contains(rest)) {
        tail = std::strcmp(suffix, ".weight") == 0 ? ".weight" : ".awq";
        rest.resize(rest.size() - n);
        break;
      }
    }
    const auto it = kLayer.find(rest);
    if (it != kLayer.end()) {
      Target t{"blk." + layer + "." + it->second.first + tail, it->second.second};
      if (tail == ".awq") t.conv = Conv::kF32;
      if (tail == ".weight" || tail.empty()) {
        if (rest == "linear_attn.in_proj_qkv") t.head_rows = value_dim, t.head_first = 2ULL * key_heads * key_dim;
        if (rest == "linear_attn.in_proj_z") t.head_rows = value_dim;
        if (rest == "linear_attn.in_proj_a" || rest == "linear_attn.in_proj_b" || rest == "linear_attn.A_log" ||
            rest == "linear_attn.dt_bias")
          t.head_rows = 1;
        if (rest == "linear_attn.conv1d.weight") t.head_rows = value_dim, t.head_first = 2ULL * key_heads * key_dim;
      }
      return t;
    }
  }
  throw std::runtime_error("hfq: unknown tensor " + hf);
}

// GGUF value head i is HF head P[i]: GGUF tiles the value heads over the key heads, HF groups them by key head.
std::vector<std::uint32_t> HeadOrder(std::uint32_t value_heads, std::uint32_t key_heads) {
  std::vector<std::uint32_t> p(value_heads);
  const std::uint32_t r = value_heads / key_heads;
  for (std::uint32_t i = 0; i < value_heads; ++i) p[i] = r * (i % key_heads) + i / key_heads;
  return p;
}

// Reorders `heads` blocks of `block` bytes starting at p: block i becomes old block order[i].
void Permute(std::uint8_t* p, std::size_t block, const std::vector<std::uint32_t>& order) {
  std::vector<std::uint8_t> old(p, p + block * order.size());
  for (std::size_t i = 0; i < order.size(); ++i) std::memcpy(p + i * block, old.data() + order[i] * block, block);
}

}  // namespace

Gguf Gguf::OpenHfq(const std::string& path, bool lock, bool weights) {
  Gguf g;
  g.fd_ = ::open(path.c_str(), O_RDONLY);
  if (g.fd_ < 0) throw std::runtime_error("hfq: cannot open " + path);
  struct stat st{};
  if (::fstat(g.fd_, &st) != 0) throw std::runtime_error("hfq: fstat failed");
  const std::size_t file = static_cast<std::size_t>(st.st_size);
  struct [[gnu::packed]] {
    char magic[4];
    std::uint32_t version, arch, count;
    std::uint64_t meta_off, data_off;
  } h{};
  if (::pread(g.fd_, &h, sizeof(h), 0) != sizeof(h)) throw std::runtime_error("hfq: truncated header");
  if ((h.version & ~kOverlayFlag) != 1 || (h.version & kOverlayFlag) != 0)
    throw std::runtime_error("hfq: only version 1 without a metadata overlay is supported");
  if (h.meta_off >= h.data_off || h.data_off > file) throw std::runtime_error("hfq: bad header offsets");

  std::string head(h.data_off - h.meta_off, '\0');
  if (::pread(g.fd_, head.data(), head.size(), static_cast<off_t>(h.meta_off)) != static_cast<ssize_t>(head.size()))
    throw std::runtime_error("hfq: truncated metadata");
  const std::size_t meta_end = JsonEnd(head);
  const Json meta = Json::parse(head.begin(), head.begin() + static_cast<std::ptrdiff_t>(meta_end));
  if (meta.value("mq4v2.symmetric", 0) != 1) throw std::runtime_error("hfq: only symmetric MQ4G256V2 is supported");
  g.meta_ = Metadata(meta);

  // tensor index
  std::size_t p = meta_end;
  const auto take = [&](void* dst, std::size_t n) {
    if (p + n > head.size()) throw std::runtime_error("hfq: truncated tensor index");
    std::memcpy(dst, head.data() + p, n);
    p += n;
  };
  std::uint32_t count = 0;
  take(&count, 4);
  if (count != h.count) throw std::runtime_error("hfq: tensor count mismatch");
  std::vector<HfqTensor> hts(count);
  std::uint64_t cur = h.data_off;
  for (auto& t : hts) {
    std::uint16_t len = 0;
    take(&len, 2);
    t.name.resize(len);
    take(t.name.data(), len);
    std::uint8_t nd = 0;
    take(&t.qt, 1);
    take(&nd, 1);
    t.shape.resize(nd);
    for (auto& d : t.shape) {
      std::uint32_t v = 0;
      take(&v, 4);
      d = v;
    }
    std::uint32_t group = 0;
    take(&group, 4);
    take(&t.size, 8);
    t.offset = cur;
    cur += t.size;
  }
  if (cur > file) throw std::runtime_error("hfq: tensor data past the end of " + path);

  const Json cfg = Sub(meta, "config");
  const Json& tcfg = cfg.contains("text_config") ? cfg["text_config"] : cfg;
  const std::uint32_t key_heads = tcfg["linear_num_key_heads"], key_dim = tcfg["linear_key_head_dim"];
  const std::uint32_t value_heads = tcfg["linear_num_value_heads"], value_dim = tcfg["linear_value_head_dim"];
  const std::vector<std::uint32_t> order = HeadOrder(value_heads, key_heads);

  // Layout: the file as is, then the converted (f32) tensors, 256 B aligned.
  struct Job {
    const HfqTensor* src;
    Target dst;
    std::uint64_t out;  // converted tensors: offset from data_off
  };
  std::vector<Job> jobs;
  jobs.reserve(count);
  std::uint64_t extra = 0;
  const std::uint64_t tail = file - h.data_off;
  for (const auto& t : hts) {
    Job j{&t, Map(t.name, key_heads, key_dim, value_dim), t.offset - h.data_off};
    std::uint64_t elements = 1;
    for (const auto d : t.shape) elements *= d;
    if (j.dst.conv != Conv::kNone) {
      j.out = tail + extra;
      extra += (elements * 4 + 255) / 256 * 256;
    }
    if (t.qt == kQtMQ4 &&
        (t.shape.size() != 2 || t.shape[1] % 256 || t.size != t.shape[0] * t.shape[1] / 256 * kMq4Block))
      throw std::runtime_error("hfq: bad MQ4G256V2 tensor " + t.name);
    if (t.qt != kQtMQ4 && t.qt != kQtQ8_0 && t.qt != kQtF16)
      throw std::runtime_error("hfq: unsupported type in " + t.name);
    if ((t.qt == kQtF16) != (j.dst.conv == Conv::kNorm1p || j.dst.conv == Conv::kF32 || j.dst.conv == Conv::kNegExp))
      throw std::runtime_error("hfq: unexpected type for " + t.name);
    jobs.push_back(std::move(j));
  }

  g.size_ = file + extra;
  g.version_ = 3;
  g.data_offset_ = h.data_off;
  constexpr std::size_t kHuge = std::size_t{2} << 20;
  const std::size_t bytes = (file + extra + kHuge - 1) & ~(kHuge - 1);
  std::uint8_t* base = weights ? Reserve(bytes, path) : nullptr;
  g.base_ = base, g.map_bytes_ = weights ? bytes : 0;
  if (weights) g.ReadAll(base, file, path);
  ::close(g.fd_);
  g.fd_ = -1;

  std::uint8_t* data = weights ? base + h.data_off : nullptr;
  std::atomic<std::size_t> next{0};
  const auto work = [&] {
    for (std::size_t i; (i = next.fetch_add(1)) < jobs.size();) {
      const Job& j = jobs[i];
      const HfqTensor& t = *j.src;
      std::uint8_t* src = base + t.offset;
      std::uint64_t elements = 1;
      for (const auto d : t.shape) elements *= d;
      if (t.qt == kQtMQ4) {
        // code - 8 as a signed nibble: xor 8 in both nibbles of every weight byte
        for (std::uint64_t b = 0; b < t.size; b += kMq4Block) {
          std::uint64_t* q = reinterpret_cast<std::uint64_t*>(src + b + kMq4Nibbles);
          for (int k = 0; k < 16; ++k) q[k] ^= 0x8888888888888888ULL;
        }
        if (j.dst.head_rows) {
          const std::size_t row = t.shape[1] / 256 * kMq4Block;
          Permute(src + j.dst.head_first * row, j.dst.head_rows * row, order);
        }
        continue;
      }
      if (j.dst.conv == Conv::kNone) continue;  // Q8_0 embedding: ggml layout already
      float* out = reinterpret_cast<float*>(data + j.out);
      if (t.qt == kQtQ8_0) {
        for (std::uint64_t b = 0; b < elements / 32; ++b) {
          std::uint16_t d = 0;
          std::memcpy(&d, src + b * 34, 2);
          const auto* q = reinterpret_cast<const std::int8_t*>(src + b * 34 + 2);
          for (int k = 0; k < 32; ++k) out[b * 32 + k] = Half(d) * static_cast<float>(q[k]);
        }
      } else {
        const auto* w = reinterpret_cast<const std::uint16_t*>(src);
        for (std::uint64_t k = 0; k < elements; ++k) {
          const float v = Half(w[k]);
          out[k] = j.dst.conv == Conv::kNorm1p ? 1.0F + v : j.dst.conv == Conv::kNegExp ? -std::exp(v) : v;
        }
      }
      if (j.dst.head_rows) {
        const std::size_t row = elements / t.shape[0] * 4;  // f32 bytes per HF row (conv1d: per channel)
        Permute(reinterpret_cast<std::uint8_t*>(out) + j.dst.head_first * row, j.dst.head_rows * row, order);
      }
    }
  };
  std::vector<std::thread> pool;
  const unsigned threads = std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
  for (unsigned i = 0; weights && i < threads; ++i) pool.emplace_back(work);
  for (auto& th : pool) th.join();

  g.tensors_.reserve(jobs.size());
  for (const Job& j : jobs) {
    const HfqTensor& t = *j.src;
    TensorInfo info;
    info.name = j.dst.name;
    info.dims.assign(t.shape.rbegin(), t.shape.rend());
    if (t.name.ends_with("conv1d.weight")) info.dims = {t.shape.back(), t.shape.front()};  // [kernel, channels]
    info.elements = 1;
    for (const auto d : info.dims) info.elements *= d;
    info.type = t.qt == kQtMQ4 ? GgmlType::kMQ4G256 : j.dst.conv != Conv::kNone ? GgmlType::kF32 : GgmlType::kQ8_0;
    info.offset = j.out;
    info.bytes = info.elements / BlockElements(info.type) * BlockBytes(info.type);
    g.tensors_.push_back(std::move(info));
  }
  if (lock && ::mlock(base, bytes) != 0)
    std::fprintf(stderr,
                 "hfq: cannot lock the model in memory (RLIMIT_MEMLOCK); it may be paged out under memory "
                 "pressure. To lock it: a line '<user> - memlock unlimited' in /etc/security/limits.d/, then log in "
                 "again\n");
  return g;
}

}  // namespace yah::core
