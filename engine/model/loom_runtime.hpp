// Thin RAII wrappers over the HRX native API: device, stream, executable, buffer, and a dispatch helper.
#ifndef YAH_MODEL_LOOM_RUNTIME_HPP_
#define YAH_MODEL_LOOM_RUNTIME_HPP_

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/gguf.hpp"
#include "hrx_runtime.h"

namespace yah::model {

class LoomError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline void LoomCheck(hrx_status_t status, const char* what) {
  if (hrx_status_is_ok(status)) return;
  std::string text = what;
  char* message = nullptr;
  size_t length = 0;
  if (hrx_status_is_ok(hrx_status_to_string(status, &message, &length)) && message) {
    text += ": ";
    text.append(message, length);
    hrx_status_free_message(message);
  }
  throw LoomError(text);
}

struct LoomBuffer {
  hrx_buffer_t handle = nullptr;
  size_t size = 0;

  LoomBuffer() = default;
  LoomBuffer(LoomBuffer&& other) noexcept : handle(other.handle), size(other.size) {
    other.handle = nullptr;
    other.size = 0;
  }
  LoomBuffer& operator=(LoomBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      handle = other.handle;
      size = other.size;
      other.handle = nullptr;
      other.size = 0;
    }
    return *this;
  }
  LoomBuffer(const LoomBuffer&) = delete;
  LoomBuffer& operator=(const LoomBuffer&) = delete;
  ~LoomBuffer() { reset(); }
  void reset() {
    if (handle) hrx_buffer_release(handle);
    handle = nullptr;
    size = 0;
  }
};

struct LoomEvent {
  hrx_event_t handle = nullptr;
  LoomEvent() = default;
  LoomEvent(LoomEvent&& other) noexcept : handle(other.handle) { other.handle = nullptr; }
  LoomEvent& operator=(LoomEvent&& other) noexcept {
    if (this != &other) {
      reset();
      handle = other.handle;
      other.handle = nullptr;
    }
    return *this;
  }
  LoomEvent(const LoomEvent&) = delete;
  LoomEvent& operator=(const LoomEvent&) = delete;
  ~LoomEvent() { reset(); }
  void reset() {
    if (handle) hrx_event_release(handle);
    handle = nullptr;
  }
};

struct LoomExecutable {
  hrx_executable_t handle = nullptr;
  std::vector<std::string> names;
  std::vector<hrx_executable_export_info_t> infos;

  LoomExecutable() = default;
  LoomExecutable(LoomExecutable&& other) noexcept
      : handle(other.handle), names(std::move(other.names)), infos(std::move(other.infos)) {
    other.handle = nullptr;
    other.infos.clear();
  }
  LoomExecutable& operator=(LoomExecutable&& other) noexcept {
    if (this != &other) {
      reset();
      handle = other.handle;
      names = std::move(other.names);
      infos = std::move(other.infos);
      other.handle = nullptr;
      other.infos.clear();
    }
    return *this;
  }
  LoomExecutable(const LoomExecutable&) = delete;
  LoomExecutable& operator=(const LoomExecutable&) = delete;
  ~LoomExecutable() { reset(); }
  void reset() {
    if (handle) hrx_executable_release(handle);
    handle = nullptr;
    names.clear();
    infos.clear();
  }
  [[nodiscard]] uint32_t Ordinal(const std::string& name) const {
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == name) return static_cast<uint32_t>(i);
    }
    throw LoomError("export not found: " + name);
  }
  [[nodiscard]] uint32_t OrdinalOrZero(const std::string& name) const {
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == name) return static_cast<uint32_t>(i);
    }
    return 0;
  }
  // The workgroup size this export was compiled with, or 0 if the metadata does not carry it.
  // Do not hardcode it: a wave64 kernel launched with 32 threads computes a wrong tile and looks fast.
  [[nodiscard]] uint32_t WorkgroupSize(uint32_t ordinal) const {
    if (ordinal < infos.size() && infos[ordinal].workgroup_size[0] != 0) {
      return infos[ordinal].workgroup_size[0];
    }
    return 0;
  }
  // How many buffers this export's dispatch binds.
  [[nodiscard]] uint32_t BindingCount(uint32_t ordinal) const {
    if (ordinal < infos.size()) return infos[ordinal].binding_count;
    return 0;
  }
};

class LoomDevice {
 public:
  LoomDevice() {
    LoomCheck(hrx_gpu_initialize(0), "hrx_gpu_initialize");
    initialized_ = true;
    if (!hrx_status_is_ok(hrx_gpu_device_get(0, &device_))) {
      hrx_gpu_shutdown();
      throw LoomError("hrx_gpu_device_get");
    }
    LoomCheck(hrx_stream_create(device_, 0, &stream_), "hrx_stream_create");
  }
  LoomDevice(const LoomDevice&) = delete;
  LoomDevice& operator=(const LoomDevice&) = delete;
  ~LoomDevice() {
    if (stream_) hrx_stream_release(stream_);
    // device_ is borrowed (hrx_gpu_device_get does not retain it); hrx_gpu_shutdown releases it.
    // Do not release it here: that clears the device early and HRX_PROFILE_FILE gets no dispatch events or session_end.
    if (initialized_) hrx_gpu_shutdown();
  }

  [[nodiscard]] hrx_device_t device() const { return device_; }

  [[nodiscard]] hrx_stream_t stream() const { return stream_; }

  [[nodiscard]] LoomExecutable Load(const std::string& path, const char* target_key = "gfx1151") {
    LoomExecutable executable;
    LoomCheck(hrx_executable_load_file(device_, path.c_str(), "amdgpu", target_key, &executable.handle),
              "hrx_executable_load_file");
    size_t count = 0;
    LoomCheck(hrx_executable_export_count(executable.handle, &count), "hrx_executable_export_count");
    executable.names.reserve(count);
    executable.infos.resize(count);
    for (size_t i = 0; i < count; ++i) {
      LoomCheck(hrx_executable_export_info(executable.handle, static_cast<uint32_t>(i), &executable.infos[i]),
                "hrx_executable_export_info");
      executable.names.emplace_back(executable.infos[i].name ? executable.infos[i].name : "");
    }
    return executable;
  }

  [[nodiscard]] LoomBuffer Allocate(size_t bytes) {
    LoomBuffer buffer;
    buffer.size = bytes;
    LoomCheck(
        hrx_buffer_allocate(stream_, bytes, HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &buffer.handle),
        "hrx_buffer_allocate");
    return buffer;
  }

  // A host-local buffer, persistently mapped at *host: the host reads what the stream copied there once it completed
  // (Mark / Wait), without a queue transfer (a synchronous D2H waits behind all queued work).
  [[nodiscard]] LoomBuffer AllocateHost(size_t bytes, const void** host) {
    LoomBuffer buffer;
    buffer.size = bytes;
    const hrx_buffer_usage_t usage =
        HRX_BUFFER_USAGE_TRANSFER | HRX_BUFFER_USAGE_MAPPING_SCOPED | HRX_BUFFER_USAGE_MAPPING_PERSISTENT;
    LoomCheck(hrx_buffer_allocate(stream_, bytes, HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_HOST_COHERENT, usage,
                                  &buffer.handle),
              "hrx_buffer_allocate(host)");
    Synchronize();
    void* p = nullptr;
    LoomCheck(hrx_buffer_map(buffer.handle, HRX_MAP_READ, 0, bytes, &p), "hrx_buffer_map");
    *host = p;
    return buffer;
  }
  void Copy(const hrx_buffer_ref_t& src, const hrx_buffer_ref_t& dst) {
    Order();
    LoomCheck(hrx_stream_copy_buffer(stream_, src.buffer, src.offset, dst.buffer, dst.offset, src.length),
              "hrx_stream_copy_buffer");
  }
  // Fill the whole buffer with a 32-bit pattern, in stream order.
  void Fill(const LoomBuffer& buffer, uint32_t pattern) {
    Order();
    LoomCheck(hrx_stream_fill_buffer(stream_, buffer.handle, 0, buffer.size, &pattern, 4), "hrx_stream_fill_buffer");
  }
  // An event after the work queued so far (submitting it); Wait blocks until the GPU has run that work.
  LoomEvent Mark() {
    Order();
    LoomEvent e;
    LoomCheck(hrx_event_create(device_, HRX_EVENT_FLAG_DISABLE_TIMING, &e.handle), "hrx_event_create");
    LoomCheck(hrx_event_record(e.handle, stream_), "hrx_event_record");
    return e;
  }
  void Wait(const LoomEvent& e) { LoomCheck(hrx_event_synchronize(e.handle), "hrx_event_synchronize"); }
  // Host data into the buffer in stream order (copied now; at most 64 KB).
  void Update(const LoomBuffer& buffer, const void* host, size_t bytes, size_t offset = 0) {
    Order();
    LoomCheck(hrx_stream_update_buffer(stream_, host, bytes, buffer.handle, offset), "hrx_stream_update_buffer");
  }
  void H2D(const LoomBuffer& buffer, const void* host, size_t bytes, size_t offset = 0) {
    LoomCheck(hrx_synchronous_h2d(device_, host, buffer.handle, offset, bytes), "hrx_synchronous_h2d");
  }
  void D2H(const LoomBuffer& buffer, void* host, size_t bytes, size_t offset = 0) {
    LoomCheck(hrx_synchronous_d2h(device_, buffer.handle, offset, host, bytes), "hrx_synchronous_d2h");
  }

  void Dispatch(const LoomExecutable& executable, uint32_t ordinal, const hrx_dispatch_config_t& config,
                const void* constants, size_t constants_size, const hrx_buffer_ref_t* bindings, size_t binding_count) {
    Order();
    DispatchRaw(executable, ordinal, config, constants, constants_size, bindings, binding_count);
  }

  // Flag-chained dispatch (tools/gen_chain.py kernels, bindings + the counter buffer, constants wait slot, need,
  // own slot): the kernel waits on the previous chained dispatch's counter instead of a queue barrier, so it launches
  // while its producer drains. join: run beside the previous dispatch instead (same producer, same counter slot).
  void ChainDispatch(const LoomExecutable& executable, const hrx_dispatch_config_t& config,
                     std::vector<hrx_buffer_ref_t> bindings, bool join = false) {
    if (!chain_cnt_.handle) {
      chain_cnt_ = Allocate(kChainSlots * 4);
      std::vector<uint32_t> z(kChainSlots, 0);
      H2D(chain_cnt_, z.data(), z.size() * 4);
      chain_shadow_.assign(kChainSlots, 0);
    }
    if (!join) {
      chain_wait_ = chain_slot_;
      chain_need_ = chain_shadow_[chain_slot_];
      chain_slot_ = chain_slot_ % (kChainSlots - 1) + 1;   // slot 0 is never signaled: a chain start waits on it
    }
    const uint32_t waves = (config.workgroup_size[0] + 31) / 32;
    chain_shadow_[chain_slot_] += config.workgroup_count[0] * config.workgroup_count[1] * config.workgroup_count[2] * waves;
    bindings.push_back({chain_cnt_.handle, 0, chain_cnt_.size});
    const uint32_t c[3] = {chain_wait_, chain_need_, chain_slot_};
    next_flags_ = 1u << 2;
    DispatchRaw(executable, 0, config, c, sizeof c, bindings.data(), bindings.size());
    chain_open_ = no_barrier_ok_;
  }
  // Orders what follows after every chained dispatch so far, and starts a new chain.
  void Order() {
    if (chain_open_) LoomCheck(hrx_stream_execution_barrier(stream_), "hrx_stream_execution_barrier");
    chain_open_ = false;
    chain_slot_ = 0;
  }

 private:
  void DispatchRaw(const LoomExecutable& executable, uint32_t ordinal, const hrx_dispatch_config_t& config,
                   const void* constants, size_t constants_size, const hrx_buffer_ref_t* bindings, size_t binding_count) {
    const uint32_t flags = no_barrier_ok_ ? next_flags_ : 0;
    next_flags_ = 0;
    hrx_status_t status = hrx_stream_dispatch(stream_, executable.handle, ordinal, &config, constants, constants_size,
                                              bindings, binding_count, flags);
    // Stock HRX rejects the no-barrier flag up front (nothing recorded): retry with ordered dispatch from now on.
    if (flags && hrx_status_code(status) == HRX_STATUS_INVALID_ARGUMENT) {
      hrx_status_ignore(status);
      no_barrier_ok_ = false;
      status = hrx_stream_dispatch(stream_, executable.handle, ordinal, &config, constants, constants_size, bindings,
                                   binding_count, 0);
    }
    LoomCheck(status, "hrx_stream_dispatch");
  }

 public:
  // The next Dispatch may overlap the one after it (no trailing ordering barrier). Needs the local libhrx flag
  // HRX_DISPATCH_FLAG_NO_ORDERING_BARRIER (bit 2, not upstream); on stock HRX dispatches stay ordered.
  void NoBarrierNext() { next_flags_ = 1u << 2; }
  // Sleep-poll synchronize: with us > 0, poll an event at the stream tail and sleep us between checks.
  // The runtime's blocking wait busy-polls a host core (ROCr); a long queued run (the prefill) opts in.
  // Off by default: on short single waits the sleep slack shows in the timing.
  void SetSleepSync(long us) { sleep_us_ = us; }
  // Submits the queued work without waiting, so the GPU runs it while the host records more.
  void Flush() { LoomCheck(hrx_stream_flush(stream_), "hrx_stream_flush"); }
  void Synchronize() {
    if (sleep_us_ > 0) {
      // hrx_stream_query reports complete while the stream timepoint is 0 (true for plain dispatches): poll an event.
      LoomEvent tail;
      LoomCheck(hrx_event_create(device_, HRX_EVENT_FLAG_NONE, &tail.handle), "hrx_event_create");
      LoomCheck(hrx_event_record(tail.handle, stream_), "hrx_event_record");
      bool complete = false;
      for (;;) {
        LoomCheck(hrx_event_query(tail.handle, &complete), "hrx_event_query");
        if (complete) break;
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us_));
      }
    }
    LoomCheck(hrx_stream_synchronize(stream_), "sync");
  }

  [[nodiscard]] LoomEvent NewEvent() {
    LoomEvent event;
    LoomCheck(hrx_event_create(device_, HRX_EVENT_FLAG_NONE, &event.handle), "hrx_event_create");
    return event;
  }
  void Record(LoomEvent& event) {
    Order();
    LoomCheck(hrx_event_record(event.handle, stream_), "hrx_event_record");
  }
  float Elapsed(LoomEvent& start, LoomEvent& stop) {
    float ms = 0.0f;
    LoomCheck(hrx_event_elapsed_time(start.handle, stop.handle, &ms), "hrx_event_elapsed_time");
    return ms;
  }

  static hrx_dispatch_config_t Config(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t sx, uint32_t sy, uint32_t sz,
                                      uint32_t subgroup = 32) {
    hrx_dispatch_config_t config = {};
    config.workgroup_count[0] = gx;
    config.workgroup_count[1] = gy;
    config.workgroup_count[2] = gz;
    config.workgroup_size[0] = sx;
    config.workgroup_size[1] = sy;
    config.workgroup_size[2] = sz;
    config.subgroup_size = subgroup;
    return config;
  }

 private:
  hrx_device_t device_ = nullptr;
  hrx_stream_t stream_ = nullptr;
  uint32_t next_flags_ = 0;
  bool no_barrier_ok_ = true;
  static constexpr uint32_t kChainSlots = 1024;   // gen_chain.CHAIN_SLOTS
  LoomBuffer chain_cnt_;
  std::vector<uint32_t> chain_shadow_;   // what each counter reads once the chained dispatches so far have run
  uint32_t chain_slot_ = 0, chain_wait_ = 0, chain_need_ = 0;
  bool chain_open_ = false;   // the last dispatch has no trailing barrier
  long sleep_us_ = 0;
  bool initialized_ = false;
};

// Records dispatches into one HRX graph and launches it on the device stream. Each dispatch waits only for what it
// touches: the last earlier write to any byte range it reads or writes, and, for a write, the reads since then.
// Independent dispatches get no edge and HRX records no barrier between them, so they can run at the same time on the
// GPU. (Stream dispatches always end with an ordering barrier; graphs are the stock-HRX way to overlap kernels.)
class LoomGraph {
 public:
  explicit LoomGraph(LoomDevice& gpu) : gpu_(gpu) {
    LoomCheck(hrx_graph_create(gpu.device(), 0, &graph_), "hrx_graph_create");
  }
  ~LoomGraph() {
    if (exec_) hrx_graph_exec_release(exec_);
    if (graph_) hrx_graph_release(graph_);
  }
  LoomGraph(const LoomGraph&) = delete;
  LoomGraph& operator=(const LoomGraph&) = delete;

  // A buffer no recorded dispatch writes (weights, lookup tables): it orders nothing.
  void ReadOnly(hrx_buffer_t buffer) { read_only_.insert(buffer); }

  // writes: bit i set if the dispatch may write binding i. A missing bit is a race; an extra one only costs overlap.
  // Returns the node's index among the graph's dispatches (Grid). Not their timestamps' command index: that counts
  // HRX's barriers and atomic stores too.
  // after: ranges this dispatch is ordered after as if it read them, though it does not (it gets the same barrier as a
  // dispatch that does, so the two can run together); not recorded as accesses.
  // also_writes: ranges recorded as written by this dispatch though it does not bind them (a wait that stands for
  // another engine's writes).
  // untracked: bit i set if the kernel orders binding i itself (completion counters it waits on): no dependency, but
  // the access is recorded, so later dispatches still order after it.
  size_t Dispatch(const LoomExecutable& executable, uint32_t ordinal, const hrx_dispatch_config_t& config,
                  const hrx_buffer_ref_t* bindings, size_t binding_count, uint64_t writes,
                  const std::vector<hrx_buffer_ref_t>* after = nullptr,
                  const std::vector<hrx_buffer_ref_t>* also_writes = nullptr, uint64_t untracked = 0) {
    std::vector<hrx_graph_node_t> deps;
    for (size_t i = 0; i < binding_count; ++i)
      if (!read_only_.count(bindings[i].buffer) && !((untracked >> i) & 1))
        Depend(bindings[i], (writes >> i) & 1, &deps);
    if (after)
      for (const hrx_buffer_ref_t& r : *after)
        if (!read_only_.count(r.buffer)) Depend(r, false, &deps);
    if (also_writes)
      for (const hrx_buffer_ref_t& r : *also_writes) Depend(r, true, &deps);
    Unique(&deps);
    // The graph keeps the binding pointer until it is instantiated.
    const auto& b = bindings_.emplace_back(bindings, bindings + binding_count);
    hrx_graph_kernel_node_attrs_t attrs{};
    attrs.executable = executable.handle;
    attrs.export_ordinal = ordinal;
    attrs.config = config;
    attrs.bindings = b.data();
    attrs.binding_count = b.size();
    hrx_graph_node_t node = nullptr;
    LoomCheck(hrx_graph_add_kernel_node(graph_, deps.data(), deps.size(), &attrs, &node), "hrx_graph_add_kernel_node");
    for (size_t i = 0; i < binding_count; ++i)
      if (!read_only_.count(bindings[i].buffer))
        accesses_[bindings[i].buffer].push_back({bindings[i].offset, bindings[i].length, ((writes >> i) & 1) != 0, node});
    if (also_writes)
      for (const hrx_buffer_ref_t& r : *also_writes) accesses_[r.buffer].push_back({r.offset, r.length, true, node});
    grids_.push_back({config.workgroup_count[0], config.workgroup_count[1], config.workgroup_count[2]});
    return grids_.size() - 1;
  }
  // The workgroup count node i was recorded with.
  [[nodiscard]] const std::array<uint32_t, 3>& Grid(size_t i) const { return grids_[i]; }
  [[nodiscard]] size_t size() const { return grids_.size(); }

  // Instantiates the graph now, so that Launch only queues it (segments joined by host waits launch back to back).
  void Instantiate() {
    if (!exec_) LoomCheck(hrx_graph_instantiate(graph_, 0, &exec_), "hrx_graph_instantiate");
  }
  // Queues the graph on the device stream, after everything queued before (instantiates it first if needed).
  void Launch() {
    Instantiate();
    gpu_.Order();
    LoomCheck(hrx_graph_exec_launch(exec_, gpu_.stream()), "hrx_graph_exec_launch");
  }

 private:
  static void Unique(std::vector<hrx_graph_node_t>* deps) {
    std::sort(deps->begin(), deps->end());
    deps->erase(std::unique(deps->begin(), deps->end()), deps->end());
  }
  struct Access {
    size_t offset, length;
    bool write;
    hrx_graph_node_t node;
  };
  // Newest first: every overlapping earlier write orders this access, a write also waits for the overlapping reads.
  // A write that covers the whole range was itself ordered after everything older, so the scan stops there.
  void Depend(const hrx_buffer_ref_t& b, bool write, std::vector<hrx_graph_node_t>* deps) {
    const auto& list = accesses_[b.buffer];
    for (auto it = list.rbegin(); it != list.rend(); ++it) {
      if (it->offset >= b.offset + b.length || b.offset >= it->offset + it->length) continue;
      if (it->write) {
        deps->push_back(it->node);
        if (it->offset <= b.offset && it->offset + it->length >= b.offset + b.length) return;
      } else if (write) {
        deps->push_back(it->node);
      }
    }
  }

  LoomDevice& gpu_;
  hrx_graph_t graph_ = nullptr;
  hrx_graph_exec_t exec_ = nullptr;
  std::set<hrx_buffer_t> read_only_;
  std::map<hrx_buffer_t, std::vector<Access>> accesses_;
  std::deque<std::vector<hrx_buffer_ref_t>> bindings_;
  std::vector<std::array<uint32_t, 3>> grids_;
};

// The model's tensor data on the GPU; Ref(t) binds a tensor. By default it is copied into device-local buffers of about
// 1 GiB cut at tensor boundaries, each host range released right after its copy (RAM holds one copy of the model):
// the GPU reads imported host pages snooped, ~6% more GEMM cycles.
class LoomWeights {
 public:
  LoomWeights(LoomDevice& gpu, core::Gguf& gguf) {
    constexpr size_t kChunk = size_t{1} << 30;
    constexpr size_t kPad = size_t{64} << 10;  // a kernel's read past its last tensor stays mapped
    std::vector<std::pair<size_t, size_t>> spans;
    for (const auto& t : gguf.tensors()) spans.emplace_back(t.offset, t.offset + t.bytes);
    std::sort(spans.begin(), spans.end());
    for (size_t i = 0; i < spans.size();) {
      const size_t begin = spans[i].first;
      size_t end = spans[i].second;
      for (++i; i < spans.size() && spans[i].second - begin <= kChunk; ++i) end = std::max(end, spans[i].second);
      LoomBuffer buffer = gpu.Allocate(end - begin + kPad);
      gpu.H2D(buffer, gguf.tensor_data_base() + begin, end - begin);
      gguf.ReleaseData(begin, end - begin);
      chunks_.push_back({begin, std::move(buffer)});
    }
  }
  [[nodiscard]] hrx_buffer_ref_t Ref(const core::TensorInfo& t) const {
    const size_t off = static_cast<size_t>(t.offset);
    const auto it = std::upper_bound(chunks_.begin(), chunks_.end(), off,
                                     [](size_t o, const Chunk& c) { return o < c.begin; }) - 1;
    return {it->buffer.handle, off - it->begin, static_cast<size_t>(t.bytes)};
  }
  template <typename F>
  void ForEachBuffer(F&& f) const {
    for (const Chunk& c : chunks_) f(c.buffer.handle);
  }

 private:
  struct Chunk {
    size_t begin;  // tensor-data offset of the buffer's first byte
    LoomBuffer buffer;
  };
  std::vector<Chunk> chunks_;
};

}  // namespace yah::model

#endif  // YAH_MODEL_LOOM_RUNTIME_HPP_