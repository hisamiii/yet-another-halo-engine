// Thin RAII wrappers over the HRX native API: device, stream, executable, buffer, and a dispatch helper.
#ifndef YAH_MODEL_LOOM_RUNTIME_HPP_
#define YAH_MODEL_LOOM_RUNTIME_HPP_

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <deque>
#include <utility>
#include <cstdlib>
#include <thread>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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
  if (hrx_status_is_ok(hrx_status_to_string(status, &message, &length)) &&
      message) {
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
  LoomBuffer(LoomBuffer&& other) noexcept
      : handle(other.handle), size(other.size) {
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
  LoomEvent(LoomEvent&& other) noexcept : handle(other.handle) {
    other.handle = nullptr;
  }
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
      : handle(other.handle),
        names(std::move(other.names)),
        infos(std::move(other.infos)) {
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
  // The workgroup size this export was compiled with, or 0 when the metadata
  // does not carry it. Callers used to hardcode this (32 at every GEMM site),
  // which is wrong for any kernel declared with a different workgroup size: a
  // wave64 kernel launched with 32 threads runs half a wavegroup, so it both
  // computes the wrong tile and looks fast.
  [[nodiscard]] uint32_t WorkgroupSize(uint32_t ordinal) const {
    if (ordinal < infos.size() && infos[ordinal].workgroup_size[0] != 0) {
      return infos[ordinal].workgroup_size[0];
    }
    return 0;
  }
  // How many buffers this export's dispatch binds. The GEMM family is not
  // uniform: the IQ grid/signs formats take (weight, grid, [ksigns], input,
  // wstage, ostage, out) -- 7 for iq3xxs/iq2xxs/iq2xs, 6 for iq3s -- while
  // every other format takes just (weight, input, wstage, ostage, out) = 5.
  // A caller that hardcodes 6 rejects the whole non-grid family with
  // "dispatch binding count mismatch; expected 5 but got 6".
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
    if (std::getenv("YAH_LOOM_SLEEP_SYNC_US"))
      std::fprintf(stderr, "sync sleeps: %ld\n", sync_sleeps_);
    if (dispatch_n_)
      std::fprintf(stderr, "dispatch timing: %ld calls, %.1f ms total in hrx_stream_dispatch, max %.1f us, first->last enqueue %.1f ms\n",
                   dispatch_n_, dispatch_us_ / 1000.0, dispatch_max_us_,
                   std::chrono::duration<double, std::milli>(last_dispatch_ - first_dispatch_).count());
    pace_events_.clear();  // before the stream and runtime go away
    if (stream_) hrx_stream_release(stream_);
    // device_ is borrowed: hrx_gpu_device_get does not retain it, so releasing
    // it here drops the runtime's own reference and clears the device before
    // hrx_gpu_shutdown runs. That silently skipped the device profiling end:
    // with HRX_PROFILE_FILE set, the profile had a session_begin and nothing
    // else (no dispatch events, no session_end). hrx_gpu_shutdown releases it.
    if (initialized_) hrx_gpu_shutdown();
  }

  [[nodiscard]] hrx_device_t device() const { return device_; }
  [[nodiscard]] hrx_stream_t stream() const { return stream_; }

  [[nodiscard]] LoomExecutable Load(const std::string& path,
                                    const char* target_key = "gfx1151") {
    LoomExecutable executable;
    LoomCheck(hrx_executable_load_file(device_, path.c_str(), "amdgpu",
                                       target_key, &executable.handle),
              "hrx_executable_load_file");
    size_t count = 0;
    LoomCheck(hrx_executable_export_count(executable.handle, &count),
              "hrx_executable_export_count");
    executable.names.reserve(count);
    executable.infos.resize(count);
    for (size_t i = 0; i < count; ++i) {
      LoomCheck(hrx_executable_export_info(executable.handle,
                                           static_cast<uint32_t>(i),
                                           &executable.infos[i]),
                "hrx_executable_export_info");
      executable.names.emplace_back(executable.infos[i].name
                                        ? executable.infos[i].name
                                        : "");
    }
    return executable;
  }

  [[nodiscard]] LoomBuffer Allocate(size_t bytes) {
    LoomBuffer buffer;
    buffer.size = bytes;
    LoomCheck(hrx_buffer_allocate(stream_, bytes, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                  HRX_BUFFER_USAGE_DEFAULT, &buffer.handle),
              "hrx_buffer_allocate");
    return buffer;
  }

  // Import an external host pointer (e.g. a GGUF mmap window) as an HRX
  // buffer. The caller must keep the mapping alive while the buffer is used.
  [[nodiscard]] LoomBuffer Import(void* host_ptr, size_t bytes) {
    LoomBuffer buffer;
    buffer.size = bytes;
    hrx_buffer_params_t params = {};
    params.type = HRX_MEMORY_TYPE_DEVICE_VISIBLE;
    params.access = HRX_MEMORY_ACCESS_READ;
    params.usage = HRX_BUFFER_USAGE_DEFAULT;
    params.queue_affinity = 0;
    LoomCheck(hrx_allocator_import_buffer(hrx_device_allocator(device_),
                                          params, host_ptr, bytes,
                                          &buffer.handle),
              "hrx_allocator_import_buffer");
    return buffer;
  }

  void H2D(const LoomBuffer& buffer, const void* host, size_t bytes,
           size_t offset = 0) {
    LoomCheck(hrx_synchronous_h2d(device_, host, buffer.handle, offset, bytes),
              "hrx_synchronous_h2d");
  }
  void D2H(const LoomBuffer& buffer, void* host, size_t bytes,
           size_t offset = 0) {
    LoomCheck(hrx_synchronous_d2h(device_, buffer.handle, offset, host, bytes),
              "hrx_synchronous_d2h");
  }

  void Dispatch(const LoomExecutable& executable, uint32_t ordinal,
                const hrx_dispatch_config_t& config, const void* constants,
                size_t constants_size, const hrx_buffer_ref_t* bindings,
                size_t binding_count) {
    static const bool time_dispatch = std::getenv("YAH_LOOM_DISPATCH_TIMING") != nullptr;
    const auto t0 = time_dispatch ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const uint32_t flags = next_flags_;
    next_flags_ = 0;
    LoomCheck(hrx_stream_dispatch(stream_, executable.handle, ordinal, &config,
                                  constants, constants_size, bindings,
                                  binding_count, flags),
              "hrx_stream_dispatch");
    if (time_dispatch) {
      const auto t1 = std::chrono::steady_clock::now();
      if (!dispatch_n_) first_dispatch_ = t0;
      last_dispatch_ = t1;
      const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
      dispatch_us_ += us; dispatch_max_us_ = us > dispatch_max_us_ ? us : dispatch_max_us_; ++dispatch_n_;
    }
    Pace();
  }

  // YAH_LOOM_PACE=N[,M]: record an event every N dispatches and keep at most M
  // (default 4) outstanding; past that, sleep-poll on the oldest instead of
  // letting hrx_stream_dispatch spin on queue backpressure (one host core for
  // the whole prefill on an APU whose CPU and GPU share one power budget).
  void Pace() {
    static const std::pair<long, long> cfg = [] {
      const char* v = std::getenv("YAH_LOOM_PACE");
      long n = v ? std::atol(v) : 0, m = 4;
      if (v && std::strchr(v, ',')) m = std::atol(std::strchr(v, ',') + 1);
      return std::make_pair(n, m);
    }();
    if (cfg.first <= 0 || ++paced_ % cfg.first != 0) return;
    LoomEvent event;
    LoomCheck(hrx_event_create(device_, HRX_EVENT_FLAG_NONE, &event.handle), "hrx_event_create");
    LoomCheck(hrx_event_record(event.handle, stream_), "hrx_event_record");
    pace_events_.push_back(std::move(event));
    while (static_cast<long>(pace_events_.size()) > cfg.second) {
      bool complete = false;
      for (;;) {
        LoomCheck(hrx_event_query(pace_events_.front().handle, &complete), "hrx_event_query");
        if (complete) break;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
      pace_events_.pop_front();
    }
  }

  // Sleep-poll synchronize: with N > 0 the wait polls an event recorded at
  // the stream tail and sleeps N us between checks, instead of the runtime's
  // blocking wait, which busy-polls a host core for the whole wait (ROCr). A
  // driver that queues a long run before one wait (the prefill) opts in with
  // SetSleepSync(); YAH_LOOM_SLEEP_SYNC_US=N overrides it (0 = runtime wait).
  // Off by default: probes time single short waits, where N us of slack shows.
  // Experimental (needs the local HRX prototype flag, bit 2: skip the
  // trailing ordering barrier): the next Dispatch may overlap the one after
  // it. Never set against stock HRX, which rejects unknown flags.
  void NoBarrierNext() { next_flags_ = 1u << 2; }
  void SetSleepSync(long us) {
    if (!std::getenv("YAH_LOOM_SLEEP_SYNC_US")) sleep_us_ = us;
  }
  void Synchronize() {
    if (sleep_us_ > 0) {
      // hrx_stream_query reports complete while the stream timepoint is 0, which
      // it is for plain dispatches: poll an event recorded at the tail instead.
      LoomEvent tail;
      LoomCheck(hrx_event_create(device_, HRX_EVENT_FLAG_NONE, &tail.handle), "hrx_event_create");
      LoomCheck(hrx_event_record(tail.handle, stream_), "hrx_event_record");
      bool complete = false;
      for (;;) {
        LoomCheck(hrx_event_query(tail.handle, &complete), "hrx_event_query");
        if (complete) break;
        ++sync_sleeps_;
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us_));
      }
    }
    LoomCheck(hrx_stream_synchronize(stream_), "sync");
  }

  [[nodiscard]] LoomEvent NewEvent() {
    LoomEvent event;
    LoomCheck(hrx_event_create(device_, HRX_EVENT_FLAG_NONE, &event.handle),
              "hrx_event_create");
    return event;
  }
  void Record(LoomEvent& event) {
    LoomCheck(hrx_event_record(event.handle, stream_), "hrx_event_record");
  }
  float Elapsed(LoomEvent& start, LoomEvent& stop) {
    float ms = 0.0f;
    LoomCheck(hrx_event_elapsed_time(start.handle, stop.handle, &ms),
              "hrx_event_elapsed_time");
    return ms;
  }

  static hrx_dispatch_config_t Config(uint32_t gx, uint32_t gy, uint32_t gz,
                                      uint32_t sx, uint32_t sy, uint32_t sz,
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
  long paced_ = 0;
  long dispatch_n_ = 0;
  long sync_sleeps_ = 0;
  uint32_t next_flags_ = 0;
  long sleep_us_ = [] {
    const char* v = std::getenv("YAH_LOOM_SLEEP_SYNC_US");
    return v ? std::atol(v) : 0L;
  }();
  std::chrono::steady_clock::time_point first_dispatch_, last_dispatch_;
  double dispatch_us_ = 0, dispatch_max_us_ = 0;
  std::deque<LoomEvent> pace_events_;
  bool initialized_ = false;
};

}  // namespace yah::model

#endif  // YAH_MODEL_LOOM_RUNTIME_HPP_