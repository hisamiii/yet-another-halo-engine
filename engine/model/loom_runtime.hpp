// Thin RAII wrappers over the HRX native API: device, stream, executable, buffer, and a dispatch helper.
#ifndef YAH_MODEL_LOOM_RUNTIME_HPP_
#define YAH_MODEL_LOOM_RUNTIME_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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
  // How many buffers this export's dispatch binds. The GEMM family is not uniform:
  // IQ grid formats bind (weight, grid, [ksigns], input, wstage, ostage, out): 7 for iq3xxs/iq2xxs/iq2xs, 6 for iq3s.
  // Every other format binds (weight, input, wstage, ostage, out) = 5.
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

  // Import an external host pointer (e.g. a GGUF mmap window) as an HRX buffer.
  // The caller must keep the mapping alive while the buffer is in use.
  [[nodiscard]] LoomBuffer Import(void* host_ptr, size_t bytes) {
    LoomBuffer buffer;
    buffer.size = bytes;
    hrx_buffer_params_t params = {};
    params.type = HRX_MEMORY_TYPE_DEVICE_VISIBLE;
    params.access = HRX_MEMORY_ACCESS_READ;
    params.usage = HRX_BUFFER_USAGE_DEFAULT;
    params.queue_affinity = 0;
    LoomCheck(hrx_allocator_import_buffer(hrx_device_allocator(device_), params, host_ptr, bytes, &buffer.handle),
              "hrx_allocator_import_buffer");
    return buffer;
  }

  void H2D(const LoomBuffer& buffer, const void* host, size_t bytes, size_t offset = 0) {
    LoomCheck(hrx_synchronous_h2d(device_, host, buffer.handle, offset, bytes), "hrx_synchronous_h2d");
  }
  void D2H(const LoomBuffer& buffer, void* host, size_t bytes, size_t offset = 0) {
    LoomCheck(hrx_synchronous_d2h(device_, buffer.handle, offset, host, bytes), "hrx_synchronous_d2h");
  }

  void Dispatch(const LoomExecutable& executable, uint32_t ordinal, const hrx_dispatch_config_t& config,
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

  // The next Dispatch may overlap the one after it (no trailing ordering barrier). Needs the local libhrx flag
  // HRX_DISPATCH_FLAG_NO_ORDERING_BARRIER (bit 2, not upstream); on stock HRX dispatches stay ordered.
  void NoBarrierNext() { next_flags_ = 1u << 2; }
  // Sleep-poll synchronize: with us > 0, poll an event at the stream tail and sleep us between checks.
  // The runtime's blocking wait busy-polls a host core (ROCr); a long queued run (the prefill) opts in.
  // Off by default: on short single waits the sleep slack shows in the timing.
  void SetSleepSync(long us) { sleep_us_ = us; }
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
  void Record(LoomEvent& event) { LoomCheck(hrx_event_record(event.handle, stream_), "hrx_event_record"); }
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
  long sleep_us_ = 0;
  bool initialized_ = false;
};

}  // namespace yah::model

#endif  // YAH_MODEL_LOOM_RUNTIME_HPP_