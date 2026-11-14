// NPU (XDNA2) GEMMs beside the GPU: libamdf for the device, the HRX xdna loader for .xdna images.
// Operands live in HRX device buffers (full GPU bandwidth), exported once as a DMA-BUF through HSA and imported per binding view into the NPU.
// Gated images only (gen_npu_gemm GATE): the host submits a chunk's jobs at once.
// The NPU waits for each job's ready word itself, and a reaper thread retires the commands (NpuSplit: the protocol).
// Needs HRX patches 0013 / 0014 (see engine/hrx/patches).
#ifndef YAH_MODEL_LOOM_NPU_HPP_
#define YAH_MODEL_LOOM_NPU_HPP_

#include <dlfcn.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "amdf/amdf.h"
#include "amdf/xdna.h"
#include "experimental/xdna/amdf_status.h"
#include "experimental/xdna/executable.h"
#include "iree/base/byte_sequence.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"
#include "model/loom_runtime.hpp"
#include "model/npu_prof.hpp"
#include "model/npu_split.hpp"

namespace yah::model {

inline void NpuCheck(iree_status_t status, const char* what) {
  if (iree_status_is_ok(status)) return;
  char* text = nullptr;
  iree_host_size_t length = 0;
  std::string message = std::string("npu: ") + what;
  const iree_allocator_t allocator = iree_allocator_system();
  if (iree_status_to_string(status, &allocator, &text, &length)) {
    message += ": " + std::string(text, length);
    iree_allocator_free(iree_allocator_system(), text);
  }
  iree_status_free(status);
  throw LoomError(message);
}
#define YAH_AMDF(expr, what) NpuCheck(IREE_HAL_AMD_STATUS_FROM_AMDF((expr), what), what)

class LoomNpu {
 public:
  // A GPU buffer shared with the NPU (views of it imported per binding through its DMA-BUF).
  struct Shared {
    LoomBuffer gpu;
    void* device = nullptr;
    std::size_t bytes = 0;
    int fd = -1;
    std::uint64_t fd_offset = 0;
    void* host = nullptr;              // host pages (CreateShared host), else none
    amdf_memory_t* native = nullptr;   // their libamdf registration, bound directly
    bool external = false;             // RegisterHost: the caller's pages (no HRX import, not unmapped)
  };
  struct View {
    Shared* shared;
    std::size_t offset, length;
  };
  // One loaded instance of an image with its bindings and native command storage.
  struct Kernel {
    const void* image_key = nullptr;
    std::string path;
    iree_hal_amd_xdna_image_t* image = nullptr;
    uint32_t entry_ordinal = 0;
    // Decoder-column images (gen_npu_gemm dcol 2) share all array state but their decoder tiles: family names the set
    // of such images (nullptr: the image alone), dec the decoder format, swap the image whose setup re-programs only
    // the decoder tiles (LOOM_EXP_SETUP_TILES; spliced into a command where the format changes).
    const void* family = nullptr;
    int dec = -1;
    const Kernel* swap = nullptr;
    // the bytes of first bound to a scratch C (the array setup and one ungated call, the state the gate expects before
    // its first job; a gfirst set's: the setup alone), spliced in where a job's family differs from the array's
    // (FusedCommand); empty: none. A copy, not a second instance: the NPU's command memory holds ~90 instances
    // (hardware.md)
    std::vector<std::uint8_t> setup;
    amdf_xdna_kernel_command_t first{};
    // first sets the array up and runs one whole call (any data; gfirst: no call). gate opens a job once its ready word
    // reaches the job's gate value; done writes the done word after the job's last C (gen_npu_gemm GATE).
    // Streamed (seven invocations): push[parity] queues a call and wait retires the oldest queued one; lead is the even
    // push with its weight fills unpaced, for the first call of a job (nothing computes yet that pacing would
    // protect). Not streamed (four invocations, one replay group, HRX patch 0017): call runs one whole call.
    bool streamed = true;
    amdf_xdna_kernel_command_t push[2]{}, wait{}, lead{}, gate{}, done{}, call{};
    std::vector<iree_hal_amd_xdna_executable_storage_t> storage;
    std::vector<amdf_host_mapping_t*> storage_maps;
    std::vector<iree_hal_buffer_t*> buffers;
  };
  static const void* FamilyOf(const Kernel* k) { return k->family ? k->family : k->image_key; }
  // One call: its kernel instance's invocation bytes as bound for it (an instance serves many calls: Rebind).
  struct Call {
    const Kernel* k = nullptr;
    std::vector<std::uint8_t> push[2], wait, lead, gate, done, call;
  };

  LoomNpu(LoomDevice& gpu, std::uint32_t columns) : gpu_(gpu), columns_(columns) {
    YAH_AMDF(amdf_query_api(AMDF_ABI_VERSION_LATEST, AMDF_ABI_VERSION_LATEST, &api_), "query_api");
    const void* ext = nullptr;
    YAH_AMDF(api_->query_extension(AMDF_EXTENSION_XDNA, AMDF_XDNA_EXTENSION_VERSION_1,
                                   AMDF_XDNA_EXTENSION_VERSION_LATEST, &ext),
             "query_extension(XDNA)");
    xdna_ = static_cast<const amdf_xdna_api_t*>(ext);
    amdf_instance_create_info_t ici{};
    ici.type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.structure_size = sizeof(ici);
    YAH_AMDF(api_->instance_create(&ici, &instance_), "instance_create");
    SelectScope();
    OpenEndpoint();
    CreateDevice();
    // HRX's own libhsa (already loaded) exports its allocations as DMA-BUFs.
    void* hsa = dlopen("libhsa-runtime64.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (hsa) export_dmabuf_ = reinterpret_cast<ExportFn>(dlsym(hsa, "hsa_amd_portable_export_dmabuf"));
    if (!export_dmabuf_) throw LoomError("npu: hsa_amd_portable_export_dmabuf not found");
    reaper_ = std::thread([this] { Reaper(); });
    attn_thread_ = std::thread([this] { AttnSubmitter(); });
    packets_ = &CreateShared(kPacketBytes, true);   // here: allocating it later would wait for the GPU mid-forward
  }
  LoomNpu(const LoomNpu&) = delete;
  LoomNpu& operator=(const LoomNpu&) = delete;
  ~LoomNpu() {
    // no GPU work may still read the shared buffers (flag words, C) once they are freed: on this GPU a read of freed
    // memory hangs the shader and the GPU's scheduler
    try {
      gpu_.Synchronize();
    } catch (...) {
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    reap_cv_.notify_all();
    attn_cv_.notify_all();
    if (attn_thread_.joinable()) attn_thread_.join();
    for (const AttnPending& q : attn_queue_) Fail({q.job});
    if (reaper_.joinable()) reaper_.join();
    // the array first: after its last job a gated image's head keeps polling its flag words (and writing its tick
    // scratch) for the rest of its supply, and streamed fills stay armed; with the IOMMU in passthrough a page freed
    // under a live context is written by the NPU after the kernel reused it (the GPU wedged after runs, 2026-10-09)
    if (queue_) api_->kernel_queue_destroy(queue_), queue_ = nullptr;
    if (context_) xdna_->context_destroy(context_), context_ = nullptr;
    for (auto& a : arenas_) {
      api_->host_mapping_destroy(a.map);
      api_->memory_destroy(a.storage.memory);
    }
    for (auto& [memory, map] : scripts_) {
      api_->host_mapping_destroy(map);
      api_->memory_destroy(memory);
    }
    for (auto& k : kernels_) {
      for (auto* m : k->storage_maps) api_->host_mapping_destroy(m);
      for (auto& s : k->storage) api_->memory_destroy(s.memory);
      for (auto* b : k->buffers) iree_hal_buffer_release(b);
    }
    for (auto& [key, m] : imports_) api_->memory_destroy(m);
    for (auto& [key, image] : images_) iree_hal_amd_xdna_image_destroy(image);
    for (auto& s : shared_) {
      if (s->external) {
        api_->memory_destroy(s->native);
      } else if (s->host) {   // the HRX import first, then the registration, then the pages
        s->gpu.reset();
        api_->memory_destroy(s->native);
        munmap(s->host, s->bytes);
      } else {
        close(s->fd);
      }
    }
    if (queue_) api_->kernel_queue_destroy(queue_);
    if (context_) xdna_->context_destroy(context_);
    if (device_) api_->device_destroy(device_);
    if (endpoint_) api_->endpoint_close(endpoint_);
    if (instance_) api_->instance_destroy(instance_);
  }

  // A zeroed GPU buffer the NPU can bind views of.
  // host: anonymous host pages registered with libamdf and imported into HRX instead of a device buffer the NPU imports.
  // C uses them: with all shared buffers in device memory, deferred data-fabric machine checks (some corrupting an NPU
  // output) were more frequent; their trigger is the SMU switching fabric / memory clocks during NPU work (results.md).
  Shared& CreateShared(std::size_t bytes, bool host = false,
                       std::source_location loc = std::source_location::current()) {
    AllocLog(host ? "npuhost" : "npudev", bytes, loc);
    auto s = std::make_unique<Shared>();
    s->bytes = Align(bytes);
    if (host) {
      s->host = mmap(nullptr, s->bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
      if (s->host == MAP_FAILED) throw LoomError("npu: mmap");
      amdf_memory_create_info_t ci{};
      ci.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
      ci.structure_size = sizeof(ci);
      ci.memory_profile_ordinal = RegisterProfile();
      ci.access_count = 1;
      ci.accesses = &access_;
      ci.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
      ci.byte_length = s->bytes;
      ci.minimum_alignment = 4096;
      ci.registered_host_pointer = s->host;
      ci.registered_host_cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
      YAH_AMDF(api_->memory_create(scope_, &ci, &s->native), "memory_create(registered host)");
      s->device = s->host;
      s->gpu.size = s->bytes;
      hrx_buffer_params_t params = {};
      params.type = HRX_MEMORY_TYPE_DEVICE_VISIBLE;
      params.access = HRX_MEMORY_ACCESS_READ | HRX_MEMORY_ACCESS_WRITE;
      params.usage = HRX_BUFFER_USAGE_DEFAULT;
      LoomCheck(hrx_allocator_import_buffer(hrx_device_allocator(gpu_.device()), params, s->host, s->bytes, &s->gpu.handle),
                "hrx_allocator_import_buffer(host)");
      gpu_.Fill(s->gpu, 0);
      gpu_.Synchronize();
      shared_.push_back(std::move(s));
      return *shared_.back();
    }
    s->gpu = gpu_.Allocate(s->bytes);
    gpu_.Fill(s->gpu, 0);
    gpu_.Synchronize();
    LoomCheck(hrx_buffer_get_device_ptr(s->gpu.handle, &s->device), "hrx_buffer_get_device_ptr");
    if (export_dmabuf_(s->device, s->bytes, &s->fd, &s->fd_offset) != 0) throw LoomError("npu: dma-buf export");
    shared_.push_back(std::move(s));
    return *shared_.back();
  }
  // Shared buffers round to pages.
  [[nodiscard]] static std::size_t Align(std::size_t bytes) { return (bytes + 4095) / 4096 * 4096; }
  // The caller's page-aligned host pages (e.g. raw GGUF weight rows) registered for NPU reads; they stay the caller's.
  // libamdf pins them for writing, so they must be writable (a private file mapping gets copies of the pages it pins).
  Shared& RegisterHost(void* pages, std::size_t bytes) {
    auto s = std::make_unique<Shared>();
    s->bytes = bytes;
    s->external = true;
    amdf_memory_create_info_t ci{};
    ci.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    ci.structure_size = sizeof(ci);
    ci.memory_profile_ordinal = RegisterProfile();
    ci.access_count = 1;
    ci.accesses = &access_;
    ci.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    ci.byte_length = bytes;
    ci.minimum_alignment = 4096;
    ci.registered_host_pointer = pages;
    ci.registered_host_cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
    YAH_AMDF(api_->memory_create(scope_, &ci, &s->native), "memory_create(registered weights)");
    s->device = pages;
    shared_.push_back(std::move(s));
    return *shared_.back();
  }

  // A new instance of the image at path (entry), its bindings the given views.
  Kernel& Load(const std::string& path, const std::string& entry, const std::vector<View>& views) {
    iree_hal_amd_xdna_image_t* image = Image(path);
    uint32_t entry_ordinal = 0;
    NpuCheck(iree_hal_amd_xdna_image_find_entry(image, iree_make_string_view(entry.data(), entry.size()),
                                                &entry_ordinal),
             "image_find_entry");
    const iree_hal_amd_xdna_image_tables_t* tables = iree_hal_amd_xdna_image_tables(image);
    const iree_xdna_elf_entry_record_t rec = iree_hal_amd_xdna_image_tables_entry(tables, entry_ordinal);
    if (rec.binding_count != views.size()) throw LoomError("npu: " + path + ": binding count");
    if (rec.invocation_count != 7 && rec.invocation_count != 4)
      throw LoomError("npu: " + path + " has no gate (re-emit it)");
    auto k = std::make_unique<Kernel>();
    k->image_key = image;
    k->path = path;
    k->image = image;
    k->entry_ordinal = entry_ordinal;
    k->streamed = rec.invocation_count == 7;
    LoadStorage(*k, views);
    kernels_.push_back(std::move(k));
    return *kernels_.back();
  }
  // k's command storage: the image's commands loaded into it, bound to views, and k's invocations in it.
  void LoadStorage(Kernel& k, const std::vector<View>& views) {
    const iree_hal_amd_xdna_image_tables_t* tables = iree_hal_amd_xdna_image_tables(k.image);
    const iree_xdna_elf_entry_record_t rec = iree_hal_amd_xdna_image_tables_entry(tables, k.entry_ordinal);
    k.storage.resize(rec.allocation_use_count);
    k.storage_maps.resize(rec.allocation_use_count);
    for (uint32_t i = 0; i < rec.allocation_use_count; ++i) {
      const uint32_t ord = iree_hal_amd_xdna_image_tables_allocation_use(tables, rec.first_allocation_use + i);
      AllocateStorage(iree_hal_amd_xdna_image_tables_allocation(tables, ord), &k.storage[i], &k.storage_maps[i]);
    }
    const auto n = static_cast<iree_host_size_t>(k.storage.size());
    NpuCheck(iree_hal_amd_xdna_executable_load(k.image, k.entry_ordinal, n, k.storage.data()), "executable_load");
    Rebind(k, views);
    NpuCheck(
        iree_hal_amd_xdna_executable_query_continuation(k.image, k.entry_ordinal, n, k.storage.data(), false, &k.first),
        "query_invocation");
    amdf_xdna_kernel_command_t* const streamed[] = {&k.push[0], &k.push[1], &k.wait, &k.lead, &k.gate, &k.done};
    amdf_xdna_kernel_command_t* const single[] = {&k.call, &k.gate, &k.done};
    for (uint32_t i = 0; i + 1 < rec.invocation_count; ++i)
      NpuCheck(iree_hal_amd_xdna_executable_query_invocation_ordinal(k.image, k.entry_ordinal, n, k.storage.data(),
                                                                     i + 1, k.streamed ? streamed[i] : single[i]),
               "query_invocation_ordinal");
  }
  // Frees k's command storage (LoadStorage brings it back). Only for an instance whose commands are all copied out
  // (Snapshot, its setup copy): no command in it was submitted, so the NPU never read these pages.
  void UnloadStorage(Kernel& k) {
    for (std::size_t i = 0; i < k.storage.size(); ++i) {
      api_->host_mapping_destroy(k.storage_maps[i]);
      api_->memory_destroy(k.storage[i].memory);
      storage_bytes_ -= storage_size_.at(k.storage[i].memory);
      storage_size_.erase(k.storage[i].memory);
      --storage_count_;
    }
    k.storage.clear(), k.storage_maps.clear();
    k.first = {}, k.push[0] = {}, k.push[1] = {}, k.wait = {}, k.lead = {}, k.gate = {}, k.done = {}, k.call = {};
  }
  [[nodiscard]] static bool Loaded(const Kernel& k) { return !k.storage.empty(); }

  // The executable bindings of views for instance k (k.buffers keeps the range wrappers). DMA-BUF views are imported
  // once per view; registered host pages bind at the view's offset.
  std::vector<iree_hal_amd_xdna_executable_binding_t> Bindings(Kernel& k, const std::vector<View>& views) {
    const iree_hal_amd_xdna_image_tables_t* tables = iree_hal_amd_xdna_image_tables(k.image);
    const iree_xdna_elf_entry_record_t rec = iree_hal_amd_xdna_image_tables_entry(tables, k.entry_ordinal);
    if (rec.binding_count != views.size()) throw LoomError("npu: binding count");
    std::vector<iree_hal_amd_xdna_executable_binding_t> bindings(views.size());
    for (std::size_t i = 0; i < views.size(); ++i) {
      const View& v = views[i];
      if (v.offset + v.length > v.shared->bytes) throw LoomError("npu: bad view");
      const iree_xdna_elf_binding_record_t contract =
          iree_hal_amd_xdna_image_tables_binding(tables, rec.first_binding + static_cast<uint32_t>(i));
      amdf_memory_t* mem = v.shared->native;
      std::uint64_t addr = 0, mem_off = 0;
      if (mem) {   // registered host pages: bound at the view's offset
        mem_off = v.offset;
      } else if (const auto it = imports_.find({v.shared, v.offset, v.length}); it != imports_.end()) {
        mem = it->second;
      } else {
        amdf_memory_import_info_t ii{};
        ii.type = AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO;
        ii.structure_size = sizeof(ii);
        ii.memory_profile_ordinal = profile_.ordinal;
        ii.access_count = 1;
        ii.minimum_alignment = contract.minimum_alignment;
        ii.accesses = &access_;
        amdf_external_memory_t em{};   // borrowed: the fd stays ours
        em.type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD;
        em.payload.file_descriptor = v.shared->fd;
        em.source_byte_offset = v.shared->fd_offset + v.offset;
        em.byte_length = v.length;
        YAH_AMDF(api_->memory_import(scope_, &ii, &em, &mem), "memory_import");
        imports_[{v.shared, v.offset, v.length}] = mem;
      }
      YAH_AMDF(api_->memory_query_address(mem, 0, AMDF_MEMORY_ADDRESS_XDNA_DMA, &addr), "memory_query_address");
      addr += mem_off;
      // executable_bind checks this wrapper's range and access only; the span is never dereferenced (the address is
      // device_address). Wrappers start 64-byte aligned: raw weight rows need not.
      std::uint8_t* span = static_cast<std::uint8_t*>(v.shared->device) + v.offset;
      const std::size_t lead = reinterpret_cast<std::uintptr_t>(span) & 63;
      iree_hal_buffer_t* buffer = nullptr;
      NpuCheck(iree_hal_heap_buffer_wrap(
                   iree_hal_buffer_placement_undefined(),
                   IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                       IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
                   IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, IREE_HAL_BUFFER_USAGE_STORAGE,
                   v.length + lead, iree_make_byte_span(span - lead, v.length + lead),
                   iree_hal_buffer_release_callback_null(), iree_allocator_system(), &buffer),
               "heap_buffer_wrap");
      k.buffers.push_back(buffer);
      bindings[i].buffer_ref = iree_hal_make_buffer_ref(buffer, 0, v.length + lead);
      bindings[i].memory = mem;
      bindings[i].memory_byte_offset = mem_off;
      bindings[i].device_address = addr;
    }
    return bindings;
  }
  // Bind instance k to other views (one instance of an image serves many calls: its invocation bytes are copied per
  // call, Snapshot).
  void Rebind(Kernel& k, const std::vector<View>& views) {
    const std::vector<iree_hal_amd_xdna_executable_binding_t> bindings = Bindings(k, views);
    const auto n = static_cast<iree_host_size_t>(k.storage.size());
    NpuCheck(iree_hal_amd_xdna_executable_bind(k.image, k.entry_ordinal, n, k.storage.data(), bindings.size(),
                                               bindings.data()),
             "executable_bind");
    for (uint32_t i = 0; i < n; ++i)
      YAH_AMDF(api_->host_mapping_cache_control(k.storage_maps[i], AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                                k.storage[i].mapping.data_length),
               "host_mapping_cache_control(storage)");
  }
  // The bytes of k's first as bound now.
  std::vector<std::uint8_t> FirstBytes(const Kernel& k) const {
    const std::uint8_t* p = CommandBytes(k, k.first);
    return {p, p + k.first.byte_length};
  }
  // The invocation bytes of k as bound now.
  Call Snapshot(const Kernel& k) const {
    Call c;
    c.k = &k;
    auto copy = [&](const amdf_xdna_kernel_command_t& cmd, std::vector<std::uint8_t>& out) {
      if (!cmd.byte_length) return;
      const std::uint8_t* src = CommandBytes(k, cmd);
      out.assign(src, src + cmd.byte_length);
    };
    copy(k.push[0], c.push[0]), copy(k.push[1], c.push[1]), copy(k.wait, c.wait), copy(k.lead, c.lead);
    copy(k.gate, c.gate), copy(k.done, c.done), copy(k.call, c.call);
    return c;
  }

  // A gated image's job (NpuSplit::Enqueue): its calls, done word and gate value; the NPU waits for its ready word
  // itself and writes the gate value to *done.
  struct GatedJob {
    std::vector<const Call*> calls;
    volatile std::uint32_t* done = nullptr;
    std::uint32_t gate = 0;
    std::string tag;
    std::size_t host_calls = 0;  // a host-gated job (EnqueueAttn): its calls; the reaper stores done itself
  };

  // NPU attention (tools/gen_npu_attn.py). The kernel takes all 32 cores, so no core can wait for the GPU: the host
  // gates it. Each call's 4 streams read the call's payloads (its own views) and then the KV head's records (views every
  // call of the head shares); a call writes its own outputs. The image's first invocation sets the array up and runs
  // one call, its continuation runs one more call on the array as the last call left it.
  struct AttnImage {
    Kernel* k = nullptr;
    std::uint32_t nb_built = 0;
    // byte offsets of the 4 record descriptors' lengths (gen_npu_attn.stream_words) in the first and continuation
    std::vector<std::size_t> first_lengths, cont_lengths;
  };
  // One call: the image's invocation bytes as bound for it.
  struct AttnCall {
    const AttnImage* image = nullptr;
    std::vector<std::uint8_t> first, cont;
  };
  // A record descriptor's length in i32 words for nb bodies (nb + 1 records of 64 bfp16 fragments).
  static constexpr std::uint32_t AttnLength(std::uint32_t nb) { return (nb + 1) * 64 * 72 / 4; }
  // An instance of the attention image at path built for nb_built bodies, bound to views (any call's).
  AttnImage LoadAttn(const std::string& path, std::uint32_t nb_built, const std::vector<View>& views) {
    iree_hal_amd_xdna_image_t* image = Image(path);
    uint32_t entry_ordinal = 0;
    NpuCheck(iree_hal_amd_xdna_image_find_entry(image, iree_make_cstring_view("probe"), &entry_ordinal),
             "image_find_entry");
    auto k = std::make_unique<Kernel>();
    k->image_key = image;
    k->path = path;
    k->image = image;
    k->entry_ordinal = entry_ordinal;
    k->streamed = false;
    const iree_hal_amd_xdna_image_tables_t* tables = iree_hal_amd_xdna_image_tables(image);
    const iree_xdna_elf_entry_record_t rec = iree_hal_amd_xdna_image_tables_entry(tables, entry_ordinal);
    k->storage.resize(rec.allocation_use_count);
    k->storage_maps.resize(rec.allocation_use_count);
    for (uint32_t i = 0; i < rec.allocation_use_count; ++i) {
      const uint32_t ord = iree_hal_amd_xdna_image_tables_allocation_use(tables, rec.first_allocation_use + i);
      AllocateStorage(iree_hal_amd_xdna_image_tables_allocation(tables, ord), &k->storage[i], &k->storage_maps[i]);
    }
    const auto n = static_cast<iree_host_size_t>(k->storage.size());
    NpuCheck(iree_hal_amd_xdna_executable_load(image, entry_ordinal, n, k->storage.data()), "executable_load");
    Rebind(*k, views);
    NpuCheck(
        iree_hal_amd_xdna_executable_query_continuation(image, entry_ordinal, n, k->storage.data(), false, &k->first),
        "query_continuation(first)");
    NpuCheck(
        iree_hal_amd_xdna_executable_query_continuation(image, entry_ordinal, n, k->storage.data(), true, &k->call),
        "query_continuation");
    AttnImage a;
    a.k = k.get();
    a.nb_built = nb_built;
    kernels_.push_back(std::move(k));
    auto find = [&](const amdf_xdna_kernel_command_t& c, std::vector<std::size_t>& at) {
      const std::uint8_t* p = CommandBytes(*a.k, c);
      for (std::size_t i = 16; i + 4 <= c.byte_length; i += 4) {
        std::uint32_t w = 0;
        std::memcpy(&w, p + i, 4);
        if (w == AttnLength(nb_built)) at.push_back(i);
      }
    };
    find(a.k->first, a.first_lengths), find(a.k->call, a.cont_lengths);
    if (a.cont_lengths.size() != 4 || (a.first_lengths.size() != 4 && !a.first_lengths.empty()))
      throw LoomError("npu: " + path + ": record lengths for nb " + std::to_string(nb_built) + " not found");
    return a;
  }
  // The invocation bytes of a call of a bound to views.
  AttnCall BindAttn(const AttnImage& a, const std::vector<View>& views) {
    Rebind(*a.k, views);
    AttnCall c;
    c.image = &a;
    const std::uint8_t* f = CommandBytes(*a.k, a.k->first);
    const std::uint8_t* r = CommandBytes(*a.k, a.k->call);
    c.first.assign(f, f + a.k->first.byte_length);
    c.cont.assign(r, r + a.k->call.byte_length);
    return c;
  }
  // A layer's calls of one image as one command (its first call sets the array up if another image ran since), nb
  // bodies each (<= the image's nb_built; the payloads carry nb too). The host thread submits it once *ready >=
  // ready_value; the reaper stores done_value to *done after it (done_value | kGateFailed after a failure, or without
  // ready once the GPU is done: Drain). Commands run in enqueue order: no gated job may queue behind an unsubmitted one
  // (EnqueueGated checks).
  struct AttnJob {
    std::vector<const AttnCall*> calls;
    std::uint32_t nb = 0;
    volatile std::uint32_t* ready = nullptr;
    std::uint32_t ready_value = 0;
    volatile std::uint32_t* done = nullptr;
    std::uint32_t done_value = 0;
    std::string tag;
  };
  void EnqueueAttn(const AttnJob& job) {
    CheckHealth();
    const auto t0 = std::chrono::steady_clock::now();
    const AttnImage& a = *job.calls.at(0)->image;
    if (job.nb > a.nb_built || job.nb < 32 || job.nb % 16) throw LoomError("npu: attention nb");
    bool resident = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      const bool idle = retired_ == submitted_ && attn_queue_.empty() && active_.time_since_epoch().count() &&
                        t0 - active_ > suspend_after_;
      resident = resident_ == a.k->image_key && !idle;
      resident_ = a.k->image_key, resident_dec_ = -1;
    }
    if (!resident) program_.Forget(), ++epoch_;  // a full setup outside FusedCommand
    std::vector<std::uint8_t> bytes;
    std::uint32_t ops = 0;
    auto append = [&](const std::vector<std::uint8_t>& b, const std::vector<std::size_t>& lengths) {
      std::uint32_t n = 0, size = 0;
      std::memcpy(&n, b.data() + 8, 4), std::memcpy(&size, b.data() + 12, 4);
      if (size != b.size()) throw LoomError("npu: unexpected command header");
      const std::size_t at = bytes.empty() ? 0 : bytes.size() - 16;
      if (bytes.empty()) bytes.assign(b.begin(), b.begin() + 16);
      bytes.insert(bytes.end(), b.begin() + 16, b.end());
      ops += n;
      for (const std::size_t i : lengths) {
        std::uint32_t w = 0;
        std::memcpy(&w, &bytes[at + i], 4);
        if (w != AttnLength(a.nb_built)) throw LoomError("npu: attention record length moved");
        w = AttnLength(job.nb);
        std::memcpy(&bytes[at + i], &w, 4);
      }
    };
    for (std::size_t i = 0; i < job.calls.size(); ++i) {
      const AttnCall& c = *job.calls[i];
      if (c.image != &a) throw LoomError("npu: an attention job mixes images");
      if (i == 0 && !resident)
        append(c.first, a.first_lengths);
      else
        append(c.cont, a.cont_lengths);
    }
    const std::uint32_t size = static_cast<std::uint32_t>(bytes.size());
    std::memcpy(&bytes[8], &ops, 4), std::memcpy(&bytes[12], &size, 4);
    AttnPending p;
    p.command = Place(bytes);
    p.ready = job.ready, p.ready_value = job.ready_value;
    p.job = GatedJob{{}, job.done, job.done_value, job.tag, job.calls.size()};
    p.t0 = t0;
    {
      std::lock_guard<std::mutex> lock(mu_);
      attn_queue_.push_back(std::move(p));
    }
    attn_cv_.notify_one();
  }
  // Gated jobs as one command, submitted now (no host between the jobs). The first job of an image first runs its setup
  // as its own command (one ungated call, any data, unless gfirst; the host waits for it once), after which the gate
  // waits for job 1.
  // After a failure the reaper stores gate | kGateFailed to the done words of this and every later command's jobs, so
  // the GPU's waits end and report it.
  // fresh: set the first job's image up even if its family is resident (a chunk's first command: the gate's job
  // sequence restarts there, NpuSplit::BeginChunk).
  void EnqueueGated(std::vector<GatedJob> jobs, bool fresh = false) {
    CheckHealth();
    AttnQueued();
    const auto t0 = std::chrono::steady_clock::now();
    {
      // Idle past the device's autosuspend delay, the NPU may have been powered down: the array kept nothing, so the
      // next setups write whole programs (as after a failure)
      std::lock_guard<std::mutex> lock(mu_);
      if (retired_ == submitted_ && active_.time_since_epoch().count() && t0 - active_ > suspend_after_) {
        resident_ = nullptr;
        resident_dec_ = -1;
        program_lost_ = true;
        ++epoch_;
      }
    }
    const Kernel* k = jobs[0].calls[0]->k;
    const void* family = k->family ? k->family : k->image_key;
    bool resident = false;
    int dec = -1;
    {
      std::lock_guard<std::mutex> lock(mu_);
      resident = resident_ == family;
      dec = resident_dec_;
    }
    if ((!resident || fresh) && !k->setup.empty()) {
      // FusedCommand splices the setup twin in front of the first job
      std::lock_guard<std::mutex> lock(mu_);
      family = nullptr;
    } else if (!resident) {
      program_.Forget();   // a full setup outside FusedCommand
      ++epoch_;
      YAH_AMDF(api_->kernel_queue_wait(queue_, Submit(k->first), AMDF_TIMEOUT_INFINITE, 0), "kernel_queue_wait(setup)");
      std::lock_guard<std::mutex> lock(mu_);
      resident_ = family;
      dec = resident_dec_ = k->dec;
    }
    std::vector<std::vector<const Call*>> runs;
    for (const GatedJob& g : jobs) runs.push_back(g.calls);
    // commands run in submission order: the next one starts with this one's last family and decoder
    const Fused fused = FusedCommand(runs, dec, family);
    const amdf_xdna_kernel_command_t command = fused.command;
    {
      std::lock_guard<std::mutex> lock(mu_);
      resident_dec_ = fused.dec;
      resident_ = fused.family;
    }
    if (recording_) {
      recording_->jobs.push_back(jobs);
      recording_->family = fused.family, recording_->dec = fused.dec;
    }
    const std::uint64_t submission = Submit(command);
    {
      std::lock_guard<std::mutex> lock(mu_);
      pending_.push_back({submission, std::move(jobs), t0});
      ++submitted_;
    }
    reap_cv_.notify_one();
  }
  // Command replay. A command's bytes depend only on its jobs and on the array state it starts from (DeltaSetup's
  // program memory and read-only data, the family and decoder set up), and a sequence of commands leaves the state of
  // everything its setups wrote set by them alone. So a chunk whose jobs repeat the previous chunk's (with no forgotten
  // program state since) builds the same commands as the next such chunk: Record keeps the commands EnqueueGated builds
  // from now on, Seal copies them into their own command storage, and Replay submits them again as they are.
  struct Script {
    std::vector<std::vector<std::uint8_t>> bytes;   // the recorded commands, until Seal
    std::vector<std::vector<GatedJob>> jobs;        // per command
    const void* family = nullptr;                   // the family and decoder set up after the last command
    int dec = -1;
    iree_hal_amd_xdna_executable_storage_t storage{};
    amdf_host_mapping_t* map = nullptr;
    std::vector<amdf_xdna_kernel_command_t> commands;
  };
  // Program state forgotten so far (a failed command, a setup outside FusedCommand): scripts recorded before a change
  // start from a state the array no longer holds.
  [[nodiscard]] std::uint64_t Epoch() const { return epoch_.load(); }
  void Record(Script* s) { recording_ = s; }
  void Seal(Script& s) {
    recording_ = nullptr;
    std::size_t size = 0;
    for (const auto& b : s.bytes) size += (b.size() + kCommandAlign - 1) / kCommandAlign * kCommandAlign;
    iree_xdna_elf_allocation_record_t req{};
    req.domain = IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND;
    req.byte_length = size;
    req.alignment = kCommandAlign;
    AllocateStorage(req, &s.storage, &s.map);
    scripts_.push_back({s.storage.memory, s.map});
    std::size_t at = 0;
    for (const auto& b : s.bytes) {
      std::memcpy(static_cast<std::uint8_t*>(s.storage.mapping.data) + at, b.data(), b.size());
      amdf_xdna_kernel_command_t c{};
      c.memory = s.storage.memory;
      c.access_ordinal = s.storage.access_ordinal;
      c.byte_offset = s.storage.memory_byte_offset + at;
      c.byte_length = b.size();
      s.commands.push_back(c);
      at += (b.size() + kCommandAlign - 1) / kCommandAlign * kCommandAlign;
    }
    YAH_AMDF(api_->host_mapping_cache_control(s.map, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, size),
             "host_mapping_cache_control(script)");
    s.bytes.clear();
  }
  // Submits a sealed script's commands (the array in the state its recording started from); queued counts the jobs
  // submitted.
  void Replay(const Script& s, std::size_t& queued) {
    CheckHealth();
    AttnQueued();
    for (std::size_t i = 0; i < s.commands.size(); ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      const std::uint64_t submission = Submit(s.commands[i]);
      std::lock_guard<std::mutex> lock(mu_);
      pending_.push_back({submission, s.jobs[i], t0});
      ++submitted_, ++commands_.replayed;
      resident_ = s.family, resident_dec_ = s.dec;
      queued += s.jobs[i].size();
      reap_cv_.notify_one();
    }
  }
  void Free(Script& s) {
    Drain();
    if (s.map) {
      api_->host_mapping_destroy(s.map), api_->memory_destroy(s.storage.memory);
      std::erase_if(scripts_, [&](const auto& m) { return m.first == s.storage.memory; });
    }
    s = Script{};
  }
  // Waits until the reaper retired every command, queued attention commands included (stats and failures are in).
  // gpu_done: the GPU has finished all work it was given, so a queued attention command whose ready word is not there
  // never gets it and fails.
  void Drain(bool gpu_done = false) {
    std::unique_lock<std::mutex> lock(mu_);
    attn_gpu_done_ = gpu_done;
    retired_cv_.wait(lock, [this] { return retired_ == submitted_ && attn_queue_.empty(); });
    attn_gpu_done_ = false;
  }
  // Per job tag: jobs, calls.
  struct TagStats {
    std::size_t jobs = 0, calls = 0;
  };
  std::map<std::string, TagStats> Stats() {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
  }
  // Commands: count, NPU time (each from the later of its submission and the previous command's completion, its
  // waits for the GPU included) and the longest (the driver kills a command after amdxdna tdr_timeout_ms, 2000 ms).
  // Host involvement: command bytes, the host's waits for a free arena, the NPU's idle time with no command queued.
  struct CommandStats {
    std::size_t commands = 0, bytes = 0, arena_waits = 0, replayed = 0;
    double busy_ms = 0, max_ms = 0, arena_wait_ms = 0, idle_ms = 0;
  };
  CommandStats Commands() {
    std::lock_guard<std::mutex> lock(mu_);
    return commands_;
  }
  // NPU time of the last retired command (CommandStats).
  [[nodiscard]] double LastCommandMs() const { return last_command_ms_.load(); }
  // One line per job tag (jobs, calls), then the commands (CommandStats).
  void Report(std::FILE* f) {
    for (const auto& [tag, st] : Stats())
      std::fprintf(f, "npu: %-5s %4zu jobs %5zu calls\n", tag.c_str(), st.jobs, st.calls);
    const CommandStats c = Commands();
    std::fprintf(f, "npu: %zu commands, NPU %.1f ms (waits for the GPU included), longest %.1f ms\n", c.commands,
                 c.busy_ms, c.max_ms);
    std::fprintf(f,
                 "npu: %.1f MB of commands, host waits for a free arena %zu (%.1f ms), NPU idle with none queued "
                 "%.1f ms\n",
                 c.bytes / 1e6, c.arena_waits, c.arena_wait_ms, c.idle_ms);
    if (c.replayed) std::fprintf(f, "npu: %zu commands replayed\n", c.replayed);
  }
  // Throws if a command failed.
  NpuProf* Prof() { return prof_.get(); }
  void CheckHealth() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!failure_.empty()) throw LoomError(failure_);
  }

 private:
  // Gated jobs as one command, each [gate c0][lead c0][push c1][wait][push c2][wait] ... [wait][wait][done cN], or for
  // an image that is not streamed [gate c0][call c0] ... [call cN][done cN].
  // Pushes alternate descriptor sets, at most two calls queued: calls overlap, and no command boundary falls while a
  // call's DMA work is in flight (that hung the NPU firmware under DRAM contention). The bodies of the calls' own
  // relocated commands go under one transaction header (libamdf's public format 0.1,
  // AMDF_XDNA_TRANSACTION_FORMAT_VERSION_0_1: operation count at byte 8, byte length at 12).
  // Built per command into a ring of command arenas (allocating per command cost ~1 ms); an arena is reused once its
  // commands retired (family switches splice whole array setups: commands of several MB, and the NPU's command memory
  // is small).
  struct Arena {
    iree_hal_amd_xdna_executable_storage_t storage{};
    amdf_host_mapping_t* map = nullptr;
    std::size_t used = 0;
    std::uint64_t last = 0;   // the last command built into it (its submission count)
  };
  static constexpr std::size_t kArenaBytes = 8u << 20, kCommandAlign = 32768, kArenas = 5;
  static const std::uint8_t* CommandBytes(const Kernel& k, const amdf_xdna_kernel_command_t& c) {
    for (const auto& st : k.storage)
      if (st.memory == c.memory) return static_cast<const std::uint8_t*>(st.mapping.data) + (c.byte_offset - st.memory_byte_offset);
    throw LoomError("npu: command outside its kernel storage");
  }
  // Decoder-column images: where a call's decoder format differs from the one set up, the queued calls drain, the
  // format's swap setup follows and the call starts like a job's first (lead; the parity restarts there). Where a job's
  // family differs from the one set up (the previous job has drained by its done), its image's setup twin runs first.
  // Returns the command and the family and decoder format set up at its end.
  struct Fused {
    amdf_xdna_kernel_command_t command{};
    int dec = -1;
    const void* family = nullptr;
  };
  // The core tiles' program memory as the array holds it once the commands built so far ran: FusedCommand builds in
  // submission order and nothing else writes programs (a failure or a setup outside it forgets them).
  struct ProgramState {
    static constexpr std::uint32_t kCols = 8, kRows = 4, kWords = 0x10000 / 4;   // rows 2-5, [0x20000, 0x30000)
    std::vector<std::uint32_t> word = std::vector<std::uint32_t>(kCols * kRows * kWords);
    std::vector<std::uint8_t> known = std::vector<std::uint8_t>(kCols * kRows * kWords);
    // per core tile, the read-only data blocks (address, bytes) of the last setup that wrote the tile: only that image
    // ran there since, and an image never writes its own read-only data
    std::vector<std::vector<std::pair<std::uint32_t, std::vector<std::uint8_t>>>> rodata =
        std::vector<std::vector<std::pair<std::uint32_t, std::vector<std::uint8_t>>>>(kCols * kRows);
    void Forget() {
      std::fill(known.begin(), known.end(), 0);
      for (auto& r : rodata) r.clear();
    }
  };
  // A setup (transaction) with its program loads cut to the words the array does not hold already and its read-only
  // data loads (decoder tables) dropped where the tile holds them already; every other operation runs in full and in
  // order (zero-filled storages, registers, DMA waits: state the kernels change at run time). Setups are ~0.5 MB,
  // 2/3 of it core programs, and images that follow each other share most of their code. Loom images load data memory
  // only as read-only sections (PROGBITS) and zero-filled storages (NOBITS: all-zero blocks), one block per section.
  //
  // Program words go to the core tiles by control packets (when `packets`, family setups): each 16-byte aligned 4-word
  // chunk a packet [control header, 4 words] that shim MM2S0 streams up lane 5 (no image uses it) to the tile's
  // TILE_CONTROL port. The microcontroller writes ~0.7 GB/s; the shim DMAs stream the packets while it runs the rest
  // of the setup. See ControlPackets.
  std::vector<std::uint8_t> DeltaSetup(const std::uint8_t* src, std::size_t length, bool packets) {
    if (program_lost_.exchange(false)) program_.Forget();
    std::uint32_t n = 0, size = 0;
    std::memcpy(&n, src + 8, 4), std::memcpy(&size, src + 12, 4);
    if (size != length || size < 16) throw LoomError("npu: unexpected setup header");
    std::vector<std::uint8_t> out(src, src + 16);
    std::uint32_t ops = 0;
    std::vector<std::vector<std::pair<std::uint32_t, std::vector<std::uint8_t>>>> rodata(program_.rodata.size());
    std::vector<std::uint8_t> touched(program_.rodata.size());
    PacketSetup ps;
    for (std::size_t o = 16; o < size;) {
      std::uint32_t len = 0, addr = 0;
      switch (src[o]) {   // libamdf transaction 0.1 operations (Loom's xdna_product.c)
        case 0: len = 24; break;
        case 3: len = 28; break;
        case 1: std::memcpy(&len, src + o + 12, 4); break;
        case 128: std::memcpy(&len, src + o + 4, 4); break;
        default: throw LoomError("npu: unknown setup operation " + std::to_string(src[o]));
      }
      if (len < 16 || o + len > size) throw LoomError("npu: setup operation outside its command");
      std::memcpy(&addr, src + o + 8, 4);
      const std::uint32_t col = addr >> 25, row = (addr >> 20) & 31, off = addr & 0xFFFFF;
      if (packets && col < ProgramState::kCols && ps.Take(src + o, len, col, row, off, out, ops)) {
        o += len;
        continue;
      }
      const bool core = col < ProgramState::kCols && row >= 2 && row < 2 + ProgramState::kRows;
      const std::size_t t = core ? col * ProgramState::kRows + row - 2 : 0;
      if (core) touched[t] = 1;
      if (src[o] == 1 && core && off < 0x10000) {   // data memory
        std::vector<std::uint8_t> payload(src + o + 16, src + o + len);
        const bool zero = std::all_of(payload.begin(), payload.end(), [](std::uint8_t b) { return b == 0; });
        const auto& prev = program_.rodata[t];
        const bool held = !zero && std::any_of(prev.begin(), prev.end(), [&](const auto& b) {
          return b.first == addr && b.second == payload;
        });
        if (!zero) rodata[t].emplace_back(addr, std::move(payload));
        if (!held) out.insert(out.end(), src + o, src + o + len), ++ops;
        o += len;
        continue;
      }
      if (src[o] != 1 || col >= ProgramState::kCols || row < 2 || row >= 2 + ProgramState::kRows || off < 0x20000 ||
          off >= 0x30000 || len % 4 || off % 4 || off - 0x20000 + (len - 16) > 0x10000) {
        out.insert(out.end(), src + o, src + o + len), ++ops, o += len;
        continue;
      }
      const std::size_t base = ((col * ProgramState::kRows + row - 2) * ProgramState::kWords) + (off - 0x20000) / 4;
      const std::uint32_t words = (len - 16) / 4;
      const std::uint8_t* w = src + o + 16;
      auto differs = [&](std::uint32_t i) {
        std::uint32_t v;
        std::memcpy(&v, w + 4 * i, 4);
        return !program_.known[base + i] || program_.word[base + i] != v;
      };
      for (std::uint32_t i = 0; i < words;) {
        if (!differs(i)) {
          ++i;
          continue;
        }
        // a run of differing words; gaps under 32 words are written too (a write op costs about as much as 150 bytes)
        std::uint32_t end = i + 1;
        for (std::uint32_t j = end; j < words && j < end + 32; ++j)
          if (differs(j)) end = j + 1;
        auto write = [&](std::uint32_t i0, std::uint32_t i1) {
          if (i0 == i1) return;
          const std::uint32_t bl = 16 + 4 * (i1 - i0), at = addr + 4 * i0;
          const std::size_t h = out.size();
          out.insert(out.end(), src + o, src + o + 16);
          std::memcpy(&out[h + 8], &at, 4), std::memcpy(&out[h + 12], &bl, 4);
          out.insert(out.end(), w + 4 * i0, w + 4 * i1);
          ++ops;
        };
        std::uint32_t ia = i;   // the aligned chunks by packet, the ends by write
        while (ia < end && (addr + 4 * ia) % 16) ++ia;
        const std::uint32_t ie = ia + (end - ia) / 4 * 4;
        if (packets && ie - ia >= 8) {
          write(i, ia), write(ie, end);
          auto& st = ps.stream[col * ProgramState::kRows + row - 2];
          for (std::uint32_t j = ia; j < ie; j += 4) {
            std::uint32_t hdr = ((addr + 4 * j) & 0xFFFFF) | 3u << 20;   // write, 4 beats
            if (!(__builtin_popcount(hdr) & 1)) hdr |= 1u << 31;          // odd parity
            st.push_back(hdr);
            st.resize(st.size() + 4);
            std::memcpy(&st[st.size() - 4], w + 4 * j, 16);
          }
        } else {
          write(i, end);
        }
        for (std::uint32_t j = i; j < end; ++j) {
          std::memcpy(&program_.word[base + j], w + 4 * j, 4);
          program_.known[base + j] = 1;
        }
        i = end;
      }
      o += len;
    }
    for (std::size_t t = 0; t < touched.size(); ++t)
      if (touched[t]) program_.rodata[t] = std::move(rodata[t]);
    if (packets) ControlPackets(ps, out, ops);
    const std::uint32_t bytes = static_cast<std::uint32_t>(out.size());
    std::memcpy(&out[8], &ops, 4), std::memcpy(&out[12], &bytes, 4);
    return out;
  }
  // DeltaSetup's control packets: lane 5 is MM2S0 -> shim north 5 -> memtile south 5 / north 5 -> core south 5, up
  // the column; each core tile takes the packets of its row (packet id) to TILE_CONTROL and passes the rest north.
  // The control port takes one 4-word write per packet: a BD repeat per packet, its iteration stepping the 5-word
  // packets, 64 at most (the iteration wrap). Setup operations that would cut the lane or the token route while the
  // packets stream (switch zero-clears) skip those registers, and the image's own routes up from MM2S0 wait for the
  // packets' end.
  struct PacketSetup {
    std::vector<std::vector<std::uint32_t>> stream =   // per core tile: packets [header, 4 words]
        std::vector<std::vector<std::uint32_t>>(ProgramState::kCols * ProgramState::kRows);
    std::array<std::array<std::uint32_t, 0xC0>, ProgramState::kCols> shim{};   // the image's shim switch (0x3F000)
    std::vector<std::uint8_t> hoist, deferred;   // operations copied to the front, moved behind the packets
    std::uint32_t hoist_ops = 0, deferred_ops = 0;
    static bool Kept(std::uint32_t row, std::uint32_t off) {   // the lane's and the token route's registers
      if (row == 0) return off == 0x3F008 || off == 0x3F044 || off == 0x3F200;   // M2, M17, S0 slot 0
      if (row == 1) return off == 0xB0040 || off == 0xB0130;                     // M16, S12
      return off == 0x3F00C || off == 0x3F048 || off == 0x3F128 || (off >= 0x3F2A0 && off < 0x3F2B0);   // M3 M18 S10
    }
    // Records and reorders one setup operation; true if it took it (written to out or deferred).
    bool Take(const std::uint8_t* op, std::uint32_t len, std::uint32_t col, std::uint32_t row, std::uint32_t off,
              std::vector<std::uint8_t>& out, std::uint32_t& ops) {
      std::uint32_t v = 0;
      if (op[0] == 0) std::memcpy(&v, op + 16, 4);
      if (row == 0 && (op[0] == 0 || op[0] == 1) && off >= 0x3F000 && off < 0x3F300) {
        const std::uint32_t nw = op[0] == 0 ? 1 : (len - 16) / 4;
        for (std::uint32_t q = 0; q < nw && off + 4 * q < 0x3F300; ++q)
          std::memcpy(&shim[col][(off - 0x3F000) / 4 + q], op + 16 + 4 * q, 4);
      }
      // the shim mux / demux, MM2S0's slave port and the token route, ahead of the packets (the first setup's array
      // has none of them)
      if (row == 0 && ((op[0] == 3 && (off == 0x1F000 || off == 0x1F004)) ||
                       (op[0] == 0 && (off == 0x3F008 || off == 0x3F100 || off == 0x3F200 || off == 0x3F114))))
        hoist.insert(hoist.end(), op, op + len), ++hoist_ops;
      if (row == 0 && op[0] == 0 && off < 0x3F058 && off >= 0x3F000 && v == 0x80000005) {   // a route up from MM2S0
        deferred.insert(deferred.end(), op, op + len), ++deferred_ops;
        return true;
      }
      if (op[0] != 1 || row > 1 + ProgramState::kRows || off < 0x3F000 || (row == 1 && off < 0xB0000)) return false;
      const std::uint32_t nw = (len - 16) / 4;
      bool cut = false;
      for (std::uint32_t q = 0; q < nw && !cut; ++q) cut = Kept(row, off + 4 * q);
      if (!cut) return false;
      for (std::uint32_t q = 0; q < nw;) {   // the block without the kept registers
        if (Kept(row, off + 4 * q)) {
          ++q;
          continue;
        }
        std::uint32_t e = q + 1;
        while (e < nw && !Kept(row, off + 4 * e)) ++e;
        std::uint32_t a = 0;
        std::memcpy(&a, op + 8, 4);
        const std::uint32_t at = a + 4 * q, bl = 16 + 4 * (e - q);
        const std::size_t h = out.size();
        out.insert(out.end(), op, op + 16);
        std::memcpy(&out[h + 8], &at, 4), std::memcpy(&out[h + 12], &bl, 4);
        out.insert(out.end(), op + 16 + 4 * q, op + 16 + 4 * e);
        ++ops, q = e;
      }
      return true;
    }
  };
  static constexpr std::size_t kPacketBytes = 32u << 20;
  // The device address of a packet stream in the packet store, 0 when it is full. Streams are stored once by content:
  // a forward repeats the same image switches, and a recorded script replays its packets by address.
  std::uint64_t StorePackets(const std::vector<std::uint32_t>& st) {
    const std::size_t bytes = 4 * st.size();
    std::string key(reinterpret_cast<const char*>(st.data()), bytes);
    if (const auto it = packet_index_.find(key); it != packet_index_.end()) return it->second;
    if (packets_used_ + bytes > packets_->bytes) return 0;
    auto* dst = static_cast<std::uint8_t*>(packets_->host) + packets_used_;
    std::memcpy(dst, st.data(), bytes);
    for (std::size_t q = 0; q < bytes; q += 64) _mm_clflush(dst + q);   // the NPU's reads do not snoop the CPU caches
    _mm_mfence();
    std::uint64_t da = 0;
    YAH_AMDF(api_->memory_query_address(packets_->native, 0, AMDF_MEMORY_ADDRESS_XDNA_DMA, &da), "packet address");
    da += packets_used_;
    packets_used_ += (bytes + 63) / 64 * 64;
    packet_index_.emplace(std::move(key), da);
    return da;
  }
  // DeltaSetup's setup out with its packet streams: the lane and the first batch of packets after the leading core
  // disables, a batch per stretch of the setup (each ends with a one-shot task, the only one with a token: a repeated
  // task issues one per repeat), the lane down and the deferred routes before the trailing core enables.
  void ControlPackets(PacketSetup& ps, std::vector<std::uint8_t>& out, std::uint32_t& ops) {
    using PS = ProgramState;
    std::vector<std::uint8_t> seq;
    std::uint32_t n = 0;
    auto w32 = [&](std::uint32_t a, std::uint32_t v) {
      std::uint8_t op[24] = {};
      const std::uint32_t sz = 24;
      std::memcpy(op + 8, &a, 4), std::memcpy(op + 16, &v, 4), std::memcpy(op + 20, &sz, 4);
      seq.insert(seq.end(), op, op + 24), ++n;
    };
    auto blk = [&](std::uint32_t a, std::initializer_list<std::uint32_t> v) {
      std::uint8_t op[16] = {1};
      const std::uint32_t sz = static_cast<std::uint32_t>(16 + 4 * v.size()), loc = (a >> 20 & 31) << 8 | a >> 25;
      std::memcpy(op + 4, &loc, 4), std::memcpy(op + 8, &a, 4), std::memcpy(op + 12, &sz, 4);
      seq.insert(seq.end(), op, op + 16);
      for (const std::uint32_t x : v) seq.insert(seq.end(), reinterpret_cast<const std::uint8_t*>(&x),
                                                 reinterpret_cast<const std::uint8_t*>(&x) + 4);
      ++n;
    };
    auto tile = [](std::uint32_t c, std::uint32_t r) { return c << 25 | r << 20; };
    struct Block {
      std::uint32_t row, n;
      std::uint64_t da;
    };
    std::array<std::vector<Block>, PS::kCols> blocks;
    std::array<std::uint32_t, PS::kCols> top{};   // rows on the lane
    for (std::uint32_t c = 0; c < PS::kCols; ++c)
      for (std::uint32_t r = 0; r < PS::kRows; ++r) {
        const auto& st = ps.stream[c * PS::kRows + r];
        if (st.empty()) continue;
        if (const std::uint64_t da = StorePackets(st)) {
          for (std::size_t k = 0; k < st.size() / 5; k += 64)
            blocks[c].push_back({r + 2, static_cast<std::uint32_t>(std::min<std::size_t>(64, st.size() / 5 - k)),
                                 da + 20 * k});
          top[c] = r + 1;
          continue;
        }
        for (std::size_t k = 0; k < st.size(); k += 5)   // the store is full: written by the microcontroller
          blk(tile(c, r + 2) | (st[k] & 0xFFFFF), {st[k + 1], st[k + 2], st[k + 3], st[k + 4]});
      }
    const std::vector<std::uint8_t> fallback = std::move(seq);
    const std::uint32_t fallback_ops = n;
    seq.clear(), n = 0;
    std::size_t batches = 0;
    for (const auto& b : blocks) batches = std::max(batches, (b.size() + 2) / 3);
    // the leading core disables and the trailing core enables (maskwrites of 0x32000)
    std::vector<std::size_t> at;
    std::size_t lead = 16, tail = out.size();
    bool other = false;
    for (std::size_t q = 16; q < out.size();) {
      std::uint32_t a = 0, len = out[q] == 0 ? 24 : out[q] == 3 ? 28 : 0;
      std::memcpy(&a, &out[q + 8], 4);
      if (out[q] == 1) std::memcpy(&len, &out[q + 12], 4);
      else if (!len) std::memcpy(&len, &out[q + 4], 4);
      const bool enable = (out[q] == 0 || out[q] == 3) && (a & 0xFFFFF) == 0x32000;
      if (!enable) other = true, tail = out.size();
      else if (!other) lead = q + len;
      else if (tail == out.size()) tail = q;
      at.push_back(q);
      q += len;
    }
    std::vector<std::uint8_t> res(out.begin(), out.begin() + lead);
    auto flush = [&] { res.insert(res.end(), seq.begin(), seq.end()), ops += n, seq.clear(), n = 0; };
    auto batch = [&](std::size_t b) {
      for (std::uint32_t c = 0; c < PS::kCols; ++c) {
        const std::size_t b0 = 3 * b, b1 = std::min(blocks[c].size(), b0 + 3);
        for (std::size_t k = b0; k < b1; ++k) {
          const Block& bl = blocks[c][k];
          const auto bd = static_cast<std::uint32_t>(k - b0);
          // 5 words a repeat, packet id = row; iteration step 5 words, wrap n (both encoded minus one)
          blk(tile(c, 0) | 0x1D000 + 0x20 * bd,
              {5, static_cast<std::uint32_t>(bl.da),
               static_cast<std::uint32_t>(bl.da >> 32 & 0xFFFF) | 1u << 30 | bl.row << 19, 0, 0xC0000000, 0x02000000,
               bl.n > 1 ? (bl.n - 1) << 20 | 4 : 0, 1u << 25});
          w32(tile(c, 0) | 0x1D214, (bl.n - 1) << 16 | bd);
        }
        if (b1 > b0) {   // the one-shot: the last packet again
          const Block& bl = blocks[c][b1 - 1];
          const std::uint64_t la = bl.da + 20 * (bl.n - 1);
          blk(tile(c, 0) | 0x1D060, {5, static_cast<std::uint32_t>(la),
                                     static_cast<std::uint32_t>(la >> 32 & 0xFFFF) | 1u << 30 | bl.row << 19, 0,
                                     0xC0000000, 0x02000000, 0, 1u << 25});
          w32(tile(c, 0) | 0x1D214, 0x80000003);
        }
      }
    };
    auto wait = [&](std::size_t b) {
      for (std::uint32_t c = 0; c < PS::kCols; ++c) {
        if (blocks[c].size() <= 3 * b) continue;
        std::uint8_t op[16] = {128};
        const std::uint32_t sz = 16, w2 = 1 | c << 16, w3 = 1u << 8 | 1u << 16;   // MM2S0 of shim (c, 0)
        std::memcpy(op + 4, &sz, 4), std::memcpy(op + 8, &w2, 4), std::memcpy(op + 12, &w3, 4);
        seq.insert(seq.end(), op, op + 16), ++n;
      }
    };
    if (batches) {
      res.insert(res.end(), ps.hoist.begin(), ps.hoist.end()), ops += ps.hoist_ops;
      for (std::uint32_t c = 0; c < PS::kCols; ++c) {
        if (!top[c]) continue;
        for (std::uint32_t m = 0; m < 22; ++m)
          if (ps.shim[c][m] == 0x80000005) w32(tile(c, 0) | 0x3F000 + 4 * m, 0);
        w32(tile(c, 0) | 0x3F044, 0x80000005);                                // M17 <- S5 (MM2S0)
        w32(tile(c, 1) | 0xB0130, 0x80000000), w32(tile(c, 1) | 0xB0040, 0x8000000C);   // S12, M16 <- S12
        for (std::uint32_t r = 0; r < top[c]; ++r) {
          const std::uint32_t t = tile(c, r + 2);
          w32(t | 0x3F128, 0xC0000000);   // S10 by packets: own row to arbiter 0 msel 0, rows above msel 1
          for (std::uint32_t k = 0; r + k < PS::kRows; ++k)
            w32(t | 0x3F2A0 + 4 * k, (r + 2 + k) << 24 | 0x1F << 16 | 1u << 8 | (k ? 1u << 4 : 0));
          w32(t | 0x3F00C, 0xC0000008);
          if (r + 1 < top[c]) w32(t | 0x3F048, 0xC0000010);
        }
      }
      batch(0);
      flush();
    }
    std::size_t from = lead;
    for (std::size_t b = 1; b < batches; ++b) {   // evenly through the setup
      const std::size_t target = lead + (tail - lead) * b / batches;
      const auto it = std::lower_bound(at.begin(), at.end(), std::max(target, from));
      const std::size_t cut = it == at.end() ? tail : std::min(*it, tail);
      res.insert(res.end(), out.begin() + from, out.begin() + cut);
      from = cut;
      wait(b - 1), batch(b), flush();
    }
    res.insert(res.end(), out.begin() + from, out.begin() + tail);
    if (batches) {
      wait(batches - 1);
      for (std::uint32_t c = 0; c < PS::kCols; ++c) {
        if (!top[c]) continue;
        w32(tile(c, 0) | 0x3F044, 0);
        w32(tile(c, 1) | 0xB0040, 0), w32(tile(c, 1) | 0xB0130, 0);
        for (std::uint32_t r = 0; r < top[c]; ++r) {
          const std::uint32_t t = tile(c, r + 2);
          w32(t | 0x3F00C, 0), w32(t | 0x3F048, 0), w32(t | 0x3F128, 0);
          for (std::uint32_t k = 0; k < 4; ++k) w32(t | 0x3F2A0 + 4 * k, 0);
        }
      }
      flush();
    }
    res.insert(res.end(), fallback.begin(), fallback.end()), ops += fallback_ops;
    res.insert(res.end(), ps.deferred.begin(), ps.deferred.end()), ops += ps.deferred_ops;
    res.insert(res.end(), out.begin() + tail, out.end());
    out = std::move(res);
  }
  ProgramState program_;
  Shared* packets_ = nullptr;   // the packet store (StorePackets)
  std::size_t packets_used_ = 0;
  std::unordered_map<std::string, std::uint64_t> packet_index_;
  std::atomic<bool> program_lost_{false};
  std::atomic<std::uint64_t> epoch_{0};   // Epoch
  Script* recording_ = nullptr;           // Record
  std::vector<std::pair<amdf_memory_t*, amdf_host_mapping_t*>> scripts_;   // sealed scripts' storage
  Fused FusedCommand(const std::vector<std::vector<const Call*>>& runs, int dec, const void* fam) {
    std::vector<std::uint8_t> bytes;
    std::uint32_t ops = 0;
    auto append_raw = [&](const std::uint8_t* src, std::size_t length) {
      std::uint32_t n = 0, size = 0;
      std::memcpy(&n, src + 8, 4), std::memcpy(&size, src + 12, 4);
      if (size != length) throw LoomError("npu: unexpected command header");
      if (bytes.empty()) bytes.assign(src, src + 16);
      bytes.insert(bytes.end(), src + 16, src + size);
      ops += n;
    };
    auto append = [&](const std::vector<std::uint8_t>& b) { append_raw(b.data(), b.size()); };
    auto append_ops = [&](const std::vector<std::uint8_t>& b, std::uint32_t n) {
      bytes.insert(bytes.end(), b.begin(), b.end());
      ops += n;
    };
    for (const auto& calls : runs) {
      int queued = 0, since = 0;
      if (const Kernel* k0 = calls[0]->k; FamilyOf(k0) != fam) {
        if (k0->setup.empty()) throw LoomError("npu: a job switches image families without a setup");
        append(DeltaSetup(k0->setup.data(), k0->setup.size(), true));
        fam = FamilyOf(k0), dec = k0->dec;
      }
      append(calls[0]->gate);
      if (prof_) append_ops(prof_->Setup(calls[0]->k->path), prof_->SetupOps());
      if (!calls[0]->k->streamed) {
        for (const Call* c : calls) append(c->call);
        append(calls.back()->done);
        if (prof_) append_ops(prof_->Read(), 1);
        continue;
      }
      for (std::size_t i = 0; i < calls.size(); ++i) {
        const Call& c = *calls[i];
        if (c.k->swap && c.k->dec != dec) {
          for (; queued > 0; --queued) append(calls[i - 1]->wait);
          // mid-job (a gate's tasks live on columns 1-2): no packets
          append(DeltaSetup(CommandBytes(*c.k->swap, c.k->swap->first), c.k->swap->first.byte_length, false));
          dec = c.k->dec, since = 0;
        }
        append(since == 0 ? c.lead : c.push[since & 1]);
        ++since;
        if (++queued == 2) append(c.wait), --queued;
      }
      for (; queued > 0; --queued) append(calls.back()->wait);
      append(calls.back()->done);
      if (prof_) append_ops(prof_->Read(), 1);
    }
    const std::uint32_t size = static_cast<std::uint32_t>(bytes.size());
    std::memcpy(&bytes[8], &ops, 4), std::memcpy(&bytes[12], &size, 4);
    if (recording_) recording_->bytes.push_back(bytes);
    return {Place(bytes), dec, fam};
  }
  // A command's bytes in the arena ring, for the next command submitted after the attention commands still queued.
  amdf_xdna_kernel_command_t Place(const std::vector<std::uint8_t>& bytes) {
    const std::size_t size = bytes.size();
    if (size > kArenaBytes) throw LoomError("npu: fused command exceeds its arena");
    if (arenas_.empty() || arenas_[arena_].used + size > kArenaBytes) {
      if (arenas_.size() < kArenas) {
        arenas_.emplace_back();
        iree_xdna_elf_allocation_record_t req{};
        req.domain = IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND;
        req.byte_length = kArenaBytes;
        req.alignment = kCommandAlign;
        AllocateStorage(req, &arenas_.back().storage, &arenas_.back().map);
        arena_ = arenas_.size() - 1;
      } else {   // the next arena, once the commands in it retired
        arena_ = (arena_ + 1) % kArenas;
        std::unique_lock<std::mutex> lock(mu_);
        const auto tw = std::chrono::steady_clock::now();
        retired_cv_.wait(lock, [&] { return retired_ >= arenas_[arena_].last; });
        commands_.arena_waits += 1;
        commands_.arena_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count();
        arenas_[arena_].used = 0;
      }
    }
    Arena& a = arenas_[arena_];
    {
      std::lock_guard<std::mutex> lock(mu_);
      a.last = submitted_ + attn_queue_.size() + 1;  // this command's submission
      commands_.bytes += size;
    }
    std::memcpy(static_cast<std::uint8_t*>(a.storage.mapping.data) + a.used, bytes.data(), size);
    YAH_AMDF(api_->host_mapping_cache_control(a.map, AMDF_HOST_CACHE_OPERATION_FLUSH, a.used, size),
             "host_mapping_cache_control(fused)");
    amdf_xdna_kernel_command_t c{};
    c.memory = a.storage.memory;
    c.access_ordinal = a.storage.access_ordinal;
    c.byte_offset = a.storage.memory_byte_offset + a.used;
    c.byte_length = size;
    a.used += (size + kCommandAlign - 1) / kCommandAlign * kCommandAlign;
    return c;
  }
  std::deque<Arena> arenas_;
  std::size_t arena_ = 0;   // the arena being filled

  static constexpr std::uint32_t kGateFailed = NpuSplit::kGateFailed;
  static void Fail(const std::vector<GatedJob>& jobs) {
    for (const GatedJob& g : jobs) __atomic_store_n(g.done, g.gate | kGateFailed, __ATOMIC_RELEASE);
  }
  std::uint64_t Submit(const amdf_xdna_kernel_command_t& command) {
    amdf_xdna_kernel_queue_submission_info_t si{};
    si.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
    si.structure_size = sizeof(si);
    si.command_count = 1;
    si.commands = &command;
    std::uint64_t submission = 0;
    YAH_AMDF(xdna_->kernel_queue_submit(queue_, &si, &submission), "kernel_queue_submit");
    return submission;
  }

  // Attention commands (EnqueueAttn) wait here, built, for their ready words; the thread submits them in order.
  struct AttnPending {
    amdf_xdna_kernel_command_t command{};
    volatile std::uint32_t* ready = nullptr;
    std::uint32_t ready_value = 0;
    GatedJob job;
    std::chrono::steady_clock::time_point t0;
  };
  std::deque<AttnPending> attn_queue_;  // under mu_
  std::condition_variable attn_cv_;
  std::thread attn_thread_;
  bool attn_gpu_done_ = false;  // under mu_ (Drain)
  void AttnQueued() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!attn_queue_.empty()) throw LoomError("npu: a command would run before queued attention commands");
  }
  // Polls each queued command's ready word (~20 us apart) and submits it. No time limit: the host queues chunks ahead
  // of the GPU, so a ready word can be many seconds away. After any failure, or without ready once the GPU is done
  // (Drain), a command fails: its done gets kGateFailed and nothing later runs.
  void AttnSubmitter() {
    for (;;) {
      AttnPending* p = nullptr;
      {
        std::unique_lock<std::mutex> lock(mu_);
        attn_cv_.wait(lock, [this] { return stop_ || !attn_queue_.empty(); });
        if (attn_queue_.empty()) return;
        p = &attn_queue_.front();  // stays queued until submitted: Place counts it
      }
      bool ok = true;
      for (;;) {
        bool stop = false;
        {
          std::lock_guard<std::mutex> lock(mu_);
          stop = stop_ || !failure_.empty();
          if (!stop && attn_gpu_done_ && __atomic_load_n(p->ready, __ATOMIC_ACQUIRE) < p->ready_value) {
            failure_ = "npu: an attention job's ready word never came (" + p->job.tag + ")";
            stop = true;
          }
        }
        if (stop) {
          ok = false;
          break;
        }
        if (__atomic_load_n(p->ready, __ATOMIC_ACQUIRE) >= p->ready_value) break;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
      }
      if (!ok) {
        // its done word fails; the commands built after it never run (the array state they expect is not there)
        std::lock_guard<std::mutex> lock(mu_);
        for (const AttnPending& q : attn_queue_) Fail({q.job});
        attn_queue_.clear();
        resident_ = nullptr, resident_dec_ = -1, program_lost_ = true, ++epoch_;
        retired_cv_.notify_all();
        continue;
      }
      const std::uint64_t submission = Submit(p->command);
      {
        std::lock_guard<std::mutex> lock(mu_);
        pending_.push_back({submission, {p->job}, p->t0});
        attn_queue_.pop_front();
        ++submitted_;
      }
      reap_cv_.notify_one();
    }
  }

  // Retires gated commands in order (EnqueueGated): stats.
  // On a failure it stores gate | kGateFailed to the done words of this and every later command's jobs, so no GPU wait
  // is left spinning.
  struct Pending {
    std::uint64_t submission;
    std::vector<GatedJob> jobs;
    std::chrono::steady_clock::time_point t0;
  };
  std::deque<Pending> pending_;
  std::chrono::steady_clock::time_point last_retired_{};  // reaper thread only
  std::condition_variable reap_cv_;
  std::thread reaper_;
  void Reaper() {
    for (;;) {
      Pending p;
      {
        std::unique_lock<std::mutex> lock(mu_);
        reap_cv_.wait(lock, [this] { return stop_ || !pending_.empty(); });
        if (pending_.empty()) return;
        p = std::move(pending_.front());
        pending_.pop_front();
      }
      bool ok = true;
      try {
        YAH_AMDF(api_->kernel_queue_wait(queue_, p.submission, AMDF_TIMEOUT_INFINITE, 0), "kernel_queue_wait(gated)");
      } catch (const LoomError& e) {
        ok = false;
        std::lock_guard<std::mutex> lock(mu_);
        if (failure_.empty()) failure_ = e.what();
        resident_ = nullptr;
        resident_dec_ = -1;
        program_lost_ = true;   // the next command's setups write whole programs (DeltaSetup)
        ++epoch_;
      }
      const auto now = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(now - std::max(p.t0, last_retired_)).count();
      // submitted after the previous command retired: the NPU sat idle with nothing queued
      const double idle = p.t0 > last_retired_ && last_retired_.time_since_epoch().count()
                              ? std::chrono::duration<double, std::milli>(p.t0 - last_retired_).count() : 0;
      last_retired_ = now;
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (!failure_.empty()) ok = false;
        if (ok) {
          for (const GatedJob& g : p.jobs) {
            auto& st = stats_[g.tag];
            st.jobs += 1, st.calls += g.calls.size() + g.host_calls;
            if (g.host_calls) __atomic_store_n(g.done, g.gate, __ATOMIC_RELEASE);
          }
          commands_.commands += 1, commands_.busy_ms += ms, commands_.max_ms = std::max(commands_.max_ms, ms);
          commands_.idle_ms += idle;
          last_command_ms_ = ms;
        }
      }
      if (!ok) Fail(p.jobs);
      {
        std::lock_guard<std::mutex> lock(mu_);
        ++retired_;
        active_ = std::chrono::steady_clock::now();
      }
      retired_cv_.notify_all();
    }
  }

  // The memory profile that registers caller-owned host pages for NPU access (CreateShared host).
  uint32_t RegisterProfile() {
    for (uint32_t ord = 0;; ++ord) {
      amdf_memory_profile_t p{};
      p.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      p.structure_size = sizeof(p);
      amdf_memory_access_capabilities_t caps{};
      caps.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
      caps.structure_size = sizeof(caps);
      const amdf_status_t st = api_->memory_scope_query_device_profile(scope_, ord, 1, &access_, &p, &caps);
      if (amdf_status_code(st) == AMDF_STATUS_CODE_OUT_OF_RANGE) throw LoomError("npu: no host registration profile");
      if (st == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) continue;
      YAH_AMDF(st, "memory_scope_query_device_profile(register)");
      if (p.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) return p.ordinal;
    }
  }

  void SelectScope() {
    uint32_t count = 0;
    api_->instance_enumerate_memory_scopes(instance_, 0, nullptr, &count);
    std::vector<amdf_memory_scope_t*> scopes(count);
    YAH_AMDF(api_->instance_enumerate_memory_scopes(instance_, count, scopes.data(), &count),
             "instance_enumerate_memory_scopes");
    for (auto* s : scopes) {
      amdf_memory_scope_info_t info{};
      info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
      info.structure_size = sizeof(info);
      YAH_AMDF(api_->memory_scope_query_info(s, &info), "memory_scope_query_info");
      if (info.kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM) {
        scope_ = s;
        return;
      }
    }
    throw LoomError("npu: no system memory scope");
  }

  void OpenEndpoint() {
    uint32_t count = 0;
    YAH_AMDF(api_->endpoint_enumerate(instance_, 0, nullptr, &count), "endpoint_enumerate(count)");
    std::vector<amdf_endpoint_summary_t> summaries(count);
    YAH_AMDF(api_->endpoint_enumerate(instance_, count, summaries.data(), &count), "endpoint_enumerate");
    for (auto& s : summaries) {
      if (s.engine_kind != AMDF_ENGINE_KIND_XDNA) continue;
      YAH_AMDF(api_->endpoint_open(instance_, &s.id, &endpoint_), "endpoint_open");
      return;
    }
    throw LoomError("npu: no XDNA endpoint");
  }

  void CreateDevice() {
    amdf_xdna_endpoint_info_t xi{};
    xi.type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO;
    xi.structure_size = sizeof(xi);
    YAH_AMDF(xdna_->endpoint_query_info(endpoint_, &xi), "xdna.endpoint_query_info");
    NpuCheck(iree_hal_amd_xdna_aie2p_npu2_target_initialize(iree_make_cstring_view(xi.target_id),
                                                            static_cast<uint16_t>(columns_), &target_),
             "npu2_target_initialize");
    amdf_xdna_device_create_info_t dci{};
    dci.type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_CREATE_INFO;
    dci.structure_size = sizeof(dci);
    YAH_AMDF(xdna_->device_create(endpoint_, &dci, &device_), "xdna.device_create");
    amdf_xdna_device_info_t di{};
    di.type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO;
    di.structure_size = sizeof(di);
    YAH_AMDF(xdna_->device_query_info(device_, &di), "xdna.device_query_info");
    target_.instruction_alignment = di.instruction.address_alignment;
    amdf_endpoint_info_t ei{};
    ei.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    ei.structure_size = sizeof(ei);
    YAH_AMDF(api_->endpoint_query_info(endpoint_, &ei), "endpoint_query_info");
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < ei.queue_family_count && family == UINT32_MAX; ++i) {
      amdf_queue_family_info_t f{};
      f.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
      f.structure_size = sizeof(f);
      YAH_AMDF(api_->endpoint_query_queue_family_info(endpoint_, i, &f), "endpoint_query_queue_family_info");
      if (f.command_type == AMDF_QUEUE_COMMAND_TYPE_XDNA && (f.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL))
        family = i;
    }
    if (family == UINT32_MAX) throw LoomError("npu: no XDNA kernel queue family");
    access_.device = device_;
    access_.requirements.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    access_.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    access_.requirements.address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA;
    profile_.ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
    for (uint32_t ord = 0;; ++ord) {
      amdf_memory_profile_t p{};
      p.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      p.structure_size = sizeof(p);
      amdf_memory_access_capabilities_t caps{};
      caps.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
      caps.structure_size = sizeof(caps);
      const amdf_status_t st = api_->memory_scope_query_device_profile(scope_, ord, 1, &access_, &p, &caps);
      if (amdf_status_code(st) == AMDF_STATUS_CODE_OUT_OF_RANGE) break;
      if (st == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) continue;
      YAH_AMDF(st, "memory_scope_query_device_profile");
      bool dmabuf = false;
      for (uint32_t e = 0; e < p.external_memory_support_count; ++e)
        dmabuf |= p.external_memory_support[e].type == AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD &&
                  (p.external_memory_support[e].flags & AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT) &&
                  (p.external_memory_support[e].flags & AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET);
      if ((p.roles & AMDF_MEMORY_PROFILE_ROLE_IMPORT) && dmabuf) {
        profile_ = p;
        break;
      }
    }
    if (profile_.ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) throw LoomError("npu: no DMA-BUF import profile");
    amdf_xdna_context_create_info_t cci{};
    cci.type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO;
    cci.structure_size = sizeof(cci);
    cci.logical_column_count = columns_;
    cci.physical_column_origin = AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY;
    cci.acceptable_scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
    YAH_AMDF(xdna_->context_create(device_, &cci, &context_), "xdna.context_create");
    amdf_xdna_kernel_queue_create_info_t qi{};
    qi.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO;
    qi.structure_size = sizeof(qi);
    qi.queue_family_ordinal = family;
    YAH_AMDF(xdna_->kernel_queue_create(context_, &qi, &queue_), "xdna.kernel_queue_create");
    if (const char* spec = std::getenv("YAH_NPU_PROF")) {
      prof_ = std::make_unique<NpuProf>(spec);
      amdf_xdna_context_info_t info{};
      info.type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_INFO;
      info.structure_size = sizeof(info);
      YAH_AMDF(xdna_->context_query_info(context_, &info), "xdna.context_query_info");
      prof_->Attach(static_cast<std::uint32_t>(info.id.words[1]));   // Linux DRM: the context's handle, low word
    }
  }

  iree_hal_amd_xdna_image_t* Image(const std::string& path) {
    auto it = images_.find(path);
    if (it != images_.end()) return it->second;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw LoomError("npu: cannot open " + path);
    const auto size = static_cast<iree_host_size_t>(f.tellg());
    f.seekg(0);
    void* data = nullptr;
    NpuCheck(iree_allocator_malloc(iree_allocator_system(), size, &data), "malloc(image)");
    f.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    iree_byte_sequence_t* seq = nullptr;
    iree_byte_span_t span = iree_make_byte_span(data, size);
    NpuCheck(iree_byte_sequence_create_from_span_move(&span, iree_allocator_system(), &seq), "byte_sequence");
    iree_hal_amd_xdna_image_t* image = nullptr;
    const iree_status_t st = iree_hal_amd_xdna_image_create(seq, &target_, iree_allocator_system(), &image);
    iree_byte_sequence_release(seq);
    NpuCheck(st, ("image_create " + path).c_str());
    images_[path] = image;
    return image;
  }

  void AllocateStorage(const iree_xdna_elf_allocation_record_t& req, iree_hal_amd_xdna_executable_storage_t* out,
                       amdf_host_mapping_t** out_map) {
    const bool command = req.domain == IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND;
    amdf_memory_scope_t* scope = scope_;
    if (command) {
      uint32_t count = 0;
      YAH_AMDF(xdna_->context_enumerate_memory_scopes(context_, 1, &scope, &count),
               "xdna.context_enumerate_memory_scopes");
    }
    const amdf_memory_address_kind_t kind = command ? AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE : AMDF_MEMORY_ADDRESS_XDNA_DMA;
    amdf_memory_device_access_t access{};
    access.device = device_;
    access.requirements.access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE | (command ? AMDF_MEMORY_ACCESS_EXECUTE : 0);
    access.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    access.requirements.address_kinds = UINT64_C(1) << kind;
    amdf_memory_profile_t p{};
    const amdf_memory_profile_roles_t roles = AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    for (uint32_t ord = 0;; ++ord) {
      p = {};
      p.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      p.structure_size = sizeof(p);
      amdf_memory_access_capabilities_t caps{};
      caps.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
      caps.structure_size = sizeof(caps);
      const amdf_status_t st = api_->memory_scope_query_device_profile(scope, ord, 1, &access, &p, &caps);
      if (amdf_status_code(st) == AMDF_STATUS_CODE_OUT_OF_RANGE) throw LoomError("npu: no storage profile");
      if (st == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) continue;
      YAH_AMDF(st, "memory_scope_query_device_profile(storage)");
      if ((p.roles & roles) == roles && (p.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE)) break;
    }
    const std::uint64_t g = p.allocation.byte_length_granularity;
    amdf_memory_create_info_t ci{};
    ci.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    ci.structure_size = sizeof(ci);
    ci.memory_profile_ordinal = p.ordinal;
    ci.access_count = 1;
    ci.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    ci.byte_length = (req.byte_length + g - 1) / g * g;
    ci.minimum_alignment = p.allocation.minimum_alignment;
    ci.accesses = &access;
    storage_bytes_ += ci.byte_length;   // the NPU's command memory is small (~64 MB per process): report the total
    ++storage_count_;
    YAH_AMDF(api_->memory_create(scope, &ci, &out->memory),
             ("memory_create(storage) " + std::to_string(storage_count_) + " storages, " +
              std::to_string(storage_bytes_ >> 10) + " KB with this one").c_str());
    storage_size_[out->memory] = ci.byte_length;
    YAH_AMDF(api_->memory_query_address(out->memory, 0, kind, &out->device_address), "memory_query_address(storage)");
    amdf_memory_map_info_t mi{};
    mi.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    mi.structure_size = sizeof(mi);
    mi.byte_length = req.byte_length;
    mi.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    YAH_AMDF(api_->memory_map(out->memory, &mi, out_map), "memory_map(storage)");
    amdf_host_mapping_info_t info{};
    info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    info.structure_size = sizeof(info);
    YAH_AMDF(api_->host_mapping_query_info(*out_map, &info), "host_mapping_query_info(storage)");
    out->mapping = iree_make_byte_span(info.pointer, static_cast<iree_host_size_t>(req.byte_length));
  }

  LoomDevice& gpu_;
  std::uint32_t columns_;
  const amdf_api_t* api_ = nullptr;
  const amdf_xdna_api_t* xdna_ = nullptr;
  amdf_instance_t* instance_ = nullptr;
  amdf_memory_scope_t* scope_ = nullptr;
  amdf_endpoint_t* endpoint_ = nullptr;
  amdf_device_t* device_ = nullptr;
  amdf_xdna_context_t* context_ = nullptr;
  std::unique_ptr<NpuProf> prof_;
  amdf_kernel_queue_t* queue_ = nullptr;
  amdf_memory_device_access_t access_{};
  amdf_memory_profile_t profile_{};
  using ExportFn = int (*)(const void*, std::size_t, int*, std::uint64_t*);
  ExportFn export_dmabuf_ = nullptr;
  iree_hal_amd_xdna_aie2p_target_t target_{};
  std::map<std::string, iree_hal_amd_xdna_image_t*> images_;
  std::vector<std::unique_ptr<Shared>> shared_;
  std::vector<std::unique_ptr<Kernel>> kernels_;
  const void* resident_ = nullptr;  // the image (family) whose array state is set up; under mu_
  std::uint64_t storage_bytes_ = 0;    // command-memory storages allocated (images' and the command arenas')
  std::uint32_t storage_count_ = 0;
  std::map<const amdf_memory_t*, std::uint64_t> storage_size_;  // per storage: its bytes in storage_bytes_
  int resident_dec_ = -1;            // its decoder format at the end of the last submitted command; under mu_
  std::map<std::tuple<const Shared*, std::size_t, std::size_t>, amdf_memory_t*> imports_;   // DMA-BUF views
  std::atomic<double> last_command_ms_{0};
  std::map<std::string, TagStats> stats_;  // under mu_
  CommandStats commands_;                  // under mu_
  std::mutex mu_;
  std::uint64_t submitted_ = 0, retired_ = 0;  // commands; under mu_
  std::chrono::steady_clock::time_point active_{};  // the last command's retirement; under mu_
  // How long the NPU may sit idle before runtime PM may power it down (amdxdna's autosuspend, 5 s by default), with a
  // margin; "on" (no runtime PM): never
  const std::chrono::milliseconds suspend_after_ = SuspendAfter();
  static std::chrono::milliseconds SuspendAfter() {
    const std::string dir = "/sys/class/accel/accel0/device/power/";
    std::ifstream c(dir + "control"), d(dir + "autosuspend_delay_ms");
    std::string control;
    long delay = 0;
    if (c >> control && control == "on") return std::chrono::hours(24 * 365);
    if (!(d >> delay) || delay <= 0) return std::chrono::milliseconds(500);
    return std::chrono::milliseconds(delay * 3 / 4);
  }
  std::condition_variable retired_cv_;
  bool stop_ = false;
  std::string failure_;
};

// The NPU side of LoomPrefill's column split: the shared A / W / C buffers, the flag words and the calls bound on them.
class LoomNpuSplit : public NpuSplit {
 public:
  LoomNpuSplit(LoomDevice& gpu, const NpuPlan& plan)
      : npu_(gpu, 8),
        a_(npu_.CreateShared(plan.a_bytes)),
        w_(npu_.CreateShared(plan.w_bytes)),
        c_(npu_.CreateShared(plan.c_bytes, true)),  // the NPU writes C (see CreateShared)
        flags_(npu_.CreateShared(plan.dcol ? kDcolFlagBytes : 65536, true)),
        cscratch_(npu_.CreateShared(plan.dcol ? plan.c_bytes : 4096, true)),
        plan_(plan) {
    if (!plan.gate_calls) throw LoomError("npu: the set has no gate protocol (dispatch.txt npugate; re-emit it)");
    if (!GpuPinnedHigh())
      throw LoomError("npu: pin the GPU performance level first (echo high > /sys/class/drm/card*/device/"
                      "power_dpm_force_performance_level): fabric / memory clock switches corrupt NPU outputs");
    if (kSignal + 2 * plan.gate_record > kGateSlot) throw LoomError("npu: the gate records do not fit a gate slot");
  }
  const LoomBuffer& A() const override { return a_.gpu; }
  const LoomBuffer& W() const override { return w_.gpu; }
  const LoomBuffer& C() const override { return c_.gpu; }
  std::uint32_t Bind(const std::string& image, NpuView a, NpuView w, NpuView c) override {
    const auto key = std::make_tuple(image, a.offset, a.length, w.offset, w.length, c.offset, c.length);
    if (const auto it = ids_.find(key); it != ids_.end()) return it->second;
    // the call's gate slot: flag record, tick scratch, signal (go, done)
    const std::size_t slot = GateSlot();
    calls_.push_back(npu_.Snapshot(npu_.Load(image, "npu_gemm",
                                             {{&a_, a.offset, a.length},
                                              {&w_, w.offset, w.length},
                                              {&c_, c.offset, c.length},
                                              {&flags_, slot, plan_.gate_record},
                                              {&flags_, slot + kTick, plan_.gate_record},
                                              {&flags_, slot + kSignal, 2 * plan_.gate_record}})));
    return ids_[key] = static_cast<std::uint32_t>(calls_.size() - 1);
  }
  void RegisterRaw(const void* p, std::size_t bytes) override {
    const auto b = reinterpret_cast<std::uintptr_t>(p) & ~std::uintptr_t{4095};
    const auto e = (reinterpret_cast<std::uintptr_t>(p) + bytes + 4095) & ~std::uintptr_t{4095};
    for (const Raw& r : raw_)
      if (r.begin <= b && e <= r.end) return;
    // libamdf pins for writing: the model is resident writable memory (Gguf::OpenResident), so the NPU shares the
    // GPU's pages (a protection change or a copy-on-write fault there would evict every GPU queue). The NPU does not
    // snoop the CPU's caches, so the rows are flushed to memory.
    auto* pages = reinterpret_cast<void*>(b);
    LoomNpu::Shared& s = npu_.RegisterHost(pages, e - b);
    for (std::uintptr_t a = b; a < e; a += 64) _mm_clflushopt(reinterpret_cast<void*>(a));
    _mm_sfence();
    raw_.push_back({b, e, &s});
  }
  std::uint32_t BindRaw(const std::string& image, const std::string& swap, int dec, NpuView a, const void* raw,
                        std::size_t raw_bytes, NpuView c, const std::string& family) override {
    const auto r0 = reinterpret_cast<std::uintptr_t>(raw);
    const auto key = std::make_tuple(image, a.offset + (a.in_c ? std::size_t{1} << 62 : 0), a.length, std::size_t{r0},
                                     raw_bytes, c.offset, c.length);
    if (const auto it = ids_.find(key); it != ids_.end()) return it->second;
    std::vector<LoomNpu::View> views = CallViews(a, c);
    views.push_back(RawView(raw, raw_bytes));
    return ids_[key] = BindViews(image, swap, dec, family, std::move(views));
  }
  std::uint32_t BindRawPair(const std::string& image, const std::string& swap, int dec, NpuView a, const void* gate,
                            std::size_t gate_bytes, const void* up, std::size_t up_bytes, NpuView c,
                            std::size_t junk_bytes, const std::string& family) override {
    const auto key = std::make_tuple(image, a.offset + (a.in_c ? std::size_t{1} << 62 : 0), a.length,
                                     std::size_t{reinterpret_cast<std::uintptr_t>(gate)},
                                     reinterpret_cast<std::uintptr_t>(up), c.offset, c.length);
    if (const auto it = ids_.find(key); it != ids_.end()) return it->second;
    if (junk_bytes > cscratch_.bytes) throw LoomError("npu: the gate columns' records exceed the scratch C");
    std::vector<LoomNpu::View> views = CallViews(a, c);
    views.push_back(RawView(gate, gate_bytes));
    views.push_back(RawView(up, up_bytes));
    views.push_back({&cscratch_, 0, junk_bytes});   // the gate columns' C records: never read
    return ids_[key] = BindViews(image, swap, dec, family, std::move(views));
  }
  // A decoder-column call's bindings up to its raw rows: A, the panel (not read), C, the gate's flag / tick / signal,
  // the fill sink (not written).
  std::vector<LoomNpu::View> CallViews(NpuView a, NpuView c) {
    const std::size_t slot = GateSlot();
    return {{a.in_c ? &c_ : &a_, a.offset, a.length},
            {&w_, 0, plan_.w_bytes},
            {&c_, c.offset, c.length},
            {&flags_, slot, plan_.gate_record},
            {&flags_, slot + kTick, plan_.gate_record},
            {&flags_, slot + kSignal, 2 * plan_.gate_record},
            {&w_, 0, plan_.w_bytes}};
  }
  LoomNpu::View RawView(const void* raw, std::size_t bytes) const {
    const auto r0 = reinterpret_cast<std::uintptr_t>(raw);
    for (const Raw& x : raw_)
      if (x.begin <= r0 && r0 + bytes <= x.end) return {x.shared, r0 - x.begin, bytes};
    throw LoomError("npu: raw weights outside the registered ranges");
  }
  std::uint32_t BindViews(const std::string& image, const std::string& swap, int dec, const std::string& family,
                          std::vector<LoomNpu::View> views) {
    // one instance per image, re-bound per call (an instance's storage is ~0.7 MB, mostly its array setup)
    LoomNpu::Kernel*& k = instances_[image];
    if (k && !LoomNpu::Loaded(*k)) {  // (EndBinding)
      npu_.LoadStorage(*k, views);
    } else if (!k) {
      k = &npu_.Load(image, "npu_gemm", views);
      LoomNpu::Kernel*& sw = instances_[swap];
      if (!sw) sw = &npu_.Load(swap, "npu_gemm", views);   // only its setup runs
      k->family = &families_[family], k->dec = dec, k->swap = sw;
      std::vector<LoomNpu::View> scratch = views;   // the setup: first with its call's C on the scratch
      scratch[2] = {&cscratch_, 0, views[2].length};
      npu_.Rebind(*k, scratch);
      k->setup = npu_.FirstBytes(*k);
      npu_.Rebind(*k, views);
      // the copy relies on C's address living in first's own bytes; a gfirst set's first is the setup alone (no C)
      if ((k->setup == npu_.FirstBytes(*k)) != plan_.gfirst)
        throw LoomError("npu: " + image +
                        (plan_.gfirst ? ": first runs a call (re-emit the set)" : ": first does not hold C's address"));
      // an image whose setup is another's byte for byte (a format's site images of K = 5120 and 6144: the NPU takes
      // K 0 - 5119 of both) sets the array up the same way: one family, so no setup runs between their jobs
      for (const auto& [path, other] : instances_)
        if (other && other != k && other->family && other->dec == k->dec && other->setup == k->setup) {
          k->family = other->family;
          break;
        }
    } else {
      npu_.Rebind(*k, views);
    }
    calls_.push_back(npu_.Snapshot(*k));
    return static_cast<std::uint32_t>(calls_.size() - 1);
  }
  // An instance's commands are copies once bound (Snapshot; FusedCommand splices its setup copy), except a swap image's
  // setup, which FusedCommand reads from its storage, and the first of an image with no setup copy, which runs as is.
  void EndBinding() override {
    std::set<const LoomNpu::Kernel*> keep;
    for (const auto& [path, k] : instances_)
      if (k && k->swap && k->swap != k) keep.insert(k->swap);
    for (const auto& [path, k] : instances_)
      if (k && LoomNpu::Loaded(*k) && !k->setup.empty() && !keep.count(k)) npu_.UnloadStorage(*k);
  }
  // A flag word from / to the host, through memory (the NPU does not snoop CPU caches).
  void HostStore(std::uint32_t word, std::uint32_t value) override {
    auto* w = static_cast<volatile std::uint32_t*>(flags_.host) + word;
    __atomic_store_n(w, value, __ATOMIC_RELEASE);
    __builtin_ia32_clflush(const_cast<std::uint32_t*>(w));
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
  }
  NpuProf* Prof() override { return npu_.Prof(); }
  std::uint32_t HostLoad(std::uint32_t word) override {
    auto* w = static_cast<volatile std::uint32_t*>(flags_.host) + word;
    __builtin_ia32_clflush(const_cast<std::uint32_t*>(w));
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    return __atomic_load_n(w, __ATOMIC_ACQUIRE);
  }
  void CheckHealthNow() { npu_.CheckHealth(); }
  // NPU time of the last command, after every submitted one retired.
  double LastCommandMs() {
    npu_.Drain();
    return npu_.LastCommandMs();
  }
  [[nodiscard]] const LoomBuffer* Flags() const override { return &flags_.gpu; }
  [[nodiscard]] std::uint32_t Word(std::uint32_t word) const override {
    return __atomic_load_n(static_cast<const std::uint32_t*>(flags_.host) + word, __ATOMIC_ACQUIRE);
  }
  Job NewJob(const std::vector<std::uint32_t>& calls) override {
    if (calls.empty() || calls.size() >= plan_.gate_calls) throw LoomError("npu: too many calls for one NPU job");
    Job j;
    j.ready = static_cast<std::uint32_t>((kGateBase + kGateSlot * calls.front()) / 4);
    j.done = static_cast<std::uint32_t>((kGateBase + kGateSlot * calls.back() + kSignal + plan_.gate_record) / 4);
    // decoder-column sets: the gate counts jobs from 1 after each image setup, which precedes every family switch and
    // every chunk's first job (BeginChunk, FusedCommand)
    if (plan_.dcol) {
      const void* fam = LoomNpu::FamilyOf(calls_.at(calls.front()).k);
      if (fam != job_family_) job_family_ = fam, seq_ = 0;
    }
    j.gate = ++seq_ * plan_.gate_calls + static_cast<std::uint32_t>(calls.size());
    return j;
  }
  // The chunk's graph clears each job's words after joining it (LoomPrefill::NpuJoin), so they read zero here.
  void BeginChunk() override {
    if (plan_.dcol) job_family_ = nullptr, fresh_ = true;
  }
  // A chunk's jobs as the previous chunk's, with the array's program state kept since that chunk started, build the same
  // commands every time (LoomNpu::Script): the second such chunk records them, later ones replay them.
  void Enqueue(const std::vector<Queued>& jobs) override {
    std::size_t queued = 0;
    const bool chunk = fresh_;
    const bool repeat = chunk && plan_.dcol && npu_.Epoch() == prev_epoch_ && SameJobs(jobs, prev_jobs_);
    if (chunk) prev_epoch_ = npu_.Epoch(), prev_jobs_ = jobs;
    const bool replay = repeat && !script_.commands.empty() && SameJobs(jobs, script_jobs_);
    const bool record = repeat && !replay && !no_script_;
    try {
      npu_.CheckHealth();
      if (Word(kFlagStatus)) throw LoomError("npu: a GPU wait for the NPU failed");
      if (replay) {
        npu_.Replay(script_, queued);
        fresh_ = false;
        return;
      }
      if (record) {
        npu_.Free(script_);
        script_jobs_.clear();
        npu_.Record(&script_);
      }
      while (queued < jobs.size()) {
        std::vector<LoomNpu::GatedJob> command;
        std::size_t calls = 0;
        const std::size_t per = plan_.dcol ? kDcolJobsPerCommand : kJobsPerCommand;
        for (std::size_t i = queued + command.size(); i < jobs.size() && command.size() < per; ++i) {
          if (!command.empty() && calls + jobs[i].calls.size() > kCallsPerCommand) break;
          // the NPU waits for the GPU between jobs: a command over many layers can outlast the 2000 ms limit
          if (!command.empty() && jobs[i].layer >= jobs[queued].layer + kLayersPerCommand) break;
          LoomNpu::GatedJob g;
          for (const std::uint32_t id : jobs[i].calls) g.calls.push_back(&calls_.at(id));
          g.done = static_cast<volatile std::uint32_t*>(flags_.host) + jobs[i].words.done;
          g.gate = jobs[i].words.gate;
          g.tag = jobs[i].tag;
          calls += jobs[i].calls.size();
          command.push_back(std::move(g));
        }
        const std::size_t n = command.size();
        npu_.EnqueueGated(std::move(command), fresh_);
        fresh_ = false;
        queued += n;
      }
      if (record && npu_.Epoch() == prev_epoch_) {
        try {
          npu_.Seal(script_);
          script_jobs_ = jobs;
        } catch (const LoomError&) {   // a chunk's script (~94 MB) past the NPU's command memory: issue every chunk anew
          npu_.Record(nullptr), npu_.Free(script_);
          no_script_ = true;
        }
      } else if (record) {   // the recording's own setups forgot program state
        npu_.Record(nullptr), npu_.Free(script_);
      }
    } catch (...) {
      if (record) npu_.Record(nullptr), npu_.Free(script_);
      prev_jobs_.clear();
      for (std::size_t i = queued; i < jobs.size(); ++i)
        HostStore(jobs[i].words.done, jobs[i].words.gate | kGateFailed);
      throw;
    }
  }
  // One line per job tag (jobs, calls), then the commands (LoomNpu::CommandStats).
  void Report(std::FILE* f) {
    npu_.Drain();
    if (Word(kFlagStatus)) std::fprintf(f, "npu: ERROR a GPU wait for the NPU failed\n");
    npu_.Report(f);
  }

  // The amdgpu card's DPM performance level is "high" (results.md: deferred data-fabric errors otherwise).
  static bool GpuPinnedHigh() {
    for (int card = 0; card < 8; ++card) {
      std::ifstream f("/sys/class/drm/card" + std::to_string(card) + "/device/power_dpm_force_performance_level");
      std::string level;
      if (f >> level) return level == "high";
    }
    return false;
  }

 private:
  // Gate slots, one per bound call from byte kGateBase of the flag words (below it: kFlagStatus), every view 64-byte
  // aligned: flag record (+0: ready), tick scratch (+kTick), signal (+kSignal: go, then done one record later).
  static constexpr std::size_t kGateBase = 256, kGateSlot = 192, kTick = 64, kSignal = 128;
  // Jobs per NPU command: few commands, while the longest (its waits for the GPU included) stays far below the driver's
  // 2000 ms command limit.
  static constexpr std::size_t kJobsPerCommand = 16, kCallsPerCommand = 256;
  // Layers one command spans at most: its first job may wait up to the gate's supply (~1.2 s, gen_npu_gemm GPS) for
  // the GPU, the rest of the command stays a few layers (~95 ms each in slow runs) under the 2000 ms limit.
  static constexpr std::uint32_t kLayersPerCommand = 4;
  // raw-row sets: layers without NPU work leave longer waits between jobs (up to the gate's ~1.2 s supply)
  static constexpr std::size_t kDcolJobsPerCommand = 8;
  // Decoder-column sets bind a few thousand calls (a gate slot each).
  static constexpr std::size_t kDcolFlagBytes = 2u << 20;
  std::size_t GateSlot() const {
    const std::size_t slot = kGateBase + kGateSlot * calls_.size();
    if (slot + kGateSlot > flags_.bytes) throw LoomError("npu: more NPU calls than gate slots");
    return slot;
  }
  struct Raw {
    std::uintptr_t begin, end;
    LoomNpu::Shared* shared;
  };
  LoomNpu npu_;
  LoomNpu::Shared &a_, &w_, &c_, &flags_;
  LoomNpu::Shared& cscratch_;   // the setup instances' C (BindRaw)
  NpuPlan plan_;
  std::uint32_t seq_ = 0;
  const void* job_family_ = nullptr;   // NewJob's current family run (decoder-column sets)
  bool fresh_ = false;                 // the next command starts a chunk (BeginChunk)
  // Enqueue's command replay: the previous chunk's jobs and the program epoch at its start, the script and its jobs
  std::vector<Queued> prev_jobs_, script_jobs_;
  std::uint64_t prev_epoch_ = 0;
  LoomNpu::Script script_;
  bool no_script_ = false;   // the script did not fit in command memory: no recording from then on
  static bool SameJobs(const std::vector<Queued>& a, const std::vector<Queued>& b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](const Queued& x, const Queued& y) {
      return x.calls == y.calls && x.tag == y.tag && x.layer == y.layer && x.words.ready == y.words.ready &&
             x.words.done == y.words.done && x.words.gate == y.words.gate;
    });
  }
  std::deque<LoomNpu::Call> calls_;   // stable addresses (LoomNpu::GatedJob)
  std::vector<Raw> raw_;
  std::map<std::string, LoomNpu::Kernel*> instances_;
  std::map<std::string, char> families_;   // the decoder-column image families (BindRaw), by name
  std::map<std::tuple<std::string, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>,
           std::uint32_t>
      ids_;
};

// The split prefill attention's NPU side on its own LoomNpu (a set splits either the GEMMs or the attention).
class LoomNpuAttn : public NpuAttn {
 public:
  // No DPM pin (the GEMM split's): on auto the 128K kv8 runs matched the pinned ones bit for bit with no machine checks
  // logged, and the pin's fabric clock costs the GPU ~10% at depth (results.md).
  explicit LoomNpuAttn(LoomDevice& gpu) : npu_(gpu, 8) {}
  Buffers Setup(const std::string& image, std::uint32_t kvn, std::uint32_t nbs, std::uint32_t B) override {
    if (!calls_.empty()) throw LoomError("npu attention: set up twice");
    constexpr std::size_t kRec = 4608, kOut = 34816, kMc = 16, kMcf = 17;
    const std::size_t slots = nbs + 1, half = slots * kRec, heads = std::size_t{2} * half, pay = 2 * kMcf * kRec;
    const std::size_t calls = std::size_t{3} * kvn * (B / 256), per_head = calls / kvn;
    LoomNpu::Shared& ks = npu_.CreateShared(kvn * heads);
    LoomNpu::Shared& vs = npu_.CreateShared(kvn * heads);
    LoomNpu::Shared& kn = npu_.CreateShared(std::size_t{4} * kvn * (16 * (nbs / 16 - 1) - 1));
    LoomNpu::Shared& kq = npu_.CreateShared(calls * pay);
    LoomNpu::Shared& vq = npu_.CreateShared(calls * pay);
    LoomNpu::Shared& out = npu_.CreateShared(calls * kMc * kOut, true);  // the NPU writes it (CreateShared)
    flags_ = &npu_.CreateShared(4096, true);
    // a call's views: its KV head's 4 record streams, its 16 cores' outputs, its 4 payload prefixes
    auto views = [&](std::size_t c) {
      const std::size_t g = c / per_head;
      std::vector<LoomNpu::View> v = {
          {&ks, g * heads, half}, {&ks, g * heads + half, half}, {&vs, g * heads, half}, {&vs, g * heads + half, half}};
      for (std::size_t p = 0; p < kMc; ++p) v.push_back({&out, (c * kMc + p) * kOut, kOut});
      for (LoomNpu::Shared* q : {&kq, &vq})
        for (std::size_t h = 0; h < 2; ++h) v.push_back({q, c * pay + h * kMcf * kRec, kMcf * kRec});
      return v;
    };
    image_ = npu_.LoadAttn(image, nbs, views(0));
    calls_.reserve(calls);
    for (std::size_t c = 0; c < calls; ++c) calls_.push_back(npu_.BindAttn(image_, views(c)));
    return {&ks.gpu, &vs.gpu, &kn.gpu, &kq.gpu, &vq.gpu, &out.gpu, &flags_->gpu};
  }
  void Enqueue(std::uint32_t nb, std::uint32_t ready, std::uint32_t done, std::uint32_t value,
               const std::string& tag) override {
    if (Word(NpuSplit::kFlagStatus)) throw LoomError("npu: a GPU wait for the NPU attention failed");
    LoomNpu::AttnJob job;
    for (const auto& c : calls_) job.calls.push_back(&c);
    job.nb = nb;
    job.ready = Words() + ready, job.ready_value = value;
    job.done = Words() + done, job.done_value = value;
    job.tag = tag;
    npu_.EnqueueAttn(job);
  }
  std::uint32_t Word(std::uint32_t word) override { return __atomic_load_n(Words() + word, __ATOMIC_ACQUIRE); }
  // Both after the GPU finished the work it was given (a job still without its ready word fails).
  void Finish() override {
    npu_.Drain(true);
    npu_.CheckHealth();
    if (Word(NpuSplit::kFlagStatus)) throw LoomError("npu: a GPU wait for the NPU attention failed");
  }
  void Report(std::FILE* f) {
    npu_.Drain(true);
    if (Word(NpuSplit::kFlagStatus)) std::fprintf(f, "npu: ERROR a GPU wait for the NPU attention failed\n");
    npu_.Report(f);
  }

 private:
  volatile std::uint32_t* Words() const { return static_cast<volatile std::uint32_t*>(flags_->host); }
  LoomNpu npu_;
  LoomNpu::Shared* flags_ = nullptr;
  LoomNpu::AttnImage image_;
  std::vector<LoomNpu::AttnCall> calls_;
};

}  // namespace yah::model

#endif  // YAH_MODEL_LOOM_NPU_HPP_
