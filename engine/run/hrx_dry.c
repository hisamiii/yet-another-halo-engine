// hrx_dry: a stand-in libhrx.so.0 that runs nothing on the GPU (engine/run/hrx_dry.sh puts it first on the library path).
// Executables load through the real libhrx (HRX_DRY_REAL, its own link namespace) for their export info; buffers are
// sizes, copies probe their host pages on the calling thread, and every dispatch, graph node and copy is checked against
// the buffers it names: a view past its buffer hangs the GPU in a graph. A HAL's <path>.fp (footprint_gate.py) gives
// the bytes its kernel may touch per binding; a view shorter than that hangs it too. Prints the errors and the memory a
// real run needs.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "hrx_runtime.h"

struct hrx_status_s {
  hrx_status_code_t code;
  char message[256];
};
struct hrx_buffer_s {
  size_t size;
  int kind;    // 0 device-local, 1 host-local, 2 imported host memory
  int alive;   // released buffers stay allocated, so a later use is reported
  void* host;  // host-local: its backing; imported: the host range
  uintptr_t va;
};
struct hrx_executable_s {
  void* real;
  char* path;
  size_t count;
  hrx_executable_export_info_t* infos;
  size_t envelope[32];  // <path>.fp: bytes the kernel may touch through each binding (0: not declared)
  uint64_t grid;        // the compiled grid's workgroups the envelopes are for (0: not known)
  int has_envelope;
  size_t unchecked;     // dispatches of a HAL with no <path>.fp
};
static hrx_executable_t executables[1024];
static size_t executable_count;
struct hrx_stream_s { int unused; };
struct hrx_event_s { int unused; };
struct hrx_graph_s { size_t nodes; };
struct hrx_graph_exec_s { struct hrx_graph_s* graph; };

static void* real_lib;
static __typeof__(hrx_gpu_initialize)* r_initialize;
static __typeof__(hrx_gpu_device_get)* r_device_get;
static __typeof__(hrx_gpu_shutdown)* r_shutdown;
static __typeof__(hrx_executable_load_file)* r_load_file;
static __typeof__(hrx_executable_export_count)* r_export_count;
static __typeof__(hrx_executable_export_info)* r_export_info;
static __typeof__(hrx_status_to_string)* r_to_string;
static __typeof__(hrx_status_free_message)* r_free_message;

static size_t enveloped, errors, dispatches, graph_nodes, copies, device_now, device_peak, host_now, imported, ram_peak;
static uintptr_t next_va = 0x100000000000ull;

// what a real run holds now: the process's resident pages plus the device-local buffers (GTT pages on this machine)
static void sample(void) {
  long pages = 0, resident = 0;
  FILE* f = fopen("/proc/self/statm", "r");
  if (f && fscanf(f, "%ld %ld", &pages, &resident) == 2) {
    const size_t now = (size_t)resident * 4096 + device_now;
    if (now > ram_peak) ram_peak = now;
  }
  if (f) fclose(f);
}

static hrx_status_t make(hrx_status_code_t code, const char* fmt, ...) {
  struct hrx_status_s* s = calloc(1, sizeof *s);
  s->code = code;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(s->message, sizeof s->message, fmt, ap);
  va_end(ap);
  return s;
}

// each distinct message once, with its count in the report
static struct {
  char text[640];
  size_t count;
} distinct[256];
static size_t distinct_count;
static void error(const char* fmt, ...) {
  ++errors;
  char text[640];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(text, sizeof text, fmt, ap);
  va_end(ap);
  for (size_t i = 0; i < distinct_count; ++i)
    if (!strcmp(distinct[i].text, text)) return (void)++distinct[i].count;
  if (distinct_count == 256) return;
  strcpy(distinct[distinct_count].text, text);
  distinct[distinct_count++].count = 1;
}

// a real libhrx status, as one of this library's
static hrx_status_t from_real(hrx_status_t s) {
  if (!s) return NULL;
  char* m = NULL;
  size_t n = 0;
  hrx_status_t f = r_to_string(s, &m, &n);
  hrx_status_t out = make(HRX_STATUS_INTERNAL, "%s", (!f && m) ? m : "real libhrx error");
  if (m) r_free_message(m);
  return out;
}

static void on_fault(int sig) {
  void* frames[64];
  const int n = backtrace(frames, 64);
  fprintf(stderr, "hrx_dry: signal %d, backtrace (addr2line -e <binary> -f -C <offset>):\n", sig);
  backtrace_symbols_fd(frames, n, 2);
  signal(sig, SIG_DFL);
  raise(sig);
}

// one byte of every page of [p, p + n), so an unmapped source faults here, on the caller's stack
static void probe(const void* p, size_t n) {
  if (!n) return;
  const volatile unsigned char* c = p;
  unsigned sum = 0;
  for (size_t i = 0; i < n; i += 4096) sum += c[i];
  sum += c[n - 1];
  (void)sum;
}

static int check_ref(const char* what, hrx_buffer_t b, size_t offset, size_t length) {
  if (!b) return error("%s: null buffer", what), 0;
  if (!b->alive) return error("%s: buffer of %zu bytes used after release", what, b->size), 0;
  if (offset > b->size || length > b->size - offset)
    return error("%s: view [%zu, +%zu) = end %zu past the buffer's %zu bytes", what, offset, length, offset + length,
                 b->size), 0;
  return 1;
}

static void check_dispatch(const char* how, hrx_executable_t e, uint32_t ordinal, const hrx_dispatch_config_t* config,
                           size_t constants_size, const hrx_buffer_ref_t* bindings, size_t binding_count) {
  ++dispatches;
  if (!e || ordinal >= e->count) {
    error("%s: export %u of %s does not exist", how, ordinal, e ? e->path : "(null executable)");
    return;
  }
  const hrx_executable_export_info_t* info = &e->infos[ordinal];
  char what[512];
  if (binding_count != info->binding_count)
    error("%s %s (%s): %zu bindings, the kernel has %u", how, info->name, e->path, binding_count, info->binding_count);
  if (constants_size != info->constant_byte_length)
    error("%s %s (%s): %zu constant bytes, the kernel takes %u", how, info->name, e->path, constants_size,
          info->constant_byte_length);
  for (int d = 0; d < 3; ++d) {
    if (!config->workgroup_count[d]) error("%s %s (%s): workgroup_count[%d] = 0", how, info->name, e->path, d);
    if (info->workgroup_size[d] && config->workgroup_size[d] && config->workgroup_size[d] != info->workgroup_size[d])
      error("%s %s (%s): workgroup_size[%d] %u, the kernel requires %u", how, info->name, e->path, d,
            config->workgroup_size[d], info->workgroup_size[d]);
  }
  if (!e->has_envelope) ++e->unchecked;
  // a dispatch of fewer workgroups than the compiled grid (the last layer's trim to one token tile, prepab's
  // max(alpha / beta, ring) lanes) touches that share of each envelope
  const uint64_t run = (uint64_t)config->workgroup_count[0] * config->workgroup_count[1] * config->workgroup_count[2];
  const uint64_t grid = e->grid ? e->grid : run;
  for (size_t i = 0; i < binding_count; ++i) {
    snprintf(what, sizeof what, "%s %s (%s) binding %zu", how, info->name, e->path, i);
    check_ref(what, bindings[i].buffer, bindings[i].offset, bindings[i].length);
    const size_t envelope = i < 32 ? e->envelope[i] : 0;
    const size_t need = run < grid ? (size_t)((envelope * run + grid - 1) / grid) : envelope;
    // past its buffer, the read lands on unmapped VA and the ring hangs; inside it, it reads a neighbor's bytes
    if (bindings[i].length < need && bindings[i].buffer)
      error("%s: the kernel touches %zu bytes (%llu of its %llu workgroups), the view has %zu: %s", what, need,
            (unsigned long long)run, (unsigned long long)grid, bindings[i].length,
            bindings[i].offset + need > bindings[i].buffer->size ? "PAST ITS BUFFER" : "inside its buffer");
  }
  for (size_t i = binding_count; i < 32; ++i)
    if (e->envelope[i]) error("%s %s (%s): no binding %zu, the kernel touches it", how, info->name, e->path, i);
}

static int reported;
static void drain(void);
static void report(void) {
  if (reported++) return;
  drain();
  sample();
  for (size_t i = 0; i < distinct_count; ++i)
    fprintf(stderr, "hrx_dry: ERROR (x%zu) %s\n", distinct[i].count, distinct[i].text);
  for (size_t i = 0; i < executable_count; ++i)
    if (executables[i]->unchecked)
      fprintf(stderr, "hrx_dry: no envelopes (%zu dispatches): %s\n", executables[i]->unchecked, executables[i]->path);
  const double gb = 1e9;
  fprintf(stderr,
          "hrx_dry: %zu dispatches, %zu graph nodes, %zu copies checked (%zu HALs with envelopes): %zu errors\n"
          "hrx_dry: device-local peak %.2f GB, host-local %.2f GB, imported %.2f GB\n"
          "hrx_dry: a real run peaks at about %.2f GB of RAM (resident pages plus device-local buffers)\n",
          dispatches, graph_nodes, copies, enveloped, errors, device_peak / gb, host_now / gb, imported / gb, ram_peak / gb);
}

hrx_status_t hrx_gpu_initialize(uint32_t flags) {
  const char* path = getenv("HRX_DRY_REAL");
  if (!path) return make(HRX_STATUS_FAILED_PRECONDITION, "hrx_dry: HRX_DRY_REAL names no real libhrx");
  real_lib = dlmopen(LM_ID_NEWLM, path, RTLD_NOW | RTLD_LOCAL);
  if (!real_lib) return make(HRX_STATUS_NOT_FOUND, "hrx_dry: %s", dlerror());
  r_initialize = dlsym(real_lib, "hrx_gpu_initialize");
  r_device_get = dlsym(real_lib, "hrx_gpu_device_get");
  r_shutdown = dlsym(real_lib, "hrx_gpu_shutdown");
  r_load_file = dlsym(real_lib, "hrx_executable_load_file");
  r_export_count = dlsym(real_lib, "hrx_executable_export_count");
  r_export_info = dlsym(real_lib, "hrx_executable_export_info");
  r_to_string = dlsym(real_lib, "hrx_status_to_string");
  r_free_message = dlsym(real_lib, "hrx_status_free_message");
  atexit(report);
  signal(SIGSEGV, on_fault);
  signal(SIGBUS, on_fault);
  fprintf(stderr, "hrx_dry: no GPU work runs; checking every binding\n");
  return from_real(r_initialize(flags));
}
hrx_status_t hrx_gpu_device_get(int index, hrx_device_t* device) { return from_real(r_device_get(index, device)); }
hrx_status_t hrx_gpu_shutdown(void) {
  report();
  hrx_status_t s = from_real(r_shutdown());
  if (errors) {
    fflush(stdout);
    _exit(3);
  }
  return s;
}

hrx_status_code_t hrx_status_code(hrx_status_t s) { return s ? s->code : HRX_STATUS_OK; }
hrx_status_t hrx_status_to_string(hrx_status_t s, char** out_message, size_t* out_length) {
  *out_message = strdup(s ? s->message : "OK");
  if (out_length) *out_length = strlen(*out_message);
  return NULL;
}
void hrx_status_free_message(char* message) { free(message); }
void hrx_status_ignore(hrx_status_t s) { free(s); }

hrx_status_t hrx_stream_create(hrx_device_t device, uint32_t flags, hrx_stream_t* stream) {
  (void)device, (void)flags;
  *stream = calloc(1, sizeof **stream);
  return NULL;
}
void hrx_stream_release(hrx_stream_t stream) { free(stream); }
hrx_status_t hrx_stream_synchronize(hrx_stream_t stream) { return (void)stream, drain(), NULL; }
hrx_status_t hrx_stream_flush(hrx_stream_t stream) { return (void)stream, NULL; }
hrx_status_t hrx_stream_execution_barrier(hrx_stream_t stream) { return (void)stream, NULL; }
hrx_status_t hrx_stream_get_timeline_position(hrx_stream_t stream, hrx_timeline_point_t* position) {
  (void)stream;
  static int timeline;   // a semaphore, so the caller waits on the point (the wait reads its pending uploads)
  position->semaphore = (hrx_semaphore_t)&timeline, position->value = 1;
  return NULL;
}
hrx_status_t hrx_semaphore_wait(hrx_semaphore_t semaphore, uint64_t value, uint64_t timeout_ns) {
  return (void)semaphore, (void)value, (void)timeout_ns, drain(), NULL;
}

static hrx_buffer_t new_buffer(size_t size, int kind, void* host) {
  hrx_buffer_t b = calloc(1, sizeof *b);
  b->size = size, b->kind = kind, b->alive = 1, b->host = host, b->va = next_va;
  next_va += (size + 0xffff) & ~(uintptr_t)0xffff;
  return b;
}
hrx_status_t hrx_buffer_allocate(hrx_stream_t stream, size_t size, hrx_memory_type_t mem_type,
                                 hrx_buffer_usage_t usage, hrx_buffer_t* buffer) {
  (void)stream, (void)usage;
  if (!size) return make(HRX_STATUS_INVALID_ARGUMENT, "hrx_dry: zero-byte allocation");
  if (mem_type & HRX_MEMORY_TYPE_HOST_LOCAL) {
    void* p = calloc(1, size);
    if (!p) return make(HRX_STATUS_OUT_OF_MEMORY, "hrx_dry: host-local %zu bytes", size);
    host_now += size;
    *buffer = new_buffer(size, 1, p);
  } else {
    device_now += size;
    if (device_now > device_peak) device_peak = device_now;
    sample();
    *buffer = new_buffer(size, 0, NULL);
  }
  return NULL;
}
hrx_allocator_t hrx_device_allocator(hrx_device_t device) { return (hrx_allocator_t)device; }
hrx_status_t hrx_allocator_import_buffer(hrx_allocator_t allocator, hrx_buffer_params_t params, void* host_ptr,
                                         size_t size, hrx_buffer_t* buffer) {
  (void)allocator, (void)params;
  if ((uintptr_t)host_ptr & 4095) error("import: host pointer %p is not page aligned", host_ptr);
  probe(host_ptr, size);
  imported += size;
  *buffer = new_buffer(size, 2, host_ptr);
  return NULL;
}
void hrx_buffer_release(hrx_buffer_t b) {
  if (!b) return;
  if (!b->alive) error("buffer of %zu bytes released twice", b->size);
  if (b->kind == 0) device_now -= b->size;
  if (b->kind == 1) host_now -= b->size, free(b->host), b->host = NULL;
  b->alive = 0;
}
hrx_status_t hrx_buffer_map(hrx_buffer_t b, hrx_map_flags_t flags, size_t offset, size_t size, void** mapped_ptr) {
  (void)flags;
  if (!check_ref("map", b, offset, size)) return make(HRX_STATUS_OUT_OF_RANGE, "hrx_dry: map");
  if (b->kind == 0) return make(HRX_STATUS_INVALID_ARGUMENT, "hrx_dry: mapping a device-local buffer");
  *mapped_ptr = (char*)b->host + offset;
  return NULL;
}
hrx_status_t hrx_buffer_get_device_ptr(hrx_buffer_t b, void** device_ptr) {
  *device_ptr = (void*)b->va;
  return NULL;
}

// A stream upload reads its source on the queue thread, any time before the next sync: it is read at that sync here,
// the latest point. A page that is no longer resident by then was released under the copy (the model is mlocked).
struct pending {
  const char* what;
  const void* src;
  hrx_buffer_t dst;
  size_t off, n;
};
static struct pending pendings[4096];
static size_t pending_count;
static void upload(const char* what, const void* src, hrx_buffer_t dst, size_t off, size_t n) {
  const uintptr_t lo = (uintptr_t)src & ~(uintptr_t)4095, hi = ((uintptr_t)src + n + 4095) & ~(uintptr_t)4095;
  unsigned char vec[4096];
  for (uintptr_t a = lo; a < hi; a += (uintptr_t)sizeof vec * 4096) {
    const size_t pages = (hi - a) / 4096 < sizeof vec ? (hi - a) / 4096 : sizeof vec;
    if (mincore((void*)a, pages * 4096, vec)) {
      error("%s: source %p (+%zu) is not mapped when the copy runs", what, src, n);
      return;
    }
    for (size_t i = 0; i < pages; ++i)
      if (!(vec[i] & 1)) {
        error("%s: source page %p (of %p +%zu) was released before the copy ran", what, (void*)(a + i * 4096), src, n);
        return;
      }
  }
  probe(src, n);
  if (dst->kind != 0) memcpy((char*)dst->host + off, src, n);
}
static void drain(void) {
  for (size_t i = 0; i < pending_count; ++i)
    upload(pendings[i].what, pendings[i].src, pendings[i].dst, pendings[i].off, pendings[i].n);
  pending_count = 0;
}
static hrx_status_t h2d(const char* what, const void* src, hrx_buffer_t dst, size_t off, size_t n, int queued) {
  ++copies;
  if (!check_ref(what, dst, off, n)) return NULL;
  if (n > ((size_t)1 << 26)) sample();
  if (!queued) {
    drain();
    upload(what, src, dst, off, n);
    return NULL;
  }
  if (pending_count == sizeof pendings / sizeof *pendings) drain();
  pendings[pending_count++] = (struct pending){what, src, dst, off, n};
  return NULL;
}
hrx_status_t hrx_synchronous_h2d(hrx_device_t device, const void* src, hrx_buffer_t dst, size_t off, size_t n) {
  return (void)device, h2d("synchronous_h2d", src, dst, off, n, 0);
}
hrx_status_t hrx_stream_copy_h2d(hrx_stream_t stream, const void* src, hrx_buffer_t dst, size_t off, size_t n) {
  return (void)stream, h2d("stream_copy_h2d", src, dst, off, n, 1);
}
hrx_status_t hrx_stream_update_buffer(hrx_stream_t stream, const void* src, size_t n, hrx_buffer_t dst, size_t off) {
  (void)stream;
  if (n > 65536) error("update_buffer: %zu bytes, the limit is 64 KB", n);
  return h2d("update_buffer", src, dst, off, n, 0);   // (recorded into the command: copied now)
}
hrx_status_t hrx_synchronous_d2h(hrx_device_t device, hrx_buffer_t src, size_t off, void* dst, size_t n) {
  (void)device;
  drain();
  ++copies;
  if (!check_ref("synchronous_d2h", src, off, n)) return NULL;
  if (src->kind != 0) memcpy(dst, (char*)src->host + off, n);
  else memset(dst, 0, n);
  return NULL;
}
hrx_status_t hrx_stream_copy_buffer(hrx_stream_t stream, hrx_buffer_t src, size_t src_off, hrx_buffer_t dst,
                                    size_t dst_off, size_t n) {
  (void)stream;
  ++copies;
  if (!check_ref("copy_buffer source", src, src_off, n) || !check_ref("copy_buffer target", dst, dst_off, n))
    return NULL;
  if (dst->kind == 0) return NULL;
  if (src->kind != 0) memmove((char*)dst->host + dst_off, (char*)src->host + src_off, n);
  else memset((char*)dst->host + dst_off, 0, n);
  return NULL;
}
hrx_status_t hrx_stream_fill_buffer(hrx_stream_t stream, hrx_buffer_t b, size_t off, size_t n, const void* pattern,
                                    size_t pattern_size) {
  (void)stream;
  ++copies;
  if (!check_ref("fill_buffer", b, off, n)) return NULL;
  if (pattern_size == 0 || n % pattern_size) error("fill_buffer: %zu bytes, pattern of %zu", n, pattern_size);
  if (b->kind != 0)
    for (size_t i = 0; i + pattern_size <= n; i += pattern_size) memcpy((char*)b->host + off + i, pattern, pattern_size);
  return NULL;
}

hrx_status_t hrx_event_create(hrx_device_t device, hrx_event_flags_t flags, hrx_event_t* event) {
  (void)device, (void)flags;
  *event = calloc(1, sizeof **event);
  return NULL;
}
void hrx_event_release(hrx_event_t event) { free(event); }
hrx_status_t hrx_event_record(hrx_event_t event, hrx_stream_t stream) { return (void)event, (void)stream, NULL; }
hrx_status_t hrx_event_synchronize(hrx_event_t event) { return (void)event, drain(), NULL; }
hrx_status_t hrx_event_query(hrx_event_t event, bool* complete) {
  (void)event;
  drain();
  *complete = true;
  return NULL;
}
hrx_status_t hrx_event_elapsed_time(hrx_event_t start, hrx_event_t stop, float* ms) {
  (void)start, (void)stop;
  *ms = 0;
  return NULL;
}

hrx_status_t hrx_executable_load_file(hrx_device_t device, const char* path, const char* target_family,
                                      const char* target_key, hrx_executable_t* executable) {
  void* real = NULL;
  hrx_status_t s = from_real(r_load_file(device, path, target_family, target_key, (hrx_executable_t*)&real));
  if (s) return s;
  hrx_executable_t e = calloc(1, sizeof *e);
  e->real = real, e->path = strdup(path);
  if ((s = from_real(r_export_count(real, &e->count)))) return s;
  e->infos = calloc(e->count ? e->count : 1, sizeof *e->infos);
  for (size_t i = 0; i < e->count; ++i) {
    if ((s = from_real(r_export_info(real, (uint32_t)i, &e->infos[i])))) return s;
    e->infos[i].name = strdup(e->infos[i].name ? e->infos[i].name : "");
  }
  char fp[4096];
  snprintf(fp, sizeof fp, "%s.fp", path);
  FILE* f = fopen(fp, "r");
  char line[256];
  unsigned arg;
  size_t bytes;
  if (f) ++enveloped, e->has_envelope = 1;
  while (f && fgets(line, sizeof line, f)) {
    unsigned gx, gy, gz;
    if (sscanf(line, "grid %u %u %u", &gx, &gy, &gz) == 3) {
      e->grid = (uint64_t)gx * gy * gz;
      continue;
    }
    if (sscanf(line, "%u %zu", &arg, &bytes) == 2 && arg < 32) e->envelope[arg] = bytes;
  }
  if (f) fclose(f);
  if (executable_count < 1024) executables[executable_count++] = e;
  *executable = e;
  return NULL;
}
hrx_status_t hrx_executable_export_count(hrx_executable_t e, size_t* count) {
  *count = e->count;
  return NULL;
}
hrx_status_t hrx_executable_export_info(hrx_executable_t e, uint32_t ordinal, hrx_executable_export_info_t* info) {
  if (ordinal >= e->count) return make(HRX_STATUS_OUT_OF_RANGE, "hrx_dry: export %u of %s", ordinal, e->path);
  *info = e->infos[ordinal];
  return NULL;
}
void hrx_executable_release(hrx_executable_t e) { (void)e; }   // (kept: dispatch errors name its path)

hrx_status_t hrx_stream_dispatch(hrx_stream_t stream, hrx_executable_t e, uint32_t ordinal,
                                 const hrx_dispatch_config_t* config, const void* constants, size_t constants_size,
                                 const hrx_buffer_ref_t* bindings, size_t binding_count, uint32_t flags) {
  (void)stream, (void)constants, (void)flags;
  check_dispatch("dispatch", e, ordinal, config, constants_size, bindings, binding_count);
  return NULL;
}

hrx_status_t hrx_graph_create(hrx_device_t device, uint32_t flags, hrx_graph_t* graph) {
  (void)device, (void)flags;
  *graph = calloc(1, sizeof **graph);
  return NULL;
}
void hrx_graph_release(hrx_graph_t graph) { free(graph); }
static hrx_graph_node_t new_node(hrx_graph_t g) {
  ++graph_nodes;
  return (hrx_graph_node_t)(uintptr_t)++g->nodes;
}
hrx_status_t hrx_graph_add_kernel_node(hrx_graph_t g, const hrx_graph_node_t* deps, size_t dep_count,
                                       const hrx_graph_kernel_node_attrs_t* a, hrx_graph_node_t* out_node) {
  (void)deps, (void)dep_count;
  check_dispatch("graph node", a->executable, a->export_ordinal, &a->config, a->constants_size, a->bindings,
                 a->binding_count);
  --dispatches;
  *out_node = new_node(g);
  return NULL;
}
hrx_status_t hrx_graph_add_atomic_store_node(hrx_graph_t g, const hrx_graph_node_t* deps, size_t dep_count,
                                             const hrx_graph_atomic_store_node_attrs_t* a, hrx_graph_node_t* out_node) {
  (void)deps, (void)dep_count;
  check_ref("graph atomic store", a->target.buffer, a->target.offset, a->target.length);
  *out_node = new_node(g);
  return NULL;
}
hrx_status_t hrx_graph_instantiate(hrx_graph_t g, uint32_t flags, hrx_graph_exec_t* exec) {
  (void)flags;
  *exec = calloc(1, sizeof **exec);
  (*exec)->graph = g;
  return NULL;
}
void hrx_graph_exec_release(hrx_graph_exec_t exec) { free(exec); }
hrx_status_t hrx_graph_exec_launch(hrx_graph_exec_t exec, hrx_stream_t stream) { return (void)exec, (void)stream, NULL; }
