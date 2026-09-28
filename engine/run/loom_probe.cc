// loom_probe: minimal HRX-native Loom dispatch probe.
//
// Loads a HAL executable emitted by iree-benchmark-loom --artifact-bundle-dir
// (hal_executables/*.hal), dispatches it over HRX buffers, and checks the
// residual-add result: out[i] = a[i] + b[i] with a = iota and b = iota + 100,
// so out is 100 + 2*i. No HIP anywhere.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "hrx_runtime.h"

static void Die(hrx_status_t status, const char* what) {
  if (hrx_status_is_ok(status)) return;
  char* message = nullptr;
  size_t length = 0;
  if (hrx_status_is_ok(hrx_status_to_string(status, &message, &length)) &&
      message) {
    std::fprintf(stderr, "%s: %.*s\n", what, static_cast<int>(length), message);
    hrx_status_free_message(message);
  } else {
    std::fprintf(stderr, "%s: unformattable hrx_status_t\n", what);
  }
  std::exit(1);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: loom_probe <hal_executable.hal> [target_key]\n");
    return 2;
  }
  const char* hal_path = argv[1];
  const char* target_key = argc > 2 ? argv[2] : "gfx1151";

  Die(hrx_gpu_initialize(0), "hrx_gpu_initialize");
  hrx_device_t device = nullptr;
  Die(hrx_gpu_device_get(0, &device), "hrx_gpu_device_get");
  hrx_stream_t stream = nullptr;
  Die(hrx_stream_create(device, 0, &stream), "hrx_stream_create");
  hrx_executable_t executable = nullptr;
  Die(hrx_executable_load_file(device, hal_path, "amdgpu", target_key,
                               &executable),
      "hrx_executable_load_file");

  size_t export_count = 0;
  Die(hrx_executable_export_count(executable, &export_count),
      "hrx_executable_export_count");
  std::fprintf(stderr, "exports: %zu\n", export_count);
  for (size_t i = 0; i < export_count; ++i) {
    hrx_executable_export_info_t info;
    Die(hrx_executable_export_info(executable, static_cast<uint32_t>(i), &info),
        "hrx_executable_export_info");
    std::fprintf(stderr, "  [%zu] name=%s bindings=%u params=%u consts=%u\n", i,
                 info.name ? info.name : "(null)", info.binding_count,
                 info.parameter_count, info.constant_byte_length);
  }
  uint32_t ordinal = 0;
  hrx_status_t lookup =
      hrx_executable_lookup_export_by_name(executable, "yah_residual_1d", &ordinal);
  if (!hrx_status_is_ok(lookup)) {
    hrx_status_ignore(lookup);
    std::fprintf(stderr, "export name not found; using ordinal 0\n");
  }

  const size_t dim = 8;
  const size_t bytes = dim * sizeof(float);
  hrx_buffer_t a = nullptr;
  hrx_buffer_t b = nullptr;
  hrx_buffer_t out = nullptr;
  Die(hrx_buffer_allocate(stream, bytes, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                          HRX_BUFFER_USAGE_DEFAULT, &a),
      "alloc a");
  Die(hrx_buffer_allocate(stream, bytes, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                          HRX_BUFFER_USAGE_DEFAULT, &b),
      "alloc b");
  Die(hrx_buffer_allocate(stream, bytes, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                          HRX_BUFFER_USAGE_DEFAULT, &out),
      "alloc out");

  std::vector<float> ha(dim), hb(dim), ho(dim, 0.0f);
  for (size_t i = 0; i < dim; ++i) {
    ha[i] = static_cast<float>(i);
    hb[i] = 100.0f + static_cast<float>(i);
  }
  Die(hrx_synchronous_h2d(device, ha.data(), a, 0, bytes), "h2d a");
  Die(hrx_synchronous_h2d(device, hb.data(), b, 0, bytes), "h2d b");

  hrx_dispatch_config_t config = {};
  config.workgroup_count[0] = 1;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = 256;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;
  hrx_buffer_ref_t bindings[3] = {
      {a, 0, bytes}, {b, 0, bytes}, {out, 0, bytes}};
  Die(hrx_stream_dispatch(stream, executable, ordinal, &config, nullptr, 0,
                          bindings, 3, HRX_DISPATCH_FLAG_NONE),
      "hrx_stream_dispatch");
  Die(hrx_stream_synchronize(stream), "hrx_stream_synchronize");
  Die(hrx_synchronous_d2h(device, out, 0, ho.data(), bytes), "d2h out");

  int failures = 0;
  for (size_t i = 0; i < dim; ++i) {
    const float want = 100.0f + 2.0f * static_cast<float>(i);
    if (ho[i] != want) {
      std::fprintf(stderr, "  out[%zu] = %f, want %f\n", i, ho[i], want);
      ++failures;
    }
  }
  if (failures == 0) {
    std::printf("LOOM PROBE PASS: out = 100 + 2*i for %zu elements\n", dim);
  } else {
    std::printf("LOOM PROBE FAIL: %d mismatches\n", failures);
  }
  hrx_buffer_release(a);
  hrx_buffer_release(b);
  hrx_buffer_release(out);
  hrx_executable_release(executable);
  hrx_stream_release(stream);
  hrx_device_release(device);
  Die(hrx_gpu_shutdown(), "hrx_gpu_shutdown");
  return failures == 0 ? 0 : 1;
}