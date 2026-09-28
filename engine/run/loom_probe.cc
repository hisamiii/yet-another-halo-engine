// loom_probe: minimal HRX-native Loom dispatch probe using the runtime layer.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "model/loom_runtime.hpp"

using namespace yah::model;  // NOLINT(google-build-using-namespace)

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: loom_probe <hal_executable.hal> [target_key]\n");
    return 2;
  }
  try {
    LoomDevice gpu;
    LoomExecutable executable = gpu.Load(argv[1], argc > 2 ? argv[2] : "gfx1151");
    std::fprintf(stderr, "exports: %zu\n", executable.infos.size());
    for (size_t i = 0; i < executable.infos.size(); ++i) {
      const auto& info = executable.infos[i];
      std::fprintf(stderr, "  [%zu] name=%s bindings=%u params=%u consts=%u\n",
                   i, executable.names[i].c_str(), info.binding_count,
                   info.parameter_count, info.constant_byte_length);
    }
    const uint32_t ordinal = executable.OrdinalOrZero("yah_residual_1d");

    const size_t dim = 8;
    const size_t bytes = dim * sizeof(float);
    LoomBuffer a = gpu.Allocate(bytes);
    LoomBuffer b = gpu.Allocate(bytes);
    LoomBuffer out = gpu.Allocate(bytes);
    std::vector<float> ha(dim), hb(dim), ho(dim, 0.0f);
    for (size_t i = 0; i < dim; ++i) {
      ha[i] = static_cast<float>(i);
      hb[i] = 100.0f + static_cast<float>(i);
    }
    gpu.H2D(a, ha.data(), bytes);
    gpu.H2D(b, hb.data(), bytes);

    const hrx_buffer_ref_t bindings[3] = {
        {a.handle, 0, bytes}, {b.handle, 0, bytes}, {out.handle, 0, bytes}};
    gpu.Dispatch(executable, ordinal, LoomDevice::Config(1, 1, 1, 256, 1, 1),
                 nullptr, 0, bindings, 3);
    gpu.Synchronize();
    gpu.D2H(out, ho.data(), bytes);

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
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "loom_probe: %s\n", error.what());
    return 1;
  }
}