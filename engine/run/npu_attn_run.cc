// NPU attention calls through LoomNpu's host-gated runner (tools/npu_attn_check.py --runner drives it).
//
// usage: npu_attn_run <image.xdna> <dir> <mc> <nb_built> <nb> <calls> <jobs> <out_bytes>
//
// dir holds the streams npu_attn_check writes: s0..s3.bin (the KV head's records, sized for nb_built) and q0..q3.bin
// (a call's payloads). Every call binds the shared records, its own copy of the payloads and its own outputs; a job
// runs all calls as one command, gated on a ready word the GPU stores from a graph. Writes the last job's outputs,
// out<call>_<pair>.bin, and prints the NPU time per job. With <passes> <ref_dir> (a fresh call's out<pair>.bin): runs
// the jobs that many times and checks every call's outputs against the reference.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "model/loom_npu.hpp"
#include "model/loom_runtime.hpp"

using yah::model::LoomBuffer;
using yah::model::LoomDevice;
using yah::model::LoomError;
using yah::model::LoomGraph;
using yah::model::LoomNpu;

namespace {

std::vector<char> ReadAll(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw LoomError("cannot open " + path);
  std::vector<char> data(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(data.data(), static_cast<std::streamsize>(data.size()));
  return data;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 9 && argc != 11) {
    std::fprintf(stderr,
                 "usage: npu_attn_run <image.xdna> <dir> <mc> <nb_built> <nb> <calls> <jobs> <out_bytes> "
                 "[<passes> <ref_dir>]\n");
    return 2;
  }
  try {
    const std::string image = argv[1], dir = argv[2];
    const std::uint32_t mc = std::atoi(argv[3]), nb_built = std::atoi(argv[4]), nb = std::atoi(argv[5]);
    const std::size_t calls = std::atoi(argv[6]), jobs = std::atoi(argv[7]), ob = std::atoll(argv[8]);
    std::vector<std::vector<char>> recs, pays;
    for (int s = 0; s < 4; ++s) {
      recs.push_back(ReadAll(dir + "/s" + std::to_string(s) + ".bin"));
      pays.push_back(ReadAll(dir + "/q" + std::to_string(s) + ".bin"));
    }
    const std::size_t rb = recs[0].size(), pb = pays[0].size();

    LoomDevice gpu;
    LoomNpu npu(gpu, std::max(2u, std::min(8u, mc)));  // the image's columns
    LoomNpu::Shared& records = npu.CreateShared(4 * rb);
    LoomNpu::Shared& payloads = npu.CreateShared(calls * 4 * pb);
    LoomNpu::Shared& outputs = npu.CreateShared(calls * mc * ob, true);
    LoomNpu::Shared& flags = npu.CreateShared(4096, true);
    for (int s = 0; s < 4; ++s) {
      gpu.H2D(records.gpu, recs[s].data(), rb, s * rb);
      for (std::size_t c = 0; c < calls; ++c) gpu.H2D(payloads.gpu, pays[s].data(), pb, (c * 4 + s) * pb);
    }
    gpu.Fill(outputs.gpu, 0x7fc00000u);  // NaN: bytes no call wrote show
    gpu.Synchronize();
    auto views = [&](std::size_t c) {
      std::vector<LoomNpu::View> v;
      for (int s = 0; s < 4; ++s) v.push_back({&records, s * rb, rb});
      for (std::uint32_t p = 0; p < mc; ++p) v.push_back({&outputs, (c * mc + p) * ob, ob});
      for (int s = 0; s < 4; ++s) v.push_back({&payloads, (c * 4 + s) * pb, pb});
      return v;
    };
    const LoomNpu::AttnImage a = npu.LoadAttn(image, nb_built, views(0));
    std::vector<LoomNpu::AttnCall> bound;
    for (std::size_t c = 0; c < calls; ++c) bound.push_back(npu.BindAttn(a, views(c)));
    std::printf("command bytes: first %zu, continuation %zu\n", bound[0].first.size(), bound[0].cont.size());

    auto* words = static_cast<volatile std::uint32_t*>(flags.host);  // ready j at 2 j, done j at 2 j + 1
    const std::size_t obytes = calls * mc * ob;
    const auto* host = static_cast<const std::uint8_t*>(outputs.host);
    // Runs the jobs once (pass it: ready and done values it + 1); returns each job's done time.
    auto run = [&](std::uint32_t it) {
      for (std::size_t j = 0; j < jobs; ++j) {
        LoomNpu::AttnJob job;
        for (const auto& b : bound) job.calls.push_back(&b);
        job.nb = nb;
        job.ready = &words[2 * j], job.ready_value = it + 1;
        job.done = &words[2 * j + 1], job.done_value = it + 1;
        job.tag = "attn";
        npu.EnqueueAttn(job);
      }
      // the GPU stores the ready words (release, system scope: as after a layer's encoders)
      const auto t0 = std::chrono::steady_clock::now();
      {
        LoomGraph g(gpu);
        for (std::size_t j = 0; j < jobs; ++j)
          g.AtomicStore({flags.gpu.handle, 8 * j, 4}, it + 1, HRX_ATOMIC_FLAG_RELEASE | HRX_ATOMIC_FLAG_SYSTEM_SCOPE,
                        {{records.gpu.handle, 0, 4 * rb}, {payloads.gpu.handle, 0, calls * 4 * pb}}, {});
        g.Launch();
      }
      std::vector<double> done_ms(jobs, 0);
      for (std::size_t j = 0; j < jobs; ++j) {
        while (__atomic_load_n(&words[2 * j + 1], __ATOMIC_ACQUIRE) < it + 1) std::this_thread::yield();
        done_ms[j] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (words[2 * j + 1] != it + 1) throw LoomError("job " + std::to_string(j) + " failed");
      }
      npu.Drain();
      npu.CheckHealth();
      gpu.Synchronize();
      return done_ms;
    };
    auto flush = [&] {
      for (std::size_t i = 0; i < obytes; i += 64) _mm_clflush(host + i);
      _mm_mfence();
    };

    // soak: every call of every pass against a fresh call's outputs, as the GPU reads them (a copy) and as the CPU does
    // after invalidating its cached lines; the CPU's plain read is counted too: the NPU's writes do not invalidate the
    // CPU's caches, so lines it cached before the NPU wrote stay stale
    if (argc == 11) {
      const int iters = std::atoi(argv[9]);
      std::vector<std::vector<char>> ref;
      for (std::uint32_t p = 0; p < mc; ++p)
        ref.push_back(ReadAll(std::string(argv[10]) + "/out" + std::to_string(p) + ".bin"));
      LoomBuffer gcopy = gpu.Allocate(obytes);
      std::vector<std::uint8_t> got(obytes);
      // lines (64 B) of O (the first 16 KB of each core's 17 KB) that differ from the reference
      auto bad_lines = [&](const std::uint8_t* b, std::string* where) {
        std::size_t n = 0;
        for (std::size_t c = 0; c < calls; ++c)
          for (std::uint32_t p = 0; p < mc; ++p)
            for (std::size_t h = 0; h < 2; ++h)
              for (std::size_t l = 0; l < 16384; l += 64) {
                const std::size_t at = (c * mc + p) * ob + h * (ob / 2) + l;
                if (std::memcmp(b + at, ref[p].data() + h * (ob / 2) + l, 64)) {
                  if (where && n < 4) *where += " " + std::to_string(at);
                  ++n;
                }
              }
        return n;
      };
      std::size_t gpu_bad = 0, cpu_bad = 0, flushed_bad = 0, passes_bad = 0;
      for (int it = 0; it < iters; ++it) {
        gpu.Fill(outputs.gpu, 0x7fc00000u);
        gpu.Synchronize();
        run(static_cast<std::uint32_t>(it));
        gpu.Copy({outputs.gpu.handle, 0, obytes}, {gcopy.handle, 0, obytes});
        gpu.Synchronize();
        gpu.D2H(gcopy, got.data(), obytes);
        std::string wg, wc, wf;
        const std::size_t g = bad_lines(got.data(), &wg);
        const std::size_t c = bad_lines(host, &wc);
        flush();
        const std::size_t f = bad_lines(host, &wf);
        gpu_bad += g, cpu_bad += c, flushed_bad += f, passes_bad += (g || f);
        if (g || c || f)
          std::printf("pass %d: bad lines gpu %zu (%s ), cpu %zu (%s ), cpu flushed %zu (%s )\n", it, g, wg.c_str(), c,
                      wc.c_str(), f, wf.c_str());
      }
      std::printf("soak: %d passes, %zu bad; bad lines gpu %zu, cpu flushed %zu (cpu unflushed %zu)\n", iters,
                  passes_bad, gpu_bad, flushed_bad, cpu_bad);
      if (passes_bad) throw LoomError("soak: outputs differ");
    } else {
      const std::vector<double> done_ms = run(0);
      for (std::size_t j = 0; j < jobs; ++j)
        std::printf("job %zu: done at %.3f ms (%.3f ms after the previous)\n", j, done_ms[j],
                    done_ms[j] - (j ? done_ms[j - 1] : 0));
      const auto st = npu.Commands();
      std::printf("commands %zu, NPU busy %.3f ms, longest %.3f ms, bytes %zu\n", st.commands, st.busy_ms, st.max_ms,
                  st.bytes);
      flush();
      for (std::size_t c = 0; c < calls; ++c)
        for (std::uint32_t p = 0; p < mc; ++p) {
          std::ofstream f(dir + "/out" + std::to_string(c) + "_" + std::to_string(p) + ".bin", std::ios::binary);
          f.write(reinterpret_cast<const char*>(host) + (c * mc + p) * ob, static_cast<std::streamsize>(ob));
        }
    }
    std::printf("npu_attn_run: ok\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "npu_attn_run: %s\n", e.what());
    return 1;
  }
  return 0;
}
