// YAH_NPU_PROF (CPU-driven runs, YAH_NPU_CPU=1): per-job AIE performance counters of every core tile.
// The fused command (LoomNpuSplit::FusedCommand) carries Setup(image) after each job's gate, so counting starts once
// the job's ready word lands, and Read() after its done; the firmware appends the READ_REGS values to a debug buffer
// attached to the context, which the host reads once the last job is done (Finish).
// Spec "c0,c1,c2,c3/m0,m1": the core module's four counters' events and the memory module's two (AIE2P event ids:
// tools/npu_prof.py, which sums them per family); each counter counts the cycles its event is set. Its cost is in the
// noise (family times within 1% of a plain run).
#ifndef YAH_MODEL_NPU_PROF_HPP_
#define YAH_MODEL_NPU_PROF_HPP_

#include <dirent.h>
#include <drm/amdxdna_accel.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "model/loom_runtime.hpp"

namespace yah::model {

class NpuProf {
 public:
  static constexpr std::uint32_t kCols = 8, kRows = 4, kRow0 = 2, kCore = 4, kMem = 2, kRegs = kCore + kMem;
  static constexpr std::uint32_t kTiles = kCols * kRows, kValues = kTiles * kRegs;

  explicit NpuProf(const char* spec) {
    if (std::sscanf(spec, "%u,%u,%u,%u/%u,%u", &c_[0], &c_[1], &c_[2], &c_[3], &m_[0], &m_[1]) != 6)
      throw LoomError(std::string("YAH_NPU_PROF: want c0,c1,c2,c3/m0,m1 event ids, got ") + spec);
    std::vector<std::uint32_t> addr;   // the counters READ_REGS reads, per tile
    for (std::uint32_t col = 0; col < kCols; ++col)
      for (std::uint32_t row = kRow0; row < kRow0 + kRows; ++row) {
        const std::uint32_t t = col << 25 | row << 20;
        for (std::uint32_t r = 0; r < kCore; ++r) addr.push_back(t + 0x31520 + 4 * r);
        for (std::uint32_t r = 0; r < kMem; ++r) addr.push_back(t + 0x11020 + 4 * r);
      }
    // READ_REGS (transaction custom op 130): {op, col, row, pad, u32 size}, u32 count, u32 pad, u64 addresses
    read_.assign(16 + 8 * addr.size(), 0);
    read_[0] = 130;
    Put(read_.data() + 4, static_cast<std::uint32_t>(read_.size()));
    Put(read_.data() + 8, static_cast<std::uint32_t>(addr.size()));
    for (std::size_t i = 0; i < addr.size(); ++i) Put(read_.data() + 16 + 8 * i, addr[i]);
  }
  ~NpuProf() {
    if (bo_) {
      drm_gem_close close{bo_, 0};
      ioctl(fd_, DRM_IOCTL_GEM_CLOSE, &close);
    }
  }

  // Before the context's first command: a debug buffer attached to the context (DRM handle ctx) on libamdf's device
  // descriptor (the process's accel descriptor that knows the context).
  void Attach(std::uint32_t ctx) {
    std::string errors;
    if (DIR* d = opendir("/proc/self/fd")) {
      for (dirent* e; (e = readdir(d)) && !map_;) {
        char path[64], link[64] = {};
        std::snprintf(path, sizeof(path), "/proc/self/fd/%s", e->d_name);
        if (readlink(path, link, sizeof(link) - 1) > 0 && !std::strncmp(link, "/dev/accel/", 11))
          if (!TryAttach(std::atoi(e->d_name), ctx))
            errors += std::string(" fd ") + e->d_name + ": " + std::strerror(errno);
      }
      closedir(d);
    }
    if (!map_) throw LoomError("npu prof: no debug buffer for context " + std::to_string(ctx) + errors);
  }

  // The counters' setup for a job of this image: stop, zero, the image's PC events, start. YAH_NPU_PROF_PC names a
  // directory of "<image name>.pc" files (emit_prefill_pp with LOOM_DIAG_PC=1), lines "col row pc0 pc1 pc2 pc3": the
  // program addresses of a core's PC events (events 20 / 21: the PC in [pc0, pc1) / [pc2, pc3)); 0 or no file: none.
  const std::vector<std::uint8_t>& Setup(const std::string& image) {
    std::vector<std::uint8_t>& ops = setup_[image];
    if (!ops.empty()) return ops;
    std::uint32_t pc[kCols][kRows][4] = {};
    if (const char* dir = std::getenv("YAH_NPU_PROF_PC")) {
      std::string name = image.substr(image.rfind('/') + 1);
      if (std::FILE* f = std::fopen((dir + ("/" + name.substr(0, name.rfind('.')) + ".pc")).c_str(), "r")) {
        for (unsigned col, row, a[4]; std::fscanf(f, "%u %u %u %u %u %u", &col, &row, &a[0], &a[1], &a[2], &a[3]) == 6;)
          if (col < kCols && row >= kRow0 && row < kRow0 + kRows)
            for (int r = 0; r < 4; ++r) pc[col][row - kRow0][r] = a[r] ? 1u << 31 | a[r] : 0;   // bit 31: valid
        std::fclose(f);
      }
    }
    auto ctl = [](unsigned a, unsigned b) { return (a & 127) | (a & 127) << 8 | (b & 127) << 16 | (b & 127) << 24; };
    for (std::uint32_t col = 0; col < kCols; ++col)
      for (std::uint32_t row = kRow0; row < kRow0 + kRows; ++row) {
        const std::uint32_t t = col << 25 | row << 20;
        Write(ops, t + 0x31500, 0), Write(ops, t + 0x31504, 0), Write(ops, t + 0x11000, 0);   // stop
        for (std::uint32_t r = 0; r < kCore; ++r) Write(ops, t + 0x31520 + 4 * r, 0);
        for (std::uint32_t r = 0; r < kMem; ++r) Write(ops, t + 0x11020 + 4 * r, 0);
        for (std::uint32_t r = 0; r < 4; ++r) Write(ops, t + 0x32020 + 4 * r, pc[col][row - kRow0][r]);
        Write(ops, t + 0x31500, ctl(c_[0], c_[1])), Write(ops, t + 0x31504, ctl(c_[2], c_[3]));
        Write(ops, t + 0x11000, ctl(m_[0], m_[1]));
      }
    return ops;
  }
  static constexpr std::uint32_t SetupOps() { return kTiles * (3 + kCore + kMem + 4 + 3); }
  const std::vector<std::uint8_t>& Read() const { return read_; }

  // After the last job's done: the firmware's log of every job's READ_REGS, kValues words per job in job order
  // (col-major tiles, core counters then memory counters). The sync queues behind the context's last command.
  std::vector<std::uint32_t> Finish(std::size_t jobs) {
    if (jobs * kValues > kBytes / 4) throw LoomError("npu prof: more jobs than the debug buffer holds");
    amdxdna_drm_sync_bo sb{};
    sb.handle = bo_, sb.direction = SYNC_DIRECT_FROM_DEVICE, sb.size = kBytes;
    if (ioctl(fd_, DRM_IOCTL_AMDXDNA_SYNC_BO, &sb)) throw LoomError("npu prof: SYNC_BO failed");
    Flush();
    if (map_[jobs * kValues] != 0) std::fprintf(stderr, "npu prof: the log runs past the last job\n");
    return std::vector<std::uint32_t>(map_, map_ + jobs * kValues);
  }

 private:
  static constexpr std::uint32_t kBytes = 1 << 20;
  static void Put(std::uint8_t* p, std::uint32_t v) { std::memcpy(p, &v, 4); }
  bool TryAttach(int fd, std::uint32_t ctx) {
    amdxdna_drm_create_bo cb{};
    cb.size = kBytes, cb.type = AMDXDNA_BO_DEV;
    if (ioctl(fd, DRM_IOCTL_AMDXDNA_CREATE_BO, &cb)) return false;
    amdxdna_drm_get_bo_info bi{};
    bi.handle = cb.handle;
    amdxdna_drm_config_hwctx ch{};
    ch.handle = ctx, ch.param_type = DRM_AMDXDNA_HWCTX_ASSIGN_DBG_BUF, ch.param_val = cb.handle;
    if (ioctl(fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &bi) || !bi.vaddr || ioctl(fd, DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &ch)) {
      const int e = errno;
      drm_gem_close close{cb.handle, 0};
      ioctl(fd, DRM_IOCTL_GEM_CLOSE, &close);
      errno = e;
      return false;
    }
    fd_ = fd, bo_ = cb.handle;
    map_ = reinterpret_cast<volatile std::uint32_t*>(bi.vaddr);
    return true;
  }
  // the NPU writes the buffer behind the CPU's caches: drop the stale lines
  void Flush() {
    for (std::uint32_t i = 0; i < kBytes / 4; i += 16) __builtin_ia32_clflush(const_cast<std::uint32_t*>(map_ + i));
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
  }
  // WRITE32: {op 0, pad[7]}, u64 address (low word), u32 value, u32 size 24
  static void Write(std::vector<std::uint8_t>& ops, std::uint32_t address, std::uint32_t value) {
    std::uint8_t op[24] = {};
    Put(op + 8, address), Put(op + 16, value), Put(op + 20, std::uint32_t{24});
    ops.insert(ops.end(), op, op + 24);
  }

  unsigned c_[kCore] = {}, m_[kMem] = {};
  std::map<std::string, std::vector<std::uint8_t>> setup_;   // per image
  std::vector<std::uint8_t> read_;
  std::uint32_t bo_ = 0;
  int fd_ = -1;
  volatile std::uint32_t* map_ = nullptr;
};

}  // namespace yah::model

#endif  // YAH_MODEL_NPU_PROF_HPP_
