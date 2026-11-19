// YAH_NPU_PROF (CPU-driven runs, YAH_NPU_CPU=1, or GPU + NPU runs: LoomPrefill::ReportNpuProf): per-job AIE
// performance counters of every core tile.
// The fused command (LoomNpuSplit::FusedCommand) carries Setup(image) after each job's gate, so counting starts once
// the job's ready word lands, and Read() after its done; the firmware appends the READ_REGS values to a debug buffer
// attached to the context, which the host reads once the last job is done (Finish).
// Spec "c0,c1,c2,c3/m0,m1": the core module's four counters' events and the memory module's two (AIE2P event ids:
// tools/npu_prof.py, which sums them per family); each counter counts the cycles its event is set. Its cost is in the
// noise (family times within 1% of a plain run).
// Optional, after a job's tile values: YAH_NPU_PROF_MT="e0,e1,e2,e3" the memory tiles' (row 1) four counters,
// YAH_NPU_PROF_MT_SEL="s0,s1,m0,m1" the DMA channels their SEL0 / SEL1 events watch (S2MM, MM2S), and
// YAH_NPU_PROF_SHIM="e0,e1" the shim tiles' (row 0) two (e.g. 36 / 32: MM2S 0 starved of memory / held back by its
// stream). Unset: those counters count nothing.
// The read also takes core tile (0, 2)'s 64-bit timer (array clock cycles, the last two values): with event 1 (TRUE) on
// core counter 0, the job's gate opened at timer - that tile's counter 0 (Finish: times()).
// YAH_NPU_PROF_ROT="s0;s1;...": several counter settings in one run, each "c0,c1,c2,c3/m0,m1[@combo][#pc dir]"
// (combo as YAH_NPU_PROF_COMBO, the PC files' directory as YAH_NPU_PROF_PC; a part left out: those variables'). An
// image's n-th job counts with setting n % count (chosen(): each Setup's setting, in job order), so the jobs of one
// image share the settings out: e.g. 12 settings over 25 like layers, 2 layers each.
#ifndef YAH_MODEL_NPU_PROF_HPP_
#define YAH_MODEL_NPU_PROF_HPP_

#include <dirent.h>
#include <drm/amdxdna_accel.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
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
  static constexpr std::uint32_t kMt = 4, kShim = 2;   // memory tile and shim counters, after the tiles'
  static constexpr std::uint32_t kTiles = kCols * kRows, kValues = kTiles * kRegs + kCols * (kMt + kShim) + 2;
  struct Times {
    std::uint64_t gate, done;   // (gate: valid with TRUE on core counter 0)
  };

  // cols: the context's columns (the engine's 8; a narrower context's run_once tools fewer), counted from column 0
  explicit NpuProf(const char* spec, std::uint32_t cols = kCols) : cols_(cols) {
    if (cols_ < 1 || cols_ > kCols) throw LoomError("npu prof: bad column count");
    if (std::sscanf(spec, "%u,%u,%u,%u/%u,%u", &c_[0], &c_[1], &c_[2], &c_[3], &m_[0], &m_[1]) != 6)
      throw LoomError(std::string("YAH_NPU_PROF: want c0,c1,c2,c3/m0,m1 event ids, got ") + spec);
    std::vector<std::uint32_t> addr;   // the counters READ_REGS reads, per tile
    for (std::uint32_t col = 0; col < cols_; ++col)
      for (std::uint32_t row = kRow0; row < kRow0 + kRows; ++row) {
        const std::uint32_t t = col << 25 | row << 20;
        for (std::uint32_t r = 0; r < kCore; ++r) addr.push_back(t + 0x31520 + 4 * r);
        for (std::uint32_t r = 0; r < kMem; ++r) addr.push_back(t + 0x11020 + 4 * r);
      }
    for (std::uint32_t col = 0; col < cols_; ++col) {
      for (std::uint32_t r = 0; r < kMt; ++r) addr.push_back((col << 25 | 1u << 20) + 0x91020 + 4 * r);
      for (std::uint32_t r = 0; r < kShim; ++r) addr.push_back((col << 25) + 0x31020 + 4 * r);
    }
    // (diag) YAH_NPU_PROF_LOCKS=1: column 0's lock values and DMA channel status instead of the counters: tiles (0, 2..5)
    // locks 0-15 and S2MM 0/1, MM2S 0/1 status, memory tile (0, 1) locks 0-15 and S2MM 0-5, MM2S 0-5 status (110
    // words with the timer)
    if (std::getenv("YAH_NPU_PROF_LOCKS")) {
      addr.clear();
      for (std::uint32_t row = kRow0; row < kRow0 + kRows; ++row)
        for (std::uint32_t i = 0; i < 16; ++i) addr.push_back((row << 20) + 0x1F000 + 0x10 * i);
      for (std::uint32_t row = kRow0; row < kRow0 + kRows; ++row)
        for (std::uint32_t o : {0x1DF00u, 0x1DF04u, 0x1DF10u, 0x1DF14u}) addr.push_back((row << 20) + o);
      for (std::uint32_t i = 0; i < 16; ++i) addr.push_back((1u << 20) + 0xC0000 + 0x10 * i);
      for (std::uint32_t i = 0; i < 6; ++i) addr.push_back((1u << 20) + 0xA0660 + 4 * i);
      for (std::uint32_t i = 0; i < 6; ++i) addr.push_back((1u << 20) + 0xA0680 + 4 * i);
    }
    addr.push_back((kRow0 << 20) + 0x340F8), addr.push_back((kRow0 << 20) + 0x340FC);   // tile (0, 2) Timer_Low / High
    nvals_ = static_cast<std::uint32_t>(addr.size());
    if (const char* e = std::getenv("YAH_NPU_PROF_MT"))
      std::sscanf(e, "%u,%u,%u,%u", &mt_[0], &mt_[1], &mt_[2], &mt_[3]);
    if (const char* e = std::getenv("YAH_NPU_PROF_MT_SEL")) {
      unsigned a = 0, b = 0, c = 0, d = 0;
      std::sscanf(e, "%u,%u,%u,%u", &a, &b, &c, &d);
      mt_sel_ = (a & 7) | (b & 7) << 8 | (c & 7) << 16 | (d & 7) << 24;
    }
    if (const char* e = std::getenv("YAH_NPU_PROF_SHIM")) std::sscanf(e, "%u,%u", &sh_[0], &sh_[1]);
    // YAH_NPU_PROF_COMBO="a,b,c,d,o0,o1,o2": the core modules' combo events (9 = a o0 b, 10 = c o1 d, 11 = 9 o2 10;
    // op 0 AND, 1 AND NOT, 2 OR, 3 OR NOT), e.g. 26,20,26,21,0,0,0: lock stalls in PC ranges 0-1 / 2-3
    if (const char* e = std::getenv("YAH_NPU_PROF_COMBO")) {
      unsigned v[7] = {};
      if (std::sscanf(e, "%u,%u,%u,%u,%u,%u,%u", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6]) != 7)
        throw LoomError("YAH_NPU_PROF_COMBO: want a,b,c,d,o0,o1,o2");
      combo_[0] = (v[0] & 127) | (v[1] & 127) << 8 | (v[2] & 127) << 16 | (v[3] & 127) << 24;
      combo_[1] = (v[4] & 3) | (v[5] & 3) << 8 | (v[6] & 3) << 16;
    }
    Setting base;
    std::copy(c_, c_ + kCore, base.c), std::copy(m_, m_ + kMem, base.m);
    base.combo[0] = combo_[0], base.combo[1] = combo_[1];
    if (const char* dir = std::getenv("YAH_NPU_PROF_PC")) base.pc = dir;
    rot_.push_back(base);
    if (const char* e = std::getenv("YAH_NPU_PROF_ROT")) {
      rot_.clear();
      for (std::string s(e); !s.empty();) {
        const std::size_t semi = s.find(';');
        std::string item = s.substr(0, semi);
        s = semi == std::string::npos ? "" : s.substr(semi + 1);
        if (item.empty()) continue;
        Setting t = base;
        if (const std::size_t h = item.find('#'); h != std::string::npos) t.pc = item.substr(h + 1), item.resize(h);
        if (const std::size_t a = item.find('@'); a != std::string::npos) {
          unsigned v[7] = {};
          if (std::sscanf(item.c_str() + a + 1, "%u,%u,%u,%u,%u,%u,%u", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6]) != 7)
            throw LoomError("YAH_NPU_PROF_ROT: a combo wants a,b,c,d,o0,o1,o2: " + item);
          t.combo[0] = (v[0] & 127) | (v[1] & 127) << 8 | (v[2] & 127) << 16 | (v[3] & 127) << 24;
          t.combo[1] = (v[4] & 3) | (v[5] & 3) << 8 | (v[6] & 3) << 16;
          item.resize(a);
        }
        if (std::sscanf(item.c_str(), "%u,%u,%u,%u/%u,%u", &t.c[0], &t.c[1], &t.c[2], &t.c[3], &t.m[0], &t.m[1]) != 6)
          throw LoomError("YAH_NPU_PROF_ROT: want c0,c1,c2,c3/m0,m1 event ids: " + item);
        rot_.push_back(t);
      }
      if (rot_.empty()) throw LoomError("YAH_NPU_PROF_ROT: no settings");
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
    const std::uint32_t k = occ_[image]++ % static_cast<std::uint32_t>(rot_.size());
    chosen_.push_back(k);
    const Setting& set = rot_[k];
    std::vector<std::uint8_t>& ops = setup_[image + "\n" + std::to_string(k)];
    if (!ops.empty()) return ops;
    std::uint32_t pc[kCols][kRows][4] = {};
    if (!set.pc.empty()) {
      const std::string& dir = set.pc;
      std::string name = image.substr(image.rfind('/') + 1);
      if (std::FILE* f = std::fopen((dir + ("/" + name.substr(0, name.rfind('.')) + ".pc")).c_str(), "r")) {
        for (unsigned col, row, a[4]; std::fscanf(f, "%u %u %u %u %u %u", &col, &row, &a[0], &a[1], &a[2], &a[3]) == 6;)
          if (col < kCols && row >= kRow0 && row < kRow0 + kRows)
            for (int r = 0; r < 4; ++r) pc[col][row - kRow0][r] = a[r] ? 1u << 31 | a[r] : 0;   // bit 31: valid
        std::fclose(f);
      }
    }
    auto ctl = [](unsigned a, unsigned b) { return (a & 127) | (a & 127) << 8 | (b & 127) << 16 | (b & 127) << 24; };
    for (std::uint32_t col = 0; col < cols_; ++col)
      for (std::uint32_t row = kRow0; row < kRow0 + kRows; ++row) {
        const std::uint32_t t = col << 25 | row << 20;
        Write(ops, t + 0x31500, 0), Write(ops, t + 0x31504, 0), Write(ops, t + 0x11000, 0);   // stop
        for (std::uint32_t r = 0; r < kCore; ++r) Write(ops, t + 0x31520 + 4 * r, 0);
        for (std::uint32_t r = 0; r < kMem; ++r) Write(ops, t + 0x11020 + 4 * r, 0);
        for (std::uint32_t r = 0; r < 4; ++r) Write(ops, t + 0x32020 + 4 * r, pc[col][row - kRow0][r]);
        Write(ops, t + 0x34400, set.combo[0]), Write(ops, t + 0x34404, set.combo[1]);   // (YAH_NPU_PROF_COMBO)
        Write(ops, t + 0x31500, ctl(set.c[0], set.c[1])), Write(ops, t + 0x31504, ctl(set.c[2], set.c[3]));
        Write(ops, t + 0x11000, ctl(set.m[0], set.m[1]));
      }
    for (std::uint32_t col = 0; col < cols_; ++col) {
      const std::uint32_t m = col << 25 | 1u << 20, sh = col << 25;
      Write(ops, m + 0x91000, 0), Write(ops, m + 0x91004, 0), Write(ops, sh + 0x31000, 0);   // stop
      for (std::uint32_t r = 0; r < kMt; ++r) Write(ops, m + 0x91020 + 4 * r, 0);
      for (std::uint32_t r = 0; r < kShim; ++r) Write(ops, sh + 0x31020 + 4 * r, 0);
      Write(ops, m + 0xA06A0, mt_sel_);
      Write(ops, m + 0x91000, ctl(mt_[0], mt_[1])), Write(ops, m + 0x91004, ctl(mt_[2], mt_[3]));
      Write(ops, sh + 0x31000, ctl(sh_[0], sh_[1]));
    }
    return ops;
  }
  std::uint32_t SetupOps() const {
    return cols_ * kRows * (3 + kCore + kMem + 4 + 2 + 3) + cols_ * (3 + kMt + kShim + 1 + 3);
  }
  // words per job (kValues with the engine's 8 columns)
  std::uint32_t Values() const { return nvals_; }
  const std::vector<std::uint8_t>& Read() const { return read_; }
  // Finish's jobs' timer stamps
  const std::vector<Times>& times() const { return times_; }
  // each Setup's setting (YAH_NPU_PROF_ROT; 0 without), in call order = job order
  const std::vector<std::uint32_t>& chosen() const { return chosen_; }

  // After the last job's done: the firmware's log of every job's READ_REGS, Values() words per job in job order
  // (col-major tiles, core counters then memory counters, then the memory tiles', the shims' and the timer; times()).
  // The sync queues behind the context's last command.
  std::vector<std::uint32_t> Finish(std::size_t jobs) {
    if (jobs * Values() > kBytes / 4) throw LoomError("npu prof: more jobs than the debug buffer holds");
    amdxdna_drm_sync_bo sb{};
    sb.handle = bo_, sb.direction = SYNC_DIRECT_FROM_DEVICE, sb.size = kBytes;
    if (ioctl(fd_, DRM_IOCTL_AMDXDNA_SYNC_BO, &sb)) throw LoomError("npu prof: SYNC_BO failed");
    Flush();
    if (map_[jobs * Values()] != 0) std::fprintf(stderr, "npu prof: the log runs past the last job\n");
    std::vector<std::uint32_t> v(map_, map_ + jobs * Values());
    times_.clear();
    for (std::size_t j = 0; j < jobs; ++j) {
      const std::uint32_t* const x = v.data() + j * Values();
      const std::uint64_t done = std::uint64_t{x[Values() - 1]} << 32 | x[Values() - 2];
      times_.push_back({done - x[0], done});
    }
    return v;
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

  struct Setting {   // one counter setting (YAH_NPU_PROF_ROT)
    unsigned c[kCore] = {}, m[kMem] = {};
    std::uint32_t combo[2] = {};
    std::string pc;   // the PC files' directory, empty: none
  };
  std::vector<Setting> rot_;                       // the settings (one without YAH_NPU_PROF_ROT)
  std::map<std::string, std::uint32_t> occ_;       // per image, its jobs set up so far
  std::vector<std::uint32_t> chosen_;              // each Setup's setting
  std::uint32_t combo_[2] = {};
  std::uint32_t cols_ = kCols;
  std::uint32_t nvals_ = 0;
  unsigned c_[kCore] = {}, m_[kMem] = {}, mt_[kMt] = {}, sh_[kShim] = {};
  std::uint32_t mt_sel_ = 0;
  std::map<std::string, std::vector<std::uint8_t>> setup_;   // per image
  std::vector<std::uint8_t> read_;
  std::vector<Times> times_;
  std::uint32_t bo_ = 0;
  int fd_ = -1;
  volatile std::uint32_t* map_ = nullptr;
};

}  // namespace yah::model

#endif  // YAH_MODEL_NPU_PROF_HPP_
