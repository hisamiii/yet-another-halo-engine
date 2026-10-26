// The GPU's DPM level pinned to high while the NPU runs (fabric DPM switches corrupt the NPU's outputs), and the
// previous level put back when the process ends however it ends: a watcher process blocks on a pipe whose write end
// only this process holds, so it restores the level on exit, an error, a signal, a crash or SIGKILL alike.
// The level is written directly when the file is writable, else through doas -n or sudo -n tee (no prompt).
#ifndef YAH_MODEL_DPM_PIN_HPP_
#define YAH_MODEL_DPM_PIN_HPP_

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

extern char** environ;

namespace yah::model {

class DpmPin {
 public:
  DpmPin() = default;
  DpmPin(const DpmPin&) = delete;
  DpmPin& operator=(const DpmPin&) = delete;
  ~DpmPin() {
    if (hold_ < 0) return;
    ::close(hold_);  // the watcher restores the level now
    ::waitpid(watcher_, nullptr, 0);
  }

  // "" once the level is high (pinned now or already), else why it is not.
  std::string Pin() {
    for (int card = 0; card < 8 && path_.empty(); ++card) {
      const std::string p = "/sys/class/drm/card" + std::to_string(card) + "/device/power_dpm_force_performance_level";
      if (!Level(p).empty()) path_ = p;
    }
    if (path_.empty()) return "no amdgpu DPM control found";
    const std::string was = Level(path_);
    if (was == "high") return "";
    int fds[2];
    if (::pipe2(fds, O_CLOEXEC) != 0) return "pipe2 failed";
    // sh: wait for the write end to close, then the level back by the same routes Write takes
    const std::string script =
        "while read -r _; do :; done; { printf '%s\\n' \"$1\" > \"$2\"; } 2> /dev/null || "
        "printf '%s\\n' \"$1\" | doas -n tee \"$2\" > /dev/null 2>&1 || "
        "printf '%s\\n' \"$1\" | sudo -n tee \"$2\" > /dev/null 2>&1 || "
        "echo \"yah: cannot put the GPU's DPM level back to $1 ($2)\" >&2";
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[0], 0);
    posix_spawnattr_t at;
    posix_spawnattr_init(&at);
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP);  // its own group: a terminal's Ctrl-C does not reach it
    posix_spawnattr_setpgroup(&at, 0);
    std::vector<std::string> args = {"/bin/sh", "-c", script, "sh", was, path_};
    const int r = ::posix_spawn(&watcher_, "/bin/sh", &fa, &at, Argv(args).data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    posix_spawnattr_destroy(&at);
    ::close(fds[0]);
    if (r != 0) {
      ::close(fds[1]);
      return "cannot start the DPM watcher";
    }
    hold_ = fds[1];
    if (Write("high") && Level(path_) == "high") return "";
    ::close(std::exchange(hold_, -1));  // nothing to restore
    ::waitpid(watcher_, nullptr, 0);
    return "the GPU's DPM level is " + was + " and yah cannot pin it to high (" + path_ +
           ": not writable, and doas -n / sudo -n tee failed). Allow one, e.g. the doas.conf line 'permit nopass " +
           User() + " cmd tee args " + path_ + "', or set it yourself: echo high | sudo tee " + path_;
  }

 private:
  static std::string Level(const std::string& path) {
    std::ifstream f(path);
    std::string level;
    f >> level;
    return level;
  }
  static std::string User() {
    const char* u = std::getenv("USER");
    return u && *u ? u : "<user>";
  }
  static std::vector<char*> Argv(std::vector<std::string>& args) {
    std::vector<char*> v;
    for (auto& a : args) v.push_back(a.data());
    v.push_back(nullptr);
    return v;
  }
  // level into path_: directly, else <tool> -n tee path_ with the level on its stdin (in the pipe before the tool
  // starts: no write to a pipe whose reader may be gone)
  bool Write(const std::string& level) const {
    if (std::ofstream f(path_); f && (f << level << "\n") && f.flush()) return true;
    const std::string line = level + "\n";
    for (const char* tool : {"doas", "sudo"}) {
      int fds[2];
      if (::pipe2(fds, O_CLOEXEC) != 0) return false;
      const bool wrote = ::write(fds[1], line.data(), line.size()) == static_cast<ssize_t>(line.size());
      ::close(fds[1]);
      posix_spawn_file_actions_t fa;
      posix_spawn_file_actions_init(&fa);
      posix_spawn_file_actions_adddup2(&fa, fds[0], 0);
      posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
      posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
      std::vector<std::string> args = {tool, "-n", "tee", path_};
      pid_t pid = 0;
      const int r = ::posix_spawnp(&pid, tool, &fa, nullptr, Argv(args).data(), environ);
      posix_spawn_file_actions_destroy(&fa);
      ::close(fds[0]);
      int status = 0;
      if (r == 0 && ::waitpid(pid, &status, 0) == pid && wrote && WIFEXITED(status) && WEXITSTATUS(status) == 0)
        return true;
    }
    return false;
  }

  std::string path_;
  pid_t watcher_ = -1;
  int hold_ = -1;  // the pipe's write end: closing it (or the process ending) restores the level
};

}  // namespace yah::model

#endif  // YAH_MODEL_DPM_PIN_HPP_
