# rocprof studio (pwilkin/ROCprofGUI): headless use on gfx1151

Rust + Tauri capture analyser for rocprofv3 / ATT / RADV traces, with an MCP
server (`rocprof-studio --headless --mcp-port N --mcp-token-file F`).
Evaluated 2026-10-01 at commit 5555964.

**What works here**
- The headless core alone builds:
  `cargo build --release -p profile-core` (Rust 1.98, ~1 min). No Node / GTK
  / WebKit is needed for that.
- It reads our rocprofv3 `--att` output directories directly (decoded UI
  JSON: code.json, filenames.json, wstates*.json, se*_wv*.json). gfx1151 is
  recognised (arch "navi").
- The raw-ATT decoder packaging (rocprof-trace-decoder 0.2.2 enforced) is not
  needed for decoded captures; TheRock ships 0.1.7.
- Commands:
  - `profile-core <capture_dir>`: capture model JSON (waves, instructions,
    dispatches, occupancy, warnings).
  - `profile-core <capture_dir> <wave_file>`: one wave's tokens.
  - `profile-core <capture_dir> --hidden`: per-instruction and per-token
    stall/idle split into hidden vs exposed, using ROCm Compute Viewer's
    pipe-priority model (MATRIX > VALU > VMEM/LDS > SMEM/SALU).
    **The useful part for us.**

**Limits we hit**
- Import cap: 2 M records (`import.rs` check_rows).
- Hidden-analysis cap: 4 M tokens (`hidden.rs:265`).
- A full attention trace exceeds both even with `--att-simd-select 0`. A
  local build multiplies both caps (diagnosis only, not upstreamed). The
  hidden JSON for 90 waves is ~830 MB.

**First finding** (int8 GQA attention, pp8192, 1 SIMD, truncated trace):

| instruction | exposed % of wave time |
|---|---:|
| `ds_load_b128` | 20.4% (only 26% hidden) |
| `v_wmma_i32_16x16x16_iu8` | 8.3% |
| `s_waitcnt lgkmcnt(1)` | 6.7% |
| `s_waitcnt vmcnt(2)` | 4.0% |
| `v_cvt_f32_i32` (score scaling) | 3.1% |

The `ds_load_b128` time is issue stalls: the LDS queue is saturated (LDS
74-86% busy by counters). Our simdtl.py charged only the following
s_waitcnt. Next lever: LDS bytes per WMMA (kv4 / iu4 halve fragment bytes
again).

Script: engine/run/research/fa/ (att.sh / attq.sh capture,
`--att-simd-select 0 --att-buffer-size 25165824` for a small window).

## MCP server setup on this machine (2026-10-01)

- **Build:** full app from /home/q/yah-scratch/ext/ROCprofGUI with
  `npm ci && npx tauri build --no-bundle`. System webkit2gtk-4.1 / gtk3 /
  libsoup3 are present, and the repo ships decoder-python.zip.
  - The build carries the local x50 caps (import.rs, hidden.rs) so full
    attention traces load.
- **Install:** `~/.local/bin/rocprof-studio`, run by the systemd user service
  `~/.config/systemd/user/rocprof-studio.service`:
  - `--headless --mcp-port 8765`
  - ROCPROFGUI_HOME=~/.rocprofgui
  - enabled, Restart=on-failure
- **Endpoint:** http://127.0.0.1:8765/mcp (Streamable HTTP, loopback only).
  Bearer token in ~/.rocprofgui/mcp/token (0600); requests without it get
  401.
- **Claude Code:** registered at user scope as `rocprof-studio` (31 tools:
  capture_open / capture_analyze / wave_analyze / hidden_latency /
  profile_run / ...).
- **Our kernels:** pass `profiler` =
  /var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0/bin/rocprofv3
  to profile_run (the default is /opt/rocm/bin/rocprofv3). Import ATT that
  rocprofv3 already decoded. Raw-ATT decoding wants rocprof-trace-decoder
  0.2.2; TheRock ships 0.1.7.
- **Manage:** `systemctl --user {status,restart,stop} rocprof-studio`.
