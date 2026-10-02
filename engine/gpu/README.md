# gpu/

- `loom/`: the Loom kernels and their Python generators (`loom/tools/`). See [docs/architecture.md](../../docs/architecture.md).
- `loomhip.hip`: runs one Loom hsaco through HIP so rocprofv3 (PMC, occupancy, ATT) can profile it; HRX dispatches are invisible to rocprofv3. Build with `engine/build_loomhip.sh`.
