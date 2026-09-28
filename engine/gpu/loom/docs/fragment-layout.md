# gfx11 f16 matrix-fragment lane layout (probe)

`probe_fragment_layout.loom` loads a 16x16 f16 logical tile (element
`(r,c) = r*16+c`) with `vector.fragment.load<lhs>` and stores the physical
per-lane carrier to a `32x16` output, so `out[L][e]` identifies the logical
element lane L holds at lane-local element e.

Result: lane L (0..15) holds logical row L, all 16 columns in order; lanes
16..31 hold a duplicate of lanes 0..15.

| lane | carrier elements e = 0..15 |
| --- | --- |
| L (0..15) | (L,0) (L,1) ... (L,15) |
| L+16 | same as lane L |

So the 16x16 LHS fragment is one 16-f16 row per participating lane, with the
upper half-wave duplicating the lower half. The carrier `vector<16xf16>` is 16
f16 = 8 dwords per lane. `probe_fragment_rhs.loom` and
`probe_fragment_result.loom` are the same probe for the rhs and result roles.

LDS: `probe_fragment_lds.loom` stages the 256 f16 into a
`buffer.alloca<workgroup>` view and repeats the fragment load. The per-lane
carrier is byte-identical to the global-buffer case. The port note that "a
fragment load or store against a `buffer.alloca<workgroup>` view compiles but
does not produce the tile the target reads" is therefore wrong: LDS staging is
correct. Measured on the `iq4xs` kStore at production shape (m_tiles=1088,
k_blocks=20), the LDS form is 2.37 ms vs 2.62 ms for the dense global tile
(~10%), both passing the in-tree case.

Consequence: a 16x16 weight tile can be decoded straight into a workgroup LDS
tile and fragment-loaded from it. A format decoder can also build the carrier
per lane with no cross-lane dependency at all, because lane L only needs row L.

## Reproduce

    bash engine/gpu/loom/loom_run.sh engine/gpu/loom/probe_fragment_layout.loom @probe_fragment_lhs_global_case -
    bash engine/gpu/loom/loom_run.sh engine/gpu/loom/probe_fragment_lds.loom @probe_fragment_lhs_lds_case -

The case writes `probe_fragment_lhs_global.npy` / `probe_fragment_lhs_lds.npy`
under `/tmp/iree-loom-benchmark/<source>.loom_<hash>/`.