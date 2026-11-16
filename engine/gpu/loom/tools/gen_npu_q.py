#!/usr/bin/env python3
"""gen_npu_q.py: layout Q' GEMM images (experimental, stage 1 of the Q' build). Every worker is an independent core
that owns a TM x TN output tile over the whole K: f32 sums in its own memory, no cascade, no roles. The leaf is
gen_npu_gemm's tail in GN_QCORE mode (plain MMAs from C, in-row sub-tile boundaries in a solved order).
Stage 1 array: one row of cols workers (array row 2). A: one stream multicast along the row (records [M block][pass]
[slab]); W: per column, its panel [pass][record] staged in the column's memory tile and replayed per M block; C: per
column, segments [M block][slab] as bf16 (the tail's last-pass convert). Same bindings as gen_npu_gemm with one K
slice, so npu_gemm_check.py checks it (NPU_Q=1, ks of one entry, NP 4)."""
import gen_npu_gemm as N


def array_program(L, cfg):
    e = L.append
    cols, nb, P, ks = cfg.cols, cfg.nb, cfg.passes, cfg.ks[0]
    MP, NP = N.MP, N.NP
    e("aie2p.target<array> @array_target\naie2p.target<core> @core_target\n")
    e(f"low.func.def public retain target<amd.xdna.aie2p.array>(@array_target) abi(array_program) @{cfg.entry}() asm {{")
    for k, v in (("zero", 0), ("one", 1), ("two", 2), ("nw", cols), ("rec", nb * P)):
        e(f"  %{k} = constant.u32 {v} : reg<aie2p.array.scalar : index>")
    e("  %origin = constant.u64 0 : reg<aie2p.array.offset : offset>")
    e("  %workers = group %nw")
    e('  %ab = binding 0, "read"')
    e('  %wb = binding 1, "read"')
    e('  %cb = binding 2, "write"')
    for i in range(max(cols, 2, MP) + 6):
        e(f"  %n{i} = constant.u32 {i} : reg<aie2p.array.scalar : index>")
    for c in range(cols):
        e(f"  %lane{c} = constant.u32 {c} : reg<aie2p.array.scalar : index>")
        e(f"  %k{c} = worker %workers, %lane{c}, @tail")
        e(f"  constrain.location %k{c}, %n{c}, %n2")
    # C first (rings are placed in channel order: C claims whole banks before the input rings)
    rec = MP * NP * 4 * 256 // MP // 8
    e(f"  %ncols = constant.u32 {cols} : reg<aie2p.array.scalar : index>")
    e(f"  %nseg = constant.u32 {nb * MP} : reg<aie2p.array.scalar : index>")
    e(f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols}x{nb * MP}x{rec}xi32>>")
    for c in range(cols):
        e(f"  %cr{c} = partition.receiver %c_all, %origin, %n{c}, %ncols : reg<aie2p.array.receiver : tile<{rec}xi32>>")
        e(f"  %sc{c} = sender %k{c}, 2 : reg<aie2p.array.sender : tile<{rec}xi32>>")
        e(f"  %chc{c} = channel %sc{c}, %cr{c}, %n{MP}, %nseg : reg<aie2p.array.channel : tile<{rec}xi32>>")
        e(f"  constrain.leaf_sync %chc{c}")
    wrec = NP * N.slab(ks)
    for c in range(cols):
        e(f"  %wo{c} = constant.u64 {c * P * wrec} : reg<aie2p.array.offset : offset>")
        e(f"  %ws{c} = sender %wb, 0 : reg<aie2p.array.sender : tile<{nb}x{P}x{wrec // 4}xi32, "
          f"#encoding.layout.strided<strides=[0, {wrec // 4}, 1]>>>")
        e(f"  %wv{c} = view.sender %ws{c}, %wo{c} : reg<aie2p.array.sender : tile<{wrec // 4}xi32>>")
        e(f"  %rw{c} = receiver %k{c}, 1 : reg<aie2p.array.receiver : tile<{wrec // 4}xi32>>")
        e(f"  %chw{c} = channel %wv{c}, %rw{c}, %two, %rec : reg<aie2p.array.channel : tile<{wrec // 4}xi32>>")
        e(f"  constrain.stage %chw{c}, %n{c if cols > 1 else 1}")
        if N.qpass_on():   # (GN_QPASS) the leaf walks the W ring: one firing per M block
            e(f"  constrain.leaf_sync %chw{c}")
    fa = N.slab(ks)
    e(f"  %as0 = sender %ab, 0 : reg<aie2p.array.sender : tile<{nb * P * MP}x{fa // 4}xi32>>")
    e(f"  %a0 = view.sender %as0, %origin : reg<aie2p.array.sender : tile<{fa // 4}xi32>>")
    e(f"  %arec = constant.u32 {nb * P * MP} : reg<aie2p.array.scalar : index>")
    for c in range(cols):
        e(f"  %ra{c} = receiver %k{c}, 0 : reg<aie2p.array.receiver : tile<{fa // 4}xi32>>")
        e(f"  %cha{c} = channel %a0, %ra{c}, %n{N.acap_tail(cfg)}, %arec : reg<aie2p.array.channel : tile<{fa // 4}xi32>>")
        e(f"  constrain.leaf_sync %cha{c}")
        e(f"  constrain.stage %cha{c}, %n{N.astage(0, cols) if cols > 1 else 2}")
    e("  return\n}\n")


def gen(cfg):
    assert len(cfg.ks) == 1 and cfg.fuse == 0 and not cfg.gate and cfg.passes >= 2 and 1 <= cfg.cols <= 8
    L = []
    array_program(L, cfg)
    N.leaf(L, cfg, "tail", cfg.ks[0])
    return "\n".join(L)
