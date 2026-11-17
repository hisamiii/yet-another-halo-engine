#!/usr/bin/env python3
"""gen_npu_q.py: layout Q' GEMM images (experimental, stage 1 of the Q' build). Every worker is an independent core
that owns a TM x TN output tile over the whole K: f32 sums in its own memory, no cascade, no roles. The leaf is
gen_npu_gemm's tail in GN_QCORE mode (plain MMAs from C, in-row sub-tile boundaries in a solved order).
Stage 1 array: one row of cols workers (array row 2). A: one stream multicast along the row (records [M block][pass]
[slab]); W: per column, its panel [pass][record] staged in the column's memory tile and replayed per M block; C: per
column, segments [M block][slab] as bf16 (the tail's last-pass convert). Same bindings as gen_npu_gemm with one K
slice, so npu_gemm_check.py checks it (NPU_Q=1, ks of one entry, NP 4)."""
import os

import gen_npu_gemm as N

# NPU_QROWS=R (1..4): R array rows (2 .. 1 + R); row r takes M blocks [r nb / R, (r + 1) nb / R) (its own A stream,
# multicast along the row); a column's W panel is multicast to its R cores; C [col][M block][slab] as before.
QROWS = int(os.environ.get("NPU_QROWS", "1"))


def array_program(L, cfg):
    e = L.append
    cols, nb, P, ks = cfg.cols, cfg.nb, cfg.passes, cfg.ks[0]
    MP, NP, R = N.MP, N.NP, QROWS
    assert 1 <= R <= 4 and nb % R == 0
    nbr = nb // R
    e("aie2p.target<array> @array_target\naie2p.target<core> @core_target\n")
    e(f"low.func.def public retain target<amd.xdna.aie2p.array>(@array_target) abi(array_program) @{cfg.entry}() asm {{")
    for k, v in (("zero", 0), ("one", 1), ("two", 2), ("nw", cols * R), ("rec", nbr * P)):
        e(f"  %{k} = constant.u32 {v} : reg<aie2p.array.scalar : index>")
    e("  %origin = constant.u64 0 : reg<aie2p.array.offset : offset>")
    e("  %workers = group %nw")
    e('  %ab = binding 0, "read"')
    e('  %wb = binding 1, "read"')
    e('  %cb = binding 2, "write"')
    for i in range(max(cols, 2, MP) + 6):
        e(f"  %n{i} = constant.u32 {i} : reg<aie2p.array.scalar : index>")
    kk = {}
    for c in range(cols):
        for r in range(R):
            kk[c, r] = f"%k{c}_{r}"
            e(f"  %lane{c}_{r} = constant.u32 {c * R + r} : reg<aie2p.array.scalar : index>")
            e(f"  {kk[c, r]} = worker %workers, %lane{c}_{r}, @tail")
            e(f"  constrain.location {kk[c, r]}, %n{c}, %n{2 + r}")
    # C first (rings are placed in channel order: C claims whole banks before the input rings)
    rec = MP * NP * 4 * 256 // MP // 8
    e(f"  %ncols = constant.u32 {cols * R} : reg<aie2p.array.scalar : index>")
    e(f"  %nseg = constant.u32 {nbr * MP} : reg<aie2p.array.scalar : index>")
    # [col][row][its M blocks] = [col][M block]: lane c R + r
    e(f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols * R}x{nbr * MP}x{rec}xi32>>")
    for c in range(cols):
        for r in range(R):
            t = f"{c}_{r}"
            e(f"  %cr{t} = partition.receiver %c_all, %origin, %lane{t}, %ncols : reg<aie2p.array.receiver : tile<{rec}xi32>>")
            e(f"  %sc{t} = sender {kk[c, r]}, 2 : reg<aie2p.array.sender : tile<{rec}xi32>>")
            e(f"  %chc{t} = channel %sc{t}, %cr{t}, %n{2 if N.qalt_on() and not N.qaltp_on() else MP}, %nseg : reg<aie2p.array.channel : tile<{rec}xi32>>")
            e(f"  constrain.leaf_sync %chc{t}")
    wrec = NP * N.slab(ks)
    fz = cfg.fuse == 2
    Kb = (8 * P * ks) // 8
    iw = len(L)   # (fuse 2 reorders the sections below: C, the fill channels, W, A, the fill constraints)
    wty = f"{wrec // 4}"
    for c in range(cols):
        e(f"  %wo{c} = constant.u64 {c * P * wrec} : reg<aie2p.array.offset : offset>")
        e(f"  %ws{c} = sender %wb, 0 : reg<aie2p.array.sender : tile<{nbr}x{P}x{wrec // 4}xi32, "
          f"#encoding.layout.strided<strides=[0, {wrec // 4}, 1]>>>")
        e(f"  %wv{c} = view.sender %ws{c}, %wo{c} : reg<aie2p.array.sender : tile<{wty}xi32>>")
        for r in range(R):   # (one view, R channels: multicast down the column)
            t = f"{c}_{r}"
            e(f"  %rw{t} = receiver {kk[c, r]}, 1 : reg<aie2p.array.receiver : tile<{wty}xi32>>")
            e(f"  %chw{t} = channel %wv{c}, %rw{t}, %two, %rec : reg<aie2p.array.channel : tile<{wty}xi32>>")
            e(f"  constrain.stage %chw{t}, %n{c if cols > 1 else 1}")
            if N.qpass_on():   # (GN_QPASS) the leaf walks the W ring: one firing per M block
                e(f"  constrain.leaf_sync %chw{t}")
    ia = len(L)
    fa = N.slab(ks)
    e(f"  %arec = constant.u32 {nbr * P * MP} : reg<aie2p.array.scalar : index>")
    for r in range(R):   # row r: its M blocks' records [M block][pass][slab], multicast along the row
        e(f"  %ao{r} = constant.u64 {r * nbr * P * MP * fa} : reg<aie2p.array.offset : offset>")
        e(f"  %as{r} = sender %ab, 0 : reg<aie2p.array.sender : tile<{nbr * P * MP}x{fa // 4}xi32>>")
        e(f"  %a{r} = view.sender %as{r}, %ao{r} : reg<aie2p.array.sender : tile<{fa // 4}xi32>>")
        for j in range(cols):
            c = (2 * r + j) % cols
            t = f"{c}_{r}"
            e(f"  %ra{t} = receiver {kk[c, r]}, 0 : reg<aie2p.array.receiver : tile<{fa // 4}xi32>>")
            e(f"  %cha{t} = channel %a{r}, %ra{t}, %n{N.acap_tail(cfg)}, %arec : reg<aie2p.array.channel : tile<{fa // 4}xi32>>")
            e(f"  constrain.leaf_sync %cha{t}")
            e(f"  constrain.stage %cha{t}, %n{N.astage(r, cols) if cols > 1 else 2}")
    # (fuse 2) the egress ring before the panel it fills (W), the panel before the A stage (gen_npu_gemm.qf_ring_at)
    jf = len(L)
    if fz:   # per core: its egress ring (dummy write binding, never transferred) and its raw rows (rawdma: through its
        # W DMA into W slot 0); raw [N][kraw / 256][BLK] -> per column, units [slab][1024-K pass][16 rows][4 blocks]
        q = N.QF["cfg"]
        U, BLK = q.passes, N.fuse_blk(q)
        RS = (cfg.kraw or 8 * P * ks) // 256 * BLK // 4
        fw, nrec = N.fuse_record(q, (0,))
        dp, sp = N.QF_PORTS
        e('  %fdb = binding 3, "write"')
        e('  %frb = binding 4, "read"')
        e(f"  %fdl = constant.u32 {cols * R} : reg<aie2p.array.scalar : index>")
        e(f"  %fun = constant.u32 {U * N.FUSE_SB * NP} : reg<aie2p.array.scalar : index>")
        e(f"  %fdn = constant.u32 {nrec} : reg<aie2p.array.scalar : index>")
        e(f"  %fd_all = receiver %fdb, 0 : reg<aie2p.array.receiver : tile<{cols * R}x{nrec}x{fw}xi32>>")
        # records [super-block][slab][16 rows][one block] (super-blocks are BLK apart in a row)
        e(f"  %fru_all = sender %frb, 0 : reg<aie2p.array.sender : tile<{U * N.FUSE_SB}x{NP}x16x{BLK // 4}xi32, "
          f"#encoding.layout.strided<strides=[{BLK // 4}, {16 * RS}, {RS}, 1]>>>")
        for c in range(cols):
            e(f"  %fro{c} = constant.u64 {c * N.TN * RS * 4} : reg<aie2p.array.offset : offset>")
            e(f"  %fru{c} = view.sender %fru_all, %fro{c} : reg<aie2p.array.sender : tile<16x{BLK // 4}xi32>>")
            for r in range(R):
                t = f"{c}_{r}"
                e(f"  %fdr{t} = partition.receiver %fd_all, %origin, %lane{t}, %fdl : reg<aie2p.array.receiver : tile<{fw}xi32>>")
                e(f"  %fds{t} = sender {kk[c, r]}, {dp} : reg<aie2p.array.sender : tile<{fw}xi32>>")
                e(f"  %chd{t} = channel %fds{t}, %fdr{t}, %two, %fdn : reg<aie2p.array.channel : tile<{fw}xi32>>")
                e(f"  constrain.leaf_sync %chd{t}")
                e(f"  %frr{t} = receiver {kk[c, r]}, {sp} : reg<aie2p.array.receiver : tile<16x{BLK // 4}xi32>>")
                e(f"  %chr{t} = channel %fru{c}, %frr{t}, %one, %fun : reg<aie2p.array.channel : tile<16x{BLK // 4}xi32>>")
                e(f"  constrain.stage %chr{t}, %n{c if cols > 1 else 1}")
                e(f"  constrain.leaf_sync %chr{t}")
        assert R == 1, "fuse 2: one row (a column's panel filled by one core)"
    jt = len(L)
    if fz:   # filled by the core's egress; its raw rows land in its W ring's first slot
        L[iw:jt] = L[jf:jt] + L[iw:ia] + L[ia:jf]
        for c in range(cols):
            t = f"{c}_0"
            e(f"  constrain.fill %chw{t}, %chd{t}")
            e(f"  %fioff{t} = constant.u32 0 : reg<aie2p.array.scalar : index>")
            e(f"  %firows{t} = constant.u32 16 : reg<aie2p.array.scalar : index>")
            e(f"  %fipitch{t} = constant.u32 {N.decoder(cfg.fmt, cfg.gord).RECROW} : reg<aie2p.array.scalar : index>")
            e(f"  constrain.intake %chr{t}, %chw{t}, %fioff{t}, %firows{t}, %fipitch{t}")
    e("  return\n}\n")


def gen(cfg):
    assert len(cfg.ks) == 1 and cfg.fuse in (0, 2) and not cfg.gate and cfg.passes >= 2 and 1 <= cfg.cols <= 8
    L = []
    if cfg.fuse == 2:   # the decode side: 1024-K passes over the call's K, one 128-k-block slice
        K = 8 * cfg.passes * cfg.ks[0]
        assert K % 1024 == 0 and 32 % cfg.ks[0] == 0, "fuse 2: K a multiple of 1024, ks dividing a super-block"
        N.QF["cfg"] = N.Config(cfg.cols, cfg.nb, (128,), K // 1024, mu=1, fuse=2, kraw=cfg.kraw, fmt=cfg.fmt, rawdma=1)
        N.QF["wslot"] = N.NP * N.slab(cfg.ks[0])
    else:
        N.QF["cfg"] = None
    array_program(L, cfg)
    N.leaf(L, cfg, "tail", cfg.ks[0], cov=(0,) if cfg.fuse == 2 else None)
    return "\n".join(L)
