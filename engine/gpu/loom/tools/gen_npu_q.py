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


def crel():
    """C relay (2 rows; NPU_QCREL=0: off): row r takes M blocks 2 f + r; row 1's C ring goes to row 0's core (its south
    neighbor reads it in place), which copies each firing's segments after its own: one C stream a column (8, not 16:
    the shim has 16 stream-to-memory channels, and the gate takes 2)."""
    return QROWS >= 2 and os.environ.get("NPU_QCREL", "1") == "1"


def crel_mid_port():
    """(C relay chain, 3-4 rows) a middle row's port for the next row's C ring (past A 0, W 1, C 2)."""
    return 3


def qsplit():
    """(NPU_QSPLIT=1, the C relay's 2 rows) the fill split: a step's raw rows arrive as one record [16 rows][slab]
    [144 B] (the decoder's 576 B row: 4 slabs side by side), row 0 through its intake, row 1 as one W record; row 1
    decodes slabs 2, 3 (slab 2 into its idle C ring, slab 3 into the split channel's ring), row 0 slabs 0, 1, then
    egresses all four (row 1's through its north window). The split channel's locks hand a step's outputs over."""
    on = os.environ.get("NPU_QSPLIT") == "1"
    assert not on or (crel() and QROWS == 2), "NPU_QSPLIT needs the C relay at 2 rows (NPU_QROWS=2)"
    return on


def qsplit_port(gate):
    """Row 0's port for the split channel (past the relay's)."""
    return {None: 6, "waiter": 9, "relay": 8}[gate]


def crel_port(gate):
    """Row 0's port for row 1's C ring (past the filler's 0-4 and the gate's)."""
    return {None: 5, "waiter": 8, "relay": 7}[gate]
# The Q' core's knobs (gen_npu_gemm GN_Q*; defaults(): environment values win) and its solved sub-tile order
QCORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_core.json")
Q_ENV = dict(GN_QFLOAT="1", GN_QWCONT="1", GN_QLOCK="1", GN_QPASS="1", GN_QBANK="3", GN_QA4="1", GN_QEAGER="1",
             GN_QDRIP="12", GN_QROW="1", GN_QLRU="aie2p.mbms,aie2p.mexa", GN_QPEEL="1", GN_QCORE=QCORE)


def defaults():
    """Set the Q' knobs that the environment leaves unset."""
    for k, v in Q_ENV.items():
        os.environ.setdefault(k, v)


def loom_env(cfg):
    """loom-compile's environment for a Q' image of cfg (gen_npu_gemm.loom_env plus the Q' core's): the C block
    no-alias facts (generator-asserted: in one leaf body no C store writes a block a later C load reads) and the ring
    pins (GN_QBANK; GN_QPINS: the whole list; fuse 2: qf_ring_at)."""
    env = dict(N.loom_env(cfg))
    env.update(LOOM_EXP_NOALIAS_ACC="2", LOOM_EXP_NOWAR_ACC="1")
    ks = cfg.ks[0]
    if os.environ.get("GN_QBANK") in ("1", "2", "3"):
        env["LOOM_EXP_RING_AT"] = {"2": N.QBANK2_RING_AT, "3": N.QA4_RING_AT if N.qa4_on() else (N.QCB1_RING_AT if N.qcb1_on() else N.QA3_RING_AT) if N.qa3_on() else N.QBANK3_RING_AT}.get(
            os.environ["GN_QBANK"], N.QBANK_RING_AT)
        # (the pins are written for ks 28: W slots NP slabs, A slots one slab)
        env["LOOM_EXP_RING_AT"] = env["LOOM_EXP_RING_AT"].replace("16128@", f"{N.NP * N.slab(ks)}@").replace(
            "4032@", f"{N.slab(ks)}@")
    if os.environ.get("GN_QPINS"):
        env["LOOM_EXP_RING_AT"] = os.environ["GN_QPINS"]
    if cfg.fuse == 2:
        env["LOOM_EXP_RING_AT"] = N.qf_ring_at(env["LOOM_EXP_RING_AT"])
    return env


def array_program(L, cfg):
    e = L.append
    cols, nb, P, ks = cfg.cols, cfg.nb, cfg.passes, cfg.ks[0]
    MP, NP, R = N.MP, N.NP, QROWS
    assert 1 <= R <= 4 and nb % R == 0
    nbr = nb // R
    gw, gr = qgate(cfg)
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
            # (fuse 2: row 0 decodes its column's panel; the rows below read it, a leaf without the fill)
            e(f"  {kk[c, r]} = worker %workers, %lane{c}_{r}, @{leaf_name(cfg, c, r)}")
            e(f"  constrain.location {kk[c, r]}, %n{c}, %n{2 + r}")
    # C first (rings are placed in channel order: C claims whole banks before the input rings)
    rec = MP * NP * 4 * 256 // MP // 8
    e(f"  %ncols = constant.u32 {cols * R} : reg<aie2p.array.scalar : index>")
    e(f"  %nseg = constant.u32 {nbr * MP} : reg<aie2p.array.scalar : index>")
    cd = f"%n{2 if N.qalt_on() and not N.qaltp_on() else MP}"
    if crel():   # [col][M block]: row 0's C stream carries every row's blocks, interleaved (row r + 1's C ring is
        # read by row r, which copies its segments after its own: a chain)
        e(f"  %ncols1 = constant.u32 {cols} : reg<aie2p.array.scalar : index>")
        e(f"  %nseg2 = constant.u32 {nb * MP} : reg<aie2p.array.scalar : index>")
        e(f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols}x{nb * MP}x{rec}xi32>>")
        for c in range(cols):
            g = "waiter" if cfg.gate and c == 1 else "relay" if cfg.gate and c == 2 else None
            e(f"  %cr{c}_0 = partition.receiver %c_all, %origin, %n{c}, %ncols1 : reg<aie2p.array.receiver : tile<{rec}xi32>>")
            e(f"  %sc{c}_0 = sender {kk[c, 0]}, 2 : reg<aie2p.array.sender : tile<{rec}xi32>>")
            e(f"  %chc{c}_0 = channel %sc{c}_0, %cr{c}_0, {cd}, %nseg2 : reg<aie2p.array.channel : tile<{rec}xi32>>")
            e(f"  constrain.leaf_sync %chc{c}_0")
            for r in range(1, R):   # row r's ring: its blocks and the rows' above, R - r a firing
                e(f"  %nsg{c}_{r} = constant.u32 {(R - r) * nbr * MP} : reg<aie2p.array.scalar : index>")
                rp = crel_port(g) if r == 1 else crel_mid_port()
                e(f"  %cr{c}_{r} = receiver {kk[c, r - 1]}, {rp} : reg<aie2p.array.receiver : tile<{rec}xi32>>")
                e(f"  %sc{c}_{r} = sender {kk[c, r]}, 2 : reg<aie2p.array.sender : tile<{rec}xi32>>")
                e(f"  %chc{c}_{r} = channel %sc{c}_{r}, %cr{c}_{r}, {cd}, %nsg{c}_{r} : "
                  f"reg<aie2p.array.channel : tile<{rec}xi32>>")
                e(f"  constrain.leaf_sync %chc{c}_{r}")
    # [col][row][its M blocks] = [col][M block]: lane c R + r
    else:
        e(f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols * R}x{nbr * MP}x{rec}xi32>>")
    for c in range(cols if not crel() else 0):
        for r in range(R):
            t = f"{c}_{r}"
            e(f"  %cr{t} = partition.receiver %c_all, %origin, %lane{t}, %ncols : reg<aie2p.array.receiver : tile<{rec}xi32>>")
            e(f"  %sc{t} = sender {kk[c, r]}, 2 : reg<aie2p.array.sender : tile<{rec}xi32>>")
            e(f"  %chc{t} = channel %sc{t}, %cr{t}, {cd}, %nseg : reg<aie2p.array.channel : tile<{rec}xi32>>")
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
        if crel():   # (row r: M blocks r, r + R, ...)
            e(f"  %ao{r} = constant.u64 {r * P * MP * fa} : reg<aie2p.array.offset : offset>")
            e(f"  %as{r} = sender %ab, 0 : reg<aie2p.array.sender : tile<{nbr}x{P * MP}x{fa // 4}xi32, "
              f"#encoding.layout.strided<strides=[{R * P * MP * fa // 4}, {fa // 4}, 1]>>>")
        else:
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
        # raw row words: a block's; with rows below the filler, padded so the raw records (multicast with the panel)
        # are whole W records to them (qf_skip_raw)
        nraw, rw = U * N.FUSE_SB * NP, BLK // 4
        while R > 1 and nraw * 64 * rw % N.QF["wslot"]:
            rw += 1
        assert 4 * rw <= N.decoder(cfg.fmt, cfg.gord).RECROW
        N.QF["rawn"], N.QF["raww"] = nraw, rw
        dp, sp = N.QF_PORTS
        e(f'  %fdb = binding {6 if cfg.gate else 3}, "write"')   # (gated: the engine's binding order)
        e(f'  %frb = binding {7 if cfg.gate else 4}, "read"')
        e(f"  %fdl = constant.u32 {cols} : reg<aie2p.array.scalar : index>")
        e(f"  %fun = constant.u32 {U * N.FUSE_SB * NP} : reg<aie2p.array.scalar : index>")
        e(f"  %fdn = constant.u32 {nrec} : reg<aie2p.array.scalar : index>")
        e(f"  %fd_all = receiver %fdb, 0 : reg<aie2p.array.receiver : tile<{cols}x{nrec}x{fw}xi32>>")
        # records [super-block][slab][16 rows][one block] (super-blocks are BLK apart in a row)
        if qsplit():   # (a step's 4 slabs side by side in each row: the decoder's 576 B record row)
            rty = f"16x{NP}x{rw}"
            e(f"  %fru_all = sender %frb, 0 : reg<aie2p.array.sender : tile<{U * N.FUSE_SB}x16x{NP}x{rw}xi32, "
              f"#encoding.layout.strided<strides=[{BLK // 4}, {RS}, {16 * RS}, 1]>>>")
            e(f"  %fsn = constant.u32 {U * N.FUSE_SB} : reg<aie2p.array.scalar : index>")
        else:
            rty = f"16x{rw}"
            e(f"  %fru_all = sender %frb, 0 : reg<aie2p.array.sender : tile<{U * N.FUSE_SB}x{NP}x16x{rw}xi32, "
              f"#encoding.layout.strided<strides=[{BLK // 4}, {16 * RS}, {RS}, 1]>>>")
        for c in range(cols):
            e(f"  %fro{c} = constant.u64 {c * N.TN * RS * 4} : reg<aie2p.array.offset : offset>")
            e(f"  %fru{c} = view.sender %fru_all, %fro{c} : reg<aie2p.array.sender : tile<{rty}xi32>>")
            for r in range(1):   # (row 0 fills; the rows below see its raw rows as W records: gen_npu_gemm.qf_skip_raw)
                t = f"{c}_{r}"
                e(f"  %fdlane{c} = constant.u32 {c} : reg<aie2p.array.scalar : index>")
                e(f"  %fdr{t} = partition.receiver %fd_all, %origin, %fdlane{c}, %fdl : reg<aie2p.array.receiver : tile<{fw}xi32>>")
                e(f"  %fds{t} = sender {kk[c, r]}, {dp} : reg<aie2p.array.sender : tile<{fw}xi32>>")
                e(f"  %chd{t} = channel %fds{t}, %fdr{t}, %two, %fdn : reg<aie2p.array.channel : tile<{fw}xi32>>")
                e(f"  constrain.leaf_sync %chd{t}")
                e(f"  %frr{t} = receiver {kk[c, r]}, {sp} : reg<aie2p.array.receiver : tile<{rty}xi32>>")
                e(f"  %chr{t} = channel %fru{c}, %frr{t}, %one, {'%fsn' if qsplit() else '%fun'} : "
                  f"reg<aie2p.array.channel : tile<{rty}xi32>>")
                e(f"  constrain.stage %chr{t}, %n{c if cols > 1 else 1}")
                e(f"  constrain.leaf_sync %chr{t}")
    jt = len(L)
    if fz:   # filled by the core's egress; its raw rows land in its W ring's first slot
        L[iw:jt] = L[jf:jt] + L[iw:ia] + L[ia:jf]
        for c in range(cols):
            e(f"  constrain.fill %chw{c}_0, %chd{c}_0")
            for r in range(1):
                t = f"{c}_{r}"
                e(f"  %fioff{t} = constant.u32 0 : reg<aie2p.array.scalar : index>")
                e(f"  %firows{t} = constant.u32 16 : reg<aie2p.array.scalar : index>")
                e(f"  %fipitch{t} = constant.u32 {N.decoder(cfg.fmt, cfg.gord).RECROW} : reg<aie2p.array.scalar : index>")
                e(f"  constrain.intake %chr{t}, %chw{t}, %fioff{t}, %firows{t}, %fipitch{t}")
    if cfg.gate:   # (last: the pinned rings first) gen_npu_gemm's gate channels on the two gate cores
        S = cfg.gate
        e('  %fb = binding 3, "read"')
        e('  %tb = binding 4, "write"')
        e('  %sb = binding 5, "write"')
        e(f"  %gsup = constant.u32 {S} : reg<aie2p.array.scalar : index>")
        e(f"  %gf_all = sender %fb, 0 : reg<aie2p.array.sender : tile<1x{S}x1x4xi32, #encoding.layout.strided<strides=[4, 0, 0, 1]>>>")
        e("  %gf = partition.sender %gf_all, %origin, %n0, %one : reg<aie2p.array.sender : tile<4xi32>>")
        kw, kr = kk[gw], kk[gr]
        p = N.QF["gports"]
        e(f"  %gpoll = receiver {kw}, {p['waiter']['poll']} : reg<aie2p.array.receiver : tile<4xi32>>")
        e("  %chgp = channel %gf, %gpoll, %two, %gsup : reg<aie2p.array.channel : tile<4xi32>>")
        e("  constrain.core_stream %chgp")
        e(f"  %gt_all = receiver %tb, 0 : reg<aie2p.array.receiver : tile<1x{S}x1x4xi32, #encoding.layout.strided<strides=[4, 0, 0, 1]>>>")
        e("  %gt = partition.receiver %gt_all, %origin, %n0, %one : reg<aie2p.array.receiver : tile<4xi32>>")
        e(f"  %gtick = sender {kw}, {p['waiter']['tick']} : reg<aie2p.array.sender : tile<4xi32>>")
        e("  %chgt = channel %gtick, %gt, %two, %gsup : reg<aie2p.array.channel : tile<4xi32>>")
        e("  constrain.core_stream %chgt")
        e("  constrain.request %chgp, %chgt")
        e(f"  %gn1s = sender {kw}, {p['waiter']['nbr']} : reg<aie2p.array.sender : tile<4xi32>>")
        e(f"  %gn1r = receiver {kr}, {p['relay']['nbr']} : reg<aie2p.array.receiver : tile<4xi32>>")
        e("  %chgn = channel %gn1s, %gn1r, %one, %one : reg<aie2p.array.channel : tile<4xi32>>")
        e("  constrain.leaf_sync %chgn")
        e("  %gs_all = receiver %sb, 0 : reg<aie2p.array.receiver : tile<1x2x4xi32>>")
        e("  %gs = partition.receiver %gs_all, %origin, %n0, %one : reg<aie2p.array.receiver : tile<4xi32>>")
        e(f"  %gsig = sender {kr}, {p['relay']['sig']} : reg<aie2p.array.sender : tile<4xi32>>")
        e("  %chgs = channel %gsig, %gs, %two, %two : reg<aie2p.array.channel : tile<4xi32>>")
        e("  constrain.core_stream %chgs")
        e("  constrain.gate %chgs")
        e("  constrain.signal %chgs")
    if fz and qsplit():   # row 1 -> row 0: a step's slab-3 output (slab 2: row 1's C ring), one record a step
        rq = 2 * N.decoder(cfg.fmt, cfg.gord).OUT_B // 4
        e(f"  %qsn = constant.u32 {N.QF['cfg'].passes * N.FUSE_SB} : reg<aie2p.array.scalar : index>")
        for c in range(cols):
            g = "waiter" if cfg.gate and c == 1 else "relay" if cfg.gate and c == 2 else None
            e(f"  %qss{c} = sender {kk[c, 1]}, 3 : reg<aie2p.array.sender : tile<{rq}xi32>>")
            e(f"  %qsr{c} = receiver {kk[c, 0]}, {qsplit_port(g)} : reg<aie2p.array.receiver : tile<{rq}xi32>>")
            e(f"  %chq{c} = channel %qss{c}, %qsr{c}, %one, %qsn : reg<aie2p.array.channel : tile<{rq}xi32>>")
            e(f"  constrain.leaf_sync %chq{c}")
    e("  return\n}\n")


def qgate(cfg):
    """(gated) the waiter and relay cores, (column 1, the last row) and (column 2, the last row), and their ports past
    the leaf's (A 0, W 1, C 2; a filler's egress 3, raw 4): gen_npu_gemm.gate_ports reads QF["gports"]."""
    if not cfg.gate:
        N.QF["gports"] = None
        return None, None
    assert cfg.cols >= 3
    r = gate_row()
    b = 5 if r == 0 and cfg.fuse == 2 else 3
    N.QF["gports"] = {"col": 1, "waiter": dict(row=r, poll=b, tick=b + 1, nbr=b + 2),
                      "relay": dict(col=2, row=r, nbr=b, sig=b + 1)}
    return (1, r), (2, r)


def gate_row():
    """The gate cores' row: the last (crel: row 0, whose C stream ends the job)."""
    return 0 if crel() else QROWS - 1


def leaf_name(cfg, c, r):
    """Worker (c, r)'s leaf: tail (fuse 2: row 0 decodes the panel), tailr (the rows below), _gw / _gr: the gate's."""
    name = "tailr" if cfg.fuse == 2 and r else "tail"
    if r and crel() and r < QROWS - 1:   # (a middle row of the relay chain: tailr<r>)
        name += str(r)
    if cfg.gate and (c, r) == (1, gate_row()):
        return name + "_gw"
    if cfg.gate and (c, r) == (2, gate_row()):
        return name + "_gr"
    return name


def leaves(cfg):
    """[(name, gate role)] of the image's distinct leaves."""
    out = {}
    for c in range(cfg.cols):
        for r in range(QROWS):
            n = leaf_name(cfg, c, r)
            out[n] = "waiter" if n.endswith("_gw") else "relay" if n.endswith("_gr") else None
    return list(out.items())


def gen(cfg):
    assert len(cfg.ks) == 1 and cfg.fuse in (0, 2) and cfg.passes >= 2 and 1 <= cfg.cols <= 8
    L = []
    if cfg.fuse == 2:   # the decode side: 1024-K passes over the call's K, one 128-k-block slice
        K = 8 * cfg.passes * cfg.ks[0]
        assert K % 1024 == 0 and 32 % cfg.ks[0] == 0, "fuse 2: K a multiple of 1024, ks dividing a super-block"
        N.QF["cfg"] = N.Config(cfg.cols, cfg.nb, (128,), K // 1024, mu=1, fuse=2, kraw=cfg.kraw, fmt=cfg.fmt, rawdma=1)
        N.QF["wslot"] = N.NP * N.slab(cfg.ks[0])
    else:
        N.QF["cfg"] = None
    N.QF["rows"] = QROWS
    N.QF["crel"] = crel_port if crel() else None
    N.QF["crelmid"] = crel_mid_port() if crel() else None
    N.QF["split"] = qsplit_port if qsplit() else None
    array_program(L, cfg)
    for name, gate in leaves(cfg):   # (tailr: the GEMM leaf without the fill, skipping the raw rows)
        fill = cfg.fuse == 2 and not name.startswith("tailr")
        N.leaf(L, cfg, "tail", cfg.ks[0], cov=(0,) if fill else None, gate=gate, name=name)
    return "\n".join(L)

