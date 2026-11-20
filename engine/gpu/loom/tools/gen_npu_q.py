#!/usr/bin/env python3
"""gen_npu_q.py: layout Q' GEMM images (experimental, stage 1 of the Q' build). Every worker is an independent core
that owns a TM x TN output tile over the whole K: f32 sums in its own memory, no cascade, no roles. The leaf is
gen_npu_gemm's tail in GN_QCORE mode (plain MMAs from C, in-row sub-tile boundaries in a solved order).
Stage 1 array: one row of cols workers (array row 2). A: one stream multicast along the row (records [M block][pass]
[slab]); W: per column, its panel [pass][record] staged in the column's memory tile and replayed per M block; C: per
column, segments [M block][slab] as bf16 (the tail's last-pass convert). Same bindings as gen_npu_gemm with one K
slice, so npu_gemm_check.py checks it (NPU_Q=1, ks of one entry, NP 4)."""
import os
import re

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


def qdec():
    """(NPU_QDEC=1, fuse 2) decoder rows: a core above the compute rows decodes its column's panel (raw on its own DMA
    ring, a step's record [16 rows][slab][144 B]; its scratch private), egresses it into the panel, and computes
    nothing; the compute rows run the GEMM leaf without the fill."""
    on = os.environ.get("NPU_QDEC") == "1"
    assert not on or (QROWS in (1, 2) and not qsplit()), "NPU_QDEC: one or two compute rows, no fill split"
    return on


def qdec_rows():
    """(NPU_QDEC) the decoder rows: one (all 4 slabs), or two with two compute rows (the second decodes slabs 2-3
    and sends them to the first over a core channel, which egresses all 4)."""
    return int(os.environ.get("NPU_QDECN", 1 if QROWS == 1 else 2)) if qdec() else 0


def qsplit_port(gate):
    """Row 0's port for the split channel (past the relay's)."""
    return {None: 6, "waiter": 9, "relay": 8}[gate]


def qfix():
    """(GN_QDIAG_FIX=a, diag: wrong output) the intake test: A reread from the core's ring, its locks dropped."""
    return os.environ.get("GN_QDIAG_FIX", "")


QFIX_A = ("  acq %am1, 0", "  acq.cond %am1, %cqn, 0", "  acq.cond %am1, %cqn_P, 0", "  rel %one, 0")
QFIX_W = ("  acq %am1, 1", "  acq.cond %am1, %waq, 1", "  acq.cond %am1, %waq_P, 1", "  rel.cond %one, %wrq, 1",
          "  rel.cond %one, %wrq_P, 1")   # (fuse 0 only: the fill takes W locks too)
QFIX_DROP = {"a": QFIX_A, "w": QFIX_W, "aw": QFIX_A + QFIX_W}


def crel_port(gate):
    """Row 0's port for row 1's C ring (past the filler's 0-4 and the gate's)."""
    return {None: 5, "waiter": 8, "relay": 7}[gate]
# The Q' core's knobs (gen_npu_gemm GN_Q*; defaults(): environment values win) and its solved sub-tile order
QCORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_core.json")
Q_ENV = dict(GN_QFLOAT="1", GN_QWCONT="1", GN_QLOCK="1", GN_QPASS="1", GN_QBANK="3", GN_QA4="1", GN_QEAGER="1",
             GN_QDRIP="12", GN_QROW="1", GN_QLRU="aie2p.mbms,aie2p.mexa", GN_QPEEL="1", GN_QCORE=QCORE)


# The 64x80 P1K image (NP 5, ks 32 x 20 passes, 4 rows, fuse 2): row 1 of each column decodes the panel straight into
# its egress ring, the vertical pairs share W by K half (GN_QP1K), each pair's row 0 relays row 1's C and the column's
# memory tile merges the two pair streams (A, C in token order, gen_npu_q.mrel). 11-17: gated run bit-exact on
# UD-IQ4_XS rows, 67K cycles a 64-token block (76% MMA share), the first block of a 2048-token call at ~364K.
# 11-17 night: K-half split (GN_QFKH: rows 0 / 1 each decode one K half, own egress lane) with the gate's tick and go /
# done packet-shared on the row 0 egress lanes of columns 1 / 2 (GN_QPSHARE, HRX 0094): real calls qkv 52.6 -> 57.2%,
# gate 38.7 -> 43.4% MMA busy; pp2048 2682.9 -> 2667.6 ms, bit-exact.
# 11-18: 4-row decode (GN_QF4R: all 4 rows decode, the raw intake split by record parity over the two W multicasts,
# Loom LOOM_EXP_INTAKE_SPLIT, HRX 0095) and the gate's leftover polls drained 32 a firing (gen_npu_gemm GB, was 8):
# real calls qkv 57.2 -> 61.0%, gate 43.4 -> 52.9% MMA busy, bit-exact.
P1K_NP, P1K_KS, P1K_PASSES = 5, (32,), 20
P1K_ENV = dict(GN_ORDC="3,2,0,1", GN_QPINS="p1k",
               GN_QZACC="1", GN_QRELAX="1", GN_QTOK="-6,0,0,10", GN_QDRIP="4", GN_QEMERGE="1", GN_QWDUP="1",
               GN_QCLAZY="1", NPU_QCREL="2", GN_QLOCKA="12,5", GN_QWCONT="0", GN_QPEEL="0", GN_QP1="1", GN_QMB1="5",
               GN_QP1K="1", GN_QCVL="1", GN_QFKH="1", GN_QPSHARE="1", GN_QF4R="1", DQ_HDRWIN="8",
               GN_QLSDEPTH="7",   # (11-18: the A streams staged 7 deep in the memory tiles: gate 796K -> 731K a call)
               GN_QPREFILL="1",   # (11-18: the gate cores check after their fill: a job's first fill in the GPU wait)
               GN_QFGUARD="1",    # (11-18: a fill waits for its core's C drain: safe at queue depth 2, see gen_npu_gemm)
               GN_QF4RSWAP="1",   # (11-18: rows 0 / 1 swap fill roles: each egress row owns its half's first record)
               GN_QF4RCOPY="2",   # (11-18: helper copies pipelined, groups of 6: copy cycles 41.9K -> 34.1K a job)
               GN_QFEGR2="1",     # (11-18: egress ring of 4 slabs: with the copies, the last call's first block -4.8K)
               GN_QLRU="aie2p.mbms")   # (11-18: no LRU on the operand vectors: memstall -1.5K a call)
P1K_LOOM = dict(LOOM_EXP_RING_AT_FORCE="1", LOOM_EXP_SPACE_AT="scratch@12288", LOOM_EXP_EGRESS_MERGE="1",
                LOOM_EXP_REPLAY_GROUP="4", LOOM_EXP_MERGE_RING_LOW="1", LOOM_EXP_PANEL_STREAM="1",
                LOOM_EXP_PANEL_STREAM_SINGLE="1",
                LOOM_EXP_LOCK_ACQREL="6")   # (11-18: locks order as acquire / release only: loop 696 -> 685 bundles)


def defaults(p1k=False):
    """Set the Q' knobs that the environment leaves unset (p1k: the 64x80 P1K image's first)."""
    for k, v in list(P1K_ENV.items() if p1k else ()) + list(Q_ENV.items()):
        os.environ.setdefault(k, v)
    N.ORDER_C = tuple(int(x) for x in os.environ.get("GN_ORDC", "0,2,1,3").split(","))   # (read at import)
    N.qf_p1kd_layout()   # (read at import too)


def loom_env(cfg):
    """loom-compile's environment for a Q' image of cfg (gen_npu_gemm.loom_env plus the Q' core's): the C block
    no-alias facts (generator-asserted: in one leaf body no C store writes a block a later C load reads) and the ring
    pins (GN_QBANK; GN_QPINS: the whole list; fuse 2: qf_ring_at)."""
    N.QF["qsw"] = cfg.qsw and qswdiag() != "intake"   # (the C pins: qpins)
    grid_flags(cfg)
    env = dict(N.loom_env(cfg))
    env.update(LOOM_EXP_NOALIAS_ACC="2", LOOM_EXP_NOWAR_ACC="1")
    if N.QF["qsw"]:   # the silu table between the C rows' f32 areas and W slot 1 (the rings are forced, and Loom's
        # own place, offset 0, is under W slot 0; qf_ring_at moves the K-half egress ring off 49152)
        env["LOOM_EXP_RODATA_AT"] = ",".join(f"{N.silu_sym(k)}@{a}" for k, (_, a) in enumerate(N.silu_pieces()))
        if len(N.silu_pieces()) > 1 and not N.qgmap_on():   # (GN_QSILU by hand: GN_QSILUK, @silu_k's address)
            env["LOOM_EXP_RODATA_AT"] += f",silu_k@{os.environ.get('GN_QSILUK') or N.QF['siluk']}"
        env["LOOM_EXP_EGRESS_MERGE_STRIDED"] = "1"
    if p1k4(cfg):   # (NP 4) the scratch pad's guard
        env["LOOM_EXP_RODATA_AT"] = ",".join(([env["LOOM_EXP_RODATA_AT"]] if "LOOM_EXP_RODATA_AT" in env else [])
                                             + [f"qfgd@{N.p1k4_guard()}"])
    if N.qgmap_on():   # (FFN gate / up, a grid format: gen_npu_gemm QG_*) the tables, @silu_k
        env["LOOM_EXP_RODATA_AT"] = ",".join([env["LOOM_EXP_RODATA_AT"]] + qgmap_pins(cfg))
    elif N.QF["grid"]:   # the decoder's tables (gen_npu_gemm P1K_GRID_*: 49152 .. 53760, the others in spare runs),
        # largest first
        at, pins, used = N.P1K_GRID_AT, [], set()
        ms = [re.match(r"global.rodata.def @(\w+) = align\((\d+)\).*bytes\(\"([0-9a-fA-F]*)\"\)", ln)
              for ln in N.decoder(cfg.fmt, cfg.gord).inline_rodata(N.dec_tag(cfg))]
        for m in sorted(ms, key=lambda m: -len(m[3])):
            a, n = -(-at // int(m[2])) * int(m[2]), len(m[3]) // 2
            if a + n > N.P1K_GRID_END:   # (a spare run, once each)
                fit = [s for s, t in N.P1K_GRID_SPARE if s % int(m[2]) == 0 and s + n <= t and s not in used]
                assert pins and fit, (cfg.fmt, m[1], n)
                a = fit[0]
                used.add(a)
            else:
                at = a + n
            pins.append(f"{m[1]}@{a}")
        env["LOOM_EXP_RODATA_AT"] = ",".join(([env["LOOM_EXP_RODATA_AT"]] if "LOOM_EXP_RODATA_AT" in env else []) + pins)
    if N.pshare():   # (array_program counted the two channels)
        env["LOOM_EXP_PACKET_SHARE"] = N.QF["pshare"]
    ks = cfg.ks[0]
    if os.environ.get("GN_QBANK") in ("1", "2", "3"):
        env["LOOM_EXP_RING_AT"] = {"2": N.QBANK2_RING_AT, "3": N.QA4_RING_AT if N.qa4_on() else (N.QCB1_RING_AT if N.qcb1_on() else N.QA3_RING_AT) if N.qa3_on() else N.QBANK3_RING_AT}.get(
            os.environ["GN_QBANK"], N.QBANK_RING_AT)
        # (the pins are written for ks 28: W slots NP slabs, A slots one slab)
        env["LOOM_EXP_RING_AT"] = env["LOOM_EXP_RING_AT"].replace("16128@", f"{N.NP * N.slab(ks)}@").replace(
            "4032@", f"{N.slab(ks)}@")
    if os.environ.get("GN_QPINS"):
        env["LOOM_EXP_RING_AT"] = N.qpins()
    if cfg.fuse == 2:
        env["LOOM_EXP_RING_AT"] = N.qf_ring_at(env["LOOM_EXP_RING_AT"])
    if N.QSWAP and os.environ.get("GN_QSWPAIR") == "1":
        # (row 5 swap image, private HRX 0099 + 0112 + 0114, not in the pin yet) 8 A streams: each A stage's odd
        # records ride the east column's second shim MM2S into a second ring of the stage's own memory tile; column
        # 1's W panel shares column 0's shim MM2S0 as packets (parted at shim 0), freeing a channel for the gate's
        # poll. 11-20: pp2048 split 1808.3 -> 1754.5 ms, same md5, ffnswap 18.16 -> 14.57 M a job.
        env.update(LOOM_EXP_INGRESS_PAIR="2", LOOM_EXP_SHIM_SHARE="36:32")
    if os.environ.get("GN_QRELAX") == "1":   # (HRX 0087) no MMA-latency drain before the row loop's back edge
        env["LOOM_EXP_HWLOOP_TERM_RELAX"] = "1"
    if N.crelp_on():   # (HRX 0089) the pair relay's channel and the P1 tokens form a leaf-synchronized cycle
        env["LOOM_EXP_LEAF_SYNC_CYCLES"] = "1"
    if N.qp1k_on() and cfg.fuse == 2:   # (the P1K image's placement and stream knobs)
        env.update(P1K_LOOM)
    if N.qgmap_on():   # (gen_npu_gemm QG_*) private storage after W slot 1, below the spare runs: the tables' pins
        # in 63488 .. 65536 grow bank 3's cursor past where Loom would place it
        assert N.QG_PRIVATE + 512 <= N.QG_SPARE[1][0]
        env["LOOM_EXP_SPACE_AT"] = ",".join(filter(None, [env.get("LOOM_EXP_SPACE_AT"), f"private@{N.QG_PRIVATE}"]))
    if N.QF.get("silu") == QSW5_SILU:   # (qsw5) private storage (<= 320 B) past the token ring, below silu piece 1:
        # Loom's own choice lands in the even C area's last row
        assert QSW5_PRIVATE + 320 <= QSW5_SILU[1][1]
        env["LOOM_EXP_SPACE_AT"] = ",".join(filter(None, [env.get("LOOM_EXP_SPACE_AT"), f"private@{QSW5_PRIVATE}"]))
        # (HRX 0100) the gate's poll and tick supplies queued 4 x 256 on one descriptor each: the raw rows' 5-run
        # chain (10 descriptors a column) leaves column 1's shim tile, which holds them, 6 of its 16
        env["LOOM_EXP_SUPPLY_QUEUE"] = "1"
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
    RW_ = R + qdec_rows()   # (worker rows: the compute rows, then the decoder rows)
    for k, v in (("zero", 0), ("one", 1), ("two", 2), ("nw", cols * RW_), ("rec", nbr * P)) + ((("qw4n", 4),) if N.qw4_on() else ()):
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
        for r in range(RW_):
            kk[c, r] = f"%k{c}_{r}"
            lane_ = c * R + r if r < R else cols * r + c   # (compute lanes first: the C partitions index them)
            e(f"  %lane{c}_{r} = constant.u32 {lane_} : reg<aie2p.array.scalar : index>")
            # (fuse 2: row 0 decodes its column's panel; the rows below read it, a leaf without the fill)
            e(f"  {kk[c, r]} = worker %workers, %lane{c}_{r}, @{leaf_name(cfg, c, r)}")
            e(f"  constrain.location {kk[c, r]}, %n{c}, %n{2 + r}")
    # C first (rings are placed in channel order: C claims whole banks before the input rings)
    rec = N.SW_REC // 4 if N.QF["qsw"] else MP * NP * 4 * 256 // MP // 8   # (qsw: a 16-token row's h records)
    e(f"  %ncols = constant.u32 {cols * R} : reg<aie2p.array.scalar : index>")
    e(f"  %nseg = constant.u32 {nbr * MP} : reg<aie2p.array.scalar : index>")
    cd = f"%n{2 if N.qalt_on() and not N.qaltp_on() else MP}"
    if crel():   # [col][M block]: row 0's C stream carries every row's blocks, interleaved (row r + 1's C ring is
        # read by row r, which copies its segments after its own: a chain)
        e(f"  %ncols1 = constant.u32 {cols} : reg<aie2p.array.scalar : index>")
        e(f"  %nseg2 = constant.u32 {nb * MP} : reg<aie2p.array.scalar : index>")
        e(f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols}x{nb * MP}x{rec}xi32>>")
        for c in range(cols):
            g = "waiter" if cfg.gate and c == gw[0] else "relay" if cfg.gate and c == gr[0] else None
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
    if N.crelp_on():   # (NPU_QCREL=2) [col][pair][its blocks]: row 2p + j takes pair p's blocks 2 f + j; row 2p's
        # stream carries both rows' (row 2p + 1's C ring read in place by row 2p, copied after its own)
        assert R % 2 == 0 and (cfg.fuse != 2 or N.qp1k_on())
        L[-1] = (f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols * R // 2}x{2 * nbr * MP}x{rec}xi32>>")
        if N.QF["qsw"]:   # (FFN gate / up) H straight in down's A layout [M block][call][slab][column][576 B], this
            # call's records at its binding offset: lane 2 c + p (the merged owner, lane / 2 = c) at column c's piece,
            # its blocks hpass calls' records apart (Loom LOOM_EXP_EGRESS_MERGE_STRIDED, HRX 0097)
            assert mrel() and cfg.hpass >= 1 and rec * 4 == N.SW_REC and R == 4
            hm_ = cfg.hpass * MP * 8 * rec
            L[-1] = (f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols * R // 2}x{2 * nbr}x{MP}x{rec}xi32, "
                     f"#encoding.layout.strided<strides=[{rec}, {hm_}, {8 * rec}, 1]>>>")
            if N.QSWAP:   # (row 5 role swap) the merged lanes' first index is the token column (= its M block of
                # H), the second the weight block (32 features): H slab-major [M block][slab][F / 32 pieces][576 B]
                # (the down reads it with GN_QHSLAB strides); this call's blocks at its binding offset
                L[-1] = (f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols * R // 2}x{2 * nbr}x{MP}x{rec}xi32, "
                         f"#encoding.layout.strided<strides=[{hm_}, {rec}, {8 * cfg.hpass * rec}, 1]>>>")
                if os.environ.get("GN_QSWHT") == "1":   # (row 5, diag) H in today's down A layout [M block][call]
                    # [slab][column][576 B]: a lane's blocks as [2 nbr / 8 calls][8 columns]
                    assert 2 * nbr % 8 == 0
                    L[-1] = (f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols * R // 2}x{2 * nbr // 8}x8x{MP}x{rec}xi32, "
                             f"#encoding.layout.strided<strides=[{hm_}, {MP * 8 * rec}, {rec}, {8 * rec}, 1]>>>")
            if qsw5_on(cfg):   # (qsw5: column c's 144 B at slot c of each of 5 passes; rows 4608 B apart)
                hm_ = cfg.hpass * MP * 1152
                L[-1] = (f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols * R // 2}x{2 * nbr}x{MP}x5x36xi32, "
                         f"#encoding.layout.strided<strides=[36, {hm_}, 1152, {MP * 1152}, 1]>>>")
        e(f"  %ncolp = constant.u32 {cols * R // 2} : reg<aie2p.array.scalar : index>")
        e(f"  %nseg2 = constant.u32 {2 * nbr * MP} : reg<aie2p.array.scalar : index>")
        cty = "5x36" if N.QF["qsw"] and qsw5_on(cfg) else f"{rec}"   # (qsw5: a record as its 5 pieces)
        for c in range(cols):
            for q in range(0, R, 2):
                t, u = f"{c}_{q}", f"{c}_{q + 1}"
                e(f"  %lanep{t} = constant.u32 {c * R // 2 + q // 2} : reg<aie2p.array.scalar : index>")
                e(f"  %cr{t} = partition.receiver %c_all, %origin, %lanep{t}, %ncolp : reg<aie2p.array.receiver : tile<{cty}xi32>>")
                e(f"  %sc{t} = sender {kk[c, q]}, 2 : reg<aie2p.array.sender : tile<{cty}xi32>>")
                e(f"  %chc{t} = channel %sc{t}, %cr{t}, {'%n8' if N.qcstage_on() else cd}, %nseg2 : reg<aie2p.array.channel : tile<{cty}xi32>>")
                e(f"  constrain.leaf_sync %chc{t}")
                if os.environ.get("GN_QEMERGE") == "1":   # (LOOM_EXP_EGRESS_MERGE) the column's memtile merges its pairs'
                    e(f"  constrain.stage %chc{t}, %n{c if cols > 1 else 1}")   # streams into one shim write
                e(f"  %cr{u} = receiver {kk[c, q]}, 5 : reg<aie2p.array.receiver : tile<{rec}xi32>>")
                e(f"  %sc{u} = sender {kk[c, q + 1]}, 2 : reg<aie2p.array.sender : tile<{rec}xi32>>")
                e(f"  %chc{u} = channel %sc{u}, %cr{u}, {cd}, %nseg : reg<aie2p.array.channel : tile<{rec}xi32>>")
                e(f"  constrain.leaf_sync %chc{u}")
    for c in range(cols if not crel() and not N.crelp_on() else 0):
        for r in range(R):
            t = f"{c}_{r}"
            e(f"  %cr{t} = partition.receiver %c_all, %origin, %lane{t}, %ncols : reg<aie2p.array.receiver : tile<{rec}xi32>>")
            e(f"  %sc{t} = sender {kk[c, r]}, 2 : reg<aie2p.array.sender : tile<{rec}xi32>>")
            e(f"  %chc{t} = channel %sc{t}, %cr{t}, {cd}, %nseg : reg<aie2p.array.channel : tile<{rec}xi32>>")
            e(f"  constrain.leaf_sync %chc{t}")
            if os.environ.get("GN_QEMERGE") == "1":   # (LOOM_EXP_EGRESS_MERGE) the column's memory tile merges its
                # cores' C rings into one shim write, a block of each row in turn (row r takes blocks b R + r)
                e(f"  constrain.stage %chc{t}, %n{c if cols > 1 else 1}")
    wrec = NP * N.slab(ks)
    fz = cfg.fuse == 2
    Kb = (8 * P * ks) // 8
    iw = len(L)   # (fuse 2 reorders the sections below: C, the fill channels, W, A, the fill constraints)
    wty = f"{wrec // 4}"
    wdup = int(os.environ.get("GN_QWDUP") or 1)   # (diag: groups of wdup columns share one W stream; 2: production's DRAM)
    assert cols % wdup == 0
    for c in range(0, cols if N.qp1_on() else 0, wdup):   # (GN_QP1) a half-record view per pair row parity: row r gets half r % 2
        assert (not fz or N.qp1k_on()) and not N.qw4_on() and "w" not in qfix() and N.qpass_on() and R % 2 == 0
        hrec = wrec // 2
        for h in range(2):
            e(f"  %wo{c}_h{h} = constant.u64 {c * P * wrec + h * P * hrec} : reg<aie2p.array.offset : offset>")
            e(f"  %ws{c}_h{h} = sender %wb, 0 : reg<aie2p.array.sender : tile<{nbr}x{P}x{hrec // 4}xi32, "
              f"#encoding.layout.strided<strides=[0, {hrec // 4}, 1]>>>")
            e(f"  %wv{c}_h{h} = view.sender %ws{c}_h{h}, %wo{c}_h{h} : reg<aie2p.array.sender : tile<{hrec // 4}xi32>>")
            for c2, r in ((c2, r) for c2 in range(c, c + wdup) for r in range(h, R, 2)):
                t = f"{c2}_{r}"
                e(f"  %rw{t} = receiver {kk[c2, r]}, 1 : reg<aie2p.array.receiver : tile<{hrec // 4}xi32>>")
                e(f"  %chw{t} = channel %wv{c}_h{h}, %rw{t}, %two, %rec : reg<aie2p.array.channel : tile<{hrec // 4}xi32>>")
                # (GN_QWDUP: half h of the column pair stages in column c + h: one W half a shim)
                e(f"  constrain.stage %chw{t}, %n{(c + h if wdup > 1 else c) if cols > 1 else 1}")
                e(f"  constrain.leaf_sync %chw{t}")
    for c in range(0 if N.qp1_on() else cols):
        e(f"  %wo{c} = constant.u64 {c * P * wrec} : reg<aie2p.array.offset : offset>")
        assert "w" not in qfix() or not fz, "GN_QDIAG_FIX=w: fuse 0 only"
        nw_, pw_ = (1, 2) if "w" in qfix() else (nbr, P)
        e(f"  %ws{c} = sender %wb, 0 : reg<aie2p.array.sender : tile<{nw_}x{pw_}x{wrec // 4}xi32, "
          f"#encoding.layout.strided<strides=[0, {wrec // 4}, 1]>>>")
        e(f"  %wv{c} = view.sender %ws{c}, %wo{c} : reg<aie2p.array.sender : tile<{wty}xi32>>")
        for r in range(R):   # (one view, R channels: multicast down the column)
            t = f"{c}_{r}"
            e(f"  %rw{t} = receiver {kk[c, r]}, 1 : reg<aie2p.array.receiver : tile<{wty}xi32>>")
            e(f"  %chw{t} = channel %wv{c}, %rw{t}, {'%qw4n' if N.qw4_on() else '%two'}, {'%two' if 'w' in qfix() else '%rec'} : "
              f"reg<aie2p.array.channel : tile<{wty}xi32>>")
            e(f"  constrain.stage %chw{t}, %n{c if cols > 1 else 1}")
            if N.qpass_on():   # (GN_QPASS) the leaf walks the W ring: one firing per M block
                e(f"  constrain.leaf_sync %chw{t}")
    ia = len(L)
    fa = N.slab(ks)
    PA = cfg.apass or P   # (Config.apass: A's passes a M block, the call's P of them at the binding's offset)
    assert PA >= P
    # (GN_QDIAG_FIX=a, diag: the A channels carry the ring's slots only; the leaves reread them without locks)
    e(f"  %arec = constant.u32 {N.acap_tail(cfg) if 'a' in qfix() else nbr * P * MP} : reg<aie2p.array.scalar : index>")
    art = f"{fa // 4}"   # (the pair relay: the host orders A so row 2 p + j's blocks are contiguous, npu_gemm_check)
    asp_ = 1 if fa // 4 <= 1023 or not (crel() or mrel()) else 2   # (strided A: the shim DMA's wrap fields hold 1023
    if asp_ > 1:                                                     # words, so a record of two halves)
        art = f"{asp_}x{fa // 4 // asp_}"
    for r in range(R):   # row r: its M blocks' records [M block][pass][slab], multicast along the row
        if crel() or mrel():   # (row r: M blocks r, r + R, ...; merged pair relay: row 2 p + j's 4 f + 2 j + p)
            b0 = r if crel() or N.qcstage_on() else 2 * (r % 2) + r // 2   # (GN_QCSTAGE: the merge takes a pair round: row r owns 4 f + r)
            e(f"  %ao{r} = constant.u64 {b0 * PA * MP * fa} : reg<aie2p.array.offset : offset>")
            na_, ra_ = (1, N.acap_tail(cfg)) if "a" in qfix() else (nbr, P * MP)
            assert fa // 4 % asp_ == 0 and fa // 4 // asp_ <= 1023
            ah_ = f"{asp_}x" if asp_ > 1 else ""
            st_ = f"{fa // 4 // asp_}, " if asp_ > 1 else ""
            e(f"  %as{r} = sender %ab, 0 : reg<aie2p.array.sender : tile<{na_}x{ra_}x{ah_}{fa // 4 // asp_}xi32, "
              f"#encoding.layout.strided<strides=[{R * PA * MP * fa // 4}, {fa // 4}, {st_}1]>>>")
        else:
            assert PA == P, "GN Config.apass: the pair relay / merge's strided A (mrel / crel) only"
            e(f"  %ao{r} = constant.u64 {r * nbr * P * MP * fa} : reg<aie2p.array.offset : offset>")
            rpl = int(os.environ.get("GN_QAREPLAY") or 0)   # (timing diag: the memory tile replays an rpl-record A panel,
            if rpl:                                        # no A from DDR in steady state; results wrong)
                assert nbr * P * MP % rpl == 0 and "a" not in qfix()
                e(f"  %as{r} = sender %ab, 0 : reg<aie2p.array.sender : tile<{nbr * P * MP // rpl}x{rpl}x{fa // 4}xi32, "
                  f"#encoding.layout.strided<strides=[0, {fa // 4}, 1]>>>")
            else:
                e(f"  %as{r} = sender %ab, 0 : reg<aie2p.array.sender : tile<{N.acap_tail(cfg) if 'a' in qfix() else nbr * P * MP}x{fa // 4}xi32>>")
        # (GN_QASPLIT=2: the row's columns in two halves, each its own stream staged in its own memory tile, so a
        # multicast stalls on 4 receivers, not 8)
        srows = os.environ.get("GN_QASPLIT_ROWS")   # (diag: split only these rows)
        asp = 2 if os.environ.get("GN_QASPLIT") == "2" and cols >= 4 and (not srows or str(r) in srows.split(",")) else 1
        for g in range(asp):
            av = f"%a{r}" if g == 0 else f"%a{r}g{g}"
            if g:   # (its own sender, an equal layout of another type: Loom keys a multicast by the sender, CSE merges equals)
                sl = next(ln for ln in reversed(L) if ln.startswith(f"  %as{r} = sender "))
                m_ = re.search(r"tile<([\dx]+)xi32(, #encoding.layout.strided<strides=\[([\d, ]+)\]>)?>>$", sl)
                dims = [int(v) for v in m_.group(1).split("x")]
                if m_.group(3):   # (replay diag: [n][rpl][rec] -> [n / 2][2][rpl][rec], both outer strides 0)
                    assert len(dims) == 3 and dims[0] % 2 == 0, sl
                    sl = sl[:m_.start()] + (f"tile<{dims[0] // 2}x2x{dims[1]}x{dims[2]}xi32, "
                                            f"#encoding.layout.strided<strides=[0, 0, {dims[2]}, 1]>>>")
                else:
                    assert len(dims) == 2 and dims[0] % 2 == 0, sl   # [n][rec] -> [n / 2][2][rec]
                    sl = sl[:m_.start()] + f"tile<{dims[0] // 2}x2x{dims[1]}xi32>>"
                e(sl.replace(f"%as{r} = ", f"%as{r}g{g} = "))
            ao = f"%ao{r}"
            if g and os.environ.get("GN_QASPLIT_COPY") == "1":   # (diag) the second copy of A (npu_gemm_check)
                ao = f"%ao{r}c"
                e(f"  {ao} = constant.u64 {r * nbr * P * MP * fa + cfg.nb * P * MP * fa} : reg<aie2p.array.offset : offset>")
            e(f"  {av} = view.sender %as{r}{f'g{g}' if g else ''}, {ao} : reg<aie2p.array.sender : tile<{art}xi32>>")
        for j in range(cols):
            c = (2 * r + j) % cols
            g = c * asp // cols
            av = f"%a{r}" if g == 0 else f"%a{r}g{g}"
            # (split: each half's stages inside its own columns, rows spread over them; 2 rows: 0, 2 / 4, 6)
            st = (g * (cols // 2) + r * (cols // 2) // R if asp > 1 else N.astage(r, cols)) if cols > 1 else 2
            if os.environ.get("GN_QASTAGE"):   # (diag) row r's A stage columns
                st = int(os.environ["GN_QASTAGE"].split(",")[r])
            if asp > 1 and os.environ.get("GN_QASPLIT_EVEN") == "1":   # (diag) stages in even columns only
                st = g * (cols // 2) + 2 * (r * (cols // 4) // R)
            if asp > 1 and os.environ.get("GN_QASPLIT_ODD") == "1":   # (diag) stages in odd columns only
                st = g * (cols // 2) + 2 * (r * (cols // 4) // R) + 1
            t = f"{c}_{r}"
            e(f"  %ra{t} = receiver {kk[c, r]}, 0 : reg<aie2p.array.receiver : tile<{art}xi32>>")
            e(f"  %cha{t} = channel {av}, %ra{t}, %n{N.acap_tail(cfg)}, %arec : reg<aie2p.array.channel : tile<{art}xi32>>")
            e(f"  constrain.leaf_sync %cha{t}")
            e(f"  constrain.stage %cha{t}, %n{st}")
    # (fuse 2) the egress ring before the panel it fills (W), the panel before the A stage (gen_npu_gemm.qf_ring_at)
    jf = len(L)
    if fz:   # per core: its egress ring (dummy write binding, never transferred) and its raw rows (rawdma: through its
        # W DMA into W slot 0); raw [N][kraw / 256][BLK] -> per column, units [slab][1024-K pass][16 rows][4 blocks]
        geo = {0: raw_geom(cfg, N.QF["cfg"])}
        if N.fpair(cfg):   # (column pairs) the up columns' format
            geo[1] = raw_geom(cfg, N.QF["cfgu"])
            assert geo[1]["fw"] == geo[0]["fw"] and geo[1]["nrec"] == geo[0]["nrec"], (geo[0], geo[1])
            assert not qsplit() and not qdec() and not N.qfraw4_on()
        cg = lambda c: geo[c % 2 if N.fpair(cfg) else 0]   # (column c's)
        q, U, BLK, RS, fw, nrec, nraw, rw, odd, r4 = (geo[0][k] for k in
                                                      ("q", "U", "BLK", "RS", "fw", "nrec", "nraw", "rw", "odd", "r4"))
        fr = 1 if N.qp1k_on() else 0   # (P1K: row 1 decodes; row 0's program holds the pair relay's copy)
        dr = R if qdec() else fr   # (the decoding worker's row: the egressing one)
        drs = tuple(range(R, R + qdec_rows())) if qdec() else (0, 1, 2, 3) if N.qf4r_on() else (0, 1) if N.qfkh_on() else (fr,)
        # (GN_QF4RSWAP=1: pair (0, 1) swaps roles, row 0 egresses, so both egress rows own their K half's first record)
        swp = N.qf4r_on() and N.qf4r_swap()
        egr = ((0, 2) if swp else (1, 2)) if N.qf4r_on() else drs if N.qfkh_on() else (dr,)   # (lane c len(egr) + i)
        h4 = ({1: 0, 3: 2} if swp else {0: 1, 3: 2}) if N.qf4r_on() else {}   # (GN_QF4R) helper row -> its egress row
        if N.qfkh_on():
            nrec //= len(egr)
        for g_ in geo.values():
            assert 4 * g_["rw"] <= N.decoder(g_["q"].fmt, g_["q"].gord).RECROW
        N.QF["rawn"], N.QF["raww"] = nraw, rw
        N.QF["rawgeo"] = {g_["q"].fmt: (g_["nraw"], g_["rw"]) for g_ in geo.values()}   # (leaves: their format's)
        if os.environ.get("GN_QDBG"):   # (diag) the raw layout
            import sys
            print(f"raw layout: fmt {cfg.fmt} nraw {nraw} rw {rw} wslot {N.QF['wslot']} fw {fw} nrec {nrec} "
                  f"recrow {N.decoder(cfg.fmt, cfg.gord).RECROW}", file=open(os.environ["GN_QDBG"], "a"))
        dp, sp = N.QF_PORTS
        e(f'  %fdb = binding {6 if cfg.gate else 3}, "write"')   # (gated: the engine's binding order)
        e(f'  %frb = binding {7 if cfg.gate else 4}, "read"')
        if N.fpair(cfg):   # (column pairs) the up tensor's rows
            e(f'  %frbu = binding {8 if cfg.gate else 5}, "read"')
        # (GN_QFLANE2) a second egress lane per egress row: its helper's records (GN_QFLANE2COLS: one side's columns)
        l2c = lambda c: N.qflane2_side("e" if N.fpair(cfg) and c % 2 else "w")
        l2any, l2all = any(l2c(c) for c in range(cols)), all(l2c(c) for c in range(cols))
        nl_ = cols * len(egr)
        e(f"  %fdl = constant.u32 {nl_} : reg<aie2p.array.scalar : index>")
        for i_, g_ in geo.items():
            e(f"  %fun{'u' if i_ else ''} = constant.u32 {g_['nraw']} : reg<aie2p.array.scalar : index>")
        eg2 = N.QF_P1KD["eb"] // N.QF_P1KD["rec"] if N.qfdirect_on() else 1   # (GN_QFEGR2: slab-sized egress records)
        if not l2all:   # (one-lane columns: every record on the first lane)
            e(f"  %fdn = constant.u32 {nrec // eg2} : reg<aie2p.array.scalar : index>")
            e(f"  %fd_all = receiver %fdb, 0 : reg<aie2p.array.receiver : tile<{nl_}x{nrec // eg2}x{fw * eg2}xi32>>")
        if l2any:   # (two-lane columns: half the records on each; the second lane's view: slab records)
            e(f"  %fdn{'' if l2all else '2'} = constant.u32 {nrec // 2} : reg<aie2p.array.scalar : index>")
            e(f"  %fd_all{'' if l2all else '2'} = receiver %fdb, 0 : "
              f"reg<aie2p.array.receiver : tile<{nl_}x{nrec // 2}x{fw}xi32>>")
            e(f"  %fdb_all = receiver %fdb, 0 : reg<aie2p.array.receiver : tile<{nl_}x{nrec // 4}x{2 * fw}xi32>>")
        # records [super-block][slab][16 rows][one block] (super-blocks are BLK apart in a row)
        if qsplit() or qdec():   # (a step's 4 slabs side by side in each row: the decoder's 576 B record row)
            rty = f"16x{NP}x{rw}"
            e(f"  %fru_all = sender %frb, 0 : reg<aie2p.array.sender : tile<{U * N.FUSE_SB}x16x{NP}x{rw}xi32, "
              f"#encoding.layout.strided<strides=[{BLK // 4}, {RS}, {16 * RS}, 1]>>>")
            e(f"  %fsn = constant.u32 {U * N.FUSE_SB} : reg<aie2p.array.scalar : index>")
        else:
            rty = f"16x{rw}"
            for i_, g_ in geo.items():   # (column pairs: the up tensor's sender too, its own format's shape)
                s_, b_ = ("%fru_allu", "%frbu") if i_ else ("%fru_all", "%frb")
                U_, BLK_, RS_, rw_ = g_["U"], g_["BLK"], g_["RS"], g_["rw"]
                # (GN_QRAWCONTIG=1, diag, wrong output: each record's 16 rows read contiguously, the same bytes a record)
                rs_ = rw_ if os.environ.get("GN_QRAWCONTIG") == "1" else RS_
                if cfg.qsw and qswdiag() != "convert" and not N.fpair(cfg):   # (FFN gate / up) per step: the
                    # column's 2 gate slabs, then the same up rows dgu bytes on
                    # (not odd: the pair axis would need the descriptor chain that the gate-to-up stride takes)
                    assert NP in (4, 5) and not r4 and cfg.dgu % 4 == 0 and not odd
                    if qsw5_on(cfg):   # (qsw5: per step slab b = the column's gate run b (8 rows, runs 256 rows
                        # apart), then up's; a record [gate / up][8 rows])
                        rty = f"2x8x{rw_}"
                        e(f"  {s_} = sender {b_}, 0 : reg<aie2p.array.sender : tile<{U_ * N.FUSE_SB}x5x2x8x{rw_}xi32, "
                          f"#encoding.layout.strided<strides=[{BLK_ // 4}, {256 * rs_}, {cfg.dgu // 4}, {rs_}, 1]>>>")
                    else:
                        e(f"  {s_} = sender {b_}, 0 : reg<aie2p.array.sender : tile<{U_ * N.FUSE_SB}x2x2x16x{rw_}xi32, "
                          f"#encoding.layout.strided<strides=[{BLK_ // 4}, {cfg.dgu // 4}, {16 * rs_}, {rs_}, 1]>>>")
                elif g_["odd"]:   # (pairs of super-blocks: the second's read starts 2 bytes before it)
                    e(f"  {s_} = sender {b_}, 0 : reg<aie2p.array.sender : tile<{U_ * N.FUSE_SB // 2}x2x{NP}x16x{rw_}xi32, "
                      f"#encoding.layout.strided<strides=[{BLK_ // 2}, {BLK_ // 4}, {16 * rs_}, {rs_}, 1]>>>")
                else:
                    e(f"  {s_} = sender {b_}, 0 : reg<aie2p.array.sender : tile<{U_ if r4 else U_ * N.FUSE_SB}x{NP}x16x{rw_}xi32, "
                      f"#encoding.layout.strided<strides=[{(N.FUSE_SB if r4 else 1) * BLK_ // 4}, {16 * rs_}, {rs_}, 1]>>>")
        for c in range(cols):   # (qsw: column c's gate rows swcol c ..; pairs: column c's 64 rows of its tensor
            # from row 64 (c // 2))
            if N.fpair(cfg):
                fo_, fs_ = (c // 2) * N.TN * cg(c)["RS"] * 4, "%fru_allu" if c % 2 else "%fru_all"
                rty = f"16x{cg(c)['rw']}"
            else:
                fo_, fs_ = c * (cfg.swcol if cfg.qsw and qswdiag() != 'convert' else N.TN) * RS * 4, "%fru_all"
                if qsw5_on(cfg) and qswdiag() != "convert":   # (qsw5: column c's runs at 8 P8[c] rows)
                    fo_ = 8 * QSW5_P8[c] * RS * 4
            e(f"  %fro{c} = constant.u64 {fo_} : reg<aie2p.array.offset : offset>")
            e(f"  %fru{c} = view.sender {fs_}, %fro{c} : reg<aie2p.array.sender : tile<{rty}xi32>>")
            for r in drs:   # (row 0 fills; the rows below see its raw rows as W records: gen_npu_gemm.qf_skip_raw)
                t = f"{c}_{r}"
                if r in egr:   # (the egressing decoders; raw multicast to every decoder row)
                    l2 = l2c(c)
                    sfx = "2" if l2 and not l2all else ""   # (this column's first-lane view and count)
                    e(f"  %fdlane{t} = constant.u32 {c * len(egr) + egr.index(r)} : reg<aie2p.array.scalar : index>")
                    e(f"  %fdr{t} = partition.receiver %fd_all{sfx}, %origin, %fdlane{t}, %fdl : reg<aie2p.array.receiver : tile<{fw * eg2}xi32>>")
                    e(f"  %fds{t} = sender {kk[c, r]}, {dp} : reg<aie2p.array.sender : tile<{fw * eg2}xi32>>")
                    e(f"  %chd{t} = channel %fds{t}, %fdr{t}, {'%n4' if N.qfdirect_on() and not l2 else '%two'}, %fdn{sfx} : "
                      f"reg<aie2p.array.channel : tile<{fw * eg2}xi32>>")
                    e(f"  constrain.leaf_sync %chd{t}")
                    if l2:   # (its helper's records: ring slots 2-3, on its C channel's DMA)
                        e(f"  %fdlaneb{t} = constant.u32 {c * len(egr) + egr.index(r)} : "
                          "reg<aie2p.array.scalar : index>")
                        # (one slot of a slab, 2 records: gen_npu_gemm.qf_ring_at pins it after the first ring)
                        e(f"  %fdnb{t} = constant.u32 {nrec // 4} : reg<aie2p.array.scalar : index>")
                        e(f"  %fdrb{t} = partition.receiver %fdb_all, %origin, %fdlaneb{t}, %fdl : "
                          f"reg<aie2p.array.receiver : tile<{2 * fw}xi32>>")
                        e(f"  %fdsb{t} = sender {kk[c, r]}, {N.QF_LANE2_PORT} : reg<aie2p.array.sender : tile<{2 * fw}xi32>>")
                        e(f"  %chdb{t} = channel %fdsb{t}, %fdrb{t}, %one, %fdnb{t} : "
                          f"reg<aie2p.array.channel : tile<{2 * fw}xi32>>")
                        e(f"  constrain.leaf_sync %chdb{t}")
                elif r in h4:   # (GN_QF4R) the helper's records to its egress neighbour (its window reads them)
                    e(f"  %fhn{t} = constant.u32 {nrec // 2} : reg<aie2p.array.scalar : index>")
                    e(f"  %fds{t} = sender {kk[c, r]}, {dp} : reg<aie2p.array.sender : tile<{fw}xi32>>")
                    e(f"  %fhr{t} = receiver {kk[c, h4[r]]}, {N.QF4R_NB} : reg<aie2p.array.receiver : tile<{fw}xi32>>")
                    e(f"  %chd{t} = channel %fds{t}, %fhr{t}, %n{N.qf4r_hslots()}, %fhn{t} : "
                      f"reg<aie2p.array.channel : tile<{fw}xi32>>")
                    e(f"  constrain.leaf_sync %chd{t}")
                e(f"  %frr{t} = receiver {kk[c, r]}, {sp} : reg<aie2p.array.receiver : tile<{rty}xi32>>")
                e(f"  %chr{t} = channel %fru{c}, %frr{t}, {'%n3' if qdec() else '%two' if N.qfdirect_on() and N.QF_P1KD['raw'] == 2 else '%one'}, "
                  f"{'%fsn' if qsplit() or qdec() else '%funu' if N.fpair(cfg) and c % 2 else '%fun'} : "
                  f"reg<aie2p.array.channel : tile<{rty}xi32>>")
                if not qdec() or os.environ.get("GN_QDSTAGE", "1") == "1":   # (decoder rows: staged too, else shim to core)
                    e(f"  constrain.stage %chr{t}, %n{c if cols > 1 else 1}")
                e(f"  constrain.leaf_sync %chr{t}")
            if qdec_rows() == 2:   # the second decoder's slabs 2-3 of a step (one record) to the first
                rq = 4 * N.decoder(cfg.fmt, cfg.gord).OUT_B // 4
                e(f"  %qds{c} = sender {kk[c, R + 1]}, {N.QDEC_SPORT} : reg<aie2p.array.sender : tile<{rq}xi32>>")
                e(f"  %qdr{c} = receiver {kk[c, R]}, {N.QDEC_RPORT} : reg<aie2p.array.receiver : tile<{rq}xi32>>")
                e(f"  %chqd{c} = channel %qds{c}, %qdr{c}, %one, %fsn : reg<aie2p.array.channel : tile<{rq}xi32>>")
                e(f"  constrain.leaf_sync %chqd{c}")
    jt = len(L)
    if fz:   # filled by the core's egress; its raw rows land in its W ring's first slot
        L[iw:jt] = L[jf:jt] + L[iw:ia] + L[ia:jf]
        for c in range(cols):
            if N.qf4r_on():   # (half 0's panel (rows 0 / 2) by row 2, half 1's (rows 1 / 3) by row 1)
                e(f"  constrain.fill %chw{c}_0, %chd{c}_2")
                e(f"  constrain.fill %chw{c}_1, %chd{c}_{0 if N.qf4r_swap() else 1}")
                if l2c(c):   # (the second writer: the second egress lane)
                    e(f"  constrain.fill %chw{c}_0, %chdb{c}_2")
                    e(f"  constrain.fill %chw{c}_1, %chdb{c}_0")
            elif N.qfkh_on():   # (a panel per K half, each filled by its decoder: rows 0 / 2 by row 0, 1 / 3 by row 1)
                e(f"  constrain.fill %chw{c}_0, %chd{c}_0")
                e(f"  constrain.fill %chw{c}_1, %chd{c}_1")
            else:
                e(f"  constrain.fill %chw{c}_0, %chd{c}_{dr}")   # (the shared panel's owner: row 0's member)
            for r in (() if qdec() else drs):
                t = f"{c}_{r}"
                e(f"  %fioff{t} = constant.u32 0 : reg<aie2p.array.scalar : index>")
                e(f"  %firows{t} = constant.u32 16 : reg<aie2p.array.scalar : index>")
                q_ = cg(c)["q"]
                e(f"  %fipitch{t} = constant.u32 {N.decoder(q_.fmt, q_.gord).RECROW} : reg<aie2p.array.scalar : index>")
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
        if not N.notick():   # (the tick channel: one fresh flag read per tick)
            e(f"  %gt_all = receiver %tb, 0 : reg<aie2p.array.receiver : tile<1x{S}x1x4xi32, #encoding.layout.strided<strides=[4, 0, 0, 1]>>>")
            e("  %gt = partition.receiver %gt_all, %origin, %n0, %one : reg<aie2p.array.receiver : tile<4xi32>>")
            e(f"  %gtick = sender {kw}, {p['waiter']['tick']} : reg<aie2p.array.sender : tile<4xi32>>")
            e("  %chgt = channel %gtick, %gt, %two, %gsup : reg<aie2p.array.channel : tile<4xi32>>")
            e("  constrain.core_stream %chgt")
            e("  constrain.request %chgp, %chgt")
        else:   # (a free-running poll, queued at the gate: Loom's constrain.request %p, %p, HRX pin)
            e("  constrain.request %chgp, %chgp")
        e(f"  %gn1s = sender {kw}, {p['waiter']['nbr']} : reg<aie2p.array.sender : tile<4xi32>>")
        e(f"  %gn1r = receiver {kr}, {p['relay']['nbr']} : reg<aie2p.array.receiver : tile<4xi32>>")
        e("  %chgn = channel %gn1s, %gn1r, %one, %one : reg<aie2p.array.channel : tile<4xi32>>")
        e("  constrain.leaf_sync %chgn")
        e("  %gs_all = receiver %sb, 0 : reg<aie2p.array.receiver : tile<1x2x4xi32>>")
        e("  %gs = partition.receiver %gs_all, %origin, %n0, %one : reg<aie2p.array.receiver : tile<4xi32>>")
        # (GN_QNOTICK: the waiter sends go / done itself on its free stream port; the relay only counts)
        gsig = f"{kw}, {p['waiter']['tick']}" if N.notick() else f"{kr}, {p['relay']['sig']}"
        e(f"  %gsig = sender {gsig} : reg<aie2p.array.sender : tile<4xi32>>")
        e("  %chgs = channel %gsig, %gs, %two, %two : reg<aie2p.array.channel : tile<4xi32>>")
        e("  constrain.core_stream %chgs")
        e("  constrain.gate %chgs")
        e("  constrain.signal %chgs")
    if fz and qsplit():   # row 1 -> row 0: a step's slab-3 output (slab 2: row 1's C ring), one record a step
        rq = 2 * N.decoder(cfg.fmt, cfg.gord).OUT_B // 4
        e(f"  %qsn = constant.u32 {N.QF['cfg'].passes * N.FUSE_SB} : reg<aie2p.array.scalar : index>")
        for c in range(cols):
            g = "waiter" if cfg.gate and c == gw[0] else "relay" if cfg.gate and c == gr[0] else None
            e(f"  %qss{c} = sender {kk[c, 1]}, 3 : reg<aie2p.array.sender : tile<{rq}xi32>>")
            e(f"  %qsr{c} = receiver {kk[c, 0]}, {qsplit_port(g)} : reg<aie2p.array.receiver : tile<{rq}xi32>>")
            e(f"  %chq{c} = channel %qss{c}, %qsr{c}, %one, %qsn : reg<aie2p.array.channel : tile<{rq}xi32>>")
            e(f"  constrain.leaf_sync %chq{c}")
    if os.environ.get("GN_QMB1") == "5":   # (diag MB2) one channel per column, row 0 -> row 1 (Loom: no channel cycles):
        # its credit / ready locks order both memories' hand-overs, since each core alternates own / other half in a fixed
        # order. Row 0 (sender, port 3): a ready pre-released at the firing start, then acquire a credit before and release
        # a ready after each half (the last half's release skipped); row 1 (receiver, port 4): acquire a ready before and
        # release a credit after each half. Per firing 2 MP passes of each.
        assert R % 2 == 0 and (not fz or N.qp1k_on()) and not crel()   # (one channel per vertical pair)
        e(f"  %mbn = constant.u32 {nbr * P * MP * 2} : reg<aie2p.array.scalar : index>")
        for c in range(cols):
            for q in range(0, R, 2):
                t = f"{c}" if q == 0 else f"{c}_{q}"
                e(f"  %mbs{t} = sender {kk[c, q]}, 3 : reg<aie2p.array.sender : tile<16xi32>>")
                e(f"  %mbr{t} = receiver {kk[c, q + 1]}, 4 : reg<aie2p.array.receiver : tile<16xi32>>")
                e(f"  %chm{t} = channel %mbs{t}, %mbr{t}, %one, %mbn : reg<aie2p.array.channel : tile<16xi32>>")
                e(f"  constrain.leaf_sync %chm{t}")
    if N.pshare():   # (Loom numbers the channels in program order) the tick and the signal on their column's row 0 egress lane
        chans = [x.split(" = channel ")[0].strip() for x in "\n".join(L).splitlines() if " = channel " in x]
        gp = N.QF["gports"]
        er = 2 if N.qf4r_on() else 0   # (the column's egress lane nearest the gate row)
        N.QF["pshare"] = ",".join(f"{chans.index(c)}:{chans.index(f'%chd{col}_{er}')}" for c, col in
                                  (("%chgt", gp["col"]), ("%chgs", gp["relay"]["col"])))
    if N.fpair(cfg) and not N.qpair_off():   # (column pairs) qpair_swap's rings: core (2 u, r) to its east neighbour, one 128 B slot (not 64: P1K's 64 B pin is the MB2 tokens', forced, the rings would share it), two records
        # a converted row (after pshare's numbering: Loom numbers channels in program order)
        assert cols % 2 == 0 and cfg.qsw
        e(f"  %nqx = constant.u32 {2 * nbr * MP} : reg<aie2p.array.scalar : index>")
        for u in range(cols // 2):
            for r in range(R):
                t = f"{u}_{r}"
                e(f"  %qxs{t} = sender {kk[2 * u, r]}, {N.QPAIR_PORT} : reg<aie2p.array.sender : tile<32xi32>>")
                e(f"  %qxr{t} = receiver {kk[2 * u + 1, r]}, {N.QPAIR_PORT} : reg<aie2p.array.receiver : tile<32xi32>>")
                e(f"  %chqx{t} = channel %qxs{t}, %qxr{t}, %one, %nqx : reg<aie2p.array.channel : tile<32xi32>>")
                e(f"  constrain.leaf_sync %chqx{t}")
    e("  return\n}\n")


def mrel():
    """The pair relay (NPU_QCREL=2) with the egress merge (GN_QEMERGE=1): each column's memory tile drains its pairs'
    C streams a block in turn, so row 2 p + j owns M blocks 4 f + 2 j + p (a strided A stream) and C is in token order."""
    return N.crelp_on() and os.environ.get("GN_QEMERGE") == "1"


def qgate(cfg):
    """(gated) the waiter and relay cores, (column 1, the last row) and (column 2, the last row), and their ports past
    the leaf's (A 0, W 1, C 2; a filler's egress 3, raw 4): gen_npu_gemm.gate_ports reads QF["gports"]."""
    if not cfg.gate:
        N.QF["gports"] = None
        return None, None
    assert cfg.cols >= 3
    r = gate_row()
    b = 5 if (r == 0 and cfg.fuse == 2) or os.environ.get("GN_QMB1") == "5" else 3   # (MB2: row q + 1 receives on 4)
    if cfg.fuse == 2 and N.qf4r_on():   # (GN_QF4R: the gate row decodes too: 6 its helper sender, 7 its raw intake)
        b = 9
    cw, cr = gate_cols(cfg)
    assert cfg.cols > cr
    N.QF["gports"] = {"col": cw, "waiter": dict(row=r, poll=b, tick=b + 1, nbr=b + 2),
                      "relay": dict(col=cr, row=r, nbr=b, sig=b + 1)}
    return (cw, r), (cr, r)


def gate_cols(cfg):
    """(gated) the waiter's and the relay's columns, east neighbours (their channel is a shared-memory hop; 2 and 4 run
    out of DMA channels): 1 and 2; column pairs: 2 and 3, the waiter on a gate column (an up column's waiter with
    the larger decoder overflows the 16 KB program memory: a Q5_K up's, 16416 B)."""
    if N.fpair(cfg) and os.environ.get("GN_QGCOLS"):   # (diag) the column pairs' waiter, relay columns
        return tuple(int(x) for x in os.environ["GN_QGCOLS"].split(","))
    return (2, 3) if N.fpair(cfg) else (1, 2)


def gate_row():
    """The gate cores' row: the last (crel: row 0, whose C stream ends the job)."""
    return 0 if crel() else QROWS - 1


def leaf_name(cfg, c, r):
    """Worker (c, r)'s leaf: tail (fuse 2: row 0 decodes the panel), tailr (the rows below), _gw / _gr: the gate's;
    column pairs (Config.ufmt): _pw the gate column's (2 u), _pe the up column's (2 u + 1, ufmt's decoder)."""
    n = leaf_name1(cfg, c, r)
    return n + ("_pe" if c % 2 else "_pw") if N.fpair(cfg) and not n.startswith("qdec") else n


def leaf_name1(cfg, c, r):
    if qdec() and r >= QROWS:
        return "qdec" if qdec_rows() == 1 else f"qdec{r - QROWS}"
    name = "tailr" if cfg.fuse == 2 and r else "tail"
    pr = r % 2 if os.environ.get("GN_QMB1") == "5" and QROWS % 2 == 0 else r   # (MB2 / P1: each vertical pair)
    if cfg.fuse == 2 and N.qp1k_on():   # (P1K fill) row 1 decodes; row 3 shares its W stream: it skips the raw records
        # (s); rows 0, 2 read the other half's stream (GN_QP1RAW1=1: the raw reaches them too)
        name = ("tailm" if r == 1 else "tailms") if pr == 1 else (
            "tailm0s" if os.environ.get("GN_QP1RAW1") == "1" else "tailm0")
        if N.qfkh_on() and pr == 0:   # (GN_QFKH) row 0 decodes K half 0 (an intake needs its W stream's source member); row 2 skips the raw
            name = "tailm0d" if r == 0 else "tailm0e" if N.qf4r_on() else "tailm0s"
        if N.qf4r_on() and r == 3:   # (GN_QF4R) row 3 decodes too
            name = "tailmh"
        return name + ("_gw" if cfg.gate and (c, r) == (gate_cols(cfg)[0], gate_row()) else
                       "_gr" if cfg.gate and (c, r) == (gate_cols(cfg)[1], gate_row()) else "")
    if cfg.fuse == 0 and pr == 1 and os.environ.get("GN_QMB1"):   # (diag: row 1 reads row 0's W, gen_npu_gemm)
        name = "tailm"
    if cfg.fuse == 0 and pr == 0 and os.environ.get("GN_QMB1") in ("2", "3", "4", "5"):   # (diag: row 0 reads W from row 1)
        name = "tailm0"
    if r and crel() and r < QROWS - 1:   # (a middle row of the relay chain: tailr<r>)
        name += str(r)
    if cfg.gate and (c, r) == (gate_cols(cfg)[0], gate_row()):
        return name + "_gw"
    if cfg.gate and (c, r) == (gate_cols(cfg)[1], gate_row()):
        return name + "_gr"
    return name


def leaves(cfg):
    """[(name, gate role)] of the image's distinct leaves."""
    out = {}
    for c in range(cfg.cols):
        for r in range(QROWS + qdec_rows()):
            n = leaf_name(cfg, c, r)
            b = n[:-3] if N.fpair(cfg) and n.endswith(("_pw", "_pe")) else n
            out[n] = "waiter" if b.endswith("_gw") else "relay" if b.endswith("_gr") else None
    return list(out.items())


def raw_geom(cfg, q):
    """(fuse 2) the raw rows' geometry for the decode side's Config q (its format): super-blocks U a row's passes, BLK
    bytes, RS row words; the egress record fw words, nrec of them; nraw raw records a call of rw words (a block's;
    with rows below the filler, padded so the raw records, multicast with the panel, are whole W records to them:
    qf_skip_raw); odd (a super-block of 2 mod 4 bytes: read in pairs, 2 bytes of slop before the odd ones)."""
    NP, R = N.NP, QROWS
    U, BLK = q.passes, N.fuse_blk(q)
    RS = (cfg.kraw or 8 * cfg.passes * cfg.ks[0]) // 256 * BLK // 4
    fw, nrec = N.fuse_record(q, (0,))
    r4 = N.qfraw4_on()   # (GN_QFRAW4: a record's rows hold the step's 4 super-blocks)
    nraw, rw = (U * NP, N.FUSE_SB * BLK // 4) if r4 else (U * N.FUSE_SB * NP, BLK // 4)
    odd = N.qf_odd2(q)
    if odd:
        assert not r4 and U * N.FUSE_SB % 2 == 0, (q.fmt, U)
        rw = (BLK + 2 + 3) // 4
    while R > 1 and nraw * 64 * rw % N.QF["wslot"]:
        rw += 1
    if qdec():   # (the split's record: a step's NP slabs side by side, 144 B each: the decoder's 576 B row)
        rw = N.decoder(q.fmt, q.gord).RECROW // (4 * NP)
    return dict(q=q, U=U, BLK=BLK, RS=RS, fw=fw, nrec=nrec, nraw=nraw, rw=rw, odd=odd, r4=r4)


def qswdiag():
    """(diag, Config.qsw) GN_QSWDIAG=intake: only the gate / up intake (plain bf16 C, [gate 32][up 32] per column);
    convert: only the swiglu epilogue and its H records (the plain intake)."""
    return os.environ.get("GN_QSWDIAG", "")


def p1k4(cfg):
    """The P1K direct fill at NP 4 (gen_npu_gemm.p1k4_on, before its QF cfg is set)."""
    return cfg.fuse == 2 and N.qp1k_on() and N.NP == 4 and os.environ.get("GN_QFDIRECT", "1") == "1"


def qgmap_pins(cfg):
    """(gen_npu_gemm.qgmap_on) RODATA_AT pins: per decoder (a pair's two formats sit in different cores, so each
    format's tables are packed alone) the largest table at 49152, the rest first-fit in QG_SPARE's runs; then @silu_k
    first-fit in what every format leaves free (it is in every core). The rest of the runs: Loom's stack."""
    def syms(cc):
        out = []
        for ln in N.decoder(cc.fmt, cc.gord).inline_rodata(N.dec_tag(cc)):
            m = re.match(r"global.rodata.def @(\w+) = align\((\d+)\).*bytes\(\"([0-9a-fA-F]*)\"\)", ln)
            out.append((m[1], int(m[2]), len(m[3]) // 2))
        return sorted(out, key=lambda t: -t[2])

    def fit(free, al, n):   # first fit in [(lo, hi)]: (address, the free list without it)
        for i, (lo, hi) in enumerate(free):
            a = -(-lo // al) * al
            if a + n <= hi:
                return a, free[:i] + [(lo, a), (a + n, hi)] + free[i + 1:]
        return None, free

    pins, frees = [], []
    for cc in [cfg] + ([N.colcfg(cfg, 1)] if N.fpair(cfg) and cfg.ufmt != cfg.fmt else []):
        if not N.decoder(cc.fmt, cc.gord).GRID:
            frees.append(list(N.QG_SPARE))
            continue
        free = list(N.QG_SPARE)
        for i, (nm, al, n) in enumerate(syms(cc)):
            if i == 0:   # (the main table at the page start: its gathers take the page from the address's high bytes)
                assert 49152 % al == 0 and 49152 + n <= 53760, (cc.fmt, nm, n)
                a, free = 49152, [(49152 + n, 53760)] + free[1:]
            else:
                a, free = fit(free, al, n)
                assert a is not None, (cc.fmt, nm, n, free)
            pins.append(f"{nm}@{a}")
        frees.append([r for r in free if r[1] > r[0]])
    nk = 64 * len(N.silu_kvals())
    cand = sorted({-(-lo // 64) * 64 for fr in frees for lo, _ in fr})
    ka = next((a for a in cand if all(any(lo <= a and a + nk <= hi for lo, hi in fr) for fr in frees)), None)
    assert ka is not None, frees
    return pins + [f"silu_k@{ka}"]


# (qsw5) the NP 5 FFN gate / up map's SiLU table: the bound at 87.5 (gen_npu_gemm.set_sl_hi: 115 blocks), in the
# runs the NP 5 site map leaves free (past the even C area's last row, past the private storage, below the token ring,
# A slot 0's end), its 9 vectors (@silu_k) at 25280 .. 25856, below the decoder's pad (26112). Not 11520 .. 12288: the
# fill's second raw intake slot (9216 .. 18432) covers it. Loom's stack (128 B) goes to 65280: the fill's rings end
# bank 2 (gen_npu_gemm QSW5_RING), where Loom's cursor would put it inside the even C area's last row.
QSW5_SILU = ((64, 49664), (26, 31104), (22, 29312), (3, 25088))
QSW5_SILUK = 25280
QSW5_PRIVATE = 30784   # (Loom's private storage: past the 64 B token ring at 30720)
QSW5_P8 = (0, 2, 1, 3, 4, 6, 5, 7)   # (P4: column c computes k-block slot c's k-block, down's decode order)


def qsw5_on(cfg):
    """The NP 5 FFN gate / up image: column c of call j takes the 8 features of k-block 8 (j % 4) + QSW5_P8[c] of
    each of 5 passes 5 (j / 4) + b (C slab b: gate block b's 8 rows, then up's), its H records [b][token half]
    at slot c of those passes (H [M block][hpass passes][16-token row][4608 B], down's A)."""
    return bool(cfg.qsw) and N.NP == 5


def grid_flags(cfg):
    """gen_npu_gemm's Q' decoder flags: the byte grid (QF qg8) and, for a grid format at fuse 2, the P1K grid map."""
    q5 = qsw5_on(cfg) and qswdiag() != "intake"
    N.set_sl_hi(0x42AF if q5 else None)
    N.QF["silu"], N.QF["siluk"] = (QSW5_SILU, QSW5_SILUK) if q5 else (None, None)
    N.QF["qg8"] = True
    N.QF["grid"] = cfg.fuse == 2 and N.qp1k_on() and any(
        N.decoder(f, g).GRID for f, g in [(cfg.fmt, cfg.gord)] + ([(cfg.ufmt, cfg.ugord or "P4")] if N.fpair(cfg) else []))


def gen(cfg):
    assert len(cfg.ks) == 1 and cfg.fuse in (0, 2) and cfg.passes >= 2 and 1 <= cfg.cols <= 8
    grid_flags(cfg)
    N.QF["gfmt"] = cfg.fmt   # (gen_npu_gemm.dec_tag)
    L = []
    if cfg.fuse == 2:   # the decode side: 1024-K passes over the call's K, one 128-k-block slice
        K = 8 * cfg.passes * cfg.ks[0]
        assert K % 1024 == 0 and 32 % cfg.ks[0] == 0, "fuse 2: K a multiple of 1024, ks dividing a super-block"
        N.QF["cfg"] = N.Config(cfg.cols, cfg.nb, (128,), K // 1024, mu=1, fuse=2, kraw=cfg.kraw, fmt=cfg.fmt, rawdma=1,
                              gord=cfg.gord)
        N.QF["cfgu"] = N.Config(cfg.cols, cfg.nb, (128,), K // 1024, mu=1, fuse=2, kraw=cfg.kraw, fmt=cfg.ufmt,
                                rawdma=1, gord=cfg.ugord or "P4") if N.fpair(cfg) else None   # (column pairs)
        N.QF["wslot"] = N.NP * N.slab(cfg.ks[0]) // (2 if N.qp1k_on() else 1)   # (P1K: a member's W record, one half)
        N.QF_PORTS = (6, 7) if N.qp1k_on() else (3, 4)   # (P1K: 3 / 4 the pair's tokens, 5 the pair C relay)
    else:
        N.QF["cfg"] = None
    N.QF["rows"] = QROWS
    N.QF["crel"] = crel_port if crel() else None
    N.QF["crelmid"] = crel_mid_port() if crel() else None
    N.QF["split"] = qsplit_port if qsplit() else None
    N.QF["dec"] = qdec()
    N.QF["qsw"] = cfg.qsw and qswdiag() != "intake"
    array_program(L, cfg)
    if N.QF["qsw"]:
        L.append(N.silu_rodata())
    if cfg.fuse == 2:   # the inlined decoder's tables (grid formats; column pairs: the up format's too)
        tags = {}
        for cc in [cfg] + ([N.colcfg(cfg, 1)] if N.fpair(cfg) else []):
            tags.setdefault(N.dec_tag(cc), cc)   # (a pair of one format: one set, "inlu")
        for t_, cc in tags.items():
            L.extend(N.decoder(cc.fmt, cc.gord, cc.swiglu).inline_rodata(t_))
    if p1k4(cfg):   # (NP 4) the scratch pad's guard (gen_npu_gemm P1K4_GUARD)
        L.append(f'global.rodata.def @qfgd = align(64) bytes("{"00" * 64}")')
    i0 = len(L)
    for name, gate in leaves(cfg):   # (tailr: the GEMM leaf without the fill, skipping the raw rows)
        if name.startswith("qdec"):
            N.qdec_leaf(L, cfg, None if name == "qdec" else int(name[4:]))
            continue
        pair = {"_pw": "w", "_pe": "e"}.get(name[-3:]) if N.fpair(cfg) else None
        b = name[:-3] if pair else name
        fill = cfg.fuse == 2 and not b.startswith("tailr") and not qdec() and (
            b == "tailm" or b.startswith(("tailm0d", "tailm0e", "tailmh")) or not N.qp1k_on())
        cl, saved = cfg, None
        if pair == "e":   # (the up column: its format's decoder, decode config and raw geometry)
            cl = N.colcfg(cfg, 1)
            saved = {k: N.QF[k] for k in ("cfg", "rawn", "raww")}
            N.QF["cfg"] = N.QF["cfgu"]
            N.QF["rawn"], N.QF["raww"] = N.QF["rawgeo"][cl.fmt]
        N.leaf(L, cl, "tail", cfg.ks[0], cov=(0,) if fill else None, gate=gate, name=name, pair=pair)
        if saved:
            N.QF.update(saved)
    if qfix():   # (any acquire / release of the dropped ports: the unrolled copies' renamed operands too)
        import re
        ports = "".join({"a": "0", "w": "1"}[c] for c in qfix())
        pat = re.compile(rf"^  (acq|rel)(\.cond)? %\w+(, %\w+)?, [{ports}]$")
        L[i0:] = [ln for ln in "\n".join(L[i0:]).split("\n") if not pat.match(ln)]
    if N.QSWAP:   # (row 5 role swap) every MMA's operands the other way round: C blocks [token][feature] (gen_npu_gemm)
        import re
        assert cfg.fuse == 0
        return re.sub(r"(= mma\.bfp16ebs8\.m8n8k8 [^,]+), ([^,]+), ([^,]+), ", r"\1, \3, \2, ", "\n".join(L))
    return "\n".join(L)

