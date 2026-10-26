#!/usr/bin/env python3
"""Tile GEMM yah_iu4_gemm[_kres|_kqg|_gu]: MQ4G256 weights x int4 activations (the A4 path) on v_wmma_i32_16x16x16_iu4.

Weights: MQ4G256 rows (136 B per 256 K: f16 scale0, zero0, scale1, zero1, 128 B of nibbles), the loader's signed codes, so
w = code * scale per 128 K. Activations: gen_rot.gen_q's int4 rows (aq, [tokens][K / 2] bytes) and f32 scales times 2^26
(ad, one per 128 values of each token).
Workgroup: BM rows x BN tokens over WM x WN wave32 waves, TM x TN each. Per 128-K phase the workgroup stages the weight
nibbles and scales and the activation nibbles and scales in LDS (double-buffered, one barrier per phase); each wave
runs 8 k steps of i32 WMMAs into accumulators seeded with the bits of 1/(2 pi), then folds them into its f32
accumulators: acc += (f32 bits - 1/(2 pi)) * (weight scale * activation scale * 2^26), the same values as f32(c) * the
scales.
The iu4 C layout (wave32): element i of lane l is row 2 i + l / 16, token l % 16, so the weight scale is a per-element
vector (LDS holds each 16-row tile's scales in that order) and the activation scale is one value per lane.
Epilogues: gen_gemm_tile.lds_epilogue (kstore, kres, kqg); the 48-row tile (TM 16) stores its f32 fragments; gu pairs
gate and up rows through LDS (gu_epilogue).
Bindings: weight, aq, ad, then kstore: output; kres: resid, output; kqg: output (q), gate_out; gu: up, output (f16).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_gemm_tile as TG  # noqa: E402

KP = 128        # K per phase: one weight scale group
WP = 20         # LDS dwords per staged row: 16 of nibbles + 4 of pad (16-byte aligned rows)
V8 = "vector<8xf32>"
VI = "vector<8xi32>"
V2 = "vector<2xi32>"
# gu silu's 1 / (1 + exp(-g)) as a bare v_rcp_f32 (1 ulp, no Newton step / inf select): an A4 redraw (kld_ref in the
# redraw spread), gu -0.2%
GU_BARE_RCP = True
# gu: a fence between the A and B fragment loads of a k step: allocation then places the bank-pinned A fragments first
# (all WMMAs 16 cyc, 237 -> 213 VGPR), gu -0.5%; kres loses with it (out +3.6%)
GU_FRAG_FENCE = True
# staged rows r and r + 8 share LDS banks at pitch WP (20 r mod 32 repeats every 8 rows), so a half-wave's fragment
# ds_load_b64 of 16 rows takes 2 rounds: rows with bit 3 ^ bit 0 set store each 16-byte chunk's two 8-byte k steps
# swapped (two b64 staging stores) and read k step st at slot st ^ 1 (a second lane base for odd k steps). Bit 0 in the
# swizzle spreads a staging wave's 8 rows over both halves of every bank group (bit 3 alone: stores 3-way conflicted).
# All iu4 GEMMs bit-identical, pp2048 GEMM cycles -1.1% (gu -1.1%, kres_down -1.2%).
LDS_SWZ = True
# gu epilogue: -log2e and 1.0 as lane * 0 + c (VGPRs): as a literal / inline source the silu's mul and add never pair
# (398 -> 381 epilogue instrs, bit-identical, gu -0.1%)
GU_EPI_VCONST = True
# flag-waiting GEMMs: the prologue issues phase 0's weight loads before the wait (weights never depend on the producer),
# so a workgroup launched early fetches them while it spins. Bit-identical, counters flat; pp2048 8 ABBA runs -11.5 ms
# mean at +-10 ms run spread (expected ~1-2 ms: one first-fetch latency per call)
WAIT_LATE = True
# fold software-pipelined: tile x's fmacs one fenced stage after its sub / mul (no dependent fmac right behind its mul:
# each s_delay_alu stall at raised priority lets another wave's 16-cycle WMMA in)
FOLD_PIPE = True
GROUP = 8   # grouped workgroup order: row tiles per launch band (group_of)


def group_of(kind, M):
    """Row tiles per launch band: the largest divisor of the row tile count <= GROUP. The grid is (G, token tiles, row
    tiles / G); x runs fastest, so a band of G weight tiles serves every token tile before the next band starts (its 16
    re-reads hit the MALL instead of DRAM: gu -2%) and consecutive workgroups still touch different weight tiles
    (staggered first fetches; token-fast order stalls on them). Nothing derives from the grid size (Loom folds
    workgroup.count to the config), so the engine's one-tile Trim launch stays right."""
    BM = tile(M)[0]
    n = M // (BM // 2 if kind == "gu" else BM)
    return max(g for g in range(1, min(GROUP, n) + 1) if n % g == 0)


def tile(M):
    """The workgroup geometry (BM, BN, WM, WN) for M rows: 8 waves of 32 x 64, so 3 workgroups fit a WGP (216 VGPRs,
    42 KB LDS) and one's fold, barrier and epilogue stalls overlap the others' WMMAs."""
    return (128, 128, 4, 2) if M % 128 == 0 else (48, 128, 3, 2)


def flag_count(M):
    """What a flagged kres (gen flag) adds to a token tile's counter: one per lane of its M / BM row tiles."""
    BM, _, WM, WN = tile(M)
    return M // BM * WM * WN * 32


GU_CHUNK = 1024   # output features per counter of a flagged gu (gen_rot.gen_q's 1024-value chunks)


def gu_flag_count(M):
    """What a flagged gu adds to a (chunk, token tile) counter: one per lane of the chunk's GU_CHUNK / (BM / 2) row
    tiles."""
    BM, _, WM, WN = tile(M)
    return GU_CHUNK // (BM // 2) * WM * WN * 32


def sym(kind):
    return "yah_iu4_gemm" + {"kres": "_kres", "kqg": "_kqg", "gu": "_gu"}.get(kind, "")


def gen(kind, M, out16=False, hperm=None, flag=False, wait=0, kpitch=0):
    """out16: kstore stores f16 (gen_gemm_tile.OUT16). flag (kres, kstore): binding cnt after the outputs, i32 per token
    tile (zero before the GEMM); each lane adds 1 to cnt[token tile] once its wave's stores are visible (device scope;
    flag_count per token tile), so a flag-waiting consumer (gen_rot.flag_wait) can read finished token tiles while the
    GEMM's last workgroups still run. A flagged gu counts per (GU_CHUNK features, token tile): cnt[chunk token tiles +
    token tile], gu_flag_count each (gen_rot.gen_q flag). Kind gu: the FFN's gate and up in one GEMM, f16 silu(gate) * up
    of M features; a workgroup stages BM / 2 gate rows and the same features' up rows (wave rows WM / 2 and up) and
    pairs them through LDS, so the gate never goes to memory.
    hperm (value heads, key heads): the output of weight rows 128 i.. (GGUF DeltaNet value head i) goes to HF head
    position r (i % key heads) + i / key heads (r = value / key heads), as hfq.cpp HeadOrder undoes it: a 128-row tile
    is one head, so only the tile's store origin moves.
    wait (the producer's count per token tile, e.g. gen_rot.nq_flag_count): binding wcnt last, i32 per token tile (zero
    before the producer); the workgroup waits until wcnt[token tile] reaches wait (gen_rot.flag_wait), so the producer
    of aq / ad runs without a graph barrier before this GEMM.
    kpitch: the weight rows' 256-blocks when the GEMM reads only the first k_blocks of each (a K window: the GPU's
    features of the NPU split's ffn_down), else k_blocks."""
    BM, BN, WM, WN = tile(M)
    GU = kind == "gu"
    BO = BM // 2 if GU else BM         # output rows of a workgroup
    t = TG.Tile(bm=BM, bn=BN, wm=WM, wn=WN)
    TM, TN = t.tm, t.tn
    FM, FN, NWAVE = TM // 16, TN // 16, WM * WN
    LANES = 32 * NWAVE
    assert TM % 16 == 0 and TN % 16 == 0 and (kind != "kqg" or TM == 32)
    s = sym(kind)
    bufs = ["weight", "aq", "ad"] + {"kstore": ["output"], "kres": ["resid", "output"], "kqg": ["output", "gate_out"],
                                     "gu": ["up", "output"]}[kind]
    assert not flag or kind in ("kres", "kstore", "gu")
    assert not (flag and GU) or M % GU_CHUNK == 0
    if flag:
        bufs.append("cnt")
    if wait:
        assert BN == 128
        bufs.append("wcnt")
    WB, AB = BM * WP, BN * WP          # dwords of one staged weight / activation tile
    SB = 2 * (WB + AB + BM + BN)       # dwords: two each of the weight, activation, weight scale, activation scale tiles
    epi = NWAVE * (TM + TG.EPAD) * 16  # f32 of the epilogue slabs (alias the tiles)
    LDSW = max(SB, epi)
    L = []
    e = L.append

    def assume(n, align):
        """%n = %{n}r, an LDS byte offset with no constant term, behind index.assume: canonicalization would fold the
        per-access constants into a fresh affine value in the loop, which the address lowering then rebuilds from its
        lane terms at every access (~10 VALU per phase)."""
        e(f"  %{n} = index.assume %{n}r [range(%{n}r, 0, {LDSW * 4}), mul(%{n}r, {align})] : index")

    e(f"// GENERATED by tools/gen_gemm_iu4.py {kind} -- edit the generator.")
    e(f"amdgpu.target<gfx1151> @yah_iu4_w32 {{subgroup_size = 32}}")
    e("")
    for c in ("m_tiles", "k_blocks", "token_tiles"):
        e(f"config.decl @{s}.{c} : %value: index where [range(%value, 1, 4096)]")
    e("")
    e(f"kernel.def target(@yah_iu4_w32) @{s}() {{")
    e("  %unit = index.constant 1 : index")
    e(f"  %m_tiles = config.get @{s}.m_tiles : index")
    e(f"  %token_tiles = config.get @{s}.token_tiles : index")
    e(f"  %wgs = index.constant {LANES} : index")
    e(f"  %rowgrp = index.constant {BO // 16} : index")
    e("  %m_groups = index.div %m_tiles, %rowgrp : index")
    G = group_of(kind, M)
    if G:
        e(f"  %grp = index.constant {G} : index")
        e("  %m_bands = index.div %m_groups, %grp : index")
        e("  kernel.launch.config workgroups(%grp, %token_tiles, %m_bands) workgroup_size(%wgs, %unit, %unit) : index")
    else:
        e("  kernel.launch.config workgroups(%m_groups, %token_tiles, %unit) workgroup_size(%wgs, %unit, %unit) : index")
    e("} launch(" + ", ".join(f"%{b}: buffer" for b in bufs) + ") {")
    e("  %base = index.constant 0 : offset")
    for v in sorted({0, 1, 2, 4, 8, 16, 32, 64, 256, WP, BM, BN, LANES}):
        e(f"  %c{v} = index.constant {v} : index")
    e("  %m = index.constant 16 : index")
    e("  %n = index.constant 16 : index")
    e("  %k = index.constant 16 : index")
    e(f"  %m_tiles = config.get @{s}.m_tiles : index")
    e(f"  %k_blocks = config.get @{s}.k_blocks : index")
    e(f"  %token_tiles = config.get @{s}.token_tiles : index")
    e(f"  %cwtok = index.constant {BN} : index")
    e("  %tokens = index.mul %token_tiles, %cwtok : index")
    e("  %m_rows = index.mul %m_tiles, %c16 : index")
    e("  %out_total = index.mul %m_rows, %tokens : index")
    e("  %c136 = index.constant 136 : index")
    e("  %ktot = index.mul %k_blocks, %c256 : index")
    e("  %kphases = index.mul %k_blocks, %c2 : index")
    e("  %plast = index.sub %kphases, %c1 : index")
    if kpitch:
        e(f"  %bpr = index.constant {kpitch * 136} : index")           # weight bytes per row (a K window of them)
    else:
        e("  %bpr = index.mul %k_blocks, %c136 : index")              # weight bytes per row
    e("  %w_bytes = index.mul %m_rows, %bpr : index")
    e("  %w_dw = index.div %w_bytes, %c4 : index")
    e("  %w_h = index.div %w_bytes, %c2 : index")
    e("  %a_rowdw = index.div %ktot, %c8 : index")                # activation dwords per token
    e("  %a_dw = index.mul %tokens, %a_rowdw : index")
    e("  %d_tot = index.mul %tokens, %kphases : index")
    e("  " + ", ".join(f"%{b}_na" for b in bufs) + " = buffer.assume.noalias " + ", ".join(f"%{b}" for b in bufs)
      + " : " + ", ".join(["buffer"] * len(bufs)))
    e("  %wv = buffer.view %weight_na[%base] : buffer -> view<[%w_bytes]xi8>")
    e("  %a_bytes = index.mul %a_dw, %c4 : index")
    e("  %aqv = buffer.view %aq_na[%base] : buffer -> view<[%a_bytes]xi8>")
    e("  %d_bytes = index.mul %d_tot, %c4 : index")
    e("  %adv = buffer.view %ad_na[%base] : buffer -> view<[%d_bytes]xi8>")
    if GU:
        e("  %uv = buffer.view %up_na[%base] : buffer -> view<[%w_bytes]xi8>")
    e(f"  %al_bytes = index.constant {LDSW * 4} : offset")
    e("  %al = buffer.alloca<workgroup> align(16) %al_bytes : buffer")
    e(f"  %lv = buffer.view %al[%base] : buffer -> view<{LDSW}xi32>")
    e(f"  %lf = buffer.view %al[%base] : buffer -> view<{LDSW}xf32>")
    e(f"  %lb = buffer.view %al[%base] : buffer -> view<{LDSW * 4}xi8>")   # K-loop accesses: byte offsets need no shifts
    if G:
        e("  %wg_gx = kernel.workgroup.id<x> : index")
        e("  %wg_gz = kernel.workgroup.id<z> : index")
        e(f"  %wg_gc = index.constant {G} : index")
        e("  %wg_gb = index.mul %wg_gz, %wg_gc : index")
        e("  %wg_x = index.add %wg_gb, %wg_gx : index")
    else:
        e("  %wg_x = kernel.workgroup.id<x> : index")
    e("  %wg_y = kernel.workgroup.id<y> : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %wave = index.div %tid, %c32 : index")
    e("  %lane = index.rem %tid, %c32 : index")
    def flag_wait():
        import gen_rot
        WV = "view<[%token_tiles]xi32>"
        e(f"  %wf_cv = buffer.view %wcnt_na[%base] : buffer -> {WV}")
        gen_rot.flag_wait(e, "%wf_cv", WV, ["%wg_y"], wait, "%tid", "wf")
    if wait and not WAIT_LATE:
        flag_wait()
    e("  %l16 = index.rem %lane, %c16 : index")
    e("  %lh = index.div %lane, %c16 : index")
    e(f"  %cwn = index.constant {WN} : index")
    e("  %wr = index.div %wave, %cwn : index")
    e("  %wt = index.rem %wave, %cwn : index")
    e(f"  %cbm = index.constant {BM} : index")
    e(f"  %cbo = index.constant {BO} : index")
    e("  %wg_row = index.mul %wg_x, %cbo : index")
    e("  %wtb = index.mul %wg_y, %cwtok : index")
    e(f"  %ctm = index.constant {TM} : index")
    e(f"  %ctn = index.constant {TN} : index")
    e("  %wr_off = index.mul %wr, %ctm : index")
    e("  %wt_off = index.mul %wt, %ctn : index")
    if hperm:
        nv, nk = hperm
        assert BM == 128 and M == nv * 128 and kind == "kstore"
        e("  %hp_c128 = index.constant 128 : index")
        e(f"  %hp_nk = index.constant {nk} : index")
        e(f"  %hp_r = index.constant {nv // nk} : index")
        e("  %hp_h = index.div %wg_row, %hp_c128 : index")
        e("  %hp_m = index.rem %hp_h, %hp_nk : index")
        e("  %hp_d = index.div %hp_h, %hp_nk : index")
        e("  %hp_m3 = index.mul %hp_m, %hp_r : index")
        e("  %hp_p = index.add %hp_m3, %hp_d : index")
        e("  %hp_row = index.mul %hp_p, %hp_c128 : index")
        e("  %m_origin = index.add %hp_row, %wr_off : index")
    else:
        e("  %m_origin = index.add %wg_row, %wr_off : index")
    e("  %token_base = index.add %wtb, %wt_off : index")
    # staging map: chunk c = tid + LANES * r (16 bytes: weight row c / 4 or token c / 4, quarter c % 4 of its 64 bytes)
    nwc, nac = -(-BM * 4 // LANES), -(-BN * 4 // LANES)
    assert not GU or nwc == 2 and BM * 2 == LANES   # gu: chunk 0 of every lane is a gate row, chunk 1 the up row
    e(f"  %w_clast = index.constant {BM * 4 - 1} : index")
    e(f"  %a_clast = index.constant {BN * 4 - 1} : index")
    e("  %w_dlim = index.sub %w_bytes, %c16 : index")
    e("  %a_dlim = index.sub %a_bytes, %c16 : index")
    e("  %w_hlim = index.sub %w_bytes, %c2 : index")
    for nm, n, last in (("w", nwc, "%w_clast"), ("a", nac, "%a_clast")):
        for r in range(n):
            p = f"%{nm}s{r}"
            e(f"  {p}o = index.constant {LANES * r} : index")
            e(f"  {p}c0 = index.add %tid, {p}o : index")
            e(f"  {p}live = index.cmp ule, {p}c0, {last} : index")   # past the tile: the store is skipped
            e(f"  {p}c = index.min {p}c0, {last} : index")
            e(f"  {p}row = index.div {p}c, %c4 : index")
            e(f"  {p}q = index.rem {p}c, %c4 : index")
            e(f"  {p}q4 = index.mul {p}q, %c4 : index")
            e(f"  {p}lr = index.mul {p}row, %c{WP} : index")
            e(f"  {p}lo0 = index.add {p}lr, {p}q4 : index")            # LDS dword within the tile
            e(f"  {p}lo = index.mul {p}lo0, %c4 : index")
            if LDS_SWZ:
                e(f"  {p}s80 = index.div {p}row, %c8 : index")
                e(f"  {p}s8 = index.add {p}s80, {p}row : index")
                e(f"  {p}s = index.rem {p}s8, %c2 : index")             # row bit 3 ^ bit 0
                e(f"  {p}sw0 = index.mul {p}s, %c8 : index")
                e(f"  {p}s9 = index.add {p}s8, %c1 : index")      # no index.sub in LDS bases (it stops offset folding)
                e(f"  {p}t = index.rem {p}s9, %c2 : index")
                e(f"  {p}sw1 = index.mul {p}t, %c8 : index")
                for ab, sw in (("A", "sw0"), ("B", "sw1")):         # A: k step 2q of the chunk, B: 2q + 1
                    e(f"  {p}lo{ab}r = index.add {p}lo, {p}{sw} : index")
                    e(f"  {p}lo{ab} = index.assume {p}lo{ab}r [range({p}lo{ab}r, 0, {LDSW * 4}), mul({p}lo{ab}r, 8)] : index")
            if nm == "w":
                if GU and r:
                    e(f"  {p}gr0 = index.add %wg_row, {p}row : index")
                    e(f"  {p}gr = index.sub {p}gr0, %cbo : index")
                else:
                    e(f"  {p}gr = index.add %wg_row, {p}row : index")
                e(f"  {p}gb = index.mul {p}gr, %bpr : index")
                e(f"  {p}qb = index.mul {p}q, %c16 : index")
                e(f"  {p}gbq = index.add {p}gb, {p}qb : index")         # + 136 * block + 8 + 64 * half per phase
            else:
                e(f"  {p}gt = index.add %wtb, {p}row : index")
                e(f"  {p}gb = index.mul {p}gt, %a_rowdw : index")
                e(f"  {p}gbq0 = index.add {p}gb, {p}q4 : index")
                e(f"  {p}gbq = index.mul {p}gbq0, %c4 : index")          # bytes; + 64 * phase
    # scales: lane tid < BM loads row tid's f16 scale, stores it at its C-layout slot: tile r / 16, half r % 2, i (r % 16) / 2
    e(f"  %sc_last = index.constant {BM - 1} : index")
    e("  %sc_r = index.min %tid, %sc_last : index")
    if GU:
        e("  %sc_up = index.cmp uge, %sc_r, %cbo : index")      # rows BO.. are the up tensor's
        e("  %sc_rf = index.rem %sc_r, %cbo : index")
        e("  %sc_gr = index.add %wg_row, %sc_rf : index")
    else:
        e("  %sc_gr = index.add %wg_row, %sc_r : index")
    e("  %sc_gb = index.mul %sc_gr, %bpr : index")
    e("  %sc_t = index.div %sc_r, %c16 : index")
    e("  %sc_rr = index.rem %sc_r, %c16 : index")
    e("  %sc_h = index.rem %sc_rr, %c2 : index")
    e("  %sc_i = index.div %sc_rr, %c2 : index")
    e("  %sc_t16 = index.mul %sc_t, %c16 : index")
    e("  %sc_h8 = index.mul %sc_h, %c8 : index")
    e("  %sc_o0 = index.add %sc_t16, %sc_h8 : index")
    e("  %sc_o1 = index.add %sc_o0, %sc_i : index")
    e("  %sc_lor = index.mul %sc_o1, %c4 : index")
    assume("sc_lo", 4)
    # activation scales: lane tid < BN loads token tid's; global loads in the phase that uses them would make its fold
    # wait (in order) for the next phase's prefetch
    e(f"  %ds_last = index.constant {BN - 1} : index")
    e("  %ds_t = index.min %tid, %ds_last : index")
    e("  %ds_gt = index.add %wtb, %ds_t : index")
    e("  %ds_gd = index.mul %ds_gt, %kphases : index")
    e("  %ds_gb = index.mul %ds_gd, %c4 : index")
    e("  %ds_lor = index.mul %ds_t, %c4 : index")
    assume("ds_lo", 4)

    def loads(pfx, kp, mid=None):
        """The global loads of phase kp into registers; returns [(kind, value, lds offset, live)]. With mid: the weight
        loads, then mid() (the prologue's flag wait), then the activation loads."""
        e(f"  %{pfx}kb = index.div {kp}, %c2 : index")
        e(f"  %{pfx}hf = index.rem {kp}, %c2 : index")
        e(f"  %{pfx}kb136 = index.mul %{pfx}kb, %c136 : index")
        e(f"  %{pfx}hf64 = index.mul %{pfx}hf, %c64 : index")
        e(f"  %{pfx}wo0 = index.add %{pfx}kb136, %{pfx}hf64 : index")
        e(f"  %{pfx}wo = index.add %{pfx}wo0, %c8 : index")            # nibble bytes of the phase within the row
        e(f"  %{pfx}ao = index.mul {kp}, %c64 : index")                  # activation bytes of the phase within the row
        out = []
        order = (("w", nwc), ("ws", 0), ("mid", 0), ("a", nac), ("d", 0)) if mid else (("w", nwc), ("a", nac), ("ws", 0), ("d", 0))
        for nm, n in order:
            if nm == "mid":
                mid()
            if nm == "ws":
                weight_scales(pfx, out)
            if nm == "d":
                act_scale(pfx, kp, out)
            for r in range(n):
                p = f"%{nm}s{r}"
                if nm == "w":
                    e(f"  %{pfx}w{r}d0 = index.add {p}gbq, %{pfx}wo : index")
                    e(f"  %{pfx}w{r}d = index.min %{pfx}w{r}d0, %w_dlim : index")
                    wv = "%uv" if GU and r else "%wv"
                    e(f"  %{pfx}w{r}8 = vector.load {wv}[%{pfx}w{r}d] : view<[%w_bytes]xi8> -> vector<16xi8>")
                    e(f"  %{pfx}w{r} = vector.bitcast %{pfx}w{r}8 : vector<16xi8> to vector<4xi32>")
                else:
                    e(f"  %{pfx}a{r}d0 = index.add {p}gbq, %{pfx}ao : index")
                    e(f"  %{pfx}a{r}d = index.min %{pfx}a{r}d0, %a_dlim : index")
                    e(f"  %{pfx}a{r}8 = vector.load %aqv[%{pfx}a{r}d] : view<[%a_bytes]xi8> -> vector<16xi8>")
                    e(f"  %{pfx}a{r} = vector.bitcast %{pfx}a{r}8 : vector<16xi8> to vector<4xi32>")
                out.append((nm, f"%{pfx}{nm}{r}", f"{p}lo", f"{p}live"))
        return out

    def weight_scales(pfx, out):
        e(f"  %{pfx}hf4 = index.mul %{pfx}hf, %c4 : index")
        e(f"  %{pfx}sb0 = index.add %sc_gb, %{pfx}kb136 : index")
        e(f"  %{pfx}sb = index.add %{pfx}sb0, %{pfx}hf4 : index")
        e(f"  %{pfx}sh = index.min %{pfx}sb, %w_hlim : index")
        for nm, v in (("g", "%wv"), ("u", "%uv")) if GU else (("", "%wv"),):
            e(f"  %{pfx}sc{nm}8 = vector.load {v}[%{pfx}sh] : view<[%w_bytes]xi8> -> vector<2xi8>")
            e(f"  %{pfx}sc{nm}h = vector.bitcast %{pfx}sc{nm}8 : vector<2xi8> to vector<1xf16>")
            e(f"  %{pfx}sc{nm} = vector.extract %{pfx}sc{nm}h[0] : vector<1xf16> -> f16")
        out.append(("s", f"%{pfx}sc", "%sc_lo", None))

    def act_scale(pfx, kp, out):
        e(f"  %{pfx}kp4 = index.mul {kp}, %c4 : index")
        e(f"  %{pfx}di = index.add %ds_gb, %{pfx}kp4 : index")
        e(f"  %{pfx}dt8 = vector.load %adv[%{pfx}di] : view<[%d_bytes]xi8> -> vector<4xi8>")
        e(f"  %{pfx}dtv = vector.bitcast %{pfx}dt8 : vector<4xi8> to vector<1xf32>")
        e(f"  %{pfx}dt = vector.extract %{pfx}dtv[0] : vector<1xf32> -> f32")
        out.append(("d", f"%{pfx}dt", "%ds_lo", None))

    def offsets(q, buf):
        """The byte offsets of LDS buffer buf (an index value) as %{q}wbo / abo / sbo / dbo."""
        e(f"  %{q}cwb = index.constant {WB * 4} : index")
        e(f"  %{q}cab = index.constant {AB * 4} : index")
        e(f"  %{q}wbo = index.mul {buf}, %{q}cwb : index")
        e(f"  %{q}abo0 = index.mul {buf}, %{q}cab : index")
        e(f"  %{q}cwb2 = index.constant {2 * WB * 4} : index")
        e(f"  %{q}abo = index.add %{q}abo0, %{q}cwb2 : index")
        e(f"  %{q}cbm4 = index.constant {BM * 4} : index")
        e(f"  %{q}cbn4 = index.constant {BN * 4} : index")
        e(f"  %{q}sbo0 = index.mul {buf}, %{q}cbm4 : index")
        e(f"  %{q}dbo0 = index.mul {buf}, %{q}cbn4 : index")
        # the scale region bases are here, not in the lane values (see assume)
        e(f"  %{q}csb = index.constant {2 * (WB + AB) * 4} : index")
        e(f"  %{q}cdb = index.constant {2 * (WB + AB + BM) * 4} : index")
        e(f"  %{q}sbo = index.add %{q}sbo0, %{q}csb : index")
        e(f"  %{q}dbo = index.add %{q}dbo0, %{q}cdb : index")

    def stores(pfx, regs, buf):
        """Store the registers of loads() into LDS buffer buf (an index value: 0 or 1)."""
        offsets(pfx, buf)
        for x, (nm, v, lo, live) in enumerate(regs):
            off = {"w": f"%{pfx}wbo", "a": f"%{pfx}abo", "s": f"%{pfx}sbo", "d": f"%{pfx}dbo"}[nm]
            e(f"  %{pfx}st{x} = index.add {lo}, {off} : index")
            # scales: every lane stores; lanes past the tile repeat the last row's (token's) value at its slot
            if nm in "sd":
                f = v
                if nm == "s":
                    if GU:   # gate / up scale selected here: next to its loads, it made the phase start wait for them all
                        e(f"  %{pfx}sc = scf.select %sc_up, %{pfx}scu, %{pfx}scg : f16")
                    f = f"%{pfx}sf"
                    e(f"  {f} = scalar.extf {v} : f16 to f32")
                e(f"  %{pfx}st{x}v = vector.from_elements {f} : vector<1xf32>")
                e(f"  %{pfx}st{x}b = vector.bitcast %{pfx}st{x}v : vector<1xf32> to vector<4xi8>")
                e(f"  vector.store %{pfx}st{x}b, %lb[%{pfx}st{x}] : vector<4xi8>, view<{LDSW * 4}xi8>")
            elif LDS_SWZ:
                e(f"  scf.if {live} {{")
                for hh, ab in enumerate("AB"):
                    e(f"    %{pfx}st{x}{ab} = index.add {lo}{ab}, {off} : index")
                    e(f"    %{pfx}st{x}{ab}w = vector.slice {v}[{2 * hh}] : vector<4xi32> -> vector<2xi32>")
                    e(f"    %{pfx}st{x}{ab}b = vector.bitcast %{pfx}st{x}{ab}w : vector<2xi32> to vector<8xi8>")
                    e(f"    vector.store %{pfx}st{x}{ab}b, %lb[%{pfx}st{x}{ab}] : vector<8xi8>, view<{LDSW * 4}xi8>")
                e("  }")
            else:
                e(f"  scf.if {live} {{")
                e(f"    vector.store {v}8, %lb[%{pfx}st{x}] : vector<16xi8>, view<{LDSW * 4}xi8>")
                e("  }")

    # prologue: phase 0 into buffer 0
    stores("p0_", loads("p0_", "%c0", flag_wait if wait and WAIT_LATE else None), "%c0")
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    # per-lane LDS byte offsets of fragment 0 in buffer 0: weight row m0 + l16, token n0 + l16 (fragment i / j is 16 i / j
    # rows further), the weight scale slots (tile * 16 + half * 8) and this lane's token's activation scale
    for i in range(FM):
        e(f"  %fr{i}c = index.constant {16 * i} : index")
    for j in range(FN):
        e(f"  %ft{j}c = index.constant {16 * j} : index")
    e("  %fr1 = index.add %wr_off, %l16 : index")
    e(f"  %frd = index.mul %fr1, %c{WP} : index")
    e("  %fr0r = index.mul %frd, %c4 : index")
    assume("fr0", 16)
    e("  %fst = index.div %wr_off, %c16 : index")
    e("  %fst16 = index.mul %fst, %c16 : index")
    e("  %fsh = index.mul %lh, %c8 : index")
    e("  %fs1 = index.add %fst16, %fsh : index")
    e("  %fs0r = index.mul %fs1, %c4 : index")
    assume("fs0", 32)
    e("  %ft1 = index.add %wt_off, %l16 : index")
    e(f"  %ft2 = index.mul %ft1, %c{WP} : index")
    e("  %ft0r = index.mul %ft2, %c4 : index")
    assume("ft0", 16)
    if LDS_SWZ:
        e("  %fsw_80 = index.div %l16, %c8 : index")
        e("  %fsw_8 = index.add %fsw_80, %l16 : index")
        e("  %fsw_r = index.rem %fsw_8, %c2 : index")
        e("  %fsw_s = index.mul %fsw_r, %c8 : index")         # 8 for rows with bit 3 ^ bit 0 set
        e("  %fsw_9 = index.add %fsw_8, %c1 : index")
        e("  %fsw_u = index.rem %fsw_9, %c2 : index")
        e("  %fsw_t = index.mul %fsw_u, %c8 : index")
        for n, sw in (("fr0e", "%fsw_s"), ("fr0o", "%fsw_t"), ("ft0e", "%fsw_s"), ("ft0o", "%fsw_t")):
            e(f"  %{n}r = index.add %{n[:3]}r, {sw} : index")
            assume(n, 8)
    e("  %fd0r = index.mul %ft1, %c4 : index")
    assume("fd0", 4)
    e("  %schema = encoding.define #encoding.operand<element_format=i4, payload_elements=16, payload_registers=2> : encoding<schema>")
    # WMMA chains start from the bits of 1/(2 pi) (an inline source): the result, read as f32, is 1/(2 pi) + c * 2^-26
    # exactly (|c| <= 8192), so one subtract converts it; rotq stores the activation scales times 2^26 to match
    e("  %mzi = scalar.constant 1042479491 : i32")
    e(f"  %mzv = vector.splat %mzi : {VI}")
    e("  %minv = scalar.constant 0.15915494 : f32")
    e(f"  %minvv = vector.splat %minv : {V8}")
    e("  %zf = scalar.constant 0.0 : f32")
    e(f"  %zfv = vector.splat %zf : {V8}")
    NA = FM * FN

    def kst(st):
        """Dword offset of k step st in a staged row (the odd-step lane base carries the swizzled 8 bytes)."""
        return 4 * (st // 2) if LDS_SWZ else 2 * st

    def sfx(st):
        return ("o" if st % 2 else "e") if LDS_SWZ else ""

    def phase(q, acc, ba, mid=None):
        """The WMMAs of the phase in the LDS buffer whose lane addresses (fragment 0, k step 0) are ba[la / lb / dt /
        ws], then mid(), then the fold into acc; returns the new accumulator names."""
        for j in range(FN):
            e(f"    %{q}dti{j}c = index.constant {64 * j} : index")
            e(f"    %{q}dti{j} = index.add {ba['dt']}, %{q}dti{j}c : index")
            e(f"    %{q}dt{j}b = vector.load %lb[%{q}dti{j}] : view<{LDSW * 4}xi8> -> vector<4xi8>")
            e(f"    %{q}dt{j}v = vector.bitcast %{q}dt{j}b : vector<4xi8> to vector<1xf32>")
            e(f"    %{q}dt{j} = vector.extract %{q}dt{j}v[0] : vector<1xf32> -> f32")
            e(f"    %{q}dtv{j} = vector.splat %{q}dt{j} : {V8}")
        def frag_loads(st):
            for i in range(FM):
                e(f"    %{q}la{st}_{i}c = index.constant {(16 * i * WP + kst(st)) * 4} : index")
                e(f"    %{q}la{st}_{i} = index.add {ba['la' + sfx(st)]}, %{q}la{st}_{i}c : index")
                e(f"    %{q}wd{st}_{i}b = vector.load %lb[%{q}la{st}_{i}] : view<{LDSW * 4}xi8> -> vector<8xi8>")
                e(f"    %{q}wd{st}_{i} = vector.bitcast %{q}wd{st}_{i}b : vector<8xi8> to {V2}")
                e(f"    %{q}lhs{st}_{i} = vector.fragment<lhs> %{q}wd{st}_{i} shape [%m, %k] using {{schema = %schema : encoding<schema>}} : {V2}")
            if GU and GU_FRAG_FENCE:
                e("    scf.schedule.fence")
            for j in range(FN):
                e(f"    %{q}lb{st}_{j}c = index.constant {(16 * j * WP + kst(st)) * 4} : index")
                e(f"    %{q}lb{st}_{j} = index.add {ba['lb' + sfx(st)]}, %{q}lb{st}_{j}c : index")
                e(f"    %{q}ad{st}_{j}b = vector.load %lb[%{q}lb{st}_{j}] : view<{LDSW * 4}xi8> -> vector<8xi8>")
                e(f"    %{q}ad{st}_{j} = vector.bitcast %{q}ad{st}_{j}b : vector<8xi8> to {V2}")
                e(f"    %{q}rhs{st}_{j} = vector.fragment<rhs> %{q}ad{st}_{j} shape [%k, %n] using {{schema = %schema : encoding<schema>}} : {V2}")

        c = {}

        def mmas(st):
            for i in range(FM):
                for j in range(FN):
                    if st == 0:
                        e(f"    %{q}init{i}_{j} = vector.fragment<init> %mzv shape [%m, %n] : {VI}")
                        c[i, j] = f"%{q}init{i}_{j}"
                    e(f"    %{q}c{st}_{i}_{j} = vector.mma %{q}lhs{st}_{i}, %{q}rhs{st}_{j}, {c[i, j]} : {V2}, {V2}, {VI}")
                    c[i, j] = f"%{q}c{st}_{i}_{j}"

        # step s + 1's fragment loads go before step s's WMMAs, behind a fence: unfenced, the scheduler hoists every
        # load of the phase (all offsets are immediates) and the extra live fragments cost the WMMA operand banks
        NST = KP // 16
        frag_loads(0)
        for st in range(NST):
            if st + 1 < NST:
                frag_loads(st + 1)
            mmas(st)
            if st < NST - 1:
                e("    scf.schedule.fence")
        # fold: acc += f32(c) * weight scale (per element) * activation scale (per lane), at raised issue priority so
        # the folding wave's VALU wins arbitration against the other waves' WMMAs and it gets back to its own sooner
        out = []
        for i in range(FM):
            e(f"    %{q}wsi{i}c = index.constant {64 * i} : index")
            e(f"    %{q}wsi{i} = index.add {ba['ws']}, %{q}wsi{i}c : index")
            e(f"    %{q}ws{i}b = vector.load %lb[%{q}wsi{i}] : view<{LDSW * 4}xi8> -> vector<32xi8>")
            e(f"    %{q}ws{i} = vector.bitcast %{q}ws{i}b : vector<32xi8> to {V8}")
        if mid:
            mid()       # priority is a scheduling boundary: work that should overlap the WMMAs goes before it
        e("    kernel.priority {level = 3}")
        def pre(x):
            i, j = divmod(x, FN)
            e(f"    %{q}cb{x} = vector.bitcast {c[i, j]} : {VI} to {V8}")
            e(f"    %{q}cf{x} = vector.subf %{q}cb{x}, %minvv : {V8}")
            e(f"    %{q}sc{x} = vector.mulf %{q}ws{i}, %{q}dtv{j} : {V8}")
        def fin(x):
            e(f"    %{q}f{x} = vector.fmaf %{q}cf{x}, %{q}sc{x}, {acc[x]} : {V8}")
        if FOLD_PIPE:
            for g in range(FM * FN + 1):
                if g < FM * FN:
                    pre(g)
                if g:
                    fin(g - 1)
                if g < FM * FN:
                    e("    scf.schedule.fence")
        else:
            for x in range(FM * FN):
                pre(x)
                fin(x)
        out += [f"%{q}f{x}" for x in range(FM * FN)]
        e("    kernel.priority {level = 0}")
        return out

    ca = ", ".join(f"%a{x} = %zfv : {V8}" for x in range(NA))
    e("  " + ", ".join(f"%acc{x}" for x in range(NA)) + f" = scf.for %kp = [%c0 to %kphases step %c2]({ca}) -> ("
      + ", ".join([V8] * NA) + ") {")
    def half(h, kp, buf, nbuf, acc):
        """One phase at K phase kp (an index) from LDS buffer buf, loading phase kp + 1 into nbuf; h: name prefix."""
        e(f"    %{h}kn0 = index.add {kp}, %c1 : index")
        e(f"    %{h}kn = index.min %{h}kn0, %plast : index")
        regs = loads(h + "n_", f"%{h}kn")
        # next phase's global loads issue first: behind a burst of this phase's fragment loads they stall at issue
        e("    scf.schedule.fence")
        offsets(h + "u_", buf)
        # one base per LDS read stream; fragment, tile and k-step offsets are immediates
        b = {}
        streams = (("la", "%fr0", "wbo"), ("lb", "%ft0", "abo"), ("dt", "%fd0", "dbo"), ("ws", "%fs0", "sbo"))
        if LDS_SWZ:
            streams = (("lae", "%fr0e", "wbo"), ("lao", "%fr0o", "wbo"), ("lbe", "%ft0e", "abo"), ("lbo", "%ft0o", "abo"),
                       ("dt", "%fd0", "dbo"), ("ws", "%fs0", "sbo"))
        for n, inv, off in streams:
            e(f"    %{h}bi_{n} = index.add {inv}, %{h}u_{off} : index")
            b[n] = f"%{h}bi_{n}"
        acc = phase(h + "u_", acc, b, lambda: stores(h + "n_", regs, nbuf))
        e("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        return acc

    # two phases per iteration (the phase count 2 k_blocks is even): constant LDS buffers, so the buffer offsets are
    # immediates instead of VALU on the parity (-11 VALU per phase, GEMMs -3.7%)
    acc = half("p0", "%kp", "%c0", "%c1", [f"%a{x}" for x in range(NA)])
    e("    %kp1 = index.add %kp, %c1 : index")
    acc = half("p1", "%kp1", "%c1", "%c0", acc)
    e("    scf.yield " + ", ".join(acc) + " : " + ", ".join([V8] * NA))
    e("  }")
    if GU:
        gu_epilogue(e, TM, FM, FN, WN, V8)
    elif TM != 32:
        # lds_epilogue reads 32-row slabs: other tiles store the result fragments directly (f32 kstore)
        assert kind == "kstore" and not out16
        e("  %out_layout = encoding.layout.strided [%c1, %m_rows] : encoding<layout>")
        OV = "view<[%m_rows]x[%tokens]xf32, %out_layout>"
        e(f"  %out_t_view = buffer.view %output_na[%base] : buffer -> {OV}")
        for i in range(FM):
            for j in range(FN):
                e(f"  %or{i}_{j} = index.add %m_origin, %fr{i}c : index")
                e(f"  %ot{i}_{j} = index.add %token_base, %ft{j}c : index")
                e(f"  vector.fragment.store<result> %acc{i * FN + j}, %out_t_view[%or{i}_{j}, %ot{i}_{j}] shape [%m, %n] : {V8}, {OV}")
    else:
        TG.OUT16, TG.OSTRIDE = out16, 0
        try:
            TG.lds_epilogue(e, t, kind == "kres", V8, qg=kind == "kqg", col_fence=kind == "kres")
        finally:
            TG.OUT16 = False
    if flag:
        # every lane, no workgroup barrier and no thread-id test (that keeps the id live and shifts the main loop's
        # registers: +0.9% vs +0.4%); same-address adds are cheap next to the stores
        e("  %kf_one = scalar.constant 1 : i32")
        if GU:
            e(f"  %kf_nch = index.constant {M // GU_CHUNK} : index")
            e("  %kf_n = index.mul %kf_nch, %token_tiles : index")
            e(f"  %kf_rt = index.constant {GU_CHUNK // BO} : index")
            e("  %kf_ch = index.div %wg_x, %kf_rt : index")
            e("  %kf_i0 = index.mul %kf_ch, %token_tiles : index")
            e("  %kf_i = index.add %kf_i0, %wg_y : index")
            e("  %kf_cv = buffer.view %cnt_na[%base] : buffer -> view<[%kf_n]xi32>")
            e("  view.atomic.reduce<addi> %kf_one, %kf_cv[%kf_i] {ordering = release, scope = device} : i32, view<[%kf_n]xi32>")
        else:
            e("  %kf_cv = buffer.view %cnt_na[%base] : buffer -> view<[%token_tiles]xi32>")
            e("  view.atomic.reduce<addi> %kf_one, %kf_cv[%wg_y] {ordering = release, scope = device} : i32, view<[%token_tiles]xi32>")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def gu_epilogue(e, TM, FM, FN, WN, V8):
    """gu: per 16-token column every wave puts its fragments in its LDS slab; then gate wave w (wave rows 0, 1) and up
    wave w + 2 WN each take half of their 32 rows: lane l, token l / 2, rows 16 (l % 2) + 8 p (p: 0 gate wave, 1 up
    wave), 8 x f16(silu(gate) * up) per 16-byte store. The silu is lds_epilogue's swiglu, with a bare rcp when
    GU_BARE_RCP."""
    W = (TM + TG.EPAD) * 16            # f32 of a slab
    e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    e(f"  %es_ctm = index.constant {TM + TG.EPAD} : index")
    e("  %es_lay = encoding.layout.strided [%c1, %es_ctm] : encoding<layout>")
    e(f"  %es_wb = index.constant {W * 4} : index")
    e("  %es_off_i = index.mul %wave, %es_wb : index")
    e("  %es_off = index.cast %es_off_i : index to offset")
    e(f"  %es_view = buffer.view %al[%es_off] : buffer -> view<{TM}x16xf32, %es_lay>")
    e(f"  %gu_all = buffer.view %al[%base] : buffer -> view<{W * 4 * WN}xf32>")
    e("  %out_h = buffer.view %output_na[%base] : buffer -> view<[%out_total]xf16>")
    e("  %negone = scalar.constant -1.0 : f32")
    # exp(-g) as exp2(g * -log2e): what expf<afn> lowers to after the negation, in one multiply (bit-identical)
    e("  %nl2e = scalar.constant -1.4426950408889634 : f32")
    if not GU_BARE_RCP:
        TG.rcp_consts(e)
    e("  %one = scalar.constant 1.0 : f32")
    nl2e, one = "%nl2e", "%one"
    if GU_EPI_VCONST:
        e("  %ev_i = index.cast %lane : index to i32")
        e("  %ev_f = scalar.sitofp %ev_i : i32 to f32")
        e("  %ev_z = scalar.constant 0.0 : f32")
        e("  %ev_0 = scalar.mulf %ev_f, %ev_z : f32")
        e("  %nl2e_v = scalar.addf %ev_0, %nl2e : f32")
        e("  %one_v = scalar.addf %ev_0, %one : f32")
        nl2e, one = "%nl2e_v", "%one_v"
    e("  %gu_p = index.div %wr, %c2 : index")               # 0: a gate wave, 1: an up wave
    e("  %gu_wr = index.rem %wr, %c2 : index")
    e(f"  %gu_cw = index.constant {WN} : index")
    e("  %gu_gw0 = index.mul %gu_wr, %gu_cw : index")
    e("  %gu_gw = index.add %gu_gw0, %wt : index")          # the pair's gate wave
    e(f"  %gu_cs = index.constant {W} : index")
    e("  %gu_gs = index.mul %gu_gw, %gu_cs : index")
    e(f"  %gu_cu = index.constant {2 * WN * W} : index")
    e("  %gu_us = index.add %gu_gs, %gu_cu : index")
    e("  %gu_t = index.div %lane, %c2 : index")
    e("  %gu_h0 = index.rem %lane, %c2 : index")
    e("  %gu_h = index.mul %gu_h0, %c16 : index")
    e("  %gu_p8 = index.mul %gu_p, %c8 : index")
    e("  %gu_r = index.add %gu_h, %gu_p8 : index")          # first of the lane's 8 rows within the wave's 32
    e("  %gu_tt = index.mul %gu_t, %es_ctm : index")
    e("  %gu_rd = index.add %gu_tt, %gu_r : index")
    e("  %gu_gr = index.add %gu_gs, %gu_rd : index")
    e("  %gu_ur = index.add %gu_us, %gu_rd : index")
    e(f"  %gu_ctm = index.constant {TM} : index")
    e("  %gu_fo = index.mul %gu_wr, %gu_ctm : index")
    e("  %gu_f0 = index.add %wg_row, %gu_fo : index")
    e("  %gu_f = index.add %gu_f0, %gu_r : index")          # the lane's first feature
    for j in range(FN):
        for i in range(FM):
            e(f"  %es_r{i}_{j} = index.constant {16 * i} : index")
            e(f"  vector.fragment.store<result> %acc{i * FN + j}, %es_view[%es_r{i}_{j}, %c0] shape [%m, %n] : {V8}, view<{TM}x16xf32, %es_lay>")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
        e(f"  %gu_tc{j} = index.constant {16 * j} : index")
        e(f"  %gu_tk{j}0 = index.add %token_base, %gu_tc{j} : index")
        e(f"  %gu_tk{j} = index.add %gu_tk{j}0, %gu_t : index")
        e(f"  %gu_tm{j} = index.mul %gu_tk{j}, %m_rows : index")
        e(f"  %gu_ob{j} = index.add %gu_tm{j}, %gu_f : index")
        hs = []
        for q in range(2):
            e(f"  %gu_gi{j}_{q} = index.add %gu_gr, %c{4 * q} : index")
            e(f"  %gu_ui{j}_{q} = index.add %gu_ur, %c{4 * q} : index")
            e(f"  %gu_g{j}_{q} = vector.load %gu_all[%gu_gi{j}_{q}] : view<{W * 4 * WN}xf32> -> vector<4xf32>")
            e(f"  %gu_u{j}_{q} = vector.load %gu_all[%gu_ui{j}_{q}] : view<{W * 4 * WN}xf32> -> vector<4xf32>")
            for x in range(4):
                y = f"{j}_{q}_{x}"
                e(f"  %g_{y} = vector.extract %gu_g{j}_{q}[{x}] : vector<4xf32> -> f32")
                e(f"  %v_{y} = vector.extract %gu_u{j}_{q}[{x}] : vector<4xf32> -> f32")
                e(f"  %ng_{y} = scalar.mulf %g_{y}, {nl2e} : f32")
                e(f"  %ex_{y} = scalar.exp2f<afn> %ng_{y} : f32")
                e(f"  %dn_{y} = scalar.addf {one}, %ex_{y} : f32")
                TG.recip1(e, f"%iv_{y}", f"%dn_{y}", bare=GU_BARE_RCP)
                e(f"  %sg_{y} = scalar.mulf %g_{y}, %iv_{y} : f32")
                e(f"  %ac_{y} = scalar.mulf %sg_{y}, %v_{y} : f32")
                e(f"  %h_{y} = scalar.fptrunc %ac_{y} : f32 to f16")
                hs.append(f"%h_{y}")
        e(f"  %hv{j} = vector.from_elements {', '.join(hs)} : vector<8xf16>")
        e(f"  vector.store %hv{j}, %out_h[%gu_ob{j}] : vector<8xf16>, view<[%out_total]xf16>")
        e("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")


if __name__ == "__main__":
    print(gen(sys.argv[1], int(sys.argv[2])), end="")
