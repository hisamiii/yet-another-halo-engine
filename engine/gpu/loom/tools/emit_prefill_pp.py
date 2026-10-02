#!/usr/bin/env python3
"""Emit the full-prompt (B-token) Loom prefill HAL set.

Same naming convention as emit_prefill.py, but *every* token dimension is bound
to B instead of the 5-token batch:

  GEMM family      token_tiles = B / TOKEN_TILE        (grid y)
  fixed kernels    batch = B
  attention        max_context = score_capacity = B, KV cache B deep
  residual reduce  dim = 5120 * B                      (the whole prompt)

The whole prompt goes through in ONE pass, so start_pos stays 0 everywhere, the
KV cache is written once, and the recurrent state (conv ring, DeltaNet state)
starts zeroed -- exactly the first-chunk case the 5-token path already
validates. No token-tile loop, and therefore no per-tile launch overhead: the
GEMM grid carries the token dimension the same way the benchmark arms do.

usage: emit_prefill_pp.py <model.gguf> <outdir> [tokens]

env:
  YAH_TOKEN_TILE  GEMM tokens per workgroup (16|64|128|256; default 64).
                  Anything but 64 goes through tools/widen_tokens.widen().
  YAH_PP_TOKENS   default token count when argv[3] is absent.
"""
import os, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emit_prefill as E  # noqa: E402


def widen_source(loomfile, text, tile):
    """Widen (or leave at 64) the N sub-tiles of a GEMM port source."""
    if tile == 64:
        return text
    import widen_tokens as W
    origin = "%m_origin_s" if "%m_origin_s" in text else "%m_origin"
    return W.widen(text, tile // 16, m_origin=origin)


def shared_kstore(fmt, mt, kb, B, out, outdir, kind="kstore"):
    """Emit the shared-decode kStore (tools/gen_gemm_shared.py) for this shape if
    it covers the format, and return its dispatch.txt geometry, else None.

    It replaces the chained kStore: same ABI and output, bit-identical, with the
    decoded weight tile shared by NW waves, a prefetched branch-free decode and a
    direct token-major epilogue. YAH_SHARED_GEMM=0 keeps the chained kernel.
    """
    import gen_gemm_shared as G
    if os.environ.get("YAH_SHARED_GEMM", "1") == "0" or fmt not in G.FMTS:
        return None
    r = tile_kstore(fmt, mt, kb, B, out, outdir, kind)
    if r:
        return r
    if mt < 4:
        # the 48-row ssm_alpha/ssm_beta: one 16-row tile per workgroup, 64 tokens
        # over 2 waves, (3, B/64) workgroups: 0.40 -> 0.11 ms standalone (Q5_K)
        r = tile_kstore(fmt, mt, kb, B, out, outdir, kind, geom=(16, 64, 1, 2))
        if r:
            return r
    if mt % 4 and mt > 4:
        return None
    # Matrices under 64 rows (the 48-row ssm_alpha/ssm_beta, m_tiles=3) get
    # m_tiles row tiles per wave and 16-token waves, so the grid is 64 workgroups
    # instead of 8: 0.70-0.92 -> 0.47 ms per dispatch, bit-identical.
    small = mt < 4
    prev = G.set_geometry(mt=mt, tok=16) if small else None
    try:
        return _emit_shared(G, fmt, mt, kb, B, out, outdir, kind, rowgrp=mt if small else 4)
    finally:
        if prev:
            G.set_geometry(*prev)


# Formats the tile GEMM (tools/gen_gemm_tile.py) has been verified bit-identical
# on in the pp2048 pipeline (q8_0 through the same kdiv as gen_gemm_shared).
TILE_FMTS = ("iq3s", "iq4xs", "iq3xxs", "q4k", "q5k", "q6k", "iq2xxs", "iq2xs", "q3k", "q8_0")


def tile_kstore(fmt, mt, kb, B, out, outdir, kind, geom=None):
    """Emit the tile GEMM (tools/gen_gemm_tile.py: 16 wave32 waves over a
    128-row x 256-token workgroup, both operands in padded LDS tiles) for this
    shape if it covers it, and return its dispatch.txt geometry, else None.
    YAH_TILE_GEMM=0 keeps the shared-decode kernel."""
    import gen_gemm_tile as TG
    if os.environ.get("YAH_TILE_GEMM", "1") == "0" or fmt not in TILE_FMTS:
        return None
    prev_geom = TG.set_geometry(*geom) if geom else None
    try:
        return _tile_kstore(TG, fmt, mt, kb, B, out, outdir, kind)
    finally:
        if prev_geom:
            TG.set_geometry(*prev_geom)


def _tile_kstore(TG, fmt, mt, kb, B, out, outdir, kind):
    tile, rowgrp = TG.geometry()
    if mt % rowgrp or B % tile:
        return None
    # grouped launch order (gen_gemm_tile SWZ) where the row groups divide by it
    swz = int(os.environ.get("YAH_TILE_SWZ", "0"))
    prev_swz, prev_da = TG.SWZ, TG.DECAHEAD_ENV
    TG.SWZ = swz if swz and (mt // rowgrp) % swz == 0 else 0
    # decode-ahead lost on this one shape in two paired pp2048 profiles
    # (IQ4_XS kres 5120 x 6144: 88.5 -> 93.7 / 95.5 ms); it keeps KSUB=64
    if (fmt, kind, kb) in DECAHEAD_SKIP and prev_da is None:
        TG.DECAHEAD_ENV = "0"
    try:
        return _tile_emit(TG, fmt, mt, kb, B, out, outdir, kind, tile, rowgrp)
    finally:
        TG.SWZ, TG.DECAHEAD_ENV = prev_swz, prev_da


# Short-K residual GEMMs: with decode-ahead, 45% of wave time is s_waitcnt
# vmcnt(0) in the K loop (ATT, Q4_K kres K=6144), full drains that serialize
# the read-ahead. Q4_K kres K=6144 11.34 -> 9.51 M cycles off (bit-identical).
DECAHEAD_SKIP = {("iq4xs", "kres", 24), ("q4k", "kres", 24)}


def _tile_emit(TG, fmt, mt, kb, B, out, outdir, kind, tile, rowgrp):
    # TG.configure() rewrites gen_gemm_shared's module globals (NW, LR, KSUB...)
    # to drive the shared decode helpers; put them back for the kernels the
    # shared generator still emits in this process.
    G = TG.G
    keep = {k: getattr(G, k) for k in ("KSUB", "PAD", "ROWP", "PH", "GPP", "GPL", "LR", "NW")}
    try:
        return _emit_gen(lambda f, k: TG.gen(f, k), tile, fmt, mt, kb, B, out, outdir, kind, rowgrp)
    finally:
        for k, v in keep.items():
            setattr(G, k, v)


def _emit_shared(G, fmt, mt, kb, B, out, outdir, kind, rowgrp):
    return _emit_gen(G.gen, G.TOK * G.NW, fmt, mt, kb, B, out, outdir, kind, rowgrp)


def _emit_gen(gen, tile, fmt, mt, kb, B, out, outdir, kind, rowgrp):
    if B % tile:
        return None
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    # Not named yah_ffn_gemm_*: E.emit applies the chain/widen/epilogue rewrites
    # to that prefix, and this source is already in its final form.
    src = os.path.join(tmp, "yah_sgemm_%s_%s.loom" % (fmt, kind))
    with open(src, "w") as fh:
        fh.write(gen(fmt, kind))
    sym = "yah_ffn_gemm_%s%s" % (fmt, {"swiglu": "_swiglu", "kres": "_kres", "kqg": "_kqg"}.get(kind, ""))
    # Refuse before emitting if any declared operand footprint exceeds the buffer
    # the driver binds (tools/footprint_gate.py): a silent overrun hangs the ring.
    import subprocess
    gate = subprocess.run([sys.executable, os.path.join(HERE, "footprint_gate.py"), src, sym, fmt,
                           kind, str(mt), str(kb), str(B // tile), str(B)],
                          capture_output=True, text=True)
    if gate.returncode != 0:
        raise SystemExit("footprint gate refused %s: %s" % (out, (gate.stdout + gate.stderr).strip()[-400:]))
    E.emit(src, ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb),
                 "%s.token_tiles=%d" % (sym, B // tile)], out, outdir)
    return (out, tile, rowgrp, B // tile)


def main():
    model, outdir = sys.argv[1], sys.argv[2]
    B = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("YAH_PP_TOKENS", "2048"))
    TILE = int(os.environ.get("YAH_TOKEN_TILE", "64"))
    # YAH_GEMM_W64=1 (with YAH_ROWGRP) rebuilds the GEMM sources the wave64/row-
    # group chain can address as the measured-best arm; the rest keep the
    # shipping wave32 geometry. dispatch.txt records which is which.
    CHAIN = os.environ.get("YAH_GEMM_W64") == "1"
    # YAH_WIDEN_ALL=1 decouples the token-tile widening from the chain. MEASURED
    # BROKEN -- do not ship, and do not assume the coupling below is incidental.
    # widen_tokens itself is fine: it completes on every FFN source (iq3xxs/iq4xs/
    # q5k included) and its output IR is textually complete (tokens 64->128, 8
    # accumulators, 8 rhs loads, the copy-out rescaled to >>7 / &127), and all 64
    # widened HALs compile. But the forward is WRONG: B=2048 argmax is 220 with
    # every family widened, and still 220 when ONLY gemm_swiglu_iq3xxs_1088_20 is
    # widened (1 file), 220 for kstore-only, 220 for residual-only, 198 for the
    # non-FFN formats -- against argmax 11751 for the untouched shipped set, run
    # interleaved on the same binary. So the coupling is load-bearing: the widened
    # UNCHAINED kernels are not usable, and the root cause is not yet established
    # (it is NOT a missing rewrite in widen_tokens; register pressure at wave32
    # with 8 live accumulators, or a layout the chain's later steps normally fix,
    # both remain open). Default OFF: unset, the emitted set is byte-identical.
    WIDEN_ALL = os.environ.get("YAH_WIDEN_ALL") == "1"
    LEVEL = os.environ.get("YAH_CHAIN_LEVEL", "full")
    ROWGRP = int(os.environ.get("YAH_ROWGRP", "4"))
    KSPLIT = int(os.environ.get("YAH_KSPLIT", "4"))
    if B % TILE:
        raise SystemExit("tokens=%d must be a multiple of the token tile=%d" % (B, TILE))
    E.TOKEN_TILE = TILE
    E.WIDEN_TILE = TILE
    TT = B // TILE
    # Chunked prefill: YAH_CTX=T (> B) emits every kernel at the chunk size B,
    # the KV cache (rope's max_context, attention's cache_capacity, the V^T
    # transpose) at T, and one rope / attention HAL per chunk i (start_pos = i B):
    # rope_c<i>.hal / wmma_c<i>.hal (rope.hal / wmma.hal are chunk 0). The
    # driver runs the T tokens in T / B passes over the 64 layers, carrying the
    # DeltaNet and conv states. dispatch.txt row "ctx" records T.
    T = int(os.environ.get("YAH_CTX", str(B)))
    if T % B:
        raise SystemExit("YAH_CTX must be a multiple of the chunk size")
    NCH = T // B
    KC = T * 4 * 256          # kv heads x head dim, i.e. max_context rows
    os.makedirs(outdir, exist_ok=True)
    rows = E.parse(model)
    combos = set()
    for nm, dims, ty in rows:
        suffix = nm.split(".", 2)[2] if nm.startswith("blk.") else nm
        info = E.FMT.get(ty)
        if not info:
            continue
        fmt, port, qk = info
        if suffix in E.KSTORE:
            kind = "kstore"
        elif suffix in E.RESIDUAL:
            kind = "residual"
        elif suffix in E.SWIGLU:
            kind = "swiglu"
        else:
            continue
        combos.add((kind, fmt, port, dims[1] // 16, dims[0] // qk))

    n = 0
    geom = []  # (<hal>, <tokens per workgroup>, <row groups>, <token_tiles>)
    for kind, fmt, port, mt, kb in sorted(combos):
        if kind == "kstore":
            f = "yah_ffn_gemm_%s_f32.loom" % port
        elif kind == "residual":
            f = "yah_ffn_gemm_%s_residual_f32.loom" % port
        else:
            f = "yah_ffn_gemm_%s_swiglu_f16.loom" % port
        # Only the sources the chain can rebuild get the wider tile. chain_applies
        # runs the real transform and fails loudly on an anchor it does not know,
        # so the tile this emitter binds and the grid the driver passes cannot
        # drift apart by accident.
        # mt % rowgrp == 0 or the row-grouped grid truncates: the iq3s family
        # contains a 48-row weight (m_tiles=3), where grid x would become 0.
        # YAH_CHAIN_LEVEL=w64 emits each source through the wave64 port only. The
        # row group needs widen_rows, which needs the iq3s lane map, so rowgrp stays
        # 1 at that level; the tile still widens because widen_tokens is
        # format-agnostic. chain_applies() probes the level actually selected,
        # because _chain() is what it runs and _chain() honours the same env var.
        # full chain where it applies, wave64-only for the rest at level=w64.
        # The two probes must stay separate: a source that supports the full chain
        # has to KEEP its row groups even when the level asks for wave64, or the
        # emit silently regresses the already-chained iq3s HALs.
        full = ((E.chain_applies(f, tile=TILE, level="full") if CHAIN else False)
                and mt % ROWGRP == 0)
        # The reduced levels are a FALLBACK for sources that cannot take the full
        # chain: a source that can must keep it, or the emit silently drops the
        # row groups it already had.
        alt_level = LEVEL if LEVEL in ("rows", "w64") else None
        alt = (E.chain_applies(f, tile=TILE, level=alt_level)
               if (CHAIN and alt_level) else False)
        # 'rows' applies widen_rows, so it divides the x grid by rowgrp exactly
        # like the full chain and needs the same divisibility guard. Without it
        # mt=3 (m_tiles=3) becomes m_groups = 3/4 = 0 and the ostage fragment
        # store cannot prove its bound: "vector_extent is 16, view_bound is 48,
        # and the maximum legal origin is 32".
        if alt_level == "rows":
            alt = alt and mt % ROWGRP == 0
        chain_level = "full" if full else (alt_level if alt else None)
        use = chain_level is not None
        # widenable: every source widen_tokens can widen, chained or not. The row
        # group still requires the chain (widen_rows), so rowgrp stays 1 without it.
        tile = TILE if (use or WIDEN_ALL) else 64
        tt = B // tile
        # 'rows' applies widen_rows, so it keeps the row-group grid; 'w64' does not.
        rowgrp = ROWGRP if chain_level in ("full", "rows") else 1
        sym = E.sym_of(f)
        if kind == "kstore":
            sg = shared_kstore(fmt, mt, kb, B, "gemm_kstore_%s_%d_%d.hal" % (fmt, mt, kb), outdir)
            if sg:
                geom.append(sg)
                # the attention q projection (12288 rows = 24 heads x [q|gate]):
                # also the variant that stores q and gate unpacked
                # (loom_forward_pp prefers it and skips yah_unpack_qg)
                if mt == 768 and os.environ.get("YAH_KQG", "1") == "1":
                    qgv = shared_kstore(fmt, mt, kb, B, "gemm_kqg_%s_%d_%d.hal" % (fmt, mt, kb),
                                        outdir, kind="kqg")
                    if qgv:
                        geom.append(qgv)
                n += 1
                continue
            cfg = ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb),
                   "%s.token_tiles=%d" % (sym, tt)]
            # The chain drops the runtime word_decode switch outright (it keeps
            # the word body), so the binding must go too or the compile rejects
            # it as an unused config.
            if port == "iq3s" and not use:
                cfg.append("%s.word_decode=%d" % (sym, 0 if mt == 1088 else 1))
            out = "gemm_kstore_%s_%d_%d.hal" % (fmt, mt, kb)
        elif kind == "residual":
            cfg = ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb),
                   "%s.token_tiles=%d" % (sym, tt),
                   "%s.k_split=%d" % (sym, KSPLIT), "%s.accum=0" % sym]
            out = "gemm_residual_%s_%d_%d.hal" % (fmt, mt, kb)
        else:
            out = "gemm_swiglu_%s_%d_%d.hal" % (fmt, mt, kb)
            sg = shared_kstore(fmt, mt, kb, B, out, outdir, kind="swiglu")
            if sg:
                geom.append(sg)
                n += 1
                continue
            cfg = ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb),
                   "%s.token_tiles=%d" % (sym, tt)]
        E.emit(f, cfg, out, outdir, widen=tile, chain=use,
               chain_level=(chain_level or "full"))
        geom.append((out, tile, rowgrp, tt))
        # YAH_KSTORE_RESIDUAL=1 additionally emits the CHAINED kStore HAL for
        # every residual shape, so a driver can run those projections on the fast
        # decode instead of the residual source (whose decode widen_rows cannot
        # address). Additive and opt-in: with the env var unset the emitted set is
        # byte-identical to before. The residual HAL is still emitted either way,
        # and the driver loads by exact name, so nothing changes until the driver
        # is taught to ask for the kStore variant.
        if kind == "residual" and os.environ.get("YAH_KSTORE_RESIDUAL") != "off":
            kf = "yah_ffn_gemm_%s_f32.loom" % port
            kfull = E.chain_applies(kf, tile=TILE, level="full") and mt % ROWGRP == 0
            kalt = (E.chain_applies(kf, tile=TILE, level=alt_level)
                    if alt_level else False)
            if alt_level == "rows":
                kalt = kalt and mt % ROWGRP == 0
            kchain_level = "full" if kfull else (alt_level if kalt else None)
            kuse = kchain_level is not None
            ktile = TILE if (kuse or WIDEN_ALL) else 64
            ktt = B // ktile
            krowgrp = ROWGRP if kchain_level in ("full", "rows") else 1
            ksym = E.sym_of(kf)
            kcfg = ["%s.m_tiles=%d" % (ksym, mt), "%s.k_blocks=%d" % (ksym, kb),
                    "%s.token_tiles=%d" % (ksym, ktt)]
            if port == "iq3s" and not kuse:
                kcfg.append("%s.word_decode=%d" % (ksym, 1))
            kout = "gemm_kstore_%s_%d_%d.hal" % (fmt, mt, kb)
            sg = shared_kstore(fmt, mt, kb, B, kout, outdir)
            if sg:
                geom.append(sg)
                # the fused-residual variant (loom_forward_pp prefers it when present)
                kr = shared_kstore(fmt, mt, kb, B, "gemm_kres_%s_%d_%d.hal" % (fmt, mt, kb),
                                   outdir, kind="kres")
                if kr:
                    geom.append(kr)
            else:
                E.emit(kf, kcfg, kout, outdir, widen=ktile, chain=kuse,
                       chain_level=(kchain_level or "full"))
                geom.append((kout, ktile, krowgrp, ktt))
        n += 1

    # Attention: tools/gen_attn_heads.py with H query heads of one GQA group per
    # workgroup sharing each K/V tile (bit-identical to yah_attn_wmma_qb.loom).
    # The driver reads H from this row's row-group field. YAH_ATTN_HEADS=0 keeps
    # the hand-written kernel (one head per workgroup).
    attn_heads = int(os.environ.get("YAH_ATTN_HEADS", "3"))
    attn_src = "yah_attn_wmma_qb.loom"
    # YAH_ATTN_HIP=1: tools/gen_attn_hip.py, bit-identical to HIP's
    # WmmaCausalAttention<32, 16, true> (32 tokens x 2 heads per workgroup),
    # reading V^T written per layer by yah_transpose_v16 (vtrans.hal row).
    attn_hip = os.environ.get("YAH_ATTN_HIP", "1") == "1"
    vtrans_src = None
    kq8_on = False
    vq8_on = False
    vq4_on = False
    if attn_hip:
        os.environ.setdefault("YAH_ATTN_MAX_TOKENS", str(max(B, 2048)))
        import gen_attn_hip
        tmp = os.path.join(outdir, ".emit_tmp")
        os.makedirs(tmp, exist_ok=True)
        attn_src = os.path.join(tmp, "yah_attn_hip.loom")
        with open(attn_src, "w") as fh:
            # tools/gen_attn_fa.py by default (register softmax, not HIP's
            # arithmetic order; accepted 2026-10-01 on kl_p999 / flips / PPL,
            # see gate/README.md), same grid, bindings and f16 output.
            # YAH_ATTN_FA=0: the HIP-order kernel (bit-identical to HIP).
            if os.environ.get("YAH_ATTN_FA", "1") == "1":
                import gen_attn_fa
                fh.write(gen_attn_fa.gen())
            else:
                fh.write(gen_attn_hip.gen())
        vtrans_src = os.path.join(tmp, "yah_transpose_v16.loom")
        with open(vtrans_src, "w") as fh:
            fh.write(gen_attn_hip.gen_vtrans())
        fa_gqa = (os.environ.get("YAH_ATTN_FA", "1") == "1"
                  and os.environ.get("YAH_ATTN_FA_GQA", "0") == "1")
        if fa_gqa:
            # GQA-packed FA: 6 heads x 16 tokens, 384-thread workgroups
            geom.append(("wmma.hal", 16, 6, (B + 15) // 16))
            geom.append(("attn_wg384", 0, 0, 0))
        else:
            geom.append(("wmma.hal", 32, 2, (B + 31) // 32))
        # Quantized K (engine/run/kvq/README.md): YAH_ATTN_FA_K8=1 (kv8a16) int8 K
        # by yah_kmean + yah_kq8; YAH_ATTN_FA_K4=1 (kv4a16) H256 + asymmetric
        # int4 K by yah_kmean + yah_kq4. The attention decodes either to f16.
        kq4_mode = os.environ.get("YAH_ATTN_FA_K4", "0") == "1"
        kq8_on = (os.environ.get("YAH_ATTN_FA", "1") == "1"
                  and (os.environ.get("YAH_ATTN_FA_K8", "0") == "1" or kq4_mode))
        if kq8_on:
            import gen_kvq
            kmean_src = os.path.join(tmp, "yah_kmean.loom")
            kq8_src = os.path.join(tmp, "yah_kq8.loom")
            open(kmean_src, "w").write(gen_kvq.gen_kmean())
            open(kq8_src, "w").write(gen_kvq.gen_kq4() if kq4_mode else gen_kvq.gen_kq8())
            geom.append(("attn_kq4" if kq4_mode else "attn_kq8", 0, 0, 0))
        # YAH_ATTN_FA_VQ8=1 (kv8a16) / YAH_ATTN_FA_VQ4=1 (kv4a16): V^T as bytes /
        # nibbles per channel per 16-key tile + (S, C') by yah_vq8 / yah_vq4
        # instead of the f16 transpose (one HAL per chunk: start_pos)
        vq4_on = (os.environ.get("YAH_ATTN_FA", "1") == "1"
                  and os.environ.get("YAH_ATTN_FA_VQ4", "0") == "1")
        if vq4_on:
            import gen_kvq
            vq4_src = os.path.join(tmp, "yah_vq4.loom")
            open(vq4_src, "w").write(gen_kvq.gen_vq4())
            geom.append(("attn_vq4", 0, 0, 0))
        vq8_on = (os.environ.get("YAH_ATTN_FA", "1") == "1"
                  and os.environ.get("YAH_ATTN_FA_VQ8", "0") == "1")
        if vq8_on:
            import gen_kvq
            vq8_src = os.path.join(tmp, "yah_vq8.loom")
            open(vq8_src, "w").write(gen_kvq.gen_vq8())
            geom.append(("attn_vq8", 0, 0, 0))
        if gen_attn_hip.F16OUT:
            # marker: the attention HAL stores its output as f16 (no half_cast)
            geom.append(("attn_f16out", 0, 0, 0))
        geom.append(("vtrans.hal", 0, 0, 0))
    elif attn_heads:
        import gen_attn_heads
        tmp = os.path.join(outdir, ".emit_tmp")
        os.makedirs(tmp, exist_ok=True)
        attn_src = os.path.join(tmp, "yah_attn_wmma_h%d.loom" % attn_heads)
        with open(attn_src, "w") as fh:
            fh.write(gen_attn_heads.gen(attn_heads))
        geom.append(("wmma.hal", 16, attn_heads, (B + 15) // 16))

    # DeltaNet: tools/gen_deltanet_hip.py, bit-identical to HIP's
    # BatchedDeltaNetRowSplitKernel<float,16,2> (tools/deltanet_vs_hip.sh), grid
    # (2, heads) recorded as the rowsplit.hal row group. YAH_DELTANET_HIP=0 keeps
    # the regtile kernel (Loom's own sequential-sum order).
    dn_src = "yah_deltanet_rowsplit_f32.loom"
    if os.environ.get("YAH_DELTANET_HIP", "1") != "0":
        import gen_deltanet_hip
        tmp = os.path.join(outdir, ".emit_tmp")
        os.makedirs(tmp, exist_ok=True)
        dn_src = os.path.join(tmp, "yah_deltanet_hip_f32.loom")
        # YAH_DN_CHUNK=1 (default): chunked WY Gated DeltaNet (tools/gen_gdn_chunk.py,
        # engine/run/research/gdn): same ABI and (2, heads) grid, f16 WMMA inputs
        # (output rel 2.1e-4 vs this kernel, end-to-end KLD ~3e-6), standalone
        # pp2048 3.57 vs 4.12 M cycles. Needs B % 32 == 0 (else the recurrent kernel).
        chunk_dn = os.environ.get("YAH_DN_CHUNK", "1") == "1" and B % 32 == 0
        if chunk_dn:
            import gen_gdn_chunk
        with open(dn_src, "w") as fh:
            fh.write(gen_gdn_chunk.gen() if chunk_dn else gen_deltanet_hip.gen())
        geom.append(("rowsplit.hal", 0, 2, 0))

    # Record the resolved launch geometry with the prepared executables.
    # loom_forward_pp reads this instead of recomputing the grid, so the dispatch
    # site and the compiled kernel cannot disagree (see tools/emit_prefill.py,
    # chain=). A mismatch is silent and wrong, not a crash.
    # Quantized K and V: attention never reads the f16 KV cache, so it becomes
    # a one-layer, one-chunk scratch (RoPE writes row cur - cache_start; the
    # quantizers read it right after): the driver sizes kv16 by this marker.
    kv16_scratch = kq8_on and (vq4_on or vq8_on)
    if kv16_scratch:
        geom.append(("kv16_scratch", 0, 0, 0))
    if NCH > 1:
        # quantized KV: K quantizers run per chunk on cache slices (kmean on
        # chunk 0 only), V quantizers per chunk with start_pos (vq*_c<i>.hal)
        geom.append(("ctx", B, 0, T))     # chunk size, total context
    with open(os.path.join(outdir, "dispatch.txt"), "w") as fh:
        for hal, tk, rg, tt in geom:
            fh.write("%s %d %d %d\n" % (hal, tk, rg, tt))

    # The output norm is the fully unrolled fused=0 form (tools/gen_half_norm.py):
    # same arithmetic in the same order, all loads issued up front; 0.54 -> 0.33
    # ms per 2048-row call, bit-identical. YAH_NORM_UNROLLED=0 keeps the loop form.
    norm_src = "yah_half_norm_f16.loom"
    if os.environ.get("YAH_NORM_UNROLLED", "1") != "0":
        import gen_half_norm
        tmp = os.path.join(outdir, ".emit_tmp")
        os.makedirs(tmp, exist_ok=True)
        norm_src = os.path.join(tmp, "yah_half_norm_unrolled.loom")
        with open(norm_src, "w") as fh:
            fh.write(gen_half_norm.gen(5120))
    fixed = [
        ("yah_residual_add_1d_f32.loom", "accum.hal",
         ["yah_residual_1d.dim=%d" % (5120 * B)]),
        (norm_src, "norm.hal",
         ["yah_half_norm.rows=%d" % B, "yah_half_norm.dim=5120",
          "yah_half_norm.eps=1e-06", "yah_half_norm.fused=0"]),
        ("yah_ssm_conv_f32.loom", "conv.hal",
         ["yah_ssm_conv.batch=%d" % B, "yah_ssm_conv.qkv_dim=10240"]),
        # conv with yah_deltanet_prep_kq fused in (the driver prefers it)
        ("yah_ssm_conv_kq_f32.loom", "convkq.hal",
         ["yah_ssm_conv_kq.batch=%d" % B, "yah_ssm_conv_kq.qkv_dim=10240",
          "yah_ssm_conv_kq.num_key_heads=16"]),
        ("yah_deltanet_prep_kq_f32.loom", "prepkq.hal",
         ["yah_deltanet_prep_kq.batch=%d" % B,
          "yah_deltanet_prep_kq.num_key_heads=16",
          "yah_deltanet_prep_kq.qkv_size=10240"]),
        ("yah_deltanet_prep_ab_f32.loom", "prepab.hal",
         ["yah_deltanet_prep_ab.batch=%d" % B,
          "yah_deltanet_prep_ab.qkv_size=10240",
          "yah_deltanet_prep_ab.num_heads=48"]),
        (dn_src, "rowsplit.hal",
         ["yah_deltanet.batch=%d" % B, "yah_deltanet.qkv_size=10240",
          "yah_deltanet.inner_size=6144", "yah_deltanet.num_key_heads=16",
          "yah_deltanet.num_heads=48"]),
        ("yah_ssm_postnorm_gate_f16.loom", "postnorm.hal",
         ["yah_ssm_postnorm_fp16.head_count=%d" % (48 * B)]),
        ("yah_unpack_qg_f32.loom", "unpack.hal",
         ["yah_unpack_qg.batch=%d" % B, "yah_unpack_qg.num_heads=24",
          "yah_unpack_qg.head_dim=256"]),
        *[("yah_fused_qk_rope_batched_f32.loom", "rope.hal" if c == 0 else "rope_c%d.hal" % c, [
            "yah_fused_qk_rope_batched.start_pos=%d" % (c * B),
            "yah_fused_qk_rope_batched.batch=%d" % B,
            "yah_fused_qk_rope_batched.layer_idx=0",
            "yah_fused_qk_rope_batched.max_context=%d" % T,
            "yah_fused_qk_rope_batched.num_heads=24",
            "yah_fused_qk_rope_batched.num_kv_heads=4",
            "yah_fused_qk_rope_batched.head_dim=256",
            "yah_fused_qk_rope_batched.rotary_dim=64",
            "yah_fused_qk_rope_batched.q_elems=%d" % (6144 * B),
            "yah_fused_qk_rope_batched.kv_elems=%d" % (1024 * B),
            "yah_fused_qk_rope_batched.cache32_elems=%d" % KC,
            "yah_fused_qk_rope_batched.cache16_elems=%d" % (1024 * B if kv16_scratch else KC),
            *(["yah_fused_qk_rope_batched.cache_start=%d" % (c * B)] if kv16_scratch else [])]) for c in range(NCH)],
        *[(attn_src, "wmma.hal" if c == 0 else "wmma_c%d.hal" % c, [
            "attention_prefill.cache_capacity=%d" % T,
            "attention_prefill.token_count=%d" % B,
            "attention_prefill.start_pos=%d" % (c * B),
            "attention_prefill.num_heads=24", "attention_prefill.num_kv_heads=4",
            "attention_prefill.head_dim=256", "attention_prefill.gqa=6"]) for c in range(NCH)],
        *([(vtrans_src, "vtrans.hal",
            ["yah_vtrans.token_count=%d" % T, "yah_vtrans.cache_capacity=%d" % T])]
          if vtrans_src else []),
        *([(kmean_src, "kmean.hal", ["yah_kvq.token_count=%d" % B, "yah_kvq.cache_capacity=%d" % B]),
           (kq8_src, "kq8.hal", ["yah_kvq.token_count=%d" % B, "yah_kvq.cache_capacity=%d" % B])]
          if kq8_on else []),
        *([(vq8_src, "vq8.hal" if c == 0 else "vq8_c%d.hal" % c,
            ["yah_kvq.token_count=%d" % B, "yah_kvq.cache_capacity=%d" % T, "yah_kvq.start_pos=%d" % (c * B)])
           for c in range(NCH)] if vq8_on else []),
        *([(vq4_src, "vq4.hal" if c == 0 else "vq4_c%d.hal" % c,
            ["yah_kvq.token_count=%d" % B, "yah_kvq.cache_capacity=%d" % T, "yah_kvq.start_pos=%d" % (c * B)])
           for c in range(NCH)]
          if vq4_on else []),
        *([("yah_conv_state_f32.loom", "convstate.hal",
            ["yah_conv_state.batch=%d" % B, "yah_conv_state.channels=10240"])] if NCH > 1 else []),
        ("yah_half_cast.loom", "cast.hal",
         ["yah_half_cast.num_elements=%d" % (6144 * B)]),
        ("yah_rmsnorm_f32.loom", "rmsnorm.hal",
         ["yah_rmsnorm.rows=1", "yah_rmsnorm.eps=1e-06"]),
        ("yah_gemv_q6k_f32.loom", "gemv.hal",
         ["yah_gemv_q6k.m_rows=248320", "yah_gemv_q6k.k_blocks=20"]),
        ("yah_argmax_f32.loom", "argmax.hal", ["yah_argmax.vocab=248320"]),
    ]
    for loom, outname, configs in fixed:
        E.emit(loom, configs, outname, outdir)
        n += 1

    tables = os.path.join(E.LOOM, "tables")
    for src, dst in [("grid_iq3s.bin", "grid_iq3s.bin"),
                     ("grid_iq3xxs.bin", "grid_iq3xxs.bin"),
                     ("grid_iq2xxs.bin", "grid_iq2xxs.bin"),
                     ("grid_iq2xs.bin", "grid_iq2xs.bin"),
                     ("ksigns_iq2xs.bin", "ksigns_iq3xxs.bin"),
                     ("ksigns_iq2xs.bin", "ksigns_iq2xxs.bin")]:
        shutil.copy(os.path.join(tables, src), os.path.join(outdir, dst))
    print("emitted %d GEMM + fixed prefill HALs for B=%d (tile=%d, token_tiles=%d)"
          % (n, B, TILE, TT))


if __name__ == "__main__":
    main()
