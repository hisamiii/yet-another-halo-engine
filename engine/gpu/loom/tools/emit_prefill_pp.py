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
TILE_FMTS = ("iq3s", "iq4xs", "iq3xxs", "q4k", "q5k", "q6k", "iq2xxs", "q3k", "q8_0")


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
    sym = "yah_ffn_gemm_%s%s" % (fmt, {"swiglu": "_swiglu", "kres": "_kres"}.get(kind, ""))
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
    KC = B * 4 * 256          # kv heads x head dim, i.e. max_context rows
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
    if attn_heads:
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
        with open(dn_src, "w") as fh:
            fh.write(gen_deltanet_hip.gen())
        geom.append(("rowsplit.hal", 0, 2, 0))

    # Record the resolved launch geometry with the prepared executables.
    # loom_forward_pp reads this instead of recomputing the grid, so the dispatch
    # site and the compiled kernel cannot disagree (see tools/emit_prefill.py,
    # chain=). A mismatch is silent and wrong, not a crash.
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
        ("yah_fused_qk_rope_batched_f32.loom", "rope.hal", [
            "yah_fused_qk_rope_batched.start_pos=0",
            "yah_fused_qk_rope_batched.batch=%d" % B,
            "yah_fused_qk_rope_batched.layer_idx=0",
            "yah_fused_qk_rope_batched.max_context=%d" % B,
            "yah_fused_qk_rope_batched.num_heads=24",
            "yah_fused_qk_rope_batched.num_kv_heads=4",
            "yah_fused_qk_rope_batched.head_dim=256",
            "yah_fused_qk_rope_batched.rotary_dim=64",
            "yah_fused_qk_rope_batched.q_elems=%d" % (6144 * B),
            "yah_fused_qk_rope_batched.kv_elems=%d" % (1024 * B),
            "yah_fused_qk_rope_batched.cache32_elems=%d" % KC,
            "yah_fused_qk_rope_batched.cache16_elems=%d" % KC]),
        (attn_src, "wmma.hal", [
            "attention_prefill.cache_capacity=%d" % B,
            "attention_prefill.token_count=%d" % B,
            "attention_prefill.start_pos=0",
            "attention_prefill.num_heads=24", "attention_prefill.num_kv_heads=4",
            "attention_prefill.head_dim=256", "attention_prefill.gqa=6"]),
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
