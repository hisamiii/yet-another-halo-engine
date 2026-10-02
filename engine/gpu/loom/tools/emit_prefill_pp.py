#!/usr/bin/env python3
"""Emit the full-prompt (B-token) Loom prefill HAL set.

Same naming convention as emit_prefill.py, but *every* token dimension is bound
to B instead of the 5-token batch:

  GEMM family      token_tiles = B / tile              (grid y)
  fixed kernels    batch = B
  attention        max_context = score_capacity = B, KV cache B deep
  residual reduce  dim = 5120 * B                      (the whole prompt)

The whole prompt goes through in ONE pass, so start_pos stays 0 everywhere, the
KV cache is written once, and the recurrent state (conv ring, DeltaNet state)
starts zeroed -- exactly the first-chunk case the 5-token path already
validates. No token-tile loop, and therefore no per-tile launch overhead: the
GEMM grid carries the token dimension the same way the benchmark arms do.

usage: emit_prefill_pp.py <model.gguf> <outdir> [tokens]   (default 2048 tokens)
"""
import os, re, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emit_prefill as E  # noqa: E402
import gen_attn_fa  # noqa: E402
import gen_deltanet_hip  # noqa: E402
import gen_gdn_chunk  # noqa: E402
import gen_half_norm  # noqa: E402
import gen_kvq  # noqa: E402


def rope_kpaged(text):
    """yah_fused_qk_rope_batched -> paged-K variant: K rows go straight to the
    paged K pool (row ptab[cur / 256] * 256 + cur % 256, k16_elems elements,
    page index clamped into the pool); V keeps writing the one-chunk scratch for
    yah_vtpage (KV paging)."""
    def r(a, b):
        nonlocal text
        assert text.count(a) == 1, a[:60]
        text = text.replace(a, b)
    r("config.decl @yah_fused_qk_rope_batched.cache16_elems : %value: index where [range(%value, 1, 1073741824)]",
      "config.decl @yah_fused_qk_rope_batched.cache16_elems : %value: index where [range(%value, 1, 1073741824)]\n"
      "config.decl @yah_fused_qk_rope_batched.k16_elems : %value: index where [range(%value, 1, 1073741824)]")
    r("%k_cache_f16: buffer, %v_cache_f16: buffer, %eps_buf: buffer) {",
      "%k_cache_f16: buffer, %v_cache_f16: buffer, %eps_buf: buffer, %ptab: buffer) {")
    r("  %cache16_elems = config.get @yah_fused_qk_rope_batched.cache16_elems : index\n",
      "  %cache16_elems = config.get @yah_fused_qk_rope_batched.cache16_elems : index\n"
      "  %k16_elems = config.get @yah_fused_qk_rope_batched.k16_elems : index\n")
    r("  %k16_view = buffer.view %k16_na[%base] : buffer -> view<[%cache16_elems]xf16>",
      "  %k16_view = buffer.view %k16_na[%base] : buffer -> view<[%k16_elems]xf16>")
    r("  %cur = index.min %cur_nn, %ctx_m1 : index\n",
      "  %cur = index.min %cur_nn, %ctx_m1 : index\n"
      "  %pg256 = index.constant 256 : index\n"
      "  %pg255 = index.constant 255 : index\n"
      "  %pgrows = index.div %k16_elems, %kv_width : index\n"
      "  %pgn0 = index.add %pgrows, %pg255 : index\n"
      "  %pgnp = index.div %pgn0, %pg256 : index\n"
      "  %pt_view = buffer.view %ptab[%base] : buffer -> view<[%pgnp]xi32>\n"
      "  %pglp = index.div %cur, %pg256 : index\n"
      "  %pglp1 = index.sub %pgnp, %c1 : index\n"
      "  %pglpc = index.min %pglp, %pglp1 : index\n"
      "  %pgv = view.load %pt_view[%pglpc] : view<[%pgnp]xi32> -> i32\n"
      "  %pgu = index.cast %pgv : i32 to index\n"
      "  %pgz = index.max %pgu, %c0 : index\n"
      "  %pg = index.min %pgz, %pglp1 : index\n"           # clamp into the pool (both sides)
      "  %pgb = index.mul %pg, %pg256 : index\n"
      "  %pgo = index.rem %cur, %pg256 : index\n"
      "  %krow = index.add %pgb, %pgo : index\n")
    r("    %f16_off = index.min %f16_off_raw, %cache16_max : index\n",
      "    %f16_off = index.min %f16_off_raw, %cache16_max : index\n"
      "    %k16_pre = index.mul %krow, %kv_width : index\n"
      "    %k16_raw = index.add %k16_pre, %hbkq : index\n"
      "    %k16_max = index.sub %k16_elems, %head_dim : index\n"
      "    %k16_base = index.min %k16_raw, %k16_max : index\n")
    for a in ("%k16_off0 = index.add %f16_off, %p3 :", "%k16_off1 = index.add %f16_off, %p3b :", "%k16_off3 = index.add %f16_off, %i3 :"):
        r(a, a.replace("%f16_off", "%k16_base"))
    text, n = re.subn(r"(%k16_view\[[^\]]*\] : f16, )view<\[%cache16_elems\]xf16>", r"\1view<[%k16_elems]xf16>", text)
    assert n == 3, n
    return text


def shared_kstore(fmt, mt, kb, B, out, outdir, kind="kstore"):
    """Emit the shared-decode kStore (tools/gen_gemm_shared.py) for this shape if
    it covers the format, and return its dispatch.txt geometry, else None.

    It replaces the hand-written kStore: same ABI and output, bit-identical, with
    the decoded weight tile shared by NW waves, a prefetched branch-free decode
    and a direct token-major epilogue.
    """
    import gen_gemm_shared as G
    if fmt not in G.FMTS:
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
    shape if it covers it, and return its dispatch.txt geometry, else None."""
    import gen_gemm_tile as TG
    if fmt not in TILE_FMTS:
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
    # decode-ahead lost on this one shape in two paired pp2048 profiles
    # (IQ4_XS kres 5120 x 6144: 88.5 -> 93.7 / 95.5 ms); it keeps KSUB=64
    decahead = (fmt, kind, kb) not in DECAHEAD_SKIP
    return _tile_emit(TG, fmt, mt, kb, B, out, outdir, kind, tile, rowgrp, decahead)


# Short-K residual GEMMs: with decode-ahead, 45% of wave time is s_waitcnt
# vmcnt(0) in the K loop (ATT, Q4_K kres K=6144), full drains that serialize
# the read-ahead. Q4_K kres K=6144 11.34 -> 9.51 M cycles off (bit-identical).
DECAHEAD_SKIP = {("iq4xs", "kres", 24), ("q4k", "kres", 24)}


def _tile_emit(TG, fmt, mt, kb, B, out, outdir, kind, tile, rowgrp, decahead):
    # TG.configure() rewrites gen_gemm_shared's module globals (NW, LR, KSUB...)
    # to drive the shared decode helpers; put them back for the kernels the
    # shared generator still emits in this process.
    G = TG.G
    keep = {k: getattr(G, k) for k in ("KSUB", "PAD", "ROWP", "PH", "GPP", "GPL", "LR", "NW")}
    try:
        return _emit_gen(lambda f, k: TG.gen(f, k, decahead), tile, fmt, mt, kb, B, out, outdir, kind, rowgrp)
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
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    # GEMM tokens per workgroup of the hand-written sources
    TILE = 64
    if B % TILE:
        raise SystemExit("tokens=%d must be a multiple of the token tile=%d" % (B, TILE))
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
        # The residual projections run as a kStore plus the fused-residual kres
        # variant (loom_forward_pp prefers kres when present).
        if kind in ("kstore", "residual"):
            f = "yah_ffn_gemm_%s_f32.loom" % port
        else:
            f = "yah_ffn_gemm_%s_swiglu_f16.loom" % port
        sym = E.sym_of(f)
        if kind == "residual":
            kout = "gemm_kstore_%s_%d_%d.hal" % (fmt, mt, kb)
            sg = shared_kstore(fmt, mt, kb, B, kout, outdir)
            if sg:
                geom.append(sg)
                kr = shared_kstore(fmt, mt, kb, B, "gemm_kres_%s_%d_%d.hal" % (fmt, mt, kb),
                                   outdir, kind="kres")
                if kr:
                    geom.append(kr)
                n += 1
                continue
            cfg = ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb),
                   "%s.token_tiles=%d" % (sym, TT)]
            if port == "iq3s":
                cfg.append("%s.word_decode=1" % sym)
            out = kout
        elif kind == "kstore":
            sg = shared_kstore(fmt, mt, kb, B, "gemm_kstore_%s_%d_%d.hal" % (fmt, mt, kb), outdir)
            if sg:
                geom.append(sg)
                # the attention q projection (12288 rows = 24 heads x [q|gate]):
                # also the variant that stores q and gate unpacked
                # (loom_forward_pp prefers it and skips yah_unpack_qg)
                if mt == 768:
                    qgv = shared_kstore(fmt, mt, kb, B, "gemm_kqg_%s_%d_%d.hal" % (fmt, mt, kb),
                                        outdir, kind="kqg")
                    if qgv:
                        geom.append(qgv)
                n += 1
                continue
            cfg = ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb),
                   "%s.token_tiles=%d" % (sym, TT)]
            if port == "iq3s":
                cfg.append("%s.word_decode=%d" % (sym, 0 if mt == 1088 else 1))
            out = "gemm_kstore_%s_%d_%d.hal" % (fmt, mt, kb)
        else:
            out = "gemm_swiglu_%s_%d_%d.hal" % (fmt, mt, kb)
            sg = shared_kstore(fmt, mt, kb, B, out, outdir, kind="swiglu")
            if sg:
                geom.append(sg)
                n += 1
                continue
            cfg = ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb),
                   "%s.token_tiles=%d" % (sym, TT)]
        E.emit(f, cfg, out, outdir)
        geom.append((out, TILE, 1, TT))
        n += 1

    # Attention: tools/gen_attn_fa.py (register softmax), 32 tokens x 2 heads per
    # workgroup, reading V^T (vtrans.hal, or the paged / quantized V pools).
    # KV paging (256-token pages) needs a context that is a multiple of 256;
    # otherwise the caches stay contiguous.
    kv_paged = T % 256 == 0
    if not kv_paged:
        print("KV paging off: needs the context to be a multiple of 256")
    gen_attn_fa.PAGED = gen_kvq.PAGED = kv_paged
    gen_attn_fa.MAX_TOKENS = max(B, 2048)
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    attn_src = os.path.join(tmp, "yah_attn_hip.loom")
    open(attn_src, "w").write(gen_attn_fa.gen())
    vtrans_src = os.path.join(tmp, "yah_transpose_v16.loom")
    open(vtrans_src, "w").write(gen_attn_fa.gen_vtrans())
    geom.append(("wmma.hal", 32, 2, (B + 31) // 32))
    # Quantized K (engine/run/kvq/README.md): YAH_ATTN_FA_K8=1 (kv8a16) int8 K
    # by yah_kmean + yah_kq8; YAH_ATTN_FA_K4=1 (kv4a16) H256 + asymmetric
    # int4 K by yah_kmean + yah_kq4. The attention decodes either to f16.
    kq4_mode = os.environ.get("YAH_ATTN_FA_K4", "0") == "1"
    kq8_on = os.environ.get("YAH_ATTN_FA_K8", "0") == "1" or kq4_mode
    if kq8_on:
        kmean_src = os.path.join(tmp, "yah_kmean.loom")
        kq8_src = os.path.join(tmp, "yah_kq8.loom")
        open(kmean_src, "w").write(gen_kvq.gen_kmean())
        open(kq8_src, "w").write(gen_kvq.gen_kq4() if kq4_mode else gen_kvq.gen_kq8())
        geom.append(("attn_kq4" if kq4_mode else "attn_kq8", 0, 0, 0))
    # YAH_ATTN_FA_VQ8=1 (kv8a16) / YAH_ATTN_FA_VQ4=1 (kv4a16): V^T as bytes /
    # nibbles per channel per 16-key tile + (S, C') by yah_vq8 / yah_vq4
    # instead of the f16 transpose (one HAL per chunk: start_pos)
    vq4_on = os.environ.get("YAH_ATTN_FA_VQ4", "0") == "1"
    if vq4_on:
        vq4_src = os.path.join(tmp, "yah_vq4.loom")
        open(vq4_src, "w").write(gen_kvq.gen_vq4())
        geom.append(("attn_vq4", 0, 0, 0))
    vq8_on = os.environ.get("YAH_ATTN_FA_VQ8", "0") == "1"
    if vq8_on:
        vq8_src = os.path.join(tmp, "yah_vq8.loom")
        open(vq8_src, "w").write(gen_kvq.gen_vq8())
        geom.append(("attn_vq8", 0, 0, 0))
    # marker: the attention HAL stores its output as f16 (no half_cast)
    geom.append(("attn_f16out", 0, 0, 0))
    geom.append(("vtrans.hal", 0, 0, 0))

    # DeltaNet: chunked WY Gated DeltaNet (tools/gen_gdn_chunk.py), grid
    # (2, heads) recorded as the rowsplit.hal row group, f16 WMMA inputs.
    # It needs B % 32 == 0; otherwise the recurrent kernel
    # (tools/gen_deltanet_hip.py, same ABI and grid).
    dn_src = os.path.join(tmp, "yah_deltanet_hip_f32.loom")
    open(dn_src, "w").write(gen_gdn_chunk.gen() if B % 32 == 0 else gen_deltanet_hip.gen())
    geom.append(("rowsplit.hal", 0, 2, 0))

    # Record the resolved launch geometry with the prepared executables.
    # loom_forward_pp reads this instead of recomputing the grid, so the dispatch
    # site and the compiled kernel cannot disagree (see tools/emit_prefill.py,
    # chain=). A mismatch is silent and wrong, not a crash.
    # Quantized K and V: attention never reads the f16 KV cache, so it becomes
    # a one-layer, one-chunk scratch (RoPE writes row cur - cache_start; the
    # quantizers read it right after): the driver sizes kv16 by this marker.
    # Paged K / V caches (kv_paged, decided above; page table bound to attention
    # and every cache writer). The f16 KV cache is then always a scratch; fp16
    # K / V go to paged pools through the paged RoPE K store / yah_vtpage (per
    # chunk) instead of the V^T re-transpose.
    if kv_paged:
        geom.append(("kv_paged", 0, 0, 0))
    kv16_scratch = kv_paged or (kq8_on and (vq4_on or vq8_on))
    if kv16_scratch:
        geom.append(("kv16_scratch", 0, 0, 0))
    paged_f16k = kv_paged and not kq8_on
    paged_f16v = kv_paged and not (vq4_on or vq8_on)
    rope_src = "yah_fused_qk_rope_batched_f32.loom"
    if paged_f16k:   # RoPE writes K rows straight into the paged pool
        rope_src = os.path.join(tmp, "yah_fused_qk_rope_batched_kpaged.loom")
        open(rope_src, "w").write(rope_kpaged(open(os.path.join(E.LOOM, "yah_fused_qk_rope_batched_f32.loom")).read()))
        geom.append(("rope_kpaged", 0, 0, 0))
    if paged_f16v:
        vtpage_src = os.path.join(tmp, "yah_vtpage.loom")
        open(vtpage_src, "w").write(gen_kvq.gen_vtpage())
        vtrans_src = None    # replaced by the paged per-chunk transpose
    if NCH > 1:
        # quantized KV: K quantizers run per chunk on cache slices (kmean on
        # chunk 0 only), V quantizers per chunk with start_pos (vq*_c<i>.hal)
        geom.append(("ctx", B, 0, T))     # chunk size, total context
    with open(os.path.join(outdir, "dispatch.txt"), "w") as fh:
        for hal, tk, rg, tt in geom:
            fh.write("%s %d %d %d\n" % (hal, tk, rg, tt))

    # The output norm (tools/gen_half_norm.py): fully unrolled, all loads issued up front.
    norm_src = os.path.join(tmp, "yah_half_norm_unrolled.loom")
    open(norm_src, "w").write(gen_half_norm.gen(5120))
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
        *[(rope_src, "rope.hal" if c == 0 else "rope_c%d.hal" % c, [
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
            *(["yah_fused_qk_rope_batched.cache_start=%d" % (c * B)] if kv16_scratch else []),
            *(["yah_fused_qk_rope_batched.k16_elems=%d" % (1024 * T)] if paged_f16k else [])]) for c in range(NCH)],
        *[(attn_src, "wmma.hal" if c == 0 else "wmma_c%d.hal" % c, [
            "attention_prefill.cache_capacity=%d" % T,
            "attention_prefill.token_count=%d" % B,
            "attention_prefill.start_pos=%d" % (c * B),
            "attention_prefill.num_heads=24", "attention_prefill.num_kv_heads=4",
            "attention_prefill.head_dim=256", "attention_prefill.gqa=6"]) for c in range(NCH)],
        *([(vtrans_src, "vtrans.hal",
            ["yah_vtrans.token_count=%d" % T, "yah_vtrans.cache_capacity=%d" % T])]
          if vtrans_src else []),
        *([(kmean_src, "kmean.hal", ["yah_kvq.token_count=%d" % B, "yah_kvq.cache_capacity=%d" % B])]
          if kq8_on else []),
        *([(kq8_src, "kq8.hal", ["yah_kvq.token_count=%d" % B, "yah_kvq.cache_capacity=%d" % B])]
          if kq8_on and not kv_paged else []),
        *([(kq8_src, "kq8.hal" if c == 0 else "kq8_c%d.hal" % c,
            ["yah_kvq.token_count=%d" % B, "yah_kvq.cache_capacity=%d" % B,
             "yah_kvq.start_pos=%d" % (c * B), "yah_kvq.pool_rows=%d" % T]) for c in range(NCH)]
          if kq8_on and kv_paged else []),
        *([(vtpage_src, "vtpage.hal" if c == 0 else "vtpage_c%d.hal" % c,
            ["yah_kvq.token_count=%d" % B, "yah_kvq.start_pos=%d" % (c * B), "yah_kvq.pool_rows=%d" % T])
           for c in range(NCH)] if paged_f16v else []),
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
