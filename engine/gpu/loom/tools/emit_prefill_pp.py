#!/usr/bin/env python3
"""Emit the prefill HAL set for B-token chunks: every kernel compiled for its shape, plus dispatch.txt.

Every token dimension is bound to B: GEMM token_tiles = B / tile (grid y), fixed kernels batch = B, residual dim = 5120 * B.
YAH_CTX=T (a multiple of B, default B) sizes the KV cache and emits one rope / attention HAL per chunk.
The driver runs T / B chunks and carries the DeltaNet and conv states. YAH_KV selects the KV format (gen_kvq.kv_bits).
dispatch.txt has one row per HAL, "<hal> <tokens per workgroup> <row groups> <token_tiles>", plus mode marker rows.
HAL names follow emit_prefill.py.

usage: emit_prefill_pp.py <model.gguf> <outdir> [tokens]   (default 2048 tokens)
"""
import dataclasses, functools, json, os, re, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emit_prefill as E  # noqa: E402
import gen_attn_fa  # noqa: E402
import gen_deltanet_hip  # noqa: E402
import gen_gdn_chunk  # noqa: E402
import gen_half_norm  # noqa: E402
import gen_kvq  # noqa: E402


def rope_kpaged(text):
    """Rewrite yah_fused_qk_rope_batched into the paged-K variant: K rows go straight to the paged K pool.
    K row = ptab[cur / 256] * 256 + cur % 256, page index clamped into the pool.
    V still writes the one-chunk scratch for yah_vtpage."""
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


def qk_of(fmt):
    return next(qk for f, _, qk in E.FMT.values() if f == fmt)


def gemm(fmt, mt, kb, B, out, outdir, kind="kstore"):
    """Emit the tile GEMM (tools/gen_gemm_tile.py) of this shape and kind; return its dispatch.txt row, or None."""
    r = tile_kstore(fmt, mt, kb, B, out, outdir, kind)
    if not r and mt < 4:
        # the 48-row ssm_alpha/ssm_beta: one 16-row tile per workgroup, 64 tokens over 2 waves, grid (3, B/64)
        r = tile_kstore(fmt, mt, kb, B, out, outdir, kind, geom=(16, 64, 1, 2))
    return r


@functools.cache
def tiles():
    """YAH_TILES=<file.json>: per-GEMM Tile overrides from a tuner, {"gemm_kstore_iq3s_1088_20": {"bn": 512, ...}}."""
    path = os.environ.get("YAH_TILES")
    return json.load(open(path)) if path else {}


# Formats the tile GEMM (tools/gen_gemm_tile.py) is verified bit-identical on in the pp2048 pipeline.
TILE_FMTS = ("iq3s", "iq4xs", "iq3xxs", "q4k", "q5k", "q6k", "iq2xxs", "iq2xs", "q3k", "q8_0")


def tile_kstore(fmt, mt, kb, B, out, outdir, kind, geom=None):
    """Emit the tile GEMM (tools/gen_gemm_tile.py) for this shape if it covers it; return its dispatch.txt row, else None.
    geom=(BM, BN, WM, WN) overrides the workgroup geometry for this one kernel."""
    import gen_gemm_tile as TG
    if fmt not in TILE_FMTS:
        return None
    t = TG.default_tile(fmt, kind, kb, geom)
    if not geom and out[:-4] in tiles():
        t = dataclasses.replace(t, **tiles()[out[:-4]])
    if mt % t.rowgrp:
        return None
    # a token tile that does not divide the chunk: the last tile is masked (gen_gemm_tile.gen masked=)
    masked = B % t.bn != 0
    return _emit_gen(lambda f, k: TG.gen(f, k, t, masked), t.bn, fmt, mt, kb, B, out, outdir, kind, t.rowgrp, masked)


def _emit_gen(gen, tile, fmt, mt, kb, B, out, outdir, kind, rowgrp, masked=False):
    if B % tile and not masked:
        return None
    tt = -(-B // tile)
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(tmp, "yah_sgemm_%s_%s.loom" % (fmt, kind))
    with open(src, "w") as fh:
        fh.write(gen(fmt, kind))
    sym = "yah_ffn_gemm_%s%s" % (fmt, {"swiglu": "_swiglu", "kres": "_kres", "kqg": "_kqg"}.get(kind, ""))
    # Refuse before emitting if any declared operand footprint exceeds the buffer the driver binds: an overrun hangs the ring.
    import subprocess
    gate = subprocess.run([sys.executable, os.path.join(HERE, "footprint_gate.py"), src, sym, fmt,
                           kind, str(mt), str(kb), str(tt), str(B)] + (["masked"] if masked else []),
                          capture_output=True, text=True)
    if gate.returncode != 0:
        raise SystemExit("footprint gate refused %s: %s" % (out, (gate.stdout + gate.stderr).strip()[-400:]))
    E.emit(src, ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb), "%s.token_tiles=%d" % (sym, tt)]
           + (["%s.tokens=%d" % (sym, B)] if masked else []), out, outdir)
    return (out, tile, rowgrp, tt)


def main():
    model, outdir = sys.argv[1], sys.argv[2]
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    # GEMM tokens per workgroup of the hand-written sources
    TILE = 64
    if B % TILE:
        raise SystemExit("tokens=%d must be a multiple of the token tile=%d" % (B, TILE))
    TT = B // TILE
    # Chunked prefill: YAH_CTX=T (> B) emits every kernel at the chunk size B and the KV cache (rope, attention, V^T) at T.
    # One rope / attention HAL per chunk i (start_pos = i * B): rope_c<i>.hal / wmma_c<i>.hal; rope.hal / wmma.hal are chunk 0.
    # The driver runs T / B passes over the 64 layers. dispatch.txt row "ctx" records T.
    T = int(os.environ.get("YAH_CTX", str(B)))
    if T % B:
        raise SystemExit("YAH_CTX must be a multiple of the chunk size")
    NCH = T // B
    KC = T * 4 * 256          # KV cache elements: T rows x 4 kv heads x 256
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
        name = lambda k: "gemm_%s_%s_%d_%d.hal" % (k, fmt, mt, kb)
        # A residual projection gets a kstore and the fused-residual kres; the attention q projection (12288 rows =
        # 24 heads x [q | gate]) also gets kqg, which stores q and gate unpacked. The driver prefers kres and kqg.
        kinds = {"kstore": ["kstore"] + (["kqg"] if mt == 768 else []), "residual": ["kstore", "kres"],
                 "swiglu": ["swiglu"]}[kind]
        for k in kinds:
            r = gemm(fmt, mt, kb, B, name(k), outdir, kind=k)
            if r:
                geom.append(r)
            elif k == "kstore" and fmt == "q2k":
                # Q2_K has no tile decoder; its only tensors here are the 48-row ssm_alpha / ssm_beta.
                sym = E.sym_of("yah_ffn_gemm_q2k_f32.loom")
                E.emit("yah_ffn_gemm_q2k_f32.loom", ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb),
                                                     "%s.token_tiles=%d" % (sym, TT)], name(k), outdir)
                geom.append((name(k), TILE, 1, TT))
            elif k == kinds[0]:
                raise SystemExit("no GEMM kernel for %s %s (%d rows, K = %d)" % (fmt, k, mt * 16, kb * qk_of(fmt)))
        n += 1

    # Attention: tools/gen_attn_fa.py, 32 tokens x 2 heads per workgroup; reads V^T (vtrans.hal, or the paged / quantized pools).
    # KV paging (256-token pages) needs a context that is a multiple of 256; otherwise the caches stay contiguous.
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
    # Quantized KV (YAH_KV, gen_kvq.kv_bits; engine/run/kvq/README.md).
    # K: int8 (yah_kq8) or H256 + asymmetric int4 (yah_kq4) after yah_kmean centres it; the attention decodes it to f16.
    kv_k, kv_v = gen_kvq.kv_bits()
    kq4_mode = kv_k == 4
    kq8_on = kv_k != 16
    if kq8_on:
        kmean_src = os.path.join(tmp, "yah_kmean.loom")
        kq8_src = os.path.join(tmp, "yah_kq8.loom")
        open(kmean_src, "w").write(gen_kvq.gen_kmean())
        open(kq8_src, "w").write(gen_kvq.gen_kq4() if kq4_mode else gen_kvq.gen_kq8())
        geom.append(("attn_kq4" if kq4_mode else "attn_kq8", 0, 0, 0))
    # V^T as bytes / nibbles per channel per 16-key tile + (S, C') by yah_vq8 / yah_vq4 instead of the f16 transpose.
    vq4_on = kv_v == 4
    # Quantized V: the decoder's open 16-key tile is seeded from the last chunk's f16 V rows (prefill ending mid-tile).
    vseed_src = os.path.join(tmp, "yah_vseed.loom")
    if kv_v != 16:
        open(vseed_src, "w").write(gen_kvq.gen_vseed())
    if vq4_on:
        vq4_src = os.path.join(tmp, "yah_vq4.loom")
        open(vq4_src, "w").write(gen_kvq.gen_vq4())
        geom.append(("attn_vq4", 0, 0, 0))
    vq8_on = kv_v == 8
    if vq8_on:
        vq8_src = os.path.join(tmp, "yah_vq8.loom")
        open(vq8_src, "w").write(gen_kvq.gen_vq8())
        geom.append(("attn_vq8", 0, 0, 0))
    # marker: the attention HAL stores f16 straight into the o-projection input
    geom.append(("attn_f16out", 0, 0, 0))
    geom.append(("vtrans.hal", 0, 0, 0))

    # DeltaNet: chunked WY Gated DeltaNet (tools/gen_gdn_chunk.py), f16 WMMA inputs.
    # Its grid (2, heads) is recorded as rowsplit.hal's row group.
    # It needs B % 32 == 0; else the recurrent kernel (tools/gen_deltanet_hip.py, same ABI and grid).
    dn_src = os.path.join(tmp, "yah_deltanet_hip_f32.loom")
    open(dn_src, "w").write(gen_gdn_chunk.gen() if B % 32 == 0 else gen_deltanet_hip.gen())
    geom.append(("rowsplit.hal", 0, 2, 0))

    # loom_forward_pp reads the launch geometry from dispatch.txt instead of recomputing the grid.
    # So the dispatch site and the compiled kernel cannot disagree; a mismatch is silent and wrong, not a crash.
    # Paged K / V: the f16 KV cache is only a scratch; f16 K / V go to the paged pools (RoPE K store, yah_vtpage per chunk).
    # Quantized K and V: attention never reads the f16 KV cache, so it is a one-layer, one-chunk scratch (kv16_scratch).
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
        # quantized KV: K quantizers run per chunk on cache slices (kmean on chunk 0 only), V quantizers per chunk (vq*_c<i>.hal)
        geom.append(("ctx", B, 0, T))     # chunk size, total context
    with open(os.path.join(outdir, "dispatch.txt"), "w") as fh:
        for hal, tk, rg, tt in geom:
            fh.write("%s %d %d %d\n" % (hal, tk, rg, tt))

    # The output norm (tools/gen_half_norm.py): fully unrolled, all loads issued up front.
    norm_src = os.path.join(tmp, "yah_half_norm_unrolled.loom")
    # 4 rows (waves) per workgroup: 46.1 -> 41.9 ms per pp2048 vs one row per workgroup. The tuner's table may set it
    # ({"norm": {"wpr": 2}}); the driver derives the grid from the kernel's workgroup size.
    open(norm_src, "w").write(gen_half_norm.gen(5120, wpr=tiles().get("norm", {}).get("wpr", 4)))
    fixed = [
        ("yah_residual_add_1d_f32.loom", "accum.hal",
         ["yah_residual_1d.dim=%d" % (5120 * B)]),
        (norm_src, "norm.hal",
         ["yah_half_norm.rows=%d" % B, "yah_half_norm.dim=5120",
          "yah_half_norm.eps=1e-06", "yah_half_norm.fused=0"]),
        # the conv with the q / k L2 norm (prep_kq) fused in
        ("yah_ssm_conv_kq_f32.loom", "convkq.hal",
         ["yah_ssm_conv_kq.batch=%d" % B, "yah_ssm_conv_kq.qkv_dim=10240",
          "yah_ssm_conv_kq.num_key_heads=16"]),
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
        *([(vseed_src, "vseed.hal", ["yah_kvq.token_count=%d" % B])] if (vq8_on or vq4_on) else []),
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
                     ("ksigns_iq2xs.bin", "ksigns_iq2xxs.bin")]:
        shutil.copy(os.path.join(tables, src), os.path.join(outdir, dst))
    print("emitted %d GEMM + fixed prefill HALs for B=%d (tile=%d, token_tiles=%d)"
          % (n, B, TILE, TT))


if __name__ == "__main__":
    main()
