#!/usr/bin/env python3
"""Emit the prefill HAL set for B-token chunks: every kernel compiled for its shape, plus dispatch.txt.

Every token dimension is bound to B: GEMM token_tiles = B / tile (grid y), fixed kernels batch = B, residual dim = 5120 * B.
YAH_CTX=T (a multiple of B, default B) sizes the KV cache and emits one rope / attention HAL per chunk.
The driver runs T / B chunks and carries the DeltaNet and conv states. YAH_KV selects the KV format (gen_kvq.kv_bits).
dispatch.txt has one row per HAL, "<hal> <tokens per workgroup> <row groups> <token_tiles>", plus mode marker rows.
HAL names follow emit_prefill.py.

usage: emit_prefill_pp.py <model.gguf> <outdir> [tokens]   (default 2048 tokens)
"""
import contextlib, dataclasses, functools, json, os, re, subprocess, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emit_prefill as E  # noqa: E402
import gen_attn_fa  # noqa: E402
import gen_gdn_chunk  # noqa: E402
import gen_gdn_ss  # noqa: E402
import gen_conv_kq  # noqa: E402
import gen_kres_persist  # noqa: E402
import gen_half_norm  # noqa: E402
import gen_decode_misc  # noqa: E402
import gen_gemv  # noqa: E402
import gen_r5fuse  # noqa: E402
import gen_kvq  # noqa: E402
import gen_npu_gemm as GN  # noqa: E402
import npu_prof  # noqa: E402


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


def rope_vtpage(text):
    """rope_kpaged's V store into the paged V^T pool instead of the one-chunk scratch (what yah_vtpage did after it):
    pool [kv head][pages * 16 tiles][256 dims][16 keys], token cur in tile ptab[cur / 256] * 16 + cur % 256 / 16,
    key cur % 16. The same f16 values, so the same pool contents; dispatch.txt "vtrope"."""
    def r(a, b):
        nonlocal text
        assert text.count(a) == 1, a[:60]
        text = text.replace(a, b)
    r("  %v16_view = buffer.view %v16_na[%base] : buffer -> view<[%cache16_elems]xf16>",
      "  %v16_view = buffer.view %v16_na[%base] : buffer -> view<[%k16_elems]xf16>")
    r("    %v_loop = scf.for ",
      "    %vt_c16 = index.constant 16 : index\n"
      "    %vt_c4096 = index.constant 4096 : index\n"
      "    %vt_pt0 = index.mul %pg, %vt_c16 : index\n"
      "    %vt_lt = index.div %pgo, %vt_c16 : index\n"
      "    %vt_ptile = index.add %vt_pt0, %vt_lt : index\n"
      "    %vt_kt = index.mul %pgnp, %vt_c16 : index\n"
      "    %vt_a = index.mul %kv_h, %vt_kt : index\n"
      "    %vt_b = index.add %vt_a, %vt_ptile : index\n"
      "    %vt_b0 = index.mul %vt_b, %vt_c4096 : index\n"
      "    %vt_key = index.rem %cur, %vt_c16 : index\n"
      "    %vt_base = index.add %vt_b0, %vt_key : index\n"
      "    %vt_last = index.sub %k16_elems, %c1 : index\n"
      "    %v_loop = scf.for ")
    r("      %v16_off = index.add %f16_off, %i4 : index\n",
      "      %vt_o0 = index.mul %i4, %vt_c16 : index\n"
      "      %vt_o1 = index.add %vt_base, %vt_o0 : index\n"
      "      %v16_off = index.min %vt_o1, %vt_last : index\n")
    r("view.store %vvh, %v16_view[%v16_off] : f16, view<[%cache16_elems]xf16>",
      "view.store %vvh, %v16_view[%v16_off] : f16, view<[%k16_elems]xf16>")
    return text


def postnorm_tiled(text):
    """Rewrite yah_ssm_postnorm_fp16 to store fragment-major for an afrag ssm_out (gen_gemm_tile Tile.atiled over
    K = 6144 = 48 heads x 128): row (token, head) column c is k = head * 128 + c of that token."""
    def r(a, b):
        nonlocal text
        assert text.count(a) == 1, a[:60]
        text = text.replace(a, b)
    r("  %c128 = index.constant 128 : index\n",
      "  %c128 = index.constant 128 : index\n"
      "  %pt16 = index.constant 16 : index\n"
      "  %pt48 = index.constant 48 : index\n"
      "  %pt256 = index.constant 256 : index\n"
      "  %pt384 = index.constant 384 : index\n")
    r("      view.store %half, %out_view[%off2] : f16, view<[%io_total]xf16>\n",
      "      %pt_tok = index.div %head, %pt48 : index\n"
      "      %pt_hh = index.rem %head, %pt48 : index\n"
      "      %pt_k0 = index.mul %pt_hh, %c128 : index\n"
      "      %pt_k = index.add %pt_k0, %column : index\n"
      "      %pt_tt = index.div %pt_tok, %pt16 : index\n"
      "      %pt_tr = index.rem %pt_tok, %pt16 : index\n"
      "      %pt_kt = index.div %pt_k, %pt16 : index\n"
      "      %pt_kr = index.rem %pt_k, %pt16 : index\n"
      "      %pt_a = index.mul %pt_tt, %pt384 : index\n"
      "      %pt_b = index.add %pt_a, %pt_kt : index\n"
      "      %pt_c = index.mul %pt_b, %pt256 : index\n"
      "      %pt_d = index.mul %pt_tr, %pt16 : index\n"
      "      %pt_e = index.add %pt_c, %pt_d : index\n"
      "      %pt_off = index.add %pt_e, %pt_kr : index\n"
      "      view.store %half, %out_view[%pt_off] : f16, view<[%io_total]xf16>\n")
    return text


def postnorm_heads(text, h0, nh, B):
    """Restrict yah_ssm_postnorm_fp16 to heads h0 .. h0 + nh - 1 (config head_count = B * nh rows): row r is token r / nh,
    head h0 + r % nh, i.e. flat row (r / nh) * 48 + h0 + r % nh of the full [B][48][128] buffers. Same math per row."""
    def r(a, b):
        nonlocal text
        assert text.count(a) == 1, a[:60]
        text = text.replace(a, b)
    r("  %io_total = index.mul %head_count, %c128 : index\n",
      f"  %pn_full = index.constant {48 * B * 128} : index\n"
      "  %io_total = index.add %pn_full, %c0 : index\n")
    r("  %head = index.add %head0, %c0 : index\n  %in_range = index.cmp ult, %head, %head_count : index\n",
      f"  %pn_nh = index.constant {nh} : index\n"
      f"  %pn_h0 = index.constant {h0} : index\n"
      "  %pn_c48 = index.constant 48 : index\n"
      "  %pn_tok = index.div %head0, %pn_nh : index\n"
      "  %pn_hr = index.rem %head0, %pn_nh : index\n"
      "  %pn_h = index.add %pn_hr, %pn_h0 : index\n"
      "  %pn_f = index.mul %pn_tok, %pn_c48 : index\n"
      "  %head = index.add %pn_f, %pn_h : index\n"
      "  %in_range = index.cmp ult, %head0, %head_count : index\n")
    return text


# O16_MT: kstore shapes whose outputs are stored as f16 (their consumers read f16): 640 = attn_qkv (conv, prep_ab), 384 =
# attn_gate (the DeltaNet z: postnorm). Not bit-exact; gate N (see engine/run/gate) with the f16 qkv + z simulation:
# mean KL 1.8e-6, p99.9 1.3e-4, 0 flips on 16 windows.
O16_MT = (640, 384)

# DNSPLIT: DeltaNet as two dispatches, heads 0-39 (80 workgroups: two full rounds of 40 slots) and 40-47 (16), with the
# postnorm split the same way; the second part writes its own output buffer, so the first part's postnorm depends only on
# the first DeltaNet part and runs beside the second (whose 16 workgroups fill 16 of 40 slots). Same math per head.
DNSPLIT = (40, 8)

# Sets without a NPU qkv split (dn_vconv): the v channels' conv and alpha / beta run inside DeltaNet
# (gen_gdn_ss natural), the conv kernel takes only the q / k heads (gen_conv_kq.gen(4096)) and prep_ab only advances
# the conv ring, after both DeltaNet parts. Bit-identical; dispatch.txt "dnvconv <conv workgroups>".
# There the DeltaNet is the state-stationary one (gen_gdn_ss: one workgroup per head, the state in registers): the v
# conv inside it, head ranges,
# its O as f32 at the heads' own columns (gen_gdn_ss natural), read by the plain postnorm parts. Sets whose NPU takes
# qkv rows keep gen_gdn_chunk (the v conv needs every qkv column). Gate N mean KL 1.06e-6 (vs the f32 recurrent
# reference 1.53e-6, gen_gdn_chunk 1.08e-6); pp2048 GPU only -0.97% (11-02: DeltaNet alone 66 -> 34 ms, conv 18 -> 9,
# the postnorm loses its overlap 14 -> 24). Fusing the postnorm into part a passed the gate but doubled part a.


def _rewrite(text, pairs):
    for a, b in pairs:
        assert text.count(a) == 1, a
        text = text.replace(a, b)
    return text


def prepab_x16(text):
    """yah_deltanet_prep_ab reading the qkv GEMM's f16 output for the conv ring (O16_MT 640)."""
    return _rewrite(text, [
        ("%qkv_view = buffer.view %qkv_na[%base] : buffer -> view<[%qkv_total]xf32>",
         "%qkv_view = buffer.view %qkv_na[%base] : buffer -> view<[%qkv_total]xf16>"),
        ("%v = view.load %qkv_view[%src] : view<[%qkv_total]xf32> -> f32\n",
         "%vh = view.load %qkv_view[%src] : view<[%qkv_total]xf16> -> f16\n        %v = scalar.extf %vh : f16 to f32\n")])


def postnorm_g16(text):
    """yah_ssm_postnorm_fp16 reading the DeltaNet z gate as the f16 output of its GEMM (O16_MT 384)."""
    return _rewrite(text, [
        ("%gate_view = buffer.view %gate_na[%base] : buffer -> view<[%io_total]xf32>",
         "%gate_view = buffer.view %gate_na[%base] : buffer -> view<[%io_total]xf16>"),
        ("%g = view.load %gate_view[%off2] : view<[%io_total]xf32> -> f32\n",
         "%gh = view.load %gate_view[%off2] : view<[%io_total]xf16> -> f16\n      %g = scalar.extf %gh : f16 to f32\n")])


NORM_SPLIT = 2
# the onorm's waves a row (gen_osum; the chain runs through the parts in order, so any split gives the same bits;
# 4: 852 -> 840 and 862 -> 806 us a call at pp2048, YAH_ONORM_SPLIT=2 the old 256 threads)
ONORM_SPLIT = int(os.environ.get("YAH_ONORM_SPLIT", "4"))


def rope_live(text):
    """yah_fused_qk_rope_batched without its dead stores: nothing in the prefill reads the f32 K / V cache copies
    (k_cache / v_cache) or the in-place f32 k_out (the attention reads the f16 caches)."""
    n = 0
    out = []
    for line in text.split("\n"):
        if re.match(r"\s*view\.store %\w+, %(ko|kc|vc)_view\[", line):
            n += 1
            continue
        out.append(line)
    assert n >= 5, n
    return "\n".join(out)


def rope_q16(text):
    """yah_fused_qk_rope_batched storing Q as f16(q * 0.0625): the attention's own Q scale and rounding (gen_attn_fa
    Q16), done where Q is produced (half the Q bytes written and read; the same bits)."""
    a = "%qo_view = buffer.view %qo_na[%base] : buffer -> view<[%q_elems]xf32>"
    assert text.count(a) == 1
    text = text.replace(a, a.replace("xf32>", "xf16>") + "\n  %qsc16 = scalar.constant 0.0625 : f32")
    stores = re.findall(r"view\.store (%\w+), %qo_view\[(%\w+)\] : f32, view<\[%q_elems\]xf32>", text)
    assert len(stores) == 3, stores
    for v, i in stores:
        text = text.replace(f"view.store {v}, %qo_view[{i}] : f32, view<[%q_elems]xf32>",
                            f"%{v[1:]}q16s = scalar.mulf {v}, %qsc16 : f32\n      %{v[1:]}q16h = scalar.fptrunc %{v[1:]}q16s : f32 to f16\n"
                            f"      view.store %{v[1:]}q16h, %qo_view[{i}] : f16, view<[%q_elems]xf16>")
    return text


def rope_konly(text):
    """yah_fused_qk_rope_batched for K / V only (gen_attn_fa QNORM: the attention normalizes and rotates its own Q):
    grid (num_kv_heads, batch), workgroup x = KV head."""
    a = "  %blocks = index.add %num_heads, %num_kv_heads : index\n  kernel.launch.config workgroups(%blocks,"
    assert text.count(a) == 1
    text = text.replace(a, "  kernel.launch.config workgroups(%num_kv_heads,")
    a = "  %block = kernel.workgroup.id<x> : index\n"
    assert text.count(a) == 1
    text = text.replace(a, "  %block0 = kernel.workgroup.id<x> : index\n  %block = index.add %block0, %num_heads : index\n")
    a = "  %is_q = index.cmp ult, %block, %num_heads : index\n"
    assert text.count(a) == 1
    text = text.replace(a, "")
    i = text.index("  scf.if %is_q {\n")
    j = text.index("\n  }\n  %kv_h_raw = ", i)
    return text[:i] + text[j + len("\n  }\n"):]


def qk_of(fmt):
    return next(qk for f, _, qk in E.FMT.values() if f == fmt)


def gemm(fmt, mt, kb, B, out, outdir, kind="kstore"):
    """Emit the tile GEMM (tools/gen_gemm_tile.py) of this shape and kind; return its dispatch.txt row, or None."""
    r = tile_kstore(fmt, mt, kb, B, out, outdir, kind)
    if not r and mt < 4:
        # the 48-row ssm_alpha/ssm_beta: one 16-row tile per workgroup, 64 tokens over 2 waves, grid (3, B/64)
        r = tile_kstore(fmt, mt, kb, B, out, outdir, kind, geom=(16, 64, 1, 2))
    return r


QNORM = os.environ.get("YAH_QNORM", "1") == "1"   # gen_attn_fa QNORM: the attention does the Q head norm + RoPE (RoPE: K / V only)
# GSIG (YAH_GSIG, default u16; 0: off): the kqg GEMM's epilogue applies the gate's sigmoid and stores it as unorm16
# bits ("u16", gen_gemm_tile KQG_GSIG), the attention multiplies (gen_attn_fa GSIG): half the gate bytes, no exp /
# reciprocal in the attention epilogue. Gate N (KL 1.07e-6); attention busiest SQ 71.6 -> 73.8% of floor at depth 0,
# +0.2 at 8K, even at 64K; pp2048 attention -3.1%, kqg -0.3% (half the stores). Only where the kqg GEMM writes every
# gate row: not with the NPU's q rows. dispatch.txt "gsig"
GSIG = os.environ.get("YAH_GSIG", "u16")
GSIG = "" if GSIG == "0" else GSIG
# QSPLIT (YAH_QSPLIT=1; with the u16 gate, GPU q rows): the kqg GEMM stores q as f16(q w / 16) plus the raw rotary dims
# and per-32-dim sums of q^2 (gen_gemm_tile KQG_QSPLIT); the attention RoPEs, and scales S by the norm's 1 / rms
# (gen_attn_fa R2P_QSPLIT). dispatch.txt "qsplit" (the engine binds the q norm weight to the kqg GEMM)
QSPLIT = os.environ.get("YAH_QSPLIT", "1") == "1"
QSPLIT_ON = False   # set by main
# YAH_ATTN_C0SEED=k: chunk 0's attention HAL from the source reordered by srcseed seed k (0: as emitted). Chunk 0
# (start_pos 0) and the deeper chunks want different orders for clean WMMA banks (rsurvey); 29 cleans QSPLIT's chunk 0
# (+0.8 points busiest SQ at depth 0). Any attention source change needs a new sweep (srcseed + rsurvey).
ATTN_C0SEED = os.environ.get("YAH_ATTN_C0SEED")
# YAH_ATTN_C0ROT16 (QSPLIT, default on): chunk 0's attention RoPEs the f16 halves (gen_attn_fa R2P_QROT16): +1.2..1.8
# points busiest SQ at depth 0 (gate N, kl_mean 1.7e-6); deeper chunks keep the f32 rotary dims (dirty banks at depth)
ATTN_C0ROT16 = os.environ.get("YAH_ATTN_C0ROT16", "1") == "1"
GSIG_ON = False   # set by main


def alt(path):
    return path.replace(".loom", "_alt.loom")


ATTN_VGPR = 192   # every attention variant fits here unless the allocator rotates the O accumulators


def hal_vgprs(hal):
    E.wait(hal)
    notes = subprocess.run(["llvm-readelf", "--notes", hal], capture_output=True, text=True).stdout
    m = re.search(r"\.vgpr_count:\s*(\d+)", notes)
    return int(m.group(1)) if m else 0


def hal_spills(hal):
    E.wait(hal)
    notes = subprocess.run(["llvm-readelf", "--notes", hal], capture_output=True, text=True).stdout
    m = re.search(r"\.private_segment_fixed_size:\s*(\d+)", notes)
    return int(m.group(1)) if m else 0


QROT_FROM = 12288   # kv4: the first context position whose chunk rotates Q in its own kernel (see the attention)
MODEL = None  # set by main(): tiles() reads this model's table


@functools.cache
def tiles():
    """The autotuner's table for MODEL (tune_table.py: YAH_TILES or engine/tune/tables/<model>.json):
    {"tiles": {hal: knobs}, "variants": {hal: [knobs incl. "bn", ...]}, "pick": {hal: {max tokens: bn}},
     "norm": {...}, "rw": {...}}. A flat {hal: knobs, ...} file is read as its "tiles"."""
    import tune_table
    t = tune_table.load(MODEL) if MODEL else {}
    if "tiles" not in t:
        t = {"tiles": {k: v for k, v in t.items() if k.startswith("gemm_")},
             **{k: v for k, v in t.items() if not k.startswith("gemm_")}}
    return t


# Narrower token tiles for chunks with few real tokens; the driver picks one per chunk (LoomPrefill::PickGemm).
# A narrow tile pads less but costs more per token row (x1.12 at 128, x1.6 at 64, measured): an 18-token prompt
# 595 -> 404 ms with 64, a 300-token prompt 909 -> 802 ms with 128. Same values as the default tile.
NARROW = ((128, 2), (64, 2))


def tkey(out):
    """A GEMM's key in the tuned tiles: its HAL name without ".hal" and a row-segment suffix (RSEG_GEMMS: the same
    tile, so the same results)."""
    return re.sub(r"_rseg\d+$", "", out[:-4])


def narrow_variants(fmt, mt, kb, B, out, outdir, kind):
    """Emit "<hal>.t<BN>.hal" for each narrow token tile of this GEMM (the tuner's "variants", else NARROW on the
    default tile), plus the tuner's per-bucket choices as "pick:<hal>:<max tokens> <max tokens> <bn> 0" rows."""
    import gen_gemm_tile as TG
    if fmt not in TILE_FMTS or mt < 4:
        return []
    base, rows = TG.default_tile(fmt, kind, kb), []
    full = dataclasses.replace(base, **tiles()["tiles"].get(tkey(out), {}))
    tuned = tiles().get("variants", {}).get(tkey(out))
    for knobs in (tuned if tuned is not None else [{"bn": bn, "wn": wn} for bn, wn in NARROW]):
        t = dataclasses.replace(base, **knobs)
        bn = t.bn
        try:
            TG.check(t)
        except ValueError:
            continue
        if bn >= full.bn or mt % t.rowgrp:
            continue
        masked = B % bn != 0
        r = _emit_gen(lambda f, k: TG.gen(f, k, t, masked), bn, fmt, mt, kb, B, out[:-4] + ".t%d.hal" % bn, outdir, kind,
                      t.rowgrp, masked)
        if r:
            rows.append(r)
    for maxtok, bn in sorted(tiles().get("pick", {}).get(out[:-4], {}).items(), key=lambda kv: int(kv[0])):
        rows.append(("pick:%s:%s" % (out, maxtok), int(maxtok), int(bn), 0))
    return rows


# Formats the tile GEMM (tools/gen_gemm_tile.py) is verified bit-identical on in the pp2048 pipeline.
TILE_FMTS = ("iq3s", "iq4xs", "iq4nl", "iq3xxs", "q4k", "q5k", "q6k", "iq2xxs", "iq2xs", "iq2s", "q3k", "q8_0", "q2k")


# Activations-from-global GEMMs (gen_gemm_tile Tile.afrag): "<hal>.af.hal" reads a fragment-major input (Tile.atiled; the
# FFN norm "norm_t.hal" and the swiglu "<hal>.af.to.hal" write it). 512-token tiles, only the decoded weights in LDS.
# Same values as the GEMM.
AF_TILE = dict(bm=128, bn=512, wm=1, wn=16, ksub=128, dbuf=False, decahead=False, afrag=True, atiled=True, stg_minwg=0,
               tallepi=True, ecoal=True, esr=32)
AF_PIPE = dict(wlate=True, bpre=2, b0early=True)
# IQ3_S: step 1's B prefetch issued with step 0's, before the decode; otherwise the allocator copies one of its fragments
# right after its load, which drains vmcnt(0) once per phase (IQ4_XS: 200 VGPRs with it, fused IQ3_S ffn: 216)
AF_PIPE3 = dict(AF_PIPE, b0early=2)
AF_KQ = dict(rhs_outer=False, rhs_fence=0)   # K-quants without B prefetch (Q6_K: 208 VGPRs with it)
# Q4_K / Q3_K / Q5_K with the B prefetch: their decoders came down to 184-192 VGPRs (fma narrowing, results.md), so it fits now:
# -3.6..-4.0% clock-free (Q4_K swiglu -5.6%, kstore 64 -5.5%)
AF_KQP = dict(AF_KQ, **AF_PIPE, lhs_stream=4)
# No B prefetch (bpre 0, no b0early, lhs_stream 0): under HRX 0051 the prefetching forms sit at 192 VGPRs with
# shared WMMA banks; without it 179-187 VGPRs, rank 0 (11-02, clock-free, bit-identical; fused ffn AF_FFN too)
AF_NOPRE = dict(AF_PIPE, bpre=0, b0early=False, lhs_stream=0)
# The GEMMs with an afrag form and their knobs on top of AF_TILE; clock-free vs the GEMM (one round each, 2026-10-03).
# The best lhs_stream depends on the decoder's register allocation: IQ3_XXS kstore without it waits vmcnt(0) twice per
# phase on the step-1 B prefetch. Q3_K: the prefetch needs 208 VGPRs (one workgroup per WGP). IQ3_XXS: lhs_stream=1 with
# the gen_gemm_decode.IQ3_SGTAB sign table (deeper A streaming splits the A-fragment tuples at the 192-VGPR tier: 8 copies
# per k step); alone it is neutral, with the table -0.7..-1.0% (one process A/B 2026-10-04). Not the fused ffn: its
# fragment-major-output form loses +1.3% with the table although its loop drops 32 VALU; nor swiglu (same, +1.3% in pp2048).
# IQ3_S (256-entry table of the block's sign bytes): kres 320x68 -0.7%, kqg -1.0%, kstore 384 -0.5%; kstore 1088 / 640
# +0.5 / +0.2% and the fragment-major ffn / swiglu +0.7 / +2.2% keep the spread.
# f16p (IQ3 decode bytes as f16 subnormal pairs, one v_perm for the odd bytes; -11..-15 loop VALU): IQ3_S kstore 1088 /
# kres 320x68 / kqg / swiglu -0.25..-0.55%, fused ffn IQ3_XXS / IQ3_S -0.35%; IQ3_XXS kstore / kqg neutral, kres +1.8% (not).
# gaddr (IQ3 grid / sign-table lookups at pre-scaled byte offsets on byte views of the LDS tables: IQ3_XXS -15, IQ3_S
# -17..-23 loop VALU; one process A/B 2026-10-04): IQ3_XXS kstore -0.3..-0.8%, kqg -0.25%, kres -0.6..-1.2%, swiglu -0.7%;
# IQ3_S kstore 640 / kqg -0.25..-0.3%, kres -0.8..-1.0%. Not: IQ3_S kstore 384 (200-208 VGPRs at every lhs_stream, +12%),
# IQ3_S kstore 1088 / swiglu and both fused ffn (neutral).
# f16s (IQ3 signed f16 subnormal pairs: two v_perm per grid word instead of sign xor / add + pair split, fma addend +0;
# IQ3_XXS loop VALU 156 -> 114; one process A/B 2026-10-04): IQ3_XXS kstore / kqg -1.2..-1.5% (lhs_stream=3), kres -0.85%
# (wlate=False: 188 VGPRs, else 200-208), swiglu -0.4%, fused ffn -1.0% (no sign table); IQ3_S kstore -0.8..-1.5%,
# kqg -1%, kres -0.7..-1.7%, fused ffn -0.5%. Not: IQ3_S swiglu (+0.2..+25%).
# tallepi (kstore / kres): the LDS epilogue in 16-row slabs; the fragment stores cost the K = 6144 kres 15%.
AF = {
    ("iq3s", "kstore", 1088, 20): dict(AF_PIPE3, lhs_stream=2, f16p=True, f16s=True, sgtab=True),  # -9.1%
    ("iq4xs", "kstore", 1088, 20): dict(AF_PIPE, lhs_stream=2),                 # -12.9%
    ("iq3xxs", "kstore", 1088, 20): dict(AF_PIPE, lhs_stream=3, sgtab=True, gaddr=True, f16s=True),  # -11.3%
    ("q3k", "kstore", 1088, 20): dict(AF_KQP),                                  # -7.2%
    ("iq3s", "swiglu", 1088, 20): dict(AF_PIPE3, lhs_stream=2, swepi=False, f16p=True),     # -11.6%
    ("iq4xs", "swiglu", 1088, 20): dict(AF_PIPE, lhs_stream=2, swepi=False),    # -10.9%
    ("iq3xxs", "swiglu", 1088, 20): dict(AF_PIPE, lhs_stream=2, swepi=False, gaddr=True, f16s=True),  # -12.1%
    # persistent lhs_stream 1 (was 2): bank cost 0.6 -> 0.1%, 191 VGPRs; down -0.4% (92.6 -> 93.0%), bit-identical (11-03)
    ("iq3s", "kres", 320, 68): dict(AF_PIPE3, sgtab=True, f16p=True, gaddr=True, f16s=True,
                                    persist=dict(b0early=True, lhs_stream=1)),               # -8.6%
    # lhs_stream 1 (was 4) since HRX 0051: bank cost 1.0 -> 0.4%, 190 VGPRs; pp2048 counters 3 runs (11-03),
    # bit-identical: down -0.66% (92.8 -> 93.4%), out -0.32%
    ("iq4xs", "kres", 320, 68): dict(AF_PIPE, lhs_stream=1, respre=1, persist=dict(pslice="top_top")), # -10.4%
    ("iq3xxs", "kres", 320, 68): dict(AF_PIPE, lhs_stream=1, sgtab=True, gaddr=True, f16s=True, wlate=False, respre=1,
                                      persist=dict(pslice="b0_a1")),                         # -8.2%
    ("q4k", "kres", 320, 68): dict(AF_KQP, respre=1, persist=True, lhs_stream=1),   # -10.4%; lhs_stream 1: see q5k 24
    ("q4k", "swiglu", 1088, 20): dict(AF_KQP, swepi=False),                     # -7.7%
    ("q3k", "swiglu", 1088, 20): dict(AF_KQP, swepi=False),
    ("q5k", "kstore", 1088, 20): dict(AF_KQP, ksl=True),
    ("q5k", "swiglu", 1088, 20): dict(AF_KQP, ksl=True, swepi=False),
    ("iq2s", "kstore", 1088, 20): dict(ksl=True, wlate=True, b0early=True, lhs_stream=2),
    ("iq2s", "swiglu", 1088, 20): dict(ksl=True, wlate=True, b0early=True, lhs_stream=2),
    # attention o-proj / DeltaNet ssm_out (K = 6144; input: the attention output, postnorm_t.hal)
    ("iq3s", "kres", 320, 24): dict(AF_PIPE3, gaddr=True, f16s=True, sgtab=True,
                                    persist=dict(b0early=True, lhs_stream=2)),               # -8.8%
    ("iq3xxs", "kres", 320, 24): dict(AF_PIPE, lhs_stream=1, sgtab=True, gaddr=True, f16s=True, wlate=False, respre=1,
                                      persist=dict(pslice="b0_a1")),                         # -8.5%
    ("iq4xs", "kres", 320, 24): dict(AF_PIPE, lhs_stream=1, respre=1, persist=dict(pslice="top_top")), # -8.8%; ls 1: see 68
    ("q4k", "kres", 320, 24): dict(AF_KQP, respre=1, persist=True, pslice="top_top", lhs_stream=1),   # -5.3%; top_top 89.6 -> 90.5%
    # DeltaNet qkv (10240 rows) / gate (6144 rows); input: norm_rt.hal (alpha / beta keep the row-major copy)
    ("iq3s", "kstore", 640, 20): dict(AF_NOPRE, gaddr=True, f16s=True),          # -9.6%; no prefetch -1.5%
    ("iq4xs", "kstore", 640, 20): dict(AF_NOPRE),                               # -10.3%; no prefetch -1.1%
    ("iq3xxs", "kstore", 640, 20): dict(AF_PIPE, lhs_stream=3, sgtab=True, gaddr=True, f16s=True),  # -11.4%
    ("q4k", "kstore", 640, 20): dict(AF_KQP, bpre=0, b0early=False, lhs_stream=1),  # -7.4%; no prefetch -1.1%
    ("q3k", "kstore", 640, 20): dict(AF_KQP),                                   # -7.7%
    ("iq3s", "kstore", 384, 20): dict(AF_PIPE3, lhs_stream=3, sgtab=True, f16s=True),       # -9.4%
    ("iq4xs", "kstore", 384, 20): dict(AF_NOPRE),                               # -9.9%; no prefetch -1.2%
    ("iq3xxs", "kstore", 384, 20): dict(AF_PIPE, lhs_stream=3, sgtab=True, gaddr=True, f16s=True),  # -10.9%
    ("q4k", "kstore", 384, 20): dict(AF_KQP, bpre=0, b0early=False, lhs_stream=1),  # -6.0%; no prefetch -0.9%
    ("q3k", "kstore", 384, 20): dict(AF_KQP),                                   # -6.4%
    ("q5k", "kstore", 384, 20): dict(AF_KQP, ksl=True),                         # -12%
    # K-quant kres lhs_stream 1 (each A fragment just before its MMAs) since HRX 0051: WMMA bank cost 1.2-1.8% -> 0.1-0.5%,
    # 187-192 VGPRs; pp2048 counters 3 runs (11-03), bit-identical: q5k out -1.2% (89.3 -> 90.4%), q5k down -1.8%, q4k
    # out -0.9%, q4k down -1.1%, q3k down -0.9%
    ("q5k", "kres", 320, 24): dict(AF_KQP, ksl=True, respre=1, persist=True, pslice="top_top", lhs_stream=1),   # -4.6%; top_top 88.3 -> 89.2%
    # 11-02: the down kres of the formats that had no afrag form (tiled 256-token kernels at 69-82% of the floor)
    ("q5k", "kres", 320, 68): dict(AF_KQP, ksl=True, respre=1, persist=True, lhs_stream=1),
    ("q3k", "kres", 320, 68): dict(AF_KQP, respre=1, persist=True, pslice="top_top", lhs_stream=1),   # top_top 90.9 -> 91.4%
    ("q3k", "kres", 320, 24): dict(AF_KQP, respre=1, persist=True, pslice="top_top", lhs_stream=1),   # top_top 88.6 -> 89.1%
    ("iq2s", "kres", 320, 68): dict(ksl=True, wlate=True, b0early=True, lhs_stream=2, respre=1, persist=True,
                                    pslice="top_top"),
    # attention q (kqg: 12288 rows, q / gate split) and k / v (1024 rows); input: attn_norm (norm_t / norm_rt)
    ("iq3s", "kqg", 768, 20): dict(AF_PIPE3, lhs_stream=1, sgtab=True, f16p=True, gaddr=True, f16s=True),  # -8.4%
    ("iq4xs", "kqg", 768, 20): dict(AF_PIPE, lhs_stream=2),                     # -9.5%
    ("iq3xxs", "kqg", 768, 20): dict(AF_PIPE, lhs_stream=3, sgtab=True, gaddr=True, f16s=True),  # -9.3%
    ("q4k", "kqg", 768, 20): dict(AF_KQP),                                      # -6.7%
    ("q3k", "kqg", 768, 20): dict(AF_KQP),                                      # -5.0%
    ("q5k", "kqg", 768, 20): dict(AF_KQP, ksl=True),                            # -10.8%
    # 11-02, the q of the two layers in these formats (tiled 96 x 8 at 69-72% of the floor): Q2_K without B prefetch
    # -19.0% (with it -13.4%); IQ4_NL (32-element blocks) spills without it, lhs_stream 4 -17.7%
    # Q2_K with the B prefetch since its fptrunc(fma) decode (187 -> 192 VGPRs): attention seg -0.15 M cycles
    ("q2k", "kqg", 768, 20): dict(AF_KQP, ksl=True),
    ("iq4nl", "kqg", 768, 160): dict(ksl=True, wlate=True, b0early=True, lhs_stream=4),
    ("iq4xs", "kstore", 64, 20): dict(AF_PIPE),                                 # -15.7%
    ("q4k", "kstore", 64, 20): dict(AF_KQP),                                    # -16.0%
    ("q5k", "kstore", 64, 20): dict(AF_KQP, ksl=True),                          # -10.4%
    # B prefetch (AF_KQP) since it fits the 192 tier (it was 208 VGPRs): the attention segs with a Q6_K v rider (layers
    # 23, 63) 16.34 / 16.37 -> 16.14 / 16.17 Mcyc per call (-1.2%, 89.7 -> 90.8% of floor), bit-identical (11-03)
    ("q6k", "kstore", 64, 20): dict(AF_KQP, ksl=True),
    # IQ2: the grid / sign lookups leave no room for B prefetch (bpre 1-2: 208-216 VGPRs, one workgroup per WGP)
    ("iq2xxs", "kstore", 1088, 20): dict(ksl=True, wlate=True, b0early=True, lhs_stream=2),   # -19.3%
    ("iq2xxs", "swiglu", 1088, 20): dict(ksl=True, wlate=True, b0early=True, lhs_stream=2),   # -16.6%
    ("iq2xs", "kstore", 1088, 20): dict(ksl=True, wlate=True, b0early=True, lhs_stream=2),    # -18.4%
}


# Fused ffn_gate + ffn_up (gen_gemm_tile kind "ffn"): one afrag GEMM writes f16(silu(gate) * up) for layers whose gate
# and up share a format (the driver binds both tensors as one: the GGUF stores ffn_up right after ffn_gate). Same
# values as kstore + swiglu. vs the two afrag GEMMs, clock-free (2026-10-03, fast reciprocal in both): IQ3_S -3.5%,
# IQ4_XS -0.9%, IQ3_XXS -3.1%.
AF_FFN = {
    # IQ3_S: rev_gf_jf (HRX 0051 compiler) WMMA bank rank 1.16 -> 0.38, -2.2% (.to), bit-identical
    # IQ3_S / IQ4_XS: no B prefetch (bpre 0, no b0early, lhs_stream 0; 11-02, HRX 0051): -0.7% / -1.1% (rank 0)
    "iq3s": dict(AF_PIPE, swepi=False, f16p=True, f16s=True, epiord="rev_gf_jf", bpre=0, b0early=False, lhs_stream=0),
    # IQ4_XS: the order steers the hot loop's WMMA banks (with HRX 0049 numbering): rev_gf_jf leaves A, B, C + 1 and
    # C in distinct banks (-2.2% vs rev_gf_jr; the default order shared banks, +3%), bit-identical
    "iq4xs": dict(AF_PIPE, swepi=False, epiord="rev_gf_jf", bpre=0, b0early=False, lhs_stream=0),
    # IQ3_XXS: f16s without a sign table (-1.0%; with the packed sgtw=1 table +0.9..+1.6%, which itself had been -0.5%
    # over the plain f16p form; the two-word table +1.3%: LDS bank conflicts 13.9 -> 23.8 M cycles, LDS issue waits +62%)
    "iq3xxs": dict(AF_PIPE, lhs_stream=1, swepi=False, f16p=True, gaddr=True, f16s=True),
}
# Mixed-format fused gate + up ((gate fmt, up fmt): gen_gemm_tile fmt "<gate>:<up>", "gemm_ffn_<gate>_<up>_<mt>_<kb>"):
# each wave loads its own format's weights at its decode (ffn_inload) and no B prefetch, the only forms within the
# 192-VGPR tier; base knobs the gate format's AF_FFN entry, else its swiglu's. vs the gate kstore + up swiglu pair
# (clock-free): iq3s:iq4xs -1.4%, iq4xs:q4k -3.0%, q3k:iq3s -2.6%, iq3s:q3k -1.0%, iq4xs:iq3s -1.0%, outputs
# bit-identical. GPU only (the driver keeps the pair while the ffn site has NPU rows).
# A pair of one format (no AF_FFN entry, so no NPU split): "gemm_ffn_<fmt>_<fmt>_..", the format's swiglu knobs without
# B prefetch: q5k -3.5%, q3k -3.8% (the prefetching forms rank 1.0 WMMA banks, -1.2..-1.4%).
AF_MIX = dict(ffn_inload=True, bpre=0, b0early=False, lhs_stream=0)
# ffn_sel: every lane loads both formats' slots at a role-selected address, carried like a one-format prefetch (the
# in-decode loads waited on DRAM each phase); lhs_stream 1 keeps it within the 192-VGPR tier. pp2048 counters, 3 runs
# (11-03), bit-identical: iq3xxs:iq3s -1.6%, iq2xs:iq3s -1.5%, iq2s:iq3xxs -1.3%, iq4xs:iq3s -1.0%, iq3xxs:iq2s -0.8%,
# iq3s:iq4xs -0.6%, q3k:iq3s -0.3%, GEMMs -0.13%; iq2xs:iq2s +1.7% (keeps the in-decode loads)
AF_SEL = dict(ffn_sel=True, bpre=0, b0early=False, lhs_stream=1)
AF_FFN_MIX = {
    ("iq3s", "iq4xs"): dict(AF_FFN["iq3s"], **AF_SEL),
    ("iq4xs", "q4k"): dict(AF_FFN["iq4xs"], **AF_SEL),
    ("q3k", "iq3s"): dict(AF[("q3k", "swiglu", 1088, 20)], **AF_SEL),
    ("iq3s", "q3k"): dict(AF_FFN["iq3s"], **AF_SEL),
    ("iq4xs", "iq3s"): dict(AF_FFN["iq4xs"], **AF_SEL),
    ("q5k", "q5k"): dict(AF[("q5k", "swiglu", 1088, 20)], bpre=0, b0early=False, lhs_stream=0),
    ("q3k", "q3k"): dict(AF[("q3k", "swiglu", 1088, 20)], bpre=0, b0early=False, lhs_stream=0),
    # both formats with lookup tables (the up's bind as gridu / ksignsu, 11-02): iq3xxs:iq3s -1.3%, iq3xxs:iq2s -3.7%,
    # iq2s:iq3xxs -2.8%, iq2xs:iq2s -4.1%, iq2xs:iq3s -1.9% (IQ2_XS: a gate only, its kstore knobs); iq2s:iq2s with its
    # swiglu knobs -4.5% (without B prefetch -4.0%)
    ("iq3xxs", "iq3s"): dict(AF_FFN["iq3xxs"], **AF_SEL),
    ("iq3xxs", "iq2s"): dict(AF_FFN["iq3xxs"], **AF_SEL),
    # lhs_stream 1 (each A fragment just before its MMAs; 176 VGPRs): pp2048 counters, 3 runs each, -0.5 M cycles per
    # call (-1.2%) for these three, bit-identical; iq3s:q3k +0.5% with it (gemm_bench had all four 1-3% better)
    ("iq2s", "iq3xxs"): dict(AF[("iq2s", "swiglu", 1088, 20)], **AF_SEL),
    ("iq2xs", "iq2s"): dict(AF[("iq2xs", "kstore", 1088, 20)], **dict(AF_MIX, lhs_stream=1)),   # ffn_sel +1.7%
    ("iq2xs", "iq3s"): dict(AF[("iq2xs", "kstore", 1088, 20)], **AF_SEL),
    ("iq2s", "iq2s"): dict(AF[("iq2s", "swiglu", 1088, 20)]),
}


def ffn_fused(rows, B, outdir):
    """Emit "gemm_ffn_<fmt>_<mt>_<kb>.af.hal" / ".af.to.hal" for each AF_FFN format some layer uses for both its gate
    and up, and "gemm_ffn_<gate>_<up>_.." for each AF_FFN_MIX pair some layer has; return the dispatch.txt rows."""
    import gen_gemm_tile as TG
    if B % AF_TILE["bn"]:
        return []
    by = {}
    for nm, dims, ty in rows:
        parts = nm.split(".")
        if len(parts) > 2 and parts[0] == "blk" and parts[2] in ("ffn_gate", "ffn_up") and E.FMT.get(ty):
            by.setdefault(parts[1], {})[parts[2]] = (E.FMT[ty][0], dims[1] // 16, dims[0] // E.FMT[ty][2])
    out = []
    jobs = {(fmt, fmt, mt, kb) for fmt, mt, kb in (l["ffn_gate"] for l in by.values() if l.get("ffn_gate") == l.get("ffn_up"))
            if fmt in AF_FFN}
    jobs |= {(g[0], u[0], g[1], g[2]) for g, u in ((l.get("ffn_gate"), l.get("ffn_up")) for l in by.values())
             if g and u and g[1:] == u[1:] and (g[0], u[0]) in AF_FFN_MIX}
    for fmt, fmt_up, mt, kb in sorted(jobs):
        pair = (fmt, fmt_up) in AF_FFN_MIX
        knobs = AF_FFN_MIX[(fmt, fmt_up)] if pair else AF_FFN[fmt]
        name = fmt + "_" + fmt_up if pair else fmt
        gfmt = fmt if fmt == fmt_up else fmt + ":" + fmt_up
        t = dataclasses.replace(TG.default_tile(fmt, "swiglu", kb), **AF_TILE, **knobs, ffn=True)
        for suffix, tt in ((".af.hal", t), (".af.to.hal", dataclasses.replace(t, tout=True))):
            TG.check(tt)
            r = _emit_gen(lambda f, k: TG.gen(f, k, tt, False), tt.bn, gfmt, mt, kb, B,
                          "gemm_ffn_%s_%d_%d%s" % (name, mt, kb, suffix), outdir, "ffn", tt.rowgrp)
            if r:
                out.append(r)
    return out


# KST2: each layer's ssm_alpha + ssm_beta (one format, stored back to back) in one kstore, "gemm_kst2_<fmt>_<mt>_<kb>.hal"
# (mt of one tensor; the engine takes it when the two are adjacent), 48-row tiles over both tensors' rows, rows of beta
# to the second output (gen_gemm_tile.SPLIT2). gemm_bench 11-02: 0.111 vs 0.157 Mcyc for the two 48-row kstores (the
# merged rows at the old 16-row tile 0.144: the tile, not the shared input, wins); pp2048 counters 20.2 -> 15.5 Mcyc.
# Same values.
KST2_GEOM = (48, 64, 3, 2)


def kst2(rows, B, outdir):
    """Emit the KST2 GEMMs for the (fmt, mt, kb) of every layer whose ssm_alpha and ssm_beta match; dispatch.txt rows."""
    import gen_gemm_tile as TG
    by = {}
    for nm, dims, ty in rows:
        parts = nm.split(".")
        if len(parts) > 2 and parts[0] == "blk" and parts[2] in ("ssm_alpha", "ssm_beta") and E.FMT.get(ty):
            by.setdefault(parts[1], {})[parts[2]] = (E.FMT[ty][0], dims[1] // 16, dims[0] // E.FMT[ty][2])
    out = []
    for fmt, mt, kb in sorted({l["ssm_alpha"] for l in by.values() if l.get("ssm_alpha") == l.get("ssm_beta")}):
        if fmt not in TILE_FMTS or B % KST2_GEOM[1]:
            continue
        t = TG.default_tile(fmt, "kstore", kb, KST2_GEOM)
        if mt % t.rowgrp:   # a workgroup's rows must lie in one tensor
            continue
        TG.SPLIT2 = True
        try:
            # ".at.hal": the same GEMM reading the norm's fragment-major copy (Tile.atin), so a deltanet layer whose
            # qkv / gate are afrag needs no row-major copy (GN_KST2AT=0: not emitted)
            for sfx, tt in [(".hal", t)] + ([(".at.hal", dataclasses.replace(t, atin=True))]
                                             if os.environ.get("GN_KST2AT", "1") != "0" else []):
                TG.check(tt)
                r = _emit_gen(lambda f, k: TG.gen(f, "kstore", tt, False), tt.bn, fmt, 2 * mt, kb, B,
                              "gemm_kst2_%s_%d_%d%s" % (fmt, mt, kb, sfx), outdir, "kst2", tt.rowgrp)
                if r:
                    out.append(r)
        finally:
            TG.SPLIT2 = False
    return out


# SEG: segmented GEMMs (gen_segcat): a layer's small GEMMs over the same activations ride in the large one's dispatch,
# each with its own weights, format, row tile and output. DeltaNet: attn_qkv + ssm_alpha + ssm_beta (48-row segments);
# attention: the q + gate kqg + attn_k + attn_v (their afrag tiles; k and v are not adjacent and often of two formats).
# "seg_<kind>_<fmt>_<mt>_<kb>__..." per segment, the large GEMM first; dispatch.txt "<hal> <tile> <groups> <tt>": the
# grid x (all segments' row groups) in the rowgrp column. Same values.
SEG_SMALL = dict(ksl=True, bm=48, tallepi=False, ecoal=False)   # the 48-row afrag tile (fragment-store epilogue)
SEG_VGPR_TIER = 192   # 1536 / 8 waves: two 16-wave workgroups per WGP
# riders' knobs when the segmented GEMM would cross the tier: each A fragment loaded just before its MMAs, one step
# ahead (kstore 64 VGPRs: Q4_K 192 -> 179, Q5_K 192 -> 187, Q6_K 208 -> 192; 64-row riders fit too but ran at 85%)
SEG_RIDER_LIGHT = dict(lhs_stream=1)


def _seg_tile(fmt, kind, mt, kb, small, light=None):
    import gen_gemm_tile as TG
    d = TG.default_tile(fmt, kind, kb)
    if small:   # the format's own K step (AF_TILE's 128 is for the 256-element formats)
        return dataclasses.replace(d, **dict(AF_TILE, **SEG_SMALL, ksub=d.ksub))
    knobs = {k: v for k, v in AF.get((fmt, kind, mt, kb), {}).items() if k != "persist"}
    return dataclasses.replace(d, **dict(AF_TILE, **dict(knobs, **(light or {}))))


def seg_gemms(rows, B, outdir):
    """Emit the SEG GEMMs some layer can use; return their dispatch.txt rows."""
    import gen_gemm_tile as TG
    import gen_segcat
    import hrx_paths
    if B % AF_TILE["bn"]:
        return []
    by = {}
    for nm, dims, ty in rows:
        parts = nm.split(".")
        if len(parts) > 2 and parts[0] == "blk" and E.FMT.get(ty):
            by.setdefault(parts[1], {})[parts[2]] = (E.FMT[ty][0], dims[1] // 16, dims[0] // E.FMT[ty][2])
    jobs = set()
    for l in by.values():
        q, a, b = l.get("attn_qkv"), l.get("ssm_alpha"), l.get("ssm_beta")
        if q and a and b and (q[0], "kstore", q[1], q[2]) in AF and a[0] in TILE_FMTS and b[0] in TILE_FMTS:
            jobs.add(((q, "kstore", False), (a, "kstore", True), (b, "kstore", True)))
        q, k, v = l.get("attn_q"), l.get("attn_k"), l.get("attn_v")
        if q and k and v and (q[0], "kqg", q[1], q[2]) in AF and all((x[0], "kstore", x[1], x[2]) in AF for x in (k, v)):
            jobs.add(((q, "kqg", False), (k, "kstore", False), (v, "kstore", False)))
    out, big = [], {}
    tt = B // AF_TILE["bn"]

    def build(segs, light=None):
        """Emit one segmented GEMM; return its dispatch.txt row."""
        texts = []
        for i, ((fmt, mt, kb), kind, small) in enumerate(segs):
            assert (fmt, mt, kb) not in DNSEG, "a segmented GEMM over K-segmented down rows"
            t = _seg_tile(fmt, kind, mt, kb, small, light if i and not small else None)
            TG.check(t)
            TG.OUT16 = kind == "kstore" and mt in O16_MT
            try:
                raw = TG.gen(fmt, kind, t, False)
            finally:
                TG.OUT16 = False
            sym = "yah_ffn_gemm_%s%s" % (fmt, {"kqg": "_kqg"}.get(kind, ""))
            # each segment's body is unchanged in the spliced kernel but for names: its own footprint gate covers it
            E._gate("seg_%s_%s_%d_%d" % (kind, fmt, mt, kb), raw, [sym, fmt, kind, str(mt), str(kb), str(tt), str(B)],
                    hrx_paths.env())
            texts.append((raw, (mt, kb, tt)))
        name = "seg_" + "__".join("%s_%s_%d_%d" % (kind, fmt, mt, kb) for (fmt, mt, kb), kind, _ in segs) + ".hal"
        text, groups = gen_segcat.segcat("yah_seg", texts)
        tmp = os.path.join(outdir, ".emit_tmp")
        os.makedirs(tmp, exist_ok=True)
        src = os.path.join(tmp, name[:-4] + ".loom")
        open(src, "w").write(text)
        E.emit(src, [], name, outdir)
        return name, AF_TILE["bn"], groups, tt

    # one VGPR tier: over 192 a WGP holds one 16-wave workgroup instead of two. A segmented GEMM the riders push over
    # it while the large GEMM alone fits loses more than it saves (11-02: IQ4_NL kqg + K-quant k / v 200, Q6_K v 204:
    # +0.3..+1.5 M cycles per layer): those retry with SEG_RIDER_LIGHT riders, else keep the separate GEMMs
    for segs in sorted(jobs):
        (fmt, mt, kb), kind, _ = segs[0]
        big = hal_vgprs(os.path.join(outdir, "gemm_%s_%s_%d_%d.af.hal" % (kind, fmt, mt, kb)))
        for light in (None, SEG_RIDER_LIGHT):
            r = build(segs, light)
            if hal_vgprs(os.path.join(outdir, r[0])) <= SEG_VGPR_TIER or big > SEG_VGPR_TIER:
                out.append(r)
                break
            os.remove(os.path.join(outdir, r[0]))
    return out


# The out projections' (K = 6144) start stagger: the second workgroup per WGP waits this many barriers (gen_gemm_tile
# STAGGER, default 8000). 11-02 pp2048 counters, out kres M cycles (iq4xs / q4k per call / q5k): 8000 252.6 / 7.06 / 86.0,
# 4000 250.8 / 7.03 / 85.8, 2000 250.6 / 7.01 / 85.3, 1000 251.5 / 7.04 / 85.8, 0 253.2 / 7.07 / 86.5. Same values.
OUT_KRES_STAGGER = 2000


def afrag_variants(fmt, mt, kb, B, out, outdir, kind):
    """Emit "<hal>.af.hal" (and for swiglu "<hal>.af.to.hal", fragment-major output) if this GEMM has an afrag form."""
    import gen_gemm_tile as TG
    knobs = AF.get((fmt, kind, mt, kb))
    if knobs is None or B % AF_TILE["bn"]:
        return []
    knobs = dict(knobs)
    if kind == "kres" and kb == 24:
        knobs.setdefault("stagger", OUT_KRES_STAGGER)
    persist = knobs.pop("persist", False)
    t = dataclasses.replace(TG.default_tile(fmt, kind, kb), **AF_TILE, **knobs)
    rows = []
    for suffix, tt in [(".af.hal", t)] + ([(".af.to.hal", dataclasses.replace(t, tout=True))] if kind == "swiglu" else []):
        TG.check(tt)
        r = _emit_gen(lambda f, k: TG.gen(f, k, tt, False), tt.bn, fmt, mt, kb, B, out[:-4] + suffix, outdir, kind,
                      tt.rowgrp)
        if r:
            rows.append(r)
    if persist and kind == "kres" and B // t.bn > 1 and 2 * kb >= gen_kres_persist.SLICES:
        # the persistent kres (gen_kres_persist): grid y = 1, each workgroup runs every token tile of its row block
        # persist: True, or a dict of knobs for the persistent variant (its slice loads need registers: IQ3_S drops b0early 2)
        pt = dataclasses.replace(t, **persist) if isinstance(persist, dict) else t
        with kseg_for(fmt, mt, kb):
            text = gen_kres_persist.persist(TG.gen(fmt, "kres", pt, False),
                                            TG.gen(fmt, "kstore", dataclasses.replace(pt, respre=0), False), B // t.bn,
                                            place=pt.pslice)
        r = _emit_gen(lambda f, k: text, t.bn, fmt, mt, kb, B, out[:-4] + ".af.p.hal", outdir, kind, t.rowgrp)
        if r:
            rows.append(r)
    return rows


# NPU column split (YAH_NPU_SPLIT="<site>=<NPU rows>,...", rows multiples of "npurows" (512), q also of 512; empty: none):
# the NPU (XDNA2, HRX .xdna) computes the trailing rows of a GEMM, the GPU the leading ones. Sites (rows x K):
#   qkv  DeltaNet attn_qkv (10240 x 5120, f16 out)     gate  DeltaNet attn_gate (6144 x 5120, f16 out)
#   q    attention attn_q (12288 x 5120, kqg: whole heads of [256 q | 256 gate])
#   out  ssm_out / attn_output (5120 x 6144, kres)
# The NPU's calls have K = 5120 (gen_npu_gemm passes): out takes a K chunk of 5120 and the GPU computes the rest of K for
# the NPU's rows (npu_rem_tile, gen_gemm_tile.KWIN), which the unpack adds.
# Per site and format, the GPU's share of every variant the driver may run ("<hal>.npu.hal": "<hal>" any of the set's
# GEMM HALs, its afrag / persistent / fragment-major-output forms included; gen_gemm_tile.OSTRIDE). Per K the activation
# encoders ("npu_enc_<K>[_c<chunk>][_t][_pk].hal", _t: fragment-major input); per site the unpack
# ("npu_unpack_<site>.hal"; qkv: "npu_unpack_qkv_tail.hal" + the conv split "convkq_g.hal" / "convkq_c.hal", which reads C
# itself). dispatch.txt: "npusplit_<site> <rows> 0 0", "npubytes_<K> <A> <W per call> <C per call>", and per kernel
# "<hal> <workgroups> <workgroup size> 0" (the split GEMMs have GEMM rows).
NPU_SPLIT = dict((k, int(v)) for k, v in (x.split("=") for x in os.environ.get("YAH_NPU_SPLIT", "").split(",") if x))
# The NPU image waits for each job's ready word itself (gen_npu_gemm GATE, HRX patch 0016): no host relay sits between
# the GPU and the NPU. dispatch.txt: "npugate <supply> <calls> <record>" (gen_npu_gemm GATE_*), and npu_flag_wait.hal
# waits for a job's done word (gen_npu_unpack.gen_flag_wait).
# The NPU decodes its weights itself from the raw GGUF rows the engine registers with it: 8 GEMM columns whose cores each
# decode their own panel slice (gen_npu_gemm fuse 2), images "npu_dcol_<fmt>_<K>.xdna" for the formats in FUSE_FMTS, NP 4:
# 512-row calls (NP 5 images no longer fit the grid decoders' raw rings), one replay group. The decoders' k-block order
# (gen_npu_dec.KPERM) permutes the activations (gen_bfp16_encode.ACT_PERM): "p4" for the fused writers (norm, postnorm,
# ffn unpack) and the plain encoders, "pk" (Q4_K) by the "_pk" encoders into A's second half.
# Tried, not kept: the GPU decoding the NPU's rows to its BFP16 stream (dqbfp, one plain image), and a decoder column
# feeding 7 GEMM columns (per-format swap images inside a command); pp2048 2126 ms came from the fused form.
FUSE_FMTS = ("iq4xs", "iq3xxs", "iq3s", "q3k", "iq2xxs", "iq2xs", "q4k", "q5k")
FUSE_FFN_FMTS = ("iq4xs", "iq3xxs", "iq3s", "q3k", "iq2xxs", "iq2xs", "q5k", "q4k", "iq2s")   # the FFN block's gate / up (NP
# 4 images: room for a grid decoder's tables; Q4_K's own k-block order would need its own activation stream: Q4_K only
# in pairs, a P4 H through its gather-free decoder)
FUSE_FFN_DOWN_FMTS = ("iq4xs", "iq3xxs", "iq3s", "q4k", "q3k", "q5k", "iq2s")
DCOL_FMTS = {"iq4xs": ("IQ4_XS", "p4"), "iq4nl": ("IQ4_NL", "p4"), "q2k": ("Q2_K", "p4"), "iq3s": ("IQ3_S", "p4"), "iq3xxs": ("IQ3_XXS", "p4"), "q4k": ("Q4_K", "pk"),
             "q3k": ("Q3_K", "p4"), "iq2xxs": ("IQ2_XXS", "p4"),
             "iq2xs": ("IQ2_XS", "p4"), "q5k": ("Q5_K", "p4"), "iq2s": ("IQ2_S", "p4"),
             "bfp": ("BFP", "p4")}   # BFP: rows pre-decoded on the host (bfp_pair_rows, bfp_site_rows)
NPU_KS = (36, 36, 36, 20)   # k-blocks per pass and K-slice row; K = 1024 * passes
# the FFN block's gate / up of the formats whose filler cores hold 40 k-block slices (IQ3_S's head A ring does not
# place): the short tail slice frees the room for its deferred epilogue (gen_npu_gemm dfr); the norm before such a
# layer writes this split's stream ("norm_t_bfp_g1s.hal", dispatch row "npuffndfr_<fmt>")
NPU_DFR_FMTS, NPU_DFR_KS = ("iq4xs", "iq3xxs", "iq3s"), (40, 40, 40, 8)
# gate and up of one format without a fused GPU gate / up whose single-format deferred image fits: their own decoder
# on the GGUF rows, no pre-decoded copy (Q3_K's with its record realign rolled: npu_images' q3roll fallback)
NPU_SAME_SW = ("q5k", "q3k", "iq2s")
# (Q') the gate / up pairs of one format on the NPU's qsw image (YAH_NPU_QSW_FMTS, default q5k): super-blocks of a
# multiple of 4 bytes only (an odd one's pair-of-super-blocks axis plus the gate-to-up step need more shim DMA address
# dimensions than there are). Q5_K: pp2048 -2.3% (11-18, layers 58-60; the GPU's side still runs as separate gate
# kstore and up swiglu splits, a fused GPU gate / up with an NPU split would gain more).
QSW_FMTS = tuple(f for f in os.environ.get("YAH_NPU_QSW_FMTS", "q5k").split(",") if f)
# (Q') gate / up of NPU_COLPAIR formats on Q' column-pair images (gen_npu_q ufmt: "npu_ffnsw_<gate>_<up>_<hord>",
# dispatch rows "npuffncp_<gate>_<up>", one image family each); YAH_NPU_QCP=0: only QSW_FMTS pairs
QCP = os.environ.get("YAH_NPU_QCP", "1") == "1"
# gate and up of two of these formats: column pairs (gen_npu_gemm ufmt: each column decodes one format from its own
# tensor's rows), the tails swap halves; "npu_ffnsw_<gate>_<up>_<hord>", dispatch rows "npuffncp_<gate>_<up>")
NPU_COLPAIR = ("iq4xs", "q4k", "q3k", "iq3s", "iq3xxs", "iq2s", "iq2xs")
# The other pairs' gate / up rows (the NPU's) are pre-decoded to bfp16 (bfp_pair_rows: npu_bfp_<prefix>ffn_gate_up.bin),
# one BFP image like a single format's with the deferred epilogue ("npu_ffnsw_bfp_<hord>", dispatch rows
# "npuffnbx_<gate>_<up>"). Site tensors of a format without an NPU decoder (IQ4_NL, Q2_K attn_q) take their NPU rows
# pre-decoded too (bfp_site_rows: npu_bfp_<tensor>.bin), on the BFP site image ("npu_dcol_bfp_<K>", dispatch rows
# "npusitebx_<tensor>").


def colpair(l):
    """A layer whose gate and up run as column pairs (NPU_COLPAIR). Q' (QCP): one format too but IQ4_XS's (the
    formats of odd super-blocks have no single-format qsw image: pairs read one tensor a column)."""
    g, u = l["ffn_gate"][0], l["ffn_up"][0]
    iq4 = os.environ.get("YAH_NPU_QCP_IQ4") == "1"   # (diag: IQ4_XS / IQ4_XS layers as column pairs too)
    return (g != u or (QCP and NPU_Q and (g != "iq4xs" or iq4))) and {g, u} <= set(NPU_COLPAIR)


def gu_fmt(fmt):
    """The NPU decoder of the FFN block's gate / up rows of fmt."""
    return DCOL_FMTS[fmt][0]


# the fused site and down images of the formats whose four cores each decode one super-block of every unit (gen_npu_gemm
# kchain), the gate's waiter and relay on tails (gtail: rows 1 decode); the grid formats' helper rings after the A rings
# (hlate: their tables leave the mids no contiguous A ring otherwise)
NPU_KCHAIN_FMTS = ("iq4xs", "q4k", "q3k", "iq3s", "iq3xxs", "q5k")
KCHAIN = lambda fmt: dict(kchain=1, hcap=2, gtail=1, hlate=int(fmt in ("iq3s", "iq3xxs"))) if fmt in NPU_KCHAIN_FMTS else {}


def npu_cfg(B, ks, passes, **kw):
    """A fused NPU image's gen_npu_gemm config: gated, its gate's waiter and relay at a job's first firing (the setup
    runs no call, so an image switch inside a command costs no dummy call; dispatch.txt "npugfirst")."""
    return GN.Config(NPU_COLS, B // 64, ks, passes, mu=1, gate=GN.GATE_SUPPLY, gwest=1, ghelp=1, fuse=2, **kw)


NPU_COLS = 8
NPU_NP = 4   # 16-row slabs per column of a call
NPU_ROWS = NPU_COLS * 16 * NPU_NP   # output rows per NPU call
NPU_SITES = {"qkv": ("kstore", 640, 20), "gate": ("kstore", 384, 20), "q": ("kqg", 768, 20), "out": ("kres", 320, 24)}
NPU_PASSES = 5   # K = 5120 per call (gen_npu_gemm passes)
# YAH_NPU_Q=<R> (experimental): the site images in layout Q' (gen_npu_q: every core a 64 x 64 tile over the whole K, no
# cascade; R array rows split the M blocks), one K slice of 16 k-blocks over 40 passes; IQ4_XS sites only, no FFN block
NPU_Q = int(os.environ.get("YAH_NPU_Q", "0"))
if NPU_Q:
    NPU_KS, NPU_PASSES = (16,), 40
# YAH_NPU_QP1K=1 (with YAH_NPU_Q=4): the 64x80 P1K image (gen_npu_q P1K_*): 640-row calls, ks 32 x 20 passes
NPU_QP1K = NPU_Q == 4 and os.environ.get("YAH_NPU_QP1K") == "1"
if NPU_QP1K:
    import gen_npu_q as _Q
    NPU_NP, NPU_KS, NPU_PASSES = _Q.P1K_NP, _Q.P1K_KS, _Q.P1K_PASSES
    NPU_ROWS = NPU_COLS * 16 * NPU_NP
NPU_KCALL = 8 * sum(NPU_KS) * NPU_PASSES   # K per call (5120)
# YAH_NPU_FFNBLK=<F> (F = 1024 n): the NPU owns the FFN's last F features of every layer whose gate, up and down are in
# FUSE_FFN_FMTS / FUSE_FFN_DOWN_FMTS (and ffn_up follows ffn_gate). Its gate / up calls (gen_npu_gemm swiglu, NP 4, raw
# rows straight from the model; call j (F / 256 of them) column c features 256 j + 32 c + [0, 32) of F, swcol 32) apply
# swiglu and write H into C: bfp16 fragments as [M block][pass][16-token slab][128 k-blocks] (hpass). Its down calls
# (ain "h", 512 rows) write their partials. The GPU runs the other features through the fused ffn's split (".npu.hal",
# nn = F) and down over them ("npuffnrem_<fmt>.hal", a K window, f32); npu_unpack_ffnblk.hal adds hidden + both after
# the job. Down's K parts: F's passes in the fewest parts of at most 5 (a fused panel holds a part), sizes balanced.
# A command switches image families at job boundaries (LoomNpu FusedCommand: the image's setup).
# With YAH_NPU_QP1K (layout Q', F's K parts of one size S, ffnblk_parts: F = 5120, 8192, 10240, ..): only layers
# whose gate, up and down are all IQ4_XS; gate / up are Q' qsw images (NP 4, F / 256 calls of 256 features, the
# swiglu epilogue writing H straight in down's A layout: H in parts of S features, each [M block][S / 256 calls][...]
# as one K = S down reads it, call j at part j / (S / 256), gen_npu_q), down is the plain P1K site image (K = S) over
# its part of H (8 calls of 640 rows a part, K window [M - F, M); dispatch.txt "npuffnq").
# Tried, not kept: the GPU window's residual epilogue waiting for the NPU (npuffnres): only 20 of its workgroups fit at
# once, so while the NPU is the late one the rest of the GPU's down starts after the NPU's (~5 ms a layer).
NPU_FFNBLK = int(os.environ.get("YAH_NPU_FFNBLK", "0"))
# YAH_NPU_FFNBLKS=<F> (Q', default YAH_NPU_FFNBLK): the F of the single-format layers (npu_ffnblk "ok": IQ4_XS gate and
# up, dispatch.txt "npuffnblk"); the pairs keep YAH_NPU_FFNBLK ("npuffnpair"). Their downs then read their own F's K
# segments: dispatch.txt "npudnsegl_<layer> <F>", the down GEMMs of that layout "<hal>_kseg<widths>" (FFNBLKS_DOWNS).
NPU_FFNBLKS = int(os.environ.get("YAH_NPU_FFNBLKS", "0")) or NPU_FFNBLK
# YAH_NPU_FFNSW5 (Q', default 1 when YAH_NPU_FFNBLKS's K parts are whole 1280-feature groups): the single-format
# (IQ4_XS) layers' gate / up at NP 5, 320 features a call (gen_npu_q qsw5: call j's 4-call group (j % 16) / 4 of
# K part j / 16, its 64 features j % 4 of each of 5 runs of 256); dispatch.txt "npuffnsw5 320", "npubytes5_ffnsw".
# 0: NP 4, 256 a call.
FFNSW5 = os.environ.get("YAH_NPU_FFNSW5", "1") == "1"
RSEG_GEMMS = set()   # (fmt, m_tiles, k_blocks) of the FFNSW5 layers' gate / up (row-segmented, gen_gemv RSEG)
FFNBLKS_DOWNS = set()   # (fmt, m_tiles, k_blocks) of the single-format layers' downs when their F differs (npu_ffnblk)
KSEG_F = [0]   # (kseg_f) the F whose K segments kseg_for gives the K-segmented downs, 0: NPU_FFNBLK's (DNSEG)
# YAH_NPU_DNSEG (Q', default 1): ffn_down's rows in K segments, reordered in place at load ([segment][row][its blocks]:
# the GPU's K part, then each NPU K part, gen_gemv.kseg_layout; LoomWeights, dispatch.txt "npudnseg_<fmt> <F>"). The
# down images read a part's rows contiguous (kraw 0): one raw block a column a call instead of 80 runs of S / 256
# super-blocks at the 17408-wide row stride (11-19: the strided read's DDR stalls, ~70K cycles a down call, ~22K
# contiguous). Every GPU kernel over those weights reads each block from its segment (gen_gemm_tile.KSEG, gen_gemv
# kseg; bit-identical results); no memory beyond the model's.
NPU_DNSEG = os.environ.get("YAH_NPU_DNSEG", "1") == "1"   # (0: the GGUF rows, 17408 wide)
DNSEG = {}   # {(fmt, m_tiles, k_blocks): K segments} of the K-segmented down weights (main, dnseg_plan)


def dnseg_plan(rows):
    """DNSEG for YAH_NPU_DNSEG with the Q' FFN block: every ffn_down whose format both GEMM generators read with
    K segments (one segment layout per format and shape: each format's down GEMMs and GEMVs read them all)."""
    import gen_gemm_tile as TG
    if not (NPU_DNSEG and NPU_FFNBLK and NPU_Q and NPU_QP1K):
        return
    for nm, dims, ty in rows:
        if nm.endswith(".ffn_down.weight") and E.FMT.get(ty):
            fmt, mt, kb = E.FMT[ty][0], dims[1] // 16, dims[0] // E.FMT[ty][2]
            if fmt in TG.G.FMTS and TG.G.FMTS[fmt].get("kdiv", 1) == 1 and gen_gemv.FMT.get(fmt, (0,))[0] == 256:
                DNSEG[(fmt, mt, kb)] = dn_kseg(kb, NPU_FFNBLK)


@contextlib.contextmanager
def kseg_f(F):
    """kseg_for's K segments for F (the single-format layers' downs, YAH_NPU_FFNBLKS) while generating."""
    KSEG_F[0] = F if F != NPU_FFNBLK else 0
    try:
        yield
    finally:
        KSEG_F[0] = 0


@contextlib.contextmanager
def kseg_for(fmt, mt, kb):
    """gen_gemm_tile.KSEG while a GEMM over (fmt, mt, kb) weights is generated (kb: whole rows' k_blocks)."""
    import gen_gemm_tile as TG
    TG.KSEG = DNSEG.get((fmt, mt, kb))
    if TG.KSEG and KSEG_F[0]:
        TG.KSEG = dn_kseg(kb, KSEG_F[0])
    try:
        yield
    finally:
        TG.KSEG = None
# (Q' P1K) the formats whose site images are NP 4 (512-row calls: the grid decoders' tables, Q4_K's program and Q5_K's
# raw ring do not fit an NP 5 core): those layers' sites take NPU rows rounded down to 512 (npu_rows4) and their own
# unpacks / conv split at TN 64 ("<hal>_4.hal"); dispatch.txt "npusplit4_<site>", "npurows4", "npubytes4_5120",
# "npunp4_<fmt>" (LoomPrefill::Np4)
QNP4_FMTS = tuple(os.environ.get("YAH_NPU_NP4", "iq3s,iq3xxs,iq2s,iq2xs,q3k,q4k,q5k").split(","))   # (YAH_NPU_NP4: a subset)
QJOB_ENV = {}   # per NPU image name: environment knobs of its generator only (npu_images source)
# YAH_NPU_RSWAP_EMIT=1 (row 5 stepping stone, Q' NP 4 gate / up, i.e. YAH_NPU_FFNSW5=0): also the swapped IQ4_XS
# gate / up image ("npu_ffnswap_iq4xs": gen_npu_q GN_QSWAP + GN_QSWHT, fuse 0: the weights as its stream, GPU-decoded
# into A, 512 tokens a call as its panel in W, H in today's down A layout), the GPU decoders of the NPU's rows
# ("npu_swdec_iq4xs_<g|u>.hal", gen_iq4xs_bfp stream) and the token panel encode ("npu_swpanel.hal", gen_bfp16_encode
# wgt p1k); dispatch.txt "npurswap <F>", "npubytes_ffnswap". The engine uses them with YAH_NPU_RSWAP=1 only.
# The Q' pairs' other gate / up formats get their decoders into the same stream too ("npu_swdec_<fmt>_<g|u>.hal",
# gen_kq_bfp stream): the swap image reads bfp16 only, so it serves every format.
RSWAP_EMIT = os.environ.get("YAH_NPU_RSWAP_EMIT") == "1"
RSWAP_KQ = {"q4k": "Q4_K", "q5k": "Q5_K", "q3k": "Q3_K", "iq3s": "IQ3_S", "iq3xxs": "IQ3_XXS", "iq2xs": "IQ2_XS",
            "iq2s": "IQ2_S"}   # (gen_kq_bfp's formats)
RSWAP_KS, RSWAP_PASSES = (32,), 20   # (the swap's stream and panel split: swq.py's, checked)
# YAH_NPU_RSWAP_NB=<n> (with RSWAP_EMIT): the swap image's weight M blocks (64 stream rows) a call, a multiple of 32
# (today's 1024 features a call); the engine reads it from npubytes_ffnswap's A bytes. 128: 2 calls a token panel
# at F 8192, so a job loads each panel twice instead of 8 times (<= 63 blocks a row a call: 252)
RSWAP_NB = int(os.environ.get("YAH_NPU_RSWAP_NB", "32"))
RSWAP_ENV = dict(GN_QSWAP="1", GN_QSWHT="1", GN_QFKH="0", GN_QPSHARE="0", GN_QF4R="0", GN_QPREFILL="0",
                 GN_QFGUARD="0", GN_QF4RSWAP="0", GN_QF4RCOPY="0", GN_QFEGR2="0")   # (no fill at fuse 0)
RSWAP_DROP = {"LOOM_EXP_PANEL_STREAM", "LOOM_EXP_PANEL_STREAM_SINGLE", "LOOM_EXP_REPLAY_GROUP"}


def ffnblk_parts(F):
    """(first pass, passes) of down's K parts over F features (engine/model/loom_prefill.hpp FfnBlkParts)."""
    assert F % 1024 == 0, F
    P = F // 1024
    n = -(-P // 5)
    sizes = [P // n + (i < P % n) for i in range(n)]
    return tuple((sum(sizes[:i]), sizes[i]) for i in range(n))


def ffnblk_units(F):
    """(first unit, units, padded units) of down's K parts over F features, a unit 256 features (a super-block;
    loom_prefill.hpp FfnBlkUnits). F of whole passes: ffnblk_parts. Else (Q' FFNSW5, F of whole 1280-feature groups):
    the fewest parts of at most 4 groups, the shorter ones first; a part of 15 units reads 16 (its down image's K): the
    16th is the next row's first block (DNSEG) times H's zero pad unit (npu_zpad_<F>.hal)."""
    if F % 1024 == 0:
        return tuple((4 * p0, 4 * n, 4 * n) for p0, n in ffnblk_parts(F))
    assert F % 1280 == 0, F
    G = F // 1280
    n = -(-G // 4)
    g = [G // n + (i >= n - G % n) for i in range(n)]
    return tuple((5 * sum(g[:i]), 5 * g[i], -(-5 * g[i] // 4) * 4) for i in range(n))


def ffnblk_apass(F):
    """(Q') H's units a 64-token M block when F's down parts differ in padded size: H one layout [M block][units], a
    part's A a window of it (gen_npu_gemm apass; the gate / up image's hpass); else 0: H in parts, [part][M block][its
    units]."""
    us = ffnblk_units(F)
    return sum(u[2] for u in us) if len({u[2] for u in us}) > 1 else 0


def dn_tag(r, p, ap):
    """A down part image's name tag ("npu_ffndn<tag>_<fmt>"): its passes (A its own, no pad), else
    "u<units>_<padded units>_<apass>"."""
    return "%d" % (p // 4) if r == p and not ap else "u%d_%d_%d" % (r, p, ap)


def ffnblk_pads(F, B):
    """H's pad units (zero, the padded parts' last units) as (first unit, units apart) over the 64-token M blocks."""
    us, ap, nb, out = ffnblk_units(F), ffnblk_apass(F), B // 64, []
    for i, (_, r, p) in enumerate(us):
        base = sum(u[2] for u in us[:i]) if ap else i * nb * p
        out.extend((base + k, ap or p) for k in range(r, p))
    return out


def dn_kseg(kb, F):
    """The K segments of a down whose last F columns the NPU reads (DNSEG): the GPU's, then each part's."""
    return gen_gemv.kseg_layout_units(kb, F, ffnblk_units(F))


def npu_chunks(site):
    """(k offset, passes) of the NPU's K chunks of a site; the GPU computes the rest of K (npu_rem) for the NPU's rows."""
    K = NPU_SITES[site][2] * 256
    return [(NPU_KCALL * i, NPU_PASSES) for i in range(K // NPU_KCALL)]


def npu_rem(site):
    """The K columns past the NPU's chunks (out: 1024 of 6144, down: 2048 of 17408): the GPU's npurem_<site>_<fmt>.hal."""
    return NPU_SITES[site][2] * 256 % NPU_KCALL


def npu_rem_tile(fmt, kbw, kbt, kind="kstore"):
    """The K remainder's tile (gen_gemm_tile.KWIN, kbw of kbt blocks): the afrag kstore (or kres) with the knobs of the
    format's K = kbt * 256 kres (emit_prefill_pp.AF; its residual and persistence knobs dropped), the plain afrag
    tile (straight-line k steps, as afrag needs) for a format with no tuned kres."""
    import gen_gemm_tile as TG
    knobs = dict(AF.get((fmt, "kres", 320, kbt)) or AF.get((fmt, "kres", 320, 24)) or dict(ksl=True))
    knobs.pop("persist", None)
    knobs["respre"] = 0
    return dataclasses.replace(TG.default_tile(fmt, kind, kbw), **AF_TILE, **knobs)


# The out site's GPU kres split (".af.npu.hal"): 1920 / 2048 rows give 15 / 16 row blocks of 128, so the AF tile's
# 4 token tiles of 512 make 60 / 64 workgroups on 40 CUs (1.5 / 1.6 rounds). 256-token tiles with 2 x 8 waves: 120 / 128
# workgroups. 11-19 standalone (n1118/kbench.py, B 2048, K 6144): IQ4_XS 1436-1678 -> 1172-1226 us, Q4_K 1691-1733 ->
# 1343-1355, Q5_K 1676-1706 -> 1349-1357, Q3_K 1847-1913 -> 1393-1397; bit-identical (no knob changes the MMA order).
# In the engine (pp2048): Q4_K / Q5_K 1593 / 1611 -> 1425 / 1442 us, IQ4_XS 1273 -> 1270.
# YAH_NPU_TILE_<SITE>="bm,bn,wm,wn" (site OUT, QKV, GATE, Q): the split's tile at that site; 0: the AF tile.
SPLIT_TILE_DEFAULT = {"out": "128,256,2,8"}


def split_tile(site):
    v = os.environ.get("YAH_NPU_TILE_" + site.upper(), SPLIT_TILE_DEFAULT.get(site, "0"))
    return dict(zip(("bm", "bn", "wm", "wn"), map(int, v.split(",")))) if v != "0" else None


# YAH_NPU_NOONORM=1: no "npu_onorm" (the out site's sum as its own unpack, then the FFN block's norm)
NO_ONORM = os.environ.get("YAH_NPU_NOONORM") == "1"


def fit_rowgrp(t, mt):
    """Tile t over mt row tiles: NPU splits can leave GPU / remainder row counts the tile's row group need not divide;
    the largest of its row group, 4, 2, 1 that does."""
    if mt % t.rowgrp == 0:
        return t
    import gen_gemm_tile as TG
    for r in (4, 2, 1):
        if r >= t.rowgrp or mt % r:
            continue
        bm = t.bm * r // t.rowgrp   # rowgrp = bm / 16 (ffn: bm / 32)
        for wm in range(min(t.wm, bm // 16), 0, -1):
            c = dataclasses.replace(t, bm=bm, wm=wm)
            try:
                TG.check(c)
                return c
            except ValueError:
                pass
    raise SystemExit("no row group of %s divides %d row tiles" % (t, mt))


QSITE_FMTS = ("iq4xs", "iq4nl", "q2k")   # Q' site images at NPU_NP (the others: NP 4, QNP4_FMTS)


def np4(fmt):
    """fmt's Q' site images are NP 4 (QNP4_FMTS)."""
    return NPU_QP1K and fmt in QNP4_FMTS


# YAH_NPU_SPLIT4="<site>=<rows>,...": the NP 4 formats' layers' NPU rows (multiples of 512; a site not named: NPU_SPLIT's
# rows rounded down). Their calls take ~20-40% more NPU cycles a row than NP 5's (each 512-row call streams the whole A,
# as a 640-row call does). 11-18 pp2048 at p2ffq's split (night log 41): qkv 4608 -2.4% vs IQ4_XS-only sites; the NPU
# spans' balance (qkv 4608, gate 2560, q 4096, out 2048) +7.0%; q 4096 alone +0.9%.
NPU_SPLIT4 = dict((k, int(v)) for k, v in (x.split("=") for x in os.environ.get("YAH_NPU_SPLIT4", "qkv=4608").split(",")
                                           if x))


def npu_rows4(site):
    """A site's NPU rows in the layers of NP 4 formats (np4)."""
    n = NPU_SPLIT4.get(site, NPU_SPLIT[site] // 512 * 512)
    assert n % 512 == 0 and 0 <= n < NPU_SITES[site][1] * 16, (site, n)
    return n


def npu_split(rows, combos, B, outdir):
    import gen_bfp16_encode as GE
    import gen_gemm_tile as TG
    import gen_npu_unpack as GU
    bad = set(NPU_SPLIT) - set(NPU_SITES)
    assert not bad, "unknown NPU split sites %s" % sorted(bad)
    assert all(v % NPU_ROWS == 0 and 0 < v < NPU_SITES[k][1] * 16 for k, v in NPU_SPLIT.items()) and B % 512 == 0
    assert NPU_SPLIT.get("q", 0) % 512 == 0, "q: whole heads"
    GE.ACT_PERM = "p4"
    out = []
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)

    def split(fmt, kind, mt, kb, t, name, nn, persist=None, tokens=B, wrap=None):
        """The GPU's share: rows [0, mt * 16 - nn) of the GEMM HAL name (tile t) at the full stride, over tokens.
        wrap(text, n, nwg, lanes): a transform of the GEMM's text that adds buffers (gen_r5fuse.fuse)."""
        mtg = mt - nn // 16
        tf = fit_rowgrp(t, mtg)
        if tf is not t and persist is not None:   # the persistent kres keeps its row blocks: the tiled form runs instead
            return
        assert mtg % tf.rowgrp == 0 and tokens % tf.bn == 0, (name, nn)
        TG.check(tf)
        if persist is not None:
            gen = lambda f, k: gen_kres_persist.persist(TG.gen(f, "kres", persist, False),
                                                       TG.gen(f, "kstore", dataclasses.replace(persist, respre=0), False),
                                                       tokens // tf.bn, place=persist.pslice)
        elif wrap is not None:
            gen = lambda f, k: wrap(TG.gen(f, k, tf, False), mt * 16, mtg // tf.rowgrp * (tokens // tf.bn), tf.lanes)
        else:
            gen = lambda f, k: TG.gen(f, k, tf, False)
        try:
            out.append(_emit_gen(gen, tf.bn, fmt, mtg, kb, tokens, name[:-4] + ".npu.hal", outdir, kind, tf.rowgrp,
                                 ostride=mt * 16, extra="_wrap" if wrap is not None else "",
                                 vtry=gen_r5fuse.VGPR_TRY if wrap is not None else None))
        except (SystemExit, RuntimeError, ValueError) as ex:
            if tf is t:
                raise
            # a split's narrower row group may not emit (spills): the engine runs that form unsplit
            print("npu split %s: not emitted with row group %d (%s)" % (name, tf.rowgrp, str(ex)[:120]))

    def variants(fmt, kind, mt, kb, base, nn, site):
        """Every form of GEMM base the set emits (tile_kstore, afrag_variants): its split."""
        t = TG.default_tile(fmt, kind, kb)
        if base[:-4] in tiles()["tiles"]:
            t = dataclasses.replace(t, **tiles()["tiles"][base[:-4]])
        if fmt in TILE_FMTS and mt % t.rowgrp == 0:
            split(fmt, kind, mt, kb, t, base, nn)
        knobs = AF.get((fmt, kind, mt, kb))
        if knobs is not None and B % AF_TILE["bn"] == 0:
            knobs = dict(knobs)
            persist = knobs.pop("persist", False)
            ta = dataclasses.replace(TG.default_tile(fmt, kind, kb), **AF_TILE, **knobs)
            ts = ta
            st = split_tile(site)
            if st and B % st["bn"] == 0:
                ts = dataclasses.replace(ta, **st)
                try:
                    TG.check(ts)
                except ValueError:
                    ts = ta
            split(fmt, kind, mt, kb, ts, base[:-4] + ".af.hal", nn)
            if kind == "swiglu":
                split(fmt, kind, mt, kb, dataclasses.replace(ta, tout=True), base[:-4] + ".af.to.hal", nn)
            if persist and kind == "kres" and B // ta.bn > 1 and 2 * kb >= gen_kres_persist.SLICES:
                pt = dataclasses.replace(ta, **persist) if isinstance(persist, dict) else ta
                # fewer row blocks than CUs: the persistent kres over all but the last token tile, run beside the tiled
                # kres on that tile (LoomPrefill::RunResidual)
                if B // ta.bn > 2:
                    split(fmt, kind, mt, kb, ta, base[:-4] + ".af.p3.hal", nn, persist=pt, tokens=B - ta.bn)

    done = set()
    images, bsites, sites4 = set(), set(), set()
    for kind, fmt, port, mt, kb in sorted(combos):
        for site, (skind, smt, skb) in NPU_SITES.items():
            if site not in NPU_SPLIT or mt != smt or kb * qk_of(fmt) != skb * 256 or fmt not in TILE_FMTS:
                continue
            nn = npu_rows4(site) if np4(fmt) else NPU_SPLIT[site]
            if not nn:
                continue
            if skind == "kres":
                if kind != "residual":
                    continue
                variants(fmt, "kres", mt, kb, "gemm_kres_%s_%d_%d.hal" % (fmt, mt, kb), nn, site)
            elif kind == "kstore":
                variants(fmt, skind, mt, kb, "gemm_%s_%s_%d_%d.hal" % (skind, fmt, mt, kb), nn, site)
            else:
                continue
            chunked = len(npu_chunks(site)) > 1 or npu_rem(site)
            if NPU_Q and (fmt in QSITE_FMTS or np4(fmt)):   # (Q': IQ4_XS, IQ4_NL, the NP 4 formats)
                images.add((fmt, skb * 256))
                if np4(fmt):
                    sites4.add(site)
            elif fmt in FUSE_FMTS and not NPU_Q:
                images.add((fmt, skb * 256))
            elif not chunked and skind != "kres" and not NPU_Q:   # (BFP site rows; Q': those layers stay on the GPU)
                images.add(("bfp", skb * 256))
                bsites.update((t, nn) for t in combos[(kind, fmt, port, mt, kb)])
            if npu_rem(site) and (site, fmt) not in done:   # the GPU's K remainder of the NPU's rows (f32 [B][nn])
                done.add((site, fmt))
                kbw = npu_rem(site) // 256
                tr = fit_rowgrp(npu_rem_tile(fmt, kbw, kb), nn // 16)

                def gen_rem(f, k, tr=tr, kb=kb, kbw=kbw):
                    TG.KWIN = (kb - kbw, kb)
                    try:
                        return TG.gen(f, "kstore", tr, False)
                    finally:
                        TG.KWIN = None
                assert nn // 16 not in O16_MT, "the remainder writes f32"
                out.append(_emit_gen(gen_rem, tr.bn, fmt, nn // 16, kbw, B, "npurem_%s_%s.hal" % (site, fmt), outdir,
                                     "kstore", tr.rowgrp, kfull=kb))
    # activation encoders: per K (chunked: per chunk), row-major and fragment-major input
    encs = set()
    for site in NPU_SPLIT:
        K = NPU_SITES[site][2] * 256
        chunks = npu_chunks(site)
        chunked = len(chunks) > 1 or npu_rem(site)
        for ci, chunk in enumerate(chunks):
            encs.add((K, ci if chunked else None, chunk))
    if NPU_FFNBLK:   # the FFN block's gate / up activations
        encs.add((5120, None, (0, NPU_PASSES)))
    for K, ci, (k_off, passes) in sorted(encs, key=lambda x: (x[0], -1 if x[1] is None else x[1])):
        for tiled in (False, True):
            for perm in ("p4", "pk"):
                name = "npu_enc_%d%s%s%s.hal" % (K, "" if ci is None else "_c%d" % ci, "_t" if tiled else "",
                                                 "_pk" if perm == "pk" else "")
                src = os.path.join(tmp, name[:-4] + ".loom")
                GE.ACT_PERM = perm
                try:
                    open(src, "w").write(GE.gen("act", B, list(NPU_KS), passes, tiled=tiled, k_off=k_off,
                                                k_src=K if 8 * sum(NPU_KS) * passes != K else None))
                finally:
                    GE.ACT_PERM = "p4"
                E.emit(src, ["nop=0"], name, outdir)   # emit_hal.py wants a config; the encoder has none
                out.append((name, B // 8 * (passes * sum(NPU_KS)) // GE.WG, GE.WG, 0))
    # the norms that feed K = 5120 sites also write the NPU's BFP16 input (gen_half_norm.gen_split bfp): "<norm>_bfp.hal"
    if {"qkv", "q"} & set(NPU_SPLIT):
        # (and "<norm>_bfp_pk.hal": the Q4_K-order stream too, into A's second half, for sites with a Q4_K weight)
        for tiled, name, pk in [(t, n[:-4] + pk * "_pk" + ".hal", pk) for pk in (False, True)
                                for t, n in ((False, "norm_bfp.hal"), (True, "norm_t_bfp.hal"),
                                             ("both", "norm_rt_bfp.hal"))]:
            src = os.path.join(tmp, name[:-4] + ".loom")
            open(src, "w").write(gen_half_norm.gen_split(5120, wpr=4, split=NORM_SPLIT, tiled=tiled,
                                                         bfp=(B, list(NPU_KS), NPU_PASSES, pk)))
            E.emit(src, ["yah_half_norm.rows=%d" % B, "yah_half_norm.dim=5120", "yah_half_norm.eps=1e-06",
                         "yah_half_norm.fused=0"], name, outdir)
        out.append(("npunormbfp", 1, 0, 0))
        out.append(("npunormbfppk", 1, 0, 0))
    # the DeltaNet postnorm over heads 0 .. DNSPLIT[0] - 1 (exactly ssm_out's NPU K chunk) also writes the NPU's BFP16
    # input (gen_postnorm_bfp): "postnorm[_t]_a_bfp.hal"
    if "out" in NPU_SPLIT and DNSPLIT and DNSPLIT[0] * 128 == NPU_KCALL:
        import gen_postnorm_bfp
        for tiled, name in ((False, "postnorm_a_bfp.hal"), (True, "postnorm_t_a_bfp.hal")):
            src = os.path.join(tmp, name[:-4] + ".loom")
            open(src, "w").write(gen_postnorm_bfp.gen(B, 0, DNSPLIT[0], list(NPU_KS), NPU_PASSES, tiled, 384 in O16_MT))
            E.emit(src, ["yah_ssm_postnorm_fp16.head_count=%d" % (DNSPLIT[0] * B)], name, outdir)
        out.append(("npupostnormbfp", 1, 0, 0))
    # unpacks (the NP 4 formats' sites: TN 64, "_4")
    def unpacks(site, n, sfx):
        skind, mt, kb = NPU_SITES[site]
        N = mt * 16
        f = {"qkv": dict(out16=True), "gate": dict(out16=True), "q": dict(qg=True),
             "out": dict(resid=True, parts=len(npu_chunks("out")), rem=True)}[site]
        name = "npu_unpack_%s%s.hal" % (site, sfx)
        src = os.path.join(tmp, name[:-4] + ".loom")
        open(src, "w").write(GU.gen(B, n // GN.TN, N // 2 if site == "q" else N, N - n, **f))
        E.emit(src, ["nop=0"], name, outdir)
        out.append((name, B * n // 8 // GU.wg(n // GN.TN), GU.wg(n // GN.TN), 0))
        if site == "out" and NPU_FFNBLK and not NO_ONORM:
            # the same sum fused into the FFN block's norm ("norm_t_bfp_g1": its fragment-major f16 and the
            # one-replay-group stream): "npu_onorm<sfx>.hal", gen_half_norm.gen_osum (LoomPrefill::RunNorm)
            import gen_half_norm
            groups, perm = GE.GROUPS, GE.ACT_PERM
            GE.GROUPS, GE.ACT_PERM = (NPU_PASSES,), "p4"
            try:
                text = gen_half_norm.gen_osum(B, N - n, n, f["parts"], ONORM_SPLIT, (B, list(NPU_KS), NPU_PASSES))
            finally:
                GE.GROUPS, GE.ACT_PERM = groups, perm
            oname = "npu_onorm%s.hal" % sfx
            src = os.path.join(tmp, oname[:-4] + ".loom")
            open(src, "w").write(text)
            E.emit(src, ["yah_half_norm.rows=%d" % B, "yah_half_norm.dim=5120", "yah_half_norm.eps=1e-06",
                         "yah_half_norm.fused=0"], oname, outdir)
            out.append((oname, B // 4, 32 * 4 * ONORM_SPLIT, 0))   # 4 rows a workgroup
        if site == "qkv":
            # the conv reads the NPU's v channels straight from C (gen_conv_kq.gen_c) and the GPU's from the f16 output
            # (gen climit); the unpack writes only the last 8 tokens, the conv ring that prep_ab reads
            for hal, text, cfg in (
                    ("npu_unpack_qkv_tail%s.hal" % sfx, GU.gen(B, n // GN.TN, N, N - n, out16=True, last8=True),
                     ["nop=0"]),
                    ("convkq_g%s.hal" % sfx, gen_conv_kq.gen(N - n), ["yah_ssm_conv_kq.batch=%d" % B,
                                                                      "yah_ssm_conv_kq.qkv_dim=10240",
                                                                      "yah_ssm_conv_kq.num_key_heads=16"]),
                    ("convkq_c%s.hal" % sfx, gen_conv_kq.gen_c(N - n, n // GN.TN, B), ["nop=0"])):
                src = os.path.join(tmp, hal[:-4] + ".loom")
                open(src, "w").write(text)
                E.emit(src, cfg, hal, outdir)
            out.append(("npu_unpack_qkv_tail%s.hal" % sfx, n // GU.wg(n // GN.TN), GU.wg(n // GN.TN), 0))   # 8 tokens x n / 8 items
            out.append(("convkq_g%s.hal" % sfx, gen_conv_kq.tiles(N - n), 256, 0))
            out.append(("convkq_c%s.hal" % sfx, n // gen_conv_kq.cw(n), gen_conv_kq.cw(n), B // gen_conv_kq.TC))
        out.append(("npusplit%s_%s" % (sfx[1:], site), n, 0, 0))
    for site, n in sorted(NPU_SPLIT.items()):
        unpacks(site, n, "")
    for site in sorted(sites4):
        with GN.np_override(4):
            unpacks(site, npu_rows4(site), "_4")
    # one graph per chunk: the GPU waits for each NPU job on a flag word (gen_npu_unpack.gen_flag_wait)
    src = os.path.join(tmp, "npu_flag_wait.loom")
    open(src, "w").write(GU.gen_flag_wait())
    E.emit(src, ["nop=0"], "npu_flag_wait.hal", outdir)
    out.append(("npu_flag_wait.hal", 1, 32, 0))
    src = os.path.join(tmp, "npu_flag_wait_log.loom")   # the same wait, logging its poll count (YAH_NPU_WAITLOG)
    open(src, "w").write(GU.gen_flag_wait(log=True))
    E.emit(src, ["nop=0"], "npu_flag_wait_log.hal", outdir)
    out.append(("npu_flag_wait_log.hal", 1, 32, 0))
    out.append(("npugate", GN.GATE_SUPPLY, GN.GATE_CALLS, GN.GATE_RECORD))
    bfp_site_rows(bsites, outdir)
    out.extend(("npusitebx_" + t, 1, 0, 0) for t, _ in sorted(bsites))
    extra = npu_ffnblk(rows, B, tmp, outdir, out, split, variants) if NPU_FFNBLK else []
    out.extend(npu_images(images, B, tmp, outdir, extra))
    GE.ACT_PERM = None
    return out


def npu_images(images, B, tmp, outdir, extra=()):
    """The NPU images: per (format, site K) the gated GEMM reading raw rows of that K; dispatch.txt "npudcol <cols>",
    "npubytes_5120" and "npurows". extra: more (cfg, name, np) images (npu_ffnblk; np: their gen_npu_gemm NP)."""
    import concurrent.futures
    sys.path.insert(0, os.path.dirname(HERE))
    import hrx_paths
    assert images, "an NPU split needs a site the NPU computes"
    if NPU_Q:   # (layout Q': gen_npu_q)
        # (IQ4_XS and the NP 4 formats: the layers of other formats keep the site on the GPU, LoomPrefill::DcolImage)
        images = {(f, K) for f, K in images if f in QSITE_FMTS or np4(f)}
        import gen_npu_q as Q
        Q.defaults(p1k=NPU_QP1K)
        Q.QROWS = NPU_Q
        jobs = [(GN.Config(NPU_COLS, B // 64, NPU_KS, NPU_PASSES, mu=1, gate=GN.GATE_SUPPLY, fuse=2, kraw=K,
                           fmt=DCOL_FMTS[fmt][0], rawdma=1), "npu_dcol_%s_%d" % (fmt, K),
                 4 if np4(fmt) else NPU_NP if NPU_QP1K else GN.NP)
                for fmt, K in sorted(images)]
    else:
        jobs = [(npu_cfg(B, NPU_KS, NPU_PASSES, kraw=K, fmt=DCOL_FMTS[fmt][0], **KCHAIN(fmt)),
                 "npu_dcol_%s_%d" % (fmt, K), GN.NP) for fmt, K in sorted(images)]
    jobs += list(extra)

    # sources and environments first (gen_npu_gemm's NP is module state), then the compiles in parallel
    def source(cfg, name, np_):
        old = {k: os.environ.get(k) for k in QJOB_ENV.get(name, {})}   # (the image's own knobs, QJOB_ENV)
        os.environ.update(QJOB_ENV.get(name, {}))
        GN.qf_p1kd_layout()
        tl = name in os.environ.get("GN_QCYCIMG", "").split(",")   # (diag: these images with GN_QCYC stamps, GN.qtlp)
        qswap = GN.QSWAP
        try:
            GN.QSWAP = qswap or name.startswith("npu_ffnswap_")   # (RSWAP_EMIT: gen_npu_gemm reads GN_QSWAP at import)
            GN.QCYC = GN.QCYC or tl
            GN.QFCYC = tl and os.environ.get("GN_QFCYC") == "1"
            with GN.np_override(np_):
                src = os.path.join(tmp, name + ".loom")
                text = Q.gen(cfg) if NPU_Q else GN.gen(cfg)
                open(src, "w").write(GN.qtlp(text) if tl and os.environ.get("GN_QTLP") == "1" else text)
                env_ = dict(hrx_paths.env(), **(Q.loom_env(cfg) if NPU_Q else GN.loom_env(cfg)))
                if name.startswith("npu_ffnswap_"):
                    for k_, v_ in Q.P1K_LOOM.items():
                        if k_ not in RSWAP_DROP:
                            env_.setdefault(k_, v_)
                    for k_ in RSWAP_DROP:
                        env_.pop(k_, None)
                return cfg, name, src, env_
        finally:
            GN.QSWAP = qswap
            GN.QCYC = GN.QCYC and not tl
            GN.QFCYC = False
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            GN.qf_p1kd_layout()

    def compile_one(b):
        cfg, name, src, env = b
        # YAH_NPU_REPORT=1: each image's text-details compile report beside it ("<name>.report": blocks, bundles)
        rep = (["--compile-report=text-details", "--compile-report-output=" + os.path.join(outdir, name + ".report")]
               if os.environ.get("YAH_NPU_REPORT") == "1" else [])
        r = subprocess.run([hrx_paths.LOOM_COMPILE, src, "--root=@" + cfg.entry,
                            "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11",
                            "--output=" + os.path.join(outdir, name + ".xdna")] + rep, capture_output=True, text=True,
                           env=env)
        if os.environ.get("LOOM_DIAG_PC") and not r.returncode:   # the image's PC ranges for YAH_NPU_PROF_PC
            open(os.path.join(outdir, name + ".pc"), "w").write(
                "\n".join(npu_prof.pc_ranges(src, r.stderr.splitlines())) + "\n")
            # the raw block table too (npu_prof.pc_ranges input), for ranges between any two labels
            open(os.path.join(outdir, name + ".pcdiag"), "w").write(
                "".join(l + "\n" for l in r.stderr.splitlines() if l.split()[:1] in (["PCBLOCKS"], ["PCTEXT"])))
        return r
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        rs = list(ex.map(compile_one, [source(*j) for j in jobs]))
        # ghelp's helper (a decoder plus the gate's waiter) can exceed the core's program memory: that image with
        # Q3_K's record realign rolled (q3roll), then no seam on the helpers, then the helpers' scale halves rolled
        # (hvroll). Not without ghelp: those images hang the NPU (docs/results.md)
        for fix, msg in ((dict(q3roll=1), "Q3_K realign rolled"), (dict(seam=2), "helper without seam"),
                         (dict(hvroll=1), "hvroll"),
                         (dict(hvroll=1, htab=0), "hvroll, helper table per unit")):
            redo = [i for i, r in enumerate(rs) if r.returncode and jobs[i][0].ghelp and "XDNA/058" in r.stderr]
            for i in redo:
                jobs[i] = (dataclasses.replace(jobs[i][0], **fix),) + tuple(jobs[i][1:])
                print("%s: %s (program memory)" % (jobs[i][1], msg), file=sys.stderr)
            for i, r in zip(redo, ex.map(compile_one, [source(*jobs[i]) for i in redo])):
                rs[i] = r
    for (cfg, name, _), r in zip(jobs, rs):
        if r.returncode:
            raise SystemExit("NPU image compile failed (%s): %s" % (name, r.stderr[-800:]))
    j0 = next(j for j in jobs if j[2] == NPU_NP)
    out = [("npudcol", NPU_COLS, 0, 0), ("npubytes_5120",) + tuple(GN.stream_bytes(j0[0])),
           ("npurows", NPU_ROWS, 0, 0)]
    j4 = [j for j in jobs if NPU_Q and j[2] == 4 and j[0].fmt != "IQ4_XS" and not j[0].qsw]
    if j4 and NPU_QP1K:   # (the NP 4 formats' site images, np4)
        with GN.np_override(4):
            out += [("npubytes4_5120",) + tuple(GN.stream_bytes(j4[0][0])), ("npurows4", NPU_COLS * 64, 0, 0)]
        out += [("npunp4_" + f, 1, 0, 0) for f in sorted({f for f, _ in images if np4(f)})]
    # (Q': the NPU queues one call at most; at depth 2 a core's next fill overwrote its undrained C (its rings overlay C
    # slots 1-3). P1K images with GN_QFGUARD wait for the drain in the fill and run at depth 2 (1 of 47 engine runs hung
    # on 11-18, NPU firmware timeout, cause open; 11-19 q50 at depth 2: NPU jobs -3%, bit-exact). GN_QDEPTH2=0: depth 1.)
    guard = NPU_QP1K and os.environ.get("GN_QFGUARD") == "1" and os.environ.get("GN_QDEPTH2", "1") == "1"
    return out + [("npugfirst", 1, 0, 0)] + ([("npudepth1", 1, 0, 0)] if NPU_Q and not guard else [])


def bfp_units(w, kperm):
    """f32 weights [n (16 | n)][K / 256][256] -> the BFP decoder's rows (bytes): per 16-row slab, its rows of K / 1024
    units of 1152 bytes (unit (slab, pass): [kb 128 in kperm order][h 2] bfp16 fragments)."""
    import npu_dec_emit as Q
    n, K = w.shape[0], w.shape[1] * 256
    w = w.reshape(n // 16, 2, 8, K // 1024, 128, 8).transpose(0, 3, 4, 1, 2, 5)[:, :, kperm]
    units = Q.bfp_hw(w.reshape(-1, 8, 8)).reshape(n // 16, K // 1024, 16, 1152)
    return units.transpose(0, 2, 1, 3).tobytes()


def bfp_site_rows(sites, outdir):
    """The site tensors (name, NPU rows) of formats without an NPU decoder, their last rows (the NPU's)
    dequantized exactly (gguf quants) as BFP rows (bfp_units) to npu_bfp_<name>.bin."""
    import gguf
    import numpy as np
    from gguf import quants
    kperm = GN.decoder("BFP").KPERM
    ts = {t.name: t for t in gguf.GGUFReader(MODEL).tensors}
    for name, nn in sorted(sites):
        t = ts[name]
        K, N = int(t.shape[0]), int(t.shape[1])
        path = os.path.join(outdir, "npu_bfp_%s.bin" % name)
        if os.path.exists(path) and os.path.getsize(path) == nn * K // 256 * 288:
            continue
        raw = np.asarray(t.data).view(np.uint8).reshape(N, -1)
        with open(path + ".tmp", "wb") as f:
            for r0 in range(N - nn, N, 512):
                w = quants.dequantize(raw[r0:r0 + 512], t.tensor_type).astype(np.float32)
                f.write(bfp_units(w.reshape(-1, K // 256, 256), kperm))
        os.replace(path + ".tmp", path)
        print("bfp site rows: %s (%d rows) -> %s" % (name, nn, path))


def bfp_pair_rows(layers, F, outdir):
    """The gate / up tensors of the pairs in layers (prefixes), their last F rows (the NPU's) decoded to
    their decoders' exact f32 weights (gen_npu_dec.weights) as BFP rows (bfp_units), gate's then
    up's, to npu_bfp_<prefix>ffn_gate_up.bin (the engine's one BFP image binds them like a single format's)."""
    import gguf
    import numpy as np
    kperm = GN.decoder("BFP").KPERM
    rd = gguf.GGUFReader(MODEL)
    ts = {t.name: t for t in rd.tensors}
    for pre in sorted(layers):
        path = os.path.join(outdir, "npu_bfp_%sffn_gate_up.bin" % pre)
        K = int(ts[pre + "ffn_gate.weight"].shape[0])
        if os.path.exists(path) and os.path.getsize(path) == 2 * F * K // 256 * 288:
            continue
        with open(path + ".tmp", "wb") as f:
            for k in ("ffn_gate", "ffn_up"):
                t = ts[pre + k + ".weight"]
                src, N = t.tensor_type.name, int(t.shape[1])
                raw = np.asarray(t.data).view(np.uint8).reshape(N, K // 256, -1)
                for r0 in range(N - F, N, 512):
                    blk = raw[r0:r0 + 512]
                    f.write(bfp_units(GN.decoder(src).weights(blk), kperm))
        os.replace(path + ".tmp", path)
        print("bfp pair rows: %s -> %s" % (pre, path))


def npu_ffnblk(rows, B, tmp, outdir, out, split, variants):
    """NPU_FFNBLK: the GPU's kernels of the FFN block (appended to out, split / variants: npu_split's) and its NPU images
    (returned as npu_images extra jobs); dispatch.txt "npuffnblk <F>", "npubytes_ffnsw" / "npubytes_ffndn<passes>"."""
    import gen_bfp16_encode as GE
    import gen_gemm_tile as TG
    import gen_npu_unpack as GU
    F, FS = NPU_FFNBLK, NPU_FFNBLKS   # (pairs, single-format layers)
    assert F % 1024 == 0 and (FS % 1024 == 0 or FS % 1280 == 0) and B % AF_TILE["bn"] == 0 and (FS == F or NPU_Q)
    by = {}
    for nm, dims, ty in rows:
        p = nm.split(".")
        if len(p) > 2 and p[0] == "blk" and p[2] in ("ffn_gate", "ffn_up", "ffn_down") and E.FMT.get(ty):
            by.setdefault(p[1], {})[p[2]] = (E.FMT[ty][0], dims[1] // 16, dims[0] // E.FMT[ty][2])
    fmts, dfmts = FUSE_FFN_FMTS, FUSE_FFN_DOWN_FMTS
    # H's k-block order: the down decoder's (P4 but Q4_K's; Q': P4, Q4_K's down decodes in P4 order, gord "GF")
    hord_of = lambda d: "Q4K" if d == "q4k" and not NPU_Q else "P4"
    # (Q', YAH_NPU_FFNQ4K=0: not the layers with Q4_K downs; at FFNBLK 5120 the NPU is the critical path, p6d
    # pp2048 2286.9 ms vs 2158.0 without them)
    q4kdn = os.environ.get("YAH_NPU_FFNQ4K", "1") == "1"
    rows = NPU_ROWS if NPU_QP1K else NPU_COLS * 64   # down rows per call (NP 4; Q': the P1K site image's)
    qcalls = 4 * ffnblk_parts(F)[0][1]   # (Q': a K part's qsw calls, its down's passes)
    qcallss = ffnblk_apass(FS) or ffnblk_units(FS)[0][2]   # (H's units a 64-token M block in a part, or in all)
    # (FFNSW5: whole 1280-feature groups)
    sw5 = FFNSW5 and NPU_Q and FS % 1280 == 0 and all(r % 5 == 0 for _, r, _ in ffnblk_units(FS))
    sw5 = sw5 and all(gen_gemv.rseg_ok(l["ffn_gate"][1] * 16, FS) for l in by.values() if l.get("ffn_gate"))
    for f_ in {F, FS}:
        assert not NPU_Q or (NPU_QP1K and ((f_ % 1024 == 0 and len({u[2] for u in ffnblk_units(f_)}) == 1)
                                           or (f_ == FS and sw5 and NPU_DNSEG))), \
            "YAH_NPU_Q: the FFN block needs YAH_NPU_QP1K and K parts of one size (F = 5120, 8192, 10240, ..; " \
            "FFNSW5 with DNSEG: any F of whole 1280-feature groups)"
    assert len(ffnblk_units(F)) == len(ffnblk_units(FS)), "the unpack sums one count of K parts"
    ok = [l for l in by.values() if l.get("ffn_gate") == l.get("ffn_up") and l["ffn_gate"][0] in fmts
          and l["ffn_gate"][0] in AF_FFN and l.get("ffn_down", ("",))[0] in dfmts]
    # gate and up of two formats, or one without a fused GPU gate / up (its gate kstore and up swiglu): pairs
    pairs = [l for l in by.values() if l.get("ffn_gate") and l.get("ffn_up")
             and (l["ffn_gate"][0] != l["ffn_up"][0] or l["ffn_gate"][0] not in AF_FFN)
             and l["ffn_gate"][1:] == l["ffn_up"][1:] and l["ffn_gate"][0] in fmts and l["ffn_up"][0] in fmts
             and l.get("ffn_down", ("",))[0] in dfmts]
    if NPU_Q:   # (Q': IQ4_XS gate and up; down IQ4_XS or an NP 4 format; the other layers keep the FFN on the GPU)
        # (YAH_NPU_FFNDN4=0: IQ4_XS downs only)
        dn4 = os.environ.get("YAH_NPU_FFNDN4", "1") != "0"
        dok = lambda l: l["ffn_down"][0] == "iq4xs" or (dn4 and np4(l["ffn_down"][0]) and (q4kdn or l["ffn_down"][0] != "q4k"))
        ok = [l for l in ok if l["ffn_gate"][0] == "iq4xs" and dok(l)]
        # pairs: gate and up of one QSW_FMTS format without a fused GPU gate / up (the GPU's side a pair's, the NPU's
        # one qsw image of that format); QCP: column pairs (colpair), one format's too (taken from ok's candidates)
        if QCP:
            pairs += [l for l in by.values() if l.get("ffn_gate") and l.get("ffn_gate") == l.get("ffn_up")
                      and l not in pairs and colpair(l) and l.get("ffn_down", ("",))[0] in dfmts]
        pairs = [l for l in pairs if (l["ffn_gate"][0] == l["ffn_up"][0] in QSW_FMTS or (QCP and colpair(l))) and dok(l)]
    # pairs of one format of NPU_SAME_SW: one image like a single format's, with the deferred epilogue
    # ("npu_ffnsw_<gate>_<up>"); column pairs; the other pairs take the BFP image (bsw)
    mixed = [l for l in pairs if l["ffn_gate"][0] == l["ffn_up"][0] in NPU_SAME_SW or colpair(l)]
    bsw = [l for l in pairs if l not in mixed]
    bfp_pair_rows({"blk.%s." % n for n, l in by.items() if l in bsw}, F, outdir)
    jobs = []
    for fmt, mt, kb in sorted({l["ffn_gate"] for l in ok}):
        hords = sorted({hord_of(l["ffn_down"][0]) for l in ok if l["ffn_gate"] == (fmt, mt, kb)})
        for hord in hords:
            if hord == hords[0]:
                t = dataclasses.replace(TG.default_tile(fmt, "swiglu", kb), **AF_TILE, **AF_FFN[fmt], ffn=True)
                for sfx, tt in ((".af.hal", t), (".af.to.hal", dataclasses.replace(t, tout=True))):
                    split(fmt, "ffn", mt, kb, tt, "gemm_ffn_%s_%d_%d%s" % (fmt, mt, kb, sfx), FS)
                    if RSWAP_EMIT and fmt == "iq4xs" and sfx == ".af.to.hal" and FS == gen_r5fuse.F and kb == 20:
                        # (row 5 swap) the same split decoding another layer's NPU gate / up rows into the swap's
                        # stream ("<hal>.af.to.r5.npu.hal", gen_r5fuse)
                        split(fmt, "ffn", mt, kb, tt, "gemm_ffn_%s_%d_%d.af.to.r5.hal" % (fmt, mt, kb), FS,
                              wrap=gen_r5fuse.fuse)
                    if sw5 and fmt == "iq4xs":   # (the row-segmented layers' GPU rows: their up rows F rows on)
                        try:
                            TG.FFUP = FS
                            split(fmt, "ffn", mt, kb, tt, "gemm_ffn_%s_%d_%d%s%s" % (fmt, mt, kb, gen_gemv.rseg_suffix(FS), sfx), FS)
                        finally:
                            TG.FFUP = 0
            blk = GN.decoder(gu_fmt(fmt)).BLK
            dfr = fmt in NPU_DFR_FMTS
            if NPU_Q:   # (Q' qsw at NP 4, or FFNSW5's NP 5; its egress ring stays 4 records, GN_QFEGR2 would reach
                # the silu table)
                np5 = sw5 and fmt == "iq4xs"
                # (np5: gate block g's up block 1280 rows on, RSEG; else up a gate tensor on)
                dgu_ = gen_gemv.RSEG_B * kb * blk if np5 else mt * 16 * kb * blk
                jobs.append((GN.Config(NPU_COLS, B // 64, NPU_KS, NPU_PASSES, mu=1, gate=GN.GATE_SUPPLY, fuse=2,
                                       kraw=kb * 256, fmt=DCOL_FMTS[fmt][0], rawdma=1, qsw=True, dgu=dgu_,
                                       swcol=40 if np5 else 32, hpass=qcallss, hord=hord),
                             "npu_ffnsw_%s_%s" % (fmt, hord.lower()), 5 if np5 else 4))
                QJOB_ENV[jobs[-1][1]] = dict(GN_QFEGR2="0")
                if RSWAP_EMIT and fmt == "iq4xs" and not np5 and hord == "P4":
                    assert B == 2048 and FS % 1024 == 0 and all(u[2] % 4 == 0 for u in ffnblk_units(FS))
                    assert RSWAP_NB % 32 == 0 and RSWAP_NB <= 252 and FS // 32 % RSWAP_NB == 0, RSWAP_NB
                    jobs.append((GN.Config(NPU_COLS, RSWAP_NB, RSWAP_KS, RSWAP_PASSES, mu=1, gate=GN.GATE_SUPPLY, fuse=0,
                                           rawdma=0, gord="P4", qsw=True, dgu=0, swcol=32, hpass=qcallss, hord="P4"),
                                 "npu_ffnswap_iq4xs", 4))
                    QJOB_ENV["npu_ffnswap_iq4xs"] = dict(RSWAP_ENV)
                continue
            jobs.append((npu_cfg(B, NPU_DFR_KS if dfr else NPU_KS, NPU_PASSES, swiglu=True, dgu=mt * 16 * kb * blk,
                                 swcol=32, hpass=FS // 1024, hord=hord, fmt=gu_fmt(fmt), dfr=int(dfr)),
                         "npu_ffnsw_%s_%s" % (fmt, hord.lower()), 4))
    parts = ffnblk_parts(F)
    lf = lambda l: FS if l in ok else F   # (a layer's F)
    if FS != F:   # (the single-format layers' downs in FS's K segments: their GEMMs, main; their layers, npudnsegl)
        FFNBLKS_DOWNS.update(l["ffn_down"] for l in ok)
        out.extend(("npudnsegl_" + n, FS, 0, 0) for n, l in sorted(by.items()) if l in ok)
    # the GPU's K window [0, 17408 - F)
    for fmt, mt, kb, kbw in sorted({l["ffn_down"] + ((l["ffn_down"][2] * 256 - lf(l)) // 256,) for l in ok + pairs}):
        tr = fit_rowgrp(npu_rem_tile(fmt, kbw, kb), mt)

        def gen_rem(f, k, tr=tr, kb=kb):
            TG.KWIN = (0, kb)
            try:
                return TG.gen(f, "kstore", tr, False)
            finally:
                TG.KWIN = None
        assert mt not in O16_MT, "the window writes f32"
        with kseg_f((kb - kbw) * 256):   # (the window's layer group's K segments)
            out.append(_emit_gen(gen_rem, tr.bn, fmt, mt, kbw, B, "npuffnrem_%s_%d.hal" % (fmt, kbw), outdir, "kstore",
                                 tr.rowgrp, kfull=kb))

        def gen_rem_rr(f, k, tr=tr, kb=kb):   # (the same window with the residual epilogue: hidden + it, "_rr")
            TG.KWIN = (0, kb)
            try:
                return TG.gen(f, "kres", tr, False)
            finally:
                TG.KWIN = None
        with kseg_f((kb - kbw) * 256):
            out.append(_emit_gen(gen_rem_rr, tr.bn, fmt, mt, kbw, B, "npuffnrem_%s_%d_rr.hal" % (fmt, kbw), outdir,
                                 "kres", tr.rowgrp, kfull=kb))
    for fmt, mt, kb, r, p, ap in sorted({l["ffn_down"] + (u[1], u[2], ffnblk_apass(lf(l))) for l in ok + pairs
                                         for u in ffnblk_units(lf(l))}):
        if True:
            if NPU_Q:   # (Q': the site image of down's raw K (a part's), A bound to H; the NP 4 formats' at NP 4)
                # (a padded part: its segment's rows r blocks apart, read p blocks wide; ap: A an H window)
                assert r == p or (fmt, mt, kb) in DNSEG, (fmt, r, p)
                jobs.append((GN.Config(NPU_COLS, B // 64, NPU_KS, p, mu=1, gate=GN.GATE_SUPPLY, fuse=2,
                                       kraw=r * 256 if r != p else 0 if (fmt, mt, kb) in DNSEG else kb * 256,
                                       fmt=DCOL_FMTS[fmt][0], rawdma=1, apass=ap,
                                       gord="GF" if fmt == "q4k" else "P4"),   # (H in P4 order)
                             "npu_ffndn%s_%s" % (dn_tag(r, p, ap), fmt), 4 if np4(fmt) else NPU_NP))
                continue
            assert r == p and not ap
            jobs.append((npu_cfg(B, NPU_KS, p // 4, kraw=kb * 256, ain="h", hpass=F // 1024, fmt=DCOL_FMTS[fmt][0],
                                 **KCHAIN(fmt)), "npu_ffndn%d_%s" % (p // 4, fmt), 4))
    for (gf, mt, kb), (uf, _, _), hord in sorted({(l["ffn_gate"], l["ffn_up"], hord_of(l["ffn_down"][0]))
                                                  for l in pairs}):
        # the GPU's other features: gate kstore and up swiglu, split (".npu.hal", nn = F)
        variants(gf, "kstore", mt, kb, "gemm_kstore_%s_%d_%d.hal" % (gf, mt, kb), F, "ffn")
        variants(uf, "swiglu", mt, kb, "gemm_swiglu_%s_%d_%d.hal" % (uf, mt, kb), F, "ffn")
        # (GN_FFNMIXSPLIT=0: off) the fused GPU gate / up's split for the pair, AF_FFN_MIX or one AF_FFN format
        # ("gemm_ffn_<gate>[_<up>]_<mt>_<kb>.af.to.npu.hal", ffn_fused's knobs): one GEMM instead of kstore + swiglu
        if os.environ.get("GN_FFNMIXSPLIT", "1") != "0" and B % AF_TILE["bn"] == 0:
            import gen_gemm_tile as TG
            mx = (gf, uf) in AF_FFN_MIX
            if mx or (gf == uf and gf in AF_FFN):
                t = dataclasses.replace(TG.default_tile(gf, "swiglu", kb), **AF_TILE,
                                        **(AF_FFN_MIX[(gf, uf)] if mx else AF_FFN[gf]), ffn=True, tout=True)
                split(gf if gf == uf else gf + ":" + uf, "ffn", mt, kb, t,
                      "gemm_ffn_%s_%d_%d.af.to.hal" % (gf + "_" + uf if mx else gf, mt, kb), F)
        if any(l in bsw for l in pairs if (l["ffn_gate"], l["ffn_up"]) == ((gf, mt, kb), (uf, mt, kb))):   # (BFP)
            if "npu_ffnsw_bfp_" + hord.lower() not in {j[1] for j in jobs}:
                jobs.append((npu_cfg(B, NPU_DFR_KS, NPU_PASSES, swiglu=True, dgu=mt * 16 * kb * 288, swcol=32,
                                     hpass=F // 1024, hord=hord, fmt="BFP", dfr=1), "npu_ffnsw_bfp_" + hord.lower(), 4))
            continue
        go = lambda f: "GF" if f == "q4k" else "P4"
        if NPU_Q and colpair({"ffn_gate": (gf,), "ffn_up": (uf,)}):   # (Q' column pairs: H as the qsw image's)
            cfg = GN.Config(NPU_COLS, B // 64, NPU_KS, NPU_PASSES, mu=1, gate=GN.GATE_SUPPLY, fuse=2, kraw=kb * 256,
                            fmt=gu_fmt(gf), gord=go(gf), rawdma=1, qsw=True, dgu=0, swcol=32, hpass=qcalls, hord=hord,
                            ufmt=gu_fmt(uf), ugord=go(uf))
            QJOB_ENV["npu_ffnsw_%s_%s_%s" % (gf, uf, hord.lower())] = dict(GN_QFEGR2="0")
        elif colpair({"ffn_gate": (gf,), "ffn_up": (uf,)}):   # (column pairs; Q4_K in P4 order: gen_npu_dec "GF")
            cfg = npu_cfg(B, NPU_DFR_KS, NPU_PASSES, swiglu=True, swcol=32, hpass=F // 1024, hord=hord, fmt=gu_fmt(gf),
                          gord=go(gf), ufmt=gu_fmt(uf), ugord=go(uf), dfr=1)
        elif NPU_Q:   # (Q': one format, the qsw image as above)
            blk = GN.decoder(gu_fmt(gf)).BLK
            cfg = GN.Config(NPU_COLS, B // 64, NPU_KS, NPU_PASSES, mu=1, gate=GN.GATE_SUPPLY, fuse=2, kraw=kb * 256,
                            fmt=DCOL_FMTS[gf][0], rawdma=1, qsw=True, dgu=mt * 16 * kb * blk, swcol=32, hpass=qcalls,
                            hord=hord)
            QJOB_ENV["npu_ffnsw_%s_%s_%s" % (gf, uf, hord.lower())] = dict(GN_QFEGR2="0")
        else:   # (mixed)
            blk = GN.decoder(gu_fmt(gf)).BLK
            cfg = npu_cfg(B, NPU_DFR_KS, NPU_PASSES, swiglu=True, dgu=mt * 16 * kb * blk, swcol=32, hpass=F // 1024,
                          hord=hord, fmt=gu_fmt(gf), dfr=1)
        jobs.append((cfg, "npu_ffnsw_%s_%s_%s" % (gf, uf, hord.lower()), 4))
    # hidden + the GPU's window + the NPU's down partials (per K part, calls of rows rows, one replay group each)
    N, calls = 5120, -(-5120 // rows)
    name = "npu_unpack_ffnblk"
    cg = GU.CG
    GU.CG = 1
    try:
        with GN.np_override(NPU_NP if NPU_Q else 4):
            text = GU.gen(B, calls * NPU_COLS, N, 0, resid=True, parts=len(parts), rem=True,
                          dup=((calls - 1) * NPU_COLS, N - rows))
            text_rr = GU.gen(B, calls * NPU_COLS, N, 0, resid=True, parts=len(parts), rem=True,
                             dup=((calls - 1) * NPU_COLS, N - rows), remres=True)
            nj, wg = GN.TN // 8, GU.wg(calls * NPU_COLS)
            # the same sum fused into the next layer's norm when it feeds a K = 5120 NPU site ("npu_unorm[_pk].hal",
            # gen_half_norm.gen_unorm: the norm_bfp stream, with _pk also the Q4_K-order one); needs the last down call's
            # overlap to keep whole NPU columns (rows a multiple of 5120 / calls)
            import gen_half_norm
            unorm = {}
            # (not yet for TN 80, the Q' P1K down: gen_unorm reads C as TN 64; the sum then runs as its own unpack)
            if {"qkv", "q"} & set(NPU_SPLIT) and (N - rows) == (calls - 1) * NPU_COLS * GN.TN and GN.TN == 64:
                for pk in (False, True):
                    unorm["npu_unorm%s.hal" % ("_pk" if pk else "")] = gen_half_norm.gen_unorm(
                        B, list(NPU_KS), NPU_PASSES, len(parts), pk)
            # after the "_rr" sum: the sum, the stream and the f16 output(s) in one pass ("npu_unorm_rr_<r|t|rt>[_pk]",
            # TN 64 or 80: gen_unorm indexes each column in full at 80)
            if {"qkv", "q"} & set(NPU_SPLIT) and (N - rows) == (calls - 1) * NPU_COLS * GN.TN and GN.TN in (64, 80):
                for pk in (False, True):
                    for f16 in ("r", "t", "rt"):
                        unorm["npu_unorm_rr_%s%s.hal" % (f16, "_pk" if pk else "")] = gen_half_norm.gen_unorm(
                            B, list(NPU_KS), NPU_PASSES, len(parts), pk, remres=True, f16=f16)
    finally:
        GU.CG = cg
    src = os.path.join(tmp, name + ".loom")
    open(src, "w").write(text)
    E.emit(src, ["nop=0"], name + ".hal", outdir)
    out.append((name + ".hal", B * calls * NPU_COLS * nj // wg, wg, 0))
    # (with npuffnrem_*_rr: the sum after the GPU's down added hidden in its epilogue, "_rr")
    src = os.path.join(tmp, name + "_rr.loom")
    open(src, "w").write(text_rr)
    E.emit(src, ["nop=0"], name + "_rr.hal", outdir)
    out.append((name + "_rr.hal", B * calls * NPU_COLS * nj // wg, wg, 0))
    for f_ in sorted({F, FS}):   # (padded down parts) zeros into H's pad units, before the gate / up job each layer
        pads = ffnblk_pads(f_, B)
        if pads:
            us = ffnblk_units(f_)
            zn = "npu_zpad_%d" % f_
            src = os.path.join(tmp, zn + ".loom")
            open(src, "w").write(GU.gen_zpad(B // 64, pads, B // 64 * sum(u[2] for u in us) * 18432))
            E.emit(src, ["nop=0"], zn + ".hal", outdir)
            out.append((zn + ".hal", B // 64 * len(pads), GU.ZPAD_WG, 0))
    dn4 = sorted({j[1].split("_")[-1] for j in jobs if j[1].startswith("npu_ffndn") and j[2] == 4 != NPU_NP})
    if dn4:   # (Q') the NP 4 downs' sum: 512-row calls at TN 64 ("npu_unpack_ffnblk_4.hal", "npuffnblk4")
        rows4, cg = NPU_COLS * 64, GU.CG
        calls4 = -(-N // rows4)
        GU.CG = 1
        try:
            with GN.np_override(4):
                text = GU.gen(B, calls4 * NPU_COLS, N, 0, resid=True, parts=len(parts), rem=True,
                              dup=((calls4 - 1) * NPU_COLS, N - rows4))
                text4_rr = GU.gen(B, calls4 * NPU_COLS, N, 0, resid=True, parts=len(parts), rem=True,
                                  dup=((calls4 - 1) * NPU_COLS, N - rows4), remres=True)
                if {"qkv", "q"} & set(NPU_SPLIT) and calls4 * rows4 == N:   # (as above, the NP 4 downs at TN 64)
                    for pk in (False, True):
                        for f16 in ("r", "t", "rt"):
                            unorm["npu_unorm4_rr_%s%s.hal" % (f16, "_pk" if pk else "")] = gen_half_norm.gen_unorm(
                                B, list(NPU_KS), NPU_PASSES, len(parts), pk, remres=True, f16=f16)
                nj4, wg4 = GN.TN // 8, GU.wg(calls4 * NPU_COLS)
        finally:
            GU.CG = cg
        src = os.path.join(tmp, name + "_4.loom")
        open(src, "w").write(text)
        E.emit(src, ["nop=0"], name + "_4.hal", outdir)
        out.append((name + "_4.hal", B * calls4 * NPU_COLS * nj4 // wg4, wg4, 0))
        src = os.path.join(tmp, name + "_4_rr.loom")
        open(src, "w").write(text4_rr)
        E.emit(src, ["nop=0"], name + "_4_rr.hal", outdir)
        out.append((name + "_4_rr.hal", B * calls4 * NPU_COLS * nj4 // wg4, wg4, 0))
        out.append(("npuffnblk4", F, 256, rows4))
        for fmt in dn4:
            out.append(("npuffndn4_" + fmt, 1, 0, 0))
    for hal, text in unorm.items():
        src = os.path.join(tmp, hal[:-4] + ".loom")
        open(src, "w").write(text)
        E.emit(src, ["yah_half_norm.rows=%d" % B, "yah_half_norm.dim=5120", "yah_half_norm.eps=1e-06",
                     "yah_half_norm.fused=0"], hal, outdir)
        out.append((hal, B // 4, 32 * 4 * 4, 0))   # gen_unorm: 4 rows x split 4
    # gate / up read their activations in one replay group: the norm before them writes that stream (fragment-major)
    import gen_half_norm
    groups, perm = GE.GROUPS, GE.ACT_PERM
    dfrs = sorted({j[0].fmt for j in jobs if j[0].dfr})
    for name, ks in (("norm_t_bfp_g1", NPU_KS),) + ((("norm_t_bfp_g1s", NPU_DFR_KS),) if dfrs else ()):
        GE.GROUPS, GE.ACT_PERM = (NPU_PASSES,), "p4"
        try:
            src = os.path.join(tmp, name + ".loom")
            open(src, "w").write(gen_half_norm.gen_split(5120, wpr=4, split=NORM_SPLIT, tiled=True,
                                                         bfp=(B, list(ks), NPU_PASSES)))
        finally:
            GE.GROUPS, GE.ACT_PERM = groups, perm
        E.emit(src, ["yah_half_norm.rows=%d" % B, "yah_half_norm.dim=5120",
                     "yah_half_norm.eps=1e-06", "yah_half_norm.fused=0"], name + ".hal", outdir)
    for fmt in NPU_DFR_FMTS:
        if gu_fmt(fmt) in dfrs:
            out.append(("npuffndfr_" + fmt, 1, 0, 0))
    for gf, uf in sorted({(l["ffn_gate"][0], l["ffn_up"][0]) for l in pairs} if not NPU_Q else ()):   # (Q': none)
        out.append(("npuffndfr_%s_%s" % (gf, uf), 1, 0, 0))
    for gf, uf in sorted({(l["ffn_gate"][0], l["ffn_up"][0]) for l in pairs if colpair(l)}):
        out.append(("npuffncp_%s_%s" % (gf, uf), 1, 0, 0))
    for gf, uf in sorted({(l["ffn_gate"][0], l["ffn_up"][0]) for l in bsw}):
        out.append(("npuffnbx_%s_%s" % (gf, uf), 1, 0, 0))
    out.append(("npuffnblk", FS, 256, rows))   # F, gate / up features per call, down rows
    swj = next((j for j in jobs if j[1] == "npu_ffnswap_iq4xs"), None)
    if swj:   # (YAH_NPU_RSWAP_EMIT) the swap's GPU side: the NPU rows' decoders, the token panel
        import gen_iq4xs_bfp
        st_ = dict(ks=RSWAP_KS, passes=RSWAP_PASSES, tile=64)
        for up_, nm_ in ((0, "npu_swdec_iq4xs_g"), (1, "npu_swdec_iq4xs_u")):
            src = os.path.join(tmp, nm_ + ".loom")
            open(src, "w").write(gen_iq4xs_bfp.gen(FS, 5120, dict(st_, up=up_)))
            E.emit(src, ["nop=0"], nm_ + ".hal", outdir)
            out.append((nm_ + ".hal", gen_iq4xs_bfp.items(FS, 5120), gen_iq4xs_bfp.WG, gen_iq4xs_bfp.out_bytes(FS, 5120, st_)))
        import gen_kq_bfp
        for fm_ in sorted({l[t_][0] for l in pairs for t_ in ("ffn_gate", "ffn_up")} & RSWAP_KQ.keys()):
            for up_, g_ in ((0, "g"), (1, "u")):
                nm_ = "npu_swdec_%s_%s" % (fm_, g_)
                src = os.path.join(tmp, nm_ + ".loom")
                open(src, "w").write(gen_kq_bfp.gen(RSWAP_KQ[fm_], FS, 5120, dict(st_, up=up_)))
                E.emit(src, ["nop=0"], nm_ + ".hal", outdir)
                out.append((nm_ + ".hal", gen_kq_bfp.items(FS, 5120), gen_kq_bfp.WG, gen_kq_bfp.out_bytes(FS, 5120, st_)))
        src = os.path.join(tmp, "npu_swpanel.loom")
        open(src, "w").write(GE.gen("wgt", B, RSWAP_KS, RSWAP_PASSES, 64, True, True, p1k=True))
        E.emit(src, ["nop=0"], "npu_swpanel.hal", outdir)
        out.append(("npu_swpanel.hal", B // 8 * 640, GE.WG, GE.layout_bytes("wgt", B, RSWAP_KS, RSWAP_PASSES)))
        out.append(("npurswap", FS, 0, 0))
        with GN.np_override(4):
            out.append(("npubytes_ffnswap",) + tuple(GN.stream_bytes(swj[0])))
    if NPU_Q:
        out.append(("npuffnq", 1, 0, 0))
        for fmt, mt, kb in sorted(DNSEG):   # (the engine reorders these formats' down rows: LoomWeights)
            out.append(("npudnseg_%s" % fmt, F, mt, kb))
    for np_, nm_ in ((4, "npubytes_ffnsw"), (5, "npubytes5_ffnsw")):   # (FFNSW5: the NP 5 image's, "npuffnsw5")
        sw = next((j for j in jobs if j[1].startswith("npu_ffnsw_") and j[2] == np_), None)
        if sw:
            with GN.np_override(np_):
                out.append((nm_,) + tuple(GN.stream_bytes(sw[0])))
    if sw5 and ok:   # (and their gate / up rows in row segments: LoomWeights, the RSEG GEMMs)
        out.append(("npuffnsw5", 320, 0, 0))
        out.extend(("npursegl_" + n, FS, 0, 0) for n, l in sorted(by.items()) if l in ok)
        RSEG_GEMMS.update(l["ffn_gate"] for l in ok)
    for tag in sorted({dn_tag(r, p, ffnblk_apass(f_)) for f_ in {F, FS} for _, r, p in ffnblk_units(f_)}):
        dns = [j for j in jobs if j[1].startswith("npu_ffndn%s_" % tag)]
        dn = next((j for j in dns if j[2] == NPU_NP), dns[0])
        with GN.np_override(dn[2]):
            out.append(("npubytes_ffndn%s" % tag,) + tuple(GN.stream_bytes(dn[0])))
        if dn4:   # (Q') the NP 4 downs'
            dn = next(j for j in dns if j[2] == 4)
            with GN.np_override(4):
                out.append(("npubytes4_ffndn%s" % tag,) + tuple(GN.stream_bytes(dn[0])))
    if pairs:   # (the mixed / BFP pairs': their GPU side's features)
        out.append(("npuffnpair", F, 0, 0))
    return jobs


def tile_kstore(fmt, mt, kb, B, out, outdir, kind, geom=None):
    """Emit the tile GEMM (tools/gen_gemm_tile.py) for this shape if it covers it; return its dispatch.txt row, else None.
    geom=(BM, BN, WM, WN) overrides the workgroup geometry for this one kernel."""
    import gen_gemm_tile as TG
    if fmt not in TILE_FMTS:
        return None
    t = TG.default_tile(fmt, kind, kb, geom)
    if not geom and tkey(out) in tiles()["tiles"]:
        t = dataclasses.replace(t, **tiles()["tiles"][tkey(out)])
    if mt % t.rowgrp:
        return None
    # a token tile that does not divide the chunk: the last tile is masked (gen_gemm_tile.gen masked=)
    masked = B % t.bn != 0
    return _emit_gen(lambda f, k: TG.gen(f, k, t, masked), t.bn, fmt, mt, kb, B, out, outdir, kind, t.rowgrp, masked)


def _emit_gen(gen, tile, fmt, mt, kb, B, out, outdir, kind, rowgrp, masked=False, ostride=0, kfull=0, extra="",
              vtry=None):
    """ostride: a GEMM over the leading mt * 16 rows of an ostride-row output (gen_gemm_tile.OSTRIDE, the NPU split's GPU share);
    f16 output then follows the full shape (O16_MT).
    kfull: a K window's whole-row k_blocks (gen_gemm_tile.KWIN).
    extra: gen adds buffers past the GEMM's (its own source file; footprint_gate does not model them: the caller
    states and checks their bounds, e.g. gen_r5fuse.lane_records). vtry: emit_prefill.emit's."""
    if B % tile and not masked:
        return None
    import gen_gemm_tile as _TG
    _TG.OUT16 = kind == "kstore" and (ostride // 16 if ostride else mt) in O16_MT
    _TG.OSTRIDE = ostride
    tt = -(-B // tile)
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(tmp, "yah_sgemm_%s_%s%s.loom" % (fmt, kind, extra))
    try:
        with open(src, "w") as fh, kseg_for(fmt, mt, kfull or kb):
            fh.write(gen(fmt, kind))
    finally:
        _TG.OSTRIDE = 0
    sym = "yah_ffn_gemm_%s%s" % (fmt.replace(":", "_"), {"swiglu": "_swiglu", "kres": "_kres", "kqg": "_kqg", "ffn": "_ffn"}.get(kind, ""))
    # Refuse before emitting if any declared operand footprint exceeds the buffer the driver binds: an overrun hangs the ring.
    gate = [sym, fmt, kind, str(mt), str(kb), str(tt), str(B)] + (["masked"] if masked else []) \
        + (["ostride=%d" % ostride] if ostride else []) + (["kfull=%d" % kfull] if kfull else []) \
        + (["rseg"] if _TG.RSEG else [])
    E.emit(src, ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb), "%s.token_tiles=%d" % (sym, tt)]
           + (["%s.tokens=%d" % (sym, B)] if masked else []), out, outdir, gate=None if extra else gate, vtry=vtry)
    return (out, tile, rowgrp, tt)


def rseg_gemms(B, T, outdir):
    """The GPU's GEMMs over the FFNSW5 layers' row-segmented gate / up (RSEG_GEMMS): the fused ffn's afrag forms and
    the gate kstore / up swiglu (and their variants), named "<plain name>_rseg<F>"; dispatch.txt rows."""
    import gen_gemm_tile as TG
    out, F = [], NPU_FFNBLKS
    for fmt, mt, kb in sorted(RSEG_GEMMS):
        M, sfx = mt * 16, gen_gemv.rseg_suffix(F)
        try:
            TG.RSEG = (M, F, 0)
            t = dataclasses.replace(TG.default_tile(fmt, "swiglu", kb), **AF_TILE, **AF_FFN[fmt], ffn=True)
            for suffix, tt in ((".af.hal", t), (".af.to.hal", dataclasses.replace(t, tout=True))):
                TG.check(tt)
                r = _emit_gen(lambda f, k: TG.gen(f, k, tt, False), tt.bn, fmt, mt, kb, B,
                              "gemm_ffn_%s_%d_%d%s%s" % (fmt, mt, kb, sfx, suffix), outdir, "ffn", tt.rowgrp)
                assert r, (fmt, suffix)
                out.append(r)
            for kind, off in (("kstore", 0), ("swiglu", M)):   # (the up rows from M in the gate-based binding)
                TG.RSEG = (M, F, off)
                name = "gemm_%s_%s_%d_%d%s.hal" % (kind, fmt, mt, kb, sfx)
                r = gemm(fmt, mt, kb, B, name, outdir, kind=kind)
                assert r, name
                out.append(r)
                if T > B:
                    out.extend(narrow_variants(fmt, mt, kb, B, name, outdir, kind))
                out.extend(afrag_variants(fmt, mt, kb, B, name, outdir, kind))
        finally:
            TG.RSEG = None
    return out


def gemm_combos(rows):
    """{(kind, fmt, port, m_tiles, k_blocks): [tensor names]} of the model's GEMM tensors; kind is kstore, residual or
    swiglu. rows: emit_prefill.parse(model)."""
    combos = {}
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
        combos.setdefault((kind, fmt, port, dims[1] // 16, dims[0] // qk), []).append(nm)
    return combos


def gemm_kinds(kind, mt):
    """HAL kinds emitted for a combo, in order; the driver runs the last one (kres over kstore, kqg over kstore)."""
    return {"kstore": ["kstore"] + (["kqg"] if mt == 768 else []), "residual": ["kstore", "kres"],
            "swiglu": ["swiglu"]}[kind]


def main():
    global MODEL, NPU_FFNBLK, GSIG_ON, QSPLIT_ON
    model, outdir = sys.argv[1], sys.argv[2]
    MODEL = model
    E.ASYNC = True
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    # GEMM tokens per workgroup of the hand-written sources
    TILE = 64
    if B % TILE:
        raise SystemExit("tokens=%d must be a multiple of the token tile=%d" % (B, TILE))
    TT = B // TILE
    # An NPU call replays each memory-tile W panel once per 64-token M block: past 63 (B > 4032) the HRX pair-replay
    # patch counts replay pairs, whose twin descriptors fit the memory tile only with one replay group per call (set in
    # the environment too: the calibration menu's worker processes keep the same groups)
    import gen_bfp16_encode as GE0
    if B // 64 > 63 and not os.environ.get("YAH_NPU_GROUPS"):
        os.environ["YAH_NPU_GROUPS"] = str(NPU_PASSES)
        GE0.GROUPS = (NPU_PASSES,)
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
    dnseg_plan(rows)
    combos = gemm_combos(rows)
    types = {nm: ty for nm, _, ty in rows}
    # R2P_LAZY (gen_attn_fa: lazy rescale threshold, nats) 5, as the npu branch: 2 x 32 busiest SQ +0.8 at 126K on real
    # kv8 inputs, native kld_ref 0.0164 = exact
    if "R2P_LAZY" not in os.environ:
        gen_attn_fa.R2P_LAZY = 5.0
    GSIG_ON = bool(GSIG) and "q" not in NPU_SPLIT
    if GSIG_ON:
        import gen_gemm_tile
        assert GSIG in ("u16", "16", "32"), GSIG
        gen_gemm_tile.KQG_GSIG, gen_gemm_tile.KQG_GSIG16 = GSIG, GSIG in ("16", "u16")
        gen_attn_fa.GSIG, gen_attn_fa.GSIG16 = GSIG, GSIG in ("16", "u16")
        os.environ["KQG_GSIG"] = os.environ["GSIG"] = GSIG   # (the calibration menu's worker processes)
    QSPLIT_ON = QSPLIT and GSIG_ON and GSIG in ("16", "u16") and QNORM and gen_attn_fa.Q16
    if QSPLIT_ON:
        import gen_gemm_tile
        gen_gemm_tile.KQG_QSPLIT = gen_attn_fa.R2P_QSPLIT = True
        os.environ["KQG_QSPLIT"] = os.environ["R2P_QSPLIT"] = "1"

    n = 0
    geom = []  # (<hal>, <tokens per workgroup>, <row groups>, <token_tiles>)
    for kind, fmt, port, mt, kb in sorted(combos):
        name = lambda k: "gemm_%s_%s_%d_%d.hal" % (k, fmt, mt, kb)
        # A residual projection gets a kstore and the fused-residual kres; the attention q projection (12288 rows =
        # 24 heads x [q | gate]) also gets kqg, which stores q and gate unpacked. The driver prefers kres and kqg.
        kinds = gemm_kinds(kind, mt)
        for k in kinds:
            r = gemm(fmt, mt, kb, B, name(k), outdir, kind=k)
            if r:
                geom.append(r)
                if T > B:  # chunked sets: their last chunk may be short; a one-pass set always runs all B tokens
                    geom.extend(narrow_variants(fmt, mt, kb, B, name(k), outdir, k))
                geom.extend(afrag_variants(fmt, mt, kb, B, name(k), outdir, k))
            elif k == "kstore" and fmt == "q2k":
                # Q2_K has no tile decoder; its only tensors here are the 48-row ssm_alpha / ssm_beta.
                assert (fmt, mt, kb) not in DNSEG, "the hand-written Q2_K GEMM reads GGUF rows"
                sym = E.sym_of("yah_ffn_gemm_q2k_f32.loom")
                E.emit("yah_ffn_gemm_q2k_f32.loom", ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb),
                                                     "%s.token_tiles=%d" % (sym, TT)], name(k), outdir)
                geom.append((name(k), TILE, 1, TT))
            elif k == kinds[0]:
                raise SystemExit("no GEMM kernel for %s %s (%d rows, K = %d)" % (fmt, k, mt * 16, kb * qk_of(fmt)))
        n += 1

    geom.extend(ffn_fused(rows, B, outdir))
    geom.extend(kst2(rows, B, outdir))
    geom.extend(seg_gemms(rows, B, outdir))
    if NPU_SPLIT or NPU_FFNBLK:
        import gen_bfp16_encode as GE
        import gen_npu_unpack as GU
        # one replay group per call (the fillers' W-DMA raw ring hangs with two), so C is one partial
        groups, cg = GE.GROUPS, GU.CG
        GE.GROUPS, GU.CG = (NPU_PASSES,), 1
        try:
            with GN.np_override(NPU_NP):
                geom.extend(npu_split(rows, combos, B, outdir))
        finally:
            GE.GROUPS, GU.CG = groups, cg
    for kind, fmt, port, mt, kb in sorted(c for c in combos if c[1:2] + c[3:] in FFNBLKS_DOWNS):
        sfx = gen_gemv.kseg_suffix(dn_kseg(kb, NPU_FFNBLKS))
        name = lambda k: "gemm_%s_%s_%d_%d%s.hal" % (k, fmt, mt, kb, sfx)
        with kseg_f(NPU_FFNBLKS):
            for k in gemm_kinds(kind, mt):
                r = gemm(fmt, mt, kb, B, name(k), outdir, kind=k)
                assert r, name(k)
                geom.append(r)
                if T > B:
                    geom.extend(narrow_variants(fmt, mt, kb, B, name(k), outdir, k))
                geom.extend(afrag_variants(fmt, mt, kb, B, name(k), outdir, k))
    geom.extend(rseg_gemms(B, T, outdir))


    # Attention: tools/gen_attn_fa.py, 32 tokens x 2 heads per workgroup; reads V^T (vtrans.hal, or the paged / quantized pools).
    # KV paging (256-token pages) needs a context that is a multiple of 256; otherwise the caches stay contiguous.
    kv_paged = T % 256 == 0
    if not kv_paged:
        print("KV paging off: needs the context to be a multiple of 256")
    gen_attn_fa.PAGED = gen_kvq.PAGED = kv_paged
    gen_attn_fa.MAX_TOKENS = max(B, 2048)
    # YAH_ATTN_SHAPE=<heads>x<tokens>: attention workgroup shape. 6 x 16 (a whole GQA group: each K / V tile staged once for
    # its 6 heads) is bit-identical to 2 x 32, pp2048 attention -4.9%, pp8192 even
    a_hpw, a_qt = map(int, os.environ.get("YAH_ATTN_SHAPE", "2x32").split("x"))
    gen_attn_fa.configure(a_hpw, a_qt)
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    # kv4 / kv8 (paged): decode once per attention call instead of in every workgroup's staging (llama.cpp's prefill
    # flash attention converts quantized K / V to f16 the same way): yah_kdeq / yah_vdeq (per chunk) fill an f16 pool pair
    # with the staging decode's own arithmetic and the attention is the fp16 paged kernel (kv4: on yah_qrot's rotated Q,
    # kv8: on RoPE's f16 Q as before), bit-identical. kv4 attention -12% at 32K depth, kv8 -21%. dispatch.txt "kvdeq".
    kvdeq = kv_paged and ((gen_attn_fa.K4 and gen_attn_fa.VQ4) or (gen_attn_fa.K8 and gen_attn_fa.VQ8))
    deq_bits = 4 if gen_attn_fa.K4 else 8
    kv8deq = kvdeq and deq_bits == 8
    A = gen_attn_fa
    kv_flags = (A.K4, A.K8, A.KDEC, A.VQ4, A.VQ8)
    if kv8deq:   # the main sources are the fp16 paged kernel
        A.K8 = A.KDEC = A.VQ8 = False
    # the 4-workgroup kernel's bit-identical VALU cuts (gen_attn_fa MAXNN, SUM0, EPI_FAST; CTX16: start_pos = c * B):
    # pp2048 attention -6.4% cycles
    # QB_ZIGZAG: the short causal blocks are mostly Q / gate / O traffic; launched last (longest first) they ran as a
    # DRAM-bound tail. Interleaved with the long ones: pp2048 attention -5.5% cycles, bit-identical, neutral at depth
    # PV_FENCE: -16 v_mov per tile, pp2048 attention -2.3% cycles (-3% in the pipeline), bit-identical
    # R2P: the reuse-2 role split (half the LDS fragment reads, no duplicated softmax), bit-identical: pp2048 attention
    # -7.3% cycles (62.8 -> 67.6% of floor), 8K -9.6% (71.6 -> 79.3%). YAH_ATTN_R2P=0: the head_dim split
    # One attention kernel at every depth: the persistent kernel (gen_attn_fa PERSIST, YAH_ATTN_PERSIST=<slots>, 80; 0:
    # the grid kernel), workgroups running whole units, bit-identical; dispatch.txt "persist_<hal> <slots> 1 <dyn>" rows.
    # Busiest SQ % of floor 11-03: 82.7 / 85.4 / 85.8 / 85.8 at 16K / 32K / 96K / 124K, where the 3-workgroup twins it
    # replaced (KT2, PAIR "deep") ran 75-77 / 73.
    persist = int(os.environ.get("YAH_ATTN_PERSIST", str(320 // gen_attn_fa.NQB)))   # 20 WGPs x 4 (2x64: x 2)
    MAIN_FLAGS = dict(MAXNN=True, SUM0=True, EPI_FAST=True, CTX16=B % 16 == 0, QB_ZIGZAG=not persist, PERSIST=persist, PV_FENCE=True,
                      R2P=os.environ.get("YAH_ATTN_R2P", "1") != "0", PRO_FAST=os.environ.get("YAH_ATTN_PROF", "1") != "0")
    # YAH_ATTN_PREV=1 (with PERSIST): half the slots run their snake rounds in reverse (gen_attn_fa PERSIST_REV):
    # 8K busiest SQ 78.9 -> 81.1% of floor (clean grid 81.3%), bit-identical
    if persist and os.environ.get("YAH_ATTN_PREV") == "1":
        MAIN_FLAGS.update(QB_ZIGZAG=False, PERSIST_REV=True)
    # YAH_ATTN_PDYN (default 1, with PERSIST): the slots take their units from an atomic queue (gen_attn_fa PERSIST_DYN), bit-identical;
    # standalone wall vs the grid kernel: tie to 8K, -2% at 16K, -5% at 64K, -11% at 126K. dispatch.txt "persist_<hal>
    # <slots> 1 1": the driver binds a zeroed queue after the others
    pdyn = bool(persist) and os.environ.get("YAH_ATTN_PDYN", "1") == "1"
    if pdyn:
        MAIN_FLAGS.update(QB_ZIGZAG=False, PERSIST_DYN=True)


    def write_attn(path):
        A = gen_attn_fa
        saved = {k: getattr(A, k) for k in MAIN_FLAGS}
        for k, v in MAIN_FLAGS.items():
            setattr(A, k, v)
        if A.KDEC:   # R2P has no in-kernel KV decode
            A.R2P = False
        open(path, "w").write(A.gen())
        if not A.R2P:   # the head_dim split kernel's "_alt" twin without DIAG_SWAP (attn_guard)
            saved["DIAG_SWAP"] = A.DIAG_SWAP
            A.DIAG_SWAP = False
            open(alt(path), "w").write(A.gen())
        elif os.path.exists(alt(path)):   # (a stale twin from an older emit into this directory)
            os.remove(alt(path))
        for k, v in saved.items():
            setattr(A, k, v)
    gen_attn_fa.QNORM = QNORM and gen_attn_fa.Q16 and gen_attn_fa.QDIRECT is not False   # (see RoPE below)
    if GSIG_ON:
        geom.append(("gsig", 0, 0, 0))
    if QSPLIT_ON:
        geom.append(("qsplit", 0, 0, 0))
    # YAH_ATTN_KDB=0|1: the 2 x 32 R2P kernels double buffer the QK waves' K fragments (gen_attn_fa R2P_KDB) with the
    # lane values recomputed per role branch (R2P_RELANE), so Loom's bank-aware allocation fits 192 VGPRs: tiled kernel
    # 2K..16K mean SQ +0.3..0.8, busiest +0.1..0.7, bit-identical (8K gate window logits equal). Chunk 0 and the 2 x 64
    # kernels keep the single buffer (depth 0 -0.4 / tiled -2.5, 64K -0.2)
    kdb_main = bool(MAIN_FLAGS["R2P"] and persist) and os.environ.get("YAH_ATTN_KDB", "1") == "1"
    gen_attn_fa.R2P_KDB = gen_attn_fa.R2P_RELANE = kdb_main
    attn_src = os.path.join(tmp, "yah_attn_hip.loom")
    write_attn(attn_src)
    if os.environ.get("YAH_ATTN_SRC_ONLY"):   # (dev loop: this attention source only, no set)
        shutil.copyfile(attn_src, os.environ["YAH_ATTN_SRC_ONLY"])
        os._exit(0)
    import srcseed
    c0seed = int(ATTN_C0SEED) if ATTN_C0SEED is not None else 29 if QSPLIT_ON else 0
    c0rot16 = QSPLIT_ON and ATTN_C0ROT16

    def write_c0(src, path):   # chunk 0's twin of an attention source (c0rot16 / kdb regenerate it)
        kdb = gen_attn_fa.R2P_KDB   # (chunk 0 keeps the single K buffer: depth 0 busiest SQ -0.4, tiled -2.5)
        gen_attn_fa.R2P_KDB = gen_attn_fa.R2P_RELANE = False
        if c0rot16:
            gen_attn_fa.R2P_QROT16 = True
            write_attn(path)
            gen_attn_fa.R2P_QROT16 = False
            src = path
        elif kdb:
            write_attn(path)
            src = path
        gen_attn_fa.R2P_KDB = gen_attn_fa.R2P_RELANE = kdb
        if c0seed:
            text = srcseed.seed_order(open(src).read(), c0seed)
            open(path, "w").write(text)
        return path if c0seed or c0rot16 or kdb else src
    attn_c0_src = write_c0(attn_src, os.path.join(tmp, "yah_attn_hip_c0.loom"))
    # afrag o-projections (K = 6144 kres) read the attention output fragment-major: wmma_t[_c<i>].hal
    af24 = any(h.startswith("gemm_kres_") and h.endswith("_320_24.af.hal") for h, *_ in geom)
    attn_t_src = os.path.join(tmp, "yah_attn_hip_t.loom")
    attn_t_c0_src = attn_t_src
    if af24:
        gen_attn_fa.TILED_OUT = True
        write_attn(attn_t_src)
        attn_t_c0_src = write_c0(attn_t_src, os.path.join(tmp, "yah_attn_hip_t_c0.loom"))
        gen_attn_fa.TILED_OUT = False
    # YAH_ATTN_W64=<chunk>: chunks from <chunk> on run the 2 x 64 R2P kernel (half the K / V staging per MMA, bit-identical).
    # Its 16-wave workgroups run 2 per WGP and phase-lock: it wins only at depth.
    gen_attn_fa.R2P_KDB = gen_attn_fa.R2P_RELANE = False
    w64_from = int(os.environ.get("YAH_ATTN_W64", "0")) or NCH
    attn64_src = attn64_t_src = None
    persist64 = persist // 2
    # YAH_ATTN_KVG64=g: the 2 x 64 kernels (and their key split) deal units g KV heads at a time (gen_attn_fa ATTN_KVG,
    # L2 reuse at depth, energy); the 2 x 32 and chunk-0 kernels keep the default order (KVG costs them busiest SQ:
    # -1.7 at depth 0, -1.4 at 8K, bit-identical).
    kvg_main, kvg64 = gen_attn_fa.KVG, int(os.environ.get("YAH_ATTN_KVG64", str(gen_attn_fa.KVG)))
    gen_attn_fa.KVG = kvg64
    # YAH_ATTN_QFOLD64=0|1|2: the 2 x 64 kernels fold r log2e into Q (gen_attn_fa R2P_QFOLD mode). mode 2 is the default
    # with YAH_ATTN_KVG64 (kv8 gpuk8v: 64K busiest SQ 88.0 -> 88.3, 126K slot KL 3.0e-7, 0 flips); mode 1 and the 2 x 32
    # kernel lose (WMMA bank rank 14-16: 64K -1.2, 8K -1.7..-2.3)
    qfold_main = gen_attn_fa.R2P_QFOLD
    gen_attn_fa.R2P_QFOLD = int(os.environ.get("YAH_ATTN_QFOLD64", "2" if os.environ.get("YAH_ATTN_KVG64") else str(qfold_main)))
    # YAH_ATTN_PCHK64=0|1: the 2 x 64 kernels test the lazy rescale on each lane's 8-key max (gen_attn_fa R2P_PCHK,
    # bit-identical): 64K busiest SQ 88.3 -> 88.9; the 2 x 32 kernel keeps the row max (8K: mean +0.0, one SQ slower)
    pchk_main = gen_attn_fa.R2P_PCHK
    gen_attn_fa.R2P_PCHK = os.environ.get("YAH_ATTN_PCHK64", "1") == "1"
    if w64_from < NCH:
        assert MAIN_FLAGS["R2P"] and persist and (a_hpw, a_qt) == (2, 32) and w64_from >= 1
        gen_attn_fa.configure(2, 64)
        MAIN_FLAGS["PERSIST"] = persist64
        attn64_src = os.path.join(tmp, "yah_attn_hip64.loom")
        write_attn(attn64_src)
        if af24:
            attn64_t_src = os.path.join(tmp, "yah_attn_hip64_t.loom")
            gen_attn_fa.TILED_OUT = True
            write_attn(attn64_t_src)
            gen_attn_fa.TILED_OUT = False
        MAIN_FLAGS["PERSIST"] = persist
        gen_attn_fa.configure(a_hpw, a_qt)
    # YAH_ATTN_KSPLIT=<chunk>: from that 2 x 64 chunk on, the tail units (units % slots) run as 5 key pieces in a
    # two-kernel split: wmma[_t]_c<c>.hal stores the pieces' f32 partials (gen_attn_fa R2P_DIAG ksstore), then
    # wmmak[_t]_c<c>.hal merges them in piece order and writes the tail units' output (ksmerge). Busiest SQ -5.7% at
    # 126K (balanced rounds), not bit-identical. dispatch.txt "ksplit_<hal> <partial floats> 0 0": the driver binds a
    # partials buffer after the queue and runs the merge after the attention. (ATTN_KSPLIT alone, the one-kernel split
    # with a cross-workgroup merge, hung the GPU: refused)
    if gen_attn_fa.KSPLIT:
        sys.exit("ATTN_KSPLIT: the one-kernel split hangs the GPU; use YAH_ATTN_KSPLIT=<chunk>")
    ks_from = int(os.environ.get("YAH_ATTN_KSPLIT", "0")) or NCH
    attn64ks_src = attn64ks_t_src = attn64km_src = attn64km_t_src = None
    ksn_srcs = {}   # (YAH_NPU_ATTN) the twins: stem -> (src, tiled src)
    ks_floats = ks_floats_npu = 0
    npu_kv = int(os.environ.get("YAH_NPU_ATTN", "0"))
    npu_from = NCH
    if ks_from < NCH:
        assert w64_from <= ks_from and pdyn
        gen_attn_fa.configure(2, 64)
        MAIN_FLAGS["PERSIST"] = persist64
        gen_attn_fa.KSPLIT = 5
        # YAH_ATTN_PRIO64=0|1: the key-split store kernels raise the QK waves' issue priority in their MMA section
        # (gen_attn_fa R2P_PRIOQ 1) on the relane pass: busiest SQ -0.5% at 16K..126K, bit-identical (alone each <= -0.15%)
        prio64 = os.environ.get("YAH_ATTN_PRIO64", "1") == "1"
        # YAH_NPU_ATTN=n (with YAH_ATTN_KSPLIT): the pair's NPU twins yah_attn_hip64{ks,km}n.loom (gen_attn_fa NPU_KV n:
        # the last n KV heads' units shared with the NPU attention) run chunks from YAH_NPU_ATTN_FROM (default: the
        # first key-split chunk; at least 1: the twins need start_pos >= 512), beside the NPU attention (gen_npu_attn,
        # npu_attn.xdna) and its encoders npua{k,v,q}_c<c>.hal (gen_npu_attn_encode's layer forms). dispatch.txt
        # "npuattn <n> <nb built> <first chunk>"; the twins' "ksplit_" rows carry their larger partials
        if npu_kv:
            npu_from = max(int(os.environ.get("YAH_NPU_ATTN_FROM", "0")) or ks_from, ks_from, 1)
            # (the NPU's q GEMM rows leave the GPU's kqg GEMM without its split-norm q: the q encoder then reads the
            # attention's raw f32 q, gen_npu_attn_encode qlay 2)
        for diag, stem in (("ksstore", "ks"), ("ksmerge", "km")) + ((("ksstore", "ksn"), ("ksmerge", "kmn")) if npu_kv else ()):
            gen_attn_fa.NPU_KV = npu_kv if stem.endswith("n") else 0
            gen_attn_fa.R2P_DIAG.add(diag)
            gen_attn_fa.R2P_RELANE = prio64 and stem.startswith("ks")
            gen_attn_fa.R2P_PRIOQ = 1 if prio64 and stem.startswith("ks") else 0
            p = os.path.join(tmp, "yah_attn_hip64%s.loom" % stem)
            write_attn(p)
            pt = None
            if af24:
                pt = os.path.join(tmp, "yah_attn_hip64%s_t.loom" % stem)
                gen_attn_fa.TILED_OUT = True
                write_attn(pt)
                gen_attn_fa.TILED_OUT = False
            gen_attn_fa.R2P_DIAG.discard(diag)
            gen_attn_fa.R2P_RELANE, gen_attn_fa.R2P_PRIOQ = False, 0
            gen_attn_fa.NPU_KV = 0
            if stem == "ks":
                attn64ks_src, attn64ks_t_src = p, pt
            elif stem == "km":
                attn64km_src, attn64km_t_src = p, pt
            else:
                ksn_srcs[stem] = (p, pt)
        p0, gen_attn_fa.PERSIST = gen_attn_fa.PERSIST, persist64   # write_attn applies MAIN_FLAGS only while it generates
        ks_floats = gen_attn_fa.ksplit_floats()
        gen_attn_fa.NPU_KV = npu_kv
        ks_floats_npu = gen_attn_fa.ksplit_floats()
        gen_attn_fa.NPU_KV = 0
        gen_attn_fa.PERSIST = p0
        gen_attn_fa.KSPLIT = 0
        MAIN_FLAGS["PERSIST"] = persist
        gen_attn_fa.configure(a_hpw, a_qt)
    gen_attn_fa.KVG = kvg_main
    gen_attn_fa.R2P_QFOLD = qfold_main
    gen_attn_fa.R2P_PCHK = pchk_main
    A.K4, A.K8, A.KDEC, A.VQ4, A.VQ8 = kv_flags
    vtrans_src = os.path.join(tmp, "yah_transpose_v16.loom")
    open(vtrans_src, "w").write(gen_attn_fa.gen_vtrans())
    # kv4 at depth: the Q rotation in its own kernel (gen_attn_fa QROT) and the attention on direct Q fragments, from the
    # chunk at QROT_FROM tokens on: 4 workgroups per WGP instead of 3, -4.2% attention at 8K..120K depth (the rotation's
    # 1.5 M cycles break even at ~8K). dispatch.txt "qrot <first chunk>"; the driver rotates q into its own f16 buffer.
    # kv4 under kvdeq: Q rotated from chunk 0 (its K is H256-rotated)
    qrot_first = 0 if kvdeq and deq_bits == 4 else -(-QROT_FROM // B) if gen_attn_fa.K4 and kv_paged and NCH > 1 else NCH
    if QSPLIT_ON and qrot_first < NCH:   # (the Q rotation kernel and its attention read the f32 q)
        sys.exit("YAH_QSPLIT: not with the Q rotation kernel (kv4 past QROT_FROM, kvdeq); set YAH_QSPLIT=0")
    attn_qr_src = os.path.join(tmp, "yah_attn_hip_qr.loom")
    attn_qr_t_src = os.path.join(tmp, "yah_attn_hip_qr_t.loom")
    qrot_src = os.path.join(tmp, "yah_qrot.loom")
    kdeq_src = os.path.join(tmp, "yah_kdeq%d.loom" % deq_bits)
    vdeq_src = os.path.join(tmp, "yah_vdeq%d.loom" % deq_bits)
    if qrot_first < NCH:
        A = gen_attn_fa
        saved = (A.Q16, A.QROT, A.VSWZ_T, A.K4, A.K8, A.KDEC, A.VQ4, A.VQ8)
        if kvdeq:   # the fp16 paged kernel (the f16 pools have the int4 pools' page layout)
            A.K4 = A.K8 = A.KDEC = A.VQ4 = A.VQ8 = False
            A.Q16 = True
        else:
            # VSWZ_T off: with direct Q the swizzled V staging tips Loom into rotating the O accumulators at the back edge
            A.Q16 = A.QROT = True
            A.VSWZ_T = False
        write_attn(attn_qr_src)
        if af24:
            A.TILED_OUT = True
            write_attn(attn_qr_t_src)
            A.TILED_OUT = False
        A.Q16, A.QROT, A.VSWZ_T, A.K4, A.K8, A.KDEC, A.VQ4, A.VQ8 = saved
        open(qrot_src, "w").write(A.gen_qrot())
        geom.append(("qrot", qrot_first, 0, 0))
    if kvdeq:
        # ATTN_KRAW=1 (env, read by gen_attn_fa): kdeq8 writes the raw codes and the kscale buffer the attention scales
        # its partials by (dispatch.txt "kraw")
        assert not gen_attn_fa.KRAW or deq_bits == 8
        open(kdeq_src, "w").write(gen_attn_fa.gen_kdeq(deq_bits, raw=gen_attn_fa.KRAW, khalf=gen_attn_fa.KH))
        open(vdeq_src, "w").write(gen_attn_fa.gen_vdeq(deq_bits))
        geom.append(("kvdeq", 0, 0, 0))
        if gen_attn_fa.KRAW:
            geom.append(("kraw", 0, 0, 0))
    geom.append(("wmma.hal", a_qt, a_hpw, (B + a_qt - 1) // a_qt))
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
    # It needs B % 32 == 0 (main's token tile 64 holds it).
    dn_src = os.path.join(tmp, "yah_deltanet_hip_f32.loom")
    open(dn_src, "w").write(gen_gdn_chunk.gen())
    # the conv pairs with the DeltaNet kernel: f16 normalized hand-off
    # the f16 hand-off conv per (256 channels, gen_conv_kq.TB tokens) workgroup: bit-identical, inputs read once
    assert 640 in O16_MT and B % gen_conv_kq.TB == 0
    dn_vconv = DNSPLIT == (40, 8) and "qkv" not in NPU_SPLIT
    if dn_vconv:
        text = gen_conv_kq.gen(4096)
        geom.append(("dnvconv", gen_conv_kq.tiles(4096), 0, 0))
    else:
        text = gen_conv_kq.gen()
    conv_src = os.path.join(tmp, "yah_ssm_conv_kq.loom")
    open(conv_src, "w").write(text)
    prepab_src = os.path.join(tmp, "yah_deltanet_prep_ab.loom")
    text = open(os.path.join(E.LOOM, "yah_deltanet_prep_ab_f32.loom")).read()
    open(prepab_src, "w").write(prepab_x16(text) if 640 in O16_MT else text)
    postnorm_base = open(os.path.join(E.LOOM, "yah_ssm_postnorm_gate_f16.loom")).read()
    if 384 in O16_MT:
        postnorm_base = postnorm_g16(postnorm_base)
    postnorm_src = os.path.join(tmp, "yah_ssm_postnorm.loom")
    open(postnorm_src, "w").write(postnorm_base)
    dn_ss = dn_vconv   # the DeltaNet parts on the state-stationary kernel
    # blocks per head of the DeltaNet grid (the unsplit rowsplit.hal is not dispatched with dnsplit)
    geom.append(("rowsplit.hal", 0, 1 if dn_ss else 2, 0))
    geom.append(("dnthreads", gen_gdn_ss.THREADS if dn_ss else 256, 0, 0))
    dn_parts = []
    if DNSPLIT:
        ha, hb = DNSPLIT
        for tag, h0, nh in (("a", 0, ha), ("b", ha, hb)):
            src = os.path.join(tmp, f"yah_deltanet_{tag}.loom")
            if dn_ss:
                open(src, "w").write(gen_gdn_ss.gen(list(range(h0, h0 + nh))))
            else:
                open(src, "w").write(gen_gdn_chunk.gen(h0, nh))
            dn_parts.append((tag, h0, nh, src))
        geom.append(("dnsplit", 0, ha, hb))
    geom.append(("convtb", gen_conv_kq.TB, 0, 0))

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
    npua = []   # (YAH_NPU_ATTN) the encoders and the GPU's wait: (src, hal)
    npu_nbs = 16 * ((NCH - 1) * B // 512 + 1)   # the image's nb: the last chunk's
    if npu_kv:
        # the paged f16 K / V^T pools, or kv8's per-layer f16 copies of them (kvdeq, not KRAW's raw codes)
        if qrot_first < NCH or not ((paged_f16k and paged_f16v) or (kv8deq and not gen_attn_fa.KRAW)):
            sys.exit("YAH_NPU_ATTN: the encoders read paged f16 K / V^T pools (fp16 or kv8) and the attention's own q")
        import gen_npu_attn
        import gen_npu_attn_encode as NE
        import gen_npu_unpack as GU
        for c in range(npu_from, NCH):
            call = NE.Call(mc=16, nb=16 * (c * B // 512 + 1), nbs=npu_nbs, pages=T // 256, B=B, start_pos=c * B,
                           kvn=npu_kv, qlay=0 if QSPLIT_ON else 2 if gen_attn_fa.QNORM else 1)
            call.check()
            for kind, gen in (("k", NE.gen_k), ("v", NE.gen_v), ("q", NE.gen_q)):
                src = os.path.join(tmp, "npua%s_c%d.loom" % (kind, c))
                open(src, "w").write(gen(call))
                npua.append((src, "npua%s_c%d.hal" % (kind, c)))
        if not (NPU_SPLIT or NPU_FFNBLK):   # (a GEMM split emits the same wait)
            src = os.path.join(tmp, "npu_flag_wait.loom")
            open(src, "w").write(GU.gen_flag_wait())
            npua.append((src, "npu_flag_wait.hal"))
        img = os.path.join(tmp, "npu_attn.loom")
        open(img, "w").write(gen_npu_attn.emit(16, npu_nbs))
        gen_npu_attn.compile_xdna(img, os.path.join(outdir, "npu_attn.xdna"))
    rope_src = "yah_fused_qk_rope_batched_f32.loom"
    if paged_f16k:   # RoPE writes K rows straight into the paged pool
        rope_src = os.path.join(tmp, "yah_fused_qk_rope_batched_kpaged.loom")
        open(rope_src, "w").write(rope_kpaged(open(os.path.join(E.LOOM, "yah_fused_qk_rope_batched_f32.loom")).read()))
        geom.append(("rope_kpaged", 0, 0, 0))
    text = open(rope_src if os.path.isabs(rope_src) else os.path.join(E.LOOM, rope_src)).read()
    rope_src = os.path.join(tmp, "yah_fused_qk_rope_batched_live.loom")
    text = rope_live(text)
    vtrope = paged_f16k and paged_f16v   # RoPE stores V^T into the paged pool: no yah_vtpage pass
    if vtrope:
        text = rope_vtpage(text)
        geom.append(("vtrope", 0, 0, 0))
    open(rope_src, "w").write(text)
    # QNORM: the attention takes the raw f32 q and does the head norm + RoPE + f16(q / 16) in its prologue (same bits);
    # RoPE does K / V only. dispatch.txt "qnorm"
    if gen_attn_fa.QNORM:
        text = open(rope_src).read()
        rope_src = os.path.join(tmp, "yah_fused_qk_rope_batched_konly.loom")
        open(rope_src, "w").write(rope_konly(text))
        geom.append(("qnorm", 0, 0, 0))
    elif gen_attn_fa.Q16:   # RoPE hands the attention f16(q / 16) (not with K4: its H256 rotates the f32 Q)
        text = open(rope_src if os.path.isabs(rope_src) else os.path.join(E.LOOM, rope_src)).read()
        rope_src = os.path.join(tmp, "yah_fused_qk_rope_batched_q16.loom")
        open(rope_src, "w").write(rope_q16(text))
        geom.append(("q16", 0, 0, 0))
    if paged_f16v:
        if not vtrope:
            vtpage_src = os.path.join(tmp, "yah_vtpage.loom")
            open(vtpage_src, "w").write(gen_kvq.gen_vtpage())
        vtrans_src = None    # replaced by the paged per-chunk transpose (yah_vtpage or RoPE's own store)
    if NCH > 1:
        # quantized KV: K quantizers run per chunk on cache slices (kmean on chunk 0 only), V quantizers per chunk (vq*_c<i>.hal)
        geom.append(("ctx", B, 0, T))     # chunk size, total context
    # Each row over NORM_SPLIT waves (gen_half_norm.gen_split: the same per-lane chain order, so the same bits):
    # 0.383 -> 0.299 ms per call at 2048 rows (latency-bound -> ~214 GB/s). The driver reads the split from dispatch.txt.
    geom.append(("norm_split", 0, NORM_SPLIT, 0))
    # The chunk's token_embd rows from its token ids on the GPU (gen_decode_misc.gen_embed_iq4xs, one workgroup per row),
    # so the host stages ids and never waits between chunks.
    embed_src = os.path.join(tmp, "yah_embed.loom")
    if types["token_embd.weight"] != 8:
        raise SystemExit("emit_prefill_pp: the embed kernel reads a Q8_0 token_embd")
    open(embed_src, "w").write(gen_decode_misc.gen_embed_q8_0(rows=B))
    geom.append(("embed.hal", 1, 5120 // 16, B))   # one token per workgroup
    geom.append(("native", 0, 0, 0))   # the engine loads the GGUF as it is (core/gguf_convert.cpp)
    # the output head: the tile-free GEMV (gen_gemv) for Q8_0 (a GGUF head that is not Q6_K, which
    # yah_gemv_q6k_f32 reads), rows per workgroup / lanes / workgroups in its row
    head_src, head_fmt = None, gen_gemv.GGML.get(types["output.weight"])
    if head_fmt:
        head_src = os.path.join(tmp, "yah_gemv_head.loom")
        open(head_src, "w").write(gen_gemv.gen("plain", [head_fmt], 248320, 5120))
        geom.append(("gemv.hal", gen_gemv.rows_per_wg(), 32 * 4, 248320 // gen_gemv.rows_per_wg()))
    with open(os.path.join(outdir, "dispatch.txt"), "w") as fh:
        for hal, tk, rg, tt in geom:
            fh.write("%s %d %d %d\n" % (hal, tk, rg, tt))

    # The output norm (tools/gen_half_norm.py): fully unrolled, all loads issued up front.
    norm_src = os.path.join(tmp, "yah_half_norm_unrolled.loom")
    # 4 rows (waves) per workgroup: 46.1 -> 41.9 ms per pp2048 vs one row per workgroup. The tuner's table may set it
    # ({"norm": {"wpr": 2}}); the driver derives the grid from the kernel's workgroup size.
    open(norm_src, "w").write(gen_half_norm.gen_split(5120, wpr=4, split=NORM_SPLIT))
    # the FFN norm for afrag GEMMs: the same values, stored fragment-major
    normt_src = os.path.join(tmp, "yah_half_norm_tiled.loom")
    open(normt_src, "w").write(gen_half_norm.gen_split(5120, wpr=4, split=NORM_SPLIT, tiled=True))
    any_af = any(h.endswith(".af.hal") for h, *_ in geom)
    # attn_norm where some of its GEMMs are afrag (DeltaNet alpha / beta never): both layouts in one pass
    normrt_src = os.path.join(tmp, "yah_half_norm_both.loom")
    open(normrt_src, "w").write(gen_half_norm.gen_split(5120, wpr=4, split=NORM_SPLIT, tiled="both"))
    # the DeltaNet postnorm, fragment-major for an afrag ssm_out
    postnorm_t_src = os.path.join(tmp, "yah_ssm_postnorm_tiled.loom")
    open(postnorm_t_src, "w").write(postnorm_tiled(postnorm_base))
    pn_parts = []
    for tag, h0, nh, _ in dn_parts:
        base = postnorm_base
        for t_, txt in (("", postnorm_heads(base, h0, nh, B)), ("_t", postnorm_tiled(postnorm_heads(base, h0, nh, B)))):
            src = os.path.join(tmp, f"yah_ssm_postnorm{t_}_{tag}.loom")
            open(src, "w").write(txt)
            pn_parts.append((f"postnorm{t_}_{tag}.hal", src, nh))
    fixed = [
        (embed_src, "embed.hal", ["nop=0"]),
        ("yah_residual_add_1d_f32.loom", "accum.hal",
         ["yah_residual_1d.dim=%d" % (5120 * B)]),
        (norm_src, "norm.hal",
         ["yah_half_norm.rows=%d" % B, "yah_half_norm.dim=5120",
          "yah_half_norm.eps=1e-06", "yah_half_norm.fused=0"]),
        *([(normt_src, "norm_t.hal",
            ["yah_half_norm.rows=%d" % B, "yah_half_norm.dim=5120",
             "yah_half_norm.eps=1e-06", "yah_half_norm.fused=0"])] if any_af else []),
        *([(normrt_src, "norm_rt.hal",
            ["yah_half_norm.rows=%d" % B, "yah_half_norm.dim=5120",
             "yah_half_norm.eps=1e-06", "yah_half_norm.fused=0"])] if any_af else []),
        *([(postnorm_t_src, "postnorm_t.hal", ["yah_ssm_postnorm_fp16.head_count=%d" % (48 * B)])] if af24 else []),
        # the conv with the q / k L2 norm (prep_kq) fused in
        (conv_src, "convkq.hal",
         ["yah_ssm_conv_kq.batch=%d" % B, "yah_ssm_conv_kq.qkv_dim=10240",
          "yah_ssm_conv_kq.num_key_heads=16"]),
        (prepab_src, "prepab.hal",
         ["yah_deltanet_prep_ab.batch=%d" % B,
          "yah_deltanet_prep_ab.qkv_size=10240",
          "yah_deltanet_prep_ab.num_heads=48"]),
        (dn_src, "rowsplit.hal",
         ["yah_deltanet.batch=%d" % B, "yah_deltanet.qkv_size=10240",
          "yah_deltanet.inner_size=6144", "yah_deltanet.num_key_heads=16",
          "yah_deltanet.num_heads=48"]),
        (postnorm_src, "postnorm.hal",
         ["yah_ssm_postnorm_fp16.head_count=%d" % (48 * B)]),
        *[(src, f"rowsplit_{tag}.hal",
           ["yah_deltanet.batch=%d" % B, "yah_deltanet.qkv_size=10240",
            "yah_deltanet.inner_size=6144", "yah_deltanet.num_key_heads=16",
            "yah_deltanet.num_heads=48"]) for tag, _, nh, src in dn_parts],
        *[(src, hal, ["yah_ssm_postnorm_fp16.head_count=%d" % (nh * B)]) for hal, src, nh in pn_parts],
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
        *([(qrot_src, "qrot.hal", ["yah_qrot.token_count=%d" % B])] if qrot_first < NCH else []),
        *[(src, "%s.hal" % stem if c == 0 else "%s_c%d.hal" % (stem, c),
           ["%s.cache_capacity=%d" % (sym, T), "%s.rows=%d" % (sym, (c + 1) * B)]
           + (["%s.out_capacity=%d" % (sym, T)] if sym.startswith("yah_vdeq") else [])
           )
          for c in range(NCH) if kvdeq
          for src, stem, sym in ((kdeq_src, "kdeq%d" % deq_bits, "yah_kdeq%d" % deq_bits),
                                 (vdeq_src, "vdeq%d" % deq_bits, "yah_vdeq%d" % deq_bits))],
        *[(attn_qr_src if c >= qrot_first else attn_c0_src if c == 0
           else ksn_srcs["ksn"][0] if c >= npu_from else attn64ks_src if c >= ks_from
           else attn64_src if c >= w64_from else attn_src,
           "wmma.hal" if c == 0 else "wmma_c%d.hal" % c, [
            "attention_prefill.cache_capacity=%d" % T,
            "attention_prefill.token_count=%d" % B,
            "attention_prefill.start_pos=%d" % (c * B),
            "attention_prefill.num_heads=24", "attention_prefill.num_kv_heads=4",
            "attention_prefill.head_dim=256", "attention_prefill.gqa=6"]) for c in range(NCH)],
        *[(attn_qr_t_src if c >= qrot_first else attn_t_c0_src if c == 0
           else ksn_srcs["ksn"][1] if c >= npu_from else attn64ks_t_src if c >= ks_from
           else attn64_t_src if c >= w64_from else attn_t_src,
           "wmma_t.hal" if c == 0 else "wmma_t_c%d.hal" % c, [
            "attention_prefill.cache_capacity=%d" % T,
            "attention_prefill.token_count=%d" % B,
            "attention_prefill.start_pos=%d" % (c * B),
            "attention_prefill.num_heads=24", "attention_prefill.num_kv_heads=4",
            "attention_prefill.head_dim=256", "attention_prefill.gqa=6"]) for c in range(NCH) if af24],
        *[(src_, "%s_c%d.hal" % (stem_, c), [   # (YAH_ATTN_KSPLIT: the merge kernels)
            "attention_prefill.cache_capacity=%d" % T,
            "attention_prefill.token_count=%d" % B,
            "attention_prefill.start_pos=%d" % (c * B),
            "attention_prefill.num_heads=24", "attention_prefill.num_kv_heads=4",
            "attention_prefill.head_dim=256", "attention_prefill.gqa=6"]) for c in range(ks_from, NCH)
          for src_, stem_ in ((ksn_srcs["kmn"][0] if c >= npu_from else attn64km_src, "wmmak"),)
          + (((ksn_srcs["kmn"][1] if c >= npu_from else attn64km_t_src, "wmmak_t"),) if af24 else ())],
        *[(src, hal, ["nop=0"]) for src, hal in npua],
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
           for c in range(NCH)] if paged_f16v and not vtrope else []),
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
        *([(head_src, "gemv.hal", ["nop=0"])] if head_src else [("yah_gemv_q6k_f32.loom", "gemv.hal",
          ["yah_gemv_q6k.m_rows=248320", "yah_gemv_q6k.k_blocks=20"])]),
        ("yah_argmax_f32.loom", "argmax.hal", ["yah_argmax.vocab=248320"]),
    ]
    # Loom low-level LICM (HRX patch 0011) for the attention kernels where it wins: it hoists the fragment loads'
    # per-lane LDS address math out of the key loop (fp16 -1..-2%; kv4 QROT -0.5% with one extra live register); the
    # kv4 staged-Q build loses 11% and the afrag GEMMs 5-20%, so they compile without it
    ksn_twins = [x for pair in ksn_srcs.values() for x in pair if x]
    def attn_env(loom):
        # the NPU split's merge twins: HRX main's allocator lands them at 216 VGPRs (7 waves); within 192 they fit
        # without spills at 8 waves, as on the old pin (npu8x2 16K +3.9% at 216). Not without the split-norm q (the
        # q on the NPU: the attention's own head norm spills 16 B within 192)
        if loom is not None and loom in ksn_srcs.get("kmn", ()) and QSPLIT_ON:
            return {"LOOM_EXP_LICM": "1", "LOOM_EXP_BUDGET": "amdgpu.vgpr=192"}
        if loom in (attn_qr_src, attn_qr_t_src):   # fp16 paged under kvdeq, else the kv4 direct-Q kernel
            return {"LOOM_EXP_LICM": "1"} if kvdeq else {"LOOM_EXP_LICM": "1", "LOOM_EXP_LICM_MAX_LIVE": "1"}
        if loom in (attn_src, attn_t_src, attn_c0_src, attn_t_c0_src, attn64_src, attn64_t_src,
                    attn64ks_src, attn64ks_t_src, attn64km_src, attn64km_t_src, *ksn_twins) and loom is not None and (
                not gen_attn_fa.KDEC or kv8deq):
            return {"LOOM_EXP_LICM": "1"}
        return None
    persist_rows, ks_rows = [], []
    for loom, outname, configs in fixed:
        E.emit(loom, configs, outname, outdir, attn_env(loom))
        if outname.startswith("wmma"):   # attn_guard: Loom's allocator can land a variant in its O-accumulator back-edge
            # rotation (the 8 O tuples one placement off the P.V results: ~128 loop moves, 200+ VGPRs); the head_dim
            # split kernel then comes from its twin without DIAG_SWAP (R2P has no such form and has never needed one)
            hal = os.path.join(outdir, outname)
            if (hal_vgprs(hal) > ATTN_VGPR or hal_spills(hal)) and os.path.exists(alt(loom)):
                print("attention %s: %d VGPRs, %d B spills, re-emitted without DIAG_SWAP" % (outname, hal_vgprs(hal), hal_spills(hal)))
                E.emit(alt(loom), configs, outname, outdir, attn_env(loom))
            if hal_spills(hal):
                raise SystemExit("attention %s: %d B spills: refusing to emit a spilling attention kernel"
                                 % (outname, hal_spills(hal)))
            if hal_vgprs(hal) > ATTN_VGPR:   # (3 workgroups per WGP instead of 4)
                print("attention %s: %d VGPRs, over %d" % (outname, hal_vgprs(hal), ATTN_VGPR))
        if outname.startswith("wmma") and persist:
            persist_rows.append((outname, persist64 if loom in (attn64_src, attn64_t_src, attn64ks_src, attn64ks_t_src,
                                                                attn64km_src, attn64km_t_src, *ksn_twins) else persist))
            if loom in (attn64ks_src, attn64ks_t_src):
                ks_rows.append((outname, ks_floats))
            elif loom in ksn_srcs.get("ksn", ()):
                ks_rows.append((outname, ks_floats_npu))
        n += 1
    if persist_rows:   # the persistent attention HALs launch persist workgroups on x only
        with open(os.path.join(outdir, "dispatch.txt"), "a") as fh:
            for hal, slots in persist_rows:
                fh.write("persist_%s %d 1 %d\n" % (hal, slots, pdyn))
            for hal, floats in ks_rows:
                fh.write("ksplit_%s %d 0 0\n" % (hal, floats))
            if npu_kv:
                fh.write("npuattn %d %d %d\n" % (npu_kv, npu_nbs, npu_from))

    tables = os.path.join(E.LOOM, "tables")
    for src, dst in [("grid_iq3s.bin", "grid_iq3s.bin"),
                     ("grid_iq3xxs.bin", "grid_iq3xxs.bin"),
                     ("grid_iq2xxs.bin", "grid_iq2xxs.bin"),
                     ("grid_iq2xs.bin", "grid_iq2xs.bin"),
                     ("grid_iq2s.bin", "grid_iq2s.bin"),
                     ("ksigns_iq2xs.bin", "ksigns_iq2xxs.bin")]:
        shutil.copy(os.path.join(tables, src), os.path.join(outdir, dst))
    E.wait()
    print("emitted %d GEMM + fixed prefill HALs for B=%d (tile=%d, token_tiles=%d)"
          % (n, B, TILE, TT))


if __name__ == "__main__":
    main()
