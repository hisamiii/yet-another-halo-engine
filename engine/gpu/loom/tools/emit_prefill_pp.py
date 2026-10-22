#!/usr/bin/env python3
"""Emit the prefill HAL set for B-token chunks: every kernel compiled for its shape, plus dispatch.txt.

Every token dimension is bound to B: GEMM token_tiles = B / tile (grid y), fixed kernels batch = B, residual dim = 5120 * B.
YAH_CTX=T (a multiple of B, default B) sizes the KV cache and emits one rope / attention HAL per chunk.
The driver runs T / B chunks and carries the DeltaNet and conv states. YAH_KV selects the KV format (gen_kvq.kv_bits).
dispatch.txt has one row per HAL, "<hal> <tokens per workgroup> <row groups> <token_tiles>", plus mode marker rows.
HAL names follow emit_prefill.py.

usage: emit_prefill_pp.py <model> <outdir> [tokens]   (default 2048 tokens; an HFQ MQ4 model)
"""
import dataclasses, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emit_prefill as E  # noqa: E402
import gen_attn_fa  # noqa: E402
import gen_gdn_chunk  # noqa: E402
import gen_gdn_ss  # noqa: E402
import gen_conv_kq  # noqa: E402
import gen_half_norm  # noqa: E402
import gen_decode_misc  # noqa: E402
import gen_gemv  # noqa: E402
import gen_rot  # noqa: E402
import gen_gemm_iu4  # noqa: E402
import gen_kvq  # noqa: E402
import gen_fpsparse  # noqa: E402


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

# DNVCONV: A4 dnperm sets without a NPU qkv split: the v channels' conv and alpha / beta run inside DeltaNet
# (gen_gdn_chunk vconv), the conv kernel takes only the q / k heads (gen_conv_kq.gen(4096)) and prep_ab only advances
# the conv ring, after both DeltaNet parts. Bit-identical; dispatch.txt "dnvconv <conv workgroups>".
# DNO16: DeltaNet's output (the pnq input) as f16(O 2^8) in A4 dnperm sets (gen_gdn_chunk o16): not bit-exact; vs the
# WT2 KL reference (24 chunks) mean 0.0671 / q99.9 2.026 / top-1 88.40% vs f32 0.0687 / 2.023 / 87.91%, inside the A4
# redraw spread (0.0693-0.0695 / 2.15-2.27). Unscaled f16 failed the tail (q99.9 2.52: subnormal outputs).
# DNSS: the dnperm parts as the state-stationary kernel (gen_gdn_ss: one workgroup per head, the state in registers, the
# head's v-independent work once). Grid (1, heads) x gen_gdn_ss.THREADS (dispatch.txt "dnthreads"). pp2048 (sums over
# 48 layers): part a 54.9 -> 36.1 ms; part b 27.3 -> 33.8 (8 workgroups, one per WGP: each head's serial chain, beside
# pnq a). A ninth wave runs the solve beside the row waves' K S^T / Q S^T (X = T1 (K S^T), not bit-identical): part a
# 34.5 -> 30.7 ms, part b 33.4 -> 31.0; WT2 KL 0.0671 -> 0.0683 / q99.9 2.026 -> 2.054, inside the redraw spread.


def _rewrite(text, pairs):
    for a, b in pairs:
        assert text.count(a) == 1, a
        text = text.replace(a, b)
    return text


def conv_tb_ok(B):
    """The f16 hand-off conv per (256 channels, gen_conv_kq.TB tokens) workgroup applies."""
    return 640 in O16_MT and B % gen_conv_kq.TB == 0 and gen_gdn_chunk.CONV16


def prepab_x16(text):
    """yah_deltanet_prep_ab reading the qkv GEMM's f16 output for the conv ring (O16_MT 640)."""
    return _rewrite(text, [
        ("%qkv_view = buffer.view %qkv_na[%base] : buffer -> view<[%qkv_total]xf32>",
         "%qkv_view = buffer.view %qkv_na[%base] : buffer -> view<[%qkv_total]xf16>"),
        ("%v = view.load %qkv_view[%src] : view<[%qkv_total]xf32> -> f32\n",
         "%vh = view.load %qkv_view[%src] : view<[%qkv_total]xf16> -> f16\n        %v = scalar.extf %vh : f16 to f32\n")])


RING_WGS = (10240 + 255) // 256   # the dnvconv ring update's grid (one lane per qkv channel)


def prepab_ring(text, dflag):
    """yah_deltanet_prep_ab's conv ring update alone (DNRING): it waits until dcnt[0] (DeltaNet part b's counter) reaches
    dflag, so the ring is no longer behind a graph barrier, and counts its waves in ocnt[0] (gen_rot.emit_oflag,
    ring_flag_count), which pnq_abfr waits on. The alpha / beta lanes (their ab_ output is not read in dnvconv sets) are
    gone with their bindings: qkv, state, valid, dcnt, ocnt."""
    i = text.index("  %in_count = index.cmp ult, %i, %count : index")
    j = text.index("  kernel.return", i)
    L = []
    gen_rot.emit_oflag(L.append, 1, "%c0", "%lane")
    text = text[:i] + "\n".join(L) + "\n" + text[j:]
    W = []
    W.append("  %dcnt_na, %ocnt_na = buffer.assume.noalias %dcnt, %ocnt : buffer, buffer")
    W.append("  %dn_cv = buffer.view %dcnt_na[%base] : buffer -> view<1xi32>")
    gen_rot.flag_wait(W.append, "%dn_cv", "view<1xi32>", ["%c0"], dflag, "%lane", "dn")
    return _rewrite(text, [
        ("launch(%alpha: buffer, %beta: buffer, %ssm_a: buffer, %ssm_dt: buffer, %qkv: buffer, %state: buffer, %ab: buffer, "
         "%valid: buffer) {", "launch(%qkv: buffer, %state: buffer, %valid: buffer, %dcnt: buffer, %ocnt: buffer) {"),
        ("  %alpha_na, %beta_na, %a_na, %dt_na, %qkv_na, %state_na, %ab_na = buffer.assume.noalias %alpha, %beta, %ssm_a, "
         "%ssm_dt, %qkv, %state, %ab : buffer, buffer, buffer, buffer, buffer, buffer, buffer\n",
         "  %qkv_na, %state_na = buffer.assume.noalias %qkv, %state : buffer, buffer\n"),
        ("  %alpha_view = buffer.view %alpha_na[%base] : buffer -> view<[%count]xf32>\n", ""),
        ("  %beta_view = buffer.view %beta_na[%base] : buffer -> view<[%count]xf32>\n", ""),
        ("  %a_view = buffer.view %a_na[%base] : buffer -> view<[%num_heads]xf32>\n", ""),
        ("  %dt_view = buffer.view %dt_na[%base] : buffer -> view<[%num_heads]xf32>\n", ""),
        ("  %ab_view = buffer.view %ab_na[%base] : buffer -> view<[%ab_total]xf32>\n", ""),
        ("  %i = index.add %wg_base, %lane : index\n", "  %i = index.add %wg_base, %lane : index\n" + "\n".join(W) + "\n")])


def ring_flag_count():
    """What prepab_ring adds to its counter: one per wave of its grid."""
    return RING_WGS * 256 // 32


def postnorm_g16(text):
    """yah_ssm_postnorm_fp16 reading the DeltaNet z gate as the f16 output of its GEMM (O16_MT 384)."""
    return _rewrite(text, [
        ("%gate_view = buffer.view %gate_na[%base] : buffer -> view<[%io_total]xf32>",
         "%gate_view = buffer.view %gate_na[%base] : buffer -> view<[%io_total]xf16>"),
        ("%g = view.load %gate_view[%off2] : view<[%io_total]xf32> -> f32\n",
         "%gh = view.load %gate_view[%off2] : view<[%io_total]xf16> -> f16\n      %g = scalar.extf %gh : f16 to f32\n")])


NORM_SPLIT = 2


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


def rope_kv(text, need=None):
    """rope_konly (KV16 + vtrope) rewritten wave-per-head: K workgroups (x < batch / 2) normalize, rotate and store 8
    (token, KV head) rows, one per wave; V workgroups (batch / 4 more) store one KV head of a 16-token tile as V^T,
    lane = dim, one 32 B row per lane. Same ops and sum order as the workgroup reduce (per-32 subgroup reduces, then
    the 8 partials as a cluster-8 tree), so the same bits. need: wait for the k / v GEMM counters (gen_gemm_iu4 flag:
    need each) before any load (ropew)."""
    S = "@yah_fused_qk_rope_batched"
    L = [text[:text.index("kernel.def ")]]
    e = L.append
    e(f"kernel.def target(@yah_wave32) {S}() {{")
    e("  %unit = index.constant 1 : index")
    e(f"  %batch = config.get {S}.batch : index")
    e("  %kw0 = index.constant 2 : index")
    e("  %kwg = index.div %batch, %kw0 : index")
    e("  %vw0 = index.constant 4 : index")
    e("  %vwg = index.div %batch, %vw0 : index")
    e("  %nwg = index.add %kwg, %vwg : index")
    e("  %c256 = index.constant 256 : index")
    e("  kernel.launch.config workgroups(%nwg, %unit, %unit) workgroup_size(%c256, %unit, %unit) : index")
    e("} launch(%q: buffer, %k: buffer, %v: buffer, %q_weight: buffer, %k_weight: buffer, %q_out: buffer, %k_out: buffer, "
      "%k_cache: buffer, %v_cache: buffer, %k_cache_f16: buffer, %v_cache_f16: buffer, %eps_buf: buffer, %ptab: buffer"
      + (", %kcnt: buffer, %vcnt: buffer" if need else "") + ") {")
    e("  %base = index.constant 0 : offset")
    for v in (0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 255, 256, 1024, 4096):
        e(f"  %c{v} = index.constant {v} : index")
    e("  %two = scalar.constant 2.0 : f32")
    e("  %one = scalar.constant 1.0 : f32")
    for n in ("start_pos", "batch", "max_context", "kv_elems", "k16_elems"):
        e(f"  %{n} = config.get {S}.{n} : index")
    e(f"  %theta = config.get {S}.theta : f32")
    e("  %head_dimf = scalar.constant 256.0 : f32")
    e("  %rotary_dimf = scalar.constant 64.0 : f32")
    e("  %k_na, %v_na, %kw_na, %k16_na, %v16_na, %eps_na, %pt_na = buffer.assume.noalias %k, %v, %k_weight, "
      "%k_cache_f16, %v_cache_f16, %eps_buf, %ptab : buffer, buffer, buffer, buffer, buffer, buffer, buffer")
    e("  %k_view = buffer.view %k_na[%base] : buffer -> view<[%kv_elems]xf16>")
    e("  %v_view = buffer.view %v_na[%base] : buffer -> view<[%kv_elems]xf16>")
    e("  %kw_view = buffer.view %kw_na[%base] : buffer -> view<[%c256]xf32>")
    e("  %k16_view = buffer.view %k16_na[%base] : buffer -> view<[%k16_elems]xf16>")
    e("  %v16_view = buffer.view %v16_na[%base] : buffer -> view<[%k16_elems]xf16>")
    e("  %eps_view = buffer.view %eps_na[%base] : buffer -> view<[%c1]xf32>")
    e("  %pgrows = index.div %k16_elems, %c1024 : index")
    e("  %pgn0 = index.add %pgrows, %c255 : index")
    e("  %pgnp = index.div %pgn0, %c256 : index")
    e("  %pglp1 = index.sub %pgnp, %c1 : index")
    e("  %pt_view = buffer.view %pt_na[%base] : buffer -> view<[%pgnp]xi32>")
    e("  %ctx_m1 = index.sub %max_context, %c1 : index")
    e("  %kwg = index.div %batch, %c2 : index")
    e("  %wg = kernel.workgroup.id<x> : index")
    e("  %tid = kernel.workitem.id<x> : index")
    e("  %lane = index.rem %tid, %c32 : index")
    e("  %wave = index.div %tid, %c32 : index")
    e("  %is_kw = index.cmp ult, %wg, %kwg : index")
    e("  %vg0 = index.sub %wg, %kwg : index")
    e("  %vg = index.max %vg0, %c0 : index")
    # the workgroup's first chunk-local token: K 2 per workgroup (waves 0-3, 4-7), V 16 (4 KV heads per tile)
    e("  %kt0 = index.mul %wg, %c2 : index")
    e("  %vtl = index.div %vg, %c4 : index")
    e("  %vt0 = index.mul %vtl, %c16 : index")
    e("  %bt = scf.select %is_kw, %kt0, %vt0 : index")
    if need:
        e("  %rw_tt0 = index.add %batch, %c128 : index")
        e("  %rw_tt1 = index.sub %rw_tt0, %c1 : index")
        e("  %rw_tt = index.div %rw_tt1, %c128 : index")
        e("  %rw_t = index.div %bt, %c128 : index")
        e("  %rw_kna, %rw_vna = buffer.assume.noalias %kcnt, %vcnt : buffer, buffer")
        e("  %rw_kv = buffer.view %rw_kna[%base] : buffer -> view<[%rw_tt]xi32>")
        e("  %rw_vv = buffer.view %rw_vna[%base] : buffer -> view<[%rw_tt]xi32>")
        e("  %rw_w0 = index.cmp ult, %tid, %c32 : index")
        e(f"  %rw_need = scalar.constant {need} : i32")
        e(f"  %rw_lim = scalar.constant {gen_rot.FLAG_POLLS} : i32")
        e("  %rw_z = scalar.constant 0 : i32")
        e("  %rw_one = scalar.constant 1 : i32")
        e("  scf.if %rw_w0 {")
        e("    %rw_n = scf.while(%rw_i = %rw_z : i32) -> (i32) {")
        e("      %rw_x = view.atomic.load %rw_kv[%rw_t] {ordering = relaxed, scope = device} : view<[%rw_tt]xi32> -> i32")
        e("      %rw_y = view.atomic.load %rw_vv[%rw_t] {ordering = relaxed, scope = device} : view<[%rw_tt]xi32> -> i32")
        e("      %rw_m = scf.select %is_kw, %rw_x, %rw_y : i32")
        e("      %rw_e = scalar.cmpi slt, %rw_m, %rw_need : i32")
        e("      %rw_l = scalar.cmpi ult, %rw_i, %rw_lim : i32")
        e("      %rw_g = scalar.andi %rw_e, %rw_l : i1")
        e("      scf.condition %rw_g, %rw_i : i1, i32")
        e("    } do(%rw_j: i32) {")
        e("      %rw_j1 = scalar.addi %rw_j, %rw_one : i32")
        e("      scf.yield %rw_j1 : i32")
        e("    }")
        e("  }")
        e("  kernel.barrier<global> scope(workgroup) ordering(acquire)")
    e("  scf.if %is_kw {")
    e("    %kh = index.rem %wave, %c4 : index")
    e("    %kwt = index.div %wave, %c4 : index")
    e("    %bi = index.add %kt0, %kwt : index")
    e("    %eps = view.load %eps_view[%c0] : view<[%c1]xf32> -> f32")
    e("    %cur_raw = index.add %start_pos, %bi : index")
    e("    %cur = index.min %cur_raw, %ctx_m1 : index")
    e("    %pglp = index.div %cur, %c256 : index")
    e("    %pglpc = index.min %pglp, %pglp1 : index")
    e("    %pgv = view.load %pt_view[%pglpc] : view<[%pgnp]xi32> -> i32")
    e("    %pgu = index.cast %pgv : i32 to index")
    e("    %pgz = index.max %pgu, %c0 : index")
    e("    %pg = index.min %pgz, %pglp1 : index")
    e("    %pgb = index.mul %pg, %c256 : index")
    e("    %pgo = index.rem %cur, %c256 : index")
    e("    %krow = index.add %pgb, %pgo : index")
    e("    %cur_i = index.cast %cur : index to i32")
    e("    %curf = scalar.sitofp %cur_i : i32 to f32")
    e("    %bkw = index.mul %bi, %c1024 : index")
    e("    %hbkq = index.mul %kh, %c256 : index")
    e("    %hbk = index.add %bkw, %hbkq : index")
    e("    %src = index.add %hbk, %lane : index")
    for j in range(8):
        e(f"    %xo{j} = index.add %src, %c{32 * j} : index")
        e(f"    %x{j}h = view.load %k_view[%xo{j}] : view<[%kv_elems]xf16> -> f16")
        e(f"    %x{j} = scalar.extf %x{j}h : f16 to f32")
        e(f"    %sq{j} = scalar.mulf %x{j}, %x{j} : f32")
        e(f"    %ps{j} = kernel.subgroup.reduce<addf> %sq{j} : f32")
    e("    %l8 = index.rem %lane, %c8 : index")
    for j in range(1, 8):
        e(f"    %lj{j}c = index.constant {j} : index")
        e(f"    %lj{j} = index.cmp eq, %l8, %lj{j}c : index")
        e(f"    %pv{j} = scf.select %lj{j}, %ps{j}, {'%ps0' if j == 1 else '%%pv%d' % (j - 1)} : f32")
    e("    %total_k = kernel.subgroup.reduce<addf> %pv7 {cluster_size = 8} : f32")
    e("    %mean_k = scalar.divf %total_k, %head_dimf : f32")
    e("    %me_k = scalar.addf %mean_k, %eps : f32")
    e("    %inv_k = scalar.rsqrtf %me_k : f32")
    e("    %k16_pre = index.mul %krow, %c1024 : index")
    e("    %k16_raw = index.add %k16_pre, %hbkq : index")
    e("    %k16_max = index.sub %k16_elems, %c256 : index")
    e("    %k16_base = index.min %k16_raw, %k16_max : index")
    e("    %p3b = index.add %lane, %c32 : index")
    e("    %kw0 = view.load %kw_view[%lane] : view<[%c256]xf32> -> f32")
    e("    %kw1 = view.load %kw_view[%p3b] : view<[%c256]xf32> -> f32")
    e("    %kn0 = scalar.mulf %x0, %inv_k : f32")
    e("    %kn0w = scalar.mulf %kn0, %kw0 : f32")
    e("    %kn1 = scalar.mulf %x1, %inv_k : f32")
    e("    %kn1w = scalar.mulf %kn1, %kw1 : f32")
    e("    %p3i = index.cast %lane : index to i32")
    e("    %p3if = scalar.sitofp %p3i : i32 to f32")
    e("    %twop3 = scalar.mulf %two, %p3if : f32")
    e("    %expo3 = scalar.divf %twop3, %rotary_dimf : f32")
    e("    %invfreq3 = scalar.powf<afn> %theta, %expo3 : f32")
    e("    %inv3 = scalar.divf %one, %invfreq3 : f32")
    e("    %angle3 = scalar.mulf %curf, %inv3 : f32")
    e("    %cv3 = scalar.cosf<afn> %angle3 : f32")
    e("    %sv3 = scalar.sinf<afn> %angle3 : f32")
    e("    %ka = scalar.mulf %kn0w, %cv3 : f32")
    e("    %kb = scalar.mulf %kn1w, %sv3 : f32")
    e("    %kr0 = scalar.subf %ka, %kb : f32")
    e("    %kc2 = scalar.mulf %kn0w, %sv3 : f32")
    e("    %kd = scalar.mulf %kn1w, %cv3 : f32")
    e("    %kr1 = scalar.addf %kc2, %kd : f32")
    e("    %ko0 = index.add %k16_base, %lane : index")
    e("    %ko1 = index.add %k16_base, %p3b : index")
    e("    %kr0h = scalar.fptrunc %kr0 : f32 to f16")
    e("    %kr1h = scalar.fptrunc %kr1 : f32 to f16")
    e("    view.store %kr0h, %k16_view[%ko0] : f16, view<[%k16_elems]xf16>")
    e("    view.store %kr1h, %k16_view[%ko1] : f16, view<[%k16_elems]xf16>")
    for j in range(2, 8):
        e(f"    %i3_{j} = index.add %lane, %c{32 * j} : index")
        e(f"    %w3_{j} = view.load %kw_view[%i3_{j}] : view<[%c256]xf32> -> f32")
        e(f"    %n3_{j} = scalar.mulf %x{j}, %inv_k : f32")
        e(f"    %n3w{j} = scalar.mulf %n3_{j}, %w3_{j} : f32")
        e(f"    %ko3_{j} = index.add %k16_base, %i3_{j} : index")
        e(f"    %n3h{j} = scalar.fptrunc %n3w{j} : f32 to f16")
        e(f"    view.store %n3h{j}, %k16_view[%ko3_{j}] : f16, view<[%k16_elems]xf16>")
    e("  } else {")
    # V^T tile: [kv head][pool tiles][256 dims][16 keys], pool tile = ptab page * 16 + tile in page
    e("    %vh = index.rem %vg, %c4 : index")
    e("    %cur0r = index.add %start_pos, %vt0 : index")
    e("    %cur0 = index.min %cur0r, %ctx_m1 : index")
    e("    %vpglp = index.div %cur0, %c256 : index")
    e("    %vpglpc = index.min %vpglp, %pglp1 : index")
    e("    %vpgv = view.load %pt_view[%vpglpc] : view<[%pgnp]xi32> -> i32")
    e("    %vpgu = index.cast %vpgv : i32 to index")
    e("    %vpgz = index.max %vpgu, %c0 : index")
    e("    %vpg = index.min %vpgz, %pglp1 : index")
    e("    %vpgo = index.rem %cur0, %c256 : index")
    e("    %vt_pt0 = index.mul %vpg, %c16 : index")
    e("    %vt_lt = index.div %vpgo, %c16 : index")
    e("    %vt_ptile = index.add %vt_pt0, %vt_lt : index")
    e("    %vt_kt = index.mul %pgnp, %c16 : index")
    e("    %vt_a = index.mul %vh, %vt_kt : index")
    e("    %vt_b = index.add %vt_a, %vt_ptile : index")
    e("    %vt_b0 = index.mul %vt_b, %c4096 : index")
    e("    %vt_o0 = index.mul %tid, %c16 : index")
    e("    %vt_o = index.add %vt_b0, %vt_o0 : index")
    e("    %vt_m = index.sub %k16_elems, %c16 : index")
    e("    %vt_d = index.min %vt_o, %vt_m : index")
    e("    %vrow0 = index.mul %vt0, %c1024 : index")
    e("    %vhd = index.mul %vh, %c256 : index")
    e("    %vs0 = index.add %vrow0, %vhd : index")
    e("    %vs = index.add %vs0, %tid : index")
    for kk in range(16):
        e(f"    %vk{kk}c = index.constant {kk * 1024} : index")
        e(f"    %vk{kk}o = index.add %vs, %vk{kk}c : index")
        e(f"    %vk{kk} = view.load %v_view[%vk{kk}o] : view<[%kv_elems]xf16> -> f16")
    for h in range(2):
        e(f"    %vq{h} = vector.from_elements {', '.join('%%vk%d' % (8 * h + i) for i in range(8))} : vector<8xf16>")
    e("    %vt_d8 = index.add %vt_d, %c8 : index")
    e("    vector.store %vq0, %v16_view[%vt_d] : vector<8xf16>, view<[%k16_elems]xf16>")
    e("    vector.store %vq1, %v16_view[%vt_d8] : vector<8xf16>, view<[%k16_elems]xf16>")
    e("  }")
    e("  kernel.return")
    e("}")
    return "\n".join(L) + "\n"


def rope_kv16(text):
    """yah_fused_qk_rope_batched reading f16 K / V (the GEMMs' f16 outputs, KV16): each load widened to f32"""
    for b in ("k", "v"):
        a = f"%{b}_view = buffer.view %{b}_na[%base] : buffer -> view<[%kv_elems]xf32>"
        assert text.count(a) == 1
        text = text.replace(a, a.replace("xf32>", "xf16>"))
    loads = re.findall(r"(%\w+) = view\.load %([kv])_view\[(%\w+)\] : view<\[%kv_elems\]xf32> -> f32", text)
    assert len(loads) >= 4, loads
    for v, b, i in loads:
        text = text.replace(f"{v} = view.load %{b}_view[{i}] : view<[%kv_elems]xf32> -> f32",
                            f"{v}_x16 = view.load %{b}_view[{i}] : view<[%kv_elems]xf16> -> f16\n"
                            f"      {v} = scalar.extf {v}_x16 : f16 to f32")
    return text


def gemm(fmt, mt, kb, B, out, outdir, kind="kstore"):
    """Emit the tile GEMM (tools/gen_gemm_tile.py) of this shape and kind; return its dispatch.txt row, or None."""
    r = tile_kstore(fmt, mt, kb, B, out, outdir, kind)
    if not r and mt < 4:
        # the 48-row ssm_alpha/ssm_beta: one 16-row tile per workgroup, 64 tokens over 2 waves, grid (3, B/64)
        r = tile_kstore(fmt, mt, kb, B, out, outdir, kind, geom=(16, 64, 1, 2))
    return r


OCC3_FROM = 86016   # first context position whose chunk's attention runs 3 workgroups per WGP (see emit: write_attn)
OCC3_LDS = 33792    # the LDS floor that caps a WGP at 3 attention workgroups (128 KB / 4 < 33 KB)


# From KT2_FROM keys of context on (fp16 attention), a chunk runs the 32-key-tile role-split kernel (gen_attn_fa KT2):
# 3 workgroups per WGP, one barrier pair / reduction / rescale vote per 32 keys; -0.8% at 16K, -1.8% at 32K, -3..-5.6% at
# 88K..126K vs the 4-workgroup kernel / the deep twin (standalone); not bit-identical (the running max moves per 32 keys).
KT2_FROM = 16384
# Block-sparse prefill attention (gen_fpsparse, FlashPrefill V2) from YAH_FPS_FROM tokens of context on (0: off), kv4 paged
# sets with the KT2 twin only (its chunks take the sparse KT2 twin whatever KT2_FROM is). 32K, alpha 0.1: from 4096 vs
# 16384 -2.5% cycles at the same quality (KL / needles / answer tile within the redraw band).
# The f16 pools grow by the pooled-row pages (every attention HAL and vdeq use the extended capacity).
# dispatch.txt "fpsparse <first chunk> <extension pages> <list stride>"
FPS_FROM = int(os.environ.get("YAH_FPS_FROM", "0"))
FPS_ORD = True   # yah_fpord launch order (longest tile list first)
QNORM = True   # gen_attn_fa QNORM: the attention does the Q head norm + RoPE (RoPE: K / V only)
# QG16 (A4 with QNORM): the kqg GEMM stores q and gate as f16, the attention reads them so. dispatch.txt "qg16"
QG16 = True
QG16_ON = False   # set by main
# KV16 (A4): the K / V GEMMs (kstore, 1024 rows) store f16, RoPE widens them (rope_kv16). dispatch.txt "kv16in"
KV16_ON = False   # set by main
# NQOV (A4): every kres also as iu4_kresf_* (gen_gemm_iu4 flag: per-token-tile completion counters) and normq as
# normqf_5120 (gen_rot.gen_nq flag: rows wait for their token tile), so the driver records the norm after its kres
# with no graph barrier and the norm runs in the GEMM's tail. Same values. dispatch.txt "nqov"
NQOV_ON = False   # set by main
# DNOV (A4 dnvconv, with DNSPLIT (40, 8)): the qkv and gate kstores also flagged (iu4_kstoref_*, iu4_kstorehf_*), the
# conv as convkqf and pnq part a as pnq_af waiting on their counters, so the conv runs in the qkv GEMM's tail and pnq
# part a in the gate GEMM's. Same values. dispatch.txt "dnov"
DNOV_ON = False   # set by main
# CONVP (with DNOV): the flagged conv as gen_conv_kq.gen_p (persistent workgroups walking (key head, token tile)
# segments, convkqp.hal) instead of gen's grid of 16-token workgroups: conv exposure 5.3-5.5 -> 4.2 ms per pp2048
# (64-token segments over 512 workgroups: -> ~1.9 ms). Same values
# GUOV (A4): the gu GEMM also as iu4_guf_* (gen_gemm_iu4 flag: a counter per 1024 features and token tile) and
# rotq_17408 as rotqf_17408 (gen_rot.gen_q flag), so the down projection's rotq runs in the gu GEMM's tail. Same values.
# dispatch.txt "guov"
GUOV_ON = False   # set by main
# NQGU (A4, with NQOV and GUOV): normq also as normqo_5120 / normqfo_5120 (gen_rot.gen_nq oflag: per-token-tile counters)
# and the flagged gu as iu4_gufw_* (gen_gemm_iu4 wait), so the driver records the gu GEMM after its norm with no graph
# barrier and its first workgroups start in the norm's tail. Same values. dispatch.txt "nqgu"
NQGU_ON = False   # set by main
# RQK (A4, with NQOV): the residual rotqs also as rotqfo_17408 / rotqo_6144 (gen_rot.gen_q oflag) and the flagged kres as
# iu4_kresfw_* (gen_gemm_iu4 wait), so the driver records the kres after its rotq with no graph barrier. Same values.
# dispatch.txt "rqk"
RQK_ON = False   # set by main
# RQP (with RQK and GUOV): the down projection's rotq also as rotqp_17408 (gen_rot.gen_qp: RQP persistent workgroups
# walking the token tiles in order), so the kres after it is not queued behind 34816 rotq workgroups and starts on the
# first token tiles in the gu's tail: rotq exposure 21.3 -> 0.3 ms, gu start .. kres end 1.546 -> 1.474 gu, pp2048 ~-1.9%
# timeline span. Same values. (P 96 / 240 lose; recorded before the gu, its LDS takes a gu workgroup per WGP: gu +44%)
RQP = 160
# RQOP (with RQK): the attention output's rotq also as rotqop_6144 (gen_rot.gen_qp, no producer counters: a graph
# barrier orders it after the attention), so its kres is not queued behind its 12288 workgroups
RQOP = 160
# AOV (with RQOP): chunk 0's attention also as wmmaf.hal (gen_attn_fa OCNT: per (1024 features, token tile) counters)
# and the attention-out rotq as rotqopf_6144 (gen_qp waiting on them), so no graph barrier separates the two: the rotq
# runs in the attention's tail (polling at priority 0 beside it): attention start .. kres end -131 us per attention
# layer. Same values. dispatch.txt "aov"
# ROPEW (with NQW and QNORM): the attention's k / v GEMMs also as iu4_kstorefw_* (per-token-tile counters) and RoPE as
# ropew[_c<i>].hal (rope_kv need: a workgroup polls its token tile's k and v counters), so no graph barrier separates
# them: RoPE runs in the v GEMM's tail. Same values. dispatch.txt "ropew"
# ROPEKV (QNORM, KV16, vtrope): RoPE as rope_kv (wave-per-head K, coalesced V^T tiles), grid (3 * batch / 4).
# Same values. dispatch.txt "ropekv"
# NQP: the flag-counting normq (normq[f]o: its GEMM waits on the counters) as NQP persistent workgroups (gen_nq persist),
# so the waiting GEMM is dispatched while the rows run. Same values. dispatch.txt "nqp <NQP>"
NQP = 64
ROPEW_ON = False   # set by main
# PNQP (with DNRING): the chained pnq also as pnq_abfrp (gen_rot.gen_pnqp: PNQP persistent workgroups walking the token
# tiles in order), for the same reason: ssm_out's kres was queued behind its 12288 workgroups. 155 VGPRs: 6 workgroups
# per WGP beside two kres workgroups
PNQP = 120
# NQW (A4): the attention and DeltaNet input GEMMs (kqg, kstore K / V / alpha / beta, the flagged qkv kstore) also as
# iu4_<kind>[f]w_* (gen_gemm_iu4 wait) on the attention norm's counters (normq[f]o_5120), so the driver records them after
# the norm with no graph barrier. Same values. dispatch.txt "nqw"
NQW_ON = False   # set by main
# DNRING (A4, with DNOV's pnq_abf, RQK, NQW): the conv ring update as prepabr (prepab_ring: waits on DeltaNet part b's
# counter, counts) before the pnq, pnq_abfr waiting on it and counting rows for ssm_out's kres (iu4_kresfw_*), so a
# DeltaNet layer's tail and the next layer's input GEMMs need no graph barrier. Same values. dispatch.txt "dnring"
DNRING_ON = False   # set by main
# QNSTREAM + QNMIX: the QNORM prologue one fragment at a time (its 255-VGPR peak had pushed both role-split twins past 240,
# so KT2 fell back and the deep twin ran 2 workgroups per WGP; 96K deep attention -13%); MAXNN + SUM0 (KT2 -1.35%)
QN_FLAGS = dict(QNSTREAM=True, QNMIX=True, MAXNN=True, SUM0=True)
KT2_FLAGS = dict(PAIR=True, PAIR_LATE=False, QKF=1, PVF=2, KT2=True, KT2_IL=1, **QN_FLAGS)
KT2_VGPR = 240   # 3 workgroups per WGP: 6 waves x 240 (VGPRs allocate in granules of 24 on gfx1151)
# The KT2 twin's workgroup: 6 heads x 16 tokens (a whole GQA group, 12 waves, 2 workgroups per WGP) stages each K / V tile
# once per 96 query rows instead of 64; the LDS that frees holds a second P^T / alpha slot, so B loads the previous
# tile's P^T at its arm start instead of carrying it (KT2_P2: -16 loop-live VGPRs). Bit-identical, attention -3.1..-3.6%
# at 16K..96K. Its chunks launch from their own dispatch.txt rows (wmma[_t]_c<i>.hal).
# KT2_RSALL (O x alpha every tile: the skip branch's join cost 128 v_movs), KT2_MIRROR (odd workgroups mirror the roles
# over the SIMDs) and KADDR_SR (K load address = uniform page part + loop-invariant lane part): bit-identical, -1.5..-2%.
KT2_SHAPE = (6, 16)
KT2_SHAPE_FLAGS = dict(KT2_P2=True, MAPS_EARLY=True, KT2_RSALL=True, KT2_MIRROR=True, KADDR_SR=True)


def deep(path):
    return path.replace(".loom", "_deep.loom")


def kt2(path):
    return path.replace(".loom", "_kt2.loom")


def alt(path):
    return path.replace(".loom", "_alt.loom")


def fps(path):
    return path.replace(".loom", "_kt2fps.loom")


ATTN_VGPR = 192   # every attention variant fits here unless the allocator rotates the O accumulators


def hal_vgprs(hal):
    notes = subprocess.run(["llvm-readelf", "--notes", hal], capture_output=True, text=True).stdout
    m = re.search(r"\.vgpr_count:\s*(\d+)", notes)
    return int(m.group(1)) if m else 0


def hal_spills(hal):
    notes = subprocess.run(["llvm-readelf", "--notes", hal], capture_output=True, text=True).stdout
    m = re.search(r"\.private_segment_fixed_size:\s*(\d+)", notes)
    return int(m.group(1)) if m else 0


QROT_FROM = 12288   # kv4: the first context position whose chunk rotates Q in its own kernel (see the attention)


# Narrower token tiles for chunks with few real tokens; the driver picks one per chunk (LoomPrefill::PickGemm).
# A narrow tile pads less but costs more per token row (x1.12 at 128, x1.6 at 64, measured): an 18-token prompt
# 595 -> 404 ms with 64, a 300-token prompt 909 -> 802 ms with 128. Same values as the default tile.
NARROW = ((128, 2), (64, 2))



def narrow_variants(fmt, mt, kb, B, out, outdir, kind):
    """Emit "<hal>.t<BN>.hal" for each NARROW token tile of this GEMM."""
    import gen_gemm_tile as TG
    if mt < 4:
        return []
    full, rows = TG.default_tile(fmt, kind, kb), []
    for bn, wn in NARROW:
        t = dataclasses.replace(full, bn=bn, wn=wn)
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
    return rows


def tile_kstore(fmt, mt, kb, B, out, outdir, kind, geom=None):
    """Emit the tile GEMM (tools/gen_gemm_tile.py) for this shape if it covers it; return its dispatch.txt row, else None.
    geom=(BM, BN, WM, WN) overrides the workgroup geometry for this one kernel."""
    import gen_gemm_tile as TG
    t = TG.default_tile(fmt, kind, kb, geom)
    if mt % t.rowgrp:
        return None
    # a token tile that does not divide the chunk: the last tile is masked (gen_gemm_tile.gen masked=)
    masked = B % t.bn != 0
    return _emit_gen(lambda f, k: TG.gen(f, k, t, masked), t.bn, fmt, mt, kb, B, out, outdir, kind, t.rowgrp, masked)


def _emit_gen(gen, tile, fmt, mt, kb, B, out, outdir, kind, rowgrp, masked=False):
    if B % tile and not masked:
        return None
    import gen_gemm_tile as _TG
    _TG.OUT16 = kind == "kstore" and mt in O16_MT
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


def iu4_gemm(kind, mt, kb, B, outdir, hperm=None, flag=False, wait=0):
    """Emit the A4 GEMM (tools/gen_gemm_iu4.py) of an MQ4 tensor; return its dispatch.txt row. hperm: a kstore whose
    output heads go to HF order (gen_gemm_iu4.gen), as "kstoreh". flag: the completion-counter variant, "<kind>f"
    (NQOV, DNOV); wait: the flag-waiting variant (gen_gemm_iu4 wait), "w" after that (NQGU)."""
    bm, bn, _, _ = gen_gemm_iu4.tile(mt * 16)
    if kind == "gu":
        bm //= 2   # a gu workgroup stages bm rows of gate and up for bm / 2 features
    if B % bn or (mt * 16) % bm:
        raise SystemExit("iu4 GEMM %s: %d rows / %d tokens do not tile" % (kind, mt * 16, B))
    tt = B // bn
    sym = gen_gemm_iu4.sym(kind)
    out = "iu4_%s_%d_%d.hal" % (kind + ("h" if hperm else "") + ("f" if flag else "") + ("w" if wait else ""), mt, kb)
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(tmp, out[:-4] + ".loom")
    out16 = (kind == "kstore" and (mt in O16_MT or (mt == 64 and KV16_ON))) or (kind == "kqg" and QG16_ON)
    open(src, "w").write(gen_gemm_iu4.gen(kind, mt * 16, out16, hperm, flag, wait))
    gate = subprocess.run([sys.executable, os.path.join(HERE, "footprint_gate.py"), src, sym, "mq4", kind, str(mt), str(kb),
                           str(tt), str(B), "aspan=%d" % gen_gemm_iu4.KP], capture_output=True, text=True)
    if gate.returncode != 0:
        raise SystemExit("footprint gate refused %s: %s" % (out, (gate.stdout + gate.stderr).strip()[-400:]))
    E.emit(src, ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb), "%s.token_tiles=%d" % (sym, tt)], out, outdir)
    return (out, bn, bm // 16, tt)


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
    """HAL kinds emitted for a combo: the attention q projection (768 row tiles) as kqg, the residual ones as kres."""
    return {"kstore": ["kqg" if mt == 768 else "kstore"], "residual": ["kres"], "swiglu": ["swiglu"]}[kind]


def main():
    global QG16_ON, KV16_ON, NQOV_ON, DNOV_ON, GUOV_ON, NQGU_ON, RQK_ON, NQW_ON, DNRING_ON, ROPEW_ON
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
    combos = gemm_combos(rows)
    types = {nm: ty for nm, _, ty in rows}
    # MQ4 (HFQ) models: A4 prefill (int4 activations, iu4 GEMMs) unless YAH_A16=1 (W4A16: f16 activations, the quality mode)
    if types.get("blk.0.ffn_down.weight") != 44 or types.get("output.weight") != 44 or types["token_embd.weight"] != 8:
        raise SystemExit("emit_prefill_pp: not an MQ4 model (the GGUF model path is gone since tag pre-cleanup-2026-10-20)")
    a4 = os.environ.get("YAH_A16") != "1"
    KV16_ON = a4
    NQOV_ON = a4
    GUOV_ON = a4
    NQGU_ON = GUOV_ON
    RQK_ON = (NQOV_ON and GUOV_ON)
    NQW_ON = a4
    # the engine runs ropew only with fp16 paged K and V (RoPE writes both pools; quantized KV has its own staging)
    ROPEW_ON = (NQW_ON and QNORM and (B % gen_rot.FLAG_TT == 0) and (gen_kvq.kv_bits() == (16, 16)) and (T % 256 == 0))
    DNOV_ON = (a4 and (DNSPLIT == (40, 8)) and conv_tb_ok(B))   # the dnvconv conv (dn_vconv below)
    QG16_ON = QG16 and a4 and QNORM and gen_attn_fa.Q16 and gen_attn_fa.QDIRECT is not False

    n = 0
    geom = []  # (<hal>, <tokens per workgroup>, <row groups>, <token_tiles>)
    for kind, fmt, port, mt, kb in sorted(combos):
        name = lambda k: "gemm_%s_%s_%d_%d.hal" % (k, fmt, mt, kb)
        # A residual projection gets a kstore and the fused-residual kres; the attention q projection (12288 rows =
        # 24 heads x [q | gate]) also gets kqg, which stores q and gate unpacked. The driver prefers kres and kqg.
        kinds = gemm_kinds(kind, mt)
        if a4:
            # A4: the FFN's gate and up run as one gu GEMM (on the up combo); the gate tensors need no kstore
            if all(t.endswith("ffn_gate.weight") for t in combos[(kind, fmt, port, mt, kb)]):
                continue
            for k in (["gu"] if kind == "swiglu" else kinds):
                # dnperm sets (DNSPLIT (40, 8)): the DeltaNet gate (attn_gate) also as kstoreh, in HF head order
                # for the pnq kernels
                gate = k == "kstore" and any(t.endswith("attn_gate.weight") for t in combos[(kind, fmt, port, mt, kb)])
                for hp in [None] + ([gen_rot.PN_HEADS[:2]] if gate and DNSPLIT == (40, 8) else []):
                    geom.append(iu4_gemm(k, mt, kb, B, outdir, hp))
                    if gen_gemm_iu4.group_of(k, mt * 16):   # grouped grid: the engine launches (G, tiles, row tiles / G)
                        geom.append(("iu4grp_" + geom[-1][0], gen_gemm_iu4.group_of(k, mt * 16), 0, 0))
                    # the flagged variants: the plain GEMM's geometry (dispatch.txt row) serves both
                    if k == "kres" and NQOV_ON and mt * 16 == 5120:
                        iu4_gemm(k, mt, kb, B, outdir, flag=True)
                    if k == "kres" and RQK_ON and mt * 16 == 5120 and kb * 256 in (6144, 17408):
                        iu4_gemm(k, mt, kb, B, outdir, flag=True, wait=gen_rot.q_flag_count(kb * 256))
                    if k == "gu" and GUOV_ON:
                        iu4_gemm(k, mt, kb, B, outdir, flag=True)
                    if k == "gu" and GUOV_ON and NQGU_ON:
                        iu4_gemm(k, mt, kb, B, outdir, flag=True, wait=gen_rot.nq_flag_count(5120))
                    if k == "kstore" and DNOV_ON and (hp is not None or any(
                            t.endswith("attn_qkv.weight") for t in combos[(kind, fmt, port, mt, kb)])):
                        iu4_gemm(k, mt, kb, B, outdir, hp, flag=True)
                    # the attention norm's readers (kb 20: K = 5120)
                    tn = combos[(kind, fmt, port, mt, kb)]
                    if NQW_ON and hp is None and kb == 20 and (k == "kqg" or k == "kstore" and any(
                            t.endswith(x) for t in tn for x in ("attn_k.weight", "attn_v.weight", "ssm_alpha.weight",
                                                                "ssm_beta.weight", "attn_qkv.weight"))):
                        qkv = any(t.endswith("attn_qkv.weight") for t in tn)
                        kv = ROPEW_ON and k == "kstore" and any(t.endswith(("attn_k.weight", "attn_v.weight")) for t in tn)
                        iu4_gemm(k, mt, kb, B, outdir, flag=qkv and DNOV_ON, wait=gen_rot.nq_flag_count(5120))
                        if kv:   # ropew: also flagged (the plain one serves chunks without ropew)
                            iu4_gemm(k, mt, kb, B, outdir, flag=True, wait=gen_rot.nq_flag_count(5120))
            n += 1
            continue
        for k in kinds:
            r = gemm(fmt, mt, kb, B, name(k), outdir, kind=k)
            if r:
                geom.append(r)
                if T > B:  # chunked sets: their last chunk may be short; a one-pass set always runs all B tokens
                    geom.extend(narrow_variants(fmt, mt, kb, B, name(k), outdir, k))
            else:
                raise SystemExit("no GEMM kernel for %s %s (%d rows, K = %d)" % (fmt, k, mt * 16, kb * 256))
        n += 1

    # Attention: tools/gen_attn_fa.py, 32 tokens x 2 heads per workgroup; reads V^T (vtrans.hal, or the paged / quantized pools).
    # KV paging (256-token pages) needs a context that is a multiple of 256; otherwise the caches stay contiguous.
    kv_paged = T % 256 == 0
    if not kv_paged:
        print("KV paging off: needs the context to be a multiple of 256")
    gen_attn_fa.PAGED = gen_kvq.PAGED = kv_paged
    gen_attn_fa.MAX_TOKENS = max(B, 2048)
    # the attention workgroup: 2 heads x 32 tokens (the KT2 twin takes 6 x 16, KT2_SHAPE)
    a_hpw, a_qt = 2, 32
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
    # From OCC3_FROM keys of context on, a chunk's attention runs 3 workgroups per WGP (an LDS floor of 33 KB) instead of
    # 4: past ~84K the 4-workgroup kernel falls off a cache cliff (73% -> 60% of the WMMA floor at 126K; 3 workgroups
    # hold ~70% at every depth, but lose ~5% up to 80K). Each attention source has a "_deep" twin for those chunks.
    # Each source also gets a "_alt" twin without DIAG_SWAP: Loom's allocator can land a variant in its O-accumulator
    # back-edge rotation (the 8 O tuples one placement off the P.V results: ~128 loop moves, 200+ VGPRs); the emit
    # loop re-emits such a HAL from the twin (attn_guard).
    # The deep twin is also the role-split kernel (gen_attn_fa PAIR: one wave of each pair runs QK + the softmax once,
    # the other P.V; bit-identical): at 3 workgroups per WGP it needs no more than the 256 VGPRs that allows.
    DEEP_FLAGS = dict(PAIR=True, PAIR_LATE=False, QKF=2, PVF=4, **QN_FLAGS)
    # the 4-workgroup kernel's bit-identical VALU cuts (gen_attn_fa MAXNN, SUM0, EPI_FAST; CTX16: start_pos = c * B):
    # pp2048 attention -6.4% cycles; the deep / KT2 twins keep their tuned code
    # QB_ZIGZAG: the short causal blocks are mostly Q / gate / O traffic; launched last (longest first) they ran as a
    # DRAM-bound tail. Interleaved with the long ones: pp2048 attention -5.5% cycles, bit-identical, neutral at depth
    # PV_FENCE: -16 v_mov per tile, pp2048 attention -2.3% cycles (-3% in the pipeline), bit-identical
    MAIN_FLAGS = dict(MAXNN=True, SUM0=True, EPI_FAST=True, CTX16=B % 16 == 0, QB_ZIGZAG=True, PV_FENCE=True)
    # the role-split twins too: EPI_FAST + CTX16 halve their epilogue VALU (B waves normalize all 256 dims alone);
    # bit-identical, KT2 -1.6% at 16K, -0.5% at 32K, -0.8% at 96K
    TWIN_EPI = dict(EPI_FAST=True, CTX16=B % 16 == 0)

    kt2_cfg = KT2_SHAPE
    kt2_shape = kt2_cfg if kt2_cfg != (a_hpw, a_qt) else None   # a shape of its own: its chunks get dispatch.txt rows
    fps_on = bool(FPS_FROM) and kvdeq and deq_bits == 4 and kt2_cfg == KT2_SHAPE == (6, 16)
    if FPS_FROM and not fps_on:
        raise SystemExit("YAH_FPS_FROM needs a kv4 paged set with the 6x16 KT2 twin")
    if fps_on and (B % gen_fpsparse.BLK or T % 256 or FPS_FROM % B or FPS_FROM < B):
        raise SystemExit("YAH_FPS_FROM: chunk a multiple of 128, context of 256, FPS_FROM a chunk start past chunk 0")
    fps_ls = gen_fpsparse.list_stride(T, B)
    fps_ext = gen_fpsparse.ext_pages(T)
    cap_attn = T + 256 * fps_ext if fps_on else T    # the f16 pools' capacity (pooled-row pages appended)
    fps_paths = set()                                # sources that get the sparse twin (set below: the kvdeq Q sources)

    def write_attn(path):
        A = gen_attn_fa
        keys = tuple(dict.fromkeys(("LDS_MIN", "DIAG_SWAP") + tuple(DEEP_FLAGS) + tuple(KT2_FLAGS) + tuple(MAIN_FLAGS)
                                   + tuple(KT2_SHAPE_FLAGS) + ("FPSPARSE", "FPS_LS", "FPS_TPAD", "FPS_ORD", "FPS_NG", "K4", "KDEC", "VQ4", "QROT")))
        saved = {k: getattr(A, k) for k in keys}
        twins = [(path, MAIN_FLAGS, None), (deep(path), dict(DEEP_FLAGS, LDS_MIN=OCC3_LDS, **TWIN_EPI), None)]
        if not A.KDEC:   # the KT2 twin (fp16 sources only)
            twins.append((kt2(path), dict(KT2_FLAGS, LDS_MIN=OCC3_LDS, **TWIN_EPI,
                                          **(KT2_SHAPE_FLAGS if kt2_cfg == KT2_SHAPE else {})), kt2_shape))
            if fps_on and path in fps_paths:   # the block-sparse KT2 twin (the kvdeq sources), on the kv4 pools directly
                twins.append((fps(path), dict(KT2_FLAGS, LDS_MIN=OCC3_LDS, **TWIN_EPI, **KT2_SHAPE_FLAGS, FPSPARSE=True,
                                              FPS_LS=fps_ls, FPS_TPAD=T, FPS_ORD=FPS_ORD, FPS_NG=B // 16 * 4, K4=True, KDEC=True, VQ4=True, QROT=True),
                              kt2_shape))
        for name, flags, shape in twins:
            for k, v in flags.items():
                setattr(A, k, v)
            if shape:
                A.configure(*shape)
            open(name, "w").write(A.gen())
            A.DIAG_SWAP = False
            open(alt(name), "w").write(A.gen())
            for k in keys:
                setattr(A, k, saved[k])
            if shape:
                A.configure(a_hpw, a_qt)
    gen_attn_fa.QNORM = QNORM and gen_attn_fa.Q16 and gen_attn_fa.QDIRECT is not False   # (see RoPE below)
    gen_attn_fa.QG16 = QG16_ON
    if QG16_ON:
        geom.append(("qg16", 0, 0, 0))
    attn_src = os.path.join(tmp, "yah_attn_hip.loom")
    write_attn(attn_src)
    attn_f_src = os.path.join(tmp, "yah_attn_hipf.loom")
    aov_need = 0
    if (a4 and RQOP and (B % gen_rot.FLAG_TT == 0)):   # the main twin with OCNT (and its alt)
        A = gen_attn_fa
        saved = {k: getattr(A, k) for k in ("DIAG_SWAP", "OCNT") + tuple(MAIN_FLAGS)}
        for k, v in dict(MAIN_FLAGS, OCNT=True).items():
            setattr(A, k, v)
        open(attn_f_src, "w").write(A.gen())
        A.DIAG_SWAP = False
        open(alt(attn_f_src), "w").write(A.gen())
        aov_need = A.ocnt_need()
        for k, v in saved.items():
            setattr(A, k, v)
    A.K4, A.K8, A.KDEC, A.VQ4, A.VQ8 = kv_flags
    vtrans_src = os.path.join(tmp, "yah_transpose_v16.loom")
    open(vtrans_src, "w").write(gen_attn_fa.gen_vtrans())
    # kv4 at depth: the Q rotation in its own kernel (gen_attn_fa QROT) and the attention on direct Q fragments, from the
    # chunk at QROT_FROM tokens on: 4 workgroups per WGP instead of 3, -4.2% attention at 8K..120K depth (the rotation's
    # 1.5 M cycles break even at ~8K). dispatch.txt "qrot <first chunk>"; the driver rotates q into its own f16 buffer.
    # kv4 under kvdeq: Q rotated from chunk 0 (its K is H256-rotated)
    qrot_first = 0 if kvdeq and deq_bits == 4 else -(-QROT_FROM // B) if gen_attn_fa.K4 and kv_paged and NCH > 1 else NCH
    attn_qr_src = os.path.join(tmp, "yah_attn_hip_qr.loom")
    qrot_src = os.path.join(tmp, "yah_qrot.loom")
    kdeq_src = os.path.join(tmp, "yah_kdeq%d.loom" % deq_bits)
    vdeq_src = os.path.join(tmp, "yah_vdeq%d.loom" % deq_bits)
    if qrot_first < NCH:
        A = gen_attn_fa
        saved = (A.Q16, A.QROT, A.K4, A.K8, A.KDEC, A.VQ4, A.VQ8)
        if kvdeq:   # the fp16 paged kernel (the f16 pools have the int4 pools' page layout)
            A.K4 = A.K8 = A.KDEC = A.VQ4 = A.VQ8 = False
            A.Q16 = True
            fps_paths.add(attn_qr_src)
        else:
            A.Q16 = A.QROT = True
        write_attn(attn_qr_src)
        A.Q16, A.QROT, A.K4, A.K8, A.KDEC, A.VQ4, A.VQ8 = saved
        open(qrot_src, "w").write(A.gen_qrot())
        geom.append(("qrot", qrot_first, 0, 0))
    if kvdeq:
        open(kdeq_src, "w").write(gen_attn_fa.gen_kdeq(deq_bits))
        open(vdeq_src, "w").write(gen_attn_fa.gen_vdeq(deq_bits))
        geom.append(("kvdeq", 0, 0, 0))
    fpmean_src = os.path.join(tmp, "yah_fpmean.loom")
    fpcopyk_src = os.path.join(tmp, "yah_fpcopyk.loom")
    fpcopyv_src = os.path.join(tmp, "yah_fpcopyv.loom")
    fpsel_src = os.path.join(tmp, "yah_fpsel.loom")
    fpord_src = os.path.join(tmp, "yah_fpord.loom")
    kdeqf_src = os.path.join(tmp, "yah_kdeq4_first.loom")
    vdeqf_src = os.path.join(tmp, "yah_vdeq4_first.loom")
    if fps_on:
        # the sparse twin reads the kv4 pools (the staging decode, as the dense kv4 kernel did): sparse tiles lose the
        # cross-workgroup L2 reuse of the f16 pools, so the 4x smaller reads win (128K last chunk attention -23%, its
        # kdeq / vdeq only decode the two chunks yah_fpmean reads). yah_fpcopyk / v write the pooled rows as int4 rows
        # and V^T tiles of the kv4 pools' extension; dispatch.txt "fpskv4"
        open(fpmean_src, "w").write(gen_fpsparse.gen_fpmean())
        open(fpcopyk_src, "w").write(gen_fpsparse.gen_fpcopyk())
        open(fpcopyv_src, "w").write(gen_fpsparse.gen_fpcopyv())
        open(fpsel_src, "w").write(gen_fpsparse.gen_fpsel(T, B, gen_fpsparse.ALPHA))
        open(kdeqf_src, "w").write(gen_attn_fa.gen_kdeq(4, first=True))
        open(vdeqf_src, "w").write(gen_attn_fa.gen_vdeq(4, first=True))
        geom.append(("fpsparse", FPS_FROM // B, fps_ext, fps_ls))
        geom.append(("fpskv4", 0, 0, 0))
        if FPS_ORD:   # the attention launches the tile lists longest first (yah_fpord after yah_fpsel; "fpord")
            open(fpord_src, "w").write(gen_fpsparse.gen_fpord(T, B))
            geom.append(("fpord", 0, 0, 0))
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

    # DeltaNet: chunked WY Gated DeltaNet (tools/gen_gdn_chunk.py), f16 WMMA inputs (B is a multiple of 64).
    # Its grid (2, heads) is recorded as rowsplit.hal's row group.
    dn_src = os.path.join(tmp, "yah_deltanet_hip_f32.loom")
    open(dn_src, "w").write(gen_gdn_chunk.gen())
    # the f16 hand-off conv per (256 channels, gen_conv_kq.TB tokens) workgroup: bit-identical, inputs read once
    conv_tb = conv_tb_ok(B)
    assert conv_tb
    dn_vconv = (conv_tb and a4 and (DNSPLIT == (40, 8)))
    if dn_vconv:
        text = gen_conv_kq.gen(4096)
        geom.append(("dnvconv", gen_conv_kq.tiles(4096), 0, 0))
    else:
        text = gen_conv_kq.gen()
    conv_src = os.path.join(tmp, "yah_ssm_conv_kq.loom")
    open(conv_src, "w").write(text)
    assert DNOV_ON <= dn_vconv
    convf_src = os.path.join(tmp, "yah_ssm_conv_kqf.loom")
    if DNOV_ON:
        open(convf_src, "w").write(gen_conv_kq.gen(4096, flag=gen_gemm_iu4.flag_count(10240)))
        geom.append(("dnov", 0, 0, 0))
    convp_src = os.path.join(tmp, "yah_ssm_conv_kq_p.loom")
    convp = (DNOV_ON and (B % gen_rot.FLAG_TT == 0))
    if convp:
        open(convp_src, "w").write(gen_conv_kq.gen_p(B))
        geom.append(("convkqp.hal", gen_conv_kq.PG, 0, 0))
    prepab_src = os.path.join(tmp, "yah_deltanet_prep_ab.loom")
    text = open(os.path.join(E.LOOM, "yah_deltanet_prep_ab_f32.loom")).read()
    open(prepab_src, "w").write(prepab_x16(text) if 640 in O16_MT else text)
    prepab_text = prepab_x16(text) if 640 in O16_MT else text
    postnorm_base = open(os.path.join(E.LOOM, "yah_ssm_postnorm_gate_f16.loom")).read()
    if 384 in O16_MT:
        postnorm_base = postnorm_g16(postnorm_base)
    postnorm_src = os.path.join(tmp, "yah_ssm_postnorm.loom")
    open(postnorm_src, "w").write(postnorm_base)
    dn_perm = a4 and DNSPLIT == (40, 8)
    dn_ss = (dn_perm and dn_vconv)
    # one flagged pnq for both DeltaNet parts (gen_rot.gen_pnq "ab"): part b's DeltaNet counts its finished lanes
    pnq_ab = DNOV_ON and dn_ss
    DNRING_ON = (pnq_ab and RQK_ON and NQW_ON and (DNSPLIT == (40, 8)))
    prepabr_src = os.path.join(tmp, "yah_deltanet_prep_abr.loom")
    if DNRING_ON:
        open(prepabr_src, "w").write(prepab_ring(prepab_text, gen_gdn_ss.done_count(DNSPLIT[1])))
        geom.append(("dnring", 0, 0, 0))
    # blocks per head of the DeltaNet grid (the unsplit rowsplit.hal is not dispatched with dnsplit)
    geom.append(("rowsplit.hal", 0, 1 if dn_ss else 2, 0))
    geom.append(("dnthreads", gen_gdn_ss.THREADS if dn_ss else 256, 0, 0))
    dn_parts = []
    # A4 sets: DeltaNet part b runs the heads of whole rotq groups (gen_rot.pn_heads), so each fused postnorm + rotq part
    # (gen_pnq) reads one part's compact output; "dnperm" tells the host the parts are head sets, not ranges
    if DNSPLIT:
        ha, hb = DNSPLIT
        for tag, h0, nh in (("a", 0, ha), ("b", ha, hb)):
            src = os.path.join(tmp, f"yah_deltanet_{tag}.loom")
            if dn_perm:
                assert len(gen_rot.pn_heads(tag)) == nh
                if dn_ss:
                    open(src, "w").write(gen_gdn_ss.gen(gen_rot.pn_heads(tag), flag=pnq_ab and tag == "b"))
                else:
                    open(src, "w").write(gen_gdn_chunk.gen(heads=gen_rot.pn_heads(tag), vconv=dn_vconv, o16=True))
            else:
                open(src, "w").write(gen_gdn_chunk.gen(h0, nh))
            dn_parts.append((tag, h0, nh, src))
        geom.append(("dnsplit", 0, ha, hb))
        if dn_perm:
            geom.append(("dnperm", 0, 0, 0))
    if conv_tb:
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
    rope_src = "yah_fused_qk_rope_batched_f32.loom"
    ropew_src = os.path.join(tmp, "yah_fused_qk_rope_batched_wait.loom")
    if paged_f16k:   # RoPE writes K rows straight into the paged pool
        rope_src = os.path.join(tmp, "yah_fused_qk_rope_batched_kpaged.loom")
        open(rope_src, "w").write(rope_kpaged(open(os.path.join(E.LOOM, "yah_fused_qk_rope_batched_f32.loom")).read()))
        geom.append(("rope_kpaged", 0, 0, 0))
    text = open(rope_src if os.path.isabs(rope_src) else os.path.join(E.LOOM, rope_src)).read()
    rope_src = os.path.join(tmp, "yah_fused_qk_rope_batched_live.loom")
    text = rope_live(text)
    if KV16_ON:
        text = rope_kv16(text)
    vtrope = paged_f16k and paged_f16v   # RoPE stores V^T into the paged pool: no yah_vtpage pass
    if vtrope:
        text = rope_vtpage(text)
        geom.append(("vtrope", 0, 0, 0))
    open(rope_src, "w").write(text)
    if KV16_ON:
        geom.append(("kv16in", 0, 0, 0))
    # QNORM: the attention takes the raw f32 q and does the head norm + RoPE + f16(q / 16) in its prologue (same bits);
    # RoPE does K / V only. dispatch.txt "qnorm"
    if gen_attn_fa.QNORM:
        text = open(rope_src).read()
        rope_src = os.path.join(tmp, "yah_fused_qk_rope_batched_konly.loom")
        konly = rope_konly(text)
        kv = (KV16_ON and vtrope and (B % 16 == 0))
        open(rope_src, "w").write(rope_kv(konly) if kv else konly)
        geom.append(("qnorm", 0, 0, 0))
        if kv:
            geom.append(("ropekv", 0, 0, 0))
        if ROPEW_ON:
            assert kv
            open(ropew_src, "w").write(rope_kv(konly, gen_gemm_iu4.flag_count(1024)))
            geom.append(("ropew", 0, 0, 0))
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
    # The chunk's token_embd rows (Q8_0) from its token ids on the GPU (gen_decode_misc.gen_embed_q8_0, one workgroup per
    # row), so the host stages ids and never waits between chunks.
    embed_src = os.path.join(tmp, "yah_embed.loom")
    open(embed_src, "w").write(gen_decode_misc.gen_embed_q8_0(rows=B))
    geom.append(("embed.hal", 1, 5120 // 16, B))   # one token per workgroup
    # MQ4 GEMMs read their input in the weights' rotated basis (gen_rot): the driver rotates each GEMM input site in place,
    # the DeltaNet postnorm's out of place (dst in HF head order); the head's f32 norm row without awq.
    # A4 sets rotate and quantize instead (rotq_*: int4 activations with one scale per 128 values, the iu4 GEMM's K phase;
    # rows "<hal> 0 <lanes> <workgroups>"). Per-token scales for the norm-fed sites and 512-value spans for the others cost
    # WT2 KL 0.064 vs 0.051 (2 chunks; p99.9 2.08 vs 1.01).
    rot_src = {}
    rots = [("rot_head.hal", (5120, 1, False, None, "f32"))]
    if not a4:
        rots += [("rot_5120.hal", (5120, B)), ("rot_6144.hal", (6144, B)), ("rot_17408.hal", (17408, B)),
                 ("rot_heads.hal", (6144, B, True, (48, 16, 128)))]
    for name, args in rots:
        rot_src[name] = os.path.join(tmp, name[:-4] + ".loom")
        open(rot_src[name], "w").write(gen_rot.gen(*args))
    for name, (k, heads) in (("rotq_5120.hal", (5120, None)), ("rotq_6144.hal", (6144, None)),
                             ("rotq_17408.hal", (17408, None)), ("rotq_heads.hal", (6144, (48, 16, 128)))) if a4 else ():
        rot_src[name] = os.path.join(tmp, name[:-4] + ".loom")
        open(rot_src[name], "w").write(gen_rot.gen_q(k, B, gen_gemm_iu4.KP, True, heads, hot=k == 17408))
        geom.append((name, 0, 64, k // 1024))
    if GUOV_ON:   # rotq_17408's geometry
        rot_src["rotqf_17408.hal"] = os.path.join(tmp, "rotqf_17408.loom")
        open(rot_src["rotqf_17408.hal"], "w").write(
            gen_rot.gen_q(17408, B, gen_gemm_iu4.KP, True, None, flag=gen_gemm_iu4.gu_flag_count(17408)))
        geom.append(("guov", 0, 0, 0))
    if RQK_ON:   # the rotq_17408 / rotq_6144 geometry
        for name, src in (("rotqfo_17408.hal", gen_rot.gen_q(17408, B, gen_gemm_iu4.KP, True, None,
                                                              flag=gen_gemm_iu4.gu_flag_count(17408), oflag=True)),
                          ("rotqo_6144.hal", gen_rot.gen_q(6144, B, gen_gemm_iu4.KP, True, None, oflag=True))):
            rot_src[name] = os.path.join(tmp, name[:-4] + ".loom")
            open(rot_src[name], "w").write(src)
        geom.append(("rqk", 0, 0, 0))
        if RQP and GUOV_ON and B % gen_rot.FLAG_TT == 0:
            rot_src["rotqp_17408.hal"] = os.path.join(tmp, "rotqp_17408.loom")
            open(rot_src["rotqp_17408.hal"], "w").write(
                gen_rot.gen_qp(17408, B, gen_gemm_iu4.KP, RQP, gen_gemm_iu4.gu_flag_count(17408)))
            geom.append(("rotqp_17408.hal", RQP, 0, 0))
        if RQOP and B % gen_rot.FLAG_TT == 0:
            rot_src["rotqop_6144.hal"] = os.path.join(tmp, "rotqop_6144.loom")
            open(rot_src["rotqop_6144.hal"], "w").write(gen_rot.gen_qp(6144, B, gen_gemm_iu4.KP, RQOP, 0))
            geom.append(("rotqop_6144.hal", RQOP, 0, 0))
            if aov_need:
                rot_src["rotqopf_6144.hal"] = os.path.join(tmp, "rotqopf_6144.loom")
                open(rot_src["rotqopf_6144.hal"], "w").write(gen_rot.gen_qp(6144, B, gen_gemm_iu4.KP, RQOP, aov_need, plow=True))
                geom.append(("rotqopf_6144.hal", RQOP, 0, 0))
                geom.append(("aov", 0, 0, 0))
    if NQW_ON:
        geom.append(("nqw", 0, 0, 0))
    if a4:
        # the norm-fed sites: RMS norm + rotq in one kernel (gen_rot.gen_nq; same bits, no f16 norm round trip)
        rot_src["normq_5120.hal"] = os.path.join(tmp, "normq_5120.loom")
        open(rot_src["normq_5120.hal"], "w").write(gen_rot.gen_nq(5120, B, gen_gemm_iu4.KP))
        geom.append(("normq_5120.hal", 0, 16 * 5120 // 256, 1))
        if NQOV_ON:
            assert gen_gemm_iu4.tile(5120)[1] == gen_rot.FLAG_TT
            rot_src["normqf_5120.hal"] = os.path.join(tmp, "normqf_5120.loom")
            open(rot_src["normqf_5120.hal"], "w").write(
                gen_rot.gen_nq(5120, B, gen_gemm_iu4.KP, flag=gen_gemm_iu4.flag_count(5120)))
            geom.append(("nqov", 0, 0, 0))
        if NQGU_ON or NQW_ON:
            for f in (0, gen_gemm_iu4.flag_count(5120)) if NQOV_ON else (0,):
                name = "normq%so_5120.hal" % ("f" if f else "")
                rot_src[name] = os.path.join(tmp, name[:-4] + ".loom")
                open(rot_src[name], "w").write(gen_rot.gen_nq(5120, B, gen_gemm_iu4.KP, flag=f, oflag=True,
                                                              persist=NQP if B % NQP == 0 else 0))
            geom.append(("nqgu", 0, 0, 0))
            if NQP and B % NQP == 0:
                geom.append(("nqp", NQP, 0, 0))
        if DNSPLIT == (40, 8):
            # the DeltaNet postnorm + rotq_heads (gen_rot.gen_pnq) per DeltaNet part (dn_perm): a beside part b, b after it
            for part in "ab":
                name = "pnq_%s.hal" % part
                rot_src[name] = os.path.join(tmp, "pnq_%s.loom" % part)
                open(rot_src[name], "w").write(gen_rot.gen_pnq(B, gen_gemm_iu4.KP, part, o16=True))   # gate: kstoreh
                geom.append((name, 0, 64, len(gen_rot.pnq_groups(part)) // 4))
            if DNOV_ON:   # part a waiting on the gate GEMM's counters (pnq_a.hal's geometry)
                rot_src["pnq_af.hal"] = os.path.join(tmp, "pnq_af.loom")
                open(rot_src["pnq_af.hal"], "w").write(
                    gen_rot.gen_pnq(B, gen_gemm_iu4.KP, "a", o16=True, flag=gen_gemm_iu4.flag_count(6144)))
            if pnq_ab:   # both parts in one grid (raw_: part b's compact f16 rows, then part a's), so part b no
                # longer dispatches behind part a's waiting workgroups
                rot_src["pnq_abf.hal"] = os.path.join(tmp, "pnq_abf.loom")
                open(rot_src["pnq_abf.hal"], "w").write(
                    gen_rot.gen_pnq(B, gen_gemm_iu4.KP, "ab", o16=True, flag=gen_gemm_iu4.flag_count(6144),
                                    dflag=gen_gdn_ss.done_count(DNSPLIT[1])))
                geom.append(("pnq_abf.hal", 0, 64, len(gen_rot.pnq_groups("ab")) // 4))
                # a trimmed chunk's gate GEMM runs unflagged (gen_gemm_iu4 flags only full token tiles): the same
                # grid waiting only on DeltaNet part b, behind the gate GEMM in the graph (pnq_abf.hal's geometry)
                if DNRING_ON:   # waiting on the ring update (which waits on part b), counting for ssm_out's kres
                    assert gen_rot.q_flag_count(6144) == 128 * len(gen_rot.pnq_groups("ab")) // 4 * 2
                    rot_src["pnq_abfr.hal"] = os.path.join(tmp, "pnq_abfr.loom")
                    open(rot_src["pnq_abfr.hal"], "w").write(
                        gen_rot.gen_pnq(B, gen_gemm_iu4.KP, "ab", o16=True, flag=gen_gemm_iu4.flag_count(6144),
                                        dflag=ring_flag_count(), oflag=True))
                    if PNQP and B % gen_rot.FLAG_TT == 0:
                        rot_src["pnq_abfrp.hal"] = os.path.join(tmp, "pnq_abfrp.loom")
                        open(rot_src["pnq_abfrp.hal"], "w").write(gen_rot.gen_pnqp(
                            B, gen_gemm_iu4.KP, PNQP, gen_gemm_iu4.flag_count(6144), ring_flag_count()))
                        geom.append(("pnq_abfrp.hal", PNQP, 0, 0))
                rot_src["pnq_abd.hal"] = os.path.join(tmp, "pnq_abd.loom")
                open(rot_src["pnq_abd.hal"], "w").write(
                    gen_rot.gen_pnq(B, gen_gemm_iu4.KP, "ab", o16=True, dflag=gen_gdn_ss.done_count(DNSPLIT[1])))
    with open(os.path.join(outdir, "rot_signs.bin"), "wb") as fh:
        fh.write(gen_rot.signs())
    geom.append(("rot", 0, 0, 0))
    if a4:
        geom.append(("a4", 0, 0, 0))
    # the output head: the tile-free GEMV (gen_gemv), rows per workgroup / lanes / workgroups in its row
    head_src = os.path.join(tmp, "yah_gemv_head.loom")
    open(head_src, "w").write(gen_gemv.gen("plain", ["mq4"], 248320, 5120))
    geom.append(("gemv.hal", gen_gemv.rows_per_wg(), 32 * 4, 248320 // gen_gemv.rows_per_wg()))
    with open(os.path.join(outdir, "dispatch.txt"), "w") as fh:
        for hal, tk, rg, tt in geom:
            fh.write("%s %d %d %d\n" % (hal, tk, rg, tt))

    # The output norm (tools/gen_half_norm.py): fully unrolled, all loads issued up front.
    norm_src = os.path.join(tmp, "yah_half_norm_unrolled.loom")
    # 4 rows (waves) per workgroup: 46.1 -> 41.9 ms per pp2048 vs one row per workgroup; the driver derives the grid from
    # the kernel's workgroup size.
    open(norm_src, "w").write(gen_half_norm.gen_split(5120, wpr=4, split=NORM_SPLIT))
    pn_parts = []
    for tag, h0, nh, _ in ([] if dn_perm else dn_parts):
        base = postnorm_base
        src = os.path.join(tmp, f"yah_ssm_postnorm_{tag}.loom")
        open(src, "w").write(postnorm_heads(base, h0, nh, B))
        pn_parts.append((f"postnorm_{tag}.hal", src, nh))
    def occ(path, c, use_kt2=True):   # the 3-workgroup twins: KT2 from KT2_FROM keys of context on, else PAIR from OCC3_FROM
        if use_kt2 and fps_on and c * B >= FPS_FROM and os.path.exists(fps(path)):
            return fps(path)
        if use_kt2 and c * B >= KT2_FROM and (c or not kt2_shape) and os.path.exists(kt2(path)):
            return kt2(path)
        return deep(path) if c * B >= OCC3_FROM else path
    fixed = [
        (embed_src, "embed.hal", ["nop=0"]),
        (norm_src, "norm.hal",
         ["yah_half_norm.rows=%d" % B, "yah_half_norm.dim=5120",
          "yah_half_norm.eps=1e-06", "yah_half_norm.fused=0"]),
        # the conv with the q / k L2 norm (prep_kq) fused in
        (conv_src, "convkq.hal",
         ["yah_ssm_conv_kq.batch=%d" % B, "yah_ssm_conv_kq.qkv_dim=10240",
          "yah_ssm_conv_kq.num_key_heads=16"]),
        *([(convf_src, "convkqf.hal",
            ["yah_ssm_conv_kq.batch=%d" % B, "yah_ssm_conv_kq.qkv_dim=10240",
             "yah_ssm_conv_kq.num_key_heads=16"])] if DNOV_ON else []),
        *([(convp_src, "convkqp.hal", ["yah_ssm_conv_kq_p.items=%d" % (16 * B // gen_conv_kq.PST)])] if convp else []),
        *([(prepabr_src, "prepabr.hal",
             ["yah_deltanet_prep_ab.batch=%d" % B,
              "yah_deltanet_prep_ab.qkv_size=10240",
              "yah_deltanet_prep_ab.num_heads=48"])] if DNRING_ON else []),
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
            "yah_deltanet.inner_size=%d" % (128 * nh if dn_perm else 6144), "yah_deltanet.num_key_heads=16",
            "yah_deltanet.num_heads=48"]) for tag, _, nh, src in dn_parts],
        *[(src, hal, ["yah_ssm_postnorm_fp16.head_count=%d" % (nh * B)]) for hal, src, nh in pn_parts],
        *[(src_, stem_ + ".hal" if c == 0 else stem_ + "_c%d.hal" % c, [
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
            *(["yah_fused_qk_rope_batched.k16_elems=%d" % (1024 * T)] if paged_f16k else [])]) for c in range(NCH)
          for src_, stem_ in ((rope_src, "rope"),) + (((ropew_src, "ropew"),) if ROPEW_ON and gen_attn_fa.QNORM else ())],
        *([(qrot_src, "qrot.hal", ["yah_qrot.token_count=%d" % B])] if qrot_first < NCH else []),
        *[(src, "%s.hal" % stem if c == 0 else "%s_c%d.hal" % (stem, c),
           ["%s.cache_capacity=%d" % (sym, T), "%s.rows=%d" % (sym, (c + 1) * B)]
           + (["%s.out_capacity=%d" % (sym, cap_attn)] if sym.startswith("yah_vdeq") else [])
           + (["%s.first=%d" % (sym, max(c - 1, 0) * B)] if src in (kdeqf_src, vdeqf_src) else []))
          for c in range(NCH) if kvdeq
          for sparse in (fps_on and c * B >= FPS_FROM,)
          for src, stem, sym in ((kdeqf_src if sparse else kdeq_src, "kdeq%d" % deq_bits, "yah_kdeq%d" % deq_bits),
                                 (vdeqf_src if sparse else vdeq_src, "vdeq%d" % deq_bits, "yah_vdeq%d" % deq_bits))],
        *[(occ(attn_qr_src if c >= qrot_first else attn_src, c), "wmma.hal" if c == 0 else "wmma_c%d.hal" % c, [
            "attention_prefill.cache_capacity=%d" % cap_attn,
            "attention_prefill.token_count=%d" % B,
            "attention_prefill.start_pos=%d" % (c * B),
            "attention_prefill.num_heads=24", "attention_prefill.num_kv_heads=4",
            "attention_prefill.head_dim=256", "attention_prefill.gqa=6"]) for c in range(NCH)],
        *([(attn_f_src, "wmmaf.hal", [
            "attention_prefill.cache_capacity=%d" % cap_attn,
            "attention_prefill.token_count=%d" % B,
            "attention_prefill.start_pos=0",
            "attention_prefill.num_heads=24", "attention_prefill.num_kv_heads=4",
            "attention_prefill.head_dim=256", "attention_prefill.gqa=6"])] if aov_need and occ(attn_src, 0) == attn_src
          and qrot_first > 0 else []),
        *[(src, "%s.hal" % stem if c == 0 else "%s_c%d.hal" % (stem, c), cfg)
          for c in range(NCH) if fps_on
          for src, stem, cfg in (
              (fpmean_src, "fpmean", ["yah_fpmean.capacity=%d" % T, "yah_fpmean.capx=%d" % cap_attn,
                                      "yah_fpmean.first=%d" % (max(c - 1, 0) * B // 128), "yah_fpmean.nend=%d" % ((c + 1) * B // 128)]),
              *[(src, stem, ["%s.capacity=%d" % (sym, T), "%s.capx=%d" % (sym, cap_attn),
                             "%s.nend=%d" % (sym, (c + 1) * B // 128)])
                for src, stem, sym in ((fpcopyk_src, "fpcopyk", "yah_fpcopyk"), (fpcopyv_src, "fpcopyv", "yah_fpcopyv"))],
              *([(fpsel_src, "fpsel", ["yah_fpsel.start_pos=%d" % (c * B), "yah_fpsel.tpad=%d" % T])]
                if c * B >= FPS_FROM else []))],
        *([(fpord_src, "fpord.hal", ["yah_fpord.ls=%d" % fps_ls])] if fps_on and FPS_ORD else []),
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
        (head_src, "gemv.hal", ["nop=0"]),
        *[(src, name, ["nop=0"]) for name, src in rot_src.items()],
        ("yah_argmax_f32.loom", "argmax.hal", ["yah_argmax.vocab=248320"]),
    ]
    # Loom low-level LICM (HRX patch 0011) for the attention kernels where it wins: it hoists the fragment loads'
    # per-lane LDS address math out of the key loop (fp16 -1..-2%; kv4 QROT -0.5% with one extra live register); the
    # kv4 staged-Q build loses 11%, so it compiles without it
    def attn_env(loom):
        if "_kt2" in loom:   # KT2 (fp16 sources only): hoisting capped at 6 new loop-live units (240 VGPRs)
            return {"LOOM_EXP_LICM": "1", "LOOM_EXP_LICM_MAX_LIVE": "6"}
        loom = loom.replace("_alt.loom", ".loom").replace("_deep.loom", ".loom").replace("_kt2.loom", ".loom")
        if loom == attn_qr_src:   # fp16 paged under kvdeq, else the kv4 direct-Q kernel
            return {"LOOM_EXP_LICM": "1"} if kvdeq else {"LOOM_EXP_LICM": "1", "LOOM_EXP_LICM_MAX_LIVE": "1"}
        if loom in (attn_src, attn_f_src) and (not gen_attn_fa.KDEC or kv8deq):
            return {"LOOM_EXP_LICM": "1"}
        return None
    kt2_rows = []
    for loom, outname, configs in fixed:
        E.emit(loom, configs, outname, outdir, attn_env(loom))
        if outname.startswith("wmma") and "_kt2" in loom:   # KT2 past 240 VGPRs loses its third workgroup: use the old twin
            hal = os.path.join(outdir, outname)
            if (hal_vgprs(hal) > KT2_VGPR or hal_spills(hal)) and "_kt2fps" in loom:
                raise SystemExit("attention %s: the block-sparse KT2 twin has %d VGPRs, %d B spills (no dense fallback)"
                                 % (outname, hal_vgprs(hal), hal_spills(hal)))
            if hal_vgprs(hal) > KT2_VGPR or hal_spills(hal):
                c = int(re.search(r"_c(\d+)\.hal$", outname).group(1)) if "_c" in outname else 0
                print("attention %s: KT2 %d VGPRs, %d B spills, re-emitted from the previous twin" % (outname, hal_vgprs(hal), hal_spills(hal)))
                loom = occ(loom.replace("_kt2.loom", ".loom"), c, use_kt2=False)
                E.emit(loom, configs, outname, outdir, attn_env(loom))
        if outname.startswith("wmma") and "_kt2" in loom and kt2_shape:
            kt2_rows.append(outname)
        if outname.startswith("wmma"):   # attn_guard (deep twins: 3 workgroups per WGP allow 256 VGPRs, never spills)
            hal = os.path.join(outdir, outname)
            limit = 256 if "_deep" in loom else KT2_VGPR if "_kt2" in loom else ATTN_VGPR
            if hal_vgprs(hal) > limit or hal_spills(hal):
                print("attention %s: %d VGPRs, %d B spills, re-emitted without DIAG_SWAP" % (outname, hal_vgprs(hal), hal_spills(hal)))
                E.emit(alt(loom), configs, outname, outdir, attn_env(loom))
                if hal_spills(hal):
                    raise SystemExit("attention %s still spills: refusing to emit a spilling attention kernel" % outname)
        n += 1
    if kt2_rows:   # the KT2 chunks' own launch geometry (the global wmma.hal row is chunk 0's)
        with open(os.path.join(outdir, "dispatch.txt"), "a") as fh:
            for hal in kt2_rows:
                fh.write("%s %d %d %d\n" % (hal, kt2_shape[1], kt2_shape[0], (B + kt2_shape[1] - 1) // kt2_shape[1]))

    print("emitted %d GEMM + fixed prefill HALs for B=%d (tile=%d, token_tiles=%d)"
          % (n, B, TILE, TT))


if __name__ == "__main__":
    main()
