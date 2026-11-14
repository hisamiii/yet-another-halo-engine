#!/usr/bin/env python3
"""Emit the prefill HAL set for B-token chunks: every kernel compiled for its shape, plus dispatch.txt.

Every token dimension is bound to B: GEMM token_tiles = B / tile (grid y), fixed kernels batch = B, residual dim = 5120 * B.
YAH_CTX=T (a multiple of B, default B) sizes the KV cache and emits one rope / attention HAL per chunk.
The driver runs T / B chunks and carries the DeltaNet and conv states. YAH_KV selects the KV format (gen_kvq.kv_bits).
dispatch.txt has one row per HAL, "<hal> <tokens per workgroup> <row groups> <token_tiles>", plus mode marker rows.
HAL names follow emit_prefill.py.

usage: emit_prefill_pp.py <model.gguf> <outdir> [tokens]   (default 2048 tokens)
"""
import dataclasses, functools, json, os, re, subprocess, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emit_prefill as E  # noqa: E402
import gen_attn_fa  # noqa: E402
import gen_deltanet_hip  # noqa: E402
import gen_gdn_chunk  # noqa: E402
import gen_gdn_ss  # noqa: E402
import gen_conv_kq  # noqa: E402
import gen_kres_persist  # noqa: E402
import gen_half_norm  # noqa: E402
import gen_decode_misc  # noqa: E402
import gen_gemv  # noqa: E402
import gen_rot  # noqa: E402
import gen_gemm_iu4  # noqa: E402
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

# DNVCONV: A4 dnperm sets without a NPU qkv split: the v channels' conv and alpha / beta run inside DeltaNet
# (gen_gdn_chunk vconv), the conv kernel takes only the q / k heads (gen_conv_kq.gen(4096)) and prep_ab only advances
# the conv ring, after both DeltaNet parts. Bit-identical; dispatch.txt "dnvconv <conv workgroups>".
DNVCONV = True
# DNO16: DeltaNet's output (the pnq input) as f16(O 2^8) in A4 dnperm sets (gen_gdn_chunk o16): not bit-exact; vs the
# WT2 KL reference (24 chunks) mean 0.0671 / q99.9 2.026 / top-1 88.40% vs f32 0.0687 / 2.023 / 87.91%, inside the A4
# redraw spread (0.0693-0.0695 / 2.15-2.27). Unscaled f16 failed the tail (q99.9 2.52: subnormal outputs).
DNO16 = os.environ.get("YAH_DN_O16", "1") == "1"
# DNSS: the dnperm parts as the state-stationary kernel (gen_gdn_ss: one workgroup per head, the state in registers, the
# head's v-independent work once). Grid (1, heads) x gen_gdn_ss.THREADS (dispatch.txt "dnthreads"). pp2048 (sums over
# 48 layers): part a 54.9 -> 36.1 ms; part b 27.3 -> 33.8 (8 workgroups, one per WGP: each head's serial chain, beside
# pnq a). A ninth wave runs the solve beside the row waves' K S^T / Q S^T (X = T1 (K S^T), not bit-identical): part a
# 34.5 -> 30.7 ms, part b 33.4 -> 31.0; WT2 KL 0.0671 -> 0.0683 / q99.9 2.026 -> 2.054, inside the redraw spread.
DNSS = True
# DNSS_NAT: f16 (non-A4) sets with the state-stationary DeltaNet too: the v conv inside it (DNVCONV's path), head ranges,
# its O as f32 at the heads' own columns (gen_gdn_ss natural), read by the plain postnorm parts. Sets whose NPU takes
# qkv rows keep gen_gdn_chunk (the v conv needs every qkv column). Gate N mean KL 1.06e-6 (vs the f32 recurrent
# reference 1.53e-6, gen_gdn_chunk 1.08e-6); pp2048 GPU only -0.97% (11-02: DeltaNet alone 66 -> 34 ms, conv 18 -> 9,
# the postnorm loses its overlap 14 -> 24). Fusing the postnorm into part a passed the gate but doubled part a.
DNSS_NAT = os.environ.get("YAH_DNSS_NAT", "1") == "1"


def _rewrite(text, pairs):
    for a, b in pairs:
        assert text.count(a) == 1, a
        text = text.replace(a, b)
    return text


# DN_REC: the f32 recurrent DeltaNet (gen_deltanet_hip) at any B, as the B % 32 != 0 sets have it: an accuracy reference
# for DeltaNet changes (engine/run/gate), not a production set
DN_REC = os.environ.get("YAH_DN_REC", "0") == "1"


def dn32(B):
    """The chunked DeltaNet (and its f16 conv hand-off) applies."""
    return B % 32 == 0 and not DN_REC


def conv_tb_ok(B):
    """The f16 hand-off conv per (256 channels, gen_conv_kq.TB tokens) workgroup applies."""
    return 640 in O16_MT and B % gen_conv_kq.TB == 0 and dn32(B) and gen_gdn_chunk.CONV16


def conv_x16(text):
    """yah_ssm_conv_kq reading the qkv GEMM's f16 output (O16_MT 640): each tap widened back to f32 at its load."""
    pairs = [("%x_view = buffer.view %x_noalias[%base] : buffer -> view<[%x_total]xf32>",
              "%x_view = buffer.view %x_noalias[%base] : buffer -> view<[%x_total]xf16>")]
    for v in ("s3", "x1", "x2", "x3"):
        off = "%s3_off" if v == "s3" else f"%{v}_off"
        pairs.append((f"    %{v} = view.load %x_view[{off}] : view<[%x_total]xf32> -> f32\n",
                      f"    %{v}h = view.load %x_view[{off}] : view<[%x_total]xf16> -> f16\n"
                      f"    %{v} = scalar.extf %{v}h : f16 to f32\n"))
    return _rewrite(text, pairs)


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


def conv_n16(text):
    """yah_ssm_conv_kq handing DeltaNet (gen_gdn_chunk CONV16) its f16 inputs: conv_out holds f16(v) for the v channels and
    f16(k * inv_k), f16(q * q_scale) for the q / k channels, the f32 products and rounding DeltaNet did itself on the f32
    conv_out, so the result is bit-identical. Lane 0 of a q / k workgroup also leaves inv_k / q_scale in LDS for the others."""
    def rep(a, b):
        nonlocal text
        assert text.count(a) == 1, a
        text = text.replace(a, b)
    rep("%out_view = buffer.view %out_noalias[%base] : buffer -> view<[%x_total]xf32>",
        "%out_view = buffer.view %out_noalias[%base] : buffer -> view<[%x_total]xf16>")
    rep("%kq_bytes = index.constant 1024 : offset", "%kq_bytes = index.constant 1040 : offset")
    text = text.replace("view<256xf32>", "view<258xf32>")
    rep("    view.store %result, %out_view[%s3_off] : f32, view<[%x_total]xf32>\n", """    %is_v = index.cmp uge, %wg, %nkh : index
    scf.if %is_v {
      %result_h = scalar.fptrunc %result : f32 to f16
      view.store %result_h, %out_view[%s3_off] : f16, view<[%x_total]xf16>
    }
""")
    rep("        view.store %kq_all, %scales_view[%o2] : f32, view<[%scale_total]xf32>\n      }\n    }\n  }\n", """        view.store %kq_all, %scales_view[%o2] : f32, view<[%scale_total]xf32>
        %ks_slot = index.constant 256 : index
        %qs_slot = index.constant 257 : index
        view.store %inv_k, %kq_lds[%ks_slot] : f32, view<258xf32>
        view.store %q_scaled, %kq_lds[%qs_slot] : f32, view<258xf32>
      }
    }
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %ks_at = index.constant 256 : index
    %qs_at = index.constant 257 : index
    %ks = view.load %kq_lds[%ks_at] : view<258xf32> -> f32
    %qs = view.load %kq_lds[%qs_at] : view<258xf32> -> f32
    %own_scale = scf.select %lane_hi, %ks, %qs : f32
    %own = view.load %kq_lds[%lane] : view<258xf32> -> f32
    %own_n = scalar.mulf %own, %own_scale : f32
    %own_h = scalar.fptrunc %own_n : f32 to f16
    %own_row = index.mul %t, %qkv_dim : index
    %own_off = index.add %own_row, %c : index
    view.store %own_h, %out_view[%own_off] : f16, view<[%x_total]xf16>
  }
""")
    return text


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


def rope_wait(text, need):
    """rope_konly polling its token tile's k and v GEMM counters (bindings kcnt, vcnt after the others; gen_gemm_iu4
    flag: need each) before any load: wave 0 polls relaxed, the barrier's global acquire after it."""
    a = "%ptab: buffer) {"
    assert text.count(a) == 1
    text = text.replace(a, "%ptab: buffer, %kcnt: buffer, %vcnt: buffer) {")
    a = "  %tid = kernel.workitem.id<x> : index\n"
    assert text.count(a) == 1
    poll = [
        "  %rw_c32 = index.constant 32 : index",
        "  %rw_c128 = index.constant 128 : index",
        "  %rw_c127 = index.constant 127 : index",
        "  %rw_b = config.get @yah_fused_qk_rope_batched.batch : index",
        "  %rw_b1 = index.add %rw_b, %rw_c127 : index",
        "  %rw_tt = index.div %rw_b1, %rw_c128 : index",
        "  %rw_t = index.div %bi, %rw_c128 : index",
        "  %rw_kna, %rw_vna = buffer.assume.noalias %kcnt, %vcnt : buffer, buffer",
        "  %rw_kv = buffer.view %rw_kna[%base] : buffer -> view<[%rw_tt]xi32>",
        "  %rw_vv = buffer.view %rw_vna[%base] : buffer -> view<[%rw_tt]xi32>",
        "  %rw_w0 = index.cmp ult, %tid, %rw_c32 : index",
        f"  %rw_need = scalar.constant {need} : i32",
        f"  %rw_lim = scalar.constant {gen_rot.FLAG_POLLS} : i32",
        "  %rw_z = scalar.constant 0 : i32",
        "  %rw_one = scalar.constant 1 : i32",
        "  scf.if %rw_w0 {",
        "    %rw_n = scf.while(%rw_i = %rw_z : i32) -> (i32) {",
        "      %rw_x = view.atomic.load %rw_kv[%rw_t] {ordering = relaxed, scope = device} : view<[%rw_tt]xi32> -> i32",
        "      %rw_y = view.atomic.load %rw_vv[%rw_t] {ordering = relaxed, scope = device} : view<[%rw_tt]xi32> -> i32",
        "      %rw_m = scalar.minsi %rw_x, %rw_y : i32",
        "      %rw_e = scalar.cmpi slt, %rw_m, %rw_need : i32",
        "      %rw_l = scalar.cmpi ult, %rw_i, %rw_lim : i32",
        "      %rw_g = scalar.andi %rw_e, %rw_l : i1",
        "      scf.condition %rw_g, %rw_i : i1, i32",
        "    } do(%rw_j: i32) {",
        "      %rw_j1 = scalar.addi %rw_j, %rw_one : i32",
        "      scf.yield %rw_j1 : i32",
        "    }",
        "  }",
        "  kernel.barrier<global> scope(workgroup) ordering(acquire)",
    ]
    return text.replace(a, a + "\n".join(poll) + "\n")


def rope_kv(text, need=None):
    """rope_konly (KV16 + vtrope) rewritten wave-per-head: K workgroups (x < batch / 2) normalize, rotate and store 8
    (token, KV head) rows, one per wave; V workgroups (batch / 4 more) store one KV head of a 16-token tile as V^T,
    lane = dim, one 32 B row per lane. Same ops and sum order as the workgroup reduce (per-32 subgroup reduces, then
    the 8 partials as a cluster-8 tree), so the same bits. need: wait for the k / v GEMM counters (rope_wait)."""
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
# QG16 (A4 with QNORM): the kqg GEMM stores q and gate as f16, the attention reads them so. dispatch.txt "qg16"
QG16 = True
QG16_ON = False   # set by main
# GSIG (YAH_GSIG, default u16; 0: off): the kqg GEMM's epilogue applies the gate's sigmoid and stores it as unorm16
# bits ("u16", gen_gemm_tile KQG_GSIG), the attention multiplies (gen_attn_fa GSIG): half the gate bytes, no exp /
# reciprocal in the attention epilogue. Gate N (KL 1.07e-6); attention busiest SQ 71.6 -> 73.8% of floor at depth 0,
# +0.2 at 8K, even at 64K; pp2048 attention -3.1%, kqg -0.3% (half the stores). Only where the kqg GEMM writes every
# gate row: not with the NPU's q rows, f16 q / gate (QG16) or half sets. dispatch.txt "gsig"
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
# KV16 (A4): the K / V GEMMs (kstore, 1024 rows) store f16, RoPE widens them (rope_kv16). dispatch.txt "kv16in"
KV16 = True
KV16_ON = False   # set by main
# NQOV (A4): every kres also as iu4_kresf_* (gen_gemm_iu4 flag: per-token-tile completion counters) and normq as
# normqf_5120 (gen_rot.gen_nq flag: rows wait for their token tile), so the driver records the norm after its kres
# with no graph barrier and the norm runs in the GEMM's tail. Same values. dispatch.txt "nqov"
NQOV = True
NQOV_ON = False   # set by main
# DNOV (A4 dnvconv, with DNSPLIT (40, 8)): the qkv and gate kstores also flagged (iu4_kstoref_*, iu4_kstorehf_*), the
# conv as convkqf and pnq part a as pnq_af waiting on their counters, so the conv runs in the qkv GEMM's tail and pnq
# part a in the gate GEMM's. Same values. dispatch.txt "dnov"
DNOV = True
DNOV_ON = False   # set by main
# CONVP (with DNOV): the flagged conv as gen_conv_kq.gen_p (persistent workgroups walking (key head, token tile)
# segments, convkqp.hal) instead of gen's grid of 16-token workgroups: conv exposure 5.3-5.5 -> 4.2 ms per pp2048
# (64-token segments over 512 workgroups: -> ~1.9 ms). Same values
CONVP = True
# GUOV (A4): the gu GEMM also as iu4_guf_* (gen_gemm_iu4 flag: a counter per 1024 features and token tile) and
# rotq_17408 as rotqf_17408 (gen_rot.gen_q flag), so the down projection's rotq runs in the gu GEMM's tail. Same values.
# dispatch.txt "guov"
GUOV = True
GUOV_ON = False   # set by main
# NQGU (A4, with NQOV and GUOV): normq also as normqo_5120 / normqfo_5120 (gen_rot.gen_nq oflag: per-token-tile counters)
# and the flagged gu as iu4_gufw_* (gen_gemm_iu4 wait), so the driver records the gu GEMM after its norm with no graph
# barrier and its first workgroups start in the norm's tail. Same values. dispatch.txt "nqgu"
NQGU = True
NQGU_ON = False   # set by main
# RQK (A4, with NQOV): the residual rotqs also as rotqfo_17408 / rotqo_6144 (gen_rot.gen_q oflag) and the flagged kres as
# iu4_kresfw_* (gen_gemm_iu4 wait), so the driver records the kres after its rotq with no graph barrier. Same values.
# dispatch.txt "rqk"
RQK = True
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
AOV = True
# ROPEW (with NQW and QNORM): the attention's k / v GEMMs also as iu4_kstorefw_* (per-token-tile counters) and RoPE as
# ropew[_c<i>].hal (rope_wait: a workgroup polls its token tile's k and v counters), so no graph barrier separates
# them: RoPE runs in the v GEMM's tail. Same values. dispatch.txt "ropew"
ROPEW = True
# ROPEKV (QNORM, KV16, vtrope): RoPE as rope_kv (wave-per-head K, coalesced V^T tiles), grid (3 * batch / 4).
# Same values. dispatch.txt "ropekv"
ROPEKV = True
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
NQW = True
NQW_ON = False   # set by main
# DNRING (A4, with DNOV's pnq_abf, RQK, NQW): the conv ring update as prepabr (prepab_ring: waits on DeltaNet part b's
# counter, counts) before the pnq, pnq_abfr waiting on it and counting rows for ssm_out's kres (iu4_kresfw_*), so a
# DeltaNet layer's tail and the next layer's input GEMMs need no graph barrier. Same values. dispatch.txt "dnring"
DNRING = True
DNRING_ON = False   # set by main


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


def narrow_variants(fmt, mt, kb, B, out, outdir, kind):
    """Emit "<hal>.t<BN>.hal" for each narrow token tile of this GEMM (the tuner's "variants", else NARROW on the
    default tile), plus the tuner's per-bucket choices as "pick:<hal>:<max tokens> <max tokens> <bn> 0" rows."""
    import gen_gemm_tile as TG
    if fmt not in TILE_FMTS or mt < 4:
        return []
    base, rows = TG.default_tile(fmt, kind, kb), []
    full = dataclasses.replace(base, **tiles()["tiles"].get(out[:-4], {}))
    tuned = tiles().get("variants", {}).get(out[:-4])
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
TILE_FMTS = ("iq3s", "iq4xs", "iq4nl", "iq3xxs", "q4k", "q5k", "q6k", "iq2xxs", "iq2xs", "iq2s", "q3k", "q8_0", "mq4", "q2k")


# Activations-from-global GEMMs (gen_gemm_tile Tile.afrag): "<hal>.af.hal" reads a fragment-major input (Tile.atiled; the
# FFN norm "norm_t.hal" and the swiglu "<hal>.af.to.hal" write it). 512-token tiles, only the decoded weights in LDS.
# Same values as the GEMM. YAH_AFRAG=0 leaves them out.
AFRAG = os.environ.get("YAH_AFRAG", "1") != "0"
AF_TILE = dict(bm=128, bn=512, wm=1, wn=16, ksub=128, dbuf=False, decahead=False, afrag=True, atiled=True, stg_minwg=0,
               tallepi=True, ecoal=True, esr=32)
AF_PIPE = dict(wlate=True, bpre=2, b0early=True)
# IQ3_S: step 1's B prefetch issued with step 0's, before the decode; otherwise the allocator copies one of its fragments
# right after its load, which drains vmcnt(0) once per phase (IQ4_XS: 200 VGPRs with it, fused IQ3_S ffn: 216)
AF_PIPE3 = dict(AF_PIPE, b0early=2)
AF_KQ = dict(rhs_outer=False, rhs_fence=0)   # K-quants without B prefetch (Q6_K: 208 VGPRs with it)
# Q4_K / Q3_K / Q5_K with the B prefetch: their decoders came down to 184-192 VGPRs (Q4DFMA, Q3_FMIX), so it fits now:
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
    # Q2_K with the B prefetch since its fptrunc(fma) decode (Q2DFMA, 187 -> 192 VGPRs): attention seg -0.15 M cycles
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
if __import__("os").environ.get("YAH_FFN_SEL", "1") == "0":
    AF_SEL = AF_MIX
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
    if not AFRAG or B % AF_TILE["bn"]:
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
KST2 = os.environ.get("YAH_KST2", "1") == "1"
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
            r = _emit_gen(lambda f, k: TG.gen(f, "kstore", t, False), t.bn, fmt, 2 * mt, kb, B,
                          "gemm_kst2_%s_%d_%d.hal" % (fmt, mt, kb), outdir, "kst2", t.rowgrp)
        finally:
            TG.SPLIT2 = False
        if r:
            out.append(r)
    return out


# SEG: segmented GEMMs (gen_segcat): a layer's small GEMMs over the same activations ride in the large one's dispatch,
# each with its own weights, format, row tile and output. DeltaNet: attn_qkv + ssm_alpha + ssm_beta (48-row segments);
# attention: the q + gate kqg + attn_k + attn_v (their afrag tiles; k and v are not adjacent and often of two formats).
# "seg_<kind>_<fmt>_<mt>_<kb>__..." per segment, the large GEMM first; dispatch.txt "<hal> <tile> <groups> <tt>": the
# grid x (all segments' row groups) in the rowgrp column. Same values.
SEG = os.environ.get("YAH_SEG", "1") == "1"
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
    if not (AFRAG and B % AF_TILE["bn"] == 0):
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
    if not AFRAG or knobs is None or B % AF_TILE["bn"]:
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
        text = gen_kres_persist.persist(TG.gen(fmt, "kres", pt, False),
                                        TG.gen(fmt, "kstore", dataclasses.replace(pt, respre=0), False), B // t.bn,
                                        place=pt.pslice)
        r = _emit_gen(lambda f, k: text, t.bn, fmt, mt, kb, B, out[:-4] + ".af.p.hal", outdir, kind, t.rowgrp)
        if r:
            rows.append(r)
    return rows


# NPU column split (YAH_NPU_SPLIT="<site>=<NPU rows>,...", rows multiples of "npurows" (640; fused sets 512), q also of 512; empty: none): the NPU (XDNA2, HRX .xdna)
# computes the trailing rows of a GEMM, the GPU the leading ones. Sites (rows x K):
#   qkv  DeltaNet attn_qkv (10240 x 5120, f16 out)     gate  DeltaNet attn_gate (6144 x 5120, f16 out)
#   q    attention attn_q (12288 x 5120, kqg: whole heads of [256 q | 256 gate])
#   out  ssm_out / attn_output (5120 x 6144, kres)      down  ffn_down (5120 x 17408, kres)
# The NPU runs one image (K = 5120 per call: switching images costs ~0.55 ms each): out and down take K chunks of 5120 and
# the GPU computes the rest of K for the NPU's rows (npu_rem_tile, gen_gemm_tile.KWIN), which the unpack adds.
#   ffn  ffn_gate + ffn_up (17408 x 5120 each): the same NPU rows of both, silu(gate) * up in the unpack
# Per site and format, the GPU's share of every variant the driver may run ("<hal>.npu.hal": "<hal>" any of the set's
# GEMM HALs, its afrag / persistent / fragment-major-output forms included; gen_gemm_tile.OSTRIDE) and the NPU's rows
# decoded straight to its BFP16 weight stream ("dqbfp_<fmt>_<mt>_<kb>[_c<chunk>].hal", gen_gemm_tile.DQ_BFP). Per K the
# activation encoders ("npu_enc_<K>[_c<chunk>][_t].hal", _t: fragment-major input) and the NPU GEMM image
# ("npu_gemm_<K>.xdna", 8 columns of gen_npu_gemm.TN = 640 rows per call: "npurows"); per site the unpack ("npu_unpack_<site>.hal", ffn also
# "npu_unpack_ffn_t.hal" with fragment-major output; qkv: "npu_unpack_qkv_tail.hal" + the conv split "convkq_g.hal" /
# "convkq_c.hal", which reads C itself). dispatch.txt: "npusplit_<site> <rows> 0 0", "npubytes_<K> <A>
# <W per call> <C per call>", and per kernel "<hal> <workgroups> <workgroup size> 0" (the split GEMMs have GEMM rows).
NPU_SPLIT = dict((k, int(v)) for k, v in (x.split("=") for x in os.environ.get("YAH_NPU_SPLIT", "").split(",") if x))
# The NPU image waits for each job's ready word itself (gen_npu_gemm GATE, HRX patch 0016): no host relay sits between
# the GPU and the NPU. dispatch.txt: "npugate <supply> <calls> <record>" (gen_npu_gemm GATE_*), and npu_flag_wait.hal
# waits for a job's done word (gen_npu_unpack.gen_flag_wait).
# YAH_NPU_DCOL=1: the NPU decodes its weights itself (gen_npu_gemm dcol 2): 7 GEMM columns read by a decoder column
# from the raw GGUF rows the engine registers with the NPU, so no GPU decode ("dqbfp") runs. Per format with a decoder
# (DCOL_FMTS) and site K the image "npu_dcol_<fmt>_<K>.xdna", and per format "npu_dcol_<fmt>.swap.xdna" (its decoder
# tiles' setup alone: the engine switches formats inside a command). The decoders' k-block order (gen_npu_dec.KPERM)
# permutes the activations (gen_bfp16_encode.ACT_PERM): "p4" for the fused writers (norm, postnorm, ffn unpack) and the
# plain encoders, "pk" (Q4_K) by the "_pk" encoders into A's second half. Tensors of other formats stay on the GPU.
# YAH_NPU_FUSE=1: the same raw-row sets on 8 fused GEMM columns (gen_npu_gemm fuse 2: every GEMM core decodes its own
# panel slice; no decoder column, no swap images), images "npu_dcol_<fmt>_<K>.xdna" for the formats in FUSE_FMTS, NP 4:
# 512-row calls (site rows multiples of 512; NP 5 images no longer fit the grid decoders' raw rings), one replay group.
NPU_FUSE = os.environ.get("YAH_NPU_FUSE") == "1"
NPU_DCOL = os.environ.get("YAH_NPU_DCOL") == "1" or NPU_FUSE
FUSE_FMTS = ("iq4xs", "iq3xxs", "iq3s", "q3k", "iq2xxs", "iq2xs", "q4k", "q5k")
FUSE_FFN_FMTS = ("iq4xs", "iq3xxs", "iq3s", "q3k", "iq2xxs", "iq2xs", "q5k", "q4k", "iq2s")   # the fused FFN block's gate / up (NP
# 4 images: room for a grid decoder's tables; Q4_K's own k-block order would need its own activation stream: Q4_K only
# in pairs, a P4 H through its gather-free decoder, gord GF)
FUSE_FFN_DOWN_FMTS = ("iq4xs", "iq3xxs", "iq3s", "q4k", "q3k", "q5k", "iq2s")
DCOL_FMTS = {"iq4xs": ("IQ4_XS", "p4"), "iq3s": ("IQ3_S", "p4"), "iq3xxs": ("IQ3_XXS", "p4"), "q4k": ("Q4_K", "pk"),
             "q3k": ("Q3_K", "p4"), "iq2xxs": ("IQ2_XXS", "p4"),
             "iq2xs": ("IQ2_XS", "p4"), "q5k": ("Q5_K", "p4"),
             "iq2s": ("BFP", "p4")}   # IQ2_S: no decoder, the rows pre-decoded (bfp_rows: npu_bfp_<tensor>.bin)
NPU_KS = (36, 36, 36, 20) if NPU_DCOL else GN.KS   # k-blocks per pass and K-slice row; K = 1024 * passes
# the fused FFN block's gate / up of the formats whose filler cores hold 40 k-block slices (IQ3_S's head A ring does not
# place): the short tail slice frees the room for its deferred epilogue (gen_npu_gemm dfr); the norm before such a
# layer writes this split's stream ("norm_t_bfp_g1s.hal", dispatch row "npuffndfr_<fmt>")
NPU_DFR_FMTS, NPU_DFR_KS = ("iq4xs", "iq3xxs"), (40, 40, 40, 8)
# YAH_NPU_IX4 (default 1): the FFN block's IQ3_S / IQ3_XXS gate / up rows transcoded on the host to IQ4_XS blocks with
# the same weights (ix4_files: npu_ix4_<...>.bin), decoded by IQ4_XS's decoder with their kvalues (gen_npu_dec IQ4_XO /
# IQ4_XX): no grid tables, so IQ3_S's fillers also hold the deferred epilogue's 40 k-block slices
NPU_IX4 = os.environ.get("YAH_NPU_IX4", "1") == "1"
IX4_FMTS = {"iq3s": "IQ4_XO", "iq3xxs": "IQ4_XX"} if NPU_IX4 else {}
IX4_FAMILY = ("IQ4_XS", "IQ4_XO", "IQ4_XX") if NPU_IX4 else ()
if NPU_IX4:
    NPU_DFR_FMTS += ("iq3s",)
# YAH_NPU_BFPSW (default 1): the other pairs' gate / up rows (the NPU's) pre-decoded to bfp16 (bfp_pair_rows:
# npu_bfp_<prefix>ffn_gate_up.bin), one BFP image like a single format's with the deferred epilogue
# ("npu_ffnsw_bfp_<hord>", dispatch rows "npuffnbx_<gate>_<up>") instead of column pairs
NPU_BFPSW = os.environ.get("YAH_NPU_BFPSW", "1") == "1"
# YAH_NPU_BFPSITE (default 1): site tensors of a format without an NPU decoder (IQ4_NL, Q2_K attn_q) take their NPU
# rows pre-decoded (bfp_site_rows: npu_bfp_<tensor>.bin), on the BFP site image ("npu_dcol_iq2s_<K>", dispatch rows
# "npusitebx_<tensor>")
NPU_BFPSITE = os.environ.get("YAH_NPU_BFPSITE", "1") == "1"
# pairs of one format (no fused GPU gate / up) whose decoder holds the deferred epilogue's slices: the NPU's side one
# image like a single format's, on the GGUF rows (Q3_K's leaf exceeds the program memory at 40 / 8; Q5_K's decode
# runs slower than the BFP image's stream: BFP with NPU_BFPSW)
SAME_SW_FMTS = ("q5k",) if NPU_IX4 and not NPU_BFPSW else ()


def gu_fmt(fmt):
    """The NPU decoder of the FFN block's gate / up rows of fmt."""
    return IX4_FMTS.get(fmt) or DCOL_FMTS[fmt][0]


# the fused site and down images of the formats whose four cores each decode one super-block of every unit (gen_npu_gemm
# kchain), the gate's waiter and relay on tails (gtail: rows 1 decode); the grid formats' helper rings after the A rings
# (hlate: their tables leave the mids no contiguous A ring otherwise)
NPU_KCHAIN_FMTS = ("iq4xs", "q4k", "q3k", "iq3s", "iq3xxs", "q5k")
KCHAIN = lambda fmt: dict(kchain=1, hcap=2, gtail=1, hlate=int(fmt in ("iq3s", "iq3xxs"))) if fmt in NPU_KCHAIN_FMTS else {}
# the decoder-column images' setups run no call (gen_npu_gemm gfirst: the gate waits at a job's first firing), so an
# image switch inside a command costs no dummy call; dispatch.txt "npugfirst"
NPU_GFIRST = 1
# Token halves (LoomPrefill halves): the qkv / q and out jobs and their GPU chains run per half of the chunk, so the NPU works during the DeltaNet / attention chain.
# main emits the set at half the chunk into "<outdir>/h" (YAH_EMIT_HALF=1: only what the halves run); the FFN block's unpack and norm get "_h<i>" forms for half i's tokens.
EMIT_HALF = os.environ.get("YAH_EMIT_HALF") == "1"


def npu_halves(B):
    split = set(NPU_SPLIT)
    return NPU_DCOL and B % 128 == 0 and bool({"qkv", "q", "out"} & split) and not {"ffn", "down"} & split


NPU_COLS = 7 if NPU_DCOL and not NPU_FUSE else 8
NPU_NP = 4 if NPU_FUSE else GN.NP   # 16-row slabs per column of a site call
NPU_ROWS = NPU_COLS * 16 * NPU_NP   # output rows per NPU call
NPU_SITES = {"qkv": ("kstore", 640, 20), "gate": ("kstore", 384, 20), "q": ("kqg", 768, 20), "out": ("kres", 320, 24),
             "down": ("kres", 320, 68), "ffn": ("ffn", 1088, 20)}
NPU_PASSES = 5   # one NPU image: K = 5120 per call (gen_npu_gemm passes)
# YAH_NPU_FFNBLK=<F> (decoder-column sets, F = 7168): the NPU owns the FFN's last F features of every layer whose
# ffn_gate and ffn_up share a p4 decoder format (and ffn_up follows ffn_gate) and whose ffn_down has a grid decoder
# (natural k-block order). Its gate / up calls (gen_npu_gemm swiglu, NP 4, raw rows straight from the model; call j's
# column c computes features 8 (128 c + 4 j) + [0, 32) of F) apply swiglu and write H into C: bfp16 fragments as
# [M block][pass 7][16-token slab][128 k-blocks] (hpass). Its down calls (ain "h": 4 + 3 passes, 560 rows each, the 10th
# call overlapping the 9th) write their partials. The GPU runs the other features through the fused ffn's split
# (".npu.hal", nn = F) and down over them ("npuffnrem_<fmt>.hal", a K window); npu_unpack_ffnblk.hal adds hidden + both.
# A command switches image families at job boundaries (LoomNpu FusedCommand: the image's setup, and one ungated call
# unless NPU_GFIRST).
# With YAH_NPU_FUSE (any F = 1024 n; layers whose three tensors are in FUSE_FMTS): gate / up on 8 fused columns, call j
# (F / 256 of them) column c features 256 j + 32 c + [0, 32) of F (gen_npu_gemm swcol 32), down in 640-row calls.
# Down's K parts (both kinds): F's passes in the fewest parts of at most 5 (a fused panel holds a part), sizes balanced.
NPU_FFNBLK = int(os.environ.get("YAH_NPU_FFNBLK", "0"))
# Fused sets: the GPU window writes f32 and npu_unpack_ffnblk adds hidden + it + the NPU's partials after the job.
# YAH_NPU_FFNRES=1: its residual epilogue waits for the NPU instead (npuffnres). Only 20 of its workgroups fit at once
# (one per WGP), so while the NPU is the late one the rest of the GPU's down starts after the NPU's (~5 ms a layer).
NPU_FFNRES = os.environ.get("YAH_NPU_FFNRES", "0") == "1"
FFNBLK_FMTS = ("iq4xs", "iq3s", "iq3xxs", "q3k", "iq2xxs", "iq2xs")
FFNBLK_DOWN_FMTS = ("iq3s", "iq3xxs", "iq4xs", "q4k")
# the down decoders' k-block order where it is not natural: their layers' gate / up images write H in it (_p4 / _q4k)
FFNBLK_HORD = {"iq4xs": "P4", "q4k": "Q4K"}
# Layers whose gate and up formats differ: column pairs (gen_npu_gemm ufmt; 192 features per call), F = 6144 and down in
# two 3-pass parts.
FFNBLK_PAIR = 6144


def ffnblk_parts(F):
    """(first pass, passes) of down's K parts over F features (engine/model/loom_prefill.hpp FfnBlkParts)."""
    P = F // 1024
    n = -(-P // 5)
    sizes = [P // n + (i < P % n) for i in range(n)]
    return tuple((sum(sizes[:i]), sizes[i]) for i in range(n))


def npu_chunks(site):
    """(k offset, passes) of the NPU's K chunks of a site; the GPU computes the rest of K (npu_rem) for the NPU's rows."""
    K = NPU_SITES[site][2] * 256
    return [(1024 * NPU_PASSES * i, NPU_PASSES) for i in range(K // (1024 * NPU_PASSES))]


def npu_rem(site):
    """The K columns past the NPU's chunks (out: 1024 of 6144, down: 2048 of 17408): the GPU's npurem_<site>_<fmt>.hal."""
    return NPU_SITES[site][2] * 256 % (1024 * NPU_PASSES)


def npu_rem_tile(fmt, kbw, kbt, kind="kstore"):
    """The K remainder's tile (gen_gemm_tile.KWIN, kbw of kbt blocks): the afrag kstore (or kres) with the knobs of the
    format's K = kbt * 256 kres (emit_prefill_pp.AF; its residual and persistence knobs dropped), the plain afrag
    tile (straight-line k steps, as afrag needs) for a format with no tuned kres."""
    import gen_gemm_tile as TG
    knobs = dict(AF.get((fmt, "kres", 320, kbt)) or AF.get((fmt, "kres", 320, 24)) or dict(ksl=True))
    knobs.pop("persist", None)
    knobs["respre"] = 0
    return dataclasses.replace(TG.default_tile(fmt, kind, kbw), **AF_TILE, **knobs)


def fit_rowgrp(t, mt):
    """Tile t over mt row tiles: decoder-column splits (560-row NPU calls) leave GPU / remainder row counts the tile's
    row group need not divide; the largest of its row group, 4, 2, 1 that does."""
    if not NPU_DCOL or mt % t.rowgrp == 0:
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


def npu_split(rows, combos, B, outdir):
    import gen_bfp16_encode as GE
    import gen_gemm_tile as TG
    import gen_npu_unpack as GU
    sys.path.insert(0, os.path.dirname(HERE))
    import hrx_paths
    bad = set(NPU_SPLIT) - set(NPU_SITES)
    assert not bad, "unknown NPU split sites %s" % sorted(bad)
    assert all(v % NPU_ROWS == 0 and 0 < v < NPU_SITES[k][1] * 16 for k, v in NPU_SPLIT.items()) and B % 512 == 0
    assert NPU_SPLIT.get("q", 0) % 512 == 0, "q: whole heads"
    assert not (NPU_COLS == 7 and "q" in NPU_SPLIT), "dcol: q's whole heads are not 560-row calls"
    GE.ACT_PERM = "p4" if NPU_DCOL else None
    out = []
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)

    def split(fmt, kind, mt, kb, t, name, nn, persist=None, tokens=B):
        """The GPU's share: rows [0, mt * 16 - nn) of the GEMM HAL name (tile t) at the full stride, over tokens."""
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
        else:
            gen = lambda f, k: TG.gen(f, k, tf, False)
        try:
            out.append(_emit_gen(gen, tf.bn, fmt, mtg, kb, tokens, name[:-4] + ".npu.hal", outdir, kind, tf.rowgrp,
                                 ostride=mt * 16))
        except (SystemExit, RuntimeError, ValueError) as ex:
            if tf is t:
                raise
            # a decoder-column split's narrower row group may not emit (spills): the engine runs that form unsplit
            print("npu split %s: not emitted with row group %d (%s)" % (name, tf.rowgrp, str(ex)[:120]))

    def variants(fmt, kind, mt, kb, base, nn):
        """Every form of GEMM base the set emits (tile_kstore, afrag_variants): its split."""
        t = TG.default_tile(fmt, kind, kb)
        if base[:-4] in tiles()["tiles"]:
            t = dataclasses.replace(t, **tiles()["tiles"][base[:-4]])
        if fmt in TILE_FMTS and mt % t.rowgrp == 0:
            split(fmt, kind, mt, kb, t, base, nn)
        knobs = AF.get((fmt, kind, mt, kb))
        if AFRAG and knobs is not None and B % AF_TILE["bn"] == 0:
            knobs = dict(knobs)
            persist = knobs.pop("persist", False)
            ta = dataclasses.replace(TG.default_tile(fmt, kind, kb), **AF_TILE, **knobs)
            split(fmt, kind, mt, kb, ta, base[:-4] + ".af.hal", nn)
            if kind == "swiglu":
                split(fmt, kind, mt, kb, dataclasses.replace(ta, tout=True), base[:-4] + ".af.to.hal", nn)
            if persist and kind == "kres" and B // ta.bn > 1 and 2 * kb >= gen_kres_persist.SLICES:
                pt = dataclasses.replace(ta, **persist) if isinstance(persist, dict) else ta
                # fewer row blocks than CUs: the persistent kres over all but the last token tile, run beside the tiled
                # kres on that tile (LoomPrefill::RunResidual)
                if B // ta.bn > 2:
                    split(fmt, kind, mt, kb, ta, base[:-4] + ".af.p3.hal", nn, persist=pt, tokens=B - ta.bn)

    def dqbfp(fmt, mt, kb, nn, ci=None, chunk=None):
        """The NPU's rows (the last nn of mt * 16) decoded to its weight stream; chunk ci: (k offset, passes) of a K chunk."""
        dt = dataclasses.replace(TG.default_tile(fmt, "kstore", kb), bm=64, wm=2, wn=4, decahead=False,
                                 ksub=TG.DQ_BFP_KSUB, dbuf=False)
        name = "dqbfp_%s_%d_%d%s.hal" % (fmt, mt, kb, "" if chunk is None else "_c%d" % ci)
        if chunk is None:
            TG.DQ_BFP, kbc, kfull = (NPU_KS, kb * qk_of(fmt) // 1024), kb, 0
        else:
            TG.DQ_BFP = (NPU_KS, chunk[1], chunk[0] // 256, kb)
            kbc, kfull = chunk[1] * 4, kb
        try:
            _emit_gen(lambda f, k: TG.gen(f, "dequant", dt), dt.bn, fmt, nn // 16, kbc, B, name, outdir, "dequant",
                      dt.rowgrp, sym="yah_dequant_%s_bfp16" % fmt, kfull=kfull)
        finally:
            TG.DQ_BFP = None
        out.append((name, TG.dq_wgs(nn // 16, kbc, dt, fmt), dt.lanes, 0))

    done = set()
    dcol_images, bsites = set(), set()
    for kind, fmt, port, mt, kb in sorted(combos):
        for site, (skind, smt, skb) in NPU_SITES.items():
            if site not in NPU_SPLIT or mt != smt or kb * qk_of(fmt) != skb * 256 or fmt not in TILE_FMTS:
                continue
            nn = NPU_SPLIT[site]
            if site == "ffn":   # the gate kstore and the up swiglu of each format (the fused ffn below)
                if kind == "kstore":
                    variants(fmt, "kstore", mt, kb, "gemm_kstore_%s_%d_%d.hal" % (fmt, mt, kb), nn)
                elif kind == "swiglu":
                    variants(fmt, "swiglu", mt, kb, "gemm_swiglu_%s_%d_%d.hal" % (fmt, mt, kb), nn)
                else:
                    continue
            elif skind == "kres":
                if kind != "residual":
                    continue
                variants(fmt, "kres", mt, kb, "gemm_kres_%s_%d_%d.hal" % (fmt, mt, kb), nn)
            elif kind == "kstore":
                variants(fmt, skind, mt, kb, "gemm_%s_%s_%d_%d.hal" % (skind, fmt, mt, kb), nn)
            else:
                continue
            chunks = npu_chunks(site)
            chunked = len(chunks) > 1 or npu_rem(site)
            for ci, chunk in (enumerate(chunks) if chunked else ((None, None),)):
                if NPU_DCOL:
                    if fmt in (FUSE_FMTS if NPU_FUSE else DCOL_FMTS):
                        dcol_images.add((fmt, kb * 256))
                    elif NPU_FUSE and NPU_BFPSITE and chunk is None and skind != "kres":   # (BFP site rows)
                        dcol_images.add(("iq2s", skb * 256))
                        bsites.update((t, nn) for t in combos[(kind, fmt, port, mt, kb)])
                    continue
                if (fmt, mt, kb, chunk) not in done:
                    done.add((fmt, mt, kb, chunk))
                    dqbfp(fmt, mt, kb, NPU_SPLIT[site], ci, chunk)
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
    if "ffn" in NPU_SPLIT and AFRAG and B % AF_TILE["bn"] == 0:   # the fused gate + up GEMMs (ffn_fused)
        by = {}
        for nm, dims, ty in rows:
            p = nm.split(".")
            if len(p) > 2 and p[0] == "blk" and p[2] in ("ffn_gate", "ffn_up") and E.FMT.get(ty):
                by.setdefault(p[1], {})[p[2]] = (E.FMT[ty][0], dims[1] // 16, dims[0] // E.FMT[ty][2])
        for fmt, mt, kb in sorted({l["ffn_gate"] for l in by.values() if l.get("ffn_gate") == l.get("ffn_up")}):
            if fmt in AF_FFN:
                t = dataclasses.replace(TG.default_tile(fmt, "swiglu", kb), **AF_TILE, **AF_FFN[fmt], ffn=True)
                for sfx, tt in ((".af.hal", t), (".af.to.hal", dataclasses.replace(t, tout=True))):
                    split(fmt, "ffn", mt, kb, tt, "gemm_ffn_%s_%d_%d%s" % (fmt, mt, kb, sfx), NPU_SPLIT["ffn"])
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
    # the ffn unpack writes down's BFP16 input for its columns (gen_npu_unpack bfp): down's chunks there need only their
    # leading k-blocks encoded ("npu_enc_17408_c<i>p")
    ffn_bfp = "ffn" in NPU_SPLIT and "down" in NPU_SPLIT
    if ffn_bfp:
        ffn_off = NPU_SITES["ffn"][1] * 16 - NPU_SPLIT["ffn"]
        assert NPU_SITES["ffn"][1] * 16 == NPU_SITES["down"][2] * 256
        for ci, (k_off, passes) in enumerate(npu_chunks("down")):
            cover = min(k_off + 1024 * passes, ffn_off) - k_off
            if 0 < cover < 1024 * passes:
                for tiled in (False, True):
                    name = "npu_enc_17408_c%dp%s.hal" % (ci, "_t" if tiled else "")
                    src = os.path.join(tmp, name[:-4] + ".loom")
                    open(src, "w").write(GE.gen("act", B, list(NPU_KS), passes, tiled=tiled, k_off=k_off, k_src=17408,
                                                kb_count=cover // 8))
                    E.emit(src, ["nop=0"], name, outdir)
                    out.append((name, B // 8 * (cover // 8) // GE.WG, GE.WG, 0))
        out.append(("npuffnbfp", ffn_off, 0, 0))
    for K, ci, (k_off, passes) in sorted(encs, key=lambda x: (x[0], -1 if x[1] is None else x[1])):
        for tiled in (False, True):
            for perm in (("p4", "pk") if NPU_DCOL else (None,)):
                name = "npu_enc_%d%s%s%s.hal" % (K, "" if ci is None else "_c%d" % ci, "_t" if tiled else "",
                                                 "_pk" if perm == "pk" else "")
                src = os.path.join(tmp, name[:-4] + ".loom")
                GE.ACT_PERM = perm
                try:
                    open(src, "w").write(GE.gen("act", B, list(NPU_KS), passes, tiled=tiled, k_off=k_off,
                                                k_src=K if 1024 * passes != K else None))
                finally:
                    GE.ACT_PERM = "p4" if NPU_DCOL else None
                E.emit(src, ["nop=0"], name, outdir)   # emit_hal.py wants a config; the encoder has none
                out.append((name, B // 8 * (passes * sum(NPU_KS)) // GE.WG, GE.WG, 0))
    # the norms that feed K = 5120 sites also write the NPU's BFP16 input (gen_half_norm.gen_split bfp): "<norm>_bfp.hal"
    if {"qkv", "q", "ffn"} & set(NPU_SPLIT):
        for tiled, name in ((False, "norm_bfp.hal"), (True, "norm_t_bfp.hal"), ("both", "norm_rt_bfp.hal")):
            src = os.path.join(tmp, name[:-4] + ".loom")
            open(src, "w").write(gen_half_norm.gen_split(5120, wpr=4, split=NORM_SPLIT, tiled=tiled,
                                                         bfp=(B, list(NPU_KS), NPU_PASSES)))
            E.emit(src, ["yah_half_norm.rows=%d" % B, "yah_half_norm.dim=5120", "yah_half_norm.eps=1e-06",
                         "yah_half_norm.fused=0"], name, outdir)
        out.append(("npunormbfp", 1, 0, 0))
    # the DeltaNet postnorm over heads 0 .. DNSPLIT[0] - 1 (exactly ssm_out's NPU K chunk) also writes the NPU's BFP16
    # input (gen_postnorm_bfp): "postnorm[_t]_a_bfp.hal"
    if "out" in NPU_SPLIT and DNSPLIT and DNSPLIT[0] * 128 == 1024 * NPU_PASSES:
        import gen_postnorm_bfp
        for tiled, name in ((False, "postnorm_a_bfp.hal"), (True, "postnorm_t_a_bfp.hal")):
            src = os.path.join(tmp, name[:-4] + ".loom")
            open(src, "w").write(gen_postnorm_bfp.gen(B, 0, DNSPLIT[0], list(NPU_KS), NPU_PASSES, tiled, 384 in O16_MT))
            E.emit(src, ["yah_ssm_postnorm_fp16.head_count=%d" % (DNSPLIT[0] * B)], name, outdir)
        out.append(("npupostnormbfp", 1, 0, 0))
    # unpacks
    for site, n in sorted(NPU_SPLIT.items()):
        skind, mt, kb = NPU_SITES[site]
        N = mt * 16
        forms = {"qkv": [dict(out16=True)], "gate": [dict(out16=True)], "q": [dict(qg=True)],
                 "out": [dict(resid=True, parts=len(npu_chunks("out")), rem=True)],
                 "down": [dict(resid=True, parts=len(npu_chunks("down")), rem=True)],
                 "ffn": [dict(swiglu=True), dict(swiglu=True, tiled=True)]}[site]
        if site == "ffn" and ffn_bfp:
            forms = [dict(f, bfp=(NPU_KS, NPU_PASSES, 1024 * NPU_PASSES, len(npu_chunks("down")))) for f in forms]
        for f in forms:
            name = "npu_unpack_%s%s.hal" % (site, "_t" if f.get("tiled") else "")
            src = os.path.join(tmp, name[:-4] + ".loom")
            open(src, "w").write(GU.gen(B, n // GN.TN, N // 2 if site == "q" else N, N - n, **f))
            E.emit(src, ["nop=0"], name, outdir)
            out.append((name, B * n // 8 // GU.wg(n // GN.TN), GU.wg(n // GN.TN), 0))
        if site == "qkv" and conv_tb_ok(B):
            # the conv reads the NPU's v channels straight from C (gen_conv_kq.gen_c) and the GPU's from the f16 output
            # (gen climit); the unpack writes only the last 8 tokens, the conv ring that prep_ab reads
            for hal, text, cfg in (
                    ("npu_unpack_qkv_tail.hal", GU.gen(B, n // GN.TN, N, N - n, out16=True, last8=True), ["nop=0"]),
                    ("convkq_g.hal", gen_conv_kq.gen(N - n), ["yah_ssm_conv_kq.batch=%d" % B,
                                                            "yah_ssm_conv_kq.qkv_dim=10240",
                                                            "yah_ssm_conv_kq.num_key_heads=16"]),
                    ("convkq_c.hal", gen_conv_kq.gen_c(N - n, n // GN.TN, B), ["nop=0"])):
                src = os.path.join(tmp, hal[:-4] + ".loom")
                open(src, "w").write(text)
                E.emit(src, cfg, hal, outdir)
            out.append(("npu_unpack_qkv_tail.hal", n // GU.wg(n // GN.TN), GU.wg(n // GN.TN), 0))   # 8 tokens x n / 8 items
            out.append(("convkq_g.hal", gen_conv_kq.tiles(N - n), 256, 0))
            out.append(("convkq_c.hal", n // gen_conv_kq.cw(n), gen_conv_kq.cw(n), B // gen_conv_kq.TC))
        out.append(("npusplit_" + site, n, 0, 0))
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
    if NPU_DCOL:
        bfp_site_rows(bsites, outdir)
        out.extend(("npusitebx_" + t, 1, 0, 0) for t, _ in sorted(bsites))
        extra = npu_ffnblk(rows, B, tmp, outdir, out, split, variants) if NPU_FFNBLK else []
        out.extend(npu_dcol_images(dcol_images, B, tmp, outdir, extra))
        GE.ACT_PERM = None
        return out
    # NPU images, one per pass count
    env = dict(hrx_paths.env(), **GN.LOOM_ENV)
    for passes in sorted({p for s in NPU_SPLIT for _, p in npu_chunks(s)}):
        K = 1024 * passes
        cfg = GN.Config(8, B // 64, NPU_KS, passes, gate=GN.GATE_SUPPLY)
        src = os.path.join(tmp, "npu_gemm_%d.loom" % K)
        open(src, "w").write(GN.gen(cfg))
        r = subprocess.run([hrx_paths.LOOM_COMPILE, src, "--root=@" + cfg.entry,
                            "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11",
                            "--output=" + os.path.join(outdir, "npu_gemm_%d.xdna" % K)], capture_output=True, text=True,
                           env=env)
        if r.returncode:
            raise SystemExit("NPU GEMM compile failed: " + r.stderr[-800:])
        out.append(("npubytes_%d" % K,) + tuple(GN.stream_bytes(cfg)))
    out.append(("npurows", NPU_ROWS, 0, 0))
    return out


def npu_dcol_images(images, B, tmp, outdir, extra=()):
    """The decoder-column images (NPU_DCOL): per (format, site K) the gated GEMM reading raw rows of that K, per format
    its decoder swap image; dispatch.txt "npudcol <cols>", "npubytes_5120" and "npurows". extra: more (cfg, name, np)
    images (npu_ffnblk; np: their gen_npu_gemm NP)."""
    import concurrent.futures
    sys.path.insert(0, os.path.dirname(HERE))
    import hrx_paths
    jobs = []
    for fmt, K in sorted(images):
        if NPU_FUSE:
            cfg = GN.Config(NPU_COLS, B // 64, NPU_KS, NPU_PASSES, mu=1, gate=GN.GATE_SUPPLY, gwest=1, ghelp=1, fuse=2, kraw=K,
                            fmt=DCOL_FMTS[fmt][0], **KCHAIN(fmt))
        else:
            cfg = GN.Config(NPU_COLS, B // 64, NPU_KS, NPU_PASSES, gate=GN.GATE_SUPPLY, dcol=2, fmt=DCOL_FMTS[fmt][0],
                            kraw=K)
        jobs.append((cfg, "npu_dcol_%s_%d" % (fmt, K), None, GN.NP))
    for fmt in sorted({f for f, _ in images} if not NPU_FUSE else ()):
        cfg = GN.Config(NPU_COLS, B // 64, NPU_KS, NPU_PASSES, gate=GN.GATE_SUPPLY, dcol=2, fmt=DCOL_FMTS[fmt][0])
        jobs.append((cfg, "npu_dcol_%s.swap" % fmt, "swap", GN.NP))
    jobs += [(cfg, name, mode, np_) for cfg, name, mode, np_ in extra]
    # sources and environments first (gen_npu_gemm's NP is module state), then the compiles in parallel
    def source(cfg, name, mode, np_):
        if cfg.gate and NPU_GFIRST:
            cfg = dataclasses.replace(cfg, gfirst=1)
        with GN.np_override(*(np_ if isinstance(np_, tuple) else (np_,))):
            src = os.path.join(tmp, name + ".loom")
            open(src, "w").write(GN.gen(cfg))
            return cfg, name, src, dict(hrx_paths.env(), **GN.loom_env(cfg, mode == "swap"))

    def compile_one(b):
        cfg, name, src, env = b
        r = subprocess.run([hrx_paths.LOOM_COMPILE, src, "--root=@" + cfg.entry,
                            "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11",
                            "--output=" + os.path.join(outdir, name + ".xdna")], capture_output=True, text=True,
                           env=env)
        if os.environ.get("LOOM_DIAG_PC") and not r.returncode:   # the image's PC ranges for YAH_NPU_PROF_PC
            open(os.path.join(outdir, name + ".pc"), "w").write(
                "\n".join(npu_prof.pc_ranges(src, r.stderr.splitlines())) + "\n")
        return r
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        rs = list(ex.map(compile_one, [source(*j) for j in jobs]))
        # ghelp's helper (a decoder plus the gate's waiter) can exceed the core's program memory: that image with
        # no seam on the helpers, then the helpers' scale halves rolled (hvroll), else without ghelp
        for fix, msg in ((dict(seam=2), "helper without seam"), (dict(hvroll=1), "hvroll"),
                         (dict(hvroll=1, htab=0), "hvroll, helper table per unit"),
                         (dict(ghelp=0, hvroll=0, htab=1), "no ghelp")):
            redo = [i for i, r in enumerate(rs) if r.returncode and jobs[i][0].ghelp and "core program memory" in r.stderr]
            for i in redo:
                jobs[i] = (dataclasses.replace(jobs[i][0], **fix),) + tuple(jobs[i][1:])
                print("%s: %s (program memory)" % (jobs[i][1], msg), file=sys.stderr)
            for i, r in zip(redo, ex.map(compile_one, [source(*jobs[i]) for i in redo])):
                rs[i] = r
    for (cfg, name, _, _), r in zip(jobs, rs):
        if r.returncode:
            raise SystemExit("NPU decoder-column compile failed (%s): %s" % (name, r.stderr[-800:]))
    # a site's K = 5120 call (with no sites, the FFN block's gate / up activations still use its A)
    cfg = jobs[0][0] if images else GN.Config(NPU_COLS, B // 64, NPU_KS, NPU_PASSES, gate=GN.GATE_SUPPLY, dcol=2,
                                              fmt="IQ4_XS")
    out = [("npudcol", NPU_COLS, 0, 0), ("npubytes_5120",) + tuple(GN.stream_bytes(cfg)), ("npurows", NPU_ROWS, 0, 0)]
    return out + ([("npugfirst", 1, 0, 0)] if NPU_GFIRST else [])


def bfp_rows(names, outdir):
    """IQ2_S tensors the NPU takes (DCOL_FMTS "BFP"): each written pre-decoded to outdir/npu_bfp_<name>.bin, rows of
    K / 1024 units of 1152 bytes: unit (16-row slab b, pass u)'s decoder output ([kb 128 in KPERM order][h 2] bfp16
    fragments of the exact f32 weights, gen_npu_dec.oracle) as its 16 rows (the engine binds it like raw rows)."""
    import gguf
    import numpy as np
    import npu_dec_emit as Q
    kperm = GN.decoder("BFP").KPERM
    grid = np.fromfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables", "grid_iq2s.bin"),
                       np.uint8).reshape(1024, 8).astype(np.float32)
    rd = gguf.GGUFReader(MODEL)
    for t in rd.tensors:
        if t.name not in names:
            continue
        path = os.path.join(outdir, "npu_bfp_%s.bin" % t.name)
        assert t.tensor_type.name == "IQ2_S", t.name
        K, N = int(t.shape[0]), int(t.shape[1])
        if os.path.exists(path) and os.path.getsize(path) == N * K // 256 * 288:
            continue
        raw = np.asarray(t.data).view(np.uint8).reshape(N, K // 256, 82)
        with open(path, "wb") as f:
            for b0 in range(0, N, 256):   # 256 rows at a time
                f.write(bfp_units(iq2s_weights(raw[b0:b0 + 256], grid), kperm))
        print("bfp rows: %s (%d x %d) -> %s" % (t.name, N, K, path))


def iq2s_weights(blk, grid):
    """IQ2_S super-blocks [n][sb][82] -> their exact f32 weights [n][sb][256]."""
    import numpy as np
    n, nsb = blk.shape[:2]
    d = blk[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float32)        # [n][sb]
    qs, sg = blk[..., 2:34].astype(np.int64), blk[..., 34:66].astype(np.int64)  # [n][sb][32]
    qh, sc = blk[..., 66:74].astype(np.int64), blk[..., 74:82].astype(np.int64)  # [n][sb][8]
    l = np.arange(32) % 4
    ib = np.arange(32) // 4
    idx = qs | ((qh[..., ib] << (8 - 2 * l)) & 0x300)                            # [n][sb][32]
    s4 = np.stack([sc & 15, sc >> 4], -1).reshape(n, nsb, 16)                   # per 16 values
    dl = (d[..., None] * (np.float32(0.5) + s4.astype(np.float32)) * np.float32(0.25)).astype(np.float32)
    g = grid[idx]                                                                # [n][sb][32][8]
    sign = 1 - 2 * ((sg[..., None] >> np.arange(8)) & 1)                         # [n][sb][32][8]
    return (np.repeat(dl, 16, -1).reshape(n, nsb, 32, 8) * g * sign).astype(np.float32).reshape(n, nsb, 256)


def bfp_units(w, kperm):
    """f32 weights [n (16 | n)][K / 256][256] -> the BFP decoder's rows (bytes): per 16-row slab, its rows of K / 1024
    units of 1152 bytes (unit (slab, pass): [kb 128 in kperm order][h 2] bfp16 fragments)."""
    import npu_dec_emit as Q
    n, K = w.shape[0], w.shape[1] * 256
    w = w.reshape(n // 16, 2, 8, K // 1024, 128, 8).transpose(0, 3, 4, 1, 2, 5)[:, :, kperm]
    units = Q.bfp_hw(w.reshape(-1, 8, 8)).reshape(n // 16, K // 1024, 16, 1152)
    return units.transpose(0, 2, 1, 3).tobytes()


def bfp_site_rows(sites, outdir):
    """YAH_NPU_BFPSITE: the site tensors (name, NPU rows) of formats without an NPU decoder, their last rows (the NPU's)
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
    """YAH_NPU_BFPSW: the gate / up tensors of the pairs in layers (prefixes), their last F rows (the NPU's) decoded to
    their decoders' exact f32 weights (gen_npu_dec.weights; IQ2_S: iq2s_weights) as BFP rows (bfp_units), gate's then
    up's, to npu_bfp_<prefix>ffn_gate_up.bin (the engine's one BFP image binds them like a single format's)."""
    import gguf
    import numpy as np
    kperm = GN.decoder("BFP").KPERM
    grid = np.fromfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables", "grid_iq2s.bin"),
                       np.uint8).reshape(1024, 8).astype(np.float32)
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
                    w = iq2s_weights(blk, grid) if src == "IQ2_S" else GN.decoder(src).weights(blk)
                    f.write(bfp_units(w, kperm))
        os.replace(path + ".tmp", path)
        print("bfp pair rows: %s -> %s" % (pre, path))


def ix4_files(layers, sw, F, outdir):
    """YAH_NPU_IX4: the FFN block's IQ3_S / IQ3_XXS gate / up tensors (layers: prefix -> kinds) transcoded
    (gen_npu_dec.ix4_rows), their last F rows (the NPU's) only: sw layers (prefixes in sw) to
    npu_ix4_<prefix>ffn_gate_up.bin (gate's rows, then up's), the others to npu_ix4_<tensor>.bin (engine Ix4Rows)."""
    import gguf
    import numpy as np
    rd = gguf.GGUFReader(MODEL)
    ts = {t.name: t for t in rd.tensors}
    for pre, kinds in sorted(layers.items()):
        if not kinds:
            continue
        files = [(pre + "ffn_gate_up", ["ffn_gate", "ffn_up"])] if pre in sw else [(pre + k, [k]) for k in kinds]
        # (an sw file's IQ4_XS tensor: its rows as they are)
        for name, ks in files:
            path = os.path.join(outdir, "npu_ix4_%s.bin" % name)
            K = int(ts[pre + ks[0] + ".weight"].shape[0])
            if os.path.exists(path) and os.path.getsize(path) == len(ks) * F * K // 256 * 136:
                continue
            with open(path + ".tmp", "wb") as f:
                for k in ks:
                    t = ts[pre + k + ".weight"]
                    src, N = t.tensor_type.name, int(t.shape[1])
                    raw = np.asarray(t.data).view(np.uint8).reshape(N, K // 256, -1)
                    if src == "IQ4_XS":
                        f.write(raw[N - F:].tobytes())
                        continue
                    D = GN.decoder({"IQ3_S": "IQ4_XO", "IQ3_XXS": "IQ4_XX"}[src])
                    for r0 in range(N - F, N, 512):
                        f.write(D.ix4_rows(raw[r0:r0 + 512], src).tobytes())
            os.replace(path + ".tmp", path)
            print("ix4 rows: %s -> %s" % (name, path))


def npu_ffnblk(rows, B, tmp, outdir, out, split, variants):
    """NPU_FFNBLK: the GPU's kernels of the FFN block (appended to out, split / variants: npu_split's) and its NPU images (returned
    as npu_dcol_images extra jobs); dispatch.txt "npuffnblk <F>", "npubytes_ffnsw" / "npubytes_ffndn<passes>"."""
    import gen_bfp16_encode as GE
    import gen_gemm_tile as TG
    import gen_npu_unpack as GU
    F = NPU_FFNBLK
    assert NPU_DCOL and (F == 7168 or NPU_FUSE and F % 1024 == 0) and "ffn" not in NPU_SPLIT and AFRAG
    assert B % AF_TILE["bn"] == 0
    by = {}
    for nm, dims, ty in rows:
        p = nm.split(".")
        if len(p) > 2 and p[0] == "blk" and p[2] in ("ffn_gate", "ffn_up", "ffn_down") and E.FMT.get(ty):
            by.setdefault(p[1], {})[p[2]] = (E.FMT[ty][0], dims[1] // 16, dims[0] // E.FMT[ty][2])
    fmts, dfmts = (FUSE_FFN_FMTS, FUSE_FFN_DOWN_FMTS) if NPU_FUSE else (FFNBLK_FMTS, FFNBLK_DOWN_FMTS)
    # H's k-block order: the down decoder's (fused: P4 but Q4_K's)
    hord_of = (lambda d: "Q4K" if d == "q4k" else "P4") if NPU_FUSE else (lambda d: FFNBLK_HORD.get(d, ""))
    rows = NPU_COLS * 64 if NPU_FUSE else NPU_ROWS   # down rows per call (fused: NP 4)
    ok = [l for l in by.values() if l.get("ffn_gate") == l.get("ffn_up") and l["ffn_gate"][0] in fmts
          and l["ffn_gate"][0] in AF_FFN and l.get("ffn_down", ("",))[0] in dfmts]
    # gate and up of two formats: column pairs (gen_npu_gemm ufmt; fused sets: F features like the others)
    pfmts, pdfmts, PF = (fmts, dfmts, F) if NPU_FUSE else (FFNBLK_FMTS, FFNBLK_DOWN_FMTS, FFNBLK_PAIR)
    # (also one format without a fused GPU gate / up: its gate kstore and up swiglu, like a pair's)
    pairs = [l for l in by.values() if l.get("ffn_gate") and l.get("ffn_up")
             and (l["ffn_gate"][0] != l["ffn_up"][0] or l["ffn_gate"][0] not in AF_FFN)
             and l["ffn_gate"][1:] == l["ffn_up"][1:] and l["ffn_gate"][0] in pfmts and l["ffn_up"][0] in pfmts
             and l.get("ffn_down", ("",))[0] in pdfmts]
    # pairs whose gate and up both take IQ4_XS's decoder (YAH_NPU_IX4): one image like a single format's, its up slabs
    # gathering from their own kvalues table (gen_npu_gemm kv2), with the deferred epilogue ("npu_ffnsw_<gate>_<up>")
    mixed = [l for l in pairs if NPU_FUSE and {gu_fmt(l[kd][0]) for kd in ("ffn_gate", "ffn_up")} <= set(IX4_FAMILY)]
    same = [l for l in pairs if NPU_FUSE and l["ffn_gate"][0] == l["ffn_up"][0] and l["ffn_gate"][0] in SAME_SW_FMTS]
    bsw = [l for l in pairs if NPU_FUSE and NPU_BFPSW and l not in mixed + same]
    bfp_rows({"blk.%s.%s.weight" % (n, kd) for n, l in by.items() if l in ok + pairs
              for kd in ("ffn_gate", "ffn_up", "ffn_down") if l.get(kd, ("",))[0] == "iq2s"
              and not (l in bsw and kd != "ffn_down")}, outdir)
    bfp_pair_rows({"blk.%s." % n for n, l in by.items() if l in bsw}, F, outdir)
    if NPU_IX4:   # sw layers: gate and up in one file (the image's dgu); pairs: per tensor; the NPU's rows only
        ix4_files({"blk.%s." % n: [kd for kd in ("ffn_gate", "ffn_up") if l[kd][0] in IX4_FMTS]
                   for n, l in by.items() if l in ok + pairs and l not in bsw},
                  {"blk.%s." % n for n, l in by.items() if l in ok + mixed}, F, outdir)
    jobs = []
    for fmt, mt, kb in sorted({l["ffn_gate"] for l in ok}):
        hords = sorted({hord_of(l["ffn_down"][0]) for l in ok if l["ffn_gate"] == (fmt, mt, kb)})
        for hord in hords:
            if hord == hords[0]:
                t = dataclasses.replace(TG.default_tile(fmt, "swiglu", kb), **AF_TILE, **AF_FFN[fmt], ffn=True)
                for sfx, tt in ((".af.hal", t), (".af.to.hal", dataclasses.replace(t, tout=True))):
                    split(fmt, "ffn", mt, kb, tt, "gemm_ffn_%s_%d_%d%s" % (fmt, mt, kb, sfx), F)
            blk = GN.decoder(gu_fmt(fmt)).BLK
            name = "npu_ffnsw_%s%s" % (fmt, hord and "_" + hord.lower())
            if NPU_FUSE:
                dfr = fmt in NPU_DFR_FMTS
                jobs.append((GN.Config(NPU_COLS, B // 64, NPU_DFR_KS if dfr else NPU_KS, NPU_PASSES, mu=1,
                                       gate=GN.GATE_SUPPLY, gwest=1, ghelp=1, fuse=2, swiglu=True, dgu=mt * 16 * kb * blk, swcol=32,
                                       hpass=F // 1024, hord=hord, fmt=gu_fmt(fmt), dfr=int(dfr)), name, None, 4))
                continue
            jobs.append((GN.Config(NPU_COLS, B // 64, NPU_KS, NPU_PASSES, gate=GN.GATE_SUPPLY, dcol=2,
                                   fmt=DCOL_FMTS[fmt][0], swiglu=True, dgu=mt * 16 * kb * blk, swcol=1024,
                                   hpass=F // 1024, hord=hord), name, None, 4))
            jobs.append((jobs[-1][0], name + ".swap", "swap", 4))
    parts = ffnblk_parts(F)
    for fmt, mt, kb, kbw in sorted({l["ffn_down"] + ((l["ffn_down"][2] * 256 - F) // 256,) for l in ok} |
                                   {l["ffn_down"] + ((l["ffn_down"][2] * 256 - PF) // 256,) for l in pairs}):
        if NPU_FUSE and NPU_FFNRES:   # the GPU's K window as the residual GEMM, adding the NPU's down partials
            with GN.np_override(4):
                dcfg = GN.Config(NPU_COLS, B // 64, NPU_KS, parts[0][1], mu=1, gate=GN.GATE_SUPPLY, gwest=1, ghelp=1, fuse=2,
                                 kraw=kb * 256, ain="h", hpass=F // 1024, fmt=DCOL_FMTS[fmt][0])
                cp, tn, calls = GN.stream_bytes(dcfg)[2], GN.TN, -(-5120 // rows)
            assert rows == NPU_COLS * tn and 5120 % rows == 0
            tr = fit_rowgrp(npu_rem_tile(fmt, kbw, kb, "kres"), mt)
            add = dict(rows=rows, tn=tn, np=4, call=cp // 2, calls=calls, parts=len(parts), tokens=B)

            def gen_res(f, k, tr=tr, kb=kb, add=add):
                TG.KWIN, TG.NPUADD = (0, kb), add
                try:
                    return TG.gen(f, "kres", tr, False)
                finally:
                    TG.KWIN = TG.NPUADD = None
            out.append(_emit_gen(gen_res, tr.bn, fmt, mt, kbw, B, "npuffnres_%s_%d.hal" % (fmt, kbw), outdir, "kres",
                                 tr.rowgrp, kfull=kb, npuc=len(parts) * calls * cp))
            continue
        # the GPU's K window [0, 17408 - F)
        tr = fit_rowgrp(npu_rem_tile(fmt, kbw, kb), mt)

        def gen_rem(f, k, tr=tr, kb=kb):
            TG.KWIN = (0, kb)
            try:
                return TG.gen(f, "kstore", tr, False)
            finally:
                TG.KWIN = None
        assert mt not in O16_MT, "the window writes f32"
        out.append(_emit_gen(gen_rem, tr.bn, fmt, mt, kbw, B, "npuffnrem_%s_%d.hal" % (fmt, kbw), outdir, "kstore",
                             tr.rowgrp, kfull=kb))
    for fmt, mt, kb in sorted({l["ffn_down"] for l in ok} | {l["ffn_down"] for l in pairs}):
        for passes in sorted({p for _, p in parts + (ffnblk_parts(PF) if pairs else ())}):
            if NPU_FUSE:
                jobs.append((GN.Config(NPU_COLS, B // 64, NPU_KS, passes, mu=1, gate=GN.GATE_SUPPLY, gwest=1, ghelp=1, fuse=2,
                                       kraw=kb * 256, ain="h", hpass=F // 1024, fmt=DCOL_FMTS[fmt][0],
                                       **KCHAIN(fmt)),
                             "npu_ffndn%d_%s" % (passes, fmt), None, 4))
                continue
            jobs.append((GN.Config(NPU_COLS, B // 64, NPU_KS, passes, gate=GN.GATE_SUPPLY, dcol=2, fmt=DCOL_FMTS[fmt][0],
                                   kraw=kb * 256, ain="h", hpass=F // 1024, gord="N"), "npu_ffndn%d_%s" % (passes, fmt),
                         None, GN.NP))
            jobs.append((jobs[-1][0], "npu_ffndn%d_%s.swap" % (passes, fmt), "swap", GN.NP))
    for (gf, mt, kb), (uf, _, _), hord in sorted({(l["ffn_gate"], l["ffn_up"], hord_of(l["ffn_down"][0]) if NPU_FUSE
                                                   else FFNBLK_HORD.get(l["ffn_down"][0], "")) for l in pairs}):
        # the GPU's other features: gate kstore and up swiglu, split (".npu.hal", nn = PF)
        variants(gf, "kstore", mt, kb, "gemm_kstore_%s_%d_%d.hal" % (gf, mt, kb), PF)
        variants(uf, "swiglu", mt, kb, "gemm_swiglu_%s_%d_%d.hal" % (uf, mt, kb), PF)
        name = "npu_ffnsp_%s_%s%s" % (gf, uf, hord and "_" + hord.lower())
        if any(l in bsw for l in pairs if (l["ffn_gate"], l["ffn_up"]) == ((gf, mt, kb), (uf, mt, kb))):   # (BFP)
            if "npu_ffnsw_bfp_" + hord.lower() not in {j[1] for j in jobs}:
                jobs.append((GN.Config(NPU_COLS, B // 64, NPU_DFR_KS, NPU_PASSES, mu=1, gate=GN.GATE_SUPPLY, gwest=1,
                                       ghelp=1, fuse=2, swiglu=True, dgu=mt * 16 * kb * 288, swcol=32, hpass=F // 1024,
                                       hord=hord, fmt="BFP", dfr=1), "npu_ffnsw_bfp_" + hord.lower(), None, 4))
            continue
        if NPU_FUSE and ({gu_fmt(gf), gu_fmt(uf)} <= set(IX4_FAMILY) or gf == uf and gf in SAME_SW_FMTS):   # (mixed)
            blk = GN.decoder(gu_fmt(gf)).BLK
            jobs.append((GN.Config(NPU_COLS, B // 64, NPU_DFR_KS, NPU_PASSES, mu=1, gate=GN.GATE_SUPPLY, gwest=1, ghelp=1,
                                   fuse=2, swiglu=True, dgu=mt * 16 * kb * blk, swcol=32, hpass=F // 1024, hord=hord,
                                   fmt=gu_fmt(gf), kv2=gu_fmt(uf) if gf != uf else "", dfr=1),
                         "npu_ffnsw_%s_%s%s" % (gf, uf, hord and "_" + hord.lower()), None, 4))
            continue
        if NPU_FUSE:   # pair u: gate column 2 u, up column 2 u + 1, features 256 j + 64 u + [0, 64) of call j
            jobs.append((GN.Config(NPU_COLS, B // 64, NPU_KS, NPU_PASSES, mu=1, gate=GN.GATE_SUPPLY, gwest=1, ghelp=1, fuse=2, swiglu=True,
                                   pnbr=1, swcol=64, hpass=F // 1024, hord=hord, fmt=gu_fmt(gf), ufmt=gu_fmt(uf),
                                   gord="GF" if "q4k" in (gf, uf) and hord == "P4" else "P4"),
                         name, None, (4, True)))
            continue
        jobs.append((GN.Config(6, B // 64, NPU_KS, NPU_PASSES, gate=GN.GATE_SUPPLY, dcol=2, fmt=DCOL_FMTS[gf][0],
                               ufmt=DCOL_FMTS[uf][0], swiglu=True, swcol=2048, hpass=F // 1024, hord=hord),
                     name, None, (4, True)))
        jobs.append((jobs[-1][0], name + ".swap", "swap", (4, True)))
    # hidden + the GPU's window + the NPU's down partials (per K part, calls of rows rows, one replay group each); fused
    # sets add them in the GPU window's residual epilogue (npuffnres) unless NPU_FFNRES is off
    N, calls = 5120, -(-5120 // rows)
    assert not pairs or len(ffnblk_parts(PF)) == len(parts)
    # token halves (LoomPrefill halves): "_h<i>" unpacks half i's tokens of the same C
    if not (NPU_FUSE and NPU_FFNRES):
        for name, tr in (("npu_unpack_ffnblk", None),) + tuple(("npu_unpack_ffnblk_h%d" % h, (h * B // 2, B // 2))
                                                                for h in range(2) if npu_halves(B)):
            cg = GU.CG
            GU.CG = 1
            try:
                with GN.np_override(4 if NPU_FUSE else GN.NP):
                    text = GU.gen(B, calls * NPU_COLS, N, 0, resid=True, parts=len(parts), rem=True,
                                  dup=((calls - 1) * NPU_COLS, N - rows), trange=tr)
                    nj, wg = GN.TN // 8, GU.wg(calls * NPU_COLS)
            finally:
                GU.CG = cg
            src = os.path.join(tmp, name + ".loom")
            open(src, "w").write(text)
            E.emit(src, ["nop=0"], name + ".hal", outdir)
            out.append((name + ".hal", (tr[1] if tr else B) * calls * NPU_COLS * nj // wg, wg, 0))
    # gate / up read their activations in one replay group: the norm before them writes that stream (fragment-major)
    import gen_half_norm
    groups, perm = GE.GROUPS, GE.ACT_PERM
    dfrs = sorted({j[0].fmt for j in jobs if j[0].dfr})
    # "_h<i>": half i's rows (token halves), into the whole chunk's stream
    for name, ks in (("norm_t_bfp_g1", NPU_KS),) + ((("norm_t_bfp_g1s", NPU_DFR_KS),) if dfrs else ()):
        for sfx, part in (("", None),) + tuple(("_h%d" % h, (h * B // 2, B // 2)) for h in range(2) if npu_halves(B)):
            GE.GROUPS, GE.ACT_PERM = (NPU_PASSES,), "p4"
            try:
                src = os.path.join(tmp, name + sfx + ".loom")
                open(src, "w").write(gen_half_norm.gen_split(5120, wpr=4, split=NORM_SPLIT, tiled=True,
                                                             bfp=(B, list(ks), NPU_PASSES), bfp_part=part))
            finally:
                GE.GROUPS, GE.ACT_PERM = groups, perm
            E.emit(src, ["yah_half_norm.rows=%d" % (part[1] if part else B), "yah_half_norm.dim=5120",
                         "yah_half_norm.eps=1e-06", "yah_half_norm.fused=0"], name + sfx + ".hal", outdir)
    for fmt in NPU_DFR_FMTS:
        if gu_fmt(fmt) in dfrs:
            out.append(("npuffndfr_" + fmt, 1, 0, 0))
    if NPU_IX4:
        out.append(("npuffnix4", 1, 0, 0))
    for gf, uf in sorted({(l["ffn_gate"][0], l["ffn_up"][0]) for l in mixed + same + bsw}):
        out.append(("npuffndfr_%s_%s" % (gf, uf), 1, 0, 0))
    for gf, uf in sorted({(l["ffn_gate"][0], l["ffn_up"][0]) for l in bsw}):
        out.append(("npuffnbx_%s_%s" % (gf, uf), 1, 0, 0))
    out.append(("npuffnblk", F, 256 if NPU_FUSE else 224, rows))   # F, gate / up features per call, down rows
    sw = next(j for j in jobs if j[1].startswith("npu_ffnsw_"))
    with GN.np_override(4):
        out.append(("npubytes_ffnsw",) + tuple(GN.stream_bytes(sw[0])))
    for passes in sorted({p for _, p in parts + (ffnblk_parts(PF) if pairs else ())}):
        dn = next(j for j in jobs if j[1].startswith("npu_ffndn%d_" % passes))
        with GN.np_override(dn[3]):
            out.append(("npubytes_ffndn%d" % passes,) + tuple(GN.stream_bytes(dn[0])))
    sp = next((j for j in jobs if j[1].startswith("npu_ffnsp_")), None)
    if sp:
        with GN.np_override(4, True):
            out.append(("npubytes_ffnsp",) + tuple(GN.stream_bytes(sp[0])))
    if pairs:   # (also the mixed / BFP pairs': their GPU side's features)
        out.append(("npuffnpair", PF, 0, 0))
    return jobs


def tile_kstore(fmt, mt, kb, B, out, outdir, kind, geom=None):
    """Emit the tile GEMM (tools/gen_gemm_tile.py) for this shape if it covers it; return its dispatch.txt row, else None.
    geom=(BM, BN, WM, WN) overrides the workgroup geometry for this one kernel."""
    import gen_gemm_tile as TG
    if fmt not in TILE_FMTS:
        return None
    t = TG.default_tile(fmt, kind, kb, geom)
    if not geom and out[:-4] in tiles()["tiles"]:
        t = dataclasses.replace(t, **tiles()["tiles"][out[:-4]])
    if mt % t.rowgrp:
        return None
    # a token tile that does not divide the chunk: the last tile is masked (gen_gemm_tile.gen masked=)
    masked = B % t.bn != 0
    return _emit_gen(lambda f, k: TG.gen(f, k, t, masked), t.bn, fmt, mt, kb, B, out, outdir, kind, t.rowgrp, masked)


def _emit_gen(gen, tile, fmt, mt, kb, B, out, outdir, kind, rowgrp, masked=False, ostride=0, sym=None, kfull=0, npuc=0):
    """ostride: a GEMM over the leading mt * 16 rows of an ostride-row output (gen_gemm_tile.OSTRIDE, the NPU split's GPU share);
    f16 output then follows the full shape (O16_MT). sym: the kernel symbol, if not the GEMM's or the dequant's.
    kfull: a K-chunk dequant's whole-row k_blocks (gen_gemm_tile.DQ_BFP). npuc: the bytes of NPU C a gen_gemm_tile.NPUADD
    epilogue reads."""
    if B % tile and not masked:
        return None
    import gen_gemm_tile as _TG
    _TG.OUT16 = kind == "kstore" and (ostride // 16 if ostride else mt) in O16_MT
    _TG.OSTRIDE = ostride
    tt = -(-B // tile)
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(tmp, "yah_sgemm_%s_%s.loom" % (fmt, kind))
    try:
        with open(src, "w") as fh:
            fh.write(gen(fmt, kind))
    finally:
        _TG.OSTRIDE = 0
    if sym is None:
        sym = "yah_ffn_gemm_%s%s" % (fmt.replace(":", "_"), {"swiglu": "_swiglu", "kres": "_kres", "kqg": "_kqg", "ffn": "_ffn"}.get(kind, ""))
        if kind == "dequant":
            sym = "yah_dequant_" + fmt
    # Refuse before emitting if any declared operand footprint exceeds the buffer the driver binds: an overrun hangs the ring.
    gate = [sym, fmt, kind, str(mt), str(kb), str(tt), str(B)] + (["masked"] if masked else []) \
        + (["ostride=%d" % ostride] if ostride else []) + (["kfull=%d" % kfull] if kfull else []) \
        + (["npuc=%d" % npuc] if npuc else [])
    E.emit(src, ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb), "%s.token_tiles=%d" % (sym, tt)]
           + (["%s.tokens=%d" % (sym, B)] if masked else []), out, outdir, gate=gate)
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
    E.emit(src, ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb), "%s.token_tiles=%d" % (sym, tt)], out, outdir,
           gate=[sym, "mq4", kind, str(mt), str(kb), str(tt), str(B), "aspan=%d" % gen_gemm_iu4.KP])
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
    """HAL kinds emitted for a combo, in order; the driver runs the last one (kres over kstore, kqg over kstore)."""
    return {"kstore": ["kstore"] + (["kqg"] if mt == 768 else []), "residual": ["kstore", "kres"],
            "swiglu": ["swiglu"]}[kind]


def main():
    global MODEL, NPU_FFNBLK, QG16_ON, GSIG_ON, QSPLIT_ON, KV16_ON, NQOV_ON, DNOV_ON, GUOV_ON, NQGU_ON, RQK_ON, NQW_ON, DNRING_ON, ROPEW_ON
    model, outdir = sys.argv[1], sys.argv[2]
    MODEL = model
    E.ASYNC = True
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
    # MQ4 weights (HFQ, or GGUF converted at load): A4 prefill (int4 activations, iu4 GEMMs) unless YAH_A16=1 (W4A16: f16
    # activations, the quality mode). YAH_NATIVE=1 (a GGUF loaded as it is): W4A16 on the file's own formats, no rotation.
    rot = types.get("blk.0.ffn_down.weight") == 44
    if rot and (types.get("output.weight") != 44 or types["token_embd.weight"] != 8):
        raise SystemExit("emit_prefill_pp: the model's weights are not MQ4G256 as yah-dump --table shows them")
    if not rot and os.environ.get("YAH_NATIVE") != "1":
        raise SystemExit("emit_prefill_pp: not MQ4 weights; a GGUF loaded natively needs YAH_NATIVE=1")
    a4 = rot and os.environ.get("YAH_A16") != "1"
    # R2P_LAZY (gen_attn_fa: lazy rescale threshold, nats) 5 on native and A16 sets, as the npu branch: 2 x 32 busiest
    # SQ +0.8 at 126K on real kv8 inputs, native kld_ref 0.0164 = exact. A4 keeps the exact kernel (kld_ref 0.0675 ->
    # 0.0692, npu_gate over its limit)
    if "R2P_LAZY" not in os.environ:
        gen_attn_fa.R2P_LAZY = 0.0 if a4 else 5.0
    KV16_ON = a4
    NQOV_ON = a4
    GUOV_ON = a4 and not EMIT_HALF
    NQGU_ON = GUOV_ON
    RQK_ON = (NQOV_ON and GUOV_ON)
    NQW_ON = a4 and not EMIT_HALF
    # the engine runs ropew only with fp16 paged K and V (RoPE writes both pools; quantized KV has its own staging)
    ROPEW_ON = ROPEW and NQW_ON and QNORM and B % gen_rot.FLAG_TT == 0 and gen_kvq.kv_bits() == (16, 16) and T % 256 == 0
    DNOV_ON = (DNOV and a4 and DNSPLIT == (40, 8) and DNVCONV and conv_tb_ok(B) and "qkv" not in NPU_SPLIT
               and not EMIT_HALF)   # the dnvconv conv (dn_vconv below)
    QG16_ON = QG16 and a4 and QNORM and gen_attn_fa.Q16 and gen_attn_fa.QDIRECT is not False
    GSIG_ON = bool(GSIG) and not (QG16_ON or EMIT_HALF or "q" in NPU_SPLIT)
    if GSIG_ON:
        import gen_gemm_tile
        assert GSIG in ("u16", "16", "32"), GSIG
        gen_gemm_tile.KQG_GSIG, gen_gemm_tile.KQG_GSIG16 = GSIG, GSIG in ("16", "u16")
        gen_attn_fa.GSIG, gen_attn_fa.GSIG16 = GSIG, GSIG in ("16", "u16")
        os.environ["KQG_GSIG"] = os.environ["GSIG"] = GSIG   # (the calibration menu's worker processes)
    QSPLIT_ON = QSPLIT and GSIG_ON and GSIG in ("16", "u16") and QNORM and gen_attn_fa.Q16 and not a4
    if QSPLIT_ON:
        import gen_gemm_tile
        gen_gemm_tile.KQG_QSPLIT = gen_attn_fa.R2P_QSPLIT = True
        os.environ["KQG_QSPLIT"] = os.environ["R2P_QSPLIT"] = "1"

    n = 0
    geom = []  # (<hal>, <tokens per workgroup>, <row groups>, <token_tiles>)
    if EMIT_HALF:   # a half set: no GEMMs of its own (the NPU split's GPU shares come from npu_split)
        NPU_FFNBLK = 0
        assert not {"ffn", "down"} & set(NPU_SPLIT) and T % (2 * B) == 0
    for kind, fmt, port, mt, kb in sorted(combos) if not EMIT_HALF else ():
        name = lambda k: "gemm_%s_%s_%d_%d.hal" % (k, fmt, mt, kb)
        # A residual projection gets a kstore and the fused-residual kres; the attention q projection (12288 rows =
        # 24 heads x [q | gate]) also gets kqg, which stores q and gate unpacked. The driver prefers kres and kqg.
        kinds = gemm_kinds(kind, mt)
        if a4 and fmt == "mq4":
            # A4: the FFN's gate and up run as one gu GEMM (on the up combo); the gate tensors need no kstore
            if all(t.endswith("ffn_gate.weight") for t in combos[(kind, fmt, port, mt, kb)]):
                continue
            for k in (["gu"] if kind == "swiglu" else kinds):
                if k != "kstore" or kind != "residual":
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
                geom.extend(afrag_variants(fmt, mt, kb, B, name(k), outdir, k))
            elif k == "kstore" and fmt == "q2k":
                # Q2_K has no tile decoder; its only tensors here are the 48-row ssm_alpha / ssm_beta.
                sym = E.sym_of("yah_ffn_gemm_q2k_f32.loom")
                E.emit("yah_ffn_gemm_q2k_f32.loom", ["%s.m_tiles=%d" % (sym, mt), "%s.k_blocks=%d" % (sym, kb),
                                                     "%s.token_tiles=%d" % (sym, TT)], name(k), outdir)
                geom.append((name(k), TILE, 1, TT))
            elif k == kinds[0]:
                raise SystemExit("no GEMM kernel for %s %s (%d rows, K = %d)" % (fmt, k, mt * 16, kb * qk_of(fmt)))
        n += 1

    if not EMIT_HALF:
        geom.extend(ffn_fused(rows, B, outdir))
        if KST2 and not a4:
            geom.extend(kst2(rows, B, outdir))
        if SEG and not a4:
            geom.extend(seg_gemms(rows, B, outdir))
    if NPU_SPLIT or NPU_FFNBLK:
        import gen_bfp16_encode as GE
        import gen_npu_unpack as GU
        # fused sets: one replay group per call (the fillers' W-DMA raw ring hangs with two), so C is one partial
        groups, cg = GE.GROUPS, GU.CG
        if NPU_FUSE:
            GE.GROUPS, GU.CG = (NPU_PASSES,), 1
        try:
            with GN.np_override(NPU_NP):
                geom.extend(npu_split(rows, combos, B, outdir))
        finally:
            GE.GROUPS, GU.CG = groups, cg


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
    # replaced (KT2, PAIR "deep") ran 75-77 / 73. Not the OCNT kernel (wmmaf.hal: its counters feed a concurrent
    # consumer) nor half sets (the grid kernel: 85.0% to 96K, 73.2% at 124K).
    persist = 0 if EMIT_HALF else int(os.environ.get("YAH_ATTN_PERSIST", str(320 // gen_attn_fa.NQB)))   # 20 WGPs x 4 (2x64: x 2)
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
    gen_attn_fa.QG16 = QG16_ON
    if QG16_ON:
        geom.append(("qg16", 0, 0, 0))
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
    attn_f_src = os.path.join(tmp, "yah_attn_hipf.loom")
    aov_need = 0
    if AOV and a4 and RQOP and B % gen_rot.FLAG_TT == 0 and not EMIT_HALF:   # the main kernel with OCNT
        A = gen_attn_fa
        saved = {k: getattr(A, k) for k in ("OCNT",) + tuple(MAIN_FLAGS)}
        for k, v in dict(MAIN_FLAGS, OCNT=True, PERSIST=0, QB_ZIGZAG=True).items():
            setattr(A, k, v)
        open(attn_f_src, "w").write(A.gen())
        aov_need = A.ocnt_need()
        for k, v in saved.items():
            setattr(A, k, v)
    # afrag o-projections (K = 6144 kres) read the attention output fragment-major: wmma_t[_c<i>].hal
    af24 = any(h.startswith("gemm_kres_") and h.endswith("_320_24.af.hal") for h, *_ in geom)
    if EMIT_HALF:   # the whole set's (its o-projections' afrag forms)
        af24 = AFRAG and B % AF_TILE["bn"] == 0 and any(
            AF.get((fmt, "kres", mt, kb)) is not None for kind, fmt, port, mt, kb in combos
            if kind == "residual" and (mt, kb) == (320, 24))
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
        assert w64_from <= ks_from and pdyn and not EMIT_HALF
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
            if NPU_SPLIT or NPU_FFNBLK:
                sys.exit("YAH_NPU_ATTN: a set splits either the GEMMs or the attention with the NPU")
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
    # It needs B % 32 == 0; else the recurrent kernel (tools/gen_deltanet_hip.py, same ABI and grid).
    dn_src = os.path.join(tmp, "yah_deltanet_hip_f32.loom")
    open(dn_src, "w").write(gen_gdn_chunk.gen() if dn32(B) else gen_deltanet_hip.gen())
    # the conv pairs with the DeltaNet kernel: f16 normalized hand-off to the chunked one, f32 conv_out otherwise
    # the f16 hand-off conv per (256 channels, gen_conv_kq.TB tokens) workgroup: bit-identical, inputs read once
    conv_tb = conv_tb_ok(B)
    dn_vconv = (DNVCONV and conv_tb and (a4 or (DNSS_NAT and B % 32 == 0)) and DNSPLIT == (40, 8) and "qkv" not in NPU_SPLIT
                and not EMIT_HALF)
    if dn_vconv:
        text = gen_conv_kq.gen(4096)
        geom.append(("dnvconv", gen_conv_kq.tiles(4096), 0, 0))
    elif conv_tb:
        text = gen_conv_kq.gen()
    else:
        text = open(os.path.join(E.LOOM, "yah_ssm_conv_kq_f32.loom")).read()
        if 640 in O16_MT:
            text = conv_x16(text)
        if dn32(B) and gen_gdn_chunk.CONV16:
            text = conv_n16(text)
    conv_src = os.path.join(tmp, "yah_ssm_conv_kq.loom")
    open(conv_src, "w").write(text)
    assert DNOV_ON <= dn_vconv
    convf_src = os.path.join(tmp, "yah_ssm_conv_kqf.loom")
    if DNOV_ON:
        open(convf_src, "w").write(gen_conv_kq.gen(4096, flag=gen_gemm_iu4.flag_count(10240)))
        geom.append(("dnov", 0, 0, 0))
    convp_src = os.path.join(tmp, "yah_ssm_conv_kq_p.loom")
    convp = CONVP and DNOV_ON and B % gen_rot.FLAG_TT == 0
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
    dn_ss = DNSS and dn_perm and dn_vconv and DNO16 and B % 32 == 0
    dn_nat = not a4 and dn_vconv   # DNSS_NAT
    dn_ss = dn_ss or dn_nat
    # one flagged pnq for both DeltaNet parts (gen_rot.gen_pnq "ab"): part b's DeltaNet counts its finished lanes
    pnq_ab = DNOV_ON and dn_ss
    DNRING_ON = DNRING and pnq_ab and RQK_ON and NQW_ON and DNSPLIT == (40, 8)
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
    if dn32(B) and DNSPLIT:
        ha, hb = DNSPLIT
        for tag, h0, nh in (("a", 0, ha), ("b", ha, hb)):
            src = os.path.join(tmp, f"yah_deltanet_{tag}.loom")
            if dn_perm:
                assert len(gen_rot.pn_heads(tag)) == nh
                if dn_ss:
                    open(src, "w").write(gen_gdn_ss.gen(gen_rot.pn_heads(tag), flag=pnq_ab and tag == "b"))
                else:
                    open(src, "w").write(gen_gdn_chunk.gen(heads=gen_rot.pn_heads(tag), vconv=dn_vconv, o16=DNO16))
            elif dn_nat:
                open(src, "w").write(gen_gdn_ss.gen(list(range(h0, h0 + nh)), natural=True))
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
                           kvn=npu_kv, qlay=0 if QSPLIT_ON else 1)
            call.check()
            for kind, gen in (("k", NE.gen_k), ("v", NE.gen_v), ("q", NE.gen_q)):
                src = os.path.join(tmp, "npua%s_c%d.loom" % (kind, c))
                open(src, "w").write(gen(call))
                npua.append((src, "npua%s_c%d.hal" % (kind, c)))
        src = os.path.join(tmp, "npu_flag_wait.loom")
        open(src, "w").write(GU.gen_flag_wait())
        npua.append((src, "npu_flag_wait.hal"))
        img = os.path.join(tmp, "npu_attn.loom")
        open(img, "w").write(gen_npu_attn.emit(16, npu_nbs))
        gen_npu_attn.compile_xdna(img, os.path.join(outdir, "npu_attn.xdna"))
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
        kv = ROPEKV and KV16_ON and vtrope and B % 16 == 0
        open(rope_src, "w").write(rope_kv(konly) if kv else konly)
        geom.append(("qnorm", 0, 0, 0))
        if kv:
            geom.append(("ropekv", 0, 0, 0))
        if ROPEW_ON:
            need = gen_gemm_iu4.flag_count(1024)
            open(ropew_src, "w").write(rope_kv(konly, need) if kv else rope_wait(konly, need))
            geom.append(("ropew", 0, 0, 0))
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
    # MQ4 GEMMs read their input in the weights' rotated basis (gen_rot): the driver rotates each GEMM input site in place,
    # the DeltaNet postnorm's out of place (dst in HF head order); the head's f32 norm row without awq.
    # A4 sets rotate and quantize instead (rotq_*: int4 activations with one scale per 128 values, the iu4 GEMM's K phase;
    # rows "<hal> 0 <lanes> <workgroups>"). Per-token scales for the norm-fed sites and 512-value spans for the others cost
    # WT2 KL 0.064 vs 0.051 (2 chunks; p99.9 2.08 vs 1.01).
    rot_src = {}
    if rot:
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
                    open(rot_src[name], "w").write(gen_rot.gen_pnq(B, gen_gemm_iu4.KP, part, o16=DNO16))   # gate: kstoreh
                    geom.append((name, 0, 64, len(gen_rot.pnq_groups(part)) // 4))
                if DNOV_ON:   # part a waiting on the gate GEMM's counters (pnq_a.hal's geometry)
                    rot_src["pnq_af.hal"] = os.path.join(tmp, "pnq_af.loom")
                    open(rot_src["pnq_af.hal"], "w").write(
                        gen_rot.gen_pnq(B, gen_gemm_iu4.KP, "a", o16=DNO16, flag=gen_gemm_iu4.flag_count(6144)))
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
    else:
        geom.append(("native", 0, 0, 0))   # the engine loads the GGUF as it is (core/gguf_convert.cpp native)
    # the output head: the tile-free GEMV (gen_gemv) for MQ4 or Q8_0 (a native GGUF's head that is not Q6_K, which
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
    for tag, h0, nh, _ in ([] if dn_perm else dn_parts):
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
        ("yah_unpack_qg_f32.loom", "unpack.hal",
         ["yah_unpack_qg.batch=%d" % B, "yah_unpack_qg.num_heads=24",
          "yah_unpack_qg.head_dim=256"]),
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
        *([(attn_f_src, "wmmaf.hal", [
            "attention_prefill.cache_capacity=%d" % T,
            "attention_prefill.token_count=%d" % B,
            "attention_prefill.start_pos=0",
            "attention_prefill.num_heads=24", "attention_prefill.num_kv_heads=4",
            "attention_prefill.head_dim=256", "attention_prefill.gqa=6"])] if aov_need and qrot_first > 0 else []),
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
        *[(src, name, ["nop=0"]) for name, src in rot_src.items()],
        ("yah_argmax_f32.loom", "argmax.hal", ["yah_argmax.vocab=248320"]),
    ]
    # Loom low-level LICM (HRX patch 0011) for the attention kernels where it wins: it hoists the fragment loads'
    # per-lane LDS address math out of the key loop (fp16 -1..-2%; kv4 QROT -0.5% with one extra live register); the
    # kv4 staged-Q build loses 11% and the afrag GEMMs 5-20%, so they compile without it
    ksn_twins = [x for pair in ksn_srcs.values() for x in pair if x]
    def attn_env(loom):
        if loom in (attn_qr_src, attn_qr_t_src):   # fp16 paged under kvdeq, else the kv4 direct-Q kernel
            return {"LOOM_EXP_LICM": "1"} if kvdeq else {"LOOM_EXP_LICM": "1", "LOOM_EXP_LICM_MAX_LIVE": "1"}
        if loom in (attn_src, attn_t_src, attn_f_src, attn_c0_src, attn_t_c0_src, attn64_src, attn64_t_src,
                    attn64ks_src, attn64ks_t_src, attn64km_src, attn64km_t_src, *ksn_twins) and loom is not None and (
                not gen_attn_fa.KDEC or kv8deq):
            return {"LOOM_EXP_LICM": "1"}
        return None
    if EMIT_HALF:   # the DeltaNet / attention chain of a half (the norms, unpacks and conv halves come from npu_split)
        fixed = [f for f in fixed if f[1].startswith(("prepab", "rowsplit", "postnorm", "rope", "vtpage", "wmma"))]
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
        if outname.startswith("wmma") and persist and outname != "wmmaf.hal":
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
    for src, dst in [] if EMIT_HALF else [("grid_iq3s.bin", "grid_iq3s.bin"),
                     ("grid_iq3xxs.bin", "grid_iq3xxs.bin"),
                     ("grid_iq2xxs.bin", "grid_iq2xxs.bin"),
                     ("grid_iq2xs.bin", "grid_iq2xs.bin"),
                     ("grid_iq2s.bin", "grid_iq2s.bin"),
                     ("ksigns_iq2xs.bin", "ksigns_iq2xxs.bin")]:
        shutil.copy(os.path.join(tables, src), os.path.join(outdir, dst))
    E.wait()
    print("emitted %d GEMM + fixed prefill HALs for B=%d (tile=%d, token_tiles=%d)"
          % (n, B, TILE, TT))
    if npu_halves(B) and not EMIT_HALF:
        env = {k: v for k, v in os.environ.items() if k != "YAH_NPU_FFNBLK"}
        env.update(YAH_EMIT_HALF="1", YAH_CTX=str(T))
        subprocess.run([sys.executable, os.path.abspath(__file__), model, os.path.join(outdir, "h"), str(B // 2)], env=env, check=True)


if __name__ == "__main__":
    main()
