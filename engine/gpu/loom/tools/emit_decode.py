#!/usr/bin/env python3
"""Emit every HAL the Loom single-token decode forward needs for a shard.

usage: emit_decode.py <model.gguf> <outdir> [max_context]       (default 4096, multiple of 256)

  gb_<fmts>_<Ms>_<K>.hal         tools/gen_gemv.py gen_bands: a layer's input projections
                                 in one dispatch (the decoder's default)
  gv_<kind>_<fmts>_<M>_<K>.hal   tools/gen_gemv.py, one per distinct (kind, formats,
                                 shape) on the shard: plain for the input projections
                                 and the head, resid for attn_output / ssm_out /
                                 ffn_down (the residual add fused), swiglu for
                                 ffn_gate + ffn_up
  dattn_{kvappend,part,reduce}   tools/gen_decode_attn.py: attention over the paged
                                 fp16 KV pools (the prefill's YAH_KV_PAGED layout)
  rmsnorm, deltanet[_conv]       tools/gen_decode_misc.py (the ports' math, 512 lanes;
                                 deltanet_conv also runs the decode conv, the default)
  unpack, rope, ssmconv, argmax: the ported HIP decode kernels.
                                 rope's own cache write goes to a one-row dummy
                                 (max_context=1); dattn_kvappend writes the pools.
  grid_*.bin, ksigns_iq2xs.bin   the IQ tables (loom/tables)
  decode.txt                     "ctx <max_context>"
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOOM = os.path.abspath(os.path.join(HERE, ".."))
EMIT = os.path.join(LOOM, "emit_hal.py")
sys.path.insert(0, HERE)
import emit_prefill as EP  # noqa: E402  (GGUF tensor table)
import gen_decode_attn as DA  # noqa: E402
import gen_decode_misc as DM  # noqa: E402
import gen_gemv as GV  # noqa: E402

NUM_HEADS, NUM_KV, HEAD_DIM, ROTARY = 24, 4, 256, 64
# YAH_GV_RW="swiglu:1,8;resid:2,4": rows per wave R and waves per workgroup W per GEMV kind
# (plain, resid, resid_norm, swiglu, bands); default 2,4. Recorded in decode.txt ("rw kind R W").
RW = {k: (2, 4) for k in ("plain", "resid", "resid_norm", "swiglu", "bands")}
for item in filter(None, os.environ.get("YAH_GV_RW", "").split(";")):
    k, v = item.split(":")
    RW[k] = tuple(int(x) for x in v.split(","))


def emit_src(text, name, outdir, configs=("nop=0",)):
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(tmp, name + ".loom")
    open(src, "w").write(text)
    emit_file(src, name, outdir, configs)


def emit_file(path, name, outdir, configs):
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    r = subprocess.run([sys.executable, EMIT, path, tmp] + list(configs), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("emit failed for %s:\n%s%s" % (path, r.stdout[-3000:], r.stderr[-3000:]))
    shutil.copy(r.stdout.strip().splitlines()[0], os.path.join(outdir, name + ".hal"))


def gemv_name(kind, fmts, M, K):
    return "gv_%s_%s_%d_%d" % (kind, "_".join(fmts), M, K)


def bands_name(fmts, Ms, K):
    return "gb_%s_%s_%d" % ("_".join(fmts), "_".join(map(str, Ms)), K)


def bands_set(model):
    """{name: (fmts, Ms, K)}: the input projections of each layer as one band-fused GEMV
    (attn_qkv / attn_gate / ssm_alpha / ssm_beta, or attn_q / attn_k / attn_v)."""
    t = {}
    for nm, dims, ty in EP.parse(model):
        t[nm] = (GV.GGML.get(ty), int(dims[0]), int(dims[1]) if len(dims) > 1 else 1)
    out = {}
    layers = sorted({int(n.split(".")[1]) for n in t if n.startswith("blk.") and n.split(".")[1].isdigit()})
    for l in layers:
        p = "blk.%d." % l
        if p + "ffn_gate.weight" not in t:
            continue
        group = ("attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta") if p + "attn_qkv.weight" in t else ("attn_q", "attn_k", "attn_v")
        ns = [p + n + ".weight" for n in group]
        fmts, K, Ms = [t[n][0] for n in ns], t[ns[0]][1], [t[n][2] for n in ns]
        out[bands_name(fmts, Ms, K)] = (fmts, Ms, K)
    return out


def gemv_set(model):
    """{name: (kind, fmts, M, K)} for every decode projection on the shard."""
    t = {}
    for nm, dims, ty in EP.parse(model):
        t[nm] = (GV.GGML.get(ty), int(dims[0]), int(dims[1]) if len(dims) > 1 else 1)
    out = {}

    def add(kind, names):
        fmts = [t[n][0] for n in names]
        if None in fmts:
            raise SystemExit("no GEMV decoder for " + ", ".join(names))
        K, M = t[names[0]][1], t[names[0]][2]
        out[gemv_name(kind, fmts, M, K)] = (kind, fmts, M, K)

    layers = sorted({int(n.split(".")[1]) for n in t if n.startswith("blk.") and n.split(".")[1].isdigit()})
    for l in layers:
        p = "blk.%d." % l
        if p + "ffn_gate.weight" not in t:
            continue
        for n in ("attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta"):
            if p + n + ".weight" in t:
                add("plain", [p + n + ".weight"])
        for n in ("attn_output", "ssm_out", "ffn_down"):
            if p + n + ".weight" in t:
                add("resid", [p + n + ".weight"])
                if GV.LPR == 32:
                    add("resid_norm", [p + n + ".weight"])
        add("swiglu", [p + "ffn_gate.weight", p + "ffn_up.weight"])
    add("plain", ["output.weight"])
    return out


def main():
    model, outdir = sys.argv[1], sys.argv[2]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    if T % 256:
        raise SystemExit("max_context must be a multiple of 256")
    os.makedirs(outdir, exist_ok=True)
    gs = gemv_set(model)
    grids = {}   # exact launch grid of every GEMV kernel: the decoder refuses any other
    for name, (kind, fmts, M, K) in sorted(gs.items()):
        R, W = RW[kind]
        if M % (R * W):   # small outputs (48 rows) keep the default geometry
            R, W = 2, 4
        emit_src(GV.gen(kind, fmts, M, K, R, W), name, outdir)
        grids[name] = M // GV.rows_per_wg(R, W)
        if GV.PERSIST and kind != "resid_norm":
            grids[name] = min(grids[name], GV.PERSIST)
    bs = bands_set(model)
    for name, (fmts, Ms, K) in sorted(bs.items()):
        emit_src(GV.gen_bands(fmts, Ms, K, *RW["bands"]), name, outdir)
        grids[name] = sum(Ms) // GV.rows_per_wg(*RW["bands"])
    for which in ("kvappend", "part", "reduce"):
        emit_src(DA.gen(which, T), "dattn_" + which, outdir)
    L = lambda f: os.path.join(LOOM, f)
    emit_src(DM.gen_rmsnorm(), "rmsnorm", outdir)          # 512 lanes (the port: one 32-lane subgroup)
    emit_file(L("yah_unpack_qg_f32.loom"), "unpack", outdir,
              ["yah_unpack_qg.batch=1", "yah_unpack_qg.num_heads=%d" % NUM_HEADS, "yah_unpack_qg.head_dim=%d" % HEAD_DIM])
    emit_file(L("yah_fused_qk_rope_f32.loom"), "rope", outdir, [
        "yah_fused_qk_rope.layer_idx=0", "yah_fused_qk_rope.max_context=1",
        "yah_fused_qk_rope.num_heads=%d" % NUM_HEADS, "yah_fused_qk_rope.num_kv_heads=%d" % NUM_KV,
        "yah_fused_qk_rope.head_dim=%d" % HEAD_DIM, "yah_fused_qk_rope.rotary_dim=%d" % ROTARY,
        "yah_fused_qk_rope.q_elems=%d" % (NUM_HEADS * HEAD_DIM), "yah_fused_qk_rope.kv_elems=%d" % (NUM_KV * HEAD_DIM),
        "yah_fused_qk_rope.cache32_elems=%d" % (NUM_KV * HEAD_DIM),
        "yah_fused_qk_rope.cache16_elems=%d" % (NUM_KV * HEAD_DIM)])
    emit_file(L("yah_ssm_conv_decode_f32.loom"), "ssmconv", outdir,
              ["yah_ssm_conv_decode.qkv_dim=10240", "yah_ssm_conv_decode.rows=1"])
    emit_src(DM.gen_deltanet(hoist=True), "deltanet", outdir)
    emit_src(DM.gen_deltanet(fuse=True, hoist=True), "deltanet_conv", outdir)   # + the decode conv
    emit_src(DM.gen_embed_iq4xs(), "embed", outdir)      # token_embd row from the device token stream        # 512 lanes per head (the port: one wave)
    emit_file(L("yah_argmax_f32.loom"), "argmax", outdir, ["yah_argmax.vocab=248320"])
    for f in os.listdir(os.path.join(LOOM, "tables")):
        shutil.copy(os.path.join(LOOM, "tables", f), os.path.join(outdir, f))
    open(os.path.join(outdir, "decode.txt"), "w").write(
        "ctx %d\n" % T + "".join("rw %s %d %d\n" % (k, r * (2 if GV.LPR == 16 else 1), w) for k, (r, w) in sorted(RW.items())) +
        ("persist gv %d 0\n" % GV.PERSIST if GV.PERSIST else "") +
        "".join("grid %s %d 0\n" % (n, g) for n, g in sorted(grids.items())))
    shutil.rmtree(os.path.join(outdir, ".emit_tmp"), ignore_errors=True)
    print("emitted %d GEMV + %d band GEMV + 10 decode HALs (max context %d) to %s" % (len(gs), len(bs), T, outdir))


if __name__ == "__main__":
    main()
