#!/usr/bin/env python3
"""Emit every HAL the Loom single-token decode forward needs for a shard.

usage: emit_decode.py <model.gguf> <outdir>

Reuses emit_prefill.py for the kStore/kSwiGLU/kResidual GEMMs and emit_hal.py
for the fixed decode kernels. The decode attention kernel bakes start_pos, so
one HAL is emitted per position up to --positions (default 32). The IQ grid/
ksigns tables are emitted by emit_prefill.py from loom/tables/.
"""
import os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
LOOM = os.path.abspath(os.path.join(HERE, ".."))
EMIT = os.path.join(LOOM, "emit_hal.py")
PREFILL = os.path.join(LOOM, "tools", "emit_prefill.py")

MAX_CONTEXT = 64
NUM_HEADS = 24
NUM_KV = 4
HEAD_DIM = 256
ROTARY = 64
CACHE = MAX_CONTEXT * NUM_KV * HEAD_DIM
POSITIONS = 32


def emit(loomfile, outdir, outname, configs):
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    r = subprocess.run([sys.executable, EMIT, os.path.join(LOOM, loomfile), tmp] + configs,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("emit failed for %s:\n%s%s" % (loomfile, r.stdout, r.stderr))
    hal = r.stdout.strip().splitlines()[-1]
    shutil.copy(hal, os.path.join(outdir, outname))


def main():
    model, outdir = sys.argv[1], sys.argv[2]
    os.environ["YAH_TOKEN_TILE"] = "64"  # decode keeps the 64-wide tile
    os.makedirs(outdir, exist_ok=True)
    subprocess.run([sys.executable, PREFILL, model, outdir], check=True)
    emit("yah_half_norm_f16.loom", outdir, "norm.hal",
         ["yah_half_norm.rows=1", "yah_half_norm.dim=5120",
          "yah_half_norm.eps=1e-6"])
    emit("yah_unpack_qg_f32.loom", outdir, "unpack.hal",
         ["yah_unpack_qg.batch=1", "yah_unpack_qg.num_heads=%d" % NUM_HEADS,
          "yah_unpack_qg.head_dim=%d" % HEAD_DIM])
    emit("yah_half_cast.loom", outdir, "cast.hal", ["yah_half_cast.num_elements=6144"])
    emit("yah_rmsnorm_f32.loom", outdir, "rmsnorm.hal",
         ["yah_rmsnorm.rows=1", "yah_rmsnorm.eps=1e-6"])
    emit("yah_gemv_q6k_f32.loom", outdir, "gemv.hal",
         ["yah_gemv_q6k.m_rows=248320", "yah_gemv_q6k.k_blocks=20"])
    emit("yah_argmax_f32.loom", outdir, "argmax.hal", ["yah_argmax.vocab=248320"])
    emit("yah_ssm_conv_decode_f32.loom", outdir, "ssmconv.hal",
         ["yah_ssm_conv_decode.qkv_dim=10240", "yah_ssm_conv_decode.rows=1"])
    emit("yah_deltanet_decode_resident_f32.loom", outdir, "deltanet.hal",
         ["yah_deltanet_decode.num_heads=48", "yah_deltanet_decode.num_key_heads=16"])
    emit("yah_fused_qk_rope_f32.loom", outdir, "rope.hal", [
        "yah_fused_qk_rope.layer_idx=0",
        "yah_fused_qk_rope.max_context=%d" % MAX_CONTEXT,
        "yah_fused_qk_rope.num_heads=%d" % NUM_HEADS,
        "yah_fused_qk_rope.num_kv_heads=%d" % NUM_KV,
        "yah_fused_qk_rope.head_dim=%d" % HEAD_DIM,
        "yah_fused_qk_rope.rotary_dim=%d" % ROTARY,
        "yah_fused_qk_rope.q_elems=%d" % (NUM_HEADS * HEAD_DIM),
        "yah_fused_qk_rope.kv_elems=%d" % (NUM_KV * HEAD_DIM),
        "yah_fused_qk_rope.cache32_elems=%d" % CACHE,
        "yah_fused_qk_rope.cache16_elems=%d" % CACHE,
    ])
    for pos in range(POSITIONS):
        emit("yah_decode_attn_f16.loom", outdir, "attn_%d.hal" % pos, [
            "yah_decode_attn.layer_idx=0",
            "yah_decode_attn.start_pos=%d" % pos,
            "yah_decode_attn.max_context=%d" % MAX_CONTEXT,
            "yah_decode_attn.num_heads=%d" % NUM_HEADS,
            "yah_decode_attn.num_kv_heads=%d" % NUM_KV,
            "yah_decode_attn.head_dim=%d" % HEAD_DIM,
            "yah_decode_attn.rows=1",
            "yah_decode_attn.has_gate=1",
        ])
    # emit_prefill.py already wrote the IQ grid/sign tables into outdir.
    print("emitted decode HALs to", outdir)


if __name__ == "__main__":
    main()