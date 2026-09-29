#!/usr/bin/env python3
"""Emit every HAL the prefill forward needs for a GGUF shard.

Naming convention so the C++ driver needs no manifest:
  <outdir>/gemm_<kind>_<fmt>_<m_tiles>_<k_blocks>.hal
  <outdir>/<fixed>.hal   for the non-GEMM kernels
where kind is kstore|residual|swiglu.
"""
import os, re, shutil, struct, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Token-tile width for the emitted GEMM family. Prefill narrows to 16 (only kB
# real tokens are read); the decode path keeps the original 64.
TOKEN_TILE = int(os.environ.get("YAH_TOKEN_TILE", "16"))
LOOM = os.path.abspath(os.path.join(HERE, ".."))
EMIT = os.path.join(LOOM, "emit_hal.py")

# ggml type -> (short, port stem, qk)
FMT = {
    12: ("q4k", "q4k", 256), 13: ("q5k", "q5k", 256), 14: ("q6k", "q6k", 256),
    11: ("q3k", "q3k", 256), 23: ("iq4xs", "iq4xs", 256), 21: ("iq3s", "iq3s", 256),
    18: ("iq3xxs", "iq3xxs", 256), 20: ("iq4nl", "iq4nl", 32),
    17: ("iq2xs", "iq2xs", 256), 8: ("q8_0", "q8_0", 32),
    16: ("iq2xxs", "iq2xxs", 256), 10: ("q2k", "q2k", 256),
}
KSTORE = {"attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight",
          "ssm_beta.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
          "ffn_gate.weight"}
RESIDUAL = {"attn_output.weight", "ssm_out.weight", "ffn_down.weight"}
SWIGLU = {"ffn_up.weight"}


def parse(model):
    f = open(model, "rb"); f.read(8)
    nt, nkv = struct.unpack("<QQ", f.read(16))
    SIZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    def rd():
        n = struct.unpack("<Q", f.read(8))[0]; return f.read(n).decode("utf-8", "replace")
    def skip(t):
        if t == 8: rd()
        elif t == 9:
            et = struct.unpack("<I", f.read(4))[0]; n = struct.unpack("<Q", f.read(8))[0]
            if et == 8:
                for _ in range(n): rd()
            else: f.seek(SIZ[et] * n, 1)
        else: f.seek(SIZ[t], 1)
    for _ in range(nkv):
        rd(); t = struct.unpack("<I", f.read(4))[0]; skip(t)
    rows = []
    for _ in range(nt):
        nm = rd(); nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        ty = struct.unpack("<I", f.read(4))[0]
        f.read(8)
        rows.append((nm, dims, ty))
    return rows


def sym_of(loomfile):
    text = open(os.path.join(LOOM, loomfile)).read()
    return re.search(r"config\.decl @([A-Za-z0-9_]+)\.m_tiles", text).group(1)


def narrow_tokens(text):
    """Rewrite a GEMM kernel to a 16-wide token tile.

    The prefill pads kB real tokens to a 64-wide tile, so 3/4 of the rhs loads,
    MMAs and epilogue stores are waste. The element loop maps lane l to column
    l&15, and the epilogue decodes (row, token) with a shift/mask pair, so the
    same source narrows structurally: only n-group 0 survives and the token
    decode becomes 16 wide. Only tokens 0..kB-1 are ever read, so the driver and
    every other kernel are unchanged."""
    keep = []
    for line in text.split(chr(10)):
        if "%tokens = index.mul %token_tiles, %c64" in line or \
           "%token_base = index.mul %wg_y, %c64" in line:
            keep.append(line.replace("%c64", "%c16"))
        elif re.search(r"%rhs[123] = vector\.fragment\.load<rhs>", line):
            continue
        elif re.search(r"%n[123] = vector\.mma", line):
            continue
        elif line.strip().startswith("scf.yield %n0, %n1, %n2, %n3"):
            keep.append(line.replace("scf.yield %n0, %n1, %n2, %n3",
                                     "scf.yield %n0, %a1, %a2, %a3"))
        elif re.search(r"vector\.fragment\.store<result> %acc[123],", line):
            continue
        elif "scf.for %j2 = [%c0 to %c32 step %c1]" in line:
            keep.append(line.replace("%c32", "%c8"))
        elif "%r2_i = scalar.shrui %e2_i, %c6i" in line:
            keep.append(line.replace("%c6i", "%c4i"))
        elif "%tok_i = scalar.andi %e2_i, %c63i" in line:
            keep.append(line.replace("%c63i", "%c15i"))
        else:
            keep.append(line)
    return chr(10).join(keep)

def emit(loomfile, configs, outname, outdir):
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(LOOM, loomfile)
    if os.path.basename(loomfile).startswith("yah_ffn_gemm_") and TOKEN_TILE == 16:
        src_tmp = os.path.join(tmp, os.path.basename(loomfile))
        with open(src_tmp, "w") as fh:
            fh.write(narrow_tokens(open(src).read()))
        src = src_tmp
    r = subprocess.run([sys.executable, EMIT, src, tmp] + configs,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("emit failed for " + loomfile + ": " + r.stdout + r.stderr)
    hal = r.stdout.strip().splitlines()[-1]
    dst = os.path.join(outdir, outname)
    subprocess.run(["cp", hal, dst], check=True)
    return dst


def main():
    model, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    rows = parse(model)
    combos = set()
    for nm, dims, ty in rows:
        suffix = nm.split(".", 2)[2] if nm.startswith("blk.") else nm
        info = FMT.get(ty)
        if not info: continue
        fmt, port, qk = info
        if suffix in KSTORE: kind = "kstore"
        elif suffix in RESIDUAL: kind = "residual"
        elif suffix in SWIGLU: kind = "swiglu"
        else: continue
        mt, kb = dims[1] // 16, dims[0] // qk
        combos.add((kind, fmt, port, mt, kb))
    n = 0
    for kind, fmt, port, mt, kb in sorted(combos):
        if kind == "kstore":
            f = f"yah_ffn_gemm_{port}_f32.loom"; sym = sym_of(f)
            emit(f, [f"{sym}.m_tiles={mt}", f"{sym}.k_blocks={kb}", f"{sym}.token_tiles=1"],
                 f"gemm_kstore_{fmt}_{mt}_{kb}.hal", outdir); n += 1
        elif kind == "residual":
            f = f"yah_ffn_gemm_{port}_residual_f32.loom"; sym = sym_of(f)
            # Every residual arm has m_tiles=320 (the projection writes hidden),
            # so a 4-way K split raises the grid from 320 to 1280 workgroups.
            emit(f, [f"{sym}.m_tiles={mt}", f"{sym}.k_blocks={kb}",
                     f"{sym}.token_tiles=1", f"{sym}.k_split=4", f"{sym}.accum=0"],
                 f"gemm_residual_{fmt}_{mt}_{kb}.hal", outdir); n += 1
        else:
            f = f"yah_ffn_gemm_{port}_swiglu_f16.loom"; sym = sym_of(f)
            emit(f, [f"{sym}.m_tiles={mt}", f"{sym}.k_blocks={kb}", f"{sym}.token_tiles=1"],
                 f"gemm_swiglu_{fmt}_{mt}_{kb}.hal", outdir); n += 1
    # The residual reduction: hidden += sum_s partial[s], dim = hidden * 64 tokens.
    emit("yah_residual_add_1d_f32.loom", ["yah_residual_1d.dim=%d" % (5120 * TOKEN_TILE)],
         "accum.hal", outdir); n += 1
    # The fixed prefill kernels at the shard's shapes.
    fixed = [
        ("yah_half_norm_f16.loom", "norm.hal",
         ["yah_half_norm.rows=5", "yah_half_norm.dim=5120", "yah_half_norm.eps=1e-06"]),
        ("yah_ssm_conv_f32.loom", "conv.hal",
         ["yah_ssm_conv.batch=5", "yah_ssm_conv.qkv_dim=10240"]),
        ("yah_deltanet_prep_kq_f32.loom", "prepkq.hal",
         ["yah_deltanet_prep_kq.batch=5", "yah_deltanet_prep_kq.num_key_heads=16",
          "yah_deltanet_prep_kq.qkv_size=10240"]),
        ("yah_deltanet_prep_ab_f32.loom", "prepab.hal",
         ["yah_deltanet_prep_ab.batch=5", "yah_deltanet_prep_ab.qkv_size=10240",
          "yah_deltanet_prep_ab.num_heads=48"]),
        ("yah_deltanet_rowsplit_f32.loom", "rowsplit.hal",
         ["yah_deltanet.batch=5", "yah_deltanet.qkv_size=10240",
          "yah_deltanet.inner_size=6144", "yah_deltanet.num_key_heads=16",
          "yah_deltanet.num_heads=48"]),
        ("yah_ssm_postnorm_gate_f16.loom", "postnorm.hal",
         ["yah_ssm_postnorm_fp16.head_count=240"]),
        ("yah_unpack_qg_f32.loom", "unpack.hal",
         ["yah_unpack_qg.batch=5", "yah_unpack_qg.num_heads=24",
          "yah_unpack_qg.head_dim=256"]),
        ("yah_fused_qk_rope_batched_f32.loom", "rope.hal", [
            "yah_fused_qk_rope_batched.start_pos=0",
            "yah_fused_qk_rope_batched.batch=5",
            "yah_fused_qk_rope_batched.layer_idx=0",
            "yah_fused_qk_rope_batched.max_context=8",
            "yah_fused_qk_rope_batched.num_heads=24",
            "yah_fused_qk_rope_batched.num_kv_heads=4",
            "yah_fused_qk_rope_batched.head_dim=256",
            "yah_fused_qk_rope_batched.rotary_dim=64",
            "yah_fused_qk_rope_batched.q_elems=30720",
            "yah_fused_qk_rope_batched.kv_elems=5120",
            "yah_fused_qk_rope_batched.cache32_elems=8192",
            "yah_fused_qk_rope_batched.cache16_elems=8192"]),
        ("yah_attn_wmma_f32.loom", "wmma.hal", [
            "yah_attn_wmma.layer_idx=0", "yah_attn_wmma.start_pos=0",
            "yah_attn_wmma.batch_size=5", "yah_attn_wmma.max_context=8",
            "yah_attn_wmma.num_heads=24", "yah_attn_wmma.num_kv_heads=4",
            "yah_attn_wmma.head_dim=256", "yah_attn_wmma.gqa=6",
            "yah_attn_wmma.score_capacity=8", "yah_attn_wmma.kv_padded=8",
            "yah_attn_wmma.has_gate=1", "yah_attn_wmma.has_lse=0",
            "yah_attn_wmma.head_major=0"]),
        ("yah_half_cast.loom", "cast.hal", ["yah_half_cast.num_elements=30720"]),
        ("yah_rmsnorm_f32.loom", "rmsnorm.hal",
         ["yah_rmsnorm.rows=1", "yah_rmsnorm.eps=1e-06"]),
        ("yah_gemv_q6k_f32.loom", "gemv.hal",
         ["yah_gemv_q6k.m_rows=248320", "yah_gemv_q6k.k_blocks=20"]),
        ("yah_argmax_f32.loom", "argmax.hal", ["yah_argmax.vocab=248320"]),
    ]
    for loom, outname, configs in fixed:
        emit(loom, configs, outname, outdir); n += 1
    # The IQ grid and sign tables, committed under loom/tables/.
    tables = os.path.join(LOOM, "tables")
    for src, dst in [("grid_iq3s.bin", "grid_iq3s.bin"),
                     ("grid_iq3xxs.bin", "grid_iq3xxs.bin"),
                     ("grid_iq2xxs.bin", "grid_iq2xxs.bin"),
                     ("grid_iq2xs.bin", "grid_iq2xs.bin"),
                     ("ksigns_iq2xs.bin", "ksigns_iq3xxs.bin"),
                     ("ksigns_iq2xs.bin", "ksigns_iq2xxs.bin")]:
        shutil.copy(os.path.join(tables, src), os.path.join(outdir, dst))
    print("emitted", n, "GEMM + fixed prefill HALs")


if __name__ == "__main__":
    main()