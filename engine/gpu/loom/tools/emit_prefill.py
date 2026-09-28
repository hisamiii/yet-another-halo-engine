#!/usr/bin/env python3
"""Emit every HAL the prefill forward needs for a GGUF shard.

Naming convention so the C++ driver needs no manifest:
  <outdir>/gemm_<kind>_<fmt>_<m_tiles>_<k_blocks>.hal
  <outdir>/<fixed>.hal   for the non-GEMM kernels
where kind is kstore|residual|swiglu.
"""
import os, re, struct, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOOM = os.path.abspath(os.path.join(HERE, ".."))
EMIT = os.path.join(LOOM, "emit_hal.py")

# ggml type -> (short, port stem, qk)
FMT = {
    12: ("q4k", "q4k", 256), 13: ("q5k", "q5k", 256), 14: ("q6k", "q6k", 256),
    11: ("q3k", "q3k", 256), 23: ("iq4xs", "iq4xs", 256), 21: ("iq3s", "iq3s", 256),
    18: ("iq3xxs", "iq3xxs", 256), 20: ("iq4nl", "iq4nl", 32),
    17: ("iq2xs", "iq2xs", 256), 8: ("q8_0", "q8_0", 32),
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


def emit(loomfile, configs, outname, outdir):
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    r = subprocess.run([sys.executable, EMIT, os.path.join(LOOM, loomfile), tmp] + configs,
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
            emit(f, [f"{sym}.m_tiles={mt}", f"{sym}.k_blocks={kb}", f"{sym}.token_tiles=1"],
                 f"gemm_residual_{fmt}_{mt}_{kb}.hal", outdir); n += 1
        else:
            f = f"yah_ffn_gemm_{port}_swiglu_f16.loom"; sym = sym_of(f)
            emit(f, [f"{sym}.m_tiles={mt}", f"{sym}.k_blocks={kb}", f"{sym}.token_tiles=1"],
                 f"gemm_swiglu_{fmt}_{mt}_{kb}.hal", outdir); n += 1
    print("emitted", n, "GEMM HALs")


if __name__ == "__main__":
    main()