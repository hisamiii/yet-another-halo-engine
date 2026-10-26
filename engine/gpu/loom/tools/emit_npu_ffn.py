#!/usr/bin/env python3
"""Emit the NPU FFN block into an A4 prefill set (emit_prefill_pp.py, B tokens): the NPU owns the FFN's last F features
of every layer (gate / up + swiglu, then down over its features), the GPU the rest; npu.txt beside dispatch.txt.

usage: emit_npu_ffn.py <model> <prefill set> <tokens B> [F]   (F a multiple of 1024, default 8192)

GPU kernels (B tokens):
  normqx_5120.hal     the FFN norm's normq that also writes the rotated row in the MQ4 decoder's positions (gen_rot xo)
  npu_enc_act.hal     that row to the gate / up image's BFP16 activation stream (one replay group, the decoder's P4 order)
  iu4_gu_<m>_20.hal   gate / up over the GPU's 17408 - F features (m = their 16-row tiles)
  rotq_<17408 - F>.hal   their rotq (down's awq over the GPU's features)
  iu4_kresk_320_<kb>.hal down's residual GEMM over that K window of its rows (gen_gemm_iu4 kpitch)
  npu_unpack_ffn.hal  hidden + the NPU's down partials (gen_npu_unpack resid, parts)
  npu_flag_wait.hal   waits for an NPU job's done word (gen_npu_unpack.gen_flag_wait)
NPU images (gen_npu_gemm, gated):
  npu_ffnsw.xdna      gate / up + swiglu, MQ4 rows decoded on the cores (fuse 2), F / 256 calls: call j's column c
                      takes features 17408 - F + 256 j + 32 c + [0, 32); H = bfp16(silu(gate) up) into C
  npu_ffndn<p>.xdna   down from H over a K part of p passes (1024 features each), IQ4_XS rows decoded on the cores: the
                      rows with the rotation and down's awq folded in, re-quantized (npu_down_pack.hpp), 10 calls of 512
npu.txt: "npuffn <F> <B> <K parts>", "npugate <supply> <calls> <record>", "npubytes_<image> a w c",
"npudgu <bytes from a gate row to its up row>", then the GPU kernels' geometry rows (dispatch.txt's format).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import emit_prefill as E  # noqa: E402
import emit_prefill_pp as P  # noqa: E402
import gen_bfp16_encode as GE  # noqa: E402
import gen_gemm_iu4  # noqa: E402
import gen_npu_gemm as N  # noqa: E402
import gen_npu_unpack as GU  # noqa: E402
import gen_rot  # noqa: E402
import hrx_paths  # noqa: E402
import subprocess  # noqa: E402

FFN, HID = 17408, 5120
# k-blocks per pass and K-slice row (K = 1024 per pass). Gate / up has its M blocks' epilogue deferred into the next
# block's passes (gen_npu_gemm dfr; -7.6%, the same sums): the short tail slice frees the room for it.
KS_SW = (40, 40, 40, 8)
KS_DN = (36, 32, 32, 28)    # down (kchain: slice ends at multiples of 4): -3% over (36, 36, 36, 20)
COLS, NP = 8, 4             # NPU columns, 16-row slabs per column: 512 rows a call
TARGET = "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11"


def parts(F):
    """(first pass, passes) of down's K parts over F features: the fewest of at most 5 passes, balanced."""
    P_, n = F // 1024, (F // 1024 + 4) // 5
    out, p0 = [], 0
    for i in range(n):
        out.append((p0, P_ // n + (i < P_ % n)))
        p0 += out[-1][1]
    return out


def sw_cfg(B, F, dgu):
    """H in down's decoder order (IQ4_XS: P4)."""
    return N.Config(COLS, B // 64, KS_SW, 5, mu=1, gate=N.GATE_SUPPLY, gwest=1, ghelp=1, gfirst=1, fuse=2, swiglu=True,
                    dgu=dgu, swcol=32, hpass=F // 1024, hord="P4", fmt="MQ4", dfr=1)


def dn_cfg(B, F, passes):
    """Down decodes its IQ4_XS rows (model/npu_down_pack.hpp) itself: the same gate layout as gate / up's (gwest), so the two
    images switch inside a command (a dense image's setup leaves column 2's gate routes of gate / up's in place: the
    array hangs)."""
    return N.Config(COLS, B // 64, KS_DN, passes, mu=1, gate=N.GATE_SUPPLY, gwest=1, ghelp=1, gfirst=1, fuse=2, kraw=F,
                    ain="h", hpass=F // 1024, fmt="IQ4_XS", kchain=1, hcap=2, gtail=1)


def compile_npu(cfg, groups, name, outdir):
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    GE.GROUPS = groups
    src = os.path.join(tmp, name + ".loom")
    open(src, "w").write(N.gen(cfg))
    r = subprocess.run([hrx_paths.LOOM_COMPILE, src, f"--root=@{cfg.entry}", TARGET,
                        "--output=" + os.path.join(outdir, name + ".xdna")],
                       env=dict(hrx_paths.env(), **N.loom_env(cfg)), capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"loom-compile {name} failed:\n{(r.stderr + r.stdout)[-3000:]}")
    return tuple(N.stream_bytes(cfg))


# A native set (a GGUF's own formats, emit_prefill_pp YAH_NATIVE): the NPU decoders per role (gen_npu_dec DQ_FMT). Gate
# and up of one format and the decoder's activation order (IQ4_XS's P4); down any of DN_FMTS, H in its k-block order.
SW_FMTS = {"iq4xs": "IQ4_XS"}
DN_FMTS = {"iq4xs": "IQ4_XS", "q4k": "Q4_K", "iq3s": "IQ3_S"}
BLK = {"iq4xs": 136, "q4k": 144, "iq3s": 110}


def hord(dfmt):
    return "Q4K" if dfmt == "q4k" else "P4"


def dn_native(B, F, passes, dfmt):
    """Down from H over a K part, the GGUF's own rows (kraw: the whole row) decoded on the cores. IQ4_XS: dn_cfg's
    k-slices; the others the old engine's (36, 36, 36, 20), grid formats with their helper rings late."""
    if dfmt == "iq4xs":
        return N.Config(COLS, B // 64, KS_DN, passes, mu=1, gate=N.GATE_SUPPLY, gwest=1, ghelp=1, gfirst=1, fuse=2,
                        kraw=FFN, ain="h", hpass=F // 1024, fmt="IQ4_XS", kchain=1, hcap=2, gtail=1)
    return N.Config(COLS, B // 64, (36, 36, 36, 20), passes, mu=1, gate=N.GATE_SUPPLY, gwest=1, ghelp=1, gfirst=1,
                    fuse=2, kraw=FFN, ain="h", hpass=F // 1024, fmt=DN_FMTS[dfmt], kchain=1, hcap=2, gtail=1,
                    hlate=int(dfmt == "iq3s"))


def native(model, outdir, B, F):
    """The NPU FFN block of a native set: the layers whose gate / up and down formats have decoders run it (npu.txt
    "npulayer_<layer> 1 0 0"); its images "npu_ffnsw_<fmt>_<hord>" and "npu_ffndn<passes>_<fmt>"; the GPU's share the
    fused ffn over the first G features (".npu.hal": OSTRIDE) and down's residual GEMM over that K window
    ("gemm_kresw_<fmt>_<G / 256>.hal": KWIN)."""
    import dataclasses
    import gen_gemm_tile as TG
    G = FFN - F
    by = {}
    for nm, dims, ty in E.parse(model):
        p = nm.split(".")
        if len(p) > 2 and p[0] == "blk" and p[2] in ("ffn_gate", "ffn_up", "ffn_down") and dims[-1] in (FFN, HID):
            by.setdefault(int(p[1]), {})[p[2]] = E.FMT[ty][0]
    ok = {l: f for l, f in by.items() if len(f) == 3 and f["ffn_gate"] == f["ffn_up"] in SW_FMTS
          and f["ffn_down"] in DN_FMTS}
    if not ok:
        sys.exit("emit_npu_ffn: no layer of %s has NPU decoders for its FFN formats" % model)
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    rows = [("npuffn", F, B, len(parts(F))), ("npugate", N.GATE_SUPPLY, N.GATE_CALLS, N.GATE_RECORD), ("npunative", 1, 0, 0)]
    rows += [("npulayer_%d" % l, 1, 0, 0) for l in sorted(ok)]

    def loom(name, text, configs=("nop=0",)):
        src = os.path.join(tmp, name[:-4] + ".loom")
        open(src, "w").write(text)
        E.emit(src, list(configs), name, outdir)
    # ---- GPU: the encoder (from the norm's row-major output), the GPU's gate / up and down, the unpack, the flag wait
    groups, perm = GE.GROUPS, GE.ACT_PERM
    GE.GROUPS, GE.ACT_PERM = (5,), "p4"
    try:
        loom("npu_enc_act.hal", GE.gen("act", B, list(KS_SW), 5))
    finally:
        GE.GROUPS, GE.ACT_PERM = groups, perm
    rows.append(("npu_enc_act.hal", GE.WG, 0, (B // 8) * 5 * sum(KS_SW) // GE.WG))
    for fmt in sorted({f["ffn_gate"] for f in ok.values()}):
        t = dataclasses.replace(TG.default_tile(fmt, "swiglu", HID // 256), **P.AF_TILE, **P.AF_FFN[fmt], ffn=True,
                                tout=True)
        TG.check(t)
        TG.OSTRIDE = FFN
        try:
            rows.append(P._emit_gen(lambda f, k: TG.gen(f, k, t, False), t.bn, fmt, G // 16, HID // 256, B,
                                    "gemm_ffn_%s_%d_%d.npu.hal" % (fmt, FFN // 16, HID // 256), outdir, "ffn",
                                    t.rowgrp, ostride=FFN))
        finally:
            TG.OSTRIDE = 0
    for fmt in sorted({f["ffn_down"] for f in ok.values()}):
        knobs = dict(P.AF[(fmt, "kres", HID // 16, FFN // 256)])
        knobs.pop("persist", None)
        knobs["respre"] = 0
        t = dataclasses.replace(TG.default_tile(fmt, "kres", G // 256), **P.AF_TILE, **knobs)
        TG.check(t)
        TG.KWIN = (0, FFN // 256)
        try:
            rows.append(P._emit_gen(lambda f, k: TG.gen(f, k, t, False), t.bn, fmt, HID // 16, G // 256, B,
                                    "gemm_kresw_%s_%d.hal" % (fmt, G // 256), outdir, "kres", t.rowgrp,
                                    kfull=FFN // 256))
        finally:
            TG.KWIN = None
    calls = HID // (COLS * 64)
    cg = GU.CG
    GU.CG = 1
    try:
        with N.np_override(NP):
            loom("npu_unpack_ffn.hal", GU.gen(B, calls * COLS, HID, 0, resid=True, parts=len(parts(F))))
            nj, wg = N.TN // 8, GU.wg(calls * COLS)
    finally:
        GU.CG = cg
    rows.append(("npu_unpack_ffn.hal", B * calls * COLS * nj // wg, wg, 0))
    loom("npu_flag_wait.hal", GU.gen_flag_wait())
    rows.append(("npu_flag_wait.hal", 1, 32, 0))
    # ---- NPU
    with N.np_override(NP):
        for fmt, ho in sorted({(f["ffn_gate"], hord(f["ffn_down"])) for f in ok.values()}):
            cfg = dataclasses.replace(sw_cfg(B, F, FFN * (HID // 256) * BLK[fmt]), hord=ho, fmt=SW_FMTS[fmt])
            name = "ffnsw_%s_%s" % (fmt, ho.lower())
            rows.append(("npubytes_" + name,) + compile_npu(cfg, (5,), "npu_" + name, outdir))
        for fmt in sorted({f["ffn_down"] for f in ok.values()}):
            for passes in sorted({p for _, p in parts(F)}):
                name = "ffndn%d_%s" % (passes, fmt)
                rows.append(("npubytes_" + name,) + compile_npu(dn_native(B, F, passes, fmt), (passes,), "npu_" + name,
                                                                outdir))
    with open(os.path.join(outdir, "npu.txt"), "w") as fh:
        for r in rows:
            fh.write(" ".join(str(v) for v in r) + "\n")
    print(f"npu ffn block F={F} B={B} (native): layers {sorted(ok)}, {len(rows)} rows in {outdir}/npu.txt")


def main():
    model, outdir, B = sys.argv[1], sys.argv[2], int(sys.argv[3])
    F = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
    assert F % 1024 == 0 and 0 < F < FFN and (FFN - F) % 256 == 0 and B % 64 == 0
    if any(ln.startswith("native ") for ln in open(os.path.join(outdir, "dispatch.txt"))):
        return native(model, outdir, B, F)
    G = FFN - F
    # up's rows follow gate's by its MQ4 bytes and the HFQ's f16 gate awq (a GGUF is converted to that layout;
    # LoomPrefill checks every layer's)
    dgu = FFN * (HID // 256) * 136 + HID * 2
    tmp = os.path.join(outdir, ".emit_tmp")
    os.makedirs(tmp, exist_ok=True)
    rows = [("npuffn", F, B, len(parts(F))), ("npugate", N.GATE_SUPPLY, N.GATE_CALLS, N.GATE_RECORD),
            ("npudgu", dgu, 0, 0)]

    def loom(name, text, configs=("nop=0",)):
        src = os.path.join(tmp, name[:-4] + ".loom")
        open(src, "w").write(text)
        E.emit(src, list(configs), name, outdir)
    # ---- GPU
    loom("normqx_5120.hal", gen_rot.gen_nq(5120, B, gen_gemm_iu4.KP, xo=True))
    rows.append(("normqx_5120.hal", 0, 16 * 5120 // 256, 1))
    groups, perm = GE.GROUPS, GE.ACT_PERM
    GE.GROUPS, GE.ACT_PERM = (5,), "p4"
    try:
        loom("npu_enc_act.hal", GE.gen("act", B, list(KS_SW), 5))
    finally:
        GE.GROUPS, GE.ACT_PERM = groups, perm
    rows.append(("npu_enc_act.hal", GE.WG, 0, (B // 8) * 5 * sum(KS_SW) // GE.WG))
    rows.append(P.iu4_gemm("gu", G // 16, HID // 256, B, outdir))
    rows.append(("iu4grp_" + rows[-1][0], gen_gemm_iu4.group_of("gu", G), 0, 0))
    loom("rotq_%d.hal" % G, gen_rot.gen_q(G, B, gen_gemm_iu4.KP, True, None))
    rows.append(("rotq_%d.hal" % G, 0, 64, G // 1024))
    rows.append(iu4_kresk(G // 256, B, outdir))
    rows.append(("iu4grp_" + rows[-1][0], gen_gemm_iu4.group_of("kres", HID), 0, 0))
    calls = HID // (COLS * 64)
    cg = GU.CG
    GU.CG = 1
    try:
        with N.np_override(NP):
            loom("npu_unpack_ffn.hal", GU.gen(B, calls * COLS, HID, 0, resid=True, parts=len(parts(F))))
            nj, wg = N.TN // 8, GU.wg(calls * COLS)
    finally:
        GU.CG = cg
    rows.append(("npu_unpack_ffn.hal", B * calls * COLS * nj // wg, wg, 0))
    loom("npu_flag_wait.hal", GU.gen_flag_wait())
    rows.append(("npu_flag_wait.hal", 1, 32, 0))
    # ---- NPU
    with N.np_override(NP):
        rows.append(("npubytes_ffnsw",) + compile_npu(sw_cfg(B, F, dgu), (5,), "npu_ffnsw", outdir))
        for passes in sorted({p for _, p in parts(F)}):
            rows.append(("npubytes_ffndn%d" % passes,) + compile_npu(dn_cfg(B, F, passes), (passes,), "npu_ffndn%d" % passes,
                                                                    outdir))
    with open(os.path.join(outdir, "npu.txt"), "w") as fh:
        for r in rows:
            fh.write(" ".join(str(v) for v in r) + "\n")
    print(f"npu ffn block F={F} B={B}: {len(rows)} rows in {outdir}/npu.txt")


def iu4_kresk(kb, B, outdir):
    """down's residual GEMM over the first kb 256-blocks of its 68 (the GPU's features): iu4_kresk_320_<kb>.hal."""
    bm, bn, _, _ = gen_gemm_iu4.tile(HID)
    sym = gen_gemm_iu4.sym("kres")
    out = "iu4_kresk_320_%d.hal" % kb
    src = os.path.join(outdir, ".emit_tmp", out[:-4] + ".loom")
    open(src, "w").write(gen_gemm_iu4.gen("kres", HID, kpitch=FFN // 256))
    E.emit(src, ["%s.m_tiles=%d" % (sym, HID // 16), "%s.k_blocks=%d" % (sym, kb), "%s.token_tiles=%d" % (sym, B // bn)],
           out, outdir)
    return (out, bn, bm // 16, B // bn)


if __name__ == "__main__":
    main()
