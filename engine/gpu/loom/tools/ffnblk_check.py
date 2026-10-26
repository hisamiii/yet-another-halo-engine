#!/usr/bin/env python3
"""Check the NPU's FFN block on an HFQ model's real rows: the MQ4 gate / up + swiglu image (gen_npu_gemm fuse 2,
swiglu) writes H, then the dense down image (ain "h", fuse 0) multiplies H by down's rows unrotated into bfp16.

usage: ffnblk_check.py <work> [model] [layer] [sw calls]

The block is the FFN's last F = 7168 features. Gate / up: the GPU's rotated activations x' = s2 / 16 H256(s1 x / awq)
(gen_rot) in the MQ4 decoder's positions (gen_npu_dec.MQ4_ELEM, then its KPERM), one replay group; call j's column c
computes features 256 j + 32 c + [0, 32) from the rows as the loader leaves them (gate, then up dgu bytes further).
Down: K part 0 (passes 0-3, H's first 4096 features), the first 512 rows, weights w_orig = s1 / awq (H256 (s2 / 16 w)),
the rotation and awq folded in, so the NPU multiplies its own H. H is checked against float64 (rel RMS), the down partial
within half a bf16 ulp of its bfp16 operands (H as the NPU wrote it, w_orig quantized).
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import bfp16_check as B  # noqa: E402
import gen_bfp16_encode as GE  # noqa: E402
import gen_npu_gemm as N  # noqa: E402
import gen_rot  # noqa: E402
import hfq_index  # noqa: E402
import hrx_paths  # noqa: E402

TOKENS, F, FFN, HID = 2048, 7168, 17408, 5120
KS = (36, 36, 36, 20)
PRE = "model.language_model.layers.%d.mlp."
TARGET = "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11"
REC = 4 * 128 * 144   # H bytes per (M block, pass) of one 16-token slab row: [slab 4][k-block 128][144]


def h256(x):
    """The natural-order Walsh-Hadamard transform over the last axis's 256-groups (unnormalized)."""
    s = x.shape
    y = x.reshape(-1, 256).astype(np.float64)
    h = 1
    while h < 256:
        y = y.reshape(-1, 256 // (2 * h), 2, h)
        y = np.stack([y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]], 2)
        h *= 2
    return y.reshape(s)


def rotate(x, awq):
    """gen_rot's x' = s2 / 16 H256(s1 x / awq), in float64."""
    sg = np.frombuffer(gen_rot.signs(), np.float32).astype(np.float64)
    k = x.shape[-1]
    return np.tile(sg[256:], k // 256) * h256(np.tile(sg[:256], k // 256) * x / awq)


def unrotate_rows(w, awq):
    """Rows w (rotated basis) -> w_orig with w . rotate(x) = w_orig . x: s1 / awq (H256 (s2 / 16 w))."""
    sg = np.frombuffer(gen_rot.signs(), np.float32).astype(np.float64)
    k = w.shape[-1]
    return np.tile(sg[:256], k // 256) / awq * h256(np.tile(sg[256:], k // 256) * w)


def mq4_values(rows):
    """[n][K / 256][136] (signed nibbles) -> f64 [n][K] in element order."""
    d = np.stack([rows[..., 0:2].copy().view(np.float16)[..., 0], rows[..., 4:6].copy().view(np.float16)[..., 0]], -1)
    q = rows[..., 8:].astype(np.int64)
    lo, hi = q & 15, q >> 4
    code = np.stack([lo, hi], -1).reshape(rows.shape[:-1] + (256,))
    code = np.where(code >= 8, code - 16, code)
    return (np.repeat(d.astype(np.float64), 128, -1) * code).reshape(rows.shape[0], -1)


def run(cmd, env, tag):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if r.returncode:
        sys.exit(f"{tag} failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")


def compile_image(cfg, src, img, groups):
    if os.path.exists(img):
        return
    GE.GROUPS = groups
    open(src, "w").write(N.gen(cfg))
    run([hrx_paths.LOOM_COMPILE, src, f"--root=@{cfg.entry}", TARGET, f"--output={img}"],
        dict(hrx_paths.env(), **N.loom_env(cfg)), "loom-compile " + os.path.basename(img))


def xdna(img, cfg, binds, out_idx, out, env):
    run([hrx_paths.XDNA_RUN, "--columns=8", "--image=" + img, "--entry=" + cfg.entry, "--binding_memory=system"]
        + ["--binding=" + b for b in binds] + [f"--output={out_idx}={out}"], env, "iree-xdna-run " + os.path.basename(img))


def h_values(h):
    """H bytes [M block][pass 7][slab 4][k-block 128][16 rows][E, 8 m] -> f64 [TOKENS][F]."""
    r = h[:TOKENS // 64 * 7 * REC].reshape(TOKENS // 64, 7, 4, 128, 16, 9)
    v = r[..., 1:].view(np.int8).astype(np.float64) * np.exp2(r[..., 0].astype(np.float64) - 133)[..., None]
    return v.transpose(0, 2, 4, 1, 3, 5).reshape(TOKENS, F)   # [M block][slab][row] x [pass][k-block][8]


def main():
    work = sys.argv[1]
    model = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/Downloads/qwen3.8-27b.mq4-xts")
    layer = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    calls = int(sys.argv[4]) if len(sys.argv) > 4 else F // 256
    os.makedirs(work, exist_ok=True)
    idx = {e[0]: e for e in hfq_index.read(model)}
    tg, tu, td = (idx[PRE % layer + n + "_proj.weight"] for n in ("gate", "up", "down"))
    awq = {n: np.array(hfq_index.raw(model, idx[PRE % layer + n + "_proj.awq_scale.weight"]).view(np.float16), np.float64)
           for n in ("gate", "up", "down")}
    assert np.array_equal(awq["gate"], awq["up"]), "gate and up share the norm's rotated activations"
    rb, f0, dgu = HID // 256 * 136, FFN - F, tu[3] - tg[3]
    env = hrx_paths.env()

    # ---- gate / up + swiglu
    D = N.decoder("MQ4")
    sw = N.Config(8, TOKENS // 64, KS, 5, mu=1, fuse=2, swiglu=True, dgu=dgu, swcol=32, hpass=F // 1024, hord="",
                  fmt="MQ4")
    with N.np_override(4):
        compile_image(sw, f"{work}/sw.loom", f"{work}/sw.xdna", (5,))
        GE.GROUPS = (5,)
        sa, sw_w, sc = N.stream_bytes(sw)
        words = max(fw * nrec for c in range(8) for fw, nrec in (N.fuse_record(sw, cov)
                                                                for cov in N.fuse_cover(sw, c).values()))
        rng = np.random.default_rng(3)
        x = rng.standard_normal((TOKENS, HID)) * np.exp(rng.uniform(-1, 1, (1, HID)))
        xr = rotate(x, awq["gate"])
        # the decoder's positions: element 32 s + MQ4_ELEM[p] at position 32 s + p, then its k-block order per pass
        perm = (32 * np.arange(HID // 32)[:, None] + np.array(D.MQ4_ELEM)[None, :]).reshape(-1)
        kperm = np.concatenate([128 * p + np.array(D.KPERM) for p in range(5)])
        a_in = xr[:, perm].reshape(TOKENS, HID // 8, 8)[:, kperm].reshape(TOKENS, HID)
        ra, _ = B.reference(a_in, "act", TOKENS, list(KS), 5, 64, True)
        assert ra.size == sa
        ra.tofile(f"{work}/a.bin")
        np.zeros(sw_w, np.uint8).tofile(f"{work}/w.bin")
        np.zeros(len(KS) * 8 * words, np.int32).tofile(f"{work}/d.bin")
    # the rows as the loader leaves them: gate's and up's nibbles signed, the bytes between them (gate's awq) as stored
    span = np.array(hfq_index.raw(model, (None, 0, 0, tg[3], tu[3] + tu[4] - tg[3])))
    for t in (tg, tu):
        o = t[3] - tg[3]
        v = span[o:o + t[4]].reshape(-1, 136)
        v[:, 8:] ^= 0x88
    span = np.concatenate([span, np.zeros(4096, np.uint8)])
    H = np.zeros(TOKENS // 64 * 7 * REC + sc, np.uint8)
    for j in range(calls):
        a0 = (f0 + 256 * j) * rb
        span[a0:a0 + dgu + 256 * rb + 4096].tofile(f"{work}/r.bin")
        c0 = (j // 4) * REC + (j % 4) * 32 * 144
        H[c0:c0 + sc].tofile(f"{work}/c.bin")
        xdna(f"{work}/sw.xdna", sw, [f"{work}/{n}.bin" for n in "awcdr"], 2, f"{work}/c_out.bin", env)
        H[c0:c0 + sc] = np.fromfile(f"{work}/c_out.bin", np.uint8)
        print(f"sw call {j} done", flush=True)
    H.tofile(f"{work}/h.bin")
    hv = h_values(H)
    wg = mq4_values(hfq_index.mq4_rows(model, tg, slice(f0, FFN)))
    wu = mq4_values(hfq_index.mq4_rows(model, tu, slice(f0, FFN)))
    g, u = xr @ wg.T, xr @ wu.T
    href = g / (1 + np.exp(-g)) * u
    n = 256 * calls
    rms = np.sqrt(np.mean((hv[:, :n] - href[:, :n]) ** 2) / np.mean(href[:, :n] ** 2))
    print(f"H: features 0..{n - 1}: rel RMS {rms:.2e} vs float64 (bfp16 operands and H)")
    if calls < 16:
        return
    # ---- down, part 0 (passes 0-3), rows 0-511
    passes, rows = 4, 512
    dn = N.Config(8, TOKENS // 64, KS, passes, mu=1, ain="h", hpass=F // 1024)
    with N.np_override(4):
        compile_image(dn, f"{work}/dn4.loom", f"{work}/dn4.xdna", (passes,))
        GE.GROUPS = (passes,)
        da, dw_b, dc = N.stream_bytes(dn)
        wd = mq4_values(hfq_index.mq4_rows(model, td, slice(0, rows)))
        wo = unrotate_rows(wd, awq["down"])
        # the fold: w . rotate(h) = w_orig . h (spot check on random h)
        hh = rng.standard_normal((4, FFN))
        assert np.allclose(rotate(hh, awq["down"]) @ wd.T, hh @ wo.T, rtol=1e-9, atol=1e-9)
        wk = wo[:, f0:f0 + 1024 * passes]
        rw, _ = B.reference(wk, "wgt", rows, list(KS), passes, N.TN, True)
        assert rw.size == dw_b
        rw.tofile(f"{work}/dw.bin")
        H[:da].tofile(f"{work}/da.bin")
        np.zeros(dc // 4, np.float32).tofile(f"{work}/dc.bin")
        xdna(f"{work}/dn4.xdna", dn, [f"{work}/da.bin", f"{work}/dw.bin", f"{work}/dc.bin"], 2, f"{work}/dc_out.bin", env)
        import npu_gemm_check as C
        raw = (np.fromfile(f"{work}/dc_out.bin", np.uint16).astype(np.uint32) << 16).view(np.float32)
        part = C.unpack_c(raw, 8, TOKENS // 64)
        ref = hv[:, :1024 * passes] @ C.value(wk).T
        ulp = np.exp2(np.floor(np.log2(np.maximum(np.abs(ref), 1e-30))) - 7)
        err = float(np.max(np.abs(part - ref) / ulp))
        exact = href[:, :1024 * passes] @ wk.T
        rms = np.sqrt(np.mean((part - exact) ** 2) / np.mean(exact ** 2))
        print(f"down part 0, rows 0..{rows - 1}: max |err| {err:.3f} bf16 ulp vs its bfp16 operands; rel RMS {rms:.2e} "
              "vs float64 (exact H, w_orig)")
        sys.exit(0 if err <= 0.501 else 1)


if __name__ == "__main__":
    main()
