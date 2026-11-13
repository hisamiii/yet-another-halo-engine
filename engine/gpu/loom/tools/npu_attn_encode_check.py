#!/usr/bin/env python3
"""Check gen_npu_attn_encode.py on the GPU against a real attention dump (loom_prefill YAH_ATTN_DUMP, qnorm + qsplit):
K / V records bit for bit against npu_attn_check.encode, the payloads against float64 (Q within a mantissa step, C,
the park starts, nb, the pad slot). The keys sit in a small shuffled page pool, so the page table is exercised.
Writes <workdir>/inputs.npz for `npu_attn_check.py <workdir> <mc> <nb> --inputs <workdir>/inputs.npz` (the NPU run on
the GPU's bytes, against the float64 softmax(q k / 16) v of the same rows).

usage: npu_attn_encode_check.py <dump prefix> <workdir> [key=value ...]   (gen_npu_attn_encode.Call; B, start_pos and
                                                                          pages come from the dump)
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_npu_attn_encode as GE  # noqa: E402
import npu_attn_check as NC  # noqa: E402
sys.path.insert(0, os.path.join(HERE, ".."))
import hrx_paths  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
EMIT = os.path.join(HERE, "..", "emit_hal.py")
HALRUN = os.path.join(ROOT, "engine", "build", "hal_run")
GPURUN = os.path.join(ROOT, "engine", "run", "gpu_run.sh")
H, KVH, D, PAGE, REC = 24, 4, 256, 256, GE.REC
f32 = np.float32


def gpu(W, c, kind, bindings):
    src = os.path.join(W, kind + ".loom")
    with open(src, "w") as f:
        f.write({"k": GE.gen_k, "v": GE.gen_v, "q": GE.gen_q}[kind](c))
    r = subprocess.run([sys.executable, EMIT, src, os.path.join(W, kind)], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    hal = r.stdout.strip().splitlines()[-1]
    gx, wg = c.grid(kind)
    cmd = [GPURUN, "npu-attn-encode", "--", HALRUN, "-", hal, str(gx), str(wg),
           ",".join(str(v) for v in c.footprints(kind))] + bindings
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=hrx_paths.env())
    if "hal_run: ok" not in r.stdout:
        sys.exit(f"hal_run {kind} failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    for ln in r.stdout.splitlines():
        if "ms per dispatch" in ln:                      # HAL_RUN_ITERS=N
            print(f"{kind}: {ln.strip()}")


def frags(b):
    """[n x 72 B] -> (m [n, 8, 8], e [n, 8])"""
    b = b.reshape(-1, 8, 9)
    return b[:, :, 1:].view(np.int8).astype(np.int64), b[:, :, 0].astype(np.int64)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    prefix, W = sys.argv[1], sys.argv[2]
    os.makedirs(W, exist_ok=True)
    B = os.path.getsize(prefix + ".out") // (H * D * 2)
    T = os.path.getsize(prefix + ".kpool") // (KVH * D * 2)
    c = GE.parse(sys.argv[3:])
    c.B, c.start_pos = B, T - B
    nk = 32 * c.M
    c.pages = 2 * -(-(c.k0 + nk) // PAGE)                # every other page unused
    c.check()
    nh, ncores = len(c.heads), 2 * c.mc
    # the pool: logical page lp at physical ptab[lp] (a shuffle), unused pages noise
    rng = np.random.default_rng(5)
    ptab = rng.permutation(c.pages).astype(np.int32)
    kp = np.memmap(prefix + ".kpool", np.float16, "r").reshape(T, KVH * D)
    vp = np.memmap(prefix + ".vpool", np.float16, "r").reshape(KVH, T // 16, D, 16)
    npg = -(-(c.k0 + nk) // PAGE)
    kl = np.asarray(kp[:npg * PAGE], np.float16)
    vl = np.asarray(vp[:, :npg * 16], np.float16)
    kphys = rng.standard_normal((c.pages * PAGE, KVH * D)).astype(np.float16)
    vphys = rng.standard_normal((KVH, c.pages * 16, D, 16)).astype(np.float16)
    for lp in range(npg):
        kphys[ptab[lp] * PAGE:(ptab[lp] + 1) * PAGE] = kl[lp * PAGE:(lp + 1) * PAGE]
        vphys[:, ptab[lp] * 16:(ptab[lp] + 1) * 16] = vl[:, lp * 16:(lp + 1) * 16]
    fs = {n: os.path.join(W, n) for n in ("kp.bin", "vp.bin", "ptab.bin", "qeps.bin", "ks.bin", "vs.bin", "kn.bin",
                                         "ks2.bin", "vs2.bin")}
    kphys.tofile(fs["kp.bin"])
    vphys.tofile(fs["vp.bin"])
    ptab.tofile(fs["ptab.bin"])
    np.array([1e-6], f32).tofile(fs["qeps.bin"])
    gpu(W, c, "k", [f"f:{fs['kp.bin']}", f"f:{fs['ptab.bin']}", f"o:{c.ks_bytes()}:{fs['ks.bin']}",
                    f"o:{4 * max(c.M, 1)}:{fs['kn.bin']}"])
    gpu(W, c, "v", [f"f:{fs['vp.bin']}", f"f:{fs['ptab.bin']}", f"o:{c.vs_bytes()}:{fs['vs.bin']}"])
    gpu(W, c, "q", [f"f:{prefix}.q", f"f:{prefix}.qnw", f"f:{fs['qeps.bin']}", f"f:{fs['kn.bin']}",
                    f"io:{fs['ks.bin']}:{fs['ks2.bin']}", f"io:{fs['vs.bin']}:{fs['vs2.bin']}"])
    ks = np.fromfile(fs["ks2.bin"], np.uint8).reshape(2, c.slots, REC)
    vs = np.fromfile(fs["vs2.bin"], np.uint8).reshape(2, c.slots, REC)
    kn = np.fromfile(fs["kn.bin"], f32)
    for n in ("kp.bin", "vp.bin", "ks.bin", "vs.bin"):
        os.remove(fs[n])
    bad = 0

    # K / V records: bit for bit
    Kf = kl[c.k0:c.k0 + nk, c.g * D:(c.g + 1) * D].astype(np.float64)
    Vf = vl[c.g].transpose(0, 2, 1).reshape(-1, D)[c.k0:c.k0 + nk].astype(np.float64)
    zero = (np.zeros((128, 8, 8), np.int64), np.zeros((128, 8), np.int64))
    Kr = [NC.encode(np.stack([Kf[32 * j + 16 * s + 8 * kq:32 * j + 16 * s + 8 * kq + 8, 8 * d:8 * d + 8]
                              for s in range(2) for d in range(32) for kq in range(2)])) if j < c.M else zero
          for j in range(c.nb + 1)]
    Vr = []
    for b in range(c.nb):
        n, rp = b // 16, (14 + b) % 16
        f = np.zeros((128, 8, 8))
        for st in range(64):
            bs, s, kq = st // 4, (st // 2) % 2, st % 2
            r = 16 * (n - 1) - 1 + bs
            if n and 0 <= r < c.M:
                for dp in range(2):
                    f[2 * st + dp] = Vf[32 * r + 16 * s + 8 * kq:32 * r + 16 * s + 8 * kq + 8,
                                        16 * rp + 8 * dp:16 * rp + 8 * dp + 8].T
        Vr.append(NC.encode(f))
    for name, img, recs in (("K", ks, Kr), ("V", vs, Vr)):
        want = np.stack([np.stack([NC.frag_bytes(m[64 * h:64 * h + 64], e[64 * h:64 * h + 64]) for m, e in recs])
                         for h in range(2)])
        got = img[:, c.mcf:c.mcf + len(recs)]
        nbad = int((got != want).sum())
        bad += nbad
        print(f"{name} records: {nbad} of {got.size} bytes differ")
    kd = np.stack([NC.decode(*r) for r in Kr[:c.M]])      # [M, 128 frags, 8 keys, 8 dims]
    kn2 = np.zeros((c.M, 32))
    for j in range(c.M):
        for f in range(128):
            s, kq = f // 64, f % 2
            kn2[j, 16 * s + 8 * kq:16 * s + 8 * kq + 8] += (kd[j, f] ** 2).sum(1)
    ker = np.abs(kn.astype(np.float64) / kn2.max(1) - 1).max()
    print(f"key norms: max rel err {ker:.2e}")
    bad += ker > 1e-5
    kmax = np.sqrt(kn2.max())

    # payloads: Q against float64, C, the park starts, nb, the pad slot. Q's exponents must match and its mantissas
    # round float64 (clamped at 127) except in the rotary dims: the attention's fast sin / cos (afn) at positions near
    # 2^17 miss float64 by up to ~2 steps, and the encoder repeats the attention's RoPE op for op
    raw = np.fromfile(prefix + ".q", np.uint8)
    q16 = raw[:B * 12288].view(np.float16).reshape(B, H, D).astype(np.float64)
    rot = raw[B * 12288:B * 18432].view(f32).reshape(B, H, 64).astype(np.float64)
    sums = raw[B * 18432:B * 19968].view(f32).reshape(B, H, 16).astype(np.float64)
    qnw = np.fromfile(prefix + ".qnw", f32).astype(np.float64)
    inv = 1.0 / 10000000.0 ** (2.0 * np.arange(32) / 64.0)
    Qs, Cs, refs, qoff, qexp, rexp, cer, rmax = [], [], [], 0, 0, 0, 0.0, 0.0
    for w in range(ncores):
        hd, toks = c.heads[w % nh], c.t0 + 16 * (w // nh) + np.arange(16)
        q = 16.0 * q16[toks, hd]
        ang = (c.start_pos + toks)[:, None].astype(np.float64) * inv[None, :]
        a, b = rot[toks, hd, :32] * qnw[:32], rot[toks, hd, 32:64] * qnw[32:64]
        q[:, :32], q[:, 32:64] = a * np.cos(ang) - b * np.sin(ang), a * np.sin(ang) + b * np.cos(ang)
        q *= (1.0 / np.sqrt(sums[toks, hd].sum(-1) / D + 1e-6))[:, None]
        S = q @ Kf.T / 16.0
        P = np.exp(S - S.max(1, keepdims=True))
        refs.append(P @ Vf / P.sum(1, keepdims=True))
        qr = q * (NC.L2E / 16.0)
        qf = np.stack([qr[8 * rb:8 * rb + 8, 8 * d:8 * d + 8] for d in range(32) for rb in range(2)])
        m, e = frags(ks[w % 2, w // 2])
        Qs.append((m, e))
        x = qf.reshape(64, 8, 8) / np.exp2(e - 133.0)[..., None]
        er = np.abs(x - m)
        er[np.abs(x) > 127.5] = 0                        # clamped
        de = e - NC.encode(qf)[1]
        qexp += int((de[16:] != 0).sum()) + int((np.abs(de[:16]) > 1).sum())
        rexp += int((de[:16] != 0).sum())
        qoff += int((er[16:] > 0.5 + 1e-3).sum())        # fragments 2d + rb, d >= 8: dims 64 ..
        rmax = max(rmax, float(er[:16].max()))
        qb = NC.decode(m, e)
        qn = np.array([np.linalg.norm(np.concatenate([qb[2 * d + r // 8][r % 8] for d in range(32)])) for r in range(16)])
        C = vs[w % 2, w // 2, :512].view(f32).reshape(2, 8, 8)
        Cs.append(C.copy())
        cw = (248.5 - qn * kmax).reshape(2, 8, 1)
        cer = max(cer, float(np.abs(C - cw).max()))
        park = vs[w % 2, w // 2, 512:].view(f32).copy()
        nbw = park[512:513].view(np.uint32)[0]
        park[512] = 128.0
        bad += int((park != 128.0).sum()) + (nbw != c.nb)
    pad = int(ks[:, c.mc:c.mcf].sum()) + int(vs[:, c.mc:c.mcf].sum())
    print(f"Q: {qexp} exponents differ from float64 ({rexp} rotary ones by 1 allowed), {qoff} mantissas (dims 64 ..) "
          f"off by more than rounding, rotary dims max {rmax:.2f} steps; C max |err| {cer:.2e}; "
          f"pad slot nonzero bytes {pad}")
    bad += qexp + qoff + (rmax > 3) + pad + (cer > 1e-3)
    # the GPU's bytes must be exactly what npu_attn_check.streams builds from the parsed inputs
    inp = dict(Qs=Qs, Cs=Cs, K=Kr, V=Vr, ref=refs)
    st = NC.streams(inp, c.mc, c.nb, c.nb)
    rb = sum(int((st[s] != (ks if s < 2 else vs)[s % 2].reshape(-1)).sum()) for s in range(4))
    print(f"streams rebuilt from the parsed inputs: {rb} bytes differ")
    bad += rb
    def pk(fr):
        return np.stack([m for m, e in fr]).astype(np.int8), np.stack([e for m, e in fr]).astype(np.uint8)
    (Qm, Qe), (Km, Ke), (Vm, Ve) = pk(Qs), pk(Kr), pk(Vr)
    np.savez(os.path.join(W, "inputs.npz"), Qm=Qm, Qe=Qe, C=np.stack(Cs), Km=Km, Ke=Ke, Vm=Vm, Ve=Ve, ref=np.stack(refs))
    print(f"mc={c.mc} nb={c.nb} g={c.g} k0={c.k0} heads={c.heads} t0={c.t0}: {'OK' if not bad else 'FAIL'}; {W}/inputs.npz")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
