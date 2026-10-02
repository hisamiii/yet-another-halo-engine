#!/usr/bin/env python3
"""Check the quantized-KV decode kernels of gen_decode_attn.py (kappend_q, vappend_q, part_q + the shared reduce).

The reference is a numpy model of the prefill's kv8a16 / kv4a16 formats (gen_kvq.py), with a scrambled page table.

usage: dattn_q_check.py <model.gguf> <workdir> <bits 8|4|kb,vb> [T] [pos]
(the model is only opened by hal_run; no tensor is bound)
"""
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_decode_attn as A  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
EMIT = os.path.join(HERE, "..", "emit_hal.py")
HALRUN = os.path.join(ROOT, "engine", "build", "hal_run")
GPURUN = os.path.join(ROOT, "engine", "run", "gpu_run.sh")
BENCH = int(os.environ.get("DATTN_BENCH", "0"))
f32, f16 = np.float32, np.float16
PERM4 = (0, 2, 1, 3)


def build(work, name, text):
    src = os.path.join(work, name + ".loom")
    open(src, "w").write(text)
    r = subprocess.run([sys.executable, EMIT, src, os.path.join(work, "hal"), "nop=0"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stdout[-1500:] + r.stderr[-1500:])
    return r.stdout.strip().splitlines()[0]


def run(model, hal, grid, wg, binds, mins, tag="", bench=False):
    cmd = [GPURUN, "dattn-check", "--", HALRUN, model, hal, grid, str(wg), ",".join(map(str, mins))] + binds
    env = dict(os.environ)
    if bench and BENCH:
        env["HAL_RUN_ITERS"] = str(BENCH)
        time.sleep(1)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    if "hal_run: ok" not in r.stdout:
        raise SystemExit(r.stdout[-1500:] + r.stderr[-1500:])
    for line in r.stdout.splitlines():
        if "ms per dispatch" in line:
            print(f"{tag}: {float(line.split()[1]) * 1000:.1f} us per dispatch ({grid} workgroups)")


def hadamard(n=256):
    i = np.arange(n)
    pc = np.vectorize(lambda x: bin(x).count("1"))(i[:, None] & i[None, :])
    return np.where(pc % 2, -1.0, 1.0)


H = hadamard()


def quant_k(k, m, bits):
    """one token's K [1024] f32 -> (dwords, scales f16 pairs) as gen_kvq yah_kq8 / yah_kq4"""
    x = k.astype(f16).astype(f32) - m
    if bits == 8:
        xc = x.reshape(8, 128)
        amax = np.abs(xc).max(1).astype(f32)
        s = np.where(amax / f32(127) > 0, amax / f32(127), f32(1)).astype(f32)
        u = (np.round(xc * (f32(1) / s)[:, None]) + 128).astype(np.uint32).reshape(256, 4)
        dw = np.zeros(256, np.uint32)
        for b, d in enumerate(PERM4):
            dw |= u[:, d] << (8 * b)
        sc = np.stack([(s * 256).astype(f16), (s * -384).astype(f16)], 1).reshape(16)
        return dw, sc
    xr = (x.reshape(4, 256).astype(np.float64) @ H.T / 16.0).astype(f32).reshape(32, 32)
    lo, hi = xr.min(1), xr.max(1)
    s0 = (hi - lo) / f32(15)
    s = np.where(s0 > 0, s0, f32(1)).astype(f32)
    u = np.clip(np.round((xr - lo[:, None]) * (f32(1) / s)[:, None]), 0, 15).astype(np.uint32).reshape(128, 8)
    dw = np.zeros(128, np.uint32)
    for i in range(8):
        dw |= u[:, i] << (4 * (i // 2) + (16 if i % 2 else 0))
    S0 = (s * 16).astype(f32)
    sc = np.stack([S0.astype(f16), (lo - S0).astype(f16)], 1).reshape(64)
    return dw, sc


def deq_k(dw, sc, bits):
    """-> K - m as the attention sees it, [1024] (K4: rotated, per head)"""
    sc = sc.astype(f32).reshape(-1, 2)
    if bits == 8:
        u = np.stack([(dw >> (8 * b)) & 255 for b in range(4)], 1)   # [256][slot]
        x = np.zeros((256, 4), f32)
        for b, d in enumerate(PERM4):
            x[:, d] = u[:, b]
        x = x.reshape(8, 128)
        return ((x - 128) * (sc[:, 0] / 256)[:, None]).reshape(1024)
    u = np.stack([(dw >> (4 * (i // 2) + (16 if i % 2 else 0))) & 15 for i in range(8)], 1).astype(f32).reshape(32, 32)
    return (u * (sc[:, 0] / 16)[:, None] + (sc[:, 1] + sc[:, 0])[:, None]).reshape(1024)


def quant_vtile(v, bits):
    """v [16 keys][n channels] -> (dwords [n][nw], stats f16 [n][2]) as gen_kvq yah_vq*"""
    lv, off, umax = (14, 7, 14) if bits == 4 else (254, 128, 255)
    smul, cmul = (16.0, -23.0) if bits == 4 else (256.0, -384.0)
    v = v.astype(f32)
    mx, mn = v.max(0), v.min(0)
    cen = (mx + mn) * f32(0.5)
    s0 = (mx - mn) / f32(lv)
    s = np.where(s0 > 0, s0, f32(1)).astype(f32)
    sf, cf = s.astype(f16).astype(f32), cen.astype(f16).astype(f32)
    u = np.clip(np.round((v - cf) * (f32(1) / sf)) + off, 0, umax).astype(np.uint32)   # [16][n]
    nw = 2 if bits == 4 else 4
    dw = np.zeros((v.shape[1], nw), np.uint32)
    for w in range(nw):
        for p in range(8 if bits == 4 else 4):
            if bits == 4:
                key, sh = 8 * w + (2 * p if p < 4 else 2 * (p - 4) + 1), 4 * p
            else:
                key, sh = 4 * w + PERM4[p], 8 * p
            dw[:, w] |= u[key] << sh
    st = np.stack([(sf * smul).astype(f16), (cf + sf * f32(cmul)).astype(f16)], 1)
    return dw, st


def deq_vtile(dw, st, bits):
    st = st.astype(f32)
    n = dw.shape[0]
    u = np.zeros((16, n), f32)
    for w in range(dw.shape[1]):
        for p in range(8 if bits == 4 else 4):
            if bits == 4:
                key, sh, mk = 8 * w + (2 * p if p < 4 else 2 * (p - 4) + 1), 4 * p, 15
            else:
                key, sh, mk = 4 * w + PERM4[p], 8 * p, 255
            u[key] = (dw[:, w] >> sh) & mk
    div = 16.0 if bits == 4 else 256.0
    return u * (st[:, 0] / div) + st[:, 0] + st[:, 1]


def main():
    model, work = sys.argv[1], sys.argv[2]
    kb, vb = (map(int, sys.argv[3].split(","))) if "," in sys.argv[3] else (int(sys.argv[3]),) * 2
    bits = kb
    T = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
    pos = int(sys.argv[5]) if len(sys.argv) > 5 else 700
    os.makedirs(work, exist_ok=True)
    rng = np.random.default_rng(5)
    npg, tiles, n = T // 256, T // 16, pos + 1
    nw = 2 if vb == 4 else 4
    kdw, ksn = (128, 64) if kb == 4 else (256, 16)
    ptab = rng.permutation(npg).astype(np.int32)
    prow = lambda p: ptab[p // 256] * 256 + p % 256
    m = (0.5 * rng.standard_normal(1024)).astype(f32)
    K = (rng.standard_normal((T, 1024)) + m).astype(f16)
    V = rng.standard_normal((T, 1024)).astype(f16)
    q = (0.3 * rng.standard_normal((24, 256))).astype(f32)
    gate = rng.standard_normal((24, 256)).astype(f32)
    kq = np.zeros((T, kdw), np.uint32)
    ks = np.zeros((T, ksn), f16)
    Kd = np.zeros((T, 1024), f32)
    for p in range(T):
        dw, sc = quant_k(K[p].astype(f32), m, bits)
        kq[prow(p)], ks[prow(p)] = dw, sc
        Kd[p] = deq_k(dw, sc, bits)
    vq = np.zeros((4, tiles, 256, nw), np.uint32)
    vs = np.zeros((4, tiles, 256, 2), f16)
    Vd = V.astype(f32).copy()
    curt = pos // 16
    for t in range(curt):
        dw, st = quant_vtile(V[16 * t:16 * t + 16], vb)
        pt = ptab[t // 16] * 16 + t % 16
        vq[:, pt], vs[:, pt] = dw.reshape(4, 256, nw), st.reshape(4, 256, 2)
        Vd[16 * t:16 * t + 16] = deq_vtile(dw, st, vb)
    vopen = np.zeros((1024, 16), f16)
    nk = pos - 16 * curt + 1
    vopen[:, :nk] = V[16 * curt:pos + 1].T
    vopen[:, nk:] = 7.0                      # stale keys past pos: masked
    f = lambda nm, a: (a.tofile(os.path.join(work, nm)), os.path.join(work, nm))[1]
    fq, fkq, fks, fvq, fvs, fvo, fp = (f("q", q), f("kq", kq), f("ks", ks), f("vq", vq), f("vs", vs),
                                       f("vo", vopen), f("pt", ptab))
    fpos, fg = f("pos", np.array([pos], np.int32)), f("gate", gate)
    used = pos // 256 + 1
    acc_b, ml_b = npg * 24 * 256 * 4, npg * 24 * 2 * 4
    sz = [24 * 256 * 4, kq.nbytes, ks.nbytes, vq.nbytes, vs.nbytes, vopen.nbytes, npg * 4, 4, acc_b, ml_b]
    hp = build(work, "part_q", A.gen_part_q(T, kb, vb))
    run(model, hp, f"4,{used}", 256, [f"f:{x}" for x in (fq, fkq, fks, fvq, fvs, fvo, fp, fpos)] +
        [f"o:{acc_b}:{work}/acc", f"o:{ml_b}:{work}/ml"], sz, "part_q", bench=True)
    hr = build(work, "reduce", A.gen("reduce", T))
    run(model, hr, "24", 256, [f"f:{work}/acc", f"f:{work}/ml", f"f:{fg}", f"f:{fpos}", f"o:{24 * 256 * 4}:{work}/out"],
        [acc_b, ml_b, 24 * 256 * 4, 4, 24 * 256 * 4])
    out = np.fromfile(f"{work}/out", f32).reshape(24, 256)

    def attn(Kx, Vx, qx):
        ref = np.zeros((24, 256))
        for hh in range(24):
            kv = hh // 6
            s = Kx[:n, kv * 256:kv * 256 + 256].astype(np.float64) @ qx[hh].astype(np.float64) / 16.0
            pr = np.exp(s - s.max())
            pr /= pr.sum()
            ref[hh] = (pr @ Vx[:n, kv * 256:kv * 256 + 256].astype(np.float64)) / (1.0 + np.exp(-gate[hh].astype(np.float64)))
        return ref
    qr = q if bits == 8 else (q.astype(np.float64) @ H.T / 16.0)   # K4 codes are rotated
    ref = attn(Kd, Vd, qr)
    exact = attn(K.astype(f32) - m, V, q)
    err = np.abs(out - ref).max() / np.abs(ref).max()
    qerr = np.abs(out - exact).max() / np.abs(exact).max()
    print(f"part_q k{kb}v{vb}+reduce T={T} pos={pos} pages {used} ptab {ptab.tolist()}: "
          f"vs dequant model {err:.2e}{'   <-- FAIL' if not err < 1e-4 else ''}; vs fp16 KV {qerr:.2e} (quantization)")

    # kappend at p2
    p2 = min(517, T - 1)
    kx = (rng.standard_normal(1024) + m).astype(f32)
    hk = build(work, "kappend_q", A.gen_kappend_q(T, bits))
    run(model, hk, "1", 128, [f"f:{f('kx', kx)}", f"f:{f('m', m)}", f"io:{fkq}:{work}/kq2", f"io:{fks}:{work}/ks2",
                              f"f:{fp}", f"f:{f('pos2', np.array([p2], np.int32))}"],
        [4096, 4096, kq.nbytes, ks.nbytes, npg * 4, 4])
    kq2 = np.fromfile(f"{work}/kq2", np.uint32).reshape(T, kdw)
    ks2 = np.fromfile(f"{work}/ks2", f16).reshape(T, ksn)
    dw, sc = quant_k(kx, m, bits)
    r = prow(p2)
    others = np.array_equal(np.delete(kq2, r, 0), np.delete(kq, r, 0)) and np.array_equal(np.delete(ks2, r, 0), np.delete(ks, r, 0))
    dq = np.abs(deq_k(kq2[r], ks2[r], bits) - deq_k(dw, sc, bits)).max()
    print(f"kappend_q{bits} pos={p2} row={r}: codes {'exact' if np.array_equal(kq2[r], dw) else 'differ'}, "
          f"scales {'exact' if np.array_equal(ks2[r], sc) else 'differ'}, max dequant diff {dq:.2e}, other rows "
          f"{'untouched' if others else 'CHANGED'}{'' if others and dq < 1e-2 else '   <-- FAIL'}")

    # vappend: a mid-tile key (open tile only), then a tile's 16th key (quantize)
    hv = build(work, "vappend_q", A.gen_vappend_q(T, vb))
    for p3 in (16 * 37 + 9, 16 * 37 + 15):
        vx = rng.standard_normal(1024).astype(f32)
        vo = np.zeros((1024, 16), f16)
        vo[:, :p3 % 16] = rng.standard_normal((1024, p3 % 16)).astype(f16)
        run(model, hv, "4", 256, [f"f:{f('vx', vx)}", f"io:{f('vo3', vo)}:{work}/vo4", f"io:{fvq}:{work}/vq2",
                                  f"io:{fvs}:{work}/vs2", f"f:{fp}", f"f:{f('pos3', np.array([p3], np.int32))}"],
            [4096, vo.nbytes, vq.nbytes, vs.nbytes, npg * 4, 4])
        vo4 = np.fromfile(f"{work}/vo4", f16).reshape(1024, 16)
        vq2 = np.fromfile(f"{work}/vq2", np.uint32).reshape(vq.shape)
        vs2 = np.fromfile(f"{work}/vs2", f16).reshape(vs.shape)
        evo = vo.copy()
        evo[:, p3 % 16] = vx.astype(f16)
        evq, evs = vq.copy(), vs.copy()
        t = p3 // 16
        if p3 % 16 == 15:
            dw, st = quant_vtile(evo.T, vb)
            pt = ptab[t // 16] * 16 + t % 16
            evq[:, pt], evs[:, pt] = dw.reshape(4, 256, nw), st.reshape(4, 256, 2)
        ok = np.array_equal(vo4, evo) and np.array_equal(vs2, evs)
        nd = int((vq2 != evq).sum())
        print(f"vappend_q{vb} pos={p3}: open tile + stats {'exact' if ok else 'MISMATCH'}, code dwords differing {nd}"
              f"{'' if ok and nd == 0 else '   <-- FAIL'}")
    if os.environ.get("DQ_KEEP"):    # debugging: keep the data, plus the model inputs
        np.savez(f"{work}/model.npz", Kd=Kd, Vd=Vd, K=K, V=V, m=m, q=q, gate=gate, kq=kq, ks=ks, ptab=ptab)
        return
    for nm in os.listdir(work):
        if nm != "hal" and not nm.endswith(".loom"):
            os.remove(os.path.join(work, nm))


if __name__ == "__main__":
    main()
