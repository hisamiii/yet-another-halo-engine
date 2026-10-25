#!/usr/bin/env python3
"""Check the GPU kernels of the NPU FFN block (emit_npu_ffn.py) in a set built for B tokens, on layer 4's down rows.

usage: npu_ffn_gpu_check.py <model.mq4-xts> <set> <work dir>

normqx_5120: q and d equal normq_5120's bit for bit; xo is the rotated row in the MQ4 decoder's positions (float64
  oracle: the norm, the awq divide and the rotation, then gen_npu_dec.MQ4_ELEM per 32 values).
iu4_kresk_320_<G / 256>: down over the first G features of its rows equals iu4_kres_320_68 bit for bit when that one's
  activations are rotq_17408 of the same rows with the other features zero (rotq_<G> of the first G columns).
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bfp16_check as BC  # noqa: E402
import gen_npu_dec  # noqa: E402
import hfq_index  # noqa: E402

FFN, HID, LAYER = 17408, 5120, 4


def geom(st):
    out = {}
    for line in open(os.path.join(st, "dispatch.txt")).read().splitlines() + open(os.path.join(st, "npu.txt")).read().splitlines():
        f = line.split()
        out[f[0]] = [int(v) for v in f[1:]]
    return out


def run(model, hal, grid, wg, binds):
    mins = ",".join(str(m) for m, _ in binds)
    cmd = [BC.GPURUN, "npu-ffn", "--", BC.HALRUN, model, hal, grid, str(wg), mins] + [b for _, b in binds]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if "hal_run: ok" not in r.stdout:
        sys.exit(f"hal_run {os.path.basename(hal)} failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")


def main():
    model, st, work = sys.argv[1:4]
    os.makedirs(work, exist_ok=True)
    g = geom(st)
    F, B = g["npuffn"][0], g["npuffn"][1]
    G = FFN - F
    rng = np.random.default_rng(4)
    w = lambda n: os.path.join(work, n)
    idx = {e[0]: e for e in hfq_index.read(model)}
    pre = f"model.language_model.layers.{LAYER}.mlp."
    sg = w("signs.f32")
    open(sg, "wb").write(open(os.path.join(st, "rot_signs.bin"), "rb").read())
    s = np.fromfile(sg, np.float32)
    s1, s2 = s[:256].astype(np.float64), s[256:].astype(np.float64)

    # ---- normqx_5120 vs normq_5120 and the float64 rotation
    x = (rng.standard_normal((B, HID)) * 3).astype(np.float32)
    nw = (1 + 0.2 * rng.standard_normal(HID)).astype(np.float32)
    aw = np.asarray(hfq_index.raw(model, idx[pre + "gate_proj.awq_scale.weight"]).view(np.float16), np.float32)
    for n, a in (("x.f32", x), ("nw.f32", nw), ("aw.f32", aw)):
        a.tofile(w(n))
    qb, db, xb = B * HID // 2, B * HID // 128 * 4, B * HID * 2
    outs = {}
    for hal in ("normq_5120.hal", "normqx_5120.hal"):
        xo = [(xb, f"o:{xb}:{w('xo.f16')}")] if "qx" in hal else []
        run(model, os.path.join(st, hal), str(B), 320,
            [(B * HID * 4, f"f:{w('x.f32')}"), (HID * 4, f"f:{w('nw.f32')}"), (HID * 4, f"f:{w('aw.f32')}"),
             (2048, f"f:{sg}"), (qb, f"o:{qb}:{w(hal + '.q')}"), (db, f"o:{db}:{w(hal + '.d')}")] + xo)
        outs[hal] = open(w(hal + ".q"), "rb").read() + open(w(hal + ".d"), "rb").read()
    print("normqx q / d:", "bit-identical" if outs["normq_5120.hal"] == outs["normqx_5120.hal"] else "DIFFER")
    xd = x.astype(np.float64)
    nx = (xd / np.sqrt((xd * xd).mean(1, keepdims=True) + 1e-6)) * nw
    nx = nx.astype(np.float16).astype(np.float64) / aw
    r = nx.reshape(B, -1, 256) * s1
    h = 1
    while h < 256:   # H256
        r = r.reshape(B, -1, 256 // (2 * h), 2, h)
        r = np.stack([r[..., 0, :] + r[..., 1, :], r[..., 0, :] - r[..., 1, :]], -2)
        h *= 2
    rot = (r.reshape(B, -1, 256) * s2).reshape(B, HID)
    want = rot.reshape(B, -1, 32)[:, :, gen_npu_dec.MQ4_ELEM].reshape(B, HID)
    got = np.fromfile(w("xo.f16"), np.float16).astype(np.float64).reshape(B, HID)
    rel = np.sqrt(((got - want) ** 2).mean() / (want ** 2).mean())
    print(f"normqx xo vs float64: rel rms {rel:.2e}, max |d| / rms {np.abs(got - want).max() / np.sqrt((want ** 2).mean()):.2e}")

    # ---- iu4_kresk vs iu4_kres over zeroed features
    dn = idx[pre + "down_proj.weight"]
    hfq_index.mq4_rows(model, dn).tofile(w("down.mq4"))
    daw = np.asarray(hfq_index.raw(model, idx[pre + "down_proj.awq_scale.weight"]).view(np.float16), np.float32)
    daw.tofile(w("daw.f32"))
    daw[:G].tofile(w("dawg.f32"))
    hx = (rng.standard_normal((B, FFN)) * 0.5).astype(np.float16)
    hx[:, G:] = 0
    hx.tofile(w("h.f16"))
    np.ascontiguousarray(hx[:, :G]).tofile(w("hg.f16"))
    res = rng.standard_normal((B, HID)).astype(np.float32)
    res.tofile(w("res.f32"))
    for k, hin, a in ((FFN, "h.f16", "daw.f32"), (G, "hg.f16", "dawg.f32")):
        rg = g[f"rotq_{k}.hal"]
        q, d = B * k // 2, B * k // 128 * 4
        run(model, os.path.join(st, f"rotq_{k}.hal"), f"{rg[2]},{B}", rg[1],
            [(B * k * 2, f"f:{w(hin)}"), (k * 4, f"f:{w(a)}"), (2048, f"f:{sg}"), (q, f"o:{q}:{w(f'aq{k}')}"),
             (d, f"o:{d}:{w(f'ad{k}')}")])
    ob = B * HID * 4
    for hal, k in ((f"iu4_kres_320_{FFN // 256}.hal", FFN), (f"iu4_kresk_320_{G // 256}.hal", G)):
        grp = g.get("iu4grp_" + hal, [0])[0]
        gx = HID // 16 // g[hal][1]
        grid = f"{grp},{B // 128},{gx // grp}" if grp else f"{gx},{B // 128}"
        run(model, os.path.join(st, hal), grid, 256,
            [(HID * FFN // 256 * 136, f"f:{w('down.mq4')}"), (B * k // 2, f"f:{w(f'aq{k}')}"),
             (B * k // 128 * 4, f"f:{w(f'ad{k}')}"), (ob, f"f:{w('res.f32')}"), (ob, f"o:{ob}:{w(hal + '.out')}")])
    a, b = (open(w(n), "rb").read() for n in (f"iu4_kres_320_{FFN // 256}.hal.out", f"iu4_kresk_320_{G // 256}.hal.out"))
    ya, yb = np.frombuffer(a, np.float32), np.frombuffer(b, np.float32)
    print(f"kresk vs kres: {'bit-identical' if a == b else 'DIFFER'} ({np.count_nonzero(ya != yb)} of {ya.size} differ, "
          f"|y - resid| rms {np.sqrt(((ya - res.ravel()) ** 2).mean()):.3f})")


if __name__ == "__main__":
    main()
