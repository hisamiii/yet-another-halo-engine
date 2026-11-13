#!/usr/bin/env python3
"""Check gen_npu_attn.py on the NPU: O and l bit for bit against a CPU emulation of the core arithmetic, the time a body,
and with --real the normalized output against a float64 softmax(q k / 16) v of a real attention dump.

usage: npu_attn_check.py <workdir> <mc> <nb> [--image built.xdna --built-nb N] [--real <dump prefix> [--kv-head g]]
                        [--inputs <npz from npu_attn_encode_check.py>] [--seed s]

Records (what the GPU encoder must produce; bfp16ebs8 fragments = 8 rows x [exponent byte, 8 int8 mantissas], value
m 2^(E - 133), one exponent a row of 8):
  Q  [64 fragments] fragment 2d + rb: rows 8 rb .. of the core's 16, dims 8d .., scaled log2(e) / 16
  C  f32 [2, 8, 8]  the QK chain inits a row: 250 - F0 - (|q| max|k| + 1) (a Cauchy-Schwarz bound: U = S + C stays in
                    the core exp's range without a max pass; F0 = 1/2, P = 2^(U - 128))
  K  record j [128 fragments] = keys 32j ..: fragment 64s + 2d + kp = keys 16s + 8kp .., dims 8d ..
  V  record b [128 fragments] (P.V group n = b // 16, dim slice rp = (14 + b) % 16): fragment 2st + dp, st = 4bs + 2s + kp
                    over K record r = 16(n - 1) - 1 + bs, keys 16s + 8kp ..: [8 dims 16rp + 8dp ..][8 keys] (zeros for
                    r < 0 or r >= M)
Record 0 of each stream is the payload of each core (Q | C | the park starts 2 x 8 x 256 B of f32 128 = P 1 a fake key;
nb in park half 1's first word), then 1 zero record if mc is even, then nb + 1 K records / nb V records, each stream
carrying the lower or the upper 64 fragments. Output a pair: 34 KB records, core h at 17 KB h: O [16 rp][4 c = 2dp + rbp]
[8][8] f32, then t1, t0 (cycles) and at + 640 the l block [8][8] (cols 0-3 rows 0-7, 4-7 rows 8-15, x 2^-63).
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import gen_npu_attn as G  # noqa: E402
import hrx_paths  # noqa: E402

f32 = np.float32
F0 = 0.5
L2E = np.log2(np.e)
OSTRIDE = 17408                                          # a core's egress: 16 O records + the stats record


def mma(acc, am, ea, wm, ew):
    """mma.bfp16ebs8.m8n8k8: acc [8, 8] f32 += A W^T. The 8-term dot is an exact integer D on the grid 2^e,
    e = Ea + Ew - 266, and enters the add as a float of exponent e + 12: the operand of the smaller exponent is rounded
    (nearest even) to the other's 24-bit grid, the sum is exact and rounds to f32."""
    d = (am.astype(np.int64) @ wm.astype(np.int64).T).astype(np.float64)
    e = (ea.reshape(8, 1) + ew.reshape(1, 8)).astype(np.float64) - 266
    a = acc.astype(np.float64)
    ex = np.where(a != 0, np.frexp(a)[1] - 1.0, -1e9)
    g = np.exp2(np.maximum(ex, e + 12) - 23)
    return ((np.rint(a / g) + np.rint(d * np.exp2(e) / g)) * g).astype(f32)


def frag_bytes(m, e):
    """fragments [n, 8, 8] + exponents [n, 8] -> the 72 B a fragment the FIFO pops"""
    return np.concatenate([np.asarray(e).reshape(-1, 8, 1), np.asarray(m)], axis=2).astype(np.uint8).reshape(-1)


def bfp(x):
    """bfp16ebs8 of 8-lane blocks as [(E, m int x 8)], E the f32 exponent field of the block max (the core's push)"""
    out = []
    for r in x.reshape(-1, 8).astype(np.float64):
        E = int(((np.abs(r.astype(f32)).view(np.uint32) >> 23) & 0xFF).max())
        m = np.rint(r / 2.0 ** (E - 133))
        if m.max() > 127:
            E += 1
            m = np.rint(r / 2.0 ** (E - 133))
        out.append((E, m.astype(np.int64)))
    return out


def encode(x):
    """[n, 8, 8] -> (m [n, 8, 8], e [n, 8]) as the GPU encoder writes inputs (gen_bfp16_encode: f32 x, E the exponent
    field of the row max, m = clamp(roundeven(x 2^(133 - E)), -128, 127), all zero for E = 0)"""
    x = np.ascontiguousarray(x, f32).reshape(-1, 8, 8)
    e = ((np.abs(x).view(np.uint32) >> 23) & 0xFF).max(2).astype(np.int64)
    sc = np.where(e > 0, np.exp2(133.0 - e), 0.0)[..., None]
    return np.clip(np.rint(x.astype(np.float64) * sc), -128, 127).astype(np.int64), e


def decode(m, e):
    return m.astype(np.float64) * np.exp2(e.astype(np.float64) - 133)[..., None]


def bf16(v):
    b = np.asarray(v, f32).view(np.uint32).astype(np.uint64)
    b = b + 0x7FFF + ((b >> 16) & 1)
    return ((b >> 16).astype(np.uint32) << 16).view(f32)


def core_exp2(U):
    """the softmax's p for U, op for op as gen_npu_attn.softmax issues them"""
    U = np.asarray(U, f32)
    mb = (bf16(U).view(np.uint32) >> 16).astype(np.int64)
    mb = np.clip(np.where(mb >= 0x8000, mb - 0x10000, mb), 0x4300, 0x437E)
    k = mb - 0x4300
    x = bf16((U - (mb.astype(np.uint32) << 16).view(f32)).astype(f32))
    v = bf16((f32(G.EXP_B1) + f32(G.EXP_A2) * x).astype(f32))
    E = ((k + 127).astype(np.uint32) << 23).view(f32)
    xE = ((((x.view(np.uint32) >> 16).astype(np.int64) + (k << 7)) & 0xFFFF).astype(np.uint32) << 16).view(f32)
    return (E + xE * v).astype(f32)


HALF = [(np.where(np.arange(8)[:, None] // 4 == h, 64, 0) * np.ones((8, 8), np.int64),
         np.where(np.arange(8) // 4 == h, 64, 0)) for h in range(2)]   # rows 4h .. 4h + 3 = 2^-63


def emulate(Q, C, K, V, nb, rot):
    """one core's O [16, 4, 8, 8] and l [8, 8]: Q (m, e), C [2, 8, 8], K / V [(m, e)] a body; rot: core 1 (its P.V
    lanes start at the P region middle)"""
    Qm, Qe = Q
    park = np.full((2, 2, 4, 8, 8), 128.0, f32)          # [half][s][c]
    R = [[(np.zeros((8, 8), np.int64), np.zeros(8, np.int64)) for _ in range(G.RGP // G.FR)] for _ in range(2)]
    O = np.zeros((16, 4, 8, 8), f32)
    lacc = np.zeros((8, 8), f32)
    for j in range(nb):
        (Km, Ke), (Vm, Ve) = K[j], V[j]
        pu = (j % G.NSL) * 576
        reg, base = pu // G.RGP, (pu % G.RGP) // G.FR
        if j % 2 == 1:                                   # l over the two slots before this one's, in the other region
            b0 = ((pu - 576) % G.RGP) // G.FR
            for kb in range(8):
                for rb in range(2):
                    lacc = mma(lacc, *R[1 - reg][b0 + 2 * kb + rb], *HALF[rb])
        for s in range(2):                               # softmaxes of the previous body's S
            for c in range(4):
                pm, pe = np.zeros((8, 8), np.int64), np.zeros(8, np.int64)
                for i, (E, m) in enumerate(bfp(core_exp2(park[j % 2, s, c].reshape(-1)))):
                    pm[i], pe[i] = m, E
                R[reg][base + 4 * s + c] = (pm, pe)
        for s in range(2):
            for c in range(4):                           # QK round: c = 2 kbp + rb
                rb, kp = c % 2, c // 2
                acc = C[rb].copy()
                for d in range(32):
                    acc = mma(acc, Qm[2 * d + rb], Qe[2 * d + rb], Km[64 * s + 2 * d + kp], Ke[64 * s + 2 * d + kp])
                park[(j + 1) % 2, s, c] = acc
        rp = (14 + j) % 16                               # P.V round: c = 2 dbp + rbp
        for c in range(4):
            rbp, dp = c % 2, c // 2
            acc = O[rp, c]
            for st in (list(range(32, 64)) + list(range(32)) if rot else range(64)):
                acc = mma(acc, *R[1 - reg][2 * st + rbp], Vm[2 * st + dp], Ve[2 * st + dp])
            O[rp, c] = acc
    return O, lacc


def frags(rng, n, elo, ehi):
    return rng.integers(-128, 128, (n, 8, 8)), rng.integers(elo, ehi, (n, 8))


def random_inputs(ncores, nb, seed):
    """small Q (S within a few units of the row constant: P finite at any nb), distinct K / V records a body"""
    rng = np.random.default_rng(seed)
    Qs = [frags(rng, 64, 120, 121) for _ in range(ncores)]
    Cs = [np.repeat((250.0 - F0 - 12.0 - rng.integers(0, 4, 16)).astype(f32).reshape(2, 8, 1), 8, 2) for _ in range(ncores)]
    K = [frags(rng, 128, 127, 128) for _ in range(nb + 1)]
    V = [frags(rng, 128, 100, 106) for _ in range(nb)]
    return dict(Qs=Qs, Cs=Cs, K=K, V=V)


def real_inputs(prefix, ncores, nb, g):
    """records from a real attention dump (loom_prefill YAH_ATTN_DUMP, qnorm + qsplit set) for KV head g: the cores'
    rows = (token, head) rows of g's 6 query heads; the keys = the M 32-key records before the chunk"""
    H, KVH, D = 24, 4, 256
    B = os.path.getsize(prefix + ".out") // (H * D * 2)
    T = os.path.getsize(prefix + ".kpool") // (KVH * D * 2)
    start = T - B
    nq = -(-16 * ncores // 6)
    raw = np.fromfile(prefix + ".q", np.uint8)
    q16 = raw[:B * 12288].view(np.float16).reshape(B, H, D).astype(np.float64)      # f16(q w / 16), dims 0-63 unused
    rot = raw[B * 12288:B * 18432].view(f32).reshape(B, H, 64).astype(np.float64)   # raw q of the rotary dims
    sums = raw[B * 18432:B * 19968].view(f32).reshape(B, H, 16).astype(np.float64)  # sums of q^2 a 16 dims
    qnw = np.fromfile(prefix + ".qnw", f32).astype(np.float64)
    kp = np.memmap(prefix + ".kpool", np.float16, "r").reshape(T, KVH, D)
    vp = np.memmap(prefix + ".vpool", np.float16, "r").reshape(KVH, T // 16, D, 16)
    rows = np.unique(np.linspace(0, B - 1, nq).astype(int))
    Q = 16.0 * q16[rows]
    inv = 1.0 / 10000000.0 ** (2.0 * np.arange(32) / 64.0)
    ang = (start + rows)[:, None].astype(np.float64) * inv[None, :]
    a, b = rot[rows][:, :, :32] * qnw[:32], rot[rows][:, :, 32:64] * qnw[32:64]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    Q[:, :, :32], Q[:, :, 32:64] = a * c - b * s, a * s + b * c
    Q *= (1.0 / np.sqrt(sums[rows].sum(-1) / D + 1e-6))[:, :, None]
    rows = Q[:, 6 * g:6 * g + 6, :].reshape(-1, D)
    M = 16 * (nb // 16 - 1) - 1
    nk = 32 * M
    k0 = start - nk
    Kf = np.asarray(kp[k0:start, g, :], np.float64)
    t = np.arange(k0, start)
    Vf = np.asarray(vp[g][t // 16, :, t % 16], np.float64)
    kmax = np.linalg.norm(decode(*encode(Kf.reshape(-1, 8, 8))).reshape(nk, D), axis=1).max()
    Qs, Cs, ref = [], [], []
    for cr in range(ncores):
        q = rows[16 * cr:16 * cr + 16] * (L2E / 16.0)
        m, e = encode(np.stack([q[8 * rb:8 * rb + 8, 8 * d:8 * d + 8] for d in range(32) for rb in range(2)]))
        Qs.append((m, e))
        qb = decode(m, e)
        qn = np.array([np.linalg.norm(np.concatenate([qb[2 * d + r // 8][r % 8] for d in range(32)])) for r in range(16)])
        Cs.append(np.repeat((250.0 - F0 - (qn * kmax + 1.0)).astype(f32).reshape(2, 8, 1), 8, 2))
        S = rows[16 * cr:16 * cr + 16] @ Kf.T / 16.0
        P = np.exp(S - S.max(1, keepdims=True))
        ref.append(P @ Vf / P.sum(1, keepdims=True))
    zero = (np.zeros((128, 8, 8), np.int64), np.zeros((128, 8), np.int64))
    K = [encode(np.stack([Kf[32 * j + 16 * s + 8 * kq:32 * j + 16 * s + 8 * kq + 8, 8 * d:8 * d + 8]
                          for s in range(2) for d in range(32) for kq in range(2)])) if j < M else zero for j in range(nb + 1)]
    V = []
    for bb in range(nb):
        n, rp = bb // 16, (14 + bb) % 16
        f = np.zeros((128, 8, 8))
        for st in range(64):
            bs, s, kq = st // 4, (st // 2) % 2, st % 2
            r = 16 * (n - 1) - 1 + bs
            if n and 0 <= r < M:
                for dp in range(2):
                    f[2 * st + dp] = Vf[32 * r + 16 * s + 8 * kq:32 * r + 16 * s + 8 * kq + 8,
                                        16 * rp + 8 * dp:16 * rp + 8 * dp + 8].T
        V.append(encode(f))
    print(f"real: {nk} keys from {k0}, {ncores} cores x 16 rows of KV head {g}", flush=True)
    return dict(Qs=Qs, Cs=Cs, K=K, V=V, ref=ref)


def unpack(O, l):
    """O [16, 4, 8, 8] + the l block -> [16 rows, 256 dims] normalized"""
    out = np.zeros((16, 256))
    for rp in range(16):
        for c in range(4):
            rbp, dp = c % 2, c // 2
            out[8 * rbp:8 * rbp + 8, 16 * rp + 8 * dp:16 * rp + 8 * dp + 8] = O[rp, c]
    return out / (np.concatenate([l[:, 0], l[:, 4]]).astype(np.float64) * 2.0 ** 63)[:, None]


def streams(inp, mc, nb, nb_built):
    """the 4 stream bindings (K lower, K upper, V lower, V upper), sized for nb_built"""
    mcf = mc + (1 - mc % 2)
    st = [[], [], [], []]
    for w in range(2 * mc):
        pl = np.concatenate([frag_bytes(*inp["Qs"][w]), inp["Cs"][w].reshape(-1).view(np.uint8),
                             np.full(2 * 8 * 64, 128.0, f32).view(np.uint8)])
        assert pl.size == G.KB + G.VB
        pl[G.KB + 512 + 2048:G.KB + 512 + 2052] = np.array([nb], np.uint32).view(np.uint8)
        st[w % 2].append(pl[:G.KB])
        st[2 + w % 2].append(pl[G.KB:])
    for s in range(4):
        h = s % 2
        st[s] += [np.zeros(G.KB, np.uint8)] * (mcf - mc)
        rec = inp["K"][:nb + 1] if s < 2 else inp["V"][:nb]
        st[s] += [frag_bytes(m[64 * h:64 * h + 64], e[64 * h:64 * h + 64]) for m, e in rec]
        full = (mcf + nb_built + 1) * G.KB if s < 2 else (mcf + nb_built) * G.VB
        b = np.concatenate(st[s])
        st[s] = np.concatenate([b, np.zeros(full - b.size, np.uint8)])
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("workdir")
    ap.add_argument("mc", type=int)
    ap.add_argument("nb", type=int)
    ap.add_argument("--image", help="a built image (else emit + compile one for nb)")
    ap.add_argument("--built-nb", type=int, help="the nb --image was built for (patched down to nb)")
    ap.add_argument("--real", help="a real attention dump prefix")
    ap.add_argument("--kv-head", type=int, default=0)
    ap.add_argument("--inputs", help="the GPU encoder's records (npu_attn_encode_check.py)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    W, mc, nb = a.workdir, a.mc, a.nb
    os.makedirs(W, exist_ok=True)
    nb_built = a.built_nb or nb
    if a.image:
        img = open(a.image, "rb").read()
        open(f"{W}/p.xdna", "wb").write(G.patch_nb(img, mc, nb_built, nb) if nb_built != nb else img)
    else:
        open(f"{W}/p.loom", "w").write(G.emit(mc, nb))
        G.compile_xdna(f"{W}/p.loom", f"{W}/p.xdna")
    if a.inputs:
        z = {k: v.astype(np.int64) if v.dtype.kind in "iu" else v for k, v in np.load(a.inputs).items()}
        inp = dict(Qs=list(zip(z["Qm"], z["Qe"])), Cs=list(z["C"]), K=list(zip(z["Km"], z["Ke"])),
                   V=list(zip(z["Vm"], z["Ve"])), ref=list(z["ref"]))
    elif a.real:
        inp = real_inputs(a.real, 2 * mc, nb, a.kv_head)
    else:
        inp = random_inputs(2 * mc, nb, a.seed)
    args = [hrx_paths.XDNA_RUN, f"--columns={max(2, min(8, mc))}", f"--image={W}/p.xdna", "--entry=probe",
            "--binding_memory=system", "--invocation_count=1"]
    for s, b in enumerate(streams(inp, mc, nb, nb_built)):
        b.tofile(f"{W}/s{s}.bin")
        args.append(f"--binding={W}/s{s}.bin")
    for p in range(mc):
        np.zeros(G.NEG * 1024, np.uint8).tofile(f"{W}/o{p}.bin")
        args.append(f"--binding={W}/o{p}.bin")
    args += [f"--output={4 + p}={W}/out{p}.bin" for p in range(mc)]
    r = subprocess.run(args, capture_output=True, text=True, env=hrx_paths.env(), timeout=600)
    if r.returncode:
        sys.exit("run failed\n" + r.stdout[-1500:] + r.stderr[-1500:])
    bad, per, errs = 0, [], []
    for p in range(mc):
        o = np.fromfile(f"{W}/out{p}.bin", np.uint8)
        for h in range(2):
            w = 2 * p + h
            rw = o[OSTRIDE * h:OSTRIDE * (h + 1)]
            O = rw[:16384].view(f32).reshape(16, 4, 8, 8)
            t1, t0 = (int(v) for v in rw[16384:16392].view(np.uint32))
            l = rw[16384 + G.LSL:16384 + G.LSL + 256].view(f32).reshape(8, 8)
            Oe, le = emulate(inp["Qs"][w], inp["Cs"][w], inp["K"], inp["V"], nb, h == 1)
            ob = int(((O != Oe) & ~(np.isnan(O) & np.isnan(Oe))).sum())
            lb = int((l != le).sum())
            bad += ob + lb
            per.append(((t1 - t0) & 0xFFFFFFFF) / nb)
            msg = f"  core {w}: {per[-1]:.1f} cycles a body, O misses {ob} of {O.size}, l misses {lb} of 64"
            if "ref" in inp:
                er = np.abs(unpack(O, l) - inp["ref"][w])
                errs.append(er)
                msg += f"; vs f64 mean |err| {er.mean():.2e} max {er.max():.2e}"
            if ob or lb or "ref" in inp:
                print(msg)
    print(f"mc={mc} nb={nb}: {2 * mc} cores, a body min {min(per):.1f} mean {sum(per) / len(per):.1f} max {max(per):.1f}; "
          f"misses {bad}" + (f"; real vs f64 mean |err| {np.mean(errs):.3e} max {np.max(errs):.3e}" if errs else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
