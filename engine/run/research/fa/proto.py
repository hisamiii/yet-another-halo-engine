# kv4a8 numerics prototype: attention-core output error (pre-gate) vs exact f64, real layer-3 pp8192 data
import numpy as np, itertools, sys
D = "/home/q/yah-scratch/attn8k/"; T = 8192
rng = np.random.default_rng(0)
qpos = np.sort(rng.choice(np.arange(256, T), 512, replace=False))
Q = np.fromfile(D + "l3.aq", np.float32).reshape(T, 24, 256)[qpos].astype(np.float64) * 0.0625
K = np.fromfile(D + "l3.ak16", np.float16).reshape(T, 4, 256).astype(np.float64)
V = np.fromfile(D + "l3.av16", np.float16).reshape(T, 4, 256).astype(np.float64)

def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n: H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)
H128 = hadamard(128)
def rot(x):  # block-diagonal H128 over the two 128-dim halves (last axis)
    return np.concatenate([x[..., :128] @ H128, x[..., 128:] @ H128], -1)

def q_sym(x, bits, group):  # symmetric per-row-group, last axis split in groups
    s = x.shape; g = x.reshape(*s[:-1], s[-1] // group, group)
    lv = 2 ** (bits - 1) - 1
    sc = np.abs(g).max(-1, keepdims=True) / lv; sc[sc == 0] = 1
    return (np.round(g / sc) * sc).reshape(s)
def q_chan(x, bits):  # per-channel (over tokens) midrange, x [T, H, D]
    lo, hi = x.min(0, keepdims=True), x.max(0, keepdims=True); c = (lo + hi) / 2
    lv = 2 ** bits - 2; sc = (hi - lo) / lv; sc[sc == 0] = 1
    return np.round((x - c) / sc) * sc + c

def attn(Qx, Kx, Vx, pq8=False):
    out = np.empty((len(qpos), 24, 256))
    for h in range(24):
        kv = h // 6
        s = Qx[:, h] @ Kx[:, kv].T                       # [512, T]
        s[np.arange(T)[None, :] > qpos[:, None]] = -np.inf
        m = s.max(1, keepdims=True); p = np.exp(s - m)
        if pq8: p = np.round(p * 255) / 255             # uint8 P with scale 1/255 (p <= 1)
        out[:, h] = (p @ Vx[:, kv]) / p.sum(1, keepdims=True)
    return out
ref = attn(Q, K, V)
def err(o): return np.sqrt(((o - ref) ** 2).mean() / (ref ** 2).mean())

Kc = K - K.mean(0, keepdims=True)                     # centring (exact for softmax)
res = []
def run(name, Qx, Kx, Vx, pq8=False, vrot=False):
    o = attn(Qx, Kx, Vx, pq8)
    if vrot: o = rot(o)                                # undo V rotation (H symmetric orthonormal)
    res.append((name, err(o))); print(f"{name:52s} rel RMS {res[-1][1]:.2e}", flush=True)

q8 = lambda x: q_sym(x, 8, 128)
run("fp16 (rounding only)", Q, K, V)
run("int8 K(c,tok-half) + Q8 / int8 V chan  [current int8]", q8(Q), q_sym(Kc, 8, 128), q_chan(V, 8))
run("int8 current + P uint8", q8(Q), q_sym(Kc, 8, 128), q_chan(V, 8), pq8=True)
for kname, kq, rq in [("K4 tok-half", q_sym(Kc, 4, 128), False), ("K4 tok-32", q_sym(Kc, 4, 32), False),
                      ("K4 H tok-half", q_sym(rot(Kc), 4, 128), True), ("K4 H tok-32", q_sym(rot(Kc), 4, 32), True)]:
    Qx = q8(rot(Q)) if rq else q8(Q)
    run(f"{kname:16s} + Q8, V fp16", Qx, kq, V)
Vr = rot(V)
for vname, vq, vr in [("V4 chan", q_chan(V, 4), False), ("V4 tok-half", q_sym(V - V.mean(0, keepdims=True), 4, 128) + V.mean(0, keepdims=True), False),
                      ("V4 H chan", q_chan(Vr, 4), True), ("V4 H tok-half", q_sym(Vr, 4, 128), True), ("V4 H tok-32", q_sym(Vr, 4, 32), True)]:
    run(f"K fp16, {vname:14s}", Q, K, vq, vrot=vr)
