# kv4a8 numerics prototype: attention-core output error (pre-gate) vs exact f64, real layer-3 pp8192 data
import numpy as np, itertools, sys
D = "/home/q/yah-scratch/attn8k/"; T = 8192
rng = np.random.default_rng(0)
qpos = np.sort(rng.choice(np.arange(256, T), 256, replace=False))
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
def q_chan_tile(x, bits, tile):  # per-channel midrange per token tile, x [T, H, D]
    t = x.reshape(T // tile, tile, *x.shape[1:]); lo, hi = t.min(1, keepdims=True), t.max(1, keepdims=True)
    c = (lo + hi) / 2; sc = (hi - lo) / (2 ** bits - 2); sc[sc == 0] = 1
    return (np.round((t - c) / sc) * sc + c).reshape(x.shape)
res = []
def run(name, Qx, Kx, Vx, pq8=False, vrot=False):
    o = attn(Qx, Kx, Vx, pq8)
    if vrot: o = rot(o)
    print(f"{name:58s} rel RMS {err(o):.2e}", flush=True)
q8 = lambda x: q_sym(x, 8, 128); q4 = lambda x, g: q_sym(x, 4, g)
K4H = q_sym(rot(Kc), 4, 128)
run("int8 config (reference point)", q8(Q), q_sym(Kc, 8, 128), q_chan(V, 8))
run("V4 chan-tile16 (K fp16)", Q, K, q_chan_tile(V, 4, 16))
run("V4 chan-tile64 (K fp16)", Q, K, q_chan_tile(V, 4, 64))
run("kv4a8: K4 H tok-half + Q8, V4 chan", q8(rot(Q)), K4H, q_chan(V, 4))
run("kv4a8: K4 H tok-half + Q8, V4 chan-tile16", q8(rot(Q)), K4H, q_chan_tile(V, 4, 16))
run("kv4a4: K4 H tok-half + Q4 H tok-half, V4 chan", q4(rot(Q), 128), K4H, q_chan(V, 4))
run("kv4a4: K4 H tok-32 + Q4 H tok-32, V4 chan", q4(rot(Q), 32), q_sym(rot(Kc), 4, 32), q_chan(V, 4))
run("kv4a4 + P uint8 (full int)", q4(rot(Q), 128), K4H, q_chan(V, 4), pq8=True)
