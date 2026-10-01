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
H256 = hadamard(256)
def rot256(x): return x @ H256
def q_asym(x, bits, group):
    s = x.shape; g = x.reshape(*s[:-1], s[-1] // group, group)
    lo, hi = g.min(-1, keepdims=True), g.max(-1, keepdims=True); lv = 2 ** bits - 1
    sc = (hi - lo) / lv; sc[sc == 0] = 1
    return (np.round((g - lo) / sc) * sc + lo).reshape(s)
def sink(Kq, Kfull, n): Kq = Kq.copy(); Kq[:n] = Kfull[:n]; return Kq
q8 = lambda x: q_sym(x, 8, 128)
def run(name, Qx, Kx):
    print(f"{name:52s} rel RMS {err(attn(Qx, Kx, V)):.2e}", flush=True)
run("K4 H128 tok-half, Q8  [current kv4]", q8(rot(Q)), q_sym(rot(Kc), 4, 128))
run("K4 H128 tok-half, Q fp16", rot(Q), q_sym(rot(Kc), 4, 128))
run("K4 H256 rotated-half scale, Q8", q8(rot256(Q)), q_sym(rot256(Kc), 4, 128))
run("K4 H256 32-group, Q8", q8(rot256(Q)), q_sym(rot256(Kc), 4, 32))
run("K4 H128 asym tok-half, Q8", q8(rot(Q)), q_asym(rot(Kc), 4, 128))
run("K4 H256 asym 32-group, Q8", q8(rot256(Q)), q_asym(rot256(Kc), 4, 32))
run("K4 H128 tok-half + sink 4 tokens fp16, Q8", q8(rot(Q)), sink(q_sym(rot(Kc), 4, 128), rot(Kc), 4))
run("K4 H128 tok-half + sink 64 tokens fp16, Q8", q8(rot(Q)), sink(q_sym(rot(Kc), 4, 128), rot(Kc), 64))
run("K8 (reference)", q8(Q), q_sym(Kc, 8, 128))
