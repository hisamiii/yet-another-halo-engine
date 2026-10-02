import numpy as np, sys
exec(open("gdn_ref.py").read().split("hs = [")[0])
def rnd(x, mode):
    if mode == "f32": return x.astype(np.float32).astype(np.float64)
    if mode == "f16": return x.astype(np.float16).astype(np.float64)
    if mode == "bf16":
        b = x.astype(np.float32).view(np.uint32); b = ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
        return b.view(np.float32).astype(np.float64)
def chunked_q(h, mode, C=64, solve="f32"):
    """log-space gating; every matmul input rounded to `mode`, f32 accumulate; S kept f32"""
    kn, qn, vv, al, be = head(h)
    S = np.zeros((D, D)); out = np.empty((T, D)); r = lambda x: rnd(x, mode); f = lambda x: rnd(x, "f32")
    for c0 in range(0, T, C):
        K, Q, V, a, b = kn[c0:c0+C], qn[c0:c0+C], vv[c0:c0+C], al[c0:c0+C], be[c0:c0+C]
        G = np.cumsum(np.log(a))
        Gm = np.exp(np.minimum(G[:, None] - G[None, :], 0))
        KK = f(r(K) @ r(K).T); QK = f(r(Q) @ r(K).T)
        A = np.tril((b[:, None] * KK) * Gm, -1)
        Tm = f(np.linalg.inv(np.eye(C) + f(A)))               # forward substitution in f32
        W = f(r(Tm) @ r(b[:, None] * K * np.exp(G)[:, None]))
        U = f(r(Tm) @ r(b[:, None] * V))
        Vn = f(U - f(r(W) @ r(S).T))
        P = np.tril(QK * Gm)
        out[c0:c0+C] = f(f(r(Q * np.exp(G)[:, None]) @ r(S).T) + f(r(P) @ r(Vn)))
        S = f(np.exp(G[-1]) * S + f(r((Vn * np.exp(G[-1] - G)[:, None]).T) @ r(K)))
    return out
for h in (0, 17, 47):
    ref, _ = recurrent(h)
    k32 = np.linalg.norm(raw[:, h] - ref) / np.linalg.norm(ref)
    line = f"head {h:2d}: current kernel (f32) {k32:.1e}"
    for mode in ("f32", "bf16", "f16"):
        c = chunked_q(h, mode)
        line += f" | chunked {mode} inputs {np.linalg.norm(c-ref)/np.linalg.norm(ref):.1e}"
    print(line, flush=True)
