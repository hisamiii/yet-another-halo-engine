import numpy as np, sys
T, KH, H, D = 2048, 16, 48, 128
conv = np.fromfile("l0.conv", np.float32).reshape(T, 10240).astype(np.float64)
kq = np.fromfile("l0.kq", np.float32).reshape(T, KH, 3).astype(np.float64)
ab = np.fromfile("l0.ab", np.float32).reshape(T, H, 2).astype(np.float64)
raw = np.fromfile("l0.raw", np.float32).reshape(T, H, D).astype(np.float64)
q = conv[:, :2048].reshape(T, KH, D); k = conv[:, 2048:4096].reshape(T, KH, D); v = conv[:, 4096:].reshape(T, H, D)
def head(h):
    kh = h % KH
    kn = k[:, kh] * kq[:, kh, 0:1]                  # k_hat
    qn = q[:, kh] * kq[:, kh, 1:2]                  # q_hat (1/sqrt(128) folded)
    return kn, qn, v[:, h], ab[:, h, 0], ab[:, h, 1]
def recurrent(h):
    kn, qn, vv, al, be = head(h)
    S = np.zeros((D, D)); out = np.empty((T, D))
    for t in range(T):
        S *= al[t]
        d = be[t] * (vv[t] - S @ kn[t])
        S += np.outer(d, kn[t])
        out[t] = S @ qn[t]
    return out, S
def chunked(h, C=64):
    """WY / UT-transform chunkwise gated delta rule (FLA chunk_gated_delta_rule)"""
    kn, qn, vv, al, be = head(h)
    S = np.zeros((D, D)); out = np.empty((T, D))
    for c0 in range(0, T, C):
        K, Q, V, a, b = kn[c0:c0+C], qn[c0:c0+C], vv[c0:c0+C], al[c0:c0+C], be[c0:c0+C]
        g = np.cumprod(a)                            # gamma_i = prod_{j<=i} alpha_j
        Gm = g[:, None] / g[None, :]                 # gamma_i / gamma_j
        A = np.tril((b[:, None] * (K @ K.T)) * Gm, -1)
        Tm = np.linalg.inv(np.eye(C) + A)            # (I + A)^-1, lower triangular
        W = Tm @ (b[:, None] * K * g[:, None])       # decayed-key correction
        U = Tm @ (b[:, None] * V)
        Vn = U - W @ S.T                             # new values (pseudo-values)
        P = np.tril((Q @ K.T) * Gm)                  # intra-chunk attention incl. diagonal
        out[c0:c0+C] = (Q * g[:, None]) @ S.T + P @ Vn
        S = g[-1] * S + (Vn.T * (g[-1] / g)[None, :]) @ K
    return out, S
hs = [int(x) for x in sys.argv[1:]] or [0, 17, 47]
for h in hs:
    r, Sr = recurrent(h)
    c, Sc = chunked(h)
    e_k = np.linalg.norm(r - raw[:, h]) / np.linalg.norm(raw[:, h])
    e_c = np.linalg.norm(c - r) / np.linalg.norm(r)
    print(f"head {h:2d}: recurrence(f64) vs kernel rel {e_k:.2e} | chunked(C=64) vs recurrence rel {e_c:.2e} | final state rel {np.linalg.norm(Sc-Sr)/np.linalg.norm(Sr):.2e} | min alpha {head(h)[3].min():.4f}")
