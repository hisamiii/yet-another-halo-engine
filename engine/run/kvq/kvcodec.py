#!/usr/bin/env python3
"""KV-cache codec library and the loom_forward_pp YAH_KV_HOOK server.

Gate v2 (engine/run/kvq/README.md) evaluates KV-cache compression schemes by fake quantization.
Each chunk's new f16 K and V rows of every attention layer go through quantize -> dequantize here.
The fp16 attention kernel then runs on the dequantized cache.
The same codec functions serve the per-layer offline evaluation (tier A), so every scheme has one implementation.

Hook usage (one persistent process per engine run):
  YAH_KV_HOOK="python3 engine/run/kvq/kvcodec.py --k kv4 --v kv4" loom_forward_pp ...
Codecs operate on f32 arrays [rows, 4 kv heads, 256].
Stateful codecs (prompt statistics from the first chunk, calibrated transforms) keep per-layer state.
This matches streaming use: a chunk is quantized when it is written.

K codecs (--k): fp16, int8, kv4, bad3, h256g32, uq (UltraQuant MXFP4), ...
V codecs (--v): fp16, int8, kv4, h256g32, uq, ...
--dump DIR also saves the f16 K/V per layer for tier A, plus the Q rows when the engine sends them (YAH_KV_HOOK_QSTRIDE).
"""
import argparse
import os
import struct
import sys

import numpy as np

MAGIC = 0x4B56484B
H, D = 4, 256


def hadamard(n):
    m = np.array([[1.0]])
    while m.shape[0] < n:
        m = np.block([[m, m], [m, -m]])
    return m / np.sqrt(n)


H256 = hadamard(256)
H128 = hadamard(128)


# ---------------------------------------------------------------- quantizers
def q_sym(x, bits, group):
    """Symmetric round-to-nearest per group along the last axis."""
    s = x.shape
    g = x.reshape(*s[:-1], s[-1] // group, group)
    lv = 2 ** (bits - 1) - 1
    sc = np.abs(g).max(-1, keepdims=True) / lv
    sc[sc == 0] = 1
    return (np.clip(np.round(g / sc), -lv, lv) * sc).reshape(s)


def q_asym(x, bits, group):
    """Asymmetric (min/max, zero point) per group along the last axis."""
    s = x.shape
    g = x.reshape(*s[:-1], s[-1] // group, group)
    lo, hi = g.min(-1, keepdims=True), g.max(-1, keepdims=True)
    lv = 2 ** bits - 1
    sc = (hi - lo) / lv
    sc[sc == 0] = 1
    return (np.round((g - lo) / sc) * sc + lo).reshape(s)


def q_chan_tile(x, levels, tile=16):
    """Per channel per `tile` tokens, midrange centre, `levels` levels (engine int4 V: 15, int8 V: 255).

    x [rows, H, D], rows a multiple of tile.
    """
    r = x.shape[0]
    t = x.reshape(r // tile, tile, *x.shape[1:])
    lo, hi = t.min(1, keepdims=True), t.max(1, keepdims=True)
    c = (lo + hi) / 2
    sc = (hi - lo) / (levels - 1)
    sc[sc == 0] = 1
    half = (levels - 1) / 2
    return (np.clip(np.round((t - c) / sc), -half, half) * sc + c).reshape(x.shape)


def q_asym_clip(x, bits, group, kappa):
    """Asymmetric per group, with the min/max range shrunk by kappa about its centre (WUSH-KV uses kappa 0.96 for K)."""
    s = x.shape
    g = x.reshape(*s[:-1], s[-1] // group, group)
    lo, hi = g.min(-1, keepdims=True), g.max(-1, keepdims=True)
    c = (lo + hi) / 2
    lo, hi = c + (lo - c) * kappa, c + (hi - c) * kappa
    lv = 2 ** bits - 1
    sc = (hi - lo) / lv
    sc[sc == 0] = 1
    return (np.clip(np.round((g - lo) / sc), 0, lv) * sc + lo).reshape(s)


def wush_transform(kc, q, gamma=1e-2):
    """WUSH-KV K transform for one KV head (arXiv 2609.38121, closed form).

    L L^T = H + g tr(H)/d I, H = sum q q^T over the GQA group's query rows.
    U Lam U^T = L^T (M + g tr(M)/d I) L, M = sum k k^T; T = Hd Lam^-1/4 U^T L^T.
    Returns T and T^-1 (k' = T k, q' = T^-T q keeps q . k).
    """
    d = kc.shape[1]
    M = kc.T @ kc
    Hq = q.T @ q
    L = np.linalg.cholesky(Hq + gamma * np.trace(Hq) / d * np.eye(d))
    A = L.T @ (M + gamma * np.trace(M) / d * np.eye(d)) @ L
    lam, U = np.linalg.eigh(A)
    T = H256 @ np.diag(np.maximum(lam, 1e-30) ** -0.25) @ U.T @ L.T
    return T, np.linalg.inv(T)


def q_sym_clip(x, group, kappa, lv=7):
    """Symmetric +-lv per group, absmax scaled by kappa (the engine's int4 K format)."""
    s = x.shape
    g = x.reshape(*s[:-1], s[-1] // group, group)
    sc = np.abs(g).max(-1, keepdims=True) * kappa / lv
    sc[sc == 0] = 1
    return (np.clip(np.round(g / sc), -lv, lv) * sc).reshape(s)


E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])


def q_mxfp4(x, group=32, c=0.156):
    """UltraQuant: E2M1 codes per `group` channels, power-of-two scale 2^round(log2(c * absmax)) (MSE-optimal clip constant c)."""
    s = x.shape
    g = x.reshape(*s[:-1], s[-1] // group, group)
    m = np.abs(g).max(-1, keepdims=True)
    e = np.round(np.log2(np.maximum(c * m, 1e-30)))
    sc = 2.0 ** e
    z = np.abs(g) / sc
    idx = np.abs(z[..., None] - E2M1).argmin(-1)
    return (np.sign(g) * E2M1[idx] * sc).reshape(s)


def sinkhorn_tile_q(x, bits=4, tile=128, iters=8):
    """KVarN (arXiv 2606.03458) per tile of `tile` tokens x 128 dims.

    Sinkhorn-balance rows (tokens) and columns (channels) to equal RMS, quantize the tile asymmetrically per row, then undo the row and column scales.
    x [rows, H, D], rows a multiple of `tile`.
    """
    r, h, d = x.shape
    t = x.reshape(r // tile, tile, h, d // 128, 128).transpose(0, 2, 3, 1, 4)   # [nt, H, halves, tok, 128]
    sr = np.ones(t.shape[:-1] + (1,))
    sc = np.ones(t.shape[:-2] + (1, 128))
    for _ in range(iters):
        b = t / (sr * sc)
        sr *= np.sqrt((b ** 2).mean(-1, keepdims=True)) + 1e-30
        b = t / (sr * sc)
        sc *= np.sqrt((b ** 2).mean(-2, keepdims=True)) + 1e-30
    b = t / (sr * sc)
    bq = q_asym(b, bits, 128) * sr * sc
    return bq.transpose(0, 3, 1, 2, 4).reshape(r, h, d)


# ---------------------------------------------------------------- codecs
class Codec:
    """Per attention layer: encode-decode one chunk [rows, H, D] (f32)."""

    def __init__(self, name):
        self.name = name
        self.state = {}

    def __call__(self, layer, x, q=None):
        n = self.name
        if n == "fp16":
            return x
        st = self.state.setdefault(layer, {})
        if "mean" not in st:            # prompt statistics from the first chunk
            st["mean"] = x.mean(0, keepdims=True)
        m = st["mean"]
        if n == "int8":                 # centred, per token-half, 127 levels (engine int8 K)
            return q_sym(x - m, 8, 128) + m
        if n == "int8v":                # per channel per 16-token tile, 255 levels
            return q_chan_tile(x, 255)
        if n == "kv4":                  # engine kv4 basic K: centred, per token-half, +-7
            return q_sym(x - m, 4, 128) + m
        if n == "kv4v":                 # engine kv4 basic V: per channel per 16-token tile, 15 levels
            return q_chan_tile(x, 15)
        if n == "bad3":                 # must-fail anchor: 3-bit, per token over 256, no centring
            return q_sym(x, 3, 256)
        if n == "h256g32":              # centred, H256, asym int4 per 32 dims
            y = (x - m) @ H256
            return q_asym(y, 4, 32) @ H256.T + m
        if n == "h256s":                # centred, fixed H256, symmetric +-7 per token-half, range .96 (int4 K kernel storage format)
            return q_sym_clip((x - m) @ H256, 128, 0.96) @ H256.T + m
        if n.startswith("h256a"):       # centred, H256, asymmetric int4 (0..15 + zero point) per group
            # h256a: per token-half (128); in a kernel the zero point is a rank-1 correction z * sum(q half).
            # h256a96: same, with the range shrunk by .96. h256ag64: per 64 dims.
            y = (x - m) @ H256
            grp = 64 if n == "h256ag64" else 128
            yq = q_asym_clip(y, 4, grp, 0.96) if n == "h256a96" else q_asym(y, 4, grp)
            return yq @ H256.T + m
        if n == "ksink":                # KVarN Sinkhorn tiles, centred, no rotation
            return sinkhorn_tile_q(x - m) + m
        if n == "h256sink":             # H256 rotation, then KVarN Sinkhorn tiles
            return sinkhorn_tile_q((x - m) @ H256) @ H256.T + m
        if n == "h256sg32":             # centred, H256, symmetric +-7 per 32 dims, free f16 scale per group
            return q_sym((x - m) @ H256, 4, 32) @ H256.T + m
        if n.startswith("h256e"):       # centred, H256, symmetric +-7, one scale per token-half (128)
            # Per 32-dim group: scale_g = s_half / 2^e, e in 0..2^bits-1, the largest e whose range still covers the group.
            # h256e2: 2-bit e, h256e1: 1-bit e.
            nb = int(n[5:])
            y = (x - m) @ H256
            sh = y.shape
            g = y.reshape(*sh[:-1], 2, 4, 32)                      # [.., half, group, 32]
            gmax = np.abs(g).max(-1, keepdims=True)
            s_half = gmax.max(-2, keepdims=True) / 7
            s_half[s_half == 0] = 1
            e = np.floor(np.log2(np.maximum(s_half * 7, 1e-30) / np.maximum(gmax, 1e-30)))
            e = np.clip(e, 0, 2 ** nb - 1)
            sc = s_half / 2.0 ** e
            yq = np.clip(np.round(g / sc), -7, 7) * sc
            return yq.reshape(sh) @ H256.T + m
        if n == "h128":                 # centred, H128 per half, per token-half
            y = np.concatenate([(x - m)[..., :128] @ H128, (x - m)[..., 128:] @ H128], -1)
            yq = q_sym(y, 4, 128)
            return np.concatenate([yq[..., :128] @ H128.T, yq[..., 128:] @ H128.T], -1) + m
        if n.startswith("wush"):        # WUSH-KV K: calibrated per-head transform (first chunk K + Q)
            # wush: asym g128, range .96; wushg32: asym g32; wushs: symmetric +-7 per token-half, range .96 (int4 K kernel storage format).
            # KVQ_WUSH_FILE=f.npz: static transforms (T{layer}_{head}, mean{layer}) from another document, not per-prompt calibration.
            if "T" not in st and os.environ.get("KVQ_WUSH_FILE"):
                z = np.load(os.environ["KVQ_WUSH_FILE"])
                st["T"] = [(z[f"T{layer}_{h}"], np.linalg.inv(z[f"T{layer}_{h}"])) for h in range(H)]
                st["mean"] = m = z[f"mean{layer}"]
            if "T" not in st:
                if q is None:
                    raise SystemExit("wush needs Q rows: set YAH_KV_HOOK_QSTRIDE (e.g. 4)")
                qs = q[1].astype(np.float64)                         # [n, 24, 256]
                kc = (x - m).astype(np.float64)
                gamma = float(os.environ.get("KVQ_WUSH_GAMMA", "1.0"))   # 1.0 gives a lower attention error than 0.01 (layer 3)
                st["T"] = [wush_transform(kc[:, h], qs[:, 6 * h:6 * h + 6].reshape(-1, D), gamma) for h in range(H)]
            y = np.stack([(x - m)[:, h] @ st["T"][h][0].T for h in range(H)], 1)
            if n == "wushs":
                yq = q_sym_clip(y, 128, 0.96)
            else:
                yq = q_asym_clip(y, 4, 32 if n == "wushg32" else 128, 0.96)
            return np.stack([yq[:, h] @ st["T"][h][1].T for h in range(H)], 1) + m
        if n == "uq":                   # UltraQuant K: H256 + MXFP4 per 32 (no centring)
            return q_mxfp4(x @ H256) @ H256.T
        if n == "uqv":                  # UltraQuant V: MXFP4 per 32, no rotation
            return q_mxfp4(x)
        raise SystemExit(f"unknown codec {n}")


def serve(kc, vc, dump):
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    dumps = {}
    while True:
        hdr = inp.read(28)
        if len(hdr) < 28:
            break
        magic, layer, chunk, first, rows, cols, qrows = struct.unpack("<7i", hdr)
        if magic != MAGIC:
            raise SystemExit("bad hook header")
        if rows == 0:
            break
        n = rows * cols
        k = np.frombuffer(inp.read(n * 2), np.float16).reshape(rows, H, D)
        v = np.frombuffer(inp.read(n * 2), np.float16).reshape(rows, H, D)
        qd = None
        if qrows:
            raw = inp.read(qrows * (4 + 6144 * 4))
            rec = np.frombuffer(raw, np.uint8).reshape(qrows, 4 + 6144 * 4)
            qpos = rec[:, :4].copy().view(np.int32)[:, 0]
            qd = (qpos, rec[:, 4:].copy().view(np.float32).reshape(qrows, 24, 256))
        if dump:
            d = dumps.setdefault(layer, {"k": [], "v": [], "qpos": [], "q": []})
            d["k"].append(k.copy()); d["v"].append(v.copy())
            if qd is not None:
                d["qpos"].append(qd[0]); d["q"].append(qd[1].astype(np.float16))
        ko = kc(layer, k.astype(np.float32), qd).astype(np.float16)
        vo = vc(layer, v.astype(np.float32), qd).astype(np.float16)
        out.write(ko.tobytes()); out.write(vo.tobytes()); out.flush()
    if dump:
        os.makedirs(dump, exist_ok=True)
        for layer, d in dumps.items():
            np.save(f"{dump}/k_l{layer:02d}.npy", np.concatenate(d["k"]))
            np.save(f"{dump}/v_l{layer:02d}.npy", np.concatenate(d["v"]))
            if d["q"]:
                np.save(f"{dump}/qpos_l{layer:02d}.npy", np.concatenate(d["qpos"]))
                np.save(f"{dump}/q_l{layer:02d}.npy", np.concatenate(d["q"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="fp16")
    ap.add_argument("--v", default="fp16")
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    serve(Codec(a.k), Codec(a.v), a.dump)


if __name__ == "__main__":
    main()
