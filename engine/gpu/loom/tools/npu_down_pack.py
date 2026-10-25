#!/usr/bin/env python3
"""Pack the NPU FFN block's down weights (emit_npu_ffn.py): every layer's ffn_down over the NPU's F features, the rows
unrotated with down's awq folded in (w_orig = s1 / awq H256(s2 / 16 w), ffnblk_check.unrotate_rows), re-quantized
to IQ4_XS rows the down image decodes (gen_npu_gemm fuse 2: the NPU's images share one gate layout, LoomPrefill
RunFfnBlock).

usage: npu_down_pack.py <model.mq4-xts> <out> [F] [layers]

out: IQ4_XS [layer][5120 rows][F / 256 blocks][136 B].
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import emit_npu_ffn as NF  # noqa: E402
import ffnblk_check as FB  # noqa: E402
import hfq_index  # noqa: E402

ROWS = NF.COLS * NF.NP * 16   # rows per call
IQ = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float64)   # kvalues
MID = (IQ[1:] + IQ[:-1]) / 2


def nearest(x):
    """The index of the IQ grid value nearest x (ties up)."""
    return np.searchsorted(MID, x, side="right").astype(np.uint8)


def iq4xs(w):
    """f64 [n][K] -> IQ4_XS blocks [n][K / 256][136] u8 (llama.cpp block_iq4_xs: d, scales_h, scales_l, qs): per 32
    values the integer scale (ls - 32) d and the codes that minimize the squared error over a few scale candidates."""
    n, K = w.shape
    b = w.reshape(n, K // 256, 8, 32)
    am = np.abs(b).max(-1)
    sub = np.zeros_like(am)   # the best scale per 32 (signed: the grid is asymmetric)
    err = np.full(am.shape, np.inf)
    for f in np.concatenate([np.linspace(0.8, 1.2, 17), -np.linspace(0.8, 1.2, 17)]):
        sc = am / 127 * f
        x = b / np.where(sc == 0, 1, sc)[..., None]
        e = ((IQ[nearest(x)] * sc[..., None] - b) ** 2).sum(-1)
        better = e < err
        sub, err = np.where(better, sc, sub), np.where(better, e, err)
    d = (np.abs(sub).max(-1) / 31).astype(np.float16)
    dd = d.astype(np.float64)
    ls = np.clip(np.round(sub / np.where(dd == 0, 1, dd)[..., None]), -32, 31).astype(np.int64)
    s = (dd[..., None] * ls)[..., None]
    q = nearest(b / np.where(s == 0, 1, s))   # [n][kb][8][32]
    out = np.zeros((n, K // 256, 136), np.uint8)
    out[..., 0:2] = d[..., None].view(np.uint8)
    u = (ls + 32).astype(np.uint16)   # 6 bits per sub-block
    sh = np.zeros((n, K // 256), np.uint16)
    for j in range(8):
        sh |= ((u[..., j] >> 4) & 3) << (2 * j)
        out[..., 4 + j // 2] |= ((u[..., j] & 15) << (4 * (j % 2))).astype(np.uint8)
    out[..., 2], out[..., 3] = (sh & 255).astype(np.uint8), (sh >> 8).astype(np.uint8)
    out[..., 8:] = (q[..., :16] | (q[..., 16:] << 4)).reshape(n, K // 256, 128)
    return out


def iq4xs_values(blk):
    """IQ4_XS blocks -> f64 [n][K] (npu_dec_emit.iq4xs_fields, the decoder's oracle)."""
    import npu_dec_emit
    d, ls, q = npu_dec_emit.iq4xs_fields(blk)
    return (d[..., None, None] * (ls - 32)[..., None] * IQ[q].reshape(q.shape[:-1] + (8, 32))).reshape(blk.shape[0], -1)


def layer_rows(model, idx, layer, F):
    """The layer's IQ4_XS rows over the NPU's features, and the relative RMS error of the re-quantization."""
    pre = FB.PRE % layer
    G = NF.FFN - F
    awq = np.asarray(hfq_index.raw(model, idx[pre + "down_proj.awq_scale.weight"]).view(np.float16), np.float64)
    rows = hfq_index.mq4_rows(model, idx[pre + "down_proj.weight"])[:, G // 256:]
    out, num, den = [], 0.0, 0.0
    for c in range(0, NF.HID, ROWS):
        wo = FB.unrotate_rows(FB.mq4_values(np.ascontiguousarray(rows[c:c + ROWS])), awq[G:])
        blk = iq4xs(wo)
        num, den = num + ((iq4xs_values(blk) - wo) ** 2).sum(), den + (wo ** 2).sum()
        out.append(blk)
    return np.concatenate(out), np.sqrt(num / den)


def main():
    model, path = sys.argv[1], sys.argv[2]
    F = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    layers = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    idx = {e[0]: e for e in hfq_index.read(model)}
    t0 = time.time()
    with open(path + ".part", "wb") as fh:
        for layer in range(layers):
            blk, rel = layer_rows(model, idx, layer, F)
            fh.write(blk.tobytes())
            print(f"layer {layer}: {time.time() - t0:.0f} s, re-quantization rel rms {rel:.4f}", flush=True)
    os.replace(path + ".part", path)


if __name__ == "__main__":
    main()
