#!/usr/bin/env python3
"""Fixtures for yah_swiglu_decode_f32.loom.

Pads each weight fixture to the swiglu band max (16 rows * 4200 = 67200 bytes)
and saves silu(dot_gate)*dot_up. The GEMV-backed formats reuse their GEMV
expected row dots; IQ2_XS/IQ2_S have no GEMV port, so their d-row dots are
decoded here from the block layout and dotted against the shared sparse x.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.dirname(HERE)
ROWS = 16
K = 5120
MAXB = 67200
GEMVF = ["q4k", "iq4xs", "iq4nl", "q5k", "q6k", "q3k", "iq3s", "iq3xxs"]
IQ2F = ["iq2xs", "iq2s"]


def decode_iq2xs(block, grid, ksigns):
    d = np.frombuffer(block[0:2].tobytes(), dtype=np.float16)[0].astype(np.float64)
    qs = block[2:66]
    sc = block[66:74]
    out = []
    for i in range(256):
        sub16 = i // 16
        ib32 = (sub16 // 2) % 8
        half = sub16 % 2
        li = (i % 16) // 8
        j = i % 8
        l = 2 * half + li
        code = int(qs[2 * (4 * ib32 + l)]) | (int(qs[2 * (4 * ib32 + l) + 1]) << 8)
        g_word = int(grid[code & 511])
        g_byte = (g_word >> (8 * j)) & 0xFF
        signs = int(ksigns[code >> 9])
        sign_bit = (signs >> j) & 1
        mag = -g_byte if sign_bit else g_byte
        nib = (int(sc[ib32]) & 0xF) if half == 0 else (int(sc[ib32]) >> 4)
        out.append(d * (0.5 + nib) * 0.25 * mag)
    return out


def decode_iq2s(block, grid):
    d = np.frombuffer(block[0:2].tobytes(), dtype=np.float16)[0].astype(np.float64)
    qs = block[2:66]
    qh = block[66:74]
    sc = block[74:82]
    out = []
    for i in range(256):
        sub16 = i // 16
        ib32 = (sub16 // 2) % 8
        half = sub16 % 2
        li = (i % 16) // 8
        j = i % 8
        l = 2 * half + li
        low = int(qs[4 * ib32 + l])
        high = (int(qh[ib32]) << (8 - 2 * l)) & 0x300
        g_word = int(grid[low | high])
        g_byte = (g_word >> (8 * j)) & 0xFF
        signs = int(qs[32 + 4 * ib32 + l])
        sign_bit = (signs >> j) & 1
        mag = -g_byte if sign_bit else g_byte
        nib = (int(sc[ib32]) & 0xF) if half == 0 else (int(sc[ib32]) >> 4)
        out.append(d * (0.5 + nib) * 0.25 * mag)
    return out


def main():
    x = np.load(os.path.join(FIX, "q4k_gemm", "gemv_x.npy")).astype(np.float32)
    np.save(os.path.join(HERE, "x.npy"), x)
    dots = {}
    for fmt in GEMVF:
        src = np.load(os.path.join(FIX, fmt + "_gemm", "input_" + fmt + "_gemm.npy")).astype(np.uint8)
        out = np.zeros(MAXB, dtype=np.uint8)
        out[:src.size] = src
        np.save(os.path.join(HERE, fmt + "_pad.npy"), out.view(np.int8))
        dots[fmt] = np.load(os.path.join(FIX, fmt + "_gemm", "gemv_expected.npy")).astype(np.float64)
        print(fmt, src.size)
    for fmt in IQ2F:
        src = np.load(os.path.join(FIX, fmt + "_gemm", "input_" + fmt + "_gemm.npy")).astype(np.uint8)
        out = np.zeros(MAXB, dtype=np.uint8)
        out[:src.size] = src
        np.save(os.path.join(HERE, fmt + "_pad.npy"), out.view(np.int8))
        nblk = src.size // ROWS
        bpb = src.size // ROWS // 20
        grid = np.load(os.path.join(FIX, fmt + "_gemm", "grid.npy"))
        ksigns = np.load(os.path.join(FIX, fmt + "_gemm", "ksigns.npy"))
        dot = np.zeros(ROWS, dtype=np.float64)
        for r in range(ROWS):
            dec = []
            for b in range(20):
                off = (r * 20 + b) * bpb
                blk = src[off:off + bpb]
                if fmt == "iq2xs":
                    dec += decode_iq2xs(blk, grid, ksigns)
                else:
                    dec += decode_iq2s(blk, grid)
            dec = np.array(dec, dtype=np.float64)
            dot[r] = np.dot(dec, x.astype(np.float64))
        dots[fmt] = dot
        print(fmt, src.size, dot[:3])
    allf = GEMVF + IQ2F
    for g in allf:
        for u in allf:
            dg = dots[g]
            du = dots[u]
            expected = (dg / (1.0 + np.exp(-dg)) * du).astype(np.float32)
            np.save(os.path.join(HERE, "%s_%s_expected.npy" % (g, u)), expected)
    print("done")


if __name__ == "__main__":
    main()