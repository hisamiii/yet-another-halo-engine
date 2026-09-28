#!/usr/bin/env python3
import sys, numpy as np

def decode_q2k_row(raw, offset, row, k):
    nb = k // 256; bpr = nb * 84
    raw.seek(offset + row * bpr)
    data = np.frombuffer(raw.read(bpr), dtype=np.uint8).reshape(nb, 84)
    scales = data[:, 0:16].astype(np.int64)
    qs = data[:, 16:80].astype(np.int64)
    d = data[:, 80:82].copy().view(np.float16).astype(np.float64).reshape(nb, 1)
    dmin = data[:, 82:84].copy().view(np.float16).astype(np.float64).reshape(nb, 1)
    out = np.empty((nb, 256), dtype=np.float64)
    for i in range(256):
        sub = i // 16; l = i % 16; j = (sub % 8) // 2
        sc = scales[:, sub]
        q = qs[:, (sub // 8) * 32 + (sub % 2) * 16 + l]
        out[:, i] = (d[:, 0] * (sc & 15) * ((q >> (2 * j)) & 3) - dmin[:, 0] * (sc >> 4))
    return out.reshape(-1)

def main():
    model = sys.argv[1]; offset = int(sys.argv[2]); out_path = sys.argv[3]
    tokens = int(sys.argv[4]); n = int(sys.argv[5]); k = int(sys.argv[6])
    checks = [int(x) for x in sys.argv[7:]] or list(range(0, min(n, 8)))
    raw = open(model, "rb"); out = np.fromfile(out_path, dtype=np.float32)
    wa = 0.0; wr = 0.0
    for t in range(tokens):
        i = np.arange(k)
        x = (((t * 7 + i * 13) % 23 - 11) * 0.125).astype(np.float16).astype(np.float64)
        for row in checks:
            w = decode_q2k_row(raw, offset, row, k).astype(np.float16).astype(np.float64)
            exp = float(np.dot(x, w)); got = float(out[t * n + row])
            a = abs(got - exp); wa = max(wa, a); wr = max(wr, a / max(abs(exp), 1e-6))
    print("tokens", tokens, "n", n, "k", k, "checks", len(checks) * tokens)
    print("max_abs", wa); print("max_rel", wr)
    ok = wr < 1e-3 and wa < 1e-3
    print("PASS" if ok else "FAIL"); return 0 if ok else 1

if __name__ == "__main__": sys.exit(main())