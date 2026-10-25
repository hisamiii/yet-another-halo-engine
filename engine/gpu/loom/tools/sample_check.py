#!/usr/bin/env python3
"""The decode sampler (gen_decode_misc.gen_sample) against a numpy model of it, the model against the exact
distribution (softmax(logits / T) cut by top-k, top-p and min-p), and the penalty kernel (gen_penalty) against numpy.

usage: sample_check.py <model> <workdir> [draws]      (<model>: any file; no case binds a tensor, hal_run skips it)

The model repeats the kernel's steps (top-k and nucleus thresholds by binary search, the min-p bound, the counter
hash, Gumbel-max) in float64, so the GPU must return its token for every (T, top_p, top_k, min_p, seed, pos) case (the
kernel's exp / log are approximate, so a near-tie could differ; none should with these logits). Over draws seeds the
model's token frequencies must match the exact cut distribution (total variation distance below a sampling-noise
bound). The penalties must match numpy bit for bit.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
EMIT = os.path.join(HERE, "..", "emit_hal.py")
HALRUN = os.path.join(ROOT, "engine", "build", "hal_run")
GPURUN = os.path.join(ROOT, "engine", "run", "gpu_run.sh")
import gen_decode_misc as DM  # noqa: E402

V = 248320
M32 = 0xFFFFFFFF


def fmix(x):
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & M32
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & M32
    return x ^ (x >> 16)


def fmix_np(x):
    x = x ^ (x >> np.uint64(16))
    x = (x * np.uint64(0x85EBCA6B)) & np.uint64(M32)
    x = x ^ (x >> np.uint64(13))
    x = (x * np.uint64(0xC2B2AE35)) & np.uint64(M32)
    return x ^ (x >> np.uint64(16))


def threshold(logits, t, top_p, top_k=0, min_p=0.0):
    """The kernel's final threshold: its float32 mid points, sums in float64."""
    mx = np.float32(logits.max())
    cut = np.float32(mx - np.float32(20.723265836946411) * np.float32(t))
    e = np.exp((logits.astype(np.float64) - mx) / t)

    def mass(lam):
        return e[logits >= lam].sum()

    def search(lo, ok):
        hi = mx
        for _ in range(DM.SAMPLE_ITERS):
            mid = np.float32(np.float32(lo + hi) * np.float32(0.5))
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return lo
    k = top_k if top_k else V + 1
    klo = search(cut, lambda lam: (logits >= lam).sum() >= k)
    target = mass(klo) * top_p
    lo = search(klo, lambda lam: mass(lam) >= target)
    if min_p > 0:
        lo = max(lo, np.float32(mx + np.float32(np.float32(np.log(np.float32(min_p))) * np.float32(t))))
    return lo


def draw(logits, lam, t, seed, pos):
    """The kernel's token for this threshold: Gumbel-max over the kept tokens, the lowest index on ties."""
    kept = np.nonzero(logits >= lam)[0]
    rng = fmix(fmix(((pos * 0x9E3779B1) & M32) ^ (seed >> 32)) ^ (seed & M32))
    h = fmix_np(((kept.astype(np.uint64) * np.uint64(0x9E3779B1)) & np.uint64(M32)) ^ np.uint64(rng))
    u = ((h >> np.uint64(8)).astype(np.float64) + 0.5) * 2.0 ** -24
    score = (logits[kept].astype(np.float64) - logits.max()) / t - np.log(-np.log(u))
    return int(kept[np.argmax(score)])


def nucleus(logits, t, top_p, top_k=0, min_p=0.0):
    """The exact distribution: softmax(logits / T), its top_k tokens, their top_p nucleus, then p >= min_p p_max
    (sorted, as vLLM and transformers apply them)."""
    p = np.exp((logits.astype(np.float64) - logits.max()) / t)
    order = np.argsort(-p, kind="stable")
    if top_k:
        order = order[:top_k]
    c = np.cumsum(p[order])
    k = int(np.searchsorted(c, top_p * c[-1])) + 1
    keep = order[:k]
    if min_p > 0:
        keep = keep[p[keep] >= min_p * p[keep].max()]
    q = np.zeros(V)
    q[keep] = p[keep] / p[keep].sum()
    return q


def penalize(logits, toks, pos, presence, frequency, repetition, last_n):
    """llama.cpp's penalties over toks[pos + 1 - last_n .. pos], in the kernel's float32 order."""
    out = logits.copy()
    window = toks[max(0, pos + 1 - min(last_n, DM.PENALTY_WINDOW)):pos + 1]
    for tok in dict.fromkeys(window.tolist()):
        c = np.float32((window == tok).sum())
        lg = out[tok]
        lr = np.float32(lg / np.float32(repetition)) if lg > 0 else np.float32(lg * np.float32(repetition))
        out[tok] = np.float32(lr - np.float32(np.float32(presence) + np.float32(np.float32(frequency) * c)))
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    model_path, work = sys.argv[1], sys.argv[2]
    draws = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    os.makedirs(work, exist_ok=True)
    src = os.path.join(work, "sample.loom")
    open(src, "w").write(DM.gen_sample(V))
    r = subprocess.run([sys.executable, EMIT, src, os.path.join(work, "sample"), "nop=0"], capture_output=True,
                       text=True)
    if r.returncode:
        sys.exit(f"emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    hal = r.stdout.strip().splitlines()[0]
    # logits shaped like a model's (a long tail and a handful of strong candidates), and flat ones (a nucleus of tens of
    # thousands, the threshold among many near-equal logits)
    g = np.random.default_rng(1)
    peaked = g.normal(-4.0, 2.0, V).astype(np.float32)
    peaked[g.choice(V, 12, replace=False)] = np.float32(12.0) + g.normal(0, 1.5, 12).astype(np.float32)
    flat = g.normal(0.0, 1.0, V).astype(np.float32)
    lf = os.path.join(work, "logits.f32")

    ok = n = 0
    for logits, t, top_p, top_k, min_p in (
            (peaked, 0.6, 0.95, 0, 0.0), (peaked, 1.0, 1.0, 0, 0.0), (peaked, 1.3, 0.5, 0, 0.0),
            (flat, 1.0, 0.9, 0, 0.0), (flat, 0.6, 0.5, 0, 0.0), (peaked, 0.6, 0.95, 20, 0.0),
            (peaked, 1.0, 1.0, 5, 0.0), (flat, 1.0, 0.95, 20, 0.0), (flat, 1.0, 1.0, 1000, 0.0),
            (flat, 0.7, 1.0, 0, 0.05), (peaked, 1.0, 0.95, 20, 0.1),
            (flat, 1.0, 0.9, 20000, 0.0), (flat, 1.0, 1.0, 9000, 0.02)):
        logits.tofile(lf)
        lam = threshold(logits, t, top_p, top_k, min_p)
        print(f"  T {t} top_p {top_p} top_k {top_k} min_p {min_p}: {int((logits >= lam).sum())} tokens kept")
        for seed, pos in ((1, 0), (0x123456789ABCDEF, 17), (42, 4095), (7, 70000), (2**64 - 1, 3)):
            pf, sf, of = (os.path.join(work, x) for x in ("params.bin", "pos.bin", "out.bin"))
            np.array([t, top_p], np.float32).tofile(pf)
            with open(pf, "ab") as f:
                np.array([seed & M32, seed >> 32, top_k], np.uint32).tofile(f)
                np.array([min_p, 0, 0], np.float32).tofile(f)
            np.array([pos], np.int32).tofile(sf)
            cmd = [GPURUN, "sample", "--", HALRUN, model_path, hal, "1", "1024", f"{V * 4},32,4,4", f"f:{lf}",
                   f"f:{pf}", f"f:{sf}", f"o:4:{of}"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if "hal_run: ok" not in r.stdout:
                sys.exit(f"hal_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
            got = int(np.fromfile(of, np.int32)[0])
            want = draw(logits, lam, t, seed, pos)
            n += 1
            ok += got == want
            if got != want:
                print(f"  T {t} top_p {top_p} seed {seed:#x} pos {pos}: GPU {got} (logit {logits[got]:.4f}), "
                      f"model {want} (logit {logits[want]:.4f}), threshold {lam:.6f}")
    print(f"sample: GPU = model on {ok} / {n} cases")

    worst = 0.0
    logits = peaked
    for t, top_p, top_k, min_p in ((0.6, 0.95, 0, 0.0), (1.3, 0.5, 0, 0.0), (1.0, 0.95, 5, 0.0), (0.6, 1.0, 0, 0.1),
                                   (1.3, 0.9, 8, 0.05)):
        q = nucleus(logits, t, top_p, top_k, min_p)
        lam = threshold(logits, t, top_p, top_k, min_p)
        counts = np.zeros(V)
        for s in range(draws):
            counts[draw(logits, lam, t, 0xABCDEF00 + s, s % 4096)] += 1
        tv = 0.5 * np.abs(counts / draws - q).sum()
        bound = 1.5 * np.sqrt((q > 0).sum() / draws)   # ~E[TV] of draws samples is <= sqrt(k / draws) / 2
        print(f"sample model: T {t} top_p {top_p} top_k {top_k} min_p {min_p}: {int((q > 0).sum())} tokens kept, "
              f"total variation {tv:.4f} over {draws} draws (bound {bound:.4f})")
        worst = max(worst, tv / bound)

    # penalties: a window with repeats, at the start of the stream (shorter than last_n) and deep in it
    T = 4096
    src = os.path.join(work, "penalty.loom")
    open(src, "w").write(DM.gen_penalty(V, T))
    r = subprocess.run([sys.executable, EMIT, src, os.path.join(work, "penalty"), "nop=0"], capture_output=True,
                       text=True)
    if r.returncode:
        sys.exit(f"emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    phal = r.stdout.strip().splitlines()[0]
    toks = g.choice(np.array([5, 77, 1000, 248319, 3, 3, 9]), T + 1).astype(np.int32)
    toks[2000:2600] = g.integers(0, V, 600)
    pok = pn = 0
    for pos, presence, frequency, repetition, last_n in ((10, 1.5, 0.0, 1.0, 64), (3000, 0.5, 0.3, 1.1, 1024),
                                                         (2500, 0.0, 0.0, 1.3, 256), (4095, -0.5, -0.2, 0.9, 64)):
        tf, pf, sf, of = (os.path.join(work, x) for x in ("toks.bin", "pparams.bin", "ppos.bin", "plogits.bin"))
        toks.tofile(tf)
        np.array([presence, frequency, repetition], np.float32).tofile(pf)
        with open(pf, "ab") as f:
            np.array([last_n], np.int32).tofile(f)
        np.array([pos], np.int32).tofile(sf)
        flat.tofile(lf)
        cmd = [GPURUN, "penalty", "--", HALRUN, model_path, phal, "1", str(DM.PENALTY_WINDOW),
               f"{V * 4},{(T + 1) * 4},16,4", f"io:{lf}:{of}", f"f:{tf}", f"f:{pf}", f"f:{sf}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if "hal_run: ok" not in r.stdout:
            sys.exit(f"hal_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
        got = np.fromfile(of, np.float32)
        want = penalize(flat, toks, pos, presence, frequency, repetition, last_n)
        changed = int((want != flat).sum())
        bad = int((got != want).sum())
        pn += 1
        pok += bad == 0
        print(f"  penalty pos {pos} last_n {last_n}: {changed} logits changed, {bad} differ from numpy")
    print(f"penalty: GPU = numpy on {pok} / {pn} cases")
    if ok != n or worst > 1.0 or pok != pn:
        sys.exit("sample: FAIL")
    print("sample: ok")


if __name__ == "__main__":
    main()
