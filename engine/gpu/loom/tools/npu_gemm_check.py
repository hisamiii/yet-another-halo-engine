#!/usr/bin/env python3
"""Check gen_npu_gemm.py on the NPU against a float64 oracle of the bfp16ebs8 math.

usage: npu_gemm_check.py <workdir> <cols> <m_blocks> <ks,ks,ks,ks> <passes> [act.f16 wgt.f16]

Operands are random (wide per-row dynamic range) or f16 files: activations [64 * m_blocks][K], weights [TN * cols][K] (gen_npu_gemm.TN).
They are packed with the encoder's numpy oracle (bfp16_check.reference), the image runs through the engine's npu_run, and each C
partial (one per replay group of passes, gen_npu_gemm.groups) is compared with the exact product of its K range of the
bfp16 values; C is bf16: within half an ulp of the exact product (f32 accumulation order is ~1e-7 below that).
Time the NPU with the counter profiler, not wall clock.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import bfp16_check as B  # noqa: E402
import gen_npu_gemm as N  # noqa: E402
import hrx_paths  # noqa: E402


def unpack_c(raw, cols, nb, groups=1, group=None):
    """C binding [col][group][M block][slab mp][slab np][chain][8][8] (as f32) -> [TM * nb][TN * cols]: the sum of the
    groups' partials in order (as the GPU unpack adds them), or partial group only."""
    c = raw.reshape(cols, groups, nb, N.MP, N.NP, 2, 2, 8, 8)
    if group is not None:
        c = c[:, group]
    else:
        acc = c[:, 0]
        for g in range(1, groups):
            acc = acc + c[:, g]
        c = acc
    return c.transpose(1, 2, 4, 6, 0, 3, 5, 7).reshape(N.TM * nb, N.TN * cols)


def value(x):
    E, m = B.quantize(x)
    rows, k = m.shape
    return (m.reshape(rows, k // 8, 8).astype(np.float64) * np.exp2(E.astype(np.float64) - 133)[..., None]).reshape(rows, k)


def run(cmd, env, tag):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if r.returncode:
        sys.exit(f"{tag} failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return r


def main():
    if len(sys.argv) < 6:
        sys.exit(__doc__)
    work, cols, nb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    ks, passes = tuple(int(v) for v in sys.argv[4].split(",")), int(sys.argv[5])
    # NPU_MU=1: one M slab per iteration (2-slot activation ring); NPU_FUSE=2: the GEMM cores decode their panels
    mu, fuse = int(os.environ.get("NPU_MU", "2")), int(os.environ.get("NPU_FUSE", "0"))
    if os.environ.get("NPU_GROUPS"):   # replay groups of passes, e.g. "5" (one group)
        N.GE.GROUPS = tuple(int(v) for v in os.environ["NPU_GROUPS"].split(","))
    # fuse 2, NPU_KRAW=<K> NPU_K0=<k>: rows of K columns, the call's K at column k (a site's chunk)
    kraw, k0r = int(os.environ.get("NPU_KRAW", "0")), int(os.environ.get("NPU_K0", "0"))
    fmt = os.environ.get("NPU_FMT", "IQ4_XS")   # fuse 2: the raw rows' format (gen_npu_gemm FUSE_FMTS)
    assert fuse in (0, 2), "NPU_FUSE: 0 or 2"
    # NPU_GATE=1 (diag): the gated image (the engine's bindings) with its flag record already open for one call
    cfg = N.Config(cols, nb, ks, passes, mu=mu, fuse=fuse, kraw=kraw, fmt=fmt, rawdma=int(fuse == 2),
                   gate=N.GATE_SUPPLY if os.environ.get("NPU_GATE") else 0)
    M, Nn, K = N.TM * nb, N.TN * cols, 8 * passes * sum(ks)
    os.makedirs(work, exist_ok=True)
    wq = None
    if fuse == 2:   # real rows of fmt (NPU_GGUF): the panels are decoded on the cores; W's value is the decoder's
        sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
        import gguf
        D = N.decoder(fmt, cfg.gord)
        rd = gguf.GGUFReader(os.path.expanduser(os.environ.get(
            "NPU_GGUF", "~/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf")))
        KR = kraw or K
        t = next(x for x in rd.tensors if x.tensor_type.name == fmt and len(x.shape) == 2
                 and int(x.shape[0]) == KR and int(x.shape[1]) >= Nn)
        wraw = np.asarray(t.data).view(np.uint8).reshape(-1, KR // 256, D.BLK)[:Nn].copy()
        wf = D.weights(np.ascontiguousarray(wraw[:, k0r // 256:(k0r + K) // 256])).reshape(Nn, K)   # exact f32
        frag = wf.reshape(Nn // 8, 8, K // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
        bb = D.Q.bfp_hw(frag).reshape(-1, 8, 9)                    # per row [E][8 mantissas]
        mq = bb[:, :, 1:].copy().view(np.int8).astype(np.float64)
        wq = (mq * np.exp2(bb[:, :, 0].astype(np.float64) - 133)[..., None]).reshape(
            Nn // 8, K // 8, 8, 8).transpose(0, 2, 1, 3).reshape(Nn, K)
    if len(sys.argv) > 7:
        a = np.fromfile(sys.argv[6], np.float16)[:M * K].reshape(M, K)
        w = np.fromfile(sys.argv[7], np.float16)[:Nn * K].reshape(Nn, K)
    else:
        rng = np.random.default_rng(11)
        a = (rng.standard_normal((M, K)) * np.exp(rng.uniform(-3, 3, (M, 1)))).astype(np.float16)
        w = (rng.standard_normal((Nn, K)) * 0.05).astype(np.float16)
        wd_ = int(os.environ.get("GN_QWDUP") or 1)   # (Q' P1 diag) columns d j + i read column d j's W (one multicast)
        for c_ in range(Nn // N.TN):
            if c_ % wd_:
                w[N.TN * c_:N.TN * (c_ + 1)] = w[N.TN * (c_ - c_ % wd_):N.TN * (c_ - c_ % wd_ + 1)]
    if os.environ.get("NPU_AONEHOT"):   # (diag) token t reads only k0 + t: C[t, n] = W[n, k0 + t] (the panel, readable)
        ko = int(os.environ["NPU_AONEHOT"])
        a = np.zeros((M, K), np.float16)
        tt = np.arange(min(M, K - ko))
        a[tt, ko + tt] = 1
    src = os.path.join(work, "npu_gemm.loom")
    img = os.path.join(work, "npu_gemm.xdna")
    if QMODE:   # layout Q' (gen_npu_q, stage 1): one K slice, the tail leaf as a Q' core (GN_QCORE = boundary json)
        import gen_npu_q as Q
        assert len(ks) == 1 and os.environ.get("GN_QCORE"), "NPU_Q: ks of one entry and GN_QCORE"
    text_ = (Q.gen if QMODE else N.gen)(cfg)
    if os.environ.get("GN_QTL") == "1" and N.QCYC:   # (diag, Q' leaves) block timeline: private words 8 / 12 / 16 =
        out_, stage_ = [], {}                           # block start / after the start W / after the start A acquire,
        for ln_ in text_.splitlines():                  # copied into words 1-3 of each row's stamped segment
            if ln_.startswith("low.func.def"):
                stage_ = {}
            out_.append(ln_)
            hit = (8 if ln_ == "  st %tsn, %tca, 0" else 12 if ln_ == "  acq %am1, 1" else
                   16 if ln_ == "  acq %am1, 0" else None)
            if hit is not None and hit not in stage_:
                stage_[hit] = 1
                out_ += [f"  %tlw{hit} = mov.cycles", f"  %tll{hit} = slice %tlw{hit}[0] : reg<aie2p.er x2> -> reg<aie2p.er>",
                         f"  st %tll{hit}, %tca, {hit}"]
            if ln_ == "  st %ptp, %ptq, 0":
                for w_, o_ in ((1, 8), (2, 12), (3, 16)):
                    out_ += [f"  %tlr{o_} = lda %pta, {o_}", f"  %tld{o_} = sub %tlr{o_}, %pts",
                             f"  st %tld{o_}, %ptq, {4 * w_}"]
        text_ = "\n".join(out_) + "\n"
    open(src, "w").write(text_)
    env = hrx_paths.env()
    env.update(Q.loom_env(cfg) if QMODE else N.loom_env(cfg))
    for k in os.environ.get("NPU_ENV_DROP", "").split(","):   # e.g. LOOM_EXP_PANEL_STREAM (diagnosis)
        env.pop(k, None)
    for kv in os.environ.get("NPU_ENV_ADD", "").split(","):    # e.g. LOOM_EXP_ROUTE_RESET=1 (diagnosis)
        if "=" in kv:
            env[kv.split("=")[0]] = kv.split("=", 1)[1]
    run([hrx_paths.LOOM_COMPILE, src, f"--root=@{cfg.entry}", "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11",
         f"--output={img}"] + ([f"--compile-report={os.environ.get('NPU_REPORT_KIND', 'text-details')}",
                                f"--compile-report-output={os.environ['NPU_REPORT']}"]
                               if os.environ.get("NPU_REPORT") else []), env, "loom-compile")   # (diag) Loom's report
    if os.environ.get("LOOM_DIAG_PC"):   # (diag) the PC block table beside the image (night/sim/dumploop.py)
        r_ = subprocess.run([hrx_paths.LOOM_COMPILE, src, f"--root=@{cfg.entry}",
                             "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11", f"--output={img}"],
                            env=dict(env, LOOM_DIAG_PC="1"), capture_output=True, text=True)
        open(os.path.join(work, "npu_gemm.pcdiag"), "w").write(
            "".join(ln + "\n" for ln in r_.stderr.splitlines() if ln.split()[:1] in (["PCBLOCKS"], ["PCTEXT"], ["PCACQ"])))
        for ln in r_.stderr.splitlines():   # (diag, HRX pin: LOOM_EXP_TRACE_SHIM in NPU_ENV_ADD) the shim DMA choices
            if ln.startswith(("SHIM ", "TWIN ", "STAGE ")):
                print(ln)
    if os.environ.get("NPU_COMPILE_ONLY"):   # (diag) stop after the image (compiler work, no NPU)
        sys.exit(0)
    a_b, w_b, c_b = (os.path.join(work, n) for n in ("a.bin", "w.bin", "c.bin"))
    sa, sw, sc = N.stream_bytes(cfg)
    a_in = a
    if fuse == 2:   # the decoders emit each pass's k-blocks in panel order (gen_npu_dec.KPERM): A follows
        kperm = np.concatenate([128 * p + D.KPERM for p in range(K // 1024 if QMODE else passes)])
        a_in = a.reshape(M, K // 8, 8)[:, kperm].reshape(M, K)
    ra, _ = B.reference(a_in, "act", M, list(ks), passes, N.TM, True)
    rw, _ = B.reference(w, "wgt", Nn, list(ks), passes, N.TN, True)
    assert ra.size == sa and rw.size == sw
    mrel = os.environ.get("NPU_QCREL") == "2" and os.environ.get("GN_QEMERGE") == "1"
    if QMODE and mrel:   # (pair relay + merge: A and C in token order, gen_npu_q.mrel)
        pass
    elif QMODE and os.environ.get("NPU_QCREL") == "2":   # (pair relay) row 2 p + j reads its blocks p nb / 2 + 2 f + j in order
        R_ = int(os.environ.get("NPU_QROWS", "2"))
        nbr_ = nb // R_
        src = [(q // 2) * 2 * nbr_ + 2 * f + q % 2 for q in range(R_) for f in range(nbr_)]
        ra = np.ascontiguousarray(ra.view(np.uint8).reshape(nb, -1)[src]).reshape(-1).view(ra.dtype)
    elif QMODE and os.environ.get("GN_QEMERGE") == "1":   # (egress merge) row r reads blocks b R + r: the merged drain
        R_ = int(os.environ.get("NPU_QROWS", "2"))         # writes [b][r] = M block order
        nbr_ = nb // R_
        src = [b * R_ + r for r in range(R_) for b in range(nbr_)]
        ra = np.ascontiguousarray(ra.view(np.uint8).reshape(nb, -1)[src]).reshape(-1).view(ra.dtype)
    if os.environ.get("GN_QASPLIT_COPY") == "1":   # (diag) a second copy of A for the split's second streams
        ra = np.concatenate([ra, ra])
    ra.tofile(a_b)
    if QMODE and N.qp1_on():   # (GN_QP1) [col][pass][record] -> [col][half][pass][half record]: each half view contiguous
        il_ = os.environ.get("GN_QP1IL")   # (P1K interleaved panel, LOOM_EXP_PANEL_INTERLEAVE: 5 = [pass][group][K half]
        if N.qp1k_on() and il_ == "5":    # the record order as is; 1 = [pass][K half][group], the fill decoder's)
            pass
        elif N.qp1k_on() and il_ == "1":
            wb_ = rw.view(np.uint8).reshape(cols, passes, N.NP, 2, -1)
            rw = np.ascontiguousarray(wb_.transpose(0, 1, 3, 2, 4)).reshape(-1)
        elif N.qp1k_on():   # (P1K) [col][pass][group][K half][half slab] -> [col][K half][pass][group][half slab]
            wb_ = rw.view(np.uint8).reshape(cols, passes, N.NP, 2, -1)
            rw = np.ascontiguousarray(wb_.transpose(0, 3, 1, 2, 4)).reshape(-1)
        else:
            wb_ = rw.view(np.uint8).reshape(cols, passes, 2, -1)
            rw = np.ascontiguousarray(wb_.transpose(0, 2, 1, 3)).reshape(-1)
    rw.tofile(w_b)
    np.zeros(sc // 4, np.float32).tofile(c_b)
    co = os.path.join(work, "c_out.bin")
    bindings = [a_b, w_b, c_b]
    if cfg.gate:   # flag (ready = the first job's GATE_CALLS + 1 call), tick, signal (go, done)
        g_b = [os.path.join(work, n) for n in ("flag.bin", "tick.bin", "sig.bin")]
        np.array([N.GATE_CALLS + max(1, int(os.environ.get('NPU_RUN_GATED') or 1)), 0, 0, 0], np.uint32).tofile(g_b[0])
        np.zeros(cfg.gate * 4, np.uint32).tofile(g_b[1])
        np.zeros(8, np.uint32).tofile(g_b[2])
        bindings += g_b
    if fuse:   # the egress lanes' dummy binding (never transferred), the raw stream binding
        d_b, r_b = os.path.join(work, "d.bin"), os.path.join(work, "r.bin")
        if QMODE:   # (gen_npu_q: one filler a core, the decode side's records)
            words = np.prod(N.fuse_record(N.QF["cfg"], (0,)))
        else:
            words = max(fw * nrec for c in range(cols) for fw, nrec in (N.fuse_record(cfg, cov)
                                                                      for cov in N.fuse_cover(cfg, c).values()))
        np.zeros(len(ks) * cols * words, np.int32).tofile(d_b)
        # the GGUF rows as they are, from the call's first column
        np.concatenate([wraw.reshape(-1)[k0r // 256 * D.BLK:], np.zeros(256, np.uint8)]).tofile(r_b)
        bindings += [d_b, r_b]
    width = None
    for cand in range(cols, 9):   # memory-tile stages can need a wider context than the compute columns
        if os.environ.get("NPU_STALL_CMD"):   # (diag) a command run 3 s into the job, while it lives (a status dump)
            import time
            pr = subprocess.Popen([hrx_paths.NPU_RUN, img, cfg.entry, str(cand), "2=" + co + os.environ.get("NPU_OUTX", "")] + bindings,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            time.sleep(float(os.environ.get("NPU_STALL_S", "3")))
            if pr.poll() is None:
                subprocess.run(os.environ["NPU_STALL_CMD"], shell=True)
            o, er = pr.communicate(timeout=600)
            r = subprocess.CompletedProcess(pr.args, pr.returncode, o, er)
        else:
            r = subprocess.run([hrx_paths.NPU_RUN, img, cfg.entry, str(cand), "2=" + co + os.environ.get("NPU_OUTX", "")] + bindings,
                               capture_output=True, text=True, env=env, timeout=600)
        if r.returncode == 0:
            width = cand
            if os.environ.get("NPU_RUN_REPEAT") or os.environ.get("YAH_NPU_PROF"):   # (npu_run's times, counters)
                print("".join(ln + "\n" for ln in r.stdout.splitlines() if " run " in ln or " prof " in ln), end="")
            print("".join(ln + "\n" for ln in r.stderr.splitlines() if "RunGatedOnce" in ln), end="")
            break
        if "does not match the admitted native context" not in r.stderr:
            sys.exit(f"npu_run failed\n{r.stderr[-2000:]}")
    if width is None:
        sys.exit("no context width admitted the image")
    if N.QCYC:   # (diag) each segment's first word: core cycles since the first firing; per M block: its last segment's
        seg = N.NP * 256 * 2   # bytes of one bf16 segment (NP sub-tiles of 16 x 16)
        cyc = np.fromfile(co, np.uint32)[::seg // 4].astype(np.int64)
        if os.environ.get("GN_QTL") == "1":   # (diag) words 0-3 of every segment: [col][block][row of MP][end, start, W, A]
            w4_ = np.fromfile(co, np.uint32).reshape(-1, seg // 4)[:, :4].astype(np.int64)
            np.save(os.path.join(work, "tl.npy"), w4_.reshape(cols, nb, N.MP, 4))
        R_ = int(os.environ.get("NPU_QROWS", "1"))
        rel_ = R_ >= 2 and (os.environ.get("NPU_QCREL", "1") == "1" or (os.environ.get("GN_QEMERGE") == "1"
                                                                      and os.environ.get("NPU_QCREL") != "2"))   # (C relay
        # or egress merge: row r's blocks r, r + R, ..)
        last = cyc.reshape(cols, nb, N.MP)[0, ::R_, -1] if rel_ else cyc.reshape(cols, nb, N.MP)[0, :nb // R_, -1]
        if os.environ.get("NPU_QCREL") == "2":   # (P1 pair relay: core 0's blocks 0, 2, .. of pair 0's half; merged: 0, 4, ..)
            last = cyc.reshape(cols, nb, N.MP)[0, ::4, -1] if mrel else cyc.reshape(cols, nb, N.MP)[0, :nb // (R_ // 2):2, -1]
        if rel_:
            for r_ in range(R_):
                st_ = np.diff(cyc.reshape(cols, nb, N.MP)[0, r_::R_, -1])
                print(f"  row {r_} (core 0): median {float(np.median(st_)):.0f} min {st_.min()} max {st_.max()}")
        if os.environ.get("GN_QFCYC") == "1" and fuse == 2:   # (the filler's fill words 1-3 of its blocks' segments)
            w4f = np.fromfile(co, np.uint32).reshape(cols, nb, N.MP, seg // 4)[0, :, :, :4].astype(np.int64)
            fb_ = ((1 if N.qcstage_on() else 2) if mrel else 1) if N.qp1k_on() and os.environ.get("NPU_QCREL") == "2" else 0
            v_ = w4f[fb_, -1]
            print(f"  fill (core 0 column, filling row's block {fb_}): egress {v_[1]} whole fill {v_[2]} raw waits {v_[3]}")
        step = np.diff(last)
        mma = passes * N.MP * N.NP * sum(ks) * 4   # MMAs a core issues per M block
        med = float(np.median(step))
        print(f"cycles per M block (core 0, {nb} blocks): median {med:.0f} min {step.min()} max {step.max()}; "
              f"{mma} MMAs -> MMA share {mma / med:.1%}; first block done at {last[0]}")
        sys.exit(0)
    raw = (np.fromfile(co, np.uint16).astype(np.uint32) << 16).view(np.float32)   # bf16 C
    if os.environ.get("NPU_AONEHOT"):   # (diag) per token (= k): how many features match the panel
        ko = int(os.environ["NPU_AONEHOT"])
        got = unpack_c(raw, cols, nb, 1)
        T = min(M, K - ko)
        want = (wq if wq is not None else value(w))[:, ko:ko + T].T   # [t][n]
        g = got[:T]
        ok = np.isclose(g, want, rtol=1e-2, atol=1e-30)
        print(f"onehot k0 {ko}: tokens {T}, match {ok.mean():.3f}, nan {np.isnan(g).mean():.3f}, zero {(g == 0).mean():.3f}")
        rowok = ok.mean(axis=1)
        for t0 in range(0, T, 64):
            print(f"  k {ko + t0:5d}..{ko + t0 + 63:5d}: " + "".join("#" if r > 0.9 else "n" if np.isnan(g[t0 + i]).all()
                  else "0" if (g[t0 + i] == 0).all() else "." for i, r in enumerate(rowok[t0:t0 + 64])))
        if os.environ.get("NPU_AONEHOT_MAP"):   # (diag) for each feature n, the panel row m whose values it holds
            wfull = (wq if wq is not None else value(w))   # [n][k]
            np.save(os.path.join(work, "oh_got.npy"), g)
            np.save(os.path.join(work, "oh_w.npy"), wfull)
            best = []
            for n in range(g.shape[1]):
                sc = [(np.isclose(g[:, n], wfull[m, ko:ko + T], rtol=1e-2, atol=1e-30).mean(), m) for m in range(wfull.shape[0])]
                r, m = max(sc)
                best.append(f"{m}" if r > 0.9 else f"{m}?{r:.2f}")
            print("  feature -> row:", " ".join(best))
            okm = np.isclose(g, want, rtol=1e-2, atol=1e-30)   # [t][n]
            for n in range(g.shape[1]):
                print(f"  f{n:3d} " + "".join("#" if okm[k0:k0 + 8, n].all() else "." if not okm[k0:k0 + 8, n].any()
                                            else "+" for k0 in range(0, T, 8)))
        for t in (0, 1, 8, 64):
            if t < T:
                print(f"  t {t}: got {g[t, :6]} want {want[t, :6]}")
        sys.exit(0)
    grp = N.groups(cfg) or (passes,)
    err, k0 = 0.0, 0
    for g, gp in enumerate(grp):   # each group's partial: its K range
        k1 = k0 + 8 * gp * sum(ks)
        part = unpack_c(raw, cols, nb, len(grp), g)
        ref = value(a[:, k0:k1]) @ (value(w[:, k0:k1]) if wq is None else wq[:, k0:k1]).T
        if os.environ.get("GN_QMB1") in ("2", "5") and not N.qp1k_on():   # (diag) row 1's column halves swapped
            R_ = int(os.environ.get("NPU_QROWS", "2"))
            tb, nbr_ = N.TM, nb // R_
            # (a block's row: the pair relay interleaves a pair's blocks, else row r holds blocks r nbr ..)
            inter_ = os.environ.get("NPU_QCREL") == "2" or os.environ.get("GN_QEMERGE") == "1"   # (row = b mod R)
            odd = [(b % 2 == 1) if inter_ else (b // nbr_) % 2 == 1 for b in range(nb)]
            for c in range(cols):
                for b in range(nb):
                    if odd[b]:
                        blk = ref[b * tb:(b + 1) * tb, N.TN * c:N.TN * (c + 1)]
                        ref[b * tb:(b + 1) * tb, N.TN * c:N.TN * (c + 1)] = np.roll(blk, -N.TN // 2, axis=1)
        ulp = np.exp2(np.floor(np.log2(np.maximum(np.abs(ref), 1e-30))) - 7)   # of bf16 at ref
        e = np.abs(part - ref) / ulp
        e = np.where(np.isnan(e), np.inf, e)   # (Python max drops NaN)
        err = max(err, float(e.max()))
        if err > 0.501:   # where: per column, the bad output rows (features) and tokens
            for c in range(cols):
                ec = e[:, N.TN * c:N.TN * (c + 1)]
                if ec.max() > 0.501:
                    bad = np.argwhere(ec > 0.501)
                    print(f"group {g} column {c}: {len(bad)} bad, features {np.unique(bad[:, 1] // 16)} (slabs), "
                          f"tokens {bad[:, 0].min()}..{bad[:, 0].max()}")
                    if os.environ.get("NPU_BADMAP"):   # (diag) M blocks, rows of a block, tokens within a row
                        tb = N.MP * 16
                        print("  M blocks:", np.unique(bad[:, 0] // tb).tolist())
                        print("  rows in a block:", np.bincount(bad[:, 0] % tb // 16, minlength=N.MP).tolist(),
                              "token in row:", np.bincount(bad[:, 0] % 16, minlength=16).tolist(),
                              "feature in slab:", np.bincount(bad[:, 1] % 16, minlength=16).tolist())
                        pc_ = part[:, N.TN * c:N.TN * (c + 1)]
                        rc_ = ref[:, N.TN * c:N.TN * (c + 1)]
                        i0 = bad[0]
                        print("  first bad:", i0.tolist(), "got", pc_[i0[0], i0[1]], "want", rc_[i0[0], i0[1]])
                        if os.environ.get("NPU_BLKMAP"):   # (diag) per output M block: the reference block it holds
                            tb = N.MP * 16
                            pb = pc_.reshape(nb, tb, -1)
                            rb = rc_.reshape(nb, tb, -1)
                            print("  block map:", [int(np.argmin([np.nanmean(np.abs(pb[b] - rb[b2])) for b2 in range(nb)]))
                                                   for b in range(nb)])
                        if os.environ.get("NPU_SRCMAP"):   # (diag) which reference column's values this column holds
                            fin_ = np.isfinite(pc_)
                            near = [float(np.mean(np.isclose(pc_[fin_], ref[:, N.TN * c2:N.TN * (c2 + 1)][fin_], rtol=1e-2)))
                                    for c2 in range(cols)]
                            print(f"  finite {fin_.mean():.3f}; share matching ref column:", [f"{v:.2f}" for v in near])
                            okm = (ec <= 0.501).reshape(nb, N.MP, 16, N.NP, 16).mean(axis=(0, 2, 4))
                            fim = np.isfinite(pc_).reshape(nb, N.MP, 16, N.NP, 16).mean(axis=(0, 2, 4))
                            print("  good share per sub-tile [slab][group]:", np.round(okm, 2).tolist())
                            print("  finite share per sub-tile [slab][group]:", np.round(fim, 2).tolist())
        k0 = k1
    got = unpack_c(raw, cols, nb, len(grp))
    exact = a.astype(np.float64) @ (w.astype(np.float64) if wq is None else wf).T
    rms = np.sqrt(np.mean((got - exact) ** 2) / np.mean(exact ** 2))
    print(f"npu gemm {M}x{Nn}x{K} ks={list(ks)} passes={passes} groups={list(grp)} (context {width} columns): max |err| "
          f"{err:.3f} bf16 ulp vs the bfp16 oracle; rel RMS {rms:.2e} vs unquantized f16")
    sys.exit(0 if err <= 0.501 else 1)   # each partial rounded once (RNE) from f32 accumulation


QMODE = os.environ.get("NPU_Q") == "1"   # layout Q' images: 64-wide tiles (NP 4)

if __name__ == "__main__":
    if QMODE:
        with N.np_override(int(os.environ.get("GN_QNP", "4"))):   # (GN_QNP: Q' experiments)
            main()
    else:
        main()
