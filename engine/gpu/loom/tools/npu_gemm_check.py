#!/usr/bin/env python3
"""Check gen_npu_gemm.py on the NPU against a float64 oracle of the bfp16ebs8 math.

usage: npu_gemm_check.py <workdir> <cols> <m_blocks> <ks,ks,ks,ks> <passes> [act.f16 wgt.f16]

Operands are random (wide per-row dynamic range) or f16 files: activations [64 * m_blocks][K], weights [TN * cols][K] (gen_npu_gemm.TN).
They are packed with the encoder's numpy oracle (bfp16_check.reference), the image runs through the engine's npu_run, and each C
partial (one per replay group of passes, gen_npu_gemm.groups) is compared with the exact product of its K range of the
bfp16 values; C is bf16: within half an ulp of the exact product (f32 accumulation order is ~1e-7 below that).
Time the NPU with the counter profiler, not wall clock.
"""
import dataclasses
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


def bf16(x):
    """f32 -> bf16 (nearest even), as f32."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def qsw_h(g, u):
    if os.environ.get("GN_QSWH") == "one":
        return np.ones_like(g)
    if os.environ.get("GN_QSWH") == "raw":
        return g.astype(np.float32)
    if os.environ.get("GN_QSWH") in ("g", "u"):   # (diag) the image's h = g or u
        return bf16(g if os.environ["GN_QSWH"] == "g" else u)
    """gen_npu_gemm convert_swiglu on f32 sums g, u: h = max(g, 0) u + t (-|g| u) in f32, bf16 operands, t = sigmoid(-|g|)
    from its table (|g| clamped to [2^-8, 128] as bf16 bits), -|g| u rounded to bf16."""
    gb, ub = bf16(g), bf16(u)
    x = (np.arange(N.SL_LO, N.SL_HI + 1, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)
    tt = (1 / (1 + np.exp(x))).astype(np.float32)
    tt = bf16(tt)
    tt[tt < np.float32(2.0 ** -126)] = 0
    la = (gb.view(np.uint32) >> 16) & 0x7FFF
    t = tt[np.clip(la, N.SL_LO, N.SL_HI) - N.SL_LO]
    fp = np.maximum(gb, np.float32(0)) * ub
    fqb = bf16(-np.abs(gb) * ub)
    return (fp + t * fqb).astype(np.float32)


def qsw_check(hraw, av, wq, D, cols, nb, hpass):
    """NPU_QSW: H ([M block][call of hpass, this one 0][slab][column][576 B]; 576 B: k-blocks in P4 order, token
    halves, 8 x [E][8 m]; the other calls' records must stay 0) against the model:
    g, u = the exact sums (f64) to f32, qsw_h, the NPU's bfp16 conversion (gen_npu_dec Q.bfp_hw). Returns the exit code."""
    F = wq.shape[0] // 2
    s_ = av.astype(np.float64) @ wq.T.astype(np.float64)
    h = qsw_h(s_[:, :F].astype(np.float32), s_[:, F:].astype(np.float32))   # [tokens][F]
    M = h.shape[0]
    perm = (0, 2, 1, 3)
    # [col][M block][slab][position p][token half][8 tokens][8 features]: k-block perm[p] of the column's 4
    hv = h.reshape(nb, N.MP, 2, 8, cols, 4, 8)[:, :, :, :, :, list(perm), :]
    hv = hv.transpose(0, 1, 4, 5, 2, 3, 6)   # [M block][slab][column][p][token half][8][8]
    want = D.Q.bfp_hw(np.ascontiguousarray(hv).reshape(-1, 8, 8)).reshape(-1)
    hj = hraw[:nb * hpass * N.MP * cols * N.SW_REC].reshape(nb, hpass, -1)
    other = int(np.count_nonzero(hj[:, 1:]))
    print(f"NPU_QSW H: the other {hpass - 1} calls' records hold {other} nonzero bytes (want 0)")
    got = np.ascontiguousarray(hj[:, 0]).reshape(-1)
    if os.environ.get("GN_QSWDUMP"):   # (diag) the sums and both H to an npz
        np.savez(os.environ["GN_QSWDUMP"], g=s_[:, :F], u=s_[:, F:], want=want, got=got)
    bad = np.flatnonzero(got != want)
    print(f"NPU_QSW H: {got.size} bytes, {bad.size} differ ({bad.size / got.size:.2e})")
    if bad.size:
        rec = bad // 9
        print(f"  first: byte {bad[0]} (record {rec[0] // 64} of [block][slab][col], row byte {bad[0] % 9}) "
              f"got {got[bad[0]]} want {want[bad[0]]}; exponent bytes {np.count_nonzero(bad % 9 == 0)}")
        dm = np.abs(got[bad].view(np.int8).astype(int) - want[bad].view(np.int8).astype(int))
        print(f"  mantissa |diff| max {dm[bad % 9 != 0].max() if np.any(bad % 9) else 0}; "
              f"records hit {np.unique(bad // 576).size} of {got.size // 576}")
    return 0 if bad.size == 0 and other == 0 else 1


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
    # NPU_QSW=1 (Q', NP 4, fuse 2): the FFN gate / up image (Config.qsw): a layer's ffn_gate rows 0 .. N / 2 and the
    # same ffn_up rows (NPU_QSW_LAYER, else the first layer with both IQ4_XS), H against an exact model of the epilogue
    qsw = os.environ.get("NPU_QSW") == "1"
    assert fuse in (0, 2), "NPU_FUSE: 0 or 2"
    # NPU_GATE=1 (diag): the gated image (the engine's bindings) with its flag record already open for one call
    # NPU_GORD: the decoder's k-block order (gen_npu_gemm Config.gord; "GF": Q4_K pushed in P4)
    cfg = N.Config(cols, nb, ks, passes, mu=mu, fuse=fuse, kraw=kraw, fmt=fmt, rawdma=int(fuse == 2),
                   gate=N.GATE_SUPPLY if os.environ.get("NPU_GATE") else 0,
                   gord=os.environ.get("NPU_GORD", "P4"))
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
        if qsw:   # W = [gate rows][up rows], N / 2 each; the raw stream's up rows dgu bytes after the gate rows
            assert QMODE and N.NP == 4 and not k0r
            ts_ = {x.name: x for x in rd.tensors}
            lay_ = os.environ.get("NPU_QSW_LAYER") or next(
                n_.split(".")[1] for n_ in ts_ if n_.endswith("ffn_gate.weight") and ts_[n_].tensor_type.name == fmt
                and ts_[n_.replace("gate", "up")].tensor_type.name == fmt)
            # (NPU_QSW_UFMT=<type>: column pairs, the up rows of that format from NPU_QSW_ULAYER's ffn_up, on their
            # own raw binding; gate columns 2 u, up columns 2 u + 1, the up decoder's own rows)
            ufmt_ = os.environ.get("NPU_QSW_UFMT", "")
            tg_ = ts_[f"blk.{lay_}.ffn_gate.weight"]
            tu_ = ts_[f"blk.{os.environ.get('NPU_QSW_ULAYER') or lay_}.ffn_up.weight"]
            assert int(tg_.shape[0]) == KR and tg_.tensor_type.name == fmt and tu_.tensor_type.name == (ufmt_ or fmt)
            print(f"NPU_QSW: {tg_.name} / {tu_.name} rows 0 .. {Nn // 2}" + (f" (column pairs, up {ufmt_})" if ufmt_ else ""))
            if ufmt_:   # (N.decoder reconfigures one shared decoder: take each format's right before its use)
                wrawg = np.asarray(tg_.data).view(np.uint8).reshape(-1, KR // 256, N.decoder(fmt, "P4").BLK)[:Nn // 2].copy()
                wrawu = np.asarray(tu_.data).view(np.uint8).reshape(-1, KR // 256, N.decoder(ufmt_, "P4").BLK)[:Nn // 2].copy()
                D = N.decoder(fmt, cfg.gord)
                cfg = dataclasses.replace(cfg, qsw=True, dgu=0, swcol=32, hord="P4", ufmt=ufmt_,
                                          ugord=os.environ.get("NPU_QSW_UGORD") or ("GF" if ufmt_ == "Q4_K" else "P4"),
                                          hpass=int(os.environ.get("NPU_QSW_HPASS", "2")))
            else:
                wraw = np.concatenate([np.asarray(x.data).view(np.uint8).reshape(-1, KR // 256, D.BLK)[:Nn // 2]
                                       for x in (tg_, tu_)])
                cfg = dataclasses.replace(cfg, qsw=True, dgu=Nn // 2 * KR // 256 * D.BLK, swcol=32, hord="P4",
                                          hpass=int(os.environ.get("NPU_QSW_HPASS", "2")))   # (H: this call is call 0)
            if int(os.environ.get("NPU_QSW_GAP", "0")):   # the up rows NPU_QSW_GAP rows past the gate rows (the
                cfg = dataclasses.replace(cfg, dgu=int(os.environ["NPU_QSW_GAP"]) * KR // 256 * D.BLK)   # engine: M)
        else:
            wraw = np.asarray(t.data).view(np.uint8).reshape(-1, KR // 256, D.BLK)[:Nn].copy()
        if qsw and cfg.ufmt:   # (column pairs) each half by its own decoder
            wfu = N.decoder(cfg.ufmt, "P4").weights(np.ascontiguousarray(wrawu)).reshape(Nn // 2, K)
            D = N.decoder(fmt, cfg.gord)
            wf = np.concatenate([D.weights(np.ascontiguousarray(wrawg)).reshape(Nn // 2, K), wfu])
        else:
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
    if os.environ.get("GN_QTL2") == "1" and N.QFCYC and os.environ.get("GN_QFW") == "1":   # (diag, P1K fill rows)
        out_, seen_ = [], set()   # absolute cycles (low word) at the fill's end (before the first C slot acquire) /
        for ln_ in text_.splitlines():   # after the block-start W / A acquires -> %qfc 32 / 36 / 40 = words 5-7
            if ln_.startswith("low.func.def"):
                seen_ = set()
            pre_, post_ = ln_ == "  acq %am1, 2" and 32 not in seen_, None
            if pre_:
                seen_.add(32)
                post_ = 32
                out_.append(ln_)
            else:
                out_.append(ln_)
                post_ = (36 if ln_.startswith("  %qfc = storage ") and 36 not in seen_ else   # (the firing's start)
                         40 if ln_ == "  acq %am1, 0" and 40 not in seen_ else None)
            if post_ is not None:
                seen_.add(post_)
                stamp_ = [f"  %tq2w{post_} = mov.cycles", f"  %tq2l{post_} = slice %tq2w{post_}[0] : reg<aie2p.er x2> -> reg<aie2p.er>",
                          f"  %tq2a{post_} = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>",
                          f"  %tq2k{post_} = mova.i32 32", f"  %tq2m{post_} = mov.modifier %tq2k{post_}",
                          f"  %tq2b{post_} = padda.modifier %tq2a{post_}, %tq2m{post_}",
                          f"  st %tq2l{post_}, %tq2b{post_}, {post_ - 32}"]
                if pre_:   # (before the acquire: the fill's end)
                    out_[-1:-1] = stamp_
                else:
                    out_ += stamp_
        text_ = "\n".join(out_) + "\n"
    if os.environ.get("GN_QTL3") in ("1", "2") and N.QFCYC and not os.environ.get("GN_QFW"):   # (diag, P1K fill rows)
        out_ = []   # absolute cycles (low word) after the firing's first / last own-slab egress acquire (%fzAd2) and
        r2_ = os.environ.get("GN_QTL3") == "2"   # at the firing's start -> %qfc 32 / 40 / 36 = words 5 / 7 / 6
        for ln_ in text_.splitlines():   # (GN_QTL3=2: word 7 = after the firing's first raw record acquire instead)
            out_.append(ln_)
            if r2_ and ln_.startswith("  acq %fzAm1, "):
                out_ += ["  %tq4w = mov.cycles", "  %tq4l = slice %tq4w[0] : reg<aie2p.er x2> -> reg<aie2p.er>",
                         "  %tq4a = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>",
                         "  %tq4k = mova.i32 32", "  %tq4m = mov.modifier %tq4k",
                         "  %tq4b = padda.modifier %tq4a, %tq4m", "  %tq4o = lda %tq4b, 8", "  %tq4e = eqz %tq4o",
                         "  %tq4v = mul %tq4e, %tq4l", "  %tq4n = add.rr %tq4o, %tq4v", "  st %tq4n, %tq4b, 8"]
                continue
            if ln_.startswith("  %qfc = storage "):
                out_ += ["  %tq3c = mov.cycles", "  %tq3cl = slice %tq3c[0] : reg<aie2p.er x2> -> reg<aie2p.er>",
                         "  %tq3a0 = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>",
                         "  %tq3k0 = mova.i32 32", "  %tq3m0 = mov.modifier %tq3k0",
                         "  %tq3b0 = padda.modifier %tq3a0, %tq3m0", "  %tq3z0 = mova.i32 0",
                         "  st %tq3z0, %tq3b0, 0", "  st %tq3cl, %tq3b0, 4"] + (["  st %tq3z0, %tq3b0, 8"] if r2_ else [])
            elif ln_.startswith("  acq %fzAd2, "):
                out_ += ["  %tq3w = mov.cycles", "  %tq3l = slice %tq3w[0] : reg<aie2p.er x2> -> reg<aie2p.er>",
                         "  %tq3a = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>",
                         "  %tq3k = mova.i32 32", "  %tq3m = mov.modifier %tq3k",
                         "  %tq3b = padda.modifier %tq3a, %tq3m", "  %tq3o = lda %tq3b, 0", "  %tq3e = eqz %tq3o",
                         "  %tq3v = mul %tq3e, %tq3l", "  %tq3n = add.rr %tq3o, %tq3v", "  st %tq3n, %tq3b, 0",
                         ] + ([] if r2_ else ["  st %tq3l, %tq3b, 8"])
        text_ = "\n".join(out_) + "\n"
    if os.environ.get("GN_QCW") == "1" and N.QCYC:   # (diag, Q' leaves) per block: cycles in the entry C acquire, the
        out_, n_ = [], 0                             # lazy C acquires and the W acquires (private words 20 / 24 / 28,
        acqs_ = {"  acq %am1, 2": 20, "  acq.cond %am1, %qclk, 2": 24, "  acq %am1, 1": 28,   # reset per firing)
                 "  acq.cond %am1, %waq, 1": 28}   # -> words 1-3 of each stamped segment
        if os.environ.get("GN_QCWA") == "1":   # (word 3 = the A acquires instead of W)
            acqs_ = {k_: v_ for k_, v_ in acqs_.items() if v_ != 28}
            acqs_.update({"  acq %am1, 0": 28, "  acq.cond %am1, %cqn, 0": 28})
        if os.environ.get("GN_QCWT") == "1":   # (words 1 / 2 / 3 = the pair tokens / the relay copy / A)
            acqs_ = {"  acq %am2_e, 3": 20, "  acq %am1, 3": 20, "  acq %am1, 4": 20, "  acq.cond %am1, %mbq, 3": 20,
                     "  acq.cond %am1, %mbq, 4": 20, "  acq %crm1, 5": 24, "  acq %crm1, 2": 24,
                     "  acq %am1, 0": 28, "  acq.cond %am1, %cqn, 0": 28}
        for ln_ in text_.splitlines():
            n_ += 1
            o_ = acqs_.get(ln_)
            if o_ is not None:
                out_ += [f"  %qcwa{n_} = mov.cycles", f"  %qcwb{n_} = slice %qcwa{n_}[0] : reg<aie2p.er x2> -> reg<aie2p.er>"]
            out_.append(ln_)
            if o_ is not None:
                out_ += [f"  %qcwc{n_} = mov.cycles", f"  %qcwd{n_} = slice %qcwc{n_}[0] : reg<aie2p.er x2> -> reg<aie2p.er>",
                         f"  %qcwe{n_} = sub %qcwd{n_}, %qcwb{n_}",
                         f"  %qcwp{n_} = storage_address %tcs : low.storage<private> -> reg<aie2p.ep>",
                         f"  %qcwf{n_} = lda %qcwp{n_}, {o_}", f"  %qcwg{n_} = add.rr %qcwf{n_}, %qcwe{n_}",
                         f"  st %qcwg{n_}, %qcwp{n_}, {o_}"]
            if o_ is not None and ln_ == "  acq.cond %am1, %qclk, 2":   # the lazy acquire's done time from the firing start: words 40 / 44 / 48 (mp 0 / 1 / 2; 36: sink)
                out_ += [f"  %qcwhc{n_} = copy %qcwp{n_} : reg<aie2p.ep> -> reg<aie2p.ep>", f"  %qcwhk{n_} = mov.i32 32", f"  %qcwhm{n_} = mov.modifier %qcwhk{n_}",
                         f"  %qcwh{n_} = padds.modifier %qcwhc{n_}, %qcwhm{n_}",
                         f"  %qcwt{n_} = lda %qcwh{n_}, 0", f"  %qcwu{n_} = sub %qcwd{n_}, %qcwt{n_}",
                         f"  %qcwk{n_} = mova.i32 4", f"  %qcwm{n_} = mul %mp, %qcwk{n_}",
                         f"  %qcwn{n_} = add.rr %qcwm{n_}, %qcwk{n_}", f"  %qcwo{n_} = mul %qcwn{n_}, %qcl",
                         f"  %qcwv{n_} = mov.modifier %qcwo{n_}", f"  %qcwq{n_} = padds.modifier %qcwh{n_}, %qcwv{n_}",
                         f"  st %qcwu{n_}, %qcwq{n_}, 4"]
            if ln_ == "  st %tsn, %tca, 0":
                out_ += [f"  %qcwz{n_} = mova.i32 0"] + [f"  st %qcwz{n_}, %tca, {w_}" for w_ in (20, 24, 28)]
                out_ += [f"  %qcws{n_} = mov.cycles", f"  %qcsl{n_} = slice %qcws{n_}[0] : reg<aie2p.er x2> -> reg<aie2p.er>",
                         f"  %qcsc{n_} = copy %tca : reg<aie2p.ep> -> reg<aie2p.ep>", f"  %qcshk{n_} = mov.i32 32", f"  %qcshm{n_} = mov.modifier %qcshk{n_}",
                         f"  %qcsh{n_} = padds.modifier %qcsc{n_}, %qcshm{n_}",
                         f"  st %qcsl{n_}, %qcsh{n_}, 0"]
            if ln_ == "  st %tsp_l, %cvds, 0":
                out_ += [f"  %qcwyc{n_} = copy %tsa_l : reg<aie2p.ep> -> reg<aie2p.ep>", f"  %qcwyk{n_} = mov.i32 32",
                         f"  %qcwym{n_} = mov.modifier %qcwyk{n_}", f"  %qcwy{n_} = padds.modifier %qcwyc{n_}, %qcwym{n_}"]
                out_ += [f"  %qcwr{n_}_{w_} = lda {b2_}, {o2_}\n  st %qcwr{n_}_{w_}, %cvds, {4 * w_}"
                         for w_, b2_, o2_ in ((1, "%tsa_l", 20), (2, "%tsa_l", 24), (3, "%tsa_l", 28), (4, f"%qcwy{n_}", 8),
                                              (5, f"%qcwy{n_}", 12), (6, f"%qcwy{n_}", 16))]
        text_ = "\n".join(out_) + "\n"
    open(src, "w").write(text_)
    env = hrx_paths.env()
    env.update(Q.loom_env(cfg) if QMODE else N.loom_env(cfg))
    for k in os.environ.get("NPU_ENV_DROP", "").split(","):   # e.g. LOOM_EXP_PANEL_STREAM (diagnosis)
        env.pop(k, None)
    for kv in os.environ.get("NPU_ENV_ADD", "").split(","):    # e.g. LOOM_EXP_ROUTE_RESET=1 (diagnosis)
        if "=" in kv:
            k_, v_ = kv.split("=", 1)
            if k_ == "LOOM_EXP_SPACE_AT" and env.get(k_):   # (the image's own pins for spaces the add leaves out)
                named = {p_.split("@")[0] for p_ in v_.split(",")}
                v_ = ",".join([v_] + [p_ for p_ in env[k_].split(",") if p_.split("@")[0] not in named])
            env[k_] = v_
    if os.environ.get("NPU_SAVE_CMD"):   # (diag) the compile's command and Loom environment, to rerun it by hand
        import json
        json.dump({"cmd": [hrx_paths.LOOM_COMPILE, src, f"--root=@{cfg.entry}",
                           "--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11", f"--output={img}"],
                   "env": {k: v for k, v in env.items() if k.startswith("LOOM_")}},
                  open(os.environ["NPU_SAVE_CMD"], "w"), indent=1)
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
            if ln.startswith(("SHIM ", "TWIN ", "STAGE ", "packet share", "route dump", "dma dump", "RING ", "STORAGE ")):
                print(ln)
    if os.environ.get("NPU_COMPILE_ONLY"):   # (diag) stop after the image (compiler work, no NPU)
        sys.exit(0)
    a_b, w_b, c_b = (os.path.join(work, n) for n in ("a.bin", "w.bin", "c.bin"))
    if qsw and os.environ.get("GN_QSWDIAG") == "intake":   # (diag) plain C: column c = gate rows 32 c .., up rows 32 c ..
        idx_ = np.concatenate([np.r_[32 * c_:32 * c_ + 32, Nn // 2 + 32 * c_:Nn // 2 + 32 * c_ + 32] for c_ in range(cols)])
        wq, wf, qsw = wq[idx_], wf[idx_], False
    if qsw and os.environ.get("GN_QSWDIAG") == "convert":   # (diag) plain intake: column c's gate = W rows 64 c .. + 32,
        # its up the next 32 (the qsw model's gate / up halves gathered from them)
        idx_ = np.r_[np.concatenate([np.r_[64 * c_:64 * c_ + 32] for c_ in range(cols)]),
                     np.concatenate([np.r_[64 * c_ + 32:64 * c_ + 64] for c_ in range(cols)])]
        wq, wf = wq[idx_], wf[idx_]
    sa, sw, sc = N.stream_bytes(cfg)
    if qsw:   # H: [M block][call][slab][column][576 B]
        sc = nb * cfg.hpass * N.MP * cols * N.SW_REC
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
    (np.full(sc, 0xFF, np.uint8) if os.environ.get("NPU_CFILL") else np.zeros(sc // 4, np.float32)).tofile(c_b)   # (diag: NPU_CFILL, unwritten C shows)
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
        gap_ = int(os.environ.get("NPU_QSW_GAP", "0")) if qsw else 0
        if qsw and cfg.ufmt:   # (column pairs) the gate rows, then the up rows on their own binding
            ru_b = os.path.join(work, "ru.bin")
            np.concatenate([wrawg.reshape(-1), np.zeros(256, np.uint8)]).tofile(r_b)
            np.concatenate([wrawu.reshape(-1), np.zeros(256, np.uint8)]).tofile(ru_b)
            bindings += [d_b, r_b, ru_b]
            env.setdefault("NPU_RUN_WEIGHTS", f"{len(bindings) - 2},{len(bindings) - 1}")
        else:
            rawv = (np.concatenate([wraw[:Nn // 2].reshape(-1), np.zeros((gap_ - Nn // 2) * KR // 256 * D.BLK, np.uint8),
                                    wraw[Nn // 2:].reshape(-1)]) if gap_ else wraw.reshape(-1))
            np.concatenate([rawv[k0r // 256 * D.BLK:], np.zeros(256, np.uint8)]).tofile(r_b)
            bindings += [d_b, r_b]
            # the raw stream reads weights (the split lead starts it before the gate; GN_QPREFILL images need that)
            env.setdefault("NPU_RUN_WEIGHTS", str(len(bindings) - 1))   # (env: copied from os.environ above)
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
        seg = N.SW_REC if qsw else N.NP * 256 * 2   # bytes of one bf16 segment (NP sub-tiles of 16 x 16; qsw: a row's H)
        cyc = np.fromfile(co, np.uint32)[::seg // 4].astype(np.int64)
        if qsw:
            print("  (qsw: the H bytes carry the stamps, not checked)")
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
            if mrel:   # (each row's own blocks: staged row q owns 4 f + q, else row 2 p + j owns 4 f + 2 j + p)
                c4_ = cyc.reshape(cols, nb, N.MP)[0, :, -1]
                for r_ in range(4):
                    b_ = r_ if N.qcstage_on() else 2 * (r_ % 2) + r_ // 2
                    st_ = np.diff(c4_[b_::4])
                    print(f"  row {r_} (column 0): median {float(np.median(st_)):.0f} min {st_.min()} max {st_.max()}")
        if rel_:
            for r_ in range(R_):
                st_ = np.diff(cyc.reshape(cols, nb, N.MP)[0, r_::R_, -1])
                print(f"  row {r_} (core 0): median {float(np.median(st_)):.0f} min {st_.min()} max {st_.max()}")
        if os.environ.get("GN_QFCYC") == "1" and fuse == 2:   # (the filler's fill words 1-3 of its blocks' segments)
            w4f = np.fromfile(co, np.uint32).reshape(cols, nb, N.MP, seg // 4)[int(os.environ.get("GN_QCOL", "0")), :, :, :8].astype(np.int64)
            fb_ = ((1 if N.qcstage_on() else 2) if mrel else 1) if N.qp1k_on() and os.environ.get("NPU_QCREL") == "2" else 0
            v_ = w4f[fb_, -1]
            print(f"  fill (core 0 column, filling row's block {fb_}): egress {v_[1]} whole fill {v_[2]} raw waits {v_[3]}")
            if os.environ.get("GN_QF4R") == "1" and os.environ.get("GN_QFCOLS") == "1":   # (diag) whole fill per column
                wall_ = np.fromfile(co, np.uint32).reshape(cols, nb, N.MP, seg // 4)[:, :, -1, 2].astype(np.int64)
                for c_ in range(cols):
                    print(f"  column {c_} whole fill by row: " + " ".join(str(wall_[c_, 2 * (r_ % 2) + r_ // 2])
                                                                      for r_ in range(4)))
            if os.environ.get("GN_QF4R") == "1":   # (every row fills: row r's first block 2 (r % 2) + r // 2)
                for r_ in range(4):
                    u_ = w4f[2 * (r_ % 2) + r_ // 2, -1]
                    print(f"  fill row {r_}: decoder/egress {u_[1]} whole fill {u_[2]} raw waits {u_[3]} hot {u_[4]}")
                    if os.environ.get("GN_QFW") == "2":   # (diag) the gate check, the helper record copies
                        print(f"  fill row {r_}: gate check {u_[5]} copies {u_[6]}")
                    if os.environ.get("GN_QFW") in ("3", "4", "5", "6"):   # (diag) the helper-record branch's cycles
                        print(f"  fill row {r_}: helper-record branch {u_[5]}")
                    if os.environ.get("GN_QTL3") in ("1", "2"):   # (diag) the last firing's own-slab acquires from its start
                        print(f"  fill row {r_} egress acquires: first done +{(u_[5] - u_[6]) % 2**32} "
                              f"last done +{(u_[7] - u_[6]) % 2**32}")
                    if os.environ.get("GN_QFW") == "7":   # (diag) own egress waits: sum, longest, its acquire (1-based)
                        print(f"  fill row {r_} own egress waits: sum {u_[5]} longest {u_[6]} at acquire {u_[7]}")
                    if os.environ.get("GN_QFW") == "8":   # (diag) timer low words: the fill's start, acquire 1's end
                        print(f"  fill row {r_} timer: fill start {u_[5]} own acquire 1 done {u_[6]} (+{u_[6] - u_[5]}) longest {u_[7]}")
                    if os.environ.get("GN_QFW") == "1":   # (diag) the egress slot waits (own, copy), helper waits
                        print(f"  fill row {r_} waits: own egress {u_[5]} helper records {u_[6]} copy egress {u_[7]}")
            if os.environ.get("DQ_CYC") == "1":   # (diag) word 1: the decoder's cycles; word 4: its hot loop's
                print(f"  decoder (DQ_CYC): {v_[1]} cycles, hot loop {v_[4]}")
        if os.environ.get("GN_QCW") == "1" and mrel:   # (diag) per row: median per block of the C / W acquire waits
            w4c = np.fromfile(co, np.uint32).reshape(cols, nb, N.MP, seg // 4)[0, :, -1, :7].astype(np.int64)
            for r_ in range(4):
                b_ = r_ if N.qcstage_on() else 2 * (r_ % 2) + r_ // 2
                v_ = w4c[b_::4][1:]   # (the row's first block apart)
                print(f"  row {r_} waits a block (median, column 0): C entry {np.median(v_[:, 1]):.0f} "
                      f"C lazy {np.median(v_[:, 2]):.0f} W {np.median(v_[:, 3]):.0f}; slots 1-3 got at "
                      f"{np.median(v_[:, 4]):.0f} {np.median(v_[:, 5]):.0f} {np.median(v_[:, 6]):.0f}; "
                      f"first block: {w4c[b_][1:].tolist()}")
        if os.environ.get("GN_QCYCABS") == "1":   # (diag) the call's span over every core: last end - first start
            w2_ = np.fromfile(co, np.uint32).reshape(-1, seg // 4)[:, :2].astype(np.int64)
            st0_, en_ = w2_[:, 1], w2_[:, 0] + w2_[:, 1]
            print(f"  call span (all cores): {en_.max() - st0_.min()} cycles; first starts spread {st0_.max() - st0_.min()}")
            b4_ = w2_.reshape(cols, nb, N.MP, 2)[0, :4, -1]   # (column 0, blocks 0-3: their last segment)
            print("  column 0 blocks 0-3 (start, end) from the first start:",
                  [(int(s - st0_.min()), int(s + e - st0_.min())) for e, s in b4_])
        if os.environ.get("GN_QSTAMPS") == "1":   # (diag) every M block's stamp, core 0
            print("  block stamps (core 0):", [int(x) for x in last])
            a_ = cyc.reshape(cols, nb, N.MP)[:, :, -1]
            print("  last block done per column:", [int(x) for x in a_.max(axis=1)], "first block per column:",
                  [int(x) for x in a_.min(axis=1)])
        step = np.diff(last)
        mma = passes * N.MP * N.NP * sum(ks) * 4   # MMAs a core issues per M block
        med = float(np.median(step))
        print(f"cycles per M block (core 0, {nb} blocks): median {med:.0f} min {step.min()} max {step.max()}; "
              f"{mma} MMAs -> MMA share {mma / med:.1%}; first block done at {last[0]}")
        sys.exit(0)
    if qsw:
        sys.exit(qsw_check(np.fromfile(co, np.uint8), value(a), wq, D, cols, nb, cfg.hpass))
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
                        if os.environ.get("NPU_WEFF") and wq is not None:   # (diag) a bad feature's effective weights
                            # (least squares over the tokens) against the true ones, per 32-K sub-block: the fit
                            # w_eff = s w + t (s, t) and the rms misfit
                            A_ = value(a[:, k0:k1]).astype(np.float64)
                            for f_ in np.unique(bad[:, 1])[:1]:
                                fg_ = N.TN * c + f_
                                we_ = np.linalg.lstsq(A_, pc_[:, f_].astype(np.float64), rcond=None)[0]
                                wt_ = wq[fg_, k0:k1].astype(np.float64)
                                np.save(os.path.join(work, f"weff_{fg_}.npy"), np.stack([we_, wt_]))
                                out_ = []
                                for j in range((k1 - k0) // 32):
                                    x_, y_ = wt_[32 * j:32 * j + 32], we_[32 * j:32 * j + 32]
                                    st_ = np.polyfit(x_, y_, 1) if np.ptp(x_) > 0 else (0.0, 0.0)
                                    out_.append((round(float(st_[0]), 2), round(float(st_[1] / (np.std(x_) + 1e-12)), 2)))
                                print(f"  feature {fg_} per sub-block (scale, offset / std):", out_)
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
