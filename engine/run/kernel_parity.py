#!/usr/bin/env python3
"""Per-kernel HIP-vs-Loom parity table from device timestamps on both sides.

  kernel_parity.py <hip_kernel_trace.csv> <loom_disp.jsonl> <loom_seq.csv>

hip:  rocprofv3 --kernel-trace --output-format csv ... yah-run ...
loom: YAH_LOOM_SEQ=seq.csv HRX_PROFILE_FILE=p.irpf HRX_PROFILE_MODE=dispatch loom_forward_pp ...
      iree-profile dispatch --dispatch_events --format=jsonl p.irpf > loom_disp.jsonl

GEMMs are keyed (epilogue, format, M, K) on both sides; everything else by a
hand mapping of kernel names. Times are summed per key over the whole pp run.
"""
import collections, csv, json, re, sys

GT = {8: "q8_0", 10: "q2k", 11: "q3k", 12: "q4k", 13: "q5k", 14: "q6k", 16: "iq2xxs",
      17: "iq2xs", 18: "iq3xxs", 21: "iq3s", 23: "iq4xs"}
QK = {"q8_0": 32}  # everything else 256 per block
EPI = {0: "store", 1: "swiglu", 2: "gateup", 3: "resid"}
# Non-GEMM kernels: (hip name fragment, loom kernel name) -> common label.
OTHER = [
    ("BatchedDeltaNetRowSplitKernel", "yah_deltanet", "deltanet"),
    ("WmmaCausalAttention", "yah_attn_wmma", "attention"),
    ("BatchedSSMConvKernel", "yah_ssm_conv", "ssm_conv"),
    ("BatchedFusedQKNormRoPEKvWriteKernel", "yah_fused_qk_rope_batched", "qk_rope"),
    ("BatchedUnpackQGKernel", "yah_unpack_qg", "unpack_qg"),
    ("BatchedDeltaNetPrepKqKernel", "yah_deltanet_prep_kq", "prep_kq"),
    ("BatchedDeltaNetPrepAlphaBetaKernel", "yah_deltanet_prep_ab", "prep_ab"),
    ("PackAttentionHeads", "yah_pack_heads", "pack_heads"),
    ("Q8KBlockGEMVKernel", "yah_gemv_q6k", "head"),
    ("BatchedEmbeddingLookupKernel", "yah_embed", "embed"),
    ("HalfNorm5120", "yah_half_norm", "half_norm"),
    ("BatchedSSMPostNormGateFp16Kernel", "yah_ssm_postnorm_fp16", "ssm_postnorm"),
    ("HalfCast", "yah_half_cast", "half_cast"),
    ("ArgmaxKernel", "yah_argmax", "argmax"),
    ("RMSNormKernel", "yah_rmsnorm", "final_norm"),
]


def hip_rows(path):
    rows = list(csv.DictReader(open(path)))
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    out = []
    for r in rows:
        name = r["Kernel_Name"].replace("(anonymous namespace)::", "")
        ms = (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e6
        gx = int(r["Grid_Size_X"]) // int(r["Workgroup_Size_X"])
        gy = int(r["Grid_Size_Y"])
        m = re.search(r"HalfPrefillGemmKernel<(\d+), (\d+), \d+, \d+, \(gufo::core::GgmlType\)(\d+), "
                      r"\(gufo::hip::HalfEpilogue\)(\d)", name)
        if m:
            bm, bn, t, ep = int(m[1]), int(m[2]), int(m[3]), int(m[4])
            M = gy * bm
            epi = EPI[ep]
            if epi == "gateup":
                M //= 2
            fmt = GT[t]
            if bm == 32 and M == 64:
                M = 48  # the ssm alpha/beta projections (48 heads) round up to a tile
            # K is filled in by align() from the Loom GEMM this one matches.
            out.append((("gemm", epi, fmt, M, None), ms, name))
            continue
        label = None
        for frag, _, lab in OTHER:
            if frag in name:
                label = lab
        if label is None:
            label = "hip:" + re.sub(r"\(.*$", "", name.replace("(anonymous namespace)::", ""))[:60] + f" g={gx}x{gy}"
        out.append((("op", label), ms, name))
    return out


def loom_rows(jsonl, seq):
    ev = [json.loads(l) for l in open(jsonl) if '"dispatch_event"' in l]
    ev.sort(key=lambda e: e["start_tick"])
    sq = list(csv.DictReader(open(seq)))
    assert len(ev) == len(sq), (len(ev), len(sq))
    out = []
    for e, s in zip(ev, sq):
        g = e["workgroup_count"]
        assert [int(s["gx"]), int(s["gy"]), int(s["gz"])] == g, (s, g)
        key = s["key"]
        ms = e["duration_ns"] / 1e6
        m = re.match(r"gemm_(kstore|swiglu|kres)_(\w+?)_(\d+)_(\d+)\.hal", key)
        if m:
            kind, fmt, mt, kb = m[1], m[2], int(m[3]), int(m[4])
            epi = {"kstore": "store", "swiglu": "swiglu", "kres": "resid"}[kind]
            M, K = mt * 16, kb * QK.get(fmt, 256)
            out.append((("gemm", epi, fmt, M, K), ms, key))
            continue
        label = None
        for _, lname, lab in OTHER:
            if key == lname:
                label = lab
        out.append((("op", label or "loom:" + key), ms, key))
    return out


def align(hip, loom):
    """Give each HIP GEMM the K of the Loom GEMM it corresponds to.

    Both engines run two half_norms per layer (attn, ffn), so the streams split
    into the same segments; within a segment GEMMs pair up by (format, M) in
    order. HIP's fused gate+up pairs with Loom's gate kStore (same M, format)."""
    def segments(rows, norm):
        seg, cur = [], []
        for r in rows:
            if r[0] == ("op", norm):
                seg.append(cur)
                cur = []
            cur.append(r)
        seg.append(cur)
        return seg
    hs = segments(hip, "half_norm")
    ls = segments(loom, "half_norm")
    assert len(hs) == len(ls), (len(hs), len(ls))
    out, misses = [], 0
    for h, l in zip(hs, ls):
        pool = [r[0] for r in l if r[0][0] == "gemm"]
        for k, ms, name in h:
            if k[0] == "gemm":
                _, epi, fmt, M, _ = k
                hit = next((c for c in pool if c[2] == fmt and c[3] == M and
                            (c[1] == "resid") == (epi == "resid")), None)
                if hit is None:
                    hit = next((c for c in pool if c[3] == M and
                                (c[1] == "resid") == (epi == "resid")), None)
                if hit is None:
                    misses += 1
                    k = ("gemm", epi, fmt, M, "?")
                else:
                    pool.remove(hit)
                    k = ("gemm", epi, fmt, M, hit[4])
            out.append((k, ms, name))
    if misses:
        print(f"warning: {misses} HIP GEMMs without a Loom counterpart", file=sys.stderr)
    return out


def agg(rows):
    d = collections.defaultdict(lambda: [0, 0.0])
    for k, ms, _ in rows:
        d[k][0] += 1
        d[k][1] += ms
    return d


def main():
    lr = loom_rows(sys.argv[2], sys.argv[3])
    hip = agg(align(hip_rows(sys.argv[1]), lr))
    loom = agg(lr)
    # HIP fuses gate+up (gateup) where Loom runs store + swiglu: compare the
    # pair against the fused kernel under one combined key.
    def merge_ffn(d):
        m = collections.defaultdict(lambda: [0, 0.0])
        for k, (n, ms) in d.items():
            if k[0] == "gemm" and k[1] in ("gateup", "swiglu") or (k[0] == "gemm" and k[1] == "store" and k[3] == 17408):
                k = ("gemm", "ffn_gate+up", k[2], 17408, k[4])
            m[k][0] += n
            m[k][1] += ms
        return m
    hip, loom = merge_ffn(hip), merge_ffn(loom)
    keys = sorted(set(hip) | set(loom), key=lambda k: -max(hip.get(k, [0, 0])[1], loom.get(k, [0, 0])[1]))
    th = sum(v[1] for v in hip.values())
    tl = sum(v[1] for v in loom.values())
    print(f"{'key':44s} {'hip n':>5s} {'hip ms':>8s} {'loom n':>6s} {'loom ms':>8s} {'loom/hip':>8s} {'gap ms':>8s}")
    for k in keys:
        hn, hm = hip.get(k, [0, 0.0])
        ln, lm = loom.get(k, [0, 0.0])
        ks = " ".join(str(x) for x in k[1:])
        r = f"{lm / hm:8.2f}" if hm and lm else "       -"
        print(f"{ks[:44]:44s} {hn:5d} {hm:8.1f} {ln:6d} {lm:8.1f} {r} {lm - hm:8.1f}")
    cat = collections.defaultdict(lambda: [0.0, 0.0])
    for k in keys:
        c = "gemm" if k[0] == "gemm" and k[3] > 64 else ("gemm_small" if k[0] == "gemm" else k[1])
        cat[c][0] += hip.get(k, [0, 0.0])[1]
        cat[c][1] += loom.get(k, [0, 0.0])[1]
    print()
    for c, (hm, lm) in sorted(cat.items(), key=lambda x: -(x[1][1] - x[1][0])):
        r = f"{lm / hm:8.2f}" if hm and lm else "       -"
        print(f"  {c:42s} {hm:8.1f} {lm:8.1f} {r} {lm - hm:8.1f}")
    print(f"{'TOTAL':44s} {'':5s} {th:8.1f} {'':6s} {tl:8.1f} {tl / th:8.2f} {tl - th:8.1f}")


if __name__ == "__main__":
    main()
