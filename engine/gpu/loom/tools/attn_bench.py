#!/usr/bin/env python3
"""Standalone prefill attention at any depth: compile attention sources for each context start, run them interleaved in one
attn_bench process under HRX counters, print busiest-SQ and mean-SQ cycles per call against the WMMA floor.

  attn_bench.py [--reps N] [--depths 0,8192,...] [--tag DIR] <name>=<source.loom>[,ENV=VAL...] ...

A depth is the chunk's start_pos (the chunk is token_count 2048 queries; cache_capacity = depth + 2048, or --cap T for
every depth, as the engine's chunk HALs: cache_capacity = the whole context, which moves the allocation). The buffer sizes
come from the kernel's own size formulas, checked line by line in the source: a source with other formulas is refused.
Example: attn_bench.py --depths 0,8192,32768 clean=oset/l53f/.emit_tmp/yah_attn_hip.loom pers=oset/x8pr/.emit_tmp/yah_attn_hip.loom
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import hrx_paths  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
B, HEADS, KV_HEADS, HD = 2048, 24, 4, 256
FLOOR_MACS_PER_CYCLE = 10240   # 40 CUs x 2 SIMDs x one 16x16x16 f16 WMMA (4096 MACs) per 32 cycles
# The kernels' buffer size formulas this harness knows how to size: fp16 paged K / V^T, f32 gate, and either f32 Q with the
# q-norm weight + eps (the main kernel) or rotated f16 Q (the "_qr" kernels kv4 runs on its dequantized f16 pools).
COMMON = [
    "%qtot = index.mul %B, %c6144 : index",
    "%npages = index.div %npg0, %c256 : index",
    "%pcap = index.mul %npages, %c256 : index",
    "%kvtot = index.mul %pcap, %c1024 : index",
    "%vpitch = index.mul %vtiles, %c16 : index",
    "%vtot = index.mul %vpitch, %c1024 : index",
    "%g_flat = buffer.view %g_na[%base] : buffer -> view<[%qtot]xf32>",
    "%o_flat = buffer.view %o_na[%base] : buffer -> view<[%qtot]xf16>",
    "%k_flat = buffer.view %k_na[%base] : buffer -> view<[%kvtot]xf16>",
    "%v_flat = buffer.view %v_na[%base] : buffer -> view<[%vtot]xf16>",
    "%pt_flat = buffer.view %pt_na[%base] : buffer -> view<[%npages]xi32>",
]
LAYOUTS = {
    "f32q": COMMON + [
        "launch(%query: buffer, %gate: buffer, %key_cache: buffer, %value_cache: buffer, %output: buffer, %lse: buffer, %ptab: buffer, %qnw: buffer, %qeps: buffer)",
        "%q_flat = buffer.view %q_na[%base] : buffer -> view<[%qtot]xf32>",
        "%qw_flat = buffer.view %qw_na[%base] : buffer -> view<[%c256]xf32>",
        "%qe_flat = buffer.view %qe_na[%base] : buffer -> view<[%c1]xf32>",
    ],
    "f16q": COMMON + [
        "launch(%query: buffer, %gate: buffer, %key_cache: buffer, %value_cache: buffer, %output: buffer, %lse: buffer, %ptab: buffer)",
        "%q_flat = buffer.view %q_na[%base] : buffer -> view<[%qtot]xf16>",
    ],
}


PQUEUE = "%pqv = buffer.view %pq_na[%base] : buffer -> view<264xi32>"


def layout(text, path):
    """The layout whose size formulas all appear in the source; refuse the source otherwise. A queued persistent kernel
    (gen_attn_fa PERSIST_DYN) binds one more buffer, its 264-word queue, after the others."""
    if ", %pqueue: buffer)" in text:
        if PQUEUE not in text:
            sys.exit(f"refused {path}: its queue is not 264 i32")
        text = text.replace(", %pqueue: buffer)", ")")
    # an f16 gate view (gen_attn_fa R2P_GSIG16) covers half the f32 gate allocation: size it as the f32 one
    text = text.replace("%g_flat = buffer.view %g_na[%base] : buffer -> view<[%qtot]xf16>",
                        "%g_flat = buffer.view %g_na[%base] : buffer -> view<[%qtot]xf32>")
    for name, lines in LAYOUTS.items():
        if all(x in text for x in lines):
            break
    else:
        sys.exit(f"refused {path}: its buffer size formulas match no harness layout")
    if re.search(r"%lse\w* = buffer\.view", text) or re.search(r"view\.store [^\n]*%lse", text):
        sys.exit(f"refused {path}: it writes lse; size it in the harness first")
    return name


def buffers(lay, cap):
    npages = (cap + 255) // 256
    pcap = npages * 256
    vpitch = (pcap + 15) // 16 * 16
    q = f"s:{B * 6144 * 4}:1" if lay == "f32q" else f"h:{B * 6144 * 2}:1"
    bufs = [q, f"s:{B * 6144 * 4}:2", f"h:{pcap * 1024 * 2}:3", f"h:{vpitch * 1024 * 2}:4", f"z:{B * 6144 * 2}",
            f"z:{B * HEADS * 4}", f"p:{npages}"]
    return bufs + (["c:256:1.0", "c:1:0.000001"] if lay == "f32q" else [])


def grid(text):
    """(gx, gy) from the source's launch config: persistent kernels launch their %cpst slots on x."""
    m = re.search(r"%cpst = index.constant (\d+) : index", text)
    if m:
        return int(m.group(1)), 1
    qt = int(re.search(r"%cqt = index.constant (\d+) : index", text).group(1))
    hpw = int(re.search(r"%chpw = index.constant (\d+) : index", text).group(1))
    return (B + qt - 1) // qt, HEADS // hpw


def compile_hal(src, text, env, depth, outdir, cap):
    configs = [f"cache_capacity={cap}", f"token_count={B}", f"start_pos={depth}", f"num_heads={HEADS}",
               f"num_kv_heads={KV_HEADS}", f"head_dim={HD}", f"gqa={HEADS // KV_HEADS}"]
    key = hashlib.sha1((text + json.dumps(sorted(env.items())) + ",".join(configs)).encode()).hexdigest()[:16]
    out = os.path.join(outdir, f"{key}.hal")
    if not os.path.exists(out):
        cmd = [env.get("LOOM_BIN") or hrx_paths.LOOM_COMPILE, src, "--root=@yah_attn_wmma", "--target=amdgpu:gfx1151", "--format=amdgpu-hsaco",
               "--output=" + out + ".tmp"] + ["--config=attention_prefill." + c for c in configs]
        r = subprocess.run(cmd, env=dict(hrx_paths.env(), **env), capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"compile failed: {src} depth {depth}\n{r.stderr[-2000:]}")
        os.replace(out + ".tmp", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--depths", default="0,8192")
    ap.add_argument("--cap", type=int, default=0, help="cache_capacity for every depth (default depth + 2048)")
    ap.add_argument("--tag", default=os.path.expanduser("~/yah-scratch/attn_bench"))
    ap.add_argument("--inputs", help="real Q / K / V: <prefix>.q, .kpool, .vpool (loom_prefill YAH_ATTN_DUMP), one depth")
    ap.add_argument("--counters", default="", help="more HRX counters (e.g. SQ_INSTS_VALU,SQ_WAVES): per-call totals")
    ap.add_argument("--att", action="store_true", help="HRX ATT thread trace (one rep): decoded to <tag>/att_<depth>.txt")
    ap.add_argument("variants", nargs="+")
    a = ap.parse_args()
    depths = [int(d) for d in a.depths.split(",")]
    if any(d % B for d in depths):
        sys.exit(f"depths must be multiples of {B}")
    os.makedirs(a.tag, exist_ok=True)
    variants = []
    for v in a.variants:
        name, spec = v.split("=", 1)
        src, *envs = spec.split(",")
        text = open(src).read()
        lay = layout(text, src)
        env = {"LOOM_EXP_LICM": "1", **({"LOOM_EXP_LICM_MAX_LIVE": "6"} if "_kt2" in src else {})}
        env.update(e.split("=", 1) for e in envs)
        variants.append((name, src, text, env, lay))
    report = []
    extra = [c for c in a.counters.split(",") if c]
    if len({v[4] for v in variants}) > 1:
        sys.exit("variants of one run must share a buffer layout (f32 Q or f16 Q)")
    for depth in depths:
        cap = a.cap or depth + B
        if depth + B > cap:
            sys.exit(f"depth {depth} + {B} queries past --cap {cap}")
        bufs = buffers(variants[0][4], cap)
        if any(", %pqueue: buffer)" in v[2] for v in variants):
            bufs.append("z:1056")
        if a.inputs:
            for i, ext in ((0, ".q"), (2, ".kpool"), (3, ".vpool")):
                want = int(bufs[i].split(":")[1])
                got = os.path.getsize(a.inputs + ext)
                if got != want:
                    sys.exit(f"refused {a.inputs + ext}: {got} bytes, the kernel reads {want}")
                bufs[i] = "f:" + a.inputs + ext
        hals = []
        for name, src, text, env, _ in variants:
            gx, gy = grid(text)
            if "HAL" in env:   # a prebuilt code object for this one depth (e.g. a regperm rewrite of the compiled one)
                hals.append(f"{env['HAL']}@{gx},{gy}")
                continue
            hals.append(f"{compile_hal(src, text, env, depth, a.tag, cap)}@{gx},{gy}")
        prof = os.path.join(a.tag, "run.irpf")
        if os.path.exists(prof):
            os.remove(prof)
        env = dict(os.environ, **({"ATTN_BENCH_ZERO": str(len(bufs) - 1)} if bufs[-1] == "z:1056" else {}), YAH_HRX=hrx_paths.HRX, CQ_MEM_GB="6", YAH_NPU="0", HRX_PROFILE_MODE="counters",
                   HRX_PROFILE_COUNTERS=",".join(["SQ_BUSY_CYCLES"] + extra), HRX_PROFILE_FILE=prof)
        if a.att:
            env.update(HRX_PROFILE_MODE="executable", HRX_PROFILE_FILTER="yah_attn*")
            env.pop("HRX_PROFILE_COUNTERS")
        cmd = ["bash", "-c", f"set +u; a=(\"$@\"); set --; source {ROOT}/engine/hrx-env.sh >/dev/null 2>&1; exec \"${{a[@]}}\"", "x",
               f"{ROOT}/engine/run/gpu_run.sh", "gpuonly", "--", f"{ROOT}/engine/build/attn_bench", str(a.reps),
               str(len(bufs))] + bufs + hals
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        firsts = re.findall(r"^first (\d+) ([0-9a-f]{16})$", r.stdout, re.M)
        if r.returncode or len(firsts) != len(hals):
            sys.exit(f"attn_bench failed at depth {depth}:\n" + (r.stdout + r.stderr)[-2000:])
        tool = os.path.join(hrx_paths.BUILD, "runtime/src/iree/tools/iree-profile/iree-profile")
        if a.att:   # decode and keep the trace text; no counters in this mode
            out = os.path.join(a.tag, f"att_{depth}.txt")
            d = subprocess.run([tool, "att", "--format=text", prof], capture_output=True, text=True)
            open(out, "w").write(d.stdout + d.stderr)
            print(f"depth {depth}: ATT rc {d.returncode}, {len(d.stdout)} bytes -> {out}", flush=True)
            continue
        samples, totals = {}, {}
        for line in subprocess.run([tool, "counter", "--format=jsonl", "--counter_samples", prof],
                                   capture_output=True, text=True).stdout.splitlines():
            d = json.loads(line)
            if d.get("type") == "counter_sample" and d["counter"] == "SQ_BUSY_CYCLES":
                v = d.get("values") or [d["value"]]
                samples[d["dispatch_event_id"]] = (sum(v) / len(v), max(v), d.get("duration_ticks", 0))
            elif d.get("type") == "counter_sample" and d["counter"] in extra:
                totals[d["dispatch_event_id"], d["counter"]] = sum(d.get("values") or [d["value"]])
        ids = sorted(samples)
        n = len(hals)
        if len(ids) != n * (1 + a.reps):
            sys.exit(f"depth {depth}: {len(ids)} counter samples, expected {n * (1 + a.reps)}")
        floor = HEADS * (depth * B + B * (B + 1) // 2) * 2 * HD / FLOOR_MACS_PER_CYCLE
        for k, (name, *_rest) in enumerate(variants):
            rows = [samples[ids[n + r * n + k]] for r in range(a.reps)]
            mean = statistics.median(x[0] for x in rows)
            mx = statistics.median(x[1] for x in rows)
            us = statistics.median(x[2] for x in rows) / 100
            report.append((depth, name, firsts[k][1], mean, mx, us, floor))
            print(f"depth {depth:6d} {name:12s} hash {firsts[k][1]}  busiest SQ {mx / 1e6:8.3f} Mcyc ({100 * floor / mx:5.1f}% "
                  f"of floor)  mean SQ {mean / 1e6:8.3f} ({100 * floor / mean:5.1f}%)  wall {us:9.1f} us" +
                  "".join(f"  {c} {statistics.median(totals[ids[n + r * n + k], c] for r in range(a.reps)):.4g}"
                          for c in extra), flush=True)
        os.remove(prof)
    return report


if __name__ == "__main__":
    main()
