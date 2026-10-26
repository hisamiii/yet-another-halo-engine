#!/usr/bin/env python3
"""npu_gate.py <model> <prefill set> [down pack] [workdir]: the NPU FFN block's smoke test.

Runs the same 4096-token prompt (two chunks, the pair schedule) through loom_forward_pp twice, on the GPU alone and
with --npu, and checks that the NPU ran its jobs without a failed wait and that its output is the model's: the last 64
positions' top-1 tokens and KL against the GPU run, and the last chunk's hidden states. The NPU path is not
bit-identical to the GPU's (bfp16 activations, IQ4_XS down rows): the bounds catch a hang, a fallback, or missing or
garbage NPU output, not a quality change (that is kld_ref.py's job: engine/run/gate). Needs numpy, an unlimited memlock
limit and a build with NPU support (engine/build_hrx.sh). The down pack defaults to <prefill set>/npu_down.bin.
Run it as a GPU job: engine/run/gpu_run.sh npugate -- engine/tests/npu_gate.py ...
"""
import os
import re
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TOKENS, TAIL, HID = 4096, 64, 5120
# GPU vs NPU on this prompt (serve-pp1024, MQ4 HFQ): top-1 1.000, KL 0.0014, cos 0.960; an all-zero down pack (the
# NPU's features dropped from every layer): 0.984, 0.0132, 0.837
TOP1_MIN, KL_MAX, COS_MIN = 0.95, 0.005, 0.93


def run(model, hal, prefix, npu_args):
    cmd = [os.path.join(ROOT, "engine", "build", "loom_forward_pp"), model, hal, prefix, str(TOKENS),
           os.path.join(ROOT, "engine", "run", "gate", "corpus", "ids2048_w00.txt"), "--pairs",
           "--logits-from", str(TOKENS - TAIL)] + npu_args
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if p.returncode:
        sys.exit("npu_gate: %s failed (%d):\n%s" % (cmd[0], p.returncode, p.stderr[-2000:]))
    ms = re.search(r"layers_ms=([\d.]+)", p.stdout)
    print("  %-4s %s ms of layers, %.0f s in all" % ("npu" if npu_args else "gpu", ms and ms.group(1), time.time() - t))
    return p.stdout


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    model, hal = sys.argv[1], sys.argv[2]
    down = sys.argv[3] if len(sys.argv) > 3 else os.path.join(hal, "npu_down.bin")
    work = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.expanduser("~"), "Development", "Scratch",
                                                              "npu-gate")
    if resource.getrlimit(resource.RLIMIT_MEMLOCK)[0] != resource.RLIM_INFINITY:
        sys.exit("npu_gate: the NPU needs an unlimited memlock limit (prlimit --pid $$ --memlock=unlimited:unlimited)")
    for f in (os.path.join(hal, "npu.txt"), down):
        if not os.path.exists(f):
            sys.exit("npu_gate: no %s (tools/emit_npu_ffn.py, tools/npu_down_pack.py)" % f)
    os.makedirs(work, exist_ok=True)
    run(model, hal, os.path.join(work, "gpu"), [])
    out = run(model, hal, os.path.join(work, "npu"), ["--npu", "--npu-down", down])

    fails = []
    jobs = sum(int(m) for m in re.findall(r"npu: \S+\s+(\d+) jobs", out))
    if not jobs or "npu: ERROR" in out:
        fails.append("the NPU ran %d jobs%s" % (jobs, ", a GPU wait failed" if "npu: ERROR" in out else ""))
    lg, ln = (np.fromfile(os.path.join(work, n + ".all_logits"), np.float32).reshape(TAIL, -1) for n in ("gpu", "npu"))
    top1 = float(np.mean(lg.argmax(1) == ln.argmax(1)))
    pg, pn = (np.exp(x - x.max(1, keepdims=True)) for x in (lg, ln))
    pg, pn = pg / pg.sum(1, keepdims=True), pn / pn.sum(1, keepdims=True)
    kl = float(np.mean(np.sum(pg * (np.log(pg + 1e-30) - np.log(pn + 1e-30)), 1)))
    hg, hn = (np.fromfile(os.path.join(work, n + ".hidden"), np.float32).reshape(-1, HID) for n in ("gpu", "npu"))
    cos = float(np.mean(np.sum(hg * hn, 1) / (np.linalg.norm(hg, axis=1) * np.linalg.norm(hn, axis=1) + 1e-30)))
    print("  %d NPU jobs; last %d positions: top-1 agreement %.3f, KL %.4f; hidden cos %.4f" % (jobs, TAIL, top1, kl,
                                                                                               cos))
    if top1 < TOP1_MIN:
        fails.append("top-1 agreement %.3f < %.2f" % (top1, TOP1_MIN))
    if kl > KL_MAX:
        fails.append("KL %.4f > %.3f" % (kl, KL_MAX))
    if cos < COS_MIN:
        fails.append("hidden cos %.4f < %.2f" % (cos, COS_MIN))
    print("NPU GATE " + ("FAIL: " + "; ".join(fails) if fails else "PASS"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
