#!/usr/bin/env python3
"""footprint_gate.py <file.loom> <sym> <fmt> <kind> <m_tiles> <k_blocks> <token_tiles> [B=2048]

Compile the source under the exact config and read each root's declared byte envelope from the compile report.
Refuse (exit 3) unless every envelope fits the buffer loom_forward_pp binds for that root.
An envelope past its buffer does not fault on this GPU: the shader reads unmapped VA and the gfx ring times out.
emit_prefill_pp runs this on every generated GEMM before it emits the HAL.
"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loom_preflight as lp
QB = {"q4k": (256, 144), "q5k": (256, 176), "q6k": (256, 210), "q3k": (256, 110), "q2k": (256, 84),
      "iq4xs": (256, 136), "iq3s": (256, 110), "iq3xxs": (256, 98), "iq2xxs": (256, 66), "iq2xs": (256, 74),
      "q8_0": (32, 34), "iq4nl": (32, 18)}
src, sym, fmt, kind, mt, kb, tt = sys.argv[1:8]
mt, kb, tt = int(mt), int(kb), int(tt)
B = int(sys.argv[8]) if len(sys.argv) > 8 else 2048
qk, bpb = QB[fmt]
M, K = mt * 16, kb * qk
bound = {"weight": M * kb * bpb, "input": B * K * 2, "resid": B * M * 4, "gate": B * M * 4,
         "output": B * M * (2 if kind in ("swiglu", "kqg") else 4),
         "gate_out": B * M * 2,
         # the driver's grid buffers (loom_forward_pp.cc): 512 / 256 / 512 / 1024 words
         "grid": {"iq3s": 2048, "iq3xxs": 1024, "iq2xxs": 2048, "iq2xs": 4096}.get(fmt, 0),
         "ksigns": 128,
         "wstage": 17408 * 16 * 2, "ostage": 20480 * B * 4}
# YAH_LOOM_HOME selects the HRX/Loom tree whose build compiles the HALs (default /home/q/hrx).
H = os.environ.get("YAH_LOOM_HOME", "/home/q/hrx")
e = dict(os.environ)
e["IREE_HAL_AMDGPU_LIBHSA_PATH"] = "/home/q/rocm10/x_runtime/opt/rocm/core-10.0/lib"
e["LD_LIBRARY_PATH"] = ":".join(["/var/lib/lemonade/.cache/lemonade/bin/therock/gfx1151-7.13.0/lib",
    H + "/libhrx/src/binding/hip", H + "/libhrx/src/libhrx", "/home/q/rocm10/x_runtime/opt/rocm/core-10.0/lib",
    "/home/q/rocm10/x_llvm/opt/rocm/core-10.0/lib/llvm/lib",
    "/home/q/rocm10/x_sysdeps/opt/rocm/core-10.0/lib/rocm_sysdeps/lib", "/opt/rocm/lib"])
rep = "/tmp/footprint_gate_report.json"
root = "@" + sym
cmd = [H + "/build/cmake/loom/src/loom/tools/loom-compile/loom-compile", src, "--root=" + root,
       "--target=amdgpu:gfx1151", "--format=amdgpu-hsaco", "--output=/tmp/footprint_gate.hsaco",
       "--compile-report=details", "--compile-report-output=" + rep,
       f"--config={sym}.m_tiles={mt}", f"--config={sym}.k_blocks={kb}", f"--config={sym}.token_tiles={tt}"]
r = subprocess.run(cmd, env=e, capture_output=True, text=True)
if r.returncode:
    print("GATE: compile failed", r.stderr[-800:]); sys.exit(2)
got = lp.declared_envelopes(rep, with_names=True)
if not got:
    print("GATE: no envelopes in report; refusing"); sys.exit(3)
rows, names = got
bad = []
for arg, nbytes in sorted(rows.items()):
    nm = names.get(arg, "?")
    lim = bound.get(nm)
    ok = lim is not None and nbytes <= lim
    if not ok:
        bad.append(f"{nm}: declares {nbytes} > bound {lim}")
if bad:
    print("GATE REFUSED " + os.path.basename(src) + f" m_tiles={mt} k_blocks={kb}: " + "; ".join(bad)); sys.exit(3)
print("gate ok " + ", ".join(f"{names.get(a,'?')}={b}" for a, b in sorted(rows.items())))
