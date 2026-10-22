#!/usr/bin/env python3
"""footprint_gate.py <file.loom> <sym> <fmt> <kind> <m_tiles> <k_blocks> <token_tiles> [B=2048] [masked]
[aspan=<K per activation scale>]

Compile the source under the exact config and read each root's declared byte envelope from the compile report.
Refuse (exit 3) unless every envelope fits the buffer loom_forward_pp binds for that root.
An envelope past its buffer does not fault on this GPU: the shader reads unmapped VA and the gfx ring times out.
emit_prefill_pp runs this on every generated GEMM before it emits the HAL.
"""
import atexit, os, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loom_preflight as lp
QB = {"mq4": (256, 136)}   # elements, bytes per block
src, sym, fmt, kind, mt, kb, tt = sys.argv[1:8]
mt, kb, tt = int(mt), int(kb), int(tt)
B = int(sys.argv[8]) if len(sys.argv) > 8 else 2048
# masked: the kernel declares its token count (config "tokens") and the last token tile is partial
qk, bpb = QB[fmt]
M, K = mt * 16, kb * qk
wbytes = M * kb * bpb
bound = {"weight": wbytes, "input": B * kb * qk * 2, "resid": B * M * 4, "gate": B * M * 4,
         "output": B * M * (2 if kind in ("swiglu", "kqg", "gu") else 4),
         # A4 gu (gen_gemm_iu4): the up tensor beside the gate tensor in "weight"
         "up": wbytes,
         "gate_out": B * M * 2,
         "wstage": 17408 * 16 * 2, "ostage": 20480 * B * 4,
         # gen_gemm_iu4 flag: a counter word per token tile (gu: per 1024 features and token tile)
         "cnt": tt * 4 * (M // 1024 if kind == "gu" else 1), "wcnt": tt * 4}
# A4 GEMMs (gen_gemm_iu4): int4 activations and their f32 scales, one per aspan values of a token
aspan = next((int(a.split("=")[1]) for a in sys.argv[9:] if a.startswith("aspan=")), 0)
if aspan:
    bound.update(aq=B * K // 2, ad=B * (K // aspan) * 4)
sys.path.insert(0, os.path.dirname(HERE))
import hrx_paths  # noqa: E402
e = hrx_paths.env()
# per-call temporaries: emits may run in parallel
tmp = tempfile.mkdtemp(prefix="footprint_gate_")
atexit.register(shutil.rmtree, tmp, True)
rep = os.path.join(tmp, "report.json")
root = "@" + sym
cmd = [hrx_paths.LOOM_COMPILE, src, "--root=" + root,
       "--target=amdgpu:gfx1151", "--format=amdgpu-hsaco", "--output=" + os.path.join(tmp, "k.hsaco"),
       "--compile-report=details", "--compile-report-output=" + rep,
       f"--config={sym}.m_tiles={mt}", f"--config={sym}.k_blocks={kb}", f"--config={sym}.token_tiles={tt}"]
if "masked" in sys.argv[9:]:
    cmd.append(f"--config={sym}.tokens={B}")
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
