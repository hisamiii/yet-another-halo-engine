#!/usr/bin/env python3
"""Segmented GEMM: several gen_gemm_tile kernels over the same input in one dispatch (same values, bit-identical).

segcat(sym, segs) splices kernels generated for one workgroup size and token tile (the afrag tiles: 512 lanes, 512
tokens) into one kernel whose grid x is the sum of their row groups: workgroups [off_i, off_i + groups_i) run segment
i's body with wg_x = id - off_i (a uniform branch on the workgroup id). Each segment keeps its own weights, format,
row tile and epilogue, so a small GEMM (ssm_alpha / ssm_beta, attn_k / attn_v) rides along with the large one that
reads the same activations (attn_qkv, the q + gate kqg) instead of paying a launch with too few rows to fill the GPU.

segs: [(text, (m_tiles, k_blocks, token_tiles))]: each text a complete gen_gemm_tile kernel, its configs baked in as
constants. Bindings: segment 0's in order, then each later segment's own ones (all but SHARED) with suffix _s<i>.
Shared LDS: one alloca per buffer name at the largest size (one segment runs per workgroup)."""
import re
from gen_kres_persist import _defs, _rename

SHARED = ("input", "wstage", "ostage")


def _parts(text):
    """(header lines before kernel.def, sym, [binding names], body lines without the final kernel.return)."""
    L = text.rstrip("\n").split("\n")
    idef = next(i for i, l in enumerate(L) if l.startswith("kernel.def "))
    sym = re.search(r"@(\w+)\(\)", L[idef]).group(1)
    ilaunch = next(i for i, l in enumerate(L) if l.startswith("} launch("))
    binds = re.findall(r"%(\w+): buffer", L[ilaunch])
    assert L[-1] == "}" and L[-2].strip() == "kernel.return", "a kernel ending in kernel.return"
    body = L[ilaunch + 1:-2]
    assert not any(l.strip() == "kernel.return" for l in body), "one kernel.return, at the end"
    return L[:idef], sym, binds, body


def _items(hdr):
    """A kernel header's module-level items: target declarations and whole low.func.def blocks (comments dropped)."""
    out, cur = [], None
    for l in hdr:
        if cur is not None:
            cur.append(l)
            if l == "}":
                out.append("\n".join(cur)); cur = None
        elif l.startswith("low.func.def "):
            cur = [l]
        elif l.startswith("amdgpu.target"):
            out.append(l)
    return out


def segcat(sym, segs, wgs=512):
    out_binds, bodies, lds, groups = [], [], {}, []
    head = None
    for i, (text, (mt, kb, tt)) in enumerate(segs):
        hdr, ssym, binds, body = _parts(text)
        if head is None:
            head = [l for l in hdr if not l.startswith("config.decl")]
        else:   # a later segment's targets and Low helpers (gen_gemm_decode.lowdefs) the merged kernel lacks
            have = "\n".join(head)
            for item in _items(hdr):
                if item not in have:
                    head += item.split("\n")
        cfg = {"m_tiles": mt, "k_blocks": kb, "token_tiles": tt}
        body = [re.sub(r"config\.get @%s\.(\w+) : index" % ssym, lambda m: "index.constant %d : index" % cfg[m.group(1)], l)
                for l in body]
        assert not any("config.get" in l for l in body), "a config left: " + ssym
        bm = int(re.search(r"%wg_row = index.mul %wg_x, %c(\d+)", "\n".join(body)).group(1))
        assert (mt * 16) % bm == 0, (ssym, mt, bm)
        groups.append(mt * 16 // bm)
        # LDS: drop the alloca, its users read the shared one
        names = {}
        for l in body:
            m = re.match(r"\s*(%\w+) = buffer\.alloca<workgroup> align\(16\) (%\w+) : buffer", l)
            if m:
                nbytes = int(next(re.search(r"index.constant (\d+) : offset", x).group(1) for x in body
                                  if x.strip().startswith(m.group(2) + " = index.constant")))
                key = m.group(1)[1:]
                lds[key] = max(lds.get(key, 0), nbytes)
                names[m.group(1)] = "%seg_lds_" + key
        body = [l for l in body if not re.match(r"\s*%\w+ = buffer\.alloca<workgroup>", l)]
        body = [re.sub(r"%\w+", lambda m: names.get(m.group(0), m.group(0)), l) for l in body]
        # the segment's row group: the global workgroup id less its offset, clamped into [0, groups) (the branch
        # already guarantees it; the clamp lets the bounds analysis see the standalone kernel's range)
        iw = [j for j, l in enumerate(body) if l.strip() == "%wg_x = kernel.workgroup.id<x> : index"]
        assert len(iw) == 1, ssym
        body[iw[0]:iw[0] + 1] = [f"  %seg_lo = index.max %seg_wx, %seg_off{i} : index",
                                 f"  %seg_rel = index.sub %seg_lo, %seg_off{i} : index",
                                 f"  %seg_last = index.constant {groups[-1] - 1} : index",
                                 f"  %wg_x = index.min %seg_rel, %seg_last : index"]
        if i:
            own = [b for b in binds if b not in SHARED]
            body = _rename(body, _defs(body) | {"%" + b for b in own}, f"_s{i}")
            out_binds += [b + f"_s{i}" for b in own]
        else:
            out_binds += binds
        bodies.append(body)
    total = sum(groups)
    offs = [sum(groups[:i]) for i in range(len(groups))]
    tt = segs[0][1][2]
    assert all(s[1][2] == tt for s in segs), "one token tiling"
    L = list(head)
    L.append(f"kernel.def target(@yah_tile_w32) @{sym}() {{")
    L += ["  %unit = index.constant 1 : index", f"  %wgs = index.constant {wgs} : index",
          f"  %groups = index.constant {total} : index", f"  %token_tiles = index.constant {tt} : index",
          "  kernel.launch.config workgroups(%groups, %token_tiles, %unit) workgroup_size(%wgs, %unit, %unit) : index"]
    L.append("} launch(" + ", ".join(f"%{b}: buffer" for b in out_binds) + ") {")
    for k, n in sorted(lds.items()):
        L += [f"  %seg_lds_{k}_bytes = index.constant {n} : offset",
              f"  %seg_lds_{k} = buffer.alloca<workgroup> align(16) %seg_lds_{k}_bytes : buffer"]
    L.append("  %seg_wx = kernel.workgroup.id<x> : index")
    for i, o in enumerate(offs):
        L.append(f"  %seg_off{i} = index.constant {o} : index")

    def chain(i):
        """Segments i.. as nested branches: the last first (each else holds the earlier ones)."""
        if i == 0:
            return list(bodies[0])
        return ([f"  %seg_in{i} = index.cmp uge, %seg_wx, %seg_off{i} : index", f"  scf.if %seg_in{i} {{"]
                + bodies[i] + ["  } else {"] + chain(i - 1) + ["  }"])
    L += chain(len(bodies) - 1)
    L += ["  kernel.return", "}"]
    return "\n".join(L) + "\n", total
