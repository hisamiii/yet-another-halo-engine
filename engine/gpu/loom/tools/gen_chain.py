#!/usr/bin/env python3
"""Flag-chained decode dispatch: a kernel waits for its producer's counter instead of a queue barrier.

chain(src) adds three i32 launch arguments and one buffer to a 1-D Loom kernel:
  %ch_cnt   i32 [CHAIN_SLOTS] counters (zero at allocation, only ever incremented)
  %ch_ws    the slot to wait on, %ch_need its target value (absolute; the host keeps a shadow of every slot)
  %ch_ss    the slot this kernel signals: lane 0 of every wave adds 1 once its stores are visible (device scope),
            at the end or at a top-level "// chain.signal" marker (stores after it are not ordered for the consumer)
Wave 0 polls cnt[ws] until (cnt - need) >= 0 as i32 (wrap-safe), the other waves wait at the barrier, then the
workgroup acquires (device scope). The host dispatches chained kernels without the ordering barrier; the queue
launches workgroups in order, so a waiting workgroup never blocks its producer.
A lost signal costs one bound of polls per workgroup (gen_rot.FLAG_POLLS, ~0.2 s), not a hang.
"""
import re

import gen_rot

CHAIN_SLOTS = 1024


def chain(src):
    lines = src.split("\n")
    li = [i for i, l in enumerate(lines) if l.startswith("} launch(")]
    ri = [i for i, l in enumerate(lines) if l.strip() == "kernel.return"]
    assert len(li) == 1 and len(ri) == 1 and ri[0] > li[0], "chain: one launch signature and one kernel.return"
    assert all(re.search(r"workgroup_size\(\S+, %\w+, %\w+\)", l) for l in lines if "workgroup_size(" in l)
    sig = lines[li[0]]
    assert sig.endswith(") {")
    lines[li[0]] = sig[:-3] + ", %ch_cnt: buffer, %ch_ws: i32, %ch_need: i32, %ch_ss: i32) {"
    # gen_gemv peel: the wait goes at its "// chain.wait" markers (one per band branch), after the work that does not
    # read the producer
    wi = [i for i, l in enumerate(lines) if l.strip() == "// chain.wait"]
    assert all(li[0] < w < ri[0] for w in wi)
    CV = f"view<{CHAIN_SLOTS}xi32>"
    pro = [
        "  %ch_base = index.constant 0 : offset",
        f"  %ch_cv = buffer.view %ch_cnt[%ch_base] : buffer -> {CV}",
        "  %ch_tid = kernel.workitem.id<x> : index",
        "  %ch_c0 = index.constant 0 : index",
        "  %ch_c32 = index.constant 32 : index",
        "  %ch_zero = scalar.constant 0 : i32",
        "  %ch_one = scalar.constant 1 : i32",
        f"  %ch_limit = scalar.constant {gen_rot.FLAG_POLLS} : i32",
        f"  %ch_mask = scalar.constant {CHAIN_SLOTS - 1} : i32",
        "  %ch_wsm = scalar.andi %ch_ws, %ch_mask : i32",
        "  %ch_ssm = scalar.andi %ch_ss, %ch_mask : i32",
        "  %ch_wsi = index.cast %ch_wsm : i32 to index",
        "  %ch_ssi = index.cast %ch_ssm : i32 to index",
    ]
    wait = [
        "  %ch_w0 = index.cmp ult, %ch_tid, %ch_c32 : index",
        "  scf.if %ch_w0 {",
        "    %ch_n, %ch_v = scf.while(%ch_i = %ch_zero : i32, %ch_p = %ch_zero : i32) -> (i32, i32) {",
        f"      %ch_x = view.atomic.load %ch_cv[%ch_wsi] {{ordering = acquire, scope = device}} : {CV} -> i32",
        "      %ch_d = scalar.subi %ch_x, %ch_need : i32",
        "      %ch_early = scalar.cmpi slt, %ch_d, %ch_zero : i32",
        "      %ch_lt = scalar.cmpi ult, %ch_i, %ch_limit : i32",
        "      %ch_go = scalar.andi %ch_early, %ch_lt : i1",
        "      scf.condition %ch_go, %ch_i, %ch_x : i1, i32, i32",
        "    } do(%ch_j: i32, %ch_w: i32) {",
        "      %ch_j1 = scalar.addi %ch_j, %ch_one : i32",
        "      scf.yield %ch_j1, %ch_w : i32, i32",
        "    }",
        "  }",
        "  kernel.barrier<global> scope(workgroup) ordering(acquire)",
    ]
    ind = lines[ri[0]][: len(lines[ri[0]]) - len(lines[ri[0]].lstrip())]
    epi = [
        "%ch_l = index.rem %ch_tid, %ch_c32 : index",
        "%ch_l0 = index.cmp eq, %ch_l, %ch_c0 : index",
        "scf.if %ch_l0 {",
        f"  view.atomic.reduce<addi> %ch_one, %ch_cv[%ch_ssi] {{ordering = release, scope = device}} : i32, {CV}",
        "}",
    ]
    body = lines[li[0] + 1: ri[0]]
    local = (set(re.findall(r"%ch_\w+", "\n".join(wait))) - set(re.findall(r"(%ch_\w+) =", "\n".join(pro)))
             - {"%ch_cnt", "%ch_ws", "%ch_need", "%ch_ss"})
    for n, w in reversed(list(enumerate(wi))):   # each marker's wait with its own local names
        k = w - li[0] - 1
        mine = [re.sub(r"%ch_\w+", lambda m: m.group(0).replace("%ch_", f"%ch{n}_") if m.group(0) in local else m.group(0), l)
                for l in wait] if len(wi) > 1 else wait
        body = body[:k] + mine + body[k + 1:]
    if not wi:
        body = wait + body
    si = [i for i, l in enumerate(body) if l.strip() == "// chain.signal"]
    assert len(si) <= 1
    if si:   # the signal before the stores no consumer of this dispatch reads
        sind = body[si[0]][: len(body[si[0]]) - len(body[si[0]].lstrip())]
        body = body[:si[0]] + [sind + l for l in epi] + body[si[0] + 1:]
    out = lines[: li[0] + 1] + pro + body + ([] if si else [ind + l for l in epi]) + lines[ri[0]:]
    return "\n".join(out)
