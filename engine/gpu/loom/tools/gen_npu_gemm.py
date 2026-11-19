#!/usr/bin/env python3
"""Emit the NPU cascade GEMM (XDNA2 array program + core leaves, Loom low asm) for loom-compile.

C[M x N] = A[M x K] . W[N x K]^T in bfp16ebs8 with f32 accumulation, N = TN = 80 per NPU column (cols <= 8).
Each column is a cascade of 4 compute rows: the head (top row) and two mids compute K-slice partials of every 16 x 16
sub-tile and pass them down the accumulator cascade; the tail (bottom row) adds its slice and the running C.
K is split per pass over the rows (ks[r] 8-wide k-blocks each, ks = [head, mid, mid, tail]); passes repeat until K.
Operand streams are the layouts of gen_bfp16_encode.py (act: [slice][M block][pass][slab], wgt: [col][slice][pass][slab]).
C: per column [M block][4 segments][sub-tile][chain][8 x 8 bf16] (see npu_gemm_check.unpack_c); the tail accumulates f32 and
packs each segment to bf16 in place on its last pass, and the C ring sends the packed half (LOOM_EXP_LS_SEND_PITCH=2).
The kernel is bound by the operand streams into the compute tiles (36 * (1 / MP + 1 / NP) bytes per MMA over two
S2MM streams of ~7.7 B/cycle): NP = 5 needs 20% less activation stream than 4; a column's whole weight panel must fit
its 512 KB memory tile, which caps NP at 5 for K = 5120.

Dataflow: weights stage once per column in its memory tile and replay per M block; each row's activations are one
stream, staged in a memory tile and multicast along the row. Activations arrive one 16-row slab per record into a
leaf-synchronized ring (the leaf acquires and releases slabs, constrain.leaf_sync); the tail holds C as 4
leaf-synchronized segments, acquired on a segment's first pass and released after its last.
A 5-pass call replays its passes in groups (gen_bfp16_encode.GROUPS: every M block over passes 0-1, then over 2-4), each
group a C partial ([col][group][M block]..., summed by the GPU unpack): a group's memory-tile slots refill with the next
call's weights while the other group computes, so calls stream back to back (Loom LOOM_EXP_PANEL_GROUPS /
LOOM_EXP_PANEL_STREAM: counting locks, a looping fill, invocations [setup | even push | odd push | waits | lead push]
and fills paced a few columns at a time so they do not starve the activation streams of DRAM; the lead push, a job's
first call, fills unpaced; LoomNpu submits the pushes one call ahead of the waits).
Needs HRX patch 0013; compile with LOOM_ENV. Config.gate > 0 adds the NPU-side gate (GATE below; HRX patch 0016).
"""
import contextlib
import dataclasses
import importlib
import json
import os
import re
import struct
import sys

import gen_bfp16_encode as GE

MBMS4 = "(reg<aie2p.mbms>, reg<aie2p.mbms>, reg<aie2p.mbms>, reg<aie2p.mbms>) -> reg<aie2p.mbms x4>"
ORDER_C = tuple(int(x) for x in os.environ.get("GN_ORDC", "0,2,1,3").split(","))   # MMA order of the 4 chains inside a k step (a-row, w-col: c >> 1, c & 1)
LAGU = 9                 # pops issue 9 MMAs ahead of their first use (the pop -> operand latency)
WB = 9                   # pops of the next sub-tile that ride in the cascade-write block (head, mids)
MP, NP = int(os.environ.get("GN_QMP", "4")), 5   # 16-row slabs per M block / per NPU column (GN_QMP: Q' experiments)
TM, TN = 16 * MP, 16 * NP
KS = (35, 35, 35, 23)    # production k-blocks per pass and K-slice row: K = 1024 per pass
# loom-compile environment for the NPU image
# GN_QCYC=1 (diag, the plain tail's convert): each last-pass segment's first word = the core cycles since the first
# firing's start (npu_gemm_check prints the per-M-block steps; the outputs are not checked)
QCYC = os.environ.get("GN_QCYC") == "1"
# GN_QFCYC=1 (diag, with GN_QCYC): the Q' fill's cycles (word 2: the whole fill; word 1: its egress phases) into every
# C segment's next words (the last call's fill; %qfc words 0 / 8: phase starts, 4: egress sum, 12: fill)
QFCYC = QCYC and os.environ.get("GN_QFCYC") == "1"
# GN_QCYCABS=1 (diag, with GN_QCYC, not GN_QFCYC): word 1 of each stamped segment = the core's first firing's absolute
# cycle (low word), so the call's span = max(word 0 + word 1) - min(word 1) over the cores
QCYCABS = QCYC and not QFCYC and os.environ.get("GN_QCYCABS") == "1"
# GN_QBANK=1 (experimental, the Q' tail with mu 1, ks <= 28): the bank plan of a one-core Q' tile, 16 KB banks
# 0 W slot 0 | 1 C rows 0, 2 + A slot 0 | 2 C rows 1, 3 + A slot 1 | 3 W slot 1: the DMA refills only a bank the core
# is not reading. The C pins reserve the sent 2 KB of each 4 KB slot; each bank's A slot comes after its C slots, so
# the allocator's cursor passes the slots' f32 halves. The check sets LOOM_EXP_RING_AT to QBANK_RING_AT.
QBANK_RING_AT = "2048@16384:32768:20480:36864,16128@0:49152,4032@24576:40960"
# GN_QBANK=2: C rows 0, 2 in bank 2 and 1, 3 in bank 1 (each row's C in the bank of the A slot the DMA refills)
QBANK2_RING_AT = "2048@32768:16384:36864:20480,16128@0:49152,4032@24576:40960"
# GN_QBANK=3 (with GN_QPASS): a C row's sub-tiles alternate banks, so a window stores one bank and loads the other:
# even sub-tiles in bank 1 (rows 2 KB apart: the C ring's send slots), odd ones in bank 2 (an 8 KB private storage,
# placed in bank 2 after A slot 1, which moves to the bank's start); the convert gathers both into the send slot.
QBANK3_RING_AT = "2048@16384:18432:20480:22528,16128@0:49152,4032@24576:32768"
# GN_QA3=1 (with GN_QBANK=3): three A slots 4 KB apart from 24576 (two in bank 1 beside C even, one at bank 2's
# start): the DMA runs a slot ahead through upstream stalls. A firing's 64 rows advance the ring one slot (64 mod 3):
# the slot a firing starts at lives in the pass counter's storage (offset 12, zero from the array setup). It keeps
# running across calls, as the DMA's slot position does. (Four slots do not fit beside C and the stack.)
QA3_RING_AT = "2048@16384:18432:20480:22528,16128@0:49152,4032@24576:28672:32768"
# GN_QPASS=1 (experimental, with GN_QROW=1 and GN_QBANK=1): one firing per M block. The leaf owns the W ring too and
# its row loop runs every (pass, row); a pass boundary is one more rotated row end (W slot released and the next
# acquired in its window); the C slots are acquired and zeroed once per M block and converted after the loop.


def qlock_on():   # (layout Q') LOOM_EXP_LOCK_READ_SEP on: reads may issue 2 cycles after an acquire
    return qpass_on() and os.environ.get("GN_QLOCK") == "1"


def notick():
    """GN_QNOTICK=1: the waiter's poll stream runs free (no tick channel, no constrain.request): its shim DMA reads
    ahead as far as the stream FIFOs hold, so a poll is staler by that depth (gate values only grow: a stale read only
    waits longer), and one southbound lane less leaves the waiter's tile (the K-half split routes gated)."""
    return os.environ.get("GN_QNOTICK") == "1"


def pshare():
    """GN_QPSHARE=1 (Loom LOOM_EXP_PACKET_SHARE, HRX pin; the K-half split takes every compute -> memory tile lane):
    the waiter's tick and the relay's go / done borrow the lane of their column's row 0 decoder egress past row 0, as
    packets (each record a header + 4 words, the last with TLAST). The relay sends done at its last firing of the job
    (after every fill of the job), so done never waits at the shim inside the shared lane while a fill needs it."""
    on = os.environ.get("GN_QPSHARE") == "1"
    assert not (on and notick()), "GN_QPSHARE keeps the tick"
    return on


def gate_tick(e, v, tag):
    """One tick (a request for one flag record; GN_QPSHARE: a packet)."""
    gate_record(e, (v, v, v, v), f"%gtid{tag}")


def qpass_on():
    on = os.environ.get("GN_QPASS") == "1"
    assert not on or (os.environ.get("GN_QROW") == "1" and os.environ.get("GN_QBANK") in ("1", "3")), "GN_QPASS: QROW, QBANK 1/3"
    return on


def qa3_on():
    on = os.environ.get("GN_QA3") == "1"
    assert not on or (qpass_on() and os.environ.get("GN_QBANK") == "3"), "GN_QA3 needs GN_QPASS, GN_QBANK=3"
    return on


# GN_QCB1=1 (with GN_QA3): all of C in Loom bank 1 (the C ring and the odd sub-tiles' area, in its own storage block),
# all A slots in bank 2 beside the small private block: the core's C and A accesses never share a bank (the A DMA
# then always writes the bank the core reads A from)
QCB1_RING_AT = "2048@16384:18432:20480:22528,16128@0:49152,4032@32768:36864:40960"


def qcb1_on():
    on = os.environ.get("GN_QCB1") == "1"
    assert not on or qa3_on(), "GN_QCB1 needs GN_QA3"
    return on


# GN_QA4=1 (with GN_QBANK=3): four A slots alternating Loom banks 1 and 2 (slot order 24576, 32768, 28672, 36864):
# the A DMA, up to three slots ahead, fills a slot in the other bank than the one the core reads (as the W ring
# does), and 4 divides a firing's 64 rows (every firing starts at slot 0). Room: the odd sub-tiles' area in its own
# block at bank 2's top, the never-run zero-fill block and its 1 KB buffer dropped (private: 128 B beside a W slot).
QA4_RING_AT = "2048@16384:18432:20480:22528,16128@0:49152,4032@24576:32768:28672:36864"


def qa4_on():
    on = os.environ.get("GN_QA4") == "1"
    assert not on or (qpass_on() and os.environ.get("GN_QBANK") == "3" and not qa3_on()), \
        "GN_QA4 needs GN_QPASS, GN_QBANK=3, not GN_QA3"
    return on


# GN_QPEEL=1 (with GN_QA4): the last pass peeled: the (pass, row) loop runs MP * (passes - 1) iterations, then a copy
# of its body runs the last pass's MP rows, storing each sub-tile's sums straight to bf16 in its row's C slot (sub-tile
# t at 512 t, as the post-loop convert packed them) and releasing the slot at the row's end: no post-loop convert, and
# the output DMA drains each slot while the next rows run. In place: a sub-tile's bf16 lands in the f32 of sub-tile
# t & 2 (even area), whose loads precede the store through the MMA / pop chain.
def qpeel_on():
    on = os.environ.get("GN_QPEEL") == "1"
    assert not on or qa4_on(), "GN_QPEEL needs GN_QA4"
    return on


# GN_QUNR=1: the main (pass, row) loop's body holds two rows (the second copy takes the first's back-edge values), so
# the loop-carried operands meet their header registers every second row only. Without GN_QPEEL the code stays the
# peel's size (the unrolled copy instead of the peel's; the post-loop convert stores C).
def qunr_on():
    return os.environ.get("GN_QUNR") == "1"


# Layout Q' with fuse 2 (gen_npu_q sets QF["cfg"]): every Q' core decodes its column's panel at a call's first firing,
# slab-major (unit order [slab][1024-K pass]), so the memory tile holds [slab][K] and the W channel reads passes of any
# ks. QF["cfg"]: the decode side's Config (ks (128,), passes K / 1024); QF["wslot"]: the Q' W slot bytes.
QF = {"cfg": None, "wslot": 0}
QF_PORTS = (3, 4)   # a Q' filler's egress sender and raw receiver (A 0, W 1, C 2)
QF_SLAB = 4         # %fzt words (the tail's parked C ring is word 0): the slab being decoded, the egress record,
QF_REC = 8          # then the slab outputs' addresses (words 12 .. 24)
QF_OUT = 12
QDEC_SPORT, QDEC_RPORT = 3, 5   # (NPU_QDEC, two decoder rows) the second's slab sender, the first's receiver
QF_OUTS = (19584, 49152, 53760, None)    # slab outputs (4608 B each): C rows after SP, W slot 1 (two), private
# (P1K fill, the 64x80 map: W slots 0 / 53760, scratch 12288, A 20480 / 32768, C rows 37376..49664) the decoder's areas,
# all idle during the fill (the A slots are not: A streams in meanwhile): IN W slot 0, SP in the scratch's tail (bank 1),
# SP2 in the C rows (bank 2), the 5 slab outputs, the egress ring (2 x 576 B) in the free run past the small pin
QF_P1K = dict(sp=16896, sp2=37376, outs=(12288, 25088, 44608, 53760, 58368), egr=30784)
# (P1K fill, GN_QFDIRECT=1, the default) the decoder writes each slab's 4608 B straight into the egress ring, 4 records
# of 2304 B (2 a slab) over C slots 1-3 and their tails (40448..49664, idle in the fill); SP2 in W slot 1
_QFRAW2 = os.environ.get("GN_QFRAW2", "1") == "1"
# raw: intake slots, 9216 B apart (GN_QFRAW2=1, the default: 2, slot 1 over the scratch %qods, time-disjoint). SP then
# leaves bank 2 (slot 1 ends at 18432): 26112, bank 3 past Loom's stack (at the bank cursor, 25088: SP 25088 gave wrong
# output); gated one-call fill 288K -> 264K. 49664 / 60992 cost +50K / +32K (banks shared with SP2).
QF_P1KD = {}


def qf_p1kd_layout():
    """QF_P1KD from the environment (at import, and again after gen_npu_q.defaults sets the P1K knobs). GN_QFKH: the
    K-half split's ring, one record a raw record, 4 x 1152 B off the C ring."""
    kh = os.environ.get("GN_QFKH") == "1"
    e2 = kh and os.environ.get("GN_QFEGR2") == "1"   # (GN_QFEGR2: the egress ring 4 BDs of a slab, 2 records each)
    QF_P1KD.update(sp=int(os.environ.get("GN_QFSP", "26112" if _QFRAW2 else "16896")), sp2=53760,
                   egr=int(os.environ.get("GN_QFEGR", "40448" if e2 or not kh else "49152")), rec=1152 if kh else 2304, slots=8 if e2 else 4,
                   eb=2304 if kh and e2 else 1152 if kh else 2304, raw=2 if _QFRAW2 else 1)


def qfegr2_on():
    """GN_QFEGR2=1 (GN_QF4R): an egress row's ring holds 4 slabs (8 records, 40448..49664, C slots 1-3), one BD and
    one lock step a slab; the address math keeps 1152 B records (8 slots)."""
    return QF_P1KD["eb"] != QF_P1KD["rec"]


qf_p1kd_layout()


def qfdirect_on():
    return qp1k_on() and QF["cfg"] is not None and os.environ.get("GN_QFDIRECT", "1") == "1"


def qfkh_on():
    """GN_QFKH=1 (P1K direct fill): the K-half split: rows 0 and 1 decode (tailm0d, tailm), each one K half of every
    super-block (record h, h = row) into its own panel (rows h, h + 2) through its own egress lane."""
    return qfdirect_on() and os.environ.get("GN_QFKH") == "1"


QF_HALF = [None]   # (GN_QFKH) the leaf's K half
QF4R = [None]      # (GN_QF4R) the leaf's role: dict(par: its super-block steps' parity, help: hands its records to its
                   # egress neighbour) or None
QF4R_NB = 8        # (GN_QF4R) an egress row's receiver of its helper's records (the helper sends on its egress port)
def qf4r_hslots():
    """(GN_QF4R) the helper ring's records over its C slots 1-3 (idle in the fill): 8, or 4 in an FFN gate / up image
    (8 would reach its silu table at 45568); GN_QF4RHS overrides."""
    return int(os.environ.get("GN_QF4RHS") or (4 if QF.get("qsw") else 8))


QF4R_HAT = 40448


def qf4r_on():
    """GN_QF4R=1 (with GN_QFKH): the 4-row decode. The raw intake splits by record parity (Loom
    LOOM_EXP_INTAKE_SPLIT, HRX pin): W multicast 0 (rows 0, 2) gets the even records, 1 (rows 1, 3) the odd ones, as
    intake siblings (one memory-tile drain feeds both rows of a multicast); each row decodes one K half of each of its
    records (rows 0, 1 half 1; rows 2, 3 half 0). Rows 1 / 2 egress halves 1 / 0 in panel order, their own records'
    halves between the copies of their helper's (rows 0 / 3: a ring in the helper, read through the window)."""
    return qfkh_on() and os.environ.get("GN_QF4R") == "1"


QF_LANE2_PORT = 9   # (GN_QFLANE2) an egress row's second egress lane (its C channel's DMA, idle in the fill)


def qfl2guard():
    """(GN_QFLANE2, GN_QFL2GUARD=1) in an egress row: its own slab acquires the second lane's slot of the helper copy
    that follows it (the copy skips that acquire), so the own decode starts only once the last copy was sent."""
    return bool(QF4R[0]) and QF4R[0]["l2"] and not QF4R[0]["help"] and os.environ.get("GN_QFL2GUARD") == "1"


def qflane2_side(side):
    """Lane 2 in a column of this side ("w": a pair's gate (even) columns, "e": its up (odd) columns; images without
    pairs are all "w"): GN_QFLANE2COLS=w|e (diag) limits it to one side."""
    m = os.environ.get("GN_QFLANE2COLS", "")
    return qflane2_on() and (not m or side == m)


def qflane2_on():
    """GN_QFLANE2=1 (GN_QF4RSWAP, the FFN gate / up's 4-record egress ring): the fill's second egress lane. An
    egress row's ring holds its own slab in slots 0-1 and its helper's copied slab in 2-3; slots 2-3 become a ring of
    their own on port QF_LANE2_PORT (pinned right after the first), sent on the row's C channel's DMA (Loom: a second
    constrain.fill of the panel, HRX 0098), so the two lanes egress a K half's panel in parallel. Not beside a Q4_K
    up column (its GF decoder: panel half 0 comes out wrong, 11-19 log; GN_QFLANE2=2, diag: there too)."""
    cu = QF.get("cfgu")
    v = os.environ.get("GN_QFLANE2")
    return (qf4r_swap() and v in ("1", "2") and not qfegr2_on() and qf4r_hslots() == 4
            and (v == "2" or not (cu is not None and cu.fmt == "Q4_K")))


def qf4r_swap():
    """GN_QF4RSWAP=1 (with GN_QF4R): rows 0 / 1 swap the fill roles (row 0 egresses half 1, row 1 helps): an egress
    row's first record in panel order is then its own in both pairs (row 1 waited ~3K a call for its helper's)."""
    return qf4r_on() and os.environ.get("GN_QF4RSWAP") == "1"


def qfhalf():
    """GN_QFHALF=h (P1K direct fill, diag): the decoder takes record h of each super-block only (K half h; None:
    both); GN_QFKH: the leaf's half."""
    if QF_HALF[0] is not None:
        return QF_HALF[0]
    v = os.environ.get("GN_QFHALF")
    return int(v) if v and qfdirect_on() else None


def qfraw4_on():
    """GN_QFRAW4=1 (P1K direct fill): a raw record holds a slab's 4 super-blocks a row (one 576 B DRAM read a row,
    25 records a call, not 100 of 144 B reads); the decoder takes each record's 4 passes in turn."""
    return qfdirect_on() and os.environ.get("GN_QFRAW4") == "1"
QF_FW = 144         # egress record words: a 16-k-block piece is 4 records
QF_A13 = {32768: 33920, 36864: 38016}   # A slots 1, 3 (bank 2) past the egress ring at its start: qf_ring_at


def qf_ring_at(pins):
    """(Q' fuse 2) LOOM_EXP_RING_AT for pins: rings are placed in channel order C, egress + raw, W, A (W's memory-tile
    panel before the A stage: one memory tile holds both at 8 columns) and a pin is skipped once its bank holds a later
    allocation, so the egress ring takes bank 2's start (32768) and A slot 1 moves past it (QF_A13; both bank-2 slots, the leaf steps slots symmetrically)."""
    fb = 4 * fuse_record(QF["cfg"], (0,))[0]
    if qfdirect_on():   # (P1K fill, direct) the 4-record egress ring over C slots 1-3
        m = QF_P1KD
        assert fb == m["rec"]
        hat = P1K_GRID_RING if QF.get("grid") else QF4R_HAT
        h4 = (f",{fb}@" + ":".join(str(hat + i * fb) for i in range(qf4r_hslots()))) if qf4r_on() else ""
        eb = m["eb"]
        # (FFN gate / up: the K-half ring over C slots 1-3 too, idle in the fill; the silu table takes 45568 .. 53760)
        egr = 40448 if QF.get("qsw") and "GN_QFEGR" not in os.environ else m["egr"]
        if QF.get("grid") and not QF.get("qsw"):   # (grid formats at NP 4: P1K_GRID_*; gate / up: qgmap_on)
            assert NP == 4 and not QF.get("qsw") and egr == 40448, "Q' grid formats: NP 4 site images"
            egr = P1K_GRID_RING
            assert egr + m["slots"] * fb <= P1K_GRID_AT and hat + qf4r_hslots() * fb <= P1K_GRID_AT
        assert not QF.get("qsw") or egr + m["slots"] * fb <= 45568
        assert not (QF.get("qsw") and qf4r_on()) or QF4R_HAT + qf4r_hslots() * fb <= 45568
        if qflane2_on():   # (GN_QFLANE2) slots 0-1 the first lane's ring, 2-3 the second's (one 2-record slot)
            assert eb == fb and m["slots"] == 4
            return pins + f",{eb}@{egr}:{egr + eb},{2 * eb}@{egr + 2 * eb}" + h4
        return pins + f",{eb}@" + ":".join(str(egr + i * eb) for i in range(m["slots"] * fb // eb)) + h4
    if qp1k_on():   # (P1K fill) the egress ring in QF_P1K's free run; the A pins stay
        return pins + f",{fb}@{QF_P1K['egr']}:{QF_P1K['egr'] + fb}"
    g = pins.split(",")
    a = [QF_A13.get(int(v), int(v)) for v in g[2].split("@")[1].split(":")]
    g[2] = g[2].split("@")[0] + "@" + ":".join(map(str, a))
    assert 32768 + 2 * fb <= QF_A13[32768], fb
    return ",".join(g) + f",{fb}@32768:{32768 + fb}"


def acap_tail(cfg):
    return len(qa4_slots()) if qa4_on() else 3 if qa3_on() else cfg.acap_t


def qpass_cslots():
    """The C row slots' byte addresses (QBANK_RING_AT's C pins)."""
    return [int(v) for v in QBANK_RING_AT.split(",")[0].split("@")[1].split(":")]


def qpins(default=None):
    """GN_QPINS (default if unset). "p1k": the P1K map's pins for the current NP: the MP C rows' bf16 slots (512 NP B)
    at the even sub-tile area's row stride (q3_strides) from 37376, the W slots (NP x 2304 B: a member's half record)
    at 0 / 53760, the A slots at 20480 / 32768, the 64 B word at 30720."""
    v = os.environ.get("GN_QPINS") or default
    if v != "p1k":
        return v
    cs = ":".join(str(37376 + q3_strides()[0] * r) for r in range(4))
    return f"{SW_REC if QF.get('qsw') else 512 * NP}@{cs},{2304 * NP}@0:53760,4608@20480:32768,64@{p1k_tok()}"


# (P1K at NP 4, grid formats) the decoder's tables from 49152 (4 KB pages: the gathers' address high bytes) to W slot 1
# (53760); the egress and helper rings move down from 40448 to 39424 (C slot 1: slots 1-3 are idle in the fill) to
# make room. A table symbol past 53760 goes to the first spare run it fits aligned: 63488 .. 65536 (after the 320 B
# storage at 62976) or 28672 .. 30656 (after the scratch pad, before its guard: P1K4_*).
# NP 5 has no room: its 25088 .. 29312 holds Loom's stack and the decoder's scratch pad, the rest of the map is full.
P1K_GRID_AT, P1K_GRID_END, P1K_GRID_RING = 49152, 53760, 39424
# (P1K at NP 4, every format) the decoder's scratch pad at 25088 (A slot 0's end), a 64 B rodata guard (@qfgd) at
# 30656: its pin grows bank 1's cursor past the pad, so Loom's stack (placed at the cursor) goes after it; the fill
# takes the pad's address from the guard's
P1K4_SP, P1K4_GUARD = 25088, 30656
P1K_GRID_SPARE = ((63488, 65536), (28672, P1K4_GUARD))
# (P1K at NP 4, FFN gate / up with a grid format: qgmap_on) the silu table in two pieces around the grid page:
# 56 blocks at 45568 .. 49152 (the even C rows' end to the page), 65 at 28288 .. 32448 (the decoder pad's end, a grid
# decoder's 3200 B); the pad's guard and the MB2 token word move past it (32448, 32704); the SL_K vectors (@silu_k),
# the grid tables' small symbols and Loom's stack take the grid page's tail and 63488 .. 65536 (gen_npu_q.qgmap_pins)
QG_SILU = ((56, 45568), (65, 28288))
QG_GUARD, QG_TOK = 32448, 32704
QG_PRIVATE = 62976   # (W slot 1's end; Loom's private storage, <= 512 B, pinned: gen_npu_q.loom_env)
QG_SPARE = ((49152, 53760), (63488, 65536))


def qgmap_on():
    """The P1K grid map of an FFN gate / up image (QF grid and qsw: gen_npu_q.grid_flags)."""
    return bool(QF.get("grid") and QF.get("qsw"))


def p1k4_guard():
    return QG_GUARD if qgmap_on() else P1K4_GUARD


def p1k_tok():
    return QG_TOK if qgmap_on() else 30720


def p1k4_on():
    """The P1K direct fill at NP 4 (gen_npu_q pins @qfgd): P1K4_SP / P1K4_GUARD."""
    return qfdirect_on() and NP == 4


def q3_strides():
    """(GN_QBANK=3) the even / odd sub-tile areas' row strides: NP odd puts one more sub-tile in the even area."""
    return 1024 * ((NP + 1) // 2), 1024 * (NP // 2)


# GN_QALT=1 (with GN_QPEEL, MP * NP odd): the C sums' two areas by strict alternation over the pass: sub-tile
# (r, t) in area E if (NP r + t) is even, else O (E ceil(MP NP / 2) KB, O the rest), each chain +1 KB a visit, so
# every window stores one area and loads the other (one same-area step a pass: the pass end, where the chains jump
# back to the areas' starts). E and O are scratch storages beside the A slots; the bf16 rows go to a 2-slot C ring
# (GN_QPINS' first group, e.g. in the W slots' banks' spare), acquired and released per row by the peel.
def qalt_on():
    on = os.environ.get("GN_QALT") in ("1", "2")
    assert not on or (qpeel_on() and MP * NP % 2 == 1), "GN_QALT needs GN_QPEEL and MP * NP odd"
    return on


# GN_QALT=2: the C ring's MP send slots pinned inside the areas (GN_QPINS' first group): row r's slot starts its
# 3-sub-tile portion (r even: E + NP r / 2 KB, else O + (NP r - 1) / 2 KB); no E / O storages (the ring pins and the
# A pins after them keep Loom's bump cursor past the areas' gaps); the bf16 row packs in place as before.
def qaltp_on():
    return os.environ.get("GN_QALT") == "2"


def qaltp_eo():
    """(GN_QALT=2) the E and O areas' byte addresses, from the C pins (checked against every row's slot)."""
    v = [int(x) for x in qpins().split(",")[0].split("@")[1].split(":")]
    E, O = v[0], v[1] - 1024 * ((NP - 1) // 2)
    assert len(v) == MP and all(x == (E + 1024 * (NP * r // 2) if r % 2 == 0 else O + 1024 * ((NP * r - 1) // 2))
                                for r, x in enumerate(v)), v
    return E, O


def qalt_sizes():
    """(GN_QALT) the E and O areas' bytes."""
    return 1024 * ((MP * NP + 1) // 2), 1024 * (MP * NP // 2)


def qalt_cslots():
    """(GN_QALT) the C ring's 2 slot addresses (GN_QPINS' first group)."""
    v = [int(x) for x in qpins().split(",")[0].split("@")[1].split(":")]
    assert len(v) == 2, v
    return v


def qa4_carry(cfg):
    """(GN_QA4) a firing's rows not a multiple of 4: the A slot runs on across firings (base %qai in private)."""
    return qa4_on() and MP * cfg.passes % len(qa4_slots()) != 0


def qa4_slots():
    """(GN_QA4) the 4 A slots' byte addresses: GN_QPINS' third group if set, else QA4_RING_AT's."""
    pins = qpins(QA4_RING_AT)
    a = [int(v) for v in pins.split(",")[2].split("@")[1].split(":")]
    return [QF_A13.get(v, v) for v in a] if QF["cfg"] is not None and not qp1k_on() else a   # (P1K: the A pins stay)


def div_mp(src, dst, x):
    """Lines: dst = src // MP (src < 4096; MP = 4: a shift)."""
    if MP == 4:   # (the names of the original lines: byte-identical 4 x 4 images)
        return ["  %qkm2 = mova.i32 -2", f"  {dst} = lshl {src}, %qkm2"]
    m = -(-65536 // MP)
    assert all(v * m >> 16 == v // MP for v in range(4096))
    return [f"  %{x}m = mov.i32 {m}", f"  %{x}p = mul {src}, %{x}m", f"  %{x}k = mova.i32 -16", f"  {dst} = lshl %{x}p, %{x}k"]


# GN_QW4=1 (with GN_QPASS, passes % 4 == 0): a 4-slot W ring (slot = pass & 3; the DMA up to 3 passes ahead), the
# step to pass pairs. Slot k = slot 0 + (k & 1) d1 + (k >> 1) d2 (GN_QPINS' W group in ring order: s0, s0 + d1,
# s0 + d2, s0 + d1 + d2); every firing starts at slot 0.
def qw4_on():
    return os.environ.get("GN_QW4") == "1"


def qw4_deltas():
    """(GN_QW4) d1, d2 from GN_QPINS' W group."""
    w = [int(v) for v in qpins().split(",")[1].split("@")[1].split(":")]
    assert len(w) == 4 and w[3] - w[0] == (w[1] - w[0]) + (w[2] - w[0]), w
    return w[1] - w[0], w[2] - w[0]


def qclazy_on():
    """GN_QCLAZY=1 (GN_QZACC): the firing acquires C slot 0 at entry, slot r at pass 0's row r - 1 end, so the DMA
    drains the last firing's segments (and a pair relay's copies) while the rows before run."""
    return os.environ.get("GN_QCLAZY") == "1" and qzacc_on()


def crelp_on():
    """NPU_QCREL=2 (GN_QP1, an even number of rows): each P1 pair's row 1 sends its C to row 0 (port 5), which copies
    each firing's MP segments after its own: one C stream per pair (2 a column at 4 rows: the shim's 16)."""
    return os.environ.get("NPU_QCREL") == "2" and qp1_on()


def qzacc_on():
    """GN_QZACC=1 (Q' QPASS + QROW + GN_QCORE): no C zeroing at the firing start; a chain's first MMA of a sub-tile
    (the one reading its loaded C) takes the MMA config with bit 0 (zero the accumulator input) on in pass 0."""
    return os.environ.get("GN_QZACC") == "1"


def qp1_on():
    """GN_QP1=1 (with GN_QMB1=5, 2 rows, NP even): layout P1, a vertical pair shares W: each core's W ring holds only its
    half of every W record (column groups 0 .. NP / 2 - 1 in row 0, the rest in row 1, written by that core's own W
    DMA); the other half is read through the neighbour window, the two cores taking turns per half (MB2's tokens)."""
    on = os.environ.get("GN_QP1") == "1"
    assert not on or (os.environ.get("GN_QMB1") == "5" and (NP % 2 == 0 or qp1k_on())), "GN_QP1 needs GN_QMB1=5, NP even"
    return on


def qp1k_on():
    """GN_QP1K=1 (with GN_QP1): P1 split by K, not by column groups (any NP, e.g. 64x80): each core's W slot holds one
    K half (128 of a pass's 256) of every group, row r half r % 2; every sub-tile reads its first 16 k steps from row
    0's memory and the rest from row 1's (one W FIFO restart mid sub-tile). The record halves are per group
    (npu_gemm_check); no group rotation, so no output column swap."""
    return os.environ.get("GN_QP1K") == "1"


def qpass_wdelta():
    """W slot 1 - W slot 0 in bytes (QBANK_RING_AT's W pins; GN_QPINS' if set)."""
    w = [int(v) for v in qpins(QBANK_RING_AT).split(",")[1].split("@")[1].split(":")]
    return w[1] - w[0]
LOOM_ENV = dict(LOOM_EXP_LOCKED_PACK="1", LOOM_EXP_LATE_STORAGE="1", LOOM_EXP_LS_SEND_PITCH="2",
                LOOM_EXP_PANEL_GROUPS=",".join(map(str, GE.GROUPS)), LOOM_EXP_PANEL_STREAM="1")


@dataclasses.dataclass(frozen=True)
class Config:
    cols: int           # NPU columns, N = 64 * cols
    nb: int             # 64-row M blocks, M = 64 * nb
    ks: tuple           # k-blocks per pass for (head, mid, mid, tail)
    passes: int         # K = 8 * passes * sum(ks)
    mu: int = 2         # M slabs per loop iteration of a leaf
    acap: int = 2       # activation ring slabs, head and mids
    acap_t: int = 2     # activation ring slabs, tail (its memory also holds C)
    entry: str = "npu_gemm"
    gate: int = 0       # NPU-side gate: poll supply per job (0: ungated); see GATE below
    fmt: str = "IQ4_XS"  # fuse 2: the raw weights' GGUF format
    kraw: int = 0       # fuse 2: the raw rows' K (row stride; the call reads K = 1024 * passes of it), 0: the call's K
    swiglu: bool = False  # one replay group: a column's first NP / 2 slabs are gate rows, the rest the up rows of the
                          # same features; the tail writes silu(gate) * up as bfp16ebs8 fragments (convert_swiglu) into
                          # H (hpass), the next GEMM's activations (ain = "h")
    ain: str = ""         # "h": the activations are read from H (a swiglu call's output) instead of the encoder's stream
    qsw: bool = False     # (Q', NP 4) FFN gate / up: column c's slabs 0, 1 are gate rows swcol c .., 2, 3 the same up
                          # rows dgu bytes on; the row convert writes h = silu(gate) up as bfp16 records (convert_swiglu,
                          # hord "P4"), 576 B a 16-token row, into H = down's A ([M block][call, hpass of them][slab]
                          # [column][576 B]; this call's at its C binding offset)
    dgu: int = 0          # swiglu fuse 2: bytes from a gate row to its up row (the raw stream per pass: [gate / up][32 rows])
    hpass: int = 0        # H as [M block][pass hpass][16-token slab][128 k-blocks] (the FFN block's swiglu output, down's
                          # input): swiglu (cols = hpass): column c writes pass c; ain "h": records [M block][pass, slab]
    swcol: int = 0        # swiglu fuse 2: raw rows between columns' first gate rows (0: TN / 2)
    gord: str = "P4"      # fuse 2: the grid decoders' k-block order (gen_npu_dec DQ_GORD; "N": natural; "GF": Q4_K in P4)
    hord: str = ""        # swiglu: H in the down decoder's k-block order: "P4" ((0, 2, 1, 3) per 4 k-blocks: IQ4_XS's
                          # and the P4 grid decoders'), "Q4K" ((0, 4, 1, 5, 2, 6, 3, 7) per 8: Q4_K's; a call's 4 gate
                          # k-blocks (one format) are then slabs 32 rows apart, the caller steps call j's rows by
                          # 64 (j / 2) + 16 (j % 2)), "": natural
    fuse: int = 0       # 0: the W binding holds the BFP16 panels (npu_gemm_check). 2: every GEMM core fills its own
                        # weight panel slice at each replay group's first firing, decoding the super-blocks covering
                        # it from the column's raw GGUF rows (kraw; gen_npu_dec.leaf_inline, in its idle W ring), its
                        # slice through a leaf-synced egress ring into its memory-tile panel (constrain.fill; HRX patch
                        # 0020); W is leaf-synchronized. Tried, not kept: 1, the same fill from a BFP16 pass-through
                        # stream.
    rawdma: int = 1     # fuse 2: each filler's raw rows reach it through its W DMA instead of its core stream: the
                        # column's raw stream lands in a memory-tile ring the fillers drain in turn, each queued ahead of
                        # its panel's records, into a ring in its idle W ring (constrain.intake, HRX patch 0022)
    gwest: int = 0      # gate (fuse 2): the relay is the waiter's east neighbor (column 2, row 1; it loads the waiter's
                        # memory through its west window) instead of the row below it, so each of the two columns sends
                        # one gate stream (tick / signal) and has room for two fill egresses (fuse_cover; HRX
                        # patch 0025)
    ghelp: int = 0      # gwest: the gate's cores (row 1 of columns 1, 2), idle while their heads decode rows 1 and 0,
                        # decode row 1 themselves and hand its records to the head over neighbor memory (fuse_helper)
    hvroll: int = 0     # ghelp: the helpers' decoder with its scale halves rolled (gen_npu_dec VROLL; program memory)
    htab: int = 1       # ghelp: the helpers' decoder table built once per fill (else every unit; program memory)
    q3roll: int = 0     # fuse 2: Q3_K's record realign of both halves in a loop (gen_npu_dec Q3ROLL; program memory)
    seam: int = 1       # head / mids: the next iteration's first sub-tile init and its WB pops ride with the last
                        # sub-tile's cascade writes (carried across the back edge), like the inner sub-tile boundaries;
                        # 2: not on ghelp's helpers (program memory)
    eg512: int = 1      # fuse 2: egress records copied 64 bytes a load / store (every source 64-byte aligned: ks and
                        # the hsplit piece multiples of 4 k-blocks), else 16
    rskip: int = 1      # fuse 2: decode only the half super-blocks (16 k-blocks) overlapping the filler's k-blocks
    hsplit: int = 1     # ghelp: the head's k-blocks past its last whole super-block come from the helper, which decodes
                        # that super-block anyway (the head, the fill's slowest core, decodes one super-block less)
    hcap: int = 4       # ghelp: the helper's record ring depth (a power of 2; 2 couples the helper to the head's egress)
    thelp: int = 0      # fuse 2: the tail decodes its own slice for the last mid (a helper like ghelp's: fuse_helpers)
    hlate: int = 0      # kchain: the helper rings declared after the A rings (rings are placed in channel order: the mids'
                        # second small ring would split their banks so no contiguous A ring is left: grid formats)
    gtail: int = 0      # kchain gwest: the gate's waiter and relay on the tails of columns 1, 2 (rows 1: program memory)
    kchain: int = 0     # fuse 2: core r decodes super-block r of each unit (the fill balanced) whatever the k-slices: each row
                        # sends the k-blocks of its super-block before its slice to the row above (kchain_send)
    dfr: int = 0        # swiglu (mu 1, not paired): each M block's epilogue deferred into the next block's passes: segment s
                        # at pass s, one h block (a push) after sub-tiles DFR_T of each iteration, in the tail's cascade
                        # slack; the C ring's other half (2 MP slots, released in the same order, so the records and
                        # their addressing are unchanged); the call's last block runs it whole on its last pass
    dfi: int = 1        # dfr: the deferred push inline (no branch: a firing without a deferred segment pushes into a
                        # dummy buffer), so no branch delay / quiescence splits the push tile
    ufmt: str = ""      # swiglu fuse 2 column pairs (dfr): pair u is gate column 2 u (fmt rows, binding 7) and up
                        # column 2 u + 1 (ufmt rows in ugord's order, binding 8; dgu unused) of the same 64 features, so
                        # each column's cores hold one decoder; the tails swap halves (pair_swap), each runs the
                        # epilogue of 32
    ugord: str = ""     # ufmt's k-block order (the activations' order is shared: Q4_K's "GF" beside a P4 format)
    ofeat: int = 0      # final f32 output, fragment-major into [token / 16][ofeat / 16][16][16]: column c's 80 features are
                        # fragments 5c .. 5c + 4; needs one replay group (GE.GROUPS = (passes,)) and
                        # LOOM_EXP_LS_SEND_PITCH=1 (the whole f32 slot is sent)


# NPU-side gate (cfg.gate = supply): column 0's head waits for each job's ready word itself and its neighbor mid relays.
# Extra bindings: 3 flag (read; 16-byte record, word 0 ready = the job's gate value), 4 tick (write; scratch), 5 signal
# (write; go at byte 0, done at byte 16). At a job's first firing the head polls request-driven
# (constrain.request: one fresh flag read per tick) until ready >= (its job count + 1) * GATE_CALLS (jobs count from 1 after the
# setup call; jobs sharing a ready word store theirs in order, after the GPU joined the earlier ones), at a pace of
# GP0 + GPR * max(0, polls - GRAMP) + GPS * max(0, polls - GFAST) delay iterations (~7.7 ns each in the engine). A poll
# sees ready ~17 polls after the GPU stored it (the reads run that far behind; pp2048 fit, and host-stamped flag records
# show the read itself is fresh), so a wait costs ~17 paces: a constant pace through the waits of layers with NPU work
# (up to ~15 ms; 512 polls of ~0.04 ms) keeps that ~0.65 ms (0.31 ms paces cost ~5 ms per waiting job, 0.66 s of
# pp2048), where backing off from the start cost 4-17 ms per job, growing with the gap. A gentle ramp then covers the
# waits across a GPU-only FFN (UD-IQ4_XS layers 0, 14-17: ~45 ms, ~5 ms late; at a constant pace into the back-off,
# layer 15's out saw ready ~40 ms late). The last 128 polls back off, so the supply still
# lasts ~1.25 s (the tick stream takes one record per poll; 1024 is the most the shim's descriptors chain). It hands
# [seq, status, polls, ncalls] to the mid over a leaf-synchronized neighbor channel and ticks / reads out the rest of the supply GB records per
# firing of the next job. The mid emits go and done (constrain.signal + constrain.gate): the control program's gate
# invocation waits for go before any data moves; its done invocation queues done after the job's egress, so done lands
# after C. done = the job's ready value, with GATE_FAILED set if the head gave up (status 2). Job state (firings done, firings per job, leftover
# supply, pending reads, sequence) lives in private storage, which the array setup of a core stream plan zeroes; a job of
# N calls is N * nb * passes firings, and before the first gated job (state zero) it is one call: the setup call's.
# The wait and the relay run at a job's first firing (gate_first; the end of a firing only counts and drains,
# gate_count): state zero is then at the gate, so the setup runs no call (LOOM_EXP_SETUP_ONLY). Tried, not kept: the
# wait after the last firing of the job before (the setup then ran one ungated call).
# GB (GN_GB): 32 (was 8) drains a job's leftover supply in the background, so its last firing does not tick + read
# the rest one round trip at a time (~580 cycles each: an 8-call P1K job 5417K -> 5198K cycles, 11-18)
GP0, GPR, GRAMP, GPS, GFAST, GB = 5000, 200, 512, 16000, 896, int(os.environ.get("GN_GB", "32"))
# The protocol the host follows (dispatch.txt "npugate <GATE_SUPPLY> <GATE_CALLS> <GATE_RECORD>"): a job's gate value is
# sequence * GATE_CALLS + its calls (jobs of 1 .. GATE_CALLS - 1 calls); done lands GATE_RECORD bytes into the signal
# binding. GATE_FAILED in done: the head gave up (the host sets it too for a failed command; gen_npu_unpack.gen_flag_wait).
GATE_SUPPLY, GATE_CALLS, GATE_RECORD, GATE_FAILED = 1024, 64, 16, 1 << 31


def decoder(fmt, gord="P4", g8=True):
    """gen_npu_dec for fmt (its constants follow DQ_FMT, DQ_GORD and DQ_G8 at import). g8: IQ3_S from the byte grid,
    which the gate / up images and every Q' image need (the 8 KB bf16 grid does not fit beside their rings); the other
    images keep the bf16 grid's modulo-scheduled loop (faster)."""
    import gen_npu_dec
    g8 = g8 or QF.get("qg8", False)
    g = "1" if g8 else "0"
    gs = ("2" if QF.get("qsw") else "1") if QF.get("grid") else "0"   # (the Q' grid map: IQ2_XS / IQ2_S tables in two
    # symbols; FFN gate / up, qgmap_on: @gz 2 KB-aligned, gen_npu_dec GS2K)
    q1 = "1" if QF.get("qg8") else "0"    # (Q' images: one super-block a record row, gen_npu_dec Q1SB)
    if (gen_npu_dec.FMT0 != fmt or os.environ.get("DQ_GORD", "P4") != gord or os.environ.get("DQ_G8", "1") != g
            or os.environ.get("DQ_GSPLIT", "0") != gs or os.environ.get("DQ_Q1SB", "0") != q1):
        os.environ["DQ_FMT"] = fmt
        os.environ["DQ_GORD"] = gord
        os.environ["DQ_G8"] = g
        os.environ["DQ_GSPLIT"] = gs
        os.environ["DQ_Q1SB"] = q1
        gen_npu_dec = importlib.reload(gen_npu_dec)
    return gen_npu_dec


def loom_env(cfg):
    """LOOM_ENV for cfg."""
    env = dict(LOOM_ENV)
    if cfg.fuse:   # the fill egress rings are contiguous (fuse_prologue steps them by a record)
        env["LOOM_EXP_LS_SEND_PITCH_RECORD"] = str(NP * 4 * 256 // 8 * 4)
    if cfg.fuse:   # the shared last-two-rows panel interleaved per unit (its filler emits both rows' slabs of a unit
        env["LOOM_EXP_PANEL_INTERLEAVE"] = str(NP)   # together; uniform slots of KS (36, 36, 36, 20) would not fit)
    if cfg.fuse == 2:   # the inlined decoder's tables low (HRX patch 0021): the contiguous W ring spans the next banks
        env["LOOM_EXP_PACK_LOW"] = "1"
    if cfg.fuse == 2 and qf4r_on():   # (GN_QF4R) the raw records split by parity over the two W multicasts
        env["LOOM_EXP_INTAKE_SPLIT"] = "1"
        env["LOOM_EXP_INTAKE_STAGE_DEPTH"] = os.environ.get("GN_QF4RDEPTH", "4")   # (2 records a multicast in flight)
    if os.environ.get("GN_QLSDEPTH"):   # the activation streams' memory-tile stage depth (records; default the core ring's)
        env["LOOM_EXP_LS_STAGE_DEPTH"] = os.environ["GN_QLSDEPTH"]
    if not groups(cfg):
        env["LOOM_EXP_PANEL_GROUPS"] = str(cfg.passes)
        if cfg.fuse == 2:   # one group, still streamed (HRX patch 0019): the panel's fill
            env["LOOM_EXP_PANEL_STREAM_SINGLE"] = "1"
    if os.environ.get("GN_QGROUPS"):   # (Q') the panel in pass groups drained block-major (HRX patch 0086): the next
        # call refills a group once the last M block passed it
        assert sum(int(v) for v in os.environ["GN_QGROUPS"].split(",")) == cfg.passes
        env["LOOM_EXP_PANEL_GROUPS"] = os.environ["GN_QGROUPS"]
        env["LOOM_EXP_PANEL_BLOCKMAJOR"] = "1"
        env.pop("LOOM_EXP_PANEL_STREAM_SINGLE", None)
    if cfg.swiglu:
        env["LOOM_EXP_LS_SEND_PITCH"] = str(SW_PITCH)
        env["LOOM_EXP_LS_SEND_PITCH_RECORD"] = str(SW_REC)
        env["LOOM_EXP_PANEL_GROUPS"] = str(cfg.passes)
    if cfg.fuse == 2:   # other layouts share the context: routes must not outlive them
        env["LOOM_EXP_ROUTE_RESET"] = "1"
        env["LOOM_EXP_DRAIN_USED"] = "1"   # (HRX patch 0048: the stale-request drain raises only the image's locks)
    if cfg.gate:
        env["LOOM_EXP_SETUP_ONLY"] = "1"
    # HRX main: the inferred concat affinity grows the decoders' frames past 16 KB (worker 16518 B on q4_k), so the
    # images keep the pin's allocation; q4_k also needs the LRU register choice to fit its dequant loads
    env["LOOM_EXP_NO_CONCAT_PROPAGATE"] = "1"
    if os.environ.get("GN_QROW") == "1":   # (layout Q' row-rotated tail) accumulators LRU: the chains come back to the
        env["LOOM_EXP_LRU"] = os.environ.get("GN_QLRU", "aie2p.mbms")  # header's registers (no moves on the back edge)
        if os.environ.get("GN_QACQREL") == "1":   # (UNSAFE: FIFO pops are not ordered behind the acquire)
            env["LOOM_EXP_LOCK_ACQREL"] = "6"
    if qpass_on() and os.environ.get("GN_QHW") != "0":   # the (pass, row) loop as a hardware loop (HRX patch 0080: the
        # authored trip count; 0083 '@': the counts go to the last loops, the GEMM's, after any fill loop)
        env["LOOM_EXP_HWLOOP_TRIP"] = "@" + (f"{MP * (cfg.passes - 1) // (2 if qunr_on() else 1)},{MP}" if qpeel_on()
                                             else str(MP * cfg.passes // (2 if qunr_on() else 1)))
    if qlock_on():   # spread lock lines, reads 2 / 1 cycles from a lock (HRX
        # patch 0081; still ordered: the first reads after an acquire hit the slot's start, which the DMA wrote first,
        # and the last ones before a release hit its end, which the DMA writes last)
        env["LOOM_EXP_LOCK_READ_SEP"] = "2,1"
    if qpass_on() and os.environ.get("GN_QFLOAT") == "1":   # the Q' body's text order stops being the issue floor
        # (HRX 0061-0065 planner controls: dependencies only; views no boundaries, pops / fills 2 cycles under)
        env.update(LOOM_EXP_LOCKED_FLOAT="all", LOOM_EXP_LOCKED_VIEWS="1", LOOM_EXP_LOCKED_HOIST="2")
        if os.environ.get("GN_QMB1") == "5":   # (MB2 / P1 tokens) a plain acquire stays an order floor: the neighbour reads
            # after a token acquire have no dependency on it (acq.cond, the rings' and the row-end token's, still float)
            env["LOOM_EXP_LOCKED_FLOAT"] = "acc,acq.,add,and," + ",".join("bcdefghijklmnopqrstuvwxyz")
        if cfg.fuse == 2:   # (HRX 0084) only the GEMM's blocks float: a locked body has no memory dependencies, and
            env["LOOM_EXP_LOCKED_FLOAT_MMA"] = "1"   # the decoder's header stores must land before its vector loads
    if cfg.fuse == 2 and cfg.fmt == "Q4_K":
        env["LOOM_EXP_LRU"] = "aie2p.vec256"
    return env


def groups(cfg):
    """The call's replay groups of passes, () for a single group."""
    if cfg.swiglu or cfg.ain:
        return ()
    g = GE.pass_groups(cfg.passes)
    assert len(g) <= 2
    return g if len(g) > 1 else ()


def bf16_round(x):
    """x rounded to bf16 (nearest even), as a float."""
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", u))[0]


def bf16_bits(x):
    return struct.unpack("<I", struct.pack("<f", bf16_round(x)))[0] >> 16


def f32_bits(x):
    v = struct.unpack("<i", struct.pack("<f", x))[0]
    return v


SW_REC = NP * 2 * 72      # swiglu: a segment's output record, NP k-blocks x 2 token halves of bfp16 fragments (720 B)
SW_PITCH = 8              # its send pitch: a C slot of 8 records (5760 B) holds the segment's f32 C (5120 B)
SW_SLOT = SW_PITCH * SW_REC
SW_HALF = 512 * NP        # column pairs: half a segment's f32 C (NP / 2 slabs)
DFR_T = (0, 2)            # Config.dfr: the sub-tiles after which a deferred push runs (2 of the 4: 8 pushes a pass)
DFT = "reg<aie2p.mstfifo>, reg<aie2p.mpfs>, reg<aie2p.mr26_fifo_st>"


@contextlib.contextmanager
def np_override(n):
    """Generate with n 16-row slabs per column (TN = 16 n) instead of NP (e.g. 4: a swiglu call's 32 features per
    column, so its k-block units line up with 1024-feature passes)."""
    global NP, TN, SW_REC, SW_PITCH, SW_SLOT, SW_HALF
    saved = NP, TN, SW_REC, SW_PITCH, SW_SLOT, SW_HALF
    NP, TN = n, 16 * n
    SW_REC = NP * 2 * 72
    SW_PITCH = 8
    SW_SLOT = SW_PITCH * SW_REC
    SW_HALF = 512 * NP
    assert SW_SLOT >= 1024 * NP
    try:
        yield
    finally:
        NP, TN, SW_REC, SW_PITCH, SW_SLOT, SW_HALF = saved


def h_col_stride(cfg):
    """swiglu: bytes between columns' records in H ([M block][pass][slab][128 k-blocks]): swcol features apart, a
    pass per 1024 (column c writes pass c) or swcol / 8 k-blocks within a pass (the columns' records side by side in a
    call's window; swcol 32: 8 columns = 256 features, call j at feature 256 j)."""
    if cfg.swcol >= 1024:
        assert cfg.swcol % 1024 == 0 and cfg.hpass >= cfg.cols * cfg.swcol // 1024
        return cfg.swcol // 1024 * MP * 128 * 144
    n, per = cfg.cols, 8 * NP
    assert 1024 % (cfg.swcol * n) == 0 and cfg.swcol == per
    return cfg.swcol // 8 * 144


def slab(ks):
    return (144 * ks + 63) // 64 * 64


def stream_bytes(cfg):
    """(activation, weight, C) binding sizes."""
    a = sum(cfg.nb * cfg.passes * MP * slab(k) for k in cfg.ks)
    if cfg.ain == "h":   # the view's extent in H ([M block][pass][slab][128 k-blocks])
        a = (144 * sum(cfg.ks[:-1]) + (cfg.nb - 1) * cfg.hpass * MP * 128 * 144 + (cfg.passes * MP - 1) * 128 * 144
             + slab(cfg.ks[-1]))
    w = cfg.cols * sum(cfg.passes * NP * slab(k) for k in cfg.ks)
    if cfg.qsw:   # (Q') the receiver's extent in H from call 0's records: [M block][hpass calls][slab][column][576 B]
        return a, w, ((cfg.nb - 1) * cfg.hpass + 1) * MP * cfg.cols * SW_REC
    if cfg.swiglu:   # the receiver's extent in H: the last 16-token row's columns
        last = (cfg.cols - 1) * h_col_stride(cfg)   # the last column's (pass's) last record in H
        return a, w, ((cfg.nb - 1) * cfg.hpass * MP * 128 * 144 + (MP - 1) * 128 * 144 + last + SW_REC)
    return a, w, max(1, len(groups(cfg))) * cfg.cols * cfg.nb * MP * NP * 2 * 256


def shuffle16_cycles():
    """The non-trivial cycles of piece i -> 2 (i % 8) + i // 8 on 16 pieces."""
    seen, out = set(), []
    for i in range(16):
        c, j = [], i
        while j not in seen:
            seen.add(j)
            c.append(j)
            j = 2 * (j % 8) + j // 8
        if len(c) > 1:
            out.append(c)
    return out


SHUFFLE16_CYCLES = shuffle16_cycles()


# lsched: slot classes and result latencies (Loom's AIE2P itineraries) of a locked straight-line block's ops
LS_FREE = ("slice", "concat")
LS_SLOT = (("vlda", "lda", 7), ("mova", "lda", 1), ("mov.scalar-to-address", "lda", 1), ("vldb", "ldb", 7),
           ("vconv", "st", 2), ("vst", "st", 1), ("vups", "mv", 3), ("vband", "mv", 2), ("vbor", "mv", 2),
           ("vmax", "mv", 2), ("vmin", "mv", 2), ("vmov", "mv", 2), ("vadd", "mv", 2), ("vbcst", "mv", 2),
           ("vshuffle", "mv", 2), ("vsel", "mv", 2), ("vlt", "mv", 2), ("vmul", "vec", 6), ("vmac", "vec", 6),
           ("mov.i32", "lng", 1))


def lsched(lines, window=24, alpha=1.0):
    """Reorders a straight-line block (SSA lines, no labels) for the locked schedule, where the emission order is the
    issue order and only adjacent independent ops of different slots pack: cycle-driven list scheduling on LS_SLOT's
    slot classes and latencies (one op a slot a cycle; priority: latency-weighted height less alpha x the original
    position, which keeps the chains' register pressure near the original order's). An op may not pass one more than
    window lines earlier in the original order. Free views (slice / concat) follow their inputs, constants
    (mov.i32 / vbcst: Loom rematerializes them at every use anyway) go just before their first user, and loads stay
    before the stores after them (the epilogue's store fifo overwrites C in place)."""
    import re
    ops = []
    for i, x in enumerate(lines):
        lhs, eq, rhs = x.partition(" = ")
        if not eq:
            lhs, rhs = "", x
        mn = rhs.split()[0] if rhs.split() else ""
        if mn.startswith(LS_FREE):
            slot, lat = None, 0
        else:
            slot, lat = next(((sl, la) for pre, sl, la in LS_SLOT if mn.startswith(pre)), ("alu", 1))
        ops.append(dict(x=x, mn=mn, defs=re.findall(r"%[\w.]+", lhs),
                        uses=re.findall(r"%[\w.]+", rhs.split(" : ")[0]), slot=slot, lat=lat))
    by = {d: i for i, o in enumerate(ops) for d in o["defs"]}
    preds = [set(by[u] for u in o["uses"] if u in by and by[u] != i) for i, o in enumerate(ops)]
    for i, o in enumerate(ops):
        if o["mn"].startswith("vst"):
            preds[i] |= {k for k in range(i) if ops[k]["mn"].startswith(("vlda", "vst"))}
    succ = [[] for _ in ops]
    for i, ps in enumerate(preds):
        for k in ps:
            succ[k].append(i)
    h = [0] * len(ops)
    for i in reversed(range(len(ops))):
        h[i] = ops[i]["lat"] + max((h[j] for j in succ[i]), default=0)
    kdef = [False] * len(ops)
    for i, o in enumerate(ops):
        kdef[i] = o["mn"].startswith(("mov.i32", "vbcst")) and all(kdef[k] for k in preds[i])

    def kneed(i, acc):
        for k in preds[i]:
            if kdef[k] and k not in acc:
                acc.add(k)
                kneed(k, acc)
        return acc
    done, ready, out, cyc = [False] * len(ops), [0] * len(ops), [], 0
    while True:
        lo = min((i for i in range(len(ops)) if not done[i] and not kdef[i]), default=None)
        if lo is None:   # (constants nobody uses)
            out.extend(o["x"] for i, o in enumerate(ops) if not done[i])
            return out
        assert cyc < 20 * len(ops) + 100, "lsched stuck"
        used = set()
        while True:
            free = [i for i in range(len(ops)) if not done[i] and not kdef[i] and ops[i]["slot"] is None and
                    all(done[k] for k in preds[i])]
            cand = free or [i for i in range(len(ops)) if not done[i] and not kdef[i] and i <= lo + window and
                            ops[i]["slot"] is not None and ops[i]["slot"] not in used and
                            all(kdef[k] or done[k] and ready[k] <= cyc for k in preds[i])]
            if not cand:
                break
            i = cand[0] if free else max(cand, key=lambda k: (h[k] - alpha * k, -k))
            for k in sorted(kneed(i, set())):
                if not done[k]:
                    done[k], ready[k] = True, cyc
                    out.append(ops[k]["x"])
            done[i] = True
            ready[i] = max([ready[k] for k in preds[i]] + [cyc]) if free else cyc + ops[i]["lat"]
            out.append(ops[i]["x"])
            if not free:
                used.add(ops[i]["slot"])
        cyc += 1


def bump(L, dst, src, off):
    cur, i = src, 0
    while off > 0:
        step = min(off, 448)
        off -= step
        nxt = dst if off == 0 else f"{dst}_b{i}"
        i += 1
        L.append(f"  {nxt} = padda {cur}, {step}")
        cur = nxt
    if cur != dst:
        L.append(f"  {dst} = copy {cur} : reg<aie2p.ep> -> reg<aie2p.ep>")


def array_program(L, cfg):
    e = L.append
    rows, roles = len(cfg.ks), ("head",) + ("mid",) * (len(cfg.ks) - 2) + ("tail",)
    cols, nb, P = cfg.cols, cfg.nb, cfg.passes
    e("aie2p.target<array> @array_target\naie2p.target<core> @core_target\n")
    e(f"low.func.def public retain target<amd.xdna.aie2p.array>(@array_target) abi(array_program) @{cfg.entry}() asm {{")
    for k, v in (("zero", 0), ("one", 1), ("two", 2), ("nw", rows * cols), ("rec", nb * P), ("orec", nb), ("fold", P)):
        e(f"  %{k} = constant.u32 {v} : reg<aie2p.array.scalar : index>")
    e("  %origin = constant.u64 0 : reg<aie2p.array.offset : offset>")
    e("  %workers = group %nw")
    e('  %ab = binding 0, "read"')
    e('  %wb = binding 1, "read"')
    e('  %cb = binding 2, "write"')
    for i in range(max(cols, 2, MP) + 6):
        e(f"  %n{i} = constant.u32 {i} : reg<aie2p.array.scalar : index>")
    panel = sum(P * NP * slab(k) for k in cfg.ks)
    for c in range(cols):
        for r, role in enumerate(roles):
            e(f"  %lane{c}_{r} = constant.u32 {c * rows + r} : reg<aie2p.array.scalar : index>")
            grole = worker_leaf(cfg, c, r)[0]
            e(f"  %k{c}_{r} = worker %workers, %lane{c}_{r}, @{grole}")
            e(f"  constrain.location %k{c}_{r}, %n{c}, %n{2 + rows - 1 - r}")
    # C first: rings are placed in channel order, so the tail's C claims whole banks before its input rings
    e(f"  %ncols = constant.u32 {cols} : reg<aie2p.array.scalar : index>")
    seg = MP * NP * 4 * 256 // MP
    ng = max(1, len(groups(cfg)))   # C segments per M block and group
    e(f"  %nseg = constant.u32 {ng * nb * MP} : reg<aie2p.array.scalar : index>")
    if cfg.swiglu:
        # column c's record (NP k-blocks of 16 tokens) at k-block NP c of its 16-token row in H
        rec = f"{SW_REC // 4}"   # column c: pass c of H [M block][pass][slab][128 k-blocks]
        e(f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols}x{nb}x{MP}x{rec}xi32, "
          f"#encoding.layout.strided<strides=[{h_col_stride(cfg) // 4}, {cfg.hpass * MP * 128 * 36}, {128 * 36}, 1]>>>")
        stype = f"tile<{rec}xi32>"
    elif cfg.ofeat:
        # a segment is [sub-tile 5][16 tokens][16 f32] (the tail's convert_f32)
        # the tail's slot is [sub-tile][16 tokens][16 f32] (convert_f32): NP consecutive fragments of one 16-token row
        # (the 1280 contiguous words as 5 x 256: a shim DMA wrap holds at most 1023)
        rec = f"{NP}x256"
        # with two replay groups each group's partial is its own tensor, stacked: [group][token / 16][ofeat / 16][16][16]
        e(f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols}x{ng * nb * MP}x{rec}xi32, "
          f"#encoding.layout.strided<strides=[{NP * 256}, {cfg.ofeat * 16}, 256, 1]>>>")
        stype = f"tile<{rec}xi32>"
    else:
        rec = f"{seg // 8}"
        e(f"  %c_all = receiver %cb, 0 : reg<aie2p.array.receiver : tile<{cols}x{ng * nb * MP}x{rec}xi32>>")
        stype = f"tile<{rec}xi32>"
    for c in range(cols):
        e(f"  %cr{c} = partition.receiver %c_all, %origin, %n{c}, %ncols : reg<aie2p.array.receiver : tile<{rec}xi32>>")
        e(f"  %sc{c} = sender %k{c}_{rows - 1}, 2 : reg<aie2p.array.sender : {stype}>")
        e(f"  %chc{c} = channel %sc{c}, %cr{c}, %n{2 if qalt_on() and not qaltp_on() else MP * (2 if cfg.dfr else 1)}, %nseg : reg<aie2p.array.channel : tile<{rec}xi32>>")
        e(f"  constrain.leaf_sync %chc{c}")
    if cfg.fuse:
        fuse_channels(e, cfg, panel)
    # weights: per-column panels [slice][pass][record], staged in that column's memory tile, replayed per M block
    for c in range(cols):
        for r in range(rows):
            wrec = NP * slab(cfg.ks[r])
            order = fuse_order(cfg, c)
            w_off = c * panel + sum(P * NP * slab(cfg.ks[j]) for j in order[:order.index(r)])
            e(f"  %wo{c}_{r} = constant.u64 {w_off} : reg<aie2p.array.offset : offset>")
            e(f"  %ws{c}_{r} = sender %wb, 0 : reg<aie2p.array.sender : tile<{nb}x{P}x{wrec // 4}xi32, "
              f"#encoding.layout.strided<strides=[0, {wrec // 4}, 1]>>>")
            e(f"  %wv{c}_{r} = view.sender %ws{c}_{r}, %wo{c}_{r} : reg<aie2p.array.sender : tile<{wrec // 4}xi32>>")
            e(f"  %rw{c}_{r} = receiver %k{c}_{r}, 1 : reg<aie2p.array.receiver : tile<{wrec // 4}xi32>>")
            e(f"  %chw{c}_{r} = channel %wv{c}_{r}, %rw{c}_{r}, %two, %rec : reg<aie2p.array.channel : tile<{wrec // 4}xi32>>")
            e(f"  constrain.stage %chw{c}_{r}, %n{c if cols > 1 else 1}")
            if cfg.fuse:   # each panel filled by its row's filler (fuse_cover); a filler's own W is leaf-synced
                f, _ = fuse_filler(cfg, c, r)
                if f == r or r in fuse_helpers(cfg, c):
                    e(f"  constrain.leaf_sync %chw{c}_{r}")
                e(f"  constrain.fill %chw{c}_{r}, %chd{c}_{f}")
    if cfg.rawdma:   # each filler's raw ring in its W ring (fuse_raw_ring), records [16 rows][RECROW bytes]
        e(f"  %firows = constant.u32 16 : reg<aie2p.array.scalar : index>")
        for c in range(cols):
            cc = colcfg(cfg, c)
            e(f"  %fipitch{c} = constant.u32 {decoder(cc.fmt, cc.gord, cc.swiglu).RECROW} : reg<aie2p.array.scalar : index>")
            for r in dict.fromkeys(list(fuse_cover(cfg, c)) + list(fuse_helpers(cfg, c))):
                cov = fuse_dcov(cfg, c, r)
                e(f"  %fioff{c}_{r} = constant.u32 {fuse_raw_ring(cc, cov)[0]} : reg<aie2p.array.scalar : index>")
                e(f"  constrain.intake %chr{c}_{r}, %chw{c}_{r}, %fioff{c}_{r}, %firows, %fipitch{c}")
    # activations: one stream per row, one slab per record, multicast along the row (first branch rotates)
    for r in range(rows):
        fa = slab(cfg.ks[r])
        a_off = sum(nb * P * MP * slab(k) for k in cfg.ks[:r])
        arec = f"{fa // 4}xi32"
        if cfg.ain == "h":   # H [M block][pass][slab][128 k-blocks]: records [M block][pass, slab]
            a_off = 144 * sum(cfg.ks[:r])
            e(f"  %ao{r} = constant.u64 {a_off} : reg<aie2p.array.offset : offset>")
            rw = fa // 4
            sp = next(k for k in range(1, 9) if rw % k == 0 and rw // k <= 1023)
            arec = f"{sp}x{rw // sp}xi32"
            e(f"  %as{r} = sender %ab, 0 : reg<aie2p.array.sender : tile<{nb}x{P * MP}x{sp}x{rw // sp}xi32, "
              f"#encoding.layout.strided<strides=[{cfg.hpass * MP * 128 * 36}, {128 * 36}, {rw // sp}, 1]>>>")
        else:
            e(f"  %ao{r} = constant.u64 {a_off} : reg<aie2p.array.offset : offset>")
            e(f"  %as{r} = sender %ab, 0 : reg<aie2p.array.sender : tile<{nb * P * MP}x{fa // 4}xi32>>")
        e(f"  %a{r} = view.sender %as{r}, %ao{r} : reg<aie2p.array.sender : tile<{arec}>>")
        e(f"  %arec{r} = constant.u32 {nb * P * MP} : reg<aie2p.array.scalar : index>")
        for j in range(cols):
            c = (2 * r + j) % cols
            acap = acap_tail(cfg) if roles[r] == "tail" else cfg.acap
            e(f"  %ra{c}_{r} = receiver %k{c}_{r}, 0 : reg<aie2p.array.receiver : tile<{arec}>>")
            e(f"  %cha{c}_{r} = channel %a{r}, %ra{c}_{r}, %n{acap}, %arec{r} : reg<aie2p.array.channel : tile<{arec}>>")
            e(f"  constrain.leaf_sync %cha{c}_{r}")
            e(f"  constrain.stage %cha{c}_{r}, %n{astage(r, cols, cfg.gwest) if cols > 1 else 2}")
    for ln in _HLATE if cfg.hlate else ():   # (fuse_channels)
        e(ln)
    _HLATE.clear()
    if cfg.gate:
        S = cfg.gate
        e('  %fb = binding 3, "read"')
        e('  %tb = binding 4, "write"')
        e('  %sb = binding 5, "write"')
        e(f"  %gsup = constant.u32 {S} : reg<aie2p.array.scalar : index>")
        e(f"  %gf_all = sender %fb, 0 : reg<aie2p.array.sender : tile<1x{S}x1x4xi32, #encoding.layout.strided<strides=[4, 0, 0, 1]>>>")
        e("  %gf = partition.sender %gf_all, %origin, %n0, %one : reg<aie2p.array.sender : tile<4xi32>>")
        gw, gr, gc = gate_ports(cfg)["waiter"], gate_ports(cfg)["relay"], gate_ports(cfg)["col"]
        kw, kr = f"%k{gc}_{gw['row']}", f"%k{gr.get('col', gc)}_{gr['row']}"
        e(f"  %gpoll = receiver {kw}, {gw['poll']} : reg<aie2p.array.receiver : tile<4xi32>>")
        e("  %chgp = channel %gf, %gpoll, %two, %gsup : reg<aie2p.array.channel : tile<4xi32>>")
        e("  constrain.core_stream %chgp")
        if not notick():   # (the tick channel: one fresh flag read per tick)
            e(f"  %gt_all = receiver %tb, 0 : reg<aie2p.array.receiver : tile<1x{S}x1x4xi32, #encoding.layout.strided<strides=[4, 0, 0, 1]>>>")
            e("  %gt = partition.receiver %gt_all, %origin, %n0, %one : reg<aie2p.array.receiver : tile<4xi32>>")
            e(f"  %gtick = sender {kw}, {gw['tick']} : reg<aie2p.array.sender : tile<4xi32>>")
            e("  %chgt = channel %gtick, %gt, %two, %gsup : reg<aie2p.array.channel : tile<4xi32>>")
            e("  constrain.core_stream %chgt")
            e("  constrain.request %chgp, %chgt")
        else:   # (a free-running poll, queued at the gate: Loom's constrain.request %p, %p, HRX pin)
            e("  constrain.request %chgp, %chgp")
        e(f"  %gn1s = sender {kw}, {gw['nbr']} : reg<aie2p.array.sender : tile<4xi32>>")
        e(f"  %gn1r = receiver {kr}, {gr['nbr']} : reg<aie2p.array.receiver : tile<4xi32>>")
        e("  %chgn = channel %gn1s, %gn1r, %one, %one : reg<aie2p.array.channel : tile<4xi32>>")
        e("  constrain.leaf_sync %chgn")
        e("  %gs_all = receiver %sb, 0 : reg<aie2p.array.receiver : tile<1x2x4xi32>>")
        e("  %gs = partition.receiver %gs_all, %origin, %n0, %one : reg<aie2p.array.receiver : tile<4xi32>>")
        e(f"  %gsig = sender {kr}, {gr['sig']} : reg<aie2p.array.sender : tile<4xi32>>")
        e("  %chgs = channel %gsig, %gs, %two, %two : reg<aie2p.array.channel : tile<4xi32>>")
        e("  constrain.core_stream %chgs")
        e("  constrain.gate %chgs")
        e("  constrain.signal %chgs")
    if fpair(cfg):   # pair_swap's rings: the gate tail's half's address, one slot, two records a segment
        e(f"  %nxs = constant.u32 {2 * cfg.nb * MP} : reg<aie2p.array.scalar : index>")
        for u in range(cols // 2):
            e(f"  %xs{u} = sender %k{2 * u}_{rows - 1}, {PAIR_PORT} : reg<aie2p.array.sender : tile<16xi32>>")
            e(f"  %xr{u} = receiver %k{2 * u + 1}_{rows - 1}, {PAIR_PORT} : reg<aie2p.array.receiver : tile<16xi32>>")
            e(f"  %chx{u} = channel %xs{u}, %xr{u}, %one, %nxs : reg<aie2p.array.channel : tile<16xi32>>")
            e(f"  constrain.leaf_sync %chx{u}")
    e("  return\n}\n")


def astage(r, cols, gwest=0):
    """The memory-tile column K-slice row r's activations stage in: one row per column (0, 2, 4, 6; 6 columns: 0, 2, 4, 1).
    gwest: row 1 in column 3 (the gate's columns 1 and 2 have two-filler panels, no descriptors left to stage)."""
    if gwest and r == 1:
        return 3
    return 2 * r if 2 * r < cols else (2 * r + 1) % cols


FUSE_REC = 200   # fused fill egress record words, at most (each replay group's fill must end on a whole record)


def gate_ports(cfg):
    """The gate's column, its workers (rows) and their port indices: the waiter's flag poll (core stream in), tick (core
    stream out) and neighbor sender, the relay's neighbor receiver and signal (core stream out). A worker core has
    one stream port per direction, so with fuse the gate moves to rows 1 (waiter) and 2 (relay) of column 1, whose
    head fills every panel (fuse_cover; the filling cores' stream input carries their raw weights; column 0's memory
    tile also stages an activation row: its northward link has no channel left for the flag stream)."""
    if QF.get("gports"):   # (layout Q': gen_npu_q places the gate)
        return QF["gports"]
    rows = len(cfg.ks)
    if cfg.fuse and cfg.gwest:   # (the relay's column: "col" is the waiter's)
        if cfg.gtail:   # (the tail's ports 0, 1, 2: A, W, C; 5, 6: kchain's raw intake and sender)
            assert cfg.kchain
            return {"col": 1, "waiter": dict(row=rows - 1, poll=3, tick=4, nbr=7),
                    "relay": dict(col=2, row=rows - 1, nbr=3, sig=4)}
        return {"col": 1, "waiter": dict(row=1, poll=2, tick=3, nbr=4), "relay": dict(col=2, row=1, nbr=2, sig=3)}
    if cfg.fuse:
        return {"col": 1 if cfg.cols > 1 else 0, "waiter": dict(row=1, poll=2, tick=3, nbr=4),
                "relay": dict(row=2, nbr=2, sig=3)}
    return {"col": 0, "waiter": dict(row=0, poll=2, tick=3, nbr=4), "relay": dict(row=1, nbr=2, sig=3)}


def worker_leaf(cfg, c, r):
    """Worker (c, r)'s leaf: (name, role, fused fill cover or None, gate role or None)."""
    rows = len(cfg.ks)
    role = "head" if r == 0 else "tail" if r == rows - 1 else "mid"
    cov = fuse_cover(cfg, c).get(r) if cfg.fuse else None
    gp = gate_ports(cfg)
    gate = None
    if cfg.gate:
        gate = ("waiter" if (c, r) == (gp["col"], gp["waiter"]["row"]) else
                "relay" if (c, r) == (gp["relay"].get("col", gp["col"]), gp["relay"]["row"]) else None)
    hs = fuse_helpers(cfg, c)
    h = fuse_helper(cfg, c)
    if cfg.kchain and c >= 0:
        name = role + "_k" + kchain_role(cfg, c, r) + "".join(map(str, cov or (r,)))
    elif r in hs:
        name = role + "_h"   # (ghelp: decodes its row for the head; thelp: the tail's for the last mid)
    elif r in hs.values() and r != 0:   # (thelp's last mid: not the plain columns' mid_l)
        name = role + "_f" + "".join(map(str, cov)) + "h"
    elif cov is None:
        name = role + ("_n" if cfg.fuse and role != "tail" else "")   # (fuse: a mid whose panel another fills)
    elif cov == fuse_cover(cfg, -1).get(r):
        name = "mid_l" if len(cov) > 1 else role   # the last mid also fills the tail's slice
    else:
        name = role + "_f" + "".join(map(str, cov)) + ("h" if h is not None else "")
    if fpair(cfg) and c >= 0 and role == "tail":   # (pair_swap: the gate tail's half is its second)
        name += "_xe" if c % 2 else "_xw"
    elif fpair(cfg) and c >= 0 and c % 2 and (fuse_dcov(cfg, c, r) is not None or r in hs):   # (a decoder of ufmt)
        name += "_u"
    return name + ("" if not gate else "_g" + gate[0] if cfg.fuse else "_g"), role, cov, gate


def fuse_ports(role):
    """(egress port, core stream port) of a filling worker: after its A and W ports."""
    return QF_PORTS if QF["cfg"] is not None else (2, 3)


def fpair(cfg):
    """Column pairs (Config.ufmt)."""
    return bool(cfg.ufmt)


def colcfg(cfg, c):
    """Column c's config: a pair's up column decodes ufmt rows in ugord's order."""
    return dataclasses.replace(cfg, fmt=cfg.ufmt, gord=cfg.ugord) if fpair(cfg) and c >= 0 and c % 2 else cfg


def qsw_roll4(cfg):
    """(FFN gate / up epilogue, program memory) the rolled j loop one half an iteration instead of two:
    GN_QSWROLL4=1 / 0, else on for the IQ2 decoders' cores (IQ2_S gate core 16816 B of 16384 with two halves)."""
    v = os.environ.get("GN_QSWROLL4")
    return v == "1" if v in ("0", "1") else cfg.fmt in ("IQ2_S", "IQ2_XS")


def qf_vroll(cfg):
    """The Q' fill's decoder with its scale halves rolled (program memory): GN_QFVROLL=1 (diag), or IQ3_XXS in an
    FFN gate / up image (its gate core 16816 B of 16384 unrolled, 16144 rolled)."""
    return os.environ.get("GN_QFVROLL") == "1" or (bool(QF.get("qsw")) and cfg.fmt == "IQ3_XXS")


def dec_tag(cfg):
    """The inlined decoder's rodata name: a pair's up columns (colcfg) their own."""
    if QF.get("gfmt"):   # (Q', gen_npu_q: the fill leaves see decode Configs) the gate format's "inl", the up's "inlu"
        return "inl" if cfg.fmt == QF["gfmt"] else "inlu"
    return "inlu" if fpair(cfg) and cfg.fmt == cfg.ufmt else "inl"


PAIR_PORT = 3   # pair_swap's ring port on both tails (after A, W and C)
QPAIR_PORT = 12   # (Q' column pairs) qpair_swap's ring port on every core (the P1K image's cores use 0 .. 11)


def pair_port():
    """The pair ring's port: Q' (gen_npu_q sets QF["cfg"]) or the old tails'."""
    return QPAIR_PORT if QF.get("cfg") is not None else PAIR_PORT


def qpair_off():
    """(diag, GN_QPAIRDIAG=nohs) Q' column pairs without qpair_swap: no handshake, no swap (wrong output; a deadlock
    that stays is not the pair rings')."""
    return QF.get("cfg") is not None and os.environ.get("GN_QPAIRDIAG") == "nohs"


def pair_copy(e, y, src, dst, swap=False, nbytes=None):
    """nbytes (default SW_HALF) from scalar address src to dst, unrolled (a loop's branch delays and load waits cost
    ~25 bundles per 256 bytes); swap: both ways (dst's bytes to src). Per KB one base pointer 512 bytes in (the
    immediates reach -512 .. 448), 64-byte loads and stores in groups of four."""
    nb_ = SW_HALF if nbytes is None else nbytes
    def padd(name, src_, off):
        cur, i = src_, 0
        while off > 0:
            step = min(off, 448)
            off -= step
            nxt = name if off == 0 else f"{name}_{i}"
            i += 1
            e(f"  {nxt} = padda {cur}, {step}")
            cur = nxt
        return cur
    e(f"  {y}sa = mov.scalar-to-address {src}")
    e(f"  {y}da = mov.scalar-to-address {dst}")
    assert nb_ % 256 == 0 or nb_ == 128   # (128: qpair_swap's GN_QPAIRCH pieces, one group of two)
    sb, db = f"{y}sa", f"{y}da"
    for g in range(-(-nb_ // 256)):
        k, qs = g // 4, range(4 * g, min(4 * g + 4, nb_ // 64))
        if g % 4 == 0:
            sb = padd(f"{y}sb{k}", sb, 512 if k == 0 else 1024)
            db = padd(f"{y}db{k}", db, 512 if k == 0 else 1024)
        for q in qs:
            e(f"  {y}v{q} = vlda.512.i32x16 {sb}, {64 * (q % 16) - 512}")
            if swap:
                e(f"  {y}u{q} = vldb.512.i32x16 {db}, {64 * (q % 16) - 512}")
        for q in qs:
            e(f"  vst.512.i32x16 {y}v{q}, {db}, {64 * (q % 16) - 512}")
            if swap:
                e(f"  vst.512.i32x16 {y}u{q}, {sb}, {64 * (q % 16) - 512}")


def pair_swap(e, pair, slot, tag):
    """Column pairs: a finished C segment at scalar slot (gate slabs then up slabs of a single format's column) is
    swapped half for half with the other tail's, so each holds a single format column's segment: the gate tail ("w")
    gives its second two slabs (the gate rows of the features the up tail keeps) and takes the up tail's first two (the
    up rows of its own features) in their place. The up tail swaps both halves in place, the gate tail's through its
    west window: the gate tail posts its half's address and its view of the ring's (a ring in its memory, one slot, two
    records a segment) and waits for the second record, which the up tail sends once it swapped; the up tail's view of
    the gate tail's memory is the ring address difference away."""
    x = f"%px{tag}"
    e(f"  {x}sp = storage_address %pxst : low.storage<private> -> reg<aie2p.ep>")
    e(f"  {x}r = lda {x}sp, 0")
    e(f"  acq %am1, {PAIR_PORT}")
    e(f"  {x}rp = mov.scalar-to-address {x}r")
    if pair == "w":
        e(f"  {x}k = mov.i32 {SW_HALF}")
        e(f"  {x}h = add.rr {slot}, {x}k")
        e(f"  st {x}h, {x}rp, 0")
        e(f"  {x}rq = mov.scalar-to-address {x}r")
        e(f"  st {x}r, {x}rq, 4")
        e(f"  rel %one, {PAIR_PORT}")
        e(f"  acq %am1, {PAIR_PORT}")
    else:
        e(f"  {x}wh = lda {x}rp, 0")
        e(f"  {x}rq = mov.scalar-to-address {x}r")
        e(f"  {x}wr = lda {x}rq, 4")
        e(f"  {x}dv = sub {x}r, {x}wr")
        e(f"  {x}src = add.rr {x}wh, {x}dv")
        # (a guard: the half must lie in the ring's 64 KB window, else no swap: wrong output, no stray access)
        e(f"  {x}hk = mov.i32 {SW_HALF - 1}")
        e(f"  {x}end = add.rr {x}src, {x}hk")
        e(f"  {x}x0 = xor {x}src, {x}r")
        e(f"  {x}x1 = xor {x}end, {x}r")
        e(f"  {x}x2 = or {x}x0, {x}x1")
        e(f"  {x}sk = mov.i32 -16")
        e(f"  {x}x3 = lshl {x}x2, {x}sk")
        e(f"  {x}ok = eqz {x}x3")
        e(f"  low.cond_br {x}ok, ^pxd{tag}, ^pxj{tag} : reg<aie2p.er>")
        e(f"^pxd{tag}:")
        pair_copy(e, f"{x}0", f"{x}src", slot, swap=True)
        e(f"  low.br ^pxj{tag}")
        e(f"^pxj{tag}:")
        e(f"  rel %one, {PAIR_PORT}")
        e(f"  acq %am1, {PAIR_PORT}")
    e(f"  rel %one, {PAIR_PORT}")


def pair_park(e, part="both"):
    """Column pairs: the tail's swap ring (one slot) parked as a scalar in private storage (the tail's pointer registers
    are spoken for). part "res": only its resource (resources come before any instruction), "park": the rest."""
    if part != "park":
        e(f"  %pxp = resource<native_pointer> {{index = {pair_port()}, source_type = buffer}} : reg<aie2p.ep>")
    if part == "res":
        return
    e("  %pxst = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
    e("  %pxsp = storage_address %pxst : low.storage<private> -> reg<aie2p.ep>")
    e("  %pxs = mov.address-to-scalar %pxp")
    e("  st %pxs, %pxsp, 0")


def qpair_swap(e, pair, ea, ob, tag):
    """(Q' column pairs, cfg.qsw) one 16-token row's swap before its convert. Gate core (2 u, r), pair "w", holds the
    gate sums of the pair's 64 features: sub-tiles 0 .. 3 at (even, odd, even + 1 KB, odd + 1 KB) of the row (q3
    areas); up core (2 u + 1, r), "e", the up sums of the same features and tokens. The up core swaps its even and odd
    first KB (up sub-tiles 0, 1) with the gate core's even and odd second KB (gate sub-tiles 2, 3) through its west
    window, so each core holds today's qsw row (gate sub-tiles 0, 1, up sub-tiles 2, 3 of its own 32 features: the gate
    core the pair's first 32, the up core the second 32) and runs the qsw convert unchanged. The ring (one slot, two
    records a row) lives in the gate core's memory: the gate posts its even row address and its own view of the ring,
    the up core maps the gate's addresses by the difference of the two views (the C areas are pinned alike in every
    core, so odd - even is the same in both), swaps, frees the slot, and takes the gate's second record."""
    P = pair_port()
    x = f"%qx{tag}"
    e(f"  {x}sp = storage_address %pxst : low.storage<private> -> reg<aie2p.ep>")
    e(f"  {x}r = lda {x}sp, 0")
    e(f"  acq %am1, {P}")
    e(f"  {x}rp = mov.scalar-to-address {x}r")
    if pair == "w":
        e(f"  {x}h = mov.address-to-scalar {ea}")
        e(f"  {x}ho = mov.address-to-scalar {ob}")
        e(f"  st {x}h, {x}rp, 0")
        e(f"  st {x}r, {x}rp, 4")
        e(f"  st {x}ho, {x}rp, 8")   # (its odd row too: the two leaves' memory maps differ, their decoders do)
        e(f"  rel %one, {P}")
        e(f"  acq %am1, {P}")
    else:
        e(f"  {x}wh = lda {x}rp, 0")
        e(f"  {x}wr = lda {x}rp, 4")
        e(f"  {x}wo = lda {x}rp, 8")
        e(f"  {x}dv = sub {x}r, {x}wr")
        e(f"  {x}g0 = add.rr {x}wh, {x}dv")
        e(f"  {x}k1 = mov.i32 1024")
        e(f"  {x}se = add.rr {x}g0, {x}k1")
        e(f"  {x}ea = mov.address-to-scalar {ea}")
        e(f"  {x}ob = mov.address-to-scalar {ob}")
        e(f"  {x}eo = sub {x}ob, {x}ea")
        e(f"  {x}go = sub {x}wo, {x}wh")         # (the gate's odd - even)
        e(f"  {x}so = add.rr {x}se, {x}go")
        # (a guard: both gate pieces inside the ring's 64 KB window, else no swap: wrong output, no stray access;
        # GN_QPAIRGUARD=mask: each piece's gate address forced into the window instead, ch-aligned, less code)
        gmask = os.environ.get("GN_QPAIRGUARD", "mask") == "mask"
        if gmask:
            e(f"  {x}wk = mov.i32 -65536")
            e(f"  {x}wb = and {x}r, {x}wk")
            e(f"  low.br ^qxd{tag}")
        else:
            e(f"  {x}hk = mov.i32 1023")
            e(f"  {x}end = add.rr {x}so, {x}hk")
            e(f"  {x}x0 = xor {x}se, {x}r")
            e(f"  {x}x1 = xor {x}end, {x}r")
            e(f"  {x}x2 = or {x}x0, {x}x1")
            e(f"  {x}sk = mov.i32 -16")
            e(f"  {x}x3 = lshl {x}x2, {x}sk")
            e(f"  {x}ok = eqz {x}x3")
            e(f"  low.cond_br {x}ok, ^qxd{tag}, ^qxj{tag} : reg<aie2p.er>")
        e(f"^qxd{tag}:")   # (8 pieces of 256 B: 4 of the even KB, then 4 of the odd one odd - even on; a loop, the
        # copy's code once: the program memory is full, 16 KB)
        ER = "reg<aie2p.er>"
        ch = int(os.environ.get("GN_QPAIRCH", "128"))   # (the copy's piece: 128 B, 16 pieces, halves its code)
        assert ch in (128, 256)
        npc, lpc = 1024 // ch, (1024 // ch).bit_length() - 1
        e(f"  {x}i0 = mova.i32 0")
        e(f"  {x}k8 = mova.i32 {2 * npc}")
        e(f"  {x}k3 = mova.i32 {npc - 1}")
        e(f"  {x}km2 = mova.i32 {-lpc}")
        e(f"  {x}kc = mova.i32 {ch.bit_length() - 1}")
        if gmask:
            e(f"  {x}lk = mov.i32 {0xFFFF & -ch}")
        e(f"  low.br ^qxl{tag}({x}i0: {ER})")
        e(f"^qxl{tag}({x}li: {ER}):")
        e(f"  {x}lm = lt {x}li, {x}k8")
        # (the step in the header: no hardware loop candidate, so LOOM_EXP_HWLOOP_TRIP's '@' counts stay the GEMM's)
        e(f"  {x}ln = add {x}li, 1")
        e(f"  low.cond_br {x}lm, ^qxb{tag}, ^qxj{tag} : {ER}")
        e(f"^qxb{tag}:")
        e(f"  {x}pc = lshl {x}li, {x}km2")      # (the KB: i >> 2, 128 B pieces i >> 3)
        e(f"  {x}po = mul {x}pc, {x}eo")
        e(f"  {x}pg = mul {x}pc, {x}go")
        e(f"  {x}qa = and {x}li, {x}k3")
        e(f"  {x}qo = lshl {x}qa, {x}kc")      # ((i & 3) << 8; 128 B pieces (i & 7) << 7)
        e(f"  {x}oo = add.rr {x}po, {x}qo")
        e(f"  {x}og = add.rr {x}pg, {x}qo")
        e(f"  {x}ls = add.rr {x}se, {x}og")
        if gmask:
            e(f"  {x}lw = and {x}ls, {x}lk")
            e(f"  {x}lv = or {x}lw, {x}wb")
        e(f"  {x}ld = add.rr {x}ea, {x}oo")
        pair_copy(e, f"{x}c", f"{x}lv" if gmask else f"{x}ls", f"{x}ld", swap=True, nbytes=ch)
        e(f"  low.br ^qxl{tag}({x}ln: {ER})")
        e(f"^qxj{tag}:")
        e(f"  rel %one, {P}")
        e(f"  acq %am1, {P}")
    e(f"  rel %one, {P}")


def fuse_groups(cfg):
    """The panels' replay groups as pass ranges."""
    g = groups(cfg) or (cfg.passes,)
    out, p = [], 0
    for n in g:
        out.append(range(p, p + n))
        p += n
    return out


def fuse_cover(cfg, c):
    """Column c's filling rows: {filler row: the rows whose slices it fills, consecutive rows in panel member order
    (their panels share its fill, interleaved per unit)}. Rows 0 .. rows - 2 fill their own, the second-to-last also
    the last row's. The gate column (fuse 2) has one filler, its head, for every row: four streams at most cross each
    downward link, and the gate's tick and signal plus C leave room for a single egress. gwest splits the gate over
    columns 1 and 2 (one gate stream each): their heads fill rows 1 and 0, their rows 2 rows 3 and 2. fuse 2 lists the filler last:
    a member's drain starts once its slice of the group's last unit lands, and the filler's drain overwrites its W ring,
    the decoder's scratch, which the other members' slices are still copied from."""
    rows = len(cfg.ks)
    if cfg.fuse == 2 and cfg.gate and cfg.gwest and c in (1, 2):   # the gate's rows 1: two fillers, head and row 2
        return {0: (0, 1), rows - 2: (rows - 2, rows - 1)} if cfg.kchain else {0: (1, 0), rows - 2: (rows - 1, rows - 2)}
    assert not (cfg.kchain and cfg.gate and cfg.fuse == 2 and c == gate_ports(cfg)["col"]), "kchain: gwest only"
    if cfg.fuse == 2 and cfg.gate and c == gate_ports(cfg)["col"]:
        return {0: tuple(range(1, rows)) + (0,)}
    last = (rows - 1, rows - 2) if cfg.fuse == 2 and not cfg.kchain else (rows - 2, rows - 1)   # (kchain: the tail's
    # member last, its drain only after the tail's last record: the tail's piece for the last mid goes first)
    return {r: (last if r == rows - 2 else (r,)) for r in range(rows - 1)}


def fuse_order(cfg, c):
    """Column c's panel slices in W binding order (the memory-tile panel members' order: fuse_cover)."""
    if cfg.fuse != 2:
        return list(range(len(cfg.ks)))
    return [r for _, cov in sorted(fuse_cover(cfg, c).items()) for r in cov]


def fuse_filler(cfg, c, r):
    """(filler row, its cover) of row r's panel in column c."""
    return next((f, cov) for f, cov in fuse_cover(cfg, c).items() if r in cov)


HELP_RAW, HELP_SEND, HELP_RECV = 5, 6, 4   # ghelp ports: the helper's raw intake and record sender, the head's receiver
HELP_RELAY = 7   # kchain: a relaying helper's receiver (the gate waiter's neighbor port is 4)


def fuse_helpers(cfg, c):
    """Column c's helpers {helper row: the helped filler's row}: ghelp's row 1 (the head) and thelp's tail (the last
    mid, whose cover lists the tail's row first like the head's (1, 0))."""
    out = {}
    h = fuse_helper(cfg, c)
    if h is not None:
        out[h] = 0
    rows = len(cfg.ks)
    if cfg.fuse == 2 and cfg.kchain and c >= 0:   # rows sending up: those without an egress (their slice too), else
        # those whose super-block starts before their slice
        assert not cfg.thelp and cfg.rawdma and not cfg.swiglu
        kchain_check(cfg)
        cov = fuse_cover(cfg, c)
        return {r: r - 1 for r in range(1, rows) if r not in cov or kchain_piece(cfg, r) > 0}
    if cfg.fuse == 2 and cfg.thelp and c >= 0 and fuse_cover(cfg, c).get(rows - 2) == (rows - 1, rows - 2):
        assert cfg.rawdma and not cfg.swiglu
        out[rows - 1] = rows - 2
    return out


def kchain_piece(cfg, r):
    """kchain: the k-blocks row r (> 0) sends the row above: its super-block's before its slice."""
    return sum(cfg.ks[:r]) - 32 * r


def kchain_check(cfg):
    assert len(cfg.ks) == 4 and sum(cfg.ks) == 128
    assert all(0 <= kchain_piece(cfg, r) < 32 and kchain_piece(cfg, r) % 4 == 0 for r in range(1, 4)), cfg.ks


def kchain_send(cfg, r, pure):
    """kchain: row r's pieces for the row above [(first k-block, words)]: the piece of the row above's slice, then (pure:
    a row without an egress, whose slice the row above's cover lists after its own) its own slice."""
    # (one piece: the slice follows the piece in the decoder's scratch, one copy loop; the rest: kchain_relay)
    n = kchain_piece(cfg, r) + (min(sum(cfg.ks[:r + 1]), 32 * r + 32) - sum(cfg.ks[:r]) if pure else 0)
    return [(32 * r, 36 * n)] if n else []


def kchain_relay(cfg, r):
    """kchain: the k-blocks of row r's slice past its super-block (a row without an egress relays them from below)."""
    return max(0, sum(cfg.ks[:r + 1]) - 32 * r - 32)


def kchain_role(cfg, c, r):
    """kchain: worker (c, r)'s fill role: "M" (no egress: sends up), else "K" + "u" (sends up) + "h" (helped)."""
    cov, hs = fuse_cover(cfg, c), fuse_helpers(cfg, c)
    if r not in cov:   # ("Mh": relays the row below's piece after its own slice)
        return "M" + ("h" if hs.get(r + 1) == r else "")
    return "K" + ("u" if r in hs else "") + ("h" if hs.get(r + 1) == r else "")


def fuse_helper(cfg, c):
    """ghelp: column c's helper row, else None. The column's head fills rows (1, 0) but decodes only row 0; row 1's core
    (no egress: four streams at most cross each downward link) takes the column's raw intake too, decodes row 1 like a
    filler and sends its records to the head over a leaf-synced neighbor channel instead of an egress, and the head
    copies them into its egress ahead of row 0's."""
    if cfg.fuse == 2 and cfg.ghelp and c >= 0 and fuse_cover(cfg, c).get(0) == (1, 0):
        assert cfg.rawdma
        return 1
    return None


def fuse_hx(cfg):
    """hsplit: the head's k-blocks the helper decodes and sends after its own slice (0: none)."""
    if cfg.kchain:
        return 0
    x = cfg.ks[0] % 32
    return x if cfg.fuse == 2 and cfg.ghelp and cfg.hsplit and cfg.ks[0] >= 32 else 0


def fuse_dcov(cfg, c, r):
    """The rows worker (c, r) decodes (fuse_cover less a helper's row; None: it decodes nothing)."""
    cov, hs = fuse_cover(cfg, c).get(r), fuse_helpers(cfg, c)
    if r in hs:
        return (r,)
    return tuple(j for j in cov if j not in hs) if cov is not None else None


def fuse_record(cfg, cov):
    """(egress record words, records per call) of a filler of the rows cov. Records are whole 4-word steps dividing
    every replay group's fill."""
    unit = NP * sum(slab(cfg.ks[j]) for j in cov) // 4
    # whole 64-byte copies, whole records per slab
    # (swiglu: not C's record size, LOOM_EXP_LS_SEND_PITCH_RECORD would pitch the egress ring like C's)
    xs = [36 * fuse_hx(cfg)] if cov == (1, 0) and fuse_hx(cfg) else []   # (hsplit: the helper's extra piece)
    if cfg.kchain:
        xs += [36 * kchain_piece(cfg, r) for r in range(1, len(cfg.ks)) if kchain_piece(cfg, r)]
    fw = max(w for w in range(16, FUSE_REC + 1, 16) if all(sw % w == 0 for sw in [sw for _, sw in fuse_pieces(cfg, cov)] + xs)
             and not (cfg.swiglu and 4 * w == SW_REC))
    if cfg is QF["cfg"] or (QF.get("cfgu") is not None and cfg is QF["cfgu"]):   # (Q': 4 records a 16-k-block
        # piece; P1K direct: half a slab's output; column pairs: the up format's decode config too)
        fw = QF_P1KD["rec"] // 4 if qfdirect_on() else QF_FW
    return fw, cfg.passes * unit // fw


FUSE_SB = 4   # fuse 2: super-blocks per row of a unit (a pass: 1024 k)
FUSE_FMTS = ("IQ4_XS", "IQ4_NL", "Q3_K", "IQ3_XXS", "IQ3_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "Q4_K", "Q5_K", "Q2_K", "BFP")   # fuse 2 decoders (cfg.fmt, in
# cfg.gord's k-block order; program and data memory decide which fit an image: gen_npu_dec sizes)


def qf_odd2(cfg):
    """(Q' P1K direct fill, one super-block a raw record) cfg.fmt's super-blocks are 2 mod 4 bytes: the raw stream reads
    them in pairs (gen_npu_q), each odd one from the word before it, and the fill starts its decode 2 bytes on."""
    b = fuse_blk(cfg)
    assert b % 2 == 0, b
    return b % 4 == 2 and not qfraw4_on()


def fuse_blk(cfg):
    """fuse 2: bytes per super-block of cfg.fmt."""
    assert cfg.fmt in FUSE_FMTS, cfg.fmt
    return decoder(cfg.fmt, cfg.gord, cfg.swiglu).BLK


def fuse_unit_raw(cfg):
    """fuse 2: a (pass, slab) unit's raw bytes: [16 rows][4 super-blocks]."""
    return 16 * FUSE_SB * fuse_blk(cfg)


def fuse_span(cfg, cov):
    """fuse 2, a filler of the rows cov: (k-block range k0, k1 of a pass; first super-block, super-block count)."""
    k0, k1 = sum(cfg.ks[:min(cov)]), sum(cfg.ks[:max(cov) + 1])
    if cfg.kchain and len(cov) == 1:   # (a decoding core: its super-block)
        k0, k1 = 32 * cov[0], 32 * cov[0] + 32
    s0, s1 = k0 // 32, -(-k1 // 32)
    return k0, k1, s0, s1 - s0


def fuse_pieces(cfg, cov):
    """fuse 2, a filler of the rows cov: per slice of its panel unit, (first k-block, padded slab words)."""
    out = [(sum(cfg.ks[:j]), slab(cfg.ks[j]) // 4) for j in cov]
    assert all(sw % 16 == 0 for _, sw in out), "fuse 2: slabs of whole 64-byte copies"
    return out


_HLATE = []   # Config.hlate: fuse_channels' helper-ring lines, emitted by array_program after the A rings


def fuse_channels(e, cfg, panel):
    """Per filling worker (fuse_cover): its leaf-synced egress ring (the source of its panel's fill; the dummy
    write binding, a lane per worker, is never transferred), and its column's raw stream, multicast over the column's
    filling core streams."""
    rows, cols = len(cfg.ks), cfg.cols
    units = cfg.passes * NP
    e(f"  %fdb = binding {6 if cfg.gate else 3}, \"write\"")
    e(f"  %frb = binding {7 if cfg.gate else 4}, \"read\"")
    e(f"  %fdl = constant.u32 {rows * cols} : reg<aie2p.array.scalar : index>")
    e(f"  %fun = constant.u32 {units} : reg<aie2p.array.scalar : index>")
    # raw rows [N][kraw / 256][136]: per column, per (pass, slab) unit [16 rows][4 super-blocks]
    rstride = lambda cc: (cfg.kraw or 8 * cfg.passes * sum(cfg.ks)) // 256 * fuse_blk(cc) // 4   # row stride in words
    if fpair(cfg):   # pair u: [two feature groups (the gate tails' columns 2 u, 2 u + 1 of a single format's image)]
        # [NP / 2 slabs]; the gate column's rows from %frb, the up column's (ufmt) from their own binding
        assert cfg.swiglu and cfg.dfr and NP == 4 and cfg.hord in ("P4", "Q4K")
        e(f"  %frbu = binding {8 if cfg.gate else 5}, \"read\"")
        for up in (0, 1):
            cc = colcfg(cfg, up)
            BLK, RS, uw = fuse_blk(cc), rstride(cc), fuse_unit_raw(cc) // 4
            gs, ss = (16, 32) if cfg.hord == "Q4K" else (cfg.swcol, 16)
            e(f"  %fru_all{up} = sender {'%frbu' if up else '%frb'}, 0 : reg<aie2p.array.sender : tile<{cfg.passes}x2x{NP // 2}x16x{uw // 16}xi32, "
              f"#encoding.layout.strided<strides=[{FUSE_SB * BLK // 4}, {gs * RS}, {ss * RS}, {RS}, 1]>>>")
    elif cfg.swiglu:   # column c: gate rows swcol c .. (its NP / 2 slabs), then the same up rows dgu bytes on; Q4K:
        # its 2 slabs 32 rows apart (fuse_rows)
        assert cfg.dgu % 4 == 0 and NP % 2 == 0 and cfg.hord in ("", "P4", "Q4K")
        BLK, RS, uw = fuse_blk(cfg), rstride(cfg), fuse_unit_raw(cfg) // 4
        e(f"  %fru_all = sender %frb, 0 : reg<aie2p.array.sender : tile<{cfg.passes}x2x{NP // 2}x16x{uw // 16}xi32, "
          f"#encoding.layout.strided<strides=[{FUSE_SB * BLK // 4}, {cfg.dgu // 4}, "
          f"{(32 if cfg.hord == 'Q4K' else 16) * RS}, {RS}, 1]>>>")
    else:
        BLK, RS, uw = fuse_blk(cfg), rstride(cfg), fuse_unit_raw(cfg) // 4
        e(f"  %fru_all = sender %frb, 0 : reg<aie2p.array.sender : tile<{cfg.passes}x{NP}x16x{uw // 16}xi32, "
          f"#encoding.layout.strided<strides=[{FUSE_SB * BLK // 4}, {16 * RS}, {RS}, 1]>>>")
    covs = list(fuse_cover(cfg, -1).values())   # an ordinary column's covers, then any other column's
    for cov in (cov for c in range(cols) for cov in fuse_cover(cfg, c).values()):
        covs += [cov] if cov not in covs else []
    tag = lambda cov: str(covs.index(cov)) if cov in covs[:rows - 1] else "f" + "".join(map(str, cov))
    for cov in covs:
        fw, nrec = fuse_record(cfg, cov)
        e(f"  %fdn{tag(cov)} = constant.u32 {nrec} : reg<aie2p.array.scalar : index>")
        e(f"  %fd_all{tag(cov)} = receiver %fdb, 0 : reg<aie2p.array.receiver : tile<{rows * cols}x{nrec}x{fw}xi32>>")
    for c in range(cols):
        # column c: rows TN c .. (swiglu: gate rows swcol c ..; Q4K: 64 (c / 2) + 16 (c % 2) ..; pairs: pair c / 2's
        # first gate rows, the up column's in its binding)
        cc = colcfg(cfg, c)
        uw = fuse_unit_raw(cc) // 4
        if fpair(cfg):
            r0 = 64 * (c // 2) if cfg.hord == "Q4K" else 2 * (c // 2) * cfg.swcol
            e(f"  %fro{c} = constant.u64 {r0 * rstride(cc) * 4} : reg<aie2p.array.offset : offset>")
            e(f"  %fru{c} = view.sender %fru_all{c % 2}, %fro{c} : reg<aie2p.array.sender : tile<16x{uw // 16}xi32>>")
        else:
            r0 = (64 * (c // 2) + 16 * (c % 2) if cfg.swiglu and cfg.hord == "Q4K" else
                  c * (cfg.swcol if cfg.swiglu else TN))
            e(f"  %fro{c} = constant.u64 {r0 * RS * 4} : reg<aie2p.array.offset : offset>")
            e(f"  %fru{c} = view.sender %fru_all, %fro{c} : reg<aie2p.array.sender : tile<16x{uw // 16}xi32>>")
        for r, cov in fuse_cover(cfg, c).items():
            dp, sp = fuse_ports("")
            fw, nrec = fuse_record(cfg, cov)
            i = tag(cov)
            e(f"  %fdr{c}_{r} = partition.receiver %fd_all{i}, %origin, %lane{c}_{r}, %fdl : reg<aie2p.array.receiver : tile<{fw}xi32>>")
            e(f"  %fds{c}_{r} = sender %k{c}_{r}, {dp} : reg<aie2p.array.sender : tile<{fw}xi32>>")
            e(f"  %chd{c}_{r} = channel %fds{c}_{r}, %fdr{c}_{r}, %two, %fdn{i} : reg<aie2p.array.channel : tile<{fw}xi32>>")
            e(f"  constrain.leaf_sync %chd{c}_{r}")
            ut = f"16x{uw // 16}"
            e(f"  %frr{c}_{r} = receiver %k{c}_{r}, {sp} : reg<aie2p.array.receiver : tile<{ut}xi32>>")
            if cfg.rawdma:   # staged in the column's memory tile; intake ring depth fuse_raw_ring
                cap = fuse_raw_ring(cc, fuse_dcov(cfg, c, r))[1]
                e(f"  %chr{c}_{r} = channel %fru{c}, %frr{c}_{r}, %n{cap}, %fun : reg<aie2p.array.channel : tile<{ut}xi32>>")
                e(f"  constrain.stage %chr{c}_{r}, %n{c if cols > 1 else 1}")
                e(f"  constrain.leaf_sync %chr{c}_{r}")
            else:
                e(f"  %chr{c}_{r} = channel %fru{c}, %frr{c}_{r}, %two, %fun : reg<aie2p.array.channel : tile<{ut}xi32>>")
                e(f"  constrain.core_stream %chr{c}_{r}")
        for h, hd in fuse_helpers(cfg, c).items():   # a helper: its raw intake, its records to the helped filler
            # (the filler's record size)
            s = "" if hd == 0 else f"_{h}"
            ut = f"16x{uw // 16}"
            fw, _ = fuse_record(cfg, fuse_cover(cfg, c).get(hd) or (hd,))   # (kchain: a relaying row has no cover)
            hx = fuse_hx(cfg) if hd == 0 and not cfg.kchain else 0
            pure = h not in fuse_cover(cfg, c)
            words = (sum(w for _, w in kchain_send(cfg, h, pure)) + 36 * (kchain_relay(cfg, h) if pure else 0)
                     if cfg.kchain else slab(cfg.ks[h]) // 4 + 36 * hx)
            assert words % fw == 0
            nrec = cfg.passes * NP * words // fw
            if h not in fuse_cover(cfg, c):   # (kchain: a filler's raw arrives through its own intake)
                e(f"  %frr{c}_{h} = receiver %k{c}_{h}, {HELP_RAW} : reg<aie2p.array.receiver : tile<{ut}xi32>>")
                cap = fuse_raw_ring(cc, (h,))[1]
                e(f"  %chr{c}_{h} = channel %fru{c}, %frr{c}_{h}, %n{cap}, %fun : reg<aie2p.array.channel : tile<{ut}xi32>>")
                e(f"  constrain.stage %chr{c}_{h}, %n{c if cols > 1 else 1}")
                e(f"  constrain.leaf_sync %chr{c}_{h}")
            eh = _HLATE.append if cfg.hlate else e   # (hlate: array_program emits them after the A rings)
            eh(f"  %fhn{c}{s} = constant.u32 {nrec} : reg<aie2p.array.scalar : index>")
            eh(f"  %fhs{c}{s} = sender %k{c}_{h}, {HELP_SEND} : reg<aie2p.array.sender : tile<{fw}xi32>>")
            rp = HELP_RELAY if hd not in fuse_cover(cfg, c) else HELP_RECV   # (kchain: a relaying row's receiver)
            eh(f"  %fhr{c}{s} = receiver %k{c}_{hd}, {rp} : reg<aie2p.array.receiver : tile<{fw}xi32>>")
            assert cfg.hcap & (cfg.hcap - 1) == 0   # (slots: record count & (hcap - 1))
            eh(f"  %chh{c}{s} = channel %fhs{c}{s}, %fhr{c}{s}, %n{cfg.hcap}, %fhn{c}{s} : reg<aie2p.array.channel : tile<{fw}xi32>>")
            eh(f"  constrain.leaf_sync %chh{c}{s}")


def fuse_prologue(L, cfg, role, ks, cov, helper=None, gate=None):
    """A filling worker's firing (row r < rows - 1): the firing counter (private), the W slot by parity (leaf-synced
    W, acquired here and released at the exit); at each replay group's first firing (firings run group, M block,
    pass), that group's fill: per unit, skip the rows before its own, take its slab (row rows - 2 also the last
    row's: their panel is interleaved per unit), skip the rest (pass-through). A group's fill ends on a whole record;
    the record count runs on across groups and calls (the egress ring's slot parity)."""
    e = L.append
    rows = len(cfg.ks)
    dp = HELP_SEND if (helper or "")[:1] == "M" else fuse_ports(role)[0]
    wrec = NP * slab(ks)
    e("  %fzs = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
    e("  %fzp = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    fuse2_entry(L, cfg, dp, cov, wrec, helper, gate, tail=role == "tail")


def fuse2_entry(L, cfg, dp, cov, wrec, helper=None, gate=None, tail=False):
    """fuse 2's entry: the firing counter and W slot parity, the leaf's pointers and the W slot offset parked in
    private storage (nothing lives across the decoding fill: it needs every register), the group dispatch, the fill
    (fuse_decode), then ^fzgemm rebuilds the pointers (W is acquired after the GEMM's constants)."""
    e = L.append
    if QFCYC:
        e("  %qfc = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
    if gate and qprefill_on(QF["cfg"]):   # (GN_QPREFILL) the gate's neighbor, parked across the fill
        e("  %gnk = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
    e("  %fzc = lda %fzp, 0")
    e("  %fzc1 = add %fzc, 1")
    q = QF["cfg"]
    # (Q': a filling core fires once per M block of its row, nb / rows a call)
    e(f"  %fzcl = mov.i32 {cfg.nb // QF.get('rows', 1) if q is not None else cfg.nb * cfg.passes}")
    e("  %fzcw = lt %fzc1, %fzcl")
    e("  %fzcn = mul %fzc1, %fzcw")
    e("  st %fzcn, %fzp, 0")
    e("  %fzc1o = sub %fzc1, %fzc")                     # 1, not a constant
    e("  %fzpar = and %fzc, %fzc1o")
    e(f"  %fzwr = mov.i32 {0 if q is not None else wrec}")
    e("  %fzwo = mul %fzpar, %fzwr")
    for o, nm in ((20, "%a_r"), (24, "%w0_r"), (28, "%fzd_r")):
        e(f"  %fzps{o} = mov.address-to-scalar {nm}")
        e(f"  st %fzps{o}, %fzp, {o}")
    if QF4R[0] and not QF4R[0]["help"]:   # (GN_QF4R) the helper's ring (word 0) and its record count (word 4)
        e("  %f4s = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
        e("  %f4sp = storage_address %f4s : low.storage<private> -> reg<aie2p.ep>")
        e("  %f4nr = mov.address-to-scalar %fzn_r")
        e("  st %f4nr, %f4sp, 0")
    if tail:   # (thelp: the tail's C ring, parked in a word of its own)
        e("  %fzt = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
        e("  %fztp = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
        e("  %fzts = mov.address-to-scalar %o_r")
        e("  st %fzts, %fztp, 0")
        if QF.get("crel"):   # (its own word: the Q' fill counts in %fzt's words 4-12)
            e("  %crst = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
            e("  %crstp = storage_address %crst : low.storage<private> -> reg<aie2p.ep>")
            e("  %fzcr = mov.address-to-scalar %crb_r")
            e("  st %fzcr, %crstp, 0")
            if QF.get("split"):
                e("  %fzqs = mov.address-to-scalar %qsr_r")
                e("  st %fzqs, %crstp, 4")
    hlp = helper == "H" or "h" in (helper or "")       # (kchain "K..h": helped too)
    # %fzx: 0 the helper's ring, 4 the helper record count (the head), 8 the gate's neighbor, (kchain) 12 the ring to
    # the row above, 16 its record count
    if hlp or (helper or "").startswith("K"):
        e("  %fzx = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
        e("  %fzxp = storage_address %fzx : low.storage<private> -> reg<aie2p.ep>")
        for o, nm in ((((0, "%fzh_r"),) if hlp else ()) +
                      (((12, "%fzq_r"),) if "u" in (helper or "") else ())):
            e(f"  %fzxs{o} = mov.address-to-scalar {nm}")
            e(f"  st %fzxs{o}, %fzxp, {o}")
    e("  st %fzwo, %fzp, 4")
    e("  %fzcf = mov.i32 780")                         # the GEMM's MMA config, loaded after the fill: as a constant,
    e("  st %fzcf, %fzp, 16")                          # the allocator's pressure repairs (for the decoder) cloned it at
                                                       # each of its 720 MMA uses (+2.3 KB of program)
    starts, s0 = [], 0
    for g in ([range(q.passes)] if q is not None else fuse_groups(cfg)):
        starts.append(s0)
        s0 += cfg.nb * len(g)
    for i, st in enumerate(starts):   # firing == st: group i's fill
        last = i + 1 == len(starts)
        if st:
            e(f"  %fzbl{i} = mov.i32 {st}")
            e(f"  %fzml{i} = lt %fzc, %fzbl{i}")
            e(f"  low.cond_br %fzml{i}, ^fzgemm, ^fzck{i} : reg<aie2p.er>")
            e(f"^fzck{i}:")
            e(f"  %fzbh{i} = add %fzbl{i}, 1")
            e(f"  %fzmh{i} = lt %fzc, %fzbh{i}")
        else:
            e(f"  %fzmh{i} = eqz %fzc")
        e(f"  low.cond_br %fzmh{i}, ^fzgo{i}, {'^fzgemm' if last else f'^fzne{i}'} : reg<aie2p.er>")
        if not last:
            e(f"^fzne{i}:")
    for i, g in enumerate([range(q.passes)] if q is not None else fuse_groups(cfg)):   # (Q': one fill a call)
        e(f"^fzgo{i}:")
        if i == 0 and q is not None and os.environ.get("GN_QFGUARD") == "1":
            # (GN_QFGUARD=1) the fill's rings overlay C slots 1-3 (the helper's records, the egress ring): wait (an
            # acquire-equal) until the last call's C left the slots. Only a queue depth 2 (the next call's tasks
            # queued while this one runs) can start a fill before that.
            e(f"  %fzgd = mova.i32 {int(os.environ.get('GN_QFGUARD_N', MP))}")
            e("  acq %fzgd, 2")
        if QFCYC and q is not None:
            e("  %qfpz = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
            e("  %qfz = mova.i32 0")
            e("  st %qfz, %qfpz, 4")
            e("  st %qfz, %qfpz, 24")
        if QFCYC and q is not None:
            e("  %qfpgo = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
            e("  %qfwgo = mov.cycles")
            e("  %qflgo = slice %qfwgo[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
            e("  st %qflgo, %qfpgo, 8")
        if i == 0 and gate and qprefill_on(q):   # (GN_QPREFILL) the gate after the fill: its neighbor parked
            e("  %gsq = storage_address %gnk : low.storage<private> -> reg<aie2p.ep>")
            e("  %gnqs = mov.address-to-scalar %gn1_r")
            e("  st %gnqs, %gsq, 0")
        elif i == 0 and gate:   # (a call's first firing: the check costs once per call)
            e("  %gsp0 = storage_address %gst : low.storage<private> -> reg<aie2p.ep>")
            if QFCYC and os.environ.get("GN_QFW") == "2":   # (diag) the gate check's cycles: %qfc byte 44
                L0_ = len(L)
                gate_first(e, cfg, gate, "%gn1_r")
                body_ = L[L0_:]
                del L[L0_:]
                qfw_wrap(e, "g", 44, body_, raw=True)
            else:
                gate_first(e, cfg, gate, "%gn1_r")
        e(f"  %fzgn{i} = mov.i32 {len(g) * ((1 if qfraw4_on() else FUSE_SB) if q is not None else NP)}")   # (Q': super-block steps)
        e(f"  low.br ^fzdq(%fzgn{i}: reg<aie2p.er>)")
    if q is not None:
        QF["gate_now"] = gate   # (the split channel's port)
        qf_decode(L, q, dp)
    else:
        fuse_decode(L, cfg, dp, cov, helper)
    e("^fzgemm:")
    if gate and qprefill_on(q):   # (GN_QPREFILL) every firing; only a job's first (state zero) waits
        e("  %gsp0 = storage_address %gst : low.storage<private> -> reg<aie2p.ep>")
        e("  %gsk = storage_address %gnk : low.storage<private> -> reg<aie2p.ep>")
        e("  %gnql = lda %gsk, 0")
        e("  %gn1q = mov.scalar-to-address %gnql")
        gate_first(e, cfg, gate, "%gn1q")
    e("  %fzpG = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    for o, nm in ((20, "a"), (24, "w0"), (28, "fzd")):
        e(f"  %fzpl{o} = lda %fzpG, {o}")
        e(f"  %{nm} = mov.scalar-to-address %fzpl{o}")
    if tail:
        e("  %fztG = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
        e("  %fztl = lda %fztG, 0")
        e("  %o = mov.scalar-to-address %fztl")
    e("  %fzwo2 = lda %fzpG, 4")
    e("  %fzwm = mov.modifier %fzwo2")
    e("  %fzw0c = copy %w0 : reg<aie2p.ep> -> reg<aie2p.ep>")
    e("  %w = padds.modifier %fzw0c, %fzwm")


FZ_SP, FZ_IN, FZ_SP2, FZ_OUT = 0, 3264, 16384, 23616   # fuse 2 scratch (from the W ring's first 2048-aligned byte):
# decoder storage A (SCR_B), its input record ([16][RECROW]), storage B (SCR2_B, another bank: the table's second
# copy), its output (2 OUT_B records a super-block)
FZ_COMPACT = 0, 28992, 21760, 3264   # (SP, IN, SP2, OUT) where a ring is too small for that (NP 4, 4 super-blocks):
# the output right after storage A, then B (still a bank apart from A) and the input record


FZ_RAW = 0, 3264, 21760, 28992   # rawdma: (SP, IN, SP2, OUT), IN two raw records (fuse_raw_ring)
FZ_SP2_OWN = -1                     # fuse_scratch: storage B outside the W ring (the worker's private %fzs2)
FZ_RAW_COMPACT = 0, 28992, 21760, 3264   # rawdma, 4 super-blocks a row: one raw record (the compact layout)


def fuse_raw_ring(cfg, cov):
    """rawdma: (byte offset from the W ring's first 2048-aligned byte, records) of the filler's raw ring: two
    [16][RECROW] records where the decoder output allows (FZ_RAW, Q5_K's packed layout), else one."""
    D = decoder(cfg.fmt, cfg.gord, cfg.swiglu)
    sp, in_, sp2, out = fuse_scratch(cfg, cov)
    if QF["cfg"] is not None:   # (Q': W slot 0 holds one raw record of one super-block)
        return in_, 1
    return in_, (2 if sp2 - in_ >= 2 * 16 * D.RECROW else 1)


def fuse_scratch(cfg, cov):
    """fuse 2: the filler's scratch offsets (SP, IN, SP2, OUT) in its W ring (its 2 NP slabs)."""
    D = decoder(cfg.fmt, cfg.gord, cfg.swiglu)
    if qfdirect_on():   # (P1K fill, direct: QF_P1KD; NP 4: P1K4_SP)
        m = dict(QF_P1KD, sp=P1K4_SP) if p1k4_on() else QF_P1KD
        assert not p1k4_on() or P1K4_SP + D.SCR_B <= P1K_GRID_SPARE[1][0], D.SCR_B
        assert not (p1k4_on() and qgmap_on()) or P1K4_SP + D.SCR_B <= QG_SILU[1][1], D.SCR_B   # (the silu piece)
        assert 16 * D.RECROW <= QF["wslot"] and (m["sp"] + D.SCR_B <= 20480 or 25088 <= m["sp"] <= 30720 - D.SCR_B or 49664 <= m["sp"] <= 53760 - D.SCR_B or m["sp"] >= m["sp2"] + D.SCR2_B) and m["sp2"] + D.SCR2_B <= 65280 and m["sp"] + D.SCR_B <= 65280
        assert m["raw"] * 16 * D.RECROW <= 20480   # (the intake slots: W slot 0, its tail, the odd C area)
        return m["sp"], 0, m["sp2"], None
    if QF["cfg"] is not None and qp1k_on():   # (P1K fill: QF_P1K)
        m = QF_P1K
        assert 16 * D.RECROW <= QF["wslot"] and D.SCR_B <= 20480 - m["sp"] and D.SCR2_B <= m["outs"][2] - m["sp2"]
        return m["sp"], 0, m["sp2"], m["outs"][0]
    if QF["cfg"] is not None:   # Q' (QA4 pins, W slot 0 at 0): areas idle until the panel's replay
        ws = QF["wslot"]   # IN: W slot 0; SP: the C rows (bank 1); SP2: the odd-C block; OUT: QF_OUTS (one per slab)
        assert 16 * D.RECROW <= ws and 16384 + D.SCR_B <= QF_OUTS[0] and QF_OUTS[0] + 2 * D.OUT_B <= 24576, ws
        assert QF_OUTS[2] + 2 * D.OUT_B <= 49152 + ws and D.SCR2_B <= 8192, ws
        return 16384, 0, 40960, QF_OUTS[0]
    _, _, _, nsb = fuse_span(cfg, cov)
    ring = 2 * NP * slab(cfg.ks[cov[-1]]) - 2047   # (the scratch starts at the ring's first 2048-aligned byte)
    if cfg.rawdma:
        sp, in_, sp2, out = FZ_RAW
        if (out + nsb * 2 * D.OUT_B <= ring and sp + D.SCR_B <= in_ and in_ + 2 * 16 * D.RECROW <= sp2
                and sp2 + D.SCR2_B <= out):
            return FZ_RAW
        sp, in_, sp2, out = FZ_RAW_COMPACT
        if (out + nsb * 2 * D.OUT_B <= sp2 and sp2 + D.SCR2_B <= in_ and in_ + 16 * D.RECROW <= ring
                and sp + D.SCR_B <= out):
            return FZ_RAW_COMPACT
        if D.GF:   # packed (no tables): A, two raw records, B, the output
            in_ = -(-D.SCR_B // 64) * 64
            sp2 = in_ + 2 * 16 * D.RECROW
            out = -(-(sp2 + D.SCR2_B) // 64) * 64
            if out + nsb * 2 * D.OUT_B <= ring:
                return sp, in_, sp2, out
        # tight (thelp's short rings): the output, one raw record, then storage B at least 16 KB past A (a bank apart)
        out = max(3264, -(-D.SCR_B // 64) * 64)
        in_ = -(-(out + nsb * 2 * D.OUT_B) // 64) * 64
        sp2 = max(in_ + 16 * D.RECROW, 16384)
        if sp2 + D.SCR2_B > ring:   # (kchain's tail at 20 k-blocks: storage B in private storage, FZ_SP2_OWN)
            assert cfg.kchain and in_ + 16 * D.RECROW <= ring, (nsb, ring)
            sp2 = FZ_SP2_OWN
        return sp, in_, sp2, out
    sp, in_, sp2, out = FZ_SP, FZ_IN, FZ_SP2, FZ_OUT
    if out + nsb * 2 * D.OUT_B > ring:
        sp, in_, sp2, out = FZ_COMPACT
        assert out + nsb * 2 * D.OUT_B <= sp2 and sp2 + D.SCR2_B <= in_ and in_ + 16 * D.RECROW <= ring
        return sp, in_, sp2, out
    assert sp + D.SCR_B <= in_ and in_ + 16 * D.RECROW <= sp2 and sp2 + D.SCR2_B <= out
    return sp, in_, sp2, out


def fuse_decode(L, cfg, dp, cov, helper=None):
    """fuse 2: a replay group's fill (^fzdq(units)): per (pass, slab) unit, the raw stream's [16 rows][4 super-blocks]
    -> the decoder's input record (all 4 super-blocks: a record row holds them), the inlined cfg.fmt decoder on this
    row's super-blocks, then its k-blocks (panel order [kb][h]) copied record by record into the egress ring. Scratch
    is the W ring (idle
    until this panel's drain, which starts after the last record). Unit index and count live in private storage
    (the decoder needs every register)."""
    e = L.append
    D = decoder(cfg.fmt, cfg.gord, cfg.swiglu)
    hrow = cov[0] if helper == "H" else None           # (the helper's row, listed first: the filler copies its records)
    K = (helper or "").startswith("K")
    dcov = tuple(j for j in cov if j != hrow) if not K else (cov[0],)
    k0, k1, s0, nsb = fuse_span(cfg, dcov)
    fw, _ = fuse_record(cfg, cov)
    RW = FUSE_SB * D.BLK // 4                          # words per row of a unit (its super-blocks need not align)
    pieces = fuse_pieces(cfg, cov)
    hx = fuse_hx(cfg) if helper and cov in ((1, 0), (1,)) else 0   # (hsplit: ghelp's head only)
    if hx and helper == "H":                           # hsplit: the head decodes its whole super-blocks only
        k1 -= hx
        nsb = -(-k1 // 32) - s0
    elif hx:                                           # the helper: its slice, then the head's last hx k-blocks
        fw = fuse_record(cfg, (1, 0))[0]
        pieces = pieces + [(cfg.ks[0] - hx, 36 * hx)]
    elif (helper or "")[:1] == "M" and cov != (1,):                # thelp's tail: the last mid's record size
        fw = fuse_record(cfg, (cov[0], cov[0] - 1))[0]
    relay = 0
    if cfg.kchain and (helper or "")[:1] == "M":       # (kchain: every record 144 words, fuse_record)
        pieces = kchain_send(cfg, cov[0], True)
        relay = kchain_relay(cfg, cov[0]) if "h" in helper else 0
    assert FUSE_SB * D.BLK <= D.RECROW and FUSE_SB * D.BLK % 4 == 0
    SP, IN, SP2, OUT = fuse_scratch(cfg, dcov)
    assert fw % 16 == 0
    e("^fzdq(%fzN: reg<aie2p.er>):")
    e("  %fzpD = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  st %fzN, %fzpD, 12")                       # the units left
    tonce = not (D.GRID or D.Q3K) and (helper != "M" or bool(cfg.htab))
    if tonce:                                       # the decoder's table: built by the fill's first unit
        e("  %fzT0 = mova.i32 0")
        e("  %fzTs = mov.address-to-scalar %fzpD")
        e(f"  %fzTt = add %fzTs, {D.TFLAG}")
        e("  %fzTr = mov.scalar-to-address %fzTt")
        e("  st %fzT0, %fzTr, 0")
    rawport = HELP_RAW if (helper or "")[:1] == "M" else fuse_ports("")[1]
    cap = fuse_raw_ring(cfg, dcov)[1] if cfg.rawdma else 0
    if cfg.rawdma:   # the W ring is idle: grant the raw ring's records
        # an acquire of 0 first: a core reset while the core stalls in an acquire (this head, parked at the end of
        # a call) leaves the lock request behind, and the restarted core's first release also lands on that lock
        # (+cap phantom raw records after a re-setup); a first acquire clears it
        e("  %fzZ0 = mova.i32 0")
        e(f"  acq %fzZ0, {rawport}")
        e(f"  %fzG = mova.i32 {cap}")
        e(f"  rel %fzG, {rawport}")
    e("  low.br ^fzu")
    e("^fzu:")
    e("  %fzpU = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzu = lda %fzpU, 12")
    e("  %fzum = eqz %fzu")
    e("  low.cond_br %fzum, ^fzux, ^fzub : reg<aie2p.er>")
    e("^fzub:")
    e("  %fzR1 = mova.i32 1")
    e("  %fzR0 = mova.i32 0")
    # scratch pointers (re-derived each unit: nothing lives across the decoder)
    e("  %fzk2047 = mov.i32 2047")
    e("  %fzw0l = lda %fzpU, 24")
    e("  %fzwb = add.rr %fzw0l, %fzk2047")
    e("  %fzkm2048 = mov.i32 -2048")
    e("  %fzsb = and %fzwb, %fzkm2048")
    for nm, off in (("sp", SP), ("in", IN), ("sp2", SP2), ("out", OUT)):
        v = "in0" if nm == "in" and s0 else nm
        if off == FZ_SP2_OWN:
            e(f"  %fzs2 = storage {{byte_alignment = 64, byte_length = {D.SCR2_B}}} : low.storage<private>")
            e("  %fzs2p = storage_address %fzs2 : low.storage<private> -> reg<aie2p.ep>")
            e(f"  %fza_{v} = mov.address-to-scalar %fzs2p")
            continue
        e(f"  %fzo_{v} = mov.i32 {off}")
        e(f"  %fza_{v} = add.rr %fzsb, %fzo_{v}")
    inb = f"%fza_{'in0' if s0 else 'in'}"
    if cfg.rawdma:   # the unit is the raw ring's record (units left) % cap, its rows RECROW apart (the DMA lands them)
        e("  %fzAm1 = mova.i32 -1")
        e(f"  acq %fzAm1, {rawport}")
        if cap == 2:
            e("  %fzsl = and %fzu, %fzR1")
            e(f"  %fzsk = mov.i32 {16 * D.RECROW}")
            e("  %fzso = mul %fzsl, %fzsk")
            e(f"  %fzinx = add.rr {inb}, %fzso")
            inb = "%fzinx"
    else:
        raw_stream_unit(e, D, RW, inb)
    if s0:   # the decoder reads this row's super-blocks
        e(f"  %fzo_ins = mov.i32 {(2 * D.OUT_B if D.BFP else D.BLK) * s0}")   # (BFP: the record is in output layout)
        e(f"  %fzinS = add.rr {inb}, %fzo_ins")
        inb = "%fzinS"
    for nm in ("sp", "in", "sp2", "out"):
        e(f"  %dq_{nm} = mov.scalar-to-address {inb if nm == 'in' else '%fza_' + nm}")
    # the decoder (its values and labels prefixed)
    ren = lambda m: m.group(0) if m.group(2) == "fzs" else m.group(1) + "dq_" + m.group(2)
    # only the records (half super-blocks: 16 k-blocks) overlapping k0 .. k1: the egress reads nothing else
    rec = ((k0 - 32 * s0) // 16, -(-(k1 - 32 * s0) // 16)) if cfg.rskip else None
    assert D.OUT_B == 16 * 144
    for ln in D.leaf_inline(nsb, dec_tag(cfg), vroll=bool((helper or "")[:1] == "M" and cfg.hvroll), tonce=tonce, rec=rec,
                             q3roll=bool(cfg.q3roll)):
        ln = re.sub(r"([%^])([A-Za-z_]\w*)", lambda m: ren(m) if not m.group(2).startswith("dq_") else m.group(0), ln)
        e(ln if ln.startswith("^") else "  " + ln.strip())
    if cfg.rawdma:   # the raw record is consumed: back to the DMA, but for the last cap units (the panel follows them)
        e("  %fzpQ = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
        e("  %fzuQ = lda %fzpQ, 12")
        e(f"  %fzcQ = mov.i32 {cap}")
        e("  %fzmQ = lt %fzcQ, %fzuQ")
        e("  low.cond_br %fzmQ, ^fzrr, ^fzrj : reg<aie2p.er>")
        e("^fzrr:")
        e("  %fzr1Q = mova.i32 1")
        e(f"  rel %fzr1Q, {rawport}")
        e("  low.br ^fzrj")
        e("^fzrj:")
    # the egress: per slice of this row's panel (rows - 2: its own and the last row's, interleaved per unit), its
    # padded slab from the decoded super-blocks (the pad copies whatever follows), a record (fw words) at a time
    e("  %fzE1 = mova.i32 1")
    e("  %fzE0 = mova.i32 0")
    e("  %fzEm1 = mova.i32 -1")
    e("  %fzk2047b = mov.i32 2047")
    e("  %fzpW = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzw0l2 = lda %fzpW, 24")
    e("  %fzwb2 = add.rr %fzw0l2, %fzk2047b")
    e("  %fzkm2048b = mov.i32 -2048")
    e("  %fzsb2 = and %fzwb2, %fzkm2048b")
    if (all(4 * sw == 144 * cfg.ks[j] for j, (_, sw) in zip(cov, pieces)) and len(pieces) > 1 and hrow is None
            and not K
            and len(pieces) == len(cov)):
        fuse_egress_rotated(e, cfg, dp, cov, pieces, fw, OUT, k0, k1, s0)
        pieces = []
    if K:   # kchain: the piece for the row above, then the egress (own records, the row below's after them)
        r, b = cov[0], sum(cfg.ks[:cov[0]])
        if "u" in helper:
            kchain_up(e, kchain_send(cfg, r, False), fw, OUT, s0, cfg.hcap, eg512(cfg))
        tl = cfg.ks[cov[1]] if len(cov) == 2 else 0    # (a second member: the row below's whole slice after its own)
        # the memory tile's fill takes each unit as padded slabs (fuse_record): the egress below sends real k-blocks
        # only, so a slice of 2 mod 4 k-blocks would leave the fill waiting for its pad words (NPU timeout)
        assert 36 * (cfg.ks[r] + tl) == sum(slab(cfg.ks[j]) for j in cov) // 4, ("kchain egress != fill", cfg.ks, cov)
        if "h" in helper:
            fuse_egress_helped(e, dp, 36 * (cfg.ks[r] + tl) // fw, fw, OUT + (b - 32 * s0) * 144,
                               36 * (kchain_piece(cfg, r + 1) + tl) // fw, cfg.hcap, eg512(cfg), nh=0)
            pieces = []
        else:
            assert not tl
            pieces = [(b, 36 * cfg.ks[r])]
    if hrow is not None:
        assert len(cov) == 2 and cov[0] == hrow and (pieces[0][1] == pieces[1][1] or not hx)
        fuse_egress_helped(e, dp, pieces[1][1] // fw, fw, OUT + (pieces[1][0] - 32 * s0) * 144, 36 * hx // fw, cfg.hcap,
                           eg512(cfg), nh=pieces[0][1] // fw)
        pieces = []
    for i, (kb0, sw) in enumerate(pieces):
        x = f"^fze{i + 1}" if i + 1 < len(pieces) else "^fzrl" if relay else "^fzex"
        e(f"  %fzoo{i} = mov.i32 {OUT + (kb0 - 32 * s0) * 144}")
        e(f"  %fzsrc{i} = add.rr %fzsb2, %fzoo{i}")
        e(f"  %fzsp{i} = mov.scalar-to-address %fzsrc{i}")
        e(f"  %fznr{i} = mov.i32 {sw // fw}")
        e(f"  %fzpE{i} = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
        e(f"  %fzen{i} = lda %fzpE{i}, 8")               # the egress record count (its parity: the ring slot)
        e(f"  %fzdl{i} = lda %fzpE{i}, 28")              # the egress ring
        e(f"  low.br ^fze{i}h(%fzE0: reg<aie2p.er>, %fzsp{i}: reg<aie2p.ep>, %fzen{i}: reg<aie2p.er>)")
        e(f"^fze{i}h(%fzei{i}: reg<aie2p.er>, %fzes{i}: reg<aie2p.ep>, %fzec{i}: reg<aie2p.er>):")
        e(f"  %fzem{i} = lt %fzei{i}, %fznr{i}")
        e(f"  low.cond_br %fzem{i}, ^fze{i}b, ^fze{i}s : reg<aie2p.er>")
        e(f"^fze{i}b:")
        sp_, b_ = egress_record(e, f"{i}", f"%fzes{i}", f"%fzec{i}", f"%fzdl{i}", fw, cfg.hcap if (helper or "")[:1] == "M" else 2,
                                eg512(cfg), acq=f"  acq %fzEm1, {dp}")
        e(f"  rel %fzE1, {dp}")
        e(f"  %fzcs{i} = padda {sp_}, {4 * fw - b_}")
        e(f"  %fzec1{i} = add.rr %fzec{i}, %fzE1")
        e(f"  %fzei1{i} = add.rr %fzei{i}, %fzE1")
        e(f"  low.br ^fze{i}h(%fzei1{i}: reg<aie2p.er>, %fzcs{i}: reg<aie2p.ep>, %fzec1{i}: reg<aie2p.er>)")
        e(f"^fze{i}s:")
        e(f"  st %fzec{i}, %fzpE{i}, 8")
        e(f"  low.br {x}")
        if i + 1 < len(pieces):
            e(f"^fze{i + 1}:")
    if relay:
        kchain_relay_loop(e, dp, 36 * relay // fw, fw, cfg.hcap, eg512(cfg))
    e("^fzex:")
    e("  %fzpX = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzu2 = lda %fzpX, 12")
    e("  %fzu21 = add %fzu2, -1")
    e("  st %fzu21, %fzpX, 12")
    e("  low.br ^fzu")
    e("^fzux:")
    e("  low.br ^fzgemm")


def qf_skip_raw(e, cfg):
    """(Q' fuse 2, a row below the filler) the memory tile's raw and panel drains share the column's multicast W
    stream: the raw records (rows padded: gen_npu_q) reach this row as whole W records ahead of the panel; at a call's
    first firing take and return them, then the GEMM."""
    n = QF["rawn"] * 64 * QF["raww"] // QF["wslot"]
    e("  %rzs = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
    e("  %rzp = storage_address %rzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %rzc = lda %rzp, 0")
    e("  %rzc1 = add %rzc, 1")
    e(f"  %rzcl = mov.i32 {cfg.nb // QF.get('rows', 1)}")
    e("  %rzcw = lt %rzc1, %rzcl")
    e("  %rzcn = mul %rzc1, %rzcw")
    e("  st %rzcn, %rzp, 0")
    e("  %rzm = eqz %rzc")
    e("  low.cond_br %rzm, ^rzs0, ^rzgo : reg<aie2p.er>")
    e("^rzs0:")
    e("  %rzps = storage_address %rzs : low.storage<private> -> reg<aie2p.ep>")
    e(f"  %rzN = mov.i32 {n}")
    e("  st %rzN, %rzps, 4")
    e("  low.br ^rzu")
    e("^rzu:")
    e("  %rzpu = storage_address %rzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %rzu = lda %rzpu, 4")
    e("  %rzum = eqz %rzu")
    e("  low.cond_br %rzum, ^rzgo, ^rzub : reg<aie2p.er>")
    e("^rzub:")
    e("  %rzA = mova.i32 -1")
    e("  acq %rzA, 1")
    e("  %rzR = mova.i32 1")
    e("  rel %rzR, 1")
    e("  %rzpb = storage_address %rzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %rzv = lda %rzpb, 4")
    e("  %rzk = mov.i32 -1")
    e("  %rzvn = add.rr %rzv, %rzk")
    e("  st %rzvn, %rzpb, 4")
    e("  low.br ^rzu")
    e("^rzgo:")


CRMID = [0]   # (C relay chain) the leaf's middle row (leaf header -> its end), 0: none
CREL0 = [False]       # (leaf: a non-filling row 0 relays row 1's C)
QODS_EARLY = [False]   # (row 1 split: %qods declared ahead of its fill, qf_split_r1)


def qf_split_r1(L, cfg):
    """(Q' fill split, row 1) at a call's first firing: per step, its raw W record (W slot j & 1: [16 rows][slab]
    [144 B]) and the split channel's credit, slabs 2, 3 decoded (slab 2 into this core's C ring at the slab-2 output
    offset, slab 3 into the split ring), then the channel's ready and the W slot back; row 0 egresses them. The
    pointers are parked across the decoder (it needs every register) and rebuilt at ^rzgo."""
    e = L.append
    D = decoder(cfg.fmt, cfg.gord, cfg.swiglu)
    q = QF["cfg"]
    S = q.passes * FUSE_SB
    firings = cfg.nb // QF.get("rows", 1)
    assert (S + firings * cfg.passes) % 2 == 0, "W slot parity at a call's start"
    ws1 = int(QA4_RING_AT.split(",")[1].split("@")[1].split(":")[1])   # W slot 1
    cs0 = int(QA4_RING_AT.split(",")[0].split("@")[1].split(":")[0])
    so_ = MP * q3_strides()[1]
    tonce = not (D.GRID or D.Q3K)
    e(f"  %qods = storage {{byte_alignment = 1024, byte_length = {so_}}} : low.storage<scratch>")
    e("  %fzs = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
    e("  %fzt = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
    e("  %rzs = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
    e("  %rzp = storage_address %rzs : low.storage<private> -> reg<aie2p.ep>")
    for o, nm in ((20, "a"), (24, "w"), (28, "o"), (12, "qss")):
        e(f"  %rzk{o} = mov.address-to-scalar %{nm}_r")
        e(f"  st %rzk{o}, %rzp, {o}")
    e("  %rzc = lda %rzp, 0")
    e("  %rzc1 = add %rzc, 1")
    e(f"  %rzcl = mov.i32 {firings}")
    e("  %rzcw = lt %rzc1, %rzcl")
    e("  %rzcn = mul %rzc1, %rzcw")
    e("  st %rzcn, %rzp, 0")
    e("  %rzm = eqz %rzc")
    e("  low.cond_br %rzm, ^rzs0, ^rzgo : reg<aie2p.er>")
    e("^rzs0:")
    e("  %rzps = storage_address %rzs : low.storage<private> -> reg<aie2p.ep>")
    e(f"  %rzN = mov.i32 {S}")
    e("  st %rzN, %rzps, 4")
    if tonce:   # the decoder's table: built by the fill's first record
        e("  %rzpT = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
        e("  %rzT0 = mova.i32 0")
        e("  %rzTs = mov.address-to-scalar %rzpT")
        e(f"  %rzTt = add %rzTs, {D.TFLAG}")
        e("  %rzTr = mov.scalar-to-address %rzTt")
        e("  st %rzT0, %rzTr, 0")
    e("  low.br ^rzu")
    e("^rzu:")
    e("  %rzpu = storage_address %rzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %rzu = lda %rzpu, 4")
    e("  %rzum = eqz %rzu")
    e("  low.cond_br %rzum, ^rzgo, ^rzub : reg<aie2p.er>")
    e("^rzub:")
    e("  %rzA = mova.i32 -1")
    e("  acq %rzA, 1")
    e("  %rzA3 = mova.i32 -1")
    e("  acq %rzA3, 3")
    e("  %rzpb0 = storage_address %rzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %rzK2 = mova.i32 2")
    e("  st %rzK2, %rzpb0, 8")
    e("  low.br ^rzh")
    e("^rzh:")
    e("  %rzph = storage_address %rzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %rzhv = lda %rzph, 8")
    e(f"  %rzh4 = mova.i32 {NP}")
    e("  %rzhm = lt %rzhv, %rzh4")
    e("  low.cond_br %rzhm, ^rzhb, ^rzhx : reg<aie2p.er>")
    e("^rzhb:")
    e("  %rzpB = storage_address %rzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %rzlf = lda %rzpB, 4")                     # steps left: this step j = S - left, its W slot j & 1
    e(f"  %rzSk = mov.i32 {S}")
    e("  %rzj = sub %rzSk, %rzlf")
    e("  %rzK1 = mova.i32 1")
    e("  %rzpar = and %rzj, %rzK1")
    e(f"  %rzwk = mov.i32 {ws1}")
    e("  %rzwo = mul %rzpar, %rzwk")
    e("  %rzwb = lda %rzpB, 24")
    e("  %rzws = add.rr %rzwb, %rzwo")
    e(f"  %rzsk = mov.i32 {4 * QF['raww']}")
    e("  %rzso = mul %rzhv, %rzsk")
    e("  %rza_in = add.rr %rzws, %rzso")
    e("  %rza_sp = lda %rzpB, 28")                   # storage A: the C ring's start (idle until the GEMM)
    e(f"  %rzq2k = mov.i32 {QF_OUTS[0] - cs0}")
    e("  %rzq2 = add.rr %rza_sp, %rzq2k")            # slab 2's output, slab 3's: the split ring
    e("  %rzq3 = lda %rzpB, 12")
    e("  %rzdq = sub %rzq3, %rzq2")
    e("  %rzKm2 = mova.i32 -2")
    e("  %rzh2 = add.rr %rzhv, %rzKm2")
    e("  %rzdm = mul %rzdq, %rzh2")
    e("  %rza_out = add.rr %rzq2, %rzdm")
    e("  %rzpQ = storage_address %qods : low.storage<scratch> -> reg<aie2p.ep>")
    e("  %rza_sp2 = mov.address-to-scalar %rzpQ")
    for nm in ("sp", "in", "sp2", "out"):
        e(f"  %dq_{nm} = mov.scalar-to-address %rza_{nm}")
    ren = lambda m: m.group(0) if m.group(2) in ("fzs", "fzt") else m.group(1) + "dq_" + m.group(2)
    for ln in ([] if os.environ.get("GN_QFNODEC") else
               D.leaf_inline(1, dec_tag(cfg), tonce=tonce, q3roll=bool(cfg.q3roll),
                             vroll=qf_vroll(cfg),   # (the halves rolled, program memory)
                             rec=(lambda h: (h, h + 1) if h is not None else None)(qfhalf()))):
        ln = re.sub(r"([%^])([A-Za-z_]\w*)", lambda m: ren(m) if not m.group(2).startswith("dq_") else m.group(0), ln)
        e(ln if ln.startswith("^") else "  " + ln.strip())
    e("  %rzpn = storage_address %rzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %rzhn0 = lda %rzpn, 8")
    e("  %rzhn = add %rzhn0, 1")
    e("  st %rzhn, %rzpn, 8")
    e("  low.br ^rzh")
    e("^rzhx:")
    e("  %rzR3 = mova.i32 1")
    e("  rel %rzR3, 3")
    e("  %rzR = mova.i32 1")
    e("  rel %rzR, 1")
    e("  %rzpx = storage_address %rzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %rzv = lda %rzpx, 4")
    e("  %rzk = mov.i32 -1")
    e("  %rzvn = add.rr %rzv, %rzk")
    e("  st %rzvn, %rzpx, 4")
    e("  low.br ^rzu")
    e("^rzgo:")
    e("  %rzpG = storage_address %rzs : low.storage<private> -> reg<aie2p.ep>")
    for o, nm in ((20, "a"), (24, "w"), (28, "o")):
        e(f"  %rzl{o} = lda %rzpG, {o}")
        e(f"  %{nm} = mov.scalar-to-address %rzl{o}")


def qf_decode(L, cfg, dp):
    """Q' fuse 2's fill (cfg: the decode side's Config). Per super-block step (passes of 1024 K, 4 steps each): each
    slab's raw record (16 rows of the super-block, W slot 0) decoded into its own output area (QF_OUTS), then 8 pieces
    of 16 k-blocks in Q' pass order ([pass][slab]: the memory-tile panel drains linearly) to the egress ring."""
    e = L.append
    D = decoder(cfg.fmt, cfg.gord, cfg.swiglu)
    SP, IN, _, _ = fuse_scratch(cfg, (0,))
    fw = QF_FW
    assert 64 * 36 % (4 * fw) == 0 or 16 * 36 == 4 * fw, fw
    rawport = QF_PORTS[1]
    e("^fzdq(%fzN: reg<aie2p.er>):")
    e("  %fzpD = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  st %fzN, %fzpD, 12")                       # the super-block steps left
    if qfdirect_on() and QF_P1KD["raw"] == 2:   # (the raw record count: its intake slot)
        e("  %fzRc0 = mova.i32 0")
        e("  %fzpTr = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
        e(f"  st %fzRc0, %fzpTr, {QF_OUT}")   # (%fzt word QF_OUT: the direct fill keeps no output table)
    tonce = not (D.GRID or D.Q3K)
    if tonce:                                       # the decoder's table: built by the fill's first record
        e("  %fzT0 = mova.i32 0")
        e("  %fzTs = mov.address-to-scalar %fzpD")
        e(f"  %fzTt = add %fzTs, {D.TFLAG}")
        e("  %fzTr = mov.scalar-to-address %fzTt")
        e("  st %fzT0, %fzTr, 0")
    spl = bool(QF.get("split"))
    dec = bool(QF.get("dec"))                       # (a decoder row: its own memory map, qdec_leaf)
    one = spl or dec                                # (a step's raw record holds its slabs side by side)
    dsp = dec and QF.get("drole") == 0              # (two decoder rows: the first, fed slabs 2-3 by the second)
    dsnd = dec and QF.get("drole") == 1             # (the second: slabs 2-3 into its sender, no egress)
    NS = 2 if spl or dsp or dsnd else NP            # (split: this row's slabs; row 1 decodes the rest)
    pport = QDEC_RPORT if dsp else QF["split"](QF["gate_now"]) if spl else None
    if not spl and not dec and not qp1k_on():   # (P1K: every slab output in QF_P1K)
        e(f"  %fzo3s = storage {{byte_alignment = 64, byte_length = {2 * D.OUT_B}}} : low.storage<private>")
    # the slab outputs' addresses (from the W ring's first 2048-aligned byte)
    e("  %fzw0a = lda %fzpD, 24")
    e("  %fzk2047a = mov.i32 2047")
    e("  %fzwba = add.rr %fzw0a, %fzk2047a")
    e("  %fzkm2048a = mov.i32 -2048")
    e("  %fzsba = and %fzwba, %fzkm2048a")
    e("  %fzpT = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
    if spl:   # slabs 2, 3: row 1's (its C ring + the slab-2 offset; the split ring), in this core's view
        cs0 = int(QA4_RING_AT.split(",")[0].split("@")[1].split(":")[0])
        e("  %fzqG = storage_address %crst : low.storage<private> -> reg<aie2p.ep>")
        e("  %fzq2b = lda %fzqG, 0")
        e(f"  %fzq2k = mov.i32 {QF_OUTS[0] - cs0}")
        e("  %fzq2 = add.rr %fzq2b, %fzq2k")
        e(f"  st %fzq2, %fzpT, {QF_OUT + 8}")
        e("  %fzq3 = lda %fzqG, 4")
        e(f"  st %fzq3, %fzpT, {QF_OUT + 12}")
    outs = QF_P1K["outs"] if qp1k_on() else QF_OUTS
    assert len(outs) >= NS and QF_OUT + 4 * NS <= 64
    direct = qfdirect_on()
    for i, off in enumerate(outs[:0 if dsnd or direct else NS]):   # (dsnd, dsp's slabs 2-3: qdec_leaf)
        if dec:   # (the decoder's outputs: one private block, 2 OUT_B a slab)
            e(f"  %fzop{i} = storage_address %dqo : low.storage<private> -> reg<aie2p.ep>")
            e(f"  %fzoq{i} = mov.address-to-scalar %fzop{i}")
            e(f"  %fzok{i} = mov.i32 {2 * D.OUT_B * i}")
            e(f"  %fzoa{i} = add.rr %fzoq{i}, %fzok{i}")
            e(f"  st %fzoa{i}, %fzpT, {QF_OUT + 4 * i}")
            continue
        if off is None:
            e(f"  %fzop{i} = storage_address %fzo3s : low.storage<private> -> reg<aie2p.ep>")
            e(f"  %fzoa{i} = mov.address-to-scalar %fzop{i}")
            e(f"  st %fzoa{i}, %fzpT, {QF_OUT + 4 * i}")
            continue
        e(f"  %fzok{i} = mov.i32 {off}")
        e(f"  %fzoa{i} = add.rr %fzsba, %fzok{i}")
        e(f"  st %fzoa{i}, %fzpT, {QF_OUT + 4 * i}")
    if not dec:   # (a decoder's raw ring is its own: its DMA holds the credits)
        e("  %fzZ0 = mova.i32 0")                   # (fuse_decode: clear a stale lock request, grant the raw slot)
        e(f"  acq %fzZ0, {rawport}")
        e(f"  %fzG = mova.i32 {QF_P1KD['raw'] if direct else 1}")
        e(f"  rel %fzG, {rawport}")
    e("  low.br ^fzu")
    e("^fzu:")
    e("  %fzpU = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzu = lda %fzpU, 12")
    e("  %fzum = eqz %fzu")
    e("  low.cond_br %fzum, ^fzux, ^fzub : reg<aie2p.er>")
    e("^fzub:")
    r4 = QF4R[0]
    if one:   # (a step's raw record: one acquire)
        e("  %fzAs = mova.i32 -1")
        e(f"  acq %fzAs, {rawport}")
    if dsnd:   # (the sender's slot: one record a step)
        e("  %fzAd = mova.i32 -1")
        e(f"  acq %fzAd, {QDEC_SPORT}")
    e("  %fzH0 = mova.i32 0")
    e("  %fzpT1 = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
    e(f"  st %fzH0, %fzpT1, {QF_SLAB}")
    e("  low.br ^fzh")
    # per slab: its raw record, decoded into its output
    e("^fzh:")
    e("  %fzpH = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
    e(f"  %fzhv = lda %fzpH, {QF_SLAB}")
    e(f"  %fzh4 = mova.i32 {NS}")
    e("  %fzhm = lt %fzhv, %fzh4")
    e("  low.cond_br %fzhm, ^fzhb, ^fzhx : reg<aie2p.er>")
    e("^fzhb:")
    if r4:   # (GN_QF4R) this row's records: the call's record index's parity, (steps left x NS + slab) % 2 (an even
        # step count): (steps left + slab) % 2 for NS odd, slab % 2 for NS even
        assert QF["rawn"] % 2 == 0 and QF["cfg"].passes * FUSE_SB % 2 == 0
        e("  %f4pV = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
        e(f"  %f4v = lda %f4pV, {QF_SLAB}")
        if NS % 2:
            e("  %f4pU = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
            e("  %f4u = lda %f4pU, 12")
            e("  %f4uv = add.rr %f4u, %f4v")
        else:
            e("  %f4uv = or %f4v, %f4v")
        e("  %f4k1 = mova.i32 1")
        e("  %f4p = and %f4uv, %f4k1")
        e(f"  low.cond_br %f4p, {'^f4own, ^f4x' if r4['par'] else '^f4x, ^f4own'} : reg<aie2p.er>")
        e("^f4own:")
    if QFCYC and os.environ.get("GN_QFRAW") == "1":   # (the raw record's wait: word 24)
        e("  %qfpra = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
        e("  %qfwra = mov.cycles")
        e("  %qflra = slice %qfwra[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
        e("  st %qflra, %qfpra, 20")
    if not one:
        e("  %fzAm1 = mova.i32 -1")
        e(f"  acq %fzAm1, {rawport}")
    raw4 = qfraw4_on()
    if raw4:   # (the record's 4 passes: %fzt word QF_REC counts them)
        e("  %fzJ0 = mova.i32 0")
        e("  %fzpJ0 = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
        e(f"  st %fzJ0, %fzpJ0, {QF_REC}")
        e("  low.br ^fzj")
        e("^fzj:")
        e("  %fzpJ = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
        e(f"  %fzjv = lda %fzpJ, {QF_REC}")
        e(f"  %fzj4 = mova.i32 {FUSE_SB}")
        e("  %fzjm = lt %fzjv, %fzj4")
        e("  low.cond_br %fzjm, ^fzjb, ^fzjx : reg<aie2p.er>")
        e("^fzjb:")
    if direct:   # (the slab's two egress records)
        e(f"  %fzAd2 = mova.i32 {-1 if qfegr2_on() and not (QF4R[0] and QF4R[0]['help']) else -2}")
        if os.environ.get("GN_QFODLY"):   # (diag) spin N trips before every own-slab egress acquire
            e(f"  %fzok = mov.i32 {int(os.environ['GN_QFODLY'])}")
            e("  %fzoz = mova.i32 0")
            e("  low.br ^fzoh(%fzok: reg<aie2p.er>)")
            e("^fzoh(%fzoc: reg<aie2p.er>):")
            e("  %fzom = lt %fzoz, %fzoc")
            e("  low.cond_br %fzom, ^fzob, ^fzox : reg<aie2p.er>")
            e("^fzob:")
            e("  %fzon = add %fzoc, -1")
            e("  low.br ^fzoh(%fzon: reg<aie2p.er>)")
            e("^fzox:")
        qfw_wrap(e, "o", 32, [f"  acq %fzAd2, {dp}"])
        if qfl2guard():   # (the own slab takes its helper copy's slot of the second lane: that ring drained first)
            e("  %fzL2m = mova.i32 -1")
            e(f"  acq %fzL2m, {QF_LANE2_PORT}")
    if QFCYC and os.environ.get("GN_QFRAW") == "1":
        e("  %qfprb = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
        e("  %qfwrb = mov.cycles")
        e("  %qflrb = slice %qfwrb[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
        e("  %qfsrb = lda %qfprb, 20")
        e("  %qfdrb = sub %qflrb, %qfsrb")
        e("  %qftrb = lda %qfprb, 24")
        e("  %qfnrb = add.rr %qftrb, %qfdrb")
        e("  st %qfnrb, %qfprb, 24")
    e("  %fzpB = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzw0l = lda %fzpB, 24")
    if dec:   # storages A and B; the input: the raw ring's slot (record count parity, %fzs word 0)
        e("  %fzpa_sp = storage_address %dqa : low.storage<private> -> reg<aie2p.ep>")
        e("  %fza_sp = mov.address-to-scalar %fzpa_sp")
        e("  %fzrc = lda %fzpB, 0")                  # (the slot: 0, 1, 2; a 3-record ring takes no W pin)
        e(f"  %fzrkb = mov.i32 {16 * D.RECROW}")
        e("  %fzro = mul %fzrc, %fzrkb")
        e("  %fzsb = add.rr %fzw0l, %fzro")
        e("  %fzo_in = mova.i32 0")
    else:
        e("  %fzk2047 = mov.i32 2047")
        e("  %fzwb = add.rr %fzw0l, %fzk2047")
        e("  %fzkm2048 = mov.i32 -2048")
        e("  %fzsb = and %fzwb, %fzkm2048")
        if p1k4_on():   # (the pad from the guard's address: the guard stays referenced, so pinned)
            e("  %fzgp = mov.local-address @qfgd")
            e("  %fzgs = mov.address-to-scalar %fzgp")
            e(f"  %fzo_sp = mov.i32 {SP - p1k4_guard()}")
            e("  %fza_sp = add.rr %fzgs, %fzo_sp")
        else:
            e(f"  %fzo_sp = mov.i32 {SP}")
            e("  %fza_sp = add.rr %fzsb, %fzo_sp")
        e(f"  %fzo_in = mov.i32 {IN}")
    if one:   # (slab s: 4 s raw words a row on)
        e(f"  %fzsk = mov.i32 {4 * QF['raww']}")
        if dsnd:   # (slabs 2, 3)
            e("  %fzhv2 = add %fzhv, 2")
            e("  %fzso = mul %fzhv2, %fzsk")
        else:
            e("  %fzso = mul %fzhv, %fzsk")
        e("  %fza_in0 = add.rr %fzsb, %fzo_in")
        e("  %fza_in = add.rr %fza_in0, %fzso")
    else:
        if direct and QF_P1KD["raw"] == 2:   # (intake slot: raw record count mod 2)
            e("  %fza_in0d = add.rr %fzsb, %fzo_in")
            e("  %fzpTd = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
            e(f"  %fzrcd = lda %fzpTd, {QF_OUT}")
            e("  %fzrk1 = mova.i32 1")
            e("  %fzrpd = and %fzrcd, %fzrk1")
            e(f"  %fzrkb = mov.i32 {16 * D.RECROW}")
            e("  %fzrod = mul %fzrpd, %fzrkb")
            if raw4:   # (pass j's super-block: BLK bytes on in each row; the count steps after the record)
                e("  %fzpJi = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
                e(f"  %fzjvi = lda %fzpJi, {QF_REC}")
                e(f"  %fzjkb = mov.i32 {D.BLK}")
                e("  %fzjo = mul %fzjvi, %fzjkb")
                e("  %fza_in0j = add.rr %fza_in0d, %fzjo")
                e("  %fza_in = add.rr %fza_in0j, %fzrod")
            else:
                if qf_odd2(QF["cfg"]):   # (odd steps start 2 bytes into the record: the step's parity is the steps
                    # left's, word 12 of %fzs, for an even step count)
                    e("  %fza_inr = add.rr %fza_in0d, %fzrod")
                    e("  %fzoU = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
                    e("  %fzou = lda %fzoU, 12")
                    e("  %fzok1 = mova.i32 1")
                    e("  %fzop1 = and %fzou, %fzok1")
                    e("  %fzrp2 = add.rr %fzop1, %fzop1")
                    e("  %fza_in = add.rr %fza_inr, %fzrp2")
                else:
                    e("  %fza_in = add.rr %fza_in0d, %fzrod")
                e("  %fzrcn = add %fzrcd, 1")
                e(f"  st %fzrcn, %fzpTd, {QF_OUT}")
        else:
            e("  %fza_in = add.rr %fzsb, %fzo_in")
    if dec:
        e("  %fzpa_sp2 = storage_address %dqb : low.storage<private> -> reg<aie2p.ep>")
        e("  %fza_sp2 = mov.address-to-scalar %fzpa_sp2")
    else:
        e(f"  %fzo_sp2 = mov.i32 {fuse_scratch(cfg, (0,))[2]}")
        e("  %fza_sp2 = add.rr %fzsb, %fzo_sp2")
    e("  %fzpO = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzpOs = mov.address-to-scalar %fzpO")
    e("  %fzh4b = mova.i32 4")
    e("  %fzhq = mul %fzhv, %fzh4b")
    e("  %fzpOq = add.rr %fzpOs, %fzhq")
    e("  %fzpOa = mov.scalar-to-address %fzpOq")
    if direct:   # the ring slot pair of egress record count (word 8) mod 4
        m_ = QF_P1KD
        e("  %fzdlo = lda %fzpB, 28")
        e("  %fzcno = lda %fzpB, 8")
        e(f"  %fzk3o = mova.i32 {(qf4r_hslots() if QF4R[0] and QF4R[0]['help'] else m_['slots']) - 1}")
        e("  %fzslo = and %fzcno, %fzk3o")
        e(f"  %fzkbo = mov.i32 {m_['rec']}")
        e("  %fzofo = mul %fzslo, %fzkbo")
        if qfhalf():   # (record h only: written at out + h OUT_B)
            e(f"  %fzofh = mov.i32 {-qfhalf() * D.OUT_B}")
            e("  %fzofo2 = add.rr %fzofo, %fzofh")
            e("  %fza_out = add.rr %fzdlo, %fzofo2")
        else:
            e("  %fza_out = add.rr %fzdlo, %fzofo")
    else:
        e(f"  %fza_out = lda %fzpOa, {QF_OUT}")
    for nm in ("sp", "in", "sp2", "out"):
        e(f"  %dq_{nm} = mov.scalar-to-address %fza_{nm}")
    ren = lambda m: m.group(0) if m.group(2) in ("fzs", "fzt", "qfc") else m.group(1) + "dq_" + m.group(2)
    qfdec = QFCYC and os.environ.get("DQ_CYC") == "1"   # (diag: the decoder's cycles summed into %qfc byte 16)
    if qfdec:
        e("  %qfdw0 = mov.cycles")
        e("  %qfd0 = slice %qfdw0[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
    for ln in ([] if os.environ.get("GN_QFNODEC") else   # (diag: no decoder, every lock and egress kept)
               D.leaf_inline(1, dec_tag(cfg), tonce=tonce, q3roll=bool(cfg.q3roll),
                             vroll=qf_vroll(cfg),   # (the halves rolled, program memory)
                             rec=(lambda h: (h, h + 1) if h is not None else None)(qfhalf()))):
        ln = re.sub(r"([%^])([A-Za-z_]\w*)", lambda m: ren(m) if not m.group(2).startswith("dq_") else m.group(0), ln)
        e(ln if ln.startswith("^") else "  " + ln.strip())
    if qfdec:
        e("  %qfdw1 = mov.cycles")
        e("  %qfd1 = slice %qfdw1[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
        e("  %qfdd = sub %qfd1, %qfd0")
        e("  %qfdp = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
        e("  %qfds = lda %qfdp, 16")
        e("  %qfdn = add.rr %qfds, %qfdd")
        e("  st %qfdn, %qfdp, 16")
    if direct:   # the slab's records to the egress DMA
        e(f"  %fzE2d = mova.i32 {1 if qfegr2_on() and not (QF4R[0] and QF4R[0]['help']) else 2}")
        e(f"  rel %fzE2d, {dp}")
        e("  %fzpWd = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
        e("  %fzend = lda %fzpWd, 8")
        e("  %fzend2 = add %fzend, 2")
        e("  st %fzend2, %fzpWd, 8")
    if raw4:   # the next pass of the record; after its 4th, the intake count steps
        e("  %fzpJn = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
        e(f"  %fzjn0 = lda %fzpJn, {QF_REC}")
        e("  %fzjn = add %fzjn0, 1")
        e(f"  st %fzjn, %fzpJn, {QF_REC}")
        e("  low.br ^fzj")
        e("^fzjx:")
        e("  %fzpTx = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
        e(f"  %fzrcx = lda %fzpTx, {QF_OUT}")
        e("  %fzrcxn = add %fzrcx, 1")
        e(f"  st %fzrcxn, %fzpTx, {QF_OUT}")
    # the raw slot back to the DMA, but for the call's last record (the panel's W records follow it)
    e("  %fzpQ = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzuQ = lda %fzpQ, 12")
    e("  %fzpQ2 = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
    e(f"  %fzhQ = lda %fzpQ2, {QF_SLAB}")
    e("  %fzQ1 = mova.i32 1")
    e("  %fzmQa = lt %fzQ1, %fzuQ")                 # more steps after this one
    e(f"  %fzQ3 = mova.i32 {NS - (QF_P1KD['raw'] if direct else 1)}")   # (the call's last slots stay granted)
    e("  %fzmQb = lt %fzhQ, %fzQ3")                 # more slabs in this step
    if r4:   # (GN_QF4R: the row's own records, counted in QF_OUT)
        e("  %f4pQo = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
        e(f"  %f4rq = lda %f4pQo, {QF_OUT}")
        e(f"  %f4nq = mova.i32 {QF['rawn'] // 2 - QF_P1KD['raw'] + 1}")
        e("  %fzmQ = lt %f4rq, %f4nq")
    elif dec:   # (decoder: every step's record back after its last slab; the ring parity counts on)
        e("  %fzmQ = eqz %fzmQb")
    elif spl:   # (split: the step's record back after its last slab, but for the call's last step)
        e("  %fzmQc = eqz %fzmQb")
        e("  %fzmQ = and %fzmQa, %fzmQc")
    else:
        e("  %fzmQ = or %fzmQa, %fzmQb")
    e("  low.cond_br %fzmQ, ^fzrr, ^fzrj : reg<aie2p.er>")
    e("^fzrr:")
    e("  %fzr1Q = mova.i32 1")
    e(f"  rel %fzr1Q, {rawport}")
    if dec:
        e("  %fzpRc = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
        e("  %fzrc2 = lda %fzpRc, 0")
        e("  %fzrc3 = add %fzrc2, 1")
        e("  %fzrk3 = mova.i32 3")
        e("  %fzrcw = lt %fzrc3, %fzrk3")
        e("  %fzrc4 = mul %fzrc3, %fzrcw")
        e("  st %fzrc4, %fzpRc, 0")
    e("  low.br ^fzrj")
    e("^fzrj:")
    e("  %fzhn = add.rr %fzhQ, %fzQ1")
    e(f"  st %fzhn, %fzpQ2, {QF_SLAB}")
    e("  low.br ^fzh")
    # the step's egress: records r = 0 .. 31: piece r / 4 (pass r / 16 of the step's two, slab r / 4 % 4), part r % 4
    e("^fzhx:")
    if dsnd:   # (the step's slabs to the first decoder; no egress)
        e("  %fzqd = mova.i32 1")
        e(f"  rel %fzqd, {QDEC_SPORT}")
        e("  low.br ^fzex")
    if direct:   # (the slabs went out as they were decoded)
        e("  low.br ^fzex")
    hx0 = len(L)
    if spl or dsp:   # (row 1's slabs 2, 3 of the step)
        e("  %fzqa = mova.i32 -1")
        e(f"  acq %fzqa, {pport}")
    if QFCYC:
        e("  %qfphx = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
        e("  %qfwhx = mov.cycles")
        e("  %qflhx = slice %qfwhx[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
        e("  st %qflhx, %qfphx, 0")
    e("  %fzE0 = mova.i32 0")
    e("  %fzpX = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
    e(f"  st %fzE0, %fzpX, {QF_REC}")
    e("  low.br ^fze")
    e("^fze:")
    e("  %fzpE = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
    e(f"  %fzr = lda %fzpE, {QF_REC}")
    e(f"  %fzr32 = mova.i32 {8 * NS}")   # (8 records a slab: 2 halves x 4 parts)
    e("  %fzrm = lt %fzr, %fzr32")
    e("  low.cond_br %fzrm, ^fzeb, ^fzex : reg<aie2p.er>")
    e("^fzeb:")
    e("  %fzEm1 = mova.i32 -1")
    e("  %fzE1 = mova.i32 1")
    e("  %fzk3 = mova.i32 3")
    e("  %fzk2 = mova.i32 -2")
    e("  %fzk4 = mova.i32 -4")
    if qp1k_on():   # (P1K: the panel interleaved per slab, [step][slab][half][part]: r = 8 slab + 4 half + part)
        e("  %fzk7 = mova.i32 7")
        e("  %fzkm3 = mova.i32 -3")
        e("  %fzr7 = and %fzr, %fzk7")
        e("  %fzpart = and %fzr, %fzk3")
        e("  %fzsl = lshl %fzr, %fzkm3")                # slab (r / 8)
        e("  %fzps = lshl %fzr7, %fzk2")                # half (r / 4 % 2)
    else:
        e("  %fzpart = and %fzr, %fzk3")                # r % 4
        e("  %fzpc = lshl %fzr, %fzk2")                 # r / 4 (lshl by a negative count shifts right)
        e("  %fzsl = and %fzpc, %fzk3")                 # slab
        e("  %fzps = lshl %fzr, %fzk4")                 # pass of the step (r / 16)
    e("  %fzpEs = mov.address-to-scalar %fzpE")
    e("  %fzk4b = mova.i32 4")
    e("  %fzslq = mul %fzsl, %fzk4b")
    e("  %fzpEq = add.rr %fzpEs, %fzslq")
    e("  %fzpEa = mov.scalar-to-address %fzpEq")
    e(f"  %fzob = lda %fzpEa, {QF_OUT}")             # the slab's output
    e(f"  %fzkps = mov.i32 {16 * 144}")
    e("  %fzpso = mul %fzps, %fzkps")
    e(f"  %fzkpt = mov.i32 {4 * fw}")
    e("  %fzpto = mul %fzpart, %fzkpt")
    e("  %fzsrc1 = add.rr %fzob, %fzpso")
    e("  %fzsrc = add.rr %fzsrc1, %fzpto")
    e("  %fzsp = mov.scalar-to-address %fzsrc")
    e("  %fzpW = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzen = lda %fzpW, 8")                     # the egress record count (its parity: the ring slot)
    e("  %fzdl = lda %fzpW, 28")                    # the egress ring
    egress_record(e, "q", "%fzsp", "%fzen", "%fzdl", fw, 2, eg512(cfg), acq=f"  acq %fzEm1, {dp}")
    e(f"  rel %fzE1, {dp}")
    e("  %fzpW2 = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzen2 = lda %fzpW2, 8")
    e("  %fzen1 = add.rr %fzen2, %fzE1")
    e("  st %fzen1, %fzpW2, 8")
    e("  %fzpE2 = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
    e(f"  %fzr2 = lda %fzpE2, {QF_REC}")
    e("  %fzr1 = add.rr %fzr2, %fzE1")
    e(f"  st %fzr1, %fzpE2, {QF_REC}")
    e("  low.br ^fze")
    if dsnd or direct:
        del L[hx0:]
    e("^fzex:")
    if spl or dsp:   # (row 1 may overwrite them)
        e("  %fzqr = mova.i32 1")
        e(f"  rel %fzqr, {pport}")
    if QFCYC:
        e("  %qfpex = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
        e("  %qfwex = mov.cycles")
        e("  %qflex = slice %qfwex[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
        e("  %qfsex = lda %qfpex, 0")
        e("  %qfdex = sub %qflex, %qfsex")
        e("  %qftex = lda %qfpex, 4")
        e("  %qfnex = add.rr %qftex, %qfdex")
        e("  st %qfnex, %qfpex, 4")
    e("  %fzpX2 = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzu2 = lda %fzpX2, 12")
    e("  %fzu21 = add %fzu2, -1")
    e("  st %fzu21, %fzpX2, 12")
    e("  low.br ^fzu")
    if r4:
        qf4r_other(e, NS, rawport, dp, direct)
    e("^fzux:")
    if QFCYC:
        e("  %qfpux = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
        e("  %qfwux = mov.cycles")
        e("  %qflux = slice %qfwux[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
        e("  %qfsux = lda %qfpux, 8")
        e("  %qfdux = sub %qflux, %qfsux")
        e("  %qftux = lda %qfpux, 12")
        e("  %qfnux = add.rr %qftux, %qfdux")
        e("  st %qfnux, %qfpux, 12")
    e("  low.br ^fzgemm")


def qfw_wrap(e, tag, word, lines, raw=False, mode="1"):
    """(diag GN_QFW=mode, with GN_QFCYC) the cycles of lines (acquires) summed into %qfc byte word. raw: lines may hold
    block labels (the gate check): emitted as they are."""
    mx = QFCYC and tag == "o" and os.environ.get("GN_QFW") == "7"   # (diag) own egress: sum 32, max 36, its acquire 40
    on = QFCYC and os.environ.get("GN_QFW") == mode or raw or mx
    if on:
        e(f"  %qw{tag}p = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
        e(f"  %qw{tag}a = mov.cycles")
        e(f"  %qw{tag}al = slice %qw{tag}a[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
        if raw:   # (the end is another block: the start through %qfc byte 52)
            e(f"  %qw{tag}k0 = mova.i32 32")
            e(f"  %qw{tag}m0 = mov.modifier %qw{tag}k0")
            e(f"  %qw{tag}p2 = padda.modifier %qw{tag}p, %qw{tag}m0")
            e(f"  st %qw{tag}al, %qw{tag}p2, 20")
    for ln in lines:
        e(ln)
    if on:
        e(f"  %qw{tag}b = mov.cycles")
        e(f"  %qw{tag}bl = slice %qw{tag}b[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
        e(f"  %qw{tag}q0 = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
        e(f"  %qw{tag}k = mova.i32 32")
        e(f"  %qw{tag}m = mov.modifier %qw{tag}k")
        e(f"  %qw{tag}q = padda.modifier %qw{tag}q0, %qw{tag}m")
        if raw:
            e(f"  %qw{tag}a2 = lda %qw{tag}q, 20")
        e(f"  %qw{tag}d = sub %qw{tag}bl, %qw{tag}{'a2' if raw else 'al'}")
        e(f"  %qw{tag}s = lda %qw{tag}q, {word - 32}")
        e(f"  %qw{tag}n = add.rr %qw{tag}s, %qw{tag}d")
        e(f"  st %qw{tag}n, %qw{tag}q, {word - 32}")
        if mx:   # (the longest single wait and the running acquire count, 1-based, at it; count at byte 56)
            e(f"  %qw{tag}c0 = lda %qw{tag}q, 24")
            e(f"  %qw{tag}c1 = add %qw{tag}c0, 1")
            e(f"  st %qw{tag}c1, %qw{tag}q, 24")
            e(f"  %qw{tag}mx = lda %qw{tag}q, 4")
            e(f"  %qw{tag}gt = lt %qw{tag}mx, %qw{tag}d")
            e(f"  %qw{tag}dm = sub %qw{tag}d, %qw{tag}mx")
            e(f"  %qw{tag}dg = mul %qw{tag}dm, %qw{tag}gt")
            e(f"  %qw{tag}m2 = add.rr %qw{tag}mx, %qw{tag}dg")
            e(f"  st %qw{tag}m2, %qw{tag}q, 4")
            e(f"  %qw{tag}ix = lda %qw{tag}q, 8")
            e(f"  %qw{tag}di = sub %qw{tag}c1, %qw{tag}ix")
            e(f"  %qw{tag}ig = mul %qw{tag}di, %qw{tag}gt")
            e(f"  %qw{tag}i2 = add.rr %qw{tag}ix, %qw{tag}ig")
            e(f"  st %qw{tag}i2, %qw{tag}q, 8")


def qf4r_other(e, NS, rawport, dp, direct):
    """(GN_QF4R) a record of the other multicast: an egress row copies its helper's two egress records of it (ring
    QF4R_NB, 8 slots, count in %f4s word 4) into its egress ring, in panel order; a helper skips it."""
    assert direct and QF_P1KD["raw"] == 2
    hlp = QF4R[0]["help"]
    e("^f4x:")
    xw = QFCYC and os.environ.get("GN_QFW") in ("3", "4", "6")   # (diag) the branch's cycles: %qfc byte 56 (start: 60;
    xw4 = xw and os.environ.get("GN_QFW") == "4"                # 4: up to its copies only; 5: the neighbour's first
    xw5 = QFCYC and os.environ.get("GN_QFW") == "5"             # load only)
    if xw:
        e("  %qxp0 = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
        e("  %qxk = mova.i32 32")
        e("  %qxm = mov.modifier %qxk")
        e("  %qxp = padda.modifier %qxp0, %qxm")
        e("  %qxa = mov.cycles")
        e("  %qxal = slice %qxa[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
        e("  st %qxal, %qxp, 28")
    if not hlp:   # the helper's slab (two records) into the egress ring
        e("  %f4A2 = mova.i32 -2")
        qfw_wrap(e, "h", 36, [f"  acq %f4A2, {QF4R_NB}"])
        if os.environ.get("GN_QFW") == "6" and QFCYC:   # (diag) branch start .. past the helper-record acquire,
            e("  low.br ^q6a")                                # in its own block (no reordering across it)
            e("^q6a:")
            e("  %qxp20 = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
            e("  %qxk2 = mova.i32 32")
            e("  %qxm2 = mov.modifier %qxk2")
            e("  %qxp2 = padda.modifier %qxp20, %qxm2")
            e("  %qxb = mov.cycles")
            e("  %qxbl = slice %qxb[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
            e("  %qxs = lda %qxp2, 28")
            e("  %qxd = sub %qxbl, %qxs")
            e("  %qxt = lda %qxp2, 24")
            e("  %qxn = add.rr %qxt, %qxd")
            e("  st %qxn, %qxp2, 24")
        l2_ = QF4R[0]["l2"]
        e(f"  %f4AE = mova.i32 {-1 if qfegr2_on() or l2_ else -2}")
        dp = QF_LANE2_PORT if l2_ else dp   # (GN_QFLANE2: the helper's slab on the second egress lane)
        qfw_wrap(e, "x", 40, [] if qfl2guard() else [f"  acq %f4AE, {dp}"])   # (guard: the own slab took it)
        if os.environ.get("GN_QFL2DLY"):   # (diag) spin N trips before the copy's loads
            e(f"  %f4dk = mov.i32 {int(os.environ['GN_QFL2DLY'])}")
            e("  %f4dz = mova.i32 0")
            e("  low.br ^f4dh(%f4dk: reg<aie2p.er>)")
            e("^f4dh(%f4dc: reg<aie2p.er>):")
            e("  %f4dm = lt %f4dz, %f4dc")
            e("  low.cond_br %f4dm, ^f4db, ^f4dx : reg<aie2p.er>")
            e("^f4db:")
            e("  %f4dn = add %f4dc, -1")
            e("  low.br ^f4dh(%f4dn: reg<aie2p.er>)")
            e("^f4dx:")
        e("  %f4pS = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
        e("  %f4pN4 = storage_address %f4s : low.storage<private> -> reg<aie2p.ep>")
        e("  %f4nb = lda %f4pN4, 0")
        e("  %f4nc = lda %f4pN4, 4")
        e(f"  %f4k7 = mova.i32 {qf4r_hslots() - 1}")
        e("  %f4ns = and %f4nc, %f4k7")
        e(f"  %f4kr = mov.i32 {QF_P1KD['rec']}")
        e("  %f4no = mul %f4ns, %f4kr")
        e("  %f4src = add.rr %f4nb, %f4no")
        e("  %f4eb = lda %f4pS, 28")
        e("  %f4ec = lda %f4pS, 8")
        e(f"  %f4k3 = mova.i32 {QF_P1KD['slots'] - 1}")
        e("  %f4es = and %f4ec, %f4k3")
        e("  %f4eo = mul %f4es, %f4kr")
        e("  %f4dst = add.rr %f4eb, %f4eo")
        e("  %f4s0 = mov.scalar-to-address %f4src")
        e("  %f4q_0 = mov.scalar-to-address %f4dst")
        # (the vector loads' address waits for a first load from the neighbour's memory: reads right past an acquire
        # can see its stores stale, crel_copy)
        body5 = ["  %f4kk = lda %f4s0, 0",
                 "  %f4z0 = mova.i32 0",
                 "  %f4k0 = mul %f4kk, %f4z0",
                 "  %f4km = mov.modifier %f4k0",
                 "  %f4p_0 = padda.modifier %f4s0, %f4km"]
        if xw5:   # (the stamp after the address that waits on the load)
            body5 += ["  %qx5s = mov.address-to-scalar %f4p_0", "  %qx5z = mul %qx5s, %f4z0"]
        if xw5:
            e("  %qx5p0 = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
            e("  %qx5k = mova.i32 32")
            e("  %qx5m = mov.modifier %qx5k")
            e("  %qx5p = padda.modifier %qx5p0, %qx5m")
            e("  %qx5a = mov.cycles")
            e("  %qx5al = slice %qx5a[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
        for ln_ in body5:
            e(ln_)
        if xw5:
            e("  %qx5b = mov.cycles")
            e("  %qx5bl = slice %qx5b[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
            e("  %qx5bz = add.rr %qx5bl, %qx5z")
            e("  %qx5d = sub %qx5bz, %qx5al")
            e("  %qx5t = lda %qx5p, 24")
            e("  %qx5n = add.rr %qx5t, %qx5d")
            e("  st %qx5n, %qx5p, 24")
        nl = 2 * QF_P1KD["rec"] // 64
        if os.environ.get("GN_QF4RCOPY", "1") == "2":   # groups of G, pipelined: group g + 1's loads beside g's stores
            G_ = int(os.environ.get("GN_QF4RCOPYG", "6"))
            ng_ = (nl + G_ - 1) // G_
            cl_, sp_, dp_ = [], ["%f4p_0"], ["%f4q_0"]
            for s_ in range(ng_ + 1):
                if s_ < ng_:
                    if s_:
                        cl_.append(f"  %f4s{s_} = padda {sp_[-1]}, {64 * G_}")
                        sp_.append(f"%f4s{s_}")
                    cl_ += [f"  %f4w{q} = {'vlda' if q % 2 == 0 else 'vldb'}.512.i32x16 {sp_[s_]}, {64 * (q % G_)}"
                            for q in range(G_ * s_, min(G_ * s_ + G_, nl))]
                if s_:
                    g_ = s_ - 1
                    if g_:
                        cl_.append(f"  %f4d{g_} = padda {dp_[-1]}, {64 * G_}")
                        dp_.append(f"%f4d{g_}")
                    cl_ += [f"  vst.512.i32x16 %f4w{q}, {dp_[g_]}, {64 * (q % G_)}"
                            for q in range(G_ * g_, min(G_ * g_ + G_, nl))]
        elif os.environ.get("GN_QF4RCOPY", "1") == "1":   # groups of 7: loads on both load units, then the stores (a
            cl_, sb_, db_ = [], "%f4p_0", "%f4q_0"           # load waits for the store before it: one pair a 7 cycles)
            for g_ in range((nl + 6) // 7):
                if g_:
                    cl_ += [f"  %f4s{g_} = padda {sb_}, 448", f"  %f4d{g_} = padda {db_}, 448"]
                    sb_, db_ = f"%f4s{g_}", f"%f4d{g_}"
                qs_ = range(7 * g_, min(7 * g_ + 7, nl))
                cl_ += [f"  %f4w{q} = {'vlda' if q % 2 == 0 else 'vldb'}.512.i32x16 {sb_}, {64 * (q % 7)}" for q in qs_]
                cl_ += [f"  vst.512.i32x16 %f4w{q}, {db_}, {64 * (q % 7)}" for q in qs_]
        else:
            cl_ = [ln for i in range(nl) for ln in (f"  %f4v_{i}, %f4p_{i + 1} = vlda.acc.post %f4p_{i}, 64",
                                                    f"  %f4q_{i + 1} = vst.acc.post %f4v_{i}, %f4q_{i}, 64")]
        if xw4:
            e("  %qxp20 = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
            e("  %qxk2 = mova.i32 32")
            e("  %qxm2 = mov.modifier %qxk2")
            e("  %qxp2 = padda.modifier %qxp20, %qxm2")
            e("  %qxb = mov.cycles")
            e("  %qxbl = slice %qxb[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
            e("  %qxs = lda %qxp2, 28")
            e("  %qxd = sub %qxbl, %qxs")
            e("  %qxt = lda %qxp2, 24")
            e("  %qxn = add.rr %qxt, %qxd")
            e("  st %qxn, %qxp2, 24")
        qfw_wrap(e, "c", 48, [] if os.environ.get("GN_QFNOCOPY") else cl_, mode="2")   # (GN_QFNOCOPY: diag, no copies)
        e("  %f4R2 = mova.i32 2")
        e(f"  %f4RE = mova.i32 {1 if qfegr2_on() or QF4R[0]['l2'] else 2}")
        e(f"  rel %f4RE, {dp}")
        e(f"  rel %f4R2, {QF4R_NB}")
        e("  %f4ec2 = add %f4ec, 2")
        e("  st %f4ec2, %f4pS, 8")
        e("  %f4nc2 = add %f4nc, 2")
        e("  st %f4nc2, %f4pN4, 4")
    e("  %f4pN = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
    e(f"  %f4hv2 = lda %f4pN, {QF_SLAB}")
    e("  %f4hv3 = add %f4hv2, 1")
    e(f"  st %f4hv3, %f4pN, {QF_SLAB}")
    if xw and not xw4 and os.environ.get("GN_QFW") != "6":
        e("  %qxp20 = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
        e("  %qxk2 = mova.i32 32")
        e("  %qxm2 = mov.modifier %qxk2")
        e("  %qxp2 = padda.modifier %qxp20, %qxm2")
        e("  %qxb = mov.cycles")
        e("  %qxbl = slice %qxb[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
        e("  %qxs = lda %qxp2, 28")
        e("  %qxd = sub %qxbl, %qxs")
        e("  %qxt = lda %qxp2, 24")
        e("  %qxn = add.rr %qxt, %qxd")
        e("  st %qxn, %qxp2, 24")
    e("  low.br ^fzh")


def qdec_leaf(L, cfg, role=None):
    """(Q' NPU_QDEC) a decoder row's leaf: per firing (one a call) its column's whole panel decoded from its raw ring
    (port QF_PORTS[1], a step's record a slot) into private outputs and egressed (port QF_PORTS[0]) into the panel.
    role 0 / 1 (two decoder rows): 0 decodes slabs 0-1 and egresses all 4, slabs 2-3 from 1's channel (QDEC_RPORT);
    1 decodes slabs 2-3 straight into its sender (QDEC_SPORT), one record a step."""
    e = L.append
    q = QF["cfg"]
    D = decoder(q.fmt, q.gord, q.swiglu)
    dp, rp = QF_PORTS
    nm = "qdec" if role is None else f"qdec{role}"
    e(f"low.func.def schedule(locked) target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @{nm}() asm {{")
    if role != 1:
        e(f"  %fzd_r = resource<native_pointer> {{index = {dp}, source_type = buffer}} : reg<aie2p.ep>")
    else:
        e(f"  %qds_r = resource<native_pointer> {{index = {QDEC_SPORT}, source_type = buffer}} : reg<aie2p.ep>")
    if role == 0:
        e(f"  %qdr_r = resource<native_pointer> {{index = {QDEC_RPORT}, source_type = buffer}} : reg<aie2p.ep>")
    e(f"  %raw_r = resource<native_pointer> {{index = {rp}, source_type = buffer}} : reg<aie2p.ep>")
    e("  set.scd-enable 1")
    e("  set.rounding 12")
    e("  %fzs = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
    e("  %fzt = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
    e(f"  %dqa = storage {{byte_alignment = 64, byte_length = {D.SCR_B}}} : low.storage<private>")
    # (A, the outputs, then B: B starts > 16 KB after A, so in another bank: the table's second copy)
    ob = 2 * D.OUT_B * (NP if role is None else 2)
    pad = int(os.environ.get("GN_QDPAD", 64))   # (diag: a full bank's pad does not fit role 1 on columns 4-7)
    if role != 1:
        e(f"  %dqo = storage {{byte_alignment = 64, byte_length = {ob + (pad if role is not None else 0)}}} : low.storage<private>")
    else:
        e(f"  %dqp = storage {{byte_alignment = 64, byte_length = {pad}}} : low.storage<private>")
    e(f"  %dqb = storage {{byte_alignment = 64, byte_length = {D.SCR2_B}}} : low.storage<private>")
    if QFCYC:
        e("  %qfc = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
    e("  %fzp = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzps24 = mov.address-to-scalar %raw_r")
    e("  st %fzps24, %fzp, 24")
    if role != 1:
        e("  %fzps28 = mov.address-to-scalar %fzd_r")
        e("  st %fzps28, %fzp, 28")
    if role is not None:   # the outputs of the slabs not in %dqo: 1's sender (0, 1), the channel's record (2, 3)
        e("  %fzpTq = storage_address %fzt : low.storage<private> -> reg<aie2p.ep>")
        e(f"  %fzqb = mov.address-to-scalar {'%qds_r' if role == 1 else '%qdr_r'}")
        e(f"  %fzqk = mov.i32 {2 * D.OUT_B}")
        e("  %fzqb1 = add.rr %fzqb, %fzqk")
        o = QF_OUT if role == 1 else QF_OUT + 8
        e(f"  st %fzqb, %fzpTq, {o}")
        e(f"  st %fzqb1, %fzpTq, {o + 4}")
    e(f"  %fzgn0 = mov.i32 {q.passes * FUSE_SB}")
    e("  low.br ^fzdq(%fzgn0: reg<aie2p.er>)")
    QF["gate_now"] = None
    QF["drole"] = role
    qf_decode(L, q, dp)
    QF["drole"] = None
    e("^fzgemm:")
    e("  return")
    e("}")


def raw_stream_unit(e, D, RW, inb):
    """fuse_decode's raw unit from the core stream: every row's 4 super-blocks into the decoder's input record row
    (RECROW holds them) at inb, 16 words a step."""
    e(f"  %fzinp = mov.scalar-to-address {inb}")
    e("  %fz32 = mova.i32 32")
    e("  %fzm32 = mov.modifier %fz32")
    e(f"  %fzrr = mov.i32 {D.RECROW}")
    e("  %fz16r = mova.i32 16")
    e(f"  %fzws = mova.i32 {RW // 16}")
    e("  low.br ^fzrw(%fzR0: reg<aie2p.er>, %fzinp: reg<aie2p.ep>)")
    e("^fzrw(%fzri: reg<aie2p.er>, %fzrp: reg<aie2p.ep>):")
    e("  %fzrm = lt %fzri, %fz16r")
    e("  low.cond_br %fzrm, ^fzrb, ^fzrx : reg<aie2p.er>")
    e("^fzrb:")
    e("  %fzrpc = copy %fzrp : reg<aie2p.ep> -> reg<aie2p.ep>")
    e("  %fzrq = padds.modifier %fzrpc, %fzm32")      # 16 words a step around the pointer (offsets -32 .. 28)
    e("  low.br ^fzw(%fzR0: reg<aie2p.er>, %fzrq: reg<aie2p.ep>)")
    e("^fzw(%fzwi: reg<aie2p.er>, %fzwq: reg<aie2p.ep>):")
    e("  %fzwlm = lt %fzwi, %fzws")
    e("  low.cond_br %fzwlm, ^fzwb, ^fzwx : reg<aie2p.er>")
    e("^fzwb:")
    for w in range(16):   # (a stream read holds the core ~10 cycles: measured, no stall event)
        e(f"  %fzv{w} = mov.ss")
    for w in range(16):
        e(f"  st %fzv{w}, %fzwq, {4 * w - 32}")
    e("  %fzwq1 = padda %fzwq, 64")
    e("  %fzwi1 = add.rr %fzwi, %fzR1")
    e("  low.br ^fzw(%fzwi1: reg<aie2p.er>, %fzwq1: reg<aie2p.ep>)")
    e("^fzwx:")
    for w in range(RW % 16):                            # the row's last words
        e(f"  %fzt{w} = mov.ss")
        e(f"  st %fzt{w}, %fzwq, {4 * w - 32}")
    e("  %fzrpa = mov.address-to-scalar %fzrp")
    e("  %fzrpb = add.rr %fzrpa, %fzrr")
    e("  %fzrpn = mov.scalar-to-address %fzrpb")
    e("  %fzri1 = add.rr %fzri, %fzR1")
    e("  low.br ^fzrw(%fzri1: reg<aie2p.er>, %fzrpn: reg<aie2p.ep>)")
    e("^fzrx:")


def eg512(cfg):
    """fuse 2: every egress source is 64-byte aligned (scratch and channel rings are; records are 64-byte multiples
    from k-block offsets that are multiples of 4: 4 k-blocks are 576 bytes)."""
    return bool(cfg.eg512) and all(k % 4 == 0 for k in cfg.ks) and fuse_hx(cfg) % 4 == 0


def egress_record(e, t, src, cnt, ring, fw, slots=2, w512=False, acq=None):
    """One egress record: fw words from src (16-byte aligned: k-blocks are 144 bytes) into the egress ring slot of
    record count cnt mod slots, straight-line in chunks of up to 12 copies (all loads, then all stores), each through
    pointers 64 bytes in (immediates reach -128 .. 112; padda steps by multiples of 64). w512 (64-byte aligned src):
    64-byte copies in chunks of up to 7, loads alternating the A / B lanes, pointers 256 bytes in: every chunk's loads
    first (loom cannot prove a store leaves the next chunk's source alone), then acq (the ring slot's lock: the
    source is the core's own), then the stores. Returns (the source pointer now: padda consumes src, its byte offset
    from src)."""
    e(f"  %fzk1{t} = mova.i32 {slots - 1}")
    e(f"  %fzer{t} = and {cnt}, %fzk1{t}")
    e(f"  %fzerb{t} = mov.i32 {4 * fw}")
    e(f"  %fzeo{t} = mul %fzer{t}, %fzerb{t}")
    e(f"  %fzea{t} = add.rr {ring}, %fzeo{t}")
    off = 0
    if w512:
        assert fw % 16 == 0
        stores = []
        for c, q0 in enumerate(range(0, fw // 16, 7)):   # (padda steps reach 448)
            o = 64 * q0 + 256
            e(f"  %fzsx{t}_{c} = padda {src}, {o - off}")
            src, off = f"%fzsx{t}_{c}", o
            e(f"  %fzeb{t}_{c} = mov.i32 {o}")
            e(f"  %fzea{t}_{c} = add.rr %fzea{t}, %fzeb{t}_{c}")
            e(f"  %fzed{t}_{c} = mov.scalar-to-address %fzea{t}_{c}")
            qs = range(q0, min(q0 + 7, fw // 16))
            for q in qs:
                e(f"  %fzv{t}_{q} = {'vldb' if q % 2 else 'vlda'}.512.i32x16 {src}, {64 * (q - q0) - 256}")
            stores += [f"  vst.512.i32x16 %fzv{t}_{q}, %fzed{t}_{c}, {64 * (q - q0) - 256}" for q in qs]
        if acq:
            e(acq)
        for ln in stores:
            e(ln)
        return src, off
    if acq:
        e(acq)
    for c, q0 in enumerate(range(0, fw // 4, 12)):
        o = 16 * q0 + 64
        e(f"  %fzsx{t}_{c} = padda {src}, {o - off}")
        src, off = f"%fzsx{t}_{c}", o
        e(f"  %fzeb{t}_{c} = mov.i32 {o}")
        e(f"  %fzea{t}_{c} = add.rr %fzea{t}, %fzeb{t}_{c}")
        e(f"  %fzed{t}_{c} = mov.scalar-to-address %fzea{t}_{c}")
        qs = range(q0, min(q0 + 12, fw // 4))
        for q in qs:
            e(f"  %fzv{t}_{q} = vlda.128.i32x4 {src}, {16 * (q - q0) - 64}")
        for q in qs:
            e(f"  vst.128.i32x4 %fzv{t}_{q}, %fzed{t}_{c}, {16 * (q - q0) - 64}")
    return src, off


def kchain_relay_loop(e, dp, n, fw, hcap, w512):
    """kchain: ^fzrl: n records from the row below's ring (HELP_RELAY; %fzx 0, 4) into this helper's own (dp; count %fzs 8)."""
    e("^fzrl:")
    e("  %fzrE = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzrc = lda %fzrE, 8")
    e("  %fzrdl = lda %fzrE, 28")
    e("  %fzrX = storage_address %fzx : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzrh = lda %fzrX, 4")
    e("  %fzrhl = lda %fzrX, 0")
    e(f"  %fzrn = mov.i32 {n}")
    e("  low.br ^fzrlh(%fzE0: reg<aie2p.er>, %fzrc: reg<aie2p.er>, %fzrh: reg<aie2p.er>)")
    e("^fzrlh(%fzri: reg<aie2p.er>, %fzrcc: reg<aie2p.er>, %fzrhh: reg<aie2p.er>):")
    e("  %fzrm = lt %fzri, %fzrn")
    e("  low.cond_br %fzrm, ^fzrlb, ^fzrls : reg<aie2p.er>")
    e("^fzrlb:")
    e(f"  acq %fzEm1, {dp}")
    e(f"  acq %fzEm1, {HELP_RELAY}")
    e(f"  %fzrk = mova.i32 {hcap - 1}")
    e("  %fzrp = and %fzrhh, %fzrk")
    e(f"  %fzrb = mov.i32 {4 * fw}")
    e("  %fzro = mul %fzrp, %fzrb")
    e("  %fzra = add.rr %fzrhl, %fzro")
    e("  %fzrsp = mov.scalar-to-address %fzra")
    egress_record(e, "r", "%fzrsp", "%fzrcc", "%fzrdl", fw, hcap, w512)
    e(f"  rel %fzE1, {HELP_RELAY}")
    e(f"  rel %fzE1, {dp}")
    e("  %fzrc1 = add.rr %fzrcc, %fzE1")
    e("  %fzrh1 = add.rr %fzrhh, %fzE1")
    e("  %fzri1 = add.rr %fzri, %fzE1")
    e("  low.br ^fzrlh(%fzri1: reg<aie2p.er>, %fzrc1: reg<aie2p.er>, %fzrh1: reg<aie2p.er>)")
    e("^fzrls:")
    e("  %fzrF = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  st %fzrcc, %fzrF, 8")
    e("  %fzrG = storage_address %fzx : low.storage<private> -> reg<aie2p.ep>")
    e("  st %fzrhh, %fzrG, 4")
    e("  low.br ^fzex")


def kchain_up(e, pieces, fw, OUT, s0, hcap, w512):
    """kchain: the pieces [(first k-block, words)] from the decoded super-block into the ring to the row above (its
    pointer and record count in %fzx at 12, 16)."""
    e("  %fzqx = storage_address %fzx : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzqn = lda %fzqx, 16")
    e("  %fzql = lda %fzqx, 12")
    for i, (kb0, sw) in enumerate(pieces):
        t = f"q{i}"
        e(f"  %fzoo{t} = mov.i32 {OUT + (kb0 - 32 * s0) * 144}")
        e(f"  %fzsrc{t} = add.rr %fzsb2, %fzoo{t}")
        e(f"  %fzsp{t} = mov.scalar-to-address %fzsrc{t}")
        e(f"  %fznr{t} = mov.i32 {sw // fw}")
        cnt = "%fzqn" if i == 0 else f"%fzqd{i - 1}"
        e(f"  low.br ^fze{t}h(%fzE0: reg<aie2p.er>, %fzsp{t}: reg<aie2p.ep>, {cnt}: reg<aie2p.er>)")
        e(f"^fze{t}h(%fzei{t}: reg<aie2p.er>, %fzes{t}: reg<aie2p.ep>, %fzec{t}: reg<aie2p.er>):")
        e(f"  %fzem{t} = lt %fzei{t}, %fznr{t}")
        e(f"  low.cond_br %fzem{t}, ^fze{t}b, ^fze{t}s : reg<aie2p.er>")
        e(f"^fze{t}b:")
        e(f"  acq %fzEm1, {HELP_SEND}")
        sp_, b_ = egress_record(e, t, f"%fzes{t}", f"%fzec{t}", "%fzql", fw, hcap, w512)
        e(f"  rel %fzE1, {HELP_SEND}")
        e(f"  %fzcs{t} = padda {sp_}, {4 * fw - b_}")
        e(f"  %fzec1{t} = add.rr %fzec{t}, %fzE1")
        e(f"  %fzei1{t} = add.rr %fzei{t}, %fzE1")
        e(f"  low.br ^fze{t}h(%fzei1{t}: reg<aie2p.er>, %fzcs{t}: reg<aie2p.ep>, %fzec1{t}: reg<aie2p.er>)")
        e(f"^fze{t}s:")
        e(f"  %fzqd{i} = copy %fzec{t} : reg<aie2p.er> -> reg<aie2p.er>")
    e(f"  %fzqy = storage_address %fzx : low.storage<private> -> reg<aie2p.ep>")
    e(f"  st %fzqd{len(pieces) - 1}, %fzqy, 16")


def fuse_egress_helped(e, dp, nr, fw, own, xr=0, hcap=2, w512=False, nh=None):
    """fuse_decode's egress for a helped head (ghelp): per unit, the helper's nr records (each copied from the helper's
    channel ring, slot: the helper record count mod hcap), then its own nr from the decoded super-blocks at own (bytes
    from the scratch base), one record loop around a single copy (program memory). hsplit: the last xr of its own come
    from the helper too (after its nr)."""
    nh = nr if nh is None else nh                      # (the helper's records: thelp's tail slice differs)
    e(f"  %fzmn = mov.i32 {nh}")
    if xr:
        e(f"  %fzmx = mov.i32 {nh + nr - xr - 1}")
    e(f"  %fzm2n = mov.i32 {nh + nr}")
    e(f"  %fzmb = mov.i32 {4 * fw}")
    e(f"  %fzmoo = mov.i32 {own}")
    e("  %fzmo0 = add.rr %fzsb2, %fzmoo")
    e("  %fzpE = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzen = lda %fzpE, 8")                         # the egress record count (its parity: the ring slot)
    e("  %fzdl = lda %fzpE, 28")                        # the egress ring
    e("  %fzxE = storage_address %fzx : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzhn = lda %fzxE, 4")                         # the helper record count
    e("  %fzhl = lda %fzxE, 0")                         # the helper's ring
    e("  low.br ^fzm(%fzE0: reg<aie2p.er>, %fzen: reg<aie2p.er>, %fzhn: reg<aie2p.er>, %fzmo0: reg<aie2p.er>)")
    e("^fzm(%fzmi: reg<aie2p.er>, %fzmc: reg<aie2p.er>, %fzmh: reg<aie2p.er>, %fzmo: reg<aie2p.er>):")
    e("  %fzmm = lt %fzmi, %fzm2n")
    e("  low.cond_br %fzmm, ^fzmb, ^fzms : reg<aie2p.er>")
    e("^fzmb:")
    e(f"  acq %fzEm1, {dp}")
    helped_rec(e, "%fzmw", xr)
    e("  low.cond_br %fzmw, ^fzmhh, ^fzmoo : reg<aie2p.er>")
    e("^fzmhh:")                                        # a helper record: its slot
    e(f"  acq %fzEm1, {HELP_RECV}")
    e(f"  %fzmhk = mova.i32 {hcap - 1}")
    e("  %fzmhp = and %fzmh, %fzmhk")
    e("  %fzmho = mul %fzmhp, %fzmb")
    e("  %fzmha = add.rr %fzhl, %fzmho")
    e("  %fzmh1 = add.rr %fzmh, %fzE1")
    e("  low.br ^fzmcp(%fzmha: reg<aie2p.er>, %fzmh1: reg<aie2p.er>, %fzmo: reg<aie2p.er>)")
    e("^fzmoo:")                                        # an own record
    e("  %fzmo1 = add.rr %fzmo, %fzmb")
    e("  low.br ^fzmcp(%fzmo: reg<aie2p.er>, %fzmh: reg<aie2p.er>, %fzmo1: reg<aie2p.er>)")
    e("^fzmcp(%fzmsrc: reg<aie2p.er>, %fzmh2: reg<aie2p.er>, %fzmo2: reg<aie2p.er>):")
    e("  %fzmsp = mov.scalar-to-address %fzmsrc")
    egress_record(e, "m", "%fzmsp", "%fzmc", "%fzdl", fw, w512=w512)
    helped_rec(e, "%fzmw2", xr)
    e("  low.cond_br %fzmw2, ^fzmrh, ^fzmj : reg<aie2p.er>")
    e("^fzmrh:")
    e(f"  rel %fzE1, {HELP_RECV}")
    e("  low.br ^fzmj")
    e("^fzmj:")
    e(f"  rel %fzE1, {dp}")
    e("  %fzmc1 = add.rr %fzmc, %fzE1")
    e("  %fzmi1 = add.rr %fzmi, %fzE1")
    e("  low.br ^fzm(%fzmi1: reg<aie2p.er>, %fzmc1: reg<aie2p.er>, %fzmh2: reg<aie2p.er>, %fzmo2: reg<aie2p.er>)")
    e("^fzms:")   # (the storage pointers re-derived: none lives across the copies)
    e("  %fzpF = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  st %fzmc, %fzpF, 8")
    e("  %fzxF = storage_address %fzx : low.storage<private> -> reg<aie2p.ep>")
    e("  st %fzmh, %fzxF, 4")
    e("  low.br ^fzex")


def helped_rec(e, v, xr):
    """v = 1 when fuse_egress_helped's record %fzmi is the helper's (i < nr, hsplit: or i > 2 nr - xr - 1)."""
    if not xr:
        e(f"  {v} = lt %fzmi, %fzmn")
        return
    e(f"  {v}a = lt %fzmi, %fzmn")
    e(f"  {v}b = lt %fzmx, %fzmi")
    e(f"  {v} = add.rr {v}a, {v}b")


def fuse_egress_rotated(e, cfg, dp, cov, pieces, fw, OUT, k0, k1, s0):
    """fuse_decode's egress as one record loop (program memory: a loop per slice cost ~300 bytes each): the slices of
    an unpadded cover are a rotation of the decoded k-blocks k0 .. k1 (the filler's own slice last), so the source walks
    from the first slice's start and wraps once at k1."""
    lo, hi = OUT + (k0 - 32 * s0) * 144, OUT + (k1 - 32 * s0) * 144
    start = OUT + (pieces[0][0] - 32 * s0) * 144
    nrec = sum(sw for _, sw in pieces) // fw
    assert all(sw % fw == 0 for _, sw in pieces) and (start - lo) % (4 * fw) == 0
    e(f"  %fzoo = mov.i32 {start}")
    e("  %fzsrc = add.rr %fzsb2, %fzoo")
    e(f"  %fzohi = mov.i32 {hi}")
    e("  %fzhi = add.rr %fzsb2, %fzohi")
    e("  %fzpE = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
    e("  %fzen = lda %fzpE, 8")                         # the egress record count (its parity: the ring slot)
    e("  %fzdl = lda %fzpE, 28")                        # the egress ring
    e(f"  %fznr = mov.i32 {nrec}")
    e(f"  %fzspan = mov.i32 {hi - lo}")
    e("  %fzB1 = mova.i32 1")
    e("  %fzBm1 = mova.i32 -1")
    e("  low.br ^fzeh(%fzE0: reg<aie2p.er>, %fzsrc: reg<aie2p.er>, %fzen: reg<aie2p.er>)")
    e("^fzeh(%fzei: reg<aie2p.er>, %fzes: reg<aie2p.er>, %fzec: reg<aie2p.er>):")
    e("  %fzem = lt %fzei, %fznr")
    e("  low.cond_br %fzem, ^fzeb, ^fzes : reg<aie2p.er>")
    e("^fzeb:")
    e(f"  acq %fzBm1, {dp}")
    e("  %fzesp = mov.scalar-to-address %fzes")
    egress_record(e, "", "%fzesp", "%fzec", "%fzdl", fw, w512=eg512(cfg))
    e(f"  rel %fzB1, {dp}")
    e("  %fzec1 = add.rr %fzec, %fzB1")
    e("  %fzei1 = add.rr %fzei, %fzB1")
    e(f"  %fzstep = mov.i32 {4 * fw}")
    e("  %fzns = add.rr %fzes, %fzstep")
    e("  %fzin = lt %fzns, %fzhi")          # 1: below k1's end, 0: wrap to k0
    e("  %fzwr0 = mul %fzin, %fzspan")
    e("  %fzwr1 = sub %fzns, %fzspan")
    e("  %fzns2 = add.rr %fzwr1, %fzwr0")
    e("  low.br ^fzeh(%fzei1: reg<aie2p.er>, %fzns2: reg<aie2p.er>, %fzec1: reg<aie2p.er>)")
    e("^fzes:")
    e("  st %fzec, %fzpE, 8")
    e("  low.br ^fzex")


MBNAME = [None]   # (GN_QMB1 diag) the leaf being generated

def leaf(L, cfg, role, ks, gate=None, cov=None, name=None, helper=None, pair=None):
    e = L.append
    tail = role == "tail"
    fa = slab(ks)
    acap = acap_tail(cfg) if tail else cfg.acap
    a_adv = acap == MP and not (qa3_on() or qa4_on())   # a ring of MP slabs is addressed like a whole record: the base advances per iteration
    # mu 1 with a 2-slot ring: the slot alternates per iteration (body: the base steps +1 / -1 slab by parity)
    assert MP % cfg.mu == 0 and (a_adv or cfg.mu % acap == 0 or (cfg.mu, acap) in ((1, 2), (1, 3), (1, 4))), \
        "ring slot of a row must be static per iteration"
    name = name or role + ("_g" if gate else "")
    full_name = name
    name = name[:-3] if name.endswith(("_pw", "_pe")) and pair else name   # (Q' pairs: the role rides in pair)
    name = name[:-3] if name.endswith(("_gw", "_gr")) and gate else name   # (the gate role rides in gate)
    skp1 = name in ("tailm0s", "tailms")   # (P1K fill) a P1 leaf that skips the raw records: its pair role without the s
    dfill = name == "tailm0d"   # (GN_QFKH) row 0: tailm0 that decodes K half 0
    f4 = name in ("tailm0e", "tailmh")   # (GN_QF4R) rows 2 / 3: tailm0 / tailm that decode too
    name = name[:-1] if skp1 or dfill or f4 else name
    fills = cov is not None
    QF_HALF[0] = (0 if dfill else 1) if fills and qfkh_on() else None
    QF4R[0] = None
    if fills and qf4r_on():   # (rows 0 / 1 half 1, rows 2 / 3 half 0; helpers 0 / 3 take the odd steps)
        hlp4 = full_name.startswith(("tailmh",) + (() if qf4r_swap() else ("tailm0d",))) or (
            qf4r_swap() and full_name.split("_")[0] == "tailm")
        QF_HALF[0] = 0 if full_name.startswith(("tailm0e", "tailmh")) else 1
        # (own records: multicast 0's rows (tailm0*) the even ones)
        QF4R[0] = dict(par=0 if full_name.startswith("tailm0") else 1, help=hlp4,
                       l2=qflane2_side("e" if pair == "e" else "w"))   # (GN_QFLANE2: this column's lanes)
    lk = "" if tail and os.environ.get("GN_QUNLOCK") else "schedule(locked) "   # (probe: Loom's scheduler orders it)
    e(f"low.func.def {lk}target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @{full_name}() asm {{")
    split_r1 = QF["cfg"] is not None and bool(QF.get("split")) and name.startswith("tailr")   # (row 1 decodes too)
    QODS_EARLY[0] = split_r1
    rs = "_r" if fills or split_r1 else ""   # parked as scalars across the decoding fill (^fzgemm / ^rzgo)
    e(f"  %a{rs} = resource<native_pointer> {{index = 0, source_type = buffer}} : reg<aie2p.ep>")
    if fills:   # the W ring's first slot and the fused fill's egress ring (fuse_prologue; a helper's: its channel to
        # the head, whose ring the head reads it from)
        e(f"  %w0{rs} = resource<native_pointer> {{index = 1, source_type = buffer}} : reg<aie2p.ep>")
        dp = HELP_SEND if (helper or "")[:1] == "M" else fuse_ports(role)[0]
        e(f"  %fzd{rs} = resource<native_pointer> {{index = {dp}, source_type = buffer}} : reg<aie2p.ep>")
        if helper == "H" or "h" in (helper or ""):
            rp = HELP_RELAY if helper[:1] == "M" else HELP_RECV
            e(f"  %fzh{rs} = resource<native_pointer> {{index = {rp}, source_type = buffer}} : reg<aie2p.ep>")
        if QF4R[0] and not QF4R[0]["help"]:   # (GN_QF4R) the helper's ring, through this core's window
            e(f"  %fzn{rs} = resource<native_pointer> {{index = {QF4R_NB}, source_type = buffer}} : reg<aie2p.ep>")
        if "u" in (helper or ""):   # (kchain: the ring to the row above)
            e(f"  %fzq{rs} = resource<native_pointer> {{index = {HELP_SEND}, source_type = buffer}} : reg<aie2p.ep>")
    elif name == "tailm" and os.environ.get("GN_QMB1") == "1":   # (GN_QMB1 diag, synthetic 2-row Q': row 1 pops row 0's W slots through its south window;
        # the multicast gives both rows the same W and a slot refills only once both rows released it)
        e("  %wmb = resource<native_pointer> {index = 1, source_type = buffer} : reg<aie2p.ep>")
    else:
        e(f"  %w{rs} = resource<native_pointer> {{index = 1, source_type = buffer}} : reg<aie2p.ep>")
    if split_r1:   # (the split channel's ring: slab 3's output)
        e("  %qss_r = resource<native_pointer> {index = 3, source_type = buffer} : reg<aie2p.ep>")
    if fills and QF.get("split"):   # (row 0: row 1's slab-3 ring, parked: fuse2_entry)
        e(f"  %qsr_r = resource<native_pointer> {{index = {QF['split'](gate)}, source_type = buffer}} : reg<aie2p.ep>")
    if fills and QF.get("crel"):   # (Q' C relay) row 1's C ring (parked: fuse2_entry)
        e(f"  %crb_r = resource<native_pointer> {{index = {QF['crel'](gate)}, source_type = buffer}} : reg<aie2p.ep>")
    if gate:   # the neighbor channel to / from the relay (leaf-synchronized; a filling gate core parks it: fuse2_entry)
        gp = gate_ports(cfg)[gate]
        e(f"  %gn1{rs} = resource<native_pointer> {{index = {gp['nbr']}, source_type = buffer}} : reg<aie2p.ep>")
    if tail:
        e(f"  %o{rs} = resource<native_pointer> {{index = 2, source_type = buffer}} : reg<aie2p.ep>")
    if name == "tailm0" and os.environ.get("GN_QMB1") == "5":
        e("  %mbt3 = resource<native_pointer> {index = 3, source_type = buffer} : reg<aie2p.ep>")
    if name == "tailm" and os.environ.get("GN_QMB1") == "5":
        e("  %mbt4 = resource<native_pointer> {index = 4, source_type = buffer} : reg<aie2p.ep>")
    if pair and not qpair_off():   # (its resource here; Q' parks it after the C relay's resources)
        pair_park(e, "res" if QF.get("cfg") is not None else "both")
    if name in ("tailm", "tailm0") and os.environ.get("GN_QMB1") == "5":   # (MB2 tokens: ports 3 own, 4 other)
        pass
    if name == "tailm" and os.environ.get("GN_QMB1") == "1":   # (GN_QMB1, after the resources)
        e("  %wmbs = mov.address-to-scalar %wmb")
        e(f"  %wmbk = mov.i32 {int(os.environ.get('GN_QMB1_DELTA', 0x40000 - 0x70000))}")
        e("  %wmbt = add.rr %wmbs, %wmbk")
        e("  %w = mov.scalar-to-address %wmbt")
    crmid = int(name[5]) if QF.get("crelmid") and name[:5] == "tailr" and name[5:6].isdigit() else 0
    CRMID[0] = crmid
    MBNAME[0] = name   # (GN_QMB1 diag: body reads it)
    # (C relay, a row 0 that does not fill: NPU_QDEC) row 1's C ring and this core's, parked for crel_copy
    crel0 = bool(QF.get("crel")) and bool(QF.get("dec")) and not fills and role == "tail" and not name.startswith("tailr")
    crel0 = crel0 or (crelp_on() and name == "tailm0")   # (NPU_QCREL=2: a P1 pair's row 0 relays row 1's C)
    CREL0[0] = crel0
    if crmid or crel0:   # (C relay chain, a middle row) the next row's C ring and this core's, parked for crel_copy
        crp = QF["crelmid"] if crmid else 5 if crelp_on() else QF["crel"](gate)
        e(f"  %crb{rs} = resource<native_pointer> {{index = {crp}, source_type = buffer}} : reg<aie2p.ep>")
        e("  %crst = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
        e("  %crstp = storage_address %crst : low.storage<private> -> reg<aie2p.ep>")
        e(f"  %crx0 = mov.address-to-scalar %crb{rs}")
        e("  st %crx0, %crstp, 0")
        e(f"  %crx4 = mov.address-to-scalar %o{rs}")
        e("  st %crx4, %crstp, 4")
    if pair and QF.get("cfg") is not None and not qpair_off():
        pair_park(e, "park")
    early = fills   # the decoding fill first: nothing of the GEMM's lives across the decoder
    if gate:   # job state (private storage, zeroed by the setup): the first job's gate before any firing
        e("  %gst = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
        if not early:   # (fuse 2: at a call's first firing, fuse2_entry)
            e("  %gsp0 = storage_address %gst : low.storage<private> -> reg<aie2p.ep>")
            gate_first(e, cfg, gate, "%gn1")
    if role != "head":
        e("  set.scd-enable 1")
    if role != "tail":
        e("  set.mcd-enable 1")
    else:
        e("  set.rounding 12")   # round to nearest even (the bf16 C)
    if early:
        fuse_prologue(L, cfg, role, ks, cov, helper, gate)
    elif split_r1:
        qf_split_r1(L, cfg)
    elif QF["cfg"] is not None and (name.startswith("tailr") or skp1) and not QF.get("dec"):
        qf_skip_raw(e, cfg)
    if early:   # (fuse2_entry)
        e("  %fzpL = storage_address %fzs : low.storage<private> -> reg<aie2p.ep>")
        e("  %conf = lda %fzpL, 16")
    else:   # a load, not a constant: the allocator would rematerialize a constant at every MMA (one MOVA each)
        if qalt_on() and not qaltp_on() and role == "tail":   # (O first in the private block: no 1 KB pad ahead of it)
            e(f"  %qods = storage {{byte_alignment = 1024, byte_length = {qalt_sizes()[1]}}} : low.storage<private>")
        e("  %cfs = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
        e("  %cfp = storage_address %cfs : low.storage<private> -> reg<aie2p.ep>")
        e("  %conf0 = mova.i32 780")
        e("  st %conf0, %cfp, 0")
    e("  %am1 = mova.i32 -1")
    if tail and os.environ.get("GN_QSPIN", "").startswith("d"):   # (diag) a straight-line delay: N dependent adds
        e("  %dl0 = mova.i32 1")
        n = int(os.environ["GN_QSPIN"][1:])
        for i in range(n):
            e(f"  %dl{i + 1} = add %dl{i}, 3")
        e(f"  st %dl{n}, %cfp, 16")
    elif tail and os.environ.get("GN_QSPIN") == "fwd":   # (diag) one forward branch: a block boundary, no loop
        e("  low.br ^spdone")
        e("^spdone:")
    elif tail and os.environ.get("GN_QSPIN"):   # (diag) a spin of GN_QSPIN iterations before the GEMM, every firing
        sp = os.environ["GN_QSPIN"]
        e(f"  %spn = mova.i32 {int(sp[1:])}" if sp[0] == "a" else f"  %spn = mov.i32 {int(sp)}")
        e("  low.br ^spin(%spn: reg<aie2p.er>)")
        e("^spin(%spv: reg<aie2p.er>):")
        e("  %spm = eqz %spv")
        e("  low.cond_br %spm, ^spdone, ^spb : reg<aie2p.er>")
        e("^spb:")
        e("  %spv1 = add %spv, -1")
        e("  low.br ^spin(%spv1: reg<aie2p.er>)")
        e("^spdone:")
    if early and not (tail and qpass_on()):
        e("  acq %am1, 1")
    e("  %addmode = mova.i32 60")
    if role != "head":
        e("  %qsel0 = mova.i32 0")
    e("  %one = mova.i32 1")
    e("  %zero = mova.i32 0")
    qpass = tail and qpass_on()
    e(f"  %nmp = mova.i32 {MP * cfg.passes if qpass else MP // cfg.mu}")
    e(f"  %nnp = mova.i32 {NP}")
    if not early:
        e("  %conf = lda %cfp, 0")   # (the store above has settled: a load right after its store reads stale)
    e("  %k144 = mova.i32 16")
    qbank = tail and os.environ.get("GN_QBANK") in ("1", "2", "3") and os.environ.get("GN_QCORE")
    q3 = qbank and os.environ.get("GN_QBANK") == "3"
    if qbank:   # (QBANK_RING_AT pins the slots) A slot 1 a 16 KB bank above slot 0 (a_next); C rows (body)
        assert (cfg.mu, acap) in ((1, 2), (1, 3), (1, 4)) and not a_adv and MP in (4, 5) and NP in (4, 5, 6) and fa <= (4608 if qp1_on() else 4096)
    assert not q3 or qpass, "GN_QBANK=3 needs GN_QPASS=1"
    e(f"  %kks = mova.i32 {fa // 16}")
    e("  %fstep = mul %k144, %kks")   # slab stride
    e("  %mf = mov.modifier %fstep")
    e("  %k512 = mova.i32 512")
    e("  %k2 = mova.i32 2")
    e("  %cstep = mul %k512, %k2")
    e("  %mcs = mov.modifier %cstep")
    e("  %mz = mov.modifier %zero")
    if q3:   # the odd sub-tiles' area; %qodb: row 0 sub-tile 1's pointer (the odd chain's first)
        sp = "scratch" if qcb1_on() or qa4_on() else "private"   # (GN_QCB1: its own block, placed apart from the private one)
        so_ = qalt_sizes()[1] if qalt_on() else MP * q3_strides()[1]
        if qalt_on():   # (E in the scratch block, O in the private one: each fits its bank beside an A slot)
            sp = "private"
        if qaltp_on():   # (the areas sit at the C pins: O from the ring's base)
            E_, O_ = qaltp_eo()
            e(f"  %qoeo = mov.i32 {O_ - E_}")
            e("  %qoem = mov.modifier %qoeo")
            e("  %qodx = copy %o : reg<aie2p.ep> -> reg<aie2p.ep>")
            e("  %qod = padds.modifier %qodx, %qoem")
        else:
            if not qalt_on() and not QODS_EARLY[0]:   # (row 1 split: declared ahead of its fill)
                e(f"  %qods = storage {{byte_alignment = 1024, byte_length = {so_}}} : low.storage<{sp}>")
            e(f"  %qod = storage_address %qods : low.storage<{sp}> -> reg<aie2p.ep>")
        e("  %qodc = copy %qod : reg<aie2p.ep> -> reg<aie2p.ep>")
        bump(L, "%qodb", "%qodc", 512)
        if qaltp_on():
            e("  %qev = copy %o : reg<aie2p.ep> -> reg<aie2p.ep>")
            e(f"  %qd0 = mov.i32 {E_ - O_ - qalt_sizes()[1]}")
        elif qalt_on():
            e(f"  %qevs = storage {{byte_alignment = 1024, byte_length = {qalt_sizes()[0]}}} : low.storage<scratch>")
            e("  %qev = storage_address %qevs : low.storage<scratch> -> reg<aie2p.ep>")
            e("  %qevsc = mov.address-to-scalar %qev")
            e("  %qodsc = mov.address-to-scalar %qod")
            e("  %qdd = sub %qevsc, %qodsc")
            e(f"  %qkso = mov.i32 {-qalt_sizes()[1]}")
            e("  %qd0 = add.rr %qdd, %qkso")
    if tail:
        # pass counter in private storage; C segments are acquired on the first pass and released after the last
        e("  %cst = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
        e("  %cnp = storage_address %cst : low.storage<private> -> reg<aie2p.ep>")
        grp = groups(cfg)
        if qpass and qa3_on():   # the firing's first A slot; the next firing's (64 rows: +1 mod 3)
            e("  %qai = lda %cnp, 12")
            e("  %q3s8 = mova.i32 8")
            e("  %qak = mul %k512, %q3s8")
            e("  %qaof = mul %qai, %qak")
            e("  %qaom = mov.modifier %qaof")
            e("  %qa0c = copy %a : reg<aie2p.ep> -> reg<aie2p.ep>")
            e("  %qa0 = padds.modifier %qa0c, %qaom")
            e("  %qai1 = add.rr %qai, %one")
            e("  %qak3 = mova.i32 3")
            e("  %qaw = lt %qai1, %qak3")
            e("  %qain = mul %qai1, %qaw")
            e("  st %qain, %cnp, 12")
        if qpass and qa4_carry(cfg):   # the firing's first A slot %qai; the next firing's: + MP * passes, mod 4
            a_ = qa4_slots()
            e("  %qai = lda %cnp, 12")
            e(f"  %q4d0 = mov.i32 {a_[1] - a_[0]}")
            e("  %q4s0 = mul %qai, %q4d0")
            if len(a_) == 4:
                e(f"  %q4d1 = mov.i32 {a_[2] - a_[1] - (a_[1] - a_[0])}")
                e("  %q4g2 = lt %one, %qai")
                e("  %q4s1 = mul %q4g2, %q4d1")
                e("  %qaof = add.rr %q4s0, %q4s1")
            else:
                e("  %qaof = copy %q4s0 : reg<aie2p.er> -> reg<aie2p.er>")
            e("  %qaom = mov.modifier %qaof")
            e("  %qa0c = copy %a : reg<aie2p.ep> -> reg<aie2p.ep>")
            e("  %qa0 = padds.modifier %qa0c, %qaom")
            e(f"  %q4nr = mova.i32 {MP * cfg.passes % len(a_)}")
            e("  %q4n1 = add.rr %qai, %q4nr")
            e(f"  %q4n3 = mova.i32 {len(a_) - 1}")
            e("  %qain = and %q4n1, %q4n3")
            e("  st %qain, %cnp, 12")
        if grp:
            # %cnt: the pass within the group; %gs: the group (0 / 1); %gd: the passes done in the group's phase
            ga, gb = grp
            e("  %cnt = lda %cnp, 0")
            e("  %gs = lda %cnp, 4")
            e("  %gd = lda %cnp, 8")
            e(f"  %gdiff = mova.i32 {gb - ga}")
            e(f"  %ga = mova.i32 {ga}")
            e("  %gsd = mul %gs, %gdiff")
            e("  %gsz = add.rr %ga, %gsd")
            e("  %first = lt %cnt, %one")
            e(f"  %gam2 = mova.i32 {ga - 2}")
            e("  %pm2 = add.rr %gam2, %gsd")
            e("  %clast = lt %pm2, %cnt")
        elif qpass:   # (GN_QPASS) one firing per M block: every firing is its first and last pass
            e("  %first = mova.i32 1")
            e("  %clast = mova.i32 1")
        else:
            e(f"  %npass = mova.i32 {cfg.passes}")
            e("  %cnt = lda %cnp, 0")
            e("  %first = lt %cnt, %one")
            e(f"  %pm2 = mova.i32 {cfg.passes - 2}")
            e("  %clast = lt %pm2, %cnt")
        if QCYC:   # (diag) the first firing's start cycle (word 0; word 4 set once it is written)
            e("  %tcs = storage {byte_alignment = 64, byte_length = 64} : low.storage<private>")
            e("  %tca = storage_address %tcs : low.storage<private> -> reg<aie2p.ep>")
            e("  %t0w = mov.cycles")
            e("  %t0l = slice %t0w[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
            e("  %tfl = lda %tca, 4")
            e("  %tso = lda %tca, 0")
            e("  %tfirst = lt %tfl, %one")
            e("  %tdl = sub %t0l, %tso")
            e("  %tdm = mul %tdl, %tfirst")
            e("  %tsn = add.rr %tso, %tdm")
            e("  st %tsn, %tca, 0")
            e("  st %one, %tca, 4")
        if cfg.dfr:   # the M block (%blk): its C on ring half blk & 1; %nfb: not the call's first, %lastb: its last
            assert cfg.swiglu and cfg.mu == 1 and not grp and cfg.nb % 2 == 0 and cfg.passes > MP
            e("  %blk = lda %cnp, 4")
            e("  %bpar = and %blk, %one")
            e("  %nfb = lt %zero, %blk")
            e(f"  %nbm2 = mova.i32 {cfg.nb - 2}")
            e("  %lastb = lt %nbm2, %blk")
            e("  %clb = mul %clast, %lastb")
            e(f"  %khalf = mov.i32 {MP * SW_SLOT}")
            e("  %ohoff = mul %bpar, %khalf")
            e("  %ohm = mov.modifier %ohoff")
            e("  %oh0 = copy %o : reg<aie2p.ep> -> reg<aie2p.ep>")
            e("  %oh = padds.modifier %oh0, %ohm")
            # %dfx: this firing (pass s < MP of a block after the first) runs segment s of the previous block,
            # whose slot (%dss) is on the other half
            e(f"  %kmpd = mova.i32 {MP}")
            e("  %dlt = lt %cnt, %kmpd")
            e("  %dfx = mul %dlt, %nfb")
            e("  %os = mov.address-to-scalar %o")
            e("  %opar = sub %one, %bpar")
            e("  %dho = mul %opar, %khalf")
            e(f"  %kslt = mov.i32 {SW_SLOT}")
            e("  %dso = mul %cnt, %kslt")
            e("  %dsh = add.rr %os, %dho")
            e("  %dss = add.rr %dsh, %dso")
        # first pass: C loads read a 1 KB zero buffer with step 0
        if not qa4_on():
            e("  %zst = storage {byte_alignment = 64, byte_length = 1024} : low.storage<private>")
            e("  %zbp = storage_address %zst : low.storage<private> -> reg<aie2p.ep>")
        if cfg.dfr:   # the zero buffer's address: the deferred store fifo's dummy source
            e("  %zsc0 = mov.address-to-scalar %zbp")
        if cfg.dfr and cfg.dfi:   # the inline push's target in a firing without a deferred segment
            e("  %dmst = storage {byte_alignment = 64, byte_length = 1024} : low.storage<private>")
            e("  %dmsp = storage_address %dmst : low.storage<private> -> reg<aie2p.ep>")
            e("  %dmsc = mov.address-to-scalar %dmsp")
        if qpass:   # the 4 C slots and the first W slot acquired, the C slots zeroed (every pass loads its C)
            assert MP in (4, 5) and not cfg.dfr and not grp
            for _ in range(0 if qalt_on() and not qaltp_on() else 1 if qclazy_on() else MP):
                e("  acq %am1, 2")
            e("  acq %am1, 1")
            e("  %qzacc = acc.clear.f32x64")
            e("  %qzp = slice %qzacc[0] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>")
            e("  %qk64 = mova.i32 64")
            cs = qpass_cslots()
            zb = ([(r, b, off - cs[0] + 1024 * b + 512, "%o") for r, off in enumerate(cs) for b in range(NP)] if not q3 else
                  [(a, b, 1024 * b + 512, base) for a, (base, st_) in enumerate(zip(("%o", "%qod"), q3_strides()))
                   for b in range(MP * st_ // 1024)] if not qalt_on() else
                  [(a, b, 1024 * b + 512, base) for a, (base, sz_) in enumerate(zip(("%qev", "%qod"), qalt_sizes()))
                   for b in range(sz_ // 1024)])
            for r, b, off, base in ([] if qzacc_on() or os.environ.get("GN_QZACC") == "n" else zb):   # (n: diag, wrong)
                e(f"  %qzm{r}_{b} = mova.i32 {off // 64}")
                e(f"  %qzo{r}_{b} = mul %qzm{r}_{b}, %qk64")
                e(f"  %qzd{r}_{b} = mov.modifier %qzo{r}_{b}")
                e(f"  %qzc{r}_{b} = copy {base} : reg<aie2p.ep> -> reg<aie2p.ep>")
                e(f"  %qzb{r}_{b} = padds.modifier %qzc{r}_{b}, %qzd{r}_{b}")
                for q in range(16):
                    e(f"  vst.acc %qzp, %qzb{r}_{b}, {64 * q - 512}")
        e("  low.cond_br %first, ^zfill, ^nofill : reg<aie2p.er>" if not qpass else "  low.br ^nofill")
        e("^nofill:")
        e(f"  %olc = copy {'%qev' if qalt_on() else '%oh' if cfg.dfr else '%o'} : reg<aie2p.ep> -> reg<aie2p.ep>")
        bump(L, "%olb", "%olc", 512)
        e("  low.br ^start(%olb: reg<aie2p.ep>)")
        if not qa4_on():
            e("^zfill:")
            e("  %zacc = acc.clear.f32x64")
            e("  %zp = slice %zacc[0] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>")
            e("  %zbc = copy %zbp : reg<aie2p.ep> -> reg<aie2p.ep>")
            bump(L, "%zbb", "%zbc", 512)
            for q in range(16):
                e(f"  vst.acc %zp, %zbb, {64 * q - 512}")
            e("  low.br ^start(%zbb: reg<aie2p.ep>)")
        e("^start(%plb: reg<aie2p.ep>):")
        if qpass:   # (the W slot as a scalar: the loop carries it)
            e("  %wsb = mov.address-to-scalar %w")
            if cfg.passes % 2:   # odd passes: the W slot parity %qwp runs on across firings (private word 16)
                e("  %qwp = lda %cnp, 16")
                e("  %qwpn = sub %one, %qwp")
                e("  st %qwpn, %cnp, 16")
                e(f"  %qwi = mova.i32 {qpass_wdelta() // 512}")
                e("  %qwib = mul %k512, %qwi")
                e("  %qwio = mul %qwp, %qwib")
                e("  %wsb0 = add.rr %wsb, %qwio")
        if grp:
            e("  %cn1 = add.rr %cnt, %one")
            e("  %cwrap = lt %cn1, %gsz")
            e("  %cnn = mul %cn1, %cwrap")
            e("  st %cnn, %cnp, 0")
            e(f"  %gnb = mova.i32 {cfg.nb}")
            e("  %gplen = mul %gnb, %gsz")
            e("  %gd1 = add.rr %gd, %one")
            e("  %gmore = lt %gd1, %gplen")   # 1 while the group's phase continues
            e("  %gdn = mul %gd1, %gmore")
            e("  st %gdn, %cnp, 8")
            e("  %gsw = lt %gmore, %one")     # 1 at the phase switch: s <- 1 - s
            e("  %gs2 = add.rr %gs, %gs")
            e("  %gt = mul %gs2, %gsw")
            e("  %gsa = add.rr %gs, %gsw")
            e("  %gtn = mul %gt, %am1")
            e("  %gsn = add.rr %gsa, %gtn")
            e("  st %gsn, %cnp, 4")
        elif not qpass:
            e("  %cn1 = add.rr %cnt, %one")
            e("  %cwrap = lt %cn1, %npass")
            e("  %cnn = mul %cn1, %cwrap")
            e("  st %cnn, %cnp, 0")
            if cfg.dfr:   # the block advances when the pass wraps, the call's blocks wrap to 0
                e("  %bstep = lt %cwrap, %one")
                e("  %bn1 = add.rr %blk, %bstep")
                e(f"  %nbk = mova.i32 {cfg.nb}")
                e("  %bmore = lt %bn1, %nbk")
                e("  %bnn = mul %bn1, %bmore")
                e("  st %bnn, %cnp, 4")
    if gate:   # job state: private storage, zeroed by the array setup of a core stream plan
        e("  %gsp = storage_address %gst : low.storage<private> -> reg<aie2p.ep>")
    if pair and cfg.dfr:   # the deferred segment's swap, before its pushes (Q' pairs: qpair_swap in the convert)
        e("  low.cond_br %dfx, ^pxd, ^pxn : reg<aie2p.er>")
        e("^pxd:")
        pair_swap(e, pair, "%dss", "d")
        e("  low.br ^pxn")
        e("^pxn:")
    body(L, cfg, role, ks, a_adv, acap, gate, fills, helper, pair)


def gate_pace(e, p, pn1, out):
    """out = the pace after pn1 polls: GP0 + GPR * max(0, pn1 - GRAMP) + GPS * max(0, pn1 - GFAST) (p: name prefix)."""
    for t, lim, k in (("r", GRAMP, GPR), ("b", GFAST, GPS)):
        e(f"  {p}{t}l = {imm32(lim)} {lim}")
        e(f"  {p}{t}s = lt {p}{t}l, {pn1}")
        e(f"  {p}{t}o = sub {pn1}, {p}{t}l")
        e(f"  {p}{t}m = mul {p}{t}o, {p}{t}s")
        e(f"  {p}{t}k = mov.i32 {k}")
        e(f"  {p}{t}g = mul {p}{t}m, {p}{t}k")
    e(f"  {p}0 = mov.i32 {GP0}")
    e(f"  {p}r = add.rr {p}0, {p}rg")
    e(f"  {out} = add.rr {p}r, {p}bg")


def imm32(v):
    """The constant op for v: mova.i32 (4 bytes, load slot) where its 11-bit immediate holds v, else mov.i32."""
    return "mova.i32" if -1024 <= v < 1024 else "mov.i32"


def gate_signal(e, exp, sts, pol, one):
    """go [ready, status, polls, 0], then done [ready (| GATE_FAILED if the waiter gave up: status 2), status, polls, 0]
    on the core's signal stream (the relay's; GN_QNOTICK: the waiter's)."""
    e("  %gs0 = mova.i32 0")
    gate_record(e, (exp, sts, pol, "%gs0"), "%gsid0")
    if pshare():
        return
    e(f"  %gsbad = lt {one}, {sts}")
    e(f"  %gsfb = mov.i32 {GATE_FAILED - (1 << 32)}")   # the waiter gave up
    e("  %gsmark = mul %gsbad, %gsfb")
    e(f"  %gsdone = add.rr {exp}, %gsmark")
    gate_record(e, ("%gsdone", sts, pol, "%gs0"), "%gsid1")


def gate_record(e, words, idr):
    """One signal record (GN_QPSHARE: a packet, header first, TLAST on the last word)."""
    if pshare():
        e(f"  {idr} = mova.i32 2")   # (Loom's LOOM_AIE2P_SHARE_CHANNEL_ID)
        e(f"  mov.ph {idr}, 0")
    for i, v in enumerate(words):
        e(f"  mov.ms{'.last' if pshare() and i == 3 else ''} {v}")


def gate_done_late(e):
    """GN_QPSHARE: the job's done at the relay's last firing, from the hand-off kept in job state (20 ready, 24 status,
    28 polls)."""
    e("  %gdv = lda %gsp, 20")
    e("  %gds = lda %gsp, 24")
    e("  %gdp = lda %gsp, 28")
    e("  %gd0 = mova.i32 0")
    e("  %gd1 = mova.i32 1")
    e("  %gdbad = lt %gd1, %gds")
    e(f"  %gdfb = mov.i32 {GATE_FAILED - (1 << 32)}")   # the waiter gave up
    e("  %gdmark = mul %gdbad, %gdfb")
    e("  %gddone = add.rr %gdv, %gdmark")
    gate_record(e, ("%gddone", "%gds", "%gdp", "%gd0"), "%gdid")


def qprefill_on(q):
    """(GN_QPREFILL=1, Q' fuse 2) the gate cores check the gate after a call's fill, not before: a job's first fill
    then runs in the wait for the GPU. The host must start the weight reads before the gate (LoomNpu::SplitLead), or
    the gate waits on a fill that waits on it."""
    return q is not None and os.environ.get("GN_QPREFILL") == "1"


def gate_first(e, cfg, gate, gn):
    """The gate at a job's first firing, before any of its data (see GATE), so a freshly set up array
    (job state zero: at the gate) runs no call. The waiter polls for the job's ready value and hands it to the relay,
    which emits its go and done; job state then holds the job's firings (gate_count counts them down, back to zero at
    its end). Fuse 2 checks at a call's first firing (before its fill), others at every firing. gn: the neighbor
    channel's pointer (the resource: no fill ran yet)."""
    # (a Q' core fires once per M block of its row: nb / rows a call)
    S, FPC = cfg.gate, (cfg.nb // QF.get("rows", 1) if qpass_on() else cfg.nb * cfg.passes)
    nbr = gate_ports(cfg)[gate]["nbr"]
    compact = gate == "waiter" and cfg.ghelp   # (the helper's waiter: firings left at word 0, less program)
    e(f"  %gfs = lda %gsp0, {0 if compact else 4}")
    e("  low.cond_br %gfs, ^gfgo, ^gfw : reg<aie2p.er>")
    e("^gfw:")
    e("  %gf1 = mova.i32 1")
    e("  %gfm1 = mova.i32 -1")
    e(f"  %gffpc = {imm32(FPC)} {FPC}")
    if gate == "relay":
        if qlock_on():   # the record's loads wait for a first load's result (their address adds 0 times it):
            # LOOM_EXP_LOCK_READ_SEP lets reads issue 2 cycles after an acquire, and the relay's first two then read
            # the waiter's record stale (tests: markers 0x51, 0xa2, 0xf3 came out 0, 0, 0xf3)
            e("  %gfzr = mova.i32 0")
        e(f"  acq %gfm1, {nbr}")
        if qlock_on():
            e(f"  %gfk = lda {gn}, 12")
            e("  %gfk0 = mul %gfk, %gfzr")
            e("  %gfkm = mov.modifier %gfk0")
            e(f"  %gfgn = padda.modifier {gn}, %gfkm")
            gn = "%gfgn"
        e(f"  %gfexp = lda {gn}, 0")
        e(f"  %gfsts = lda {gn}, 4")
        e(f"  %gfpol = lda {gn}, 8")
        e(f"  %gfnc = lda {gn}, 12")
        e(f"  rel %gf1, {nbr}")
        if not notick():
            gate_signal(e, "%gfexp", "%gfsts", "%gfpol", "%gf1")
        if pshare():   # (done at the job's last firing: gate_count)
            e("  st %gfexp, %gsp0, 20")
            e("  st %gfsts, %gsp0, 24")
            e("  st %gfpol, %gsp0, 28")
        e("  %gfz = lt %gfnc, %gf1")   # (given up: one call)
        e("  %gfnc1 = add.rr %gfnc, %gfz")
        e("  %gfNn = mul %gfnc1, %gffpc")
        e("  st %gfNn, %gsp0, 4")
        e("  low.br ^gfgo")
        e("^gfgo:")
        return
    # the waiter: ready >= the last job's + GATE_CALLS; poll, then spin the pace (gate_pace)
    e(f"  %gf64 = mova.i32 {GATE_CALLS}")
    if compact:
        e("  %gfE = lda %gsp0, 16")
        e("  %gfexp = add.rr %gfE, %gf64")
    else:
        e("  %gfseq = lda %gsp0, 16")
        e("  %gfnext = add.rr %gfseq, %gf1")
        e("  %gfexp = mul %gfnext, %gf64")
    e("  %gf0 = mova.i32 0")
    e("  low.br ^gfp(%gf0: reg<aie2p.er>, %gf0: reg<aie2p.er>)")
    e("^gfp(%gfpn: reg<aie2p.er>, %gfpw: reg<aie2p.er>):")
    e("  low.cond_br %gfpw, ^gfspin, ^gfask : reg<aie2p.er>")
    e("^gfspin:")
    e("  %gfpw1 = add %gfpw, -1")
    e("  low.br ^gfp(%gfpn: reg<aie2p.er>, %gfpw1: reg<aie2p.er>)")
    e("^gfask:")
    if not notick():   # (ghelp's ticks carry the read bound: the tick binding is scratch)
        gate_tick(e, '%gfpn' if compact else '%gf0', "f")
    e("  %gfrdy = mov.ss")
    e("  %gfy1 = mov.ss")
    e("  %gfy2 = mov.ss")
    e("  %gfy3 = mov.ss")
    e("  %gfpn1 = add %gfpn, 1")
    e("  %gflate = lt %gfrdy, %gfexp")
    e(f"  %gfsl = {imm32(S - 1)} {S - 1}")
    e("  %gfquit = lt %gfsl, %gfpn1")   # (pn + 1 >= S)
    e("  %gfwait = lt %gfquit, %gflate")
    gate_pace(e, "%gfv", "%gfpn1", "%gfpace")
    e("  low.cond_br %gfwait, ^gfpgo, ^gfhand : reg<aie2p.er>")
    e("^gfpgo:")
    e("  low.br ^gfp(%gfpn1: reg<aie2p.er>, %gfpace: reg<aie2p.er>)")
    e("^gfhand:")   # opened: [ready, 1, polls, calls]; gave up after S polls: [expected, 2, S, 0] (the job: one call)
    e("  %gfls27 = copy %gflate : reg<aie2p.er> -> reg<aie2p.mr27_select>")
    e("  %gfhs = add %gflate, 1")
    e("  %gfhv = sel.nez %gfexp, %gfrdy, %gfls27")
    e("  %gfhc = sub %gfhv, %gfexp")
    e(f"  acq %gfm1, {nbr}")
    e(f"  st %gfhv, {gn}, 0")
    e(f"  st %gfhs, {gn}, 4")
    e(f"  st %gfpn1, {gn}, 8")
    e(f"  st %gfhc, {gn}, 12")
    e(f"  rel %gf1, {nbr}")
    if notick():
        gate_signal(e, "%gfhv", "%gfhs", "%gfpn1", "%gf1")
    e("  %gfz = lt %gfhc, %gf1")
    e("  %gfhc1 = add.rr %gfhc, %gfz")
    e("  %gfNn = mul %gfhc1, %gffpc")
    e("  %gfLn1 = sub %gfsl, %gfpn1")
    e("  %gfLn = add %gfLn1, 1")   # (S - (pn + 1); pending reads: 0, the gate's state)
    e(f"  st %gfNn, %gsp0, {0 if compact else 4}")
    e("  st %gfLn, %gsp0, 8")
    e(f"  st {'%gfexp' if compact else '%gfnext'}, %gsp0, 16")
    e("  low.br ^gfgo")
    e("^gfgo:")


QCSTAGE_AT = (25088, 27648, 30208, 49152)   # (GN_QCSTAGE) the relay row's staging slots (free core memory)


def qcstage_on():
    """GN_QCSTAGE=1 (pair relay, NPU_QCREL=2): the relay row's C ring has 2 MP slots, its own MP (the accumulator
    rows) then MP staging slots (QCSTAGE_AT; GN_QPINS needs the 2 MP-slot entry) that take the partner's copied
    segments, so its next block waits for its own segments' drain only, not for the copies' too."""
    return os.environ.get("GN_QCSTAGE") == "1" and crelp_on()


def crel_copy_staged(e, port, own):
    """(GN_QCSTAGE) crel_copy into the staging slots: segment i of the partner's ring to QCSTAGE_AT[i] (the own ring
    pointer, slot 0, parked at own: its address + QCSTAGE_AT[i] - slot 0's)."""
    pins = qpins(QA4_RING_AT)
    sb = int(pins.split(",")[0].split("@")[0])
    cs = [int(v) for v in pins.split(",")[0].split("@")[1].split(":")]
    st = QCSTAGE_AT
    assert len(st) == MP == 4 and all(b - a == sb for a, b in zip(st[:3], st[1:3])) and st[3] >= st[2] + sb
    sd = cs[1] - cs[0]
    e(f"  %crG = storage_address {own[0]} : low.storage<private> -> reg<aie2p.ep>")
    e(f"  %crol = lda %crG, {own[1]}")
    e("  %crS = storage_address %crst : low.storage<private> -> reg<aie2p.ep>")
    e("  %crrl = lda %crS, 0")
    e("  %crr = mov.scalar-to-address %crrl")
    e("  %crm1 = mova.i32 -1")
    e("  %cr1 = mova.i32 1")
    e("  %crz = mova.i32 0")
    e(f"  %crK = mova.i32 {MP}")
    e("  low.br ^crl(%crz: reg<aie2p.er>, %crr: reg<aie2p.ep>)")
    e("^crl(%cri: reg<aie2p.er>, %crs: reg<aie2p.ep>):")
    e("  %crm = lt %cri, %crK")
    e("  %crin = add.rr %cri, %cr1")
    e("  low.cond_br %crm, ^crbd, ^crdn : reg<aie2p.er>")
    e("^crbd:")
    e(f"  acq %crm1, {port}")
    e("  acq %crm1, 2")
    # staging slot i: own + (st0 - cs0) + sb i + (i > 2) (st3 - st2 - sb)
    e(f"  %crok = mov.i32 {sb}")
    e("  %croi = mul %cri, %crok")
    e("  %crk2 = mova.i32 2")
    e("  %crg3 = lt %crk2, %cri")
    e(f"  %crjk = mov.i32 {st[3] - st[2] - sb}")
    e("  %crj = mul %crg3, %crjk")
    e(f"  %crb0 = mov.i32 {st[0] - cs[0]}")
    e("  %crs1 = add.rr %crol, %crb0")
    e("  %crs2 = add.rr %crs1, %croi")
    e("  %crs3 = add.rr %crs2, %crj")
    e("  %crq_0 = mov.scalar-to-address %crs3")
    e("  %crk = lda %crs, 0")   # (the vector loads' address waits for a first load: crel_copy)
    e("  %crk0 = mul %crk, %crz")
    e("  %crkm = mov.modifier %crk0")
    e("  %crp_0 = padda.modifier %crs, %crkm")
    nl = sb // 64
    for i in range(nl):
        e(f"  %crv_{i}, %crp_{i + 1} = vlda.acc.post %crp_{i}, 64")
        e(f"  %crq_{i + 1} = vst.acc.post %crv_{i}, %crq_{i}, 64")
    e("  rel %cr1, 2")
    e(f"  rel %cr1, {port}")
    if sd > sb:   # (the source pointer past the slot's tail)
        e(f"  %crgk = mov.i32 {sd - sb}")
        e("  %crgm = mov.modifier %crgk")
        e(f"  %crp_{nl}g = padda.modifier %crp_{nl}, %crgm")
        nl = f"{nl}g"
    e(f"  low.br ^crl(%crin: reg<aie2p.er>, %crp_{nl}: reg<aie2p.ep>)")
    e("^crdn:")


def crel_copy(e, port, nseg=None, own=("%fzt", 0)):
    """(Q' C relay) the firing's end on row 0: row 1's MP C segments (its ring, through the north window) copied into
    this core's C ring after its own, so its C stream carries block 2 f, then block 2 f + 1. The pointers come back
    from fuse2_entry's parking word. A segment is the slot's first 2 KB (the peeled pass's bf16)."""
    pins = qpins(QA4_RING_AT)
    sb = int(pins.split(",")[0].split("@")[0])   # (a segment: one C ring slot; 2048 B, the P1 layout's 3072 B)
    cs = [int(v) for v in pins.split(",")[0].split("@")[1].split(":")]
    assert sb % 64 == 0 and len(cs) == MP and qa4_on() and (qpeel_on() or qunr_on() or os.environ.get("GN_QBANK") == "3") and not qalt_on()
    e(f"  %crG = storage_address {own[0]} : low.storage<private> -> reg<aie2p.ep>")
    e(f"  %crol = lda %crG, {own[1]}")
    e("  %cro = mov.scalar-to-address %crol")
    e("  %crS = storage_address %crst : low.storage<private> -> reg<aie2p.ep>")
    e("  %crrl = lda %crS, 0")
    e("  %crr = mov.scalar-to-address %crrl")
    e("  %crm1 = mova.i32 -1")
    e("  %cr1 = mova.i32 1")
    e("  %crz = mova.i32 0")
    e(f"  %crK = mova.i32 {nseg or MP}")
    sd = cs[1] - cs[0] if len(cs) > 1 else sb   # (the slot stride: odd NP's C slots hold an f32 tail past the bf16)
    assert sd >= sb and sd % 64 == 0 and all(b - a == sd for a, b in zip(cs, cs[1:])), cs
    # (a loop over the segments, its counter stepped in the header: no hardware loop candidate, so
    # LOOM_EXP_HWLOOP_TRIP's '@' counts stay the GEMM's)
    e("  low.br ^crl(%crz: reg<aie2p.er>, %crr: reg<aie2p.ep>, %cro: reg<aie2p.ep>)")
    e("^crl(%cri: reg<aie2p.er>, %crs: reg<aie2p.ep>, %crq_0: reg<aie2p.ep>):")
    e("  %crm = lt %cri, %crK")
    e("  %crin = add.rr %cri, %cr1")
    e("  low.cond_br %crm, ^crbd, ^crdn : reg<aie2p.er>")
    e("^crbd:")
    e(f"  acq %crm1, {port}")
    e("  acq %crm1, 2")
    # the vector loads' address waits for a first load (LOOM_EXP_LOCK_READ_SEP: reads 2 cycles past an acquire read
    # row 1's stores stale; see the gate relay's record)
    e("  %crk = lda %crs, 0")
    e("  %crk0 = mul %crk, %crz")
    e("  %crkm = mov.modifier %crk0")
    e("  %crp_0 = padda.modifier %crs, %crkm")
    nl = sb // 64
    nocopy = os.environ.get("GN_QCRELNOCOPY") == "1"   # (diag: the locks only, pointers stepped)
    if nocopy:
        e("  %crnk = mova.i32 64")
        e("  %crnm = mov.modifier %crnk")
    for i in range(nl):
        if nocopy:
            e(f"  %crp_{i + 1} = padds.modifier %crp_{i}, %crnm")
            e(f"  %crq_{i + 1} = padds.modifier %crq_{i}, %crnm")
            continue
        e(f"  %crv_{i}, %crp_{i + 1} = vlda.acc.post %crp_{i}, 64")
        e(f"  %crq_{i + 1} = vst.acc.post %crv_{i}, %crq_{i}, 64")
    e("  rel %cr1, 2")
    e(f"  rel %cr1, {port}")
    if sd > sb:   # (both pointers past the slot's tail)
        e(f"  %crgk = mov.i32 {sd - sb}")
        e("  %crgm = mov.modifier %crgk")
        e(f"  %crp_{nl}g = padda.modifier %crp_{nl}, %crgm")
        e(f"  %crq_{nl}g = padda.modifier %crq_{nl}, %crgm")
        nl = f"{nl}g"
    if (nseg or MP) > MP:   # (a chain: both rings wrap every MP segments)
        assert MP & (MP - 1) == 0
        e(f"  %crwm = mova.i32 {MP - 1}")
        e("  %crw = and %crin, %crwm")
        e("  %crwz = eqz %crw")
        e(f"  %crwk = mov.i32 {-MP * sd}")
        e("  %crwo = mul %crwz, %crwk")
        e("  %crwmd = mov.modifier %crwo")
        e(f"  %crpw = padda.modifier %crp_{nl}, %crwmd")
        e(f"  %crqw = padda.modifier %crq_{nl}, %crwmd")
        e("  low.br ^crl(%crin: reg<aie2p.er>, %crpw: reg<aie2p.ep>, %crqw: reg<aie2p.ep>)")
    else:
        e(f"  low.br ^crl(%crin: reg<aie2p.er>, %crp_{nl}: reg<aie2p.ep>, %crq_{nl}: reg<aie2p.ep>)")
    e("^crdn:")


def gate_count(e, cfg, gate):
    """The gated leaves' end of firing: the job's firings counted down (zero at its end: the next firing waits at the
    gate, gate_first) and the waiter's leftover supply drained in the background
    (all of it by the job's end)."""
    e("  %g0 = mova.i32 0")
    e("  %g1 = mova.i32 1")
    if gate == "relay":   # [firings done, firings of the job]
        e("  %gF = lda %gsp, 0")
        e("  %gN = lda %gsp, 4")
        e("  %gF1 = add.rr %gF, %g1")
        e("  %gcont = lt %gF1, %gN")
        e("  %gFn = mul %gF1, %gcont")
        e("  %gNn = mul %gN, %gcont")
        e("  st %gFn, %gsp, 0")
        e("  st %gNn, %gsp, 4")
        if pshare():   # the job's last firing: its done
            e("  low.cond_br %gcont, ^grx, ^grd : reg<aie2p.er>")
            e("^grd:")
            gate_done_late(e)
            e("  low.br ^grx")
            e("^grx:")
        e("  return")
        return
    if cfg.ghelp:   # compact: [firings left, -, leftover, pending reads, the last job's ready value]
        R27 = "reg<aie2p.er> -> reg<aie2p.mr27_select>"
        e("  %gR = lda %gsp, 0")
        e("  %gL = lda %gsp, 8")
        e("  %gQ = lda %gsp, 12")
        e("  %gR1 = add %gR, -1")                    # this firing ends the job: R1 == 0
        e(f"  %ggb = mova.i32 {GB}")
        e("  %gls = lt.select %gL, %ggb")             # m = min(L, GB), or L at the end
        e("  %gmin = sel.nez %gL, %ggb, %gls")
        e("  %gcs = copy %gR1 : " + R27)
        e("  %gm = sel.nez %gmin, %gL, %gcs")
        e("  %gQL = add.rr %gQ, %gL")
        e("  %gn = sel.nez %gQ, %gQL, %gcs")         # n = Q, or Q + L at the end
        e("  low.br ^gd(%g0: reg<aie2p.er>)")
        e("^gd(%gdi: reg<aie2p.er>):")
        e("  %gdtm = lt %gdi, %gm")
        e("  %gdrm = lt %gdi, %gn")
        e("  %gdm = or %gdtm, %gdrm")
        e("  %gdi1 = add %gdi, 1")
        e("  low.cond_br %gdm, ^gdr, ^gdx : reg<aie2p.er>")
        e("^gdr:")
        e("  low.cond_br %gdrm, ^gdr1, ^gdt : reg<aie2p.er>")
        e("^gdr1:")
        for i in range(4):
            e(f"  %gdx{i} = mov.ss")
        e("  low.br ^gdt")
        e("^gdt:")
        e("  low.cond_br %gdtm, ^gdt1, ^gdn : reg<aie2p.er>")
        e("^gdt1:")
        for i in range(4):
            e("  mov.ms %gn")
        e("  low.br ^gdn")
        e("^gdn:")
        e("  low.br ^gd(%gdi1: reg<aie2p.er>)")
        e("^gdx:")
        e("  st %gR1, %gsp, 0")
        e("  %gL2 = sub %gL, %gm")
        e("  st %gL2, %gsp, 8")
        e("  %gQn = sel.nez %gm, %g0, %gcs")         # (all read at the end)
        e("  st %gQn, %gsp, 12")
        e("  return")
        return
    # the waiter's: [firings done, firings of the job, leftover, pending reads, sequence]
    # (Q': each drain loop's step in its header, so no drain loop is a hardware loop candidate: LOOM_EXP_HWLOOP_TRIP's
    # '@' list, HRX patch 0083, goes to a leaf's last candidates, which must stay the GEMM's)
    hstep = qpass_on()

    def step(nxt, cur):
        e(f"  {nxt} = add.rr {cur}, %g1")
    e("  %gF = lda %gsp, 0")
    e("  %gN = lda %gsp, 4")
    e("  %gL = lda %gsp, 8")
    e("  %gQ = lda %gsp, 12")
    e("  low.br ^gq(%g0: reg<aie2p.er>)")   # read back last firing's ticks, then tick up to GB more of the leftover
    e("^gq(%gqi: reg<aie2p.er>):")
    e("  %gqm = lt %gqi, %gQ")
    if hstep:
        step("%gqn", "%gqi")
    e("  low.cond_br %gqm, ^gq1, ^gqd : reg<aie2p.er>")
    e("^gq1:")
    for i in range(4):
        e(f"  %gqx{i} = mov.ss")
    if not hstep:
        step("%gqn", "%gqi")
    e("  low.br ^gq(%gqn: reg<aie2p.er>)")
    e("^gqd:")
    e(f"  %ggb = mov.i32 {GB}")
    e("  %gls = lt %gL, %ggb")
    e("  %glsn = sub %g1, %gls")
    e("  %gb1 = mul %gL, %gls")
    e("  %gb2 = mul %ggb, %glsn")
    e("  %gbt = add.rr %gb1, %gb2")
    e("  low.br ^gt(%g0: reg<aie2p.er>)")
    e("^gt(%gti: reg<aie2p.er>):")
    e("  %gtm = lt %gti, %gbt")
    if hstep:
        step("%gtk", "%gti")
    e("  low.cond_br %gtm, ^gt1, ^gtd : reg<aie2p.er>")
    e("^gt1:")
    if notick():
        for i in range(4):
            e(f"  %gtx{i} = mov.ss")
    else:
        gate_tick(e, "%g0", "t")
    if not hstep:
        step("%gtk", "%gti")
    e("  low.br ^gt(%gtk: reg<aie2p.er>)")
    e("^gtd:")
    e("  %gL2 = sub %gL, %gbt")
    e("  %gF1 = add.rr %gF, %g1")
    e("  %gcont = lt %gF1, %gN")
    e("  low.cond_br %gcont, ^gmid, ^gend : reg<aie2p.er>")
    e("^gmid:")
    e("  st %gF1, %gsp, 0")
    e("  st %gL2, %gsp, 8")
    e(f"  st {'%g0' if notick() else '%gbt'}, %gsp, 12")
    e("  return")
    e("^gend:")   # finish the leftover: the pending reads, then one tick + read per remaining record
    e("  low.br ^gr(%g0: reg<aie2p.er>)")
    e("^gr(%gri: reg<aie2p.er>):")
    e(f"  %grm = lt %gri, {'%g0' if notick() else '%gbt'}")
    if hstep:
        step("%grn", "%gri")
    e("  low.cond_br %grm, ^gr1, ^grd : reg<aie2p.er>")
    e("^gr1:")
    for i in range(4):
        e(f"  %grx{i} = mov.ss")
    if not hstep:
        step("%grn", "%gri")
    e("  low.br ^gr(%grn: reg<aie2p.er>)")
    e("^grd:")
    e("  low.br ^gl(%g0: reg<aie2p.er>)")
    e("^gl(%gli: reg<aie2p.er>):")
    e("  %glm = lt %gli, %gL2")
    if hstep:
        step("%gln", "%gli")
    e("  low.cond_br %glm, ^gl1, ^gld : reg<aie2p.er>")
    e("^gl1:")
    if not notick():
        gate_tick(e, "%g0", "l")
    for i in range(4):
        e(f"  %glx{i} = mov.ss")
    if not hstep:
        step("%gln", "%gli")
    e("  low.br ^gl(%gln: reg<aie2p.er>)")
    e("^gld:")
    for o in (0, 4, 8, 12):
        e(f"  st %g0, %gsp, {o}")
    e("  return")


def body(L, cfg, role, ks, a_adv, acap, gate=None, fills=False, helper=None, pair=None):
    """The locked stream: per sub-tile MMA + pop bundles, then the cascade boundary.
    head: chains start with mmul; after each sub-tile, 16 cascade writes (the next sub-tile's pops ride with them)
    mid:  mmul at k 0, then k 1-4 add the incoming partial quarter by quarter (fused cascade MMA); writes like the head
    tail: chains start from C, k 0-3 add the incoming quarters; after each sub-tile C is stored and the next loaded"""
    e = L.append
    tail = role == "tail"
    mid = role == "mid"
    qbank = tail and os.environ.get("GN_QBANK") in ("1", "2", "3") and os.environ.get("GN_QCORE")   # (leaf)
    q3 = qbank and os.environ.get("GN_QBANK") == "3"
    NT = NP * cfg.mu      # sub-tiles per iteration, M-slab major
    # GN_QROW=1 (layout Q', with GN_QCORE): the row end is a rotated window too, into the next row's sub-tile 0, whose
    # state after its first E2 k-blocks (accumulators, FIFOs, pops ahead) crosses the back edge; the join's scalar work
    # rides in sub-tile 1; a last pass converts every row after the loop
    qrow = tail and bool(os.environ.get("GN_QCORE")) and os.environ.get("GN_QROW") == "1"
    if qrow:
        assert cfg.mu == 1 and acap in (2, 3, 4) and not a_adv and not cfg.swiglu and not cfg.dfr and not cfg.ofeat
        QEr, QE2r = (json.load(open(os.environ["GN_QCORE"]))[k] for k in ("E", "E2"))
    qpass = qrow and qpass_on()
    # swiglu tail: the A pointer is loop-carried as a scalar, a short-lived pointer per sub-tile. Its store fifo pins
    # p2; a pointer live across the loop would otherwise be pushed into p0 / p1, which the operand load fifos need.
    spa = tail and cfg.swiglu
    pat = "reg<aie2p.er>" if spa else "reg<aie2p.ep>"
    fa_, fw_ = "%a", "%w"
    if tail and qpass and qa4_carry(cfg):
        fa_ = "%qa0"
    if tail and qpass and cfg.passes % 2:
        e("  %qw0a = mov.scalar-to-address %wsb0")
        fw_ = "%qw0a"
    e(f"  %lo_a = vlda.load-fifo.low512 {fa_}, 0")
    e(f"  %lo_w = vlda.load-fifo.low512 {fw_}, 0")
    e(f"  %fa0 = vlda.load-fifo.high512 {fa_}, %lo_a, 64")
    e(f"  %fw0 = vlda.load-fifo.high512 {fw_}, %lo_w, 64")
    if tail:
        e(f"  %oc = copy {'%qev' if qalt_on() else '%oh' if cfg.dfr else '%o'} : reg<aie2p.ep> -> reg<aie2p.ep>")
        bump(L, "%ob", "%oc", 512)
        po0, pct = "%ob", "reg<aie2p.ep>"
        # C load pointer: the zero buffer with step 0 on an M block's first pass, the C slot otherwise
        e("  %nf = mova.i32 1" if qpass else "  %nf = lt %zero, %cnt")
        e("  %lstep = mul %cstep, %nf")
        e("  %mls = mov.modifier %lstep")
        if cfg.swiglu:   # a segment's C slot is SW_SLOT bytes: the pointers skip its tail past the f32 segment
            e(f"  %swgap = mov.i32 {SW_SLOT - 1024 * NP}")
            e("  %lgap = mul %swgap, %nf")
            e("  %mlb = mov.modifier %lgap")
            e("  %mcb = mov.modifier %swgap")
        pla, plp = ", %plb: reg<aie2p.ep>", ", %pl: reg<aie2p.ep>"
    else:
        po0, pct = "%zero", "reg<aie2p.er>"
        pla = plp = ""
    a0 = "%a"
    if spa:
        e("  %as0 = mov.address-to-scalar %a")
        a0 = "%as0"
    dft = DFT.split(", ")
    dfa = lambda v: "".join(f", {x}: {ty}" for x, ty in zip(v, dft)) if tail and cfg.dfr else ""
    if tail and cfg.dfr:   # the deferred segment's store fifo (when there is none: dfi's dummy buffer, else the zero
        # buffer, never pushed)
        dz = "%dmsc" if cfg.dfi else "%zsc0"
        e(f"  %dsd = sub %dss, {dz}")
        e("  %dsm = mul %dfx, %dsd")
        e(f"  %dsa = add.rr {dz}, %dsm")
        e("  %dsp = mov.scalar-to-address %dsa")
        e("  %dsl = vlda.store-fifo.low512 %dsp, 0")
        e("  %dqf0 = vlda.store-fifo.high512 %dsp, %dsl, 64")
        e("  %dqo0 = copy %dsp : reg<aie2p.ep> -> reg<aie2p.mpfs>")
        e("  %dqp0 = mova.fifo.store.position 0")
    hdr = len(L)   # (seam: the entry and the header are rewritten once the first sub-tile's carried values are known)
    e(f"  low.br ^outer(%zero: reg<aie2p.er>, {a0}: {pat}, {po0}: {pct}, %fa0: reg<aie2p.eldfiforeg>, %fw0: reg<aie2p.eldfiforeg>{pla}{dfa(('%dqf0', '%dqo0', '%dqp0'))})")
    e(f"^outer(%mp: reg<aie2p.er>, %pa: {pat}, %po: {pct}, %fa: reg<aie2p.eldfiforeg>, %fw: reg<aie2p.eldfiforeg>{plp}{dfa(("%dqf", "%dqo", "%dqp"))}):")
    dfifo = ("%dqf", "%dqo", "%dqp")
    e("  %omore = lt %mp, %nmp")
    e("  low.cond_br %omore, ^obody, ^exit : reg<aie2p.er>")
    e("^obody:")
    pc = ["%po"] + [f"%pc{t}" for t in range(1, NT)]
    pl = (["%pl"] + [f"%pl{t}" for t in range(1, NT)]) if tail else pc
    if qrow:
        pc, pl = pc + [f"%pc{NT}"], pl + [f"%pl{NT}"]
    fifo = {"a": "%fa", "w": "%fw"}

    wend = {}   # (GN_QWCONT) each sub-tile's final W FIFO state
    name = MBNAME[0]
    mb2 = os.environ.get("GN_QMB1") in ("2", "3", "4", "5") and name in ("tailm", "tailm0") and qpass

    def units_for(t):
        x = f"s{t}"
        tw, tm = t % NP, t // NP
        # (GN_QWCONT, Q' passes) a row's W blocks sit back to back and a sub-tile pops exactly its 63 lines: the W
        # FIFO runs on into the next sub-tile (no pointer copy, no position reset; only the first fill)
        wcont = qpass and os.environ.get("GN_QWCONT") == "1" and 0 < t < NT
        p1k_mid = None   # (P1K) the second K half's W base: (scalar base, byte offset)
        if tw == 0 and not (qpass and t == NT):
            wp, wsrc = [], ("%qw0a" if qpass and cfg.passes % 2 else "%w")
        else:   # (qpass: from the carried W slot, the next row's at t = NT)
            wp = ([f"  %{x}w0 = mov.scalar-to-address {'%wcn' if t == NT else '%wc'}"] if qpass else
                  [f"  %{x}w0 = copy %w : reg<aie2p.ep> -> reg<aie2p.ep>"])
            for j in range(tw):
                wp.append(f"  %{x}w{j + 1} = padds.modifier %{x}w{j}, %mf")
            wsrc = f"%{x}w{tw}"
        if mb2 and not wcont:   # (GN_QMB1=2 diag, 2-row synthetic: W column halves split between the two rows' memories,
            # row 0 owns groups 0, 1, row 1 groups 2, 3; row 1 visits its groups rotated by NP / 2, so at any sub-tile the two
            # cores read different memories; the other half through the neighbour window. Row 1's output column halves swap.)
            mb3 = os.environ.get("GN_QMB1") == "3"   # (=3: every W pop from the other row's memory, no rotation)
            rot = name == "tailm" and os.environ.get("GN_QMB1") in ("2", "5") and not qp1k_on()   # (=4: no rotation)
            g = (tw + (NP // 2 if rot else 0)) % NP
            own = (g < NP // 2) == (name == "tailm0") and not mb3
            d = 0 if own else (0x60000 - 0x70000 if name == "tailm0" else 0x40000 - 0x70000)
            bs = "%wcn" if t == NT else "%wc" if t else f"%{x}mbw"
            wp = ([f"  %{x}mbw = mov.address-to-scalar %w"] if t == 0 else []) + [
                  f"  %{x}mbk = mov.i32 {(g % (NP // 2) if qp1_on() else g) * slab(ks) + d}", f"  %{x}mbs = add.rr {bs}, %{x}mbk",
                  f"  %{x}w0 = mov.scalar-to-address %{x}mbs"]
            wsrc = f"%{x}w0"
            if qp1k_on():   # (P1K) group g's K half 0 sits in row 0's memory, half 1 in row 1's, both at g half;
                # each core reads its own memory first (GN_QP1K_REV=1: row 1 runs K half 1 first, A too), so the two never
                # read one memory at once; =0: both in k order (row 1 starts on row 0's memory)
                dn = 0x60000 - 0x70000 if name == "tailm0" else 0x40000 - 0x70000
                hs = slab(ks) // 2
                rev = name == "tailm" and os.environ.get("GN_QP1K_REV", "1") == "1"
                o0, o1 = (g * hs, g * hs + dn) if name == "tailm0" or rev else (g * hs + dn, g * hs)
                wp[-3] = f"  %{x}mbk = mov.i32 {o0}"
                p1k_mid = ("%wc" if t == 0 else bs, o1)   # (t 0: the carried slot; its own base is per entry)
            elif t == 0 and (g % (NP // 2) if qp1_on() else g) == 0 and d == 0:
                wp, wsrc = [], "%w"
        asrc, ap = "%pa", []
        # spa: a distinct opaque zero per sub-tile (outer counter >> 8 + t) keeps the conversions from being merged
        # into one long-lived pointer
        zs = [f"  %{x}kz = mova.i32 {-(8 + t)}", f"  %{x}oz = lshl %mp, %{x}kz", f"  %{x}pas = add.rr %pa, %{x}oz"] if spa else []
        if spa:
            ap, asrc = zs + [f"  %{x}pae = mov.scalar-to-address %{x}pas"], f"%{x}pae"
        sm = tm if a_adv else tm % acap
        if sm:
            ap = zs + [f"  %{x}am0 = mov.scalar-to-address %{x}pas"] if spa else [f"  %{x}am0 = copy %pa : reg<aie2p.ep> -> reg<aie2p.ep>"]
            for j in range(sm):
                ap.append(f"  %{x}am{j + 1} = padds.modifier %{x}am{j}, %mf")
            asrc = f"%{x}am{sm}"
        if tw == 0:   # the previous row's slab is fully popped; wait for this row's slab
            ap = (["  rel %one, 0"] if t > 0 else []) + ["  acq %am1, 0"] + ap
        if qrow and t == NT:   # (qrow) the next row's sub-tile 0: slab pointer from a_next, locks placed below
            asrc, ap = "%pan", []
        p1k_a = None   # (P1K, row 1 reversed) the slab start as a scalar: A starts at its K half 1, restarts there
        if p1k_mid and name == "tailm" and os.environ.get("GN_QP1K_REV", "1") == "1":
            p1k_a = f"%{x}zra"
            ap = ap + [f"  %{x}zra = mov.address-to-scalar {asrc}", f"  %{x}zrk = mov.i32 {slab(ks) // 2}",
                       f"  %{x}zrs = add.rr %{x}zra, %{x}zrk", f"  %{x}zre = mov.scalar-to-address %{x}zrs"]
            asrc = f"%{x}zre"
        positions = [f"  %{x}qa = mova.fifo.position 0", f"  %{x}qw = mova.fifo.position 0"]
        # the tail is one pointer short for early W derivation (it has slack)
        early = (wp + positions) if not tail and t > 0 else []
        if wcont:
            assert not early and t - 1 in wend
            pw0, fw0, qw0 = wend[t - 1]
            init = ap + [f"  %{x}pa = copy {asrc} : reg<aie2p.ep> -> reg<aie2p.eps>", positions[0],
                         f"  %{x}pa1, %{x}fa1, %{x}qa1 = vlda.fill.512 %{x}pa, {fifo['a']}, %{x}qa",
                         f"  %{x}pw1, %{x}fw1, %{x}qw1 = vldb.fill.512 {pw0}, {fw0}, {qw0}"]
        else:
            init = ap + [f"  %{x}pa = copy {asrc} : reg<aie2p.ep> -> reg<aie2p.eps>"] + ([] if early else wp) + [
                    f"  %{x}pw = copy {wsrc} : reg<aie2p.ep> -> reg<aie2p.eps>"] + ([] if early else positions) + [
                    f"  %{x}pa1, %{x}fa1, %{x}qa1 = vlda.fill.512 %{x}pa, {fifo['a']}, %{x}qa",
                    f"  %{x}pw1, %{x}fw1, %{x}qw1 = vldb.fill.512 %{x}pw, {fifo['w']}, %{x}qw"]
        st = {"pa": f"%{x}pa1", "fa": f"%{x}fa1", "qa": f"%{x}qa1", "pw": f"%{x}pw1", "fw": f"%{x}fw1", "qw": f"%{x}qw1"}
        units, carry, n = [], [], 0
        for k in range(ks):
            for i, (ln, h) in enumerate([("a", 0), ("w", 0), ("a", 1), ("w", 1)]):
                kk, nm = ("a", "a") if ln == "a" else ("b", "w")
                line = (f"  %{x}{nm}{k}_{h}, %{x}p{ln}{k}_{h}, %{x}f{ln}{k}_{h}, %{x}q{ln}{k}_{h} = "
                        f"vld{kk}.pop.bfp16ebs8 {st['p' + ln]}, {st['f' + ln]}, {st['q' + ln]}")
                st["p" + ln], st["f" + ln], st["q" + ln] = f"%{x}p{ln}{k}_{h}", f"%{x}f{ln}{k}_{h}", f"%{x}q{ln}{k}_{h}"
                units.append(carry + [line])
                carry = []
                if i % 2 == 0:
                    continue
                n += 1
                if p1k_mid and n == ks:   # (P1K) the W FIFO restarts on the other memory's K half: base early,
                    # pointer copy (eps budget 2: after the half's last pop), position and fill ahead of the next pop
                    assert ks % 8 == 0, "P1K: the restart on a refill boundary"
                    early_ = [f"  %{x}mk = mov.i32 {p1k_mid[1]}", f"  %{x}ms = add.rr {p1k_mid[0]}, %{x}mk",
                              f"  %{x}mw = mov.scalar-to-address %{x}ms"]
                    if p1k_a:   # (row 1 reversed) A restarts at the slab start
                        early_.append(f"  %{x}zrm = mov.scalar-to-address {p1k_a}")
                    for j, ln_ in enumerate(early_):
                        units[-12 + 2 * j].append(ln_)
                    carry = []
                    if p1k_a:
                        carry = [f"  %{x}paM = copy %{x}zrm : reg<aie2p.ep> -> reg<aie2p.eps>", f"  %{x}qaM = mova.fifo.position 0",
                                 f"  %{x}paM1, %{x}faM1, %{x}qaM1 = vlda.fill.512 %{x}paM, {st['fa']}, %{x}qaM"]
                        st["pa"], st["fa"], st["qa"] = f"%{x}paM1", f"%{x}faM1", f"%{x}qaM1"
                    else:
                        fa_ = f"  %{x}rpaM, %{x}rfaM, %{x}rqaM = vlda.fill.512 {st['pa']}, {st['fa']}, {st['qa']}"
                        st["pa"], st["fa"], st["qa"] = f"%{x}rpaM", f"%{x}rfaM", f"%{x}rqaM"
                        units[-1].append(fa_)
                    carry += [f"  %{x}pwM = copy %{x}mw : reg<aie2p.ep> -> reg<aie2p.eps>", f"  %{x}qwM = mova.fifo.position 0",
                              f"  %{x}pwM1, %{x}fwM1, %{x}qwM1 = vldb.fill.512 %{x}pwM, {st['fw']}, %{x}qwM"]
                    st["pw"], st["fw"], st["qw"] = f"%{x}pwM1", f"%{x}fwM1", f"%{x}qwM1"
                    continue
                if n % 8 == 0 and n < 2 * ks:   # FIFO refill: pops fetch 512 bits and consume 576
                    fills = []
                    for ln2, kk2 in (("a", "a"), ("w", "b")):
                        fills.append(f"  %{x}rp{ln2}{n}, %{x}rf{ln2}{n}, %{x}rq{ln2}{n} = vld{kk2}.fill.512 {st['p' + ln2]}, {st['f' + ln2]}, {st['q' + ln2]}")
                        st["p" + ln2], st["f" + ln2], st["q" + ln2] = f"%{x}rp{ln2}{n}", f"%{x}rf{ln2}{n}", f"%{x}rq{ln2}{n}"
                    units[-1].append(fills[0])
                    carry = [fills[1]]
        if carry:
            units.append(carry)
        fifo["a"], fifo["w"] = st["fa"], st["fw"]
        wend[t] = (st["pw"], st["fw"], st["qw"])
        return early, init, units

    # each unit (a pop, maybe a fill) rides with one MMA; at a sub-tile boundary of the head and mids the next
    # sub-tile's FIFO init fills the MMA -> cascade-write latency window and its first WB pops pair with the writes
    slots, prologue, wblock, held = {}, [], {}, {}
    wb = 0 if tail else WB
    seam = bool(cfg.seam and wb) and not (cfg.seam == 2 and (helper or "")[:1] == "M")
    for t in range(NT + (1 if qrow else 0)):
        base = t * ks * 4 - LAGU
        early, init, units = units_for(t)
        if wb and t > 0:
            s0 = t * ks * 4
            for i, line in enumerate(early):
                slots.setdefault(base - len(early) + i, []).append(line)
            wblock[t - 1] = [init] + units[:wb]
            for u in range(wb, len(units)):
                slots.setdefault(s0 + u - wb, []).extend(units[u])
            continue
        if seam:   # t 0: init + the first WB units are carried in (entry: before the loop; back edge: the last writes)
            assert wb == LAGU and not early
            seam0 = init + [ln for u in units[:wb] for ln in u]
            for u in range(wb, len(units)):
                slots.setdefault(u - wb, []).extend(units[u])
            continue
        if tail and os.environ.get("GN_QCORE") and t > 0:   # Q' core: the init's address work one line per MMA ahead;
            # the FIFO pointer copies (eps, budget 2: free only after t - 1's last pop) and the fills stay with the pop
            keep = [ln for ln in init if "eps>" in ln or "fill.512" in ln]
            ahead = [ln for ln in init if ln not in keep]
            for i, line in enumerate(ahead):
                slots.setdefault(base - len(ahead) + i, []).append(line)
            if qrow and t == NT:   # the row's slab released after its last pop, the next acquired if that row runs
                keep = ["  rel %one, 0", "  %cqn = copy %nxm : reg<aie2p.er> -> reg<aie2p.mr26_lock>",
                        "  acq.cond %am1, %cqn, 0"] + keep
                if qpass and os.environ.get("GN_QLOCK") == "1":   # the lock work spread over the MMAs around the
                    # switch (a lock line orders the stream: lumped they leave ~20 bundles without an MMA): the next A slab
                    # and W slot acquired 9 / 5 MMAs ahead, the releases after the new FIFOs' first pops (one r26 value
                    # at a time: each conditional line 3 MMAs after its copy)
                    ra, rw = (int(x) for x in os.environ.get("GN_QLOCKR", "3,24").split(","))   # release offsets
                    assert base + max(ra, rw) < (NT * ks + QE2r) * 4, "a lock line past the last window is dropped"
                    qa_, qw_ = (int(x) for x in os.environ.get("GN_QLOCKA", "9,5").split(","))   # acquire offsets
                    assert qa_ - 3 > qw_, "the A acquire ahead of the W slot's r26 copy"
                    for g, ln in ((base - qa_ - 3, "  %cqn = copy %nxm : reg<aie2p.er> -> reg<aie2p.mr26_lock>"),
                                  (base - qa_, "  acq.cond %am1, %cqn, 0"),
                                  (base - qw_ - 3, "  %waq = copy %wwa : reg<aie2p.er> -> reg<aie2p.mr26_lock>"),
                                  (base - qw_, "  acq.cond %am1, %waq, 1"),
                                  (base + ra, "  rel %one, 0"),
                                  (base + rw - 3, "  %wrq = copy %wrap : reg<aie2p.er> -> reg<aie2p.mr26_lock>"),
                                  (base + rw, "  rel.cond %one, %wrq, 1")):
                        slots.setdefault(g, []).append(ln)
                    keep = keep[3:]
                elif qpass:   # the W slot at a pass end (each lock line with its r26 copy: one r26 value at a time)
                    keep =["  rel %one, 0", "  %cqn = copy %nxm : reg<aie2p.er> -> reg<aie2p.mr26_lock>\n  acq.cond %am1, %cqn, 0",
                            "  %wrq = copy %wrap : reg<aie2p.er> -> reg<aie2p.mr26_lock>\n  rel.cond %one, %wrq, 1",
                            "  %waq = copy %wwa : reg<aie2p.er> -> reg<aie2p.mr26_lock>\n  acq.cond %am1, %waq, 1"] + keep[3:]
            units[0] = keep + units[0]
        else:
            units[0] = init + units[0]
        if mb2 and os.environ.get("GN_QMB1") == "5" and t in (0, NP // 2, NT):   # (MB2) the memory tokens, ahead of
            # the sub-tile's W fill: first half = own memory's token (port 3, sender), second = the other's (port 4)
            # (GN_QP1) a firing starts with a handshake: row 1 releases one more credit after its first W acquire (entry
            # block), row 0 takes two at t 0: its first remote reads (floated a few cycles above their token acquire:
            # no dependency) then never precede row 1's first W slot. Later reads trail the partner by a half row.
            if name == "tailm0":   # (sender, port 3)
                tl = ((["  rel %one, 3", "  %am2 = mova.i32 -2", "  acq %am2, 3"] if qp1_on() else
                       ["  rel %one, 3", "  acq %am1, 3"]) if t == 0 else
                      ["  rel %one, 3", "  acq %am1, 3"] if t == NP // 2 else
                      ["  rel.cond %one, %mbq, 3", "  acq.cond %am1, %mbq, 3"])
            else:   # (receiver, port 4)
                tl = ((["  rel %one, 4", "  acq %am1, 4"] if qp1_on() else ["  acq %am1, 4"]) if t == 0 else
                      ["  rel %one, 4", "  acq %am1, 4"] if t == NP // 2 else
                      ["  rel %one, 4", "  acq.cond %am1, %mbq, 4"])
            if t == NT:
                (prologue if base - 3 < 0 else slots.setdefault(base - 3, [])).append(
                    "  %mbq = copy %nxm : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
            # GN_QTOK=r1,a1,r2,a2 (MMA slots from base): the t = NP / 2 release / acquire and the row end's, spread (a
            # lumped lock pair leaves bundles without an MMA). The half-switch acquire stays ahead of the remote fills
            # (a1 <= 0); the row end's lines stay within the r26 copy's window (>= -2, < 21: %wrq's copy)
            ofs = [int(v) for v in os.environ.get("GN_QTOK", "0,0,0,0").split(",")]
            o_ = (0, 0) if t == 0 else (ofs[0], ofs[1]) if t == NP // 2 else (ofs[2], ofs[3])
            assert t != NP // 2 or o_[1] <= 0
            assert t != NT or (min(o_) >= -2 and max(o_) < 21)
            if t and len(tl) == 2:
                for ln, o in zip(tl, o_):
                    (prologue if base + o < 0 else slots.setdefault(base + o, [])).append(ln)
                tl = []
            (prologue if base < 0 else slots.setdefault(base, [])).extend(tl)
        for i, line in enumerate(early):
            g = base - (len(early) - i)
            (prologue if g < 0 else slots.setdefault(g, [])).append(line)
        for u, lines in enumerate(units):
            g = base + u
            if qrow and t == NT and g >= (NT * ks + QE2r) * 4:
                continue
            if tail and cfg.dfr and t - 1 in DFR_T and g < t * ks * 4:   # held past the deferred push (vector registers)
                held.setdefault(t, []).extend(lines)
                continue
            (prologue if g < 0 else slots.setdefault(g, [])).extend(lines)

    def a_next(e):
        """%pan: the next iteration's A slab pointer (seam: the last iteration's stays put, the carried pops read it)."""
        steps = cfg.mu if a_adv else 1
        mf = "%mf"
        if seam and a_adv:
            e("  %fsn = mul %fstep, %nxm")
            e("  %mfn = mov.modifier %fsn")
            mf = "%mfn"
        qa2_ = False
        if not a_adv and cfg.mu == 1 and acap in (2, 4) and qa4_on():   # (GN_QA4) slot i = mp & 3 (or mp & 1: 2 slots)
            q4m = "%mp"
            if qa4_carry(cfg):
                e("  %q4mp = add.rr %mp, %qai")
                q4m = "%q4mp"
            a_ = qa4_slots()
            if len(a_) == 2:   # (two slots: +d, -d)
                e(f"  %q4o = and {q4m}, %one")
                e(f"  %q4a = mov.i32 {a_[1] - a_[0]}")
                e(f"  %q4b = mov.i32 {2 * (a_[0] - a_[1])}")
                e("  %q4x = mul %q4b, %q4o")
                e("  %pst = add.rr %q4a, %q4x")
                qa2_ = True
        if qa2_:
            pass
        elif not a_adv and cfg.mu == 1 and acap == 4 and qa4_on():
            e("  %q4k3 = mova.i32 3")
            e(f"  %q4i = and {q4m}, %q4k3")
            e(f"  %q4o = and {q4m}, %one")
            e("  %q4l = lt %k2, %q4i")
            d_ = [a_[(i + 1) % 4] - a_[i] for i in range(4)]
            e(f"  %q4a = mov.i32 {d_[0]}")
            e(f"  %q4b = mov.i32 {d_[1] - d_[0]}")
            e(f"  %q4c = mov.i32 {d_[3] - d_[1] - (d_[2] - d_[0])}")
            e("  %q4x = mul %q4b, %q4o")
            e("  %q4y = mul %q4c, %q4l")
            e("  %q4xy = add.rr %q4x, %q4y")
            if d_[0] != d_[2]:   # (slots not evenly paired, e.g. 64x96 tiles) + (d2 - d0) for i >= 2
                e(f"  %q4g = lt %one, %q4i")
                e(f"  %q4h = mov.i32 {d_[2] - d_[0]}")
                e("  %q4z = mul %q4h, %q4g")
                e("  %q4xz = add.rr %q4xy, %q4z")
                e("  %pst = add.rr %q4a, %q4xz")
            else:
                e("  %pst = add.rr %q4a, %q4xy")
        elif not a_adv and (cfg.mu, acap) == (1, 3):   # (GN_QA3) slot j = (mp + %qai) % 3: +4 KB, from slot 2 -8 KB
            e("  %q3j = add.rr %mp, %qai")
            e("  %q3k = mova.i32 171")   # j // 3 = (j * 171) >> 9 for j < 256
            e("  %q3m = mul %q3j, %q3k")
            e("  %q3s = mova.i32 -9")
            e("  %q3d = lshl %q3m, %q3s")
            e("  %q3t = mova.i32 3")
            e("  %q3d3 = mul %q3d, %q3t")
            e("  %q3i = sub %q3j, %q3d3")
            e("  %q3w = lt %one, %q3i")
            e("  %q3a = mul %k512, %q3s8")
            e("  %q3bk = mova.i32 -24")
            e("  %q3b = mul %k512, %q3bk")
            e("  %q3x = mul %q3b, %q3w")
            e("  %pst = add.rr %q3a, %q3x")
        elif not a_adv and (cfg.mu, acap) == (1, 2):   # alternate slots: step = fstep (1 - 2 (mp & 1))
            e("  %ph = lshl %mp, %am1")
            e("  %ph2 = add.rr %ph, %ph")
            e("  %par = sub %mp, %ph2")
            fs = "%fstep"
            if qbank:   # (GN_QBANK) the A slots 16 KB apart; W keeps the slab stride (%mf)
                e(f"  %kqa = mova.i32 {16 if q3 else 32}")   # (q3: A slot 1 at bank 2's start, 8 KB above slot 0)
                e("  %fsq = mul %k512, %kqa")
                fs = "%fsq"
            e(f"  %fst2 = add.rr {fs}, {fs}")
            e("  %pst0 = mul %par, %fst2")
            if seam:
                e(f"  %pstr = sub {fs}, %pst0")
                e("  %pst = mul %pstr, %nxm")
            else:
                e(f"  %pst = sub {fs}, %pst0")
        if spa and not a_adv and (cfg.mu, acap) in ((1, 2), (1, 3), (1, 4)):
            e("  %pan = add.rr %pa, %pst")
        elif not a_adv and (cfg.mu, acap) in ((1, 2), (1, 3), (1, 4)):
            e("  %pac = copy %pa : reg<aie2p.ep> -> reg<aie2p.ep>")
            e("  %pmod = mov.modifier %pst")
            e("  %pan = padds.modifier %pac, %pmod")
        elif spa:   # scalar A pointer: advance by steps slab strides
            e(f"  %kpst = mova.i32 {steps if a_adv else 0}")
            e("  %pastep = mul %fstep, %kpst")
            e("  %pan = add.rr %pa, %pastep")
        else:
            e("  %pac = copy %pa : reg<aie2p.ep> -> reg<aie2p.ep>")
            e(f"  %pan0 = padds.modifier %pac, {mf if a_adv else '%mz'}")
            for j in range(1, steps):
                e(f"  %pan{j}c = copy %pan{j - 1} : reg<aie2p.ep> -> reg<aie2p.ep>")
                e(f"  %pan{j} = padds.modifier %pan{j}c, {mf}")
            e(f"  %pan = copy %pan{steps - 1} : reg<aie2p.ep> -> reg<aie2p.ep>")

    if seam:
        defd, used = [], set()
        for ln in seam0:
            lhs, _, rhs = ln.partition("=")
            if rhs:
                defd += re.findall(r"%\w+", lhs)
            used |= set(re.findall(r"%\w+", rhs if rhs else lhs))
        kind = lambda n: ("mexa" if re.match(r"%s0[aw]\d", n) else "eps" if n.startswith("%s0p") else
                          "eldfiforeg" if n.startswith("%s0f") else "erf2" if n.startswith("%s0q") else None)
        carried = [n for n in defd if kind(n) == "mexa" or (kind(n) and n not in used)]
        assert all(kind(n) for n in carried) and len([n for n in carried if kind(n) == "mexa"]) == wb

        def seam_copy(pfx, pa_, fa_, fw_):
            """seam0 with its own names prefixed pfx, the A pointer pa_, the fifos fa_ / fw_."""
            out = []
            for ln in seam0:
                ln = re.sub(r"%s0(\w*)", lambda m: f"%{pfx}" + m.group(1) if "%s0" + m.group(1) in defd else m.group(0), ln)
                ln = re.sub(r"%pa\b", pa_, ln)
                out.append(re.sub(r"%fw\b", fw_, re.sub(r"%fa\b", fa_, ln)))
            return out
        ren = lambda pfx: [f"%{pfx}" + n[3:] for n in carried]
        cty = lambda names: "".join(f", {n}: reg<aie2p.{kind('%s0' + n[3:])}>" for n in names)
        # the header tests the latch's condition (%more; the entry's: MP // mu > 0)
        assert L[hdr + 2].strip() == "%omore = lt %mp, %nmp"
        L[hdr:hdr + 4] = seam_copy("e0", a0, "%fa0", "%fw0") + [
            f"  low.br ^outer(%zero: reg<aie2p.er>, {a0}: {pat}, {po0}: {pct}, %one: reg<aie2p.er>{cty(ren('e0'))})",
            f"^outer(%mp: reg<aie2p.er>, %pa: {pat}, %po: {pct}, %more: reg<aie2p.er>{cty(carried)}):",
            "  low.cond_br %more, ^obody, ^exit : reg<aie2p.er>"]
        # the latch: its scalar work rides with the last sub-tile's MMAs, the next iteration's A slab (acquired only if
        # it runs) and init with its last 9 (no pops: the last pop is 10 MMAs before the end), the WB pops with the writes
        nxt0 = seam_copy("n0", "%pan", fifo["a"], fifo["w"])
        assert nxt0[0].strip() == "acq %am1, 0"
        lat = ["  %mpn = add.rr %mp, %one", "  %nxm = lt %mpn, %nmp",
               "  %cqn = copy %nxm : reg<aie2p.er> -> reg<aie2p.mr26_lock>"]
        a_next(lat.append)
        init_n = [ln for ln in nxt0[1:] if ".pop." not in ln]
        pos = [ln for ln in init_n if "fifo.position" in ln]
        # the last 9 MMAs: the positions, the release (pops done), the conditional acquire (lock latency apart); the
        # MMA -> write gap: the pointer copies and fills (as at the inner boundaries)
        G = NT * ks * 4
        assert len(pos) == 2 and len(lat) <= ks * 4 - 9 - wb and max(slots) < G - 9
        # one op per B-pop MMA (a rematerialized constant's MOVA takes the A load slot; every other MMA hides latency)
        bs = [g for g in range(G - 10, G - ks * 4, -1) if any("vldb.pop" in ln for ln in slots.get(g, []))
              and not any("vlda." in ln for ln in slots[g])][:len(lat)]
        assert len(bs) == len(lat)
        for g, ln in zip(sorted(bs), lat):
            slots[g].append(ln)
        for g, ln in ((G - 9, pos[0]), (G - 8, pos[1]), (G - 7, "  rel %one, 0"), (G - 3, "  acq.cond %am1, %cqn, 0")):
            slots.setdefault(g, []).append(ln)
        wblock[NT - 1] = [[ln for ln in init_n if ln not in pos]] + [[ln] for ln in nxt0 if ".pop." in ln]

    if qrow:   # one line per two MMAs in sub-tile 1: the next row's slab pointer, C row deltas, counter
        lat = []
        a_next(lat.append)
        if qbank:
            v2 = os.environ.get("GN_QBANK") == "2"
            if q3:   # each area's rows a fixed stride apart (q3_strides): a chain's +1 KB step lands on the next row;
                # row MP - 1 wraps (NP odd: the chains' strides differ, each its own wrap)
                ses, sos = q3_strides()
                rowq = (["  %qk3 = mova.i32 3", "  %qmq = and %mp, %qk3"] if MP == 4 else div_mp("%mp", "%qmd", "qmd") +
                        [f"  %qk5 = mova.i32 {MP}", "  %qm5 = mul %qmd, %qk5", "  %qmq = sub %mp, %qm5"])
                lat += rowq + [f"  %qk2 = mova.i32 {MP - 2}", "  %wrap = lt %qk2, %qmq"]
                if qalt_on():   # t0 <- the O chain + d0 (E's start), t1 <- the E chain - d0 - MP NP KB (O's start)
                    lat += ["  %cdel = mul %wrap, %qd0", f"  %qkz = mova.i32 {-MP * NP * 2}", "  %qkzb = mul %k512, %qkz",
                            "  %qwz = mul %wrap, %qkzb", "  %cdelo = sub %qwz, %cdel", "  %mcdo = mov.modifier %cdelo"]
                else:
                    lat += [f"  %qkw = mova.i32 {-ses * MP // 512}", "  %qkwa = mul %k512, %qkw", "  %cdel = mul %wrap, %qkwa"]
                if NP % 2 and not qalt_on():
                    lat += [f"  %qkwo = mova.i32 {-sos * MP // 512}", "  %qkwao = mul %k512, %qkwo",
                            "  %cdelo = mul %wrap, %qkwao", "  %mcdo = mov.modifier %cdelo"]
            else:
                lat += [f"  %kcd0a = mova.i32 {40 if v2 else 24}", "  %kcd0 = mul %k512, %kcd0a",
                        f"  %kcd1a = mova.i32 {72 if v2 else 56}", "  %kcd1 = mul %k512, %kcd1a", "  %cdp = mul %par, %kcd1"]
            if q3:
                pass
            elif qpass:   # row MP - 1 -> row 0 of the next pass: the C rows wrap (%wrap: the row ends a pass)
                cs_ = qpass_cslots()
                wadj = cs_[0] - (cs_[-1] + 1024 * NP + 12288 - 28672)
                lat += ["  %cdel0 = sub %kcd0, %cdp", "  %qk3 = mova.i32 3", "  %qmq = and %mp, %qk3", "  %qk2 = mova.i32 2",
                        "  %wrap = lt %qk2, %qmq", f"  %qkw = mova.i32 {wadj // 512}", "  %qkwa = mul %k512, %qkw",
                        "  %cwr = mul %wrap, %qkwa", "  %cdel = add.rr %cdel0, %cwr"]
            else:
                lat += ["  %cdel = sub %cdp, %kcd0" if v2 else "  %cdel = sub %kcd0, %cdp"]
            lat += ["  %mcd = mov.modifier %cdel"] + ([] if qpass else ["  %cdl = mul %cdel, %nf", "  %mcdl = mov.modifier %cdl"])
        lat += ["  %mpn = add.rr %mp, %one", "  %nxm = lt %mpn, %nmp"]
        if qpass:   # the W slot: acquired at a pass end if a row follows; the next row's slot (pass parity)
            assert (qw4_deltas()[0] if qw4_on() else qpass_wdelta()) % 512 == 0, "W slot delta: a multiple of 512"
            lat +=["  %wwa = mul %wrap, %nxm"] + div_mp("%mpn", "%qpq", "qpd") + [
                    *(["  %qpqw = add.rr %qpq, %qwp", "  %qppn = and %qpqw, %one"] if cfg.passes % 2 else
                      ["  %qppn = and %qpq, %one"]), f"  %qwda = mova.i32 {(qw4_deltas()[0] if qw4_on() else qpass_wdelta()) // 512}",
                    "  %qwd = mul %k512, %qwda", "  %qwo = mul %qppn, %qwd"]
            if qw4_on():   # (GN_QW4) + (pass & 2) d2 / 2
                d2 = qw4_deltas()[1]
                assert cfg.passes % 4 == 0 and d2 % 1024 == 0, (cfg.passes, d2)
                lat += ["  %qw4k = mova.i32 2", "  %qw4b = and %qpq, %qw4k", f"  %qw4a = mova.i32 {d2 // 1024}",
                        "  %qw4d = mul %k512, %qw4a", "  %qw4o = mul %qw4b, %qw4d", "  %qwo4 = add.rr %qwo, %qw4o",
                        "  %wcn = add.rr %wsb, %qwo4"]
            else:
                lat += ["  %wcn = add.rr %wsb, %qwo"]
        elif os.environ.get("GN_QROW") == "1":   # (first pass: rows 1.. acquire their C slot at the row-end window)
            lat += ["  %cfn = mul %first, %nxm"]
        # (the pointer itself just before the row-end window: one fewer pointer live through the row)
        late_ = [ln for ln in lat if ln.split(" = ")[0].strip() in ("%pac", "%pan")]
        lat = [ln for ln in lat if ln not in late_]
        # one line per two MMAs from sub-tile 1's 3rd k-block after its window to 6 before the next, then sub-tile 2's
        gs_ = [g for s_ in range(1, NT - 1) for g in range((s_ * ks + QE2r + 2) * 4, (s_ * ks + ks - 6) * 4, 2)]
        if len(gs_) < len(lat):   # (short sub-tiles, ks < 28: one line per MMA)
            gs_ = [g for s_ in range(1, NT - 1) for g in range((s_ * ks + QE2r + 2) * 4, (s_ * ks + ks - 6) * 4)]
        if len(gs_) < len(lat):   # (ks 16: sub-tile 0's too, and up to 3 lines an MMA)
            gs_ = list(range((QE2r + 2) * 4, (ks - 6) * 4)) + gs_
            gs_ = sorted(gs_ * 3)
        assert len(gs_) >= len(lat) and len(late_) == 2, (len(gs_), len(lat), len(late_), ks, QE2r, QEr)
        for g, ln in zip(gs_, lat):
            slots.setdefault(g, []).append(ln)
        for i, ln in enumerate(late_):
            slots.setdefault((NT * ks - QEr) * 4 - 4 + i, []).append(ln)

    def emit_writes(t):
        blk = wblock.get(t, [[]])
        for line in blk[0]:
            e(line)
        units = blk[1:]
        wire = [(c, q) for q in range(4) for c in ORDER_C]   # the order the consumer's fused MMAs read
        for i, (c, q) in enumerate(wire):
            e(f"  %o{t}_{c}_{q} = slice {cur[c]}[{q}] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>")
            e(f"  vmov.mcd.acc %o{t}_{c}_{q}")
            for line in (units[i] if i < len(units) else []):
                e(line)
        for unit in units[len(wire):]:
            for line in unit:
                e(line)

    n_init = 6
    cur = [None] * 4
    if qrow:   # the first row's sub-tile 0 start: the entry, before the loop (renamed and spliced in at the end)
        ent, rest = list(prologue[:n_init]), prologue[n_init:]
        lds = [f"  %l0_{c}_{q} = vlda.acc {pl[0]}, {256 * c + 64 * q - 512}" for c in range(4) for q in range(4)]
        for i, line in enumerate(lds):
            ent.append(line)
            if i < len(rest):
                ent.append(rest[i])
        ent += rest[len(lds):]
        c0 = []
        for c in range(4):
            ent.append(f"  %z0_{c} = concat(%l0_{c}_0, %l0_{c}_1, %l0_{c}_2, %l0_{c}_3) : {MBMS4}")
            c0.append(f"%z0_{c}")
        if qzacc_on():   # (the entry is row 0 of pass 0: its first MMAs ignore the unzeroed C)
            assert qpass
            ent.append("  %qzc0 = add.rr %conf, %one")
        for k in range(QE2r):
            for ci, c in enumerate(ORDER_C):
                cf_ = "%qzc0" if qzacc_on() and k == 0 else "%conf"
                ent.append(f"  %m0_{k}_{c} = mma.bfp16ebs8.m8n8k8 {c0[c]}, %s0a{k}_{c >> 1}, %s0w{k}_{c & 1}, {cf_}")
                c0[c] = f"%m0_{k}_{c}"
                ent += slots.pop(k * 4 + ci, [])
        if not qpass:   # (qpass: the leaf acquired every C slot)
            e("  %cqa0 = copy %first : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
            e("  acq.cond %am1, %cqa0, 2")
        cur = c0
        prologue = rest = []
    for line in prologue[:n_init]:
        e(line)
    rest = prologue[n_init:]
    if qrow:
        pass
    elif tail and cfg.dfr:   # (r26 holds the deferred fifo's position: a branch, not acq.cond)
        e("  low.cond_br %first, ^qa, ^qj : reg<aie2p.er>")
        e("^qa:")
        e("  acq %am1, 2")
        e("  low.br ^qj")
        e("^qj:")
    elif tail:
        e("  %cqa0 = copy %first : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
        e("  acq.cond %am1, %cqa0, 2")
    if tail and not qrow:
        lds = [f"  %l0_{c}_{q} = vlda.acc {pl[0]}, {256 * c + 64 * q - 512}" for c in range(4) for q in range(4)]
        for i, line in enumerate(lds):
            e(line)
            if i < len(rest):
                e(rest[i])
        for line in rest[len(lds):]:
            e(line)
        for c in range(4):
            e(f"  %z0_{c} = concat(%l0_{c}_0, %l0_{c}_1, %l0_{c}_2, %l0_{c}_3) : {MBMS4}")
            cur[c] = f"%z0_{c}"
    elif not qrow:
        for line in rest:
            e(line)
    # GN_QCORE=<boundary json> (experimental, layout Q'): the tail without a cascade input (plain MMAs from C) and each
    # in-row sub-tile boundary emitted in a solved order (night/sim/bound.py DUMP): the old sub-tile's last E k-blocks,
    # its C stores, the next sub-tile's C loads and first E2 k-blocks interleaved. Row ends stay serial.
    qcore = tail and bool(os.environ.get("GN_QCORE"))
    if qcore:
        assert not cfg.swiglu and not cfg.dfr and not cfg.ofeat
        QW = json.load(open(os.environ["GN_QCORE"]))
        QE, QE2 = QW["E"], QW["E2"]
        assert ks >= QE + QE2
    # (GN_QNOROT=1: every boundary serial, a control for the windows)
    QDRIP = int(os.environ.get("GN_QDRIP", "0"))
    QEAGER = int(os.environ.get("GN_QEAGER", "0"))
    rot = lambda t: qcore and not os.environ.get("GN_QNOROT") and (t + 1 < NT and (t + 1) % NP != 0 or qrow and t == NT - 1)

    def emit_window(t):
        """The rotated boundary t -> t + 1 (both sub-tiles' window k-blocks, stores, loads); leaves cur = t + 1's."""
        if qzacc_on() and t == 0:   # (GN_QZACC) conf + 1 while this row / the next row is in pass 0 (row mp < MP)
            assert qpass and qrow
            e(f"  %qzk = mova.i32 {MP}")
            e("  %qzl = lt %mp, %qzk")
            e("  %qzc = add.rr %conf, %qzl")
            e("  %qzm = add.rr %mp, %one")
            e("  %qzl1 = lt %qzm, %qzk")
            e("  %qzcn = add.rr %conf, %qzl1")
        if qpass and qclazy_on() and t == NT - 1:   # (GN_QCLAZY) pass 0's next row's C slot (rows 1 .. MP - 1)
            e("  %qcm = add.rr %mp, %one")
            e(f"  %qck = mova.i32 {MP}")
            e("  %qcl = lt %qcm, %qck")
            e("  %qclk = copy %qcl : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
            e("  acq.cond %am1, %qclk, 2")
        if qrow and not qpass and t == NT - 1:   # the next row's C slot on the first pass (before its loads)
            e("  %cqc = copy %cfn : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
            e("  acq.cond %am1, %cqc, 2")
        if q3:   # (q3) two chains, even and odd sub-tiles: t + 1's pointer is t - 1's + 1 KB (+ the row wrap at the row
            # end); sub-tile 1's is the carried %pl; a sub-tile's loads and stores share its pointer
            if t == 0:
                pl[1] = "%pl"
            else:
                src_ = pc[t] if t == NT - 1 and NP % 2 and not qalt_on() else pc[t - 1]
                e(f"  %plc{t + 1} = copy {src_} : reg<aie2p.ep> -> reg<aie2p.ep>")
                e(f"  %pl{t + 1}{'g' if t == NT - 1 else ''} = padds.modifier %plc{t + 1}, %mcs")
                if t == NT - 1:
                    e(f"  %pl{t + 1} = padds.modifier %pl{t + 1}g, %mcd")
                pl[t + 1] = f"%pl{t + 1}"
        else:
            e(f"  %plc{t + 1} = copy {pl[t]} : reg<aie2p.ep> -> reg<aie2p.ep>")
            if qrow and qbank and t == NT - 1:   # the next row's C slot
                e(f"  %pl{t + 1}g = padds.modifier %plc{t + 1}, %mls")
                e(f"  %pl{t + 1} = padds.modifier %pl{t + 1}g, {"%mcd" if qpass else "%mcdl"}")
            else:
                e(f"  %pl{t + 1} = padds.modifier %plc{t + 1}, %mls")
        g0, gi, new = (t * ks + ks - QE) * 4, 0, {}
        # the window's pops and fills in stream order, pulled by need: before each MMA every line defining an operand
        # of the next LAGU MMAs (the window's own order, which moves some new-chain MMAs ahead of their slots)
        pend = [ln for g in range(g0, g0 + 4 * (QE + QE2)) for ln in slots.get(g, [])]
        mm = [ev for _, ev in QW["events"] if ev[0] == "mma"]
        ops = [{f"%s{t if ev[1] == 'o' else t + 1}a{(ks if ev[1] == 'o' else 0) + ev[3]}_{ev[2] >> 1}",
                f"%s{t if ev[1] == 'o' else t + 1}w{(ks if ev[1] == 'o' else 0) + ev[3]}_{ev[2] & 1}"} for ev in mm]
        defd = set()
        pdefs = {n for ln in pend if "=" in ln for n in re.findall(r"%\w+", ln.partition("=")[0])}

        def side_of(ln):
            if qpass and ("%wrq" in ln or "%waq" in ln):   # (in W stream order)
                return "w"
            if qrow and ("%cqn" in ln or ln.strip() == "rel %one, 0"):   # (in A stream order)
                return "a"
            lhs, eq, _ = ln.partition("=")
            m = re.match(r"%s\d+r?[pfq]?(am|a|w)", (re.findall(r"%\w+", lhs) or [""])[0]) if eq else None
            return ("w" if m.group(1) == "w" else "a") if m else None

        def pull(name):
            """Emit this FIFO side's pending lines (and side-less ones) up to the one defining name."""
            sd = side_of(next(ln for ln in pend if name in re.findall(r"%\w+", ln.partition("=")[0])))
            i = 0
            while name not in defd:
                ln = pend[i]
                if side_of(ln) in (sd, None):
                    pend.pop(i)
                    defd.update(re.findall(r"%\w+", ln.partition("=")[0]) if "=" in ln else [])
                    e(ln)
                else:
                    i += 1

        def eager():
            """(GN_QEAGER) after an MMA: per FIFO side, its earliest pending line when it is not a pop and its inputs
            are defined (the next sub-tile's pointer copies, positions, fills ride with MMAs instead of clustering
            before its first pop); side-less lines ahead of it go first. QEAGER 2: at most one line per MMA."""
            n = 0
            for sd in ("a", "w"):
                i = next((k for k, ln in enumerate(pend) if side_of(ln) == sd), None)
                if i is None or ".pop." in pend[i] or n and QEAGER == 2:
                    continue
                if any(side_of(pend[k]) is None for k in range(i)):
                    continue
                uses = set(re.findall(r"%\w+", pend[i].partition("=")[2] if "=" in pend[i] else pend[i]))
                if (uses & pdefs) - defd:
                    continue
                ln = pend.pop(i)
                defd.update(re.findall(r"%\w+", ln.partition("=")[0]) if "=" in ln else [])
                e(ln)
                n += 1
        for _, ev in QW["events"]:
            if ev[0] == "pop":   # the solver's pop time
                s_, sd, r_, j_ = ev[1:]
                nm = f"%s{t if s_ == 'o' else t + 1}{sd}{(ks if s_ == 'o' else 0) + j_}_{r_}"
                if nm in pdefs and nm not in defd:
                    pull(nm)
                continue
            if ev[0] == "mma":
                for nm in sorted(ops[gi] & pdefs - defd):
                    pull(nm)
                side, c, j = ev[1], ev[2], ev[3]
                tt, kk = (t, ks + j) if side == "o" else (t + 1, j)
                cf_ = "%conf"
                if side == "n" and c not in new:
                    e(f"  %z{t + 1}_{c} = concat(%l{t + 1}_{c}_0, %l{t + 1}_{c}_1, %l{t + 1}_{c}_2, %l{t + 1}_{c}_3) : {MBMS4}")
                    new[c] = f"%z{t + 1}_{c}"
                    if qzacc_on():   # (the row-end window starts the next row's sub-tile 0: the next row's pass)
                        cf_ = "%qzcn" if t == NT - 1 else "%qzc"
                acc = cur[c] if side == "o" else new[c]
                e(f"  %m{tt}_{kk}_{c} = mma.bfp16ebs8.m8n8k8 {acc}, %s{tt}a{kk}_{c >> 1}, %s{tt}w{kk}_{c & 1}, {cf_}")
                if side == "o":
                    cur[c] = f"%m{tt}_{kk}_{c}"
                else:
                    new[c] = f"%m{tt}_{kk}_{c}"
                gi += 1
                if QEAGER:
                    eager()
                if QDRIP and gi > 4 * (QE + QE2) - QDRIP and pend:   # (GN_QDRIP=n) the pops ahead for the MMAs after the
                    ln = pend.pop(0)                                  # window, one per MMA over its last n MMAs
                    defd.update(re.findall(r"%\w+", ln.partition("=")[0]) if "=" in ln else [])
                    e(ln)
            elif ev[0] == "drain":
                c, q = ev[1], ev[2]
                e(f"  %o{t}_{c}_{q} = slice {cur[c]}[{q}] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>")
                e(f"  vst.acc %o{t}_{c}_{q}, {pc[t]}, {256 * c + 64 * q - 512}")
            else:
                n, q = ev[1], ev[2]
                e(f"  %l{t + 1}_{n}_{q} = vlda.acc {pl[t + 1]}, {256 * n + 64 * q - 512}")
        for ln in pend:   # (pops ahead for the MMAs after the window)
            e(ln)
        assert gi == 4 * (QE + QE2) and len(new) == 4
        if q3:   # t + 1 stores where it loaded; after the row end, the next row's odd chain from this row's sub-tile 3
            pc[t + 1] = pl[t + 1]
            if t == NT - 1:
                e(f"  %pqoc = copy {pc[t - 1] if NP % 2 and not qalt_on() else pc[t]} : reg<aie2p.ep> -> reg<aie2p.ep>")
                e("  %pqog = padds.modifier %pqoc, %mcs")
                e(f"  %pqo = padds.modifier %pqog, {'%mcdo' if NP % 2 else '%mcd'}")
        else:
            e(f"  %pcc{t + 1} = copy {pc[t]} : reg<aie2p.ep> -> reg<aie2p.ep>")
            if qrow and qbank and t == NT - 1:
                e(f"  %pc{t + 1}g = padds.modifier %pcc{t + 1}, %mcs")
                e(f"  %pc{t + 1} = padds.modifier %pc{t + 1}g, %mcd")
            else:
                e(f"  %pc{t + 1} = padds.modifier %pcc{t + 1}, %mcs")
        for c in range(4):
            cur[c] = new[c]

    def convert(base, lp, back):
        """Last pass: pack the segment's f32 C (NP sub-tiles) in place to bf16 in its first half, which the C ring sends
        (LOOM_EXP_LS_SEND_PITCH=2). base is a store pointer back bytes past the segment's slot + 512; on a last pass the C
        load pointer lp equals it, so lp walks the f32 source (post-increment loads) while base walks the bf16
        destination (post-increment stores); both are rebuilt for the join.
        It runs between a sub-tile's C stores and the next C loads, so all accumulators are free and loads run 16 ahead.
        Returns the join's (base, lp)."""
        e(f"  low.cond_br %clast, ^cv{t}, ^nc{t} : reg<aie2p.er>")
        e(f"^nc{t}:")
        e(f"  low.br ^cj{t}({base}: reg<aie2p.ep>, {lp}: reg<aie2p.ep>)")
        e(f"^cv{t}:")
        d, k, rest = base, 0, back + 512
        while rest > 0:
            step = min(rest, 448)
            rest -= step
            e(f"  %cvb{t}_{k} = padda {d}, {-step}")
            d, k = f"%cvb{t}_{k}", k + 1
        e(f"  %cvs{t}_0 = copy {d} : reg<aie2p.ep> -> reg<aie2p.ep>")
        if QCYC:
            e(f"  %cvt{t} = copy {d} : reg<aie2p.ep> -> reg<aie2p.ep>")
        s_ = f"%cvs{t}_0"
        n = 16 * NP

        def load(i):
            nonlocal s_
            e(f"  %cvl{t}_{i}, %cvs{t}_{i + 1} = vlda.acc.post {s_}, 64")
            s_ = f"%cvs{t}_{i + 1}"
        for i in range(16):
            load(i)
        for i in range(n):
            e(f"  %cvd{t}_{i} = vst.convert.f32x16.to.bf16x16.post %cvl{t}_{i}, {d}, 32")
            d = f"%cvd{t}_{i}"
            if i + 16 < n:
                load(i + 16)
        if QCYC:   # (diag) the cycles since the first firing's start into the segment's first word
            e(f"  %tcw{t} = mov.cycles")
            e(f"  %tcl{t} = slice %tcw{t}[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
            e(f"  %tsa{t} = storage_address %tcs : low.storage<private> -> reg<aie2p.ep>")
            e(f"  %tst{t} = lda %tsa{t}, 0")
            e(f"  %tsp{t} = sub %tcl{t}, %tst{t}")
            e(f"  st %tsp{t}, %cvt{t}, 0")
            if QCYCABS:
                e(f"  st %tst{t}, %cvt{t}, 4")
            if QFCYC and tail and QF.get("cfg") is not None and fills is True:
                e(f"  %qfa{t} = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
                e(f"  %qfe{t} = lda %qfa{t}, 4")
                e(f"  %qff{t} = lda %qfa{t}, 12")
                e(f"  st %qfe{t}, %cvt{t}, 4")
                e(f"  st %qff{t}, %cvt{t}, 8")
        bump(L, f"%cvr{t}", d, back + 512 - 32 * n)
        e(f"  %cvq{t} = copy %cvr{t} : reg<aie2p.ep> -> reg<aie2p.ep>")
        e(f"  low.br ^cj{t}(%cvr{t}: reg<aie2p.ep>, %cvq{t}: reg<aie2p.ep>)")
        e(f"^cj{t}(%cj{t}p: reg<aie2p.ep>, %cj{t}l: reg<aie2p.ep>):")
        e(f"  %cqr{t} = copy %clast : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
        e(f"  rel.cond %one, %cqr{t}, 2")
        return f"%cj{t}p", f"%cj{t}l"
    def convert_swiglu(base, lp, back, segs=1, unit=None, carry=None, q3=None):
        """Last pass (cfg.swiglu): slabs 0 .. NP / 2 - 1 of a segment are gate rows, the rest up rows of the same
        features. Per 8 x 8 block pair (8 tokens, 8 features) h = silu(g) * u through bf16 products with f32
        results (sigmoid below); a segment's 2 NP h blocks ([feature block][token half] in push order) go out as
        bfp16ebs8 fragments (SW_REC) at its slot start, in place (every block is read before the store fifo reaches
        it). A loop over the iteration's segs segments (the earlier ones deferred to here; the 16 KB program memory
        holds one block's code). Returns the join's (base, lp).
        unit (cfg.dfr): instead one push of the deferred segment at %dss (its h block 2 %mp + t's place in DFR_T)
        on the store fifo unit; returns the join's (base, lp, fifo). carry (cfg.dfr, the whole epilogue): the
        deferred fifo, through the join (dead on the epilogue's path: one store fifo); returns (base, lp, fifo).
        q3 (cfg.qsw, the Q' row convert): (even area, odd area) pointers of one row (gate sub-tiles 0, 1 at each one's
        start, up sub-tiles 2, 3 a KB on), h at the even one's start; no branch, no loop, nothing returned."""
        ca = "".join(f", {x}: {ty}" for x, ty in zip(carry, DFT.split(", "))) if carry else ""
        cnd = "%clb" if cfg.dfr else "%clast"
        if unit:   # the C pointers parked as scalars across the push (the tail's ep budget is 8)
            e(f"  %dub{t} = mov.address-to-scalar {base}")
            e(f"  %dul{t} = mov.address-to-scalar {lp}")
        if unit and not cfg.dfi:
            e(f"  low.cond_br %dfx, ^du{t}, ^dn{t} : reg<aie2p.er>")
            e(f"^dn{t}:")
            e(f"  low.br ^dj{t}({unit[0]}: reg<aie2p.mstfifo>, {unit[1]}: reg<aie2p.mpfs>, {unit[2]}: reg<aie2p.mr26_fifo_st>)")
            e(f"^du{t}:")
        elif not unit and not q3:
            e(f"  low.cond_br {cnd}, ^cv{t}, ^nc{t} : reg<aie2p.er>")
            e(f"^nc{t}:")
            e(f"  low.br ^cj{t}({base}: reg<aie2p.ep>, {lp}: reg<aie2p.ep>{ca})")
            e(f"^cv{t}:")
        d, k, rest = base, 0, (0 if unit or q3 else back + 512 + SW_SLOT * (segs - 1))   # the first deferred segment's slot
        while rest > 0:
            step = min(rest, 448)
            rest -= step
            e(f"  %cvb{t}_{k} = padda {d}, {-step}")
            d, k = f"%cvb{t}_{k}", k + 1
        p = f"%sw{t}_"
        V1, V2 = "reg<aie2p.vec256>", "reg<aie2p.vec256 x2>"
        M1, M2, M4 = "reg<aie2p.mbms>", "reg<aie2p.mbms x2>", "reg<aie2p.mbms x4>"
        ER = "reg<aie2p.er>"
        if unit:   # push u = 2 mp + (t's place in DFR_T): (j, i, r) = (u >> 2, u >> 1 & 1, u & 1), in P4 / Q4K order
            # (one format: both per two slabs): slab i, half j, the gate block at 1024 (mp & 1) + 256 (mp >> 1) +
            # 512 r + 512 (jloop's gate block address)
            assert cfg.hord and NP == 4
            e(f"  {p}ua = and %mp, %one")
            e(f"  {p}k10 = mova.i32 10")
            e(f"  {p}ub = lshl {p}ua, {p}k10")
            e(f"  {p}km1 = mova.i32 -1")
            e(f"  {p}uc = lshl %mp, {p}km1")
            e(f"  {p}k8 = mova.i32 8")
            e(f"  {p}ud = lshl {p}uc, {p}k8")
            e(f"  {p}ue = add.rr {p}ub, {p}ud")
            e(f"  {p}kr = mov.i32 {512 + 512 * DFR_T.index(t)}")
            e(f"  {p}uf = add.rr %dss, {p}kr")
        elif q3:
            e(f"  {p}ss = mov.address-to-scalar {q3[0]}")
            e(f"  {p}os = mov.address-to-scalar {q3[1]}")
        else:
            e(f"  {p}ds = mov.address-to-scalar {d}")
            e(f"  {p}k0 = mova.i32 0")
            e(f"  {p}nseg = mova.i32 {segs}")
            e(f"  {p}kslot = mov.i32 {SW_SLOT}")
            e(f"  low.br ^swl{t}({p}k0: {ER})")
            e(f"^swl{t}({p}k: {ER}):")
            e(f"  {p}more = lt {p}k, {p}nseg")
            e(f"  low.cond_br {p}more, ^swb{t}, ^swx{t} : {ER}")
            e(f"^swb{t}:")
            e(f"  {p}so = mul {p}k, {p}kslot")
            e(f"  {p}ss = add.rr {p}ds, {p}so")
            if pair:
                pair_swap(e, pair, f"{p}ss", f"w{t}")
        e(f"  {p}c = mova.i32 60")

        kn, kmemo, kch = [0], {}, [None]   # kmemo: a chain's constants, made once (per chain: sharing them
        # across the body's four chains pins too many accumulators)

        def kbf(x):                                         # a bf16 constant in 32 lanes
            return kbits(bf16_bits(x))

        def kbits(b):
            if ("b", b, kch[0]) in kmemo:
                return kmemo[("b", b, kch[0])]
            kn[0] += 1
            if b in silu_kvals():   # loaded from the table's tail (a broadcast would be rematerialized at every use)
                if ("p", kch[0]) not in kmemo:
                    e(f"  {p}kvp{kn[0]} = mov.scalar-to-address {p}cts")
                    kmemo[("p", kch[0])] = f"{p}kvp{kn[0]}"
                e(f"  {p}kv{kn[0]} = vlda.512.i8x64 {kmemo[('p', kch[0])]}, {64 * silu_kvals().index(b) - 512}")
            else:
                e(f"  {p}kc{kn[0]} = mov.i32 {b}")
                e(f"  {p}kv{kn[0]} = vbcst.16 {p}kc{kn[0]}")
            kmemo[("b", b, kch[0])] = f"{p}kv{kn[0]}"
            return f"{p}kv{kn[0]}"

        def kacc(x):                                        # an f32 constant (exact in bf16) in 32 accumulator lanes
            if ("a", x, kch[0]) in kmemo:
                return kmemo[("a", x, kch[0])]
            kn[0] += 1
            nm, one_, x_ = f"{p}ka{kn[0]}", kbf(1.0), kbf(x)
            e(f"  {nm} = vmul.bf16x32 {one_}, {x_}, {p}c")
            kmemo[("a", x, kch[0])] = nm
            return nm
        def weave(*bodies):
            """Emits the bodies' instructions round-robin: independent chains (their own SSA names, shared
            constants made at first use) whose latencies the locked order would otherwise expose one after the other."""
            segs, kpre = [], (f"  {p}kc", f"  {p}kv", f"  {p}ka")
            for b in bodies:
                s0 = len(L)
                b()
                steps, kdefs = [], []   # a shared constant rides with the first step that uses it
                for x in L[s0:]:
                    if x.startswith(kpre):
                        kdefs.append(x)
                    else:
                        steps.append(kdefs + [x])
                        kdefs = []
                del L[s0:]
                segs.append(steps)
            for i in range(max(map(len, segs))):   # same program: chain 0's step i (with its constants) leads
                for sg in segs:
                    if i < len(sg):
                        L.extend(sg[i])
        e(f"  {p}sh2 = mov.shift 2")
        e(f"  {p}m24 = mova.i32 24")                        # vshuffle: each 8-byte piece's first value
        e(f"  {p}km2 = mova.i32 -2")
        sp_, b0_ = silu_pieces(), 0
        for k_, (nb_, _) in enumerate(sp_):   # (each piece: its address / 4 less its first index, in 16 bits)
            x_ = f"{p}tla{k_ or ''}"
            e(f"  {x_} = mov.local-address @{silu_sym(k_)}")
            e(f"  {x_}0 = mov.address-to-scalar {x_}")
            e(f"  {p}tq4{k_ or ''} = lshl {x_}0, {p}km2")
            e(f"  {p}tqk{k_ or ''} = mov.i32 {SL_LO + 16 * b0_}")
            e(f"  {p}tq{k_ or ''} = sub {p}tq4{k_ or ''}, {p}tqk{k_ or ''}")   # (the address - 4 SL_LO) / 4: index = |g|'s bits
            b0_ += nb_
        e(f"  {p}ctk = mov.i32 {silu_kofs() + 512}")
        if len(sp_) > 1:   # (pieces: the vectors in @silu_k)
            e(f"  {p}tlk = mov.local-address @silu_k")
            e(f"  {p}tlk0 = mov.address-to-scalar {p}tlk")
        e(f"  {p}cts = add.rr {p + 'tlk0' if len(sp_) > 1 else p + 'tla0'}, {p}ctk")   # SL_K's vectors (+ 512)
        if unit:
            fifo = unit
        else:
            e(f"  {p}sp = mov.scalar-to-address {p}ss")
            e(f"  {p}sl = vlda.store-fifo.low512 {p}sp, 0")
            e(f"  {p}f0 = vlda.store-fifo.high512 {p}sp, {p}sl, 64")
            e(f"  {p}op = copy {p}sp : reg<aie2p.ep> -> reg<aie2p.mpfs>")
            e(f"  {p}ps = mova.fifo.store.position 0")
            fifo = (f"{p}f0", f"{p}op", f"{p}ps")
        ptrn = [0]

        def ptr(sub):                                       # this block's sub-tile: offsets -512 .. 448 around its middle
            ptrn[0] += 1                                    # short-lived (the tail's ep budget is 8)
            q_ = f"{p}pp{ptrn[0]}"
            e(f"  {q_} = mov.scalar-to-address {sub}")
            return q_
        nf = 0
        MF, MP_, MS = "reg<aie2p.mstfifo>", "reg<aie2p.mpfs>", "reg<aie2p.mr26_fifo_st>"

        def jloop(lo, hi, kind, tag, fifo, one=None):
            """j over the gate slabs lo .. hi - 1, i over their two 8-row halves, unrolled into one schedule
            (lsched); kind "sw": g and u from this C, h pushed. Returns the exit's store fifo.
            one: (j, i) registers, a single push (r 0).
            """
            q, T = p + tag, f"{t}{tag}"
            s0 = len(L)
            if one:   # (j, i) as the gate block's address: one[0] + one[1]
                e(f"  {q}gib = add.rr {one[0]}, {one[1]}")
                e(f"  {q}kup = mov.i32 {1024 * NP // 2}")
                e(f"  {q}uib = add.rr {q}gib, {q}kup")
                fifo = jpush(q, T, kind, fifo, [0])
            jr = os.environ.get("GN_QSWJROLL", "1")   # ("auto": unrolled only in IQ4_XS gate / up images, program memory)
            if not one and (cfg.hord or not q3) and hi - lo > 1 and (
                    jr == "1" or (jr == "auto" and not (cfg.fmt == "IQ4_XS" and cfg.ufmt in ("", "IQ4_XS")))):
                # (program memory) j a runtime loop: the gate block's offset is linear in j (hord: 256 j, else
                # 1024 j; q3 with hord: half i's area, 256 j on); the counter steps in the header, so no
                # hardware-loop candidate (LOOM_EXP_HWLOOP_TRIP)
                assert not cfg.hord or (lo, hi) == (0, 2)
                r4 = qsw_roll4(cfg)   # (one half an iteration: u = 2 j + i)
                e(f"  {q}jk0 = mova.i32 {(2 if r4 else 1) * lo}")
                e(f"  {q}jkn = mova.i32 {(2 if r4 else 1) * hi}")
                e(f"  {q}jks = mov.i32 {256 if cfg.hord else 1024}")
                if r4:   # (half i's offset: q3 the odd area's, hord 1024, else 256)
                    e(f"  {q}jk1 = mova.i32 1")
                    e(f"  {q}jkm1 = mova.i32 -1")
                    if q3:
                        e(f"  {q}jks2 = sub {p}os, {p}ss")
                    else:
                        e(f"  {q}jks2 = mov.i32 {1024 if cfg.hord else 256}")
                e(f"  low.br ^jl{T}({q}jk0: {ER}, {fifo[0]}: {MF}, {fifo[1]}: {MP_}, {fifo[2]}: {MS})")
                e(f"^jl{T}({q}jv: {ER}, {q}jf: {MF}, {q}jp: {MP_}, {q}jq: {MS}):")
                e(f"  {q}jm = lt {q}jv, {q}jkn")
                e(f"  {q}jx = add {q}jv, 1")
                e(f"  low.cond_br {q}jm, ^jb{T}, ^je{T} : {ER}")
                e(f"^jb{T}:")
                s1 = len(L)
                fifo = (f"{q}jf", f"{q}jp", f"{q}jq")
                if r4:
                    e(f"  {q}ju = lshl {q}jv, {q}jkm1")
                    e(f"  {q}ji = and {q}jv, {q}jk1")
                    e(f"  {q}jo = mul {q}ju, {q}jks")
                    e(f"  {q}jo2 = mul {q}ji, {q}jks2")
                    e(f"  {q}jg0 = add.rr {p}ss, {q}jo")
                    qq = f"{q}u"
                    e(f"  {qq}jg = add.rr {q}jg0, {q}jo2")
                    go, du = 512, 1024 if q3 else 1024 * NP // 2
                    e(f"  {qq}gk = mov.i32 {go}")
                    e(f"  {qq}gib = add.rr {qq}jg, {qq}gk")
                    e(f"  {qq}uk = mov.i32 {go + du}")
                    e(f"  {qq}uib = add.rr {qq}jg, {qq}uk")
                    fifo = jpush(qq, T, kind, fifo, range(2))
                else:
                    e(f"  {q}jo = mul {q}jv, {q}jks")
                for i in range(0 if r4 else 2):
                    qq = f"{q}u{i}"
                    go, du = (1024 * i + 512) if cfg.hord else (256 * i + 512), 1024 * NP // 2
                    gb = f"{p}ss"
                    if q3:
                        go, du, gb = 512, 1024, f"{p}os" if i else f"{p}ss"
                    e(f"  {qq}jg = add.rr {gb}, {q}jo")
                    e(f"  {qq}gk = mov.i32 {go}")
                    e(f"  {qq}gib = add.rr {qq}jg, {qq}gk")
                    e(f"  {qq}uk = mov.i32 {go + du}")
                    e(f"  {qq}uib = add.rr {qq}jg, {qq}uk")
                    fifo = jpush(qq, T, kind, fifo, range(2))
                L[s1:] = lsched(L[s1:], 24)   # (12 or 40: more code)
                e(f"  low.br ^jl{T}({q}jx: {ER}, {fifo[0]}: {MF}, {fifo[1]}: {MP_}, {fifo[2]}: {MS})")
                e(f"^je{T}:")
                return (f"{q}jf", f"{q}jp", f"{q}jq")
            for j in range(lo, hi) if not one else ():
                for i in range(2):
                    # gate rows: slab j, half i (sub-tiles at 256 i, 512 + 256 i); up: NP / 2 slabs on. The record
                    # takes the 8-feature blocks 2 j + i in loop order; P4 (and Q4K's 4 gate k-blocks of one
                    # format, its slabs 32 rows apart): per two slabs blocks 0, 2, 1, 3, so (j, i) stand for slab
                    # (j & ~1) | i, half j & 1
                    qq = f"{q}u{j}{i}"
                    jj, ii = j, i
                    if cfg.hord:
                        jj, ii = (j & ~1) + i, j & 1
                    go, gb, du = 1024 * jj + 256 * ii + 512, f"{p}ss", 1024 * NP // 2
                    if q3:   # (sub-tile jj at the even (jj 0) or odd area's start, its up sub-tile jj + 2 a KB on)
                        go, gb, du = 256 * ii + 512, f"{p}os" if jj else f"{p}ss", 1024
                    e(f"  {qq}gk = mov.i32 {go}")
                    e(f"  {qq}gib = add.rr {gb}, {qq}gk")
                    e(f"  {qq}uk = mov.i32 {go + du}")
                    e(f"  {qq}uib = add.rr {gb}, {qq}uk")
                    fifo = jpush(qq, T, kind, fifo, range(2))
            L[s0:] = lsched(L[s0:], 24)
            return fifo

        def jpush(q, T, kind, fifo, rs):
            nonlocal nf
            kmemo.clear()   # (constants made before the body stay in their own blocks; shared across the rolled
            # loop's halves they spill: +600 B)
            for r in rs:                                    # token half: chain row bit
                n = f"{q}b{r}"

                def load(sub, tag):
                    pp = ptr(sub)
                    q4 = []
                    for q_ in range(4):
                        x = f"{n}{tag}l{q_}"
                        e(f"  {x} = vlda.acc {pp}, {512 * r + 64 * q_ - 512}")
                        q4.append(x)
                    return q4

                # vmul / vmac.bf16x32: 32 f32 lanes in the first two units of the x4 accumulator
                def lo32(v, tag):
                    q0, q1, r2 = f"{n}{tag}0", f"{n}{tag}1", f"{n}{tag}2"
                    e(f"  {q0} = slice {v}[0] : {M4} -> {M1}")
                    e(f"  {q1} = slice {v}[1] : {M4} -> {M1}")
                    e(f"  {r2} = concat({q0}, {q1}) : ({M1}, {M1}) -> {M2}")
                    return q0, q1, r2

                def lut_t(h, fill=()):
                    """t = sigmoid(-|g|) (bf16) of {n}gb{h} from the table (silu_rodata), gathered at |g|'s bits
                    (clamped). fill: independent instructions (emitters) spread between the gathers, whose load slot
                    leaves the vector units free. Returns t."""
                    g_ = f"{n}gb{h}"
                    e(f"  {n}la{h} = vband {g_}, {kbits(0x7FFF)}")                 # |g|
                    # clamped as integers (a float compare lets NaN through: an address past the table)
                    e(f"  {n}lc0{h}, {n}lc0m{h} = vmax_lt.s16x32 {n}la{h}, {kbf(2.0 ** -8)}")
                    e(f"  {n}lc{h}, {n}lcm{h} = vmin_ge.s16x32 {n}lc0{h}, {kbf(128.0)}")
                    # + the table's address / 4 less SL_LO in 16 bits (the table ends below 2^18), then x 4 as i32:
                    # the gathers' byte addresses
                    sp_ = silu_pieces()
                    for k_ in range(len(sp_)):
                        if ("q", k_, kch[0]) not in kmemo:
                            kn[0] += 1
                            e(f"  {p}kv{kn[0]} = vbcst.16 {p}tq{k_ or ''}")
                            kmemo[("q", k_, kch[0])] = f"{p}kv{kn[0]}"
                    tb_, b0_ = kmemo[("q", 0, kch[0])], 0
                    for k_ in range(1, len(sp_)):   # (GN_QSILU pieces: index >= the piece's first takes its base)
                        b0_ += sp_[k_ - 1][0]
                        e(f"  {n}lp{h}{k_} = vlt.s16x32.el.low32 {n}lc{h}, {kbits(SL_LO + 16 * b0_)}")
                        e(f"  {n}lt{h}s{k_} = vsel.16.mask64 {kmemo[('q', k_, kch[0])]}, {tb_}, {n}lp{h}{k_}")
                        tb_ = f"{n}lt{h}s{k_}"
                    e(f"  {n}lb{h} = vadd.16 {n}lc{h}, {tb_}")
                    e(f"  {n}lu{h} = vups.2x.x-to-c.unsigned {n}lb{h}, {p}sh2")
                    fill, tv = list(fill), []

                    def addr(q):   # 16 lanes' 8 lane addresses x 2
                        e(f"  {n}lq{h}{q} = slice {n}lu{h}[{q}] : {M2} -> {M1}")
                        e(f"  {n}ld{h}{q} = vmov.accumulator512.to.vector512 {n}lq{h}{q}")
                    addr(0)
                    for q in range(2):   # 4 gathers, the next half's addresses (gather latency), the 16 values
                        pr = []
                        for w in range(2):
                            x_ = f"{n}l{h}{q}{w}"
                            e(f"  {x_}a = slice {n}ld{h}{q}[{w}] : {V2} -> {V1}")
                            e(f"  {x_}l = vldb.4x16.lo {x_}a")
                            if fill:
                                fill.pop(0)()
                            e(f"  {x_}h = vldb.4x16.hi {x_}a")
                            if fill:
                                fill.pop(0)()
                        if q == 0:
                            addr(1)
                        for w in range(2):
                            x_ = f"{n}l{h}{q}{w}"
                            e(f"  {x_}c = concat({x_}l, {x_}h) : ({V1}, {V1}) -> {V2}")
                            pr.append(f"{x_}c")
                        e(f"  {n}ls{h}{q} = vshuffle {pr[0]}, {pr[1]}, {p}m24")   # each piece's first value
                        e(f"  {n}lt{h}{q} = slice {n}ls{h}{q}[0] : {V2} -> {V1}")
                        tv.append(f"{n}lt{h}{q}")
                    for f_ in fill:
                        f_()
                    e(f"  {n}lt{h} = concat({tv[0]}, {tv[1]}) : ({V1}, {V1}) -> {V2}")
                    return f"{n}lt{h}"

                def silu(h):
                    """silu(g) as bf16 from {n}gb{h}: silu = g t (g < 0), g - g t (g >= 0), t = lut_t: one
                    multiply-accumulate. Exhaustive over bf16 g on the core: max 1.59 ulp, mean 0.075; NaN, inf -> NaN
                    (yah-scratch/npu/silu/lprobe.py)."""
                    g_ = f"{n}gb{h}"
                    t_ = lut_t(h)
                    e(f"  {n}ln{h} = vbor {t_}, {kbf(-0.0)}")                     # -t
                    e(f"  {n}lg{h} = vlt.s16x32.el.low32 {g_}, {kbf(0.0)}")      # g < 0 (sign bit)
                    e(f"  {n}lm{h} = vsel.16.mask64 {kbf(1.0)}, {kbf(0.0)}, {n}lg{h}")
                    e(f"  {n}lst{h} = vsel.16.mask64 {n}ln{h}, {t_}, {n}lg{h}")
                    e(f"  {n}lp{h} = vmul.bf16x32 {g_}, {n}lm{h}, {p}c")             # g [g >= 0]
                    e(f"  {n}si{h} = vmac.bf16x32 {n}lp{h}, {g_}, {n}lst{h}, {p}c")  # silu(g)
                    e(f"  {n}sib{h} = vconv.bf16.fp32 {lo32(f'{n}si{h}', f'siq{h}')[2]}")
                    return f"{n}sib{h}"

                def swiglu_h(h):
                    """{n}h{h} = silu(g) u = max(g, 0) u + t (-|g| u), t = lut_t: both products wait on nothing but
                    g and u (they fill the gathers' slots), then one multiply-accumulate after the table. -|g| u
                    rounds to bf16 (the t term's operand); max(g, 0) u is exact."""
                    g_, u_ = f"{n}gb{h}", f"{n}ub{h}"

                    def f1():
                        e(f"  {n}fg{h}, {n}fgm{h} = vmax_lt.bf16 {g_}, {kbf(0.0)}")   # max(g, 0)

                    def f2():
                        e(f"  {n}fn{h} = vbor {g_}, {kbf(-0.0)}")                      # -|g|

                    def f3():
                        e(f"  {n}fp{h} = vmul.bf16x32 {n}fg{h}, {u_}, {p}c")

                    def f4():
                        e(f"  {n}fq{h} = vmul.bf16x32 {n}fn{h}, {u_}, {p}c")

                    def f5():
                        e(f"  {n}fqb{h} = vconv.bf16.fp32 {lo32(f'{n}fq{h}', f'fqq{h}')[2]}")
                    t_ = lut_t(h, fill=(f1, f3, f2, f4, f5))
                    e(f"  {n}h{h} = vmac.bf16x32 {n}fp{h}, {t_}, {n}fqb{h}, {p}c")
                g4, u4 = load(f"{q}gib", "g"), load(f"{q}uib", "u")
                hs = {}

                def hbody(h):
                    kch[0] = r   # the woven pair shares its constants
                    u2 = f"{n}u2{h}"
                    e(f"  {u2} = concat({u4[2 * h]}, {u4[2 * h + 1]}) : ({M1}, {M1}) -> {M2}")
                    g2 = f"{n}g2{h}"
                    e(f"  {g2} = concat({g4[2 * h]}, {g4[2 * h + 1]}) : ({M1}, {M1}) -> {M2}")
                    e(f"  {n}gb{h} = vconv.bf16.fp32 {g2}")
                    e(f"  {n}ub{h} = vconv.bf16.fp32 {u2}")
                    if os.environ.get("GN_QSWH") == "one":   # (diag) h = 1
                        e(f"  {n}h{h} = vmul.bf16x32 {kbf(1.0)}, {kbf(1.0)}, {p}c")
                    elif os.environ.get("GN_QSWH") in ("g", "u"):   # (diag) h = g or u (x 1.0)
                        e(f"  {n}h{h} = vmul.bf16x32 {n}{os.environ['GN_QSWH']}b{h}, {kbf(1.0)}, {p}c")
                    else:
                        swiglu_h(h)
                    hs[h] = lo32(f"{n}h{h}", f"hq{h}")[:2]
                if os.environ.get("GN_QSWH") == "raw":   # (diag) h = the loaded g sums, no conversion
                    e(f"  {n}H = concat({g4[0]}, {g4[1]}, {g4[2]}, {g4[3]}) : ({M1}, {M1}, {M1}, {M1}) -> {M4}")
                else:
                    weave(lambda: hbody(0), lambda: hbody(1))
                    hs = hs[0] + hs[1]
                    e(f"  {n}H = concat({hs[0]}, {hs[1]}, {hs[2]}, {hs[3]}) : ({M1}, {M1}, {M1}, {M1}) -> {M4}")
                nf += 1
                nxt = (f"{p}xf{nf}", f"{p}xp{nf}", f"{p}xq{nf}")
                e(f"  {nxt[0]}, {nxt[1]}, {nxt[2]} = vst.push.bfp16ebs8.from.fp32 {fifo[0]}, {n}H, {fifo[1]}, {fifo[2]}")
                fifo = nxt
            return fifo

        H2 = NP // 2
        if unit and cfg.dfi:
            fifo = jloop(0, H2, "sw", "", fifo, one=(f"{p}uf", f"{p}ue"))
            e(f"  %duB{t} = mov.scalar-to-address %dub{t}")
            e(f"  %duL{t} = mov.scalar-to-address %dul{t}")
            return f"%duB{t}", f"%duL{t}", fifo
        elif unit:
            fifo = jloop(0, H2, "sw", "", fifo, one=(f"{p}uf", f"{p}ue"))
            e(f"  low.br ^dj{t}({fifo[0]}: {MF}, {fifo[1]}: {MP_}, {fifo[2]}: {MS})")
            e(f"^dj{t}(%dj{t}f: {MF}, %dj{t}o: {MP_}, %dj{t}q: {MS}):")
            e(f"  %duB{t} = mov.scalar-to-address %dub{t}")
            e(f"  %duL{t} = mov.scalar-to-address %dul{t}")
            return f"%duB{t}", f"%duL{t}", (f"%dj{t}f", f"%dj{t}o", f"%dj{t}q")
        else:
            fifo = jloop(0, H2, "sw", "", fifo)
        # the flush writes whole 64-byte lines only, and drops the last one when the stream ends on a 128-byte
        # boundary: pad with zero fragments (72 B each) until it ends at 64 mod 128 (the slot's bytes below
        # the pads have all been read by now)
        e(f"  {p}zc = acc.clear.f32x64")
        frags = 2 * NP
        for z in range(next(n for n in range(16) if 72 * (frags + n) % 128 == 64)):
            nf += 1
            nxt = (f"{p}xf{nf}", f"{p}xp{nf}", f"{p}xq{nf}")
            e(f"  {nxt[0]}, {nxt[1]}, {nxt[2]} = vst.push.bfp16ebs8.from.fp32 {fifo[0]}, {p}zc, {fifo[1]}, {fifo[2]}")
            fifo = nxt
        e(f"  {p}yf, {p}yp, {p}yq = vst.flush.512 {fifo[0]}, {fifo[1]}, {fifo[2]}")
        if q3:
            return None
        e(f"  {p}k1 = add.rr {p}k, %one")
        e(f"  low.br ^swl{t}({p}k1: {ER})")
        e(f"^swx{t}:")
        e(f"  {p}rb = mov.i32 {back + 512 + SW_SLOT * (segs - 1)}")
        e(f"  {p}ra = add.rr {p}ds, {p}rb")
        e(f"  %cvr{t} = mov.scalar-to-address {p}ra")
        e(f"  %cvq{t} = copy %cvr{t} : reg<aie2p.ep> -> reg<aie2p.ep>")
        if carry:   # a fresh (unused) fifo for the join: the zero buffer's
            e(f"  {p}czp = mov.scalar-to-address %zsc0")
            e(f"  {p}czl = vlda.store-fifo.low512 {p}czp, 0")
            e(f"  {p}czf = vlda.store-fifo.high512 {p}czp, {p}czl, 64")
            e(f"  {p}czo = copy {p}czp : reg<aie2p.ep> -> reg<aie2p.mpfs>")
            e(f"  {p}czq = mova.fifo.store.position 0")
            ca = "".join(f", {x}: {ty}" for x, ty in zip((f"{p}czf", f"{p}czo", f"{p}czq"), DFT.split(", ")))
        e(f"  low.br ^cj{t}(%cvr{t}: reg<aie2p.ep>, %cvq{t}: reg<aie2p.ep>{ca})")
        cf = (f"%cj{t}f", f"%cj{t}o", f"%cj{t}q")
        e(f"^cj{t}(%cj{t}p: reg<aie2p.ep>, %cj{t}l: reg<aie2p.ep>"
          + ("".join(f", {x}: {ty}" for x, ty in zip(cf, DFT.split(", "))) if carry else "") + "):")
        if carry:   # (r26 holds the deferred fifo's position)
            e(f"  low.cond_br {cnd}, ^cr{t}, ^ck{t} : reg<aie2p.er>")
            e(f"^cr{t}:")
            for j in range(segs):
                e("  rel %one, 2")
            e(f"  low.br ^ck{t}")
            e(f"^ck{t}:")
        for j in range(segs if not carry else 0):
            e(f"  %cqr{t}_{j} = copy {cnd} : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
            e(f"  rel.cond %one, %cqr{t}_{j}, 2")
        if carry:
            return f"%cj{t}p", f"%cj{t}l", cf
        return f"%cj{t}p", f"%cj{t}l"
    for t in range(NT):
        for k in range(QE2 if t > 0 and rot(t - 1) or qrow and t == 0 else 0, ks - QE if rot(t) else ks):
            fq = -1 if qcore else (k if tail else (k - 1 if mid else -1))   # incoming quarter added at this k step
            if fq == 0:
                e(f"  %q{t}_{k} = copy %qsel0 : reg<aie2p.er> -> reg<aie2p.mr31_scd>")
                sel = f"%q{t}_{k}"
            for ci, c in enumerate(ORDER_C):
                g = (t * ks + k) * 4 + ci
                a_op, w_op = f"%s{t}a{k}_{c >> 1}", f"%s{t}w{k}_{c & 1}"
                if 0 <= fq < 4:
                    # the k step's last fused MMA advances r31 to the next quarter
                    if ci == 3 and fq < 3:
                        e(f"  %m{t}_{k}_{c}, %q{t}_{k}n = maddmac.bfp.ex-ex.scd.increment {cur[c]}, {sel}, {a_op}, {w_op}, %conf")
                        sel = f"%q{t}_{k}n"
                    else:
                        e(f"  %m{t}_{k}_{c} = maddmac.bfp.ex-ex.scd {cur[c]}, {sel}, {a_op}, {w_op}, %conf")
                elif not tail and k == 0:
                    e(f"  %m{t}_{k}_{c} = mmul.bfp16ebs8.m8n8k8 {a_op}, {w_op}, %conf")
                else:
                    e(f"  %m{t}_{k}_{c} = mma.bfp16ebs8.m8n8k8 {cur[c]}, {a_op}, {w_op}, %conf")
                cur[c] = f"%m{t}_{k}_{c}"
                for line in slots.get(g, []):
                    e(line)
        nxt = t + 1 < NT
        if not tail:
            emit_writes(t)
            continue
        if rot(t):
            emit_window(t)
            continue
        # store sub-tile t, load the next sub-tile's C (or zeros on the first pass)
        row_end = (t + 1) % NP == 0
        if row_end and nxt:
            e(f"  %cqa{t + 1} = copy %first : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
            e(f"  acq.cond %am1, %cqa{t + 1}, 2")
        if nxt:   # loads go through pl, so pc steps after the stores (one fewer live pointer)
            e(f"  %plc{t + 1} = copy {pl[t]} : reg<aie2p.ep> -> reg<aie2p.ep>")
            if cfg.swiglu and row_end:
                e(f"  %pl{t + 1}g = padds.modifier %plc{t + 1}, %mls")
                e(f"  %pl{t + 1} = padds.modifier %pl{t + 1}g, %mlb")
            else:
                e(f"  %pl{t + 1} = padds.modifier %plc{t + 1}, %mls")
        late = row_end and nxt or (cfg.dfr and t in DFR_T)   # the convert (push) runs between the stores and the next C loads
        for c in range(4):
            for q in range(4):
                if nxt and not late:
                    e(f"  %l{t + 1}_{c}_{q} = vlda.acc {pl[t + 1]}, {256 * c + 64 * q - 512}")
                e(f"  %o{t}_{c}_{q} = slice {cur[c]}[{q}] : reg<aie2p.mbms x4> -> reg<aie2p.mbms>")
                e(f"  vst.acc %o{t}_{c}_{q}, {pc[t]}, {256 * c + 64 * q - 512}")
        def convert_f32(base, lp, back):
            """Last pass (cfg.ofeat): permute each (sub-tile, mh) half of the segment's f32 C in place from [nh][row][8]
            to [row][nh][8] (whole 32-byte chain rows), so the slot is [sub-tile][16 tokens][16 f32] for the strided C
            receiver (fragment-major output). A half's 16 rows are all loaded before any is stored. Returns the join's (base, lp)."""
            e(f"  low.cond_br %clast, ^cv{t}, ^nc{t} : reg<aie2p.er>")
            e(f"^nc{t}:")
            e(f"  low.br ^cj{t}({base}: reg<aie2p.ep>, {lp}: reg<aie2p.ep>)")
            e(f"^cv{t}:")
            e(f"  %cvk{t} = copy {base} : reg<aie2p.ep> -> reg<aie2p.ep>")
            d, k, rest = f"%cvk{t}", 0, back + 512 - 256   # the first half's middle
            while rest > 0:
                step = min(rest, 448)
                rest -= step
                e(f"  %cvb{t}_{k} = padda {d}, {-step}")
                d, k = f"%cvb{t}_{k}", k + 1
            for h in range(2 * NP):
                if h:
                    bump(L, f"%cvh{t}_{h}", d, 512)
                    d = f"%cvh{t}_{h}"
                # piece i = 8 nh + row (32 bytes at 32 i) belongs at 2 row + nh: a perfect shuffle, cycles of <= 4 pieces
                for cyc in SHUFFLE16_CYCLES:
                    for i in cyc:
                        e(f"  %cvx{t}_{h}_{i} = vlda.256.i32x8 {d}, {32 * i - 256}")
                    for i in cyc:
                        e(f"  vst.256.i32x8 %cvx{t}_{h}_{i}, {d}, {32 * (2 * (i % 8) + i // 8) - 256}")
            e(f"  %cvq{t} = copy {base} : reg<aie2p.ep> -> reg<aie2p.ep>")
            e(f"  low.br ^cj{t}({base}: reg<aie2p.ep>, %cvq{t}: reg<aie2p.ep>)")
            e(f"^cj{t}(%cj{t}p: reg<aie2p.ep>, %cj{t}l: reg<aie2p.ep>):")
            e(f"  %cqr{t} = copy %clast : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
            e(f"  rel.cond %one, %cqr{t}, 2")
            return f"%cj{t}p", f"%cj{t}l"
        if cfg.swiglu:
            assert not cfg.ofeat and not groups(cfg), "swiglu needs the full sum: one replay group, bf16-slot C"
            convert = convert_swiglu
        if cfg.ofeat:
            convert = convert_f32
        if cfg.dfr and t in DFR_T:
            pc[t], pl[t], dfifo = convert_swiglu(pc[t], pl[t], 0, unit=dfifo)
        if row_end and not nxt:
            if cfg.swiglu and cfg.dfr:
                pc[t], pl[t], dfifo = convert(pc[t], pl[t], 1024 * (NP - 1), segs=NT // NP, carry=dfifo)
            elif cfg.swiglu:   # the iteration's segments (mu), the earlier ones deferred to here
                pc[t], pl[t] = convert(pc[t], pl[t], 1024 * (NP - 1), segs=NT // NP)
            else:
                pc[t], pl[t] = convert(pc[t], pl[t], 1024 * (NP - 1))
        if nxt:
            e(f"  %pcc{t + 1} = copy {pc[t]} : reg<aie2p.ep> -> reg<aie2p.ep>")
            if cfg.swiglu and row_end:
                e(f"  %pc{t + 1}g = padds.modifier %pcc{t + 1}, %mcs")
                e(f"  %pc{t + 1} = padds.modifier %pc{t + 1}g, %mcb")
            else:
                e(f"  %pc{t + 1} = padds.modifier %pcc{t + 1}, %mcs")
            if late:   # from the next store pointer, so this one is dead
                if not cfg.swiglu:   # swiglu: deferred to the iteration end
                    pc[t + 1], pl[t + 1] = convert(pc[t + 1], pl[t + 1], 1024 * NP)
                for c in range(4):
                    for q in range(4):
                        e(f"  %l{t + 1}_{c}_{q} = vlda.acc {pl[t + 1]}, {256 * c + 64 * q - 512}")
            for line in held.get(t + 1, []):
                e(line)
            for c in range(4):
                e(f"  %z{t + 1}_{c} = concat(%l{t + 1}_{c}_0, %l{t + 1}_{c}_1, %l{t + 1}_{c}_2, %l{t + 1}_{c}_3) : {MBMS4}")
                cur[c] = f"%z{t + 1}_{c}"
    if qrow:   # header arguments: the body's live-ins (t 0 names); entry values: the entry's, renamed
        defs, uses = set(), set()
        for ln in L[hdr + 5:]:
            lhs, eq, rhs = ln.partition(" = ")
            (defs.update(re.findall(r"%\w+", lhs)), uses.update(re.findall(r"%\w+", rhs))) if eq else uses.update(re.findall(r"%\w+", ln))
        edefs = {n for ln in ent if " = " in ln for n in re.findall(r"%\w+", ln.partition(" = ")[0])}
        hv = {"%mp": "%zero", "%pa": "%qa0" if qa3_on() or qa4_carry(cfg) else a0, "%po": po0, "%pl": "%qodb" if q3 else "%plb", "%more": "%one",
              "%wc": "%wsb0" if qpass and cfg.passes % 2 else "%wsb"}
        live = sorted(n for n in uses - defs if n in edefs or n in hv) + ["%more"]

        def qkind(n):
            if n in ("%mp", "%more", "%wc") or re.match(r"%s0z", n):   # (%s0z*: P1K's slab-start scalar)
                return "reg<aie2p.er>"
            if n in ("%pa", "%po", "%pl"):
                return "reg<aie2p.ep>"
            if re.match(r"%m0_\d+_\d$", n):
                return "reg<aie2p.mbms x4>"
            for pat_, k_ in ((r"%s0[aw]\d", "mexa"), (r"%s0r?p", "eps"), (r"%s0r?f", "eldfiforeg"), (r"%s0r?q", "erf2")):
                if re.match(pat_, n):
                    return f"reg<aie2p.{k_}>"
            raise AssertionError(n)
        bm = {"%mp": "%mpn", "%pa": "%pan", "%po": pc[NT], "%pl": "%pqo" if q3 else pl[NT], "%more": "%nxm", "%wc": "%wcn"}
        bname = lambda n: bm.get(n) or re.sub(r"^%s0(?=[a-z])", f"%s{NT}", re.sub(r"^%m0_", f"%m{NT}_", n))
        for n in live:
            assert bname(n) in defs, (n, bname(n))
        e(f"  low.br ^outer({', '.join(f'{bname(n)}: {qkind(n)}' for n in live)})")
        ren = {n: n + "_e" for n in edefs}
        ren.update({"%pa": "%qa0" if qa3_on() or qa4_carry(cfg) else a0, "%pl": "%plb", "%fa": "%fa0", "%fw": "%fw0", "%po": po0, "%mp": "%zero"})
        ent2 = [re.sub(r"%\w+", lambda m: ren.get(m.group(0), m.group(0)), ln) for ln in ent]
        ev_ = lambda n: ren[n] if n in edefs else hv[n]
        assert L[hdr + 2].strip() == "%omore = lt %mp, %nmp" and L[hdr + 4] == "^obody:"
        L[hdr:hdr + 4] = ent2 + [
            f"  low.br ^outer({', '.join(f'{ev_(n)}: {qkind(n)}' for n in live)})",
            f"^outer({', '.join(f'{n}: {qkind(n)}' for n in live)}):",
            "  low.cond_br %more, ^obody, ^exit : reg<aie2p.er>"]
    else:
        if not seam:
            e("  rel %one, 0")
            a_next(e)
        if tail:
            e(f"  %pcl = copy {pc[-1]} : reg<aie2p.ep> -> reg<aie2p.ep>")
            if cfg.swiglu:
                e("  %pong = padds.modifier %pcl, %mcs")
                e("  %pon = padds.modifier %pong, %mcb")
            elif qbank:   # the next row's slot (QBANK_C): +12288 after an even row, -16384 after an odd one
                # (GN_QBANK=2, each row in the other A slot's bank: -20480 after an even row, +16384 after an odd one)
                v2 = os.environ.get("GN_QBANK") == "2"
                e(f"  %kcd0a = mova.i32 {40 if v2 else 24}")
                e("  %kcd0 = mul %k512, %kcd0a")   # 20480 / 12288
                e(f"  %kcd1a = mova.i32 {72 if v2 else 56}")
                e("  %kcd1 = mul %k512, %kcd1a")   # 36864 / 28672
                e("  %cdp = mul %par, %kcd1")
                e("  %cdel = sub %cdp, %kcd0" if v2 else "  %cdel = sub %kcd0, %cdp")
                e("  %mcd = mov.modifier %cdel")
                e("  %pong = padds.modifier %pcl, %mcs")
                e("  %pon = padds.modifier %pong, %mcd")
            else:
                e("  %pon = padds.modifier %pcl, %mcs")
            e(f"  %plx = copy {pl[-1]} : reg<aie2p.ep> -> reg<aie2p.ep>")
            if cfg.swiglu:
                e("  %plng = padds.modifier %plx, %mls")
                e("  %pln = padds.modifier %plng, %mlb")
            elif qbank:   # (the first pass's loads stay on the zero buffer: step 0)
                e("  %cdl = mul %cdel, %nf")
                e("  %mcdl = mov.modifier %cdl")
                e("  %plng = padds.modifier %plx, %mls")
                e("  %pln = padds.modifier %plng, %mcdl")
            else:
                e("  %pln = padds.modifier %plx, %mls")
        else:
            e("  %pon = copy %po : reg<aie2p.er> -> reg<aie2p.er>")
        if seam:
            e(f"  low.br ^outer(%mpn: reg<aie2p.er>, %pan: {pat}, %pon: {pct}, %nxm: reg<aie2p.er>{cty(ren('n0'))})")
        else:
            e("  %mpn = add.rr %mp, %one")
            pln = ", %pln: reg<aie2p.ep>" if tail else ""
            e(f"  low.br ^outer(%mpn: reg<aie2p.er>, %pan: {pat}, %pon: {pct}, {fifo['a']}: reg<aie2p.eldfiforeg>, {fifo['w']}: reg<aie2p.eldfiforeg>{pln}{dfa(dfifo)})")
    e("^exit:")
    if qpeel_on() or qunr_on():
        assert q3 and qrow and qpass and NT == NP and not cfg.dfr
        i0 = L.index("^obody:", hdr)
        hl = next(ln for ln in L[hdr:i0] if ln.startswith("^outer("))
        hargs = [a.split(": ", 1) for a in hl[len("^outer("):-2].split(", ")]
        back = L[-2]
        assert L[-1] == "^exit:" and back.startswith("  low.br ^outer(") and back.endswith(")"), (back, L[-1])
        body = L[i0 + 1:-2]
        if qunr_on():   # (the peel below copies one row: body / back as captured here)
            assert MP * (cfg.passes - (1 if qpeel_on() else 0)) % 2 == 0
            bvals = [a.split(": ", 1)[0] for a in back[len("  low.br ^outer("):-1].split(", ")]
            assert len(bvals) == len(hargs)
            bd = {n for ln in body if " = " in ln for n in re.findall(r"%\w+", ln.partition(" = ")[0])}
            uren = {n: n + "_U" for n in bd}
            uren.update({a: v for (a, _), v in zip(hargs, bvals)})
            sub_u = lambda ln: re.sub(r"%\w+", lambda m: uren.get(m.group(0), m.group(0)), ln)
            back_u = "  low.br ^outer(" + ", ".join(f"{v}_U: {ty}" if v in bd else f"{uren.get(v, v)}: {ty}"
                                                      for v, (_, ty) in zip(bvals, hargs)) + ")"
            L[i0 + 1:-1] = body + [sub_u(ln) for ln in body] + [back_u]
    if qpeel_on():
        bdefs = {n for ln in body if " = " in ln for n in re.findall(r"%\w+", ln.partition(" = ")[0])}
        pren = {n: n + "_P" for n in bdefs | {a for a, _ in hargs}}
        sub_ = lambda ln: re.sub(r"%\w+", lambda m: pren.get(m.group(0), m.group(0)), ln)
        e("  %ppk = mova.i32 256")
        e("  %ppm = mov.modifier %ppk")
        if qaltp_on():
            e("  %ppc = copy %o : reg<aie2p.ep> -> reg<aie2p.ep>")
            e("  %pp0 = padds.modifier %ppc, %ppm")
        elif qalt_on():   # the block's first C slot s; the next block's (s + MP) & 1; c: (s + MP (passes - 1)) & 1
            cd_ = qalt_cslots()[1] - qalt_cslots()[0]
            e("  %cnq = storage_address %cst : low.storage<private> -> reg<aie2p.ep>")
            e("  %qos = lda %cnq, 24")
            e("  %qok1 = mova.i32 1")
            e("  %qosa = add.rr %qos, %qok1")
            e("  %qosn = and %qosa, %qok1")
            e("  st %qosn, %cnq, 24")
            e(f"  %qocd = mov.i32 {cd_}")
            e("  %qoo = mul %qos, %qocd")
            e("  %qoom = mov.modifier %qoo")
            e(f"  %qock = mova.i32 {MP * (cfg.passes - 1) % 2}")
            e("  %qoca = add.rr %qos, %qock")
            e("  %qoc = and %qoca, %qok1")
            e("  %ppc = copy %o : reg<aie2p.ep> -> reg<aie2p.ep>")
            e("  %pp0a = padds.modifier %ppc, %ppm")
            e("  %pp0 = padds.modifier %pp0a, %qoom")
        else:
            e("  %ppc = copy %o : reg<aie2p.ep> -> reg<aie2p.ep>")
            e("  %pp0 = padds.modifier %ppc, %ppm")   # (row 0's slot + 256: the stores' offsets -256 .. 224)
        e("  %pqm = mov.modifier %k512")
        e(f"  low.br ^pouter({', '.join(f'{a}: {ty}' for a, ty in hargs)}, %pp0: reg<aie2p.ep>)")
        e(f"^pouter({', '.join(f'{a}_P: {ty}' for a, ty in hargs)}, %psl_P: reg<aie2p.ep>):")
        e("  low.cond_br %more_P, ^pbody, ^pexit : reg<aie2p.er>")
        e("^pbody:")
        if qalt_on() and not qaltp_on():
            e("  acq %am1, 2")
        q = "%psl_P"
        nst, made = 0, {0}
        for ln in body:
            m = re.match(r"  vst\.acc (%o(\d+)_(\d)_(\d)), (%\w+), (-?\d+)$", ln)
            if m:
                t, c, qq = int(m.group(2)), int(m.group(3)), int(m.group(4))
                assert 0 <= t < NP and int(m.group(6)) == 256 * c + 64 * qq - 512, ln
                if t not in made:   # the sub-tile's bf16 pointer (512 on from the previous), at its first store
                    assert t - 1 in made
                    made.add(t)
                    e(f"  %pqc{t}_P = copy {q} : reg<aie2p.ep> -> reg<aie2p.ep>")
                    e(f"  %pq{t}_P = padds.modifier %pqc{t}_P, %pqm")
                    q = f"%pq{t}_P"
                e(f"  vst.convert.f32x16.to.bf16x16 {pren[m.group(1)]}, {'%psl_P' if t == 0 else f'%pq{t}_P'}, "
                  f"{128 * c + 32 * qq - 256}")
                nst += 1
            else:
                e(sub_(ln))
        assert nst == 16 * NP, nst
        if QCYC:   # (diag) the cycles since the first firing's start into the row's first word
            e("  %ptw = mov.cycles")
            e("  %ptl = slice %ptw[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
            e("  %pta = storage_address %tcs : low.storage<private> -> reg<aie2p.ep>")
            e("  %pts = lda %pta, 0")
            e("  %ptp = sub %ptl, %pts")
            e("  %ptk = mova.i32 -256")
            e("  %ptm = mov.modifier %ptk")
            e("  %ptc = copy %psl_P : reg<aie2p.ep> -> reg<aie2p.ep>")
            e("  %ptq = padds.modifier %ptc, %ptm")
            e("  st %ptp, %ptq, 0")
            if QFCYC and QF.get("cfg") is not None and fills is True:   # (fill cycles: words 1, 2)
                e("  %qfaP = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
                e("  %qfeP = lda %qfaP, 4")
                e("  %qffP = lda %qfaP, 12")
                e("  st %qfeP, %ptq, 4")
                e("  st %qffP, %ptq, 8")
                e("  %qfrP = lda %qfaP, 24")
                e("  st %qfrP, %ptq, 12")
        e("  rel %one, 2")
        e(f"  %pnc = copy {q} : reg<aie2p.ep> -> reg<aie2p.ep>")
        if qaltp_on():   # the next row's slot: row r even -> O - E + 2 KB on, odd -> E - O + 3 KB on (NP 5)
            E_, O_ = qaltp_eo()
            cs_ = [int(x) for x in qpins().split(",")[0].split("@")[1].split(":")]
            de_, do_ = cs_[1] - cs_[0], cs_[2] - cs_[1]
            assert all(cs_[r + 1] - cs_[r] == (de_ if r % 2 == 0 else do_) for r in range(MP - 1)), cs_
            e("  %pqk1 = mova.i32 1")
            e(f"  %pqkc = mova.i32 {MP * (cfg.passes - 1) % 2}")
            e("  %pqa = add.rr %mp_P, %pqkc")
            e("  %pqb = and %pqa, %pqk1")
            e(f"  %pqk = mov.i32 {do_ - de_}")
            e("  %pqd = mul %pqb, %pqk")
            e(f"  %pqe0 = mov.i32 {de_ - 512 * (NT - 1)}")
            e("  %pqe = add.rr %pqe0, %pqd")
            e("  %pnm = mov.modifier %pqe")
            e("  %pnx = padds.modifier %pnc, %pnm")
        elif qalt_on():   # the other slot: + d if this row's is slot 0, else - d (cur = (mp + c) & 1)
            e("  %pqa = add.rr %mp_P, %qoc")
            e("  %pqb = and %pqa, %qok1")
            e(f"  %pqk = mov.i32 {-2 * cd_}")
            e("  %pqd = mul %pqb, %pqk")
            e(f"  %pqe0 = mov.i32 {cd_ - 512 * (NT - 1)}")
            e("  %pqe = add.rr %pqe0, %pqd")
            e("  %pnm = mov.modifier %pqe")
            e("  %pnx = padds.modifier %pnc, %pnm")
        elif q3_strides()[0] == 512 * NT:
            e("  %pnx = padds.modifier %pnc, %pqm")   # (the next row's slot: the even area's row stride on)
        else:
            e(f"  %pnk = mova.i32 {(q3_strides()[0] - 512 * (NT - 1)) // 512}")
            e("  %pnk2 = mul %k512, %pnk")
            e("  %pnm = mov.modifier %pnk2")
            e("  %pnx = padds.modifier %pnc, %pnm")
        e(sub_(back)[:-1] + ", %pnx: reg<aie2p.ep>)")
        L[-1] = L[-1].replace("^outer(", "^pouter(")
        e("^pexit:")
    if q3 and qpeel_on():
        pass
    elif q3 and os.environ.get("GN_QCVL", "1" if qunr_on() else "0") == "1":
        # (GN_QCVL: the convert below as a loop over the rows, ~1/4 of its code: the unroll's room without the peel)
        # a row's loads: 16 from the even area, 16 odd, 16 even, ... (post-incremented: each ends a row on; d too, but
        # for odd NP: its bf16 row is 512 B short of the even area's row, stepped below)
        n = 16 * NP
        assert q3_strides() == (1024 * ((NP + 1) // 2), 1024 * (NP // 2))
        e("  %cvz = mova.i32 0")
        e("  %cvo0 = copy %o : reg<aie2p.ep> -> reg<aie2p.ep>")
        e("  %cvq0 = copy %qod : reg<aie2p.ep> -> reg<aie2p.ep>")
        e(f"  %cvK = mova.i32 {MP}")
        e("  low.br ^cvl(%cvz: reg<aie2p.er>, %cvo0: reg<aie2p.ep>, %cvo0: reg<aie2p.ep>, %cvq0: reg<aie2p.ep>)")
        e("^cvl(%cvr: reg<aie2p.er>, %cvd_0: reg<aie2p.ep>, %cva_0: reg<aie2p.ep>, %cvb_0: reg<aie2p.ep>):")
        e("  %cvm = lt %cvr, %cvK")
        e("  %cvrn = add %cvr, 1")
        e("  low.cond_br %cvm, ^cvb, ^cvx : reg<aie2p.er>")
        e("^cvb:")
        if QCYC:   # (the row's first word, for the stamp: the stores consume %cvd_0)
            e("  %cvds = copy %cvd_0 : reg<aie2p.ep> -> reg<aie2p.ep>")
        cur = {"a": "%cva_0", "b": "%cvb_0"}

        def load_(i):
            g = "a" if (i // 16) % 2 == 0 else "b"
            e(f"  %cvl_{i}, %cv{g}_{i + 1} = vlda.acc.post {cur[g]}, 64")
            cur[g] = f"%cv{g}_{i + 1}"
        if cfg.qsw and QF.get("qsw"):   # (FFN gate / up) the row's h = silu(g) u: 576 B of bfp16 records at its slot; pointers a row on
            assert NP == 4 and cfg.hord == "P4" and q3_strides() == (2048, 2048) and not qalt_on()
            e("  %cvsa = copy %cva_0 : reg<aie2p.ep> -> reg<aie2p.ep>")
            e("  %cvsb = copy %cvb_0 : reg<aie2p.ep> -> reg<aie2p.ep>")
            if pair and not qpair_off():   # (Q' column pairs) the row's halves swapped with the partner core's first
                qpair_swap(e, pair, "%cvsa", "%cvsb", "q")
            t = "q"
            convert_swiglu(None, None, 0, q3=("%cvsa", "%cvsb"))
            e("  %cvsk = mov.i32 2048")
            e("  %cvsm = mov.modifier %cvsk")
            for g_ in ("a", "b"):
                e(f"  %cvn{g_} = padda.modifier %cv{g_}_0, %cvsm")
                cur[g_] = f"%cvn{g_}"
            e("  %cvnd = padda.modifier %cvd_0, %cvsm")
            d = "%cvnd"
        else:
            for i in range(16):
                load_(i)
            d = "%cvd_0"
            for i in range(n):
                e(f"  %cvd_{i + 1} = vst.convert.f32x16.to.bf16x16.post %cvl_{i}, {d}, 32")
                d = f"%cvd_{i + 1}"
                if i + 16 < n:
                    load_(i + 16)
        if QCYC:   # (diag) the cycles since the first firing's start into the row's first word
            e("  %tcw_l = mov.cycles")
            e("  %tcl_l = slice %tcw_l[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
            e("  %tsa_l = storage_address %tcs : low.storage<private> -> reg<aie2p.ep>")
            e("  %tst_l = lda %tsa_l, 0")
            e("  %tsp_l = sub %tcl_l, %tst_l")
            e("  st %tsp_l, %cvds, 0")
            if QCYCABS:
                e("  st %tst_l, %cvds, 4")
            if QFCYC and QF.get("cfg") is not None and fills is True:   # (fill cycles: words 1-3)
                e("  %qfaL = storage_address %qfc : low.storage<private> -> reg<aie2p.ep>")
                dqc_ = os.environ.get("DQ_CYC") == "1"   # (diag: word 1 the decoder's cycles, word 4 its hot loop's)
                for w_, o_ in ((1, 16 if dqc_ else 4), (2, 12), (3, 24)) + (((4, 28),) if dqc_ else ()) + (
                        ((5, 32), (6, 36), (7, 40)) if os.environ.get("GN_QFW") in ("1", "7") or os.environ.get("GN_QTL3") in ("1", "2") else
                        ((5, 44), (6, 48)) if os.environ.get("GN_QFW") == "2" else
                        ((5, 56),) if os.environ.get("GN_QFW") in ("3", "4", "5", "6") else ()):
                    if o_ >= 32 and w_ == 5:   # (offsets past the lda immediate: a pointer 32 on)
                        e("  %qfkL = mova.i32 32")
                        e("  %qfmL = mov.modifier %qfkL")
                        e("  %qfaL2 = padda.modifier %qfaL, %qfmL")
                    e(f"  %qfvL{w_} = lda {'%qfaL2' if o_ >= 32 else '%qfaL'}, {o_ - 32 if o_ >= 32 else o_}")
                    e(f"  st %qfvL{w_}, %cvds, {4 * w_}")
        e("  rel %one, 2")
        if q3_strides()[0] > 32 * n:   # (odd NP: d to the next even-area row)
            e(f"  %cvgk = mov.i32 {q3_strides()[0] - 32 * n}")
            e("  %cvgm = mov.modifier %cvgk")
            e(f"  %cvdg = padda.modifier {d}, %cvgm")
            d = "%cvdg"
        e(f"  low.br ^cvl(%cvrn: reg<aie2p.er>, {d}: reg<aie2p.ep>, {cur['a']}: reg<aie2p.ep>, {cur['b']}: reg<aie2p.ep>)")
        e("^cvx:")
    elif q3:   # each row: its f32 sub-tiles (even area, odd area, alternating) packed to bf16 in its even half
        for r in range(MP):
            t = 100 + r
            ptr = {}
            evs, ods = q3_strides()   # (sub-tile t: the even area's (t even) or the odd area's, 1 KB each a row)
            for nm, base, off in ((("d", "%o", evs * r),) +
                                  tuple((f"s{t_}", "%o" if t_ % 2 == 0 else "%qod",
                                         (evs if t_ % 2 == 0 else ods) * r + 1024 * (t_ // 2)) for t_ in range(NP))):
                e(f"  %c3{nm}a{r} = mova.i32 {off // 512}")
                e(f"  %c3{nm}o{r} = mul %k512, %c3{nm}a{r}")
                e(f"  %c3{nm}m{r} = mov.modifier %c3{nm}o{r}")
                e(f"  %c3{nm}c{r} = copy {base} : reg<aie2p.ep> -> reg<aie2p.ep>")
                e(f"  %c3{nm}p{r} = padds.modifier %c3{nm}c{r}, %c3{nm}m{r}")
                ptr[nm] = f"%c3{nm}p{r}"
            d = ptr["d"]
            n = 16 * NP

            def load3(i):
                sg = f"s{i // 16}"
                e(f"  %cvl{t}_{i}, %c3n{t}_{i} = vlda.acc.post {ptr[sg]}, 64")
                ptr[sg] = f"%c3n{t}_{i}"
            for i in range(16):
                load3(i)
            for i in range(n):
                e(f"  %cvd{t}_{i} = vst.convert.f32x16.to.bf16x16.post %cvl{t}_{i}, {d}, 32")
                d = f"%cvd{t}_{i}"
                if i + 16 < n:
                    load3(i + 16)
            if QCYC:   # (diag) the cycles since the first firing's start into the row's first word
                e(f"  %tcw{t} = mov.cycles")
                e(f"  %tcl{t} = slice %tcw{t}[0] : reg<aie2p.er x2> -> reg<aie2p.er>")
                e(f"  %tsa{t} = storage_address %tcs : low.storage<private> -> reg<aie2p.ep>")
                e(f"  %tst{t} = lda %tsa{t}, 0")
                e(f"  %tsp{t} = sub %tcl{t}, %tst{t}")
                e(f"  %c3q{r} = copy %o : reg<aie2p.ep> -> reg<aie2p.ep>")
                e(f"  %c3qq{r} = padds.modifier %c3q{r}, %c3dm{r}")
                e(f"  st %tsp{t}, %c3qq{r}, 0")
                if QCYCABS:
                    e(f"  st %tst{t}, %c3qq{r}, 4")
            e("  rel %one, 2")
    elif qrow:   # last pass: each row's f32 C stays in its slot until here (row offsets: the C pins, else 4 KB apart)
        if qbank:
            cs = [int(v) for v in (QBANK2_RING_AT if os.environ.get("GN_QBANK") == "2" else QBANK_RING_AT)
                  .split(",")[0].split("@")[1].split(":")]
        else:
            cs = [1024 * NP * r for r in range(MP)]
        # from %po (the row after the last one, as the latch left it: one fewer pointer live through the loop)
        par_ = (MP - 1) & 1
        dl = (0 if not qbank else par_ * 36864 - 20480 if os.environ.get("GN_QBANK") == "2" else 12288 - par_ * 28672)
        cx = cs[0] if qpass else cs[-1] + 1024 * NP + dl   # (qpass: the last row's latch wrapped to row 0)
        for r, off in enumerate(cs):
            v = off - cx + 1024 * (NP - 1)
            e(f"  %cvoa{r} = mova.i32 {v // 512}")
            e(f"  %cvo{r} = mul %k512, %cvoa{r}")
            e(f"  %cvm{r} = mov.modifier %cvo{r}")
            e(f"  %cvpc{r} = copy %po : reg<aie2p.ep> -> reg<aie2p.ep>")
            e(f"  %cvp{r} = padds.modifier %cvpc{r}, %cvm{r}")
            t = 100 + r
            convert(f"%cvp{r}", f"%cvp{r}", 1024 * (NP - 1))
    if tail and cfg.dfr:   # the deferred segment: pads, flush (convert_swiglu's), its slot released
        e("  low.cond_br %dfx, ^dfl, ^dfe : reg<aie2p.er>")
        e("^dfl:")
        e("  %dzc = acc.clear.f32x64")
        f_ = ("%dqf", "%dqo", "%dqp")
        for z in range(next(n for n in range(16) if 72 * (2 * NP + n) % 128 == 64)):
            e(f"  %dzf{z}, %dzo{z}, %dzq{z} = vst.push.bfp16ebs8.from.fp32 {f_[0]}, %dzc, {f_[1]}, {f_[2]}")
            f_ = (f"%dzf{z}", f"%dzo{z}", f"%dzq{z}")
        e(f"  %dyf, %dyp, %dyq = vst.flush.512 {f_[0]}, {f_[1]}, {f_[2]}")
        e("  low.br ^dfe")
        e("^dfe:")
        e("  %dqr = copy %dfx : reg<aie2p.er> -> reg<aie2p.mr26_lock>")
        e("  rel.cond %one, %dqr, 2")
    if fills and not (role == "tail" and qpass_on()):
        e("  rel %one, 1")
    if fills and QF.get("crel"):
        crel_copy(e, QF["crel"](gate), MP * (QF.get("rows", 2) - 1))
    elif CRMID[0]:
        crel_copy(e, QF["crelmid"], MP * (QF["rows"] - 1 - CRMID[0]), own=("%crst", 4))
    elif CREL0[0] and crelp_on():
        (crel_copy_staged(e, 5, ("%crst", 4)) if qcstage_on() else crel_copy(e, 5, MP, own=("%crst", 4)))
    elif CREL0[0]:
        crel_copy(e, QF["crel"](gate), MP * (QF.get("rows", 2) - 1), own=("%crst", 4))
    if gate:
        gate_count(L.append, cfg, gate)
        e("}\n")
    else:
        e("  return\n}\n")


# swiglu's sigmoid table: t = sigmoid(-|g|) (bf16) for the bf16 bits of |g| in [SL_LO, SL_HI] (2^-8 .. 128; |g| is
# clamped into it: below, t = 1/2 to bf16; above, t = 0). Gather payload (Loom's immutable gather layout): each 32 bytes
# twice (odd address lanes read a 64-byte block's second half), then a pad for the last pieces.
SL_LO, SL_HI = 0x3B80, 0x4300


SL_K = (0x7FFF, 0x3B80, 0x4300, 0x0000, 0x8000, 0x3F80)   # bf16x32 constants after the table (epilogue loads)


SL_NB = (SL_HI - SL_LO + 1 + 15) // 16   # the table's 16-entry blocks


def silu_pieces():
    """GN_QSILU="blocks@address,...": the table in pieces of whole blocks (their sum SL_NB), each its own symbol
    (silu_sym) pinned at its byte address (gen_npu_q.loom_env), no pad after one (a gather's over-read past a piece
    lands in memory that only needs to exist: its bytes are dropped); the SL_K vectors in @silu_k.
    Default: one piece with the pad at 45568. [(blocks, address)]"""
    v = os.environ.get("GN_QSILU")
    if not v:
        return list(QG_SILU) if qgmap_on() else [(SL_NB, 45568)]
    ps = [(int(a), int(b)) for a, b in (x.split("@") for x in v.split(","))]
    assert sum(n for n, _ in ps) == SL_NB, (ps, SL_NB)
    return ps


def silu_kvals():
    """The bf16x32 constant vectors at the table's tail: SL_K, then (pieces) each later piece's first index."""
    sp, b0, ks = silu_pieces(), 0, list(SL_K)
    for n, _ in sp[:-1]:
        b0 += n
        ks.append(SL_LO + 16 * b0)
    return tuple(ks)


def silu_sym(k):
    return "silu_t" if k == 0 else f"silu_t{k}"


def silu_kofs():
    """The byte offset of SL_K's first vector from the last piece's symbol."""
    sp = silu_pieces()
    if len(sp) == 1:
        n = SL_NB * 32 + 32
        return 2 * (n + (-n % 32))
    return 0   # (pieces: @silu_k)


def silu_rodata():
    import numpy as np
    x = (np.arange(SL_LO, SL_HI + 1, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)
    t = (1 / (1 + np.exp(x))).astype(np.float32)
    u = t.view(np.uint32).astype(np.uint64)
    tb = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    tb[t < np.float32(2.0 ** -126)] = 0                 # no denormals (the core flushes them)
    blk = np.zeros((len(tb) + 15) // 16 * 16, np.uint16)
    blk[:len(tb)] = tb
    blk = blk.reshape(-1, 16)
    sp = silu_pieces()
    kv = [np.full(32, k, np.uint16) for k in silu_kvals()]
    if len(sp) == 1:
        pay = np.concatenate([np.stack([blk, blk], 1).reshape(-1), np.zeros(32, np.uint16)])
        pay = np.concatenate([pay, np.zeros(-len(pay) % 32, np.uint16)] + kv)
        return f'global.rodata.def @silu_t = align(64) bytes("{pay.view(np.uint8).tobytes().hex()}")'
    out, b0 = [], 0
    for k, (nb, _) in enumerate(sp):
        pay = np.stack([blk[b0:b0 + nb], blk[b0:b0 + nb]], 1).reshape(-1)
        out.append(f'global.rodata.def @{silu_sym(k)} = align(64) bytes("{pay.view(np.uint8).tobytes().hex()}")')
        b0 += nb
    out.append(f'global.rodata.def @silu_k = align(64) bytes("{np.concatenate(kv).view(np.uint8).tobytes().hex()}")')
    return "\n".join(out)


def gen(cfg):
    assert len(cfg.ks) == 4 and cfg.passes >= 2 and 1 <= cfg.cols <= 8
    assert cfg.fuse in (0, 2) and (cfg.fuse == 2 or not cfg.gate), "fuse 0 or 2; gated images are fused"
    L = []
    array_program(L, cfg)
    assert len(set(cfg.ks[1:-1])) == 1 or cfg.kchain, "the mids share one leaf"   # (kchain: a leaf per mid)
    if cfg.swiglu:
        L.append(silu_rodata())
    if cfg.fuse:   # every worker's leaf, once (worker_leaf)
        if cfg.fuse == 2:   # the inlined decoder's tables (grid formats)
            for cc in [cfg] + ([colcfg(cfg, 1)] if fpair(cfg) else []):
                L.extend(decoder(cc.fmt, cc.gord, cc.swiglu).inline_rodata(dec_tag(cc)))
        seen = set()
        used = {worker_leaf(cfg, c, r)[0] for c in range(cfg.cols) for r in range(len(cfg.ks))}
        for c in [-1] + list(range(cfg.cols)):   # (-1: an ordinary column first: head, mid, mid_l, tail)
            for r in range(len(cfg.ks)):
                name, role, cov, gate = (worker_leaf(cfg, c, r) if c >= 0 else
                                         (lambda w: (w[0], w[1], w[2], None))(worker_leaf(cfg, -1, r)))
                # (thelp: no column runs the unhelped last mid; pairs: every tail swaps)
                if c < 0 and (cfg.thelp or cfg.kchain or fpair(cfg)) and name not in used:
                    continue
                hs = fuse_helpers(cfg, c)
                helper = "M" if r in hs else "H" if r in hs.values() else None
                if cfg.kchain and c >= 0:   # (kchain_role)
                    helper = kchain_role(cfg, c, r)
                if (helper or "")[:1] == "M":
                    cov = (r,)
                if name not in seen:
                    seen.add(name)
                    pair = ("e" if c % 2 else "w") if fpair(cfg) and c >= 0 and role == "tail" else None
                    leaf(L, colcfg(cfg, c), role, cfg.ks[r], gate=gate, cov=cov, name=name, helper=helper, pair=pair)
    else:
        leaf(L, cfg, "head", cfg.ks[0])
        leaf(L, cfg, "mid", cfg.ks[1])
        leaf(L, cfg, "tail", cfg.ks[-1])
    return "\n".join(L)


def main():
    if len(sys.argv) < 5:
        sys.exit("usage: gen_npu_gemm.py <cols> <m_blocks> <ks,ks,ks,ks> <passes> [entry]")
    cfg = Config(int(sys.argv[1]), int(sys.argv[2]), tuple(int(v) for v in sys.argv[3].split(",")), int(sys.argv[4]),
                 entry=sys.argv[5] if len(sys.argv) > 5 else "npu_gemm", gate=int(sys.argv[6]) if len(sys.argv) > 6 else 0)
    sys.stdout.write(gen(cfg))


if __name__ == "__main__":
    main()
