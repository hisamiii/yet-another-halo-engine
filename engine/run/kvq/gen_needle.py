#!/usr/bin/env python3
"""Multi-key retrieval prompts for gate v2 tier C.

  gen_needle.py <book .ids.raw> <length> <seed> <out prefix>

Builds a sequence of exactly <length> tokens: book text (token slices of a tokenized book) with NKEYS facts at depths 5%..95%:
    "\\nThe access code of agent Falcon-17 is 4829173.\\n"
Keys are near-duplicates (agent names share a prefix, numbers differ in one or two digits); values are random 7-digit codes.
The prompt ends with NQ questions:
    "\\nThe access code of agent Falcon-17 is" + " 4829173" + ".\\n"
Their answer tokens are scored teacher-forced (the logits at the position before each answer token).
Segments are tokenized separately and joined at the token level, so every answer position is exact.

Harder variant (env KVQ_NKEYS / KVQ_NQ / KVQ_NUPD / KVQ_WARM): more keys, and NUPD of the queried keys are reassigned later in the text.
A reassignment reads "The access code of agent X was changed to Y."; the question asks for the current code.
WARM unscored warm-up questions come before the scored ones, so format uncertainty does not dominate the first scored query.

Writes <prefix>.ids (space-separated ids), <prefix>.pos (positions to score, one per line) and <prefix>.json (keys, values, depths, answer spans).
"""
import json
import os
import subprocess
import sys

import numpy as np

TOK = "/var/lib/lemonade/.cache/lemonade/bin/llamacpp/rocm-stable/llama-tokenize"
MODEL = "/home/q/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf"
NKEYS = int(os.environ.get("KVQ_NKEYS", "32"))
NQ = int(os.environ.get("KVQ_NQ", "8"))
NUPD = int(os.environ.get("KVQ_NUPD", "0"))
WARM = int(os.environ.get("KVQ_WARM", "0"))
_cache = {}


def tok(text):
    if text not in _cache:
        out = subprocess.run([TOK, "-m", MODEL, "-p", text, "--ids", "--no-bos", "--log-disable"],
                             capture_output=True, text=True, check=True).stdout
        _cache[text] = json.loads(out.strip().splitlines()[-1])
    return _cache[text]


def main():
    book, L, seed, prefix = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    rng = np.random.default_rng(seed)
    hay = json.loads(open(book).read())[2000:]
    names = ["Falcon", "Falcon", "Heron", "Heron"]          # shared prefixes: near-duplicate keys
    keys, vals = [], []
    while len(keys) < NKEYS:
        k = f"{names[len(keys) % 4]}-{rng.integers(10, 99 if NKEYS <= 64 else 999)}"
        if k not in keys:
            keys.append(k)
            vals.append(str(rng.integers(1_000_000, 9_999_999)))
    needles = [tok(f"\nThe access code of agent {k} is {v}.\n") for k, v in zip(keys, vals)]
    qidx = rng.choice(NKEYS, WARM + NQ, replace=False)
    # reassignments: (key index, new value); placed after the original fact
    upd = {int(i): str(rng.integers(1_000_000, 9_999_999)) for i in qidx[WARM:WARM + NUPD]}
    unl = {i: tok(f"\nThe access code of agent {keys[i]} was changed to {v}.\n") for i, v in upd.items()}
    cur = {i: upd.get(i, vals[i]) for i in range(NKEYS)}
    qpre = [tok(f"\nThe access code of agent {keys[i]} is") for i in qidx]
    qans = [tok(f" {cur[i]}") for i in qidx]
    qend = tok(".\n")
    tail = sum(len(a) + len(b) + len(qend) for a, b in zip(qpre, qans))
    body = L - tail - sum(len(n) for n in needles) - sum(len(u) for u in unl.values())
    nslot = NKEYS + len(upd)
    depths = np.linspace(0.05, 0.95, nslot)
    # slot order: a random permutation of facts, each reassignment moved after its original
    events = [("set", int(k)) for k in rng.permutation(NKEYS)]
    for i in upd:
        j = next(n for n, e in enumerate(events) if e == ("set", i))
        events.insert(int(rng.integers(j + 1, len(events) + 1)), ("upd", i))
    ids, spans, kdepth, udepth = [], [], {}, {}
    cut = 0
    for slot, (kind, ki) in enumerate(events):
        at = int(depths[slot] * body)
        ids += hay[cut:at]; cut = at
        if kind == "set":
            kdepth[ki] = len(ids) / L
            ids += needles[ki]
        else:
            udepth[ki] = len(ids) / L
            ids += unl[ki]
    ids += hay[cut:body]
    for i, (a, b) in enumerate(zip(qpre, qans)):
        ids += a
        ki = int(qidx[i])
        if i >= WARM:
            spans.append({"key": keys[ki], "value": cur[ki], "depth": udepth.get(ki, kdepth[ki]),
                          "updated": ki in upd, "old_depth": kdepth[ki], "start": len(ids), "tokens": b})
        ids += b + qend
    assert len(ids) == L, (len(ids), L)
    pos = [p - 1 for s in spans for p in range(s["start"], s["start"] + len(s["tokens"]))]
    open(prefix + ".ids", "w").write(" ".join(map(str, ids)))
    open(prefix + ".pos", "w").write("\n".join(map(str, pos)) + "\n")
    json.dump({"length": L, "keys": keys, "values": vals, "queries": spans}, open(prefix + ".json", "w"), indent=1)
    print(f"{prefix}: {L} tokens, {NKEYS} keys, {NQ} queries, {len(pos)} scored positions")


if __name__ == "__main__":
    main()
