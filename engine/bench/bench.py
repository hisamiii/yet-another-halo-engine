#!/usr/bin/env python3
"""End-to-end benchmark of an OpenAI-compatible chat server (yah_server, llama.cpp's llama-server, ...): prefill and
decode speed at several prompt lengths, as a client sees them.

usage: bench.py [--url http://127.0.0.1:8080] [--lengths 512,2048,8192,16384,30000] [--gen 128] [--cool 55]
                [--json out.json]

One greedy request per length, thinking off, max_tokens --gen with ignore_eos so every run decodes the same count.
Each prompt starts with a random nonce, so no prefix cache serves it. The numbers come from the server's llama.cpp
timings object when it sends one (yah_server, llama-server): prefill = prompt_n / prompt_ms (to the first token),
decode = the 2nd to the last token. Without timings the client times the stream: first token, then the rest.
Prompt lengths are approximate: one probe request measures tokens per filler line, the table shows the real counts.

Measuring rules (docs/build-and-run.md, Measuring): one round per point, no repeats; before each run the APU cools to
--cool C (k10temp Tctl; 0: don't wait), then 15 s, 30 s for prompts of 8K and up. A run reports the Tctl it started at.
"""
import argparse
import glob
import json
import os
import random
import sys
import time
import urllib.request

WORDS = ("river stone lantern copper meadow signal harbor quiet ember orchard violet north thread window summit "
         "candle marble autumn falcon garden silver echo timber canyon ribbon planet cedar velvet compass").split()


def tctl():
    """The APU's Tctl in C (k10temp), or None."""
    for d in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if open(os.path.join(d, "name")).read().strip() == "k10temp":
                return int(open(os.path.join(d, "temp1_input")).read()) / 1000.0
        except OSError:
            pass
    return None


def cool(limit, settle):
    t = tctl()
    if limit and t is not None:
        while t > limit:
            time.sleep(2)
            t = tctl()
        time.sleep(settle)
        t = tctl()
    return t


def filler(lines, seed):
    g = random.Random(seed)
    return "\n".join(f"Note {i}: " + " ".join(g.choice(WORDS) for _ in range(12)) + "." for i in range(lines))


def run(url, model, lines, gen):
    """One streamed request; returns its numbers."""
    nonce = "%016x" % random.getrandbits(64)
    content = f"Session {nonce}.\n" + filler(lines, nonce) + "\nSummarize the notes above."
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": gen, "temperature": 0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    usage = timings = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            for c in chunk.get("choices") or []:
                d = c.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    last = time.perf_counter()
                    first = first or last
            usage = chunk.get("usage") or usage
            timings = chunk.get("timings") or timings
    n = usage["prompt_tokens"] if usage else None
    out = {"prompt_tokens": n, "gen_tokens": usage["completion_tokens"] if usage else None}
    if timings:
        out.update(cached=timings.get("cache_n", 0), prefill_ms=timings["prompt_ms"],
                   prefill_tps=timings["prompt_per_second"], decode_tps=timings["predicted_per_second"], source="server")
    else:
        k = out["gen_tokens"] or 0
        out.update(cached=0, prefill_ms=(first - t0) * 1000.0, prefill_tps=n * 1000.0 / ((first - t0) * 1000.0),
                   decode_tps=(k - 1) / (last - first) if k > 1 and last > first else 0.0, source="client")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--lengths", default="512,2048,8192,16384,30000")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--cool", type=float, default=55.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    with urllib.request.urlopen(url + "/v1/models", timeout=30) as r:
        model = json.loads(r.read())["data"][0]["id"]
    # tokens per filler line, from one probe (its own nonce: it warms nothing the runs reuse)
    probe = run(url, model, 40, 2)
    base = run(url, model, 0, 2)["prompt_tokens"]
    per_line = (probe["prompt_tokens"] - base) / 40.0
    print(f"{model} at {url}: {per_line:.1f} tokens per filler line, {base} tokens of template and question", flush=True)
    rows = []
    print(f"{'prompt':>7} {'cached':>6} {'prefill ms':>10} {'prefill t/s':>11} {'gen':>4} {'decode t/s':>10} {'Tctl':>5}",
          flush=True)
    for want in (int(x) for x in a.lengths.split(",")):
        lines = max(0, round((want - base) / per_line))
        t = cool(a.cool, 30 if want >= 8192 else 15)
        r = run(url, model, lines, a.gen)
        r.update(target=want, tctl_start=t)
        rows.append(r)
        print(f"{r['prompt_tokens']:>7} {r['cached']:>6} {r['prefill_ms']:>10.1f} {r['prefill_tps']:>11.1f} "
              f"{r['gen_tokens']:>4} {r['decode_tps']:>10.2f} {t if t is not None else float('nan'):>5.1f}", flush=True)
        if r["cached"]:
            print("  (the server reused a cache for this prompt: its prefill numbers are not a cold prefill)")
    if rows and rows[0]["source"] == "client":
        print("the server sends no timings: prefill is time to the first token as the client sees it")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"url": url, "model": model, "gen": a.gen, "date": time.strftime("%Y-%m-%d %H:%M"),
                       "rows": rows}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
