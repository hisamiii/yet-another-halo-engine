#!/usr/bin/env python3
"""GPU smoke test of a release bundle: its yah_server with an empty environment (only HOME and PATH), so nothing of a
development setup (hrx-env.sh, LD_LIBRARY_PATH) can stand in for a missing piece.

usage: smoke.py <bundle dir> <model> [port]

Checks: greedy chat completion (plain and streamed, same text), a Responses request, a sampled request with top-k and
penalties, and a follow-up turn that reuses the cache. Takes about a minute after the model loads.
"""
import json
import os
import glob
import subprocess
import sys
import threading
import time
import urllib.request

bundle, model = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
port = int(sys.argv[3]) if len(sys.argv) > 3 else 18995
url = f"http://127.0.0.1:{port}"


def post(path, body, stream=False):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        if not stream:
            return json.loads(r.read())
        text = ""
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                for c in json.loads(line[6:])["choices"]:
                    text += c["delta"].get("content") or ""
        return text


fails = 0


def check(ok, what):
    global fails
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    fails += not ok


gtt = (glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used") or [None])[0]
peak = [0]


def watch():   # the GTT high-water mark while the server runs
    while gtt and server.poll() is None:
        peak[0] = max(peak[0], int(open(gtt).read()))
        time.sleep(0.5)


env = {"HOME": os.environ.get("HOME", "/"), "PATH": "/usr/bin:/bin"}
log = open(os.path.join(os.path.dirname(bundle), "smoke-server.log"), "w")
server = subprocess.Popen([os.path.join(bundle, "bin", "yah_server"), "--model", model, "--port", str(port)], env=env,
                          stdout=log, stderr=subprocess.STDOUT)
threading.Thread(target=watch, daemon=True).start()
try:
    for _ in range(300):
        if server.poll() is not None:
            sys.exit(f"smoke: yah_server exited ({server.returncode}); see {log.name}")
        try:
            urllib.request.urlopen(url + "/health", timeout=2)
            break
        except OSError:
            time.sleep(1)
    else:
        sys.exit("smoke: no /health after 300 s")
    ask = [{"role": "user", "content": "Name three primary colors, comma separated."}]
    base = {"model": "qwen", "messages": ask, "reasoning_effort": "none", "max_tokens": 32, "temperature": 0}
    plain = post("/v1/chat/completions", base)["choices"][0]["message"]["content"]
    print("  reply:", repr(plain))
    check(len(plain) > 3, "chat completion")
    check(post("/v1/chat/completions", {**base, "stream": True}, stream=True) == plain, "streamed = plain")
    r = post("/v1/responses", {"model": "qwen", "input": ask[0]["content"], "reasoning": {"effort": "none"},
                               "max_output_tokens": 32, "temperature": 0})
    text = "".join(p["text"] for o in r["output"] if o["type"] == "message" for p in o["content"])
    check(text == plain, "responses = chat completions")
    s = post("/v1/chat/completions", {**base, "temperature": 0.8, "top_k": 20, "min_p": 0.05, "seed": 7,
                                      "presence_penalty": 0.5, "repetition_penalty": 1.1})
    check(len(s["choices"][0]["message"]["content"]) > 3, "sampled with top-k, min-p and penalties")
    facts = "\n".join(f"Fact {i}: {i} times {i + 3} is {i * (i + 3)}." for i in range(1, 260))
    turn1 = [{"role": "user", "content": facts + "\nWhat is fact 100?"}]
    a1 = post("/v1/chat/completions", {**base, "messages": turn1})["choices"][0]["message"]["content"]
    turn2 = turn1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "And fact 7?"}]
    r2 = post("/v1/chat/completions", {**base, "messages": turn2})
    cached = r2["usage"]["prompt_tokens_details"]["cached_tokens"]
    check(cached >= 2048, f"follow-up reuses the cache ({cached} of {r2['usage']['prompt_tokens']} tokens)")
finally:
    server.terminate()
    server.wait(timeout=60)
if gtt:
    print(f"  GTT peak {peak[0] / 2**30:.1f} GiB (the desktop's share included)")
print("smoke: FAIL" if fails else "smoke: ok")
sys.exit(1 if fails else 0)
