#!/usr/bin/env python3
"""GPU smoke test of a release bundle: its yah_server with an empty environment (only HOME and PATH), so nothing of a
development setup (hrx-env.sh, LD_LIBRARY_PATH) can stand in for a missing piece.

usage: smoke.py <bundle dir> <model> [port] [--npu]

Checks: greedy chat completion (plain and streamed, same text), a Responses request, a sampled request with top-k and
penalties, a function tool's call and result through both APIs, and a follow-up turn that reuses the cache. --npu: the
server runs with --npu, must report it on, and must answer from deep in a two-chunk prompt (the NPU's pair
schedule). Takes about a minute after the model loads.
"""
import json
import os
import glob
import subprocess
import sys
import threading
import time
import urllib.request

args = [a for a in sys.argv[1:] if a != "--npu"]
npu = "--npu" in sys.argv
bundle, model = os.path.abspath(args[0]), os.path.abspath(args[1])
port = int(args[2]) if len(args) > 2 else 18995
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
server = subprocess.Popen([os.path.join(bundle, "bin", "yah_server"), "--model", model, "--port", str(port)] +
                          (["--npu"] if npu else []), env=env, stdout=log, stderr=subprocess.STDOUT)
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
    tool = {"name": "get_population", "description": "The population of a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
    ask_tool = [{"role": "user", "content": "What is the population of Lyon? Use the tool."}]
    t = post("/v1/chat/completions", {**base, "messages": ask_tool, "max_tokens": 200,
                                      "tools": [{"type": "function", "function": tool}]})["choices"][0]
    calls = t["message"].get("tool_calls") or []
    check(t["finish_reason"] == "tool_calls" and calls and json.loads(calls[0]["function"]["arguments"]).get("city"),
          "chat completions tool call: " + (calls[0]["function"]["arguments"] if calls else "none"))
    if calls:
        msgs = ask_tool + [t["message"], {"role": "tool", "tool_call_id": calls[0]["id"], "content": "522250"}]
        a = post("/v1/chat/completions", {**base, "messages": msgs, "max_tokens": 200,
                                          "tools": [{"type": "function", "function": tool}]})["choices"][0]
        check("522" in (a["message"]["content"] or "").replace(",", "").replace(" ", ""), "answer from the tool result")
    r = post("/v1/responses", {"model": "qwen", "input": ask_tool, "reasoning": {"effort": "none"},
                               "max_output_tokens": 200, "temperature": 0, "tools": [{"type": "function", **tool}]})
    fc = [o for o in r["output"] if o["type"] == "function_call"]
    check(fc and fc[0]["call_id"] and json.loads(fc[0]["arguments"]).get("city"), "responses function_call")
    if fc:
        inp = ask_tool + r["output"] + [{"type": "function_call_output", "call_id": fc[0]["call_id"],
                                         "output": "522250"}]
        r = post("/v1/responses", {"model": "qwen", "input": inp, "reasoning": {"effort": "none"},
                                   "max_output_tokens": 200, "temperature": 0, "tools": [{"type": "function", **tool}]})
        text = "".join(p["text"] for o in r["output"] if o["type"] == "message" for p in o["content"])
        check("522" in text.replace(",", "").replace(" ", ""), "responses answer from function_call_output")
    facts = "\n".join(f"Fact {i}: {i} times {i + 3} is {i * (i + 3)}." for i in range(1, 260))
    turn1 = [{"role": "user", "content": facts + "\nWhat is fact 100?"}]
    a1 = post("/v1/chat/completions", {**base, "messages": turn1})["choices"][0]["message"]["content"]
    turn2 = turn1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "And fact 7?"}]
    r2 = post("/v1/chat/completions", {**base, "messages": turn2})
    cached = r2["usage"]["prompt_tokens_details"]["cached_tokens"]
    check(cached >= 2048, f"follow-up reuses the cache ({cached} of {r2['usage']['prompt_tokens']} tokens)")
    if npu:
        check("yah_server: npu on" in open(log.name).read(), "npu on")
        facts = "\n".join(f"Fact {i}: {i} times {i + 3} is {i * (i + 3)}." for i in range(1, 560))
        deep = [{"role": "user", "content": facts + "\nWhat is fact 300?"}]
        r = post("/v1/chat/completions", {**base, "messages": deep})
        a = r["choices"][0]["message"]["content"]
        check("90900" in a.replace(",", ""), f"two-chunk prompt ({r['usage']['prompt_tokens']} tokens): {a!r}")
finally:
    server.terminate()
    server.wait(timeout=60)
if gtt:
    print(f"  GTT peak {peak[0] / 2**30:.1f} GiB (the desktop's share included)")
print("smoke: FAIL" if fails else "smoke: ok")
sys.exit(1 if fails else 0)
