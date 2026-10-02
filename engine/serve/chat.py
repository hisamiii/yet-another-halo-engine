#!/usr/bin/env python3
"""Terminal chat client for yah_server (OpenAI Responses API, streaming). Standard library only.

usage: chat.py [--url http://127.0.0.1:8080] [--effort none|low|medium|high] [--temperature T] [--max N] [--system TEXT]

Type a message and press Enter. Commands: /reset (new conversation), /effort E, /temp T, /max N, /system TEXT, /quit.
Reasoning streams dimmed, the answer in normal text; a stats line follows each reply.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DIM, RESET = "\033[2m", "\033[0m"


def stream_response(url, body):
    """Yield (event type, data dict) from the server's SSE stream."""
    req = urllib.request.Request(url + "/v1/responses", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    with urllib.request.urlopen(req) as resp:
        event = None
        for raw in resp:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                yield event, json.loads(line[6:])
                event = None


def main():
    ap = argparse.ArgumentParser(description="chat with yah_server")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--effort", default="none", choices=["none", "low", "medium", "high"])
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max", type=int, default=2048, help="max output tokens per reply")
    ap.add_argument("--system", default=None)
    a = ap.parse_args()
    history = []  # Responses API input items: user messages and the model's previous output items
    print(f"yah chat ({a.url}), effort={a.effort}. /reset /effort E /temp T /max N /system TEXT /quit")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text.startswith("/"):
            cmd, _, arg = text.partition(" ")
            if cmd == "/quit":
                return
            elif cmd == "/reset":
                history = []
                print("(new conversation)")
            elif cmd == "/effort" and arg in ("none", "low", "medium", "high"):
                a.effort = arg
            elif cmd == "/temp":
                a.temperature = float(arg) if arg else None
            elif cmd == "/max" and arg.isdigit():
                a.max = int(arg)
            elif cmd == "/system":
                a.system = arg or None
            else:
                print("commands: /reset /effort none|low|medium|high /temp T /max N /system TEXT /quit")
            continue
        history.append({"role": "user", "content": text})
        body = {"model": "yah", "input": history, "stream": True, "max_output_tokens": a.max,
                "reasoning": {"effort": a.effort}}
        if a.temperature is not None:
            body["temperature"] = a.temperature
        if a.system:
            body["instructions"] = a.system
        t0 = time.time()
        first = None
        final = None
        try:
            for ev, d in stream_response(a.url, body):
                if ev in ("response.reasoning_text.delta", "response.output_text.delta"):
                    first = first or time.time()
                    dim = ev.startswith("response.reasoning")
                    sys.stdout.write((DIM if dim else "") + d["delta"] + (RESET if dim else ""))
                    sys.stdout.flush()
                elif ev == "response.output_item.done" and d["item"]["type"] == "reasoning":
                    sys.stdout.write("\n\n")
                elif ev in ("response.completed", "response.incomplete", "response.failed"):
                    final = d["response"]
        except urllib.error.HTTPError as e:
            print(f"error {e.code}: {e.read().decode()}")
            history.pop()
            continue
        except urllib.error.URLError as e:
            print(f"cannot reach {a.url}: {e.reason}")
            history.pop()
            continue
        except KeyboardInterrupt:
            print("\n(interrupted)")
            history.pop()
            continue
        if final is None:
            print("\n(stream ended without a final response)")
            history.pop()
            continue
        history.extend(final["output"])  # keep the model's turn (reasoning + message) for the next request
        u = final.get("usage") or {}
        out_tok = u.get("output_tokens", 0)
        ttft = (first - t0) if first else 0.0
        gen_s = time.time() - (first or t0)
        rate = (out_tok - 1) / gen_s if out_tok > 1 and gen_s > 0 else 0.0
        status = final.get("status")
        print(f"\n{DIM}[{u.get('input_tokens', 0)} in, {out_tok} out, first token {ttft:.2f} s, {rate:.1f} tok/s"
              f"{', ' + status if status != 'completed' else ''}]{RESET}")


if __name__ == "__main__":
    main()
