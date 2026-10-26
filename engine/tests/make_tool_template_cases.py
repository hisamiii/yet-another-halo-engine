#!/usr/bin/env python3
"""Render the tool-calling cases of serve/chat_template.cpp with the model's own Jinja template (the GGUF's or the
HFQ's tokenizer.chat_template), as transformers' apply_chat_template renders it (its tojson: json.dumps with
ensure_ascii=False), into serve/testdata/chat_template_tool_cases.json for yah_serve_test.

usage: make_tool_template_cases.py <model.gguf | model.hfq> [out.json]   (needs jinja2)
"""
import json
import os
import struct
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))


def template_of(path):
    with open(path, "rb") as f:
        head = f.read(64 << 20)
    if head[:4] == b"GGUF":   # the metadata key, then its string value (u64 length)
        key = b"tokenizer.chat_template"
        at = head.index(struct.pack("<Q", len(key)) + key) + 8 + len(key)
        n = struct.unpack_from("<Q", head, at + 4)[0]
        return head[at + 12:at + 12 + n].decode()
    at = head.index(b'"chat_template":"')   # the HFQ's JSON header
    return json.JSONDecoder().raw_decode(head[at + 16:].decode("utf-8", "replace"))[0]


def render(template, messages, tools, effort, add_generation_prompt=True):
    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    kw = {"messages": messages, "tools": tools, "add_generation_prompt": add_generation_prompt}
    if effort == "none":
        kw["enable_thinking"] = False
    elif effort != "high":
        kw["reasoning_effort"] = effort
    return env.from_string(template).render(**kw)


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                                                    "days": {"type": "integer"}, "metric": {"type": "boolean"}},
                   "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search", "description": "Search <docs> & \"notes\"\nfor a query. Gr\u00fc\u00dfe, \u6771\u4eac",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "number"},
                                                    "filters": {"type": "object"}, "tags": {"type": "array"}}}}}
NOARGS = {"type": "function", "function": {"name": "now", "description": "The time.",
                                           "parameters": {"type": "object", "properties": {}}}}


def call(name, args, id_="call_1"):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": args}}


CASES = [
    ("tools, user", "high", [{"role": "user", "content": "Weather in Paris?"}], [WEATHER]),
    ("tools, system, effort none", "none",
     [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Paris?"}], [WEATHER, SEARCH]),
    ("tools, effort low", "low", [{"role": "user", "content": "Search for cats"}], [SEARCH, NOARGS]),
    ("one call and its result", "high", [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "Need the tool.",
         "tool_calls": [call("get_weather", {"city": "Paris", "days": 3, "metric": True})]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 18}"}], [WEATHER]),
    ("text, two calls, two results, answer, follow-up", "medium", [
        {"role": "system", "content": "You search."},
        {"role": "user", "content": "Find two things"},
        {"role": "assistant", "content": "Searching now.", "tool_calls": [
            call("search", {"query": "a \"b\"\nc", "limit": 0.5, "filters": {"lang": "en", "years": [2020, 2021]},
                            "tags": ["x", "\u00fc"]}, "call_a"),
            call("search", {"query": "tiny", "limit": 1e-05}, "call_b")]},
        {"role": "tool", "tool_call_id": "call_a", "content": "first"},
        {"role": "tool", "tool_call_id": "call_b", "content": "second"},
        {"role": "assistant", "content": "Both found."},
        {"role": "user", "content": "Thanks, and big numbers?"}], [SEARCH]),
    ("call without arguments, tool last, no generation prompt", "high", [
        {"role": "user", "content": "What time is it?"},
        {"role": "assistant", "content": None, "tool_calls": [call("now", {})]},
        {"role": "tool", "content": "12:00"}], [NOARGS]),
    ("numbers", "high", [
        {"role": "user", "content": "n"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("search", {"limit": 1e16, "query": "q", "filters": {"a": 100.0, "b": -2.5e-07, "c": 123456789012}})]},
        {"role": "tool", "content": "ok"},
        {"role": "user", "content": "<tool_response>\nmore\n</tool_response>"}], [SEARCH]),
]


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    template = template_of(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "serve", "testdata",
                                                            "chat_template_tool_cases.json")
    cases = []
    for name, effort, messages, tools in CASES:
        gen = "no generation prompt" not in name
        cases.append({"name": name, "effort": effort, "add_generation_prompt": gen, "tools": tools,
                      "messages": messages, "expected": render(template, messages, tools, effort, gen)})
    with open(out, "w") as f:
        json.dump(cases, f, indent=1)
        f.write("\n")
    print(f"{len(cases)} cases -> {out}")


if __name__ == "__main__":
    main()
