#!/usr/bin/env python3
"""Golden test: the C++ chat template (yah_server --render-chat) against the GGUF's Jinja template rendered by jinja2.

usage: PYTHONPATH=<llama.cpp>/gguf-py python test_chat_template.py [model.gguf] [yah_server]
"""
import json
import subprocess
import sys
from pathlib import Path

from gguf import GGUFReader
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[2]
MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/q/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf"
SERVER = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "engine/build/yah_server")


def user(content):
    return {"role": "user", "content": content}


def assistant(content, reasoning=None):
    m = {"role": "assistant", "content": content}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    return m


def parts(*texts, kind="input_text"):
    return [{"type": kind, "text": t} for t in texts]


HISTORY = [
    {"role": "system", "content": "You are terse."},
    user("What is 2+2?"),
    assistant("4", "Simple arithmetic.\n2+2=4."),
    user("And 3+3?"),
]

CASES = [
    ("single user", [user("Hello")], "high"),
    ("system + user", [{"role": "system", "content": "Be nice."}, user("Hi")], "high"),
    ("developer + user", [{"role": "developer", "content": "Answer in French."}, user("Hi")], "high"),
    ("multi-turn with reasoning", HISTORY, "high"),
    ("effort none", [user("Hello")], "none"),
    ("effort low", [user("Hello")], "low"),
    ("effort medium", [user("Hello")], "medium"),
    ("effort none, system", [{"role": "system", "content": "S"}, user("Hi")], "none"),
    ("effort medium, system", [{"role": "system", "content": "S"}, user("Hi")], "medium"),
    ("effort low, multi-turn", HISTORY, "low"),
    ("effort none, multi-turn", HISTORY, "none"),
    ("content parts", [user(parts("Part one. ", "Part two.")), assistant(parts("Reply.", kind="output_text")),
                       user(parts("Third", kind="text"))], "high"),
    ("system parts", [{"role": "system", "content": parts("A", " B")}, user("x")], "medium"),
    ("trim whitespace", [{"role": "system", "content": "\n  sys \t\n"}, user("  \n hi there \n\n"),
                         assistant("\n answer \n", "\n\n  why \n"), user("next ")], "high"),
    ("trim unicode whitespace", [user("\u3000\u00a0hi\u2028\u00a0"), assistant("\u2003a\u3000", "\u00a0r\u00a0"),
                                 user("q\u0085")], "low"),
    ("empty system, medium", [{"role": "system", "content": "  "}, user("Hi")], "medium"),
    ("empty system, high", [{"role": "system", "content": ""}, user("Hi")], "high"),
    ("assistant without reasoning", [user("a"), assistant("b"), user("c")], "high"),
    ("unicode text", [user("\u041f\u0440\u0438\u0432\u0435\u0442 \u4f60\u597d \U0001f642 caf\u00e9"),
                      assistant("\U0001d518", "\u0434\u0443\u043c\u0430\u044e")], "none"),
    ("tool_response user turn", [user("real question"), assistant("ok"),
                                 user("<tool_response>\nresult\n</tool_response>")], "high"),
    ("assistant last", [user("Hi"), assistant("Partial")], "medium"),
    ("system merge", [{"role": "system", "content": " One. "}, {"role": "developer", "content": "Two."},
                      user("Hi"), {"role": "system", "content": "Three."}], "high"),
    ("no generation prompt", [user("Hi"), assistant("Hello", "greet")], "high", False),
    ("no generation prompt, none", [user("Hi")], "none", False),
    ("error: no user", [{"role": "system", "content": "S"}, assistant("x")], "high"),
    ("error: only tool_response", [user("<tool_response>r</tool_response>")], "high"),
]


def part_text(content):
    if content is None or isinstance(content, str):
        return content or ""
    return "".join(p["text"] for p in content)


def normalize(messages):
    # The template takes one system message, first. The server merges system and developer messages into one.
    system = [m for m in messages if m["role"] in ("system", "developer")]
    if not system or (len(system) == 1 and messages[0] is system[0] and system[0]["role"] == "system"):
        return messages
    texts = [part_text(m["content"]).strip() for m in system]
    merged = {"role": "system", "content": "\n\n".join(t for t in texts if t)}
    return [merged] + [m for m in messages if m["role"] not in ("system", "developer")]


def raise_exception(message):
    raise TemplateError(message)


def main():
    reader = GGUFReader(MODEL)
    field = reader.fields["tokenizer.chat_template"]
    source = bytes(field.parts[field.data[0]]).decode()
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(source)

    requests = []
    expected = []
    for case in CASES:
        name, messages, effort = case[:3]
        add_generation_prompt = case[3] if len(case) > 3 else True
        requests.append({"messages": messages, "effort": effort, "add_generation_prompt": add_generation_prompt})
        kwargs = {"messages": normalize(messages), "add_generation_prompt": add_generation_prompt}
        if effort == "none":
            kwargs["enable_thinking"] = False
        elif effort in ("low", "medium"):
            kwargs["reasoning_effort"] = effort
        try:
            expected.append(template.render(**kwargs))
        except TemplateError as error:
            expected.append({"error": str(error)})

    out = subprocess.run([SERVER, "--render-chat"], input=json.dumps(requests), capture_output=True, text=True,
                         check=True)
    got = json.loads(out.stdout)
    passed = 0
    for case, want, have in zip(CASES, expected, got):
        both_errors = isinstance(want, dict) and isinstance(have, dict)
        if want == have or both_errors:
            passed += 1
            continue
        print(f"FAIL {case[0]}\n  jinja: {want!r}\n  c++:   {have!r}")
    print(f"chat template: {passed}/{len(CASES)} cases match")
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
