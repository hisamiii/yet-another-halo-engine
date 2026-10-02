#!/usr/bin/env python3
"""End-to-end test of yah_server --fake (no GPU) through the official openai SDK and raw HTTP.

usage: python test_responses.py [model.gguf] [yah_server]   (needs the openai package)
"""
import http.client
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import openai
from openai.types.responses import Response

ROOT = Path(__file__).resolve().parents[2]
MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/q/Downloads/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf"
SERVER = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "engine/build/yah_server")

# FakeGenerator's canned reply (fake_generator.hpp).
REASONING = "The user wants a reply. I give the canned one."
ANSWER = "Hello! This is a fake reply. \U0001D518 caf\u00e9."

REASONING_EVENTS = ["response.output_item.added", "response.content_part.added", "response.reasoning_text.delta",
                    "response.reasoning_text.done", "response.content_part.done", "response.output_item.done"]
MESSAGE_EVENTS = ["response.output_item.added", "response.content_part.added", "response.output_text.delta",
                  "response.output_text.done", "response.content_part.done", "response.output_item.done"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def request(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    conn.request(method, path, body, {"Content-Type": "application/json"})
    res = conn.getresponse()
    data = res.read()
    conn.close()
    return res.status, data


def check_usage(r):
    u = r.usage
    assert u.total_tokens == u.input_tokens + u.output_tokens, u
    assert u.input_tokens > 0 and u.input_tokens_details.cached_tokens == 0, u
    assert 0 <= u.output_tokens_details.reasoning_tokens <= u.output_tokens, u


def check_full(r, effort="high"):
    """A complete reply to one of the fake's prompts."""
    assert r.status == "completed" and r.error is None and r.incomplete_details is None, r
    assert r.output_text == ANSWER, r.output_text
    check_usage(r)
    if effort in ("none", "minimal"):
        assert [o.type for o in r.output] == ["message"], r.output
        assert r.usage.output_tokens_details.reasoning_tokens == 0
    else:
        assert [o.type for o in r.output] == ["reasoning", "message"], r.output
        assert r.output[0].content[0].type == "reasoning_text" and r.output[0].content[0].text == REASONING
        assert r.output[0].summary == [] and r.output[0].id.startswith("rs_")
        assert 0 < r.usage.output_tokens_details.reasoning_tokens < r.usage.output_tokens
    msg = r.output[-1]
    assert msg.id.startswith("msg_") and msg.status == "completed" and msg.role == "assistant"
    assert r.id.startswith("resp_") and r.reasoning.effort == effort


def collect(stream):
    """Returns the event list; checks sequence numbers and that every delta is whole UTF-8."""
    events = list(stream)
    assert [e.sequence_number for e in events] == list(range(len(events))), [e.sequence_number for e in events]
    for e in events:
        assert "\ufffd" not in getattr(e, "delta", ""), e
    return events


def squash(types):
    """Event types with runs of the same delta event collapsed into one."""
    out = []
    for t in types:
        if not (out and out[-1] == t and t.endswith(".delta")):
            out.append(t)
    return out


def check_stream_events(events, effort="high", final="response.completed"):
    types = [e.type for e in events]
    expect = ["response.created", "response.in_progress"]
    if effort != "none":
        expect += REASONING_EVENTS
    expect += MESSAGE_EVENTS + [final]
    assert squash(types) == expect, squash(types)
    by_type = {}
    for e in events:
        by_type.setdefault(e.type, []).append(e)
    if effort != "none":
        reasoning = "".join(e.delta for e in by_type["response.reasoning_text.delta"])
        assert reasoning == REASONING and by_type["response.reasoning_text.done"][0].text == REASONING
    answer = "".join(e.delta for e in by_type["response.output_text.delta"])
    assert answer == ANSWER and by_type["response.output_text.done"][0].text == ANSWER
    done = events[-1].response
    Response.model_validate(done.model_dump())
    assert done.id == events[0].response.id
    return done


def main():
    port = free_port()
    log = tempfile.TemporaryFile("w+")
    server = subprocess.Popen([SERVER, "--model", MODEL, "--fake", "--port", str(port)], stderr=log)
    base = f"http://127.0.0.1:{port}"
    checks = 0
    try:
        for _ in range(300):
            try:
                if request(port, "GET", "/health")[0] == 200:
                    break
            except OSError:
                time.sleep(0.1)
        client = openai.OpenAI(base_url=base + "/v1", api_key="unused")

        def ok(name):
            nonlocal checks
            checks += 1
            print(f"ok  {name}")

        models = client.models.list()
        assert len(models.data) == 1 and models.data[0].owned_by == "yah", models
        ok("GET /v1/models")

        # Non-streaming, string input, default effort. Validate the raw JSON strictly too.
        raw = client.responses.with_raw_response.create(model="qwen-test", input="Hi there")
        Response.model_validate(raw.http_response.json())
        r = raw.parse()
        check_full(r)
        assert r.model == "qwen-test" and r.metadata == {} and r.tools == [] and r.parallel_tool_calls is False
        ok("create: string input, reasoning on")

        # Message list with content parts, instructions, metadata, thinking off.
        r = client.responses.create(
            model="qwen-test",
            instructions="Be brief.",
            input=[{"role": "user", "content": "Hi"},
                   {"role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]},
                   {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Again"}]}],
            reasoning={"effort": "none"},
            metadata={"k": "v"},
            temperature=0.2,
            top_p=0.5,
            store=False,
        )
        check_full(r, "none")
        assert r.instructions == "Be brief." and r.metadata == {"k": "v"}
        assert r.temperature == 0.2 and r.top_p == 0.5
        ok("create: message list, instructions, effort none")

        # A previous response's output items fed back as input (multi-turn).
        first = client.responses.create(model="m", input="Hi", reasoning={"effort": "low"})
        check_full(first, "low")
        history = [{"role": "user", "content": "Hi"}] + [o.model_dump(exclude_none=True) for o in first.output]
        r = client.responses.create(model="m", input=history + [{"role": "user", "content": "More"}],
                                    reasoning={"effort": "medium"})
        check_full(r, "medium")
        assert r.usage.input_tokens > first.usage.input_tokens
        r = client.responses.create(model="m", input="Hi", reasoning={"effort": "minimal"})
        check_full(r, "minimal")
        ok("create: output items as input, effort low / medium / minimal")

        # Streaming.
        events = collect(client.responses.create(model="m", input="Hi", stream=True))
        done = check_stream_events(events)
        check_full(done)
        events = collect(client.responses.create(model="m", input=[{"role": "user", "content": "Hi"}],
                                                 reasoning={"effort": "none"}, stream=True))
        check_full(check_stream_events(events, "none"), "none")
        ok("stream: event order, sequence numbers, deltas, effort high and none")

        # The SDK's stream helper accumulates the snapshot from the events.
        with client.responses.stream(model="m", input="Hi") as stream:
            for _ in stream:
                pass
            final = stream.get_final_response()
        check_full(final)
        ok("responses.stream helper")

        # max_output_tokens: stops inside the reasoning, then inside the answer.
        r = client.responses.create(model="m", input="Hi", max_output_tokens=5)
        assert r.status == "incomplete" and r.incomplete_details.reason == "max_output_tokens", r
        assert [o.type for o in r.output] == ["reasoning"] and REASONING.startswith(r.output[0].content[0].text)
        assert r.usage.output_tokens == 5 and r.usage.output_tokens_details.reasoning_tokens == 5
        assert r.max_output_tokens == 5
        check_usage(r)
        r = client.responses.create(model="m", input="Hi", max_output_tokens=20)
        assert r.status == "incomplete" and [o.type for o in r.output] == ["reasoning", "message"], r
        assert r.output[1].status == "incomplete" and ANSWER.startswith(r.output_text) and r.output_text != ANSWER
        assert r.usage.output_tokens == 20
        events = collect(client.responses.create(model="m", input="Hi", max_output_tokens=5, stream=True))
        assert events[-1].type == "response.incomplete" and events[-1].response.status == "incomplete"
        Response.model_validate(events[-1].response.model_dump())
        events = collect(client.responses.create(model="m", input="Hi", max_output_tokens=5,
                                                 reasoning={"effort": "none"}, stream=True))
        assert events[-1].type == "response.incomplete" and events[-1].response.output[0].status == "incomplete"
        ok("max_output_tokens: status incomplete (reasoning, answer, streamed)")

        # Errors.
        def post(body, raw_body=None):
            content = raw_body if raw_body is not None else json.dumps(body)
            status, data = request(port, "POST", "/v1/responses", content)
            return status, json.loads(data)

        def bad(body, param=None, code=None, raw_body=None):
            status, data = post(body, raw_body)
            assert status == 400, (status, data)
            err = data["error"]
            assert err["type"] == "invalid_request_error" and err["message"], err
            assert set(err) == {"message", "type", "param", "code"}, err
            if param is not None:
                assert err["param"] == param, err
            if code is not None:
                assert err["code"] == code, err

        tools = [{"type": "function", "name": "f"}]
        bad({"model": "m", "input": "Hi", "tools": tools}, "tools", "unsupported_parameter")
        bad({"model": "m", "input": "Hi", "previous_response_id": "resp_1"}, "previous_response_id")
        bad({"model": "m", "input": "Hi", "tool_choice": "required"}, "tool_choice")
        bad({}, raw_body="{not json")
        bad([1, 2])
        bad({"input": "Hi"}, "model")
        bad({"model": "m"}, "input")
        bad({"model": "m", "input": []}, "input")
        bad({"model": "m", "input": [{"type": "function_call", "name": "f"}]}, "input[0].type")
        bad({"model": "m", "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]},
            "input[0].content[0].type")
        bad({"model": "m", "input": [{"role": "tool", "content": "x"}]}, "input[0].role")
        bad({"model": "m", "input": [{"role": "assistant", "content": "x"}]}, "input")
        bad({"model": "m", "input": "Hi", "reasoning": {"effort": "huge"}}, "reasoning.effort")
        bad({"model": "m", "input": "Hi", "temperature": 3}, "temperature")
        bad({"model": "m", "input": "Hi", "max_output_tokens": 0}, "max_output_tokens")
        bad({"model": "m", "input": "Hi", "max_output_tokens": 100000}, "max_output_tokens", "context_length_exceeded")
        bad({"model": "m", "input": "hello " * 5000}, "input", "context_length_exceeded")
        status, _ = post({"model": "m", "input": "Hi", "tools": [], "tool_choice": "auto", "previous_response_id": None,
                          "text": {"format": {"type": "text"}}, "user": "u", "store": True, "max_output_tokens": 3})
        assert status == 200
        try:
            client.responses.create(model="m", input="Hi", tools=[{"type": "function", "name": "f", "parameters": {}}])
            raise AssertionError("tools accepted")
        except openai.BadRequestError as error:
            assert error.status_code == 400 and error.body["param"] == "tools", error.body
        status, data = request(port, "GET", "/v1/nothing")
        assert status == 404 and json.loads(data)["error"]["message"]
        ok("errors: unsupported fields, bad JSON, bad input, context overflow, 404")

        # A client that disconnects mid-stream stops the generation; the server keeps serving.
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("POST", "/v1/responses", json.dumps({"model": "m", "input": "Hi", "stream": True}))
        res = conn.getresponse()
        assert res.status == 200 and res.getheader("Content-Type").startswith("text/event-stream")
        for _ in range(12):
            res.readline()
        conn.close()
        check_full(client.responses.create(model="m", input="Hi"))
        ok("client disconnect mid-stream")
    finally:
        server.terminate()
        server.wait()
        log.seek(0)
        text = log.read()
        log.close()
    print(text, end="")
    assert "finish=cancelled" in text, "disconnect not logged as cancelled"
    print(f"responses: all {checks} checks passed")


if __name__ == "__main__":
    main()
