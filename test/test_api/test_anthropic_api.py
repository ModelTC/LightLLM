#!/usr/bin/env python3
"""Manual integration test for the Anthropic API compatibility layer.

Requires:
  1. A running LightLLM server started with --enable_anthropic_api
  2. ``pip install anthropic``

Usage:
  python test/test_api/test_anthropic_api.py \
      --base-url http://localhost:8088 \
      --model my-local-model

Each assertion exits the script with non-zero status on failure so it can
be used as a CI gate once a GPU runner is available.
"""
from __future__ import annotations

import argparse
import sys


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def test_non_streaming_text(client, model: str) -> None:
    resp = client.messages.create(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
    )
    print("[non-stream]", resp)
    if resp.type != "message":
        _fail(f"expected type=message, got {resp.type}")
    if not resp.content or resp.content[0].type != "text":
        _fail(f"expected a text content block, got {resp.content}")
    if resp.stop_reason not in {"end_turn", "stop_sequence", "max_tokens"}:
        _fail(f"unexpected stop_reason: {resp.stop_reason}")
    if resp.usage.input_tokens <= 0 or resp.usage.output_tokens <= 0:
        _fail(f"unexpected usage: {resp.usage}")


def test_streaming_text(client, model: str) -> None:
    collected = []
    stop_reason = None
    with client.messages.stream(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
    ) as stream:
        for text in stream.text_stream:
            collected.append(text)
        final = stream.get_final_message()
        stop_reason = final.stop_reason

    full = "".join(collected)
    print(f"[stream] stop_reason={stop_reason!r} text={full!r}")
    if not full.strip():
        _fail("streaming produced no text")
    if stop_reason not in {"end_turn", "max_tokens"}:
        _fail(f"unexpected stop_reason after stream: {stop_reason}")


def test_system_prompt(client, model: str) -> None:
    resp = client.messages.create(
        model=model,
        max_tokens=32,
        system="Always reply with exactly the word: banana",
        messages=[{"role": "user", "content": "What fruit?"}],
    )
    text = resp.content[0].text if resp.content else ""
    print(f"[system] text={text!r}")
    if "banana" not in text.lower():
        print(f"WARN: system prompt may not be routed — got {text!r}", file=sys.stderr)


def test_tool_use(client, model: str) -> None:
    resp = client.messages.create(
        model=model,
        max_tokens=256,
        tools=[
            {
                "name": "get_weather",
                "description": "Return the current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    )
    print(f"[tool] stop_reason={resp.stop_reason} content={resp.content}")
    tool_blocks = [b for b in resp.content if b.type == "tool_use"]
    if resp.stop_reason == "tool_use" and not tool_blocks:
        _fail("stop_reason=tool_use but no tool_use content block")
    if tool_blocks:
        tb = tool_blocks[0]
        if tb.name != "get_weather":
            _fail(f"unexpected tool name: {tb.name}")
        if not isinstance(tb.input, dict):
            _fail(f"tool input is not a dict: {tb.input!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8088")
    parser.add_argument("--model", default="default")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["non_stream", "stream", "system", "tool"],
        help="Tests to skip",
    )
    args = parser.parse_args()

    try:
        import anthropic
    except ImportError:
        _fail("anthropic SDK not installed. Run: pip install anthropic")
        return

    client = anthropic.Anthropic(base_url=args.base_url, api_key=args.api_key)

    if "non_stream" not in args.skip:
        test_non_streaming_text(client, args.model)
    if "stream" not in args.skip:
        test_streaming_text(client, args.model)
    if "system" not in args.skip:
        test_system_prompt(client, args.model)
    if "tool" not in args.skip:
        test_tool_use(client, args.model)

    print("\nAll selected tests passed.")


if __name__ == "__main__":
    main()
