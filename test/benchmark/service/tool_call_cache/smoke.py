"""End-to-end correctness probes; run under exp with the service manifest saved."""

import argparse
import json
import urllib.request


def post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18235")
    parser.add_argument("--model", default="tool-cache-qwen4b")
    parser.add_argument("--prefix-words", type=int, default=0)
    parser.add_argument("--cache-granularity", type=int, default=1)
    args = parser.parse_args()
    tool = {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo the supplied text.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
    }
    messages = [
        {
            "role": "user",
            "content": "Reference:"
            + " ledger" * args.prefix_words
            + "\nCall echo with text hello. Use the tool, then wait for its result.",
        }
    ]
    first = post(
        args.url + "/v1/chat/completions",
        {
            "model": args.model,
            "messages": messages,
            "tools": [tool],
            "temperature": 0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    choice = first["choices"][0]
    assert choice["finish_reason"] == "tool_calls", first
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "echo", first
    messages.append(choice["message"])
    messages.append({"role": "tool", "tool_call_id": choice["message"]["tool_calls"][0]["id"], "content": "hello"})
    messages.append({"role": "user", "content": "Now give the final answer in one word. Do not call any tools."})
    second = post(
        args.url + "/v1/chat/completions",
        {
            "model": args.model,
            "messages": messages,
            "tools": [tool],
            "temperature": 0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert second["choices"][0]["finish_reason"] == "stop", second
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] >= (
        first["usage"]["prompt_tokens"] - args.cache_granularity - 16
    ), second
    payload = {
        "model": args.model,
        "messages": messages[:1],
        "tools": [tool],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        args.url + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    tool_chunks = 0
    stream_reason = None
    stream_usage = None
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            event = json.loads(line[5:])
            if event.get("usage"):
                stream_usage = event["usage"]
            for choice in event.get("choices", []):
                tool_chunks += len(choice.get("delta", {}).get("tool_calls") or [])
                stream_reason = choice.get("finish_reason") or stream_reason
    assert stream_reason == "tool_calls" and tool_chunks > 0 and stream_usage, (
        stream_reason,
        tool_chunks,
        stream_usage,
    )
    print(
        json.dumps(
            {
                "tool_round": first,
                "final_round": second,
                "stream_tool_chunks": tool_chunks,
                "stream_finish_reason": stream_reason,
                "stream_usage": stream_usage,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
