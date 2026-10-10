"""Exercise structured output on a running server with overlap enabled.

Example: python test/format_out/validate_structured_output.py \
    --base-url http://127.0.0.1:28881 --model qwen35_27b
Use --chunked_prefill_size 256 on the server to exercise multiple prompt chunks.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import count
import json
from pathlib import Path
import sys
from typing import Literal

from pydantic import BaseModel
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from format_out.impl import ChatSession, SamplingParams


request_ids = count()
request_id_prefix = "structured-output"


class HelperResult(BaseModel):
    difficulty: Literal["easy", "hard"]
    thoughts: list[str]
    answer: str


def post(base_url, path, payload, stream=False):
    constrained = "response_format" in payload or any(
        payload.get("parameters", {}).get(key) for key in ("guided_json", "guided_grammar", "regular_constraint")
    )
    kind = "grammar" if constrained else "plain"
    response = requests.post(
        base_url + path,
        json=payload,
        timeout=120,
        stream=stream,
        headers={"X-Request-Id": f"{request_id_prefix}-{kind}-{next(request_ids)}"},
    )
    assert response.status_code == 200, (response.status_code, response.text)
    return response


def chat_payload(model, prompt, response_format=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 256,
        "temperature": 0,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    return payload


def schema_request(base_url, model, index, stream=False, long_prompt=False, thinking=False):
    schema = {
        "type": "object",
        "properties": {
            "id": {"const": index % 4},
            "status": {"enum": ["ok", "好"]},
            "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        },
        "required": ["id", "status", "items"],
        "additionalProperties": False,
    }
    prompt = f'Return JSON with id {index % 4}, status "好", and items [1, 2].'
    if long_prompt:
        prompt = "This is background context. " * 180 + prompt
    payload = chat_payload(model, prompt, {"type": "json_schema", "json_schema": {"name": "result", "schema": schema}})
    if thinking is None:
        payload.pop("chat_template_kwargs")  # Exercise the model's default thinking mode.
    else:
        payload["chat_template_kwargs"]["enable_thinking"] = thinking
    if thinking is not False:
        payload["messages"][0]["content"] = "Think briefly without repeating your analysis. " + prompt
        payload["max_tokens"] = 3072
        payload["stream_reasoning"] = True
    # Exercise both stochastic and greedy sampling, sharing compiled schemas.
    # Completed-reasoning cases use greedy decoding; stochastic reasoning can
    # legitimately consume its whole budget before reaching the answer.
    if index % 2 and thinking is False:
        payload.update(temperature=0.7, top_p=0.9)
    payload["stream"] = stream
    response = post(base_url, "/v1/chat/completions", payload, stream=stream)
    if stream:
        parts = []
        reasoning = []
        finished = False
        for line in response.iter_lines():
            if not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            event = json.loads(data)
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                parts.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning") or delta.get("reasoning_content") or "")
                if choice.get("finish_reason") is not None:
                    assert choice["finish_reason"] == "stop", choice
                    finished = True
        response.close()
        assert finished
        value = json.loads("".join(parts))
        if thinking is not False:
            assert "".join(reasoning), "No reasoning returned before the constrained answer"
    else:
        data = response.json()
        assert data["choices"][0]["finish_reason"] == "stop", data
        value = json.loads(data["choices"][0]["message"]["content"])
        if thinking is not False:
            message = data["choices"][0]["message"]
            assert message.get("reasoning") or message.get("reasoning_content"), data
            assert data["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0, data
        if long_prompt:
            assert data["usage"]["prompt_tokens"] > 512, data["usage"]
    assert set(value) == {"id", "status", "items"}, value
    assert value["id"] == index % 4 and value["status"] in ("ok", "好"), value
    assert len(value["items"]) == 2 and all(type(item) is int for item in value["items"]), value


def plain_request(base_url, model):
    data = post(base_url, "/v1/chat/completions", chat_payload(model, "Reply with the word hello.")).json()
    assert data["choices"][0]["message"]["content"], data


def native_request(base_url, **parameters):
    return post(
        base_url,
        "/generate",
        {"inputs": "Answer briefly.", "parameters": {"max_new_tokens": 32, "do_sample": False, **parameters}},
    ).json()


def main():
    global request_id_prefix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:28881")
    parser.add_argument("--model", default="qwen35_27b")
    parser.add_argument("--request-id-prefix", default="structured-output")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    request_id_prefix = args.request_id_prefix

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for index in range(16):
            futures.append(pool.submit(schema_request, base_url, args.model, index, index % 3 == 0, index == 2))
            if index % 4 == 0:
                futures.append(pool.submit(plain_request, base_url, args.model))
        for future in futures:
            future.result()
    print("PASS: 20 mixed concurrent requests, schema reuse, streaming, and chunked prefill", flush=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(schema_request, base_url, args.model, index, index % 2 == 0, index == 0, thinking)
            for index, thinking in enumerate([True, False, None, True, False, True, None, False])
        ]
        for future in futures:
            future.result()
    print("PASS: mixed thinking on/off/default, shared schemas, streaming, and chunked prefill", flush=True)

    payload = chat_payload(args.model, 'Return a JSON object with a key "ok".', {"type": "json_object"})
    data = post(base_url, "/v1/chat/completions", payload).json()
    assert data["choices"][0]["finish_reason"] == "stop", data
    assert isinstance(json.loads(data["choices"][0]["message"]["content"]), dict), data
    print("PASS: json_object", flush=True)

    payload["chat_template_kwargs"]["enable_thinking"] = True
    payload["messages"][0]["content"] = 'Think briefly, then return a JSON object with a key "ok".'
    payload["max_tokens"] = 3072
    data = post(base_url, "/v1/chat/completions", payload).json()
    assert data["choices"][0]["finish_reason"] == "stop", data
    assert isinstance(json.loads(data["choices"][0]["message"]["content"]), dict), data
    assert data["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0, data
    print("PASS: thinking followed by json_object", flush=True)

    payload["max_tokens"] = 1
    data = post(base_url, "/v1/chat/completions", payload).json()
    assert data["choices"][0]["finish_reason"] == "length", data
    assert not data["choices"][0]["message"]["content"], data
    schema_request(base_url, args.model, 0, thinking=True)
    print("PASS: truncation during thinking, then complete another constrained request", flush=True)

    for constraint in ({"regular_constraint": "OK"}, {"guided_grammar": 'root ::= "OK"'}):
        data = native_request(base_url, exponential_decay_length_penalty=[1, 2.0], **constraint)
        assert data["generated_text"] == ["OK"] and data["finish_reason"] == "stop", data
    data = native_request(base_url, guided_json={"const": {"中文": '引号"和换行\n'}})
    assert json.loads(data["generated_text"][0]) == {"中文": '引号"和换行\n'}, data
    print("PASS: regex, EBNF, Unicode schema, and EOS penalty", flush=True)

    data = native_request(base_url, regular_constraint="OK", ignore_eos=True)
    assert data["generated_text"] == ["OK"] and data["finish_reason"] == "stop", data
    data = native_request(base_url, regular_constraint="abcdefghijklmnopqrstuvwxyz", max_new_tokens=1)
    assert data["finish_reason"] == "length", data
    print("PASS: grammar termination with ignore_eos, and output-length truncation", flush=True)

    session = ChatSession(
        chat_his='What is 1 + 1? Return JSON with difficulty "easy", thoughts [], and answer "2".',
        sampling_param=SamplingParams(do_sample=False),
        url=base_url + "/generate",
        disable_log=True,
    )
    output = session.gen_json_object(HelperResult, max_new_tokens=256, prefix_regex=r"\s{0,20}")
    HelperResult.model_validate_json(output)
    print("PASS: format_out helper with schema-expanded EBNF and whitespace prefix", flush=True)

    for parameters in (
        {"regular_constraint": "["},
        {"guided_json": "not json"},
        {"guided_grammar": "bad grammar"},
    ):
        response = requests.post(base_url + "/generate", json={"inputs": "test", "parameters": parameters}, timeout=30)
        assert response.status_code == 400, (parameters, response.status_code, response.text)
    print("PASS: malformed constraints rejected", flush=True)

    payload = chat_payload(
        args.model, "Return a JSON object containing a very long list of numbers.", {"type": "json_object"}
    )
    payload["stream"] = True
    with post(base_url, "/v1/chat/completions", payload, stream=True) as response:
        for line in response.iter_lines():
            if line.startswith(b"data: "):
                break
    schema_request(base_url, args.model, 0)
    plain_request(base_url, args.model)
    print("PASS: cancel stream, then complete constrained and ordinary requests", flush=True)


if __name__ == "__main__":
    main()
