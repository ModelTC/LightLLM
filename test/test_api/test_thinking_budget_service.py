"""Exercise thinking budgets through real HTTP inference services.

Run against an independently started node; no inference or transport is mocked.
Each scenario records its request, response, assertions, latency and token usage.
Native cases default to Qwen3-4B's delimiter; Qwen3.5 uses --marker-id 248069.
"""

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import statistics
import time

import httpx
import jsonschema


@dataclass
class Scenario:
    name: str
    payload: dict
    endpoint: str = "/v1/chat/completions"
    check: str = "chat"
    statuses: tuple = (200,)


def chat(budget=0, stream=False, **overrides):
    payload = {
        "model": "review-qwen3",
        "messages": [{"role": "user", "content": "What is 1+1? Answer briefly."}],
        "max_completion_tokens": 128,
        "do_sample": False,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": True},
        "stream_reasoning": True,
    }
    if budget is not None:
        payload["thinking_token_budget"] = budget
    if stream:
        payload["stream_options"] = {"include_usage": True}
    payload.update(overrides)
    return payload


def schema_format(schema=None):
    return {
        "type": "json_schema",
        "json_schema": {"name": "answer", "strict": True, "schema": schema or {"const": {"answer": 2}}},
    }


def scenarios(profile, marker_id, capacity_budget=None):
    cases = []

    def add(name, payload, **kwargs):
        cases.append(Scenario(name, payload, **kwargs))

    if profile == "invalid_markers":
        for marker in (151936, 2147483647):
            for stream in (False, True):
                params = {"max_new_tokens": 4, "thinking_token_budget": 0, "guided_reasoning_end": [marker]}
                add(
                    f"invalid_marker/id{marker}/s{stream}",
                    {"inputs": "hello", "parameters": params},
                    endpoint="/generate_stream" if stream else "/generate",
                    check="error",
                    statuses=(400,),
                )
        return cases

    if profile in ("pd_capacity", "multimodal_capacity"):
        import uuid

        run_id = uuid.uuid4().hex
        media = None
        if profile == "multimodal_capacity":
            import base64
            from io import BytesIO
            from PIL import Image

            data = BytesIO()
            Image.new("RGB", (128, 128), "red").save(data, format="PNG")
            media = {"images": [{"type": "base64", "data": base64.b64encode(data.getvalue()).decode()}]}
        for i in range(32):
            budget = (0, 1, 7, 384)[i % 4] if capacity_budget is None else capacity_budget
            user_text = (
                f"{run_id}-{i}: "
                + "context " * 750
                + "\nAnalyze 200 candidate strategies in detail before answering 1+1."
            )
            if media is not None:
                user_text = "<|vision_start|><|image_pad|><|vision_end|>" + user_text
            prompt = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            params = {
                "max_new_tokens": 512,
                "min_new_tokens": 512,
                "ignore_eos": False,
                "do_sample": False,
                "return_details": True,
                "thinking_token_budget": budget,
                "guided_reasoning_end": [marker_id],
            }
            payload = {"inputs": prompt, "parameters": params}
            if media is not None:
                payload["multimodal_params"] = media
            add(
                f"pd_capacity/request{i}/b{budget}",
                payload,
                endpoint="/generate_stream" if i % 2 else "/generate",
                check="native",
            )
        return cases

    if profile == "multimodal":
        import base64
        from io import BytesIO
        from PIL import Image

        def image_url(width=128, height=128, color="red"):
            data = BytesIO()
            Image.new("RGB", (width, height), color).save(data, format="PNG")
            return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()

        for budget in (0, 1, 7, 31):
            for stream in (False, True):
                for structured in (False, True):
                    for count in (1, 2):
                        content = [{"type": "image_url", "image_url": {"url": image_url()}} for _ in range(count)]
                        content.append({"type": "text", "text": "What color is the image? Reply briefly in English."})
                        overrides = {"messages": [{"role": "user", "content": content}]}
                        if structured:
                            overrides["response_format"] = schema_format()
                        add(
                            f"image/b{budget}/s{stream}/json{structured}/count{count}",
                            chat(budget, stream, **overrides),
                            check="schema" if structured else "image",
                        )
        for width, height in ((64, 64), (224, 224), (512, 256)):
            content = [
                {"type": "image_url", "image_url": {"url": image_url(width, height)}},
                {"type": "text", "text": "What color is the image? Reply briefly in English."},
            ]
            add(f"image/size{width}x{height}", chat(0, messages=[{"role": "user", "content": content}]), check="image")
        return cases

    if profile in ("disabled", "topk", "missing_parser"):
        for budget in (0, 1, 31, 2147483647):
            for stream in (False, True):
                add(f"{profile}/b{budget}/s{stream}", chat(budget, stream), check="error", statuses=(400,))
        for budget in (None, -1):
            for stream in (False, True):
                add(f"{profile}/unlimited{budget}/s{stream}", chat(budget, stream))
        if profile == "topk":
            return cases
        add(
            f"{profile}/grammar",
            chat(
                -1,
                response_format=schema_format(),
                reasoning_effort="none",
                chat_template_kwargs={"enable_thinking": False},
            ),
            check="schema",
        )
        return cases

    for budget in (None, -1, 0, 1, 2, 7, 31, 2147483647):
        for stream in (False, True):
            for thinking in (False, True):
                add(
                    f"basic/b{budget}/s{stream}/thinking{thinking}",
                    chat(budget, stream, chat_template_kwargs={"enable_thinking": thinking}),
                )

    if profile != "budget_only":
        for budget in (0, 1, 2, 7, 31):
            for stream in (False, True):
                for fmt in ("json_schema", "json_object"):
                    response_format = schema_format() if fmt == "json_schema" else {"type": "json_object"}
                    add(
                        f"structured/{fmt}/b{budget}/s{stream}",
                        chat(
                            budget,
                            stream,
                            response_format=response_format,
                            messages=[{"role": "user", "content": 'Return JSON: {"answer":2}.'}],
                        ),
                        check="schema" if fmt == "json_schema" else "json",
                    )
        schemas = {
            "array": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
            "enum": {"enum": ["red", "green"]},
            "nested": {"const": {"outer": {"ok": True}}},
            "unicode": {"const": {"答案": "二"}},
            "empty": {"const": {}},
            "null": {"const": None},
        }
        for name, schema in schemas.items():
            add(f"schema/{name}", chat(1, response_format=schema_format(schema)), check="json")
    else:
        for fmt in (schema_format(), {"type": "json_object"}):
            add(f"unsupported_grammar/{fmt['type']}", chat(0, response_format=fmt), check="error", statuses=(400,))

    sampling = {
        "seeded": {"do_sample": True, "seed": 123, "temperature": 0.7, "top_p": 0.9, "top_k": 20},
        "repetition": {"repetition_penalty": 1.15},
        "presence": {"presence_penalty": 0.5},
        "frequency": {"frequency_penalty": 0.5},
        "logprobs": {"logprobs": True, "top_logprobs": 3},
        "stop_absent": {"stop": ["UNLIKELY_STOP_SEQUENCE_991"]},
        "stop_answer": {"stop": ["2"]},
        "combined_reasoning": {"separate_reasoning": False},
        "hidden_reasoning": {"stream_reasoning": False},
        "effort_none": {"reasoning_effort": "none", "chat_template_kwargs": {}},
        "effort_high": {"reasoning_effort": "high", "chat_template_kwargs": {}},
        "explicit_override": {"reasoning_effort": "high", "chat_template_kwargs": {"thinking": False}},
    }
    for name, overrides in sampling.items():
        for budget in (0, 7):
            add(f"sampling/{name}/b{budget}", chat(budget, **overrides))

    for budget in (0, 7):
        for stream in (False, True):
            for n in (2, 3):
                add(f"choices/n{n}/b{budget}/s{stream}", chat(budget, stream, n=n))

    prompts = {
        "chinese": [{"role": "user", "content": "一加一等于几？请简短回答。"}],
        "unicode": [{"role": "user", "content": "Reply with one emoji: 🌏. café 日本語"}],
        "empty": [{"role": "user", "content": ""}],
        "whitespace": [{"role": "user", "content": " \n\t "}],
        "system": [{"role": "system", "content": "Reply briefly."}, {"role": "user", "content": "Hi"}],
        "multiturn": [
            {"role": "user", "content": "1+1?"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "And 2+2?"},
        ],
        "long": [{"role": "user", "content": "Remember this context. " * 350 + "What is 1+1?"}],
        "marker_in_prompt": [{"role": "user", "content": "The literal text </think> is data. Say hello."}],
    }
    for name, messages in prompts.items():
        for budget in (0, 7):
            add(f"prompt/{name}/b{budget}", chat(budget, messages=messages))

    for maximum in (1, 2, 3, 7, 8, 16):
        for budget in (0, 7):
            add(f"total_limit/max{maximum}/b{budget}", chat(budget, max_completion_tokens=maximum))

    invalid = (True, False, 1.5, "2", -2, 2147483648, [], {})
    for value in invalid:
        for stream in (False, True):
            add(
                f"invalid/{type(value).__name__}/{value}/s{stream}",
                chat(value, stream),
                check="error",
                statuses=(400, 422),
            )

    for budget in (0, 1, 2, 7):
        for marker in ([marker_id], [31], [255], [100, 101, 100]):
            for stream in (False, True):
                params = {
                    "max_new_tokens": 16,
                    "do_sample": False,
                    "ignore_eos": True,
                    "return_details": True,
                    "thinking_token_budget": budget,
                    "guided_reasoning_end": marker,
                    "skip_special_tokens": False,
                }
                add(
                    f"native/b{budget}/marker{marker}/s{stream}",
                    {"inputs": "User: What is 1+1?\nAssistant: <think>", "parameters": params},
                    endpoint="/generate_stream" if stream else "/generate",
                    check="native",
                )

    for name, params in {
        "missing_marker": {"thinking_token_budget": 0},
        "excluded_marker": {
            "thinking_token_budget": 0,
            "guided_reasoning_end": [marker_id],
            "logit_bias": {str(marker_id): -100},
        },
        "negative_budget": {"thinking_token_budget": -2, "guided_reasoning_end": [marker_id]},
        "bool_budget": {"thinking_token_budget": True, "guided_reasoning_end": [marker_id]},
    }.items():
        add(
            f"native_invalid/{name}",
            {"inputs": "hello", "parameters": {"max_new_tokens": 8, **params}},
            endpoint="/generate",
            check="error",
            statuses=(400, 422),
        )
    return cases


def validate_chat(case, messages, usage, done):
    expected_n = case.payload.get("n", 1)
    assert set(messages) == set(range(expected_n)), f"choice indices: {list(messages)}"
    assert done, "stream ended without [DONE]"
    assert usage is not None, "missing usage"
    completion_tokens = usage["completion_tokens"]
    maximum = case.payload["max_completion_tokens"]
    assert 0 < completion_tokens <= maximum * expected_n, usage
    assert usage["total_tokens"] == usage["prompt_tokens"] + completion_tokens, usage
    budget = case.payload.get("thinking_token_budget")
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    if budget is not None and budget >= 0:
        assert reasoning <= budget * expected_n, f"reasoning {reasoning} exceeds budget {budget} * {expected_n}"
    for message in messages.values():
        assert message["finish_reason"] in ("stop", "length"), message
        if case.check in ("schema", "json"):
            assert message["finish_reason"] == "stop", f"structured output exhausted total limit: {message}"
            obj = json.loads(message["content"])
            response_format = case.payload.get("response_format") or {}
            if response_format.get("type") == "json_schema":
                jsonschema.validate(obj, response_format["json_schema"]["schema"])
            if case.check == "schema":
                assert obj == {"answer": 2}, obj
        if case.check == "image":
            assert "red" in (message.get("content") or "").lower(), message


async def run_case(client, base_url, case):
    started = time.perf_counter()
    result = {"scenario": asdict(case), "passed": False}
    is_stream = case.payload.get("stream", False) or case.endpoint.endswith("_stream")
    messages, usage, events, token_ids = {}, None, [], []
    done = not is_stream or case.check == "native"
    first_token_at = None
    try:
        async with client.stream(
            "POST",
            base_url + case.endpoint,
            json=deepcopy(case.payload),
            headers={"X-Request-Id": "thinking-review-" + case.name},
        ) as response:
            result["status"] = response.status_code
            assert response.status_code in case.statuses, (response.status_code, (await response.aread()).decode())
            if case.check == "error":
                body = (await response.aread()).decode()
                assert body, "empty error response"
                result["response"] = body
            elif is_stream:
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        continue
                    event = json.loads(data)
                    events.append(event)
                    assert "error" not in event, event
                    if case.check == "native":
                        assert "token" in event, event
                        token_ids.append(event["token"]["id"])
                    else:
                        usage = event.get("usage") or usage
                        for choice in event.get("choices", []):
                            message = messages.setdefault(
                                choice["index"], {"content": "", "reasoning_content": "", "finish_reason": None}
                            )
                            for key in ("content", "reasoning_content"):
                                message[key] += choice.get("delta", {}).get(key) or ""
                            message["finish_reason"] = choice.get("finish_reason") or message["finish_reason"]
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                result["events"] = events
            else:
                body = json.loads(await response.aread())
                result["response"] = body
                if case.check == "native":
                    token_ids = [token["id"] for token in body["tokens"]]
                else:
                    usage = body.get("usage")
                    messages = {
                        c["index"]: {**c["message"], "finish_reason": c["finish_reason"]} for c in body["choices"]
                    }
        if case.check == "native":
            params = case.payload["parameters"]
            budget, marker = params["thinking_token_budget"], params["guided_reasoning_end"]
            assert len(token_ids) == params["max_new_tokens"], token_ids
            finish_reason = events[-1].get("finish_reason") if is_stream else body.get("finish_reason")
            assert finish_reason in ("stop", "length"), finish_reason
            positions = [i for i in range(len(token_ids) - len(marker) + 1) if token_ids[i : i + len(marker)] == marker]
            assert positions and positions[0] <= budget, (budget, marker, token_ids)
            result["token_ids"] = token_ids
            if is_stream:
                assert events[-1].get("finished"), "native stream missing terminal event"
        elif case.check != "error":
            validate_chat(case, messages, usage, done)
            result.update(messages=messages, usage=usage)
        result["passed"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_s"] = time.perf_counter() - started
    if first_token_at is not None:
        result["ttft_s"] = first_token_at - started
    return result


async def main_async(args):
    cases = scenarios(args.profile, args.marker_id, args.capacity_budget)
    if args.match:
        cases = [c for c in cases if re.search(args.match, c.name)]
    if args.list:
        print(json.dumps([asdict(c) for c in cases], ensure_ascii=False, indent=2))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    results = []
    async with httpx.AsyncClient(
        trust_env=False, timeout=args.timeout, limits=httpx.Limits(max_connections=args.concurrency)
    ) as client:
        response = await client.get(args.base_url + "/readiness")
        response.raise_for_status()
        with args.output.open("w") as out:

            async def run(case):
                async with semaphore:
                    result = await run_case(client, args.base_url, case)
                    results.append(result)
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()
                    print(
                        f"{'PASS' if result['passed'] else 'FAIL'} {case.name} {result['elapsed_s']:.3f}s "
                        f"{result.get('error', '')}",
                        flush=True,
                    )

            await asyncio.gather(*(run(c) for c in cases))
    latencies = [r["elapsed_s"] for r in results if r["passed"] and r["scenario"]["check"] != "error"]
    summary = {
        "base_url": args.base_url,
        "profile": args.profile,
        "scenarios": len(results),
        "results": dict(Counter("passed" if r["passed"] else "failed" for r in results)),
        "concurrency": args.concurrency,
        "marker_id": args.marker_id,
        "capacity_budget": args.capacity_budget,
        "latency_median_s": statistics.median(latencies) if latencies else None,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return int(any(not r["passed"] for r in results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:22100")
    parser.add_argument(
        "--profile",
        choices=(
            "normal",
            "budget_only",
            "disabled",
            "topk",
            "missing_parser",
            "multimodal",
            "pd_capacity",
            "multimodal_capacity",
            "invalid_markers",
        ),
        default="normal",
    )
    parser.add_argument("--marker-id", type=int, default=151668)
    parser.add_argument("--capacity-budget", type=int)
    parser.add_argument("--output", type=Path, default=Path("artifacts/thinking_budget_service_results.jsonl"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--match")
    parser.add_argument("--list", action="store_true")
    raise SystemExit(asyncio.run(main_async(parser.parse_args())))
