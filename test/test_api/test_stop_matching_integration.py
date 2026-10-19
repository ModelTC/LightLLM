"""真实服务回归：LIGHTLLM_STOP_TEST_URLS=http://127.0.0.1:8000[,PD_MASTER_URL] pytest -q 本文件。

使用 Qwen3 思维链标记的模型，正常服务及 P/D 节点和 Master 均开启 --enable_prompt_logprobs。
服务采用默认 model_name；先确保 P/D 的注册和 KV 状态上报连接均已就绪。
"""

import json
import os

import pytest
import requests


@pytest.fixture(scope="module", params=os.getenv("LIGHTLLM_STOP_TEST_URLS", "").split(","))
def server_url(request):
    if not request.param:
        pytest.skip("Set LIGHTLLM_STOP_TEST_URLS to test running normal/PD services")
    return request.param.rstrip("/")


def _native(server_url, parameters, prompt="Count from one to ten: 1, 2,", stream=False):
    response = requests.post(
        server_url + ("/generate_stream" if stream else "/generate"),
        json={
            "inputs": prompt,
            "parameters": {"do_sample": False, "max_new_tokens": 30, "return_details": True, **parameters},
        },
        timeout=60,
        stream=stream,
    )
    response.raise_for_status()
    if not stream:
        return response.json()
    with response:
        events = [json.loads(line[5:]) for line in response.iter_lines() if line.startswith(b"data:")]
    assert events and events[-1]["finished"], events
    assert sum(event["finished"] for event in events) == 1
    return {
        "generated_text": ["".join(event["token"]["text"] for event in events)],
        "tokens": [event["token"] for event in events],
        "finish_reason": events[-1]["finish_reason"],
        **{key: events[-1][key] for key in ("prompt_token_ids", "prompt_logprobs") if key in events[-1]},
    }


@pytest.fixture(scope="module")
def baseline(server_url):
    result = _native(server_url, {})
    assert len(result["tokens"]) > 6
    assert result["finish_reason"] == "length"
    return result


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("kind", ["string", "ids", "mixed"])
@pytest.mark.parametrize("stream", [False, True])
def test_native_stop_scope_and_token_boundaries(server_url, baseline, mode, include, kind, stream):
    tokens = baseline["tokens"]
    if kind == "string":
        stop = "".join(token["text"] for token in tokens[4:6])
        position = baseline["generated_text"][0].index(stop)
        expected = baseline["generated_text"][0][: position + (len(stop) if include else 0)]
    else:
        stop = [[token["id"] for token in tokens[4:6]]]
        if kind == "mixed":
            stop.append("UNMATCHED_STOP")
        expected = "".join(token["text"] for token in tokens[:6])
    result = _native(
        server_url,
        {"stop_sequences": stop, "stop_str_match_mode": mode, "include_stop_str_in_output": include},
        stream=stream,
    )
    assert result["generated_text"] == [expected]
    assert result["finish_reason"] == "stop"
    assert [token["id"] for token in result["tokens"]] == [token["id"] for token in tokens[:6]]


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("stream", [False, True])
def test_id_stop_on_prefill_first_token(server_url, baseline, mode, stream):
    first = baseline["tokens"][0]
    result = _native(server_url, {"stop_sequences": [[first["id"]]], "stop_str_match_mode": mode}, stream=stream)
    assert [token["id"] for token in result["tokens"]] == [first["id"]]
    assert result["generated_text"] == [first["text"]]
    assert result["finish_reason"] == "stop"


@pytest.mark.parametrize("stream", [False, True])
def test_prompt_logprobs_survive_backend_stop(server_url, baseline, stream):
    stop = [[token["id"] for token in baseline["tokens"][4:6]]]
    result = _native(server_url, {"stop_sequences": stop, "prompt_logprobs": 0}, stream=stream)
    assert result["finish_reason"] == "stop"
    assert len(result["prompt_token_ids"]) == len(result["prompt_logprobs"])
    assert result["prompt_logprobs"][0] is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("api", ["completions", "chat/completions"])
def test_openai_multiple_choices_stop_independently(server_url, stream, api):
    payload = {"model": "default_model_name", "temperature": 0, "max_tokens": 30, "n": 3, "stream": stream}
    if api == "completions":
        payload.update(prompt="Count from one to ten: 1, 2,", stop="4,", best_of=3)
    else:
        payload.update(
            messages=[{"role": "user", "content": "Repeat exactly: START END AFTER"}],
            stop="END",
            chat_template_kwargs={"enable_thinking": False},
        )
    response = requests.post(server_url + "/v1/" + api, json=payload, timeout=60, stream=stream)
    response.raise_for_status()
    if stream:
        texts = {index: "" for index in range(3)}
        finishes = {}
        with response:
            for line in response.iter_lines():
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                assert "error" not in chunk, chunk
                for choice in chunk.get("choices", []):
                    index = choice["index"]
                    texts[index] += choice.get("text", choice.get("delta", {}).get("content") or "")
                    if choice.get("finish_reason"):
                        assert index not in finishes
                        finishes[index] = choice["finish_reason"]
    else:
        choices = response.json()["choices"]
        texts = {choice["index"]: choice.get("text", choice.get("message", {}).get("content")) for choice in choices}
        finishes = {choice["index"]: choice["finish_reason"] for choice in choices}
    assert set(texts) == set(finishes) == {0, 1, 2}
    assert set(finishes.values()) == {"stop"}
    assert all(payload["stop"] not in text for text in texts.values())
    assert all(text for text in texts.values())


def test_openai_batch_prompts_keep_stop_and_choice_ids_independent(server_url):
    response = requests.post(
        server_url + "/v1/completions",
        json={
            "model": "default_model_name",
            "prompt": ["Count from one to ten: 1, 2,", "Count from one to ten: 2, 3,"],
            "temperature": 0,
            "max_tokens": 30,
            "n": 2,
            "best_of": 2,
            "stop": "6,",
        },
        timeout=60,
    )
    response.raise_for_status()
    choices = response.json()["choices"]
    assert {choice["index"] for choice in choices} == {0, 1, 2, 3}
    assert all(choice["finish_reason"] == "stop" and "6," not in choice["text"] for choice in choices)
    assert choices[0]["text"] != choices[2]["text"]


@pytest.mark.parametrize(
    "force,marker,in_reasoning", [(0, "<think>", True), (1, "</think>", True), (0, "</think>", False)]
)
def test_prompt_marker_pretransition_only_applies_to_content(server_url, force, marker, in_reasoning):
    prompt = "<|im_start|>user\nWhat is 2 + 3? Think briefly.<|im_end|>\n<|im_start|>assistant\n" + marker
    params = {"max_new_tokens": 80, "skip_special_tokens": False, "_initial_reasoning_state": force}
    baseline = _native(server_url, params, prompt)
    # 模型可能立刻结束思考；只取首个非空 token，验证它是否受初始阶段保护。
    stop_text = next(token["text"] for token in baseline["tokens"] if token["text"])
    result = _native(server_url, {**params, "stop_str_match_mode": 2, "stop_sequences": stop_text}, prompt)
    if in_reasoning:
        assert result["generated_text"][0].startswith(stop_text)
    else:
        assert result["generated_text"] == [""]
        assert result["finish_reason"] == "stop"


@pytest.mark.parametrize("stream", [False, True])
def test_ids_stop_inside_reasoning_while_strings_skip_it(server_url, stream):
    prompt = (
        "<|im_start|>user\nWhat is 2 + 3? Think briefly and then answer with a digit.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    parameters = {"max_new_tokens": 256, "skip_special_tokens": False, "_initial_reasoning_state": 1}
    baseline = _native(server_url, parameters, prompt)
    tokens = baseline["tokens"]
    assert "</think>" in baseline["generated_text"][0], baseline["generated_text"]
    assert all("</think>" not in token["text"] for token in tokens[:3])
    stop_ids = [token["id"] for token in tokens[:3]]
    stop_text = "".join(token["text"] for token in tokens[:3])
    result = _native(server_url, {**parameters, "stop_str_match_mode": 2, "stop_sequences": [stop_ids]}, prompt, stream)
    assert [token["id"] for token in result["tokens"]] == stop_ids
    assert result["finish_reason"] == "stop"
    result = _native(
        server_url, {**parameters, "stop_str_match_mode": 2, "stop_sequences": [stop_text]}, prompt, stream
    )
    assert result["generated_text"][0].startswith(stop_text)
    assert "</think>" in result["generated_text"][0]
