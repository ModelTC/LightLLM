import asyncio
import json
from types import SimpleNamespace

import pytest

from lightllm.server import api_openai
from lightllm.server.api_models import ChatCompletionRequest, CompletionRequest
from lightllm.server.api_openai import (
    _build_completion_response,
    _collect_generation_results,
)
from lightllm.server.core.objs import FinishStatus
from lightllm.server.httpserver.manager import StopSequenceOutput


class _FinishStatus:
    def __init__(self, finished=False, reason=None):
        self.finished = finished
        self.reason = reason

    def is_finished(self):
        return self.finished

    def get_finish_reason(self):
        return self.reason


def test_collect_generation_results_keeps_choices_separate(monkeypatch):
    async def generate_results():
        metadata = {
            "prompt_tokens": 4,
            "prompt_cache_len": 1,
            "prompt_token_ids": [1, 2, 3, 4],
        }
        yield 82, "C", {**metadata, "logprob": -0.5, "id": 14}, _FinishStatus()
        yield 81, "B", {**metadata, "logprob": -0.2, "id": 11}, _FinishStatus()
        yield 80, "A", {**metadata, "logprob": -0.1, "id": 10}, _FinishStatus()
        yield 82, "3", {**metadata, "logprob": -0.6, "id": 15}, _FinishStatus(True, "length")
        yield 81, "2", {**metadata, "logprob": -0.4, "id": 13}, _FinishStatus(True, "length")
        yield 80, "1", {**metadata, "logprob": -0.3, "id": 12}, _FinishStatus(True, "length")

    request = CompletionRequest(
        model="test-model",
        prompt="Prompt",
        n=3,
        best_of=3,
        max_tokens=2,
        logprobs=1,
    )
    results = asyncio.run(_collect_generation_results(generate_results(), request, "Prompt"))

    assert [result["text"] for result in results] == ["A1", "B2", "C3"]
    assert [result["completion_tokens"] for result in results] == [2, 2, 2]
    assert [[token["id"] for token in result["token_infos"]] for result in results] == [
        [10, 12],
        [11, 13],
        [14, 15],
    ]

    from lightllm.server.api_http import g_objs

    monkeypatch.setattr(g_objs, "httpserver_manager", SimpleNamespace(tokenizer=None), raising=False)
    response = _build_completion_response([results], request, created_time=123, is_batch=False)

    assert [choice.index for choice in response.choices] == [0, 1, 2]
    assert [choice.text for choice in response.choices] == ["A1", "B2", "C3"]
    assert response.usage.prompt_tokens == 4
    assert response.usage.completion_tokens == 6
    assert response.usage.total_tokens == 10
    assert response.usage.prompt_tokens_details.cached_tokens == 1

    second_prompt_results = [
        {
            **result,
            "prompt_tokens": 5,
            "prompt_cache_len": 2,
            "prompt_text": "Another prompt",
        }
        for result in results
    ]
    batch_response = _build_completion_response(
        [results, second_prompt_results], request, created_time=123, is_batch=True
    )

    assert [choice.index for choice in batch_response.choices] == list(range(6))
    assert batch_response.usage.prompt_tokens == 9
    assert batch_response.usage.completion_tokens == 12
    assert batch_response.usage.total_tokens == 21
    assert batch_response.usage.prompt_tokens_details.cached_tokens == 3


def test_non_streaming_chat_usage_sums_all_choices(monkeypatch):
    async def generate_results():
        metadata = {"prompt_tokens": 4, "prompt_cache_len": 1}
        for sub_req_id, text in [(80, "A"), (81, "B"), (82, "C")]:
            yield sub_req_id, text, metadata, _FinishStatus()
        for sub_req_id, text in [(80, "1"), (81, "2"), (82, "3")]:
            yield sub_req_id, text, metadata, _FinishStatus(True, "length")

    class _SamplingParams:
        def init(self, **_kwargs):
            pass

        def verify(self):
            pass

    async def build_prompt(_request, _tools):
        return "Prompt"

    manager = SimpleNamespace(tokenizer=None, generate=lambda *_args, **_kwargs: generate_results())
    monkeypatch.setattr(api_openai, "SamplingParams", _SamplingParams)
    monkeypatch.setattr(api_openai, "build_prompt", build_prompt)
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: SimpleNamespace(reasoning_parser=None))

    from lightllm.server.api_http import g_objs

    monkeypatch.setattr(g_objs, "httpserver_manager", manager, raising=False)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        n=3,
        max_tokens=2,
    )

    response = asyncio.run(api_openai.chat_completions_impl(request, SimpleNamespace()))

    assert [choice.index for choice in response.choices] == [0, 1, 2]
    assert [choice.message.content for choice in response.choices] == ["A1", "B2", "C3"]
    assert response.usage.prompt_tokens == 4
    assert response.usage.completion_tokens == 6
    assert response.usage.total_tokens == 10
    assert response.usage.prompt_tokens_details.cached_tokens == 1


@pytest.mark.parametrize("api", ["chat", "completion"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("include", [None, False, True])
def test_include_stop_str_in_output_api(monkeypatch, api, stream, include):
    from lightllm.server.api_http import g_objs

    def generate(prompt, sampling_params, multimodal_params, request):
        assert sampling_params.include_stop_str_in_output is (include is True)
        assert sampling_params.stop_sequences.to_strings() == ["END"]

        async def results():
            output = StopSequenceOutput(sampling_params)
            metadata = {"prompt_tokens": 4, "prompt_cache_len": 0, "logprobs": {}}
            for index, text in enumerate(["helloE", "NDextra"], 1):
                finish = FinishStatus(FinishStatus.FINISHED_STOP if index == 2 else FinishStatus.NO_FINISH)
                token_metadata = {**metadata, "id": index, "logprob": -index}
                if index == 2:
                    token_metadata["_stop_output_offset"] = -5 if include else -8
                for result in output.process(80, text, token_metadata, finish):
                    yield result

        return results()

    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [ord(c) for c in text])
    monkeypatch.setattr(g_objs, "httpserver_manager", SimpleNamespace(tokenizer=tokenizer, generate=generate))
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: SimpleNamespace(reasoning_parser=None))

    async def build_prompt(*args):
        return "Prompt"

    monkeypatch.setattr(api_openai, "build_prompt", build_prompt)
    kwargs = {"include_stop_str_in_output": include} if include is not None else {}

    async def run():
        common = dict(model="test-model", stop=["END"], stream=stream, max_completion_tokens=10, **kwargs)
        if api == "chat":
            request = ChatCompletionRequest(messages=[{"role": "user", "content": "Hello"}], **common)
            response = await api_openai.chat_completions_impl(request, SimpleNamespace())
        else:
            request = CompletionRequest(prompt="Prompt", **common)
            response = await api_openai.completions_impl(request, SimpleNamespace())

        if stream:
            body = "".join(
                [chunk.decode() if isinstance(chunk, bytes) else chunk async for chunk in response.body_iterator]
            )
            events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
            choices = [choice for event in events for choice in event["choices"]]
            text = (
                "".join(choice["delta"].get("content") or "" for choice in choices)
                if api == "chat"
                else "".join(choice.get("text", "") for choice in choices)
            )
            assert choices[-1]["finish_reason"] == "stop"
        else:
            text = response.choices[0].message.content if api == "chat" else response.choices[0].text
            assert response.choices[0].finish_reason == "stop"
            assert response.usage.completion_tokens == 2
        assert text == ("helloEND" if include else "hello")

    asyncio.run(run())


@pytest.mark.parametrize("thinking", [False, True])
def test_stop_scope_receives_request_thinking_mode(monkeypatch, thinking):
    from lightllm.server.api_http import g_objs
    from lightllm.server import reasoning_parser

    def generate(prompt, sampling_params, multimodal_params, request):
        assert sampling_params._reasoning_status == int(thinking)

        async def results():
            yield 80, "hello", {"prompt_tokens": 4, "id": 1}, FinishStatus(FinishStatus.FINISHED_STOP)

        return results()

    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [ord(c) for c in text])
    monkeypatch.setattr(g_objs, "httpserver_manager", SimpleNamespace(tokenizer=tokenizer, generate=generate))
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: SimpleNamespace(reasoning_parser="qwen3"))
    monkeypatch.setattr(api_openai, "_is_force_thinking_mode", lambda request: thinking)
    monkeypatch.setattr(reasoning_parser, "get_token_id", lambda token: 1001)

    async def build_prompt(*args):
        return "Prompt"

    monkeypatch.setattr(api_openai, "build_prompt", build_prompt)
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello"}],
        stop=["END"],
        max_completion_tokens=10,
    )
    asyncio.run(api_openai.chat_completions_impl(request, SimpleNamespace()))
