import asyncio
from types import SimpleNamespace

from lightllm.server import api_openai
from lightllm.server.api_models import ChatCompletionRequest, CompletionRequest
from lightllm.server.api_openai import (
    _build_completion_response,
    _collect_generation_results,
)


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
    sampling_params = SimpleNamespace(stop_sequences=SimpleNamespace(size=0))

    results = asyncio.run(_collect_generation_results(generate_results(), request, "Prompt", sampling_params))

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
