import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightllm.server.api_lightllm import lightllm_generate, lightllm_generate_stream
from lightllm.server.core.objs import FinishStatus, SamplingParams, StopMatchMode
from lightllm.server.generation import generate_with_stop


@pytest.mark.parametrize("stream", [False, True])
def test_native_api_stops_without_consuming_later_metadata(stream):
    async def run():
        closed = []

        async def raw_results(*args, **kwargs):
            metadata = {"prompt_tokens": 2, "prompt_cache_len": 0, "logprobs": {}, "logprob": -0.1}
            try:
                yield 80, "helloE", {**metadata, "id": 10}, FinishStatus()
                yield 80, "NDextra", {**metadata, "id": 11}, FinishStatus()
                pytest.fail("stop must finish without reading subsequent tokens or metadata")
            finally:
                closed.append(True)

        manager = SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda text, **kwargs: [ord(c) for c in text]),
            args=SimpleNamespace(reasoning_parser=None, enable_return_routed_experts=False),
            generate=raw_results,
            stop_request=AsyncMock(),
        )
        request = SimpleNamespace(
            json=AsyncMock(
                return_value={
                    "inputs": "prompt",
                    "parameters": {"stop_sequences": ["END"], "max_new_tokens": 10},
                }
            )
        )
        response = await (lightllm_generate_stream if stream else lightllm_generate)(request, manager)
        if stream:
            events = [json.loads(chunk.decode().split(":", 1)[1]) async for chunk in response.body_iterator]
            assert "".join(event["token"]["text"] for event in events) == "hello"
            assert events[-1]["finished"]
            assert events[-1]["finish_reason"] == "stop"
        else:
            data = json.loads(response.body)
            assert data["generated_text"] == ["hello"]
            assert data["count_output_tokens"] == 2
            assert data["finish_reason"] == "stop"
        manager.stop_request.assert_awaited_once_with(80)
        assert closed == [True]

    asyncio.run(run())


@pytest.mark.parametrize(
    "mode", [StopMatchMode.BACKEND_ALL, StopMatchMode.OUTPUT_ALL, StopMatchMode.OUTPUT_CONTENT_ONLY]
)
@pytest.mark.parametrize("feature", ["enable_rl", "enable_return_routed_experts", "prompt_logprobs"])
@pytest.mark.parametrize("stop", [None, "END"])
def test_rl_and_final_metadata_use_backend_stop_only(mode, feature, stop):
    async def run():
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [ord(c) for c in text])
        params = SamplingParams()
        params.init(tokenizer, stop_sequences=stop, stop_match_mode=mode)
        if feature == "prompt_logprobs":
            params.prompt_logprobs = 0
        metadata = {"id": 10, "prompt_token_ids": [1, 2], "routed_experts": [[2]]}

        async def raw_results(*args, **kwargs):
            yield 80, "helloEND", metadata, FinishStatus(FinishStatus.FINISHED_STOP)

        manager = SimpleNamespace(
            tokenizer=tokenizer,
            args=SimpleNamespace(
                reasoning_parser=None,
                enable_rl=feature == "enable_rl",
                enable_return_routed_experts=feature == "enable_return_routed_experts",
            ),
            generate=raw_results,
            stop_request=AsyncMock(),
        )
        if mode != StopMatchMode.BACKEND_ALL:
            with pytest.raises(AssertionError, match="BACKEND_ALL"):
                generate_with_stop(manager, "prompt", params, None, None)
            return
        results = [result async for result in generate_with_stop(manager, "prompt", params, None, None)]
        assert len(results) == 1
        assert results[0][1] == "helloEND"
        assert results[0][2] is metadata
        manager.stop_request.assert_not_awaited()

    asyncio.run(run())
