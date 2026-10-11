import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightllm.server.api_lightllm import lightllm_generate, lightllm_generate_stream
from lightllm.server.core.objs import FinishStatus, SamplingParams, StrStopMatchMode
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.pd_io_struct import NodeRole


def _manager(tokenizer, args, raw_results):
    manager = HttpServerManager.__new__(HttpServerManager)
    manager.tokenizer = tokenizer
    manager.args = args
    manager.pd_mode = NodeRole.NORMAL
    manager.is_multinode_tp_slave = False
    manager._generate = raw_results
    manager.stop_request = AsyncMock()
    return manager


@pytest.mark.parametrize("stream", [False, True])
def test_native_api_discards_output_after_application_stop(stream):
    async def run():
        closed = []
        exhausted = []

        async def raw_results(*args, **kwargs):
            metadata = {"prompt_tokens": 2, "prompt_cache_len": 0, "logprobs": {}, "logprob": -0.1}
            try:
                yield 80, "helloE", {**metadata, "id": 10}, FinishStatus()
                yield 80, "NDextra", {**metadata, "id": 11}, FinishStatus()
                yield 80, "late", {**metadata, "id": 12}, FinishStatus(FinishStatus.FINISHED_LENGTH)
                exhausted.append(True)
            finally:
                closed.append(True)

        manager = _manager(
            SimpleNamespace(encode=lambda text, **kwargs: [ord(c) for c in text]),
            SimpleNamespace(reasoning_parser=None, enable_rl=False, enable_return_routed_experts=False),
            raw_results,
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
        assert exhausted == [True]

    asyncio.run(run())


@pytest.mark.parametrize("mode", [NodeRole.NORMAL, NodeRole.P, NodeRole.D])
@pytest.mark.parametrize("slave", [False, True])
def test_http_generate_only_wraps_client_output(mode, slave):
    async def run():
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [ord(c) for c in text])
        params = SamplingParams()
        params.init(tokenizer, stop_sequences="END", max_new_tokens=10)
        websocket, pd_event = object(), object()

        async def raw_results(prompt, sampling_params, multimodal, request, upload, event):
            assert upload is websocket
            assert event is pd_event
            yield 80, "helloENDextra", {"id": 10}, FinishStatus(FinishStatus.FINISHED_STOP)

        manager = _manager(
            tokenizer,
            SimpleNamespace(reasoning_parser=None, enable_rl=False, enable_return_routed_experts=False),
            raw_results,
        )
        manager.pd_mode = mode
        manager.is_multinode_tp_slave = slave
        results = [result async for result in manager.generate("prompt", params, None, None, websocket, pd_event)]
        expected = "hello" if mode == NodeRole.NORMAL and not slave else "helloENDextra"
        assert "".join(result[1] for result in results) == expected
        manager.stop_request.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize(
    "mode", [StrStopMatchMode.DETOKEN_ALL, StrStopMatchMode.OUTPUT_ALL, StrStopMatchMode.OUTPUT_CONTENT_ONLY]
)
@pytest.mark.parametrize("feature", ["enable_rl", "enable_return_routed_experts", "prompt_logprobs"])
@pytest.mark.parametrize("stop", [None, "END"])
def test_rl_and_final_metadata_use_backend_stop_only(mode, feature, stop):
    async def run():
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [ord(c) for c in text])
        params = SamplingParams()
        params.init(tokenizer, stop_sequences=stop, str_stop_match_mode=mode)
        if feature == "prompt_logprobs":
            params.prompt_logprobs = 0
        metadata = {"id": 10, "prompt_token_ids": [1, 2], "routed_experts": [[2]]}

        async def raw_results(*args, **kwargs):
            yield 80, "helloEND", metadata, FinishStatus(FinishStatus.FINISHED_STOP)

        manager = _manager(
            tokenizer,
            SimpleNamespace(
                reasoning_parser=None,
                enable_rl=feature == "enable_rl",
                enable_return_routed_experts=feature == "enable_return_routed_experts",
            ),
            raw_results,
        )
        if mode != StrStopMatchMode.DETOKEN_ALL:
            with pytest.raises(AssertionError, match="DETOKEN_ALL"):
                manager.generate("prompt", params, None, None)
            return
        results = [result async for result in manager.generate("prompt", params, None, None)]
        assert len(results) == 1
        assert results[0][1] == "helloEND"
        assert results[0][2] is metadata
        manager.stop_request.assert_not_awaited()

    asyncio.run(run())
