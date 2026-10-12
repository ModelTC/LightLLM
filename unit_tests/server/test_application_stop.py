import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightllm.server.api_lightllm import lightllm_generate, lightllm_generate_stream
from lightllm.server.api_tgi import tgi_generate_impl, tgi_generate_stream_impl
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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("include", [False, True])
def test_tgi_api_uses_shared_stop_wrapper(stream, include):
    async def run():
        async def raw_results(*args, **kwargs):
            metadata = {"prompt_tokens": 2, "logprobs": {}, "logprob": -0.1, "cumlogprob": -0.1}
            yield 80, "helloE", {**metadata, "id": 10}, FinishStatus()
            yield 80, "NDextra", {**metadata, "id": 11}, FinishStatus()
            yield 80, "late", {**metadata, "id": 12}, FinishStatus(FinishStatus.FINISHED_LENGTH)

        manager = _manager(
            SimpleNamespace(encode=lambda text, **kwargs: [ord(c) for c in text]),
            SimpleNamespace(reasoning_parser=None, enable_rl=False, enable_return_routed_experts=False),
            raw_results,
        )
        request = SimpleNamespace(
            json=AsyncMock(
                return_value={
                    "inputs": "prompt",
                    "parameters": {"stop": ["END"], "details": True, "include_stop_str_in_output": include},
                }
            )
        )
        response = await (tgi_generate_stream_impl if stream else tgi_generate_impl)(request, manager)
        expected = "helloEND" if include else "hello"
        if stream:
            events = []
            async for chunk in response.body_iterator:
                chunk = chunk.decode() if isinstance(chunk, bytes) else chunk
                events.append(json.loads(chunk.split(":", 1)[1]))
            assert "".join(event["token"]["text"] for event in events) == expected
            assert events[-1]["finish_reason"] == "stop"
        else:
            data = json.loads(response.body)
            data = data[0] if isinstance(data, list) else data
            assert data["generated_text"] == expected
            assert data["details"]["finish_reason"] == "stop"
        manager.stop_request.assert_awaited_once_with(80)

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
@pytest.mark.parametrize("stop", [None, "END", [[10]], ["END", [10]]])
def test_rl_and_final_metadata_skip_both_stop_wrappers(mode, feature, stop):
    async def run():
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [ord(c) for c in text])
        params = SamplingParams()
        params.init(tokenizer, stop_sequences=stop, str_stop_match_mode=mode)
        if feature == "prompt_logprobs":
            params.prompt_logprobs = 0
        metadata = {"id": 11, "prompt_token_ids": [1, 2], "prompt_logprobs": [None, {}], "routed_experts": [[2]]}

        async def raw_results(*args, **kwargs):
            # 第一条输出就命中 stop；最终 metadata 只在后端结束输出中出现。
            yield 80, "helloEND", {"id": 10}, FinishStatus()
            yield 80, "tail", metadata, FinishStatus(FinishStatus.FINISHED_STOP)

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
        assert len(results) == 2
        assert results[0][1] == "helloEND"
        assert not results[0][3].is_finished()
        assert results[1][1] == "tail"
        assert results[1][2] is metadata
        assert results[1][3].is_stopped()
        manager.stop_request.assert_not_awaited()

    asyncio.run(run())
