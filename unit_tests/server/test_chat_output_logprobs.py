import json
import pickle
import uuid
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast

from lightllm.server import api_http, api_openai, build_prompt as prompt_builder, reasoning_parser
from lightllm.server.api_models import ChatCompletionRequest
from lightllm.server.core.objs import FinishStatus, Req
from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.server.httpserver_for_pd_master import manager as pd_master


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize(
    "fields",
    [
        {"top_logprobs": 0},
        {"logprobs": None, "top_logprobs": 1},
        {"logprobs": True, "top_logprobs": -1},
        {"logprobs": True, "top_logprobs": 21},
        {"logprobs": True, "top_logprobs": 1.5},
        {"logprobs": True, "top_logprobs": True},
    ],
)
def test_invalid_chat_logprobs(fields):
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[], **fields)


@pytest.mark.parametrize(
    "fields,topk",
    [
        ({}, -1),
        ({"logprobs": None}, -1),
        ({"logprobs": True}, 0),
        ({"logprobs": True, "top_logprobs": 0}, 0),
        ({"logprobs": True, "top_logprobs": 20}, 20),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("separate_reasoning", [False, True])
@pytest.mark.anyio
async def test_chat_logprobs_response(monkeypatch, fields, topk, stream, separate_reasoning):
    # The first two ByteLevel tokens complete their UTF-8 character only when
    # the third arrives. Think markers and the suppressed EOS are invisible.
    vocab = {"<think>": 0, "think": 1, "</think>": 2, "ä": 3, "½": 4, "ł": 5, "Ġyes": 6, "<eos>": 7}
    backend = Tokenizer(models.BPE(vocab, []))
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>")

    async def build_prompt(*args):
        return "prompt"

    async def generate(prompt, params, *args, **kwargs):
        assert params.output_top_logprobs == topk
        assert pickle.loads(pickle.dumps(params)).output_top_logprobs == topk
        for token_id, delta in enumerate(["<think>", "think", "</think>", "", "", "你", " yes", ""]):
            metadata = {
                "id": token_id,
                "logprob": -0.25,
                "prompt_tokens": 10,
                "output_top_logprobs": [(6, -0.125)] if topk > 0 else [],
            }
            status = FinishStatus(FinishStatus.FINISHED_STOP if token_id == 7 else FinishStatus.NO_FINISH)
            yield 0, delta, metadata, status

    monkeypatch.setattr(api_openai, "build_prompt", build_prompt)
    monkeypatch.setattr(api_openai, "get_env_start_args", lambda: SimpleNamespace(reasoning_parser="deepseek-v4"))
    monkeypatch.setattr(prompt_builder, "tokenizer_supports_force_thinking", lambda: True)
    monkeypatch.setattr(reasoning_parser, "get_token_id", tokenizer.convert_tokens_to_ids)
    monkeypatch.setattr(
        api_http.g_objs,
        "httpserver_manager",
        SimpleNamespace(tokenizer=tokenizer, generate=generate, output_grammar_compiler=None),
        raising=False,
    )
    monkeypatch.setattr(api_http.g_objs, "args", SimpleNamespace(tool_call_parser="deepseekv4"), raising=False)
    request = ChatCompletionRequest(messages=[], stream=stream, separate_reasoning=separate_reasoning, **fields)
    response = await api_openai.chat_completions_impl(request, SimpleNamespace())
    if stream:
        choices = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode()
            for line in chunk.splitlines():
                if line.startswith("data: {"):
                    choices.extend(json.loads(line[6:])["choices"])
    else:
        choices = response.model_dump()["choices"]
    entries = [choice["logprobs"] for choice in choices if choice["logprobs"] is not None]
    if topk < 0:
        assert entries == []
        return
    reasoning = [entry for part in entries for entry in (part["reasoning_content"] or [])]
    content = [entry for part in entries for entry in (part["content"] or [])]
    assert [entry["token"] for entry in reasoning] == (["think"] if separate_reasoning else [])
    expected_content = "你 yes" if separate_reasoning else "think你 yes"
    assert bytes(byte for entry in content for byte in entry["bytes"]).decode() == expected_content
    assert len(content) == (4 if separate_reasoning else 5)
    assert all(entry["logprob"] == -0.25 for entry in content + reasoning)
    expected = [{"token": " yes", "logprob": -0.125, "bytes": [32, 121, 101, 115]}] if topk > 0 else []
    assert all(entry["top_logprobs"] == expected for entry in content + reasoning)


def test_candidate_shm_is_reused_and_resized(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", f"logprob-test-{uuid.uuid4().hex}")
    owner, worker = Req(), Req()
    for req in (owner, worker):
        req.alloc_shm_numpy_len = 8
        req.sample_params.output_top_logprobs = 2
        req.sample_params.max_new_tokens = 3
    owner.create_logprobs_shm_array()
    try:
        worker.link_logprobs_shm_array()
        assert worker.shm_logprobs.arr.shape == (8,)
        assert worker.shm_output_topk_logprobs.arr.shape == (3 + 128, 2, 2)
        assert worker.shm_output_topk_logprobs.shm.size == worker.shm_output_topk_logprobs.arr.nbytes
        assert worker.output_topk_logprobs_row(0)[0].tolist() == [-1, -1]
        worker.shm_logprobs.arr[3] = (-0.25, 1)
        worker.output_topk_logprobs_row(0)[:] = ([9, 8], [-0.5, -1.0])
        worker.output_topk_logprobs_row(1)[:] = ([7, 6], [-0.25, -1.5])
        worker.detach_shm_arrays()

        owner.detach_shm_arrays()
        owner.link_logprobs_shm_array()
        assert owner.shm_logprobs.arr[3].tolist() == (-0.25, 1)
        assert owner.output_topk_logprobs_row(1).tolist() == [[7, 6], [-0.25, -1.5]]

        owner.detach_shm_arrays()
        owner.create_logprobs_shm_array()
        assert owner.output_topk_logprobs_row(0)[0].tolist() == [-1, -1]
        owner.shm_output_topk_logprobs.close_shm()

        owner.sample_params.output_top_logprobs = 0
        owner.create_logprobs_shm_array()
        assert owner.shm_logprobs.arr.shape == (8,)
        assert owner.shm_output_topk_logprobs.shm is None
        assert owner.shm_logprobs.arr[3].tolist() == (0.0, -1)
    finally:
        worker.detach_shm_arrays()
        topk_shm = getattr(owner, "shm_output_topk_logprobs", None)
        if topk_shm is not None:
            topk_shm.close_shm()
        owner.shm_logprobs.close_shm()


@pytest.mark.anyio
@pytest.mark.parametrize("needs_prefill", [True, False])
@pytest.mark.parametrize("end_status", [FinishStatus.FINISHED_LENGTH, FinishStatus.FINISHED_PD_DECODE_CAPACITY])
async def test_pd_first_token_uses_the_node_that_sampled_it(monkeypatch, needs_prefill, end_status):
    prompt_ids = [1, 2, 3]
    params = SamplingParams()
    params.max_new_tokens = 10
    params.output_top_logprobs = 2
    params.group_request_id = 0
    finish = FinishStatus()
    d_first = (0, "first", {"count_output_tokens": 1, "node_mode": "decode", "output_top_logprobs": []}, finish)
    p_first = (
        0,
        "first",
        {"count_output_tokens": 1, "node_mode": "prefill", "output_top_logprobs": [(1, -0.5)]},
        finish,
    )
    d_next = (0, "second", {"count_output_tokens": 2}, FinishStatus(end_status))
    decode_info = SimpleNamespace(ready_kv_len=1 if needs_prefill else len(prompt_ids) - 1)
    batches = deque([[d_first, d_next], [p_first]] if needs_prefill else [[d_first, d_next]])
    status = SimpleNamespace(
        prefill_prompt_ids_event=SimpleNamespace(prompt_ids=prompt_ids),
        up_status_event=SimpleNamespace(upkv_status=SimpleNamespace(pd_kv_trans_params=pickle.dumps(decode_info))),
        wait_to_ready=AsyncMock(),
        raise_if_error=lambda: None,
        can_read=AsyncMock(return_value=True),
        pop_all_tokens=AsyncMock(side_effect=batches.popleft),
        put_tokens_to_front=AsyncMock(side_effect=batches.appendleft),
    )
    monkeypatch.setattr(pd_master, "ReqStatus", lambda *args: status)
    manager = pd_master.HttpServerManagerForPDMaster.__new__(pd_master.HttpServerManagerForPDMaster)
    manager.args = SimpleNamespace(pd_node_id=1)
    manager.req_id_to_out_inf = {}
    manager._wait_for_event_or_disconnect = AsyncMock()
    node = SimpleNamespace(websocket=SimpleNamespace(send_bytes=AsyncMock()))
    request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
    stream = pd_master.HttpServerManagerForPDMaster.fetch_pd_stream(
        manager, node, node, prompt_ids, params, {}, request
    )
    try:
        _, text, metadata, _ = await stream.__anext__()
        assert text == "first"
        assert metadata["output_top_logprobs"] == ([(1, -0.5)] if needs_prefill else [])
        _, text, _, status = await stream.__anext__()
        assert text == "second" and status.status == end_status
    finally:
        await stream.aclose()
