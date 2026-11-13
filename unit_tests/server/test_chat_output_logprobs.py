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
        assert worker.output_topk_logprobs_row(output_index=0)[0].tolist() == [-1, -1]
        worker.shm_logprobs.arr[3] = (-0.25, 1)
        worker.output_topk_logprobs_row(output_index=0)[:] = ([9, 8], [-0.5, -1.0])
        worker.output_topk_logprobs_row(output_index=1)[:] = ([7, 6], [-0.25, -1.5])
        worker.detach_shm_arrays()

        owner.detach_shm_arrays()
        owner.link_logprobs_shm_array()
        assert owner.shm_logprobs.arr[3].tolist() == (-0.25, 1)
        assert owner.output_topk_logprobs_row(output_index=1).tolist() == [[7, 6], [-0.25, -1.5]]

        owner.detach_shm_arrays()
        owner.create_logprobs_shm_array()
        assert owner.output_topk_logprobs_row(output_index=0)[0].tolist() == [-1, -1]
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
    d_first = (
        0,
        "first",
        {"count_output_tokens": 1, "node_mode": "decode", "output_top_logprobs": [], "logprob_prompt_ids": prompt_ids},
        finish,
    )
    p_first = (
        0,
        "first",
        {
            "count_output_tokens": 1,
            "node_mode": "prefill",
            "output_top_logprobs": [(1, -0.5)],
            "logprob_prompt_ids": prompt_ids,
        },
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
        assert metadata["logprob_prompt_ids"] == prompt_ids
        assert metadata["output_top_logprobs"] == ([(1, -0.5)] if needs_prefill else [])
        _, text, _, status = await stream.__anext__()
        assert text == "second" and status.status == end_status
    finally:
        await stream.aclose()


@pytest.fixture
def chat_response(monkeypatch):
    async def run(
        tokenizer,
        pieces,
        *,
        stream,
        separate_reasoning=True,
        model_type=None,
        tools=None,
        topk=0,
        rank=2,
        tool_parser="qwen25",
        scores=None,
        first_metadata=None,
        candidates=None,
        logprobs=True,
        return_choices=False,
    ):
        async def build_prompt(*args):
            return "prompt"

        async def generate(prompt, params, *args, **kwargs):
            for index, (token_id, delta) in enumerate(pieces):
                yield 0, delta, {
                    "id": token_id,
                    **(first_metadata or {} if index == 0 else {}),
                    "logprob": scores[index] if scores is not None else -2.401099,
                    "logprobs": {token_id: {"rank": rank}},
                    "prompt_tokens": 10,
                    "output_top_logprobs": candidates if candidates is not None else [(0, -0.125)] * topk,
                }, FinishStatus(FinishStatus.FINISHED_LENGTH if index == len(pieces) - 1 else FinishStatus.NO_FINISH)

        monkeypatch.setattr(api_openai, "build_prompt", build_prompt)
        monkeypatch.setattr(api_openai, "get_env_start_args", lambda: SimpleNamespace(reasoning_parser=model_type))
        monkeypatch.setattr(prompt_builder, "tokenizer_supports_force_thinking", lambda: True)
        monkeypatch.setattr(reasoning_parser, "get_token_id", tokenizer.convert_tokens_to_ids)
        monkeypatch.setattr(
            api_http.g_objs,
            "httpserver_manager",
            SimpleNamespace(
                tokenizer=tokenizer,
                generate=generate,
                output_grammar_compiler=None,
            ),
            raising=False,
        )
        monkeypatch.setattr(api_http.g_objs, "args", SimpleNamespace(tool_call_parser=tool_parser), raising=False)
        request = ChatCompletionRequest(
            messages=[],
            stream=stream,
            separate_reasoning=separate_reasoning,
            logprobs=logprobs,
            top_logprobs=topk if logprobs else None,
            tools=tools,
        )
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
        if return_choices:
            return choices
        messages = [choice["delta" if stream else "message"] for choice in choices]
        logprobs = [choice["logprobs"] for choice in choices if choice["logprobs"]]
        return (
            "".join(message.get("content") or "" for message in messages),
            "".join(message.get("reasoning_content") or "" for message in messages),
            [token for entry in logprobs for token in entry["content"] or []],
            [token for entry in logprobs for token in entry["reasoning_content"] or []],
        )

    return run


def make_tokenizer(pieces, decoder=None):
    backend = Tokenizer(models.BPE(dict(zip(pieces, range(len(pieces)))), []))
    if decoder is not None:
        backend.decoder = decoder
    return PreTrainedTokenizerFast(tokenizer_object=backend)


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("topk", [0, 1, 3, 20])
@pytest.mark.parametrize("rank", [2, 20, 21])
async def test_chat_sampled_score_uses_fixed_rank_cutoff(chat_response, stream, topk, rank):
    tokenizer = make_tokenizer(["answer"])
    _, _, content, _ = await chat_response(tokenizer, [(0, "answer")], stream=stream, topk=topk, rank=rank)
    assert content[0]["logprob"] == (-9999.0 if rank > 20 else -2.401099)
    assert len(content[0]["top_logprobs"]) == topk


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
async def test_byte_fallback_keeps_every_generated_token(chat_response, stream, sequence):
    decoder = decoders.ByteFallback()
    if sequence:
        decoder = decoders.Sequence([decoder, decoders.Fuse()])
    tokenizer = make_tokenizer(["<0xE4>", "<0xBD>", "<0xA0>"], decoder)
    text, _, content, _ = await chat_response(tokenizer, [(0, ""), (1, ""), (2, "你")], stream=stream)
    assert text == "你"
    assert len(content) == 3
    assert [entry["bytes"] for entry in content] == [[0xE4], [0xBD], [0xA0]]
    assert bytes(byte for entry in content for byte in entry["bytes"]).decode() == text


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "decoder",
    [
        decoders.Metaspace(),
        decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(" ", 1, 0),
            ]
        ),
    ],
)
async def test_sentencepiece_word_spaces(chat_response, stream, decoder):
    tokenizer = make_tokenizer(["▁Hello", "▁world"], decoder)
    text, _, content, _ = await chat_response(tokenizer, [(0, "Hello"), (1, " world")], stream=stream)
    assert text == "Hello world"
    assert [token["token"] for token in content] == ["Hello", " world"]
    assert bytes(byte for token in content for byte in token["bytes"]).decode() == text


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("split_marker", [False, True])
async def test_tool_logprobs_only_contain_message_content(chat_response, stream, split_marker):
    pieces = [
        "Checking.",
        "<tool_call>",
        '{"name":',
        '"weather",',
        '"arguments":',
        '{"city":',
        '"Paris"}}',
        "</tool_call>",
    ]
    if split_marker:
        pieces[1:2] = ["<tool", "_call>"]
    tokenizer = make_tokenizer(pieces, decoders.Fuse())
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
    text, _, content, reasoning = await chat_response(
        tokenizer,
        list(enumerate(pieces)),
        stream=stream,
        tools=tools,
    )
    assert text == "Checking."
    assert [entry["token"] for entry in content] == ["Checking."]
    assert reasoning == []


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("separate_reasoning", [False, True])
@pytest.mark.parametrize("split_header", [False, True])
async def test_harmony_headers_are_not_visible_logprobs(chat_response, stream, separate_reasoning, split_header):
    pieces = [
        "<|channel|>",
        "analysis",
        "<|message|>",
        "Thinking",
        "<|end|>",
        "<|start|>",
        "assistant",
        "<|channel|>",
        "final",
        "<|message|>",
        "Answer",
        "<|return|>",
    ]
    if split_header:
        pieces[5:6] = ["<|sta", "rt|>"]
    tokenizer = make_tokenizer(list(dict.fromkeys(pieces)), decoders.Fuse())
    text, reasoning_text, content, reasoning = await chat_response(
        tokenizer,
        [(tokenizer.convert_tokens_to_ids(piece), piece) for piece in pieces],
        stream=stream,
        separate_reasoning=separate_reasoning,
        model_type="gpt-oss",
    )
    assert text == ("Answer" if separate_reasoning else "ThinkingAnswer")
    assert [entry["token"] for entry in content] == (["Answer"] if separate_reasoning else ["Thinking", "Answer"])
    assert [entry["token"] for entry in reasoning] == (["Thinking"] if separate_reasoning else [])


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("separate_reasoning", [False, True])
async def test_harmony_repeated_header_words_are_real_content(chat_response, stream, separate_reasoning):
    pieces = [
        "<|channel|>",
        "analysis",
        "<|message|>",
        "analysis",
        "<|end|>",
        "<|start|>",
        "assistant",
        "<|channel|>",
        "final",
        "<|message|>",
        "final",
        "<|return|>",
    ]
    tokenizer = make_tokenizer(list(dict.fromkeys(pieces)), decoders.Fuse())
    text, _, content, reasoning = await chat_response(
        tokenizer,
        [(tokenizer.convert_tokens_to_ids(piece), piece) for piece in pieces],
        stream=stream,
        separate_reasoning=separate_reasoning,
        model_type="gpt-oss",
    )
    assert text == ("final" if separate_reasoning else "analysisfinal")
    assert [entry["token"] for entry in content] == (["final"] if separate_reasoning else ["analysis", "final"])
    assert [entry["token"] for entry in reasoning] == (["analysis"] if separate_reasoning else [])


@pytest.mark.anyio
@pytest.mark.parametrize("valid_tool", [False, True])
async def test_nonstream_tool_logprobs_follow_actual_extraction(chat_response, valid_tool):
    pieces = ["Checking.", '<tool_call>{"name":"weather","arguments":{"city":"Paris"}}']
    if valid_tool:
        pieces += ["</tool_call>"]
    tokenizer = make_tokenizer(pieces, decoders.Fuse())
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
    text, _, content, _ = await chat_response(tokenizer, list(enumerate(pieces)), stream=False, tools=tools)
    assert text == ("Checking." if valid_tool else "".join(pieces))
    assert [entry["token"] for entry in content] == (pieces[:1] if valid_tool else pieces)


def test_unknown_token_bytes_do_not_drop_empty_increments():
    from lightllm.server.chat_logprob_buffer import ChatLogprobBuffer

    tokenizer = SimpleNamespace(
        decode=lambda ids, **kwargs: "\ufffd" if len(ids) == 1 else "你",
        convert_ids_to_tokens=lambda token_id: "unknown-byte",
    )
    buffer = ChatLogprobBuffer(tokenizer, separate_reasoning=True)
    for token_id, text in enumerate(["", "", "你"]):
        span = buffer.record(text, {"id": token_id, "logprob": -0.25}, FinishStatus())
        buffer.assign([span], len(text))
    content, _ = buffer.message_logprobs()
    assert len(content) == 3
    assert all(token["bytes"] is None for token in content)


def test_combined_reasoning_and_content_token_is_only_returned_once(monkeypatch):
    from lightllm.server.chat_logprob_buffer import ChatLogprobBuffer

    pieces = ["<think>", "Thinking</think>Answer"]
    tokenizer = make_tokenizer(pieces, decoders.Fuse())
    monkeypatch.setattr(reasoning_parser, "get_token_id", tokenizer.convert_tokens_to_ids)
    buffer = ChatLogprobBuffer(tokenizer, separate_reasoning=True)
    parser = reasoning_parser.ReasoningParser("qwen3", force_reasoning=False)
    for token_id, text in enumerate(pieces):
        buffer.record(text, {"id": token_id, "logprob": -0.25}, FinishStatus())
        _, content, spans = parser.parse_stream_chunk_with_spans(text)
        buffer.assign(spans, len(content))
    content, reasoning = buffer.message_logprobs()
    assert content is None
    assert [token["token"] for token in reasoning] == pieces[1:]


def test_reasoning_span_api_preserves_literal_delimiters_and_flush(monkeypatch):
    monkeypatch.setattr(reasoning_parser, "get_token_id", lambda token: 0)
    parser = reasoning_parser.ReasoningParser("qwen3", force_reasoning=False)
    text = "literal </think>"
    reasoning, content, spans = parser.parse_stream_chunk_with_spans(text)
    assert not reasoning and content == text
    assert "".join(text[span.start : span.end] for span in spans) == text
    parser = reasoning_parser.ReasoningParser("qwen3", force_reasoning=True)
    assert parser.parse_stream_chunk_with_spans("</")[2] == []
    reasoning, content, spans = parser.flush_with_spans()
    assert reasoning == "</" and not content
    assert [(span.start, span.end, span.channel) for span in spans] == [(0, 2, "reasoning")]


@pytest.mark.anyio
@pytest.mark.parametrize("prefix", ["", "Paris"])
async def test_tool_body_uses_generated_position(chat_response, prefix):
    pieces = ([prefix] if prefix else []) + [
        '<|python_tag|>{"name":"weather","arguments":{"city":"',
        "Paris",
        '"}};',
        "Paris",
    ]
    tokenizer = make_tokenizer(list(dict.fromkeys(pieces)), decoders.Fuse())
    scores = [-0.1 * (index + 1) for index in range(len(pieces))]
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
    text, _, content, _ = await chat_response(
        tokenizer,
        [(tokenizer.convert_tokens_to_ids(piece), piece) for piece in pieces],
        stream=False,
        tools=tools,
        tool_parser="llama3",
        scores=scores,
    )
    assert text == prefix + "Paris"
    assert [item["token"] for item in content] == ([prefix] if prefix else []) + ["Paris"]
    assert [item["logprob"] for item in content] == ([scores[0]] if prefix else []) + [scores[-1]]


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("separate_reasoning", [False, True])
async def test_nonstream_reasoning_uses_message_trimming(chat_response, stream, separate_reasoning):
    pieces = ["Thinking", "</think>", "\n\n", "Answer"]
    tokenizer = make_tokenizer(pieces, decoders.Fuse())
    text, reasoning_text, content, reasoning = await chat_response(
        tokenizer,
        list(enumerate(pieces)),
        stream=stream,
        separate_reasoning=separate_reasoning,
        model_type="qwen3",
        scores=[-0.1, -0.2, -0.3, -0.4],
    )
    expected = (["Thinking"] if not separate_reasoning else []) + (["\n\n"] if stream else []) + ["Answer"]
    assert text == "".join(expected)
    assert [item["token"] for item in content] == expected
    assert [item["logprob"] for item in content] == (
        ([-0.1] if not separate_reasoning else []) + ([-0.3] if stream else []) + [-0.4]
    )
    assert bytes(byte for item in content for byte in item["bytes"]).decode() == text
    assert [item["token"] for item in reasoning] == (["Thinking"] if separate_reasoning else [])


async def first_token_metadata(prompt_ids, output_index=1):
    import asyncio
    from unittest.mock import Mock
    from lightllm.server.httpserver.manager import HttpServerManager

    manager = HttpServerManager.__new__(HttpServerManager)
    manager.is_multinode_tp_slave = False
    manager.first_time_costs = Mock()
    manager.latest_success_infer_time_mark = Mock()
    status = SimpleNamespace(
        event=asyncio.Event(),
        lock=asyncio.Lock(),
        has_timed_out_waiting_for_inference=lambda: False,
        out_token_info_list=[(0, "", {"count_output_tokens": output_index}, FinishStatus())],
    )
    status.event.set()
    params = SamplingParams()
    params.best_of = 1
    params.output_top_logprobs = 1
    generator = manager._wait_to_token_package(0, prompt_ids, 0, params, status, None)
    try:
        _, _, metadata, _ = await generator.__anext__()
        return metadata
    finally:
        await generator.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
async def test_prompt_generation_sentencepiece_boundary(chat_response, stream, sequence):
    import numpy as np
    from lightllm.server.detokenization.decode import decode_token
    from lightllm.server.detokenization.decode_req import DecodeReq
    from lightllm.server.embed_cache.utils import MULTIMODAL_TOKEN_ID_START

    decoder = (
        decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(" ", 1, 0),
            ]
        )
        if sequence
        else decoders.Metaspace()
    )
    tokenizer = make_tokenizer(["▁Hello", "▁world", "▁there"], decoder)
    prompt_ids = [0] * 6 + [MULTIMODAL_TOKEN_ID_START]
    req = SimpleNamespace(
        request_id=0,
        group_req_id=0,
        input_len=len(prompt_ids),
        shm_prompt_ids=SimpleNamespace(arr=np.array(prompt_ids + [1])),
        sample_params=SimpleNamespace(
            print_eos_token=False, skip_special_tokens=True, add_spaces_between_special_tokens=True
        ),
    )
    decode_req = DecodeReq(req)
    decode_req.output_ids.append(1)
    delta = decode_token(tokenizer, decode_req, 1, [])
    metadata = await first_token_metadata(prompt_ids)
    assert metadata["logprob_prompt_ids"] == prompt_ids[-5:]
    assert "logprob_prompt_ids" not in await first_token_metadata(prompt_ids, output_index=2)
    text, _, content, _ = await chat_response(
        tokenizer,
        [(1, delta)],
        stream=stream,
        first_metadata=metadata,
        topk=1,
        candidates=[(2, -0.2)],
    )
    assert text == delta == " world"
    assert content[0]["token"] == " world"
    assert bytes(content[0]["bytes"]).decode() == text
    candidate = content[0]["top_logprobs"][0]
    assert candidate["token"] == " there"
    assert bytes(candidate["bytes"]).decode() == " there"


@pytest.mark.anyio
@pytest.mark.parametrize("parser_name", ["glm47", "qwen3_coder"])
@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
async def test_stream_tool_body_uses_generated_position(parser_name, chunk_size):
    from lightllm.server.chat_logprob_buffer import ChatLogprobBuffer
    from lightllm.server.function_call_parser import FunctionCallParser
    from lightllm.server.api_models import Tool

    call = (
        "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
        if parser_name == "glm47"
        else "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>"
    )
    pieces = [call, "Paris"]
    tokenizer = make_tokenizer(pieces, decoders.Fuse())
    tools = [
        Tool(
            type="function",
            function={"name": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}},
        )
    ]
    parser = FunctionCallParser(tools, parser_name)
    raw = "".join(pieces)
    normal_text, all_calls, positions = "", [], []
    for start in range(0, len(raw), chunk_size):
        normal, calls, spans = parser.parse_stream_chunk_with_spans(raw[start : start + chunk_size])
        normal_text += normal
        all_calls.extend(calls)
        positions.extend(pos for span in spans for pos in range(span.start, span.end))
    assert normal_text == "Paris" and all_calls
    assert positions == list(range(len(call), len(raw)))
    buffer = ChatLogprobBuffer(tokenizer, separate_reasoning=True)
    parser = FunctionCallParser(tools, parser_name)
    for index, piece in enumerate(pieces):
        raw_span = buffer.record(piece, {"id": index, "logprob": -0.1 * (index + 1)}, FinishStatus())
        buffer.assign([raw_span], len(piece))
        _, _, spans = parser.parse_stream_chunk_with_spans(piece)
    content = buffer.content_logprobs(spans)["content"]
    assert [(item["token"], item["logprob"]) for item in content] == [("Paris", -0.2)]


@pytest.mark.anyio
@pytest.mark.parametrize("text", ['{"name":"weather","arguments":{"city":"Paris"}}', '{"name":123}'])
@pytest.mark.parametrize("tool_parser", [None, "llama3"])
async def test_plain_json_with_tools_keeps_scores(chat_response, text, tool_parser):
    tokenizer = make_tokenizer([text], decoders.Fuse())
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
    for enabled in (False, True):
        choices = await chat_response(
            tokenizer,
            [(0, text)],
            stream=False,
            tools=tools,
            tool_parser=tool_parser,
            scores=[-0.6],
            logprobs=enabled,
            return_choices=True,
        )
        assert choices[0]["message"]["content"] == text
        assert choices[0]["message"]["tool_calls"] is None
        if enabled:
            assert [(item["token"], item["logprob"]) for item in choices[0]["logprobs"]["content"]] == [(text, -0.6)]
        else:
            assert choices[0]["logprobs"] is None


@pytest.mark.anyio
@pytest.mark.parametrize("tool_parser", ["glm47", "qwen3_coder"])
async def test_suppressed_tool_whitespace_has_no_scores(chat_response, tool_parser):
    call = (
        "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
        if tool_parser == "glm47"
        else "<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>"
    )
    pieces = [call, "\n", "Done"]
    tokenizer = make_tokenizer(pieces, decoders.Fuse())
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    for enabled in (False, True):
        text, _, content, _ = await chat_response(
            tokenizer,
            list(enumerate(pieces)),
            stream=True,
            tools=tools,
            tool_parser=tool_parser,
            scores=[-0.1, -0.2, -0.3],
            logprobs=enabled,
        )
        assert text == "Done"
        assert [(item["token"], item["logprob"]) for item in content] == ([("Done", -0.3)] if enabled else [])
        if enabled:
            assert bytes(byte for item in content for byte in item["bytes"]).decode() == text


def test_candidate_after_pending_byte_fallback():
    import numpy as np
    from lightllm.server.core.objs.logprob_utils import ChatTokenDecoder
    from lightllm.server.detokenization.decode import decode_token
    from lightllm.server.detokenization.decode_req import DecodeReq

    tokenizer = make_tokenizer(
        ["▁Hello", "<0xE4>", "<0xBD>", "<0xA0>", "▁world"],
        decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(" ", 1, 0),
            ]
        ),
    )
    req = SimpleNamespace(
        request_id=0,
        group_req_id=0,
        input_len=1,
        shm_prompt_ids=SimpleNamespace(arr=np.array([0, 1, 2, 3])),
        sample_params=SimpleNamespace(
            print_eos_token=False, skip_special_tokens=True, add_spaces_between_special_tokens=True
        ),
    )
    decode_req = DecodeReq(req)
    decoder = ChatTokenDecoder(tokenizer)
    decoder.set_prompt([0])
    deltas = []
    for token_id in (1, 2, 3):
        decode_req.output_ids.append(token_id)
        deltas.append(decode_token(tokenizer, decode_req, token_id, []))
        assert decoder.token_logprob(token_id, -0.1)["bytes"] == [[0xE4], [0xBD], [0xA0]][token_id - 1]
        decoder.advance(token_id)
        candidate = decoder.token_logprob(4, -0.2)
        assert candidate == {"token": " world", "logprob": -0.2, "bytes": list(b" world")}
    assert deltas == ["", "", "你"]


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_special_candidates_keep_identity(chat_response, monkeypatch, stream):
    from lightllm.server.core.objs import sampling_params as params_module

    tokenizer = make_tokenizer(["prefix", "answer", "<eos>", "<pad>"], decoders.Fuse())
    tokenizer.add_special_tokens({"eos_token": "<eos>", "pad_token": "<pad>"})
    monkeypatch.setattr(params_module, "SKIP_SPECIAL_TOKENS", True)
    _, _, content, _ = await chat_response(
        tokenizer,
        [(1, "answer")],
        stream=stream,
        topk=2,
        first_metadata={"logprob_prompt_ids": [0]},
        candidates=[(2, -0.2), (3, -0.3)],
    )
    assert content[0]["top_logprobs"] == [
        {"token": "<eos>", "logprob": -0.2, "bytes": list(b"<eos>")},
        {"token": "<pad>", "logprob": -0.3, "bytes": list(b"<pad>")},
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_pending_byte_candidates_in_chat(chat_response, stream):
    tokenizer = make_tokenizer(
        ["▁Hello", "<0xE4>", "<0xBD>", "<0xA0>", "▁world"],
        decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(" ", 1, 0),
            ]
        ),
    )
    text, _, content, _ = await chat_response(
        tokenizer,
        [(1, ""), (2, ""), (3, "你"), (4, " world")],
        stream=stream,
        first_metadata={"logprob_prompt_ids": [0]},
        topk=1,
        candidates=[(4, -0.2)],
    )
    assert text == "你 world"
    assert len(content) == 4
    assert bytes(byte for item in content for byte in item["bytes"]).decode() == text
    assert all(
        item["top_logprobs"] == [{"token": " world", "logprob": -0.2, "bytes": list(b" world")}] for item in content
    )


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("separate_reasoning", [False, True])
async def test_reasoning_then_tool_uses_api_spans(chat_response, stream, separate_reasoning):
    pieces = [
        "Thinking",
        "</think>",
        "Checking.",
        "<tool_call>",
        '{"name":"weather","arguments":{"city":"Paris"}}',
        "</tool_call>",
    ]
    tokenizer = make_tokenizer(pieces, decoders.Fuse())
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
    text, _, content, reasoning = await chat_response(
        tokenizer,
        list(enumerate(pieces)),
        stream=stream,
        separate_reasoning=separate_reasoning,
        model_type="qwen3",
        tools=tools,
    )
    expected = ["Checking."] if separate_reasoning else ["Thinking", "Checking."]
    assert text == "".join(expected)
    assert [item["token"] for item in content] == expected
    assert [item["token"] for item in reasoning] == (["Thinking"] if separate_reasoning else [])


@pytest.mark.anyio
async def test_reasoning_flush_does_not_include_hidden_tool_scores(chat_response):
    call = "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
    pieces = [call, "<thi"]
    tokenizer = make_tokenizer(pieces, decoders.Fuse())
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    text, _, content, _ = await chat_response(
        tokenizer,
        list(enumerate(pieces)),
        stream=True,
        separate_reasoning=False,
        model_type="qwen3",
        tools=tools,
        tool_parser="glm47",
        scores=[-0.1, -0.2],
    )
    assert text == "<thi"
    assert [(item["token"], item["logprob"]) for item in content] == [("<thi", -0.2)]
