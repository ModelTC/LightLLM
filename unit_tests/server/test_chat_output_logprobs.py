import json
import pickle
import uuid
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
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
    assert [entry["token"] for entry in content] == (
        ["", "", "你", " yes"] if separate_reasoning else ["think", "", "", "你", " yes"]
    )
    expected_logprobs = [-0.25, -0.25, -0.25, -0.25] if separate_reasoning else [-0.25] * 5
    assert [entry["logprob"] for entry in content] == expected_logprobs
    assert [entry["logprob"] for entry in reasoning] == ([-0.25] if separate_reasoning else [])
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


@pytest.fixture
def chat_response(monkeypatch):
    async def run(tokenizer, pieces, *, stream, separate_reasoning=True, model_type=None, tools=None, topk=0, rank=2):
        async def build_prompt(*args):
            return "prompt"

        async def generate(prompt, params, *args, **kwargs):
            for index, (token_id, delta) in enumerate(pieces):
                yield 0, delta, {
                    "id": token_id,
                    "logprob": -2.401099,
                    "logprobs": {token_id: {"rank": rank}},
                    "prompt_tokens": 10,
                    "output_top_logprobs": [(0, -0.125)] * topk,
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
        monkeypatch.setattr(api_http.g_objs, "args", SimpleNamespace(tool_call_parser="qwen25"), raising=False)
        request = ChatCompletionRequest(
            messages=[],
            stream=stream,
            separate_reasoning=separate_reasoning,
            logprobs=True,
            top_logprobs=topk,
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
async def test_chat_keeps_the_sampled_logprob(chat_response, stream, topk, rank):
    tokenizer = make_tokenizer(["answer"])
    _, _, content, _ = await chat_response(tokenizer, [(0, "answer")], stream=stream, topk=topk, rank=rank)
    assert content[0]["logprob"] == -2.401099
    assert len(content[0]["top_logprobs"]) == topk


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_empty_increments_keep_their_own_logprob(chat_response, stream):
    tokenizer = make_tokenizer(["<0xE4>", "<0xBD>", "<0xA0>"])
    text, _, content, _ = await chat_response(tokenizer, [(0, ""), (1, ""), (2, "你")], stream=stream)
    assert text == "你"
    assert [entry["token"] for entry in content] == ["", "", "你"]
    assert [entry["logprob"] for entry in content] == pytest.approx([-2.401099, -2.401099, -2.401099])
    assert [entry["bytes"] for entry in content] == [[], [], list("你".encode())]


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
    generated = [(tokenizer.convert_tokens_to_ids(piece), piece) for piece in pieces]
    text, reasoning_text, content, reasoning = await chat_response(
        tokenizer,
        generated,
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
async def test_tool_delimiter_fused_into_another_token_is_rejected(chat_response):
    pieces = ["Checking.", '<tool_call>{"name":"weather","arguments":{"city":"Paris"}}', "</tool_call>"]
    tokenizer = make_tokenizer(pieces, decoders.Fuse())
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
    with pytest.raises(ValueError, match="token sequence"):
        await chat_response(tokenizer, list(enumerate(pieces)), stream=False, tools=tools)


def test_empty_increments_keep_each_position_including_a_trailing_one():
    from lightllm.server.chat_logprob_buffer import ChatLogprobBuffer

    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "")
    buffer = ChatLogprobBuffer(tokenizer, None, False, True)
    for token_id, text in enumerate(["", "", "你"]):
        buffer.record(text, {"id": token_id, "logprob": -0.25}, FinishStatus())
    content, _ = buffer.message_logprobs()
    assert [token["token"] for token in content] == ["", "", "你"]
    assert [token["logprob"] for token in content] == [-0.25, -0.25, -0.25]
    assert content[2]["bytes"] == list("你".encode())

    buffer.record("", {"id": 3, "logprob": -0.5}, FinishStatus(FinishStatus.FINISHED_STOP))
    content, reasoning = buffer.message_logprobs()
    assert [(token["token"], token["logprob"]) for token in content] == [("", -0.5)]
    assert reasoning is None


def test_reasoning_delimiter_must_be_one_token():
    from lightllm.server.chat_logprob_buffer import ChatLogprobBuffer

    tokenizer = make_tokenizer(["<think>", "Thinking</think>Answer"], decoders.Fuse())
    with pytest.raises(ValueError, match="token sequence"):
        ChatLogprobBuffer(tokenizer, "qwen3", False, True)


def test_multitoken_reasoning_delimiter_includes_newline():
    from lightllm.server.chat_logprob_buffer import ChatLogprobBuffer

    pieces = ["<|channel>", "thought", "\n", "secret", "<channel|>", "answer"]
    tokenizer = make_tokenizer(pieces, decoders.Fuse())
    buffer = ChatLogprobBuffer(tokenizer, "gemma4", False, True)
    for piece in pieces:
        buffer.record(
            piece,
            {"id": tokenizer.convert_tokens_to_ids(piece), "logprob": -0.2},
            FinishStatus(FinishStatus.FINISHED_STOP if piece == pieces[-1] else FinishStatus.NO_FINISH),
        )
    content, reasoning = buffer.message_logprobs()
    assert [token["token"] for token in reasoning] == ["secret"]
    assert [token["token"] for token in content] == ["answer"]


def test_gemma4_answer_without_thought_header_stays_in_content():
    from lightllm.server.chat_logprob_buffer import ChatLogprobBuffer

    pieces = ["<|channel>", "thought", "\n", "<channel|>", "2"]
    tokenizer = make_tokenizer(pieces, decoders.Fuse())
    buffer = ChatLogprobBuffer(tokenizer, "gemma4", True, True)
    buffer.record("2", {"id": tokenizer.convert_tokens_to_ids("2"), "logprob": -0.2}, FinishStatus())
    content, reasoning = buffer.message_logprobs()
    assert reasoning is None
    assert [token["token"] for token in content] == ["2"]


def test_incomplete_delimiter_prefix_is_visible_text():
    from lightllm.server.chat_logprob_buffer import ChatLogprobBuffer

    pieces = ["<think>", "</", "think>", "nope"]
    tokenizer = make_tokenizer(pieces, decoders.Fuse())
    buffer = ChatLogprobBuffer(tokenizer, "qwen3", False, True)
    for piece in ["</", "nope"]:
        buffer.record(
            piece,
            {"id": tokenizer.convert_tokens_to_ids(piece), "logprob": -0.2},
            FinishStatus(FinishStatus.FINISHED_STOP if piece == "nope" else FinishStatus.NO_FINISH),
        )
    content, reasoning = buffer.message_logprobs()
    assert reasoning is None
    assert [token["token"] for token in content] == ["</", "nope"]


def test_tool_parser_without_end_token_is_supported():
    from lightllm.server.chat_logprob_buffer import ChatLogprobBuffer

    tokenizer = make_tokenizer(["<|python_tag|>", "ok"], decoders.Fuse())
    buffer = ChatLogprobBuffer(
        tokenizer,
        None,
        False,
        True,
        tools=[{"type": "function", "function": {"name": "ok"}}],
        tool_call_parser="llama3",
    )
    assert buffer._sequences


def test_byte_level_candidate_preserves_raw_bytes():
    from lightllm.server.chat_logprob_buffer import _candidate_logprob

    backend = Tokenizer(models.BPE({"Ã": 0}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel()
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    assert _candidate_logprob(tokenizer, 0, -0.5)["bytes"] == [195]


def test_harmony_commentary_attributes_are_hidden_logprobs():
    from lightllm.server.chat_logprob_buffer import ChatLogprobBuffer

    pieces = [
        "<|start|>",
        "assistant",
        "<|channel|>",
        "commentary to=functions.weather",
        "<|message|>",
        '{"city":"Paris"}',
        "<|end|>",
        "<|start|>",
        "assistant",
        "<|channel|>",
        "final",
        "<|message|>",
        "Answer",
        "<|return|>",
    ]
    unique = list(dict.fromkeys(pieces))
    ids = {piece: index for index, piece in enumerate(unique)}
    tokenizer = make_tokenizer(unique, decoders.Fuse())
    buffer = ChatLogprobBuffer(tokenizer, "gpt-oss", False, True)
    for index, piece in enumerate(pieces):
        buffer.record(
            piece,
            {"id": ids[piece], "logprob": -0.1},
            FinishStatus(FinishStatus.FINISHED_STOP if index == len(pieces) - 1 else FinishStatus.NO_FINISH),
        )
    content, reasoning = buffer.message_logprobs()
    assert [token["token"] for token in content] == ["Answer"]
    assert reasoning is None


def test_mistral_tool_parser_is_rejected():
    from lightllm.server.chat_logprob_buffer import ChatLogprobBuffer

    tokenizer = make_tokenizer(["[TOOL_CALLS] [", "]"], decoders.Fuse())
    with pytest.raises(ValueError, match="mistral"):
        ChatLogprobBuffer(
            tokenizer,
            None,
            False,
            True,
            tools=[{"type": "function", "function": {"name": "ok"}}],
            tool_call_parser="mistral",
        )
