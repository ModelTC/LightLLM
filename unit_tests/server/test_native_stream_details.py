import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from lightllm.server import api_lightllm


@pytest.mark.parametrize("details", [None, False, True])
@pytest.mark.parametrize("reported", [False, True])
def test_stream_details_are_opt_in_and_preserve_metadata(monkeypatch, details, reported):
    class Sampling:
        best_of = 1

        def init(self, **kwargs):
            assert "return_details" not in kwargs

        async def verify_cpu_async(self, compiler):
            pass

    monkeypatch.setattr(api_lightllm, "SamplingParams", Sampling)
    parameters = {"do_sample": False}
    if details is not None:
        parameters["return_details"] = details

    class Request:
        async def json(self):
            return {"inputs": [11, 12], "parameters": parameters}

    metadata = dict(
        id=7,
        prompt_tokens=2,
        count_output_tokens=1,
        logprob=-0.2,
        logprobs={},
        prompt_cache_len=1,
        mtp_accepted_token_num=2,
        mtp_verify_token_num=3,
        mtp_verify_step_num=1,
        input_usage=dict(input_text_tokens=2),
    )
    fields = ("prompt_cache_len", "mtp_accepted_token_num", "mtp_verify_token_num", "mtp_verify_step_num")
    if not reported:
        for name in fields:
            metadata.pop(name)
    before = copy.deepcopy(metadata)

    async def generate(*args, **kwargs):
        yield 0, "token", metadata, SimpleNamespace(is_finished=lambda: True, get_finish_reason=lambda: "length")

    async def collect():
        response = await api_lightllm.lightllm_generate_stream(
            Request(), SimpleNamespace(tokenizer=None, generate=generate, output_grammar_compiler=None)
        )
        return [json.loads(chunk.decode().removeprefix("data:")) async for chunk in response.body_iterator]

    bodies = asyncio.run(collect())
    assert len(bodies) == 1
    body = bodies[0]
    for name in fields:
        if details and reported:
            assert body["token"][name] == metadata[name]
        else:
            assert name not in body["token"]
    assert body["input_usage"] == metadata["input_usage"] and body["details"] is None
    assert body["token"]["id"] == 7 and body["finish_reason"] == "length"
    assert metadata == before
