import asyncio
import json
from types import SimpleNamespace

import pytest

from lightllm.server import api_lightllm
from lightllm.server.core.objs.req import FinishStatus


@pytest.mark.parametrize("enabled", [False, True])
def test_stream_only_adds_acceptance_metrics_on_opted_in_final_frame(monkeypatch, enabled):
    class Params:
        best_of = 1

        def init(self, **kwargs):
            pass

        async def verify_cpu_async(self, compiler):
            pass

    class Request:
        async def json(self):
            return {"inputs": "test", "parameters": {}}

    async def generate(*args, **kwargs):
        for index in range(2):
            meta = dict(id=17 + index, logprob=-0.5, count_output_tokens=index + 1, logprobs={})
            if index == 1 and enabled:
                meta.update(
                    spec_acceptance_metrics={"schema_version": 2},
                    mtp_accepted_token_num=3,
                    mtp_verify_token_num=12,
                    mtp_verify_step_num=2,
                )
            yield 8, "a", meta, FinishStatus(FinishStatus.FINISHED_LENGTH if index == 1 else FinishStatus.NO_FINISH)

    monkeypatch.setattr(api_lightllm, "SamplingParams", Params)

    async def run():
        manager = SimpleNamespace(tokenizer=None, output_grammar_compiler=None, generate=generate)
        response = await api_lightllm.lightllm_generate_stream(Request(), manager)
        return [json.loads(chunk.removeprefix(b"data:").decode()) async for chunk in response.body_iterator]

    rows = asyncio.run(run())
    assert len(rows) == 2
    assert "spec_acceptance_metrics" not in rows[0]
    if enabled:
        assert rows[1]["spec_acceptance_metrics"] == {"schema_version": 2}
        assert rows[1]["token"]["mtp_verify_step_num"] == 2
    else:
        assert "spec_acceptance_metrics" not in rows[1]
        assert "mtp_verify_step_num" not in rows[1]["token"]
