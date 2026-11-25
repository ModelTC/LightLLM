import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightllm.server import api_lightllm
from lightllm.server.core.objs import FinishStatus
from lightllm.server.httpserver.stop_seq_match import wrap_stop_match_stream
from unit_tests.server.core.objs.test_stop_sequence_output import _http_manager, _params


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("token_stop", [False, True])
@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize(
    "backend_finish", [FinishStatus.FINISHED_ABORTED, FinishStatus.FINISHED_LENGTH, FinishStatus.FINISHED_STOP]
)
def test_metrics_preserve_application_stop_and_collect_final_metadata(
    monkeypatch, stream, token_stop, include, split, backend_finish
):
    final_metrics = dict(
        schema_version=2,
        visible_output_tokens=4,
        inferred_output_tokens=6,
        verification_output_ranges=[[True, 6, 1, 2], [False, 6, 4, 3]],
        proposals={"rounds": 1, "accepted_draft_tokens": 3},
    )
    counters = dict(mtp_accepted_token_num=3, mtp_verify_token_num=12, mtp_verify_step_num=2)

    class Request:
        async def json(self):
            return {"inputs": [], "parameters": {"return_details": True}}

    async def run(enabled):
        params = _params(stop=[[9]] if token_stop else "END", include=include)
        params.init = lambda **kwargs: None
        params.verify_cpu_async = AsyncMock()
        monkeypatch.setattr(api_lightllm, "SamplingParams", lambda: params)
        manager = _http_manager()
        manager.args.enable_spec_acceptance_metrics = enabled
        manager.output_grammar_compiler = None
        manager.stop_request = AsyncMock()
        raw_finishes = []

        async def raw():
            chunks = ["helloE", "NDextra", "late", "later"] if split else ["hello", "ENDextra", "late", "later"]
            for index, text in enumerate(chunks):
                finish = FinishStatus(backend_finish if index == 3 else FinishStatus.NO_FINISH)
                raw_finishes.append(finish)
                metadata = dict(
                    id=9 if index == 1 else index + 20, logprob=-0.5, logprobs={}, count_output_tokens=index + 1
                )
                if enabled and index == 3:
                    metadata.update(spec_acceptance_metrics=final_metrics, **counters)
                yield 80, text, metadata, finish

        manager.generate = lambda *args, **kwargs: wrap_stop_match_stream(manager, raw(), params, [])
        response = await (api_lightllm.lightllm_generate_stream if stream else api_lightllm.lightllm_generate)(
            Request(), manager
        )
        if stream:
            frames = [json.loads(chunk.removeprefix(b"data:").decode()) async for chunk in response.body_iterator]
            assert sum(row["finished"] for row in frames) == 1
            final = frames[-1]
            result = dict(
                text="".join(row["token"]["text"] for row in frames),
                finish=final["finish_reason"],
                ids=[row["token"]["id"] for row in frames],
                count=final["token"]["count_output_tokens"],
            )
            if enabled:
                metrics = final["spec_acceptance_metrics"]
                assert metrics["visible_output_tokens"] == 2
                assert metrics["backend_visible_output_tokens"] == 4
                assert metrics["committed"] == dict(
                    proposal_rounds=0, output_tokens=0, accepted_draft_tokens=0, unused_proposal_rounds=1
                )
                assert metrics["proposals"] == final_metrics["proposals"]
                assert final_metrics["visible_output_tokens"] == 4
                assert all(final["token"][name] == value for name, value in counters.items())
        else:
            body = json.loads(response.body)
            final = body["tokens"][-1]
            result = dict(
                text=body["generated_text"][0],
                finish=body["finish_reason"],
                ids=[row["id"] for row in body["tokens"]],
                count=body["count_output_tokens"],
            )
            if enabled:
                metrics = final["spec_acceptance_metrics"]
                assert metrics["visible_output_tokens"] == 2
                assert metrics["backend_visible_output_tokens"] == 4
                assert metrics["committed"] == dict(
                    proposal_rounds=0, output_tokens=0, accepted_draft_tokens=0, unused_proposal_rounds=1
                )
                assert metrics["proposals"] == final_metrics["proposals"]
                assert final_metrics["visible_output_tokens"] == 4
                assert all(final[name] == value for name, value in counters.items())
        manager.stop_request.assert_awaited_once_with(80)
        assert [finish.status for finish in raw_finishes] == [0, 0, 0, backend_finish]
        return result

    off = asyncio.run(run(False))
    on = asyncio.run(run(True))
    assert on == off
    assert on["finish"] == "stop"


@pytest.mark.parametrize("legacy", ["enable_rl", "enable_return_routed_experts", "prompt_logprobs"])
def test_metrics_keep_existing_final_metadata_mode(monkeypatch, legacy):
    async def run(enabled):
        manager = _http_manager()
        manager.args.enable_spec_acceptance_metrics = enabled
        params = _params()
        if legacy == "prompt_logprobs":
            params.prompt_logprobs = 0
        else:
            setattr(manager.args, legacy, True)
        manager.stop_request = AsyncMock()

        async def raw():
            yield 80, "END", {"id": 1}, FinishStatus()
            yield 80, "late", {"id": 2, "routed_experts": [7]}, FinishStatus(FinishStatus.FINISHED_LENGTH)

        return [
            (text, meta, finish.status)
            async for _, text, meta, finish in wrap_stop_match_stream(manager, raw(), params, [])
        ]

    assert asyncio.run(run(True)) == asyncio.run(run(False))


@pytest.mark.parametrize("token_stop", [False, True])
@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("backend_finish", [1, 2, 3, 4])
def test_metrics_preserve_stop_when_hit_packet_already_finishes(token_stop, include, backend_finish):
    from lightllm.server.httpserver.stop_seq_match import wrap_stop_match_stream
    from unit_tests.server.core.objs.test_stop_sequence_output import _http_manager, _params
    from lightllm.server.core.objs.req import FinishStatus

    async def run(enabled):
        manager = _http_manager()
        manager.args.enable_spec_acceptance_metrics = enabled
        manager.stop_request = AsyncMock()
        params = _params(stop=[[9]] if token_stop else "END", include=include)

        async def raw():
            yield 80, "hello", {"id": 1}, FinishStatus()
            yield 80, "ENDextra", {"id": 9, "spec_acceptance_metrics": {"schema_version": 2}}, FinishStatus(
                backend_finish
            )

        rows = [
            (text, meta, finish.status)
            async for _, text, meta, finish in wrap_stop_match_stream(manager, raw(), params, [])
        ]
        manager.stop_request.assert_not_awaited()
        return rows

    assert asyncio.run(run(True)) == asyncio.run(run(False))
