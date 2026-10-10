import asyncio
import copy
from types import SimpleNamespace

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from lightllm.server.httpserver_for_pd_master.pd_selector import PDSelectionExtraInfo


def _make_manager(monkeypatch):
    monkeypatch.setattr(
        "lightllm.server.httpserver.manager.HttpServerManager._check_and_repair_length",
        classmethod(lambda cls, *a, **k: asyncio.sleep(0)),
    )
    monkeypatch.setattr(SamplingParams, "from_buffer_copy", classmethod(lambda cls, other: copy.copy(other)))
    mgr = object.__new__(HttpServerManagerForPDMaster)
    mgr.args = SimpleNamespace(disable_pd_master_decode_capacity_limit=True, reasoning_parser=None)
    mgr.enable_pd_node_self_request_limit = True
    mgr.pd_node_resource_wait_timeout_seconds = -1
    mgr.pd_node_continuation_resource_wait_timeout_seconds = 60
    mgr.pd_node_busy_retry_timeout_seconds = 120
    mgr.pd_cache_high_priority_max_age_seconds = 60
    mgr.pd_cache_high_priority_min_prompt_tokens = 8192
    mgr.disable_pd_cache_high_priority = False
    mgr.running_request_count = 0
    counter = [0]

    def gen_id():
        counter[0] += 1
        return counter[0]

    mgr.id_gen = SimpleNamespace(generate_id=gen_id)
    mgr.metric_client = SimpleNamespace(counter_inc=lambda *a, **k: None, histogram_observe=lambda *a, **k: None)
    mgr.tokens = lambda *a, **k: 10
    mgr._log_req_header = lambda *a, **k: asyncio.sleep(0)
    mgr.recorded_cache_hit_rates = []
    mgr.inserted_prompt_caches = []
    mgr.pd_manager = SimpleNamespace(
        selector=SimpleNamespace(
            record_prompt_cache_hit_rate=mgr.recorded_cache_hit_rates.append,
            insert_prompt_cache=lambda prompt, p_node: mgr.inserted_prompt_caches.append((prompt, p_node)),
        )
    )
    p_node = SimpleNamespace(dispatched_prompt_chars=0, dispatched_req_num=0)
    mgr.select_p_d_node = lambda *a, **k: asyncio.sleep(0, result=(p_node, 1, PDSelectionExtraInfo()))
    mgr.remove_req = lambda *a, **k: asyncio.sleep(0)
    return mgr


def _collect(mgr, sampling_params, monkeypatch, segments, first_node_mode="prefill", needs_prefill_first_token=True):
    segment_iter = iter(segments)

    async def fake_wait(p_node, d_node, start_time, prompt, sp, multimodal_params, request):
        sub_req_id = sp.group_request_id
        token_count, final_status = next(segment_iter)
        hit = sampling_params.max_new_tokens * 10
        for token_index in range(1, token_count + 1):
            finish_status = FinishStatus()
            if token_index == token_count:
                finish_status = FinishStatus(final_status)
            metadata = {
                "prompt_tokens": 100,
                "prompt_cache_len": hit if token_index == 1 else 0,
                "count_output_tokens": token_index,
                "node_mode": first_node_mode if token_index == 1 else "decode",
            }
            if token_index == 1 and needs_prefill_first_token is not None:
                metadata["needs_prefill_first_token"] = needs_prefill_first_token
            yield sub_req_id, "x", metadata, finish_status

    monkeypatch.setattr(mgr, "_wait_to_token_package", fake_wait)

    async def run():
        out = []
        async for sub_id, out_str, metadata, finish in mgr.generate(
            prompt="hello",
            sampling_params=sampling_params,
            multimodal_params=SimpleNamespace(images=[], audios=[], verify_and_preload=lambda req: asyncio.sleep(0)),
            request=None,
        ):
            assert "needs_prefill_first_token" not in metadata
            out.append(metadata.get("prompt_cache_len", -1))
        return out

    return asyncio.run(run())


def test_single_block_prefill_hit_persists_past_decode_zeros(monkeypatch):
    mgr = _make_manager(monkeypatch)
    sp = SamplingParams()
    sp.n = 1
    sp.max_new_tokens = 3
    sp.best_of = 1
    sp.group_request_id = 0
    cached = _collect(mgr, sp, monkeypatch, segments=[(3, FinishStatus.FINISHED_STOP)])
    assert cached and all(c == 30 for c in cached), cached
    assert mgr.recorded_cache_hit_rates == [pytest.approx(0.3)]
    assert len(mgr.inserted_prompt_caches) == 1
    assert mgr.inserted_prompt_caches[0][0] == "hello"


def test_dynamic_split_keeps_first_segment_hit(monkeypatch):
    mgr = _make_manager(monkeypatch)
    sp = SamplingParams()
    sp.n = 1
    sp.max_new_tokens = 5
    sp.best_of = 1
    sp.group_request_id = 0
    cached = _collect(
        mgr,
        sp,
        monkeypatch,
        segments=[
            (3, FinishStatus.FINISHED_PD_DECODE_CAPACITY),
            (1, FinishStatus.FINISHED_STOP),
        ],
    )
    assert cached == [50, 50, 50], cached
    assert mgr.recorded_cache_hit_rates == [pytest.approx(0.5)]
    assert len(mgr.inserted_prompt_caches) == 1
    assert mgr.inserted_prompt_caches[0][0] == "hello"


def test_error_result_records_hit_rate_without_inserting_prompt_cache(monkeypatch):
    mgr = _make_manager(monkeypatch)
    sampling_params = SamplingParams()
    sampling_params.n = 1
    sampling_params.best_of = 1
    sampling_params.max_new_tokens = 1

    async def failed_wait(_p_node, _d_node, _start_time, _prompt, sp, *_args):
        yield (
            sp.group_request_id,
            "",
            {"prompt_tokens": 100, "prompt_cache_len": 20, "count_output_tokens": 0, "needs_prefill_first_token": True},
            FinishStatus(FinishStatus.FINISHED_ERROR),
        )

    monkeypatch.setattr(mgr, "_wait_to_token_package", failed_wait)

    async def run():
        async for _ in mgr.generate(
            prompt="hello",
            sampling_params=sampling_params,
            multimodal_params=SimpleNamespace(
                images=[],
                audios=[],
                verify_and_preload=lambda req: asyncio.sleep(0),
            ),
            request=None,
        ):
            pass

    asyncio.run(run())

    assert mgr.recorded_cache_hit_rates == [pytest.approx(0.2)]
    assert mgr.inserted_prompt_caches == []


@pytest.mark.parametrize("in_reasoning", [False, True])
def test_pd_continuation_uses_local_reasoning_state(monkeypatch, in_reasoning):
    from lightllm.server.detokenization import stop_sequence

    monkeypatch.setattr(stop_sequence, "get_env_start_args", lambda: SimpleNamespace(reasoning_parser="qwen3"))
    monkeypatch.setattr("lightllm.server.httpserver_for_pd_master.manager.get_stop_in_reasoning", lambda: False)
    manager = _make_manager(monkeypatch)
    manager.args.reasoning_parser = "qwen3"
    manager.tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: {"<think>": [1000], "</think>": [1001]}.get(text, [ord(c) for c in text])
    )
    params = SamplingParams()
    params.init(manager.tokenizer, stop_sequences="END", max_new_tokens=3, _initial_reasoning_state=1)
    segment_modes = []

    async def fake_wait(p_node, d_node, start_time, prompt, sp, multimodal_params, request):
        segment_modes.append(sp._initial_reasoning_state)
        assert sp.enable_stop_str_match_in_inference is False
        metadata = {"prompt_tokens": 10, "prompt_cache_len": 0, "count_output_tokens": 1}
        if len(segment_modes) == 1:
            yield sp.group_request_id, "reason", {**metadata, "id": 10}, FinishStatus()
            if not in_reasoning:
                yield sp.group_request_id, "", {**metadata, "id": 1001}, FinishStatus()
            # A simulated delimiter at the capacity boundary must not change state.
            yield (
                sp.group_request_id,
                "",
                {**metadata, "id": 1001},
                FinishStatus(FinishStatus.FINISHED_PD_DECODE_CAPACITY),
            )
        else:
            yield sp.group_request_id, "END", {**metadata, "id": 11}, FinishStatus(FinishStatus.FINISHED_STOP)

    monkeypatch.setattr(manager, "_wait_to_token_package", fake_wait)

    async def run():
        return [
            result
            async for result in manager.generate(
                "hello",
                params,
                SimpleNamespace(images=[], audios=[], verify_and_preload=lambda req: asyncio.sleep(0)),
                None,
            )
        ]

    results = asyncio.run(run())
    assert segment_modes == [1, int(in_reasoning)]
    assert "".join(result[1] for result in results) == "reason" + ("END" if in_reasoning else "")


@pytest.mark.parametrize("node_mode", ["decode", None])
def test_skipped_prefill_does_not_insert_prompt_cache(monkeypatch, node_mode):
    mgr = _make_manager(monkeypatch)
    sampling_params = SamplingParams()
    sampling_params.n = sampling_params.best_of = 1
    sampling_params.max_new_tokens = 3
    cached = _collect(
        mgr,
        sampling_params,
        monkeypatch,
        segments=[(3, FinishStatus.FINISHED_STOP)],
        first_node_mode=node_mode,
        needs_prefill_first_token=False,
    )
    assert cached == [30, 30, 30]
    assert mgr.recorded_cache_hit_rates == [pytest.approx(0.3)]
    assert mgr.inserted_prompt_caches == []


@pytest.mark.parametrize("needs_prefill_first_token", [True, None])
def test_completed_prefill_inserts_cache_when_first_output_is_decode(monkeypatch, needs_prefill_first_token):
    mgr = _make_manager(monkeypatch)
    sampling_params = SamplingParams()
    sampling_params.n = sampling_params.best_of = 1
    sampling_params.max_new_tokens = 3
    cached = _collect(
        mgr,
        sampling_params,
        monkeypatch,
        segments=[(3, FinishStatus.FINISHED_STOP)],
        first_node_mode="decode",
        needs_prefill_first_token=needs_prefill_first_token,
    )
    assert cached == [30, 30, 30]
    assert len(mgr.inserted_prompt_caches) == 1
    assert mgr.inserted_prompt_caches[0][0] == "hello"


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("parser", [None, "qwen3"])
@pytest.mark.parametrize(
    "continuation,node_finish",
    [
        ("NDextra", FinishStatus.NO_FINISH),
        ("NDextra", FinishStatus.FINISHED_STOP),
        ("NX", FinishStatus.FINISHED_LENGTH),
        ("N", FinishStatus.FINISHED_LENGTH),
    ],
)
def test_pd_stop_prefix_survives_capacity_split(monkeypatch, include, parser, continuation, node_finish):
    from unittest.mock import AsyncMock
    from lightllm.server.detokenization import stop_sequence

    monkeypatch.setattr(stop_sequence, "get_env_start_args", lambda: SimpleNamespace(reasoning_parser=parser))
    monkeypatch.setattr("lightllm.server.httpserver_for_pd_master.manager.get_stop_in_reasoning", lambda: False)
    manager = _make_manager(monkeypatch)
    manager.args.reasoning_parser = parser
    manager.tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: {"<think>": [1000], "</think>": [1001]}.get(text, [ord(c) for c in text])
    )
    manager.abort = AsyncMock()
    params = SamplingParams()
    params.init(
        manager.tokenizer,
        stop_sequences="END",
        include_stop_str_in_output=include,
        _initial_reasoning_state=int(parser is not None),
        max_new_tokens=10,
    )
    segments = []

    async def fake_wait(p_node, d_node, start_time, prompt, sp, multimodal_params, request):
        segments.append(prompt)
        assert sp.enable_stop_str_match_in_inference is (parser is None)
        metadata = {"prompt_tokens": 10, "prompt_cache_len": 0, "count_output_tokens": 1}
        if len(segments) == 1:
            if parser:
                yield sp.group_request_id, "END", {**metadata, "id": 10}, FinishStatus()
                yield sp.group_request_id, "", {**metadata, "id": 1001}, FinishStatus()
            yield sp.group_request_id, "helloE", {**metadata, "id": 11}, FinishStatus()
            yield sp.group_request_id, "", dict(metadata), FinishStatus(FinishStatus.FINISHED_PD_DECODE_CAPACITY)
        else:
            yield sp.group_request_id, continuation, {**metadata, "id": 12}, FinishStatus(node_finish)
            if continuation == "NDextra" and node_finish == FinishStatus.NO_FINISH:
                pytest.fail("Master must stop the active segment as soon as the cross-segment stop matches")

    monkeypatch.setattr(manager, "_wait_to_token_package", fake_wait)

    async def run():
        return [
            result
            async for result in manager.generate(
                "prompt",
                params,
                SimpleNamespace(images=[], audios=[], verify_and_preload=lambda req: asyncio.sleep(0)),
                None,
            )
        ]

    results = asyncio.run(run())
    assert len(segments) == 2
    assert segments[1] == "prompt" + ("END" if parser else "") + "helloE"
    expected = "hello" + ("END" if include else "") if continuation == "NDextra" else "helloE" + continuation
    assert "".join(result[1] for result in results) == ("END" if parser else "") + expected
    assert [result[2]["id"] for result in results] == ([10, 1001] if parser else []) + [11, 12]
    assert results[-1][3].get_finish_reason() == ("stop" if continuation == "NDextra" else "length")
    assert all(not result[3].is_finished() for result in results[:-1])
    if continuation == "NDextra" and node_finish == FinishStatus.NO_FINISH:
        manager.abort.assert_awaited_once()
    else:
        manager.abort.assert_not_awaited()
