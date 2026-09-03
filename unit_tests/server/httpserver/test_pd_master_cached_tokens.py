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
    mgr.args = SimpleNamespace(disable_pd_master_decode_capacity_limit=True)
    mgr.pd_high_priority_request_time_out_seconds = 60
    mgr.pd_cache_high_priority_max_age_seconds = 60
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


def _collect(mgr, sampling_params, monkeypatch, segments):
    segment_iter = iter(segments)

    async def fake_wait(p_node, d_node, start_time, prompt, sp, multimodal_params, request):
        sub_req_id = sp.group_request_id
        token_count, final_status = next(segment_iter)
        hit = sampling_params.max_new_tokens * 10
        for token_index in range(1, token_count + 1):
            finish_status = FinishStatus()
            if token_index == token_count:
                finish_status = FinishStatus(final_status)
            yield (
                sub_req_id,
                "x",
                {
                    "prompt_tokens": 100,
                    "prompt_cache_len": hit if token_index == 1 else 0,
                    "count_output_tokens": token_index,
                },
                finish_status,
            )

    monkeypatch.setattr(mgr, "_wait_to_token_package", fake_wait)

    async def run():
        out = []
        async for sub_id, out_str, metadata, finish in mgr.generate(
            prompt="hello",
            sampling_params=sampling_params,
            multimodal_params=SimpleNamespace(images=[], audios=[], verify_and_preload=lambda req: asyncio.sleep(0)),
            request=None,
        ):
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
            (3, FinishStatus.FINISHED_LENGTH),
            (1, FinishStatus.FINISHED_STOP),
        ],
    )
    assert cached and all(c == 50 for c in cached), cached
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
            {"prompt_tokens": 100, "prompt_cache_len": 20, "count_output_tokens": 0},
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
