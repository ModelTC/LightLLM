import asyncio
import copy
from types import SimpleNamespace

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster


def _make_manager(monkeypatch):
    monkeypatch.setattr(
        "lightllm.server.httpserver.manager.HttpServerManager._check_and_repair_length",
        classmethod(lambda cls, *a, **k: asyncio.sleep(0)),
    )
    monkeypatch.setattr(SamplingParams, "from_buffer_copy", classmethod(lambda cls, other: copy.copy(other)))
    mgr = object.__new__(HttpServerManagerForPDMaster)
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
    mgr.select_p_d_node = lambda *a, **k: asyncio.sleep(0, result=(p_node, 1, 0.0))
    mgr.remove_req = lambda *a, **k: asyncio.sleep(0)
    return mgr


def _collect(mgr, sampling_params, monkeypatch, split):
    mgr._split_max_new_tokens = lambda *a, **k: list(split)

    async def fake_wait(p_node, d_node, start_time, prompt, sp, multimodal_params, request):
        sub_req_id = sp.group_request_id
        hit = sp.max_new_tokens * 10
        yield sub_req_id, "x", {"prompt_tokens": 100, "prompt_cache_len": hit}, FinishStatus()
        for _ in range(2):
            yield sub_req_id, "y", {"prompt_tokens": 100, "prompt_cache_len": 0}, FinishStatus()
        yield sub_req_id, "z", {"prompt_tokens": 100, "prompt_cache_len": 0}, FinishStatus(FinishStatus.FINISHED_STOP)

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
    cached = _collect(mgr, sp, monkeypatch, split=[3])
    assert cached and all(c == 30 for c in cached), cached
    assert mgr.recorded_cache_hit_rates == [pytest.approx(0.3)]
    assert len(mgr.inserted_prompt_caches) == 1
    assert mgr.inserted_prompt_caches[0][0] == "hello"


def test_multi_block_keeps_first_block_hit(monkeypatch):
    mgr = _make_manager(monkeypatch)
    sp = SamplingParams()
    sp.n = 1
    sp.max_new_tokens = 5
    sp.best_of = 1
    sp.group_request_id = 0
    cached = _collect(mgr, sp, monkeypatch, split=[3, 2])
    assert cached[-1] == 30, cached
    assert mgr.recorded_cache_hit_rates == [pytest.approx(0.3)]
    assert len(mgr.inserted_prompt_caches) == 1
    assert mgr.inserted_prompt_caches[0][0] == "hello"


def test_error_result_records_hit_rate_without_inserting_prompt_cache(monkeypatch):
    mgr = _make_manager(monkeypatch)
    mgr._split_max_new_tokens = lambda *a, **k: [1]
    sampling_params = SamplingParams()
    sampling_params.n = 1
    sampling_params.best_of = 1
    sampling_params.max_new_tokens = 1

    async def failed_wait(_p_node, _d_node, _start_time, _prompt, sp, *_args):
        yield (
            sp.group_request_id,
            "",
            {"prompt_tokens": 100, "prompt_cache_len": 20},
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
