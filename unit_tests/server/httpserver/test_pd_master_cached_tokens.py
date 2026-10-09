import asyncio
import copy
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from lightllm.server.httpserver_for_pd_master.pd_selector import PDSelectionExtraInfo
from lightllm.server.pd_io_struct import ObjType, PD_Client_Obj, PDUpKVStatus
from lightllm.utils.error_utils import ServerBusyError
from lightllm.server.router.req_queue.dp_base_queue import DpQueue


def _make_manager(monkeypatch):
    monkeypatch.setattr(
        "lightllm.server.httpserver.manager.HttpServerManager._check_and_repair_length",
        classmethod(lambda cls, *a, **k: asyncio.sleep(0)),
    )
    monkeypatch.setattr(SamplingParams, "from_buffer_copy", classmethod(lambda cls, other: copy.copy(other)))
    mgr = object.__new__(HttpServerManagerForPDMaster)
    mgr.args = SimpleNamespace(disable_pd_master_decode_capacity_limit=True)
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
            insert_prompt_cache=lambda prompt, p_node, dp_rank: mgr.inserted_prompt_caches.append(
                (prompt, p_node, dp_rank)
            ),
        )
    )
    p_node = SimpleNamespace(dispatched_prompt_chars=0, dispatched_req_num=0)
    mgr.select_p_d_node = lambda *a, **k: asyncio.sleep(0, result=(p_node, 1, PDSelectionExtraInfo()))
    mgr.remove_req = lambda *a, **k: asyncio.sleep(0)
    return mgr


def _collect(mgr, sampling_params, monkeypatch, segments):
    segment_iter = iter(segments)

    async def fake_wait(p_node, d_node, start_time, prompt, sp, multimodal_params, request, **_kwargs):
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
                    "_prefill_dp_rank": 0,
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

    async def failed_wait(_p_node, _d_node, _start_time, _prompt, sp, *_args, **_kwargs):
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

    assert mgr.recorded_cache_hit_rates == []
    assert mgr.inserted_prompt_caches == []


@pytest.mark.parametrize("skip_prefill", [False, True])
@pytest.mark.parametrize("master_dp_rank", [None, 3])
def test_pd_dispatch_separates_dp_ranks_and_tracks_actual_prefill_rank(skip_prefill, master_dp_rank):
    async def run():
        mgr = object.__new__(HttpServerManagerForPDMaster)
        mgr.args = SimpleNamespace(pd_node_id=42)
        mgr.req_id_to_out_inf = {}
        p_node = PD_Client_Obj(1, "p:8000", "prefill", {"dp": 4, "nnodes": 1})
        d_node = PD_Client_Obj(2, "d:8000", "decode", {"dp": 1, "nnodes": 1})
        p_node.websocket = SimpleNamespace(send_bytes=AsyncMock())
        d_node.websocket = SimpleNamespace(send_bytes=AsyncMock())
        params = SamplingParams()
        params.init(None, max_new_tokens=8)
        params.group_request_id = 123
        params.suggested_dp_index = 1
        actual_dp_rank = 1 if master_dp_rank is None else 3
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))

        async def ready(_event, _request, **kwargs):
            status = mgr.req_id_to_out_inf[123]
            if kwargs["stage"] == "prefill":
                status.prefill_prompt_ids_event.prompt_ids = [1, 2, 3, 4]
            else:
                status.up_status_event.upkv_status = PDUpKVStatus(
                    123, 42, pickle.dumps(SimpleNamespace(ready_kv_len=3 if skip_prefill else 0))
                )
                # D may report the first token before P. Cache attribution must still use P.
                status.out_token_info_list = [
                    (123, "D", {"count_output_tokens": 1, "node_mode": "decode", "dp_rank": 0}, FinishStatus())
                ]
                if not skip_prefill:
                    status.out_token_info_list.append(
                        (
                            123,
                            "P",
                            {
                                "count_output_tokens": 1,
                                "node_mode": "prefill",
                                "dp_rank": actual_dp_rank,
                                "prompt_cache_len": 2,
                            },
                            FinishStatus(FinishStatus.FINISHED_LENGTH),
                        )
                    )
                status.event.set()

        mgr._wait_for_event_or_disconnect = ready
        stream = mgr.fetch_pd_stream(
            p_node, d_node, "prompt", params, SimpleNamespace(), request, prefill_dp_rank=master_dp_rank
        )
        try:
            result = await asyncio.wait_for(stream.__anext__(), timeout=1)
        finally:
            await stream.aclose()
        p_params = pickle.loads(p_node.websocket.send_bytes.await_args_list[0].args[0])[1][1]
        d_params = pickle.loads(d_node.websocket.send_bytes.await_args.args[0])[1][1]
        assert p_params.suggested_dp_index == actual_dp_rank
        assert d_params.suggested_dp_index == (1 if master_dp_rank is None else -1)
        assert p_params.max_new_tokens == 1
        assert d_params.max_new_tokens == params.max_new_tokens == 8
        assert params.suggested_dp_index == 1
        assert p_params.group_request_id == d_params.group_request_id == 123
        assert result[2]["_prefill_dp_rank"] == (None if skip_prefill else actual_dp_rank)
        queue = object.__new__(DpQueue)
        queue.dp_size_in_node = 4
        queue.inner_queues = [SimpleNamespace(extend=MagicMock()) for _ in range(4)]
        queue.reqs_waiting_for_dp_index = []
        group = [SimpleNamespace(sample_params=p_params)]
        queue.extend(group)
        queue.inner_queues[actual_dp_rank].extend.assert_called_once_with(group)
        assert queue.reqs_waiting_for_dp_index == []
        for rank in range(4):
            if rank != actual_dp_rank:
                queue.inner_queues[rank].extend.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["success", "skip_prefill", "busy", "cancel", "split"])
def test_rank_inflight_accounting_and_cache_feedback(monkeypatch, outcome):
    mgr = _make_manager(monkeypatch)
    node = PD_Client_Obj(1, "p:8000", "prefill", {"dp": 4, "nnodes": 1})
    mgr.select_p_d_node = lambda *_args: asyncio.sleep(0, result=(node, 1, PDSelectionExtraInfo(prefill_dp_rank=3)))
    mgr.abort = AsyncMock()
    calls = []

    async def results(_p, _d, _start, prompt, sp, *_args, prefill_dp_rank):
        calls.append(prompt)
        assert prefill_dp_rank == 3
        assert node.dp_ranks[3].dispatched_req_num == 1
        assert node.dp_ranks[3].dispatched_prompt_chars == len(prompt)
        assert all(rank.dispatched_req_num == 0 for rank in node.dp_ranks[:3])
        if outcome == "busy":
            raise ServerBusyError("busy")
        if outcome == "cancel":
            raise asyncio.CancelledError()
        yield (
            sp.group_request_id,
            "x",
            {
                "prompt_tokens": 100,
                "prompt_cache_len": 20,
                "_prefill_dp_rank": None if outcome == "skip_prefill" else 3,
            },
            FinishStatus(
                FinishStatus.FINISHED_STOP if outcome != "split" or len(calls) > 1 else FinishStatus.NO_FINISH
            ),
        )
        if outcome == "split" and len(calls) == 1:
            yield sp.group_request_id, "", {}, FinishStatus(FinishStatus.FINISHED_PD_DECODE_CAPACITY)

    mgr._wait_to_token_package = results
    sp = SimpleNamespace(max_new_tokens=2, group_request_id=123)

    async def run():
        async for _, _, metadata, _ in mgr._generate_one_attempt("hello", sp, None, None, 0, 123, 100):
            assert "_prefill_dp_rank" not in metadata

    if outcome in ("busy", "cancel"):
        with pytest.raises(ServerBusyError if outcome == "busy" else asyncio.CancelledError):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert all(rank.dispatched_req_num == rank.dispatched_prompt_chars == 0 for rank in node.dp_ranks)
    assert len(calls) == (2 if outcome == "split" else 1)
    if outcome in ("success", "split"):
        assert mgr.inserted_prompt_caches == [("hello", node, 3)]
    else:
        assert mgr.inserted_prompt_caches == []
