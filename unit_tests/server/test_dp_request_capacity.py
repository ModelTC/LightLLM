from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lightllm.server.core.objs.shm_req_manager import ShmReqManager
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.router.batch import Batch
from lightllm.server.router.req_queue.chunked_prefill.impl import ChunkedPrefillQueue
from lightllm.server.router.req_queue.dp_base_queue import DpQueue
from lightllm.utils.config_utils import get_running_max_req_size_per_dp


@pytest.mark.parametrize(
    "global_capacity,dp,nnodes,expected",
    [(32, 8, 1, 4), (33, 8, 1, 5), (4, 8, 1, 1), (32, 8, 2, 8), (32, 1, 2, 32), (32, 1, 1, 32)],
)
def test_hybrid_capacity_uses_local_dp_count(monkeypatch, global_capacity, dp, nnodes, expected):
    monkeypatch.setattr("lightllm.utils.config_utils.is_hybrid_att_model", lambda _: True)
    args = StartArgs(running_max_req_size=global_capacity, dp=dp, nnodes=nnodes)
    assert get_running_max_req_size_per_dp(args) == expected
    # HTTP/SHM indexes remain global even when GPU request indexes are local.
    monkeypatch.setattr("lightllm.server.core.objs.shm_req_manager.get_env_start_args", lambda: args)
    assert ShmReqManager.get_max_req_num(None) == global_capacity


@pytest.mark.parametrize("mode", ["full_attention", "diverse", "cache_fetch"])
def test_modes_requiring_existing_capacity_keep_global_default(monkeypatch, mode):
    monkeypatch.setattr("lightllm.utils.config_utils.is_hybrid_att_model", lambda _: mode != "full_attention")
    args = StartArgs(
        running_max_req_size=32,
        dp=8,
        diverse_mode=mode == "diverse",
        enable_dp_prompt_cache_fetch=mode == "cache_fetch",
    )
    assert get_running_max_req_size_per_dp(args) == 32


@pytest.mark.parametrize("capacity", [0, -1])
def test_reject_invalid_global_capacity(capacity):
    args = StartArgs(running_max_req_size=capacity, dp=8)
    with pytest.raises(ValueError, match="running_max_req_size"):
        get_running_max_req_size_per_dp(args)


def _make_dp_queue(monkeypatch, balancer):
    monkeypatch.setattr("lightllm.utils.config_utils.is_hybrid_att_model", lambda _: True)
    monkeypatch.setattr("lightllm.server.router.req_queue.base_queue.get_fixed_kv_len", lambda: 0)
    args = StartArgs(
        running_max_req_size=32,
        dp=8,
        max_total_token_num=1048576,
        batch_max_tokens=4096,
        router_token_ratio=0.85,
        dp_balancer=balancer,
    )
    router = SimpleNamespace(
        router_statics=SimpleNamespace(ema_req_out_len=16),
        get_used_tokens=lambda dp: 0,
        shared_token_load=MagicMock(),
    )
    return DpQueue(args, router, ChunkedPrefillQueue, dp_size_in_node=8)


def _req(request_id, dp=-1):
    return SimpleNamespace(
        request_id=request_id,
        sample_params=SimpleNamespace(suggested_dp_index=dp, pd_high_priority_request=False),
        is_aborted=False,
        get_tuple_tokens=lambda busy, ema: (64, 16),
        get_first_router_need_tokens=lambda: 64,
        get_decode_need_tokens=lambda: 3,
    )


@pytest.mark.parametrize("balancer", ["bs_balancer", "round_robin"])
def test_global_32_requests_run_as_four_per_rank(monkeypatch, balancer):
    queue = _make_dp_queue(monkeypatch, balancer)
    for i in range(32):
        queue.extend([_req(i)])
    batch = queue.generate_new_batch(None)
    assert batch.get_all_dp_req_num() == [4] * 8
    assert queue.get_wait_req_num() == 0


def test_pinned_overflow_waits_until_a_local_slot_is_released(monkeypatch):
    queue = _make_dp_queue(monkeypatch, "bs_balancer")
    # A skewed client can use any global SHM slot, but only four may enter
    # inference on this rank. The fifth must not exhaust its GPU request pool.
    pinned = [_req(i, dp=2) for i in range(6)]
    for req in pinned:
        queue.extend([req])
    queue.extend([_req(6, dp=3)])
    batch = queue.generate_new_batch(None)
    assert batch.get_all_dp_req_num() == [0, 0, 4, 1, 0, 0, 0, 0]
    assert queue.get_wait_req_num() == 2
    assert queue.generate_new_batch(batch) is None
    # Releasing a request on a different rank must not open a slot on rank 2.
    batch.pop_req(6)
    assert queue.generate_new_batch(batch) is None
    batch.pop_req(0)
    resumed = queue.generate_new_batch(batch)
    assert [req.request_id for req in resumed.reqs] == [4]
    batch.merge(resumed)
    assert batch.get_all_dp_req_num()[2] == 4
    assert queue.get_wait_req_num() == 1
    assert queue.generate_new_batch(batch) is None
    rest = queue.generate_new_batch(Batch(-1, [], dp_size_in_node=8))
    assert [req.request_id for req in rest.reqs] == [5]
    assert queue.get_wait_req_num() == 0
