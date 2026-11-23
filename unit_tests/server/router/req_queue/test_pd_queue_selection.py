from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lightllm.server.router.req_queue import _get_req_queue_class
from lightllm.server.router.batch import Batch, Req
from lightllm.server.router.req_queue.chunked_prefill.impl_for_pd_decode import PDDecodeQueue
from lightllm.server.router.req_queue.chunked_prefill.impl_for_pd_prefill import PDPrefillQueue
from lightllm.server.router.req_queue.dp_balancer.cache_aware import DpCacheAwareBalancer


def _make_args(run_mode: str):
    return SimpleNamespace(
        run_mode=run_mode,
        diverse_mode=False,
        output_constraint_mode="none",
        disable_chunked_prefill=False,
    )


def test_pd_prefill_uses_prefill_queue():
    args = _make_args("prefill")

    assert _get_req_queue_class(args, router=None, dp_size_in_node=1) is PDPrefillQueue


def test_pd_decode_uses_decode_queue():
    args = _make_args("decode")

    assert _get_req_queue_class(args, router=None, dp_size_in_node=1) is PDDecodeQueue


def test_pd_prefill_peak_tokens_sum_request_tuples():
    queue = PDPrefillQueue.__new__(PDPrefillQueue)
    queue.dp_index = 0
    queue.args = SimpleNamespace(page_size=16)
    queue.is_busy = lambda: False
    queue.router = SimpleNamespace(router_statics=SimpleNamespace(ema_req_out_len=10))
    reqs = [
        SimpleNamespace(
            request_id="req-0",
            input_len=10,
            sample_params=SimpleNamespace(max_new_tokens=1, suggested_dp_index=0),
            get_tuple_tokens=MagicMock(return_value=(11, 34)),
        ),
        SimpleNamespace(
            request_id="req-1",
            input_len=20,
            sample_params=SimpleNamespace(max_new_tokens=1, suggested_dp_index=0),
            get_tuple_tokens=MagicMock(return_value=(21, 34)),
        ),
        SimpleNamespace(
            request_id="req-2",
            input_len=100,
            sample_params=SimpleNamespace(max_new_tokens=1, suggested_dp_index=1),
            get_tuple_tokens=MagicMock(return_value=(101, 34)),
        ),
    ]
    batch = Batch(batch_id=1, reqs=reqs, dp_size_in_node=2)

    assert queue._caclu_batch_estimated_peak_token_num(batch) == 100
    reqs[0].get_tuple_tokens.assert_called_once_with(False, 10)
    reqs[1].get_tuple_tokens.assert_called_once_with(False, 10)
    reqs[2].get_tuple_tokens.assert_not_called()


def test_pd_decode_sums_request_tuples_for_non_decode_requests():
    queue = PDDecodeQueue.__new__(PDDecodeQueue)
    queue.args = SimpleNamespace(page_size=16)
    queue.dp_index = 0
    queue.is_busy = lambda: False
    queue.router = SimpleNamespace(router_statics=SimpleNamespace(ema_req_out_len=10))
    reqs = [
        SimpleNamespace(
            request_id="req-0",
            input_len=20,
            sample_params=SimpleNamespace(max_new_tokens=1, suggested_dp_index=0),
            is_infer_decode=lambda: False,
            get_tuple_tokens=MagicMock(return_value=(21, 34)),
        ),
        SimpleNamespace(
            request_id="req-1",
            input_len=100,
            sample_params=SimpleNamespace(max_new_tokens=1, suggested_dp_index=1),
            is_infer_decode=lambda: False,
            get_tuple_tokens=MagicMock(return_value=(101, 34)),
        ),
    ]
    batch = Batch(batch_id=1, reqs=reqs, dp_size_in_node=2)

    assert queue._caclu_batch_estimated_peak_token_num(batch) == 55
    reqs[0].get_tuple_tokens.assert_called_once_with(False, 10)
    reqs[1].get_tuple_tokens.assert_not_called()


def test_pd_decode_uses_tuple_estimation():
    queue = PDDecodeQueue.__new__(PDDecodeQueue)
    queue.args = SimpleNamespace(page_size=16)
    queue.dp_index = 0
    queue.is_busy = lambda: False
    queue.router = SimpleNamespace(router_statics=SimpleNamespace(ema_req_out_len=10))
    req = SimpleNamespace(
        request_id="req-0",
        sample_params=SimpleNamespace(suggested_dp_index=0),
        is_infer_decode=lambda: True,
        get_tuple_tokens=lambda is_busy, ema_req_out_len: (16, 32),
    )
    batch = Batch(batch_id=1, reqs=[req], dp_size_in_node=1)

    assert queue._caclu_batch_estimated_peak_token_num(batch) == 48


@pytest.mark.parametrize(
    "run_mode,running_counts,waiting_counts,group_sizes,cache_hit,expected",
    [
        ("prefill", (2, 4), (0, 0), (1,), False, (0,)),
        ("prefill", (2, 4), (0, 0), (1,), True, (0,)),
        ("prefill", (2, 3), (0, 0), (1,), True, (1,)),
        ("prefill", (4, 4), (0, 0), (1,), True, (1,)),
        ("prefill", (2, 1), (0, 1), (2, 1), True, (1, 0)),
        ("decode", (2, 4), (0, 0), (1,), True, (1,)),
    ],
)
def test_cache_aware_prefill_respects_rank_request_slots(
    run_mode, running_counts, waiting_counts, group_sizes, cache_hit, expected
):
    prompt = np.arange(1025, dtype=np.int64)

    def req(dp, length, req_id):
        return SimpleNamespace(
            request_id=req_id,
            input_len=length,
            shm_cur_kv_len=0,
            shm_prompt_ids=prompt,
            sample_params=SimpleNamespace(suggested_dp_index=dp, disable_prompt_cache=False),
            get_prompt_ids_numpy=lambda: prompt,
        )

    running = [
        req(dp, (8192, 128)[dp], (dp, index)) for dp, count in enumerate(running_counts) for index in range(count)
    ]
    queues = []
    for dp, count in enumerate(waiting_counts):
        waiting = [req(dp, (8192, 128)[dp], ("waiting", dp, index)) for index in range(count)]
        queues.append(SimpleNamespace(waiting_req_list=waiting, running_max_req_size=4, extend=waiting.extend))
    balancer = DpCacheAwareBalancer(2, queues, run_mode)
    if cache_hit:
        balancer.prefix_cache.insert(balancer.prefix_cache.hash_prefixes(prompt), 1)

    groups = [
        [req(-1, len(prompt), ("new", index, sub)) for sub in range(count)] for index, count in enumerate(group_sizes)
    ]
    pending = groups.copy()
    balancer.assign_reqs_to_dp(Batch(1, running, dp_size_in_node=2), pending)
    assert not pending
    assert [[request.sample_params.suggested_dp_index for request in group] for group in groups] == [
        [dp] * count for dp, count in zip(expected, group_sizes)
    ]


@pytest.mark.parametrize(
    "host_len,expected_rank",
    [(0, 1), (512, 1), (1024, 0)],
)
def test_dp_prefill_rotates_host_only_hits_without_losing_longer_gpu_hints(host_len, expected_rank):
    prompt = np.arange(1025, dtype=np.int64)

    def req(dp, length, host=0):
        return SimpleNamespace(
            input_len=length,
            shm_cur_kv_len=0,
            shm_prompt_ids=prompt,
            sample_params=SimpleNamespace(suggested_dp_index=dp, disable_prompt_cache=False, prompt_logprobs=-1),
            get_prompt_ids_numpy=lambda: prompt,
            get_host_cache_match_len=lambda: host,
        )

    waiting = [[], []]
    queues = [SimpleNamespace(waiting_req_list=q, running_max_req_size=4, extend=q.extend) for q in waiting]
    balancer = DpCacheAwareBalancer(2, queues, "prefill", enable_host_cache=True)
    balancer.prefix_cache.insert(balancer.prefix_cache.hash_prefixes(prompt), 1)
    running = [req(0, 256), req(1, 512)]
    incoming = req(-1, len(prompt), host_len)

    balancer.assign_reqs_to_dp(SimpleNamespace(reqs=running), [[incoming]])

    assert incoming.sample_params.suggested_dp_index == expected_rank


def test_host_cache_boundary_comes_from_the_last_staged_page():
    req = SimpleNamespace(
        input_len=5001,
        sample_params=SimpleNamespace(disable_prompt_cache=False, prompt_logprobs=-1),
        cpu_cache_match_page_indexes=SimpleNamespace(size=2),
        token_hash_page_len_list=SimpleNamespace(items=[4096, 4608, 5120]),
    )
    assert Req.get_host_cache_match_len(req) == 4608
    req.sample_params.prompt_logprobs = 0
    assert Req.get_host_cache_match_len(req) == 0


@pytest.mark.parametrize(
    "strategy,blocked_priority,expect_bypass",
    [("hrrn_resume", False, True), ("hrrn_resume", True, False), ("default", False, False)],
)
def test_pd_prefill_bypasses_a_blocked_head_once_for_a_host_hit(strategy, blocked_priority, expect_bypass):
    queue = PDPrefillQueue.__new__(PDPrefillQueue)
    queue.dp_index = 0
    queue.dp_size_in_node = 1
    queue.max_total_tokens = 100
    queue.running_max_req_size = 6
    queue._bypassed_request_id = None
    queue.args = SimpleNamespace(
        prefill_queue_strategy=strategy, enable_cpu_cache=True, chunked_prefill_size=8, diverse_mode=False
    )
    queue.is_busy = lambda: False
    queue.filter_aborted_reqs = lambda: None
    queue.router = SimpleNamespace(router_statics=SimpleNamespace(ema_req_out_len=10), shared_token_load=MagicMock())

    def req(req_id, length, host=0, high_priority=False):
        return SimpleNamespace(
            request_id=req_id,
            input_len=length,
            sample_params=SimpleNamespace(suggested_dp_index=0, high_priority_request=high_priority),
            get_host_cache_match_len=lambda: host,
            get_tuple_tokens=lambda *args: (length, 0),
        )

    running = Batch(1, [req("running", 50)], dp_size_in_node=1)
    small = req("small", 10, high_priority=blocked_priority)
    blocked = req("blocked", 70, high_priority=blocked_priority)
    cold = req("cold", 20)
    too_large = req("too_large", 45, host=40)
    warm = req("warm", 20, host=16)
    another_warm = req("another_warm", 20, host=16)
    queue.waiting_req_list = [small, blocked, cold, too_large, warm, another_warm]

    batch = queue.generate_new_batch(running)
    assert batch.reqs == ([small, warm] if expect_bypass else [small])
    assert queue.waiting_req_list == (
        [blocked, cold, too_large, another_warm] if expect_bypass else [blocked, cold, too_large, warm, another_warm]
    )
    assert queue.generate_new_batch(running) is None
    assert queue.generate_new_batch(None).reqs[0] is blocked
