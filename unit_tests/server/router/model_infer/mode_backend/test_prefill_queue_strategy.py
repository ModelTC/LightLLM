from types import SimpleNamespace

import pytest

from lightllm.server.api_cli import make_argument_parser
from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.prefill_queue_strategy import (
    PREFILL_QUEUE_STRATEGIES,
    PrefillQueueStrategy,
    create_prefill_queue_strategy,
)


def req(
    group_id,
    input_len,
    kv_len=0,
    priority=0,
    router_priority=False,
    total_len=None,
    finish_status=FinishStatus.NO_FINISH,
    infer_aborted=False,
    filter_mark=False,
    wait_pause=False,
    paused=False,
):
    request = InferReq.__new__(InferReq)
    request.req_id = group_id
    request.cur_kv_len = kv_len
    request.finish_status = FinishStatus(finish_status)
    request.infer_aborted = infer_aborted
    request.filter_mark = filter_mark
    request.wait_pause = wait_pause
    request.paused = paused
    request.shm_req = SimpleNamespace(
        input_len=input_len,
        group_req_id=group_id,
        sample_params=SimpleNamespace(
            high_priority_request=router_priority,
            infer_high_priority=priority,
        ),
    )
    request.get_cur_total_len = lambda: input_len if total_len is None else total_len
    return request


def strategy(name="default", processed_prefill_tokens=0):
    backend = ModeBackend.__new__(ModeBackend)
    backend.args = SimpleNamespace(prefill_queue_strategy=name)
    backend.processed_prefill_tokens = processed_prefill_tokens
    return create_prefill_queue_strategy(backend)


@pytest.mark.parametrize("name", PREFILL_QUEUE_STRATEGIES)
def test_empty_and_singleton(name):
    policy = strategy(name)
    assert isinstance(policy, PrefillQueueStrategy)
    assert policy.reorder([]) == []
    request = req(1, 10)
    assert policy.reorder([request]) == [request]


@pytest.mark.parametrize(
    "name, expected",
    [
        ("default", [2, 4, 1, 3]),
        ("promote_shortest", [2, 4, 3, 1]),
        ("hrrn", [2, 4, 1, 3]),
    ],
)
def test_ordering_and_input_preservation(name, expected):
    requests = [req(1, 20), req(2, 50, priority=-2), req(3, 102, 100), req(4, 5, priority=-1)]
    original = list(requests)
    result = strategy(name).reorder(requests)
    assert [r.shm_req.group_req_id for r in result] == expected
    assert all(a is b for a, b in zip(requests, original))
    assert sorted(map(id, result)) == sorted(map(id, original))


def test_inference_priority_is_independent_from_router_priority():
    router_high_priority = req(1, 10, priority=0, router_priority=True)
    infer_high_priority = req(2, 10, priority=-1, router_priority=False)

    assert strategy().reorder([router_high_priority, infer_high_priority]) == [
        infer_high_priority,
        router_high_priority,
    ]


def test_promote_shortest_clamps_remaining_tokens_and_preserves_ties():
    requests = [req(9, 5, 6), req(8, 5, 10), req(7, 5, 5), req(6, 10, 8), req(1, 2)]
    assert strategy("promote_shortest").reorder(requests) == requests


def test_promote_shortest_ignores_finished_request_when_selecting_shortest():
    finished = req(1, 5, kv_len=5, finish_status=FinishStatus.FINISHED_LENGTH)
    long_request = req(2, 20)
    short_request = req(3, 2)

    assert strategy("promote_shortest").reorder([finished, long_request, short_request], no_decode=True) == [
        short_request,
        long_request,
        finished,
    ]


def test_promote_shortest_preserves_arrival_order_for_equal_positive_remaining_tokens():
    long_request = req(1, 20)
    first_short = req(3, 10, kv_len=8)
    second_short = req(2, 2)

    assert strategy("promote_shortest").reorder([long_request, first_short, second_short]) == [
        first_short,
        long_request,
        second_short,
    ]


@pytest.mark.parametrize("name", PREFILL_QUEUE_STRATEGIES)
@pytest.mark.parametrize("mode", [{}, {"no_decode": True}, {"strict_prefill": True}])
def test_finished_aborted_and_cleanup_requests_stay_last(name, mode):
    finished_normal = req(9, 5, kv_len=5, finish_status=FinishStatus.FINISHED_LENGTH)
    finished_high = req(8, 2, priority=-10, finish_status=FinishStatus.FINISHED_STOP)
    finished_decode = req(7, 5, kv_len=4, finish_status=FinishStatus.FINISHED_ERROR)
    aborted = req(6, 2, priority=-10, infer_aborted=True)
    cleanup = req(5, 5, kv_len=4, filter_mark=True)
    long_request = req(1, 20)
    short_request = req(2, 2)
    high_priority = req(3, 100, priority=-1)
    decode = req(4, 5, kv_len=4)
    active = [long_request, short_request, high_priority, decode]
    finished = [finished_normal, finished_high, finished_decode, aborted, cleanup]
    requests = [
        finished_normal,
        long_request,
        finished_high,
        short_request,
        finished_decode,
        high_priority,
        aborted,
        decode,
        cleanup,
    ]
    original = list(requests)
    policy = strategy(name)

    expected_active = policy.reorder(active, **mode)
    result = policy.reorder(requests, **mode)

    assert result == expected_active + finished
    assert all(a is b for a, b in zip(requests, original))
    assert sorted(map(id, result)) == sorted(map(id, original))
    for request in finished:
        assert not hasattr(request, "arrival_processed_prefill_tokens")
        assert not hasattr(request, "uncached_prefill_tokens")


@pytest.mark.parametrize("name", PREFILL_QUEUE_STRATEGIES)
def test_all_finished_requests_preserve_arrival_order(name):
    requests = [
        req(6, 2, finish_status=FinishStatus.FINISHED_ABORTED),
        req(5, 5, kv_len=5, finish_status=FinishStatus.FINISHED_LENGTH),
        req(4, 10, priority=-10, finish_status=FinishStatus.FINISHED_STOP),
        req(3, 5, kv_len=4, infer_aborted=True),
        req(2, 2, filter_mark=True),
        req(1, 5, kv_len=4, finish_status=FinishStatus.FINISHED_PD_DECODE_CAPACITY),
    ]

    assert strategy(name).reorder(requests) == requests


@pytest.mark.parametrize("pause_state", ["wait_pause", "paused"])
def test_promote_shortest_ignores_paused_request_when_selecting_shortest(pause_state):
    suspended = req(1, 2, **{pause_state: True})
    long_request = req(2, 20)
    short_request = req(3, 5)

    assert strategy("promote_shortest").reorder([suspended, long_request, short_request]) == [
        short_request,
        long_request,
        suspended,
    ]
    assert getattr(suspended, pause_state) is True


@pytest.mark.parametrize("name", PREFILL_QUEUE_STRATEGIES)
@pytest.mark.parametrize("mode", [{}, {"no_decode": True}, {"strict_prefill": True}])
def test_paused_requests_keep_arrival_order_at_tail(name, mode):
    paused_high = req(5, 2, priority=-10, paused=True)
    waiting = req(6, 5, kv_len=4, wait_pause=True)
    paused_normal = req(4, 2, paused=True)
    finished = req(7, 5, finish_status=FinishStatus.FINISHED_STOP)
    active = [req(1, 20), req(2, 5), req(3, 100, priority=-1)]
    requests = [paused_high, active[0], waiting, active[1], finished, paused_normal, active[2]]
    original = list(requests)
    policy = strategy(name)

    assert policy.reorder(requests, **mode) == policy.reorder(active, **mode) + [
        paused_high,
        waiting,
        finished,
        paused_normal,
    ]
    assert all(a is b for a, b in zip(requests, original))


@pytest.mark.parametrize("pause_state", ["wait_pause", "paused"])
def test_hrrn_initializes_paused_request_only_after_recovery(pause_state):
    suspended = req(1, 100, **{pause_state: True})
    active = req(2, 20)
    policy = strategy("hrrn", processed_prefill_tokens=1000)

    assert policy.reorder([suspended, active]) == [active, suspended]
    assert not hasattr(suspended, "arrival_processed_prefill_tokens")
    assert not hasattr(suspended, "uncached_prefill_tokens")

    setattr(suspended, pause_state, False)
    suspended.cur_kv_len = 90
    policy.backend.processed_prefill_tokens = 2000
    assert policy.reorder([suspended, active]) == [active, suspended]
    assert suspended.arrival_processed_prefill_tokens == 2000
    assert suspended.uncached_prefill_tokens == 10


def test_hrrn_preserves_existing_aging_snapshot_across_pause_and_recovery():
    request = req(1, 100)
    policy = strategy("hrrn", processed_prefill_tokens=1000)
    policy.reorder([request])

    request.paused = True
    request.cur_kv_len = 0
    policy.backend.processed_prefill_tokens = 2000
    assert policy.reorder([request]) == [request]
    assert request.paused is True

    request.paused = False
    request.cur_kv_len = 90
    policy.backend.processed_prefill_tokens = 3000
    assert policy.reorder([request]) == [request]
    assert request.arrival_processed_prefill_tokens == 1000
    assert request.uncached_prefill_tokens == 100


def test_promote_shortest_keeps_high_priority_requests_first():
    long_high_priority = req(1, 1000, priority=-1)
    short_normal_priority = req(2, 2, priority=0)

    assert strategy("promote_shortest").reorder([short_normal_priority, long_high_priority]) == [
        long_high_priority,
        short_normal_priority,
    ]


def test_promote_shortest_preserves_high_priority_arrival_order():
    first_high_priority = req(1, 100, priority=-1)
    second_high_priority = req(2, 100, priority=-2)

    assert strategy("promote_shortest").reorder([first_high_priority, second_high_priority]) == [
        first_high_priority,
        second_high_priority,
    ]


def test_promote_shortest_moves_only_one_normal_request_and_preserves_other_order():
    first_normal = req(1, 20)
    shortest_normal = req(2, 2)
    third_normal = req(3, 10)
    fourth_normal = req(4, 5)

    assert strategy("promote_shortest").reorder([first_normal, shortest_normal, third_normal, fourth_normal]) == [
        shortest_normal,
        first_normal,
        third_normal,
        fourth_normal,
    ]


def test_hrrn_prioritizes_short_work_for_equally_aged_requests():
    requests = [req(1, 100), req(2, 10), req(3, 2)]
    policy = strategy("hrrn")
    policy.reorder(requests)
    policy.backend.processed_prefill_tokens = 1000

    assert policy.reorder(requests) == [
        requests[2],
        requests[1],
        requests[0],
    ]


def test_hrrn_aging_allows_long_request_to_overtake_new_short_request():
    long_old = req(1, 100)
    short_new = req(2, 2)
    policy = strategy("hrrn")
    policy.reorder([long_old])
    policy.backend.processed_prefill_tokens = 100_000

    assert policy.reorder([short_new, long_old]) == [
        long_old,
        short_new,
    ]


def test_hrrn_aging_starts_at_first_sort_and_survives_later_sorts():
    long_request = req(1, 100)
    short_request = req(2, 2)
    requests = [short_request, long_request]
    policy = strategy("hrrn", processed_prefill_tokens=1000)

    # 首轮两个请求的等待量均为零，使用请求 ID 打破平局。
    assert policy.reorder(requests) == [long_request, short_request]
    policy.backend.processed_prefill_tokens = 2000
    # 后续排序保留首次快照，等龄请求中短请求的响应比更高。
    assert policy.reorder(requests) == [short_request, long_request]


def test_hrrn_binds_uncached_tokens_once_and_uses_request_id_tiebreaker():
    less_cached = req(2, 100, kv_len=0)
    equally_cached_later_id = req(3, 100, kv_len=90)
    more_cached = req(1, 100, kv_len=90)
    policy = strategy("hrrn")

    requests = [less_cached, equally_cached_later_id, more_cached]
    policy.reorder(requests)
    policy.backend.processed_prefill_tokens = 1000
    assert policy.reorder(requests) == [more_cached, equally_cached_later_id, less_cached]
    assert [request.uncached_prefill_tokens for request in requests] == [100, 10, 10]

    less_cached.cur_kv_len = 95
    more_cached.cur_kv_len = 0
    policy.backend.processed_prefill_tokens = 2000
    assert policy.reorder(requests) == [more_cached, equally_cached_later_id, less_cached]


def test_hrrn_keeps_internal_high_priority_requests_first():
    shortest_normal = req(1, 2, priority=0)
    long_high_priority = req(2, 1000, priority=-1)

    assert strategy("hrrn", processed_prefill_tokens=1000).reorder([shortest_normal, long_high_priority]) == [
        long_high_priority,
        shortest_normal,
    ]


@pytest.mark.parametrize("name", PREFILL_QUEUE_STRATEGIES)
def test_decode_requests_stay_left_and_are_not_reordered(name):
    decode_normal = req(1, 5, kv_len=4, priority=0)
    prefill_high = req(2, 100, priority=-2)
    decode_high = req(3, 5, kv_len=4, priority=-10)
    prefill_normal = req(4, 2, priority=0)

    assert strategy(name).reorder([decode_normal, prefill_normal, decode_high, prefill_high]) == [
        decode_normal,
        decode_high,
        prefill_high,
        prefill_normal,
    ]


def test_no_decode_and_strict_prefill_match_backend_classification_rules():
    boundary_req = req(1, 5, kv_len=4, priority=0)
    high_priority_prefill = req(2, 10, priority=-1)

    policy = strategy()
    assert policy.reorder([boundary_req, high_priority_prefill]) == [boundary_req, high_priority_prefill]
    assert policy.reorder([boundary_req, high_priority_prefill], no_decode=True) == [
        high_priority_prefill,
        boundary_req,
    ]
    assert policy.reorder([boundary_req, high_priority_prefill], strict_prefill=True) == [
        high_priority_prefill,
        boundary_req,
    ]


@pytest.mark.parametrize("name", PREFILL_QUEUE_STRATEGIES)
def test_cli_strategy_selection(name):
    args = make_argument_parser().parse_args(["--prefill_queue_strategy", name])
    backend = SimpleNamespace(args=args)
    policy = create_prefill_queue_strategy(backend)
    assert isinstance(policy, PREFILL_QUEUE_STRATEGIES[name])
    assert policy.backend is backend


def test_cli_default_and_invalid_strategy():
    parser = make_argument_parser()
    assert parser.parse_args([]).prefill_queue_strategy == "auto"
    assert parser.parse_args(["--prefill_queue_strategy", "auto"]).prefill_queue_strategy == "auto"
    assert set(PREFILL_QUEUE_STRATEGIES) == {"default", "promote_shortest", "hrrn"}
    for name in ("unknown", "none"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--prefill_queue_strategy", name])
        with pytest.raises(ValueError, match="Unknown prefill queue strategy"):
            strategy(name)


def test_strategy_base_is_abstract():
    with pytest.raises(TypeError):
        PrefillQueueStrategy(SimpleNamespace())
