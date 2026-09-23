from types import SimpleNamespace

import pytest

from lightllm.server.api_cli import make_argument_parser
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.mode_backend.prefill_queue_strategy import (
    PREFILL_QUEUE_STRATEGIES,
    PrefillQueueStrategy,
    create_prefill_queue_strategy,
)


def req(group_id, input_len, kv_len=0, priority=0, router_priority=False, total_len=None):
    return SimpleNamespace(
        cur_kv_len=kv_len,
        shm_req=SimpleNamespace(
            input_len=input_len,
            group_req_id=group_id,
            sample_params=SimpleNamespace(
                high_priority_request=router_priority,
                infer_high_priority=priority,
            ),
        ),
        get_cur_total_len=lambda: input_len if total_len is None else total_len,
    )


def strategy(name="default"):
    backend = ModeBackend.__new__(ModeBackend)
    backend.args = SimpleNamespace(prefill_queue_strategy=name)
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
    assert parser.parse_args([]).prefill_queue_strategy == "default"
    assert set(PREFILL_QUEUE_STRATEGIES) == {"default", "promote_shortest"}
    with pytest.raises(SystemExit):
        parser.parse_args(["--prefill_queue_strategy", "unknown"])
    with pytest.raises(ValueError, match="Unknown prefill queue strategy"):
        strategy("unknown")


def test_strategy_base_is_abstract():
    with pytest.raises(TypeError):
        PrefillQueueStrategy(SimpleNamespace())
