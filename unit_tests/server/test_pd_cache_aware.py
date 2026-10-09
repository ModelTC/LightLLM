import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from lightllm.server.pd_io_struct import PD_Client_Obj

from lightllm.server.httpserver_for_pd_master.pd_selector.cache_aware import (
    BalanceRelThresholdController,
    CacheAwareConfig,
    CacheAwarePolicy,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.pd_selector import (
    LoadBalancedCacheAwareSelector,
    PDSelectionExtraInfo,
)


def _worker(address: str, dispatched_prompt_chars: int = 0, dispatched_req_num: int = 0):
    node = PD_Client_Obj(
        node_id=1,
        client_ip_port=address,
        mode="prefill",
        start_args={"dp": 1, "nnodes": 1, "disable_dynamic_prompt_cache": False},
    )
    rank = node.dp_ranks[0]
    rank.dispatched_prompt_chars = dispatched_prompt_chars
    rank.dispatched_req_num = dispatched_req_num
    return rank


def test_balance_threshold_controller_adjusts_threshold_each_window():
    config = CacheAwareConfig(cache_hit_rate_window_size=3, balance_rel_threshold_step=0.1)
    controller = BalanceRelThresholdController()

    for cache_hit_rate, expected_threshold in (
        (0.5, 1.5),
        (0.6, 1.6),
        (0.4, 1.5),
        (0.4, 1.5),
    ):
        for _ in range(config.cache_hit_rate_window_size):
            controller.append(cache_hit_rate)
        controller.update_config(config)
        assert config.balance_rel_threshold == pytest.approx(expected_threshold)


@pytest.mark.parametrize(
    ("initial_threshold", "first_hit_rate", "second_hit_rate", "expected_threshold"),
    (
        (1.95, 0.5, 0.6, 2.0),
        (1.05, 0.5, 0.4, 1.0),
    ),
)
def test_balance_threshold_controller_limits_threshold_range(
    initial_threshold,
    first_hit_rate,
    second_hit_rate,
    expected_threshold,
):
    config = CacheAwareConfig(
        balance_rel_threshold=initial_threshold,
        cache_hit_rate_window_size=1,
        balance_rel_threshold_step=0.1,
    )
    controller = BalanceRelThresholdController()

    controller.append(first_hit_rate)
    controller.update_config(config)
    controller.append(second_hit_rate)
    controller.update_config(config)

    assert config.balance_rel_threshold == pytest.approx(expected_threshold)


def test_cache_aware_updates_threshold_from_inference_cache_hit_rate():
    policy = CacheAwarePolicy(CacheAwareConfig(cache_hit_rate_window_size=2))

    for _ in range(2):
        policy.record_prompt_cache_hit_rate(0.25)
    assert policy.config.balance_rel_threshold == 1.5
    for _ in range(2):
        policy.record_prompt_cache_hit_rate(0.75)

    assert policy.config.balance_rel_threshold == pytest.approx(1.55)


def test_cache_aware_selector_returns_selected_nodes_and_estimated_cache_hit_rate():
    selector = LoadBalancedCacheAwareSelector(pd_manager=None)
    p_rank = _worker("10.0.0.1:8000")
    p_node = p_rank.node
    d_node = SimpleNamespace(
        client_ip_port="10.0.0.2:8000",
        run_status=SimpleNamespace(total_token_usage_rate=0.0),
    )
    selector.update_nodes([p_node], [d_node])
    prompt = "x" * 1025
    with patch(
        "lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree.time.monotonic",
        return_value=123.0,
    ):
        selector.policy.prompt_cache_tree.insert(prompt, p_rank.cache_key)

    selected_p_node, selected_d_node, selection_extra_info = selector.select_p_d_node(prompt, None, None)

    assert selected_p_node is p_node
    assert selected_d_node is d_node
    assert selection_extra_info == PDSelectionExtraInfo(
        estimated_cache_hit_rate=pytest.approx(1.0),
        cache_last_insert_time=123.0,
        prefill_dp_rank=0,
    )


def test_cache_aware_reinsert_refreshes_matched_node_time():
    policy = CacheAwarePolicy()
    worker = _worker("10.0.0.1:8000")
    prompt = "shared prefix " * 100

    with patch(
        "lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree.time.monotonic",
        return_value=100.0,
    ):
        policy.insert_prompt_cache(prompt, worker)
        first_result = policy.prompt_cache_tree.prefix_match(prompt)
    with patch(
        "lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree.time.monotonic",
        return_value=160.0,
    ):
        policy.insert_prompt_cache(prompt, worker)
        second_result = policy.prompt_cache_tree.prefix_match(prompt)

    assert first_result.last_insert_time == 100.0
    assert second_result.last_insert_time == 160.0


def test_cache_aware_inserts_prompt_only_when_explicitly_recorded():
    policy = CacheAwarePolicy()
    worker = _worker("10.0.0.1:8000")
    prompt = "new prompt"

    assert policy.select_worker([worker], prompt) is worker
    assert policy.prompt_cache_tree.prefix_match(prompt).prefill_node is None

    policy.insert_prompt_cache(prompt, worker)

    assert policy.prompt_cache_tree.prefix_match(prompt).prefill_node == worker.cache_key


def test_cache_aware_keeps_cache_worker_when_inflight_load_is_balanced():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=110, dispatched_req_num=2)
    least_loaded_worker = _worker("10.0.0.2:8000", dispatched_prompt_chars=100, dispatched_req_num=2)
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.cache_key)

    selected_worker = policy.select_worker([cache_worker, least_loaded_worker], prompt)
    cache_info = policy.get_estimated_cache_info(selected_worker, prompt)

    assert selected_worker is cache_worker
    # 前缀树每 512 个字符抽样一次，命中率保持原有的保守估算方式。
    assert cache_info.estimated_cache_hit_rate == pytest.approx(1025 / len(prompt))


def test_cache_aware_uses_least_loaded_worker_when_cache_worker_is_overloaded():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=2000, dispatched_req_num=2)
    least_loaded_worker = _worker("10.0.0.2:8000", dispatched_prompt_chars=100, dispatched_req_num=2)
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.cache_key)

    selected_worker = policy.select_worker([cache_worker, least_loaded_worker], prompt)
    cache_info = policy.get_estimated_cache_info(selected_worker, prompt)

    assert selected_worker is least_loaded_worker
    assert cache_info.estimated_cache_hit_rate == 0.0


def test_cache_aware_keeps_cache_worker_when_both_workers_are_idle():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000")
    other_worker = _worker("10.0.0.2:8000")
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.cache_key)

    selected_worker = policy.select_worker([other_worker, cache_worker], prompt)

    assert selected_worker is cache_worker


def test_cache_aware_forces_idle_worker_over_busy_cache_worker():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=100, dispatched_req_num=1)
    idle_worker = _worker("10.0.0.2:8000")
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.cache_key)

    selected_worker = policy.select_worker([cache_worker, idle_worker], prompt)

    assert selected_worker is idle_worker


def test_cache_aware_matches_cache_only_within_idle_workers():
    policy = CacheAwarePolicy()
    busy_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=100, dispatched_req_num=1)
    cache_idle_worker = _worker("10.0.0.2:8000")
    other_idle_worker = _worker("10.0.0.3:8000")
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_idle_worker.cache_key)

    selected_worker = policy.select_worker([other_idle_worker, busy_worker, cache_idle_worker], prompt)

    assert selected_worker is cache_idle_worker


def test_master_selects_local_dp_rank_not_tp_rank():
    node = PD_Client_Obj(
        node_id=1,
        client_ip_port="10.0.0.1:8000",
        mode="prefill",
        start_args={"dp": 8, "nnodes": 2, "tp": 16, "disable_dynamic_prompt_cache": False},
    )
    selector = LoadBalancedCacheAwareSelector(None)
    d_node = SimpleNamespace(run_status=SimpleNamespace(total_token_usage_rate=0), client_ip_port="d:8000")
    selector.update_nodes([node], [d_node])
    prompt = "x" * 1025
    selector.insert_prompt_cache(prompt, node, 3)

    selected_node, _, extra = selector.select_p_d_node(prompt, None, None)

    assert len(node.dp_ranks) == 4
    assert selected_node is node
    assert extra.prefill_dp_rank == 3
    assert extra.estimated_cache_hit_rate == 1.0
    node.dp_ranks[3].dispatched_req_num = 1
    node.dp_ranks[3].dispatched_prompt_chars = 100
    _, _, extra = selector.select_p_d_node(prompt, None, None)
    assert extra.prefill_dp_rank != 3
    assert extra.estimated_cache_hit_rate == 0.0


def test_reconnecting_node_does_not_inherit_rank_cache_history():
    rank = _worker("10.0.0.1:8000")
    replacement = _worker("10.0.0.1:8000")
    policy = CacheAwarePolicy()
    prompt = "x" * 1025
    policy.insert_prompt_cache(prompt, rank)

    assert replacement.cache_key != rank.cache_key
    assert policy.get_estimated_cache_info(replacement, prompt).estimated_cache_hit_rate == 0.0
    assert policy._get_cache_worker([replacement], prompt) is None


def test_capacity_pressure_overrides_rank_cache_affinity():
    cache_rank = _worker("p:8000")
    other_rank = _worker("other:8000", dispatched_req_num=1)
    policy = CacheAwarePolicy()
    prompt = "x" * 1025
    policy.insert_prompt_cache(prompt, cache_rank)
    cache_rank.token_usage_rate = 1.1

    assert policy.select_worker([cache_rank, other_rank], prompt) is other_rank
    other_rank.token_usage_rate = 1.2
    assert policy.select_worker([cache_rank, other_rank], prompt) is cache_rank


def test_disabled_cache_does_not_record_rank_affinity():
    rank = _worker("p:8000")
    rank.node.start_args["disable_dynamic_prompt_cache"] = True
    selector = LoadBalancedCacheAwareSelector(None)
    selector.insert_prompt_cache("x" * 1025, rank.node, 0)
    assert selector.policy.prompt_cache_tree.prefix_match("x" * 1025).prefill_node is None


@pytest.mark.parametrize("inflight_req_num", [0, 2])
def test_serial_cold_prompts_spread_across_equal_load_dp_ranks(monkeypatch, inflight_req_num):
    monkeypatch.setattr(random, "choice", random.Random(0).choice)
    node = PD_Client_Obj(1, "p:8000", "prefill", {"dp": 4, "nnodes": 1, "disable_dynamic_prompt_cache": False})
    selector = LoadBalancedCacheAwareSelector(None)
    d_node = SimpleNamespace(run_status=SimpleNamespace(total_token_usage_rate=0), client_ip_port="d:8000")
    selector.update_nodes([node], [d_node])
    initial_load = 100 if inflight_req_num else 0
    for rank in node.dp_ranks:
        rank.dispatched_req_num = inflight_req_num
        rank.dispatched_prompt_chars = initial_load
    selected_ranks = []
    for index in range(16):
        prompt = chr(0x400 + index) * 1025
        _, _, extra = selector.select_p_d_node(prompt, None, None)
        assert extra.estimated_cache_hit_rate == 0.0
        rank = node.dp_ranks[extra.prefill_dp_rank]
        selected_ranks.append(rank.dp_rank)
        rank.dispatched_req_num += 1
        rank.dispatched_prompt_chars += len(prompt)
        selector.insert_prompt_cache(prompt, node, rank.dp_rank)
        rank.dispatched_req_num -= 1
        rank.dispatched_prompt_chars -= len(prompt)

    assert set(selected_ranks) == {0, 1, 2, 3}
    assert all(rank.dispatched_req_num == inflight_req_num for rank in node.dp_ranks)
    assert all(rank.dispatched_prompt_chars == initial_load for rank in node.dp_ranks)


@pytest.mark.parametrize("all_busy", [False, True])
def test_random_tie_break_only_considers_least_loaded_candidates(monkeypatch, all_busy):
    node = PD_Client_Obj(1, "p:8000", "prefill", {"dp": 3, "nnodes": 1})
    for rank in node.dp_ranks:
        rank.dispatched_req_num = 2 if all_busy else 0
        rank.dispatched_prompt_chars = 5 if all_busy else 0
    node.dp_ranks[1].dispatched_req_num = 2
    node.dp_ranks[1].dispatched_prompt_chars = 10

    def choose(candidates):
        assert [rank.dp_rank for rank in candidates] == [0, 2]
        return candidates[-1]

    monkeypatch.setattr(random, "choice", choose)
    assert CacheAwarePolicy().select_worker(node.dp_ranks, "cold prompt") is node.dp_ranks[2]
