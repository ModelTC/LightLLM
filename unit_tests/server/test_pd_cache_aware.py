import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lightllm.server.httpserver_for_pd_master.pd_selector.cache_aware import (
    BalanceRelThresholdController,
    CacheAwareConfig,
    CacheAwarePolicy,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.pd_selector import (
    LoadBalancedCacheAwareSelector,
    PDSelectionExtraInfo,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree import PromptCacheTree


def test_prompt_cache_tree_long_prompt_insert_match_and_reinsert():
    tree = PromptCacheTree()
    # The sampled key must exceed the actual interpreter recursion limit.
    prompt = "x" * ((sys.getrecursionlimit() + 512) * tree.sample_stride + 1)
    tree.insert(prompt, "P0")
    result = tree.prefix_match(prompt)
    assert result.prefill_node == "P0"
    assert result.matched_char_count == len(prompt)

    tree.insert(prompt, "P1")
    result = tree.prefix_match(prompt + "different suffix")
    assert result.prefill_node == "P1"
    assert result.matched_char_count == len(prompt)
    assert len(tree._leaf_lru) == 1


def test_prompt_cache_tree_long_prompt_eviction_and_branch_recency():
    tree = PromptCacheTree(sample_stride=1, max_node_count=8, evict_node_batch=1)
    tree.insert("abc-old", "P0")
    tree.insert("abc-new", "P1")
    assert tree.prefix_match("abc-new").prefill_node == "P1"
    # Eviction must still run after a key longer than the recursion limit.
    tree.insert("z" * (sys.getrecursionlimit() + 512), "P2")
    assert tree._node_count <= tree.max_node_count
    assert tree.prefix_match("abc-old").matched_char_count == 0
    assert tree.prefix_match("abc-new").matched_char_count == 0
    assert len(tree._leaf_lru) <= 1


def test_prompt_cache_tree_extends_leaf_and_reinserts_internal_prefix():
    tree = PromptCacheTree(sample_stride=1, max_node_count=6, evict_node_batch=1)
    tree.insert("ab", "P0")
    tree.insert("abcd", "P1")
    tree.insert("abef", "P2")
    tree.insert("ab", "P3")

    assert tree.prefix_match("ab").prefill_node == "P3"
    assert tree.prefix_match("abcd").prefill_node == "P1"
    tree.insert("abgh", "P4")

    assert tree.prefix_match("abcd").matched_char_count == 2
    assert tree.prefix_match("abef").matched_char_count == 3
    assert tree.prefix_match("abgh").matched_char_count == 4
    assert tree.prefix_match("abgh").prefill_node == "P4"


def test_prompt_cache_tree_refreshes_branch_before_eviction():
    tree = PromptCacheTree(sample_stride=1, max_node_count=8, evict_node_batch=1)
    tree.insert("abcx", "P0")
    tree.insert("abcy", "P1")
    tree.insert("abcw", "P2")
    tree.insert("abcx", "P3")
    tree.insert("abcz12", "P4")

    assert tree.prefix_match("abcy").matched_char_count == 3
    assert tree.prefix_match("abcw").matched_char_count == 3
    assert tree.prefix_match("abcx").matched_char_count == 4
    assert tree.prefix_match("abcx").prefill_node == "P3"
    assert tree.prefix_match("abcz12").matched_char_count == 6


@pytest.mark.parametrize("prefix", ("abc", "🙂中文"))
def test_prompt_cache_tree_split_preserves_suffix_worker_and_time(prefix):
    tree = PromptCacheTree(sample_stride=1)
    with patch("time.monotonic", return_value=100.0):
        tree.insert(prefix + "def", "P0")
    with patch("time.monotonic", return_value=200.0):
        tree.insert(prefix + "XYZ", "P1")

    old_suffix = tree.prefix_match(prefix + "de?")
    assert old_suffix.matched_char_count == len(prefix) + 2
    assert old_suffix.prefill_node == "P0"
    assert old_suffix.last_insert_time == 100.0
    shared_prefix = tree.prefix_match(prefix + "?")
    assert shared_prefix.matched_char_count == len(prefix)
    assert shared_prefix.prefill_node == "P1"
    assert shared_prefix.last_insert_time == 200.0

    with patch("time.monotonic", return_value=300.0):
        tree.insert(prefix[:2], "P2")
    assert tree.prefix_match(prefix[:2]).last_insert_time == 300.0
    assert tree.prefix_match(prefix[:2]).prefill_node == "P2"
    assert tree.prefix_match(prefix + "de?") == old_suffix
    assert tree.prefix_match(prefix + "?") == shared_prefix


def test_prompt_cache_tree_trims_long_leaf_to_logical_capacity():
    tree = PromptCacheTree(sample_stride=1, max_node_count=4, evict_node_batch=1)
    tree.insert("abcdef", "P0")
    result = tree.prefix_match("abcdef")
    assert result.matched_char_count == 3
    assert result.prefill_node == "P0"
    tree.insert("abcXYZ", "P1")
    result = tree.prefix_match("abcXYZ")
    assert result.matched_char_count == 3
    assert result.prefill_node == "P1"


@pytest.mark.parametrize("stride", (1, 7))
def test_prompt_cache_tree_matches_edge_prefix_boundaries(stride):
    for edge_len in (2, 7, 31, 32, 33, 65):
        tree = PromptCacheTree(sample_stride=stride)
        key = ("ab中🙂" * 17)[:edge_len]
        text = "".join(char + " " * (stride - 1) for char in key)
        with patch("time.monotonic", return_value=123.0):
            tree.insert(text, "P0")
        for matched in (0, 1, 2, 3, 7, 15, 31, 32, 33, 63, 64, edge_len):
            if matched > edge_len:
                continue
            prefix = text[: matched * stride]
            for query in (prefix, prefix + "!" + text[matched * stride + 1 :]):
                result = tree.prefix_match(query)
                assert result.matched_char_count == ((matched - 1) * stride + 1 if matched else 0)
                assert result.input_char_count == len(query)
                assert result.prefill_node == ("P0" if matched else None)
                assert result.last_insert_time == (123.0 if matched else None)


def test_prompt_cache_tree_coalesces_appended_prompts_and_refreshed_prefixes():
    tree = PromptCacheTree(sample_stride=1)
    prompt = ""
    for index in range(20):
        prompt += f"message {index};"
        tree.insert(prompt, "P0")
        assert len(tree.root.children) == 1
        assert not next(iter(tree.root.children.values())).children
        assert tree.prefix_match(prompt).matched_char_count == len(prompt)

    with patch("time.monotonic", return_value=100.0):
        tree.insert(prompt[:10], "P1")
    assert tree.prefix_match(prompt[:10]).prefill_node == "P1"
    assert tree.prefix_match(prompt).prefill_node == "P0"
    with patch("time.monotonic", return_value=200.0):
        tree.insert(prompt, "P2")
    assert not next(iter(tree.root.children.values())).children
    assert tree.prefix_match(prompt[:10]).prefill_node == "P2"
    assert tree.prefix_match(prompt[:10]).last_insert_time == 200.0
    assert tree.prefix_match(prompt).prefill_node == "P2"


def _worker(address: str, dispatched_prompt_chars: int = 0, dispatched_req_num: int = 0):
    return SimpleNamespace(
        client_ip_port=address,
        connection_key=address,
        dispatched_prompt_chars=dispatched_prompt_chars,
        dispatched_req_num=dispatched_req_num,
    )


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
    p_node = _worker("10.0.0.1:8000")
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
        selector.policy.prompt_cache_tree.insert(prompt, p_node.client_ip_port)

    selected_p_node, selected_d_node, selection_extra_info = selector.select_p_d_node(prompt, None, None)

    assert selected_p_node is p_node
    assert selected_d_node is d_node
    assert selection_extra_info == PDSelectionExtraInfo(
        estimated_cache_hit_rate=pytest.approx(1.0),
        cache_last_insert_time=123.0,
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

    assert policy.prompt_cache_tree.prefix_match(prompt).prefill_node == worker.client_ip_port


def test_cache_aware_keeps_cache_worker_when_inflight_load_is_balanced():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=110, dispatched_req_num=2)
    least_loaded_worker = _worker("10.0.0.2:8000", dispatched_prompt_chars=100, dispatched_req_num=2)
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.client_ip_port)

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
    policy.prompt_cache_tree.insert(prompt, cache_worker.client_ip_port)

    selected_worker = policy.select_worker([cache_worker, least_loaded_worker], prompt)
    cache_info = policy.get_estimated_cache_info(selected_worker, prompt)

    assert selected_worker is least_loaded_worker
    assert cache_info.estimated_cache_hit_rate == 0.0


def test_cache_aware_keeps_cache_worker_when_both_workers_are_idle():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000")
    other_worker = _worker("10.0.0.2:8000")
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.client_ip_port)

    selected_worker = policy.select_worker([other_worker, cache_worker], prompt)

    assert selected_worker is cache_worker


def test_cache_aware_forces_idle_worker_over_busy_cache_worker():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=100, dispatched_req_num=1)
    idle_worker = _worker("10.0.0.2:8000")
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.client_ip_port)

    selected_worker = policy.select_worker([cache_worker, idle_worker], prompt)

    assert selected_worker is idle_worker


def test_cache_aware_matches_cache_only_within_idle_workers():
    policy = CacheAwarePolicy()
    busy_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=100, dispatched_req_num=1)
    cache_idle_worker = _worker("10.0.0.2:8000")
    other_idle_worker = _worker("10.0.0.3:8000")
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_idle_worker.client_ip_port)

    selected_worker = policy.select_worker([other_idle_worker, busy_worker, cache_idle_worker], prompt)

    assert selected_worker is cache_idle_worker


def test_cache_aware_selector_matches_once_when_load_redirects_request():
    selector = LoadBalancedCacheAwareSelector(pd_manager=None)
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=2000, dispatched_req_num=1)
    other_worker = _worker("10.0.0.2:8000", dispatched_prompt_chars=100, dispatched_req_num=1)
    d_node = SimpleNamespace(client_ip_port="10.0.0.3:8000", run_status=SimpleNamespace(total_token_usage_rate=0.0))
    selector.update_nodes([cache_worker, other_worker], [d_node])
    prompt = "x" * 1025
    tree = selector.policy.prompt_cache_tree
    tree.insert(prompt, cache_worker.connection_key)
    with patch.object(tree, "prefix_match", wraps=tree.prefix_match) as match:
        selected, _, info = selector.select_p_d_node(prompt, None, None)
    assert selected is other_worker
    assert info == PDSelectionExtraInfo()
    match.assert_called_once_with(prompt)


def test_cache_aware_selector_keeps_concurrent_request_cache_info_separate():
    selector = LoadBalancedCacheAwareSelector(pd_manager=None)
    workers = [_worker("10.0.0.1:8000"), _worker("10.0.0.2:8000")]
    d_node = SimpleNamespace(client_ip_port="10.0.0.3:8000", run_status=SimpleNamespace(total_token_usage_rate=0.0))
    selector.update_nodes(workers, [d_node])
    prompts = ["x" * 1025, "y" * 1537]
    tree = selector.policy.prompt_cache_tree
    for index, prompt in enumerate(prompts):
        with patch("time.monotonic", return_value=100.0 + index):
            tree.insert(prompt, workers[index].connection_key)

    barrier = Barrier(8)
    prefix_match = tree.prefix_match

    def concurrent_match(prompt):
        barrier.wait(timeout=10)
        return prefix_match(prompt)

    with patch.object(tree, "prefix_match", side_effect=concurrent_match) as match:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda index: selector.select_p_d_node(prompts[index], None, None), [0, 1] * 4))
    assert match.call_count == 8
    for index, (selected, _, info) in zip([0, 1] * 4, results):
        assert selected is workers[index]
        assert info == PDSelectionExtraInfo(estimated_cache_hit_rate=1.0, cache_last_insert_time=100.0 + index)
