import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree_client import (
    create_prompt_cache_tree_client,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.cache_aware import (
    BalanceRelThresholdController,
    CacheAwareConfig,
    CacheAwarePolicy,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree import PromptCacheTree
from lightllm.server.httpserver_for_pd_master.pd_selector.pd_selector import (
    LoadBalancedCacheAwareSelector,
    PDSelectionExtraInfo,
)

pytestmark = pytest.mark.usefixtures("single_http_worker_start_args")


def _worker(address: str, dispatched_prompt_chars: int = 0, dispatched_req_num: int = 0):
    return SimpleNamespace(
        client_ip_port=address,
        connection_key=address,
        dispatched_prompt_chars=dispatched_prompt_chars,
        dispatched_req_num=dispatched_req_num,
    )


def _local_policy(config):
    return CacheAwarePolicy(prompt_cache_tree=create_prompt_cache_tree_client(config), config=config)


def _insert_prompt(policy, prompt, worker):
    sample_stride = policy.config.sample_stride
    asyncio.run(policy.prompt_cache_tree.insert_sampled(prompt[::sample_stride], worker.connection_key, sample_stride))


def _prefix_match(policy, prompt):
    sample_stride = policy.config.sample_stride
    return asyncio.run(
        policy.prompt_cache_tree.prefix_match_sampled(prompt[::sample_stride], len(prompt), sample_stride)
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
    policy = _local_policy(CacheAwareConfig(cache_hit_rate_window_size=2))

    for _ in range(2):
        policy.record_prompt_cache_hit_rate(0.25)
    assert policy.config.balance_rel_threshold == 1.5
    for _ in range(2):
        policy.record_prompt_cache_hit_rate(0.75)

    assert policy.config.balance_rel_threshold == pytest.approx(1.55)


@pytest.mark.parametrize("operation", ["insert", "prefix_match"])
def test_prompt_cache_tree_rejects_mismatched_sample_stride(operation):
    tree = PromptCacheTree(sample_stride=4)

    with pytest.raises(AssertionError, match="sample_stride mismatch"):
        if operation == "insert":
            tree.insert_sampled("po", "prefill", 2)
        else:
            tree.prefix_match_sampled("po", len("prompt"), 2)


def test_prompt_cache_tree_handles_key_deeper_than_python_recursion_limit():
    tree = PromptCacheTree(sample_stride=1)
    key = "x" * 5000

    tree.insert_sampled(key, "prefill", sample_stride=1)
    result = tree.prefix_match_sampled(key, input_char_count=len(key), sample_stride=1)

    assert result.prefill_node == "prefill"
    assert result.matched_char_count == len(key)


def test_prompt_cache_tree_empty_key_does_not_create_nodes():
    tree = PromptCacheTree(sample_stride=1)

    tree.insert_sampled("", "prefill", sample_stride=1)
    result = tree.prefix_match_sampled("", input_char_count=0, sample_stride=1)

    assert tree._node_count == 0
    assert not tree.root.children
    assert result.prefill_node is None
    assert result.matched_char_count == 0
    assert result.input_char_count == 0
    assert result.last_insert_time is None


def test_prompt_cache_tree_zero_capacity_evicts_the_complete_path():
    tree = PromptCacheTree(sample_stride=1, max_node_count=0, evict_node_batch=1)

    tree.insert_sampled("你好🙂", "prefill", sample_stride=1)
    result = tree.prefix_match_sampled("你好🙂", input_char_count=3, sample_stride=1)

    assert tree._node_count == 0
    assert not tree.root.children
    assert result.prefill_node is None
    assert result.matched_char_count == 0
    assert result.input_char_count == 3
    assert result.last_insert_time is None


def test_prompt_cache_tree_matches_unicode_partial_prefix():
    tree = PromptCacheTree(sample_stride=1)
    tree.insert_sampled("你好🙂", "prefill", sample_stride=1)

    result = tree.prefix_match_sampled("你好🙃", input_char_count=3, sample_stride=1)

    assert result.prefill_node == "prefill"
    assert result.matched_char_count == 2
    assert result.input_char_count == 3
    assert result.last_insert_time is not None


def test_prompt_cache_tree_iterative_insert_keeps_lru_eviction_order():
    tree = PromptCacheTree(sample_stride=1, max_node_count=3, evict_node_batch=1)
    tree.insert_sampled("ab", "prefill-ab", sample_stride=1)
    tree.insert_sampled("ac", "prefill-ac", sample_stride=1)

    # 再次访问 ab，使 ac 成为最早未使用的叶节点。
    tree.insert_sampled("ab", "prefill-ab", sample_stride=1)
    # 插入 d 后超出容量，依次驱逐 ac 和 ab；共同父节点 a 随后成为叶节点。
    tree.insert_sampled("d", "prefill-d", sample_stride=1)

    assert tree._node_count == 2
    assert set(tree.root.children) == {"a", "d"}
    result = tree.prefix_match_sampled("ab", input_char_count=2, sample_stride=1)
    assert result.prefill_node == "prefill-ab"
    assert result.matched_char_count == 1


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
        _insert_prompt(selector.policy, prompt, p_node)

    selected_p_node, selected_d_node, selection_extra_info = asyncio.run(selector.select_p_d_node(prompt, None, None))

    assert selected_p_node is p_node
    assert selected_d_node is d_node
    assert selection_extra_info == PDSelectionExtraInfo(
        estimated_cache_hit_rate=pytest.approx(1.0),
        cache_last_insert_time=123.0,
    )


def test_cache_aware_reinsert_refreshes_matched_node_time():
    policy = _local_policy(CacheAwareConfig())
    worker = _worker("10.0.0.1:8000")
    prompt = "shared prefix " * 100

    with patch(
        "lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree.time.monotonic",
        return_value=100.0,
    ):
        _insert_prompt(policy, prompt, worker)
        first_result = _prefix_match(policy, prompt)
    with patch(
        "lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree.time.monotonic",
        return_value=160.0,
    ):
        _insert_prompt(policy, prompt, worker)
        second_result = _prefix_match(policy, prompt)

    assert first_result.last_insert_time == 100.0
    assert second_result.last_insert_time == 160.0


def test_cache_aware_inserts_prompt_only_when_explicitly_recorded():
    policy = _local_policy(CacheAwareConfig())
    worker = _worker("10.0.0.1:8000")
    prompt = "new prompt"

    selected_worker, cache_info = asyncio.run(policy.select_worker([worker], prompt))
    assert selected_worker is worker
    assert cache_info == PDSelectionExtraInfo()
    assert _prefix_match(policy, prompt).prefill_node is None

    _insert_prompt(policy, prompt, worker)

    assert _prefix_match(policy, prompt).prefill_node == worker.connection_key


def test_cache_aware_keeps_cache_worker_when_inflight_load_is_balanced():
    policy = _local_policy(CacheAwareConfig())
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=110, dispatched_req_num=2)
    least_loaded_worker = _worker("10.0.0.2:8000", dispatched_prompt_chars=100, dispatched_req_num=2)
    prompt = "shared prefix " * 100
    _insert_prompt(policy, prompt, cache_worker)

    selected_worker, cache_info = asyncio.run(policy.select_worker([cache_worker, least_loaded_worker], prompt))

    assert selected_worker is cache_worker
    # 前缀树每 512 个字符抽样一次，命中率保持原有的保守估算方式。
    assert cache_info.estimated_cache_hit_rate == pytest.approx(1025 / len(prompt))


def test_cache_aware_uses_least_loaded_worker_when_cache_worker_is_overloaded():
    policy = _local_policy(CacheAwareConfig())
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=2000, dispatched_req_num=2)
    least_loaded_worker = _worker("10.0.0.2:8000", dispatched_prompt_chars=100, dispatched_req_num=2)
    prompt = "shared prefix " * 100
    _insert_prompt(policy, prompt, cache_worker)

    selected_worker, cache_info = asyncio.run(policy.select_worker([cache_worker, least_loaded_worker], prompt))

    assert selected_worker is least_loaded_worker
    assert cache_info.estimated_cache_hit_rate == 0.0


def test_cache_aware_keeps_cache_worker_when_both_workers_are_idle():
    policy = _local_policy(CacheAwareConfig())
    cache_worker = _worker("10.0.0.1:8000")
    other_worker = _worker("10.0.0.2:8000")
    prompt = "shared prefix " * 100
    _insert_prompt(policy, prompt, cache_worker)

    selected_worker, _ = asyncio.run(policy.select_worker([other_worker, cache_worker], prompt))

    assert selected_worker is cache_worker


def test_cache_aware_forces_idle_worker_over_busy_cache_worker():
    policy = _local_policy(CacheAwareConfig())
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=100, dispatched_req_num=1)
    idle_worker = _worker("10.0.0.2:8000")
    prompt = "shared prefix " * 100
    _insert_prompt(policy, prompt, cache_worker)

    selected_worker, _ = asyncio.run(policy.select_worker([cache_worker, idle_worker], prompt))

    assert selected_worker is idle_worker


def test_cache_aware_randomly_selects_an_idle_worker_without_cache_match():
    policy = _local_policy(CacheAwareConfig())
    first = _worker("10.0.0.1:8000")
    second = _worker("10.0.0.2:8000")

    with patch(
        "lightllm.server.httpserver_for_pd_master.pd_selector.cache_aware.random.choice", return_value=second
    ) as choice:
        selected_worker, _ = asyncio.run(policy.select_worker([first, second], "new prompt"))

    assert selected_worker is second
    choice.assert_called_once_with([first, second])


def test_cache_aware_matches_cache_only_within_idle_workers():
    policy = _local_policy(CacheAwareConfig())
    busy_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=100, dispatched_req_num=1)
    cache_idle_worker = _worker("10.0.0.2:8000")
    other_idle_worker = _worker("10.0.0.3:8000")
    prompt = "shared prefix " * 100
    _insert_prompt(policy, prompt, cache_idle_worker)

    selected_worker, _ = asyncio.run(policy.select_worker([other_idle_worker, busy_worker, cache_idle_worker], prompt))

    assert selected_worker is cache_idle_worker
