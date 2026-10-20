"""Local PD policies use shared cache/load snapshots without remote selection."""

import asyncio
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from lightllm.server.core.objs import FinishStatus, SamplingParams
from lightllm.server.httpserver_for_pd_master.manager import (
    HttpServerManagerForPDMaster,
    PDManager,
)
from lightllm.server.httpserver_for_pd_master.pd_selector import pd_node_stats_client
from lightllm.server.httpserver_for_pd_master.pd_selector import pd_selector
from lightllm.server.httpserver_for_pd_master.pd_selector import prompt_cache_tree_client
from lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree_client import (
    LocalPromptCacheTreeClient,
    RemotePromptCacheTreeClient,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.pd_selector import (
    AdaptiveLoadSelector,
    LoadBalancedCacheAwareSelector,
    PDSelectionExtraInfo,
    RandomSelector,
    RoundRobinSelector,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree import (
    PromptCacheMatchResult,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.pd_node_stats_server import (
    PDNodeStatsState,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree_server import (
    PromptCacheTreeState,
)
from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.utils.error_utils import ServerBusyError


PROMPT = "x" * 2048
COUNTER_METHODS = ("add_dispatched_info", "sub_dispatched_info")


@pytest.fixture(autouse=True)
def local_pd_master_mode(monkeypatch):
    start_args = SimpleNamespace(httpserver_workers=1)
    monkeypatch.setattr(pd_node_stats_client, "get_env_start_args", lambda: start_args)
    monkeypatch.setattr(
        prompt_cache_tree_client, "get_env_start_args", lambda: start_args
    )


def _node(
    node_id,
    address,
    *,
    mode="prefill",
    chars=0,
    requests=0,
    load=0.0,
    connection_id=None,
):
    node = PD_Client_Obj(
        node_id=node_id,
        client_ip_port=address,
        mode=mode,
        start_args={},
        connection_id=connection_id or f"connection-{node_id}",
        dispatched_prompt_chars=chars,
        dispatched_req_num=requests,
    )
    node.run_status.total_token_usage_rate = load
    return node


def _match(node=None, *, matched=1500):
    return PromptCacheMatchResult(
        prefill_node=None if node is None else node.connection_key,
        matched_char_count=0 if node is None else matched,
        input_char_count=len(PROMPT),
        last_insert_time=None if node is None else 123.0,
    )


def _assert_backend_client(selector):
    assert isinstance(
        selector.policy.prompt_cache_tree,
        (LocalPromptCacheTreeClient, RemotePromptCacheTreeClient),
    )


def _set_nodes(manager, prefill_nodes, decode_nodes):
    manager.pd_manager.prefill_nodes = prefill_nodes
    manager.pd_manager.decode_nodes = decode_nodes
    manager.pd_manager.url_to_pd_nodes = {
        node.connection_key: node for node in [*prefill_nodes, *decode_nodes]
    }
    manager.pd_manager.selector.update_nodes(prefill_nodes, decode_nodes)


def _manager(
    strategy="cache_aware", *, prefill_nodes=None, decode_nodes=None, mock_prefix=True
):
    manager = HttpServerManagerForPDMaster.__new__(HttpServerManagerForPDMaster)
    manager.args = SimpleNamespace(
        select_p_d_node_strategy=strategy, pd_master_mode="elastic"
    )
    manager.pd_manager = PDManager(manager.args)
    _set_nodes(
        manager,
        prefill_nodes
        if prefill_nodes is not None
        else [_node(100, "p1:8000"), _node(101, "p2:8000")],
        decode_nodes
        if decode_nodes is not None
        else [_node(200, "d:8000", mode="decode")],
    )
    if mock_prefix:
        selector = manager.pd_manager.selector
        selector.pd_node_stats.update_dispatched_info = AsyncMock(
            wraps=selector.pd_node_stats.update_dispatched_info
        )
        if strategy == "cache_aware":
            selector.policy.prompt_cache_tree.prefix_match_sampled = AsyncMock(
                return_value=_match()
            )
    manager.pd_node_stats = manager.pd_manager.selector.pd_node_stats
    return manager


@pytest.mark.parametrize("busy", [False, True])
def test_shared_tree_is_used_for_selection_and_cache_metadata(monkeypatch, busy):
    selector = LoadBalancedCacheAwareSelector(None)
    cached = _node(100, "p2:8000", chars=110, requests=int(busy))
    other = _node(101, "p1:8000", chars=100, requests=int(busy))
    decode = _node(200, "d:8000", mode="decode")
    selector.update_nodes([other, cached], [decode])
    _assert_backend_client(selector)
    selector.policy.prompt_cache_tree.prefix_match_sampled = AsyncMock(
        return_value=_match(cached)
    )

    p_node, d_node, extra = asyncio.run(selector.select_p_d_node(PROMPT, None, None))

    assert p_node is cached
    assert d_node is decode
    assert extra == PDSelectionExtraInfo(1500 / len(PROMPT), 123.0)


def test_cache_metadata_comes_from_the_selection_query(monkeypatch):
    selector = LoadBalancedCacheAwareSelector(None)
    cached, other = _node(100, "p2:8000"), _node(101, "p1:8000")
    selector.update_nodes([other, cached], [_node(200, "d:8000", mode="decode")])
    _assert_backend_client(selector)
    selector.policy.prompt_cache_tree.prefix_match_sampled = AsyncMock(
        return_value=_match(cached)
    )

    selected, _, extra = asyncio.run(selector.select_p_d_node(PROMPT, None, None))

    assert selected is cached
    assert extra == PDSelectionExtraInfo(1500 / len(PROMPT), 123.0)
    selector.policy.prompt_cache_tree.prefix_match_sampled.assert_awaited_once_with(
        PROMPT[:: selector.policy.config.sample_stride],
        len(PROMPT),
        selector.policy.config.sample_stride,
    )


@pytest.mark.parametrize("restarted", [False, True])
def test_cache_hint_survives_endpoint_reconnect_or_process_restart(
    monkeypatch, restarted
):
    original = _node(100, "p2:8000", connection_id="old-socket")
    current = _node(300 if restarted else 100, "p2:8000", connection_id="new-socket")
    other = _node(101, "p1:8000")
    selector = LoadBalancedCacheAwareSelector(None)
    selector.update_nodes([other, current], [_node(200, "d:8000", mode="decode")])
    _assert_backend_client(selector)
    selector.policy.prompt_cache_tree.prefix_match_sampled = AsyncMock(
        return_value=_match(original)
    )

    selected, _, extra = asyncio.run(selector.select_p_d_node(PROMPT, None, None))

    assert original.connection_key == current.connection_key
    assert selected is current
    assert extra == PDSelectionExtraInfo(1500 / len(PROMPT), 123.0)


@pytest.mark.parametrize("match_kind", ["miss", "remote_worker"])
def test_shared_match_never_falls_back_to_local_tree_or_remote_candidates(
    monkeypatch, match_kind
):
    busy = _node(100, "p1:8000", chars=2000, requests=1)
    available = _node(101, "p2:8000", chars=100, requests=1)
    selector = LoadBalancedCacheAwareSelector(None)
    selector.update_nodes([busy, available], [_node(200, "d:8000", mode="decode")])
    asyncio.run(
        selector.policy.prompt_cache_tree.insert_sampled(
            PROMPT[:: selector.policy.config.sample_stride],
            busy.connection_key,
            selector.policy.config.sample_stride,
        )
    )
    _assert_backend_client(selector)
    shared_match = (
        _match() if match_kind == "miss" else _match(_node(999, "remote:8000"))
    )
    selector.policy.prompt_cache_tree.prefix_match_sampled = AsyncMock(
        return_value=shared_match
    )

    selected, _, extra = asyncio.run(selector.select_p_d_node(PROMPT, None, None))

    assert selected is available
    assert extra == PDSelectionExtraInfo()


def test_default_cache_selection_uses_connection_keys():
    selector = LoadBalancedCacheAwareSelector(None)
    cached, other = _node(100, "p2:8000"), _node(101, "p1:8000")
    selector.update_nodes([other, cached], [_node(200, "d:8000", mode="decode")])
    asyncio.run(
        selector.policy.prompt_cache_tree.insert_sampled(
            PROMPT[:: selector.policy.config.sample_stride],
            cached.connection_key,
            selector.policy.config.sample_stride,
        )
    )

    selected, _, extra = asyncio.run(selector.select_p_d_node(PROMPT, None, None))

    assert selected is cached
    assert extra.estimated_cache_hit_rate > 0.5
    assert (
        asyncio.run(
            selector.policy.prompt_cache_tree.prefix_match_sampled(
                PROMPT[:: selector.policy.config.sample_stride],
                len(PROMPT),
                selector.policy.config.sample_stride,
            )
        ).prefill_node
        == cached.connection_key
    )


def test_round_robin_progress_is_local_to_each_worker():
    nodes = [_node(100, "p1:8000"), _node(101, "p2:8000")]
    decode = [
        _node(200, "d1:8000", mode="decode"),
        _node(201, "d2:8000", mode="decode"),
    ]
    first, second = RoundRobinSelector(None), RoundRobinSelector(None)
    for selector in (first, second):
        selector.update_nodes(nodes, decode)

    async def run():
        assert (await first.select_p_d_node(PROMPT, None, None))[:2] == (
            nodes[0],
            decode[0],
        )
        assert (await first.select_p_d_node(PROMPT, None, None))[:2] == (
            nodes[1],
            decode[1],
        )
        assert (await second.select_p_d_node(PROMPT, None, None))[:2] == (
            nodes[0],
            decode[0],
        )

    asyncio.run(run())


@pytest.mark.parametrize("selector_type", [RandomSelector, AdaptiveLoadSelector])
def test_random_and_adaptive_keep_their_local_load_preferences(
    monkeypatch, selector_type
):
    monkeypatch.setattr(pd_selector, "random", random.Random(23))
    busy, available = _node(100, "p1:8000", load=1.0), _node(101, "p2:8000", load=0.0)
    selector = selector_type(None)
    selector.update_nodes([busy, available], [_node(200, "d:8000", mode="decode")])

    async def select_many():
        return [
            (await selector.select_p_d_node(PROMPT, None, None))[0] for _ in range(100)
        ]

    selections = asyncio.run(select_many())

    assert all(node is busy or node is available for node in selections)
    available_count = sum(node is available for node in selections)
    if selector_type is AdaptiveLoadSelector:
        assert available_count > 90
    else:
        assert 20 < available_count < 80


@pytest.mark.parametrize(
    "strategy", ["random", "round_robin", "adaptive_load", "cache_aware"]
)
def test_only_cache_aware_policy_queries_prefix_data_and_all_policies_choose_local_nodes(
    monkeypatch, strategy
):
    manager = _manager(strategy)
    local_nodes = manager.pd_manager.prefill_nodes
    if strategy == "cache_aware":
        _assert_backend_client(manager.pd_manager.selector)
        manager.pd_manager.selector.policy.prompt_cache_tree.prefix_match_sampled.return_value = _match(
            local_nodes[-1]
        )

    selected, decode, _ = asyncio.run(manager.select_p_d_node(PROMPT, None, None))

    assert any(selected is node for node in local_nodes)
    assert decode is manager.pd_manager.decode_nodes[0]
    if strategy == "cache_aware":
        assert manager.pd_manager.selector.policy.prompt_cache_tree.prefix_match_sampled.await_count == 1
        manager.pd_manager.selector.policy.prompt_cache_tree.prefix_match_sampled.assert_awaited_with(
            PROMPT[:: manager.pd_manager.selector.policy.config.sample_stride],
            len(PROMPT),
            manager.pd_manager.selector.policy.config.sample_stride,
        )
        manager.pd_manager.selector.pd_node_stats.update_dispatched_info.assert_awaited_once_with(
            p_nodes=local_nodes
        )
    else:
        manager.pd_manager.selector.pd_node_stats.update_dispatched_info.assert_not_awaited()


def test_manager_uses_shared_node_load_and_local_balancing_threshold_for_selection(
    monkeypatch,
):
    cached, available = _node(100, "p1:8000"), _node(101, "p2:8000")
    manager = _manager(prefill_nodes=[cached, available])
    _assert_backend_client(manager.pd_manager.selector)

    async def update_dispatched_info(p_nodes):
        assert p_nodes == [cached, available]
        cached.dispatched_prompt_chars, cached.dispatched_req_num = 500, 1
        available.dispatched_prompt_chars, available.dispatched_req_num = 100, 1

    manager.pd_manager.selector.pd_node_stats.update_dispatched_info.side_effect = (
        update_dispatched_info
    )
    manager.pd_manager.selector.policy.prompt_cache_tree.prefix_match_sampled.return_value = _match(cached)

    manager.pd_manager.selector.policy.config.balance_rel_threshold = 1.1
    first = asyncio.run(manager.select_p_d_node(PROMPT, None, None))
    manager.pd_manager.selector.policy.config.balance_rel_threshold = 2.0
    second = asyncio.run(manager.select_p_d_node(PROMPT, None, None))

    assert first[0] is available
    assert first[2] == PDSelectionExtraInfo()
    assert second[0] is cached
    assert second[2].estimated_cache_hit_rate == 1500 / len(PROMPT)


@pytest.mark.parametrize("replacement_load", [(0, 0), (999, 9)])
def test_query_completion_uses_current_connections_without_overwriting_their_load(
    monkeypatch, replacement_load
):
    async def run():
        old_p, old_d = _node(100, "p:8000"), _node(200, "d:8000", mode="decode")
        new_p = _node(
            300, "p:8000", chars=replacement_load[0], requests=replacement_load[1]
        )
        new_d = _node(400, "d:8000", mode="decode")
        manager = _manager(prefill_nodes=[old_p], decode_nodes=[old_d])
        _assert_backend_client(manager.pd_manager.selector)
        started, complete = asyncio.Event(), asyncio.Event()

        async def update_dispatched_info(p_nodes):
            assert len(p_nodes) == 1
            if not started.is_set():
                assert p_nodes[0] is old_p
                started.set()
                await complete.wait()
                old_p.dispatched_prompt_chars, old_p.dispatched_req_num = 800, 2

        manager.pd_manager.selector.pd_node_stats.update_dispatched_info.side_effect = (
            update_dispatched_info
        )
        manager.pd_manager.selector.policy.prompt_cache_tree.prefix_match_sampled.return_value = _match(old_p)
        task = asyncio.create_task(manager.select_p_d_node(PROMPT, None, None))
        await asyncio.wait_for(started.wait(), timeout=2)
        _set_nodes(manager, [new_p], [new_d])
        complete.set()
        selected, decode, extra = await task

        assert selected is new_p and decode is new_d
        assert (
            new_p.dispatched_prompt_chars,
            new_p.dispatched_req_num,
        ) == replacement_load
        assert extra == PDSelectionExtraInfo(1500 / len(PROMPT), 123.0)
        assert manager.pd_manager.selector.policy.prompt_cache_tree.prefix_match_sampled.await_count == 1
        manager.pd_manager.selector.policy.prompt_cache_tree.prefix_match_sampled.assert_awaited_with(
            PROMPT[:: manager.pd_manager.selector.policy.config.sample_stride],
            len(PROMPT),
            manager.pd_manager.selector.policy.config.sample_stride,
        )

    asyncio.run(run())


@pytest.mark.parametrize("replaced", ["prefill", "decode"])
def test_selection_query_uses_current_decode_and_rejects_a_replaced_prefill(replaced):
    async def run():
        manager = _manager()
        selector = manager.pd_manager.selector
        prefill, decode = (
            manager.pd_manager.prefill_nodes,
            manager.pd_manager.decode_nodes,
        )
        selected_prefill = prefill[0]
        replacement = (
            _node(300, "p1:8000")
            if replaced == "prefill"
            else _node(400, "d:8000", mode="decode")
        )
        started, complete = asyncio.Event(), asyncio.Event()

        async def query(*_args):
            started.set()
            await complete.wait()
            return _match(selected_prefill)

        selector.policy.prompt_cache_tree.prefix_match_sampled.side_effect = query
        task = asyncio.create_task(manager.select_p_d_node(PROMPT, None, None))
        await asyncio.wait_for(started.wait(), timeout=2)
        _set_nodes(
            manager,
            [replacement, *prefill[1:]] if replaced == "prefill" else prefill,
            [replacement] if replaced == "decode" else decode,
        )
        complete.set()
        if replaced == "prefill":
            with pytest.raises(ServerBusyError, match="connection changed"):
                await task
        else:
            selected, current_decode, _ = await task
            assert selected is selected_prefill
            assert current_decode is replacement

    asyncio.run(run())


@pytest.fixture(params=["local", "rpyc"])
def backend(request, monkeypatch):
    start_args = SimpleNamespace(
        httpserver_workers=2 if request.param == "rpyc" else 1
    )
    monkeypatch.setattr(pd_node_stats_client, "get_env_start_args", lambda: start_args)
    monkeypatch.setattr(
        prompt_cache_tree_client, "get_env_start_args", lambda: start_args
    )
    monkeypatch.setattr(
        pd_node_stats_client,
        "get_pd_master_unix_socket_address",
        lambda service_name: f"\0lightllm_test-pd-info_{service_name}".encode(),
    )
    monkeypatch.setattr(
        prompt_cache_tree_client,
        "get_pd_master_unix_socket_address",
        lambda service_name: f"\0lightllm_test-pd-info_{service_name}".encode(),
    )
    if request.param == "rpyc":
        monkeypatch.setattr(pd_node_stats_client, "unix_connect", Mock())
        monkeypatch.setattr(prompt_cache_tree_client, "unix_connect", Mock())
    return request.param


def _generator_manager(backend, strategy="round_robin"):
    manager = _manager(
        strategy,
        prefill_nodes=[_node(100, "p:8000")],
        mock_prefix=False,
    )
    states = SimpleNamespace(stats=None, tree=None)
    if backend != "local":
        states.stats = PDNodeStatsState()
        stats = manager.pd_manager.selector.pd_node_stats
        stats._get_dispatched_info = states.stats.get_dispatched_info
        stats._add_dispatched_info = states.stats.add_dispatched_info
        stats._sub_dispatched_info = states.stats.sub_dispatched_info
        if strategy == "cache_aware":
            states.tree = PromptCacheTreeState()
            tree = manager.pd_manager.selector.policy.prompt_cache_tree
            tree._prefix_match_sampled = states.tree.prefix_match_sampled
            tree._insert_sampled = states.tree.insert_sampled
    for name in COUNTER_METHODS:
        stats = manager.pd_manager.selector.pd_node_stats
        setattr(stats, name, AsyncMock(wraps=getattr(stats, name)))
    manager.pd_manager.selector.record_prompt_cache_hit_rate = Mock(
        wraps=manager.pd_manager.selector.record_prompt_cache_hit_rate
    )
    if strategy == "cache_aware":
        tree = manager.pd_manager.selector.policy.prompt_cache_tree
        tree.insert_sampled = AsyncMock(wraps=tree.insert_sampled)
    manager.active_req_ids = set()
    manager.disable_pd_cache_high_priority = True
    manager.enable_pd_node_self_request_limit = False
    manager.remove_req = AsyncMock()
    manager.abort = AsyncMock()
    ids = iter(range(8, 80, 8))
    manager.id_gen = SimpleNamespace(generate_id=lambda: next(ids))
    return manager, states


def _pending(state, node):
    if state.stats is None:
        return node.dispatched_prompt_chars, node.dispatched_req_num
    return state.stats.get_dispatched_info([node.connection_key])[
        node.connection_key
    ]


def test_backend_tree_queries_preserve_cache_hint_for_restarted_endpoint(
    monkeypatch, backend
):
    async def run():
        manager, state = _generator_manager(backend, "cache_aware")
        cached, other = _node(100, "p2:8000"), _node(101, "p1:8000")
        decode = manager.pd_manager.decode_nodes
        _set_nodes(manager, [other, cached], decode)
        selector = manager.pd_manager.selector
        await selector.policy.prompt_cache_tree.insert_sampled(
            PROMPT[:: selector.policy.config.sample_stride],
            cached.connection_key,
            selector.policy.config.sample_stride,
        )
        injected_tree = selector.policy.prompt_cache_tree
        tree = (
            selector.policy.prompt_cache_tree.prompt_cache_tree
            if state.tree is None
            else state.tree.prompt_cache_tree
        )
        prefix_match = Mock(wraps=tree.prefix_match_sampled)
        monkeypatch.setattr(tree, "prefix_match_sampled", prefix_match)
        assert isinstance(
            injected_tree,
            (LocalPromptCacheTreeClient, RemotePromptCacheTreeClient),
        )

        selected, _, extra = await manager.select_p_d_node(PROMPT, None, None)

        assert selected is cached
        assert extra.estimated_cache_hit_rate > 0.5
        assert extra.cache_last_insert_time is not None
        original_extra = extra
        assert prefix_match.call_count == 1

        restarted = _node(300, cached.client_ip_port)
        _set_nodes(manager, [other, restarted], decode)
        selected, _, extra = await manager.select_p_d_node(PROMPT, None, None)
        assert selected is restarted
        assert extra == original_extra
        assert prefix_match.call_count == 2
        assert selector.policy.prompt_cache_tree is injected_tree

    asyncio.run(run())


def test_cache_feedback_and_dynamic_thresholds_stay_private_to_each_worker(backend):
    first, state = _generator_manager(backend, "cache_aware")
    second, _ = _generator_manager(backend, "cache_aware")
    stats_calls = tree_calls = None
    if state.stats is not None:
        stats_calls = Mock(wraps=state.stats.get_dispatched_info)
        tree_calls = Mock(wraps=state.tree.prefix_match_sampled)
        for manager in (first, second):
            selector = manager.pd_manager.selector
            selector.pd_node_stats._get_dispatched_info = stats_calls
            selector.policy.prompt_cache_tree._prefix_match_sampled = tree_calls

    for manager, rates in (
        (first, (0.2, 0.2, 0.8, 0.8)),
        (second, (0.8, 0.8, 0.2, 0.2)),
    ):
        selector = manager.pd_manager.selector
        selector.policy.config.cache_hit_rate_window_size = 2
        for rate in rates:
            assert selector.record_prompt_cache_hit_rate(rate) is None

    first_policy, second_policy = (
        first.pd_manager.selector.policy,
        second.pd_manager.selector.policy,
    )
    assert first_policy.config.balance_rel_threshold == pytest.approx(1.55)
    assert second_policy.config.balance_rel_threshold == pytest.approx(1.45)
    assert (
        first_policy.balance_rel_threshold_controller
        is not second_policy.balance_rel_threshold_controller
    )
    if stats_calls is not None:
        stats_calls.assert_not_called()
        tree_calls.assert_not_called()

    async def query_both_workers():
        for manager in (first, second):
            await manager.select_p_d_node(PROMPT, None, None)

    asyncio.run(query_both_workers())

    assert first_policy.config.balance_rel_threshold == pytest.approx(1.55)
    assert second_policy.config.balance_rel_threshold == pytest.approx(1.45)
    if stats_calls is not None:
        assert stats_calls.call_count == 2
        assert tree_calls.call_count == 2


def _params(max_new_tokens=3):
    params = SamplingParams()
    params.init(None, max_new_tokens=max_new_tokens)
    return params


async def _consume(manager, params):
    return [
        item
        async for item in manager._generate_one_attempt(
            "prompt", params, None, None, 0, 80, 2
        )
    ]


@pytest.mark.parametrize(
    "strategy", ["random", "round_robin", "adaptive_load", "cache_aware"]
)
@pytest.mark.parametrize("needs_prefill", [True, False])
def test_segment_counters_clear_at_first_token_and_restart_for_continuation(
    backend, strategy, needs_prefill
):
    manager, state = _generator_manager(backend, strategy)
    prefill = manager.pd_manager.prefill_nodes[0]
    segments = []

    async def generate_segment(
        p_node, d_node, start_time, prompt, params, multimodal, request
    ):
        segments.append(prompt)
        assert _pending(state, p_node) == (len(prompt), 1)
        yield (
            params.group_request_id,
            "x",
            {
                "prompt_tokens": 100,
                "prompt_cache_len": 20,
                "needs_prefill_first_token": needs_prefill,
            },
            FinishStatus(),
        )
        assert _pending(state, p_node) == (0, 0)
        finish = (
            FinishStatus.FINISHED_PD_DECODE_CAPACITY
            if len(segments) == 1
            else FinishStatus.FINISHED_STOP
        )
        yield params.group_request_id, "", {"prompt_tokens": 100}, FinishStatus(finish)

    manager._wait_to_token_package = generate_segment
    results = asyncio.run(_consume(manager, _params()))

    assert segments == ["prompt", "promptx"]
    assert len(results) == 3
    assert _pending(state, prefill) == (0, 0)
    assert not manager.active_req_ids
    stats = manager.pd_manager.selector.pd_node_stats
    assert stats.add_dispatched_info.await_args_list == [
        call(p_node=prefill, chars=len("prompt"), req_num=1),
        call(p_node=prefill, chars=len("promptx"), req_num=1),
    ]
    assert stats.sub_dispatched_info.await_args_list == [
        call(p_node=prefill, chars=len("prompt"), req_num=1),
        call(p_node=prefill, chars=len("promptx"), req_num=1),
    ]
    selector = manager.pd_manager.selector
    selector.record_prompt_cache_hit_rate.assert_called_once_with(0.2)
    if strategy == "cache_aware":
        assert selector.policy.balance_rel_threshold_controller._cache_hit_rates == [
            0.2
        ]
        if needs_prefill:
            sample_stride = manager.pd_manager.selector.policy.config.sample_stride
            selector.policy.prompt_cache_tree.insert_sampled.assert_awaited_once_with(
                "prompt"[::sample_stride], prefill.connection_key, sample_stride
            )
        else:
            selector.policy.prompt_cache_tree.insert_sampled.assert_not_awaited()


def test_error_output_records_hit_rate_without_inserting_cache(backend):
    manager, state = _generator_manager(backend, "cache_aware")
    prefill = manager.pd_manager.prefill_nodes[0]

    async def generate_segment(
        p_node, d_node, start_time, prompt, params, multimodal, request
    ):
        yield (
            params.group_request_id,
            "",
            {
                "prompt_tokens": 100,
                "prompt_cache_len": 20,
                "needs_prefill_first_token": True,
            },
            FinishStatus(FinishStatus.FINISHED_ERROR),
        )

    manager._wait_to_token_package = generate_segment
    results = asyncio.run(_consume(manager, _params()))

    assert len(results) == 1 and results[0][3].is_error_finished()
    assert _pending(state, prefill) == (0, 0)
    selector = manager.pd_manager.selector
    selector.record_prompt_cache_hit_rate.assert_called_once_with(0.2)
    assert selector.policy.balance_rel_threshold_controller._cache_hit_rates == [0.2]
    selector.policy.prompt_cache_tree.insert_sampled.assert_not_awaited()


def test_error_before_first_token_clears_segment_counters(backend):
    manager, state = _generator_manager(backend)
    prefill = manager.pd_manager.prefill_nodes[0]

    async def generate_segment(*_args):
        assert _pending(state, prefill) == (len("prompt"), 1)
        raise RuntimeError("before first token")
        yield

    manager._wait_to_token_package = generate_segment
    with pytest.raises(RuntimeError, match="before first token"):
        asyncio.run(_consume(manager, _params()))

    assert _pending(state, prefill) == (0, 0)
    assert not manager.active_req_ids
    manager.abort.assert_awaited_once()
    manager.remove_req.assert_awaited_once()
    manager.pd_node_stats.sub_dispatched_info.assert_awaited_once_with(
        p_node=prefill, chars=len("prompt"), req_num=1
    )


@pytest.mark.parametrize("failure", ["before_segment", "zero_output_budget"])
def test_no_segment_means_no_counter_updates(monkeypatch, backend, failure):
    manager, _ = _generator_manager(backend)
    manager._wait_to_token_package = Mock()
    params = _params()
    if failure == "zero_output_budget":
        params.max_new_tokens = 0
    else:

        def fail_copy(*_args, **_kwargs):
            raise RuntimeError("segment initialization failed")

        monkeypatch.setattr(SamplingParams, "from_buffer_copy", fail_copy)

    if failure == "before_segment":
        with pytest.raises(RuntimeError, match="segment initialization"):
            asyncio.run(_consume(manager, params))
    else:
        assert asyncio.run(_consume(manager, params)) == []
    for name in COUNTER_METHODS:
        getattr(manager.pd_node_stats, name).assert_not_awaited()
    manager._wait_to_token_package.assert_not_called()
    manager.remove_req.assert_awaited_once()
    assert not manager.active_req_ids


@pytest.mark.parametrize("replaced", ["prefill", "decode"])
def test_connection_closed_during_counter_update_fails_at_dispatch_and_clears_load(
    backend, replaced
):
    async def run():
        manager, state = _generator_manager(backend)
        prefill, decode = (
            manager.pd_manager.prefill_nodes[0],
            manager.pd_manager.decode_nodes[0],
        )
        created, complete = asyncio.Event(), asyncio.Event()
        original_add_info = manager.pd_node_stats.add_dispatched_info

        async def add_info(**kwargs):
            await original_add_info(**kwargs)
            created.set()
            await complete.wait()

        async def generate_segment(p_node, d_node, *_args):
            assert p_node is prefill and d_node is decode
            raise ConnectionError("selected PD connection closed")
            yield

        manager.pd_node_stats.add_dispatched_info = add_info
        manager._wait_to_token_package = Mock(side_effect=generate_segment)
        task = asyncio.create_task(_consume(manager, _params()))
        await asyncio.wait_for(created.wait(), timeout=2)
        previous = prefill if replaced == "prefill" else decode
        replacement = _node(
            previous.node_id,
            previous.client_ip_port,
            mode=previous.mode,
            chars=23,
            requests=2,
        )
        _set_nodes(
            manager,
            [replacement] if replaced == "prefill" else [prefill],
            [replacement] if replaced == "decode" else [decode],
        )
        complete.set()
        with pytest.raises(ConnectionError, match="selected PD connection closed"):
            await task

        manager._wait_to_token_package.assert_called_once()
        manager.abort.assert_awaited_once()
        manager.remove_req.assert_awaited_once()
        assert _pending(state, prefill) == (0, 0)
        assert (
            replacement.dispatched_prompt_chars,
            replacement.dispatched_req_num,
        ) == (23, 2)
        assert not manager.active_req_ids

    asyncio.run(run())


def test_cancelled_request_clears_counters_before_removing_request(backend):
    manager, state = _generator_manager(backend)
    prefill = manager.pd_manager.prefill_nodes[0]

    async def run():
        started = asyncio.Event()

        async def remove_req(*_args):
            assert _pending(state, prefill) == (0, 0)

        manager.remove_req.side_effect = remove_req

        async def generate_segment(*_args):
            started.set()
            await asyncio.Event().wait()
            yield

        manager._wait_to_token_package = generate_segment
        task = asyncio.create_task(_consume(manager, _params()))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            assert _pending(state, prefill) == (len("prompt"), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    assert _pending(state, prefill) == (0, 0)
    assert not manager.active_req_ids
    manager.remove_req.assert_awaited_once()
