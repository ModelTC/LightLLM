import asyncio
import json
import socket
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI

from lightllm.server.config_server.cache_metadata import CacheMetadataRelay, relays, router
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.httpserver_for_pd_master.pd_selector.pd_selector import LoadBalancedCacheAwareSelector
from lightllm.server.pd_io_struct import PD_Client_Obj

pytestmark = pytest.mark.usefixtures("single_http_worker_start_args")


def _prefix_match(tree, prompt):
    return tree.prefix_match_sampled(prompt[:: tree.sample_stride], len(prompt), tree.sample_stride)


async def _estimated(selector, worker, prompt):
    return (await selector.policy.select_worker([worker], prompt))[1]


def _selector(master=1, workers=2, high_priority=False, model="/models/test"):
    args = StartArgs(
        config_server_host="127.0.0.1",
        config_server_port=60088,
        pd_node_id=master,
        model_dir=model,
        disable_pd_cache_high_priority=not high_priority,
    )
    selector = LoadBalancedCacheAwareSelector(SimpleNamespace(args=args))
    selector.update_nodes(
        [
            PD_Client_Obj(100 + i, f"10.0.0.{i + 1}:8000", "prefill", {}, connection_id=f"{master}-{i}")
            for i in range(workers)
        ],
        [PD_Client_Obj(200, "10.0.0.9:8000", "decode", {})],
    )
    return selector


def _connect(relay, state):
    peer, initial = relay.subscribe(state.owner, state.master_id, state.epoch, state.cursor)
    for payload in initial:
        state.apply(json.loads(payload))
    _drain(peer, state)
    return peer


def _drain(peer, state):
    while not peer.queue.empty():
        state.apply(json.loads(peer.queue.get_nowait()))


def _hint(key="abc", worker="10.0.0.1:8000/dpNone", node_id=100, seq=1, observed_at=None):
    return {
        "kind": "hint",
        "id": seq,
        "key": key,
        "worker": worker,
        "node_id": node_id,
        "observed_at": time.time() if observed_at is None else observed_at,
    }


def _load(selector):
    return {
        "kind": "load",
        "observed_at": time.time(),
        "workers": {
            worker.connection_key: [worker.node_id, worker.dispatched_req_num, worker.dispatched_prompt_chars]
            for worker in selector.prefill_nodes
        },
    }


def test_four_masters_share_success_hints_in_one_order_and_choose_same_p():
    relay = CacheMetadataRelay()
    selectors = [_selector(master=i) for i in range(4)]
    states = [selector.shared_cache for selector in selectors]
    peers = [_connect(relay, state) for state in states]
    for peer, selector in zip(peers, selectors):
        relay.publish(peer, _load(selector))
    for peer, state in zip(peers, states):
        _drain(peer, state)

    prompt = "shared prefix " * 100
    # Opposite worker histories from two publishers must be applied in relay order,
    # including each publisher's own echoed event.
    for index, target in ((0, 1), (1, 0)):
        selector = selectors[index]
        asyncio.run(selector.insert_prompt_cache(prompt, selector.prefill_nodes[target]))
        assert _prefix_match(selector.shared_cache.tree, prompt).prefill_node is None
        relay.publish(peers[index], states[index].outgoing.get_nowait())
    for peer, state, selector in zip(peers, states, selectors):
        _drain(peer, state)
        p, _, extra = asyncio.run(selector.select_p_d_node(prompt, None, None))
        assert p is selector.prefill_nodes[0]
        assert extra.estimated_cache_hit_rate > 0.5
        assert state.cursor == 2


@pytest.mark.parametrize("other_master", [1, 2])
def test_global_load_keeps_own_latest_counters_without_double_counting(other_master):
    relay = CacheMetadataRelay()
    a, b = _selector(1), _selector(other_master)
    pa, pb = _connect(relay, a.shared_cache), _connect(relay, b.shared_cache)
    b.prefill_nodes[0].dispatched_req_num = 10
    b.prefill_nodes[0].dispatched_prompt_chars = 100000
    relay.publish(pb, _load(b))
    relay.publish(pa, _load(a))
    _drain(pa, a.shared_cache)
    # Local counters advance immediately, without waiting for a snapshot echo.
    a.prefill_nodes[1].dispatched_req_num = 1
    a.prefill_nodes[1].dispatched_prompt_chars = 123
    loads = a.shared_cache.cluster_loads(a.prefill_nodes)
    assert loads[a.prefill_nodes[0].connection_key] == [10, 100000]
    assert loads[a.prefill_nodes[1].connection_key] == [1, 123]
    assert asyncio.run(a.select_p_d_node("uncached prompt", None, None))[0] is a.prefill_nodes[1]
    assert a.prefill_nodes[0].dispatched_req_num == 0
    assert a.prefill_nodes[0].dispatched_prompt_chars == 0


@pytest.mark.parametrize("problem", ["missing", "stale", "future", "different_epoch", "disconnected"])
def test_unknown_remote_load_falls_back_to_worker_reported_load(problem):
    selector = _selector()
    state = selector.shared_cache
    state.ready = True
    snapshot = _load(selector)
    snapshot["observed_at"] = time.time()
    if problem == "missing":
        snapshot["workers"].pop(selector.prefill_nodes[0].connection_key)
    elif problem == "stale":
        snapshot["observed_at"] -= 10
    elif problem == "future":
        snapshot["observed_at"] += 10
    elif problem == "different_epoch":
        snapshot["workers"][selector.prefill_nodes[0].connection_key][0] += 1
    elif problem == "disconnected":
        state.ready = False
    state.remote_loads["other"] = snapshot
    assert state.cluster_loads(selector.prefill_nodes) is None
    with patch.object(
        selector, "_importance_sampling", side_effect=[selector.prefill_nodes[1], selector.decode_nodes[0]]
    ):
        assert asyncio.run(selector.select_p_d_node("hello world", None, None))[0] is selector.prefill_nodes[1]


def test_metadata_disconnect_retains_load_until_master_registration_is_gone():
    relay = CacheMetadataRelay()
    a, b = _selector(1), _selector(2)
    pa, pb = _connect(relay, a.shared_cache), _connect(relay, b.shared_cache)
    relay.publish(pb, _load(b))
    relay.unsubscribe(pb)
    assert b.shared_cache.owner in relay._loads()
    relay.forget_master(2)
    _drain(pa, a.shared_cache)
    assert b.shared_cache.owner not in a.shared_cache.remote_loads
    assert a.shared_cache.cluster_loads(a.prefill_nodes) is not None


def test_old_registration_cleanup_keeps_live_metadata_owner():
    relay = CacheMetadataRelay()
    selector = _selector()
    _connect(relay, selector.shared_cache)
    relay.forget_master(selector.shared_cache.master_id)
    assert selector.shared_cache.owner in relay._loads()


def test_unknown_master_snapshot_is_not_treated_as_zero_load():
    relay = CacheMetadataRelay()
    selector = _selector()
    peer = _connect(relay, selector.shared_cache)
    relay.subscribe("other", 2, None, 0)
    _drain(peer, selector.shared_cache)
    assert selector.shared_cache.remote_loads["other"] is None
    assert selector.shared_cache.cluster_loads(selector.prefill_nodes) is None


def test_relay_does_not_refresh_a_delayed_load_snapshot():
    relay = CacheMetadataRelay()
    a, b = _selector(1), _selector(2)
    pa, pb = _connect(relay, a.shared_cache), _connect(relay, b.shared_cache)
    snapshot = _load(b)
    snapshot["observed_at"] -= 60
    relay.publish(pb, snapshot)
    _drain(pa, a.shared_cache)
    assert a.shared_cache.cluster_loads(a.prefill_nodes) is None


def test_worker_restart_at_same_address_invalidates_hint_but_connection_change_does_not():
    selector = _selector()
    state = selector.shared_cache
    peer = _connect(CacheMetadataRelay(), state)
    del peer
    prompt = "x" * 1025
    event = {**_hint(key="xxx"), "seq": 1}
    state.apply(event)
    worker = selector.prefill_nodes[0]
    worker.connection_id = "new-master-connection"
    assert asyncio.run(_estimated(selector, worker, prompt)).estimated_cache_hit_rate == 1
    worker.node_id += 1000
    assert asyncio.run(_estimated(selector, worker, prompt)).estimated_cache_hit_rate == 0
    assert (asyncio.run(selector.policy._get_cache_worker(selector.prefill_nodes, prompt)))[0] is None


def test_dp_ranks_are_distinct_cache_targets():
    selector = _selector()
    worker = selector.prefill_nodes[0]
    worker.dp_index = 3
    state = selector.shared_cache
    relay = CacheMetadataRelay()
    peer = _connect(relay, state)
    relay.publish(peer, _hint(key="xxx", worker=worker.connection_key))
    _drain(peer, state)
    assert asyncio.run(_estimated(selector, worker, "x" * 1025)).estimated_cache_hit_rate == 1
    worker.dp_index = 4
    assert asyncio.run(_estimated(selector, worker, "x" * 1025)).estimated_cache_hit_rate == 0


def test_reconnect_replays_missing_events_without_refreshing_old_hint_age():
    relay = CacheMetadataRelay()
    selector = _selector()
    state = selector.shared_cache
    peer = _connect(relay, state)
    relay.publish(peer, _hint(key="xxx", observed_at=time.time() - 120))
    _drain(peer, state)
    relay.unsubscribe(peer)
    publisher, _ = relay.subscribe("other", 2, None, 0)
    relay.publish(publisher, _hint(key="yyy", seq=2))
    _connect(relay, state)
    assert state.cursor == 2
    result = _prefix_match(selector.shared_cache.tree, "x" * 1025)
    assert result.prefill_node is not None
    assert time.monotonic() - result.last_insert_time >= 120
    assert _prefix_match(selector.shared_cache.tree, "y" * 1025).prefill_node is not None


def test_duplicate_hint_does_not_change_order_owner_age_or_lru():
    relay = CacheMetadataRelay()
    selector = _selector()
    state = selector.shared_cache
    peer = _connect(relay, state)
    hint = _hint(key="xxx")
    relay.publish(peer, hint)
    _drain(peer, state)
    tree = selector.shared_cache.tree
    mark = tree._time_mark
    before = _prefix_match(tree, "x" * 1025)
    relay.publish(peer, hint)
    assert peer.queue.empty()
    state.apply(json.loads(relay.history[0][2]))
    assert tree._time_mark == mark
    assert _prefix_match(tree, "x" * 1025) == before


@pytest.mark.parametrize("restart", [False, True])
def test_unrecoverable_gap_or_relay_restart_clears_old_tree_and_queued_hints(restart):
    relay = CacheMetadataRelay(max_log_bytes=500)
    selector = _selector()
    state = selector.shared_cache
    peer = _connect(relay, state)
    relay.publish(peer, _hint(key="xxx"))
    _drain(peer, state)
    relay.unsubscribe(peer)
    asyncio.run(selector.insert_prompt_cache("unsent" * 200, selector.prefill_nodes[0]))
    assert not state.outgoing.empty()
    if restart:
        relay = CacheMetadataRelay(max_log_bytes=500)
    publisher, _ = relay.subscribe("other", 2, None, 0)
    for seq in range(2, 8):
        relay.publish(publisher, _hint(key="yyy", seq=seq))
    _connect(relay, state)
    assert state.outgoing.empty()
    assert state.cursor == relay.sequence
    assert _prefix_match(selector.shared_cache.tree, "x" * 1025).prefill_node is None
    assert _prefix_match(selector.shared_cache.tree, "y" * 1025).prefill_node is not None


def test_future_timestamp_is_not_eligible_for_fresh_cache_priority():
    selector = _selector()
    state = selector.shared_cache
    _connect(CacheMetadataRelay(), state)
    state.apply({**_hint(key="xxx", observed_at=time.time() + 100), "seq": 1})
    result = asyncio.run(_estimated(selector, selector.prefill_nodes[0], "x" * 1025))
    assert result.estimated_cache_hit_rate == 1
    assert result.cache_last_insert_time is None


def test_single_p_with_priority_disabled_skips_tree_and_metadata_queue():
    selector = _selector(workers=1)
    assert not selector.shared_cache.enabled()
    selector.shared_cache.ready = True
    asyncio.run(selector.insert_prompt_cache("x" * 1025, selector.prefill_nodes[0]))
    assert selector.shared_cache.outgoing.empty()
    with patch.object(
        selector.shared_cache.tree, "prefix_match_sampled", side_effect=AssertionError("unneeded lookup")
    ):
        assert asyncio.run(selector.select_p_d_node("x" * 1025, None, None))[0] is selector.prefill_nodes[0]
    assert _selector(workers=1, high_priority=True).shared_cache.enabled()


def test_bounded_queues_do_not_block_publishers_and_slow_subscriber_can_replay():
    relay = CacheMetadataRelay()
    selector = _selector()
    state = selector.shared_cache
    slow, _ = relay.subscribe("slow", 2, None, 0)
    publisher = _connect(relay, state)
    for seq in range(300):
        relay.publish(publisher, _hint(seq=seq))
    assert slow.overflow.is_set()
    assert slow.queue.qsize() == slow.queue.maxsize
    selector.shared_cache.ready = True
    for _ in range(300):
        asyncio.run(selector.insert_prompt_cache("xx", selector.prefill_nodes[0]))
    assert state.outgoing.qsize() == 256
    assert state.dropped_hints == 44
    replacement, initial = relay.subscribe("slow", 2, relay.epoch, 0)
    assert sum(json.loads(payload)["kind"] == "hint" for payload in initial) == 300
    relay.unsubscribe(slow)
    assert relay.subscribers["slow"] is replacement


def test_namespaces_separate_models_and_sampling_schema():
    a, b = _selector(model="/models/a"), _selector(model="/models/b")
    assert a.shared_cache.namespace != b.shared_cache.namespace


def test_real_websocket_clients_share_hints_and_cancel_cleanly():
    async def run():
        import uvicorn

        app = FastAPI()
        app.include_router(router)
        app.state.registered_pd_master_objs = {i: object() for i in range(1, 5)}
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        server_task = asyncio.create_task(server.serve(sockets=[sock]))
        clients = []
        try:

            async def wait_for(predicate):
                async def poll():
                    while not predicate():
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(poll(), 10)

            await wait_for(lambda: server.started)
            selectors = [_selector(i) for i in range(1, 5)]
            a, b = selectors[:2]
            for selector in selectors:
                selector.shared_cache.uri = f"ws://127.0.0.1:{sock.getsockname()[1]}/pd_cache_metadata"
                clients.append(asyncio.create_task(selector.shared_cache.run()))
            await wait_for(lambda: all(selector.shared_cache.ready for selector in selectors))
            prompt = "shared over actual websocket " * 100
            (await a.insert_prompt_cache(prompt, a.prefill_nodes[1]))
            await wait_for(lambda: all(selector.shared_cache.cursor == 1 for selector in selectors))
            await wait_for(lambda: b.shared_cache.cluster_loads(b.prefill_nodes) is not None)
            assert (await b.select_p_d_node(prompt, None, None))[0] is b.prefill_nodes[1]
            assert a.shared_cache.cursor == b.shared_cache.cursor == 1
            # Disconnect only B's metadata socket; A keeps publishing and B catches up.
            relay = relays[a.shared_cache.namespace]
            old_peer = relay.subscribers[b.shared_cache.owner]
            old_peer.overflow.set()
            await wait_for(lambda: not b.shared_cache.ready)
            second_prompt = "new prefix after disconnect " * 100
            (await a.insert_prompt_cache(second_prompt, a.prefill_nodes[0]))
            await wait_for(lambda: a.shared_cache.cursor == 2)
            await wait_for(lambda: b.shared_cache.ready and b.shared_cache.cursor == 2)
            assert relay.subscribers[b.shared_cache.owner] is not old_peer
            assert (await b.select_p_d_node(second_prompt, None, None))[0] is b.prefill_nodes[0]
            # Control characters expand during JSON encoding; dropping the hint
            # must not repeatedly disconnect the metadata stream.
            a.shared_cache.outgoing.put_nowait(_hint(key="\0" * 16384, seq=99))
            await wait_for(lambda: a.shared_cache.dropped_hints == 1)
            assert relay.sequence == 2
            assert all(selector.shared_cache.ready for selector in selectors)
        finally:
            for client in clients:
                client.cancel()
            await asyncio.gather(*clients, return_exceptions=True)
            server.should_exit = True
            await asyncio.wait_for(server_task, 10)
            relays.clear()

    asyncio.run(run())
