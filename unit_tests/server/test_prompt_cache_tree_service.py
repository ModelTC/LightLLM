"""Prompt cache tree local and RPyC component behavior."""

import asyncio
import socket
import threading
import uuid
from unittest.mock import Mock, patch

import pytest
from rpyc.utils.server import ThreadedServer

from lightllm.server.httpserver_for_pd_master.pd_selector import prompt_cache_tree_client
from lightllm.server.httpserver_for_pd_master.pd_selector.pd_node_stats_server import PDNodeStatsState
from lightllm.server.httpserver_for_pd_master.pd_selector.cache_aware import (
    CacheAwareConfig,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree import (
    PromptCacheMatchResult,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree_client import (
    LocalPromptCacheTreeClient,
    RemotePromptCacheTreeClient,
    create_prompt_cache_tree_client,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree_server import PromptCacheTreeState
from lightllm.utils.net_utils import get_abstract_unix_socket_address

SAMPLE_STRIDE = CacheAwareConfig().sample_stride


def _key(prompt):
    return prompt[::SAMPLE_STRIDE]


def _remote_tree(ipc_name):
    with patch.object(
        prompt_cache_tree_client,
        "get_pd_master_unix_socket_address",
        return_value=get_abstract_unix_socket_address("prompt_cache_tree", ipc_name),
    ):
        return RemotePromptCacheTreeClient()


def test_local_client_owns_an_independent_tree(monkeypatch):
    monkeypatch.setattr(
        prompt_cache_tree_client,
        "get_env_start_args",
        lambda: Mock(httpserver_workers=1),
    )
    config = CacheAwareConfig()
    first = create_prompt_cache_tree_client(config)
    second = create_prompt_cache_tree_client(config)
    assert isinstance(first, LocalPromptCacheTreeClient)
    assert isinstance(second, LocalPromptCacheTreeClient)
    prompt = "shared prefix" * 200

    async def run():
        await first.insert_sampled(_key(prompt), "prefill", SAMPLE_STRIDE)
        match = await first.prefix_match_sampled(
            _key(prompt), len(prompt), SAMPLE_STRIDE
        )
        missing = await second.prefix_match_sampled(
            _key(prompt), len(prompt), SAMPLE_STRIDE
        )
        assert match.prefill_node == "prefill"
        assert missing.prefill_node is None

    asyncio.run(run())


def test_remote_client_uses_symmetric_sampled_operations(monkeypatch):
    monkeypatch.setattr(prompt_cache_tree_client, "unix_connect", Mock())
    tree = _remote_tree("test-prompt-cache")
    prompt = "0123456789abcdef" * 128
    result = PromptCacheMatchResult(None, 0, len(prompt), None)
    tree._prefix_match_sampled = Mock(return_value=result)
    tree._insert_sampled = Mock()

    async def run():
        assert (
            await tree.prefix_match_sampled(_key(prompt), len(prompt), SAMPLE_STRIDE)
            is result
        )
        await tree.insert_sampled(_key(prompt), "prefill", SAMPLE_STRIDE)

    asyncio.run(run())
    tree._prefix_match_sampled.assert_called_once_with(
        _key(prompt), len(prompt), SAMPLE_STRIDE
    )
    tree._insert_sampled.assert_called_once_with(
        _key(prompt), "prefill", SAMPLE_STRIDE
    )


def test_tree_state_does_not_create_node_statistics():
    tree_state = PromptCacheTreeState()
    stats_state = PDNodeStatsState()
    prompt = "prefix" * 200
    tree_state.insert_sampled(_key(prompt), "prefill", SAMPLE_STRIDE)
    match = tree_state.prefix_match_sampled(_key(prompt), len(prompt), SAMPLE_STRIDE)
    assert match.prefill_node == "prefill"
    assert not stats_state.stats


@pytest.fixture
def rpc_tree_cluster():
    state = PromptCacheTreeState()
    ipc_name = uuid.uuid4().hex
    server = ThreadedServer(
        state,
        socket_path=get_abstract_unix_socket_address("prompt_cache_tree", ipc_name),
        protocol_config={"allow_pickle": True},
    )
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    clients = [_remote_tree(ipc_name) for _ in range(4)]
    try:
        yield state, clients
    finally:
        for client in clients:
            connection = client.connection
            if connection is not None:
                connection.close()
        server.close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_real_rpc_clients_share_prompt_cache(rpc_tree_cluster):
    _, clients = rpc_tree_cluster
    prompt = "shared prefix" * 200

    async def run():
        await clients[0].insert_sampled(_key(prompt), "prefill", SAMPLE_STRIDE)
        match = await clients[1].prefix_match_sampled(
            _key(prompt), len(prompt), SAMPLE_STRIDE
        )
        assert match.prefill_node == "prefill"
        assert match.matched_char_count > len(prompt) // 2
        transport = clients[0].connection._channel.stream.sock
        assert transport.family == socket.AF_UNIX

    asyncio.run(run())


def test_concurrent_prompts_remain_separate(rpc_tree_cluster):
    _, clients = rpc_tree_cluster
    prompts = ["a" * 2048, "b" * 2048]

    async def run():
        await asyncio.gather(
            *(
                clients[index].insert_sampled(
                    _key(prompt), f"prefill-{index}", SAMPLE_STRIDE
                )
                for index, prompt in enumerate(prompts)
            )
        )
        matches = await asyncio.gather(
            *(
                clients[index + 2].prefix_match_sampled(
                    _key(prompt), len(prompt), SAMPLE_STRIDE
                )
                for index, prompt in enumerate(prompts)
            )
        )
        assert [match.prefill_node for match in matches] == [
            "prefill-0",
            "prefill-1",
        ]

    asyncio.run(run())


def test_tree_and_statistics_servers_use_independent_sockets_and_state():
    ipc_name = uuid.uuid4().hex
    tree_address = get_abstract_unix_socket_address("prompt_cache_tree", ipc_name)
    stats_address = get_abstract_unix_socket_address("pd_node_stats", ipc_name)
    tree_state, stats_state = PromptCacheTreeState(), PDNodeStatsState()
    tree_server = ThreadedServer(
        tree_state,
        socket_path=tree_address,
        protocol_config={"allow_pickle": True},
    )
    stats_server = ThreadedServer(
        stats_state,
        socket_path=stats_address,
        protocol_config={"allow_pickle": True},
    )
    tree_thread = threading.Thread(target=tree_server.start, daemon=True)
    stats_thread = threading.Thread(target=stats_server.start, daemon=True)
    tree_thread.start()
    stats_thread.start()
    tree = _remote_tree(ipc_name)
    try:
        prompt = "prefix" * 200
        asyncio.run(tree.insert_sampled(_key(prompt), "prefill", SAMPLE_STRIDE))
        assert not stats_state.stats
        tree_server.close()
        tree_thread.join(timeout=5)
        assert not tree_thread.is_alive()
        assert stats_thread.is_alive()
        assert stats_state.get_dispatched_info(["prefill"]) == {
            "prefill": (0, 0)
        }
    finally:
        if tree.connection is not None:
            tree.connection.close()
        tree_server.close()
        stats_server.close()
        tree_thread.join(timeout=5)
        stats_thread.join(timeout=5)
