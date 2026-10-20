"""Worker-side access to the local or shared prompt cache tree."""

import asyncio
import threading

from rpyc.utils.classic import obtain
from rpyc.utils.factory import unix_connect

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.net_utils import get_pd_master_unix_socket_address
from ..async_utils import await_cancellation_safe
from .prompt_cache_tree import PromptCacheMatchResult, PromptCacheTree

RPC_TIMEOUT_SECONDS = 5


class LocalPromptCacheTreeClient:
    """Adapt the worker-local prompt cache tree to the async client interface."""

    def __init__(self, config):
        self.prompt_cache_tree = PromptCacheTree(
            sample_stride=config.sample_stride,
            max_node_count=config.max_node_count,
            evict_node_batch=config.evict_node_batch,
        )

    async def prefix_match_sampled(self, key: str, input_char_count: int, sample_stride: int) -> PromptCacheMatchResult:
        return self.prompt_cache_tree.prefix_match_sampled(key, input_char_count, sample_stride)

    async def insert_sampled(self, key: str, prefill_node: str, sample_stride: int) -> None:
        self.prompt_cache_tree.insert_sampled(key, prefill_node, sample_stride)


class RemotePromptCacheTreeClient:
    """Share the prompt cache tree between PD master workers through RPyC."""

    def __init__(self):
        self.socket_path = get_pd_master_unix_socket_address("prompt_cache_tree")
        self.lock = threading.Lock()
        self.async_lock = asyncio.Lock()
        self.connection = unix_connect(
            self.socket_path,
            config={
                "sync_request_timeout": RPC_TIMEOUT_SECONDS,
                "allow_pickle": True,
            },
        )

    async def prefix_match_sampled(self, key: str, input_char_count: int, sample_stride: int) -> PromptCacheMatchResult:
        async with self.async_lock:
            return await await_cancellation_safe(
                asyncio.to_thread(
                    self._prefix_match_sampled,
                    key,
                    input_char_count,
                    sample_stride,
                )
            )

    async def insert_sampled(self, key: str, prefill_node: str, sample_stride: int) -> None:
        async with self.async_lock:
            await await_cancellation_safe(
                asyncio.to_thread(
                    self._insert_sampled,
                    key,
                    prefill_node,
                    sample_stride,
                )
            )

    def _prefix_match_sampled(self, key: str, input_char_count: int, sample_stride: int) -> PromptCacheMatchResult:
        with self.lock:
            return obtain(self.connection.root.prefix_match_sampled(key, input_char_count, sample_stride))

    def _insert_sampled(self, key: str, prefill_node: str, sample_stride: int) -> None:
        with self.lock:
            obtain(self.connection.root.insert_sampled(key, prefill_node, sample_stride))


def create_prompt_cache_tree_client(config):
    if get_env_start_args().httpserver_workers > 1:
        return RemotePromptCacheTreeClient()
    return LocalPromptCacheTreeClient(config=config)
