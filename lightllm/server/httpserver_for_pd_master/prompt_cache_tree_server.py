"""Process-owned prompt cache tree and its RPyC service."""

import threading

import rpyc
from rpyc.utils.classic import obtain

from lightllm.utils.net_utils import get_pd_master_unix_socket_address
from .pd_selector.cache_aware import CacheAwareConfig
from .pd_selector.prompt_cache_tree import PromptCacheTree


@rpyc.service
class PromptCacheTreeState(rpyc.Service):
    """RPyC service owning the shared prompt cache tree."""

    def __init__(self):
        self.lock = threading.Lock()
        self.config = CacheAwareConfig()
        self.prompt_cache_tree = PromptCacheTree(
            sample_stride=self.config.sample_stride,
            max_node_count=self.config.max_node_count,
            evict_node_batch=self.config.evict_node_batch,
        )

    @rpyc.exposed
    def prefix_match_sampled(self, key, input_char_count, sample_stride):
        key, input_char_count, sample_stride = obtain((key, input_char_count, sample_stride))
        with self.lock:
            return self.prompt_cache_tree.prefix_match_sampled(key, input_char_count, sample_stride)

    @rpyc.exposed
    def insert_sampled(self, key, prefill_node, sample_stride) -> None:
        key, prefill_node, sample_stride = obtain((key, prefill_node, sample_stride))
        with self.lock:
            self.prompt_cache_tree.insert_sampled(key, prefill_node, sample_stride)


def start_prompt_cache_tree_server(args, pipe_writer):
    from lightllm.utils.graceful_utils import graceful_registry
    from lightllm.utils.process_check import start_parent_check_thread
    from lightllm.utils.envs_utils import get_unique_server_name
    from rpyc.utils.server import ThreadedServer
    import setproctitle

    start_parent_check_thread()
    graceful_registry("start_prompt_cache_tree_server")
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::prompt_cache_tree_server")
    server = ThreadedServer(
        PromptCacheTreeState(),
        socket_path=get_pd_master_unix_socket_address("prompt_cache_tree"),
        protocol_config={"allow_pickle": True},
    )
    pipe_writer.send("init ok")
    server.start()
