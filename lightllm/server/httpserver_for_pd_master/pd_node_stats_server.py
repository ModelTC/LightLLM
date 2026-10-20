"""Process-owned PD node dispatch statistics and their RPyC service."""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import rpyc
from rpyc.utils.classic import obtain

from lightllm.utils.net_utils import get_pd_master_unix_socket_address

MAX_NODE_STATS = 100_000
NODE_STATS_IDLE_SECONDS = 1800


@dataclass
class NodeStats:
    pending_chars: int = 0
    pending_requests: int = 0
    last_access: float = 0.0


@rpyc.service
class PDNodeStatsState(rpyc.Service):
    """RPyC service owning advisory dispatch counters."""

    def __init__(self):
        self.lock = threading.Lock()
        self.stats: OrderedDict[str, NodeStats] = OrderedDict()

    @rpyc.exposed
    def get_dispatched_info(self, connection_keys) -> dict[str, tuple[int, int]]:
        connection_keys = obtain(connection_keys)
        with self.lock:
            snapshot = {}
            for key in connection_keys:
                entry = self._get_stats(key)
                snapshot[key] = (entry.pending_chars, entry.pending_requests)
            return snapshot

    @rpyc.exposed
    def add_dispatched_info(self, connection_key, chars, req_num) -> None:
        connection_key, chars, req_num = obtain((connection_key, chars, req_num))
        with self.lock:
            chars, req_num = int(chars), int(req_num)
            entry = self._get_stats(connection_key)
            entry.pending_chars = max(0, entry.pending_chars + chars)
            entry.pending_requests = max(0, entry.pending_requests + req_num)

    @rpyc.exposed
    def sub_dispatched_info(self, connection_key, chars, req_num) -> None:
        connection_key, chars, req_num = obtain((connection_key, chars, req_num))
        with self.lock:
            chars, req_num = int(chars), int(req_num)
            entry = self._get_stats(connection_key)
            entry.pending_chars = max(0, entry.pending_chars - chars)
            entry.pending_requests = max(0, entry.pending_requests - req_num)

    def _get_stats(self, connection_key: str) -> NodeStats:
        """获取节点统计并刷新 LRU 顺序。

        每次访问先从 OrderedDict 头部清理超过空闲时间的节点；新节点使缓存
        达到容量上限时淘汰最久未访问的节点。最后更新访问时间并将当前节点
        移到队尾，因此字典头部始终是最久未访问的节点。
        """
        if not isinstance(connection_key, str) or not connection_key:
            raise ValueError("PD statistics key must be a nonempty string")
        now = time.monotonic()
        while self.stats:
            oldest = next(iter(self.stats.values()))
            if now - oldest.last_access < NODE_STATS_IDLE_SECONDS:
                break
            self.stats.popitem(last=False)
        entry = self.stats.get(connection_key)
        if entry is None:
            if len(self.stats) >= MAX_NODE_STATS:
                self.stats.popitem(last=False)
            entry = self.stats[connection_key] = NodeStats()
        entry.last_access = now
        self.stats.move_to_end(connection_key)
        return entry


def start_pd_node_stats_server(args, pipe_writer):
    from lightllm.utils.graceful_utils import graceful_registry
    from lightllm.utils.process_check import start_parent_check_thread
    from lightllm.utils.envs_utils import get_unique_server_name
    from rpyc.utils.server import ThreadedServer
    import setproctitle

    start_parent_check_thread()
    graceful_registry("start_pd_node_stats_server")
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::pd_node_stats_server")
    server = ThreadedServer(
        PDNodeStatsState(),
        socket_path=get_pd_master_unix_socket_address("pd_node_stats"),
        protocol_config={"allow_pickle": True},
    )
    pipe_writer.send("init ok")
    server.start()
