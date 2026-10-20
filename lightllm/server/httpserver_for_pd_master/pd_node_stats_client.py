"""Worker-side access to PD node dispatch statistics."""

import asyncio
import threading

from rpyc.utils.classic import obtain
from rpyc.utils.factory import unix_connect

from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.net_utils import get_pd_master_unix_socket_address
from .async_utils import await_cancellation_safe

RPC_TIMEOUT_SECONDS = 5


class LocalPDNodeStats:
    """Keep dispatch statistics on the worker's node objects."""

    async def update_dispatched_info(self, p_nodes: list[PD_Client_Obj]) -> None:
        return

    async def add_dispatched_info(self, p_node: PD_Client_Obj, chars: int, req_num: int) -> None:
        p_node.dispatched_prompt_chars += chars
        p_node.dispatched_req_num += req_num

    async def sub_dispatched_info(self, p_node: PD_Client_Obj, chars: int, req_num: int) -> None:
        p_node.dispatched_prompt_chars = max(0, p_node.dispatched_prompt_chars - chars)
        p_node.dispatched_req_num = max(0, p_node.dispatched_req_num - req_num)


class RemotePDNodeStats:
    """Share dispatch statistics between PD master workers through RPyC."""

    def __init__(self):
        self.socket_path = get_pd_master_unix_socket_address("pd_node_stats")
        self.lock = threading.Lock()
        self.async_lock = asyncio.Lock()
        self.connection = unix_connect(
            self.socket_path,
            config={
                "sync_request_timeout": RPC_TIMEOUT_SECONDS,
                "allow_pickle": True,
            },
        )

    async def update_dispatched_info(self, p_nodes: list[PD_Client_Obj]) -> None:
        p_nodes = list(p_nodes)
        async with self.async_lock:
            result = await await_cancellation_safe(
                asyncio.to_thread(
                    self._get_dispatched_info,
                    [node.connection_key for node in p_nodes],
                )
            )
        for node in p_nodes:
            node.dispatched_prompt_chars, node.dispatched_req_num = result[node.connection_key]

    async def add_dispatched_info(self, p_node: PD_Client_Obj, chars: int, req_num: int) -> None:
        async with self.async_lock:
            await await_cancellation_safe(
                asyncio.to_thread(
                    self._add_dispatched_info,
                    p_node.connection_key,
                    chars,
                    req_num,
                )
            )

    async def sub_dispatched_info(self, p_node: PD_Client_Obj, chars: int, req_num: int) -> None:
        async with self.async_lock:
            await await_cancellation_safe(
                asyncio.to_thread(
                    self._sub_dispatched_info,
                    p_node.connection_key,
                    chars,
                    req_num,
                )
            )

    def _get_dispatched_info(self, connection_keys: list[str]):
        with self.lock:
            return obtain(self.connection.root.get_dispatched_info(connection_keys))

    def _add_dispatched_info(self, connection_key: str, chars: int, req_num: int) -> None:
        with self.lock:
            obtain(self.connection.root.add_dispatched_info(connection_key, chars, req_num))

    def _sub_dispatched_info(self, connection_key: str, chars: int, req_num: int) -> None:
        with self.lock:
            obtain(self.connection.root.sub_dispatched_info(connection_key, chars, req_num))


def create_pd_node_stats_client():
    if get_env_start_args().httpserver_workers > 1:
        return RemotePDNodeStats()
    return LocalPDNodeStats()
