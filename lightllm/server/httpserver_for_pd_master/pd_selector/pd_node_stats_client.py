"""Worker-side access to PD node dispatch statistics."""

import asyncio
import os
import signal
import threading

from rpyc.utils.classic import obtain
from rpyc.utils.factory import unix_connect

from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.net_utils import get_pd_master_unix_socket_address
from ..async_utils import await_cancellation_safe

RPC_TIMEOUT_SECONDS = 5
logger = init_logger(__name__)


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
        result = await self._run_rpc(
            self._get_dispatched_info,
            tuple(node.connection_key for node in p_nodes),
        )
        if len(result) != len(p_nodes):
            raise RuntimeError("Shared PD node statistics returned an invalid result")
        for node, dispatched_info in zip(p_nodes, result):
            node.dispatched_prompt_chars, node.dispatched_req_num = dispatched_info

    async def add_dispatched_info(self, p_node: PD_Client_Obj, chars: int, req_num: int) -> None:
        await self._run_rpc(
            self._add_dispatched_info,
            p_node.connection_key,
            chars,
            req_num,
        )

    async def sub_dispatched_info(self, p_node: PD_Client_Obj, chars: int, req_num: int) -> None:
        await self._run_rpc(
            self._sub_dispatched_info,
            p_node.connection_key,
            chars,
            req_num,
        )

    async def _run_rpc(self, operation, *args):
        async def run_serialized():
            async with self.async_lock:
                return await asyncio.to_thread(operation, *args)

        try:
            # 从等待异步锁开始保护整个操作，调用者取消只能在 RPC 完成后生效。
            return await await_cancellation_safe(run_serialized())
        except asyncio.CancelledError:
            raise
        except BaseException:
            # 连接异常时无法确认远端是否已经修改计数。继续服务会使用未知状态，
            # 因此终止当前 worker，由 launcher 停止整组服务并重新初始化共享状态。
            logger.exception("Shared PD node statistics RPC failed; terminating worker")
            os.kill(os.getpid(), signal.SIGTERM)
            raise

    def _get_dispatched_info(self, connection_keys: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
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
