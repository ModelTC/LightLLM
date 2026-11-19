"""Decode 推理进程把上报推到这里，这里再写给目标 PD Master 的 /pd_report。"""

import asyncio
import inspect
import os
import pickle
import socket
from multiprocessing.connection import Connection
from typing import Dict

import setproctitle

from lightllm.server.core.objs import StartArgs
from lightllm.server.pd_io_struct import ObjType, PDUpKVStatus, PD_Master_Obj
from lightllm.server.pd_report.ipc import ReportIpc
from lightllm.utils.envs_utils import get_lightllm_websocket_max_message_size, get_unique_server_name
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.log_utils import init_logger
from lightllm.utils.net_utils import get_pd_report_ipc_address
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.websocket_utils import PD_WEBSOCKET_WRITE_LIMIT

logger = init_logger(__name__)

_HEARTBEAT_SECONDS = 15


def start_pd_report_process(args: StartArgs, pipe_writer: Connection) -> None:
    start_parent_check_thread(parent_pid=os.getppid())
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::pd_report")
    pull = ReportIpc.Pull(address=get_pd_report_ipc_address())
    pull.bind()
    pipe_writer.send("init ok")
    asyncio.run(PdReportManager(args=args, pull=pull).run())


class PdReportManager:
    """按 pd_master_node_id 把本机上报写到对应 Master。"""

    def __init__(self, args: StartArgs, pull: ReportIpc.Pull) -> None:
        self._args = args
        self._pull = pull
        self._queues: Dict[int, asyncio.Queue] = {}
        self._tasks: Dict[int, asyncio.Task] = {}

    async def run(self) -> None:
        recv_task = asyncio.create_task(self._recv_reports())
        try:
            await self._watch_masters()
        finally:
            recv_task.cancel()
            for task in self._tasks.values():
                task.cancel()
            await asyncio.gather(recv_task, *self._tasks.values(), return_exceptions=True)
            self._pull.close()

    def forward_report(self, message: object) -> None:
        if not isinstance(message, tuple) or len(message) != 2:
            logger.error(f"pd report dropped malformed local message {message!r}")
            os._exit(1)
        master_id, status = message
        if not isinstance(status, PDUpKVStatus) or status.pd_master_node_id != master_id:
            logger.error(f"pd report dropped status for master {master_id}: {status}")
            return
        queue = self._queues.get(master_id)
        if queue is None:
            logger.error(f"pd report has no connection to master {master_id}")
            return
        queue.put_nowait((ObjType.PD_UPLOAD_KV_STATUS, status))

    async def _recv_reports(self) -> None:
        while True:
            self.forward_report(message=await self._pull.recv())

    async def _watch_masters(self) -> None:
        from lightllm.server.httpserver.pd_loop import _get_pd_master_objs

        while True:
            try:
                masters = await _get_pd_master_objs(self._args)
                if masters is not None:
                    for node_id in list(self._tasks):
                        if node_id not in masters:
                            self._tasks.pop(node_id).cancel()
                    for node_id, master in masters.items():
                        if node_id in self._tasks:
                            continue
                        queue: asyncio.Queue = asyncio.Queue()
                        self._queues[node_id] = queue
                        self._tasks[node_id] = asyncio.create_task(self._serve_master(master=master, outgoing=queue))
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(f"pd report discovery failed: {exc}")
                await asyncio.sleep(10)

    async def _serve_master(self, master: PD_Master_Obj, outgoing: asyncio.Queue) -> None:
        import websockets

        uri = f"ws://{master.host_ip_port}/pd_report"

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(_HEARTBEAT_SECONDS)
                outgoing.put_nowait((ObjType.HEARTBEAT,))

        try:
            while True:
                heartbeat_task = None
                try:
                    async with websockets.connect(
                        uri=uri,
                        compression=None,
                        ping_interval=None,
                        max_size=get_lightllm_websocket_max_message_size(),
                        write_limit=PD_WEBSOCKET_WRITE_LIMIT,
                    ) as websocket:
                        sock = websocket.transport.get_extra_info("socket")
                        if sock is not None:
                            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        heartbeat_task = asyncio.create_task(heartbeat())
                        while True:
                            payload = await outgoing.get()
                            await websocket.send(pickle.dumps(payload))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception(f"pd report master {master.node_id} failed: {exc}")
                finally:
                    if heartbeat_task is not None:
                        heartbeat_task.cancel()
                        await asyncio.gather(heartbeat_task, return_exceptions=True)
                await asyncio.sleep(10)
        finally:
            if self._queues.get(master.node_id) is outgoing:
                self._queues.pop(master.node_id, None)
