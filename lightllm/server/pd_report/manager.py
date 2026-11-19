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


def start_pd_report_process(args: StartArgs, pipe_writer: Connection) -> None:
    start_parent_check_thread(parent_pid=os.getppid())
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::pd_report")
    pull = ReportIpc.Pull(address=get_pd_report_ipc_address())
    pull.bind()
    pipe_writer.send("init ok")
    asyncio.run(PdReportManager(args=args, pull=pull).run())


class PdReportManager:
    """把本机 Decode 推理进程的 KV 握手转到对应 PD Master 的 /pd_report。

    本地 ZMQ PULL 收到 PDUpKVStatus 后，按它自己的 pd_master_node_id 放进该 Master
    的发送队列。每个 Master 一条 websocket，断线只重连这一条，不碰其他 Master。
    """

    def __init__(self, args: StartArgs, pull: ReportIpc.Pull) -> None:
        """记下启动参数、本地 PULL，以及每个 Master 的发送队列和连接任务。

        _queues 的 key 是 pd_master_node_id。上报比连接更早到时，先建好队列把握手留住。
        队列在对应的 _serve_master 任务里被消费。重连期间队列保留，没写成功的握手会先补发。
        """
        self._args = args
        self._pull = pull
        self._queues: Dict[int, asyncio.Queue] = {}
        self._tasks: Dict[int, asyncio.Task] = {}
        logger.info("pd report manager created")

    async def run(self) -> None:
        """同时跑收包和 Master 发现。本协程结束时停掉这两边并关闭 PULL。

        收包任务一直从本地 ZMQ 取上报。发现循环按当前 Master 列表建连或取消连接。
        发现循环被取消后，这里取消收包和全部 Master 连接，等它们退出，再关闭 PULL。
        """
        logger.info("pd report manager started")
        recv_task = asyncio.create_task(self._recv_reports())
        try:
            await self._watch_masters()
        finally:
            logger.info("pd report manager stopping")
            recv_task.cancel()
            for task in self._tasks.values():
                task.cancel()
            await asyncio.gather(recv_task, *self._tasks.values(), return_exceptions=True)
            self._pull.close()
            logger.info("pd report manager stopped")

    async def _recv_reports(self) -> None:
        """从本地 PULL 收 PDUpKVStatus，按它的 pd_master_node_id 放进对应发送队列。

        收到的不是这个对象，说明本机通道坏了，记日志后退出进程。目标 Master 还没有队列时先建一个，
        握手留在里面，等连接建立后由发送循环写出。收包或反序列化失败由 PULL 记日志并退出进程。
        本循环被取消时，CancelledError 继续向外抛。
        """
        logger.info("pd report recv loop started")
        while True:
            status = await self._pull.recv()
            if not isinstance(status, PDUpKVStatus):
                logger.error(f"pd report dropped malformed local message {status!r}")
                os._exit(1)
            master_id = status.pd_master_node_id
            queue = self._queues.get(master_id)
            # 连接可能还没建立。先备好队列把这条握手留住，等 _serve_master 连上再发。
            if queue is None:
                queue = asyncio.Queue()
                self._queues[master_id] = queue
                logger.info(f"pd report create queue for master {master_id}, send after it connects")

                async def drop_if_absent(master_id: int, queue: asyncio.Queue) -> None:
                    # 给发现结果 3 分钟。到时这个 master 还没有连接任务，说明它不会来了，清掉自己的队列。
                    await asyncio.sleep(3 * 60)
                    if master_id not in self._tasks and self._queues.get(master_id) is queue:
                        logger.info(f"pd report drop stale queue for master {master_id}")
                        self._queues.pop(master_id, None)

                asyncio.create_task(drop_if_absent(master_id=master_id, queue=queue))
            queue.put_nowait((ObjType.PD_UPLOAD_KV_STATUS, status))
            logger.info(f"pd report queued upkv_status for master {master_id}: {status}")

    async def _watch_masters(self) -> None:
        """每 30 秒按当前 PD Master 列表对齐连接。

        列表里已经没有的 Master，取消它的连接任务，并摘掉它的发送队列。新出现的 Master 复用已经有的发送队列，
        没有再新建，然后启动 _serve_master。已经在连的 Master 保持原任务和原队列，不重新建连。
        查询失败记日志，10 秒后再查。本循环被取消时，CancelledError 继续向外抛。
        """
        from lightllm.server.httpserver.pd_loop import _get_pd_master_objs

        logger.info("pd report master watch started")
        while True:
            try:
                masters = await _get_pd_master_objs(self._args)
                if masters is not None:
                    for node_id in list(self._tasks):
                        if node_id not in masters:
                            logger.info(f"pd report master {node_id} left, cancel connection and drop queue")
                            self._tasks.pop(node_id).cancel()
                            # 马上摘掉队列，离开后的旧队列不再接收新上报。
                            # 连接任务要等事件循环才真正结束，结束时还会按队列身份再核对一次。
                            self._queues.pop(node_id, None)
                    for node_id, master in masters.items():
                        if node_id in self._tasks:
                            continue
                        logger.info(f"pd report start master {node_id} {master.host_ip_port}")
                        # 复用上报提前建好的队列，连上后发出积压的握手。换成新的空队列会把这些握手丢掉。
                        queue = self._queues.get(node_id)
                        if queue is None:
                            queue = asyncio.Queue()
                            self._queues[node_id] = queue
                        self._tasks[node_id] = asyncio.create_task(self._serve_master(master=master, outgoing=queue))
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                logger.info("pd report master watch stopped")
                raise
            except Exception as exc:
                logger.exception(f"pd report discovery failed: {exc}")
                await asyncio.sleep(10)

    async def _serve_master(self, master: PD_Master_Obj, outgoing: asyncio.Queue) -> None:
        """维持到一个 PD Master /pd_report 的 websocket，并把发送队列里的内容写出去。

        连上后打开 TCP_NODELAY。存活靠协议层 Ping：每 60 秒发一次，60 秒等不到 Pong 就断开。
        已经从队列取出但没写成功的报告留在本地，重连后先发它。这条连接失败只重连这一个
        Master，10 秒后再连。任务结束时，只有队列仍是自己这把才删掉，避免删掉后继任务新建的队列。
        """
        import websockets

        uri = f"ws://{master.host_ip_port}/pd_report"

        logger.info(f"pd report serve master {master.node_id} {uri}")
        pending = None
        try:
            while True:
                try:
                    async with websockets.connect(
                        uri=uri,
                        compression=None,
                        ping_interval=60,
                        ping_timeout=60,
                        max_size=get_lightllm_websocket_max_message_size(),
                        write_limit=PD_WEBSOCKET_WRITE_LIMIT,
                    ) as websocket:
                        logger.info(f"pd report connected master {master.node_id} {uri}")
                        sock = websocket.transport.get_extra_info("socket")
                        if sock is not None:
                            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        while True:
                            if pending is None:
                                pending = await outgoing.get()
                            await websocket.send(pickle.dumps(pending))
                            pending = None
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    logger.exception(f"pd report master {master.node_id} failed: {exc}")
                logger.info(f"pd report reconnect master {master.node_id} in 10s")
                await asyncio.sleep(10)
        finally:
            logger.info(f"pd report stopped master {master.node_id}")
            # 发现循环可能已经摘掉这把队列，空档里的新上报会再建一把。
            # 只删除仍然是自己这把的队列，避免把新建的那把删掉。
            if self._queues.get(master.node_id) is outgoing:
                self._queues.pop(master.node_id, None)
