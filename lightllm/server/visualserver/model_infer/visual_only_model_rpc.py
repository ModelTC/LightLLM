import os
import queue
import threading
import dataclasses
import rpyc
import socket
import asyncio
import inspect
import uuid
from lightllm.utils.retry_utils import retry
from rpyc.utils.factory import unix_connect
from typing import List, Any
from .model_rpc import VisualModelRpcServer, VisualModelRpcClient
from lightllm.server.multimodal_params import ImageItem
from lightllm.server.embed_cache.afs_utils import SepEmbedHandler
from rpyc.utils.server import ThreadedServer
from lightllm.utils.envs_utils import get_env_start_args
from rpyc.utils.classic import obtain
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class _Task:
    images: List["ImageItem"]
    ret: Any
    event: threading.Event
    hasError: bool = False

    def wait(self, timeout: float = None):
        self.event.wait(timeout=timeout)


class VisualOnlyModelRpcServer(VisualModelRpcServer):
    """
    完善这个代码:
    1. 创建一个队列, 用于接受别人放入的task,
    2. 创建一个线程，从队列中取出任务，完成后，修改task中的event，让放入的人得到结果和通知。这是任务循环。
    3. 能不能封装比较易读的流程。
    """

    def __init__(self):
        super().__init__()

        # 异步队列, 用于接受任务
        self.task_queue = queue.Queue()
        # 限制并发, 主要控制内存用量，防止过多照成爆炸。
        self.sempare = threading.Semaphore(3)

        self.afs_handler = SepEmbedHandler(
            afs_embed_dir=get_env_start_args().afs_embed_dir,
            redis_host=get_env_start_args().config_server_host,
            redis_port=get_env_start_args().config_server_vit_redis_port,
            capacity=get_env_start_args().afs_embed_capacity,
        )

        # 启动任务处理线程
        self.worker_thread = threading.Thread(target=self._task_worker, daemon=True)
        self.worker_thread.start()

    def _task_worker(self):
        """
        任务处理循环: 从队列中取出任务, 执行完成后通知调用者
        """
        while True:
            try:
                # 从队列获取任务, 阻塞等待
                task: _Task = self.task_queue.get()

                # 执行任务: 调用父类的forward方法处理图像
                try:
                    all_img_embeds, uuids, valid_ids = self.forward(task.images)
                    all_img_embeds = all_img_embeds.detach().cpu()

                    # 存储结果到task.ret
                    task.ret = {"embeds": all_img_embeds, "valid_ids": valid_ids}
                except Exception as e:
                    task.hasError = True
                    logger.exception(str(e))
                    raise e
                finally:
                    # 标记任务完成, 唤醒等待的调用者
                    task.event.set()
                    self.task_queue.task_done()

            except Exception as e:
                logger.exception(str(e))
                raise e

    def exposed_run_task(self, images: List["ImageItem"]):
        """
        添加任务到队列

        Args:
            images: 要处理的图像列表

        Returns:
            _Task: 任务对象, 包含ret和event
        """
        images = obtain(images)
        with self.sempare:
            event = threading.Event()
            task = _Task(images=images, ret=None, event=event)
            self.task_queue.put(task)
            task.event.wait(timeout=8888)

        all_img_embeds = task.ret["embeds"]
        valid_ids = task.ret["valid_ids"]

        if self.tp_rank_id == 0:
            for i in enumerate(len(images)):
                start, end = valid_ids[i]
                image = images[i]
                self.afs_handler.insert(image.md5, all_img_embeds[start:end])
        return


def _init_env(socket_path: str, device_id: int, success_event):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    import lightllm.utils.rpyc_fix_utils as _

    t = ThreadedServer(VisualOnlyModelRpcServer(), socket_path=socket_path, protocol_config={"allow_pickle": True})
    success_event.set()
    t.start()
    return


async def start_model_process(vit_tp, device_id):
    import multiprocessing

    socket_path = _generate_unix_socket_path()
    if os.path.exists(socket_path):
        os.remove(socket_path)

    success_event = multiprocessing.Event()
    proc = multiprocessing.Process(
        target=_init_env,
        args=(
            socket_path,
            device_id,
            success_event,
        ),
    )
    proc.start()
    await asyncio.to_thread(success_event.wait, timeout=40)
    assert proc.is_alive()

    conn = retry(max_attempts=20, wait_time=2)(unix_connect)(socket_path, config={"allow_pickle": True})
    assert proc.is_alive()
    return VisualModelRpcClient(conn.root, vit_tp, rpc_server_process=proc)


def _generate_unix_socket_path() -> str:
    """Generate a random Unix socket path"""
    unique_id = uuid.uuid4().hex[:8]
    return f"/tmp/lightllm_model_infer_{unique_id}.sock"
