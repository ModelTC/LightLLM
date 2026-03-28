import os
import queue
import threading
import asyncio
import inspect
import uuid
import rpyc
from lightllm.utils.retry_utils import retry
from rpyc.utils.factory import unix_connect
from typing import List, Any, Deque, Tuple
from .model_rpc import VisualModelRpcServer, VisualModelRpcClient
from lightllm.server.multimodal_params import ImageItem
from lightllm.server.embed_cache.afs_utils import SepEmbedHandler
from rpyc.utils.server import ThreadedServer
from lightllm.utils.envs_utils import get_env_start_args
from rpyc.utils.classic import obtain
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

class VisualOnlyModelRpcServer(VisualModelRpcServer):
    """
    完善这个代码:
    1. 创建一个队列, 用于接受别人放入的task,
    2. 创建一个线程，从队列中取出任务，完成后，修改task中的event，让放入的人得到结果和通知。这是任务循环。
    3. 能不能封装比较易读的流程。
    """

    def __init__(self):
        super().__init__()

        # 控制每次的最大推理图片数量，防止爆显存
        self.max_infer_batch_size = get_env_start_args().visual_infer_batch_size

        # 异步队列, 用于接受任务
        self.infer_queue = queue.Queue()
        # 将计算得到的结果放入 afs 的queue
        self.put_afs_queue = queue.Queue()

        # 限制并发, 主要控制内存用量，防止过多造成内存OOM
        self.sempare = threading.Semaphore(self.max_infer_batch_size * 8)

        self.afs_handler = SepEmbedHandler(
            afs_embed_dir=get_env_start_args().afs_embed_dir,
            redis_host=get_env_start_args().config_server_host,
            redis_port=get_env_start_args().config_server_vit_redis_port,
            capacity=get_env_start_args().afs_embed_capacity,
        )

        # 启动任务处理线程
        self._infer_thread = threading.Thread(target=self._infer_worker, daemon=True)
        self._infer_thread.start()

        self._put_afs_thread = threading.Thread(target=self._put_afs_worker, daemon=True)
        self._put_afs_thread.start()

    def exposed_run_task(self, images: List["ImageItem"], ref_event: threading.Event):
        try:
            images = obtain(images)
            images[-1].event = ref_event

            for image in images:
                self.infer_queue.put(image)

        except BaseException as e:
            logger.exception(str(e))
            raise e
        return

    def _get_image_items_from_queue(self, max_num: int) -> List[ImageItem]:
        """
        从队列中批量获取任务，直到达到 max_num 或队列为空。
        """
        tasks = []
        # 至少获取一个任务，阻塞
        self.sempare.acquire()
        task = self.infer_queue.get(block=True)
        tasks.append(task)  
        
        # 尝试继续获取更多任务，直到达到 max_num
        while len(tasks) < max_num:
            try:
                self.sempare.acquire()
                task = self.infer_queue.get(block=False)
                tasks.append(task)
            except queue.Empty:
                self.sempare.release()
                break

        return tasks

    def _infer_worker(self):
        """
        任务处理循环: 从队列中取出任务, 执行完成后通知调用者
        """
        while True:
            try:
                # 从队列获取任务, 阻塞等待
                images = self._get_image_items_from_queue(max_num=self.max_infer_batch_size)

                # 执行任务: 调用父类的forward方法处理图像
                all_img_embeds, uuids, valid_ids = self.forward(images)
                all_img_embeds = all_img_embeds.detach().cpu()
                for image, valid_id in zip(images, valid_ids):
                    start, end = valid_id
                    self.put_afs_queue.put((image, all_img_embeds[start:end]))

            except Exception as e:
                logger.exception(str(e))
                raise e
            
    def _put_afs_worker(self):
        """
        任务处理循环: 从队列中取出ImageItem和embed 放入 afs中, 执行完成后通知调用者
        """
        while True:
            try:
                # 从队列获取任务, 阻塞等待
                image, embed = self.put_afs_queue.get(block=True)
                # 只有 0 rank 执行真的写入操作。
                if self.tp_rank_id == 0:
                    self.afs_handler.insert(image.md5, embed)
                if hasattr(image, "event"):
                    image.event.set()
                self.sempare.release()
            except Exception as e:
                logger.exception(str(e))
                raise e


def _init_env(socket_path: str, success_event):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    import lightllm.utils.rpyc_fix_utils as _

    t = ThreadedServer(VisualOnlyModelRpcServer(), socket_path=socket_path, protocol_config={"allow_pickle": True})
    success_event.set()
    t.start()
    return


async def start_model_process():
    import multiprocessing

    socket_path = _generate_unix_socket_path()
    if os.path.exists(socket_path):
        os.remove(socket_path)

    success_event = multiprocessing.Event()
    proc = multiprocessing.Process(
        target=_init_env,
        args=(
            socket_path,
            success_event,
        ),
    )
    proc.start()
    await asyncio.to_thread(success_event.wait, timeout=40)
    assert proc.is_alive()

    conn = retry(max_attempts=20, wait_time=2)(unix_connect)(socket_path, config={"allow_pickle": True})
    assert proc.is_alive()
    # 服务端需要调用event所以，客户端需要一个后台线程进行相关的处理。
    conn._bg_thread = rpyc.BgServingThread(conn)
    return VisualModelRpcClient(conn)


def _generate_unix_socket_path() -> str:
    """Generate a random Unix socket path"""
    unique_id = uuid.uuid4().hex[:8]
    return f"/tmp/lightllm_model_infer_{unique_id}.sock"
