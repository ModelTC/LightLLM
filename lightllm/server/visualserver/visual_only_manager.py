import asyncio
import uvloop
import inspect
import setproctitle
import threading
import queue
import dataclasses
import rpyc
from typing import List, Any
from lightllm.server.core.objs.io_objs.group_req import GroupReqIndexes
from lightllm.server.core.objs import StartArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from .model_infer.model_rpc import start_model_process, VisualModelRpcClient
from lightllm.common.basemodel.attention_vit.create_utils import init_vit_att_backend
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from rpyc.utils.classic import obtain


logger = init_logger(__name__)


class VisualManager(rpyc.Service):
    def __init__(
        self,
        args: StartArgs,
    ):
        self.args = args
        self.waiting_reqs: List[GroupReqIndexes] = []
        self.model_weightdir = args.model_dir
        self.vit_dp = args.visual_dp
        self.vit_tp = args.visual_tp
        assert self.vit_dp == 1
        # image 最大推理 batch size
        self.infer_batch_size = args.visual_infer_batch_size
        self.send_batch_size = args.visual_send_batch_size

        # 工作线程
        self.task_queue = queue.Queue()
        # 限制并发, 主要控制内存用量，防止过多照成爆炸。
        self.sempare = threading.Semaphore(3)
        # 启动任务处理线程
        self.worker_thread = threading.Thread(target=self._task_worker, daemon=True)
        self.worker_thread.start()

    async def wait_to_model_ready(self):

        self.model_rpcs: List[List[VisualModelRpcClient]] = [[] for _ in range(self.vit_dp)]
        self.model_rpcs_1: List[List[VisualModelRpcClient]] = [[] for _ in range(self.vit_dp)]
        self.vit_attn_backend = init_vit_att_backend(index=0)
        for dp_rank_id in range(self.vit_dp):
            for tp_rank_id in range(self.vit_tp):

                rpc_model = await start_model_process()
                self.model_rpcs[dp_rank_id].append(rpc_model[0])
                self.model_rpcs_1[dp_rank_id].append(rpc_model[1])

        init_model_ret = []
        for dp_rank_id in range(self.vit_dp):  # async init model process
            for tp_rank_id in range(self.vit_tp):
                device_id = self.args.visual_gpu_ids[dp_rank_id * self.vit_tp + tp_rank_id]
                kvargs = {
                    "weight_dir": self.model_weightdir,
                    "device_id": device_id,
                    "vit_tp": self.vit_tp,
                    "cache_port": self.args.cache_port,
                    "tp_rank_id": tp_rank_id,
                    "dp_rank_id": dp_rank_id,
                    "data_type": self.args.data_type,
                    "visual_nccl_port": self.args.visual_nccl_ports[dp_rank_id],
                    "quant_type": self.args.vit_quant_type,
                    "quant_cfg": self.args.vit_quant_cfg,
                    "max_batch_size": min(self.infer_batch_size // self.vit_dp, 1),
                    "vit_attn_backend": self.vit_attn_backend,
                }
                init_model_ret.append(self.model_rpcs[dp_rank_id][tp_rank_id].init_model(kvargs))
        await asyncio.gather(*init_model_ret)
        return

    async def infer_imgs(self, images: List[ImageItem]):
        if len(images) == 0:
            return

        tasks = []
        for vit_dp_rank in range(self.vit_dp):
            assigned_images = [images[i] for i in range(vit_dp_rank, len(images), self.vit_dp)]
            if assigned_images:
                for vit_tp_rank in range(self.vit_tp):
                    task = asyncio.create_task(self.model_rpcs[vit_dp_rank][vit_tp_rank].encode(assigned_images))
                    tasks.append(task)

        await asyncio.gather(*tasks)
        return

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

    def clean_up(self):
        return


def start_visual_process(args, pipe_writer):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::visual_server")
    start_parent_check_thread()
    try:
        visualserver = VisualManager(args=args)
        asyncio.run(visualserver.wait_to_model_ready())
        t = rpyc.ThreadedServer(visualserver, port=None, protocol_config={"allow_pickle": True})
    except Exception as e:
        logger.exception(str(e))
        visualserver.clean_up()
        raise e

    pipe_writer.send("init ok")

    t.start()
    return


@dataclasses.dataclass
class _Task:
    images: List["ImageItem"]
    ret: Any
    event: threading.Event
    hasError: bool = False

    def wait(self, timeout: float = None):
        self.event.wait(timeout=timeout)
