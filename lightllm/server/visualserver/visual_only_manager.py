import asyncio
import uvloop
import rpyc
import inspect
import setproctitle
import threading
import collections
from typing import List
from lightllm.server.core.objs import StartArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from .model_infer import start_model_process, VisualModelRpcClient
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
        self.model_weightdir = args.model_dir
        self.vit_dp = args.visual_dp
        assert self.vit_dp == 1
        self.vit_tp = args.visual_tp
        # image 最大推理 batch size
        self.infer_batch_size = args.visual_infer_batch_size
        self.cur_dp_index = 0
        self.lock = threading.Lock()

        self.new_loop = asyncio.new_event_loop()
        t = threading.Thread(target=self.event_loop, args=(self.new_loop,), daemon=True)
        t.start()

    def event_loop(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()
        return

    async def wait_to_model_ready(self):

        self.model_rpcs: List[List[VisualModelRpcClient]] = [[] for _ in range(self.vit_dp)]
        self.vit_attn_backend = init_vit_att_backend(index=0)
        for dp_rank_id in range(self.vit_dp):
            for tp_rank_id in range(self.vit_tp):

                rpc_model = await start_model_process()
                self.model_rpcs[dp_rank_id].append(rpc_model)

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

    async def handle_reqs(self, images_need_infer: List[ImageItem]):
        # case 2
        dp_to_handle_images = collections.defaultdict(list)
        for image in images_need_infer:
            self.cur_dp_index += 1
            select_dp = self.cur_dp_index % self.vit_dp
            dp_to_handle_images[select_dp].append((image, threading.Event()))

        taskes = []
        for dp_index in range(self.vit_dp):
            _images = dp_to_handle_images[dp_index]
            if _images:
                taskes.extend(self.run_task(dp_index, images=[e[0] for e in _images], events=[e[1] for e in _images]))

        with self.lock:
            await asyncio.gather(*taskes)

        for dp_index in range(self.vit_dp):
            _images = dp_to_handle_images[dp_index]
            if _images:
                await asyncio.to_thread(_images[-1][1].wait)
        return

    def run_task(self, dp_index: int, images, events):
        taskes = []
        for vit_tp_rank in range(self.vit_tp):
            task = self.model_rpcs[dp_index][vit_tp_rank].run_task(images, events)
            taskes.append(task)
        return taskes

    def clean_up(self):
        return

    def exposed_infer_images(self, images: List[ImageItem], ref_event: threading.Event):
        try:
            images = obtain(images)
            # 将 images 的内容写入到 shm 中，

            handle = asyncio.run_coroutine_threadsafe(self.handle_reqs(images_need_infer=images), loop=self.new_loop)
            handle.result()

            ref_event.set()
        except BaseException as e:
            logger.exception(str(e))
            raise e
        finally:
            # 将 shm 进行删除
            pass

        return


def start_visual_process(args, pipe_writer):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::visual_server")
    start_parent_check_thread()

    try:
        visualserver = VisualManager(args=args)
        future = asyncio.run_coroutine_threadsafe(visualserver.wait_to_model_ready(), loop=visualserver.new_loop)
        future.result()
        t = rpyc.ThreadedServer(visualserver, port=None, protocol_config={"allow_pickle": True})
    except Exception as e:
        logger.exception(str(e))
        visualserver.clean_up()
        raise e

    pipe_writer.send("init ok")

    t.start()
    return
