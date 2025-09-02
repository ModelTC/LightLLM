import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import pickle
import inspect
import setproctitle
from typing import List
from lightllm.server.core.objs.io_objs.group_req import GroupReqIndexes
from lightllm.server.core.objs import ShmReqManager

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from .model_infer.model_rpc import start_model_process, VisualModelRpcClient
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from rpyc.utils.classic import obtain


logger = init_logger(__name__)


class VisualManager:
    def __init__(
        self,
        args,
        next_module_port,
        visual_port,
        cache_port,
        visual_model_rpc_ports,
    ):
        context = zmq.Context(2)
        self.send_to_next_module = context.socket(zmq.PUSH)  # router or audio server (if --enable_multimodal_audio)
        self.send_to_next_module.connect(f"{args.zmq_mode}127.0.0.1:{next_module_port}")

        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"{args.zmq_mode}127.0.0.1:{visual_port}")
        self.cache_client = rpyc.connect("localhost", cache_port, config={"allow_pickle": True})
        self.cache_port = cache_port
        self.waiting_reqs: List[GroupReqIndexes] = []
        self.model_weightdir = args.model_dir
        self.tp_world_size = args.tp
        self.vit_dp = args.visual_dp
        self.vit_tp = args.visual_tp
        self.infer_batch_size = args.visual_infer_batch_size
        self.trust_remote_code = args.trust_remote_code
        self.args = args
        self.visual_model_rpc_ports = visual_model_rpc_ports
        self.shm_req_manager = ShmReqManager()

    async def wait_to_model_ready(self):

        self.model_rpcs: List[List[VisualModelRpcClient]] = [[] for _ in range(self.vit_dp)]

        for dp_rank_id in range(self.vit_dp):
            tp_ports_each_dp = self.visual_model_rpc_ports[dp_rank_id]
            for tp_rank_id in range(self.vit_tp):
                device_id = self.args.visual_gpu_ids[dp_rank_id * self.vit_tp + tp_rank_id]
                rpc_model = await start_model_process(
                    port=tp_ports_each_dp[tp_rank_id], vit_tp=self.vit_tp, device_id=device_id
                )
                self.model_rpcs[dp_rank_id].append(rpc_model)

        init_model_ret = []
        for dp_rank_id in range(self.vit_dp):  # async init model process
            for tp_rank_id in range(self.vit_tp):
                kvargs = {
                    "weight_dir": self.model_weightdir,
                    "trust_remote_code": self.trust_remote_code,
                    "vit_dp": self.vit_dp,
                    "vit_tp": self.vit_tp,
                    "cache_port": self.cache_port,
                    "tp_rank_id": tp_rank_id,
                    "dp_rank_id": dp_rank_id,
                    "vit_rank_id": dp_rank_id * self.vit_tp + tp_rank_id,
                    "data_type": self.args.data_type,
                    "visual_nccl_port": self.args.visual_nccl_ports[dp_rank_id],
                    "visual_gpu_ids": self.args.visual_gpu_ids,
                    "quant_type": self.args.vit_quant_type,
                    "quant_cfg": self.args.vit_quant_cfg,
                    "max_batch_size": min(self.infer_batch_size // self.vit_dp, 1),
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

    async def loop_for_fwd(self):
        while True:
            if len(self.waiting_reqs) == 0:
                await asyncio.sleep(0.01)  # 10ms
            else:
                processing_group_reqs = []
                images_need_infer = []
                while len(self.waiting_reqs) > 0:
                    group_req_indexes = self.waiting_reqs.pop(0)
                    shm_req = self.shm_req_manager.get_req_obj_by_index(group_req_indexes.shm_req_indexes[0])
                    is_aborted = shm_req.is_aborted
                    self.shm_req_manager.put_back_req_obj(shm_req)
                    if is_aborted:
                        # 因为连接断开 aborted 掉的请求也需要传输到后续的模块进行处理
                        # 因为采用 shm 来映射所有的 req 对象以后，引用管理情况复杂了
                        # 需要一些一致的流程来保证不出现异步问题。
                        self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                        continue

                    multimodal_params = group_req_indexes.multimodal_params

                    img_uuids = [img.uuid for img in multimodal_params.images]
                    ready_image = obtain(self.cache_client.root.get_items_embed(img_uuids))

                    for img, ready in zip(multimodal_params.images, ready_image):
                        if not ready:
                            images_need_infer.append(img)

                        if len(images_need_infer) == self.infer_batch_size:
                            await self.infer_imgs(images_need_infer)
                            images_need_infer = []
                            for _group_req_indexes in processing_group_reqs:
                                self.send_to_next_module.send_pyobj(
                                    _group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL
                                )
                            processing_group_reqs = []

                    if len(images_need_infer) == 0:
                        self.send_to_next_module.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        processing_group_reqs.append(group_req_indexes)

                if len(images_need_infer) > 0:
                    await self.infer_imgs(images_need_infer)
                    for _group_req_indexes in processing_group_reqs:
                        self.send_to_next_module.send_pyobj(_group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                    processing_group_reqs = []
                    images_need_infer = []

    async def loop_for_netio_req(self):
        if not hasattr(self, "visual_recv_max_count"):
            self.visual_recv_max_count = 64

        while True:
            try:
                for _ in range(self.visual_recv_max_count):
                    recv_req: GroupReqIndexes = self.recv_from_httpserver.recv_pyobj(zmq.NOBLOCK)
                    if isinstance(recv_req, GroupReqIndexes):
                        self.waiting_reqs.append(recv_req)
                    else:
                        assert False, f"Error Req Inf {recv_req}"
                self.visual_recv_max_count = min(self.visual_recv_max_count * 1.3, 256)
            except zmq.ZMQError:
                # 当队列已经开始清空的时候，将一次接受数量下调
                self.visual_recv_max_count = 64
            await asyncio.sleep(0.01)

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return


def start_visual_process(args, next_module_port, visual_port, cache_port, model_rpc_ports, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::visual_server")
    start_parent_check_thread()
    try:
        visualserver = VisualManager(args, next_module_port, visual_port, cache_port, model_rpc_ports)
        asyncio.run(visualserver.wait_to_model_ready())
    except Exception as e:
        logger.exception(str(e))
        visualserver.clean_up()
        raise e

    pipe_writer.send("init ok")

    def handle_exception(loop, context):
        logger.exception(f"VisualServer Caught exception: {str(context)}")

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    asyncio.set_event_loop(loop)
    loop.create_task(visualserver.loop_for_fwd())
    loop.run_until_complete(visualserver.loop_for_netio_req())
    return
