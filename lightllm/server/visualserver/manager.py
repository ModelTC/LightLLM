import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import pickle
import inspect
from typing import List
from lightllm.server.core.objs.io_objs.group_req import GroupReqIndexes
from lightllm.server.core.objs import ShmReqManager

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from .model_infer.model_rpc import start_model_process, VisualModelRpcClient
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread


logger = init_logger(__name__)


class VisualManager:
    def __init__(
        self,
        args,
        router_port,
        visual_port,
        cache_port,
        visual_model_rpc_ports,
    ):
        context = zmq.asyncio.Context(2)
        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"{args.zmq_mode}127.0.0.1:{router_port}")

        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"{args.zmq_mode}127.0.0.1:{visual_port}")
        self.cache_client = rpyc.connect("localhost", cache_port)
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
                rpc_model = await start_model_process(port=tp_ports_each_dp[tp_rank_id], vit_tp=self.vit_tp)
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
                }
                init_model_ret.append(self.model_rpcs[dp_rank_id][tp_rank_id].init_model(kvargs))
        await asyncio.gather(*init_model_ret)
        return

    async def infer_imgs(self, uuids):
        if len(uuids) == 0:
            return

        tasks = []
        for vit_dp_rank in range(self.vit_dp):
            assigned_uuids = [uuids[i] for i in range(vit_dp_rank, len(uuids), self.vit_dp)]
            if assigned_uuids:
                for vit_tp_rank in range(self.vit_tp):
                    task = asyncio.create_task(self.model_rpcs[vit_dp_rank][vit_tp_rank].encode(assigned_uuids))
                    tasks.append(task)

        await asyncio.gather(*tasks)
        return

    async def loop_for_fwd(self):
        while True:
            if len(self.waiting_reqs) == 0:
                await asyncio.sleep(0.01)  # 10ms
            else:
                cur_batch_size = 0
                reqs_need_infer = []
                uuids_need_infer = []
                while cur_batch_size < self.infer_batch_size and len(self.waiting_reqs) > 0:
                    group_req_indexes = self.waiting_reqs.pop(0)
                    shm_req = self.shm_req_manager.get_req_obj_by_index(group_req_indexes.shm_req_indexes[0])
                    is_aborted = shm_req.is_aborted
                    self.shm_req_manager.put_back_req_obj(shm_req)

                    if is_aborted:
                        # 因为连接断开 aborted 掉的请求也需要传输到后续的模块进行处理
                        # 因为采用 shm 来映射所有的 req 对象以后，引用管理情况复杂了
                        # 需要一些一致的流程来保证不出现异步问题。
                        self.send_to_router.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                        continue

                    multimodal_params = group_req_indexes.multimodal_params

                    cur_uuids_need_infer = []
                    for img in multimodal_params.images:
                        if not self.cache_client.root.get_item_embed(img.uuid):
                            cur_batch_size += 1
                            cur_uuids_need_infer.append(img.uuid)

                    if len(cur_uuids_need_infer) == 0:
                        self.send_to_router.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        uuids_need_infer.extend(cur_uuids_need_infer)
                        reqs_need_infer.append((group_req_indexes, len(uuids_need_infer) - 1))

                for start_index in range(0, len(uuids_need_infer), self.infer_batch_size):
                    await self.infer_imgs(uuids_need_infer[start_index : (start_index + self.infer_batch_size)])
                    finished_req_indexes = [
                        group_req_indexes
                        for group_req_indexes, mark_index in reqs_need_infer
                        if mark_index < start_index + self.infer_batch_size
                    ]
                    reqs_need_infer = [
                        (group_req_indexes, mark_index)
                        for group_req_indexes, mark_index in reqs_need_infer
                        if mark_index >= start_index + self.infer_batch_size
                    ]

                    for group_req_indexes in finished_req_indexes:
                        self.send_to_router.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)

    async def loop_for_netio_req(self):
        while True:
            recv_req: GroupReqIndexes = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, GroupReqIndexes):
                self.waiting_reqs.append(recv_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return


def start_visual_process(args, router_port, visual_port, cache_port, model_rpc_ports, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    start_parent_check_thread()

    try:
        visualserver = VisualManager(args, router_port, visual_port, cache_port, model_rpc_ports)
        asyncio.run(visualserver.wait_to_model_ready())
    except Exception as e:
        logger.exception(str(e))
        visualserver.clean_up()
        raise e

    pipe_writer.send("init ok")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(visualserver.loop_for_fwd())
    loop.run_until_complete(visualserver.loop_for_netio_req())
    return
