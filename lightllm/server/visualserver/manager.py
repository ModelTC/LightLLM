import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
from typing import List
from transformers import AutoConfig
from ..io_struct import AbortReq
from ..embed_cache.utils import tensor2bytes, read_shm, create_shm, get_shm_name_data, get_shm_name_embed
from rpyc.utils.classic import obtain

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from .model_infer.model_rpc import start_model_process, VisualModelRpcClient
from io import BytesIO
from PIL import Image
import time
import torch


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
        self.send_to_router.connect(f"tcp://127.0.0.1:{router_port}")

        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{visual_port}")
        self.cache_client = rpyc.connect("localhost", cache_port)
        self.cache_port = cache_port
        self.waiting_reqs = []
        self.model_weightdir = args.model_dir
        self.tp_world_size = args.tp
        self.vit_dp = args.visual_dp
        self.vit_tp = args.visual_tp
        self.infer_batch_size = args.visual_infer_batch_size
        self.trust_remote_code = args.trust_remote_code
        self.args = args
        self.visual_model_rpc_ports = visual_model_rpc_ports

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
                }
                init_model_ret.append(self.model_rpcs[dp_rank_id][tp_rank_id].init_model(kvargs))
        await asyncio.gather(*init_model_ret)
        return

    async def abort(self, group_req_id):
        abort_req = AbortReq(group_req_id=group_req_id)
        self.send_to_router.send_pyobj(abort_req)
        # 过滤掉被 aborted的请求。
        self.waiting_reqs = [req for req in self.waiting_reqs if req[3] != group_req_id]
        return

    async def infer_imgs(self, uuids):
        if len(uuids) == 0:
            return
        # uuids -> PIL Images
        tasks = []
        for vit_dp_rank in range(self.vit_dp):
            assigned_uuids = [uuids[i] for i in range(vit_dp_rank, len(uuids), self.vit_dp)]
            if assigned_uuids:
                for vit_tp_rank in range(self.vit_tp):
                    task = asyncio.create_task(self.model_rpcs[vit_dp_rank][vit_tp_rank].encode(assigned_uuids))
                    tasks.append(task)

        # rets = [self.model_rpcs[tp_rank].encode(images) for tp_rank in range(self.world_size)]
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
                    req = self.waiting_reqs.pop(0)
                    _, _, multimodal_params, _, _ = req
                    for img in multimodal_params.images:
                        if not self.cache_client.root.get_item_embed(img.uuid):
                            cur_batch_size += 1
                            uuids_need_infer.append(img.uuid)
                    if len(uuids_need_infer) > 0:
                        reqs_need_infer.append(req)
                    else:
                        self.send_to_router.send_pyobj(req)

                await self.infer_imgs(uuids_need_infer)
                for req in reqs_need_infer:
                    self.send_to_router.send_pyobj(req)

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 5:
                self.waiting_reqs.append(recv_req)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                group_req_id = abort_req.group_req_id
                await self.abort(group_req_id)
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
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    try:
        visualserver = VisualManager(args, router_port, visual_port, cache_port, model_rpc_ports)
        asyncio.run(visualserver.wait_to_model_ready())
    except Exception as e:
        import traceback

        err_str = "\n".join(traceback.format_exception(e))
        pipe_writer.send(err_str)
        visualserver.clean_up()
        raise
    pipe_writer.send("init ok")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(visualserver.loop_for_fwd())
    loop.run_until_complete(visualserver.loop_for_netio_req())
    return
