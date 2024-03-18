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
        client_port,
        infer_batch_size=4,
    ):
        context = zmq.asyncio.Context(2)
        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{router_port}")

        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{visual_port}")
        self.cache_client = rpyc.connect("localhost", client_port)
        self.waiting_reqs = []
        self.model_weightdir = args.model_dir
        self.tp_world_size = args.tp
        self.world_size = 1
        self.infer_batch_size = infer_batch_size
        self.trust_remote_code = args.trust_remote_code
        self.args = args

    async def wait_to_model_ready(self):

        self.model_rpcs: List[VisualModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):  # async init model process
            kvargs = {
                "weight_dir": self.model_weightdir,
                "trust_remote_code": self.trust_remote_code,
                "rank_id": rank_id,
            }
            init_model_ret.append(self.model_rpcs[rank_id].init_model(kvargs))
        await asyncio.gather(*init_model_ret)
        return

    async def abort(self, request_id):
        abort_req = AbortReq(req_id=request_id)
        self.send_to_router.send_pyobj(abort_req)
        return

    async def infer_imgs(self, uuids):
        if len(uuids) == 0:
            return
        # uuids -> PIL Images
        images = []
        for uid in uuids:
            image_data = read_shm(get_shm_name_data(uid))
            images.append(Image.open(BytesIO(image_data)))
            # print(" + got pil image:", images[-1].size, images[-1].mode)
        rets = [self.model_rpcs[tp_rank].encode(images) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            img_embed = obtain(ans[0])
        else:
            img_embed = ans[0]
        torch.cuda.synchronize()
        # b = time.time()
        for i in range(len(uuids)):
            # print(" + set_item_embed:", uuids[i], img_embed[i].shape)
            if not self.cache_client.root.get_item_embed(uuids[i]):
                cur_embed_bytes = tensor2bytes(img_embed[i])
                create_shm(get_shm_name_embed(uuids[i]), cur_embed_bytes)
                self.cache_client.root.set_item_embed(uuids[i])
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
                    _, _, multimodal_params, _ = req
                    for img in multimodal_params.images:
                        if not self.cache_client.root.get_item_embed(img.uuid):
                            cur_batch_size += 1
                            uuids_need_infer.append(img.uuid)
                    if len(uuids_need_infer) > 0:
                        reqs_need_infer.append(req)
                    else:
                        # print(" + no need need infer, send to router...")
                        self.send_to_router.send_pyobj(req)

                await self.infer_imgs(uuids_need_infer)
                for req in reqs_need_infer:
                    # print(" + after infer_imgs, send to router...")
                    self.send_to_router.send_pyobj(req)

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 4:
                self.waiting_reqs.append(recv_req)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                await self.abort(request_id)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return


def start_visual_process(args, router_port, visual_port, client_port, pipe_writer):
    try:
        visualserver = VisualManager(args, router_port, visual_port, client_port)
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
