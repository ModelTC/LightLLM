import zmq
import asyncio
import uvloop
import inspect
import setproctitle
import pickle
import torch
import time
import multiprocessing as mp
import os
from typing import List
from lightllm.server.core.objs import StartArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.server.core.objs.x2i_params import X2IParams, X2IResponse, X2ICacheRelease, CfgNormType
from lightllm.utils.dist_utils import set_current_device_id
from lightllm.utils.start_utils import start_submodule_processes
from .past_kv_cache_client import PastKVCacheClient

logger = init_logger(__name__)


"""
manage a generation service,
1. start x2v pipelines
2. receive generation request from http_server.
3. call llm gen to obtain past key values
4. call x2v to generate images and pass the key values to it
5. return the generated images.
"""

class X2IManager:
    def __init__(
        self,
        args: StartArgs,
    ):
        context = zmq.Context(2)
        self.args = args

        # from http server
        self.zmq_recv_socket = context.socket(zmq.PULL)
        self.zmq_recv_socket.bind(f"{args.zmq_mode}127.0.0.1:{args.x2i_port}")

        # to http server
        self.send_to_httpserver = context.socket(zmq.PUSH)
        self.send_to_httpserver.connect(f"{args.zmq_mode}127.0.0.1:{args.http_server_port_for_x2i}")

        self.use_naive_x2i = args.x2i_use_naive_impl
        self.world_size = args.x2i_server_used_gpus

        if not self.use_naive_x2i and self.world_size > 1:
            # send to workers
            self.worker_pub = context.socket(zmq.PUSH)
            self.worker_pub.bind(f"{args.zmq_mode}127.0.0.1:{args.x2i_worker_task_port}")

        self.waiting_reqs: List[X2IParams] = []

        self.past_kv_cache_client = PastKVCacheClient(only_create_meta_data=False, init_shm_data=True)


    async def wait_to_model_ready(self):

        if self.world_size <= 1:
            if self.use_naive_x2i:
                from lightllm.server.x2i_server.naive.modeling_neo_chat import NEOX2I
                self.naive_x2i = NEOX2I(self.args.model_dir, torch.cuda.current_device())
            else:
                from lightx2v import LightX2VPipeline

                self.gen_pipe = LightX2VPipeline(
                    model_path=self.args.model_dir,
                    model_cls="neopp",
                    support_tasks=["t2i", "i2i"],
                )
                self.gen_pipe.create_generator(
                    config_json=self.args.x2v_gen_model_config,
                )
                self.gen_pipe.modify_config({"load_kv_cache_in_pipeline_for_debug": False, "save_result_for_debug": False})
        else:
            # distribted x2v
            from lightllm.server.x2i_server.lightx2v.adapter import start_x2v_process
            funcs = [start_x2v_process] * self.world_size
            args = [(self.args, rank, self.world_size) for rank in range(self.world_size)]
            start_submodule_processes(funcs, args)

    async def t2i_generate(self, past_kv_cache, past_kv_cache_text, param: X2IParams):
        if self.use_naive_x2i:
            images = self.naive_x2i.t2i(past_kv_cache, past_kv_cache_text, param)
            return images

        self.gen_pipe.runner.set_inference_params(
            index_offset_cond=param.past_kvcache.get_compressed_len(),
            index_offset_uncond=param.past_kvcache_text.get_compressed_len(),
            cfg_interval=param.cfg_interval,
            cfg_scale=param.guidance_scale,
            cfg_norm=CfgNormType(param.cfg_norm).as_str(),
            timestep_shift=param.timestep_shift,
        )
        self.gen_pipe.runner.set_kvcache(past_kv_cache, past_kv_cache_text)
        image = self.gen_pipe.generate(
            seed=param.seed + param.past_kvcache.img_len,
            save_result_path="",  # 返回base64，不需要指定路径了
            target_shape=[param.height, param.width],  # Height, Width
        )
        # images = self.naive_x2i.t2i(past_kv_cache, past_kv_cache_text, param)
        return [image]

    async def it2i_generate(self, past_kv_cache, past_kv_cache_text, past_kv_cache_img, param: X2IParams):
        if self.use_naive_x2i:
            images = self.naive_x2i.it2i(past_kv_cache, past_kv_cache_text, past_kv_cache_img, param)
            return images

        self.gen_pipe.runner.set_inference_params(
            index_offset_cond=param.past_kvcache.get_compressed_len(),
            index_offset_uncond=param.past_kvcache_text.get_compressed_len(),
            cfg_interval=param.cfg_interval,
            cfg_scale=param.guidance_scale,
            cfg_norm=CfgNormType(param.cfg_norm).as_str(),
            timestep_shift=param.timestep_shift,
        )
        self.gen_pipe.runner.set_kvcache_i2i(past_kv_cache, past_kv_cache_text, past_kv_cache_img)
        image = self.gen_pipe.generate(
            seed=param.seed + param.past_kvcache_img.img_len,
            save_result_path="",  # 返回base64，不需要指定路径了
            target_shape=[param.height, param.width],  # Height, Width
        )
        return [image]

    async def loop_for_fwd(self):
        while True:
            try:
                if len(self.waiting_reqs) == 0:
                    await asyncio.sleep(0.01)
                    continue

                x2i_param = self.waiting_reqs.pop(0)

                if not self.use_naive_x2i and self.world_size > 1:
                    # broadcast to workers
                    self.worker_pub.send_pyobj(x2i_param, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    past_kv_cache = self.past_kv_cache_client.get_kv_cache_for_x2i(
                        x2i_param.past_kvcache.get_all(), x2i_param.past_kvcache.token_len, self.use_naive_x2i
                    )

                    past_kv_cache_text = self.past_kv_cache_client.get_kv_cache_for_x2i(
                        x2i_param.past_kvcache_text.get_all(), x2i_param.past_kvcache_text.token_len, self.use_naive_x2i
                    )
                    is_t2i = x2i_param.past_kvcache_img.is_empty()

                    past_kv_cache_img = None
                    if not is_t2i:  # t2i
                        past_kv_cache_img = self.past_kv_cache_client.get_kv_cache_for_x2i(
                            x2i_param.past_kvcache_img.get_all(), x2i_param.past_kvcache_img.token_len, self.use_naive_x2i
                        )

                    # release
                    self.send_to_httpserver.send_pyobj(
                        X2ICacheRelease(request_id=x2i_param.request_id), protocol=pickle.HIGHEST_PROTOCOL
                    )

                    images = []
                    logger.info(f"{'t2i' if is_t2i else 'it2i'} generate images with: {x2i_param}")
                    start_t = time.time()
                    if is_t2i:
                        images = await self.t2i_generate(past_kv_cache, past_kv_cache_text, x2i_param)
                    else:
                        images = await self.it2i_generate(past_kv_cache, past_kv_cache_text, past_kv_cache_img, x2i_param)
                    logger.info(f"generate {len(images)} images done, cost {time.time() - start_t:.2f}s")

                    self.send_to_httpserver.send_pyobj(
                        X2IResponse(request_id=x2i_param.request_id, images=images), protocol=pickle.HIGHEST_PROTOCOL
                    )

            except Exception as e:
                self.send_to_httpserver.send_pyobj(
                    X2IResponse(request_id=x2i_param.request_id, images=None), protocol=pickle.HIGHEST_PROTOCOL
                )
                logger.error(e, exc_info=e)

    async def loop_for_netio_req(self):
        while True:
            try:
                recv_req: X2IParams = self.zmq_recv_socket.recv_pyobj(zmq.NOBLOCK)
                self.waiting_reqs.append(recv_req)

            except zmq.ZMQError:
                await asyncio.sleep(0.1)

            await asyncio.sleep(0.01)

    def clean_up(self):
        pass


def setup_devices(args: StartArgs):
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    logger.info(f"current devices: {devices} {torch.cuda.device_count()}")
    if not devices:
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [int(x.strip()) for x in devices.split(",") if x.strip()]

    llm_need_gpus = args.tp * args.dp
    # llm_need_gpus = 0
    x2i_need_gpus = args.x2i_server_used_gpus
    if len(devices) < llm_need_gpus + x2i_need_gpus:
        raise ValueError(f"devices {devices} not enough, need {llm_need_gpus} and {x2i_need_gpus}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices[llm_need_gpus : llm_need_gpus + x2i_need_gpus]))

    logger.info(
        f"setup devices for x2i server: {os.environ['CUDA_VISIBLE_DEVICES']}, "
        f"{torch.cuda.device_count()} {torch.cuda.current_device()}"
    )


def start_x2i_process(args, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::x2i_server")
    start_parent_check_thread()
    set_current_device_id(torch.cuda.current_device())
    try:
        x2iserver = X2IManager(
            args=args,
        )
        asyncio.run(x2iserver.wait_to_model_ready())
    except Exception as e:
        logger.exception(str(e))
        x2iserver.clean_up()
        raise e

    pipe_writer.send("init ok")

    def handle_exception(loop, context):
        logger.exception(f"X2IServer Caught exception: {str(context)}")

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    asyncio.set_event_loop(loop)
    loop.create_task(x2iserver.loop_for_fwd())
    loop.run_until_complete(x2iserver.loop_for_netio_req())
    return
