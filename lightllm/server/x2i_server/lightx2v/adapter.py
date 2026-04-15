import datetime
import inspect
import torch
import torch.distributed as dist
import zmq
import zmq.asyncio
import setproctitle
import asyncio
import os

from lightllm.server.core.objs import StartArgs
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.server.core.objs.x2i_params import X2IParams, X2IResponse, X2ICacheRelease, CfgNormType
from ..past_kv_cache_client import PastKVCacheClient

logger = init_logger(__name__)


class LightX2VServer:
    def __init__(self, args: StartArgs, rank: int, world_size: int):
        self.args = args
        self.rank = rank
        self.world_size = world_size

        # receive task from manager
        if self.rank == 0:
            context = zmq.asyncio.Context(2)
            self.task_socket = context.socket(zmq.PULL)
            self.task_socket.connect(f"{args.zmq_mode}127.0.0.1:{self.args.x2i_worker_task_port}")

            # send result back
            self.result_socket = context.socket(zmq.PUSH)
            self.result_socket.connect(f"{args.zmq_mode}127.0.0.1:{self.args.http_server_port_for_x2i}")

        self.past_kv_cache_client = PastKVCacheClient(only_create_meta_data=False, init_shm_data=False)
        torch.cuda.set_device(rank)
        self._init_pipeline()

        self.task_dist_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(days=30))

    def _init_pipeline(self):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(self.args.x2i_worker_nccl_port)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)

        from lightx2v import LightX2VPipeline

        self.pipe = LightX2VPipeline(
            model_path=self.args.model_dir,
            model_cls="neopp",
            support_tasks=["t2i", "i2i"],
        )
        self.pipe.create_generator(config_json=self.args.x2v_gen_model_config)
        self.pipe.modify_config({
            "load_kv_cache_in_pipeline_for_debug": False,
            "save_result_for_debug": False})

    async def run(self):
        while True:

            try:
                if self.rank == 0:
                    param: X2IParams = await self.task_socket.recv_pyobj()
                    dist.broadcast_object_list([param], src=0, group=self.task_dist_group)
                else:
                    params = [None]
                    dist.broadcast_object_list(params, src=0, group=self.task_dist_group)
                    param: X2IParams = params[0]

                assert param is not None, "Received None param in x2v worker, this should not happen."

                images = await self._process(param)

                if self.rank == 0:
                    await self.result_socket.send_pyobj(X2IResponse(request_id=param.request_id, images=images))

            except Exception as e:
                logger.error(f"Error processing request {param.request_id}: {str(e)}", exc_info=e)
                if self.rank == 0:
                    await self.result_socket.send_pyobj(X2IResponse(request_id=param.request_id, images=None))

    async def _process(self, param: X2IParams):
        is_t2i = param.past_kvcache_img.is_empty()

        self.pipe.runner.set_inference_params(
            index_offset_cond=param.past_kvcache.get_compressed_len(),
            index_offset_uncond=param.past_kvcache_text.get_compressed_len(),
            cfg_interval=param.cfg_interval,
            cfg_scale=param.guidance_scale,
            cfg_norm=CfgNormType(param.cfg_norm).as_str(),
            timestep_shift=param.timestep_shift,
        )
        past_kv_cache = self.past_kv_cache_client.get_kv_cache_for_x2i(
            param.past_kvcache.get_all(), param.past_kvcache.token_len)
        past_kv_cache_text = self.past_kv_cache_client.get_kv_cache_for_x2i(
            param.past_kvcache_text.get_all(), param.past_kvcache_text.token_len)
        past_kv_cache_img = None
        if not is_t2i:
            past_kv_cache_img = self.past_kv_cache_client.get_kv_cache_for_x2i(
                param.past_kvcache_img.get_all(), param.past_kvcache_img.token_len)

        dist.barrier()  # ensure all workers have got the kv cache before generation starts

        if self.rank == 0:
            # release
            await self.result_socket.send_pyobj(X2ICacheRelease(request_id=param.request_id))

        logger.info(f"{'t2i' if is_t2i else 'it2i'} generate images with: {param}")

        if is_t2i:
            self.pipe.runner.set_kvcache(
                past_kv_cache,
                past_kv_cache_text,
            )
        else:
            self.pipe.runner.set_kvcache_i2i(
                past_kv_cache,
                past_kv_cache_text,
                past_kv_cache_img,
            )
        image = self.pipe.generate(
            seed=param.seed + param.past_kvcache.img_len,
            save_result_path="",
            target_shape=[param.height, param.width],
        )

        return [image]


def start_x2v_process(args: StartArgs, rank: int, world_size: int, pipe_writer):

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::x2v_server_{rank}")
    start_parent_check_thread()

    try:
        x2v_server = LightX2VServer(args=args, rank=rank, world_size=world_size)
    except Exception as e:
        logger.exception(str(e), exc_info=e)
        raise e

    pipe_writer.send("init ok")

    def handle_exception(loop, context):
        logger.exception(f"X2VServer Caught exception: {str(context)}")

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    asyncio.set_event_loop(loop)

    loop.run_until_complete(x2v_server.run())

    return
