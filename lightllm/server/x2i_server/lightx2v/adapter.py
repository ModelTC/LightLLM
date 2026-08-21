import datetime
import inspect
import torch
import torch.distributed as dist
import zmq
import zmq.asyncio
import setproctitle
import asyncio
import os
import time
from PIL import Image
import io
import base64

from lightllm.server.core.objs import StartArgs
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.server.core.objs.x2i_params import X2IParams, X2IResponse, X2ICacheRelease, CfgNormType, OutputFormatType
from ..past_kv_cache_client import PastKVCacheClient
from ..rng_state_cache import RngStateCache

logger = init_logger(__name__)


def process_images_after_vae_decoder(self):
    r_height, r_width = -1, -1
    if hasattr(self, "target_shape") and self.target_shape is not None:
        r_height, r_width = self.target_shape

    image = self._denorm(self.scheduler.image_prediction.float())
    image = (image.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu()
    # CHW -> HWC
    image = image[0].permute(1, 2, 0).contiguous().numpy()
    pil_image = Image.fromarray(image)

     # Resize if requested
    if r_height > 0 and r_width > 0:
        if (
            pil_image.height != r_height
            or pil_image.width != r_width
        ):
            logger.info(f"resize image to user request size ({r_width}x{r_height}) from "
                        f"x2i server generated image size ({pil_image.width}x{pil_image.height})")
            pil_image = pil_image.resize(
                (r_width, r_height),
                Image.Resampling.LANCZOS,
            )

    buffer = io.BytesIO()

    output_format = self.output_format.lower() or "jpeg"
    if output_format in ("jpg", "jpeg"):
        pil_image.save(buffer, format="JPEG", quality=90)
    elif output_format == "png":
        pil_image.save(buffer, format="PNG", compress_level=1)
    elif output_format == "webp":
        pil_image.save(buffer, format="WEBP", quality=90, method=0)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def set_resolution(self, r_width, r_height):
    self.target_shape = (r_height, r_width)


class LightX2VServer:
    def __init__(self, args: StartArgs, rank: int, world_size: int, nccl_port: int):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.nccl_port = nccl_port

        # receive task from manager
        if self.rank == 0:
            context = zmq.asyncio.Context(2)
            self.task_socket = context.socket(zmq.PULL)
            self.task_socket.connect(f"{args.zmq_mode}127.0.0.1:{self.args.x2i_worker_task_port}")

            # send result back
            self.result_socket = context.socket(zmq.PUSH)
            self.result_socket.connect(f"{args.zmq_mode}127.0.0.1:{self.args.http_server_port_for_x2i}")

        self.past_kv_cache_client = PastKVCacheClient(only_create_meta_data=False, init_shm_data=False)

        logger.info(f"set device for x2v server {rank}/{world_size} {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        torch.cuda.set_device(self.rank)

        self._init_pipeline()

        if self.world_size > 1:
            self.task_dist_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(days=30))

        self.enable_cfg = args.x2i_enable_cfg

        # Per-chat-session RNG snapshot, see lightllm/server/x2i_server/rng_state_cache.py.
        # 各 worker 各自维护一份；由于本进程是单 stream 的串行 _process，且任务通过 broadcast
        # 同步，因此各 rank 之间的 RNG 演进保持一致。
        self.rng_state_cache = RngStateCache()

    def _init_pipeline(self):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(self.nccl_port)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)

        from lightx2v import LightX2VPipeline
        from lightx2v.models.runners.neopp.neopp_runner import NeoppRunner
        # monkey patch
        NeoppRunner.process_images_after_vae_decoder = process_images_after_vae_decoder
        NeoppRunner.set_resolution = set_resolution

        self.pipe = LightX2VPipeline(
            model_path=self.args.model_dir,
            model_cls="neopp",
            support_tasks=["t2i", "i2i"],
        )
        self.pipe.create_generator(config_json=self.args.x2v_gen_model_config)
        self.pipe.modify_config({"load_kv_cache_in_pipeline_for_debug": False, "save_result_for_debug": False})

    async def run(self):
        while True:

            try:
                if self.rank == 0:
                    param: X2IParams = await self.task_socket.recv_pyobj()
                    if self.world_size > 1:
                        dist.broadcast_object_list([param], src=0, group=self.task_dist_group)
                else:
                    params = [None]
                    dist.broadcast_object_list(params, src=0, group=self.task_dist_group)
                    param: X2IParams = params[0]

                assert param is not None, "Received None param in x2v worker, this should not happen."

                start = time.time()
                images = await self._process(param)
                logger.info(
                    f"[{self.rank}/{self.world_size}/{self.nccl_port}] generate images cost {time.time() - start} seconds"
                )

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
            index_offset_uncond=param.past_kvcache_text.get_compressed_len() if self.enable_cfg else None,
            cfg_interval=param.cfg_interval,
            cfg_scale=param.guidance_scale,
            cfg_norm=CfgNormType(param.cfg_norm).as_str(),
            timestep_shift=param.timestep_shift,
            output_format=OutputFormatType(param.output_format).as_str(),
        )

        past_kv_cache = self.past_kv_cache_client.get_kv_cache_for_x2i(
            param.past_kvcache.get_all(), param.past_kvcache.token_len
        )

        past_kv_cache_text = (
            self.past_kv_cache_client.get_kv_cache_for_x2i(
                param.past_kvcache_text.get_all(), param.past_kvcache_text.token_len
            )
            if self.enable_cfg
            else None
        )

        past_kv_cache_img = (
            self.past_kv_cache_client.get_kv_cache_for_x2i(
                param.past_kvcache_img.get_all(), param.past_kvcache_img.token_len
            )
            if not is_t2i
            else None
        )

        if self.world_size > 1:
            dist.barrier()  # ensure all workers have got the kv cache before generation starts

        if self.rank == 0:
            # release
            await self.result_socket.send_pyobj(X2ICacheRelease(request_id=param.request_id))

        logger.info(
            f"[{self.rank}/{self.world_size}/{self.nccl_port}] {'t2i' if is_t2i else 'it2i'} generate images with: {param}"
        )

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

        session_id = param.session_id
        if param.first_image:
            # 本 session 第一张图：用传入的 seed 初始化全局 RNG
            seed = param.seed
        else:
            # 后续图：恢复本 session 上次结束时的 RNG，避免被其他 session 的 seed_all 污染
            restored = self.rng_state_cache.restore(session_id)
            seed = None
            if not restored:
                logger.warning(
                    f"session {session_id} rng state miss (maybe expired or first call after restart), "
                    f"fallback to current global rng"
                )
        logger.info(f"seed: {seed} param.seed: {param.seed} first_image: {param.first_image} session_id: {session_id}")
        self.pipe.runner.set_resolution(param.r_width, param.r_height)
        image = self.pipe.generate(
            seed=seed,
            save_result_path="",
            target_shape=[param.height, param.width],
        )

        # 保存当前 RNG state，供同一 session 下一张图使用
        self.rng_state_cache.save(session_id)

        return [image]


def start_x2v_process(args: StartArgs, rank: int, world_size: int, nccl_port: int, pipe_writer):

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::x2v_server_{rank}")
    start_parent_check_thread()

    try:
        x2v_server = LightX2VServer(args=args, rank=rank, world_size=world_size, nccl_port=nccl_port)
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
