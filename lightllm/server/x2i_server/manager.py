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
from typing import List, Any
from lightllm.server.core.objs import StartArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.server.core.objs.x2i_params import X2IParams, X2IResponse, X2ICacheRelease, CfgNormType
from lightllm.utils.dist_utils import set_current_device_id
from lightllm.utils.start_utils import start_submodule_processes
from lightllm.utils.net_utils import alloc_can_use_network_port
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.platform import get_backend

from .past_kv_cache_client import PastKVCacheClient
from .rng_state_cache import RngStateCache

logger = init_logger(__name__)


def get_visible_device_env_name(args: StartArgs) -> str:
    """Return the accelerator visibility variable used by the selected platform."""
    if args.hardware_platform == "ascend":
        return "ASCEND_RT_VISIBLE_DEVICES"
    if args.hardware_platform == "musa":
        return "MUSA_VISIBLE_DEVICES"
    return "CUDA_VISIBLE_DEVICES"


def _get_visible_devices(args: StartArgs) -> List[str]:
    env_name = get_visible_device_env_name(args)
    value = os.environ.get(env_name, "").strip()
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(index) for index in range(get_backend().runtime.device_count())]


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
        self.x2i_server_used_gpus = args.x2i_server_used_gpus

        if not self.use_naive_x2i:
            # send to workers
            self.worker_pub = context.socket(zmq.PUSH)
            self.worker_pub.bind(f"{args.zmq_mode}127.0.0.1:{args.x2i_worker_task_port}")

        self.waiting_reqs: List[X2IParams] = []

        self.past_kv_cache_client = PastKVCacheClient(only_create_meta_data=False, init_shm_data=True)

        self.enable_cfg = args.x2i_enable_cfg

        # Per-chat-session RNG snapshot, so concurrent sessions don't clobber each other's
        # global torch / cuda RNG state between successive image generations.
        self.rng_state_cache = RngStateCache()

    async def wait_to_model_ready(self):

        if self.use_naive_x2i:
            from lightllm.server.x2i_server.naive.modeling_neo_chat import NEOX2I

            self.naive_x2i = NEOX2I(self.args.model_dir, get_backend().runtime.current_device())

        else:
            # x2v server use separate processes
            from lightllm.server.x2i_server.lightx2v.adapter import start_x2v_process

            x2v_world_size = get_x2v_world_size()

            assert (
                self.x2i_server_used_gpus >= x2v_world_size
            ), "x2i_server_used_gpus must be greater than x2v_world_size"

            if self.x2i_server_used_gpus < x2v_world_size or self.x2i_server_used_gpus % x2v_world_size != 0:
                logger.warning(
                    f"x2i_server_used_gpus {self.x2i_server_used_gpus} is not divisible by x2v_world_size {x2v_world_size}"
                )

            x2v_dp_size = self.x2i_server_used_gpus // x2v_world_size

            x2v_dp_dist_ports = alloc_can_use_network_port(
                num=x2v_dp_size,
                from_port_num=15000,
            )

            visible_devices = _get_visible_devices(self.args)
            visible_device_env = get_visible_device_env_name(self.args)

            logger.info(
                f"x2v_dp_dist_ports: {x2v_dp_dist_ports}, "
                f"x2v_world_size: {x2v_world_size}, x2v_dp_size: {x2v_dp_size}"
            )

            funcs = [start_x2v_process] * x2v_dp_size * x2v_world_size
            args = [
                (self.args, rank, x2v_world_size, x2v_dp_dist_ports[dp_rank])
                for dp_rank in range(x2v_dp_size)
                for rank in range(x2v_world_size)
            ]

            envs = [
                {
                    visible_device_env: ",".join(
                        visible_devices[dp_rank * x2v_world_size : (dp_rank + 1) * x2v_world_size]
                    )
                }
                for dp_rank in range(x2v_dp_size)
                for _ in range(x2v_world_size)
            ]

            start_submodule_processes(funcs, args, envs)

    async def t2i_generate(self, past_kv_cache, past_kv_cache_text, param: X2IParams):
        assert self.use_naive_x2i, "t2i is not supported for non naive x2i"
        images = self.naive_x2i.t2i(past_kv_cache, past_kv_cache_text, param)
        return images

    async def it2i_generate(self, past_kv_cache, past_kv_cache_text, past_kv_cache_img, param: X2IParams):
        assert self.use_naive_x2i, "it2i is not supported for non naive x2i"
        images = self.naive_x2i.it2i(past_kv_cache, past_kv_cache_text, past_kv_cache_img, param)
        return images

    async def loop_for_fwd(self):
        while True:
            try:
                if len(self.waiting_reqs) == 0:
                    await asyncio.sleep(0.01)
                    continue

                x2i_param = self.waiting_reqs.pop(0)

                if not self.use_naive_x2i:
                    # broadcast to workers
                    self.worker_pub.send_pyobj(x2i_param, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    past_kv_cache = self.past_kv_cache_client.get_kv_cache_for_x2i(
                        x2i_param.past_kvcache.get_all(), x2i_param.past_kvcache.token_len, self.use_naive_x2i
                    )

                    past_kv_cache_text = (
                        self.past_kv_cache_client.get_kv_cache_for_x2i(
                            x2i_param.past_kvcache_text.get_all(),
                            x2i_param.past_kvcache_text.token_len,
                            self.use_naive_x2i,
                        )
                        if self.enable_cfg
                        else None
                    )

                    is_t2i = x2i_param.past_kvcache_img.is_empty()

                    past_kv_cache_img = None
                    if not is_t2i:  # t2i
                        past_kv_cache_img = self.past_kv_cache_client.get_kv_cache_for_x2i(
                            x2i_param.past_kvcache_img.get_all(),
                            x2i_param.past_kvcache_img.token_len,
                            self.use_naive_x2i,
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
                        images = await self.it2i_generate(
                            past_kv_cache, past_kv_cache_text, past_kv_cache_img, x2i_param
                        )
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


class X2VConfig:
    _data = None

    @staticmethod
    def get(args: StartArgs = None):
        if X2VConfig._data is None:
            args = args or get_env_start_args()
            if not hasattr(args, "x2v_gen_model_config") or not args.x2v_gen_model_config:
                return None
            import json
            with open(args.x2v_gen_model_config, "r") as f:
                X2VConfig._data = json.load(f)
            logger.info(f"lightllm get x2v config: {X2VConfig._data}")
        return X2VConfig._data

    @staticmethod
    def get_value(key: str, default: Any = None):
        conf = X2VConfig.get()
        if not conf:
            return default
        return conf.get(key, default)


def get_enable_cfg() -> bool:
    return X2VConfig.get_value("enable_cfg", True)


def get_x2v_world_size() -> int:
    cfg_p_size = X2VConfig.get_value("parallel", {}).get("cfg_p_size", 1)
    seq_p_size = X2VConfig.get_value("parallel", {}).get("seq_p_size", 1)
    return cfg_p_size * seq_p_size


def get_default_image_config() -> dict:
    default_config = {}
    for key in ["timestep_shift", "cfg_interval", "cfg_norm"]:
        value = X2VConfig.get_value(key)
        if value is not None:
            default_config[key] = value
    cfg_scale = X2VConfig.get_value("cfg_scale")
    if cfg_scale is not None:
        default_config["guidance_scale"] = cfg_scale
    return default_config


def get_min_max_pixels() -> tuple[int, int]:
    min_pixels = X2VConfig.get_value("min_pixels", 2048 * 2048)
    max_pixels = X2VConfig.get_value("max_pixels", 4096 * 4096)
    return min_pixels, max_pixels


def get_input_min_max_pixels(input_image_num: int) -> tuple[int, int]:
    min_pixels = X2VConfig.get_value("input_min_pixels", 2048 * 2048)
    max_pixels = X2VConfig.get_value("input_max_pixels", 2048 * 2048)
    total_pixels = X2VConfig.get_value("input_total_pixels", 5 * 2048 * 2048)

    max_pixels = min(max_pixels, total_pixels // max(1, input_image_num))
    min_pixels = min(min_pixels, max_pixels)

    return min_pixels, max_pixels


def setup_devices(args: StartArgs):
    runtime = get_backend().runtime
    devices = _get_visible_devices(args)
    logger.info(
        f"current {get_visible_device_env_name(args)} devices: {devices}; "
        f"runtime device count: {runtime.device_count()}"
    )

    llm_need_gpus = 0 if args.x2i_server_deploy_mode == "colocate" else args.tp * args.dp
    x2i_need_gpus = args.x2i_server_used_gpus
    if len(devices) < llm_need_gpus + x2i_need_gpus:
        raise ValueError(f"devices {devices} not enough, need {llm_need_gpus} and {x2i_need_gpus}")

    visible_devices = ",".join(devices[llm_need_gpus : llm_need_gpus + x2i_need_gpus])
    logger.info(f"setup devices for x2i server: {visible_devices}")
    return visible_devices


def start_x2i_process(args, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::x2i_server")
    start_parent_check_thread()
    runtime = get_backend().runtime
    set_current_device_id(runtime.current_device())
    x2iserver = None
    try:
        x2iserver = X2IManager(
            args=args,
        )
        asyncio.run(x2iserver.wait_to_model_ready())
    except Exception as e:
        logger.exception(str(e))
        if x2iserver is not None:
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
