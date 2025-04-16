import os
import json
import torch
from easydict import EasyDict
from functools import lru_cache
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


def set_unique_server_name(args):
    if args.run_mode == "pd_master":
        os.environ["LIGHTLLM_UNIQUE_SERVICE_NAME_ID"] = str(args.port) + "_pd_master"
    else:
        os.environ["LIGHTLLM_UNIQUE_SERVICE_NAME_ID"] = str(args.nccl_port) + "_" + str(args.node_rank)
    return


@lru_cache(maxsize=None)
def get_unique_server_name():
    service_uni_name = os.getenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID")
    return service_uni_name


def set_cuda_arch(args):
    if args.enable_flashinfer_prefill or args.enable_flashinfer_decode:
        capability = torch.cuda.get_device_capability()
        arch = f"{capability[0]}.{capability[1]}"
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{arch}{'+PTX' if arch == '9.0' else ''}"


def set_env_start_args(args):
    set_cuda_arch(args)
    if not isinstance(args, dict):
        args = vars(args)
    os.environ["LIGHTLLM_START_ARGS"] = json.dumps(args)
    return


@lru_cache(maxsize=None)
def get_env_start_args():
    from lightllm.server.core.objs.start_args_type import StartArgs

    start_args: StartArgs = json.loads(os.environ["LIGHTLLM_START_ARGS"])
    start_args: StartArgs = EasyDict(start_args)
    return start_args


@lru_cache(maxsize=None)
def enable_env_vars(args):
    return os.getenv(args, "False").upper() in ["ON", "TRUE", "1"]


@lru_cache(maxsize=None)
def get_deepep_num_max_dispatch_tokens_per_rank():
    return int(os.getenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK", 256))


def get_lightllm_gunicorn_time_out_seconds():
    return int(os.getenv("LIGHTLMM_GUNICORN_TIME_OUT", 180))


@lru_cache(maxsize=None)
def get_lightllm_websocket_max_message_size():
    """
    Get the maximum size of the WebSocket message.
    :return: Maximum size in bytes.
    """
    return int(os.getenv("LIGHTLLM_WEBSOCKET_MAX_SIZE", 16 * 1024 * 1024))
