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
    if not torch.cuda.is_available():
        return
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
    # 该参数需要大于单卡最大batch size，且是8的倍数。该参数与显存占用直接相关，值越大，显存占用越大，如果出现显存不足，可以尝试调小该值
    return int(os.getenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK", 256))


def get_lightllm_gunicorn_time_out_seconds():
    return int(os.getenv("LIGHTLMM_GUNICORN_TIME_OUT", 180))


def get_lightllm_gunicorn_keep_alive():
    return int(os.getenv("LIGHTLMM_GUNICORN_KEEP_ALIVE", 10))


@lru_cache(maxsize=None)
def get_lightllm_websocket_max_message_size():
    """
    Get the maximum size of the WebSocket message.
    :return: Maximum size in bytes.
    """
    return int(os.getenv("LIGHTLLM_WEBSOCKET_MAX_SIZE", 16 * 1024 * 1024))


# get_redundancy_expert_ids and get_redundancy_expert_num are primarily
# used to obtain the IDs and number of redundant experts during inference.
# They depend on a configuration file specified by ep_redundancy_expert_config_path,
# which is a JSON formatted text file.
# The content format is as follows:
# {
#   "redundancy_expert_num": 1,  # Number of redundant experts per rank
#   "0": [0],                    # Key: layer_index (string),
#                                # Value: list of original expert IDs that are redundant for this layer
#   "1": [0],
#   "default": [0]               # Default list of redundant expert IDs if layer-specific entry is not found
# }


@lru_cache(maxsize=None)
def get_redundancy_expert_ids(layer_index: int):
    """
    Get the redundancy expert ids from the environment variable.
    :return: List of redundancy expert ids.
    """
    args = get_env_start_args()
    if args.ep_redundancy_expert_config_path is None:
        return []

    with open(args.ep_redundancy_expert_config_path, "r") as f:
        config = json.load(f)
    if str(layer_index) in config:
        return config[str(layer_index)]
    else:
        return config.get("default", [])


@lru_cache(maxsize=None)
def get_redundancy_expert_num():
    """
    Get the number of redundancy experts from the environment variable.
    :return: Number of redundancy experts.
    """
    args = get_env_start_args()
    if args.ep_redundancy_expert_config_path is None:
        return 0

    with open(args.ep_redundancy_expert_config_path, "r") as f:
        config = json.load(f)
    if "redundancy_expert_num" in config:
        return config["redundancy_expert_num"]
    else:
        return 0


@lru_cache(maxsize=None)
def get_redundancy_expert_update_interval():
    return int(os.getenv("LIGHTLLM_REDUNDANCY_EXPERT_UPDATE_INTERVAL", 30 * 60))


@lru_cache(maxsize=None)
def get_redundancy_expert_update_max_load_count():
    return int(os.getenv("LIGHTLLM_REDUNDANCY_EXPERT_UPDATE_MAX_LOAD_COUNT", 1))


@lru_cache(maxsize=None)
def get_kv_quant_calibration_warmup_count():
    # 服务启动后前warmup次推理不计入量化校准统计，该参数可以控制在一个更大的校准数据集的不同位置处开始校准。
    return int(os.getenv("LIGHTLLM_KV_QUANT_CALIBRARTION_WARMUP_COUNT", 0))


@lru_cache(maxsize=None)
def get_kv_quant_calibration_inference_count():
    # warmup后开始进行量化校准统计，推理次数达到inference_count后输出统计校准结果，通过该参数可以控制对量化校准数据的采集量。
    return int(os.getenv("LIGHTLLM_KV_QUANT_CALIBRARTION_INFERENCE_COUNT", 4000))


@lru_cache(maxsize=None)
def get_triton_autotune_level():
    return int(os.getenv("LIGHTLLM_TRITON_AUTOTUNE_LEVEL", 0))


g_model_init_done = False


def get_model_init_status():
    global g_model_init_done
    return g_model_init_done


def set_model_init_status(status: bool):
    global g_model_init_done
    g_model_init_done = status
    return g_model_init_done


def use_whisper_sdpa_attention() -> bool:
    """
    whisper重训后,使用特定的实现可以提升精度，用该函数控制使用的att实现。
    """
    return enable_env_vars("LIGHTLLM_USE_WHISPER_SDPA_ATTENTION")
