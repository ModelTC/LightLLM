import os
import json
import torch
import uuid
from easydict import EasyDict
from functools import lru_cache
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


def set_unique_server_name(args):
    node_uuid = uuid.uuid4().hex[0:16]

    if args.run_mode == "pd_master":
        os.environ["LIGHTLLM_UNIQUE_SERVICE_NAME_ID"] = str(node_uuid) + "_pd_master"
    else:
        os.environ["LIGHTLLM_UNIQUE_SERVICE_NAME_ID"] = str(node_uuid) + "_" + str(args.node_rank)
    return


@lru_cache(maxsize=None)
def get_unique_server_name():
    service_uni_name = os.getenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID")
    return service_uni_name


def set_cuda_arch(args):
    if not torch.cuda.is_available():
        return
    return


def set_env_start_args(args):
    set_cuda_arch(args)
    if not isinstance(args, dict):
        args = vars(args)
    os.environ["LIGHTLLM_START_ARGS"] = json.dumps(args)
    if args["enable_ep_moe"]:
        if args["run_mode"] == "prefill":
            decode_capacity = args["running_max_req_size"] * (args["mtp_step"] + 1)
            decode_capacity = ((decode_capacity + 7) // 8) * 8
            configured_decode_capacity = int(os.getenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE", decode_capacity))
            if configured_decode_capacity != decode_capacity:
                logger.warning(
                    "NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE=%d differs from the automatically derived value %d.",
                    configured_decode_capacity,
                    decode_capacity,
                )
            decode_capacity = max(configured_decode_capacity, decode_capacity)
        else:
            decode_capacity = get_deepep_num_max_dispatch_tokens_per_rank_decode()
        min_qp_depth = 2 * (decode_capacity + 1)
        derived_qp_depth = 1 << (min_qp_depth - 1).bit_length()
        configured_qp_depth = int(os.getenv("NVSHMEM_QP_DEPTH", derived_qp_depth))
        if configured_qp_depth < derived_qp_depth:
            logger.warning(
                "NVSHMEM_QP_DEPTH=%d is below the required minimum; using %d instead.",
                configured_qp_depth,
                derived_qp_depth,
            )
        os.environ["NVSHMEM_QP_DEPTH"] = str(max(configured_qp_depth, derived_qp_depth))
    return


@lru_cache(maxsize=None)
def get_env_start_args():
    from lightllm.server.core.objs.start_args_type import StartArgs

    start_args: StartArgs = json.loads(os.environ["LIGHTLLM_START_ARGS"])
    start_args: StartArgs = EasyDict(start_args)
    return start_args


@lru_cache(maxsize=None)
def get_llm_data_type() -> torch.dtype:
    data_type: str = get_env_start_args().data_type
    if data_type in ["fp16", "float16"]:
        data_type = torch.float16
    elif data_type in ["bf16", "bfloat16"]:
        data_type = torch.bfloat16
    elif data_type in ["fp32", "float32"]:
        data_type = torch.float32
    else:
        raise ValueError(f"Unsupported datatype {data_type}!")
    return data_type


@lru_cache(maxsize=None)
def enable_env_vars(args):
    return os.getenv(args, "False").upper() in ["ON", "TRUE", "1"]


@lru_cache(maxsize=None)
def get_deepep_num_max_dispatch_tokens_per_rank_prefill():
    # 该参数需要大于单卡最大batch size，且是8的倍数。该参数与显存占用直接相关，值越大，显存占用越大。
    # 如果未显式配置，则默认至少覆盖当前进程的 `batch_max_tokens`，避免 DeepEP V2 在 autotune
    # warmup 或大 prefill batch 时因为 buffer 上界过小而报错。
    configured = os.getenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK_PREFILL", None)
    if configured is not None:
        return int(configured)

    batch_max_tokens = get_env_start_args().batch_max_tokens or 256
    return ((int(batch_max_tokens) + 7) // 8) * 8


@lru_cache(maxsize=None)
def get_deepep_num_max_dispatch_tokens_per_rank_decode():
    args = get_env_start_args()
    per_dp_running_max_req_size = getattr(args, "per_dp_running_max_req_size", None)
    if per_dp_running_max_req_size is None:
        per_dp_running_max_req_size = args.running_max_req_size

    graph_max_batch_size = 0
    if not args.disable_cudagraph:
        graph_max_batch_size = args.graph_max_batch_size
        if args.enable_decode_microbatch_overlap:
            graph_max_batch_size //= 2

    required = max(per_dp_running_max_req_size, graph_max_batch_size) * (args.mtp_step + 1)
    required = ((required + 7) // 8) * 8
    configured = int(os.getenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE", required))
    if configured != required:
        logger.warning(
            "NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE=%d differs from the automatically derived value %d.",
            configured,
            required,
        )
    return max(configured, required)


@lru_cache(maxsize=None)
def get_lightllm_websocket_max_message_size():
    """
    Get the maximum size of the WebSocket message.
    :return: Maximum size in bytes.
    """
    return int(os.getenv("LIGHTLLM_WEBSOCKET_MAX_SIZE", 128 * 1024 * 1024))


@lru_cache(maxsize=None)
def get_prefill_eplb_step_interval():
    """Return the number of prefill forwards between EPLB attempts."""
    interval = int(os.getenv("LIGHTLLM_PREFILL_EPLB_STEP_INTERVAL", 20))
    if interval <= 0:
        raise ValueError("LIGHTLLM_PREFILL_EPLB_STEP_INTERVAL must be greater than 0")
    return interval


@lru_cache(maxsize=None)
def get_eplb_rebalance_gain_threshold() -> float:
    """Return the EPLB gain threshold: estimated critical-load reduction ratio; 0.05 means 5%."""
    env_name = "LIGHTLLM_EPLB_REBALANCE_GAIN_THRESHOLD"
    raw_value = os.getenv(env_name, "0.05")
    value = float(raw_value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{env_name} must be a ratio between 0.0 and 1.0, got {raw_value!r}")
    return value


@lru_cache(maxsize=None)
def get_eplb_placement_stickiness() -> float:
    """Return the EPLB placement stickiness: a keep-bonus, as a fraction of the mean per-layer expert load."""
    env_name = "LIGHTLLM_EPLB_PLACEMENT_STICKINESS"
    raw_value = os.getenv(env_name, "0.1")
    value = float(raw_value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{env_name} must be a ratio between 0.0 and 1.0, got {raw_value!r}")
    return value


def get_triton_autotune_level():
    return int(os.getenv("LIGHTLLM_TRITON_AUTOTUNE_LEVEL", 0))


@lru_cache(maxsize=None)
def enable_full_att_decode_tune() -> bool:
    """
    Whether to run FA3 full-attention decode num_splits warmup/autotune at model init.

    Env: ENABLE_FULL_ATT_DECODE_TUNE
      - ON / TRUE / 1: enable
      - otherwise (default False): skip this operator-specific tuning
    """
    return enable_env_vars("ENABLE_FULL_ATT_DECODE_TUNE")


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


@lru_cache(maxsize=None)
def enable_radix_tree_timer_merge() -> bool:
    """
    使能定期合并 radix tree的叶节点, 防止插入查询性能下降。
    """
    return enable_env_vars("LIGHTLLM_RADIX_TREE_MERGE_ENABLE")


@lru_cache(maxsize=None)
def get_radix_tree_merge_update_delta() -> int:
    return int(os.getenv("LIGHTLLM_RADIX_TREE_MERGE_DELTA", 6000))


@lru_cache(maxsize=None)
def get_diverse_max_batch_shared_group_size() -> int:
    return int(os.getenv("LIGHTLLM_MAX_BATCH_SHARED_GROUP_SIZE", 4))


@lru_cache(maxsize=None)
def enable_diverse_mode_gqa_decode_fast_kernel() -> bool:
    return get_env_start_args().diverse_mode and "int8kv" == get_env_start_args().llm_kv_type


@lru_cache(maxsize=None)
def get_disk_cache_prompt_limit_length():
    return int(os.getenv("LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH", 2048))


@lru_cache(maxsize=None)
def get_dsv4_cpu_cache_max_pages_per_task() -> int:
    return int(os.getenv("LIGHTLLM_DSV4_CPU_CACHE_MAX_PAGES_PER_TASK", 4))


@lru_cache(maxsize=None)
def enable_huge_page():
    """
    大页模式：启动后可大幅缩短cpu kv cache加载时间
    "sudo sed -i 's/^GRUB_CMDLINE_LINUX=\"/& default_hugepagesz=1G \
        hugepagesz=1G hugepages={需要启用的大页容量}/' /etc/default/grub"
    "sudo update-grub"
    "sudo reboot"
    """
    return enable_env_vars("LIGHTLLM_HUGE_PAGE_ENABLE")


@lru_cache(maxsize=None)
def enable_cpu_cache_numa_interleave() -> bool:
    """是否启用 CPU KV cache 共享内存的 NUMA 交错分配策略。"""
    return enable_env_vars("LIGHTLLM_ENABLE_NUMA_INTERLEAVE")


@lru_cache(maxsize=None)
def get_added_mtp_kv_layer_num() -> int:
    # mtp 模式下需要在mem manger上扩展draft model使用的layer
    added_mtp_layer_num = 0
    if get_env_start_args().mtp_mode == "eagle_with_att":
        added_mtp_layer_num += 1
    elif get_env_start_args().mtp_mode == "vanilla_with_att":
        added_mtp_layer_num += get_env_start_args().mtp_step

    return added_mtp_layer_num


@lru_cache(maxsize=None)
def get_pd_split_max_new_tokens() -> int:
    return int(os.getenv("LIGHTLLM_PD_SPLIT_MAX_NEW_TOKENS", 2048))


@lru_cache(maxsize=None)
def get_lightllm_url_pool_maxsize() -> int:
    return int(os.getenv("LIGHTLLM_URL_POOL_MAXSIZE", 512))
