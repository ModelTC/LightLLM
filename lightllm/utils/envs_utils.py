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
    # 在单卡最大 prefill batch 之外额外保留 128 个 token，并向上对齐到 8，
    # 避免 autotune warmup 或调度边界波动使实际输入超过 DeepEP buffer 上限。
    batch_max_tokens = get_env_start_args().batch_max_tokens or 256
    capacity = ((int(batch_max_tokens) + 128 + 7) // 8) * 8
    logger.info(
        "DeepEP prefill buffer capacity: batch_max_tokens=%s, safety_margin=128, capacity=%s",
        batch_max_tokens,
        capacity,
    )
    return capacity


@lru_cache(maxsize=None)
def get_deepep_num_max_dispatch_tokens_per_rank_decode():
    # 每个请求最多产生 mtp_step + 1 个 verify token；额外保留 12 个 token
    # 处理调度和 CUDA Graph 的边界波动，最后向上对齐到 DeepEP 要求的 8。
    args = get_env_start_args()
    required_tokens = int(args.running_max_req_size) * (int(args.mtp_step) + 1)
    capacity_with_margin = required_tokens + 12
    capacity = ((capacity_with_margin + 7) // 8) * 8
    logger.info(
        "DeepEP decode buffer capacity: running_max_req_size=%s, mtp_step=%s, "
        "required_tokens=%s, safety_margin=12, capacity=%s",
        args.running_max_req_size,
        args.mtp_step,
        required_tokens,
        capacity,
    )
    return capacity


@lru_cache(maxsize=None)
def get_deepep_num_max_dispatch_tokens_per_rank() -> int:
    """返回同时覆盖 prefill 和 decode 的单 rank DeepEP buffer 容量。"""
    prefill_capacity = get_deepep_num_max_dispatch_tokens_per_rank_prefill()
    decode_capacity = get_deepep_num_max_dispatch_tokens_per_rank_decode()
    return max(prefill_capacity, decode_capacity)


@lru_cache(maxsize=None)
def get_lightllm_websocket_max_message_size():
    """
    Get the maximum size of the WebSocket message.
    :return: Maximum size in bytes.
    """
    return int(os.getenv("LIGHTLLM_WEBSOCKET_MAX_SIZE", 128 * 1024 * 1024))


@lru_cache(maxsize=None)
def get_eplb_step_interval():
    """返回两次 EPLB 评估之间的推理步数。"""
    interval = int(os.getenv("LIGHTLLM_EPLB_STEP_INTERVAL", 20))
    if interval <= 0:
        raise ValueError("LIGHTLLM_EPLB_STEP_INTERVAL must be greater than 0")
    return interval


@lru_cache(maxsize=None)
def get_eplb_transfer_layer_parallelism():
    """返回 EPLB 权重迁移时允许并行处理的最大层数。"""
    parallelism = int(os.getenv("LIGHTLLM_EPLB_TRANSFER_LAYER_PARALLELISM", 16))
    assert parallelism > 0
    return parallelism


@lru_cache(maxsize=None)
def get_triton_autotune_level():
    return int(os.getenv("LIGHTLLM_TRITON_AUTOTUNE_LEVEL", 0))


@lru_cache(maxsize=None)
def get_decode_attn_autotune_seq_len() -> int:
    """Decode attention 调优的代表性 KV 长度（token），默认 32768；调优时的 run key 按该长度分桶。"""
    seq_len = int(os.getenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", "32768"))
    if seq_len <= 0:
        raise ValueError("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN must be positive")
    return seq_len


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
def enable_triton_mtp_kernel() -> bool:
    """
    启用 Triton MTP 解码专用 kernel
    通过启动参数 --mtp_step > 0 和 --llm_decode_att_backend=triton 控制
    """
    return (get_env_start_args().mtp_step > 0) and ("triton" in get_env_start_args().llm_decode_att_backend)


@lru_cache(maxsize=None)
def get_disk_cache_prompt_limit_length():
    return int(os.getenv("LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH", 2048))


def get_cache_placement_gpu_capacity_ratio() -> float:
    ratio = float(os.getenv("LIGHTLLM_CACHE_PLACEMENT_GPU_CAPACITY_RATIO", 0.8))
    assert 0 < ratio <= 1
    return ratio


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
    args = get_env_start_args()
    mtp_mode = args.mtp_mode

    if mtp_mode is None:
        return 0
    if mtp_mode == "vanilla_no_att":
        return 0
    if mtp_mode == "eagle_no_att":
        return 0
    if mtp_mode == "vanilla_with_att":
        return args.mtp_step
    if mtp_mode == "eagle_with_att":
        return 1
    if mtp_mode == "eagle3":
        return _get_mtp_draft_backbone_layer_num(args.mtp_draft_model_dir[0])
    if mtp_mode == "dspark":
        return _get_mtp_draft_backbone_layer_num(args.mtp_draft_model_dir[0])
    if mtp_mode == "dflash":
        return _get_mtp_draft_backbone_layer_num(args.mtp_draft_model_dir[0])

    raise ValueError(f"unsupported mtp_mode: {mtp_mode}")


@lru_cache(maxsize=None)
def get_mtp_weight_layer_num() -> int:
    args = get_env_start_args()
    mtp_mode = args.mtp_mode

    if mtp_mode is None:
        return 0
    if mtp_mode == "vanilla_no_att":
        return args.mtp_step
    if mtp_mode == "eagle_no_att":
        return 1
    return get_added_mtp_kv_layer_num()


def _get_mtp_draft_backbone_layer_num(draft_model_dir: str) -> int:
    with open(os.path.join(draft_model_dir, "config.json"), "r") as json_file:
        draft_config = json.load(json_file)
    # Use the effective draft backbone config when the checkpoint stores it nested.
    draft_config.update(draft_config.get("dflash_config", {}))
    # A draft model may contain multiple attention layers; each layer needs a
    # separate KV-cache slot after the target model's layers.
    layer_num = draft_config.get("num_hidden_layers", draft_config.get("n_layer"))
    assert layer_num is not None, f"missing num_hidden_layers or n_layer in draft config: {draft_model_dir}"
    return int(layer_num)


@lru_cache(maxsize=None)
def get_pd_node_resource_wait_timeout_seconds() -> int:
    """P/D 节点的资源等待超时，单位为秒；负数表示永久等待。"""
    return int(os.getenv("LIGHTLLM_PD_NODE_RESOURCE_WAIT_TIMEOUT_SECONDS", 20))


@lru_cache(maxsize=None)
def get_pd_node_continuation_resource_wait_timeout_seconds() -> int:
    """P/D 节点处理续跑分段时的资源等待超时，单位为秒。"""
    return max(0, int(os.getenv("LIGHTLLM_PD_NODE_CONTINUATION_RESOURCE_WAIT_TIMEOUT_SECONDS", 60)))


@lru_cache(maxsize=None)
def get_pd_node_busy_retry_timeout_seconds() -> int:
    """PD Master 收到节点繁忙错误后的最长重试时间，单位为秒。"""
    return max(0, int(os.getenv("LIGHTLLM_PD_NODE_BUSY_RETRY_TIMEOUT_SECONDS", 120)))


@lru_cache(maxsize=None)
def get_pd_cache_high_priority_max_age_seconds() -> int:
    """cache 命中请求提升为 PD 高优先级时允许的最大缓存年龄，单位为秒。"""
    return max(0, int(os.getenv("LIGHTLLM_PD_CACHE_HIGH_PRIORITY_MAX_AGE_SECONDS", 180)))


@lru_cache(maxsize=None)
def get_pd_cache_high_priority_min_prompt_tokens() -> int:
    """cache 命中请求提升为 PD 高优先级时要求的最小 prompt token 数。"""
    return max(0, int(os.getenv("LIGHTLLM_PD_CACHE_HIGH_PRIORITY_MIN_PROMPT_TOKENS", 2048)))


@lru_cache(maxsize=None)
def get_lightllm_url_pool_maxsize() -> int:
    return int(os.getenv("LIGHTLLM_URL_POOL_MAXSIZE", 512))
