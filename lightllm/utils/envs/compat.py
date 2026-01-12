"""
向后兼容模块

提供与旧 envs_utils.py 相同的 API，底层使用新的环境变量管理框架。
用于平滑迁移，新代码应直接使用 lightllm.utils.envs.registry。

迁移指南：
    旧代码:
        from lightllm.utils.envs_utils import get_lightllm_gunicorn_time_out_seconds
        timeout = get_lightllm_gunicorn_time_out_seconds()

    新代码:
        from lightllm.utils.envs import env
        timeout = env.server.GUNICORN_TIMEOUT.get()
"""

import warnings
from functools import lru_cache

from lightllm.utils.envs.registry import env


def _deprecation_warning(old_func: str, new_path: str):
    """发出废弃警告"""
    warnings.warn(
        f"{old_func}() 已废弃，请使用 {new_path}",
        DeprecationWarning,
        stacklevel=3,
    )


# ============================================================================
# 布尔值辅助函数
# ============================================================================
@lru_cache(maxsize=None)
def enable_env_vars(name: str) -> bool:
    """
    检查环境变量是否启用（兼容旧接口）

    新代码应直接使用 EnvVar.get() 并定义为 BOOL 类型
    """
    import os

    return os.getenv(name, "False").upper() in ["ON", "TRUE", "1"]


# ============================================================================
# 服务配置
# ============================================================================
def get_lightllm_gunicorn_time_out_seconds() -> int:
    """获取 Gunicorn 超时时间（秒）"""
    _deprecation_warning("get_lightllm_gunicorn_time_out_seconds", "env.server.GUNICORN_TIMEOUT.get()")
    return env.server.GUNICORN_TIMEOUT.get()


def get_lightllm_gunicorn_keep_alive() -> int:
    """获取 Gunicorn 保活时间（秒）"""
    _deprecation_warning("get_lightllm_gunicorn_keep_alive", "env.server.GUNICORN_KEEP_ALIVE.get()")
    return env.server.GUNICORN_KEEP_ALIVE.get()


@lru_cache(maxsize=None)
def get_lightllm_websocket_max_message_size() -> int:
    """获取 WebSocket 最大消息大小"""
    _deprecation_warning("get_lightllm_websocket_max_message_size", "env.server.WEBSOCKET_MAX_SIZE.get()")
    return env.server.WEBSOCKET_MAX_SIZE.get()


@lru_cache(maxsize=None)
def get_unique_server_name() -> str:
    """获取服务唯一标识"""
    _deprecation_warning("get_unique_server_name", "env.server.UNIQUE_SERVICE_NAME_ID.get()")
    return env.server.UNIQUE_SERVICE_NAME_ID.get()


# ============================================================================
# KV 缓存和量化
# ============================================================================
@lru_cache(maxsize=None)
def get_kv_quant_calibration_warmup_count() -> int:
    """获取 KV 量化校准预热次数"""
    _deprecation_warning("get_kv_quant_calibration_warmup_count", "env.kv_cache.QUANT_CALIBRATION_WARMUP_COUNT.get()")
    return env.kv_cache.QUANT_CALIBRATION_WARMUP_COUNT.get()


@lru_cache(maxsize=None)
def get_kv_quant_calibration_inference_count() -> int:
    """获取 KV 量化校准推理次数"""
    _deprecation_warning(
        "get_kv_quant_calibration_inference_count", "env.kv_cache.QUANT_CALIBRATION_INFERENCE_COUNT.get()"
    )
    return env.kv_cache.QUANT_CALIBRATION_INFERENCE_COUNT.get()


@lru_cache(maxsize=None)
def get_disk_cache_prompt_limit_length() -> int:
    """获取磁盘缓存提示长度限制"""
    _deprecation_warning("get_disk_cache_prompt_limit_length", "env.kv_cache.DISK_CACHE_PROMPT_LIMIT_LENGTH.get()")
    return env.kv_cache.DISK_CACHE_PROMPT_LIMIT_LENGTH.get()


# ============================================================================
# 专家冗余
# ============================================================================
@lru_cache(maxsize=None)
def get_redundancy_expert_update_interval() -> int:
    """获取冗余专家更新间隔"""
    _deprecation_warning("get_redundancy_expert_update_interval", "env.redundancy.UPDATE_INTERVAL.get()")
    return env.redundancy.UPDATE_INTERVAL.get()


@lru_cache(maxsize=None)
def get_redundancy_expert_update_max_load_count() -> int:
    """获取冗余专家更新最大加载次数"""
    _deprecation_warning("get_redundancy_expert_update_max_load_count", "env.redundancy.UPDATE_MAX_LOAD_COUNT.get()")
    return env.redundancy.UPDATE_MAX_LOAD_COUNT.get()


# ============================================================================
# GPU 和 Triton
# ============================================================================
@lru_cache(maxsize=None)
def get_triton_autotune_level() -> int:
    """获取 Triton 自动调优级别"""
    _deprecation_warning("get_triton_autotune_level", "env.gpu.TRITON_AUTOTUNE_LEVEL.get()")
    return env.gpu.TRITON_AUTOTUNE_LEVEL.get()


# ============================================================================
# Radix Tree
# ============================================================================
@lru_cache(maxsize=None)
def enable_radix_tree_timer_merge() -> bool:
    """检查是否启用 Radix Tree 定时合并"""
    _deprecation_warning("enable_radix_tree_timer_merge", "env.radix_tree.MERGE_ENABLE.get()")
    return env.radix_tree.MERGE_ENABLE.get()


@lru_cache(maxsize=None)
def get_radix_tree_merge_update_delta() -> int:
    """获取 Radix Tree 合并更新增量"""
    _deprecation_warning("get_radix_tree_merge_update_delta", "env.radix_tree.MERGE_DELTA.get()")
    return env.radix_tree.MERGE_DELTA.get()


# ============================================================================
# 批处理和调度
# ============================================================================
@lru_cache(maxsize=None)
def get_diverse_max_batch_shared_group_size() -> int:
    """获取最大批处理共享组大小"""
    _deprecation_warning("get_diverse_max_batch_shared_group_size", "env.scheduling.MAX_BATCH_SHARED_GROUP_SIZE.get()")
    return env.scheduling.MAX_BATCH_SHARED_GROUP_SIZE.get()


# ============================================================================
# 分布式
# ============================================================================
@lru_cache(maxsize=None)
def get_deepep_num_max_dispatch_tokens_per_rank() -> int:
    """获取每个 rank 最大派发 Token 数"""
    _deprecation_warning(
        "get_deepep_num_max_dispatch_tokens_per_rank", "env.distributed.NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()"
    )
    return env.distributed.NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()


# ============================================================================
# 内存
# ============================================================================
@lru_cache(maxsize=None)
def enable_huge_page() -> bool:
    """检查是否启用大页内存"""
    _deprecation_warning("enable_huge_page", "env.memory.HUGE_PAGE_ENABLE.get()")
    return env.memory.HUGE_PAGE_ENABLE.get()


# ============================================================================
# 模型相关
# ============================================================================
def use_whisper_sdpa_attention() -> bool:
    """检查是否使用 Whisper SDPA 注意力"""
    _deprecation_warning("use_whisper_sdpa_attention", "env.model.USE_WHISPER_SDPA_ATTENTION.get()")
    return env.model.USE_WHISPER_SDPA_ATTENTION.get()
