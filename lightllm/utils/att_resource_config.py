"""Shared State definitions for model initialization and CPU page metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING
import hashlib
import json
from pathlib import Path

import torch

from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.full_att import (
    FullAttentionConfig, FullAttentionState, create_gqa_state, create_mla_state,
)
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num, get_env_start_args
from lightllm.utils.torch_dtype_utils import get_torch_dtype

if TYPE_CHECKING:
    from collections.abc import Mapping
    from easydict import EasyDict
    from lightllm.server.core.objs.start_args_type import StartArgs


def load_static_fp8_calibration(
    args: StartArgs | EasyDict,
    *,
    per_head: bool,
    storage_layer_count: int,
    heads: int,
    tp_rank: int = 0,
    tp_world_size: int = 1,
    storage_layer_start: int | None = None,
):
    """读取模型和 CPU cache 共用的静态标定参数。

    storage_layer_count 是当前 State 的物理层数；storage_layer_start 是其在标定表中的
    物理行起点，两者都不是逻辑层号。共享逻辑层使用同一行标定参数，不重复占用行。
    返回 Tensor 保留在 CPU；当前 TP 分片在每行中选取对应的 head 参数。
    """
    path = args.kv_quant_calibration_config_path
    if path is None:
        if per_head:
            raise ValueError("fp8kv_sph requires kv_quant_calibration_config_path with q_calibration")
        return {"scale": "unit"}, torch.ones((storage_layer_count, 2), dtype=torch.float32), None
    with open(path) as f:
        cfg = json.load(f)
    calibrated_storage_layer_count = cfg["num_layers"]
    expected_draft_storage_layer_count = get_added_mtp_kv_layer_num() if storage_layer_start is None else 0
    if storage_layer_start is not None:
        if storage_layer_start < 0 or storage_layer_start + storage_layer_count > calibrated_storage_layer_count:
            raise ValueError("calibration does not cover this model's attention layers")
    elif per_head and ("num_target_layers" in cfg or "num_draft_layers" in cfg):
        calibrated_target_storage_layer_count = cfg["num_target_layers"]
        calibrated_draft_storage_layer_count = cfg["num_draft_layers"]
        if (
            calibrated_draft_storage_layer_count < 0
            or calibrated_target_storage_layer_count + calibrated_draft_storage_layer_count != calibrated_storage_layer_count
            or calibrated_target_storage_layer_count != storage_layer_count - expected_draft_storage_layer_count
            or (
                expected_draft_storage_layer_count > 0
                and calibrated_draft_storage_layer_count != expected_draft_storage_layer_count
            )
        ):
            raise ValueError("calibration target/draft layer layout does not match the runtime")
    elif calibrated_storage_layer_count != storage_layer_count:
        raise ValueError(
            f"calibration num_layers {calibrated_storage_layer_count} does not match runtime {storage_layer_count}"
        )

    if not per_head:
        from lightllm.utils.config_utils import get_model_architectures

        if cfg["architectures"] != get_model_architectures(args.model_dir):
            raise ValueError("calibration architecture does not match the model")
    mode = "per_head" if per_head else "per_tensor"
    if cfg.get("quant_type", mode) != mode:
        raise ValueError("calibration quantization mode does not match the State")
    limits = torch.finfo(torch.float8_e4m3fn)
    if cfg.get("qmin", limits.min) != limits.min or cfg.get("qmax", limits.max) != limits.max:
        raise ValueError("calibration range does not match float8_e4m3fn")

    def scale_tensor(values: Mapping[str, object], shape: tuple[int, ...]):
        tensor = torch.tensor(values["scales"], dtype=torch.float32)
        if list(values["scales_shape"]) != list(shape) or list(tensor.shape) != list(shape):
            raise ValueError(f"calibration scales and scales_shape must match {shape}")
        if not bool((torch.isfinite(tensor) & (tensor > 0)).all()):
            raise ValueError("calibration scales must be finite and positive")
        first_storage_layer_index = storage_layer_start or 0
        return tensor[first_storage_layer_index : first_storage_layer_index + storage_layer_count]

    def shard(tensor: torch.Tensor, calibrated_heads: int):
        if calibrated_heads <= 0 or (heads * tp_world_size) % calibrated_heads:
            raise ValueError("calibration heads do not cover the model's KV heads")
        expanded = tensor.repeat_interleave(heads * tp_world_size // calibrated_heads, dim=-1)
        return expanded[:, tp_rank * heads : (tp_rank + 1) * heads].contiguous()

    if per_head:
        calibrated_heads = cfg["num_head"]
        all_scales = scale_tensor(cfg, (calibrated_storage_layer_count, 2 * calibrated_heads))
        k, v = all_scales.split(calibrated_heads, dim=-1)
        scales = torch.cat((shard(k, calibrated_heads), shard(v, calibrated_heads)), dim=-1)
        q_cfg = cfg.get("q_calibration")
        if not q_cfg:
            raise ValueError("fp8kv_sph calibration config requires q_calibration")
        q_scales = shard(scale_tensor(q_cfg, (calibrated_storage_layer_count, q_cfg["num_head"])), q_cfg["num_head"])
    else:
        scales, q_scales = scale_tensor(cfg, (calibrated_storage_layer_count, 2)), None
    return cfg, scales, q_scales


def build_full_attention_state(
    args: StartArgs | EasyDict,
    *,
    heads: int,
    dim: int,
    logical_layer_groups: tuple[tuple[int, ...], ...],
    mla: bool = False,
    packed: bool = False,
    dtype: torch.dtype | None = None,
    replicated: bool = False,
    tp_rank: int = 0,
    tp_world_size: int = 1,
    calibration_storage_layer_start: int | None = None,
) -> FullAttentionState:
    """将模型和启动配置转换为算法 create 参数，返回容量规划所需的 meta State。"""
    dtype = get_torch_dtype(args.data_type) if dtype is None else dtype
    config = FullAttentionConfig(
        logical_layer_groups=logical_layer_groups,
        global_head_num=heads,
        head_dim=dim,
        dtype=dtype,
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
        replicated=replicated or mla or packed,
    )
    if mla or packed:
        return create_mla_state(
            config=config, kv_type=args.llm_kv_type, packed=packed, page_size=args.page_size, device="meta",
        )
    scales = q_scales = None
    calibration_id = ""
    if args.llm_kv_type in ("fp8kv_sph", "fp8kv_spt"):
        calibration, scales, q_scales = load_static_fp8_calibration(
            args,
            per_head=args.llm_kv_type == "fp8kv_sph",
            storage_layer_count=len(logical_layer_groups),
            heads=config.head_num,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            storage_layer_start=calibration_storage_layer_start,
        )
        calibration_id = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    return create_gqa_state(
        config=config, kv_type=args.llm_kv_type, page_size=args.page_size, device="meta",
        scales=scales, q_scales=q_scales, calibration_id=calibration_id,
    )


def build_cpu_cache_layouts():
    """使用启动流程相同的 meta State 定义，由管理器统一生成 CPU 服务页布局。"""
    from transformers.configuration_utils import PretrainedConfig
    from lightllm.models import get_draft_model_class
    from lightllm.common.att_resources.utils import MAX_DRAFT_MODELS
    from lightllm.common.basemodel.basemodel_utils import (
        create_full_attention_definition,
        create_bounded_attention_definition,
        get_resource_config,
    )
    from lightllm.utils.llm_utils import get_llm_model_class

    args = get_env_start_args()
    main_config, _ = PretrainedConfig.get_config_dict(args.model_dir)
    models = [("main", get_llm_model_class(), main_config)]
    if args.mtp_mode is not None and args.mtp_mode not in ("vanilla_no_att", "eagle_no_att"):
        count = args.mtp_step if args.mtp_mode == "vanilla_with_att" else 1
        assert args.mtp_draft_model_dir is not None and len(args.mtp_draft_model_dir) >= count
        assert 0 < count <= MAX_DRAFT_MODELS, "at most 36 draft models are supported"
        for index, directory in enumerate(args.mtp_draft_model_dir[:count]):
            config, _ = PretrainedConfig.get_config_dict(directory)
            models.append((f"draft{index}", get_draft_model_class(config, args.mtp_mode), config))

    manager = AttResourceManager(request_capacity=0)  # 只描述 meta 布局，不分配请求或缓存存储。
    logical_layer_start = calibration_storage_layer_start = 0
    for role, model_class, config in models:
        count = get_resource_config(config, model_class=model_class)["num_hidden_layers"]
        logical_layer_ids = tuple(range(logical_layer_start, logical_layer_start + count))
        full = create_full_attention_definition(
            model_class, config, args, model_logical_layer_ids=logical_layer_ids, calibration_storage_layer_start=calibration_storage_layer_start
        )
        bounded = create_bounded_attention_definition(
            model_class, config, args, model_logical_layer_ids=logical_layer_ids
        )
        for state in (full, bounded):
            if state is None:
                continue
            manager.set_att_state(role, state)
        logical_layer_start += count
        calibration_storage_layer_start += len(full.config.logical_layer_groups) if full is not None else 0

    manager.finish_state_creation()
    if manager.get_bounded_attention_states():
        assert (
            args.cpu_cache_token_page_size == args.linear_att_hash_page_size * args.linear_att_page_block_num
        ), "hybrid CPU pages must match the large state-page token boundary"
    return manager.get_cpu_transfer_layouts()


def cpu_cache_namespace():
    """按 main/draft 模型路径及文件版本隔离磁盘缓存。"""
    args = get_env_start_args()
    directories = [args.model_dir, *(args.mtp_draft_model_dir or [])]
    revisions = []
    for directory in directories:
        directory = Path(directory).resolve()
        artifacts = []
        for pattern in ("*.safetensors", "*.bin", "*.index.json", "config.json"):
            for path in sorted(directory.glob(pattern)):
                stat = path.stat()
                artifacts.append((path.name, stat.st_size, stat.st_mtime_ns))
        revisions.append((str(directory), artifacts))
    return hashlib.sha256(repr(revisions).encode()).hexdigest()
