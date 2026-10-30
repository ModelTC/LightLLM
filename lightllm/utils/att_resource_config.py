"""静态 FP8 标定加载与 CPU cache 服务布局构建。"""

from __future__ import annotations

from typing import TYPE_CHECKING
import json

import torch

from lightllm.common.att_resources import AttResourceManager
from lightllm.utils.envs_utils import get_env_start_args

if TYPE_CHECKING:
    from collections.abc import Mapping
    from lightllm.common.att_resources.types import StateTransferLayout
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
    storage_layer_start: int = 0,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor | None]:
    """读取模型和 CPU cache 共用的静态标定参数。

    storage_layer_count 是当前 State 的物理层数；storage_layer_start 是其在标定表中的
    物理行起点，两者都不是逻辑层号。共享逻辑层使用同一行标定参数，不重复占用行。
    返回 Tensor 保留在 CPU；当前 TP 分片在每行中选取对应的 head 参数。
    """
    path = args.kv_quant_calibration_config_path
    if path is None:
        assert not per_head, "fp8kv_sph requires kv_quant_calibration_config_path with q_calibration"
        return {"scale": "unit"}, torch.ones((storage_layer_count, 2), dtype=torch.float32), None
    with open(path) as f:
        cfg = json.load(f)
    calibrated_storage_layer_count = cfg["num_layers"]
    # 当前模型只读取自己的物理层区间；main/draft 偏移由装配流程明确提供。
    storage_layer_end = storage_layer_start + storage_layer_count
    assert 0 <= storage_layer_start < storage_layer_end <= calibrated_storage_layer_count, (
        "calibration does not cover this model's attention layers"
    )

    if not per_head:
        from lightllm.utils.config_utils import get_model_architectures

        assert cfg["architectures"] == get_model_architectures(args.model_dir), (
            "calibration architecture does not match the model"
        )
    mode = "per_head" if per_head else "per_tensor"
    assert cfg.get("quant_type", mode) == mode, "calibration quantization mode does not match the State"
    limits = torch.finfo(torch.float8_e4m3fn)
    assert cfg.get("qmin", limits.min) == limits.min and cfg.get("qmax", limits.max) == limits.max, (
        "calibration range does not match float8_e4m3fn"
    )

    def scale_tensor(values: Mapping[str, object], shape: tuple[int, ...]) -> torch.Tensor:
        tensor = torch.tensor(values["scales"], dtype=torch.float32)
        assert tuple(values["scales_shape"]) == tensor.shape == shape, (
            f"calibration scales and scales_shape must match {shape}"
        )
        assert bool((torch.isfinite(tensor) & (tensor > 0)).all()), "calibration scales must be finite and positive"
        return tensor[storage_layer_start:storage_layer_end]

    def shard(tensor: torch.Tensor, calibrated_heads: int) -> torch.Tensor:
        assert calibrated_heads > 0 and (heads * tp_world_size) % calibrated_heads == 0, (
            "calibration heads do not cover the model's KV heads"
        )
        expanded = tensor.repeat_interleave(heads * tp_world_size // calibrated_heads, dim=-1)
        return expanded[:, tp_rank * heads : (tp_rank + 1) * heads].contiguous()

    if per_head:
        calibrated_heads = cfg["num_head"]
        all_scales = scale_tensor(cfg, (calibrated_storage_layer_count, 2 * calibrated_heads))
        k, v = all_scales.split(calibrated_heads, dim=-1)
        scales = torch.cat((shard(k, calibrated_heads), shard(v, calibrated_heads)), dim=-1)
        q_cfg = cfg.get("q_calibration")
        assert q_cfg, "fp8kv_sph calibration config requires q_calibration"
        q_scales = shard(scale_tensor(q_cfg, (calibrated_storage_layer_count, q_cfg["num_head"])), q_cfg["num_head"])
    else:
        scales, q_scales = scale_tensor(cfg, (calibrated_storage_layer_count, 2)), None
    return cfg, scales, q_scales


def build_cpu_cache_layouts() -> Mapping[str, StateTransferLayout]:
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

    # 阶段一：读取 main 和启用 attention 的 draft 配置，不构造实际模型。
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

    # 阶段二：复用模型定义入口创建 meta State，并累计逻辑层号和标定物理行偏移。
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

    # 阶段三：确认混合模型的页边界一致，由管理器聚合 CPU 联合页布局。
    manager.finish_state_creation()
    if manager.get_bounded_attention_states():
        assert (
            args.cpu_cache_token_page_size == args.linear_att_hash_page_size * args.linear_att_page_block_num
        ), "hybrid CPU pages must match the large state-page token boundary"
    return manager.get_cpu_transfer_layouts()
