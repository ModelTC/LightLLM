"""EPLB 公共工具。"""

from typing import List, Optional, Protocol, Tuple

import torch

from lightllm.utils.envs_utils import get_env_start_args

NamedTensor = Tuple[str, torch.Tensor]


class ExpertWeightPack(Protocol):
    """EPLB 需要迁移的单组专家权重。"""

    weight: torch.Tensor
    weight_scale: Optional[torch.Tensor]
    weight_zero_point: Optional[torch.Tensor]


class EPLBExpertWeight(Protocol):
    """包含门控投影和下投影专家权重的 MoE 层。"""

    w13: ExpertWeightPack
    w2: ExpertWeightPack


def get_eplb_dispatch_mode(is_prefill: bool) -> str:
    """查询 ``(is_prefill, plan_mode)`` 对应的 EPLB 副本分发模式。"""
    default_dispatch_mode = {
        True: "current_node_first",
        False: "current_gpu_first",
    }
    # (是否为 prefill, planner 模式): 副本分发模式
    dispatch_mode_table = {
        (True, "global_balance"): "global_first",
        (True, "topology_aware"): "current_node_first",
        (False, "global_balance"): "current_gpu_first",
        (False, "topology_aware"): "current_gpu_first",
    }
    plan_mode = get_env_start_args().eplb_plan_mode
    dispatch_key = (is_prefill, plan_mode)
    return dispatch_mode_table.get(dispatch_key, default_dispatch_mode[is_prefill])


def should_record_prefill_route(is_prefill: bool) -> bool:
    """仅在 prefill 定制模式的 prefill 请求中记录路由负载。"""
    run_mode = get_env_start_args().eplb_run_mode
    return run_mode == "prefill" and is_prefill


def should_record_decode_route(is_prefill: bool) -> bool:
    """仅在 decode 定制模式的 decode 请求中记录专家共现。"""
    run_mode = get_env_start_args().eplb_run_mode
    return run_mode == "decode" and not is_prefill


def extract_eplb_expert_tensors(weight: EPLBExpertWeight) -> List[NamedTensor]:
    """按固定顺序返回 EPLB 必须迁移的权重及量化参数。"""
    named_tensors: List[NamedTensor] = []
    for pack_name in ("w13", "w2"):
        weight_pack: ExpertWeightPack = getattr(weight, pack_name)
        for value_name in ("weight", "weight_scale", "weight_zero_point"):
            tensor: Optional[torch.Tensor] = getattr(weight_pack, value_name, None)
            if tensor is not None:
                assert tensor.ndim >= 1 and tensor.is_contiguous(), f"{pack_name}.{value_name} must be contiguous"
                named_tensors.append((f"{pack_name}.{value_name}", tensor))
    return named_tensors
