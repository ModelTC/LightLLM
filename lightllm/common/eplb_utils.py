"""传输与模型分析共用的轻量 EPLB 工具。"""

from typing import List, Optional, Protocol, Tuple

import torch

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
