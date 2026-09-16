from typing import Optional

import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.expert_parallel_state import (
    ExpertParallelState,
)
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.distributed import dist_group_manager

from .triton_impl import FuseMoeTriton


class FuseMoeTritonEP(FuseMoeTriton):
    """Triton MoE backend for expert-parallel symmetric-memory execution."""

    def __init__(self, *args, expert_parallel_state: ExpertParallelState, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_parallel_state = expert_parallel_state

    def _fused_experts(
        self,
        input_tensor: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        is_prefill: Optional[bool] = None,
        clamp_limit: Optional[float] = None,
        alloc_tensor_func=torch.empty,
    ) -> torch.Tensor:
        buffer = dist_group_manager.ep_triton_moe_buffer
        return buffer.forward(
            input_tensor,
            (w13.weight, w13.weight_scale),
            (w2.weight, w2.weight_scale),
            topk_weights,
            topk_ids.to(torch.long),
            float("inf") if clamp_limit is None else clamp_limit,
            alloc_tensor_func=alloc_tensor_func,
        )
