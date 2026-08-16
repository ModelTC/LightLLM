import torch
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
from lightllm.common.quantization.quantize_method import (
    WeightPack,
    QuantizationMethod,
)


class FuseMoeBaseImpl(ABC):
    def __init__(
        self,
        n_routed_experts: int,
        num_fused_shared_experts: int,
        routed_scaling_factor: float,
        quant_method: QuantizationMethod,
    ):
        self.n_routed_experts = n_routed_experts
        self.num_fused_shared_experts = num_fused_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.quant_method = quant_method

    def __call__(
        self,
        input_tensor: torch.Tensor,
        router_logits: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        correction_bias: Optional[torch.Tensor],
        scoring_func: str,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: int,
        num_expert_group: int,
        is_prefill: Optional[bool] = None,
        # Callback to capture MoE topk expert ids (routed experts metadata).
        moe_capture_callback: Optional[Callable[[torch.Tensor], None]] = None,
        per_expert_scale: Optional[torch.Tensor] = None,
        # Qwen3.5 uses this gate to control fused shared expert aggregation weights.
        shared_expert_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids, origin_topk_ids = self._select_experts(
            input_tensor=input_tensor,
            router_logits=router_logits,
            correction_bias=correction_bias,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
            per_expert_scale=per_expert_scale,
            shared_expert_gate=shared_expert_gate,
            is_prefill=is_prefill,
            preserve_logical_ids=moe_capture_callback is not None,
        )
        if moe_capture_callback is not None:
            moe_capture_callback(origin_topk_ids)
        return self._fused_experts(
            input_tensor=input_tensor,
            w13=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
            is_prefill=is_prefill,
        )

    @abstractmethod
    def _select_experts(
        self,
        input_tensor: torch.Tensor,
        router_logits: torch.Tensor,
        correction_bias: Optional[torch.Tensor],
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: int,
        num_expert_group: int,
        scoring_func: str,
        per_expert_scale: Optional[torch.Tensor] = None,
        shared_expert_gate: Optional[torch.Tensor] = None,
        is_prefill: Optional[bool] = None,
        preserve_logical_ids: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return top-k weights, execution IDs, and logical IDs for capture."""
        pass

    @abstractmethod
    def _fused_experts(
        self,
        input_tensor: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        is_prefill: Optional[bool] = None,
    ) -> torch.Tensor:
        """Apply selected experts and return the fused-MoE output."""
        pass
