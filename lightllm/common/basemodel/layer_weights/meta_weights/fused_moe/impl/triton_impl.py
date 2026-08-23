import torch
from typing import Optional
from lightllm.common.quantization.no_quant import WeightPack
from .base_impl import FuseMoeBaseImpl


class FuseMoeTriton(FuseMoeBaseImpl):
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
    ):
        """Select experts and return topk weights and ids."""
        from lightllm.common.basemodel.triton_kernel.fused_moe.topk_select import select_experts

        topk_weights, topk_ids = select_experts(
            hidden_states=input_tensor,
            router_logits=router_logits,
            correction_bias=correction_bias,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
        )
        if self.routed_scaling_factor != 1.0:
            topk_weights.mul_(self.routed_scaling_factor)
        if per_expert_scale is not None:
            topk_weights = topk_weights * per_expert_scale[topk_ids.to(torch.long)].to(topk_weights.dtype)
        origin_topk_ids = topk_ids
        if self.num_fused_shared_experts > 0:
            from lightllm.common.basemodel.triton_kernel.fused_moe.append_shared_expert_topk import (
                append_fused_shared_experts,
            )

            topk_weights, topk_ids = append_fused_shared_experts(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_expert_start_id=self.n_routed_experts,
                num_fused_shared_experts=self.num_fused_shared_experts,
                shared_expert_gate=shared_expert_gate,
            )
        return topk_weights, topk_ids, origin_topk_ids

    def _fused_experts(
        self,
        input_tensor: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
        clamp_limit: Optional[float] = None,
        alloc_tensor_func=torch.empty,
    ):
        w13_weight, w13_scale = w13.weight, w13.weight_scale
        w2_weight, w2_scale = w2.weight, w2.weight_scale
        use_fp8_w8a8 = w13_weight.dtype == torch.float8_e4m3fn

        from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe import fused_experts

        fused_experts(
            hidden_states=input_tensor,
            w1=w13_weight,
            w2=w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            use_fp8_w8a8=use_fp8_w8a8,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            limit=clamp_limit,
        )
        return input_tensor

    def fused_experts_with_topk(
        self,
        input_tensor: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_prefill: Optional[bool] = None,
        clamp_limit: Optional[float] = None,
        alloc_tensor_func=torch.empty,
    ):
        return self._fused_experts(
            input_tensor=input_tensor,
            w13=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            is_prefill=is_prefill,
            clamp_limit=clamp_limit,
            alloc_tensor_func=alloc_tensor_func,
        )
