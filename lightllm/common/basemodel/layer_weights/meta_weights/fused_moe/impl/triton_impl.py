import torch
from typing import Optional
from lightllm.common.quantization.no_quant import WeightPack
from lightllm.common.quantization.quantize_method import QuantizationMethod
from .base_impl import FuseMoeBaseImpl


class FuseMoeTriton(FuseMoeBaseImpl):
    def __init__(
        self,
        n_routed_experts: int,
        num_fused_shared_experts: int,
        routed_scaling_factor: float,
        quant_method: QuantizationMethod,
        redundancy_expert_num: int,
        redundancy_expert_ids_tensor: torch.Tensor,
        routed_expert_counter_tensor: torch.Tensor,
        auto_update_redundancy_expert: bool,
    ):
        super().__init__(
            n_routed_experts=n_routed_experts,
            num_fused_shared_experts=num_fused_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            quant_method=quant_method,
            redundancy_expert_num=redundancy_expert_num,
            redundancy_expert_ids_tensor=redundancy_expert_ids_tensor,
            routed_expert_counter_tensor=routed_expert_counter_tensor,
            auto_update_redundancy_expert=auto_update_redundancy_expert,
        )

    def create_workspace(self):
        return None

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
        return topk_weights, topk_ids

    def _fused_experts(
        self,
        input_tensor: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
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
        )
        return input_tensor

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
        per_expert_scale: Optional[torch.Tensor] = None,
        shared_expert_gate: Optional[torch.Tensor] = None,
    ):
        topk_weights, topk_ids = self._select_experts(
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
        )
        output = self._fused_experts(
            input_tensor=input_tensor,
            w13=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
            is_prefill=is_prefill,
        )
        return output
