import torch
from typing import Optional
from lightllm.common.quantization.no_quant import WeightPack
from lightllm.common.quantization.quantize_method import QuantizationMethod
from .base_impl import FuseMoeBaseImpl
from lightllm.common.basemodel import routing_manager as _routing_mgr


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
        topk_weights.mul_(self.routed_scaling_factor)
        if self.num_fused_shared_experts > 0:
            pad_topk_ids = (
                torch.arange(
                    start=self.n_routed_experts,
                    end=self.n_routed_experts + self.num_fused_shared_experts,
                    step=1,
                    dtype=topk_ids.dtype,
                    device="cuda",
                )
                .view(1, self.num_fused_shared_experts)
                .repeat(topk_ids.shape[0], 1)
            )
            pad_topk_weights = torch.full(
                (topk_weights.shape[0], self.num_fused_shared_experts),
                fill_value=1.0,
                device="cuda",
                dtype=topk_weights.dtype,
            )

            topk_ids = torch.cat([topk_ids, pad_topk_ids], dim=1)
            topk_weights = torch.cat([topk_weights, pad_topk_weights], dim=1)
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
        moe_layer_index: Optional[int] = None,
        microbatch_index: int = 0,
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
        )

        if _routing_mgr.g_routing_capture_manager is not None and moe_layer_index is not None:
            _routing_mgr.g_routing_capture_manager.capture(moe_layer_index, topk_ids, microbatch_index)

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
