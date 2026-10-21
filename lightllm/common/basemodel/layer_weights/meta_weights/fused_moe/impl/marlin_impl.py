import torch
from .triton_impl import FuseMoeTriton
from lightllm.common.quantization.quantize_method import (
    WeightPack,
)
from lightllm.common.quantization.awq import (
    AWQMARLINW4A16QuantizationMethod,
)
from typing import Optional
from lightllm.utils.config_utils import ffn_use_tanh_approximate_gelu


class FuseMoeMarlin(FuseMoeTriton):
    def create_workspace(self):
        from lightllm.utils.vllm_utils import HAS_VLLM

        assert HAS_VLLM, "moe awq marlin quantization requires kernels of vllm"
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new,
        )

        return marlin_make_workspace_new(torch.device("cuda"), 4)

    def _fused_experts(
        self,
        input_tensor: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        is_prefill: Optional[bool] = None,
        alpha: Optional[float] = None,
        limit: Optional[float] = None,
        clamp_up_add_one: bool = True,
    ):
        if self.quant_method.method_name == "mxfp4w4a16-b32-marlin":
            from vllm.model_executor.layers.fused_moe.activation import MoEActivation
            from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
            from vllm.scalar_type import scalar_types

            if alpha not in (None, 1.0) or (limit is not None and clamp_up_add_one):
                raise ValueError("MXFP4 Marlin supports standard clamped SwiGLU")
            return fused_marlin_moe(
                hidden_states=input_tensor,
                w1=w13.weight,
                w2=w2.weight,
                bias1=None,
                bias2=None,
                w1_scale=w13.weight_scale,
                w2_scale=w2.weight_scale,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                quant_type_id=scalar_types.float4_e2m1f.id,
                global_num_experts=self.n_routed_experts,
                activation=MoEActivation.SILU,
                clamp_limit=limit,
                workspace=self.workspace,
            )
        if alpha is not None or limit is not None:
            raise NotImplementedError("FuseMoeMarlin does not support clamped SwiGLU")

        w1_weight, w1_scale, w1_zero_point = w13.weight, w13.weight_scale, w13.weight_zero_point
        w2_weight, w2_scale, w2_zero_point = w2.weight, w2.weight_scale, w2.weight_zero_point

        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe

        self.quant_method: AWQMARLINW4A16QuantizationMethod = self.quant_method

        activation = "silu" if not ffn_use_tanh_approximate_gelu() else "gelu"

        fused_marlin_moe(
            input_tensor,
            w1_weight,
            w2_weight,
            None,
            None,
            w1_scale,
            w2_scale,
            router_logits,
            topk_weights,
            topk_ids,
            quant_type_id=self.quant_method.vllm_quant_type.id,
            apply_router_weight_on_input=False,
            global_num_experts=-1,
            activation=activation,
            expert_map=None,
            w1_zeros=w1_zero_point,
            w2_zeros=w2_zero_point,
            workspace=self.workspace,
            inplace=True,
        )
        return input_tensor
