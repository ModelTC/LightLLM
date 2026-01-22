import torch
from .triton_impl import FuseMoeTriton
from lightllm.common.quantization.quantize_method import (
    WeightPack,
)
from typing import Optional


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
    ):

        w1_weight, w1_scale, w1_zero_point = w13.weight, w13.weight_scale, w13.weight_zero_point
        w2_weight, w2_scale, w2_zero_point = w2.weight, w2.weight_scale, w2.weight_zero_point

        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe

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
            expert_map=None,
            w1_zeros=w1_zero_point,
            w2_zeros=w2_zero_point,
            workspace=self.workspace,
            inplace=True,
        )
        return input_tensor
