import torch
from typing import Optional

from lightllm.common.quantization.quantize_method import WeightPack
from .triton_impl import FuseMoeTriton


class FuseMoeMXFP4(FuseMoeTriton):
    def create_workspace(self):
        return None

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
    ):
        try:
            from vllm.model_executor.layers.fused_moe.activation import MoEActivation
            from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
            from vllm.scalar_type import scalar_types
        except Exception as e:
            raise RuntimeError(f"MXFP4 fused MoE requires vLLM fused kernels, error={repr(e)}") from e

        return fused_marlin_moe(
            hidden_states=input_tensor.contiguous(),
            w1=w13.weight,
            w2=w2.weight,
            bias1=None,
            bias2=None,
            w1_scale=w13.weight_scale,
            w2_scale=w2.weight_scale,
            topk_weights=topk_weights.to(torch.float32).contiguous(),
            topk_ids=topk_ids.to(torch.long).contiguous(),
            quant_type_id=scalar_types.float4_e2m1f.id,
            global_num_experts=self.n_routed_experts,
            activation=MoEActivation.SILU,
            clamp_limit=clamp_limit,
        )
