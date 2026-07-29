from .triton_impl import FuseMoeTriton


class FuseMoeMXFP4(FuseMoeTriton):
    def _fused_experts(
        self,
        input_tensor,
        w13,
        w2,
        topk_weights,
        topk_ids,
        router_logits=None,
        is_prefill=False,
        shared_expert_out=None,
        shared_expert_gate=None,
    ):
        del router_logits, is_prefill
        if shared_expert_out is not None or shared_expert_gate is not None:
            raise NotImplementedError("MXFP4 fused shared experts are not supported")

        from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_mxfp4 import (
            fused_experts_mxfp4,
        )

        return fused_experts_mxfp4(
            hidden_states=input_tensor,
            w1=w13.weight,
            w2=w2.weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1_scale=w13.weight_scale,
            w2_scale=w2.weight_scale,
            activation=getattr(self, "activation", "silu"),
            activation_situ_beta=getattr(self, "activation_situ_beta", None),
            activation_situ_linear_beta=getattr(self, "activation_situ_linear_beta", None),
        )
