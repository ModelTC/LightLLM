import os
import torch
import torch.nn.functional as F
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.mistral.layer_infer.transformer_layer_infer import MistralTransformerLayerInfer
from lightllm.models.mixtral.layer_infer._custom_ops import fused_topk
from lightllm.models.mixtral.layer_weights.transformer_layer_weight import MixtralTransformerLayerWeight
from lightllm.models.deepseek2.layer_infer.fused_moe import fused_experts


class MixtralTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.num_local_experts = network_config["num_local_experts"]
        self.num_experts_per_tok = network_config["num_experts_per_tok"]
        self.renormalize = True
        return

    def _ffn(self, input, infer_state: InferStateInfo, layer_weight: MixtralTransformerLayerWeight) -> torch.Tensor:
        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        router_logits = torch.mm(input.view(-1, self.embed_dim_), layer_weight.moe_gate)
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.num_experts_per_tok,
            renormalize=self.renormalize,
            alloc_tensor_func=self.alloc_tensor,
        )
        # print(topk_weights, topk_ids)

        return fused_experts(
            hidden_states=hidden_states,
            w1=layer_weight.w1,
            w2=layer_weight.w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            alloc_tensor_func=self.alloc_tensor,
        )
