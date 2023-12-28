import os
import torch
import torch.nn.functional as F
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.mistral.layer_infer.transformer_layer_infer import MistralTransformerLayerInfer
from lightllm.models.mixtral.layer_weights.transformer_layer_weight import MixtralTransformerLayerWeight

class MixtralTransformerLayerInfer(LlamaTransformerLayerInfer):
    def _ffn(self, input, infer_state: InferStateInfo, layer_weight: MixtralTransformerLayerWeight) -> torch.Tensor:
        router_logits = torch.mm(input.view(-1, self.embed_dim_), layer_weight.gate)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       infer_state.experts_topk,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = None
        for expert_idx in range(infer_state.num_local_experts):
            expert_layer = layer_weight.experts[expert_idx]
            expert_mask = (selected_experts == expert_idx)
            expert_weights = (routing_weights * expert_mask).sum(dim=-1,
                                                                 keepdim=True)
            w1_out = torch.mm(input.view(-1, self.embed_dim_), expert_layer['w1'])
            torch.nn.functional.silu(w1_out, inplace=True)
            w3_out = torch.mm(input.view(-1, self.embed_dim_), expert_layer['w3'])
            current_hidden_states = w1_out * w3_out
            w1_out, w3_out = None, None
            current_hidden_states = torch.mm(current_hidden_states, expert_layer['w2'])
            current_hidden_states = current_hidden_states.mul_(expert_weights)
            
            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states.add_(current_hidden_states)
        input = None
        return final_hidden_states