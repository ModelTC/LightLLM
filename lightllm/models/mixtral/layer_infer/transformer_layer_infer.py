import torch
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.mixtral.layer_weights.transformer_layer_weight import MixtralTransformerLayerWeight


class MixtralTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.num_local_experts = network_config["num_local_experts"]
        self.num_experts_per_tok = network_config["num_experts_per_tok"]
        self.renormalize = True
        return

    def _ffn(self, input, infer_state: InferStateInfo, layer_weight: MixtralTransformerLayerWeight) -> torch.Tensor:
        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        router_logits = layer_weight.moe_gate.mm(hidden_states)
        layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.renormalize,
            use_grouped_topk=False,
            topk_group=None,
            num_expert_group=None,
            microbatch_index=getattr(infer_state, "microbatch_index", 0),
        )
        return hidden_states.view(num_tokens, hidden_dim)
