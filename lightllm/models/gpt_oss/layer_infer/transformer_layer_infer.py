from lightllm.common.basemodel.moe_route_info_manager import get_moe_capture_callback
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer


class GptOssTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.head_dim_ = network_config["head_dim"]
        self.is_moe = True

    def _ffn(self, input, infer_state, layer_weight):
        hidden_states = self._tpsp_allgather(input.view(-1, self.embed_dim_), infer_state)
        router_logits = layer_weight.moe_gate.mm(hidden_states)
        hidden_states = self.bound_moe.run(
            hidden_states,
            router_logits,
            capture=get_moe_capture_callback(infer_state, self.layer_num_),
        )
        return self._tpsp_reduce(hidden_states, infer_state)
