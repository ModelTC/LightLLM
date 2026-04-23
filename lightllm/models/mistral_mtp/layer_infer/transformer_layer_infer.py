from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.mistral.layer_infer.transformer_layer_infer import MistralTransformerLayerInfer


class MistralMTPTransformerLayerInfer(MistralTransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        return

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings
