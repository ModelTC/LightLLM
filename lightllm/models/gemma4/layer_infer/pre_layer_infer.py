import torch
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer


class Gemma4PreLayerInfer(LlamaPreLayerInfer):
    """
    Text-only pre-layer for Gemma-4 (Phase A). Applies the Gemma embedding
    scale (sqrt(hidden_size)) to the token embeddings. Multimodal embed-scatter
    handling will be added alongside the vision tower port.
    """

    def __init__(self, network_config):
        super().__init__(network_config)
        self.embed_scale = float(network_config["hidden_size"]) ** 0.5

    def context_forward(self, input_ids, infer_state, layer_weight):
        input_embdings = super().context_forward(input_ids, infer_state, layer_weight)
        input_dtype = input_embdings.dtype
        return (input_embdings.float() * self.embed_scale).to(input_dtype)

    def token_forward(self, input_ids, infer_state, layer_weight):
        input_embdings = super().token_forward(input_ids, infer_state, layer_weight)
        input_dtype = input_embdings.dtype
        return (input_embdings.float() * self.embed_scale).to(input_dtype)
