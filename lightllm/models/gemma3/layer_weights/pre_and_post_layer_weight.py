import torch
import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


# add key: language_model.xxx -> xxx
# only change keys at PreAndPostLayerWeight load, TransformLayerWeight is correct now
def rename_weight_keys(weights):
    prefix = "language_model."
    keys = list(weights.keys())
    for k in keys:
        if prefix in k:
            weights[k[len(prefix) :]] = weights[k]


class Gemma3PreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        network_config["tie_word_embeddingse"] = True
        super().__init__(data_type, network_config, mode)
        return

    def load_hf_weights(self, weights):
        rename_weight_keys(weights)
        super().load_hf_weights(weights)
        return
