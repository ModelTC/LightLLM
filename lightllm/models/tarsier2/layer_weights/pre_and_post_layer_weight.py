import torch
import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight

from lightllm.models.internlm2.layer_weights.pre_and_post_layer_weight import Internlm2PreAndPostLayerWeight
from lightllm.models.qwen2.layer_weights.pre_and_post_layer_weight import Qwen2PreAndPostLayerWeight


# add key: language_model.xxx -> xxx
# only change keys at PreAndPostLayerWeight load, TransformLayerWeight is correct now
def rename_weight_keys(weights):
    prefix = "language_model."
    keys = list(weights.keys())
    for k in keys:
        if prefix in k:
            weights[k[len(prefix) :]] = weights[k]


class Tarsier2Qwen2PreAndPostLayerWeight(Qwen2PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        return

    def load_hf_weights(self, weights):
        rename_weight_keys(weights)
        super().load_hf_weights(weights)
        return


class Tarsier2LlamaPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        return

    def load_hf_weights(self, weights):
        rename_weight_keys(weights)
        super().load_hf_weights(weights)
        return
