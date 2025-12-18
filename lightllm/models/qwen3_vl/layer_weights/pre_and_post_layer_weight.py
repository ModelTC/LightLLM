import numpy as np
from lightllm.models.qwen2.layer_weights.pre_and_post_layer_weight import Qwen2PreAndPostLayerWeight

# add key: language_model.xxx -> xxx
# only change keys at PreAndPostLayerWeight load, TransformLayerWeight is correct now
def rename_weight_keys(weights):
    prefix = "model.language_model."
    keys = list(weights.keys())
    for k in keys:
        if prefix in k:
            weights[k.replace(prefix, "model.")] = weights.pop(k)


class Qwen3VLPreAndPostLayerWeight(Qwen2PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        return

    def load_hf_weights(self, weights):
        rename_weight_keys(weights)
        super().load_hf_weights(weights)
        return
