from lightllm.models.qwen3next.layer_weights.pre_and_post_layer_weight import Qwen3NextPreAndPostLayerWeight
from lightllm.models.qwen3_vl.layer_weights.pre_and_post_layer_weight import rename_weight_keys


class Qwen35PreAndPostLayerWeight(Qwen3NextPreAndPostLayerWeight):
    def load_hf_weights(self, weights):
        rename_weight_keys(weights)
        super().load_hf_weights(weights)
        return
