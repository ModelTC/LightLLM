import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class MistralMTPPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        self.wte_weight_ = None
        self.lm_head_weight_ = None
        self.final_norm_weight_ = None
        return

    def load_hf_weights(self, weights):
        rename_weights(weights)
        if "model.eh_proj.weight" in weights:
            self.eh_proj_weight_ = self._cuda(weights["model.eh_proj.weight"]).t()
        if "model.enorm.weight" in weights:
            self.enorm_weight_ = self._cuda(weights["model.enorm.weight"])
        if "model.hnorm.weight" in weights:
            self.hnorm_weight_ = self._cuda(weights["model.hnorm.weight"])
        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.eh_proj_weight_, self.enorm_weight_, self.hnorm_weight_]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return


def rename_weights(weights):
    all_keys = list(weights.keys())
    for key in all_keys:
        if key.startswith("mtp."):
            weights[key.replace("mtp.", "model.")] = weights.pop(key)
    return weights
