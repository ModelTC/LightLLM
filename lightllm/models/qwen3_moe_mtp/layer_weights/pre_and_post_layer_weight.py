import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class Qwen3MOEMTPPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        # 与Qwen3MOE模型共享
        self.wte_weight_ = None
        self.lm_head_weight_ = None
        return

    def load_hf_weights(self, weights):
        if "model.layers.0.proj.weight" in weights:
            self.eh_proj_weight_ = self._cuda(weights["model.layers.0.proj.weight"]).t()
        if "model.layers.0.norm_after_embedding.weight" in weights:
            self.enorm_weight_ = self._cuda(weights["model.layers.0.norm_after_embedding.weight"])
        if "model.layers.0.norm_before_output.weight" in weights:
            self.hnorm_weight_ = self._cuda(weights["model.layers.0.norm_before_output.weight"])
        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.eh_proj_weight_, self.enorm_weight_, self.hnorm_weight_]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return
