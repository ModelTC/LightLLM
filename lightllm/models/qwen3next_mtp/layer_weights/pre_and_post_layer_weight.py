import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class Qwen3NextMTPPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        self.wte_weight_ = None
        self.lm_head_weight_ = None
        return

    def load_hf_weights(self, weights):
        if "mtp.fc.weight" in weights:
            self.fc_weight_ = self._cuda(weights["mtp.fc.weight"]).t()
        if "mtp.pre_fc_norm_embedding.weight" in weights:
            self.pre_fc_norm_embedding_weight_ = self._cuda(weights["mtp.pre_fc_norm_embedding.weight"])
        if "mtp.pre_fc_norm_hidden.weight" in weights:
            self.pre_fc_norm_hidden_weight_ = self._cuda(weights["mtp.pre_fc_norm_hidden.weight"])
        if "mtp.norm.weight" in weights:
            self.final_norm_weight_ = self._cuda(weights["mtp.norm.weight"])

        return

    def verify_load(self):
        return True
