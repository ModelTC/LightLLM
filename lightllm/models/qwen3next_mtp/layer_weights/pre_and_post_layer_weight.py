import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class Qwen3NextMTPPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        self.wte_weight_ = None
        self.lm_head_weight_ = None
        return

    def load_hf_weights(self, weights):
        # Load MTP-specific pre-layer weights
        # mtp.fc.weight - projection from (embedding + hidden) to hidden_size
        if "mtp.fc.weight" in weights:
            self.fc_weight_ = self._cuda(weights["mtp.fc.weight"]).t()

        # mtp.pre_fc_norm_embedding.weight - normalize embedding before fc
        if "mtp.pre_fc_norm_embedding.weight" in weights:
            self.pre_fc_norm_embedding_weight_ = self._cuda(weights["mtp.pre_fc_norm_embedding.weight"])

        # mtp.pre_fc_norm_hidden.weight - normalize hidden state before fc
        if "mtp.pre_fc_norm_hidden.weight" in weights:
            self.pre_fc_norm_hidden_weight_ = self._cuda(weights["mtp.pre_fc_norm_hidden.weight"])

        # mtp.norm.weight - final norm before lm_head
        if "mtp.norm.weight" in weights:
            self.final_norm_weight_ = self._cuda(weights["mtp.norm.weight"])

        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.fc_weight_,
            self.pre_fc_norm_embedding_weight_,
            self.pre_fc_norm_hidden_weight_,
            self.final_norm_weight_,
        ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return
