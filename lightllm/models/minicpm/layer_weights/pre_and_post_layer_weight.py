import copy
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class MiniCPMPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        hidden_size = self.network_config_["hidden_size"]
        dim_model_base = self.network_config_.get("dim_model_base", hidden_size)
        self.lm_head_scale = hidden_size / dim_model_base
        self.scale_emb = self.network_config_.get("scale_emb", 1)
        return

    def verify_load(self):
        if self.lm_head_weight_ == self.wte_weight_:
            self.lm_head_weight_ = copy.copy(self.lm_head_weight_)

        self.lm_head_weight_.weight = self.lm_head_weight_.weight / self.lm_head_scale
        self.wte_weight_.weight = self.wte_weight_.weight * self.scale_emb
        errors = "weights load not ok"
        weights = [self.wte_weight_, self.lm_head_weight_, self.final_norm_weight_]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return
