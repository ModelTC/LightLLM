import torch
import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class Deepseek3MTPPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        # 与DeepseekV3模型共享
        self.wte_weight_ = None
        self.lm_head_weight_ = None
        return

    def _create_weight(self):
        hidden_size = self.network_config_["hidden_size"]
        moe_intermediate_size = self.network_config_["moe_intermediate_size"]

        # Pre-allocate memory for weights
        self.eh_proj_weight_ = torch.empty((moe_intermediate_size, hidden_size), dtype=self.data_type_).cuda()
        self.enorm_weight_ = torch.empty(hidden_size, dtype=self.data_type_).cuda()
        self.hnorm_weight_ = torch.empty(hidden_size, dtype=self.data_type_).cuda()
        self.final_norm_weight_ = torch.empty(hidden_size, dtype=self.data_type_).cuda()
        return

    def load_hf_weights(self, weights):
        if "model.layers.0.eh_proj.weight" in weights:
            self.eh_proj_weight_.copy_(weights["model.layers.0.eh_proj.weight"].t())
        if "model.layers.0.enorm.weight" in weights:
            self.enorm_weight_.copy_(weights["model.layers.0.enorm.weight"])
        if "model.layers.0.hnorm.weight" in weights:
            self.hnorm_weight_.copy_(weights["model.layers.0.hnorm.weight"])
        if "model.layers.0.shared_head.norm.weight" in weights:
            self.final_norm_weight_.copy_(weights["model.layers.0.shared_head.norm.weight"])
        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.eh_proj_weight_, self.enorm_weight_, self.hnorm_weight_, self.final_norm_weight_]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return
