import torch
import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class MiniCPMPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(tp_rank, world_size, data_type, network_config, mode)
        hidden_size = self.network_config_["hidden_size"]
        dim_model_base = self.network_config_.get("dim_model_base", hidden_size)
        self.lm_head_scale = hidden_size / dim_model_base
        self.scale_emb = self.network_config_.get("scale_emb", 1)
        return

    def load_hf_weights(self, weights):
        vob_size = self.network_config_["vocab_size"]
        split_indexes = np.linspace(0, vob_size, self.world_size_ + 1, dtype=np.int64)
        split_start = split_indexes[self.tp_rank_]
        split_end = split_indexes[self.tp_rank_ + 1]
        if "model.embed_tokens.weight" in weights:
            # print(weights['model.embed_tokens.weight'].shape)
            self.wte_weight_ = self._cuda(weights["model.embed_tokens.weight"][split_start:split_end, :])
        if "lm_head.weight" in weights:
            # print(weights['lm_head.weight'].shape)
            self.lm_head_weight_ = self._cuda(weights["lm_head.weight"][split_start:split_end, :]) / self.lm_head_scale
        if "model.norm.weight" in weights:
            self.final_norm_weight_ = self._cuda(weights["model.norm.weight"])

        return

    def verify_load(self):
        if not hasattr(self, "lm_head_weight_"):
            self.lm_head_weight_ =  self.wte_weight_ / self.lm_head_scale
        self.wte_weight_ = self.wte_weight_ * self.scale_emb
        errors = "weights load not ok"
        weights = [self.wte_weight_, self.lm_head_weight_, self.final_norm_weight_]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return
