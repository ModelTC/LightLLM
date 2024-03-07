import torch
import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class Gemma_2bPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(tp_rank, world_size, data_type, network_config, mode)
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
            self.lm_head_weight_ = self._cuda(weights["lm_head.weight"][split_start:split_end, :])
        else:
            self.lm_head_weight_ = self.wte_weight_
        if "model.norm.weight" in weights:
            self.final_norm_weight_ = self._cuda(weights["model.norm.weight"])
            self.final_norm_weight_ = self.final_norm_weight_ + 1

        return