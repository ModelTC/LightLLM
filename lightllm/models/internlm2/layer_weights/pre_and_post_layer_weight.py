import torch
import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class Internlm2PreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(tp_rank, world_size, data_type, network_config, mode)
        return

    def load_hf_weights(self, weights):
        vob_size = self.network_config_["vocab_size"]
        split_vob_size = vob_size // self.world_size_
        n_embed = self.network_config_["hidden_size"]
        if "model.tok_embeddings.weight" in weights:
            self.wte_weight_ = self._cuda(weights['model.tok_embeddings.weight'][split_vob_size *
                                                                    self.tp_rank_: split_vob_size * (self.tp_rank_ + 1), :])
        if 'output.weight' in weights:
            self.lm_head_weight_ = self._cuda(weights['output.weight'][split_vob_size * self.tp_rank_: split_vob_size *
                                                            (self.tp_rank_ + 1), :])
        if 'model.norm.weight' in weights:
            self.final_norm_weight_ = self._cuda(weights['model.norm.weight'])

        return
