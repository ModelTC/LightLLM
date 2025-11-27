import torch
import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class Internlm2PreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        return

    def _create_weight(self):
        vob_size = self.network_config_["vocab_size"]
        hidden_size = self.network_config_["hidden_size"]
        split_indexes = np.linspace(0, vob_size, self.tp_world_size_ + 1, dtype=np.int64)
        split_start = split_indexes[self.tp_rank_]
        split_end = split_indexes[self.tp_rank_ + 1]
        split_vob_size = split_end - split_start

        # Pre-allocate memory for weights
        self.wte_weight_ = torch.empty((split_vob_size, hidden_size), dtype=self.data_type_).cuda()
        self.lm_head_weight_ = torch.empty((split_vob_size, hidden_size), dtype=self.data_type_).cuda()
        self.final_norm_weight_ = torch.empty(hidden_size, dtype=self.data_type_).cuda()
        return

    def load_hf_weights(self, weights):
        vob_size = self.network_config_["vocab_size"]
        split_indexes = np.linspace(0, vob_size, self.tp_world_size_ + 1, dtype=np.int64)
        split_start = split_indexes[self.tp_rank_]
        split_end = split_indexes[self.tp_rank_ + 1]
        if "model.tok_embeddings.weight" in weights:
            self.wte_weight_.copy_(weights["model.tok_embeddings.weight"][split_start:split_end, :])
        if "output.weight" in weights:
            self.lm_head_weight_.copy_(weights["output.weight"][split_start:split_end, :])
        if "model.norm.weight" in weights:
            self.final_norm_weight_.copy_(weights["model.norm.weight"])

        return
