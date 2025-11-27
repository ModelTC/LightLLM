import torch
import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class StableLMPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
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
        self.final_norm_bias_ = torch.empty(hidden_size, dtype=self.data_type_).cuda()
        return

    def load_hf_weights(self, weights):
        vob_size = self.network_config_["vocab_size"]
        split_indexes = np.linspace(0, vob_size, self.tp_world_size_ + 1, dtype=np.int64)
        split_start = split_indexes[self.tp_rank_]
        split_end = split_indexes[self.tp_rank_ + 1]
        if "model.embed_tokens.weight" in weights:
            # print(weights['model.embed_tokens.weight'].shape)
            self.wte_weight_.copy_(weights["model.embed_tokens.weight"][split_start:split_end, :])
        if "lm_head.weight" in weights:
            # print(weights['lm_head.weight'].shape)
            self.lm_head_weight_.copy_(weights["lm_head.weight"][split_start:split_end, :])
        if "model.norm.weight" in weights:
            self.final_norm_weight_.copy_(weights["model.norm.weight"])
        if "model.norm.bias" in weights:
            self.final_norm_bias_.copy_(weights["model.norm.bias"])

        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.wte_weight_, self.lm_head_weight_, self.final_norm_weight_, self.final_norm_bias_]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return
