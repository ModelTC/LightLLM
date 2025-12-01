import torch
import numpy as np
from lightllm.common.basemodel import PreAndPostLayerWeight


class BloomPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        self._create_weight()

    def _create_weight(self):
        vob_size = self.network_config_["vocab_size"]
        hidden_size = self.network_config_["hidden_size"]
        split_vob_size = vob_size // self.tp_world_size_

        # Pre-allocate memory for weights
        self.pre_norm_weight_ = torch.empty(hidden_size, dtype=self.data_type_).cuda()
        self.pre_norm_bias_ = torch.empty(hidden_size, dtype=self.data_type_).cuda()
        self.final_norm_weight_ = torch.empty(hidden_size, dtype=self.data_type_).cuda()
        self.final_norm_bias_ = torch.empty(hidden_size, dtype=self.data_type_).cuda()
        self.wte_weight_ = torch.empty((split_vob_size, hidden_size), dtype=self.data_type_).cuda()
        self.lm_head_weight_ = torch.empty((split_vob_size, hidden_size), dtype=self.data_type_).cuda()
        return

    def load_hf_weights(self, weights):

        if "word_embeddings_layernorm.weight" in weights:
            self.pre_norm_weight_.copy_(weights["word_embeddings_layernorm.weight"])
        if "word_embeddings_layernorm.bias" in weights:
            self.pre_norm_bias_.copy_(weights["word_embeddings_layernorm.bias"])
        if "ln_f.weight" in weights:
            self.final_norm_weight_.copy_(weights["ln_f.weight"])
        if "ln_f.bias" in weights:
            self.final_norm_bias_.copy_(weights["ln_f.bias"])
        if "word_embeddings.weight" in weights:
            vob_size = self.network_config_["vocab_size"]
            split_vob_size = vob_size // self.tp_world_size_
            self.wte_weight_.copy_(
                weights["word_embeddings.weight"][
                    split_vob_size * self.tp_rank_ : split_vob_size * (self.tp_rank_ + 1), :
                ]
            )
            self.lm_head_weight_ = self.wte_weight_
        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.pre_norm_weight_,
            self.pre_norm_bias_,
            self.final_norm_weight_,
            self.final_norm_bias_,
            self.wte_weight_,
            self.lm_head_weight_,
        ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return
