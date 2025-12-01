import torch
import numpy as np
from lightllm.common.basemodel import PreAndPostLayerWeight


class QwenPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        self._create_weight()
        return

    def _create_weight(self):
        vob_size = self.network_config_["vocab_size"]
        hidden_size = self.network_config_["hidden_size"]
        split_vob_size = vob_size // self.tp_world_size_

        # Pre-allocate memory for weights
        self.wte_weight_ = torch.empty((split_vob_size, hidden_size), dtype=self.data_type_).cuda()
        self.lm_head_weight_ = torch.empty((split_vob_size, hidden_size), dtype=self.data_type_).cuda()
        self.final_norm_weight_ = torch.empty(hidden_size, dtype=self.data_type_).cuda()
        return

    def load_hf_weights(self, weights):

        vob_size = self.network_config_["vocab_size"]
        split_vob_size = vob_size // self.tp_world_size_

        if "transformer.wte.weight" in weights:
            self.wte_weight_.copy_(
                weights["transformer.wte.weight"][
                    split_vob_size * self.tp_rank_ : split_vob_size * (self.tp_rank_ + 1), :
                ]
            )
        if "lm_head.weight" in weights:
            self.lm_head_weight_.copy_(
                weights["lm_head.weight"][split_vob_size * self.tp_rank_ : split_vob_size * (self.tp_rank_ + 1), :]
            )
        if "transformer.ln_f.weight" in weights:
            self.final_norm_weight_.copy_(weights["transformer.ln_f.weight"])

        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.wte_weight_,
            self.lm_head_weight_,
            self.final_norm_weight_,
        ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return
