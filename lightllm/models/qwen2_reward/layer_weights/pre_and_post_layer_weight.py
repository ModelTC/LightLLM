import torch
import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, NormWeight


class Qwen2RewardPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        return

    def load_hf_weights(self, weights):
        vob_size = self.network_config_["vocab_size"]
        split_indexes = np.linspace(0, vob_size, self.tp_world_size_ + 1, dtype=np.int64)
        split_start = split_indexes[self.tp_rank_]
        split_end = split_indexes[self.tp_rank_ + 1]
        if "model.embed_tokens.weight" in weights:
            self.wte_weight_ = self._cuda(weights["model.embed_tokens.weight"][split_start:split_end, :])
            tie_word_embeddings = self.network_config_.get("tie_word_embeddings", False)
            if tie_word_embeddings:
                self.lm_head_weight_ = self.wte_weight_

        if "model.norm.weight" in weights:
            self.final_norm_weight_ = self._cuda(weights["model.norm.weight"])

        if "score.0.weight" in weights:
            self.score_up_weight = self._cuda(weights["score.0.weight"]).transpose(0, 1)
        if "score.0.bias" in weights:
            self.score_up_bias = self._cuda(weights["score.0.bias"])

        if "score.2.weight" in weights:
            self.score_down_weight = self._cuda(weights["score.2.weight"]).transpose(0, 1)
        if "score.2.bias" in weights:
            self.score_down_bias = self._cuda(weights["score.2.bias"])

        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.wte_weight_,
            self.final_norm_weight_,
            self.score_up_weight,
            self.score_up_bias,
            self.score_down_weight,
            self.score_down_bias,
        ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return
