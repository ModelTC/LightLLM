import torch
import numpy as np

from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class CoherePreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def load_hf_weights(self, weights):
        vob_size = self.network_config_["vocab_size"]
        tie_weight = self.network_config_.get("tie_word_embeddings", True)
        split_indexes = np.linspace(0, vob_size, self.tp_world_size_ + 1, dtype=np.int64)
        split_start = split_indexes[self.tp_rank_]
        split_end = split_indexes[self.tp_rank_ + 1]
        if "model.embed_tokens.weight" in weights:
            # print(weights['model.embed_tokens.weight'].shape)
            self.wte_weight_ = self._cuda(weights["model.embed_tokens.weight"][split_start:split_end, :])
            if tie_weight:
                self.lm_head_weight_ = self.wte_weight_
        if "model.norm.weight" in weights:
            self.final_norm_weight_ = self._cuda(weights["model.norm.weight"])
        if "model.lm_head.weight" in weights:
            self.lm_head_weight_ = self._cuda(weights["model.lm_head.weight"])
        return

    def verify_load(self):
        super().verify_load()

        errors = "tie weights load not ok"
        tie_weight = self.network_config_.get("tie_word_embeddings", True)
        if tie_weight:
            assert self.lm_head_weight_ is not None, errors
            assert self.wte_weight_ is self.lm_head_weight_, errors
        else:
            assert self.lm_head_weight_ is not None, errors
            assert self.wte_weight_ is not None, errors
            assert self.wte_weight_ is not self.lm_head_weight_, errors
