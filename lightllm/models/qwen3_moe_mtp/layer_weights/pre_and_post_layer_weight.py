import numpy as np
from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    ROWMMWeight,
    LMHeadWeight,
    NoTpNormWeight,
)


class Qwen3MOEMTPPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)

        self.eh_proj_weight_ = ROWMMWeight(
            weight_names="model.0.proj.weight",
            data_type=self.data_type_,
            name="eh_proj",
            tp_rank=0,
            tp_world_size=1,
        )
        self.enorm_weight_ = NoTpNormWeight(
            weight_name="model.0.norm_after_embedding.weight",
            data_type=self.data_type_,
            bias_name=None,
        )
        self.hnorm_weight_ = NoTpNormWeight(
            weight_name="model.0.norm_before_output.weight",
            data_type=self.data_type_,
            bias_name=None,
        )
        # 与Qwen3MOE模型共享
        self.wte_weight_: EmbeddingWeight = None
        self.lm_head_weight_: LMHeadWeight = None
        self.final_norm_weight_: NoTpNormWeight = None
        return

    def load_hf_weights(self, weights):
        vob_size = self.network_config_["vocab_size"]
        split_indexes = np.linspace(0, vob_size, self.tp_world_size_ + 1, dtype=np.int64)
        split_start = split_indexes[self.tp_rank_]
        split_end = split_indexes[self.tp_rank_ + 1]
        if "model.layers.0.proj.weight" in weights:
            self.eh_proj_weight_ = self._cuda(weights["model.layers.0.proj.weight"]).t()
        if "model.layers.0.norm_after_embedding.weight" in weights:
            self.enorm_weight_ = self._cuda(weights["model.layers.0.norm_after_embedding.weight"])
        if "model.layers.0.norm_before_output.weight" in weights:
            self.hnorm_weight_ = self._cuda(weights["model.layers.0.norm_before_output.weight"])
        if "lm_head.weight" in weights:
            self.lm_head_weight_ = self._cuda(weights["lm_head.weight"][split_start:split_end, :])
        if "model.norm.weight" in weights:
            self.final_norm_weight_ = self._cuda(weights["model.norm.weight"])
        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.eh_proj_weight_, self.enorm_weight_, self.hnorm_weight_]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return
