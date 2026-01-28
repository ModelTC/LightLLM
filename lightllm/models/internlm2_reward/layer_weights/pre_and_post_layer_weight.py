from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, RMSNormWeight, ROWMMWeight


class Internlm2RewardPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        hidden_size = network_config["hidden_size"]
        vocab_size = network_config["vocab_size"]
        self.wte_weight_ = EmbeddingWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="model.tok_embeddings.weight",
            data_type=self.data_type_,
        )
        self.score_head_ = ROWMMWeight(
            weight_names="v_head.weight",
            data_type=self.data_type_,
            name="score_head",
            tp_rank=0,
            tp_world_size=1,
        )
        self.final_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name="model.norm.weight",
            data_type=self.data_type_,
        )
        return
