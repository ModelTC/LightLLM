from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight, LMHeadWeight, RMSNormWeight


class DeepseekV41PreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        hidden, vocab = network_config["hidden_size"], network_config["vocab_size"]
        self.wte_weight_ = EmbeddingWeight(
            dim=hidden, vocab_size=vocab, weight_name="embed.weight", data_type=data_type
        )
        self.lm_head_weight_ = LMHeadWeight(
            dim=hidden, vocab_size=vocab, weight_name="head.weight", data_type=data_type
        )
        self.final_norm_weight_ = RMSNormWeight(dim=hidden, weight_name="norm.weight", data_type=data_type)
