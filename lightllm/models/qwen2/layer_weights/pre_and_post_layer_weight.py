from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import LMHeadWeight


class Qwen2PreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        hidden_size = network_config["hidden_size"]
        vocab_size = network_config["vocab_size"]
        tie_word_embeddings = network_config.get("tie_word_embeddings", False)
        self.lm_head_weight_ = LMHeadWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="thinker.lm_head.weight",
            data_type=self.data_type_,
            embedding_weight=self.wte_weight_ if tie_word_embeddings else None,
        )
        return
