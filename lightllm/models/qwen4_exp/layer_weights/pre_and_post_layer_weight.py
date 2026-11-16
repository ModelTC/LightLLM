from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import EmbeddingWeight
from lightllm.models.qwen4_exp.checkpoint import normalize_text_weight_names
from .components import gated_residual_weight
from .lm_head_weight import Qwen4ExpLMHeadWeight


class Qwen4ExpPreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        hidden, vocab = network_config["hidden_size"], network_config["vocab_size"]
        self.wte_weight_ = EmbeddingWeight(hidden, vocab, "model.embed_tokens.weight", data_type)
        self.lm_head_weight_ = Qwen4ExpLMHeadWeight(
            hidden,
            vocab,
            "lm_head.weight",
            data_type,
            self.wte_weight_ if network_config.get("tie_word_embeddings", False) else None,
        )
        self.hyper_connection_mixer = gated_residual_weight(
            "model.hyper_connection_mixer", network_config, data_type, use_combine=False
        )

    def load_hf_weights(self, weights):
        normalize_text_weight_names(weights)
        super().load_hf_weights(weights)
