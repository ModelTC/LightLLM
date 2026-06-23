import torch
from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    RMSNormWeight,
    ParameterWeight,
)


class DeepseekV4PreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)

        hidden = network_config["hidden_size"]
        vocab = network_config["vocab_size"]
        hc_mult = network_config["hc_mult"]

        # embeddings / lm_head / final norm (bf16, vocab tensor-parallel). V4 has no `model.` prefix
        # and does not tie embeddings (tie_word_embeddings=false).
        self.wte_weight_ = EmbeddingWeight(
            dim=hidden, vocab_size=vocab, weight_name="embed.weight", data_type=self.data_type_
        )
        self.lm_head_weight_ = LMHeadWeight(
            dim=hidden, vocab_size=vocab, weight_name="head.weight", data_type=self.data_type_
        )
        self.final_norm_weight_ = RMSNormWeight(dim=hidden, weight_name="norm.weight", data_type=self.data_type_)

        # final hyper-connection head (collapses the hc_mult residual streams before the lm_head)
        self.hc_head_fn_ = ParameterWeight(
            weight_name="hc_head_fn", data_type=torch.float32, weight_shape=(hc_mult, hc_mult * hidden)
        )
        self.hc_head_base_ = ParameterWeight(
            weight_name="hc_head_base", data_type=torch.float32, weight_shape=(hc_mult,)
        )
        self.hc_head_scale_ = ParameterWeight(weight_name="hc_head_scale", data_type=torch.float32, weight_shape=(1,))
        return
