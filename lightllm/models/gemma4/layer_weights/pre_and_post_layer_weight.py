from lightllm.common.basemodel import PreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    EmbeddingWeight,
    LMHeadWeight,
    ROWMMWeight,
    RMSNormWeight,
)


class Gemma4PreAndPostLayerWeight(PreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        hidden_size = network_config["hidden_size"]
        vocab_size = network_config["vocab_size"]

        self.wte_weight_ = EmbeddingWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="model.language_model.embed_tokens.weight",
            data_type=self.data_type_,
        )
        # lm_head is tied to input embedding for Gemma-4 (no separate lm_head.weight).
        self.lm_head_weight_ = LMHeadWeight(
            dim=hidden_size,
            vocab_size=vocab_size,
            weight_name="lm_head.weight",
            data_type=self.data_type_,
            embedding_weight=self.wte_weight_,
        )

        # Gemma-4 uses standard RMSNorm (not the gemma2/3 (1+w) variant).
        self.final_norm_weight_ = RMSNormWeight(
            dim=hidden_size,
            weight_name="model.language_model.norm.weight",
            data_type=self.data_type_,
        )

        if network_config.get("hidden_size_per_layer_input"):
            num_layers = network_config["num_hidden_layers"]
            ple_dim = network_config["hidden_size_per_layer_input"]
            ple_vocab = network_config.get("vocab_size_per_layer_input", vocab_size)
            self.embed_tokens_per_layer_weight_ = EmbeddingWeight(
                dim=num_layers * ple_dim,
                vocab_size=ple_vocab,
                weight_name="model.language_model.embed_tokens_per_layer.weight",
                data_type=self.data_type_,
            )
            # nn.Linear(in=hidden_size, out=num_layers*ple_dim); HF storage is
            # (out, in). Replicated across TP ranks.
            self.per_layer_model_projection_weight_ = ROWMMWeight(
                in_dim=hidden_size,
                out_dims=[num_layers * ple_dim],
                weight_names="model.language_model.per_layer_model_projection.weight",
                data_type=self.data_type_,
                tp_rank=0,
                tp_world_size=1,
            )
            # RMSNorm over the ple_dim of the projection output.
            self.per_layer_projection_norm_weight_ = RMSNormWeight(
                dim=ple_dim,
                weight_name="model.language_model.per_layer_projection_norm.weight",
                data_type=self.data_type_,
            )
        return
