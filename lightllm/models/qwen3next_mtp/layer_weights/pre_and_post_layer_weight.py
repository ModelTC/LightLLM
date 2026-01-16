from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import NoTpGEMMANormWeight


class Qwen3NextMTPPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        self.wte_weight_ = None
        self.lm_head_weight_ = None

        # Use Gemma-style normalization for all MTP norm layers
        self.final_norm_weight_ = NoTpGEMMANormWeight(
            weight_name="mtp.norm.weight",
            data_type=self.data_type_,
            bias_name=None,
        )
        self.pre_fc_norm_embedding_weight_ = NoTpGEMMANormWeight(
            weight_name="mtp.pre_fc_norm_embedding.weight",
            data_type=self.data_type_,
            bias_name=None,
        )
        self.pre_fc_norm_hidden_weight_ = NoTpGEMMANormWeight(
            weight_name="mtp.pre_fc_norm_hidden.weight",
            data_type=self.data_type_,
            bias_name=None,
        )
        return

    def load_hf_weights(self, weights):
        if "mtp.fc.weight" in weights:
            self.fc_weight_ = self._cuda(weights["mtp.fc.weight"]).t()

        # Load weights for norm weight objects
        self.final_norm_weight_.load_hf_weights(weights)
        self.pre_fc_norm_embedding_weight_.load_hf_weights(weights)
        self.pre_fc_norm_hidden_weight_.load_hf_weights(weights)

        return

    def verify_load(self):
        # Verify all norm weights loaded correctly
        return (
            self.final_norm_weight_.verify_load()
            and self.pre_fc_norm_embedding_weight_.verify_load()
            and self.pre_fc_norm_hidden_weight_.verify_load()
        )
