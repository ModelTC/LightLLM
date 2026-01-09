from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight, NoTpNormWeight


class StableLMPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        self.final_norm_weight_ = NoTpNormWeight(
            weight_name="model.norm.weight",
            data_type=self.data_type_,
            bias_name="model.norm.bias",
        )
        return
