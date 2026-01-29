from lightllm.models.deepseek_mtp.layer_weights.pre_and_post_layer_weight import Deepseek3MTPPreAndPostLayerWeight


class Glm4MoeLiteMTPPreAndPostLayerWeight(Deepseek3MTPPreAndPostLayerWeight):
    def __init__(self, data_type, network_config, quant_cfg, layer_idx=0):
        super().__init__(data_type, network_config, quant_cfg, network_config["num_hidden_layers"])
