from lightllm.common.basemodel.layer_weights.meta_weights import GEMMANormWeight
from lightllm.models.qwen3_vl.layer_weights.pre_and_post_layer_weight import Qwen3VLPreAndPostLayerWeight


class Qwen35PreAndPostLayerWeight(Qwen3VLPreAndPostLayerWeight):
    def __init__(self, data_type, network_config):
        super().__init__(data_type, network_config)
        self.final_norm_weight_ = GEMMANormWeight(
            dim=network_config["hidden_size"],
            weight_name="model.norm.weight",
            data_type=self.data_type_,
        )
