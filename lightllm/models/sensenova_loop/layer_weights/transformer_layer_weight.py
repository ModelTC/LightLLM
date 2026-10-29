import torch

from lightllm.common.basemodel.layer_weights.meta_weights import ParameterWeight
from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import Qwen35TransformerLayerWeight


class SenseNovaLoopTransformerLayerWeight(Qwen35TransformerLayerWeight):
    def _init_weight(self):
        execution_index = self.network_config_["loop_physical_indices"].index(self.layer_num_)
        self.is_linear_attention_layer = self.network_config_["layer_types"][execution_index] == "linear_attention"
        super()._init_weight()
        self.mhc = self.network_config_.get("mhc")
        if self.mhc is not None and self.mhc["start"] <= self.layer_num_ < self.mhc["end"]:
            self._init_mhc()

    def _init_mhc(self):
        prefix = f"model.layers.{self.layer_num_}"
        streams = self.mhc["streams"]
        mix_size = (2 + streams) * streams
        flattened_hidden = streams * self.n_embed
        self.mhc_attn = ParameterWeight(
            weight_name=f"{prefix}.mhc_attn.weight",
            bias_name=f"{prefix}.mhc_attn.bias",
            data_type=torch.float32,
            weight_shape=(mix_size, flattened_hidden),
            bias_shape=(mix_size,),
        )
        self.mhc_ffn = ParameterWeight(
            weight_name=f"{prefix}.mhc_ffn.weight",
            bias_name=f"{prefix}.mhc_ffn.bias",
            data_type=torch.float32,
            weight_shape=(mix_size, flattened_hidden),
            bias_shape=(mix_size,),
        )
