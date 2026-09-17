from lightllm.common.basemodel.layer_weights.meta_weights import ParameterWeight, ROWMMWeight
from lightllm.models.qwen3_dflash.layer_weights.transformer_layer_weight import Qwen3DFlashTransformerLayerWeight


class Qwen3DFlash2TransformerLayerWeight(Qwen3DFlashTransformerLayerWeight):
    """DFlash decoder weights plus the two dynamic convolutions."""

    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)

        hidden_size = network_config["hidden_size"]
        kernel_size = int(network_config["conv_kernel_size"])
        group_size = int(network_config["conv_group_size"])
        assert hidden_size % group_size == 0
        group_num = hidden_size // group_size
        dynamic_size = 2 * kernel_size * group_num
        weight_prefix = f"layers.{self.layer_num_}"

        self.attention_conv_projection_weight_ = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[dynamic_size],
            weight_names=f"{weight_prefix}.attention_conv.kernel_projection.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.attention_conv_base_weight_ = ParameterWeight(
            weight_name=f"{weight_prefix}.attention_conv.base_kernel",
            data_type=self.data_type_,
            weight_shape=(2, kernel_size, hidden_size),
        )
        self.mlp_conv_projection_weight_ = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[dynamic_size],
            weight_names=f"{weight_prefix}.mlp_conv.kernel_projection.weight",
            data_type=self.data_type_,
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )
        self.mlp_conv_base_weight_ = ParameterWeight(
            weight_name=f"{weight_prefix}.mlp_conv.base_kernel",
            data_type=self.data_type_,
            weight_shape=(2, kernel_size, hidden_size),
        )
