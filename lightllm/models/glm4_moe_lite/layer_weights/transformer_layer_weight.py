import torch
from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import Deepseek2TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, FusedMoeWeight


class Glm4MoeLiteTransformerLayerWeight(Deepseek2TransformerLayerWeight):
    def __init__(self, layer_num, data_type, network_config, quant_cfg=None):
        super().__init__(layer_num, data_type, network_config, quant_cfg)

    def _parse_config(self):
        super()._parse_config()

        self.is_moe = self.network_config_.get(
            "n_routed_experts"
        ) is not None and self.layer_num_ >= self.network_config_.get("first_k_dense_replace", 0)

        from lightllm.utils.envs_utils import get_env_start_args

        self.num_fused_shared_experts = 0
        if get_env_start_args().enable_fused_shared_experts and self.is_moe:
            assert not get_env_start_args().enable_ep_moe, "enable_fused_shared_experts can only work with tp mode."
            self.num_fused_shared_experts = self.network_config_.get("n_shared_experts", 0)

    def _init_moe(self):
        moe_intermediate_size = self.network_config_["moe_intermediate_size"]
        hidden_size = self.network_config_["hidden_size"]

        self.moe_gate = ROWMMWeight(
            in_dim=hidden_size,
            out_dims=[self.n_routed_experts],
            weight_names=f"model.layers.{self.layer_num_}.mlp.gate.weight",
            data_type=torch.float32,  # Router gate needs float32 for numerical stability
            quant_method=None,
            tp_rank=0,
            tp_world_size=1,
        )

        if self.num_fused_shared_experts == 0:
            self._load_mlp(f"model.layers.{self.layer_num_}.mlp.shared_experts", is_shared_experts=True)

        self.experts = FusedMoeWeight(
            gate_proj_name="gate_proj",
            down_proj_name="down_proj",
            up_proj_name="up_proj",
            e_score_correction_bias_name=self.e_score_correction_bias_name,
            weight_prefix=f"model.layers.{self.layer_num_}.mlp.experts",
            n_routed_experts=self.n_routed_experts,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            data_type=self.data_type_,
            quant_method=self.quant_cfg.get_quant_method(self.layer_num_, "fused_moe"),
            num_fused_shared_experts=self.num_fused_shared_experts,
            layer_num=self.layer_num_,
            network_config=self.network_config_,
        )
